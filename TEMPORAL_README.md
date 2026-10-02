# Temporal TEASER (`--temporalize_teaser`)

TEASER is a per-frame regressor: each frame's face crop goes through four
independent MobileNetV3 encoders and a linear head, and nothing ties frame *t*
to *t+1* -- no temporal layer, no temporal loss, video datasets sampled one
frame at a time (`K: 1` in both configs, and `K` / `LRS3_temporal_sampling`
are not read anywhere in the code). Applied to sign-language video this gives
visible jitter in expression/jaw, worst when a hand comes close to or covers
the mouth. Today the coherence is added afterwards: Savitzky-Golay window 9
on the fitted SMPL-X (SG9), then a cubic Hermite splice over short hand-face
occlusion episodes (`rgb2smplx/stages/face_jitter_fix.py` in RGB2SMPLX). The
splice is wrong exactly where it matters: the mouth keeps articulating
(mouthing) under the hand, and the frames next to the occlusion are already
corrupted.

This branch adds a small temporal module that works on the latent features of
TEASER's frozen encoders, before the regression heads, trained with offline
pseudo-GT from Pixel3DMM and self-distillation from TEASER itself, plus
synthetic hand occlusion so that occluded frames have a real target.

**Status:** plan agreed 2026-10-02; implementation in progress (see
[Progress](#progress)).

---

## 1. Ground rule: everything behind a flag

- Every change is behind `--temporalize_teaser` (and `temporal.enabled` in the
  YAML configs under `configs/temporal/`). **Default OFF.**
- With the flag OFF, training, demo and inference behave **identically** to the
  original: same outputs, same checkpoints. A test checks it numerically.
- New code lives in `src/temporal/` and `tools_temporal/`. Existing files get
  at most an `if args.temporalize_teaser:` branch. `TeaserEncoder` itself is
  not modified: the split forward (features -> heads) is reimplemented in
  `src/temporal/split_encoder.py` and a test checks it equals
  `TeaserEncoder.forward` bit for bit.
- Sub-options are dedicated flags, only read when `--temporalize_teaser` is ON.
- Work happens on the `temporal` branch of the fork (worktree
  `/leonardo_work/IscrC_SLPSCALE/TEASER_temporal`), not on the submodule
  checkout that RGB2SMPLX production runs from. The RGB2SMPLX pipeline is not
  touched until the ablations pick a winner (see 9).

---

## 2. What the code actually is (Phase 0 findings)

Measured by running the real model (TEASER env, `pretrained_models/TEASER.pt`;
`Teaser.pt`, `TEASER.pt`, `TEASER_v1.pt` are the same file, md5 `ba1bbcae...`).

| Encoder | timm backbone | pooled feature | head |
|---|---|---|---|
| Pose | `tf_mobilenetv3_small_minimal_100` | **576** | `Linear(576->6)`: `[0:3]` pose (axis-angle), `[3:6]` cam `[s,tx,ty]` (orthographic) |
| Shape | `tf_mobilenetv3_large_minimal_100` | **960** | `Linear(960->300)` |
| Expression | `tf_mobilenetv3_large_minimal_100` | **960** | `Linear(960->55)`: `[0:50]` expr, `[50:52]` eyelid `clamp(0,1)`, `[52]` jaw open `ReLU`, `[53:55]` jaw `clamp(+-0.2)` |
| Token | `tf_mobilenetv3_small_minimal_100` | 4 x 256 (multi-scale) | only used by the training-time generator |

- Pooling: `adaptive_avg_pool2d(features[-1], 1)` on the 7x7 map, then a
  **single** linear layer. `--temporal_feats all` = 960 + 576 + 960 = 2496.
- Consequence: with a frozen linear head, a feature delta acts on the output
  only through `W @ delta_f`, so a feature-space adapter is equivalent to a
  55-dim parameter residual whose *input* is the 960-dim feature. Its edge over
  a parameter-space smoother (SmoothNet) is the richer input. It also means
  LoRA on the head is meaningless (the head is 53k parameters, as cheap as
  full fine-tuning): `--temporal_head` is `frozen | full` only (decision D).
- FLAME: `assets/FLAME2020/generic_model.pkl`, 300 shape + 50 expression;
  neck and eyeballs fixed at defaults (**no gaze**); eyelids are two extra
  vertex blendshapes (`assets/l_eyelid.npy`, `r_eyelid.npy`), the same ones
  Pixel3DMM uses.
- Region masks: `assets/FLAME_masks/FLAME_masks.pkl` (tracked in the repo).
  Regions used by the losses and metrics (`src/temporal/flame_regions.py`):
  - **mouth** = `lips` (250 unique vertices)
  - **eyes** = (`eye_region` U eyelid-blendshape support) - eyeballs
  - **rest** = `face` - mouth - eyes
  Eyeballs are excluded everywhere: TEASER predicts no gaze.
- Pixel3DMM runs FLAME 2020 with 100 expressions; TEASER's 50 are the first 50
  of the same basis. So `refit50` is a per-frame least squares (linear in
  expression given the jaw, small nonlinear jaw fit), and the `vertices` target
  carries an irreducible floor (components 51-100).
- Original training (`main/train.py`, OmegaConf + `src/teaser_trainer.py`)
  cannot run here: `datasets/preprocess_scripts/landmark.onnx` (203-landmark
  model) and the training datasets are missing. Temporal training is therefore
  a separate script, `tools_temporal/train_temporal.py`, that never imports
  the original trainer.
- Video inference in RGB2SMPLX (`rgb2smplx/stages/teaser.py`): pose-ROI ->
  FaceLandmarker landmarks (gaps interpolated), crop at scale 1.4 to 224,
  per-frame forward (`--batch-size` only concatenates). `teaser.npz` does not
  store the crop transform; the feature cache does.
- Batch size matters numerically: batch 1 reproduces the per-frame output bit
  for bit; batched convs move expression by ~5e-3 (cuDNN kernel choice). Cache
  extraction defaults to batch 1 so the cached outputs equal `teaser.npz`.

---

## 3. Data

| Set | Clips | Pixel3DMM | TEASER | hands (`wilor.npz`) | use |
|---|---|---|---|---|---|
| PHOENIX train subset | 300 (to run) | to run | yes | yes | train |
| CSL-Daily train subset | 300 (to run) | to run (30 fps) | yes | yes | train |
| PHOENIX test (`phoenix_facesmooth/test`) | 100 | yes | in the corpus | yes | test only |
| Multiface (`multiface/mf50`) | 49 | yes | yes | n/a (no hands) | test, **real 3D GT** |

- Selection of the train subsets by `c_t` from `wilor.npz`: **~75 % clean,
  ~25 % with real hand-face occlusion**. Clean clips give pseudo-GT that is
  valid on every frame, so every synthetic occlusion pasted on them has a full
  target; one clean clip yields many training samples (different hands,
  trajectories, timings each epoch). Real occluded clips are kept for the
  frames near occlusion (where Pixel3DMM is still usable), to train the gate on
  real `c_t`, and for the real-world test.
- Pixel3DMM costs ~490 s/clip/GPU (PHOENIX): 600 clips ~ 80 GPU-h. The tracker
  per-frame loss is dumped too when re-running (gives an optional confidence
  weight; without it the weight is 1).
- Never train on PHOENIX test, Multiface or the reel clips.

### Data access (found 2026-10-02) -- blocks training

| corpus / split | frames readable by this account | per-clip npz |
|---|---|---|
| PHOENIX train (`phoenix/phoenix/train`, 7101 clips) | **no**: `frames/` link into `/leonardo_scratch/fast/IscrC_SIGMA/signdata/...` | `wilor/teaser/mediapipe.npz` mode 600 (owner only); `fit_gvhmr*.npz` readable |
| PHOENIX test | yes (`RGB2SMPLX/phoenix_data`, 642 clips; the local tarball holds only test) | yes |
| CSL-Daily train | **no**: `/leonardo_work/IscrC_SIGMA/rgb2smplx/csl*` permission denied | -- |
| CSL-Daily test (`csl/test`, 123 clips) | yes | yes |

Pixel3DMM and TEASER both need the frames, so the train subsets cannot be
built until read access to those frames exists (or PHOENIX-2014-T is fetched
in full). Clip selection does not need frames: it runs on the readable fits.

- Clip selection: `tools_temporal/select_train_clips.py` (RGB2SMPLX env) scores
  clips with RGB2SMPLX's own jitter-fix occlusion measure on the fitted
  SMPL-X. Fully clean clips are rare (PHOENIX train sample: 14 of 20 clips had
  an occluded frame, the clean ones all short), so "clean" = at most 5 %
  occluded frames, "occluded" = at least 10 %, clips of at least 32 frames.
- Pixel3DMM throughput on PHOENIX: ~12 clips per GPU-hour (2 workers on one
  A100, `experiments/occlusion_protocols_smplx/pixel3dmm/run_pixel3dmm_phoenix.sbatch`
  in RGB2SMPLX, reused unchanged): 300 clips ~ 25 GPU-h.
- Dumping the tracker loss (decision E) would mean changing RGB2SMPLX's
  `rgb2smplx/stages/pixel3dmm.py`; not done (main pipeline stays untouched),
  so the Pixel3DMM confidence weight is 1.

### Why synthetic occlusion is the training signal and real occlusion still matters

On a real occluded frame there is no target: Pixel3DMM is unreliable there
too (its loss is switched off by `(1 - c_t)`). Pasting a hand on a clean clip
gives the target for free (the clean frame's pseudo-GT / TEASER output). Real
occluded clips still matter because (1) pasted PNG hands miss shadows, blur,
contact deformation and depth order, and a model trained only on pastes can
learn paste artefacts; (2) under real occlusion the face *detector* drifts and
the crop itself jumps -- reproduced only if the hand is pasted on the **full
frame before detection and cropping** (the default here); (3) the goal is real
video, so the final test must include real occlusion, measured with metrics
that need no GT (jitter/jerk, near-occlusion behaviour, mouth continuity).

---

## 4. Components

### 4.1 Feature extraction and cache -- `tools_temporal/extract_features.py`

Runs TEASER's frozen encoders on a work directory's `frames/` with the exact
crop pipeline of `rgb2smplx/stages/teaser.py` and writes one `.npz` per clip:

| key | shape | notes |
|---|---|---|
| `feat_expr` | (T, 960) fp16 | always |
| `feat_pose` | (T, 576) fp16 | `--temporal_feats expr+pose` or `all` |
| `feat_shape` | (T, 960) fp16 | `--temporal_feats all` |
| `expression`, `jaw_pose`, `eyelid`, `pose_params`, `cam`, `shape_params` | as `teaser.npz` | TEASER's own outputs |
| `tform` | (T, 3, 3) | frame px -> 224 crop px |
| `landmarks` | (T, 478, 2) | frame px, interpolated where not detected |
| `face_detected`, `valid` | (T,) bool | |
| `frame_index` | (T,) | position in `frames/` |
| `fps`, `crop_scale`, `checkpoint_md5` | scalars | |

`--save_spatial` additionally stores the pre-pool 7x7x960 map (fp16,
~94 KB/frame) in case spatial information turns out to matter under occlusion.
With synthetic occlusion active the forward is done online instead (encoders
frozen, `no_grad`).

### 4.2 Pixel3DMM pseudo-GT -- `tools_temporal/prepare_p3dmm_targets.py`

Reads the `pixel3dmm.npz` the RGB2SMPLX stage already writes (same `frames/`,
so frame alignment is by index, checked), never runs Pixel3DMM.
`--p3dmm_target`:
- `vertices`: canonical face vertices, FLAME(shape = reference identity,
  expr100, jaw, eyelid, global = neck = eyes = 0). Reference identity is the
  same for prediction and target so the loss isolates expression/jaw/eyelid.
- `refit50`: per-frame fit of 50 expr + jaw in TEASER's space minimising
  per-vertex L2 to the `vertices` mesh; eyelids copied (same blendshapes).
  Stores the fit residual per frame.
Optional per-frame tracker loss -> `p3dmm_conf`.

### 4.3 Occlusion confidence `c_t` -- `src/temporal/occlusion_conf.py`, `tools_temporal/compute_occlusion_conf.py`

Hand polygons (convex hull of 2D keypoints; `wilor.npz` `hand_keypoints_xy`,
or a generic JSON/NPZ of `(T, H, K, 2)`) mapped into the 224 crop with
`tform`; mouth / eye region polygons from TEASER's FLAME vertices projected
with its own orthographic `cam`. `c_mouth`, `c_eyes` = intersection over the
region's area, in [0, 1]. Missing keypoints -> `c = 0` with a warning.

### 4.4 Temporal adapter -- `src/temporal/adapter.py`

```
TemporalAdapter(feat_dim, arch='transformer', d_model=256, n_layers=2, window=16,
                causal=False, fusion='gated', mask_token=False, cond_dim=2)
  .forward(f: (B,T,F), c: (B,T,2) | None, frame_mask: (B,T) bool | None)
      -> f_tilde (B,T,F), aux {'delta', 'gate'}
```
- `--temporal_arch transformer | tcn | gru` (2-4 layers / dilated residual
  1D conv / bidirectional GRU), input/output projections to `d_model`,
  positional encoding for the transformer.
- `--temporal_window` (default 16 frames, sweep 8/16/32), `--temporal_causal`.
- `--temporal_fusion`: `residual` (`f + delta`), `gated`
  (`f + g * delta`, `g = sigmoid(MLP([f, delta, c]))`), `replace`
  (`alpha * f + T(f)`, `alpha` learned, starts at 1 -- TCMR-style but still
  identity at init, decision F).
- Output layer zero-initialised: at init the output equals TEASER for every
  fusion mode (tested).
- `--temporal_mask_token`: learned `[MASK]` replacing random frames in
  training (`--frame_mask_p 0.3`) and, optionally at inference, frames with
  `c_t > --occ_mask_thresh`.
- `--temporal_head frozen | full` (default `frozen`).
- Long videos: sliding window, default **overlap-add** (Hann, stride 4);
  `center` mode optional; edges padded by reflection.

### 4.5 Losses -- `src/temporal/losses.py` (weight 0 = off)

- `L_p3dmm_vertices`: L1 on canonical vertices, region weights
  `--w_region_eyes/--w_region_mouth/--w_region_rest`, weighted by
  `(1 - c_t)` per region and by `p3dmm_conf` if present. On synthetic
  occlusion, the target is the clean frame's, unweighted.
- `L_teaser_mouth`: position-only self-distillation on lip vertices and jaw
  against TEASER on the **clean** frame, pre-filtered with **SG5** (raw would
  teach the jitter, SG9 blunts fast mouthing). Decision C.
- `L_params` (optional, `refit50`): L2 on parameters.
- `L_accel`: vertex acceleration error **against Pixel3DMM** only (never
  against TEASER, which would re-teach the jitter; never towards zero).
- `L_lipread` (`--lipread_loss`): documented interface / stub for SPECTRE's
  lipreader; no weights downloaded without asking.

### 4.6 Synthetic occlusion -- `src/temporal/occlusion_aug.py` (`--synthetic_occlusion`)

RGBA hand patches (user-provided folder) pasted over mouth/eyes on the **full
frame before detection and cropping**, smooth trajectories within the window,
random scale/rotation, Lab mean/std colour transfer towards the face skin.
`--occ_aug_p` per clip, duration 2..T frames. Returns the synthetic `c_t`
computed from the pasted alpha mask.

### 4.7 Training and inference

- `tools_temporal/train_temporal.py`: windows of T frames from the caches,
  OmegaConf config (`configs/temporal/*.yaml`), logging, checkpoints of the
  adapter only (+ head if `full`).
- Demo (`main/demo_video.py`) and `tools_temporal/infer_clip.py`: with
  `--temporalize_teaser --temporal_ckpt path`, encoder -> adapter -> heads.
  Output for RGB2SMPLX: `teaser_temporal.npz`, same keys as `teaser.npz`, so
  `--face teaser` reads it unchanged.

### 4.8 Baselines -- `src/temporal/postproc.py` (`--postproc`)

- `sg9`: Savitzky-Golay window 9 on the parameters.
- `interp`: faithful copy of RGB2SMPLX's Hermite splice (episodes <= 15
  frames, +-1 extension, IoA threshold 0.20) -- copied, not imported, because
  TEASER's env is Python 3.9.
- `smoothnet`: light SmoothNet (per-dimension residual FC over time, sliding
  window) on the 55 parameters, trained on TEASER outputs with Pixel3DMM
  targets, same dataloader.
- SPECTRE: not integrated; `eval_temporal.py --external_preds dir` loads its
  predictions (same 50 FLAME 2020 expr + jaw space).

### 4.9 Evaluation -- `tools_temporal/eval_temporal.py`

Reported separately on **clean**, **near-occlusion** (+-`--near_k`, default
3) and **occluded** frames:
- vertex acceleration error and jitter;
- lip vertex error (per-frame max L2 over lip vertices, averaged);
- eye/eyelid vertex error, eye-aspect-ratio on blinks;
- full-face vertex error;
- FPS and ms/frame.
Synthetic-occlusion test: occlude clean clips and compare with (main) the
**same model on the clean clip** (consistency), (secondary) Pixel3DMM.
**Multiface + synthetic hands** against the real 3D GT is the headline number.
Output CSV/JSON; `tools_temporal/aggregate_results.py` builds the table.

---

## 5. Ablations (`configs/temporal/`, `tools_temporal/run_ablations.sh`)

Baselines: **B0** TEASER per-frame; **B1** B0 + SG9; **B2** B0 + Hermite
interpolation (current pipeline); **B3** B0 + SmoothNet (Pixel3DMM target);
**B4** SPECTRE (external predictions).

Temporal TEASER (all `--temporalize_teaser`):
- **A1** transformer, `residual`, T=16, `L_p3dmm_vertices` + `L_accel`
- **A2** A1 + `L_teaser_mouth`
- **A3** A2 + `gated` with `c_t`
- **A4** A3 + `--temporal_mask_token` + frame masking
- **A5** A4 + `--synthetic_occlusion`
- **A6** A5 + `--lipread_loss` (if integrated)
- **A7** A5 + `--temporal_head full` (replaces the LoRA variant, decision D)

Secondary, on A5: transformer / tcn / gru; T = 8 / 16 / 32 (also the fps
check: if 32 wins only on CSL, switch the window to seconds); causal vs not;
features `expr` / `expr+pose` / `all`; target `vertices` / `refit50`.

Defaults: `--temporal_feats expr`, `gated`, transformer 2 layers d=256
(~1.5 M parameters, negligible next to the encoders).

---

## 6. Decisions taken (2026-10-02)

| | question | decision |
|---|---|---|
| A | training data | Pixel3DMM on 300 PHOENIX train + 300 CSL train, ~75 % clean / 25 % real occlusion, chosen by `c_t`; first train A1 on 100+100 before scaling |
| B | RGB2SMPLX integration | none until a winner; temporal output written to `teaser_temporal.npz` |
| C | `L_accel` vs mouth distillation | `L_accel` against Pixel3DMM only; mouth distillation position-only against SG5-filtered TEASER-clean |
| D | `--temporal_head` | `frozen` / `full`; no LoRA |
| E | Pixel3DMM residual | weight 1 by default; dump tracker loss when re-running |
| F | `replace` fusion | `alpha * f + T(f)`, `alpha` starts at 1, so identity at init |
| G | GT for synthetic occlusion | same model on the clean clip (main), Pixel3DMM (secondary), Multiface real GT (headline) |
| H | window / inference | window in frames, default 16; overlap-add (Hann, stride 4) default |

---

## 7. Tests -- `tests/temporal/`

- split forward == `TeaserEncoder.forward` (bit for bit, CPU and GPU);
- cache outputs == existing `teaser.npz` on a real clip (GPU, batch 1);
- adapter shapes for every arch/fusion; identity at init;
- flag OFF unchanged (demo output before vs after);
- losses finite; augmentation `c_t` consistent with the pasted mask.

GPU tests run on the debug queue (`boost_qos_dbg`, 30 min cap).

---

## 8. Commands

(filled in as each component lands)

```bash
TPY=/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python
cd /leonardo_work/IscrC_SLPSCALE/TEASER_temporal
export PYTHONPATH=.

# unit tests (CPU part)
$TPY -m pytest tests/temporal -q

# 4.1 feature cache for one RGB2SMPLX work directory (batch 1 = identical to teaser.npz)
$TPY tools_temporal/extract_features.py <work_dir> <out.npz> \
    --checkpoint pretrained_models/TEASER.pt --temporal_feats expr

# step 1 checks on a GPU (debug queue): unit tests + cache == RGB2SMPLX stage on a real clip
sbatch tools_temporal/sbatch/test_step1.sbatch          # CLIP=<work_dir> to change clip

# clip lists for the Pixel3DMM train subsets (RGB2SMPLX env, reads the fits only)
sbatch tools_temporal/sbatch/select_phoenix_train.sbatch
```

Verified 2026-10-02 (job 59220919, A100): unit tests OK on CPU and CUDA; on
`csl/test/S005996_P0006_T00` (105 frames) every `teaser.npz` key of the cache
is identical to `rgb2smplx.stages.teaser --batch-size 1`.

---

## 9. Progress

- [x] Phase 0: exploration, plan, decisions (this file)
- [x] 4.1 split encoder + identity test + feature cache (verified, see 8)
- [ ] clip lists + Pixel3DMM sbatch for the train subsets (user launches) -- **blocked on frame access** (3)
- [ ] 4.2 pseudo-GT, 4.3 `c_t`
- [ ] 4.4 adapter, 4.5 losses, 4.7 training
- [ ] 4.6 synthetic occlusion
- [ ] 4.8 baselines, 4.9 evaluation, 5 ablations
