# Temporal TEASER (`--temporalize_teaser`)

TEASER is a per-frame regressor: each frame's face crop goes through four
independent MobileNetV3 encoders and a linear head, and nothing ties frame *t*
to *t+1* -- no temporal layer, no temporal loss, video datasets sampled one
frame at a time (`K: 1` in both configs; `K` and `LRS3_temporal_sampling` are
not read anywhere in the code).

In sign-language video the hands often pass in front of the face. TEASER,
seeing one frame at a time, is then wrong on the occluded frames and jittery
around them. Today coherence is added afterwards: Savitzky-Golay window 9 on
the fitted SMPL-X (SG9), then a cubic Hermite splice over short hand-face
occlusion episodes (`rgb2smplx/stages/face_jitter_fix.py` in RGB2SMPLX). The
splice is wrong where it matters most: the mouth keeps articulating
(mouthing) under the hand, and the frames next to the occlusion are already
corrupted.

## 0. Two problems, two tracks

| | goal | status |
|---|---|---|
| **Track 1** -- temporal coherence | make TEASER use neighbouring frames, so it is stable and robust to hand occlusion; teacher = TEASER itself | **this document, in progress** |
| Track 2 -- better face representation | improve what TEASER predicts (eyes, mouth detail) with extra supervision, e.g. Pixel3DMM pseudo-GT | parked, see 10 |

The two were first planned as one (Pixel3DMM as the target of the temporal
module); they are now separate so each result can be attributed. Track 1
does not depend on Track 2 in any way.

**What Track 1 can and cannot do.** The gain comes from two things the model
learns: to use the visible frames before and after the hand, and that the
features of a covered face are unreliable. Real hand-face occlusions are
**short**: 4.3 frames on average, median 3, 95 % within 12 frames (table in
4.3). So the job is mostly bridging a few frames from good context on both
sides, and cleaning up the frames next to the occlusion, which TEASER already
gets wrong; partial occlusions are where the image still helps most. The rare
long total occlusions (1-2 % over 15 frames) are where no model recovers the
mouthing: the best is a plausible continuation. Because episodes are short,
the interpolation baseline (T2) is a strong one to beat, and the success
criteria (7) are what tell a real temporal model from aggressive smoothing.

**Status:** Track 1 plan agreed 2026-10-03. Step 1 (split encoder + feature
cache) done and verified; see 12.

---

## 1. Ground rule: everything behind a flag

- Every change is behind `--temporalize_teaser` (and `temporal.enabled` in the
  YAML configs under `configs/temporal/`). **Default OFF.**
- With the flag OFF, training, demo and inference behave **identically** to the
  original: same outputs, same checkpoints. A test checks it numerically.
- New code lives in `src/temporal/` and `tools_temporal/`. Existing files get
  at most an `if args.temporalize_teaser:` branch. `TeaserEncoder` itself is
  not modified: the split forward (features -> heads) lives in
  `src/temporal/split_encoder.py`, and a test checks it equals
  `TeaserEncoder.forward` bit for bit.
- Sub-options are dedicated flags, only read when `--temporalize_teaser` is ON.
- Work happens on the `temporal` branch of the fork (worktree
  `/leonardo_work/IscrC_SLPSCALE/TEASER_temporal`), not on the submodule
  checkout RGB2SMPLX production runs from. RGB2SMPLX is not touched until the
  ablations pick a winner; then the output is `teaser_temporal.npz` with the
  same keys as `teaser.npz`, so `--face teaser` reads it unchanged.

---

## 2. The model as it is (Phase 0 findings)

Measured on the real model (TEASER env, `pretrained_models/TEASER.pt`;
`Teaser.pt`, `TEASER.pt`, `TEASER_v1.pt` are the same file).

| Encoder | timm backbone | pooled feature | head |
|---|---|---|---|
| Pose | `tf_mobilenetv3_small_minimal_100` | **576** | `Linear(576->6)`: `[0:3]` pose (axis-angle), `[3:6]` cam `[s,tx,ty]` (orthographic) |
| Shape | `tf_mobilenetv3_large_minimal_100` | 960 | `Linear(960->300)` |
| Expression | `tf_mobilenetv3_large_minimal_100` | **960** | `Linear(960->55)`: `[0:50]` expr, `[50:52]` eyelid `clamp(0,1)`, `[52]` jaw open `ReLU`, `[53:55]` jaw `clamp(+-0.2)` |
| Token | `tf_mobilenetv3_small_minimal_100` | 4 x 256 | only used by the training-time generator |

- Pooling is `adaptive_avg_pool2d(features[-1], 1)` on the 7x7 map, then a
  **single** linear layer. The adapter sees the expression feature (960), and
  optionally the pose feature too (`--temporal_feats expr+pose`, 1536). Shape
  is per-identity, never an adapter input.
- With a frozen linear head a feature delta acts on the output only through
  `W @ delta_f`: a feature-space adapter is a 55-dim parameter residual whose
  *input* is the 960-dim feature. Its edge over SmoothNet (parameters in,
  parameters out) is that richer input -- the 960 features can carry "this
  face is covered" in a way 55 parameters cannot. That is exactly what T3 vs
  T4 measures. LoRA on a 53k-parameter head is pointless:
  `--temporal_head frozen | full`.
- FLAME 2020 (`assets/FLAME2020/generic_model.pkl`), 300 shape + 50
  expression; neck and eyeballs fixed (**no gaze**); eyelids are two extra
  vertex blendshapes (`assets/l_eyelid.npy`, `r_eyelid.npy`).
- Regions for losses and metrics (`src/temporal/flame_regions.py`), from
  `assets/FLAME_masks/FLAME_masks.pkl`:
  **mouth** = `lips` (250 vertices); **eyes** = (`eye_region` U eyelid
  blendshape support) - eyeballs; **rest** = `face` - mouth - eyes. Eyeballs
  are excluded everywhere (no gaze). Vertices are canonical: FLAME with a fixed
  shared identity, global/neck/eye rotation zero, so the comparison isolates
  expression, jaw and eyelids.
- The original training (`main/train.py`) cannot run here
  (`datasets/preprocess_scripts/landmark.onnx` and its datasets are missing);
  temporal training is a separate script that never imports it.
- Crops in RGB2SMPLX (`rgb2smplx/stages/teaser.py`): pose-ROI ->
  FaceLandmarker landmarks (gaps interpolated), crop at scale 1.4 to 224.
  Batch size matters numerically: batch 1 reproduces per-frame output bit for
  bit, batched convs move expression by ~5e-3.

---

## 3. Teacher, targets and real occlusion

The teacher is **the original, frozen TEASER**. No external pseudo-GT.

- On a frame of the original video, the target is TEASER's own per-frame
  prediction on that frame.
- **Frames that are really occluded in the original video have no valid
  target** (TEASER is wrong there -- that is the problem being solved). Every
  per-frame target loss is multiplied by `(1 - c_t_real)`, in every window,
  clean or augmented. With `--real_occ_hard_thresh` (default 0.5) a frame
  above the threshold gets weight 0 outright.
- A synthetic hand pasted on a frame that is already really occluded gives
  no target either: weight 0.
- Real occlusion therefore enters training only as *input*: the model sees
  those frames but is never pulled towards TEASER's output on them.

`c_t_real` comes from the real hands (4.2), per region (`c_mouth`, `c_eyes`).
For the mouth/jaw parameters the weight uses `c_mouth`, for the eyelids
`c_eyes`; for vertex losses each region uses its own `c`.

---

## 4. Data

### 4.1 Corpora (continuous signing, signer-disjoint)

Checked 2026-10-03:

| corpus | clips with a SMPL-X fit (readable) | frames for TEASER | signers | fps |
|---|---|---|---|---|
| PHOENIX-2014-T (DGS) `/leonardo_work/IscrC_SLPSCALE/phoenix/phoenix/{train,dev,test}` | 7101 / 523 / 647 (Chang Liu) | readable (`frames/` links into the IscrC_SIGMA scratch) | 9 (`speaker` in the corpus CSV) | 25 |
| CSL-Daily (CSL, continuous) `/leonardo_work/IscrC_SIGMA/rgb2smplx/csl_daily/csldaily` | 9680 (~40 % of the corpus) | `frames/` deleted; source videos readable in `/leonardo_work/IscrC_SIGMA/csl_daily_crop/videos` (1280x1280, 30 fps); fits made with `--target-fps 30`, so frames = every video frame | P0000-02, P0004-05, P0007-08 (`_P00NN_` in the name) | 30 |
| CSL-Daily test (ours) `/leonardo_work/IscrC_SLPSCALE/csl/test` | 123 | readable | P0000, P0002-09 | 30 |
| How2Sign (ASL) | none yet | to extract | ~11 | 24 |

Not used: the isolated-sign CSL in `/leonardo_work/IscrC_SIGMA/rgb2smplx/csl`
(Track 1 needs continuous signing).

PHOENIX and CSL-Daily already went through the full RGB2SMPLX pipeline, so
for them Track 1 needs only the frames and its own TEASER feature cache:
occlusion and hand masks come from the existing fits. How2Sign needs the full
pipeline first (no fit = no occlusion meter, no hand masks); it joins in round
two.

### 4.2 Pilot dataset: 100 + 100 + 100 training clips

**Training uses clean video only.** Every occlusion the model learns from is
synthetic, pasted by us on clean frames, so every occluded frame has a true
unoccluded target. Real occlusion is measured only to throw it away, to size
the synthetic occlusions (4.3) and for the real-world test.

Data live in `data/temporal/<corpus>/` (git-ignored): `lists/` (candidates,
splits, `segments.json`, reports), `occ/`, `clips/` (frames), `cache/`,
`variants/`.

1. **Candidates**: CSL-Daily 150 per extracted signer (1050); PHOENIX 1500
   at random over train/dev/test (signers from the corpus CSVs).
2. **Occlusion meter (a) on the fits** (`compute_real_occlusion.py`): no
   frames needed.
3. **Selection** (`select_clean_clips.py`): a clip is eligible when <= 5 % of
   its frames are occluded (`c_mnc > 0.2`), no episode is longer than 15
   frames, and its **clean segments** hold >= 48 frames. Clean segment = a run
   of >= 24 frames with `c_mnc < 0.05`, at least 2 frames from any real
   episode. Training windows and synthetic hands stay inside the clean
   segments (`data.segments`, `make_synthetic_variants.py --segments`).
4. **Pick** per split, round-robin over signers, at most 30 per signer.
5. **Frames**: PHOENIX linked; CSL-Daily regenerated with RGB2SMPLX's
   `prepare --target-fps 30` and checked against the fit's frame count.
6. **TEASER cache** (expr+pose, batch 1), hand bank, 3 synthetic variants.

Selection, 2026-10-03:

| corpus | candidates | too much real occlusion | not enough clean frames | eligible | train | val | test |
|---|---|---|---|---|---|---|---|
| PHOENIX | 1500 (13 without a readable fit) | 872 | 133 | 482 | 100 clips, 9027 clean frames | 20 (Signer07) | 30 (Signer04, 08) |
| CSL-Daily | 1050 | 547 | 17 | 486 | 100 clips, 14001 clean frames | 20 (P0001) | 30 (P0005, P0007) |
| How2Sign | 168 chunks of 4 s from 12 shard videos | 17 | 1 | 150 | 87 clips, 7594 clean frames (woman) | 23 (man) | 40 (man) |

PHOENIX train per signer: Signer01 29, 03 28, 05 28, 09 14, 02 1 (Signer06:
no eligible clip). CSL-Daily train: 25 each for P0000, P0002, P0004, P0008.

### 4.2b Splits

- PHOENIX: test Signer04 + Signer08, validation Signer07, training the rest
  (signers span the official train/dev/test, so clips are drawn from all three).
- CSL-Daily: test P0005 + P0007, validation P0001, training P0000, P0002,
  P0004, P0008. (Our older `csl/test` clips were prepared at 15 fps and the
  three signers missing from the extracted 40 % have no 30 fps fit, so the
  test signers come from the extracted ones.)
- How2Sign: by signer once the signer of each video is known (4.2d).
- Lists: `data/temporal/<corpus>/lists/{train,val,test}.txt`; never changed.
- Our existing evaluation sets (the 100 PHOENIX test clips, the CSL test
  reel) are kept as an extra real-occlusion check.

### 4.2c Valid targets

Inside clean segments every frame is a valid target by construction. Real
occlusion outside them never enters training.

### 4.2d How2Sign (ASL)

Raw videos are long (2-3 min, 1280x720, 24 or 30 fps), so they are cut into
chunks, go through the full RGB2SMPLX pipeline (needed for meter (a) and the
hand masks), then the same selection. GPU budget: 2 hours, up to 4 GPUs in
parallel.

### 4.3 How long real occlusions are

Measured in RGB2SMPLX (`experiments/occlusion_protocols_smplx/results/segment_durations.json`):
episodes where the hands cover more than 20 % of the mouth/nose/chin region
(IoA > 0.2), at 25 fps.

| | episodes | mean | median | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| CSL test | 341 | 3.7 | 3 | 5 | 8 | 10 | 17 | 23 |
| PHOENIX test (100 clips) | 199 | 5.4 | 4 | 6 | 10 | 14 | 21 | 53 |
| combined | 540 | **4.3** | **3** | 5 | 9 | 12 | 18 | 53 |

Durations in frames. 40 % of episodes last 1-2 frames, 2.4 % more than 15,
0.4 % more than 25. This histogram is the starting distribution for the
synthetic occlusions until 5.2 recomputes it on the training signers.
Consequences: a 16-frame window always has context on both sides of a
typical episode (p99 = 18), so the window sweep is 8 / 16, with 32 only as a
check; synthetic durations follow this distribution instead of "anything up
to the whole window".

---

## 5. Components

### 5.1 Feature cache -- `tools_temporal/extract_features.py` (done)

Runs TEASER's frozen encoders on `<clip>/frames` with the exact crop pipeline
of `rgb2smplx/stages/teaser.py`; one `.npz` per clip:

| key | shape | notes |
|---|---|---|
| `feat_expr` | (T, 960) fp32 | always (fp32 so the heads reproduce TEASER exactly) |
| `feat_pose` | (T, 576) fp32 | `--temporal_feats expr+pose` |
| `expression`, `jaw_pose`, `eyelid`, `pose_params`, `cam`, `shape_params` | as `teaser.npz` | TEASER's outputs = the teacher |
| `tform` | (T, 3, 3) | frame px -> 224 crop px |
| `landmarks` | (T, 478, 2) | frame px, interpolated where not detected |
| `face_detected`, `pose_valid`, `valid` | (T,) bool | |
| `frame_index`, `fps`, `crop_scale`, `checkpoint_md5` | | |

`--save_spatial` adds the pre-pool 7x7x960 map (~94 KB/frame), in case spatial
information turns out to matter under occlusion. MediaPipe Pose's tracker is
reset per clip (the RGB2SMPLX stage does not reset it between clips run in
one process; for one clip in a fresh process the two are identical).

### 5.2 Real occlusion `c_t_real` and its statistics -- `src/temporal/occlusion_conf.py`, `tools_temporal/occlusion_stats.py`

- Hand polygons: convex hull of the 2D hand keypoints (WiLoR / MediaPipe Hands
  / a generic `(T, H, K, 2)` NPZ), mapped into the 224 crop with `tform`.
- Region polygons: TEASER's canonical-region vertices projected with its own
  orthographic `cam`.
- `c_mouth`, `c_eyes` = area(hands n region) / area(region), in [0, 1].
  Missing hands -> `c = 0` with a warning.
- `occlusion_stats.py` aggregates `c_t_real` over the **training** signers of
  each corpus into `configs/temporal/occlusion_stats_<corpus>.json`: episode
  length distribution, episodes per second, IoA distribution inside an
  episode (partial vs total), which region. Synthetic occlusions are sampled
  from it (5.4).

### 5.3 Temporal adapter -- `src/temporal/adapter.py`

```
TemporalAdapter(feat_dim, arch='transformer', d_model=256, n_layers=2, window=16,
                causal=False, fusion='gated', mask_token=False, cond_dim=2)
  .forward(f: (B,T,F), c: (B,T,2) | None, frame_mask: (B,T) bool | None)
      -> f_tilde (B,T,F), aux {'delta', 'gate'}
```
- `--temporal_arch transformer | tcn | gru` (2-4 layers / dilated residual 1D
  conv / bidirectional GRU), in/out projections to `d_model`, positional
  encoding for the transformer.
- `--temporal_window` (default 16 frames; sweep 8/16/32), `--temporal_causal`.
- `--temporal_fusion`: `residual` (`f + delta`); `gated` (`f + g * delta`,
  `g = sigmoid(MLP([f, delta, c]))`); `replace` (`alpha * f + T(f)`, `alpha`
  learned and starting at 1, so still identity at init).
- Output layer zero-initialised: at init the output **equals TEASER** for every
  fusion mode (tested).
- `--temporal_mask_token`: learned `[MASK]` that replaces the features of
  random frames in training (`--frame_mask_p`, default 0.15) and, optionally
  at inference, frames with `c_t > --occ_mask_thresh`.
- `c_t` given to the adapter at inference is the real one (5.2); in training
  it is real `c` combined with the synthetic one (`max`).
- `--temporal_head frozen | full` (default `frozen`).
- Long videos: sliding window, default **overlap-add** (Hann, stride 4);
  `center` mode optional; edges padded by reflection.

### 5.4 Synthetic occlusion -- `src/temporal/occlusion_aug.py` (`--synthetic_occlusion`)

Hands are pasted on the **full frame before face detection and cropping**, so
the face detector and the crop react as they do to a real hand, and only
inside clean segments, so every covered frame has a clean target. Variants
are made offline (`tools_temporal/make_synthetic_variants.py`): each is a
full re-extraction stored like a clean cache plus `c_syn` (coverage of lips
and eyes per frame), `syn_mask` (any pasted hand), `core_mask`, the plan and
the clean cache it belongs to. A synthetic frame counts as occluded at
coverage > 0.2, like a real one.

**Regions and coverage** are those of real occlusion: the fit's FLAME lips /
eye region projected with the fit's camera (`export_face_regions.py`),
coverage = share of the region's convex hull under the hand (alpha > 0.5).

**Hand bank** (`build_hand_bank.py`, per corpus, `data/temporal/hand_bank/<corpus>/{train,heldout}`):
cut-outs from our own frames where the hand is away from the face; the fit's
MANO hand plus a forearm stub seeds GrabCut; scale-free sharpness, a
skin-colour check against mostly-sleeve cut-outs, and the forearm faded out
so the arm does not end in a straight cut. Training hands come from training
signers, held-out hands from validation / test signers; hands are pasted on
the same corpus only.

**Pasting**: the hand's Lab means move halfway to the face's skin (texture
kept), the edge is feathered in proportion to the hand's size, a soft shadow
darkens the face beside it.

Two ways to move the hand (both made, compared on the review page):

- **Parametric (v2)**: episodes sampled from the real statistics -- duration,
  peak coverage, region (eyes instead of mouth at the real *eyes-only* rate,
  4-5 %) -- with 2-4 entry and exit frames; on every core frame the hand's
  offset from the region centre is bisected to the target coverage (median
  error 0.009 on a preview).
- **Replay** (`build_replay_bank.py`): the dynamics of real episodes from the
  same corpus and split, read from the fits and occlusion files: per frame
  the hand's position relative to the lips in the face's frame, its size,
  its wrist-to-hand direction and the source's lip-coverage curve, with the
  episode's own approach and retreat; the hand is a bank cut-out of the same
  side with a near MANO pose; each frame is calibrated to the source's
  coverage and motion-blurred by the hand's speed. Durations, motion and
  coverage curves are real by construction.

Tried and dropped: transplanting the real occluding hand's *pixels*
(`build_transplant_bank.py`). In front of the face the fitted hand is not
accurate enough to separate hand skin from face skin, and the cut-outs carry
the source signer's lips, nose or eye -- the worst possible corruption for a
model meant to read the mouth.

History: the first variants (`variants_v1/`) put 44 % of PHOENIX hands on the
eyes (the rate of mouth episodes whose hand *also* touches the eyes was read
as an eyes-instead-of-mouth rate), under-covered the lips (median 0.69 vs
0.97 real) and missed the region in 15-29 % of episodes; v2 fixes all three.

`HandBank('procedural')` draws a skin-coloured palm-and-fingers shape: smoke
tests only.

### 5.5 Losses -- `src/temporal/losses.py` (weight 0 = off)

All per-frame target terms carry the real-occlusion weight of 3.

- `L_self`: on frames with a valid target, L1 between the adapter's output and
  TEASER per-frame, on canonical vertices (per-region weights
  `--w_region_mouth/eyes/rest`) or on the 55 parameters --
  `--self_target vertices | params`.
- `L_occ`: on frames covered by a synthetic hand (and their +-k neighbours),
  the target is TEASER on the **original, unoccluded** frame. Same form and
  weights as `L_self`, separate weight `--w_occ`.
- `L_accel`: penalises the second difference of the **predicted** vertices
  (towards zero), weight `--w_accel`. Its sweep is the main stability vs
  fidelity trade-off; the success criteria (7.1) are what keeps it honest.
- `--self_target_smoothed`: variant where the target of `L_self`/`L_occ` is
  TEASER + SG9 instead of raw TEASER (an ablation, not the default).

### 5.6 Training recipe -- `tools_temporal/train_temporal.py`

Per training window of T frames:
1. With probability `--occ_aug_p` (default 0.5) the window gets synthetic
   occlusion; otherwise it stays clean.
2. In an occluded window the hand covers a contiguous run of frames whose
   length is sampled from the real distribution (4.3: median 3, mean ~4, tail
   to ~20), placed so that most episodes have visible frames on both sides,
   with total and partial coverage. The visible frames before and after are
   the ones the model has to learn to use. A window can hold more than one
   episode, at the real episode rate.
3. Independently, frame masking (`[MASK]` in place of some features) as a
   third, simpler kind of disturbance.
4. In every window, clean or not, per-frame target losses are attenuated by
   `(1 - c_t_real)` (3).

Clean windows come from the clean caches, occluded windows from the
precomputed variants (5.4), placed so the episode has visible frames on
both sides; neither needs the encoder at training time (one step ~35 ms on
an A100 at batch 32). `--occ_curriculum`: optional, a few epochs with p = 0,
then p ramps up -- only if training is unstable at the start.

OmegaConf configs in `configs/temporal/*.yaml`; checkpoints hold the adapter
only (+ head if `full`).

### 5.7 Inference

Demo (`main/demo_video.py`) and `tools_temporal/infer_clip.py`: with
`--temporalize_teaser --temporal_ckpt path`, encoder -> adapter -> heads;
without the flag, as before. Output `teaser_temporal.npz`.

### 5.8 Baselines -- `src/temporal/postproc.py` (`--postproc`)

- `sg9`: Savitzky-Golay window 9 on the parameters.
- `interp`: faithful copy of RGB2SMPLX's Hermite splice (episodes <= 15
  frames, +-1 extension, IoA threshold 0.20); copied, not imported (TEASER's
  env is Python 3.9).
- `smoothnet`: light SmoothNet (per-dimension residual FC over time, sliding
  window) on the 55 parameters, trained with **the same self-supervised
  recipe** as the adapter (same windows, synthetic occlusion, `L_self`,
  `L_occ`, `L_accel`), so the only difference is its input.
- SPECTRE (optional): not integrated; `eval_temporal.py --external_preds dir`.

---

## 6. Evaluation -- `tools_temporal/eval_temporal.py`

Everything reported separately on **clean**, **near-occlusion** (+-`--near_k`,
default 3) and **occluded** frames.

- **Stability:** vertex acceleration and jitter (mouth, eyes, whole face).
- **Fidelity on clean frames:** vertex error vs TEASER per-frame, per region.
- **Recovery under synthetic occlusion:** occlude clips of the test set (7.2)
  and compare with **TEASER on the same clip unoccluded** (mouth, eyes,
  total). Secondary: the same model on the unoccluded clip (consistency).
- **Real occlusion** (no GT): stability and continuity of the mouth across
  real episodes, on the test signers and on our existing PHOENIX/CSL reels.
- **Speed:** FPS and ms/frame, encoder vs adapter.
- CSV/JSON per run; `tools_temporal/aggregate_results.py` builds the table.

---

## 7. When Track 1 counts as a success

### 7.1 Better on occlusion without being worse elsewhere

A model can do very well on synthetic occlusions by smoothing everything.
Training succeeded only if, **at the same time**, on the test set:

1. on occluded and near-occlusion frames, the error vs the unoccluded clip is
   clearly lower than TEASER, SG9, interpolation **and** SmoothNet;
2. on clean frames, the error vs TEASER stays small -- the mouthing has not
   been erased;
3. jitter and acceleration are lower than TEASER and at least comparable to
   SG9.

If only 1 holds, the result is aggressive smoothing, not a temporal model.

### 7.2 A synthetic test set that is really new

Otherwise it measures how well the model memorised the training hands:
- **different signers** from training (4.2);
- **different hands**: a held-out group of hand PNGs never used in training,
  and test trajectories generated differently (replayed real hand motion);
- **realistic durations and positions**: length, frequency and coverage of
  the synthetic occlusions sampled from the real statistics (4.3, 5.2) of
  PHOENIX and CSL -- short episodes dominate (median 3 frames) -- with the
  tail (10-20 frames) and partial coverage represented, and results reported
  per duration bucket (1-2, 3-5, 6-10, >10 frames) so a win on the common
  short case cannot hide a loss on the long one.

The test set is generated once with a fixed seed and stored
(`tools_temporal/build_occlusion_testset.py`), so every method is scored on the
same occlusions.

---

## 8. Ablations (`configs/temporal/`, `tools_temporal/run_ablations.sh`)

Baselines and models:
- **T0** TEASER per-frame
- **T1** T0 + SG9
- **T2** T0 + interpolation (current pipeline)
- **T3** T0 + SmoothNet (same self-supervised recipe)
- **T4** adapter, `residual`, `L_self` + `L_accel`
- **T5** T4 + `gated` with `c_t`
- **T6** T5 + mask token and frame masking
- **T7** T6 + synthetic occlusion with `L_occ`

Sweeps (on T7 unless stated):
- `w_accel` (the main stability/fidelity trade-off; also on T4)
- `occ_aug_p` 0.3 / 0.5 / 0.7 -- read together the occluded-frame error and
  the clean-frame fidelity; higher p should help the first and cost a little
  on the second, and the test set decides
- window 8 / 16 frames (32 only as a check; real episodes are short, 4.3)
- architecture transformer / tcn / gru; causal vs not
- features `expr` vs `expr+pose`; `--self_target vertices | params`;
  `--self_target_smoothed`
- optional: `--occ_curriculum`, `--temporal_head full`

Defaults: `--temporal_feats expr`, `gated`, transformer 2 layers d=256
(~2.4-3.6 M parameters depending on gate / mask token; ~0.02 ms per frame,
negligible next to the encoders).

---

## 9. Decisions

| | question | decision |
|---|---|---|
| 1 | teacher | original frozen TEASER; no Pixel3DMM in Track 1 (2026-10-03) |
| 2 | real occlusion in training | input only; every target loss x `(1 - c_t_real)`, 0 above a threshold |
| 3 | `L_accel` | on the predicted vertices, towards zero, swept; kept honest by 7.1 |
| 4 | data | several sign languages, signer-disjoint splits; frames + hands only |
| 5 | GT for synthetic occlusion | TEASER on the unoccluded clip (main), same-model consistency (secondary) |
| 6 | `--temporal_head` | `frozen` / `full`; no LoRA |
| 7 | `replace` fusion | `alpha * f + T(f)`, `alpha` starts at 1 |
| 8 | window / inference | in frames, default 16; overlap-add (Hann, stride 4) |
| 10 | synthetic occlusion durations | from the real distribution: mean ~4 frames, median 3, tail to ~20 |
| 11 | hand patches | cut from our own frames, split by signer |
| 9 | RGB2SMPLX | untouched until a winner; then `teaser_temporal.npz` |

---

## 10. Track 2 (parked): better representation

Not part of Track 1; notes kept for later.
- Pixel3DMM pseudo-GT (FLAME 2020, 100 expressions, gaze) exists for the 100
  PHOENIX test clips and 49 Multiface clips; RGB2SMPLX's stage
  `rgb2smplx/stages/pixel3dmm.py` writes it aligned to `frames/`. ~12 clips per
  GPU-hour on PHOENIX.
- TEASER's 50 expressions are the first 50 of the same FLAME 2020 basis, so a
  Pixel3DMM target can be expressed as canonical vertices or refit to 50
  coefficients by least squares.
- Multiface (`multiface/mf50`) has real 3D GT: the right place to measure a
  representation change.
- SPECTRE's lipreading loss belongs here too (mouth detail), not in Track 1.

---

## 11. Tests -- `tests/temporal/`

- split forward == `TeaserEncoder.forward` (bit for bit, CPU and CUDA);
- cache outputs == RGB2SMPLX's `teaser.npz` on a real clip (GPU, batch 1);
- adapter shapes for every arch/fusion; identity at init;
- flag OFF unchanged (demo output before vs after);
- losses finite; real-occlusion weighting zeroes the right frames;
  augmentation `c_t` consistent with the pasted mask.

No pytest in the TEASER env: tests are `unittest`. GPU tests run on the debug
queue (`boost_qos_dbg`, 30 min cap).

---

## 12. Pipeline, commands and progress

| step | script | env | output |
|---|---|---|---|
| clip dirs (PHOENIX we can read) | `tools_temporal/make_phoenix_clips.py` | TEASER | `<root>/<clip>/frames`, `video.json`, `signers.tsv` |
| real occlusion | `tools_temporal/compute_real_occlusion.py` (from the TEASER-face SMPL-X fit) | RGB2SMPLX | `<clip>.occ.npz` |
| occlusion statistics | `tools_temporal/occlusion_stats.py` | any | `occlusion_stats.json` |
| clean features | `tools_temporal/extract_features.py` | TEASER | `<clip>.npz` |
| synthetic variants | `tools_temporal/make_synthetic_variants.py` | TEASER | `<clip>.v<k>.npz` |
| training | `tools_temporal/train_temporal.py configs/temporal/T*.yaml key=value ...` | TEASER | `config.yaml`, `log.jsonl`, `last.pt`, `best.pt` |
| evaluation | `tools_temporal/eval_temporal.py --method NAME=teaser|sg9|sg9+interp|ckpt.pt` | TEASER | JSON + table + 7.1 checks |
| inference | `tools_temporal/infer_clip.py` | TEASER | `teaser_temporal.npz` (keys of `teaser.npz`) |
| demo | `main/demo_video.py ... --temporalize_teaser --temporal_ckpt ckpt.pt` | TEASER | crop / TEASER / temporal video |

End-to-end smoke test: `tools_temporal/sbatch/phoenix_smoke.sbatch` (6
PHOENIX test clips, one per signer, from 30 candidates outside the 100
evaluation clips), then `tools_temporal/sbatch/smoke_pipeline.sbatch`.

```bash
TPY=/leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser/bin/python
cd /leonardo_work/IscrC_SLPSCALE/TEASER_temporal
export PYTHONPATH=.

# unit tests (CPU, and CUDA when visible)
$TPY -m unittest discover -s tests/temporal -p 'test_*.py' -v

# feature cache for one clip directory (batch 1 = identical to teaser.npz)
$TPY tools_temporal/extract_features.py <clip_dir> <out.npz> \
    --checkpoint pretrained_models/TEASER.pt --temporal_feats expr

# step 1 checks on a GPU (debug queue): unit tests + cache == RGB2SMPLX stage
sbatch tools_temporal/sbatch/test_step1.sbatch          # CLIP=<work_dir> to change clip
```

Verified 2026-10-02 (job 59220919, A100): unit tests OK on CPU and CUDA; on
`csl/test/S005996_P0006_T00` (105 frames) every `teaser.npz` key of the cache
is identical to `rgb2smplx.stages.teaser --batch-size 1`.

- [x] Phase 0 and plan (this file)
- [x] 5.1 split encoder + identity test + feature cache
- [x] 5.2 real occlusion (from the TEASER-face SMPL-X fits) + statistics
- [x] 5.3 adapter, 5.5 losses, 5.8 baselines (SG9, Hermite splice, SmoothNet)
- [x] 5.4 synthetic occlusion (procedural hands until the hand set exists)
- [x] 5.6 training, 6 evaluation, 5.7 inference + demo flag
- [x] whole pipeline run end to end on 6 PHOENIX test clips (2026-10-03, see below)
- [x] pilot dataset built (2026-10-03): 300 + 100 val/test clips over PHOENIX, CSL-Daily,
  How2Sign; hand banks (1688 training / 974 held-out cut-outs); 1187 synthetic variants;
  review page with statistics and examples
- [ ] fix before training: hand placement (anchor on the hand, not the cut-out box),
  hand scale (hand part only), cut-out artefacts, flat colour transfer; then regenerate variants
- [ ] held-out test trajectories replaying real hand motion (7.2)
- [ ] training data (several corpora, signer splits, 4.1-4.2)
- [ ] real trainings and 8 ablations

Smoke test, 2026-10-03 (jobs 59256529, 59256974, 59257396, A100): 4 train /
2 val clips (different signers), 2 procedural-hand variants per clip, 200
steps. Every stage runs; unit tests pass; `teaser_temporal.npz` has the keys
and shapes of `teaser.npz`; **demo with the flag OFF gives output identical
to the production TEASER checkout** (130/130 frames); demo with the flag ON
writes the 3-panel video. The numbers are not results (4 clips, 200 steps),
only a sanity check that the losses do what they should: on the val
variants T7 lowers the mouth error under synthetic occlusion from 14.8 mm
(T0) to 6.2 mm (T2: 9.7) with 0.18 mm clean-frame drift from TEASER, while
T4 (no synthetic occlusion) stays at TEASER (0.08 mm drift) and does not
help under occlusion. Jerk is still far above SG9 at this length of
training (`w_accel` untuned). Training ~35 ms/step, the adapter ~0.02 ms
per frame at inference.

### Pilot dataset v3, 2026-10-04 (review page: `tools_temporal/sbatch/pilot_stats_report.sbatch`)

Fakes calibrated on the protocol's mouth/nose/chin coverage (5.4). Per episode,
real (whole corpus) vs parametric vs replay:

| corpus | median duration (ms) | median peak mnc | peak >= 0.6 |
|---|---|---|---|
| PHOENIX | 120 / 120 / 120 | 0.42 / 0.40 / 0.39 | 28 % / 22 % / 23 % |
| CSL-Daily | 133 / 133 / 100 | 0.41 / 0.40 / 0.37 | 24 % / 22 % / 21 % |
| How2Sign | 100 / 100 / 100 | 0.26 / 0.27 / 0.40 (pooled dynamics) | 6 % / 6 % / 24 % |

Occluded frames: real corpus 10.5 % (PHOENIX), 7.6 % (CSL-Daily), 1.3 %
(How2Sign); selected training clips 2.0 / 2.0 / 0.4 % (outside their clean
segments); training windows at occ_aug_p 0.5: 9-11 % (0.3: ~6 %, 0.7: 13-15 %).

### Ablations, 2026-10-05 (validation; `runs/eval_phase*_val.json`, review page)

Mouth error under the hand / next to it (mm), all followed by SG9 unless noted:
current pipeline (SG9 + Hermite) 5.26 / 2.13; SmoothNet 4.60 ± 0.02; T7 replay
1.95 ± 0.01; **T7 mixed fakes 1.92 ± 0.03 / 1.02** (3 seeds; clean-frame drift
0.29 mm and jerk 0.18, as SG9 alone). Without fakes (T4-T6) the adapter stays
at TEASER. Architecture: transformer best; window 32 slightly better (1.86);
pose features / trained head no gain; GRU 2.12, window 8 2.18, causal 2.34,
TCN 2.41; loss on parameters needs rebalancing (not comparable as run). All
three corpora together beat any single corpus on every corpus.
