"""Train temporal TEASER (or the SmoothNet baseline) -- TEMPORAL_README.md 5.6.

    python tools_temporal/train_temporal.py configs/temporal/T7.yaml out_dir=runs/T7 \
        data.cache_dir=... data.occ_dir=... data.variant_dir=... data.train_list=... data.val_list=...

Config = configs/temporal/base.yaml, then the run's YAML, then ``key=value``
overrides. Teacher = the frozen TEASER outputs in the caches. Per step:

- a batch of windows, a fraction ``train.occ_aug_p`` from synthetic variants;
- ``L_self`` on frames away from any pasted hand, ``L_occ`` on pasted frames
  and their +-``loss.near_k`` neighbours, both against TEASER on the
  unoccluded frames and both weighted by the real occlusion (1 - c_real, 0
  above ``loss.hard_thresh``);
- ``L_accel`` on the predicted canonical vertices.

Vertices are in millimetres. Writes ``<out_dir>/config.yaml``, ``log.jsonl``
(training losses and validation metrics) and ``last.pt`` / ``best.pt``
(best = lowest validation score, see ``val_score``).
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(argv):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(REPO_ROOT / "configs/temporal/base.yaml")
    rest = list(argv)
    if rest and rest[0].endswith((".yaml", ".yml")):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(rest.pop(0)))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(rest))
    if not cfg.temporal.enabled:
        raise SystemExit("temporal.enabled is false: nothing to train")
    return cfg


def collate(samples):
    import torch

    out = {}
    for key, value in samples[0].items():
        if isinstance(value, dict):
            out[key] = {k: torch.stack([s[key][k] for s in samples]) for k in value}
        else:
            out[key] = torch.stack([s[key] for s in samples])
    return out


def to_device(batch, device):
    return {k: ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device))
            for k, v in batch.items()}


def compute_losses(pred, batch, flame, region_index, cfg):
    import torch
    from src.temporal import losses as L

    lc = cfg.loss
    valid = batch["valid"]
    weights = L.region_weights(batch["c_real"][..., 0], batch["c_real"][..., 1], lc.hard_thresh)
    weights = {k: v * valid.to(v.dtype) for k, v in weights.items()}
    syn_near = L.near_frames(batch["syn_mask"], lc.near_k) & valid
    clean_sel = valid & ~syn_near

    pred_v = flame(pred["expression"], pred["jaw"], pred["eyelid"]) * 1000.0
    out = {}
    if lc.self_target == "vertices":
        with torch.no_grad():
            t = batch["teacher"]
            target_v = flame(t["expression"], t["jaw"], t["eyelid"]) * 1000.0
        region_w = dict(lc.w_region)
        out["self"], parts = L.vertex_target_loss(pred_v, target_v, weights, region_index, region_w, clean_sel)
        out["occ"], _ = L.vertex_target_loss(pred_v, target_v, weights, region_index, region_w, syn_near)
        for region, value in parts.items():
            out[f"self_{region}"] = value.detach()
    elif lc.self_target == "params":
        out["self"], _ = L.param_target_loss(pred, batch["teacher"], weights, clean_sel)
        out["occ"], _ = L.param_target_loss(pred, batch["teacher"], weights, syn_near)
    else:
        raise ValueError(f"unknown self_target {lc.self_target!r}")
    out["accel"] = L.accel_loss(pred_v, valid)
    out["total"] = lc.w_self * out["self"] + lc.w_occ * out["occ"] + lc.w_accel * out["accel"]
    return out


def evaluate(model, clips, variants, flame, region_names, cfg, device, with_baseline=False):
    """Validation metrics pooled over clips: real clips (fidelity, stability) and variants (recovery)."""
    import torch
    from src.temporal import metrics as M
    from src.temporal.data import full_clip

    region_index = {r: flame.region_index(r) for r in region_names}
    region_index["face"] = np.arange(flame.vertex_ids.numel())
    model.eval()
    results = {"real": [], "synthetic": []}
    baseline = {"real": [], "synthetic": []}
    for kind, items in (("real", clips), ("synthetic", variants)):
        for clip in items:
            batch = to_device(full_clip(clip), device)
            pred = model.infer_clip(batch, window=cfg.train.window, stride=cfg.train.infer_stride)
            pred = {k: v[0].cpu().numpy() for k, v in pred.items() if not k.startswith("_")}
            ref = clip.teacher
            valid = clip.valid
            real_occ = clip.c_mnc > 0.2
            if kind == "real":
                groups = M.frame_groups(real_occ, valid, cfg.loss.near_k + 1)
            else:
                groups = M.frame_groups(clip.syn_mask, valid, cfg.loss.near_k + 1, exclude=real_occ)
            ref_v = M.canonical_mm(flame, ref, device)
            episodes = clip.episodes if kind == "synthetic" else None
            results[kind].append(M.clip_metrics(M.canonical_mm(flame, pred, device), ref_v, groups,
                                                region_index, episodes))
            if with_baseline:
                baseline[kind].append(M.clip_metrics(M.canonical_mm(flame, clip.input_params, device), ref_v,
                                                     groups, region_index, episodes))
    model.train()
    out = {kind: M.pool(r) for kind, r in results.items() if r}
    if with_baseline:
        out["teaser"] = {kind: M.pool(r) for kind, r in baseline.items() if r}
    return out


def val_score(metrics):
    """Lower is better: recovery under synthetic occlusion + fidelity on clean real frames + stability."""
    score, parts = 0.0, 0
    syn = metrics.get("synthetic", {})
    for group in ("occluded", "near"):
        value = syn.get(group, {}).get("err_mouth", float("nan"))
        if not math.isnan(value):
            score, parts = score + value, parts + 1
    real = metrics.get("real", {}).get("clean", {})
    for key in ("err_face", "accel"):
        value = real.get(key, float("nan"))
        if not math.isnan(value):
            score, parts = score + value, parts + 1
    return score if parts else float("inf")


def main():
    cfg = load_config(sys.argv[1:])
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    from omegaconf import OmegaConf
    from src.temporal.data import WindowDataset, load_clips, load_variants, read_list
    from src.temporal.flame_regions import CanonicalFlame
    from src.temporal.losses import REGIONS
    from src.temporal.model import build_model
    from src.temporal.split_encoder import load_teaser_encoder

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    dc = cfg.data
    train_clips = load_clips(read_list(dc.train_list), dc.cache_dir, dc.occ_dir, dc.feature_set, dc.teacher_smoothing,
                             dc.get("segments"))
    val_clips = load_clips(read_list(dc.val_list), dc.cache_dir, dc.occ_dir, dc.feature_set, dc.teacher_smoothing)
    train_variants = load_variants(train_clips, dc.variant_dir, dc.feature_set) if dc.variant_dir else []
    val_variants = load_variants(val_clips, dc.variant_dir, dc.feature_set) if dc.variant_dir else []
    print(f"[train] {len(train_clips)} train clips ({sum(len(c) for c in train_clips)} frames), "
          f"{len(train_variants)} variants; {len(val_clips)} val clips, {len(val_variants)} variants")

    encoder = load_teaser_encoder(cfg.checkpoint, device)
    encoder.requires_grad_(False)
    flame = CanonicalFlame(vertex_ids="face").to(device)
    region_index = {r: torch.as_tensor(flame.region_index(r), device=device) for r in REGIONS}
    model = build_model(cfg.model, encoder, dc.feature_set, cfg.train.window).to(device)
    if cfg.model.get("kind", "adapter") == "smoothnet":
        model.set_normalisation(train_clips)
    del encoder
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[train] {sum(p.numel() for p in trainable) / 1e6:.2f} M trainable parameters")

    tc = cfg.train
    dataset = WindowDataset(train_clips, train_variants, tc.window, tc.samples_per_epoch, tc.occ_aug_p, cfg.seed)
    optimizer = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)
    warmup = min(100, tc.steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(warmup, 1)) * 0.5 * (1 + math.cos(math.pi * min(s, tc.steps) / tc.steps)))

    log = open(out_dir / "log.jsonl", "a")
    initial = evaluate(model, val_clips, val_variants, flame, REGIONS, cfg, device, with_baseline=True)
    log.write(json.dumps({"step": 0, "val": initial}) + "\n")
    log.flush()
    best = val_score(initial)
    print(f"[train] step 0 (= TEASER) val score {best:.4f}")

    rng = np.random.default_rng(cfg.seed)
    started, running = time.time(), {}
    for step in range(1, tc.steps + 1):
        if tc.occ_curriculum_steps:
            dataset.occ_aug_p = (tc.occ_aug_p * min(1.0, step / tc.occ_curriculum_steps)
                                 if dataset.variants else 0.0)
        dataset.epoch = step
        batch = to_device(collate([dataset[int(i)] for i in rng.integers(0, len(dataset), tc.batch_size)]), device)
        pred = model(batch)
        losses = compute_losses(pred, batch, flame, region_index, cfg)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v)
        if step % tc.log_every == 0:
            entry = {"step": step, "lr": scheduler.get_last_lr()[0],
                     **{k: v / tc.log_every for k, v in running.items()},
                     "s_per_step": (time.time() - started) / step}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print("[train] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in entry.items()))
            running = {}
        if step % tc.val_every == 0 or step == tc.steps:
            metrics = evaluate(model, val_clips, val_variants, flame, REGIONS, cfg, device)
            score = val_score(metrics)
            log.write(json.dumps({"step": step, "val": metrics, "score": score}) + "\n")
            log.flush()
            state = {"config": OmegaConf.to_container(cfg), "step": step, "score": score,
                     "model": {k: v for k, v in model.state_dict().items()}}
            torch.save(state, out_dir / "last.pt")
            if score < best:
                best = score
                torch.save(state, out_dir / "best.pt")
            print(f"[train] step {step} val score {score:.4f} (best {best:.4f})")
    log.close()


if __name__ == "__main__":
    main()
