"""Temporal TEASER on one clip -> ``teaser_temporal.npz`` (TEMPORAL_README.md 5.7).

    python tools_temporal/infer_clip.py <work_dir> <out.npz> --temporal_ckpt runs/T7/best.pt \
        [--cache <clip feature cache>] [--occ <clip.occ.npz>]

Same keys as RGB2SMPLX's ``teaser.npz`` (so ``--face teaser`` reads it
unchanged): ``expression``, ``jaw_pose``, ``eyelid`` come from the temporal
model; ``pose_params``, ``cam``, ``shape_params``, ``valid``, ``frame_index``
are TEASER's own per-frame values. Without ``--cache`` the features are
extracted first (the same crops as RGB2SMPLX's stage). Without ``--occ`` the
adapter's occlusion input is 0 on every frame.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--temporal_ckpt", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--occ", type=Path)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    work_dir, output, ckpt = args.work_dir.resolve(), args.output.resolve(), args.temporal_ckpt.resolve()
    cache_path = args.cache.resolve() if args.cache else None
    occ_path = args.occ.resolve() if args.occ else None
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    import tempfile

    import torch
    from src.temporal.data import Clip, full_clip
    from tools_temporal.eval_temporal import load_checkpoint
    from tools_temporal.train_temporal import to_device

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(ckpt, device)
    started = time.time()
    if cache_path is None:
        from src.temporal.split_encoder import load_teaser_encoder
        from tools_temporal.extract_features import extract_clip

        encoder = load_teaser_encoder(REPO_ROOT / cfg.checkpoint, device)
        arrays = extract_clip(work_dir, encoder, device, cfg.data.feature_set)
        cache_path = Path(tempfile.mkdtemp()) / "cache.npz"
        np.savez(cache_path, **arrays)
    clip = Clip(cache_path, occ_path, cfg.data.feature_set)
    pred = model.infer_clip(to_device(full_clip(clip), device), window=cfg.train.window,
                            stride=cfg.train.infer_stride)
    with np.load(cache_path) as cache:
        out = {k: cache[k] for k in ("frame_index", "valid", "pose_params", "cam", "shape_params")}
    out["expression"] = pred["expression"][0].cpu().numpy().astype(np.float32)
    out["jaw_pose"] = pred["jaw"][0].cpu().numpy().astype(np.float32)
    out["eyelid"] = pred["eyelid"][0].cpu().numpy().astype(np.float32)
    out["temporal_checkpoint"] = np.array(str(ckpt))
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, output)
    print(f"[infer_clip] {len(clip)} frames, {time.time() - started:.1f} s -> {output}")


if __name__ == "__main__":
    main()
