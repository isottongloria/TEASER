"""The feature cache's TEASER outputs equal RGB2SMPLX's teaser.npz on a real clip (GPU).

    python tests/temporal/check_cache_vs_stage.py <work_dir> [--rgb2smplx /path/to/RGB2SMPLX] [--out-dir DIR]

Runs, each in its own fresh process (both import top-level ``src``/``utils``
packages, from different roots, so they cannot share one): RGB2SMPLX's
``rgb2smplx.stages.teaser`` with the production TEASER checkout, batch 1;
then ``tools_temporal/extract_features.py`` from this tree, batch 1. Compares
every key teaser.npz has, exactly. Exit code 0 = identical.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
KEYS = ("frame_index", "valid", "pose_params", "jaw_pose", "expression", "eyelid", "cam", "shape_params")


def _env(pythonpath):
    env = dict(os.environ, PYTHONPATH=str(pythonpath), PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--rgb2smplx", type=Path, default=Path("/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX"))
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "pretrained_models/TEASER.pt")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="cache_vs_stage_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_npz, cache_npz = out_dir / "teaser_stage.npz", out_dir / "cache.npz"
    checkpoint = args.checkpoint.resolve()

    subprocess.run([sys.executable, "-m", "rgb2smplx.stages.teaser", str(args.work_dir), str(stage_npz),
                    "--checkpoint", str(checkpoint), "--batch-size", "1"],
                   cwd=args.rgb2smplx, env=_env(args.rgb2smplx),
                   check=True)
    subprocess.run([sys.executable, str(REPO_ROOT / "tools_temporal/extract_features.py"),
                    str(args.work_dir), str(cache_npz), "--checkpoint", str(checkpoint),
                    "--temporal_feats", "all", "--batch-size", "1"],
                   cwd=REPO_ROOT, env=_env(REPO_ROOT), check=True)

    stage, cache = np.load(stage_npz), np.load(cache_npz)
    ok = True
    for key in KEYS:
        a, b = stage[key], cache[key]
        same = a.shape == b.shape and np.array_equal(a, b)
        diff = float(np.abs(a.astype(np.float64) - b).max()) if a.shape == b.shape and a.size else float("nan")
        print(f"{key:14s} {str(a.shape):14s} {'identical' if same else f'DIFFERENT max|d|={diff:.3g}'}")
        ok &= same
    print("feature keys:", {k: cache[k].shape for k in cache.files if k.startswith("feat_")})
    print("RESULT:", "IDENTICAL" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
