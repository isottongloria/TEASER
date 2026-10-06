"""Spot-check that rendered variants (render_variants.py) are the frames TEASER was given in training / evaluation.

    python tools_temporal/verify_rendered_variants.py --fraction 0.1 --shard 0/8 --out check_0.json

Draws a seeded sample of the rendered variants (the same fraction of every
split / corpus / kind), re-runs the feature extraction on the rendered folder
and compares with the variant cache it came from: TEASER features (expr, pose),
expression / jaw / eyelid and landmarks. As the noise floor of a re-run on the
same frames, ``--clean`` re-extracts that many clean clips and compares them
with their own caches.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
KEYS = ("feat_expr", "feat_pose", "expression", "jaw_pose", "eyelid", "landmarks")


def sample(out_root, fraction, seed):
    groups = defaultdict(list)
    for d in sorted(out_root.glob("*/*/*")):
        if (d / "occlusion.npz").is_file():
            kind = "replay" if ".replay" in d.name else "param"
            groups[(d.parent.parent.name, d.parent.name, kind)].append(d)
    rng = np.random.default_rng(seed)
    picked = []
    for key in sorted(groups):
        dirs = groups[key]
        n = max(1, int(round(fraction * len(dirs))))
        picked += [dirs[i] for i in sorted(rng.choice(len(dirs), n, replace=False))]
    return picked, {"/".join(k): len(v) for k, v in groups.items()}


def compare(arrays, ref_path, rows=None):
    out = {}
    with np.load(ref_path) as ref:
        for key in KEYS:
            a, b = np.asarray(arrays[key], np.float64), np.asarray(ref[key], np.float64)
            if rows is not None:
                a, b = a[rows], b[rows]
            d = np.abs(a - b)
            out[key] = {"max": float(d.max()) if d.size else 0.0, "mean": float(d.mean()) if d.size else 0.0,
                        "ref_scale": float(np.abs(b).mean()) if b.size else 0.0}
        out["valid_equal"] = bool(np.array_equal(arrays["valid"], ref["valid"]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_root", type=Path, default=REPO_ROOT / "data/temporal/occluded_videos")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data/temporal")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--clean", type=int, default=0, help="clean clips to re-extract (noise floor)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "pretrained_models/TEASER.pt")
    args = parser.parse_args()
    out_root, root = args.out_root.resolve(), args.root.resolve()
    shard, n_shards = (int(x) for x in args.shard.split("/"))
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    from src.temporal.split_encoder import load_teaser_encoder
    from tools_temporal.extract_features import extract_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = load_teaser_encoder(args.checkpoint.resolve(), device)
    picked, population = sample(out_root, args.fraction, args.seed)
    results = {"population": population, "sampled": len(picked), "variants": {}, "clean": {}}
    for d in picked[shard::n_shards]:
        with np.load(d / "occlusion.npz") as occ:
            source, syn = Path(str(occ["source"])), occ["syn_mask"].astype(bool)
        arrays = extract_clip(d, encoder, device, "expr+pose", progress=False)
        r = {"all": compare(arrays, source), "pasted": compare(arrays, source, syn),
             "frames": int(len(syn)), "pasted_frames": int(syn.sum())}
        results["variants"][str(d.relative_to(out_root))] = r
        print(f"[verify] {d.relative_to(out_root)}: feat_expr max {r['all']['feat_expr']['max']:.2e}, "
              f"expression max {r['all']['expression']['max']:.2e}", flush=True)
    clean = sorted({(d.parent.name, d.name.split(".")[0]) for d in picked})[shard::n_shards][:args.clean]
    for corpus, clip in clean:
        arrays = extract_clip(root / corpus / "clips" / clip, encoder, device, "expr+pose", progress=False)
        r = compare(arrays, root / corpus / "cache" / f"{clip}.npz")
        results["clean"][f"{corpus}/{clip}"] = r
        print(f"[verify] clean {corpus}/{clip}: feat_expr max {r['feat_expr']['max']:.2e}", flush=True)
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
