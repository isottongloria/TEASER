"""Synthetic-occlusion variants of clips (TEMPORAL_README.md 5.4, 5.6).

    python tools_temporal/make_synthetic_variants.py --clips_root <dir of clip dirs> --cache <clean caches> \
        --out <variant caches> --hands <PNG folder | procedural> [--hand_list names.txt] \
        [--stats occlusion_stats.json] --variants 3 [--names list.txt] [--seed 0]

For each clip and each variant k: sample an occlusion plan (seeded by clip
name, k and ``--seed``) from the real statistics, paste the hands on the
clip's frames **before** face detection and cropping, and run the whole
TEASER feature extraction on the pasted frames. Writes
``<out>/<clip>.v<k>.npz``: the same keys as a clean cache (``expression`` etc.
there are TEASER *on the occluded frames*, i.e. the T0 baseline under
occlusion), plus:

``c_syn`` (T, 2)   synthetic coverage [mouth, eyes] from the pasted alpha
``syn_mask`` (T,)  frames with a pasted hand
``plan``           the episodes, JSON (inside the clip's clean segments with ``--segments``)
``clean_cache``    the clean cache's file name: the training target (TEASER on
                   the unoccluded frames) and the clean landmarks come from it.

The face regions where the hand goes come from the clean cache's landmarks,
so the plan does not depend on what the detector does under the hand.
"""

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips_root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="clean caches (<clip>.npz)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hands", required=True, help="folder of RGBA PNGs, or 'procedural' (smoke tests only)")
    parser.add_argument("--hand_list", type=Path, help="names of the hand PNGs to use (train or test identities)")
    parser.add_argument("--stats", type=Path, help="occlusion_stats.py JSON (default: geometric, mean 4.3 frames)")
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--segments", type=Path, help="segments.json of select_clean_clips.py: paste only "
                                                      "inside each clip's clean segments")
    parser.add_argument("--names", type=Path, help="clip names (default: every clean cache)")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "pretrained_models/TEASER.pt")
    parser.add_argument("--temporal_feats", choices=("expr", "expr+pose"), default="expr+pose")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    clips_root, cache_dir, out_dir = args.clips_root.resolve(), args.cache.resolve(), args.out.resolve()
    hands = args.hands if args.hands == "procedural" else str(Path(args.hands).resolve())
    hand_list = args.hand_list.resolve() if args.hand_list else None
    stats_path = args.stats.resolve() if args.stats else None
    names_path = args.names.resolve() if args.names else None
    segments = json.loads(args.segments.read_text()) if args.segments else {}
    checkpoint = args.checkpoint.resolve()
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    from src.temporal import occlusion_aug as oa
    from src.temporal import video_crops as vc
    from src.temporal.split_encoder import load_teaser_encoder
    from tools_temporal.extract_features import extract_clip

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = ([l.split()[0] for l in names_path.read_text().splitlines() if l.strip()] if names_path
             else sorted(p.name[:-4] for p in cache_dir.glob("*.npz") if p.name.count(".") == 1))
    bank = oa.HandBank(hands, [l.strip() for l in hand_list.read_text().splitlines() if l.strip()]
                       if hand_list else None)
    stats = oa.OcclusionStats(stats_path)
    encoder = load_teaser_encoder(checkpoint, device)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        with np.load(cache_dir / f"{name}.npz") as clean:
            landmarks = clean["landmarks"].astype(np.float64)
            fps = float(clean["fps"])
        frames = vc.Frames(vc.list_frames(clips_root / name / "frames"), 3 << 30)
        for k in range(args.variants):
            target = out_dir / f"{name}.v{k}.npz"
            if target.is_file():
                continue
            started = time.time()
            rng = np.random.default_rng([args.seed, k, zlib.crc32(name.encode())])
            plan = oa.sample_plan(len(frames), stats, bank.names, rng, segments=segments.get(name))
            occluder = oa.Occluder(frames, landmarks, plan, bank)
            arrays = extract_clip(clips_root / name, encoder, device, args.temporal_feats,
                                  frames=occluder, progress=False)
            arrays.update(c_syn=occluder.c_syn, syn_mask=occluder.syn_mask,
                          plan=np.array(json.dumps(plan)), clean_cache=np.array(f"{name}.npz"),
                          fps=np.float32(fps))
            tmp = target.with_name(f".{target.name}.tmp.npz")
            np.savez(tmp, **arrays)
            os.replace(tmp, target)
            covered = occluder.c_syn[occluder.syn_mask]
            print(f"[variants] {name} v{k}: {len(plan)} episodes, {int(occluder.syn_mask.sum())}/{len(frames)} "
                  f"frames, mouth coverage median {np.median(covered[:, 0]) if len(covered) else 0:.2f}, "
                  f"{time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
