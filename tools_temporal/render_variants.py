"""Write the synthetic-occlusion variants out as frame folders, exactly as training and evaluation saw them.

    python tools_temporal/render_variants.py --corpus phoenix [--shard 0/8] [--root data/temporal] \
        [--out data/temporal/occluded_videos]

make_synthetic_variants.py pasted the hands in memory and kept only TEASER's
features. This re-renders the same frames from what each variant saved: its
``plan`` (hand, flip, per-frame position / size / angle, entry and exit), the
clip's clean frames, its FLAME regions and the hand bank it was drawn from
(train hands for training clips, held-out hands for val / test) -- the
Occluder has no randomness, so the pixels are the ones TEASER was given.

Output, per variant: ``<out>/<split>/<corpus>/<clip>.<kind><k>/``
  ``frames/``        frames with a hand: PNG (lossless, the exact pixels);
                     the others: links to the clean frames (identical)
  ``occlusion.npz``  c_syn (T, 2: lips, eyes), syn_mask, core_mask, plan, source
                     variant cache, clean frames dir
  ``video.json``     link to the clip's
``<kind>`` is ``param`` (variants/) or ``replay`` (variants_replay/); ``k`` the
variant's index there. Training reads variants_mix/, which links the param
ones as v0-v2 and the replay ones as v3-v5.

Check on every variant: the coverage the Occluder measures while pasting must
equal the saved ``c_syn``. verify_rendered_variants.py re-runs TEASER on a
sample and compares the features.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
KINDS = (("param", "variants"), ("replay", "variants_replay"))


def read_list(path):
    return [line.split()[0] for line in Path(path).read_text().splitlines() if line.strip()]


def jobs(root, corpus):
    """(split, clip, kind, variant path), in a fixed order."""
    out = []
    for split in ("train", "val", "test"):
        for clip in read_list(root / corpus / "lists" / f"{split}.txt"):
            for kind, folder in KINDS:
                for path in sorted((root / corpus / folder).glob(f"{clip}.v*.npz")):
                    out.append((split, clip, kind, path))
    return out


def render(root, out_root, corpus, split, clip, kind, variant, bank, oa, vc):
    import cv2

    k = variant.name[len(clip) + 2:-4]
    out = out_root / split / corpus / f"{clip}.{kind}{k}"
    if (out / "occlusion.npz").is_file():
        return None
    with np.load(variant) as v:
        plan = json.loads(str(v["plan"]))
        c_syn, syn_mask, core_mask = v["c_syn"], v["syn_mask"].astype(bool), v["core_mask"].astype(bool)
    clip_dir = root / corpus / "clips" / clip
    paths = vc.list_frames(clip_dir / "frames")
    frames = vc.Frames(paths, 1 << 30)
    regions = oa.FaceRegions.from_fit(root / corpus / "regions" / f"{clip}.regions.npz")
    occluder = oa.Occluder(frames, regions, plan, bank)
    if len(occluder) != len(syn_mask) or not np.array_equal(occluder.syn_mask, syn_mask):
        raise ValueError(f"{variant}: frames or plan do not match the saved masks")
    tmp = out.with_name(f".{out.name}.tmp")
    if tmp.exists():
        for p in (tmp / "frames").glob("*"):
            p.unlink()
    (tmp / "frames").mkdir(parents=True, exist_ok=True)
    for t, path in enumerate(paths):
        target = tmp / "frames" / f"{Path(path).stem}.png" if syn_mask[t] else tmp / "frames" / Path(path).name
        if syn_mask[t]:
            if not cv2.imwrite(str(target), occluder[t]):
                raise OSError(f"could not write {target}")
        else:
            os.symlink(Path(path).resolve(), target)
    diff = float(np.abs(occluder.c_syn - c_syn).max()) if len(c_syn) else 0.0
    np.savez(tmp / "occlusion.npz", c_syn=c_syn, syn_mask=syn_mask, core_mask=core_mask,
             plan=np.array(json.dumps(plan)), source=np.array(str(variant.resolve())),
             clean_frames=np.array(str((clip_dir / "frames").resolve())), c_syn_rerendered_maxdiff=np.float32(diff))
    if (clip_dir / "video.json").exists():
        os.symlink((clip_dir / "video.json").resolve(), tmp / "video.json")
    os.replace(tmp, out)
    return out, int(syn_mask.sum()), len(paths), diff


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data/temporal")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/temporal/occluded_videos")
    parser.add_argument("--shard", default="0/1", help="i/n: every n-th variant from the i-th")
    args = parser.parse_args()
    root, out_root = args.root.resolve(), args.out.resolve()
    shard, n_shards = (int(x) for x in args.shard.split("/"))
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    from src.temporal import occlusion_aug as oa
    from src.temporal import video_crops as vc

    banks = {s: oa.HandBank(root / "hand_bank" / args.corpus / s,
                            read_list(root / args.corpus / "lists" / f"hands_{s}.txt"))
             for s in ("train", "heldout")}
    todo = jobs(root, args.corpus)[shard::n_shards]
    started, done, worst = time.time(), 0, 0.0
    for split, clip, kind, variant in todo:
        result = render(root, out_root, args.corpus, split, clip, kind, variant,
                        banks["train" if split == "train" else "heldout"], oa, vc)
        if result is None:
            continue
        out, pasted, n, diff = result
        done += 1
        worst = max(worst, diff)
        flag = "" if diff < 1e-6 else "  <-- c_syn MISMATCH"
        print(f"[render] {split}/{args.corpus}/{out.name}: {pasted}/{n} frames pasted, "
              f"c_syn max diff {diff:.2e}{flag}", flush=True)
    print(f"[render] {args.corpus} shard {args.shard}: {done}/{len(todo)} variants in "
          f"{time.time() - started:.0f} s, worst c_syn diff {worst:.2e}")


if __name__ == "__main__":
    main()
