"""Frame counts of the temporal training data, per split and corpus: clean and occluded, real and synthetic.

    python tools_temporal/training_set_stats.py --root data/temporal/all --variants train=variants_mix val=variants_replay test=variants_replay --out stats.json

Real occlusion = mouth/nose/chin IoA > 0.2 (c_mnc, as everywhere else); "near"
= within 2 frames of it; "clean segments" = the spans training's clean windows
are drawn from (segments.json). Synthetic = per variant, frames with a pasted
hand (entry/core/exit) and those whose covered region is above 0.2.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def runs(mask):
    """Lengths of the True runs of a bool array."""
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return edges[1::2] - edges[::2]


def dilate(mask, k):
    out = mask.copy()
    for s in range(1, k + 1):
        out[s:] |= mask[:-s]
        out[:-s] |= mask[s:]
    return out


def corpus_of(name, members):
    for corpus, names in members.items():
        if name in names:
            return corpus
    if name.startswith("h2s_"):
        return "how2sign"
    return "csl_daily" if name.startswith("S0") else "phoenix"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/temporal/all"))
    parser.add_argument("--variants", nargs="+", default=["train=variants_mix", "val=variants_replay",
                                                           "test=variants_replay"],
                        help="split=variant_dir: the variants each split is used with")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    variant_dirs = dict(spec.split("=", 1) for spec in args.variants)
    lists = args.root / "lists"
    read = lambda p: [l.split()[0] for l in p.read_text().splitlines() if l.strip()]
    segments = json.loads((lists / "segments.json").read_text())
    corpora = ("phoenix", "csl_daily", "how2sign")
    members = {c: set(read(lists / f"train_{c}.txt")) | set(read(lists / f"val_{c}.txt")) for c in corpora}
    stats = defaultdict(lambda: defaultdict(float))
    for split in ("train", "val", "test"):
        for name in read(lists / f"{split}.txt"):
            key = f"{split}/{corpus_of(name, members)}"
            s = stats[key]
            with np.load(args.root / "cache" / f"{name}.npz") as cache:
                valid = cache["valid"].astype(bool)
                fps = float(cache["fps"]) if "fps" in cache.files else 25.0
            occ_path = args.root / "occ" / f"{name}.occ.npz"
            if occ_path.is_file():
                with np.load(occ_path) as occ:
                    c = np.nan_to_num(occ["c_mnc"])
            else:
                c = np.zeros(len(valid))
            occluded = (c > 0.2) & valid
            near = dilate(occluded, 2) & valid & ~occluded
            s["clips"] += 1
            s["frames"] += len(valid)
            s["seconds"] += len(valid) / fps
            s["valid"] += valid.sum()
            s["real_occluded"] += occluded.sum()
            s["real_near"] += near.sum()
            s["real_clean"] += (valid & ~occluded & ~near).sum()
            s["real_episodes"] += len(runs(occluded))
            s["clean_segment_frames"] += sum(e - b for b, e in segments.get(name, [[0, len(valid)]]))
            for path in sorted((args.root / variant_dirs[split]).glob(f"{name}.v*.npz")):
                with np.load(path) as v:
                    syn = v["syn_mask"].astype(bool) & valid
                    syn_occ = syn & (v["c_syn"].max(1) > 0.2)
                    plan = json.loads(str(v["plan"]))
                s["variants"] += 1
                s["variant_frames"] += valid.sum()
                s["syn_hand_frames"] += syn.sum()
                s["syn_occluded"] += syn_occ.sum()
                s["syn_episodes"] += len(plan)
                s["syn_core_len_sum"] += sum(ep["length"] for ep in plan)
                for ep in plan:
                    s[f"syn_type_{ep.get('type', '?')}"] += 1
                    s[f"syn_region_{ep.get('region', '?')}"] += 1
    out = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in sorted(stats.items())}
    for split in ("train", "val", "test"):
        total = defaultdict(float)
        for k, v in out.items():
            if k.startswith(split + "/"):
                for kk, vv in v.items():
                    total[kk] += vv
        out[f"{split}/ALL"] = dict(total)
    text = json.dumps(out, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
