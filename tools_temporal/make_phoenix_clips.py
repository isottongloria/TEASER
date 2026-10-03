"""Clip directories for PHOENIX-2014-T clips whose frames we can read (development / test samples).

    python tools_temporal/make_phoenix_clips.py --split test --out <root> \
        [--occ <dir of *.occ.npz> --pick 40 --exclude eval100.txt]

Creates ``<root>/<clip>/frames/000000.png ...`` (symlinks to the release's
``imagesNNNN.png``, the naming of RGB2SMPLX work directories) and
``<root>/<clip>/video.json`` (``output_fps`` 25), so every tool here reads it
like an RGB2SMPLX work directory. Writes ``<root>/signers.tsv`` (clip ->
signer, from the corpus CSV).

With ``--pick N`` only N clips are linked: round-robin over signers, half of
them with at least one real occlusion episode (``c_mnc > 0.2`` in ``--occ``)
and half without, seeded; ``--exclude`` keeps given clips out (e.g. the 100
PHOENIX test clips kept for evaluation).
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PHOENIX = Path("/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX/phoenix_data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T")


def signers(split, phoenix=PHOENIX):
    path = phoenix / f"annotations/manual/PHOENIX-2014-T.{split}.corpus.csv"
    with open(path, newline="") as handle:
        return {row["name"]: row["speaker"] for row in csv.DictReader(handle, delimiter="|")}


def has_occlusion(occ_path, tau=0.2):
    with np.load(occ_path) as occ:
        return bool((np.nan_to_num(occ["c_mnc"], nan=0.0) > tau).any())


def pick_clips(names, signer_of, occ_dir, n, seed=0):
    rng = np.random.default_rng(seed)
    groups = defaultdict(lambda: {True: [], False: []})
    for name in names:
        occ = occ_dir / f"{name}.occ.npz"
        if occ.is_file():
            groups[signer_of[name]][has_occlusion(occ)].append(name)
    for g in groups.values():
        for k in g:
            rng.shuffle(g[k])
    picked, flag = [], True
    while len(picked) < n and any(g[True] or g[False] for g in groups.values()):
        for signer in sorted(groups):
            g = groups[signer]
            source = g[flag] or g[not flag]
            if source and len(picked) < n:
                picked.append(source.pop())
        flag = not flag
    return sorted(picked)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--occ", type=Path)
    parser.add_argument("--pick", type=int)
    parser.add_argument("--exclude", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    signer_of = signers(args.split)
    frames_root = PHOENIX / "features/fullFrame-210x260px" / args.split
    names = sorted(n for n in signer_of if (frames_root / n).is_dir())
    if args.exclude:
        excluded = {l.strip() for l in args.exclude.read_text().splitlines() if l.strip()}
        names = [n for n in names if n not in excluded]
    if args.pick:
        if not args.occ:
            raise SystemExit("--pick needs --occ")
        names = pick_clips(names, signer_of, args.occ, args.pick, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        clip = args.out / name
        frames = clip / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        for k, src in enumerate(sorted(frames_root.joinpath(name).glob("images*.png"))):
            link = frames / f"{k:06d}.png"
            if not link.is_symlink():
                link.symlink_to(src)
        (clip / "video.json").write_text(json.dumps({"output_fps": 25.0, "source": "PHOENIX-2014-T",
                                                     "split": args.split, "signer": signer_of[name]}))
    with open(args.out / "signers.tsv", "w") as handle:
        for name in names:
            handle.write(f"{name}\t{signer_of[name]}\n")
    counts = defaultdict(int)
    for name in names:
        counts[signer_of[name]] += 1
    print(f"[phoenix] {len(names)} clips -> {args.out}; per signer: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
