"""Keep the clips with little real occlusion and cut them into fully clean segments (TEMPORAL_README.md 4.2).

    python tools_temporal/select_clean_clips.py --occ <dir of *.occ.npz> --signers signers.tsv \
        --split train=Signer01,Signer03 --split val=Signer07 --split test=Signer04,Signer08 \
        --pick train=100 --pick val=20 --pick test=30 --max_per_signer 30 --out <dir>

Training uses clean video only: occlusions are all synthetic, pasted by us,
so every occluded frame has a true unoccluded target. Real occlusion is
measured (compute_real_occlusion.py, on the fits) only to throw it away here
and for the statistics / the real-world test.

Per clip, from ``c_mnc`` (hands over mouth/nose/chin, IoA):
- episodes = runs of ``c_mnc > tau`` (0.2);
- dirty frames = episode frames dilated by ``--margin`` (2: the frames next to
  an occlusion are already corrupted) plus frames with ``c_mnc >= clean_max``
  (0.05) or a NaN region;
- clean segments = runs of the remaining frames at least ``--min_segment``
  (24) long.
A clip is eligible when its occluded fraction is <= ``--max_occ_frac`` (5 %),
no episode is longer than ``--max_episode`` (15), and its clean segments hold
at least ``--min_clean`` (48) frames. From the eligible clips, ``--pick`` takes
N per split, at most ``--max_per_signer`` per signer, round-robin over
signers, seeded.

Writes ``<out>/<split>.txt`` (clip<TAB>signer), ``<out>/segments.json``
(clip -> [[start, end_exclusive], ...]) and ``<out>/selection_report.json``.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def runs(flags):
    out, start = [], None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(flags)))
    return out


def analyse(c_mnc, tau=0.2, clean_max=0.05, margin=2, min_segment=24):
    nan = np.isnan(c_mnc)
    c = np.nan_to_num(c_mnc, nan=1.0)
    occluded = c > tau
    episodes = runs(occluded)
    dirty = occluded.copy()
    for s, e in episodes:
        dirty[max(0, s - margin):min(len(c), e + margin)] = True
    dirty |= (c >= clean_max) | nan
    segments = [(s, e) for s, e in runs(~dirty) if e - s >= min_segment]
    return {
        "frames": len(c), "occ_frac": float(occluded.mean()) if len(c) else 0.0,
        "episodes": len(episodes), "longest_episode": max((e - s for s, e in episodes), default=0),
        "segments": segments, "clean_frames": int(sum(e - s for s, e in segments)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--occ", type=Path, required=True)
    parser.add_argument("--signers", type=Path, required=True, help="clip<TAB>signer for every candidate")
    parser.add_argument("--split", action="append", default=[], help="NAME=signer,signer,...")
    parser.add_argument("--pick", action="append", default=[], help="NAME=N clips")
    parser.add_argument("--max_per_signer", type=int, default=30)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--clean_max", type=float, default=0.05)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--min_segment", type=int, default=24)
    parser.add_argument("--min_clean", type=int, default=48)
    parser.add_argument("--max_occ_frac", type=float, default=0.05)
    parser.add_argument("--max_episode", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    signer_of = dict(line.split("\t")[:2] for line in args.signers.read_text().splitlines() if line.strip())
    split_of_signer = {}
    for spec in args.split:
        name, signers = spec.split("=", 1)
        for signer in signers.split(","):
            split_of_signer[signer] = name
    picks = {k: int(v) for k, v in (p.split("=", 1) for p in args.pick)}

    analysis, reasons = {}, Counter()
    eligible = defaultdict(lambda: defaultdict(list))
    for clip, signer in sorted(signer_of.items()):
        path = args.occ / f"{clip}.occ.npz"
        if not path.is_file():
            reasons["no occlusion file (no fit?)"] += 1
            continue
        with np.load(path) as occ:
            a = analyse(occ["c_mnc"].astype(np.float64), args.tau, args.clean_max, args.margin, args.min_segment)
        analysis[clip] = a
        if a["occ_frac"] > args.max_occ_frac:
            reasons["too much real occlusion"] += 1
        elif a["longest_episode"] > args.max_episode:
            reasons["an episode too long"] += 1
        elif a["clean_frames"] < args.min_clean:
            reasons["not enough clean frames"] += 1
        elif signer not in split_of_signer:
            reasons["signer in no split"] += 1
        else:
            reasons["eligible"] += 1
            eligible[split_of_signer[signer]][signer].append(clip)

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"criteria": {k: getattr(args, k) for k in ("tau", "clean_max", "margin", "min_segment", "min_clean",
                                                          "max_occ_frac", "max_episode", "max_per_signer")},
              "candidates": len(signer_of), "outcome": dict(reasons), "splits": {}}
    segments = {}
    for split, n in picks.items():
        pools = {s: list(rng.permutation(c)) for s, c in sorted(eligible[split].items())}
        chosen, taken = [], Counter()
        while len(chosen) < n and any(pools.values()):
            for signer in sorted(pools):
                if pools[signer] and taken[signer] < args.max_per_signer and len(chosen) < n:
                    chosen.append((pools[signer].pop(), signer))
                    taken[signer] += 1
            if all(not p or taken[s] >= args.max_per_signer for s, p in pools.items()):
                break
        chosen.sort()
        (args.out / f"{split}.txt").write_text("".join(f"{c}\t{s}\n" for c, s in chosen))
        for clip, _ in chosen:
            segments[clip] = analysis[clip]["segments"]
        report["splits"][split] = {
            "picked": len(chosen), "asked": n, "per_signer": dict(sorted(taken.items())),
            "eligible_per_signer": {s: len(c) for s, c in sorted(eligible[split].items())},
            "clean_frames": int(sum(analysis[c]["clean_frames"] for c, _ in chosen)),
            "frames": int(sum(analysis[c]["frames"] for c, _ in chosen)),
        }
    (args.out / "segments.json").write_text(json.dumps(segments))
    (args.out / "selection_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"candidates": report["candidates"], "outcome": report["outcome"],
                      "splits": {k: {kk: v[kk] for kk in ("picked", "asked", "per_signer", "clean_frames", "frames")}
                                 for k, v in report["splits"].items()}}, indent=1))


if __name__ == "__main__":
    main()
