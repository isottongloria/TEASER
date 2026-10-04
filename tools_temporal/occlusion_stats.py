"""Statistics of real hand-face occlusion, for sizing synthetic occlusions (TEMPORAL_README.md 4.3, 5.2).

    python tools_temporal/occlusion_stats.py --occ <dir of *.occ.npz> --out stats.json [--clips list.txt]

Reads the per-clip ``*.occ.npz`` of compute_real_occlusion.py. An episode is a
run of frames with ``c_mnc > tau`` (0.2, the measure of RGB2SMPLX's occlusion
protocol, so durations are comparable with its segment_durations.json).
Writes, per corpus run:

- ``durations``: histogram (frames -> count), mean, median, percentiles;
- ``episodes_per_second`` and ``occluded_fraction`` of all frames;
- ``coverage``: per episode, the peak IoA over mouth/nose/chin and over the
  lips, the fraction of mouth episodes that also reach the eyes (a hand over
  the mouth often touches the eye region too), and ``eyes_only_rate``: eye
  episodes with no mouth occlusion at all, over all episodes -- the rate at
  which a synthetic hand should go to the eyes instead of the mouth;
- ``gap_frames``: distance between consecutive episodes in a clip.

The synthetic-occlusion sampler (src/temporal/occlusion_aug.py) draws from
``durations.histogram`` and ``coverage`` directly.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def segments(flags):
    out, start = [], None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def _percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {}
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            **{f"p{p}": float(np.percentile(values, p)) for p in (75, 90, 95, 99)},
            "max": float(values.max())}


def clip_episodes(occ, tau):
    c_mnc = np.nan_to_num(occ["c_mnc"], nan=0.0)
    c_mouth = np.nan_to_num(occ["c_mouth"], nan=0.0)
    c_eyes = np.nan_to_num(occ["c_eyes"], nan=0.0)
    episodes = []
    for start, end in segments(c_mnc > tau):
        sl = slice(start, end + 1)
        episodes.append({"start": start, "end": end, "frames": end - start + 1,
                         "peak_mnc": float(c_mnc[sl].max()), "peak_mouth": float(c_mouth[sl].max()),
                         "peak_eyes": float(c_eyes[sl].max())})
    return episodes, len(c_mnc)


def summarise(paths, tau=0.2, fps=25.0):
    durations, peaks_mnc, peaks_mouth, eyes_hit, gaps = [], [], [], [], []
    total_frames = occluded_frames = eyes_only = 0
    for path in paths:
        with np.load(path) as occ:
            episodes, n = clip_episodes(occ, tau)
            mouth_flags = np.nan_to_num(occ["c_mnc"], nan=0.0) > tau
            for start, end in segments(np.nan_to_num(occ["c_eyes"], nan=0.0) > tau):
                if not mouth_flags[start:end + 1].any():
                    eyes_only += 1
        total_frames += n
        for k, ep in enumerate(episodes):
            durations.append(ep["frames"])
            occluded_frames += ep["frames"]
            peaks_mnc.append(ep["peak_mnc"])
            peaks_mouth.append(ep["peak_mouth"])
            eyes_hit.append(ep["peak_eyes"] > tau)
            if k:
                gaps.append(ep["start"] - episodes[k - 1]["end"] - 1)
    histogram = dict(sorted(Counter(durations).items()))
    return {
        "tau": tau, "fps": fps, "clips": len(paths), "frames": total_frames,
        "episodes": len(durations),
        "episodes_per_second": len(durations) / max(total_frames / fps, 1e-9),
        "occluded_fraction": occluded_frames / max(total_frames, 1),
        "durations": {"histogram": {str(k): v for k, v in histogram.items()}, **_percentiles(durations)},
        "coverage": {"peak_mnc": _percentiles(peaks_mnc), "peak_mouth": _percentiles(peaks_mouth),
                     "peak_mouth_values": [round(v, 4) for v in peaks_mouth],
                     "fraction_reaching_eyes": float(np.mean(eyes_hit)) if eyes_hit else 0.0,
                     "fraction_total_mouth": float(np.mean(np.array(peaks_mouth) > 0.9)) if peaks_mouth else 0.0,
                     "eyes_only_episodes": eyes_only,
                     "eyes_only_rate": eyes_only / max(len(durations) + eyes_only, 1)},
        "gap_frames": _percentiles(gaps),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--occ", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clips", type=Path, help="restrict to these clip names (e.g. the training signers')")
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()
    if args.clips:
        names = [l.strip() for l in args.clips.read_text().splitlines() if l.strip()]
        paths = [args.occ / f"{n}.occ.npz" for n in names if (args.occ / f"{n}.occ.npz").is_file()]
    else:
        paths = sorted(args.occ.glob("*.occ.npz"))
    stats = summarise(paths, args.tau, args.fps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=1))
    d = stats["durations"]
    print(f"[stats] {stats['clips']} clips, {stats['episodes']} episodes, "
          f"{100 * stats['occluded_fraction']:.1f}% frames occluded; duration mean {d.get('mean', 0):.2f} "
          f"median {d.get('median', 0):.0f} p95 {d.get('p95', 0):.0f} max {d.get('max', 0):.0f}; "
          f"peak lips IoA median {stats['coverage']['peak_mouth'].get('median', 0):.2f} -> {args.out}")


if __name__ == "__main__":
    main()
