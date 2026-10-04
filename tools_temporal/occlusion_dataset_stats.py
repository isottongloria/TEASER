"""Occlusion statistics of the pilot dataset, real vs synthetic (TEMPORAL_README.md 4.2, 5.4).

    python tools_temporal/occlusion_dataset_stats.py --data data/temporal --corpus phoenix --workers 32 \
        --out data/temporal/report/stats_phoenix.json

Everything on the region and threshold of RGB2SMPLX's occlusion protocol: IoA
of the hands over the mouth/nose/chin hull, a frame occluded above 0.2. For
the synthetic hands the coverage is recomputed on that hull by replaying each
variant's plan (no image needed: the hand's placement depends only on the
face regions, so blank frames of the right size are used).

Reported, per corpus:
1. occluded-frame share
   - ``corpus``: every candidate clip measured (the corpus as it is),
     and the training signers' candidates only;
   - ``selected``: the selected clips (train / val / test), whole clips, and
     inside their clean segments (0 by construction);
   - ``param`` / ``replay``: the synthetic variants, whole clips (real +
     synthetic occlusion) and inside the clean segments (synthetic only);
   - ``training_windows``: the share of frames a training window shows
     occluded, for ``occ_aug_p`` 0.3 / 0.5 / 0.7 -- clean windows (none) and
     variant windows placed around an episode as data.WindowDataset does;
2. per-clip severity with the classes of the multilingual occlusion dataset
   (experiments/multilingual_occlusion_dataset/build_dataset.py): ``none`` = no
   occluded frame; ``severe`` = peak IoA >= 0.60 and longest episode >= 300 ms;
   ``moderate`` = the rest;
3. per-episode severity: duration (ms) and peak IoA histograms.
"""

import argparse
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TAU = 0.20
SEVERE_PEAK = 0.60
SEVERE_MS = 300.0
FPS = {"phoenix": 25.0, "csl_daily": 30.0, "how2sign": 30.0}
DUR_BINS_MS = [0, 80, 160, 300, 500, 800, 1e9]
PEAK_BINS = [0.2, 0.4, 0.6, 0.8, 0.95, 1.01]


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


def clip_summary(series, fps):
    """series: per-frame IoA over mouth/nose/chin. -> frames, occluded, class, episodes."""
    c = np.nan_to_num(np.asarray(series, np.float64), nan=0.0)
    occ = c > TAU
    episodes = [{"ms": (e - s) * 1000.0 / fps, "frames": e - s, "peak": float(c[s:e].max())} for s, e in runs(occ)]
    longest = max((ep["ms"] for ep in episodes), default=0.0)
    peak = float(c.max()) if len(c) else 0.0
    if not occ.any():
        cls = "none"
    elif peak >= SEVERE_PEAK and longest >= SEVERE_MS:
        cls = "severe"
    else:
        cls = "moderate"
    return {"frames": int(len(c)), "occluded": int(occ.sum()), "class": cls, "episodes": episodes}


def histogram(values, bins):
    counts = np.histogram(values, bins=bins)[0] if len(values) else np.zeros(len(bins) - 1, int)
    total = max(int(counts.sum()), 1)
    return [round(float(x) / total, 4) for x in counts]


def aggregate(summaries):
    frames = sum(s["frames"] for s in summaries)
    occluded = sum(s["occluded"] for s in summaries)
    episodes = [ep for s in summaries for ep in s["episodes"]]
    classes = Counter(s["class"] for s in summaries)
    n = max(len(summaries), 1)
    return {
        "clips": len(summaries), "frames": frames, "occluded_frames": occluded,
        "occluded_share": occluded / max(frames, 1),
        "classes": {k: classes.get(k, 0) for k in ("none", "moderate", "severe")},
        "class_share": {k: classes.get(k, 0) / n for k in ("none", "moderate", "severe")},
        "episodes": len(episodes),
        "episode_ms_median": float(np.median([e["ms"] for e in episodes])) if episodes else 0.0,
        "episode_peak_median": float(np.median([e["peak"] for e in episodes])) if episodes else 0.0,
        "episode_ms_hist": histogram([e["ms"] for e in episodes], DUR_BINS_MS),
        "episode_peak_hist": histogram([e["peak"] for e in episodes], PEAK_BINS),
    }


def variant_job(task):
    """Replay one variant's plan on blank frames; -> per-frame synthetic mnc coverage + summaries."""
    sys.path.insert(0, str(REPO_ROOT))
    import cv2
    from src.temporal import occlusion_aug as oa

    data, corpus, mode, path, segments, window = task
    root = Path(data) / corpus
    name = Path(path).name[:-4]
    clip = name.split(".v")[0]
    with np.load(path) as v:
        plan = json.loads(str(v["plan"]))
    with np.load(root / "occ" / f"{clip}.occ.npz") as occ:
        real = np.nan_to_num(occ["c_mnc"], nan=0.0)
    first = sorted((root / "clips" / clip / "frames").iterdir())[0]
    shape = cv2.imread(str(first)).shape
    n = len(real)

    class Blank:
        def __len__(self):
            return n

        def __getitem__(self, t):
            return np.zeros(shape, np.uint8)

    bank_split = task_bank_split(root, clip)
    hands = [l.strip() for l in (root / "lists" / f"hands_{bank_split}.txt").read_text().splitlines() if l.strip()]
    bank = oa.HandBank(str(Path(data) / "hand_bank" / corpus / bank_split), hands)
    occluder = oa.Occluder(Blank(), oa.FaceRegions.from_fit(root / "regions" / f"{clip}.regions.npz"), plan, bank)
    for t in range(n):
        if occluder.syn_mask[t]:
            occluder[t]
    syn = occluder.c_syn_mnc.astype(np.float64)
    fps = FPS[corpus]
    inside = np.zeros(n, bool)
    for s, e in segments:
        inside[s:e] = True
    combined = np.maximum(real, syn)
    # Training windows around each episode, as data.WindowDataset places them.
    windows = []
    for ep in plan:
        first_f = ep["start"] - ep.get("entry", 0)
        span = ep.get("entry", 0) + ep["length"] + ep.get("exit", 0)
        room = window - span
        for lead in (range(1, room) if room >= 2 else [0]):
            s = int(np.clip(first_f - lead, 0, max(0, n - window)))
            windows.append(float((syn[s:s + window] > TAU).mean()))
    return {
        "mode": mode, "variant": name,
        "whole": clip_summary(combined, fps),
        "inside": clip_summary(syn[inside], fps) | {"frames": int(inside.sum())},
        "synthetic_episodes": clip_summary(syn, fps)["episodes"],
        "window_occluded_mean": float(np.mean(windows)) if windows else 0.0,
    }


def task_bank_split(root, clip):
    train = {l.split("\t")[0] for l in (root / "lists" / "train.txt").read_text().splitlines() if l.strip()}
    return "train" if clip in train else "heldout"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data/temporal")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--max_variants", type=int, default=0, help="per mode, 0 = all (to cap the run time)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = args.data.resolve()
    root = data / args.corpus
    fps = FPS[args.corpus]
    lists = root / "lists"
    segments = json.loads((lists / "segments.json").read_text())
    read = lambda p: [l.split("\t") for l in p.read_text().splitlines() if l.strip()]
    signer_of = dict(r[:2] for r in read(lists / "candidates.tsv"))
    train_signers = {r[1] for r in read(lists / "train.txt")}

    def real_summary(clip):
        with np.load(root / "occ" / f"{clip}.occ.npz") as occ:
            return clip_summary(occ["c_mnc"], fps)

    candidates = [c for c in signer_of if (root / "occ" / f"{c}.occ.npz").is_file()]
    real = {c: real_summary(c) for c in candidates}
    out = {"corpus": args.corpus, "fps": fps, "tau": TAU,
           "dur_bins_ms": DUR_BINS_MS[:-1], "peak_bins": PEAK_BINS[:-1],
           "corpus_all": aggregate(list(real.values())),
           "corpus_train_signers": aggregate([real[c] for c in candidates if signer_of[c] in train_signers]),
           "selected": {}}
    for split in ("train", "val", "test"):
        clips = [r[0] for r in read(lists / f"{split}.txt")]
        out["selected"][split] = aggregate([real[c] for c in clips])

    tasks = []
    for mode, folder in (("param", "variants"), ("replay", "variants_replay")):
        paths = sorted((root / folder).glob("*.npz"))
        if args.max_variants:
            paths = paths[:args.max_variants]
        for p in paths:
            clip = p.name.split(".v")[0]
            tasks.append((str(data), args.corpus, mode, str(p), segments.get(clip, []), args.window))
    with Pool(args.workers) as pool:
        results = pool.map(variant_job, tasks, chunksize=2)
    for mode in ("param", "replay"):
        rs = [r for r in results if r["mode"] == mode]
        if not rs:
            continue
        syn_eps = [{"episodes": r["synthetic_episodes"], "frames": 0, "occluded": 0, "class": "none"} for r in rs]
        window_share = float(np.mean([r["window_occluded_mean"] for r in rs]))
        out[mode] = {
            "whole": aggregate([r["whole"] for r in rs]),
            "inside_segments": aggregate([r["inside"] for r in rs]),
            "synthetic_episodes": aggregate(syn_eps) | {"clips": len(rs)},
            "occluded_share_in_variant_windows": window_share,
            "training_windows": {str(p): p * window_share for p in (0.3, 0.5, 0.7)},
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    show = lambda a: f"{100 * a['occluded_share']:.1f}% frames occluded, classes {a['classes']}"
    print(f"[stats] {args.corpus}: corpus {show(out['corpus_all'])}")
    for mode in ("param", "replay"):
        if mode in out:
            print(f"[stats] {args.corpus} {mode}: whole {show(out[mode]['whole'])}; inside segments "
                  f"{100 * out[mode]['inside_segments']['occluded_share']:.1f}%; training windows at p=0.5 "
                  f"{100 * out[mode]['training_windows']['0.5']:.1f}%")


if __name__ == "__main__":
    main()
