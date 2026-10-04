"""Summarise a training run's log.jsonl: learning curves, overfitting check, baselines, per corpus.

    python tools_temporal/summarize_run.py runs/T7_replay_long [--json out.json]

Prints, per validation step, the validation score and the same score on the
training subset (train.eval_train_clips), and the key metrics:
synthetic occluded / near-occlusion mouth error (recovery), real clean face
error (fidelity to TEASER) and real clean jerk (stability), next to TEASER
per-frame and SG9 from step 0. A training score that keeps falling while
validation rises is overfitting.
"""

import argparse
import json
from pathlib import Path

KEYS = (("synthetic", "occluded", "err_mouth", "occl. mouth err"),
        ("synthetic", "near", "err_mouth", "near mouth err"),
        ("synthetic", "clean", "err_face", "syn clean face err"),
        ("real", "clean", "err_face", "real clean face err"),
        ("real", "clean", "jerk", "real clean jerk"),
        ("real", "near", "jerk", "real near jerk"))


def get(m, kind, group, key):
    return m.get(kind, {}).get(group, {}).get(key, float("nan"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    entries = [json.loads(l) for l in (args.run / "log.jsonl").read_text().splitlines() if l.strip()]
    vals = [e for e in entries if "val" in e]
    trains = [e for e in entries if "total" in e]
    first = vals[0]["val"]

    print(f"{'':22s}" + "".join(f"{k[3]:>20s}" for k in KEYS))
    for name, m in (("TEASER (T0)", first.get("teaser", {})), ("SG9 (T1)", first.get("sg9", {}))):
        print(f"{name:22s}" + "".join(f"{get(m, *k[:3]):20.3f}" for k in KEYS))
    print()
    print(f"{'step':>6s} {'val score':>10s} {'train score':>12s}" + "".join(f"{k[3]:>20s}" for k in KEYS))
    rows = []
    for e in vals:
        m = e["val"]
        row = {"step": e["step"], "score": e.get("score"), "train_score": e.get("train_score"),
               **{k[3]: get(m, *k[:3]) for k in KEYS}}
        if "train_eval" in e:
            row.update({"train " + k[3]: get(e["train_eval"], *k[:3]) for k in KEYS})
        rows.append(row)
        sc = f"{e['score']:.3f}" if e.get("score") is not None else "-"
        ts = f"{e['train_score']:.3f}" if e.get("train_score") is not None else "-"
        print(f"{e['step']:6d} {sc:>10s} {ts:>12s}" + "".join(f"{row[k[3]]:20.3f}" for k in KEYS))
    best = min((e for e in vals if e.get("score") is not None), key=lambda e: e["score"], default=None)
    last = vals[-1]
    print()
    if best:
        print(f"best val score {best['score']:.3f} at step {best['step']}; last {last.get('score', float('nan')):.3f} "
              f"at step {last['step']}")
        if "by_corpus" in best["val"]:
            print("per corpus at the best step:")
            for corpus, m in sorted(best["val"]["by_corpus"].items()):
                print(f"  {corpus:10s}" + "".join(f"{k[3]}={get(m, *k[:3]):.3f}  " for k in KEYS))
    if trains:
        t = trains[-1]
        print(f"last training losses (step {t['step']}): " +
              ", ".join(f"{k}={t[k]:.4g}" for k in ("self", "occ", "accel", "total") if k in t))
    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "teaser": first.get("teaser"), "sg9": first.get("sg9"),
                                         "best_step": best and best["step"], "training": trains}, indent=1))


if __name__ == "__main__":
    main()
