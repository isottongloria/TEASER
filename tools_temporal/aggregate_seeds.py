"""Average eval_temporal.py results over seeds; split a big evaluation into tables for the review page.

    python tools_temporal/aggregate_seeds.py --eval runs/eval_phase3_val.json \
        --group "T7 replay=T7_replay_s0,T7_replay_s1,T7_replay_s2" ... \
        --keep T0 --keep T2 --out runs/eval_phase3_seeds.json

Each ``--group NAME=a,b,c`` becomes one method whose metrics are the mean over
the listed methods, with the standard deviation stored under ``_std``
(same layout) and printed; ``--keep`` copies methods unchanged. The output has
the eval_temporal.py layout, so build_review_page.py shows it as a table.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def walk(entries, fn):
    """Combine the numeric leaves of several same-shaped dicts with fn(list) -> value."""
    first = entries[0]
    if isinstance(first, dict):
        return {k: walk([e[k] for e in entries if k in e], fn) for k in first
                if all(isinstance(e, dict) and k in e for e in entries)}
    if isinstance(first, (int, float)):
        return fn([float(e) for e in entries])
    return first


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--group", action="append", default=[], help="NAME=method,method,...")
    parser.add_argument("--keep", action="append", default=[], help="method to copy unchanged (NAME or NAME=new name)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.eval.read_text())
    out = {}
    for spec in args.keep:
        old, _, new = spec.partition("=")
        out[new or old] = data[old]
    for spec in args.group:
        name, members = spec.split("=", 1)
        members = members.split(",")
        entries = [{k: v for k, v in data[m].items() if k in ("real", "synthetic")} for m in members]
        mean = walk(entries, lambda xs: float(np.mean(xs)))
        std = walk(entries, lambda xs: float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0)
        out[name] = {"spec": "mean of " + ",".join(members), "kind": data[members[0]].get("kind", "adapter"),
                     **mean, "_std": std, "_members": members}
        g = lambda d, k, gr, key: d.get(k, {}).get(gr, {}).get(key, float("nan"))
        print(f"{name:28s} occl {g(mean, 'synthetic', 'occluded', 'err_mouth'):.3f}±{g(std, 'synthetic', 'occluded', 'err_mouth'):.3f}"
              f"  near {g(mean, 'synthetic', 'near', 'err_mouth'):.3f}±{g(std, 'synthetic', 'near', 'err_mouth'):.3f}"
              f"  drift {g(mean, 'real', 'clean', 'err_face'):.3f}±{g(std, 'real', 'clean', 'err_face'):.3f}"
              f"  jerk {g(mean, 'real', 'clean', 'jerk'):.3f}±{g(std, 'real', 'clean', 'jerk'):.3f}")
    args.out.write_text(json.dumps(out, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
