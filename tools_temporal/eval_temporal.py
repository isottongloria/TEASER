"""Evaluate Track 1 methods on real clips and synthetic-occlusion variants (TEMPORAL_README.md 6, 7).

    python tools_temporal/eval_temporal.py --cache <dir> --occ <dir> --variants <dir> --list test.txt \
        --method T0=teaser --method T1=sg9 --method T2=sg9+interp \
        --method T3=runs/T3/last.pt --method T7=runs/T7/last.pt --out results/test.json

Methods: ``teaser`` (per-frame, T0), ``sg9`` (T1), ``sg9+interp`` (T2, the current
RGB2SMPLX pipeline: SG9 then the Hermite splice on occlusion episodes), or a
checkpoint of train_temporal.py (adapter or SmoothNet); ``--external NAME=dir``
adds predictions from ``dir/<clip>.npz`` (``expression``, ``jaw_pose``,
``eyelid``), e.g. SPECTRE.

Every method sees the frames a clip shows -- for a variant, the pasted ones
-- and the occlusion signal the pipeline would have: real ``c_mnc`` and, on a
variant, the pasted hand's coverage. References: TEASER on the unoccluded
frames. Results per method, for ``real`` and ``synthetic`` clips, per frame
group (clean / near / occluded) and, for synthetic, per episode duration; plus
ms/frame of the temporal part. Also prints the 7.1 success checks for each
learned method.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def postproc_method(name, clip):
    from src.temporal import postproc

    params = {k: v.copy() for k, v in clip.input_params.items()}
    if name == "teaser":
        return params
    smoothed = {k: postproc.savgol(v, 9, 2) for k, v in params.items()}
    if name == "sg9":
        return smoothed
    if name == "sg9+interp":
        occ = np.maximum(clip.c_mnc, clip.c_syn[:, 0])
        spliced, _ = postproc.hermite_interp({"expression": smoothed["expression"], "jaw": smoothed["jaw"]}, occ)
        return {"expression": spliced["expression"], "jaw": spliced["jaw"], "eyelid": smoothed["eyelid"]}
    raise ValueError(f"unknown method {name!r}")


def load_checkpoint(path, device):
    import torch
    from omegaconf import OmegaConf
    from src.temporal.model import build_model
    from src.temporal.split_encoder import load_teaser_encoder

    state = torch.load(path, map_location=device)
    cfg = OmegaConf.create(state["config"])
    encoder = load_teaser_encoder(REPO_ROOT / cfg.checkpoint, device)
    model = build_model(cfg.model, encoder, cfg.data.feature_set, cfg.train.window).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--occ", type=Path)
    parser.add_argument("--variants", type=Path)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--method", action="append", default=[],
                        help="NAME=teaser|sg9|sg9+interp|<ckpt.pt>|<ckpt.pt>+sg9")
    parser.add_argument("--external", action="append", default=[], help="NAME=<dir of <clip>.npz>")
    parser.add_argument("--near_k", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    paths = {k: (v.resolve() if v else None) for k, v in vars(args).items()
             if k in ("cache", "occ", "variants", "list", "out")}
    methods = [m.split("=", 1) for m in args.method]
    def resolve(v):
        # "<ckpt.pt>+sg9": the model's output, then SG9 (stability of SG9 on top of the model).
        if v in ("teaser", "sg9", "sg9+interp"):
            return v
        if v.endswith("+sg9"):
            return str(Path(v[:-4]).resolve()) + "+sg9"
        return str(Path(v).resolve())
    methods = [(n, resolve(v)) for n, v in methods]
    externals = [(n, Path(v).resolve()) for n, v in (e.split("=", 1) for e in args.external)]
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    from src.temporal import metrics as M
    from src.temporal.data import full_clip, load_clips, load_variants, read_list
    from src.temporal.flame_regions import CanonicalFlame
    from tools_temporal.train_temporal import to_device

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    flame = CanonicalFlame(vertex_ids="face").to(device)
    region_index = {r: flame.region_index(r) for r in ("mouth", "eyes", "rest")}
    region_index["face"] = np.arange(flame.vertex_ids.numel())
    names = read_list(paths["list"])

    models = {}
    for name, spec in methods:
        if spec.endswith(".pt") or spec.endswith(".pt+sg9"):
            models[name] = load_checkpoint(spec.removesuffix("+sg9") if hasattr(str, "removesuffix")
                                           else spec[:-4] if spec.endswith("+sg9") else spec, device)

    report = {}
    for name, spec in methods + [(n, f"external:{d}") for n, d in externals]:
        feature_set = models[name][1].data.feature_set if name in models else "expr"
        clips = load_clips(names, paths["cache"], paths["occ"], feature_set)
        variants = load_variants(clips, paths["variants"], feature_set) if paths["variants"] else []
        results = {"real": [], "synthetic": []}
        frames, seconds = 0, 0.0
        for kind, items in (("real", clips), ("synthetic", variants)):
            for clip in items:
                if name in models:
                    model, cfg = models[name]
                    batch = to_device(full_clip(clip), device)
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    t0 = time.time()
                    pred = model.infer_clip(batch, window=cfg.train.window, stride=cfg.train.infer_stride)
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    seconds += time.time() - t0
                    frames += len(clip)
                    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
                    if spec.endswith("+sg9"):
                        from src.temporal.postproc import savgol
                        pred = {k: savgol(v, 9, 2) for k, v in pred.items()}
                elif spec.startswith("external:"):
                    ext_name = clip.name if kind == "real" else clip.name  # variants: <clip>.v<k>.npz
                    with np.load(Path(spec[len("external:"):]) / f"{ext_name}.npz") as ext:
                        pred = {"expression": ext["expression"], "jaw": ext["jaw_pose"], "eyelid": ext["eyelid"]}
                else:
                    pred = postproc_method(spec, clip)
                real_occ = clip.c_mnc > 0.2
                if kind == "real":
                    groups = M.frame_groups(real_occ, clip.valid, args.near_k)
                else:
                    groups = M.frame_groups(clip.syn_occ, clip.valid, args.near_k, exclude=real_occ)
                results[kind].append(M.clip_metrics(M.canonical_mm(flame, pred, device),
                                                    M.canonical_mm(flame, clip.teacher, device), groups,
                                                    region_index, clip.episodes if kind == "synthetic" else None))
        kind_of = (models[name][1].model.get("kind", "adapter") + ("+sg9" if spec.endswith("+sg9") else "")) if name in models else \
            ("external" if spec.startswith("external:") else "postproc")
        report[name] = {"spec": spec, "kind": kind_of, **{k: M.pool(v) for k, v in results.items() if v}}
        if frames:
            report[name]["ms_per_frame"] = 1000 * seconds / frames

    report["_success"] = success_checks(report)
    paths["out"].parent.mkdir(parents=True, exist_ok=True)
    paths["out"].write_text(json.dumps(report, indent=1))
    print_table(report)
    print(f"-> {paths['out']}")


def success_checks(report):
    """TEMPORAL_README.md 7.1, for every learned method against T0 / T1 / T2 / T3 when present."""
    def get(name, kind, group, key):
        return report.get(name, {}).get(kind, {}).get(group, {}).get(key, float("nan"))

    baselines = [n for n in report if report[n]["kind"] in ("postproc", "smoothnet", "external")]
    teaser = next((n for n in report if report[n]["spec"] == "teaser"), None)
    sg9 = next((n for n in report if report[n]["spec"] == "sg9"), None)
    out = {}
    for name, entry in report.items():
        if not entry["kind"].startswith("adapter"):
            continue
        checks = {}
        for group in ("occluded", "near"):
            mine = get(name, "synthetic", group, "err_mouth")
            others = {b: get(b, "synthetic", group, "err_mouth") for b in baselines if b != name}
            checks[f"1_{group}_beats_baselines"] = bool(others) and all(mine < v for v in others.values())
        if teaser:
            checks["2_clean_fidelity_mm"] = get(name, "real", "clean", "err_face")
        if teaser and sg9:
            mine = get(name, "real", "clean", "jerk")
            checks["3_jerk_below_teaser"] = mine < get(teaser, "real", "clean", "jerk")
            checks["3_jerk_vs_sg9"] = mine / get(sg9, "real", "clean", "jerk")
        out[name] = checks
    return out


def print_table(report):
    rows = [("synthetic", "occluded", "err_mouth"), ("synthetic", "near", "err_mouth"),
            ("synthetic", "occluded", "err_face"), ("synthetic", "clean", "err_face"),
            ("real", "clean", "err_face"), ("real", "clean", "err_mouth"),
            ("real", "clean", "jerk"), ("real", "occluded", "jerk"), ("real", "near", "jerk")]
    names = [n for n in report if not n.startswith("_")]
    print(f"{'metric (mm)':34s}" + "".join(f"{n:>12s}" for n in names))
    for kind, group, key in rows:
        values = [report[n].get(kind, {}).get(group, {}).get(key, float("nan")) for n in names]
        print(f"{kind + '/' + group + '/' + key:34s}" + "".join(f"{v:12.3f}" for v in values))
    buckets = report[names[0]].get("synthetic", {}).get("by_duration", {})
    for bucket in buckets:
        values = [report[n].get("synthetic", {}).get("by_duration", {}).get(bucket, float("nan")) for n in names]
        print(f"{'synthetic/err_mouth/dur ' + bucket:34s}" + "".join(f"{v:12.3f}" for v in values))
    timing = [report[n].get("ms_per_frame", float("nan")) for n in names]
    print(f"{'ms/frame (temporal part)':34s}" + "".join(f"{v:12.3f}" for v in timing))
    for name, checks in report.get("_success", {}).items():
        print(f"[7.1] {name}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in checks.items()))


if __name__ == "__main__":
    main()
