"""Pick the train clips Pixel3DMM pseudo-GT is run on (TEMPORAL_README.md, 3).

Runs in RGB2SMPLX's own env, not TEASER's (it reuses RGB2SMPLX's occlusion
detector, read-only):

    PYTHONPATH=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX ~/miniconda3/envs/rgb2smplx/bin/python \
        tools_temporal/select_train_clips.py --corpus /leonardo_work/IscrC_SLPSCALE/phoenix/phoenix/train \
        --out tools_temporal/lists/phoenix_train --rounds 100 200 --clean-frac 0.75

Scores every clip with the hand-face occlusion measure of RGB2SMPLX's jitter
fix (``rgb2smplx/stages/face_jitter_fix.py``), computed from the clip's fitted
SMPL-X (``fit_gvhmr_face_teaser_upper.npz`` by default -- on PHOENIX train
the only per-clip file this account can read): the fitted mesh projected
with the clip's "source" camera, IoA = area(WiLoR-valid hand hulls n
mouth/nose/chin hull) / area(mouth/nose/chin hull); a frame is occluded above
``--tau`` (0.20, the jitter fix's TAU). Same functions, so the same frames the
production jitter fix would splice. This is only for choosing clips: the
training-time ``c_t`` is computed in the TEASER crop (src/temporal/occlusion_conf.py).

Classes, by the fraction of occluded frames: "clean" = at most
``--clean-max-frac`` (5 %); "occluded" = at least ``--occ-min-frac`` (10 %)
and ``--min-occ-frames``; the rest is "ambiguous" and never picked. Fully
clean clips are rare in sign language (in a PHOENIX train sample 14 of 20
clips had an occluded frame, and the clean ones were all short), so "clean"
means mostly clean: synthetic occlusion is pasted on its clean windows, and
the clean frames of every clip carry valid pseudo-GT anyway. Rounds are nested and disjoint (round 2 adds
clips to round 1), each with the same clean/occluded mix. Writes one clip
name per line per round, plus a TSV with every clip's scores.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def ioa_series(data, model, groups, device, fjf):
    """Per-frame IoA as the jitter fix computes it (NaN where the face hull is degenerate)."""
    # Older fits (e.g. PHOENIX train, 2026-09-22) carry 10 expression
    # coefficients and today's loader builds 100: zero-padding gives the same mesh.
    expr = np.asarray(data["smplx_expr"], np.float32)
    width = model.num_expression_coeffs
    if expr.shape[1] < width:
        data = dict(data, smplx_expr=np.pad(expr, ((0, 0), (0, width - expr.shape[1]))))
    vertices = fjf._forward_vertices(data, model, device)
    principal = data["source_principal_xy"][0]
    frame_shape = (int(round(2 * principal[1])), int(round(2 * principal[0])))
    wilor_valid = data.get("wilor_valid")
    values = np.full(len(vertices), np.nan)
    for i in range(len(vertices)):
        proj = fjf._project_frame(vertices[i], data, i)
        target = fjf._convex_hull_polygon(proj[groups["mouth_nose_chin"]])
        hulls = [fjf._convex_hull_polygon(proj[groups[side]])
                 for column, side in enumerate(("left_hand", "right_hand"))
                 if wilor_valid is None or bool(wilor_valid[i][column])]
        values[i] = fjf._ioa(hulls, target, frame_shape)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True, help="directory of RGB2SMPLX clip work dirs")
    parser.add_argument("--fit-name", default="fit_gvhmr_face_teaser_upper.npz")
    parser.add_argument("--out", type=Path, required=True, help="output prefix")
    parser.add_argument("--rounds", type=int, nargs="+", default=[100, 200],
                        help="clips added per round (nested, disjoint)")
    parser.add_argument("--clean-frac", type=float, default=0.75)
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument("--clean-max-frac", type=float, default=0.05)
    parser.add_argument("--occ-min-frac", type=float, default=0.10)
    parser.add_argument("--min-occ-frames", type=int, default=3)
    parser.add_argument("--min-frames", type=int, default=32, help="shorter clips are never picked")
    parser.add_argument("--exclude", type=Path, nargs="*", default=[],
                        help="files of clip names never to pick")
    parser.add_argument("--limit", type=int, help="score only the first N clips (smoke test)")
    parser.add_argument("--sample", type=int, help="score a random sample of N clips (seeded)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", type=Path,
                        default=Path("/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX/models/human_model_files"))
    args = parser.parse_args()

    import torch
    from rgb2smplx.face_identity import model_kwargs
    from rgb2smplx.face_model import face_model_of, model_dir_for
    from rgb2smplx.stages import face_jitter_fix as fjf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    excluded = set()
    for path in args.exclude:
        excluded |= {line.strip() for line in path.read_text().splitlines() if line.strip()}
    clips = sorted(p for p in args.corpus.iterdir()
                   if (p / args.fit_name).is_file() and p.name not in excluded)
    if args.sample and args.sample < len(clips):
        pick = np.random.default_rng(args.seed + 1).choice(len(clips), args.sample, replace=False)
        clips = [clips[i] for i in sorted(pick)]
    if args.limit:
        clips = clips[:args.limit]
    print(f"[select] {len(clips)} candidate clips in {args.corpus} ({device})", file=sys.stderr)

    models = {}
    rows, clean, occluded = [], [], []
    for k, clip in enumerate(clips):
        try:
            data = dict(np.load(clip / args.fit_name))
            model_dir = model_dir_for(args.models / "smplx", face_model_of(data))
            v_template = model_kwargs(data, model_dir.parent / "smplx").get("v_template")
            key = (str(model_dir), None if v_template is None else hash(v_template.tobytes()))
            if key not in models:
                models[key] = fjf._load_smplx_model(model_dir, device, v_template)
            values = ioa_series(data, models[key], fjf._vertex_groups(model_dir), device, fjf)
        except Exception as error:  # unreadable/corrupt file: skip, but say so
            print(f"[select] skip {clip.name}: {error}", file=sys.stderr)
            continue
        seen = values[~np.isnan(values)]
        s = {"frames": len(values), "face_frames": int(len(seen)),
             "occ_frames": int((seen > args.tau).sum()),
             "max_ioa": float(seen.max()) if len(seen) else float("nan"),
             "mean_ioa": float(seen.mean()) if len(seen) else float("nan")}
        long_enough = s["frames"] >= args.min_frames and s["face_frames"] == s["frames"]
        occ_frac = s["occ_frames"] / max(s["frames"], 1)
        if long_enough and occ_frac <= args.clean_max_frac:
            label = "clean"
            clean.append(clip.name)
        elif long_enough and occ_frac >= args.occ_min_frac and s["occ_frames"] >= args.min_occ_frames:
            label = "occluded"
            occluded.append(clip.name)
        else:
            label = "ambiguous"
        rows.append((clip.name, label, s))
        if (k + 1) % 250 == 0:
            print(f"[select] {k + 1}/{len(clips)}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{args.out}_scores.tsv", "w") as handle:
        handle.write("clip\tlabel\tframes\tface_frames\tocc_frames\tmax_ioa\tmean_ioa\n")
        for name, label, s in rows:
            handle.write(f"{name}\t{label}\t{s['frames']}\t{s['face_frames']}\t{s['occ_frames']}\t"
                         f"{s['max_ioa']:.4f}\t{s['mean_ioa']:.4f}\n")

    rng = np.random.default_rng(args.seed)
    clean = list(rng.permutation(clean))
    occluded = list(rng.permutation(occluded))
    print(f"[select] clean {len(clean)}, occluded {len(occluded)}, "
          f"ambiguous {len(rows) - len(clean) - len(occluded)}", file=sys.stderr)
    for r, size in enumerate(args.rounds, start=1):
        n_clean = int(round(size * args.clean_frac))
        n_occ = size - n_clean
        if n_clean > len(clean) or n_occ > len(occluded):
            raise SystemExit(f"round {r}: not enough clips (need {n_clean} clean / {n_occ} occluded)")
        picked = sorted(clean[:n_clean] + occluded[:n_occ])
        clean, occluded = clean[n_clean:], occluded[n_occ:]
        Path(f"{args.out}_round{r}.txt").write_text("\n".join(picked) + "\n")
        print(f"[select] round {r}: {n_clean} clean + {n_occ} occluded -> {args.out}_round{r}.txt",
              file=sys.stderr)


if __name__ == "__main__":
    main()
