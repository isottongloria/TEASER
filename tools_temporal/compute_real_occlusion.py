"""Real hand-face occlusion per frame, from a clip's TEASER-face SMPL-X fit (TEMPORAL_README.md 5.2).

Runs in RGB2SMPLX's env (it needs smplx and RGB2SMPLX's validated geometry):

    PYTHONPATH=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX ~/miniconda3/envs/rgb2smplx/bin/python \
        tools_temporal/compute_real_occlusion.py --fits <dir of clip dirs> --out <dir> [--clips list.txt]

For every clip ``<fits>/<clip>/fit_gvhmr_face_teaser_upper.npz`` (the SMPL-X
fit whose face comes from TEASER) writes ``<out>/<clip>.occ.npz``:

``c_mnc`` (T,)
    IoA of the WiLoR-observed hands over the mouth/nose/chin hull -- exactly
    the measure of RGB2SMPLX's occlusion protocol and jitter fix (SMPL-X mesh
    projected with the fit's own camera, ``experiments/occlusion_protocols_smplx/
    smplx_geometry.py``), so ``c_mnc > 0.2`` reproduces their episodes.
``c_mouth``, ``c_eyes`` (T,)
    the same IoA over the FLAME ``lips`` and eye regions of
    ``src/temporal/flame_regions.py``, mapped to SMPL-X through
    ``SMPL-X__FLAME_vertex_ids.npy``. These are the training ``c_t_real``.
``hand_valid`` (T, 2)
    WiLoR observed the left / right hand that frame (an unobserved hand's
    mesh is interpolated and does not count as an occluder).
``teaser_expr50``, ``teaser_jaw``, ``teaser_eyelid``
    TEASER's expression / jaw / eyelids as carried by the fit (interpolated
    and SG9-smoothed by RGB2SMPLX), for cross-checking our own TEASER run
    and frame alignment; not used as a training target.
``frame_shape``, ``fps``.

IoA is rasterised on the full frame (frame size = 2 x principal point, as in
RGB2SMPLX). A frame whose region hull is degenerate gets NaN.
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

RGB2SMPLX = Path("/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX")
TEASER_ROOT = Path(__file__).resolve().parents[1]
FIT_NAME = "fit_gvhmr_face_teaser_upper.npz"


def _hull(points):
    points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.convexHull(points).astype(np.int32).reshape(-1, 2)


def ioa(hand_hulls, region_hull, frame_shape):
    """area(union of hand hulls n region hull) / area(region hull); NaN if the region is empty."""
    height, width = frame_shape
    region = np.zeros((height, width), np.uint8)
    cv2.fillPoly(region, [region_hull.reshape(-1, 1, 2)], 1)
    area = int(region.sum())
    if area == 0:
        return float("nan")
    if not hand_hulls:
        return 0.0
    hands = np.zeros((height, width), np.uint8)
    for hull in hand_hulls:
        if len(hull) >= 3:
            cv2.fillPoly(hands, [hull.reshape(-1, 1, 2)], 1)
    return float(np.logical_and(region, hands).sum()) / area


def flame_region_ids_in_smplx():
    """lips / eyes FLAME regions as SMPL-X vertex indices."""
    sys.path.insert(0, str(TEASER_ROOT))
    from src.temporal.flame_regions import region_vertex_ids

    flame_to_smplx = np.load(RGB2SMPLX / "models/human_model_files/smplx/SMPL-X__FLAME_vertex_ids.npy")
    regions = region_vertex_ids(TEASER_ROOT / "assets")
    return {"mouth": flame_to_smplx[regions["mouth"]], "eyes": flame_to_smplx[regions["eyes"]]}


def clip_occlusion(fit_path, geometry, extra_regions):
    npz = dict(np.load(fit_path))
    vertices = geometry.forward_vertices(npz)
    projected = geometry.project_all_frames(vertices, npz)
    groups = geometry.vertex_groups()
    principal = npz["source_principal_xy"][0] if "source_principal_xy" in npz else None
    if principal is None:
        raise ValueError("fit without a source camera: cannot recover the frame size")
    frame_shape = (int(round(2 * principal[1])), int(round(2 * principal[0])))
    n = len(projected)
    hand_valid = np.asarray(npz.get("wilor_valid", np.ones((n, 2), bool)), dtype=bool)

    out = {name: np.full(n, np.nan, np.float32) for name in ("c_mnc", "c_mouth", "c_eyes")}
    region_ids = {"c_mnc": groups["mouth_nose_chin"], "c_mouth": extra_regions["mouth"],
                  "c_eyes": extra_regions["eyes"]}
    for t in range(n):
        hands = [_hull(projected[t, groups[side]])
                 for column, side in enumerate(("left_hand", "right_hand")) if hand_valid[t, column]]
        for name, ids in region_ids.items():
            out[name][t] = ioa(hands, _hull(projected[t, ids]), frame_shape)
    out.update(
        hand_valid=hand_valid,
        teaser_expr50=np.asarray(npz[geometry.expression_key(npz)], np.float32),
        teaser_jaw=np.asarray(npz["smplx_jaw_pose"], np.float32),
        teaser_eyelid=np.asarray(npz.get("smplx_eyelid", np.zeros((n, 2))), np.float32),
        frame_shape=np.asarray(frame_shape),
        fps=np.float32(geometry.DEFAULT_FPS),
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fits", type=Path, required=True, help="directory of clip directories")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clips", type=Path, help="file of clip names (default: every clip with a fit)")
    parser.add_argument("--fit-name", default=FIT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(RGB2SMPLX / "experiments/occlusion_protocols_smplx"))
    import smplx_geometry as geometry

    if args.clips:
        names = [line.strip() for line in args.clips.read_text().splitlines() if line.strip()]
    else:
        names = sorted(p.name for p in args.fits.iterdir() if (p / args.fit_name).is_file())
    extra = flame_region_ids_in_smplx()
    args.out.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    for k, name in enumerate(names):
        target = args.out / f"{name}.occ.npz"
        if target.is_file() and not args.overwrite:
            continue
        try:
            arrays = clip_occlusion(args.fits / name / args.fit_name, geometry, extra)
        except Exception as error:
            print(f"[occlusion] FAILED {name}: {error}", file=sys.stderr)
            failed += 1
            continue
        tmp = target.with_name(f".{target.name}.tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(target)
        done += 1
        if (k + 1) % 50 == 0:
            print(f"[occlusion] {k + 1}/{len(names)}", file=sys.stderr)
    print(f"[occlusion] {done} written, {failed} failed -> {args.out}")


if __name__ == "__main__":
    main()
