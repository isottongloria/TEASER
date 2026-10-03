"""Hand bank: RGBA hand cut-outs from our own frames, guided by the pipeline's fits (TEMPORAL_README.md 5.4).

Runs in RGB2SMPLX's env (smplx + RGB2SMPLX's validated geometry):

    PYTHONPATH=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX ~/miniconda3/envs/rgb2smplx/bin/python \
        tools_temporal/build_hand_bank.py --corpus phoenix --clips_root data/temporal/phoenix/clips \
        --fits_of fits.tsv --split train=lists/train.txt --split test=lists/test.txt --out data/temporal/hand_bank

For every clip and every 2nd frame, each hand WiLoR actually observed
(``wilor_valid``) is a candidate when it is

- large enough (``--min_px``, the hand's projected height),
- away from the face (no overlap with the frontal face hull grown by 0.15 face size) and
  from the other hand (hulls disjoint),
- sharp (Laplacian variance inside the hand, measured at 64 px hand height so
  that it does not depend on the resolution, >= ``--min_sharpness``).

The silhouette is the fit's MANO hand plus a short forearm stub (SMPL-X
vertices driven by the elbow, within 8 cm of the wrist), projected with the
fit's camera and rasterised triangle by triangle (finger gaps stay open). It
seeds GrabCut on the image: eroded silhouette = sure hand, dilated band =
probable, outside = background. A cut-out is kept when GrabCut and the
silhouette agree (IoU >= ``--min_iou``) and at least ``--min_skin`` of it has a
skin colour (not mostly sleeve); its alpha is feathered by 1 px.
At most ``--per_clip`` cut-outs per clip and hand side, the sharpest ones.

Writes ``<out>/<split>/<corpus>__<signer>__<clip>__f<frame>__<side>.png`` and
``<out>/index.tsv`` (file, split, corpus, signer, clip, frame, side, height px,
IoU, sharpness, and the 45 MANO pose values for handshape clustering).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

RGB2SMPLX = Path("/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX")
FOREARM_CM = 0.08


def hand_vertex_sets(geometry):
    """{'left'/'right': (vertex ids, face rows)} = MANO hand + forearm stub."""
    import torch

    model = geometry._model()
    groups = geometry.vertex_groups()
    weights = model.lbs_weights.argmax(1).numpy()
    with torch.no_grad():
        joints = torch.einsum("jv,vx->jx", model.J_regressor, model.v_template).numpy()
    template = model.v_template.numpy()
    faces = np.asarray(model.faces, dtype=np.int64)
    out = {}
    for side, elbow, wrist in (("left", 18, 20), ("right", 19, 21)):
        stub = np.where((weights == elbow) & (np.linalg.norm(template - joints[wrist], axis=1) < FOREARM_CM))[0]
        ids = np.union1d(np.asarray(groups[f"{side}_hand"]), stub)
        inside = np.zeros(len(template), bool)
        inside[ids] = True
        out[side] = (ids, faces[inside[faces].all(1)])
    return out


def silhouette(points_xy, faces, shape):
    mask = np.zeros(shape, np.uint8)
    tris = np.round(points_xy[faces]).astype(np.int32)
    for tri in tris:
        cv2.fillConvexPoly(mask, tri, 1)
    return mask


def sharpness(frame, sil, height=64):
    """Laplacian variance inside the hand, on the hand box resized to ``height`` px (scale-free)."""
    ys, xs = np.nonzero(sil)
    box = frame[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    mask = sil[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = height / max(box.shape[0], 1)
    size = (max(8, int(round(box.shape[1] * scale))), height)
    gray = cv2.cvtColor(cv2.resize(box, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) > 0
    return float(cv2.Laplacian(gray, cv2.CV_64F)[mask].var()) if mask.any() else 0.0


def skin_fraction(rgba):
    """Share of the cut-out's pixels with a skin colour (YCrCb 133<=Cr<=173, 77<=Cb<=127)."""
    inside = rgba[..., 3] > 128
    if not inside.any():
        return 0.0
    ycrcb = cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return float(skin[inside].mean())


def hull_mask(points_xy, shape):
    mask = np.zeros(shape, np.uint8)
    hull = cv2.convexHull(np.round(points_xy).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    return mask


def cut_out(frame, sil, margin):
    """GrabCut seeded by the silhouette, inside the silhouette's box + margin. -> (rgba, iou) or None."""
    ys, xs = np.nonzero(sil)
    h, w = sil.shape
    y0, y1 = max(0, ys.min() - margin), min(h, ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(w, xs.max() + margin + 1)
    crop, s = frame[y0:y1, x0:x1], sil[y0:y1, x0:x1]
    k = max(1, int(round(0.04 * max(s.shape))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    sure = cv2.erode(s, kernel)
    probable = cv2.dilate(s, kernel)
    gc = np.full(s.shape, cv2.GC_BGD, np.uint8)
    gc[probable > 0] = cv2.GC_PR_BGD
    gc[s > 0] = cv2.GC_PR_FGD
    gc[sure > 0] = cv2.GC_FGD
    if (gc == cv2.GC_FGD).sum() < 10:
        return None
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:  # keep the largest blob only
        fg = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    union = np.logical_or(fg, s).sum()
    iou = float(np.logical_and(fg, s).sum() / union) if union else 0.0
    alpha = cv2.GaussianBlur(fg.astype(np.float32) * 255, (3, 3), 0)
    return np.dstack([crop, alpha.astype(np.uint8)]), iou


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--clips_root", type=Path, required=True, help="our clip dirs with frames/")
    parser.add_argument("--fits_of", type=Path, required=True, help="clip<TAB>fit npz path")
    parser.add_argument("--split", action="append", required=True, help="NAME=list file (clip<TAB>signer)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min_px", type=int, default=24)
    parser.add_argument("--min_sharpness", type=float, default=20.0)
    parser.add_argument("--min_iou", type=float, default=0.6)
    parser.add_argument("--per_clip", type=int, default=3)
    parser.add_argument("--min_skin", type=float, default=0.4, help="reject cut-outs that are mostly sleeve")
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    sys.path.insert(0, str(RGB2SMPLX / "experiments/occlusion_protocols_smplx"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import smplx_geometry as geometry
    from compute_real_occlusion import fit_for_geometry

    fit_of = dict(l.split("\t")[:2] for l in args.fits_of.read_text().splitlines() if l.strip())
    sets = hand_vertex_sets(geometry)
    # The frontal face only (FLAME 'face' region), not the whole head: hands in
    # front of the neck or beside the ears are fine occluders to cut out.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.temporal.flame_regions import region_vertex_ids
    flame_to_smplx = np.load(RGB2SMPLX / "models/human_model_files/smplx/SMPL-X__FLAME_vertex_ids.npy")
    face_ids = flame_to_smplx[region_vertex_ids(Path(__file__).resolve().parents[1] / "assets")["face"]]
    args.out.mkdir(parents=True, exist_ok=True)
    index = open(args.out / "index.tsv", "a")
    kept_total = 0
    for spec in args.split:
        split, list_path = spec.split("=", 1)
        (args.out / split).mkdir(exist_ok=True)
        rows = [l.split("\t")[:2] for l in Path(list_path).read_text().splitlines() if l.strip()]
        for clip, signer in rows:
            npz = fit_for_geometry(dict(np.load(fit_of[clip])), geometry)
            projected = geometry.project_all_frames(geometry.forward_vertices(npz), npz)
            frames = sorted((args.clips_root / clip / "frames").iterdir())
            valid = np.asarray(npz.get("wilor_valid", np.ones((len(projected), 2), bool)), bool)
            if len(frames) != len(projected):
                print(f"[hands] skip {clip}: {len(frames)} frames, fit has {len(projected)}", file=sys.stderr)
                continue
            candidates = {"left": [], "right": []}
            for t in range(0, len(frames), args.stride):
                frame = None
                face = projected[t, face_ids]
                face_size = np.ptp(face, axis=0).max()
                for col, side in enumerate(("left", "right")):
                    if not valid[t, col]:
                        continue
                    ids, faces = sets[side]
                    pts = projected[t]
                    hand = pts[ids]
                    height = np.ptp(hand, axis=0).max()
                    if height < args.min_px:
                        continue
                    if frame is None:
                        frame = cv2.imread(str(frames[t]))
                        shape = frame.shape[:2]
                        face_m = cv2.dilate(hull_mask(face, shape), np.ones((3, 3), np.uint8),
                                            iterations=max(1, int(0.15 * face_size)))
                    sil = silhouette(pts, faces, shape)
                    if sil.sum() < 20 or (sil & face_m).any():
                        continue
                    other = "right" if side == "left" else "left"
                    if valid[t, 1 - col] and (hull_mask(pts[sets[other][0]], shape) & sil).any():
                        continue
                    sharp = sharpness(frame, sil)
                    if sharp < args.min_sharpness:
                        continue
                    candidates[side].append((sharp, t, height))
            for side, items in candidates.items():
                pose_key = "smplx_lhand_pose" if side == "left" else "smplx_rhand_pose"
                kept = 0
                for sharp, t, height in sorted(items, reverse=True)[:args.per_clip * 3]:
                    if kept >= args.per_clip:
                        break
                    frame = cv2.imread(str(frames[t]))
                    pts = projected[t]
                    sil = silhouette(pts, sets[side][1], frame.shape[:2])
                    result = cut_out(frame, sil, margin=max(4, int(0.15 * height)))
                    if result is None or result[1] < args.min_iou:
                        continue
                    rgba, iou = result
                    if skin_fraction(rgba) < args.min_skin:
                        continue
                    name = f"{args.corpus}__{signer}__{clip}__f{t:05d}__{side}.png"
                    cv2.imwrite(str(args.out / split / name), rgba)
                    pose = np.asarray(npz[pose_key][t], np.float32).reshape(-1)
                    index.write("\t".join([name, split, args.corpus, signer, clip, str(t), side, f"{height:.0f}",
                                           f"{iou:.3f}", f"{sharp:.1f}"] + [f"{v:.4f}" for v in pose]) + "\n")
                    kept += 1
                    kept_total += 1
            index.flush()
        print(f"[hands] {args.corpus} {split}: {len(list((args.out / split).glob(args.corpus + '__*.png')))} cut-outs")
    index.close()
    print(f"[hands] {kept_total} new cut-outs -> {args.out}")


if __name__ == "__main__":
    main()
