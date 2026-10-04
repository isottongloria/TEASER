"""Transplant bank: real hand-over-face occlusion episodes, cut out frame by frame (TEMPORAL_README.md 5.4b).

Runs in RGB2SMPLX's env:

    PYTHONPATH=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX ~/miniconda3/envs/rgb2smplx/bin/python \
        tools_temporal/build_transplant_bank.py --corpus phoenix --occ data/temporal/phoenix/occ \
        --fits_of fits.tsv --frames_of frames.tsv --signers candidates.tsv \
        --split train=Signer01,Signer03 --split heldout=Signer04,Signer07 --out data/temporal/transplant_bank

Instead of a still hand moved along a made-up path, a *real* occlusion is
lifted from a video where it happened and replayed on a clean clip: its
duration, approach and retreat, hand shape, motion blur and coverage are
real by construction.

Sources: every episode of real occlusion (``c_mnc > 0.2``, compute_real_occlusion.py)
in the candidate clips of the given signers, 1-20 frames long, plus
``--context`` frames on each side (the approach and the retreat). Per frame,
the occluding hands are the WiLoR-observed hands whose silhouette lies within
half a face size of the face; an episode is dropped when a core frame has no
observed occluding hand.

Cut-out, per frame and hand: the fit's MANO hand + forearm stub (faded along
the forearm) projected with the fit's camera is a **tight constraint** -- the
hand is in front of a face of the same colour, so GrabCut may only remove
pixels from the silhouette or add a thin band (3 % of the hand's size), never
more; the frame is rejected when GrabCut and the silhouette disagree (IoU <
``--min_iou``), and the episode when a core frame is rejected. Both hands'
cut-outs are merged into one RGBA crop per frame.

Writes ``<out>/<split>/<corpus>__<signer>__<clip>__e<start>/``: ``fNN.png``
(RGBA crop per frame of the window) and ``meta.json`` with, per frame, the
crop origin in the source frame and the source face anchors (lips centre,
face size = eye-region width, eye-axis angle), plus the episode's core
start / length inside the window and the source c_mnc.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

RGB2SMPLX = Path("/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX")


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


def eye_axis(points):
    """Angle (degrees) of the eye region's principal axis, folded to [-90, 90)."""
    centred = points - points.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    angle = float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0])))
    return (angle + 90) % 180 - 90


class FrameSource:
    """Frames of a clip: a frames/ directory, or a video read by frame index."""

    def __init__(self, path):
        self.path = Path(path)
        self.files = sorted(self.path.iterdir()) if self.path.is_dir() else None

    def __len__(self):
        if self.files is not None:
            return len(self.files)
        cap = cv2.VideoCapture(str(self.path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    def read(self, indices):
        if self.files is not None:
            return {t: cv2.imread(str(self.files[t])) for t in indices}
        out, cap = {}, cv2.VideoCapture(str(self.path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(indices))
        for t in range(min(indices), max(indices) + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if t in indices:
                out[t] = frame
        cap.release()
        return out


def constrained_cut(frame, sil, fade, band):
    """GrabCut confined to the silhouette plus a thin band. -> (rgba crop, origin, iou) or None."""
    ys, xs = np.nonzero(sil)
    h, w = sil.shape
    m = band + 2
    y0, y1 = max(0, ys.min() - m), min(h, ys.max() + m + 1)
    x0, x1 = max(0, xs.min() - m), min(w, xs.max() + m + 1)
    crop, s, f = frame[y0:y1, x0:x1], sil[y0:y1, x0:x1], fade[y0:y1, x0:x1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    allowed = cv2.dilate(s, kernel)
    sure = cv2.erode(s, kernel)
    gc = np.full(s.shape, cv2.GC_BGD, np.uint8)
    gc[allowed > 0] = cv2.GC_PR_BGD
    gc[s > 0] = cv2.GC_PR_FGD
    gc[sure > 0] = cv2.GC_FGD
    if (gc == cv2.GC_FGD).sum() < 10:
        return None
    try:
        cv2.grabCut(crop, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = (((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)) & (allowed > 0)).astype(np.uint8)
    union = np.logical_or(fg, s).sum()
    iou = float(np.logical_and(fg, s).sum() / union) if union else 0.0
    alpha = cv2.GaussianBlur(fg.astype(np.float32) * 255, (3, 3), 0) * f
    return np.dstack([crop, alpha.astype(np.uint8)]), (int(x0), int(y0)), iou


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--occ", type=Path, required=True)
    parser.add_argument("--fits_of", type=Path, required=True, help="clip<TAB>fit npz")
    parser.add_argument("--frames_of", type=Path, required=True, help="clip<TAB>frames dir or video file")
    parser.add_argument("--signers", type=Path, required=True, help="clip<TAB>signer (candidates)")
    parser.add_argument("--split", action="append", required=True, help="NAME=signer,signer,...")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--min_iou", type=float, default=0.6)
    parser.add_argument("--max_per_split", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(RGB2SMPLX / "experiments/occlusion_protocols_smplx"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import smplx_geometry as geometry
    from build_hand_bank import fade_map, hand_vertex_sets, silhouette
    from compute_real_occlusion import fit_for_geometry, flame_region_ids_in_smplx

    fit_of = dict(l.split("\t")[:2] for l in args.fits_of.read_text().splitlines() if l.strip())
    frames_of = dict(l.split("\t")[:2] for l in args.frames_of.read_text().splitlines() if l.strip())
    signer_of = dict(l.split("\t")[:2] for l in args.signers.read_text().splitlines() if l.strip())
    split_of = {}
    for spec in args.split:
        name, signers = spec.split("=", 1)
        for signer in signers.split(","):
            split_of[signer] = name
    sets = hand_vertex_sets(geometry)
    regions = flame_region_ids_in_smplx()
    rng = np.random.default_rng(args.seed)

    # Episodes per split, from the occlusion files only (cheap), then shuffled and capped.
    episodes = {name: [] for name in set(split_of.values())}
    for clip, signer in signer_of.items():
        split = split_of.get(signer)
        occ_path = args.occ / f"{clip}.occ.npz"
        if split is None or clip not in fit_of or clip not in frames_of or not occ_path.is_file():
            continue
        with np.load(occ_path) as occ:
            c = np.nan_to_num(occ["c_mnc"], nan=0.0)
        for s, e in runs(c > 0.2):
            if e - s <= args.max_len and s - args.context >= 0 and e + args.context <= len(c):
                episodes[split].append((clip, signer, s, e))
    kept = {}
    for split, items in episodes.items():
        order = rng.permutation(len(items))
        kept[split], done = 0, 0
        by_clip = {}
        for k in order:
            by_clip.setdefault(items[k][0], []).append(items[k])
        for clip, eps in by_clip.items():
            if kept[split] >= args.max_per_split:
                break
            npz = fit_for_geometry(dict(np.load(fit_of[clip])), geometry)
            projected = geometry.project_all_frames(geometry.forward_vertices(npz), npz)
            valid = np.asarray(npz.get("wilor_valid", np.ones((len(projected), 2), bool)), bool)
            source = FrameSource(frames_of[clip])
            for _, signer, s, e in eps:
                if kept[split] >= args.max_per_split:
                    break
                lo, hi = s - args.context, e + args.context
                frames = source.read(set(range(lo, hi)))
                if len(frames) != hi - lo:
                    continue
                meta, crops, ok = [], [], True
                for t in range(lo, hi):
                    frame = frames[t]
                    pts = projected[t]
                    lips, eyes = pts[regions["mouth"]], pts[regions["eyes"]]
                    face_size = max(float(np.ptp(eyes[:, 0])), 4.0)
                    centre = lips.mean(0)
                    sil = np.zeros(frame.shape[:2], np.uint8)
                    fade = np.zeros(frame.shape[:2], np.float32)
                    height = 0.0
                    for col, side in enumerate(("left", "right")):
                        ids, faces, fades = sets[side]
                        hand = pts[ids]
                        near = np.linalg.norm(hand.mean(0) - centre) < 1.5 * face_size
                        if not (valid[t, col] and near):
                            continue
                        sil |= silhouette(pts, faces, frame.shape[:2])
                        fade = np.maximum(fade, fade_map(pts, faces, fades, frame.shape[:2], blur=1.0))
                        height = max(height, float(np.ptp(hand, axis=0).max()))
                    core = s <= t < e
                    if sil.sum() < 20:
                        if core:
                            ok = False
                            break
                        crops.append(None)
                        meta.append(None)
                        continue
                    result = constrained_cut(frame, sil, fade, band=max(1, int(round(0.03 * height))))
                    if result is None or result[2] < args.min_iou:
                        if core:
                            ok = False
                            break
                        crops.append(None)
                        meta.append(None)
                        continue
                    rgba, origin, iou = result
                    crops.append(rgba)
                    meta.append({"origin": origin, "centre": centre.tolist(), "face_size": face_size,
                                 "angle": eye_axis(eyes), "iou": round(iou, 3)})
                if not ok:
                    continue
                name = f"{args.corpus}__{signer}__{clip}__e{s:05d}"
                folder = args.out / split / name
                folder.mkdir(parents=True, exist_ok=True)
                for j, rgba in enumerate(crops):
                    if rgba is not None:
                        cv2.imwrite(str(folder / f"f{j:02d}.png"), rgba)
                with np.load(args.occ / f"{clip}.occ.npz") as occ:
                    c_src = np.nan_to_num(occ["c_mnc"][lo:hi]).round(3).tolist()
                (folder / "meta.json").write_text(json.dumps({
                    "corpus": args.corpus, "signer": signer, "clip": clip, "window": [lo, hi],
                    "core_start": s - lo, "length": e - s, "frames": meta, "c_mnc_source": c_src}))
                kept[split] += 1
        print(f"[transplant] {args.corpus} {split}: {kept[split]} episodes kept of {len(items)}")


if __name__ == "__main__":
    main()
