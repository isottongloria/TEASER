"""Replay bank: the dynamics of real hand-over-face occlusions, without their pixels (TEMPORAL_README.md 5.4b).

Runs in RGB2SMPLX's env:

    PYTHONPATH=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX ~/miniconda3/envs/rgb2smplx/bin/python \
        tools_temporal/build_replay_bank.py --corpus phoenix --occ data/temporal/phoenix/occ \
        --fits_of fits.tsv --signers candidates.tsv \
        --split train=Signer01,Signer03 --split heldout=Signer04,Signer07 --out data/temporal/replay_bank

Cutting the real occluding hand out of the frame does not work: in front of
the face the fitted hand mesh is not accurate enough to separate hand skin
from face skin, and the cut-outs carry the source signer's lips, nose or eye
(tools_temporal/build_transplant_bank.py, kept for reference). So only the
*dynamics* are taken from real episodes, from the fit and the occlusion file,
no frames needed:

per frame of the episode window (the episode -- ``c_mnc > 0.2``, 1-20
frames -- plus ``--context`` frames of approach and retreat on each side),
for the occluding hand (the WiLoR-observed hand closest to the lips):
- ``rel``: the hand's centroid relative to the lips centre, in the eye
  axis's frame, divided by the face size (eye-region width);
- ``height``: the hand's projected size / face size;
- ``angle``: the wrist-to-hand direction relative to the eye axis (degrees,
  image coordinates, y down), so fingers up and fingers down differ;
- ``side``; and the source's lip coverage ``c_mouth`` (and ``c_mnc``).
Per episode: the MANO pose of that hand averaged over the core, for picking a
bank hand of similar shape.

Writes ``<out>/<split>/<corpus>.jsonl``, one episode per line.
"""

import argparse
import json
import sys
from pathlib import Path

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


def principal_angle(points):
    centred = points - points.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0])))


def fold(angle):
    return (angle + 90) % 180 - 90


def rotate(v, degrees):
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--occ", type=Path, required=True)
    parser.add_argument("--fits_of", type=Path, required=True)
    parser.add_argument("--signers", type=Path, required=True)
    parser.add_argument("--split", action="append", required=True, help="NAME=signer,signer,...")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=20)
    args = parser.parse_args()

    sys.path.insert(0, str(RGB2SMPLX / "experiments/occlusion_protocols_smplx"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import smplx_geometry as geometry
    from compute_real_occlusion import fit_for_geometry, flame_region_ids_in_smplx

    fit_of = dict(l.split("\t")[:2] for l in args.fits_of.read_text().splitlines() if l.strip())
    signer_of = dict(l.split("\t")[:2] for l in args.signers.read_text().splitlines() if l.strip())
    split_of = {}
    for spec in args.split:
        name, signers = spec.split("=", 1)
        for signer in signers.split(","):
            split_of[signer] = name
    regions = flame_region_ids_in_smplx()
    hands = geometry.vertex_groups()
    # Vertices driven by each wrist joint: their centroid stands in for the wrist.
    weights = geometry._model().lbs_weights.argmax(1).numpy()
    wrist_ids = {"left": np.where(weights == 20)[0], "right": np.where(weights == 21)[0]}
    counts = {}
    writers = {}
    for clip, signer in sorted(signer_of.items()):
        split = split_of.get(signer)
        occ_path = args.occ / f"{clip}.occ.npz"
        if split is None or clip not in fit_of or not occ_path.is_file():
            continue
        with np.load(occ_path) as occ:
            c_mnc = np.nan_to_num(occ["c_mnc"], nan=0.0)
            c_mouth = np.nan_to_num(occ["c_mouth"], nan=0.0)
        episodes = [(s, e) for s, e in runs(c_mnc > 0.2)
                    if e - s <= args.max_len and s - args.context >= 0 and e + args.context <= len(c_mnc)]
        if not episodes:
            continue
        npz = fit_for_geometry(dict(np.load(fit_of[clip])), geometry)
        projected = geometry.project_all_frames(geometry.forward_vertices(npz), npz)
        valid = np.asarray(npz.get("wilor_valid", np.ones((len(projected), 2), bool)), bool)
        for s, e in episodes:
            lo, hi = s - args.context, e + args.context
            # The occluding hand: observed, and closest to the lips over the core.
            dists = []
            for col, side in enumerate(("left", "right")):
                ids = hands[f"{side}_hand"]
                d = [np.linalg.norm(projected[t, ids].mean(0) - projected[t, regions["mouth"]].mean(0))
                     if valid[t, col] else np.inf for t in range(s, e)]
                dists.append(np.mean(d))
            col = int(np.argmin(dists))
            if not np.isfinite(dists[col]):
                continue
            side = ("left", "right")[col]
            ids = hands[f"{side}_hand"]
            frames = []
            for t in range(lo, hi):
                pts = projected[t]
                lips, eyes, hand = pts[regions["mouth"]], pts[regions["eyes"]], pts[ids]
                face_size = max(float(np.ptp(eyes[:, 0])), 4.0)
                eye_angle = fold(principal_angle(eyes))
                rel = rotate(hand.mean(0) - lips.mean(0), -eye_angle) / face_size
                pointing = hand.mean(0) - pts[wrist_ids[side]].mean(0)   # wrist -> hand
                frames.append({"rel": [round(float(x), 4) for x in rel],
                               "height": round(float(np.ptp(hand, axis=0).max()) / face_size, 4),
                               "angle": round(float(np.degrees(np.arctan2(pointing[1], pointing[0]))) - eye_angle, 2),
                               "observed": bool(valid[t, col]),
                               "c_mouth": round(float(c_mouth[t]), 3), "c_mnc": round(float(c_mnc[t]), 3)})
            pose_key = "smplx_lhand_pose" if side == "left" else "smplx_rhand_pose"
            pose = np.asarray(npz[pose_key][s:e], np.float32).reshape(e - s, -1).mean(0)
            record = {"corpus": args.corpus, "signer": signer, "clip": clip, "window": [lo, hi],
                      "core_start": s - lo, "length": e - s, "side": side,
                      "pose": [round(float(x), 4) for x in pose], "frames": frames}
            if split not in writers:
                (args.out / split).mkdir(parents=True, exist_ok=True)
                writers[split] = open(args.out / split / f"{args.corpus}.jsonl", "w")
            writers[split].write(json.dumps(record) + "\n")
            counts[split] = counts.get(split, 0) + 1
    for w in writers.values():
        w.close()
    print(f"[replay] {args.corpus}: {counts}")


if __name__ == "__main__":
    main()
