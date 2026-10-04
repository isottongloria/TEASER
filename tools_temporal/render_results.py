"""Visual results of a trained model on chosen clips, for the review page (TEASER env, GPU preferred).

    python tools_temporal/render_results.py --ckpt runs/T7_replay_long/best.pt --data data/temporal \
        --split train --out results.json

Per corpus, two examples from clips of ``--split``:
- **synthetic**: a replay variant's episode (hand pasted on a clean clip); the
  target exists (TEASER on the unoccluded frames);
- **real**: the longest real hand-over-face episode (c_mnc > 0.2) in the clips.
  In training clips these frames lie outside the clean segments, so they were
  never a training target; there is no ground truth, only plausibility.

For every frame of the window (episode +-4 frames): the face crop TEASER saw,
and the FLAME face (canonical: the clip's mean identity, head pose zeroed, same
camera for all) rendered for TEASER per-frame (T0), SG9 + interpolation (T2,
the current pipeline), the model, and the target when there is one. Also the
mouth opening per frame (vertical extent of the lips, mm) over a longer stretch
of the clip, per method.
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def b64(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data/temporal")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    ckpt, data, out = args.ckpt.resolve(), args.data.resolve(), args.out.resolve()
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    from skimage.transform import SimilarityTransform, warp
    from src.FLAME.FLAME import FLAME
    from src.renderer.renderer import Renderer
    from src.temporal import video_crops as vc
    from src.temporal.data import Clip, VariantClip, full_clip
    from src.temporal.flame_regions import region_vertex_ids
    from tools_temporal.eval_temporal import load_checkpoint, postproc_method
    from tools_temporal.train_temporal import to_device

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_checkpoint(ckpt, device)
    flame, renderer = FLAME().to(device), Renderer(render_full_head=False).to(device)
    lips = torch.as_tensor(region_vertex_ids()["mouth"], device=device)
    rng = np.random.default_rng(args.seed)

    @torch.no_grad()
    def faces(params, shape, frames_idx):
        """Rendered canonical faces (BGR uint8) and the mouth opening (mm) for the given frames."""
        n = len(frames_idx)
        t = lambda k: torch.as_tensor(np.asarray(params[k])[frames_idx], dtype=torch.float32, device=device)
        out = flame.forward({"shape_params": torch.as_tensor(shape, dtype=torch.float32, device=device)[None].expand(n, -1),
                             "expression_params": t("expression"), "jaw_params": t("jaw"),
                             "eyelid_params": t("eyelid"), "pose_params": torch.zeros(n, 3, device=device)})
        v = out["vertices"]
        opening = (v[:, lips, 1].max(1).values - v[:, lips, 1].min(1).values) * 1000.0
        cam = torch.tensor([[8.0, 0.0, 0.0]], device=device).expand(n, -1)
        img = renderer.forward(v, cam)["rendered_img"]
        imgs = [(im.permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).astype(np.uint8) for im in img]
        return [cv2.resize(i, (112, 112)) for i in imgs], opening.cpu().numpy().round(2).tolist()

    def predict(clip):
        pred = model.infer_clip(to_device(full_clip(clip), device), window=cfg.train.window,
                                stride=cfg.train.infer_stride)
        return {k: v[0].cpu().numpy() for k, v in pred.items()}

    def crop(frames, tform, t):
        img = frames[t]
        return cv2.resize(warp(img, SimilarityTransform(matrix=tform[t].astype(float)).inverse,
                               output_shape=(224, 224), preserve_range=True).astype(np.uint8), (112, 112))

    results = []
    for corpus in ("phoenix", "csl_daily", "how2sign"):
        root = data / corpus
        names = [l.split("\t")[0] for l in (root / "lists" / f"{args.split}.txt").read_text().splitlines() if l.strip()]
        segments = json.loads((root / "lists" / "segments.json").read_text())

        # Synthetic example: a replay variant with a 5-10 frame episode of high coverage.
        order = rng.permutation(len(names))
        for i in order:
            name = names[i]
            paths = sorted((root / "variants_replay").glob(f"{name}.v*.npz"))
            if not paths:
                continue
            clean = Clip(root / "cache" / f"{name}.npz", root / "occ" / f"{name}.occ.npz", cfg.data.feature_set,
                         segments=segments.get(name))
            var = VariantClip(paths[0], clean, cfg.data.feature_set)
            eps = [e for e in var.episodes if 5 <= e["length"] <= 10 and e["coverage_mnc"] > 0.5]
            if not eps:
                continue
            ep = eps[0]
            lo = max(0, ep["start"] - ep["entry"] - 2)
            hi = min(len(var), ep["start"] + ep["length"] + ep["exit"] + 2)
            frames_occ = None
            from src.temporal import occlusion_aug as oa
            split_bank = "train" if args.split == "train" else "heldout"
            hands = [l.strip() for l in (root / "lists" / f"hands_{split_bank}.txt").read_text().splitlines() if l.strip()]
            frames = vc.Frames(vc.list_frames(root / "clips" / name / "frames"), 1 << 30)
            occ = oa.Occluder(frames, oa.FaceRegions.from_fit(root / "regions" / f"{name}.regions.npz"),
                              var.episodes, oa.HandBank(str(data / "hand_bank" / corpus / split_bank), hands))
            for t in range(hi):
                if occ.syn_mask[t]:
                    occ[t]
            with np.load(paths[0]) as z:
                tform = z["tform"]
                shape = z["shape_params"][z["valid"]].mean(0)
            idx = list(range(lo, hi))
            methods = {"TEASER (T0)": var.input_params, "SG9 + interp (T2)": postproc_method("sg9+interp", var),
                       "Temporal (T7)": predict(var), "Target": var.teacher}
            rows = {}
            curve_idx = list(range(max(0, lo - 12), min(len(var), hi + 12)))
            curves = {}
            for label, p in methods.items():
                rows[label], _ = faces(p, shape, idx)
                _, curves[label] = faces(p, shape, curve_idx)
            results.append({
                "corpus": corpus, "kind": "synthetic", "clip": name, "variant": paths[0].name[:-4],
                "frames": idx, "core": [ep["start"], ep["start"] + ep["length"]],
                "crops": [b64(crop(occ, tform, t)) for t in idx],
                "renders": {k: [b64(i) for i in v] for k, v in rows.items()},
                "curve_frames": curve_idx, "curves": curves,
                "pasted": [bool(occ.syn_mask[t]) for t in curve_idx]})
            break

        # Real example: the longest real episode (outside the clean segments) in these clips.
        best = None
        for name in names:
            with np.load(root / "occ" / f"{name}.occ.npz") as z:
                c = np.nan_to_num(z["c_mnc"], nan=0.0)
            flags = c > 0.2
            t = 0
            while t < len(flags):
                if flags[t]:
                    e = t
                    while e < len(flags) and flags[e]:
                        e += 1
                    if best is None or (e - t) > best[2] - best[1]:
                        best = (name, t, e)
                    t = e
                else:
                    t += 1
        if best:
            name, s, e = best
            clip = Clip(root / "cache" / f"{name}.npz", root / "occ" / f"{name}.occ.npz", cfg.data.feature_set,
                        segments=segments.get(name))
            with np.load(root / "cache" / f"{name}.npz") as z:
                tform = z["tform"]
                shape = z["shape_params"][z["valid"]].mean(0)
            frames = vc.Frames(vc.list_frames(root / "clips" / name / "frames"), 1 << 30)
            lo, hi = max(0, s - 4), min(len(clip), e + 4)
            idx = list(range(lo, hi))
            methods = {"TEASER (T0)": clip.input_params, "SG9 + interp (T2)": postproc_method("sg9+interp", clip),
                       "Temporal (T7)": predict(clip)}
            rows, curves = {}, {}
            curve_idx = list(range(max(0, lo - 12), min(len(clip), hi + 12)))
            for label, p in methods.items():
                rows[label], _ = faces(p, shape, idx)
                _, curves[label] = faces(p, shape, curve_idx)
            results.append({
                "corpus": corpus, "kind": "real", "clip": name, "frames": idx, "core": [s, e],
                "c_mnc": [round(float(x), 2) for x in clip.c_mnc[lo:hi]],
                "crops": [b64(crop(frames, tform, t)) for t in idx],
                "renders": {k: [b64(i) for i in v] for k, v in rows.items()},
                "curve_frames": curve_idx, "curves": curves,
                "pasted": [bool(clip.c_mnc[t] > 0.2) for t in curve_idx]})
        print(f"[results] {corpus}: {sum(r['corpus'] == corpus for r in results)} examples")
    out.write_text(json.dumps({"checkpoint": str(ckpt), "split": args.split, "examples": results}))
    print(f"[results] -> {out}")


if __name__ == "__main__":
    main()
