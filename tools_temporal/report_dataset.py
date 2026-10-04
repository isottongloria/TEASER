"""Collect the pilot dataset's statistics and example images into one JSON (for the review page).

    python tools_temporal/report_dataset.py --data data/temporal --out report.json

Per corpus: selection outcome, splits (clips, signers, frames, clean frames),
real occlusion statistics of the training signers, synthetic-variant
statistics (episode durations, coverage, frames with a pasted hand), the hand
bank (counts, sizes, a sample of cut-outs) and two example episodes as frame
strips (clean frame, frame with the pasted hand, the crop TEASER sees).
Images are embedded as base64 JPEG/PNG.
"""

import argparse
import base64
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPORA = ("phoenix", "csl_daily", "how2sign")


def b64(img, ext=".jpg"):
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 85] if ext == ".jpg" else [])
    return f"data:image/{'jpeg' if ext == '.jpg' else 'png'};base64," + base64.b64encode(buf).decode()


def lines(path):
    return [l.split("\t") for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).is_file() else []


def hand_tile(path, height=96):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    width = max(8, int(im.shape[1] * height / im.shape[0]))
    im = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
    return b64(im, ".png")


def corpus_report(data, corpus, rng):
    from skimage.transform import SimilarityTransform, warp
    from src.temporal import occlusion_aug as oa
    from src.temporal import video_crops as vc

    root = data / corpus
    lists = root / "lists"
    sel = json.loads((lists / ("selection_report.json" if corpus != "how2sign" else "eligibility/selection_report.json")).read_text())
    segments = json.loads((lists / "segments.json").read_text())
    out = {"selection": {"candidates": sel["candidates"], "outcome": sel["outcome"]}, "splits": {}}
    for split in ("train", "val", "test"):
        rows = lines(lists / f"{split}.txt")
        frames = clean = 0
        for clip, _ in rows:
            with np.load(root / "cache" / f"{clip}.npz") as c:
                frames += len(c["valid"])
            clean += sum(e - s for s, e in segments.get(clip, []))
        out["splits"][split] = {"clips": len(rows), "signers": dict(sorted(Counter(s for _, s in rows).items())),
                                "frames": frames, "clean_frames": clean}

    stats = json.loads((lists / "occlusion_stats_train.json").read_text())
    out["real"] = {"clips": stats["clips"], "episodes": stats["episodes"],
                   "occluded_fraction": stats["occluded_fraction"],
                   "durations": stats["durations"], "fraction_reaching_eyes": stats["coverage"]["fraction_reaching_eyes"],
                   "peak_mouth": stats["coverage"]["peak_mouth"],
                   "peak_mouth_values": stats["coverage"]["peak_mouth_values"]}

    syn_dur, syn_cov, syn_frames, total_frames, n_var, regions = [], [], 0, 0, 0, Counter()
    variants = sorted((root / "variants").glob("*.npz"))
    for path in variants:
        with np.load(path) as v:
            plan = json.loads(str(v["plan"]))
            mask, c_syn = v["syn_mask"], v["c_syn"]
        n_var += 1
        syn_frames += int(mask.sum())
        total_frames += len(mask)
        for ep in plan:
            syn_dur.append(ep["length"])
            regions[ep["region"]] += 1
            sl = slice(ep["start"], ep["start"] + ep["length"])
            col = 0 if ep["region"] == "mouth" else 1
            syn_cov.append(float(c_syn[sl, col].max()) if c_syn[sl].size else 0.0)
    out["synthetic"] = {"variants": n_var, "episodes": len(syn_dur), "frames_with_hand": syn_frames,
                        "frames": total_frames, "durations": dict(sorted(Counter(syn_dur).items())),
                        "duration_mean": float(np.mean(syn_dur)) if syn_dur else 0.0,
                        "duration_median": float(np.median(syn_dur)) if syn_dur else 0.0,
                        "coverage_values": [round(x, 3) for x in syn_cov], "regions": dict(regions)}

    bank = {}
    for split in ("train", "heldout"):
        names = [l[0] for l in lines(lists / f"hands_{split}.txt")]
        files = [data / "hand_bank" / split / f"{n}.png" for n in names]
        heights = [cv2.imread(str(f), cv2.IMREAD_UNCHANGED).shape[0] for f in files[:400]]
        pick = [files[i] for i in rng.choice(len(files), min(12, len(files)), replace=False)] if files else []
        bank[split] = {"count": len(files), "signers": len({n.split("__")[1] for n in names}),
                       "height_median": float(np.median(heights)) if heights else 0.0,
                       "samples": [hand_tile(f) for f in pick]}
    out["hands"] = bank

    # Two example episodes: one from a training variant, one from a test variant.
    examples = []
    for split in ("train", "test"):
        names = {c for c, _ in lines(lists / f"{split}.txt")}
        cands = [p for p in variants if p.name.split(".v")[0] in names]
        rng.shuffle(cands)
        for path in cands:
            clip = path.name.split(".v")[0]
            with np.load(path) as v:
                plan = json.loads(str(v["plan"]))
                tform_occ = v["tform"]
                c_syn = v["c_syn"]
            good = [ep for ep in plan if ep["region"] == "mouth" and 3 <= ep["length"] <= 8
                    and c_syn[ep["start"]:ep["start"] + ep["length"], 0].max() >= 0.5]
            if not good:
                continue
            ep = good[0]
            with np.load(root / "cache" / f"{clip}.npz") as c:
                landmarks = c["landmarks"].astype(np.float64)
                tform_clean = c["tform"]
            frames = vc.Frames(vc.list_frames(root / "clips" / clip / "frames"), 1 << 30)
            hands = oa.HandBank(str(data / "hand_bank" / ("train" if split == "train" else "heldout")),
                                [l[0] for l in lines(lists / f"hands_{'train' if split == 'train' else 'heldout'}.txt")])
            regions_path = root / "regions" / f"{clip}.regions.npz"
            regions = oa.FaceRegions.from_fit(regions_path) if regions_path.is_file() \
                else oa.FaceRegions.from_landmarks(landmarks)
            occ = oa.Occluder(frames, regions, plan, hands)
            lo = max(0, ep["start"] - ep.get("entry", 0) - 1)
            hi = min(len(frames), ep["start"] + ep["length"] + ep.get("exit", 0) + 1)
            crop = lambda img, T: warp(img, SimilarityTransform(matrix=T.astype(float)).inverse,
                                       output_shape=(224, 224), preserve_range=True).astype(np.uint8)
            strip = []
            for t in range(lo, hi):
                clean_img, occ_img = frames[t], occ[t]
                h = 200
                fit = lambda im: cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_AREA)
                strip.append({"t": t, "in_episode": bool(occ.core_mask[t]), "pasted": bool(occ.syn_mask[t]),
                              "coverage": round(float(occ.c_syn[t, 0]), 2),
                              "clean": b64(fit(clean_img)), "occluded": b64(fit(occ_img)),
                              "crop_occ": b64(cv2.resize(crop(occ_img, tform_occ[t]), (128, 128))),
                              "crop_clean": b64(cv2.resize(crop(clean_img, tform_clean[t]), (128, 128)))})
            examples.append({"split": split, "clip": clip, "variant": path.name[:-4], "hand": ep["hand"],
                             "length": ep["length"], "strip": strip})
            break
    out["examples"] = examples
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data/temporal")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    rng = np.random.default_rng(args.seed)
    report = {c: corpus_report(args.data.resolve(), c, rng) for c in CORPORA if (args.data / c / "variants").is_dir()}
    args.out.write_text(json.dumps(report))
    for c, r in report.items():
        print(c, {s: (v["clips"], v["clean_frames"]) for s, v in r["splits"].items()},
              "variants", r["synthetic"]["variants"], "hands", {k: v["count"] for k, v in r["hands"].items()})


if __name__ == "__main__":
    main()
