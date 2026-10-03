"""Feature cache for temporal TEASER (TEMPORAL_README.md, 4.1).

    python tools_temporal/extract_features.py <work_dir> <out.npz> \
        --checkpoint pretrained_models/TEASER.pt [--temporal_feats expr|expr+pose] [--save_spatial]

Runs TEASER's frozen encoders on ``<work_dir>/frames`` (an RGB2SMPLX work
directory) with the crops of ``rgb2smplx/stages/teaser.py`` and writes one
``.npz`` per clip: the pooled features the adapter trains on, TEASER's own
per-frame outputs (the same keys as ``teaser.npz``), the crop transform and
the landmarks. The expression feature is always stored; pose and shape
feature only when ``--temporal_feats expr+pose`` asks for it.

One difference from the RGB2SMPLX stage, on purpose: MediaPipe Pose's tracker
is reset before each clip, so a clip's first frames never depend on which clip
the same process ran before. For a clip run alone in a fresh process the two
are identical (tests/temporal/check_cache_vs_stage.py).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _md5(path, chunk=1 << 22):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _clip_fps(work_dir):
    meta = Path(work_dir) / "video.json"
    if meta.is_file():
        return float(json.loads(meta.read_text()).get("output_fps", np.nan))
    return float("nan")


def extract_clip(work_dir, encoder, device, feature_set="expr", save_spatial=False,
                 face_detection_mode="pose_roi", crop_scale=1.4, roi_size=512,
                 image_size=224, batch_size=1, frame_cache_mb=3072, progress=True):
    """Returns the dict of arrays written to the cache (see TEMPORAL_README.md 4.1)."""
    from tqdm import tqdm
    from src.temporal import split_encoder as se
    from src.temporal import video_crops as vc

    names = se.FEATURE_SETS[feature_set]
    frames = vc.Frames(vc.list_frames(Path(work_dir) / "frames"), int(frame_cache_mb) << 20)
    n = len(frames)

    vc.reset_pose_tracker()
    if face_detection_mode == "pose_roi":
        landmarks, face_detected, pose_valid = vc.pose_roi_landmarks(frames, roi_size, progress)
        have = np.ones(n, dtype=bool)
    else:
        landmarks, face_detected, pose_valid = vc.direct_landmarks(frames, progress)
        have = face_detected.copy()

    out = {
        "frame_index": np.arange(n, dtype=np.int64),
        "valid": np.zeros(n, dtype=bool),
        "pose_params": np.zeros((n, 3), np.float32),
        "jaw_pose": np.zeros((n, 3), np.float32),
        "expression": np.zeros((n, encoder.expression_encoder.n_exp), np.float32),
        "eyelid": np.zeros((n, 2), np.float32),
        "cam": np.zeros((n, 3), np.float32),
        "shape_params": np.zeros((n, 300), np.float32),
        "tform": np.full((n, 3, 3), np.nan, np.float32),
    }
    for name in names:
        out["feat_" + name] = np.zeros((n, se.FEATURE_DIMS[name]), np.float16)
        if save_spatial:
            out[f"feat_{name}_map"] = np.zeros((n, se.FEATURE_DIMS[name], 7, 7), np.float16)
    # Pose and shape features are always computed (the heads need them for
    # TEASER's own outputs); they are only stored when asked for.
    run_names = ("expr", "pose", "shape")

    pending, tensors = [], []

    def flush():
        if not pending:
            return
        batch = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        feats = se.extract_features(encoder, batch.to(device), run_names, spatial=save_spatial)
        heads = se.apply_heads(encoder, feats)
        for pos, i in enumerate(pending):
            out["valid"][i] = True
            out["pose_params"][i] = heads["pose_params"][pos].cpu().numpy()
            out["jaw_pose"][i] = heads["jaw_params"][pos].cpu().numpy()
            out["expression"][i] = heads["expression_params"][pos].cpu().numpy()
            out["eyelid"][i] = heads["eyelid_params"][pos].cpu().numpy()
            out["cam"][i] = heads["cam"][pos].cpu().numpy()
            out["shape_params"][i] = heads["shape_params"][pos].cpu().numpy()
            for name in names:
                out["feat_" + name][i] = feats[name][pos].cpu().numpy().astype(np.float16)
                if save_spatial:
                    out[f"feat_{name}_map"][i] = feats[name + "_map"][pos].cpu().numpy().astype(np.float16)
        pending.clear()
        tensors.clear()

    with torch.no_grad():
        for i in tqdm(range(n), desc="TEASER features", unit="frame", disable=not progress):
            if not have[i]:
                continue
            tform = vc.crop_face_transform(landmarks[i, :, :2], crop_scale, image_size)
            out["tform"][i] = tform.params.astype(np.float32)
            tensors.append(vc.crop_tensor(frames[i], tform, image_size))
            pending.append(i)
            if len(pending) >= batch_size:
                flush()
        flush()

    out.update(
        landmarks=landmarks[:, :, :2].astype(np.float32),
        face_detected=face_detected,
        pose_valid=pose_valid,
        fps=np.float32(_clip_fps(work_dir)),
        crop_scale=np.float32(crop_scale),
        feature_set=np.array(feature_set),
        face_detection_mode=np.array(face_detection_mode),
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "pretrained_models/TEASER.pt")
    parser.add_argument("--temporal_feats", choices=("expr", "expr+pose"), default="expr")
    parser.add_argument("--save_spatial", action="store_true",
                        help="also store the pre-pool 7x7 maps (fp16, ~94 KB/frame for expr)")
    parser.add_argument("--face-detection-mode", choices=("pose_roi", "direct"), default="pose_roi")
    parser.add_argument("--crop-scale", type=float, default=1.4)
    parser.add_argument("--roi-size", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="1 (default) reproduces teaser.npz bit for bit; larger is faster "
                             "but moves the outputs by ~5e-3 (cuDNN kernel choice)")
    parser.add_argument("--frame-cache-mb", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    output = args.output.resolve()
    checkpoint = args.checkpoint.resolve()
    os.chdir(REPO_ROOT)  # mediapipe_utils loads assets/ by a relative path at import
    sys.path.insert(0, str(REPO_ROOT))
    from src.temporal.split_encoder import load_teaser_encoder

    started = time.time()
    encoder = load_teaser_encoder(checkpoint, args.device)
    arrays = extract_clip(work_dir, encoder, args.device, args.temporal_feats, args.save_spatial,
                          args.face_detection_mode, args.crop_scale, args.roi_size,
                          args.image_size, args.batch_size, args.frame_cache_mb)
    arrays["checkpoint_md5"] = np.array(_md5(checkpoint))

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, output)
    n = len(arrays["valid"])
    print(f"[extract_features] {int(arrays['valid'].sum())}/{n} frames, "
          f"{int(arrays['face_detected'].sum())} with a direct face detection, "
          f"{time.time() - started:.0f} s -> {output}")


if __name__ == "__main__":
    main()
