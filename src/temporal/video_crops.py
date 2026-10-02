"""Face crops for a clip, exactly as RGB2SMPLX's ``rgb2smplx/stages/teaser.py`` makes them.

The feature cache must hold the same per-frame TEASER outputs as the
``teaser.npz`` the RGB2SMPLX pipeline writes, so this follows that stage step
for step: frames from ``<work_dir>/frames`` in sorted order, MediaPipe Pose
(tracking mode) -> face ROI -> FaceLandmarker on the 512 ROI crop, boxes and
landmarks interpolated over time, TEASER's own ``crop_face`` similarity at
scale 1.4 to 224, and the input tensor built the same way (from_numpy on the
HWC RGB crop, permute, unsqueeze, /255 -- a different memory layout makes
cuDNN pick other kernels and moves the outputs by ~6e-3).

It is a copy rather than an import because RGB2SMPLX is the parent repo of
this submodule; ``tests/temporal/check_cache_vs_stage.py`` runs both on the
same clip and compares.

MediaPipe is imported lazily: ``utils/mediapipe_utils.py`` loads
``assets/face_landmarker.task`` by a path relative to the working directory at
import time, so the caller must import from the repo root (the tools do
``os.chdir`` to it).
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.transform import estimate_transform, warp

FRAME_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def list_frames(frames_dir):
    frames_dir = Path(frames_dir)
    paths = sorted(p for p in frames_dir.iterdir()
                   if p.suffix.lower() in FRAME_SUFFIXES) if frames_dir.is_dir() else []
    if not paths:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    return paths


class Frames:
    """Decode each frame once while a byte budget lasts, re-read after (same pixels)."""

    def __init__(self, paths, budget_bytes):
        self.paths = list(paths)
        self._cache = {}
        self._budget = budget_bytes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        frame = cv2.imread(str(self.paths[index]))
        if frame is None:
            raise OSError(f"Could not read frame: {self.paths[index]}")
        if self._budget >= frame.nbytes:
            self._cache[index] = frame
            self._budget -= frame.nbytes
        return frame


def crop_face_transform(landmarks_xy, scale=1.4, image_size=224):
    """TEASER's ``crop_face`` (main/demo_video.py): frame px -> crop px similarity."""
    left = np.min(landmarks_xy[:, 0])
    right = np.max(landmarks_xy[:, 0])
    top = np.min(landmarks_xy[:, 1])
    bottom = np.max(landmarks_xy[:, 1])

    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)

    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2],
                        [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    dst_pts = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    return estimate_transform("similarity", src_pts, dst_pts)


def crop_tensor(frame_bgr, tform, image_size=224):
    """(1, 3, S, S) float tensor in [0, 1], built as the RGB2SMPLX stage builds it."""
    cropped_bgr = warp(frame_bgr, tform.inverse, output_shape=(image_size, image_size),
                       preserve_range=True).astype(np.uint8)
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(cropped_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def reset_pose_tracker():
    """Forget MediaPipe Pose's tracking state, so one clip never seeds the next.

    ``utils.mediapipe_utils`` caches one tracking-mode detector per process;
    without a reset, the first frames of a clip are tracked from the last
    frame of the previous clip in the same process.
    """
    from utils import mediapipe_utils
    for detector in mediapipe_utils._pose_cache.values():
        detector.close()
    mediapipe_utils._pose_cache.clear()


def pose_roi_landmarks(frames, roi_size=512, progress=True):
    """(landmarks (T,478,3), face_detected (T,), pose_valid (T,)) in frame pixels."""
    from tqdm import tqdm
    from utils.mediapipe_utils import detect_pose, run_mediapipe
    from utils.pose_roi import (crop_roi, face_roi_from_pose, interpolate_boxes,
                                interpolate_landmarks, landmarks_crop_to_frame)

    n = len(frames)
    boxes = np.zeros((n, 4), dtype=np.float32)
    pose_valid = np.zeros(n, dtype=bool)
    for i in tqdm(range(n), desc="MediaPipe Pose", unit="frame", disable=not progress):
        result = detect_pose(frames[i], static_image_mode=False)
        if result is None:
            continue
        box = face_roi_from_pose(*result)
        if box is not None:
            boxes[i] = box
            pose_valid[i] = True
    if not np.any(pose_valid):
        raise RuntimeError(f"MediaPipe Pose did not localize a face ROI in any of {n} frames")
    boxes = interpolate_boxes(boxes, pose_valid)

    landmarks = np.full((n, 478, 3), np.nan, dtype=np.float32)
    face_detected = np.zeros(n, dtype=bool)
    for i in tqdm(range(n), desc="FaceLandmarker (ROI)", unit="frame", disable=not progress):
        crop, tform = crop_roi(frames[i], boxes[i], out_size=roi_size)
        landmarks_crop = run_mediapipe(crop, face_detection_mode="direct")
        if landmarks_crop is None:
            continue
        landmarks[i] = landmarks_crop_to_frame(landmarks_crop, tform)
        face_detected[i] = True
    if not np.any(face_detected):
        raise RuntimeError(f"FaceLandmarker detected no face in any of {n} pose-derived ROIs")
    return interpolate_landmarks(landmarks, face_detected), face_detected, pose_valid


def direct_landmarks(frames, progress=True):
    """TEASER's original full-frame FaceLandmarker; undetected frames stay NaN."""
    from tqdm import tqdm
    from utils.mediapipe_utils import run_mediapipe

    n = len(frames)
    landmarks = np.full((n, 478, 3), np.nan, dtype=np.float32)
    face_detected = np.zeros(n, dtype=bool)
    for i in tqdm(range(n), desc="FaceLandmarker (direct)", unit="frame", disable=not progress):
        result = run_mediapipe(frames[i], face_detection_mode="direct")
        if result is not None:
            landmarks[i] = result
            face_detected[i] = True
    return landmarks, face_detected, face_detected.copy()
