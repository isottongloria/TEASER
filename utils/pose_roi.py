"""Pose-guided face ROI utilities for the ``pose_roi`` face detection mode.

Adapted verbatim from the sibling SMIRK repository
(``/leonardo_work/IscrC_SLPSCALE/smirk/utils/pose_roi.py``, commit d26bd94,
"Add pose_roi face detection mode robust to small/distant faces"), where it
was introduced to fix the same MediaPipe FaceLandmarker failure mode on
small/distant faces (full-body, camera-far shots) also seen in TEASER on the
CSY sample videos. These functions implement the "MediaPipe Pose -> face ROI
-> FaceLandmarker" pipeline. They are intentionally free of any MediaPipe
import so they can be unit tested with simulated detectors; the
MediaPipe-backed adapters that call these live in ``utils/mediapipe_utils.py``.
"""

import numpy as np
from skimage.transform import estimate_transform, warp

# MediaPipe Pose landmark indices that lie on the face (nose, eyes, ears, mouth corners).
FACE_LANDMARK_INDICES = tuple(range(11))  # 0..10
LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
NOSE_INDEX = 0


def face_roi_from_pose(
    landmarks_xy,
    visibility,
    visibility_threshold=0.35,
    min_face_points=4,
    primary_scale=2.5,
    primary_top_shift=0.15,
    fallback_scale=0.9,
    fallback_top_shift=0.05,
):
    """Build a square face ROI (in pixel coordinates) from MediaPipe Pose landmarks.

    Primary strategy: use the visible face landmarks (indices 0-10). The ROI side is
    ``primary_scale`` times the larger of the horizontal/vertical span of those points,
    shifted up by ``primary_top_shift`` of the side length so the forehead is included.

    Fallback strategy (used when fewer than ``min_face_points`` face landmarks are visible):
    if the nose and both shoulders are visible, build a box centered on the nose with side
    ``fallback_scale`` times the shoulder-to-shoulder distance, shifted up by
    ``fallback_top_shift`` of the side length.

    Returns a ``[x1, y1, x2, y2]`` float32 array, or ``None`` if neither strategy applies.
    """
    landmarks_xy = np.asarray(landmarks_xy, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=np.float64)

    face_idx = np.array(FACE_LANDMARK_INDICES)
    face_visible = visibility[face_idx] >= visibility_threshold

    if np.count_nonzero(face_visible) >= min_face_points:
        points = landmarks_xy[face_idx][face_visible]
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        span = max(x2 - x1, y2 - y1, 1e-6)
        side = primary_scale * span
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        cy -= primary_top_shift * side
        return _square_box(cx, cy, side)

    nose_visible = visibility[NOSE_INDEX] >= visibility_threshold
    shoulders_visible = (
        visibility[LEFT_SHOULDER_INDEX] >= visibility_threshold
        and visibility[RIGHT_SHOULDER_INDEX] >= visibility_threshold
    )
    if nose_visible and shoulders_visible:
        shoulder_dist = np.linalg.norm(
            landmarks_xy[LEFT_SHOULDER_INDEX] - landmarks_xy[RIGHT_SHOULDER_INDEX]
        )
        side = fallback_scale * max(shoulder_dist, 1e-6)
        cx, cy = landmarks_xy[NOSE_INDEX]
        cy -= fallback_top_shift * side
        return _square_box(cx, cy, side)

    return None


def _square_box(cx, cy, side):
    half = side / 2.0
    return np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)


def interpolate_boxes(boxes, valid):
    """Fill missing ROI boxes by linear interpolation over frame index.

    ``boxes`` is a ``(T, 4)`` array and ``valid`` a ``(T,)`` boolean mask of frames with a
    directly detected box. Frames before the first / after the last valid detection use the
    nearest valid box (constant extrapolation, matching ``np.interp`` semantics).
    """
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        raise ValueError("No valid pose ROI was detected in any frame")

    boxes = np.asarray(boxes, dtype=np.float32).copy()
    frame_ids = np.arange(len(boxes))
    known = frame_ids[valid]
    for coord in range(4):
        boxes[:, coord] = np.interp(frame_ids, known, boxes[valid, coord])
    return boxes


def roi_crop_transform(box, out_size=512):
    """Similarity transform mapping original-frame coordinates to an ``out_size`` crop."""
    x1, y1, x2, y2 = box
    src_pts = np.array([[x1, y1], [x1, y2], [x2, y1]], dtype=np.float64)
    dst_pts = np.array([[0, 0], [0, out_size - 1], [out_size - 1, 0]], dtype=np.float64)
    return estimate_transform("similarity", src_pts, dst_pts)


def crop_roi(frame, box, out_size=512):
    """Extract the ``box`` region of ``frame`` as an ``out_size x out_size`` crop.

    Regions of ``box`` that fall outside the frame are padded by edge replication
    (equivalent to ``cv2.BORDER_REPLICATE``). Returns ``(crop, tform)`` where ``tform`` maps
    original-frame coordinates to crop coordinates (``tform.inverse`` maps back).
    """
    tform = roi_crop_transform(box, out_size=out_size)
    crop = warp(
        frame,
        tform.inverse,
        output_shape=(out_size, out_size),
        mode="edge",
        preserve_range=True,
    ).astype(frame.dtype)
    return crop, tform


def landmarks_crop_to_frame(landmarks_crop, tform):
    """Map ``(N, 3)`` landmarks in crop pixel coordinates back to the original frame.

    Only x, y are transformed; z is kept as-is (MediaPipe's normalized depth, unaffected by
    the 2D crop transform).
    """
    landmarks_crop = np.asarray(landmarks_crop, dtype=np.float64)
    out = landmarks_crop.copy()
    out[:, :2] = tform.inverse(landmarks_crop[:, :2])
    return out


def interpolate_landmarks(landmarks, detected):
    """Fill frames with no FaceLandmarker detection by linear interpolation over time.

    ``landmarks`` is ``(T, 478, 3)`` and ``detected`` a ``(T,)`` boolean mask. Frames before
    the first / after the last detection use the nearest detected frame.
    """
    detected = np.asarray(detected, dtype=bool)
    if not np.any(detected):
        raise ValueError("FaceLandmarker did not detect a face in any frame")

    landmarks = np.asarray(landmarks, dtype=np.float32).copy()
    n = landmarks.shape[0]
    frame_ids = np.arange(n)
    known = frame_ids[detected]
    flat = landmarks.reshape(n, -1)
    for col in range(flat.shape[1]):
        flat[:, col] = np.interp(frame_ids, known, flat[detected, col])
    return flat.reshape(landmarks.shape)


def extract_pose_roi_landmarks(frames, detect_pose, detect_face, roi_size=512, progress=False):
    """Run the full Pose -> ROI -> FaceLandmarker pipeline over a sequence of frames.

    ``detect_pose(frame)`` must return ``None`` or ``(landmarks_xy, visibility)`` for the 33
    MediaPipe Pose landmarks in pixel coordinates of ``frame``. ``detect_face(crop)`` must
    return ``None`` or a ``(478, 3)`` array of Face Landmarker landmarks in pixel coordinates
    of ``crop``. Both are injected so this function (and the geometry it relies on) can be
    unit tested without MediaPipe.

    Returns a dict with the aligned ``landmarks`` (``(T, 478, 3)``, no NaNs), and the
    ``pose_valid`` / ``face_detected`` boolean masks recording which frames were detected
    directly versus interpolated, plus a ``frame_count``.

    Note: this whole-video convenience wrapper takes an already-decoded list of frames (as
    the original SMIRK caller did) and is kept here for API parity, but TEASER's own video
    script (``scripts/reconstruct_video_3panel.py``) does NOT use it directly -- it re-reads
    the video from disk once per pass instead, to avoid holding every decoded frame in RAM
    at once, and calls the lower-level per-frame functions above itself.
    """
    n = len(frames)
    if n == 0:
        raise ValueError("No frames to process")

    boxes = np.zeros((n, 4), dtype=np.float32)
    pose_valid = np.zeros(n, dtype=bool)

    frame_iter = range(n)
    if progress:
        from tqdm import tqdm

        frame_iter = tqdm(frame_iter, desc="MediaPipe Pose", unit="frame")

    for i in frame_iter:
        result = detect_pose(frames[i])
        if result is None:
            continue
        xy, visibility = result
        box = face_roi_from_pose(xy, visibility)
        if box is not None:
            boxes[i] = box
            pose_valid[i] = True

    if not np.any(pose_valid):
        raise RuntimeError(
            f"MediaPipe Pose did not produce a valid face ROI in any of the {n} frames"
        )

    boxes = interpolate_boxes(boxes, pose_valid)

    landmarks = np.full((n, 478, 3), np.nan, dtype=np.float32)
    face_detected = np.zeros(n, dtype=bool)

    frame_iter = range(n)
    if progress:
        from tqdm import tqdm

        frame_iter = tqdm(frame_iter, desc="MediaPipe FaceLandmarker", unit="frame")

    for i in frame_iter:
        crop, tform = crop_roi(frames[i], boxes[i], out_size=roi_size)
        landmarks_crop = detect_face(crop)
        if landmarks_crop is None:
            continue
        landmarks[i] = landmarks_crop_to_frame(landmarks_crop, tform)
        face_detected[i] = True

    if not np.any(face_detected):
        raise RuntimeError(
            f"FaceLandmarker did not detect a face in any of the {n} pose-derived ROIs"
        )

    landmarks = interpolate_landmarks(landmarks, face_detected)

    return {
        "landmarks": landmarks,
        "pose_valid": pose_valid,
        "face_detected": face_detected,
        "frame_count": n,
        "pose_roi_real": int(np.count_nonzero(pose_valid)),
        "pose_roi_interpolated": int(n - np.count_nonzero(pose_valid)),
        "face_real": int(np.count_nonzero(face_detected)),
        "face_interpolated": int(n - np.count_nonzero(face_detected)),
    }
