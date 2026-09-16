import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

from utils.pose_roi import crop_roi, face_roi_from_pose, landmarks_crop_to_frame

base_options = python.BaseOptions(model_asset_path='assets/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                    output_face_blendshapes=True,
                                    output_facial_transformation_matrixes=True,
                                    num_faces=1,
                                    min_face_detection_confidence=0.1,
                                    min_face_presence_confidence=0.1
                                    )
detector = vision.FaceLandmarker.create_from_options(options)

# Lazily-created MediaPipe Pose detectors, keyed by static_image_mode. Only
# instantiated the first time `face_detection_mode='pose_roi'` is actually
# used, so the 'direct' path (the original TEASER behavior) is unaffected.
_pose_cache = {}


def _detect_face_landmarks(image_bgr):
    """Run TEASER's own FaceLandmarker (`detector` above, same model/weights
    used everywhere else) on a BGR image. Returns None or a (478, 3) ndarray
    of landmarks in `image_bgr`'s own pixel coordinates.
    """
    image_numpy = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_numpy)
    detection_result = detector.detect(mp_image)

    if len(detection_result.face_landmarks) == 0:
        return None

    face_landmarks = detection_result.face_landmarks[0]
    face_landmarks_numpy = np.zeros((478, 3))
    for i, landmark in enumerate(face_landmarks):
        face_landmarks_numpy[i] = [landmark.x * mp_image.width, landmark.y * mp_image.height, landmark.z]
    return face_landmarks_numpy


def get_pose_detector(static_image_mode=False):
    """Return a cached `mp.solutions.pose.Pose` instance (used only by the
    'pose_roi' face detection mode, to localize a face ROI on frames where
    the face is too small/distant for direct FaceLandmarker detection).
    """
    if static_image_mode not in _pose_cache:
        _pose_cache[static_image_mode] = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=2,
            smooth_landmarks=not static_image_mode,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _pose_cache[static_image_mode]


def detect_pose(image_bgr, static_image_mode=False):
    """Run MediaPipe Pose on a BGR image. Returns None or (xy(33,2), visibility(33,))
    in `image_bgr`'s own pixel coordinates.
    """
    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = get_pose_detector(static_image_mode=static_image_mode).process(image_rgb)
    if result.pose_landmarks is None:
        return None
    landmarks = result.pose_landmarks.landmark
    xy = np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float64)
    visibility = np.array([lm.visibility for lm in landmarks], dtype=np.float64)
    return xy, visibility


def run_mediapipe(image, face_detection_mode='direct'):
    """Detect the 478 TEASER face landmarks in a single BGR image.

    face_detection_mode:
      - 'direct' (default, original TEASER behavior): run FaceLandmarker on the
        full frame. Fails when the face occupies too few pixels (e.g. a
        full-body, camera-far shot).
      - 'pose_roi': locate the face with MediaPipe Pose first, then run the
        exact same FaceLandmarker on a zoomed-in crop around it. More robust
        to small/distant faces; see utils/pose_roi.py.
    """
    if face_detection_mode == 'direct':
        landmarks = _detect_face_landmarks(image)
        if landmarks is None:
            print('No face detected')
        return landmarks

    if face_detection_mode == 'pose_roi':
        return run_mediapipe_pose_roi_image(image)

    raise ValueError(f"Unknown face_detection_mode: {face_detection_mode!r}")


def run_mediapipe_pose_roi_image(image, roi_size=512, static_image_mode=True):
    """Single-image variant of the pose_roi pipeline: MediaPipe Pose -> face ROI
    -> TEASER's own FaceLandmarker on the zoomed-in crop, mapped back to the
    original image's coordinates.
    """
    pose_result = detect_pose(image, static_image_mode=static_image_mode)
    if pose_result is None:
        print('No person detected by MediaPipe Pose')
        return None

    xy, visibility = pose_result
    box = face_roi_from_pose(xy, visibility)
    if box is None:
        print('MediaPipe Pose could not localize a face ROI')
        return None

    crop, tform = crop_roi(image, box, out_size=roi_size)
    landmarks_crop = _detect_face_landmarks(crop)
    if landmarks_crop is None:
        print('No face detected')
        return None

    return landmarks_crop_to_frame(landmarks_crop, tform)
