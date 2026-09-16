"""
Three-panel TEASER visual-evaluation video.

For a given input video, produces ONE output MP4 with three synchronized
panels placed side by side, at the original video's FPS and frame count:

    [ RGB ] | [ RGB + TEASER overlay ] | [ TEASER 3D ]

- LEFT:   the original RGB frame, uncropped, full resolution.
- CENTER: the original RGB frame with the TEASER mesh reconstruction
          rendered back into the frame's own coordinates and alpha-blended
          on top (so alignment with the real face can be judged in context).
- RIGHT:  the TEASER mesh reconstruction alone, face-centered, on a clean
          black background (same crop TEASER uses internally for inference).

This intentionally reuses the exact same building blocks already used by
`main/test_image.py` / `main/demo_video.py` (`run_mediapipe`, `crop_face`,
`TeaserEncoder`, `FLAME`, `Renderer`, and the same crop/warp-back math for
mapping the 224x224 reconstruction back onto the full-resolution frame) so
that this script's reconstruction is identical to the one already verified
to work. It does not modify or import from those demo scripts, so they are
left untouched.

Frames are read and written one at a time (no whole-video buffering).
"""
import argparse
import os

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import estimate_transform, warp

from src.teaser_encoder import TeaserEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
from utils.mediapipe_utils import run_mediapipe, detect_pose
from utils.pose_roi import (
    face_roi_from_pose,
    interpolate_boxes,
    crop_roi,
    landmarks_crop_to_frame,
    interpolate_landmarks,
)


def crop_face(frame, landmarks, scale=1.0, image_size=224):
    """Identical to the helper in main/test_image.py / main/demo_video.py."""
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2],
                         [center[0] - size / 2, center[1] + size / 2],
                         [center[0] + size / 2, center[1] - size / 2]])
    dst_pts = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, dst_pts)
    return tform


def make_title_bar(width, height, text):
    bar = np.full((height, width, 3), (32, 32, 32), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, height / 40.0)
    thickness = max(1, height // 20)
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(0, (width - tw) // 2)
    y = max(th + 2, (height + th) // 2)
    cv2.putText(bar, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return bar


def stack_panel(content_rgb, title, bar_height):
    bar = make_title_bar(content_rgb.shape[1], bar_height, title)
    return np.concatenate([bar, content_rgb], axis=0)


def mark_no_face(panel_rgb, text="no face detected"):
    panel_rgb = panel_rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, panel_rgb.shape[1] / 500.0)
    thickness = max(1, panel_rgb.shape[1] // 400)
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(4, (panel_rgb.shape[1] - tw) // 2)
    y = panel_rgb.shape[0] - 10
    cv2.rectangle(panel_rgb, (0, y - th - 8), (panel_rgb.shape[1], panel_rgb.shape[0]), (0, 0, 0), -1)
    cv2.putText(panel_rgb, text, (x, y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    return panel_rgb


class TeaserReconstructor:
    """Thin wrapper around the encoder/FLAME/renderer used by the official demos.

    Only builds the mesh reconstruction (no teaser_generator neural image
    synthesis) since the "TEASER 3D" panel is the rendered mesh, not the
    GAN-based image-to-image output.
    """

    def __init__(self, checkpoint_path, device='cuda', image_size=224):
        self.device = device
        self.image_size = image_size

        self.encoder = TeaserEncoder().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder_sd = {k.replace('teaser_encoder.', ''): v
                      for k, v in checkpoint.items() if 'teaser_encoder' in k}
        self.encoder.load_state_dict(encoder_sd)
        self.encoder.eval()

        self.flame = FLAME().to(device)
        self.renderer = Renderer().to(device)

    @torch.no_grad()
    def reconstruct(self, cropped_rgb_uint8):
        """cropped_rgb_uint8: HxWx3 uint8 RGB, already cropped to image_size.

        Returns rendered_rgb_uint8: image_size x image_size x 3 uint8 RGB,
        black background where no face geometry projects.
        """
        img = torch.tensor(cropped_rgb_uint8).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img = img.to(self.device)

        outputs = self.encoder(img)
        flame_output = self.flame.forward(outputs)
        renderer_output = self.renderer.forward(
            flame_output['vertices'], outputs['cam'],
            landmarks_fan=flame_output['landmarks_fan'],
            landmarks_mp=flame_output['landmarks_mp'])

        rendered_img = renderer_output['rendered_img']  # 1x3xHxW, RGB in [0,1]
        rendered_np = (rendered_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0)
        return rendered_np.astype(np.uint8)


def warp_back_to_original(rendered_rgb_uint8, tform, orig_h, orig_w):
    """Maps the image_size x image_size rendered reconstruction back into the
    coordinate frame of the original full-resolution frame, using the exact
    same warp direction already used by test_image.py / demo_video.py's
    `--render_orig` code path (`warp(rendered_img, tform, output_shape=...)`).
    """
    rendered_orig = warp(rendered_rgb_uint8, tform, output_shape=(orig_h, orig_w),
                          preserve_range=True).astype(np.uint8)

    face_mask = (rendered_rgb_uint8.sum(axis=-1) > 0).astype(np.float64)
    mask_orig = warp(face_mask, tform, output_shape=(orig_h, orig_w),
                      order=0, preserve_range=True)
    return rendered_orig, (mask_orig > 0.5)


def _iter_frames(input_path, max_frames=None, resize_to=None):
    """Yields resized BGR frames one at a time (never holds the whole video)."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")
    try:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and idx >= max_frames:
                break
            idx += 1
            if resize_to is not None:
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            yield frame
    finally:
        cap.release()


def precompute_pose_roi_landmarks(input_path, max_frames=None, resize_to=None, roi_size=512):
    """Two streaming passes over the video (re-reading from disk each time, so no
    whole-video frame buffer is ever held) implementing SMIRK's "pose_roi" face
    detection mode: MediaPipe Pose locates a face ROI on frames where TEASER's
    own direct FaceLandmarker would fail (small/distant face), then that same
    FaceLandmarker runs again on a zoomed-in crop of that ROI. Missing per-frame
    detections are interpolated over time so every frame ends up with a usable
    478-point landmark set (same format `run_mediapipe` returns), exactly as in
    SMIRK's implementation this was ported from.

    Only these compact per-frame arrays (a handful of floats/frame) are kept in
    memory across the whole video -- never the decoded frames themselves.

    Returns (landmarks, stats) where landmarks is a (T, 478, 3) ndarray already
    in the *original* (resized, if resize_to given) frame's pixel coordinates,
    or (None, stats) if pose_roi could not localize/detect a face in any frame
    at all (caller should then fall back to RGB-only for the whole video).
    """
    # Pass 1: MediaPipe Pose -> face ROI boxes.
    boxes = []
    pose_valid = []
    for frame in _iter_frames(input_path, max_frames=max_frames, resize_to=resize_to):
        result = detect_pose(frame, static_image_mode=False)
        if result is None:
            boxes.append(np.zeros(4, dtype=np.float32))
            pose_valid.append(False)
            continue
        xy, visibility = result
        box = face_roi_from_pose(xy, visibility)
        if box is None:
            boxes.append(np.zeros(4, dtype=np.float32))
            pose_valid.append(False)
        else:
            boxes.append(box)
            pose_valid.append(True)

    boxes = np.stack(boxes, axis=0)
    pose_valid = np.array(pose_valid, dtype=bool)
    stats = {"frame_count": len(boxes), "pose_roi_real": int(pose_valid.sum()),
              "pose_roi_interpolated": int((~pose_valid).sum())}

    if not np.any(pose_valid):
        return None, stats

    boxes = interpolate_boxes(boxes, pose_valid)

    # Pass 2: FaceLandmarker (TEASER's own, same model/weights) on each ROI crop.
    landmarks = np.full((len(boxes), 478, 3), np.nan, dtype=np.float32)
    face_detected = np.zeros(len(boxes), dtype=bool)
    for i, frame in enumerate(_iter_frames(input_path, max_frames=max_frames, resize_to=resize_to)):
        crop, tform = crop_roi(frame, boxes[i], out_size=roi_size)
        landmarks_crop = run_mediapipe(crop, face_detection_mode='direct')
        if landmarks_crop is None:
            continue
        landmarks[i] = landmarks_crop_to_frame(landmarks_crop, tform)
        face_detected[i] = True

    stats["face_real"] = int(face_detected.sum())
    stats["face_interpolated"] = int((~face_detected).sum())

    if not np.any(face_detected):
        return None, stats

    landmarks = interpolate_landmarks(landmarks, face_detected)
    return landmarks, stats


def process_video(input_path, output_path, checkpoint, device='cuda',
                   image_size=224, crop_scale=1.4, overlay_alpha=0.7,
                   title_bar_height=None, max_frames=None, max_output_height=None,
                   face_detection_mode='direct', roi_size=512):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if orig_w <= 0 or orig_h <= 0:
        raise RuntimeError(f"Could not read frame size from: {input_path}")

    # Optional downscale (never upscale) of the *whole* frame, preserving aspect
    # ratio and full field of view (no cropping) — only changes pixel density.
    resize_to = None
    if max_output_height is not None and orig_h > max_output_height:
        scale = max_output_height / orig_h
        resize_to = (int(round(orig_w * scale)), int(round(orig_h * scale)))
        orig_w, orig_h = resize_to

    pose_roi_landmarks = None
    if face_detection_mode == 'pose_roi':
        pose_roi_landmarks, pose_roi_stats = precompute_pose_roi_landmarks(
            input_path, max_frames=max_frames, resize_to=resize_to, roi_size=roi_size)
        print(f"[pose_roi] {pose_roi_stats}")
        if pose_roi_landmarks is None:
            print(f"[warn] pose_roi could not localize/detect a face in any frame of "
                  f"{input_path}; falling back to RGB-only for the whole video.")

    if title_bar_height is None:
        title_bar_height = max(24, orig_h // 12)

    right_panel_size = orig_h  # square reconstruction resized to match content height
    separator_w = max(2, orig_w // 200)
    separator = np.full((title_bar_height + orig_h, separator_w, 3), (200, 200, 200), dtype=np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    reconstructor = TeaserReconstructor(checkpoint, device=device, image_size=image_size)

    writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                 quality=8, macro_block_size=None)

    n_frames_total = 0
    n_frames_no_face = 0
    frame_idx = 0
    try:
        for frame_bgr in _iter_frames(input_path, max_frames=max_frames, resize_to=resize_to):
            n_frames_total += 1

            orig_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            face_found = False
            try:
                if pose_roi_landmarks is not None:
                    kpt_mediapipe = pose_roi_landmarks[frame_idx]
                else:
                    kpt_mediapipe = run_mediapipe(frame_bgr)
            except Exception:
                kpt_mediapipe = None
            frame_idx += 1

            if kpt_mediapipe is not None:
                try:
                    kpt2d = kpt_mediapipe[..., :2]
                    tform = crop_face(frame_bgr, kpt2d, scale=crop_scale, image_size=image_size)
                    cropped_bgr = warp(frame_bgr, tform.inverse, output_shape=(image_size, image_size),
                                        preserve_range=True).astype(np.uint8)
                    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

                    rendered_rgb = reconstructor.reconstruct(cropped_rgb)
                    face_found = True
                except Exception as e:
                    print(f"[warn] frame {frame_idx}: reconstruction failed ({e}); "
                          f"falling back to RGB-only for this frame.")
                    face_found = False
            else:
                n_frames_no_face += 1

            # ---- LEFT: original RGB, untouched ----
            left = orig_rgb

            # ---- RIGHT: TEASER 3D reconstruction alone, clean background ----
            if face_found:
                right = cv2.resize(rendered_rgb, (right_panel_size, right_panel_size),
                                    interpolation=cv2.INTER_LINEAR)
            else:
                right = np.zeros((right_panel_size, right_panel_size, 3), dtype=np.uint8)
                right = mark_no_face(right)

            # ---- CENTER: RGB + reconstruction warped back & alpha-blended ----
            if face_found:
                rendered_orig, mask_orig = warp_back_to_original(rendered_rgb, tform, orig_h, orig_w)
                center = orig_rgb.copy()
                mask3 = mask_orig[..., None]
                blended = (orig_rgb.astype(np.float32) * (1 - overlay_alpha)
                           + rendered_orig.astype(np.float32) * overlay_alpha)
                center = np.where(mask3, blended.astype(np.uint8), center)
            else:
                center = mark_no_face(orig_rgb.copy())

            left_panel = stack_panel(left, "RGB", title_bar_height)
            center_panel = stack_panel(center, "RGB + TEASER", title_bar_height)
            right_panel = stack_panel(right, "TEASER 3D", title_bar_height)

            composed = np.concatenate(
                [left_panel, separator, center_panel, separator, right_panel], axis=1)

            writer.append_data(composed)
    finally:
        writer.close()

    print(f"Done: {input_path} -> {output_path}")
    print(f"  frames processed: {n_frames_total}, frames with no detected face: {n_frames_no_face}")
    return {"frames": n_frames_total, "frames_no_face": n_frames_no_face,
            "fps": fps, "width": orig_w, "height": orig_h}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=str, required=True, help='Path to the input video')
    parser.add_argument('--output', type=str, required=True, help='Path to the output comparison MP4')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the TEASER checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--image_size', type=int, default=224, help='TEASER crop/input size')
    parser.add_argument('--crop_scale', type=float, default=1.4, help='Face crop scale (as in the official demos)')
    parser.add_argument('--overlay_alpha', type=float, default=0.7,
                         help='Blend strength of the rendered mesh over the RGB frame in the center panel')
    parser.add_argument('--title_bar_height', type=int, default=None,
                         help='Height in pixels of the panel title bars (default: scaled to frame height)')
    parser.add_argument('--max_frames', type=int, default=None,
                         help='Optional cap on number of frames (for quick testing only)')
    parser.add_argument('--max_output_height', type=int, default=None,
                         help='Optional cap on the RGB frame height (aspect-ratio preserved, '
                              'no cropping) to keep very high-resolution videos (e.g. 1080p) '
                              'from producing oversized output frames. Off by default.')
    parser.add_argument('--face_detection_mode', type=str, default='direct', choices=['direct', 'pose_roi'],
                         help="'direct' (default): TEASER's original full-frame FaceLandmarker. "
                              "'pose_roi': MediaPipe Pose locates a face ROI first (robust to "
                              "small/distant faces, e.g. full-body shots), then the exact same "
                              "FaceLandmarker runs on a zoomed-in crop of it; missing per-frame "
                              "detections are interpolated over time. Ported from SMIRK's own fix "
                              "for the same issue (see utils/pose_roi.py).")
    parser.add_argument('--roi_size', type=int, default=512,
                         help='Crop size used for the pose_roi zoomed-in FaceLandmarker pass.')
    args = parser.parse_args()

    process_video(
        input_path=args.input,
        output_path=args.output,
        checkpoint=args.checkpoint,
        device=args.device,
        image_size=args.image_size,
        crop_scale=args.crop_scale,
        overlay_alpha=args.overlay_alpha,
        title_bar_height=args.title_bar_height,
        max_frames=args.max_frames,
        max_output_height=args.max_output_height,
        face_detection_mode=args.face_detection_mode,
        roi_size=args.roi_size,
    )


if __name__ == '__main__':
    main()
