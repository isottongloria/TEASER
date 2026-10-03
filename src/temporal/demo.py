"""``main/demo_video.py --temporalize_teaser``: per-frame TEASER vs temporal TEASER, side by side.

For every video in ``--input_path``: TEASER's own per-frame crops (MediaPipe
FaceLandmarker on the full frame, ``crop_face`` at scale 1.4, as the original
demo), the pooled features of every frame, the temporal model over the whole
clip (sliding window), then one output video per input with three panels:
crop | TEASER per-frame | temporal TEASER. Frames without a face are skipped
like in the original demo. The original demo path is not touched.
"""

import os

import cv2
import imageio
import numpy as np
import torch
from skimage.transform import warp


def run_temporal_demo(args, crop_face):
    from src.FLAME.FLAME import FLAME
    from src.renderer.renderer import Renderer
    from src.temporal import split_encoder as se
    from tools_temporal.eval_temporal import load_checkpoint
    from utils.mediapipe_utils import run_mediapipe

    if not args.temporal_ckpt:
        raise SystemExit("--temporalize_teaser needs --temporal_ckpt")
    device = args.device
    model, cfg = load_checkpoint(args.temporal_ckpt, device)
    encoder = se.load_teaser_encoder(args.checkpoint, device)
    flame = FLAME().to(device)
    renderer = Renderer().to(device)
    os.makedirs(args.out_path, exist_ok=True)

    for video in sorted(os.listdir(args.input_path)):
        cap = cv2.VideoCapture(os.path.join(args.input_path, video))
        if not cap.isOpened():
            print(f"Error opening video file {video}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        crops, per_frame, feats = [], [], []
        while True:
            ret, image = cap.read()
            if not ret:
                break
            kpt = run_mediapipe(image)
            if kpt is None:
                continue
            tform = crop_face(image, kpt[..., :2], scale=1.4, image_size=224)
            crop = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = torch.tensor(crop).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            with torch.no_grad():
                f = se.extract_features(encoder, tensor, ("expr", "pose", "shape"))
                per_frame.append({k: v.cpu() for k, v in se.apply_heads(encoder, f).items()})
            feats.append(se.concat_features(f, cfg.data.feature_set).cpu())
            crops.append(crop)
        cap.release()
        if not crops:
            continue

        n = len(crops)
        stack = lambda k: torch.cat([p[k] for p in per_frame])
        batch = {"feats": torch.cat(feats)[None].to(device),
                 "c_real": torch.zeros(1, n, 2, device=device), "c_syn": torch.zeros(1, n, 2, device=device),
                 "valid": torch.ones(1, n, dtype=torch.bool, device=device),
                 "input_params": {"expression": stack("expression_params")[None].to(device),
                                  "jaw": stack("jaw_params")[None].to(device),
                                  "eyelid": stack("eyelid_params")[None].to(device)}}
        temporal = model.infer_clip(batch, window=cfg.train.window, stride=cfg.train.infer_stride)

        frames = []
        with torch.no_grad():
            for t in range(n):
                base = {k: v[t:t + 1].to(device) for k, v in per_frame[t].items() if k != "token"}
                panels = [crops[t]]
                for expression, jaw, eyelid in (
                        (base["expression_params"], base["jaw_params"], base["eyelid_params"]),
                        (temporal["expression"][:, t], temporal["jaw"][:, t], temporal["eyelid"][:, t])):
                    params = dict(base, expression_params=expression, jaw_params=jaw, eyelid_params=eyelid)
                    out = flame.forward(params)
                    img = renderer.forward(out["vertices"], params["cam"])["rendered_img"]
                    panels.append((img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                frames.append(np.concatenate(panels, axis=1))
        target = os.path.join(args.out_path, f"{os.path.splitext(video)[0]}_temporal.mp4")
        imageio.mimsave(target, frames, fps=fps or 25)
        print(f"[temporal demo] {n} frames -> {target}")
