"""Track 1 losses (TEMPORAL_README.md 3 and 5.5). Teacher = the original, frozen TEASER.

- ``L_self``: on frames with a valid target, L1 between the adapter's output
  and TEASER per-frame, on canonical vertices (per region) or on the
  parameters (``self_target='vertices' | 'params'``).
- ``L_occ``: same form, on frames covered by a synthetic hand (and their
  neighbours); the target there is TEASER on the *unoccluded* frame.
- ``L_accel``: mean norm of the second difference of the predicted canonical
  vertices (towards zero).

Every per-frame target term is multiplied by the real-occlusion weight of
its region: ``(1 - c_real)``, and 0 where ``c_real > hard_thresh``. A frame that
is really occluded in the source video is therefore input only, never a
target. NaN in ``c_real`` (degenerate region hull) counts as not occluded.

Regions and their occlusion: mouth <- ``c_mouth``, eyes <- ``c_eyes``,
rest <- max of the two. In ``params`` mode expression and jaw use the mouth
weight, eyelids the eye weight.
"""

import torch

REGIONS = ("mouth", "eyes", "rest")


def real_occlusion_weight(c, hard_thresh=0.5):
    """(...,) occlusion in [0,1] -> weight in [0,1]: 1 - c, and 0 above ``hard_thresh``."""
    c = torch.nan_to_num(c, nan=0.0).clamp(0.0, 1.0)
    weight = 1.0 - c
    if hard_thresh is not None:
        weight = torch.where(c > hard_thresh, torch.zeros_like(weight), weight)
    return weight


def region_weights(c_mouth, c_eyes, hard_thresh=0.5):
    """Per-frame weights {'mouth','eyes','rest'} -> (B,T), from the real occlusion."""
    return {
        "mouth": real_occlusion_weight(c_mouth, hard_thresh),
        "eyes": real_occlusion_weight(c_eyes, hard_thresh),
        "rest": real_occlusion_weight(torch.maximum(torch.nan_to_num(c_mouth), torch.nan_to_num(c_eyes)),
                                      hard_thresh),
    }


def _weighted_mean(per_frame, weight):
    """sum(w * x) / sum(w) over (B,T); 0 when no frame has weight."""
    total = weight.sum()
    if total <= 0:
        return per_frame.sum() * 0.0
    return (per_frame * weight).sum() / total


def vertex_target_loss(pred_v, target_v, frame_weights, region_index, region_w, select=None):
    """L1 on canonical vertices, per region.

    pred_v, target_v (B,T,V,3); frame_weights {region: (B,T)}; region_index
    {region: LongTensor of vertex positions}; region_w {region: float};
    select (B,T) bool restricts the frames (None = all).
    Returns (loss, {region: loss}).
    """
    parts, total = {}, pred_v.sum() * 0.0
    for region in REGIONS:
        if region_w.get(region, 0) == 0:
            continue
        idx = region_index[region]
        per_frame = (pred_v[:, :, idx] - target_v[:, :, idx]).abs().mean(dim=(-1, -2))
        weight = frame_weights[region]
        if select is not None:
            weight = weight * select.to(weight.dtype)
        parts[region] = _weighted_mean(per_frame, weight)
        total = total + region_w[region] * parts[region]
    return total, parts


def param_target_loss(pred, target, frame_weights, select=None):
    """L1 on the 55 parameters. pred/target dicts with 'expression' (B,T,50), 'jaw' (B,T,3), 'eyelid' (B,T,2)."""
    mouth = torch.cat([(pred["expression"] - target["expression"]).abs(),
                       (pred["jaw"] - target["jaw"]).abs()], dim=-1).mean(-1)
    eyes = (pred["eyelid"] - target["eyelid"]).abs().mean(-1)
    w_mouth, w_eyes = frame_weights["mouth"], frame_weights["eyes"]
    if select is not None:
        w_mouth = w_mouth * select.to(w_mouth.dtype)
        w_eyes = w_eyes * select.to(w_eyes.dtype)
    parts = {"mouth": _weighted_mean(mouth, w_mouth), "eyes": _weighted_mean(eyes, w_eyes)}
    # Weighted by how many parameters each part has (53 vs 2), as one L1 over 55.
    return (53 * parts["mouth"] + 2 * parts["eyes"]) / 55, parts


def accel_loss(pred_v, valid=None):
    """Mean L2 norm of the vertices' second difference over time. pred_v (B,T,V,3); valid (B,T) bool."""
    if pred_v.shape[1] < 3:
        return pred_v.sum() * 0.0
    accel = pred_v[:, 2:] - 2 * pred_v[:, 1:-1] + pred_v[:, :-2]
    per_frame = accel.norm(dim=-1).mean(dim=-1)  # (B, T-2)
    if valid is None:
        return per_frame.mean()
    triplet = (valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]).to(per_frame.dtype)
    return _weighted_mean(per_frame, triplet)


def near_frames(mask, k):
    """Dilate a (B,T) bool mask by k frames on both sides."""
    if k <= 0:
        return mask
    x = mask.to(torch.float32)[:, None]
    x = torch.nn.functional.max_pool1d(x, 2 * k + 1, stride=1, padding=k)
    return x[:, 0] > 0
