"""Track 1 metrics (TEMPORAL_README.md 6, 7), shared by validation and eval_temporal.py.

Everything is computed on canonical FLAME vertices in millimetres
(``CanonicalFlame``: identity and head rotation fixed), split into frame
groups:

- ``occluded``: frames covered by a hand (synthetic: ``syn_mask``; real:
  ``c_mnc > tau``);
- ``near``: within ``near_k`` frames of an occluded frame, not occluded;
- ``clean``: the rest.

Error metrics need a reference: TEASER on the unoccluded clip for a synthetic
variant, TEASER per-frame for a real clip (fidelity; meaningless on really
occluded frames, so those are left out of every error). Frames really
occluded in the source video are excluded from the error groups of a
synthetic variant as well.

Per group:
- ``err_<region>``: mean per-vertex L2 distance to the reference (mm), region
  in mouth / eyes / rest / face;
- ``lip_max``: per-frame max L2 over lip vertices, averaged (mm);
- ``accel``: mean norm of the predicted vertices' second difference (mm/frame^2),
  ``jerk``: of the third difference -- stability, no reference needed;
- ``accel_err``: mean norm of the difference between predicted and reference
  second differences.
For synthetic variants also ``by_duration``: ``err_mouth`` of occluded+near
frames per episode-length bucket (1-2, 3-5, 6-10, >10 frames).
"""

import numpy as np
import torch

BUCKETS = ((1, 2), (3, 5), (6, 10), (11, 10 ** 6))


def dilate(mask, k):
    out = mask.copy()
    for s in range(1, k + 1):
        out[s:] |= mask[:-s]
        out[:-s] |= mask[s:]
    return out


def frame_groups(occluded, valid, near_k=3, exclude=None):
    occluded = occluded & valid
    near = dilate(occluded, near_k) & ~occluded & valid
    clean = valid & ~occluded & ~near
    groups = {"clean": clean, "near": near, "occluded": occluded}
    if exclude is not None:
        groups = {k: v & ~exclude for k, v in groups.items()}
    return groups


@torch.no_grad()
def canonical_mm(flame, params, device="cpu"):
    """params dict of (T, ·) arrays -> (T, V, 3) numpy, millimetres."""
    t = lambda k: torch.as_tensor(np.asarray(params[k]), dtype=torch.float32, device=device)
    return (flame(t("expression"), t("jaw"), t("eyelid")) * 1000.0).cpu().numpy()


def _central(values, n, offset):
    """Assign a per-difference value to the frame at its centre; NaN elsewhere."""
    out = np.full(n, np.nan)
    out[offset:offset + len(values)] = values
    return out


def clip_metrics(pred_v, ref_v, groups, region_index, episodes=None):
    """pred_v / ref_v (T, V, 3) mm; groups {name: (T,) bool}; region_index {region: idx}."""
    n = len(pred_v)
    per_frame = {}
    dist = np.linalg.norm(pred_v - ref_v, axis=-1)  # (T, V)
    for region, idx in region_index.items():
        per_frame[f"err_{region}"] = dist[:, idx].mean(1)
    per_frame["lip_max"] = dist[:, region_index["mouth"]].max(1)
    if n >= 3:
        acc_p = pred_v[2:] - 2 * pred_v[1:-1] + pred_v[:-2]
        acc_r = ref_v[2:] - 2 * ref_v[1:-1] + ref_v[:-2]
        per_frame["accel"] = _central(np.linalg.norm(acc_p, axis=-1).mean(1), n, 1)
        per_frame["accel_err"] = _central(np.linalg.norm(acc_p - acc_r, axis=-1).mean(1), n, 1)
    if n >= 4:
        jerk = pred_v[3:] - 3 * pred_v[2:-1] + 3 * pred_v[1:-2] - pred_v[:-3]
        per_frame["jerk"] = _central(np.linalg.norm(jerk, axis=-1).mean(1), n, 1)

    out = {}
    for group, mask in groups.items():
        out[group] = {"frames": int(mask.sum())}
        for key, values in per_frame.items():
            v = values[mask]
            v = v[~np.isnan(v)]
            out[group][key] = (float(v.sum()), int(len(v)))  # (sum, count): poolable over clips
    if episodes:
        out["by_duration"] = {}
        for lo, hi in BUCKETS:
            mask = np.zeros(n, bool)
            for ep in episodes:
                if lo <= ep["length"] <= hi:
                    mask[max(0, ep["start"] - 3):ep["start"] + ep["length"] + 3] = True
            mask &= groups["occluded"] | groups["near"]
            v = per_frame["err_mouth"][mask]
            out["by_duration"][f"{lo}-{hi if hi < 10 ** 6 else 'inf'}"] = (float(v.sum()), int(len(v)))
    return out


def pool(results):
    """Pool a list of clip_metrics dicts into means."""
    pooled = {}
    for result in results:
        for group, values in result.items():
            g = pooled.setdefault(group, {})
            for key, value in values.items():
                if key == "frames":
                    g["frames"] = g.get("frames", 0) + value
                else:
                    s, c = g.get(key, (0.0, 0))
                    g[key] = (s + value[0], c + value[1])
    return {group: {k: (v if k == "frames" else (v[0] / v[1] if v[1] else float("nan")))
                    for k, v in values.items()} for group, values in pooled.items()}
