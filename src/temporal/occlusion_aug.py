"""Synthetic hand occlusion (TEMPORAL_README.md 5.4).

A *plan* is a list of occlusion episodes for one clip, sampled once (seeded)
from the real statistics (tools_temporal/occlusion_stats.py): start, length,
region (mouth / eyes), target coverage, hand patch, scale, rotation and a
smooth in-plane path. ``Occluder`` applies a plan to the clip's frames as they
are read, on the **full frame, before face detection and cropping**, and
returns the exact synthetic coverage per frame (``c_syn``, from the pasted
alpha mask over the region polygon).

Regions in the frame come from the clean clip's MediaPipe face landmarks
(stored in its feature cache): lips = outer lip contour, scaled 1.3x about
its centre (FLAME's ``lips`` region reaches past the vermilion); eyes = both
eye contours, scaled 1.5x (eyelids and the skin around them).

Hand patches: a folder of RGBA PNGs (``HandBank(path)``), split into train /
test identities by file list; ``HandBank('procedural')`` draws a skin-coloured
palm-and-fingers shape and exists **only for smoke tests**.
"""

import json
from pathlib import Path

import cv2
import numpy as np

# MediaPipe FaceMesh (478) contours.
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
CHEEKS = [50, 280, 205, 425, 123, 352]  # skin samples for the colour transfer
REGION_SCALE = {"mouth": 1.3, "eyes": 1.5}


def _scaled(points, scale):
    centre = points.mean(0)
    return centre + scale * (points - centre)


def region_polygons(landmarks_xy):
    """{'mouth': (K,2), 'eyes': [(K,2), (K,2)]} in frame pixels from 478 landmarks."""
    return {
        "mouth": [_scaled(landmarks_xy[LIPS_OUTER], REGION_SCALE["mouth"])],
        "eyes": [_scaled(landmarks_xy[LEFT_EYE], REGION_SCALE["eyes"]),
                 _scaled(landmarks_xy[RIGHT_EYE], REGION_SCALE["eyes"])],
    }


def coverage(alpha, polygons):
    """Fraction of the region (union of polygons) covered by alpha > 0.5."""
    mask = np.zeros(alpha.shape, np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 1)
    area = int(mask.sum())
    return float(((alpha > 0.5) & (mask > 0)).sum()) / area if area else 0.0


class HandBank:
    """RGBA hand patches. ``source`` = a folder of PNGs (optionally a list file of names), or 'procedural'."""

    def __init__(self, source, names=None):
        self.procedural = source == "procedural"
        if self.procedural:
            self.names = [f"procedural_{k}" for k in range(64)]
            return
        folder = Path(source)
        files = sorted(folder.glob("*.png"))
        if names is not None:
            keep = set(names)
            files = [f for f in files if f.stem in keep or f.name in keep]
        self.files = {f.stem: f for f in files}
        self.names = sorted(self.files)
        if not self.names:
            raise FileNotFoundError(f"no hand PNGs in {folder}")

    def get(self, name):
        """(H, W, 4) uint8 BGRA."""
        if self.procedural:
            return _procedural_hand(int(name.split("_")[1]))
        patch = cv2.imread(str(self.files[name]), cv2.IMREAD_UNCHANGED)
        if patch is None or patch.ndim != 3 or patch.shape[2] != 4:
            raise ValueError(f"{self.files[name]} is not an RGBA PNG")
        return patch


def _procedural_hand(seed, size=128):
    """A skin-coloured palm with fingers: smoke tests only, not a training occluder."""
    rng = np.random.default_rng(seed)
    alpha = np.zeros((size, size), np.uint8)
    cv2.ellipse(alpha, (size // 2, int(size * 0.62)), (int(size * 0.26), int(size * 0.3)), 0, 0, 360, 255, -1)
    for k in range(4):
        x = int(size * (0.32 + 0.12 * k))
        length = int(size * rng.uniform(0.28, 0.4))
        cv2.line(alpha, (x, int(size * 0.45)), (x + int(rng.normal(0, 3)), int(size * 0.45) - length),
                 255, int(size * 0.09))
    cv2.line(alpha, (int(size * 0.3), int(size * 0.7)), (int(size * 0.1), int(size * 0.5)), 255, int(size * 0.1))
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    base = np.array([rng.uniform(90, 140), rng.uniform(120, 170), rng.uniform(170, 220)])  # BGR skin
    shade = np.linspace(0.85, 1.1, size)[:, None, None]
    colour = np.clip(base[None, None, :] * shade + rng.normal(0, 4, (size, size, 3)), 0, 255)
    return np.dstack([colour.astype(np.uint8), alpha])


def _lab_transfer(patch_bgr, alpha, target_bgr):
    """Match the patch's Lab mean/std (inside alpha) to ``target_bgr`` pixels (N,3)."""
    if len(target_bgr) < 4:
        return patch_bgr
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(target_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    inside = alpha > 128
    if inside.sum() < 4:
        return patch_bgr
    src = lab[inside]
    mean_s, std_s = src.mean(0), src.std(0) + 1e-3
    mean_t, std_t = target.astype(np.float32).mean(0), target.astype(np.float32).std(0) + 1e-3
    lab = (lab - mean_s) / std_s * std_t + mean_t
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------- planning


class OcclusionStats:
    """Duration and coverage distributions from tools_temporal/occlusion_stats.py's JSON."""

    def __init__(self, path=None, fallback_mean=4.3):
        self.durations, self.weights, self.coverages = None, None, None
        if path is not None:
            stats = json.loads(Path(path).read_text())
            hist = stats["durations"]["histogram"]
            self.durations = np.array([int(k) for k in hist], dtype=np.int64)
            self.weights = np.array(list(hist.values()), dtype=np.float64)
            self.weights /= self.weights.sum()
            values = stats["coverage"].get("peak_mouth_values") or []
            self.coverages = np.array(values, dtype=np.float64) if values else None
            self.eyes_fraction = stats["coverage"].get("fraction_reaching_eyes", 0.1)
        else:
            self.eyes_fraction = 0.1
        self.fallback_mean = fallback_mean

    def duration(self, rng, max_len):
        if self.durations is None:
            # Geometric with the measured mean (4.3 frames) when no stats are given.
            d = int(rng.geometric(1.0 / self.fallback_mean))
        else:
            d = int(rng.choice(self.durations, p=self.weights))
        return int(np.clip(d, 1, max_len))

    def coverage(self, rng):
        if self.coverages is None or not len(self.coverages):
            return float(rng.uniform(0.25, 1.0))
        return float(np.clip(rng.choice(self.coverages) + rng.normal(0, 0.05), 0.2, 1.0))


def sample_plan(n_frames, stats, hand_names, rng, gap=(8, 24), max_len=20, eyes_fraction=None):
    """Episodes over a clip, separated by ``gap`` frames, leaving context at both ends."""
    eyes_fraction = stats.eyes_fraction if eyes_fraction is None else eyes_fraction
    episodes, t = [], int(rng.integers(3, max(4, gap[0])))
    while True:
        length = stats.duration(rng, max_len)
        if t + length + 2 > n_frames:
            break
        angle = rng.uniform(0, 2 * np.pi)
        episodes.append({
            "start": t, "length": length,
            "region": "eyes" if rng.random() < eyes_fraction else "mouth",
            "coverage": stats.coverage(rng),
            "hand": str(rng.choice(hand_names)),
            "scale": float(rng.uniform(1.8, 2.8)),      # hand height / inter-ocular distance
            "rotation": float(rng.uniform(-50, 50)),
            "flip": bool(rng.random() < 0.5),
            "direction": [float(np.cos(angle)), float(np.sin(angle))],
            "drift": float(rng.uniform(0.0, 0.4)),       # path length / hand size across the episode
            "seed": int(rng.integers(0, 2 ** 31)),
        })
        t += length + int(rng.integers(gap[0], gap[1] + 1))
    return episodes


# ---------------------------------------------------------------- applying


class Occluder:
    """Wraps a frame source; ``occluder[i]`` is frame i with the plan's hand pasted on it."""

    def __init__(self, frames, landmarks_xy, plan, bank):
        self.frames, self.landmarks, self.plan, self.bank = frames, landmarks_xy, plan, bank
        n = len(frames)
        self.c_syn = np.zeros((n, 2), np.float32)   # [mouth, eyes]
        self.syn_mask = np.zeros(n, bool)
        self.by_frame = {}
        for k, ep in enumerate(plan):
            for j in range(ep["length"]):
                t = ep["start"] + j
                if t < n:
                    self.by_frame[t] = (k, j)
                    self.syn_mask[t] = True
        self._patch_cache = {}

    def __len__(self):
        return len(self.frames)

    def _patch(self, ep, face_size):
        key = (ep["hand"], ep["scale"], ep["rotation"], ep["flip"], int(face_size))
        if key not in self._patch_cache:
            patch = self.bank.get(ep["hand"])
            if ep["flip"]:
                patch = patch[:, ::-1].copy()
            height = max(8, int(ep["scale"] * face_size))
            width = max(8, int(patch.shape[1] * height / patch.shape[0]))
            patch = cv2.resize(patch, (width, height), interpolation=cv2.INTER_AREA)
            side = int(np.hypot(height, width)) + 2
            canvas = np.zeros((side, side, 4), np.uint8)
            y0, x0 = (side - height) // 2, (side - width) // 2
            canvas[y0:y0 + height, x0:x0 + width] = patch
            rot = cv2.getRotationMatrix2D((side / 2, side / 2), ep["rotation"], 1.0)
            self._patch_cache[key] = cv2.warpAffine(canvas, rot, (side, side), flags=cv2.INTER_LINEAR)
        return self._patch_cache[key]

    def __getitem__(self, t):
        frame = self.frames[t]
        if t not in self.by_frame:
            return frame
        k, j = self.by_frame[t]
        ep = self.plan[k]
        lm = self.landmarks[t]
        polys = region_polygons(lm)[ep["region"]]
        centre = np.concatenate(polys).mean(0)
        face_size = max(np.linalg.norm(lm[33] - lm[263]), 4.0)
        patch = self._patch(ep, face_size)
        side = patch.shape[0]
        # Distance of the hand centre from the region centre sets the coverage:
        # 0 -> centred (total), up to ~0.45 hand sizes -> partial.
        direction = np.array(ep["direction"])
        progress = j / max(ep["length"] - 1, 1)
        offset = (1.0 - ep["coverage"]) * 0.45 * side + (progress - 0.5) * ep["drift"] * side
        cx, cy = centre + direction * offset
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))

        out = frame.copy()
        h, w = out.shape[:2]
        fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
        alpha_full = np.zeros((h, w), np.float32)
        if fx1 > fx0 and fy1 > fy0:
            crop = patch[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
            cheeks = lm[CHEEKS].round().astype(int)
            cheeks = cheeks[(cheeks[:, 0] >= 0) & (cheeks[:, 0] < w) & (cheeks[:, 1] >= 0) & (cheeks[:, 1] < h)]
            skin = frame[cheeks[:, 1], cheeks[:, 0]] if len(cheeks) else np.zeros((0, 3))
            colour = _lab_transfer(np.ascontiguousarray(crop[..., :3]), crop[..., 3], skin)
            a = crop[..., 3:4].astype(np.float32) / 255.0
            region = out[fy0:fy1, fx0:fx1].astype(np.float32)
            out[fy0:fy1, fx0:fx1] = np.clip(a * colour + (1 - a) * region, 0, 255).astype(np.uint8)
            alpha_full[fy0:fy1, fx0:fx1] = a[..., 0]
        all_polys = region_polygons(lm)
        self.c_syn[t] = [coverage(alpha_full, all_polys["mouth"]), coverage(alpha_full, all_polys["eyes"])]
        return out
