"""Synthetic hand occlusion (TEMPORAL_README.md 5.4).

A *plan* is a list of occlusion episodes for one clip, sampled once (seeded)
from the real statistics (tools_temporal/occlusion_stats.py). Each episode has
an entry (the hand comes in from outside the face), a core of ``length``
frames at a target coverage, and an exit, all inside one clean segment.
``Occluder`` applies a plan to the clip's frames as they are read, on the
**full frame, before face detection and cropping**, and records the exact
synthetic coverage per frame (``c_syn``).

Face regions (``FaceRegions``): by default the FLAME lips / eye region of the
clip's SMPL-X fit projected with the fit's camera
(tools_temporal/export_face_regions.py) -- the very regions real occlusion is
measured on, so real and synthetic coverage are the same quantity. Coverage
= area(hand alpha > 0.5 n convex hull of the region) / area(hull). A fallback
from MediaPipe landmarks exists for clips without a fit (smoke tests).

Placement: the hand's centre sits at the region's centre plus an offset
along a random direction. For every episode the offset is **calibrated** on
each core frame by bisection until the measured coverage reaches
the episode's target (sampled from the real per-episode peaks) -- on every
core frame, so the coverage holds while the face moves; entry and exit move
the hand in from and back out to one hand-size away. The hand is placed by
its centroid, not by the centre of its cut-out.

Paste quality: the patch's Lab means are moved halfway towards the face's
skin (texture and contrast kept), the alpha is feathered in proportion to the
hand's size, and a soft shadow darkens the face under and beside the hand.

Hand patches: a folder of RGBA PNGs (``HandBank(path)``), restricted to a
file list (train / held-out identities); ``HandBank('procedural')`` draws a
skin-coloured palm-and-fingers shape and exists **only for smoke tests**.
"""

import json
from pathlib import Path

import cv2
import numpy as np

# MediaPipe FaceMesh (478) contours, for the landmark fallback only.
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
COLOUR_STRENGTH = 0.5      # share of the Lab mean difference towards the skin
SHADOW_DARKEN = 0.35       # darkening at the shadow's core
REGIONS = ("mouth", "eyes")


def _hull(points):
    return cv2.convexHull(np.round(points).astype(np.int32).reshape(-1, 1, 2)).reshape(-1, 2)


def _scaled(points, scale):
    centre = points.mean(0)
    return centre + scale * (points - centre)


class FaceRegions:
    """Per-frame region hulls, face size and skin samples."""

    def __init__(self, mouth_points, eyes_points):
        self.mouth, self.eyes = mouth_points, eyes_points  # (T, K, 2) each

    @classmethod
    def from_fit(cls, path):
        with np.load(path) as z:
            return cls(z["mouth"].astype(np.float64), z["eyes"].astype(np.float64))

    @classmethod
    def from_landmarks(cls, landmarks_xy):
        """Fallback without a fit: MediaPipe lips (x1.3) and eye contours (x1.5)."""
        mouth = np.stack([_scaled(lm[LIPS_OUTER], 1.3) for lm in landmarks_xy])
        eyes = np.stack([np.concatenate([_scaled(lm[LEFT_EYE], 1.5), _scaled(lm[RIGHT_EYE], 1.5)])
                         for lm in landmarks_xy])
        return cls(mouth, eyes)

    def __len__(self):
        return len(self.mouth)

    def hulls(self, t):
        return {"mouth": _hull(self.mouth[t]), "eyes": _hull(self.eyes[t])}

    def centre(self, t, region):
        return (self.mouth if region == "mouth" else self.eyes)[t].mean(0)

    def face_size(self, t):
        """Width of the eye region (about the outer eye corners' distance)."""
        return max(float(np.ptp(self.eyes[t][:, 0])), 4.0)

    def skin_mask(self, t, shape):
        """Cheeks / nose: the hull of mouth + eyes minus both regions."""
        mask = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(mask, _hull(np.concatenate([self.mouth[t], self.eyes[t]])), 1)
        for hull in self.hulls(t).values():
            cv2.fillConvexPoly(mask, hull, 0)
        return mask > 0


def coverage(alpha, hull):
    mask = np.zeros(alpha.shape, np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    area = int(mask.sum())
    return float(((alpha > 0.5) & (mask > 0)).sum()) / area if area else 0.0


class HandBank:
    """RGBA hand patches. ``source`` = a folder of PNGs (optionally a list of names), or 'procedural'."""

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


def _colour_match(patch_bgr, alpha, skin_bgr, strength=COLOUR_STRENGTH):
    """Move the patch's Lab means part of the way to the skin's; keep its texture and contrast."""
    inside = alpha > 128
    if len(skin_bgr) < 8 or inside.sum() < 8:
        return patch_bgr
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    skin = cv2.cvtColor(skin_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    shift = strength * (skin.astype(np.float32).mean(0) - lab[inside].mean(0))
    return cv2.cvtColor(np.clip(lab + shift, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------- planning


class OcclusionStats:
    """Duration, coverage and region distributions from tools_temporal/occlusion_stats.py's JSON."""

    def __init__(self, path=None, fallback_mean=4.3):
        self.durations, self.weights, self.coverages = None, None, None
        self.eyes_fraction = 0.05
        if path is not None:
            stats = json.loads(Path(path).read_text())
            hist = stats["durations"]["histogram"]
            self.durations = np.array([int(k) for k in hist], dtype=np.int64)
            self.weights = np.array(list(hist.values()), dtype=np.float64)
            self.weights /= self.weights.sum()
            values = [v for v in (stats["coverage"].get("peak_mouth_values") or []) if v > 0.2]
            self.coverages = np.array(values, dtype=np.float64) if values else None
            # Episodes on the eyes *instead of* the mouth. (fraction_reaching_eyes counts
            # mouth episodes whose hand also touches the eye region: not the same thing.)
            self.eyes_fraction = stats["coverage"].get("eyes_only_rate", 0.05)
        self.fallback_mean = fallback_mean

    def duration(self, rng, max_len):
        if self.durations is None:
            d = int(rng.geometric(1.0 / self.fallback_mean))  # measured mean, 4.3 frames
        else:
            d = int(rng.choice(self.durations, p=self.weights))
        return int(np.clip(d, 1, max_len))

    def coverage(self, rng):
        if self.coverages is None:
            return float(rng.uniform(0.25, 1.0))
        return float(np.clip(rng.choice(self.coverages) + rng.normal(0, 0.03), 0.21, 1.0))


def sample_plan(n_frames, stats, hand_names, rng, gap=(8, 24), max_len=20, eyes_fraction=None,
                segments=None, entry=(2, 4)):
    """Episodes (entry + core of ``length`` frames + exit) inside the clean segments.

    ``segments`` ([[start, end_exclusive], ...], select_clean_clips.py): every
    pasted frame, entry and exit included, lies in one, so it has a clean
    target; default the whole clip. Episodes are ``gap`` frames apart.
    """
    eyes_fraction = stats.eyes_fraction if eyes_fraction is None else eyes_fraction
    episodes = []
    for seg_start, seg_end in (segments if segments is not None else [(0, n_frames)]):
        t = seg_start + int(rng.integers(1, max(2, gap[0] // 2)))
        while True:
            length = stats.duration(rng, max_len)
            n_in, n_out = (int(rng.integers(entry[0], entry[1] + 1)) for _ in range(2))
            start = t + n_in
            if start + length + n_out > seg_end:
                break
            angle = rng.uniform(0, 2 * np.pi)
            episodes.append({
                "start": int(start), "length": int(length), "entry": n_in, "exit": n_out,
                "region": "eyes" if rng.random() < eyes_fraction else "mouth",
                "coverage": stats.coverage(rng),
                "hand": str(rng.choice(hand_names)),
                "scale": float(rng.uniform(1.0, 1.6)),       # hand height / eye-region width
                "rotation": float(rng.uniform(-50, 50)),
                "flip": bool(rng.random() < 0.5),
                "direction": [float(np.cos(angle)), float(np.sin(angle))],
                "seed": int(rng.integers(0, 2 ** 31)),
            })
            t = start + length + n_out + int(rng.integers(gap[0], gap[1] + 1))
    return episodes


# ---------------------------------------------------------------- applying


class Occluder:
    """Wraps a frame source; ``occluder[i]`` is frame i with the plan's hands pasted on it."""

    def __init__(self, frames, regions, plan, bank):
        self.frames, self.plan, self.bank = frames, plan, bank
        self.regions = regions if isinstance(regions, FaceRegions) else FaceRegions.from_landmarks(regions)
        n = len(frames)
        self.c_syn = np.zeros((n, 2), np.float32)   # [mouth, eyes]
        self.syn_mask = np.zeros(n, bool)            # any pasted hand (entry, core, exit)
        self.core_mask = np.zeros(n, bool)           # core frames only
        self.by_frame = {}
        for k, ep in enumerate(plan):
            n_in, n_out = ep.get("entry", 0), ep.get("exit", 0)
            for t in range(ep["start"] - n_in, ep["start"] + ep["length"] + n_out):
                if 0 <= t < n:
                    self.by_frame[t] = k
                    self.syn_mask[t] = True
                    self.core_mask[t] = ep["start"] <= t < ep["start"] + ep["length"]
        self._patch_cache, self._offset = {}, {}

    def __len__(self):
        return len(self.frames)

    def _patch(self, ep, face_size):
        size = int(round(face_size))
        key = (ep["hand"], ep["scale"], ep["rotation"], ep["flip"], size)
        if key not in self._patch_cache:
            patch = self.bank.get(ep["hand"])
            if ep["flip"]:
                patch = patch[:, ::-1].copy()
            height = max(8, int(ep["scale"] * size))
            width = max(8, int(patch.shape[1] * height / patch.shape[0]))
            patch = cv2.resize(patch, (width, height), interpolation=cv2.INTER_AREA)
            side = int(np.hypot(height, width)) + 2
            canvas = np.zeros((side, side, 4), np.uint8)
            y0, x0 = (side - height) // 2, (side - width) // 2
            canvas[y0:y0 + height, x0:x0 + width] = patch
            rot = cv2.getRotationMatrix2D((side / 2, side / 2), ep["rotation"], 1.0)
            patch = cv2.warpAffine(canvas, rot, (side, side), flags=cv2.INTER_LINEAR)
            # Feathered edge, in proportion to the hand's size.
            sigma = max(0.6, 0.012 * height)
            alpha = cv2.GaussianBlur(patch[..., 3].astype(np.float32), (0, 0), sigma) / 255.0
            # Where the hand's mass is, relative to the canvas centre: the hand is
            # placed by its centroid, not by the centre of its (mostly empty) box.
            ys, xs = np.nonzero(alpha > 0.5)
            mass = np.array([xs.mean() - side / 2, ys.mean() - side / 2]) if len(xs) else np.zeros(2)
            self._patch_cache[key] = (patch[..., :3].copy(), alpha, height, mass)
        return self._patch_cache[key]

    @staticmethod
    def _alpha_at(alpha, centre, shape, mass=(0.0, 0.0)):
        """The hand's alpha placed with its centroid at ``centre`` on a full-frame canvas."""
        side = alpha.shape[0]
        x0 = int(round(centre[0] - side / 2 - mass[0]))
        y0 = int(round(centre[1] - side / 2 - mass[1]))
        h, w = shape
        out = np.zeros(shape, np.float32)
        fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
        if fx1 > fx0 and fy1 > fy0:
            out[fy0:fy1, fx0:fx1] = alpha[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
        return out, (x0, y0)

    def _calibrated_offset(self, k, t, shape):
        """Offset (pixels) along the episode's direction that reaches its target coverage on frame t.

        Coverage falls as the hand moves out, so bisection on the offset; when even the
        centred hand covers less than the target, it stays centred.
        """
        key = (k, t)
        if key in self._offset:
            return self._offset[key]
        ep = self.plan[k]
        _, alpha, height, mass = self._patch(ep, self.regions.face_size(t))
        hull = self.regions.hulls(t)[ep["region"]]
        centre = self.regions.centre(t, ep["region"])
        direction = np.array(ep["direction"])
        cov = lambda d: coverage(self._alpha_at(alpha, centre + direction * d, shape, mass)[0], hull)
        lo, hi = 0.0, 1.2 * height
        if cov(lo) >= ep["coverage"]:
            for _ in range(12):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if cov(mid) >= ep["coverage"] else (lo, mid)
        self._offset[key] = (lo, height)
        return self._offset[key]

    def __getitem__(self, t):
        frame = self.frames[t]
        if t not in self.by_frame:
            return frame
        k = self.by_frame[t]
        ep = self.plan[k]
        shape = frame.shape[:2]
        start, end = ep["start"], ep["start"] + ep["length"]
        colour, alpha, _, mass = self._patch(ep, self.regions.face_size(t))
        if t < start:                        # entry: from one hand-size out to the first core position
            offset, height = self._calibrated_offset(k, start, shape)
            frac = (start - t) / (ep.get("entry", 1) + 1)
            d = offset + frac * height
        elif t >= end:                       # exit: back out from the last core position
            offset, height = self._calibrated_offset(k, end - 1, shape)
            frac = (t - end + 1) / (ep.get("exit", 1) + 1)
            d = offset + frac * height
        else:                                # core: at the target coverage on every frame
            d, height = self._calibrated_offset(k, t, shape)
        centre = self.regions.centre(t, ep["region"]) + np.array(ep["direction"]) * d
        a_full, (x0, y0) = self._alpha_at(alpha, centre, shape, mass)

        out = frame.astype(np.float32)
        # Soft shadow, down and to the side of the hand, on what is under it.
        shadow = cv2.GaussianBlur(a_full, (0, 0), max(1.0, 0.05 * height))
        shift = np.float32([[1, 0, 0.03 * height], [0, 1, 0.05 * height]])
        shadow = cv2.warpAffine(shadow, shift, (shape[1], shape[0]))
        out *= (1.0 - SHADOW_DARKEN * shadow)[..., None]
        # The hand itself, colour-matched to the face's skin.
        skin = frame[self.regions.skin_mask(t, shape)]
        side = alpha.shape[0]
        canvas = np.zeros((shape[0], shape[1], 3), np.float32)
        h, w = shape
        fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
        if fx1 > fx0 and fy1 > fy0:
            matched = _colour_match(np.ascontiguousarray(colour), (alpha * 255).astype(np.uint8), skin)
            canvas[fy0:fy1, fx0:fx1] = matched[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
        out = a_full[..., None] * canvas + (1.0 - a_full[..., None]) * out
        hulls = self.regions.hulls(t)
        self.c_syn[t] = [coverage(a_full, hulls["mouth"]), coverage(a_full, hulls["eyes"])]
        return np.clip(out, 0, 255).astype(np.uint8)
