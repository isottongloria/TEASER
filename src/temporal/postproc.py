"""Baselines on TEASER's per-frame parameters (TEMPORAL_README.md 5.8).

- ``savgol``: Savitzky-Golay, window 9, order 2, ``mode='nearest'`` -- RGB2SMPLX's
  ``scripts/postfilter.py`` (SG9). RGB2SMPLX smooths rotations (the jaw) as
  quaternions; here the jaw's axis-angle is smoothed linearly, which for
  TEASER's small jaw angles (opening <= ~0.5 rad, the rest clamped to 0.2)
  is the same to well below the metric's resolution.
- ``hermite_interp``: RGB2SMPLX's jitter-fix splice (``rgb2smplx/stages/
  face_jitter_fix.py``), copied because TEASER's env is Python 3.9: episodes of
  ``c > 0.2`` up to 15 frames, extended by 1 frame each side, replaced by a
  cubic Hermite spline through the frames around them, with velocities from
  a 2-frame look-back / look-ahead; skipped without context on both sides.
  The current pipeline (T2) is SG9 then this splice, in that order.
- ``SmoothNet``: light reimplementation of SmoothNet (Zeng et al. 2022):
  per-dimension fully connected layers over the time axis of a fixed window,
  residual, last layer zero-initialised (identity at start). Trained with the
  same recipe as the adapter (T3).
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicHermiteSpline
from scipy.signal import savgol_filter

TAU = 0.20
MAX_EPISODE_FRAMES = 15
EXTEND_FRAMES = 1
VELOCITY_BASELINE_FRAMES = 2


def savgol(values, window=9, order=2):
    """Zero-phase SG along time of a (T, D) array, as RGB2SMPLX's postfilter."""
    values = np.asarray(values, dtype=np.float32)
    if window < 3 or len(values) < window:
        return values.copy()
    window = window if window % 2 else window + 1
    return savgol_filter(values, window, min(order, window - 1), axis=0, mode="nearest").astype(np.float32)


def segments(flags):
    """Run-length encode a bool series into (start, end_inclusive, length)."""
    out, start = [], None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i - 1, i - start))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1, len(flags) - start))
    return out


def _hermite_fill(params, start, end, baseline=VELOCITY_BASELINE_FRAMES):
    n = len(params)
    pre, post = start - 1, end + 1
    if pre - baseline < 0 or post + baseline >= n:
        return False
    v_pre = (params[pre] - params[pre - baseline]) / baseline
    v_post = (params[post + baseline] - params[post]) / baseline
    spline = CubicHermiteSpline(np.array([pre, post], dtype=np.float64),
                                np.stack([params[pre], params[post]]),
                                np.stack([v_pre, v_post]), axis=0)
    params[start:end + 1] = spline(np.arange(start, end + 1, dtype=np.float64))
    return True


def hermite_interp(tracks, occlusion, tau=TAU, max_len=MAX_EPISODE_FRAMES, extend=EXTEND_FRAMES):
    """Splice every short occlusion episode in each (T, D) track of ``tracks`` (dict).

    ``occlusion`` (T,) is the hand-face IoA (NaN = not occluded). A track is
    only changed where every track could be spliced, as RGB2SMPLX requires both
    expression and jaw to succeed. Returns (new tracks, report).
    """
    occluded = np.nan_to_num(np.asarray(occlusion, dtype=np.float64), nan=0.0) > tau
    tracks = {k: np.array(v, dtype=np.float64, copy=True) for k, v in tracks.items()}
    n = len(occluded)
    report = {"n_episodes": 0, "n_corrected": 0, "n_skipped_too_long": 0, "n_skipped_no_context": 0}
    for start, end, length in segments(occluded):
        report["n_episodes"] += 1
        if length > max_len:
            report["n_skipped_too_long"] += 1
            continue
        s, e = max(0, start - extend), min(n - 1, end + extend)
        trial = {k: v.copy() for k, v in tracks.items()}
        if all(_hermite_fill(v, s, e) for v in trial.values()):
            tracks = trial
            report["n_corrected"] += 1
        else:
            report["n_skipped_no_context"] += 1
    return {k: v.astype(np.float32) for k, v in tracks.items()}, report


class SmoothNet(nn.Module):
    """Per-dimension temporal MLP on windows of ``window`` frames: x (B, T=window, D) -> (B, T, D)."""

    def __init__(self, window=16, hidden=128, n_blocks=3, dropout=0.1):
        super().__init__()
        self.window = window
        self.encoder = nn.Sequential(nn.Linear(window, hidden), nn.LeakyReLU(0.1))
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU(0.1), nn.Dropout(dropout),
                          nn.Linear(hidden, hidden), nn.LeakyReLU(0.1))
            for _ in range(n_blocks))
        self.decoder = nn.Linear(hidden, window)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        if x.shape[1] != self.window:
            raise ValueError(f"SmoothNet expects windows of {self.window} frames, got {x.shape[1]}")
        h = self.encoder(x.transpose(1, 2))  # (B, D, hidden): each dimension on its own
        for block in self.blocks:
            h = h + block(h)
        return x + self.decoder(h).transpose(1, 2)


@torch.no_grad()
def smoothnet_clip(model, values, stride=4):
    """Overlap-add SmoothNet over a whole (T, D) tensor, edge-replicated to fit the window."""
    window, t_len = model.window, values.shape[0]
    half = window // 2
    x = torch.cat([values[:1].expand(half, -1), values, values[-1:].expand(window - half, -1)])
    starts = list(range(0, x.shape[0] - window + 1, stride))
    if starts[-1] != x.shape[0] - window:
        starts.append(x.shape[0] - window)
    out = torch.zeros_like(x)
    weight = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
    hann = torch.hann_window(window, periodic=False, dtype=x.dtype, device=x.device).clamp_min(1e-3)[:, None]
    y = model(torch.stack([x[s:s + window] for s in starts]))
    for s, ys in zip(starts, y):
        out[s:s + window] += hann * ys
        weight[s:s + window] += hann
    return (out / weight)[half:half + t_len]
