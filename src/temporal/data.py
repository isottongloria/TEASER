"""Training windows from feature caches, real occlusion and synthetic variants (TEMPORAL_README.md 5.6).

A clean clip = one feature cache (tools_temporal/extract_features.py) plus,
when available, its real-occlusion file (tools_temporal/compute_real_occlusion.py);
a variant = the same clip with synthetic hands pasted on it
(tools_temporal/make_synthetic_variants.py). Everything is indexed by the
clip's ``frames/`` position; lengths are checked.

Every sample is a window of ``window`` frames (edge-padded past the end,
padded frames ``valid = False``) with:

``feats``         (T, F)  adapter input: features of the frames the model sees
``input_params``  dict    TEASER per-frame on those same frames (SmoothNet's input, T0)
``teacher``       dict    TEASER per-frame on the *unoccluded* frames (the target);
                          with ``teacher_smoothing='sg9'`` its clip-level SG9
``c_real``        (T, 2)  real occlusion [mouth, eyes] of the source video
``c_mnc``         (T,)    real mouth/nose/chin IoA
``c_syn``         (T, 2)  synthetic coverage [mouth, eyes] (0 in a clean window)
``syn_mask``      (T,)    frames with a pasted hand
``valid``         (T,)

Params dicts have ``expression`` (T, 50), ``jaw`` (T, 3), ``eyelid`` (T, 2).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.temporal.postproc import savgol
from src.temporal.split_encoder import FEATURE_SETS

PARAM_KEYS = {"expression": "expression", "jaw": "jaw_pose", "eyelid": "eyelid"}


def read_list(path):
    return [line.split()[0] for line in Path(path).read_text().splitlines() if line.strip()]


def _features(cache, feature_set, path):
    missing = [f"feat_{n}" for n in FEATURE_SETS[feature_set] if f"feat_{n}" not in cache.files]
    if missing:
        raise KeyError(f"{path} has no {missing} (extract with --temporal_feats {feature_set})")
    return np.concatenate([cache[f"feat_{n}"].astype(np.float32) for n in FEATURE_SETS[feature_set]], axis=1)


def _params(cache):
    return {k: np.asarray(cache[v], np.float32) for k, v in PARAM_KEYS.items()}


class Clip:
    """A clean clip: what the model sees and the target are the same frames."""

    def __init__(self, cache_path, occ_path=None, feature_set="expr", teacher_smoothing=None):
        with np.load(cache_path) as cache:
            self.feats = _features(cache, feature_set, cache_path)
            self.input_params = _params(cache)
            self.valid = cache["valid"].astype(bool)
            self.fps = float(cache["fps"]) if "fps" in cache.files else 25.0
        if teacher_smoothing == "sg9":
            self.teacher = {k: savgol(v, 9, 2) for k, v in self.input_params.items()}
        elif teacher_smoothing:
            raise ValueError(f"unknown teacher_smoothing {teacher_smoothing!r}")
        else:
            self.teacher = self.input_params
        self.name = Path(cache_path).name.split(".")[0]
        n = len(self.feats)
        if occ_path is not None and Path(occ_path).is_file():
            with np.load(occ_path) as occ:
                if len(occ["c_mnc"]) != n:
                    raise ValueError(f"{occ_path}: {len(occ['c_mnc'])} frames, cache has {n}")
                self.c_real = np.nan_to_num(np.stack([occ["c_mouth"], occ["c_eyes"]], 1)).astype(np.float32)
                self.c_mnc = np.nan_to_num(occ["c_mnc"]).astype(np.float32)
        else:
            warnings.warn(f"no real occlusion for {self.name}: c_real = 0")
            self.c_real = np.zeros((n, 2), np.float32)
            self.c_mnc = np.zeros(n, np.float32)
        self.c_syn = np.zeros((n, 2), np.float32)
        self.syn_mask = np.zeros(n, bool)
        self.episodes = []

    def __len__(self):
        return len(self.feats)

    def arrays(self):
        """Every per-frame array of the clip, by sample key."""
        return {"feats": self.feats, "input_params": self.input_params, "teacher": self.teacher,
                "c_real": self.c_real, "c_mnc": self.c_mnc, "c_syn": self.c_syn,
                "syn_mask": self.syn_mask, "valid": self.valid}

    def window(self, start, length):
        n = len(self)
        positions = np.arange(start, start + length)
        idx = np.clip(positions, 0, n - 1)
        out = {}
        for key, value in self.arrays().items():
            if isinstance(value, dict):
                out[key] = {k: torch.from_numpy(v[idx]) for k, v in value.items()}
            else:
                out[key] = torch.from_numpy(value[idx])
        out["valid"] = out["valid"] & torch.from_numpy((positions >= 0) & (positions < n))
        return out


class VariantClip(Clip):
    """A synthetic-occlusion variant: sees the pasted frames, targets the clean clip's TEASER."""

    def __init__(self, variant_path, clean, feature_set="expr"):
        with np.load(variant_path) as cache:
            self.feats = _features(cache, feature_set, variant_path)
            self.input_params = _params(cache)
            valid = cache["valid"].astype(bool)
            self.c_syn = cache["c_syn"].astype(np.float32)
            self.syn_mask = cache["syn_mask"].astype(bool)
            self.episodes = json.loads(str(cache["plan"]))
        if len(self.feats) != len(clean):
            raise ValueError(f"{variant_path}: {len(self.feats)} frames, clean clip has {len(clean)}")
        self.teacher, self.c_real, self.c_mnc = clean.teacher, clean.c_real, clean.c_mnc
        self.valid = valid & clean.valid
        self.fps = clean.fps
        self.name = Path(variant_path).name[:-4]
        self.clean = clean


def load_clips(names, cache_dir, occ_dir=None, feature_set="expr", teacher_smoothing=None):
    clips = []
    for name in names:
        cache = Path(cache_dir) / f"{name}.npz"
        if not cache.is_file():
            warnings.warn(f"no cache for {name}, skipped")
            continue
        occ = Path(occ_dir) / f"{name}.occ.npz" if occ_dir else None
        clips.append(Clip(cache, occ, feature_set, teacher_smoothing))
    return clips


def load_variants(clips, variant_dir, feature_set="expr"):
    variants = []
    for clip in clips:
        for path in sorted(Path(variant_dir).glob(f"{clip.name}.v*.npz")):
            variants.append(VariantClip(path, clip, feature_set))
    return variants


class WindowDataset(Dataset):
    """Random windows. With probability ``occ_aug_p`` a window comes from a synthetic
    variant, placed so that one of its episodes lies inside with context on both
    sides when the window allows it; otherwise from a clean clip. Clips are drawn
    proportionally to their length."""

    def __init__(self, clips, variants=(), window=16, samples_per_epoch=10000, occ_aug_p=0.0, seed=0):
        if not clips:
            raise ValueError("no clips")
        self.clips, self.variants = clips, [v for v in variants if v.episodes]
        self.window, self.samples, self.seed = window, samples_per_epoch, seed
        self.occ_aug_p = occ_aug_p if self.variants else 0.0
        lengths = np.array([len(c) for c in clips], dtype=np.float64)
        self.p = lengths / lengths.sum()
        self.epoch = 0

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = np.random.default_rng([self.seed, self.epoch, index])
        if self.occ_aug_p > 0 and rng.random() < self.occ_aug_p:
            clip = self.variants[rng.integers(len(self.variants))]
            ep = clip.episodes[rng.integers(len(clip.episodes))]
            room = self.window - ep["length"]
            lead = int(rng.integers(1, room)) if room >= 2 else 0
            start = int(np.clip(ep["start"] - lead, 0, max(0, len(clip) - self.window)))
        else:
            clip = self.clips[rng.choice(len(self.clips), p=self.p)]
            start = int(rng.integers(0, max(1, len(clip) - self.window + 1)))
        return clip.window(start, self.window)


def full_clip(clip):
    """The whole clip as a batch of one (validation / evaluation)."""
    sample = clip.window(0, len(clip))
    return {k: ({kk: vv[None] for kk, vv in v.items()} if isinstance(v, dict) else v[None])
            for k, v in sample.items()}
