"""Temporal TEASER = frozen TEASER features -> adapter -> TEASER's expression head (TEMPORAL_README.md 5.3).

Also the SmoothNet baseline (T3) behind the same interface, so training,
inference and evaluation treat them alike:

    model = build_model(cfg.model, encoder, feature_set)
    params = model(batch)                 # training windows: dict of (B, T, ·)
    params = model.infer_clip(clip_batch, window, stride)   # a whole clip, sliding window

Output params: ``expression`` (·, 50), ``jaw`` (·, 3), ``eyelid`` (·, 2).
"""

import copy

import torch
import torch.nn as nn

from src.temporal.adapter import TemporalAdapter, sliding_window
from src.temporal.postproc import SmoothNet, smoothnet_clip
from src.temporal.split_encoder import (expression_postprocess, feature_dim, split_features)

PARAM_DIMS = (("expression", 50), ("jaw", 3), ("eyelid", 2))


def condition(batch):
    """Occlusion the adapter is told about: real or synthetic, whichever is larger."""
    return torch.maximum(batch["c_real"], batch["c_syn"])


class TemporalTeaser(nn.Module):
    def __init__(self, encoder, feature_set="expr", head="frozen", frame_mask_p=0.0, **adapter_kw):
        super().__init__()
        self.feature_set = feature_set
        self.n_exp = encoder.expression_encoder.n_exp
        self.adapter = TemporalAdapter(feature_dim(feature_set), **adapter_kw)
        # TEASER's expression head; a trainable copy with head='full', frozen otherwise.
        self.head = copy.deepcopy(encoder.expression_encoder.expression_layers)
        self.head.requires_grad_(head == "full")
        self.frame_mask_p = frame_mask_p

    def _params(self, f_tilde):
        raw = self.head(split_features(f_tilde, self.feature_set)["expr"])
        out = expression_postprocess(raw, self.n_exp)
        return {"expression": out["expression_params"], "jaw": out["jaw_params"], "eyelid": out["eyelid_params"]}

    def forward(self, batch):
        frame_mask = None
        if self.training and self.frame_mask_p > 0 and self.adapter.mask_token is not None:
            frame_mask = (torch.rand(batch["valid"].shape, device=batch["valid"].device) < self.frame_mask_p) \
                & batch["valid"]
        f_tilde, aux = self.adapter(batch["feats"], condition(batch), frame_mask)
        params = self._params(f_tilde)
        params["_aux"] = aux
        return params

    @torch.no_grad()
    def infer_clip(self, batch, window=16, stride=4, mode="overlap_add", occ_mask_thresh=None):
        """batch: a full clip as a batch of one (data.full_clip)."""
        frame_mask = None
        if occ_mask_thresh is not None and self.adapter.mask_token is not None:
            frame_mask = condition(batch)[0].max(-1).values > occ_mask_thresh
        f_tilde = sliding_window(self.adapter, batch["feats"][0], condition(batch)[0], frame_mask,
                                 window=window, stride=stride, mode=mode)
        return {k: v[None] for k, v in self._params(f_tilde).items()}


class SmoothNetModel(nn.Module):
    """SmoothNet on TEASER's 55 per-frame parameters, normalised per dimension."""

    def __init__(self, window=16, hidden=128, n_blocks=3, dropout=0.1):
        super().__init__()
        self.net = SmoothNet(window, hidden, n_blocks, dropout)
        self.register_buffer("mean", torch.zeros(55))
        self.register_buffer("std", torch.ones(55))

    def set_normalisation(self, clips):
        stacked = torch.cat([torch.cat([torch.from_numpy(c.input_params[k]) for k, _ in PARAM_DIMS], 1)
                             for c in clips])
        self.mean.copy_(stacked.mean(0))
        self.std.copy_(stacked.std(0).clamp_min(1e-4))

    def _pack(self, params):
        return (torch.cat([params[k] for k, _ in PARAM_DIMS], -1) - self.mean) / self.std

    def _unpack(self, x):
        x = x * self.std + self.mean
        out, start = {}, 0
        for k, d in PARAM_DIMS:
            out[k] = x[..., start:start + d]
            start += d
        return out

    def forward(self, batch):
        return self._unpack(self.net(self._pack(batch["input_params"])))

    @torch.no_grad()
    def infer_clip(self, batch, window=None, stride=4, mode=None, occ_mask_thresh=None):
        x = self._pack({k: v[0] for k, v in batch["input_params"].items()})
        return {k: v[None] for k, v in self._unpack(smoothnet_clip(self.net, x, stride)).items()}


def build_model(model_cfg, encoder, feature_set, window):
    kind = model_cfg.get("kind", "adapter")
    if kind == "smoothnet":
        return SmoothNetModel(window=window, hidden=model_cfg.get("hidden", 128),
                              n_blocks=model_cfg.get("n_blocks", 3), dropout=model_cfg.get("dropout", 0.1))
    if kind != "adapter":
        raise ValueError(f"unknown model kind {kind!r}")
    return TemporalTeaser(
        encoder, feature_set=feature_set, head=model_cfg.head, frame_mask_p=model_cfg.frame_mask_p,
        arch=model_cfg.arch, d_model=model_cfg.d_model, n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads, kernel_size=model_cfg.kernel_size, dropout=model_cfg.dropout,
        causal=model_cfg.causal, fusion=model_cfg.fusion, mask_token=model_cfg.mask_token,
        use_cond=model_cfg.use_cond)
