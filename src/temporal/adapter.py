"""Temporal adapter on TEASER's pooled features (TEMPORAL_README.md 5.3).

    adapter = TemporalAdapter(feat_dim=960, arch='transformer', fusion='gated', use_cond=True)
    f_tilde, aux = adapter(f, c=c, frame_mask=mask)   # f (B,T,F), c (B,T,2), mask (B,T) bool

The output feature goes through TEASER's own (frozen) heads. The core's
output layer is zero-initialised, so at initialisation ``f_tilde == f``
exactly (for unmasked frames) in every fusion mode: training starts from
TEASER and only moves away from it where the losses ask.

Masked frames (training-time frame masking, or inference-time frames with
high occlusion) have their feature replaced by a learned ``[MASK]`` token
*before* the core and as the base of the fusion, so the core has to rebuild
them from their neighbours.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHS = ("transformer", "tcn", "gru")
FUSIONS = ("residual", "gated", "replace")


class SinusoidalPosition(nn.Module):
    """Relative-length-agnostic sinusoidal positions, so any window length works."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
        freq = torch.exp(torch.arange(0, self.d_model, 2, device=x.device, dtype=x.dtype)
                         * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(x.shape[1], self.d_model, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(t * freq)
        pe[:, 1::2] = torch.cos(t * freq[: pe[:, 1::2].shape[1]])
        return x + pe


class TransformerCore(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dropout, causal):
        super().__init__()
        self.position = SinusoidalPosition(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.causal = causal

    def forward(self, x):
        mask = None
        if self.causal:
            t = x.shape[1]
            mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        return self.norm(self.encoder(self.position(x), mask=mask))


class TCNBlock(nn.Module):
    def __init__(self, d_model, kernel_size, dilation, dropout, causal):
        super().__init__()
        span = (kernel_size - 1) * dilation
        self.pad = (span, 0) if causal else (span // 2, span - span // 2)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(d_model, d_model, 1)
        # Normalise each frame over its channels only: a norm over time
        # (GroupNorm) would leak future frames into a causal TCN.
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # (B, C, T)
        y = self.conv1(F.pad(x, self.pad, mode="replicate"))
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        y = self.conv2(self.dropout(F.gelu(y)))
        return x + y


class TCNCore(nn.Module):
    """Dilated residual 1D convs; receptive field (k-1)(2^L - 1) + 1 frames."""

    def __init__(self, d_model, n_layers, kernel_size, dropout, causal):
        super().__init__()
        self.blocks = nn.ModuleList(TCNBlock(d_model, kernel_size, 2 ** i, dropout, causal)
                                    for i in range(n_layers))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.norm(x.transpose(1, 2))


class GRUCore(nn.Module):
    def __init__(self, d_model, n_layers, dropout, causal):
        super().__init__()
        hidden = d_model if causal else d_model // 2
        self.gru = nn.GRU(d_model, hidden, n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0, bidirectional=not causal)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.gru(x)[0])


class TemporalAdapter(nn.Module):
    def __init__(self, feat_dim, arch="transformer", d_model=256, n_layers=2, n_heads=4,
                 kernel_size=3, dropout=0.1, causal=False, fusion="gated", mask_token=False,
                 use_cond=False, cond_dim=2):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}")
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}")
        self.feat_dim, self.fusion, self.causal = feat_dim, fusion, causal
        self.use_cond, self.cond_dim = use_cond, cond_dim

        in_dim = feat_dim + (cond_dim if use_cond else 0)
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Linear(in_dim, d_model)
        if arch == "transformer":
            self.core = TransformerCore(d_model, n_layers, n_heads, dropout, causal)
        elif arch == "tcn":
            self.core = TCNCore(d_model, n_layers, kernel_size, dropout, causal)
        else:
            self.core = GRUCore(d_model, n_layers, dropout, causal)
        self.out_proj = nn.Linear(d_model, feat_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        if fusion == "gated":
            gate_in = 2 * feat_dim + (cond_dim if use_cond else 0)
            self.gate = nn.Sequential(nn.Linear(gate_in, d_model), nn.GELU(), nn.Linear(d_model, feat_dim))
        if fusion == "replace":
            # f_tilde = alpha * f + T(f); alpha starts at 1 so the start is still TEASER.
            self.alpha = nn.Parameter(torch.ones(()))
        self.mask_token = nn.Parameter(torch.randn(feat_dim) * 0.02) if mask_token else None

    def forward(self, f, c=None, frame_mask=None):
        """f (B,T,F), c (B,T,cond_dim) or None, frame_mask (B,T) bool or None -> (f_tilde, aux)."""
        if frame_mask is not None and frame_mask.any():
            if self.mask_token is None:
                raise ValueError("frame_mask given but the adapter has no mask token")
            f = torch.where(frame_mask[..., None], self.mask_token.to(f.dtype).expand_as(f), f)
        if self.use_cond:
            if c is None:
                c = torch.zeros(*f.shape[:2], self.cond_dim, device=f.device, dtype=f.dtype)
            x = torch.cat([f, c], dim=-1)
        else:
            x = f
        delta = self.out_proj(self.core(self.in_proj(self.in_norm(x))))

        aux = {"delta": delta}
        if self.fusion == "residual":
            f_tilde = f + delta
        elif self.fusion == "gated":
            gate_in = [f, delta] + ([c] if self.use_cond else [])
            gate = torch.sigmoid(self.gate(torch.cat(gate_in, dim=-1)))
            f_tilde = f + gate * delta
            aux["gate"] = gate
        else:
            f_tilde = self.alpha * f + delta
            aux["alpha"] = self.alpha
        return f_tilde, aux


@torch.no_grad()
def sliding_window(adapter, f, c=None, frame_mask=None, window=16, stride=4, mode="overlap_add"):
    """Run the adapter over a whole clip. f (T,F), c (T,cond) / None, frame_mask (T,) / None -> (T,F).

    The clip is reflection-padded by window//2 on both sides so edge frames
    get context too. ``overlap_add`` (default) blends windows with a Hann
    weight; ``center`` keeps only each window's centre frame (stride 1);
    a causal adapter uses ``last`` (each frame from the window ending on it).
    """
    if mode not in ("overlap_add", "center", "last"):
        raise ValueError(f"unknown mode {mode!r}")
    if adapter.causal and mode != "last":
        mode = "last"
    t_len = f.shape[0]
    half = window // 2
    pad_left, pad_right = (window - 1, 0) if mode == "last" else (half, window - half)

    def pad(x):
        if x is None:
            return None
        squeeze = x.dim() == 1
        x = x[:, None].float() if squeeze else x
        y = x.T[None]  # (1, C, T)
        mode_pad = "reflect" if t_len > max(pad_left, pad_right) else "replicate"
        y = F.pad(y, (pad_left, pad_right), mode=mode_pad)[0].T
        return y[:, 0].bool() if squeeze else y

    fp, cp, mp = pad(f), pad(c), pad(frame_mask)
    total = fp.shape[0]
    starts = list(range(0, total - window + 1, 1 if mode in ("center", "last") else stride))
    if starts[-1] != total - window:
        starts.append(total - window)
    out = torch.zeros_like(fp)
    weight = torch.zeros(total, 1, device=f.device, dtype=f.dtype)
    hann = torch.hann_window(window, periodic=False, device=f.device, dtype=f.dtype).clamp_min(1e-3)[:, None]
    batch = 64
    for k in range(0, len(starts), batch):
        chunk = starts[k:k + batch]
        fb = torch.stack([fp[s:s + window] for s in chunk])
        cb = torch.stack([cp[s:s + window] for s in chunk]) if cp is not None else None
        mb = torch.stack([mp[s:s + window] for s in chunk]) if mp is not None else None
        yb, _ = adapter(fb, cb, mb)
        for s, y in zip(chunk, yb):
            if mode == "overlap_add":
                out[s:s + window] += hann * y
                weight[s:s + window] += hann
            elif mode == "center":
                out[s + half] = y[half]
                weight[s + half] = 1
            else:
                out[s + window - 1] = y[-1]
                weight[s + window - 1] = 1
    out = out / weight.clamp_min(1e-8)
    return out[pad_left:pad_left + t_len]
