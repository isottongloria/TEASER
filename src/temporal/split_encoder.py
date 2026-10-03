"""TeaserEncoder split into "pooled features" and "heads", without modifying it.

``TeaserEncoder.forward`` runs, per branch, backbone -> global average pool ->
one linear layer (-> clamp/ReLU for the expression branch). The temporal
adapter has to sit between the pool and the linear layer, so this module
re-expresses that forward as two halves:

    feats = extract_features(encoder, img)       # {'expr': (B,960), 'pose': (B,576), 'shape': (B,960)}
    out   = apply_heads(encoder, feats)          # same dict as encoder(img), minus 'token'

using the encoder's own submodules and weights, so nothing is copied but the
few lines of post-processing. ``tests/temporal/test_split_encoder.py`` checks
that ``apply_heads(extract_features(img))`` equals ``encoder(img)`` bit for bit;
if TeaserEncoder's forward ever changes, that test is what catches it.

``apply_heads`` also accepts features with extra leading dimensions, e.g.
(B, T, F) from the adapter.
"""

import torch
import torch.nn.functional as F

# Name used throughout src/temporal -> TeaserEncoder attribute.
BRANCHES = {
    "expr": "expression_encoder",
    "pose": "pose_encoder",
    "shape": "shape_encoder",
}

# Width of the pooled feature, i.e. the last feature map's channels:
# tf_mobilenetv3_large_minimal_100 -> 960, tf_mobilenetv3_small_minimal_100 -> 576.
FEATURE_DIMS = {"expr": 960, "pose": 576, "shape": 960}

# --temporal_feats choices -> branches whose features the adapter sees. Shape
# is per-identity, not per-frame, so it is never an adapter input; its head
# still runs for TEASER's own outputs.
FEATURE_SETS = {
    "expr": ("expr",),
    "expr+pose": ("expr", "pose"),
}


def load_teaser_encoder(checkpoint, device, n_exp=50):
    """TeaserEncoder with the ``teaser_encoder.*`` weights of a full TEASER checkpoint, in eval mode.

    Same loading as RGB2SMPLX's stage and scripts/reconstruct_video_3panel.py.
    """
    from src.teaser_encoder import TeaserEncoder

    encoder = TeaserEncoder(n_exp=n_exp).to(device)
    state_dict = torch.load(str(checkpoint), map_location=device)
    encoder_sd = {k.replace("teaser_encoder.", ""): v
                  for k, v in state_dict.items() if "teaser_encoder" in k}
    if not encoder_sd:
        raise ValueError(f"No 'teaser_encoder.*' keys in {checkpoint}")
    encoder.load_state_dict(encoder_sd)
    return encoder.eval()


def feature_dim(feature_set):
    return sum(FEATURE_DIMS[name] for name in FEATURE_SETS[feature_set])


def _pool(feature_map):
    # Exactly the op sequence of the original encoders' forward.
    return F.adaptive_avg_pool2d(feature_map, (1, 1)).squeeze(-1).squeeze(-1)


def extract_features(encoder, img, names=("expr", "pose", "shape"), spatial=False):
    """Pooled last-stage features of the requested branches for a (B,3,224,224) batch.

    With ``spatial=True`` also returns the pre-pool maps under ``'<name>_map'``
    ((B, C, 7, 7)), for the optional spatial cache.
    """
    out = {}
    for name in names:
        feature_map = getattr(encoder, BRANCHES[name]).encoder(img)[-1]
        out[name] = _pool(feature_map)
        if spatial:
            out[name + "_map"] = feature_map
    return out


def _flat(feature):
    return feature.reshape(-1, feature.shape[-1]), feature.shape[:-1]


def _unflat(tensor, lead):
    return tensor.reshape(*lead, tensor.shape[-1])


def pose_head(encoder, feature):
    flat, lead = _flat(feature)
    pose_cam = encoder.pose_encoder.pose_cam_layers(flat).reshape(flat.size(0), -1)
    return {
        "pose_params": _unflat(pose_cam[..., :3], lead),
        "cam": _unflat(pose_cam[..., 3:], lead),
    }


def shape_head(encoder, feature):
    flat, lead = _flat(feature)
    parameters = encoder.shape_encoder.shape_layers(flat).reshape(flat.size(0), -1)
    return {"shape_params": _unflat(parameters, lead)}


def expression_head_raw(encoder, feature):
    """The expression branch's linear output (n_exp + 5) before clamp/ReLU."""
    flat, lead = _flat(feature)
    raw = encoder.expression_encoder.expression_layers(flat).reshape(flat.size(0), -1)
    return _unflat(raw, lead)


def expression_postprocess(raw, n_exp):
    """ExpressionEncoder.forward's post-processing, on the raw linear output."""
    return {
        "expression_params": raw[..., :n_exp],
        "eyelid_params": torch.clamp(raw[..., n_exp:n_exp + 2], 0, 1),
        "jaw_params": torch.cat([F.relu(raw[..., n_exp + 2].unsqueeze(-1)),
                                 torch.clamp(raw[..., n_exp + 3:n_exp + 5], -.2, .2)], dim=-1),
    }


def expression_head(encoder, feature):
    raw = expression_head_raw(encoder, feature)
    return expression_postprocess(raw, encoder.expression_encoder.n_exp)


_HEADS = {"pose": pose_head, "shape": shape_head, "expr": expression_head}


def apply_heads(encoder, feats):
    """TeaserEncoder's outputs from pooled features (any subset of expr/pose/shape).

    Keys follow ``TeaserEncoder.forward``'s insertion order (pose, shape,
    expression); ``'token'`` is not produced (it needs the token encoder's
    own multi-scale maps and only feeds the training-time generator).
    """
    out = {}
    for name in ("pose", "shape", "expr"):
        if name in feats:
            out.update(_HEADS[name](encoder, feats[name]))
    return out


def split_forward(encoder, img):
    """``TeaserEncoder.forward`` reassembled from the two halves (for the identity test)."""
    out = apply_heads(encoder, extract_features(encoder, img))
    out["token"] = encoder.token_encoder(img)
    return out


def concat_features(feats, feature_set):
    """Concatenate the branches of ``feature_set`` along the last dim (adapter input)."""
    return torch.cat([feats[name] for name in FEATURE_SETS[feature_set]], dim=-1)


def split_features(flat, feature_set):
    """Inverse of ``concat_features``."""
    out, start = {}, 0
    for name in FEATURE_SETS[feature_set]:
        out[name] = flat[..., start:start + FEATURE_DIMS[name]]
        start += FEATURE_DIMS[name]
    return out
