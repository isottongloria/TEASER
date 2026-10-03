"""FLAME face regions and canonical vertices for the temporal losses and metrics.

Regions (vertex indices into FLAME 2020's 5023 vertices), from the
``FLAME_masks.pkl`` shipped in ``assets/FLAME_masks`` plus the support of
TEASER's eyelid blendshapes:

- ``mouth`` = ``lips``
- ``eyes``  = (``eye_region`` U eyelid support) - eyeballs
- ``rest``  = ``face`` - mouth - eyes
- ``face``  = mouth U eyes U rest

Eyeballs are left out everywhere: TEASER predicts no gaze. The eyelid
support is every vertex an eyelid blendshape moves by more than 5 % of its
largest displacement (~310 per side).

Canonical vertices: FLAME with one fixed shared identity (the template, all
shape coefficients 0) and global / neck / eye rotations at zero, so only
expression, jaw and eyelids move the mesh. Comparing two predictions there
isolates exactly what the temporal module changes.

``region_vertex_ids`` needs only numpy and works in RGB2SMPLX's env too;
``CanonicalFlame`` needs TEASER's FLAME (imported lazily).
"""

import os
import pickle
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
EYELID_SUPPORT_FRAC = 0.05


def region_vertex_ids(assets_dir=REPO_ROOT / "assets"):
    """{'mouth', 'eyes', 'rest', 'face'} -> sorted unique int64 vertex indices."""
    assets_dir = Path(assets_dir)
    with open(assets_dir / "FLAME_masks/FLAME_masks.pkl", "rb") as handle:
        masks = pickle.load(handle, encoding="latin1")
    as_set = lambda key: set(np.asarray(masks[key]).astype(np.int64).tolist())

    eyelid = set()
    for name in ("l_eyelid.npy", "r_eyelid.npy"):
        displacement = np.linalg.norm(np.load(assets_dir / name), axis=1)
        eyelid |= set(np.where(displacement > EYELID_SUPPORT_FRAC * displacement.max())[0].tolist())

    eyeballs = as_set("left_eyeball") | as_set("right_eyeball")
    mouth = as_set("lips") - eyeballs
    eyes = ((as_set("eye_region") | eyelid) - eyeballs) - mouth
    rest = as_set("face") - mouth - eyes - eyeballs
    to_array = lambda ids: np.array(sorted(ids), dtype=np.int64)
    return {"mouth": to_array(mouth), "eyes": to_array(eyes), "rest": to_array(rest),
            "face": to_array(mouth | eyes | rest)}


@contextmanager
def _in_repo_root():
    # TEASER's FLAME loads several assets by paths relative to the working directory.
    previous = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


class CanonicalFlame(torch.nn.Module):
    """(expression (..., 50), jaw (..., 3), eyelid (..., 2)) -> canonical face vertices (..., V, 3).

    Differentiable, any leading dimensions. ``vertex_ids`` defaults to the
    ``face`` region; pass ``None`` for all 5023 vertices.
    """

    def __init__(self, n_exp=50, vertex_ids="face"):
        super().__init__()
        from src.FLAME.FLAME import FLAME

        with _in_repo_root():
            flame = FLAME(n_exp=n_exp, n_shape=300)
        self.n_exp = n_exp
        dtype = flame.dtype
        # Only what LBS needs; FLAME's landmark machinery is not used.
        self.register_buffer("v_template", flame.v_template.clone())
        self.register_buffer("shapedirs", flame.shapedirs[:, :, 300:300 + n_exp].clone())
        self.register_buffer("posedirs", flame.posedirs.clone())
        self.register_buffer("J_regressor", flame.J_regressor.clone())
        self.register_buffer("lbs_weights", flame.lbs_weights.clone())
        self.register_buffer("parents", flame.parents.clone())
        self.register_buffer("l_eyelid", flame.l_eyelid[0].clone())
        self.register_buffer("r_eyelid", flame.r_eyelid[0].clone())
        self.register_buffer("eye_pose", flame.eye_pose.detach().clone())
        self.dtype = dtype
        if isinstance(vertex_ids, str):
            vertex_ids = region_vertex_ids()[vertex_ids]
        self.register_buffer("vertex_ids", None if vertex_ids is None
                             else torch.as_tensor(vertex_ids, dtype=torch.long))

    def forward(self, expression, jaw, eyelid=None):
        from src.FLAME.lbs import lbs

        lead = expression.shape[:-1]
        expression = expression.reshape(-1, expression.shape[-1])
        jaw = jaw.reshape(-1, 3)
        n = expression.shape[0]
        # With zero shape, betas = expression only and shapedirs = the expression block.
        zeros3 = torch.zeros(n, 3, dtype=expression.dtype, device=expression.device)
        full_pose = torch.cat([zeros3, zeros3, jaw, self.eye_pose.expand(n, -1)], dim=1)
        vertices, _ = lbs(expression, full_pose, self.v_template.unsqueeze(0).expand(n, -1, -1),
                          self.shapedirs, self.posedirs, self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype)
        if eyelid is not None:
            eyelid = eyelid.reshape(-1, 2)
            vertices = vertices + self.r_eyelid * eyelid[:, 1, None, None] \
                                + self.l_eyelid * eyelid[:, 0, None, None]
        if self.vertex_ids is not None:
            vertices = vertices[:, self.vertex_ids]
        return vertices.reshape(*lead, vertices.shape[-2], 3)

    def region_index(self, region):
        """Positions of ``region``'s vertices inside this module's output."""
        regions = region_vertex_ids()
        ids = regions[region]
        if self.vertex_ids is None:
            return np.asarray(ids)
        lookup = {int(v): i for i, v in enumerate(self.vertex_ids.tolist())}
        return np.array([lookup[int(v)] for v in ids if int(v) in lookup], dtype=np.int64)
