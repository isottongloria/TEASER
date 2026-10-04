"""Adapter, losses, regions, canonical FLAME and baselines (TEMPORAL_README.md 11).

    cd <TEASER root> && PYTHONPATH=. python -m unittest discover -s tests/temporal -v
"""

import itertools
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from src.temporal import losses as L  # noqa: E402
from src.temporal import postproc  # noqa: E402
from src.temporal.adapter import ARCHS, FUSIONS, TemporalAdapter, sliding_window  # noqa: E402
from src.temporal.flame_regions import CanonicalFlame, region_vertex_ids  # noqa: E402

F_DIM = 32


def _adapter(arch="transformer", fusion="gated", **kw):
    torch.manual_seed(0)
    return TemporalAdapter(F_DIM, arch=arch, d_model=16, n_layers=2, n_heads=2, fusion=fusion, **kw).eval()


class AdapterTest(unittest.TestCase):
    def test_shapes_and_identity_at_init(self):
        f = torch.randn(3, 12, F_DIM)
        c = torch.rand(3, 12, 2)
        for arch, fusion, causal, use_cond in itertools.product(ARCHS, FUSIONS, (False, True), (False, True)):
            with self.subTest(arch=arch, fusion=fusion, causal=causal, cond=use_cond):
                adapter = _adapter(arch, fusion, causal=causal, use_cond=use_cond)
                out, aux = adapter(f, c)
                self.assertEqual(out.shape, f.shape)
                self.assertTrue(torch.equal(out, f), "not identity at init")
                self.assertTrue(torch.equal(aux["delta"], torch.zeros_like(f)))

    def test_moves_once_trained(self):
        """Zero init must not block learning: one step changes the output."""
        adapter = _adapter(fusion="gated").train()
        f = torch.randn(2, 8, F_DIM)
        loss = (adapter(f)[0] - f - 1).pow(2).mean()
        loss.backward()
        self.assertGreater(adapter.out_proj.weight.grad.abs().sum().item(), 0)

    def test_causal_ignores_future(self):
        for arch in ARCHS:
            with self.subTest(arch=arch):
                adapter = _adapter(arch, "residual", causal=True)
                torch.nn.init.normal_(adapter.out_proj.weight, std=0.1)
                f = torch.randn(1, 10, F_DIM)
                g = f.clone()
                g[:, 6:] += torch.randn_like(g[:, 6:])
                a, b = adapter(f)[0], adapter(g)[0]
                self.assertTrue(torch.allclose(a[:, :6], b[:, :6], atol=1e-5))
                self.assertFalse(torch.allclose(a[:, 6:], b[:, 6:], atol=1e-3))

    def test_mask_token(self):
        adapter = _adapter(mask_token=True)
        f = torch.randn(2, 8, F_DIM)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[:, 3] = True
        out, _ = adapter(f, frame_mask=mask)
        # At init delta = 0, so a masked frame becomes the token, the others stay.
        self.assertTrue(torch.allclose(out[:, 3], adapter.mask_token.expand(2, -1)))
        self.assertTrue(torch.equal(out[:, :3], f[:, :3]))
        with self.assertRaises(ValueError):
            _adapter(mask_token=False)(f, frame_mask=mask)

    def test_sliding_window(self):
        f = torch.randn(37, F_DIM)
        for mode, causal in (("overlap_add", False), ("center", False), ("last", True)):
            with self.subTest(mode=mode):
                adapter = _adapter(causal=causal)
                out = sliding_window(adapter, f, window=16, stride=4, mode=mode)
                self.assertEqual(out.shape, f.shape)
                self.assertTrue(torch.allclose(out, f, atol=1e-5))
        short = torch.randn(5, F_DIM)  # shorter than the window
        self.assertTrue(torch.allclose(sliding_window(_adapter(), short, window=16), short, atol=1e-5))


class LossTest(unittest.TestCase):
    def test_real_occlusion_weight(self):
        c = torch.tensor([0.0, 0.3, 0.6, float("nan"), 1.0])
        w = L.real_occlusion_weight(c, hard_thresh=0.5)
        self.assertTrue(torch.allclose(w, torch.tensor([1.0, 0.7, 0.0, 1.0, 0.0])))

    def test_really_occluded_frames_are_not_targets(self):
        """Changing the target on a really occluded frame must not change the loss."""
        pred = torch.randn(1, 6, 10, 3)
        target = torch.randn(1, 6, 10, 3)
        c_mouth = torch.zeros(1, 6)
        c_mouth[0, 2] = 0.9
        weights = L.region_weights(c_mouth, torch.zeros(1, 6))
        index = {"mouth": torch.arange(0, 4), "eyes": torch.arange(4, 7), "rest": torch.arange(7, 10)}
        region_w = {"mouth": 1.0, "eyes": 0.0, "rest": 0.0}
        a, _ = L.vertex_target_loss(pred, target, weights, index, region_w)
        target[0, 2] += 100.0
        b, _ = L.vertex_target_loss(pred, target, weights, index, region_w)
        self.assertTrue(torch.allclose(a, b))
        self.assertTrue(torch.isfinite(a))

    def test_param_loss_and_accel(self):
        pred = {"expression": torch.randn(2, 5, 50), "jaw": torch.randn(2, 5, 3), "eyelid": torch.rand(2, 5, 2)}
        weights = L.region_weights(torch.zeros(2, 5), torch.zeros(2, 5))
        zero, _ = L.param_target_loss(pred, pred, weights)
        self.assertEqual(zero.item(), 0.0)
        line = torch.linspace(0, 1, 7)[None, :, None, None].expand(1, 7, 4, 3)
        self.assertLess(L.accel_loss(line).item(), 1e-6)  # constant velocity -> no acceleration
        self.assertGreater(L.accel_loss(torch.randn(1, 7, 4, 3)).item(), 0)

    def test_no_weight_gives_zero_not_nan(self):
        pred, target = torch.randn(1, 4, 10, 3), torch.randn(1, 4, 10, 3)
        weights = {r: torch.zeros(1, 4) for r in L.REGIONS}
        index = {r: torch.arange(10) for r in L.REGIONS}
        loss, _ = L.vertex_target_loss(pred, target, weights, index, {"mouth": 1.0, "eyes": 1.0, "rest": 1.0})
        self.assertEqual(loss.item(), 0.0)

    def test_near_frames(self):
        mask = torch.zeros(1, 9, dtype=torch.bool)
        mask[0, 4] = True
        self.assertEqual(L.near_frames(mask, 2)[0].nonzero().flatten().tolist(), [2, 3, 4, 5, 6])


class FlameTest(unittest.TestCase):
    def test_regions_disjoint(self):
        r = region_vertex_ids()
        self.assertEqual(len(set(r["mouth"]) & set(r["eyes"])), 0)
        self.assertEqual(len(set(r["rest"]) & (set(r["mouth"]) | set(r["eyes"]))), 0)
        self.assertEqual(len(r["face"]), len(r["mouth"]) + len(r["eyes"]) + len(r["rest"]))
        self.assertGreater(len(r["mouth"]), 200)

    def test_canonical_equals_teaser_flame(self):
        """CanonicalFlame == TEASER's FLAME with zero shape and zero global pose."""
        from src.FLAME.FLAME import FLAME

        torch.manual_seed(0)
        flame = FLAME(n_exp=50, n_shape=300)
        canonical = CanonicalFlame(vertex_ids=None)
        expr, jaw, eyelid = torch.randn(4, 50), torch.randn(4, 3) * 0.1, torch.rand(4, 2)
        reference = flame.forward({"shape_params": torch.zeros(4, 300), "expression_params": expr,
                                   "pose_params": torch.zeros(4, 3), "jaw_params": jaw,
                                   "eyelid_params": eyelid})["vertices"]
        ours = canonical(expr, jaw, eyelid)
        self.assertTrue(torch.allclose(ours, reference, atol=1e-6))
        batched = canonical(expr.reshape(2, 2, 50), jaw.reshape(2, 2, 3), eyelid.reshape(2, 2, 2))
        self.assertTrue(torch.allclose(batched.reshape(4, -1, 3), ours, atol=1e-6))


class OcclusionAugTest(unittest.TestCase):
    def test_calibrated_coverage_and_masks(self):
        """The pasted hand reaches the planned coverage on the core frames; c_syn is consistent."""
        from src.temporal import occlusion_aug as oa

        n, h, w = 20, 240, 240
        frames = [np.full((h, w, 3), 150, np.uint8) for _ in range(n)]
        t = np.arange(n)[:, None]
        mouth = np.stack([np.array([[100, 150], [140, 150], [140, 170], [100, 170]], float)] * n)
        eyes = np.stack([np.array([[80, 90], [160, 90], [160, 110], [80, 110]], float)] * n)
        regions = oa.FaceRegions(mouth, eyes)
        plan = [{"start": 6, "length": 5, "entry": 2, "exit": 2, "region": "mouth", "coverage": target,
                 "hand": "procedural_3", "scale": 1.3, "rotation": 10.0, "flip": False,
                 "direction": [0.6, 0.8], "drift": 0.0, "seed": 0} for target in (0.5,)]
        occ = oa.Occluder(frames, regions, plan, oa.HandBank("procedural"))
        for k in range(n):
            occ[k]
        core = occ.c_syn[6:11, 0]
        self.assertTrue(np.all(np.abs(core - 0.5) < 0.08), core)
        self.assertEqual(occ.syn_mask.nonzero()[0].tolist(), list(range(4, 13)))
        self.assertEqual(occ.core_mask.nonzero()[0].tolist(), list(range(6, 11)))
        self.assertTrue(np.all(occ.c_syn[:4] == 0) and np.all(occ.c_syn[13:] == 0))
        self.assertLess(occ.c_syn[4, 0], core.min())  # entry: the hand is still coming in


class BaselineTest(unittest.TestCase):
    def test_savgol_keeps_a_parabola(self):
        t = np.arange(30, dtype=np.float32)[:, None]
        x = 0.01 * t ** 2
        self.assertTrue(np.allclose(postproc.savgol(x)[5:-5], x[5:-5], atol=1e-3))

    def test_hermite_fills_short_episode_only(self):
        t = np.arange(40, dtype=np.float64)
        track = np.stack([np.sin(t / 5), np.cos(t / 7)], 1)
        noisy = track.copy()
        noisy[10:13] += 5.0
        occ = np.zeros(40)
        occ[10:13] = 0.6
        occ[20:37] = 0.6  # 17 frames: too long, left alone
        noisy[20:37] += 5.0
        out, report = postproc.hermite_interp({"x": noisy}, occ)
        self.assertEqual(report, {"n_episodes": 2, "n_corrected": 1,
                                  "n_skipped_too_long": 1, "n_skipped_no_context": 0})
        self.assertLess(np.abs(out["x"][10:13] - track[10:13]).max(), 0.05)
        self.assertTrue(np.array_equal(out["x"][20:37], noisy[20:37].astype(np.float32)))

    def test_smoothnet_identity_at_init(self):
        torch.manual_seed(0)
        model = postproc.SmoothNet(window=16).eval()
        x = torch.randn(3, 16, 55)
        self.assertTrue(torch.equal(model(x), x))
        clip = torch.randn(41, 55)
        self.assertTrue(torch.allclose(postproc.smoothnet_clip(model, clip), clip, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
