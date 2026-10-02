"""Split forward (features -> heads) must equal TeaserEncoder.forward bit for bit.

    cd <TEASER root> && PYTHONPATH=. python -m unittest discover -s tests/temporal -v

Runs on the real checkpoint when present (pretrained_models/TEASER.pt), and
on CUDA too when a GPU is visible.
"""

import os
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from src.teaser_encoder import TeaserEncoder  # noqa: E402
from src.temporal import split_encoder as se  # noqa: E402

CHECKPOINT = REPO_ROOT / "pretrained_models/TEASER.pt"


def _encoders(device):
    torch.manual_seed(0)
    random_init = TeaserEncoder().to(device).eval()
    # The heads are initialised near zero, which would make the clamp/ReLU
    # branches trivially pass; spread the weights so every branch is exercised.
    with torch.no_grad():
        for layer in (random_init.expression_encoder.expression_layers[-1],
                      random_init.pose_encoder.pose_cam_layers[-1],
                      random_init.shape_encoder.shape_layers[-1]):
            layer.weight.normal_(0, 0.5)
            layer.bias.normal_(0, 0.5)
    out = {"random": random_init}
    if CHECKPOINT.is_file():
        out["TEASER.pt"] = se.load_teaser_encoder(CHECKPOINT, device)
    return out


def _devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class SplitEncoderTest(unittest.TestCase):
    def test_split_forward_equals_forward(self):
        for device in _devices():
            for name, encoder in _encoders(device).items():
                with self.subTest(device=device, encoder=name), torch.no_grad():
                    img = torch.rand(3, 3, 224, 224, generator=torch.Generator().manual_seed(1)).to(device)
                    reference = encoder(img)
                    split = se.split_forward(encoder, img)
                    self.assertEqual(list(reference), list(split))
                    for key in reference:
                        self.assertTrue(torch.equal(reference[key], split[key]), f"{key} differs")

    def test_clamps_exercised(self):
        """The random-weight case really hits the clamp/ReLU branches."""
        encoder = _encoders("cpu")["random"]
        with torch.no_grad():
            out = encoder(torch.rand(8, 3, 224, 224, generator=torch.Generator().manual_seed(2)))
        self.assertTrue(((out["eyelid_params"] == 0) | (out["eyelid_params"] == 1)).any())
        self.assertTrue((out["jaw_params"][:, 0] == 0).any())

    def test_heads_accept_leading_dims(self):
        encoder = _encoders("cpu")["random"]
        torch.manual_seed(3)
        feats = {name: torch.randn(2, 5, dim) for name, dim in se.FEATURE_DIMS.items()}
        with torch.no_grad():
            batched = se.apply_heads(encoder, feats)
            flat = se.apply_heads(encoder, {k: v.reshape(10, -1) for k, v in feats.items()})
        for key in flat:
            self.assertEqual(batched[key].shape[:2], (2, 5))
            self.assertTrue(torch.allclose(batched[key].reshape(10, -1), flat[key], atol=1e-6), key)

    def test_concat_split_roundtrip(self):
        torch.manual_seed(4)
        feats = {name: torch.randn(2, 4, dim) for name, dim in se.FEATURE_DIMS.items()}
        for feature_set, names in se.FEATURE_SETS.items():
            flat = se.concat_features(feats, feature_set)
            self.assertEqual(flat.shape[-1], se.feature_dim(feature_set))
            back = se.split_features(flat, feature_set)
            self.assertEqual(tuple(back), names)
            for name in names:
                self.assertTrue(torch.equal(back[name], feats[name]))


if __name__ == "__main__":
    unittest.main()
