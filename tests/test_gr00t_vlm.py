"""Tests for the top-level VLM connector (``diff_vla_wbc.gr00t_vlm``).

These tests never download or load the 4.9 GB Cosmos-Reason2-2B weights: the
resolution logic is exercised on temporary directories, and the tensor plumbing
(processor outputs -> backbone -> flow-matching head -> tokens) runs with a fake
backbone plus the *real* (tiny) ``Gr00tN1d7ActionHead``.

The real-weights smoke lives in ``scripts/smoke_vlm.py`` because it needs a GPU
and the local VLM checkpoint.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from transformers.feature_extraction_utils import BatchFeature

from diff_vla_wbc import gr00t_vlm
from diff_vla_wbc.gr00t_fm import sample_model_with_grad
from diff_vla_wbc.gr00t_vlm import (
    DEFAULT_SELECT_LAYER,
    Gr00tVLM,
    Gr00tVLMTokenSource,
    local_vlm_candidates,
    resolve_vlm_source,
    vlm_available,
)
from diff_vla_wbc.gr00t_tiny import ensure_gr00t_importable, tiny_config

_GR00T_IMPORT_ERROR = None
try:
    _Gr00tN1d7Config, Gr00tN1d7ActionHead = ensure_gr00t_importable()
except Exception as exc:  # pragma: no cover - depends on the vendored checkout
    Gr00tN1d7ActionHead = None  # type: ignore[assignment]
    _GR00T_IMPORT_ERROR = exc


def _write_fake_vlm(root: Path, *, weights: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type": "qwen3_vl"}')
    (root / "tokenizer.json").write_text("{}")
    if weights:
        (root / "model.safetensors").write_bytes(b"\0" * 16)
    return root


class VlmResolutionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_env_var_wins_and_requires_weights(self):
        vlm_dir = _write_fake_vlm(self.tmp / "vlm")
        with mock.patch.dict(os.environ, {gr00t_vlm._VLM_ENV: str(vlm_dir)}):
            self.assertEqual(resolve_vlm_source(), (str(vlm_dir), "env"))
            self.assertTrue(vlm_available())

        empty = self.tmp / "empty"
        empty.mkdir()
        with mock.patch.dict(os.environ, {gr00t_vlm._VLM_ENV: str(empty)}):
            with self.assertRaises(FileNotFoundError):
                resolve_vlm_source()
            self.assertFalse(vlm_available())

    def test_checkout_copy_is_preferred_over_cache(self):
        checkout = _write_fake_vlm(self.tmp / "checkout" / "nvidia" / "Cosmos-Reason2-2B")
        with mock.patch.object(gr00t_vlm, "LOCAL_VLM_DIR", checkout):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(gr00t_vlm._VLM_ENV, None)
                self.assertEqual(resolve_vlm_source(), (str(checkout), "checkout"))
            self.assertIn(checkout, local_vlm_candidates())

    def test_explicit_argument_bypasses_resolution(self):
        self.assertEqual(resolve_vlm_source("/nope/Cosmos-Reason2-2B"), ("/nope/Cosmos-Reason2-2B", "argument"))
        with mock.patch.object(gr00t_vlm, "LOCAL_VLM_DIR", self.tmp / "missing"):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(gr00t_vlm._VLM_ENV, None)
                with mock.patch.object(gr00t_vlm, "local_vlm_candidates", lambda: []):
                    self.assertEqual(resolve_vlm_source(), (gr00t_vlm.VLM_REPO_ID, "repo-id"))
                    self.assertFalse(vlm_available())

    def test_default_select_layer_matches_gr00t_truncation(self):
        self.assertEqual(DEFAULT_SELECT_LAYER, 16)


class _FakeProcessor:
    """Minimal stand-in for the Qwen3-VL processor (chat template + tensors)."""

    def __init__(self, tokens: int = 6):
        self.tokens = tokens
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = messages[0]["content"][-1]["text"]
        return f"<image>{text}"

    def __call__(self, text, images=None, return_tensors="pt", padding=True):
        batch = len(text)
        return {
            "input_ids": torch.arange(self.tokens).repeat(batch, 1),
            "attention_mask": torch.ones(batch, self.tokens, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(batch, self.tokens, dtype=torch.long),
            "pixel_values": torch.zeros(batch * 4, 12),
            "image_grid_thw": torch.tensor([[1, 2, 2]]).repeat(batch, 1),
        }


class _FakeBackbone(torch.nn.Module):
    """Returns deterministic features shaped like the Qwen3-VL backbone output."""

    def __init__(self, hidden: int = 16, tokens: int = 6):
        super().__init__()
        self.proj = torch.nn.Linear(1, hidden)
        self.tokens = tokens

    def forward(self, vl_input):
        batch = int(vl_input["input_ids"].shape[0])
        seq = int(vl_input["input_ids"].shape[1])
        features = self.proj(torch.ones(batch, seq, 1, dtype=self.proj.weight.dtype))
        # ``Gr00tN1d7ActionHead.process_backbone_output`` indexes the backbone
        # output like a BatchFeature, so mirror the real backbone exactly.
        return BatchFeature(
            data={
                "backbone_features": features,
                "backbone_attention_mask": torch.ones(batch, seq, dtype=torch.bool),
                "image_mask": torch.zeros(batch, seq, dtype=torch.bool),
            }
        )


@unittest.skipIf(Gr00tN1d7ActionHead is None, f"vendored GR00T head unavailable: {_GR00T_IMPORT_ERROR}")
class Gr00tVLMPlumbingTest(unittest.TestCase):
    def setUp(self):
        self.cfg = tiny_config(action_horizon=12, action_dim=64, backbone_dim=16, state_dim=8)
        self.cfg.state_history_length = 1
        self.head = Gr00tN1d7ActionHead(self.cfg)
        self.backbone = _FakeBackbone(hidden=self.cfg.backbone_embedding_dim)
        self.vlm = Gr00tVLM(
            self.cfg, self.backbone, self.head, processor=_FakeProcessor(), vlm_source="fake"
        )

    def test_build_batch_pads_state_and_sets_embodiment(self):
        batch = self.vlm.build_batch(
            torch.zeros(3, 3, 8, 8), ["pick"] * 2, torch.zeros(2, 5), embodiment_id=7
        )
        self.assertEqual(tuple(batch["state"].shape), (2, 1, self.cfg.max_state_dim))
        self.assertEqual(batch["state"][..., 5:].abs().sum().item(), 0.0)
        self.assertEqual(batch["embodiment_id"].tolist(), [7, 7])
        self.assertIn("mm_token_type_ids", batch)

    def test_build_batch_rejects_state_wider_than_max(self):
        with self.assertRaises(ValueError):
            self.vlm.build_batch(torch.zeros(1, 3, 8, 8), ["pick"], torch.zeros(1, self.cfg.max_state_dim + 1))

    def test_build_batch_accepts_a_single_pil_image(self):
        """A bare PIL image has no ``__len__``; it must broadcast to the batch."""
        from PIL import Image

        image = Image.fromarray(torch.zeros(8, 8, 3, dtype=torch.uint8).numpy())
        batch = self.vlm.build_batch(image, ["pick"] * 3, torch.zeros(3, 4))
        self.assertEqual(batch["state"].shape[0], 3)
        self.assertEqual(len(self.vlm._as_pil_list(image, 3)), 3)

    def test_prepare_input_splits_and_rejects_vlm_content(self):
        batch = self.vlm.build_batch(torch.zeros(2, 3, 8, 8), ["pick"] * 2, torch.zeros(2, 4))
        backbone_inputs, action_inputs = self.vlm.prepare_input(batch)
        self.assertEqual(
            set(backbone_inputs),
            {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"},
        )
        self.assertEqual(set(action_inputs), {"state", "embodiment_id"})
        # ``gr00t_fm`` reads these via attribute access, so they must be BatchFeatures.
        self.assertTrue(hasattr(action_inputs, "embodiment_id"))
        self.assertTrue(hasattr(action_inputs, "state"))
        with self.assertRaises(ValueError):
            self.vlm.prepare_input({"vlm_content": ["x"]})
        with self.assertRaises(KeyError):
            self.vlm.prepare_input({"input_ids": torch.zeros(1, 4)})

    def test_real_head_produces_grad_tokens_from_vlm_features(self):
        torch.manual_seed(3)
        batch = self.vlm.build_batch(torch.zeros(2, 3, 8, 8), ["pick"] * 2, torch.zeros(2, 4))
        tokens = sample_model_with_grad(self.vlm, batch, num_steps=2)
        self.assertEqual(tuple(tokens.shape), (2, self.cfg.action_horizon, self.cfg.max_action_dim))
        self.assertTrue(tokens.requires_grad)
        tokens.square().mean().backward()
        self.assertGreater(float(self.head.model.proj_out_2.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(self.backbone.proj.weight.grad.abs().sum()), 0.0)

    def test_token_source_uses_fixed_noise(self):
        source = Gr00tVLMTokenSource(
            self.vlm,
            num_steps=2,
            initial_noise=torch.randn(2, self.cfg.action_horizon, self.cfg.max_action_dim),
        )
        batch = self.vlm.build_batch(torch.zeros(2, 3, 8, 8), ["pick"] * 2, torch.zeros(2, 4))
        first = source(batch)
        second = source(batch)
        self.assertEqual(tuple(first.shape), (2, self.cfg.action_horizon, self.cfg.max_action_dim))
        self.assertTrue(torch.allclose(first.detach(), second.detach()))


if __name__ == "__main__":
    unittest.main()
