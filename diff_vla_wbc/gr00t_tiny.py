"""Helpers for exercising the *real* GR00T action-head code without a 2B VLM.

The full VLA needs a checkpoint and a GPU with enough memory.  For unit tests
and gradient smoke runs we instantiate the actual ``Gr00tN1d7ActionHead`` with a
tiny DiT configuration and feed it random backbone features.  This is the same
Euler sampler / action encoder / decoder code path that the real model uses.

``ensure_gr00t_importable`` tolerates the partial source checkout in this
workspace: the upstream ``gr00t/model/__init__.py`` eagerly imports optional
pipeline/data modules that may be absent from a code-only checkout, so we install
namespace packages for the three package levels and import the model module
directly.  With a complete install, the normal import path is used.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

from .compat import WORKSPACE_ROOT

GR00T_ROOT = WORKSPACE_ROOT / "Isaac-GR00T"


def _drop_gr00t_modules() -> None:
    for name in list(sys.modules):
        if name == "gr00t" or name.startswith("gr00t."):
            del sys.modules[name]


def ensure_gr00t_importable():
    if not GR00T_ROOT.is_dir():
        raise FileNotFoundError(f"Isaac-GR00T checkout not found at {GR00T_ROOT}")
    root = str(GR00T_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

        return Gr00tN1d7Config, Gr00tN1d7ActionHead
    except Exception as first_error:
        _drop_gr00t_modules()
        for name, rel in (
            ("gr00t", Path("gr00t")),
            ("gr00t.model", Path("gr00t/model")),
            ("gr00t.model.gr00t_n1d7", Path("gr00t/model/gr00t_n1d7")),
        ):
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(GR00T_ROOT / rel)]  # type: ignore[attr-defined]
            sys.modules[name] = pkg
        try:
            from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
            from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

            return Gr00tN1d7Config, Gr00tN1d7ActionHead
        except Exception as second_error:
            raise ImportError(
                "Could not import the GR00T action head. The checkout may be incomplete. "
                f"normal import: {first_error!r}; namespace import: {second_error!r}"
            ) from second_error


def tiny_config(
    *,
    action_horizon: int = 12,
    action_dim: int = 64,
    backbone_dim: int = 16,
    state_dim: int = 8,
    hidden: int = 32,
    num_layers: int = 1,
    num_attention_heads: int = 2,
):
    Gr00tN1d7Config, _ = ensure_gr00t_importable()
    return Gr00tN1d7Config(
        model_name="dummy",
        backbone_embedding_dim=backbone_dim,
        max_state_dim=state_dim,
        max_action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_size=hidden,
        input_embedding_dim=hidden,
        state_history_length=1,
        add_pos_embed=False,
        use_vlln=False,
        use_alternate_vl_dit=False,
        diffusion_model_cfg=dict(
            positional_embeddings=None,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            attention_head_dim=max(1, hidden // num_attention_heads),
            norm_type="ada_norm",
            dropout=0.0,
            final_dropout=False,
            output_dim=hidden,
            interleave_self_attention=False,
        ),
        num_inference_timesteps=3,
        max_num_embodiments=4,
        max_seq_len=16,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
    )


def make_tiny_action_head(**kwargs):
    _, Gr00tN1d7ActionHead = ensure_gr00t_importable()
    return Gr00tN1d7ActionHead(tiny_config(**kwargs))


class TinyBackbone(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)

    def forward(self, inputs):
        from transformers.feature_extraction_utils import BatchFeature

        return BatchFeature(data={"backbone_features": self.proj(inputs["backbone_features"])})


class TinyGr00tModel(nn.Module):
    """A 2-layer stand-in for the full VLA that still runs the real action head."""

    def __init__(self, action_head: nn.Module, backbone_dim: int = 16):
        super().__init__()
        self.action_head = action_head
        self.backbone = TinyBackbone(backbone_dim)

    def prepare_input(self, inputs: dict):
        from transformers.feature_extraction_utils import BatchFeature

        backbone_inputs = BatchFeature(
            data={"backbone_features": inputs["backbone_features"]}
        )
        action_inputs = BatchFeature(
            data={
                "state": inputs["state"],
                "embodiment_id": inputs.get(
                    "embodiment_id",
                    torch.zeros(inputs["state"].shape[0], dtype=torch.long),
                ),
            }
        )
        return backbone_inputs, action_inputs

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def tiny_model_inputs(
    batch_size: int = 2,
    *,
    seq_len: int = 5,
    backbone_dim: int = 16,
    state_dim: int = 8,
    device: str | torch.device = "cpu",
) -> dict:
    return {
        "backbone_features": torch.randn(batch_size, seq_len, backbone_dim, device=device),
        "state": torch.randn(batch_size, 1, state_dim, device=device),
        "embodiment_id": torch.zeros(batch_size, dtype=torch.long, device=device),
    }


__all__ = [
    "GR00T_ROOT",
    "TinyGr00tModel",
    "ensure_gr00t_importable",
    "make_tiny_action_head",
    "tiny_config",
    "tiny_model_inputs",
]
