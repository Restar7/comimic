"""Gradient-enabled GR00T N1.7 flow-matching sampler.

The upstream implementation decorates both inference entry points with
``@torch.no_grad()``.  The Euler integration itself is differentiable, so this
module reproduces the exact loop without the decorator and keeps every step in
the autograd graph.

This is the *pathwise gradient / BPTT* variant: gradients flow through all
``num_steps`` Euler steps.  It is deliberately kept as a separate backend so a
Q-VGM / adjoint estimator can be swapped in later without changing the
SuperMimic side.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:  # transformers is optional when only the adapter/tests are used
    from transformers.feature_extraction_utils import BatchFeature
except Exception:  # pragma: no cover
    BatchFeature = dict  # type: ignore[assignment,misc]


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _contains(obj: Any, name: str) -> bool:
    try:
        return name in obj
    except TypeError:
        return hasattr(obj, name)


def _horizon(action_head: nn.Module) -> int:
    return int(_get(action_head, "action_horizon", _get(_get(action_head, "config"), "action_horizon")))


def _action_dim(action_head: nn.Module) -> int:
    return int(_get(action_head, "action_dim", _get(_get(action_head, "config"), "max_action_dim")))


def _num_steps(action_head: nn.Module, num_steps: int | None) -> int:
    steps = num_steps if num_steps is not None else _get(
        action_head, "num_inference_timesteps", _get(_get(action_head, "config"), "num_inference_timesteps")
    )
    steps = int(steps)
    if steps <= 0:
        raise ValueError("num_steps must be positive")
    return steps


def sample_fm_from_features(
    action_head: nn.Module,
    backbone_features: torch.Tensor,
    state_features: torch.Tensor,
    embodiment_id: torch.Tensor,
    backbone_output: Any | None = None,
    action_input: Any | None = None,
    *,
    initial_noise: torch.Tensor | None = None,
    num_steps: int | None = None,
    options: dict[str, Any] | None = None,
    return_features: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable equivalent of ``get_action_with_features``.

    ``backbone_features`` must already have been passed through
    ``action_head.process_backbone_output`` (that is what ``_encode_features``
    does).  Use :func:`sample_fm_with_grad` for the high-level entry point.
    """
    if not torch.is_grad_enabled():
        raise RuntimeError(
            "sample_fm_from_features was called under torch.no_grad(); the joint "
            "training graph would be cut. Call it from a normal grad-enabled context."
        )
    vl_embeds = backbone_features
    batch_size = vl_embeds.shape[0]
    horizon = _horizon(action_head)
    action_dim = _action_dim(action_head)
    steps = _num_steps(action_head, num_steps)
    expected = (batch_size, horizon, action_dim)
    if initial_noise is None:
        actions = torch.randn(expected, dtype=vl_embeds.dtype, device=vl_embeds.device)
    else:
        if tuple(initial_noise.shape) != expected:
            raise ValueError(f"initial_noise must be {expected}, got {tuple(initial_noise.shape)}")
        actions = initial_noise.to(device=vl_embeds.device, dtype=vl_embeds.dtype)

    config = _get(action_head, "config")
    use_alt = bool(_get(config, "use_alternate_vl_dit", False))
    add_pos = bool(_get(config, "add_pos_embed", False))
    buckets = int(_get(action_head, "num_timestep_buckets", _get(config, "num_timestep_buckets")))
    dt = 1.0 / steps
    vel_strength = torch.ones_like(actions)

    # Optional RTC/inpainting path, copied from the upstream implementation.
    # Training normally passes no `action` key and therefore ignores this block.
    if action_input is not None and _contains(action_input, "action"):
        if options is None:
            raise ValueError("RTC options are required when action_input contains 'action'")
        horizon_before_padding = int(options["action_horizon"])
        overlap = int(options["rtc_overlap_steps"])
        frozen = int(options["rtc_frozen_steps"])
        actions[:, :overlap, :] = action_input["action"][
            :, horizon_before_padding - overlap : horizon_before_padding, :
        ]
        vel_strength[:, :frozen, :] = 0.0
        intermediate = overlap - frozen
        if intermediate > 0:
            t = torch.linspace(0.0, 1.0, intermediate + 2, device=vl_embeds.device)
            ramp = 1 - torch.exp(-float(options["rtc_ramp_rate"]) * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)
            vel_strength[:, frozen:overlap, :] = ramp[1:-1][None, :, None]

    for index in range(steps):
        t_cont = index / float(steps)
        timestep = int(t_cont * buckets)
        timesteps = torch.full((batch_size,), timestep, dtype=torch.long, device=vl_embeds.device)
        action_embeddings = action_head.action_encoder(actions, timesteps, embodiment_id)
        if add_pos:
            positions = torch.arange(
                action_embeddings.shape[1], dtype=torch.long, device=vl_embeds.device
            )
            action_embeddings = action_embeddings + action_head.position_embedding(positions).unsqueeze(0)
        joined = torch.cat((state_features, action_embeddings), dim=1)
        kwargs = dict(
            hidden_states=joined,
            encoder_hidden_states=vl_embeds,
            timestep=timesteps,
        )
        if use_alt:
            if backbone_output is None:
                raise ValueError("backbone_output is required for AlternateVLDiT")
            kwargs["image_mask"] = backbone_output.image_mask
            kwargs["backbone_attention_mask"] = backbone_output.backbone_attention_mask
        hidden = action_head.model(**kwargs)
        predicted = action_head.action_decoder(hidden, embodiment_id)
        predicted_velocity = predicted[:, -horizon:]
        actions = actions + dt * predicted_velocity * vel_strength

    if return_features:
        return actions, vl_embeds, state_features
    return actions


def sample_fm_with_grad(
    action_head: nn.Module,
    backbone_output: Any,
    action_input: Any,
    *,
    initial_noise: torch.Tensor | None = None,
    num_steps: int | None = None,
    options: dict[str, Any] | None = None,
    return_features: bool = False,
):
    """Differentiable equivalent of ``Gr00tN1d7ActionHead.get_action``.

    ``action_head`` is the native head; ``backbone_output`` and ``action_input``
    are the native ``BatchFeature`` objects returned by ``Gr00tN1d7.prepare_input``.
    """
    features = action_head._encode_features(backbone_output, action_input)
    return sample_fm_from_features(
        action_head,
        features.backbone_features,
        features.state_features,
        action_input.embodiment_id,
        backbone_output=backbone_output,
        action_input=action_input,
        initial_noise=initial_noise,
        num_steps=num_steps,
        options=options,
        return_features=return_features,
    )


def sample_model_with_grad(
    gr00t_model: nn.Module,
    model_inputs: dict,
    *,
    initial_noise: torch.Tensor | None = None,
    num_steps: int | None = None,
    options: dict[str, Any] | None = None,
    return_features: bool = False,
):
    """Full VLA path: preprocess -> VLM backbone -> differentiable FM sampling.

    ``model_inputs`` must be a GR00T-ready batch (the same dict accepted by
    ``Gr00tN1d7.forward``).  The dict is shallow-copied because
    ``prepare_input`` may pop the raw ``vlm_content`` entry.
    """
    if not torch.is_grad_enabled():
        raise RuntimeError("sample_model_with_grad must be called with grad enabled")
    backbone_inputs, action_inputs = gr00t_model.prepare_input(dict(model_inputs))
    backbone_output = gr00t_model.backbone(backbone_inputs)
    return sample_fm_with_grad(
        gr00t_model.action_head,
        backbone_output,
        action_inputs,
        initial_noise=initial_noise,
        num_steps=num_steps,
        options=options,
        return_features=return_features,
    )


class Gr00tTokenSource(nn.Module):
    """Drop-in token source for the joint trainer.

    It is intentionally stateless apart from the wrapped GR00T model and the
    sampler settings, so the same object can be used for training and for
    offline token caching.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        num_steps: int | None = None,
        initial_noise: torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.register_buffer(
            "fixed_noise", initial_noise if initial_noise is not None else torch.empty(0), persistent=True
        )

    def forward(self, model_inputs: dict) -> torch.Tensor:
        noise = self.fixed_noise if self.fixed_noise.numel() else None
        return sample_model_with_grad(
            self.model,
            model_inputs,
            initial_noise=noise,
            num_steps=self.num_steps,
        )


def trainable_parameter_groups(module: nn.Module, lr: float) -> list[dict]:
    return [{"params": [p for p in module.parameters() if p.requires_grad], "lr": lr}]


__all__ = [
    "Gr00tTokenSource",
    "sample_fm_from_features",
    "sample_fm_with_grad",
    "sample_model_with_grad",
    "trainable_parameter_groups",
]
