"""K-candidate sampling on the native N1.7 flow sampler (M1).

The official sampler draws its initial noise inside
``Gr00tN1d7ActionHead.get_action_with_features`` and its signature is pinned by
``tests/scripts/deployment/test_dit_only_forward_signature.py``.  We therefore do
**not** patch upstream.  Instead:

* backbone/state features are encoded exactly once (``_encode_features``);
* the identical observation is replicated ``K`` times along the batch axis, so
  the internally drawn ``randn(B*K, H, D)`` *is* K independent Gaussian
  initializations of the same conditioning;
* one batched sampler call returns ``[B*K, H, D]`` which reshapes to
  ``[B, K, H, D]``;
* decoding to the SONIC78 layout stays in the official processor.

RTC is opt-in and only reachable through :class:`mosaic_n17.policy.rtc.RtcRequest`:
all ``K`` candidates share the same committed chunk tail (replication is
interleaved, so the injected prefix is identical per row) and the frozen ticks
are pinned by the official sampler.  A token prefix is still *not* a joint-space
command: it is only ever the already-committed 78D block re-encoded in the
model's normalized space, never an invented motion-token target.  Candidate
selection must not rewrite executed ticks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from ..contracts.sonic78 import ACTION78_DIM, SONIC78_V1, Sonic78Spec
from .decode import decode_candidates
from .rtc import RtcRequest, RtcWindow, validate_prefix_tensor


@dataclass(frozen=True)
class CandidateSet:
    """Validated ``[B, K, H, 78]`` candidates plus masks, seeds and digests."""

    candidates: np.ndarray
    motion_valid: np.ndarray
    hand_valid: np.ndarray
    seeds: tuple[int, ...]
    token_sha256: tuple[tuple[str, ...], ...]

    @property
    def batch_size(self) -> int:
        return int(self.candidates.shape[0])

    @property
    def num_candidates(self) -> int:
        return int(self.candidates.shape[1])


def _leading_size(value: torch.Tensor) -> int:
    return int(value.shape[0]) if value.dim() > 0 else -1


def replicate_action_input(action_input, batch_size: int, K: int):
    """Repeat every batch-aligned tensor ``K`` times along dim 0 (interleaved)."""

    from transformers.feature_extraction_utils import BatchFeature

    if not isinstance(action_input, BatchFeature):
        action_input = BatchFeature(data=dict(action_input))
    data = {}
    for key, value in action_input.items():
        if isinstance(value, torch.Tensor) and _leading_size(value) == batch_size:
            data[key] = torch.repeat_interleave(value, K, dim=0)
        else:
            data[key] = value
    return BatchFeature(data=data)


def _head_int(action_head, field: str, *, config_alias: str | None = None) -> int:
    """Read an int from the action head itself, then from its config.

    ``Gr00tN1d7ActionHead`` stores ``action_dim``/``action_horizon`` as instance
    attributes (``gr00t_n1d7.py:62-63``) and its config only has
    ``max_action_dim``/``action_horizon``; duck-typed stubs in tests expose the
    same numbers through ``config``.  Both shapes must resolve, and a missing
    value must raise instead of silently sampling a wrong width.
    """

    value = getattr(action_head, field, None)
    if value is not None:
        return int(value)
    config = getattr(action_head, "config", None)
    if config is not None:
        value = getattr(config, field, None)
        if value is None and config_alias is not None:
            value = getattr(config, config_alias, None)
        if value is not None:
            return int(value)
    alias = f" or config.{config_alias}" if config_alias else ""
    raise AttributeError(f"action head must expose {field}{alias}")


def _action_dim(action_head) -> int:
    return _head_int(action_head, "action_dim", config_alias="max_action_dim")


def _action_horizon(action_head) -> int:
    return _head_int(action_head, "action_horizon")


def encode_once(action_head, backbone_output, action_input):
    """Run ``_encode_features`` once, refusing a silent fallback if it disappears."""

    encode = getattr(action_head, "_encode_features", None)
    if encode is None:
        raise AttributeError(
            "Gr00tN1d7ActionHead._encode_features is missing; the pinned sampler contract changed"
        )
    return encode(backbone_output, action_input)


def sample_from_features(
    action_head,
    features,
    action_input,
    backbone_output,
    *,
    K: int,
    options=None,
) -> torch.Tensor:
    """One batched official sampler call over ``B*K`` replicated observations."""

    if isinstance(K, bool) or not isinstance(K, int) or K <= 0:
        raise ValueError(f"K must be a positive int, got {K!r}")
    window = RtcWindow.from_model_options(options) if options is not None else None
    if window is not None:
        if "action" not in action_input:
            raise ValueError(
                "RTC options require action_input['action'] (the committed chunk in the "
                "model's normalized padded space); refusing to sample without a prefix"
            )
        expected = (window.horizon, _action_dim(action_head))
        if tuple(action_input["action"].shape[1:]) != expected:
            raise ValueError(
                f"prefix must be [B, {expected[0]}, {expected[1]}], "
                f"got {tuple(action_input['action'].shape)}"
            )
    elif "action" in action_input:
        raise ValueError(
            "action_input carries a committed chunk but no RTC options; the action head would "
            "assert on a missing options dict"
        )
    backbone_features = features.backbone_features
    state_features = features.state_features
    batch_size = int(backbone_features.shape[0])
    if int(state_features.shape[0]) != batch_size:
        raise ValueError("backbone and state features disagree on batch size")
    replicated = replicate_action_input(action_input, batch_size, K)
    embodiment_id = replicated.get("embodiment_id")
    if embodiment_id is None:
        raise ValueError("action_input must carry embodiment_id")
    # ``backbone_output`` carries the batch-aligned image/attention masks that the
    # alternate-VL DiT consumes; reusing the un-replicated mapping makes the
    # model reshape a B=1 mask into B*K and crash (or worse, silently broadcast).
    if K > 1:
        if not isinstance(backbone_output, Mapping):
            raise TypeError(
                "K>1 needs a mapping backbone_output (BatchFeature) so its batch-aligned "
                f"masks can be replicated, got {type(backbone_output).__name__}"
            )
        backbone_output = replicate_action_input(backbone_output, batch_size, K)
    output = action_head.get_action_with_features(
        backbone_features=torch.repeat_interleave(backbone_features, K, dim=0),
        state_features=torch.repeat_interleave(state_features, K, dim=0),
        embodiment_id=embodiment_id,
        backbone_output=backbone_output,
        action_input=replicated,
        options=options,
    )
    prediction = output["action_pred"]
    horizon, dim = _action_horizon(action_head), _action_dim(action_head)
    if tuple(prediction.shape) != (batch_size * K, horizon, dim):
        raise ValueError(
            f"sampler returned {tuple(prediction.shape)}, expected {(batch_size * K, horizon, dim)}"
        )
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("sampler produced non-finite actions")
    return prediction.reshape(batch_size, K, horizon, dim)


def sample_raw_candidates(
    model, inputs: dict, *, K: int, rtc: RtcRequest | None = None
) -> torch.Tensor:
    """Sample ``[B,K,H,action_dim]`` normalized actions for one model-ready batch.

    With ``rtc`` the committed chunk is injected into ``inputs["action"]``; every
    candidate then shares that prefix because ``replicate_action_input`` repeats
    the batch axis after the injection.
    """

    if isinstance(K, bool) or not isinstance(K, int) or K <= 0:
        raise ValueError(f"K must be a positive int, got {K!r}")
    model_inputs = dict(inputs)
    options = None
    if rtc is not None:
        rtc.window.validate()
        prefix = validate_prefix_tensor(
            rtc.previous_action,
            rtc.window,
            action_dim=_action_dim(model.action_head),
        )
        model_inputs["action"] = torch.from_numpy(prefix)
        options = rtc.window.to_model_options()
    elif "action" in model_inputs:
        raise ValueError(
            "inputs already carry an 'action' field but no RTC window; pass the window "
            "explicitly instead of letting the sampler guess"
        )
    backbone_inputs, action_inputs = model.prepare_input(model_inputs)
    backbone_output = model.backbone(backbone_inputs)
    features = encode_once(model.action_head, backbone_output, action_inputs)
    return sample_from_features(
        model.action_head,
        features,
        action_inputs,
        backbone_output,
        K=K,
        options=options,
    )


def build_candidate_set(
    candidates: np.ndarray,
    *,
    seeds: tuple[int, ...] | None = None,
    spec: Sonic78Spec = SONIC78_V1,
    provenance: str | None = None,
) -> CandidateSet:
    """Validate decoded candidates and bind their per-candidate digests."""

    array = np.asarray(candidates, dtype=np.float64)
    if array.ndim != 4 or array.shape[-1] != ACTION78_DIM:
        raise ValueError(f"candidates must be [B,K,H,{ACTION78_DIM}], got {array.shape}")
    spec.validate_block(array, provenance=provenance, require_provenance=provenance is not None)
    motion_valid = np.ones(array.shape[:-1], dtype=bool)
    hand_valid = np.ones(array.shape[:-1] + (14,), dtype=bool)
    hashes = tuple(
        tuple(hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in batch)
        for batch in array
    )
    resolved_seeds = tuple(range(array.shape[1])) if seeds is None else tuple(seeds)
    if len(resolved_seeds) != array.shape[1]:
        raise ValueError("seeds must contain one entry per candidate")
    return CandidateSet(
        candidates=array,
        motion_valid=motion_valid,
        hand_valid=hand_valid,
        seeds=resolved_seeds,
        token_sha256=hashes,
    )


def sample_candidates_and_normalized(
    model,
    processor,
    inputs: dict,
    *,
    K: int,
    rtc: RtcRequest | None = None,
    spec: Sonic78Spec = SONIC78_V1,
    provenance: str | None = None,
    embodiment_tag: str = "UNITREE_G1_SONIC",
    seeds: tuple[int, ...] | None = None,
) -> tuple[CandidateSet, np.ndarray]:
    """Sample K candidates and also return their ``[B,K,H,D]`` normalized block.

    The normalized block is what the next RTC window must reuse: decoding to 78D
    and re-encoding would introduce a float round trip, so callers that commit a
    candidate keep this tensor instead.
    """

    raw = sample_raw_candidates(model, inputs, K=K, rtc=rtc)
    normalized = raw.detach().float().cpu().numpy()
    decoded = decode_candidates(processor, normalized, spec=spec, embodiment_tag=embodiment_tag)
    candidate_set = build_candidate_set(decoded, seeds=seeds, spec=spec, provenance=provenance)
    return candidate_set, normalized


def sample_candidates(
    model,
    processor,
    inputs: dict,
    *,
    K: int,
    spec: Sonic78Spec = SONIC78_V1,
    provenance: str | None = None,
    embodiment_tag: str = "UNITREE_G1_SONIC",
    seeds: tuple[int, ...] | None = None,
) -> CandidateSet:
    """Sample + decode + validate K candidates for one model-ready batch."""

    candidate_set, _ = sample_candidates_and_normalized(
        model,
        processor,
        inputs,
        K=K,
        spec=spec,
        provenance=provenance,
        embodiment_tag=embodiment_tag,
        seeds=seeds,
    )
    return candidate_set


__all__ = [
    "CandidateSet",
    "build_candidate_set",
    "encode_once",
    "replicate_action_input",
    "sample_candidates",
    "sample_candidates_and_normalized",
    "sample_from_features",
    "sample_raw_candidates",
]
