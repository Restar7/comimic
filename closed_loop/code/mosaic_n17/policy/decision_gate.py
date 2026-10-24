"""Candidate arbitration (M4): hard filters first, then typed/stage scores.

The gate never invents a motion command.  It only orders *complete* 78D
candidates that already passed the contract, and it fails closed: when no
candidate is admissible it returns ``fallback=True`` and the caller must keep the
current executed plan (or hold), never send a "best effort" token.

Score terms are explicit and weighted; a term whose weight is zero may be
omitted, but a non-zero term must be supplied, so a silently missing world-model
score can never look like a neutral one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..contracts.sonic78 import ACTION78_DIM, SONIC78_V1, TOKEN_ABS_BOUND, Sonic78Spec


SCORE_TERMS = ("stage_gain", "support", "uncertainty")


@dataclass(frozen=True)
class GateConfig:
    stage_weight: float = 1.0
    support_weight: float = 0.5
    uncertainty_weight: float = 1.0
    selection_margin: float = 0.0
    motion_abs_bound: float = TOKEN_ABS_BOUND
    hand_abs_bound: float | None = None

    def __post_init__(self) -> None:
        for name in ("stage_weight", "support_weight", "uncertainty_weight"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(self.selection_margin) or self.selection_margin < 0.0:
            raise ValueError("selection_margin must be finite and non-negative")
        if (
            not np.isfinite(self.motion_abs_bound)
            or not 0.0 < self.motion_abs_bound <= TOKEN_ABS_BOUND
        ):
            raise ValueError(
                f"motion_abs_bound must be in (0, {TOKEN_ABS_BOUND}]; a looser bound is not the "
                "official contract"
            )
        if self.hand_abs_bound is not None:
            if not np.isfinite(self.hand_abs_bound) or self.hand_abs_bound <= 0.0:
                raise ValueError("hand_abs_bound must be positive when set")
        if self.stage_weight + self.support_weight + self.uncertainty_weight <= 0.0:
            raise ValueError("at least one score weight must be positive")

    @classmethod
    def from_mapping(cls, mapping: Mapping) -> "GateConfig":
        allowed = {
            "stage_weight",
            "support_weight",
            "uncertainty_weight",
            "selection_margin",
            "motion_abs_bound",
            "hand_abs_bound",
        }
        unknown = set(mapping) - allowed
        if unknown:
            raise ValueError(f"unknown gate config keys {sorted(unknown)}")
        return cls(**dict(mapping))

    def active_terms(self) -> tuple[str, ...]:
        weights = {
            "stage_gain": self.stage_weight,
            "support": self.support_weight,
            "uncertainty": self.uncertainty_weight,
        }
        return tuple(name for name in SCORE_TERMS if weights[name] > 0.0)


@dataclass(frozen=True)
class GateDecision:
    index: int
    fallback: bool
    reason: str
    scores: np.ndarray
    valid: np.ndarray
    margin: float
    ambiguous: bool

    @property
    def selected(self) -> bool:
        return not self.fallback and self.index >= 0


def _component(name: str, components: Mapping[str, np.ndarray], K: int) -> np.ndarray:
    if name not in components:
        raise ValueError(f"score term {name!r} is required by the gate config but was not supplied")
    value = np.asarray(components[name], dtype=np.float64)
    if value.shape != (K,):
        raise ValueError(f"score term {name!r} must be [{K}], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"score term {name!r} must be finite")
    return value


def decide(
    candidates: np.ndarray,
    *,
    components: Mapping[str, np.ndarray] | None = None,
    config: GateConfig = GateConfig(),
    spec: Sonic78Spec = SONIC78_V1,
) -> GateDecision:
    """Rank one observation's ``[K, H, 78]`` candidates; fail closed when empty."""

    array = np.asarray(candidates, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != ACTION78_DIM:
        raise ValueError(f"gate candidates must be [K,H,{ACTION78_DIM}], got {array.shape}")
    if array.shape[-2] != spec.horizon:
        raise ValueError(f"gate candidates must carry horizon {spec.horizon}, got {array.shape}")
    K = int(array.shape[0])
    if K <= 0:
        raise ValueError("gate needs at least one candidate")

    valid = np.isfinite(array).all(axis=(-2, -1))
    motion = array[..., :64]
    valid &= np.abs(motion).max(axis=(-2, -1)) <= config.motion_abs_bound
    if config.hand_abs_bound is not None:
        valid &= np.abs(array[..., 64:]).max(axis=(-2, -1)) <= config.hand_abs_bound
    if not valid.any():
        return GateDecision(
            index=-1,
            fallback=True,
            reason="no_valid_candidate",
            scores=np.full(K, -np.inf, dtype=np.float64),
            valid=valid,
            margin=0.0,
            ambiguous=False,
        )

    components = {} if components is None else components
    active = config.active_terms()
    weights = {
        "stage_gain": config.stage_weight,
        "support": config.support_weight,
        "uncertainty": config.uncertainty_weight,
    }
    scores = np.zeros(K, dtype=np.float64)
    for name in active:
        value = _component(name, components, K)
        scores += (-weights[name] * value) if name == "uncertainty" else weights[name] * value
    scores = np.where(valid, scores, -np.inf)

    order = np.argsort(scores)[::-1]
    best = int(order[0])
    runner_up = next((int(index) for index in order[1:] if np.isfinite(scores[index])), None)
    margin = float(scores[best] - scores[runner_up]) if runner_up is not None else float("inf")
    ambiguous = runner_up is not None and margin < config.selection_margin
    return GateDecision(
        index=best,
        fallback=False,
        reason="selected_ambiguous" if ambiguous else "selected",
        scores=scores,
        valid=valid,
        margin=margin,
        ambiguous=ambiguous,
    )


__all__ = ["GateConfig", "GateDecision", "SCORE_TERMS", "decide"]
