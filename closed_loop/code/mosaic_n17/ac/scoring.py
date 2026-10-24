"""Candidate scoring on predicted typed slots (M3/M4/M7).

Two independent quantities are exposed, never fused here:

* ``stage_gain``: how much closer the candidate's predicted Stage slots are to
  the frontier goal than the current Stage slots are;
* ``support``: how plausible the predicted C/B/E change is relative to a
  calibrated scale (a large change that the data never shows is penalized).

Both are pure functions of slots and masks.  Weights and calibration constants
must come from the training run that produced the AC checkpoint; this module
only refuses to invent them (a missing scale or goal raises instead of
defaulting).
"""

from __future__ import annotations

import numpy as np

from .slots import SLOT_DIM, SLOT_TYPES, TOTAL_SLOTS, slot_slices


def _pool(slots: np.ndarray, name: str) -> np.ndarray:
    span = slot_slices()[name]
    return slots[..., span, :].mean(axis=-2)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    denominator = np.clip(norm_a * norm_b, 1e-12, None)
    return 1.0 - (a * b).sum(axis=-1) / denominator


def typed_distance(predicted: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """Per-type cosine distance on the type-pooled slots."""

    predicted = _validate(predicted)
    target = _validate(target)
    if predicted.shape != target.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {target.shape}")
    return {
        name: _cosine_distance(_pool(predicted, name), _pool(target, name)) for name in SLOT_TYPES
    }


def stage_gain(
    current_slots: np.ndarray,
    predicted_slots: np.ndarray,
    goal_slots: np.ndarray,
) -> np.ndarray:
    """``d_S(current, goal) - d_S(predicted, goal)`` (positive is progress)."""

    current = _validate(current_slots)
    predicted = _validate(predicted_slots)
    goal = _validate(goal_slots)
    if goal.ndim != current.ndim:
        raise ValueError("goal slots must share the batch layout with current slots")
    d_current = _cosine_distance(_pool(current, "S"), _pool(goal, "S"))
    d_predicted = _cosine_distance(_pool(predicted, "S"), _pool(goal, "S"))
    return d_current - d_predicted


def plausibility(
    current_slots: np.ndarray,
    predicted_slots: np.ndarray,
    *,
    scale: np.ndarray | float,
) -> np.ndarray:
    """Negative normalized change magnitude over C/B/E (0 = unchanged)."""

    current = _validate(current_slots)
    predicted = _validate(predicted_slots)
    if current.shape != predicted.shape:
        raise ValueError("current and predicted slots must share a shape")
    scale_array = np.asarray(scale, dtype=np.float64)
    if not np.isfinite(scale_array).all() or np.any(scale_array <= 0.0):
        raise ValueError("scale must be finite and strictly positive")
    delta = predicted - current
    normalized = delta / scale_array
    per_sample = (normalized**2).mean(axis=(-1, -2))
    return -per_sample


def score_components(
    *,
    current_slots: np.ndarray,
    predicted_slots: np.ndarray,
    goal_slots: np.ndarray | None,
    change_scale: np.ndarray | float,
    uncertainty: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Assemble the gate's score terms; a missing goal/uncertainty is explicit."""

    components = {
        "support": plausibility(current_slots, predicted_slots, scale=change_scale),
    }
    if goal_slots is not None:
        components["stage_gain"] = stage_gain(current_slots, predicted_slots, goal_slots)
    if uncertainty is not None:
        value = np.asarray(uncertainty, dtype=np.float64)
        if not np.isfinite(value).all():
            raise ValueError("uncertainty must be finite")
        components["uncertainty"] = value
    return components


def _validate(slots: np.ndarray) -> np.ndarray:
    array = np.asarray(slots)
    if array.shape[-2:] != (TOTAL_SLOTS, SLOT_DIM):
        raise ValueError(f"slots must end in ({TOTAL_SLOTS}, {SLOT_DIM}), got {array.shape}")
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError("slots must be finite")
    return array


__all__ = [
    "plausibility",
    "score_components",
    "stage_gain",
    "typed_distance",
]
