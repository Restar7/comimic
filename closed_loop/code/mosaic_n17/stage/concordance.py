"""Candidate concordance against a Stage frontier (M7).

``gain_k = d_S(Z_t, g_t) - d_S(Zhat_{t+Delta,k}, g_t)`` on the Stage branch.
The report keeps the raw per-candidate gains so ranking can be audited; it never
turns a gain into a predicted task success.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ac.scoring import stage_gain
from ..ac.slots import SLOT_DIM, TOTAL_SLOTS


def _validate(slots: object, label: str) -> np.ndarray:
    array = np.asarray(slots, dtype=np.float64)
    if array.shape[-2:] != (TOTAL_SLOTS, SLOT_DIM):
        raise ValueError(f"{label} must end in ({TOTAL_SLOTS}, {SLOT_DIM}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return array


@dataclass(frozen=True)
class ConcordanceReport:
    gains: np.ndarray
    order: np.ndarray
    best_index: int

    @property
    def best_gain(self) -> float:
        return float(self.gains[self.best_index])


def rank_candidates(gains: np.ndarray) -> np.ndarray:
    values = np.asarray(gains, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("gains must be a non-empty 1-D array")
    if not np.isfinite(values).all():
        raise ValueError("gains must be finite")
    return np.argsort(values)[::-1]


def concordance(
    current_slots: object,
    predicted_slots: object,
    goal_slots: object,
) -> ConcordanceReport:
    current = _validate(current_slots, "current slots")
    predicted = _validate(predicted_slots, "predicted slots")
    goal = _validate(goal_slots, "goal slots")
    candidates_ok = predicted.shape == current.shape or (
        predicted.ndim == current.ndim + 1 and predicted.shape[1:] == current.shape
    )
    if not candidates_ok:
        raise ValueError(
            f"predicted slots {predicted.shape} must be {current.shape} or [K]+{current.shape}"
        )
    gains = np.atleast_1d(stage_gain(current, predicted, goal))
    order = rank_candidates(gains)
    return ConcordanceReport(gains=gains, order=order, best_index=int(order[0]))


def top1_agreement(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Fraction of rows where two rankers pick the same candidate."""

    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim < 1:
        raise ValueError("score arrays must share a shape")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("scores must be finite")
    return float((a.argmax(axis=-1) == b.argmax(axis=-1)).mean())


__all__ = ["ConcordanceReport", "concordance", "rank_candidates", "top1_agreement"]
