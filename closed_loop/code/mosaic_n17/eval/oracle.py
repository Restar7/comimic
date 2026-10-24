"""Offline candidate evaluation: Oracle@N, ranking regret and diversity.

These metrics answer whether K samples contain something better than the model's
single guess under a *given* scorer.  They are not a physics result: Oracle@N
still has to be re-measured with replay receipts before it supports any claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OracleReport:
    candidates: int
    valid_rate: float
    oracle_score_mean: float
    selected_score_mean: float
    oracle_gap_mean: float
    pairwise_diversity_mean: float
    regret_mean: float


def _validate(candidates: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(candidates, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if array.ndim != 4:
        raise ValueError(f"candidates must be [B,K,H,D], got {array.shape}")
    if values.shape != array.shape[:2]:
        raise ValueError(f"scores must be [B,K]={array.shape[:2]}, got {values.shape}")
    if not np.isfinite(array).all() or not np.isfinite(values).all():
        raise ValueError("candidates and scores must be finite")
    return array, values


def oracle_at_n(candidates: np.ndarray, scores: np.ndarray) -> OracleReport:
    """Oracle/selected/gap/diversity for ``[B,K,H,D]`` candidates with ``[B,K]`` scores."""

    array, values = _validate(candidates, scores)
    batch, K = values.shape
    flat = array.reshape(batch, K, -1)
    pairwise = np.linalg.norm(flat[:, :, None, :] - flat[:, None, :, :], axis=-1)
    upper = pairwise[:, np.triu_indices(K, k=1)[0], np.triu_indices(K, k=1)[1]]
    diversity = float(upper.mean()) if upper.size else 0.0
    oracle = values.max(axis=1)
    selected = values[:, 0]
    return OracleReport(
        candidates=int(batch * K),
        valid_rate=1.0,
        oracle_score_mean=float(oracle.mean()),
        selected_score_mean=float(selected.mean()),
        oracle_gap_mean=float((oracle - selected).mean()),
        pairwise_diversity_mean=diversity,
        regret_mean=float((oracle - selected).mean()),
    )


def ranking_regret(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Reference-truth score lost by taking the ranker's top candidate.

    ``reference`` is the ground-truth outcome (e.g. verified replay score); the
    ranker being judged is ``scores``.  Regret 0 means the ranker picked an
    outcome-maximal candidate.
    """

    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(reference, dtype=np.float64)
    if values.shape != truth.shape or values.ndim != 2:
        raise ValueError("scores and reference must share a [B,K] shape")
    pick = values.argmax(axis=1)
    picked = np.take_along_axis(truth, pick[:, None], axis=1)[:, 0]
    return truth.max(axis=1) - picked


def top1_agreement(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("scores must share a shape")
    return float((left.argmax(axis=-1) == right.argmax(axis=-1)).mean())


__all__ = ["OracleReport", "oracle_at_n", "ranking_regret", "top1_agreement"]
