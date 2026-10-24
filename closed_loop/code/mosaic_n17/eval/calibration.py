"""Scorer calibration and latency budgets for the arbitration (M4 gates).

A ranking that is never calibrated cannot be trusted to pick actions, so the
report exposes ECE / Brier over binarized outcomes and the Spearman correlation
between score and outcome.  Latency is measured end to end (backbone + K DiT
sampling + AC scoring) and compared with the deployment replan period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationReport:
    expected_calibration_error: float
    brier_score: float
    spearman: float
    samples: int


def _validate(probabilities: np.ndarray, outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(outcomes, dtype=np.float64)
    if probs.shape != labels.shape or probs.ndim != 1:
        raise ValueError("probabilities and outcomes must share a [N] shape")
    if not np.isfinite(probs).all() or not np.isfinite(labels).all():
        raise ValueError("inputs must be finite")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("outcomes must be binary")
    if probs.size == 0:
        raise ValueError("need at least one sample")
    return probs, labels


def expected_calibration_error(
    probabilities: np.ndarray, outcomes: np.ndarray, *, bins: int = 10
) -> float:
    probs, labels = _validate(probabilities, outcomes)
    if bins <= 0:
        raise ValueError("bins must be positive")
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, bins - 1)
    error = 0.0
    for bucket in range(bins):
        mask = index == bucket
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probs[mask].mean()))
    return error


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    probs, labels = _validate(probabilities, outcomes)
    return float(((probs - labels) ** 2).mean())


def calibration_report(
    probabilities: np.ndarray, outcomes: np.ndarray, *, bins: int = 10
) -> CalibrationReport:
    from scipy.stats import spearmanr

    probs, labels = _validate(probabilities, outcomes)
    rho = spearmanr(probs, labels).statistic
    return CalibrationReport(
        expected_calibration_error=expected_calibration_error(probs, labels, bins=bins),
        brier_score=brier_score(probs, labels),
        spearman=float(rho) if np.isfinite(rho) else 0.0,
        samples=int(probs.size),
    )


@dataclass(frozen=True)
class LatencyReport:
    samples: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    budget_ms: float
    within_budget: bool


def latency_report(samples_ms: np.ndarray, *, budget_ms: float) -> LatencyReport:
    values = np.asarray(samples_ms, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("need at least one latency sample")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("latencies must be finite and non-negative")
    if not np.isfinite(budget_ms) or budget_ms <= 0.0:
        raise ValueError("budget_ms must be positive")
    percentiles = np.percentile(values, [50, 95, 99])
    return LatencyReport(
        samples=int(values.size),
        p50_ms=float(percentiles[0]),
        p95_ms=float(percentiles[1]),
        p99_ms=float(percentiles[2]),
        budget_ms=float(budget_ms),
        within_budget=bool(percentiles[2] <= budget_ms),
    )


__all__ = [
    "CalibrationReport",
    "LatencyReport",
    "brier_score",
    "calibration_report",
    "expected_calibration_error",
    "latency_report",
]
