"""Offline evaluation for the arbitration: oracle, calibration, latency, receipts."""

from __future__ import annotations

from .calibration import (
    CalibrationReport,
    LatencyReport,
    brier_score,
    calibration_report,
    expected_calibration_error,
    latency_report,
)
from .oracle import OracleReport, oracle_at_n, ranking_regret, top1_agreement
from .physics_replay import (
    RECEIPT_SCHEMA,
    STATUSES,
    PhysicsReceipt,
    validate_receipt,
    verification_digest,
)


__all__ = [
    "CalibrationReport",
    "LatencyReport",
    "OracleReport",
    "PhysicsReceipt",
    "RECEIPT_SCHEMA",
    "STATUSES",
    "brier_score",
    "calibration_report",
    "expected_calibration_error",
    "latency_report",
    "oracle_at_n",
    "ranking_regret",
    "top1_agreement",
    "validate_receipt",
    "verification_digest",
]
