"""Physics-replay receipt contract.

A candidate becomes a *verified teacher* only after a simulator/controller replay
produced a receipt that binds: the candidate checksum, the controller and
simulator identities, the exact tick window, and the applied/rejected outcome.
This module validates such receipts and derives the digest the teacher bank
stores; it does not run a simulator, and it refuses to verify anything else.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json


RECEIPT_SCHEMA = "mosaic.n17.physics_receipt.v1"
STATUSES = frozenset({"applied", "rejected"})


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-char hex digest")
    int(value, 16)
    return value


@dataclass(frozen=True)
class PhysicsReceipt:
    candidate_sha256: str
    controller_sha256: str
    simulator_sha256: str
    start_tick: int
    ticks: int
    status: str
    contact: bool = False
    schema: str = RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise ValueError(f"unsupported receipt schema {self.schema!r}")
        _digest(self.candidate_sha256, "candidate_sha256")
        _digest(self.controller_sha256, "controller_sha256")
        _digest(self.simulator_sha256, "simulator_sha256")
        for name in ("start_tick", "ticks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if self.ticks <= 0:
            raise ValueError("a receipt must cover at least one tick")
        if self.status not in STATUSES:
            raise ValueError(f"unsupported receipt status {self.status!r}")

    def canonical_digest(self) -> str:
        payload = json.dumps(
            {
                "schema": self.schema,
                "candidate_sha256": self.candidate_sha256,
                "controller_sha256": self.controller_sha256,
                "simulator_sha256": self.simulator_sha256,
                "start_tick": self.start_tick,
                "ticks": self.ticks,
                "status": self.status,
                "contact": self.contact,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def assert_applied(self) -> None:
        if self.status != "applied":
            raise ValueError(
                f"receipt status is {self.status!r}; a rejected rollout cannot verify a teacher"
            )


def validate_receipt(payload: Mapping, *, expected_candidate_sha256: str) -> PhysicsReceipt:
    if not isinstance(payload, Mapping):
        raise TypeError("receipt must be a mapping")
    required = (
        "schema",
        "candidate_sha256",
        "controller_sha256",
        "simulator_sha256",
        "start_tick",
        "ticks",
        "status",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"receipt is missing {missing}")
    receipt = PhysicsReceipt(
        candidate_sha256=str(payload["candidate_sha256"]),
        controller_sha256=str(payload["controller_sha256"]),
        simulator_sha256=str(payload["simulator_sha256"]),
        start_tick=int(payload["start_tick"]),
        ticks=int(payload["ticks"]),
        status=str(payload["status"]),
        contact=bool(payload.get("contact", False)),
        schema=str(payload["schema"]),
    )
    if receipt.candidate_sha256 != expected_candidate_sha256:
        raise ValueError(
            "receipt candidate digest does not match the candidate it claims to verify"
        )
    receipt.assert_applied()
    return receipt


def verification_digest(candidate_sha256: str, receipt: PhysicsReceipt) -> str:
    """Digest stored in the teacher bank when this receipt verifies a candidate."""

    if receipt.candidate_sha256 != candidate_sha256:
        raise ValueError("receipt and candidate digests disagree")
    receipt.assert_applied()
    return receipt.canonical_digest()


__all__ = [
    "RECEIPT_SCHEMA",
    "STATUSES",
    "PhysicsReceipt",
    "validate_receipt",
    "verification_digest",
]
