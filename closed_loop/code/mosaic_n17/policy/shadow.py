"""Shadow execution (M4): keep controlling with the baseline, log the gate.

Shadow mode exists to prove the arbitration is useful before it is allowed to
choose.  It executes the baseline candidate (index 0 by default) and records what
the gate *would* have chosen, its margin and whether it agreed.  If the baseline
candidate itself is inadmissible, nothing is executed and the caller must fall
back to the previously verified plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts.sonic78 import ACTION78_DIM, SONIC78_V1, Sonic78Spec
from .candidate_sampler import CandidateSet
from .decision_gate import GateDecision


@dataclass(frozen=True)
class ShadowRecord:
    executed_index: int
    executed_source: str
    gate_index: int
    gate_fallback: bool
    gate_reason: str
    agrees: bool
    margin: float
    executed_sha256: str


def _row_sha256(row: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()


def run_shadow_step(
    candidates: CandidateSet,
    decision: GateDecision,
    *,
    primary_index: int = 0,
    config: "object | None" = None,
    spec: Sonic78Spec = SONIC78_V1,
) -> tuple[np.ndarray | None, ShadowRecord]:
    """Execute the primary candidate, record the gate's choice.

    Returns ``(executed_action_or_none, record)``.  ``executed_action`` has shape
    ``[H, 78]`` and is a *validated* row of ``candidates``.
    """

    array = np.asarray(candidates.candidates, dtype=np.float64)
    if array.ndim != 4 or array.shape[-1] != ACTION78_DIM:
        raise ValueError(f"candidates must be [B,K,H,{ACTION78_DIM}], got {array.shape}")
    if array.shape[0] != 1:
        raise ValueError("shadow step handles exactly one observation (B=1)")
    if array.shape[-2] != spec.horizon:
        raise ValueError(f"candidates must carry horizon {spec.horizon}, got {array.shape}")
    K = int(array.shape[1])
    if not 0 <= primary_index < K:
        raise ValueError(f"primary_index must be in [0,K), got {primary_index}")
    if decision.scores.shape != (K,):
        raise ValueError("decision scores must align with candidates")

    row = array[0, primary_index]
    # Re-check through the contract rather than duplicating bounds here.
    admissible = bool(np.isfinite(row).all())
    if admissible:
        try:
            spec.validate_block(row[None, ...])
        except (ValueError, TypeError):
            admissible = False

    if not admissible:
        return None, ShadowRecord(
            executed_index=-1,
            executed_source="fallback",
            gate_index=decision.index,
            gate_fallback=decision.fallback,
            gate_reason=decision.reason,
            agrees=False,
            margin=decision.margin,
            executed_sha256="",
        )

    agrees = decision.selected and decision.index == primary_index
    return row, ShadowRecord(
        executed_index=primary_index,
        executed_source="primary",
        gate_index=decision.index,
        gate_fallback=decision.fallback,
        gate_reason=decision.reason,
        agrees=bool(agrees),
        margin=decision.margin,
        executed_sha256=_row_sha256(row),
    )


__all__ = ["ShadowRecord", "run_shadow_step"]
