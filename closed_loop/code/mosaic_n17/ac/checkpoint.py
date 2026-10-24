"""Phase B checkpoint contract (typed slots + SONIC78 actions).

A Phase B checkpoint is only usable when its observation/action contract matches
the deployment exactly: slot layout, state46 layout, horizon/execute cadence,
target semantics, and the frozen Phase A / joint-order digests.  Loading refuses
anything else instead of re-interpreting tensors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from ..contracts.sonic78 import SONIC78_V1, Sonic78Spec
from ..contracts.state46 import AC_STATE46_DIM, AC_STATE46_LAYOUT, joint_order_sha256
from .phase_b_runtime import TARGET_SEMANTICS, TARGET_SEMANTICS_EXECUTED_PREFIX
from .slots import SLOT_COUNTS, SLOT_DIM


PHASE_B_SCHEMA = "mosaic.n17.phase_b_checkpoint.v1"


@dataclass(frozen=True)
class PhaseBContract:
    phase_a_sha256: str
    state_dim: int = AC_STATE46_DIM
    state_layout: str = AC_STATE46_LAYOUT
    slot_dim: int = SLOT_DIM
    horizon: int = SONIC78_V1.horizon
    execute_steps: int = SONIC78_V1.execute_steps
    target_semantics: str = TARGET_SEMANTICS_EXECUTED_PREFIX
    joint_order_sha256: str = ""
    schema: str = PHASE_B_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PHASE_B_SCHEMA:
            raise ValueError(f"unsupported schema {self.schema!r}")
        if len(self.phase_a_sha256) != 64:
            raise ValueError("phase_a_sha256 must be a 64-char hex digest")
        if self.state_dim != AC_STATE46_DIM or self.state_layout != AC_STATE46_LAYOUT:
            raise ValueError(
                f"Phase B requires state46 layout {AC_STATE46_LAYOUT!r} ({AC_STATE46_DIM}D), "
                f"got {self.state_layout!r} ({self.state_dim}D)"
            )
        if self.slot_dim != SLOT_DIM:
            raise ValueError(f"slot width is frozen at {SLOT_DIM}")
        if self.target_semantics not in TARGET_SEMANTICS:
            raise ValueError(f"unsupported target semantics {self.target_semantics!r}")
        if self.joint_order_sha256 and self.joint_order_sha256 != joint_order_sha256():
            raise ValueError("joint_order_sha256 does not match the frozen G1 order table")

    def assert_matches(self, spec: Sonic78Spec = SONIC78_V1) -> None:
        if (self.horizon, self.execute_steps) != (spec.horizon, spec.execute_steps):
            raise ValueError(
                f"checkpoint cadence {(self.horizon, self.execute_steps)} != "
                f"{spec.name} {(spec.horizon, spec.execute_steps)}"
            )


def _dump(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(
        json.dumps({"sha256": digest, "contract": payload["contract"]}, indent=1) + "\n",
        encoding="utf-8",
    )
    return digest


def save_phase_b_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    config: Any,
    contract: PhaseBContract,
    extra: dict[str, Any] | None = None,
) -> str:
    contract.assert_matches()
    payload = {
        "schema": PHASE_B_SCHEMA,
        "contract": asdict(contract),
        "config": config if isinstance(config, dict) else vars(config),
        "model": model.state_dict(),
        "extra": dict(extra or {}),
    }
    return _dump(Path(path), payload)


def load_phase_b_checkpoint(
    path: str | os.PathLike[str],
    *,
    spec: Sonic78Spec = SONIC78_V1,
    expected_phase_a_sha256: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != PHASE_B_SCHEMA:
        raise ValueError("not a mosaic_n17 Phase B checkpoint")
    contract = PhaseBContract(**payload["contract"])
    contract.assert_matches(spec)
    if expected_phase_a_sha256 is not None and contract.phase_a_sha256 != expected_phase_a_sha256:
        raise ValueError(
            "Phase A digest mismatch: the AC was trained on a different frozen encoder"
        )
    return payload


__all__ = [
    "PHASE_B_SCHEMA",
    "PhaseBContract",
    "SLOT_COUNTS",
    "load_phase_b_checkpoint",
    "save_phase_b_checkpoint",
]
