"""Frozen typed Phase A runtime (C6/B3/E3/S2 x 384).

Phase A is trained in FlowBody (``extensions/structured_slot_ac``) on top of a
frozen V-JEPA encoder.  This module does **not** re-implement it: it binds an
explicit checkpoint whose contract is validated before use, and adds the N1.7
side checks (slot counts, width, finiteness, batch shape).

Loading from FlowBody is opt-in and explicit.  If the source tree or checkpoint
is absent the adapter refuses to construct: there is no fallback that silently
feeds raw pixels or zero slots into the consequence model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import sys
from typing import Callable

import numpy as np
import torch

from .slots import SLOT_COUNTS, SLOT_DIM, TOTAL_SLOTS


PHASE_A_ARCHITECTURES = frozenset(
    {
        "structured_slot_phase_a_v2",
        "structured_slot_phase_a_v31_mixer",
        "structured_slot_phase_a_v31",
    }
)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PhaseAContract:
    architecture_id: str
    dense_dim: int
    checkpoint_sha256: str
    slot_counts: tuple[tuple[str, int], ...] = tuple(SLOT_COUNTS.items())
    slot_dim: int = SLOT_DIM

    def __post_init__(self) -> None:
        if self.architecture_id not in PHASE_A_ARCHITECTURES:
            raise ValueError(
                f"unsupported Phase A architecture {self.architecture_id!r}; "
                f"expected one of {sorted(PHASE_A_ARCHITECTURES)}"
            )
        if self.dense_dim <= 0:
            raise ValueError("dense_dim must be positive")
        if dict(self.slot_counts) != dict(SLOT_COUNTS):
            raise ValueError(f"typed slot counts are frozen at {dict(SLOT_COUNTS)}")
        if self.slot_dim != SLOT_DIM:
            raise ValueError(f"typed slot width is frozen at {SLOT_DIM}")
        if len(self.checkpoint_sha256) != 64:
            raise ValueError("checkpoint_sha256 must be a 64-char hex digest")


class PhaseAAdapter:
    """Validated callable: dense V-JEPA tokens ``[B,N,D]`` -> slots ``[B,14,384]``."""

    def __init__(self, encode_fn: Callable[[torch.Tensor], torch.Tensor], contract: PhaseAContract):
        if not callable(encode_fn):
            raise TypeError("encode_fn must be callable")
        self._encode_fn = encode_fn
        self.contract = contract

    def encode(self, dense_tokens: torch.Tensor) -> np.ndarray:
        if not torch.is_tensor(dense_tokens):
            raise TypeError("dense_tokens must be a torch.Tensor")
        if dense_tokens.ndim != 3:
            raise ValueError(f"dense_tokens must be [B,N,D], got {tuple(dense_tokens.shape)}")
        if dense_tokens.shape[-1] != self.contract.dense_dim:
            raise ValueError(
                f"dense_tokens last dim must be {self.contract.dense_dim}, "
                f"got {dense_tokens.shape[-1]}"
            )
        with torch.no_grad():
            slots = self._encode_fn(dense_tokens)
        slots = torch.as_tensor(slots)
        if tuple(slots.shape[-2:]) != (TOTAL_SLOTS, SLOT_DIM):
            raise ValueError(
                f"Phase A must return [B,{TOTAL_SLOTS},{SLOT_DIM}], got {tuple(slots.shape)}"
            )
        if not bool(torch.isfinite(slots).all()):
            raise ValueError("Phase A produced non-finite slots")
        return slots.detach().float().cpu().numpy()

    @classmethod
    def from_flowbody(
        cls,
        flowbody_root: str | os.PathLike[str],
        checkpoint_path: str | os.PathLike[str],
        *,
        device: str = "cpu",
    ) -> "PhaseAAdapter":
        """Load a frozen Phase A checkpoint from the FlowBody source tree."""

        root = os.path.abspath(os.fspath(flowbody_root))
        package_root = os.path.join(root, "flowbody")
        if not os.path.isdir(package_root):
            raise FileNotFoundError(f"{package_root} does not exist; pass the FlowBody repo root")
        checkpoint_path = os.path.abspath(os.fspath(checkpoint_path))
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)

        if package_root not in sys.path:
            sys.path.insert(0, package_root)
        from extensions.structured_slot_ac.phase_a import (  # type: ignore[import-not-found]
            StructuredSlotPhaseAModel,
            phase_a_architecture_id,
        )

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "architecture" not in payload:
            raise ValueError("not a Phase A checkpoint (missing 'architecture')")
        contract_meta = payload.get("contract") or {}
        if contract_meta.get("phase") != "A":
            raise ValueError("checkpoint contract does not declare phase A")
        architecture = payload.get("architecture") or {}
        architecture_id = str(architecture.get("id", "")) if isinstance(architecture, dict) else ""
        if not architecture_id:
            raise ValueError("Phase A checkpoint does not record its architecture id")
        model = StructuredSlotPhaseAModel(
            enable_slot_mixer=architecture_id.endswith("_mixer"),
            body_readout_hidden=int(contract_meta.get("body_readout_hidden", 0)),
        )
        state = payload.get("model")
        if state is None:
            raise ValueError("Phase A checkpoint is missing its model state dict")
        model.load_state_dict(state)
        model.eval().to(device)

        expected_id = phase_a_architecture_id(enable_slot_mixer=architecture_id.endswith("_mixer"))
        dense_dim = int(architecture.get("dense_dim", 0))
        contract = PhaseAContract(
            architecture_id=expected_id,
            dense_dim=dense_dim,
            checkpoint_sha256=sha256_file(checkpoint_path),
        )
        # Contract claims are recorded in the checkpoint; a mismatch means the
        # architecture id or the slot layout drifted between repos.
        claimed_id = str(contract_meta.get("architecture_id", expected_id))
        if claimed_id != expected_id:
            raise ValueError(f"checkpoint architecture_id {claimed_id!r} != {expected_id!r}")

        def encode_fn(dense_tokens: torch.Tensor) -> torch.Tensor:
            encoded = model.encode(dense_tokens.to(device))
            concat = encoded["concat"]
            if torch.is_tensor(concat) and concat.dim() == 4:
                if concat.shape[1] != 1:
                    raise ValueError("Phase A encode must receive a single frame per call")
                concat = concat[:, 0]
            return concat

        return cls(encode_fn, contract)


__all__ = ["PHASE_A_ARCHITECTURES", "PhaseAAdapter", "PhaseAContract", "sha256_file"]
