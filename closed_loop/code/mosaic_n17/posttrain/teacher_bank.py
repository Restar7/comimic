"""Verified teacher bank for SONIC78 candidates.

Only rollout-verified candidates may teach: a model-screened candidate is a
self-assessment, not evidence.  The bank keeps the verification receipt digest,
per-candidate checksums, a generation counter and protected anchors; loading a
bank whose contract digest does not match refuses instead of re-interpreting it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from ..contracts.sonic78 import ACTION78_DIM, SONIC78_V1, Sonic78Spec


BANK_SCHEMA = "mosaic.n17.teacher_bank.v1"
PROVISIONAL = "provisional"
VERIFIED = "rollout_verified"
STATUSES = frozenset({PROVISIONAL, VERIFIED})


def _checksum(candidate: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(candidate, dtype=np.float32).tobytes()).hexdigest()


@dataclass
class _Slot:
    candidates: np.ndarray
    mask: np.ndarray
    scores: np.ndarray
    checksums: tuple[str, ...]
    status: str
    generation: int
    protected: bool
    receipt_sha256: str = ""
    lineage: dict = field(default_factory=dict)


class TeacherBank:
    """State-id keyed bank of ``[K,H,78]`` candidates with explicit evidence."""

    def __init__(self, *, spec: Sonic78Spec = SONIC78_V1, generation: int = 0) -> None:
        if spec.name not in {"SONIC78_V1", "SONIC78_LOW_LATENCY"}:
            raise ValueError(f"unsupported contract {spec.name!r}")
        self.spec = spec
        self.generation = int(generation)
        self._slots: dict[str, _Slot] = {}

    def add(
        self,
        state_id: str,
        candidates: object,
        mask: object | None = None,
        scores: object | None = None,
        *,
        status: str = PROVISIONAL,
        protected: bool = False,
        lineage: dict | None = None,
    ) -> None:
        if status not in STATUSES:
            raise ValueError(f"unsupported teacher status {status!r}")
        if not isinstance(state_id, str) or not state_id:
            raise ValueError("state_id must be a non-empty string")
        array = np.asarray(candidates, dtype=np.float64)
        if array.ndim != 3 or array.shape[-1] != ACTION78_DIM:
            raise ValueError(f"candidates must be [K,H,{ACTION78_DIM}], got {array.shape}")
        if array.shape[-2] != self.spec.horizon:
            raise ValueError(
                f"candidates must carry horizon {self.spec.horizon}, got {array.shape[-2]}"
            )
        resolved_mask = (
            np.ones(array.shape, dtype=bool) if mask is None else np.asarray(mask).astype(bool)
        )
        if resolved_mask.shape != array.shape:
            raise ValueError("mask must match candidates")
        resolved_scores = (
            np.zeros(array.shape[0], dtype=np.float64)
            if scores is None
            else np.asarray(scores, dtype=np.float64)
        )
        if resolved_scores.shape != (array.shape[0],):
            raise ValueError("scores must be [K]")
        if not np.isfinite(array[resolved_mask]).all():
            raise ValueError("candidates must be finite")
        self.spec.validate_block(array, resolved_mask)
        self._slots[state_id] = _Slot(
            candidates=array,
            mask=resolved_mask,
            scores=resolved_scores,
            checksums=tuple(_checksum(row) for row in array),
            status=status,
            generation=self.generation,
            protected=bool(protected),
            lineage=dict(lineage or {}),
        )

    def verify(self, state_id: str, receipt_sha256: str) -> None:
        """Promote a provisional slot to verified; requires a sim receipt digest."""

        slot = self._get_slot(state_id)
        if not isinstance(receipt_sha256, str) or len(receipt_sha256) != 64:
            raise ValueError("receipt_sha256 must be a 64-char digest")
        slot.status = VERIFIED
        slot.receipt_sha256 = receipt_sha256

    def _get_slot(self, state_id: str) -> _Slot:
        slot = self._slots.get(str(state_id))
        if slot is None:
            raise KeyError(f"unknown teacher state {state_id!r}")
        if slot.generation != self.generation:
            raise ValueError(f"teacher state {state_id!r} generation mismatch")
        return slot

    def get(self, state_id: str, *, require_verified: bool = True) -> _Slot:
        slot = self._get_slot(state_id)
        if require_verified and slot.status != VERIFIED:
            raise ValueError(f"teacher state {state_id!r} is {slot.status}, not verified")
        return slot

    def eligible_ids(self) -> list[str]:
        """Verified, non-protected state ids usable as post-training supervision."""

        return sorted(
            state_id
            for state_id, slot in self._slots.items()
            if slot.status == VERIFIED and not slot.protected and slot.generation == self.generation
        )

    def advance_generation(self) -> int:
        """Invalidate every stored slot and bump the generation counter."""

        self.generation += 1
        return self.generation

    def contract_digest(self) -> str:
        payload = json.dumps(
            {
                "schema": BANK_SCHEMA,
                "spec": self.spec.name,
                "horizon": self.spec.horizon,
                "execute_steps": self.spec.execute_steps,
                "action_dim": ACTION78_DIM,
                "generation": self.generation,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema": BANK_SCHEMA,
            "contract_sha256": self.contract_digest(),
            "spec": self.spec.name,
            "generation": self.generation,
            "slots": {
                state_id: {
                    "checksums": list(slot.checksums),
                    "status": slot.status,
                    "generation": slot.generation,
                    "protected": slot.protected,
                    "receipt_sha256": slot.receipt_sha256,
                    "lineage": slot.lineage,
                }
                for state_id, slot in self._slots.items()
            },
        }
        arrays = {}
        for index, (state_id, slot) in enumerate(self._slots.items()):
            arrays[f"candidates_{index}"] = slot.candidates.astype(np.float32)
            arrays[f"mask_{index}"] = slot.mask
            arrays[f"scores_{index}"] = slot.scores.astype(np.float64)
        arrays["meta_json"] = np.frombuffer(
            json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8
        )
        arrays["state_order"] = np.array(list(self._slots), dtype=np.str_)
        np.savez_compressed(target, **arrays)
        written = target if target.exists() else target.with_suffix(".npz")
        if not written.exists():
            raise RuntimeError(f"teacher bank was not written next to {target}")
        return hashlib.sha256(written.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path: str | Path, *, spec: Sonic78Spec = SONIC78_V1) -> "TeacherBank":
        with np.load(Path(path), allow_pickle=False) as archive:
            raw = {key: archive[key] for key in archive.files}
        meta = json.loads(bytes(raw["meta_json"]).decode("utf-8"))
        if meta.get("schema") != BANK_SCHEMA:
            raise ValueError("not a mosaic_n17 teacher bank")
        bank = cls(spec=spec, generation=int(meta["generation"]))
        if str(meta["contract_sha256"]) != bank.contract_digest():
            raise ValueError("teacher bank contract digest mismatch")
        order = [str(value) for value in raw["state_order"].tolist()]
        for index, state_id in enumerate(order):
            record = dict(meta["slots"][state_id])
            record.pop("checksums", None)
            bank._slots[state_id] = _Slot(
                candidates=np.asarray(raw[f"candidates_{index}"], dtype=np.float64),
                mask=np.asarray(raw[f"mask_{index}"]).astype(bool),
                scores=np.asarray(raw[f"scores_{index}"], dtype=np.float64),
                checksums=tuple(meta["slots"][state_id]["checksums"]),
                status=str(record["status"]),
                generation=int(record["generation"]),
                protected=bool(record["protected"]),
                receipt_sha256=str(record["receipt_sha256"]),
                lineage=dict(record["lineage"]),
            )
        return bank


__all__ = [
    "BANK_SCHEMA",
    "PROVISIONAL",
    "STATUSES",
    "TeacherBank",
    "VERIFIED",
]
