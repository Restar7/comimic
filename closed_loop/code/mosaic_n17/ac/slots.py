"""Typed slot contract shared by Phase A, Phase B and the stage scorer.

The typed representation is ``C6 + B3 + E3 + S2 = 14`` slots of width 384
(FlowBody ``extensions/structured_slot_ac/contracts.py:14-19``).  Keeping the
counts frozen here means a checkpoint trained elsewhere cannot be silently
re-interpreted with a different split.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


SLOT_TYPES: tuple[str, ...] = ("C", "B", "E", "S")
SLOT_COUNTS: Mapping[str, int] = {"C": 6, "B": 3, "E": 3, "S": 2}
SLOT_DIM = 384
TOTAL_SLOTS = 14

if sum(SLOT_COUNTS.values()) != TOTAL_SLOTS:  # pragma: no cover - frozen table
    raise AssertionError("typed slot counts must sum to 14")


def slot_slices(counts: Mapping[str, int] | None = None) -> dict[str, slice]:
    resolved = dict(SLOT_COUNTS if counts is None else counts)
    if set(resolved) != set(SLOT_TYPES):
        raise ValueError(f"slot counts must describe exactly {SLOT_TYPES}")
    offsets: dict[str, slice] = {}
    cursor = 0
    for name in SLOT_TYPES:
        offsets[name] = slice(cursor, cursor + resolved[name])
        cursor += resolved[name]
    return offsets


def validate_slots(slots: object, *, label: str = "slots") -> torch.Tensor:
    """Accept ``[..., 14, 384]`` float slots and return them unchanged."""

    if not torch.is_tensor(slots):
        raise TypeError(f"{label} must be a torch.Tensor, got {type(slots).__name__}")
    if slots.ndim < 2 or tuple(slots.shape[-2:]) != (TOTAL_SLOTS, SLOT_DIM):
        raise ValueError(
            f"{label} must end in ({TOTAL_SLOTS}, {SLOT_DIM}), got {tuple(slots.shape)}"
        )
    if not slots.dtype.is_floating_point:
        raise ValueError(f"{label} must be floating point, got {slots.dtype}")
    if not bool(torch.isfinite(slots).all()):
        raise ValueError(f"{label} must be finite")
    return slots


def validate_slots_np(slots: object, *, label: str = "slots") -> np.ndarray:
    array = np.asarray(slots)
    if array.ndim < 2 or array.shape[-2:] != (TOTAL_SLOTS, SLOT_DIM):
        raise ValueError(f"{label} must end in ({TOTAL_SLOTS}, {SLOT_DIM}), got {array.shape}")
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return array


def split_slots(slots: torch.Tensor) -> dict[str, torch.Tensor]:
    validate_slots(slots)
    return {name: slots[..., span, :] for name, span in slot_slices().items()}


def masked_huber(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, delta: float = 1.0
) -> torch.Tensor:
    """Huber over valid entries; returns a scalar mean or 0 when nothing is valid."""

    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("pred, target and mask must share one shape")
    if mask.dtype != torch.bool:
        raise ValueError("slot mask must be boolean")
    error = pred - target.detach()
    abs_error = error.abs()
    quadratic = torch.clamp(abs_error, max=delta)
    loss = 0.5 * quadratic**2 + delta * (abs_error - quadratic)
    selected = loss[mask]
    if selected.numel() == 0:
        return pred.sum() * 0.0
    return selected.mean()


__all__ = [
    "SLOT_COUNTS",
    "SLOT_DIM",
    "SLOT_TYPES",
    "TOTAL_SLOTS",
    "masked_huber",
    "slot_slices",
    "split_slots",
    "validate_slots",
    "validate_slots_np",
]
