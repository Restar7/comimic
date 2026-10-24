"""Typed consequence model and typed-slot utilities for the SONIC78 route."""

from __future__ import annotations

from .checkpoint import (
    PHASE_B_SCHEMA,
    PhaseBContract,
    load_phase_b_checkpoint,
    save_phase_b_checkpoint,
)
from .phase_a_runtime import PhaseAAdapter, PhaseAContract, sha256_file
from .phase_b_runtime import (
    TARGET_SEMANTICS,
    TARGET_SEMANTICS_EXECUTED_PREFIX,
    TARGET_SEMANTICS_FULL_HORIZON,
    PhaseBConfig,
    Sonic78PhaseB,
    split_action78,
    validate_consequence_supervision,
)
from .scoring import plausibility, score_components, stage_gain, typed_distance
from .slots import (
    SLOT_COUNTS,
    SLOT_DIM,
    SLOT_TYPES,
    TOTAL_SLOTS,
    masked_huber,
    slot_slices,
    split_slots,
    validate_slots,
    validate_slots_np,
)


__all__ = [
    "PHASE_B_SCHEMA",
    "PhaseAAdapter",
    "PhaseAContract",
    "PhaseBConfig",
    "PhaseBContract",
    "SLOT_COUNTS",
    "SLOT_DIM",
    "SLOT_TYPES",
    "Sonic78PhaseB",
    "TARGET_SEMANTICS",
    "TARGET_SEMANTICS_EXECUTED_PREFIX",
    "TARGET_SEMANTICS_FULL_HORIZON",
    "TOTAL_SLOTS",
    "load_phase_b_checkpoint",
    "masked_huber",
    "plausibility",
    "save_phase_b_checkpoint",
    "score_components",
    "sha256_file",
    "slot_slices",
    "split_action78",
    "split_slots",
    "stage_gain",
    "typed_distance",
    "validate_consequence_supervision",
    "validate_slots",
    "validate_slots_np",
]
