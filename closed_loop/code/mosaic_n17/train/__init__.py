"""Training entries: Phase B consequence model and the VLM aux (world-model) head."""

from __future__ import annotations

from .train_phase_b import EpisodeArrays, build_dataset, train as train_phase_b
from .train_world_model_vlm import AUX_SCHEMA, AuxSlotHead, AuxTrainingConfig, train as train_aux


__all__ = [
    "AUX_SCHEMA",
    "AuxSlotHead",
    "AuxTrainingConfig",
    "EpisodeArrays",
    "build_dataset",
    "train_aux",
    "train_phase_b",
]
