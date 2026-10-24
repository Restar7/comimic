"""Offline data builders: typed-slot targets and teacher LeRobot datasets."""

from __future__ import annotations

from .aux_targets import AuxTargetConfig, build_aux_targets
from .slots_h5 import read_slots_h5, write_slots_h5
from .teacher_lerobot import TEACHER_SCHEMA, TeacherSample, write_teacher_dataset


__all__ = [
    "AuxTargetConfig",
    "TEACHER_SCHEMA",
    "TeacherSample",
    "build_aux_targets",
    "read_slots_h5",
    "write_slots_h5",
    "write_teacher_dataset",
]
