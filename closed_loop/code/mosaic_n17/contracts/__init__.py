"""Versioned contracts for the SONIC78 route (action, state, time, provenance).

Every consumer of a candidate action, a state vector or a control tick must go
through these modules.  They are deliberately dependency-light (numpy only) so
the contracts can be validated without loading a 3B policy.
"""

from __future__ import annotations

from .provenance import (
    PLACEHOLDER_PROVENANCE,
    VALID_PROVENANCE,
    ProvenanceRecord,
    validate_provenance,
)
from .sonic78 import (
    ACTION78_DIM,
    LEFT_HAND_DIM,
    LEFT_HAND_SLICE,
    MOTION_SLICE,
    MOTION_TOKEN_DIM,
    RIGHT_HAND_DIM,
    RIGHT_HAND_SLICE,
    SONIC78_LOW_LATENCY,
    SONIC78_V1,
    TOKEN_ABS_BOUND,
    Sonic78Spec,
)
from .state46 import (
    AC_STATE46_DIM,
    AC_STATE46_LAYOUT,
    G1_JOINT_ORDER_43,
    N17_STATE46_DIM,
    N17_STATE46_LAYOUT,
    joint_order_sha256,
    joints43_to_ac_state46,
    n17_state46_to_ac_state46,
    quaternion_wxyz_to_rpy,
    split_sonic43,
)
from .timebase import PlanCommitment, TickSchedule


__all__ = [
    "ACTION78_DIM",
    "AC_STATE46_DIM",
    "AC_STATE46_LAYOUT",
    "G1_JOINT_ORDER_43",
    "LEFT_HAND_DIM",
    "LEFT_HAND_SLICE",
    "MOTION_SLICE",
    "MOTION_TOKEN_DIM",
    "N17_STATE46_DIM",
    "N17_STATE46_LAYOUT",
    "PLACEHOLDER_PROVENANCE",
    "PlanCommitment",
    "ProvenanceRecord",
    "RIGHT_HAND_DIM",
    "RIGHT_HAND_SLICE",
    "SONIC78_LOW_LATENCY",
    "SONIC78_V1",
    "Sonic78Spec",
    "TOKEN_ABS_BOUND",
    "TickSchedule",
    "VALID_PROVENANCE",
    "joint_order_sha256",
    "joints43_to_ac_state46",
    "n17_state46_to_ac_state46",
    "quaternion_wxyz_to_rpy",
    "split_sonic43",
    "validate_provenance",
]
