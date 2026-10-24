"""Observation adapters between N1.7's ``state46`` and the typed-AC ``state46``.

Both vectors have 46 entries but they are *different* vectors:

* N1.7 (``unitree_g1_sonic``): ``observation.state[43]`` in SONIC wbc order
  (left body22 + left hand7 + right body7 + right hand7) followed by
  ``observation.projected_gravity[3]``.
* typed AC: ``[root_rpy3, leg_waist15, arm14, hand14]`` with the hand order
  ``thumb/index/middle``.

Feeding the N1.7 vector into the AC would silently swap root attitude for
gravity and permute the hands, so the conversion lives here, is joint-name
based, and fails closed on any shape mismatch.  ``root_rpy`` comes from
``observation.root_orientation`` (a world-frame unit quaternion); it is never
reconstructed from gravity or from measured root positions.
"""

from __future__ import annotations

import hashlib

import numpy as np


N17_STATE46_LAYOUT = "n17_joints43_projected_gravity3"
N17_STATE46_DIM = 46
AC_STATE46_LAYOUT = "sonic_state46_root_rpy3_leg15_arm14_hand14"
AC_STATE46_DIM = 46

#: ``observation.state[43]`` order in the SONIC parquet (same order as
#: ``action.wbc``); left body22 + left hand7 + right body7 + right hand7.
G1_JOINT_ORDER_43 = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
)

#: Dex3 order used by the SONIC parquet hands (``index, middle, thumb``).
OFFICIAL_DEX3_LEFT_HAND_JOINT_NAMES = (
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
)
OFFICIAL_DEX3_RIGHT_HAND_JOINT_NAMES = (
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
)
#: AC / SIMPLE hand order (``thumb, index, middle``).
SIMPLE_LEFT_HAND_JOINT_NAMES = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
)
SIMPLE_RIGHT_HAND_JOINT_NAMES = tuple(
    name.replace("left_", "right_") for name in SIMPLE_LEFT_HAND_JOINT_NAMES
)

SONIC_WBC43_BODY_INDICES = tuple(range(0, 22)) + tuple(range(29, 36))
SONIC_WBC43_LEFT_HAND_INDICES = tuple(range(22, 29))
SONIC_WBC43_RIGHT_HAND_INDICES = tuple(range(36, 43))
LEG_WAIST_SLICE = slice(0, 15)
LEFT_ARM_SLICE = slice(15, 22)
RIGHT_ARM_SLICE = slice(29, 36)


def _official_to_simple(
    prefix: str, official: tuple[str, ...], simple: tuple[str, ...]
) -> tuple[int, ...]:
    index = {name: position for position, name in enumerate(official)}
    try:
        permutation = tuple(index[name] for name in simple)
    except KeyError as exc:  # pragma: no cover - the tables are frozen above
        raise ValueError(f"{prefix} hand order table is inconsistent: missing {exc}") from exc
    if len(set(permutation)) != 7:
        raise ValueError(f"{prefix} hand permutation must be one-to-one")
    return permutation


#: ``hand_simple = hand_official[..., SIMPLE_TO_OFFICIAL]``
SIMPLE_TO_OFFICIAL_LEFT = _official_to_simple(
    "left", OFFICIAL_DEX3_LEFT_HAND_JOINT_NAMES, SIMPLE_LEFT_HAND_JOINT_NAMES
)
SIMPLE_TO_OFFICIAL_RIGHT = _official_to_simple(
    "right", OFFICIAL_DEX3_RIGHT_HAND_JOINT_NAMES, SIMPLE_RIGHT_HAND_JOINT_NAMES
)

if len(G1_JOINT_ORDER_43) != 43:  # pragma: no cover - frozen table
    raise ValueError("SONIC wbc43 order must contain exactly 43 names")
if tuple(G1_JOINT_ORDER_43[index] for index in SONIC_WBC43_BODY_INDICES) != tuple(
    name for name in G1_JOINT_ORDER_43 if "hand_" not in name
):
    raise ValueError("wbc43 body indices must reproduce the 29 body joints in order")


def _as_float_array(value: object, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object or array.dtype.kind not in "fiu":
        raise TypeError(f"{label} must be numeric, got dtype {array.dtype}")
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return array


def quaternion_wxyz_to_matrix(quaternion: object) -> np.ndarray:
    """Unit-quaternion (w, x, y, z) to rotation matrix, mirroring FlowBody."""

    quat = _as_float_array(quaternion, "quaternion_wxyz")
    if quat.shape[-1] != 4:
        raise ValueError(f"quaternion_wxyz must end in 4, got {quat.shape}")
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("quaternion_wxyz must have non-zero norm")
    w, x, y, z = np.moveaxis(quat / norm, -1, 0)
    matrix = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def matrix_to_rpy(rotation: object) -> np.ndarray:
    """``[roll, pitch, yaw]`` with FlowBody's convention; singular pitch raises."""

    matrix = _as_float_array(rotation, "rotation")
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must end in (3, 3), got {matrix.shape}")
    pitch = np.arcsin(np.clip(-matrix[..., 2, 0], -1.0, 1.0))
    if np.any(np.abs(np.cos(pitch)) < 1e-7):
        raise ValueError("RPY conversion is singular at pitch +/- pi/2")
    return np.stack(
        [
            np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2]),
            pitch,
            np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0]),
        ],
        axis=-1,
    )


def quaternion_wxyz_to_rpy(quaternion: object) -> np.ndarray:
    """World-frame unit quaternion (w, x, y, z) to ``[roll, pitch, yaw]``."""

    return matrix_to_rpy(quaternion_wxyz_to_matrix(quaternion))


def split_sonic43(values: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split raw ``[..., 43]`` SONIC wbc/state into ``(body29, left7, right7)``."""

    array = _as_float_array(values, "sonic43")
    if array.shape[-1] != 43:
        raise ValueError(f"SONIC wbc/state must end in 43, got {array.shape}")
    return (
        array[..., list(SONIC_WBC43_BODY_INDICES)],
        array[..., list(SONIC_WBC43_LEFT_HAND_INDICES)],
        array[..., list(SONIC_WBC43_RIGHT_HAND_INDICES)],
    )


def reorder_hands_official_to_simple(hands14: object) -> np.ndarray:
    """``[..., 14]`` official Dex3 hands -> AC ``thumb/index/middle`` order."""

    array = _as_float_array(hands14, "hands14")
    if array.shape[-1] != 14:
        raise ValueError(f"hands14 must end in 14, got {array.shape}")
    left = array[..., :7][..., list(SIMPLE_TO_OFFICIAL_LEFT)]
    right = array[..., 7:][..., list(SIMPLE_TO_OFFICIAL_RIGHT)]
    return np.concatenate((left, right), axis=-1)


def n17_state46_to_ac_state46(state46: object, root_quat_w: object) -> np.ndarray:
    """Convert N1.7 ``state46`` + root quaternion into the typed-AC ``state46``.

    ``projected_gravity`` is dropped on purpose: the AC was trained on root RPY,
    and gravity is not a substitute for it.
    """

    state = _as_float_array(state46, "n17_state46")
    if state.shape[-1] != N17_STATE46_DIM:
        raise ValueError(f"N1.7 state46 must end in {N17_STATE46_DIM}, got {state.shape}")
    quat = _as_float_array(root_quat_w, "root_quat_w")
    if quat.shape[:-1] != state.shape[:-1]:
        raise ValueError(
            f"root_quat_w leading shape {quat.shape[:-1]} must match state46 {state.shape[:-1]}"
        )
    joints = state[..., :43]
    body29, left7, right7 = split_sonic43(joints)
    # ``body29`` is leg12 + waist3 + left_arm7 + right_arm7 in MuJoCo order.
    leg_waist15 = body29[..., LEG_WAIST_SLICE]
    arm14 = np.concatenate((body29[..., 15:22], body29[..., 22:29]), axis=-1)
    hand14 = reorder_hands_official_to_simple(np.concatenate((left7, right7), axis=-1))
    root_rpy3 = matrix_to_rpy(quaternion_wxyz_to_matrix(quat))
    return np.concatenate((root_rpy3, leg_waist15, arm14, hand14), axis=-1)


def joints43_to_ac_state46(joints43: object, root_quat_w: object) -> np.ndarray:
    """Convert SONIC ``joints43`` + root quaternion into the typed-AC ``state46``.

    Used wherever the projected gravity is not part of the observation (e.g. the
    Phase B training reader): the AC never consumes it, so requiring a fabricated
    46D vector would invent data.
    """

    joints = _as_float_array(joints43, "joints43")
    if joints.shape[-1] != 43:
        raise ValueError(f"joints43 must end in 43, got {joints.shape}")
    quat = _as_float_array(root_quat_w, "root_quat_w")
    if quat.shape[:-1] != joints.shape[:-1]:
        raise ValueError(
            f"root_quat_w leading shape {quat.shape[:-1]} must match joints43 {joints.shape[:-1]}"
        )
    body29, left7, right7 = split_sonic43(joints)
    leg_waist15 = body29[..., LEG_WAIST_SLICE]
    arm14 = np.concatenate((body29[..., 15:22], body29[..., 22:29]), axis=-1)
    hand14 = reorder_hands_official_to_simple(np.concatenate((left7, right7), axis=-1))
    root_rpy3 = matrix_to_rpy(quaternion_wxyz_to_matrix(quat))
    return np.concatenate((root_rpy3, leg_waist15, arm14, hand14), axis=-1)


def joint_order_sha256() -> str:
    """Canonical digest of the 43-body order and the two hand permutations."""

    payload = "|".join(
        (
            ",".join(G1_JOINT_ORDER_43),
            ",".join(str(index) for index in SIMPLE_TO_OFFICIAL_LEFT),
            ",".join(str(index) for index in SIMPLE_TO_OFFICIAL_RIGHT),
            ",".join(str(index) for index in SONIC_WBC43_BODY_INDICES),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "AC_STATE46_DIM",
    "AC_STATE46_LAYOUT",
    "G1_JOINT_ORDER_43",
    "LEG_WAIST_SLICE",
    "LEFT_ARM_SLICE",
    "N17_STATE46_DIM",
    "N17_STATE46_LAYOUT",
    "OFFICIAL_DEX3_LEFT_HAND_JOINT_NAMES",
    "OFFICIAL_DEX3_RIGHT_HAND_JOINT_NAMES",
    "RIGHT_ARM_SLICE",
    "SIMPLE_LEFT_HAND_JOINT_NAMES",
    "SIMPLE_RIGHT_HAND_JOINT_NAMES",
    "SIMPLE_TO_OFFICIAL_LEFT",
    "SIMPLE_TO_OFFICIAL_RIGHT",
    "SONIC_WBC43_BODY_INDICES",
    "SONIC_WBC43_LEFT_HAND_INDICES",
    "SONIC_WBC43_RIGHT_HAND_INDICES",
    "joint_order_sha256",
    "joints43_to_ac_state46",
    "matrix_to_rpy",
    "n17_state46_to_ac_state46",
    "quaternion_wxyz_to_matrix",
    "quaternion_wxyz_to_rpy",
    "reorder_hands_official_to_simple",
    "split_sonic43",
]
