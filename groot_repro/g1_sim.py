"""Minimal MuJoCo driver for the G1 29-DOF + hands model.

Constants mirror gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp
(MuJoCo joint order, which is what `q_target` from the decoder is expressed in).
"""

from __future__ import annotations

import numpy as np

# --- motor model -----------------------------------------------------------
ARMATURE = {
    "5020": 0.003609725,
    "7520_14": 0.010177520,
    "7520_22": 0.025101925,
    "4010": 0.00425,
}
EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
NATURAL_FREQ = 10 * 2.0 * np.pi
DAMPING_RATIO = 2.0

MOTOR = (
    ["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020"]  # left leg
    + ["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020"]  # right leg
    + ["7520_14", "5020", "5020"]  # waist
    + ["5020", "5020", "5020", "5020", "5020", "4010", "4010"]  # left arm
    + ["5020", "5020", "5020", "5020", "5020", "4010", "4010"]  # right arm
)

KP = np.array(
    [
        ARMATURE[m] * NATURAL_FREQ**2 * (2.0 if i in (4, 5, 10, 11, 13, 14) else 1.0)
        for i, m in enumerate(MOTOR)
    ]
)
KD = np.array(
    [
        2.0 * DAMPING_RATIO * ARMATURE[m] * NATURAL_FREQ * (2.0 if i in (4, 5, 10, 11, 13, 14) else 1.0)
        for i, m in enumerate(MOTOR)
    ]
)
ACTION_SCALE = np.array([0.25 * EFFORT[m] / (ARMATURE[m] * NATURAL_FREQ**2) for m in MOTOR])

DEFAULT_ANGLES = np.array(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)

# IsaacLab order -> MuJoCo/hardware order, verbatim from policy_parameters.hpp
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)


def quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """w, x, y, z -> 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def projected_gravity(q_wxyz: np.ndarray) -> np.ndarray:
    """Gravity direction expressed in the body frame (IsaacLab convention)."""
    rot = quat_to_rotmat(q_wxyz)
    return rot.T @ np.array([0.0, 0.0, -1.0])


# ---------------------------------------------------------------------------
# Scene-index mapping.
#
# `default_angles`, `action_scale` and the decoder output are in MuJoCo joint
# order (legs, waist, left arm, right arm).  The deployment MJCFs however store
# joints/actuators in the *dataset* order:
#     left_leg, right_leg, waist, left_arm, LEFT HAND, right_arm, RIGHT HAND
# so every qpos/ctrl write has to go through a name lookup, never a slice.
# ---------------------------------------------------------------------------

BODY_JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
HAND_JOINT_NAMES = [
    f"{side}_hand_{finger}_joint" for side in ("left", "right")
    for finger in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1",
                   "index_0", "index_1")
]


class SceneMap:
    """Name-based qpos/qvel/actuator indices for a MuJoCo G1 scene."""

    def __init__(self, model):
        import mujoco

        def look(obj, name):
            i = mujoco.mj_name2id(model, obj, name)
            if i < 0:
                raise KeyError(f"{name} not found")
            return i

        self.body_q = [int(model.jnt_qposadr[look(mujoco.mjtObj.mjOBJ_JOINT, n + "_joint")])
                       for n in BODY_JOINT_NAMES]
        self.body_v = [int(model.jnt_dofadr[look(mujoco.mjtObj.mjOBJ_JOINT, n + "_joint")])
                       for n in BODY_JOINT_NAMES]
        self.body_a = [look(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in BODY_JOINT_NAMES]

        self.hand_q = [int(model.jnt_qposadr[look(mujoco.mjtObj.mjOBJ_JOINT, n)])
                       for n in HAND_JOINT_NAMES]
        self.hand_v = [int(model.jnt_dofadr[look(mujoco.mjtObj.mjOBJ_JOINT, n)])
                       for n in HAND_JOINT_NAMES]
        self.hand_a = [look(mujoco.mjtObj.mjOBJ_ACTUATOR, n.replace("_joint", ""))
                       for n in HAND_JOINT_NAMES]

    def q(self, data):
        return data.qpos[self.body_q]

    def dq(self, data):
        return data.qvel[self.body_v]

    def hands(self, data):
        return data.qpos[self.hand_q]

    def hand_dq(self, data):
        return data.qvel[self.hand_v]

    def set_q(self, data, values):
        data.qpos[self.body_q] = values

    def set_hands(self, data, values):
        data.qpos[self.hand_q] = values

    def set_ctrl(self, data, values):
        data.ctrl[self.body_a] = values

    def set_hand_ctrl(self, data, values):
        data.ctrl[self.hand_a] = values
