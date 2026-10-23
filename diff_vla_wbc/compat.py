"""Compatibility layer for the cloned SuperMimic-Reproduced repository.

The upstream reproduction imports a handful of quaternion helpers from
``mjlab.utils.lab_api.math`` at module import time.  We do not want to install
the full mjlab / MuJoCo-Warp / Isaac Lab stack just to run the *differentiable*
part of the model, so this module installs a tiny, API-compatible shim before
``supermimic`` is imported.

The math functions are a compact re-implementation of the conventions used by
mjlab (quaternions are ``wxyz``).  They are plain PyTorch so they stay
differentiable inside the joint training graph.

It also centralises the G1 asset / action-scale constants used by both the
world-model reward head and the MuJoCo closed-loop evaluator.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SUPERMIMIC_ROOT = WORKSPACE_ROOT / "third_party" / "SuperMimic-Reproduced"
SUPERMIMIC_SRC = SUPERMIMIC_ROOT / "src"
DEFAULT_G1_XML = (
    WORKSPACE_ROOT
    / "GR00T-WholeBodyControl"
    / "gear_sonic_deploy"
    / "g1"
    / "g1_29dof.xml"
)

# ---------------------------------------------------------------------------
# mjlab.math compatible quaternion helpers (wxyz)
# ---------------------------------------------------------------------------


def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_unique(q: torch.Tensor) -> torch.Tensor:
    return torch.where(q[..., 0:1] < 0, -q, q)


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(q, -1)
    two_s = 2.0 / (q * q).sum(-1).clamp_min(1e-12)
    out = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return out.reshape(q.shape[:-1] + (3, 3))


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = torch.unbind(q1, -1)
    w2, x2, y2, z2 = torch.unbind(q2, -1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        -1,
    )


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    theta = (angle * 0.5).unsqueeze(-1)
    xyz = normalize(axis) * torch.sin(theta)
    w = torch.cos(theta)
    return normalize(torch.cat((w, xyz), dim=-1))


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz = q[..., 1:]
    t = torch.cross(xyz, v, dim=-1) * 2.0
    return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(q), v)


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = torch.unbind(q, -1)
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    out = torch.zeros_like(q)
    out[..., 0] = torch.cos(yaw / 2)
    out[..., 3] = torch.sin(yaw / 2)
    return normalize(out)


def axis_angle_from_quat(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    q = q * (1.0 - 2.0 * (q[..., 0:1] < 0.0))
    mag = torch.linalg.norm(q[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, q[..., 0])
    angle = 2.0 * half_angle
    sin_half_over_angle = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48.0
    )
    return q[..., 1:4] / sin_half_over_angle.unsqueeze(-1)


def quat_box_minus(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return axis_angle_from_quat(quat_mul(q1, quat_conjugate(q2)))


def quat_box_plus(q: torch.Tensor, delta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(eps)
    dq = quat_from_angle_axis(norm.squeeze(-1), delta / norm)
    return quat_unique(quat_mul(dq, q))


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(quat_box_minus(q1, q2), dim=-1)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Import shim
# ---------------------------------------------------------------------------


def install_mjlab_shim() -> None:
    """Expose just enough ``mjlab`` for the pure-torch SuperMimic modules."""
    math_name = "mjlab.utils.lab_api.math"
    if math_name in sys.modules:
        return
    try:
        if importlib.util.find_spec("mjlab") is not None:
            return
    except (ValueError, ModuleNotFoundError):
        pass

    module = types.ModuleType(math_name)
    exported = {
        "normalize": normalize,
        "quat_unique": quat_unique,
        "matrix_from_quat": matrix_from_quat,
        "quat_conjugate": quat_conjugate,
        "quat_inv": quat_inv,
        "quat_mul": quat_mul,
        "quat_from_angle_axis": quat_from_angle_axis,
        "quat_apply": quat_apply,
        "quat_apply_inverse": quat_apply_inverse,
        "yaw_quat": yaw_quat,
        "axis_angle_from_quat": axis_angle_from_quat,
        "quat_box_minus": quat_box_minus,
        "quat_box_plus": quat_box_plus,
        "quat_error_magnitude": quat_error_magnitude,
        "wrap_to_pi": wrap_to_pi,
    }
    for name, fn in exported.items():
        setattr(module, name, fn)
    module.__all__ = tuple(exported)

    packages = {
        "mjlab": [],
        "mjlab.utils": [],
        "mjlab.utils.lab_api": [],
    }
    for name in packages:
        pkg = types.ModuleType(name)
        pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = pkg
    sys.modules["mjlab.utils.lab_api.math"] = module
    sys.modules["mjlab.utils"].lab_api = sys.modules["mjlab.utils.lab_api"]  # type: ignore[attr-defined]
    sys.modules["mjlab.utils.lab_api"].math = module  # type: ignore[attr-defined]


def ensure_supermimic_importable(lightweight: bool = True) -> Path:
    install_mjlab_shim()
    src = str(SUPERMIMIC_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    if not (SUPERMIMIC_SRC / "supermimic").is_dir():
        raise FileNotFoundError(
            f"SuperMimic-Reproduced is missing at {SUPERMIMIC_SRC}. "
            "Run: git clone --depth 1 https://github.com/CuboiLeo/SuperMimic-Reproduced.git "
            "third_party/SuperMimic-Reproduced"
        )
    if lightweight:
        # ``supermimic.envs.__init__`` eagerly imports the mjlab environment
        # stack.  The differentiable bridge only needs the pure-torch MDP /
        # reward / motion-library modules, so give the package a namespace
        # ``__init__`` and let submodules import normally.
        pkg_name = "supermimic.envs"
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(SUPERMIMIC_SRC / "supermimic" / "envs")]  # type: ignore[attr-defined]
            sys.modules[pkg_name] = pkg
    return SUPERMIMIC_SRC


# ---------------------------------------------------------------------------
# G1 constants
# ---------------------------------------------------------------------------

G1_JOINT_NAMES: tuple[str, ...] = (
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
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

TRACKED_BODIES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
ANCHOR_BODY = "torso_link"
ROOT_BODY = "pelvis"

# Armature / effort limits from GR00T-WholeBodyControl's deploy header.
_ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
_EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
_FREQ = 10.0 * 2.0 * math.pi
_STIFFNESS = {k: v * _FREQ * _FREQ for k, v in _ARMATURE.items()}
_DAMPING = {k: 2.0 * 2.0 * v * _FREQ for k, v in _ARMATURE.items()}


def _joint_motor(name: str) -> str:
    if name.endswith(("hip_pitch_joint", "hip_roll_joint", "knee_joint")):
        return "7520_22"
    if name.endswith(("hip_yaw_joint", "waist_yaw_joint")):
        return "7520_14"
    if name.endswith(("wrist_pitch_joint", "wrist_yaw_joint")):
        return "4010"
    return "5020"


def _g1_gain_arrays() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    action_scale, kp, kd, armature = [], [], [], []
    for name in G1_JOINT_NAMES:
        motor = _joint_motor(name)
        action_scale.append(0.25 * _EFFORT[motor] / _STIFFNESS[motor])
        # Ankles and waist roll/pitch are driven by two 5020 units through a linkage.
        parallel = name.endswith(("ankle_pitch_joint", "ankle_roll_joint", "waist_roll_joint", "waist_pitch_joint"))
        kp.append(_STIFFNESS[motor] * (2.0 if parallel else 1.0))
        kd.append(_DAMPING[motor] * (2.0 if parallel else 1.0))
        armature.append(_ARMATURE[motor] * (2.0 if parallel else 1.0))
    return (
        torch.tensor(action_scale, dtype=torch.float32),
        torch.tensor(kp, dtype=torch.float32),
        torch.tensor(kd, dtype=torch.float32),
        torch.tensor(armature, dtype=torch.float32),
    )


G1_ACTION_SCALE, G1_KP, G1_KD, G1_ARMATURE = _g1_gain_arrays()


def g1_xml_path(path: str | Path | None = None) -> Path:
    if path is not None:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    candidates = [
        DEFAULT_G1_XML,
        WORKSPACE_ROOT / "groot_repro" / "scenes" / "g1_29dof_with_hand.xml",
        WORKSPACE_ROOT / "reproduce_scenes" / "g1" / "g1_29dof_with_hand.xml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No G1 MJCF found. Pass --xml /path/to/g1_29dof.xml; expected one at "
        f"{DEFAULT_G1_XML}"
    )


def load_g1_model(path: str | Path | None = None):
    import mujoco

    xml = g1_xml_path(path)
    model = mujoco.MjModel.from_xml_path(str(xml))
    if model.nq != 36 or model.nv != 35:
        raise ValueError(f"Expected 29-DoF G1 (nq=36, nv=35), got nq={model.nq}, nv={model.nv}")
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)
    ]
    if tuple(joint_names) != G1_JOINT_NAMES:
        raise ValueError(
            "G1 joint order mismatch. Expected SuperMimic order; got "
            f"{joint_names[:6]}... Use the unmodified g1_29dof.xml."
        )
    return model


def default_qpos0(model) -> torch.Tensor:
    """MuJoCo ``qpos0`` joint entries (the hinge zero offsets used by FK)."""
    return torch.tensor(model.qpos0[7:36], dtype=torch.float32)


def g1_default_joint_pos() -> torch.Tensor:
    """Training default pose used by mjlab's ``KNEES_BENT_KEYFRAME``.

    SuperMimic's action is a *joint-position offset* around this pose, and the
    68-d goal stores ``q_ref - q_default``.  Keeping this separate from
    ``model.qpos0`` (which is all zeros in the deploy MJCF) is essential.
    """
    q = torch.zeros(len(G1_JOINT_NAMES), dtype=torch.float32)
    for i, name in enumerate(G1_JOINT_NAMES):
        if name.endswith("hip_pitch_joint"):
            q[i] = -0.312
        elif name.endswith("knee_joint"):
            q[i] = 0.669
        elif name.endswith("ankle_pitch_joint"):
            q[i] = -0.363
        elif name.endswith("elbow_joint"):
            q[i] = 0.6
        elif name == "left_shoulder_roll_joint":
            q[i] = 0.2
        elif name == "right_shoulder_roll_joint":
            q[i] = -0.2
        elif name in ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"):
            q[i] = 0.2
    return q


def soft_joint_limits(model, factor: float = 0.9) -> torch.Tensor:
    q0 = g1_default_joint_pos()
    lo = torch.tensor(model.jnt_range[1:30, 0], dtype=torch.float32)
    hi = torch.tensor(model.jnt_range[1:30, 1], dtype=torch.float32)
    return torch.stack((q0 + factor * (lo - q0), q0 + factor * (hi - q0)), dim=-1)


def resolve_paths(paths: Iterable[str | Path]) -> list[Path]:
    out = []
    for p in paths:
        path = Path(p).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        out.append(path)
    return out
