"""Independent 14-DoF hand channel, decoupled from the 29-DoF body policy.

Why this exists
---------------
The SuperMimic line tracks a 29-DoF Unitree G1 and emits 29 joint offsets; it has
no notion of fingers (``ACT_DIM = 29``, ``TRACKED_BODIES`` stops at
``*_wrist_yaw_link``).  The SONIC / ``gear_sonic`` line in this workspace keeps a
64-D latent whole-body action and drives the hands through a *separate* 14-D
stream, without touching the 29-DoF action space::

    # gear_sonic/envs/wrapper/manager_env_wrapper.py:246
    def _apply_motion_hand_dof_targets(self):
        \"\"\"Apply the external hand stream without changing SONIC's 29-DOF action space.\"\"\"
        hand_targets = self.motion_command.hand_dof_pos          # (N, 14) joint angles
        hand_targets = torch.clamp(hand_targets, min=..., max=...)  # soft limits
        robot.set_joint_position_target(hand_targets, joint_ids=self._hand_joint_indices)

This module is the SuperMimic-side equivalent: the body keeps producing 29
joint offsets (converted to PD torques exactly like ``diff_vla_wbc/mujoco_env.py``),
and the hands are an independent channel that can be fed by

* ``pose``   -- a named hand pose (``open`` / ``grasp`` / ``pinch`` / ``fist``)
                with slew-rate limiting.  Useful because BONES-SEED's ``g1.csv``
                carries **no finger channels** (36 columns = root + 29 joints);
* ``motion`` -- a ``(T, 14)`` array of hand joint angles in radians, e.g. from a
                43-DOF motion library, clamped to the model joint limits.

Control mapping on ``reproduce_scenes/g1/g1_29dof_with_hand.xml`` (verified):

    ctrl[0:22]   left leg / waist / left arm body actuators   (G1_JOINT_NAMES[0:22])
    ctrl[22:29]  left hand  (thumb_0/1/2, middle_0/1, index_0/1)
    ctrl[29:36]  right arm body actuators                     (G1_JOINT_NAMES[22:29])
    ctrl[36:43]  right hand

Note the hand actuators are **interleaved**, not appended: never assemble the
43-D actuator vector by ``concat(body, hand)``.  Use :meth:`HandStream.assemble_ctrl`,
which scatters by name and validates the model layout at construction time.

Torque convention (same as ``mujoco_env.py``)::

    tau = kp * (q_des - q) - kd * dq,  clipped by the actuator force range.

Example
-------
>>> from diff_vla_wbc.hand_stream import HandStream, HandStreamCfg, load_hand_model
>>> model = load_hand_model()
>>> hand = HandStream(model, HandStreamCfg(pose="open"))
>>> hand.set_pose("grasp")            # smooth close
>>> for _ in range(50):
...     hand.step(0.02)
...     tau = hand.pd_torque(data)    # (14,)
...     ctrl = hand.assemble_ctrl(body_tau_29, tau)   # (43,)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

HAND_JOINT_NAMES: tuple[str, ...] = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
)
HAND_DIM = len(HAND_JOINT_NAMES)  # 14 = 2 x 7 (three-finger hand)

# Joint ranges in the MJCF, for reference (radians):
#   thumb_0  [-1.0472, 1.0472]   thumb_1 [-0.724312, 1.0472]   thumb_2 [0, 1.74533]
#   middle_0 [-1.5708, 0]        middle_1 [-1.74533, 0]
#   index_0  [-1.5708, 0]        index_1  [-1.74533, 0]
_LEFT_GRASP = (0.55, 0.85, 1.25, -1.15, -1.35, -1.15, -1.35)
_LEFT_PINCH = (0.90, 0.50, 1.45, -0.25, -0.35, -0.95, -1.25)
_LEFT_FIST = (1.00, 1.00, 1.70, -1.50, -1.70, -1.50, -1.70)


def _both_sides(left: tuple[float, ...]) -> np.ndarray:
    """Left-hand pose + its mirror image.

    The right hand actuators are geometrically mirrored in the MJCF, so their
    joint ranges are sign-flipped (verified: ``left_hand_middle_0`` is
    ``[-1.5708, 0]`` while ``right_hand_middle_0`` is ``[0, 1.5708]``; same for
    thumb_1/2 and index_0/1).  The same grasp therefore has opposite joint signs
    on the right side, i.e. ``right = -left``.
    """
    arr = np.asarray(left, dtype=np.float64)
    return np.concatenate([arr, -arr])


HAND_POSES: dict[str, np.ndarray] = {
    "open": np.zeros(HAND_DIM, dtype=np.float64),
    "grasp": _both_sides(_LEFT_GRASP),
    "pinch": _both_sides(_LEFT_PINCH),
    "fist": _both_sides(_LEFT_FIST),
}

DEFAULT_HAND_XML = (
    Path(__file__).resolve().parents[1] / "reproduce_scenes" / "g1" / "g1_29dof_with_hand.xml"
)


def load_hand_model(xml_path: str | Path | None = None):
    """Load the 43-actuator G1 MJCF (29 body + 14 hand) and validate its layout."""
    import mujoco

    path = Path(xml_path) if xml_path is not None else DEFAULT_HAND_XML
    model = mujoco.MjModel.from_xml_path(str(path))
    if model.nu != 43:
        raise ValueError(f"expected 43 actuators (29 body + 14 hand), got {model.nu} in {path}")
    return model


def _body_joint_names() -> tuple[str, ...]:
    """The 29 body joint names, in the order the SuperMimic action uses.

    Prefer ``diff_vla_wbc.compat`` (torch-only, no mjlab); fall back to the
    upstream module, which is only importable when SuperMimic's optional deps
    are installed.
    """
    try:
        from .compat import G1_JOINT_NAMES

        return tuple(G1_JOINT_NAMES)
    except Exception:  # pragma: no cover - depends on optional installs
        from supermimic.data.motion_convert import G1_JOINT_NAMES as names

        return tuple(names)


@dataclass
class HandStreamCfg:
    source: Literal["pose", "motion", "vla"] = "pose"
    pose: str = "open"
    kp: float = 8.0  # Nm/rad; small finger links, actuator force range is ~1.4 Nm
    kd: float = 0.4  # Nm*s/rad
    slew_rate: float = 6.0  # rad/s used when blending between poses
    motion_fps: float = 50.0
    motion: np.ndarray | None = None  # (T, 14) radians, required when source == "motion"
    max_target_step: float = 0.25  # rad; hard cap per step() call
    warn: bool = True


class HandStream:
    """14-DoF hand command generator + PD torque mapper for a 29/43-DoF G1 MJCF."""

    def __init__(self, model, cfg: HandStreamCfg | None = None):
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.cfg = cfg or HandStreamCfg()
        self.nu = int(model.nu)
        self._index = self._resolve(model)
        self.enabled = bool(self._index["enabled"])
        self._pose_goal = HAND_POSES.get(self.cfg.pose, HAND_POSES["open"]).copy()
        self._target = self._pose_goal.copy()
        self._motion = None
        #: Last 14-D target written by :meth:`command` (``source="vla"``).
        self._vla_target = HAND_POSES.get(self.cfg.pose, HAND_POSES["open"]).copy()
        if self.enabled and self.cfg.source == "motion":
            if self.cfg.motion is None:
                raise ValueError('source="motion" requires cfg.motion, shape (T, 14)')
            m = np.asarray(self.cfg.motion, dtype=np.float64)
            if m.ndim != 2 or m.shape[1] != HAND_DIM:
                raise ValueError(f"cfg.motion must be (T, {HAND_DIM}), got {m.shape}")
            self._motion = np.clip(m, self._index["lo"], self._index["hi"])
        elif self.enabled and self.cfg.source not in ("pose", "vla"):
            raise ValueError(f'unknown source {self.cfg.source!r}; use "pose", "motion" or "vla"')

    # ------------------------------------------------------------- model layout

    def _resolve(self, model) -> dict:
        mj = self.mujoco
        joint_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        missing = [n for n in HAND_JOINT_NAMES if n not in joint_names]
        if missing:
            if self.cfg.warn:
                print(
                    f"[hand_stream] model has no hand joints ({len(missing)} missing); "
                    "hand channel disabled — body-only control stays unchanged"
                )
            return {"enabled": False}

        jids = [joint_names.index(n) for n in HAND_JOINT_NAMES]
        # Hand joints must appear in the model in exactly HAND_JOINT_NAMES order.
        hand_block = [n for n in joint_names if n.startswith(("left_hand", "right_hand"))]
        if tuple(hand_block) != HAND_JOINT_NAMES:
            raise ValueError(
                "hand joint order in the model differs from HAND_JOINT_NAMES: "
                f"{hand_block}"
            )

        # Actuator -> joint map; hand actuators are interleaved with body ones.
        act_joint = [int(model.actuator_trnid[i, 0]) for i in range(model.nu)]
        ctrl_ids = []
        for jid in jids:
            hits = [i for i, tj in enumerate(act_joint) if tj == jid]
            if len(hits) != 1:
                raise ValueError(
                    f"expected exactly one actuator for {joint_names[jid]}, found {len(hits)}"
                )
            ctrl_ids.append(hits[0])

        body_ids = [i for i in range(model.nu) if i not in ctrl_ids]
        body_joint_names = [joint_names[act_joint[i]] for i in body_ids]
        body_order = _body_joint_names()

        if tuple(body_joint_names) != body_order:
            raise ValueError(
                "body actuator order differs from G1_JOINT_NAMES; refusing to build ctrl "
                f"(got {body_joint_names[:3]}...)"
            )

        lo = model.jnt_range[jids, 0].astype(np.float64).copy()
        hi = model.jnt_range[jids, 1].astype(np.float64).copy()
        frc_cols = np.asarray(model.jnt_actfrcrange[jids], dtype=np.float64).reshape(len(jids), -1)
        frc = np.nanmax(np.abs(frc_cols), axis=1)
        if not np.all(np.isfinite(frc)) or not np.any(frc > 0):
            frc = np.full(HAND_DIM, np.inf)
        frc[frc <= 0] = np.inf
        return {
            "enabled": True,
            "jids": jids,
            "qpos_adr": model.jnt_qposadr[jids].astype(int),
            "dof_adr": model.jnt_dofadr[jids].astype(int),
            "ctrl_ids": ctrl_ids,
            "body_ctrl_ids": body_ids,
            "lo": lo,
            "hi": hi,
            "effort": frc,
        }

    # ------------------------------------------------------------------- source

    def set_pose(self, name: str, *, immediate: bool = False) -> None:
        """Switch the target hand pose (smooth by default, slew-limited in step())."""
        if name not in HAND_POSES:
            raise KeyError(f"unknown pose {name!r}; available: {sorted(HAND_POSES)}")
        self._pose_goal = HAND_POSES[name].copy()
        if immediate:
            self._target = self._pose_goal.copy()

    def command(self, targets) -> np.ndarray:
        """Set hand targets directly (``source="vla"``: the VLA's raw output).

        This is the "used directly" path -- values are clamped to the model's
        joint limits here and nowhere else.
        """
        if not self.enabled:
            return self.targets()
        v = np.asarray(targets, dtype=np.float64).reshape(-1)
        if v.shape[0] != HAND_DIM:
            raise ValueError(f"expected {HAND_DIM} hand joint angles, got {v.shape[0]}")
        self._vla_target = np.clip(v, self._index["lo"], self._index["hi"])
        return self.targets()

    def step(self, dt: float) -> np.ndarray:
        """Advance the pose blend by ``dt`` seconds. Returns the current target."""
        if not self.enabled or self.cfg.source != "pose":
            return self.targets()
        max_step = min(float(self.cfg.slew_rate) * float(dt), float(self.cfg.max_target_step))
        delta = np.clip(self._pose_goal - self._target, -max_step, max_step)
        self._target = self._target + delta
        self._target = np.clip(self._target, self._index["lo"], self._index["hi"])
        return self.targets()

    def targets(self, frame: int | None = None) -> np.ndarray:
        """Current 14-D hand joint targets in radians (HAND_JOINT_NAMES order)."""
        if not self.enabled:
            return np.zeros(HAND_DIM, dtype=np.float64)
        if self._motion is not None:
            idx = 0 if frame is None else int(frame)
            idx = max(0, min(idx, self._motion.shape[0] - 1))
            return self._motion[idx].copy()
        if self.cfg.source == "vla":
            return self._vla_target.copy()
        return self._target.copy()

    # ------------------------------------------------------------------ control

    def read_state(self, data) -> tuple[np.ndarray, np.ndarray]:
        """Hand joint positions / velocities from ``mujoco.MjData``."""
        if not self.enabled:
            return np.zeros(HAND_DIM), np.zeros(HAND_DIM)
        q = np.asarray(data.qpos[self._index["qpos_adr"]], dtype=np.float64).copy()
        dq = np.asarray(data.qvel[self._index["dof_adr"]], dtype=np.float64).copy()
        return q, dq

    def pd_torque(self, data, targets: np.ndarray | None = None) -> np.ndarray:
        """PD torque for the 14 hand joints, clipped to the actuator force range."""
        if not self.enabled:
            return np.zeros(HAND_DIM, dtype=np.float64)
        q, dq = self.read_state(data)
        q_des = self.targets() if targets is None else np.asarray(targets, dtype=np.float64)
        q_des = np.clip(q_des, self._index["lo"], self._index["hi"])
        tau = self.cfg.kp * (q_des - q) - self.cfg.kd * dq
        return np.clip(tau, -self._index["effort"], self._index["effort"])

    def assemble_ctrl(
        self, body_tau, hand_tau=None, data=None
    ) -> np.ndarray:
        """Build the full ``(nu,)`` actuator vector.

        Parameters
        ----------
        body_tau : array of 29 torques in ``G1_JOINT_NAMES`` order.
        hand_tau : optional array of 14 torques; computed from ``data`` when omitted.
        data     : ``MjData`` used to compute the hand PD torque if ``hand_tau`` is None.
        """
        ctrl = np.zeros(self.nu, dtype=np.float64)
        body = np.asarray(body_tau, dtype=np.float64).reshape(-1)
        if self.enabled:
            if body.shape[0] != len(self._index["body_ctrl_ids"]):
                raise ValueError(f"body_tau must have 29 entries, got {body.shape[0]}")
            ctrl[self._index["body_ctrl_ids"]] = body
            if hand_tau is None:
                if data is None:
                    raise ValueError("pass hand_tau or data to compute hand torques")
                hand_tau = self.pd_torque(data)
            hand = np.asarray(hand_tau, dtype=np.float64).reshape(-1)
            if hand.shape[0] != HAND_DIM:
                raise ValueError(f"hand_tau must have {HAND_DIM} entries, got {hand.shape[0]}")
            ctrl[self._index["ctrl_ids"]] = hand
        else:
            if body.shape[0] != self.nu:
                raise ValueError(f"body-only model expects {self.nu} torques, got {body.shape[0]}")
            ctrl[:] = body
        return ctrl

    # -------------------------------------------------------------------- misc

    @property
    def ctrl_ids(self) -> list[int]:
        """Actuator indices of the 14 hand joints (empty when disabled)."""
        return list(self._index.get("ctrl_ids", []))

    def info(self) -> dict:
        return {
            "enabled": self.enabled,
            "source": self.cfg.source,
            "pose": self.cfg.pose,
            "hand_ctrl_ids": self.ctrl_ids,
            "body_ctrl_ids": list(self._index.get("body_ctrl_ids", [])),
            "kp": self.cfg.kp,
            "kd": self.cfg.kd,
        }


class G1BodyIndex:
    """Name-based body joint addresses for a 29- or 43-actuator G1 MJCF.

    The 43-D actuator model interleaves the hands with the arms (see the module
    docstring), so the body state is **not** the contiguous ``qpos[7:36]`` slice
    that works for the 29-DoF deploy MJCF.  This helper resolves the 29 body
    joints by name and keeps the SuperMimic action order.
    """

    def __init__(self, model):
        import mujoco

        joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        body = _body_joint_names()
        missing = [n for n in body if n not in joint_names]
        if missing:
            raise ValueError(f"model is missing body joints: {missing}")
        jids = [joint_names.index(n) for n in body]
        self.joint_names = body
        self.qpos_idx = np.asarray([model.jnt_qposadr[j] for j in jids], dtype=int)
        self.dof_idx = np.asarray([model.jnt_dofadr[j] for j in jids], dtype=int)
        act_joint = [int(model.actuator_trnid[i, 0]) for i in range(model.nu)]
        self.ctrl_idx = np.asarray([act_joint.index(j) for j in jids], dtype=int)
        self.lo = model.jnt_range[jids, 0].astype(np.float64)
        self.hi = model.jnt_range[jids, 1].astype(np.float64)
        self.nu = int(model.nu)

    def q(self, data) -> np.ndarray:
        return np.asarray(data.qpos[self.qpos_idx], dtype=np.float64).copy()

    def dq(self, data) -> np.ndarray:
        return np.asarray(data.qvel[self.dof_idx], dtype=np.float64).copy()

    def write_state(self, data, q: np.ndarray, dq: np.ndarray | None = None) -> None:
        data.qpos[self.qpos_idx] = q
        if dq is not None:
            data.qvel[self.dof_idx] = dq

    def assemble_ctrl(self, body_tau) -> np.ndarray:
        ctrl = np.zeros(self.nu, dtype=np.float64)
        ctrl[self.ctrl_idx] = np.asarray(body_tau, dtype=np.float64).reshape(-1)
        return ctrl
