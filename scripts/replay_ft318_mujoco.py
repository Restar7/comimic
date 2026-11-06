#!/usr/bin/env python3
"""Replay a pinned ft318 episode in MuJoCo through the SONIC deploy stack.

Three arms (``--mode``), all driven by one episode of the 318-episode dataset:

``target``
    ``ctrl = action.wbc[:29]`` (the recorded joint *targets*, STANDARD order).
    Pure plant + PD test: if this cannot hold the recorded motion, the MuJoCo
    plant is not the plant that produced the data, and neither token nor VLA
    replay can be trusted.

``token``
    ``ctrl = DEFAULT + ACTION_SCALE * decoder(`` token ``)`` with the official
    ``model_decoder.onnx`` and a closed-loop 10-frame history built from the
    *simulated* robot (exactly the deploy loop, only the token comes from disk).
    This is the "replay route": dataset motion token -> SONIC decoder -> MuJoCo.

``recorded_obs``
    The decoder is fed the *recorded* state history (open loop, no feedback).
    Diagnostic only: it isolates decoder+target from plant dynamics.

Everything (PD gains, action scale, joint maps, default pose, control rate) is
imported from ``groot_repro/g1_sim.py`` / ``sonic_loop.py``, i.e. copied from
``gear_sonic_deploy/.../policy_parameters.hpp``.

Dataset layout (verified by FK against ``observation.eef_state`` on 2026-11-04,
see ``docs/MUJOCO_SONIC_REPLAY_PLAN_20261104.md``):

    observation.state / action.wbc (43) =
        body[0:22] (STANDARD_29 order) | left_hand(7) | body[22:29] | right_hand(7)

The 29 body joints are **STANDARD_29** (MuJoCo MJCF order, left leg -> right
leg -> waist -> left arm -> right arm), *not* the interleaved IsaacLab order;
the decoder itself still emits IsaacLab order, hence ``ISAACLAB_TO_MUJOCO``.

Usage::

    PY=third_party/SuperMimic-Reproduced/.venv/bin/python
    $PY scripts/replay_ft318_mujoco.py --mode target --episodes 0 --steps 600 \
        --out runs/mujoco_replay/target_ep0.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "groot_repro"))

import mujoco  # noqa: E402

import g1_sim as G  # noqa: E402
import joint_contract as JC  # noqa: E402
from sonic_loop import HISTORY, SIGNALS, Decoder  # noqa: E402

WS = Path("/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace")

#: HA (HumanoidArena) actuator table -- the plant the SONIC data was recorded in.
#: Copied from ``trackerlab-supermimic/.../supermimic/robot.py::HA_ACTUATOR_TABLE``
#: (itself verbatim from HumanoidArena ``robots/unitree.py::G129_CFG_WITH_DEX3_WHOLEBODY``).
#: (effort_limit, stiffness, damping) per joint-family
HA_ACTUATORS = {
    "hip_pitch": (100.0, 130.0, 3.0),
    "hip_roll": (110.0, 100.0, 3.0),
    "hip_yaw": (100.0, 100.0, 2.0),
    "knee": (150.0, 160.0, 4.5),
    "ankle_pitch": (40.0, 60.0, 6.5),
    "ankle_roll": (40.0, 40.0, 2.0),
    "waist_yaw": (150.0, 150.0, 4.0),
    "waist_roll": (150.0, 150.0, 4.5),
    "waist_pitch": (150.0, 150.0, 8.0),
    "shoulder_pitch": (60.0, 40.0, 5.0),
    "shoulder_roll": (60.0, 40.0, 5.0),
    "shoulder_yaw": (60.0, 40.0, 5.0),
    "elbow": (60.0, 40.0, 5.0),
    "wrist_roll": (60.0, 20.0, 1.0),
    "wrist_pitch": (60.0, 20.0, 1.0),
    "wrist_yaw": (60.0, 20.0, 1.0),
}


def ha_joint_suffix(name: str) -> str:
    for suffix in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll",
                   "waist_yaw", "waist_roll", "waist_pitch", "shoulder_pitch", "shoulder_roll",
                   "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"):
        if name.endswith(suffix + "_joint") or name.endswith(suffix):
            return suffix
    raise KeyError(name)
DEFAULT_SCENE = WS / "GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
DEFAULT_DECODER = WS / "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"
MANIFEST = ROOT / "data/sonic_datasets.json"


# --------------------------------------------------------------------------- data
def dataset_dirs() -> list[tuple[str, Path]]:
    manifest = json.loads(MANIFEST.read_text())
    active = manifest["sets"][manifest["active"]]
    root = Path(active["root"])
    return [(d["task_id"], root / d["path"]) for d in active["datasets"]]


def load_episode(set_index: int, episode: int) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pa

    task, d = dataset_dirs()[set_index]
    files = sorted((d / "data" / "chunk-000").glob("*.parquet"))
    if not files:
        # some subsets nest one level deeper
        files = sorted(d.glob("data/chunk-*/**/*.parquet"))
    if episode >= len(files):
        raise IndexError(f"{task}: episode {episode} >= {len(files)}")
    table = pa.read_table(files[episode]).to_pydict()

    def col(name: str) -> np.ndarray:
        return np.asarray(table[name], dtype=np.float64)

    state = col("observation.state")
    wbc = col("action.wbc")
    # 43 -> body(29) in STANDARD order, hands split out
    body_state = np.concatenate([state[:, 0:22], state[:, 29:36]], axis=1)
    body_wbc = np.concatenate([wbc[:, 0:22], wbc[:, 29:36]], axis=1)
    hands_state = np.concatenate([state[:, 22:29], state[:, 36:43]], axis=1)
    hands_wbc = np.concatenate([wbc[:, 22:29], wbc[:, 36:43]], axis=1)
    return {
        "task": np.array([task]),
        "file": np.array([str(files[episode])]),
        "n": np.array([state.shape[0]]),
        "tokens": col("action.motion_token"),
        "state": state,
        "wbc": wbc,
        "body_state": body_state,
        "body_wbc": body_wbc,
        "hands_state": hands_state,
        "hands_wbc": hands_wbc,
        "gravity": col("observation.projected_gravity"),
        "root_quat": col("observation.root_orientation"),
    }


# --------------------------------------------------------------------------- model
class _JointsOnlyMap:
    """Name-based joint map for bare robot MJCFs (no actuators, no hand joints)."""

    def __init__(self, model):
        import mujoco as _mj

        def jid(name):
            for cand in (name + "_joint", name):
                i = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, cand)
                if i >= 0:
                    return i
            raise KeyError(name)

        self.body_q = [int(model.jnt_qposadr[jid(n)]) for n in G.BODY_JOINT_NAMES]
        self.body_v = [int(model.jnt_dofadr[jid(n)]) for n in G.BODY_JOINT_NAMES]
        self.body_a: list[int] = []
        self.hand_q: list[int] = []
        self.hand_v: list[int] = []
        self.hand_a: list[int] = []

    def q(self, data):
        return data.qpos[self.body_q]

    def dq(self, data):
        return data.qvel[self.body_v]

    def hands(self, data):
        return np.zeros(14)

    def set_q(self, data, values):
        data.qpos[self.body_q] = values

    def set_hands(self, data, values):
        return None

    def set_ctrl(self, data, values):
        return None

    def set_hand_ctrl(self, data, values):
        return None


class Plant:
    """Torque-motor G1 driven by the deploy PD law at 50 Hz (4 x 5 ms)."""

    def __init__(self, xml: Path, control_dt: float = 0.02, substeps: int = 4,
                 integrator: str = "implicitfast", gains: str = "ha"):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = control_dt / substeps
        if integrator == "implicitfast":
            self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.control_dt = control_dt
        self.substeps = substeps
        try:
            self.scene = G.SceneMap(self.model)
        except KeyError:
            self.scene = _JointsOnlyMap(self.model)
        self.nu_body = len(self.scene.body_a)
        #: bare robot MJCFs (the mjlab asset zoo) carry no actuators -> torque goes
        #: straight into ``qfrc_applied`` on the joint dof.
        self.use_qfrc = self.model.nu == 0
        if self.use_qfrc:
            self.body_dof = np.asarray(self.scene.body_v, dtype=np.int64)
        # joint limits in STANDARD order
        lo, hi = [], []
        for name in G.BODY_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name + "_joint")
            lo.append(self.model.jnt_range[jid, 0])
            hi.append(self.model.jnt_range[jid, 1])
        self.lo = np.asarray(lo)
        self.hi = np.asarray(hi)
        if self.model.nu == 0:
            self.tau_lim_override = True
        # torque limits straight from the MJCF motors
        self.tau_lim = np.asarray([self.model.actuator_ctrlrange[i, 1] for i in self.scene.body_a])
        self.kp = G.KP.astype(np.float64).copy()
        self.kd = G.KD.astype(np.float64).copy()
        if gains == "ha":
            kp = np.zeros(29)
            kd = np.zeros(29)
            lim = np.zeros(29)
            for i, name in enumerate(G.BODY_JOINT_NAMES):
                effort, stiff, damp = HA_ACTUATORS[ha_joint_suffix(name)]
                kp[i] = stiff
                kd[i] = damp
                lim[i] = effort
            self.kp, self.kd, self.tau_lim = kp, kd, lim
        self.up_axis = 2
        self.pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    # -- state -------------------------------------------------------------
    def _lowest_geom_z(self) -> float:
        """Exact lowest mesh vertex (per geom local frame) over all robot geoms."""
        MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
        zmin = np.inf
        for gid in range(self.model.ngeom):
            if int(self.model.geom_bodyid[gid]) == 0:  # world / floor
                continue
            if int(self.model.geom_type[gid]) != MESH:
                continue
            did = int(self.model.geom_dataid[gid])
            adr = int(self.model.mesh_vertadr[did])
            num = int(self.model.mesh_vertnum[did])
            verts = self.model.mesh_vert[adr:adr + num]
            rot = self.data.geom_xmat[gid].reshape(3, 3)
            pos = self.data.geom_xpos[gid]
            zs = verts @ rot[2] + pos[2]
            zmin = min(zmin, float(zs.min()))
        return zmin

    def root_height_for_contact(self, q_body: np.ndarray) -> float:
        """Root z that puts the lowest foot vertex on the floor (q is STANDARD order)."""
        self.data.qpos[:] = 0.0
        self.data.qpos[3] = 1.0
        self.data.qpos[2] = 1.0
        self.scene.set_q(self.data, q_body)
        mujoco.mj_forward(self.model, self.data)
        zmin = self._lowest_geom_z()
        if not np.isfinite(zmin):
            return 0.79
        return float(1.0 - zmin)

    def reset(self, q_body, hands, z: float | None = None, quat=(1, 0, 0, 0), xy=(0.0, 0.0),
              dq=None, base_lin_vel=None, base_ang_vel=None):
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[0:2] = xy
        self.data.qpos[2] = self.root_height_for_contact(q_body) if z is None else z
        self.data.qpos[3:7] = quat
        self.scene.set_q(self.data, q_body)
        self.scene.set_hands(self.data, hands)
        if base_lin_vel is not None:                        # world frame
            self.data.qvel[0:3] = np.asarray(base_lin_vel, float)
        if base_ang_vel is not None:                        # free joint: body frame omega
            self.data.qvel[3:6] = np.asarray(base_ang_vel, float)
        if dq is not None:
            self.data.qvel[self.scene.body_v] = np.asarray(dq, float)
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def q(self):
        return self.scene.q(self.data).copy()

    def dq(self):
        return self.scene.dq(self.data).copy()

    def quat(self):
        return self.data.qpos[3:7].copy()

    def root(self):
        return self.data.qpos[0:3].copy()

    def up(self):
        rot = G.quat_to_rotmat(self.data.qpos[3:7])
        return float(rot[2, 2])

    def tilt_deg(self):
        return float(np.degrees(np.arccos(np.clip(self.up(), -1, 1))))

    # -- control -----------------------------------------------------------
    def apply_target(self, q_target, hands_target, tau_clip=True):
        """Hold ``q_target`` for one control tick, recomputing PD every physics step.

        The real controller closes its PD loop at the physics rate (500 Hz), not at
        the 50 Hz policy rate; holding one torque for 20 ms is measurably unstable
        and made step-0 transients look like plant failure.
        """
        q_target = np.clip(q_target, self.lo, self.hi)
        hands_target = np.clip(hands_target, -2.4, 2.4)
        tau = np.zeros(self.nu_body)
        for _ in range(self.substeps):
            q = self.q()
            dq = self.dq()
            tau = self.kp * (q_target - q) - self.kd * dq
            if tau_clip:
                tau = np.clip(tau, -self.tau_lim, self.tau_lim)
            if self.use_qfrc:
                self.data.qfrc_applied[self.body_dof] = tau
            else:
                self.scene.set_ctrl(self.data, tau)
                self.scene.set_hand_ctrl(self.data, hands_target)
            mujoco.mj_step(self.model, self.data)
        return tau


# --------------------------------------------------------------------------- init
def joint_velocity(ep: dict) -> np.ndarray:
    """Central-difference joint velocity (STANDARD order) — zero velocity is wrong for a walker."""
    S = ep["body_state"]
    dq = np.zeros_like(S)
    dq[1:-1] = (S[2:] - S[:-2]) / 0.04
    dq[0] = (S[1] - S[0]) / 0.02
    dq[-1] = (S[-1] - S[-2]) / 0.02
    return dq


def foot_lock_base_velocity(plant: "Plant", ep: dict, frame: int) -> np.ndarray:
    """Estimate world base velocity from the stance foot being static (FK with root at origin)."""
    import mujoco as _mj

    m, d = plant.model, plant.data
    lo = _mj.mj_name2id(m, _mj.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    ro = _mj.mj_name2id(m, _mj.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

    def fk(q):
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        plant.scene.set_q(d, q)
        _mj.mj_forward(m, d)
        return d.xpos[lo].copy(), d.xpos[ro].copy()

    f1 = min(frame + 1, ep["body_state"].shape[0] - 1)
    l0, r0 = fk(ep["body_state"][frame])
    l1, r1 = fk(ep["body_state"][f1])
    delta = (l1 - l0) if l0[2] <= r0[2] else (r1 - r0)
    return -delta / 0.02


def frame_init_state(plant: "Plant", ep: dict, frame: int) -> dict:
    """Everything a faithful replay needs at ``frame``: quat from the recording, qd, base velocity."""
    dq_all = joint_velocity(ep)
    return {
        "quat": ep["root_quat"][frame].copy(),
        "dq": dq_all[frame].copy(),
        "base_lin_vel": foot_lock_base_velocity(plant, ep, frame),
        "base_ang_vel": np.zeros(3),
    }


# --------------------------------------------------------------------------- loops
def run_target(plant: Plant, ep: dict, steps: int, hands_source: str = "wbc", init=None,
               start: int = 0):
    n = min(steps, ep["state"].shape[0] - 1 - start)
    init = init or {}
    plant.reset(ep["body_state"][start], ep["hands_state"][start],
                quat=init.get("quat", (1, 0, 0, 0)),
                dq=init.get("dq"), base_lin_vel=init.get("base_lin_vel"),
                base_ang_vel=init.get("base_ang_vel"))
    rows = []
    for t in range(n):
        t += start
        hands = ep["hands_wbc"][t] if hands_source == "wbc" else ep["hands_state"][t]
        tau = plant.apply_target(ep["body_wbc"][t], hands)
        q = plant.q()
        rows.append(
            {
                "step": t,
                "q": q,
                "dq": plant.dq(),
                "quat": plant.quat(),
                "root": plant.root(),
                "up": plant.up(),
                "tau": tau,
                "mae_state": float(np.abs(q - ep["body_state"][t + 1]).mean()),
                "mae_target": float(np.abs(q - ep["body_wbc"][t]).mean()),
                "z": float(plant.root()[2]),
            }
        )
        if plant.up() < 0.3:
            break
    return rows, n


def run_token(plant: Plant, ep: dict, steps: int, decoder: Decoder, hands_source: str = "wbc",
              init=None):
    n = min(steps, ep["state"].shape[0] - 1)
    q0 = ep["body_state"][0]
    init = init or {}
    plant.reset(q0, ep["hands_state"][0], quat=init.get("quat", (1, 0, 0, 0)),
                dq=init.get("dq"), base_lin_vel=init.get("base_lin_vel"),
                base_ang_vel=init.get("base_ang_vel"))
    hist = []
    first = {
        "base_ang_vel": np.zeros(3),
        "q": (q0 - G.DEFAULT_ANGLES)[G.MUJOCO_TO_ISAACLAB],
        "dq": np.zeros(29),
        "last_action": np.zeros(29),
        "gravity": G.projected_gravity(plant.quat()),
    }
    for _ in range(HISTORY):
        hist.append({k: v.copy() for k, v in first.items()})
    rows = []
    for t in range(n):
        parts = [np.asarray(ep["tokens"][t], np.float64).reshape(-1)]
        for name in SIGNALS:
            parts.append(np.concatenate([f[name] for f in hist]))
        obs = np.concatenate(parts)
        action = decoder(obs)
        q_target = G.DEFAULT_ANGLES + action[G.ISAACLAB_TO_MUJOCO] * G.ACTION_SCALE
        hands = ep["hands_wbc"][t] if hands_source == "wbc" else ep["hands_state"][t]
        # angular velocity in body frame for the history
        qvel = plant.data.qvel
        w = plant.data.qvel[0:3].copy() if False else None
        rot = G.quat_to_rotmat(plant.quat())
        ang_vel = rot.T @ plant.data.qvel[0:3]
        tau = plant.apply_target(q_target, hands)
        q = plant.q()
        hist.append(
            {
                "base_ang_vel": ang_vel,
                "q": (q - G.DEFAULT_ANGLES)[G.MUJOCO_TO_ISAACLAB],
                "dq": plant.dq()[G.MUJOCO_TO_ISAACLAB],
                "last_action": action,
                "gravity": G.projected_gravity(plant.quat()),
            }
        )
        hist = hist[-HISTORY:]
        rows.append(
            {
                "step": t,
                "q": q,
                "dq": plant.dq(),
                "quat": plant.quat(),
                "root": plant.root(),
                "up": plant.up(),
                "tau": tau,
                "action": action,
                "q_target": q_target,
                "mae_state": float(np.abs(q - ep["body_state"][t + 1]).mean()),
                "mae_target": float(np.abs(q - ep["body_wbc"][t]).mean()),
                "mae_dec_vs_wbc": float(np.abs(q_target - ep["body_wbc"][t]).mean()),
                "z": float(plant.root()[2]),
            }
        )
        if plant.up() < 0.3:
            break
    return rows, n


def summarize(rows, n, key_state="mae_state", key_target="mae_target") -> dict:
    if not rows:
        return {"steps": 0}
    step = np.array([r["step"] for r in rows])
    mae_s = np.array([r[key_state] for r in rows])
    mae_t = np.array([r[key_target] for r in rows])
    up = np.array([r["up"] for r in rows])
    z = np.array([r["z"] for r in rows])
    root = np.stack([r["root"] for r in rows])
    drift = np.linalg.norm(root[:, :2] - root[0, :2], axis=1)
    out = {
        "steps_done": int(len(rows)),
        "steps_planned": int(n),
        "fell": bool(up[-1] < 0.3),
        "fall_step": int(step[-1]) if up[-1] < 0.3 else -1,
        "mae_state_mean": float(mae_s.mean()),
        "mae_state_p90": float(np.percentile(mae_s, 90)),
        "mae_target_mean": float(mae_t.mean()),
        "mae_target_p90": float(np.percentile(mae_t, 90)),
        "up_min": float(up.min()),
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "xy_drift_end": float(drift[-1]),
        "mae_state_block": [float(mae_s[i:i + 50].mean()) for i in range(0, len(mae_s), 50)][:12],
        "mae_target_block": [float(mae_t[i:i + 50].mean()) for i in range(0, len(mae_t), 50)][:12],
        "up_block": [float(up[i:i + 50].min()) for i in range(0, len(up), 50)][:12],
    }
    return out


def save(rows, out: Path, ep: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k in rows[0] if k != "step"] if rows else []
    arr = {k: np.stack([r[k] for r in rows]) for k in keys}
    np.savez_compressed(out, step=np.array([r["step"] for r in rows]), **arr)


def run_segments(plant: Plant, ep: dict, starts: list[int], seg_len: int, decoder: Decoder | None,
                 mode: str, hands_source: str = "wbc", init_faithful: bool = False) -> list[dict]:
    """Short-segment replay: reset to the reference every ``seg_len`` control ticks.

    This is the shape a data collector would use: the reference state is restored at
    each segment start, so global drift never accumulates inside the teacher-forced
    rollout.  Returns one row per segment with the errors *inside* the segment.
    """
    T = ep["state"].shape[0]
    out = []
    for f0 in starts:
        f1 = min(f0 + seg_len, T - 1)
        if f1 - f0 < 2:
            continue
        init = frame_init_state(plant, ep, f0) if init_faithful else {}
        plant.reset(ep["body_state"][f0], ep["hands_state"][f0],
                    quat=init.get("quat", (1, 0, 0, 0)), dq=init.get("dq"),
                    base_lin_vel=init.get("base_lin_vel"), base_ang_vel=init.get("base_ang_vel"))
        hist = None
        if mode == "token":
            q0 = ep["body_state"][f0]
            first = {
                "base_ang_vel": np.zeros(3),
                "q": (q0 - G.DEFAULT_ANGLES)[G.MUJOCO_TO_ISAACLAB],
                "dq": np.zeros(29),
                "last_action": np.zeros(29),
                "gravity": G.projected_gravity(plant.quat()),
            }
            hist = [{k: v.copy() for k, v in first.items()} for _ in range(HISTORY)]
        errs_s, errs_t, ups, zs, roots = [], [], [], [], []
        target_err = []
        fell = False
        for t in range(f0, f1):
            if mode == "target":
                q_target = ep["body_wbc"][t]
            else:
                parts = [np.asarray(ep["tokens"][t], np.float64).reshape(-1)]
                for name in SIGNALS:
                    parts.append(np.concatenate([f[name] for f in hist]))
                action = decoder(np.concatenate(parts))
                q_target = G.DEFAULT_ANGLES + action[G.ISAACLAB_TO_MUJOCO] * G.ACTION_SCALE
                rot = G.quat_to_rotmat(plant.quat())
                hist.append(
                    {
                        "base_ang_vel": rot.T @ plant.data.qvel[0:3],
                        "q": (plant.q() - G.DEFAULT_ANGLES)[G.MUJOCO_TO_ISAACLAB],
                        "dq": plant.dq()[G.MUJOCO_TO_ISAACLAB],
                        "last_action": action,
                        "gravity": G.projected_gravity(plant.quat()),
                    }
                )
                hist = hist[-HISTORY:]
                target_err.append(float(np.abs(q_target - ep["body_wbc"][t]).mean()))
            hands = ep["hands_wbc"][t] if hands_source == "wbc" else ep["hands_state"][t]
            plant.apply_target(q_target, hands)
            errs_s.append(float(np.abs(plant.q() - ep["body_state"][t + 1]).mean()))
            errs_t.append(float(np.abs(plant.q() - ep["body_wbc"][t]).mean()))
            ups.append(plant.up())
            zs.append(float(plant.root()[2]))
            roots.append(plant.root()[:2].copy())
            if plant.up() < 0.5:
                fell = True
                break
        roots = np.stack(roots)
        tilt = float(np.degrees(np.arccos(np.clip(ups[-1], -1, 1))))
        rec_tilt = float(np.degrees(np.arccos(np.clip(ep["root_quat"][f1, :][0] ** 2
                                                   - np.sum(ep["root_quat"][f1, :][1:] ** 2), -1, 1))))
        out.append(
            {
                "start": int(f0),
                "steps": len(errs_s),
                "fell": fell,
                "mae_state": float(np.mean(errs_s)),
                "mae_state_max": float(np.max(errs_s)),
                "mae_target": float(np.mean(errs_t)),
                "mae_dec_vs_wbc": float(np.mean(target_err)) if target_err else 0.0,
                "xy_drift": float(np.linalg.norm(roots[-1] - roots[0])),
                "z_drop": float(roots[0][0] * 0 + (zs[0] - zs[-1])),
                "tilt_deg": tilt,
                "rec_tilt_deg": rec_tilt,
                "up_min": float(np.min(ups)),
            }
        )
    return out


def summarize_segments(rows: list[dict]) -> dict:
    if not rows:
        return {"segments": 0}
    def stat(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return {"mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
                "p90": float(np.percentile(v, 90)), "max": float(v.max())}
    return {
        "segments": len(rows),
        "fell": int(sum(r["fell"] for r in rows)),
        "steps": int(rows[0]["steps"]),
        "mae_state": stat("mae_state"),
        "mae_target": stat("mae_target"),
        "mae_dec_vs_wbc": stat("mae_dec_vs_wbc"),
        "xy_drift": stat("xy_drift"),
        "z_drop": stat("z_drop"),
        "tilt_deg": stat("tilt_deg"),
        "up_min": stat("up_min"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="target", choices=("target", "token", "recorded_obs"))
    ap.add_argument("--set-index", type=int, default=0, help="which of the 9 sub-datasets")
    ap.add_argument("--episodes", default="0", help="comma separated episode indices")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--xml", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    ap.add_argument("--hands", default="wbc", choices=("wbc", "state"))
    ap.add_argument("--gains", default="ha", choices=("ha", "deploy"))
    ap.add_argument("--init", default="rest", choices=("rest", "recorded"),
                    help="初值：rest=单位四元数/零速；recorded=录制 root 姿态+关节速度+脚锁定基座速度")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/mujoco_replay/replay.npz")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--list", action="store_true", help="list sub-datasets and episode counts")
    ap.add_argument("--segments", type=int, default=0, help="N start frames for segmented replay")
    ap.add_argument("--segment-len", type=int, default=20)
    args = ap.parse_args()

    if args.list:
        for i, (task, d) in enumerate(dataset_dirs()):
            n = len(list((d / "data" / "chunk-000").glob("*.parquet")))
            print(f"[{i}] {task:45s} {n:4d} episodes")
        return 0

    decoder = Decoder(str(args.decoder)) if args.mode in ("token", "recorded_obs") else None
    plant = Plant(args.xml, gains=args.gains)
    report = {}
    for epi in [int(x) for x in args.episodes.split(",") if x.strip()]:
        ep = load_episode(args.set_index, epi)
        tag = f"{ep['task'][0]}/ep{epi}"
        t0 = time.time()
        if args.segments:
            T = ep["state"].shape[0]
            stride = max(1, (T - args.segment_len - 2) // args.segments)
            starts = list(range(0, T - args.segment_len - 2, stride))[: args.segments]
            seg_rows = run_segments(plant, ep, starts, args.segment_len, decoder, args.mode,
                                    args.hands, init_faithful=args.init == "recorded")
            summary = summarize_segments(seg_rows)
            summary["mode"] = args.mode
            summary["segment_len"] = args.segment_len
            summary["episode"] = tag
            report[tag] = {"summary": summary, "rows": seg_rows}
            print(f"[{args.mode}/seg{args.segment_len}] {tag}: {json.dumps(summary, ensure_ascii=False)}")
            continue
        init = frame_init_state(plant, ep, 0) if args.init == "recorded" else {}
        if args.mode == "target":
            rows, n = run_target(plant, ep, args.steps, args.hands, init=init)
        elif args.mode == "token":
            rows, n = run_token(plant, ep, args.steps, decoder, args.hands, init=init)
        else:
            raise SystemExit("recorded_obs arm is not wired yet")
        dt = time.time() - t0
        summary = summarize(rows, n)
        summary["wall_s"] = round(dt, 2)
        summary["steps_per_s"] = round(len(rows) / max(dt, 1e-6), 1)
        summary["episode"] = tag
        summary["mode"] = args.mode
        report[tag] = summary
        print(f"[{args.mode}] {tag}: {json.dumps(summary, ensure_ascii=False)}")
        if len(report) == 1 and args.report is None:
            save(rows, args.out, ep)
            print(f"[saved] {args.out}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False))
        print(f"[report] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
