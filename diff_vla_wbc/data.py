"""Motion data plumbing and batch providers for joint training.

Two paths are supported:

* ``npz`` clips in the upstream ``MotionLibrary`` format (LAFAN1-style, full
  body states).  ``prepare_motion_npz`` converts the SONIC-style per-body CSV
  clips shipped in this workspace into that format using the G1 MJCF and
  differentiable FK.
* packed BONES-SEED compact ``.pt`` libraries produced by the upstream
  ``scripts/data/build_library.py``.  Those can be loaded directly with
  ``load_motion_library(..., compact=True)``.

The batch provider is intentionally agnostic about where FM tokens come from:
pass a real ``Gr00tTokenSource`` for end-to-end training, or random/offline
tokens for gradient smoke tests.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from .compat import G1_JOINT_NAMES, ensure_supermimic_importable, g1_xml_path, load_g1_model, quat_apply_inverse
from .supermimic_backend import (
    ACTION_DIM,
    GOAL_DIM,
    HISTORY,
    STATE_DIM,
    SuperMimicBundle,
    default_xi,
)

ensure_supermimic_importable()


def _read_csv(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None]
    return arr


def _batch_fk(fk, qpos: torch.Tensor, qvel: torch.Tensor, batch: int = 1024):
    out = {"pos": [], "quat": [], "lin_vel": [], "ang_vel": []}
    with torch.no_grad():
        for start in range(0, qpos.shape[0], batch):
            r = fk(qpos[start : start + batch], qvel[start : start + batch])
            for key in out:
                out[key].append(r[key].cpu())
    return {k: torch.cat(v, 0).numpy().astype(np.float32) for k, v in out.items()}


def motion_npz_from_body_csv_dir(
    csv_dir: str | Path,
    output: str | Path,
    *,
    xml_path: str | Path | None = None,
    fps: float = 50.0,
    device: str = "cpu",
) -> Path:
    """Convert a SONIC ``joint_pos.csv`` + ``body_*.csv`` clip into a MotionLibrary npz.

    The per-body CSV order is not trusted.  Root pose + joint angles are fed
    through the G1 MJCF FK, which produces body states in the canonical MuJoCo
    body order.  Joint velocities are finite differences of the joint angles
    (same convention as mjlab's converter).
    """
    csv_dir = Path(csv_dir)
    out = Path(output)
    joint_q = _read_csv(csv_dir / "joint_pos.csv")
    body_pos = _read_csv(csv_dir / "body_pos.csv")
    body_quat = _read_csv(csv_dir / "body_quat.csv")
    body_lin = _read_csv(csv_dir / "body_lin_vel.csv")
    body_ang = _read_csv(csv_dir / "body_ang_vel.csv")
    if joint_q.shape[1] != len(G1_JOINT_NAMES):
        raise ValueError(f"{joint_q}: expected {len(G1_JOINT_NAMES)} joints, got {joint_q.shape[1]}")
    if not (joint_q.shape[0] == body_pos.shape[0] == body_quat.shape[0] == body_lin.shape[0] == body_ang.shape[0]):
        raise ValueError("CSV frame counts differ")
    # body_0 is the pelvis in the SONIC exports; use it for the free joint.
    root_pos = body_pos[:, 0:3]
    root_quat = body_quat[:, 0:4]
    root_quat = root_quat / np.linalg.norm(root_quat, axis=-1, keepdims=True)
    # Joint velocities: central differences (same convention as mjlab).
    joint_vel = np.gradient(joint_q, 1.0 / fps, axis=0)
    root_lin_vel = body_lin[:, 0:3]
    root_ang_w = body_ang[:, 0:3]
    q = torch.tensor(root_quat, dtype=torch.float32)
    root_ang_b = quat_apply_inverse(q, torch.tensor(root_ang_w, dtype=torch.float32)).numpy()

    qpos_np = np.concatenate([root_pos, root_quat, joint_q], axis=1).astype(np.float32)
    qvel_np = np.concatenate([root_lin_vel, root_ang_b, joint_vel], axis=1).astype(np.float32)
    if qpos_np.shape[1] != 36 or qvel_np.shape[1] != 35:
        raise ValueError(f"bad qpos/qvel shapes: {qpos_np.shape}, {qvel_np.shape}")

    model = load_g1_model(xml_path)
    from supermimic.kinematics.g1_fk import G1Kinematics

    fk = G1Kinematics(model, device=device)
    bodies = _batch_fk(
        fk,
        torch.tensor(qpos_np, dtype=torch.float32, device=device),
        torch.tensor(qvel_np, dtype=torch.float32, device=device),
    )
    # FK body 0 must be the pelvis and should agree with root_pos (within numerical noise).
    if fk.body_names[0] != "pelvis":
        raise ValueError(f"unexpected FK root body {fk.body_names[0]}")
    root_mae = float(np.abs(bodies["pos"][:, 0] - root_pos).mean())
    if root_mae > 5e-3:
        raise ValueError(
            f"root FK/trajectory mismatch (MAE {root_mae:.4f} m); joint or quaternion convention is wrong"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        fps=np.float32(fps),
        joint_pos=joint_q.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=bodies["pos"],
        body_quat_w=bodies["quat"],
        body_lin_vel_w=bodies["lin_vel"],
        body_ang_vel_w=bodies["ang_vel"],
        joint_names=np.array(G1_JOINT_NAMES),
        body_names=np.array(fk.body_names),
        source=str(csv_dir),
        input_fps=np.float32(fps),
    )
    return out


def motion_npz_from_lafan1_csv(
    csv_path: str | Path,
    output: str | Path,
    *,
    xml_path: str | Path | None = None,
    input_fps: float = 30.0,
    output_fps: float = 50.0,
    device: str = "cpu",
) -> Path:
    """Convert a 36-column LAFAN1/G1 retarget CSV to the motion-library npz.

    CSV layout: root xyz (m), root quaternion **xyzw**, 29 joint angles (rad).
    Resampling/velocity conventions follow the upstream ``motion_convert``
    module (linear interpolation, finite-difference velocities, SO(3) mean for
    angular velocity).
    """
    raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 36:
        raise ValueError(f"{csv_path}: expected (N,36), got {raw.shape}")
    from supermimic.data.motion_convert import resample_motion, quat_xyzw_to_wxyz

    motion = resample_motion(raw, input_fps=input_fps, output_fps=output_fps)
    joint_q = motion.joint_pos.astype(np.float32)
    qpos_np = np.concatenate([motion.root_pos, motion.root_quat, motion.joint_pos], axis=1).astype(np.float32)
    root_ang_w = motion.root_ang_vel_w
    q = torch.tensor(motion.root_quat, dtype=torch.float32)
    root_ang_b = quat_apply_inverse(q, torch.tensor(root_ang_w, dtype=torch.float32)).numpy()
    qvel_np = np.concatenate(
        [motion.root_lin_vel_w, root_ang_b, motion.joint_vel], axis=1
    ).astype(np.float32)
    model = load_g1_model(xml_path)
    from supermimic.kinematics.g1_fk import G1Kinematics

    fk = G1Kinematics(model, device=device)
    bodies = _batch_fk(
        fk,
        torch.tensor(qpos_np, device=device),
        torch.tensor(qvel_np, device=device),
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        fps=np.float32(output_fps),
        joint_pos=joint_q,
        joint_vel=motion.joint_vel.astype(np.float32),
        body_pos_w=bodies["pos"],
        body_quat_w=bodies["quat"],
        body_lin_vel_w=bodies["lin_vel"],
        body_ang_vel_w=bodies["ang_vel"],
        joint_names=np.array(G1_JOINT_NAMES),
        body_names=np.array(fk.body_names),
        source=str(csv_path),
        input_fps=np.float32(input_fps),
    )
    return out


@dataclass
class MotionBatch:
    fm_tokens: torch.Tensor
    teacher_goals: torch.Tensor
    teacher_states: torch.Tensor
    start_state: torch.Tensor
    obs_hist: torch.Tensor
    history: torch.Tensor
    prev_action: torch.Tensor
    xi: torch.Tensor
    frames: torch.Tensor
    references: list
    teacher_goal_sequence: torch.Tensor | None = None
    model_inputs: dict | None = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class MotionBatchProvider:
    """Sample paired (start state, teacher goal/reference) motion windows."""

    def __init__(
        self,
        bundle: SuperMimicBundle,
        library,
        *,
        batch_size: int = 2,
        horizon: int = 8,
        token_source: Callable[[dict], torch.Tensor] | None = None,
        token_dim: int = 64,
        token_horizon: int = 40,
        seed: int | None = None,
    ):
        self.bundle = bundle
        self.library = library
        self.batch_size = int(batch_size)
        self.horizon = int(horizon)
        self.token_source = token_source
        self.token_dim = token_dim
        self.token_horizon = token_horizon
        self.rng = torch.Generator(device="cpu")
        if seed is not None:
            self.rng.manual_seed(seed)

    def sample(self) -> MotionBatch:
        max_start = self.library.num_frames - self.horizon - 2
        if max_start <= HISTORY:
            raise ValueError("motion library is shorter than the observation/rollout horizon")
        frames = torch.randint(
            HISTORY, max_start, (self.batch_size,), generator=self.rng, dtype=torch.long
        ).to(self.bundle.device)
        start = self.bundle.reference_start(self.library, frames)
        refs = self.bundle.teacher_references(self.library, frames, self.horizon)
        teacher_goals = self.bundle.teacher_goal(self.library, frames)
        teacher_goal_sequence = torch.stack(
            [
                self.bundle.teacher_goal(self.library, frames + t)
                for t in range(self.horizon)
            ],
            dim=1,
        )
        teacher_states = self.bundle.teacher_states(self.library, frames, self.horizon)
        meta = {
            "frames": frames,
            "start_state": start["state"],
            "history": start["observations"],
            "obs_hist": start["obs_hist"],
            "prev_action": start["prev_action"],
            "xi": start["xi"],
            "teacher_goals": teacher_goals,
            "references": refs,
        }
        if self.token_source is not None:
            tokens = self.token_source(meta)
        else:
            tokens = torch.randn(
                self.batch_size,
                self.token_horizon,
                self.token_dim,
                device=self.bundle.device,
            )
        if tokens.ndim != 3 or tokens.shape[0] != self.batch_size:
            raise ValueError(f"token source returned {tuple(tokens.shape)}")
        return MotionBatch(
            fm_tokens=tokens,
            teacher_goals=teacher_goals,
            teacher_states=teacher_states,
            start_state=start["state"],
            obs_hist=start["obs_hist"],
            history=start["observations"],
            prev_action=start["prev_action"],
            xi=start["xi"],
            frames=frames,
            references=refs,
            teacher_goal_sequence=teacher_goal_sequence,
        )


def _reference_state_at(bundle: SuperMimicBundle, library, frames: torch.Tensor) -> torch.Tensor:
    r = library.ref(frames)
    return torch.cat(
        (r.root_pos_w, r.root_quat_w, r.joint_pos, r.root_lin_vel_w, r.root_ang_vel_w, r.joint_vel),
        dim=-1,
    )


def sample_reference_windows(
    bundle: SuperMimicBundle,
    library,
    *,
    batch_size: int = 8,
    window: int = 8,
    seed: int | None = None,
):
    """Reference windows for world-model pretraining (no simulator required)."""
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    max_start = library.num_frames - window - 2
    frames = torch.randint(2, max_start, (batch_size,), generator=g, dtype=torch.long).to(bundle.device)
    offs = torch.arange(window + 1, device=bundle.device)
    idx = frames[:, None] + offs[None, :]
    states = _reference_state_at(bundle, library, idx.reshape(-1)).reshape(batch_size, window + 1, STATE_DIM)
    actions = (states[:, 1:, 7:36] - bundle.default_joint_pos) / bundle.action_scale
    xi = default_xi(batch_size, bundle.device)
    done = torch.zeros(batch_size, window, device=bundle.device)
    root = library.ref(idx.reshape(-1))
    from supermimic.world_model.heads import ReferenceFrame

    root_pos = root.root_pos_w.reshape(batch_size, window + 1, 3)
    root_quat = root.root_quat_w.reshape(batch_size, window + 1, 4)
    refs = [
        ReferenceFrame(None, None, None, None, root_pos[:, t], root_quat[:, t])
        for t in range(window)
    ]
    return {
        "states": states.permute(1, 0, 2).contiguous(),  # (k+1,B,71)
        "actions": actions.permute(1, 0, 2).contiguous(),  # (k,B,29)
        "xi": xi,
        "done": done.permute(1, 0).contiguous(),
        "refs": refs,
    }


__all__ = [
    "MotionBatch",
    "MotionBatchProvider",
    "motion_npz_from_body_csv_dir",
    "motion_npz_from_lafan1_csv",
    "sample_reference_windows",
]


class CachedTokenSource:
    """Use offline-cached GR00T tokens for a motion library.

    ``tokens`` is indexed by the *global* motion-library frame, e.g. the result
    of running ``sample_model_with_grad`` on the RGB/language clip paired with
    each frame.  This is the recommended way to train the adapter/WBC without
    keeping the 2B VLM in the loop.
    """

    def __init__(self, tokens: torch.Tensor):
        if tokens.ndim != 3:
            raise ValueError("cached tokens must be (num_frames, token_horizon, token_dim)")
        self.tokens = tokens

    def __call__(self, meta: dict) -> torch.Tensor:
        frames = meta["frames"].to(self.tokens.device)
        return self.tokens[frames]
