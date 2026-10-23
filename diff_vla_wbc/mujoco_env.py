"""Minimal MuJoCo G1 closed-loop evaluator for the differentiable policy.

This is intentionally small and dependency-light (CPU MuJoCo, no mjlab/Warp).
It is used to sanity-check that the learned WBC actually tracks a reference
motion in the same 29-DoF G1 model that defines the FK/world-model state.

Control convention matches the upstream environment:

* policy output          : 29 normalized joint offsets
* joint position target  : ``q_default + action_scale * action``
* actuator command       : ``tau = kp * (q_target - q) - kd * dq``
* control rate           : 50 Hz (4 x 5 ms physics substeps)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .compat import ensure_supermimic_importable
from .supermimic_backend import (
    ACTION_DIM,
    GOAL_FRAMES,
    HISTORY,
    STATE_DIM,
    SuperMimicBundle,
    default_xi,
)

ensure_supermimic_importable()


@dataclass
class EvalResult:
    clip: str
    frames: int
    steps: int
    norm_return: float
    mean_reward: float
    failed: bool
    fail_step: int
    joint_rmse: float
    root_rmse: float
    survival: float
    trajectory: dict[str, np.ndarray] = field(default_factory=dict)


class G1MujocoEnv:
    def __init__(self, bundle: SuperMimicBundle, *, control_dt: float = 0.02, substeps: int = 4):
        import mujoco

        self.mujoco = mujoco
        self.bundle = bundle
        self.model = mujoco.MjModel.from_xml_path(str(bundle.assets.xml_path))
        # The deploy MJCF has zero armature; the mjlab training model adds the
        # reflected motor inertia. Without it the 10 Hz PD gains are unstable
        # at a 5 ms step. Add it and use the implicit integrator.
        self.model.dof_armature[6:35] = bundle.assets.armature.detach().cpu().numpy()
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.timestep = control_dt / substeps
        self.data = mujoco.MjData(self.model)
        self.control_dt = control_dt
        self.substeps = substeps
        self.q0 = bundle.assets.default_joint_pos.detach().cpu().numpy().astype(np.float64)
        self.action_scale = bundle.assets.action_scale.detach().cpu().numpy().astype(np.float64)
        self.kp = bundle.assets.kp.detach().cpu().numpy().astype(np.float64)
        self.kd = bundle.assets.kd.detach().cpu().numpy().astype(np.float64)
        self.joint_lo = self.model.jnt_range[1:30, 0]
        self.joint_hi = self.model.jnt_range[1:30, 1]
        self.effort = np.full(ACTION_DIM, 200.0)
        if getattr(self.model, "actuator_forcerange", None) is not None:
            fr = self.model.actuator_forcerange
            finite = np.isfinite(fr).all(axis=1) & (np.abs(fr).max(axis=1) > 0)
            self.effort[finite] = np.maximum(np.abs(fr[finite, 0]), np.abs(fr[finite, 1]))
        self.ctrl = np.zeros(self.model.nu, dtype=np.float64)

    # ------------------------------------------------------------------ state

    def reset(self, state: np.ndarray | torch.Tensor | None = None) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        if state is not None:
            s = np.asarray(state.detach().cpu() if torch.is_tensor(state) else state, dtype=np.float64).reshape(-1)
            if s.shape[0] != STATE_DIM:
                raise ValueError(f"state must have {STATE_DIM} entries, got {s.shape}")
            self.data.qpos[:] = s[:36]
            self.data.qvel[:] = s[36:]
        self.data.time = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def state(self) -> torch.Tensor:
        s = np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])
        return torch.tensor(s, dtype=torch.float32, device=self.bundle.device).unsqueeze(0)

    def state_numpy(self) -> np.ndarray:
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    # ------------------------------------------------------------------ control

    def apply_action(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        a = np.asarray(action.detach().cpu() if torch.is_tensor(action) else action, dtype=np.float64).reshape(-1)
        if a.shape[0] != ACTION_DIM:
            raise ValueError(f"action must have {ACTION_DIM} entries, got {a.shape}")
        q = self.data.qpos[7:36].copy()
        dq = self.data.qvel[6:35].copy()
        target = np.clip(self.q0 + self.action_scale * a, self.joint_lo, self.joint_hi)
        tau = self.kp * (target - q) - self.kd * dq
        tau = np.clip(tau, -self.effort, self.effort)
        self.ctrl[:] = tau
        for _ in range(self.substeps):
            self.data.ctrl[:] = tau
            self.mujoco.mj_step(self.model, self.data)

    @torch.no_grad()
    def rollout(
        self,
        library,
        *,
        clip_index: int = 0,
        max_steps: int | None = None,
        terminated_ok: bool = True,
        mode: str = "policy",
    ) -> EvalResult:
        clip = library.clips[clip_index]
        start = int(clip.start)
        end = int(clip.end)
        steps = min(max_steps or (end - start - 1), end - start - 1)
        if steps <= 0:
            raise ValueError("clip is too short")
        start_state = library.ref_qpos_qvel(torch.tensor([start], device=library.device))
        qpos, qvel = start_state
        self.reset(torch.cat((qpos[0], qvel[0]), dim=-1))

        # History from the reference (same construction as training/eval batches).
        frames = torch.tensor([start], device=library.device)
        start_batch = self.bundle.reference_start(library, frames)
        history = start_batch["observations"].to(self.bundle.device)
        prev_action = start_batch["prev_action"]
        xi = default_xi(1, self.bundle.device)
        frame = start
        term_names = None
        term_sum = None
        rewards = []
        joint_err, root_err = [], []
        traj = {"qpos": [], "qvel": [], "actions": [], "rewards": []}
        failed = False
        fail_step = -1

        for t in range(steps):
            goal = library.goal_window(torch.tensor([frame], device=library.device), GOAL_FRAMES)
            if mode == "teacher":
                # Inverse-dynamics-style baseline: directly target the next
                # reference joint configuration through the same action scaling.
                next_frame = min(frame + 1, end - 1)
                target_q = library.ref(torch.tensor([next_frame], device=library.device)).joint_pos[0]
                action = ((target_q - self.bundle.default_joint_pos) / self.bundle.action_scale).unsqueeze(0)
            elif mode == "policy":
                obs_flat = self.bundle.flatten_history(history)
                action = self.bundle.act(obs_flat, goal)
            else:
                raise ValueError(f"unknown rollout mode {mode!r}")
            self.apply_action(action)
            state = self.state()
            frame = min(frame + 1, end - 1)
            ref = self.bundle.reference_frame(library, torch.tensor([frame], device=library.device))
            bodies = self.bundle.assets.heads.robot_bodies(state)
            terms = self.bundle.reward(state, action, prev_action, ref, bodies)
            if term_names is None:
                term_names = [k for k in terms if k != "total"]
                term_sum = {k: 0.0 for k in term_names}
            for k in term_names:
                term_sum[k] += float(terms[k].mean().detach())
            rewards.append(float(terms["total"].mean().detach()))
            # Tracking diagnostics.
            current_ref = library.ref(torch.tensor([frame], device=library.device))
            joint_err.append(float((state[0, 7:36] - current_ref.joint_pos[0]).abs().mean().detach()))
            root_err.append(float((state[0, 0:3] - current_ref.root_pos_w[0]).norm().detach()))
            clean = self.bundle.observation(state, action, xi)
            history = torch.cat((history[1:], clean.unsqueeze(0)), dim=0)
            prev_action = action
            traj["qpos"].append(self.data.qpos.copy())
            traj["qvel"].append(self.data.qvel.copy())
            traj["actions"].append(action.detach().cpu().numpy())
            traj["rewards"].append(rewards[-1])
            if terminated_ok:
                f = self.bundle.assets.heads.termination_features(state[..., 0:3], state[..., 3:7], ref)
                if bool((f[..., 0].abs() > 0.25).any() or (f[..., 1].abs() > 0.8).any()):
                    failed = True
                    fail_step = t
                    break

        max_step = self.bundle.assets.heads.reward_cfg.max_step_reward
        norm_return = float(np.sum(rewards) / (max_step * max(len(rewards), 1)))
        return EvalResult(
            clip=clip.name,
            frames=steps,
            steps=len(rewards),
            norm_return=norm_return,
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            failed=failed,
            fail_step=fail_step,
            joint_rmse=float(np.mean(joint_err)) if joint_err else float("nan"),
            root_rmse=float(np.mean(root_err)) if root_err else float("nan"),
            survival=float(len(rewards) / steps),
            trajectory={k: np.asarray(v) for k, v in traj.items()},
        )

    def render_frame(self, width: int = 640, height: int = 480):
        """Best-effort RGB frame; requires EGL/OSMesa and a working GL backend."""
        import mujoco

        renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            renderer.update_scene(self.data)
            return renderer.render()
        finally:
            renderer.close()


__all__ = ["EvalResult", "G1MujocoEnv"]
