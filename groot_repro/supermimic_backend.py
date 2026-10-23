"""SuperMimic policy as a deployment decoder: token(64) + 10-frame history -> q_target.

This is the counterpart of ``sonic_loop.SonicController`` (which uses SONIC's
``model_decoder.onnx``).  Same interface::

    ctrl = SuperMimicController("logs/e9_bones_seed_token/ckpt_latest.pt")
    q_target = ctrl.step(token, q_mujoco, dq_mujoco, quat_wxyz, body_ang_vel)

Internals:

* SuperMimic policy input is ``obs_hist`` ``(1, 930)`` in **term-major** order
  ``[joint_pos(29), joint_vel(29), base_ang_vel(3), projected_gravity(3), actions(29)] × 10``;
* the action is a normalized joint offset: ``q_target = q0 + action_scale * action``,
  with SuperMimic's own ``q0`` / ``action_scale`` (``diff_vla_wbc.compat``), *not*
  SONIC's deploy numbers;
* the token comes from the VLA (64-D), exactly like the SONIC decoder's first 64
  inputs.

NOTE (validation status): the obs construction here mirrors the training code
(``supermimic.world_model.heads.analytic_observation``); a numeric parity check
against a logged training observation is still TODO.  Use this for A/B runs, not
for blind hardware deployment.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import g1_sim as G  # noqa: E402

# Import the real SuperMimic package *before* diff_vla_wbc.compat: the compat shim
# would otherwise replace ``supermimic.envs`` with a lightweight namespace module.
from supermimic.eval.benchmark import load_policy_from_checkpoint  # noqa: E402

from diff_vla_wbc.compat import (  # noqa: E402
    G1_ACTION_SCALE,
    G1_KD,
    G1_KP,
    g1_default_joint_pos,
)
from diff_vla_wbc.sonic_encoder import mjcf_to_isaaclab_perm  # noqa: E402

HISTORY = 10
OBS_TERM_DIMS = (29, 29, 3, 3, 29)  # joint_pos, joint_vel, base_ang_vel, gravity, actions


class SuperMimicController:
    """Token + proprio history -> joint targets, using the trained SuperMimic policy."""

    def __init__(self, ckpt: str | Path, device: str = "cuda:0", init_q: np.ndarray | None = None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.policy, self.cfg = load_policy_from_checkpoint(ckpt, str(self.device))
        if int(getattr(self.policy, "token_dim", 0)) <= 0:
            raise ValueError("checkpoint is not token-conditioned (policy.token_dim == 0)")

        # SuperMimic joint conventions (MJCF order -> IsaacLab order used by mjlab state).
        perm = mjcf_to_isaaclab_perm()
        self.q0_isaac = g1_default_joint_pos().numpy()[perm].astype(np.float64)
        self.action_scale = G1_ACTION_SCALE.numpy().astype(np.float64)[perm]
        self.kp = G1_KP.numpy().astype(np.float64)[perm]
        self.kd = G1_KD.numpy().astype(np.float64)[perm]
        self.mujoco_to_isaac = G.MUJOCO_TO_ISAACLAB  # index arrays from g1_sim
        self.isaac_to_mujoco = G.ISAACLAB_TO_MUJOCO

        q = self.q0_isaac if init_q is None else np.asarray(init_q, float)
        first = {
            "joint_pos": np.zeros(29),
            "joint_vel": np.zeros(29),
            "base_ang_vel": np.zeros(3),
            "gravity": np.array([0.0, 0.0, -1.0]),
            "actions": np.zeros(29),
        }
        self.hist: deque[dict[str, np.ndarray]] = deque(maxlen=HISTORY)
        for _ in range(HISTORY):
            self.hist.append({k: v.copy() for k, v in first.items()})
        self.last_action = np.zeros(29)

    # ------------------------------------------------------------------ obs
    def build_obs(self) -> np.ndarray:
        """Term-major flattened history, shape ``(1, 930)``."""
        parts = []
        for term in ("joint_pos", "joint_vel", "base_ang_vel", "gravity", "actions"):
            parts.append(np.stack([f[term] for f in self.hist], axis=0).reshape(-1))
        obs = np.concatenate(parts)
        assert obs.shape == (sum(OBS_TERM_DIMS) * HISTORY,), obs.shape
        return obs[None]

    # ----------------------------------------------------------------- step
    @torch.no_grad()
    def step(
        self,
        token: np.ndarray,
        q_mujoco: np.ndarray,
        dq_mujoco: np.ndarray,
        quat_wxyz: np.ndarray,
        body_ang_vel: np.ndarray,
    ) -> np.ndarray:
        """One 50 Hz control tick. Returns ``q_target`` in MuJoCo joint order."""
        obs = torch.as_tensor(self.build_obs(), dtype=torch.float32, device=self.device)
        tok = torch.as_tensor(np.asarray(token, np.float32).reshape(1, -1), device=self.device)
        action = self.policy(obs, tok, interface="token").cpu().numpy().reshape(-1)  # IsaacLab order

        q0_mujoco = self.q0_isaac[self.isaac_to_mujoco]
        scale_mujoco = self.action_scale[self.isaac_to_mujoco]
        q_target = q0_mujoco + action[self.isaac_to_mujoco] * scale_mujoco
        self.last_action = action
        self.hist.append(
            {
                "joint_pos": (np.asarray(q_mujoco, float) - q0_mujoco)[self.mujoco_to_isaac],
                "joint_vel": np.asarray(dq_mujoco, float)[self.mujoco_to_isaac],
                "base_ang_vel": np.asarray(body_ang_vel, float),
                "gravity": G.projected_gravity(quat_wxyz),
                "actions": action,
            }
        )
        return q_target

    def torque(self, q_target: np.ndarray, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        return self.kp[self.isaac_to_mujoco] * (q_target - q) - self.kd[self.isaac_to_mujoco] * dq
