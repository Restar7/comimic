"""Python port of the SONIC deployment control loop.

Replaces the C++ `g1_deploy_onnx_ref` binary with:
  token(64) + 10-frame history  ->  model_decoder.onnx  ->  action(29)
  q_target = default_angles + action[isaaclab_to_mujoco] * action_scale
  tau      = kp * (q_target - q) - kd * dq

Everything (PD gains, action scale, joint maps, default pose) is taken from
gear_sonic_deploy/.../policy_parameters.hpp so the numbers match the robot.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import onnxruntime as ort

import g1_sim as G

CONTROL_DT = 0.02  # 50 Hz, from g1_deploy_onnx_ref.cpp
HISTORY = 10
SIGNALS = ("base_ang_vel", "q", "dq", "last_action", "gravity")


class Decoder:
    def __init__(self, onnx_path: str):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def __call__(self, obs_994: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input: obs_994.reshape(1, 994).astype(np.float32)})[0].ravel()


class SonicController:
    """Maintains the 10-frame observation history and turns tokens into joint targets."""

    def __init__(self, onnx_path: str, init_q: np.ndarray | None = None):
        self.decoder = Decoder(onnx_path)
        q = G.DEFAULT_ANGLES if init_q is None else np.asarray(init_q, float)
        self.hist: deque[dict[str, np.ndarray]] = deque(maxlen=HISTORY)
        first = {
            "base_ang_vel": np.zeros(3),
            "q": q.copy(),
            "dq": np.zeros(29),
            "last_action": np.zeros(29),
            "gravity": np.array([0.0, 0.0, -1.0]),
        }
        for _ in range(HISTORY):
            self.hist.append({k: v.copy() for k, v in first.items()})

    def build_obs(self, token: np.ndarray) -> np.ndarray:
        parts = [np.asarray(token, np.float64).reshape(-1)]
        for name in SIGNALS:
            parts.append(np.concatenate([f[name] for f in self.hist]))
        obs = np.concatenate(parts)
        assert obs.shape == (994,), obs.shape
        return obs

    def step(self, token: np.ndarray, q_mujoco: np.ndarray, dq_mujoco: np.ndarray,
             quat_wxyz: np.ndarray, ang_vel: np.ndarray) -> np.ndarray:
        """One control tick. Returns q_target in MuJoCo joint order."""
        action = self.decoder(self.build_obs(token))
        q_target = G.DEFAULT_ANGLES + action[G.ISAACLAB_TO_MUJOCO] * G.ACTION_SCALE
        self.hist.append(
            {
                "base_ang_vel": np.asarray(ang_vel, float),
                "q": (np.asarray(q_mujoco, float) - G.DEFAULT_ANGLES)[G.MUJOCO_TO_ISAACLAB],
                "dq": np.asarray(dq_mujoco, float)[G.MUJOCO_TO_ISAACLAB],
                "last_action": action,
                "gravity": G.projected_gravity(quat_wxyz),
            }
        )
        return q_target


def pd_torque(q_target: np.ndarray, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    return G.KP * (q_target - q) - G.KD * dq
