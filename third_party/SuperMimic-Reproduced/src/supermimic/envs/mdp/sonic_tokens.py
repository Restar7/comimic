"""SONIC motion-token builder for the SuperMimic training loop (B mode).

The token conditions the policy instead of the 680-d goal: every imagination step
computes ``token = frozen_encoder(future reference, simulated robot root)``.  The
encoder weights are extracted from the deployed ``model_encoder.onnx`` (g1 branch)
and the token is **detached** — it is an input, not a learned representation.

This module deliberately lives in the upstream tree (it only needs torch + onnx)
so training does not depend on the surrounding workspace package.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

NUM_FUTURE = 10
TOKEN_DIM = 64
INPUT_DIM = 1762
#: ``obs[0]`` is dropped before slicing, so real offsets are the graph constants + 1.
COMMAND_OFFSET = 4
ANCHOR_OFFSET = 601

G1_MJCF_ORDER: tuple[str, ...] = (
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
G1_ISAACLAB_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
FSQ_OFFSET = 0.03223739564418793
FSQ_SCALE = 15.515501022338867
FSQ_BIAS = 0.5
FSQ_DIV = 16.0


def default_encoder_path() -> Path | None:
    env = os.environ.get("SONIC_ENCODER_ONNX")
    if env and Path(env).is_file():
        return Path(env)
    here = Path(__file__).resolve()
    repo = here.parents[4]  # third_party/SuperMimic-Reproduced
    candidates = [repo / "data/assets/model_encoder.onnx"]
    for base in here.parents[4:9]:
        candidates.append(base / "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx")
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


class G1TokenEncoder(nn.Module):
    """640 -> 64 MLP + FSQ, weights from the deployed ONNX (g1 branch)."""

    def __init__(self, linears: list[nn.Linear]):
        super().__init__()
        self.linears = nn.ModuleList(linears)
        self.register_buffer("perm", torch.as_tensor([G1_MJCF_ORDER.index(n) for n in G1_ISAACLAB_ORDER]))

    @classmethod
    def from_onnx(cls, path: str | Path) -> "G1TokenEncoder":
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(path))
        w = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        dims = ((2048, 640), (1024, 2048), (512, 1024), (512, 512), (64, 512))
        layers = []
        for k, (out_dim, in_dim) in enumerate(dims):
            lin = nn.Linear(in_dim, out_dim)
            key = f"module.encoders.g1.module.{2 * k}"
            with torch.no_grad():
                lin.weight.copy_(torch.from_numpy(w[f"{key}.weight"].copy()).float())
                lin.bias.copy_(torch.from_numpy(w[f"{key}.bias"].copy()).float())
            layers.append(lin)
        return cls(layers)

    def forward(self, command: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        x = torch.cat([command, anchor], dim=-1).reshape(command.shape[0], -1)
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1:
                x = torch.nn.functional.silu(x)
        return torch.round(torch.tanh(x + FSQ_OFFSET) * FSQ_SCALE - FSQ_BIAS) / FSQ_DIV


class SonicTokenBuilder:
    """``__call__(frame, ref, state) -> (N, 64)`` for the imagination rollout."""

    def __init__(self, lib, device: torch.device, num_future: int = NUM_FUTURE, encoder: nn.Module | None = None):
        self.lib = lib
        self.num_future = int(num_future)
        self.device = torch.device(device)
        if encoder is None:
            path = default_encoder_path()
            if path is None:
                raise FileNotFoundError(
                    "SONIC encoder ONNX not found; set $SONIC_ENCODER_ONNX to model_encoder.onnx"
                )
            encoder = G1TokenEncoder.from_onnx(path)
        self.encoder = encoder.to(self.device).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.perm = self.encoder.perm

    @torch.no_grad()
    def __call__(self, frame: torch.Tensor, ref=None, state: torch.Tensor | None = None, robot_quat: torch.Tensor | None = None) -> torch.Tensor:
        """Token for the current frames.

        ``robot_quat`` (``(N,4)`` wxyz) is used by the env-side rollout; otherwise
        the robot orientation is taken from the 71-d canonical ``state`` (used by
        the imagination rollout).
        """
        n = frame.numel()
        nf = self.num_future
        last = self.lib.last_frame_of(frame)
        offs = torch.arange(nf, device=frame.device)
        idx = torch.minimum(frame[:, None] + offs[None, :], last[:, None])
        flat = idx.reshape(-1)
        r = self.lib.ref(flat)
        pos = r.joint_pos[..., self.perm.to(r.joint_pos.device)].reshape(n, nf * 29)
        vel = r.joint_vel[..., self.perm.to(r.joint_vel.device)].reshape(n, nf * 29)
        command = torch.cat([pos, vel], dim=-1).reshape(n, nf, 58)
        if robot_quat is None:
            if state is None:
                raise ValueError("pass robot_quat or state")
            robot_quat = state[..., 3:7]
        q_robot = robot_quat.repeat_interleave(nf, dim=0).reshape(n, nf, 4)
        q_ref = r.root_quat_w.reshape(n, nf, 4)
        anchor = _rot6d(_quat_mul(_quat_conj(q_robot), q_ref))
        return self.encoder(command, anchor).detach()


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.stack((q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]), dim=-1)


def _rot6d(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    r = torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1),
        ),
        dim=-2,
    )
    return r[..., :, :2].reshape(*r.shape[:-2], 6)


__all__ = [
    "COMMAND_OFFSET",
    "ANCHOR_OFFSET",
    "G1TokenEncoder",
    "INPUT_DIM",
    "NUM_FUTURE",
    "SonicTokenBuilder",
    "TOKEN_DIM",
    "default_encoder_path",
]
