"""Frozen SONIC g1 motion encoder (10 x 64 -> FSQ token) re-implemented in pure PyTorch.

The deployed ``model_encoder.onnx`` contains three motion encoders (g1 / teleop /
smpl) selected by ``encoder_index``.  Only the g1 branch is needed here: its input
is a 10-frame window of ``[dof_pos(29), dof_vel(29), anchor_6d(6)]`` (interleaved
per frame, 640 = 10 x 64), and its output is a 64-D FSQ token.

Why re-implement instead of calling onnxruntime: the token is computed **inside the
training loop** (option B in the design), batched over thousands of envs, on GPU.
A PyTorch module avoids per-step Python/device copies and lets the token be
detached like any other conditioning input.

Weights are extracted from the ONNX initializers at load time (no separate
checkpoint is shipped).  The FSQ step is reproduced exactly::

    token = round(tanh(h + 0.032237) * 15.515501 - 0.5) / 16.0

with ``h`` the 64-D MLP output (640 -> 2048 -> 1024 -> 512 -> 512 -> 64, SiLU).

Layout constants recovered from the ONNX graph (``model_encoder.onnx``):

    obs[0]         encoder_index
    obs[3:583]     command_multi_future_nonflat (10 x 58 = dof_pos, dof_vel)
    obs[594:600]   motion_anchor_ori_b            (current frame, teleop branch)
    obs[600:660]   motion_anchor_ori_b_mf_nonflat (10 x 6, g1 branch anchor)

Usage::

    enc = G1MotionEncoder.from_onnx()
    token = enc(command, anchor_mf)          # (B, 10, 58), (B, 10, 6) -> (B, 64)
    token = token.detach()                   # conditioning input: never backprop through it
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

#: Candidate locations of the deployed encoder (first hit wins); ``$SONIC_ENCODER_ONNX`` overrides.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENCODER_REL = Path("GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx")
DEFAULT_ENCODER_ONNX = _REPO_ROOT / _ENCODER_REL


def resolve_encoder_onnx(path: str | Path | None = None) -> Path:
    """Find ``model_encoder.onnx``: explicit arg, then env var, then repo/sibling workspace."""
    import os

    if path is not None:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    env = os.environ.get("SONIC_ENCODER_ONNX")
    candidates = [Path(env)] if env else []
    sm_root = _REPO_ROOT / "third_party" / "SuperMimic-Reproduced"
    candidates += [
        sm_root / "data" / "assets" / "model_encoder.onnx",
        _REPO_ROOT / "data" / "assets" / "model_encoder.onnx",
        _REPO_ROOT / _ENCODER_REL,
        _REPO_ROOT.parent / _ENCODER_REL,
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "SONIC model_encoder.onnx not found; tried: "
        + ", ".join(str(c) for c in candidates)
        + " (set $SONIC_ENCODER_ONNX)"
    )

# --- obs layout inside the 1762-D ONNX input ---
# NOTE: the graph first slices ``obs[1:]`` (dropping ``encoder_index``) and only
# then slices the g1 fields, so every downstream offset is +1 relative to the
# constants printed inside the ONNX graph ([3] -> 4, [600] -> 601).  Verified by
# exact parity against onnxruntime (see tests/test_sonic_encoder.py).
ENCODER_INDEX_OFFSET = 0
COMMAND_OFFSET = 4
COMMAND_FRAMES = 10
COMMAND_FRAME_DIM = 58  # [dof_pos(29), dof_vel(29)]
ANCHOR_MF_OFFSET = 601
ANCHOR_FRAME_DIM = 6
INPUT_TOTAL_DIM = 1762

# --- FSQ quantizer constants (scalars in the exported graph) ---
FSQ_OFFSET = 0.03223739564418793
FSQ_SCALE = 15.515501022338867
FSQ_BIAS = 0.5
FSQ_DIV = 16.0

#: MLP hidden sizes of the g1 branch, in ONNX order.
_G1_LAYERS = ((2048, 640), (1024, 2048), (512, 1024), (512, 512), (64, 512))


class G1MotionEncoder(nn.Module):
    """640 -> 64 MLP + FSQ, weights extracted from ``model_encoder.onnx``."""

    def __init__(self, linears: list[nn.Linear]):
        super().__init__()
        if len(linears) != len(_G1_LAYERS):
            raise ValueError(f"expected {len(_G1_LAYERS)} linear layers, got {len(linears)}")
        self.linears = nn.ModuleList(linears)

    # ------------------------------------------------------------- construction
    @classmethod
    def from_onnx(cls, path: str | Path | None = None) -> "G1MotionEncoder":
        import onnx
        from onnx import numpy_helper

        src = resolve_encoder_onnx(path)
        model = onnx.load(str(src))
        weights = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        linears: list[nn.Linear] = []
        for out_dim, in_dim in _G1_LAYERS:
            wkey = f"module.encoders.g1.module.{2 * len(linears)}.weight"
            bkey = f"module.encoders.g1.module.{2 * len(linears)}.bias"
            if wkey not in weights or bkey not in weights:
                raise KeyError(f"missing g1 encoder weights {wkey} / {bkey} in {src}")
            w = torch.from_numpy(np.array(weights[wkey], copy=True)).float()
            b = torch.from_numpy(np.array(weights[bkey], copy=True)).float()
            if tuple(w.shape) != (out_dim, in_dim):
                raise ValueError(f"{wkey}: shape {tuple(w.shape)} != {(out_dim, in_dim)}")
            lin = nn.Linear(in_dim, out_dim)
            with torch.no_grad():
                lin.weight.copy_(w)
                lin.bias.copy_(b)
            linears.append(lin)
        return cls(linears)

    # ------------------------------------------------------------------ forward
    def forward_raw(self, command: torch.Tensor, anchor_mf: torch.Tensor) -> torch.Tensor:
        """Continuous 64-D embedding (pre-FSQ). ``command (B,10,58)``, ``anchor_mf (B,10,6)``."""
        if command.shape[-2:] != (COMMAND_FRAMES, COMMAND_FRAME_DIM):
            raise ValueError(f"command must be (B, {COMMAND_FRAMES}, {COMMAND_FRAME_DIM}), got {tuple(command.shape)}")
        if anchor_mf.shape[-2:] != (COMMAND_FRAMES, ANCHOR_FRAME_DIM):
            raise ValueError(f"anchor_mf must be (B, {COMMAND_FRAMES}, {ANCHOR_FRAME_DIM}), got {tuple(anchor_mf.shape)}")
        if command.shape[0] != anchor_mf.shape[0]:
            raise ValueError("command and anchor_mf batch sizes differ")
        x = torch.cat([command, anchor_mf], dim=-1).reshape(command.shape[0], -1)  # (B, 640)
        h = x
        for i, lin in enumerate(self.linears):
            h = lin(h)
            if i < len(self.linears) - 1:
                h = torch.nn.functional.silu(h)
        return h

    def fsq(self, h: torch.Tensor) -> torch.Tensor:
        """Exact FSQ step from the exported graph: round to the 1/16 grid."""
        q = torch.round(torch.tanh(h + FSQ_OFFSET) * FSQ_SCALE - FSQ_BIAS) / FSQ_DIV
        return q

    def forward(self, command: torch.Tensor, anchor_mf: torch.Tensor) -> torch.Tensor:
        """Quantized 64-D token (1/16 grid), shape ``(B, 64)``."""
        return self.fsq(self.forward_raw(command, anchor_mf))

    # --------------------------------------------------------------- convenience
    def info(self) -> dict:
        return {
            "layers": [tuple(lin.weight.shape) for lin in self.linears],
            "params": sum(p.numel() for p in self.parameters()),
            "fsq": {"offset": FSQ_OFFSET, "scale": FSQ_SCALE, "bias": FSQ_BIAS, "div": FSQ_DIV},
            "input": {"frames": COMMAND_FRAMES, "frame_dim": COMMAND_FRAME_DIM, "anchor_dim": ANCHOR_FRAME_DIM},
        }


def build_onnx_input(
    command: "torch.Tensor | None" = None,
    anchor_mf: "torch.Tensor | None" = None,
    *,
    batch: int = 1,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Assemble the full 1762-D vector the ONNX encoder expects (g1 fields only)."""
    x = torch.zeros(batch, INPUT_TOTAL_DIM, device=device)
    if command is not None:
        x[:, COMMAND_OFFSET : COMMAND_OFFSET + COMMAND_FRAMES * COMMAND_FRAME_DIM] = command.reshape(batch, -1)
    if anchor_mf is not None:
        x[:, ANCHOR_MF_OFFSET : ANCHOR_MF_OFFSET + COMMAND_FRAMES * ANCHOR_FRAME_DIM] = anchor_mf.reshape(batch, -1)
    return x


__all__ = [
    "ANCHOR_FRAME_DIM",
    "resolve_encoder_onnx",
    "ANCHOR_MF_OFFSET",
    "COMMAND_FRAME_DIM",
    "COMMAND_FRAMES",
    "COMMAND_OFFSET",
    "DEFAULT_ENCODER_ONNX",
    "ENCODER_INDEX_OFFSET",
    "FSQ_BIAS",
    "FSQ_DIV",
    "FSQ_OFFSET",
    "FSQ_SCALE",
    "G1MotionEncoder",
    "INPUT_TOTAL_DIM",
    "build_onnx_input",
]


# ---------------------------------------------------------------------------
# Motion-window construction (what feeds the g1 encoder)
# ---------------------------------------------------------------------------
#: SuperMimic / MJCF joint order is ``G1_JOINT_NAMES``; SONIC expects IsaacLab order.
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


def mjcf_to_isaaclab_perm(mjcf_order: tuple[str, ...] | list[str] | None = None) -> np.ndarray:
    """Permutation ``p`` such that ``q_isaaclab = q_mjcf[..., p]``."""
    from .compat import G1_JOINT_NAMES

    mjcf = list(mjcf_order) if mjcf_order is not None else list(G1_JOINT_NAMES)
    if len(mjcf) != len(G1_ISAACLAB_ORDER):
        raise ValueError(f"expected 29 joints, got {len(mjcf)}")
    missing = [n for n in G1_ISAACLAB_ORDER if n not in mjcf]
    if missing:
        raise ValueError(f"joint order missing {missing[:3]}...")
    return np.asarray([mjcf.index(n) for n in G1_ISAACLAB_ORDER], dtype=int)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product for wxyz quaternions."""
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


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion -> 3x3 rotation matrix (normalising first)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1),
        ),
        dim=-2,
    )


def rot6d(q_robot_wxyz: torch.Tensor, q_ref_wxyz: torch.Tensor) -> torch.Tensor:
    """First two columns of ``R_robot^T R_ref`` (body-frame orientation difference).

    Matches SONIC's ``motion_anchor_ori_b_mf_nonflat`` convention
    (``rot_mat[..., :2].reshape(6)``, row-major).
    """
    q_diff = _quat_mul(_quat_conj(q_robot_wxyz), q_ref_wxyz)
    r = quat_to_rotmat(q_diff)
    return r[..., :, :2].reshape(*r.shape[:-2], 6)


def build_motion_windows(
    joint_pos_mjcf: torch.Tensor,
    joint_vel_mjcf: torch.Tensor,
    ref_root_quat_wxyz: torch.Tensor,
    robot_root_quat_wxyz: torch.Tensor | None = None,
    *,
    num_future: int = COMMAND_FRAMES,
    dt: float = 0.02,
    window: str = "hold",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(command, anchor_mf)`` for every frame from a motion reference.

    Parameters
    ----------
    joint_pos_mjcf, joint_vel_mjcf : ``(T, 29)`` in MJCF / SuperMimic joint order.
    ref_root_quat_wxyz : ``(T, 4)`` reference root orientation (wxyz).
        HA 数据必须是 ``human_body_quat_w_aligned``（heading-aligned 参考朝向），不是
        ``human_body_quat_w``：实测 ``anchor == rot6d(conj(robot_root_orientation) *
        human_body_quat_w_aligned)``，与录制 ``consumed_anchor_rot6d`` 的 mean|Δ| ≤ 6e-5，
        7 个任务上重算 token 与 ``encoder_latent`` 逐位相等（exact_frac 1.0000）。
    robot_root_quat_wxyz : ``(T, 4)`` robot root orientation.  ``None`` means
        "perfect tracking" (the in-loop B-mode should always pass the simulated
        robot state; the identity fallback is only for offline smoke tests).
    num_future : future-frame horizon (SONIC config uses 10).
    dt : unused placeholder for callers that may re-time windows.
    window : ``"hold"``（部署/录制口径：当前参考帧重复 num_future 次，默认）或
        ``"future"``（旧口径：未来 num_future 帧升序，仅作对照）。

    Returns
    -------
    command : ``(T, num_future, 58)``，内部按 **term-major** 580 维排布
        （``[pos_0..pos_9 | vel_0..vel_9]``）；frame 维只为形状，喂 encoder 时会被 flatten。
    anchor  : ``(T, num_future, 6)`` relative 6D orientation difference
    """
    if joint_pos_mjcf.dim() != 2 or joint_pos_mjcf.shape[1] != 29:
        raise ValueError(f"joint_pos must be (T, 29), got {tuple(joint_pos_mjcf.shape)}")
    if joint_vel_mjcf.shape != joint_pos_mjcf.shape:
        raise ValueError("joint_vel must have the same shape as joint_pos")
    if ref_root_quat_wxyz.shape[0] != joint_pos_mjcf.shape[0]:
        raise ValueError("ref_root_quat must have the same T as joint_pos")

    perm = torch.as_tensor(mjcf_to_isaaclab_perm(), dtype=torch.long, device=joint_pos_mjcf.device)
    pos = joint_pos_mjcf[..., perm]
    vel = joint_vel_mjcf[..., perm]

    T = pos.shape[0]
    if str(window).lower() == "hold":
        # 部署/录制口径（S1，2026-10-26）：当前参考帧重复 num_future 次。
        idx = torch.arange(T, device=pos.device).unsqueeze(1).expand(T, num_future).contiguous()
    elif str(window).lower() == "future":
        idx = torch.arange(T, device=pos.device).unsqueeze(1) + torch.arange(num_future, device=pos.device)
        idx = idx.clamp(max=T - 1)
    else:
        raise ValueError(f"window must be hold|future, got {window!r}")
    # ❗布局：encoder 的 command 必须是 **term-major** 的 580 维
    #   ``[pos_0..pos_9 (290) | vel_0..vel_9 (290)]``（= 录制 obs[4:584] 的那个块）。
    #   按帧交错 ``cat([pos[idx], vel[idx]], -1)``（每帧 [pos_j, vel_j]）会得到完全不同的
    #   token：实测 hold 窗口下 exact_frac 0.28 vs 1.00（HOI_double_desk 首条 clip，逐位对拍
    #   `encoder_latent`）。中间那一维只是为了凑 (T, F, 58) 的形状，语义是"扁平 580"。
    command = torch.cat([pos[idx].reshape(T, -1), vel[idx].reshape(T, -1)], dim=-1)
    command = command.reshape(T, num_future, 2 * pos.shape[-1])  # (T, F, 58)，见上注释

    q_ref = ref_root_quat_wxyz[idx]  # (T, F, 4)
    if robot_root_quat_wxyz is None:
        anchor = torch.zeros(T, num_future, 6, dtype=pos.dtype, device=pos.device)
        anchor[..., 0] = 1.0
        anchor[..., 3] = 1.0  # identity rotation's first two columns
    else:
        if robot_root_quat_wxyz.shape != ref_root_quat_wxyz.shape:
            raise ValueError("robot_root_quat must have the same shape as ref_root_quat")
        q_robot = robot_root_quat_wxyz[idx]
        anchor = rot6d(q_robot, q_ref)
    return command, anchor


@torch.no_grad()
def tokens_from_motion(
    encoder: "G1MotionEncoder",
    joint_pos_mjcf: torch.Tensor,
    joint_vel_mjcf: torch.Tensor,
    ref_root_quat_wxyz: torch.Tensor,
    robot_root_quat_wxyz: torch.Tensor | None = None,
    *,
    batch_size: int = 4096,
    num_future: int = COMMAND_FRAMES,
) -> torch.Tensor:
    """Encode a whole clip to ``(T, 64)`` tokens (quantized, detached)."""
    command, anchor = build_motion_windows(
        joint_pos_mjcf, joint_vel_mjcf, ref_root_quat_wxyz, robot_root_quat_wxyz, num_future=num_future
    )
    out = []
    for start in range(0, command.shape[0], batch_size):
        out.append(encoder(command[start : start + batch_size], anchor[start : start + batch_size]))
    return torch.cat(out, dim=0)
