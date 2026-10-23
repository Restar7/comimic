"""B-mode SONIC token provider: reference motion + simulated robot state -> 64-D token.

This is the in-loop counterpart of the frozen encoder (``sonic_encoder``).  At
every SuperMimic training step:

1. gather the future reference window from ``MotionLibrary`` (10 frames of
   ``joint_pos`` / ``joint_vel`` / ``root_quat_w``),
2. build the encoder input (IsaacLab joint order + ``R_robot^T R_ref`` anchor),
3. run the frozen encoder and return the **quantized, detached** 64-D token.

The token is a conditioning input, never a learned representation: ``detach`` is
mandatory, otherwise gradients would flow back into the robot state and the
policy could learn to game the encoder.

Usage::

    provider = SonicTokenProvider(device=sim_device)
    token = provider.from_state(lib, frame, state_71)      # (N, 64)
    a = policy(obs_hist, token, interface="token")
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .sonic_encoder import COMMAND_FRAMES, G1MotionEncoder, mjcf_to_isaaclab_perm


@dataclass
class SonicTokenProvider:
    """Builds SONIC motion tokens on the fly (B mode: robot state in the loop)."""

    encoder: G1MotionEncoder | None = None
    num_future: int = COMMAND_FRAMES
    device: str | torch.device = "cuda"
    quantize: bool = True

    def __post_init__(self) -> None:
        if self.encoder is None:
            self.encoder = G1MotionEncoder.from_onnx()
        self.encoder = self.encoder.to(torch.device(self.device)).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self._perm: torch.Tensor | None = None

    # ------------------------------------------------------------------ helpers
    def _permute(self, device: torch.device) -> torch.Tensor:
        if self._perm is None or self._perm.device != device:
            self._perm = torch.as_tensor(mjcf_to_isaaclab_perm(), dtype=torch.long, device=device)
        return self._perm

    @torch.no_grad()
    def from_reference(
        self,
        ref_frames,
        robot_root_quat_wxyz: torch.Tensor | None,
    ) -> torch.Tensor:
        """``ref_frames`` = ``RefState`` for frames ``[t .. t+F)`` flattened to ``(N*F,)``.

        ``robot_root_quat_wxyz`` is ``(N*F, 4)`` (``None`` means perfect tracking,
        i.e. the offline A-mode).  Returns a ``(N, 64)`` detached token.
        """
        nf = int(self.num_future)
        joint_pos = ref_frames.joint_pos
        joint_vel = ref_frames.joint_vel
        if joint_pos.shape[0] % nf != 0:
            raise ValueError(f"expected N*{nf} reference frames, got {joint_pos.shape[0]}")
        n = joint_pos.shape[0] // nf
        device = joint_pos.device

        joint_pos = joint_pos.reshape(n, nf, 29)
        joint_vel = joint_vel.reshape(n, nf, 29)
        ref_quat = ref_frames.root_quat_w.reshape(n, nf, 4)

        # Build the windows directly in (N, F) form: the window is already
        # gathered by the library, so reuse the per-frame layout instead of
        # ``build_motion_windows`` (which slides over a contiguous clip).
        perm = self._permute(device)
        pos = joint_pos[..., perm].reshape(n, nf * 29)
        vel = joint_vel[..., perm].reshape(n, nf * 29)
        command = torch.cat([pos, vel], dim=-1).reshape(n, nf, 58)

        if robot_root_quat_wxyz is None:
            anchor = torch.zeros(n, nf, 6, dtype=joint_pos.dtype, device=device)
            anchor[..., 0] = 1.0
            anchor[..., 3] = 1.0
        else:
            q_robot = robot_root_quat_wxyz.reshape(n, nf, 4)
            q_diff = _quat_mul(_quat_conj(q_robot), ref_quat)
            anchor = _rot6d(q_diff)

        token = self.encoder(command, anchor) if self.quantize else self.encoder.forward_raw(command, anchor)
        return token.detach()

    @torch.no_grad()
    def from_state(self, lib, frame: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        """Convenience: gather the future window from ``MotionLibrary`` + 71-d state.

        ``state`` is SuperMimic's canonical 71-d vector
        ``[root pos(3), root quat(4), joint pos(29), root lin vel(3), root ang vel(3), joint vel(29)]``.
        ``state=None`` falls back to perfect-tracking (A mode).
        """
        frame = frame.reshape(-1)
        nf = int(self.num_future)
        last = lib.last_frame_of(frame)
        offs = torch.arange(nf, device=frame.device)
        idx = torch.minimum(frame[:, None] + offs[None, :], last[:, None]).reshape(-1)
        ref = lib.ref(idx)
        robot_q = None if state is None else state[..., 3:7].repeat_interleave(nf, dim=0)
        return self.from_reference(ref, robot_q)


# --- local geometry helpers (avoid importing diff_vla_wbc here) --------------
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


__all__ = ["SonicTokenProvider"]
