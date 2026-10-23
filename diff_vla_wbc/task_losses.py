"""Examples of differentiable task losses built on a predicted rollout.

The existing SuperMimic world model is robot-only.  These helpers let you add
end-effector / root / posture objectives now; a manipulation task still needs
an object + contact world model (not included here).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .compat import TRACKED_BODIES
from .rollout import Rollout
from .supermimic_backend import SuperMimicBundle


def body_position_loss(
    bundle: SuperMimicBundle,
    rollout: Rollout,
    body_name: str,
    target_xyz: torch.Tensor,
    *,
    start: int = 1,
) -> torch.Tensor:
    """Track a world-frame link position (target ``(B,3)`` or ``(H,B,3)``)."""
    if body_name not in TRACKED_BODIES:
        raise ValueError(f"{body_name} is not a tracked body: {TRACKED_BODIES}")
    idx = TRACKED_BODIES.index(body_name)
    bodies = bundle.assets.heads.robot_bodies(rollout.states[start:])
    pred = bodies["pos"][:, :, idx]  # (h,B,3)
    if target_xyz.ndim == 2:
        target = target_xyz[None].expand(pred.shape[0], -1, -1)
    elif target_xyz.ndim == 3:
        target = target_xyz
    else:
        raise ValueError("target_xyz must be (B,3) or (H,B,3)")
    return F.mse_loss(pred, target)


def end_effector_loss(
    bundle: SuperMimicBundle,
    rollout: Rollout,
    target_xyz: torch.Tensor,
    *,
    body_name: str = "left_wrist_yaw_link",
) -> torch.Tensor:
    return body_position_loss(bundle, rollout, body_name, target_xyz)


def root_height_loss(rollout: Rollout, target_height: float | torch.Tensor = 0.76) -> torch.Tensor:
    h = rollout.states[1:, :, 2]
    target = torch.as_tensor(target_height, dtype=h.dtype, device=h.device)
    return F.mse_loss(h, target.expand_as(h))


def upright_loss(rollout: Rollout) -> torch.Tensor:
    """Penalize the predicted torso up-vector deviating from world +z."""
    from .compat import quat_apply_inverse

    q = rollout.states[1:, :, 3:7]
    up = quat_apply_inverse(q, torch.tensor([0.0, 0.0, 1.0], dtype=q.dtype, device=q.device).expand(*q.shape[:-1], 3))
    return (1.0 - up[..., 2]).square().mean()


__all__ = ["body_position_loss", "end_effector_loss", "root_height_loss", "upright_loss"]
