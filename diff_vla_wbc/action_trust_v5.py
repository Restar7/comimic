"""v5 **物理动作 trust**（29D rad）：把想象的归一化动作换算成关节目标角再量预算。

规范 §7.1 / §7.2 的硬口径：
* ``q_target = default_joint_pos + action_scale * a``（**rad**；`isaac_joint.py:754` 同口径）；
* 参照必须是"**同起点、同 SONIC obs、Δ=0**"的动作（on-the-fly 下每步现算一次 Δ=0 的
  no_grad rollout；**不许**换成"上一批的参照"）；
* ``e = q_corr − q_base``（detach 掉的参照），``L_act = relu(RMS_M(e)/τ − 1)²``，τ=0.020 rad；
* ``L_smooth = mean((e_{t+1}−e_t)²/29) / τ²``；
* ``L_anchor = Σ(Δ²·mask) / (K·B) / cap²``。

本模块只做**纯张量运算**（不碰 bundle / 设备），因此单测可以在 CPU 上逐位核对。
"""
from __future__ import annotations

import torch

ACTION_DIM = 29


def q_target_from_action(default_joint_pos: torch.Tensor, action_scale: torch.Tensor,
                         action: torch.Tensor) -> torch.Tensor:
    """归一化动作 → **物理关节目标角(rad)**。

    ``default_joint_pos`` / ``action_scale`` 是 ``(29,)``（HA_V31 序，与 ``bundle.act()`` 的
    输出同序）；``action`` 可以是 ``(H,B,29)`` / ``(B,H,29)`` / ``(...,29)``。
    """
    a = action.float()
    if int(a.shape[-1]) != ACTION_DIM:
        raise ValueError(f"[trust:v5] 动作最后一维必须是 {ACTION_DIM}，得到 {tuple(a.shape)}")
    dq = default_joint_pos.float().reshape(-1)
    sc = action_scale.float().reshape(-1)
    if int(dq.numel()) != ACTION_DIM or int(sc.numel()) != ACTION_DIM:
        raise ValueError(f"[trust:v5] default_joint_pos/action_scale 必须是 {ACTION_DIM} 维，"
                         f"得到 {int(dq.numel())}/{int(sc.numel())}")
    return dq.to(a.device).to(a.dtype) + sc.to(a.device).to(a.dtype) * a


def joint_error_rad(q_corr: torch.Tensor, q_base: torch.Tensor) -> torch.Tensor:
    """``e = q_corr − q_base``（``q_base`` 必须 detach：参照是常数）。"""
    if tuple(q_corr.shape) != tuple(q_base.shape):
        raise ValueError(f"[trust:v5] 形状不一致 {tuple(q_corr.shape)} vs {tuple(q_base.shape)}")
    if q_base.requires_grad:
        raise ValueError("[trust:v5] q_base 必须 detach（参照是常数，不许把梯度带回参照）")
    return q_corr.float() - q_base.float()


def masked_rms(e: torch.Tensor, mask: torch.Tensor, dim: int = ACTION_DIM) -> tuple[torch.Tensor, torch.Tensor]:
    """逐样本 29D RMS（rad）与有效步数。``e (B,H,29)``、``mask (B,H)``（1=计权）。"""
    if e.ndim != 3 or int(e.shape[-1]) != int(dim):
        raise ValueError(f"[trust:v5] e 必须是 (B,H,{dim})，得到 {tuple(e.shape)}")
    m = mask.float()
    if tuple(m.shape) != tuple(e.shape[:2]):
        raise ValueError(f"[trust:v5] mask 必须是 {tuple(e.shape[:2])}，得到 {tuple(m.shape)}")
    num = (e.pow(2).sum(-1) * m).sum(-1)
    den = (float(dim) * m.sum(-1)).clamp_min(1e-8)
    return (num / den).clamp_min(0.0).sqrt(), m.sum(-1)


def physical_action_trust(e: torch.Tensor, mask: torch.Tensor, *,
                          tau: float = 0.020) -> tuple[torch.Tensor, torch.Tensor]:
    """``L_act``（标量）与逐样本 RMS（rad）。``e (B,H,29)``、``mask (B,H)``。"""
    if not (tau > 0):
        raise ValueError(f"[trust:v5] tau={tau} 必须 > 0")
    rms, _ = masked_rms(e, mask)
    loss = torch.relu(rms / float(tau) - 1.0).pow(2).mean()
    return loss, rms


def smoothness_penalty(e: torch.Tensor, mask: torch.Tensor, *,
                       tau: float = 0.020) -> torch.Tensor:
    """``L_smooth``：相邻控制步的关节目标差（只用两端都有效的步）。"""
    if e.ndim != 3:
        raise ValueError(f"[trust:v5] e 必须是 (B,H,29)，得到 {tuple(e.shape)}")
    if int(e.shape[1]) < 2:
        return e.sum() * 0.0
    de = e[:, 1:] - e[:, :-1]
    m = mask[:, 1:] * mask[:, :-1]
    denom = m.sum().clamp_min(1e-8)
    return (de.pow(2).mean(-1) * m).sum() / denom / float(tau) ** 2


def anchor_penalty(delta: torch.Tensor, mask: torch.Tensor, *, cap: float) -> torch.Tensor:
    """``L_anchor = Σ(Δ²·mask)/(K·B)/cap²``。``delta (B,T,64)``、``mask (B,T)``。"""
    if delta.ndim != 3 or tuple(delta.shape[:2]) != tuple(mask.shape):
        raise ValueError(f"[trust:v5] delta/mask 形状不符：{tuple(delta.shape)} vs {tuple(mask.shape)}")
    if not (cap > 0):
        raise ValueError(f"[trust:v5] cap={cap} 必须 > 0")
    m = mask.float()[..., None]
    k = m.sum().clamp_min(1e-8)                     # = K·B（有效位置数）
    return (delta.float().pow(2) * m).sum() / k / float(cap) ** 2


@torch.no_grad()
def action_budget_stats(e: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """动作预算报表（规范 §0 验收）：29D rad 增量的 RMS / peak / p99（mm）。"""
    rms, n = masked_rms(e.detach(), mask)
    m = mask.detach().bool()
    if int(m.sum()) == 0:
        return dict(rms_mrad=0.0, peak_mrad=0.0, p99_mrad=0.0, steps=0.0, samples=0.0)
    sel = e.detach().permute(0, 1, 2)[m]            # (N,29)
    peak = sel.abs().max()
    return dict(
        rms_mrad=float(rms.mean()) * 1e3,
        peak_mrad=float(peak) * 1e3,
        p99_mrad=float(sel.abs().flatten().quantile(0.99)) * 1e3,
        steps=float(n.mean()),
        samples=float(int(m.sum())),
    )


__all__ = [
    "ACTION_DIM",
    "action_budget_stats",
    "anchor_penalty",
    "joint_error_rad",
    "masked_rms",
    "physical_action_trust",
    "q_target_from_action",
    "smoothness_penalty",
]
