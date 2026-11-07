"""Reward-only Teacher for the **shared broadcast Δtoken**, plus its gradient probes.

部署口径（唯一被允许的施加方式）
------------------------------
::

    token'_t = token_t + Δθ(token_0, state64)          # Δ 在整个 chunk 上**共享**
    a_t      = WBC(token'_t, obs_t)                     # 冻结执行器
    s_{t+1}  = WM(s_t, a_t, ξ)                          # 冻结世界模型
    J        = Σ_t γ^t · survival_t · r(s_t, a_t, a_{t-1}, ref_t)

因为 Δ 是**同一个向量**被广播到 T 个位置，BPTT 对它的梯度 = 各时间步 token 梯度之和::

    g = dJ/dΔ = Σ_t ∂J/∂token_t

本模块给出两种**独立**的算法去算同一个量，用于互相验证：

* :func:`shared_delta_reward_grad`  —— 对 ``eps`` (B,1,64) 求导（部署口径，Teacher 用的就是它）
* :func:`tokenwise_reward_grad`     —— 对 ``tokens`` (B,T,64) 逐步求导再沿 T 求和（链式法则对照）
* :func:`finite_difference_directional` —— 中心差分方向导数（不经过 autograd 的第三方对照）

三者必须在确定性 FP32、同一 ``J`` 定义下数值一致；不一致就说明图里有 detach / 重复索引 /
非共享广播之类的口径错误。这里**不包含任何训练 loss**（critic / trust / act / 归一化项），
那些项在 ``v3_step`` 的 ``J1_EPS_GRAD`` 诊断里单独量。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .rollout import discounted_return, imagine_with_tokens

#: Teacher 目标版本号（写进 checkpoint meta；会话 01 的部署侧按它判断标签语义）
TEACHER_OBJECTIVE_VERSION = "shared_delta_reward_only_v2"
#: 旧 Teacher（单 token0 重复 H 步）的版本号，仅作 legacy 对照
LEGACY_TEACHER_OBJECTIVE_VERSION = "legacy_token0_repeat_v1"


@dataclass
class TeacherGrad:
    """一次 Teacher 求导的全部中间量。"""

    J: torch.Tensor            # (B,) 每窗口回报（已 detach）
    J_undisc: torch.Tensor     # (B,) 未折扣 Σ_t r_t（口径对照）
    g: torch.Tensor            # (B,D) dJ/dΔ（已 detach）
    g_undisc: torch.Tensor     # (B,D) dJ_undisc/dΔ（口径对照）
    eps: torch.Tensor          # (B,1,D) 求导点（detach 后的值）
    epss: torch.Tensor | None  # (B,T,D) 被替换的 token chunk（诊断用，已 detach）
    horizon: int
    rollout: Any = None


def as_chunk(tokens: torch.Tensor) -> torch.Tensor:
    """``(B,D)`` / ``(B,1,D)`` / ``(B,T,D)`` → ``(B,T,D)``（**不做**任何 expand/repeat）。"""
    if tokens.ndim == 2:
        tokens = tokens[:, None, :]
    if tokens.ndim != 3:
        raise ValueError(f"tokens must be (B,T,D) / (B,1,D) / (B,D), got {tuple(tokens.shape)}")
    return tokens


def shared_eps(base_tokens: torch.Tensor, *, eps0: torch.Tensor | None = None) -> torch.Tensor:
    """共享可微变量 ``eps`` (B,1,D)，初值 = ``eps0``（默认 0）。"""
    base = as_chunk(base_tokens)
    if eps0 is None:
        eps = torch.zeros(base.shape[0], 1, base.shape[-1],
                          dtype=base.dtype, device=base.device)
    else:
        eps = as_chunk(eps0).clone()
        if eps.shape[1] != 1:
            raise ValueError("eps0 必须是共享的 (B,1,D)（部署口径只在 chunk 上广播一个 Δ）")
    return eps.detach().clone().requires_grad_(True)


def _rollout(bundle, tokens, *, state, references, xi, prev_action, history, obs_hist, horizon):
    return imagine_with_tokens(
        bundle,
        tokens=tokens,
        state=state,
        obs_hist=obs_hist,
        history=history,
        xi=xi,
        references=references,
        prev_action=prev_action,
        horizon=int(horizon),
    )


def _J_pair(rollout, gamma: float, reward_steps: int):
    """同一 rollout 上的两种回报定义：部署/训练口径（折扣）与未折扣和。"""
    J = discounted_return(rollout, float(gamma), steps=int(reward_steps or 0))
    J_undisc = rollout.rewards.sum(0)
    return J, J_undisc


def shared_delta_reward_grad(
    bundle,
    base_tokens: torch.Tensor,
    *,
    state: torch.Tensor,
    references: Sequence[Any],
    xi: torch.Tensor | None = None,
    prev_action: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
    obs_hist: torch.Tensor | None = None,
    horizon: int = 20,
    gamma: float = 0.99,
    reward_steps: int = 0,
    eps0: torch.Tensor | None = None,
    keep_rollout: bool = False,
    grad_undisc: bool = True,
) -> TeacherGrad:
    """``g = dJ/dΔ``，Δ 广播到**完整 base token chunk**。

    这是 Teacher 的唯一正确算法：base chunk 来自与训练同一次冻结 VLA 前向，
    **不允许**用录制 token 替换（除非调用方显式传入录制 chunk 做对照）。
    """
    base = as_chunk(base_tokens).detach()
    eps = shared_eps(base, eps0=eps0)
    tok = base + eps
    roll = _rollout(bundle, tok, state=state, references=references, xi=xi,
                    prev_action=prev_action, history=history, obs_hist=obs_hist,
                    horizon=horizon)
    J, J_undisc = _J_pair(roll, gamma, reward_steps)
    need_graph = grad_undisc
    g = torch.autograd.grad(J.sum(), eps, retain_graph=need_graph)[0][:, 0, :].detach()
    if grad_undisc:
        g_undisc = torch.autograd.grad(J_undisc.sum(), eps)[0][:, 0, :].detach()
    else:
        g_undisc = torch.zeros_like(g)
    return TeacherGrad(J=J.detach(), J_undisc=J_undisc.detach(), g=g, g_undisc=g_undisc,
                       eps=eps.detach(), epss=base.detach(),
                       horizon=int(horizon), rollout=roll if keep_rollout else None)


def tokenwise_reward_grad(
    bundle,
    base_tokens: torch.Tensor,
    *,
    state: torch.Tensor,
    references: Sequence[Any],
    xi: torch.Tensor | None = None,
    prev_action: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
    obs_hist: torch.Tensor | None = None,
    horizon: int = 20,
    gamma: float = 0.99,
    reward_steps: int = 0,
    eps0: torch.Tensor | None = None,
    grad_undisc: bool = True,
) -> TeacherGrad:
    """``g = Σ_t dJ/dtoken_t``：独立 autograd，**不做**共享广播。

    ``tokens`` 是叶子 (B,T,D)；``imagine_with_tokens`` 内部按 ``tokens[:, min(t,T-1)]``
    取用（H > T 时最后一个 token 被重复使用 ⇒ 它对梯度的贡献自动累加），所以
    ``dJ/dtoken.sum(1)`` 与共享 ``eps`` 的梯度必须逐位一致（在相同 ``J`` 下）。
    """
    base = as_chunk(base_tokens).detach()
    _init = base + (as_chunk(eps0).detach() if eps0 is not None else 0.0)
    tok = _init.detach().clone().requires_grad_(True)
    roll = _rollout(bundle, tok, state=state, references=references, xi=xi,
                    prev_action=prev_action, history=history, obs_hist=obs_hist,
                    horizon=horizon)
    J, J_undisc = _J_pair(roll, gamma, reward_steps)
    need_graph = grad_undisc
    g_tok = torch.autograd.grad(J.sum(), tok, retain_graph=need_graph)[0]
    g = g_tok.sum(1).detach()
    if grad_undisc:
        g_undisc = torch.autograd.grad(J_undisc.sum(), tok)[0].sum(1).detach()
    else:
        g_undisc = torch.zeros_like(g)
    return TeacherGrad(J=J.detach(), J_undisc=J_undisc.detach(), g=g, g_undisc=g_undisc,
                       eps=torch.zeros(base.shape[0], 1, base.shape[-1], device=base.device),
                       epss=base.detach() if eps0 is None else (base + as_chunk(eps0).detach()),
                       horizon=int(horizon), rollout=None)


@torch.no_grad()
def _J_only(bundle, tokens, *, state, references, xi, prev_action, history, obs_hist,
            horizon, gamma, reward_steps):
    roll = _rollout(bundle, tokens, state=state, references=references, xi=xi,
                    prev_action=prev_action, history=history, obs_hist=obs_hist,
                    horizon=horizon)
    J, _ = _J_pair(roll, gamma, reward_steps)
    return J


@torch.no_grad()
def finite_difference_directional(
    bundle,
    base_tokens: torch.Tensor,
    *,
    direction: torch.Tensor,
    h: float,
    state: torch.Tensor,
    references: Sequence[Any],
    xi: torch.Tensor | None = None,
    prev_action: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
    obs_hist: torch.Tensor | None = None,
    horizon: int = 20,
    gamma: float = 0.99,
    reward_steps: int = 0,
    eps0: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(J(ε+hv) − J(ε−hv)) / 2h``，返回每窗口标量 ``(B,)``。

    方向 ``v`` 必须是共享的 ``(B,1,D)``（与 Δ 同口径）。不做 autograd。
    """
    base = as_chunk(base_tokens).detach()
    v = as_chunk(direction).detach()
    if v.shape[1] != 1:
        raise ValueError("finite-difference direction must be shared (B,1,D)")
    e0 = torch.zeros(base.shape[0], 1, base.shape[-1], device=base.device)
    if eps0 is not None:
        e0 = as_chunk(eps0).detach()
    h = float(h)
    Jp = _J_only(bundle, base + e0 + h * v, state=state, references=references, xi=xi,
                 prev_action=prev_action, history=history, obs_hist=obs_hist,
                 horizon=horizon, gamma=gamma, reward_steps=reward_steps)
    Jm = _J_only(bundle, base + e0 - h * v, state=state, references=references, xi=xi,
                 prev_action=prev_action, history=history, obs_hist=obs_hist,
                 horizon=horizon, gamma=gamma, reward_steps=reward_steps)
    return ((Jp - Jm) / (2.0 * h)).detach()


def unit_labels(g: torch.Tensor, eta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``η·unit(ĝ)`` 标签 + 每窗口梯度范数 + 有效掩码（有限且非零）。"""
    n = g.norm(dim=-1)
    finite = torch.isfinite(g).all(dim=-1)
    valid = finite & (n > 1e-12)
    den = n.clamp_min(1e-12)[:, None]
    tgt = (float(eta) * g / den).masked_fill(~valid[:, None], 0.0)
    return tgt, n, valid


def cosine(a: torch.Tensor, b: torch.Tensor, *, mask: torch.Tensor | None = None) -> torch.Tensor:
    """逐行余弦 ``(N,)``；``mask`` 之外的行走 0。"""
    if a.shape != b.shape:
        raise ValueError(f"cosine shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    na = a.norm(dim=-1).clamp_min(1e-12)
    nb = b.norm(dim=-1).clamp_min(1e-12)
    c = (a * b).sum(-1) / (na * nb)
    if mask is not None:
        c = c.masked_fill(~mask, 0.0)
    return c


__all__ = [
    "TEACHER_OBJECTIVE_VERSION",
    "LEGACY_TEACHER_OBJECTIVE_VERSION",
    "TeacherGrad",
    "as_chunk",
    "shared_eps",
    "shared_delta_reward_grad",
    "tokenwise_reward_grad",
    "finite_difference_directional",
    "unit_labels",
    "cosine",
]
