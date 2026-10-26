"""token 级 critic + GAE(λ)（照抄 SuperMimic 的信用分配，但条件换成 token 路线）。

SuperMimic 的 WBC 训练里（`trackerlab/.../runner.py`）：

    J += γ^H · c_H · V(s_H, f, ξ, goal)        # bootstrap
    critic 用 GAE(λ) 在**想象转移**上回归

它刻意**不吃 token**（"the critic never takes the token"）—— value 是"在这个状态、这个参考下往后能拿多少"。
我们照这个设计：``V(state71, ref_state71, xi33)``，~1.4e5 参数的小 MLP（和他们的 critic 同量级）。

为什么需要它：我们的 per-step 奖励很噪（速度通道），16/32 步 pathwise 梯度方差大；
GAE 平滑 + bootstrap 能把"短视密集奖励"变成"长视回报"，这是 SuperMimic 能在想象里稳住的关键。
**注意**：critic 会把 WM 的偏差一起自举进去，所以必须配合 WM 保鲜（`wm_refresh.py`）。
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class TokenCritic(nn.Module):
    """``V(state, ref_state, xi)``：状态 71 + 参考状态 71 + 上下文 33。"""

    def __init__(self, state_dim: int = 71, ref_dim: int = 71, xi_dim: int = 33,
                 hidden: int = 256, depth: int = 2):
        super().__init__()
        dims = [state_dim + ref_dim + xi_dim] + [hidden] * depth
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.SiLU()]
        layers += [nn.Linear(dims[-1], 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, ref_state: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state.float(), ref_state.float(), xi.float()), dim=-1)
        return self.net(x).squeeze(-1)


def gae_targets(rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor,
                *, gamma: float = 0.99, lam: float = 0.95) -> torch.Tensor:
    """``(H,B)`` 奖励/done/价值 → GAE(λ) 目标 ``(H,B)``（全部 detach）。"""
    H = rewards.shape[0]
    R = values[-1]
    out = []
    for t in range(H - 1, -1, -1):
        R = rewards[t] + gamma * (1.0 - dones[t]) * ((1.0 - lam) * values[t + 1] + lam * R)
        out.append(R)
    return torch.stack(out[::-1])


@torch.no_grad()
def value_estimate(critic: nn.Module, states: torch.Tensor, ref_states: torch.Tensor,
                   xi: torch.Tensor) -> torch.Tensor:
    return critic(states, ref_states, xi)


def critic_update(
    critic: nn.Module,
    opt: torch.optim.Optimizer,
    states: torch.Tensor,          # (H+1,B,71) 想象状态（含起点）
    ref_states: torch.Tensor,      # (H+1,B,71) 参考状态
    xi: torch.Tensor,              # (B,33)
    rewards: torch.Tensor,         # (H,B)
    dones: torch.Tensor,           # (H,B)
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
    epochs: int = 5,
    minibatches: int = 4,
    clip_norm: float = 1.0,
    clip_value: float = 0.1,
    log: Callable | None = None,
) -> dict[str, float]:
    """在想象转移上回归 GAE 目标（SuperMimic 的 critic 训练口径）。"""
    Hp1, B = states.shape[0], states.shape[1]
    with torch.no_grad():
        # 展平 (H+1)*B；ref/xi 按步重复
        flat_s = states.reshape(-1, states.shape[-1])
        flat_ref = ref_states.reshape(-1, ref_states.shape[-1])
        flat_xi = xi.unsqueeze(0).expand(Hp1, -1, -1).reshape(-1, xi.shape[-1])
        V = value_estimate(critic, flat_s, flat_ref, flat_xi).view(Hp1, B)
        targets = gae_targets(rewards, dones, V, gamma=gamma, lam=lam)
        X_s, X_r, X_xi = flat_s[:-B], flat_ref[:-B], flat_xi[:-B]
        Y = targets.reshape(-1)
    n = X_s.shape[0]
    if n == 0:
        return {"v_loss": 0.0, "v_ev": 0.0}
    losses = []
    v_pred_last = None
    for _ in range(int(epochs)):
        perm = torch.randperm(n, device=X_s.device)
        mb = max(1, n // int(minibatches))
        for i in range(int(minibatches)):
            idx = perm[i * mb:(i + 1) * mb]
            v = critic(X_s[idx], X_r[idx], X_xi[idx])
            loss = ((v - Y[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), clip_norm)
            torch.nn.utils.clip_grad_value_(critic.parameters(), clip_value)
            opt.step()
            losses.append(float(loss.detach()))
            v_pred_last = v.detach()
    with torch.no_grad():
        ev = float(1.0 - ((v_pred_last - Y[:v_pred_last.shape[0]]) ** 2).mean()
                   / ((Y[:v_pred_last.shape[0]] - Y[:v_pred_last.shape[0]].mean()) ** 2).mean().clamp_min(1e-9))
    return {"v_loss": sum(losses) / len(losses), "v_ev": ev}


__all__ = ["TokenCritic", "critic_update", "gae_targets", "value_estimate"]
