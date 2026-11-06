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
                 hidden: int = 256, depth: int = 2, use_diff: bool = False):
        super().__init__()
        # use_diff：把 ``s − ref`` 显式喂进去。价值几乎完全由"偏离参考多少"决定，
        # 让 MLP 自己从两个 71 维向量里减，样本效率很差（离线拟合实测 v_ev < 0）。
        self.use_diff = bool(use_diff)
        dims = [state_dim + ref_dim + (state_dim if self.use_diff else 0) + xi_dim] + [hidden] * depth
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.SiLU()]
        layers += [nn.Linear(dims[-1], 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, ref_state: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        parts = [state.float(), ref_state.float()]
        if self.use_diff:
            parts.append(state.float() - ref_state.float())
        parts.append(xi.float())
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


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
    # ★ 2026-10-29 修：原来 ev 用"最后一个随机 minibatch 的预测"去比"Y 的前一段"，
    #   索引对不上 ⇒ 想象模式下 v_ev 恒在 −1 附近（纯噪声，不是 critic 坏）。
    #   现在切出 20% 做**held-out**切片，只在它上面算 explained variance。
    n_eval = max(1, n // 5)
    g = torch.Generator(device="cpu").manual_seed(20261029)
    perm_all = torch.randperm(n, generator=g).to(X_s.device)
    idx_eval, idx_train = perm_all[:n_eval], perm_all[n_eval:]
    if idx_train.numel() == 0:
        idx_train = idx_eval
    Xt, Rt, Xit, Yt = X_s[idx_train], X_r[idx_train], X_xi[idx_train], Y[idx_train]
    nt = int(Xt.shape[0])
    losses = []
    for _ in range(int(epochs)):
        perm = torch.randperm(nt, device=Xt.device)
        mb = max(1, nt // int(minibatches))
        for i in range(int(minibatches)):
            idx = perm[i * mb:(i + 1) * mb]
            v = critic(Xt[idx], Rt[idx], Xit[idx])
            loss = ((v - Yt[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), clip_norm)
            torch.nn.utils.clip_grad_value_(critic.parameters(), clip_value)
            opt.step()
            losses.append(float(loss.detach()))
    with torch.no_grad():
        v_ev = critic(X_s[idx_eval], X_r[idx_eval], X_xi[idx_eval])
        Y_ev = Y[idx_eval]
        mse = ((v_ev - Y_ev) ** 2).mean()
        var = Y_ev.var(unbiased=False).clamp_min(1e-9)
        ev = float(1.0 - mse / var)
    return {"v_loss": sum(losses) / len(losses), "v_ev": ev}


__all__ = ["TokenCritic", "critic_update", "gae_targets", "value_estimate"]


def critic_fit_supervised(critic, opt, states, ref_states, xi, targets, *,
                          epochs: int = 5, minibatches: int = 4, batch: int = 256,
                          clip_norm: float = 1.0, clip_value: float = 0.1) -> dict:
    """用**真数据**监督 critic（不经过 WM）：``V(s, ref, xi) → ret_to_go``。

    与 ``critic_update``（想象转移上的 GAE）互补：这一条提供 grounded 的目标，
    避免"critic 拟合想象回报 → bootstrap 又把 WM 偏差放大进策略梯度"。
    """
    import torch.nn.functional as F

    n = int(states.shape[0])
    bs = min(int(batch), n)
    last_v, ev_last, n_seen = 0.0, 0.0, 0
    for _ in range(max(1, int(epochs))):
        for _ in range(max(1, int(minibatches))):
            idx = torch.randint(0, n, (bs,), device=states.device)
            v = critic(states[idx], ref_states[idx], xi[idx])
            y = targets[idx]
            loss = F.mse_loss(v, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), clip_norm)
            torch.nn.utils.clip_grad_value_(critic.parameters(), clip_value)
            opt.step()
            with torch.no_grad():
                var = float(y.var(unbiased=False))
                ev = 1.0 - float(loss) / max(var, 1e-9)
            last_v, ev_last = float(loss), ev
            n_seen += bs
    return {"v_loss": last_v, "v_ev": ev_last, "n": n_seen}


def gae_targets_padded(rewards: torch.Tensor, values: torch.Tensor, groups: torch.Tensor, *,
                       gamma: float = 0.99, lam: float = 0.95) -> torch.Tensor:
    """**真轨迹**上的 TD(λ) 回归目标：``target_t = V_t + A_t``（A 用 GAE 递推）。

    为什么需要（2026-10-30）：
      `critic_source=imagined` 时 critic 只在 8 步想象窗上拟合（H=8=0.16 s），而 token
      实际执行 20 tick=0.4 s，再往后的漂移要几秒才成形。原来靠 `boot=γ^H·V` 自举把视野
      带出去，但 V 自己也是短窗训的 ⇒ 载体失效。把 V 放到**真轨迹**上拟合（整集范围），
      自举项才有长视野。这份函数提供两种目标中的 TD(λ) 那种（另一种是 MC return-to-go）。

    参数
      rewards/values : (N,)  每行的即时奖励 / 当前 V(s)
      groups         : (N,)  episode 或 clip 的 id，**必须连续**（同一组行相邻）
    rows 在组内必须按时间递增（transitions.pt 的布局）。

    说明：组内最后一个元素用 A_T = δ_T（无自举），组与组之间不串味。
    """
    r = rewards.reshape(-1).double()
    v = values.reshape(-1).double()
    g = groups.reshape(-1)
    n = int(r.numel())
    if n == 0:
        return r.float()
    dev = r.device
    new = torch.ones(n, dtype=torch.bool, device=dev)
    new[1:] = g[1:] != g[:-1]
    starts = new.nonzero().flatten()
    ends = torch.cat([starts[1:], torch.tensor([n], device=dev)])
    lens = (ends - starts).long()
    G, L = int(lens.numel()), int(lens.max().item())

    flat_idx = torch.zeros((G, L), dtype=torch.long, device=dev)
    mask = torch.zeros((G, L), dtype=torch.bool, device=dev)
    for i in range(G):
        s0, l = int(starts[i]), int(lens[i])
        flat_idx[i, :l] = torch.arange(s0, s0 + l, device=dev)
        mask[i, :l] = True
    flat_idx = flat_idx.clamp(max=n - 1)
    zero = torch.zeros((), dtype=r.dtype, device=dev)
    rp = torch.where(mask, r[flat_idx], zero)
    vp = torch.where(mask, v[flat_idx], zero)

    adv = torch.zeros((G, L), dtype=r.dtype, device=dev)
    next_v = torch.zeros((G,), dtype=r.dtype, device=dev)
    next_a = torch.zeros((G,), dtype=r.dtype, device=dev)
    for k in range(L - 1, -1, -1):
        m = mask[:, k]
        has_next = (k + 1) < lens                      # 该行在组内是否还有后继
        delta = rp[:, k] + gamma * next_v - vp[:, k]
        a = delta + gamma * lam * torch.where(has_next, next_a, torch.zeros_like(next_a))
        a = torch.where(m, a, torch.zeros_like(a))
        adv[:, k] = a
        next_v = torch.where(m, vp[:, k], next_v)
        next_a = a

    out = torch.zeros(n, dtype=r.dtype, device=dev)
    out[flat_idx[mask]] = (adv + vp)[mask]
    return out.float()


def mc_targets_grouped(rewards: torch.Tensor, groups: torch.Tensor, *,
                       gamma: float = 0.99) -> torch.Tensor:
    """真轨迹上的 MC return-to-go（组内反向累加，无自举）。仅用于对比/自检。"""
    r = rewards.reshape(-1).double()
    g = groups.reshape(-1)
    n = int(r.numel())
    dev = r.device
    new = torch.ones(n, dtype=torch.bool, device=dev)
    new[1:] = g[1:] != g[:-1]
    starts = new.nonzero().flatten().tolist()
    out = torch.zeros(n, dtype=r.dtype, device=dev)
    acc = 0.0
    for i in range(n - 1, -1, -1):
        acc = float(r[i]) + gamma * acc
        out[i] = acc
        if i in starts and i > 0:
            acc = 0.0
    return out.float()
