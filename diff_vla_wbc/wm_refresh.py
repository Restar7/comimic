"""WM「保鲜」：用**真实转移**（HA 录制，非仿真器）微调世界模型，并锚定原始 WM 防漂移。

为什么要它：冻结 WM 在旧策略分布上训；token 一漂，想象就进外推区，奖励变成"模型的错"。
SuperMimic 的解法是每个 iter 用 replay buffer 的真转移重训 WM（`runner.py` 的 wm_loss）；
我们没有仿真器，但 `scripts/build_real_transitions.py` 从 HA 录制造了等价的真转移。

用法（训练循环里每 N 步调用一次）：

    data = torch.load("runs/wm_real/ha_transitions.pt")
    m = trainer.refresh_wm(data, steps=200)
    print(m)   # {"real_loss":..., "pos_mse":..., "qd_corr":...}

设计要点：
* **stop-gradient**：刷新只更新 WM 参数，不和策略的奖励图混在一起（策略回路里 WM 仍冻结）；
* **KL/L2 锚定**：`kl_anchor * Σ‖θ−θ0‖²`，θ0 是刷新前的权重 —— 防"用真转移把 WM 带偏"；
* 速度通道单独加权（`vel_weight`），因为它是历史短板（qd_corr 曾被训练打到 0.09）。
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .isaac_joint import default_xi, flat_state_tuple


@torch.no_grad()
def eval_wm_real(data: dict, wm, *, batch: int = 4096, vel_weight: float = 5.0,
                 device: str | torch.device = "cuda", seed: int = 0) -> dict:
    """在真转移上评一步预测质量（不更新）。"""
    dev = torch.device(device)
    g = torch.Generator(device="cpu"); g.manual_seed(int(seed))
    n = int(data["states"].shape[0])
    idx = torch.randperm(n, generator=g)[:batch]
    states = data["states"][idx].to(dev); actions = data["actions"][idx].to(dev)
    nxt = data["next_states"][idx].to(dev)
    xi = default_xi(states.shape[0], dev)
    S, S1 = flat_state_tuple(states), flat_state_tuple(nxt)
    p, R, v, w, q, qd = wm.step(S, actions, xi)
    qd_corr = float(((qd - qd.mean(0)) * (S1[5] - S1[5].mean(0))).mean()
                    / (qd.std() * S1[5].std() + 1e-9))
    return {
        "pos_mse": float((p - S1[0]).square().mean()),
        "quat_mse": float((R - S1[1]).square().mean()),
        "q_mse": float((q - S1[4]).square().mean()),
        "qd_corr": qd_corr,
        "qd_ratio": float(qd.abs().mean() / (S1[5].abs().mean() + 1e-9)),
    }


def refresh_wm(
    wm,
    data: dict,
    *,
    steps: int = 200,
    batch: int = 2048,
    lr: float = 1e-4,
    vel_weight: float = 5.0,
    kl_anchor: float = 1e-3,
    device: str | torch.device = "cuda",
    generator: torch.Generator | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """用真转移微调 WM（`steps` 个 minibatch），返回审计指标。"""
    dev = torch.device(device)
    states_all = data["states"]
    n = int(states_all.shape[0])
    if n < batch:
        raise ValueError(f"真转移只有 {n} 条，少于 batch={batch}")
    wm = wm.to(dev)
    # 刷新期间解冻（调用方在策略回路里是冻结的）；结束时会重新冻结
    wm.requires_grad_(True)
    theta0 = [p.detach().clone() for p in wm.parameters()]
    params = list(wm.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    g = generator or torch.Generator(device="cpu")
    was_training = wm.training
    wm.train()
    before = eval_wm_real(data, wm, vel_weight=vel_weight, device=dev) if steps > 0 else {}
    losses = []
    for it in range(int(steps)):
        idx = torch.randint(n, (batch,), generator=g, dtype=torch.long)
        states = data["states"][idx].to(dev)
        actions = data["actions"][idx].to(dev)
        nxt = data["next_states"][idx].to(dev)
        xi = default_xi(states.shape[0], dev)
        S, S1 = flat_state_tuple(states), flat_state_tuple(nxt)
        p, R, v, w, q, qd = wm.step(S, actions, xi)
        # 位置/朝向/关节（用各自的尺度归一，避免量纲压制）
        # 位置按量纲归一；朝向用 1−cos = (3−trace)/2（identity 时为 0，不是 −0.5）
        pos = (p - S1[0]).square().mean() / (S1[0].square().mean() + 1e-6)
        ori = ((3.0 - (R * S1[1]).sum(-1).sum(-1)) * 0.5).mean()
        qj = (q - S1[4]).square().mean()
        # 速度通道按数据 std 归一（否则 rad/s 的量纲会把它独自变成主项）
        sv = S1[2].std(0).mean().clamp_min(1e-6)
        sw = S1[3].std(0).mean().clamp_min(1e-6)
        sq = S1[5].std(0).mean().clamp_min(1e-6)
        vel = (((v - S1[2]).square().mean() / sv ** 2)
               + ((w - S1[3]).square().mean() / sw ** 2)
               + ((qd - S1[5]).square().mean() / sq ** 2))
        anchor = sum((pp - p0).square().sum() for pp, p0 in zip(params, theta0))
        loss = pos + ori + qj + vel_weight * vel + kl_anchor * anchor
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if (it + 1) % max(1, int(steps) // 4) == 0:
            log(f"  [wm-refresh {it+1}/{steps}] loss={losses[-1]:.5f} "
                f"(pos={float(pos):.5f} ori={float(ori):.5f} q={float(qj):.6f} vel={float(vel):.5f})")
    after = eval_wm_real(data, wm, vel_weight=vel_weight, device=dev)
    wm.requires_grad_(False)
    wm.eval() if not was_training else None
    out = {
        "steps": int(steps), "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "before": before, "after": after,
    }
    log("[AUDIT][wm-refresh] steps={} loss {:.4f}→{:.4f} | qd_corr {:.3f}→{:.3f} | pos {:.2e}→{:.2e}".format(
        out["steps"], out["loss_first"] or 0.0, out["loss_last"] or 0.0,
        before.get("qd_corr", float("nan")), after.get("qd_corr", float("nan")),
        before.get("pos_mse", float("nan")), after.get("pos_mse", float("nan"))))
    return out


__all__ = ["eval_wm_real", "refresh_wm"]
