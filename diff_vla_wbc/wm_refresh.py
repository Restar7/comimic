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


def _wm_step_flat(wm, states: torch.Tensor, actions: torch.Tensor, xi: torch.Tensor,
                  step_fn=None) -> torch.Tensor:
    """``(B,71) + a + xi -> (B,71)``，与后端无关。

    * mjlab/upstream ``WorldModel.step(s_flat, a, xi)`` 本来就是 flat in / flat out；
    * trackerlab 的 WM 吃 6 元组、吐 6 元组，由 bundle 提供的 ``step_fn`` 适配。
    """
    if step_fn is not None:
        return step_fn(states, actions, xi)
    return wm.step(states, actions, xi)


def _split_parts(flat: torch.Tensor):
    """扁平 71 → (p, quat, v, w, q, qd)，用**本仓状态布局**（非 trackerlab 元组序）。"""
    return (flat[..., 0:3], flat[..., 3:7], flat[..., 36:39],
            flat[..., 39:42], flat[..., 7:36], flat[..., 42:71])


@torch.no_grad()
def eval_wm_real(data: dict, wm, *, batch: int = 4096, vel_weight: float = 5.0,
                 device: str | torch.device = "cuda", seed: int = 0,
                 step_fn=None) -> dict:
    """在真转移上评一步预测质量（不更新）。"""
    dev = torch.device(device)
    g = torch.Generator(device="cpu"); g.manual_seed(int(seed))
    n = int(data["states"].shape[0])
    idx = torch.randperm(n, generator=g)[:batch]
    states = data["states"][idx].to(dev); actions = data["actions"][idx].to(dev)
    nxt = data["next_states"][idx].to(dev)
    xi = default_xi(states.shape[0], dev)
    s_next = _wm_step_flat(wm, states, actions, xi, step_fn)
    p, R, v, w, q, qd = _split_parts(s_next)
    S1 = _split_parts(nxt)
    qd_corr = float(((qd - qd.mean(0)) * (S1[5] - S1[5].mean(0))).mean()
                    / (qd.std() * S1[5].std() + 1e-9))
    return {
        "pos_mse": float((p - S1[0]).square().mean()),
        "quat_mse": float((R - S1[1]).square().mean()),
        "q_mse": float((q - S1[4]).square().mean()),
        "qd_corr": qd_corr,
        "qd_ratio": float(qd.abs().mean() / (S1[5].abs().mean() + 1e-9)),
    }



# ============ 原版 SuperMimic 的 WM 损失（2026-11-04 对齐）================
# 依据 `trackerlab-supermimic/source/trackerLab/trackerLab/algorithms/supermimic/runner.py`：
#
#     ERR_FLOOR = cat(3e-3 ×3[pos], 5e-3 ×3[rot], 2e-2 ×3[linvel], 5e-2 ×3[angvel],
#                     3e-3 ×J[q], 5e-2 ×J[qd])                      # 共 70 维
#     std = max(wm.out_norm.std(), ERR_FLOOR)
#     e   = (wm.state_error(S_hat, S) / std) ** 2
#     loss = e.mean()
#
# 为什么必须换成它（实测）：
#   我们之前自造的 `pos/(p².mean())` 把位置项压到 ~1e-6（|p|~1-2m ⇒ 分母 ~m²），
#   等于**完全没训位置** ⇒ 20 步 root 漂移想象 28.6cm vs 真实 6.8cm（4.18 倍），
#   与真实漂移的相关只有 0.205。原版逐维归一化后位置/朝向/关节/速度同量级，
#   而且和这个 WM 当初被训练时的口径一致（更可能复现原版质量）。
def ssm_err_floor(num_joints: int, device) -> torch.Tensor:
    return torch.cat((
        torch.full((3,), 3e-3), torch.full((3,), 5e-3),
        torch.full((3,), 2e-2), torch.full((3,), 5e-2),
        torch.full((int(num_joints),), 3e-3), torch.full((int(num_joints),), 5e-2),
    )).to(device)


def ssm_wm_loss_parts(wm, pred_flat: torch.Tensor, tgt_flat: torch.Tensor):
    """返回 ``(loss, parts)``，``parts`` 是**按块**归一化平方误差（pos/rot/linvel/angvel/q/qd）。

    为什么要分块打印：之前日志里的 `pos=0.00000 ori=0.00003` 是**旧口径的诊断量**
    （pos 用 |p|² 归一 ⇒ 恒为 1e-6），看那个会误以为位置没在训。真正在训的损失里，
    位置是被 `std/ERR_FLOOR` 归一化后一起优化的，必须按块看才知道它降没降。
    """
    from .isaac_joint import flat_state_tuple, NUM_JOINTS  # noqa: PLC0415
    std = torch.maximum(wm.out_norm.std().to(pred_flat.device),
                        ssm_err_floor(NUM_JOINTS, pred_flat.device))
    e = wm.state_error(flat_state_tuple(pred_flat), flat_state_tuple(tgt_flat))
    n = int(e.shape[-1])
    nj = int(NUM_JOINTS)
    blocks = {"pos": (0, 3), "rot": (3, 6), "linvel": (6, 9), "angvel": (9, 12),
              "q": (12, 12 + nj), "qd": (12 + nj, n)}
    sq = (e / std) ** 2
    parts = {k: float(sq[..., a:b].mean()) for k, (a, b) in blocks.items()}
    return sq.mean(), parts


def ssm_wm_loss(wm, pred_flat: torch.Tensor, tgt_flat: torch.Tensor) -> torch.Tensor:
    """原版口径的 WM 主损失（逐维归一化平方误差），``pred/tgt`` 都是扁平 (N,71)。"""
    from .isaac_joint import flat_state_tuple, NUM_JOINTS  # noqa: PLC0415
    std = torch.maximum(wm.out_norm.std().to(pred_flat.device),
                        ssm_err_floor(NUM_JOINTS, pred_flat.device))
    e = wm.state_error(flat_state_tuple(pred_flat), flat_state_tuple(tgt_flat))
    return ((e / std) ** 2).mean()


def increment_loss(pred: torch.Tensor, prev: torch.Tensor, tgt: torch.Tensor,
                   vel_weight: float = 1.0) -> torch.Tensor:
    """**动作可归因的增量损失**：``Δpred`` vs ``Δtrue``，按真值增量的逐维 std 归一（2026-11-04）。

    为什么必须单独加这一项
    ----------------------
    ``ssm_wm_loss`` 的分母是 ``max(out_norm.std(), ERR_FLOOR)``，位置那 3 维的 std 是 **~0.5 m**
    量级；而"动作"对一步位置的贡献只有**毫米级**。实测（``probe_wm_real_gain.py``）：
    σ=0.05 的保持型扰动在 WM 里只把终点推动 **0.08 cm**，真机是 **0.72 cm**（差 9 倍）——
    因为 8e-4 m 的效应按 0.5 m 归一化后在损失里只有 **2.6e-6**，等于没训。

    增量口径把尺度换成 ``std(一步增量)``（毫米~厘米），动作效应才落到损失能分辨的尺度上；
    而增量里**只有动作能解释**（``p'-p`` 的惯性部分由 v 给出，残差才是 a 的作用），
    所以这一项等于专门给 ∂s'/∂a 供梯度。
    """
    p, _R, v, w, q, qd = _split_parts(pred)
    p0, _R0, v0, w0, q0, qd0 = _split_parts(prev)
    T = _split_parts(tgt)

    def _n(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        s = b.detach().std(0).mean().clamp_min(1e-6)
        return (a - b).square().mean() / (s * s)

    return (_n(p - p0, T[0] - p0)
            + float(vel_weight) * (_n(v - v0, T[2] - v0) + _n(w - w0, T[3] - w0))
            + _n(q - q0, T[4] - q0)
            + _n(qd - qd0, T[5] - qd0))


def gain_loss(wm, step_fn, s: torch.Tensor, a: torch.Tensor, s1: torch.Tensor,
              xi_fn, block_weight=None) -> torch.Tensor:
    """**动作增益损失**：同一个状态、不同动作 ⇒ 预测的下一状态差必须等于真值差。

    为什么需要（2026-11-05 实测）
    ---------------------------
    A' 里"同状态、不同动作"的配对只有 **2 / 60,000**（状态最近邻 p50 = 4.86），
    所以 ``∂s'/∂a`` 完全没被数据约束：自由优化动作能把 20 步误差降 **89%**，
    但跨模型只转移 **8%** ⇒ 那是模型专属的伪方向，奖励反传等于优化假目标。

    配对探针（同一状态 × K 个不同动作）给出真实差的**直接观测**，这一项把
    "共享的动作分支权重"钉住。

    参数
    ----
    s   : (M, 71)   每个配对组的**同一个**状态
    a   : (M, K, 29) 组内 K 个不同动作
    s1  : (M, K, 71) 对应的 K 个真实下一状态
    step_fn : ``bundle.step``（吃 (B,71),(B,29),xi → (B,71)）
    """
    M, K = int(a.shape[0]), int(a.shape[1])
    s_rep = s[:, None, :].expand(M, K, s.shape[-1]).reshape(M * K, -1)
    pred = step_fn(s_rep, a.reshape(M * K, -1), xi_fn(M * K)).reshape(M, K, -1)
    d_pred = pred - pred[:, :1]              # (M,K,71)
    d_true = (s1 - s1[:, :1]).detach()       # (M,K,71)
    # 逐块归一化（位置/朝向/关节/速度量级差很大，不归一化会被 qd 淹没）
    n = d_pred.shape[-1]
    nj = (n - 12) // 2
    blocks = [(0, 3), (3, 7), (7, 36), (36, 39), (39, 42), (42, 71)]
    w = block_weight if block_weight is not None else [1.0] * len(blocks)
    tot = pred.new_zeros(())
    for wi, (i, j) in zip(w, blocks):
        if float(wi) <= 0.0:
            continue
        scale = d_true[..., i:j].std().clamp_min(1e-4)
        tot = tot + float(wi) * ((d_pred[..., i:j] - d_true[..., i:j]) / scale).pow(2).mean()
    return tot / max(1e-9, sum(float(x) for x in w))


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
    step_fn=None,
    mag_weight: float = 0.0,
) -> dict:
    """用真转移微调 WM（`steps` 个 minibatch），返回审计指标。

    ``mag_weight`` > 0 时额外加一项**幅值比正则**：直接压 ``mean|pred_qd| / mean|ref_qd|`` 到 1。
    为什么需要（2026-10-29）：MSE 主要惩罚"形状/逐元素误差"，对"整体高估 10%"几乎不敏感；
    而 pathwise 的 reward 恰好吃"预测得多快"，幅值高估会被 policy 利用。实测诊断显示
    高估是**全局**的（各 root_err 桶 ratio 1.09–1.13），不是尾部，所以必须显式正则。
    """
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
    scales = data.get("scales") if isinstance(data, dict) else None
    was_training = wm.training
    wm.train()
    before = eval_wm_real(data, wm, vel_weight=vel_weight, device=dev,
                          step_fn=step_fn) if steps > 0 else {}
    losses = []
    for it in range(int(steps)):
        idx = torch.randint(n, (batch,), generator=g, dtype=torch.long)
        states = data["states"][idx].to(dev)
        actions = data["actions"][idx].to(dev)
        nxt = data["next_states"][idx].to(dev)
        xi = default_xi(states.shape[0], dev)
        s_next = _wm_step_flat(wm, states, actions, xi, step_fn)
        p, R, v, w, q, qd = _split_parts(s_next)
        S1 = _split_parts(nxt)
        # 位置/朝向/关节（用各自的尺度归一，避免量纲压制）
        # 位置按量纲归一；朝向用 1−cos = (3−trace)/2（identity 时为 0，不是 −0.5）
        pos = (p - S1[0]).square().mean() / (S1[0].square().mean() + 1e-6)
                # ★★ 2026-11-04 **真 bug 修复**：原来写的是
                #   `((3.0 - (R * S1[1]).sum(-1).sum(-1)) * 0.5).mean()`
                #   —— `.sum(-1)` 写了两次，第二次把**整个 batch 求和成一个标量**，
                #   于是 ori 的量级 ≈ B 倍、梯度等价于放大 B 倍 lr：
                #     stage2(batch 8192) 日志里 `pos=0.00000 ori=-3492`（位置项被完全淹没）；
                #     stage2b(batch 512) 跑到 1000 步 loss 从 58 涨到 5938（发散）。
        _align = (R * S1[1]).sum(-1)
        ori = (1.0 - _align.pow(2)).mean()
        qj = (q - S1[4]).square().mean()
        # 速度通道按数据 std 归一（否则 rad/s 的量纲会把它独自变成主项）
        sv = (scales["v"].to(dev).mean() if scales is not None else S1[2].std(0).mean().clamp_min(1e-2))
        sw = (scales["w"].to(dev).mean() if scales is not None else S1[3].std(0).mean().clamp_min(1e-2))
        sq = (scales["qd"].to(dev).mean() if scales is not None else S1[5].std(0).mean().clamp_min(1e-2))
        vel = (((v - S1[2]).square().mean() / sv ** 2)
               + ((w - S1[3]).square().mean() / sw ** 2)
               + ((qd - S1[5]).square().mean() / sq ** 2))
        anchor = sum((pp - p0).square().sum() for pp, p0 in zip(params, theta0))
        # ★ 2026-11-04：主损失换成**原版 SuperMimic 口径**（逐维归一化平方误差）。
        #   旧口径把 pos 归一化到 1e-6（等于不训位置）⇒ 20 步 root 漂移 4.18 倍。
        _ssm, _parts = ssm_wm_loss_parts(wm, s_next, nxt)
        loss = _ssm + kl_anchor * anchor
        if mag_weight > 0:
            # 幅值比正则：**只压 qd**（关节速度）。
            # 2026-10-29 消融发现：三个通道一起压会误伤 w（root 角速度）—— base head + 新 WM 时
            # root 角速度从 0.072 掉到 0.056、root 朝向从 4.86° 恶化到 11.24°（+131%）。
            # 门的判据只用 qd_ratio，所以只压 qd 就够，v/w 不碰。
            mp = qd.abs().mean()
            mr = S1[5].abs().mean().clamp_min(1e-6)
            loss = loss + mag_weight * ((mp - mr) / mr).square()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if (it + 1) % max(1, int(steps) // 4) == 0:
            log(f"  [wm-refresh {it+1}/{steps}] loss={losses[-1]:.5f} "
                f"(SSM块: " + " ".join(f"{k}={v:.4f}" for k, v in _parts.items()) + ")")
    after = eval_wm_real(data, wm, vel_weight=vel_weight, device=dev, step_fn=step_fn)
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


def train_multistep(
    wm,
    S: "torch.Tensor",
    A: "torch.Tensor",
    *,
    steps: int = 600,
    batch: int = 64,
    lr: float = 3e-5,
    k: int | None = None,
    vel_weight: float = 5.0,
    mag_weight: float = 30.0,
    inc_weight: float = 0.0,
    kl_anchor: float = 0.0,
    device: Any = "cuda",
    step_fn: Callable | None = None,
    xi_fn: Callable | None = None,
    sim_data: dict | None = None,
    sim_weight: float = 1.0,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> dict:
    """**k 步自回归**微调：每步都喂模型自己上一步的输出，逐 k 对齐真值。

    为什么单步不够（2026-10-29 实测）：``refresh_wm`` 的 ``s_next = wm(state, action)``
    里 state 永远是**真实**状态，模型从未见过自己的输出 ⇒ 累积误差不受任何约束。
    实测 k=1 的 qd_ratio 0.977（门禁只量这一步，永远 PASS），而 k=20 是 1.329。
    从 B 线连续窗口滚 8 步训练 400 步（CPU 40 s）后，held-out 上 k=20 的
    qd_ratio 1.279→1.095、qd_corr 0.771→0.817、joint_mae 1.14°→0.80°，三项同时变好。

    ``S``: (N,k+1,71) 连续真值；``A``: (N,k,29) 真实动作。参数与 ``refresh_wm`` 同构。
    """
    from .isaac_joint import default_xi  # noqa: PLC0415

    dev = torch.device(device)
    n_all, K = int(S.shape[0]), int(A.shape[1])
    K = min(K, int(k) if k else K)
    # 与 refresh_wm 一致：刷新期间解冻（策略回路里它是冻结的），调用方负责之后重新冻结
    wm.requires_grad_(True)
    params = list(wm.parameters())
    if not params:
        raise RuntimeError("wm 没有参数")
    theta0 = [p.detach().clone() for p in params]
    opt = torch.optim.Adam(params, lr=float(lr))
    g = torch.Generator().manual_seed(int(seed))
    m = mag_weight > 0
    wm.train()
    run = []
    for _ in range(int(steps)):
        idx = torch.randint(n_all, (int(batch),), generator=g, dtype=torch.long)
        s = S[idx, 0].to(dev)
        loss = torch.zeros((), device=dev)
        for t in range(K):
            xi = xi_fn(s.shape[0]) if xi_fn is not None else default_xi(s.shape[0], dev)
            s_prev = s
            s = _wm_step_flat(wm, s, A[idx, t].to(dev), xi, step_fn)
            tgt_t = S[idx, t + 1].to(dev)
            l, mag, _, _ = _step_losses(s, tgt_t, float(vel_weight), wm=wm)
            loss = loss + l + (float(mag_weight) * mag if m else 0.0)
            if inc_weight > 0:
                # 动作可归因增量项：把 ∂s'/∂a 的梯度放到 mm 尺度上（见 increment_loss 注释）
                loss = loss + float(inc_weight) * increment_loss(s, s_prev, tgt_t)
        loss = loss / K
        # **交织**单步 sim 项：只跑多步会把 sim 单步门打坏（实测 600 步 lr 3e-5 就让
        # sim corr 0.9896→0.9356、ratio 0.995→1.041，直接 exit 2）。每步都掺一条
        # 真实场景单步样本，等价于"多步学习 + 单步保鲜"，两个分布一起守。
        if sim_data is not None:
            ns = int(sim_data["states"].shape[0])
            j = torch.randint(ns, (int(batch),), generator=g, dtype=torch.long)
            s0 = sim_data["states"][j].to(dev)
            xi_s = xi_fn(s0.shape[0]) if xi_fn is not None else default_xi(s0.shape[0], dev)
            pred0 = _wm_step_flat(wm, s0, sim_data["actions"][j].to(dev), xi_s, step_fn)
            l0, mag0, _, _ = _step_losses(pred0, sim_data["next_states"][j].to(dev),
                                          float(vel_weight), wm=wm)
            loss = loss + float(sim_weight) * (l0 + (float(mag_weight) * mag0 if m else 0.0))
        if float(kl_anchor) > 0:
            loss = loss + float(kl_anchor) * sum(
                (p - p0).square().sum() for p, p0 in zip(params, theta0))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        run.append(float(loss.detach()))
        if log is not None and steps and (len(run) % max(1, int(steps) // 6) == 0):
            log(f"  [wm-multistep {len(run)}/{int(steps)}] loss={run[-1]:.4f} "
                f"(k={K})")
    wm.eval()
    return {"k": K, "steps": int(steps), "loss_last": run[-1] if run else 0.0,
            "loss_first": run[0] if run else 0.0}


def _step_losses(pred, tgt, vel_weight: float = 5.0, wm=None):
    """WM 单步损失。**给了 wm 就用原版 SuperMimic 口径**（逐维归一化平方误差）。

    返回 ``(loss, mag, q_abs_err, (qd, qd_tgt))``，后两项只用于诊断。
    """
    p, R, v, w, q, qd = _split_parts(pred)
    S1 = _split_parts(tgt)
    pos = (p - S1[0]).square().mean() / (S1[0].square().mean() + 1e-6)
            # ★★ 2026-11-04 **真 bug 修复**：原来写的是
            #   `((3.0 - (R * S1[1]).sum(-1).sum(-1)) * 0.5).mean()`
            #   —— `.sum(-1)` 写了两次，第二次把**整个 batch 求和成一个标量**，
            #   于是 ori 的量级 ≈ B 倍、梯度等价于放大 B 倍 lr：
            #     stage2(batch 8192) 日志里 `pos=0.00000 ori=-3492`（位置项被完全淹没）；
            #     stage2b(batch 512) 跑到 1000 步 loss 从 58 涨到 5938（发散）。
    _align = (R * S1[1]).sum(-1)
    ori = (1.0 - _align.pow(2)).mean()
    qj = (q - S1[4]).square().mean()
    sv = S1[2].std(0).mean().clamp_min(1e-2)
    sw = S1[3].std(0).mean().clamp_min(1e-2)
    sq = S1[5].std(0).mean().clamp_min(1e-2)
    vel = (((v - S1[2]).square().mean() / sv ** 2)
           + ((w - S1[3]).square().mean() / sw ** 2)
           + ((qd - S1[5]).square().mean() / sq ** 2))
    mag = ((qd.abs().mean() - S1[5].abs().mean().clamp_min(1e-6))
           / S1[5].abs().mean().clamp_min(1e-6)).square()
    if wm is not None:
        main = ssm_wm_loss(wm, pred, tgt)            # ★ 原版口径（位置/朝向/关节/速度同量级）
    else:
        main = pos + ori + qj + vel_weight * vel     # 旧口径（保留做对照）
    return main, mag, (q - S1[4]).abs(), (qd, S1[5])
