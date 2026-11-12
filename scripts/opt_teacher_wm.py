#!/usr/bin/env python3
"""**WM 内受约束的轨迹优化教师**（摊销式轨迹优化 / Amortized Trajectory Optimization）2026-11-07。

和旧教师的区别
--------------
* 旧：`Δ = η·unit(∇J₂₀)` —— **一次**反传得到的**瞬时梯度方向**，幅值统一 η。
* 新：把 `U ∈ R^{T×64}` 当成**每个窗口自己的可训练变量**，在冻结的
  `token → SONIC → WM → reward` 可微链上做 **N 步**约束优化，取 `Δ* = NormCap(U*)`。
  幅值由优化 + 约束决定，不再是统一 η。

优化目标::

    min_U  -J₂₀^WM(Z + Δ(U)) + λ_lim·L_lim + λ_peak·L_peak + λ_rate·L_rate + λ_trust·L_trust

其中 `L_lim` 用 `bundle.mdp.kin` 的真实关节限位（逐关节非对称），
`L_peak/L_rate` 是**动作空间**的峰值与变化率预算（真机探针用的是 δ=0.03 rad RMS），
`L_trust` 是 token 空间的 L2 预算。

为什么要有 `--iters 0` 这一档
---------------------------
`iters=0` 就是旧教师（单次梯度、单位化到 η）。**同一批窗口、同一个 WM、同一个 reward** 下和新教师对比，
才能回答"多轮优化到底有没有更好"，而不是跨实验比。

⚠️ 已知风险（本脚本要量出来）
----------------------------
WM 的 `J` 幅值高估 1.79×（中位）/2.17×（均值），`ΔJ` 排序相关只有 0.092。
**"在 WM 里最大化 J" 会主动去找 WM 错得最厉害的方向**（旧 BPTT 的"薅 WM"就是这个机制）。
所以本脚本除了 WM 收益，还输出三组"作弊指纹"：

1. `act_rms / budget`：动作是否顶满预算；
2. `peak / rate`：是否出现尖峰/抖动；
3. `gain_per_rad`：每 rad 动作换来的 WM 收益 —— 旧教师是已知可比的基准，
   新教师如果**收益涨得比动作幅度快得多**，就是利用了模型误差而不是真的更好。

真实裁判只能是 Isaac 上 32 个窗口的配对（`scripts/isaac_probe_action_rollout.py`）。

用法::

    python3 scripts/opt_teacher_wm.py --dump "runs/harness_02_teacher/dump_h20/teacher_*.npz" \
        --windows 256 --iters 0 5 10 20 --out runs/teacher_opt_pilot
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(ROOT / "runs/harness_02_teacher/dump_h20/teacher_*.npz"))
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--iters", nargs="+", type=int, default=[0, 5, 10, 20])
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--token-cap", type=float, default=0.08)
    ap.add_argument("--eta", type=float, default=0.15, help="iters=0（旧教师）的固定幅值")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--act-budget", type=float, default=0.03,
                    help="动作 RMS 预算：真机真值探针就是在这个动作空间用 δ=0.03 验证的")
    ap.add_argument("--rate-budget", type=float, default=0.06, help="逐步动作变化预算（rad）")
    ap.add_argument("--lam-lim", type=float, default=1.0)
    ap.add_argument("--lam-peak", type=float, default=5.0)
    ap.add_argument("--lam-rate", type=float, default=2.0)
    ap.add_argument("--lam-trust", type=float, default=1.0)
    ap.add_argument("--lam-feas", type=float, default=10.0, help="状态可行性权重（0=关）")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/teacher_opt_pilot")
    a = ap.parse_args()

    import numpy as np
    import torch

    from diff_vla_wbc import teacher_probe as tp
    from diff_vla_wbc.isaac_joint import FrameRef, load_isaac_bundle
    from diff_vla_wbc.rollout import discounted_return

    dev = torch.device(a.device)
    files = sorted(glob.glob(a.dump))
    if not files:
        raise SystemExit(f"没有匹配 {a.dump}")
    buf: dict[str, list] = {}
    for f in files:
        d = np.load(f)
        for k in ("token_chunk", "state", "xi", "prev_action", "history", "obs_hist",
                  "frames", "clip_ids", "step"):
            buf.setdefault(k, []).append(d[k])
    ax = 1 if len(np.shape(buf["history"][0])) == 3 and np.shape(buf["history"][0])[0] == 10 else 0
    D = {k: np.concatenate(v, axis=(1 if k in ("history", "frames", "clip_ids") else 0))
         for k, v in buf.items() if k != "step"}
    N = min(int(D["token_chunk"].shape[0]), int(a.windows))
    K = int(D["frames"].shape[0])
    print(f"[data] {len(files)} dumps → 用 {N} 个窗口（共 {D['token_chunk'].shape[0]}），K={K}")

    bundle = load_isaac_bundle(a.ckpt, device=dev,
                               npz=sorted((ROOT / "data/npz/ha_tokens").glob("*.npz")),
                               packed_motion=ROOT / "runs/isaac_motion/ha_cmd/packed.pt",
                               obs_mode=None, hist=None, token_dim=64)
    kin = bundle.mdp.kin
    q_lo = kin.q_lo.detach().to(dev)
    q_hi = kin.q_hi.detach().to(dev)
    # ★ 实测 `bundle.action_scale` 全 1.0 ⇒ 动作就是"相对齐次位姿的关节角"，
    #   env 口径：`q_target = default_joint_pos + xi_bias + a * action_scale`（见 isaac_joint.py:754）。
    default_q = bundle.default_joint_pos.detach().to(dev)
    a_scale = bundle.action_scale.detach().to(dev)
    q_lo, q_hi = q_lo.to(dev), q_hi.to(dev)
    # ★ MJCF 里**无限制**的关节 `jnt_range=(0,0)`，代码默认填 0/0 ⇒ 直接拿去当限位会把
    #   所有非零角度都判成越界（实测 base 自身"越界"0.31 rad 就是这么来的假信号）。只保留真限位。
    _lim = (q_hi - q_lo) > 1e-6
    q_lo = torch.where(_lim, q_lo, torch.full_like(q_lo, -1e3))
    q_hi = torch.where(_lim, q_hi, torch.full_like(q_hi, 1e3))
    print(f"[limits] 真限位关节 {int(_lim.sum())}/{len(_lim)}；其余按无限处理")

    def joint_target(act, xi_b):
        """act (H,B,29) → 关节目标 (H,B,29)，用 env 同款公式。"""
        # xi_b (B,29) → (1,B,29)；act (H,B,29)
        return (default_q.view(1, 1, -1) + xi_b.unsqueeze(0) + act * a_scale.view(1, 1, -1))

    tok_all = torch.as_tensor(D["token_chunk"], dtype=torch.float32)
    st = torch.as_tensor(D["state"], dtype=torch.float32)
    xi = torch.as_tensor(D["xi"], dtype=torch.float32)
    prev = torch.as_tensor(D["prev_action"], dtype=torch.float32)
    hist = torch.as_tensor(D["history"], dtype=torch.float32)
    oh = torch.as_tensor(D["obs_hist"], dtype=torch.float32)
    fr = torch.as_tensor(D["frames"], dtype=torch.long)

    def _mask(u):
        m = torch.zeros_like(u)
        m[:, :a.horizon] = 1.0
        return m

    from diff_vla_wbc.delta_token import apply_norm_cap   # ★ 与部署逐字同一个限幅

    def cap(u):
        """★ 与部署/学生**同一个** L2 软限幅（`apply_norm_cap`），t≥horizon 置零。

        旧实现是逐元素 `cap*tanh(u)`，最大 ‖Δ‖ 可达 cap*√64 = 0.64，是学生的 8 倍 ——
        教师因此会优化到学生根本表达不了的集合里（2026-11-07 契约修复）。

        为什么不用"按范数归一化"：`u*(c/‖u‖)` 在 ‖u‖=0 处的局部导数是 c/1e-12 ≈ 8e10，
        反传时会把梯度打爆（实测让整个优化回退到零）。tanh 参数化没有除零。
        """
        return apply_norm_cap(u, float(a.token_cap)) * _mask(u)

    def normcap(d):
        """旧教师的 NormCap（低于 cap 时**不变**，只有超了才按范数缩）。"""
        n = d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        c = float(a.token_cap)
        return torch.where(n > c, d * (c / n), d) * _mask(d)

    rows: list[dict] = []
    dumps: dict[int, list] = {n: [] for n in a.iters}
    for s in range(0, N, a.batch):
        e = min(N, s + a.batch)
        b = e - s
        refs = [FrameRef(fr[t, s:e].to(dev)) for t in range(K)]
        base = tok_all[s:e].to(dev)
        ctx = dict(state=st[s:e].to(dev), references=refs, xi=xi[s:e].to(dev),
                   prev_action=prev[s:e].to(dev), history=hist[:, s:e].to(dev), obs_hist=None)

        # ---- 基线 ----
        with torch.no_grad():
            r0 = tp._rollout(bundle, base, horizon=a.horizon, **ctx)
            J0 = discounted_return(r0, a.gamma)
            _sf = bool(torch.isfinite(r0.states).all())
            a0 = r0.actions.detach()                       # (H,B,29)
            z0 = r0.states[0][:, 2].detach()                # 起点骨盆高度
        a0_rms = float(a0.pow(2).mean().sqrt())
        xi_b = ctx["xi"][:, 1:1 + 29]
        with torch.no_grad():
            q0 = joint_target(a0, xi_b)
            viol0 = float(((torch.relu(q_lo.view(1, 1, -1) - q0).sum()
                            + torch.relu(q0 - q_hi.view(1, 1, -1)).sum())
                           / q0.numel()))

        # ---- 单次梯度教师（iters=0，旧口径）----
        # ★ `tokenwise_reward_grad().g` 是 **dJ/dtoken 对 T 求和后的 (B,64)**
        #   （= 与共享 ε 逐位一致的窗口级梯度）。旧教师是"共享 Δ"，所以广播到 T 帧。
        g = tp.tokenwise_reward_grad(bundle, base, horizon=a.horizon, gamma=a.gamma,
                                     reward_steps=0, **ctx).g          # (B,64)
        gu = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        d_old = normcap(torch.as_tensor(a.eta, device=dev) * gu[:, None, :].expand(-1, base.shape[1], -1))
        with torch.no_grad():
            r_old = tp._rollout(bundle, base + d_old, horizon=a.horizon, **ctx)
            J_old = discounted_return(r_old, a.gamma)
            a_old = r_old.actions.detach()

        for n_it in a.iters:
            if n_it == 0:
                d = d_old.detach()
                Jn, an = J_old, a_old
                gain = Jn - J0
            else:
                # U 覆盖整段 chunk（T=40），但 cap() 会把 t≥horizon 的帧置零
                U = torch.zeros(b, base.shape[1], 64, device=dev, requires_grad=True)
                opt = torch.optim.Adam([U], lr=a.lr)
                _lr = float(a.lr)
                _best_U, _best_J = U.detach().clone(), None
                for _ in range(int(n_it)):
                    d = cap(U)
                    roll = tp._rollout(bundle, base + d, horizon=a.horizon, **ctx)
                    J = discounted_return(roll, a.gamma)
                    act = roll.actions                               # (H,B,29)
                    # ★ 关键更正（2026-11-07）：**动作不是关节偏置**。
                    #   实测 dump 里的 `prev_action` RMS=1.38 / max=5.3，而参考 q 推出的
                    #   关节偏置 RMS=0.24 ⇒ 两者差 5.7×（`action_scale` 是 1.0 还是 spec 都对不上）。
                    #   所以"逐关节限位"这一层在动作空间里无法正确施加。
                    #   改用同量纲、且**已被真机验证过**的约束：
                    #   ① 动作空间信任域（真值探针就是在这个空间里用 δ=0.03 验证的）
                    #   ② 状态可行性（骨盆高度不塌、不倾覆），直接读 rollout 的状态
                    L_lim = torch.zeros((), device=dev)
                    # ★ 用**均方**比较，不要用 sqrt(RMS)：`x.pow(2).mean().sqrt()` 在 x=0 处
                    #   导数无穷，而外层 `relu(·-budget)` 在预算内梯度为 0 ⇒ 0·∞ = NaN，
                    #   实测让第一次 Adam 步的梯度就变 NaN、整个窗口回退到零修正。
                    dev_ms = (act - a0).pow(2).mean()
                    L_peak = torch.relu(dev_ms - float(a.act_budget) ** 2).pow(2)
                    diff_ms = (act[1:] - act[:-1]).pow(2).mean()
                    L_rate = torch.relu(diff_ms - float(a.rate_budget) ** 2).pow(2)
                    L_trust = d.pow(2).mean()
                    # 状态可行性：骨盆高度相对起点塌陷/抬升、躯干倾覆（R22 = cos(tilt)）
                    st_ = roll.states[1:]
                    # WM 的状态通道可能出 inf/nan（只有奖励通道被检查过）；
                    # 直接算 relu 会污染整条 loss ⇒ 先做有界替换（发散了就当最差）。
                    st_ = torch.nan_to_num(st_, nan=0.0, posinf=1e3, neginf=-1e3)
                    z_ = st_[..., 2]
                    dz = z_ - z0.view(1, -1)
                    L_h = (torch.relu(-dz - 0.10).pow(2) + torch.relu(dz - 0.10).pow(2)).mean()
                    qt = st_[..., 3:7]
                    R22 = 1.0 - 2.0 * (qt[..., 1] ** 2 + qt[..., 2] ** 2)
                    L_tilt = torch.relu(0.90 - R22).pow(2).mean()
                    L_feas = float(a.lam_feas) * (L_h + L_tilt)
                    loss = (-J.mean() + float(a.lam_lim) * L_lim
                            + float(a.lam_peak) * L_peak + float(a.lam_rate) * L_rate
                            + float(a.lam_trust) * L_trust + L_feas)
                    opt.zero_grad(set_to_none=True)
                    if not torch.isfinite(loss):
                        # 回退到最近一次有限解，并缩小步长（line-search）
                        with torch.no_grad():
                            U.data.copy_(_best_U)
                        _lr = max(_lr * 0.5, 1e-4)
                        for g in opt.param_groups:
                            g["lr"] = _lr
                        if _lr <= 1e-4:
                            break
                        continue
                    loss.backward()
                    _gn = torch.nn.utils.clip_grad_norm_([U], 1.0)
                    if not torch.isfinite(_gn):
                        with torch.no_grad():
                            U.data.copy_(_best_U)
                        _lr = max(_lr * 0.5, 1e-4)
                        for g in opt.param_groups:
                            g["lr"] = _lr
                        if _lr <= 1e-4:
                            break
                        continue
                    with torch.no_grad():
                        _prev = U.detach().clone()
                    opt.step()
                    with torch.no_grad():
                        U.clamp_(-0.08, 0.08)         # 逐元素硬盒，防跑飞
                        if not torch.isfinite(U).all():
                            U.data.copy_(_best_U)
                        else:
                            _best_U = U.detach().clone()
                with torch.no_grad():
                    d = normcap(cap(U)).detach()
                    roll = tp._rollout(bundle, base + d, horizon=a.horizon, **ctx)
                    Jn = discounted_return(roll, a.gamma)
                    an = roll.actions.detach()
                    gain = Jn - J0
            dumps[n_it].append(d.detach().cpu().float())

            with torch.no_grad():
                act_dev = (an - a0)                                       # (H,B,29)
                act_rms = act_dev.pow(2).mean().sqrt().item()
                peak = act_dev.abs().amax(dim=(0, 2)).mean().item()
                rate = (an[1:] - an[:-1]).pow(2).mean().sqrt().item()
                qn = joint_target(an, xi_b)
                viol = float((torch.relu(q_lo.view(1, 1, -1) - qn).sum()
                              + torch.relu(qn - q_hi.view(1, 1, -1)).sum()) / qn.numel())
                tok_rms = d.pow(2).mean().sqrt().item()
                dJ = gain.cpu()
            rows.append({
                "iters": int(n_it), "s": s, "n": b, "viol_base": viol0,
                "J0": float(J0.mean()), "J": float(Jn.mean()),
                "dJ_mean": float(dJ.mean()), "dJ_p50": float(dJ.median()),
                "dJ_pos_frac": float((dJ > 0).float().mean()),
                "act_rms": act_rms, "act_rms_over_budget": act_rms / max(a.act_budget, 1e-9),
                "peak": peak, "rate": rate, "limit_viol": viol, "tok_rms": tok_rms,
                "a0_rms": a0_rms,
                "gain_per_rad": float(dJ.mean()) / max(act_rms, 1e-9),
            })
        print(f"[opt] {e}/{N}  J0={float(J0.mean()):.4f} states_finite={_sf}  "
              + "  ".join(f"n{r['iters']}:dJ{r['dJ_mean']:+.4f}/a{r['act_rms']:.3f}"
                          for r in rows[-len(a.iters):]), flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    for n_it, lst in dumps.items():
        if lst:
            np.savez_compressed(a.out / f"delta_iters{n_it}.npz", delta=torch.cat(lst).numpy())
    (a.out / "teacher_opt_pilot.json").write_text(json.dumps(rows, indent=1))

    print("\n" + "=" * 104)
    print("iters   J0      J       dJ均值   dJ>0比例  act_rms  /预算   peak    rate   限位越界  ‖Δtok‖  gain/rad")
    print("-" * 104)
    for n_it in a.iters:
        sel = [r for r in rows if r["iters"] == n_it]
        m = lambda k: float(np.mean([r[k] for r in sel]))
        print(f"{n_it:>5}{m('J0'):>8.4f}{m('J'):>8.4f}{m('dJ_mean'):>9.4f}"
              f"{m('dJ_pos_frac'):>10.2f}{m('act_rms'):>9.4f}{m('act_rms_over_budget'):>8.2f}"
              f"{m('peak'):>8.3f}{m('rate'):>8.3f}{m('limit_viol'):>10.1f}"
              f"{m('tok_rms'):>8.4f}{m('gain_per_rad'):>10.3f}")
        print(f"{'':>5}（base 自身限位越界={m('viol_base'):.4f} rad·关节⁻¹；动作=关节角，scale 全 1.0）")
    print("\n判读：`dJ` 涨但 `act_rms` 顶满预算 + `gain/rad` 远高于 iters=0 → 疑似「薅 WM」；")
    print("      真裁判是 Isaac 上 32 窗口的配对（Δ* vs 0），不是这里的 J。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
