#!/usr/bin/env python3
"""**因果门 / 天花板检查**：在 WM 里自由优化 20 步动作，看还有多少"假空间"。

判据（2026-11-05 定论）
----------------------
如果**连不受约束的动作优化**（直接对动作梯度下降，没有策略网络、没有 trust、没有 token 瓶颈）
都能把 20 步跟踪误差砍掉一大半，说明 WM 里的 `a` 是一个**能吸收自身预测误差的旋钮**，
而不是因果控制输入。用这种 surrogate 反传 = 优化假目标 ⇒ 真机越训越差。

    旧 WM（A' leash，perturb_hold=20）：headroom **92%** ⇒ 不及格
    新数据（perturb_hold=1，白噪声）：期望 **≤20%**

用法（门禁）::

    python3 scripts/probe_wm_headroom.py --ckpt <wm.pt> --gate     # 过门 exit 0，否则 exit 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wbc-ckpt", type=Path, default=ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--windows", type=Path,
                    default=ROOT / "runs/a_prime_sonic_leash/roll_windows.pt")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--bounds", type=float, nargs="*", default=[0.02, 0.05, 0.10, 0.20, 0.40],
                    help="受限 headroom 扫：把**平均每步动作改动**限制在这些幅度内。"
                         "0.05 ≈ 我们策略实际能改动的量级（round-4 act/rmse≈0.05），"
                         "0.9 ≈ 数据里相邻帧动作变化的量级 ⇒ 这个扫法比单一阈值有意义")
    ap.add_argument("--gate-bound", type=float, default=0.10,
                    help="门禁看**这个动作幅度内**的 headroom")
    ap.add_argument("--max-headroom", type=float, default=0.20,
                    help="门禁阈值：自由优化最多只能再降这么多（默认 20%%）")
    ap.add_argument("--gate", action="store_true", help="启用门禁退出码")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415

    dev = torch.device(a.device)
    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    b = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=npz,
                          packed_motion=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
                          obs_mode=None, hist=None, token_dim=64)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    b.wm.load_state_dict(ck.get("wm", ck), strict=True)
    b.wm.requires_grad_(False)
    b.wm.eval()
    print(f"[head] WM = {a.ckpt}", flush=True)

    w = torch.load(a.windows, map_location="cpu", weights_only=False)
    S, A = w["states"].float(), w["actions"].float()
    K = int(a.k)
    g = torch.Generator().manual_seed(0)
    sel = torch.randperm(int(S.shape[0]), generator=g)[: a.n]
    s0 = S[sel, 0].to(dev)
    acts0 = A[sel, :K].to(dev)
    tgt = S[sel, 1:K + 1].to(dev)
    xi = b.default_xi(s0.shape[0], dev)

    def rollout(seq: torch.Tensor) -> torch.Tensor:
        s = s0
        traj = []
        for t in range(K):
            out = b.step(s, seq[:, t], xi)
            s = out[0] if isinstance(out, (tuple, list)) else out
            traj.append(s)
        return torch.stack(traj, 1)

    def track_err(traj: torch.Tensor):
        dp = (traj[..., 0:3] - tgt[..., 0:3]).norm(dim=-1).mean(1)
        dq = (traj[..., 7:36] - tgt[..., 7:36]).abs().mean(dim=(1, 2))
        return dp, dq

    def med(seq: torch.Tensor):
        with torch.no_grad():
            dp, dq = track_err(rollout(seq))
        return float(dp.median()) * 100, float(dq.median())

    with torch.no_grad():
        base_p, base_q = track_err(rollout(acts0))
    base_p_cm, base_q_r = float(base_p.median()) * 100, float(base_q.median())
    print(f"[head] baseline（录制动作）: 位置 p50 {base_p_cm:.2f} cm  关节 p50 {base_q_r:.4f} rad",
          flush=True)

    acts = acts0.clone().requires_grad_(True)
    opt = torch.optim.Adam([acts], lr=float(a.lr))
    for it in range(int(a.iters)):
        dp, dq = track_err(rollout(acts))
        loss = dp.mean() + 0.5 * dq.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % max(1, int(a.iters) // 3) == 0 or it == int(a.iters) - 1:
            print(f"  [it {it:3d}] 位置 p50 {float(dp.median())*100:.2f} cm  "
                  f"关节 p50 {float(dq.median()):.4f}", flush=True)

    with torch.no_grad():
        free = acts.detach()
        opt_p, opt_q = med(free)
        headroom_p = 1.0 - opt_p / max(base_p_cm, 1e-9)
        headroom_q = 1.0 - opt_q / max(base_q_r, 1e-9)
        delta = free - acts0
        d = delta.reshape(-1, 29)
        m, sd = d.mean(0), d.std(0)
        print()
        print(f"[head] 自由优化后: 位置 p50 {opt_p:.2f} cm  关节 p50 {opt_q:.4f} rad")
        print(f"[head] ★ headroom 位置 {headroom_p*100:+.1f}%  关节 {headroom_q*100:+.1f}%")
        print(f"[head] 最优 Δa: |mean| {float(m.abs().mean()):.4f}  |std| {float(sd.mean()):.4f}"
              f"  （mean/std {float(m.abs().mean())/max(float(sd.mean()),1e-9):.2f}；"
              f"≈0 ⇒ 逐样本伪方向，不是可校准偏置）")
        tr_o = rollout(free)
        tr_b = rollout(acts0)
        for nm, tr in (("录制动作", tr_b), ("优化动作", tr_o)):
            print(f"[head] {nm}: 预测 z p50 {float(tr[..., 2].median()):.3f} m  "
                  f"|v| p50 {float(tr[..., 36:39].norm(dim=-1).median()):.3f} m/s  "
                  f"|qd| p50 {float(tr[..., 42:71].norm(dim=-1).median()):.2f} rad/s")
        zr = S[sel, 1:K + 1].to(dev)
        print(f"[head] 真实数据: z p50 {float(zr[..., 2].median()):.3f} m  "
              f"|v| p50 {float(zr[..., 36:39].norm(dim=-1).median()):.3f}  "
              f"|qd| p50 {float(zr[..., 42:71].norm(dim=-1).median()):.2f}")

    # ---- 受限 headroom 扫（动作改动封顶），这才是"策略真能拿到的空间" ----
    bounded = []
    for _bd in list(a.bounds or []):   # 注意：不能叫 b —— 会把 bundle 变量遮住
        acts2 = acts0.clone().requires_grad_(True)
        opt2 = torch.optim.Adam([acts2], lr=float(a.lr))
        for it in range(int(a.iters)):
            dp, dq = track_err(rollout(acts2))
            loss = dp.mean() + 0.5 * dq.mean()
            opt2.zero_grad(set_to_none=True)
            loss.backward()
            opt2.step()
            with torch.no_grad():
                dlt = acts2 - acts0
                m = dlt.abs().mean()
                if float(m) > float(_bd):
                    acts2.copy_(acts0 + dlt * (float(_bd) / float(m)))
        p_b, _q_b = med(acts2.detach())
        hr = 1.0 - p_b / max(base_p_cm, 1e-9)
        bounded.append((float(_bd), p_b, hr))
        print(f"[head] 受限 |Δa|≤{_bd:.2f}: 位置 p50 {p_b:.3f} cm ⇒ headroom {hr*100:+.1f}%", flush=True)
    hr_gate = None
    for b, _p, hr in bounded:
        if abs(b - float(a.gate_bound)) < 1e-9:
            hr_gate = hr
    if hr_gate is None and bounded:
        hr_gate = bounded[-1][2]
    print(f"[head] 门禁口径：自由优化 headroom {headroom_p*100:.1f}%，"
          f"受限(|Δa|≤{a.gate_bound}) headroom {(hr_gate or 0)*100:.1f}%")

    ok = headroom_p <= float(a.max_headroom) + 1e-9
    if a.gate:
        print(f"[head] {'✅ PASS' if ok else '❌ FAIL'}（自由优化口径）：headroom {headroom_p*100:.1f}% "
              f"{'≤' if ok else '>'} 阈值 {a.max_headroom*100:.0f}%；"
              f"受限口径 {(hr_gate or 0)*100:.1f}%")
        if not ok:
            print("[head] 说明：WM 的 `a` 仍能吸收自身预测误差 ⇒ 反传是在优化假目标。"
                  "先解决采集里的动作–意图混淆（逐步独立扰动），别开联合微调。")
            return 3
    print("[head] 判读：headroom ≲20% ⇒ 录制动作已接近 WM 最优，`a` 可当控制输入；"
          "≳50% ⇒ 假空间太大，反传不可信")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
