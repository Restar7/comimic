#!/usr/bin/env python3
"""**判据：WM 的梯度方向在真物理里是不是"上升方向"？**（终结性实验）

做法（用户口径：梯度反传，不是排序）：
    1) 从 B 线 dump 取 N 个窗口（真实状态 s0 + 真实动作块 a0）
    2) 用 WM 反传：g = ∂J_WM/∂a，把 g 归一化到 RMS=`--delta`（使真实状态变化落在 1–5 mm）
    3) 生成同一份 job：a± = a0 ± g ⇒ `scripts/isaac_probe_action_rollout.py` 在**真物理**里
       从同一个 s0 各滚 H 步，录下每步状态
    4) 用**同一个** `bundle.reward` 算 J_true(±)，统计：
         · 符号正确率 = P( sign(J_true(+)−J_true(−)) == sign(J_WM(+)−J_WM(−)) )
         · ΔJ 相关、幅度比
判据：符号正确率 ≥0.65 ⇒ **梯度可用**（能反传）；≈0.5 ⇒ 只能拿 WM 做排序（best-of-N）。

用法：
    PYTHONPATH=. python3 scripts/probe_wm_gradient_truth.py plan \
        --ckpt runs/S1_handover/wm_a24_k20.pt --dump "runs/wm_real/traj_wmuse2_val/v_11*.pt" \
        --horizon 2 --windows 96 --delta 0.03 --out /tmp/job_h2.pt --device cuda:0
    # ... 跑 isaac_probe_action_rollout.py（Isaac venv）...
    PYTHONPATH=. python3 scripts/probe_wm_gradient_truth.py score \
        --ckpt ... --job /tmp/job_h2.pt --roll /tmp/roll_h2.pt --device cuda:0
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _common(ap):
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")


def _bundle(a):
    import joint_contract as JC  # noqa: F401, PLC0415
    from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415

    return load_isaac_bundle(a.ckpt, device=torch.device(a.device),
                             npz=sorted((ROOT / "data/npz/ha_tokens").glob("*.npz")),
                             packed_motion=ROOT / "runs/isaac_motion/ha_cmd/packed.pt",
                             obs_mode=None, hist=None, token_dim=64)


def _windows(a):
    """从 dump 取窗口：起点帧、状态(ha)、真实动作块(standard)、参考帧序列。"""
    import joint_contract as JC  # noqa: PLC0415

    files = sorted(glob.glob(a.dump))
    if not files:
        raise SystemExit(f"没有匹配 {a.dump}")
    rows = []
    jn = None
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        S = d["states"].float()
        A = d.get("actions")
        if A is None:
            A = d["q_target_applied"]
        A = A.float().reshape(-1, 29)
        FR = d.get("frames")
        FR = FR.long() if torch.is_tensor(FR) else None
        if jn is None and d.get("joint_names"):
            jn = list(d["joint_names"])
        T = int(S.shape[0])
        for st in range(10, T - a.horizon - 1, max(1, int(a.stride))):
            rows.append({"file": Path(f).name, "start": st, "S": S, "A": A, "FR": FR})
            if len(rows) >= 4 * int(a.windows):
                break
        if len(rows) >= 4 * int(a.windows):
            break
    if not rows:
        raise SystemExit("[plan] 没取到窗口")
    g = torch.Generator().manual_seed(0)
    pick = torch.randperm(len(rows), generator=g)[: min(int(a.windows), len(rows))]
    return [rows[i] for i in pick.tolist()], jn


def do_plan(a) -> int:
    dev = torch.device(a.device)
    bundle = _bundle(a)
    H = int(a.horizon)
    from diff_vla_wbc.isaac_joint import FrameRef, default_xi  # noqa: PLC0415

    rows, jn = _windows(a)
    N = len(rows)
    S0 = torch.stack([r["S"][r["start"]] for r in rows]).to(dev)          # ha 序
    A0 = torch.stack([r["A"][r["start"]:r["start"] + H] for r in rows]).to(dev)  # standard
    PREV = torch.stack([r["A"][max(0, r["start"] - 1)] for r in rows]).to(dev)
    F0 = torch.tensor([int(r["FR"][r["start"]]) if r["FR"] is not None else 0 for r in rows])
    # 参考帧序列（每步一帧；没有 frames 就用常量）
    FSEQ = torch.stack([
        torch.stack([r["FR"][r["start"] + t] if r["FR"] is not None
                     else torch.tensor(0) for t in range(H)]) for r in rows])

    import joint_contract as JC  # noqa: PLC0415
    from diff_vla_wbc.b_start import state_to_standard  # noqa: PLC0415

    perm = torch.as_tensor(JC.permutation("ha_v31", "standard"), dtype=torch.long)

    def j_wm(s0_ha, acts):
        s = state_to_standard(s0_ha, perm).to(dev)
        xi = default_xi(int(acts.shape[0]), dev)
        tot = torch.zeros(int(acts.shape[0]), device=dev)
        a_prev = PREV
        for t in range(H):
            act = acts[:, t]
            s = bundle.step(s, act, xi, step_idx=t + 1)
            r = bundle.reward(s, act, a_prev, FrameRef(FSEQ[:, t].to(dev)))["total"]
            tot = tot + r
            a_prev = act
        return tot

    A0g = A0.clone().requires_grad_(True)
    j0 = j_wm(S0, A0g).mean()
    grad, = torch.autograd.grad(j0, A0g)
    rms = grad.pow(2).mean().sqrt() + 1e-12
    unit = grad / rms                                        # 单位 RMS 方向
    step = float(a.delta)
    gp = A0 + step * unit
    gm = A0 - step * unit
    with torch.no_grad():
        jp = j_wm(S0, gp)
        jm = j_wm(S0, gm)
    d = jp - jm
    print(f"[plan] H={H} N={N}  J_WM(a0)={float(j0):.5f}  |g| RMS={float(rms):.3f}")
    print(f"[plan] 沿 ĝ 走 δ={step:g}（动作 RMS）⇒ J_WM：+ {float(jp.mean()):.5f} / "
          f"− {float(jm.mean()):.5f}  ⇒ ΔJ_WM 均 {float(d.mean()):+.6f}（应 >0）")
    print(f"[plan] 动作块位移 RMS = {float((step*unit).pow(2).mean().sqrt()):.4f} rad")
    job = {"states_ha": S0.cpu(), "a0": A0.cpu(), "acts_plus": gp.detach().cpu(),
           "acts_minus": gm.detach().cpu(), "prev": PREV.cpu(), "frames": FSEQ,
           "horizon": H, "delta": step, "ckpt": str(a.ckpt),
           "J_WM_plus": jp.cpu(), "J_WM_minus": jm.cpu(), "dumps": [r["file"] for r in rows],
           "joint_names": jn}
    torch.save(job, a.out)
    print(f"[plan] 写好 job → {a.out}（下一步跑 isaac_probe_action_rollout.py）")
    return 0


def do_score(a) -> int:
    dev = torch.device(a.device)
    bundle = _bundle(a)
    job = torch.load(a.job, map_location="cpu", weights_only=False)
    roll = torch.load(a.roll, map_location="cpu", weights_only=False)
    H = int(job["horizon"])
    S0, sp, sm = job["states_ha"], job["acts_plus"], job["acts_minus"]
    prev, FSEQ = job["prev"], job["frames"]
    stp, stm = roll["states_plus"], roll["states_minus"]        # (N,H,71) ha 序（每步后）
    print(f"[score] N={int(S0.shape[0])} H={H}  roll={Path(a.roll).name}")

    import joint_contract as JC  # noqa: PLC0415
    from diff_vla_wbc.b_start import state_to_standard  # noqa: PLC0415
    from diff_vla_wbc.isaac_joint import FrameRef  # noqa: PLC0415

    perm = torch.as_tensor(JC.permutation("ha_v31", "standard"), dtype=torch.long)

    def j_true(states_ha, acts):
        tot = torch.zeros(int(acts.shape[0]))
        a_prev = prev
        for t in range(H):
            s_std = state_to_standard(states_ha[:, t].float(), perm)
            r = bundle.reward(s_std.to(dev), acts[:, t].to(dev), a_prev.to(dev),
                              FrameRef(FSEQ[:, t].to(dev)))["total"].cpu()
            tot = tot + r
            a_prev = acts[:, t]
        return tot

    with torch.no_grad():
        tp = j_true(stp, sp)
        tm = j_true(stm, sm)
    dt = tp - tm
    dw = (job["J_WM_plus"] - job["J_WM_minus"]).float()
    ok = (torch.sign(dt) == torch.sign(dw)).float()
    nz = dt.abs() > dt.abs().median() * 0.25
    corr = float(((dt - dt.mean()) * (dw - dw.mean())).mean() / (dt.std() * dw.std() + 1e-12))
    print("\n" + "=" * 78)
    print(f"H={H}：沿 WM 梯度方向 ±δ 走一步，真物理的回报变化 vs WM 的预测")
    print("=" * 78)
    print(f"  符号一致率            = {float(ok.mean()):.3f}"
          f"   （去掉 |ΔJ_true| 过小的 {int((~nz).sum())} 段后 "
          f"{float(ok[nz].mean()) if int(nz.sum()) > 4 else float('nan'):.3f}）")
    print(f"  ΔJ 相关               = {corr:+.3f}")
    print(f"  |ΔJ_true| 中位/均     = {float(dt.abs().median()):.6f} / {float(dt.mean()):+.6f}")
    print(f"  |ΔJ_WM|   中位/均     = {float(dw.abs().median()):.6f} / {float(dw.mean()):+.6f}")
    print(f"  WM 认为变好的比例      = {float((dw > 0).float().mean()):.3f}（应≈1，因为 ĝ 就是它的梯度）")
    print(f"  真物理变好的比例      = {float((dt > 0).float().mean()):.3f}"
          f"  ← **这个数就是'梯度能不能信'**")
    print("\n判读：真物理变好比例 ≥0.65 ⇒ 梯度可用；0.45–0.65 ⇒ 弱信号；≤0.45 ⇒ 梯度不可信（只能做排序）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    _common(p)
    p.add_argument("--dump", required=True)
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--windows", type=int, default=96)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--delta", type=float, default=0.03, help="动作块沿梯度的 RMS（rad）")
    p.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    _common(s)
    s.add_argument("--job", type=Path, required=True)
    s.add_argument("--roll", type=Path, required=True)
    a = ap.parse_args()
    return do_plan(a) if a.mode == "plan" else do_score(a)


if __name__ == "__main__":
    raise SystemExit(main())
