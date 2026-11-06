#!/usr/bin/env python3
"""把 B 线闭环的 root 位置误差**分解**开：多少是"不可控的常数偏置"，多少是"随时间的波动"。

为什么需要（2026-10-30，对齐 mjlab `ISSUES_ROOT_XY_YAW` 的第 3/4 条）
--------------------------------------------------------------------
机器人观测（93 维）里**没有位置、也没有线速度** ⇒ 策略看不到自己在哪，
只能从 q/重力/上一动作间接推断"我跟参考差多少"。所以位置误差里：

* **沿轨迹方向的常数滞后**（一路都慢半拍/少走一点）—— 策略理论上只能通过"走快些"间接修正；
* **横向偏移**（往旁边偏）—— 与 heading/yaw 耦合，而 yaw 对策略不可观测；
* **随时间的抖动** —— 这部分才是"可反馈"的。

如果误差绝大部分是"**每集一个常数**"（集内方差远小于集间方差），那说明
这不是"反馈没学好"，而是**系统性的偏置**（冻结 WBC 的站姿倾向 / 步幅偏小），
用奖励去磨它效率极低，应该直接查执行器层面的偏置。

输出
----
每集：`|Δp|` 均值、沿轨迹分量均值、横向分量均值、集内标准差；
汇总：集间/集内方差分解，以及"常数占比"。

用法：
    python3 scripts/analyze_root_error.py --dump 'runs/B_c*/vla_*/dump/*.pt' --n 40
"""
from __future__ import annotations

import argparse
import glob
import math
import statistics as st

import torch

ROOT_PACKED = "runs/isaac_motion/ha_cmd/packed.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--packed", default=ROOT_PACKED)
    ap.add_argument("--n", type=int, default=40, help="最多分析多少条（等间隔抽样）")
    args = ap.parse_args()

    p = torch.load(args.packed, map_location="cpu", weights_only=False)
    lib = torch.as_tensor(p["ref"]).float()
    n_lib = int(lib.shape[0])

    files = sorted(glob.glob(args.dump))
    if not files:
        raise SystemExit(f"[rooterr] 没匹配到 {args.dump}")
    if len(files) > args.n:
        step = len(files) / args.n
        files = [files[int(i * step)] for i in range(args.n)]
    print(f"[rooterr] 分析 {len(files)} 条（共匹配 {len(glob.glob(args.dump))} 条）")

    rows = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        S = d.get("states")
        F = d.get("frames")
        if S is None or F is None:
            continue
        S = S.float()
        F = F.long()
        if int(F.max()) >= n_lib:
            continue
        if int(F.numel()) != int(S.shape[0]):
            # vla 臂只在重规划时刻记 frames，按段展开到每 tick
            n_f = int(F.numel())
            stride = max(1, (int(S.shape[0]) + n_f - 1) // n_f)
            F = F[torch.clamp(torch.arange(int(S.shape[0])) // stride, max=n_f - 1)]
        R = lib[F]
        dp = S[:, :2] - R[:, :2]                     # (T,2) 世界系 xy 误差
        # 参考速度方向（库里的 w 在 36:39 是线速度；用相邻参考帧差分更稳）
        v = torch.zeros_like(dp)
        v[1:] = R[1:, :2] - R[:-1, :2]
        sp = v.norm(dim=-1)
        vhat = v / sp.clamp_min(1e-6)[:, None]
        along = (dp * vhat).sum(-1)
        cross = (dp[:, 0] * (-vhat[:, 1]) + dp[:, 1] * vhat[:, 0])
        mag = dp.norm(dim=-1)
        # 只在"参考确实在动"的帧上算沿轨迹/横向（静止帧方向没定义）
        m = sp > 1e-4
        rows.append(dict(f=f.split("/")[-3],
                         mag=float(mag.mean()),
                         along=float(along[m].mean()) if int(m.sum()) > 5 else float("nan"),
                         cross=float(cross[m].abs().mean()) if int(m.sum()) > 5 else float("nan"),
                         within=float(mag.std()),                 # 集内波动
                         n=int(S.shape[0])))

    if not rows:
        print("[rooterr] 没有可用 dump")
        return 1

    mag = [r["mag"] for r in rows]
    within = [r["within"] for r in rows]
    along = [r["along"] for r in rows if r["along"] == r["along"]]
    cross = [r["cross"] for r in rows if r["cross"] == r["cross"]]
    between = st.pstdev(mag) if len(mag) > 1 else 0.0
    win = st.mean(within)
    print()
    print(f"  |Δp_root| 均值            : {st.mean(mag)*100:6.2f} cm   (p50 {st.median(mag)*100:.2f})")
    print(f"  沿轨迹方向 均值            : {st.mean(along)*100:+6.2f} cm  (>0 = 落后于参考)")
    print(f"  横向 |偏移| 均值           : {st.mean(cross)*100:6.2f} cm")
    print(f"  集内波动（每集 |Δp| 的 sd）: {win*100:6.2f} cm")
    print(f"  集间波动（各集均值的 sd）  : {between*100:6.2f} cm")
    tot = (win ** 2 + between ** 2) ** 0.5
    print(f"  ⇒ 常数(集间)占比 {(between**2)/(tot**2+1e-12)*100:5.1f}%   "
          f"抖动(集内)占比 {(win**2)/(tot**2+1e-12)*100:5.1f}%")
    print()
    print("  解读：常数占比高 ⇒ 是**系统性偏置**（步幅/站姿），奖励磨它效率低；")
    print("        抖动占比高 ⇒ 是**反馈不足**，才值得用奖励/闭环去改。")

    # ---- 横向误差是不是由 heading(yaw) 漂移驱动？------------------------------
    # 若 corr(|横向|, |Δyaw|) 高，说明"走偏"主要是**朝向漂了**，
    # 而 93 维观测里**没有 yaw**（mjlab ISSUES 第 2 条）⇒ 策略看不见、学不到。
    def quat_yaw(q):
        w, x, y, z = q.unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    xs, ys = [], []
    for r in rows:
        pass
    pairs = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        S = d.get("states"); F = d.get("frames")
        if S is None or F is None:
            continue
        S = S.float(); F = F.long()
        if int(F.max()) >= n_lib:
            continue
        if int(F.numel()) != int(S.shape[0]):
            n_f = int(F.numel())
            stride = max(1, (int(S.shape[0]) + n_f - 1) // n_f)
            F = F[torch.clamp(torch.arange(int(S.shape[0])) // stride, max=n_f - 1)]
        R = lib[F]
        dp = S[:, :2] - R[:, :2]
        v = torch.zeros_like(dp); v[1:] = R[1:, :2] - R[:-1, :2]
        sp = v.norm(dim=-1); vhat = v / sp.clamp_min(1e-6)[:, None]
        cross = (dp[:, 0] * (-vhat[:, 1]) + dp[:, 1] * vhat[:, 0])
        dyaw = (quat_yaw(S[:, 3:7]) - quat_yaw(R[:, 3:7]))
        dyaw = torch.remainder(dyaw + 3.14159265, 6.2831853) - 3.14159265
        m = sp > 1e-4
        if int(m.sum()) > 5:
            pairs.append((float(cross[m].abs().mean()), float(dyaw[m].abs().mean())))
    if len(pairs) > 3:
        a1 = [q[0] for q in pairs]; a2 = [q[1] for q in pairs]
        m1, m2 = st.mean(a1), st.mean(a2)
        cov = sum((u - m1) * (v - m2) for u, v in pairs) / len(pairs)
        rho = cov / (st.pstdev(a1) * st.pstdev(a2) + 1e-12)
        print()
        print(f"  |横向误差| 均值 {m1*100:.1f} cm   |Δyaw| 均值 {math.degrees(m2):.1f}°")
        print(f"  corr(|横向|, |Δyaw|) = {rho:+.2f}  （高 ⇒ 走偏主要由 heading 漂移驱动，"
              f"而 yaw 对策略不可观测）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
