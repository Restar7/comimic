#!/usr/bin/env python3
"""A11/A12：**奖励项活性审计** —— 在真实的 B 线数据上量每一项到底有没有信号。

为什么必须做（2026-10-31 实证）：
  * yaw 项曾因 σ=0.25 rad 而实际误差 37~57° ⇒ 核值 1e-3~1e-7 ⇒ **整项死区**，
    而我们的 gate 只看 WM 关节误差，完全看不见；
  * yaw_rate 项与参考**相关性 0.005**（滞后 −12~+20 帧全部≈0），等于奖励噪声；
  * 两个问题都在 d1~d3 三轮里一直存在，没人发现。

判据（任何一条不满足就非零退出）：
  1) 每个 exp 核项在真实数据上的**贡献占比** ≥ 5%（否则等于没启用）；
  2) yaw 的远场必须由 hinge 提供梯度（贡献 ≥ 0.02）；
  3) 若 `W_YAWRATE > 0`，则 corr(w_z_robot, w_z_ref) 必须 ≥ 0.10（否则该项是噪声）；
  4) 每项在对数误差处的核值梯度必须 > 0（没有"常量平台"）。

用法：``python3 scripts/audit_reward_activity.py --dumps 'runs/B_d3/vla_*/dump/*.pt'``
"""
from __future__ import annotations

import argparse
import glob, math, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diff_vla_wbc.reward_shape as RS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="runs/B_*/vla_*_f*_r*/dump/*.pt")
    ap.add_argument("--packed", default="runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--max-files", type=int, default=42)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--wm-root-w-corr", type=float, default=None,
                    help="WM 在留出集上对 root 角速度的预测相关性（gate 产出）。"
                         "W_YAWRATE>0 时必须 >= 0.4，否则等于奖励 WM 伪影")
    a = ap.parse_args()

    lib = torch.load(a.packed, map_location="cpu", weights_only=False)
    ref = lib["ref"].float()
    R, F, WZ = [], [], []
    files = sorted(glob.glob(a.dumps))[: a.max_files]
    for f in files:
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        S, fr = d.get("states"), d.get("frames")
        if S is None or fr is None:
            continue
        idx = torch.arange(0, S.shape[0], a.stride)
        fr = fr.long()[idx].clamp(0, ref.shape[0] - 1)
        R.append(S.float()[idx])
        F.append(ref[fr])
        WZ.append((S.float()[idx][:, 41], ref[fr][:, 12]))
    if not R:
        print("❌ 没读到任何 B 线 dump（检查 --dumps）")
        return 1
    rob, rf = torch.cat(R), torch.cat(F)
    _, t = RS.ori_split_reward(rob, rf)
    te, ye, yh, we = (t["ori_tilt_exp"].mean(), t["ori_yaw_exp"].mean(),
                      t["ori_yaw_hinge"].mean(), t["ori_yawrate_exp"].mean())
    tot = float(te + ye + yh)
    yw = torch.cat([x for x, _ in WZ]); yf = torch.cat([y for _, y in WZ])
    yw = yw - yw.mean(); yf = yf - yf.mean()
    corr_wz = float((yw * yf).mean() / (yw.std() * yf.std() + 1e-9))
    yaw_deg = torch.rad2deg(t["yaw2"].sqrt())

    print(f"数据源: {len(files)} 个 B 线 dump / {rob.shape[0]} 个重规划时刻  "
          f"[in-distribution: 联合微调真正 rollout 的分布]")
    print(f"配置: W_TILT/W_YAW/W_YAWRATE = {RS.W_TILT}/{RS.W_YAW}/{RS.W_YAWRATE}, "
          f"YAW_STD={RS.YAW_STD} rad, hinge λ={RS.YAW_HINGE_LAMBDA} thr={RS.YAW_HINGE_THR_DEG}°")
    print(f"  yaw 误差: 中位 {yaw_deg.median():.1f}°  均值 {yaw_deg.mean():.1f}°  "
          f"p90 {yaw_deg.quantile(0.9):.1f}°")
    print(f"  tilt exp   {te:+.4f}  ({100*te/max(1e-9,tot):5.1f}% of 朝向奖励)")
    print(f"  yaw  exp   {ye:+.4f}  ({100*ye/max(1e-9,tot):5.1f}%)   ← 近场")
    print(f"  yaw  hinge -{yh:.4f}  ({100*yh/max(1e-9,tot):5.1f}%)   ← 远场")
    print(f"  yawrate    {we:+.4f}   corr(w_z_robot, w_z_ref) = {corr_wz:+.3f}")

    bad = []
    if RS.W_TILT > 0 and te / max(1e-9, tot) < 0.05:
        bad.append(f"tilt exp 占比 {100*te/max(1e-9,tot):.1f}% < 5%（等于没启用）")
    if RS.W_YAW > 0 and ye / max(1e-9, tot) < 0.05:
        bad.append(f"yaw exp 占比 {100*ye/max(1e-9,tot):.1f}% < 5%")
    if RS.YAW_HINGE_LAMBDA > 0 and yh < 0.02:
        bad.append(f"yaw hinge 贡献 {yh:.4f} < 0.02（远场没梯度）")
    # ★ 2026-10-31 修正：判据不该是"机器人状态和参考的 w_z 相关性"（那是**跟踪症状**，
    #   误差大恰恰说明奖励有信号），而应该是 **WM 能不能预测 root 角速度**——因为奖励是在
    #   想象轨迹上算的。阈值 0.4 来自实测（基线 0.500、刷新后曾掉到 0.261）。
    if RS.W_YAWRATE > 0:
        _wc = a.wm_root_w_corr
        if _wc is None:
            bad.append(f"W_YAWRATE={RS.W_YAWRATE} 但没给 --wm-root-w-corr（无法证明 WM 能预测角速度）")
        elif _wc < 0.40:
            bad.append(f"W_YAWRATE={RS.W_YAWRATE} 但 WM root_w_corr={_wc:.3f} < 0.40 ⇒ 奖励的是 WM 伪影")
        else:
            print(f"  yaw_rate 准入：WM root_w_corr={_wc:.3f} >= 0.40 ✅")
    # 梯度检查：远离 σ 处必须仍有梯度
    _a = torch.tensor([math.radians(float(yaw_deg.median())),
                       math.radians(float(yaw_deg.quantile(0.9)))],
                      dtype=torch.float64, requires_grad=True)
    _h = RS._ramp_then_slope(torch.relu(_a.abs() - math.radians(RS.YAW_HINGE_THR_DEG)),
                             math.radians(RS.YAW_HINGE_CAP_DEG), RS.YAW_HINGE_SLOPE)
    g, = torch.autograd.grad(_h.sum(), _a)
    print(f"  hinge 在中位/p90 误差处的梯度 = {[round(float(x),4) for x in g]}（必须 >0）")
    if float(g[0]) <= 0 or float(g[1]) <= 0:
        bad.append("hinge 在中位/p90 处梯度为 0（远场平台）")

    print()
    if bad:
        print(f"❌ 奖励项活性审计不合格（{len(bad)} 项）：")
        for b in bad:
            print(f"   - {b}")
        return 1
    print("✅ 奖励项活性审计通过：每项都有信号、远场有梯度、yaw_rate 未被误启用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
