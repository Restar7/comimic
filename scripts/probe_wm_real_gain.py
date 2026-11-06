#!/usr/bin/env python3
"""WM 的「根漂移增益」是否和真实一致？—— 决定奖励梯度能不能迁移。

用户口径（2026-11-04）
----------------------
奖励就是"WM 想象执行出来的 root/body 位置 vs 理想"的漂移，压它本身没错。
那么"压了不生效"只可能是**想象这一步的增益失真**：如果对**同一档动作扰动**，
WM 预测的根漂移增量远小于真实系统上实测到的增量，优化器就会：
    以为"再推一点才有效" → 真机上早就过头 → 动作过大/相位错乱/站桩。
反之如果 WM 增益**偏大**，优化器会过分保守（少动）。

本脚本两侧同口径测：
  * **真实侧**：采集 A/B 已实测（σ=0 → 20 步漂移 0.84 cm；σ=0.05 → 1.56 cm），
    即"σ=0.05 的保持型扰动让真实 20 步根漂移增加 **+0.72 cm**"。可用 `--real-delta-cm` 覆盖。
  * **WM 侧**：同窗口、真值动作 vs 真值动作 + 保持型 σ 噪声（与 `perturb_hold` 同口径），
    比较 20 步后根位置相对起点的漂移，取 p50 增量。

判据：|gain_wm / gain_real − 1| ≤ 0.5 才算"梯度可迁移"。

用法：
    python3 scripts/probe_wm_real_gain.py --ckpt runs/a_prime_sonic_leash/wm_ssm_s0_more.pt
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
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "runs/a_prime_sonic_leash/wm_ssm_s0_more.pt")
    ap.add_argument("--windows", type=Path,
                    default=ROOT / "runs/a_prime_sonic_leash/roll_windows.pt")
    ap.add_argument("--n", type=int, default=768)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--real-delta-cm", type=float, default=0.72,
                    help="σ=0.05 保持型扰动在**真机**上带来的 20 步根漂移增量（采集 A/B 实测）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415

    dev = torch.device(a.device)
    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    bundle = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=npz,
                               packed_motion=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
                               obs_mode=None, hist=None, token_dim=64)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    bundle.wm.load_state_dict(ck.get("wm", ck), strict=True)
    bundle.wm.eval()
    print(f"[gain] WM = {a.ckpt}", flush=True)

    w = torch.load(a.windows, map_location="cpu", weights_only=False)
    S, A = w["states"].float(), w["actions"].float()
    K = int(a.k)
    n = min(a.n, int(S.shape[0]))
    g = torch.Generator().manual_seed(int(a.seed))
    sel = torch.randperm(int(S.shape[0]), generator=g)[:n]
    s0 = S[sel, 0].to(dev)
    acts = A[sel, :K].to(dev)
    ref_end = S[sel, K].to(dev)          # 窗口最后一帧真值 = "理想执行"（与奖励同口径）
    xi = bundle.default_xi(s0.shape[0], dev)

    def roll(seq):
        s_ = s0.clone()
        for t in range(K):
            out = bundle.step(s_, seq[:, t], xi)
            s_ = out[0] if isinstance(out, (tuple, list)) else out
        return s_

    with torch.no_grad():
        def drift_cm(x):
            """与奖励同一口径：终点**相对理想执行**的根位置误差（漂移），不是位移。"""
            return (x[:, 0:3] - ref_end[:, 0:3]).norm(dim=-1)

        def disp_cm(x):
            return (x[:, 0:3] - s0[:, 0:3]).norm(dim=-1)

        base = roll(acts)
        dr_base, dp_base = drift_cm(base), disp_cm(base)
        print(f"[gain] WM 基线：20 步终点漂移 p50 = {float(dr_base.median()) * 100:.2f} cm "
              f"（位移 {float(dp_base.median()) * 100:.2f} cm）", flush=True)
        rows = []
        for sg in a.sigmas:
            gg = torch.Generator().manual_seed(int(a.seed) + 3)
            noise = torch.randn(acts.shape, generator=gg).to(dev) * float(sg)
            pert = roll(acts + noise)      # 保持型：整段 20 步用同一噪声场（与 perturb_hold 同口径）
            dr_pert = drift_cm(pert)
            d_cm = (float(dr_pert.median()) - float(dr_base.median())) * 100
            rows.append((float(sg), float(dr_pert.median()) * 100, d_cm))
            print(f"[gain] σ={sg:.2f}: WM 20 步终点漂移 p50 = {float(dr_pert.median()) * 100:.2f} cm "
                  f"⇒ 增量 {d_cm:+.3f} cm", flush=True)

    # 真机侧：σ=0.05 的增量是 --real-delta-cm（采集 A/B 实测），按比例外推其他 σ
    print()
    print(f"[gain] 真实侧（采集 A/B 实测）：σ=0.05 ⇒ +{a.real_delta_cm:.2f} cm")
    for sg, dp, d_cm in rows:
        real_here = a.real_delta_cm * (sg / 0.05)
        ratio = d_cm / real_here if real_here > 1e-9 else float("nan")
        flag = "✅" if abs(ratio - 1.0) <= 0.5 else ("⚠ WM 低估" if ratio < 1 else "⚠ WM 高估")
        print(f"[gain] σ={sg:.2f}: WM 增量 {d_cm:+.3f} cm / 真实增量 ≈ {real_here:+.3f} cm "
              f"⇒ 增益比 {ratio:.2f}  {flag}")
    print("\n[gain] 判据：增益比 ≈1 ⇒ 奖励梯度可迁移；≪1 ⇒ 优化器会越推越多（真机过头）；"
          "≫1 ⇒ 过分保守（少动）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
