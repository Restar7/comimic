#!/usr/bin/env python3
"""WM **动作敏感度**测试（2026-11-04）

为什么必须测（用户口径）：
    WM 是 ``s' = f(s, a)``。如果训练数据里 ``a`` 永远等于"给定 s 时那个唯一的动作"
    （跟踪误差小 + 噪声小），WM 完全可以学会**忽略 a**，只用 s 就预测得不错。
    那时 ``∂s'/∂a ≈ 0`` ⇒ 联合微调里 ``∂奖励/∂token ≈ 0`` ⇒ 训练没有梯度。
    这与实测的两个现象吻合：headroom 探针 GT vs VLA 只差 0.5%；
    微调 ``imag_delta_mean −0.014`` vs ``std 0.06``（信噪比 < 0.2）。

测法：同一批状态 s，比较
    base  = wm(s, a)
    perturbed = wm(s, a + δ·N(0,1))       δ = 0.05 / 0.10 / 0.20（= 采集噪声量级）
量 ||Δp||（root）、||Δq||（关节），并与两个"标尺"比：
    step_scale = E||s'_{t} − s'_t||      相邻真实转移的状态变化（单步正常变化量）
    cross_scale = E||s'_i − s'_j||（i≠j）  不同样本之间的差异
判据：
    若 δ=0.20 时的 ||Δs'|| < 0.1 × step_scale  ⇒ **WM 基本不看 action**
    ⇒ 采集必须提高噪声（persistent excitation），否则微调注定没梯度。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wbc-ckpt", type=Path,
                    default=ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt")
    ap.add_argument("--data", type=Path, default=ROOT / "runs/a_prime_sonic/filtered.pt")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="被评的 WM ckpt（**必须给**才是在评刷新后的 WM；不给 = 用 WBC 里绑定的旧 WM）")
    ap.add_argument("--windows", type=Path,
                    default=ROOT / "runs/a_prime_sonic_leash/roll_windows.pt",
                    help="k=20 连续窗口，用于测**累积**动作效应（1 步的 ∂s'/∂a 在 dt=0.02 下天生很小）")
    ap.add_argument("--accum-k", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415
    from diff_vla_wbc.wm_refresh import _split_parts  # noqa: PLC0415

    dev = torch.device(a.device)
    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    bundle = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=npz,
                               packed_motion=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
                               obs_mode=None, hist=None, token_dim=64)
    wm = bundle.wm
    if a.ckpt is not None:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        bundle.wm.load_state_dict(ck.get("wm", ck), strict=True)
        print(f"[sens] WM 权重来自 {a.ckpt}（刷新后的 WM）", flush=True)
    else:
        print("[sens] ⚠ 没给 --ckpt：评的是 WBC 里绑定的旧 WM，结论不能用于当前流程", flush=True)
    wm = bundle.wm
    wm.eval()

    data = torch.load(a.data, map_location="cpu", weights_only=False)
    n = int(data["states"].shape[0])
    g = torch.Generator().manual_seed(int(a.seed))
    idx = torch.randperm(n, generator=g)[: a.batch]
    s = data["states"][idx].to(dev)
    act = data["actions"][idx].to(dev)
    xi = bundle.default_xi(s.shape[0], dev)

    with torch.no_grad():
        def step(s_, a_):
            out = bundle.step(s_, a_, xi)
            return out[0] if isinstance(out, (tuple, list)) else out

        base = step(s, act)
        # 标尺 1：相邻样本的真实单步变化
        s2 = data["next_states"][idx].to(dev)
        step_scale = (s2 - s).abs().mean().item()
        # 标尺 2：不同样本之间的差异
        perm = torch.randperm(s.shape[0], generator=g)[: s.shape[0]]
        cross_scale = (s2 - s2[perm]).abs().mean().item()

        rows = []
        for d in a.deltas:
            gg = torch.Generator().manual_seed(int(a.seed) + 1)
            noise = torch.randn(act.shape, generator=gg).to(dev) * float(d)
            pert = step(s, act + noise)
            d_p = (pert[:, 0:3] - base[:, 0:3]).norm(dim=-1).mean().item()
            d_q = (pert[:, 7:36] - base[:, 7:36]).abs().mean().item()
            d_all = (pert - base).abs().mean().item()
            rows.append({"delta": float(d), "d_root_m": d_p, "d_joint_rad": d_q,
                         "d_all": d_all,
                         "ratio_vs_step": d_all / (step_scale + 1e-12)})
            print(f"[sens] δ={d:.2f}: |Δroot|={d_p*100:.2f}cm  |Δjoint|={d_q:.4f}rad  "
                  f"|Δs'|={d_all:.5f}  = {d_all/(step_scale+1e-12)*100:.1f}% 的单步变化量",
                  flush=True)

    print(f"\n[sens] 标尺：单步真实变化 step_scale={step_scale:.5f}；跨样本差异 cross_scale={cross_scale:.3f}")
    worst = max(rows, key=lambda r: r["ratio_vs_step"])
    verdict = "IGNORES_ACTION" if worst["ratio_vs_step"] < 0.10 else (
        "WEAK" if worst["ratio_vs_step"] < 0.50 else "OK")
    print(f"[sens] δ={worst['delta']:.2f} 时最强敏感度 = {worst['ratio_vs_step']*100:.1f}% ⇒ **{verdict}**")

    # ---- 累积动作效应 vs WM 的 k 步误差（**这才是联合微调真正用的信噪比**）----
    # 单步 ∂s'/∂a 在 dt=0.02 下天然很小（20 ms 内 PD 只走一小段），
    # 微调的 rollout 是 k=20 步，所以要看"动作扰动坚持 k 步"放大后的效果，
    # 再和 WM 自己在 k 步上的误差相比。
    snr_rows = []
    if a.windows.exists() and int(a.accum_k) > 1:
        w = torch.load(a.windows, map_location="cpu", weights_only=False)
        S, A = w["states"].float(), w["actions"].float()
        K = int(a.accum_k)
        g2 = torch.Generator().manual_seed(int(a.seed) + 7)
        sel = torch.randperm(int(S.shape[0]), generator=g2)[: a.batch]
        s0 = S[sel, 0].to(dev)
        acts = A[sel, :K].to(dev)
        realK = S[sel, K].to(dev)
        xi2 = bundle.default_xi(s0.shape[0], dev)

        def roll(seq):
            s_ = s0.clone()
            for t in range(K):
                out = bundle.step(s_, seq[:, t], xi2)
                s_ = out[0] if isinstance(out, (tuple, list)) else out
            return s_

        with torch.no_grad():
            s_gt = roll(acts)
            # WM 自己在 k 步上的误差（= surrogate 的噪声底）
            err_k_all = (s_gt - realK).abs().mean().item()
            err_k_q = (s_gt[:, 7:36] - realK[:, 7:36]).abs().mean().item()
            print(f"\n[sens] WM 在 k={K} 步上的误差（噪声底）：|Δs|={err_k_all:.5f}  "
                  f"|Δq|={err_k_q:.5f}", flush=True)
            for d in a.deltas:
                gg = torch.Generator().manual_seed(int(a.seed) + 11)
                noise = torch.randn(acts.shape, generator=gg).to(dev) * float(d)
                s_pert = roll(acts + noise)
                eff_all = (s_pert - s_gt).abs().mean().item()
                eff_q = (s_pert[:, 7:36] - s_gt[:, 7:36]).abs().mean().item()
                snr = eff_all / (err_k_all + 1e-12)
                snr_rows.append({"delta": float(d), "effect_k": eff_all,
                                 "effect_k_q": eff_q, "wm_err_k": err_k_all, "snr": snr})
                print(f"[sens] k={K} 步累积：δ={d:.2f} 使预测状态改变 |Δs|={eff_all:.5f} "
                      f"(关节 {eff_q:.5f}) ⇒ SNR = {snr:.3f}", flush=True)

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"data": str(a.data), "ckpt": str(a.ckpt) if a.ckpt else None,
                                     "batch": a.batch, "step_scale": step_scale,
                                     "cross_scale": cross_scale, "one_step_rows": rows,
                                     "verdict_one_step": verdict, "accum_rows": snr_rows},
                                    indent=2))
        print(f"[sens] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
