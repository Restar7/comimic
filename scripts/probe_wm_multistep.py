#!/usr/bin/env python3
"""WM **多步保真度**验收（imag/real 20 步漂移比）—— 2026-11-04

为什么必须有这一关（实测依据）：
    单步指标（`qd_corr` / `pos_mse`）好看不代表能当微调的"可微 plant"。
    旧 WM 在 20 步上：想象漂移 **28.6 cm** vs 真实 **6.8 cm（4.18 倍）**，
    与真实漂移的相关只有 **0.205** ⇒ 用它算的奖励是"一套夸张的动力学"，
    联合微调整体在优化错误的东西（两轮微调曲线都不动就是这么来的）。

判据（写进 `docs/PIPELINE_SPEC_V2_20261104.md` §3.5）：
    ratio_root = p50(imag 20 步 root 漂移) / p50(real 20 步 root 漂移)  ∈ [0.8, 1.2]
    ratio_joint 同上；另外看终点误差与相关系数。

用法：
    python3 scripts/probe_wm_multistep.py --ckpt runs/a_prime_sonic_leash/wm_scratch_s0.pt
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
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="被评的 WM ckpt（给了就替换 bundle 的 wm 权重）")
    ap.add_argument("--data", type=Path,
                    default=ROOT / "runs/a_prime_sonic_leash/filtered_std.pt")
    ap.add_argument("--windows", type=Path,
                    default=ROOT / "runs/a_prime_sonic_leash/roll_windows.pt",
                    help="k=20 连续窗口（build_wm_roll_windows.py 产物）；给了就直接用")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415

    dev = torch.device(a.device)
    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    bundle = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=npz,
                               packed_motion=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
                               obs_mode=None, hist=None, token_dim=64)
    if a.ckpt is not None:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        wsd = ck.get("wm", ck)
        bundle.wm.load_state_dict(wsd, strict=True)
        print(f"[multi] WM 权重来自 {a.ckpt}", flush=True)
    wm = bundle.wm
    wm.eval()

    if a.windows.exists():
        w = torch.load(a.windows, map_location="cpu", weights_only=False)
        S, A = w["states"].float(), w["actions"].float()   # (N,k+1,71) / (N,k,29)
        n = min(a.n, int(S.shape[0]))
        g = torch.Generator().manual_seed(int(a.seed))
        idx = torch.randperm(int(S.shape[0]), generator=g)[:n]
        s0 = S[idx, 0].to(dev)
        acts = A[idx].to(dev)
        realK = S[idx, int(a.k)].to(dev)
        print(f"[multi] 用预构建窗口 {tuple(S.shape)}，抽 {n} 个", flush=True)
    else:
        d = torch.load(a.data, map_location="cpu", weights_only=False)
        S, A, F, C = d["states"], d["actions"], d["frames"], d["clip_ids"]
        order = torch.argsort(F)
        S, A, F, C = S[order], A[order], F[order], C[order]
        rows = []
        i = 0
        K = int(a.k)
        while i < len(F) - K:
            if C[i] == C[i + K] and int(F[i + K] - F[i]) == K:
                rows.append(i); i += K
            else:
                i += 1
        if not rows:
            raise SystemExit("[multi] 找不到连续窗口；先用 build_wm_roll_windows.py")
        sel = torch.tensor(rows[: a.n])
        s0 = S[sel].to(dev)
        acts = A[sel[:, None] + torch.arange(K)].to(dev)
        realK = S[sel + K].to(dev)

    xi = bundle.default_xi(s0.shape[0], dev)
    with torch.no_grad():
        s = s0.clone()
        for t in range(int(a.k)):
            out = bundle.step(s, acts[:, t], xi)
            s = out[0] if isinstance(out, (tuple, list)) else out
        dp_pred = (s[:, 0:3] - s0[:, 0:3]).norm(dim=-1)
        dp_real = (realK[:, 0:3] - s0[:, 0:3]).norm(dim=-1)
        dq_pred = (s[:, 7:36] - s0[:, 7:36]).abs().mean(-1)
        dq_real = (realK[:, 7:36] - s0[:, 7:36]).abs().mean(-1)
        e_root = (s[:, 0:3] - realK[:, 0:3]).norm(dim=-1)
        e_q = (s[:, 7:36] - realK[:, 7:36]).abs().mean(-1)
        corr = float(torch.corrcoef(torch.stack([dp_pred, dp_real]))[0, 1])

    r_root = float(dp_pred.median() / dp_real.median().clamp_min(1e-9))
    r_joint = float(dq_pred.median() / dq_real.median().clamp_min(1e-9))
    ok = 0.8 <= r_root <= 1.2 and 0.8 <= r_joint <= 1.2
    res = {
        "ckpt": str(a.ckpt) if a.ckpt else "<base>",
        "n": int(s0.shape[0]), "k": int(a.k),
        "imag_drift_root_cm": float(dp_pred.median()) * 100,
        "real_drift_root_cm": float(dp_real.median()) * 100,
        "ratio_root": r_root,
        "imag_drift_joint_rad": float(dq_pred.median()),
        "real_drift_joint_rad": float(dq_real.median()),
        "ratio_joint": r_joint,
        "end_err_root_cm_p50": float(e_root.median()) * 100,
        "end_err_root_cm_p90": float(e_root.quantile(0.9)) * 100,
        "end_err_joint_rad_p50": float(e_q.median()),
        "corr_drift": corr,
        "pass": bool(ok),
    }
    print(f"[multi] 20 步 root 漂移：想象 {res['imag_drift_root_cm']:.2f} cm / "
          f"真实 {res['real_drift_root_cm']:.2f} cm  ⇒ ratio {r_root:.2f}", flush=True)
    print(f"[multi] 20 步关节漂移：想象 {res['imag_drift_joint_rad']:.4f} / "
          f"真实 {res['real_drift_joint_rad']:.4f}  ⇒ ratio {r_joint:.2f}", flush=True)
    print(f"[multi] 终点误差 root p50={res['end_err_root_cm_p50']:.2f}cm "
          f"p90={res['end_err_root_cm_p90']:.2f}cm；关节 p50={res['end_err_joint_rad_p50']:.4f}rad", flush=True)
    print(f"[multi] corr(想象漂移, 真实漂移) = {corr:.3f}", flush=True)
    print(f"[multi] {'✅ PASS' if ok else '❌ FAIL'}（判据 ratio ∈ [0.8, 1.2]）", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))
        print(f"[multi] -> {a.out}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
