#!/usr/bin/env python3
"""拟合 **root 角速度的逐视界幅度标定** α_k（不做任何重训）。

问题：WM 的 ``root_w_ratio`` 随视界衰减（in-dist 实测 k=1 0.81 → k=5 0.62 → k=10 0.44）。
原因是 L2 训练出的确定性模型回归**条件均值**，速度通道幅度系统性偏小，自回归时逐步累积。

为什么不能用"每步乘一个常数"：那会**指数复利**（α=1.10 时 k=20 放大 6.7 倍，直接爆掉）。
所以按视界给一张标定表 α_k，只作用于 **root 角速度 w**（blast radius 最小）。

标定口径：``α_k = Σ|w_true| / Σ|w_pred|``（逐 k、跨样本与维度），
使 ``mean|α_k·w_pred| / mean|w_true| = 1``。

**必须在独立标定集上拟合**：留出集（SitSofa）是 gate 用的，拿它拟合等于用评测集调参。
本脚本默认用**非留出**的 B dump 做标定。

用法：
  python3 scripts/fit_wm_w_calib.py --ckpt runs/S1_handover/wm_d3.pt \
      --dumps 'runs/wm_real/traj_d3_train/*.pt' --horizon 20 --out runs/S1_handover/wm_d3_wcalib.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diff_vla_wbc.isaac_joint import flat_state_tuple, load_isaac_bundle  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dumps", required=True, help="glob：**标定集**（不能是 gate 的留出集）")
    ap.add_argument("--packed", default="runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--max-files", type=int, default=0, help="0=全部")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    bundle = load_isaac_bundle(a.ckpt, device=a.device, obs_mode=None, token_dim=64,
                              packed_motion=a.packed)
    files = sorted(glob.glob(a.dumps))
    if a.max_files:
        files = files[: a.max_files]
    if not files:
        print(f"❌ 没匹配到 {a.dumps}")
        return 1
    pred = [[] for _ in range(a.horizon + 1)]
    true = [[] for _ in range(a.horizon + 1)]
    n_starts = 0
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        S, A = d.get("states"), d.get("actions")
        if S is None or A is None or S.shape[0] <= a.horizon + 1:
            continue
        S = S.float()
        idx = torch.arange(0, S.shape[0] - a.horizon - 1, a.stride)
        if idx.numel() == 0:
            continue
        n_starts += int(idx.numel())
        s = S[idx].to(a.device)
        A = A.float().to(a.device)
        xi = torch.zeros(s.shape[0], 33, device=a.device)
        pred[0].append(s[:, 39:42].cpu())
        true[0].append(s[:, 39:42].cpu())
        for k in range(1, a.horizon + 1):
            s = bundle.step(s, A[idx + k - 1], xi)
            pred[k].append(s[:, 39:42].cpu())
            true[k].append(S[idx + k][:, 39:42].cpu())
        if n_starts >= 6000:
            break
    alphas, ratios_before = [], []
    for k in range(a.horizon + 1):
        if not pred[k]:
            alphas.append(1.0)
            continue
        p = torch.cat(pred[k]);
        t = torch.cat(true[k])
        r = float(p.abs().mean() / (t.abs().mean() + 1e-9))
        alphas.append(float(t.abs().mean() / (p.abs().mean() + 1e-9)) if p.abs().mean() > 1e-9 else 1.0)
        ratios_before.append((k, r))
    # ★ 关键：`alpha_k` 是**累计**修正（第 k 步共需要乘多少），而 bundle 每步只乘一次 ⇒
    #   必须换算成**增量** gamma_k = alpha_k / alpha_{k-1}，否则会指数复利
    #   （实测：直接乘累计值时 k=5 的 ratio 掉到 0.52、k=10 掉到 0.24）。
    gammas = [1.0] * len(alphas)
    for k in range(1, len(alphas)):
        gammas[k] = alphas[k] / alphas[k - 1] if alphas[k - 1] > 1e-9 else 1.0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "ckpt": str(a.ckpt), "calib_dumps": str(a.dumps), "n_starts": n_starts,
        "horizon": a.horizon, "alpha_w_cumulative": alphas, "gamma_w": gammas,
    }, indent=1))
    print(f"标定集：{len(files)} 个 dump / {n_starts} 个起点（**非留出集**）")
    print(" k   原始 ratio   α_k(累计)  γ_k(每步)")
    for k, r in ratios_before:
        print(f"{k:2d}   {r:8.3f}   {alphas[k]:9.3f}   {gammas[k]:7.3f}")
    print(f"\n⇒ 写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
