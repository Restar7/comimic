#!/usr/bin/env python3
"""G3: multi-step WM prediction error on the ft318 segmented A' (the joint-FT imagination).

`docs/MUJOCO_SONIC_JOINTFT_PLAN_20261104.md` §3 的 G3 门：**WM 20 步预测误差 p90**。
joint FT 里的"想象"就是这件事，所以这个数是 mm 目标的地板。

真值来自 A' 的分段结构：每段 ``seg_len`` 行是**连续**的 20 步（frames f..f+seg_len-1），
所以 ``states[i..i+H]`` 就是同一条真轨迹；用 ``actions[i..i+H-1]`` 自回归滚 WM，
逐步比较位置 / 朝向 / 关节。

用法::

    third_party/SuperMimic-Reproduced/.venv/bin/python scripts/eval_wm_g3_ft318.py \
        --ckpt runs/S1_handover/sonic_wm_r1.pt \
        --transitions runs/wm_real/ft318_seg/a_prime_ft318.pt \
        --horizon 20 --starts 2000 --out runs/S1_handover/sonic_wm_r1_g3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diff_vla_wbc.mjlab_joint import load_mjlab_bundle  # noqa: E402


def quat_err_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Angle between two wxyz quaternions (deg)."""
    dot = (q1 * q2).sum(-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(dot))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--transitions", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--starts", type=int, default=2000, help="抽多少个段起点（0=全部）")
    ap.add_argument("--stride", type=int, default=20, help="A' 的段长（行）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bundle = load_mjlab_bundle(args.ckpt, device=dev, token_dim=64, freeze=True)
    bundle.eval()
    data = torch.load(str(args.transitions), map_location="cpu", weights_only=False)
    S = data["states"].float()
    A = data["actions"].float()
    n = int(S.shape[0])
    starts = list(range(0, n - args.horizon - 1, args.stride))
    if args.starts and len(starts) > args.starts:
        idx = torch.randperm(len(starts))[: args.starts].sort().values
        starts = [starts[i] for i in idx.tolist()]
    print(f"[g3] ckpt={args.ckpt} rows={n} segments={len(starts)} H={args.horizon} device={dev}")

    errs = {k: {"pos": [], "ori": [], "q": []} for k in range(1, args.horizon + 1)}
    xi = bundle.default_xi(len(starts), dev)
    cur = S[[s for s in starts]].to(dev)
    gt = S[[s for s in starts]].to(dev)
    for k in range(args.horizon):
        act = A[[s + k for s in starts]].to(dev)
        cur = bundle.step(cur, act, xi)
        gt = S[[s + k + 1 for s in starts]].to(dev)
        pos = (cur[:, 0:3] - gt[:, 0:3]).norm(dim=-1)
        ori = quat_err_deg(cur[:, 3:7], gt[:, 3:7])
        q = (cur[:, 7:36] - gt[:, 7:36]).abs().mean(dim=-1)
        errs[k + 1]["pos"] += pos.cpu().tolist()
        errs[k + 1]["ori"] += ori.cpu().tolist()
        errs[k + 1]["q"] += q.cpu().tolist()

    summary = {}
    for k in sorted(errs):
        row = {}
        for name in ("pos", "ori", "q"):
            v = np.asarray(errs[k][name])
            row[name] = {"mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
                         "p90": float(np.percentile(v, 90)), "max": float(v.max())}
        summary[k] = row
        if k in (1, 5, 10, 20) or k == args.horizon:
            print(f"[g3] k={k:2d} pos p90={row['pos']['p90'] * 1000:7.2f} mm  "
                  f"ori p90={row['ori']['p90']:6.2f} deg  q p90={row['q']['p90']:.4f} rad")
    out = {"ckpt": str(args.ckpt), "rows": n, "segments": len(starts),
           "horizon": args.horizon, "error": summary}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))
        print(f"[g3] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
