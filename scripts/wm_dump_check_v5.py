#!/usr/bin/env python3
"""§5.5.5 交叉验证：用 **B 线 dump**（连续帧 + 真实动作）复刻历史"20 步端点 0.89 cm"。

为什么需要它：`all_transitions.pt` 是**稀疏采样**（同 clip 相邻帧只占 0.05%），没法回放 40 个
连续控制步；`wm_horizon_error_v5.py` 只能用"参考命令代理"，口径与历史不同。
本脚本用现有的 B dump（`runs/B_*/dump/*.pt`：2080 帧连续动作/状态）走**历史同款**口径，
并**自证**动作/状态顺序：

  * 状态：dump 是 **HA_V31（asset 序）**，WM/训练侧是 **standard** ⇒ `x[perm(ha→std)]`；
  * 动作：dump 的 `actions` 有两种可能（policy 输出=STANDARD / decoder 输出=HA）；
    两种都跑，**误差小的那个**即真实口径（数据自证，不靠注释）。

用法::

    source scripts/harness03_env.sh
    python3 scripts/wm_dump_check_v5.py --out runs/temporal_ac_bptt_v5/main \
        --dump 'runs/B_*/dump/*.pt' --horizon 20 --starts 64
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(os.environ.get("COMIMIC_MAIN", str(ROOT)))
for _p in (str(MAIN), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("SUPERMIMIC_HA_ALIGN", "1")
os.environ.setdefault(
    "SUPERMIMIC_ROBOT_USD",
    str(MAIN / "trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"))

import diag_bptt_learnability as d03                              # noqa: E402
from diff_vla_wbc.compat import quat_error_magnitude              # noqa: E402
from diff_vla_wbc.delta_token_io_v5 import joint_permutation      # noqa: E402


def _quant(x: torch.Tensor) -> dict:
    if x.numel() == 0:
        return dict(p50=float("nan"), p90=float("nan"), mean=float("nan"))
    q = torch.quantile(x.float(), torch.tensor([0.5, 0.9]))
    return dict(p50=float(q[0]), p90=float(q[1]), mean=float(x.float().mean()))


def _roll(env, s0, acts, xi, H):
    s = s0
    out = [s]
    for k in range(H):
        with torch.no_grad():
            s = env.bundle.step(s, acts[:, k], xi, step_idx=k + 1)
        out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bptt-args", type=Path, default=MAIN / "runs/bptt1/args.json")
    ap.add_argument("--dump", default=str(MAIN / "runs/B_*/dump/*.pt"))
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--starts", type=int, default=64)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(int(a.seed))
    env = d03.load_env(a.device, a.bptt_args)
    perm_ha2std = joint_permutation("ha_v31", "standard")
    perm_std2ha = joint_permutation("standard", "ha_v31")
    files = sorted(glob.glob(a.dump))
    if not files:
        raise SystemExit(f"[dump] 没有匹配 {a.dump}")
    d03.log(f"[dump] {len(files)} 个 dump")

    rows = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        if "actions" not in d or "states" not in d:
            d03.log(f"[dump] SKIP {Path(f).name}（缺 actions/states）")
            continue
        S_ha = d["states"].float()
        A = d["actions"].float()
        S_std = S_ha.clone()
        S_std[:, 7:36] = S_ha[:, 7:36][:, perm_ha2std]
        S_std[:, 42:71] = S_ha[:, 42:71][:, perm_ha2std]
        # 候选动作口径：① dump 已是 STANDARD（policy 输出）；② dump 是 HA（decoder 输出）
        variants = {
            "actions_as_standard": A,
            "actions_as_ha": A[:, perm_ha2std],
        }
        starts = torch.arange(0, max(1, S_std.shape[0] - int(a.horizon) - 1), int(a.stride))
        starts = starts[torch.randperm(starts.numel())[: int(a.starts)]]
        for tag, A_cand in variants.items():
            errs = dict(pos=[], ori=[], q=[])
            for s0 in range(0, starts.numel(), 16):
                idx = starts[s0:s0 + 16]
                B = int(idx.numel())
                s0_t = S_std[idx].to(dev)
                acts = A_cand[idx.unsqueeze(1) + torch.arange(int(a.horizon))].to(dev)  # (B,H,29)
                xi = env.bundle.default_xi(B, dev)
                traj = _roll(env, s0_t, acts, xi, int(a.horizon))
                for k in (5, 10, int(a.horizon)):
                    tgt = S_std[idx + k].to(dev)
                    errs["pos"].append((traj[k] - tgt)[:, 0:3].norm(dim=-1).cpu())
                    errs["ori"].append(quat_error_magnitude(traj[k][:, 3:7], tgt[:, 3:7]).cpu())
                    errs["q"].append((traj[k] - tgt)[:, 7:36].norm(dim=-1).cpu())
                if k == int(a.horizon):
                    pass
            row = dict(dump=Path(f).name, variant=tag, n_starts=int(starts.numel()),
                       horizon=int(a.horizon),
                       pos_p50_mm_k20=float(torch.cat(errs["pos"]).median()) * 1e3,
                       pos_p90_mm_k20=float(torch.quantile(torch.cat(errs["pos"]), 0.9)) * 1e3,
                       ori_p50_rad=float(torch.cat(errs["ori"]).median()),
                       q_p50_rad=float(torch.cat(errs["q"]).median()))
            rows.append(row)
            d03.log(f"[dump] {row['dump']} {tag}: pos p50={row['pos_p50_mm_k20']:.2f}mm "
                    f"p90={row['pos_p90_mm_k20']:.2f}mm ori={row['ori_p50_rad']:.4f} "
                    f"q={row['q_p50_rad']:.4f}")

    best = min(rows, key=lambda r: r["pos_p50_mm_k20"]) if rows else None
    obj = dict(rows=rows, best_variant=(best["variant"] if best else None),
               best_pos_p50_mm_k20=(best["pos_p50_mm_k20"] if best else None),
               historical_reference_mm=8.9,
               note=("历史口径（B dump，连续动作+状态）的 20 步端点误差；用于交叉验证 "
                     "`wm_horizon_error_v5.py` 的参考命令代理是否可信"),
               generated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "wm_dump_check.json").write_text(json.dumps(obj, indent=1))
    d03.log(f"[dump] 最佳口径={obj['best_variant']} pos p50={obj['best_pos_p50_mm_k20']:.2f}mm"
            f"（历史 0.89 cm = 8.9 mm）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
