#!/usr/bin/env python3
"""harness02：定位有限差分不一致的来源（**不允许**把 FD 误差直接忽略）。

在 H=20 上，中心差分 ``(J(ε+hv)−J(ε−hv))/2h`` 与 autograd ``g·v`` 在 h≈1e-2 处仍有
~3.5% 偏差。本脚本逐项定位：

1. **确定性**：同一点重复算 J 的差（排除非确定性算子）。
2. **视界扫描**：k=1,2,5,10,20 的 FD 误差 —— 判断是"逐步累积"还是"某一步的奇点"。
3. **分项拆解**：对 ``rollout.reward_terms`` 每一项分别做 autograd vs FD，
   找出是哪个奖励项在 h 内非光滑（saturating 核 exp / 尾部 hinge relu / 关节限位 …）。
4. **h 曲线**：0.3 → 3e-5，看截断（h 大）与浮点噪声（h 小）两侧的行为。

输出：``runs/harness_02_teacher/fd_diagnose.json`` + 逐项打印。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("SUPERMIMIC_HA_ALIGN", "1")
os.environ.setdefault(
    "SUPERMIMIC_ROBOT_USD",
    str(ROOT / "trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"),
)

import torch  # noqa: E402

from diff_vla_wbc import teacher_probe as tp  # noqa: E402
from diff_vla_wbc.rollout import discounted_return, imagine_with_tokens  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from harness02_gradient_consistency import (  # noqa: E402
    build_bundle,
    windows_from_transitions,
)


@torch.no_grad()
def J_at(bundle, tok, ctx, horizon):
    ro = imagine_with_tokens(bundle, tokens=tok, **ctx, horizon=horizon)
    return ro, discounted_return(ro, 0.99)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "runs/harness_02_teacher"))
    args = ap.parse_args()

    bundle = build_bundle(args.device)
    torch.manual_seed(0)
    win = windows_from_transitions(bundle, args.n, args.horizon, args.seed, args.device)
    ctx = dict(state=win["state"], references=win["references"], xi=win["xi"],
               prev_action=win["prev_action"], history=win["history"],
               obs_hist=win["obs_hist"])
    tok = win["tokens"]
    out = {"seed": args.seed, "n": args.n, "horizon": args.horizon}

    # ---------------- 1) 确定性 ----------------
    _ro1, J1 = J_at(bundle, tok, ctx, args.horizon)
    _ro2, J2 = J_at(bundle, tok, ctx, args.horizon)
    det = float((J1 - J2).abs().max())
    out["determinism_max_abs_dJ"] = det
    out["J_mean"] = float(J1.mean())
    print(f"[determinism] max|J1-J2| = {det:.3e}  (J_mean={float(J1.mean()):.6f})", flush=True)

    # ---------------- 2) 视界扫描（同一 v = 归一化 g_H）----------------
    ga = tp.shared_delta_reward_grad(bundle, tok, **ctx, horizon=args.horizon, gamma=0.99)
    v = ga.g / ga.g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v = v[:, None, :]
    gv_full = (ga.g * v[:, 0]).sum(-1)
    hs = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4, 3e-5]
    horizons = [1, 2, 5, 10, 20]
    rows = []
    for k in horizons:
        if k > args.horizon:
            continue
        gk = tp.shared_delta_reward_grad(bundle, tok, **ctx, horizon=k, gamma=0.99)
        gkv = (gk.g * v[:, 0]).sum(-1)
        for h in hs:
            fd = tp.finite_difference_directional(bundle, tok, direction=v, h=h, **ctx,
                                                  horizon=k, gamma=0.99)
            rel = float((fd - gkv).norm() / gkv.norm().clamp_min(1e-12))
            rows.append({"k": k, "h": h, "rel_err": rel,
                         "fd_mean": float(fd.mean()), "gv_mean": float(gkv.mean()),
                         "rel_err_per_window_max": float(
                             ((fd - gkv).abs() / gkv.abs().clamp_min(1e-12)).max())})
            print(f"[fd] k={k:<3d} h={h:<8g} rel={rel:.3e} fd={float(fd.mean()):+.5e} "
                  f"gv={float(gkv.mean()):+.5e}", flush=True)
    out["fd_sweep"] = rows

    # ---------------- 3) 分项拆解（H=args.horizon, h=0.01）----------------
    h = 0.01
    base = tp.as_chunk(tok).detach()
    e0 = torch.zeros(base.shape[0], 1, base.shape[-1], device=base.device)
    with torch.enable_grad():
        eps = torch.zeros(base.shape[0], 1, base.shape[-1], device=base.device).requires_grad_(True)
        ro = imagine_with_tokens(bundle, tokens=base + eps, **ctx, horizon=args.horizon)
        terms = dict(ro.reward_terms)
        tg = {}
        for name, val in terms.items():
            # 奖励 dict 里有**诊断用的 detached 项**（vel_dropped / huber_* 等），跳过
            if not torch.is_tensor(val) or not val.requires_grad:
                continue
            tg[name] = torch.autograd.grad(val.sum(), eps, retain_graph=True)[0][:, 0, :].detach()
        terms = {k: v for k, v in terms.items() if k in tg}
        g_sum = torch.autograd.grad(ro.rewards.sum(), eps)[0][:, 0, :].detach()
    with torch.no_grad():
        ro_p, _ = J_at(bundle, base + h * v, ctx, args.horizon)
        ro_m, _ = J_at(bundle, base - h * v, ctx, args.horizon)
    term_rows = []
    for name in terms:
        fd_k = (ro_p.reward_terms[name].sum(0) - ro_m.reward_terms[name].sum(0)) / (2 * h)
        gk = tg[name]
        gkv = (gk * v[:, 0]).sum(-1)
        # 该分项自身的绝对量级 vs 其方向导数：非光滑项的特征是 |J_k| 大而 g·v 小
        rel = float((fd_k - gkv).norm() / gkv.norm().clamp_min(1e-12))
        term_rows.append({
            "term": name,
            "J_mean": float(terms[name].sum(0).mean()),
            "grad_dot_v_mean": float(gkv.mean()),
            "fd_dot_v_mean": float(fd_k.mean()),
            "rel_err": rel,
            "grad_norm_mean": float(gk.norm(dim=-1).mean()),
        })
        print(f"[term] {name:<14s} J={float(terms[name].sum(0).mean()):+.4f} "
              f"g·v={float(gkv.mean()):+.4e} fd·v={float(fd_k.mean()):+.4e} rel={rel:.3e}",
              flush=True)
    out["term_breakdown_h001"] = term_rows
    _gv = (g_sum * v[:, 0]).sum(-1)
    _fd = (ro_p.rewards.sum(0) - ro_m.rewards.sum(0)) / (2 * h)
    out["total_h001"] = {"gv_mean": float(_gv.mean()), "fd_mean": float(_fd.mean()),
                         "rel_err": float((_fd - _gv).norm() / _gv.norm().clamp_min(1e-12))}

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "fd_diagnose.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[done] -> {Path(args.out) / 'fd_diagnose.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
