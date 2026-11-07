#!/usr/bin/env python3
"""harness02：定位 H=20 有限差分残差的来源——是**执行器(ONNX)**还是**WM+奖励**？

做法：绕开 token→action，直接把参考 rollout 的动作序列 A:(H,B,29) 当自由变量，
比较 autograd ``dJ/dA`` 与中心差分。若这里的 FD 一致而 token 空间的 FD 不一致，
残差就在**可微 ONNX decoder**（含 clamp/tanh 类非光滑点）；若这里也不一致，
残差在 WM+奖励的非线性/非光滑（exp 核、尾部 hinge、huber…）。
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("SUPERMIMIC_HA_ALIGN", "1")
os.environ.setdefault("SUPERMIMIC_ROBOT_USD", str(ROOT / "trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"))
import torch
from diff_vla_wbc import teacher_probe as tp
from diff_vla_wbc.rollout import discounted_return, imagine_with_tokens
from harness02_gradient_consistency import build_bundle, windows_from_transitions

def J_from_A(bundle, A, ctx, H, gamma=0.99):
    s = ctx["state"]
    J = torch.zeros(A.shape[1], device=A.device, dtype=A.dtype)
    prev = ctx["prev_action"]
    for t in range(H):
        nxt = bundle.step(s, A[t], ctx["xi"], step_idx=t + 1)
        r = bundle.reward(nxt, A[t], prev if prev is not None else A[max(t-1,0)], ctx["references"][t])
        J = J + (gamma ** t) * r["total"]
        s = nxt
    return J

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--out", default=str(ROOT/"runs/harness_02_teacher"))
    a = ap.parse_args()
    bundle = build_bundle(a.device); torch.manual_seed(0)
    win = windows_from_transitions(bundle, a.n, 20, a.seed, a.device)
    ctx = dict(state=win["state"], references=win["references"], xi=win["xi"], prev_action=win["prev_action"], history=win["history"], obs_hist=win["obs_hist"])
    with torch.no_grad():
        ro = imagine_with_tokens(bundle, tokens=win["tokens"], **ctx, horizon=20)
    A0 = ro.actions.detach()
    out = {"seed": a.seed, "n": a.n, "rows": []}
    for H in (5, 20):
        A = A0[:H].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            Jv = J_from_A(bundle, A, ctx, H)
            g = torch.autograd.grad(Jv.sum(), A)[0].detach()
        v = g / g.norm().clamp_min(1e-12)
        gv = (g * v).sum(dim=(0, 2))
        for h in (3e-2, 1e-2, 3e-3, 1e-3):
            with torch.no_grad():
                Jp = J_from_A(bundle, A0[:H] + h * v, ctx, H)
                Jm = J_from_A(bundle, A0[:H] - h * v, ctx, H)
            fd = (Jp - Jm) / (2 * h)
            rel = float((fd - gv).norm() / gv.norm().clamp_min(1e-12))
            out["rows"].append({"H": H, "h": h, "rel_err": rel, "gv_mean": float(gv.mean()), "fd_mean": float(fd.mean()), "g_norm_mean": float(g.norm(dim=(0,2)).mean())})
            print(f"[action-FD] H={H:<3d} h={h:<7g} rel={rel:.3e} gv={float(gv.mean()):+.4e} fd={float(fd.mean()):+.4e} |gA|={float(g.norm(dim=(0,2)).mean()):.3f}", flush=True)
    Path(a.out, "fd_action_probe.json").write_text(json.dumps(out, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
