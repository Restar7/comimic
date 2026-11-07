#!/usr/bin/env python3
"""诊断：**这个评测集上 Δtoken 的可提升上界**（per-window oracle 方向）+ Actor 方向对齐度。

为什么需要：训练到 it=100 时 held-out ΔJ ≈ 0。要区分两种解释——
(a) 优化还没走到（上界很大，Actor 只拿到几 %）；
(b) 评测集对 Δ 不敏感（上界本身≈0，怎么训都白搭）。

做法（每个窗口独立，纯 `no_grad` + 一次 `autograd.grad`）：
1. Δ=0 的 J0；
2. 把 Δ_t (t<K) 当自由变量（形状 (K,64)），求 g = ∂J/∂Δ；
3. oracle 方向 Δ* = cap · g/‖g‖（逐窗口），量 J(Δ*) − J0（**上界**）；
4. 同样把当前 Actor 的 Δ 拿出来，量 cos(Δ_actor, g) 与 J(Δ_actor) − J0。

用法::

    source scripts/harness03_env.sh
    python3 scripts/diag_oracle_upper_v5.py --eval-set runs/.../eval_set --split val \
        --ckpt runs/.../seed0/checkpoint_actor_best.pt --device cuda:0 \
        --out runs/.../seed0/oracle_upper_val.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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

import diag_bptt_learnability as d03                                   # noqa: E402
from diff_vla_wbc.delta_token_io_v5 import load_delta_token_v5_from_checkpoint  # noqa: E402
from diff_vla_wbc.delta_token_temporal_v5 import HIST_DIM              # noqa: E402


def _J(env, b, tokens, H):
    J, _ = d03.window_returns(env.bundle, dict(b, tokens=tokens), H)
    return J


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=Path, required=True)
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--cap", type=float, default=0.08)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    H = 20
    env = d03.load_env(a.device, MAIN / "runs/bptt1/args.json")
    ws = d03.load_windows(env, SimpleNamespace(windows=a.eval_set / "windows.pt",
                                               vla_tokens=a.eval_set / "vla_tokens.pt",
                                               out=a.eval_set, tokens="vla"), H)
    w_all = d03.split_sets(ws)[a.split]
    actor = None
    if a.ckpt is not None and Path(a.ckpt).exists():
        obj = torch.load(str(a.ckpt), map_location="cpu", weights_only=False)
        actor, spec = load_delta_token_v5_from_checkpoint(obj, device=str(dev))
        K = int(spec.K)
        mask = torch.zeros(1, 40, device=dev)
        mask[:, :K] = 1.0
    else:
        K = int(a.K)
        mask = None
    dJ_oracle, dJ_actor, cos_all, gnorm = [], [], [], []
    for s0 in range(0, len(w_all), int(a.batch)):
        w = w_all.index(slice(s0, s0 + int(a.batch)))
        b = w.to(dev, H)
        B = len(w)
        base = b["tokens"]                                        # (B,40,64)
        with torch.no_grad():
            J0 = _J(env, b, base, H)
        dvar = torch.zeros(B, K, 64, device=dev, requires_grad=True)
        tok = base.clone()
        tok[:, :K] = tok[:, :K] + dvar
        with torch.enable_grad():
            J = _J(env, b, tok, H)
            g = torch.autograd.grad(J.sum(), dvar, retain_graph=False)[0]  # (B,K,64)
        gn = g.flatten(1).norm(dim=1)
        d_or = (a.cap * g / gn.clamp_min(1e-9)[:, None, None]).detach()
        tok_or = base.clone()
        tok_or[:, :K] = tok_or[:, :K] + d_or
        with torch.no_grad():
            J_or = _J(env, b, tok_or, H)
        dJ_oracle.append((J_or - J0).detach().cpu())
        gnorm.append(gn.detach().cpu())
        if actor is not None:
            hist = w.history[:, :, :HIST_DIM].contiguous().to(dev)
            with torch.no_grad():
                d_act = actor(base, b["state64"], hist, mask.expand(B, -1), cap=float(a.cap))
                J_act = _J(env, b, base + d_act, H)
            dJ_actor.append((J_act - J0).detach().cpu())
            da = d_act[:, :K].flatten(1)
            cos = (da * g.flatten(1)).sum(1) / (da.norm(dim=1).clamp_min(1e-9)
                                                * gn.clamp_min(1e-9))
            cos_all.append(cos.detach().cpu())
    out = dict(
        split=a.split, n=int(len(w_all)), cap=float(a.cap), K=int(K),
        dJ_oracle_mean=float(torch.cat(dJ_oracle).mean()),
        dJ_oracle_p50=float(torch.quantile(torch.cat(dJ_oracle), 0.5)),
        dJ_oracle_frac_pos=float((torch.cat(dJ_oracle) > 0).float().mean()),
        grad_norm_mean=float(torch.cat(gnorm).mean()),
        ckpt=(str(a.ckpt) if a.ckpt else None),
        dJ_actor_mean=(float(torch.cat(dJ_actor).mean()) if dJ_actor else None),
        cos_actor_grad_mean=(float(torch.cat(cos_all).mean()) if cos_all else None),
        note=("oracle = 逐窗口 ∂J/∂Δ 方向 + cap 范数（不可部署，仅上界）；"
              "Actor 的方向余弦说明它有没有走在正确方向上"),
    )
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
