#!/usr/bin/env python3
"""按**梯度尺度**标定 v5 的损失权重（`λ_act / λ_smooth / λ_anchor / λ_raw`）。

问题（实测）：规范锁定的 `λ_act=0.010, λ_smooth=0.002` 在本数据上让正则项的**参数梯度**
是 `J` 的 27× / 7.5×；而按旧梯度比例硬套 `λ_act=3e-4` 仍会有 ~82% 的干扰。
⇒ 必须先修"同观测动作 trust"，**再重新测各项梯度**，然后按"每项 ≈ ρ·‖g_J‖"反解 λ。

用法::

    source scripts/harness03_env.sh
    # 用缓存真实 VLA chunk 的池（pilot 口径）
    python3 scripts/calibrate_v5_weights.py \
        --windows $MAIN/runs/harness_03_bptt/big/windows.pt \
        --vla-tokens $MAIN/runs/harness_03_bptt/big/vla_tokens.pt \
        --batch 64 --target-ratio 0.1 --device cuda:0

输出：每项的 ‖g‖（λ=1 时）、反解出的 λ、以及 `‖u‖` 的尺度（用于 `L_raw`）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(os.environ.get("COMIMIC_MAIN", str(ROOT)))
for _p in (str(MAIN), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(MAIN / "scripts"))

os.environ.setdefault("SUPERMIMIC_HA_ALIGN", "1")
os.environ.setdefault(
    "SUPERMIMIC_ROBOT_USD",
    str(MAIN / "trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"))

import diag_bptt_learnability as d03                                    # noqa: E402
from diff_vla_wbc.action_trust_v5 import (                              # noqa: E402
    anchor_penalty, joint_error_rad, physical_action_trust, q_target_from_action,
    smoothness_penalty)
from diff_vla_wbc.delta_token_temporal_v5 import (                      # noqa: E402
    HIST_DIM, TemporalActorCfg, TemporalDeltaActorV5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--vla-tokens", type=Path, required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--cap", type=float, default=0.08)
    ap.add_argument("--tau", type=float, default=0.020)
    ap.add_argument("--target-ratio", type=float, default=0.1,
                    help="目标：每项正则的梯度 ≈ ratio·‖g_J‖")
    ap.add_argument("--head-scale", type=float, default=1.0,
                    help="把末层权重整体放大 k 倍，模拟 ‖u‖ 已经长大的段（测 λ_raw 用）")
    ap.add_argument("--ckpt", type=Path, default=None, help="可选：在给定 ckpt 的状态上标定（推荐用饱和前的真实 ckpt）")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    H = int(a.horizon)
    torch.manual_seed(0)
    env = d03.load_env(a.device, MAIN / "runs/bptt1/args.json")
    ws = d03.load_windows(env, SimpleNamespace(windows=a.windows, vla_tokens=a.vla_tokens,
                                               out=a.windows.parent, tokens="vla"), H)
    w_all = d03.split_sets(ws)[a.split]
    w = w_all.index(slice(0, int(a.batch)))
    b = w.to(dev, H)
    hist = w.history[:, :, :HIST_DIM].contiguous().to(dev)
    B, T = int(b["tokens"].shape[0]), int(b["tokens"].shape[1])
    K = 20
    mask = torch.zeros(1, T, device=dev)
    mask[:, :K] = 1.0
    actor = TemporalDeltaActorV5(TemporalActorCfg(chunk_len=T, cap=float(a.cap))).to(dev)
    if a.ckpt is not None and Path(a.ckpt).exists():
        from diff_vla_wbc.delta_token_io_v5 import load_delta_token_v5_from_checkpoint
        _obj = torch.load(str(a.ckpt), map_location="cpu", weights_only=False)
        _net, _spec = load_delta_token_v5_from_checkpoint(_obj, device=str(dev))
        if _net is None:
            raise SystemExit("[calib] 不是 v5 temporal ckpt")
        actor.load_state_dict(_net.state_dict())
        print(f"[calib] loaded ckpt {a.ckpt}")
    if float(a.head_scale) != 1.0:
        with torch.no_grad():
            actor.head[-1].weight.mul_(float(a.head_scale))
    actor.train()
    for p in actor.parameters():
        p.requires_grad_(True)
    params = [p for p in actor.parameters() if p.requires_grad]

    def grads_for(terms: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        names = list(terms)
        for i, name in enumerate(names):
            actor.zero_grad(set_to_none=True)
            terms[name].backward(retain_graph=(i < len(names) - 1))
            out[name] = torch.cat([
                (p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten())
                for p in params]).detach()
        return out

    delta, raw_u = actor(b["tokens"], b["state64"], hist, mask.expand(B, -1),
                         cap=float(a.cap), return_raw=True)
    # 同观测 trust：记录修正 rollout 的每步 obs，再用同一 obs + 基础 token 解一次
    recorded: list = []
    orig_exec = env.bundle.executor.act

    def _rec(token, obs93):
        recorded.append((token.detach(), obs93))
        return orig_exec(token, obs93)

    env.bundle.executor.act = _rec
    try:
        J, ex = d03.window_returns(env.bundle, dict(b, tokens=b["tokens"] + delta), H,
                                   want_actions=True)
    finally:
        env.bundle.executor.act = orig_exec
    q_corr = q_target_from_action(env.bundle.default_joint_pos, env.bundle.action_scale,
                                  ex["actions"].float().permute(1, 0, 2))
    with torch.no_grad():
        rows = [q_target_from_action(env.bundle.default_joint_pos, env.bundle.action_scale,
                                     orig_exec(b["tokens"][:, t].detach(), o.detach()).float())
                for t, (tk, o) in enumerate(recorded[:H])]
        q_base_same = torch.stack(rows, dim=1)   # (B,H,29)
    e_same = joint_error_rad(q_corr, q_base_same.detach())
    M_surv = ex["rollout"].survival[:H].permute(1, 0).detach()
    L_act, _ = physical_action_trust(e_same, M_surv, tau=float(a.tau))
    L_sm = smoothness_penalty(e_same, M_surv, tau=float(a.tau))
    L_anc = anchor_penalty(delta, mask.expand(B, -1), cap=float(a.cap))
    un = raw_u.norm(dim=-1)
    L_raw = (torch.relu(un / (2.0 * float(a.cap)) - 1.0).pow(2) * mask).sum() / mask.sum()

    terms = {"J": -J.mean(), "act": L_act, "smooth": L_sm, "anchor": L_anc, "raw": L_raw}
    vals = {k: float(v.detach()) for k, v in terms.items()}
    gs = grads_for(terms)
    gJ = float(gs["J"].norm())
    rows_out = []
    for name in ("act", "smooth", "anchor", "raw"):
        gn = float(gs[name].norm())
        lam = (float(a.target_ratio) * gJ / gn) if gn > 0 else float("inf")
        rows_out.append(dict(term=name, value=vals[name], grad_norm=gn,
                             ratio_to_J=gn / max(gJ, 1e-12), lambda_for_target=lam))
    print(f"[calib] split={a.split} B={B} cap={a.cap} head_scale={a.head_scale}")
    print(f"[calib] ‖g_J‖ = {gJ:.4e}")
    print(f"[calib] ‖Δ‖(t<K) mean={float(delta.norm(dim=-1)[:, :K].mean()):.4f} | "
          f"‖u‖ mean={float(un.mean()):.4f} p95={float(torch.quantile(un.flatten(), 0.95)):.4f} "
          f"| ‖u‖/cap p50={float((un/a.cap).median()):.3f} "
          f"| frac>2c={float((un > 2*a.cap).float().mean()):.3f}")
    print(f"[calib] 目标：每项梯度 = {a.target_ratio}·‖g_J‖；"
          f"当前锁定的 λ 下实测比值见下表")
    for r in rows_out:
        print(f"  {r['term']:7s} L={r['value']:.5f} ‖g(λ=1)‖={r['grad_norm']:.3e} "
              f"ratio={r['ratio_to_J']:.2f}× ⇒ λ*={r['lambda_for_target']:.3e}")
    obj = dict(split=a.split, batch=B, cap=a.cap, head_scale=a.head_scale,
               tau=a.tau, target_ratio=a.target_ratio, g_J=gJ,
               delta_norm_mean=float(delta.norm(dim=-1)[:, :K].mean()),
               u_norm_mean=float(un.mean()),
               u_norm_p95=float(torch.quantile(un.flatten(), 0.95)),
               u_over_cap_p50=float((un / a.cap).median()),
               frac_u_gt_2c=float((un > 2 * a.cap).float().mean()),
               terms=rows_out)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
        print(f"[calib] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
