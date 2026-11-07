"""诊断：29D 动作增量的**逐关节**分布（找出 peak 2 rad 是哪个关节/哪一步）。

用法::

    source scripts/harness03_env.sh
    python3 scripts/diag_action_budget_v5.py \
        --ckpt runs/temporal_ac_bptt_v5/main/seed0/checkpoint_actor_best.pt \
        --eval-set runs/temporal_ac_bptt_v5/eval_set --device cuda:0
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

import joint_contract as jc                                            # noqa: E402
import diag_bptt_learnability as d03                                   # noqa: E402
from diff_vla_wbc.action_trust_v5 import (                             # noqa: E402
    joint_error_rad, q_target_from_action)
from diff_vla_wbc.delta_token_temporal_v5 import HIST_DIM              # noqa: E402
from diff_vla_wbc.delta_token_io_v5 import load_delta_token_v5_from_checkpoint  # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--eval-set", type=Path, required=True)
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    env = d03.load_env(a.device, MAIN / "runs/bptt1/args.json")
    es = a.eval_set
    ws = d03.load_windows(env, SimpleNamespace(windows=es / "windows.pt",
                                               vla_tokens=es / "vla_tokens.pt",
                                               out=es, tokens="vla"), 20)
    sp = d03.split_sets(ws)
    w_all = sp[a.split]
    obj = torch.load(str(a.ckpt), map_location="cpu", weights_only=False)
    net, spec = load_delta_token_v5_from_checkpoint(obj, device=str(dev))
    assert net is not None
    K = int(spec.K)
    mask = torch.zeros(1, 40, device=dev)
    mask[:, :K] = 1.0
    e_all, per_joint_max, per_step_max = [], np.zeros(29), np.zeros(20)
    step_rms_sum = np.zeros(20); step_n = 0; div = []
    for s0 in range(0, len(w_all), int(a.batch)):
        w = w_all.index(slice(s0, s0 + int(a.batch)))
        b = w.to(dev, 20)
        hist = w.history[:, :, :HIST_DIM].contiguous().to(dev)
        delta = net(b["tokens"], b["state64"], hist, mask.expand(len(w), -1), cap=float(spec.cap))
        _, ex = d03.window_returns(env.bundle, {**b, "tokens": b["tokens"] + delta}, 20,
                                   want_actions=True)
        _, ex0 = d03.window_returns(env.bundle, b, 20, want_actions=True)
        q_corr = q_target_from_action(env.bundle.default_joint_pos, env.bundle.action_scale,
                                      ex["actions"].float().permute(1, 0, 2))
        q_base = q_target_from_action(env.bundle.default_joint_pos, env.bundle.action_scale,
                                      ex0["actions"].float().permute(1, 0, 2))
        e = joint_error_rad(q_corr, q_base.detach()).cpu().numpy()        # (B,H,29)
        e_all.append(e)
        per_joint_max = np.maximum(per_joint_max, np.abs(e).max(axis=(0, 1)))
        per_step_max = np.maximum(per_step_max, np.abs(e).max(axis=(0, 2)))
        step_rms_sum += np.sqrt((e ** 2).mean(axis=(0, 2)))        # (H,)
        step_n += 1
        _, ex_ro = d03.window_returns(env.bundle, {**b, "tokens": b["tokens"] + delta}, 20,
                                      want_actions=False)
        _, ex_ro0 = d03.window_returns(env.bundle, b, 20, want_actions=False)
        s1 = ex_ro["rollout"].states[-1]; s0 = ex_ro0["rollout"].states[-1]
        div.append((s1 - s0)[:, 0:3].norm(dim=-1).cpu().numpy())
    E = np.concatenate(e_all, 0)                                          # (N,H,29)
    _std = list(getattr(jc, "STANDARD_29", []))
    names = [n.replace("_joint", "") for n in _std] if len(_std) == 29 else [f"j{i}" for i in range(29)]
    order = np.argsort(-per_joint_max)
    print(f"[diag] split={a.split} N={E.shape[0]} cap={spec.cap}")
    print("[diag] 逐关节 |Δq| 的 max（rad）top-10：")
    for i in order[:10]:
        print(f"  j{i:02d} {names[i] if i < len(names) else '?':24s} "
              f"max={per_joint_max[i]:.4f} rad  rms={np.sqrt((E[:,:,i]**2).mean()):.5f} rad")
    print("[diag] 逐控制步的全局 max（rad）：",
          np.round(per_step_max, 3).tolist())
    # 找到最大样本
    flat = np.abs(E).reshape(-1)
    idx = int(np.argmax(flat))
    n_i, rest = divmod(idx, 20 * 29)
    h_i, j_i = divmod(rest, 29)
    print(f"[diag] 全局 peak={flat[idx]:.4f} rad @ 窗口{n_i} 步{h_i} 关节{j_i} "
          f"({names[j_i] if j_i < len(names) else '?'})")
    step_rms = step_rms_sum / max(1, step_n)
    div_all = np.concatenate(div, 0) if div else np.zeros(0)
    out = dict(split=a.split, n=int(E.shape[0]), cap=float(spec.cap),
               per_step_rms_rad=step_rms.tolist(),
               rollout_end_state_div_m=dict(p50=float(np.median(div_all)) if div_all.size else 0.0,
                                            p90=float(np.quantile(div_all, 0.9)) if div_all.size else 0.0,
                                            max=float(div_all.max()) if div_all.size else 0.0),
               per_joint_max_rad=per_joint_max.tolist(),
               per_joint_rms_rad=np.sqrt((E ** 2).mean(axis=(0, 1))).tolist(),
               per_step_max_rad=per_step_max.tolist(),
               joint_names=names,
               global_peak_rad=float(flat[idx]), peak_window=n_i, peak_step=h_i,
               peak_joint=j_i)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(f"[diag] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
