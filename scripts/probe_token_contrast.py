#!/usr/bin/env python3
"""S3 诊断：**想象回报对 token 的敏感度**（奖励反传有没有信号）。

只加载冻结的 WBC + WM（不需要 VLA，约 20 s），在同一批起点上把 conditioning 换成
{GT, 零, 随机, batch 内打乱, 全库均值}，比较 H 步想象回报。

为什么要这个探针：S3.4 的 burn-in 把起点做"脏"了，但**回报是否随 token 变化**是另一件事。
本轮实测（WBC v1/v2）GT 与零 token 差 <0.1%（窗口间噪声 4%），而随机 token 会让回报转负 ——
说明链路通、但冻结 WBC 基本不看 conditioning。这个数字决定了 λ_r 能不能开。

用法：
    python3 scripts/probe_token_contrast.py --windows 64 \
        --wbc-ckpt trackerlab-supermimic/logs/wbc_mix2_obs93/ckpt_016000_spec.pt \
        --token-space port --out runs/S3_gates/contrast_v2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SUPERMIMIC_HA_ALIGN", "1")
os.environ.setdefault(
    "SUPERMIMIC_ROBOT_USD",
    str(ROOT / "trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"),
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wbc-ckpt", type=Path,
                    default=ROOT / "trackerlab-supermimic/logs/wbc_mix2_obs93/ckpt_016000_spec.pt")
    ap.add_argument("--motion", type=Path, default=ROOT / "data/npz/ha_tokens")
    ap.add_argument("--motion-packed", type=Path, default=ROOT / "runs/isaac_motion/ha_tokens/packed.pt")
    ap.add_argument("--port-dir", type=Path, default=ROOT / "data/port_tokens/ha_tokens")
    ap.add_argument("--token-space", choices=("label", "port"), default="port",
                    help="GT 用哪个空间（必须与冻结 WBC 的训练空间一致）")
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--burnin", type=int, default=10)
    ap.add_argument("--no-burnin", action="store_true")
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    from diff_vla_wbc.burnin import sample_burnin
    from diff_vla_wbc.data import RecordedTokenSource
    from diff_vla_wbc.isaac_joint import HISTORY, load_isaac_bundle
    from diff_vla_wbc.rollout import discounted_return, imagine_with_tokens

    dev = torch.device(a.device)
    files = sorted(a.motion.glob("*.npz"))
    bundle = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=files,
                               packed_motion=a.motion_packed, token_dim=64)
    lib = bundle.library
    chunk_src = RecordedTokenSource(a.motion, lib, token_horizon=40, token_dim=64,
                                    device=dev, port_dir=a.port_dir, token_space=a.token_space)
    frame_src = RecordedTokenSource(a.motion, lib, token_horizon=1, token_dim=64,
                                    device=dev, port_dir=a.port_dir, token_space=a.token_space)
    print(f"[probe] wbc={a.wbc_ckpt} token_space={a.token_space} H={a.horizon} "
          f"K_burn={0 if a.no_burnin else a.burnin}", flush=True)

    g = torch.Generator(device="cpu")
    g.manual_seed(int(a.seed))
    max_start = lib.num_frames - a.horizon - 2
    variants = ("gt", "zeros", "randn", "shuffled", "mean")
    acc = {f"{k}_{m}": [] for k in variants for m in ("ret", "r0", "surv", "state")}
    n_done = 0
    with torch.no_grad():
        while n_done < a.windows:
            bs = min(a.batch, a.windows - n_done)
            frames = torch.randint(HISTORY, max_start, (bs,), generator=g, dtype=torch.long).to(dev)
            start = bundle.reference_start(lib, frames)
            if a.no_burnin:
                start_state, hist, prev, xi = (
                    start["state"], start["observations"], start["prev_action"], start["xi"])
            else:
                bi = sample_burnin(
                    bundle, lib, frames,
                    port_tokens=lambda f: frame_src({"frames": f})[:, 0],
                    horizon=a.horizon, k=a.burnin, generator=None, perturb=True)
                start_state, hist, prev, xi = bi.state, bi.history, bi.prev_action, bi.xi
            refs = bundle.teacher_references(lib, frames, a.horizon)
            tstates = bundle.teacher_states(lib, frames, a.horizon)
            gt = chunk_src({"frames": frames})
            mean_tok = gt.mean(0, keepdim=True).expand_as(gt).contiguous()
            perm = torch.randperm(bs, device=dev)
            toks = {
                "gt": gt,
                "zeros": torch.zeros_like(gt),
                "randn": torch.randn_like(gt),
                "shuffled": gt[perm],
                "mean": mean_tok,
            }
            for k, tok in toks.items():
                ro = imagine_with_tokens(
                    bundle, tokens=tok, state=start_state, obs_hist=None, history=hist,
                    xi=xi, references=refs, prev_action=prev, horizon=a.horizon, obs_noise=0.0)
                ret = discounted_return(ro, 0.99)
                acc[f"{k}_ret"].append(ret.mean().item())
                acc[f"{k}_r0"].append(ro.rewards[0].mean().item())
                acc[f"{k}_surv"].append(ro.survival[-1].mean().item())
                d = (ro.states[-1, :, :3] - tstates[:, -1, :3]).norm(dim=-1)
                acc[f"{k}_state"].append(d.mean().item())
            n_done += bs

    def m(k: str) -> float:
        return float(np.mean(acc[k]))

    def se(k: str) -> float:
        v = np.asarray(acc[k], np.float64)
        return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")

    contrast = m("gt_ret") - m("zeros_ret")
    paired = np.asarray(acc["gt_ret"]) - np.asarray(acc["zeros_ret"])
    paired_se = float(paired.std(ddof=1) / np.sqrt(len(paired))) if len(paired) > 1 else float("nan")
    out = {
        "wbc_ckpt": str(a.wbc_ckpt), "token_space": a.token_space,
        "horizon": a.horizon, "burnin": 0 if a.no_burnin else a.burnin,
        "windows": n_done,
        "mean": {k: round(m(k), 6) for k in acc},
        "se": {k: round(se(k), 6) for k in acc},
        "contrast_gt_minus_zeros": round(float(contrast), 6),
        "contrast_rel": round(float(contrast / max(abs(m("gt_ret")), 1e-9)), 6),
        "paired_se": round(paired_se, 6),
        "paired_t": round(float(contrast / paired_se), 3) if paired_se and paired_se > 0 else None,
        "gt_wins_frac": round(float((paired > 0).mean()), 4),
    }
    print(json.dumps(out, indent=1))
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
        print(f"[probe] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
