#!/usr/bin/env python3
"""端到端上限验证（oracle）：**奖励梯度最多能把执行误差纠回多少**。

从"偏离起点"（burn-in 大扰动）出发，直接对 64 维 token 做奖励梯度上升，
得到"上帝视角最优 token"，再和标签/VLA token 比执行回报。

    差（oracle − label）= 端到端反传的天花板优势
    差 ≈ 0            = 端到端在这套执行器上纠不了误差（先改接口，不是调权重）

同时报两种 oracle：
  oracle_free    : 只最大化回报（可以看到它漂到多离谱 —— 上界）
  oracle_anchor  : 回报 + λ·‖T − label‖²（λ=0.1，带轻度流形约束的现实上界）

    python3 scripts/probe_token_oracle.py --windows 16 --horizon 32 --scale 5 --device cuda:0
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
    ap.add_argument("--wbc-ckpt", type=Path, default=ROOT / "runs/S1_handover/WBC_A20000_wmvel_cmd.pt")
    ap.add_argument("--vlm-ckpt", type=Path,
                    default=ROOT / "runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000")
    ap.add_argument("--vlm-path", type=Path, default=ROOT / "groot_repro/nvidia/Cosmos-Reason2-2B-weights")
    ap.add_argument("--motion", type=Path, default=ROOT / "data/npz/ha_tokens")
    ap.add_argument("--motion-packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--burnin", type=int, default=10)
    ap.add_argument("--scale", type=float, default=5.0, help="burn-in 扰动倍数（1x = 5cm/0.1rad）")
    ap.add_argument("--opt-steps", type=int, default=60)
    ap.add_argument("--opt-lr", type=float, default=0.05)
    ap.add_argument("--anchor-lambda", type=float, default=0.1)
    ap.add_argument("--fm-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/S3_gates/oracle.json")
    a = ap.parse_args()

    from diff_vla_wbc.action_codec import ActionCodec
    from diff_vla_wbc.burnin import sample_burnin
    from diff_vla_wbc.data import RecordedTokenSource
    from diff_vla_wbc.fm_bc import sample_once_with_grad
    from diff_vla_wbc.g1_tasks import (TaskFrameIndex, TaskObservationBuilder,
                                       load_gr00t_processor, processed_batch)
    from diff_vla_wbc.gr00t_vlm import Gr00tVLM
    from diff_vla_wbc.isaac_joint import HISTORY, load_isaac_bundle
    from diff_vla_wbc.rollout import discounted_return, imagine_with_tokens
    from diff_vla_wbc.sonic_frontend import LibraryTokenBuilder

    dev = torch.device(a.device)
    files = sorted(a.motion.glob("*.npz"))
    bundle = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=files,
                               packed_motion=a.motion_packed, token_dim=64)
    lib = bundle.library
    online = LibraryTokenBuilder(bundle.motion, device=dev, window="hold", library_path=a.motion_packed)
    frame_src = RecordedTokenSource(a.motion, lib, token_horizon=1, token_dim=64,
                                   device=dev, token_space="label")
    chunk_src = RecordedTokenSource(a.motion, lib, token_horizon=40, token_dim=64,
                                    device=dev, token_space="label")

    processor = load_gr00t_processor(a.vlm_ckpt)
    codec = ActionCodec(processor, device=dev)
    vlm = Gr00tVLM.from_pretrained(a.vlm_path, checkpoint=a.vlm_ckpt, device=dev,
                                   action_horizon=40, max_state_dim=132,
                                   freeze_vlm=True, attn_implementation="sdpa")
    vlm.action_head.eval()
    builder = TaskObservationBuilder(TaskFrameIndex(files))
    lo = codec.raw_min[:64].copy(); hi = codec.raw_max[:64].copy()
    lo_t = torch.tensor(lo, device=dev, dtype=torch.float32)
    hi_t = torch.tensor(hi, device=dev, dtype=torch.float32)

    g = torch.Generator(device="cpu"); g.manual_seed(int(a.seed))
    max_start = lib.num_frames - a.horizon - 2
    frames = torch.randint(HISTORY, max_start, (a.windows,), generator=g, dtype=torch.long).to(dev)

    rows = []
    for s in range(0, a.windows, a.batch):
        fr = frames[s:s + a.batch]
        bi = sample_burnin(bundle, lib, fr, port_tokens=lambda f: frame_src({"frames": f})[:, 0],
                           token_fn=online.__call__, horizon=a.horizon, k=a.burnin, perturb=True,
                           xy=0.05 * a.scale, yaw=0.10 * a.scale, dq=0.05 * a.scale, dqd=0.30 * a.scale)
        idx = fr[:, None] + torch.arange(a.horizon, device=dev)[None, :]
        label = chunk_src({"frames": fr})[:, :a.horizon]
        gt = online.chunk(idx, robot_quat=bi.state[:, 3:7])

        # VLA token（正常观测）
        batch = processed_batch(processor, builder, fr)
        bb_in, act_in = vlm.prepare_input(dict(batch))
        with torch.no_grad():
            bb_out = vlm.backbone(bb_in)
        head = vlm.action_head
        noise = torch.randn(int(act_in["state"].shape[0]), int(head.action_horizon),
                            int(head.action_dim), device=dev, dtype=head.dtype,
                            generator=torch.Generator(device=dev).manual_seed(int(a.seed)))
        with torch.enable_grad():
            x_norm = sample_once_with_grad(head, bb_out, act_in, initial_noise=noise,
                                           num_steps=a.fm_steps)
        vla = codec.decode(x_norm.detach())[..., :64].float()[:, :a.horizon]

        def ret_of(tok):
            ro = imagine_with_tokens(bundle, tokens=tok, state=bi.state, obs_hist=None,
                                     history=bi.history, xi=bi.xi, references=bi.references,
                                     prev_action=bi.prev_action, horizon=a.horizon, obs_noise=0.0)
            return discounted_return(ro, 0.99), ro

        def optimise(anchor):
            T = torch.nn.Parameter(gt.clone())
            opt = torch.optim.Adam([T], lr=a.opt_lr)
            best = -1e9
            for _ in range(a.opt_steps):
                opt.zero_grad(set_to_none=True)
                r, _ = ret_of(T)
                loss = -r.mean()
                if anchor:
                    loss = loss + a.anchor_lambda * ((T - label) ** 2).mean()
                loss.backward(); opt.step()
                with torch.no_grad():
                    T.clamp_(min=lo_t, max=hi_t)
                    best = max(best, float(r.mean()))
            return T.detach(), best

        r_gt, _ = ret_of(gt)
        r_vla, _ = ret_of(vla)
        T_free, _ = optimise(anchor=False)
        T_anch, _ = optimise(anchor=True)
        r_free, _ = ret_of(T_free)
        r_anch, _ = ret_of(T_anch)
        drift = lambda T: float((T - label).abs().mean())
        row = {
            "n": int(fr.numel()),
            "ret_gt": float(r_gt.mean()), "ret_vla": float(r_vla.mean()),
            "ret_oracle_free": float(r_free.mean()), "ret_oracle_anchor": float(r_anch.mean()),
            "drift_gt": drift(gt), "drift_vla": drift(vla),
            "drift_oracle_free": drift(T_free), "drift_oracle_anchor": drift(T_anch),
        }
        rows.append(row)
        print("[oracle] " + json.dumps({k: round(v, 5) for k, v in row.items()}), flush=True)

    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "n"}
    m["headroom_oracle_free_rel"] = (m["ret_oracle_free"] - m["ret_gt"]) / max(abs(m["ret_gt"]), 1e-9)
    m["headroom_oracle_anchor_rel"] = (m["ret_oracle_anchor"] - m["ret_gt"]) / max(abs(m["ret_gt"]), 1e-9)
    m["scale"] = a.scale; m["horizon"] = a.horizon; m["windows"] = a.windows
    print("[oracle][summary] " + json.dumps({k: round(v, 5) for k, v in m.items()}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"rows": rows, "summary": m}, indent=1))
    print(f"[oracle] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
