#!/usr/bin/env python3
"""S3 headroom 判定：**token 到底有没有因果影响**，以及 VLA 离 GT 有多远。

回答一个问题：*在哪个想象场景下，token 才对结果有区分度？* 只测量、不训练。

三臂（同一起点、同一参考、同一 H）：
  gt   : hold builder + 起点状态的仿真 anchor（WBC 训练口径）
  vla  : 官方 VLA（checkpoint-8000）在同一起点观测上采出来的 token（decode 后）
  zeros: 零 token（负对照）

四格：burn-in 起点扰动 {1x, 5x} × 想象步数 {16, 32}
  （1x = xy 5 cm / yaw 0.1 rad；5x = 25 cm / 0.5 rad，WBC 自己救不回来的量级）

判读：gt-vs-zeros 就是"token 有没有被使用"；gt-vla 是"VLA 离最优有多远"= **headroom**。
headroom 太小 → 任何算法（含 critic/BC）都没有东西可学。

    python3 scripts/probe_headroom.py --windows 32 --device cuda:0
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


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wbc-ckpt", type=Path, default=ROOT / "runs/S1_handover/WBC_A20000_wmvel_cmd.pt")
    ap.add_argument("--vlm-ckpt", type=Path,
                    default=ROOT / "runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000")
    ap.add_argument("--vlm-path", type=Path,
                    default=ROOT / "groot_repro/nvidia/Cosmos-Reason2-2B-weights")
    ap.add_argument("--motion", type=Path, default=ROOT / "data/npz/ha_tokens")
    ap.add_argument("--motion-packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--burnin", type=int, default=10)
    ap.add_argument("--perturb-scales", type=float, nargs="+", default=[1.0, 5.0])
    ap.add_argument("--horizons", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--fm-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/S3_gates/headroom.json")
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
    online = LibraryTokenBuilder(bundle.motion, device=dev, window="hold",
                                 library_path=a.motion_packed)
    frame_src = RecordedTokenSource(a.motion, lib, token_horizon=1, token_dim=64,
                                   device=dev, token_space="label")
    chunk_src = RecordedTokenSource(a.motion, lib, token_horizon=40, token_dim=64,
                                    device=dev, token_space="label")
    print(f"[headroom] wbc={a.wbc_ckpt.name} windows={a.windows} "
          f"scales={a.perturb_scales} horizons={a.horizons}", flush=True)

    # ---------------- 固定窗口 + 官方 VLA 的 token（只算一次，跨格复用）----------------
    g = torch.Generator(device="cpu"); g.manual_seed(int(a.seed))
    max_start = lib.num_frames - max(a.horizons) - 2
    frames_all = torch.randint(HISTORY, max_start, (a.windows,), generator=g, dtype=torch.long).to(dev)

    vla_tokens = None
    if a.vlm_ckpt is not None and Path(a.vlm_ckpt).exists():
        processor = load_gr00t_processor(a.vlm_ckpt)
        codec = ActionCodec(processor, device=dev)
        vlm = Gr00tVLM.from_pretrained(a.vlm_path, checkpoint=a.vlm_ckpt, device=dev,
                                       action_horizon=40, max_state_dim=132,
                                       freeze_vlm=True, attn_implementation="sdpa")
        vlm.action_head.eval()
        builder = TaskObservationBuilder(TaskFrameIndex(files))
        outs = []
        for s in range(0, a.windows, a.batch):
            fr = frames_all[s:s + a.batch]
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
            outs.append(codec.decode(x_norm.detach())[..., :64].float())
            print(f"  [vla] {s + len(fr)}/{a.windows} windows", flush=True)
        vla_tokens = torch.cat(outs, 0)
        with torch.no_grad():
            lab = chunk_src({"frames": frames_all})
            n = min(lab.shape[1], vla_tokens.shape[1])
            print(f"  [vla] token_mse(vs label) = {float(((vla_tokens[:, :n] - lab[:, :n]) ** 2).mean()):.5f}",
                  flush=True)
        del vlm
        torch.cuda.empty_cache()

    # ---------------- 四格 ----------------
    rows = []
    for scale in a.perturb_scales:
        for H in a.horizons:
            rets = {"gt": [], "vla": [], "zeros": []}
            surv = {"gt": [], "vla": [], "zeros": []}
            for s in range(0, a.windows, a.batch):
                fr = frames_all[s:s + a.batch]
                bi = sample_burnin(
                    bundle, lib, fr,
                    port_tokens=lambda f: frame_src({"frames": f})[:, 0],
                    token_fn=online.__call__, horizon=H, k=a.burnin, perturb=True,
                    xy=0.05 * scale, yaw=0.10 * scale, dq=0.05 * scale, dqd=0.30 * scale,
                )
                idx = fr[:, None] + torch.arange(H, device=dev)[None, :]
                gt = online.chunk(idx, robot_quat=bi.state[:, 3:7])
                toks = {"gt": gt, "zeros": torch.zeros_like(gt)}
                if vla_tokens is not None:
                    toks["vla"] = vla_tokens[s:s + fr.numel()]
                for k, tok in toks.items():
                    ro = imagine_with_tokens(bundle, tokens=tok, state=bi.state, obs_hist=None,
                                             history=bi.history, xi=bi.xi, references=bi.references,
                                             prev_action=bi.prev_action, horizon=H, obs_noise=0.0)
                    rets[k].append(float(discounted_return(ro, 0.99).mean()))
                    surv[k].append(float(ro.survival[-1].mean()))
            m = {f"ret_{k}": float(np.mean(v)) for k, v in rets.items()}
            sv = {f"surv_{k}": float(np.mean(v)) for k, v in surv.items()}
            m.update(sv)
            m["scale"] = scale
            m["H"] = H
            m["contrast_gt_minus_zeros"] = m["ret_gt"] - m["ret_zeros"]
            m["contrast_rel"] = (m["ret_gt"] - m["ret_zeros"]) / max(abs(m["ret_gt"]), 1e-9)
            if "ret_vla" in m:
                m["headroom_gt_minus_vla"] = m["ret_gt"] - m["ret_vla"]
                m["headroom_rel"] = (m["ret_gt"] - m["ret_vla"]) / max(abs(m["ret_gt"]), 1e-9)
            rows.append(m)
            print(f"[cell] scale={scale:g}x H={H:2d} | ret gt={m['ret_gt']:.4f} "
                  f"vla={m.get('ret_vla', float('nan')):.4f} zeros={m['ret_zeros']:.4f} "
                  f"| token contrast={m['contrast_rel']:+.1%} "
                  f"| headroom(gt-vla)={m.get('headroom_rel', float('nan')):+.1%} "
                  f"| surv gt={m['surv_gt']:.4f}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"wbc": str(a.wbc_ckpt), "windows": a.windows,
                                 "burnin": a.burnin, "rows": rows}, indent=1))
    print(f"[headroom] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
