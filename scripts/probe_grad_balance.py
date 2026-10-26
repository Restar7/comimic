#!/usr/bin/env python3
"""S3 诊断：**奖励梯度到底有多大**（以及和 v1 的 sample-MSE 锚定比）。

背景：`grad_ratio = ‖∇L_r‖/‖∇L_bc‖` 只有 0.15–0.3，容易被读成"奖励反传被取消了"。
本探针把**绝对梯度范数**量出来，并同时算 v1（已经作废的）两种监督的梯度：

    L_r_norm = -J/(H·MAX_STEP_REWARD)      # v2 的奖励项（归一化回报）
    L_r_v1   = -J                          # v1 原来用的（未归一化）
    L_bc     = GR00T 原生 FM 损失           # v2 的 BC 主项
    L_anchor = MSE(sample, label_tokens)    # v1 的 token_anchor（sample-MSE，已判为错）

输出：每个损失的 ‖∇‖（同一组 head 参数、同一 batch、同一采样），以及两两比值。

    python3 scripts/probe_grad_balance.py --wbc-ckpt runs/S1_handover/WBC_A20000_wmvel_cmd.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
                    default=ROOT / "runs/S1_handover/WBC_A20000_wmvel_cmd.pt")
    ap.add_argument("--vlm-ckpt", type=Path,
                    default=ROOT / "runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000")
    ap.add_argument("--motion", type=Path, default=ROOT / "data/npz/ha_tokens")
    ap.add_argument("--motion-packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--fm-steps", type=int, default=4)
    ap.add_argument("--sub-batches", type=int, default=3, help="重复几次取平均（不同窗口）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/S3_gates/grad_balance.json")
    a = ap.parse_args()

    from diff_vla_wbc.action_codec import ActionCodec
    from diff_vla_wbc.data import RecordedTokenSource
    from diff_vla_wbc.fm_bc import bc_target, fm_bc_loss, sample_once_with_grad, grad_ratio_parameters
    from diff_vla_wbc.g1_tasks import TaskFrameIndex, TaskObservationBuilder, load_gr00t_processor, processed_batch
    from diff_vla_wbc.gr00t_vlm import Gr00tVLM
    from diff_vla_wbc.isaac_joint import load_isaac_bundle
    from diff_vla_wbc.rollout import discounted_return, imagine_with_tokens
    from diff_vla_wbc.train_v2 import BurnInBatchProvider

    dev = torch.device(a.device)
    files = sorted(a.motion.glob("*.npz"))
    bundle = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=files,
                               packed_motion=a.motion_packed, token_dim=64)
    processor = load_gr00t_processor(a.vlm_ckpt)
    codec = ActionCodec(processor, device=dev)
    vlm = Gr00tVLM.from_pretrained(a.vlm_ckpt, device=dev, action_horizon=40, max_state_dim=132,
                                   freeze_vlm=True, attn_implementation="sdpa")
    vlm.action_head.eval()
    builder = TaskObservationBuilder(TaskFrameIndex(files))

    from diff_vla_wbc.sonic_frontend import LibraryTokenBuilder

    online = LibraryTokenBuilder(bundle.motion, device=dev, window="hold", library_path=a.motion_packed)
    _tdir = a.motion
    chunk_src = RecordedTokenSource(_tdir, bundle.library, token_horizon=40, token_dim=64,
                                    device=dev, token_space="label")
    frame_src = RecordedTokenSource(_tdir, bundle.library, token_horizon=1, token_dim=64,
                                    device=dev, token_space="label")
    provider = BurnInBatchProvider(
        bundle, bundle.library,
        port_source=lambda f: chunk_src({"frames": f}),
        port_frame_source=lambda f: frame_src({"frames": f})[:, 0],
        online_tokens=online,
        batch_size=a.batch, horizon=a.horizon, seed=7, hand_dir=_tdir,
    )
    params = grad_ratio_parameters(vlm.action_head)
    print(f"[probe] params_subset={sum(p.numel() for p in params):,}  batch={a.batch} H={a.horizon} K={a.fm_steps}",
          flush=True)

    acc: dict[str, list[float]] = {}

    def add(k, v):
        acc.setdefault(k, []).append(float(v))

    for i in range(a.sub_batches):
        b = provider.sample()
        b.model_inputs = processed_batch(processor, builder, b.frames)
        bb_in, act_in = vlm.prepare_input(dict(b.model_inputs))
        with torch.no_grad():
            bb_out = vlm.backbone(bb_in)
        from diff_vla_wbc.fm_bc import copy_backbone_output, copy_action_input

        head = vlm.action_head
        K, dim = int(a.fm_steps), int(head.action_dim)
        noise = torch.randn(int(act_in["state"].shape[0]), int(head.action_horizon), dim,
                            device=dev, dtype=head.dtype)
        x_norm = sample_once_with_grad(head, bb_out, act_in, initial_noise=noise, num_steps=K)
        x_raw = codec.decode(x_norm)
        tokens = x_raw[..., :64]
        ro = imagine_with_tokens(bundle, tokens=tokens, state=b.start_state, obs_hist=b.obs_hist,
                                 history=b.history, xi=b.xi, references=b.references,
                                 prev_action=b.prev_action, horizon=a.horizon, obs_noise=0.0)
        J = discounted_return(ro, 0.99).mean()
        from diff_vla_wbc.train_v2 import _MAX_STEP_REWARD

        loss_r_norm = -J / (a.horizon * float(_MAX_STEP_REWARD))
        loss_r_v1 = -J
        tgt, mask = bc_target(codec, b.port_tokens, b.hand_ref, head_dim=dim,
                              dtype=head.dtype, device=dev)
        loss_bc = fm_bc_loss(head, copy_backbone_output(bb_out), copy_action_input(act_in), tgt, mask)
        n = min(tokens.shape[1], b.port_tokens.shape[1])
        loss_anchor = torch.nn.functional.mse_loss(tokens[:, :n].float(),
                                                  b.port_tokens[:, :n].float())

        def norm(loss):
            g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            return float(torch.sqrt(sum((x.detach().float() ** 2).sum() for x in g if x is not None)))

        add("L_bc", float(loss_bc.detach())); add("g_bc", norm(loss_bc))
        add("L_r_norm", float(loss_r_norm.detach())); add("g_r_norm", norm(loss_r_norm))
        add("L_r_v1", float(loss_r_v1.detach())); add("g_r_v1", norm(loss_r_v1))
        add("L_anchor", float(loss_anchor.detach())); add("g_anchor", norm(loss_anchor))
        print(f"  [{i}] L_bc={float(loss_bc):.4f} L_r_norm={float(loss_r_norm):.3f} "
              f"L_anchor={float(loss_anchor):.5f}", flush=True)

    m = {k: sum(v) / len(v) for k, v in acc.items()}
    out = {
        "wbc_ckpt": str(a.wbc_ckpt), "batch": a.batch, "horizon": a.horizon,
        "fm_steps": a.fm_steps, "n_repeats": a.sub_batches,
        "params_subset": int(sum(p.numel() for p in params)),
        "means": {k: round(v, 8) for k, v in m.items()},
        "grad_ratio_r_norm_over_bc": round(m["g_r_norm"] / max(m["g_bc"], 1e-12), 5),
        "grad_ratio_r_v1_over_bc": round(m["g_r_v1"] / max(m["g_bc"], 1e-12), 5),
        "grad_ratio_anchor_over_bc": round(m["g_anchor"] / max(m["g_bc"], 1e-12), 5),
        "grad_ratio_r_norm_over_anchor": round(m["g_r_norm"] / max(m["g_anchor"], 1e-12), 5),
    }
    print(json.dumps(out, indent=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"[probe] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
