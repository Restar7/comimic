#!/usr/bin/env python3
"""2×2 交叉实验：闭环 token 塌缩，是**图像**那一路还是**状态**那一路？

离线结果已经排除 state46 布局（四种拼法 |dt| 都 0.19–0.23，都接近/超过 GT 的 0.09），
所以嫌疑只剩图像。本脚本把闭环抓包里的 (rgb, state64) 和录制帧的 (rgb, g1_state_46)
交叉配对，看哪一路决定 chunk 内部的时间变化：

    (闭环图, 闭环状态) / (录制图, 闭环状态) / (闭环图, 录制状态) / (录制图, 录制状态)

用法::

    PYTHONPATH=... python3 scripts/diag_2x2_image_state.py \
        --probe runs/closedloop_r3/probe/srv/req_000001.npz --device cuda:1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_video(path: str, start: int, n: int) -> np.ndarray:
    import av  # noqa: PLC0415

    frames = []
    with av.open(path) as cont:
        for i, frame in enumerate(cont.decode(video=0)):
            if i < start:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= n:
                break
    return np.stack(frames, 0).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000")
    ap.add_argument("--head-ckpt", default="")
    ap.add_argument("--probe", default="runs/closedloop_r3/probe/srv/req_000001.npz")
    ap.add_argument("--npz", default="data/npz/ha_tokens/HOI_double_desk__tw__ep000000.npz")
    ap.add_argument("--task", default="Pick up the object from the double desk and place it into the basket.")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--fm-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default="runs/closedloop_r3/imgcmp")
    args = ap.parse_args()

    p = np.load(args.probe, allow_pickle=True)
    rgb_loop, s64_loop, s46_loop = p["rgb"], p["state64"], p["state46"]
    chunk_loop = p["chunk64"]

    d = np.load(args.npz, allow_pickle=True)
    rgb_rec = _read_video(str(d["g1_ego_view"]), 0, 1)[0]
    s46_rec = d["g1_state_46"][0].astype(np.float32)
    gt_tok = d["g1_motion_token_64"][0].astype(np.float64)
    gt_chunk = d["g1_motion_token_64"][:40].astype(np.float64)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{args.save_dir}/loop.npy", rgb_loop)
    np.save(f"{args.save_dir}/rec.npy", rgb_rec)
    np.save(f"{args.save_dir}/loop_state64.npy", s64_loop)

    def stats(name, im):
        print(f"  {name:<8} shape {im.shape} mean {im.mean():6.1f} std {im.std():6.1f} "
              f"|Δ| vs 对方 {np.abs(im.astype(np.float32) - im2.astype(np.float32)).mean():6.1f}")

    im2 = rgb_loop if False else rgb_loop
    stats("闭环图", rgb_loop)
    im2 = rgb_rec
    stats("录制图", rgb_rec)

    from scripts.ha_vla_latent64_server import Engine  # noqa: PLC0415

    ns = argparse.Namespace(
        ckpt=Path(args.ckpt), vlm_path=ROOT / "groot_repro" / "nvidia" / "Cosmos-Reason2-2B-weights",
        device=args.device, bf16=False, horizon=40, action_dim=78, fm_steps=args.fm_steps,
        token_quant="none", sample_avg=1, decode="codec", state_order="ha_v31",
        state_layout="legacy", hand_source="feedback", dry_run=False, log_dir="",
        head_ckpt=args.head_ckpt, lora_rank=32, lora_targets="attn1,attn2,ff",
        lora_also_train="proj_out", lead_beta=0.0, seed=args.seed, verbose=False, port=0,
        host="127.0.0.1",
    )
    eng = Engine(ns)

    print("\n[2x2] 每个格子只看 chunk(40 帧) 内部的时间变化 |dt|：")
    print(f"  闭环实际 chunk: |dt| {np.linalg.norm(np.diff(chunk_loop, axis=0), axis=1).mean():.4f}")
    print(f"  录制 GT  chunk: |dt| {np.linalg.norm(np.diff(gt_chunk, axis=0), axis=1).mean():.4f}")

    res = {}
    for img_name, img in (("闭环图", rgb_loop), ("录制图", rgb_rec)):
        for st_name, s46 in (("闭环状态", s46_loop), ("录制状态", s46_rec)):
            chunk, _ = eng.infer(np.asarray(img, np.uint8), s64_loop, args.task, state46_in=s46)
            chunk = np.asarray(chunk, np.float32).astype(np.float64)
            dt = np.linalg.norm(np.diff(chunk, axis=0), axis=1).mean()
            mse = float(((chunk[0] - gt_tok) ** 2).mean())
            res[(img_name, st_name)] = dt
            print(f"  ({img_name}, {st_name})  chunk |dt| {dt:.4f}   tok[0] vs GT[0] mse {mse:.5f}"
                  f"   |tok0| {np.linalg.norm(chunk[0]):.3f}")
    print("\n  结论：|dt| 随【图】翻倍 ⇒ 图像那一路；随【状态】翻倍 ⇒ 状态那一路。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
