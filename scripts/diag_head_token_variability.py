#!/usr/bin/env python3
"""VLA action head 到底会不会输出**会变的 token**？——离线、不碰 Isaac。

为什么需要
----------
2026-11-04 官方闭环（DoubleDesk / SONIC release decoder / 600 步）实测：

    数据集 GT token      absmean 0.134  |t| 1.33  相邻帧 |Δt| 0.139  逐维时间 std 0.119
    闭环 base  head 输出  absmean 0.093  |t| 0.87  相邻帧 |Δt| 0.037  逐维时间 std 0.022
    闭环 FT    head 输出  absmean 0.090  |t| 0.86  相邻帧 |Δt| 0.053  逐维时间 std 0.023

token 几乎是**常数** ⇒ SONIC decoder 收到"站着别动" ⇒ 闭环净位移 0.017 m（参考要求 1.17 m/12 s）。
换贴图（TEXTURES=1）没用（|Δt| 0.0369→0.0381），所以不是灰模 OOD。

剩下两种可能，必须分开：
  (a) head 本身塌了（哪怕喂**训练集原图 + 训练集原 state46**也只吐均值 token）；
  (b) head 没问题，是**闭环那一路的输入**跟训练口径不一致（图像/state46 构造/时序）。

本脚本喂 (a) 的条件：直接从 `data/npz/ha_tokens/*.npz` 取 `g1_ego_view` + `g1_state_46`
连续帧，让 server 的 Engine 逐帧推，看 token 的时间变化率能不能到 GT 的量级。

用法::

    PYTHONPATH=... python3 scripts/diag_head_token_variability.py \
        --ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
        [--head-ckpt runs/joint_v3_round3A/vla_000250.pt] --n 60 --device cuda:1
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOKEN = "g1_motion_token_64"


def _read_video(path: str, start: int, n: int) -> np.ndarray:
    """按顺序解出 [start, start+n) 帧，uint8 RGB。"""
    if not Path(path).exists():
        raise SystemExit(f"ego view 视频不存在: {path}")
    frames = []
    try:
        import av  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"需要 av 解码 mp4（PYTHONPATH 里要有 third_party/gr00t_deps）: {exc}") from exc
    with av.open(path) as cont:
        for i, frame in enumerate(cont.decode(video=0)):
            if i < start:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= n:
                break
    if len(frames) < n:
        raise SystemExit(f"{path} 只解出 {len(frames)} 帧，要 {n}")
    return np.stack(frames, 0).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000")
    ap.add_argument("--head-ckpt", default="")
    ap.add_argument("--npz-glob", default="data/npz/ha_tokens/HOI_double_desk__tw__*.npz")
    ap.add_argument("--n", type=int, default=60, help="连续帧数")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--task", default="Pick up the object from the double desk and place it into the basket.")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--fm-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(args.npz_glob))
    if not files:
        raise SystemExit(f"没有匹配 {args.npz_glob} 的 npz")
    d = np.load(files[0], allow_pickle=True)
    n = min(args.n, len(d[TOKEN]) - args.start)
    # `g1_ego_view` 是一个 **mp4 路径**（0 维字符串数组），不是图像张量。
    img = _read_video(str(d["g1_ego_view"]), args.start, n)
    s46 = d["g1_state_46"][args.start: args.start + n].astype(np.float32)
    gt = d[TOKEN][args.start: args.start + n].astype(np.float64)
    print(f"[diag] {Path(files[0]).name}  frames {args.start}..{args.start + n - 1}  "
          f"img {img.shape} s46 {s46.shape}")

    # ---- server Engine 复用（不重写任何预处理）--------------------------------
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

    toks = []
    for i in range(n):
        chunk, _hands = eng.infer(np.asarray(img[i], dtype=np.uint8), np.zeros(64, np.float32),
                                  args.task, state46_in=s46[i])
        toks.append(np.asarray(chunk, np.float32)[0])
    T = np.array(toks, dtype=np.float64)

    def rep(name, X):
        dt = np.linalg.norm(np.diff(X, axis=0), axis=1).mean()
        print(f"  {name:<14} absmean {np.abs(X).mean():.4f}  |t| {np.linalg.norm(X, axis=1).mean():.3f}"
              f"  |dt| {dt:.4f}  std {X.std(0).mean():.4f}")

    print("[diag] 首帧 token 序列（同一 chunk 的第 0 帧）：")
    rep("GT", gt)
    rep("head", T)
    cos = float(T.mean(0) @ gt.mean(0) / (np.linalg.norm(T.mean(0)) * np.linalg.norm(gt.mean(0)) + 1e-12))
    print(f"  cos(mean_head, mean_GT) {cos:.3f}")
    print(f"  逐帧 MSE(head, GT) {float(((T - gt) ** 2).mean()):.5f}")
    # token 塌缩指数：head 的时间变化 / GT 的时间变化（1.0 = 一样会动，0 = 常数）
    print(f"  ★ 塌缩指数 |dt|_head/|dt|_GT = "
          f"{np.linalg.norm(np.diff(T, axis=0), axis=1).mean() / max(1e-9, np.linalg.norm(np.diff(gt, axis=0), axis=1).mean()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
