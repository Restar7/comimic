#!/usr/bin/env python3
"""证明 / 证伪：部署侧 `state64_to_state46` 的布局选错了，把 head 打成"吐均值 token"。

2026-11-04 的证据链
-------------------
* 闭环抓包（`--log-dir`）里 `state64[6:35]` 与录制 `joint_pos` **逐位相同**（provider 发得对）。
* 但 `state46` 拼成了 `[jp[0:22] | left_hand | jp[22:29] | right_hand | gravity]`
  （server 默认 `--state-layout train_v1`）：
    - `state46[22:29]` = 手（部署时全 0）⇒ **7 个关节被清零**；
    - `state46[29:36]` = `jp[22:29]` ⇒ 被当成 left_hand；
    - 也就是说 46 维里有 21 维是错的。
* 而训练数据 `data/npz/ha_tokens/*.npz::g1_state_46` 是 **`[joints29 | hands14 | gravity3]`**
  （`g1_state_46[:29]` 与 `joint_pos` 逐位相同，`[29:43]` 是手）。

本脚本喂同一段录制帧，只换 state46 的拼法，看 token 时间变化率：

    PYTHONPATH=... python3 scripts/diag_state46_layout.py --n 40 --device cuda:1
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
    import av  # noqa: PLC0415

    frames = []
    with av.open(path) as cont:
        for i, frame in enumerate(cont.decode(video=0)):
            if i < start:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= n:
                break
    if len(frames) < n:
        raise SystemExit(f"只解出 {len(frames)} 帧，要 {n}")
    return np.stack(frames, 0).astype(np.uint8)


def _quat_wxyz_to_rot6d(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    # 与 HA 的 vla_state_root_rot6d 同口径：取前两列（col-major 6 维）
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000")
    ap.add_argument("--head-ckpt", default="")
    ap.add_argument("--npz", default="data/npz/ha_tokens/HOI_double_desk__tw__ep000000.npz")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--task", default="Pick up the object from the double desk and place it into the basket.")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--fm-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    n = min(args.n, len(d[TOKEN]))
    img = _read_video(str(d["g1_ego_view"]), 0, n)
    g46 = d["g1_state_46"][:n].astype(np.float32)
    gt = d[TOKEN][:n].astype(np.float64)
    joints = g46[:, :29]
    hands = g46[:, 29:43]
    quat = d["body_quat_w"][:n, 0]                     # (n,4) wxyz
    rot6d = np.stack([_quat_wxyz_to_rot6d(q) for q in quat], 0)
    s64 = np.concatenate([rot6d, joints, np.zeros_like(joints)], 1).astype(np.float32)

    from scripts.ha_vla_latent64_server import Engine, state64_to_state46  # noqa: PLC0415

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

    def run(s46_seq, name):
        T = []
        for i in range(n):
            chunk, _ = eng.infer(np.asarray(img[i], np.uint8), s64[i], args.task,
                                 state46_in=s46_seq[i])
            T.append(np.asarray(chunk, np.float32)[0])
        T = np.array(T, np.float64)
        dt = np.linalg.norm(np.diff(T, axis=0), axis=1).mean()
        cos = float(T.mean(0) @ gt.mean(0) /
                    (np.linalg.norm(T.mean(0)) * np.linalg.norm(gt.mean(0)) + 1e-12))
        mse = float(((T - gt) ** 2).mean())
        print(f"  {name:<44} |t| {np.linalg.norm(T, axis=1).mean():.3f}  "
              f"|dt| {dt:.4f}  cos {cos:+.3f}  mse {mse:.5f}")
        return T

    print("[layout] 同一段录制帧，只换 state46 的拼法（GT |dt| = %.4f）:"
          % np.linalg.norm(np.diff(gt, axis=0), axis=1).mean())
    run(g46, "(a) 训练真值 g1_state_46（legacy + 真手）")
    run(np.stack([state64_to_state46(s, np.zeros(14, np.float32), "ha_v31", "legacy")
                  for s in s64]), "(b) legacy 布局 + 手补 0（部署修复后）")
    run(np.stack([state64_to_state46(s, np.zeros(14, np.float32), "ha_v31", "train_v1")
                  for s in s64]), "(c) train_v1 布局 + 手补 0（当前闭环默认）★")
    run(np.stack([state64_to_state46(s, h, "ha_v31", "legacy")
                  for s, h in zip(s64, hands)]), "(d) legacy 布局 + 真手（最理想）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
