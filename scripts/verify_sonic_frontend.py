#!/usr/bin/env python3
"""S1 / W6 收尾：用 HA **录制数据**把 SONIC encoder 的前端逐位对上，并给出可复现配方。

S0.4 的结论是"标签 = encoder_latent（逐位相等），端口侧车装配错"。本脚本把"错在哪"钉死，
并验证修好后的配方（供 `SonicTokenBuilder` / 侧车重建使用）：

1. 从 npz 的**已记录字段**重建 1762 维 `encoder_input`（hold 窗口 + anchor），跑 encoder
   → 必须与 `encoder_latent` 逐位相等（否则偏移/顺序还不对）；
2. 用**运动库视角**能拿到的量重建同一份输入（`joint_pos/joint_vel` + anchor）→ 与
   `g1_motion_token_64` 比对（这决定 builder 能不能给 BONES-SEED 造同类 token）；
3. 对比 legacy "未来 10 帧" 窗口 vs 录制用的 "hold 当前帧" 窗口，给出两者的 token 差。

    python3 scripts/verify_sonic_frontend.py --clips 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diff_vla_wbc.sonic_encoder import (  # noqa: E402
    ANCHOR_MF_OFFSET, COMMAND_FRAMES, COMMAND_OFFSET, G1MotionEncoder, mjcf_to_isaaclab_perm, rot6d,
)

NPZ_DIR = ROOT / "data/npz/ha_tokens"


def _stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, np.float64).reshape(len(a), -1)
    b = np.asarray(b, np.float64).reshape(len(b), -1)
    d = float(np.abs(a - b).max())
    ac, bc = a - a.mean(0), b - b.mean(0)
    corr = float(((ac * bc).mean(0) / (ac.std(0) * bc.std(0) + 1e-12)).mean())
    return d, corr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--frames", type=int, default=5, help="每个 clip 抽查的帧数")
    ap.add_argument("--encoder", default=None)
    a = ap.parse_args()

    enc = G1MotionEncoder.from_onnx(Path(a.encoder) if a.encoder else None)
    perm = mjcf_to_isaaclab_perm()
    print(f"[frontend] encoder={enc.info()['layers']} perm={perm.tolist()}")

    files = sorted(NPZ_DIR.glob("*.npz"))[: a.clips]
    rows = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        raw = np.load(str(z["source"]), allow_pickle=True)
        q, qd = z["joint_pos"], z["joint_vel"]
        label = z["g1_motion_token_64"]
        latent = raw["encoder_latent"]
        T = min(len(q), len(label), len(latent))
        anchor_rec = raw["consumed_anchor_rot6d"][:T]                     # (T,6) 录制时的 anchor
        q_isa = q[:, perm].astype(np.float32)
        qd_isa = qd[:, perm].astype(np.float32)

        # ---- 1. 用录制字段重建 1762 输入 → 必须等于 encoder_latent ----
        cmd_hold = np.concatenate(
            [np.repeat(q_isa[:, None, :], COMMAND_FRAMES, 1).reshape(T, -1),
             np.repeat(qd_isa[:, None, :], COMMAND_FRAMES, 1).reshape(T, -1)], 1)
        anc_hold = np.repeat(anchor_rec[:, None, :], COMMAND_FRAMES, 1).reshape(T, -1)
        x = np.zeros((T, 1762), np.float32)
        x[:, COMMAND_OFFSET:COMMAND_OFFSET + cmd_hold.shape[1]] = cmd_hold
        x[:, ANCHOR_MF_OFFSET:ANCHOR_MF_OFFSET + anc_hold.shape[1]] = anc_hold
        # 与录制 1762 向量逐位比对（只比这两段）
        rec = raw["encoder_input"][:T]
        d_cmd = float(np.abs(rec[:, COMMAND_OFFSET:COMMAND_OFFSET + 580] - cmd_hold).max())
        d_anc = float(np.abs(rec[:, ANCHOR_MF_OFFSET:ANCHOR_MF_OFFSET + 60] - anc_hold).max())

        with torch.no_grad():
            tok_rec = enc(torch.from_numpy(x[:, COMMAND_OFFSET:COMMAND_OFFSET + 580].reshape(T, COMMAND_FRAMES, 58)),
                          torch.from_numpy(x[:, ANCHOR_MF_OFFSET:ANCHOR_MF_OFFSET + 60].reshape(T, COMMAND_FRAMES, 6))).numpy()
            # ---- 2. legacy：未来 10 帧（端口侧车的老配方）----
            idx = np.minimum(np.arange(T)[:, None] + np.arange(COMMAND_FRAMES)[None, :], T - 1)
            cmd_fut = np.concatenate([q_isa[idx].reshape(T, -1), qd_isa[idx].reshape(T, -1)], 1)
            anc_fut = np.repeat(anchor_rec[:, None, :], COMMAND_FRAMES, 1).reshape(T, -1)
            tok_fut = enc(torch.from_numpy(cmd_fut.reshape(T, COMMAND_FRAMES, 58)),
                          torch.from_numpy(anc_fut.reshape(T, COMMAND_FRAMES, 6))).numpy()
            # ---- 3. anchor 能不能用参考自车（offline）推出来 ----
            ref_quat = torch.from_numpy(z["body_quat_w"][:T, 0].astype(np.float32))
            rob_quat = torch.from_numpy(raw["robot_root_orientation"][:T].astype(np.float32))
            anc_calc = rot6d(rob_quat, ref_quat).numpy()

        d_lat, _ = _stats(tok_rec, latent[:T])
        d_lab, c_lab = _stats(tok_rec, label[:T])
        d_fut, c_fut = _stats(tok_fut, label[:T])
        d_anchor, _ = _stats(anc_calc, anchor_rec)
        rows.append((f.stem, d_cmd, d_anc, d_lat, d_lab, c_lab, d_fut, c_fut, d_anchor))
        print(f"[{f.stem}] 输入重建: cmd max|Δ|={d_cmd:.2e} anchor max|Δ|={d_anc:.2e} | "
              f"token(重建) vs latent: {d_lat:.2e} vs label: {d_lab:.2e} corr={c_lab:.6f} | "
              f"future 窗口 vs label: {d_fut:.3f} corr={c_fut:.3f} | rot6d(robot,ref) vs 录制 anchor: {d_anchor:.2e}")

    print("\n[frontend] 结论：")
    ok_rec = all(r[3] < 1e-6 for r in rows)
    ok_lab = all(r[4] < 1e-6 for r in rows)
    ok_anc = all(r[8] < 1e-6 for r in rows)
    print(f"  · 1762 重建 == encoder_latent : {'PASS' if ok_rec else 'FAIL'}")
    print(f"  · 重建 token == g1_motion_token_64 : {'PASS' if ok_lab else 'FAIL'}")
    print(f"  · rot6d(robot_root, ref_root) == consumed_anchor_rot6d : {'PASS' if ok_anc else 'FAIL'}")
    print(f"  · legacy future-10 窗口与 label 的相关性: {np.mean([r[7] for r in rows]):.3f} "
          f"(越低越说明 old sidecar 是另一个空间)")
    return 0 if (ok_rec and ok_lab) else 1


if __name__ == "__main__":
    raise SystemExit(main())
