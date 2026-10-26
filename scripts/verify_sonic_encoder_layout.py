#!/usr/bin/env python3
"""验证 SONIC encoder 的 1762 输入布局（2026-10-26 实测破译）。

结论（在 `data/ha_raw` 的录制帧上逐位验证，token 相等比例 100%）：

    col[4:584]    = [ q_t (29) × 10 | qd_t (29) × 10 ]   # **当前参考帧**重复 10 次
    col[601:661]  = anchor_rot6d (6) × 10                 # 锚点 6 维，10 帧恒定
    （col[595:601] 是同一锚点的另一份拷贝；其余列不参与 g1 分支）

    640 = 580 (qdq) + 60 (anchor×10)  →  MLP(640→2048→1024→512→512→64) → FSQ

`q_t` = `human_joint_pos[t]`（重定向参考，MJCF/SDK 顺序，**不做** perm）；
`qd_t` = `encoder_motion_joint_vel_hist[t, -1]`；
`anchor_rot6d` = `consumed_anchor_rot6d[t]`（HA runtime 产生，公式未知 → 只能从录制数据取）。

与旧 `SonicTokenBuilder` 的差异（这就是端口 token 与标签 corr 仅 0.057 的根因）：
  * 旧：未来 10 帧 q/qd + 未来 10 帧 anchor；
  * 真：**当前帧 q/qd 重复 10 次** + 恒定的 anchor 6 维。

用法::

    python3 scripts/verify_sonic_encoder_layout.py --raw data/ha_raw/HOI_double_desk/tw/<...>.npz
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np


def build_encoder_input(q, qd, anchor6):
    """返回 (1762,) 的 encoder 输入：580 qdq + 60 anchor（其余为 0）。"""
    x = np.zeros(1762, dtype=np.float32)
    x[4:294] = np.tile(q, 10)          # 当前 q × 10
    x[294:584] = np.tile(qd, 10)       # 当前 qd × 10
    x[601:661] = np.tile(anchor6, 10)  # 锚点 × 10
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="")
    ap.add_argument("--onnx", default="../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx")
    ap.add_argument("--frames", type=int, default=100)
    a = ap.parse_args()
    raw = a.raw or str(sorted(Path("data/ha_raw/HOI_double_desk/tw").glob("*.npz"))[0])

    import onnxruntime as ort
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    with np.load(raw) as r:
        lat = np.asarray(r["encoder_latent"], np.float32)
        q = np.asarray(r["human_joint_pos"], np.float32)
        qd = np.asarray(r["encoder_motion_joint_vel_hist"], np.float32)[:, -1]
        anc = np.asarray(r["consumed_anchor_rot6d"], np.float32)
        T = len(lat)

    ok = []
    for t in range(min(a.frames, T)):
        x = build_encoder_input(q[t], qd[t], anc[t])[None]
        tok = sess.run(None, {iname: x})[0].reshape(-1)
        ok.append(float((tok == lat[t]).mean()))
    print(f"[layout] {Path(raw).name}: token 相等比例 = {np.mean(ok):.4f}  "
          f"({'PASS' if np.mean(ok) > 0.999 else 'FAIL'})")
    return 0 if np.mean(ok) > 0.999 else 1


if __name__ == "__main__":
    raise SystemExit(main())
