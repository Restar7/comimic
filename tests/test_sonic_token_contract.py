#!/usr/bin/env python3
"""门禁：SONIC token 前端契约（2026-10-26 逐位对拍破译）。

背景与证据
----------
`docs/SONIC_ENCODER_INPUT_LAYOUT.md` + `docs/S2_BASE_ALIGNED_PLAN_20261026.md` §1：

* ``encoder_input[4:584]`` = command，**term-major** 580 维 = ``[q_t ×10 | qd_t ×10]``，
  其中 ``q_t = human_joint_pos[t]``（参考/命令流，**当前帧重复 10 次**）；
* ``encoder_input[601:661]`` = anchor，``consumed_anchor_rot6d[t]`` 重复 10 次；
* **anchor 公式（本次解出）**：

      anchor6 = rot6d(conj(q_robot) ⊗ q_ref_aligned)
      q_robot       = robot_root_orientation
      q_ref_aligned = human_body_quat_w_aligned

  与录制 ``consumed_anchor_rot6d`` 的 mean|Δ| ≤ 6e-5；
* 关节顺序：HA 数据是 encoder 序（= Isaac/HA_V31），**不做** ``PERM_MJCF_TO_ISAACLAB``；
  库侧若按 MJCF 序存（`build_ha_command_lib.py`），则 `build_motion_windows` 内部置换是对的。

这两条测试就是"S1 造 token 的验收线"：改坏窗口/布局/置换/anchor，立即红。
需要 `data/npz/ha_tokens` + `data/ha_raw`（不在 git 里，缺失时 skip）。
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from diff_vla_wbc.sonic_encoder import (
    G1MotionEncoder,
    mjcf_to_isaaclab_perm,
    rot6d,
    tokens_from_motion,
)

ROOT = Path(__file__).resolve().parents[1]
TOK_DIR = ROOT / "data" / "npz" / "ha_tokens"
RAW_DIR = ROOT / "data" / "ha_raw"
HAS_DATA = TOK_DIR.is_dir() and any(TOK_DIR.glob("*.npz"))


def _first_clip(max_frames: int = 600):
    """取一条 token npz，并用它的 ``source`` 找到 raw npz（含 anchor 与参考流）。"""
    for path in sorted(TOK_DIR.glob("*.npz")):
        with np.load(path) as z:
            src = Path(str(z["source"]))
            label = np.asarray(z["g1_motion_token_64"], dtype=np.float32)
        cand = list(RAW_DIR.glob(f"*/*/{src.name}.npz")) or list(RAW_DIR.glob(f"*/*/*{src.stem}*.npz"))
        if not cand:
            continue
        T = min(int(len(label)), max_frames)
        return path, cand[0], T
    return None


@unittest.skipUnless(HAS_DATA, "需要 data/npz/ha_tokens + data/ha_raw（本机数据，不在 git 里）")
class TokenContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clip, cls.raw, cls.T = _first_clip()
        cls.enc = G1MotionEncoder.from_onnx()

    def test_anchor_formula_matches_recording(self):
        with np.load(self.raw) as z:
            a_true = np.asarray(z["consumed_anchor_rot6d"][: self.T], dtype=np.float64)
            q_robot = np.asarray(z["robot_root_orientation"][: self.T], dtype=np.float64)
            q_ref = np.asarray(z["human_body_quat_w_aligned"][: self.T], dtype=np.float64)
            # 录制里的 anchor window 必须与 consumed_anchor_rot6d 一致（hold 语义）
            win = np.asarray(z["encoder_anchor_window"][: self.T], dtype=np.float64)
        self.assertLess(float(np.abs(win[:, 0] - a_true).mean()), 1e-3)
        a_calc = rot6d(torch.from_numpy(q_robot), torch.from_numpy(q_ref)).numpy()
        self.assertLess(float(np.abs(a_calc - a_true).mean()), 1e-3, "anchor 公式口径变了")

    def test_ha_label_parity(self):
        """库口径（MJCF 序）+ hold 窗口 + 公式 anchor → token 与标签逐位相等。"""
        inv = np.argsort(mjcf_to_isaaclab_perm())
        with np.load(self.raw) as z:
            label = np.asarray(z["encoder_latent"][: self.T], dtype=np.float32)
            q = np.asarray(z["human_joint_pos"][: self.T], dtype=np.float32)[:, inv]
            qd = np.asarray(z["encoder_motion_joint_vel_hist"][: self.T, -1], dtype=np.float32)[:, inv]
            q_ref = np.asarray(z["human_body_quat_w_aligned"][: self.T], dtype=np.float32)
            q_robot = np.asarray(z["robot_root_orientation"][: self.T], dtype=np.float32)
        tok = tokens_from_motion(
            self.enc,
            torch.from_numpy(q),
            torch.from_numpy(qd),
            torch.from_numpy(q_ref),
            torch.from_numpy(q_robot),
            num_future=10,
        ).numpy()
        exact = float((np.abs(tok - label) < 1e-9).mean())
        self.assertGreater(exact, 0.999, f"token 前端口径漂了：exact_frac={exact:.4f}（应 ≥0.999）")


if __name__ == "__main__":
    unittest.main()
