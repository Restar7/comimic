#!/usr/bin/env python3
"""同一个 WBC/世界模型，同一批窗口，只换 64 维 token 流 —— 谁被"读懂"。

背景（2026-10-27）：盘上有两份 HA motion token，训练/评测时各用各的：

* ``label`` = npz 的 ``g1_motion_token_64``：编码器吃**部署前端 1762 维**（命令流 + 机器人状态 +
  SMPL + 手腕 …）→ 64 维。VLA 的默认监督目标就是它。
* ``port``  = ``data/port_tokens/ha_tokens/*.npy``（10-25 建）：编码器只吃 640 维切片
  （未来 10 帧 ``joint_pos``/``joint_vel`` + anchor）。**注意**：这份是从 npz 的 ``joint_pos``
  算的，而 npz 的 ``joint_pos`` 是**实际执行轨迹**（``robot_qpos_before_decimation``），
  不是命令流 —— 见 ``build_ha_command_lib.py`` 开头的口径说明。

于是"两份 token 不一致"有两个叠加原因：① 输入被截断（1762 → 640）；② 执行轨迹 vs 命令流。
本脚本不改任何权重，只回答一个问题：**当前 WBC（含其世界模型）读得懂哪一份。**

用法：
    python3 scripts/probe_token_ab.py --supermimic-ckpt runs/S1_handover/WBC_A20000_wmvel_cmd.pt \
        --packed runs/isaac_motion/ha_cmd/packed.pt --batches 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "trackerlab-supermimic/source/trackerLab"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diff_vla_wbc.data import MotionBatchProvider, RecordedTokenSource  # noqa: E402
from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: E402
from diff_vla_wbc.rollout import compute_joint_loss, imagine_with_tokens  # noqa: E402
from diff_vla_wbc.train import JointTrainerConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", type=Path, default=ROOT / "data/npz/ha_tokens")
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--port-dir", type=Path, default=ROOT / "data/port_tokens/ha_tokens")
    ap.add_argument("--supermimic-ckpt", type=Path,
                    default=ROOT / "runs/S1_handover/WBC_A20000_wmvel_cmd.pt")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--batches", type=int, default=4, help="抽几批窗口（结果取平均）")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--hist", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    dev = torch.device(a.device)
    npz = sorted(a.motion.glob("*.npz"))
    bundle = load_isaac_bundle(a.supermimic_ckpt, device=dev, npz=npz, packed_motion=a.packed,
                               packed_dir=None, obs_mode=None, hist=a.hist, token_dim=64)
    lib = bundle.library
    src = {
        "label": RecordedTokenSource(a.motion, lib, token_horizon=40, token_dim=64,
                                     device=dev, token_space="label"),
        "port": RecordedTokenSource(a.motion, lib, token_horizon=40, token_dim=64,
                                    device=dev, port_dir=a.port_dir, token_space="port"),
    }
    prov = MotionBatchProvider(
        bundle, lib, batch_size=a.batch_size, horizon=40, token_source=None, token_dim=64,
        token_horizon=40, seed=a.seed, hand_dir=None,
    )
    cfg = JointTrainerConfig(horizon=a.horizon)

    acc: dict[str, list] = {k: [] for k in src}
    mse_pair: list[float] = []
    for bi in range(a.batches):
        b = prov.sample()
        toks = {k: s({"frames": b.frames}) for k, s in src.items()}
        mse_pair.append(float(((toks["label"] - toks["port"]) ** 2).mean()))
        for k, tok in toks.items():
            with torch.no_grad():
                ro = imagine_with_tokens(
                    bundle, tokens=tok, state=b.start_state, obs_hist=b.obs_hist,
                    history=b.history, xi=b.xi, references=b.references,
                    prev_action=b.prev_action, horizon=a.horizon, obs_noise=0.0,
                )
                _, terms = compute_joint_loss(
                    ro, teacher_states=b.teacher_states, gamma=cfg.gamma, weights=cfg.weights,
                )
            acc[k].append((float(ro.rewards.mean()), float(terms.get("state", torch.zeros(()))),
                           float(ro.survival[-1].mean())))
        print(f"[batch {bi+1}/{a.batches}] done", flush=True)

    print(f"\n[cmp] MSE(label, port) = {sum(mse_pair)/len(mse_pair):.5f}")
    print(f"{'tokens':6s}  {'imagined_reward':>15s}  {'state_term':>10s}  {'survival':>8s}")
    for k, rows in acc.items():
        n = len(rows)
        print(f"{k:6s}  {sum(r[0] for r in rows)/n:15.5f}  {sum(r[1] for r in rows)/n:10.4f}  "
              f"{sum(r[2] for r in rows)/n:8.5f}")
    print("\n[读法] 谁能让世界模型给出**更高奖励 + 更低 state_term**，谁就是执行器训练时吃的空间。",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
