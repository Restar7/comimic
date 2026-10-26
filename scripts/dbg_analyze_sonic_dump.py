#!/usr/bin/env python3
"""分析 `SONIC_DEBUG_DUMP` 的逐帧转储，并与 HA 录制 npz 对齐比较。

转储由 `action_provider_sonic.py` 的 `_dbg_step` 写入（env `SONIC_DEBUG_DUMP=<dir>`）：
    steps.jsonl   每控制步一行：latent(64) / obs994 / raw(29) / target_predelay(29) /
                  applied(29) / q(29) / qd(29) / root_pos(3) / root_quat_wxyz(4) /
                  root_lin_vel_b(3) / root_ang_vel_b(3) / grav(3) / sim_ms
    chunks.jsonl  每次 VLA 请求：state64 / chunk(40x64)
    req_XXX_rgb.npy  每次请求的相机图

用法：
    python3 scripts/dbg_analyze_sonic_dump.py \
        --dump runs/dbg_sonic_gt/dump \
        --npz data/ha_raw/HOI_double_desk/tw/xxx.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# 994 = latent(64) + ang_vel(10*3) + joint_pos_delta(10*29) + joint_vel(10*29)
#       + last_action(10*29) + gravity(10*3)
OBS_BLOCKS = (("ang_vel", 30), ("joint_pos", 290), ("joint_vel", 290),
              ("last_action", 290), ("gravity", 30))


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _arr(rows: list[dict], k: str) -> np.ndarray:
    return np.stack([np.asarray(r[k], np.float64) for r in rows])


def _quat_to_rpy_deg(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return np.array([roll, pitch, yaw])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--npz", default="", help="同一集/同一任务的录制，用于逐帧对齐")
    ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()

    dump = Path(args.dump)
    rows = _load_jsonl(dump / "steps.jsonl")
    if not rows:
        raise SystemExit(f"{dump}/steps.jsonl 为空")
    print(f"[dump] {len(rows)} 控制步  frames {rows[0]['frame']}..{rows[-1]['frame']}")

    applied = _arr(rows, "applied")
    target = _arr(rows, "target_predelay")
    raw = _arr(rows, "raw")
    q = _arr(rows, "q")
    qd = _arr(rows, "qd")
    latent = _arr(rows, "latent")
    obs = _arr(rows, "obs994")
    rpos = _arr(rows, "root_pos")
    rquat = _arr(rows, "root_quat_wxyz")
    grav = _arr(rows, "grav")
    t = np.array([r["t"] for r in rows])

    rpy = np.stack([_quat_to_rpy_deg(x) for x in rquat])
    height = rpos[:, 2]
    # 站立时 root 高度约 0.75~0.8 m；<0.45 视为已经倒下（与 fall detector 同量级）
    fell = height < 0.45
    first_fall = int(np.argmax(fell)) if fell.any() else -1
    print(f"[root] z: start={height[0]:.3f}  min={height.min():.3f}  end={height[-1]:.3f}")
    print(f"[root] |roll|/|pitch| max = {np.abs(rpy[:,0]).max():.1f}/{np.abs(rpy[:,1]).max():.1f} deg")
    print(f"[root] xy 位移 = {np.linalg.norm(rpos[-1,:2]-rpos[0,:2]):.3f} m")
    if first_fall >= 0:
        print(f"[root] 首次 z<0.45 出现在第 {first_fall} 步 ({first_fall*0.02:.2f} s)")
    else:
        print("[root] 全程 z>=0.45（没倒）")

    # 关节跟踪误差（目标 vs 实际）
    pos_err = q - applied
    print(f"\n[tracking] |target-actual| 均值={np.abs(pos_err).mean():.4f} "
          f"p95={np.percentile(np.abs(pos_err),95):.4f} max={np.abs(pos_err).max():.4f} rad")
    worst = np.argsort(-np.abs(pos_err).mean(0))[: args.topk]
    print(f"[tracking] 误差最大的关节 idx={list(map(int,worst))} "
          f"mean={np.round(np.abs(pos_err).mean(0)[worst],4)}")
    print(f"[tokens] latent mean={latent.mean():+.4f} std={latent.std():.4f} "
          f"|lat|max={np.abs(latent).max():.3f}")
    print(f"[tokens] 相邻帧 latent 跳变 mean={np.linalg.norm(np.diff(latent,axis=0),axis=1).mean():.4f}")
    print(f"[timing] 控制周期 mean={np.diff(t).mean()*1000:.2f} ms  "
          f"min={np.diff(t).min()*1000:.2f}  max={np.diff(t).max()*1000:.2f}  "
          f"sim_ms mean={np.mean([r['sim_ms'] for r in rows]):.1f}")

    # ---- 与录制对齐（仅 GT token 臂有意义：帧号 = 录制帧号）----
    if not args.npz:
        return 0
    z = np.load(args.npz, allow_pickle=True)
    n = min(len(rows), len(z["decoder_target_action"]))
    print(f"\n[compare] 与录制逐帧比较（前 {n} 帧）")
    rec_tgt = np.asarray(z["decoder_target_action"], np.float64)[:n]
    rec_raw = np.asarray(z["decoder_raw_action"], np.float64)[:n]
    rec_obs = np.asarray(z["decoder_obs"], np.float64)[:n]
    rec_lat = np.asarray(z["encoder_latent"], np.float64)[:n]
    rec_h = np.asarray(z["robot_root_position"], np.float64)[:n, 2]

    print(f"  latent     MAE={np.abs(latent[:n]-rec_lat).mean():.4f} "
          f"(corr={np.corrcoef(latent[:n].ravel(), rec_lat.ravel())[0,1]:.4f})")
    print(f"  raw action MAE={np.abs(raw[:n]-rec_raw).mean():.4f}")
    print(f"  target     MAE={np.abs(target[:n]-rec_tgt).mean():.4f} rad")
    print(f"  root z     录制={rec_h.mean():.3f} 本run={height[:n].mean():.3f} "
          f"MAE={np.abs(height[:n]-rec_h).mean():.4f} m")

    off = 64
    for name, width in OBS_BLOCKS:
        a = obs[:n, off:off + width]
        b = rec_obs[:n, off:off + width]
        print(f"  obs[{name:11s}] MAE={np.abs(a-b).mean():.5f}  "
              f"run|mean|={np.abs(a).mean():.4f} rec|mean|={np.abs(b).mean():.4f}")
        off += width
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
