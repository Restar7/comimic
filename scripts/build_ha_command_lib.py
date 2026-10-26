#!/usr/bin/env python3
"""S1-A：用 HA 的 **command 流**重建 packed 运动库（标签 token 空间的自洽参考）。

为什么不用 `data/npz/ha_tokens`：那份 `joint_pos` 是**机器人实际执行轨迹**
（`robot_qpos_before_decimation`，逐位相等），而 `encoder_latent` 是用
`encoder_motion_joint_pos_hist`（**命令流**）算出来的 —— 两者 mean|Δ|=0.39 rad、最佳 lag +8 帧。
拿执行轨迹配标签 token = token 与参考系统性错位 0.16 s。

本脚本按 **canonical 布局**（`p(3) quat(4) v(3) w(3) q(29) qd(29)` = 71）打包：

* `q` / `qd`   ← `encoder_motion_joint_{pos,vel}_hist[t][-1]`（命令流，encoder 顺序 → 转回 MJCF 序）
* `p` / `v` / `w` ← `robot_root_position` / `_lin_vel_world` / `_ang_vel_world`（命令流没有根轨迹，用执行轨迹作代理）
* `quat`       ← `robot_root_orientation`（同上；anchor 已在 token 里单独编码，这里只给跟踪用的根姿态）

    python3 scripts/build_ha_command_lib.py --out runs/isaac_motion/ha_cmd --limit 3   # 冒烟
    python3 scripts/build_ha_command_lib.py --out runs/isaac_motion/ha_cmd             # 全量 699
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diff_vla_wbc.sonic_encoder import mjcf_to_isaaclab_perm  # noqa: E402

RAW = ROOT / "data/ha_raw"


def clip_files() -> list[Path]:
    return sorted(RAW.glob("*/*/*.npz"))


def clip_name(p: Path, task_dir: str, src_dir: str) -> str:
    ep = p.stem.rsplit("_", 1)[-1]
    return f"{task_dir}__{src_dir}__ep{int(ep):06d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    perm = mjcf_to_isaaclab_perm()
    inv = np.argsort(perm)                      # isaaclab -> mjcf

    refs, names, lens = [], [], []
    files = clip_files()
    if a.limit:
        files = files[: a.limit]
    for p in files:
        z = np.load(p, allow_pickle=True)
        T = len(z["robot_root_position"])
        # ---- 命令流（encoder 顺序）→ MJCF 顺序 ----
        q_cmd = z["encoder_motion_joint_pos_hist"][:T, -1][:, inv].astype(np.float32)
        qd_cmd = z["encoder_motion_joint_vel_hist"][:T, -1][:, inv].astype(np.float32)
        p_root = z["robot_root_position"][:T].astype(np.float32)
        quat = z["robot_root_orientation"][:T].astype(np.float32)      # wxyz
        v = z["robot_root_lin_vel_world"][:T].astype(np.float32)
        w = z["robot_root_ang_vel_world"][:T].astype(np.float32)
        ref = np.concatenate([p_root, quat, v, w, q_cmd, qd_cmd], axis=1)
        assert ref.shape[1] == 71, ref.shape
        refs.append(torch.from_numpy(ref).to(torch.float16))
        names.append(clip_name(p, p.parent.parent.name, p.parent.name))
        lens.append(T)

    packed = {"ref": torch.cat(refs), "names": names,
              "starts": [int(s) for s in np.cumsum([0] + lens[:-1])], "lengths": lens}
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / "packed.pt"
    torch.save(packed, out)
    meta = {"ref_layout": "canonical", "src_fps": 50.0, "control_dt": 0.02, "dtype": "float16",
            "num_clips": len(names), "num_frames": int(sum(lens)),
            "source": "data/ha_raw encoder_motion_joint_{pos,vel}_hist[-1] (command stream)",
            "note": "root p/v/w = robot trajectory (proxy); q/qd = encoder command stream"}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[ha-cmd] wrote {out} clips={len(names)} frames={sum(lens)} ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
