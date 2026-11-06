#!/usr/bin/env python3
"""把 compact 库里的**根位置 xy** 灌进 `data/npz/g1_tasks` 那批 npz（其余字段不动）。

为什么需要（2026-10-30）
  `prepare_g1_tasks_npz.py` 明确把 root_xy 钉在原点（"任务是原地操作"），产出
  `data/npz/g1_tasks/*.npz` 里 pelvis 的 x/y 恒为 0。而 `$MOTION`（这个 npz 目录）是
  `train_joint_mjlab.py --motion` 的**参考库**、也是 `probe_wm_qd_corr.py` 的基准。
  一旦 WBC 训练改用带真位移的 compact 库（v3），联合微调却仍读"原地"npz ⇒ 两个目标打架
  （2026-10-30 实测过这个病：位置项与关节跟踪不可同时满足，策略放弃位置项）。

做法
  复刻 npz 里的 FK 只做了「根在原点」的位移，所以把这个偏移**整体平移**到所有 body 上即可：
      body_pos_w[..., 0:2] += root_xy[t]
  关节/朝向/时间轴/`g1_*` 侧车全部原样写回。

用法
  python3 scripts/patch_npz_root_xy.py --library data/library/ft318_compact_v3.pt \
      --src data/npz/g1_tasks --dst data/npz/g1_tasks_v3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, default=ROOT / "data/library/ft318_compact_v3.pt")
    ap.add_argument("--src", type=Path, default=ROOT / "data/npz/g1_tasks")
    ap.add_argument("--dst", type=Path, default=ROOT / "data/npz/g1_tasks_v3")
    a = ap.parse_args()

    lib = torch.load(str(a.library), map_location="cpu", weights_only=False)
    ref = lib["ref"].float()
    names = [Path(str(n)).stem for n in lib["names"]]
    starts = [int(x) for x in lib["starts"]]
    lengths = [int(x) for x in lib["lengths"]]
    by_name = {n: (s, L) for n, s, L in zip(names, starts, lengths)}

    files = sorted(a.src.glob("*.npz"))
    if not files:
        raise SystemExit(f"[fail] {a.src} 里没有 npz")
    a.dst.mkdir(parents=True, exist_ok=True)
    used, miss, shifts = 0, [], []
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        stem = f.stem
        if stem not in by_name:
            miss.append(stem)
            continue
        s, L = by_name[stem]
        T = int(d["body_pos_w"].shape[0])
        n = min(L, T)
        xy = ref[s:s + n, 0:2].numpy()
        bp = d["body_pos_w"].copy()
        bp[:n, :, 0] += xy[:, 0:1]
        bp[:n, :, 1] += xy[:, 1:2]
        d["body_pos_w"] = bp
        # 根线速度（npz 里有就一起补，没有就算了）
        if "body_lin_vel_w" in d and n > 1:
            v = np.zeros((n, 2), np.float32)
            v[:-1] = (xy[1:] - xy[:-1]) * 50.0
            blv = d["body_lin_vel_w"].copy()
            blv[:n, :, 0] += v[:, 0:1]
            blv[:n, :, 1] += v[:, 1:2]
            d["body_lin_vel_w"] = blv
        if "source" in d and isinstance(d["source"], np.ndarray) is False:
            pass
        np.savez(a.dst / f.name, **d)
        used += 1
        shifts.append(float(np.abs(xy).max()))
    print(f"[patch] {used}/{len(files)} 个 npz 已平移 → {a.dst}")
    if miss:
        print(f"[warn] {len(miss)} 个 npz 在库里找不到同名 clip（例 {miss[:2]}）")
    if shifts:
        print(f"[stat] 平移量 max|xy|: 中位 {np.median(shifts):.2f} m、max {max(shifts):.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
