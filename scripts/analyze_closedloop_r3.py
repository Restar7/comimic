#!/usr/bin/env python3
"""官方闭环（SonicActionProvider）逐帧 dump 的体检：摔没摔、走没走、手动没动。

为什么不用 ξ
------------
DoubleDesk 的官方 eval **没有 reference root**（`sonic_pose_source=redis`，command 驱动，
result.json 的 final_reward 恒 0），所以"相对参考的漂移 ξ"在这条链路上不可定义。
能测的就是：**站立高度 / 姿态是否翻倒 / 位移与路径 / 手有没有动作 / token 有没有塌**。

用法::

    python3 scripts/analyze_closedloop_r3.py --dump runs/closedloop_r3/ft/dump
    python3 scripts/analyze_closedloop_r3.py --compare runs/closedloop_r3/ft/dump runs/closedloop_r3/base/dump
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _quat_up(quat_wxyz: np.ndarray) -> np.ndarray:
    """世界 z 轴在机体系的分量（站立 = +1，平躺 ≈ 0）。"""
    w, x, y, z = quat_wxyz
    # R^T e_z，第三行 of R^T == 第三列 of R
    r20 = 2.0 * (x * z + w * y)
    r21 = 2.0 * (y * z - w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return np.array([r20, r21, r22], dtype=np.float64)


def load(dump: str | Path) -> dict:
    p = Path(dump) / "steps.jsonl"
    if not p.exists():
        raise SystemExit(f"没有 {p}")
    rows = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{p} 是空的")
    root = np.array([r["root_pos"] for r in rows], dtype=np.float64)
    quat = np.array([r["root_quat_wxyz"] for r in rows], dtype=np.float64)
    q = np.array([r["q"] for r in rows], dtype=np.float64)
    qd = np.array([r["qd"] for r in rows], dtype=np.float64)
    lat = np.array([r["latent"] for r in rows], dtype=np.float64)
    act = np.array([r["applied"] for r in rows], dtype=np.float64)
    up = np.array([_quat_up(x) for x in quat])
    return dict(rows=rows, root=root, quat=quat, q=q, qd=qd, lat=lat, act=act,
                up_z=up[:, 2], n=len(rows))


def report(d: dict, tag: str, every: int = 20) -> dict:
    root, n = d["root"], d["n"]
    z = root[:, 2]
    # 站立高度的参考：前 10 帧的中位数（还没被推倒）
    z0 = float(np.median(z[:10]))
    fallen = z < max(0.45, 0.6 * z0)
    # 摔倒帧的判据再加一条：躯干翻倒（世界 z 在机体系 < 0.5）
    tipped = d["up_z"] < 0.5
    dxy = np.diff(root[:, :2], axis=0)
    path = float(np.linalg.norm(dxy, axis=1).sum()) if n > 1 else 0.0
    net = float(np.linalg.norm(root[-1, :2] - root[0, :2]))
    lat_range = float(root[:, 1].ptp()) if n > 1 else 0.0
    # 手的活动度：q 若宽 43（29 body + 14 hand）后 14 维算手
    nh = max(0, d["q"].shape[1] - 29)
    hand_amp = float(d["q"][:, 29:].std()) if nh > 0 else float("nan")
    # token 是否塌（相邻帧 latent 变化太小 = 站着不动）
    dlat = float(np.linalg.norm(np.diff(d["lat"], axis=0), axis=1).mean()) if n > 1 else 0.0
    a_rms = float(np.sqrt((d["act"][:, :29] ** 2).mean()))

    print(f"===== {tag} =====")
    print(f"  帧数              {n}")
    print(f"  时长              {n * 0.02:.1f} s")
    print(f"  root z  首/末/最低  {z[0]:.3f} / {z[-1]:.3f} / {z.min():.3f}  (站立基准 {z0:.3f})")
    print(f"  躯干翻倒帧        {int(tipped.sum())}  ({100.0 * tipped.mean():.1f}%)  up_z 最低 {d['up_z'].min():.3f}")
    print(f"  低姿态帧(<0.45m)  {int(fallen.sum())}")
    print(f"  平面位移 净/路径   {net:.3f} m / {path:.3f} m")
    print(f"  y 方向摆动        {lat_range:.3f} m")
    print(f"  手活动 std        {hand_amp:.4f}")
    print(f"  相邻帧 |Δlatent|  {dlat:.4f}   |latent| 均值 {float(np.linalg.norm(d['lat'], axis=1).mean()):.2f}")
    print(f"  下发动作 RMS      {a_rms:.4f}")
    print("  每 replan 采样（每 %d 帧）:" % every)
    for i in range(0, n, every):
        j = min(i + every, n)
        seg = root[i:j, :2]
        print(f"    f{i:4d}-{j - 1:4d}  z={z[i:j].mean():.3f}  "
              f"x={root[i:j, 0].mean():+.2f} y={root[i:j, 1].mean():+.2f}  "
              f"up={d['up_z'][i:j].mean():+.2f}")
    return dict(n=n, z_min=float(z.min()), tipped=int(tipped.sum()),
                low=int(fallen.sum()), net=net, path=path,
                hand_amp=hand_amp, dlat=dlat, a_rms=a_rms, up_min=float(d["up_z"].min()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", help="单个 dump 目录")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="两个 dump 目录对比")
    ap.add_argument("--every", type=int, default=20)
    args = ap.parse_args()
    if args.compare:
        out = []
        for tag, dd in zip(args.compare, ("A", "B")):
            out.append(report(load(tag), f"{tag}  [{dd}]", args.every))
        a, b = out
        print("\n===== A vs B =====")
        for k in ("n", "z_min", "tipped", "low", "net", "path", "hand_amp", "dlat", "a_rms"):
            print(f"  {k:<9} A={a[k]:>10.4f}   B={b[k]:>10.4f}")
    elif args.dump:
        report(load(args.dump), args.dump, args.every)
    else:
        raise SystemExit("给 --dump 或 --compare")


if __name__ == "__main__":
    main()
