#!/usr/bin/env python3
"""闭环结果的多 seed 客观判读：不只看"净位移"，还看**是否走在示范路径上/走了多远**。

为什么净位移不能单独当判据
--------------------------
闭环是混沌的：token 差一点点，轨迹就分叉。单 seed 的 1.268 m vs 1.006 m 可能只是分叉，
不是"更好"。所以要：
 1. **多 seed**（每个 head 至少 3 个 seed）报均值/范围；
 2. 再加两个与任务相关的量：
    * `d_path` = 轨迹每个点到**DoubleDesk 示范路径**（`ha_cmd/packed.pt` 的 100 条 clip）的
      最近距离（中位数）——用来判断"是不是在干这件事"，站桩/乱走都会大；
    * `prog`   = 终点在示范路径上的**进度**（0=起点，1=示范终点，取最近点的累积弧长比例）。

用法::

    python3 scripts/eval_closedloop_progress.py --runs base:=runs/closedloop_r3/hands \
        base:=runs/closedloop_r3/base_s1 v4:=runs/closedloop_r3/v4 v4:=runs/closedloop_r3/v4_s1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def demo_paths(min_len: int = 200):
    pk = torch.load(ROOT / "runs/isaac_motion/ha_cmd/packed.pt", map_location="cpu",
                    weights_only=False)
    names, starts, lengths = list(pk["names"]), pk["starts"], pk["lengths"]
    out = []
    for i, n in enumerate(names):
        if "double_desk" not in str(n):
            continue
        s, L = int(starts[i]), int(lengths[i])
        if L < min_len:
            continue
        P = pk["ref"][s:s + L, 0:3].double().numpy()
        out.append(P)
    return out


def load_run(d: Path):
    p = d / "dump" / "steps.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.open()]
    return np.array([r["root_pos"] for r in rows], np.float64)


def score(traj: np.ndarray, paths):
    xy = traj[:, :2]
    best = None
    for P in paths:
        d = np.linalg.norm(xy[:, None, :] - P[None, :, :2], axis=2)     # (T, L)
        m = d.min(axis=1)                                               # 每帧到该示范的最近距离
        if best is None or float(np.median(m)) < float(np.median(best[0])):
            best = (m, P, d.argmin(axis=1))
    m, P, idx = best
    seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    prog = cum[idx] / max(cum[-1], 1e-9)
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    return dict(n=len(traj), net=net, path=path,
                d_path_p50=float(np.median(m)), d_path_p90=float(np.quantile(m, 0.9)),
                prog_end=float(prog[-1]), prog_max=float(prog.max()),
                z_min=float(traj[:, 2].min()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, metavar="TAG=DIR")
    a = ap.parse_args()
    paths = demo_paths()
    print(f"[eval] 示范路径 {len(paths)} 条（HOI_double_desk）")
    agg: dict[str, list[dict]] = {}
    for spec in a.runs:
        tag, _, d = spec.partition("=")
        traj = load_run(Path(d))
        if traj is None:
            print(f"  [skip] {spec} 没有 dump")
            continue
        s = score(traj, paths)
        agg.setdefault(tag, []).append(s)
        print(f"  {tag:<8} n={s['n']:4d} net={s['net']:.3f} path={s['path']:.3f} "
              f"d_path p50={s['d_path_p50']:.3f} p90={s['d_path_p90']:.3f} "
              f"prog_end={s['prog_end']:.2f} prog_max={s['prog_max']:.2f} z_min={s['z_min']:.3f}")
    print("\n[eval] 按 head 汇总（多 seed）：")
    for tag, rows in agg.items():
        def ms(k):
            v = np.array([r[k] for r in rows])
            return f"{v.mean():.3f}±{v.std():.3f}"
        print(f"  {tag:<8} seeds={len(rows)}  net {ms('net')}  d_path {ms('d_path_p50')}  "
              f"prog_end {ms('prog_end')}  z_min {min(r['z_min'] for r in rows):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
