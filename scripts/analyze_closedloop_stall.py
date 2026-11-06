#!/usr/bin/env python3
"""闭环"站桩/冻结"诊断：不只看终点，还要看**什么时候停止前进、为什么**。

2026-11-05 夜间的发现（本脚本的由来）
------------------------------------
`eval_closedloop_progress.py` 只报 `prog_end`，把"走到 40% 然后原地站 12 s"和
"以 40% 的速度一直走"混成同一个数。实际 dump 里两种完全不同：
  * `speed_t` = 逐帧根节点水平速度（m/s）；
  * `stall_t` = 第一次之后**再也没超过** `--stall-speed` 的帧号；
  * `flat`    = chunk 的"时间结构"——同一 chunk 内相邻帧 token 差的均值。
                站桩时它会掉一个量级（token 块被采平），这是闭环进入吸收态的直接指纹。

用法::

    python3 scripts/analyze_closedloop_stall.py --runs base_s0 base_s1 oracle fm4 fm20
    python3 scripts/analyze_closedloop_stall.py --glob 'runs/closedloop_r3/base_s*'
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def demo_paths(min_len: int = 200, task: str = "double_desk"):
    pk = torch.load(ROOT / "runs/isaac_motion/ha_cmd/packed.pt", map_location="cpu", weights_only=False)
    names, starts, lengths = list(pk["names"]), pk["starts"], pk["lengths"]
    out = []
    for i, n in enumerate(names):
        if task not in str(n):
            continue
        s, L = int(starts[i]), int(lengths[i])
        if L < min_len:
            continue
        out.append(pk["ref"][s:s + L, 0:3].double().numpy())
    return out


def analyse(run: Path, paths, stall_speed: float):
    steps = run / "dump" / "steps.jsonl"
    if not steps.exists():
        return None
    rows = [json.loads(l) for l in steps.open() if l.strip()]
    P = np.array([r["root_pos"] for r in rows], np.float64)
    T = len(P)
    sp = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1) * 50.0          # m/s per frame
    sp = np.concatenate([sp, sp[-1:]]) if len(sp) else sp
    # 第一次之后再也没回到 stall_speed 以上的帧
    above = np.nonzero(sp > stall_speed)[0]
    stall = int(above[-1] + 1) if len(above) else 0
    # 示范路径进度
    xy = P[:, :2]
    best = None
    for D in paths:
        d = np.linalg.norm(xy[:, None, :] - D[None, :, :2], axis=2)
        m = d.min(axis=1)
        if best is None or float(np.median(m)) < float(np.median(best[0])):
            best = (m, D, d.argmin(axis=1))
    m, D, idx = best
    seg = np.linalg.norm(np.diff(D[:, :2], axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    prog = cum[idx] / max(cum[-1], 1e-9)
    # chunk 时间结构（站桩时会被采平）
    flat = float("nan")
    ch = run / "dump" / "chunks.jsonl"
    if ch.exists():
        C = np.array([json.loads(l)["chunk"] for l in ch.open() if l.strip()], np.float32)
        if C.ndim == 3 and C.shape[1] > 1:
            flat = float(np.abs(np.diff(C, axis=1)).mean())
    return dict(
        run=run.name, steps=T,
        net=float(np.linalg.norm(xy[-1] - xy[0])),
        arc=float(cum[-1]),
        prog_end=float(prog[-1]), prog_max=float(prog.max()),
        prog_at_stall=float(prog[min(stall, T - 1)]),
        stall_step=stall, stall_s=stall / 50.0,
        speed_mean=float(sp.mean()), speed_after_stall=float(sp[min(stall, T - 1):].mean()),
        chunk_flat=flat,
        z_min=float(P[:, 2].min()),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="目录名（相对 runs/closedloop_r3 或绝对路径）")
    ap.add_argument("--glob", default="")
    ap.add_argument("--base", default="runs/closedloop_r3")
    ap.add_argument("--stall-speed", type=float, default=0.06, help="m/s；低于它算站桩")
    a = ap.parse_args()

    dirs = [Path(d) if Path(d).is_absolute() else Path(a.base) / d for d in a.runs]
    if a.glob:
        dirs += [Path(p) for p in sorted(glob.glob(a.glob))]
    paths = demo_paths()
    print(f"[stall] 示范 {len(paths)} 条；站桩阈值 {a.stall_speed:.2f} m/s（=1.2 cm/帧）")
    hdr = (f"{'run':<16}{'steps':>6}{'net':>7}{'arc':>6}{'prog_end':>9}{'prog_max':>9}"
           f"{'stall@':>7}{'prog@st':>8}{'v_mean':>8}{'v_post':>8}{'chunk_flat':>11}{'z_min':>7}")
    print(hdr)
    print("-" * len(hdr))
    for d in dirs:
        r = analyse(d, paths, a.stall_speed)
        if r is None:
            print(f"{d.name:<16}  [skip] 没有 dump/steps.jsonl")
            continue
        flat = "nan" if np.isnan(r["chunk_flat"]) else f"{r['chunk_flat']:.4f}"
        print(f"{r['run']:<16}{r['steps']:>6}{r['net']:>7.3f}{r['arc']:>6.2f}{r['prog_end']:>9.3f}"
              f"{r['prog_max']:>9.3f}{r['stall_step']:>7d}{r['prog_at_stall']:>8.3f}"
              f"{r['speed_mean']:>8.3f}{r['speed_after_stall']:>8.3f}{flat:>11}{r['z_min']:>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
