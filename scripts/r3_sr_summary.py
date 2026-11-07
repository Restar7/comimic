#!/usr/bin/env python3
"""汇总 r3_sr_cell.sh 产出的成功率（v13 vs base，按任务分组）。

用法: python3 scripts/r3_sr_summary.py --dir runs/closedloop_sr/w1 [--md out.md]

每格读 `<cell>/ep_seed*.json`（episode-batch 的每集 result.json）。
输出每格 n / success / 成功率 + Wilson 95% 置信区间，并给 v13-vs-base 的差值区间。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

CELLS = [
    ("SitSofa", "v13", "sits_v13"),
    ("SitSofa", "base", "sits_base"),
    ("VisionNavi", "v13", "vn_v13"),
    ("VisionNavi", "base", "vn_base"),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_cell(d: Path) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(str(d / "ep_seed*.json"))):
        try:
            rows.append(json.load(open(p)))
        except Exception:
            pass
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    base = Path(a.dir)

    stats: dict[tuple[str, str], dict] = {}
    lines: list[str] = []
    for task, arm, cell in CELLS:
        rows = load_cell(base / cell)
        n = len(rows)
        k = sum(1 for r in rows if r.get("success"))
        lo, hi = wilson(k, n)
        steps = [int(r.get("episode_steps") or 0) for r in rows]
        falls = sum(1 for r in rows if r.get("failure_reason") == "fall")
        tos = sum(1 for r in rows if r.get("failure_reason") == "timeout")
        durs = [float(r.get("duration_sec") or 0) for r in rows]
        stats[(task, arm)] = dict(n=n, k=k, lo=lo, hi=hi, steps=steps,
                                  falls=falls, timeouts=tos, durs=durs,
                                  seeds=sorted(int(r.get("seed") or 0) for r in rows))
        pct = 100.0 * k / n if n else float("nan")
        step_med = sorted(steps)[len(steps) // 2] if steps else 0
        dur_med = sorted(durs)[len(durs) // 2] if durs else 0.0
        lines.append(
            f"{task:<11} {arm:<5} n={n:<3} success={k:<3} {pct:5.1f}%  "
            f"95%CI[{100*lo:4.1f},{100*hi:4.1f}]  fall={falls} timeout={tos}  "
            f"steps_med={step_med}  dur_med={dur_med:.0f}s"
        )

    print(f"[summary] dir={base}")
    for ln in lines:
        print(ln)

    print("\n[summary] 差值（v13 − base，逐任务）：")
    for task in ("SitSofa", "VisionNavi"):
        v = stats.get((task, "v13"), {})
        b = stats.get((task, "base"), {})
        if not v.get("n") or not b.get("n"):
            print(f"  {task}: 数据不全")
            continue
        pv, pb = v["k"] / v["n"], b["k"] / b["n"]
        # 两比例差的正态近似区间（Wald，够用）
        se = math.sqrt(pv * (1 - pv) / v["n"] + pb * (1 - pb) / b["n"])
        d = pv - pb
        print(f"  {task:<11} v13 {100*pv:5.1f}% ({v['k']}/{v['n']})  "
              f"base {100*pb:5.1f}% ({b['k']}/{b['n']})  "
              f"Δ={100*d:+5.1f}pp  ±{100*1.96*se:4.1f}  "
              f"→ [{100*(d-1.96*se):+5.1f}, {100*(d+1.96*se):+5.1f}]pp")

    print("\n[summary] 逐集明细：")
    for task, arm, cell in CELLS:
        rows = load_cell(base / cell)
        if not rows:
            continue
        print(f"  {cell}:")
        for r in rows:
            print(f"    seed={r.get('seed')} success={r.get('success')} "
                  f"reason={r.get('failure_reason')} steps={r.get('episode_steps')} "
                  f"dur={round(float(r.get('duration_sec') or 0), 1)}s")

    if a.md:
        Path(a.md).write_text("\n".join(lines) + "\n")
        print(f"\n[summary] md -> {a.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
