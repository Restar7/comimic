#!/usr/bin/env python3
"""OpenDoor 成功率配对判读：读 runs/opendoor_sweep.csv，输出配对表 + 成功率检验 + 步数检验。

判据（先写死）：
  · 成功率：Fisher 精确检验（双侧），p < 0.05 才算差异显著；同时报 8 对里的胜负。
  · 步数（只比两边都成功的对）：配对差值的中位数 + 符号检验。
  · 任一项"打成平手"都算该维度无差异。

用法：
    python3 scripts/judge_opendoor_sweep.py [--csv runs/opendoor_sweep.csv]
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """双侧 Fisher 精确检验：[[a,b],[c,d]]。"""
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def hyper(x: int) -> float:
        return (math.comb(r1, x) * math.comb(n - r1, c1 - x)) / math.comb(n, c1)

    p_obs = hyper(a)
    return min(1.0, sum(hyper(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1)
                        if hyper(x) <= p_obs + 1e-12))


def sign_test(k: int, n: int) -> float:
    """双侧符号检验（n 个有效对里 k 个为正）。"""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs/opendoor_sweep.csv")
    ap.add_argument("--a", default="v13")
    ap.add_argument("--b", default="base")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    if not rows:
        print("(CSV 还是空的)"); return 0

    by = defaultdict(dict)
    for r in rows:
        if r["success"] in ("NA", ""):
            continue
        by[int(r["seed"])][r["arm"]] = dict(
            succ=int(r["success"]),
            steps=int(r["steps"]) if r["steps"] not in ("", "None") else None,
            sec=float(r["sec"]) if r["sec"] not in ("", "None") else None,
            disp=float(r["disp_m"]) if r["disp_m"] not in ("", "None") else None,
        )

    seeds = sorted(s for s in by if a.a in by[s] and a.b in by[s])
    print(f"配对集：{len(seeds)} 对（{a.a} vs {a.b}），seed={seeds}\n")
    print(f"{'seed':>4s} {'@A结果':>10s} {'A步数':>6s} {'@B结果':>10s} {'B步数':>6s} {'Δ步数(B-A)':>11s}")
    tot = {a.a: 0, a.b: 0}
    dsteps = []
    for s in seeds:
        x, y = by[s][a.a], by[s][a.b]
        tot[a.a] += x["succ"]; tot[a.b] += y["succ"]
        ds = ""
        if x["succ"] and y["succ"] and x["steps"] and y["steps"]:
            d = y["steps"] - x["steps"]
            dsteps.append(d); ds = f"{d:+d}"
        print(f"{s:4d} {'✅' if x['succ'] else '❌':>9s} {str(x['steps'] or '-'):>6s} "
              f"{'✅' if y['succ'] else '❌':>9s} {str(y['steps'] or '-'):>6s} {ds:>11s}")

    n = len(seeds)
    print(f"\n成功率：{a.a} {tot[a.a]}/{n}  vs  {a.b} {tot[a.b]}/{n}")
    if n:
        p = fisher_exact_2x2(tot[a.a], n - tot[a.a], tot[a.b], n - tot[a.b])
        print(f"  Fisher 精确（双侧）p = {p:.4f}  ⇒ {'显著' if p < 0.05 else '不显著'}")
    if dsteps:
        dsteps_sorted = sorted(dsteps)
        med = dsteps_sorted[len(dsteps) // 2]
        pos = sum(1 for d in dsteps if d < 0)          # B 步数更少 = A 更快
        print(f"\n双成功对：{len(dsteps)} 对；Δ步数中位 = {med:+d}（负 = {a.a} 更快）"
              f"；{a.a} 更快的对 = {pos}/{len(dsteps)}")
        print(f"  符号检验 p = {sign_test(pos, len(dsteps)):.4f}")
    else:
        print("\n没有'两边都成功'的对，步数无法配对比较。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
