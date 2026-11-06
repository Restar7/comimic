#!/usr/bin/env python3
"""把若干个 ``eval_wm_multistep.py --out`` 的 json 汇成一张对比表（markdown + 控制台）。

为什么需要：WM 实验一轮会产出 5~10 个 ckpt 的评测 json（不同 arch / 阶段 / 早停点），
手工抄数字既慢又容易抄错口径（k=1/5/10/20、root 单位 cm、rot 度）。这里只做机械汇总，
并显式标注 ``data_source`` / ``warmup`` —— **跨口径的数字不能放同一列比较**。

用法：
    python3 scripts/compare_wm_eval.py runs/S1_handover/eval_wm_*.json
    python3 scripts/compare_wm_eval.py --baseline wm_d3.pt runs/S1_handover/eval_wm_*.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(paths):
    rows = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"[skip] {p} 不存在")
            continue
        d = json.loads(p.read_text())
        res = d.get("results") or {}
        if not res:
            print(f"[skip] {p} 里没有 results")
            continue
        for ck, per_k in res.items():
            if not per_k:
                continue
            rows.append({
                "file": p.name,
                "ckpt": Path(ck).name,
                "horizon": d.get("horizon"),
                "warmup": d.get("warmup", 0),
                "k": per_k,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--baseline", default="", help="相对基准的 ckpt 文件名（算 Δ%）")
    ap.add_argument("--csv", action="store_true", help="输出 csv 而不是 markdown")
    a = ap.parse_args()
    rows = load(a.json)
    # 同一个 ckpt 可能在多个 json 里出现（比如每次都带基准做并排）；按 (ckpt, warmup) 去重
    uniq, seen = [], set()
    for r in rows:
        key = (r["ckpt"], r["warmup"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    rows = uniq
    if not rows:
        return 1
    base = None
    if a.baseline:
        cand = [r for r in rows if r["ckpt"] == a.baseline]
        if cand:
            base = cand[0]
    cols = [("k=1", 1), ("k=5", 5), ("k=10", 10), ("k=20", 20)]
    head = ["ckpt", "warmup"] + [f"{n} root(cm)" for n, _ in cols] + \
           [f"{n} joint(rad)" for n, _ in cols] + [f"{n} rot(°)" for n, _ in cols]
    if a.csv:
        print(",".join(head))
    else:
        print("| " + " | ".join(head) + " |")
        print("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
    for r in rows:
        vals = [r["ckpt"], str(r["warmup"])]
        for _, k in cols:
            cell = r["k"].get(str(k)) or r["k"].get(k) or {}
            vals.append(f"{100 * cell['root']:.1f}" if cell else "-")
        for _, k in cols:
            cell = r["k"].get(str(k)) or r["k"].get(k) or {}
            vals.append(f"{cell['joint']:.3f}" if cell else "-")
        for _, k in cols:
            cell = r["k"].get(str(k)) or r["k"].get(k) or {}
            vals.append(f"{cell['rot']:.1f}" if cell else "-")
        if a.csv:
            print(",".join(vals))
        else:
            print("| " + " | ".join(vals) + " |")
    # Δ 表（k=20 主判据）
    if base is not None:
        print("\n相对基准 " + a.baseline + "（k=20，负=更好）：")
        b = base["k"].get("20") or base["k"].get(20)
        for r in rows:
            if r is base:
                continue
            c = r["k"].get("20") or r["k"].get(20)
            if not (b and c):
                continue
            dj = (c["joint"] / b["joint"] - 1) * 100
            dr = (c["root"] / max(b["root"], 1e-9) - 1) * 100
            dt = (c["rot"] / max(b["rot"], 1e-9) - 1) * 100
            print(f"  {r['ckpt']:28s} joint {dj:+6.1f}%  root {dr:+6.1f}%  rot {dt:+6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
