#!/usr/bin/env python3
"""同起点**配对**对比两套策略的闭环漂移（apples-to-apples）。

为什么必须配对（2026-10-30）：
  闭环评测的方差极大 —— 用 π0 自己的 80 集重新抽 20 集，pooled `pos p50` 就在
  [5.11, 7.04] cm、`pos p90` 在 [12.0, 36.1] cm 之间跳。所以"π0 的 80 集" vs
  "R1 的 20 集"这种**不同起点集合**的比法，量到的主要是起点差异，不是策略差异。
  正确做法：两套策略跑**同一批起点文件**（dump 文件名里带 clip 内帧号），然后逐集配对。

输出：
  1. 每个指标的**逐集值**（先每集聚合，再跨集取中位数）—— A / B 两侧
  2. **配对差** d = A − B 的中位数 + bootstrap 95% CI（符号一致性 = 赢/输多少集）
  3. pooled 分位数（只为和旧报告对齐，**不作为判据**）

用法：
  python3 scripts/eval_paired.py --a 'runs/j1_v2/round_1/eval80_seed1/b_*.pt' \
                                 --b 'runs/j1_scene2/round_1/bline/b_*.pt' \
                                 --name-a R1 --name-b pi0
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_drift_full as E  # noqa: E402

# 每集聚合函数：尾部看 p90，中位看 p50，整体看 mean
PER_EP = {
    "pos_mean_cm": ("root_pos_m", lambda x: x.mean() * 100),
    "pos_p50_cm": ("root_pos_m", lambda x: x.median() * 100),
    "pos_p90_cm": ("root_pos_m", lambda x: torch.quantile(x, 0.90) * 100),
    "tilt_mean_deg": ("tilt_err_deg", lambda x: x.mean()),
    "tilt_p90_deg": ("tilt_err_deg", lambda x: torch.quantile(x, 0.90)),
    "yaw_mean_deg": ("yaw_err_deg", lambda x: x.mean()),
    "yaw_p90_deg": ("yaw_err_deg", lambda x: torch.quantile(x, 0.90)),
    "yaw_rate_p50": ("yaw_rate_err_rad_s", lambda x: x.median()),
    "joint_p50_rad": ("joint_rad", lambda x: x.median()),
    "joint_p90_rad": ("joint_rad", lambda x: torch.quantile(x, 0.90)),
    "ori_p90_deg": ("root_ori_deg", lambda x: torch.quantile(x, 0.90)),
}


def _load_ref(path: Path):
    lib = torch.load(str(path), map_location="cpu", weights_only=False)
    ref = lib["ref"].float()
    starts, lengths = lib["starts"], lib["lengths"]
    last = torch.tensor([s + l - 1 for s, l in zip(starts, lengths)], dtype=torch.long)
    clip = torch.searchsorted(torch.tensor(starts, dtype=torch.long),
                              torch.arange(ref.shape[0], dtype=torch.long), right=True) - 1
    E.CLIP_LAST = last[clip.clamp(0, len(starts) - 1)]
    return ref


def _per_episode(files: list[str], ref: torch.Tensor) -> list[dict]:
    rows = []
    for f in files:
        comp = E.episode_components(Path(f), ref)
        row = {"file": os.path.basename(f), "fell": comp["_fell"],
               "max_tilt_deg": comp["_max_tilt"]}
        for name, (key, fn) in PER_EP.items():
            row[name] = float(fn(comp[key]))
        rows.append(row)
    return rows


def _bootstrap_ci(d: torch.Tensor, n: int = 5000, seed: int = 0):
    if d.numel() == 0:
        return (float("nan"), float("nan"))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, d.numel(), (n, d.numel()), generator=g)
    med = d[idx].median(dim=1).values
    return (float(torch.quantile(med, 0.025)), float(torch.quantile(med, 0.975)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="A 组 dump glob（新策略）")
    ap.add_argument("--b", required=True, help="B 组 dump glob（基线）")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--ref", type=Path, default=ROOT / "data/library/ft318_compact_v8.pt")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tasks", nargs="*", default=None, help="只看某些 task，如 task1 task2")
    a = ap.parse_args()

    ref = _load_ref(a.ref)
    fa, fb = sorted(glob.glob(a.a)), sorted(glob.glob(a.b))
    if a.tasks:
        want = tuple(f"b_{t}_" for t in a.tasks)
        fa = [f for f in fa if os.path.basename(f).startswith(want)]
        fb = [f for f in fb if os.path.basename(f).startswith(want)]
    na = {os.path.basename(f) for f in fa}
    nb = {os.path.basename(f) for f in fb}
    common = sorted(na & nb)
    if not common:
        raise SystemExit(f"[fail] 没有同起点：A={len(na)} B={len(nb)}")
    print(f"[配对] A={a.name_a} {len(na)} 集 / B={a.name_b} {len(nb)} 集 → 同起点交集 {len(common)} 集")
    miss_a, miss_b = sorted(nb - na), sorted(na - nb)
    if miss_a:
        print(f"  [warn] B 有但 A 没跑完: {len(miss_a)} 集（如 {miss_a[:3]}）")
    if miss_b:
        print(f"  [warn] A 有但 B 没有: {len(miss_b)} 集")

    map_a = {os.path.basename(f): f for f in fa}
    map_b = {os.path.basename(f): f for f in fb}
    pa = {r["file"]: r for r in _per_episode([map_a[n] for n in common], ref)}
    pb = {r["file"]: r for r in _per_episode([map_b[n] for n in common], ref)}

    print(f"\n{'指标':<18}{a.name_a:>10}{a.name_b:>10}{'Δ中位':>10}{'95%CI':>20}{'A更好':>9}")
    rep = {"n_paired": len(common), "name_a": a.name_a, "name_b": a.name_b, "metrics": {}}
    for name in PER_EP:
        va = torch.tensor([pa[n][name] for n in common])
        vb = torch.tensor([pb[n][name] for n in common])
        d = va - vb
        lo, hi = _bootstrap_ci(d)
        win = int((d < 0).sum())
        print(f"{name:<18}{va.median():10.3f}{vb.median():10.3f}{d.median():10.3f}"
              f"   [{lo:7.3f},{hi:7.3f}]{win:5d}/{len(common)}")
        rep["metrics"][name] = {
            "a_median": float(va.median()), "b_median": float(vb.median()),
            "delta_median": float(d.median()), "ci95": [lo, hi],
            "a_better": win, "n": len(common),
            "a_mean": float(va.mean()), "b_mean": float(vb.mean()),
            "a_p90": float(torch.quantile(va, 0.90)), "b_p90": float(torch.quantile(vb, 0.90)),
            "a_max": float(va.max()), "b_max": float(vb.max()),
        }

    fell_a = sum(1 for n in common if pa[n]["fell"])
    fell_b = sum(1 for n in common if pb[n]["fell"])
    mt_a = max(pa[n]["max_tilt_deg"] for n in common)
    mt_b = max(pb[n]["max_tilt_deg"] for n in common)
    print(f"\n摔倒: {a.name_a} {fell_a}/{len(common)}  {a.name_b} {fell_b}/{len(common)}"
          f"   | 最大倾角: {mt_a:.1f}° / {mt_b:.1f}°")
    rep["fell"] = {a.name_a: fell_a, a.name_b: fell_b}
    rep["max_tilt_deg"] = {a.name_a: mt_a, a.name_b: mt_b}

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        print(f"[out] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
