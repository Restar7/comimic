#!/usr/bin/env python3
"""从 sweep 的 dump 画/统计闭环漂移曲线（2026-11-02）。

不再只看整集均值，而是输出：
  - 每条 episode 的 root error vs t（每 100 步一个点）；
  - 首次发散时间（e > --diverge）；
  - 前 1000 步误差、全集均值、p90、final；
  - 两臂（main/ctrl）的汇总。

用法：
  python3 scripts/eval_drift_curve.py \
      --sweep runs/S3_loop/sweeps/exp_ref_base_vs_base_s0 ... \
      --packed runs/isaac_motion/ha_cmd_exec/packed.pt \
      --out runs/S3_loop/sweeps/exp_drift_curve.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch
import numpy as np


def _load_ref(packed: Path):
    p = torch.load(str(packed), map_location="cpu", weights_only=False)
    return p["ref"].float()


def _curve(dump: Path, ref, diverge: float) -> dict:
    d = torch.load(str(dump), map_location="cpu", weights_only=False)
    S = d["states"].float()
    fr = d["frames"].long().clamp(0, ref.shape[0] - 1)
    e = (S[:, :3] - ref[fr, :3]).norm(dim=-1).numpy()
    T = len(e)
    step = 100
    pts = [float(e[t:min(t + step, T)].mean()) for t in range(0, T, step)]
    over = np.nonzero(e > diverge)[0]
    first = int(over[0]) if len(over) else None
    return {
        "dump": str(dump),
        "T": int(T),
        "mean": float(e.mean()),
        "p90": float(torch.quantile(torch.as_tensor(e), 0.90)),
        "final": float(e[-1]),
        "first1000": float(e[:1000].mean()) if T > 0 else None,
        "diverge": first,
        "curve_100": pts,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", nargs="+", required=True)
    ap.add_argument("--packed", type=Path,
                    default=Path("runs/isaac_motion/ha_cmd_exec/packed.pt"))
    ap.add_argument("--out", type=Path, default=Path("runs/exp_drift_curve.json"))
    ap.add_argument("--arms", default="main,ctrl")
    ap.add_argument("--diverge", type=float, default=0.30)
    a = ap.parse_args()

    ref = _load_ref(a.packed)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    out = {"packed": str(a.packed), "diverge_threshold": a.diverge, "arms": {}}

    rows = []
    for sweep in a.sweep:
        for arm in arms:
            for dump in sorted(glob.glob(str(Path(sweep) / f"{arm}_r*" / "dump" / "*.pt"))):
                r = _curve(Path(dump), ref, a.diverge)
                r["arm"] = arm
                r["sweep"] = sweep
                rows.append(r)

    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        n = len(rs)
        out["arms"][arm] = {
            "n": n,
            "mean_root": round(sum(r["mean"] for r in rs) / n, 4),
            "mean_first1000": round(sum(r["first1000"] or 0 for r in rs) / n, 4),
            "mean_final": round(sum(r["final"] for r in rs) / n, 4),
            "n_diverge": sum(1 for r in rs if r["diverge"] is not None),
            "mean_curve_100": [
                round(sum(r["curve_100"][i] for r in rs if i < len(r["curve_100"])) /
                      max(1, sum(1 for r in rs if i < len(r["curve_100"]))), 4)
                for i in range(max(len(r["curve_100"]) for r in rs))
            ],
        }

    out["episodes"] = rows
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for arm, s in out["arms"].items():
        print(f"[drift] {arm:5s} n={s['n']:2d} mean={s['mean_root']:.3f} "
              f"first1000={s['mean_first1000']:.3f} final={s['mean_final']:.3f} "
              f"diverge={s['n_diverge']}/{s['n']}")
    print(f"[drift] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
