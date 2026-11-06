#!/usr/bin/env python3
"""Calibrate the root-position hinge for the 5-round full-length curriculum.

Why this exists
---------------
The old hinge (`e_h=0.35`, `lambda=1.5`, `cap=1.0`) was tuned for the old
`E_start=0.5` curriculum.  The new curriculum starts at `E_start=1.0`, so the
start-error distribution (and the H=20 rollout errors from those starts) is
different.  Keeping the old constants would silently change the reward shape.

Method
------
* Use the real B_901 closed-loop dumps and their per-tick instantaneous root
  error (`e_inst` from `self_targets_all.pt`).
* Select the same starts the trainer would select: stride=20, history=10,
  train split (drop every 5th replan), and `e_inst[start] <= E_start`.
* For every selected start, take the **actual next H=20 ticks** (not a WM
  hallucination) and pool their root errors.  This is the real closed-loop
  error distribution the hinge is applied to.
* Anchor: reproduce the original R1 calibration (`E=0.5`, `e_h=0.35`,
  `lambda=1.5`, `cap=1.0`) only as a reference; v2 does **not** reuse its
  sparse tail ratio.
* v2 objective (fixes the mid-field dead zone found on 2026-11-01):
  - `e_h = p45` of the pooled H=20 errors, so hinge is active for ~55% of
    states and there is a non-zero gradient at p50 as well as p90;
  - `cap = p95`;
  - solve `lambda` so `mean(lambda*hinge_raw) / mean(exp) = 0.30`
    (about 2.4x the old anchor ratio, but still below the exp term).
  The previous v1 used `e_h=p75` (active only 25%), which left 0.4–0.7 m
  errors with neither exp nor hinge gradient — the policy could stand still
  at ~0.6 m with no penalty.

Outputs a JSON table and prints a human-readable table.  No training/GPU.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import torch


def _tail_raw(e: torch.Tensor, eh: float, cap: float, slope: float = 0.5) -> torch.Tensor:
    y = torch.clamp(e - eh, min=0.0)
    return torch.clamp(y, max=cap) + slope * torch.clamp(y - cap, min=0.0)


def _pool_errors(episodes: dict[str, torch.Tensor], E: float, horizon: int) -> tuple[int, torch.Tensor]:
    errs: list[float] = []
    starts = 0
    for e in episodes.values():
        T = int(e.numel())
        for t in range(10, T, 20):
            if (t // 20) % 5 == 4:       # split="train"
                continue
            if float(e[t]) > E:
                continue
            if t + horizon > T:
                continue
            starts += 1
            errs.extend(float(x) for x in e[t:t + horizon])
    if not errs:
        raise SystemExit(f"[calib] no starts/errors for E={E}")
    return starts, torch.tensor(errs, dtype=torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="runs/B_901/*/dump/*.pt")
    ap.add_argument("--self-targets", default="runs/wm_real/self_targets_all.pt")
    ap.add_argument("--out", default="runs/S1_handover/hinge_calibration_5rounds.json")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--root-std", type=float, default=0.2,
                    help="root_pos_std used by the exp kernel")
    ap.add_argument("--rounds", default="1.0 1.2 1.4 1.6 1.8",
                    help="space-separated E_start values for R1..R5")
    ap.add_argument("--anchor-e", type=float, default=0.5)
    ap.add_argument("--anchor-eh", type=float, default=0.35)
    ap.add_argument("--anchor-lam", type=float, default=1.5)
    ap.add_argument("--anchor-cap", type=float, default=1.0)
    ap.add_argument("--e-quantile", type=float, default=0.45,
                    help="e_h quantile of pooled H=20 errors (v2 default p45; v1 used p75)")
    ap.add_argument("--target-ratio", type=float, default=0.30,
                    help="target mean(hinge)/mean(exp) (v2 default 0.30)")
    a = ap.parse_args()

    st = torch.load(a.self_targets, map_location="cpu", weights_only=False)
    st.pop("_meta", None)
    episodes: dict[str, torch.Tensor] = {}
    for f in sorted(glob.glob(a.dumps)):
        key = os.path.abspath(f)
        rec = st.get(key)
        if rec is None:
            continue
        e = rec.get("e_inst")
        if e is None:
            continue
        episodes[key] = e.float()
    if not episodes:
        raise SystemExit(f"[calib] no self-target entries matched {a.dumps}")

    # ---- anchor: the old, validated R1 hinge ----
    n0, e0 = _pool_errors(episodes, a.anchor_e, a.horizon)
    exp0 = torch.exp(-e0.square() / (a.root_std ** 2))
    raw0 = _tail_raw(e0, a.anchor_eh, a.anchor_cap)
    anchor_ratio = float((a.anchor_lam * raw0).mean() / exp0.mean().clamp_min(1e-12))
    anchor = {
        "E_start": a.anchor_e,
        "e_h": a.anchor_eh,
        "lambda": a.anchor_lam,
        "cap": a.anchor_cap,
        "starts": n0,
        "n_error_rows": int(e0.numel()),
        "e_p50": round(float(torch.quantile(e0, 0.50)), 4),
        "e_p75": round(float(torch.quantile(e0, 0.75)), 4),
        "e_p90": round(float(torch.quantile(e0, 0.90)), 4),
        "e_p95": round(float(torch.quantile(e0, 0.95)), 4),
        "active_frac": round(float((e0 > a.anchor_eh).float().mean()), 4),
        "cap_sat_frac": round(float((e0 > a.anchor_cap).float().mean()), 4),
        "mean_exp": round(float(exp0.mean()), 6),
        "mean_hinge": round(float((a.anchor_lam * raw0).mean()), 6),
        "hinge_over_exp": round(anchor_ratio, 6),
    }

    rounds = []
    for i, E in enumerate((float(x) for x in a.rounds.split()), start=1):
        n, e = _pool_errors(episodes, E, a.horizon)
        eh = round(float(torch.quantile(e, a.e_quantile)), 4)
        cap = round(float(torch.quantile(e, 0.95)), 4)
        # avoid a degenerate cap below e_h
        cap = max(cap, round(eh + 0.10, 4))
        exp = torch.exp(-e.square() / (a.root_std ** 2))
        raw = _tail_raw(e, eh, cap)
        lam = a.target_ratio * float(exp.mean()) / max(1e-12, float(raw.mean()))
        lam = round(lam, 4)
        row = {
            "round": f"R{i}",
            "E_start": E,
            "e_h": eh,
            "lambda": lam,
            "cap": cap,
            "starts": n,
            "n_error_rows": int(e.numel()),
            "e_p45": round(float(torch.quantile(e, 0.45)), 4),
            "e_p50": round(float(torch.quantile(e, 0.50)), 4),
            "e_p75": round(float(torch.quantile(e, 0.75)), 4),
            "e_p90": round(float(torch.quantile(e, 0.90)), 4),
            "e_p95": round(float(torch.quantile(e, 0.95)), 4),
            "active_frac": round(float((e > eh).float().mean()), 4),
            "cap_sat_frac": round(float((e > cap).float().mean()), 4),
            "mean_exp": round(float(exp.mean()), 6),
            "mean_hinge": round(float((lam * raw).mean()), 6),
            "hinge_over_exp": round(float((lam * raw).mean() / exp.mean().clamp_min(1e-12)), 6),
            "e_h_lt_E": bool(eh < E),
            "e_h_lt_cap": bool(eh < cap),
        }
        # local gradient at p90 (if it is below cap, slope = lambda; above = 0.5*lambda)
        _p50 = float(torch.quantile(e, 0.50))
        _p90 = float(torch.quantile(e, 0.90))
        row["grad_p50"] = round(lam if _p50 <= cap else 0.5 * lam, 5)
        row["grad_p90"] = round(lam * (0.5 if _p90 > cap else 1.0), 5)
        rounds.append(row)

    # hard calibration gates: the tail must be alive but not dominant
    bad = []
    for r in rounds:
        if not (0.45 <= r["active_frac"] <= 0.65):
            bad.append(f"{r['round']} active_frac={r['active_frac']:.3f} outside [0.45,0.65]")
        if r["cap_sat_frac"] > 0.08:
            bad.append(f"{r['round']} cap_sat_frac={r['cap_sat_frac']:.3f} > 0.08")
        if r["grad_p50"] <= 0 or r["grad_p90"] <= 0:
            bad.append(f"{r['round']} grad_p50/p90 not both >0")
        if not r["e_h_lt_E"] or not r["e_h_lt_cap"]:
            bad.append(f"{r['round']} e_h not inside E/cap")
    out = {
        "source": a.dumps,
        "self_targets": a.self_targets,
        "horizon": a.horizon,
        "root_std": a.root_std,
        "target": "e_h=p45, cap=p95, lambda solves mean(hinge)/mean(exp)=0.30 (mid-field gradient)",
        "anchor": anchor,
        "rounds": rounds,
        "calibration_ok": not bad,
        "failures": bad,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[calib] source={a.dumps} episodes={len(episodes)} horizon={a.horizon}")
    print(f"[calib] anchor R1 ratio={anchor_ratio:.6f} starts={n0} rows={anchor['n_error_rows']}")
    print(f"{'round':5s} {'E_start':>8s} {'e_h':>7s} {'lambda':>8s} {'cap':>6s} "
          f"{'p45':>6s} {'p50':>6s} {'p90':>6s} {'active':>7s} {'cap_sat':>8s} {'grad50':>8s} {'grad90':>8s}")
    for r in rounds:
        print(f"{r['round']:5s} {r['E_start']:8.2f} {r['e_h']:7.3f} {r['lambda']:8.4f} "
              f"{r['cap']:6.3f} {r['e_p45']:6.3f} {r['e_p50']:6.3f} {r['e_p90']:6.3f} "
              f"{r['active_frac']:7.1%} {r['cap_sat_frac']:8.1%} {r['grad_p50']:8.4f} {r['grad_p90']:8.4f}")
    print(f"[calib] saved {a.out}  ok={not bad}")
    if bad:
        print("[calib] ❌ " + "; ".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
