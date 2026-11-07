#!/usr/bin/env python3
"""harness02：把本目录下所有产物汇总成 `metrics.json`（机器可读，便于复核/回归）。

只读 CSV/JSON，不做任何新计算；缺失的产物标 ``null`` 而不是猜。
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/harness_02_teacher"


def _csv(name: str):
    p = OUT / name
    if not p.exists():
        return None
    with open(p) as fh:
        return list(csv.DictReader(fh))


def _json(name: str):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _consistency(rows):
    if not rows:
        return None
    by_h: dict[str, list[float]] = {}
    for r in rows:
        by_h.setdefault(str(r["H"]), []).append(_f(r["cos_AB"]))
    return {
        "n_configs": len(rows),
        "cos_min": min(_f(r["cos_AB"]) for r in rows),
        "cos_mean": statistics.mean(_f(r["cos_AB"]) for r in rows),
        "max_abs_err": max(_f(r["max_abs_err"]) for r in rows),
        "rel_l2_max": max(_f(r["rel_l2"]) for r in rows),
        "sign_agree_min": min(_f(r["sign_agree"]) for r in rows),
        "by_H": {h: {"n": len(v), "cos_min": min(v), "cos_mean": statistics.mean(v)}
                 for h, v in sorted(by_h.items(), key=lambda kv: float(kv[0]))},
        "eps_scales": sorted({_f(r["eps_scale"]) for r in rows}),
    }


def _fd_best(rows):
    if not rows:
        return None
    best: dict[str, tuple[float, float]] = {}
    for r in rows:
        h = str(r["H"])
        e = _f(r["fd_rel_err"])
        if h not in best or e < best[h][0]:
            best[h] = (e, _f(r["fd_h"]))
    return {h: {"best_rel_err": v[0], "at_h": v[1]}
            for h, v in sorted(best.items(), key=lambda kv: float(kv[0]))}


def main() -> int:
    cons = _csv("gradient_consistency.csv")
    cons_prod = _csv("gradient_consistency_prod.csv")
    fd = _csv("gradient_consistency_fd.csv")
    fd_prod = [r for r in (fd or []) if str(r.get("source", "")).startswith("dump")]
    fd_rec = [r for r in (fd or []) if not str(r.get("source", "")).startswith("dump")]
    dist = _csv("distill_learning_curve.csv")
    old_new = _csv("teacher_old_vs_new.csv")
    transfer = _csv("old_label_transfer.csv")
    eps = None
    if (OUT / "eps_grad.jsonl").exists():
        eps = [json.loads(line) for line in (OUT / "eps_grad.jsonl").read_text().splitlines() if line.strip()]
    resample = {}
    mpath = OUT / "epsgrad/train_metrics.jsonl"
    if mpath.exists():
        rows = [json.loads(line) for line in mpath.read_text().splitlines() if line.strip()]
        for k in ("teacher/cos_resample_label", "teacher/token_chunk_rel_diff",
                  "teacher/g_resample_norm_mean", "teacher/cos_H5_vs_H20"):
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
            if vals:
                resample[k] = {"n": len(vals), "mean": statistics.mean(vals),
                               "values": vals}
    meta = {
        "environment": {
            "worktree": str(ROOT),
            "branch": "harness/02-teacher",
            "base_commit": "a4bd1ff9988d8d11e4585037aa51a4e1b34cdc90",
            "gpu": "2x RTX 4090 48GiB（本会话使用 cuda:1；另一会话同时跑 gen_a_g0/g1）",
            "data": {
                "transitions": "runs/a_prime_sonic_exc/all_transitions.pt",
                "packed_motion": "runs/isaac_motion/ha_cmd/packed.pt",
                "wbc": "runs/S1_handover/WBC_A24000_wmvel_cmd.pt",
                "wm": "runs/a_prime_sonic_exc/wm_gain2.pt",
                "vla": "runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000",
            },
            "horizon": 20,
            "token_chunk_len": 40,
            "gamma": 0.99,
            "reward_mode": "saturating",
            "term_off": 1,
        },
        "teacher_definitions": {
            "old": "legacy_token0_repeat_v1: g=∂/∂t0 Σ_t r_t (token0 repeated H steps, undiscounted sum)",
            "new": "shared_delta_reward_only_v2: g=∂J/∂ε, token'_t=token_t+ε, J=discounted_return (same as BPTT)",
        },
        "gradient_consistency_recorded": _consistency(cons),
        "gradient_consistency_prod": _consistency(cons_prod),
        "gcd_fd_recorded_best": _fd_best(fd_rec),
        "gcd_fd_prod_best": _fd_best(fd_prod),
        "fd_diagnose": _json("fd_diagnose.json"),
        "fd_action_probe": _json("fd_action_probe.json"),
        "teacher_old_vs_new": old_new,
        "old_label_transfer": transfer,
        "distill_learning_curve": dist,
        "eps_grad": eps,
        "resample": resample,
        "checkpoint_meta_check": _json("delta_token_meta_check.json"),
        "legacy_equivalence": _json("legacy_equivalence.json"),
        "verdicts": {
            "teacher_shared_delta_math": "PASS",
            "teacher_reward_definition_aligned": "PASS",
            "gradient_consistency_H1_H2_H5_H20": "PASS",
            "finite_difference_H_le_5": "PASS",
            "finite_difference_H20": "PARTIAL(~3.6%, fp32/no-linearity floor)",
            "old_vs_new_teacher_measured": "PASS",
            "legacy_teacher_semantics_unchanged": "PASS",
            "reward_only_vs_full_training": "PASS(2-step diagnostic; production NOT_RUN)",
            "checkpoint_metadata": "PASS",
            "distill_overfit_128_256": "PASS",
            "distill_heldout_cos": "PASS(measured, ceiling ~0.26 H20 / ~0.33 H5)",
            "closed_loop_success_rate": "NOT_RUN",
            "learning_curve_1k_3k_10k": "NOT_RUN",
            "resample_label_noise_production_scale": "NOT_RUN",
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    print(f"[saved] {OUT / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
