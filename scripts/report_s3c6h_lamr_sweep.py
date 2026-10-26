#!/usr/bin/env python3
"""S3 λ_r 消融汇总（C6h 执行器 / ha_cmd 命令流库 / 64 窗口固定评测）。

把 runs/s3c6h_lam* 各臂的 eval_curve.jsonl + train_metrics.jsonl + summary.json
并排打印，并按交接文档 §8 给出判读用读数（不替代闭环）。

用法：
    python3 scripts/report_s3c6h_lamr_sweep.py
    python3 scripts/report_s3c6h_lamr_sweep.py --arms s3c6h_lam0 s3c6h_lam0.1 s3c6h_lam1.0
    python3 scripts/report_s3c6h_lamr_sweep.py --json      # 机器可读

判读口径（§8）：
  1. 不退化（硬）：token_mse_decoded / hand_mse_decoded 与 λ_r=0 同 step 持平（±20%）；
     hand_gt170_pred 与标签(0.11384)的差距不扩大。
  2. 有增益（软）：imag_return_vla 高于 λ_r=0 且逼近 imag_return_gt；imag_survival_vla 不降。
  3. 稳定性：grad_ratio 落在 0.1–0.5。
  4. L_fm(val) 仅作诊断，不是门禁。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ARMS = ["s3c6h_lam0", "s3c6h_lam0.3", "s3c6h_lam0.1", "s3c6h_lam1.0"]

EVAL_KEYS = [
    "L_fm_val",
    "token_mse_decoded",
    "hand_mse_decoded",
    "hand_gt170_pred",
    "token_shift",
    "imag_return_vla",
    "imag_return_gt",
    "imag_survival_vla",
    "imag_reward_vla",
]
SHORT = {
    "L_fm_val": "L_fm",
    "token_mse_decoded": "tok",
    "hand_mse_decoded": "hand",
    "hand_gt170_pred": "gt170",
    "token_shift": "tshift",
    "imag_return_vla": "ret_vla",
    "imag_return_gt": "ret_gt",
    "imag_survival_vla": "surv",
    "imag_reward_vla": "rew",
}
# 硬条件阈值（相对 λ_r=0 同 step）
TOL = 0.20
GRAD_BAND = (0.1, 0.5)


def _jl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_arm(runs: str, arm: str) -> Dict[str, Any]:
    d = os.path.join(runs, arm)
    evals = _jl(os.path.join(d, "eval_curve.jsonl"))
    metrics = _jl(os.path.join(d, "train_metrics.jsonl"))
    summary: Optional[Dict[str, Any]] = None
    sp = os.path.join(d, "summary.json")
    if os.path.exists(sp):
        try:
            with open(sp, "r", encoding="utf-8") as fh:
                summary = json.load(fh)
        except json.JSONDecodeError:
            summary = None

    ratios = [m["grad_ratio"] for m in metrics if isinstance(m.get("grad_ratio"), (int, float))]
    oor = [m["codec/out_of_range"] for m in metrics if isinstance(m.get("codec/out_of_range"), (int, float))]
    return {
        "arm": arm,
        "dir": d,
        "evals": {int(e["step"]): e for e in evals},
        "metrics": metrics,
        "summary": summary,
        "grad_ratio": ratios,
        "out_of_range": oor,
        "last_step": int(metrics[-1]["step"]) if metrics else None,
        "done": summary is not None,
    }


def fmt(x: Any, nd: int = 5) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(REPO, "runs"))
    ap.add_argument("--arms", nargs="*", default=DEFAULT_ARMS)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    arms = [load_arm(args.runs, a) for a in args.arms]
    arms = [a for a in arms if a["evals"] or a["metrics"]]
    if not arms:
        print("[report] 没有可读的臂（检查 runs/<arm>/eval_curve.jsonl）")
        return 1

    base = next((a for a in arms if a["arm"] == "s3c6h_lam0"), None)
    steps = sorted({s for a in arms for s in a["evals"]})

    if args.json:
        print(json.dumps(
            {
                "arms": [
                    {
                        "arm": a["arm"],
                        "lambda_r": a["evals"][max(a["evals"])]["lambda_r"] if a["evals"] else None,
                        "last_step": a["last_step"],
                        "done": a["done"],
                        "grad_ratio": {
                            "n": len(a["grad_ratio"]),
                            "median": statistics.median(a["grad_ratio"]) if a["grad_ratio"] else None,
                            "min": min(a["grad_ratio"]) if a["grad_ratio"] else None,
                            "max": max(a["grad_ratio"]) if a["grad_ratio"] else None,
                            "last": a["grad_ratio"][-1] if a["grad_ratio"] else None,
                        },
                        "out_of_range_median": statistics.median(a["out_of_range"]) if a["out_of_range"] else None,
                        "eval": [{k: v for k, v in e.items()} for e in a["evals"].values()],
                        "summary": a["summary"],
                    }
                    for a in arms
                ],
                "label_hand_gt170": (base["evals"][min(base["evals"])].get("hand_gt170_label") if base else None),
            },
            ensure_ascii=False,
            indent=1,
        ))
        return 0

    print("=" * 108)
    print("S3 λ_r 消融（C6h / ha_cmd / 64 窗口固定评测）")
    print("=" * 108)
    for a in arms:
        lam = a["evals"][max(a["evals"])]["lambda_r"] if a["evals"] else "?"
        gr = a["grad_ratio"]
        gr_s = (f"grad_ratio n={len(gr)} med={statistics.median(gr):.3f} "
                f"[{min(gr):.3f},{max(gr):.3f}] last={gr[-1]:.3f}") if gr else "grad_ratio —（无奖励项/未采）"
        print(f"  {a['arm']:<18} λ_r={lam:<5} 到 step {str(a['last_step']):>5}  "
              f"{'完成' if a['done'] else '未完成'}  {gr_s}")
    print()

    print("-" * 108)
    print("逐 step 读数（每格 = 64 窗口）")
    print("-" * 108)
    hdr = f"{'step':>5}  " + "  ".join(f"{a['arm'].replace('s3c6h_',''):>10}" for a in arms)
    for key in EVAL_KEYS:
        print(f"\n[{SHORT[key]}] {key}")
        print(hdr)
        for st in steps:
            row = f"{st:>5}  "
            cells = []
            for a in arms:
                e = a["evals"].get(st)
                cells.append(f"{fmt(e.get(key)):>10}" if e else f"{'—':>10}")
            print(row + "  ".join(cells))

    # §8 判读
    print()
    print("=" * 108)
    print("§8 判读（相对 λ_r=0 同 step；基线缺失时退化为绝对读数）")
    print("=" * 108)
    label_gt170 = None
    if base and base["evals"]:
        label_gt170 = base["evals"][min(base["evals"])].get("hand_gt170_label")
    for a in arms:
        if a is base:
            continue
        print(f"\n--- {a['arm']} ---")
        for st in sorted(a["evals"]):
            e = a["evals"][st]
            b = base["evals"].get(st) if base else None
            parts = []
            for key, tol in (("token_mse_decoded", TOL), ("hand_mse_decoded", TOL)):
                if b and b.get(key):
                    rel = e[key] / b[key] - 1.0
                    parts.append(f"{SHORT[key]} {rel:+.1%}{'  OK' if abs(rel) <= tol else '  ✗超±20%'}")
            if b and b.get("imag_return_vla") is not None:
                d = e["imag_return_vla"] - b["imag_return_vla"]
                parts.append(f"ret_vla {d:+.4f}{'  ↑' if d > 0 else ('  =' if abs(d) < 1e-4 else '  ↓')}")
            if b and b.get("imag_survival_vla") is not None:
                d = e["imag_survival_vla"] - b["imag_survival_vla"]
                parts.append(f"surv {d:+.4f}{'  OK' if d > -1e-3 else '  ↓'}")
            if label_gt170 is not None and e.get("hand_gt170_pred") is not None:
                gap = abs(e["hand_gt170_pred"] - label_gt170)
                bgap = abs(b["hand_gt170_pred"] - label_gt170) if b and b.get("hand_gt170_pred") is not None else None
                txt = f"gt170 gap {gap:.4f}"
                if bgap is not None:
                    txt += f" (λ0 {bgap:.4f}, {gap - bgap:+.4f}){'  OK' if gap <= bgap + 0.01 else '  ✗扩大'}"
                parts.append(txt)
            print(f"  step {st:>4}: " + " | ".join(parts))

        gr = a["grad_ratio"]
        if gr:
            lo, hi = min(gr), max(gr)
            med = statistics.median(gr)
            inband = GRAD_BAND[0] <= med <= GRAD_BAND[1]
            print(f"  grad_ratio: med={med:.3f} range=[{lo:.3f},{hi:.3f}] "
                  f"→ {'在 0.1–0.5 带内' if inband else '⚠ 中位数越界（按比例调 λ_r，勿动 lr/H/K）'}")
        if a["out_of_range"]:
            print(f"  codec/out_of_range: med={statistics.median(a['out_of_range']):.3f} "
                  f"(正常 2–11%)")

    print()
    print("注：L_fm(val) 只作“在学习、没发散”诊断，不是门禁；最终结论仍以闭环为准。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
