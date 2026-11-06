#!/usr/bin/env python3
"""对比多个联合微调臂的**训练内指标**（不涉及闭环，纯读 jsonl）。

用途：H=20 vs H=8 这类"只翻一个旋钮"的消融，先看训练内信号有没有按预期变化，
再去跑闭环。口径固定为**最后 50 个 optimizer step 的迭代（microbatch）均值**，
因为历史臂是 `batch16×accum4×800 iter = 200 updates`，新臂是
`batch8/rank×8卡×accum1×200 iter = 200 updates`，两者更新次数相同。

用法：
    python3 scripts/compare_train_metrics.py \
      --arm d5jT runs/S3_loop/d5jT_joint/train_main \
      --arm h20_8g runs/S3_loop/h20_8g_joint/train_main
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TRAIN_KEYS = [
    "loss", "loss/reward_raw", "loss/reward_norm", "rollout/reward",
    "act/rmse", "act/frac_out", "critic/v_ev", "critic/v_adv_H",
    "adv/ret_std", "adv/std_ratio", "adv/frac_inside_tau",
    "codec/out_of_range", "grad/head_preclip",
]
EVAL_KEYS = [
    "L_fm_val", "token_mse_decoded", "token_shift", "hand_mse_decoded",
    "imag_reward_vla", "imag_return_vla", "imag_return_norm_vla",
    "imag_state_vla", "imag_delta_mean", "imag_delta_std",
]


def _tail_mean(rows: list[dict], key: str, n: int = 50) -> float | None:
    vals = [float(r[key]) for r in rows[-n:] if key in r and r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs=2, action="append", required=True,
                    metavar=("LABEL", "DIR"))
    ap.add_argument("--tail", type=int, default=50)
    args = ap.parse_args()

    data = {}
    for label, d in args.arm:
        d = Path(d)
        mp, ep = d / "train_metrics.jsonl", d / "eval_curve.jsonl"
        rows = [json.loads(l) for l in open(mp)] if mp.exists() else []
        ev = [json.loads(l) for l in open(ep)] if ep.exists() else []
        ap_json = json.loads((d / "args.json").read_text()) if (d / "args.json").exists() else {}
        data[label] = dict(rows=rows, ev=ev, args=ap_json)

    print("## 训练配置\n")
    print("| 臂 | H | steps(iter) | batch/iter | accum | 有效 batch | updates |")
    print("|---|---|---|---|---|---|---|")
    for label, dat in data.items():
        a = dat["args"]
        bs, ac, st = a.get("batch_size"), a.get("grad_accum"), a.get("steps")
        upd = int(st) // max(1, int(ac)) if st and ac else "-"
        print(f"| {label} | {a.get('horizon')} | {st} | {bs} | {ac} | "
              f"{int(bs) * int(ac) if bs and ac else '-'} | {upd} |")

    print(f"\n## 训练内指标（最后 {args.tail} iter 均值）\n")
    print("| 指标 | " + " | ".join(data) + " |")
    print("|---|" + "---|" * len(data))
    for k in TRAIN_KEYS:
        cells = []
        for dat in data.values():
            v = _tail_mean(dat["rows"], k, args.tail)
            cells.append("-" if v is None else f"{v:.4g}")
        print(f"| `{k}` | " + " | ".join(cells) + " |")

    print("\n## 固定留出评测（eval_curve 最后一行）\n")
    print("| 指标 | " + " | ".join(data) + " |")
    print("|---|" + "---|" * len(data))
    for k in EVAL_KEYS:
        cells = []
        for dat in data.values():
            ev = dat["ev"]
            v = ev[-1].get(k) if ev else None
            cells.append("-" if v is None else f"{v:.4g}")
        print(f"| `{k}` | " + " | ".join(cells) + " |")

    print("\n## 相对第一步的变化（末行 − 首行）\n")
    print("| 指标 | " + " | ".join(data) + " |")
    print("|---|" + "---|" * len(data))
    for k in EVAL_KEYS:
        cells = []
        for dat in data.values():
            ev = dat["ev"]
            if not ev or k not in ev[-1] or k not in ev[0]:
                cells.append("-")
                continue
            d = float(ev[-1][k]) - float(ev[0][k])
            pct = (d / float(ev[0][k]) * 100.0) if abs(float(ev[0][k])) > 1e-12 else float("nan")
            cells.append(f"{d:+.4g} ({pct:+.1f}%)")
        print(f"| `{k}` | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
