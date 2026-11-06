#!/usr/bin/env python3
"""联合微调跑完之后的**客观体检门**（2026-11-05）。

为什么需要它
------------
这三轮的历史摆在一起非常刺眼：

| 轮次 | 闭环真实结果 | act/rmse(末) | act/frac_out | 更新次数 | 想象 win 率 |
|---|---|---|---|---|---|
| round-3 | **0.111 m（崩）** | 1.463 | 0.90 | 1000 | **0.88（全场最好看）** |
| round-4 | **1.014 m ≈ base** | 0.055 | 0.00 | **100** | 0.69 |
| round-5 | 待测 | 0.92（step252 时） | 0.85 | 500 | 0.875 |

⇒ **"想象回报在涨 / win 率在涨"在本系统里是负信号**：它要么是真的学会了，
要么是动作被奖励推出训练流形、开始薅冻结 WM。这两者的唯一区别是**闭环**。
所以本门只做一件事：把"不能凭想象指标宣布成功"这件事变成可执行判据。

判据（三条，全部来自实测锚点）
------------------------------
1. **动作漂移 `act/rmse`**：round-4（唯一 ≈base 的一轮）全程 0.055–0.112；
   round-3（崩）从 1.09 涨到 1.46。`act/rmse > 0.5` 或 `act/frac_out > 0.5` ⇒ 出流形区。
2. **更新次数**：`updates = steps / grad_accum`。round-4 只有 100 次更新就停了；
   漂移是"更新次数"的函数 ⇒ 报出 updates 才能跟历史对齐（这一步最容易搞错）。
3. **机械不变量**：`grad/head_preclip` 必须有限、`rollout/survival` 不能塌
   （后者塌了说明 WM 的 soft_termination 又在误判，见 `--term-off` 的来由）。

用法::

    python3 scripts/check_ft_run.py runs/joint_v5                 # 体检，打印判读
    python3 scripts/check_ft_run.py runs/joint_v5 --gate           # 出流形区就 exit 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 历史锚点（实测，别改；改了就等于把这三轮的教训擦掉）
ANCHOR = {
    "round3_collapse": {"run": "joint_v3_round3A", "loop_m": 0.111,
                        "act_rmse_end": 1.463, "updates": 1000},
    "round4_ok": {"run": "joint_v4_trusttight", "loop_m": 1.014,
                  "act_rmse_end": 0.055, "updates": 100},
}
DRIFT_WARN = 0.50      # act/rmse
FRACOUT_WARN = 0.50    # 视线内关节占比
PINNED_WARN = 0.02     # 低于它就是"被钉死"（实测 weight=1.0 时 0.013，完全不学）


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--gate", action="store_true", help="出流形区时 exit 3")
    ap.add_argument("--curve", type=int, default=0,
                    help="额外逐 step 打印前 N 个里程碑（0=不打印）")
    a = ap.parse_args()

    run = a.run if a.run.is_absolute() else (Path.cwd() / a.run)
    if not run.exists():
        run = ROOT / a.run
    rows = load_jsonl(run / "train_metrics.jsonl")
    if not rows:
        print(f"[ft] ❌ 没有 train_metrics.jsonl: {run}")
        return 2
    try:
        args = json.loads((run / "args.json").read_text())
    except Exception:
        args = {}
    steps = int(rows[-1]["step"])
    accum = int(args.get("grad_accum", 1) or 1)
    batch = int(args.get("batch_size", 1) or 1)
    updates = max(1, steps // accum)
    print(f"[ft] {run.name}: steps={steps} batch={batch} accum={accum} "
          f"⇒ **更新次数 {updates}**、每更新 {batch} 窗/卡")

    mine = [r for r in rows if r.get("act/rmse") is not None]
    last = rows[-1]

    def tail(key: str, n: int = 30) -> float | None:
        v = [r[key] for r in mine[-n:] if r.get(key) is not None]
        return sum(v) / len(v) if v else None

    act_end = tail("act/rmse")
    frac_end = tail("act/frac_out")
    trust_end = tail("trust/rmse")
    surv_min = min((r.get("rollout/survival", 1.0) for r in rows), default=1.0)
    grad_max = max((abs(r.get("grad/head_preclip", 0.0)) for r in rows), default=0.0)
    rew = [r.get("loss/reward_norm") for r in rows if r.get("loss/reward_norm") is not None]

    print(f"[ft] 动作漂移 act/rmse(近30均) = {act_end if act_end is None else f'{act_end:.3f}'}"
          f"   frac_out = {frac_end if frac_end is None else f'{frac_end:.2f}'}")
    print(f"[ft] token 锚 trust/rmse = {trust_end if trust_end is None else f'{trust_end:.4f}'}"
          f"   survival_min = {surv_min:.3f}   grad_max = {grad_max:.2f}")
    if rew:
        print(f"[ft] 训练奖励 reward_norm: 前10均 {sum(rew[:10])/min(10, len(rew)):.3f}"
              f" → 末10均 {sum(rew[-10:])/min(10, len(rew)):.3f}")

    # 想象指标（有的话只用来**反证**，不能用来判成功）
    ev = load_jsonl(run / "eval_curve.jsonl")
    if len(ev) >= 2:
        e0, e1 = ev[0], ev[-1]
        print(f"[ft] 想象回报 {e0['imag_return_vla']:.3f} → {e1['imag_return_vla']:.3f}   "
              f"win 率 {e0.get('imag_delta_win', 0):.3f} → {e1.get('imag_delta_win', 0):.3f}"
              "   ⚠ 只作反证：round-3 崩前 win 率 0.88（全场最好）")

    print("\n[ft] 历史锚点：")
    for k, v in ANCHOR.items():
        print(f"       {k:<16} 闭环 {v['loop_m']:.3f} m   act/rmse_end {v['act_rmse_end']:.3f}"
              f"   更新 {v['updates']}")

    bad = []
    warn = []
    if not math.isfinite(grad_max) or grad_max == 0.0:
        bad.append("梯度非有限/为 0")
    # survival < 1 在 2026-11-05 之前的所有 run 里都有（那时还没 --term-off），
    # 所以它只报警不当处决；round-5 driver 已默认 TERM_OFF=1。
    if surv_min < 0.99:
        warn.append(f"survival 最低 {surv_min:.3f}（soft_termination 误判 ⇒ 该有 --term-off）")
    if grad_max > 50:
        warn.append(f"grad 峰值 {grad_max:.1f}（round-3 崩时 683；round-4 峰值 15）")
    drift_out = (act_end is not None and act_end > DRIFT_WARN) or \
                (frac_end is not None and frac_end > FRACOUT_WARN)

    # ★ 2026-11-05：**刹车踩过头**也要判出来。原来 act/rmse=0.013 会被判"✅ 在可信域内"，
    #   但那是"策略被钉在 base 上、几乎没学"（实测：想象回报 −2.809→−2.807 完全不动）。
    pinned = act_end is not None and act_end < PINNED_WARN
    if pinned:
        warn.append(f"动作漂移只有 {act_end:.4f}（< {PINNED_WARN}）⇒ **刹车踩过头，策略被钉在 base 上**"
                    f"；实测这一型想象回报完全不涨。调小 ACT_WEIGHT 重跑，别指望它自己学会")
    print()
    for w in warn:
        print(f"[ft] ⚠ {w}")
    if bad:
        print("[ft] ❌ 机械层面就有问题：" + "; ".join(bad))
        return 2
    if drift_out:
        print("[ft] ⚠ **动作已出可信域**：act/rmse"
              f" {act_end:.3f} / frac_out {frac_end:.2f} 超过 warn 线 "
              f"({DRIFT_WARN:.2f} / {FRACOUT_WARN:.2f})。")
        print("       这一型与 round-3（闭环 0.111 m 崩）同构，不是与 round-4（1.014 m）同构。")
        print("       **想象指标再好也不能宣布成功**：必须闭环 A/B，且优先测早期检查点"
              "（round-4 只跑 100 次更新就停）。")
        return 3 if a.gate else 0
    print(f"[ft] ✅ 在可信域内（act/rmse"
          f" {act_end if act_end is not None else float('nan'):.3f} ≤ {DRIFT_WARN:.2f}）"
          "：这一型与 round-4 同构，闭环大概率不劣于 base。仍需闭环确认。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
