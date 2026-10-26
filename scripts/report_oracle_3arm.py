#!/usr/bin/env python3
"""oracle-token 三臂闭环结果汇总（按交接文档 §4 的四项判据出表）。

输入：`scripts/ha_isaac_batch.sh` 的产出目录（含 `runs/**/result.json`）。
判读只看四个数（同一任务、同一 seeds 并排）：
  1. 成功率            oracle > gt ?
  2. 摔倒率 / 存活步数  oracle 不摔的比例 ≥ gt ?
  3. 到目标最小距离     本仓库 result.json 不直接落这个量，用 `max_reward(_scaled)` 作代理（奖励里含接近目标项）
  4. 手闭合事件数       三臂必须一致 —— GT/ORACLE 臂由同一 npz 回放 `g1_hand_joints_14`，
                        用 `--hands-npz` 起服务即可保证；日志里看 `[replay] infer ... hand_max==`

用法：
    python3 scripts/report_oracle_3arm.py runs/oracle_3arm_DoubleDesk_20261026_150000
    python3 scripts/report_oracle_3arm.py <OUT> --json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys

ARMS_ORDER = {"gt": 0, "oracle": 1, "vla": 2}


def load_rows(out: pathlib.Path) -> list[dict]:
    rows = []
    for f in sorted(out.glob("runs/**/result.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        d.setdefault("model_label", "?")
        rows.append(d)
    return rows


def pct(a: int, b: int) -> str:
    return f"{a / b:.1%}" if b else "—"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    out = a.out
    if not out.exists():
        print(f"[report] 目录不存在: {out}", file=sys.stderr)
        return 1
    rows = load_rows(out)
    if not rows:
        print(f"[report] {out}/runs/**/result.json 里没有 episode（服务没起或 Isaac 没跑起来）")
        return 1

    agg = collections.defaultdict(lambda: {"n": 0, "ok": 0, "fall": 0, "timeout": 0,
                                           "steps": [], "maxrew": []})
    for d in rows:
        key = (str(d.get("model_label")), str(d.get("task", "?")))
        g = agg[key]
        g["n"] += 1
        g["ok"] += int(bool(d.get("success")))
        fr = str(d.get("failure_reason") or "")
        g["fall"] += int(fr == "fall")
        g["timeout"] += int(fr == "timeout")
        g["steps"].append(int(d.get("episode_steps") or 0))
        g["maxrew"].append(float(d.get("max_reward_scaled") or d.get("max_reward") or 0.0))

    keys = sorted(agg, key=lambda k: (k[1], ARMS_ORDER.get(k[0], 9), k[0]))
    if a.json:
        print(json.dumps({f"{v}|{t}": {"episodes": g["n"], "success": g["ok"],
                                        "success_rate": round(g["ok"] / g["n"], 4),
                                        "fall": g["fall"], "timeout": g["timeout"],
                                        "avg_steps": round(statistics.mean(g["steps"]), 1) if g["steps"] else 0,
                                        "avg_max_reward_scaled": round(statistics.mean(g["maxrew"]), 4) if g["maxrew"] else 0}
                            for (v, t), g in agg.items()}, ensure_ascii=False, indent=1))
        return 0

    print("=" * 96)
    print(f"oracle 三臂闭环：{out}")
    print("=" * 96)
    print(f"{'task':<14}{'variant':<10}{'episodes':>9}{'success':>9}{'rate':>8}"
          f"{'fall':>7}{'timeout':>9}{'avg_steps':>11}{'avg_maxR':>10}")
    for (v, t), g in [(k, agg[k]) for k in keys]:
        print(f"{t:<14}{v:<10}{g['n']:>9}{g['ok']:>9}{pct(g['ok'], g['n']):>8}"
              f"{g['fall']:>7}{g['timeout']:>9}"
              f"{(statistics.mean(g['steps']) if g['steps'] else 0):>11.1f}"
              f"{(statistics.mean(g['maxrew']) if g['maxrew'] else 0):>10.4f}")

    # §4 三条正式判据（GT 为基准，按任务对齐）
    print()
    print("-" * 96)
    print("§4 判读（每任务内以 gt 为基准）")
    print("-" * 96)
    verdict_lines = []
    tasks = sorted({t for _, t in agg})
    for t in tasks:
        gt = agg.get(("gt", t))
        if not gt:
            print(f"  [{t}] 没有 gt 臂，无法判读")
            continue
        for arm in ("oracle", "vla"):
            g = agg.get((arm, t))
            if not g:
                continue
            dr = g["ok"] / g["n"] - gt["ok"] / gt["n"]
            dfr = g["fall"] / g["n"] - gt["fall"] / gt["n"]
            dstep = (statistics.mean(g["steps"]) if g["steps"] else 0) - (statistics.mean(gt["steps"]) if gt["steps"] else 0)
            drew = (statistics.mean(g["maxrew"]) if g["maxrew"] else 0) - (statistics.mean(gt["maxrew"]) if gt["maxrew"] else 0)
            line = (f"  [{t}] {arm:6s} vs gt: 成功率 {dr:+.1%} | 摔倒率 {dfr:+.1%} | "
                    f"平均步数 {dstep:+.1f} | maxR {drew:+.4f}")
            print(line)
            verdict_lines.append((t, arm, dr, dfr))
    print()
    print("读法（交接文档 §4）：")
    print("  oracle > gt ≥ vla  → 想象增益可迁移 → 投 v3 训练（DR + 前验锚定 + WM 集成不确定性）")
    print("  oracle ≈ gt ≈ vla  → 搬不过去 → 收手，资源转数据与执行器")
    print("  oracle < gt        → WM 奖励方向有偏（模型利用）→ 先修 WM，别谈训练")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
