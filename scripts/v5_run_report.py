#!/usr/bin/env python3
"""v5 run 报表：从**运行中或已完成**的 run 目录生成 markdown 报表（不依赖训练进程）。

用法::

    python3 scripts/v5_run_report.py --run runs/temporal_ac_bptt_v5/main/seed0
    python3 scripts/v5_run_report.py --run runs/temporal_ac_bptt_v5/main/seed0 \
        --out runs/temporal_ac_bptt_v5/main/seed0/interim_report.md

读取：`config.json`、`summary.json`（若有）、`train.jsonl`、`val.jsonl`、
`test_once.json`、`action_budget.csv`、`history_parity.json`、`deploy_contract_check.txt`、
`critic_gate.json`、`git_revision.txt`。所有数字都注明来源，不猜。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _f(x, nd=5):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:  # noqa: BLE001
        return "—"


def _mean(xs):
    xs = [float(x) for x in xs if x is not None and float(x) == float(x)]
    return float(np.mean(xs)) if xs else float("nan")


def build(run: Path) -> str:
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    train = [r for r in _read_jsonl(run / "train.jsonl") if "J" in r]
    val = [r for r in _read_jsonl(run / "val.jsonl") if r.get("eval") == "val"]
    test = json.loads((run / "test_once.json").read_text()) if (run / "test_once.json").exists() else None
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else None
    parity = json.loads((run / "history_parity.json").read_text()) if (run / "history_parity.json").exists() else {}
    gate = json.loads((run / "critic_gate.json").read_text()) if (run / "critic_gate.json").exists() else {}
    rev = (run / "git_revision.txt").read_text().strip().splitlines() if (run / "git_revision.txt").exists() else []
    deploy = (run / "deploy_contract_check.txt").read_text().strip().splitlines() if (run / "deploy_contract_check.txt").exists() else []

    L: list[str] = ["# v5 run 报表", "",
                    f"- run: `{run}`", f"- git: {' | '.join(rev)}"]
    if cfg:
        L += [f"- 配置: data_mode={'on-the-fly' if int(cfg.get('on_the_fly', 0)) else 'cached_pool'}"
              f" | seed={cfg.get('seed')} | updates={cfg.get('updates')} | batch={cfg.get('batch')}"
              f" | H={cfg.get('horizon')} | K={cfg.get('exec_k')} | cap={cfg.get('cap')}"
              f" | beta={cfg.get('beta')}",
              f"- eval_set: `{cfg.get('eval_set')}`"]
    if train:
        done = int(train[-1]["update"])
        sps = [r["s_per_step"] for r in train if r.get("s_per_step")]
        last = train[-1]
        L += ["", "## 训练进度（train.jsonl）", "",
              f"- 已到 update **{done}** / {cfg.get('updates', '?')}"
              f"，最近 10 次 log 的平均 J = {_mean([r['J'] for r in train[-10:]]):.5f}",
              f"- s/step 最近 10 次均值 = {_mean([r['s_per_step'] for r in train[-10:]]):.1f}"
              f"（整轮累计 {_mean(sps):.1f}）",
              f"- 最近一次：J={_f(last['J'])} loss={_f(last['loss'])} |Δ|={_f(last['delta_norm'],4)}"
              f" actRMS={_f(last['act_rms_mrad'],1)} mrad gn={_f(last['grad_norm'],2)}"
              f" cap={_f(last['cap'],3)}"]
    else:
        L += ["", "## 训练进度", "", "- 还没有 train.jsonl 记录"]

    if val:
        best = max(val, key=lambda r: r["dJ"])
        L += ["", "## held-out val（每 50 步）", "",
              "| update | J | base | ΔJ | actRMS mrad | peak mrad | delta_norm |",
              "|---|---|---|---|---|---|---|"]
        for r in val[-12:]:
            L.append(f"| {r['update']} | {_f(r['J'])} | {_f(r.get('J_base'))} | "
                     f"{r['dJ']:+.5f} | {_f(r.get('rms_mrad'),1)} | "
                     f"{_f(r.get('peak_mrad'),1)} | {_f(r.get('delta_norm'),4)} |")
        L += ["", f"- **best val ΔJ = {best['dJ']:+.5f} @ update {best['update']}**"
                  f"（最近一次 {val[-1]['dJ']:+.5f}）"]
    else:
        L += ["", "## held-out val", "", "- 还没有 val.jsonl 记录（每 50 步写一次）"]

    if test:
        L += ["", "## held-out test（独立 clip）", "",
              f"- J = {_f(test['J'])}，Base = {_f(test.get('J_base'))}，"
              f"**ΔJ = {test['dJ']:+.5f}**（n={test.get('n')}）",
              f"- 29D 动作预算：RMS={_f(test.get('rms_mrad'),2)} mrad，"
              f"peak={_f(test.get('peak_mrad'),2)} mrad，p99={_f(test.get('p99_mrad'),2)} mrad"]
    else:
        L += ["", "## held-out test", "", "- 尚未评测（训练结束时写 `test_once.json`）"]

    L += ["", "## 口径自检", "",
          f"- history parity: `{json.dumps(parity, ensure_ascii=False)}`",
          f"- critic: status={gate.get('status')} enabled={gate.get('critic_enabled')} "
          f"beta_cfg={gate.get('beta_cfg')}",
          "- deploy 契约自检: " + (" | ".join(deploy[-3:]) if deploy else "（未跑）")]

    if summary:
        L += ["", "## summary.json", "", "```json",
              json.dumps({k: summary[k] for k in summary if k in
                          ("data_mode", "seed", "updates", "best_val", "test_once", "base_J",
                           "T", "K", "H", "params", "uniform_over_eligible", "critic",
                           "wall_clock_s")}, ensure_ascii=False, indent=1), "```"]
    L += ["", f"_生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}_", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    text = build(a.run)
    out = a.out or (a.run / "interim_report.md")
    out.write_text(text)
    print(text)
    print(f"\n[report] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
