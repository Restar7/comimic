#!/usr/bin/env python3
"""Collect the three ablation runs into one markdown report.

Reads whatever ``scripts/run_ablations.sh`` produced under ``--root`` (missing
arms are reported as "not run" instead of crashing, so the report is always
regenerable mid-sweep):

    <root>/a1_success_rate/summary.json          wall success rates + Wilson CI
    <root>/a2_fmbackends/<backend>/eval_curve.jsonl
    <root>/a3_bptt_vs_ppo/{bptt,ppo}/eval_curve.jsonl

Usage:
    python3 scripts/ablation_report.py --root runs/ablations
    python3 scripts/ablation_report.py --root runs/ablations --out docs/ABLATIONS_REPORT.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def fmt(x: float | None, nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def curve_table(title: str, arms: dict[str, list[dict]], note: str = "") -> str:
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    if not arms or all(not v for v in arms.values()):
        lines += ["_not run_", ""]
        return "\n".join(lines)
    lines += [
        "| arm | evals | step@first | step@last | token_mse first → last | state_term first → last "
        "| rollout_reward first → last |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, rows in arms.items():
        if not rows:
            lines.append(f"| {name} | 0 | — | — | — | — | — |")
            continue
        first, last = rows[0], rows[-1]
        lines.append(
            f"| {name} | {len(rows)} | {first.get('step')} | {last.get('step')} | "
            f"{fmt(first.get('token_mse'))} → {fmt(last.get('token_mse'))} | "
            f"{fmt(first.get('state_term'))} → {fmt(last.get('state_term'))} | "
            f"{fmt(first.get('rollout_reward'))} → {fmt(last.get('rollout_reward'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def success_table(path: Path) -> str:
    lines = ["### Ablation 1 — closed-loop success rate", ""]
    if not path.exists():
        lines += ["_not run_ (need `runs/ckpt_vla_ft4000` and the MuJoCo closed loop)", ""]
        return "\n".join(lines)
    summary = json.loads(path.read_text())
    arms = list(summary.get("per_arm", {}))
    if not arms:
        lines += ["_empty summary_", ""]
        return "\n".join(lines)
    tasks = sorted({t for a in arms for t in summary["per_arm"][a]["per_task"]})
    lines.append("| task | " + " | ".join(arms) + " |")
    lines.append("|---|" + "|".join(["---"] * len(arms)) + "|")
    for task in tasks:
        cells = []
        for arm in arms:
            v = summary["per_arm"][arm]["per_task"][task]
            cells.append(f"{v['success']}/{v['episodes']}")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    cells = []
    for arm in arms:
        v = summary["per_arm"][arm]
        lo, hi = v["ci95"]
        cells.append(f"**{v['success']}/{v['episodes']} ({v['rate'] * 100:.0f}%)** "
                     f"[{lo * 100:.0f}, {hi * 100:.0f}]")
    lines.append("| **all** | " + " | ".join(cells) + " |")
    lines += ["", "Wilson 95% CI in brackets. Binary episodes → ≥2 episode difference to mean anything.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("runs/ablations"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.root

    parts = ["# 三个消融实验汇总", "", f"_source: `{root}`_", ""]

    parts.append(success_table(root / "a1_success_rate" / "summary.json"))

    ab2 = {
        p.name: read_jsonl(p / "eval_curve.jsonl")
        for p in sorted((root / "a2_fmbackends").glob("*"))
        if p.is_dir()
    }
    parts.append(curve_table(
        "Ablation 2 — FM gradient backends (equal wall-clock budget)",
        ab2,
        note="`rollout_reward` is the quantity every backend is trying to maximise; "
             "`token_mse` only measures the Euler chain's output, not its gradient.",
    ))

    ab3 = {
        "BPTT (pathwise, exact)": read_jsonl(root / "a3_bptt_vs_ppo" / "bptt" / "eval_curve.jsonl"),
        "PPO (score function)": read_jsonl(root / "a3_bptt_vs_ppo" / "ppo" / "eval_curve.jsonl"),
    }
    parts.append(curve_table(
        "Ablation 3 — BPTT vs PPO under an identical wall-clock budget",
        ab3,
        note="Question is binary first (does each arm move at all), then comparative.",
    ))

    text = "\n".join(parts)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"[wrote] {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
