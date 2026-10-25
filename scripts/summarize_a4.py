#!/usr/bin/env python3
"""Summarise the A4 grid (``runs/ablations/A4_wm_horizon/*/eval_curve.jsonl``) into markdown.

Reports, per cell: number of eval points, steps reached in the fixed wall clock, and the
curves of ``rollout_reward`` / ``state_term`` / ``token_mse`` / ``hand_mse`` (first, last,
best, and the slope of a least-squares fit against elapsed seconds — the only thing that is
comparable across arms with different per-step cost).

Usage:
    python3 scripts/summarize_a4.py --root runs/ablations/A4_wm_horizon
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = ("rollout_reward", "state_term", "token_mse", "hand_mse")
#: 跨 H 可比的固定前缀口径（前 8 步，见 train_joint.run_eval 的 PREFIX_STEPS）
PREFIX_KEYS = ("rollout_reward@8", "state_term@8", "survival@8")


def load_curve(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def fmt(v) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return "—"
    return f"{f:.4g}"


def describe(rows: list[dict], key: str) -> dict:
    xs, ys = [], []
    for r in rows:
        if key in r and isinstance(r[key], (int, float)) and r[key] == r[key]:
            xs.append(float(r.get("elapsed_s", 0.0)))
            ys.append(float(r[key]))
    if not ys:
        return {"n": 0, "first": None, "last": None, "best": None, "slope_per_min": None}
    return {
        "n": len(ys),
        "first": ys[0],
        "last": ys[-1],
        "best": max(ys) if (key.startswith("rollout_reward") or key in ("token_mse", "survival", "survival@8")) else min(ys),
        "slope_per_min": slope(xs, ys) * 60.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("runs/ablations/A4_wm_horizon"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cells = sorted(p.name for p in args.root.iterdir() if p.is_dir()) if args.root.exists() else []
    if not cells:
        raise SystemExit(f"no cells under {args.root}")

    lines: list[str] = []
    lines.append("# A4 · 世界模型冻结/更新 × imagination horizon（Isaac 栈，纯 HA 数据）\n")
    lines.append("### 满 horizon 口径（每格自己的 H 步均值，**跨 H 不可比**）\n")
    lines.append("| 格子 | H | 评测点 | 末步 | rollout_reward 首→末 (best) | Δ/min | state_term 首→末 | "
                 "token_mse 末 | hand_mse 末 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cell in cells:
        rows = load_curve(args.root / cell / "eval_curve.jsonl")
        if not rows:
            lines.append(f"| {cell} | 0 | — | — | — | — | — | — |")
            continue
        last = rows[-1]
        d_r = describe(rows, "rollout_reward")
        d_s = describe(rows, "state_term")
        d_t = describe(rows, "token_mse")
        d_h = describe(rows, "hand_mse")
        lines.append(
            f"| {cell} | {fmt(last.get('h'))} | {len(rows)} | {fmt(last.get('step'))} | "
            f"{fmt(d_r['first'])} → {fmt(d_r['last'])} ({fmt(d_r['best'])}) | {fmt(d_r['slope_per_min'])} | "
            f"{fmt(d_s['first'])} → {fmt(d_s['last'])} | {fmt(d_t['last'])} | {fmt(d_h['last'])} |"
        )

    # ---- 固定前缀口径（H 扫描就看这张表）----
    has_prefix = any(any(k in r for r in load_curve(args.root / c / "eval_curve.jsonl"))
                     for c in cells for k in PREFIX_KEYS)
    if has_prefix:
        lines.append("\n### 固定前缀口径（前 8 步，**跨 H 可比**）\n")
        lines.append("| 格子 | H | rollout_reward@8 首→末 (best) | Δ/min | state_term@8 首→末 | survival@8 末 |")
        lines.append("|---|---|---|---|---|---|")
        for cell in cells:
            rows = load_curve(args.root / cell / "eval_curve.jsonl")
            if not rows:
                lines.append(f"| {cell} | — | — | — | — | — |")
                continue
            last = rows[-1]
            d_r = describe(rows, "rollout_reward@8")
            d_s = describe(rows, "state_term@8")
            d_v = describe(rows, "survival@8")
            lines.append(
                f"| {cell} | {fmt(last.get('h'))} | {fmt(d_r['first'])} → {fmt(d_r['last'])} "
                f"({fmt(d_r['best'])}) | {fmt(d_r['slope_per_min'])} | "
                f"{fmt(d_s['first'])} → {fmt(d_s['last'])} | {fmt(d_v['last'])} |"
            )

    # 显存 / 日志里的 WM 更新次数
    memdir = Path("logs/a4")
    lines.append("\n## 训练侧证据（峰值显存 / WM 更新次数）\n")
    lines.append("| 格子 | 峰值显存 MiB | wm/* 更新行数 | 最后一行日志 |")
    lines.append("|---|---|---|---|")
    for cell in cells:
        peak = "—"
        mem = memdir / f"{cell}.mem"
        if mem.exists():
            best = 0.0
            for line in mem.read_text().splitlines():
                parts = line.split()
                for i in range(1, len(parts) - 1, 2):
                    try:
                        best = max(best, float(parts[i + 1]))
                    except ValueError:
                        pass
            peak = f"{best:.0f}" if best else "—"
        wm_updates = 0
        lastline = "—"
        log = memdir / f"{cell}.log"
        if log.exists():
            text = log.read_text(errors="replace").splitlines()
            wm_updates = sum(1 for ln in text if "\"wm/loss\"" in ln or "\"wm/state_loss\"" in ln)
            for ln in reversed(text):
                if ln.startswith("[step") or ln.startswith("[budget"):
                    lastline = ln[:160]
                    break
        lines.append(f"| {cell} | {peak} | {wm_updates} | {lastline} |")

    lines.append("\n## 判读\n")
    lines.append("- `wm_live` 必须真的更新过 WM：上面的 `wm/*` 行数 > 0；为 0 说明 warmup 太长，该格等同冻结。")
    lines.append("- horizon 变大 → 反传链更长、单步更贵（同墙钟下步数更少）；比较 Δ/min 而不是原始末值。")
    lines.append("- 所有格子共享：Isaac ckpt policy/WM、99 维 MDP 观测、Table A.2 奖励、软终止、")
    lines.append("  14-body 可微 FK、纯 HA 参考运动库；只有被消融的那一个变量不同。")
    lines.append("")
    text = "\n".join(lines)
    print(text)
    out = args.out or (args.root / "A4_SUMMARY.md")
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
