#!/usr/bin/env python3
"""Summarise A2 (FM gradient backends) from the on-disk artefacts.

Reads, per arm ``<root>/<backend>/``:

* ``eval_curve.jsonl``  — the fixed-batch eval curve (`step, elapsed_s, token_mse,
  state_term, rollout_reward, hand_mse`);
* ``<log_dir>/a2_<backend>.log`` — training log (backend echo, steps, last line);

plus ``<root>/gpu_mem.csv`` — the nvidia-smi sampler written by ``launch_a2_fm.sh``.

The report deliberately separates *what the run claims* (header: requested backend, pinned
checkpoint, data root) from *what the log proves* (the ``[fm-grad] backend=`` line emitted by
``train_joint.py`` for every non-default backend, the packed-library clip count, the checkpoint
iteration), because A2's whole point is that only ONE variable changes.

Usage:
    python3 scripts/a2_fm_report.py                       # -> docs/ABLATION_A2_RUN.md
    python3 scripts/a2_fm_report.py --root runs/ablations/A2_fm_backend --out /tmp/x.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

KEYS = ("rollout_reward", "state_term", "token_mse", "hand_mse")
BACKENDS = ("pathwise", "checkpointed", "truncated", "adjoint")


def load_curve(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
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


def fmt(v, nd: int = 4) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return "—"
    return f"{f:.{nd}g}"


def describe(rows: list[dict], key: str) -> dict:
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and v == v:
            xs.append(float(r.get("elapsed_s", 0.0) or 0.0))
            ys.append(float(v))
    if not ys:
        return {"n": 0, "first": None, "last": None, "best": None, "slope_per_min": None}
    return {
        "n": len(ys),
        "first": ys[0],
        "last": ys[-1],
        "best": max(ys) if key in ("rollout_reward", "token_mse") else min(ys),
        "slope_per_min": slope(xs, ys) * 60.0,
    }


def peak_mem(csv_path: Path) -> dict[int, float]:
    peaks: dict[int, float] = {}
    if not csv_path.exists():
        return peaks
    with csv_path.open(newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 3:
                continue
            try:
                idx = int(row[1].strip())
                # "46084 MiB" or "46084"
                mem = float(row[2].strip().split()[0])
            except (ValueError, IndexError):
                continue
            peaks[idx] = max(peaks.get(idx, 0.0), mem)
    return peaks


def log_facts(log_path: Path, backend: str) -> dict:
    """Pull the proof lines out of one arm's training log."""
    facts = {
        "exists": log_path.exists(),
        "backend_echo": [],
        "clips": None,
        "frames": None,
        "iteration": None,
        "obs_mode": None,
        "hand_labels": None,
        "steps_seen": 0,
        "budget_line": None,
        "last_line": None,
        "rc_line": None,
        "requested": [],
        "errors": 0,
    }
    if not log_path.exists():
        return facts
    clips_re = re.compile(r"clips['\"]?\s*:\s*(\d+)")
    frames_re = re.compile(r"frames['\"]?\s*:\s*(\d+)")
    for line in log_path.read_text(errors="replace").splitlines():
        if "[fm-grad]" in line:
            facts["backend_echo"].append(line.strip())
        if "[gate] Isaac WBC ok" in line:
            m = re.search(r"iteration=(\d+)", line)
            if m:
                facts["iteration"] = int(m.group(1))
            m = re.search(r"obs_mode=([\w\-]+)", line)
            if m:
                facts["obs_mode"] = m.group(1)
        if "packed library" in line:
            m = clips_re.search(line)
            if m:
                facts["clips"] = int(m.group(1))
            m = frames_re.search(line)
            if m:
                facts["frames"] = int(m.group(1))
        if "[hand] 辅助监督开启" in line:
            facts["hand_labels"] = line.strip()
        if line.startswith("[step "):
            facts["steps_seen"] += 1
        if line.startswith("[budget"):
            facts["budget_line"] = line.strip()
        if line.startswith("[arm a2_") and "rc=" in line:
            facts["rc_line"] = line.strip()
        if line.startswith("[arm a2_") and "requested" in line:
            facts["requested"].append(line.strip())
        if "Error" in line or "Traceback" in line or "CUDA out of memory" in line:
            facts["errors"] += 1
        if line.strip():
            facts["last_line"] = line.strip()
    facts["backend_echo"] = sorted(set(facts["backend_echo"]))
    return facts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("runs/ablations/A2_fm_backend"))
    ap.add_argument("--log-dir", type=Path, default=Path("logs"))
    ap.add_argument("--master-log", type=Path, default=Path("logs/a2_master.log"),
                    help="launcher transcript; carries the one-shot [gate] Isaac WBC line "
                         "that every arm shares (it is printed before the arms start)")
    ap.add_argument("--out", type=Path, default=Path("docs/ABLATION_A2_RUN.md"))
    args = ap.parse_args()

    arms = [b for b in BACKENDS if (args.root / b).is_dir()]
    if not arms:
        arms = sorted(p.name for p in args.root.iterdir() if p.is_dir()) if args.root.exists() else []
    if not arms:
        raise SystemExit(f"no arms under {args.root}")

    peaks = peak_mem(args.root / "gpu_mem.csv")
    facts = {b: log_facts(args.log_dir / f"a2_{b}.log", b) for b in arms}

    # [gate] 行只打一次（launcher 起臂之前），它描述的是**整个序列共享的底模**，
    # 所以按 master log 判定，再套到每一臂上；各臂日志里没有这行属正常。
    gate = {"iteration": None, "obs_mode": None, "interface": None, "line": None}
    if args.master_log.exists():
        for line in args.master_log.read_text(errors="replace").splitlines():
            if "[gate] Isaac WBC ok" in line:
                m = re.search(r"iteration=(\d+)", line)
                if m:
                    gate["iteration"] = int(m.group(1))
                m = re.search(r"obs_mode=([\w\-]+)", line)
                if m:
                    gate["obs_mode"] = m.group(1)
                m = re.search(r"interface=([\w\-]+)", line)
                if m:
                    gate["interface"] = m.group(1)
                gate["line"] = line.strip()
                break
    for b in arms:
        if facts[b]["iteration"] is None:
            facts[b]["iteration"] = gate["iteration"]
            facts[b]["obs_mode"] = gate["obs_mode"]

    lines: list[str] = []
    lines.append("# A2 运行报告：FM 反传方法（pathwise / checkpointed / truncated / adjoint）\n")
    lines.append("> 自动生成：`python3 scripts/a2_fm_report.py --root runs/ablations/A2_fm_backend`\n")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| 卡 | 本容器（wscoli2）2× RTX 4090 48 GB，`CUDA_VISIBLE_DEVICES=0,1`，2 卡 DDP |")
    lines.append("| 数据 | `data/npz/ha_tokens`（纯 HA；mixed / ft318 被入口硬拒） |")
    lines.append("| 打包运动库 | `runs/isaac_motion/ha_tokens/packed.pt`（trackerlab CompactMotionLib） |")
    lines.append("| SuperMimic（冻结底模） | `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt` |")
    lines.append("| VLA 头 | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（VLM backbone 冻结，不设 `--train-vlm`） |")
    lines.append("| 共享超参 | `--batch-size 2 --horizon 8 --hand-loss-weight 0.2 --hand-grasp-boost 5`，"
                 "同墙钟 `--time-budget-minutes 30` |")
    lines.append("| 唯一变量 | `--fm-grad-backend` |")
    lines.append("")

    lines.append("## 1. 曲线（同墙钟比较）\n")
    lines.append("| 臂 | 评测点 | 末步 | 墙钟 s | rollout_reward 首→末 (best) | Δ/min | "
                 "state_term 首→末 | token_mse 首→末 | hand_mse 末 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for b in arms:
        rows = load_curve(args.root / b / "eval_curve.jsonl")
        if not rows:
            lines.append(f"| {b} | 0 | — | — | — | — | — | — | — |")
            continue
        last = rows[-1]
        d = {k: describe(rows, k) for k in KEYS}
        lines.append(
            f"| {b} | {len(rows)} | {fmt(last.get('step'), 6)} | {fmt(last.get('elapsed_s'), 6)} | "
            f"{fmt(d['rollout_reward']['first'])} → {fmt(d['rollout_reward']['last'])} "
            f"({fmt(d['rollout_reward']['best'])}) | {fmt(d['rollout_reward']['slope_per_min'])} | "
            f"{fmt(d['state_term']['first'])} → {fmt(d['state_term']['last'])} | "
            f"{fmt(d['token_mse']['first'])} → {fmt(d['token_mse']['last'])} | "
            f"{fmt(d['hand_mse']['last'])} |"
        )
    lines.append("")

    lines.append("## 2. 训练侧证据（每臂真的按指定后端在跑）\n")
    lines.append("| 臂 | backend 回声 | 日志步数 | 墙钟退出 | 错误行 | rc |")
    lines.append("|---|---|---|---|---|---|")
    for b in arms:
        f = facts[b]
        echo = "；".join(f["backend_echo"]) if f["backend_echo"] else (
            "（pathwise 是默认值，代码不打印回声；见 launcher 的 requested 行）" if b == "pathwise" else "**缺失**"
        )
        lines.append(
            f"| {b} | {echo} | {f['steps_seen']} | {f['budget_line'] or '—'} | "
            f"{f['errors']} | {f['rc_line'] or '—'} |"
        )
    peak_line = ", ".join(f"gpu{i}={v:.0f} MiB" for i, v in sorted(peaks.items())) or "—"
    lines.append("")
    lines.append(f"显存采样（`gpu_mem.csv`，整轮 A2 的最高值，48 GB 卡）：**{peak_line}**。"
                 "每臂起点/终点见 `logs/a2.launcher.log`，采样时间列可切成对应窗口。")
    lines.append("")

    lines.append("## 3. 口径自检（每条都必须成立）\n")
    if gate["line"]:
        lines.append(f"共享底模门禁（来自 `{args.master_log}`，一次、对所有臂生效）：`{gate['line']}`\n")
    checks: list[tuple[str, bool, str]] = []
    for b in arms:
        f = facts[b]
        req_ok = any(f"--fm-grad-backend={b}" in ln for ln in f["requested"])
        checks.append((f"{b}: launcher 请求的后端 = {b}", req_ok,
                       "；".join(f["requested"]) or "未找到 requested 行"))
        if b != "pathwise":
            ok = any(f"backend={b}" in ln for ln in f["backend_echo"])
            checks.append((f"{b}: 日志回显 `backend={b}`", ok, "；".join(f["backend_echo"]) or "无回声"))
        else:
            others = [ln for ln in f["backend_echo"] if "backend=pathwise" not in ln]
            checks.append((f"{b}: 日志没有其他后端回声（pathwise 是默认值，不打印）", not others,
                           "；".join(f["backend_echo"]) or "无回声（符合预期）"))
    for b in arms:
        f = facts[b]
        checks.append((f"{b}: 数据 = 纯 HA 699 clips", f["clips"] == 699, f"clips={f['clips']}, frames={f['frames']}"))
        checks.append((f"{b}: 底模 = Isaac ckpt iteration 10500", f["iteration"] == 10500,
                       (gate["line"] if gate["line"] and gate["iteration"] == 10500
                        else f"iteration={f['iteration']}, obs_mode={f['obs_mode']}")))
        hl = f["hand_labels"] or ""
        checks.append((f"{b}: 手部辅助监督用 HA 标签", "ha_tokens" in hl and "ft318" not in hl, hl or "未开启"))
        checks.append((f"{b}: 无 OOM / traceback", f["errors"] == 0, f"错误行={f['errors']}"))
    lines.append("| 检查 | 结果 | 证据 |")
    lines.append("|---|---|---|")
    for name, ok, ev in checks:
        lines.append(f"| {name} | {'✅' if ok else '❌'} | {ev} |")
    lines.append("")

    lines.append("## 4. 判读\n")
    lines.append("- 主指标：`rollout_reward` 的同墙钟斜率（Δ/min）与终值；`token_mse` 只衡量 Euler 链输出、"
                 "不衡量梯度，不能单独当结论。")
    lines.append("- `truncated` 预期更快（步数更多）但更差（有偏）；`adjoint` 与 `pathwise` 应几乎重合，"
                 "若明显偏离 → 实现 bug，先查 `diff_vla_wbc/fm_grad.py` 而不是下结论。")
    lines.append("- `checkpointed` 与 `pathwise` 数学等价，差异应只在吞吐（步数）与显存。")
    lines.append("- `qvgm_lite` 需要 chunk critic，真实栈**不提供**（会显式报错），只在 `ablation_fm_grad/` 的 toy 里保留，"
                 "不得混进本表。")
    lines.append("- VLM backbone 两臂都冻结：`adjoint` 会 detach backbone 特征，所以本消融不叠加 `--train-vlm`。")
    lines.append("")

    text = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
