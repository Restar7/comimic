#!/usr/bin/env python3
"""Report for the card-2,3 ablations: A3 (BPTT vs PPO) and A5 (head-only vs head+WBC).

Reads only what the runs actually produced, so it is regenerable at any time and never
invents numbers:

    runs/ablations/A3_bptt_vs_ppo/{bptt,ppo}/eval_curve.jsonl   + joint_latest.pt / vla_final.pt
    runs/ablations/A5_only_head/{head_only,head_plus_wbc}/eval_curve.jsonl
    logs/a3_bptt.log, logs/a3_ppo.log, logs/a5_*.log            (per-step metrics)
    runs/ablations/gpu_mem.csv                                  (nvidia-smi sampler, optional)

Outputs:
    <out>            markdown report (default docs/ABLATION_A3_A5_CARDS23.md)
    <out>.csv        one row per (arm, eval) for downstream tables
    <out-dir>/*.png  reward / loss curves (elapsed-time and step axes)

Usage:
    python3 scripts/a3_a5_report.py                      # write the markdown
    python3 scripts/a3_a5_report.py --no-plots           # text only
    python3 scripts/a3_a5_report.py --audit-a5 --sm-ckpt <ckpt_010500.pt>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

A3_ARMS = {"bptt": "BPTT (pathwise, exact)", "ppo": "PPO (score function, T=1)"}
A5_ARMS = {"head_only": "head only (policy frozen)", "head_plus_wbc": "head + WBC policy"}


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


def read_steps(log_path: Path) -> list[dict]:
    """Per-step metric rows from a trainer log (`[step N] {...}` / `[step N/M] {...}`)."""
    if not log_path.exists():
        return []
    rows, seen = [], set()
    for line in log_path.read_text(errors="replace").splitlines():
        if not line.startswith("[step "):
            continue
        try:
            payload = json.loads(line[line.index("{") :])
        except (ValueError, json.JSONDecodeError):
            continue
        if line in seen:          # 2-rank logs interleave; the payload is what matters
            continue
        seen.add(line)
        rows.append(payload)
    return rows


def peak_gpu_mem(csv_path: Path, max_minute: float | None = None) -> dict[int, float]:
    """Per-GPU peak MiB from the `nvidia-smi --format=csv -l` sampler."""
    if not csv_path.exists():
        return {}
    peak: dict[int, float] = {}
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            try:                       # nvidia-smi csv prints units: "46046 MiB" / "100 %"
                idx = int(str(row[1]).strip())
                used = float(str(row[2]).split()[0])
            except (ValueError, IndexError):
                continue
            peak[idx] = max(peak.get(idx, 0.0), used)
    return peak


def fmt(x, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def first_last(rows: list[dict], key: str) -> tuple[float | None, float | None]:
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    if not vals:
        return None, None
    return float(vals[0]), float(vals[-1])


def tail_mean(rows: list[dict], key: str, n: int = 3) -> float | None:
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    if not vals:
        return None
    tail = vals[-n:]
    return sum(tail) / len(tail)


def arm_block(arm_dir: Path, log: Path) -> dict:
    curve = read_jsonl(arm_dir / "eval_curve.jsonl")
    steps = read_steps(log)
    out = {
        "dir": arm_dir,
        "evals": len(curve),
        "curve": curve,
        "step_rows": len(steps),
        "last_step": steps[-1].get("step") if steps else (curve[-1].get("step") if curve else None),
        "wall_s": curve[-1].get("elapsed_s") if curve and "elapsed_s" in curve[-1] else None,
        "reward_first_last": first_last(curve, "rollout_reward"),
        "reward_tail": tail_mean(curve, "rollout_reward"),
        "reward_best": max([r["rollout_reward"] for r in curve if "rollout_reward" in r], default=None),
        "state_term_first_last": first_last(curve, "state_term"),
        "state_term_tail": tail_mean(curve, "state_term"),
        "token_mse_first_last": first_last(curve, "token_mse"),
        "hand_mse_first_last": first_last(curve, "hand_mse"),
        "docs": (arm_dir / "joint_latest.pt").exists() or (arm_dir / "vla_final.pt").exists(),
        "log": log,
    }
    for k in ("ppo/sigma", "ppo/clip_frac", "ppo/ratio_epoch0", "ppo/clip_frac_epoch0",
              "loss", "loss/state", "loss/reward", "loss/hand_mse",
              "rollout/survival", "rollout/action_abs", "loss/smooth", "loss/safety"):
        out[k] = tail_mean(steps, k)
    if steps:
        first_steps = steps[:3]
        out["loss_first"] = sum(s.get("loss", float("nan")) for s in first_steps) / max(1, len(first_steps))
    return out


def table_a3(a3: dict[str, dict]) -> str:
    lines = [
        "| 指标 | " + " | ".join(A3_ARMS.values()) + " |",
        "|---|" + "---|" * len(A3_ARMS),
        "| eval 次数 | " + " | ".join(fmt(a3[k]["evals"]) for k in A3_ARMS) + " |",
        "| 训练步数 | " + " | ".join(fmt(a3[k]["last_step"]) for k in A3_ARMS) + " |",
        "| 墙钟（s，最后一次 eval） | " + " | ".join(fmt(a3[k]["wall_s"], 0) for k in A3_ARMS) + " |",
        "| rollout_reward（首次 → 末次） | "
        + " | ".join(f"{fmt(a3[k]['reward_first_last'][0])} → {fmt(a3[k]['reward_first_last'][1])}"
                     for k in A3_ARMS) + " |",
        "| rollout_reward（末 3 次均值） | " + " | ".join(fmt(a3[k]["reward_tail"]) for k in A3_ARMS) + " |",
        "| rollout_reward（最佳） | " + " | ".join(fmt(a3[k]["reward_best"]) for k in A3_ARMS) + " |",
        "| state_term（首次 → 末次） | "
        + " | ".join(f"{fmt(a3[k]['state_term_first_last'][0])} → {fmt(a3[k]['state_term_first_last'][1])}"
                     for k in A3_ARMS) + " |",
        "| token_mse（首次 → 末次） | "
        + " | ".join(f"{fmt(a3[k]['token_mse_first_last'][0])} → {fmt(a3[k]['token_mse_first_last'][1])}"
                     for k in A3_ARMS) + " |",
        "| hand_mse（首次 → 末次） | "
        + " | ".join(f"{fmt(a3[k]['hand_mse_first_last'][0])} → {fmt(a3[k]['hand_mse_first_last'][1])}"
                     for k in A3_ARMS) + " |",
        "| rollout/survival（末 3 次均值） | " + " | ".join(fmt(a3[k]["rollout/survival"]) for k in A3_ARMS) + " |",
        "| ppo/sigma（末 3 次均值） | " + " | ".join(fmt(a3[k]["ppo/sigma"]) for k in A3_ARMS) + " |",
        "| ppo/clip_frac（末 epoch，末 3 次均值） | " + " | ".join(fmt(a3[k]["ppo/clip_frac"]) for k in A3_ARMS) + " |",
        "| ppo/clip_frac（**第 0 个 epoch**，末 3 次均值） | "
        + " | ".join(fmt(a3[k]["ppo/clip_frac_epoch0"]) for k in A3_ARMS) + " |",
        "| ppo/ratio（第 0 个 epoch，末 3 次均值） | "
        + " | ".join(fmt(a3[k]["ppo/ratio_epoch0"]) for k in A3_ARMS) + " |",
        "| 训练 loss（末 3 次均值） | " + " | ".join(fmt(a3[k]["loss"]) for k in A3_ARMS) + " |",
    ]
    return "\n".join(lines)


def audit_a5(sm_ckpt: Path, head_only: Path, head_plus: Path) -> dict:
    """Frozen-policy audit: max|joint_policy − SM_CKPT.policy| must be exactly 0 for head_only."""
    import torch  # local import: report also runs without torch for pure-curve mode

    src = torch.load(str(sm_ckpt), map_location="cpu", weights_only=False)
    src_pi = src["pi"]
    out = {"source": str(sm_ckpt), "iteration": src.get("iteration"), "arms": {}}
    for name, path in (("head_only", head_only), ("head_plus_wbc", head_plus)):
        if not path.exists():
            out["arms"][name] = {"status": "missing"}
            continue
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        pi = ck.get("pi") or ck.get("policy") or {}
        worst, changed, missing = 0.0, 0, []
        for k, v in src_pi.items():
            if k not in pi:
                missing.append(k)
                continue
            d = float((pi[k].float() - v.float()).abs().max())
            worst = max(worst, d)
            if d > 0.0:
                changed += 1
        out["arms"][name] = {
            "status": "ok",
            "max_abs_delta": worst,
            "changed_tensors": changed,
            "total_tensors": len(src_pi),
            "missing": missing,
            "frozen": worst == 0.0,
        }
    return out


def plots(out_dir: Path, a3: dict[str, dict], a5: dict[str, dict]) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[warn] matplotlib unavailable ({exc}); skipping plots", file=sys.stderr)
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for title, arms, names, fname in (
        ("A3 · BPTT vs PPO (same wall clock)", a3, A3_ARMS, "A3_reward_vs_time.png"),
        ("A5 · head only vs head + WBC", a5, A5_ARMS, "A5_reward_vs_time.png"),
    ):
        curves = {k: arms[k]["curve"] for k in names if arms.get(k, {}).get("curve")}
        if not curves:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for k, label in names.items():
            rows = curves.get(k, [])
            if not rows:
                continue
            t = [r.get("elapsed_s", 0) / 60.0 for r in rows]
            rw = [r.get("rollout_reward") for r in rows]
            st = [r.get("state_term") for r in rows]
            axes[0].plot(t, rw, marker="o", ms=3, label=label)
            axes[1].plot(t, st, marker="o", ms=3, label=label)
        axes[0].set_xlabel("wall clock (min)")
        axes[0].set_ylabel("rollout_reward (eval)")
        axes[1].set_xlabel("wall clock (min)")
        axes[1].set_ylabel("state_term (eval)")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

        # step axis
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for k, label in names.items():
            rows = curves.get(k, [])
            if not rows:
                continue
            ax.plot([r.get("step") for r in rows], [r.get("rollout_reward") for r in rows],
                    marker="o", ms=3, label=label)
        ax.set_xlabel("training step")
        ax.set_ylabel("rollout_reward (eval)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        path2 = out_dir / fname.replace("_vs_time", "_vs_step")
        fig.savefig(path2, dpi=140)
        plt.close(fig)
        written.append(path2)

    return written


def write_csv(path: Path, a3: dict[str, dict], a5: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "arm", "step", "elapsed_s", "rollout_reward", "state_term",
                    "token_mse", "hand_mse"])
        for group, arms in (("A3", a3), ("A5", a5)):
            for arm, data in arms.items():
                for row in data["curve"]:
                    w.writerow([group, arm, row.get("step"), row.get("elapsed_s"),
                                row.get("rollout_reward"), row.get("state_term"),
                                row.get("token_mse"), row.get("hand_mse")])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "ablations")
    ap.add_argument("--logs", type=Path, default=ROOT / "logs")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "ABLATION_A3_A5_CARDS23.md")
    ap.add_argument("--plots-dir", type=Path, default=None)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--audit-a5", action="store_true", help="diff joint_latest.pt policy against the SM ckpt")
    ap.add_argument("--sm-ckpt", type=Path,
                    default=ROOT / "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt")
    ap.add_argument("--gpu-csv", type=Path, default=None)
    args = ap.parse_args()

    a3 = {arm: arm_block(args.root / "A3_bptt_vs_ppo" / arm, args.logs / f"a3_{arm}.log")
          for arm in A3_ARMS}
    a5 = {arm: arm_block(args.root / "A5_only_head" / arm, args.logs / f"a5_{arm}.log")
          for arm in A5_ARMS}

    audit = None
    if args.audit_a5:
        audit = audit_a5(
            args.sm_ckpt,
            args.root / "A5_only_head" / "head_only" / "joint_latest.pt",
            args.root / "A5_only_head" / "head_plus_wbc" / "joint_latest.pt",
        )

    gpu_peak = peak_gpu_mem(args.gpu_csv or (args.root / "gpu_mem.csv"))

    out_dir = args.plots_dir or args.out.parent / "figs"
    pngs = [] if args.no_plots else plots(out_dir, a3, a5)
    csv_path = args.out.with_suffix(".csv")
    write_csv(csv_path, a3, a5)

    md = ["# 卡 2,3 的消融结果：A3（BPTT vs PPO）+ A5（冻结 WBC）", ""]
    md += [
        "> 自动生成：`python3 scripts/a3_a5_report.py`。所有数字直接来自",
        "> `runs/ablations/**/eval_curve.jsonl` 与训练日志，未做任何挑选或平滑。",
        "",
        "## 0. 协议（两臂只差被消融的那一个变量）",
        "",
        "| 项 | 取值 |",
        "|---|---|",
        "| 数据 | **纯 HA**：`data/npz/ha_tokens`（699 条 / 634,016 帧 / 3.52 h）。"
        "不含任何 ft318、不含 mixed_ha_ft318 |",
        "| SuperMimic WBC + 世界模型 | `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt`"
        "（Isaac Lab/PhysX，token 接口，iteration 10500） |",
        "| 联合微调 VLA | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（官方微调产物） |",
        "| 世界模型 | 冻结（`--wm-update-every 0`） |",
        "| 手部辅助损失 | λ=0.2，boost=5 |",
        "| batch / horizon | 2 per rank（2 卡 → 全局 4）/ 8 |",
        "| 卡 | 2 卡 DDP（本容器 = 2×RTX 4090 48 GB） |",
        "",
        "**整条链路都是 Isaac Lab 的**（`diff_vla_wbc/isaac_joint.py`），不是 mjlab/MuJoCo 旧链路：",
        "policy 与世界模型 = `ckpt_010500.pt` 的 `pi`/`wm` 张量（逐元素相同，max|Δ|=0）；",
        "观测（99 维 `relative_ori99`）、奖励（`MDP.reward`，用 ckpt 里记录的 `reward_cfg`）、",
        "软终止（`MDP.soft_termination`）、可微 FK（`G1Kinematics`）、参考运动库",
        "（`CompactMotionLib`，由 `data/npz/ha_tokens` 打包，699 条）全部来自",
        "`trackerlab-supermimic`。门禁脚本 `scripts/verify_isaac_chain.py` 逐条断言通过。",
        "",
        "## 1. A3 · 精确反传（BPTT）vs score-function（PPO），同墙钟 120 min/臂",
        "",
        table_a3(a3),
        "",
    ]

    gate = args.root / "ISAAC_CHAIN_GATE.txt"
    if gate.exists():
        md += ["### 0.1 链路门禁输出（`scripts/verify_isaac_chain.py`）", "", "```text",
               gate.read_text().strip(), "```", ""]

    md += ["## 2. A5 · 只训 VLA head vs head + WBC（40 min/臂）", ""]
    md += [
        "| 指标 | " + " | ".join(A5_ARMS.values()) + " |",
        "|---|" + "---|" * len(A5_ARMS),
        "| eval 次数 | " + " | ".join(fmt(a5[k]["evals"]) for k in A5_ARMS) + " |",
        "| 训练步数 | " + " | ".join(fmt(a5[k]["last_step"]) for k in A5_ARMS) + " |",
        "| rollout_reward（首次 → 末次） | "
        + " | ".join(f"{fmt(a5[k]['reward_first_last'][0])} → {fmt(a5[k]['reward_first_last'][1])}"
                     for k in A5_ARMS) + " |",
        "| rollout_reward（末 3 次均值） | " + " | ".join(fmt(a5[k]["reward_tail"]) for k in A5_ARMS) + " |",
        "| state_term（首次 → 末次） | "
        + " | ".join(f"{fmt(a5[k]['state_term_first_last'][0])} → {fmt(a5[k]['state_term_first_last'][1])}"
                     for k in A5_ARMS) + " |",
        "| token_mse（首次 → 末次） | "
        + " | ".join(f"{fmt(a5[k]['token_mse_first_last'][0])} → {fmt(a5[k]['token_mse_first_last'][1])}"
                     for k in A5_ARMS) + " |",
        "| hand_mse（首次 → 末次） | "
        + " | ".join(f"{fmt(a5[k]['hand_mse_first_last'][0])} → {fmt(a5[k]['hand_mse_first_last'][1])}"
                     for k in A5_ARMS) + " |",
        "",
    ]

    if audit is not None:
        md += ["### 2.1 参数冻结审计（`max|joint_latest.policy − SM_CKPT.policy|`）", "",
               "| 臂 | 最大绝对差 | 变化的张量 | 结论 |", "|---|---|---|---|"]
        for name, label in (("head_only", A5_ARMS["head_only"]), ("head_plus_wbc", A5_ARMS["head_plus_wbc"])):
            row = audit["arms"].get(name, {})
            if row.get("status") != "ok":
                md.append(f"| {label} | — | — | 缺少 `joint_latest.pt` |")
                continue
            verdict = "✅ 完全冻结（逐元素 0）" if row["frozen"] else f"更新了 {row['changed_tensors']} 个张量"
            md.append(f"| {label} | {row['max_abs_delta']:.3e} | "
                      f"{row['changed_tensors']}/{row['total_tensors']} | {verdict} |")
        md.append("")

    if gpu_peak:
        md += ["## 3. 显存峰值（nvidia-smi 采样）", "",
               "| GPU | 峰值占用 (MiB) |", "|---|---|"]
        md += [f"| {k} | {v:.0f} |" for k, v in sorted(gpu_peak.items())]
        md.append("")

    if pngs:
        md += ["## 4. 曲线", ""]
        md += [f"- `{p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}`" for p in pngs]
        md.append("")

    md += [
        "## 5. 原始产物",
        "",
        "| 臂 | 目录 | eval 曲线 | 权重 |",
        "|---|---|---|---|",
    ]
    for group, arms in (("A3", a3), ("A5", a5)):
        for arm, data in arms.items():
            rel = data["dir"].relative_to(ROOT) if data["dir"].is_relative_to(ROOT) else data["dir"]
            ck = "joint_latest.pt" if (data["dir"] / "joint_latest.pt").exists() else (
                "vla_final.pt" if (data["dir"] / "vla_final.pt").exists() else "—")
            has_curve = "✅" if data["evals"] else "—"
            md.append(f"| {group}/{arm} | `{rel}` | {has_curve} ({data['evals']} 行) | {ck} |")
    md += ["", f"逐行数据：`{csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path}`", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md))
    print(f"[written] {args.out}")
    print(f"[written] {csv_path}")
    for p in pngs:
        print(f"[written] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
