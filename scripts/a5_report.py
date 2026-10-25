#!/usr/bin/env python3
"""A5 汇总 + 冻结审计：head_only vs head_plus_wbc。

两件事，缺一不可：

1. **曲线**（同墙钟 40 min/臂）：`<root>/{head_only,head_plus_wbc}/eval_curve.jsonl`；
2. **参数审计**（这是 A5 结论成立的前提）：把每臂 `joint_latest.pt` 的 `pi` / `wm` 与源底模
   `ckpt_010500.pt` 逐张量比对 —— `head_only` 必须 `max|Δ| == 0`（policy 真的没动），
   `head_plus_wbc` 必须 `> 0`（policy 真的在更新），两臂的 `wm` 都应 `== 0`（WM 冻结口径）。

用法：
    python3 scripts/a5_report.py                       # -> docs/ABLATION_A5_RUN.md
    python3 scripts/a5_report.py --root runs/ablations/A5_only_head --out /tmp/a5.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

KEYS = ("rollout_reward", "state_term", "token_mse", "hand_mse")
ARMS = ("head_only", "head_plus_wbc")


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
    return {"n": len(ys), "first": ys[0], "last": ys[-1],
            "best": max(ys) if key in ("rollout_reward", "token_mse") else min(ys),
            "slope_per_min": slope(xs, ys) * 60.0}


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
                mem = float(row[2].strip().split()[0])
            except (ValueError, IndexError):
                continue
            peaks[idx] = max(peaks.get(idx, 0.0), mem)
    return peaks


def param_audit(ckpt: Path, arm_ckpt: Path) -> dict:
    """逐张量比较 arm ckpt 与源 ckpt 的 pi / wm。"""
    out = {"available": False, "error": None, "pi": None, "wm": None, "steps": None,
           "has_adapter": None}
    if not (ckpt.exists() and arm_ckpt.exists()):
        out["error"] = "missing ckpt"
        return out
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"torch unavailable: {exc}"
        return out
    try:
        base = torch.load(ckpt, map_location="cpu", weights_only=False)
        arm = torch.load(arm_ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"load failed: {exc}"
        return out
    base_pi = base.get("pi") or base.get("policy") or {}
    base_wm = base.get("wm") or {}
    arm_pi = arm.get("pi") or arm.get("policy") or {}
    arm_wm = arm.get("wm") or {}
    out["has_adapter"] = "adapter" in arm
    out["steps"] = arm.get("steps")

    def diff(a: dict, b: dict) -> dict:
        keys = sorted(set(a) | set(b))
        max_abs = 0.0
        changed = 0
        missing: list[str] = []
        for k in keys:
            if k not in a or k not in b:
                missing.append(k)
                continue
            try:
                d = (a[k].to(torch.float64) - b[k].to(torch.float64)).abs().max().item()
            except Exception:  # noqa: BLE001
                missing.append(k)
                continue
            if d > 0:
                changed += 1
            max_abs = max(max_abs, d)
        return {"tensors": len(keys) - len(missing), "changed": changed,
                "max_abs": max_abs, "missing": missing}

    out["pi"] = diff(base_pi, arm_pi)
    out["wm"] = diff(base_wm, arm_wm)
    out["available"] = True
    return out


def log_facts(log_path: Path) -> dict:
    facts = {"exists": log_path.exists(), "requested": [], "budget_line": None,
             "errors": 0, "steps_seen": 0, "last_line": None, "rc_line": None}
    if not log_path.exists():
        return facts
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("[arm a5_") and "requested" in line:
            facts["requested"].append(line.strip())
        if line.startswith("[budget"):
            facts["budget_line"] = line.strip()
        if line.startswith("[step "):
            facts["steps_seen"] += 1
        if line.startswith("[arm a5_") and "rc=" in line:
            facts["rc_line"] = line.strip()
        if "CUDA out of memory" in line or "Traceback" in line:
            facts["errors"] += 1
        if line.strip():
            facts["last_line"] = line.strip()
    return facts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("runs/ablations/A5_only_head"))
    ap.add_argument("--log-dir", type=Path, default=Path("logs"))
    ap.add_argument("--sm-ckpt", type=Path,
                    default=Path("trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt"))
    ap.add_argument("--out", type=Path, default=Path("docs/ABLATION_A5_RUN.md"))
    args = ap.parse_args()

    arms = [a for a in ARMS if (args.root / a).is_dir()] or (
        sorted(p.name for p in args.root.iterdir() if p.is_dir()) if args.root.exists() else [])
    if not arms:
        raise SystemExit(f"no arms under {args.root}")

    facts = {a: log_facts(args.log_dir / f"a5_{a}.log") for a in arms}
    audits = {a: param_audit(args.sm_ckpt, args.root / a / "joint_latest.pt") for a in arms}
    peaks = peak_mem(args.root / "gpu_mem.csv")

    L: list[str] = []
    L.append("# A5 运行报告：冻结 SuperMimic，只训 VLA head（Isaac 栈，纯 HA）\n")
    L.append("> 自动生成：`python3 scripts/a5_report.py --root runs/ablations/A5_only_head`\n")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 卡 | 本容器（wscoli2）2× RTX 4090 48 GB，`CUDA_VISIBLE_DEVICES=0,1`，2 卡 DDP |")
    L.append("| 数据 | `data/npz/ha_tokens`（纯 HA；mixed / ft318 被入口硬拒） |")
    L.append("| 底模（WBC + WM） | `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt`（iteration 10500） |")
    L.append("| VLA | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`，VLM backbone 冻结 |")
    L.append("| 共享超参 | `--batch-size 2 --horizon 8 --hand-loss-weight 0.2 --hand-grasp-boost 5`，同墙钟 40 min/臂 |")
    L.append("| 唯一变量 | `--policy-lr/--adapter-lr`：head_only = 0/0；head_plus_wbc = 默认 1e-5 |")
    L.append("| WM | 冻结（`--wm-update-every 0`，两臂相同） |")
    L.append("")

    L.append("## 1. 曲线（同墙钟）\n")
    L.append("| 臂 | 评测点 | 末步 | 墙钟 s | rollout_reward 首→末 (best) | Δ/min | "
             "state_term 首→末 | token_mse 首→末 | hand_mse 末 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        rows = load_curve(args.root / a / "eval_curve.jsonl")
        if not rows:
            L.append(f"| {a} | 0 | — | — | — | — | — | — | — |")
            continue
        last = rows[-1]
        d = {k: describe(rows, k) for k in KEYS}
        L.append(
            f"| {a} | {len(rows)} | {fmt(last.get('step'), 6)} | {fmt(last.get('elapsed_s'), 6)} | "
            f"{fmt(d['rollout_reward']['first'])} → {fmt(d['rollout_reward']['last'])} "
            f"({fmt(d['rollout_reward']['best'])}) | {fmt(d['rollout_reward']['slope_per_min'])} | "
            f"{fmt(d['state_term']['first'])} → {fmt(d['state_term']['last'])} | "
            f"{fmt(d['token_mse']['first'])} → {fmt(d['token_mse']['last'])} | "
            f"{fmt(d['hand_mse']['last'])} |")
    L.append("")

    L.append("## 2. 冻结审计（A5 结论成立的前提）\n")
    L.append(f"源底模：`{args.sm_ckpt}`；每臂 `joint_latest.pt` 的 `pi`/`wm` 与之逐张量比较。\n")
    L.append("| 臂 | pi 张量 | pi 变化张量数 | pi max\\|Δ\\| | wm 变化张量数 | wm max\\|Δ\\| | steps | 判定 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for a in arms:
        au = audits[a]
        if not au["available"]:
            L.append(f"| {a} | — | — | — | — | — | — | ⚠️ {au['error']} |")
            continue
        pi, wm = au["pi"], au["wm"]
        if a == "head_only":
            verdict = "✅ policy 未动" if pi["max_abs"] == 0 else "❌ policy 竟然变了"
        else:
            verdict = "✅ policy 在更新" if pi["max_abs"] > 0 else "❌ policy 没动（臂名/参数错）"
        if wm["max_abs"] != 0:
            verdict += "；⚠️ WM 也变了（应冻结）"
        L.append(f"| {a} | {pi['tensors']} | {pi['changed']} | {pi['max_abs']:.3e} | "
                 f"{wm['changed']} | {wm['max_abs']:.3e} | {au['steps']} | {verdict} |")
    L.append("")

    L.append("## 3. 训练侧证据\n")
    L.append("| 臂 | 日志步数 | 墙钟退出 | 错误行 | rc |")
    L.append("|---|---|---|---|---|")
    for a in arms:
        f = facts[a]
        L.append(f"| {a} | {f['steps_seen']} | {f['budget_line'] or '—'} | {f['errors']} | {f['rc_line'] or '—'} |")
    peak_line = ", ".join(f"gpu{i}={v:.0f} MiB" for i, v in sorted(peaks.items())) or "—"
    L.append("")
    L.append(f"显存采样（整轮最高，48 GB 卡）：**{peak_line}**。")
    L.append("")

    L.append("## 4. 判读\n")
    L.append("- 若 head_only 已经能到 head_plus_wbc 同量级的 `rollout_reward`，说明收益主要来自 VLA head，"
             "可以省掉 WBC 更新（顺带避免“训练代理与部署执行器不一致”的风险放大）。")
    L.append("- 两臂同墙钟、同 batch/horizon/辅助损失，只有 `--policy-lr` 不同；`wm` 两臂都必须 `max|Δ| == 0`。")
    L.append("- 冻结审计里 `max|Δ| == 0` 是**精确**相等（fp32 逐元素），不是阈值判断。")
    L.append("")

    text = "\n".join(L)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
