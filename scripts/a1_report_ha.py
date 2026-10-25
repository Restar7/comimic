#!/usr/bin/env python3
"""Aggregate the A1 pure-HA IsaacLab closed-loop results.

Reads ``<root>/<arm>/runs/**/result.json`` produced by
``scripts/ha_isaac_batch.sh`` (Isaac Sim 5.0 + IsaacLab 2.2 + HumanoidArena)
and writes ``summary.json`` / ``SUMMARY.md`` next to them.

Metrics
-------
* success rate with Wilson 95% CI, overall and per task;
* non-fall rate (``failure_reason != "fall"``), timeout/fall counts;
* survival steps and ``max_reward`` (mean / median / max);
* paired exact McNemar test over the same (task, seed, repeat);
* the joint-finetune training curve (``eval_curve.jsonl``) as a secondary signal.

Usage
-----
    python3 scripts/a1_report_ha.py --root runs/ablations/A1_joint_finetune_ha
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    "DoubleDesk": "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody",
    "PPBox": "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby",
    "Boxing": "Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody",
    "OpenDoor": "Isaac-Move-Open-Door-G129-Dex3-Wholebody",
    "SitSofa": "Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody",
    "VisionNavi": "Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody",
}


def task_alias(task_id: str) -> str:
    for alias, full in TASKS.items():
        if task_id == full or alias.lower() in task_id.lower():
            return alias
    return task_id


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b01: int, b10: int) -> float:
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return round(min(1.0, 2.0 * tail), 4)


def mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    m = len(s) // 2
    return round(s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2, 4)


def load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.glob("*/runs/**/result.json")):
        try:
            d = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001 - a bad json must not hide the rest
            print(f"[warn] skip {path}: {exc}")
            continue
        arm = path.relative_to(root).parts[0]
        d["_arm"] = d.get("model_label") or arm
        d["_task_alias"] = task_alias(str(d.get("task", "?")))
        d["_result"] = str(path)
        rows.append(d)
    return rows


def stats(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(1 for r in rows if r.get("success"))
    lo, hi = wilson(k, n)
    reasons = Counter(str(r.get("failure_reason") or ("success" if r.get("success") else "?"))
                      for r in rows)
    steps = [float(r.get("episode_steps") or 0) for r in rows]
    rewards = [float(r.get("max_reward") or 0.0) for r in rows]
    durations = [float(r.get("duration_sec") or 0.0) for r in rows]
    finals = [float(r.get("final_reward") or 0.0) for r in rows]
    not_fell = sum(1 for r in rows if str(r.get("failure_reason") or "") != "fall")
    return {
        "episodes": n, "success": k, "rate": round(k / n, 4) if n else 0.0,
        "ci95": [round(lo, 4), round(hi, 4)],
        "not_fell": not_fell, "not_fell_rate": round(not_fell / n, 4) if n else 0.0,
        "fall": reasons.get("fall", 0), "timeout": reasons.get("timeout", 0),
        "failure_reasons": dict(reasons),
        "steps_mean": mean(steps), "steps_median": median(steps),
        "steps_max": max(steps) if steps else None,
        "max_reward_mean": mean(rewards), "max_reward_max": max(rewards) if rewards else None,
        "final_reward_mean": mean(finals),
        "duration_sec_mean": mean(durations),
    }


def summarize(rows: list[dict]) -> dict:
    arms = sorted({r["_arm"] for r in rows})
    tasks = sorted({r["_task_alias"] for r in rows})
    out: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "episodes": len(rows), "arms": arms, "tasks": tasks,
        "per_arm": {}, "paired": {},
    }
    for arm in arms:
        ar = [r for r in rows if r["_arm"] == arm]
        per_task = {t: stats([r for r in ar if r["_task_alias"] == t]) for t in tasks}
        out["per_arm"][arm] = {"overall": stats(ar), "per_task": per_task}

    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b = arms[i], arms[j]
            key = f"{a}_vs_{b}"
            by = {(r["_arm"], r["_task_alias"], r.get("seed"), r.get("repeat_idx")): r
                  for r in rows if r["_arm"] in (a, b)}
            per_task = {}
            tot = {"pairs": 0, "a_only": 0, "b_only": 0, "both": 0, "neither": 0}
            for task in tasks:
                keys = sorted({k[1:] for k in by if k[0] == a and k[1] == task})
                pairs = [(by[(a,) + k], by[(b,) + k]) for k in keys if (b,) + k in by]
                c = {"pairs": len(pairs), "a_only": 0, "b_only": 0, "both": 0, "neither": 0}
                for ra, rb in pairs:
                    sa, sb = bool(ra.get("success")), bool(rb.get("success"))
                    if sa and sb:
                        c["both"] += 1
                    elif sa:
                        c["a_only"] += 1
                    elif sb:
                        c["b_only"] += 1
                    else:
                        c["neither"] += 1
                c["net_b_minus_a"] = c["b_only"] - c["a_only"]
                c["mcnemar_p"] = mcnemar_exact(c["a_only"], c["b_only"])
                per_task[task] = c
                for k in tot:
                    tot[k] += c[k]
            tot["net_b_minus_a"] = tot["b_only"] - tot["a_only"]
            tot["mcnemar_p"] = mcnemar_exact(tot["a_only"], tot["b_only"])
            out["paired"][key] = {"per_task": per_task, "overall": tot}
    return out


def training_curve(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return rows


def markdown(s: dict, curve: list[dict], meta: dict) -> str:
    arms = s["arms"]
    tasks = s["tasks"]
    lines = ["# A1 — 有无联合微调（纯 HA + IsaacLab/HumanoidArena）", "",
             f"_generated: {s['generated_at']} · episodes: {s['episodes']}_", ""]
    lines += ["## 0. 口径与检查点", "",
              "| 项 | 值 |", "|---|---|"]
    for k, v in meta.items():
        lines.append(f"| {k} | `{v}` |")
    lines += ["", "> 硬约束：纯 HA 数据 + IsaacLab/PhysX WBC；mixed/ft318/MuJoCo 一律禁用"
              "（`docs/DATA_POLICY_HA_ONLY.md`）。", ""]

    lines += ["## 1. 成功率（主指标）", "",
              "| task | " + " | ".join(arms) + " |",
              "|---|" + "|".join(["---"] * len(arms)) + "|"]
    for task in tasks:
        cells = []
        for arm in arms:
            v = s["per_arm"][arm]["per_task"][task]
            cells.append(f"{v['success']}/{v['episodes']} ({v['rate'] * 100:.0f}%) "
                         f"[{v['ci95'][0] * 100:.0f}, {v['ci95'][1] * 100:.0f}]")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    cells = []
    for arm in arms:
        v = s["per_arm"][arm]["overall"]
        cells.append(f"**{v['success']}/{v['episodes']} ({v['rate'] * 100:.0f}%)** "
                     f"[{v['ci95'][0] * 100:.0f}, {v['ci95'][1] * 100:.0f}]")
    lines += ["| **all** | " + " | ".join(cells) + " |", ""]

    lines += ["## 2. 未摔倒率 / 失败原因", "",
              "| task | " + " | ".join(arms) + " |",
              "|---|" + "|".join(["---"] * len(arms)) + "|"]
    for task in tasks:
        cells = []
        for arm in arms:
            v = s["per_arm"][arm]["per_task"][task]
            cells.append(f"{v['not_fell']}/{v['episodes']} ({v['not_fell_rate'] * 100:.0f}%) · "
                         f"fall={v['fall']} timeout={v['timeout']}")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    lines += ["", "> `not_fell` 包含 timeout（没摔但没完成）；正式成功率只看上面的 success。", ""]

    lines += ["## 3. 存活步数 / max_reward（连续量）", "",
              "| task | arm | steps mean | steps median | steps max | max_reward mean | max_reward max |",
              "|---|---|---|---|---|---|---|"]
    for task in tasks:
        for arm in arms:
            v = s["per_arm"][arm]["per_task"][task]
            lines.append(f"| {task} | {arm} | {v['steps_mean']} | {v['steps_median']} | "
                         f"{v['steps_max']} | {v['max_reward_mean']} | {v['max_reward_max']} |")
    lines.append("")

    if s["paired"]:
        lines += ["## 4. 配对检验（同 task/seed/repeat；McNemar 精确检验）", ""]
        for key, block in s["paired"].items():
            a, b = key.split("_vs_")
            lines += [f"`{a}` vs `{b}`：b_only = {b} 独赢，a_only = {a} 独赢。", "",
                      "| task | pairs | a_only | b_only | both | neither | net(b-a) | p |",
                      "|---|---|---|---|---|---|---|---|"]
            for task, v in block["per_task"].items():
                lines.append(f"| {task} | {v['pairs']} | {v['a_only']} | {v['b_only']} | "
                             f"{v['both']} | {v['neither']} | {v['net_b_minus_a']} | {v['mcnemar_p']} |")
            v = block["overall"]
            lines.append(f"| **all** | {v['pairs']} | {v['a_only']} | {v['b_only']} | "
                         f"{v['both']} | {v['neither']} | {v['net_b_minus_a']} | {v['mcnemar_p']} |")
            lines.append("")

    if curve:
        lines += ["## 5. 联合微调训练曲线（纯 HA，`train_joint.py`）", "",
                  "| step | token_mse | state_term | rollout_reward | hand_mse |",
                  "|---|---|---|---|---|"]
        for r in curve:
            lines.append(f"| {r.get('step')} | {r.get('token_mse')} | {r.get('state_term')} | "
                         f"{r.get('rollout_reward')} | {r.get('hand_mse')} |")
        lines += ["", "> 这是训练侧的想象 rollout 指标，不等于闭环成功率；闭环以第 1 节为准。", ""]

    if s.get("expected_episodes"):
        lines += ["## 5.5 运行完整性（缺集检查）", "",
                  "| arm | expected | episodes | missing |",
                  "|---|---|---|---|"]
        for arm in arms:
            n = s["per_arm"][arm]["overall"]["episodes"]
            miss = s["missing_episodes"][arm]
            lines.append(f"| {arm} | {s['expected_episodes']} | {n} | "
                         f"{'**' + str(miss) + '**' if miss else 0} |")
        lines.append("")
    lines += ["## 6. 判读边界", "",
              "- 6 个 HA 任务；`GrapCup` 在本 checkout 缺资产，未跑。",
              "- 两臂共用同一个纯 HA WBC（`bones_seed_token_ha_v2/ckpt_*.pt`，显式冻结），"
              "只消融 VLA 的 action head。",
              "- 二进制 episode 的粒度是 1/n；配对 McNemar 的 p 值优先于肉眼比较。",
              "- 若两臂 success 都是 0，说明当前 WBC 闭环还站不住，"
              "此时只能比存活步数/max_reward，不能声称任何成功率结论。", ""]
    return "\n".join(lines) + "\n"


def expected_episodes(root: Path) -> int | None:
    """Expected episodes per arm from the driver-written manifest (None if unknown)."""
    path = root / "manifest.json"
    if not path.exists():
        return None
    try:
        proto = json.loads(path.read_text()).get("protocol", {})
        tasks = str(proto.get("tasks", "")).split()
        seeds = str(proto.get("seeds", "")).split()
        repeats = int(proto.get("repeats", 1))
    except Exception:  # noqa: BLE001
        return None
    return len(tasks) * len(seeds) * repeats if tasks and seeds else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--train-curve", type=Path,
                    default=ROOT / "runs" / "joint_ha_a1" / "eval_curve.jsonl")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = args.root
    rows = load_rows(root)
    if not rows:
        raise SystemExit(f"no result.json under {root}/*/runs/**")
    s = summarize(rows)
    for arm in s["arms"]:
        n = s["per_arm"][arm]["overall"]["episodes"]
        if n == 0:
            raise SystemExit(f"arm {arm} has zero episodes")
    expected = expected_episodes(root)
    if expected is not None:
        s["expected_episodes"] = expected
        s["missing_episodes"] = {
            arm: expected - s["per_arm"][arm]["overall"]["episodes"] for arm in s["arms"]
        }
    curve = training_curve(args.train_curve)

    meta = {
        "root": str(root.resolve()),
        "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip() or "?",
        "tasks": " ".join(s["tasks"]),
        "arms": ", ".join(s["arms"]),
        "training_curve": str(args.train_curve),
    }
    # checkpoint provenance written by scripts/run_a1_ha_isaac.sh
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            meta.update(json.loads(manifest_path.read_text()))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] bad manifest {manifest_path}: {exc}")
    for arm in s["arms"]:
        meta[f"{arm}_dir"] = str((root / arm).resolve())

    out = args.out or root
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        {"summary": s, "manifest": meta, "training_curve": curve,
         "per_episode": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]},
        indent=2, default=str))
    (out / "SUMMARY.md").write_text(markdown(s, curve, meta))
    print(f"[wrote] {out / 'summary.json'}")
    print(f"[wrote] {out / 'SUMMARY.md'}")
    print(markdown(s, curve, meta))
    if any(v > 0 for v in s.get("missing_episodes", {}).values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
