#!/usr/bin/env python3
"""把两条闭环路线（L1 官方 SONIC decoder / L2 联合微调 SuperMimic）的产物汇总成一张对比表。

用法：
    python3 scripts/compare_two_routes.py --l1 runs/compare_two_routes/L1 \
                                          --l2 runs/compare_two_routes/L2 \
                                          --out runs/compare_two_routes/COMPARE.md

只读 `summary.json`（由 `scripts/ha_isaac_batch.sh` 产出）和逐集日志，不改任何东西。
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import re
from pathlib import Path

ROUTE_META = {
    "l1": {
        "title": "L1 官方线",
        "vla": "官方微调 `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`",
        "exec": "**SONIC release decoder**（`model_decoder.onnx`，994→29）",
        "provider": "SonicActionProvider",
        "decoder_re": re.compile(r"decoder=(\S*model_decoder\.onnx)"),
    },
    "l2": {
        "title": "L2 联合线",
        "vla": "联合微调 ckpt（merged，见同目录 `vla_merge_report.json`）",
        "exec": "**SuperMimic WBC**（token 条件化 policy）",
        "provider": "SuperMimic",
        "decoder_re": None,
    },
}


def load_rows(root: Path) -> list[dict]:
    f = root / "summary.json"
    if not f.is_file():
        return []
    try:
        data = json.loads(f.read_text())
    except Exception:  # noqa: BLE001
        return []
    return list(data.get("rows") or [])


def per_task(rows: list[dict]) -> dict[str, dict]:
    agg: dict[str, dict] = collections.OrderedDict()
    for r in rows:
        task = str(r.get("task", "?"))
        a = agg.setdefault(task, {"n": 0, "ok": 0, "steps": [], "rew": []})
        a["n"] += 1
        a["ok"] += int(bool(r.get("success")))
        a["steps"].append(int(r.get("episode_steps") or 0))
        a["rew"].append(float(r.get("max_reward") or 0.0))
    for a in agg.values():
        a["avg_steps"] = round(sum(a["steps"]) / len(a["steps"]), 1) if a["steps"] else 0.0
        a["avg_rew"] = round(sum(a["rew"]) / len(a["rew"]), 4) if a["rew"] else 0.0
        a["rate"] = round(a["ok"] / a["n"], 4) if a["n"] else 0.0
    return agg


def failure_modes(rows: list[dict]) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for r in rows:
        if r.get("success"):
            c["success"] += 1
        else:
            c[str(r.get("failure_reason") or "unknown")] += 1
    return c


def provider_hits(root: Path, keyword: str) -> tuple[int, int]:
    logs = list(root.glob("**/*.log"))
    hits = sum(1 for f in logs if keyword in f.read_text(errors="ignore"))
    return hits, len(logs)


def decoder_paths(root: Path, pattern: re.Pattern | None) -> set[str]:
    if pattern is None:
        return set()
    found: set[str] = set()
    for f in root.glob("**/*.log"):
        found.update(pattern.findall(f.read_text(errors="ignore")))
    return found


def video_counts(root: Path) -> tuple[int, int]:
    ok = len(list(root.glob("**/videos_success/*.mp4")))
    fail = len(list(root.glob("**/videos_failure/*.mp4")))
    return ok, fail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1", type=Path, required=True)
    ap.add_argument("--l2", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # L2 实际用的是哪个 vla_*.pt：读 merge report，别写死
    mr = args.l2.parent / "ckpt_vla_joint4000" / "vla_merge_report.json"
    if mr.is_file():
        try:
            d = json.loads(mr.read_text())
            ROUTE_META["l2"]["vla"] = f"联合微调 `{Path(d['update']).name}`（step {d.get('step')}，merged）"
        except Exception:  # noqa: BLE001
            pass

    roots = {"l1": args.l1, "l2": args.l2}
    rows = {k: load_rows(v) for k, v in roots.items()}
    tasks = {k: per_task(v) for k, v in rows.items()}
    modes = {k: failure_modes(v) for k, v in rows.items()}
    lines: list[str] = []

    lines.append("# 两条路线闭环对比：L1 官方微调 + SONIC decoder vs L2 联合微调 + SuperMimic")
    lines.append("")
    lines.append(f"- 生成时间：{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- 协议：HumanoidArena Isaac Sim 5.0 / IsaacLab 2.2，6 任务，seed 见各表，max_steps 见逐集")
    lines.append("- 视频：`videos_success/`、`videos_failure/`（逐集前视相机）")
    lines.append("")

    # ---- 总览 ----
    lines.append("## 1. 总览（成功率）")
    lines.append("")
    lines.append("| 路线 | VLA | 执行器 | episodes | success | **成功率** | avg_steps | avg_max_reward |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k in ("l1", "l2"):
        meta, rs = ROUTE_META[k], rows[k]
        n = len(rs)
        ok = sum(1 for r in rs if r.get("success"))
        steps = [int(r.get("episode_steps") or 0) for r in rs]
        rew = [float(r.get("max_reward") or 0.0) for r in rs]
        rate = f"**{ok / n:.1%}**" if n else "—"
        avg_s = round(sum(steps) / len(steps), 1) if steps else "—"
        avg_r = round(sum(rew) / len(rew), 4) if rew else "—"
        lines.append(
            f"| {meta['title']} | {meta['vla']} | {meta['exec']} | {n} | {ok} | {rate} | {avg_s} | {avg_r} |"
        )
    lines.append("")

    # ---- 口径校验 ----
    lines.append("## 2. 口径校验（执行器对不对）")
    lines.append("")
    lines.append("| 路线 | 期望 provider | 命中 episode 日志 | 实际 decoder | 判定 |")
    lines.append("|---|---|---|---|---|")
    for k in ("l1", "l2"):
        meta = ROUTE_META[k]
        hit, total = provider_hits(roots[k], meta["provider"])
        dec = ", ".join(sorted(decoder_paths(roots[k], meta["decoder_re"]))) or "—"
        verdict = "✅" if hit else ("❌ 没有命中（结果不可用）" if total else "— 没有日志")
        lines.append(f"| {meta['title']} | `{meta['provider']}` | {hit}/{total} | `{dec}` | {verdict} |")
    lines.append("")

    # ---- 分任务 ----
    lines.append("## 3. 分任务")
    lines.append("")
    lines.append("| task | L1 success | L1 rate | L1 avg_steps | L2 success | L2 rate | L2 avg_steps |")
    lines.append("|---|---|---|---|---|---|---|")
    all_tasks = list(dict.fromkeys(list(tasks["l1"].keys()) + list(tasks["l2"].keys())))
    for t in sorted(all_tasks):
        a, b = tasks["l1"].get(t), tasks["l2"].get(t)
        def cell(d, field, fmt):  # noqa: ANN001
            if not d:
                return "—"
            return fmt.format(d[field])
        lines.append(
            "| {t} | {a} | {ar} | {as_} | {b} | {br} | {bs} |".format(
                t=t,
                a=cell(a, "ok", "{:.0f}"), ar=cell(a, "rate", "{:.0%}"), as_=cell(a, "avg_steps", "{}"),
                b=cell(b, "ok", "{:.0f}"), br=cell(b, "rate", "{:.0%}"), bs=cell(b, "avg_steps", "{}"),
            )
        )
    lines.append("")

    # ---- 失败原因 ----
    lines.append("## 4. 失败原因分布")
    lines.append("")
    lines.append("| 路线 | success | fall | timeout | other |")
    lines.append("|---|---|---|---|---|")
    for k in ("l1", "l2"):
        m = modes[k]
        other = sum(v for kk, v in m.items() if kk not in ("success", "fall", "timeout"))
        lines.append(
            f"| {ROUTE_META[k]['title']} | {m.get('success', 0)} | {m.get('fall', 0)} | "
            f"{m.get('timeout', 0)} | {other} |"
        )
    lines.append("")

    # ---- 视频 ----
    lines.append("## 5. 视频产物")
    lines.append("")
    lines.append("| 路线 | 成功视频 | 失败视频 | 目录 |")
    lines.append("|---|---|---|---|")
    for k in ("l1", "l2"):
        ok_v, fail_v = video_counts(roots[k])
        lines.append(f"| {ROUTE_META[k]['title']} | {ok_v} | {fail_v} | `{roots[k]}` |")
    lines.append("")

    # ---- 逐集 ----
    lines.append("## 6. 逐集明细")
    lines.append("")
    lines.append("| 路线 | task | seed | success | reason | steps | max_reward | video |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k in ("l1", "l2"):
        for r in sorted(rows[k], key=lambda x: (str(x.get("task")), int(x.get("seed") or 0))):
            vid = Path(str(r.get("video_path") or "")).name or "—"
            lines.append(
                f"| {ROUTE_META[k]['title']} | {r.get('task', '?')} | {r.get('seed')} | "
                f"{'✅' if r.get('success') else '❌'} | {r.get('failure_reason') or '—'} | "
                f"{r.get('episode_steps')} | {r.get('max_reward')} | {vid} |"
            )
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"[compare] -> {args.out}")
    for k in ("l1", "l2"):
        n, ok = len(rows[k]), sum(1 for r in rows[k] if r.get("success"))
        print(f"[compare] {ROUTE_META[k]['title']}: {ok}/{n} = {ok / n:.1%}" if n else f"[compare] {ROUTE_META[k]['title']}: 无数据")


if __name__ == "__main__":
    main()
