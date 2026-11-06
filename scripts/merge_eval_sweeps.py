#!/usr/bin/env python3
"""把多条**同臂对**的 closed-loop sweep 合并成一个大 n 的配对读数。

背景（2026-11-01，8 卡）：单条 sweep 只占 2 张卡、12 集 ⇒ 配对 95%CI ≈ ±1.3 cm，
分辨不了 1 cm 级的效应。8 卡机器上可以把 12 集拆成 4 条 sweep 并行（各 2 卡），
每条各自成对（同 server seed、同 reset seed），再把**逐集配对差**拼起来。

用法：
    python3 scripts/merge_eval_sweeps.py \
        runs/S3_loop/sweeps/d2b_vs_h20_8g_s0 \
        runs/S3_loop/sweeps/d2b_vs_h20_8g_s1 \
        runs/S3_loop/sweeps/d2b_vs_h20_8g_s2 \
        runs/S3_loop/sweeps/d2b_vs_h20_8g_s3 \
        --out runs/S3_loop/sweeps/d2b_vs_h20_8g_merged.json

判读口径与 `s3_eval_sweep.sh` 一致：**只看配对差**（ctrl − main）与自助法 95%CI。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return float(s[i])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps", nargs="+", help="sweep 目录（含 summary.json）")
    ap.add_argument("--out", default="", help="合并结果 json（默认打印到 stdout）")
    ap.add_argument("--label", default="", help="这组合并结果的名字")
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args()

    main_root: list[float] = []
    ctrl_root: list[float] = []
    diffs: list[float] = []
    main_tilt: list[float] = []
    ctrl_tilt: list[float] = []
    main_joint: list[float] = []
    ctrl_joint: list[float] = []
    labels: dict[str, str] = {}
    used: list[str] = []
    for s in args.sweeps:
        p = Path(s) / "summary.json"
        if not p.exists():
            print(f"[skip] 没有 summary.json: {p}")
            continue
        d = json.loads(p.read_text())
        if "main" not in d or "ctrl" not in d:
            print(f"[skip] summary 缺 main/ctrl: {p}")
            continue
        a = d["main"].get("per_seed") or []
        b = d["ctrl"].get("per_seed") or []
        if not a or len(a) != len(b):
            print(f"[skip] per_seed 长度不一致/为空: {p} ({len(a)} vs {len(b)})")
            continue
        main_root += [float(x) for x in a]
        ctrl_root += [float(x) for x in b]
        diffs += [float(y) - float(x) for x, y in zip(a, b)]
        main_tilt += [float(x) for x in (d["main"].get("per_tilt_max") or [])]
        ctrl_tilt += [float(x) for x in (d["ctrl"].get("per_tilt_max") or [])]
        if d["main"].get("joint") is not None:
            main_joint.append(float(d["main"]["joint"]))
            ctrl_joint.append(float(d["ctrl"]["joint"]))
        for k in ("main", "ctrl"):
            labels[k] = d[k].get("label", "")
        used.append(str(p.parent))

    n = len(diffs)
    if n == 0:
        raise SystemExit("[fail] 没有任何可合并的配对集")

    mu = sum(diffs) / n
    sd = (sum((x - mu) ** 2 for x in diffs) / max(1, n - 1)) ** 0.5
    sem = sd / n ** 0.5
    random.seed(0)
    boots = []
    for _ in range(int(args.boot)):
        smp = [random.choice(diffs) for _ in range(n)]
        boots.append(sum(smp) / n)
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[min(len(boots) - 1, int(0.975 * len(boots)))]
    # 双侧符号检验（配对差 < 0 = ctrl 更好）
    k = sum(1 for x in diffs if x < 0)

    def _mean(v):
        return sum(v) / len(v) if v else float("nan")

    def _sd(v):
        if len(v) < 2:
            return float("nan")
        m = _mean(v)
        return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5

    out = {
        "label": args.label or f"{labels.get('ctrl', 'ctrl')}_minus_{labels.get('main', 'main')}",
        "sweeps": used,
        "n_pairs": n,
        "main": {
            "label": labels.get("main", "main"),
            "root_mean": round(_mean(main_root), 4),
            "root_sd": round(_sd(main_root), 4),
            "root_median": round(_pct(main_root, 0.5), 4),
            "root_p90": round(_pct(main_root, 0.9), 4),
            "joint_mean": round(_mean(main_joint), 4) if main_joint else None,
            "fell": int(sum(1 for v in main_tilt if v > 45.0)),
        },
        "ctrl": {
            "label": labels.get("ctrl", "ctrl"),
            "root_mean": round(_mean(ctrl_root), 4),
            "root_sd": round(_sd(ctrl_root), 4),
            "root_median": round(_pct(ctrl_root, 0.5), 4),
            "root_p90": round(_pct(ctrl_root, 0.9), 4),
            "joint_mean": round(_mean(ctrl_joint), 4) if ctrl_joint else None,
            "fell": int(sum(1 for v in ctrl_tilt if v > 45.0)),
        },
        "paired": {
            "mean": round(mu, 4),
            "sem": round(sem, 4),
            "sd": round(sd, 4),
            "ci95_boot": [round(lo, 4), round(hi, 4)],
            "better": int(k),
            "n": n,
            "median": round(_pct(ctrl_root, 0.5) - _pct(main_root, 0.5), 4),
            "per_pair": [round(x, 4) for x in diffs],
        },
    }
    txt = json.dumps(out, indent=1, ensure_ascii=False)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt)
        print(f"[merge] -> {args.out}")
    verdict = "候选更好且 95%CI 不含 0" if hi < 0 else (
        "候选更差且 95%CI 不含 0" if lo > 0 else "CI 跨 0：不可分辨")
    print(f"[merge] {out['label']}: 配对差 {mu:+.4f} m  "
          f"95%CI [{lo:+.4f}, {hi:+.4f}]  n={n}  候选更好 {k}/{n}  ⇒ {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
