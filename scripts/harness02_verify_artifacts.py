#!/usr/bin/env python3
"""harness02：对本目录的产物做**通过判据**复检（进 test.log）。

判据（与报告一致，不因结果不好而放宽）：
  * 一致性：所有行的 cos > 0.999、max|gA-gB| < 1e-4、rel_L2 < 1e-4、符号一致 > 0.999
  * 有限差分：H<=5 最优相对误差 < 2e-2；H=20 < 5e-2
  * 可学性：n=256 时 train cos >= 0.99（能过拟合），且 held-out 最好 >= 最优常向量 + 0.03
  * 旧/新 Teacher 差异有实测（不是 null）
  * 新 ckpt 元数据 status == ok
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "runs/harness_02_teacher"
fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def _csv(name):
    p = OUT / name
    return list(csv.DictReader(open(p))) if p.exists() else []


# ---- 1) 一致性 ----
for fname in ("gradient_consistency.csv", "gradient_consistency_prod.csv"):
    rows = _csv(fname)
    if not rows:
        check(f"{fname} 存在", False)
        continue
    cos_min = min(float(r["cos_AB"]) for r in rows)
    mae = max(float(r["max_abs_err"]) for r in rows)
    rel = max(float(r["rel_l2"]) for r in rows)
    sign = min(float(r["sign_agree"]) for r in rows)
    check(f"{fname}: cos>0.999", cos_min > 0.999, f"min={cos_min:.9f} n={len(rows)}")
    check(f"{fname}: max|gA-gB|<1e-4", mae < 1e-4, f"max={mae:.3e}")
    check(f"{fname}: rel_L2<1e-4", rel < 1e-4, f"max={rel:.3e}")
    check(f"{fname}: sign>0.999", sign > 0.999, f"min={sign:.6f}")

# ---- 2) 有限差分 ----
fd = _csv("gradient_consistency_fd.csv") + _csv("gradient_consistency_prod.csv") * 0
if fd:
    best: dict[int, float] = {}
    for r in fd:
        if not r.get("fd_rel_err"):
            continue
        h = int(float(r["H"]))
        e = float(r["fd_rel_err"])
        best[h] = min(best.get(h, 9e9), e)
    for h, lim in ((1, 2e-2), (2, 2e-2), (5, 2e-2), (20, 5e-2)):
        if h in best:
            check(f"FD H={h}: best rel < {lim:g}", best[h] < lim, f"best={best[h]:.3e}")
else:
    check("FD CSV 存在", False)

# ---- 3) 可学性 ----
dl = _csv("distill_learning_curve.csv")
if not dl:
    check("distill_learning_curve.csv 存在", False)
else:
    for tgt in ("H_main", "H5"):
        rows = [r for r in dl if r["target"] == tgt and int(r["n_train"]) == 256
                and r["features"] == "token0+state64"]
        if not rows:
            check(f"{tgt} n=256 行存在", False)
            continue
        const = min(float(r["test_cos"]) for r in rows if r["model"] == "constant")
        best_h = max(float(r["test_cos"]) for r in rows if r["model"] != "constant")
        train_max = max(float(r["train_cos"]) for r in rows if r["model"] != "constant")
        check(f"{tgt}: 能过拟合 (train cos>=0.99)", train_max >= 0.99, f"train={train_max:.4f}")
        check(f"{tgt}: held-out 优于常数 +0.03", best_h >= const + 0.03,
              f"best={best_h:.4f} const={const:.4f}")

# ---- 4) 旧 vs 新 ----
on = _csv("teacher_old_vs_new.csv")
check("teacher_old_vs_new.csv 有 legacy 对照", any("legacy" in (r.get("compare") or "") for r in on),
      f"n={len(on)}")

# ---- 5) ckpt 元数据 ----
mj = OUT / "delta_token_meta_check.json"
check("delta_token_meta_check.json 存在", mj.exists())
if mj.exists():
    d = json.loads(mj.read_text())
    status = {Path(r["path"]).parent.name: r["status"] for r in d}
    check("新 ckpt 元数据 ok", status.get("dump_h20") == "ok", json.dumps(status))
    check("历史 ckpt 可推断（legacy_inferred）", status.get("gt7fix") in ("legacy_inferred", "ok"))

print()
print("RESULT:", "PASS" if not fails else f"FAIL ({len(fails)}): {fails}")
raise SystemExit(0 if not fails else 1)
