#!/usr/bin/env python3
"""把多轮闭环的 eval dump 汇总成一张 markdown 表（root / joint / tilt / 摔倒）。

用法：
    PYTHONPATH=. python3 scripts/s3_round_table.py --out runs/S3_loop/ROUNDS.md
    PYTHONPATH=. python3 scripts/s3_round_table.py            # 只打印

口径与 `plot_drift.py` **完全一致**（states[:, :3] / states[:, 7:36] / 四元数 tilt），
所以表里的数字和图上、和文档里的 0.082 / 0.200 可以直接比。

约定（都按"同一条起跑线"排）：DoubleDesk / row 0 / seed 0 / 300 步。
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def stats(p: Path, lib: torch.Tensor) -> dict:
    d = torch.load(str(p), map_location="cpu", weights_only=False)
    S = d["states"].float()
    fr = d["frames"].long().clamp(0, int(lib.shape[0]) - 1)
    ref = lib[fr]
    e_root = (S[:, :3] - ref[:, :3]).norm(dim=-1)
    joint = ((S[:, 7:36] - ref[:, 7:36]) ** 2).mean(-1).sqrt()
    w, x, y, z = S[:, 3:7].unbind(-1)
    up = 1 - 2 * (x * x + y * y)
    tilt = torch.rad2deg(torch.acos(up.clamp(-1, 1)))
    n = int(e_root.numel())
    head = min(100, n)
    return {
        "n": n,
        "root_mean": float(e_root.mean()),
        "root_p50": float(e_root.quantile(.5)),
        "root_p90": float(e_root.quantile(.9)),
        "root_head100": float(e_root[:head].mean()),
        "joint_mean": float(joint.mean()),
        "tilt_max": float(tilt.max()),
        "fell": bool(float(tilt.max()) > 60.0),
        "file": str(p),
    }


def collect_specs() -> list[tuple[str, str]]:
    """(label, glob) —— 顺序就是表里的顺序。"""
    C = ROOT
    specs: list[tuple[str, str]] = [
        ("GT online (π 无 VLA)", str(C / "runs/B_r1/gt_DoubleDesk_f0_r0/dump/*.pt")),
        ("π₀ 底模 (baseline)",
         str(C / "runs/B_r1/vla_DoubleDesk_f0_r0/dump/*.pt")),
    ]
    for r in (1, 2, 3, 4, 5):
        rd = C / f"runs/S3_loop/r{r}"
        if not rd.exists():
            continue
        for d in sorted(rd.glob("eval_*")):
            g = str(d / "dump/*.pt")
            if not glob.glob(g):
                continue
            bits = d.name.split("_", 2)
            if len(bits) == 2:      # eval_main / eval_ctrl
                nm = "主臂 λ_R=0.3" if bits[1] == "main" else "对照 λ_R=0"
                label = f"π{r} {nm}"
            else:                   # eval_main_m50_100 = 扫描 ckpt
                label = f"π{r} {bits[1]} @{bits[2]}"
            specs.append((label, g))
    return specs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None, help="写 markdown 文件")
    ap.add_argument("--ref", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    a = ap.parse_args()

    lib = torch.load(str(a.ref), map_location="cpu", weights_only=False)["ref"].float()
    rows = []
    for label, g in collect_specs():
        files = sorted(glob.glob(g))
        if not files:
            continue
        per = [stats(Path(f), lib) for f in files]
        # 多 dump 时按 root_mean 汇总（每集等权）
        rows.append({
            "label": label,
            "n": sum(p["n"] for p in per),
            "root_mean": sum(p["root_mean"] for p in per) / len(per),
            "root_p50": sum(p["root_p50"] for p in per) / len(per),
            "root_p90": sum(p["root_p90"] for p in per) / len(per),
            "root_head100": sum(p["root_head100"] for p in per) / len(per),
            "joint_mean": sum(p["joint_mean"] for p in per) / len(per),
            "tilt_max": max(p["tilt_max"] for p in per),
            "fell": any(p["fell"] for p in per),
            "episodes": len(per),
        })

    base = next((r for r in rows if r["label"].startswith("π₀")), None)
    lines = [
        "| 臂 | 集数 | 步 | root 均值 | p50 | p90 | 前100步 | joint | tilt max | 摔 | vs π₀ |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rel = "—" if not base else f"{100*(r['root_mean']/base['root_mean']-1):+.1f}%"
        lines.append(
            f"| {r['label']} | {r['episodes']} | {r['n']} | **{r['root_mean']:.3f}** | "
            f"{r['root_p50']:.3f} | {r['root_p90']:.3f} | {r['root_head100']:.3f} | "
            f"{r['joint_mean']:.3f} | {r['tilt_max']:.1f}° | "
            f"{'**是**' if r['fell'] else '否'} | {rel} |")

    # ---- 多种子配对扫描（s3_eval_sweep.sh 的产物）单独一节 ----
    sweeps = sorted((ROOT / "runs/S3_loop/sweeps").glob("*/summary.json"))
    if sweeps:
        lines += ["", "## 多种子配对扫描（row 0 / 300 步；同一时段两卡同时跑）", "",
                  "| sweep | 槽位/实际 head | n | root 均值 | SD | 每 seed |", "|---|---|---|---|---|---|"]
        for f in sweeps:
            d = json.loads(f.read_text())
            heads = {}
            hj = f.parent / "heads.json"
            if hj.exists():
                heads = json.loads(hj.read_text())
            for arm, slot in (("main", "GPU0"), ("ctrl", "GPU1")):
                if arm not in d:
                    continue
                r = d[arm]
                # 槽位名不代表策略！优先用 heads.json 里记录的真实 head
                label = heads.get(arm, {}).get("label") or r.get("label") or slot
                seeds = " / ".join(f"{v:.3f}" for v in r.get("per_seed", []))
                lines.append(f"| {f.parent.name} | {slot} `{label}` | {r['n']} | "
                             f"**{r['root_mean']:.3f}** | {r['root_sd']:.3f} | {seeds} |")

    md = "\n".join(lines) + "\n"
    print(md)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(md)
        (a.out.with_suffix(".json")).write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        print(f"[table] saved {a.out} (+ .json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
