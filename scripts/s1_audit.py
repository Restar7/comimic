#!/usr/bin/env python3
"""S1.0：WBC 侧审计快照 —— 把 spec / provenance / 权重指纹 / 已有门禁产物收进一个 json。

产出一份可复核的 `runs/S1_audit/audit.json`，回答三个问题：

1. 这个 ckpt 是用哪套执行器、哪个库、哪个 obs/interface 训出来的？（S1.1 的 `robot_spec` /
   `provenance` 键，训练端 `runner.py` 现在自带；旧 ckpt 用 `scripts/stamp_robot_spec.py` 回填）
2. 它的权重是哪一个（sha256）——闭环 / 评测表的每一行都要能指回来。
3. 已经跑过哪些门禁（`<ckpt>_eval.json` 里的 `n_failed` / `norm_return`）。

    python3 scripts/s1_audit.py
    python3 scripts/s1_audit.py --add path/to/other.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = [
    "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint_spec.pt",
    "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint_obs93_spec.pt",
    "trackerlab-supermimic/logs/wbc_mix_obs93/ckpt_012250_spec.pt",
    "trackerlab-supermimic/logs/wbc_mix_obs93/ckpt_013000_spec.pt",
    "trackerlab-supermimic/logs/wbc_mix2_obs93/ckpt_015000_spec.pt",
    "trackerlab-supermimic/logs/wbc_mix2_obs93/ckpt_015750_spec.pt",
    "trackerlab-supermimic/logs/wbc_mix2_obs93/ckpt_016000_spec.pt",
]

ARGS_KEYS = ("obs_mode", "interface", "lr", "iterations", "motion_packed", "max_episode_steps",
             "freeze_wm", "no_critic", "wm_only_iters", "hist", "goal_frames", "horizon", "amp")


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def machine_facts() -> dict:
    facts = {"hostname": platform.node(), "python": platform.python_version(),
             "date": subprocess.run(["date", "-Is"], capture_output=True, text=True).stdout.strip()}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        facts["gpus"] = [line.strip() for line in out.splitlines() if line.strip()]
    except Exception as exc:  # noqa: BLE001
        facts["gpus"] = f"unavailable: {exc}"
    return facts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs="*", default=[], help="额外要收进来的 ckpt（相对仓库根或绝对路径）")
    ap.add_argument("--out", default=str(ROOT / "runs/S1_audit/audit.json"))
    a = ap.parse_args()

    import torch

    paths = [ROOT / p for p in CANDIDATES] + [Path(p) if Path(p).is_absolute() else ROOT / p for p in a.add]
    rows, missing = [], []
    for p in paths:
        if not p.exists():
            missing.append(str(p))
            continue
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        args = ck.get("args") or {}
        ev = p.with_name(p.stem.replace("_spec", "") + "_eval.json")
        if not ev.exists():
            ev = p.with_name(p.stem + "_eval.json")
        gate = None
        if ev.exists():
            try:
                d = json.loads(ev.read_text())
                gate = {"eval_json": str(ev.relative_to(ROOT)) if ev.is_relative_to(ROOT) else str(ev),
                        "clips": len(d.get("per_clip", []) or []),
                        "n_failed": d.get("n_failed"), "norm_return": d.get("norm_return"),
                        "obs_mode": (d.get("protocol") or {}).get("observation_mode")}
            except Exception as exc:  # noqa: BLE001
                gate = {"eval_json": str(ev), "error": str(exc)}
        rows.append({
            "ckpt": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
            "iteration": ck.get("iteration"),
            "wm_integration": ck.get("wm_integration"),
            "has_robot_spec": "robot_spec" in ck,
            "has_provenance": "provenance" in ck,
            "robot_spec": ck.get("robot_spec"),
            "provenance": ck.get("provenance"),
            "train_args": {k: args.get(k) for k in ARGS_KEYS if k in args},
            "gate": gate,
        })
        print(f"[audit] {p.name}: it={ck.get('iteration')} spec={'robot_spec' in ck} "
              f"gate={'yes' if gate else '-'} sha={rows[-1]['sha256'][:12]}", flush=True)

    doc = {"machine": machine_facts(), "ckpts": rows, "missing": missing}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(f"[audit] wrote {out} ({len(rows)} ckpt, {len(missing)} missing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
