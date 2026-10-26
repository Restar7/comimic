#!/usr/bin/env python3
"""S0.1 / S1.1：把执行器 spec 快照写进 WBC ckpt（`robot_spec` 键）。

WBC 训练用的 spec 全靠环境变量（`SUPERMIMIC_HA_ALIGN=1` + `SUPERMIMIC_ROBOT_USD`），
ckpt 里不存 —— 任何消费方（联合训练 / provider / evaluate）都可能悄悄用错口径。

**训练端已经修好**：`runner.py` 现在保存时自带 `robot_spec` + `provenance`
（`trackerTask.supermimic.robot.robot_spec_snapshot`）。本脚本只给**旧 ckpt** 回填，
用的是同一个函数，所以格式逐字段一致。

    # 单个
    python3 scripts/stamp_robot_spec.py --src <old.pt> --dst <new.pt>
    # 批量（S1.1 的 6 个候选，原地加前缀 _spec）
    python3 scripts/stamp_robot_spec.py --batch
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TT = ROOT / "trackerlab-supermimic/source/trackerTask"
USD = ROOT / "trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"

#: S1.1 的候选：多跑一个就多一条，别覆盖原文件（`--batch` 会跳过已存在的 `_spec` 件）。
BATCH = [
    "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt",
    "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint_obs93.pt",
    "trackerlab-supermimic/logs/wbc_mix_obs93/ckpt_012250.pt",
    "trackerlab-supermimic/logs/wbc_mix_obs93/ckpt_013000.pt",
    "trackerlab-supermimic/logs/wbc_mix2_obs93/ckpt_015000.pt",
    "trackerlab-supermimic/logs/wbc_mix2_obs93/ckpt_015750.pt",
    "trackerlab-supermimic/logs/wbc_mix2_obs93/ckpt_016000.pt",
    "trackerlab-supermimic/logs/wbc_mix2_obs93/last.pt",
]


def _spec_of(args) -> dict:
    import torch  # noqa: F401  (import early so a broken env fails before we write anything)
    from trackerTask.supermimic import robot as rm
    from trackerTask.supermimic.robots import get_robot

    r = get_robot("g1")
    spec = rm.robot_spec_snapshot(r)
    names = spec["joint_names"]
    knee = spec["default_q"][names.index("left_knee_joint")]
    ankle = spec["default_q"][names.index("left_ankle_pitch_joint")]
    elbow = spec["default_q"][names.index("left_elbow_joint")]
    sc = spec["action_scale"]
    print(f"[stamp] HA_ALIGN={spec['ha_align']} usd={Path(spec['usd']).name} "
          f"default_q[knee/ankle/elbow]={knee:.3f}/{ankle:.3f}/{elbow:.3f} "
          f"action_scale mean/max={sum(sc) / len(sc):.4f}/{max(sc):.4f} "
          f"kp mean={sum(spec['kp']) / len(spec['kp']):.1f}", flush=True)
    return spec


def stamp_one(src: Path, dst: Path, spec: dict, provenance: dict | None, force: bool) -> int:
    import torch

    if dst.exists() and not force:
        print(f"[skip] {dst} 已存在（--force 覆盖）")
        return 0
    ck = torch.load(src, map_location="cpu", weights_only=False)
    ck["robot_spec"] = spec
    prov = dict(provenance or {})
    a = ck.get("args") or {}
    prov.setdefault("obs_mode", a.get("obs_mode"))
    prov.setdefault("interface", a.get("interface"))
    prov.setdefault("token_dim", 64 if a.get("interface") == "token" else None)
    prov.setdefault("library", a.get("motion_packed"))
    prov.setdefault("backfilled_by", "scripts/stamp_robot_spec.py")
    ck["provenance"] = {**prov, **(ck.get("provenance") or {})}
    ck["robot_spec"] = {**spec, **(ck.get("robot_spec") or {})}
    torch.save(ck, dst)
    (dst.parent / (dst.name + ".spec.json")).write_text(json.dumps(ck["robot_spec"], indent=1))
    print(f"[stamp] {src.name} -> {dst.name} (robot_spec + provenance)")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument("--dst", default=None)
    ap.add_argument("--batch", action="store_true", help="回填 BATCH 里列的旧 ckpt（写成 <stem>_spec.pt）")
    ap.add_argument("--ha-align", default="1")
    ap.add_argument("--usd", default=str(USD))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    os.environ["SUPERMIMIC_HA_ALIGN"] = str(a.ha_align)
    os.environ.setdefault("SUPERMIMIC_ROBOT_USD", a.usd)
    sys.path.insert(0, str(TT))

    spec = _spec_of(a)
    if a.batch:
        n = 0
        for rel in BATCH:
            src = ROOT / rel
            if not src.exists():
                print(f"[miss] {rel}")
                continue
            n += stamp_one(src, src.with_name(src.stem + "_spec.pt"), spec, None, a.force)
        print(f"[stamp] batch done: {n} file(s)")
        return 0

    if not a.src:
        ap.error("需要 --src/--dst 或 --batch")
    src = Path(a.src)
    dst = Path(a.dst) if a.dst else src.with_name(src.stem + "_spec.pt")
    return 0 if stamp_one(src, dst, spec, None, a.force) == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
