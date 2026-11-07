#!/usr/bin/env python3
"""harness02：Δtoken checkpoint 元数据校验 / 历史 ckpt 兼容推断。

为什么需要：`train_joint_v2.py` 用 ``load_state_dict(..., strict=False)`` 恢复 Δ 权重，
**形状不匹配不会报错**（只会被算成 missing/unexpected）。历史上 `feat_dim` 从 35 改成 64
（`DELTA_TOKEN_FULLSTATE=1`），旧 ckpt 的 meta 里没有这个字段 ⇒ 部署侧只能靠猜。

本脚本给出唯一可信的推断路径：
  * ``net.0.weight`` 形状 ``(hidden, token_dim + feat_dim)`` ⇒ feat_dim = 第二维 − token_dim
  * 线性层个数 ⇒ depth
  * meta 缺失字段按形状补齐（标 ``inferred``），meta 与形状冲突时 **FAIL**

用法::

    python3 scripts/harness02_check_delta_ckpt.py runs/gt7fix/vla_final.pt \
        runs/harness_02_teacher/dump_h20/vla_final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def inspect(path: Path) -> dict:
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ck.get("delta_token_state")
    if sd is None:
        return {"path": str(path), "status": "no_delta_token"}
    meta = dict(ck.get("delta_token_meta") or {})
    w0 = sd.get("net.0.weight")
    if w0 is None:
        return {"path": str(path), "status": "bad_state_dict", "keys": sorted(sd)}
    hidden = int(w0.shape[0])
    token_dim = int(meta.get("token_dim", 64))
    feat_dim = int(w0.shape[1]) - token_dim
    # depth = **隐层**个数；总 Linear 层数 = depth + 1（最后一个 Linear 是输出层）
    n_lin = sum(1 for k in sd if k.endswith(".weight") and k.startswith("net."))
    depth = max(0, n_lin - 1)
    inferred = {}
    if "feat_dim" not in meta:
        inferred["feat_dim"] = feat_dim
    if "token_dim" not in meta:
        inferred["token_dim"] = token_dim
    if "hidden" not in meta:
        inferred["hidden"] = hidden
    if "depth" not in meta:
        inferred["depth"] = depth
    if "cap" not in meta:
        inferred["cap"] = 0.0
    if "input_features" not in meta:
        inferred["input_features"] = ("state64[rot6d6|dof_pos29|dof_vel29]" if feat_dim == 64
                                      else "rot6d6+dof_vel29")
    conflicts = []
    for k, v in (("feat_dim", feat_dim), ("token_dim", token_dim),
                 ("hidden", hidden), ("depth", depth)):
        if k in meta and int(meta[k]) != int(v):
            conflicts.append(f"{k}: meta={meta[k]} vs state_dict={v}")
    status = "inconsistent" if conflicts else ("legacy_inferred" if meta.get("schema_version") is None
                                               else "ok")
    merged = {**meta, **inferred}
    merged.setdefault("schema_version", 1)
    merged.setdefault("apply", "broadcast_shared_over_chunk")
    merged.setdefault("norm_cap", "apply_norm_cap(tanh)")
    merged.setdefault("teacher_objective", "unknown(legacy)")
    return {"path": str(path), "status": status, "meta": merged,
            "inferred": inferred, "conflicts": conflicts,
            "state_shapes": {k: list(v.shape) for k, v in sd.items() if k.endswith(".weight")},
            "trained_steps": ck.get("step"), "git_commit": merged.get("git_commit")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    out = []
    bad = 0
    for p in a.ckpts:
        r = inspect(Path(p))
        out.append(r)
        print(f"== {p} ==")
        print(f"   status = {r['status']}")
        if r.get("conflicts"):
            print(f"   conflicts = {r['conflicts']}")
            bad += 1
        if r.get("inferred"):
            print(f"   inferred = {json.dumps(r['inferred'], ensure_ascii=False)}")
        if r.get("meta"):
            print(f"   meta = {json.dumps(r['meta'], ensure_ascii=False)}")
        if r.get("state_shapes"):
            print(f"   shapes = {json.dumps(r['state_shapes'])}")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"[saved] {a.json_out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
