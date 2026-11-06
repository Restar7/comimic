#!/usr/bin/env python3
"""P1a：构建 WBC 训练库 ``bones_seed_plus_ha_exec_x20.pt``。

配方（docs/PLAN_REFSTART_WM_PERTURB_20261102.md §6.1）::

    bones_seed 129,943 clips 原样
  + ha_cmd_exec 699 clips × 20（名字 ``<clip>__HA_CMDr0..r19``）
  = 143,923 clips / 60,156,000 frames

要点：

* **bones 段逐位不变**（只读 ``bones_seed_compact.pt``，不重采样、不改 q/root）；
* HA 段用**执行参考库** ``runs/isaac_motion/ha_cmd_exec/packed.pt``（q/root/速度全是执行量），
  ×20 只是**上采样权重**（同一 clip 的 20 份 ref 完全相同，仅名字差 ``__HA_CMDr<k>``），
  与旧库 ``bones_seed_plus_ha_cmd_x20.pt`` 的 HA 段索引结构逐项同构；
* token sidecar 不动：训练时 token 由 ``SonicTokenBuilder`` 从库行在线算（hold 窗口），
  这条链不需要额外文件。

用法::

    python3 scripts/build_wbc_lib.py --limit-bones 8 --limit-ha 3 --x20 2 \\
        --out /tmp/bones_plus_exec_smoke.pt                                # 冒烟
    python3 scripts/build_wbc_lib.py \\
        --bones trackerlab-supermimic/data/bones_seed/bones_seed_compact.pt \\
        --ha-exec runs/isaac_motion/ha_cmd_exec/packed.pt --x20 20 \\
        --out trackerlab-supermimic/data/bones_seed/bones_seed_plus_ha_exec_x20.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NUM_JOINTS = 29
REF_DIM = 13 + 2 * NUM_JOINTS


def _load_packed(path: Path) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    ref = d["ref"]
    names = list(d["names"])
    starts = [int(x) for x in d["starts"]]
    lengths = [int(x) for x in d["lengths"]]
    if ref.shape[1] != REF_DIM:
        raise ValueError(f"{path}: ref width {ref.shape[1]} != {REF_DIM}")
    if starts[-1] + lengths[-1] != ref.shape[0]:
        raise ValueError(f"{path}: index does not cover ref")
    return {"ref": ref, "names": names, "starts": starts, "lengths": lengths}


def _starts(lengths: list[int]) -> list[int]:
    return [int(s) for s in np.cumsum([0] + lengths[:-1])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bones", type=Path,
                    default=ROOT / "trackerlab-supermimic/data/bones_seed/bones_seed_compact.pt")
    ap.add_argument("--ha-exec", type=Path,
                    default=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt")
    ap.add_argument("--x20", type=int, default=20, help="HA 段重复次数（上采样权重）")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "trackerlab-supermimic/data/bones_seed/"
                                   "bones_seed_plus_ha_exec_x20.pt")
    ap.add_argument("--limit-bones", type=int, default=0, help="调试用：只取前 N 条 bones")
    ap.add_argument("--limit-ha", type=int, default=0, help="调试用：只取前 N 条 HA")
    ap.add_argument("--verify-against", type=Path,
                    default=ROOT / "trackerlab-supermimic/data/bones_seed/"
                                   "bones_seed_plus_ha_cmd_x20.pt",
                    help="旧 x20 库：只用来断言索引结构同构（空字符串关闭）")
    a = ap.parse_args()

    t0 = time.time()
    bones = _load_packed(a.bones)
    ha = _load_packed(a.ha_exec)
    print(f"[wbc-lib] bones: {len(bones['names'])} clips / {bones['ref'].shape[0]} frames; "
          f"ha-exec: {len(ha['names'])} clips / {ha['ref'].shape[0]} frames", flush=True)

    nb = a.limit_bones or len(bones["names"])
    b_names = bones["names"][:nb]
    b_lens = bones["lengths"][:nb]
    if nb != len(bones["names"]):
        b_rows = sum(b_lens)
        b_ref = bones["ref"][:b_rows]
    else:
        b_ref = bones["ref"]

    nh = a.limit_ha or len(ha["names"])
    h_names = ha["names"][:nh]
    h_lens = ha["lengths"][:nh]
    h_starts = ha["starts"][:nh]
    x = max(int(a.x20), 1)
    if any("__HA_CMDr" in n for n in h_names):
        raise SystemExit("[wbc-lib] --ha-exec 看起来已经是 x20 展开版；请传 packed.pt（699 条）")
    # 每份 HA clip 的行切片（索引结构不变，只重复）
    ha_rows = [ha["ref"][h_starts[j]: h_starts[j] + h_lens[j]] for j in range(nh)]
    ha_frames = int(sum(h_lens))

    # ---- 组装：bones 原样 + HA 段 round-major（r 外层、clip 内层，与旧 x20 库同构）----
    out_names: list[str] = list(b_names)
    out_lens: list[int] = list(b_lens)
    ha_names: list[str] = []
    for k in range(x):
        for j, name in enumerate(h_names):
            ha_names.append(f"{name}__HA_CMDr{k}")
            out_lens.append(h_lens[j])
    out_names.extend(ha_names)

    total = int(sum(out_lens))
    out = torch.empty((total, REF_DIM), dtype=torch.float16)
    out[: b_ref.shape[0]].copy_(b_ref.to(torch.float16))
    row = b_ref.shape[0]
    for k in range(x):
        for j in range(nh):
            r = ha_rows[j]
            out[row: row + r.shape[0]].copy_(r)
            row += r.shape[0]
    assert row == total, (row, total)
    del b_ref, ha_rows
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    starts = _starts(out_lens)
    print(f"[wbc-lib] assembled: clips={len(out_names)} frames={total} "
          f"({total / 50.0 / 3600.0:.1f} h) in {time.time() - t0:.1f}s", flush=True)

    # ---- 断言：与旧 x20 库索引结构同构（名字/长度/起点逐项一致）------------------
    structure_check = None
    if a.verify_against and Path(a.verify_against).is_file() and not (
            a.limit_bones or a.limit_ha):
        v = torch.load(a.verify_against, map_location="cpu", weights_only=False, mmap=True)
        v_names = list(v["names"])
        v_lens = [int(z) for z in v["lengths"]]
        same_names = v_names == out_names
        same_lens = v_lens == out_lens
        structure_check = {
            "ref_lib": str(a.verify_against),
            "names_identical": bool(same_names),
            "lengths_identical": bool(same_lens),
            "num_clips_identical": len(v_names) == len(out_names),
        }
        if not (same_names and same_lens):
            bad = next((i for i, (p, q) in enumerate(zip(out_names, v_names)) if p != q), None)
            raise SystemExit(f"[wbc-lib] FATAL: 与旧 x20 库结构不同构 (first diff {bad}): "
                             f"{out_names[bad] if bad is not None else 'len'}")
        # 旧库的 HA 段（ha_cmd）vs 新库 HA 段：q 应该差 ~0.39 rad，root 相同
        v_ref = v["ref"]
        b_frames = int(sum(b_lens))
        dq = (out[b_frames:].float()[:, 13:42] - v_ref[b_frames:].float()[:, 13:42]).abs()
        droot = (out[b_frames:, 0:7].float() - v_ref[b_frames:, 0:7].float()).abs()
        structure_check["ha_q_abs_mean_vs_ha_cmd"] = float(dq.mean())
        structure_check["ha_root_abs_max_vs_ha_cmd"] = float(droot.max())
        # bones 段必须逐位相同
        same_bones = bool(torch.equal(out[:b_frames], v_ref[:b_frames].to(torch.float16)))
        structure_check["bones_bitwise_identical"] = same_bones
        if not same_bones:
            raise SystemExit("[wbc-lib] FATAL: bones 段与旧库不一致")
        print(f"[wbc-lib] structure identical to {Path(a.verify_against).name}; "
              f"HA mean|Δq| vs ha_cmd = {structure_check['ha_q_abs_mean_vs_ha_cmd']:.4f} rad, "
              f"bones bitwise identical = True", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ref": out, "names": out_names, "starts": starts, "lengths": out_lens}, a.out)
    print(f"[wbc-lib] wrote {a.out} ({a.out.stat().st_size / 1e9:.2f} GB) "
          f"in {time.time() - t0:.1f}s")

    frames_bones = int(sum(b_lens))
    frames_ha = int(sum(h_lens) * x)
    meta = {
        "packed_ref": a.out.name,
        "ref_layout": "canonical",
        "num_clips": len(out_names),
        "num_frames": total,
        "hours_50hz": round(total / 50.0 / 3600.0, 3),
        "src_fps": 50.0,
        "bones": {"src": str(a.bones), "clips": nb, "frames": frames_bones},
        "ha_exec": {"src": str(a.ha_exec), "clips": nh, "frames_per_copy": int(sum(h_lens)),
                    "x20": x, "frames": frames_ha,
                    "name_pattern": "<clip>__HA_CMDr<0..%d>" % (x - 1)},
        "index_structure_vs_ha_cmd_x20": structure_check,
        "note": ("HA 段换成执行参考库：q/qd/root/速度 = 机器人实测执行轨迹；"
                 "token 由 SonicTokenBuilder 从库行在线计算（hold 窗口），无需 sidecar"),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = a.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print("[wbc-lib] " + json.dumps({k: meta[k] for k in ("num_clips", "num_frames")}))
    print(f"[wbc-lib] meta -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
