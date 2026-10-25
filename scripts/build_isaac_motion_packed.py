#!/usr/bin/env python3
"""Pack our ``data/npz/<set>`` clips into the **Isaac-side** (trackerlab) motion library.

Why
---
The joint finetune must not mix stacks: the WBC policy and world model come from
``trackerlab-supermimic`` (Isaac Lab / PhysX training), so the reference data, the
observation spec, the reward and the termination must come from that same implementation
(``trackerLab.motion_buffer.motion_lib.CompactMotionLib`` + ``trackerLab.tracker_env.mdp.MDP``),
not from the MuJoCo/mjlab port in ``third_party/SuperMimic-Reproduced``.

``CompactMotionLib`` can read CSVs (``from_dir``) or a packed tensor (``from_packed``).
This script produces the packed form because it is ~1 % of the size and loads in seconds,
while reproducing exactly what ``MotionLib`` computes from CSVs:

* reference layout **canonical** = ``[root_pos(3), root_quat_wxyz(4), root_lin_vel_w(3),
  root_ang_vel_w(3), q(J), qd(J)]`` (world-frame root velocities, as in the Isaac env);
* velocities are the same finite differences ``MotionLib`` uses (``_gradient`` for v and qd,
  central differences of ``so3_log(R_t R_{t-1}^T)`` for world ω) — **not** copied from the npz;
* frames are 1:1 with the npz: 50 Hz in, ``CONTROL_DT = 0.02`` out, so the resampling fast path
  is the identity.  That keeps VLA frame indices (images / instruction / tokens / hand labels
  in the same npz) aligned with motion-library frame indices, which the joint trainer relies on.

Usage
-----
    python3 scripts/build_isaac_motion_packed.py \
        --npz data/npz/mixed_ha_ft318 \
        --out runs/isaac_motion/mixed_ha_ft318.pt

Output: ``<out>`` = ``{"ref", "names", "starts", "lengths"}`` and ``<out>.meta.json``
(``ref_layout: canonical``, clip/frame counts, source hashes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
_TL = ROOT / "trackerlab-supermimic" / "source"
for _p in (_TL / "trackerLab", _TL / "trackerTask"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from trackerLab.kinematics.supermimic.rotations import quat_to_mat, so3_log  # noqa: E402
from trackerLab.motion_buffer.motion_lib.supermimic import CONTROL_DT, _gradient  # noqa: E402

NUM_JOINTS = 29


def _sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def clip_reference(npz_path: Path) -> torch.Tensor:
    """One npz clip → trackerlab canonical reference rows ``(T, 13 + 2*29)``."""
    with np.load(npz_path) as d:
        joint_pos = torch.as_tensor(np.asarray(d["joint_pos"]), dtype=torch.float32)
        body_pos = torch.as_tensor(np.asarray(d["body_pos_w"]), dtype=torch.float32)
        body_quat = torch.as_tensor(np.asarray(d["body_quat_w"]), dtype=torch.float32)
    if joint_pos.shape[-1] != NUM_JOINTS:
        raise SystemExit(f"{npz_path}: expected {NUM_JOINTS} joints, got {joint_pos.shape}")
    # body 0 is the pelvis in every converter in this repo (SONIC / HA / BONES-SEED exports)
    p = body_pos[:, 0]
    quat = body_quat[:, 0]
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    R = quat_to_mat(quat)
    v = _gradient(p, CONTROL_DT)
    qd = _gradient(joint_pos, CONTROL_DT)
    w = torch.zeros_like(p)
    w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
    w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
    w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
    return torch.cat((p, quat, v, w, joint_pos, qd), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, nargs="+", required=True,
                    help="one or more npz directories (or individual .npz files)")
    ap.add_argument("--out", type=Path, required=True, help="packed .pt to write")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--limit", type=int, default=0, help="debug: keep only the first N clips")
    args = ap.parse_args()

    files: list[Path] = []
    for entry in args.npz:
        if entry.is_dir():
            files.extend(sorted(entry.glob("*.npz")))
        elif entry.suffix == ".npz":
            files.append(entry)
        else:
            raise SystemExit(f"not an npz file or directory: {entry}")
    if not files:
        raise SystemExit("no .npz clips found")
    # 本工程永久只走纯 HA 数据；mixed / ft318 不得进入 Isaac 参考运动库。
    banned = [str(f) for f in files if "mixed" in str(f).lower() or "ft318" in str(f).lower()]
    if banned:
        raise SystemExit(
            f"拒绝打包：{len(banned)} 个 clip 来自 mixed/ft318 数据集（例如 {banned[0]}）。"
            "只允许纯 HA（data/npz/ha_tokens）。"
        )
    if args.limit:
        files = files[: args.limit]

    rows: list[torch.Tensor] = []
    names: list[str] = []
    lengths: list[int] = []
    total = 0
    hashes = {}
    for i, f in enumerate(files, 1):
        ref = clip_reference(f)
        if ref.shape[0] < 2:
            raise SystemExit(f"{f}: needs at least two frames")
        rows.append(ref)
        names.append(f.stem)
        lengths.append(int(ref.shape[0]))
        total += int(ref.shape[0])
        if len(hashes) < 64:          # enough to fingerprint the source, not the whole dir
            hashes[f.name] = _sha1(f)
        if i % 100 == 0 or i == len(files):
            print(f"[pack] {i}/{len(files)} clips, {total} frames", flush=True)

    ref = torch.cat(rows, 0)
    starts = [0]
    for n in lengths[:-1]:
        starts.append(starts[-1] + n)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ref": ref.to(dtype), "names": names, "starts": starts, "lengths": lengths}, args.out)
    meta = {
        "ref_layout": "canonical",     # [p, quat_wxyz, v_w, w_w, q, qd] — CompactMotionLib canonical
        "src_fps": 1.0 / CONTROL_DT,   # 50 Hz: MotionLib's resampling is the identity path
        "control_dt": CONTROL_DT,
        "dtype": args.dtype,
        "num_clips": len(names),
        "num_frames": total,
        "npz_dirs": [str(p) for p in args.npz],
        "npz_files_sampled_sha1": hashes,
        "note": "root = body 0 (pelvis); velocities recomputed with trackerlab's own "
                "finite differences so the Isaac world model / MDP see the same convention "
                "as its Isaac-side training data.",
    }
    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    # ``CompactMotionLib.from_packed`` reads a fixed ``meta.json`` next to the packed tensor to
    # learn the ref layout; keep exactly one packed library per directory so that file always
    # describes *this* tensor (otherwise the layout silently falls back to "bones_seed").
    sidecar = args.out.parent / "meta.json"
    sidecar.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[done] {args.out}  clips={len(names)} frames={total} "
          f"({total / 50.0 / 3600.0:.2f} h) dtype={args.dtype}")
    print(f"[done] {meta_path} (+ {sidecar})")


if __name__ == "__main__":
    main()
