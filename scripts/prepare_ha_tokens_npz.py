#!/usr/bin/env python3
"""Convert HumanoidArena **raw** SONIC episodes into the ft318 npz our trainer eats.

The LeRobot release ``HumanoidArenaV3.1_SONIC40`` only ships a 40-D reference-pose
action plus a 2-D binary hand switch, so it cannot supervise the 64+14 head.  The raw
release (``Twang2026/HumanoidArena_raw``) keeps the full SONIC recording, which is
exactly what we need.  Field mapping -- every target key matches the ft318 npz that
``scripts/prepare_g1_tasks_npz.py`` writes, so ``--recorded-tokens`` and the motion
library consumer accept both interchangeably:

===================  ===================================  ================================
ft318 key            HumanoidArena raw field               note
===================  ===================================  ================================
``joint_pos``        ``robot_qpos_before_decimation``     measured, canonical 29 order
``joint_vel``        ``robot_qvel_before_decimation``     measured (no gradient diff)
``body_*``           FK of (root, joint_pos)              same ``G1Kinematics`` as ft318
``g1_motion_token_64`` ``encoder_latent``                 bit-exact SONIC encoder output
``g1_hand_joints_14``  ``hand_action_left/right``         left 7 then right 7
``g1_state_46``      joints29 + hands14 + gravity3        processor order
root pose            ``robot_root_position/orientation``  measured, xy kept (navigation!)
``g1_instruction``   task table (see ``TASK_INSTRUCTIONS``)
===================  ===================================  ================================

Two differences from the ft318 conversion, both because the raw recording is richer:

* root ``xy`` is **kept** (ft318 pinned it to the origin; the arena has navigation
  tasks where the root genuinely moves);
* root ``z`` is **measured** (ft318 estimated it from foot contact through FK).

Usage::

    python scripts/prepare_ha_tokens_npz.py --limit 2
    python scripts/prepare_ha_tokens_npz.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import joint_contract as JC  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "ha_raw"
OUT = ROOT / "data" / "npz" / "ha_tokens"

TASK_INSTRUCTIONS = {
    "HOI_double_desk": "Pick up the object from the double desk and place it into the basket.",
    "HOI_grap_cup": "Grasp the cup on the table and lift it up.",
    "HOI_pp_box": "Pick up the box and place it at the target location.",
    "HSI_boxing": "Throw punches at the target following the reference motion.",
    "HSI_open_door": "Walk to the door, open it, and step through.",
    "HSI_sit_sofa": "Walk to the sofa and sit down.",
    "HSI_vision_navi": "Navigate to the goal location shown in the instruction.",
}

_FK_STATE: dict = {}


def _fk():
    """Lazily build the G1 FK helper (one per process)."""
    if "fk" not in _FK_STATE:
        from diff_vla_wbc.compat import (
            G1_JOINT_NAMES,
            ensure_supermimic_importable,
            load_g1_model,
        )

        ensure_supermimic_importable()
        from supermimic.kinematics.g1_fk import G1Kinematics

        _FK_STATE["names"] = G1_JOINT_NAMES
        _FK_STATE["fk"] = G1Kinematics(load_g1_model(), device="cpu")
    return _FK_STATE["fk"], _FK_STATE["names"]


def _batch_fk(fk, qpos: np.ndarray, qvel: np.ndarray, batch: int = 1024):
    import torch

    out = {"pos": [], "quat": [], "lin_vel": [], "ang_vel": []}
    q = torch.as_tensor(qpos, dtype=torch.float32)
    v = torch.as_tensor(qvel, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, q.shape[0], batch):
            r = fk(q[start : start + batch], v[start : start + batch])
            for key in out:
                out[key].append(r[key].cpu())
    return {k: torch.cat(v2, 0).numpy().astype(np.float32) for k, v2 in out.items()}


def _quat_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """``(T,4)`` wxyz -> ``(T,3,3)`` rotation matrices."""
    q = quat_wxyz.astype(np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-9)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - z * w)
    m[:, 0, 2] = 2 * (x * z + y * w)
    m[:, 1, 0] = 2 * (x * y + z * w)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - x * w)
    m[:, 2, 0] = 2 * (x * z - y * w)
    m[:, 2, 1] = 2 * (y * z + x * w)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def _gravity_body(quat_wxyz: np.ndarray) -> np.ndarray:
    """Projected gravity in the body frame -- the GR00T ``projected_gravity`` term."""
    m = _quat_to_mat(quat_wxyz)
    g_world = np.array([0.0, 0.0, -1.0])
    return np.einsum("tji,j->ti", m, g_world).astype(np.float32)


def convert_one(raw_path: Path, out_dir: Path, task: str, op: str, index: int,
                overwrite: bool) -> tuple[Path, str]:
    out = out_dir / f"{task}__{op}__ep{index:06d}.npz"
    if out.exists() and not overwrite:
        return out, "skip"

    with np.load(raw_path, allow_pickle=False) as z:
        # HA records joints in its v3.1 order (interleaved left/right); `joint_contract`
        # names the two orders explicitly.  Everything that touches FK/physics must use
        # STANDARD_29, while `state_46` must stay in HA_V31_29 because that is what the
        # deployed server feeds the VLA.
        joint_pos_ha = np.asarray(z["robot_qpos_before_decimation"], dtype=np.float32)
        joint_vel = np.asarray(z["robot_qvel_before_decimation"], dtype=np.float32)
        root_pos = np.asarray(z["robot_root_position"], dtype=np.float32)
        root_quat = np.asarray(z["robot_root_orientation"], dtype=np.float32)
        token = np.asarray(z["encoder_latent"], dtype=np.float32)
        hands = np.concatenate(
            [
                np.asarray(z["hand_action_left"], dtype=np.float32),
                np.asarray(z["hand_action_right"], dtype=np.float32),
            ],
            axis=1,
        )
        n = int(z["num_frames"])

    fk, names = _fk()
    if tuple(names) != JC.STANDARD_29:
        raise RuntimeError(
            f"FK joint order != joint_contract.STANDARD_29\n  got: {list(names)[:6]}\n"
            f"  fix: joint_contract.STANDARD_29 / scripts/prepare_ha_tokens_npz.py"
        )
    if joint_pos_ha.shape != (n, len(names)):
        raise ValueError(f"{raw_path}: joint_pos {joint_pos_ha.shape}, expected ({n},{len(names)})")
    joint_pos = JC.to_standard(joint_pos_ha, src="ha_v31")   # FK / physics order
    if token.shape != (n, 64):
        raise ValueError(f"{raw_path}: encoder_latent {token.shape}, expected ({n},64)")
    if hands.shape != (n, 14):
        raise ValueError(f"{raw_path}: hands {hands.shape}, expected ({n},14)")
    if root_quat.shape != (n, 4) or root_pos.shape != (n, 3):
        raise ValueError(f"{raw_path}: root {root_pos.shape}/{root_quat.shape}")

    gravity = _gravity_body(root_quat)
    # state_46.joints29 stays in HA order on purpose: scripts/ha_vla_latent64_server.py
    # passes the sim's vla_state (HA order) straight through, so train == deploy.
    state_46 = np.concatenate([joint_pos_ha, hands, gravity], axis=1).astype(np.float32)
    if state_46.shape[1] != 46:
        raise ValueError(f"{raw_path}: state_46 width {state_46.shape[1]}")

    zero = np.zeros((n, 3), dtype=np.float32)
    qpos = np.concatenate([root_pos, root_quat, joint_pos], axis=1).astype(np.float64)
    qvel = np.concatenate([zero, zero, joint_vel], axis=1).astype(np.float32)
    bodies = _batch_fk(fk, qpos, qvel)

    # FK must reproduce the sampled root, otherwise joint order or quat layout is wrong.
    pelvis = list(names).index("pelvis") if "pelvis" in names else 0
    root_mae = float(np.abs(bodies["pos"][:, pelvis] - root_pos).mean())
    if root_mae > 5e-3:
        raise ValueError(f"{raw_path}: FK root mismatch MAE {root_mae:.4f} m")

    video = raw_path.parent / "videos" / f"{raw_path.stem}_front_rgb.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        fps=np.float32(50.0),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=bodies["pos"],
        body_quat_w=bodies["quat"],
        body_lin_vel_w=bodies["lin_vel"],
        body_ang_vel_w=bodies["ang_vel"],
        joint_names=np.array(names),
        body_names=np.array(list(fk.body_names)),
        source=str(raw_path),
        input_fps=np.float32(50.0),
        g1_state_46=state_46,
        g1_motion_token_64=token,
        g1_hand_joints_14=hands,
        g1_instruction=np.array(TASK_INSTRUCTIONS.get(task, task)),
        g1_ego_view=np.array(str(video)),
        g1_dataset_root=np.array(str(raw_path.parent)),
        g1_episode_index=np.int32(index),
        g1_root_height_mode=np.array("measured"),
        g1_root_xy_mode=np.array("measured"),
        g1_task=np.array(task),
        g1_operator=np.array(op),
    )
    return out, "written"


def _job(args_tuple) -> tuple[str, str]:
    raw_path, out_dir, task, op, index, overwrite = args_tuple
    try:
        out, status = convert_one(Path(raw_path), Path(out_dir), task, op, index, overwrite)
        return f"{status}\t{out.name}", status
    except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the batch
        return f"error\t{raw_path}\t{exc}", "error"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = []
    for task_dir in sorted(p for p in args.raw.iterdir() if p.is_dir()):
        task = task_dir.name
        if args.tasks and task not in set(args.tasks):
            continue
        for op_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            npzs = sorted(op_dir.glob("*.npz"))
            for index, raw_path in enumerate(npzs):
                jobs.append((str(raw_path), str(args.out_dir), task, op_dir.name, index,
                             args.overwrite))
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        raise SystemExit(f"no raw npz under {args.raw}")
    print(f"[info] raw episodes={len(jobs)} out={args.out_dir}", flush=True)
    if args.dry_run:
        for j in jobs[:20]:
            print("  ", j[0])
        return 0

    counts = {"written": 0, "skip": 0, "error": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_job, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            msg, status = fut.result()
            counts[status] += 1
            if status == "error":
                print(f"[error] {msg}", flush=True)
            if n % 25 == 0 or n == len(jobs):
                print(f"  [{n}/{len(jobs)}] {counts}", flush=True)
    print(f"[done] {counts} -> {args.out_dir}")
    summary = args.out_dir / "convert_summary.json"
    summary.write_text(json.dumps(counts, indent=1) + "\n")
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
