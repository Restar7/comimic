#!/usr/bin/env python3
"""Convert HumanoidArena raw SONIC episodes into LeRobot session roots for the
**official** GR00T-N1.7 finetune (``examples/finetune.sh --embodiment-tag UNITREE_G1_SONIC``).

The official loader wants, per session root::

    meta/{info,episodes,episodes_stats,tasks,modality,stats,relative_stats}.json*
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.ego_view/episode_000000.mp4

The schema is taken verbatim from the ft318 root that produced our current 20k
checkpoint (``data/ha_session_template/``): the parquet schema, ``modality.json``
and the ``info.json`` feature block are reused byte-for-byte, only the counts and
the per-episode data are new.  Field mapping from the raw npz:

========================================  ==========================================
LeRobot column                            raw field
========================================  ==========================================
``observation.state`` (43)                joints29 + hands14, interleaved
``action.wbc`` (43)                       ``final_full_action``
``observation.eef_state`` (14)            FK wrist yaw links (pos + wxyz quat)
``observation.root_orientation`` (4)      ``robot_root_orientation``
``observation.projected_gravity`` (3)     quat -> body-frame gravity
``observation.cpp_rotation_offset`` (4)   ``anchor_init_base_quat_wxyz``
``observation.init_base_quat`` (4)        ``anchor_init_ref_quat_wxyz``
``teleop.delta_heading``                  ``human_heading_increment``
``action.motion_token`` (64)              ``encoder_latent``
``teleop.smpl_joints`` (72)               ``human_smpl_joints`` (24x3)
``teleop.smpl_pose`` (63)                 ``human_smpl_pose`` (21x3)
``teleop.body_quat_w`` (4)                ``human_body_quat_w``
``teleop.target_body_orientation`` (6)    ``consumed_anchor_rot6d``
``teleop.left/right_hand_joints`` (7)     ``hand_action_left/right``
``teleop.left/right_wrist_joints`` (3)    wrist slice of the 29 joints
``teleop.vr_3pt_position`` (9)            ``human_vr_position``
``teleop.vr_3pt_orientation`` (18)        ``human_vr_orientation`` (12) + 6 zeros
``teleop.stream/planner_*``               zeros (no planner in the arena recording)
========================================  ==========================================

Usage::

    python scripts/prepare_ha_sessions.py --limit 1          # one-episode smoke
    python scripts/prepare_ha_sessions.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
OUT = ROOT / "data" / "ha_sessions"
TEMPLATE = ROOT / "data" / "ha_session_template"

TASK_INSTRUCTIONS = {
    "HOI_double_desk": "Pick up the object from the double desk and place it into the basket.",
    "HOI_grap_cup": "Grasp the cup on the table and lift it up.",
    "HOI_pp_box": "Pick up the box and place it at the target location.",
    "HSI_boxing": "Throw punches at the target following the reference motion.",
    "HSI_open_door": "Walk to the door, open it, and step through.",
    "HSI_sit_sofa": "Walk to the sofa and sit down.",
    "HSI_vision_navi": "Navigate to the goal location shown in the instruction.",
}

_STATE = {}


def _fk():
    if "fk" not in _STATE:
        from diff_vla_wbc.compat import G1_JOINT_NAMES, ensure_supermimic_importable, load_g1_model

        ensure_supermimic_importable()
        from supermimic.kinematics.g1_fk import G1Kinematics

        _STATE["fk"] = G1Kinematics(load_g1_model(), device="cpu")
        _STATE["names"] = G1_JOINT_NAMES
    return _STATE["fk"], _STATE["names"]


def _quat_to_mat(q_wxyz: np.ndarray) -> np.ndarray:
    q = q_wxyz.astype(np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-9)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((q.shape[0], 3, 3))
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


def _gravity_body(q_wxyz: np.ndarray) -> np.ndarray:
    m = _quat_to_mat(q_wxyz)
    return np.einsum("tji,j->ti", m, np.array([0.0, 0.0, -1.0]))


def _columns(raw_path: Path) -> tuple[dict[str, np.ndarray], int]:
    """Build every LeRobot column for one raw episode."""
    import torch

    fk, names = _fk()
    if tuple(names) != JC.STANDARD_29:
        raise RuntimeError(
            f"FK joint order != joint_contract.STANDARD_29\n  got: {list(names)[:6]}"
        )
    with np.load(raw_path, allow_pickle=False) as z:
        # HA records joints in v3.1 (interleaved) order; FK/physics and the wrist slices
        # below are written in STANDARD_29, so convert once, here, and keep it explicit.
        jp_ha = np.asarray(z["robot_qpos_before_decimation"], dtype=np.float64)
        jp = JC.to_standard(jp_ha, src="ha_v31")
        root_pos = np.asarray(z["robot_root_position"], dtype=np.float64)
        root_quat = np.asarray(z["robot_root_orientation"], dtype=np.float64)
        hl = np.asarray(z["hand_action_left"], dtype=np.float64)
        hr = np.asarray(z["hand_action_right"], dtype=np.float64)
        n = int(z["num_frames"])
        token = np.asarray(z["encoder_latent"], dtype=np.float64)
        wbc = np.asarray(z["final_full_action"], dtype=np.float64)
        dhdg = np.asarray(z["human_heading_increment"], dtype=np.float64)[:, 0]
        smpl_joints = np.asarray(z["human_smpl_joints"], dtype=np.float32).reshape(n, -1)
        smpl_pose = np.asarray(z["human_smpl_pose"], dtype=np.float32).reshape(n, -1)
        body_quat = np.asarray(z["human_body_quat_w"], dtype=np.float32)
        tgt_ori = np.asarray(z["consumed_anchor_rot6d"], dtype=np.float32)
        offset = np.asarray(z["anchor_init_base_quat_wxyz"], dtype=np.float64)
        init_q = np.asarray(z["anchor_init_ref_quat_wxyz"], dtype=np.float64)
        vr_pos = np.asarray(z["human_vr_position"], dtype=np.float32)
        vr_ori = np.asarray(z["human_vr_orientation"], dtype=np.float32)
        jvel = np.asarray(z["robot_qvel_before_decimation"], dtype=np.float32)

    if jp.shape != (n, 29):
        raise ValueError(f"{raw_path}: robot_qpos {jp.shape}")
    state = np.concatenate([jp[:, 0:22], hl, jp[:, 22:29], hr], axis=1)

    zero3 = np.zeros((n, 3))
    qpos = np.concatenate([root_pos, root_quat, jp], axis=1).astype(np.float32)
    qvel = np.concatenate([zero3, zero3, jvel], axis=1).astype(np.float32)
    with torch.no_grad():
        r = fk(torch.as_tensor(qpos), torch.as_tensor(qvel))
    bpos = r["pos"].cpu().numpy().astype(np.float64)
    bquat = r["quat"].cpu().numpy().astype(np.float64)
    bnames = list(fk.body_names)
    lw = bnames.index("left_wrist_yaw_link")
    rw = bnames.index("right_wrist_yaw_link")
    root_mae = float(np.abs(bpos[:, bnames.index("pelvis")] - root_pos).mean())
    if root_mae > 5e-3:
        raise ValueError(f"{raw_path}: FK root mismatch MAE {root_mae:.4f} m")
    eef = np.concatenate([bpos[:, lw], bquat[:, lw], bpos[:, rw], bquat[:, rw]], axis=1)

    vr_ori18 = np.concatenate([vr_ori, np.zeros((n, 6), np.float32)], axis=1)

    cols = {
        "observation.state": state,
        "observation.eef_state": eef,
        "action.wbc": wbc,
        "observation.root_orientation": root_quat,
        "observation.projected_gravity": _gravity_body(root_quat),
        # Both anchor quats are stored per frame (T,4), not as episode scalars.
        "observation.cpp_rotation_offset": offset,
        "observation.init_base_quat": init_q,
        "teleop.delta_heading": dhdg,
        "action.motion_token": token,
        "teleop.smpl_joints": smpl_joints,
        "teleop.smpl_pose": smpl_pose,
        "teleop.body_quat_w": body_quat,
        "teleop.target_body_orientation": tgt_ori,
        "teleop.left_hand_joints": hl.astype(np.float32),
        "teleop.right_hand_joints": hr.astype(np.float32),
        "teleop.smpl_frame_index": np.arange(n, dtype=np.int64),
        "teleop.left_wrist_joints": jp[:, 19:22].astype(np.float32),
        "teleop.right_wrist_joints": jp[:, 26:29].astype(np.float32),
        "teleop.stream_mode": np.zeros(n, dtype=np.int32),
        "teleop.planner_mode": np.zeros(n, dtype=np.int32),
        "teleop.planner_movement": np.zeros((n, 3), dtype=np.float32),
        "teleop.planner_facing": np.zeros((n, 3), dtype=np.float32),
        "teleop.planner_speed": np.zeros(n, dtype=np.float32),
        "teleop.planner_height": np.zeros(n, dtype=np.float32),
        "teleop.vr_3pt_position": vr_pos,
        "teleop.vr_3pt_orientation": vr_ori18,
        "timestamp": (np.arange(n, dtype=np.float64) / 50.0).astype(np.float32),
    }
    return cols, n


STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")


def _stats_block(values: np.ndarray) -> dict[str, list[float]]:
    v = values.reshape(values.shape[0], -1).astype(np.float64)
    return {
        "mean": v.mean(0).tolist(),
        "std": v.std(0).tolist(),
        "min": v.min(0).tolist(),
        "max": v.max(0).tolist(),
        "q01": np.percentile(v, 1, axis=0).tolist(),
        "q99": np.percentile(v, 99, axis=0).tolist(),
    }


def _write_parquet(path: Path, cols: dict[str, np.ndarray], n: int, ep: int,
                   index_offset: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if "schema" not in _STATE:
        _STATE["schema"] = pq.read_schema(_STATE["ref_parquet"])
    schema = _STATE["schema"]

    arrays = []
    for field in schema:
        name = field.name
        typ = field.type
        if name == "frame_index":
            col = np.arange(n, dtype=np.int64)
        elif name == "episode_index":
            col = np.full(n, ep, dtype=np.int64)
        elif name == "index":
            col = np.arange(index_offset, index_offset + n, dtype=np.int64)
        elif name == "task_index":
            col = np.zeros(n, dtype=np.int64)
        else:
            col = cols[name]
        if pa.types.is_fixed_size_list(typ):
            width = typ.list_size
            flat = np.asarray(col).reshape(-1).astype(typ.value_type.to_pandas_dtype())
            arrays.append(pa.FixedSizeListArray.from_arrays(pa.array(flat, type=typ.value_type), width))
        else:
            arrays.append(pa.array(np.asarray(col), type=typ))
    table = pa.Table.from_arrays(arrays, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _convert_session(task: str, op: str, raw_dir: Path, out_root: Path, limit: int | None,
                     overwrite: bool, ref_parquet: str, stats_ref: dict) -> str:
    _STATE["ref_parquet"] = ref_parquet
    npzs = sorted(raw_dir.glob("*.npz"))
    if limit is not None:
        npzs = npzs[:limit]
    if not npzs:
        return f"skip\t{task}/{op}: no npz"

    root = out_root / f"{task}__{op}"
    if (root / "meta" / "stats.json").exists() and not overwrite:
        return f"skip\t{root.name}"

    video_dir = root / "videos" / "chunk-000" / "observation.images.ego_view"
    data_dir = root / "data" / "chunk-000"
    video_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    all_values: dict[str, list[np.ndarray]] = {k: [] for k in stats_ref}
    ep_stats: list[dict] = []
    lengths: list[int] = []
    cursor = 0
    for ep, raw_path in enumerate(npzs):
        cols, n = _columns(raw_path)
        _write_parquet(data_dir / f"episode_{ep:06d}.parquet", cols, n, ep, cursor)
        cursor += n
        lengths.append(n)

        src_video = raw_dir / "videos" / f"{raw_path.stem}_front_rgb.mp4"
        dst_video = video_dir / f"episode_{ep:06d}.mp4"
        if dst_video.exists():
            dst_video.unlink()
        if src_video.is_file():
            try:
                os.link(src_video, dst_video)
            except OSError:
                shutil.copy2(src_video, dst_video)
        elif not _STATE.get("allow_missing_video"):
            raise FileNotFoundError(f"{raw_path}: missing video {src_video}")

        block = {}
        for key in stats_ref:
            if key in cols:
                block[key] = _stats_block(cols[key])
                all_values[key].append(np.asarray(cols[key]).reshape(n, -1))
        ep_stats.append({"episode_index": ep, "stats": block})

    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "modality.json").write_text((TEMPLATE / "modality.json").read_text())
    (meta / "relative_stats.json").write_text((TEMPLATE / "relative_stats.json").read_text())

    info = json.loads((TEMPLATE / "info.json").read_text())
    info["total_episodes"] = len(npzs)
    info["total_frames"] = int(sum(lengths))
    info["total_videos"] = len(npzs)
    info["total_tasks"] = 1
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(npzs)}"}
    info["discarded_episode_indices"] = []
    (meta / "info.json").write_text(json.dumps(info, indent=4))

    instruction = TASK_INSTRUCTIONS.get(task, task)
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": instruction}) + "\n")
    with (meta / "episodes.jsonl").open("w") as fh:
        for ep, length in enumerate(lengths):
            fh.write(json.dumps({"episode_index": ep, "tasks": ["demo"], "length": length}) + "\n")
    with (meta / "episodes_stats.jsonl").open("w") as fh:
        for row in ep_stats:
            fh.write(json.dumps(row) + "\n")

    stats = {key: _stats_block(np.concatenate(all_values[key], axis=0))
             for key in stats_ref if all_values[key]}
    stats["__fingerprints__"] = {}
    (meta / "stats.json").write_text(json.dumps(stats))

    return f"written\t{root.name} eps={len(npzs)} frames={sum(lengths)}"


def _job(args) -> str:
    task, op, raw_dir, out_root, limit, overwrite, ref_parquet, stats_ref = args
    try:
        return _convert_session(task, op, Path(raw_dir), Path(out_root), limit, overwrite,
                                ref_parquet, stats_ref)
    except Exception as exc:  # noqa: BLE001
        return f"error\t{task}/{op}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out-root", type=Path, default=OUT)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="episodes per session (smoke)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-missing-video", action="store_true",
                    help="CPU-only smoke: build meta+parquet even if the mp4 is not downloaded yet")
    ap.add_argument("--ref-parquet", type=str,
                    default=os.environ.get("HA_REF_PARQUET", ""),
                    help="a real ft318 episode parquet; its schema is reused verbatim")
    args = ap.parse_args()

    if not args.ref_parquet or not Path(args.ref_parquet).is_file():
        raise SystemExit("--ref-parquet (or $HA_REF_PARQUET) must point at a reference parquet")
    _STATE["allow_missing_video"] = args.allow_missing_video
    stats_ref = list(json.loads((TEMPLATE / "stats.json").read_text()).keys())
    stats_ref = [k for k in stats_ref if k != "__fingerprints__"]

    jobs = []
    for task_dir in sorted(p for p in args.raw.iterdir() if p.is_dir()):
        if args.tasks and task_dir.name not in set(args.tasks):
            continue
        for op_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            jobs.append((task_dir.name, op_dir.name, str(op_dir), str(args.out_root),
                         args.limit, args.overwrite, args.ref_parquet, stats_ref))
    if not jobs:
        raise SystemExit(f"no raw sessions under {args.raw}")
    print(f"[info] sessions={len(jobs)}", flush=True)

    ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_job, j) for j in jobs]
        for fut in as_completed(futs):
            msg = fut.result()
            print(msg, flush=True)
            if msg.startswith("written") or msg.startswith("skip"):
                ok += 1
    roots = sorted(p for p in args.out_root.iterdir() if p.is_dir())
    (args.out_root / "roots.txt").write_text(":".join(str(p) for p in roots) + "\n")
    print(f"[done] {ok}/{len(jobs)} sessions -> {args.out_root} (roots.txt: {len(roots)} roots)")
    return 0 if ok == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
