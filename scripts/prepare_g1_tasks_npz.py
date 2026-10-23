#!/usr/bin/env python3
"""Convert the G1 whole-body task recordings into ``MotionLibrary`` npz files.

Input layout (one directory per recording session, LeRobot style)::

    <root>/task_*/<session>/
        data/chunk-000/episode_000000.parquet     # 50 fps rows, one per frame
        videos/chunk-000/observation.images.ego_view/episode_000000.mp4
        meta/tasks.jsonl                          # the instruction for this session
        meta/modality.json                        # slices of observation.state
        meta/info.json

Output: one ``.npz`` per episode in the format ``supermimic``'s ``MotionLibrary``
reads (``joint_pos`` / ``joint_vel`` / ``body_*`` / names / fps), plus the extra
keys this framework needs to line the episode up with the VLM:

===========================  ==================================================
``g1_state_46``              state in ``Gr00tN1d7Processor`` key order
``g1_motion_token_64``       the recorded GR00T motion token (supervision target)
``g1_hand_joints_14``        left+right hand joints
``g1_instruction``           the task description (language conditioning)
``g1_ego_view``              absolute path to the episode's ego_view mp4
``g1_dataset_root``          absolute path to the session directory
``g1_episode_index``         episode index inside the session
``g1_root_height_mode``      how the free-joint height was obtained
===========================  ==================================================

Why the root is reconstructed
-----------------------------
The recordings carry ``observation.root_orientation`` but **no root position and
no root velocity**: the body translation lives inside the motion token and is
decoded by SONIC's own WBC.  SuperMimic's state needs a free-joint pose, so:

* ``root_quat`` is the measured orientation (the robot is upright, so this is
  accurate);
* ``root_xy`` is pinned to the origin (the tasks are in-place manipulation; a
  navigation segment would be misrepresented);
* ``root_z`` is estimated per frame from **foot contact via FK** -- the lowest
  ankle-roll link is placed on the floor -- which tracks squatting and standing;
* root velocities are zero (nothing in the recording implies otherwise).

Use ``--root-height-mode const --const-height 0.76`` for a flat assumption.

Usage::

    python scripts/prepare_g1_tasks_npz.py \
        --root /inspire/.../flowbody/FlowBody/dataset/groot_sonic_vla_ft271_frozen_20261023 \
        --out-dir data/npz/g1_tasks --limit 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diff_vla_wbc.compat import (  # noqa: E402
    G1_JOINT_NAMES,
    ensure_supermimic_importable,
    load_g1_model,
)

#: ``observation.state`` slices of the recording, from ``meta/modality.json``.
#: Note the interleaving: hands sit between the left and right arm blocks.
_STATE_SLICES = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "left_hand": (22, 29),
    "right_arm": (29, 36),
    "right_hand": (36, 43),
}
#: The order ``Gr00tN1d7Processor`` concatenates state keys in.
_PROCESSOR_STATE_ORDER = (
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
)
#: Body joints (29) in ``G1_JOINT_NAMES`` order: left leg, right leg, waist, left
#: arm, right arm -- i.e. every slice except the two hand blocks.
_BODY_JOINT_ORDER = ("left_leg", "right_leg", "waist", "left_arm", "right_arm")
_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
_EGO_VIEW_DIR = Path("videos/chunk-000/observation.images.ego_view")


def _episode_frames(session: Path) -> list[tuple[int, Path, Path, int]]:
    """``(episode_index, parquet, video, length)`` for one session directory."""
    meta = session / "meta"
    info = json.loads((meta / "info.json").read_text())
    lengths: dict[int, int] = {}
    for line in (meta / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            lengths[int(row["episode_index"])] = int(row["length"])
    out = []
    for parquet in sorted((session / "data").rglob("*.parquet")):
        index = int(parquet.stem.split("_")[-1])
        video = session / _EGO_VIEW_DIR / f"episode_{index:06d}.mp4"
        if not video.is_file():
            raise FileNotFoundError(f"{parquet} has no matching ego_view video")
        length = lengths.get(index, 0)
        if length <= 0:
            raise ValueError(f"{session}: episode {index} missing from meta/episodes.jsonl")
        out.append((index, parquet, video, length))
    if not out:
        raise FileNotFoundError(f"no episode parquet under {session}")
    total = sum(e[3] for e in out)
    if total != info.get("total_frames"):
        print(f"[warn] {session}: episodes sum {total} != info.total_frames {info.get('total_frames')}")
    return out


def _instruction(session: Path) -> str:
    tasks = [
        json.loads(line)["task"]
        for line in (session / "meta" / "tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not tasks:
        raise ValueError(f"{session}: meta/tasks.jsonl is empty")
    if len(set(tasks)) != 1:
        # Multi-task sessions would need a per-frame task_index; refuse rather than guess.
        raise ValueError(f"{session}: expected one task description, got {len(set(tasks))}")
    return tasks[0]


def _batch_fk(fk, qpos: np.ndarray, qvel: np.ndarray, batch: int = 1024):
    out = {"pos": [], "quat": [], "lin_vel": [], "ang_vel": []}
    q = torch.as_tensor(qpos, dtype=torch.float32)
    v = torch.as_tensor(qvel, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, q.shape[0], batch):
            r = fk(q[start : start + batch], v[start : start + batch])
            for key in out:
                out[key].append(r[key].cpu())
    return {k: torch.cat(v, 0).numpy().astype(np.float32) for k, v in out.items()}


def convert_episode(
    session: Path,
    episode_index: int,
    parquet: Path,
    video: Path,
    *,
    out_dir: Path,
    fk,
    fps: float,
    root_height_mode: str,
    const_height: float,
    instruction: str,
) -> Path:
    import pandas as pd

    df = pd.read_parquet(parquet)
    frames = len(df)
    if frames < 2:
        raise ValueError(f"{parquet}: only {frames} frames")

    raw = np.stack(df["observation.state"].values).astype(np.float32)
    if raw.shape[-1] != sum(b - a for a, b in _STATE_SLICES.values()):
        raise ValueError(f"{parquet}: unexpected observation.state width {raw.shape[-1]}")

    joint_pos = np.concatenate(
        [raw[:, _STATE_SLICES[key][0] : _STATE_SLICES[key][1]] for key in _BODY_JOINT_ORDER],
        axis=1,
    ).astype(np.float32)
    if joint_pos.shape[1] != len(G1_JOINT_NAMES):
        raise ValueError(f"{parquet}: {joint_pos.shape[1]} body joints, expected {len(G1_JOINT_NAMES)}")

    joint_vel = np.gradient(joint_pos, 1.0 / fps, axis=0).astype(np.float32)

    root_quat = np.stack(df["observation.root_orientation"].values).astype(np.float64)
    root_quat /= np.linalg.norm(root_quat, axis=-1, keepdims=True)
    root_quat = root_quat.astype(np.float32)

    gravity = np.stack(df["observation.projected_gravity"].values).astype(np.float32)
    state_46 = np.concatenate(
        [raw[:, _STATE_SLICES[key][0] : _STATE_SLICES[key][1]] for key in _PROCESSOR_STATE_ORDER]
        + [gravity],
        axis=1,
    ).astype(np.float32)
    if state_46.shape[1] != 46:
        raise ValueError(f"{parquet}: state_46 has width {state_46.shape[1]}, expected 46")

    zero_root = np.zeros((frames, 3), dtype=np.float32)
    qpos0 = np.concatenate([zero_root, root_quat, joint_pos], axis=1)
    qvel0 = np.concatenate([zero_root, zero_root, joint_vel], axis=1)
    if root_height_mode == "foot":
        probe = _batch_fk(fk, qpos0, qvel0)
        names = list(fk.body_names)
        foot_idx = [names.index(n) for n in _FOOT_BODIES]
        # Put the lowest ankle-roll link on the floor.
        root_z = -probe["pos"][:, foot_idx, 2].min(axis=1)
        root_z = np.clip(root_z, 0.3, 1.2).astype(np.float32)
    elif root_height_mode == "const":
        root_z = np.full((frames,), float(const_height), dtype=np.float32)
    else:
        raise ValueError(f"unknown root height mode {root_height_mode!r}")

    root_pos = np.concatenate([np.zeros((frames, 2), np.float32), root_z[:, None]], axis=1)
    qpos = np.concatenate([root_pos, root_quat, joint_pos], axis=1)
    qvel = np.concatenate([zero_root, zero_root, joint_vel], axis=1)
    bodies = _batch_fk(fk, qpos, qvel)

    # FK must reproduce the root we asked for; catches a joint/quaternion order bug.
    names = list(fk.body_names)
    root_mae = float(np.abs(bodies["pos"][:, names.index("pelvis")] - root_pos).mean())
    if root_mae > 5e-3:
        raise ValueError(f"{parquet}: FK root mismatch (MAE {root_mae:.4f} m)")

    token = np.stack(df["action.motion_token"].values).astype(np.float32)
    hands = np.concatenate(
        [
            np.stack(df["teleop.left_hand_joints"].values).astype(np.float32),
            np.stack(df["teleop.right_hand_joints"].values).astype(np.float32),
        ],
        axis=1,
    )
    if token.shape != (frames, 64):
        raise ValueError(f"{parquet}: motion_token {token.shape}, expected ({frames}, 64)")

    out = out_dir / f"{session.parent.name}__{session.name}__ep{episode_index:06d}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        fps=np.float32(fps),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=bodies["pos"],
        body_quat_w=bodies["quat"],
        body_lin_vel_w=bodies["lin_vel"],
        body_ang_vel_w=bodies["ang_vel"],
        joint_names=np.array(G1_JOINT_NAMES),
        body_names=np.array(names),
        source=str(parquet),
        input_fps=np.float32(fps),
        g1_state_46=state_46,
        g1_motion_token_64=token,
        g1_hand_joints_14=hands,
        g1_instruction=np.array(instruction),
        g1_ego_view=np.array(str(video)),
        g1_dataset_root=np.array(str(session)),
        g1_episode_index=np.int32(episode_index),
        g1_root_height_mode=np.array(root_height_mode),
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="dataset root holding task_*/ directories")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "npz" / "g1_tasks")
    ap.add_argument("--tasks", nargs="+", default=None, help="only these task directory names")
    ap.add_argument("--limit", type=int, default=None, help="convert at most N episodes (for a smoke run)")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--root-height-mode", choices=("foot", "const"), default="foot")
    ap.add_argument("--const-height", type=float, default=0.76)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list the episodes that would be converted")
    args = ap.parse_args()

    ensure_supermimic_importable()
    from supermimic.kinematics.g1_fk import G1Kinematics

    fk = G1Kinematics(load_g1_model(), device="cpu")
    sessions = sorted(p for p in args.root.glob("task_*/[0-9]*") if p.is_dir())
    if args.tasks:
        sessions = [s for s in sessions if s.parent.name in set(args.tasks)]
    if not sessions:
        raise SystemExit(f"no task_*/<session> directories under {args.root}")

    jobs = []
    for session in sessions:
        instruction = _instruction(session)
        for index, parquet, video, length in _episode_frames(session):
            jobs.append((session, index, parquet, video, instruction))
    if args.limit is not None:
        jobs = jobs[: args.limit]

    print(f"[info] sessions={len(sessions)} episodes={len(jobs)} out={args.out_dir}")
    if args.dry_run:
        for session, index, parquet, video, _ in jobs:
            print(f"  {session.parent.name}/{session.name} ep{index:06d} <- {parquet.name}")
        return 0

    written = skipped = 0
    for n, (session, index, parquet, video, instruction) in enumerate(jobs, 1):
        out = args.out_dir / f"{session.parent.name}__{session.name}__ep{index:06d}.npz"
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        path = convert_episode(
            session,
            index,
            parquet,
            video,
            out_dir=args.out_dir,
            fk=fk,
            fps=args.fps,
            root_height_mode=args.root_height_mode,
            const_height=args.const_height,
            instruction=instruction,
        )
        written += 1
        if n % 10 == 0 or n == len(jobs):
            print(f"  [{n}/{len(jobs)}] {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"[done] written={written} skipped={skipped} -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
