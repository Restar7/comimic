"""Materialise AC-selected teacher actions as a LeRobot v2 dataset.

N1.7's flow-matching loss is a masked element mean with no per-sample weight
(``gr00t/model/gr00t_n1d7/gr00t_n1d7.py:276-278``), so
``L_demo + lambda * L_teacher`` can only be expressed as *data mixing*.  This
module writes a teacher dataset that is structurally identical to the demo
datasets: every sample is one short episode whose action rows are the chosen
78D candidate and whose observation is the frame/state the candidate was scored
at.  The training launcher then mixes demo and teacher roots by ``mix_ratio``.

Every sample must carry a verified provenance record; an unverified or
placeholder candidate is rejected before any file is written.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np

from ..contracts.provenance import ProvenanceRecord
from ..contracts.sonic78 import SONIC78_V1, Sonic78Spec


TEACHER_SCHEMA = "mosaic.n17.teacher_lerobot.v1"
VIDEO_KEY = "observation.images.ego_view"
MODALITY = {
    "state": {
        "left_leg": {"start": 0, "end": 6},
        "right_leg": {"start": 6, "end": 12},
        "waist": {"start": 12, "end": 15},
        "left_arm": {"start": 15, "end": 22},
        "left_hand": {"start": 22, "end": 29},
        "right_arm": {"start": 29, "end": 36},
        "right_hand": {"start": 36, "end": 43},
        "projected_gravity": {
            "start": 0,
            "end": 3,
            "original_key": "observation.projected_gravity",
        },
        "root_orientation": {
            "start": 0,
            "end": 4,
            "original_key": "observation.root_orientation",
            "rotation_type": "quaternion",
        },
    },
    "action": {
        "motion_token": {"start": 0, "end": 64, "original_key": "action.motion_token"},
        "left_hand_joints": {
            "start": 0,
            "end": 7,
            "original_key": "teleop.left_hand_joints",
        },
        "right_hand_joints": {
            "start": 0,
            "end": 7,
            "original_key": "teleop.right_hand_joints",
        },
    },
    "video": {"ego_view": {"original_key": VIDEO_KEY}},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}
FEATURES = {
    VIDEO_KEY: {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": [43], "names": []},
    "observation.projected_gravity": {"dtype": "float32", "shape": [3], "names": []},
    "observation.root_orientation": {"dtype": "float32", "shape": [4], "names": []},
    "action.motion_token": {"dtype": "float32", "shape": [64], "names": []},
    "teleop.left_hand_joints": {"dtype": "float32", "shape": [7], "names": []},
    "teleop.right_hand_joints": {"dtype": "float32", "shape": [7], "names": []},
    "timestamp": {"dtype": "float32", "shape": [1], "names": ["timestamp"]},
    "frame_index": {"dtype": "int64", "shape": [1], "names": ["frame_index"]},
    "episode_index": {"dtype": "int64", "shape": [1], "names": ["episode_index"]},
    "index": {"dtype": "int64", "shape": [1], "names": ["index"]},
    "task_index": {"dtype": "int64", "shape": [1], "names": ["task_index"]},
}


@dataclass(frozen=True)
class TeacherSample:
    episode_index: int
    instruction: str
    state43: np.ndarray
    projected_gravity: np.ndarray
    root_orientation: np.ndarray
    action: np.ndarray
    frame: np.ndarray
    provenance: ProvenanceRecord

    def validate(self, *, spec: Sonic78Spec = SONIC78_V1) -> None:
        for name, value, width in (
            ("state43", self.state43, 43),
            ("projected_gravity", self.projected_gravity, 3),
            ("root_orientation", self.root_orientation, 4),
        ):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (width,):
                raise ValueError(f"{name} must be [{width}], got {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite")
        action = np.asarray(self.action, dtype=np.float64)
        spec.validate_block(
            action[None, ...],
            provenance=self.provenance.provenance,
            require_provenance=True,
        )
        frame = np.asarray(self.frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame must be uint8 [H,W,3], got {frame.dtype} {frame.shape}")
        if not self.provenance.is_sim_verified:
            raise ValueError(
                "teacher samples must be rollout verified (sim_receipt_sha256 present); "
                "model-screened candidates are not teachable data"
            )
        if not self.instruction:
            raise ValueError("instruction must be non-empty")


def _fixed_list_column(rows: Sequence[np.ndarray], width: int) -> "object":
    import pyarrow as pa

    flat = np.concatenate([np.asarray(row, dtype=np.float32).reshape(-1) for row in rows])
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), width)


def _episode_table(sample: TeacherSample, *, horizon: int) -> "object":
    import pyarrow as pa

    rows = horizon
    state = np.repeat(np.asarray(sample.state43, dtype=np.float32)[None, :], rows, axis=0)
    gravity = np.repeat(
        np.asarray(sample.projected_gravity, dtype=np.float32)[None, :], rows, axis=0
    )
    quat = np.repeat(np.asarray(sample.root_orientation, dtype=np.float32)[None, :], rows, axis=0)
    action = np.asarray(sample.action, dtype=np.float32)
    columns = {
        "observation.state": _fixed_list_column(state, 43),
        "observation.projected_gravity": _fixed_list_column(gravity, 3),
        "observation.root_orientation": _fixed_list_column(quat, 4),
        "action.motion_token": _fixed_list_column(action[:, :64], 64),
        "teleop.left_hand_joints": _fixed_list_column(action[:, 64:71], 7),
        "teleop.right_hand_joints": _fixed_list_column(action[:, 71:78], 7),
        "timestamp": pa.array([i / 50.0 for i in range(rows)], type=pa.float32()),
        "frame_index": pa.array(list(range(rows)), type=pa.int64()),
        "episode_index": pa.array([sample.episode_index] * rows, type=pa.int64()),
        "index": pa.array(list(range(rows)), type=pa.int64()),
        "task_index": pa.array([0] * rows, type=pa.int64()),
    }
    return pa.table(columns)


def _write_video(path: Path, frame: np.ndarray, *, rows: int, fps: int) -> None:
    """Encode ``rows`` copies of one frame as h264 via the ffmpeg CLI.

    ``torchcodec`` 0.8 ships no video encoder and torchvision's writer needs
    PyAV, so the pinned ffmpeg binary is used directly.  GR00T decodes with
    torchcodec, which requires FFmpeg 4-7; H.264 + yuv420p is the supported path.
    """

    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode teacher videos")
    frames = np.repeat(np.asarray(frame, dtype=np.uint8)[None, ...], rows, axis=0)
    height, width = frames.shape[1:3]
    command = [
        ffmpeg,
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    completed = subprocess.run(command, input=frames.tobytes(), capture_output=True)
    if completed.returncode != 0 or not path.exists():
        raise RuntimeError(
            f"ffmpeg failed to encode {path}: {completed.stderr.decode('utf-8', 'replace')[:400]}"
        )


def write_teacher_dataset(
    root: str | os.PathLike[str],
    samples: Sequence[TeacherSample],
    *,
    horizon: int = SONIC78_V1.horizon,
    fps: int = 50,
    spec: Sonic78Spec = SONIC78_V1,
) -> dict:
    """Write ``samples`` as a LeRobot v2 dataset root; refuses to overwrite."""

    import pyarrow.parquet as pq

    if not samples:
        raise ValueError("no teacher samples")
    if horizon != spec.horizon:
        raise ValueError(f"horizon must be {spec.horizon} for {spec.name}")
    target = Path(root)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    for sample in samples:
        sample.validate(spec=spec)
    (target / "meta").mkdir(parents=True)
    (target / "data" / "chunk-000").mkdir(parents=True)
    (target / "videos" / "chunk-000" / VIDEO_KEY).mkdir(parents=True)

    instructions: list[str] = []
    for sample in samples:
        if sample.instruction not in instructions:
            instructions.append(sample.instruction)

    tasks_lines = [
        json.dumps({"task_index": index, "task": text}, ensure_ascii=False)
        for index, text in enumerate(instructions)
    ]
    (target / "meta" / "tasks.jsonl").write_text("\n".join(tasks_lines) + "\n", encoding="utf-8")
    (target / "meta" / "modality.json").write_text(
        json.dumps(MODALITY, indent=1) + "\n", encoding="utf-8"
    )
    info = {
        "codebase_version": "v2.1",
        "robot_type": "unitree_g1_sonic",
        "total_episodes": len(samples),
        "total_frames": len(samples) * horizon,
        "total_tasks": len(instructions),
        "total_videos": len(samples),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(samples)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": f"videos/chunk-{{episode_chunk:03d}}/{VIDEO_KEY}/episode_{{episode_index:06d}}.mp4",
        "features": FEATURES,
        "mosaic_teacher_schema": TEACHER_SCHEMA,
        "mosaic_teacher_horizon": horizon,
    }
    (target / "meta" / "info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")

    episodes = []
    provenance_lines = []
    for index, sample in enumerate(samples):
        table = _episode_table(sample, horizon=horizon)
        pq.write_table(table, target / "data" / "chunk-000" / f"episode_{index:06d}.parquet")
        _write_video(
            target / "videos" / "chunk-000" / VIDEO_KEY / f"episode_{index:06d}.mp4",
            sample.frame,
            rows=horizon,
            fps=fps,
        )
        episodes.append(
            json.dumps({"episode_index": index, "tasks": [sample.instruction], "length": horizon})
        )
        provenance_lines.append(
            json.dumps(
                {
                    "episode_index": index,
                    "source_episode_index": sample.episode_index,
                    "instruction": sample.instruction,
                    **sample.provenance.__dict__,
                },
                ensure_ascii=False,
            )
        )
    (target / "meta" / "episodes.jsonl").write_text("\n".join(episodes) + "\n", encoding="utf-8")
    (target / "provenance.jsonl").write_text("\n".join(provenance_lines) + "\n", encoding="utf-8")
    return {
        "schema": TEACHER_SCHEMA,
        "root": str(target),
        "episodes": len(samples),
        "frames": len(samples) * horizon,
        "tasks": instructions,
    }


__all__ = [
    "FEATURES",
    "MODALITY",
    "TEACHER_SCHEMA",
    "TeacherSample",
    "VIDEO_KEY",
    "write_teacher_dataset",
]
