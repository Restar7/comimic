"""Data-store tests: slot HDF5 and teacher LeRobot materialisation (CPU only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mosaic_n17.ac import SLOT_DIM, TOTAL_SLOTS
from mosaic_n17.contracts import ProvenanceRecord
from mosaic_n17.data.slots_h5 import read_slots_h5, write_slots_h5
from mosaic_n17.data.teacher_lerobot import TeacherSample, write_teacher_dataset


def _slots(steps: int = 4) -> np.ndarray:
    return np.zeros((steps, TOTAL_SLOTS, SLOT_DIM), dtype=np.float32)


def test_slot_store_roundtrip(tmp_path) -> None:
    path = tmp_path / "episode_000000.slots.h5"
    digest = write_slots_h5(
        path,
        episode_index=0,
        slots=_slots(),
        meta={
            "phase_a_sha256": "a" * 64,
            "dense_dim": 1408,
            "video_key": "ego_view",
            "fps": 50,
            "stride": 1,
        },
    )
    assert len(digest) == 64
    payload = read_slots_h5(path)
    assert payload["slots"].shape == (4, TOTAL_SLOTS, SLOT_DIM)
    assert payload["valid"].all()
    assert payload["attrs"]["phase_a_sha256"] == "a" * 64
    assert payload["attrs"]["schema"] == "mosaic.n17.typed_slots.v1"


def test_slot_store_rejects_bad_shape_and_missing_meta(tmp_path) -> None:
    with pytest.raises(ValueError, match="slots must be"):
        write_slots_h5(
            tmp_path / "bad.h5",
            episode_index=0,
            slots=np.zeros((4, 13, SLOT_DIM)),
            meta={
                "phase_a_sha256": "a" * 64,
                "dense_dim": 1408,
                "video_key": "ego_view",
                "fps": 50,
                "stride": 1,
            },
        )
    with pytest.raises(ValueError, match="missing"):
        write_slots_h5(
            tmp_path / "bad2.h5", episode_index=0, slots=_slots(), meta={"phase_a_sha256": "a" * 64}
        )


def _sample(index: int, *, verified: bool = True) -> TeacherSample:
    action = np.zeros((40, 78))
    action[:, :64] = 0.05
    record = ProvenanceRecord(
        candidate_id=f"c{index}",
        provenance="selected_teacher",
        generator_sha256="1" * 64,
        processor_sha256="2" * 64,
        stats_sha256="3" * 64,
        ac_sha256="4" * 64,
        sim_receipt_sha256="5" * 64 if verified else "",
    )
    return TeacherSample(
        episode_index=index,
        instruction="Navigate to the table ahead.",
        state43=np.zeros(43),
        projected_gravity=np.array([0.0, 0.0, -1.0]),
        root_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        action=action,
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        provenance=record,
    )


def test_teacher_dataset_is_lerobot_readable(tmp_path) -> None:
    root = tmp_path / "teacher"
    summary = write_teacher_dataset(root, [_sample(0), _sample(1)])
    assert summary["episodes"] == 2 and summary["frames"] == 80
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2 and info["fps"] == 50
    assert info["mosaic_teacher_schema"] == "mosaic.n17.teacher_lerobot.v1"
    tasks = [json.loads(line) for line in (root / "meta" / "tasks.jsonl").read_text().splitlines()]
    assert tasks[0]["task"] == "Navigate to the table ahead."
    modality = json.loads((root / "meta" / "modality.json").read_text())
    assert modality["annotation"]["human.task_description"]["original_key"] == "task_index"

    import pyarrow.parquet as pq

    table = pq.read_table(root / "data" / "chunk-000" / "episode_000000.parquet")
    assert table.num_rows == 40
    assert table["action.motion_token"].type.list_size == 64
    assert table["teleop.left_hand_joints"].type.list_size == 7
    assert table["observation.state"].type.list_size == 43
    video = root / "videos" / "chunk-000" / "observation.images.ego_view" / "episode_000000.mp4"
    assert video.exists() and video.stat().st_size > 0


def test_teacher_dataset_rejects_unverified_and_overwrite(tmp_path) -> None:
    root = tmp_path / "teacher"
    with pytest.raises(ValueError, match="rollout verified"):
        write_teacher_dataset(root, [_sample(0, verified=False)])
    assert not root.exists(), "a rejected sample must not leave a partial dataset"
    write_teacher_dataset(root, [_sample(0)])
    with pytest.raises(FileExistsError):
        write_teacher_dataset(root, [_sample(1)])


def test_teacher_sample_rejects_placeholder_action() -> None:
    sample = _sample(0)
    bad_action = np.zeros((40, 78))
    bad_action[:, :64] = 2.0  # beyond 1.25
    with pytest.raises(ValueError, match="1.25"):
        TeacherSample(
            episode_index=sample.episode_index,
            instruction=sample.instruction,
            state43=sample.state43,
            projected_gravity=sample.projected_gravity,
            root_orientation=sample.root_orientation,
            action=bad_action,
            frame=sample.frame,
            provenance=sample.provenance,
        ).validate()
