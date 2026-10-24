"""Training-entry tests: aux head shapes and a CPU Phase B end-to-end run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mosaic_n17.ac import SLOT_DIM, TOTAL_SLOTS
from mosaic_n17.data.slots_h5 import write_slots_h5
from mosaic_n17.train import AuxSlotHead, AuxTrainingConfig, train_phase_b


def _episode_root(tmp_path: Path, *, episodes: int = 1, frames: int = 60) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "demo"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "total_episodes": episodes,
        "total_frames": episodes * frames,
        "chunks_size": 1000,
        "fps": 50,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    rng = np.random.default_rng(0)
    for episode in range(episodes):
        state = rng.normal(scale=0.01, size=(frames, 43)).astype(np.float32)
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, 1))
        motion = (rng.normal(scale=0.01, size=(frames, 64))).astype(np.float32)
        hands = np.zeros((frames, 7), dtype=np.float32)
        table = pa.table(
            {
                "observation.state": pa.FixedSizeListArray.from_arrays(
                    pa.array(state.reshape(-1)), 43
                ),
                "observation.root_orientation": pa.FixedSizeListArray.from_arrays(
                    pa.array(quat.reshape(-1)), 4
                ),
                "action.motion_token": pa.FixedSizeListArray.from_arrays(
                    pa.array(motion.reshape(-1)), 64
                ),
                "teleop.left_hand_joints": pa.FixedSizeListArray.from_arrays(
                    pa.array(hands.reshape(-1)), 7
                ),
                "teleop.right_hand_joints": pa.FixedSizeListArray.from_arrays(
                    pa.array(hands.reshape(-1)), 7
                ),
            }
        )
        pq.write_table(table, root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet")
    return root


def _slots_dir(tmp_path: Path, *, episodes: int = 1, frames: int = 60) -> Path:
    directory = tmp_path / "slots"
    directory.mkdir()
    for episode in range(episodes):
        write_slots_h5(
            directory / f"episode_{episode:06d}.slots.h5",
            episode_index=episode,
            slots=np.zeros((frames, TOTAL_SLOTS, SLOT_DIM), dtype=np.float32),
            meta={
                "phase_a_sha256": "a" * 64,
                "dense_dim": 1408,
                "video_key": "ego_view",
                "fps": 50,
                "stride": 1,
            },
        )
    return directory


def test_phase_b_delta_weighted_loss_uses_per_dimension_weights() -> None:
    from mosaic_n17.train.train_phase_b import phase_b_loss

    predicted = torch.zeros(2, TOTAL_SLOTS, SLOT_DIM)
    target = torch.full((2, TOTAL_SLOTS, SLOT_DIM), 0.1)
    target[:, 6:] = 0.5  # B/E/S move; C is static
    weight = torch.zeros(TOTAL_SLOTS, SLOT_DIM)
    weight[6:] = 1.0
    equal = phase_b_loss(predicted, target, mode="equal")
    weighted = phase_b_loss(predicted, target, mode="delta_weighted", dim_weight=weight)
    assert weighted > equal
    uniform = torch.ones(TOTAL_SLOTS, SLOT_DIM)
    assert torch.allclose(
        phase_b_loss(predicted, target, mode="delta_weighted", dim_weight=uniform), equal
    )
    with pytest.raises(ValueError, match="dim_weight"):
        phase_b_loss(predicted, target, mode="delta_weighted")


def test_ranking_margin_loss_prefers_recorded_action() -> None:
    from mosaic_n17.train.train_phase_b import ranking_margin_loss

    recorded = torch.tensor([0.1, 0.2])
    wrong = torch.tensor([[0.2, 0.3], [0.1, 0.1]])
    expected = torch.relu(recorded.unsqueeze(1) - wrong).mean()
    assert torch.allclose(ranking_margin_loss(recorded, wrong), expected)
    # A margin penalises even a tiny win.
    assert ranking_margin_loss(recorded, wrong, margin=0.05) > ranking_margin_loss(recorded, wrong)
    with pytest.raises(ValueError):
        ranking_margin_loss(recorded, wrong, margin=-1.0)
    with pytest.raises(ValueError):
        ranking_margin_loss(recorded, wrong[None])


def test_phase_b_train_roundtrip_on_cpu(tmp_path) -> None:
    root = _episode_root(tmp_path)
    slots = _slots_dir(tmp_path)
    summary = train_phase_b(
        dataset_root=root,
        slots_dir=slots,
        output_path=tmp_path / "phase_b.pt",
        phase_a_sha256="a" * 64,
        max_steps=2,
        batch_size=2,
        device="cpu",
    )
    assert summary["steps"] == 2
    assert np.isfinite(summary["final_loss"])
    assert (tmp_path / "phase_b.pt").exists()
    assert (tmp_path / "phase_b.pt.json").exists()


def test_phase_b_rejects_missing_slots(tmp_path) -> None:
    root = _episode_root(tmp_path)
    slots = _slots_dir(tmp_path)
    (slots / "episode_000000.slots.h5").unlink()
    with pytest.raises(FileNotFoundError):
        train_phase_b(
            dataset_root=root,
            slots_dir=slots,
            output_path=tmp_path / "phase_b.pt",
            phase_a_sha256="a" * 64,
            max_steps=1,
        )


def test_aux_head_shapes_and_rejections() -> None:
    head = AuxSlotHead(token_dim=32, hidden_dim=64, depth=1, num_heads=4)
    features = torch.randn(2, 10, 32)
    actions = torch.zeros(2, 40, 78)
    output = head(features, actions)
    assert output.shape == (2, TOTAL_SLOTS, SLOT_DIM)
    with pytest.raises(ValueError, match="H=40"):
        head(features, torch.zeros(2, 39, 78))
    with pytest.raises(ValueError, match="78"):
        head(features, torch.zeros(2, 40, 64))
    with pytest.raises(ValueError, match="batch size"):
        head(torch.randn(3, 10, 32), actions)
    with pytest.raises(ValueError, match="horizon must be"):
        AuxSlotHead(horizon=20)


def test_aux_config_rejects_unexecuted_targets(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"\[1,20\]"):
        AuxTrainingConfig(
            checkpoint_path="ckpt",
            dataset_root="data",
            slots_dir="slots",
            output_path="out.pt",
            phase_a_sha256="c" * 64,
            supervision_step=40,
        )
    with pytest.raises(ValueError, match="phase_a_sha256"):
        AuxTrainingConfig(
            checkpoint_path="ckpt",
            dataset_root="data",
            slots_dir="slots",
            output_path="out.pt",
            phase_a_sha256="short",
        )


def test_action_chunk_from_step_assembles_78d() -> None:
    from types import SimpleNamespace

    from mosaic_n17.train.train_world_model_vlm import action_chunk_from_step

    step = SimpleNamespace(
        actions={
            "motion_token": np.zeros((40, 64), dtype=np.float32),
            "left_hand_joints": np.zeros((40, 7), dtype=np.float32),
            "right_hand_joints": np.zeros((40, 7), dtype=np.float32),
        }
    )
    chunk = action_chunk_from_step(step)
    assert chunk.shape == (40, 78) and chunk.dtype == np.float64
    bad = SimpleNamespace(actions={"motion_token": np.zeros((40, 64), dtype=np.float32)})
    with pytest.raises(ValueError, match="missing action groups"):
        action_chunk_from_step(bad)


def test_aux_train_fails_closed_without_processor(tmp_path) -> None:
    from mosaic_n17.train import AuxTrainingConfig, train_aux

    with pytest.raises(FileNotFoundError, match="processor_config.json"):
        train_aux(
            AuxTrainingConfig(
                checkpoint_path=str(tmp_path / "missing_ckpt"),
                dataset_root="unused",
                slots_dir="unused",
                output_path=str(tmp_path / "out.pt"),
                phase_a_sha256="d" * 64,
                max_steps=1,
                device="cpu",
            )
        )


def test_phase_b_multi_source_split_writes_diagnostics(tmp_path) -> None:
    from mosaic_n17.train.train_phase_b import SOURCE_MANIFEST_SCHEMA, load_source_manifest, train

    entries = []
    for name in ("alpha", "beta"):
        root = _episode_root(tmp_path / name, episodes=2)
        slots = _slots_dir(tmp_path / name, episodes=2)
        entries.append(
            {
                "dataset_root": str(root),
                "slots_dir": str(slots),
                "train_episodes": [0],
                "val_episodes": [1],
            }
        )
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"schema": SOURCE_MANIFEST_SCHEMA, "sources": entries}))
    sources = load_source_manifest(manifest)
    assert [source.train_episodes for source in sources] == [(0,), (0,)]
    output = tmp_path / "phase_b.pt"
    summary = train(
        sources=sources,
        output_path=output,
        phase_a_sha256="a" * 64,
        max_steps=2,
        batch_size=2,
        device="cpu",
        eval_stride=3,
    )
    assert summary["train_episodes"] == 2
    assert summary["val_episodes"] == 2
    report = json.loads((tmp_path / "phase_b.diagnostics.json").read_text())
    assert set(report["splits"]) == {"train", "val"}
    assert report["splits"]["val"]["rows"] > 0
    assert "overall" in report["splits"]["val"]


def test_phase_b_stride2_slots_and_prefix_only(tmp_path) -> None:
    from mosaic_n17.train.train_phase_b import train

    root = _episode_root(tmp_path, episodes=1, frames=120)
    slots = tmp_path / "slots2"
    slots.mkdir()
    write_slots_h5(
        slots / "episode_000000.slots.h5",
        episode_index=0,
        slots=np.zeros((60, TOTAL_SLOTS, SLOT_DIM), dtype=np.float32),
        meta={
            "phase_a_sha256": "a" * 64,
            "dense_dim": 1408,
            "video_key": "ego_view",
            "fps": 50,
            "stride": 2,
        },
    )
    for horizon in (20, 40):
        summary = train(
            dataset_root=root,
            slots_dir=slots,
            output_path=tmp_path / f"phase_b_{horizon}.pt",
            phase_a_sha256="a" * 64,
            max_steps=2,
            batch_size=2,
            device="cpu",
            action_horizon=horizon,
        )
        assert summary["steps"] == 2
        assert (tmp_path / f"phase_b_{horizon}.pt").exists()
