"""Post-training tests: bank lifecycle, teacher join and launcher config (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from mosaic_n17.contracts import ProvenanceRecord
from mosaic_n17.posttrain import (
    PosttrainConfig,
    SourceRow,
    TeacherBank,
    build_command,
    materialize_samples,
    write_config,
)
from mosaic_n17.posttrain.teacher_bank import PROVISIONAL


def _candidates(value: float = 0.1, K: int = 2) -> np.ndarray:
    candidates = np.zeros((K, 40, 78))
    candidates[..., :64] = value
    return candidates


def _bank() -> TeacherBank:
    bank = TeacherBank()
    bank.add("s0", _candidates(), scores=np.array([0.1, 0.9]))
    return bank


def test_bank_requires_verification_receipt() -> None:
    bank = _bank()
    assert bank.eligible_ids() == []
    with pytest.raises(ValueError, match="not verified"):
        bank.get("s0")
    with pytest.raises(ValueError, match="receipt_sha256"):
        bank.verify("s0", "short")
    bank.verify("s0", "a" * 64)
    assert bank.eligible_ids() == ["s0"]


def test_bank_protected_and_generation_semantics() -> None:
    bank = TeacherBank()
    bank.add("anchor", _candidates(), status=PROVISIONAL, protected=True)
    bank.verify("anchor", "b" * 64)
    assert bank.eligible_ids() == [], "protected anchors are never teachable"
    bank.add("s1", _candidates())
    bank.verify("s1", "c" * 64)
    assert bank.eligible_ids() == ["s1"]
    bank.advance_generation()
    with pytest.raises(ValueError, match="generation mismatch"):
        bank.get("s1")


def test_bank_roundtrip_and_digest_guard(tmp_path) -> None:
    bank = _bank()
    bank.verify("s0", "d" * 64)
    path = tmp_path / "bank.npz"
    digest = bank.save(path)
    assert len(digest) == 64
    restored = TeacherBank.load(path)
    assert restored.eligible_ids() == ["s0"]
    slot = restored.get("s0")
    np.testing.assert_allclose(slot.candidates, bank.get("s0").candidates)
    from mosaic_n17.contracts import Sonic78Spec

    low_latency = Sonic78Spec(name="SONIC78_LOW_LATENCY", horizon=20, execute_steps=10)
    with pytest.raises(ValueError, match="contract digest"):
        TeacherBank.load(path, spec=low_latency)


def _source(index: int) -> SourceRow:
    return SourceRow(
        dataset_root="/data/demo",
        episode_index=index,
        tick=10,
        instruction="Place the red cup into the drawer.",
        state43=np.zeros(43),
        projected_gravity=np.array([0.0, 0.0, -1.0]),
        root_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
    )


def test_materialize_joins_best_verified_candidate() -> None:
    bank = _bank()
    bank.verify("s0", "e" * 64)
    samples = materialize_samples(
        bank,
        {"s0": _source(3)},
        generator_sha256="1" * 64,
        processor_sha256="2" * 64,
        stats_sha256="3" * 64,
    )
    assert len(samples) == 1
    sample = samples[0]
    np.testing.assert_allclose(sample.action, _candidates()[1])
    assert sample.provenance.is_sim_verified
    assert sample.instruction.startswith("Place the red cup")


def test_materialize_rejects_missing_source_and_empty_bank() -> None:
    bank = _bank()
    bank.verify("s0", "e" * 64)
    with pytest.raises(KeyError, match="no source observation"):
        materialize_samples(
            bank,
            {"other": _source(3)},
            generator_sha256="1" * 64,
            processor_sha256="2" * 64,
            stats_sha256="3" * 64,
        )
    empty = TeacherBank()
    with pytest.raises(ValueError, match="no source rows"):
        materialize_samples(
            empty, {}, generator_sha256="1" * 64, processor_sha256="2" * 64, stats_sha256="3" * 64
        )


def test_materialize_rejects_masked_best_candidate() -> None:
    bank = TeacherBank()
    candidates = _candidates()
    mask = np.ones(candidates.shape, dtype=bool)
    mask[1, 0, 64] = False
    bank.add("s0", candidates, mask=mask, scores=np.array([0.1, 0.9]))
    bank.verify("s0", "f" * 64)
    with pytest.raises(ValueError, match="masked channels"):
        materialize_samples(
            bank,
            {"s0": _source(0)},
            generator_sha256="1" * 64,
            processor_sha256="2" * 64,
            stats_sha256="3" * 64,
        )


def test_posttrain_config_and_command(tmp_path) -> None:
    config = PosttrainConfig(
        base_model_path="nvidia/GR00T-N1.7-3B",
        demo_roots=("/data/demo_a", "/data/demo_b"),
        teacher_roots=("/data/teacher",),
        output_dir="/runs/post",
        teacher_mix_ratio=0.25,
    )
    path = write_config(tmp_path / "post.yaml", config)
    text = path.read_text()
    assert "mix_ratio: 0.25" in text and "start_from_checkpoint" in text
    command = build_command(path)
    assert command[-2:] == ["--load-config-path", str(path)]
    with pytest.raises(FileExistsError):
        write_config(path, config)
    with pytest.raises(ValueError, match="either full tune_llm"):
        PosttrainConfig(
            base_model_path="m",
            demo_roots=("a",),
            teacher_roots=("b",),
            output_dir="o",
            tune_llm=True,
            tune_top_llm_layers=2,
        )


def test_read_source_row_from_written_teacher_dataset(tmp_path) -> None:
    pytest.importorskip("gr00t")
    from gr00t.data.stats import generate_stats

    from mosaic_n17.data.teacher_lerobot import TeacherSample, write_teacher_dataset
    from mosaic_n17.posttrain import read_source_row

    action = np.zeros((40, 78))
    action[:, :64] = 0.05
    record = ProvenanceRecord(
        candidate_id="c0",
        provenance="selected_teacher",
        generator_sha256="1" * 64,
        processor_sha256="2" * 64,
        stats_sha256="3" * 64,
        sim_receipt_sha256="4" * 64,
    )
    root = tmp_path / "teacher"
    write_teacher_dataset(
        root,
        [
            TeacherSample(
                episode_index=0,
                instruction="Navigate to the table ahead.",
                state43=np.arange(43, dtype=np.float64) / 100.0,
                projected_gravity=np.array([0.0, 0.0, -1.0]),
                root_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                action=action,
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                provenance=record,
            )
        ],
    )
    generate_stats(root)
    row = read_source_row(root, 0, 5)
    np.testing.assert_allclose(row.state43, np.arange(43) / 100.0)
    assert row.instruction == "Navigate to the table ahead."
    assert row.frame.shape == (480, 640, 3)
