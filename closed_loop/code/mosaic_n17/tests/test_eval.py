"""Evaluation tooling tests: oracle, calibration, latency, physics receipts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mosaic_n17.eval import (
    brier_score,
    calibration_report,
    expected_calibration_error,
    latency_report,
    oracle_at_n,
    ranking_regret,
    top1_agreement,
    validate_receipt,
    verification_digest,
)


def _candidates(batch: int = 2, K: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(batch, K, 40, 78))


def test_oracle_report_metrics() -> None:
    candidates = _candidates()
    scores = np.array([[0.1, 0.5, 0.2], [0.3, 0.3, 0.6]])
    report = oracle_at_n(candidates, scores)
    assert report.candidates == 6
    assert report.oracle_score_mean == pytest.approx((0.5 + 0.6) / 2)
    assert report.selected_score_mean == pytest.approx((0.1 + 0.3) / 2)
    assert report.oracle_gap_mean == pytest.approx(((0.5 - 0.1) + (0.6 - 0.3)) / 2)
    assert report.pairwise_diversity_mean > 0.0


def test_oracle_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="must be \[B,K\]"):
        oracle_at_n(_candidates(), np.zeros((3, 3)))


def test_ranking_regret_and_top1_agreement() -> None:
    scores = np.array([[0.1, 0.9], [0.8, 0.2]])
    truth = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(ranking_regret(scores, truth), [0.0, 0.0])
    assert top1_agreement(scores, truth) == 1.0
    swapped = truth[:, ::-1]
    assert top1_agreement(scores, swapped) == 0.0


def test_calibration_metrics() -> None:
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(probs, labels, bins=2) == pytest.approx(0.0)
    assert brier_score(probs, labels) == pytest.approx(0.0)
    report = calibration_report(probs, labels)
    assert report.samples == 4 and report.spearman == pytest.approx(1.0)
    with pytest.raises(ValueError, match="binary"):
        expected_calibration_error(probs, np.array([0.0, 0.5, 1.0, 1.0]))


def test_latency_budget() -> None:
    report = latency_report(np.arange(1, 101, dtype=np.float64), budget_ms=250.0)
    assert report.samples == 100
    assert report.p99_ms <= 101.0
    assert report.within_budget is True
    fail = latency_report(np.full(10, 300.0), budget_ms=250.0)
    assert fail.within_budget is False
    with pytest.raises(ValueError, match="budget_ms"):
        latency_report(np.ones(3), budget_ms=0.0)


def _receipt(**overrides) -> dict:
    payload = {
        "schema": "mosaic.n17.physics_receipt.v1",
        "candidate_sha256": "a" * 64,
        "controller_sha256": "b" * 64,
        "simulator_sha256": "c" * 64,
        "start_tick": 100,
        "ticks": 20,
        "status": "applied",
        "contact": True,
    }
    payload.update(overrides)
    return payload


def test_receipt_validation_and_digest() -> None:
    receipt = validate_receipt(_receipt(), expected_candidate_sha256="a" * 64)
    digest = verification_digest("a" * 64, receipt)
    assert len(digest) == 64
    with pytest.raises(ValueError, match="does not match"):
        validate_receipt(_receipt(), expected_candidate_sha256="d" * 64)
    with pytest.raises(ValueError, match="rejected rollout"):
        validate_receipt(_receipt(status="rejected"), expected_candidate_sha256="a" * 64)
    with pytest.raises(ValueError, match="missing"):
        validate_receipt({"schema": "x"}, expected_candidate_sha256="a" * 64)
    with pytest.raises(ValueError, match="at least one tick"):
        validate_receipt(_receipt(ticks=0), expected_candidate_sha256="a" * 64)


def _probe_slots(steps: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    # every branch is a smooth per-dimension random walk (rank >> 2, lag-1 ~1)
    walks = np.cumsum(rng.normal(scale=0.005, size=(steps, 14, 384)), axis=0)
    state = np.cumsum(rng.normal(scale=0.2, size=(steps, 46)), axis=0)
    slots = walks.copy()
    slots[:, 6:9, :46] += state[:, None, :]
    return slots, state


def test_phase_a_probe_passes_on_informative_slots() -> None:
    from mosaic_n17.eval.phase_a_probe import probe_slots

    slots, state = _probe_slots()
    report = probe_slots(slots, state)
    assert report.branches["B"].non_collapsed
    assert report.branches["B"].readable
    assert report.action_corr > 0.2
    assert report.verdict == "pass"


def test_phase_a_probe_fails_on_collapsed_slots() -> None:
    from mosaic_n17.eval.phase_a_probe import probe_slots

    slots = np.zeros((40, 14, 384))
    state = np.zeros((40, 46))
    report = probe_slots(slots, state)
    assert report.verdict == "fail"
    assert not report.branches["C"].non_collapsed
    assert any("collapsed" in reason for reason in report.reasons)


def test_phase_a_probe_rejects_bad_shapes() -> None:
    from mosaic_n17.eval.phase_a_probe import probe_slots

    with pytest.raises(ValueError, match="slots must be"):
        probe_slots(np.zeros((10, 13, 384)), np.zeros((10, 46)))
    slots, state = _probe_slots(20)
    with pytest.raises(ValueError, match="state46"):
        probe_slots(slots, np.zeros((20, 49)))


def _cross_episode_cases(episodes: int = 4, steps: int = 60, *, informative: bool = True) -> list[tuple[str, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(1)
    projection = rng.normal(size=(46, 384))
    cases = []
    for episode in range(episodes):
        state = np.cumsum(rng.normal(scale=0.01, size=(steps, 46)), axis=0)
        if informative:
            slots = np.stack(
                [state @ projection * (0.2 + 0.05 * branch) for branch in range(14)], axis=1
            ) + 0.01 * rng.normal(size=(steps, 14, 384))
        else:
            slots = rng.normal(size=(steps, 14, 384))
        cases.append((f"ep{episode}", slots, state))
    return cases


def test_cross_episode_probe_recovers_shared_readout() -> None:
    from mosaic_n17.eval.phase_a_probe import probe_cross_episode

    report = probe_cross_episode(_cross_episode_cases())
    assert report["overall"] == "pass"
    assert report["branches"]["B"]["readout_r2_mean"] > 0.8
    assert report["branches"]["B"]["readout_r2_min"] > 0.5
    assert report["branches"]["B"]["readout_r2_mean"] > report["branches"]["B"]["shifted_r2_mean"] + 0.1


def test_cross_episode_probe_fails_on_uninformative_slots() -> None:
    from mosaic_n17.eval.phase_a_probe import probe_cross_episode

    report = probe_cross_episode(_cross_episode_cases(informative=False))
    assert report["overall"] == "fail"
    assert not report["branches"]["B"]["readable"]
    assert any("B branch" in reason for reason in report["reasons"])


def test_cross_episode_probe_requires_three_episodes() -> None:
    from mosaic_n17.eval.phase_a_probe import probe_cross_episode

    with pytest.raises(ValueError, match="at least three"):
        probe_cross_episode(_cross_episode_cases()[:2])


class _ActionDrivenModel(torch.nn.Module):
    """Prediction = current + action[0] * weight: the control test for diagnostics."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, current_slots, state46, actions):  # noqa: ANN001
        scale = actions[:, 0, 0].reshape(-1, 1, 1)
        return {"predicted_slots": current_slots + scale * self.weight}


class _CopyCurrentModel(torch.nn.Module):
    def forward(self, current_slots, state46, actions):  # noqa: ANN001
        return {"predicted_slots": current_slots}


class _SyntheticEpisode:
    def __init__(self, length: int, weight: torch.Tensor, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.slots = torch.randn(length, 14, 384, generator=generator).numpy().astype("float32")
        self.state46 = torch.zeros(length, 46).numpy().astype("float32")
        # One 78D command per tick, exactly the LeRobot/T train layout: an episode row's
        # action block is ``actions[t : t + horizon]`` and therefore [40, 78].
        actions = torch.zeros(length, 78)
        # A different command per start tick and per episode so both the shuffled control
        # and the candidate-ranking control are real errors (no cross-episode ties).
        actions[:, 0] = (
            torch.arange(length).float() + 0.5 + seed * (length + 1)
        ) / (4 * length)
        self.actions = actions.numpy().astype("float32")
        for t in range(length - 20):
            self.slots[t + 20] = self.slots[t] + float(actions[t, 0]) * weight.numpy()
        self.valid = np.ones(length, dtype=bool)
        self.source = f"synthetic-{seed}"


def _diagnostics_episodes(count: int = 2, length: int = 120) -> list:
    weight = torch.zeros(14, 384)
    weight[:, 0] = 0.5
    return [_SyntheticEpisode(length, weight, seed) for seed in range(count)]


def test_phase_b_diagnostics_separate_action_use_from_copy_current() -> None:
    from mosaic_n17.eval.phase_b_diagnostics import DiagnosticsConfig, evaluate_phase_b

    episodes = _diagnostics_episodes()
    weight = torch.zeros(14, 384)
    weight[:, 0] = 0.5
    report = evaluate_phase_b(
        _ActionDrivenModel(weight),
        episodes,
        config=DiagnosticsConfig(stride=1, clear_margin=0.0),
        device="cpu",
    )
    assert report["verdict"] == "pass"
    assert report["overall"]["beats_copy_current"]
    assert report["overall"]["shuffled_action_hurts"]
    assert report["overall"]["copy_current_l1"] > report["overall"]["model_l1"] * 10
    assert report["overall"]["shuffled_action_l1"] > report["overall"]["model_l1"]
    assert report["overall"]["action_sensitivity_max"] > 1e-3
    # Candidate ranking: the recorded action must be strictly preferred over the wrong
    # actions, otherwise the AC could not pick a candidate at deployment.
    assert report["overall"]["ranking_top1_accuracy"] == 1.0
    assert report["overall"]["ranking_mean_rank"] == 1.0
    assert report["overall"]["ranking_mrr"] == 1.0
    assert report["overall"]["ranking_clear_fraction"] == 1.0


def test_phase_b_diagnostics_reject_copy_current_model() -> None:
    from mosaic_n17.eval.phase_b_diagnostics import DiagnosticsConfig, evaluate_phase_b

    report = evaluate_phase_b(
        _CopyCurrentModel(),
        _diagnostics_episodes(),
        config=DiagnosticsConfig(stride=1, clear_margin=0.0),
        device="cpu",
    )
    assert report["verdict"] == "fail"
    assert not report["overall"]["beats_copy_current"]
    assert any("copy-current" in reason for reason in report["reasons"])
    # A copy-current model ties every candidate, so it must not score as a selector.
    assert report["overall"]["ranking_top1_accuracy"] == 0.0
    assert report["overall"]["ranking_mean_rank"] > 1.0
    assert report["overall"]["ranking_clear_fraction"] == 0.0


def test_phase_b_diagnostics_single_row_batches_still_rank() -> None:
    from mosaic_n17.eval.phase_b_diagnostics import DiagnosticsConfig, evaluate_phase_b

    weight = torch.zeros(14, 384)
    weight[:, 0] = 0.5
    report = evaluate_phase_b(
        _ActionDrivenModel(weight),
        _diagnostics_episodes(count=1),
        config=DiagnosticsConfig(stride=1, batch_size=1),
        device="cpu",
    )
    assert report["overall"]["ranking_rows"] == 4
    assert report["overall"]["ranking_top1_accuracy"] == 1.0


def test_phase_b_diagnostics_builds_history_for_history_model() -> None:
    from mosaic_n17.ac.phase_b_runtime import PhaseBConfig, Sonic78PhaseB
    from mosaic_n17.eval.phase_b_diagnostics import DiagnosticsConfig, evaluate_phase_b

    model = Sonic78PhaseB(
        PhaseBConfig(control_horizon=20, prefix_only=True, history_slots=1, prev_action_ticks=20)
    )
    report = evaluate_phase_b(
        model,
        _diagnostics_episodes(count=1),
        config=DiagnosticsConfig(stride=1, batch_size=2, ranking_candidates=2, action_horizon=20),
        device="cpu",
    )
    assert report["overall"]["ranking_rows"] == 5


def test_phase_b_diagnostics_skip_invalid_targets() -> None:
    from mosaic_n17.eval.phase_b_diagnostics import DiagnosticsConfig, select_rows

    episode = _SyntheticEpisode(80, torch.zeros(14, 384), 0)
    episode.valid[:] = True
    episode.valid[25:40] = False
    rows = select_rows([episode], DiagnosticsConfig(stride=1))
    assert rows
    assert all(not (25 <= t + 20 < 40) for _, t in rows)
    assert all(not (25 <= t < 40) for _, t in rows)
