"""Stage frontier / concordance tests (CPU only)."""

from __future__ import annotations

import numpy as np
import pytest

from mosaic_n17.ac import SLOT_DIM, TOTAL_SLOTS, slot_slices
from mosaic_n17.stage import (
    StageFrontier,
    build_stage_goal,
    concordance,
    rank_candidates,
    top1_agreement,
)


def _goal_slots(value: float = 1.0) -> np.ndarray:
    slots = np.zeros((TOTAL_SLOTS, SLOT_DIM))
    slots[slot_slices()["S"], : SLOT_DIM // 2] = value
    return slots


def test_stage_goal_binds_digest() -> None:
    goal = build_stage_goal(_goal_slots(), weights_sha256="e" * 64)
    assert len(goal.frontier_sha256) == 64
    with pytest.raises(ValueError, match="does not match"):
        type(goal)(
            slots=goal.slots + 1.0,
            frontier_sha256=goal.frontier_sha256,
            weights_sha256=goal.weights_sha256,
        )


def test_stage_goal_rejects_bad_layout_and_weights() -> None:
    with pytest.raises(ValueError, match="must end in"):
        build_stage_goal(np.zeros((TOTAL_SLOTS, 128)), weights_sha256="e" * 64)
    with pytest.raises(ValueError, match="weights_sha256"):
        build_stage_goal(_goal_slots(), weights_sha256="short")


def test_frontier_wraps_a_callable() -> None:
    frontier = StageFrontier(
        lambda current, instruction: np.ones_like(current), weights_sha256="f" * 64
    )
    current = np.zeros((TOTAL_SLOTS, SLOT_DIM))
    goal = frontier.goal(current)
    assert goal.slots.shape == (TOTAL_SLOTS, SLOT_DIM)
    assert goal.source == "stage_jepa"


def test_concordance_ranking_and_report() -> None:
    current = np.zeros((TOTAL_SLOTS, SLOT_DIM))
    current[slot_slices()["S"], SLOT_DIM // 2 :] = 1.0
    goal = _goal_slots()
    predicted = np.stack([current, goal])  # candidate 0: no progress, candidate 1: at goal
    report = concordance(current, predicted, goal)
    assert report.best_index == 1
    assert report.gains[1] > report.gains[0]
    np.testing.assert_array_equal(report.order, [1, 0])


def test_concordance_rejects_mismatched_batch() -> None:
    current = np.zeros((TOTAL_SLOTS, SLOT_DIM))
    with pytest.raises(ValueError, match="must be"):
        concordance(current, np.zeros((2, 3, TOTAL_SLOTS, SLOT_DIM)), current)
    with pytest.raises(ValueError, match="must end in"):
        concordance(current, np.zeros((2, TOTAL_SLOTS, SLOT_DIM // 2)), current)


def test_rank_candidates_and_top1_agreement() -> None:
    gains = np.array([0.1, 0.5, -0.2])
    np.testing.assert_array_equal(rank_candidates(gains), [1, 0, 2])
    a = np.array([[0.1, 0.9], [0.8, 0.2]])
    b = np.array([[0.2, 0.7], [0.1, 0.9]])
    assert top1_agreement(a, b) == 0.5
    with pytest.raises(ValueError, match="share a shape"):
        top1_agreement(a, b[:, :1])
