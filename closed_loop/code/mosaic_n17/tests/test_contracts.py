"""Contract tests for the SONIC78 route (CPU only, no gated weights)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from mosaic_n17.contracts import (
    AC_STATE46_DIM,
    G1_JOINT_ORDER_43,
    N17_STATE46_DIM,
    PLACEHOLDER_PROVENANCE,
    SONIC78_LOW_LATENCY,
    SONIC78_V1,
    PlanCommitment,
    ProvenanceRecord,
    TickSchedule,
    n17_state46_to_ac_state46,
    quaternion_wxyz_to_rpy,
    split_sonic43,
    validate_provenance,
)
from mosaic_n17.contracts.state46 import matrix_to_rpy, quaternion_wxyz_to_matrix


def _action(horizon: int = 40, *, motion: float = 0.1) -> np.ndarray:
    action = np.zeros((horizon, 78), dtype=np.float64)
    action[:, :64] = motion
    action[:, 64:] = 0.01
    return action


def test_spec_matches_official_cadence() -> None:
    assert SONIC78_V1.horizon == 40 and SONIC78_V1.execute_steps == 20
    assert SONIC78_V1.replan_hz == 2.5
    assert SONIC78_LOW_LATENCY.replan_hz == 5.0
    with pytest.raises(ValueError):
        type(SONIC78_V1)(horizon=40, execute_steps=10)


def test_split_and_shapes() -> None:
    motion, left, right = SONIC78_V1.split(_action())
    assert motion.shape == (40, 64) and left.shape == (40, 7) and right.shape == (40, 7)


def test_rejects_49d_and_short_horizon() -> None:
    with pytest.raises(ValueError, match="49D"):
        SONIC78_V1.validate(np.zeros((40, 49)))
    with pytest.raises(ValueError, match="horizon 40"):
        SONIC78_V1.validate_block(np.zeros((39, 78)))


def test_motion_mask_must_be_all_or_none() -> None:
    mask = np.ones((40, 78), dtype=bool)
    mask[0, 3] = False
    with pytest.raises(ValueError, match="all-or-none"):
        SONIC78_V1.validate(_action(), mask)
    mask[0, :64] = False
    motion_ok, hand_ok = SONIC78_V1.validate(_action(), mask)[3:]
    assert motion_ok[0] == False and hand_ok[0].all()  # noqa: E712 - explicit numpy bool
    assert motion_ok[1] == True  # noqa: E712


def test_token_bound_and_non_finite() -> None:
    action = _action()
    action[5, 7] = 1.26
    with pytest.raises(ValueError, match="1.25"):
        SONIC78_V1.validate(action)
    action = _action()
    action[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        SONIC78_V1.validate(action)


def test_provenance_rules() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        SONIC78_V1.validate(_action(), provenance=PLACEHOLDER_PROVENANCE)
    with pytest.raises(ValueError, match="provenance"):
        SONIC78_V1.validate(_action(), require_provenance=True)
    SONIC78_V1.validate(_action(), provenance="recorded_token", require_provenance=True)
    with pytest.raises(ValueError):
        validate_provenance("invented_tag")


def test_hand_order_matches_simple_layout() -> None:
    state = np.zeros(N17_STATE46_DIM)
    state[:43] = np.arange(43)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    ac46 = n17_state46_to_ac_state46(state, quat)
    assert ac46.shape == (AC_STATE46_DIM,)
    # root rpy == 0 for identity quaternion
    np.testing.assert_allclose(ac46[:3], 0.0, atol=1e-12)
    # leg/waist slots keep the raw body order
    np.testing.assert_allclose(ac46[3:18], np.arange(15))
    # arms are left(15..21) then right(29..35)
    np.testing.assert_allclose(ac46[18:32], np.r_[np.arange(15, 22), np.arange(29, 36)])
    # hands: official index/middle/thumb -> thumb/index/middle
    left_official = np.arange(22, 29)
    np.testing.assert_allclose(ac46[32:39], left_official[[4, 5, 6, 0, 1, 2, 3]])


def test_gravity_is_not_used_as_root_attitude() -> None:
    state_a = np.zeros(N17_STATE46_DIM)
    state_b = state_a.copy()
    state_b[43:] = [0.0, 0.0, -1.0]  # different projected gravity
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        n17_state46_to_ac_state46(state_a, quat), n17_state46_to_ac_state46(state_b, quat)
    )


def test_root_rpy_roundtrip_and_singularity() -> None:
    quat = np.array([0.9553365, 0.0, 0.2955202, 0.0])  # pitch ~= 0.6 rad
    rpy = quaternion_wxyz_to_rpy(quat)
    np.testing.assert_allclose(rpy[1], 0.6, atol=1e-6)
    singular = quat_from_pitch(np.pi / 2)
    with pytest.raises(ValueError, match="singular"):
        quaternion_wxyz_to_rpy(singular)


def quat_from_pitch(pitch: float) -> np.ndarray:
    return np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])


def test_split_sonic43_body_and_hands() -> None:
    values = np.arange(43)
    body, left, right = split_sonic43(values)
    np.testing.assert_allclose(body, np.r_[np.arange(22), np.arange(29, 36)])
    np.testing.assert_allclose(left, np.arange(22, 29))
    np.testing.assert_allclose(right, np.arange(36, 43))


def test_joint_order_table_is_frozen() -> None:
    assert len(G1_JOINT_ORDER_43) == 43
    assert len(set(G1_JOINT_ORDER_43)) == 43
    assert G1_JOINT_ORDER_43[12] == "waist_yaw_joint"
    assert G1_JOINT_ORDER_43[22] == "left_hand_index_0_joint"


def test_tick_schedule_and_commitment() -> None:
    schedule = TickSchedule()
    assert schedule.replan_hz == 2.5 and schedule.execute_seconds == pytest.approx(0.4)
    assert schedule.ticks_into_plan(100, 90) == 10
    with pytest.raises(ValueError):
        schedule.ticks_into_plan(130, 90)

    first = PlanCommitment(
        plan_id="p0",
        observation_tick=90,
        ready_tick=95,
        first_executable_tick=96,
        prefix_len=0,
        consumed_ticks=(96, 97),
    )
    assert first.last_consumed_tick == 97
    # The executor keeps consuming the previous chunk until the new plan is
    # ready, so the successor may start later than last_consumed + 1.
    next_plan = PlanCommitment(
        plan_id="p1",
        observation_tick=98,
        ready_tick=103,
        first_executable_tick=103,
        prefix_len=0,
    )
    next_plan.assert_can_replace(first, schedule=schedule)

    stale = PlanCommitment(
        plan_id="p_stale",
        observation_tick=97,
        ready_tick=99,
        first_executable_tick=99,
        prefix_len=0,
    )
    with pytest.raises(ValueError, match="last consumed tick"):
        stale.assert_can_replace(first, schedule=schedule)


def test_commitment_rejects_unordered_ticks() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        PlanCommitment(
            plan_id="p",
            observation_tick=0,
            ready_tick=1,
            first_executable_tick=2,
            prefix_len=0,
            consumed_ticks=(3, 2),
        )


def test_provenance_record_requires_receipt_for_verification() -> None:
    record = ProvenanceRecord(
        candidate_id="c0",
        provenance="selected_teacher",
        generator_sha256="0" * 64,
        processor_sha256="1" * 64,
        stats_sha256="2" * 64,
    )
    assert record.is_sim_verified is False
    verified = ProvenanceRecord(
        candidate_id="c0",
        provenance="selected_teacher",
        generator_sha256="0" * 64,
        processor_sha256="1" * 64,
        stats_sha256="2" * 64,
        sim_receipt_sha256="3" * 64,
        bank_generation=2,
    )
    assert verified.is_sim_verified is True


def test_parity_with_flowbody_joint_order_when_available() -> None:
    root = os.environ.get("FLOWBODY_ROOT")
    if not root:
        pytest.skip("set FLOWBODY_ROOT to run the cross-repo joint-order parity test")
    sys.path.insert(0, os.path.join(root, "flowbody"))
    try:
        from mosaic_sonic78.data import joint_order as fb  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    assert tuple(fb.SONIC_WBC43_JOINT_NAMES) == G1_JOINT_ORDER_43
    assert tuple(fb.SIMPLE_TO_OFFICIAL_HAND_INDEX) == tuple(
        int(index) for index in _left_hand_permutation()
    )
    with pytest.raises(ValueError):
        matrix_to_rpy(quaternion_wxyz_to_matrix(quat_from_pitch(np.pi / 2)))


def _left_hand_permutation() -> tuple[int, ...]:
    from mosaic_n17.contracts.state46 import SIMPLE_TO_OFFICIAL_LEFT

    return SIMPLE_TO_OFFICIAL_LEFT
