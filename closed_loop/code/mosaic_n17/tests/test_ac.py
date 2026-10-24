"""Typed-slot / Phase A / Phase B / scoring / checkpoint tests (CPU only)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mosaic_n17.ac import (
    SLOT_COUNTS,
    SLOT_DIM,
    TOTAL_SLOTS,
    PhaseAAdapter,
    PhaseAContract,
    PhaseBConfig,
    PhaseBContract,
    Sonic78PhaseB,
    load_phase_b_checkpoint,
    masked_huber,
    plausibility,
    save_phase_b_checkpoint,
    score_components,
    slot_slices,
    split_action78,
    stage_gain,
    typed_distance,
    validate_consequence_supervision,
    validate_slots,
)
from mosaic_n17.contracts import SONIC78_LOW_LATENCY, SONIC78_V1


def _slots(batch: int = 2, value: float = 0.1) -> torch.Tensor:
    return torch.full((batch, TOTAL_SLOTS, SLOT_DIM), value)


def test_slot_slices_cover_all_slots() -> None:
    slices = slot_slices()
    assert tuple(slices) == ("C", "B", "E", "S")
    assert sum(SLOT_COUNTS.values()) == TOTAL_SLOTS == 14
    cursor = 0
    for name in slices:
        span = slices[name]
        assert span.start == cursor
        cursor = span.stop
    assert cursor == TOTAL_SLOTS


def test_validate_slots_rejects_bad_shape_and_nan() -> None:
    with pytest.raises(ValueError):
        validate_slots(torch.zeros(2, 13, SLOT_DIM))
    bad = _slots()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_slots(bad)


def test_masked_huber_empty_mask_is_differentiable_zero() -> None:
    pred = torch.randn(2, 14, SLOT_DIM, requires_grad=True)
    target = torch.zeros(2, 14, SLOT_DIM)
    mask = torch.zeros(2, 14, SLOT_DIM, dtype=torch.bool)
    loss = masked_huber(pred, target, mask)
    loss.backward()
    assert float(loss) == 0.0 and pred.grad is not None


def test_split_action78_widths() -> None:
    actions = torch.zeros(2, 40, 78)
    motion, left, right = split_action78(actions)
    assert motion.shape[-1] == 64 and left.shape[-1] == 7 and right.shape[-1] == 7
    with pytest.raises(ValueError, match="78"):
        split_action78(torch.zeros(2, 40, 49))


def test_phase_b_forward_shapes_and_rejections() -> None:
    model = Sonic78PhaseB(PhaseBConfig())
    out = model(_slots(), torch.zeros(2, 46), torch.zeros(2, 40, 78))
    assert out["predicted_slots"].shape == (2, TOTAL_SLOTS, SLOT_DIM)
    assert out["horizon"] == 40
    with pytest.raises(ValueError, match="horizon"):
        model(_slots(), torch.zeros(2, 46), torch.zeros(2, 39, 78))
    with pytest.raises(ValueError, match="state46"):
        model(_slots(), torch.zeros(2, 49), torch.zeros(2, 40, 78))
    bad_state = torch.zeros(2, 46)
    bad_state[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        model(_slots(), bad_state, torch.zeros(2, 40, 78))


def test_phase_b_aux_state46_head_is_opt_in() -> None:
    plain = Sonic78PhaseB(PhaseBConfig())
    assert (
        plain(_slots(), torch.zeros(2, 46), torch.zeros(2, 40, 78))["predicted_state46"]
        is None
    )
    aux = Sonic78PhaseB(PhaseBConfig(aux_state46=True))
    output = aux(_slots(), torch.zeros(2, 46), torch.zeros(2, 40, 78))
    assert output["predicted_state46"].shape == (2, 46)


def test_phase_b_history_inputs_are_opt_in() -> None:
    config = PhaseBConfig(control_horizon=20, prefix_only=True, history_slots=1, prev_action_ticks=20)
    model = Sonic78PhaseB(config)
    out = model(
        _slots(),
        torch.zeros(2, 46),
        torch.zeros(2, 20, 78),
        history_slots=torch.zeros(2, 1, TOTAL_SLOTS, SLOT_DIM),
        prev_actions=torch.zeros(2, 20, 78),
    )
    assert out["predicted_slots"].shape == (2, TOTAL_SLOTS, SLOT_DIM)
    with pytest.raises(ValueError, match="history_slots"):
        model(_slots(), torch.zeros(2, 46), torch.zeros(2, 20, 78))
    with pytest.raises(ValueError, match="prev_actions"):
        model(
            _slots(),
            torch.zeros(2, 46),
            torch.zeros(2, 20, 78),
            history_slots=torch.zeros(2, 1, TOTAL_SLOTS, SLOT_DIM),
        )
    with pytest.raises(ValueError, match="history_slots must be in"):
        PhaseBConfig(history_slots=5)


def test_executed_prefix_supervision() -> None:
    validate_consequence_supervision(supervision_step=20)
    with pytest.raises(ValueError, match=r"\[1,20\]"):
        validate_consequence_supervision(supervision_step=30)
    with pytest.raises(ValueError, match="unexecuted tail"):
        validate_consequence_supervision(supervision_step=40, target_semantics="full_horizon")
    validate_consequence_supervision(
        supervision_step=10, spec=SONIC78_LOW_LATENCY, target_semantics="executed_prefix"
    )


def test_phase_a_adapter_validates_contract_and_output() -> None:
    def encode_fn(dense: torch.Tensor) -> torch.Tensor:
        return torch.ones(dense.shape[0], TOTAL_SLOTS, SLOT_DIM)

    contract = PhaseAContract(
        architecture_id="structured_slot_phase_a_v31",
        dense_dim=1408,
        checkpoint_sha256="a" * 64,
    )
    adapter = PhaseAAdapter(encode_fn, contract)
    slots = adapter.encode(torch.zeros(3, 256, 1408))
    assert slots.shape == (3, 14, 384)

    with pytest.raises(ValueError, match="last dim"):
        adapter.encode(torch.zeros(3, 256, 1024))
    with pytest.raises(ValueError, match="unsupported Phase A architecture"):
        PhaseAContract(architecture_id="other", dense_dim=1408, checkpoint_sha256="a" * 64)
    with pytest.raises(ValueError, match="slot counts"):
        PhaseAContract(
            architecture_id="structured_slot_phase_a_v31",
            dense_dim=1408,
            checkpoint_sha256="a" * 64,
            slot_counts=(("C", 5), ("B", 3), ("E", 3), ("S", 2)),
        )


def test_phase_a_adapter_rejects_wrong_output_shape() -> None:
    contract = PhaseAContract(
        architecture_id="structured_slot_phase_a_v31",
        dense_dim=16,
        checkpoint_sha256="b" * 64,
    )
    adapter = PhaseAAdapter(lambda dense: torch.zeros(dense.shape[0], 3, SLOT_DIM), contract)
    with pytest.raises(ValueError, match="Phase A must return"):
        adapter.encode(torch.zeros(1, 4, 16))


def test_scoring_stage_gain_and_plausibility() -> None:
    stages = slot_slices()["S"]
    # Orthogonal Stage directions: the typed score pools slots, so direction
    # (not magnitude) is what separates candidates.
    goal = np.zeros((2, TOTAL_SLOTS, SLOT_DIM))
    goal[:, stages, : SLOT_DIM // 2] = 1.0
    current = np.zeros((2, TOTAL_SLOTS, SLOT_DIM))
    current[:, stages, SLOT_DIM // 2 :] = 1.0

    np.testing.assert_allclose(stage_gain(current, current, goal), 0.0, atol=1e-12)
    np.testing.assert_allclose(stage_gain(current, goal, goal), 1.0, atol=1e-9)

    opposite = -goal.copy()
    assert np.all(stage_gain(current, opposite, goal) < 0.0)

    same_change = plausibility(current, current, scale=0.1)
    np.testing.assert_allclose(same_change, 0.0)
    changed = plausibility(current, goal, scale=0.1)
    assert np.all(changed < 0.0)
    with pytest.raises(ValueError, match="scale"):
        plausibility(current, goal, scale=0.0)


def test_score_components_are_explicit() -> None:
    current = np.zeros((3, TOTAL_SLOTS, SLOT_DIM))
    predicted = np.ones((3, TOTAL_SLOTS, SLOT_DIM)) * 0.01
    components = score_components(
        current_slots=current, predicted_slots=predicted, goal_slots=None, change_scale=0.1
    )
    assert set(components) == {"support"}
    with_goal = score_components(
        current_slots=current,
        predicted_slots=predicted,
        goal_slots=current,
        change_scale=0.1,
        uncertainty=np.zeros(3),
    )
    assert set(with_goal) == {"support", "stage_gain", "uncertainty"}
    distances = typed_distance(current, predicted)
    assert set(distances) == {"C", "B", "E", "S"}


def test_phase_b_checkpoint_roundtrip_and_contract(tmp_path) -> None:
    model = Sonic78PhaseB(PhaseBConfig())
    contract = PhaseBContract(phase_a_sha256="c" * 64)
    path = tmp_path / "phase_b.pt"
    digest = save_phase_b_checkpoint(path, model=model, config=PhaseBConfig(), contract=contract)
    assert len(digest) == 64
    payload = load_phase_b_checkpoint(path)
    assert payload["schema"] == "mosaic.n17.phase_b_checkpoint.v1"
    with pytest.raises(ValueError, match="Phase A digest mismatch"):
        load_phase_b_checkpoint(path, expected_phase_a_sha256="d" * 64)
    with pytest.raises(ValueError, match="state46 layout"):
        PhaseBContract(phase_a_sha256="c" * 64, state_layout="n17_joints43_projected_gravity3")
    with pytest.raises(ValueError, match="cadence"):
        PhaseBContract(phase_a_sha256="c" * 64, horizon=20, execute_steps=10).assert_matches(
            SONIC78_V1
        )
