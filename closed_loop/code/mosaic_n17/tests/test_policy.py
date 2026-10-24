"""Policy-layer tests: decoding, K sampling, gate and shadow (CPU only)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mosaic_n17.policy import (
    GateConfig,
    assemble_sonic78,
    decide,
    decode_candidates,
    run_shadow_step,
)
from mosaic_n17.policy.candidate_sampler import (
    CandidateSet,
    build_candidate_set,
    replicate_action_input,
    sample_from_features,
    sample_raw_candidates,
)


class FakeProcessor:
    """Mirrors ``Gr00tN1d7Processor.decode_action`` group splitting."""

    def __init__(self, horizon: int = 40, action_dim: int = 132) -> None:
        self.horizon = horizon
        self.action_dim = action_dim
        self.calls = 0

    def decode_action(self, action, embodiment_tag, state=None):
        array = np.asarray(action)
        if array.shape[-2:] != (self.horizon, self.action_dim):
            raise ValueError(f"decode_action expected (..., {self.horizon}, {self.action_dim})")
        self.calls += 1
        return {
            "motion_token": array[..., : self.horizon, 0:64],
            "left_hand_joints": array[..., : self.horizon, 64:71],
            "right_hand_joints": array[..., : self.horizon, 71:78],
        }


class FakeHead:
    config = SimpleNamespace(action_horizon=40, action_dim=132)

    def __init__(self, scale: float = 0.05, encode_log: list[int] | None = None) -> None:
        self.scale = scale
        self.encode_log = encode_log if encode_log is not None else []
        self.received_batch = None

    def _encode_features(self, backbone_output, action_input):
        self.encode_log.append(1)
        state = action_input["state"]
        return SimpleNamespace(
            backbone_features=torch.ones(state.shape[0], 4, 8),
            state_features=torch.zeros(state.shape[0], 1, 8),
        )

    def get_action_with_features(
        self,
        backbone_features,
        state_features,
        embodiment_id,
        backbone_output,
        action_input,
        options=None,
    ):
        batch = int(backbone_features.shape[0])
        self.received_batch = batch
        generator = torch.Generator().manual_seed(7)
        noise = torch.randn(batch, 40, 132, generator=generator)
        return {"action_pred": noise * self.scale + embodiment_id.float().view(-1, 1, 1)}


class FakeModel:
    def __init__(self, head: FakeHead) -> None:
        self.action_head = head

    def prepare_input(self, inputs):
        from transformers.feature_extraction_utils import BatchFeature

        state = inputs["state"]
        action_input = BatchFeature(data={"state": state, "embodiment_id": inputs["embodiment_id"]})
        return BatchFeature(data={"input_ids": inputs["input_ids"]}), action_input

    def backbone(self, backbone_inputs):
        from transformers.feature_extraction_utils import BatchFeature

        return BatchFeature(data={"backbone_features": torch.zeros(1, 4, 8)})


def _inputs(batch: int = 1) -> dict:
    return {
        "input_ids": torch.zeros(batch, 8, dtype=torch.long),
        "state": torch.zeros(batch, 132),
        "embodiment_id": torch.zeros(batch, dtype=torch.long),
    }


def test_decode_preserves_leading_dims_and_validates() -> None:
    raw = np.zeros((2, 3, 40, 132), dtype=np.float32)
    raw[..., :64] = 0.1
    decoded = decode_candidates(FakeProcessor(), raw)
    assert decoded.shape == (2, 3, 40, 78)
    with pytest.raises(ValueError, match="missing action group"):
        assemble_sonic78(
            {"motion_token": np.zeros((40, 64)), "left_hand_joints": np.zeros((40, 7))}
        )


def test_decode_rejects_short_horizon() -> None:
    raw = np.zeros((1, 39, 132), dtype=np.float32)
    with pytest.raises(ValueError):
        decode_candidates(FakeProcessor(), raw)


def test_sampler_encodes_once_and_returns_k_rows() -> None:
    head = FakeHead()
    model = FakeModel(head)
    raw = sample_raw_candidates(model, _inputs(), K=4)
    assert raw.shape == (1, 4, 40, 132)
    assert len(head.encode_log) == 1, "features must be encoded exactly once"
    assert head.received_batch == 4
    assert not torch.allclose(raw[0, 0], raw[0, 1])


def test_sampler_rejects_incomplete_rtc_options() -> None:
    head = FakeHead()
    with pytest.raises(ValueError, match="must be exactly"):
        sample_from_features(
            head,
            SimpleNamespace(
                backbone_features=torch.ones(1, 4, 8), state_features=torch.zeros(1, 1, 8)
            ),
            {
                "state": torch.zeros(1, 132),
                "embodiment_id": torch.zeros(1, dtype=torch.long),
            },
            SimpleNamespace(),
            K=2,
            options={"rtc_overlap_steps": 5},
        )


def test_sampler_requires_prefix_when_rtc_options_are_complete() -> None:
    head = FakeHead()
    with pytest.raises(ValueError, match="require action_input"):
        sample_from_features(
            head,
            SimpleNamespace(
                backbone_features=torch.ones(1, 4, 8), state_features=torch.zeros(1, 1, 8)
            ),
            {
                "state": torch.zeros(1, 132),
                "embodiment_id": torch.zeros(1, dtype=torch.long),
            },
            SimpleNamespace(),
            K=2,
            options={
                "action_horizon": 40,
                "rtc_overlap_steps": 20,
                "rtc_frozen_steps": 2,
                "rtc_ramp_rate": 6.0,
            },
        )


def test_replicate_action_input_interleaves_batch() -> None:
    from transformers.feature_extraction_utils import BatchFeature

    ai = BatchFeature(
        data={
            "state": torch.arange(3).view(3, 1),
            "embodiment_id": torch.tensor([0, 1, 2]),
            "scalar": 5,
        }
    )
    out = replicate_action_input(ai, 3, 2)
    assert out["state"].shape == (6, 1)
    assert out["embodiment_id"].tolist() == [0, 0, 1, 1, 2, 2]
    assert out["scalar"] == 5


def test_candidate_set_hashes_and_provenance() -> None:
    candidates = np.zeros((1, 3, 40, 78))
    candidates[..., :64] = 0.01
    cs = build_candidate_set(candidates, provenance="selected_teacher")
    assert cs.num_candidates == 3 and len(cs.token_sha256[0]) == 3
    assert len(set(cs.token_sha256[0])) == 1
    with pytest.raises(ValueError, match="placeholder"):
        build_candidate_set(candidates, provenance="placeholder")


def test_gate_fails_closed_when_no_candidate_is_valid() -> None:
    candidates = np.zeros((2, 40, 78))
    candidates[..., :64] = 4.0  # beyond 1.25
    decision = decide(candidates, components={"stage_gain": np.zeros(2)}, config=GateConfig())
    assert decision.fallback and decision.index == -1
    assert decision.reason == "no_valid_candidate"


def test_gate_requires_active_score_terms() -> None:
    candidates = np.zeros((2, 40, 78))
    with pytest.raises(ValueError, match="required by the gate config"):
        decide(candidates, components={}, config=GateConfig(stage_weight=1.0))


def test_gate_ranks_and_reports_margin() -> None:
    candidates = np.zeros((3, 40, 78))
    candidates[..., :64] = 0.01
    decision = decide(
        candidates,
        components={
            "stage_gain": np.array([0.1, 0.9, 0.2]),
            "support": np.array([0.0, 0.0, 0.0]),
            "uncertainty": np.array([0.5, 0.1, 0.0]),
        },
        config=GateConfig(stage_weight=1.0, support_weight=0.0, uncertainty_weight=1.0),
    )
    assert decision.index == 1 and not decision.fallback
    # scores = stage_gain - uncertainty -> [0.4-? ] ; k0=-0.4, k1=0.8, k2=0.2
    assert decision.margin == pytest.approx(0.6)


def test_gate_marks_ambiguity() -> None:
    candidates = np.zeros((2, 40, 78))
    decision = decide(
        candidates,
        components={
            "stage_gain": np.array([0.5, 0.5001]),
            "support": np.zeros(2),
            "uncertainty": np.zeros(2),
        },
        config=GateConfig(support_weight=0.0, uncertainty_weight=0.0, selection_margin=0.01),
    )
    assert decision.ambiguous and decision.reason == "selected_ambiguous"


def test_shadow_executes_primary_and_records_gate() -> None:
    candidates = np.zeros((1, 3, 40, 78))
    candidates[..., :64] = 0.01
    cs = build_candidate_set(candidates)
    decision = decide(
        cs.candidates[0],
        components={
            "stage_gain": np.array([0.0, 1.0, 0.5]),
            "support": np.zeros(3),
            "uncertainty": np.zeros(3),
        },
        config=GateConfig(support_weight=0.0, uncertainty_weight=0.0),
    )
    executed, record = run_shadow_step(cs, decision)
    assert executed is not None and executed.shape == (40, 78)
    assert record.executed_index == 0 and record.gate_index == 1
    assert record.agrees is False and record.executed_source == "primary"


def test_shadow_returns_nothing_when_primary_invalid() -> None:
    candidates = np.zeros((1, 2, 40, 78))
    candidates[0, 0, :, :64] = 9.0
    candidates[0, 1, :, :64] = 0.01
    cs = CandidateSet(
        candidates=candidates,
        motion_valid=np.ones((1, 2, 40), dtype=bool),
        hand_valid=np.ones((1, 2, 40, 14), dtype=bool),
        seeds=(0, 1),
        token_sha256=(("a", "b"),),
    )
    decision = decide(
        candidates[0],
        components={"stage_gain": np.zeros(2), "support": np.zeros(2), "uncertainty": np.zeros(2)},
        config=GateConfig(support_weight=0.0, uncertainty_weight=0.0),
    )
    executed, record = run_shadow_step(cs, decision)
    assert executed is None and record.executed_source == "fallback"


def test_wrapper_requires_scorer_for_multiple_candidates() -> None:
    from mosaic_n17.policy.policy_wrapper import MosaicPolicyWrapper

    with pytest.raises(ValueError, match="requires a scorer"):
        MosaicPolicyWrapper(object(), K=2)


def test_wrapper_returns_already_decoded_rows() -> None:
    from mosaic_n17.policy.policy_wrapper import MosaicPolicyWrapper

    wrapper = MosaicPolicyWrapper(object(), K=3, scorer=lambda candidates, state: {})
    candidates = build_candidate_set(np.random.default_rng(0).random((1, 3, 40, 78)) * 0.01)
    actions = wrapper._select_action_dict(candidates, [2])
    assert actions["motion_token"].shape == (1, 40, 64)
    assert actions["left_hand_joints"].shape == (1, 40, 7)
    np.testing.assert_allclose(actions["motion_token"][0], candidates.candidates[0, 2, :, :64])
    assert all(value.dtype == np.float32 for value in actions.values())
