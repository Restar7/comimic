"""RTC window, prefix encoding and K-candidate prefix sharing (CPU only)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mosaic_n17.policy.candidate_sampler import (
    sample_candidates_and_normalized,
    sample_from_features,
    sample_raw_candidates,
)
from mosaic_n17.policy.rtc import (
    RtcRequest,
    RtcWindow,
    align_prefix_tail,
    encode_committed_action,
    prefix_equal,
    validate_prefix_tensor,
)


HORIZON = 40
MODEL_DIM = 132


class ScaleProcessor:
    """Processor double: normalization is a scale, so tests can prove delegation."""

    def __init__(self, scale: float = 1.0, model_dim: int = MODEL_DIM) -> None:
        self.max_action_dim = model_dim
        self.scale = scale
        self.modality_configs = {
            "unitree_g1_sonic": {
                "action": SimpleNamespace(
                    modality_keys=["motion_token", "left_hand_joints", "right_hand_joints"]
                )
            }
        }
        self.state_action_processor = SimpleNamespace(apply_action=self._apply_action)

    def _apply_action(self, groups, embodiment_tag, state=None):
        expected = {"motion_token", "left_hand_joints", "right_hand_joints"}
        if set(groups) != expected:
            raise KeyError(f"unexpected groups {sorted(groups)}")
        return {
            key: np.asarray(value, dtype=np.float32) * self.scale for key, value in groups.items()
        }

    def decode_action(self, action, embodiment_tag, state=None):
        array = np.asarray(action)
        if array.shape[-2:] != (HORIZON, self.max_action_dim):
            raise ValueError(f"decode_action expected (..., {HORIZON}, {self.max_action_dim})")
        return {
            "motion_token": array[..., :HORIZON, 0:64],
            "left_hand_joints": array[..., :HORIZON, 64:71],
            "right_hand_joints": array[..., :HORIZON, 71:78],
        }

    def __call__(self, messages):
        return {"state": np.zeros((1, 46), dtype=np.float32)}


class RtcHead:
    """Replicates the official RTC math (``gr00t_n1d7.py:373-394``) without weights."""

    def __init__(self, horizon: int = HORIZON, action_dim: int = MODEL_DIM) -> None:
        self.config = SimpleNamespace(action_horizon=horizon, action_dim=action_dim)
        self.seen: dict | None = None

    def _encode_features(self, backbone_output, action_input):
        batch = int(action_input["state"].shape[0])
        return SimpleNamespace(
            backbone_features=torch.zeros(batch, 4, 8),
            state_features=torch.zeros(batch, 1, 8),
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
        generator = torch.Generator().manual_seed(11)
        shape = (batch, self.config.action_horizon, self.config.action_dim)
        noise = 0.1 * torch.randn(*shape, generator=generator)
        actions = noise.clone()
        if "action" in action_input:
            window = RtcWindow.from_model_options(options)
            overlap, frozen = window.overlap, window.frozen
            actions[:, :overlap, :] = action_input["action"][
                :, window.horizon - overlap : window.horizon, :
            ]
            steps = torch.linspace(0.0, 1.0, overlap - frozen + 2)[1:-1]
            ramp = 1 - torch.exp(-window.ramp_rate * steps)
            actions[:, frozen:overlap, :] += 0.5 * ramp[None, :, None] * noise[:, frozen:overlap, :]
        self.seen = {"options": options, "action": action_input.get("action"), "batch": batch}
        return {"action_pred": actions}


class RtcModel:
    config = SimpleNamespace(max_action_dim=MODEL_DIM, action_horizon=HORIZON)

    def __init__(self, head: RtcHead) -> None:
        self.action_head = head

    def prepare_input(self, inputs):
        from transformers.feature_extraction_utils import BatchFeature

        data = {key: value for key, value in inputs.items()}
        data.setdefault("embodiment_id", torch.zeros(int(data["state"].shape[0]), dtype=torch.long))
        return BatchFeature(data={"input_ids": torch.zeros(1, 8, dtype=torch.long)}), BatchFeature(
            data=data
        )

    def backbone(self, backbone_inputs):
        from transformers.feature_extraction_utils import BatchFeature

        return BatchFeature(
            data={
                "backbone_features": torch.zeros(1, 4, 8),
                "image_mask": torch.zeros(1, 4, dtype=torch.bool),
                "backbone_attention_mask": torch.ones(1, 4, dtype=torch.bool),
            }
        )


def _committed(*, j: int = 20) -> np.ndarray:
    """[1, 40, 132] normalized committed chunk with a distinctive tail."""

    chunk = np.zeros((1, HORIZON, MODEL_DIM), dtype=np.float32)
    chunk[0, :, :3] = np.arange(HORIZON, dtype=np.float32)[:, None]
    chunk[0, j:] = np.linspace(-0.5, 0.5, HORIZON - j, dtype=np.float32)[:, None]
    return chunk


def _request(j: int = 20, overlap: int = 6, frozen: int = 2) -> RtcRequest:
    """RTC request with the old plan's next rows re-aligned into the tail slots."""

    window = RtcWindow(overlap=overlap, frozen=frozen)
    aligned = align_prefix_tail(_committed(), start=j, overlap=overlap, horizon=HORIZON)
    return RtcRequest(window=window, previous_action=aligned)


def _inputs(batch: int = 1) -> dict:
    return {
        "input_ids": torch.zeros(batch, 8, dtype=torch.long),
        "state": torch.zeros(batch, MODEL_DIM),
        "embodiment_id": torch.zeros(batch, dtype=torch.long),
    }


def test_window_defaults_match_the_control_contract() -> None:
    window = RtcWindow()
    assert window.horizon == 40 and window.action_horizon == 40
    assert window.overlap == 20 and window.frozen == 0
    window.validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 39},
        {"overlap": 0},
        {"overlap": 41},
        {"frozen": -1},
        {"frozen": 20},
        {"overlap": 2, "frozen": 1},
        {"ramp_rate": 0.0},
        {"ramp_rate": 64.0},
        {"action_horizon": 20},
    ],
)
def test_window_rejects_unexecutable_windows(kwargs) -> None:
    with pytest.raises(ValueError):
        RtcWindow(**kwargs)


def test_window_roundtrips_through_model_options() -> None:
    window = RtcWindow(overlap=18, frozen=2, ramp_rate=6.0)
    options = window.to_model_options()
    assert set(options) == {
        "action_horizon",
        "rtc_overlap_steps",
        "rtc_frozen_steps",
        "rtc_ramp_rate",
    }
    assert RtcWindow.from_model_options(options) == window
    with pytest.raises(ValueError, match="missing=.*unknown="):
        RtcWindow.from_model_options({"rtc_overlap_steps": 18})
    with pytest.raises(ValueError, match="unknown"):
        RtcWindow.from_model_options({**options, "rtc_prefix_len": 4})


def test_request_mapping_roundtrip_requires_every_key() -> None:
    window = RtcWindow(overlap=20, frozen=2)
    request = RtcRequest(
        window=window, previous_action=np.zeros((1, HORIZON, MODEL_DIM), np.float32)
    )
    mapping = request.as_mapping()
    assert RtcRequest.from_mapping(mapping).window == window
    with pytest.raises(ValueError, match="missing"):
        RtcRequest.from_mapping({"horizon": 40})


def test_prefix_tensor_rejects_decoded_block_and_bad_shapes() -> None:
    window = RtcWindow(overlap=20, frozen=2)
    with pytest.raises(ValueError, match="78D"):
        validate_prefix_tensor(np.zeros((1, HORIZON, 78), np.float32), window, action_dim=MODEL_DIM)
    with pytest.raises(ValueError, match="must be"):
        validate_prefix_tensor(
            np.zeros((HORIZON, MODEL_DIM), np.float32), window, action_dim=MODEL_DIM
        )
    with pytest.raises(ValueError, match="prefix must be"):
        validate_prefix_tensor(
            np.zeros((1, HORIZON - 1, MODEL_DIM), np.float32), window, action_dim=MODEL_DIM
        )
    nan = np.zeros((1, HORIZON, MODEL_DIM), np.float32)
    nan[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_prefix_tensor(nan, window, action_dim=MODEL_DIM)


def test_encode_committed_action_delegates_normalization_and_pads() -> None:
    processor = ScaleProcessor(scale=2.0)
    raw = np.full((HORIZON, 78), 0.25, dtype=np.float32)
    encoded = encode_committed_action(processor, raw, window=RtcWindow(overlap=20, frozen=2))
    assert encoded.shape == (1, HORIZON, MODEL_DIM)
    assert np.allclose(encoded[..., :78], 0.5)
    assert np.allclose(encoded[..., 78:], 0.0)
    with pytest.raises(ValueError, match="end in 78"):
        encode_committed_action(processor, np.zeros((HORIZON, 49)), window=RtcWindow())


def test_prefix_equal_compares_only_the_frozen_tail() -> None:
    window = RtcWindow(overlap=20, frozen=2)
    committed = _committed()
    new = np.zeros_like(committed)
    new[:, :2] = committed[:, 20:22]
    assert prefix_equal(committed, new, window)
    new[:, 1] = new[:, 1] + 1e-3
    assert not prefix_equal(committed, new, window)


def test_sampler_shares_prefix_across_k_and_forwards_options() -> None:
    head = RtcHead()
    model = RtcModel(head)
    committed = _committed()
    request = _request(j=20, overlap=6, frozen=2)

    raw = sample_raw_candidates(model, _inputs(), K=4, rtc=request)

    assert raw.shape == (1, 4, HORIZON, MODEL_DIM)
    assert head.seen["options"] == request.window.to_model_options()
    replicated = head.seen["action"]
    assert replicated.shape == (4, HORIZON, MODEL_DIM)
    for index in range(replicated.shape[0]):
        assert torch.equal(replicated[index], torch.from_numpy(request.previous_action[0]))
    for index in range(4):
        assert torch.equal(raw[0, index, :2], torch.from_numpy(committed[0, 20:22])), (
            f"candidate {index} did not inherit the frozen prefix"
        )
    # rows [frozen, overlap) ramp away from the pinned tail, rows >= overlap are free
    assert not torch.allclose(raw[0, 0, 2:6], raw[0, 1, 2:6])
    assert not torch.allclose(raw[0, 0, 6:], raw[0, 1, 6:])


def test_sampler_fails_closed_when_prefix_or_window_is_missing() -> None:
    head = RtcHead()
    features = SimpleNamespace(
        backbone_features=torch.zeros(1, 4, 8), state_features=torch.zeros(1, 1, 8)
    )
    action_input = {
        "state": torch.zeros(1, MODEL_DIM),
        "embodiment_id": torch.zeros(1, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="require action_input"):
        sample_from_features(
            head,
            features,
            action_input,
            SimpleNamespace(),
            K=2,
            options=RtcWindow().to_model_options(),
        )
    with pytest.raises(ValueError, match="no RTC options"):
        sample_from_features(
            head,
            features,
            {**action_input, "action": torch.zeros(1, HORIZON, MODEL_DIM)},
            SimpleNamespace(),
            K=2,
            options=None,
        )
    with pytest.raises(ValueError, match="no RTC window"):
        sample_raw_candidates(
            RtcModel(head),
            {**_inputs(), "action": torch.zeros(1, HORIZON, MODEL_DIM)},
            K=2,
        )


def test_candidate_sampler_returns_normalized_block_for_commit() -> None:
    head = RtcHead()
    model = RtcModel(head)
    candidate_set, normalized = sample_candidates_and_normalized(
        model,
        ScaleProcessor(),
        _inputs(),
        K=3,
        rtc=_request(j=20, overlap=6, frozen=2),
    )
    assert candidate_set.candidates.shape == (1, 3, HORIZON, 78)
    assert normalized.shape == (1, 3, HORIZON, MODEL_DIM)
    assert np.allclose(normalized[..., :78], candidate_set.candidates)


def test_rtc_policy_delegates_without_a_request(monkeypatch) -> None:
    from mosaic_n17.policy.rtc_policy import RtcGr00tPolicy

    calls: list[tuple] = []

    def fake_super(self, observation, options=None):
        calls.append((observation, options))
        return {"motion_token": np.zeros((1, HORIZON, 64), np.float32)}, {}

    monkeypatch.setattr("gr00t.policy.gr00t_policy.Gr00tPolicy._get_action", fake_super)
    policy = object.__new__(RtcGr00tPolicy)
    observation = {"marker": 1}
    delegated, info = RtcGr00tPolicy._get_action(policy, observation)
    assert delegated["motion_token"].shape == (1, HORIZON, 64)
    assert info == {}
    RtcGr00tPolicy._get_action(policy, observation, {"anchor_tick": 3})
    assert calls == [(observation, None), (observation, {"anchor_tick": 3})]


def test_rtc_policy_rejects_other_embodiments() -> None:
    from gr00t.data.embodiment_tags import EmbodimentTag

    from mosaic_n17.policy.rtc_policy import RtcGr00tPolicy

    policy = object.__new__(RtcGr00tPolicy)
    policy.embodiment_tag = EmbodimentTag.REAL_G1
    with pytest.raises(ValueError, match="UNITREE_G1_SONIC"):
        RtcGr00tPolicy._get_action(
            policy, {}, {"rtc": RtcRequest(RtcWindow(), _committed()).as_mapping()}
        )


def test_rtc_policy_injects_prefix_and_returns_normalized_chunk() -> None:
    from transformers.feature_extraction_utils import BatchFeature

    from mosaic_n17.policy.rtc_policy import RtcGr00tPolicy

    committed = _committed()
    request = _request(j=20, overlap=6, frozen=2)
    seen: dict = {}

    class Collator:
        def __call__(self, processed):
            return BatchFeature(
                data={
                    "inputs": {
                        "state": torch.zeros(1, 1, MODEL_DIM),
                        "embodiment_id": torch.zeros(1, dtype=torch.long),
                    }
                }
            )

    class Model:
        config = SimpleNamespace(max_action_dim=MODEL_DIM)
        device = torch.device("cpu")
        dtype = torch.float32

        def get_action(self, inputs, options=None):
            seen["options"] = options
            seen["action"] = inputs["action"].clone()
            prediction = torch.zeros(1, HORIZON, MODEL_DIM)
            prediction[:, :2] = inputs["action"][:, HORIZON - 6 : HORIZON - 4]
            return {"action_pred": prediction}

    from gr00t.data.embodiment_tags import EmbodimentTag

    policy = object.__new__(RtcGr00tPolicy)
    policy.embodiment_tag = EmbodimentTag.UNITREE_G1_SONIC
    policy.processor = ScaleProcessor()
    policy.collate_fn = Collator()
    policy.model = Model()
    policy.modality_configs = {"state": SimpleNamespace(modality_keys=["projected_gravity"])}
    policy._unbatch_observation = lambda observation: [observation]
    policy._to_vla_step_data = lambda observation: SimpleNamespace(
        states={"projected_gravity": np.zeros((1, 3), np.float32)}
    )

    action, info = RtcGr00tPolicy._get_action(
        policy,
        {"video": {}, "state": {}, "language": {}},
        {"anchor_tick": 20, "rtc": request.as_mapping()},
    )

    assert seen["options"] == request.window.to_model_options()
    assert torch.equal(seen["action"], torch.from_numpy(request.previous_action))
    assert action["motion_token"].shape == (1, HORIZON, 64)
    assert np.asarray(action["right_hand_joints"]).dtype == np.float32
    assert info["action_normalized"].shape == (1, HORIZON, MODEL_DIM)
    assert np.array_equal(info["action_normalized"][:, :2], committed[:, 20:22])


def test_closed_loop_rtc_request_pins_the_old_plans_next_rows() -> None:
    from mosaic_n17.tools.closed_loop_mujoco import _rtc_request

    args = SimpleNamespace(rtc_ramp_rate=6.0)
    # First plan: there is no committed chunk yet, so the window must be skipped
    # instead of forwarding ``previous_action=None`` into the model.
    assert _rtc_request(True, None, 0, 0, 0, args) is None
    assert _rtc_request(False, _committed(), 20, 6, 2, args) is None
    assert _rtc_request(True, _committed(), 20, 6, 6, args) is None  # frozen >= overlap

    request = _rtc_request(True, _committed(), 20, 6, 2, args)
    assert request is not None and request.window.overlap == 6
    prefix = np.asarray(request.previous_action)
    assert prefix.shape == (1, HORIZON, MODEL_DIM)
    # only the tail slots are used by the sampler, and they hold rows [j, j+overlap)
    assert np.array_equal(prefix[0, HORIZON - 6 :], _committed()[0, 20:26])
    assert np.allclose(prefix[0, : HORIZON - 6], 0.0)


def test_closed_loop_decode_fails_closed_on_token_bound() -> None:
    from mosaic_n17.tools.closed_loop_mujoco import _decode_chunk

    action = {
        "motion_token": np.zeros((HORIZON, 64), np.float32),
        "left_hand_joints": np.zeros((HORIZON, 7), np.float32),
        "right_hand_joints": np.zeros((HORIZON, 7), np.float32),
    }
    tokens, hands_l, hands_r = _decode_chunk(action)
    assert tokens.shape == (HORIZON, 64) and hands_l.shape == hands_r.shape == (HORIZON, 7)
    action["motion_token"][0, 0] = 1.5
    with pytest.raises(RuntimeError, match="out of contract"):
        _decode_chunk(action)


class _BasePolicyStub:
    """Duck-typed Gr00tPolicy: only the members the wrapper touches."""

    def __init__(self, model, processor) -> None:
        from transformers.feature_extraction_utils import BatchFeature

        self.model = model
        self.processor = processor

        class _Collator:
            def __call__(self, processed):
                return BatchFeature(
                    data={
                        "inputs": {
                            "state": torch.zeros(1, 1, MODEL_DIM),
                            "embodiment_id": torch.zeros(1, dtype=torch.long),
                        }
                    }
                )

        self.collate_fn = _Collator()

    def _unbatch_observation(self, observation):
        return [observation]

    def _to_vla_step_data(self, observation):
        return SimpleNamespace(states={"projected_gravity": np.zeros((1, 3), np.float32)})

    def reset(self, options=None):
        return {}


def _wrapper(*, K: int, selection: str = "first"):
    from mosaic_n17.policy.policy_wrapper import MosaicPolicyWrapper

    model = RtcModel(RtcHead())
    return MosaicPolicyWrapper(
        _BasePolicyStub(model, ScaleProcessor()),
        K=K,
        selection=selection,
        shadow=False,
    ), model


def test_wrapper_fixed_selection_reuses_the_shared_prefix() -> None:
    committed = _committed()
    wrapper, _model = _wrapper(K=3)
    options = {"anchor_tick": 20, "rtc": _request(j=20, overlap=6, frozen=2).as_mapping()}
    actions, info = wrapper.get_action({"video": {}, "state": {}, "language": {}}, options)
    assert actions["motion_token"].shape == (1, HORIZON, 64)
    assert info["selected_index"] == [0] and info["selection"] == "first"
    assert info["mosaic_decisions"][0]["selected_by"] == "fixed"
    normalized = info["action_normalized"]
    assert normalized.shape == (1, HORIZON, MODEL_DIM)
    assert np.array_equal(normalized[0, :2], committed[0, 20:22])


def test_wrapper_get_candidates_returns_decoded_and_normalized_rows() -> None:
    window = RtcWindow(overlap=20, frozen=2)
    committed = _committed()
    wrapper, _model = _wrapper(K=2)
    payload = wrapper.get_candidates(
        {"video": {}, "state": {}, "language": {}},
        {
            "anchor_tick": 20,
            "rtc": RtcRequest(window=window, previous_action=committed).as_mapping(),
        },
    )
    assert payload["candidates"].shape == (1, 2, HORIZON, 78)
    assert payload["normalized"].shape == (1, 2, HORIZON, MODEL_DIM)
    assert payload["normalized"].dtype == np.float32
    for index in range(2):
        assert np.array_equal(payload["normalized"][0, index, :2], committed[0, 20:22])


def test_wrapper_rejects_a_broken_shared_prefix() -> None:
    from mosaic_n17.policy.policy_wrapper import MosaicPolicyWrapper

    request = _request(j=20, overlap=6, frozen=2)
    normalized = np.zeros((1, 2, HORIZON, MODEL_DIM), np.float32)
    with pytest.raises(RuntimeError, match="prefix mismatch"):
        MosaicPolicyWrapper._assert_shared_prefix(normalized, request)


def test_wrapper_selection_validation() -> None:
    from mosaic_n17.policy.policy_wrapper import MosaicPolicyWrapper

    with pytest.raises(ValueError, match="primary_index"):
        MosaicPolicyWrapper(object(), K=2, selection="first", primary_index=2)
    with pytest.raises(ValueError, match="selection must be"):
        MosaicPolicyWrapper(object(), K=2, selection="random")
    with pytest.raises(ValueError, match="requires a scorer"):
        MosaicPolicyWrapper(object(), K=2, selection="gate")


def test_rtc_server_argument_contract() -> None:
    from mosaic_n17.tools import run_rtc_server

    parser = run_rtc_server.build_parser()
    ok = parser.parse_args(["--ckpt", "x", "--k", "4"])
    run_rtc_server.validate_args(ok)
    for argv in (
        ["--ckpt", "x", "--k", "0"],
        ["--ckpt", "x", "--k", "2", "--k-select", "gate"],
        ["--ckpt", "x", "--k", "1", "--k-select", "gate"],
    ):
        with pytest.raises(SystemExit):
            run_rtc_server.validate_args(parser.parse_args(argv))


def test_sampler_replicates_backbone_masks_for_k() -> None:
    head = RtcHead()
    model = RtcModel(head)
    captured: dict = {}

    original = head.get_action_with_features

    def spy(**kwargs):
        captured["image_mask"] = kwargs["backbone_output"]["image_mask"].shape
        captured["attention_mask"] = kwargs["backbone_output"]["backbone_attention_mask"].shape
        return original(**kwargs)

    head.get_action_with_features = spy
    window = RtcWindow(overlap=20, frozen=2)
    sample_raw_candidates(
        model,
        _inputs(),
        K=3,
        rtc=RtcRequest(window=window, previous_action=_committed()),
    )
    assert captured["image_mask"] == (3, 4)
    assert captured["attention_mask"] == (3, 4)


def test_place_success_predicate_uses_target_footprint_and_top() -> None:
    from mosaic_n17.tools.closed_loop_mujoco import _place_success

    target = {"xy": [1.0, 0.0], "half_xy": [0.2, 0.4], "top_z": 0.45}

    def rows(z, x=1.0, count=25):
        return [{"tracked_bodies": {"can": {"position": [x, 0.0, z]}}} for _ in range(count)]

    assert _place_success(rows(0.50), "can", target)["success"] is True
    assert _place_success(rows(0.54, x=1.15), "can", target)["success"] is True
    assert _place_success(rows(0.20), "can", target)["success"] is False  # on the floor
    assert _place_success(rows(0.50, x=1.5), "can", target)["success"] is False  # outside
    assert _place_success(rows(0.70), "can", target)["success"] is False  # still held high
    moving = rows(0.50)
    moving[0]["tracked_bodies"]["can"]["position"][2] = 0.58  # 8 cm drift -> not resting
    assert _place_success(moving, "can", target)["success"] is False

def test_check_success_routes_place_kind_with_target_frames() -> None:
    from mosaic_n17.tools.closed_loop_mujoco import _check_success

    rows = [
        {"tracked_bodies": {"can": {"position": [1.0, 0.0, 0.50]}}} for _ in range(30)
    ]
    target = {"xy": [1.0, 0.0], "half_xy": [0.2, 0.4], "top_z": 0.45}
    out = _check_success("place:can:table", rows, None, {"table": target})
    assert out["success"] is True and "place" in out["stage"]
