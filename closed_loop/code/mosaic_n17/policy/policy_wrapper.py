"""Deployment wrapper: K candidates -> (RTC prefix) -> typed scorer -> gate/shadow.

``Gr00tPolicy._get_action`` hard-codes a single model call and does not forward
``options`` (``gr00t/policy/gr00t_policy.py:417``), so both the RTC prefix and the
K-candidate path live here as a ``PolicyWrapper`` (the same extension point the
repository uses for ``Gr00tSimPolicyWrapper``) instead of editing the official
policy.

Two selection modes:

* ``"gate"`` -- the existing behaviour: a ``scorer`` is mandatory for ``K > 1``
  and ``decide``/``run_shadow_step`` arbitrate.  The score terms come from the
  AC; as long as Phase B is not better than copy-current this mode must not be
  used online.
* ``"first"`` -- fixed selection of ``primary_index`` without a scorer, for
  exercising K sampling and shared RTC prefixes before the AC is calibrated.  It
  still validates the candidate block and the shared prefix; it never invents a
  score.

The wrapper never commits anything implicitly: ``get_action`` returns the
selected row's normalized chunk in ``info["action_normalized"]`` and the caller
decides what becomes the next committed plan.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from ..contracts.sonic78 import SONIC78_V1, Sonic78Spec
from .candidate_sampler import CandidateSet, sample_candidates_and_normalized
from .decision_gate import GateConfig, GateDecision, decide
from .rtc import RtcRequest, validate_prefix_tensor
from .shadow import run_shadow_step


ScoreFn = Callable[[CandidateSet, dict[str, np.ndarray]], dict[str, np.ndarray]]


class MosaicPolicyWrapper:
    """Wrap a ``Gr00tPolicy`` with K sampling, RTC prefixes, arbitration and shadow logging."""

    def __init__(
        self,
        policy,
        *,
        K: int = 4,
        scorer: ScoreFn | None = None,
        selection: str = "gate",
        gate_config: GateConfig = GateConfig(),
        shadow: bool = True,
        primary_index: int = 0,
        spec: Sonic78Spec = SONIC78_V1,
    ) -> None:
        if isinstance(K, bool) or not isinstance(K, int) or K <= 0:
            raise ValueError(f"K must be a positive int, got {K!r}")
        if selection not in ("gate", "first"):
            raise ValueError(f"selection must be 'gate' or 'first', got {selection!r}")
        if selection == "gate" and K > 1 and scorer is None:
            raise ValueError("K>1 requires a scorer; refusing to rank candidates without one")
        if selection == "first" and not 0 <= int(primary_index) < K:
            raise ValueError(f"primary_index must be in [0, {K}), got {primary_index!r}")
        self.policy = policy
        self.K = K
        self.scorer = scorer
        self.selection = selection
        self.gate_config = gate_config
        self.shadow = bool(shadow)
        self.primary_index = int(primary_index)
        self.spec = spec
        self.last_decisions: list[dict[str, Any]] = []

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.policy.reset(options)

    def _collated_inputs(self, observation: dict[str, Any]):
        from gr00t.policy.gr00t_policy import MessageType, _rec_to_dtype

        unbatched = self.policy._unbatch_observation(observation)
        processed, states = [], []
        for obs in unbatched:
            step = self.policy._to_vla_step_data(obs)
            states.append(step.states)
            processed.append(
                self.policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
            )
        collated = self.policy.collate_fn(processed)
        return _rec_to_dtype(collated, dtype=torch.bfloat16)["inputs"], states

    def _rtc_request(self, options: dict[str, Any] | None) -> RtcRequest | None:
        if options is None:
            return None
        unknown = set(options) - {"rtc", "anchor_tick"}
        if unknown:
            raise ValueError(f"unsupported policy options {sorted(unknown)}")
        anchor = options.get("anchor_tick")
        if anchor is not None and (
            isinstance(anchor, bool) or not isinstance(anchor, int) or anchor < 0
        ):
            raise ValueError(f"anchor_tick must be a non-negative int, got {anchor!r}")
        raw_request = options.get("rtc")
        if raw_request is None:
            return None
        request = RtcRequest.from_mapping(raw_request)
        request.window.validate()
        validate_prefix_tensor(
            request.previous_action,
            request.window,
            action_dim=int(self.policy.model.config.max_action_dim),
        )
        return request

    @staticmethod
    def _assert_shared_prefix(normalized: np.ndarray, request: RtcRequest) -> None:
        """Every candidate must carry the committed tail, bit for bit."""

        window = request.window
        if window.frozen == 0:
            return
        previous = np.asarray(request.previous_action, dtype=np.float32)
        start = window.horizon - window.overlap
        expected = previous[:, start : start + window.frozen, :]
        head = normalized[:, :, : window.frozen, :]
        if not np.allclose(head, expected[:, None, :, :], rtol=0.0, atol=0.0):
            raise RuntimeError(
                "RTC prefix mismatch: candidate frozen heads differ from the committed tail "
                f"at overlap {window.overlap} / frozen {window.frozen}"
            )

    def _select_action_dict(
        self, candidates: CandidateSet, rows: list[int]
    ) -> dict[str, np.ndarray]:
        """Split the *already decoded* 78D rows into the official action groups.

        The processor un-normalizes exactly once (inside ``sample_candidates``);
        re-decoding here would apply the inverse normalization twice.
        """

        selected = np.stack(
            [candidates.candidates[index, row] for index, row in enumerate(rows)], axis=0
        ).astype(np.float32)
        return {
            "motion_token": selected[..., :64],
            "left_hand_joints": selected[..., 64:71],
            "right_hand_joints": selected[..., 71:78],
        }

    @staticmethod
    def _selected_normalized(normalized: np.ndarray, rows: list[int]) -> np.ndarray:
        return normalized[np.arange(normalized.shape[0]), np.asarray(rows)]

    def _gate_choice(self, candidate_set: CandidateSet, states) -> list[int]:
        selected_rows: list[int] = []
        self.last_decisions = []
        for index in range(candidate_set.batch_size):
            single = CandidateSet(
                candidates=candidate_set.candidates[index : index + 1],
                motion_valid=candidate_set.motion_valid[index : index + 1],
                hand_valid=candidate_set.hand_valid[index : index + 1],
                seeds=candidate_set.seeds,
                token_sha256=candidate_set.token_sha256[index : index + 1],
            )
            components = self.scorer(single, states[index]) if self.scorer is not None else None
            decision: GateDecision = decide(
                single.candidates[0], components=components, config=self.gate_config, spec=self.spec
            )
            if self.shadow:
                executed, record = run_shadow_step(
                    single, decision, primary_index=self.primary_index, spec=self.spec
                )
                if executed is None:
                    raise RuntimeError(
                        "the primary candidate is inadmissible; keep the previous verified plan"
                    )
                chosen = self.primary_index
            else:
                if decision.fallback:
                    raise RuntimeError("no admissible candidate; keep the previous verified plan")
                chosen = decision.index
                record = None
            selected_rows.append(chosen)
            self.last_decisions.append(
                {
                    "index": chosen,
                    "gate_index": decision.index,
                    "gate_reason": decision.reason,
                    "margin": decision.margin,
                    "scores": decision.scores.tolist(),
                    "shadow": self.shadow,
                    "selected_by": "gate",
                    "agrees": None if record is None else record.agrees,
                }
            )
        return selected_rows

    def _choose(self, candidate_set: CandidateSet, states) -> list[int]:
        if self.selection == "gate":
            return self._gate_choice(candidate_set, states)
        self.last_decisions = [
            {"index": self.primary_index, "selected_by": "fixed"}
            for _ in range(candidate_set.batch_size)
        ]
        return [self.primary_index] * candidate_set.batch_size

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request = self._rtc_request(options)
        inputs, states = self._collated_inputs(observation)
        candidate_set, normalized = sample_candidates_and_normalized(
            self.policy.model,
            self.policy.processor,
            inputs,
            K=self.K,
            rtc=request,
            spec=self.spec,
        )
        if request is not None:
            self._assert_shared_prefix(normalized, request)
        selected_rows = self._choose(candidate_set, states)
        actions = self._select_action_dict(candidate_set, selected_rows)
        spread = float(np.max(np.ptp(normalized, axis=1))) if self.K > 1 else 0.0
        return actions, {
            "mosaic_decisions": self.last_decisions,
            "action_normalized": self._selected_normalized(normalized, selected_rows),
            "selected_index": selected_rows,
            "candidate_spread": spread,
            "selection": self.selection,
        }

    def get_candidates(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Sample K candidates for one observation without selecting or committing."""

        request = self._rtc_request(options)
        inputs, _states = self._collated_inputs(observation)
        candidate_set, normalized = sample_candidates_and_normalized(
            self.policy.model,
            self.policy.processor,
            inputs,
            K=self.K,
            rtc=request,
            spec=self.spec,
        )
        if request is not None:
            self._assert_shared_prefix(normalized, request)
        return {
            "candidates": candidate_set.candidates.astype(np.float32),
            "normalized": normalized.astype(np.float32),
            "candidate_ids": [f"c{index}" for index in range(self.K)],
            "seeds": list(candidate_set.seeds),
            "selection": self.selection,
            "anchor_tick": None if options is None else options.get("anchor_tick"),
        }


__all__ = ["MosaicPolicyWrapper", "ScoreFn"]
