"""Stage frontier adapter and candidate concordance (M7).

The frontier is the *goal* Stage representation ``g_t^S``; concordance measures
how much closer a candidate's predicted Stage slots get to it.  Prediction of
``g_t^S`` belongs to Stage-JEPA (FlowBody ``extensions/stage_jepa``), so this
module takes an explicit callable plus the identity of the weights that produced
it.  Nothing here invents a goal from language text or from the current frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os

import numpy as np

from ..ac.slots import SLOT_DIM, TOTAL_SLOTS


def _sha256_bytes(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def _validate_slots(slots: object, label: str) -> np.ndarray:
    array = np.asarray(slots, dtype=np.float64)
    if array.shape[-2:] != (TOTAL_SLOTS, SLOT_DIM):
        raise ValueError(f"{label} must end in ({TOTAL_SLOTS}, {SLOT_DIM}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return array


@dataclass(frozen=True)
class StageGoal:
    slots: np.ndarray
    frontier_sha256: str
    weights_sha256: str
    source: str = "stage_jepa"

    def __post_init__(self) -> None:
        slots = _validate_slots(self.slots, "goal slots")
        object.__setattr__(self, "slots", slots)
        if not isinstance(self.weights_sha256, str) or len(self.weights_sha256) != 64:
            raise ValueError("weights_sha256 must be a 64-char digest")
        expected = _sha256_bytes(slots)
        if self.frontier_sha256 != expected:
            raise ValueError(
                "frontier_sha256 does not match the goal slots; refusing an unbound frontier"
            )


def build_stage_goal(
    slots: object, *, weights_sha256: str, source: str = "stage_jepa"
) -> StageGoal:
    array = _validate_slots(slots, "goal slots")
    return StageGoal(
        slots=array,
        frontier_sha256=_sha256_bytes(array),
        weights_sha256=weights_sha256,
        source=source,
    )


class StageFrontier:
    """Callable wrapper that binds predicted goals to their weights digest."""

    def __init__(self, predict_fn, *, weights_sha256: str, source: str = "stage_jepa") -> None:
        if not callable(predict_fn):
            raise TypeError("predict_fn must be callable")
        if not isinstance(weights_sha256, str) or len(weights_sha256) != 64:
            raise ValueError("weights_sha256 must be a 64-char digest")
        self._predict_fn = predict_fn
        self.weights_sha256 = weights_sha256
        self.source = source

    def goal(self, current_slots: object, language_embedding: object | None = None) -> StageGoal:
        current = _validate_slots(current_slots, "current slots")
        predicted = self._predict_fn(current, language_embedding)
        return build_stage_goal(predicted, weights_sha256=self.weights_sha256, source=self.source)

    @classmethod
    def from_flowbody(
        cls,
        flowbody_root: str | os.PathLike[str],
        checkpoint_path: str | os.PathLike[str],
    ) -> "StageFrontier":
        """Load Stage-JEPA's frontier predictor from a FlowBody checkpoint."""

        package_root = os.path.join(os.path.abspath(os.fspath(flowbody_root)), "flowbody")
        import sys

        if package_root not in sys.path:
            sys.path.insert(0, package_root)
        from extensions.stage_jepa.model import StageJEPAPredictor  # type: ignore[import-not-found]
        import torch  # noqa: F401  (kept local so the module stays import-light)

        checkpoint = os.path.abspath(os.fspath(checkpoint_path))
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError("Stage-JEPA checkpoint is missing its model state dict")
        architecture = payload.get("architecture")
        if not isinstance(architecture, dict):
            raise ValueError(
                "Stage-JEPA checkpoint must carry its constructor kwargs under 'architecture'"
            )
        model = StageJEPAPredictor(**architecture)
        model.load_state_dict(payload["model"])
        model.eval()
        digest = hashlib.sha256(open(checkpoint, "rb").read()).hexdigest()

        def predict_fn(current: np.ndarray, language_embedding: object | None) -> np.ndarray:
            if language_embedding is None:
                raise ValueError(
                    "Stage-JEPA needs an instruction embedding; refusing to fabricate one"
                )
            with torch.no_grad():
                tokens = torch.from_numpy(np.asarray(current, dtype=np.float32))
                instruction = torch.from_numpy(np.asarray(language_embedding, dtype=np.float32))
                if instruction.ndim == 1:
                    instruction = instruction.unsqueeze(0)
                output = model(tokens, instruction)
            array = np.asarray(output, dtype=np.float64)
            if array.shape[-2:] != (TOTAL_SLOTS, SLOT_DIM):
                raise ValueError(
                    f"Stage-JEPA must be configured for the typed-slot readout "
                    f"({TOTAL_SLOTS}, {SLOT_DIM}), got {array.shape}"
                )
            return array

        return cls(predict_fn, weights_sha256=digest, source="stage_jepa")


__all__ = ["StageFrontier", "StageGoal", "build_stage_goal"]
