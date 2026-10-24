"""Inference-time RTC window for the SONIC78 route (zero-shot, no retraining).

The official sampler already implements real-time chunking inpainting
(``gr00t/model/gr00t_n1d7/gr00t_n1d7.py:358-394``): it pins the new chunk's
first ``overlap`` rows to the **last** ``overlap`` rows of the tensor passed as
``action_input["action"]``.  That API therefore requires an *aligned* prefix:
when the controller replans after ``j`` executed ticks, the tensor handed to the
model must place the old plan's rows ``[j, j+overlap)`` in its last ``overlap``
rows (the rest is padding the model never reads).  ``overlap`` is a latency
budget (a handful of ticks), **not** the whole remaining tail: pinning
``H - j`` rows would make every executed tick a replay of the old plan's
never-executed future.  The first ``frozen`` ticks never move, ``[frozen,
overlap)`` are denoised with an exponential ramp, and the rest is free.  This module owns the *contract* around that behaviour and nothing
else -- it never samples, decodes, clips or retrains.

Why a separate module:

* the window has invariants that must fail closed (an out-of-range prefix
  changes commands that were already sent);
* the model consumes a **normalized, zero-padded** ``[B, H, max_action_dim]``
  action, never the deployed 78D token block.  Encoding the committed chunk is
  the only place that may call the official normalizer;
* sim and real deployment disagree on what ``frozen`` means: the MuJoCo loop is
  synchronous (inference costs wall-clock, not sim ticks), so ``frozen`` is a
  continuity parameter there, while on a real 50 Hz controller it must be
  ``ceil(latency / 20 ms) + 1`` in *control ticks*.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..contracts.sonic78 import ACTION78_DIM, SONIC78_V1
from ..contracts.timebase import TickSchedule


#: The model needs at least one tick that is only partially frozen; a window
#: with ``overlap - frozen < 2`` degenerates into "freeze the whole overlap".
MIN_TRANSITION_TICKS = 2
#: Upper bound for the exponential ramp rate; larger values behave like a step.
MAX_RAMP_RATE = 32.0
#: The four option keys the action head reads (``gr00t_n1d7.py:365-368``).
MODEL_OPTION_KEYS = (
    "action_horizon",
    "rtc_overlap_steps",
    "rtc_frozen_steps",
    "rtc_ramp_rate",
)
REQUEST_KEYS = ("previous_action", "horizon", "overlap", "frozen", "ramp_rate", "action_horizon")


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an int, got {value!r}")
    return value


@dataclass(frozen=True)
class RtcWindow:
    """One RTC inpainting window, validated at construction time.

    ``horizon`` is the action horizon of both chunks, ``overlap`` is ``O`` (the
    old chunk's unexecuted tail length), ``frozen`` is ``F`` (ticks that must not
    move) and ``ramp_rate`` is forwarded verbatim to the model.
    """

    horizon: int = SONIC78_V1.horizon
    overlap: int = SONIC78_V1.execute_steps
    frozen: int = 0
    ramp_rate: float = 6.0
    action_horizon: int = SONIC78_V1.horizon

    def __post_init__(self) -> None:
        self.validate()

    def validate(self, schedule: TickSchedule | None = None) -> None:
        """Fail closed on every window that cannot be executed honestly."""

        schedule = TickSchedule.from_spec() if schedule is None else schedule
        for label in ("horizon", "overlap", "frozen", "action_horizon"):
            _require_int(getattr(self, label), label)
        if not np.isfinite(self.ramp_rate):
            raise ValueError("ramp_rate must be finite")
        if self.horizon != schedule.plan_horizon:
            raise ValueError(
                f"horizon must equal the control plan horizon {schedule.plan_horizon}, "
                f"got {self.horizon}"
            )
        if self.action_horizon != self.horizon:
            raise ValueError(
                "action_horizon must equal horizon: the model slices the previous tail with "
                f"[{self.horizon} - O, {self.horizon}), got action_horizon={self.action_horizon}"
            )
        if not 0 < self.overlap <= self.horizon:
            raise ValueError(f"overlap must be in (0, {self.horizon}], got {self.overlap}")
        if not 0 <= self.frozen < self.overlap:
            raise ValueError(f"frozen must be in [0, {self.overlap}), got {self.frozen}")
        if self.overlap - self.frozen < MIN_TRANSITION_TICKS:
            raise ValueError(
                f"overlap - frozen must be >= {MIN_TRANSITION_TICKS} (got "
                f"{self.overlap - self.frozen}); otherwise the whole overlap is frozen"
            )
        if not 0.0 < self.ramp_rate <= MAX_RAMP_RATE:
            raise ValueError(f"ramp_rate must be in (0, {MAX_RAMP_RATE}], got {self.ramp_rate}")

    def to_model_options(self) -> dict[str, float | int]:
        """The four keys forwarded to ``action_head.get_action_with_features``."""

        return {
            "action_horizon": self.action_horizon,
            "rtc_overlap_steps": self.overlap,
            "rtc_frozen_steps": self.frozen,
            "rtc_ramp_rate": float(self.ramp_rate),
        }

    @classmethod
    def from_model_options(cls, options: dict) -> "RtcWindow":
        """Rebuild a window from the model-facing options (strict key set)."""

        if not isinstance(options, dict):
            raise TypeError(f"rtc options must be a dict, got {type(options).__name__}")
        unknown = set(options) - set(MODEL_OPTION_KEYS)
        missing = set(MODEL_OPTION_KEYS) - set(options)
        if missing or unknown:
            raise ValueError(
                f"rtc model options must be exactly {list(MODEL_OPTION_KEYS)}; "
                f"missing={sorted(missing)} unknown={sorted(unknown)}"
            )
        return cls(
            horizon=_require_int(options["action_horizon"], "action_horizon"),
            action_horizon=_require_int(options["action_horizon"], "action_horizon"),
            overlap=_require_int(options["rtc_overlap_steps"], "rtc_overlap_steps"),
            frozen=_require_int(options["rtc_frozen_steps"], "rtc_frozen_steps"),
            ramp_rate=float(options["rtc_ramp_rate"]),
        )

    @classmethod
    def from_request(cls, request: dict) -> "RtcWindow":
        """Rebuild a window from a policy request ``options["rtc"]`` (strict key set)."""

        if not isinstance(request, dict):
            raise TypeError(f"rtc request must be a dict, got {type(request).__name__}")
        unknown = set(request) - set(REQUEST_KEYS)
        missing = set(REQUEST_KEYS) - set(request)
        if missing or unknown:
            raise ValueError(
                f"rtc request must be exactly {list(REQUEST_KEYS)}; "
                f"missing={sorted(missing)} unknown={sorted(unknown)}"
            )
        return cls(
            horizon=_require_int(request["horizon"], "horizon"),
            action_horizon=_require_int(request["action_horizon"], "action_horizon"),
            overlap=_require_int(request["overlap"], "overlap"),
            frozen=_require_int(request["frozen"], "frozen"),
            ramp_rate=float(request["ramp_rate"]),
        )

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class RtcRequest:
    """A committed chunk plus the window it should be reused with.

    ``previous_action`` is kept as a tensor/array because the caller owns the
    model width (132 for this checkpoint); :func:`validate_prefix_tensor` is the
    single place that checks it.
    """

    window: RtcWindow
    previous_action: object

    def as_mapping(self) -> dict:
        return {"previous_action": self.previous_action, **self.window.as_dict()}

    @classmethod
    def from_mapping(cls, request: dict) -> "RtcRequest":
        window = RtcWindow.from_request(request)
        return cls(window=window, previous_action=request["previous_action"])


def validate_prefix_tensor(prefix: object, window: RtcWindow, *, action_dim: int) -> np.ndarray:
    """Validate a committed chunk in model space and return it as float32.

    ``prefix`` must be ``[B, H, action_dim]`` in the *normalized* action space.
    A decoded 78D token block is rejected: the width check alone is enough to
    catch the common mistake, and a 132-wide raw block is still caught by the
    finite/normalized-space callers that own the statistics.
    """

    if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim <= 0:
        raise ValueError(f"action_dim must be a positive int, got {action_dim!r}")
    array = np.asarray(prefix, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"prefix must be [B, H, {action_dim}], got {array.shape}")
    if array.shape[2] == ACTION78_DIM and action_dim != ACTION78_DIM:
        raise ValueError(
            "prefix looks like a decoded 78D token block; the model needs the normalized "
            "zero-padded action (width = max_action_dim)"
        )
    if array.shape[1:] != (window.horizon, action_dim):
        raise ValueError(f"prefix must be [B, {window.horizon}, {action_dim}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("prefix must be finite")
    return array


def align_prefix_tail(committed: object, *, start: int, overlap: int, horizon: int) -> np.ndarray:
    """Place the old plan's rows ``[start, start+overlap)`` in the model's tail slots.

    The official sampler reads ``action_input["action"][:, horizon - overlap : horizon]``
    and pins the new chunk's first ``overlap`` rows to those values.  With
    ``start = j`` (executed ticks) the pinned rows are exactly the ticks the old
    plan would execute next, so the seam is continuous while rows
    ``[overlap, execute_steps)`` stay freshly denoised.
    """

    array = np.asarray(committed, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"committed chunk must be [B, H, D], got {array.shape}")
    if array.shape[1] != horizon:
        raise ValueError(f"committed chunk must have horizon {horizon}, got {array.shape}")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError(f"start must be a non-negative int, got {start!r}")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 < overlap <= horizon:
        raise ValueError(f"overlap must be in (0, {horizon}], got {overlap!r}")
    if start + overlap > horizon:
        raise ValueError(
            f"rows [{start}, {start + overlap}) exceed the committed horizon {horizon}"
        )
    tail = array[:, start : start + overlap, :]
    if not np.isfinite(tail).all():
        raise ValueError("committed tail must be finite")
    aligned = np.zeros_like(array)
    aligned[:, horizon - overlap : horizon, :] = tail
    return aligned


def _resolve_embodiment(embodiment_tag: object) -> str:
    """Resolve an ``EmbodimentTag`` (or its name/value) to the processor's string key."""

    from gr00t.data.embodiment_tags import EmbodimentTag

    tag = (
        embodiment_tag
        if isinstance(embodiment_tag, EmbodimentTag)
        else EmbodimentTag.resolve(embodiment_tag)
    )
    return tag.value


def encode_committed_action(
    processor,
    raw_action: np.ndarray,
    *,
    window: RtcWindow,
    embodiment_tag: str = "UNITREE_G1_SONIC",
    state: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Encode a committed 78D chunk back into the model's normalized padded space.

    This is the fallback path (restart, replay, cross-process): whenever the
    sampler's own ``action_pred`` is still available it must be reused instead,
    because re-encoding introduces a float round trip.

    The normalization is the official one -- ``q01``/``q99`` min-max with
    clipping for this checkpoint -- so the same ``StateActionProcessor`` instance
    that decoded the action has to encode it.
    """

    from ..contracts.sonic78 import SONIC78_V1

    tag = _resolve_embodiment(embodiment_tag)
    array = np.asarray(raw_action, dtype=np.float64)
    if array.ndim == 1:
        raise ValueError(f"raw action must be [H, {ACTION78_DIM}], got {array.shape}")
    if array.shape[-1] != ACTION78_DIM:
        raise ValueError(f"raw action must end in {ACTION78_DIM}, got {array.shape}")
    if array.shape[-2] != window.horizon:
        raise ValueError(f"raw action must carry horizon {window.horizon}, got {array.shape}")
    SONIC78_V1.validate_block(array)
    batched = array if array.ndim == 3 else array[None]
    groups = {
        "motion_token": batched[..., :64],
        "left_hand_joints": batched[..., 64:71],
        "right_hand_joints": batched[..., 71:78],
    }
    normalized = processor.state_action_processor.apply_action(groups, tag, state=state)
    keys = processor.modality_configs[tag]["action"].modality_keys
    missing = [key for key in keys if key not in normalized]
    if missing:
        raise ValueError(f"processor did not normalize groups {missing}")
    concatenated = np.concatenate([np.asarray(normalized[key]) for key in keys], axis=-1)
    model_dim = int(processor.max_action_dim)
    if concatenated.shape[-1] > model_dim:
        raise ValueError(
            f"normalized action width {concatenated.shape[-1]} exceeds max_action_dim {model_dim}"
        )
    padded = np.zeros(concatenated.shape[:-1] + (model_dim,), dtype=np.float32)
    padded[..., : concatenated.shape[-1]] = concatenated
    if not np.isfinite(padded).all():
        raise ValueError("normalized action must be finite")
    return padded


def prefix_equal(
    committed: np.ndarray, new_chunk: np.ndarray, window: RtcWindow, *, atol: float = 0.0
) -> bool:
    """Compare the frozen prefix of a new chunk with the committed tail.

    The model pins ``new[:, :F]`` to ``committed[:, H-O : H-O+F]``; anything else
    means the two tensors were not produced by the same normalizer / commit.
    """

    left = np.asarray(committed, dtype=np.float32)
    right = np.asarray(new_chunk, dtype=np.float32)
    if left.ndim != right.ndim or left.shape[-2:] != right.shape[-2:]:
        return False
    start = window.horizon - window.overlap
    tail = left[..., start : start + window.frozen, :]
    head = right[..., : window.frozen, :]
    if atol == 0.0:
        return bool(np.array_equal(tail, head))
    return bool(np.allclose(tail, head, rtol=0.0, atol=atol))


__all__ = [
    "MAX_RAMP_RATE",
    "MIN_TRANSITION_TICKS",
    "MODEL_OPTION_KEYS",
    "REQUEST_KEYS",
    "RtcRequest",
    "align_prefix_tail",
    "RtcWindow",
    "encode_committed_action",
    "prefix_equal",
    "validate_prefix_tensor",
]
