"""SONIC78 action contract: ``motion_token64 + left_hand7 + right_hand7``.

This is the N1.7-side port of ``flowbody/mosaic_sonic78/contract.py``.  The 64
motion channels are one latent control token, not joints: nothing here may
invent per-channel joint limits, token velocities or accelerations.  The three
hand groups are the only per-joint semantics.

Every function fails closed: wrong layout, non-finite values, out-of-bound
tokens, placeholder provenance and inconsistent masks raise instead of being
clipped or repaired.  A clipped token would silently change the command sent to
the SONIC decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np


MOTION_TOKEN_DIM = 64
LEFT_HAND_DIM = 7
RIGHT_HAND_DIM = 7
ACTION78_DIM = 78
MOTION_SLICE = slice(0, MOTION_TOKEN_DIM)
LEFT_HAND_SLICE = slice(MOTION_TOKEN_DIM, MOTION_TOKEN_DIM + LEFT_HAND_DIM)
RIGHT_HAND_SLICE = slice(MOTION_TOKEN_DIM + LEFT_HAND_DIM, ACTION78_DIM)
#: The official VLA rejects ``|motion_token| > 1.25``; this is an asset bound,
#: not a physical joint limit.
TOKEN_ABS_BOUND = 1.25

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _finite_float_array(value: object, label: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - numpy already raises
        raise ValueError(f"{label} is not array-like: {exc}") from exc
    if array.dtype == object or array.dtype.kind not in "fiu":
        raise TypeError(f"{label} must be numeric, got dtype {array.dtype}")
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return array


def _bool_mask(mask: object, action: np.ndarray) -> np.ndarray:
    raw = np.asarray(mask)
    if raw.dtype != np.bool_:
        if raw.dtype.kind not in "fiu":
            raise TypeError(f"action mask must be boolean/integer/float, got {raw.dtype}")
        values = raw.astype(np.float64)
        if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
            raise ValueError("float action mask must contain only 0/1")
        raw = raw != 0
    try:
        return np.broadcast_to(raw, action.shape).copy()
    except ValueError as exc:
        raise ValueError(f"mask {raw.shape} does not broadcast to action {action.shape}") from exc


@dataclass(frozen=True)
class Sonic78Spec:
    """Versioned, immutable action contract for one SONIC deployment variant."""

    name: str = "SONIC78_V1"
    action_dim: int = ACTION78_DIM
    horizon: int = 40
    execute_steps: int = 20
    control_hz: int = 50
    encoder_sha256: str = ""
    decoder_sha256: str = ""
    observation_config_sha256: str = ""
    normalizer_sha256: str = ""

    def __post_init__(self) -> None:
        expected = {
            "SONIC78_V1": (40, 20),
            "SONIC78_LOW_LATENCY": (20, 10),
        }
        if self.name not in expected:
            raise ValueError(f"unsupported action contract {self.name!r}")
        if (self.horizon, self.execute_steps) != expected[self.name]:
            raise ValueError(
                f"{self.name} requires horizon/execute_steps={expected[self.name]}, "
                f"got {(self.horizon, self.execute_steps)}"
            )
        if self.action_dim != ACTION78_DIM:
            raise ValueError(f"SONIC78 must have action_dim={ACTION78_DIM}, got {self.action_dim}")
        if self.control_hz != 50:
            raise ValueError(f"SONIC78 executes at 50 Hz, got {self.control_hz}")
        if not 0 < self.execute_steps <= self.horizon:
            raise ValueError("execute_steps must be in (0, horizon]")
        if self.horizon % self.execute_steps:
            raise ValueError("horizon must be an integer multiple of execute_steps")

    @property
    def replan_hz(self) -> float:
        """VLA replanning rate; the decoder/token rate is always ``control_hz``."""

        return self.control_hz / self.execute_steps

    @property
    def slices(self) -> dict[str, slice]:
        return {
            "motion": MOTION_SLICE,
            "left_hand": LEFT_HAND_SLICE,
            "right_hand": RIGHT_HAND_SLICE,
        }

    def require_assets(self) -> None:
        """Fail closed unless every locked asset digest is present."""

        for field in (
            "encoder_sha256",
            "decoder_sha256",
            "observation_config_sha256",
            "normalizer_sha256",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"{self.name} requires a locked {field}")

    def split(self, action: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        array = _finite_float_array(action, "action")
        if array.shape[-1] != ACTION78_DIM:
            raise ValueError(f"action must end in {ACTION78_DIM}, got {array.shape}")
        return (
            array[..., MOTION_SLICE],
            array[..., LEFT_HAND_SLICE],
            array[..., RIGHT_HAND_SLICE],
        )

    def validate(
        self,
        action: object,
        mask: object | None = None,
        *,
        provenance: str | None = None,
        require_provenance: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validate one ``[..., 78]`` block; returns ``(motion, left, right, motion_ok, hand_ok)``.

        ``motion_ok`` is all-or-none across the 64 channels (a partially valid
        latent token is meaningless), ``hand_ok`` is per channel.
        """

        array = _finite_float_array(action, "action")
        if array.shape[-1] != ACTION78_DIM:
            detail = " (a 49D block is a different contract)" if array.shape[-1] == 49 else ""
            raise ValueError(f"action must end in {ACTION78_DIM}, got {array.shape}{detail}")
        if array.ndim < 2:
            raise ValueError(f"action must carry a horizon dimension, got {array.shape}")

        resolved = _bool_mask(mask, array) if mask is not None else np.ones(array.shape, dtype=bool)
        motion_ok = resolved[..., MOTION_SLICE]
        if not ((motion_ok.all(axis=-1)) | (~motion_ok.any(axis=-1))).all():
            raise ValueError("motion token mask must be all-or-none across the 64 channels")
        hand_ok = resolved[..., LEFT_HAND_SLICE.start :]

        if require_provenance and provenance is None:
            raise ValueError("this action requires an explicit provenance tag")
        if provenance is not None:
            from .provenance import PLACEHOLDER_PROVENANCE, VALID_PROVENANCE

            if provenance == PLACEHOLDER_PROVENANCE:
                raise ValueError("placeholder actions are never valid supervision or commands")
            if require_provenance and provenance not in VALID_PROVENANCE:
                raise ValueError(
                    f"provenance {provenance!r} is not one of {sorted(VALID_PROVENANCE)}"
                )

        valid_motion_values = array[..., MOTION_SLICE][motion_ok]
        if valid_motion_values.size and np.abs(valid_motion_values).max() > TOKEN_ABS_BOUND:
            raise ValueError(
                f"motion token exceeds |x|<={TOKEN_ABS_BOUND}; the official VLA rejects it"
            )
        return (
            array[..., MOTION_SLICE],
            array[..., LEFT_HAND_SLICE],
            array[..., RIGHT_HAND_SLICE],
            motion_ok.all(axis=-1),
            hand_ok,
        )

    def validate_block(
        self,
        action: object,
        mask: object | None = None,
        *,
        provenance: str | None = None,
        require_provenance: bool = False,
    ) -> np.ndarray:
        """Validate an ``[..., horizon, 78]`` block and return its canonical copy."""

        array = _finite_float_array(action, "action")
        if array.ndim < 2 or array.shape[-2] != self.horizon:
            raise ValueError(
                f"{self.name} blocks must carry horizon {self.horizon}, got {array.shape}"
            )
        self.validate(array, mask, provenance=provenance, require_provenance=require_provenance)
        return array


SONIC78_V1 = Sonic78Spec()
SONIC78_LOW_LATENCY = Sonic78Spec(name="SONIC78_LOW_LATENCY", horizon=20, execute_steps=10)

__all__ = [
    "ACTION78_DIM",
    "LEFT_HAND_DIM",
    "LEFT_HAND_SLICE",
    "MOTION_SLICE",
    "MOTION_TOKEN_DIM",
    "RIGHT_HAND_DIM",
    "RIGHT_HAND_SLICE",
    "SONIC78_LOW_LATENCY",
    "SONIC78_V1",
    "TOKEN_ABS_BOUND",
    "Sonic78Spec",
]
