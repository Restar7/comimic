"""Decode N1.7 action tensors into the SONIC78 ``[H, 78]`` layout.

N1.7 emits a normalized, zero-padded ``[H, max_action_dim=132]`` tensor.  The
official processor owns the inverse normalization and the per-group split
(``motion_token`` / ``left_hand_joints`` / ``right_hand_joints``); this module
only reassembles those groups in the frozen order and validates them.  Doing the
split by hand here would risk diverging from ``processor.decode_action``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..contracts.sonic78 import ACTION78_DIM, SONIC78_V1, Sonic78Spec


GROUP_ORDER = ("motion_token", "left_hand_joints", "right_hand_joints")
GROUP_WIDTHS = (64, 7, 7)


def decode_group_dict(
    processor,
    raw_action: object,
    *,
    embodiment_tag: str = "UNITREE_G1_SONIC",
) -> dict[str, np.ndarray]:
    """Call the official ``processor.decode_action`` and return its group dict.

    ``raw_action`` is the model-space tensor (normalized, padded).  The
    processor is duck-typed so this stays testable without a checkpoint.
    """

    array = np.asarray(raw_action, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError(f"raw action must carry (..., H, D) dims, got {array.shape}")
    from gr00t.data.embodiment_tags import EmbodimentTag

    tag = EmbodimentTag.resolve(embodiment_tag)
    groups = processor.decode_action(array, tag)
    if not isinstance(groups, Mapping):
        raise TypeError("processor.decode_action must return a mapping of groups")
    missing = [key for key in GROUP_ORDER if key not in groups]
    if missing:
        raise ValueError(f"decoded action is missing SONIC78 groups {missing}")
    return {key: np.asarray(groups[key]) for key in GROUP_ORDER}


def assemble_sonic78(
    groups: Mapping[str, np.ndarray], *, spec: Sonic78Spec = SONIC78_V1
) -> np.ndarray:
    """Assemble ``motion64 + left7 + right7`` into ``[..., H, 78]`` and validate."""

    arrays = []
    for key, width in zip(GROUP_ORDER, GROUP_WIDTHS):
        if key not in groups:
            raise ValueError(f"missing action group {key!r}")
        value = np.asarray(groups[key], dtype=np.float64)
        if value.shape[-1] != width:
            raise ValueError(f"{key} must end in {width}, got {value.shape}")
        arrays.append(value)
    horizons = {value.shape[-2] for value in arrays}
    if len(horizons) != 1:
        raise ValueError(f"all action groups must share a horizon, got {sorted(horizons)}")
    candidates = np.concatenate(arrays, axis=-1)
    if candidates.shape[-1] != ACTION78_DIM:
        raise ValueError(f"assembled action must end in {ACTION78_DIM}, got {candidates.shape}")
    return spec.validate_block(candidates)


def decode_candidates(
    processor,
    raw_actions: object,
    *,
    spec: Sonic78Spec = SONIC78_V1,
    embodiment_tag: str = "UNITREE_G1_SONIC",
) -> np.ndarray:
    """Decode ``[..., H, max_action_dim]`` model output into ``[..., H, 78]``.

    Leading candidate/batch dimensions are preserved: every row is a *complete*
    78D block, never a recombination of groups from different candidates.
    """

    array = np.asarray(raw_actions, dtype=np.float32)
    if array.ndim < 3:
        raise ValueError(f"raw actions must be (..., K, H, D), got {array.shape}")
    leading = array.shape[:-2]
    flat = array.reshape((-1,) + array.shape[-2:])
    decoded = [decode_group_dict(processor, row, embodiment_tag=embodiment_tag) for row in flat]
    stacked = {
        key: np.stack([item[key] for item in decoded], axis=0).reshape(
            leading + decoded[0][key].shape[-2:]
        )
        for key in GROUP_ORDER
    }
    return assemble_sonic78(stacked, spec=spec)


__all__ = [
    "GROUP_ORDER",
    "GROUP_WIDTHS",
    "assemble_sonic78",
    "decode_candidates",
    "decode_group_dict",
]
