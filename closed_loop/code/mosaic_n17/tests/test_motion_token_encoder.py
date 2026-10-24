"""Layout pins for the frozen SONIC encoder input (CPU only, no weights)."""

from __future__ import annotations

import numpy as np
import pytest

from mosaic_n17.tools.convert_motion_tokens import body_reference, rpy_to_quat_wxyz
from mosaic_n17.tools.motion_token_encoder import (
    ENCODER_INPUT_DIM,
    MUJOCO_TO_ISAACLAB,
    assemble_g1_input,
    relative_orientation_6d,
    rotation_6d,
)


def _yaw_matrix(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.asarray(((cos, -sin, 0.0), (sin, cos, 0.0), (0.0, 0.0, 1.0)))


def test_rotation_6d_is_row_major_over_the_first_two_matrix_columns() -> None:
    matrix = _yaw_matrix(0.5)
    assert rotation_6d(matrix) == pytest.approx(
        [matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1], matrix[2, 0], matrix[2, 1]]
    )
    # A stacked-column flatten would give [R00, R10, R20, R01, R11, R21] here.
    assert rotation_6d(matrix) != pytest.approx(
        [matrix[0, 0], matrix[1, 0], matrix[2, 0], matrix[0, 1], matrix[1, 1], matrix[2, 1]]
    )


def test_relative_orientation_6d_is_identity_for_equal_quaternions() -> None:
    quat = np.asarray((np.cos(0.3), np.sin(0.3), 0.0, 0.0))
    future = np.repeat(quat[None], 4, axis=0)
    identity_6d = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    np.testing.assert_allclose(relative_orientation_6d(future, quat), np.tile(identity_6d, (4, 1)))


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def test_relative_orientation_6d_is_world_frame_invariant() -> None:
    base = np.asarray((np.cos(0.2), 0.0, np.sin(0.2), 0.0))
    future = np.asarray((np.cos(0.4), 0.0, 0.0, np.sin(0.4)))
    shift = np.asarray((np.cos(0.15), np.sin(0.15), 0.0, 0.0))
    relative = relative_orientation_6d(future[None], base)[0]
    shifted = relative_orientation_6d(
        _quat_mul(shift, future)[None], _quat_mul(shift, base)
    )[0]
    np.testing.assert_allclose(relative, shifted, atol=1e-12)


def test_encoder_input_offsets_match_the_pinned_release_layout() -> None:
    q = np.arange(10 * 29, dtype=np.float64).reshape(10, 29) / 1000.0
    dq = -q
    anchor = np.arange(10 * 6, dtype=np.float64).reshape(10, 6) / 1000.0
    encoder_input = assemble_g1_input(q, dq, anchor)
    assert encoder_input.shape == (1, ENCODER_INPUT_DIM) == (1, 1762)
    assert float(encoder_input[0, 0]) == 0.0
    assert float(encoder_input[0, 1]) == 1.0
    np.testing.assert_allclose(encoder_input[0, 4:294], q.reshape(-1))
    np.testing.assert_allclose(encoder_input[0, 294:584], dq.reshape(-1))
    np.testing.assert_allclose(encoder_input[0, 601:661], anchor.reshape(-1))
    reserved = np.concatenate(
        (encoder_input[0, 2:4], encoder_input[0, 584:601], encoder_input[0, 661:])
    )
    assert float(np.abs(reserved).max(initial=0.0)) == 0.0


def test_mujoco_to_isaaclab_is_the_pinned_policy_permutation() -> None:
    assert len(MUJOCO_TO_ISAACLAB) == 29
    assert sorted(MUJOCO_TO_ISAACLAB) == list(range(29))
    # waist_yaw / waist_roll / waist_pitch slots from ``policy_parameters.hpp``.
    assert MUJOCO_TO_ISAACLAB[12] == 22
    assert MUJOCO_TO_ISAACLAB[13] == 4
    assert MUJOCO_TO_ISAACLAB[14] == 10
    assert MUJOCO_TO_ISAACLAB[15] == 16


def test_body_reference_uses_the_arm_first_49d_layout() -> None:
    actions = np.arange(3 * 49, dtype=np.float64).reshape(3, 49)
    body = body_reference(actions)
    assert body.shape == (3, 29)
    np.testing.assert_allclose(body[:, :15], actions[:, 28:43])
    np.testing.assert_allclose(body[:, 15:], actions[:, 0:14])


def test_body_reference_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match="must be \\[T,49\\]"):
        body_reference(np.zeros((4, 43)))


def test_rpy_to_quat_wxyz_round_trips_the_yaw_axis() -> None:
    quat = rpy_to_quat_wxyz(np.asarray((0.0, 0.0, 0.5)))
    np.testing.assert_allclose(quat, [np.cos(0.25), 0.0, 0.0, np.sin(0.25)], atol=1e-12)
