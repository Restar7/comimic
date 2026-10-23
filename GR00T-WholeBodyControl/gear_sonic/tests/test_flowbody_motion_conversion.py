import numpy as np
import pytest

from gear_sonic.data_process.convert_flowbody_actions_to_motion_lib import (
    HAND_FLEXION_SIGN,
    STANDING_BODY_DOF,
    build_motion_entry,
)


def test_build_motion_entry_preserves_arm_and_masked_hand_contract():
    np.testing.assert_array_equal(
        HAND_FLEXION_SIGN,
        [-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    )
    actions = np.zeros((2, 49), dtype=np.float32)
    actions[:, :14] = np.arange(14, dtype=np.float32)
    actions[:, 14:28] = np.arange(14, dtype=np.float32) + 1.0
    mask = np.zeros_like(actions, dtype=bool)
    mask[:, :26] = True

    entry, active_hand_dims = build_motion_entry(actions, mask, fps=30.0, root_height=0.76)

    assert entry["dof"].shape == (2, 43)
    np.testing.assert_allclose(
        entry["dof"][:, :15], np.repeat(STANDING_BODY_DOF[None, :], 2, axis=0)
    )
    np.testing.assert_allclose(entry["dof"][:, 15:29], actions[:, :14])
    np.testing.assert_allclose(
        entry["dof"][:, 29:41], actions[:, 14:26] * HAND_FLEXION_SIGN[None, :12]
    )
    np.testing.assert_allclose(entry["dof"][:, 41:43], 0.0)
    assert active_hand_dims == list(range(12))
    assert entry["pose_aa"].shape == (2, 30, 3)
    np.testing.assert_allclose(entry["root_trans_offset"][:, 2], 0.76)


def test_build_motion_entry_rejects_unsupervised_arm():
    actions = np.zeros((1, 49), dtype=np.float32)
    mask = np.ones_like(actions, dtype=bool)
    mask[:, 0] = False

    with pytest.raises(ValueError, match="arm14"):
        build_motion_entry(actions, mask, fps=30.0, root_height=0.76)
