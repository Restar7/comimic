import unittest

import torch

from diff_vla_wbc import build_g1_assets
from diff_vla_wbc.compat import (
    ANCHOR_BODY,
    G1_ACTION_SCALE,
    G1_JOINT_NAMES,
    TRACKED_BODIES,
    axis_angle_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_box_minus,
    quat_box_plus,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_mul,
)


class MathAndAssetTest(unittest.TestCase):
    def test_quat_identity_and_inverse(self):
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        v = torch.tensor([[1.0, 2.0, 3.0]])
        self.assertTrue(torch.allclose(quat_apply(q, v), v, atol=1e-6))
        r = matrix_from_quat(q)
        self.assertTrue(torch.allclose(r, torch.eye(3).expand(1, 3, 3), atol=1e-6))

    def test_quat_composition_and_box_ops(self):
        axis = torch.tensor([[0.0, 0.0, 1.0]])
        q = quat_from_angle_axis(torch.tensor([0.7]), axis)
        v = torch.tensor([[1.0, 0.0, 0.0]])
        rotated = quat_apply(q, v)
        self.assertTrue(torch.allclose(quat_apply_inverse(q, rotated), v, atol=1e-5))
        self.assertTrue(torch.allclose(quat_mul(q, q), quat_from_angle_axis(torch.tensor([1.4]), axis), atol=1e-5))
        delta = quat_box_minus(q, q)
        self.assertTrue(torch.allclose(delta, torch.zeros(1, 3), atol=1e-5))
        q2 = quat_box_plus(q, axis * 0.1)
        self.assertLess(float(quat_error_magnitude(q2, q)), 0.11)
        aa = axis_angle_from_quat(q)
        self.assertAlmostEqual(float(aa.norm()), 0.7, places=4)

    def test_g1_assets(self):
        assets = build_g1_assets(device="cpu")
        self.assertEqual(tuple(assets.default_joint_pos.shape), (29,))
        self.assertEqual(tuple(assets.action_scale.shape), (29,))
        self.assertEqual(len(G1_JOINT_NAMES), 29)
        self.assertEqual(len(TRACKED_BODIES), 14)
        self.assertIn(ANCHOR_BODY, assets.fk.body_names)
        self.assertTrue(all(n in assets.fk.body_names for n in TRACKED_BODIES))
        self.assertTrue(torch.all(G1_ACTION_SCALE > 0))
        # Default knees-bent pose from mjlab's G1 keyframe.
        self.assertAlmostEqual(float(assets.default_joint_pos[0]), -0.312, places=6)
        self.assertAlmostEqual(float(assets.default_joint_pos[3]), 0.669, places=6)


if __name__ == "__main__":
    unittest.main()
