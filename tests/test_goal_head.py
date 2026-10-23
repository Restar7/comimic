"""Tests for the direct VLA -> SuperMimic goal interface (``diff_vla_wbc.goal_head``).

The design replaces the token adapter: the VLA predicts SuperMimic's *physical*
10x68 goal directly (``q - q0``, ``qd``, gravity, body/world velocities, height),
so no learned latent-to-physical conversion sits in between.  The hand head is a
separate 14-D output that never enters SuperMimic.

These tests pin down the contract that makes the swap safe:

* the goal layout per frame matches ``GOAL_SLICES`` and SuperMimic's
  ``MotionLibrary.goal_features`` order;
* at initialisation the head emits the neutral physical prior (upright gravity,
  0.76 m) rather than noise;
* the flattened goal is accepted by ``SupermimicBundle.act`` unchanged.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from diff_vla_wbc.goal_head import (
    GOAL_DIM,
    GOAL_FEATURE_DIM,
    GOAL_FRAMES,
    GOAL_SLICES,
    GoalHead,
    GoalHeadCfg,
    GoalSpec,
    flatten_goal,
    unflatten_goal,
)
from diff_vla_wbc.gr00t_vlm import Gr00tVLM
from diff_vla_wbc.hand_head import HandHead, HandHeadCfg


class GoalSpecTest(unittest.TestCase):
    def test_scale_vector_layout(self):
        spec = GoalSpec()
        vec = spec.scale_vector()
        self.assertEqual(vec.numel(), GOAL_DIM)
        # first frame, per-term scale values
        for name, expected in [
            ("joint_pos", 0.5),
            ("joint_vel", 5.0),
            ("gravity", 1.0),
            ("root_ang_vel", 5.0),
            ("root_lin_vel", 1.0),
            ("root_height", 0.1),
        ]:
            sl = GOAL_SLICES[name]
            self.assertAlmostEqual(float(vec[sl.start]), expected, msg=name)

    def test_normalize_denormalize_roundtrip(self):
        spec = GoalSpec()
        goal = torch.randn(3, GOAL_FRAMES, GOAL_FEATURE_DIM)
        back = spec.denormalize(spec.normalize(goal))
        torch.testing.assert_close(back, goal)

    def test_flatten_unflatten_roundtrip(self):
        goal = torch.randn(2, GOAL_FRAMES, GOAL_FEATURE_DIM)
        flat = flatten_goal(goal)
        self.assertEqual(tuple(flat.shape), (2, GOAL_DIM))
        torch.testing.assert_close(unflatten_goal(flat), goal)

    def test_flatten_is_frame_major(self):
        goal = torch.zeros(1, GOAL_FRAMES, GOAL_FEATURE_DIM)
        goal[0, 3, GOAL_SLICES["root_height"].start] = 1.234  # frame 3, height
        flat = flatten_goal(goal)[0]
        self.assertAlmostEqual(float(flat[3 * GOAL_FEATURE_DIM + GOAL_SLICES["root_height"].start]), 1.234)


class GoalHeadTest(unittest.TestCase):
    def test_shape(self):
        head = GoalHead(GoalHeadCfg(in_dim=32, hidden=16))
        out = head(torch.randn(2, 7, 32))
        self.assertEqual(tuple(out.shape), (2, GOAL_FRAMES, GOAL_FEATURE_DIM))

    def test_initial_output_is_physical_prior(self):
        head = GoalHead(GoalHeadCfg(in_dim=16, hidden=16, dropout=0.0)).eval()
        out = head(torch.randn(2, 5, 16))
        # zero-init last layer -> neutral prior: no joint offset, gravity down, 0.76 m
        torch.testing.assert_close(out[:, :, GOAL_SLICES["joint_pos"]], torch.zeros(2, GOAL_FRAMES, 29))
        torch.testing.assert_close(out[:, :, GOAL_SLICES["gravity"]], torch.tensor([0.0, 0.0, -1.0]).expand(2, GOAL_FRAMES, 3))
        self.assertAlmostEqual(float(out[0, 0, GOAL_SLICES["root_height"].start]), 0.76, places=4)

    def test_mask_excludes_padding(self):
        head = GoalHead(GoalHeadCfg(in_dim=8, hidden=8)).eval()
        feats = torch.randn(1, 5, 8)
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        garbage = feats.clone()
        garbage[:, 3:, :] = 1e6
        with torch.no_grad():
            torch.testing.assert_close(head(feats, mask=mask), head(garbage, mask=mask))

    def test_loss_and_gradients(self):
        head = GoalHead(GoalHeadCfg(in_dim=8, hidden=8))
        feats = torch.randn(2, 4, 8, requires_grad=True)
        pred = head(feats)
        target = pred.detach() + 0.1
        loss = head.loss(pred, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(feats.grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))

    def test_loss_shape_mismatch_raises(self):
        head = GoalHead(GoalHeadCfg(in_dim=8, hidden=8))
        with self.assertRaises(ValueError):
            head.loss(torch.zeros(1, GOAL_FRAMES, GOAL_FEATURE_DIM), torch.zeros(1, GOAL_FRAMES, 4))


class ForwardGoalAndHandsTest(unittest.TestCase):
    def test_two_outputs_without_tokens(self):
        feats = torch.randn(2, 9, 16)
        goal_head = GoalHead(GoalHeadCfg(in_dim=16, hidden=8))
        hand_head = HandHead(HandHeadCfg(in_dim=16, hidden=8, horizon=2))
        vlm = Gr00tVLM(
            config=None,
            backbone=torch.nn.Identity(),
            action_head=torch.nn.Identity(),
            goal_head=goal_head,
            hand_head=hand_head,
        )
        with mock.patch.object(Gr00tVLM, "backbone_features", return_value=feats):
            goal, hands = vlm.forward_goal_and_hands({"dummy": 1})
        self.assertEqual(tuple(goal.shape), (2, GOAL_FRAMES, GOAL_FEATURE_DIM))
        self.assertEqual(tuple(hands.shape), (2, 2, 14))

    def test_heads_optional(self):
        vlm = Gr00tVLM(config=None, backbone=torch.nn.Identity(), action_head=torch.nn.Identity())
        with mock.patch.object(Gr00tVLM, "backbone_features", return_value=torch.randn(1, 4, 8)):
            goal, hands = vlm.forward_goal_and_hands({"dummy": 1})
        self.assertIsNone(goal)
        self.assertIsNone(hands)


class SuperMimicInterfaceTest(unittest.TestCase):
    """The flattened goal must be accepted by the real (small) SuperMimic bundle."""

    @classmethod
    def setUpClass(cls):
        from diff_vla_wbc.supermimic_backend import SuperMimicBundle, build_g1_assets

        cls.assets = build_g1_assets(device="cpu")
        cls.bundle = SuperMimicBundle.random(cls.assets, small=True)

    def test_goal_drives_policy(self):
        head = GoalHead(GoalHeadCfg(in_dim=32, hidden=16))
        goal = head(torch.randn(1, 6, 32))
        obs = torch.zeros(1, 10 * 93)  # history=10 x obs_dim=93
        action = self.bundle.act(obs, flatten_goal(goal))
        self.assertEqual(tuple(action.shape), (1, 29))
        self.assertTrue(torch.isfinite(action).all())

    def test_arbitrary_goal_in_range(self):
        # A physically plausible goal (upright, walking-height) must not explode the policy.
        goal = torch.zeros(1, GOAL_FRAMES, GOAL_FEATURE_DIM)
        goal[:, :, GOAL_SLICES["gravity"]] = torch.tensor([0.0, 0.0, -1.0])
        goal[:, :, GOAL_SLICES["root_height"]] = 0.76
        action = self.bundle.act(torch.zeros(1, 10 * 93), flatten_goal(goal))
        self.assertEqual(tuple(action.shape), (1, 29))


if __name__ == "__main__":
    unittest.main()
