"""Tests for the VLA dual-output split: 64-D token -> SuperMimic, 14-D hand -> direct.

The design under test (GR00T-N1.6-style path):

```text
VLM backbone ──► flow-matching head ──► tokens [B, 40, 64] ──► adapter ──► SuperMimic
             └─► hand head           ──► hands  [B, H, 14]  ──► HandStream (direct)
```

The hand angles are *not* part of the motion token and never enter the adapter,
the world model or the policy; they are clamped once at the actuator layer.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from diff_vla_wbc.hand_head import HandHead, HandHeadCfg, attach_hand_head
from diff_vla_wbc.hand_stream import (
    HAND_DIM,
    HAND_JOINT_NAMES,
    HAND_POSES,
    HandStream,
    HandStreamCfg,
    load_hand_model,
)
from diff_vla_wbc.gr00t_vlm import Gr00tVLM


def _hand_ranges(model):
    import mujoco

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    jids = [names.index(n) for n in HAND_JOINT_NAMES]
    return model.jnt_range[jids, 0], model.jnt_range[jids, 1]


class HandHeadTest(unittest.TestCase):
    def test_shapes(self):
        cfg = HandHeadCfg(in_dim=32, hidden=16, horizon=3)
        head = HandHead(cfg)
        feats = torch.randn(2, 7, 32)
        out = head(feats)
        self.assertEqual(tuple(out.shape), (2, 3, HAND_DIM))

    def test_mask_excludes_padding(self):
        cfg = HandHeadCfg(in_dim=8, hidden=8, horizon=1)
        head = HandHead(cfg).eval()
        feats = torch.randn(1, 5, 8)
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        garbage = feats.clone()
        garbage[:, 3:, :] = 1e6  # masked-out tokens must not change the pool
        with torch.no_grad():
            a = head(feats, mask=mask)
            b = head(garbage, mask=mask)
        torch.testing.assert_close(a, b)

    def test_loss_and_gradients(self):
        head = HandHead(HandHeadCfg(in_dim=8, hidden=8, horizon=2))
        feats = torch.randn(2, 4, 8, requires_grad=True)
        pred = head(feats)
        target = torch.zeros_like(pred)
        loss = HandHead.loss(pred, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(feats.grad)
        self.assertTrue(any(p.grad is not None for p in head.parameters()))

    def test_loss_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            HandHead.loss(torch.zeros(1, 1, HAND_DIM), torch.zeros(1, 1, HAND_DIM - 1))

    def test_attach_helper(self):
        model = nn.Module()
        head = attach_hand_head(model, HandHeadCfg(in_dim=8, hidden=8))
        self.assertIs(model.hand_head, head)
        self.assertIsInstance(model.hand_head, HandHead)


class ForwardWithHandsTest(unittest.TestCase):
    """``Gr00tVLM.forward_with_hands`` must return both outputs, hand independent."""

    def test_returns_two_outputs(self):
        head = HandHead(HandHeadCfg(in_dim=16, hidden=8, horizon=2))
        vlm = Gr00tVLM(config=None, backbone=nn.Identity(), action_head=nn.Identity(), hand_head=head)
        tokens = torch.randn(2, 40, 64)
        features = torch.randn(2, 9, 16)
        with mock.patch(
            "diff_vla_wbc.gr00t_fm.sample_model_with_grad",
            return_value=(tokens, features, torch.zeros(2, 1)),
        ):
            got_tokens, hands = vlm.forward_with_hands({"dummy": 1})
        self.assertIs(got_tokens, tokens)
        self.assertEqual(tuple(hands.shape), (2, 2, HAND_DIM))

    def test_without_hand_head_is_plain_forward(self):
        vlm = Gr00tVLM(config=None, backbone=nn.Identity(), action_head=nn.Identity())
        tokens = torch.randn(2, 40, 64)
        features = torch.randn(2, 9, 16)
        with mock.patch(
            "diff_vla_wbc.gr00t_fm.sample_model_with_grad",
            return_value=(tokens, features, torch.zeros(2, 1)),
        ):
            got_tokens, hands = vlm.forward_with_hands({"dummy": 1})
        self.assertIs(got_tokens, tokens)
        self.assertIsNone(hands)

    def test_hand_head_params_disjoint_from_adapter(self):
        from diff_vla_wbc.adapter import AdapterConfig, MotionGoalAdapter

        head = HandHead(HandHeadCfg(in_dim=16, hidden=8))
        adapter = MotionGoalAdapter(AdapterConfig(token_dim=64, token_horizon=40))
        hand_ids = {id(p) for p in head.parameters()}
        adapter_ids = {id(p) for p in adapter.parameters()}
        self.assertFalse(hand_ids & adapter_ids)
        self.assertEqual(sum(p.numel() for p in head.parameters()) > 0, True)


class HandRoutingTest(unittest.TestCase):
    """End-to-end: hand output -> HandStream -> the 14 hand actuators, body untouched."""

    def test_vla_hand_goes_directly_to_hand_actuators(self):
        import mujoco

        model = load_hand_model()
        data = mujoco.MjData(model)
        hand = HandStream(model, HandStreamCfg(source="vla"))
        self.assertTrue(hand.enabled)

        rng = np.random.default_rng(0)
        raw = rng.normal(0.0, 3.0, size=HAND_DIM)  # deliberately out of range
        targets = hand.command(raw)
        lo, hi = _hand_ranges(model)
        self.assertTrue(np.all(targets >= lo - 1e-9) and np.all(targets <= hi + 1e-9))

        hand.command(HAND_POSES["grasp"])  # inside limits on both sides
        body_tau = np.arange(29, dtype=np.float64)
        ctrl = hand.assemble_ctrl(body_tau, data=data)
        self.assertEqual(ctrl.shape, (43,))
        # Body slots carry the body torques exactly; hand torques only in hand slots.
        np.testing.assert_allclose(ctrl[0:22], body_tau[0:22])
        np.testing.assert_allclose(ctrl[29:36], body_tau[22:29])
        hand_tau = hand.pd_torque(data)
        np.testing.assert_allclose(ctrl[hand.ctrl_ids], hand_tau)
        self.assertGreater(int(np.count_nonzero(hand_tau)), 10)  # every finger driven

    def test_changing_hands_does_not_change_body_ctrl(self):
        import mujoco

        model = load_hand_model()
        data = mujoco.MjData(model)
        hand = HandStream(model, HandStreamCfg(source="vla"))
        body_tau = np.arange(29, dtype=np.float64)

        hand.command(np.zeros(HAND_DIM))
        ctrl_open = hand.assemble_ctrl(body_tau, data=data)
        hand.command(np.full(HAND_DIM, 1.0))
        ctrl_closed = hand.assemble_ctrl(body_tau, data=data)

        body_slots = [0, 1, 2, 21, 29, 30, 35]
        np.testing.assert_allclose(ctrl_open[body_slots], ctrl_closed[body_slots])


if __name__ == "__main__":
    unittest.main()
