"""Tests for the B-mode SONIC token path and SuperMimic's token policy interface.

Two integration points are pinned here:

* ``SonicTokenProvider`` builds ``(N, 64)`` tokens from future reference frames and
  the **simulated** robot root orientation (B mode), detached and on the FSQ grid;
* ``supermimic.policy.nets.Policy`` accepts ``interface="token"``: the 64-D token is
  the latent (no learned goal encoder), and gradients reach the decoder.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from diff_vla_wbc.compat import ensure_supermimic_importable
from diff_vla_wbc.sonic_token import SonicTokenProvider

ensure_supermimic_importable()

from supermimic.policy.nets import Policy, PolicyCfg  # noqa: E402


def _ref(n: int, nf: int, quat: torch.Tensor | None = None) -> SimpleNamespace:
    if quat is None:
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n * nf, 4)
    return SimpleNamespace(
        joint_pos=torch.randn(n * nf, 29),
        joint_vel=torch.randn(n * nf, 29) * 0.5,
        root_quat_w=quat,
    )


class SonicTokenProviderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provider = SonicTokenProvider(device="cpu")

    def test_a_mode_shape_grid_detached(self):
        tok = self.provider.from_reference(_ref(3, 10), None)
        self.assertEqual(tuple(tok.shape), (3, 64))
        self.assertFalse(tok.requires_grad)
        self.assertTrue(torch.allclose(tok * 16, torch.round(tok * 16), atol=1e-6))

    def test_b_mode_with_perfect_tracking_equals_a_mode(self):
        ref = _ref(2, 10)
        tok_a = self.provider.from_reference(ref, None)
        tok_b = self.provider.from_reference(ref, ref.root_quat_w.clone())
        self.assertTrue(torch.equal(tok_a, tok_b))

    def test_b_mode_responds_to_robot_rotation(self):
        ref = _ref(1, 10)
        half = torch.tensor(torch.pi / 4) / 2
        q_robot = torch.tensor([[torch.cos(half), 0.0, 0.0, torch.sin(half)]])
        tok_a = self.provider.from_reference(ref, None)
        tok_b = self.provider.from_reference(ref, q_robot.expand(10, 4))
        self.assertFalse(torch.equal(tok_a, tok_b))

    def test_from_state_uses_state_quat(self):
        class FakeLib:
            def last_frame_of(self, frame):
                return frame + 100

            def ref(self, idx):
                return _ref(1, idx.numel(), torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(idx.numel(), 4))

        lib = FakeLib()
        state = torch.zeros(2, 71)
        state[:, 3] = 1.0  # wxyz identity
        tok = self.provider.from_state(lib, torch.tensor([0, 5]), state)
        self.assertEqual(tuple(tok.shape), (2, 64))

    def test_bad_frame_count_raises(self):
        ref = _ref(1, 10)
        ref.joint_pos = ref.joint_pos[:-1]
        with self.assertRaises(ValueError):
            self.provider.from_reference(ref, None)


class TokenPolicyTest(unittest.TestCase):
    def test_token_interface_forward_backward(self):
        policy = Policy(PolicyCfg(token_dim=64))
        obs = torch.zeros(4, 930)
        token = torch.randn(4, 64, requires_grad=True)
        action = policy(obs, token, interface="token")
        self.assertEqual(tuple(action.shape), (4, 29))
        action.sum().backward()
        self.assertIsNotNone(token.grad)
        self.assertTrue(any(p.grad is not None for p in policy.decoder.parameters()))

    def test_motion_interface_still_works(self):
        policy = Policy(PolicyCfg())
        action = policy(torch.zeros(2, 930), torch.zeros(2, 680))
        self.assertEqual(tuple(action.shape), (2, 29))

    def test_token_interface_requires_token_dim(self):
        policy = Policy(PolicyCfg())
        with self.assertRaises(RuntimeError):
            policy(torch.zeros(1, 930), torch.zeros(1, 64), interface="token")

    def test_token_dim_mismatch_raises(self):
        policy = Policy(PolicyCfg(token_dim=64))
        with self.assertRaises(ValueError):
            policy(torch.zeros(1, 930), torch.zeros(1, 32), interface="token")


if __name__ == "__main__":
    unittest.main()
