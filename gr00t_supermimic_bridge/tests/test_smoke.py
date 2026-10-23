"""Shape + end-to-end autograd smoke test on toy GR00T/WBC/WM modules.

This is NOT a real MuJoCo training / task-success test.
"""
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from gr00t_supermimic_bridge.core import (
    MotionGoalAdapter, compute_loss, freeze_world_model, imagined_rollout,
    normalize_context, proprio_obs, rot6d_to_mat, mat_to_6d,
    sample_fm_with_grad,
)


class DummyActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 16)
    def forward(self, actions, timesteps, embodiment_id):
        return self.proj(actions)


class DummyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(16, 16)
    def forward(self, hidden_states, encoder_hidden_states, timestep, **unused):
        return self.net(hidden_states) + encoder_hidden_states.mean(1, keepdim=True)


class DummyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(16, 64)
    def forward(self, hidden, embodiment_id):
        return self.net(hidden)


class DummyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_horizon = 12
        self.action_dim = 64
        self.num_inference_timesteps = 3
        self.num_timestep_buckets = 1000
        self.config = SimpleNamespace(add_pos_embed=False, use_alternate_vl_dit=False)
        self.action_encoder = DummyActionEncoder()
        self.action_decoder = DummyDecoder()
        self.model = DummyDiT()
        self.state_encoder = nn.Linear(6, 16)
    def _encode_features(self, backbone_output, action_input):
        return SimpleNamespace(
            backbone_features=backbone_output.backbone_features,
            state_features=self.state_encoder(action_input.state.reshape(-1, 6)).unsqueeze(1)
        )


class DummyPi(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(730 + 990, 32), nn.Tanh(), nn.Linear(32, 29))
    def forward(self, obs_hist, goal):
        return self.net(torch.cat((obs_hist.flatten(1), goal), -1))


class DummyWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = nn.Linear(29, 29)
    def step(self, state, action, normalized_context):
        p, R, v, w, q, dq = state
        acceleration = self.transform(action)
        new_q = q + .02 * acceleration
        new_dq = dq + .04 * acceleration
        new_p = p + .01 * torch.stack(
            (acceleration.mean(-1), acceleration[:, 0], acceleration[:, 1]), -1
        )
        return (new_p, R, v, w, new_q, new_dq)


class BridgeSmokeTest(unittest.TestCase):
    def test_sixd_rotation_identity(self):
        r = torch.eye(3).expand(2, 3, 3)
        self.assertTrue(torch.allclose(rot6d_to_mat(mat_to_6d(r)), r, atol=1e-5))

    def test_full_task_gradient_to_fm_head(self):
        torch.manual_seed(2026)
        b = 2
        head = DummyHead()
        adapter = MotionGoalAdapter(hidden=48)
        pi = DummyPi()
        wm = DummyWorldModel()
        freeze_world_model(wm)

        backbone = SimpleNamespace(backbone_features=torch.randn(b, 4, 16))
        action_input = SimpleNamespace(
            state=torch.randn(b, 2, 3),
            embodiment_id=torch.zeros(b, dtype=torch.long),
        )
        tokens = sample_fm_with_grad(
            head, backbone, action_input, initial_noise=torch.randn(b, 12, 64),
        )
        identity = torch.eye(3).expand(b, 3, 3).clone()
        state = (
            torch.zeros(b, 3),
            identity,
            torch.zeros(b, 3),
            torch.zeros(b, 3),
            torch.zeros(b, 29),
            torch.zeros(b, 29),
        )
        hist = torch.zeros(b, 10, 99)
        context = torch.zeros(b, 33)
        context[:, 0] = .75
        r = imagined_rollout(tokens, adapter, pi, wm, state, hist,
                             context, torch.zeros(29), horizon=4)
        self.assertEqual(tuple(r.goals.shape), (2, 12, 730))
        self.assertEqual(tuple(r.actions.shape), (2, 4, 29))
        self.assertEqual(tuple(r.joint_positions.shape), (2, 4, 29))
        loss, terms = compute_loss(
            r,
            teacher_joint_positions=torch.full((b, 4, 29), .1),
            teacher_root_positions=torch.full((b, 4, 3), .2),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        grads = {
            "fm_head": head.model.net.weight.grad,
            "goal_adapter": adapter.net[-1].weight.grad,
            "supermimic_policy": pi.net[-1].weight.grad,
            "frozen_world_model": wm.transform.weight.grad,
        }
        for name in ("fm_head", "goal_adapter", "supermimic_policy"):
            value = grads[name]
            self.assertIsNotNone(value, f"{name} gradient is absent")
            self.assertGreater(value.abs().sum().item(), 0.0, f"{name} zero gradient")
            self.assertTrue(torch.isfinite(value).all(), f"{name} invalid gradient")
        self.assertIsNone(grads["frozen_world_model"])
        print("\nGradient smoke test:", {
            k: (None if v is None else round(float(v.norm()), 8))
            for k, v in grads.items()
        })
        print("Expected: nonzero FM/adapter/WBC grads; frozen WM parameter grad=None")


if __name__ == "__main__":
    unittest.main()
