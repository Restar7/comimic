import unittest
from pathlib import Path

import torch

from diff_vla_wbc.compat import WORKSPACE_ROOT
from diff_vla_wbc.mujoco_env import G1MujocoEnv
from diff_vla_wbc.supermimic_backend import (
    SuperMimicBundle,
    build_g1_assets,
    load_motion_library,
)


class MuJoCoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.npz = WORKSPACE_ROOT / "data/npz/example/dance_in_da_party_001__A464.npz"
        if not cls.npz.exists():
            raise unittest.SkipTest("run scripts/prepare_motion_npz.py first")
        cls.assets = build_g1_assets(device="cpu")
        cls.bundle = SuperMimicBundle.random(cls.assets, small=True)
        cls.library = load_motion_library([cls.npz], cls.assets, device="cpu")
        cls.env = G1MujocoEnv(cls.bundle)

    def test_reset_and_step(self):
        qpos, qvel = self.library.ref_qpos_qvel(torch.tensor([0]))
        self.env.reset(torch.cat((qpos[0], qvel[0]), -1))
        state = self.env.state()
        self.assertEqual(tuple(state.shape), (1, 71))
        self.assertTrue(torch.isfinite(state).all())
        for _ in range(5):
            self.env.apply_action(torch.zeros(1, 29))
        self.assertTrue(torch.isfinite(self.env.state()).all())

    def test_policy_rollout(self):
        result = self.env.rollout(self.library, max_steps=10)
        self.assertGreaterEqual(result.steps, 1)
        self.assertTrue(0.0 <= result.survival <= 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(result.norm_return)))
        self.assertEqual(tuple(result.trajectory["qpos"].shape)[1:], (36,))


if __name__ == "__main__":
    unittest.main()
