import tempfile
import unittest
from pathlib import Path

import torch

from diff_vla_wbc.compat import WORKSPACE_ROOT
from diff_vla_wbc.data import motion_npz_from_body_csv_dir
from diff_vla_wbc.mujoco_env import G1MujocoEnv
from diff_vla_wbc.supermimic_backend import (
    SuperMimicBundle,
    build_g1_assets,
    load_motion_library,
)

SAMPLE_CLIP = (
    WORKSPACE_ROOT
    / "GR00T-WholeBodyControl"
    / "gear_sonic_deploy"
    / "reference"
    / "example"
    / "dance_in_da_party_001__A464"
)


class MuJoCoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SAMPLE_CLIP.is_dir():
            raise unittest.SkipTest("no sample motion clip to convert")
        # Convert into a throwaway directory: the repo ships no motion npz dataset.
        cls._tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.npz = Path(cls._tmp.name) / "sample.npz"
        motion_npz_from_body_csv_dir(SAMPLE_CLIP, cls.npz)
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
