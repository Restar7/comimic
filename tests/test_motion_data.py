import tempfile
import unittest
from pathlib import Path

import torch

from diff_vla_wbc.compat import WORKSPACE_ROOT
from diff_vla_wbc.data import (
    MotionBatchProvider,
    motion_npz_from_body_csv_dir,
    sample_reference_windows,
)
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


class MotionDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.npz = Path(cls.tmp.name) / "dance.npz"
        if not SAMPLE_CLIP.is_dir():
            raise unittest.SkipTest(f"missing sample clip {SAMPLE_CLIP}")
        motion_npz_from_body_csv_dir(SAMPLE_CLIP, cls.npz)
        cls.assets = build_g1_assets(device="cpu")
        cls.bundle = SuperMimicBundle.random(cls.assets, small=True)
        cls.library = load_motion_library([cls.npz], cls.assets, device="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_library_contract(self):
        self.assertEqual(self.library.num_clips, 1)
        self.assertGreater(self.library.num_frames, 100)
        goal = self.library.goal_window(torch.tensor([10]), 10)
        self.assertEqual(tuple(goal.shape), (1, 680))
        qpos, qvel = self.library.ref_qpos_qvel(torch.tensor([10]))
        self.assertEqual(tuple(qpos.shape), (1, 36))
        self.assertEqual(tuple(qvel.shape), (1, 35))

    def test_reference_start_and_provider(self):
        start = self.bundle.reference_start(self.library, torch.tensor([50, 60]))
        self.assertEqual(tuple(start["state"].shape), (2, 71))
        self.assertEqual(tuple(start["obs_hist"].shape), (2, 930))
        self.assertEqual(tuple(start["observations"].shape), (10, 2, 93))
        self.assertEqual(tuple(start["prev_action"].shape), (2, 29))
        provider = MotionBatchProvider(
            self.bundle, self.library, batch_size=2, horizon=4, seed=0
        )
        batch = provider.sample()
        self.assertEqual(tuple(batch.fm_tokens.shape), (2, 40, 64))
        self.assertEqual(tuple(batch.teacher_goals.shape), (2, 680))
        self.assertEqual(tuple(batch.teacher_goal_sequence.shape), (2, 4, 680))
        self.assertEqual(tuple(batch.teacher_states.shape), (2, 4, 71))
        self.assertEqual(len(batch.references), 4)
        self.assertEqual(tuple(batch.references[0].body_pos_w.shape), (2, 14, 3))

    def test_reference_windows(self):
        windows = sample_reference_windows(
            self.bundle, self.library, batch_size=3, window=4, seed=0
        )
        self.assertEqual(tuple(windows["states"].shape), (5, 3, 71))
        self.assertEqual(tuple(windows["actions"].shape), (4, 3, 29))
        self.assertEqual(tuple(windows["xi"].shape), (3, 33))
        self.assertEqual(len(windows["refs"]), 4)
        self.assertTrue(torch.isfinite(windows["states"]).all())


if __name__ == "__main__":
    unittest.main()
