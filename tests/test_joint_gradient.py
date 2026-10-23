import tempfile
import unittest
from pathlib import Path

import torch

from diff_vla_wbc.adapter import AdapterConfig, MotionGoalAdapter
from diff_vla_wbc.compat import WORKSPACE_ROOT
from diff_vla_wbc.data import MotionBatchProvider, motion_npz_from_body_csv_dir
from diff_vla_wbc.gr00t_fm import sample_model_with_grad
from diff_vla_wbc.gr00t_tiny import TinyGr00tModel, make_tiny_action_head, tiny_model_inputs
from diff_vla_wbc.supermimic_backend import (
    SuperMimicBundle,
    build_g1_assets,
    load_motion_library,
)
from diff_vla_wbc.train import JointTrainer, JointTrainerConfig

SAMPLE_CLIP = (
    WORKSPACE_ROOT
    / "GR00T-WholeBodyControl"
    / "gear_sonic_deploy"
    / "reference"
    / "example"
    / "dance_in_da_party_001__A464"
)


class EndToEndGradientTest(unittest.TestCase):
    def test_real_gr00t_head_to_supermimic(self):
        torch.manual_seed(7)
        head = make_tiny_action_head()
        model = TinyGr00tModel(head)
        inputs = tiny_model_inputs(2)
        fixed_noise = torch.randn(2, 12, 64)
        tokens = sample_model_with_grad(model, inputs, initial_noise=fixed_noise, num_steps=3)
        self.assertEqual(tuple(tokens.shape), (2, 12, 64))
        self.assertTrue(tokens.requires_grad)
        tokens.square().mean().backward()
        self.assertGreater(float(head.model.proj_out_2.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.backbone.proj.weight.grad.abs().sum()), 0.0)

        # Fresh head for the full joint step.
        head = make_tiny_action_head()
        model = TinyGr00tModel(head)
        fixed_noise = torch.randn(2, 12, 64)
        inputs = tiny_model_inputs(2)

        def token_source(meta):
            return sample_model_with_grad(model, inputs, initial_noise=fixed_noise, num_steps=3)

        assets = build_g1_assets(device="cpu")
        bundle = SuperMimicBundle.random(assets, small=True)
        if SAMPLE_CLIP.is_dir():
            # Convert into a throwaway directory: the repo does not ship a
            # motion npz dataset, and tests must not repopulate one.
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            npz = Path(tmp.name) / "sample.npz"
            motion_npz_from_body_csv_dir(SAMPLE_CLIP, npz)
            files = [npz]
        else:
            self.skipTest("no sample motion clip")
        library = load_motion_library(files, assets, device="cpu")
        adapter = MotionGoalAdapter(
            AdapterConfig(token_dim=64, token_horizon=12, hidden=64, transformer_layers=1, transformer_heads=4)
        )
        provider = MotionBatchProvider(
            bundle, library, batch_size=2, horizon=4, token_source=token_source, seed=0
        )
        trainer = JointTrainer(
            bundle,
            adapter,
            fm_sampler=token_source,
            fm_parameters=head.parameters(),
            cfg=JointTrainerConfig(horizon=4),
            device="cpu",
        )
        metrics = trainer.train_step(provider.sample())
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertGreater(float(head.model.proj_out_2.weight.grad.abs().sum()), 0.0)
        adapter_grad = sum(
            float(p.grad.abs().sum()) for p in adapter.parameters() if p.grad is not None
        )
        policy_grad = sum(
            float(p.grad.abs().sum()) for p in bundle.policy.parameters() if p.grad is not None
        )
        self.assertGreater(adapter_grad, 0.0)
        self.assertGreater(policy_grad, 0.0)
        self.assertFalse(any(p.grad is not None for p in bundle.wm.parameters()))


if __name__ == "__main__":
    unittest.main()
