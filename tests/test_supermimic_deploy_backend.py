"""Smoke test for the SuperMimic deployment decoder (``groot_repro/supermimic_backend``).

The decoder turns VLA tokens into joint targets in the MuJoCo deployment loop,
replacing SONIC's ``model_decoder.onnx``.  It needs the SuperMimic venv (torch +
mjlab + supermimic), so the test is skipped when those are unavailable.
"""

from __future__ import annotations

import unittest

import numpy as np

import os

# 默认跳过：这套解码器需要 SuperMimic 训练 venv（torch+mjlab+supermimic）。
# 在训练 venv 里用 SUPERMIMIC_DEPLOY_TEST=1 打开。
SKIP_REASON = None
if os.environ.get("SUPERMIMIC_DEPLOY_TEST") != "1":
    SKIP_REASON = "set SUPERMIMIC_DEPLOY_TEST=1 (run inside the SuperMimic venv)"
else:
    try:  # pragma: no cover - environment dependent
        import supermimic  # noqa: F401
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        SKIP_REASON = "supermimic/torch not importable in this interpreter"


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class DeployBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "groot_repro"))
        from supermimic_backend import SuperMimicController

        ckpt = root / "third_party" / "SuperMimic-Reproduced" / "logs" / "_token_rand" / "ckpt_latest.pt"
        if not ckpt.is_file():
            raise unittest.SkipTest(f"no random token checkpoint at {ckpt}")
        cls.ctrl = SuperMimicController(ckpt, device="cpu")

    def test_obs_layout(self):
        obs = self.ctrl.build_obs()
        self.assertEqual(obs.shape, (1, 930))

    def test_one_step(self):
        q = self.ctrl.q0_isaac[self.ctrl.isaac_to_mujoco].copy()
        qt = self.ctrl.step(
            np.zeros(64, np.float32), q, np.zeros(29), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)
        )
        self.assertEqual(qt.shape, (29,))
        self.assertTrue(np.isfinite(qt).all())

    def test_torque(self):
        q = self.ctrl.q0_isaac[self.ctrl.isaac_to_mujoco].copy()
        tau = self.ctrl.torque(q, q, np.zeros(29))
        self.assertEqual(tau.shape, (29,))
        self.assertTrue(np.isfinite(tau).all())


if __name__ == "__main__":
    unittest.main()
