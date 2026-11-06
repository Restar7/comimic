"""Tests for the FSQ grid / STE helper (``diff_vla_wbc.quant``).

三件事被钉死：

1. **网格口径**与 release encoder 导出图一致（``q = n/16``、值域 ``[-1, 0.9375]``）。
   写错这一处不会报错，只会让"量化"变成另一个函数，所以逐点对拍。
2. **STE 的反向严格等于 1**（``∂τ_use/∂τ ≡ 1``），包括越界被夹住的位置——这是"梯度能不能
   穿回 VLA"的唯一数学前提。
3. **``hard`` 模式必须断梯度**。它是消融臂的对照：如果 ``hard`` 也能降 loss，说明梯度从
   trust/BC 项漏进来了，那一轮的结论就不能用。
"""

from __future__ import annotations

import unittest

import torch

from diff_vla_wbc.quant import (
    FSQ_DIV,
    GRID_STEP,
    LEVEL_MAX,
    LEVEL_MIN,
    MODES,
    TOKEN_MAX,
    TOKEN_MIN,
    apply_token_quant,
    fsq_grid,
    grid_alignment,
    grid_stats,
    is_on_grid,
    ste_quantize,
)


class TestGridConstants(unittest.TestCase):
    def test_constants_match_release_encoder(self):
        # q = round(tanh(h+off)*15.5155 - 0.5)/16 ⇒ n ∈ [-16, 15]，步长 1/16
        self.assertEqual(FSQ_DIV, 16)
        self.assertEqual((LEVEL_MIN, LEVEL_MAX), (-16, 15))
        self.assertAlmostEqual(GRID_STEP, 0.0625)
        self.assertAlmostEqual(TOKEN_MIN, -1.0)
        self.assertAlmostEqual(TOKEN_MAX, 0.9375)


class TestFsqGrid(unittest.TestCase):
    def test_lands_on_multiples_of_one_sixteenth(self):
        x = torch.rand(4096) * 4.0 - 2.0
        q = fsq_grid(x)
        n = q * FSQ_DIV
        self.assertTrue(torch.allclose(n, torch.round(n), atol=1e-5))
        self.assertLessEqual(float(q.max()), TOKEN_MAX + 1e-6)
        self.assertGreaterEqual(float(q.min()), TOKEN_MIN - 1e-6)

    def test_known_values(self):
        x = torch.tensor([-1.0, -0.5, 0.0, 0.031, 0.0375, 0.0625, 0.9, 0.9375])
        # 0.031*16=0.496→0 ; 0.0375*16=0.6→0.0625 ; 0.9*16=14.4→14/16=0.875
        want = torch.tensor([-1.0, -0.5, 0.0, 0.0, 0.0625, 0.0625, 0.875, 0.9375])
        self.assertTrue(torch.allclose(fsq_grid(x), want), f"got {fsq_grid(x).tolist()}")

    def test_clamps_out_of_range(self):
        q = fsq_grid(torch.tensor([-9.0, 9.0, 1.2, -1.5]))
        self.assertTrue(torch.allclose(q, torch.tensor([-1.0, 0.9375, 0.9375, -1.0])))

    def test_no_gradient(self):
        x = torch.randn(8, requires_grad=True)
        self.assertFalse(fsq_grid(x).requires_grad)

    def test_preserves_shape_and_dtype(self):
        x = torch.rand(3, 5, 64, dtype=torch.float32)
        self.assertEqual(tuple(fsq_grid(x).shape), (3, 5, 64))
        self.assertEqual(fsq_grid(x).dtype, torch.float32)


class TestSte(unittest.TestCase):
    def test_forward_is_the_grid_value(self):
        x = torch.rand(1024) * 3.0 - 1.5
        self.assertTrue(torch.allclose(ste_quantize(x), fsq_grid(x), atol=1e-6))

    def test_backward_is_identity(self):
        x = torch.rand(3, 4, requires_grad=True) * 3.0 - 1.5   # 含越界值
        y = ste_quantize(x)
        self.assertTrue(y.requires_grad)
        g = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[0]
        self.assertTrue(torch.allclose(g, torch.ones_like(g)), f"grad={g.tolist()}")

    def test_backward_through_attention_like_reduction(self):
        """梯度要能穿过任意上游缩放：y 的导数不依赖上游形状。"""
        x = torch.randn(2, 3, 64, requires_grad=True)
        w = torch.randn_like(x)
        (ste_quantize(x) * w).sum().backward()
        self.assertTrue(torch.allclose(x.grad, w))


class TestApplyMode(unittest.TestCase):
    def test_none_is_identity(self):
        x = torch.randn(2, 2)
        out, stats = apply_token_quant(x, "none")
        self.assertIs(out, x)
        self.assertIsNone(stats)

    def test_ste_keeps_gradient_and_reports_stats(self):
        x = torch.randn(2, 64, requires_grad=True)
        out, stats = apply_token_quant(x, "ste")
        self.assertTrue(out.requires_grad)
        self.assertIsInstance(stats, dict)
        for key in ("dist_mean", "dist_p90", "dist_max", "near_edge", "out_range", "cell"):
            self.assertIn(key, stats)

    def test_hard_breaks_the_graph(self):
        """消融臂的判据：hard 不能有梯度，否则"量化无害"的结论是假的。"""
        x = torch.randn(2, 64, requires_grad=True)
        out, _ = apply_token_quant(x, "hard")
        self.assertFalse(out.requires_grad)
        self.assertIsNone(out.grad_fn)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            apply_token_quant(torch.zeros(1), "quantize-pls")

    def test_modes_exported(self):
        self.assertEqual(set(MODES), {"none", "ste", "hard"})


class TestStats(unittest.TestCase):
    def test_distance_is_bounded_inside_range(self):
        x = torch.rand(2048) * (TOKEN_MAX - TOKEN_MIN) + TOKEN_MIN
        s = grid_stats(x)
        self.assertLessEqual(s["dist_max"], 0.5 * GRID_STEP + 1e-6)
        self.assertEqual(s["out_range"], 0.0)
        self.assertAlmostEqual(s["cell"], GRID_STEP)

    def test_out_of_range_is_reported(self):
        s = grid_stats(torch.tensor([1.5, 0.0, -2.0, 0.0]))
        self.assertAlmostEqual(s["out_range"], 0.5)

    def test_on_grid(self):
        self.assertTrue(is_on_grid(fsq_grid(torch.rand(256) * 2 - 1)))
        self.assertFalse(is_on_grid(torch.tensor([0.01])))


class TestAlignment(unittest.TestCase):
    def test_release_range_is_aligned(self):
        a = grid_alignment(torch.tensor([TOKEN_MIN]), torch.tensor([TOKEN_MAX]))
        self.assertEqual(a["lo_ok"], 1.0)
        self.assertEqual(a["hi_ok"], 1.0)

    def test_off_grid_range_is_flagged(self):
        a = grid_alignment(torch.tensor([-0.93]), torch.tensor([0.9]))
        self.assertEqual(a["lo_ok"], 0.0)
        self.assertEqual(a["hi_ok"], 0.0)


if __name__ == "__main__":
    unittest.main()


class TestNumpyDeployPath(unittest.TestCase):
    """部署侧走 ``quantize_np``（闭环服务里不搬 torch 张量），必须与 torch 版逐位一致。"""

    def test_parity_with_torch(self):
        from diff_vla_wbc.quant import quantize_np

        x = torch.rand(64, 40, dtype=torch.float32) * 3.0 - 1.5   # 含越界值
        q_np, dist_np = quantize_np(x.numpy(), "ste")
        q_t = fsq_grid(x)
        self.assertTrue(torch.allclose(q_t, torch.from_numpy(q_np), atol=0, rtol=0),
                        f"max|Δ|={(q_t - torch.from_numpy(q_np)).abs().max()}")
        self.assertAlmostEqual(dist_np, float((x - q_t).abs().mean()), places=6)

    def test_none_is_passthrough(self):
        from diff_vla_wbc.quant import quantize_np

        x = (torch.rand(8, 4) * 2 - 1).numpy()
        out, dist = quantize_np(x, "none")
        self.assertEqual(dist, 0.0)
        self.assertTrue((out == x).all())

    def test_output_is_on_grid(self):
        from diff_vla_wbc.quant import quantize_np

        q, _ = quantize_np((torch.rand(32) * 4 - 2).numpy(), "hard")
        self.assertTrue(is_on_grid(torch.from_numpy(q)))


class TestDeployModeHelpers(unittest.TestCase):
    """两个 server 共用的部署口径实现（``quant.resolve_deploy_mode`` / ``check_meta_mode``）。"""

    def test_resolve_accepts_and_normalizes(self):
        from diff_vla_wbc.quant import resolve_deploy_mode

        self.assertEqual(resolve_deploy_mode("STE"), "ste")
        self.assertEqual(resolve_deploy_mode(" none "), "none")
        self.assertEqual(resolve_deploy_mode(None), "none")

    def test_resolve_rejects_garbage(self):
        from diff_vla_wbc.quant import resolve_deploy_mode

        with self.assertRaises(ValueError):
            resolve_deploy_mode("bogus")

    def test_check_meta_missing_is_none(self):
        import tempfile
        from diff_vla_wbc.quant import check_meta_mode

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(check_meta_mode(d, "none", log=lambda *_: None), "none")

    def test_check_meta_match_and_mismatch(self):
        import json
        import os
        import tempfile
        from unittest import mock

        from diff_vla_wbc.quant import check_meta_mode

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "vla_meta.json"), "w") as fh:
                json.dump({"fm_steps": 4, "token_quant": "ste"}, fh)
            # ste ↔ hard 在部署侧等价 ⇒ 放行
            self.assertEqual(check_meta_mode(d, "hard", log=lambda *_: None), "ste")
            os.environ.pop("ALLOW_TOKEN_QUANT_MISMATCH", None)
            with self.assertRaises(SystemExit):
                check_meta_mode(d, "none", log=lambda *_: None)
            with mock.patch.dict(os.environ, {"ALLOW_TOKEN_QUANT_MISMATCH": "1"}):
                self.assertEqual(check_meta_mode(d, "none", log=lambda *_: None), "ste")

    def test_ste_equals_hard_on_deploy_forward(self):
        """部署侧没有梯度：ste 与 hard 必须给出**逐位相同**的值。"""
        from diff_vla_wbc.quant import quantize_np

        x = torch.rand(8, 64).numpy() * 3 - 1.5
        a, _ = quantize_np(x, "ste")
        b, _ = quantize_np(x, "hard")
        self.assertTrue((a == b).all())
