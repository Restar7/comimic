"""SONIC release decoder 的可微 torch 版（``diff_vla_wbc.sonic_decoder``）。

只钉两件在训练/度量里会静默出错的事情：

1. **``act`` 对 token 可微**，且和"先把 token 拼进 994 维再 forward"逐位一致。
   ``x[..., :64] = token`` 这种原地写法会静默断图（反传时 ``∂a/∂token`` 直接变 0，不报错）,
   所以实现用 ``torch.cat``;这里用一个**方向导数**对拍把它钉住。
2. **参数冻结**：只借它的 Jacobian，不允许它被 optimizer 顺手更新。

ONNX 不在本机时整个类跳过（它来自 workspace 层的 release 目录，不是仓库资产）。
"""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from diff_vla_wbc.sonic_decoder import DECODER_IN_DIM, TOKEN_DIM, SonicDecoderTorch

_ONNX = (Path(__file__).resolve().parents[1].parent / "GR00T-WholeBodyControl"
         / "gear_sonic_deploy" / "policy" / "release" / "model_decoder.onnx")


@unittest.skipUnless(_ONNX.is_file(), f"release decoder onnx 不在本机: {_ONNX}")
class TestSonicDecoderTorch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dec = SonicDecoderTorch.from_onnx(_ONNX)
        g = torch.Generator().manual_seed(0)
        cls.obs = torch.rand(4, DECODER_IN_DIM, generator=g) * 0.5
        cls.tok = torch.rand(4, TOKEN_DIM, generator=g) * 2 - 1

    def test_shape_and_frozen(self):
        with torch.no_grad():
            a = self.dec(self.obs)
        self.assertEqual(tuple(a.shape), (4, 29))
        self.assertTrue(all(not p.requires_grad for p in self.dec.parameters()),
                        "decoder 参数必须冻结（只借 Jacobian）")
        self.assertEqual(sum(p.numel() for p in self.dec.parameters()), 10_184_221)

    def test_act_matches_manual_concat(self):
        with torch.no_grad():
            a_act = self.dec.act(self.tok, self.obs)
            a_man = self.dec(torch.cat([self.tok, self.obs[..., TOKEN_DIM:]], dim=-1))
        self.assertTrue(torch.allclose(a_act, a_man, atol=0, rtol=0))

    def test_token_gradient_is_alive(self):
        """断图的典型症状：梯度为 None 或全 0。"""
        tk = self.tok.clone().requires_grad_(True)
        a = self.dec.act(tk, self.obs)
        g = torch.autograd.grad(a.sum(), tk)[0]
        self.assertIsNotNone(g)
        self.assertGreater(float(g.abs().max()), 0.0, "∂a/∂token 全 0 ⇒ 图断了（原地赋值？）")

    def test_directional_derivative_matches_finite_difference(self):
        """∑a 的**方向导数** vs 有限差分。

        注意：这里只能比"标量输出的方向导数"。拿 ``grad_outputs=ones(29)`` 得到的标量梯度
        去和**逐维**差分比是错的（第一版脚本就踩了这个，cos 恒为 0.2）。
        """
        tk = self.tok.clone().requires_grad_(True)
        a = self.dec.act(tk, self.obs)
        g = torch.autograd.grad(a.sum(), tk)[0]

        d = torch.randn(TOKEN_DIM, generator=torch.Generator().manual_seed(1))
        d = d / d.norm()
        eps = 1e-4
        with torch.no_grad():
            a0 = self.dec.act(tk.detach(), self.obs).sum()
            a1 = self.dec.act(tk.detach() + eps * d, self.obs).sum()
        fd = float((a1 - a0) / eps)
        an = float((g * d).sum())
        self.assertAlmostEqual(fd, an, delta=max(1e-3, 0.02 * abs(an)),
                               msg=f"FD={fd:.6f} analytic={an:.6f}")


if __name__ == "__main__":
    unittest.main()


class TestLayout994(unittest.TestCase):
    """994 布局（block-major，每块由旧到新，``last_action`` = 解码器自己的输出）。

    用合成张量钉住**拼接顺序与形状**；真正的逐位验证在
    ``tests`` 之外的采集数据上做过（HA raw npz，`max|Δ| = 0.0`）。
    """

    def test_roundtrip(self):
        from diff_vla_wbc.sonic_decoder import (HIST, build_obs_994, split_obs_994)

        B = 3
        tok = torch.rand(B, 64)
        blk = {
            "ang_vel": torch.rand(B, HIST, 3), "joint_pos": torch.rand(B, HIST, 29),
            "joint_vel": torch.rand(B, HIST, 29), "last_action": torch.rand(B, HIST, 29),
            "gravity": torch.rand(B, HIST, 3),
        }
        obs = build_obs_994(tok, **blk)
        self.assertEqual(tuple(obs.shape), (B, 994))
        back = split_obs_994(obs)
        self.assertTrue(torch.equal(back["token"], tok))
        for k in blk:
            self.assertTrue(torch.equal(back[k], blk[k]), k)

    def test_block_offsets_match_spec(self):
        from diff_vla_wbc.sonic_decoder import BLOCKS

        # block-major：每个块连续放在 [start, end)，宽度 = 10 × per-frame
        for name, a, z, w in BLOCKS:
            self.assertEqual(z - a, 10 * w, name)
        self.assertEqual([b[1] for b in BLOCKS], [64, 94, 384, 674, 964])
        self.assertEqual(BLOCKS[-1][2], 994)

    def test_supermimic_permutation(self):
        """q_rel 加回 default_q/dq0；五个通道按 SONIC 顺序落位。"""
        from diff_vla_wbc.sonic_decoder import HIST, supermimic_obs_to_blocks

        B = 2
        o = torch.zeros(B, HIST, 93)
        o[..., 0:29] = 0.5          # q_rel
        o[..., 29:58] = 0.25        # qd
        o[..., 58:61] = 0.75        # ang_vel
        o[..., 61:64] = 0.1         # gravity
        o[..., 64:93] = 0.9         # last_action
        b = supermimic_obs_to_blocks(o, default_q=torch.full((29,), 0.2),
                                     dq0=torch.full((29,), 0.05))
        self.assertAlmostEqual(float(b["joint_pos"].mean()), 0.75, places=6)   # 0.5+0.2+0.05
        self.assertAlmostEqual(float(b["joint_vel"].mean()), 0.25, places=6)
        self.assertAlmostEqual(float(b["ang_vel"].mean()), 0.75, places=6)
        self.assertAlmostEqual(float(b["gravity"].mean()), 0.10, places=6)
        self.assertAlmostEqual(float(b["last_action"].mean()), 0.90, places=6)


@unittest.skipUnless(_ONNX.is_file(), f"release decoder onnx 不在本机: {_ONNX}")
class TestDecodeBatch1(unittest.TestCase):
    """batch=1 回归：opset 13 的 Squeeze 把 axes 放在**第二个输入**，取不到就会删掉 batch 维。

    这个 bug 只在 batch=1 时暴露，而部署侧就是 batch=1 —— 属于"静默但致命"。
    """

    def test_batch_one_shape_and_parity(self):
        import numpy as np

        dec = SonicDecoderTorch.from_onnx(_ONNX)
        torch.manual_seed(0)
        x = torch.randn(1, DECODER_IN_DIM)
        with torch.no_grad():
            y = dec(x)
        self.assertEqual(tuple(y.shape), (1, 29))
        try:
            import onnxruntime as ort
        except Exception:  # pragma: no cover
            self.skipTest("onnxruntime 不可用")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(str(_ONNX), so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        yo = sess.run(None, {name: x.numpy()})[0]
        self.assertLess(float(np.abs(y.numpy() - yo).max()), 1e-4)
