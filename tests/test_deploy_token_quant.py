"""部署侧 token 量化口径的一致性校验（``scripts/ha_vla_latent64_server.py``）。

为什么值得单独测：``token_quant=ste`` 时**训练前向喂的是格点值**。如果服务端还喂连续值，
学到的补偿就是对着一个不存在的执行器学的（`docs/TOKEN_QUANT_STE_20261104.md` §4）。
这套校验和 ``fm_steps`` 的校验同构：读 ``vla_meta.json``，不一致就 fail fast，
只留 ``ALLOW_TOKEN_QUANT_MISMATCH=1`` 一个人为放行开关。

测试直接 ``importlib`` 加载脚本（``scripts/`` 不是包），只构造 ``dry_run=True`` 的 Engine，
不加载任何权重。
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

SERVER = Path(__file__).resolve().parents[1] / "scripts" / "ha_vla_latent64_server.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_ha_vla_latent64_server_t", SERVER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)           # 只 import，不跑 main()
    return mod


class TestTokenQuantMetaCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def _engine(self, tmp: Path, arg_q: str):
        return self.mod.Engine(Namespace(dry_run=True, log_dir="", ckpt=tmp, token_quant=arg_q))

    def _meta(self, tmp: Path, payload: dict) -> None:
        (tmp / "vla_meta.json").write_text(json.dumps(payload))

    def test_missing_meta_falls_back_to_none(self):
        with tempfile.TemporaryDirectory() as d:
            eng = self._engine(Path(d), "none")
            self.assertEqual(eng._check_token_quant(), "none")

    def test_match(self):
        with tempfile.TemporaryDirectory() as d:
            self._meta(Path(d), {"fm_steps": 4, "token_quant": "ste"})
            self.assertEqual(self._engine(Path(d), "ste")._check_token_quant(), "ste")

    def test_none_meta_with_quantized_arg_also_raises(self):
        """口径必须**逐项一致**，双向都挡。

        train=none + deploy=量化：学到的预补偿是对着"连续输入的执行器"学的，部署却喂格点值，
        差半格（实测 0.8% 动作方向误差）—— 不是灾难，但足以让 A/B 结论不可比，
        所以同样要 ``ALLOW_TOKEN_QUANT_MISMATCH=1`` 才放行。
        """
        with tempfile.TemporaryDirectory() as d:
            self._meta(Path(d), {"fm_steps": 4, "token_quant": "none"})
            eng = self._engine(Path(d), "ste")
            import os
            os.environ.pop("ALLOW_TOKEN_QUANT_MISMATCH", None)
            with self.assertRaises(SystemExit):
                eng._check_token_quant()

    def test_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            self._meta(Path(d), {"fm_steps": 4, "token_quant": "ste"})
            eng = self._engine(Path(d), "none")
            with mock.patch.dict("os.environ", {}, clear=False):
                import os
                os.environ.pop("ALLOW_TOKEN_QUANT_MISMATCH", None)
                with self.assertRaises(SystemExit):
                    eng._check_token_quant()

    def test_mismatch_allowed_by_env(self):
        with tempfile.TemporaryDirectory() as d:
            self._meta(Path(d), {"fm_steps": 4, "token_quant": "ste"})
            eng = self._engine(Path(d), "none")
            with mock.patch.dict("os.environ", {"ALLOW_TOKEN_QUANT_MISMATCH": "1"}):
                self.assertEqual(eng._check_token_quant(), "ste")

    def test_ste_and_hard_are_equivalent_on_deploy(self):
        """推理没有梯度，ste/hard 的前向一样；只要"量化 or 不量化"一致就放行。"""
        with tempfile.TemporaryDirectory() as d:
            self._meta(Path(d), {"fm_steps": 4, "token_quant": "ste"})
            self.assertEqual(self._engine(Path(d), "hard")._check_token_quant(), "ste")


class TestEngineQuantModeDefault(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_default_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            eng = self.mod.Engine(Namespace(dry_run=True, log_dir="", ckpt=Path(d)))
            self.assertEqual(eng._quant_mode, "none")
            import math
            self.assertTrue(math.isnan(eng._quant_dist))   # 还没量化过 ⇒ 恒 NaN

    def test_mode_is_lowercased(self):
        with tempfile.TemporaryDirectory() as d:
            eng = self.mod.Engine(Namespace(dry_run=True, log_dir="", ckpt=Path(d),
                                            token_quant="STE"))
            self.assertEqual(eng._quant_mode, "ste")


if __name__ == "__main__":
    unittest.main()


class TestResolveTokenQuant(unittest.TestCase):
    """argparse 不校验字符串默认值 ⇒ 环境变量 TOKEN_QUANT=bogus 必须在这里被挡住。"""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_accepts_modes(self):
        for m in ("none", "ste", "hard", "STE", " none "):
            self.assertEqual(self.mod.resolve_token_quant(m), m.strip().lower())

    def test_empty_is_none(self):
        for v in (None, "", 0):
            self.assertEqual(self.mod.resolve_token_quant(v), "none")

    def test_rejects_garbage(self):
        with self.assertRaises(SystemExit):
            self.mod.resolve_token_quant("bogus")
