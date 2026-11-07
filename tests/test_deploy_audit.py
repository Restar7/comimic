#!/usr/bin/env python3
"""部署链路的**运行时口径**测试（任务书 §7 第 9 项 + 第 10 项的数据部分）。

不启动 Isaac/仿真，只测"决定结论是否可信"的那几个判定：

1. ``audit_provider_check``：日志里只有 SonicActionProvider / 只有 SuperMimic /
   两者都有 / 都没有 —— 四种情形必须给出确定的判定，且期望不符时**非零退出**。
2. ``audit_delta_required``：含 ``delta_token_state`` 的 head ckpt 在
   ``HA_ALLOW_SONIC!=1`` 时必须 FAIL（Δ 只在 SONIC 执行器上语义成立）。
3. ``audit_lib_clip_row``：参考库 clip 的起始行必须与 ``SM_TOKEN_START`` 一致
   （真实数据：pp_box 从 246502 行开始，默认 0 ⇒ 必须 FAIL）。
4. 关节序契约：``joint_contract`` 的 ha↔std 置换互逆；真实参考库 ``packed.pt`` 的 q 块
   与录制里的 HA 序 q 在该置换下对齐（数据驱动，不靠猜）。

跑法::

    python3 -m pytest tests/test_deploy_audit.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HELPER = ROOT / "scripts" / "_deploy_audit.sh"
def _data_root() -> Path:
    """数据根：worktree 里没有 gitignore 的 runs/data ⇒ 回退到主 checkout。"""
    env = os.environ.get("COMIMIC_DATA_ROOT")
    if env:
        return Path(env)
    if (ROOT / "data" / "ha_raw").exists():
        return ROOT
    alt = ROOT.parent / "comimic"
    return alt if alt.exists() else ROOT


DATA_ROOT = _data_root()

pytestmark = pytest.mark.skipif(not HELPER.exists(), reason="缺少 scripts/_deploy_audit.sh")


def _bash(script: str, env: dict | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=e,
                          cwd=str(ROOT), timeout=300)


# --------------------------------------------------------------------------- 1) provider
def test_provider_check_sonic_ok(tmp_path):
    log = tmp_path / "a.log"
    log.write_text("[SonicActionProvider] init\n[SonicActionProvider] ready\n")
    r = _bash(f"source '{HELPER}'; audit_init '{tmp_path}/aud'; "
              f"audit_provider_check '{log}' sonic")
    assert r.returncode == 0, r.stderr
    assert "provider=sonic" in r.stdout


def test_provider_check_supermimic_mismatch_fails(tmp_path):
    log = tmp_path / "b.log"
    log.write_text("[SuperMimic] ckpt loaded\n[SuperMimic] online token source\n")
    r = _bash(f"source '{HELPER}'; audit_init '{tmp_path}/aud'; "
              f"audit_provider_check '{log}' sonic")
    assert r.returncode != 0
    assert "FAIL" in r.stderr and "supermimic" in (r.stdout + r.stderr)


def test_provider_check_mixed_fails(tmp_path):
    log = tmp_path / "c.log"
    log.write_text("[SonicActionProvider] x\n[SuperMimic] y\n")
    r = _bash(f"source '{HELPER}'; audit_init '{tmp_path}/aud'; "
              f"audit_provider_check '{log}' sonic")
    assert r.returncode != 0


def test_provider_check_unknown_fails_even_for_any(tmp_path):
    log = tmp_path / "d.log"
    log.write_text("nothing useful here\n")
    r = _bash(f"source '{HELPER}'; audit_init '{tmp_path}/aud'; "
              f"audit_provider_check '{log}' any")
    assert r.returncode != 0
    assert "不能判定执行器" in r.stderr


def test_provider_check_supermimic_expected_ok(tmp_path):
    log = tmp_path / "e.log"
    log.write_text("[SuperMimic] ok\n")
    r = _bash(f"source '{HELPER}'; audit_init '{tmp_path}/aud'; "
              f"audit_provider_check '{log}' supermimic")
    assert r.returncode == 0


# --------------------------------------------------------------------------- 2) Δtoken 门槛
def _make_head(tmp_path: Path, with_delta: bool) -> Path:
    import torch

    sys.path.insert(0, str(ROOT))
    from diff_vla_wbc.delta_token import DeltaTokenNet

    net = DeltaTokenNet(feat_dim=64, hidden=8, depth=2, init_std=0.0)
    ck = {"fm": {}, "step": 1}
    if with_delta:
        ck["delta_token_state"] = net.state_dict()
        ck["delta_token_meta"] = {"hidden": 8, "depth": 2, "cap": 0.0}
    p = tmp_path / "vla_test.pt"
    torch.save(ck, p)
    return p


def test_delta_head_requires_sonic(tmp_path):
    head = _make_head(tmp_path, True)
    r = _bash(f"source '{HELPER}'; audit_init '{tmp_path}/aud'; audit_delta_required '{head}'",
              env={"HA_ALLOW_SONIC": "0"})
    assert r.returncode != 0, r.stdout
    assert "delta_token_state" in r.stderr


def test_delta_head_ok_with_sonic(tmp_path):
    head = _make_head(tmp_path, True)
    r = _bash(f"source '{HELPER}'; audit_init '{tmp_path}/aud'; audit_delta_required '{head}'",
              env={"HA_ALLOW_SONIC": "1"})
    assert r.returncode == 0, r.stderr


def test_delta_head_escape_hatch(tmp_path):
    head = _make_head(tmp_path, True)
    r = _bash(f"source '{HELPER}'; audit_init '{tmp_path}/aud'; audit_delta_required '{head}'",
              env={"HA_ALLOW_SONIC": "0", "ALLOW_DELTA_WITH_SUPERMIMIC": "1"})
    assert r.returncode == 0


def test_head_without_delta_passes(tmp_path):
    head = _make_head(tmp_path, False)
    r = _bash(f"source '{HELPER}'; audit_init '{tmp_path}/aud'; audit_delta_required '{head}'",
              env={"HA_ALLOW_SONIC": "0"})
    assert r.returncode == 0


# --------------------------------------------------------------------------- 3) 参考库行号
LIB = DATA_ROOT / "runs/isaac_motion/ha_cmd/packed.pt"


@pytest.mark.skipif(not LIB.exists(), reason="缺少 runs/isaac_motion/ha_cmd/packed.pt")
def test_lib_clip_row_double_desk_start0_ok(tmp_path):
    r = _bash(f"source '{HELPER}'; audit_init '{tmp_path}/aud'; "
              f"audit_lib_clip_row '{LIB}' HOI_double_desk 0")
    assert r.returncode == 0, r.stderr
    assert "match=YES" in r.stdout


@pytest.mark.skipif(not LIB.exists(), reason="缺少 packed.pt")
def test_lib_clip_row_ppbox_start0_fails(tmp_path):
    """**真 bug 的回归测试**：pp_box 的 clip 不在第 0 行，默认 TOKEN_START=0 必须 FAIL。"""
    r = _bash(f"source '{HELPER}'; audit_init '{tmp_path}/aud'; "
              f"audit_lib_clip_row '{LIB}' HOI_pp_box 0")
    assert r.returncode != 0
    assert "246502" in (r.stdout + r.stderr)


# --------------------------------------------------------------------------- 4) 关节序
def test_joint_permutation_inverse():
    import joint_contract as JC

    p = np.asarray(JC.permutation("ha_v31", "standard"))
    inv = np.asarray(JC.permutation("standard", "ha_v31"))
    x = np.arange(29)
    assert np.array_equal(x[p][inv], x)
    assert np.array_equal(x[inv][p], x)
    assert len(set(p.tolist())) == 29


def test_std_to_ha_matches_joint_contract():
    from diff_vla_wbc.sonic_mjlab import ha_to_std, std_to_ha

    import joint_contract as JC

    assert std_to_ha() == list(JC.permutation(src="standard", dst="ha_v31"))
    assert ha_to_std() == list(JC.permutation(src="ha_v31", dst="standard"))


@pytest.mark.skipif(not (DATA_ROOT / "data/ha_raw").exists(), reason="缺少 data/ha_raw")
def test_packed_lib_q_is_standard_order_vs_raw_ha():
    """数据驱动证明：训练参考库 ``packed.pt`` 的 q 是 STANDARD 序，而录制 ``vla_state``
    的 dof_pos 是 HA 序 —— 这才是 Δtoken 特征必须置换的依据（不是猜的）。"""
    import torch

    import joint_contract as JC

    p = np.asarray(JC.permutation("ha_v31", "standard"))
    files = sorted((DATA_ROOT / "data/ha_raw").glob("*/*/*.npz"))
    if not files:
        pytest.skip("没有录制")
    raw = np.load(files[0], allow_pickle=True)
    q_ha = np.asarray(raw["vla_state_dof_pos_29"], np.float32)      # HA 序（与 env 逐位相等）
    d = torch.load(str(LIB), map_location="cpu", weights_only=False)
    q_lib = d["ref"][: len(q_ha), 13:42].float().numpy()            # 库的 q 块

    def _agree(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(29)]))

    as_is = _agree(q_lib, q_ha)
    permuted = _agree(q_lib, q_ha[:, p])
    # 库与录制不是同一帧对齐（脚本库是命令流），但相对关系必须成立
    assert permuted > as_is + 0.2, f"as_is={as_is:.3f} permuted={permuted:.3f}"
