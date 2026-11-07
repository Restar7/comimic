"""部署观测布局（那 930 维）的**逐位对齐**测试。

背景：训练侧 `mdp.obs()` 每帧 93 维、块顺序 ``[q-def, qd, ω, g, a_prev]``、关节 STANDARD 序；
部署侧（SONIC provider）每帧同样的量但**块顺序** `[ω, q-def, qd, a_prev, g]`、
关节 **HA_V31** 序、且 obs 里没有训练时那个随机偏置 ``dq0``（部署 xi=0）。

本文件用**同一份物理量**走两条路：
  ① 训练侧：先拼 93 维 obs（含 xi 偏置）→ `to_deploy_obs_hist(...)`
  ② 部署侧：`from_raw_histories(...)`（provider 的同款拼装）
断言两者逐位相等。相等 ⇒ 以后"训练拿这批特征"和"部署喂那批特征"是同一个东西。
"""
from __future__ import annotations

import numpy as np
import pytest

from diff_vla_wbc.deploy_obs import (
    DEPLOY_BLOCKS,
    DEPLOY_HIST_DIM,
    DEPLOY_OBS_DIM,
    from_raw_histories,
    gravity_dir_from_quat_wxyz,
    to_deploy_obs_hist,
)

FRAMES = 10
NJ = 29


def _rand_inputs(seed: int = 0, frames: int = FRAMES, batch: int = 4):
    g = np.random.default_rng(seed)
    return dict(
        joint_pos=g.normal(0.0, 0.3, (frames, batch, NJ)),
        joint_vel=g.normal(0.0, 0.5, (frames, batch, NJ)),
        ang_vel=g.normal(0.0, 0.5, (frames, batch, 3)),
        gravity=g.normal(0.0, 0.3, (frames, batch, 3)),
        last_action=g.normal(0.0, 0.2, (frames, batch, NJ)),
        default_q=g.normal(0.0, 0.2, (NJ,)),
        xi_dq0=g.normal(0.0, 0.05, (batch, NJ)),
    )


def test_dims_match_provider_layout():
    assert DEPLOY_HIST_DIM == 930
    assert DEPLOY_OBS_DIM == 994
    assert [n for n, _ in DEPLOY_BLOCKS] == [
        "ang_vel", "joint_pos_delta", "joint_vel", "last_action", "gravity"]


def test_training_path_equals_deployment_path():
    """① 训练侧拼装 + 转换  ==  ② 部署侧同款拼装（逐位）。"""
    d = _rand_inputs()
    frames, batch, _ = d["joint_pos"].shape

    # ---- ① 训练侧：按 mdp.obs() 的公式拼 93 维（关节块仍是 STANDARD 序） ----
    train_hist = np.concatenate([
        d["joint_pos"] - d["default_q"][None, None, :] - d["xi_dq0"][None],
        d["joint_vel"],
        d["ang_vel"],
        d["gravity"],
        d["last_action"],
    ], axis=-1)                      # (frames, B, 93)
    assert train_hist.shape[-1] == 93
    got = to_deploy_obs_hist(train_hist, xi_dq0=d["xi_dq0"])

    # ---- ② 部署侧：直接从原始量拼 ----
    want = from_raw_histories(
        joint_pos=d["joint_pos"], joint_vel=d["joint_vel"], ang_vel=d["ang_vel"],
        gravity=d["gravity"], last_action=d["last_action"],
        default_q=d["default_q"])

    assert got.shape == want.shape == (batch, 930)
    assert np.allclose(got, want, atol=1e-12), \
        f"max|Δ|={np.abs(got - want).max():.3e}"


def test_block_contents_in_deployment_order():
    """块的**顺序**和每个块的**语义**都不能错（逐个块核对，而不是只看总误差）。"""
    d = _rand_inputs(seed=1)
    want = from_raw_histories(
        joint_pos=d["joint_pos"], joint_vel=d["joint_vel"], ang_vel=d["ang_vel"],
        gravity=d["gravity"], last_action=d["last_action"], default_q=d["default_q"])

    import joint_contract as JC
    perm = np.asarray(JC.permutation("standard", "ha_v31"))
    frames = d["joint_pos"].shape[0]
    off = 0
    expect_blocks = {
        "ang_vel": d["ang_vel"],
        "joint_pos_delta": (d["joint_pos"] - d["default_q"][None, None, :])[..., perm],
        "joint_vel": d["joint_vel"][..., perm],
        "last_action": d["last_action"][..., perm],
        "gravity": d["gravity"],
    }
    for name, width in DEPLOY_BLOCKS:
        seg = want[:, off:off + frames * width].reshape(len(want), FRAMES, width)
        want_blk = expect_blocks[name].transpose(1, 0, 2)   # (B, frames, width)
        assert np.allclose(seg, want_blk, atol=1e-12), f"块 {name} 不对"
        off += FRAMES * width
    assert off == DEPLOY_HIST_DIM


def test_joint_order_is_material():
    """如果忘了换关节序，结果必须**明显不同**（防止"其实没换也没人发现"）。"""
    d = _rand_inputs(seed=2)
    a = to_deploy_obs_hist(
        np.concatenate([d["joint_pos"] - d["default_q"][None, None, :] - d["xi_dq0"][None],
                        d["joint_vel"], d["ang_vel"], d["gravity"], d["last_action"]], -1),
        xi_dq0=d["xi_dq0"], src_order="standard", dst_order="ha_v31")
    b = to_deploy_obs_hist(
        np.concatenate([d["joint_pos"] - d["default_q"][None, None, :] - d["xi_dq0"][None],
                        d["joint_vel"], d["ang_vel"], d["gravity"], d["last_action"]], -1),
        xi_dq0=d["xi_dq0"], src_order="standard", dst_order="standard")
    assert np.abs(a - b).max() > 1e-3


def test_gravity_convention_matches_training():
    """``g_b = -R[...,2,:]``（训练侧） == ``gravity_dir_from_quat_wxyz``（部署侧）。"""
    rng = np.random.default_rng(3)
    q = rng.normal(size=(64, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q.T
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    train_g = -np.stack([r20, r21, r22], -1)
    deploy_g = gravity_dir_from_quat_wxyz(q)
    assert np.allclose(train_g, deploy_g, atol=1e-6)


def test_wrong_width_raises():
    d = _rand_inputs(seed=4)
    bad = np.zeros((FRAMES, 4, 92), np.float32)
    with pytest.raises(ValueError):
        to_deploy_obs_hist(bad)
