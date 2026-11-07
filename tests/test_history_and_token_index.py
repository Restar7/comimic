"""History10 对齐 + 时间轴索引（规范 §4.5 / §9 `tests/test_history_and_token_index.py`）。

必须是**真实窗口**：用主仓库 `runs/harness_03_bptt/big/windows.pt`（3264 个窗口）取 N=32，
逐位检查：
1. 训练侧 ``history[:, :, :58]``（obs93 的 q/qd 块，oldest→newest）；
2. 部署路径 ``hist10_from_hist64``（同一状态序列的 ``state64`` 的 q/qd 块，减 default）；
3. 二者 ``max|Δ| ≤ 1e-6``；
4. 决策帧（``[:, -1]``）的 q 块 = ``state64[6:35] − default``、qd 块 = ``state64[35:64]``；
5. 帧序 = oldest→newest（``[-1]`` 是决策时刻，不是 t+1）。

外加时间轴合同：``ℓ(k)=k``、``K=min(H,T)``、``reorder_joints`` 往返。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from diff_vla_wbc.delta_token_io import reorder_joints
from diff_vla_wbc.delta_token_io_v5 import joint_permutation
from diff_vla_wbc.delta_token_io_v5 import (
    DeltaTokenContractError, DeltaTokenSpecV5, delta_hist_from_history10,
    exec_token_count, hist10_from_blocks, hist10_from_hist64, token_index)

MAIN = Path(os.environ.get("COMIMIC_MAIN",
                           "/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic"))
BIG = MAIN / "runs/harness_03_bptt/big/windows.pt"


def _spec(default_joint_pos) -> DeltaTokenSpecV5:
    return DeltaTokenSpecV5(T=40, K=20, H=20, use_history=True,
                            meta=dict(default_joint_pos=np.asarray(default_joint_pos,
                                                                   np.float32).tolist()))


@pytest.mark.skipif(not BIG.exists(), reason=f"缺真实窗口 {BIG}")
def test_hist10_parity_real_windows():
    d = torch.load(str(BIG), map_location="cpu", weights_only=False)
    n = 32
    h = d["history"].float()[:n]                 # (32,10,93)
    st = d["state"].float()[:n]                  # (32,71) 决策帧状态
    hist10_train = h[:, :, :58].contiguous()     # [q(29)|qd(29)]，oldest→newest

    # ---- 4/5：决策帧的 q/qd 必须与 state64 的两个块逐位一致（差一个常数 default）----
    off = hist10_train[:, -1, :29] - st[:, 7:36]          # = −default（应与窗口无关）
    d_est = -off.mean(0)
    assert float((off - off.mean(0, keepdim=True)).abs().max()) <= 1e-6, \
        "q 块与 state64 的 dof_pos 差不是常数 ⇒ 关节序不一致（规范 §1.3）"
    assert float((hist10_train[:, -1, :29] - (st[:, 7:36] - d_est)).abs().max()) <= 1e-6
    assert float((hist10_train[:, -1, 29:] - st[:, 42:71]).abs().max()) <= 1e-6

    # ---- 2/3：部署路径（state64 序列的 q/qd 块，**HA_V31 序**）反推 hist10 ----
    # 注意：训练侧 hist10 是 **standard 序**；部署序 HA_V31 必须先换过来再比。
    spec = _spec(d_est)                 # spec 默认声明入站 = HA_V31
    perm_s2h = joint_permutation("standard", "ha_v31")
    q_abs_std = (hist10_train[:, :, :29] + d_est[None, None]).reshape(-1, 29).numpy()
    qd_std = hist10_train[:, :, 29:].reshape(-1, 29).numpy()
    hist64 = torch.zeros(n, 10, 64)
    hist64[:, :, 6:35] = torch.from_numpy(q_abs_std[:, perm_s2h].reshape(n, 10, 29))
    hist64[:, :, 35:64] = torch.from_numpy(qd_std[:, perm_s2h].reshape(n, 10, 29))
    hist10_deploy = torch.stack(
        [torch.from_numpy(hist10_from_hist64(hist64[i].numpy(), spec)) for i in range(n)], 0)
    assert float((hist10_train - hist10_deploy).abs().max()) <= 1e-6
    # 块序断言（等价于 hist10_from_blocks 的两块）
    perm_s2h_t = torch.from_numpy(perm_s2h)
    for i in range(n):
        # 入站块：q_offset 与 qd 都在部署序（HA_V31）
        q_off_ha = (hist10_train[i, :, :29] + 0.0)[:, perm_s2h_t].numpy()
        qd_ha = hist10_train[i, :, 29:][:, perm_s2h_t].numpy()
        blk = hist10_from_blocks(q_off_ha, qd_ha, spec)
        assert float(np.abs(blk - hist10_train[i].numpy()).max()) <= 1e-6
    # 帧序：最后一帧是决策时刻，且不是重复帧（否则 oldest→newest 说明不成立）
    diff_frames = (hist10_train[:, -1] - hist10_train[:, 0]).abs().max(-1).values
    assert float((diff_frames > 1e-6).float().mean()) >= 0.9


def test_hist10_from_blocks_algebra():
    rng = np.random.default_rng(0)
    q_std = rng.normal(size=(10, 29)).astype(np.float32)
    qd_std = rng.normal(size=(10, 29)).astype(np.float32)
    spec = _spec(np.zeros(29, np.float32))          # 入站声明 = HA_V31
    p_s2h = joint_permutation("standard", "ha_v31")
    q_ha = q_std[:, p_s2h]
    qd_ha = qd_std[:, p_s2h]
    out = hist10_from_blocks(q_ha, qd_ha, spec)
    assert out.shape == (10, 58) and out.dtype == np.float32
    assert np.allclose(out[:, :29], q_std) and np.allclose(out[:, 29:], qd_std)
    # 绝对角 + default（**standard 序**）⇒ 等价于"已减 default"
    dq = np.full(29, 0.37, np.float32)
    spec2 = _spec(dq)
    out2 = hist10_from_blocks(q_ha + dq[p_s2h][None, :], qd_ha, spec2, default_joint_pos=dq)
    assert np.allclose(out2, out)


def test_hist10_fail_closed():
    spec = _spec(np.zeros(29, np.float32))
    with pytest.raises(DeltaTokenContractError):
        hist10_from_blocks(np.zeros((9, 29), np.float32), np.zeros((10, 29), np.float32), spec)
    with pytest.raises(DeltaTokenContractError):
        hist10_from_hist64(np.zeros((10, 63), np.float32), spec)
    with pytest.raises(DeltaTokenContractError):
        delta_hist_from_history10(np.zeros((10, 58), np.float64), spec)     # dtype 不符
    with pytest.raises(DeltaTokenContractError):
        delta_hist_from_history10(np.zeros((10, 57), np.float32), spec)     # 形状不符
    with pytest.raises(DeltaTokenContractError):                            # 缺 default
        hist10_from_hist64(np.zeros((10, 64), np.float32), DeltaTokenSpecV5(T=40, K=20, H=20))


def test_token_index_is_identity():
    T = 40
    for k in range(T):
        assert token_index(k, T) == k
    for k in range(T, T + 5):
        assert token_index(k, T) == T - 1


def test_exec_k_is_min_h_t():
    assert exec_token_count(20, 40) == 20
    assert exec_token_count(50, 40) == 40
    assert exec_token_count(10, 40) == 10


def test_reorder_joints_roundtrip():
    x = np.arange(29, dtype=np.float32)
    y = reorder_joints(x, "ha_v31", "standard")
    z = reorder_joints(y, "standard", "ha_v31")
    assert np.array_equal(z, x)
    assert not np.array_equal(y, x)
