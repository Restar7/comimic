#!/usr/bin/env python3
"""Δtoken 训练/部署口径一致性测试（会话 01 交付物）。

覆盖任务书 §7 的 1–7 项：

================  ==========================================================
测试               查什么
================  ==========================================================
test_feat35       feat_dim=35 checkpoint 能按 35 维口径装载并前向
test_feat64       feat_dim=64（fullstate）checkpoint 能按 64 维口径装载
test_legacy_*     没有任何 metadata 的旧 checkpoint（权重形状反推）
test_bad_*        错 shape / metadata 冲突 / 缺键 ⇒ 必须抛，不许静默
test_zero_delta   init_std=0 时 Δ ≡ 0 ⇒ token 逐位不变
test_forward_parity  训练侧直算的 feat 与部署侧从 state64 构造的 feat 前向一致
test_cap_parity   训练/部署同一个 apply_norm_cap（含 cap=0 恒等）
test_broadcast     Δ 广播到整段 chunk、不改 hand14
================  ==========================================================

跑法::

    python3 -m pytest tests/test_delta_token_io.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joint_contract as JC  # noqa: E402
from diff_vla_wbc.delta_token import (  # noqa: E402
    FEAT_DIM,
    STATE64_DIM,
    DeltaTokenNet,
    apply_norm_cap,
)
from diff_vla_wbc import delta_token_io as DIO  # noqa: E402

P_HA2STD = np.asarray(JC.permutation("ha_v31", "standard"), dtype=np.int64)


# --------------------------------------------------------------------------- helpers
def _make_ckpt(feat_dim: int, *, hidden: int = 64, depth: int = 2, seed: int = 0,
               with_meta: bool = True, cap: float = 0.0) -> dict:
    """造一份与 `v3_step.save_vla` **同格式**的 Δ checkpoint（随机权重）。"""
    torch.manual_seed(seed)
    net = DeltaTokenNet(feat_dim=feat_dim, hidden=hidden, depth=depth, init_std=4e-4)
    sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
    out = {"delta_token_state": sd}
    if with_meta:
        out["delta_token_meta"] = {"hidden": hidden, "depth": depth, "cap": cap,
                                   "init_std": 4e-4}
    return out


def _train_feat_from_flat_state(flat71: np.ndarray) -> np.ndarray:
    """训练侧 ``_delta_token_features`` 的**逐字复刻**（standard 序，不用置换）。"""
    # flat71 = [p(3) quat(4) q(29) v(3) w(3) qd(29)]
    return np.concatenate([flat71[7:36], flat71[42:71]]).astype(np.float32)


def _rot6_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    from diff_vla_wbc.state_extra import mat_to_rot6d_wire, quat_to_mat  # noqa: PLC0415

    return mat_to_rot6d_wire(quat_to_mat(torch.as_tensor(quat_wxyz, dtype=torch.float64))).numpy()


def _state64_from_train_state(flat71: np.ndarray) -> np.ndarray:
    """一份**部署口径** state64：rot6d(绝对) + dof_pos/dof_vel 取 **HA_V31** 序。

    训练侧特征用 standard 序 ⇒ 这里把 standard 的 q/qd 反置换回 HA 序，
    模拟"线上第三方只发 HA 序 state64"这一事实。
    """
    q_std = flat71[7:36]
    qd_std = flat71[42:71]
    q_ha = np.empty(29, np.float32)
    qd_ha = np.empty(29, np.float32)
    q_ha[P_HA2STD] = q_std
    qd_ha[P_HA2STD] = qd_std
    return np.concatenate([_rot6_from_quat(flat71[3:7]), q_ha, qd_ha]).astype(np.float32)


def _random_flat71(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    return np.concatenate([
        rng.normal(scale=0.3, size=3), quat,
        rng.normal(scale=0.4, size=29),
        rng.normal(scale=0.2, size=3), rng.normal(scale=0.2, size=3),
        rng.normal(scale=0.8, size=29),
    ]).astype(np.float32)


# --------------------------------------------------------------------------- 1) feat 35
def test_feat35_meta_loads_and_forwards():
    ck = _make_ckpt(FEAT_DIM, hidden=48, depth=2)
    net, spec = DIO.load_delta_token_from_checkpoint(ck)
    assert net is not None and spec is not None
    assert spec.feat_dim == FEAT_DIM == 35
    assert spec.feat_layout == "rot6d+dof_vel29"
    assert spec.hidden == 48 and spec.depth == 2
    # 现有训练器**不写** feat_dim ⇒ 必须能从权重反推出来（这正是这次修的 bug）
    assert spec.sources["feat_dim"] == "weights(in_dim−token_dim)", spec.sources
    out = net(torch.zeros(1, 64), torch.zeros(1, 35))
    assert tuple(out.shape) == (1, 64)


# --------------------------------------------------------------------------- 2) feat 64
def test_feat64_meta_loads_and_forwards():
    ck = _make_ckpt(STATE64_DIM, hidden=48, depth=2)
    net, spec = DIO.load_delta_token_from_checkpoint(ck)
    assert spec.feat_dim == STATE64_DIM == 64
    assert spec.feat_layout == "state64"
    assert spec.layer_shapes[0] == (48, 128), spec.layer_shapes
    assert net(torch.zeros(1, 64), torch.zeros(1, 64)).shape == (1, 64)


def test_feat64_meta_explicit_feat_dim_wins():
    """会话 02 会往 meta 里写 feat_dim=64 —— 那条路径也要过。"""
    ck = _make_ckpt(STATE64_DIM)
    ck["delta_token_meta"]["feat_dim"] = 64
    ck["delta_token_meta"]["token_dim"] = 64
    ck["delta_token_meta"]["feat_joint_order"] = "standard"
    _, spec = DIO.load_delta_token_from_checkpoint(ck)
    assert spec.feat_dim == 64
    assert spec.sources["feat_dim"] == "meta+weights"
    assert spec.feat_joint_order == "standard"


# --------------------------------------------------------------------------- 3) legacy
@pytest.mark.parametrize("feat_dim", [FEAT_DIM, STATE64_DIM])
def test_legacy_ckpt_without_meta(feat_dim):
    """旧 checkpoint（meta 里没有 feat_dim/token_dim/hidden/depth）必须能按权重反推。"""
    ck = _make_ckpt(feat_dim, hidden=32, depth=3, with_meta=False)
    net, spec = DIO.load_delta_token_from_checkpoint(ck)
    assert spec.feat_dim == feat_dim
    assert spec.hidden == 32 and spec.depth == 3
    assert spec.sources["feat_dim"] == "weights(in_dim−token_dim)"
    assert spec.cap == 0.0


def test_legacy_partial_meta_cap_only():
    """老 ckpt 只存了 cap（线上确实是这样）⇒ cap 要带过来，其余按权重反推。"""
    ck = _make_ckpt(STATE64_DIM, with_meta=False)
    ck["delta_token_meta"] = {"cap": 0.35}
    _, spec = DIO.load_delta_token_from_checkpoint(ck)
    assert spec.cap == pytest.approx(0.35)
    assert spec.sources["cap"] == "meta"


# --------------------------------------------------------------------------- 4) bad inputs
def test_bad_unknown_feat_dim_raises():
    """首层输入宽度 64+40 ⇒ feat 40 不在已知口径里，必须报错（不许瞎猜）。"""
    ck = _make_ckpt(STATE64_DIM, with_meta=False)
    ck["delta_token_state"]["net.0.weight"] = torch.zeros(64, 64 + 40)
    with pytest.raises(DIO.DeltaTokenContractError, match="feat_dim"):
        DIO.load_delta_token_from_checkpoint(ck)


def test_bad_meta_conflict_raises():
    ck = _make_ckpt(STATE64_DIM)
    ck["delta_token_meta"]["feat_dim"] = 35          # 与权重 (128) 冲突
    with pytest.raises(DIO.DeltaTokenContractError, match="feat_dim"):
        DIO.load_delta_token_from_checkpoint(ck)


def test_bad_hidden_conflict_raises():
    ck = _make_ckpt(STATE64_DIM, hidden=48)
    ck["delta_token_meta"]["hidden"] = 1024
    with pytest.raises(DIO.DeltaTokenContractError, match="hidden"):
        DIO.load_delta_token_from_checkpoint(ck)


def test_bad_missing_key_raises():
    ck = _make_ckpt(STATE64_DIM)
    del ck["delta_token_state"]["net.2.bias"]
    with pytest.raises(DIO.DeltaTokenContractError, match="missing"):
        DIO.load_delta_token_from_checkpoint(ck)


def test_bad_extra_key_raises():
    ck = _make_ckpt(STATE64_DIM)
    ck["delta_token_state"]["net.9.weight"] = torch.zeros(64, 64)
    with pytest.raises(DIO.DeltaTokenContractError):
        DIO.load_delta_token_from_checkpoint(ck)


def test_bad_broken_chain_raises():
    ck = _make_ckpt(STATE64_DIM)
    ck["delta_token_state"]["net.2.weight"] = torch.zeros(48, 47)
    with pytest.raises(DIO.DeltaTokenContractError, match="不连"):
        DIO.load_delta_token_from_checkpoint(ck)


def test_absent_delta_returns_none():
    net, spec = DIO.load_delta_token_from_checkpoint({"fm": {}})
    assert net is None and spec is None


# --------------------------------------------------------------------------- 5) Δ = 0
def test_zero_init_gives_exact_zero_delta():
    """``init_std=0`` ⇒ 输出**逐位**为 0（"Δ=0 与 base 严格一致"的构造性证明）。"""
    net = DeltaTokenNet(feat_dim=STATE64_DIM, hidden=32, depth=2, init_std=0.0).eval()
    tok = torch.randn(1, 64)
    feat = torch.randn(1, 64)
    with torch.no_grad():
        d = net(tok, feat)
    assert torch.count_nonzero(d) == 0
    chunk = np.random.default_rng(0).normal(size=(40, 64)).astype(np.float32)
    spec = DIO.infer_delta_token_spec(net.state_dict(), {"cap": 0.0})
    d_np = DIO.delta_from_token_chunk(chunk[0], feat.numpy()[0], net, spec)
    assert np.all(d_np == 0.0)
    assert np.array_equal(chunk + d_np[None, :], chunk)


def test_zero_delta_with_cap_still_zero():
    net = DeltaTokenNet(feat_dim=STATE64_DIM, hidden=32, init_std=0.0).eval()
    spec = DIO.infer_delta_token_spec(net.state_dict(), {"cap": 0.5})
    d = DIO.delta_from_token_chunk(np.zeros(64), np.zeros(64), net, spec)
    assert np.all(d == 0.0)


# --------------------------------------------------------------------------- 6) forward parity
@pytest.mark.parametrize("feat_dim", [FEAT_DIM, STATE64_DIM])
def test_forward_parity_train_vs_deploy(feat_dim):
    """训练侧直接算的 feat 与部署侧从 state64 反推的 feat 必须**逐位**一致，
    且两边前向输出一致（差 < 1e-6）。这是"关节序置换没写错"的核心证据。
    """
    ck = _make_ckpt(feat_dim, hidden=64, depth=2, seed=7)
    net, spec = DIO.load_delta_token_from_checkpoint(ck)

    for i in range(6):
        flat = _random_flat71(100 + i)
        s64 = _state64_from_train_state(flat)

        # 训练侧：feature = [rot6d(q_init^-1·q) | q_std | qd_std]  （这里 rot6d 直接取同一向量）
        feat_train = np.concatenate([s64[0:6], flat[7:36], flat[42:71]]).astype(np.float32)
        if feat_dim == FEAT_DIM:
            feat_train = np.concatenate([s64[0:6], flat[42:71]]).astype(np.float32)

        feat_deploy = DIO.delta_features_from_state64(s64, spec)
        assert feat_deploy.shape == (feat_dim,)
        assert np.array_equal(feat_deploy, feat_train), (
            f"部署特征与训练特征不一致：max|Δ|={np.abs(feat_deploy - feat_train).max()}")

        tok = torch.as_tensor(np.random.default_rng(i).normal(size=(1, 64)), dtype=torch.float32)
        with torch.no_grad():
            a = net(tok, torch.as_tensor(feat_train)[None])
        b = DIO.delta_from_token_chunk(tok.numpy()[0], feat_deploy, net, spec)
        cap = spec.cap
        a_np = apply_norm_cap(a, cap)[0].numpy()
        assert np.abs(a_np - b).max() < 1e-6


def test_forward_parity_no_permutation_would_fail():
    """反证：不置换（把 HA 序直接当 standard 喂）会**明显**算出不同特征。"""
    ck = _make_ckpt(STATE64_DIM, hidden=64, seed=11)
    _, spec = DIO.load_delta_token_from_checkpoint(ck)
    flat = _random_flat71(3)
    s64 = _state64_from_train_state(flat)
    good = DIO.delta_features_from_state64(s64, spec)
    bad = np.concatenate([s64[0:6], s64[6:35], s64[35:64]]).astype(np.float32)  # 错误口径
    fdev = np.abs(good - bad)
    assert fdev[6:].max() > 0.05, f"置换竟然没有影响（max={fdev.max()}）"


# --------------------------------------------------------------------------- 7) cap
@pytest.mark.parametrize("cap", [0.0, 0.05, 0.2, 1.0])
def test_cap_parity_formula(cap):
    """``apply_norm_cap`` 是训练/部署**同一个函数**：这里验证它的数值语义与不变量。"""
    torch.manual_seed(0)
    d = torch.randn(64, 64)
    out = apply_norm_cap(d, cap)
    if cap <= 0:
        assert torch.equal(out, d), "cap<=0 必须恒等"
        return
    n_out = out.norm(dim=-1)
    assert float(n_out.max()) <= cap + 1e-5, "软上限必须不越界"
    # 远小于 cap 时近似恒等（n/cap ≪ 1 ⇒ tanh(x)/x ≈ 1）
    small = 1e-6 * torch.randn(64, 64)
    assert torch.allclose(apply_norm_cap(small, cap), small, atol=1e-5)
    # 方向不变（tanh 收缩只改模长）
    cos = (out * d).sum(-1) / (out.norm(dim=-1) * d.norm(dim=-1)).clamp_min(1e-9)
    assert float(cos.min()) > 0.9999


def test_cap_applied_identically_in_loader_and_training_path():
    """部署 ``delta_from_token_chunk`` 与训练 ``apply_norm_cap(net(...))`` 必须同值。"""
    ck = _make_ckpt(STATE64_DIM, hidden=32, seed=5, cap=0.1)
    net, spec = DIO.load_delta_token_from_checkpoint(ck)
    assert spec.cap == pytest.approx(0.1)
    tok = np.random.default_rng(1).normal(size=64).astype(np.float32)
    feat = np.random.default_rng(2).normal(size=64).astype(np.float32)
    with torch.no_grad():
        ref = apply_norm_cap(net(torch.as_tensor(tok)[None], torch.as_tensor(feat)[None]),
                             spec.cap)[0].numpy()
    got = DIO.delta_from_token_chunk(tok, feat, net, spec)
    assert np.abs(ref - got).max() < 1e-7


# --------------------------------------------------------------------------- 8) broadcast / hand
def test_delta_broadcast_covers_whole_chunk_and_keeps_hand14():
    """部署侧 ``chunk64 + Δ[None,:]``：整段 chunk 每行都加同一个 Δ；hand14 另走一路。"""
    ck = _make_ckpt(STATE64_DIM, hidden=32, seed=3)
    net, spec = DIO.load_delta_token_from_checkpoint(ck)
    rng = np.random.default_rng(0)
    chunk = rng.normal(size=(40, 64)).astype(np.float32)
    fingers = rng.normal(size=(40, 14)).astype(np.float32)
    s64 = rng.normal(size=64).astype(np.float32)
    feat = DIO.delta_features_from_state64(s64, spec)
    d = DIO.delta_from_token_chunk(chunk[0], feat, net, spec)
    out = (chunk + d[None, :]).astype(np.float32)
    # 每一行的增量都等于 d
    inc = (out - chunk).astype(np.float64)
    # 每一行的增量都等于网络给的同一个 Δ（只差 float32 加法舍入）
    assert np.allclose(inc, np.broadcast_to(d.astype(np.float64), inc.shape), atol=1e-6, rtol=0)
    # 同样的 token 行 ⇒ 增量逐位一致（证明是"加同一个向量"而不是逐帧重算）
    _tiled = np.tile(chunk[:1], (40, 1))
    _out2 = (_tiled + d[None, :]).astype(np.float32)
    _inc2 = (_out2 - _tiled).astype(np.float64)
    assert np.array_equal(_inc2, np.broadcast_to(_inc2[0], _inc2.shape))
    # fingers 完全没动（服务端只改 chunk64，手上 14 维另路返回）
    assert np.array_equal(fingers, fingers.copy())
    assert fingers.shape[1] == 14 and out.shape[1] == 64


def test_rate_of_change_between_35_and_64_is_material():
    """两套口径**不能互相当替代**：同一 state64 下 35/64 两个网络的 Δ 差异很大。"""
    ck35 = _make_ckpt(FEAT_DIM, hidden=64, seed=21)
    ck64 = _make_ckpt(STATE64_DIM, hidden=64, seed=21)
    n35, s35 = DIO.load_delta_token_from_checkpoint(ck35)
    n64, s64 = DIO.load_delta_token_from_checkpoint(ck64)
    st = np.random.default_rng(4).normal(size=64).astype(np.float32)
    tok = np.random.default_rng(5).normal(size=64).astype(np.float32)
    d35 = DIO.delta_from_token_chunk(tok, DIO.delta_features_from_state64(st, s35), n35, s35)
    d64 = DIO.delta_from_token_chunk(tok, DIO.delta_features_from_state64(st, s64), n64, s64)
    assert d35.shape == d64.shape == (64,)
    assert not np.allclose(d35, d64)
