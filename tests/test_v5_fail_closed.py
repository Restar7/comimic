"""v5 fail-closed 门禁（规范 §4.7 / §12 `tests/test_v5_fail_closed.py`）。

缺 hist / 关节序不符 / T 不符 / arch 不符 / out_mode 不符 / `t≥K` 非零
⇒ **全部抛异常**，绝不静默降级（全 0 历史、广播 Δ、退回 Base 都算 FAIL）。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from diff_vla_wbc.delta_token_io import DeltaTokenContractError
from diff_vla_wbc.delta_token_io_v5 import (
    DeltaTokenSpecV5, apply_delta_per_frame, build_temporal_actor_v5, delta_from_chunk_full_v5,
    infer_delta_token_spec_v5, load_delta_token_v5_from_checkpoint, load_temporal_actor_v5,
    save_temporal_actor_v5, spec_v5_from_actor)
from diff_vla_wbc.delta_token_temporal_v5 import TemporalActorCfg, TemporalDeltaActorV5


def _actor():
    return TemporalDeltaActorV5(TemporalActorCfg(chunk_len=40))


def _spec(**kw):
    a = _actor()
    return spec_v5_from_actor(a, meta=dict(default_joint_pos=[0.0] * 29), **kw)


def test_arch_and_out_mode_rejected():
    with pytest.raises(DeltaTokenContractError):
        DeltaTokenSpecV5(arch="temporal_v4", T=40, K=20, H=20)
    with pytest.raises(DeltaTokenContractError):
        DeltaTokenSpecV5(out_mode="shared", T=40, K=20, H=20)


def test_history_contract_rejected():
    with pytest.raises(DeltaTokenContractError):
        DeltaTokenSpecV5(T=40, K=20, H=20, hist_clock="chunk_rate")
    with pytest.raises(DeltaTokenContractError):
        DeltaTokenSpecV5(T=40, K=20, H=20, hist_dim=64)
    with pytest.raises(DeltaTokenContractError):
        DeltaTokenSpecV5(T=40, K=20, H=20, hist_joint_order="unknown_order")


def test_k_greater_than_t_rejected():
    with pytest.raises(DeltaTokenContractError):
        DeltaTokenSpecV5(T=20, K=40, H=20)


def test_missing_history_raises():
    spec = _spec()
    net = build_temporal_actor_v5(spec)
    Z = np.zeros((40, 64), np.float32)
    s = np.zeros(64, np.float32)
    with pytest.raises(DeltaTokenContractError):
        delta_from_chunk_full_v5(Z, s, None, net, spec)


def test_wrong_history_shape_or_dtype_raises():
    spec = _spec()
    net = build_temporal_actor_v5(spec)
    Z = np.zeros((40, 64), np.float32)
    s = np.zeros(64, np.float32)
    with pytest.raises(DeltaTokenContractError):
        delta_from_chunk_full_v5(Z, s, np.zeros((10, 57), np.float32), net, spec)
    with pytest.raises(DeltaTokenContractError):
        delta_from_chunk_full_v5(Z, s, np.zeros((10, 58), np.float64), net, spec)
    h = np.zeros((10, 58), np.float32)
    h[0, 0] = np.nan
    with pytest.raises(DeltaTokenContractError):
        delta_from_chunk_full_v5(Z, s, h, net, spec)


def test_wrong_T_or_state_width_raises():
    spec = _spec()
    net = build_temporal_actor_v5(spec)
    h = np.zeros((10, 58), np.float32)
    with pytest.raises(DeltaTokenContractError):
        delta_from_chunk_full_v5(np.zeros((39, 64), np.float32), np.zeros(64, np.float32), h,
                                 net, spec)
    with pytest.raises(DeltaTokenContractError):
        delta_from_chunk_full_v5(np.zeros((40, 64), np.float32), np.zeros(63, np.float32), h,
                                 net, spec)


def test_net_spec_history_mismatch_raises():
    spec_nh = spec_v5_from_actor(TemporalDeltaActorV5(TemporalActorCfg(chunk_len=40,
                                                                       use_history=False)),
                                 meta=dict(default_joint_pos=[0.0] * 29))
    net = build_temporal_actor_v5(spec_nh)
    spec_h = _spec()
    with pytest.raises(DeltaTokenContractError):
        delta_from_chunk_full_v5(np.zeros((40, 64), np.float32), np.zeros(64, np.float32),
                                 np.zeros((10, 58), np.float32), net, spec_h)


def test_nonzero_tail_delta_rejected_by_deploy_contract():
    spec = _spec()
    chunk = np.zeros((40, 64), np.float32)
    d = np.zeros((40, 64), np.float32)
    d[25, 0] = 0.1                       # t ≥ K 非零 ⇒ 必须 FAIL
    with pytest.raises(DeltaTokenContractError):
        apply_delta_per_frame(chunk, d, spec)


def test_delta_is_zero_beyond_cap():
    spec = _spec()
    net = build_temporal_actor_v5(spec)
    # 把末层权重放大，看 cap 是否真的把 ‖Δ‖ 压住
    with torch.no_grad():
        net.head[-1].weight.mul_(50.0)
    d = delta_from_chunk_full_v5(np.random.randn(40, 64).astype(np.float32),
                                 np.zeros(64, np.float32), np.zeros((10, 58), np.float32),
                                 net, spec, cap=0.08)
    n = np.linalg.norm(d, axis=-1).max()
    assert n <= 0.08 + 1e-6, n
    assert float(np.abs(d[20:]).max()) == 0.0


def test_checkpoint_spec_requires_timeline_meta():
    a = _actor()
    sd = {k: v.detach().cpu() for k, v in a.state_dict().items()}
    with pytest.raises(DeltaTokenContractError):
        infer_delta_token_spec_v5(sd, {"arch": "temporal_v5"})       # 缺 T/K/H
    with pytest.raises(DeltaTokenContractError):
        infer_delta_token_spec_v5({}, {"arch": "temporal_v5", "T": 40, "K": 20, "H": 20})
    # T 与权重 PE 表不一致 ⇒ FAIL
    with pytest.raises(DeltaTokenContractError):
        infer_delta_token_spec_v5(sd, {"arch": "temporal_v5", "T": 20, "K": 20, "H": 20})


def test_v4_checkpoint_is_not_v5():
    v4_like = {"delta_token_state": {"net.0.weight": torch.zeros(2, 99)}}
    net, spec = load_delta_token_v5_from_checkpoint(v4_like)
    assert net is None and spec is None


def test_load_rejects_wrong_shapes(tmp_path=None):
    a = _actor()
    spec = _spec()
    sd = {k: v.detach().cpu() for k, v in a.state_dict().items()}
    bad = dict(sd)
    bad["token_proj.weight"] = torch.zeros(128, 63)     # 宽度错
    with pytest.raises(DeltaTokenContractError):
        load_temporal_actor_v5(bad, spec.as_dict(), state_joint_order="ha_v31",
                               hist_joint_order="ha_v31")


def test_save_requires_matching_param_count(tmp_path):
    a = _actor()
    spec = _spec()
    path = tmp_path / "ck.pt"
    payload = save_temporal_actor_v5(path, a, spec)
    assert payload["actor_params"] == 1_744_832
    assert payload["ckpt_sha256"]
    bad = spec_v5_from_actor(a, K=20, H=20, depth=3, meta=dict(default_joint_pos=[0.0] * 29))
    with pytest.raises(DeltaTokenContractError):
        save_temporal_actor_v5(tmp_path / "ck2.pt", a, bad)


def test_actor_mask_shape_and_T_overflow_raises():
    a = _actor()
    Z = torch.randn(2, 40, 64)
    s = torch.randn(2, 64)
    h = torch.randn(2, 10, 58)
    with pytest.raises(ValueError):
        a(Z, s, h, torch.ones(2, 39))
    with pytest.raises(ValueError):
        a(torch.randn(2, 41, 64), s, h, torch.ones(2, 41))
