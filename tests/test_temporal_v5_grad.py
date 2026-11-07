"""v5 Actor 梯度/形状/掩码门禁（规范 §9 `tests/test_temporal_v5_grad.py`）。

断言：
1. ``[B,T,64] + [B,64] + [B,10,58] → [B,T,64]``；
2. attention / GRU / MLP **每一层**都有**非零有限**梯度；
3. ``mask_t = 0`` 的位置 ``∂L/∂Δ_t`` **严格为 0**（不是"很小"）；
4. 参数量 = 1,744,832（T=40、use_history）；no-history 分支 = 1,541,568（d_in=272）；
5. ``t ≥ K`` 的输出**逐位为 0**；``cap`` 是软上限（‖Δ‖ ≤ cap + eps）。
"""
from __future__ import annotations

import torch

from diff_vla_wbc.delta_token_temporal_v5 import (
    HIST_DIM, TemporalActorCfg, TemporalDeltaActorV5)


def _batch(b: int = 3, T: int = 40):
    torch.manual_seed(0)
    return (torch.randn(b, T, 64), torch.randn(b, 64), torch.randn(b, 10, HIST_DIM))


def _mask(b: int, T: int, K: int = 20) -> torch.Tensor:
    m = torch.zeros(b, T)
    m[:, :K] = 1.0
    return m


def test_shape_and_dtype():
    actor = TemporalDeltaActorV5(TemporalActorCfg(chunk_len=40))
    Z, s, h = _batch()
    d = actor(Z, s, h, _mask(3, 40))
    assert tuple(d.shape) == (3, 40, 64)
    assert d.dtype == torch.float32 and torch.isfinite(d).all()


def test_param_count_matches_spec():
    actor = TemporalDeltaActorV5(TemporalActorCfg(chunk_len=40))
    assert actor.n_params == 1_744_832, actor.n_params
    assert actor.d_in == 400
    nh = TemporalDeltaActorV5(TemporalActorCfg(chunk_len=40, use_history=False))
    assert nh.n_params == 1_541_568, nh.n_params
    assert nh.d_in == 272


def test_every_layer_has_nonzero_finite_grad():
    actor = TemporalDeltaActorV5(TemporalActorCfg(chunk_len=40))
    Z, s, h = _batch()
    d = actor(Z, s, h, _mask(3, 40))
    target = torch.randn_like(d)
    loss = (d * target).sum()
    loss.backward()
    bad = []
    for name, p in actor.named_parameters():
        if p.grad is None:
            bad.append((name, "grad=None"))
            continue
        if not torch.isfinite(p.grad).all():
            bad.append((name, "non-finite"))
        elif float(p.grad.norm()) == 0.0:
            bad.append((name, "zero"))
    assert not bad, f"这些层没有非零有限梯度：{bad}"


def test_masked_positions_have_zero_gradient():
    actor = TemporalDeltaActorV5(TemporalActorCfg(chunk_len=40))
    Z, s, h = _batch(b=2)
    d, raw = actor(Z, s, h, _mask(2, 40, K=20), return_raw=True)
    raw.retain_grad()
    g = torch.randn_like(d)
    (d * g).sum().backward()
    assert raw.grad is not None
    # mask 乘在 cap 之后、逐位置独立 ⇒ ∂L/∂u_t = g_t · mask_t ⇒ t≥K 必须逐位 0
    assert float(raw.grad[:, 20:].abs().max()) == 0.0
    assert float(raw.grad[:, :20].abs().max()) > 0.0


def test_executed_tail_is_exactly_zero_and_capped():
    actor = TemporalDeltaActorV5(TemporalActorCfg(chunk_len=40, cap=0.08))
    Z, s, h = _batch(b=8)
    d = actor(Z, s, h, _mask(8, 40, K=20), cap=0.08)
    assert float(d[:, 20:].abs().max()) == 0.0
    n = d.norm(dim=-1)
    assert float(n.max()) <= 0.08 + 1e-6, float(n.max())
    # 位置 0..19 不应全为 0（否则"逐帧"退化成空操作）
    assert float(d[:, :20].abs().max()) > 0.0


def test_missing_history_raises():
    actor = TemporalDeltaActorV5(TemporalActorCfg(chunk_len=40, use_history=True))
    Z, s, _ = _batch(b=2)
    try:
        actor(Z, s, None, _mask(2, 40))
    except ValueError as exc:
        assert "hist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("use_history=True 但缺 hist 必须抛错（禁止零历史冒充）")


def test_wrong_history_width_raises():
    actor = TemporalDeltaActorV5(TemporalActorCfg(chunk_len=40, use_history=True))
    Z, s, _ = _batch(b=2)
    try:
        actor(Z, s, torch.randn(2, 10, 64), _mask(2, 40))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("hist 宽度不是 58 时必须抛错")
