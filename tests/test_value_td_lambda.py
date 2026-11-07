"""TD(λ) + EMA target + Value Readiness Gate（规范 §6.3 / §9 `tests/test_value_td_lambda.py`）。

断言：
1. ``lam=0`` ⇒ ``G_t = r_t + γ m_t V̄(y_{t+1})``；``lam=1`` ⇒ MC 折扣回报；
2. 真终止（``done=1``）之后不再 bootstrap；段尾 timeout 用 ``V̄(y_L)``，真终止用 0；
3. EMA 语义：``V̄ ← keep·V̄ + (1−keep)·V_ψ``（含 buffer），且 **target 参数不需要梯度**；
4. 目标全部 detach（对值网络无梯度）；
5. ``CriticFeatureAdapter.in_dim`` 运行时给出 = 1086（规范 §6.2 的 71+71+71+33+580+256+4）；
   超过 ``K−1`` 的 plan token **不得填入**（严格 0）。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from diff_vla_wbc.imagined_value_v5 import (
    CriticBatch, CriticCfgV5, CriticFeatureAdapter, ImaginedValueV5, ValueReadinessGate,
    bootstrap_identifiable, build_value_net, ema_update, mc_returns, td_lambda_targets)


def _rand_seq(H=5, B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(H, B, generator=g), torch.zeros(H, B),
            torch.rand(H + 1, B, generator=g))


def test_td_lambda_endpoints():
    H, B, gamma = 6, 2, 0.9
    r = torch.ones(H, B)
    m = torch.zeros(H, B)
    v = torch.rand(H + 1, B)
    g0 = td_lambda_targets(r, m, v, gamma=gamma, lam=0.0)
    # lam=0: G_t = r_t + γ V_{t+1}
    for t in range(H):
        assert torch.allclose(g0[t], r[t] + gamma * v[t + 1])
    g1 = td_lambda_targets(r, m, v, gamma=gamma, lam=1.0)
    mc = mc_returns(r, m, gamma=gamma, end_bootstrap=v[H])
    assert torch.allclose(g1, mc)


def test_td_lambda_terminal_masks_bootstrap():
    r = torch.ones(3, 2)
    done = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    v = torch.full((4, 2), 0.5)
    g = td_lambda_targets(r, done, v, gamma=0.9, lam=0.5)
    # 第 1 步 done ⇒ G_1 = r_1 = 1（不再 bootstrap）
    assert torch.allclose(g[1], torch.ones(2))
    # G_0 = 1 + 0.9*0*... 与 G_2 无关的 done/lam 组合仍需 non-negative
    assert float(g.min()) >= 0.0


def test_end_bootstrap_true_terminal_vs_timeout():
    r = torch.ones(2, 1)
    done = torch.zeros(2, 1)
    v = torch.full((3, 1), 1.0)
    g_to = td_lambda_targets(r, done, v, gamma=1.0, lam=1.0,
                             end_bootstrap=torch.zeros(1))        # 真终止
    g_mc = mc_returns(r, done, gamma=1.0, end_bootstrap=torch.zeros(1))
    assert torch.allclose(g_to, g_mc)
    assert float(g_to[0]) == 2.0
    # timeout：末步按 V̄(y_L) 自举
    g_v = td_lambda_targets(r, done, v, gamma=1.0, lam=1.0,
                            end_bootstrap=torch.full((1,), 5.0))
    assert float(g_v[0]) == 7.0


def test_targets_are_detached():
    r = torch.ones(4, 2, requires_grad=True)
    v = torch.ones(5, 2, requires_grad=True)
    g = td_lambda_targets(r, torch.zeros(4, 2), v, gamma=0.9, lam=0.95)
    assert not g.requires_grad
    with pytest.raises(RuntimeError):
        g.sum().backward()


def test_ema_semantics():
    online = torch.nn.Linear(3, 1)
    target = torch.nn.Linear(3, 1)
    with torch.no_grad():
        for p in target.parameters():
            p.fill_(1.0)
        for p in online.parameters():
            p.fill_(3.0)
    for p in target.parameters():
        p.requires_grad_(False)
    ema_update(target, online, 0.995)
    for p in target.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.995 * 1.0 + 0.005 * 3.0))
        assert not p.requires_grad


def test_adapter_in_dim_and_plan_mask():
    ad = CriticFeatureAdapter()
    assert ad.in_dim == 1086, ad.in_dim           # 71+71+71+33+580+256+4
    B, T, H, K = 2, 40, 20, 20
    class R:
        pass
    r = R()
    r.states = torch.zeros(H + 1, B, 71)
    z = torch.ones(B, T, 64)
    batch = CriticBatch(ref_states=torch.zeros(H + 1, B, 71), xi=torch.zeros(B, 33),
                        z=z, hist10=torch.zeros(B, 10, 58))
    y = ad.build(r, batch, t=H, K=K, T=T)         # t=K ⇒ 所有 plan token 必须为 0
    assert tuple(y.shape) == (B, ad.in_dim)
    plan = y[:, -4 - 4 * 64:-4].reshape(B, 4, 64)
    assert float(plan.abs().max()) == 0.0
    y2 = ad.build(r, batch, t=0, K=K, T=T)        # t=0 ⇒ 前 4 个都合法
    plan2 = y2[:, -4 - 4 * 64:-4].reshape(B, 4, 64)
    assert float(plan2.abs().min()) == 1.0
    # hist10 缺失（hist_frames>0）⇒ 抛
    with pytest.raises(ValueError):
        ad.build(r, CriticBatch(ref_states=torch.zeros(H + 1, B, 71), xi=torch.zeros(B, 33),
                                z=z, hist10=None), t=0, K=K, T=T)


def test_value_net_and_gate():
    ad = CriticFeatureAdapter()
    net = build_value_net(ad)
    assert isinstance(net, ImaginedValueV5)
    assert net.in_dim == ad.in_dim
    assert net(torch.zeros(4, ad.in_dim)).shape == (4,)
    assert not bootstrap_identifiable(20, 20)
    assert bootstrap_identifiable(20, 12)
    gate = ValueReadinessGate(K=20, H=20)
    res = gate.update(v_ev_heldout=0.9, mae=0.1, mae_const_baseline=0.5, dir_consistency=0.9)
    assert res.status.startswith("BLOCKED") and gate.beta_at(100) == 0.0
    gate2 = ValueReadinessGate(K=20, H=12)
    assert gate2.update(v_ev_heldout=0.9, mae=0.1, mae_const_baseline=0.5,
                        dir_consistency=0.9).status == "PASS"
    assert gate2.beta_at(gate2.pass_update + 50) == pytest.approx(0.5, abs=1e-6)
    gate3 = ValueReadinessGate(K=20, H=12)
    assert gate3.update(v_ev_heldout=0.3, mae=0.1, mae_const_baseline=0.5,
                        dir_consistency=0.9).status == "FAIL"
    assert gate3.beta_at(1000) == 0.0
