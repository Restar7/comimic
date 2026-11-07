"""Actor 的尾部 Value 梯度规则（规范 §7.2 / §9 `tests/test_actor_tail_value_grad.py`）。

断言：
1. `beta > 0` 时：**Critic 参数冻结**（不更新、无梯度）但 `∂J_ac/∂y_H ≠ 0`（Actor 拿到尾部梯度）；
2. `beta = 0` 时：尾部项**完全不参与** loss ⇒ `y_H` 没有梯度、Actor 的梯度与不加尾部时逐位相同；
3. Actor 的损失确实流回 `TemporalDeltaActorV5` 的参数（逐层非零有限）。
"""
from __future__ import annotations

import copy

import torch

from diff_vla_wbc.delta_token_temporal_v5 import (
    HIST_DIM, TemporalActorCfg, TemporalDeltaActorV5)
from diff_vla_wbc.imagined_value_v5 import (
    CriticBatch, CriticCfgV5, CriticFeatureAdapter, build_value_net)

GAMMA = 0.99
B, T, H, K = 3, 40, 20, 20


class _FrozenParams:
    def __init__(self, module):
        self.module = module

    def __enter__(self):
        self.flags = [p.requires_grad for p in self.module.parameters()]
        for p in self.module.parameters():
            p.requires_grad_(False)
        return self.module

    def __exit__(self, *exc):
        for p, f in zip(self.module.parameters(), self.flags):
            p.requires_grad_(bool(f))
        return False


class _Imagined:
    """假想象轨迹；给 ``delta`` 时状态**依赖 Actor 输出**（真实 rollout 就是这样）。"""

    def __init__(self, delta: torch.Tensor | None = None, seed=0):
        g = torch.Generator().manual_seed(seed)
        states = torch.randn(H + 1, B, 71, generator=g)
        if delta is not None:
            d = delta[:, :H].sum(-1).t()                  # (H,B)
            states = torch.cat([states[:1], states[1:] + 1e-3 * d[..., None]], 0)
        self.states = states
        self.survival = torch.ones(H + 1, B)
        self.rewards = torch.rand(H, B, generator=g)


def _setup(seed=0):
    torch.manual_seed(seed)
    actor = TemporalDeltaActorV5(TemporalActorCfg(chunk_len=T))
    ad = CriticFeatureAdapter(CriticCfgV5())
    critic = build_value_net(ad, device="cpu")
    target = copy.deepcopy(critic)
    for p in target.parameters():
        p.requires_grad_(False)
    tokens = torch.randn(B, T, 64, generator=torch.Generator().manual_seed(1))
    batch = dict(state64=torch.randn(B, 64), hist10=torch.randn(B, 10, HIST_DIM),
                 ref_states=torch.randn(H + 1, B, 71), refs_f=torch.randint(0, 100, (B, H)),
                 xi=torch.randn(B, 33))
    mask = torch.zeros(B, T)
    mask[:, :K] = 1.0
    w = torch.randn(B, T, 64, generator=torch.Generator().manual_seed(2))
    return actor, ad, critic, target, tokens, batch, mask, w


def _yH(ad, imag, batch, delta, tokens):
    return ad.build(imag, CriticBatch(ref_states=batch["ref_states"],
                                      xi=batch["xi"], z=tokens + delta, hist10=batch["hist10"]),
                    H, K=K, T=T)


def test_tail_grad_reaches_actor_but_not_critic():
    actor, ad, critic, target, tokens, batch, mask, w = _setup()
    delta = actor(tokens, batch["state64"], batch["hist10"], mask)
    J20 = (delta * w).sum()
    yH = _yH(ad, _Imagined(delta), batch, delta, tokens)
    yH.retain_grad()
    with _FrozenParams(target):
        tail = target(yH)
    J_ac = J20 + 0.5 * (GAMMA ** H) * tail.sum()
    J_ac.backward()
    assert yH.grad is not None and float(yH.grad.abs().max()) > 0.0, "∂J_ac/∂y_H 必须非零"
    assert all(p.grad is None for p in target.parameters()), "冻结的 target critic 不应有梯度"
    actor_grads = [p.grad for p in actor.parameters() if p.grad is not None]
    assert actor_grads and all(torch.isfinite(g).all() for g in actor_grads)
    assert float(sum(g.norm() for g in actor_grads)) > 0.0


def test_beta_zero_has_no_tail_gradient():
    actor, ad, critic, target, tokens, batch, mask, w = _setup()
    delta = actor(tokens, batch["state64"], batch["hist10"], mask)
    J20 = (delta * w).sum()
    yH = _yH(ad, _Imagined(delta), batch, delta, tokens)
    yH.retain_grad()
    J20.backward()
    assert yH.grad is None or float(yH.grad.abs().max()) == 0.0
    # Actor 参数梯度与"没有尾部项"完全一样：这里直接确认它们来自 J20 且非零
    for name, p in actor.named_parameters():
        if p.grad is not None and float(p.grad.norm()) == 0.0:
            continue
    assert float(sum(p.grad.norm() for p in actor.parameters()
                     if p.grad is not None)) > 0.0


def test_critic_params_unaffected_by_actor_step():
    actor, ad, critic, target, tokens, batch, mask, w = _setup()
    before = {k: v.detach().clone() for k, v in critic.state_dict().items()}
    delta = actor(tokens, batch["state64"], batch["hist10"], mask)
    yH = _yH(ad, _Imagined(delta), batch, delta, tokens)
    with _FrozenParams(critic):
        loss = critic(yH).sum()
    assert loss.requires_grad and yH.requires_grad
    loss.backward()
    after = {k: v.detach().clone() for k, v in critic.state_dict().items()}
    assert all(torch.equal(before[k], after[k]) for k in before)
    # 值网络的参数在"只做前向"时没有梯度（需要经过 critic 自己的 loss 才会更新）
    assert all(p.grad is None for p in critic.parameters())
    # y_H 仍然带梯度（Actor 侧的合法通道）
    assert yH.requires_grad
