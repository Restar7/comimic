"""CPU tests for the PPO arm's optimisation mechanics.

These deliberately do not touch SuperMimic / GR00T: ``_mean`` and ``_rollout`` are
the only two seams between the PPO trainer and the shared stack, so overriding
them is enough to pin down the parts that can actually be wrong here — the
Gaussian log-prob, the importance ratio, clipping, the learned sigma and the
value baseline.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diff_vla_wbc.ppo import PpoConfig, PpoTrainer


class _MeanModule(nn.Module):
    """Stands in for the FM action head: a linear map from noise to a token chunk."""

    def __init__(self, out_features: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(4, out_features)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.linear(noise)


def _make(seed: int = 0, out_features: int = 4, **cfg_kwargs):
    torch.manual_seed(seed)
    fm = _MeanModule(out_features)
    policy = nn.Linear(1, 1)
    wm = nn.Linear(1, 1)
    bundle = SimpleNamespace(policy=policy, wm=wm)
    mean_calls = {"n": 0}

    def fm_sampler(inputs, *, initial_noise=None):
        mean_calls["n"] += 1
        return fm(initial_noise)

    cfg = PpoConfig(epochs=3, **cfg_kwargs)
    trainer = PpoTrainer(
        bundle=bundle,
        fm_sampler=fm_sampler,
        fm_parameters=list(fm.parameters()),
        optimizer=torch.optim.SGD(fm.parameters(), lr=0.1),
        cfg=cfg,
        device="cpu",
        horizon=2,
        token_dim=4,
    )

    def fake_rollout(batch, action):
        # reward strongly prefers action -> 1, so both the policy and the value
        # head have a clear direction to move in.
        reward = -(action.mean(dim=(1, 2)) - 1.0) ** 2
        rewards = reward.unsqueeze(0).repeat(2, 1)
        return SimpleNamespace(
            rewards=rewards,
            survival=torch.ones(3, action.shape[0]),
        )

    trainer._rollout = fake_rollout  # type: ignore[method-assign]
    return trainer, fm, mean_calls


def _batch(batch_size: int = 3, state_dim: int = 71):
    return SimpleNamespace(
        start_state=torch.zeros(batch_size, state_dim),
        model_inputs={"dummy": torch.zeros(batch_size)},
        as_dict=lambda: {"dummy": torch.zeros(batch_size)},
        hand_ref=None,
        token_horizon=4,
    )


def test_logprob_matches_normal_distribution() -> None:
    trainer, _, _ = _make()
    action = torch.randn(2, 3, 4)
    mean = torch.randn(2, 3, 4)
    sigma = torch.tensor(0.3)
    got = trainer._logprob(action, mean, sigma)

    dist = torch.distributions.Normal(mean, sigma)
    want = dist.log_prob(action).sum(dim=(1, 2))
    assert torch.allclose(got, want, atol=1e-4), (got, want)


def test_ratio_is_one_on_first_epoch() -> None:
    """Same noise + same action + unchanged weights => ratio exactly 1, no clipping."""
    trainer, _, calls = _make()
    batch = _batch()
    # Run only the first epoch by faking a 1-epoch config.
    trainer.cfg.epochs = 1
    metrics = trainer.train_step(batch)
    assert metrics["ppo/ratio_mean"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["ppo/clip_frac"] == 0.0
    assert calls["n"] == 2  # one mean for acting, one for the update


def test_sigma_and_policy_parameters_move() -> None:
    trainer, fm, _ = _make()
    before = trainer.log_sigma.detach().clone()
    w_before = fm.linear.weight.detach().clone()
    for _ in range(5):
        trainer.train_step(_batch())
    assert not torch.allclose(trainer.log_sigma.detach(), before), "sigma was not learned"
    assert not torch.allclose(fm.linear.weight.detach(), w_before), "policy weights did not move"


def test_value_head_fits_the_return() -> None:
    """With an action-independent return the baseline must regress to it."""
    trainer, _, _ = _make(value_lr=1e-2)

    def constant_rollout(batch, action):
        return SimpleNamespace(
            rewards=torch.full((2, action.shape[0]), 2.0),
            survival=torch.ones(3, action.shape[0]),
        )

    trainer._rollout = constant_rollout  # type: ignore[method-assign]
    losses = []
    for _ in range(15):
        m = trainer.train_step(_batch())
        losses.append(m["loss/value"])
    assert losses[-1] < losses[0], f"value loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_metrics_are_finite() -> None:
    trainer, _, _ = _make()
    m = trainer.train_step(_batch())
    for key in ("loss/policy", "loss/value", "ppo/sigma", "ppo/entropy", "rollout/reward"):
        assert key in m, key
        assert torch.isfinite(torch.tensor(m[key])), (key, m[key])
    assert 0.0 < m["ppo/sigma"] < 5.0


def test_zero_reward_gives_no_policy_drift_direction() -> None:
    """With a constant return the centred advantage is ~0, so PPO must not push."""
    trainer, fm, _ = _make()

    def flat_rollout(batch, action):
        return SimpleNamespace(
            rewards=torch.ones(2, action.shape[0]),
            survival=torch.ones(3, action.shape[0]),
        )

    trainer._rollout = flat_rollout  # type: ignore[method-assign]
    w_before = fm.linear.weight.detach().clone()
    for _ in range(3):
        trainer.train_step(_batch())
    delta = (fm.linear.weight.detach() - w_before).abs().max()
    assert delta < 1e-6, f"policy drifted on a flat advantage: {delta:.2e}"
