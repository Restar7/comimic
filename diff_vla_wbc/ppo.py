"""PPO baseline for the fixed-time-budget ablation (BPTT vs PPO).

Why this file looks the way it does
-----------------------------------
The joint trainer (``diff_vla_wbc.train.JointTrainer``) optimises

    L = w_r * (-R(a)) + w_s * state_term + w_m * smooth + ... ,
    a = FM_theta(obs, z)                       # z ~ N(0, I), fixed per batch

i.e. it differentiates **through** the flow-matching ODE, the WBC decoder and
the frozen world model.  That is BPTT with an exact gradient.

The baseline here replaces exactly one thing: the gradient of the reward term.
The parameterised object stays the same — the VLA action head — but it is turned
into an explicit stochastic policy by putting a diagonal Gaussian on top of the
FM output:

    a = FM_theta(obs, z) + sigma * eps,  eps ~ N(0, I)
    log pi(a | obs) = -0.5 * ||(a - mu_theta) / sigma||^2 - sum(log sigma) - const

``z`` is drawn once per batch and **stored**, so the importance ratio
``pi_theta / pi_theta_old`` re-evaluates the same mean and is a genuine PPO ratio
rather than extra sampling noise.  The update reuses each collected batch for
``epochs`` passes with the usual clipped objective and a learned scalar baseline
``V(obs)`` on the start state.

One-step horizon
----------------
The VLA emits one 40x64 token chunk and the world model then rolls H steps with
that single conditioning (``imagine_with_tokens`` conditions step t on
``tokens[:, min(t, T-1)]``).  The action therefore does not change inside the
imagination horizon, so the MDP has a **horizon of one**: the discounted return
of the rollout is the return of a single action, and the clipped PPO objective
reduces to a clipped policy gradient with a baseline (PPO with T=1).  Reported
as such — do not call it a multi-step GAE implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F
from torch import nn

from .rollout import compute_joint_loss, discounted_return, imagine_with_tokens


@dataclass
class PpoConfig:
    gamma: float = 0.99
    clip_eps: float = 0.2
    epochs: int = 4
    sigma_init: float = 0.25
    sigma_min: float = 1e-3
    sigma_max: float = 5.0
    value_hidden: int = 256
    value_lr: float = 1e-3
    sigma_lr: float = 1e-3
    value_clip: float = 10.0
    #: Trust-region safeguards (2026-10-25).  The action is a 40x64 token chunk, so a
    #: per-dimension mean shift of even 0.1 rad turns into a log-prob difference of ~1.6
    #: per dim; summed over 2560 dims the PPO ratio leaves the representable range in one
    #: Adam step and the surrogate collapses.  Clamping the *log* ratio keeps the objective
    #: well posed without changing the policy parameterisation.
    log_ratio_clip: float = 5.0
    #: Gradient-norm clip for the flow-matching parameters (was reusing ``value_clip``=10).
    grad_clip: float = 1.0
    #: Skip an update if the loss or the ratio is not finite (keeps a bad batch from
    #: poisoning the optimiser state).
    skip_nonfinite: bool = True
    entropy_coef: float = 0.0
    aux_hand_weight: float = 0.0
    aux_grasp_boost: float = 5.0
    aux_hand_threshold: float = 0.05   # same as --hand-mask-threshold in train_joint.py
    reward_weights: dict[str, float] = field(default_factory=dict)


class ValueHead(nn.Module):
    """Scalar baseline on the start state (71-d SONIC state)."""

    def __init__(self, in_dim: int = 71, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class PpoTrainer:
    """Fixed-time PPO arm sharing every piece of the BPTT stack except the gradient."""

    def __init__(
        self,
        *,
        bundle,
        fm_sampler,
        fm_parameters: list[nn.Parameter],
        optimizer: torch.optim.Optimizer,
        cfg: PpoConfig | None = None,
        device: str | torch.device = "cuda",
        horizon: int = 8,
        token_dim: int = 64,
        noise_dim: int | None = None,
        hand_sampler=None,
        value_optimizer: torch.optim.Optimizer | None = None,
        sync_gradients: bool = False,
    ) -> None:
        self.bundle = bundle
        self.fm_sampler = fm_sampler
        self.fm_parameters = list(fm_parameters)
        self.optimizer = optimizer
        self.cfg = cfg or PpoConfig()
        self.device = torch.device(device)
        self.horizon = horizon
        self.token_dim = token_dim
        # The FM sampler's *initial noise* has the checkpoint's action width (132 for the GR00T
        # ckpt), not the 64-d motion-token width we optimise.  Both are needed: noise for the
        # Euler chain, token for the world model / log-prob.
        self.noise_dim = int(noise_dim or token_dim)
        self.hand_sampler = hand_sampler
        self.sync_gradients = sync_gradients
        self.log_sigma = nn.Parameter(torch.full((), math.log(self.cfg.sigma_init), device=self.device))
        self.value = ValueHead(hidden=self.cfg.value_hidden).to(self.device)
        self.value_opt = value_optimizer or torch.optim.Adam(self.value.parameters(), lr=self.cfg.value_lr)
        self.steps = 0
        # The exploration scale is learned too, so it needs its own param group;
        # otherwise it would never move (it is not part of the FM parameter list).
        self.optimizer.add_param_group({"params": [self.log_sigma], "lr": self.cfg.sigma_lr})

        # The world model is a frozen environment; the WBC policy is frozen too.
        for p in self.bundle.wm.parameters():
            p.requires_grad_(False)
        for p in self.bundle.policy.parameters():
            p.requires_grad_(False)
        self.bundle.wm.eval()
        self.bundle.policy.eval()

    # ------------------------------------------------------------------ helpers

    @property
    def sigma(self) -> torch.Tensor:
        s = self.log_sigma.exp().clamp(self.cfg.sigma_min, self.cfg.sigma_max)
        return s

    def _mean(self, model_inputs, noise: torch.Tensor) -> torch.Tensor:
        """Deterministic policy mean: FM sample with the *same* initial noise, sliced to the
        motion-token width the world model consumes (chunk columns 64:78 are the hand stream)."""
        mean = self.fm_sampler(model_inputs, initial_noise=noise)
        if mean.shape[-1] != self.token_dim:
            mean = mean[..., : self.token_dim]
        return mean

    def _logprob(self, action: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        var = (sigma**2).clamp_min(1e-12)
        sq = ((action - mean) ** 2).sum(dim=tuple(range(1, action.ndim)))
        return -0.5 * sq / var - 0.5 * action[0].numel() * torch.log(2 * math.pi * var)

    def _rollout(self, batch, action: torch.Tensor):
        rollout = imagine_with_tokens(
            self.bundle,
            tokens=action,
            state=batch.start_state,
            obs_hist=batch.obs_hist,
            history=batch.history,
            xi=batch.xi,
            references=batch.references,
            prev_action=batch.prev_action,
            horizon=self.horizon,
            obs_noise=0.0,
        )
        return rollout

    # ------------------------------------------------------------------ training

    def train_step(self, batch) -> dict[str, float]:
        inputs = batch.model_inputs if getattr(batch, "model_inputs", None) is not None else batch.as_dict()
        noise = torch.randn(
            batch.start_state.shape[0],
            getattr(batch, "token_horizon", 40),
            self.noise_dim,
            device=self.device,
            dtype=next(iter(self.fm_parameters)).dtype,
        )

        # ---- 1. act (no grad through the policy), then evaluate the frozen env
        with torch.no_grad():
            mean_old = self._mean(inputs, noise)
            sigma = self.sigma.detach()
            action = mean_old + sigma * torch.randn_like(mean_old)
            logp_old = self._logprob(action, mean_old, sigma)
            rollout = self._rollout(batch, action)
            returns = discounted_return(rollout, self.cfg.gamma)
            value_old = self.value(batch.start_state)
            adv = returns - value_old
            adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
            reward_mean = float(rollout.rewards.mean())
            survival = float(rollout.survival[-1].mean())

        # ---- 2. clipped policy-gradient epochs (same batch, same z, same action)
        stats = {}
        for epoch in range(self.cfg.epochs):
            mean = self._mean(inputs, noise)
            logp = self._logprob(action, mean, self.sigma)
            log_ratio_raw = logp - logp_old
            if self.cfg.log_ratio_clip > 0:
                log_ratio = log_ratio_raw.clamp(-self.cfg.log_ratio_clip, self.cfg.log_ratio_clip)
            else:
                log_ratio = log_ratio_raw
            ratio = torch.exp(log_ratio)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            entropy = -logp.mean()
            loss = policy_loss - self.cfg.entropy_coef * entropy

            aux = None
            if (
                self.cfg.aux_hand_weight > 0
                and self.hand_sampler is not None
                and getattr(batch, "hand_ref", None) is not None
            ):
                full = self.hand_sampler(batch.as_dict())
                hands = full[..., 64:78]
                ref = batch.hand_ref.to(hands.dtype)
                n = min(hands.shape[1], ref.shape[1])
                grasp = (ref[:, :n].abs().mean(-1) > 0.10).to(hands.dtype)
                # Same mask as JointTrainer.train_step: frames whose recorded hand label is all
                # zero carry no supervision (60% of the HA frames), otherwise the aux term just
                # teaches "keep the hand still" and the two arms would not be comparable.
                active = (ref[:, :n].abs().amax(-1) > self.cfg.aux_hand_threshold).to(hands.dtype)
                w = (1.0 + self.cfg.aux_grasp_boost * grasp) * active
                per = ((hands[:, :n] - ref[:, :n]) ** 2).mean(-1)
                aux = (per * w).sum() / w.sum().clamp(min=1.0)
                loss = loss + self.cfg.aux_hand_weight * aux

            self.optimizer.zero_grad(set_to_none=True)
            self.value_opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.sync_gradients:
                self._sync()
            gn = torch.nn.utils.clip_grad_norm_(self.fm_parameters, self.cfg.grad_clip)
            if self.cfg.skip_nonfinite and not bool(torch.isfinite(gn)):
                self.optimizer.zero_grad(set_to_none=True)
                self.steps += 1
                return {"loss": float("nan"), "ppo/sigma": float(self.sigma.detach()), "skipped": 1.0}
            self.optimizer.step()

            # value regression towards the (frozen) return of this batch
            pred = self.value(batch.start_state)
            vloss = F.mse_loss(pred, returns.detach())
            vloss.backward()
            self.value_opt.step()

            if epoch == 0:
                # 诊断用：第 0 个 epoch 的 ratio 才是"更新前"的真实值；后几个 epoch 会被
                # 同 batch 多步 Adam 推到 clip 外（2560 维 token 动作 + 标量 sigma），
                # 打印它才能区分"策略没动"和"信任域饱和"。
                first_ratio = float(ratio.mean().detach())
                first_clip = float(((ratio - 1).abs() > self.cfg.clip_eps).float().mean())
            stats = {
                "loss/policy": float(policy_loss.detach()),
                "ppo/ratio_epoch0": first_ratio,
                "ppo/clip_frac_epoch0": first_clip,
                "loss/value": float(vloss.detach()),
                "ppo/ratio_mean": float(ratio.mean().detach()),
                "ppo/log_ratio_absmax": float(log_ratio_raw.abs().max().detach()),
                "ppo/clip_frac": float(((ratio - 1).abs() > self.cfg.clip_eps).float().mean()),
                "ppo/sigma": float(self.sigma.detach()),
                "ppo/logp": float(logp.mean().detach()),
                "ppo/entropy": float(entropy.detach()),
            }
            if aux is not None:
                stats["loss/hand_mse"] = float(aux.detach())
            if epoch == self.cfg.epochs - 1:
                break

        self.steps += 1
        out = {"loss": stats.get("loss/policy", 0.0), "ppo/sigma": float(self.sigma.detach())}
        out.update(stats)
        out["rollout/reward"] = reward_mean
        out["rollout/survival"] = survival
        out["rollout/action_abs"] = float(action.abs().mean())
        out["advantage/mean"] = float(adv.mean())
        return out

    def _sync(self) -> None:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return
        grads = [p.grad for p in self.fm_parameters if p.grad is not None]
        if not grads:
            return
        flat = torch.cat([g.reshape(-1) for g in grads])
        dist.all_reduce(flat, op=dist.ReduceOp.AVG)
        off = 0
        for p in self.fm_parameters:
            if p.grad is None:
                continue
            n = p.grad.numel()
            p.grad.copy_(flat[off : off + n].view_as(p.grad))
            off += n

    # ------------------------------------------------------------------ eval

    @torch.no_grad()
    def evaluate(self, batches) -> dict[str, float]:
        out = {"token_mse": [], "state_term": [], "rollout_reward": []}
        for batch in batches:
            inputs = batch.model_inputs if getattr(batch, "model_inputs", None) is not None else batch.as_dict()
            noise = torch.randn(
                batch.start_state.shape[0],
                getattr(batch, "token_horizon", 40),
                self.noise_dim,
                device=self.device,
                dtype=next(iter(self.fm_parameters)).dtype,
            )
            mean = self._mean(inputs, noise)
            rollout = self._rollout(batch, mean)
            _, terms = compute_joint_loss(
                rollout, teacher_states=batch.teacher_states, gamma=self.cfg.gamma
            )
            if getattr(batch, "recorded_tokens", None) is not None:
                rec = batch.recorded_tokens
                n = min(mean.shape[1], rec.shape[1])
                out["token_mse"].append(float(((mean[:, :n] - rec[:, :n]) ** 2).mean()))
            out["state_term"].append(float(terms.get("state", torch.zeros(()))))
            out["rollout_reward"].append(float(rollout.rewards.mean()))
        return {
            "token_mse": sum(out["token_mse"]) / max(1, len(out["token_mse"])),
            "state_term": sum(out["state_term"]) / max(1, len(out["state_term"])),
            "rollout_reward": sum(out["rollout_reward"]) / max(1, len(out["rollout_reward"])),
        }


__all__ = ["PpoConfig", "PpoTrainer", "ValueHead"]
