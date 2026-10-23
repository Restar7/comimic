"""Joint VLA / WBC / world-model training loop.

The world model is frozen for policy improvement, but its forward pass is
differentiable w.r.t. the action (and hence w.r.t. the WBC, adapter and FM
head).  A separate optimizer updates the world model from real transition
windows, which are *not* differentiated into the policy.

For the real experiment use:

* ``fm_sampler=Gr00tTokenSource(gr00t)`` with real preprocessed batches;
* ``bundle = SuperMimicBundle.from_checkpoint("runs/.../ckpt_012000.pt", assets)``;
* a ``MotionBatchProvider`` whose token source returns the paired GR00T tokens
  for the same motion window.

For the smoke run in this repository the token source is random and the WBC/WM
are randomly initialized, which still exercises every differentiable edge.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn

from .adapter import MotionGoalAdapter
from .rollout import (
    LossWeights,
    compute_joint_loss,
    freeze_world_model,
    imagine_with_goals,
    unfreeze_world_model,
)
from .supermimic_backend import SuperMimicBundle


@dataclass
class JointTrainerConfig:
    horizon: int = 8
    gamma: float = 0.99
    adapter_lr: float = 1e-4
    policy_lr: float = 1e-5
    fm_lr: float = 1e-5
    wm_lr: float = 2e-4
    weight_decay: float = 1e-4
    adapter_clip: float = 1.0
    policy_clip: float = 1.0
    fm_clip: float = 0.5
    wm_clip: float = 1.0
    obs_noise: float = 0.0
    pretrain_wm_steps: int = 0
    pretrain_wm_batch: int = 8
    pretrain_wm_window: int = 8
    weights: LossWeights = field(default_factory=LossWeights)


class JointTrainer:
    def __init__(
        self,
        bundle: SuperMimicBundle,
        adapter: MotionGoalAdapter,
        *,
        fm_sampler: Callable | None = None,
        fm_parameters: Iterable[torch.nn.Parameter] | None = None,
        cfg: JointTrainerConfig | None = None,
        device: str | torch.device | None = None,
    ):
        self.bundle = bundle
        self.adapter = adapter
        self.fm_sampler = fm_sampler
        self.cfg = cfg or JointTrainerConfig()
        self.device = torch.device(device or bundle.device)
        self.bundle.to(self.device)
        self.adapter.to(self.device)
        if self.bundle.goal_dim != self.adapter.goal_dim:
            raise ValueError(
                f"adapter goal_dim {self.adapter.goal_dim} != policy goal_dim {self.bundle.goal_dim}"
            )

        params = list(self.adapter.parameters()) + list(self.bundle.policy.parameters())
        if fm_parameters is None and isinstance(fm_sampler, nn.Module):
            fm_parameters = fm_sampler.parameters()
        self.fm_parameters = [p for p in (fm_parameters or []) if p.requires_grad]
        if self.fm_parameters:
            params += self.fm_parameters
        self.optimizer = torch.optim.AdamW(
            [
                {"params": list(self.adapter.parameters()), "lr": self.cfg.adapter_lr},
                {"params": list(self.bundle.policy.parameters()), "lr": self.cfg.policy_lr},
                *(
                    [{"params": self.fm_parameters, "lr": self.cfg.fm_lr}]
                    if self.fm_parameters
                    else []
                ),
            ],
            weight_decay=self.cfg.weight_decay,
        )
        self.wm_optimizer = torch.optim.AdamW(
            self.bundle.wm.parameters(), lr=self.cfg.wm_lr, weight_decay=self.cfg.weight_decay
        )
        # Freeze by default: policy improvement must not update the dynamics.
        freeze_world_model(self.bundle.wm)
        self.steps = 0

    # ------------------------------------------------------------------ policy / FM

    def _tokens(self, batch) -> torch.Tensor:
        if self.fm_sampler is None:
            return batch.fm_tokens
        if getattr(batch, "model_inputs", None) is None:
            # The sampler may only need the metadata dict; let it decide.
            return self.fm_sampler(batch.as_dict())
        return self.fm_sampler(batch.model_inputs)

    def train_step(self, batch) -> dict[str, float]:
        self.bundle.policy.train()
        self.adapter.train()
        self.bundle.wm.eval()
        freeze_world_model(self.bundle.wm)

        tokens = self._tokens(batch)
        rollout = imagine_with_goals(
            self.bundle,
            fm_tokens=tokens,
            adapter=self.adapter,
            state=batch.start_state,
            obs_hist=batch.obs_hist,
            history=batch.history,
            xi=batch.xi,
            references=batch.references,
            prev_action=batch.prev_action,
            horizon=self.cfg.horizon,
            obs_noise=self.cfg.obs_noise,
        )
        loss, terms = compute_joint_loss(
            rollout,
            teacher_goals=batch.teacher_goals,
            teacher_goal_sequence=batch.teacher_goal_sequence,
            teacher_states=batch.teacher_states,
            gamma=self.cfg.gamma,
            weights=self.cfg.weights,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.fm_parameters:
            torch.nn.utils.clip_grad_norm_(self.fm_parameters, self.cfg.fm_clip)
        torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.cfg.adapter_clip)
        torch.nn.utils.clip_grad_norm_(self.bundle.policy.parameters(), self.cfg.policy_clip)
        self.optimizer.step()
        self.steps += 1

        metrics = {"loss": float(loss.detach())}
        metrics.update({f"loss/{k}": float(v.detach()) for k, v in terms.items()})
        metrics["rollout/reward"] = float(rollout.rewards.mean().detach())
        metrics["rollout/survival"] = float(rollout.survival[-1].mean().detach())
        metrics["rollout/action_abs"] = float(rollout.actions.abs().mean().detach())
        return metrics

    # ------------------------------------------------------------------ world model

    def train_world_model(self, windows: dict) -> dict[str, float]:
        """Fit ``F_phi`` on real transition windows (no policy gradient)."""
        self.bundle.wm.train()
        unfreeze_world_model(self.bundle.wm)
        loss, metrics = self.bundle.wm.window_loss(
            windows["states"],
            windows["actions"],
            windows["xi"],
            windows["done"],
            windows["refs"],
        )
        self.wm_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.bundle.wm.parameters(), self.cfg.wm_clip)
        self.wm_optimizer.step()
        freeze_world_model(self.bundle.wm)
        out = {f"wm/{k}": float(v.detach()) for k, v in metrics.items()}
        out["wm/loss"] = float(loss.detach())
        return out

    # ------------------------------------------------------------------ checkpoints

    @torch.no_grad()
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "adapter": self.adapter.state_dict(),
                "adapter_config": asdict(self.adapter.cfg),
                "policy": self.bundle.policy.state_dict(),
                "wm": self.bundle.wm.state_dict(),
                "policy_config": asdict(self.bundle.policy_cfg),
                "world_model_config": asdict(self.bundle.world_model_cfg),
                "config": asdict(self.cfg),
                "steps": self.steps,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.adapter.load_state_dict(ck["adapter"])
        self.bundle.policy.load_state_dict(ck["policy"])
        self.bundle.wm.load_state_dict(ck["wm"])
        self.steps = int(ck.get("steps", 0))

    @staticmethod
    def load_bundle_from_checkpoint(path: str | Path, assets):
        """Rebuild a bundle with the exact architecture saved by ``JointTrainer.save``."""
        from .supermimic_backend import SuperMimicBundle, _coerce_cfg
        from supermimic.policy.nets import PolicyCfg
        from supermimic.world_model.world_model import WorldModelCfg

        ck = torch.load(path, map_location=assets.device, weights_only=False)
        if "policy_config" not in ck or "world_model_config" not in ck:
            raise ValueError("checkpoint lacks policy/world-model config; pass --supermimic-ckpt instead")
        policy_cfg = _coerce_cfg(PolicyCfg, ck["policy_config"])
        wm_cfg = _coerce_cfg(WorldModelCfg, ck["world_model_config"])
        bundle = SuperMimicBundle.from_configs(assets, policy_cfg, wm_cfg)
        bundle.policy.load_state_dict(ck["policy"])
        bundle.wm.load_state_dict(ck["wm"])
        return bundle

    @staticmethod
    def load_adapter_from_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
        from .adapter import AdapterConfig, MotionGoalAdapter

        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = AdapterConfig(**ck["adapter_config"])
        adapter = MotionGoalAdapter(cfg).to(device)
        adapter.load_state_dict(ck["adapter"])
        adapter.eval()
        return adapter


def pretrain_world_model(
    trainer: JointTrainer,
    provider,
    *,
    steps: int,
    batch_size: int = 8,
    window: int = 8,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Warm-start the world model from reference windows (no simulator)."""
    from .data import sample_reference_windows

    logs = []
    for i in range(steps):
        windows = sample_reference_windows(
            trainer.bundle,
            provider.library,
            batch_size=batch_size,
            window=window,
            seed=seed + i,
        )
        logs.append(trainer.train_world_model(windows))
    return logs


__all__ = ["JointTrainer", "JointTrainerConfig", "pretrain_world_model"]
