"""GR00T flow-matching tokens -> SuperMimic 10x68 motion goal.

The upstream policy is conditioned on ten future reference frames.  GR00T emits
a 40-step action chunk (64-D motion tokens in this setup), so the adapter pools
four consecutive tokens per goal frame and refines the ten pooled features with
a small temporal transformer.  The last layer is zero-initialised and a learned
bias supplies a physically sensible prior (upright gravity, nominal height).

Nothing here assumes the upstream weights are trained: the adapter is the only
new module that must be trained to align GR00T's latent action chunk with
SuperMimic's physical goal representation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

GOAL_FEATURE_DIM = 68
GOAL_FRAMES = 10
GOAL_SLICES = {
    "joint_pos": slice(0, 29),
    "joint_vel": slice(29, 58),
    "gravity": slice(58, 61),
    "root_ang_vel": slice(61, 64),
    "root_lin_vel": slice(64, 67),
    "root_height": slice(67, 68),
}


def default_goal_prior(goal_frames: int = GOAL_FRAMES, goal_frame_dim: int = GOAL_FEATURE_DIM) -> torch.Tensor:
    """A neutral goal: zero joint offset/velocity, upright gravity, 0.76 m height."""
    goal = torch.zeros(goal_frames, goal_frame_dim)
    if goal_frame_dim == GOAL_FEATURE_DIM:
        goal[:, 58:61] = torch.tensor([0.0, 0.0, -1.0])
        goal[:, 67] = 0.76
    elif goal_frame_dim == 73:
        # Motphys/Uni-SuperMimic layout (kept as an optional compatibility mode):
        # root delta 3, 6D rotation 6, lin/ang vel 3+3, q 29, 0.1*dq 29.
        goal[:, 3] = 1.0  # identity-ish 6D rotation first column
        goal[:, 7] = 1.0
    return goal


@dataclass
class AdapterConfig:
    token_dim: int = 64
    token_horizon: int = 40
    goal_frames: int = GOAL_FRAMES
    goal_frame_dim: int = GOAL_FEATURE_DIM
    hidden: int = 256
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.0


class MotionGoalAdapter(nn.Module):
    """Map ``[B, T, token_dim]`` FM tokens to ``[B, 10, 68]`` physical goals."""

    def __init__(self, cfg: AdapterConfig | None = None):
        super().__init__()
        self.cfg = cfg or AdapterConfig()
        cfg = self.cfg
        if cfg.goal_frames <= 0 or cfg.goal_frame_dim <= 0:
            raise ValueError("goal_frames and goal_frame_dim must be positive")
        self.pool = nn.AdaptiveAvgPool1d(cfg.goal_frames)
        self.inp = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.hidden),
            nn.SiLU(),
        )
        self.frame_pos = nn.Parameter(torch.zeros(1, cfg.goal_frames, cfg.hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.hidden * 2,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.hidden),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.goal_frame_dim),
        )
        # Do *not* zero-init: a zero output projection would block the gradient
        # from the WBC/world-model loss back to the FM tokens at step 0.
        nn.init.normal_(self.out[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out[-1].bias)
        self.register_buffer(
            "goal_prior",
            default_goal_prior(cfg.goal_frames, cfg.goal_frame_dim),
            persistent=True,
        )
        self.register_buffer("goal_scale", self._goal_scale(cfg.goal_frame_dim), persistent=True)

    @staticmethod
    def _goal_scale(dim: int) -> torch.Tensor:
        if dim == GOAL_FEATURE_DIM:
            return torch.tensor(
                [1.0] * 29 + [10.0] * 29 + [1.0] * 3 + [0.5] * 3 + [0.5] * 3 + [1.0]
            )
        if dim == 73:
            return torch.tensor(
                [1.0] * 3 + [1.0] * 6 + [1.0] * 3 + [0.5] * 3 + [1.0] * 29 + [10.0] * 29
            )
        return torch.ones(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Returns ``(B, goal_frames, goal_frame_dim)`` raw goals (physical units)."""
        if tokens.ndim != 3:
            raise ValueError(f"expected FM tokens [B,T,C], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.cfg.token_dim:
            raise ValueError(
                f"token dim {tokens.shape[-1]} != adapter token_dim {self.cfg.token_dim}"
            )
        # (B,T,C) -> (B,C,T) -> pool to K frames -> (B,K,C)
        pooled = self.pool(tokens.float().transpose(1, 2)).transpose(1, 2)
        h = self.inp(pooled) + self.frame_pos
        h = self.temporal(h)
        delta = self.out(h)
        return self.goal_prior + delta * self.goal_scale

    def forward_flat(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward(tokens).reshape(tokens.shape[0], -1)

    @property
    def goal_dim(self) -> int:
        return self.cfg.goal_frames * self.cfg.goal_frame_dim


class TeacherGoalAdapter(nn.Module):
    """Identity-style adapter used to debug the rollout with teacher goals."""

    def __init__(self, goal_frames: int = GOAL_FRAMES, goal_frame_dim: int = GOAL_FEATURE_DIM):
        super().__init__()
        self.cfg = AdapterConfig(goal_frames=goal_frames, goal_frame_dim=goal_frame_dim)

    def forward(self, teacher_goal: torch.Tensor) -> torch.Tensor:
        return teacher_goal

    def forward_flat(self, teacher_goal: torch.Tensor) -> torch.Tensor:
        return teacher_goal.reshape(teacher_goal.shape[0], -1)


def goal_statistics(goal: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-term means for logging a 68-D goal batch ``[B,10,68]`` or ``[B,680]``."""
    if goal.ndim == 2:
        goal = goal.reshape(goal.shape[0], GOAL_FRAMES, -1)
    return {name: goal[..., sl].mean() for name, sl in GOAL_SLICES.items() if sl.stop <= goal.shape[-1]}


__all__ = [
    "AdapterConfig",
    "GOAL_FEATURE_DIM",
    "GOAL_FRAMES",
    "GOAL_SLICES",
    "MotionGoalAdapter",
    "TeacherGoalAdapter",
    "default_goal_prior",
    "goal_statistics",
]
