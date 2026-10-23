"""14-D hand head on the VLA backbone — the "direct hand output" channel.

Architecture this implements (what the GR00T-N1.6-style path does):

```text
VLM backbone  ──►  action head (flow matching)  ──►  tokens [B, 40, 64] ──► adapter ──► SuperMimic (WBC/WM) ──► 29 body joints
              └─►  hand head (this module)       ──►  hand   [B, H, 14]   ──────────────────────────────────────► 14 hand joints
```

The 14 hand joint angles are **not** part of the 64-D motion token and never go
through the adapter, SuperMimic's world model or its reward: the VLM predicts
them directly and they are used directly (see
``diff_vla_wbc/hand_stream.py`` for the PD/actuator side).  Only the 64-D token
enters SuperMimic.

Why a separate head instead of widening the action head:

* the flow-matching head's output width is GR00T's ``max_action_dim`` (64) and is
  consumed by the adapter with a fixed contract; widening it would invalidate
  cached tokens and checkpoints;
* a separate head has its own loss, its own optimizer group and can be trained
  with hand-only labels (the body branch stays untouched);
* it is exactly the split ``gear_sonic`` uses (``hand_action_dim`` for a separate
  hand decoder: ``gear_sonic/trl/modules/universal_token_modules.py:90``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hand_stream import HAND_DIM


@dataclass
class HandHeadCfg:
    """Configuration of the hand head.

    ``horizon`` is the number of future control steps predicted per call.  The
    values are hand joint angles in radians, clamped to the model's joint limits
    when they reach the actuator layer (``HandStream.command``).
    """

    in_dim: int = 2048  # Cosmos-Reason2-2B / GR00T backbone width
    hidden: int = 512
    horizon: int = 1
    out_dim: int = HAND_DIM
    dropout: float = 0.0


class HandHead(nn.Module):
    """Masked-mean pool over the VLM token sequence, then an MLP per hand step."""

    def __init__(self, cfg: HandHeadCfg | None = None):
        super().__init__()
        self.cfg = cfg or HandHeadCfg()
        c = self.cfg
        self.norm = nn.LayerNorm(c.in_dim)
        self.net = nn.Sequential(
            nn.Linear(c.in_dim, c.hidden),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden, c.hidden),
            nn.GELU(),
            nn.Linear(c.hidden, c.horizon * c.out_dim),
        )
        self.register_buffer("_zero", torch.zeros(()), persistent=False)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``features [B, T, D]`` -> hand joint angles ``[B, horizon, out_dim]``.

        ``mask`` is an optional ``[B, T]`` 0/1 float/bool mask (padded tokens are
        excluded from the pool).
        """
        if features.dim() != 3:
            raise ValueError(f"expected features [B, T, D], got {tuple(features.shape)}")
        h = self.norm(features)
        if mask is not None:
            m = mask.to(dtype=h.dtype).unsqueeze(-1)  # [B, T, 1]
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        else:
            pooled = h.mean(dim=1)
        out = self.net(pooled)
        return out.view(features.shape[0], self.cfg.horizon, self.cfg.out_dim)

    @staticmethod
    def loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        beta: float = 0.1,
    ) -> torch.Tensor:
        """Smooth-L1 on hand joint angles (radians). ``target`` is ``[B, H, 14]``."""
        if pred.shape != target.shape:
            raise ValueError(f"pred {tuple(pred.shape)} != target {tuple(target.shape)}")
        per = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
        if mask is not None:
            m = mask.to(per.dtype).unsqueeze(-1)
            return (per * m).sum() / m.sum().clamp_min(1.0)
        return per.mean()


def attach_hand_head(model: nn.Module, cfg: HandHeadCfg | None = None) -> HandHead:
    """Attach a fresh hand head to a ``Gr00tVLM`` instance and return it."""
    head = HandHead(cfg)
    model.hand_head = head
    return head


__all__ = ["HandHead", "HandHeadCfg", "attach_hand_head"]
