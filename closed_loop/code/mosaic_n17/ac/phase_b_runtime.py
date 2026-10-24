"""Phase B consequence model for the SONIC78 route (M3).

``(current_typed_slots, state46, commanded motion64+hands14[H]) -> future typed
slots``.  The model is deliberately small: one shared dynamics Transformer whose
memory is ``14 typed slots + 1 state token + H*3 action tokens``, and one decoder
query per slot type.  It never consumes 49D joints and never reads a future
action stream.

Supervision is restricted to the *executed* prefix: a 40-step candidate is the
condition, but deployment executes only the first ``execute_steps`` before
replanning, so labelling the tail as an observed outcome would train on a future
the robot never reaches.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..contracts.sonic78 import ACTION78_DIM, SONIC78_V1, Sonic78Spec
from .slots import SLOT_DIM, TOTAL_SLOTS, validate_slots


TARGET_SEMANTICS_EXECUTED_PREFIX = "executed_prefix"
TARGET_SEMANTICS_FULL_HORIZON = "full_horizon"
TARGET_SEMANTICS = frozenset({TARGET_SEMANTICS_EXECUTED_PREFIX, TARGET_SEMANTICS_FULL_HORIZON})
GROUP_NAMES = ("motion", "left_hand", "right_hand")
GROUP_WIDTHS = (64, 7, 7)


def validate_consequence_supervision(
    *,
    supervision_step: int,
    spec: Sonic78Spec = SONIC78_V1,
    target_semantics: str = TARGET_SEMANTICS_EXECUTED_PREFIX,
) -> None:
    """Fail closed when a target claims an unexecuted consequence."""

    if target_semantics not in TARGET_SEMANTICS:
        raise ValueError(
            f"target_semantics must be one of {sorted(TARGET_SEMANTICS)}, got {target_semantics!r}"
        )
    if isinstance(supervision_step, bool) or not isinstance(supervision_step, int):
        raise TypeError("supervision_step must be an int")
    if target_semantics == TARGET_SEMANTICS_FULL_HORIZON:
        if spec.execute_steps != spec.horizon:
            raise ValueError(
                f"{spec.name} predicts H={spec.horizon} but executes {spec.execute_steps} steps; "
                "a full_horizon target would label the unexecuted tail"
            )
        limit = spec.horizon
    else:
        limit = spec.execute_steps
    if not 1 <= supervision_step <= limit:
        raise ValueError(
            f"supervision_step must be in [1,{limit}] for {target_semantics} targets of "
            f"{spec.name}, got {supervision_step}"
        )


def split_action78(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``[B,H,78]`` into ``motion64 / left7 / right7`` (widths enforced)."""

    if not torch.is_tensor(actions):
        raise TypeError("actions must be a torch.Tensor")
    if actions.ndim != 3 or actions.shape[-1] != ACTION78_DIM:
        raise ValueError(f"actions must be [B,H,{ACTION78_DIM}], got {tuple(actions.shape)}")
    if not actions.dtype.is_floating_point or not bool(torch.isfinite(actions).all()):
        raise ValueError("actions must be finite floating point")
    return torch.split(actions, GROUP_WIDTHS, dim=-1)


@dataclass(frozen=True)
class PhaseBConfig:
    slot_dim: int = SLOT_DIM
    state_dim: int = 46
    control_horizon: int = SONIC78_V1.horizon
    #: ``prefix_only`` scores just the executed prefix (20 ticks) instead of the whole 40-token
    #: candidate.  The recorded future at t+40 is the consequence of the *executed stream*
    #: (first 20 + replanned 20), not of the candidate's tail, so the strict supervision target
    #: for candidate scoring is the executed prefix.
    prefix_only: bool = False
    depth: int = 2
    num_heads: int = 8
    dropout: float = 0.0
    token_dim: int = SLOT_DIM
    #: Adds a low-dimensional auxiliary head predicting the future state46.  The slot target is
    #: dominated by dimensions that do not move, so this extra supervision is the cheapest way to
    #: force action-conditioned information into the shared trunk.
    aux_state46: bool = False
    #: Number of past slot frames (t-1..t-history_slots) fed as extra memory tokens.  A single
    #: frame does not reveal joint velocity/contact, so the future is underdetermined without it.
    history_slots: int = 0
    #: Number of already-executed action ticks (``actions[t-prev:t]``) fed as past action tokens.
    prev_action_ticks: int = 0

    def __post_init__(self) -> None:
        if self.slot_dim != SLOT_DIM:
            raise ValueError(f"typed slots are {SLOT_DIM}-wide, got {self.slot_dim}")
        if self.state_dim != 46:
            raise ValueError(f"N1.7/AC state46 is 46-wide, got {self.state_dim}")
        expected_horizon = SONIC78_V1.execute_steps if self.prefix_only else SONIC78_V1.horizon
        if self.control_horizon != expected_horizon:
            raise ValueError(
                f"control horizon must be {expected_horizon} for prefix_only={self.prefix_only}"
            )
        if self.depth <= 0 or self.num_heads <= 0 or self.slot_dim % self.num_heads:
            raise ValueError("depth/num_heads must be positive and divide the slot width")
        if not 0 <= self.history_slots <= 4:
            raise ValueError("history_slots must be in [0,4]")
        if self.prev_action_ticks not in (0, SONIC78_V1.execute_steps):
            raise ValueError(
                f"prev_action_ticks must be 0 or {SONIC78_V1.execute_steps}, got {self.prev_action_ticks}"
            )


class Sonic78PhaseB(nn.Module):
    """Typed action-conditioned consequence model."""

    def __init__(self, config: PhaseBConfig = PhaseBConfig()) -> None:
        super().__init__()
        self.config = config
        self.motion_proj = nn.Linear(GROUP_WIDTHS[0], config.token_dim)
        self.left_hand_proj = nn.Linear(GROUP_WIDTHS[1], config.token_dim)
        self.right_hand_proj = nn.Linear(GROUP_WIDTHS[2], config.token_dim)
        self.group_embedding = nn.Parameter(torch.zeros(1, 1, 3, config.token_dim))
        self.step_embedding = nn.Embedding(config.control_horizon, config.token_dim)
        self.state_projection = nn.Sequential(
            nn.Linear(config.state_dim, config.token_dim),
            nn.SiLU(),
            nn.Linear(config.token_dim, config.token_dim),
        )
        self.slot_type_embedding = nn.Parameter(torch.zeros(TOTAL_SLOTS, config.token_dim))
        self.memory_norm = nn.LayerNorm(config.token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.token_dim,
            nhead=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.depth)
        self.future_query = nn.Parameter(torch.zeros(TOTAL_SLOTS, config.token_dim))
        self.decoder = nn.MultiheadAttention(
            config.token_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(config.token_dim)
        self.delta_head = nn.Linear(config.token_dim, SLOT_DIM)
        self.state_head = (
            nn.Linear(config.token_dim, config.state_dim) if config.aux_state46 else None
        )
        self.history_offset_embedding = (
            nn.Embedding(config.history_slots, config.token_dim) if config.history_slots else None
        )
        self.past_action_embedding = (
            nn.Parameter(torch.zeros(1, 1, config.token_dim))
            if config.prev_action_ticks
            else None
        )

    def action_memory(self, actions: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Project ``[B,H,78]`` into ``[B,H*3,token_dim]`` memory."""

        if actions.ndim != 3 or actions.shape[-1] != ACTION78_DIM:
            raise ValueError(f"actions must be [B,H,{ACTION78_DIM}], got {tuple(actions.shape)}")
        horizon = int(actions.shape[1])
        if horizon != self.config.control_horizon:
            raise ValueError(
                f"actions must carry the contract horizon {self.config.control_horizon}, got {horizon}"
            )
        motion, left, right = split_action78(actions)
        tokens = torch.stack(
            (
                self.motion_proj(motion),
                self.left_hand_proj(left),
                self.right_hand_proj(right),
            ),
            dim=2,
        )  # [B,H,3,D]
        tokens = tokens + self.group_embedding
        steps = torch.arange(horizon, device=actions.device)
        tokens = tokens + self.step_embedding(steps).view(1, horizon, 1, -1)
        return tokens.reshape(actions.shape[0], horizon * 3, -1), horizon

    def forward(
        self,
        current_slots: torch.Tensor,
        state46: torch.Tensor,
        actions: torch.Tensor,
        *,
        history_slots: torch.Tensor | None = None,
        prev_actions: torch.Tensor | None = None,
    ) -> dict[str, object]:
        validate_slots(current_slots)
        if current_slots.shape[0] != actions.shape[0]:
            raise ValueError("slots and actions must share a batch size")
        if state46.ndim != 2 or state46.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"state46 must be [B,{self.config.state_dim}], got {tuple(state46.shape)}"
            )
        if state46.shape[0] != current_slots.shape[0]:
            raise ValueError("slots and state46 must share a batch size")
        if not bool(torch.isfinite(state46).all()):
            raise ValueError("state46 must be finite")

        action_tokens, horizon = self.action_memory(actions)
        slot_memory = current_slots + self.slot_type_embedding.to(current_slots.dtype)
        state_memory = self.state_projection(state46).unsqueeze(1)
        memories = [slot_memory, state_memory, action_tokens]
        if self.config.history_slots:
            if history_slots is None or history_slots.shape[1] != self.config.history_slots:
                raise ValueError(
                    f"history_slots must be [B,{self.config.history_slots},14,{SLOT_DIM}]"
                )
            offsets = torch.arange(self.config.history_slots, device=current_slots.device)
            frame_embedding = self.history_offset_embedding(offsets).view(
                1, self.config.history_slots, 1, -1
            )
            past_frames = history_slots + self.slot_type_embedding.view(
                1, 1, *self.slot_type_embedding.shape
            ).to(history_slots.dtype)
            memories.append((past_frames + frame_embedding).flatten(1, 2))
        if self.config.prev_action_ticks:
            if prev_actions is None or prev_actions.shape[1] != self.config.prev_action_ticks:
                raise ValueError(
                    f"prev_actions must be [B,{self.config.prev_action_ticks},78]"
                )
            past_action_tokens, _ = self.action_memory(prev_actions)
            memories.append(past_action_tokens + self.past_action_embedding)
        memory = self.memory_norm(torch.cat(memories, dim=1))
        decoded = self.encoder(memory)
        queries = (
            self.future_query.unsqueeze(0) + self.slot_type_embedding.to(current_slots.dtype)
        ).expand(current_slots.shape[0], -1, -1)
        query_tokens, _ = self.decoder(queries, decoded, decoded, need_weights=False)
        normalized = self.output_norm(queries + query_tokens)
        normalized_delta = self.delta_head(normalized)
        predicted = current_slots + normalized_delta
        predicted_state46 = (
            None if self.state_head is None else self.state_head(normalized.mean(dim=1))
        )
        return {
            "predicted_slots": predicted,
            "delta": normalized_delta,
            "predicted_state46": predicted_state46,
            "horizon": horizon,
        }


__all__ = [
    "GROUP_NAMES",
    "GROUP_WIDTHS",
    "PhaseBConfig",
    "Sonic78PhaseB",
    "TARGET_SEMANTICS",
    "TARGET_SEMANTICS_EXECUTED_PREFIX",
    "TARGET_SEMANTICS_FULL_HORIZON",
    "split_action78",
    "validate_consequence_supervision",
]
