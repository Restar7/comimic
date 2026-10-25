"""Differentiable imagined rollout: FM goal -> WBC -> world model -> task loss.

The rollout is the core of the co-training graph:

    a_theta = FM_theta(...)
    g_eta   = Adapter_eta(a_theta)
    u_psi   = pi_SM_psi(o_hist, g_eta)
    s_{t+1} = F_phi(s_t, u_psi, xi)
    L       = L_task(s_{1:H}) + L_track + L_goal + L_FM

``F_phi``'s parameters may be frozen, but its *inputs* remain differentiable.
Never wrap this function in ``torch.no_grad()`` when training the policy or FM
head; wrap only the separate world-model update.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch
import torch.nn.functional as F

if TYPE_CHECKING:  # annotations only: the Isaac path must not import the mjlab stack
    from .isaac_joint import FrameRef  # noqa: F401

# State/observation contract shared by every bundle (plain constants, no implementation import):
#   state 71 = [p(3), quat(4), q(29), v_w(3), w_w(3), qd(29)]
#   per-frame observation = 93 (paper) or 99 (relative_ori99)
ACTION_DIM = 29
HISTORY = 10
STATE_DIM = 71
GOAL_FRAMES = 10
GOAL_FEATURE_DIM = 68


@dataclass
class Rollout:
    states: torch.Tensor                 # (H+1, B, 71)
    actions: torch.Tensor                # (H, B, 29)
    rewards: torch.Tensor                # (H, B)
    reward_terms: dict[str, torch.Tensor]  # each (H, B)
    termination_probs: torch.Tensor      # (H+1, B)
    survival: torch.Tensor               # (H+1, B)
    goals: torch.Tensor                  # (H+1, B, 680) — or the raw token window on the token route
    goal_sequence: torch.Tensor          # (B, 10, 680)
    references: list[Any] = field(default_factory=list)
    history: torch.Tensor | None = None  # final (H, B, 93|99)


def _obs_terms_from_flat(flat: torch.Tensor, hist: int = HISTORY,
                         term_dims: tuple[int, ...] | None = None) -> torch.Tensor:
    """Split a flat term-major observation history into ``(B, hist, W)``.

    The default term split is the paper's 93-dim layout; a bundle that asks for a different
    width (``relative_ori99`` → 99) passes its own ``term_dims``.
    """
    dims = tuple(term_dims) if term_dims is not None else (29, 29, 3, 3, 29)
    if flat.ndim != 2:
        raise ValueError(f"obs_hist must be (B, hist*{sum(dims)}), got {tuple(flat.shape)}")
    b = flat.shape[0]
    if flat.shape[1] != sum(dims) * hist:
        raise ValueError(f"expected {sum(dims) * hist} obs dims, got {flat.shape[1]}")
    terms = []
    offset = 0
    for dim in dims:
        chunk = flat[:, offset : offset + dim * hist].reshape(b, hist, dim)
        terms.append(chunk)
        offset += dim * hist
    return torch.cat(terms, dim=-1)  # (B,H,W), term order restored


def _call_observation(bundle, next_state, action, xi, ref=None):
    """``bundle.observation(...)`` with an optional ``ref`` (needed by 99-dim bundles)."""
    try:
        return bundle.observation(next_state, action, xi, ref=ref)
    except TypeError:
        return bundle.observation(next_state, action, xi)


def _call_act(bundle, obs_flat, cond, ref=None):
    """``bundle.act(...)`` with an optional ``ref`` (ignored by stock bundles)."""
    try:
        return bundle.act(obs_flat, cond, ref=ref)
    except TypeError:
        return bundle.act(obs_flat, cond)


def _obs_width(bundle) -> int:
    """Per-frame observation width the bundle wants (93 by default, 99 for ``relative_ori99``)."""
    return int(getattr(bundle, "obs_width", 93 if bundle is not None else 93))


def _call_robot_bodies(bundle, state):
    """FK of the tracked bodies through the bundle (falls back to the raw analytic head).

    The bundle-level hook matters when the learned layer and the analytic layer disagree about
    the root-velocity convention: the trackerlab bridge (world-frame ω) converts to the body
    frame the mjlab FK expects.  Calling ``bundle.assets.heads.robot_bodies`` here would bypass
    that conversion and silently feed the reward world-frame angular velocity.
    """
    fn = getattr(bundle, "robot_bodies", None)
    if fn is not None:
        return fn(state)
    return bundle.assets.heads.robot_bodies(state)


def _ensure_history(
    obs_hist: torch.Tensor | None,
    history: torch.Tensor | None,
    batch: int,
    device: torch.device,
    width: int = 93,
    term_dims: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if history is not None:
        if history.shape == (batch, HISTORY, width):
            return history.permute(1, 0, 2).contiguous()
        if history.shape != (HISTORY, batch, width):
            raise ValueError(
                f"history must be {(HISTORY, batch, width)} or {(batch, HISTORY, width)}, "
                f"got {tuple(history.shape)}"
            )
        return history
    if obs_hist is None:
        return torch.zeros(HISTORY, batch, width, device=device)
    if obs_hist.shape != (batch, HISTORY * width):
        raise ValueError(f"obs_hist must be {(batch, HISTORY * width)}, got {tuple(obs_hist.shape)}")
    # (B,H,W) -> (H,B,W)
    return _obs_terms_from_flat(obs_hist, term_dims=term_dims).permute(1, 0, 2).contiguous()


def _flat_history(bundle, history: torch.Tensor) -> torch.Tensor:
    return bundle.flatten_history(history)


def imagine_with_goals(
    bundle,
    *,
    fm_tokens: torch.Tensor | None = None,
    adapter: torch.nn.Module | None = None,
    goals: torch.Tensor | None = None,
    state: torch.Tensor,
    obs_hist: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
    xi: torch.Tensor | None = None,
    references: Sequence[Any] | None = None,
    prev_action: torch.Tensor | None = None,
    horizon: int = 8,
    obs_noise: float = 0.0,
) -> Rollout:
    """Roll the WBC + world model forward with differentiable generated goals.

    Supply either ``fm_tokens`` + ``adapter``, or a precomputed ``goals``
    tensor of shape ``(B,10,68)`` / ``(B,680)``.  If ``references`` is omitted,
    zero references are not supported: the tracking reward needs a real
    reference frame, so pass the teacher references for every step.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    batch = state.shape[0]
    device = state.device
    xi = torch.zeros(batch, 33, device=device) if xi is None else xi
    width = _obs_width(bundle)
    history = _ensure_history(obs_hist, history, batch, device, width,
                              term_dims=getattr(bundle, "term_dims", None))
    if prev_action is None:
        prev_action = torch.zeros(batch, ACTION_DIM, device=device)
    if references is not None and len(references) < horizon:
        raise ValueError(f"need {horizon} reference frames, got {len(references)}")

    if adapter is not None:
        if fm_tokens is None:
            raise ValueError("fm_tokens are required when an adapter is supplied")
        goal_frames = adapter(fm_tokens)
    elif goals is not None:
        goal_frames = goals
    else:
        raise ValueError("pass either fm_tokens+adapter or goals")

    if goal_frames.ndim == 2:
        goal_frames = goal_frames.reshape(goal_frames.shape[0], GOAL_FRAMES, -1)
    if goal_frames.ndim != 3 or goal_frames.shape[0] != batch:
        raise ValueError(f"goals must be (B,{GOAL_FRAMES},D), got {tuple(goal_frames.shape)}")
    goal_dim = goal_frames.shape[-1] * goal_frames.shape[1]
    if goal_dim != bundle.goal_dim:
        raise ValueError(
            f"goal dim {goal_dim} != policy goal dim {bundle.goal_dim}; "
            "check AdapterConfig.goal_frame_dim and the policy config"
        )
    goal_sequence = goal_frames

    def goal_window_at(step: int) -> torch.Tensor:
        idx = torch.clamp(torch.arange(goal_frames.shape[1], device=device) + step, max=goal_frames.shape[1] - 1)
        return goal_frames[:, idx].reshape(batch, -1)

    current = state
    a_prev = prev_action
    all_states = [current]
    all_actions, all_rewards, all_terms = [], [], []
    all_probs = [torch.zeros(batch, device=device)]
    all_survival = [torch.ones(batch, device=device)]
    all_goals = [goal_window_at(0)]
    term_names = None
    survival = torch.ones(batch, device=device)

    for t in range(horizon):
        if references is None:
            raise ValueError("references are required for tracking reward in the rollout")
        ref = references[t]
        g = goal_window_at(t)
        obs_flat = _flat_history(bundle, history)
        action = _call_act(bundle, obs_flat, g, ref)
        next_state = bundle.step(current, action, xi)
        bodies = _call_robot_bodies(bundle, next_state)
        terms = bundle.reward(next_state, action, a_prev, ref, bodies)
        logits = bundle.termination_logits(next_state, ref)
        prob = torch.sigmoid(logits)
        survival = survival * (1.0 - prob)
        clean_obs = _call_observation(bundle, next_state, action, xi, ref)
        if obs_noise > 0:
            clean_obs = clean_obs + (torch.rand_like(clean_obs) * 2 - 1) * obs_noise
        history = torch.cat((history[1:], clean_obs.unsqueeze(0)), dim=0)

        if term_names is None:
            term_names = [k for k in terms if k != "total"]
            all_terms = {k: [] for k in term_names}
        all_terms = {k: v + [terms[k]] for k, v in all_terms.items()} if all_terms else all_terms
        all_states.append(next_state)
        all_actions.append(action)
        all_rewards.append(terms["total"])
        all_probs.append(prob)
        all_survival.append(survival)
        all_goals.append(g)
        a_prev = action
        current = next_state

    return Rollout(
        states=torch.stack(all_states, dim=0),
        actions=torch.stack(all_actions, dim=0),
        rewards=torch.stack(all_rewards, dim=0),
        reward_terms={k: torch.stack(v, dim=0) for k, v in (all_terms or {}).items()},
        termination_probs=torch.stack(all_probs, dim=0),
        survival=torch.stack(all_survival, dim=0),
        goals=torch.stack(all_goals, dim=0),
        goal_sequence=goal_sequence,
        references=list(references) if references is not None else [],
        history=history,
    )


@dataclass
class LossWeights:
    goal: float = 1.0
    reward: float = 1.0
    state: float = 0.5
    smooth: float = 1e-3
    task: float = 1.0
    safety: float = 0.01
    fm: float = 1.0
    termination: float = 0.0
    #: 速度通道（``v/w/qd``）在 ``state`` 项里的额外权重。实测：Isaac WM 想象出来的速度与参考
    #: 速度相关只有 0.077（位置/关节是 1e-4 / 0.03 的精度），速度通道占了 state 项 96% 的能量
    #: 且不可学习 —— 训练侧用 ``--state-vel-weight 0.05`` 降权，默认 1.0 保持旧行为。
    state_vel: float = 1.0


def _state_error(pred: torch.Tensor, target: torch.Tensor, vel_weight: float = 1.0) -> torch.Tensor:
    """Position / orientation / velocity error with a geodesic quaternion term."""
    from .compat import quat_error_magnitude

    pos = F.mse_loss(pred[..., 0:3], target[..., 0:3])
    ori = quat_error_magnitude(pred[..., 3:7], target[..., 3:7]).square().mean()
    q = F.mse_loss(pred[..., 7:36], target[..., 7:36])
    vel = F.mse_loss(pred[..., 36:], target[..., 36:])
    return pos + ori + q + float(vel_weight) * vel


def discounted_return(
    rollout: Rollout,
    gamma: float = 0.99,
    bootstrap: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable discounted tracking return (per batch element)."""
    h = rollout.rewards.shape[0]
    disc = gamma ** torch.arange(h, device=rollout.rewards.device, dtype=rollout.rewards.dtype)
    out = (disc[:, None] * rollout.survival[:h] * rollout.rewards).sum(0)
    if bootstrap is not None:
        out = out + (gamma**h) * rollout.survival[h] * bootstrap
    return out


def imagine_with_tokens(
    bundle,
    *,
    tokens: torch.Tensor,
    state: torch.Tensor,
    obs_hist: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
    xi: torch.Tensor | None = None,
    references: Sequence[Any] | None = None,
    prev_action: torch.Tensor | None = None,
    horizon: int = 8,
    obs_noise: float = 0.0,
) -> Rollout:
    """Token route: the conditioning is the VLA's own 64-d motion token.

    Same rollout/return contract as :func:`imagine_with_goals`, but step ``t`` is conditioned on
    ``tokens[:, min(t, T-1)]`` directly (no goal encoder, no ``adapter``) — this is the interface a
    token-conditioned SuperMimic checkpoint expects (``PolicyCfg.token_dim > 0``).
    ``Rollout.goals`` / ``goal_sequence`` then carry the tokens themselves, so the goal-alignment
    loss is simply not computed (``compute_joint_loss`` skips it when no teacher goals are passed).
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if tokens.ndim != 3:
        raise ValueError(f"tokens must be (B,T,D), got {tuple(tokens.shape)}")
    batch = state.shape[0]
    device = state.device
    if tokens.shape[0] != batch:
        raise ValueError(f"tokens batch {tokens.shape[0]} != {batch}")
    token_dim = int(tokens.shape[-1])
    if token_dim != int(getattr(bundle.policy, "token_dim", 0)):
        raise ValueError(f"token dim {token_dim} != policy token_dim {bundle.policy.token_dim}")

    xi = torch.zeros(batch, 33, device=device) if xi is None else xi
    width = _obs_width(bundle)
    history = _ensure_history(obs_hist, history, batch, device, width,
                              term_dims=getattr(bundle, "term_dims", None))
    if prev_action is None:
        prev_action = torch.zeros(batch, ACTION_DIM, device=device)
    if references is not None and len(references) < horizon:
        raise ValueError(f"need {horizon} reference frames, got {len(references)}")

    def token_at(step: int) -> torch.Tensor:
        return tokens[:, min(step, tokens.shape[1] - 1)]

    current = state
    a_prev = prev_action
    all_states = [current]
    all_actions, all_rewards, all_terms = [], [], []
    all_probs = [torch.zeros(batch, device=device)]
    all_survival = [torch.ones(batch, device=device)]
    all_tokens = [token_at(0)]
    term_names = None
    survival = torch.ones(batch, device=device)

    for t in range(horizon):
        if references is None:
            raise ValueError("references are required for tracking reward in the rollout")
        ref = references[t]
        cond = token_at(t)
        obs_flat = _flat_history(bundle, history)
        action = _call_act(bundle, obs_flat, cond, ref)
        next_state = bundle.step(current, action, xi)
        bodies = _call_robot_bodies(bundle, next_state)
        terms = bundle.reward(next_state, action, a_prev, ref, bodies)
        logits = bundle.termination_logits(next_state, ref)
        prob = torch.sigmoid(logits)
        survival = survival * (1.0 - prob)
        clean_obs = _call_observation(bundle, next_state, action, xi, ref)
        if obs_noise > 0:
            clean_obs = clean_obs + (torch.rand_like(clean_obs) * 2 - 1) * obs_noise
        history = torch.cat((history[1:], clean_obs.unsqueeze(0)), dim=0)

        if term_names is None:
            term_names = [k for k in terms if k != "total"]
            all_terms = {k: [] for k in term_names}
        all_terms = {k: v + [terms[k]] for k, v in all_terms.items()} if all_terms else all_terms
        all_states.append(next_state)
        all_actions.append(action)
        all_rewards.append(terms["total"])
        all_probs.append(prob)
        all_survival.append(survival)
        all_tokens.append(cond)
        a_prev = action
        current = next_state

    return Rollout(
        states=torch.stack(all_states, dim=0),
        actions=torch.stack(all_actions, dim=0),
        rewards=torch.stack(all_rewards, dim=0),
        reward_terms={k: torch.stack(v, dim=0) for k, v in (all_terms or {}).items()},
        termination_probs=torch.stack(all_probs, dim=0),
        survival=torch.stack(all_survival, dim=0),
        goals=torch.stack(all_tokens, dim=0),
        goal_sequence=tokens,
        references=list(references) if references is not None else [],
        history=history,
    )


def compute_joint_loss(
    rollout: Rollout,
    *,
    teacher_goals: torch.Tensor | None = None,
    teacher_goal_sequence: torch.Tensor | None = None,
    teacher_states: torch.Tensor | None = None,
    task_loss_fn: Callable[[Rollout], torch.Tensor] | None = None,
    fm_supervised_loss: torch.Tensor | None = None,
    done: torch.Tensor | None = None,
    gamma: float = 0.99,
    weights: LossWeights = LossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine tracking, goal alignment, task, smoothness, safety and FM losses."""
    terms: dict[str, torch.Tensor] = {}

    if weights.reward:
        ret = discounted_return(rollout, gamma)
        terms["reward"] = -ret.mean()
    if teacher_goal_sequence is not None:
        target = teacher_goal_sequence
        if target.ndim == 2:
            target = target.reshape(target.shape[0], GOAL_FRAMES, -1)
        pred = rollout.goals[:-1].permute(1, 0, 2)  # (B,H,680)
        steps = min(pred.shape[1], target.shape[1])
        terms["goal"] = F.smooth_l1_loss(pred[:, :steps], target[:, :steps])
    elif teacher_goals is not None:
        target = teacher_goals.reshape(teacher_goals.shape[0], GOAL_FRAMES, -1)
        terms["goal"] = F.smooth_l1_loss(rollout.goal_sequence, target)
    if teacher_states is not None:
        steps = min(rollout.states.shape[0] - 1, teacher_states.shape[1])
        target_states = teacher_states[:, :steps].permute(1, 0, 2)  # (H,B,71)
        terms["state"] = _state_error(rollout.states[1 : steps + 1], target_states,
                                      vel_weight=weights.state_vel)
    if rollout.actions.shape[0] > 1:
        terms["smooth"] = (rollout.actions[1:] - rollout.actions[:-1]).square().mean()
    if task_loss_fn is not None:
        terms["task"] = task_loss_fn(rollout)
    if weights.safety and rollout.termination_probs.shape[0] > 1:
        terms["safety"] = rollout.termination_probs[1:].mean()
    if done is not None:
        probs = rollout.termination_probs[1:].clamp(1e-6, 1 - 1e-6)
        steps = min(probs.shape[0], done.shape[1])
        terms["termination"] = F.binary_cross_entropy(probs[:steps], done[:, :steps].float())
    if fm_supervised_loss is not None:
        terms["fm"] = fm_supervised_loss

    if not terms:
        raise ValueError("no loss term produced; check the supplied targets and weights")
    total = sum(getattr(weights, name) * value for name, value in terms.items())
    return total, terms


def freeze_world_model(world_model: torch.nn.Module) -> None:
    """Freeze WM parameters but keep gradients flowing through its inputs."""
    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad_(False)


def unfreeze_world_model(world_model: torch.nn.Module) -> None:
    for p in world_model.parameters():
        p.requires_grad_(True)


__all__ = [
    "LossWeights",
    "Rollout",
    "compute_joint_loss",
    "discounted_return",
    "freeze_world_model",
    "imagine_with_goals",
    "unfreeze_world_model",
]
