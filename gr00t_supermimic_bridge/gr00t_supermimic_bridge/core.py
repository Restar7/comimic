"""PyTorch bridge: GR00T FM -> physical motion goals -> SuperMimic -> world model.

Integration scaffold, not a drop-in trained controller. Tested with toy network
modules, not with the original large VLA checkpoint, gated BONES-SEED or a robot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

State = Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor, torch.Tensor, torch.Tensor]


def sample_fm_with_grad(
    action_head: nn.Module,
    backbone_output: Any,
    action_input: Any,
    initial_noise: Optional[torch.Tensor] = None,
    num_steps: Optional[int] = None,
) -> torch.Tensor:
    """Gradient-enabled equivalent of the repository's no_grad get_action.

    No RTC/inpainting here; extend it explicitly if that training regime is needed.
    backbone_output and action_input are the native GR00T BatchFeature objects.
    """
    features = action_head._encode_features(backbone_output, action_input)
    vl_embeddings = features.backbone_features
    state_embeddings = features.state_features
    embodiment_id = action_input.embodiment_id

    batch_size = vl_embeddings.shape[0]
    steps = int(num_steps or action_head.num_inference_timesteps)
    if steps <= 0:
        raise ValueError("num_steps must be positive")
    expected = (batch_size, action_head.action_horizon, action_head.action_dim)
    if initial_noise is None:
        actions = torch.randn(expected, device=vl_embeddings.device, dtype=vl_embeddings.dtype)
    else:
        if tuple(initial_noise.shape) != expected:
            raise ValueError(f"expected initial_noise {expected}, got {tuple(initial_noise.shape)}")
        actions = initial_noise
    dt = 1.0 / steps

    for index in range(steps):
        time_bucket = int((index / float(steps)) * action_head.num_timestep_buckets)
        timestep = torch.full(
            (batch_size,), time_bucket, dtype=torch.long, device=actions.device
        )
        action_embeddings = action_head.action_encoder(actions, timestep, embodiment_id)
        if action_head.config.add_pos_embed:
            positions = torch.arange(
                action_embeddings.shape[1], device=actions.device, dtype=torch.long
            )
            action_embeddings = (
                action_embeddings + action_head.position_embedding(positions).unsqueeze(0)
            )
        joined = torch.cat((state_embeddings, action_embeddings), dim=1)
        args = dict(
            hidden_states=joined,
            encoder_hidden_states=vl_embeddings,
            timestep=timestep,
        )
        if action_head.config.use_alternate_vl_dit:
            args["image_mask"] = backbone_output.image_mask
            args["backbone_attention_mask"] = backbone_output.backbone_attention_mask
        hidden = action_head.model(**args)
        predicted_velocity = action_head.action_decoder(hidden, embodiment_id)[
            :, -action_head.action_horizon:
        ]
        actions = actions + dt * predicted_velocity
    return actions


def sample_model_with_grad(
    gr00t_model: nn.Module,
    model_inputs: dict,
    initial_noise: Optional[torch.Tensor] = None,
    num_steps: Optional[int] = None,
) -> torch.Tensor:
    """Keep everything inside one Torch graph; do not use Gr00tPolicy.get_action."""
    backbone_inputs, action_inputs = gr00t_model.prepare_input(dict(model_inputs))
    backbone_output = gr00t_model.backbone(backbone_inputs)
    return sample_fm_with_grad(
        gr00t_model.action_head, backbone_output, action_inputs, initial_noise, num_steps
    )


class MotionGoalAdapter(nn.Module):
    """Map GR00T [B,40,64] to per-control-frame [B,40,10*73].

    A 73D frame has (root offset 3, root rotation 6, linear velocity 3,
    angular velocity 3, joint position 29, 0.1*joint velocity 29).
    Supervise against MotionLib.goal(...) BEFORE trusting imagined rollouts.
    """
    def __init__(self, in_dim: int = 64, goal_frames: int = 10,
                 goal_frame_dim: int = 73, hidden: int = 256):
        super().__init__()
        self.goal_dim = goal_frames * goal_frame_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.goal_dim),
        )

    def forward(self, fm_tokens: torch.Tensor) -> torch.Tensor:
        if fm_tokens.ndim != 3:
            raise ValueError("Expected FM tokens [B,T,C]")
        return self.net(fm_tokens.float())


def mat_to_6d(rot: torch.Tensor) -> torch.Tensor:
    # Exactly the column convention used in Uni-SuperMimic geometry.py.
    return rot[..., :, :2].reshape(rot.shape[:-2] + (6,))


def rot6d_to_mat(six: torch.Tensor) -> torch.Tensor:
    # Inverse of the repository's row-major-flattened first-two-column convention.
    axes = six.reshape(six.shape[:-1] + (3, 2))
    first = F.normalize(axes[..., 0], dim=-1, eps=1e-6)
    second_raw = axes[..., 1]
    second = second_raw - (first * second_raw).sum(-1, keepdim=True) * first
    second = F.normalize(second, dim=-1, eps=1e-6)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def yaw_mat(rot: torch.Tensor) -> torch.Tensor:
    """Use the same yaw-extraction convention as SuperMimic geometry.py."""
    forward_x, forward_y = rot[..., 0, 0], rot[..., 1, 0]
    norm = torch.sqrt(forward_x.square() + forward_y.square() + 1e-9)
    c, s = forward_x / norm, forward_y / norm
    zero = torch.zeros_like(c)
    one = torch.ones_like(c)
    return torch.stack(
        (c, -s, zero, s, c, zero, zero, zero, one), -1
    ).reshape(rot.shape[:-2] + (3, 3))


def normalize_context(context: torch.Tensor) -> torch.Tensor:
    """SuperMimic MDP.normalize_context, xi is [B,33]."""
    assert context.shape[-1] == 33
    return torch.cat((
        (context[..., :1] - 0.75) / 0.45,
        context[..., 1:30] / 0.01,
        context[..., 30:33] / 0.05,
    ), dim=-1)


def proprio_obs(
    state: State, prev_action: torch.Tensor, context: torch.Tensor,
    default_q: torch.Tensor, reference_world_rot: torch.Tensor
) -> torch.Tensor:
    """Native 99D unscaled proprioception, matching SuperMimic mdp.obs."""
    _, rot, _, angvel, q, dq = state
    r_t = rot.transpose(-1, -2)
    gravity = -rot[..., 2, :]
    rel = mat_to_6d(r_t @ reference_world_rot)
    return torch.cat(
        (
            q - default_q - context[..., 1:30],
            dq,
            (r_t @ angvel.unsqueeze(-1)).squeeze(-1),
            gravity,
            prev_action,
            rel,
        ), dim=-1
    )


@dataclass
class Rollout:
    goals: torch.Tensor
    positions: torch.Tensor
    rotations: torch.Tensor
    joint_positions: torch.Tensor
    joint_velocities: torch.Tensor
    actions: torch.Tensor


def imagined_rollout(
    fm_tokens: torch.Tensor,
    adapter: MotionGoalAdapter,
    pi: nn.Module,
    wm: nn.Module,
    initial_state: State,
    obs_hist: torch.Tensor,
    context: torch.Tensor,
    default_q: torch.Tensor,
    *,
    horizon: int = 8,
    reference_world_rotations: Optional[torch.Tensor] = None,
) -> Rollout:
    """Differentiate through goal adapter + WBC policy + short world-model rollout.

    If reference_world_rotations is supplied, use teacher/data references
    [B,horizon,3,3]. Otherwise estimate from each generated motion goal and the
    predicted robot yaw. The latter is an approximation that MUST be calibrated
    on the actual control dataset before real deployment.
    """
    goals = adapter(fm_tokens)
    batch_size, available_steps, goal_dim = goals.shape
    if goal_dim != 730:
        raise ValueError("Expected 10x73 physical goal format (730D)")
    if not 1 <= horizon <= available_steps:
        raise ValueError(f"horizon={horizon} exceeds FM horizon={available_steps}")
    if obs_hist.shape != (batch_size, 10, 99):
        raise ValueError(f"Expected obs_hist [B,10,99], got {obs_hist.shape}")
    if reference_world_rotations is not None and (
        reference_world_rotations.shape != (batch_size, horizon, 3, 3)
    ):
        raise ValueError("reference_world_rotations shape mismatch")

    normalized_context = normalize_context(context)
    state = initial_state
    all_positions, all_rotations, all_q, all_dq, all_actions = [], [], [], [], []
    for step in range(horizon):
        goal = goals[:, step]                       # [B,730] future motion goal
        raw_action = pi(obs_hist, goal)              # [B,29], normalized offset
        state = wm.step(state, raw_action, normalized_context)  # differentiable
        if reference_world_rotations is None:
            goal_first_frame = goal.view(batch_size, 10, 73)[:, 0]
            # 6D target root rotation expressed in the reference heading frame.
            # This is an on-policy approximation, not a substitute for a
            # calibrated global root pose / heading estimator.
            ref_R = yaw_mat(state[1]) @ rot6d_to_mat(goal_first_frame[:, 3:9])
        else:
            ref_R = reference_world_rotations[:, step]
        observation = proprio_obs(state, raw_action, context, default_q, ref_R)
        obs_hist = torch.cat((obs_hist[:, 1:], observation.unsqueeze(1)), dim=1)
        all_positions.append(state[0])
        all_rotations.append(state[1])
        all_q.append(state[4])
        all_dq.append(state[5])
        all_actions.append(raw_action)

    return Rollout(
        goals=goals,
        positions=torch.stack(all_positions, dim=1),
        rotations=torch.stack(all_rotations, dim=1),
        joint_positions=torch.stack(all_q, dim=1),
        joint_velocities=torch.stack(all_dq, dim=1),
        actions=torch.stack(all_actions, dim=1),
    )


@dataclass
class LossWeights:
    goal: float = 1.0
    joints: float = 1.0
    root: float = 1.0
    smooth: float = 1e-3
    task: float = 1.0
    fm: float = 1.0


def compute_loss(
    rollout: Rollout,
    *,
    teacher_goals: Optional[torch.Tensor] = None,
    teacher_joint_positions: Optional[torch.Tensor] = None,
    teacher_root_positions: Optional[torch.Tensor] = None,
    task_loss_fn: Optional[Callable[[Rollout], torch.Tensor]] = None,
    fm_supervised_loss: Optional[torch.Tensor] = None,
    weights: LossWeights = LossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply physically meaningful supervision plus optional differentiable task loss.

    task_loss_fn can use predicted robot state. For object manipulation,
    attach a separately trained differentiable robot+object/contact model.
    """
    terms: dict[str, torch.Tensor] = {}
    if teacher_goals is not None:
        terms["goal"] = F.smooth_l1_loss(rollout.goals, teacher_goals)
    if teacher_joint_positions is not None:
        steps = rollout.joint_positions.shape[1]
        terms["joints"] = F.mse_loss(
            rollout.joint_positions, teacher_joint_positions[:, :steps]
        )
    if teacher_root_positions is not None:
        steps = rollout.positions.shape[1]
        terms["root"] = F.mse_loss(rollout.positions, teacher_root_positions[:, :steps])
    if rollout.actions.shape[1] > 1:
        terms["smooth"] = (
            rollout.actions[:, 1:] - rollout.actions[:, :-1]
        ).square().mean()
    if task_loss_fn is not None:
        terms["task"] = task_loss_fn(rollout)
    if fm_supervised_loss is not None:
        terms["fm"] = fm_supervised_loss

    if not terms:
        raise ValueError("At least one loss term is required")
    total = sum(getattr(weights, name) * value for name, value in terms.items())
    return total, terms


def freeze_world_model(world_model: nn.Module) -> None:
    """Weights frozen; gradients STILL propagate through its inputs."""
    world_model.eval()
    for parameter in world_model.parameters():
        parameter.requires_grad_(False)


def load_supermimic_checkpoint(path: str, device: str = "cpu"):
    """Instantiate official Uni-SuperMimic Policy and WorldModel from checkpoint.

    Requires cloned/installed Uni-SuperMimic import path. Supports normal training
    checkpoints with no test_hidden_dims; returns modules without their environment.
    """
    from unisupermimic.rl.algos.supermimic.networks import Policy, WorldModel

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    args = checkpoint["args"]
    if args.get("test_hidden_dims") is not None:
        raise ValueError("Test-shape checkpoints need learner-specific reconstruction")
    hist = int(args["hist"])
    goal_frames = int(args["goal_frames"])
    if hist != 10 or goal_frames != 10:
        raise ValueError("This bridge currently targets hist=10, goal_frames=10")
    wm = WorldModel(hidden=tuple(args["wm_hidden"]), num_joints=29, context_dim=33)
    pi = Policy(
        obs_dim=99, goal_dim=goal_frames * 73, hist=hist,
        z_dim=int(args["z_dim"]), num_joints=29,
        hidden=tuple(args["policy_hidden"]),
    )
    wm.load_state_dict(checkpoint["wm"], strict=True)
    pi.load_state_dict(checkpoint["pi"], strict=True)
    return wm.to(device), pi.to(device)
