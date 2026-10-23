"""Typed access to the cloned SuperMimic-Reproduced policy / world model.

This module is deliberately *thin*: it builds the exact upstream ``Policy`` and
``WorldModel`` objects, loads their checkpoints, and exposes the state / goal /
observation contracts used by the differentiable bridge.

Upstream contract (CuboiLeo/SuperMimic-Reproduced):

* state       ``s``  = ``[root_pos(3), root_quat wxyz(4), q(29), v_w(3), w_b(3), dq(29)]``  (71)
* observation ``o``  = ``[q - q0(29), dq(29), w_b(3), projected_gravity(3), a_prev(29)]``   (93)
* goal        ``g``  = 10 future frames x 68
                       ``[q - q0(29), dq(29), gravity(3), w_b(3), v_heading(3), height(1)]``
* action      ``a``  = 29 normalized joint-position offsets
* q_target           = ``q0 + G1_ACTION_SCALE * a``  (no encoder-bias compensation in sim)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from .compat import (
    ANCHOR_BODY,
    G1_ACTION_SCALE,
    G1_ARMATURE,
    G1_JOINT_NAMES,
    G1_KD,
    G1_KP,
    ROOT_BODY,
    TRACKED_BODIES,
    WORKSPACE_ROOT,
    default_qpos0,
    ensure_supermimic_importable,
    g1_default_joint_pos,
    g1_xml_path,
    load_g1_model,
    soft_joint_limits,
)

ensure_supermimic_importable()

from supermimic.envs.mdp.motion_library import MotionLibrary  # noqa: E402
from supermimic.kinematics.g1_fk import G1Kinematics  # noqa: E402
from supermimic.policy.nets import Policy, PolicyCfg  # noqa: E402
from supermimic.world_model.heads import (  # noqa: E402
    OBS_DIM,
    AnalyticHeads,
    ReferenceFrame,
    analytic_observation,
    flatten_history_term_major,
)
from supermimic.world_model.state_spec import StateSpecCfg  # noqa: E402
from supermimic.world_model.world_model import WorldModel, WorldModelCfg  # noqa: E402

STATE_DIM = 71
ACTION_DIM = 29
GOAL_FRAMES = 10
GOAL_FEATURE_DIM = 68
GOAL_DIM = GOAL_FRAMES * GOAL_FEATURE_DIM
XI_DIM = 33
HISTORY = 10


@dataclass
class G1Assets:
    """Robot-side quantities shared by the WM reward head and MuJoCo eval."""

    xml_path: Path
    model: Any
    fk: G1Kinematics
    heads: AnalyticHeads
    default_joint_pos: torch.Tensor
    qpos0: torch.Tensor
    soft_limits: torch.Tensor
    action_scale: torch.Tensor
    kp: torch.Tensor
    kd: torch.Tensor
    armature: torch.Tensor
    device: torch.device
    dtype: torch.dtype = torch.float32

    def q_target(self, action: torch.Tensor, joint_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Normalized policy action -> MuJoCo joint position target."""
        target = self.default_joint_pos + self.action_scale * action
        if joint_bias is not None:
            target = target + joint_bias
        return target

    def state_from_qpos_qvel(self, qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
        return torch.cat((qpos, qvel), dim=-1)

    def split_state(self, state: torch.Tensor):
        return (
            state[..., 0:3],
            state[..., 3:7],
            state[..., 7:36],
            state[..., 36:39],
            state[..., 39:42],
            state[..., 42:71],
        )


def build_g1_assets(
    xml_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> G1Assets:
    device = torch.device(device)
    resolved_xml = g1_xml_path(xml_path)
    model = load_g1_model(resolved_xml)
    fk = G1Kinematics(model, device=device, dtype=dtype)
    default_q = g1_default_joint_pos().to(device=device, dtype=dtype)
    limits = soft_joint_limits(model).to(device=device, dtype=dtype)
    heads = AnalyticHeads(
        fk,
        TRACKED_BODIES,
        ANCHOR_BODY,
        ROOT_BODY,
        limits,
        default_q,
    )
    return G1Assets(
        xml_path=resolved_xml,
        model=model,
        fk=fk,
        heads=heads,
        default_joint_pos=default_q,
        qpos0=default_qpos0(model).to(device=device, dtype=dtype),
        soft_limits=limits,
        action_scale=G1_ACTION_SCALE.to(device=device, dtype=dtype),
        kp=G1_KP.to(device=device, dtype=dtype),
        kd=G1_KD.to(device=device, dtype=dtype),
        armature=G1_ARMATURE.to(device=device, dtype=dtype),
        device=device,
        dtype=dtype,
    )


def default_xi(batch_size: int, device: str | torch.device = "cuda") -> torch.Tensor:
    """Nominal sim context: friction 0.75, zero encoder bias, zero CoM offset."""
    xi = torch.zeros(batch_size, XI_DIM, device=device)
    xi[:, 0] = 0.75
    return xi


def _coerce_cfg(cls, data: dict):
    """Turn JSON lists back into tuples, recursively for nested dataclasses."""
    kwargs = {}
    for key, value in data.items():
        if isinstance(value, list) and key in {
            "hidden",
            "encoder_hidden",
            "decoder_hidden",
            "obs_term_scale",
            "goal_term_scale",
            "err_scale",
            "out_scale",
            "in_scale",
            "xi_offset",
            "xi_scale",
            "term_hidden",
            "betas",
        }:
            value = tuple(value)
        if key == "state_spec" and isinstance(value, dict):
            value = _coerce_cfg(StateSpecCfg, value)
        kwargs[key] = value
    return cls(**kwargs)


class SuperMimicBundle(nn.Module):
    """The upstream WBC policy + learned differentiable world model."""

    def __init__(
        self,
        policy: Policy,
        world_model: WorldModel,
        assets: G1Assets,
        policy_cfg: PolicyCfg,
        world_model_cfg: WorldModelCfg,
    ):
        super().__init__()
        self.policy = policy
        self.wm = world_model
        self.assets = assets
        self.policy_cfg = policy_cfg
        self.world_model_cfg = world_model_cfg
        self.register_buffer("default_joint_pos", assets.default_joint_pos.clone())
        self.register_buffer("action_scale", assets.action_scale.clone())

    @property
    def device(self) -> torch.device:
        return self.assets.device

    @property
    def goal_frames(self) -> int:
        return GOAL_FRAMES

    @property
    def goal_feature_dim(self) -> int:
        return GOAL_FEATURE_DIM

    @property
    def goal_dim(self) -> int:
        return int(self.policy_cfg.goal_dim)

    # ------------------------------------------------------------------ construction

    @classmethod
    def random(
        cls,
        assets: G1Assets,
        *,
        small: bool = False,
        goal_dim: int = GOAL_DIM,
    ) -> "SuperMimicBundle":
        """Randomly initialized models, primarily for tests / smoke runs."""
        if small:
            policy_cfg = PolicyCfg(
                encoder_hidden=(128, 128),
                decoder_hidden=(128, 128),
                latent_dim=16,
                goal_dim=goal_dim,
            )
            wm_cfg = WorldModelCfg(hidden=(128, 128), term_hidden=(16, 16))
        else:
            policy_cfg = PolicyCfg(goal_dim=goal_dim)
            wm_cfg = WorldModelCfg()
        policy = Policy(policy_cfg).to(assets.device)
        wm = WorldModel(wm_cfg, assets.heads, assets.device)
        return cls(policy, wm, assets, policy_cfg, wm_cfg)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        assets: G1Assets | None = None,
        *,
        xml_path: str | Path | None = None,
        device: str | torch.device = "cuda",
        strict: bool = True,
    ) -> "SuperMimicBundle":
        assets = assets or build_g1_assets(xml_path, device)
        ck = torch.load(checkpoint, map_location=assets.device, weights_only=False)
        cfg = ck.get("config", {})
        if not cfg:
            raise ValueError(f"{checkpoint} has no 'config' entry; not a SuperMimic checkpoint")
        policy_cfg = _coerce_cfg(PolicyCfg, cfg["policy"])
        wm_raw = dict(cfg["world_model"])
        wm_raw.setdefault("state_spec", StateSpecCfg())
        wm_cfg = _coerce_cfg(WorldModelCfg, wm_raw)
        policy = Policy(policy_cfg).to(assets.device)
        wm = WorldModel(wm_cfg, assets.heads, assets.device)
        policy.load_state_dict(ck["policy"], strict=strict)
        wm.load_state_dict(ck["wm"], strict=strict)
        return cls(policy, wm, assets, policy_cfg, wm_cfg)

    @classmethod
    def from_configs(
        cls,
        assets: G1Assets,
        policy_cfg: PolicyCfg,
        world_model_cfg: WorldModelCfg,
    ) -> "SuperMimicBundle":
        policy = Policy(policy_cfg).to(assets.device)
        wm = WorldModel(world_model_cfg, assets.heads, assets.device)
        return cls(policy, wm, assets, policy_cfg, world_model_cfg)

    def save_checkpoint(self, path: str | Path, **extra) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "wm": self.wm.state_dict(),
                "config": {
                    "policy": asdict(self.policy_cfg),
                    "world_model": {
                        **asdict(self.world_model_cfg),
                        "state_spec": asdict(self.world_model_cfg.state_spec),
                    },
                },
                **extra,
            },
            path,
        )

    # ------------------------------------------------------------------ forward pieces

    def act(self, obs_hist_flat: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """``obs_hist_flat`` is ``(B, 930)`` in upstream term-major order."""
        return self.policy(obs_hist_flat, goal)

    def step(self, state: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        return self.wm.step(state, action, xi)

    def reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        reference: ReferenceFrame,
        bodies: dict | None = None,
    ):
        return self.assets.heads.reward(state, action, prev_action, reference, bodies)

    def termination_logits(self, state: torch.Tensor, reference: ReferenceFrame) -> torch.Tensor:
        return self.wm.termination_logits(state, reference)

    def observation(
        self,
        state: torch.Tensor,
        prev_action: torch.Tensor,
        xi: torch.Tensor,
    ) -> torch.Tensor:
        return analytic_observation(state, prev_action, xi, self.default_joint_pos)

    def flatten_history(self, hist: torch.Tensor) -> torch.Tensor:
        """``(H,B,93)`` oldest -> newest to upstream ``(B,930)`` term-major."""
        return flatten_history_term_major(hist)

    def reference_frame(self, library: MotionLibrary, frame: torch.Tensor) -> ReferenceFrame:
        ref = library.ref(frame)
        bodies = library.bodies(frame)
        return ReferenceFrame(
            bodies.pos_w,
            bodies.quat_w,
            bodies.lin_vel_w,
            bodies.ang_vel_w,
            ref.root_pos_w,
            ref.root_quat_w,
        )

    # ------------------------------------------------------------------ data helpers

    def reference_start(
        self,
        library: MotionLibrary,
        frames: torch.Tensor,
        *,
        history: int = HISTORY,
        xi: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build an observation history from reference states.

        This is used for evaluation and for smoke-training a policy from a
        motion library.  For real joint fine-tuning, replace the history with
        the replay buffer's ``hist_states`` / ``hist_actions`` (as upstream
        does) so the observation matches the demonstrator's action history.
        """
        b = frames.shape[0]
        device = frames.device
        offsets = torch.arange(history, device=device) - (history - 1)
        hist_frames = frames[:, None] + offsets[None, :]
        hist_frames = torch.clamp(hist_frames, min=0)
        ref = library.ref(hist_frames.reshape(-1))
        states = torch.cat(
            (
                ref.root_pos_w,
                ref.root_quat_w,
                ref.joint_pos,
                ref.root_lin_vel_w,
                ref.root_ang_vel_w,
                ref.joint_vel,
            ),
            dim=-1,
        ).reshape(b, history, STATE_DIM)
        actions = (ref.joint_pos - self.default_joint_pos) / self.action_scale
        actions = actions.reshape(b, history, ACTION_DIM)
        # Observation at t sees the action applied at t-1.
        prev_actions = torch.cat((torch.zeros_like(actions[:, :1]), actions[:, :-1]), dim=1)
        xi = default_xi(b, device) if xi is None else xi
        obs = []
        for t in range(history):
            obs.append(
                analytic_observation(
                    states[:, t],
                    prev_actions[:, t],
                    xi,
                    self.default_joint_pos,
                )
            )
        obs = torch.stack(obs, dim=0)  # (H,B,93)
        return {
            "state": states[:, -1],
            "hist_states": states,
            "hist_actions": prev_actions,
            "observations": obs,
            "obs_hist": self.flatten_history(obs),
            "prev_action": prev_actions[:, -1],
            "xi": xi,
        }

    def teacher_goal(self, library: MotionLibrary, frames: torch.Tensor) -> torch.Tensor:
        return library.goal_window(frames, GOAL_FRAMES)

    def teacher_states(self, library: MotionLibrary, frames: torch.Tensor, horizon: int) -> torch.Tensor:
        offs = torch.arange(1, horizon + 1, device=frames.device)
        idx = torch.clamp(frames[:, None] + offs[None, :], max=library.num_frames - 1)
        ref = library.ref(idx.reshape(-1))
        return torch.cat(
            (
                ref.root_pos_w,
                ref.root_quat_w,
                ref.joint_pos,
                ref.root_lin_vel_w,
                ref.root_ang_vel_w,
                ref.joint_vel,
            ),
            dim=-1,
        ).reshape(frames.shape[0], horizon, STATE_DIM)

    def teacher_references(
        self, library: MotionLibrary, frames: torch.Tensor, horizon: int
    ) -> list[ReferenceFrame]:
        offs = torch.arange(1, horizon + 1, device=frames.device)
        idx = torch.clamp(frames[:, None] + offs[None, :], max=library.num_frames - 1)
        return [self.reference_frame(library, idx[:, t]) for t in range(horizon)]


def configs_from_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Read only the policy/world-model configs from an upstream checkpoint."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    if not cfg:
        raise ValueError(f"{path} has no 'config' entry; not a SuperMimic checkpoint")
    policy_cfg = _coerce_cfg(PolicyCfg, cfg["policy"])
    wm_raw = dict(cfg["world_model"])
    wm_raw.setdefault("state_spec", StateSpecCfg())
    return policy_cfg, _coerce_cfg(WorldModelCfg, wm_raw)


def load_motion_library(
    npz_files: Sequence[str | Path],
    assets: G1Assets,
    *,
    compact: bool = False,
    device: str | torch.device | None = None,
) -> MotionLibrary:
    """Load full LAFAN-style npz clips, or a packed BONES compact ``.pt`` library."""
    device = device or assets.device
    files: list[str] = []
    for p in npz_files:
        path = Path(p)
        if path.is_dir():
            found = sorted(path.glob("*.npz")) + sorted(path.glob("*.pt"))
            files.extend(str(f) for f in found)
        else:
            files.append(str(path))
    if not files:
        raise ValueError("No motion files supplied")
    if compact:
        # The compact path imports mjlab's G1 XML loader lazily; point it at our asset.
        import supermimic.data.motion_convert as motion_convert

        motion_convert.load_g1_model = lambda *a, **k: load_g1_model(assets.xml_path)  # type: ignore[assignment]
    return MotionLibrary(
        files,
        TRACKED_BODIES,
        assets.default_joint_pos,
        device=device,
        compact=compact,
    )


__all__ = [
    "ACTION_DIM",
    "GOAL_DIM",
    "GOAL_FEATURE_DIM",
    "GOAL_FRAMES",
    "G1Assets",
    "HISTORY",
    "OBS_DIM",
    "STATE_DIM",
    "SuperMimicBundle",
    "XI_DIM",
    "build_g1_assets",
    "default_xi",
    "load_motion_library",
]
