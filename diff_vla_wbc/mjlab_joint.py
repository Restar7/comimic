"""mjlab / MuJoCo 后端的联合微调 bundle：**观测 93 维**，接口与 :mod:`isaac_joint` 对齐。

把 upstream ``supermimic`` 的 ``Policy`` / ``WorldModel`` / ``AnalyticHeads`` 包成
``diff_vla_wbc.train_v2`` / ``v3_step`` / ``rollout`` / ``burnin`` / ``data`` 需要的鸭子类型，
使那四个模块**一行都不改**就能跑在 MuJoCo 侧（它们只通过这里的方法访问物理/解析件）。

与 Isaac 版的差异（全部有据可查）：

| 项 | Isaac（trackerlab） | mjlab（本模块） |
|---|---|---|
| 观测宽度 | 99（``relative_ori99``） | **93**（`OBS_TERMS`，paper 口径） |
| state 元组 | trackerlab 顺序 | `[p(3), quat(4), q(29), v_w(3), w_b(3), qd(29)]`（MuJoCo `qpos|qvel`） |
| 库里的角速度 | 世界系，bridge 转换 | **世界系 → 体系**，本模块在 `_ref71_body` 里转（§1.3） |
| 动作语义 | `(q-q0-bias)/scale` | `q_target = q0 + scale·a`（upstream `q_target`） |
| 终止 | trackerlab 软终止概率 | `wm.termination_logits`（**原生 logits**，rollout 再 sigmoid） |
| FK / 奖励 | trackerlab 解析件 | `AnalyticHeads`（mjlab 可微 FK，14 tracked bodies） |

红线：
* ``OBS_DIM`` 固定 93，``term_dims`` 固定 ``(29,29,3,3,29)`` —— 改这两个数字等于换模型；
* 状态元组的 ``[39:42]`` **必须是体系角速度**（仿真 `qvel` / `state_spec` / `base_ang_vel` 都是体系）。
  运动库存的是世界系，所以只能在这里转换，不能指望 rollout（它没有 bridge）。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

from .compat import ensure_supermimic_importable

ACTION_DIM = 29
HISTORY = 10
STATE_DIM = 71
NUM_JOINTS = 29
XI_DIM = 33

#: 每帧观测宽度（`world_model/heads.py:21` 的 `OBS_DIM`）。**禁止**改成 99。
OBS_DIM = 93
#: term-major 历史布局（`world_model/heads.py:22` 的 `OBS_TERMS`）。
TERM_DIMS: tuple[int, ...] = (29, 29, 3, 3, 29)

#: 被跟踪的 14 个 body（抄自 `supermimic/envs/tracking_env_cfg.py:49-64`）。
#: 单独放这里是为了让**系统 python**（无 mjlab）也能构造 MotionLibrary，而不触发
#: `tracking_env_cfg` 里的 `mjlab.asset_zoo` 导入。
TRACKED_BODIES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
ANCHOR_BODY = "torso_link"
ROOT_BODY = "pelvis"

GOAL_FRAMES = 10
GOAL_FEATURE_DIM = 68


# ---------------------------------------------------------------------------- imports
def _sm():
    """Import the upstream (pure-torch) modules once ``supermimic`` is importable."""
    ensure_supermimic_importable(lightweight=True)
    from supermimic.envs.mdp.motion_library import MotionLibrary  # noqa: PLC0415
    from supermimic.kinematics.rotations import quat_apply_inverse  # noqa: PLC0415
    from supermimic.world_model.heads import (  # noqa: PLC0415
        ReferenceFrame,
        analytic_observation,
        flatten_history_term_major,
    )

    return {
        "MotionLibrary": MotionLibrary,
        "quat_apply_inverse": quat_apply_inverse,
        "ReferenceFrame": ReferenceFrame,
        "analytic_observation": analytic_observation,
        "flatten_history_term_major": flatten_history_term_major,
    }


# ---------------------------------------------------------------------------- helpers
def default_xi(batch: int, device: Any) -> torch.Tensor:
    """``[friction=0.75, encoder bias 29, CoM 3]``（与 ``isaac_joint.default_xi`` 同值）。"""
    xi = torch.zeros(int(batch), XI_DIM, device=device)
    xi[:, 0] = 0.75
    return xi


def is_mjlab_checkpoint(path: str | Path) -> bool:
    """B1（`sm-train` 产物）或 B2（联合微调产物）：有 ``policy``+``wm``，且**没有** ``pi``。"""
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    return ("policy" in ck and "wm" in ck) and ("pi" not in ck)


def _ref71_body(library, frames: torch.Tensor) -> torch.Tensor:
    """运动库 71 维参考 → **MuJoCo 状态语义**（``[39:42]`` 世界系 → 体系）。

    上游 ``SuperMimicBundle.reference_start`` / ``teacher_states`` 没做这一步（它自己的注释写了
    "仅供 evaluation/smoke"）。用未转换的值会让 obs 的 ``base_ang_vel`` 与训练分布错位：
    loss 照常降、闭环整体漂。
    """
    api = _sm()
    r = library.ref71(frames).float()
    w_b = api["quat_apply_inverse"](r[:, 3:7], r[:, 39:42])
    return torch.cat((r[:, :39], w_b, r[:, 42:71]), dim=-1)


class MDPShim:
    """``burnin`` 只用到 ``bundle.mdp.obs(...)``，这里给最小面（93 维，不需要 frame 索引）。"""

    __slots__ = ("_b",)

    def __init__(self, bundle: "MjlabJointBundle"):
        self._b = bundle

    def obs(self, states, actions, xi, frames=None):  # noqa: ARG002 - signature parity
        if isinstance(states, (tuple, list)):
            states = self._b.state_from_tuple(states)
        return self._b.observation(states, actions, xi)


# ---------------------------------------------------------------------------- bundle
class MjlabJointBundle:
    """mjlab 侧的执行器 + 可微动力学 + 解析件（策略/世界模型默认冻结）。"""

    def __init__(
        self,
        base,
        library=None,
        *,
        token_dim: int = 64,
        source_checkpoint: str = "",
        source_iteration: int = -1,
        mdp=None,
    ):
        self.base = base                     # diff_vla_wbc.supermimic_backend.SuperMimicBundle
        self.policy = base.policy
        self.wm = base.wm
        self.assets = base.assets
        self.policy_cfg = base.policy_cfg
        self.world_model_cfg = base.world_model_cfg
        self.library = library
        self.motion = library                # 兼容 isaac_joint 的属性名（v3_step 用）
        self.token_dim = int(getattr(base.policy, "token_dim", 0) or 0)
        if self.token_dim != int(token_dim):
            raise ValueError(f"checkpoint token_dim={self.token_dim} != requested {token_dim}")
        self.source_checkpoint = str(source_checkpoint)
        self.source_iteration = int(source_iteration)
        self.obs_width = OBS_DIM
        self.term_dims = TERM_DIMS
        self.mdp = mdp if mdp is not None else MDPShim(self)
        self._api = None

    # ---------------------------------------------------------------- basics
    @property
    def device(self) -> torch.device:
        return self.assets.device

    @property
    def default_joint_pos(self) -> torch.Tensor:
        return self.assets.default_joint_pos

    @property
    def action_scale(self) -> torch.Tensor:
        return self.assets.action_scale

    @property
    def num_joints(self) -> int:
        return NUM_JOINTS

    @property
    def goal_dim(self) -> int:
        return int(getattr(self.policy_cfg, "goal_dim", 0) or 0)

    @property
    def max_step_reward(self) -> float:
        return float(self.assets.heads.reward_cfg.max_step_reward)      # = 0.1（dt*5）

    @property
    def spec(self) -> dict:
        """审计用快照（``train_v2`` 存 ckpt 时会读；Isaac 版是 robot_spec_snapshot）。"""
        return {
            "backend": "mjlab",
            "robot": "g1",
            "num_joints": NUM_JOINTS,
            "obs_dim": OBS_DIM,
            "term_dims": list(TERM_DIMS),
            "token_dim": self.token_dim,
            "history": HISTORY,
            "action_scale": [float(v) for v in self.action_scale.detach().cpu().flatten()[:4]],
            "source_checkpoint": self.source_checkpoint,
        }

    def to(self, device):
        self.base.to(device)
        self.assets.device = torch.device(device)
        return self

    def train(self):
        self.policy.train()
        return self

    def eval(self):
        self.policy.eval()
        self.wm.eval()
        return self

    def _a(self) -> dict:
        if self._api is None:
            self._api = _sm()
        return self._api

    # ---------------------------------------------------------------- learned layer
    def act(self, obs_hist_flat: torch.Tensor, cond: torch.Tensor, ref: Any | None = None):
        """``obs_hist_flat`` 必须是 ``(B, 930)`` term-major；``cond`` 是 64 维 token。"""
        if obs_hist_flat.shape[-1] != HISTORY * OBS_DIM:
            raise ValueError(
                f"obs_hist must be (B,{HISTORY * OBS_DIM}) — 93 per frame, term-major; "
                f"got {tuple(obs_hist_flat.shape)}"
            )
        return self.policy(obs_hist_flat, cond, interface="token")

    def step(self, state: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        return self.wm.step(state, action, xi)

    # ---------------------------------------------------------------- analytic layer
    def observation(self, next_state: torch.Tensor, action: torch.Tensor, xi: torch.Tensor,
                    ref: Any | None = None) -> torch.Tensor:
        """93 维本体感觉观测（无噪声）；``ref`` 只为与 Isaac 的参数表对齐，这里不用。"""
        return self._a()["analytic_observation"](
            next_state, action, xi, self.assets.default_joint_pos
        )

    def flatten_history(self, history: torch.Tensor) -> torch.Tensor:
        """``(H,B,93)`` oldest→newest → ``(B,930)`` term-major。"""
        return self._a()["flatten_history_term_major"](history)

    def reward(self, state, action, prev_action, reference, bodies=None):
        return self.assets.heads.reward(state, action, prev_action, reference, bodies)

    def termination_logits(self, state, reference):
        return self.wm.termination_logits(state, reference)

    def robot_bodies(self, state: torch.Tensor, bodies=None):
        return self.assets.heads.robot_bodies(state)

    # ---------------------------------------------------------------- data helpers
    def default_xi(self, batch: int, device) -> torch.Tensor:
        """与 ``isaac_joint.default_xi`` 同值（``[0]=0.75`` 摩擦 + 29 偏置 + 3 CoM）。"""
        return default_xi(batch, device)

    def clip_start_of(self, frames: torch.Tensor) -> torch.Tensor:
        lib = self.library
        return lib.clip_start[lib.frame_clip[frames]].to(frames.device)

    def clip_end_of(self, frames: torch.Tensor) -> torch.Tensor:
        lib = self.library
        return lib.clip_last[lib.frame_clip[frames]].to(frames.device)

    def ref_state_at(self, frames: torch.Tensor) -> torch.Tensor:
        """库帧 → 扁平 71 状态（**体系角速度**）。v3 的 ``_ref_states`` 用它。"""
        return _ref71_body(self.library, frames)

    def ref_joint_pos(self, frames: torch.Tensor) -> torch.Tensor:
        """库里的 29 维关节位置（**MuJoCo 关节序**，索引 7:36）。"""
        return self.library.ref71(frames).float()[:, 7:36]

    def state_from_tuple(self, state_tuple) -> torch.Tensor:
        """``(p, R, v, w, q, qd)`` → 扁平 71（本后端的顺序）。"""
        p, R, v, w, q, qd = state_tuple
        return torch.cat((p, R, q, v, w, qd), dim=-1)

    def obs_from_states(self, states_flat: torch.Tensor, actions: torch.Tensor,
                        xi: torch.Tensor, frames: torch.Tensor | None = None) -> torch.Tensor:
        """给 ``burnin`` 用：直接从扁平状态造 93 维观测（``frames`` 忽略）。"""
        return self.observation(states_flat, actions, xi)

    def reference_start(self, library, frames: torch.Tensor, *, history: int = HISTORY, xi=None):
        """``frames`` 结尾的 ``history`` 帧观测历史（从参考运动本身推，与 Isaac 版同构）。"""
        if library is None:
            library = self.library
        b = int(frames.shape[0])
        device = frames.device
        offsets = torch.arange(history, device=device) - (history - 1)
        clip_first = library.clip_start[library.frame_clip[frames]].to(device)
        hf = (frames[:, None] + offsets[None, :]).clamp(min=clip_first[:, None])       # (B,H)
        xi = default_xi(b, device) if xi is None else xi
        rows = _ref71_body(library, hf.reshape(-1)).reshape(b, history, STATE_DIM)      # (B,H,71)
        q_ref = rows[..., 7:36]
        a_ref = (q_ref - self.assets.default_joint_pos) / self.assets.action_scale      # (B,H,29)
        a_prev = torch.cat((torch.zeros_like(a_ref[:, :1]), a_ref[:, :-1]), dim=1)
        xi_hist = xi[:, None].expand(b, history, XI_DIM)
        obs = self.observation(
            rows.reshape(-1, STATE_DIM), a_prev.reshape(-1, ACTION_DIM), xi_hist.reshape(-1, XI_DIM)
        ).reshape(b, history, OBS_DIM)
        obs_out = obs.permute(1, 0, 2).contiguous()          # (H,B,93) — rollout 顺序
        return {
            "state": rows[:, -1],
            "hist_states": rows,
            "hist_actions": a_prev,
            "observations": obs_out,
            "obs_hist": self.flatten_history(obs_out),
            "prev_action": a_prev[:, -1],
            "xi": xi,
            "frames": frames,
        }

    def reference_frame(self, library, frame: torch.Tensor):
        """``ReferenceFrame``（奖励/终止用）：与 `IsaacJointBundle.reference_frame` 同形。"""
        lib = library if library is not None else self.library
        r = lib.ref(frame)
        b = lib.bodies(frame)
        return self._a()["ReferenceFrame"](
            b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w, r.root_pos_w, r.root_quat_w
        )

    def teacher_states(self, library, frames: torch.Tensor, horizon: int) -> torch.Tensor:
        if library is None:
            library = self.library
        last = library.clip_last[library.frame_clip[frames]].to(frames.device)
        offs = torch.arange(1, horizon + 1, device=frames.device)
        idx = torch.minimum(frames[:, None] + offs[None, :], last[:, None])
        return _ref71_body(library, idx.reshape(-1)).reshape(frames.shape[0], horizon, STATE_DIM)

    def teacher_references(self, library, frames: torch.Tensor, horizon: int):
        if library is None:
            library = self.library
        ref_cls = self._a()["ReferenceFrame"]
        last = library.clip_last[library.frame_clip[frames]].to(frames.device)
        offs = torch.arange(1, horizon + 1, device=frames.device)
        idx = torch.minimum(frames[:, None] + offs[None, :], last[:, None])
        out = []
        for t in range(horizon):
            f = idx[:, t]
            r = library.ref(f)
            b = library.bodies(f)
            out.append(
                ref_cls(b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w, r.root_pos_w, r.root_quat_w)
            )
        return out

    def teacher_goal(self, library, frames: torch.Tensor):
        """token 路由下没有 680 维目标（`MotionBatchProvider` 会跳过这个调用）。"""
        return None

    # ---------------------------------------------------------------- checkpoints
    def save_checkpoint(self, path, **extra) -> None:
        """B2 布局（与 ``runs/joint_*.pt`` 同键），额外记录后端与结构开关。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        wm_cfg = asdict(self.world_model_cfg)
        wm_cfg["state_spec"] = asdict(self.world_model_cfg.state_spec)
        torch.save(
            {
                "policy": {k: v.detach().cpu() for k, v in self.policy.state_dict().items()},
                "wm": {k: v.detach().cpu() for k, v in self.wm.state_dict().items()},
                # 旧 JointTrainer.save 的键（`JointTrainer.load_bundle_from_checkpoint` 与
                # `scripts/closed_loop_vla_supermimic.py` 读这两个；没有就加载不了 E1 的闭合）
                "policy_config": asdict(self.policy_cfg),
                "world_model_config": wm_cfg,
                "config": {"policy": asdict(self.policy_cfg), "world_model": wm_cfg},
                "backend": "mjlab",
                "obs_dim": OBS_DIM,
                "term_dims": list(TERM_DIMS),
                "source_checkpoint": self.source_checkpoint,
                "source_iteration": self.source_iteration,
                "wm_integration": (
                    f"explicit_dt{float(self.world_model_cfg.state_spec.dt):g}"
                    if bool(getattr(self.world_model_cfg.state_spec, "explicit_integration", True))
                    else "residual"
                ),
                **extra,
            },
            path,
        )


# ---------------------------------------------------------------------------- factory
def load_mjlab_bundle(
    ckpt: str | Path,
    *,
    device: str | torch.device = "cuda",
    npz: Sequence[str | Path] | None = None,
    packed: str | Path | None = None,
    token_dim: int = 64,
    xml_path: str | Path | None = None,
    freeze: bool = True,
) -> MjlabJointBundle:
    """加载 mjlab 权重（B1/B2），可选挂一份参考库。

    * ``npz``：**full** 库（318 的 npz，含 ``body_*``，参考帧/奖励直接用）
    * ``packed``：**compact** 库（``build_library.py`` 产物，body 靠可微 FK 现算）

    两者给一个即可；J1 用 ``npz``（参考帧要 ``bodies``，full 更快更稳）。
    """
    if not is_mjlab_checkpoint(ckpt):
        raise SystemExit(
            f"{ckpt}: 不是 mjlab 训练 ckpt（需要 policy+wm 且没有 pi）。"
            "Isaac/trackerlab 的 {pi,wm} 布局请走 load_isaac_bundle。"
        )
    from .supermimic_backend import SuperMimicBundle, build_g1_assets, load_motion_library

    assets = build_g1_assets(xml_path, device)
    base = SuperMimicBundle.from_checkpoint(ckpt, assets, device=device)
    library = None
    if npz:
        library = load_motion_library(list(npz), assets, compact=False, device=device)
    elif packed:
        library = load_motion_library([packed], assets, compact=True, device=device)
    bundle = MjlabJointBundle(
        base,
        library,
        token_dim=token_dim,
        source_checkpoint=str(ckpt),
        source_iteration=int(torch.load(str(ckpt), map_location="cpu", weights_only=False).get("iteration", -1)),
    )
    if freeze:
        # 联合微调只训 VLA head：执行器与动力学在这里就冻上（训练器还会再调一次
        # `_freeze_executor`，两处都不改语义，只是让"加载即可用"这件事成立）。
        for p in bundle.policy.parameters():
            p.requires_grad_(False)
        for p in bundle.wm.parameters():
            p.requires_grad_(False)
        bundle.policy.eval()
        bundle.wm.eval()
    return bundle


__all__ = [
    "ANCHOR_BODY",
    "ROOT_BODY",
    "TRACKED_BODIES",
    "ACTION_DIM",
    "HISTORY",
    "MDPShim",
    "MjlabJointBundle",
    "NUM_JOINTS",
    "OBS_DIM",
    "STATE_DIM",
    "TERM_DIMS",
    "XI_DIM",
    "default_xi",
    "is_mjlab_checkpoint",
    "load_mjlab_bundle",
]
