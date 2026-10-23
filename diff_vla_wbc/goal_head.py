"""VLA -> SuperMimic 的直连通道：直接回归 ``[10, 68]`` 物理 goal，不再经过 token/adapter。

为什么不再走 token
------------------
SuperMimic 的策略吃的是**物理参考量**（``motion_library.py:66``）::

    g = [ q_ref - q_default (29, rad),  q̇ (29, rad/s),  projected_gravity (3, unit),
          ω_body (3, rad/s),  v_yaw (3, m/s),  root_height (1, m) ]        # 10 帧拼成 680

而 GR00T 的 64 维动作 token 是无单位隐空间，仓库里又没有 GR00T 预训练权重
（action head 随机初始化），所以 ``token -> 68 维`` 的 adapter 只能在任务损失下
学一个不可辨识的映射。SONIC 之所以能用 token，是因为它的 64 维是 motion tokenizer
在动作数据上训出来的、decoder 有重建监督——我们没有那个锚点。

因此本模块让 VLA **直接预测 goal**：目标本身就是从动作数据算出来的，天生可监督，
且和 SuperMimic 的输入口径逐项对齐，中间不存在任何学习式转换。

输出布局（逐项、按帧，行主序展平成 680）::

    frame = [ joint_pos(29) | joint_vel(29) | gravity(3) | root_ang_vel(3)
              | root_lin_vel(3) | root_height(1) ]

归一化只发生在 head 内部（``GoalSpec``），**输出的永远是物理量**，与
``MotionLibrary.goal_window`` 和 ``SupermimicBundle.act`` 的单位一致。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import GOAL_FEATURE_DIM, GOAL_FRAMES, GOAL_SLICES

GOAL_DIM = GOAL_FRAMES * GOAL_FEATURE_DIM  # 680

#: 各物理项的典型量级，只用于 head 内部的输入输出条件化（不改变对外单位）。
#: 顺序与 ``GOAL_SLICES`` 一致：关节(rad)、关节速度(rad/s)、重力(单位)、
#: 机体系角速度(rad/s)、航向系线速度(m/s)、根高度(m)。
DEFAULT_TERM_SCALES: tuple[float, ...] = (0.5, 5.0, 1.0, 5.0, 1.0, 0.1)


@dataclass
class GoalSpec:
    frames: int = GOAL_FRAMES
    frame_dim: int = GOAL_FEATURE_DIM
    term_scales: tuple[float, ...] = DEFAULT_TERM_SCALES

    def __post_init__(self) -> None:
        sizes = tuple(s.stop - s.start for s in GOAL_SLICES.values())
        if self.frame_dim != GOAL_FEATURE_DIM:
            raise ValueError(f"frame_dim must be {GOAL_FEATURE_DIM}, got {self.frame_dim}")
        if len(self.term_scales) != len(sizes):
            raise ValueError(f"term_scales needs {len(sizes)} entries, got {len(self.term_scales)}")

    @property
    def dim(self) -> int:
        return self.frames * self.frame_dim

    def scale_vector(self, *, device=None, dtype=torch.float32) -> torch.Tensor:
        """``[68]`` 每项尺度，按帧重复成 ``[680]``（与 goal 展平顺序一致）。"""
        per_frame = torch.cat(
            [
                torch.full((s.stop - s.start,), float(v), device=device, dtype=dtype)
                for s, v in zip(GOAL_SLICES.values(), self.term_scales)
            ]
        )
        return per_frame.repeat(self.frames)

    def normalize(self, goal: torch.Tensor) -> torch.Tensor:
        """物理量 -> 条件化空间（除以尺度）。"""
        scale = self.scale_vector(device=goal.device, dtype=goal.dtype)
        return goal.reshape(goal.shape[0], -1) / scale

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        """条件化空间 -> 物理量（乘以尺度），返回 ``[B, frames, 68]``。"""
        scale = self.scale_vector(device=normalized.device, dtype=normalized.dtype)
        out = normalized.reshape(normalized.shape[0], -1) * scale
        return out.view(normalized.shape[0], self.frames, self.frame_dim)


def flatten_goal(goal: torch.Tensor) -> torch.Tensor:
    """``[B, 10, 68]`` -> ``[B, 680]``，行主序，直接喂 ``SupermimicBundle.act``。"""
    if goal.dim() == 3 and goal.shape[1:] != (GOAL_FRAMES, GOAL_FEATURE_DIM):
        raise ValueError(f"expected [B, {GOAL_FRAMES}, {GOAL_FEATURE_DIM}], got {tuple(goal.shape)}")
    return goal.reshape(goal.shape[0], -1).contiguous()


def unflatten_goal(goal_flat: torch.Tensor) -> torch.Tensor:
    if goal_flat.shape[-1] != GOAL_DIM:
        raise ValueError(f"expected last dim {GOAL_DIM}, got {goal_flat.shape[-1]}")
    return goal_flat.reshape(goal_flat.shape[0], GOAL_FRAMES, GOAL_FEATURE_DIM)


@dataclass
class GoalHeadCfg:
    in_dim: int = 2048  # VLM backbone width
    hidden: int = 512
    layers: int = 2
    dropout: float = 0.0
    frames: int = GOAL_FRAMES
    frame_dim: int = GOAL_FEATURE_DIM
    term_scales: tuple[float, ...] = DEFAULT_TERM_SCALES


class GoalHead(nn.Module):
    """Masked-mean pool over VLM tokens -> ``[B, 10, 68]`` 物理 goal（单位同 SuperMimic）。"""

    def __init__(self, cfg: GoalHeadCfg | None = None):
        super().__init__()
        self.cfg = cfg or GoalHeadCfg()
        c = self.cfg
        self.spec = GoalSpec(c.frames, c.frame_dim, c.term_scales)
        self.norm = nn.LayerNorm(c.in_dim)
        layers: list[nn.Module] = []
        last = c.in_dim
        for _ in range(max(1, c.layers)):
            layers += [nn.Linear(last, c.hidden), nn.GELU(), nn.Dropout(c.dropout)]
            last = c.hidden
        layers.append(nn.Linear(last, self.spec.dim))
        self.net = nn.Sequential(*layers)
        # 零初始化最后一层 + 物理合理的先验（直立重力、~0.76 m 高度），
        # 与 adapter 的做法一致：训练初期输出接近静止姿态而不是随机抖。
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("prior", _goal_prior(self.spec), persistent=True)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError(f"expected features [B, T, D], got {tuple(features.shape)}")
        h = self.norm(features)
        if mask is not None:
            m = mask.to(dtype=h.dtype).unsqueeze(-1)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        else:
            pooled = h.mean(dim=1)
        delta = self.net(pooled)
        normalized = self.spec.normalize(self.prior.unsqueeze(0).expand(pooled.shape[0], -1, -1)) + delta
        return self.spec.denormalize(normalized)

    def loss(self, pred: torch.Tensor, target: torch.Tensor, *, beta: float = 0.05) -> torch.Tensor:
        """在条件化空间做 Smooth-L1（各物理项量级已对齐，训练更稳）。"""
        if pred.shape != target.shape:
            raise ValueError(f"pred {tuple(pred.shape)} != target {tuple(target.shape)}")
        return F.smooth_l1_loss(
            self.spec.normalize(pred), self.spec.normalize(target), beta=beta, reduction="mean"
        )


def _goal_prior(spec: GoalSpec) -> torch.Tensor:
    """中性目标：零关节偏移/速度、直立重力 ``[0, 0, -1]``、0.76 m 根高度。"""
    goal = torch.zeros(spec.frames, spec.frame_dim)
    goal[:, GOAL_SLICES["gravity"]] = torch.tensor([0.0, 0.0, -1.0])
    goal[:, GOAL_SLICES["root_height"]] = 0.76
    return goal


def attach_goal_head(model: nn.Module, cfg: GoalHeadCfg | None = None) -> GoalHead:
    head = GoalHead(cfg)
    model.goal_head = head
    return head


__all__ = [
    "GOAL_DIM",
    "GOAL_FRAMES",
    "GOAL_FEATURE_DIM",
    "GOAL_SLICES",
    "GoalHead",
    "GoalHeadCfg",
    "GoalSpec",
    "attach_goal_head",
    "flatten_goal",
    "unflatten_goal",
]
