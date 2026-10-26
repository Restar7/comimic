"""S3.4 / J7：想象 rollout 的起点必须由**冻结 WBC 自己**走出来，而不是参考帧本身。

原实现（``data.MotionBatchProvider`` + ``IsaacJointBundle.reference_start``）把
``state = 参考帧 f``、``history = 参考帧 f 的观测``、``a_prev = (q_ref − default_q)/scale``
直接当起点。于是：

* 起点误差恒为 0（``state`` 就是 ``teacher_states[:, 0]``），H 步内 WM 几乎不偏 →
  "token 好不好"在回报上没有区别（原文档 D1 12%、D2 存活≈1 的真实根因）；
* ``a_prev`` 是**换算**出来的，依赖 ``COMIMIC_ACTION_SCALE`` 这批口径 bug（J1''）。

本模块按重设计 §3.5 的做法重建起点：

1. 参考帧 f 往前退 K 帧到 f₀（夹到 clip 起点）；
2. 对 f₀ 的参考状态加扰动（root xy ±5 cm、yaw ±0.1 rad、q ±0.05 rad、qd ±0.3 rad/s）；
3. **no_grad** 下让冻结的 WBC 拿 f₀..f₀+K−1 的**端口 GT token** 在冻结 WM 里走 K 步；
4. 返回 ``state / history / prev_action / references(从 f 开始)`` —— 全部由 WBC 自己产生，
   与 ``action_scale`` 的换算口径无关。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

HISTORY = 10
NUM_JOINTS = 29
XI_DIM = 33


def _flat_tuple(flat: torch.Tensor):
    """扁平 state (…,71) → trackerlab 状态元组 (p, R, v, w, q, qd)。"""
    from .isaac_joint import flat_state_tuple  # noqa: PLC0415

    return flat_state_tuple(flat)


#: 默认扰动幅度（重设计 §3.5）
PERTURB_DEFAULTS = {
    "xy": 0.05,      # m
    "yaw": 0.10,     # rad
    "dq": 0.05,      # rad
    "dqd": 0.30,     # rad/s
}


@dataclass
class BurnInResult:
    frames: torch.Tensor           # (B,) 参考帧 f（rollout 的 reward 参考从 f+1 开始）
    f0: torch.Tensor               # (B,) burn-in 起始帧
    state: torch.Tensor            # (B,71) K 步之后的状态（= 帧 f 附近）
    history: torch.Tensor          # (H_in,B,W) oldest→newest，rollout 口径
    obs_hist: torch.Tensor         # (B, H_in*W) term-major 平铺
    prev_action: torch.Tensor      # (B,29)
    xi: torch.Tensor               # (B,33)
    references: list = field(default_factory=list)   # H 个 FrameRef，从 f+1 开始
    teacher_states: torch.Tensor | None = None       # (B,H,71)
    burnin_reward: torch.Tensor | None = None        # (B,) burn-in 段平均奖励（诊断）
    burnin_survival: torch.Tensor | None = None      # (B,) burn-in 段末存活概率（诊断）
    hist_states: torch.Tensor | None = None          # (B,K+1,71) burn-in 轨迹（诊断）


def _rand(batch: int, device, generator, scale: float) -> torch.Tensor:
    """U(-scale, scale) ``(B,)``；``generator=None`` 时用全局 RNG。

    生成器与目标张量可能不同设备（provider 的种子生成器在 CPU 上，为了让同一 seed 在
    任意设备上给出同一组扰动），所以先按生成器设备采样再搬到目标设备。
    """
    if scale <= 0:
        return torch.zeros(batch, device=device)
    dev = torch.device(device)
    if generator is not None and torch.device(generator.device).type != dev.type:
        u = torch.rand(batch, device=generator.device, generator=generator).to(dev)
    else:
        u = torch.rand(batch, device=dev, generator=generator)
    return (u * 2.0 - 1.0) * float(scale)


def perturb_state(
    state: torch.Tensor,
    *,
    xy: float = PERTURB_DEFAULTS["xy"],
    yaw: float = PERTURB_DEFAULTS["yaw"],
    dq: float = PERTURB_DEFAULTS["dq"],
    dqd: float = PERTURB_DEFAULTS["dqd"],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """给 flat state ``(B,71)`` 加 §3.5 规定的那组扰动（不改输入）。"""
    if int(state.shape[-1]) != 71:
        raise ValueError(f"expected flat state (B,71), got {tuple(state.shape)}")
    out = state.clone()
    b, device = out.shape[0], out.device
    if xy > 0:
        out[:, 0:2] += _rand(b, device, generator, xy)[:, None]
    if yaw > 0:
        from trackerLab.kinematics.supermimic.rotations import (  # noqa: PLC0415
            axis_angle_to_mat,
            mat_to_quat,
            quat_to_mat,
        )

        ang = _rand(b, device, generator, yaw)
        axis = torch.zeros(b, 3, device=device)
        axis[:, 2] = 1.0
        R = axis_angle_to_mat(axis, ang) @ quat_to_mat(out[:, 3:7])
        out[:, 3:7] = mat_to_quat(R)
    if dq > 0:
        out[:, 7:36] += _rand(b, device, generator, dq)[:, None]
    if dqd > 0:
        out[:, 42:71] += _rand(b, device, generator, dqd)[:, None]
    return out


@torch.no_grad()
def sample_burnin(
    bundle: Any,
    library: Any,
    frames: torch.Tensor,
    *,
    port_tokens: Callable[[torch.Tensor], torch.Tensor],
    token_fn: Callable[..., torch.Tensor] | None = None,
    recorded_states: torch.Tensor | None = None,
    horizon: int,
    k: int = 10,
    xi: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    perturb: bool = True,
    xy: float = PERTURB_DEFAULTS["xy"],
    yaw: float = PERTURB_DEFAULTS["yaw"],
    dq: float = PERTURB_DEFAULTS["dq"],
    dqd: float = PERTURB_DEFAULTS["dqd"],
) -> BurnInResult:
    """用**冻结 WBC + 冻结 WM + 端口 GT token** 生成 rollout 起点（重设计 §3.5 第 1–3 步）。

    ``port_tokens(frames) -> (B,64)``：侧车 token（旧端口空间）。
    ``token_fn(frames, state=...) -> (B,64)``：**环内** token（hold builder + 当时仿真机器人的
    anchor，交接文档 §5.3）；给了它就用它，``port_tokens`` 变成回退。``bundle`` 只需要 ``reference_start`` /
    ``flatten_history`` / ``observation`` / ``act`` / ``step`` / ``reward`` /
    ``termination_logits`` —— ``IsaacJointBundle`` 全部满足。
    """
    if k < 0:
        raise ValueError("burn-in steps must be >= 0")
    motion = library
    b = int(frames.shape[0])
    device = frames.device
    if xi is None:
        from .isaac_joint import default_xi  # noqa: PLC0415

        xi = default_xi(b, device)

    state_device = frames.device
    # 1) f 往前退 K 帧（夹到 clip 起点）
    clip_first = motion.clip_start[motion.clip_id[frames]].to(device)
    f0 = torch.clamp(frames - int(k), min=clip_first)

    # 2) 参考状态 + 参考历史（在 f0 处），然后加扰动
    if recorded_states is not None:
        # ★ v3：起点用**真录制机器人状态**（部署口径），历史也用真状态 + 命令推出的 a_prev 重建
        clip_first64 = motion.clip_start[motion.clip_id[f0]].to(f0.device)
        hrows = torch.clamp(
            f0[:, None] + (torch.arange(HISTORY, device=f0.device) - (HISTORY - 1)),
            min=clip_first64[:, None],
        )
        st_hist = recorded_states[hrows.reshape(-1)].float().to(state_device).reshape(b, HISTORY, -1)
        q_cmd = motion.ref[hrows.reshape(-1).to(motion.ref.device), 13:13 + NUM_JOINTS].float()
        a_hist = (q_cmd - bundle.default_joint_pos) / bundle.action_scale
        xi_rep = xi[:, None].expand(b, HISTORY, XI_DIM).reshape(-1, XI_DIM)
        obs = bundle.mdp.obs(
            _flat_tuple(st_hist.reshape(-1, 71)), a_hist, xi_rep, hrows.reshape(-1)
        ).reshape(b, HISTORY, -1)
        state = recorded_states[f0].float().to(state_device)
        history = obs.permute(1, 0, 2).contiguous()          # (H_in,B,W)
        a_prev = a_hist.reshape(b, HISTORY, -1)[:, -1]
    else:
        start = bundle.reference_start(motion, f0)
        state = start["state"]
        history = start["observations"].clone()        # (H_in,B,W)
        a_prev = start["prev_action"]
    if perturb:
        state = perturb_state(state, xy=xy, yaw=yaw, dq=dq, dqd=dqd, generator=generator)

    refs_k = bundle.teacher_references(motion, f0, int(k)) if k > 0 else []
    hist_states = [state]
    rewards, survival = [], torch.ones(b, device=device)
    for t in range(int(k)):
        ref = refs_k[t]
        if token_fn is not None:
            # B 模式：命令取该帧参考、anchor 取**当前仿真机器人 root**（部署/录制口径）
            cond = token_fn(f0 + t, state=state)
        else:
            cond = port_tokens(f0 + t)
        obs_flat = bundle.flatten_history(history)
        action = bundle.act(obs_flat, cond, ref)
        nxt = bundle.step(state, action, xi)
        terms = bundle.reward(nxt, action, a_prev, ref)
        prob = torch.sigmoid(bundle.termination_logits(nxt, ref))
        rewards.append(terms["total"])
        survival = survival * (1.0 - prob)
        obs = bundle.observation(nxt, action, xi, ref)
        history = torch.cat((history[1:], obs.unsqueeze(0)), dim=0)
        a_prev = action
        state = nxt
        hist_states.append(state)

    refs = bundle.teacher_references(motion, frames, int(horizon))
    teacher_states = bundle.teacher_states(motion, frames, int(horizon))
    obs_hist = bundle.flatten_history(history)
    return BurnInResult(
        frames=frames,
        f0=f0,
        state=state,
        history=history,
        obs_hist=obs_hist,
        prev_action=a_prev,
        xi=xi,
        references=refs,
        teacher_states=teacher_states,
        hist_states=torch.stack(hist_states, dim=1) if k > 0 else state[:, None],
        burnin_reward=torch.stack(rewards, dim=0).mean(0) if rewards else None,
        burnin_survival=survival,
    )


__all__ = ["PERTURB_DEFAULTS", "BurnInResult", "perturb_state", "sample_burnin"]
