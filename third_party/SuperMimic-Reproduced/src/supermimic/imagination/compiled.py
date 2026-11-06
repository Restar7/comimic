"""One imagined step as a single compiled function (plan 3.6: launch-overhead control).

``ImagStep`` bundles world-model dynamics, analytic FK reward, termination logits and the
next noise-free observation into one call so ``torch.compile`` can fuse the hundreds of
small kernels of the per-body FK loop. Reference quantities are passed as plain tensors
(already gathered from the motion library) so the compiled graph has static shapes.
"""

from __future__ import annotations

import torch

from supermimic.world_model.heads import ReferenceFrame, analytic_observation
from supermimic.world_model.world_model import WorldModel

# 必须是 ``tracking_reward_terms`` 返回的 key 全集（顺序无所谓，但**不能缺**）。
# 2026-10-30 事故：奖励把 `motion_global_root_ori` 拆成 tilt/yaw/yaw_rate 并加了 4 个 hinge，
# 这份写死的名单没跟着改 ⇒ 重新训 WBC 时 `terms[k]` 直接 KeyError，训练一启动就崩。
# 以后改奖励项**必须**同步这个常量（已在 build 时校验名字集合，见 _self_check）。
REWARD_TERM_NAMES = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_root_tilt",
    "motion_root_yaw",
    "motion_root_yaw_rate",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "motion_root_pos_tail",
    "motion_root_ori_tail",
    "motion_body_pos_tail_legs",
    "motion_body_pos_tail_arms",
    "motion_tilt_pen",
    "motion_height_pen",
    "action_rate_l2",
    "joint_limit",
)


class ImagStep:
    def __init__(self, wm: WorldModel, default_joint_pos: torch.Tensor, compile_mode: str | None = None):
        self.wm = wm
        self.q0 = default_joint_pos
        self._fn = self._step
        if compile_mode:
            mode = None if compile_mode == "default" else compile_mode
            self._fn = torch.compile(self._step, mode=mode, dynamic=False)

    def _step(self, s, a, a_prev, xi, ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat):
        ref = ReferenceFrame(ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat)
        s_next = self.wm.step(s, a, xi)
        bodies = self.wm.heads.robot_bodies(s_next)
        terms = self.wm.heads.reward(s_next, a, a_prev, ref, bodies)
        r = terms["total"]
        term_vec = torch.stack([terms[k] for k in REWARD_TERM_NAMES], dim=0)  # (8, n)
        logits = self.wm.termination_logits(s_next, ref)
        obs_clean = analytic_observation(s_next, a, xi, self.q0)
        ai = self.wm.heads.anchor_index
        anchor_pose = torch.cat([bodies["pos"][..., ai, :], bodies["quat"][..., ai, :]], dim=-1)  # (n, 7)
        return s_next, r, term_vec, logits, obs_clean, anchor_pose

    def __call__(self, s, a, a_prev, xi, ref: ReferenceFrame):
        return self._fn(s, a, a_prev, xi, ref.body_pos_w, ref.body_quat_w, ref.body_lin_vel_w, ref.body_ang_vel_w, ref.root_pos_w, ref.root_quat_w)
