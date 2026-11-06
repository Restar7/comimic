"""Analytic BeyondMimic tracking reward in torch (plan D4/D7).

One differentiable implementation shared by the simulator side (tests / logging)
and the imagination side (world-model reward head). Mirrors mjlab's
``tasks/tracking/mdp/rewards.py`` term by term, including the re-anchoring of the
reference bodies to the robot anchor in xy + yaw only
(``MotionCommand.update_relative_body_poses``).

Inputs are world-frame link quantities of the *tracked* bodies (same ordering as
``TRACKED_BODIES``), the anchor being one of them (``anchor_index``).
Rewards are returned already weighted and multiplied by ``dt`` like mjlab's reward
manager (``scale_rewards_by_dt=True``).
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat


@dataclass
class TrackingRewardCfg:
    std_body_pos: float = 0.3
    std_body_ori: float = 0.4
    std_body_lin_vel: float = 1.0
    std_body_ang_vel: float = 3.14
    # 2026-10-29 用户口径 B：收紧 root 的 std。
    # 原因：std=0.3/0.4 时 root_pos 4cm ⇒ reward 0.981、root_ori 10° ⇒ 0.907，梯度只剩满量程的 2–10%
    # （饱和）。收紧到 0.2/0.25 后 4cm ⇒ 0.961、10° ⇒ 0.836，小误差处重新有梯度。
    # 注意：只改 std、不改 w_*，所以 max_step_reward 仍是 0.1 ⇒ rounds.csv 的 mean_reward 跨轮可比。
    std_root_pos: float = 0.2
    # 2026-10-29 二次收紧：0.25 → 0.15。
    # 实测（同 WM、只换 head）：root_ori 4.13° → 8.81°（+113%），而 root_pos 反而 −47%
    # ⇒ reward 的"甜区"偏向位置（pos std 0.2 时 4cm=0.961；ori std 0.25 时 10°=0.852）。
    # 收到 0.15 后 10° 处 reward 0.852 → 0.64，姿态梯度约 ×2.8，且不改 w_* ⇒ max_step_reward 仍 0.12。
    std_root_ori: float = 0.25   # 2026-10-29 回滚：0.15 实测在 80 集总体层面**没有**改善 ori
                                # (p50 10.7→10.0 微降，p90 反而 24.7→45.6 恶化)，却把策略
                                # 推向大朝向摆动（R3 起 80 集摔倒 2→4/80）。收益没拿到，风险全吃。
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_lin_vel: float = 1.0
    w_body_ang_vel: float = 1.0
    # 2026-10-29 用户口径：root 权重 0.5 → 1.0（root 占总权重 5.0 → 6.0 的 33%）。
    # 注意 max_step_reward = dt * Σw 会从 0.10 变 0.12 ⇒ rounds.csv 的 mean_reward 是**新尺度**，
    # 与 2026-10-29 早间的 R1=0.0732 / R2=0.0598 不可直接比（那两轮从 R1 重跑）。
    # 2026-10-30：**关掉绝对位置项**。理由：参考的 root x/y 是我们用 planner 重建的
    # （官方那份 .pkl 没有），而实测"关节全部跟住"只能走出参考的 ~0.4 倍 ⇒ 这一项不可满足；
    # WBC 训练侧已同样关掉（`official_tracking_reward_cfg().w_root_pos = 0`）。
    # 置 0 会同时归零 `motion_global_root_pos` 与 `motion_root_pos_tail`；
    # `max_step_reward` 因此 0.13 → 0.11（跨轮比较 mean_reward 要用新基线）。
    w_root_pos: float = 0.0
    w_root_ori: float = 1.0
    # ---- 尾部铰链（2026-10-29，针对 80 集 pos p90 恶化）----
    # 诊断：`exp(-e/σ²)` 在 e>2σ 后梯度≈0。实测 π4 有 **21.2% 的时刻误差>0.4 m，
    # 却只提供 0.22% 的位置梯度** ⇒ 均值目标主动"拿尾部换中位"：p50 5.8→4.1（显著改善）
    # 的同时 p90 24.9→52–60（2.2 倍恶化）。这条铰链让远处误差**线性付费**。
    # 形式： -tail_pos_weight * relu(||Δp|| - tail_pos_e0)，单位米，结果按 dt*w_root_pos 缩放。
    # e0=0.35 约 1.75σ，正好卡在"中位都在内侧、p90 在外侧"的分界；λ=2.0 时 0.5 m 误差
    # 罚 0.006（= 理想位置项的 30%），1.0 m 封顶罚 0.04（= max_step_reward 的 33%）。
    tail_pos_weight: float = 2.0
    tail_pos_e0: float = 0.35
    tail_pos_cap: float = 1.0
    # 朝向尾部铰链（同款）：`e_root_ori` 是**平方测地角**(rad²)，所以 θ = sqrt(e)。
    # 实测 ori p90 也从 24.7 涨到 45.6°（p50 只 10–13°）⇒ 同样的"中位好、尾部坏"。
    # e0 = 20°（0.349 rad）卡在 p75~p90 之间；λ=2.0 时 45° 罚 0.017（= 理想朝向项的 85%）。
    tail_ori_weight: float = 2.0
    tail_ori_e0_rad: float = 0.3490658503988659   # 20°
    tail_ori_cap_rad: float = 1.0                 # 57°，封顶罚 0.04 = max_step_reward 的 33%
    # ---- root 朝向拆分（2026-10-30）----
    # 背景：我们的 VLA 路径**没有** anchor 输入（官方 `unitree_g1_sonic` 的 state 就 8 项，
    # 不含 `motion_anchor_orientation`；anchor 只在 encoder 路径里给）。而 SONIC 的 token 按定义
    # 是"相对机器人自身朝向"的（`anchor = rot6d(quat_conj(q_robot) ⊗ q_ref)`）⇒ **绝对 yaw 从来不进 token**。
    # 实测：token 对 yaw 误差的 CV R² 只有 0.226（对 root_pos 是 0.331）⇒ 绝对 yaw 只有 ~23% 可学。
    # 而 "转向" 的可观测部分是 **yaw 角速度**（`w_b` 陀螺就在观测里）。
    # 所以把 std_root_ori 的权重按可观测性拆三份（**总和仍 = w_root_ori，max_step_reward 不变**）：
    ori_split_tilt: float = 1.0        # roll/pitch，`proj_grav` 直接可观测 ⇒ **满权重**
    ori_split_yaw: float = 0.2         # 绝对 yaw，只有 ~23% 通道 ⇒ 降权
    ori_split_yaw_rate: float = 0.3    # yaw 角速度，"转向"的正确落点 ⇒ 专门项
    std_root_yaw_rate: float = 1.0     # rad/s，1σ = 1 rad/s 的转向速率差
    # ---- tilt / pelvis 高度每步惩罚（2026-10-30 B 项）----
    # 动机：环境侧早就有 `root_tilt_threshold=0.8`(≈45.8°) 和 pelvis 高度终止，但**奖励里一项都没有**
    # ⇒ 摔倒只在"终止时丢后续奖励"里体现，而 H=8=0.16 s 根本走不到终止（摔倒要 ~1 s 成形）。
    # A4 已证明拉长 H 会让效果变差（H8 0.0820 > H16 0.0778 > H32 0.0734 > H40 0.0681），
    # 所以正确入口是**让前兆在每一步可见**，而不是拉长视野。
    # 形式：-w_tilt·relu(tilt − e0)（弧度）+ -w_height·relu(h_min − pelvis_z)（米）。
    # e0=20° 与 ori 铰链一致；h_min=0.70 m（正常 pelvis z 实测 0.74–0.80）。
    # ---- body_pos 铰链（分组，2026-10-30）----
    # 实测（铰链臂 80 集，重锚定后）：手臂 6 body 中位 0.126 m、腿+躯干 8 body 中位 0.046 m，
    # 差 2.8 倍。若用"14 body 平均 + 单一阈值"，铰链实际只会去拽手臂 —— 而手臂那 0.13 m
    # 是 `arm=vla` 的正常偏置（肩俯仰 +34°），不是缺陷。所以按组各用自己的尾部定阈值。
    # TRACKED_BODIES 顺序：0 pelvis, 1-3 左腿, 4-6 右腿, 7 torso, 8-13 左右肩/肘/腕
    tail_body_legs_weight: float = 2.0
    tail_body_legs_e0: float = 0.10          # ≈ 腿+躯干当前 p90
    tail_body_arms_weight: float = 2.0
    tail_body_arms_e0: float = 0.25          # 只在手臂真尾部咬合
    tail_body_arms_idx: tuple = (8, 9, 10, 11, 12, 13)
    w_tilt: float = 0.0            # 2026-10-30 关闭：库内参考自身 max tilt=34.4°、56/318 clip 超 20°
                                   # ⇒ 绝对 tilt 惩罚会压合法弯腰动作。而"相对参考"的 tilt 误差
                                   # 已由 `motion_root_tilt` 覆盖；摔倒另有 80 集硬门 + 坏集过滤。
    tilt_e0_rad: float = 0.6981317007977318       # 40°：只当**摔倒前兆**。
                                                  # 原来的 20° 会惩罚 3.2% 的合法帧、
                                                  # 56/318 个 clip 的弯腰动作（库内 max tilt=34.4°）。
    w_height: float = 0.0          # 2026-10-30 关闭：库里 pelvis z p1=0.413/min=0.385，
                                   # **13.4% 的合法帧 < 0.70 m** ⇒ 绝对高度惩罚会硬压"不许蹲"。
                                   # 只在**确实需要**的地面站立任务里再开。
    height_min: float = 0.70
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    # ---- 官方口径专用项（2026-10-30）----
    # `ori_split_ori` = **绝对四元数误差**这一项的权重份额。默认 0 ⇒ 联合微调（走默认值）
    # 的 terms/total **逐位不变**；只有 WBC 训练用的 `official_tracking_reward_cfg()` 把它置 1.0
    # 同时把 tilt/yaw/yaw_rate 三个拆分项置 0，用来复现 SuperMimic/GEAR-SONIC 官方的
    # 单一 `tracking_anchor_ori`（exp(-θ²/std²)，std=0.4）。
    ori_split_ori: float = 0.0
    std_root_ori_abs: float = 0.40
    dt: float = 0.02

    @property
    def max_step_reward(self) -> float:
        # 注意：root 朝向被拆成 tilt/yaw/yaw_rate 三项，权重总和可能 ≠ w_root_ori
        # （2026-10-30 B 方案：1.0+0.2+0.3 = 1.5×）⇒ 这里必须按拆分总和算，否则会低估真实上限。
        _ori = (self.ori_split_tilt + self.ori_split_yaw + self.ori_split_yaw_rate
                + self.ori_split_ori)
        return self.dt * (self.w_body_pos + self.w_body_ori + self.w_body_lin_vel
                          + self.w_body_ang_vel + self.w_root_pos + self.w_root_ori * _ori)


def reanchor_reference(
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express the reference bodies in a frame that shares the robot anchor's xy position and yaw.

    Exactly mjlab's ``update_relative_body_poses``: z, roll and pitch stay those of the reference.
    Shapes: bodies (N, B, ·), anchors (N, ·).
    """
    nb = ref_body_pos_w.shape[-2]
    delta_pos = robot_anchor_pos_w.clone()
    delta_pos = torch.cat([delta_pos[..., :2], ref_anchor_pos_w[..., 2:3]], dim=-1)  # (N,3)
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat_w)))  # (N,4)
    delta_ori_b = delta_ori[..., None, :].expand(*delta_ori.shape[:-1], nb, 4)
    rel_quat = quat_mul(delta_ori_b, ref_body_quat_w)
    rel_pos = delta_pos[..., None, :] + quat_apply(delta_ori_b, ref_body_pos_w - ref_anchor_pos_w[..., None, :])
    return rel_pos, rel_quat


def _sq_quat_err(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Squared geodesic angle between quaternions（θ²），**在 θ→0 处也光滑**。

    ⚠️ 2026-10-30 06:35 事故：原实现是 `quat_box_minus` → `axis_angle_from_quat`（log map），
    在"两个朝向完全一致"时是 0/0 奇异。**而训练越好越容易撞上这个点**（R1 第 535 步
    `grad/head_preclip` 变 NaN，当时 rollout/reward 已是本轮最好 0.105）。之前的轮次能跑完
    800 步只是因为跟踪还差、从没接近过 θ=0。

    改用单位四元数的**双覆盖欧氏距离**：min(‖q1−q2‖², ‖q1+q2‖²) = 2−2|q1·q2| ≈ θ²/2，
    再 ×4 得到 θ²（小角度下与旧口径**逐位同量纲**，所以 std 标定不用改），且梯度线性有界。
    """
    d2p = (q1 - q2).pow(2).sum(-1)
    d2m = (q1 + q2).pow(2).sum(-1)
    return 4.0 * torch.minimum(d2p, d2m)



def official_tracking_reward_cfg() -> "TrackingRewardCfg":
    """SuperMimic / GEAR-SONIC 官方 decoder 训练奖励（**只给 WBC 训练用**）。

    来源：``GR00T-WholeBodyControl/gear_sonic/config/manager_env/rewards/tracking/base.yaml``
    + ``terms/tracking_anchor_pos.yaml`` / ``tracking_anchor_ori.yaml`` 等：

        tracking_anchor_pos          0.5   std 0.30   （绝对世界坐标）
        tracking_anchor_ori          0.5   std 0.40   （**单一四元数误差**，含 yaw）
        tracking_relative_body_pos   1.0   std 0.30
        tracking_relative_body_ori   1.0   std 0.40
        tracking_body_linvel         1.0   std 1.00
        tracking_body_angvel         1.0   std 3.14
        action_rate_l2              -0.1
        joint_limit                -10.0
        undesired_contacts          -0.1   （需要接触传感器，本实现暂缺）

    与联合微调那份（``TrackingRewardCfg()`` 默认值：root 1.0/σ0.20、ori 拆 tilt/yaw/yaw_rate、
    4 条尾部铰链、joint_limit −2.0）**完全分开**，互不影响。
    """
    return TrackingRewardCfg(
        std_root_pos=0.30,
        std_root_ori=0.40,          # 拆分项（本口径下权重为 0，仅占位）
        std_root_ori_abs=0.40,      # 官方单一朝向项
        # 2026-10-30 实验结论：**绝对位置项在这份参考上不可满足** ⇒ 关掉。
        # 证据（三版都收敛到同一处）：
        #   基座 1.51 → 无锚 ep3s: 500 步 0.35；
        #             → 锚1e-3 ep10s+speed: 500/1000/2000/3500 = 0.61/0.56/0.31/0.24
        # 每次都是"保关节、弃位置"（tilt 一直 3-9°，位置 p50 0.5-1.2 m）。
        # 另外实测：录制关节本身的速度（bones_seed 标定）与参考要求同一量级，
        # 说明冲突不在"关节该不该走"，而在"参考的根轨迹与这批关节不自洽"
        # （参考 x/y 是我们用 planner 重建的，官方那份 .pkl 我们没有）。
        # ⇒ 位置项留在训练里只会教出"站着的折中"，故置 0；位置只在评测里看。
        w_root_pos=0.0,
        w_root_ori=0.5,
        w_body_pos=1.0,
        w_body_ori=1.0,
        w_body_lin_vel=1.0,
        w_body_ang_vel=1.0,
        ori_split_tilt=0.0,         # ← 关掉我们的拆分，改用单一 ori
        ori_split_yaw=0.0,
        ori_split_yaw_rate=0.0,
        ori_split_ori=1.0,
        tail_pos_weight=0.0,        # ← 官方没有铰链
        tail_ori_weight=0.0,
        tail_body_legs_weight=0.0,
        tail_body_arms_weight=0.0,
        w_tilt=0.0,
        w_height=0.0,
        w_action_rate=-0.1,
        # SuperMimic 官方（trackerlab-supermimic RewardCfg / 论文 Table A.2）= **-2.0**。
        # 之前我按 GEAR-SONIC 的 tracking/base.yaml 写成 -10.0，那是 SONIC decoder 的口径，
        # 不是 SuperMimic 的 —— 已改回 -2.0（2026-10-30 与 trackerlab RewardCfg 逐项对拍确认）。
        w_joint_limit=-2.0,
    )

def tracking_reward_terms(
    *,
    robot_body_pos_w: torch.Tensor,
    robot_body_quat_w: torch.Tensor,
    robot_body_lin_vel_w: torch.Tensor,
    robot_body_ang_vel_w: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_body_lin_vel_w: torch.Tensor,
    ref_body_ang_vel_w: torch.Tensor,
    anchor_index: int,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_limits: torch.Tensor,  # (29, 2) or (N, 29, 2)
    cfg: TrackingRewardCfg,
) -> dict[str, torch.Tensor]:
    """Per-term weighted, dt-scaled rewards (N,) plus 'total'."""
    ra_pos = robot_body_pos_w[..., anchor_index, :]
    ra_quat = robot_body_quat_w[..., anchor_index, :]
    ma_pos = ref_body_pos_w[..., anchor_index, :]
    ma_quat = ref_body_quat_w[..., anchor_index, :]
    rel_pos, rel_quat = reanchor_reference(ref_body_pos_w, ref_body_quat_w, ma_pos, ma_quat, ra_pos, ra_quat)

    e_root_pos = ((ma_pos - ra_pos) ** 2).sum(-1)
    # 官方口径用的"绝对朝向"误差：双覆盖平方距离 ≈ θ²（与官方 quat_error_magnitude 同形式）
    e_root_ori_abs = _sq_quat_err(ma_quat, ra_quat)
    # 注意：root 朝向已拆成 tilt / yaw / yaw_rate 三项（见 cfg.ori_split_*），
    # 所以不再有 `e_root_ori = _sq_quat_err(ma_quat, ra_quat)` —— 那行曾在这里，已删除。
    e_body_pos = ((rel_pos - robot_body_pos_w) ** 2).sum(-1).mean(-1)
    e_body_ori = _sq_quat_err(rel_quat, robot_body_quat_w).mean(-1)
    e_body_lin = ((ref_body_lin_vel_w - robot_body_lin_vel_w) ** 2).sum(-1).mean(-1)
    e_body_ang = ((ref_body_ang_vel_w - robot_body_ang_vel_w) ** 2).sum(-1).mean(-1)

    lo = soft_limits[..., 0]
    hi = soft_limits[..., 1]
    out = (-(joint_pos - lo)).clamp(min=0.0) + (joint_pos - hi).clamp(min=0.0)
    joint_limit = out.sum(-1)
    action_rate = ((action - prev_action) ** 2).sum(-1)

    dt = cfg.dt
    # 尾部铰链：只用 ||Δp||（米），σ 内的行为完全不受影响，σ 外线性付费
    # tilt / 高度前兆：与参考无关的**绝对**量（参考本身 tilt≈0、pelvis z≈0.76）
    _z = torch.zeros_like(ma_quat[..., :3])
    _z[..., 2] = 1.0
    up_z = quat_apply(ma_quat, _z)[..., 2].clamp(-1.0, 1.0)
    # ⚠️ 2026-10-30 06:35 事故：这里原来是 `torch.acos(up_z)`。机器人直立时 up_z≈1，
    # acos 的导数 -1/√(1-up_z²) → ∞ ⇒ 第 535 步 grad 炸成 NaN，白烧 266 步。
    # 换成光滑的 1-cosθ（≈θ²/2，单调、无奇点）；阈值同步换算成 1-cos 口径（20° → 0.0603）。
    tilt_1mcos = 1.0 - up_z
    tilt_pen = (tilt_1mcos - (1.0 - math.cos(cfg.tilt_e0_rad))).clamp(min=0.0)
    height_pen = (cfg.height_min - ma_pos[..., 2]).clamp(min=0.0)
    # 分组 body_pos 铰链：逐 body 平方距离 → 各自组内均值
    _d2b = ((rel_pos - robot_body_pos_w) ** 2).sum(-1)              # (N, nb)
    _arms = list(cfg.tail_body_arms_idx)
    _legs = [i for i in range(_d2b.shape[-1]) if i not in _arms]
    _e_legs = _d2b[..., _legs].mean(-1).clamp_min(1e-8).sqrt()
    _e_arms = _d2b[..., _arms].mean(-1).clamp_min(1e-8).sqrt()
    tail_body_legs = (_e_legs - cfg.tail_body_legs_e0).clamp(min=0.0)
    tail_body_arms = (_e_arms - cfg.tail_body_arms_e0).clamp(min=0.0)
    tail_pos = (e_root_pos.clamp_min(1e-8).sqrt() - cfg.tail_pos_e0).clamp(min=0.0, max=cfg.tail_pos_cap)
    # --- root 朝向三分量（见 cfg 里 ori_split_* 的注释）---
    _up_r = torch.stack([quat_apply(ma_quat, _z)[..., i] for i in range(3)], dim=-1)
    _up_f = torch.stack([quat_apply(ra_quat, _z)[..., i] for i in range(3)], dim=-1)
    # 只 roll/pitch。**必须用光滑形式**：`acos(dot)` 的导数 -1/√(1-dot²) 在"上下方向几乎平行"
    # 时（dot≈1，而我们几乎总是这种情况）会放大几千倍 ⇒ 2026-10-30 06:35 实测第 535 步 grad 炸成 NaN。
    # 平方距离 = 2(1-cosθ) = 4sin²(θ/2) ≈ θ²，量纲与旧口径一致，且处处可导、梯度有界。
    e_root_tilt = (_up_r - _up_f).pow(2).sum(-1)
    def _yaw(q):
        w_, x_, y_, z_ = q.unbind(-1)
        return torch.atan2(2.0 * (w_ * z_ + x_ * y_), 1.0 - 2.0 * (y_ * y_ + z_ * z_))
    _dyaw = _yaw(ma_quat) - _yaw(ra_quat)
    _dyaw = torch.remainder(_dyaw + math.pi, 2.0 * math.pi) - math.pi
    e_root_yaw = _dyaw.pow(2)
    # yaw 角速度：锚点 body 的世界系角速度 z 分量（转向速率）
    _ai = int(anchor_index)
    e_root_yaw_rate = (ref_body_ang_vel_w[..., _ai, 2] - robot_body_ang_vel_w[..., _ai, 2]).pow(2)
    _theta_tilt = e_root_tilt.clamp_min(1e-8).sqrt()
    tail_ori = (_theta_tilt - cfg.tail_ori_e0_rad).clamp(min=0.0, max=cfg.tail_ori_cap_rad)
    terms = {
        "motion_global_root_pos": dt * cfg.w_root_pos * torch.exp(-e_root_pos / cfg.std_root_pos**2),
        "motion_global_root_ori": dt * cfg.w_root_ori * cfg.ori_split_ori
                                  * torch.exp(-e_root_ori_abs / cfg.std_root_ori_abs**2),
        "motion_root_tilt": dt * cfg.w_root_ori * cfg.ori_split_tilt
                            * torch.exp(-e_root_tilt / cfg.std_root_ori**2),
        "motion_root_yaw": dt * cfg.w_root_ori * cfg.ori_split_yaw
                           * torch.exp(-e_root_yaw / cfg.std_root_ori**2),
        "motion_root_yaw_rate": dt * cfg.w_root_ori * cfg.ori_split_yaw_rate
                                * torch.exp(-e_root_yaw_rate / cfg.std_root_yaw_rate**2),
        "motion_body_pos": dt * cfg.w_body_pos * torch.exp(-e_body_pos / cfg.std_body_pos**2),
        "motion_body_ori": dt * cfg.w_body_ori * torch.exp(-e_body_ori / cfg.std_body_ori**2),
        "motion_body_lin_vel": dt * cfg.w_body_lin_vel * torch.exp(-e_body_lin / cfg.std_body_lin_vel**2),
        "motion_body_ang_vel": dt * cfg.w_body_ang_vel * torch.exp(-e_body_ang / cfg.std_body_ang_vel**2),
        "motion_root_pos_tail": -dt * cfg.w_root_pos * cfg.tail_pos_weight * tail_pos,
        "motion_root_ori_tail": -dt * cfg.w_root_ori * cfg.tail_ori_weight * tail_ori,
        "motion_body_pos_tail_legs": -dt * cfg.w_body_pos * cfg.tail_body_legs_weight * tail_body_legs,
        "motion_body_pos_tail_arms": -dt * cfg.w_body_pos * cfg.tail_body_arms_weight * tail_body_arms,
        "motion_tilt_pen": -dt * cfg.w_tilt * tilt_pen,
        "motion_height_pen": -dt * cfg.w_height * height_pen,
        "action_rate_l2": dt * cfg.w_action_rate * action_rate,
        "joint_limit": dt * cfg.w_joint_limit * joint_limit,
    }
    total = terms["motion_global_root_pos"]
    for k, v in terms.items():
        if k != "motion_global_root_pos":
            total = total + v
    terms["total"] = total
    return terms
