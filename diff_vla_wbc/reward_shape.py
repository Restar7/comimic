#!/usr/bin/env python3
"""奖励形状：`exp` 主体 + **hinge 尾部**，以及 root 朝向的 tilt / yaw / yaw_rate 拆项。

对标 `docs/HANDOVER_TO_ISAACLAB_FULL_ARCH_20261030.md` §6.4 / §6.5（mjlab 侧实测有效的形状）。

为什么要加尾部（他们的实测，80 集分桶）
----------------------------------------
`exp(-e/σ²)` 的梯度是 `-2e/σ²·exp(-e/σ²)`，**两端都趋 0**：

| 误差区间 | 占时刻 | 提供的位置梯度 |
|---|---|---|
| 0–0.1 m | 72.4% | **86.3%** |
| >0.4 m  | **21.2%** | **0.22%** |

⇒ 纯 exp 目标会**主动"拿尾部换中位"**：中位显著变好（5.79→4.11 cm），
但 p90 翻 2.4 倍（24.9→59.6）。换成"exp 主体 + hinge 尾部"后：
p50 4.41、**p90 26.9（−48%，自助 CI 不重叠）**、摔倒 0/80。

尾部必须是 **hinge** 而不是 MSE/Huber：
纯 MSE 在 0~e₀ 与 exp 的小误差展开（二次）**完全重复**；Huber 在主体区间也重复；
只有 hinge 在 0~e₀ 恒 0、>e₀ 线性，才不和 exp 抢梯度。

为什么要拆 root 朝向
--------------------
SONIC token **按定义 yaw-相对**（`anchor = _rot6d(quat_conj(q_robot) ⊗ q_ref)`），
官方 VLA 路径没有 anchor 输入 ⇒ 绝对 yaw 只能从相机猜（token→yaw 的 R²=0.226）。
实测 ori 误差分解：**yaw 7.53°（p90 24.7°）** / pitch 5.34° / roll 1.85°
—— 10.24° 里约 **7.5° 架构上拿不到**。所以按"可观测性"配权重：

| term | 权重 | 可观测性 |
|---|---|---|
| tilt（只用 roll/pitch） | 1.0 | ✅ `proj_grav` 直接可观测 |
| yaw（绝对方位） | 0.2 | ⚠️ 仅 23% 通道 |
| **yaw_rate** | 0.3 | ✅ 陀螺 `w_b` |

坐标约定（与仓库其它地方一致）
------------------------------
* state 71 = `[p(3), quat(4,**wxyz**), q(29), v_world(3), w_world(3), qd(29)]`
* 索引：p 0:3 / quat 3:7 / q 7:36 / v 36:39 / w 39:42 / qd 42:71
"""
from __future__ import annotations

import math

import torch

# ---- 默认阈值（全部来自 mjlab §6.5 的实测配方）---------------------------------
ROOT_POS_THR = 0.35          # m
ROOT_POS_CAP = 1.00          # m
TILT_THR_DEG = 20.0          # 度
TILT_CAP_DEG = 57.0          # 度
ROOT_POS_STD = 0.20          # m（主体 exp 的 σ）
TILT_STD = 0.25              # rad
# ★ 2026-10-31（P1，用户口径"必须改"）：YAW_STD 0.25(14.3°) 与实际误差严重不匹配——
#   在 B_d3 的 930 个真实重规划时刻上，yaw 误差 **中位 37.1°、均值 57.0°**（p90 143°），
#   核值 exp(-e²/σ²) 分别只有 **1.2e-3 / 1.3e-7** ⇒ 这一项在典型窗口**梯度≈0**（死区）。
#   单纯"加权重"没用（0.2→1.0 也只是 1e-3 量级），必须先把 σ 拉回敏感区。
# ★ 2026-10-31（A1，用户口径："偏差大的应该用 hinge，不该拿 σ 调"）：
#   σ 只管**近场精度**，拉大 σ 会把近场也调钝（3° 的偏差从"被重罚"变成"几乎满额"）。
#   所以 σ 回到 0.25(14.3°)，远场交给下面的 hinge 线性补坡。
YAW_STD = 0.25               # rad(14.3°)：近场精度，别动
YAWRATE_STD = 1.00           # rad/s
# 权重重配（用户口径："tilt 太高，最重要的其实是 yaw"）：
#   tilt 1.0→0.5（它本来就活得好：实测核值 0.63，占奖励 95%，压住其它项）
#   yaw  0.2→1.0（最重要的量，之前被死区 + 低权重双杀）
#   yaw_rate 0.3→0.0（B_d3 上 corr(w_z_robot, w_z_ref)=+0.005、ratio 0.56 ⇒ 该项奖励的是噪声）
# 2026-10-31 第二轮校准（用户口径："tilt 还是太大、yaw 的 hinge 也很小"）：
#   实测旧配比 = tilt 55% / yaw-exp 25.7% / yaw-hinge 19.3%（tilt 仍过半）；把 tilt 再砍一半、
#   hinge 坡度翻到 0.3，让 yaw（近场 exp + 远场 hinge）占到七成左右。
W_TILT, W_YAW, W_YAWRATE = 0.2, 1.0, 0.0
# yaw 的 hinge 项：exp 在大误差处仍会饱和（p90 143° = 4.2σ），补一条线性坡度，
# 让"偏得离谱"的窗口**始终有梯度**（与 hinge_tail 的 cap_slope 同一思路）。
YAW_HINGE_THR_DEG = 15.0     # 超过 15° 开始线性罚（近场仍由 exp 管）
YAW_HINGE_LAMBDA = 0.3       # 每弧度罚多少。标定：实测 B 线上 λ=0.2 ⇒ hinge 只占 19%；
                             # 0.3 ⇒ 约 37%（占 yaw 的 53%）。λ=0.5 时平均罚 0.35 会把 exp 抵掉
YAW_HINGE_CAP_DEG = 60.0     # 到 60° 后坡度减半（个别 180° 窗口不再独占梯度）
YAW_HINGE_SLOPE = 0.5


def quat_to_up(q: torch.Tensor) -> torch.Tensor:
    """quat(wxyz) → 机体 z 轴在世界系的单位向量（即旋转矩阵第 3 列）。"""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ], dim=-1)


def quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """quat(wxyz) → 绕世界 z 的 yaw（rad，ZYX 分解的 atan2 形式）。"""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_pi(a: torch.Tensor) -> torch.Tensor:
    """把角度绕回 [-π, π]。"""
    return torch.remainder(a + math.pi, 2.0 * math.pi) - math.pi


def up_in_heading_frame(q: torch.Tensor) -> torch.Tensor:
    """把机体 up 向量转到**该姿态自己的朝向系**（绕世界 z 反向转 yaw）。

    为什么必须这么做：直接比较两个世界系 up 向量时，**yaw 差异会漏进 tilt**
    （倾斜的机体绕 z 转一点，up 向量就会动）。实测 dyaw=10°/30°/90° 分别
    产生 tilt2=3.75e-3 / 3.3e-2 / 0.25 —— 正是要避免的泄漏。
    各自转到自己的朝向系后，纯 yaw 偏移对 tilt 的贡献恒为 0（自检项 [3] 验证）。
    """
    up = quat_to_up(q)
    yaw = quat_to_yaw(q)
    c, s_ = torch.cos(-yaw), torch.sin(-yaw)
    x = c * up[..., 0] - s_ * up[..., 1]
    y = s_ * up[..., 0] + c * up[..., 1]
    return torch.stack([x, y, up[..., 2]], dim=-1)


def tilt_yaw_yawrate_err(rob: torch.Tensor, ref: torch.Tensor) -> dict[str, torch.Tensor]:
    """返回三项**平方误差**（与 exp(-e/σ²) 的 e 同量纲）。

    rob/ref: (B,71) 状态。yaw_rate 用 state 里的 `w_world`（索引 39:42）的 z 分量——
    这正是 mjlab 说的"锚点 body 的世界系角速度 z"。
    """
    # tilt：各自转到自己的朝向系再比 ⇒ 只含 roll/pitch，与绝对方位无关。
    # ★ 2026-10-30 修（NaN 根因）：**不能用 acos**。
    #   `acos'(x) = -1/sqrt(1-x²)`，在 x→1（完美跟踪）处**发散到无穷**；
    #   训练里第一步 rollout 的起点**恰好等于参考状态** ⇒ cos_t 精确等于 1 ⇒ 梯度 NaN
    #   ⇒ NaN-guard 在第 19 步中止。这正是 mjlab §10 列的 "_sq_quat_err / acos / sqrt(0)"
    #   那一类坑，他们的建议是**全线照抄他们的安全写法**。
    #   这里改用"up 向量差的平方"（= 4·sin²(θ/2)，小角下 ≈ θ²，与原来同量纲），
    #   它没有 acos/sqrt，梯度 = 2(up_r−up_f)，在 0 处平滑为 0。
    up_r = up_in_heading_frame(rob[:, 3:7])
    up_f = up_in_heading_frame(ref[:, 3:7])
    tilt2 = (up_r - up_f).pow(2).sum(-1)

    dyaw = wrap_pi(quat_to_yaw(rob[:, 3:7]) - quat_to_yaw(ref[:, 3:7]))
    yaw2 = dyaw.pow(2)

    dwz = rob[:, 39 + 2] - ref[:, 39 + 2]
    yawrate2 = dwz.pow(2)
    # dyaw 一并返回：hinge 项要用 |dyaw|，但**不能**写成 sqrt(yaw2)——
    # sqrt 在 0 处导数发散 ⇒ 完美跟踪时梯度 inf ⇒ NaN（2026-10-31 实测踩过）。
    return {"tilt2": tilt2, "yaw2": yaw2, "yawrate2": yawrate2, "dyaw": dyaw}


def ori_split_reward(rob: torch.Tensor, ref: torch.Tensor,
                     *, w_tilt: float = W_TILT, w_yaw: float = W_YAW,
                     w_yawrate: float = W_YAWRATE,
                     hinge_lambda: float = YAW_HINGE_LAMBDA,
                     std_tilt: float = TILT_STD, std_yaw: float = YAW_STD,
                     std_yawrate: float = YAWRATE_STD) -> tuple[torch.Tensor, dict]:
    """拆项后的朝向奖励（**未乘 dt**，与 body 的 exp 核同式）。

    返回 (r_ori, terms)，r_ori = Σ w·exp(-e/σ²)。
    """
    e = tilt_yaw_yawrate_err(rob, ref)
    t_exp = torch.exp(-e["tilt2"] / std_tilt ** 2)
    y_exp = torch.exp(-e["yaw2"] / std_yaw ** 2)
    w_exp = torch.exp(-e["yawrate2"] / std_yawrate ** 2)
    # ★ 2026-10-31：这里**不要**补 hinge。第一版用 `sqrt(yaw2)` 取 |Δyaw|，
    #   而 sqrt 在 0 处导数发散 ⇒ 完美跟踪时梯度 = inf ⇒ NaN（和 acos 那次的坑同类，
    #   test_reward_shape 的"完美跟踪: 梯度≈0"当场抓到 max|g|=nan）。
    #   重新算 |Δyaw| 虽然可以用 abs（梯度有限），但实测它的量级（p90 143° ⇒ 罚 0.95）
    #   会把 exp 项（+0.41）直接抵掉，等于又变成"只罚不奖"。
    #   结论：yaw 只用 exp，靠 **σ=0.6 + 权重 1.0** 解决死区——中位误差处核值已从 1.2e-3 升到 0.311。
    # 远场：|Δyaw| 超过 thr 之后线性罚（exp 在 ~2σ 外已饱和到 0，靠这条保证"始终有梯度"）。
    # 用 abs(wrap_pi(·)) 而不是 sqrt(yaw2)：sqrt 在 0 处导数无穷 ⇒ NaN。
    _yaw_abs = e["dyaw"].abs()
    y_hinge = _ramp_then_slope(torch.relu(_yaw_abs - math.radians(YAW_HINGE_THR_DEG)),
                               math.radians(YAW_HINGE_CAP_DEG), YAW_HINGE_SLOPE)
    # ★ 2026-10-31（AB6）：λ 允许**按轮覆盖**（bundle.yaw_hinge_lambda，默认仍是 0.3）。
    #   原来写死模块常量 ⇒ "关掉远场 hinge"这个消融根本没法做（只能改代码）。
    _ylam = float(hinge_lambda)
    r = (w_tilt * t_exp + w_yaw * y_exp + w_yawrate * w_exp
         - _ylam * y_hinge)
    out = {**{k: v.detach() for k, v in e.items() if k != "dyaw"},
           "ori_tilt_exp": (w_tilt * t_exp).detach(),
           "ori_yaw_exp": (w_yaw * y_exp).detach(),
           "ori_yaw_hinge": (_ylam * y_hinge).detach(),
           "ori_yawrate_exp": (w_yawrate * w_exp).detach(),
           "ori_split": r.detach()}
    return r, out


def _ramp_then_slope(x: torch.Tensor, cap: float, slope: float) -> torch.Tensor:
    """``min(x, cap)`` + 越过 cap 之后仍保留 ``slope`` 的斜率。

    为什么不能只用 ``clamp(max=cap)``：超过 cap 后惩罚变成**常数**，梯度恰好为 0 ——
    越是"烂到没边"的样本越没有梯度，等于把远场整段扔掉。我们 B 线真值误差 p50=0.745 m、
    52% 的时刻在 0.5 m 以外，正好都落在这段平台上（2026-10-31 查）。
    ``slope=0`` 保持旧行为（逐位相同），便于和 d1–d2b 对拍。
    """
    if cap <= 0.0:
        return x
    _in = torch.clamp(x, max=cap)
    if slope <= 0.0:
        return _in
    return _in + float(slope) * torch.relu(x - cap)


def hinge_tail(rob: torch.Tensor, ref: torch.Tensor,
               *, pos_thr: float = ROOT_POS_THR, pos_cap: float = ROOT_POS_CAP,
               tilt_thr_deg: float = TILT_THR_DEG,
               tilt_cap_deg: float = TILT_CAP_DEG,
               cap_slope: float = 0.0) -> tuple[torch.Tensor, dict]:
    """尾部 hinge 惩罚（**惩罚量，已取正**，调用方自己乘 -λ）。

    * root 位置：`relu(‖Δp‖ − pos_thr)`，上限 pos_cap（``cap_slope>0`` 时越过上限仍保留斜率）
    * root 倾斜：`relu(θ_tilt − tilt_thr)`，上限 tilt_cap（同上）
      （θ_tilt 用"机器人 up 与参考 up 的夹角"，即**跟踪**意义上的倾斜误差；
       不用绝对倾角，否则会惩罚参考动作本身就有的倾身，如 Boxing）
    """
    dpos = (rob[:, 0:3] - ref[:, 0:3]).norm(dim=-1)
    pos_tail = _ramp_then_slope(torch.relu(dpos - pos_thr), float(pos_cap), float(cap_slope))

    up_r = up_in_heading_frame(rob[:, 3:7])
    up_f = up_in_heading_frame(ref[:, 3:7])
    # 同上：不用 acos。|Δup| = 2·sin(θ/2) ⇒ θ = 2·asin(|Δup|/2)，asin 在 0 处梯度有限（=1）。
    _chord = (up_r - up_f).norm(dim=-1)
    tilt = 2.0 * torch.asin((0.5 * _chord).clamp(max=1.0 - 1e-6))
    thr = math.radians(tilt_thr_deg)
    cap = math.radians(tilt_cap_deg)
    tilt_tail = _ramp_then_slope(torch.relu(tilt - thr), cap, float(cap_slope))

    total = pos_tail + tilt_tail
    return total, {"root_pos_tail": pos_tail.detach(), "root_tilt_tail": tilt_tail.detach()}


def group_body_pos_tail(per_body_err: torch.Tensor, legs: slice, arms: slice,
                        *, legs_thr: float = 0.10, arms_thr: float = 0.25) -> torch.Tensor:
    """分组的 body 位置尾部（**需要逐 body 误差**；MDP 目前只给聚合值时不用）。

    per_body_err: (B, n_body) 每个 body 的位置误差（米）。
    """
    legs_rms = per_body_err[:, legs].pow(2).mean(-1).sqrt()
    arms_rms = per_body_err[:, arms].pow(2).mean(-1).sqrt()
    return torch.relu(legs_rms - legs_thr) + torch.relu(arms_rms - arms_thr)
