"""把"参考帧相对的姿态误差"送进 VLA 的 state 输入（46 → 52 维）。

为什么要它（2026-10-31 用户提问："横向 14cm、偏航 12.7° 没办法避免吗"）
------------------------------------------------------------------------
VLA 的 state 输入（46 维）是 ``[关节22 | 左手7 | 关节7 | 右手7 | 重力3]``，构造见
``scripts/ha_vla_latent64_server.py:state64_to_state46``。关键的一行是::

    gravity = gravity_from_rot6d(s[0:6])      # 只保留重力方向

重力在体坐标系里的方向 ``g_b = -R^T e_z`` 对**绕世界 z 的旋转完全不变** ⇒ 偏航对策略
**不可观测**（位置则从来没有进过这个向量）。而 token 是"以当前朝向左乘"的相对帧，
于是朝向一偏，命令就整段转歪，下一步再乘一次错朝向 ⇒ 正反馈放大。

补救不是给 WBC 加 6 维（那条是 ``paper_obs93``，``convert_obs99_to_obs93.py`` 已经把
``rot6d(RᵀR_ref)`` 手术式删掉了，属于执行器输入，冻结不动），而是给 **VLA 的 state**
补上同样量的"参考相对姿态误差"：``rot6d(R_robotᵀ R_ref)``。

为什么用"相对"而不是绝对
------------------------
绝对朝向 ``rot6d(R)`` 在换初始摆位/换任务时分布会漂；而"我和参考帧差多少"是**恒定
语义**的控制误差，跨任务同分布，且正是策略需要清零的东西。

为什么是 6 维而不是 2 维（sin/cos Δyaw）
---------------------------------------
6D 表示同时编码了 roll/pitch 的耦合误差，且与 mjlab/trackerLab 原版口径一致
（``trackerlab_adapter`` 的 99 维第 6 项就是这个量），便于对齐与复用。

落位
----
``state`` 在进模型前被零填充到 ``max_state_dim=132``（``gr00t_vlm.prepare_observation``），
本 ckpt 只用前 46 维 ⇒ 新 6 维写在 **46:52**，**不改任何权重形状**。
"""

from __future__ import annotations

import torch

#: 现有 state 用到的宽度（29 关节 + 14 手 + 3 重力）
STATE_BASE_DIM = 46
#: 新增的"参考相对姿态误差"占用的槽位
STATE_EXTRA_SLICE = (46, 52)
STATE_EXTRA_DIM = 6


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """``(...,4)`` wxyz → ``(...,3,3)``（与 ``trackerlab_adapter.quat_to_mat`` 同式）。"""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q.unbind(-1)
    m = torch.empty(*q.shape[:-1], 3, 3, dtype=q.dtype, device=q.device)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def mat_to_rot6d(m: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 → 6D（前两**行**摊平）。``R = I`` → ``[1,0,0,0,1,0]``。

    行优先（不是列）是刻意的：与 ``state64`` 里 ``rot6d(quat)`` 的既有实现保持一致，
    否则同一个量在两处会静默地"转置"。
    """
    return torch.cat((m[..., 0, :], m[..., 1, :]), dim=-1)


def rel_ori6(quat_robot: torch.Tensor, quat_ref: torch.Tensor) -> torch.Tensor:
    """``rot6d(R_robotᵀ R_ref)``：机器人在**参考朝向下**还差多少旋转（体坐标系）。

    * 完全对齐 ⇒ ``[1,0,0,0,1,0]``
    * 只差 Δyaw ⇒ 前两行的 ``(cosΔ, sinΔ, -sinΔ, cosΔ)`` 结构
    """
    Rr = quat_to_mat(quat_robot)
    Rf = quat_to_mat(quat_ref)
    return mat_to_rot6d(Rr.transpose(-1, -2) @ Rf)


def yaw_error_from_rel6(rel6: torch.Tensor) -> torch.Tensor:
    """从 6D 里取回 Δyaw（弧度）——只用于审计/诊断打印。

    符号口径：**参考比我多转了多少**（机器人落后于参考 ⇒ 正值）。
    ``R_rᵀ R_ref = Rz(φ)`` 时 rel6 的前两行是 ``[cosφ, −sinφ, 0, sinφ, cosφ, 0]``，
    所以 ``atan2(−rel6[1], rel6[0]) = φ``。
    """
    return torch.atan2(-rel6[..., 1], rel6[..., 0])


def write_state_extra(state: torch.Tensor, extra6: torch.Tensor) -> torch.Tensor:
    """把 6 维写进 ``state`` 的 46:52 槽位（``state`` 可以是 ``(B,1,132)`` 或 ``(B,132)``）。"""
    a, b = STATE_EXTRA_SLICE
    if state.shape[-1] < b:
        raise ValueError(f"state 宽度 {state.shape[-1]} < {b}，无法写入参考相对姿态")
    if extra6.shape[-1] != STATE_EXTRA_DIM:
        raise ValueError(f"extra 宽度 {extra6.shape[-1]} != {STATE_EXTRA_DIM}")
    if extra6.shape[0] != state.shape[0]:
        raise ValueError(f"extra 批 {extra6.shape[0]} != state 批 {state.shape[0]}")
    if state.dim() == 3 and extra6.dim() == 2:
        extra6 = extra6.unsqueeze(1)          # (B,6) → (B,1,6)，对齐 state 的 (B,1,132)
    out = state.clone()
    out[..., a:b] = extra6.to(dtype=out.dtype, device=out.device)
    return out


def vel6_body(v_world: torch.Tensor, w_world: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """体坐标系 6 维速度 ``[Rᵀv, Rᵀw]``。

    为什么用它（2026-10-31，e1 的**可落地**版本）：``rot6d(R_robotᵀR_ref)`` 需要参考帧，
    而闭环采集器里没有参考帧（`closed_loop_vla_c6h.py` 用 ``FrameRef(zeros)`` 占位）⇒
    训练有、评测无 = 静默错配。速度则是**两边都有**的：
      * dump/state71：[p(3), quat(4), q(29), **v(3)**, **w(3)**, qd(29)]，v/w 为**世界系**
        （见 ``isaac_joint.flat_state_tuple`` 与 WM 积分 ``v1 = v + R dv``）；
      * 闭环：MuJoCo 自由关节 ``qvel[0:3]``=线速度、``qvel[3:6]``=角速度，同样是**世界系**。

    而 ``w_b[2]``（体坐标偏航角速度）正是"压偏航漂移"需要的导数信号，且 ``w_b`` 的前两维
    给出 roll/pitch 角速度——这些在现有 46 维 state 里**一个都没有**。
    """
    R = quat_to_mat(quat)
    Rt = R.transpose(-1, -2)
    vb = (Rt @ v_world.unsqueeze(-1)).squeeze(-1)
    wb = (Rt @ w_world.unsqueeze(-1)).squeeze(-1)
    return torch.cat((vb, wb), dim=-1)


def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """四元数 → 偏航角（弧度）。只用于采集器的坐标系自检。"""
    R = quat_to_mat(quat)
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])


def extra6_for(mode: str, *, quat: torch.Tensor,
               v_world: torch.Tensor | None = None,
               w_world: torch.Tensor | None = None,
               quat_ref: torch.Tensor | None = None,
               q_init: torch.Tensor | None = None) -> torch.Tensor:
    """统一入口：训练侧与闭环侧**必须**都走这里（口径漂移的唯一防线）。"""
    m = (mode or "none").lower()
    if m == "vel6":
        if v_world is None or w_world is None:
            raise ValueError("vel6 需要 v_world/w_world")
        return vel6_body(v_world, w_world, quat)
    if m == "rotlocal6":
        if q_init is None:
            raise ValueError("rotlocal6 需要 q_init（回合起始姿态）")
        return rotlocal6(q_init, quat)
    if m == "abs6":
        return abs6_from_quat(quat)
    if m == "ori6":
        if quat_ref is None:
            raise ValueError("ori6 需要 quat_ref（闭环里拿不到，见 vel6_body 的说明）")
        return rel_ori6(quat, quat_ref)
    raise ValueError(f"未知 state_extra mode={mode!r}（none|vel6|ori6）")


def extra6_checksum(extra6: torch.Tensor) -> list[float]:
    """前 6 维的校验和（训练侧/闭环侧对拍用；不一致说明口径漂移）。"""
    x = extra6.detach().reshape(-1)[:STATE_EXTRA_DIM].to("cpu", dtype=torch.float64)
    return [round(float(v), 6) for v in x]


def mat_to_rot6d_wire(m: torch.Tensor) -> torch.Tensor:
    """**与 HA 请求线上的 rot6d 完全一致**的 6D 编码。

    线上实现（`closed_loop_vla_c6h.rot6d`，官方 provider 同款）是
    ``m[:, :2].reshape(6)``，即**列块**按行优先摊平：
    ``[m00, m01, m10, m11, m20, m21]``。

    ⚠️ 这和 ``mat_to_rot6d``（按**行**摊平 ``[m00,m01,m02,m10,m11,m12]``）**不是同一个东西**。
    两处用错会静默产生一个"看着像旋转、其实顺序错了"的 6 维——所以 abs6 一律走这个函数，
    并用 ``scripts/test_state_extra.py`` 对 scipy 的线上公式逐位对拍。
    """
    return torch.stack((m[..., 0, 0], m[..., 0, 1],
                        m[..., 1, 0], m[..., 1, 1],
                        m[..., 2, 0], m[..., 2, 1]), dim=-1)


def abs6_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """绝对朝向 6D（线上口径）。**闭环采集器零改动**——它本来就在 state64[0:6] 里发这 6 个数。"""
    return mat_to_rot6d_wire(quat_to_mat(quat))


# ---------------------------------------------------------------------------
# rotlocal6：**线上协议里已经有的那 6 维**（相对本回合起始朝向的旋转）
# ---------------------------------------------------------------------------
def quat_heading(q: torch.Tensor) -> torch.Tensor:
    """纯 yaw（heading）四元数。与 HA 的 ``quat_heading_wxyz`` 同语义。"""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    h = 0.5 * yaw
    zeros = torch.zeros_like(h)
    return torch.stack((torch.cos(h), zeros, zeros, torch.sin(h)), dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.stack((q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]), dim=-1)


def rotlocal6(q_init: torch.Tensor, q_now: torch.Tensor) -> torch.Tensor:
    """``rot6d( conj(heading(q_init)) * q_now )`` —— **闭环协议里的那 6 维**。

    语义（已对拍第三方实现，逐位一致）：
      * 开局时恒为 ``[1,0,0,0,1,0]``（identity）；
      * 之后等于"从开局到现在转了多少"（yaw 分量就是累计转角）；
      * **与回合无关**（每集都从 identity 起），所以不存在"绝对朝向在不同集里
        意义不同"的病态问题。

    注意：这是**自转量**，不是"与参考的误差"（那是 ``rel_ori6``，需要闭环侧发参考帧）。
    但它构造良好、且**线上本来就在发**（``state64[:6]``），因此零客户端改动。
    """
    return mat_to_rot6d_wire(quat_to_mat(quat_mul(quat_conj(quat_heading(q_init)), q_now)))
