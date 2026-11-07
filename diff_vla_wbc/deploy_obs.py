"""**部署观测布局**（那 930 维）的唯一实现 —— 训练/部署两边都必须走这里。

为什么需要它
------------
部署侧（`action_provider_sonic.py::_build_live_decoder_obs_with_latent`）喂给 SONIC
解码器的输入是 **994 维**：

```
994 = motion_token(64)
    + ang_vel_hist      (10 × 3)     # robot.root_ang_vel_b        （body 系）
    + joint_pos_hist    (10 × 29)    # joint_pos - default_q       （HA_V31 序）
    + joint_vel_hist    (10 × 29)    # joint_vel                   （HA_V31 序，不减不缩放）
    + last_action_hist  (10 × 29)    # 上一步 decoder 的原始输出    （HA_V31 序）
    + grav_dir_hist     (10 × 3)     # gravity_dir_from_base_quat_wxyz
```

而训练侧 `mdp.obs()` 的每帧是 **93 维**，块顺序完全不同：

```
93 = (q - default_q - xi_dq0)(29) | qd(29) | ω_body(3) | g_b(3) | a_prev(29)
```

两边是**同一组量、不同排列**（且关节序一个是 STANDARD、一个是 HA_V31）。

本模块提供：
    to_deploy_obs_hist(...)       训练侧历史 → 部署那 930 维（**逐位同构**）
    from_raw_histories(...)       原始 state/action 历史 → 同上（部署侧同款拼装）

两者对同一份物理量必须**逐位相等**，由 `tests/test_deploy_obs_alignment.py` 保证。
"""
from __future__ import annotations

import numpy as np
import torch

#: 部署侧每帧的块顺序与宽度（与 provider 的 concat 顺序逐字对应）
DEPLOY_BLOCKS: tuple[tuple[str, int], ...] = (
    ("ang_vel", 3),
    ("joint_pos_delta", 29),
    ("joint_vel", 29),
    ("last_action", 29),
    ("gravity", 3),
)
#: 10 帧历史拼起来的总宽度（= 930）
DEPLOY_HIST_DIM = sum(d for _, d in DEPLOY_BLOCKS) * 10
#: 加上 motion token 后的完整 decoder 输入宽度（= 994）
DEPLOY_OBS_DIM = 64 + DEPLOY_HIST_DIM

#: 训练侧 `OBS_TERM_DIMS_93` 的语义顺序（与 `mdp.obs()` 的 parts 顺序一致）
TRAIN_TERMS: tuple[tuple[str, int], ...] = (
    ("joint_pos_delta", 29),   # q - default_q - xi_dq0
    ("joint_vel", 29),
    ("ang_vel", 3),
    ("gravity", 3),
    ("last_action", 29),
)


def _perm(dst: str, src: str) -> np.ndarray:
    import joint_contract as JC  # noqa: PLC0415

    return np.asarray(JC.permutation(src, dst), dtype=np.int64)


def to_deploy_obs_hist(
    train_hist: torch.Tensor | np.ndarray,
    *,
    xi_dq0: torch.Tensor | np.ndarray | None = None,
    src_order: str = "standard",
    dst_order: str = "ha_v31",
    frames: int = 10,
) -> np.ndarray:
    """训练侧历史 → ``(B, 930)``（块顺序 == 部署 994 去掉 token 后的那 930）。

    ``train_hist`` 两种布局都接受：``(frames,B,W)`` 或 ``(B,frames,W)``
    （按 "哪一维等于 ``frames``" 判定，默认 10 = ``HISTORY``）。
    """
    h = np.asarray(train_hist.detach().cpu() if torch.is_tensor(train_hist) else train_hist,
                   dtype=np.float64)
    if h.ndim != 3:
        raise ValueError(f"train_hist 必须是 (frames,B,W) 或 (B,frames,W)，得到 {h.shape}")
    w = sum(d for _, d in TRAIN_TERMS)
    if h.shape[-1] != w:
        raise ValueError(f"最后一维应为 {w}（93），得到 {h.shape[-1]}")
    # 统一成 (frames, B, W)
    if h.shape[0] == frames:                 # (frames, B, W)
        pass
    elif h.shape[1] == frames:               # (B, frames, W)
        h = np.transpose(h, (1, 0, 2))
    else:
        raise ValueError(f"无法判定布局：{h.shape} 里没有一维等于 frames={frames}")
    _frames, B, _ = h.shape

    off = {}
    acc = 0
    for name, d in TRAIN_TERMS:
        off[name] = (acc, acc + d)
        acc += d

    def blk(name: str) -> np.ndarray:
        a, b = off[name]
        return h[:, :, a:b]                   # (frames, B, d)

    q_delta = blk("joint_pos_delta").copy()
    if xi_dq0 is not None:
        x = np.asarray(xi_dq0.detach().cpu() if torch.is_tensor(xi_dq0) else xi_dq0, np.float64)
        if x.shape[-1] != 29:
            raise ValueError(f"xi_dq0 最后一维应为 29，得到 {x.shape}")
        # 广播到 (frames,B,29)
        if x.ndim == 2:
            x = np.broadcast_to(x[None], q_delta.shape)
        q_delta = q_delta + x

    perm = _perm(dst_order, src_order)

    def reorder_j(x: np.ndarray) -> np.ndarray:
        return x[..., perm]

    q_delta = reorder_j(q_delta)
    joint_vel = reorder_j(blk("joint_vel"))
    last_act = reorder_j(blk("last_action"))

    deploy_blocks = {
        "ang_vel": blk("ang_vel"),
        "joint_pos_delta": q_delta,
        "joint_vel": joint_vel,
        "last_action": last_act,
        "gravity": blk("gravity"),
    }
    # 部署是"每个块内部 10 帧摊平、块间按 DEPLOY_BLOCKS 顺序拼接"
    parts = []
    for name, d in DEPLOY_BLOCKS:
        x = deploy_blocks[name]
        if x.shape[-1] != d:
            raise ValueError(f"块 {name} 宽度 {x.shape[-1]} != {d}")
        parts.append(x.transpose(1, 0, 2).reshape(B, frames * d))   # (B, frames*d)
    return np.concatenate(parts, axis=1)


def from_raw_histories(
    *,
    joint_pos: np.ndarray,          # (frames,B,29) 原始关节位置（src_order）
    joint_vel: np.ndarray,          # (frames,B,29)
    ang_vel: np.ndarray,            # (frames,B,3)  body 系
    gravity: np.ndarray,            # (frames,B,3)  = gravity_dir_from_base_quat_wxyz
    last_action: np.ndarray,        # (frames,B,29) 上一步原始动作（src_order）
    default_q: np.ndarray,          # (29,)  default 姿态（src_order）
    src_order: str = "standard",
    dst_order: str = "ha_v31",
) -> np.ndarray:
    """**部署侧同款拼装**：直接从原始量算出那 930 维（供对拍/独立实现用）。

    ``last_action`` 在部署里是"上一步 decoder 的原始输出"（HA_V31 序、未减 default_q），
    所以这里与 ``joint_pos_delta`` 的处理不同：只换序、不减 default_q。
    """
    jp = np.asarray(joint_pos, np.float64)
    jv = np.asarray(joint_vel, np.float64)
    av = np.asarray(ang_vel, np.float64)
    gv = np.asarray(gravity, np.float64)
    la = np.asarray(last_action, np.float64)
    dq = np.asarray(default_q, np.float64).reshape(-1)
    if dq.size != 29:
        raise ValueError("default_q 必须是 29 维")
    frames, B, _ = jp.shape
    perm = _perm(dst_order, src_order)

    qd = (jp - dq[None, None, :])[..., perm]
    jvd = jv[..., perm]
    lad = la[..., perm]

    blocks = {"ang_vel": av, "joint_pos_delta": qd, "joint_vel": jvd,
              "last_action": lad, "gravity": gv}
    parts = []
    for name, d in DEPLOY_BLOCKS:
        x = blocks[name]
        parts.append(x.transpose(1, 0, 2).reshape(B, frames * d))
    return np.concatenate(parts, axis=1)


def gravity_dir_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    """``quat_conj(q) * [0,0,-1]`` —— 与 provider 的 ``gravity_dir_from_base_quat_wxyz`` 同一数学。

    与训练侧 ``g_b = -R[..., 2, :]`` 的等价性由
    ``tests/test_deploy_obs_alignment.py::test_gravity_convention_matches_training`` 保证。
    """
    q = np.asarray(quat, np.float64)
    if q.shape[-1] != 4:
        raise ValueError("quat 最后一维应为 4 (wxyz)")
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-12
    w, x, y, z = w / n, x / n, y / n, z / n
    # R 的第三列取负：g_body = R^T * [0,0,-1] = -R[2, :]
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return -np.stack([r20, r21, r22], -1).astype(np.float32)
