"""域随机化（DR）：把「闭环推理时可能遇到的偏差」变成一个**分布**，而不是固定一个点。

动机（用户提出，正确）：固定 25 cm 扰动只是 DR 分布里的一个点，而且偏。
要学"闭环里可能遇到的情况"，就应该按**实测到的偏差分布**随机化起点 /
过程扰动 / 观测噪声，并且**分层抽样**（nominal / mid / hard），否则 batch 全是
"已经摔了"的窗口，回报方差爆掉、critic 学不动。

三层 DR（本模块实现前两层，观测噪声在入口里可选）：

    start   : root xy / yaw / joint q / joint qd   ← burnin 的扰动幅度
    process : rollout 内 root 线/角速度推扰          ← rollout.push_std/push_ang
    obs     : VLA 输入的 46 维状态噪声（可选）        ← 入口里加在 processor 输出上

分层（每个 batch 内按比例抽）：
    nominal : 扰动 = 0            （保证不遗忘标称行为）
    mid     : U(0, 0.6] × max
    hard    : U(0.6, 1.0] × max
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

BUCKET_NAMES = ("nominal", "mid", "hard")


@dataclass
class DRConfig:
    enabled: bool = True
    p_nominal: float = 0.25
    p_mid: float = 0.45            # 剩下的是 hard
    xy_max: float = 0.20           # m
    yaw_max: float = 0.25          # rad
    dq_max: float = 0.05           # rad（单关节）
    dqd_max: float = 0.30          # rad/s
    mid_hi: float = 0.6            # mid 桶的幅度上界（相对 max）
    push_std: float = 0.03         # rollout 内线速度推扰（m/s）
    push_ang: float = 0.05         # rollout 内角速度推扰（rad/s）
    obs_noise: float = 0.0         # VLA 观测噪声（46 维状态的统一噪声，0=关）


def sample_batch(n: int, cfg: DRConfig, generator: torch.Generator | None = None):
    """→ ``(scales (n,4), buckets (n,))``，scales 已乘上各自 max。"""
    if not cfg.enabled:
        return torch.zeros(n, 4), torch.zeros(n, dtype=torch.long)
    u = torch.rand(n, generator=generator)
    b = torch.zeros(n, dtype=torch.long)
    m_mid = (u >= cfg.p_nominal) & (u < cfg.p_nominal + cfg.p_mid)
    b[m_mid] = 1
    b[u >= cfg.p_nominal + cfg.p_mid] = 2
    amp = torch.zeros(n)
    amp[b == 1] = torch.rand(int(m_mid.sum()), generator=generator) * cfg.mid_hi
    hard = b == 2
    amp[hard] = cfg.mid_hi + torch.rand(int(hard.sum()), generator=generator) * (1.0 - cfg.mid_hi)
    scales = torch.stack([
        amp * cfg.xy_max, amp * cfg.yaw_max, amp * cfg.dq_max, amp * cfg.dqd_max
    ], dim=1)
    return scales, b


@torch.no_grad()
def estimate_ranges(data: dict, npz_path=None) -> dict:
    """从真转移数据里估「命令 vs 执行」的偏差量级（只用于报告/定 DR 量级，不参与训练）。

    ``data`` 来自 ``scripts/build_real_transitions.py``：``actions`` 是命令推出的动作，
    ``states`` 是执行状态；``q_exec = states[7:36]``，``q_cmd = a*scale + default_q``。
    """
    out = {}
    try:
        st = data["states"]
        act = data["actions"]
        n = int(st.shape[0])
        idx = torch.arange(0, n, max(1, n // 200000))       # 子采样：601k 帧上 quantile 会炸
        st, act = st[idx], act[idx]
        out["root_lin_std"] = float(st[:, 36:39].std())
        out["action_abs_mean"] = float(act.abs().mean())
        qd = st[:, 42:71]
        out["qd_abs_p95"] = float(qd.abs().reshape(-1).quantile(0.95))
        row = st[:, 7:36]
        out["q_range_p95"] = float((row.max(0).values - row.min(0).values).quantile(0.95))
        out["samples"] = int(idx.numel())
    except Exception as exc:  # 报告失败不能挡训练
        out["error"] = str(exc)
    return out


__all__ = ["BUCKET_NAMES", "DRConfig", "estimate_ranges", "sample_batch"]
