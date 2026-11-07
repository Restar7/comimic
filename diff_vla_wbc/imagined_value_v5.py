"""v5 **Imagined Target Critic**：TD(λ) + EMA Target + Value Readiness Gate。

本模块是 `docs/TEMPORAL_AC_BPTT_V5_MAINLINE_SPEC.md` §6 的落地实现。
**默认不启用**（§5.5.3 / §6.1）：当前运行时 ``K = 20``（一次 chunk 实际执行 20 帧）、
``H = 20``（可微 WM 步数）⇒ 尾部区间 ``K − H = 0`` ⇒ ``V(y_H)`` **不可识别**，
``beta`` 必须为 0，日志必须写 ``BOOTSTRAP_NOT_IDENTIFIABLE (K==H)``。
代码路径仍然完整实现 + 单测，供备选 A（``H_actor = 10/12 < K = 20``）和
备选 B（运行时改成 ``K = 40``）打开时使用。

关键约束（逐条对应规范）
------------------------
* ``y_t`` 只允许**合法可得**字段：想象状态 ``s_t``、参考状态 ``ref_t``、``s_t − ref_t``、
  ``xi``、``hist10``（**Actor 输入的那一份**）、**实际仍会执行**的 plan token
  （``Z_{ℓ(t)..ℓ(t)+3}``，``ℓ(t)=t``，超过 ``K−1`` 的位置**不得填入**）、
  ``remaining_steps`` 与 ``t/T、t/K、K/T``。
  明确禁止：未来真实 state、未来 RGB、未执行 token 的 label。
* ``in_dim`` **由 :class:`CriticFeatureAdapter` 运行时给出**，配置文件里不写死。
* 目标：``G_t = r_t + γ m_t [(1−λ)·sg(V̄(y_{t+1})) + λ G_{t+1}]``，末步按"真终止 / 合法 bootstrap"处理。
* 只更新 ψ；``V̄ ← keep·V̄ + (1−keep)·V_ψ``，每个 critic optimizer step 一次。
* 另算一份**不依赖自身 bootstrap 的 MC 目标**（:func:`mc_returns`）用于 ``v_ev`` 校验。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn

#: 状态 / 参考 / xi 的宽度（与 `token_critic.py` 同一套合同）
STATE_DIM = 71
XI_DIM = 33


# --------------------------------------------------------------------------- 配置
@dataclass(frozen=True)
class CriticCfgV5:
    """Critic 的形状与训练超参（**形状**全由数据在运行时决定，见 ``CriticFeatureAdapter``）。"""

    state_dim: int = STATE_DIM          # WM 的 flat state
    ref_dim: int = STATE_DIM            # 同一布局的参考状态
    xi_dim: int = XI_DIM
    use_diff: bool = True               # 显式喂 ``s − ref``
    hist_frames: int = 10               # 0 = 不用历史
    hist_dim: int = 58                  # q(29)+qd(29)
    plan_dim: int = 64                  # 单个 token 的宽度
    plan_steps: int = 4                 # ``ℓ(t)..ℓ(t)+3``
    extra_dim: int = 4                  # remaining_steps(1) + t/T + t/K + K/T
    hidden: int = 512
    depth: int = 2
    gamma: float = 0.99
    td_lambda: float = 0.95
    ema_keep: float = 0.995
    huber_delta: float = 1.0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    updates_per_actor: int = 4

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- 特征
@dataclass
class CriticBatch:
    """构造 ``y_t`` 所需的**合法字段**（全部来自想象轨迹与 Actor 的输入）。

    ``ref_states`` 支持 ``(H+1,B,71)`` / ``(B,H+1,71)``；``z`` 是 Actor 看到的那段
    ``(B,T,64)`` token chunk（**未加 Δ 或加 Δ 都行，调用方给哪份就记哪份**）。
    """

    ref_states: torch.Tensor
    xi: torch.Tensor
    z: torch.Tensor
    hist10: torch.Tensor | None = None


def _as_B_H(t: torch.Tensor, b: int) -> torch.Tensor:
    """``(C,B,D)`` 与 ``(B,C,D)`` 统一成 ``(B,C,D)``。"""
    if t.ndim != 3:
        raise ValueError(f"[critic:v5] 期望 3 维张量，得到 {tuple(t.shape)}")
    if int(t.shape[1]) == b and int(t.shape[0]) > b:
        return t.permute(1, 0, 2)
    if int(t.shape[0]) == b:
        return t
    raise ValueError(f"[critic:v5] 第一维既不是 B={b} 也不是时间维：{tuple(t.shape)}")


class CriticFeatureAdapter:
    """``(imagined rollout, batch, t) → y_t``；``in_dim`` **运行时**给出。

    ``build`` 第一次被调用时会锁定 ``in_dim``，之后每个 ``t`` 都必须一致（fail-closed）。
    """

    def __init__(self, cfg: CriticCfgV5 | None = None):
        self.cfg = cfg or CriticCfgV5()
        c = self.cfg
        plan_dim = int(c.plan_steps) * int(c.plan_dim) if int(c.plan_steps) > 0 else 0
        hist_dim = int(c.hist_frames) * int(c.hist_dim) if int(c.hist_frames) > 0 else 0
        self._in_dim = (int(c.state_dim) + int(c.ref_dim)
                        + (int(c.state_dim) if c.use_diff else 0)
                        + int(c.xi_dim) + hist_dim + plan_dim + int(c.extra_dim))
        self._locked_dim: int | None = None
        self._K: int | None = None

    # ---- 元信息 ----
    @property
    def in_dim(self) -> int:
        """**运行时**特征宽度（不是配置文件里的常数）。"""
        return int(self._in_dim)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.cfg.as_dict(), in_dim=int(self.in_dim))

    # ---- 构造 ----
    def build(self, imagined, batch: CriticBatch, t: int, *,
              K: int, T: int) -> torch.Tensor:
        """第 ``t`` 步（0..L）的 ``y_t``，``(B, in_dim)``。

        ``imagined`` 需要 ``states (H+1,B,71)``；参考状态从 ``batch.ref_states`` 取同一 ``t``。
        """
        c = self.cfg
        states = imagined.states
        b = int(states.shape[1])
        if not (0 <= int(t) < int(states.shape[0])):
            raise ValueError(f"[critic:v5] t={t} 超出想象轨迹长度 {int(states.shape[0])}")
        s = states[int(t)].float()                                  # (B,71)
        if int(s.shape[-1]) != int(c.state_dim):
            raise ValueError(f"[critic:v5] state 宽度 {int(s.shape[-1])} != {c.state_dim}")
        ref_all = _as_B_H(batch.ref_states.float(), b)                # (B,C,71)
        if int(t) >= int(ref_all.shape[1]):
            raise ValueError(f"[critic:v5] ref_states 只有 {int(ref_all.shape[1])} 步，需要 t={t}")
        ref = ref_all[:, int(t)]
        parts = [s, ref]
        if c.use_diff:
            parts.append(s - ref)
        xi = batch.xi.float()
        if tuple(xi.shape) != (b, int(c.xi_dim)):
            raise ValueError(f"[critic:v5] xi 必须是 ({b},{c.xi_dim})，得到 {tuple(xi.shape)}")
        parts.append(xi)
        if int(c.hist_frames) > 0:
            if batch.hist10 is None:
                raise ValueError("[critic:v5] cfg.hist_frames>0 但 batch.hist10 为空；"
                                 "禁止用零历史冒充（v5 §4.7）")
            h = batch.hist10.float()
            if tuple(h.shape) != (b, int(c.hist_frames), int(c.hist_dim)):
                raise ValueError(f"[critic:v5] hist10 必须是 "
                                 f"({b},{c.hist_frames},{c.hist_dim})，得到 {tuple(h.shape)}")
            parts.append(h.reshape(b, -1))
        # 只填**实际仍会执行**的 plan token（t..K-1）；超过 K-1 的位置保持 0
        if int(c.plan_steps) > 0:
            z = batch.z.float()
            if z.ndim != 3 or int(z.shape[-1]) != int(c.plan_dim):
                raise ValueError(f"[critic:v5] z 必须是 (B,T,{c.plan_dim})，得到 {tuple(z.shape)}")
            step = max(1, int(c.plan_steps))
            rows = []
            for j in range(step):
                idx = int(t) + j
                if idx < int(K) and idx < int(z.shape[1]):
                    rows.append(z[:, idx])
                else:                                     # 未执行 ⇒ 不得填入
                    rows.append(torch.zeros_like(z[:, 0]))
            parts.append(torch.cat(rows, dim=-1))
        # extra：remaining_steps / t·T⁻¹ / t·K⁻¹ / K·T⁻¹
        rem = float(max(0, int(K) - 1 - int(t)))
        extra = torch.tensor(
            [rem / max(1, int(K)), float(t) / max(1, int(T)),
             float(t) / max(1, int(K)), float(K) / max(1, int(T))],
            device=s.device, dtype=s.dtype).expand(b, -1)
        if int(extra.shape[-1]) != int(c.extra_dim):
            raise ValueError(f"[critic:v5] extra_dim={c.extra_dim} 与实现 4 不一致")
        parts.append(extra)
        y = torch.cat(parts, dim=-1)
        if int(y.shape[-1]) != self.in_dim:
            raise ValueError(f"[critic:v5] y 宽度 {int(y.shape[-1])} != adapter.in_dim {self.in_dim}")
        if self._locked_dim is None:
            self._locked_dim = int(y.shape[-1])
        elif int(y.shape[-1]) != self._locked_dim:
            raise ValueError("[critic:v5] y 宽度在步与步之间变了："
                             f"{int(y.shape[-1])} vs {self._locked_dim}")
        return y


# --------------------------------------------------------------------------- 价值网络
class ImaginedValueV5(nn.Module):
    """``y_t → V(y_t)``（**原始回报**，不做缩放；SILU 的 MLP）。"""

    def __init__(self, in_dim: int, *, hidden: int = 512, depth: int = 2):
        super().__init__()
        if int(in_dim) <= 0:
            raise ValueError(f"[critic:v5] in_dim={in_dim} 非法（必须由 adapter 给出）")
        self.in_dim = int(in_dim)
        dims = [int(in_dim)] + [int(hidden)] * max(1, int(depth))
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.SiLU()]
        layers += [nn.Linear(dims[-1], 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 2 or int(y.shape[-1]) != self.in_dim:
            raise ValueError(f"[critic:v5] y 必须是 (B,{self.in_dim})，得到 {tuple(y.shape)}")
        return self.net(y.float()).squeeze(-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_value_net(adapter: CriticFeatureAdapter, cfg: CriticCfgV5 | None = None,
                    *, device="cpu") -> ImaginedValueV5:
    """用 ``adapter.in_dim``（运行时值）建网——**不读配置文件里的任何 in_dim**。"""
    c = cfg or adapter.cfg
    net = ImaginedValueV5(int(adapter.in_dim), hidden=int(c.hidden), depth=int(c.depth))
    return net.to(device=device)


# --------------------------------------------------------------------------- 目标
def td_lambda_targets(
    rewards: torch.Tensor,             # (H,B)
    dones: torch.Tensor,               # (H,B)  m_t = 1 − done_t
    values_bar: torch.Tensor,          # (H+1,B) **target** 网络输出（已 sg）
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
    end_bootstrap: torch.Tensor | None = None,   # (B,) 真终止填 0；否则 sg(V̄(y_L))
) -> torch.Tensor:
    """``G_t = r_t + γ m_t [(1−λ)·sg(V̄(y_{t+1})) + λ·G_{t+1}]``；结果全部 detach。

    ``end_bootstrap=None`` ⇒ 末步 ``G_H = sg(V̄(y_H))``（合法未终止）；
    传 ``zeros`` ⇒ 段尾是真终止（``G_H = 0``）。
    """
    if rewards.ndim != 2 or dones.shape != rewards.shape:
        raise ValueError(f"[critic:v5] rewards/dones 必须是同形 (H,B)："
                         f"{tuple(rewards.shape)} vs {tuple(dones.shape)}")
    H, B = int(rewards.shape[0]), int(rewards.shape[1])
    if tuple(values_bar.shape) != (H + 1, B):
        raise ValueError(f"[critic:v5] values_bar 必须是 ({H+1},{B})，"
                         f"得到 {tuple(values_bar.shape)}")
    if end_bootstrap is None:
        g = values_bar[H].detach()
    else:
        if tuple(end_bootstrap.shape) != (B,):
            raise ValueError(f"[critic:v5] end_bootstrap 必须是 ({B},)")
        g = end_bootstrap.detach()
    v = values_bar.detach()
    out: list[torch.Tensor] = []
    for t in range(H - 1, -1, -1):
        m = 1.0 - dones[t].detach().float()
        g = rewards[t].detach() + float(gamma) * m * (
            (1.0 - float(lam)) * v[t + 1] + float(lam) * g)
        out.append(g)
    return torch.stack(out[::-1], dim=0)


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, keep: float) -> None:
    """``V̄ ← keep·V̄ + (1−keep)·V_ψ``（参数与 buffer 都同步，逐元素原地写）。"""
    keep = float(keep)
    if not (0.0 <= keep < 1.0):
        raise ValueError(f"[critic:v5] ema_keep={keep} 必须在 [0,1)")
    tgt = dict(target.named_parameters())
    for name, p in online.named_parameters():
        if name not in tgt:
            raise ValueError(f"[critic:v5] target 缺少参数 {name!r}")
        tgt[name].mul_(keep).add_(p.detach(), alpha=1.0 - keep)
    for name, b in online.named_buffers():
        tb = dict(target.named_buffers()).get(name)
        if tb is not None:
            tb.mul_(keep).add_(b.detach(), alpha=1.0 - keep)


def mc_returns(rewards: torch.Tensor, dones: torch.Tensor | None = None, *,
               gamma: float = 0.99,
               end_bootstrap: torch.Tensor | None = None) -> torch.Tensor:
    """**不依赖自身 bootstrap** 的 MC return-to-go ``(H,B)``（仅用于校验 ``v_ev``）。"""
    H, B = int(rewards.shape[0]), int(rewards.shape[1])
    g = (torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
         if end_bootstrap is None else end_bootstrap.detach())
    out: list[torch.Tensor] = []
    for t in range(H - 1, -1, -1):
        m = 1.0 if dones is None else (1.0 - dones[t].detach().float())
        g = rewards[t].detach() + float(gamma) * m * g
        out.append(g)
    return torch.stack(out[::-1], dim=0)


# --------------------------------------------------------------------------- 门控
def bootstrap_identifiable(K: int, H: int) -> bool:
    """``K == H``（或更小）⇒ 尾部区间为 0 ⇒ ``V(y_H)`` 不可识别。"""
    return int(K) > int(H)


@dataclass
class GateResult:
    status: str                      # PASS / FAIL / INCONCLUSIVE / BLOCKED_*
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class ValueReadinessGate:
    """``β`` 的唯一入口（规范 §6.4）。

    * 合法 continuation 不存在（``K == H``）⇒ ``BLOCKED_BOOTSTRAP_NOT_IDENTIFIABLE``；
    * held-out ``v_ev < 0.5`` / MAE 不优于常量均值基线 / 方向一致性 < 0.60
      ⇒ ``FAIL``（β=0）；
    * 全过 ⇒ ``PASS``；之后 ``β`` 在 ``ramp_updates`` 内从 0 线性升到 1。
    """

    def __init__(self, *, K: int, H: int, min_v_ev: float = 0.5,
                 min_dir_consistency: float = 0.60, ramp_updates: int = 100):
        self.K, self.H = int(K), int(H)
        self.min_v_ev = float(min_v_ev)
        self.min_dir = float(min_dir_consistency)
        self.ramp_updates = int(ramp_updates)
        self.status = ("PASS_PENDING" if bootstrap_identifiable(self.K, self.H)
                       else "BLOCKED_BOOTSTRAP_NOT_IDENTIFIABLE")
        self.reasons: list[str] = ([] if bootstrap_identifiable(self.K, self.H)
                                   else [f"K==H=={self.K} ⇒ 尾部不可识别，β 必须为 0"])
        self.pass_update: int | None = None
        self.history: list[dict[str, Any]] = []

    @property
    def blocked(self) -> bool:
        return self.status.startswith("BLOCKED")

    @property
    def passed(self) -> bool:
        return self.status in ("PASS", "PASS_RAMP")

    def update(self, *, v_ev_heldout: float | None, mae: float | None,
               mae_const_baseline: float | None, dir_consistency: float | None,
               update: int | None = None) -> GateResult:
        """用**held-out（独立 clip）**的指标判定；任何一项缺失 ⇒ INCONCLUSIVE。"""
        m = dict(v_ev_heldout=v_ev_heldout, mae=mae,
                 mae_const_baseline=mae_const_baseline, dir_consistency=dir_consistency)
        metrics = {k: float(v) for k, v in m.items() if v is not None and v == v}
        self.history.append(dict(update=update, **metrics))
        if self.blocked:
            return GateResult(self.status, self.reasons, metrics)
        reasons: list[str] = []
        if v_ev_heldout is None or mae is None or mae_const_baseline is None:
            self.status = "INCONCLUSIVE"
            return GateResult(self.status, ["缺 held-out 指标（VALUE_GATE_INCONCLUSIVE）"], metrics)
        if float(v_ev_heldout) < self.min_v_ev:
            reasons.append(f"v_ev={float(v_ev_heldout):.3f} < {self.min_v_ev}")
        if float(mae) >= float(mae_const_baseline):
            reasons.append(f"MAE={float(mae):.4f} 不优于常量均值基线 {float(mae_const_baseline):.4f}")
        if dir_consistency is not None and float(dir_consistency) < self.min_dir:
            reasons.append(f"dir_consistency={float(dir_consistency):.3f} < {self.min_dir}")
        if reasons:
            self.status = "FAIL"
            self.reasons = reasons
            return GateResult(self.status, reasons, metrics)
        if self.status == "PASS_RAMP":
            return GateResult(self.status, [], metrics)
        self.status = "PASS"
        self.pass_update = int(update or 0)
        self.reasons = []
        return GateResult(self.status, [], metrics)

    def beta_at(self, update: int) -> float:
        """PASS 之后在 ``ramp_updates`` 个 actor update 内线性 0 → 1；未 PASS 恒 0。"""
        if not self.passed or self.pass_update is None:
            return 0.0
        f = (int(update) - int(self.pass_update)) / max(1, self.ramp_updates)
        return float(min(1.0, max(0.0, f)))

    def as_dict(self) -> dict[str, Any]:
        return dict(status=self.status, blocked=bool(self.blocked), K=self.K, H=self.H,
                    reasons=list(self.reasons), pass_update=self.pass_update,
                    history=list(self.history))


__all__ = [
    "CriticBatch",
    "CriticCfgV5",
    "CriticFeatureAdapter",
    "GateResult",
    "ImaginedValueV5",
    "ValueReadinessGate",
    "bootstrap_identifiable",
    "build_value_net",
    "ema_update",
    "mc_returns",
    "td_lambda_targets",
]
