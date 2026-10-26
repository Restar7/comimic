"""S3.3 / J4：单次采样 + **GR00T 原生 flow-matching 损失**做 BC（取代 token_anchor / hand MSE）。

重设计 §3.5 的计算图::

    obs → GR00T(backbone 冻结) → head → x_norm (B,40,132)     ← 每步只采样一次，K = 服务端 fm_steps
    x_raw = codec.decode(x_norm)                              ← 可微反归一化（J2）
    token = x_raw[..., :64] → WBC(冻结) → WM(冻结) → reward
    L_bc = action_head(bb, {action=encode(port⊕hand), mask})  ← FM 损失（J4）

为什么不能对采样结果做 MSE（原 ``token_anchor`` / hand 分支）：
对一次 FM 采样做 MSE，在多模态条件（导航这种"去左/去右都行"）下最优解是**条件均值**，
行为就被"平均"成在目标之间打转；而且 token 与手各采样一次（噪声不同、多付一次前向+反传）。
GR00T 自己的训练目标不是"复现某次采样"，而是对 (noise, t) 的期望速度场回归 —— 用它做 BC
才是这套生成式 head 的正确极大似然目标。

实现细节（踩过的坑，别改）：

* ``action_head._encode_features`` / ``forward`` 都会**原地改写**传入的 ``BatchFeature``
  （``process_backbone_output`` 写回 ``backbone_features``，``forward`` 里
  ``action_input.state = ...view(...)``），所以采样与 BC 必须各拿一份浅拷贝；
  若两次复用同一份，vlln / vl_self_attention 会被算两遍（数值错，且不报错）。
* 目标是**归一化空间**的 78 维（64 token + 7+7 手），pad 到 ``action_head.action_dim``（132），
  ``action_mask`` 只在 78 维上为 1 —— 与 GR00T 数据侧的 collator 完全一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

try:  # transformers is optional only for the pure-numpy paths
    from transformers.feature_extraction_utils import BatchFeature
except Exception:  # pragma: no cover
    BatchFeature = dict  # type: ignore[assignment,misc]

#: codec 覆盖的 78 维（``motion_token(64) + left_hand(7) + right_hand(7)``）
BC_DIM = 78
TOKEN_DIM = 64
HAND_DIM = 14


# --------------------------------------------------------------------------- 输入拷贝


def copy_action_input(action_input: Any) -> Any:
    """浅拷贝 GR00T 的 ``action_input``（新 BatchFeature + 新 dict，张量共享）。"""
    data = dict(action_input)
    if "state" in data and torch.is_tensor(data["state"]):
        data["state"] = data["state"].clone()
    if isinstance(action_input, BatchFeature):
        return BatchFeature(data=data)
    return data


def copy_backbone_output(backbone_output: Any) -> Any:
    """浅拷贝 backbone 输出（``process_backbone_output`` 会原地写回字典）。"""
    data = dict(backbone_output)
    if isinstance(backbone_output, BatchFeature):
        return BatchFeature(data=data)
    return data


# --------------------------------------------------------------------------- BC 目标


def bc_target(
    codec,
    port_tokens: torch.Tensor,
    hands: torch.Tensor,
    *,
    head_dim: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(port_tok (B,T,64), hands (B,T,14))`` → ``(target (B,T,head_dim), mask)``。

    目标在**归一化空间**（``codec.encode``，min-max → [-1,1] + clamp），因为 GR00T 的
    flow-matching 头就是在归一化空间里回归速度场；直接喂原始 token 会重演 J2 的尺度错配。
    """
    if port_tokens.shape[-1] != TOKEN_DIM:
        raise ValueError(f"port tokens must be (B,T,{TOKEN_DIM}), got {tuple(port_tokens.shape)}")
    if hands.shape[-1] != HAND_DIM:
        raise ValueError(f"hand labels must be (B,T,{HAND_DIM}), got {tuple(hands.shape)}")
    t = min(port_tokens.shape[1], hands.shape[1])
    raw = torch.cat((port_tokens[:, :t].float(), hands[:, :t].float()), dim=-1)
    norm = codec.encode(raw)
    dim = int(norm.shape[-1])
    if dim != BC_DIM:
        raise ValueError(f"codec dim {dim} != {BC_DIM}")
    device = device or port_tokens.device
    dtype = dtype or port_tokens.dtype
    tgt = torch.zeros(norm.shape[0], t, int(head_dim), device=device, dtype=dtype)
    mask = torch.zeros_like(tgt)
    tgt[..., :dim] = norm.to(dtype)
    mask[..., :dim] = 1.0
    return tgt, mask


def make_bc_action_input(action_input: Any, target: torch.Tensor, mask: torch.Tensor) -> Any:
    """在 ``action_input`` 的浅拷贝里塞进 ``action`` / ``action_mask``。"""
    out = copy_action_input(action_input)
    out["action"] = target
    out["action_mask"] = mask
    return out


# --------------------------------------------------------------------------- 单次采样


def sample_once_with_grad(
    action_head,
    backbone_output: Any,
    action_input: Any,
    *,
    initial_noise: torch.Tensor | None = None,
    num_steps: int,
    grad_backend: str = "pathwise",
    truncated_tail: int = 1,
) -> torch.Tensor:
    """一次（有梯度的）FM 采样，返回归一化空间的 ``(B,horizon,head_dim)``。"""
    from .gr00t_fm import sample_fm_with_grad  # noqa: PLC0415

    return sample_fm_with_grad(
        action_head,
        copy_backbone_output(backbone_output),
        copy_action_input(action_input),
        initial_noise=initial_noise,
        num_steps=int(num_steps),
        grad_backend=grad_backend,
        truncated_tail=truncated_tail,
    )


def fm_bc_loss(
    action_head,
    backbone_output: Any,
    action_input: Any,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """GR00T 原生 flow-matching 损失（``action_head.forward`` 的 ``loss``）。"""
    out = action_head(
        copy_backbone_output(backbone_output),
        make_bc_action_input(action_input, target, mask),
    )
    loss = out["loss"] if isinstance(out, dict) else out.loss
    return loss


@dataclass
class BCLosses:
    total: torch.Tensor
    fm: torch.Tensor


def grad_norm_ratio(
    loss_a: torch.Tensor,
    loss_b: torch.Tensor,
    params: Iterable[torch.Tensor],
) -> float:
    """``‖∇loss_a‖ / ‖∇loss_b‖`` 在同一组参数上（只用于打印，不写回 ``.grad``）。

    两个损失共享同一次采样图，所以先算 ``loss_a`` 的梯度并 ``retain_graph``，再算 ``loss_b``。
    只在打印步调用（每 50 步），额外开销 = 2 次反向，可忽略。
    """
    plist = [p for p in params if p.requires_grad]
    if not plist:
        return float("nan")
    ga = torch.autograd.grad(loss_a, plist, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(loss_b, plist, retain_graph=True, allow_unused=True)
    na = torch.sqrt(sum((g.detach().float() ** 2).sum() for g in ga if g is not None))
    nb = torch.sqrt(sum((g.detach().float() ** 2).sum() for g in gb if g is not None))
    return float(na / nb.clamp_min(1e-12))


def grad_ratio_parameters(action_head) -> list[torch.Tensor]:
    """取一组**代表性**参数算梯度比：投影头 + 一个小 DiT 子集（省显存/时间）。"""
    picked: list[torch.Tensor] = []
    prefixes = ("action_decoder", "action_encoder", "state_encoder", "vlln", "vl_self_attention")
    for name, p in action_head.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(prefixes):
            picked.append(p)
    if not picked:  # 全冻结的 head（不该发生）→ 退回前 8 个参数
        picked = [p for _, p in list(action_head.named_parameters())[:8] if p.requires_grad]
    return picked


__all__ = [
    "BC_DIM",
    "BCLosses",
    "HAND_DIM",
    "TOKEN_DIM",
    "bc_target",
    "copy_action_input",
    "copy_backbone_output",
    "fm_bc_loss",
    "grad_norm_ratio",
    "grad_ratio_parameters",
    "make_bc_action_input",
    "sample_once_with_grad",
]
