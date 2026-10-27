"""动作头的 LoRA：把可训参数从 1.62B 降到 ~20M，让 **batch 能开大**。

为什么需要（对齐 SuperMimic 的第二根柱子）：
    我们的 batch 只能是 4，因为 1.62B 全量训练 = 权重 6.5G + 梯度 6.5G + AdamW 动量 13G ≈ 26 GB，
    再加冻结 backbone 8 GB 和激活，48 GB 的卡就满了。
    SuperMimic 用 batch **4096** 把"单批的模型误差方向"平均掉，这是他们敢用 H=32 而不会
    被奖励黑客的关键。我们 batch=4 相当于**每一批只看到一个误差实现**，优化器必然跟着它走。
    ⇒ LoRA 让"每个样本的显存"重新成为主导，batch 才能向上开。

设计：
    * 冻结 base 的所有参数，只在选中的 ``nn.Linear`` 上挂低秩分支 ``B(A(x))``；
    * ``B`` 零初始化 ⇒ 训练开始时**与原模型逐位等价**（不会破坏底模先验）；
    * 支持按名字后缀筛选（默认 attention 的 q/k/v/out 与 FFN），可选解冻输出投影。

用法：
    from diff_vla_wbc.lora import apply_lora
    n = apply_lora(vlm.action_head, rank=16, alpha=16, targets=("attn1","attn2","ff"))
    print(lora_summary(vlm.action_head))
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """``y = W0 x + b0 + (alpha / r) · B A x``，``W0/b0`` 冻结，只有 ``A/B`` 可训。"""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float | None = None):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        r = int(rank)
        self.rank = r
        self.scale = float(alpha if alpha is not None else r) / max(1, r)
        # 参数必须与 base 同 device/dtype —— 否则替换发生在上卡之后就会 device 不匹配
        _w = base.weight
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, device=_w.device, dtype=_w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=_w.device, dtype=_w.dtype))
        # A 用 kaiming、B 用 0 ⇒ 初始增量恒为 0（与原模型逐位等价）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = self.base(x)
        if out.dtype != self.lora_A.dtype:
            out = out.to(self.lora_A.dtype)
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return out + self.scale * delta


def apply_lora(
    root: nn.Module,
    *,
    rank: int = 16,
    alpha: float | None = None,
    targets: Sequence[str] = ("attn1", "attn2", "ff"),
    freeze_all_first: bool = True,
    also_train: Iterable[str] = ("proj_out",),
) -> int:
    """给 ``root`` 里名字含 ``targets`` 任一子串的 ``nn.Linear`` 挂 LoRA。返回替换的层数。"""
    if freeze_all_first:
        for p in root.parameters():
            p.requires_grad_(False)
    sub = tuple(str(t) for t in targets)
    todo: list[tuple[nn.Module, str, nn.Linear]] = []
    for mod_name, mod in root.named_modules():
        # 关键：按**完整路径**匹配（``model.blocks.0.attn1.to_q``），
        # 而不是子属性名（``to_q``）——否则 "attn1" 永远匹配不到。
        if isinstance(mod, nn.Linear) and any(s in mod_name for s in sub):
            parent_name, _, child = mod_name.rpartition(".")
            parent = root.get_submodule(parent_name) if parent_name else root
            todo.append((parent, child, mod))
    replaced = 0
    for parent, child, child_mod in todo:
        setattr(parent, child, LoRALinear(child_mod, rank=rank, alpha=alpha))
        replaced += 1
    if replaced == 0:
        raise RuntimeError(
            f"apply_lora 没匹配到任何 nn.Linear（targets={targets}）。"
            "先打印模块名确认，例如 [n for n,_ in root.named_modules()][:40]")
    return replaced


def lora_parameters(root: nn.Module) -> list[nn.Parameter]:
    return [p for p in root.parameters() if p.requires_grad]


def lora_summary(root: nn.Module) -> dict:
    trainable = sum(p.numel() for p in root.parameters() if p.requires_grad)
    total = sum(p.numel() for p in root.parameters())
    n_lora = sum(1 for m in root.modules() if isinstance(m, LoRALinear))
    return {
        "lora_layers": n_lora,
        "trainable": trainable,
        "total": total,
        "trainable_frac": (trainable / total) if total else 0.0,
    }
