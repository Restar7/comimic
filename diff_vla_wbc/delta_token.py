"""命令层残差（token residual）：``token' = token + g(token, 偏差特征)``。

背景（2026-11-06，四轮实测之后的结论）
------------------------------------
- 漂移是**命令层**的病：底层 WBC 跟踪 token 的误差只有几 cm（exc 起点偏差 p50 4.35cm），
  但"偏了以后命令不纠正"会让偏差继续累积。
- `LoRA 微调 / state_extra / 反常量项` 三条路都是"**给更多信息，指望模型自己学**"。
  实测：残差/均值 0.098 → 0.164（阈值 0.5），而 残差/噪声 3.79 → 4.00 **没变**
  ⇒ 学到的那点"变化"主要是采样噪声，不是状态依赖。
- 这里把"状态相关"写进**结构**：`g` 的输入就是偏差特征，同一个 token 在不同偏差下
  **必然**得到不同命令。末层权重初始化为 0 ⇒ 第 0 步与底模**逐位相同**（纯增量）。

部署口径（**零客户端改动**）
--------------------------
服务器本来就在收 64 维 state：

    state64 = [ root_rot6d(6) | dof_pos_29(6:35) | dof_vel_29(35:64) ]

`g` 只用其中的 ``[0:6]``（朝向，heading-canonical，与训练侧 rotlocal6 同口径）
和 ``[35:64]``（关节速度；这 29 维当前 state46 **根本没用**，等于白送的额外信号）。
"""

from __future__ import annotations

import torch
from torch import nn

#: 底模 token 宽度（action head 输出的前 64 维）
TOKEN_DIM = 64
#: 偏差特征宽度 = rot6d(6) + dof_vel(29)（旧口径，信息不全：**没有 dof_pos**）
FEAT_DIM = 6 + 29
#: 部署 ``state64`` 的完整宽度 = root_rot6d(6) + dof_pos(29) + dof_vel(29)。
#: ★ 2026-11-06 实测：只用 FEAT_DIM 时"状态→ĝ"的 cos 只有 0.24，喂全 64 维/71 维可到 0.39~0.47
#:   （`docs/GRAD_OPTIMUM_WHAT_IT_DEPENDS_ON_20261106.md` §2）。服务端本来就收到这 64 维 ⇒ 白送。
STATE64_DIM = 64


def apply_norm_cap(delta: torch.Tensor, cap: float) -> torch.Tensor:
    """对 ``‖Δtoken‖`` 做**软**上限（tanh 收缩，保留梯度）。训练/部署**必须**都走这里。

    ``cap <= 0`` ⇒ 恒等。小范数时近似恒等、接近 cap 时饱和，绝不硬截断
    （硬截断会让超界样本梯度为 0，正好是最需要修的那批）。
    """
    cap = float(cap or 0.0)
    if cap <= 0.0:
        return delta
    n = delta.norm(dim=-1, keepdim=True) + 1e-9
    return delta * (torch.tanh(n / cap) * cap / n)


class DeltaTokenNet(nn.Module):
    """``(token, feats) → Δtoken``。末层零初始化 ⇒ 初始输出恒为 0。

    **为什么把 token 也当输入**：token 本身就是参考运动的 encoder latent，天然带相位；
    同一个偏差在支撑相/摆动相该补的东西不同，只给偏差学不出相位条件。
    """

    def __init__(
        self,
        token_dim: int = TOKEN_DIM,
        feat_dim: int = FEAT_DIM,
        hidden: int = 1024,
        depth: int = 2,
        init_std: float = 4e-4,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.feat_dim = int(feat_dim)
        layers: list[nn.Module] = []
        d = self.token_dim + self.feat_dim
        for _ in range(max(1, int(depth))):
            layers += [nn.Linear(d, int(hidden)), nn.SiLU()]
            d = int(hidden)
        last = nn.Linear(d, self.token_dim)
        # ★ 极小初始化（不是严格 0）：`dL/dW_前面的层 ∝ W_out`，严格 0 会让**前两层在第 0 步
        #   完全没有梯度**（实测 |grad|=0），要等输出层长起来才"解锁"。
        #   用 ~1e-4 的 std ⇒ 初始 ‖Δtoken‖ ≈ 1e-2（底模 token 分量 ~0.1 的量级之下），
        #   等价于"从底模出发"，但三层从第 0 步就一起可训。
        #   `init_std=0` 可退回严格零初始化（做 A/B 用）。
        _std = float(init_std or 0.0)
        if _std > 0.0:
            nn.init.normal_(last.weight, std=_std)
        else:
            nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, token: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """``token`` (..., 64) / ``feats`` (..., 35) → Δtoken (..., 64)。"""
        if feats.shape[-1] != self.feat_dim:
            raise ValueError(f"[delta-token] feats 宽度 {feats.shape[-1]} != {self.feat_dim}")
        x = torch.cat([token.to(torch.float32), feats.to(torch.float32)], dim=-1)
        return self.net(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["DeltaTokenNet", "apply_norm_cap", "TOKEN_DIM", "FEAT_DIM", "STATE64_DIM"]
