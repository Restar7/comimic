"""Temporal-Conditioned Δ Actor（借鉴 v5 任务书的**结构**，不引入 Critic/BPTT）。

结构（与 `COMIMIC v5` 文档 §3 一致，尺寸严格固定）：

    Z (B,T,64) ──Linear(64→128) + PE128(t) ──TransformerEncoder(1 block, 4 heads, FFN256, pre-LN)
                                                   │
                                            c_t (B,T,128)
    H10 (B,10,58) ──GRU(128, 1 layer) ──► h (B,128)
    s0 (B,64)                                  │
    PE16(t) (B,T,16)                           │
                                               ▼
              x_t = [ z_t(64) ; c_t(128) ; h(128) ; s0(64) ; PE16(t)(16) ] = 400
                                               ▼
                   MLP 400→1024→1024→64 （SiLU，末层 N(0,4e-4) 初始化）
                                               ▼
                    Δ_t = mask_t · apply_norm_cap(raw_t, cap)
                    z'_t = z_t + Δ_t

**为什么这样**：`c_t` 让每个位置看到整段计划；`z_t` 是被修正的局部 token；`PE16(t)` 明确位置；
`h` 提供 10 帧真实历史。即使同一 chunk 内 `state/history` 不变，网络仍能输出**随 t 变化**的修正。

**部署含义**：本模块需要 (整段 chunk, 10 帧历史, state64) ——
- 整段 chunk 与 state64：服务端已有；
- **10 帧历史：服务端拿不到**（它只收 state64）⇒ 本模块必须在 **provider 侧**（decoder 旁边）执行，
  那里有机器人每步的 q/qd。详见 `docs/DELTA_TOKEN_RUNTIME_CONTRACT.md`。
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .delta_token import apply_norm_cap

TOKEN_DIM = 64
HIST_DIM = 58            # q29 + qd29
STATE_DIM = 64
CTX_DIM = 128            # chunk context via attention
HIDDEN = 128             # history GRU hidden
PE_CTX = 128
PE_POS = 16
MLP_IN = TOKEN_DIM + CTX_DIM + HIDDEN + STATE_DIM + PE_POS   # 400


def positional_encoding(n: int, dim: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """固定正弦位置编码 ``(n, dim)``（无参数 ⇒ 训练/部署完全一致）。"""
    if dim % 2:
        raise ValueError("dim 必须为偶数")
    pos = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(0, dim, 2, device=device, dtype=dtype).unsqueeze(0)
    div = torch.exp(-math.log(10000.0) * i / dim)
    pe = torch.zeros(n, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TemporalDeltaActor(nn.Module):
    def __init__(self, *, token_dim: int = TOKEN_DIM, hist_dim: int = HIST_DIM,
                 state_dim: int = STATE_DIM, ctx_dim: int = CTX_DIM,
                 gru_hidden: int = HIDDEN, heads: int = 4, ffn: int = 256,
                 hidden: int = 1024, depth: int = 2, init_std: float = 4e-4,
                 cap: float = 0.08) -> None:
        super().__init__()
        self.token_dim, self.hist_dim, self.state_dim = token_dim, hist_dim, state_dim
        self.ctx_dim, self.gru_hidden, self.cap = ctx_dim, gru_hidden, float(cap)
        self.token_proj = nn.Linear(token_dim, ctx_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=ctx_dim, nhead=heads, dim_feedforward=ffn,
            activation="gelu", batch_first=True, norm_first=True, dropout=0.0)
        self.chunk_attn = nn.TransformerEncoder(layer, num_layers=1)
        self.hist_gru = nn.GRU(hist_dim, gru_hidden, num_layers=1, batch_first=True)
        d = MLP_IN if (token_dim, ctx_dim, gru_hidden, state_dim, PE_POS) == (
            TOKEN_DIM, CTX_DIM, HIDDEN, STATE_DIM, PE_POS) else (
            token_dim + ctx_dim + gru_hidden + state_dim + PE_POS)
        self.mlp_in = d
        layers: list[nn.Module] = []
        for _ in range(max(1, int(depth))):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        last = nn.Linear(d, token_dim)
        nn.init.normal_(last.weight, std=float(init_std or 0.0))
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.head = nn.Sequential(*layers)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, z: torch.Tensor, hist: torch.Tensor, state64: torch.Tensor,
                *, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``z (B,T,64)`` · ``hist (B,10,58)`` · ``state64 (B,64)`` → Δ ``(B,T,64)``。"""
        B, T, _ = z.shape
        if hist.shape[1:] != (10, self.hist_dim):
            raise ValueError(f"hist 形状应为 (B,10,{self.hist_dim})，得到 {tuple(hist.shape)}")
        if state64.shape[-1] != self.state_dim:
            raise ValueError(f"state64 应为 {self.state_dim} 维，得到 {state64.shape[-1]}")
        c = self.chunk_attn(self.token_proj(z) + positional_encoding(T, self.ctx_dim, z.device, z.dtype))
        h = self.hist_gru(hist.float())[0][:, -1]                       # (B,128)
        pe = positional_encoding(T, PE_POS, z.device, z.dtype).unsqueeze(0).expand(B, T, PE_POS)
        x = torch.cat([z, c, h.unsqueeze(1).expand(B, T, self.gru_hidden),
                       state64.unsqueeze(1).expand(B, T, self.state_dim), pe], dim=-1)
        raw = self.head(x)                                              # (B,T,64)
        d = apply_norm_cap(raw, self.cap)
        if mask is not None:
            d = d * mask.to(d.dtype).unsqueeze(-1)
        return d


__all__ = ["TemporalDeltaActor", "positional_encoding", "MLP_IN", "HIST_DIM"]
