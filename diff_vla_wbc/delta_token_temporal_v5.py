"""v5 Actor：**逐帧 Δtoken**（Chunk Self-Attention + History GRU + 原 1024×2 头）。

设计约束（来自 `docs/TEMPORAL_AC_BPTT_V5_IMPLEMENTATION_PLAN.md`）
--------------------------------------------------------------
* 输入只用**部署可获得**的量：
    - ``Z (B,T,64)``   冻结官方 VLA 本次生成的整段 motion token chunk；
    - ``state64 (B,64)`` chunk **决策时刻**的观测 ``[rot6d6 | q29 | qd29]``（HA_V31 序）；
    - ``hist (B,10,58)`` 过去 10 个控制 tick 的 ``q29+qd29``（最老→最新）。
      ★ 已逐位验证：obs93 的 ``q/qd`` 块与 ``state64`` 的 ``dof_pos/dof_vel`` **同一套关节序**，
        只差一个 ``−SONIC_default`` 常数偏移（``max|Δ|=0.0``，见计划 §1.3）。
* 输出 ``Δ (B,T,64)``：**每帧一个**，``t ≥ K`` 的位置恒为 0（``K=min(H,T)``，默认 20）。
  这与部署端"一次算整段、按帧加"的语义一致；**不是**把某个 Δ 广播到整段。
* 骨架保持 **1024×2**（``400→1024→1024→64``，SiLU），末层 ``N(0,4e-4)`` + bias 0 ⇒ 第 0 步 ≈ Base。
* 参数量（T=40）：``8,320 + 5,120 + 132,480 + 72,192 + 1,525,824 + 640 = 1,744,576``。

为什么保留 ``z_t`` 与 ``PE16(t)``：``c_t`` 已含全 chunk 计划，但 ``z_t`` 是**本帧要改的那个 token**，
``PE16`` 精确标识位置；三者同时给，网络才能在"同一 chunk 同一 state 同一 history"下输出随 t 变化的修正。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .delta_token import TOKEN_DIM, apply_norm_cap

#: 历史特征宽度 = q(29) + qd(29)（**不含** a_prev：训练侧是参考库推的，部署侧是实际动作）
HIST_DIM = 58


@dataclass(frozen=True)
class TemporalActorCfg:
    tok_dim: int = TOKEN_DIM          # 64
    d_model: int = 128               # token_proj / chunk 上下文宽度
    n_heads: int = 4
    ffn: int = 256
    n_blocks: int = 1
    hist_frames: int = 10
    hist_dim: int = HIST_DIM         # 58
    state_dim: int = 64
    pe_pos_dim: int = 16
    hidden: int = 1024
    depth: int = 2
    init_std: float = 4e-4
    cap: float = 0.08
    chunk_len: int = 40              # PE 表长度（T）
    use_history: bool = True         # False ⇒ 独立 no-history 架构（不许用假历史冒充）

    def as_dict(self) -> dict:
        return asdict(self)


class TemporalDeltaActorV5(nn.Module):
    """``(Z, state64, hist) → Δ (B,T,64)``；``mask`` 里为 0 的位置输出严格为 0。"""

    def __init__(self, cfg: TemporalActorCfg | None = None):
        super().__init__()
        self.cfg = cfg or TemporalActorCfg()
        c = self.cfg
        self.token_proj = nn.Linear(int(c.tok_dim), int(c.d_model))
        self.pe_chunk = nn.Parameter(torch.zeros(int(c.chunk_len), int(c.d_model)))
        nn.init.normal_(self.pe_chunk, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=int(c.d_model), nhead=int(c.n_heads), dim_feedforward=int(c.ffn),
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.chunk_encoder = nn.TransformerEncoder(layer, num_layers=int(c.n_blocks))
        self.chunk_norm = nn.LayerNorm(int(c.d_model))
        self.use_history = bool(c.use_history)
        if self.use_history:
            self.hist_gru = nn.GRU(int(c.hist_dim), int(c.d_model), num_layers=1,
                                   batch_first=True)
        self.pe_pos = nn.Parameter(torch.zeros(int(c.chunk_len), int(c.pe_pos_dim)))
        nn.init.normal_(self.pe_pos, std=0.02)
        d_in = (int(c.tok_dim) + int(c.d_model) + (int(c.d_model) if self.use_history else 0)
                + int(c.state_dim) + int(c.pe_pos_dim))
        self.d_in = d_in
        layers: list[nn.Module] = []
        d = d_in
        for _ in range(max(1, int(c.depth))):
            layers += [nn.Linear(d, int(c.hidden)), nn.SiLU()]
            d = int(c.hidden)
        last = nn.Linear(d, int(c.tok_dim))
        _std = float(c.init_std or 0.0)
        if _std > 0:
            nn.init.normal_(last.weight, std=_std)
        else:
            nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.head = nn.Sequential(*layers)

    # ------------------------------------------------------------------ forward
    def forward(self, Z: torch.Tensor, state64: torch.Tensor,
                hist: torch.Tensor | None = None, mask: torch.Tensor | None = None,
                *, cap: float | None = None,
                return_raw: bool = False) -> torch.Tensor:
        """``Z (B,T,64)`` / ``state64 (B,64)`` / ``hist (B,10,58)`` / ``mask (B,T)`` → ``Δ (B,T,64)``。

        注意 ``mask`` 的语义：``1`` = 该帧会被执行且需要修正（``t<K``），``0`` = **保持不动**。
        """
        c = self.cfg
        if Z.ndim != 3 or int(Z.shape[-1]) != int(c.tok_dim):
            raise ValueError(f"[v5] Z 必须是 (B,T,{c.tok_dim})，得到 {tuple(Z.shape)}")
        B, T, _ = Z.shape
        if T > int(c.chunk_len):
            raise ValueError(f"[v5] T={T} > chunk_len={c.chunk_len}（PE 表不够）")
        # chunk 上下文（self-attention 允许每帧看到整段计划）
        hz = self.token_proj(Z) + self.pe_chunk[:T][None]
        c_t = self.chunk_norm(self.chunk_encoder(hz))                      # (B,T,128)
        parts = [Z, c_t]
        if self.use_history:
            if hist is None:
                raise ValueError("[v5] use_history=True 但没给 hist；禁止用零历史冒充")
            if hist.ndim != 3 or int(hist.shape[-1]) != int(c.hist_dim):
                raise ValueError(f"[v5] hist 必须是 (B,{c.hist_frames},{c.hist_dim})，"
                                 f"得到 {tuple(hist.shape)}")
            _, h_n = self.hist_gru(hist.float())                           # (1,B,128)
            parts.append(h_n[-1][:, None, :].expand(B, T, -1))
        parts.append(state64[:, None, :].expand(B, T, -1))
        parts.append(self.pe_pos[:T][None].expand(B, -1, -1))
        x = torch.cat(parts, dim=-1)
        if int(x.shape[-1]) != self.d_in:
            raise ValueError(f"[v5] 拼接宽度 {int(x.shape[-1])} != d_in {self.d_in}")
        u = self.head(x)
        d = apply_norm_cap(u, float(c.cap if cap is None else cap))
        if mask is not None:
            if mask.shape != (B, T):
                raise ValueError(f"[v5] mask 必须是 ({B},{T})，得到 {tuple(mask.shape)}")
            d = d * mask[..., None].to(d.dtype)
        # `return_raw`：给单测检查"mask_t=0 处 ∂L/∂u = 0"（∂L/∂(masked d) 当然等于上游梯度，
        # 因为它已经是输出；真正的 mask 语义要在**掩码前**的张量上验）
        if return_raw:
            return d, u
        return d

    # ------------------------------------------------------------------ 元信息
    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_groups(self, head_lr: float, enc_lr: float) -> list[dict]:
        """两档学习率：1024×2 头 / 其余编码器（计划 §7：head 3e-4、encoder 1e-4）。"""
        enc_names = {"token_proj", "pe_chunk", "chunk_encoder", "chunk_norm", "hist_gru", "pe_pos"}
        enc, head = [], []
        for n_, p_ in self.named_parameters():
            (enc if n_.split(".")[0] in enc_names else head).append(p_)
        return [dict(params=enc, lr=float(enc_lr), name="encoder"),
                dict(params=head, lr=float(head_lr), name="head")]


class PositionTemplateActor(nn.Module):
    """**逐帧位置常量模板** `c_t (T,64)`（不看任何输入）——规范里的"防伪对照" A2。

    用途：v5 的验收要求"条件化 Actor 必须打赢逐帧常量模板"（§11 "学成常量 ⇒ FAIL"）。
    这个臂与 `TemporalDeltaActorV5` **共享同一套 cap/mask/优化器/数据**，唯一区别是
    `u_t` 完全不依赖输入。
    """

    def __init__(self, chunk_len: int = 40, tok_dim: int = TOKEN_DIM, cap: float = 0.08,
                 init_std: float = 4e-4):
        super().__init__()
        self.chunk_len = int(chunk_len)
        self.tok_dim = int(tok_dim)
        self.cap = float(cap)
        self.d_in = 0          # 常量模板没有输入拼接（日志/报表兼容）
        self.template = nn.Parameter(torch.zeros(int(chunk_len), int(tok_dim)))
        if float(init_std) > 0:
            nn.init.normal_(self.template, std=float(init_std))

    def forward(self, Z: torch.Tensor, state64: torch.Tensor | None = None,
                hist: torch.Tensor | None = None, mask: torch.Tensor | None = None,
                *, cap: float | None = None,
                return_raw: bool = False) -> torch.Tensor:
        B, T = int(Z.shape[0]), int(Z.shape[1])
        if T > self.chunk_len:
            raise ValueError(f"[v5/pos-template] T={T} > chunk_len={self.chunk_len}")
        u = self.template[:T][None].expand(B, -1, -1)
        d = apply_norm_cap(u, float(self.cap if cap is None else cap))
        if mask is not None:
            d = d * mask[..., None].to(d.dtype)
        if return_raw:
            return d, u
        return d

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["HIST_DIM", "PositionTemplateActor", "TemporalActorCfg", "TemporalDeltaActorV5"]
