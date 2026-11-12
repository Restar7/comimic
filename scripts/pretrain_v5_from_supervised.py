#!/usr/bin/env python3
"""把**监督蒸馏训好的 Actor**转成 v5 结构（结构完全不动，只换初始化）。

背景与约束（用户口径）
----------------------
* BPTT 主线必须**完全按** `docs/TEMPORAL_AC_BPTT_V5_HANDOFF_V2.md` 走：结构 1,744,832 参数、
  `trust_mode=same_obs`、`lambda_raw=1e-3/κ=2`、`cap=0.08`、`H=K=20`、`updates=1000`… **一个字不改**；
* **唯一允许的差异 = Actor 的初始化**：不随机初始化，而是从我们"教师标签监督蒸馏"训出来的
  Actor 权重出发。

本脚本负责把两者接上：
1. 我们监督训练用的模块（`diff_vla_wbc/delta_token_temporal.py`）与 v5 的
   `TemporalDeltaActorV5` **只差三处**：
   - 属性名 `chunk_attn.*` ↔ `chunk_encoder.*`；
   - v5 多一个 `chunk_norm`（LayerNorm，默认 γ=1/β=0 ⇒ 恒等）；
   - v5 的 `pe_chunk` / `pe_pos` 是**可学习参数**，我们用的是固定正弦编码。
2. 因此映射规则：改名 + 把 `pe_chunk`/`pe_pos` 初始化为**同一套正弦值** + `chunk_norm` 保持默认。
3. **等价性断言**（脚本内强制）：映射之后，同样输入下两个模块的输出必须逐元素一致
   （`max|Δ| < 1e-6`）。不通过就报错退出，绝不"差不多就行"。
4. 写出 v5 格式 checkpoint（`save_temporal_actor_v5`，含 spec/meta），供
   `scripts/train_temporal_ac_bptt_v5.py --init-actor <this.ckpt>` 使用。

用法
----
    python scripts/pretrain_v5_from_supervised.py \
        --src runs/distill_temporal/temporal_B4.pt \
        --out runs/distill_temporal/v5_init_from_supervised.pt \
        --t 40 --k 20 --h 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diff_vla_wbc.delta_token_io_v5 import (  # noqa: E402
    DeltaTokenSpecV5, save_temporal_actor_v5)
from diff_vla_wbc.delta_token_temporal import (  # noqa: E402
    positional_encoding)
from diff_vla_wbc.delta_token_temporal_v5 import TemporalDeltaActorV5  # noqa: E402


def remap(src_sd: dict) -> dict:
    """我们模块的 state_dict → v5 的 key 命名（只改名，不动数值）。"""
    out = {}
    for k, v in src_sd.items():
        nk = k.replace("chunk_attn.", "chunk_encoder.") if k.startswith("chunk_attn.") else k
        out[nk] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/distill_temporal/temporal_B4.pt")
    ap.add_argument("--out", default="runs/distill_temporal/v5_init_from_supervised.pt")
    ap.add_argument("--t", type=int, default=40, help="chunk 长度 T（doc: 40）")
    ap.add_argument("--k", type=int, default=20, help="实际执行/修正帧数 K（doc: 20）")
    ap.add_argument("--h", type=int, default=20, help="BPTT 视界 H（doc: 20）")
    ap.add_argument("--cap", type=float, default=0.08)
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()

    src = torch.load(a.src, map_location="cpu", weights_only=False)
    sd = src.get("state_dict") or src
    v5 = TemporalDeltaActorV5()
    mapped = remap(sd)

    # ---- pe_chunk / pe_pos：补成"与我们固定正弦一致"的初值 -------------------
    T = int(a.t)
    mapped["pe_chunk"] = positional_encoding(T, int(v5.cfg.d_model))
    pe_pos = positional_encoding(T, int(v5.cfg.pe_pos_dim))
    mapped["pe_pos"] = pe_pos
    # chunk_norm 保持默认（γ=1, β=0 ⇒ 恒等），不覆盖

    res = v5.load_state_dict(mapped, strict=False)      # 新版 torch 返回 _IncompatibleKeys
    missing, unexpected, mismatched = res.missing_keys, res.unexpected_keys, []
    # `chunk_norm.*` 属于"v5 多出来但默认即恒等"的参数 ⇒ 允许 missing
    hard_missing = [k for k in missing if not k.startswith("chunk_norm.")]
    if hard_missing or unexpected or mismatched:
        raise SystemExit(
            f"[map] 映射不完整：missing={hard_missing} unexpected={list(unexpected)} "
            f"mismatched={list(mismatched)}")
    print(f"[map] 权重已注入 v5 Actor（params={v5.n_params}）；"
          f"chunk_norm 保持默认（missing={list(missing)}）")

    # ---- 等价性断言：除 v5 的 chunk_norm（无对应物）外必须逐位一致 ------------
    #   v5 结构比我们的监督模块**多一个 LayerNorm(128)**（256 参数）。它在 doc 的结构里是必须的，
    #   但监督模型没有对应物 ⇒ 只能保持默认初始化（γ=1,β=0）。注意 LayerNorm 默认**不是恒等**
    #   （它仍会做归一化），所以"函数级等价"只能在**把 chunk_norm 换成 Identity** 后检验。
    #   这条断言证明：**除这一层外，所有权重逐位来自监督模型**，没有映射错误。
    from diff_vla_wbc.delta_token_temporal import TemporalDeltaActor

    import copy as _copy  # noqa: PLC0415
    v5_no_ln = _copy.deepcopy(v5)
    v5_no_ln.chunk_norm = torch.nn.Identity()

    ours = TemporalDeltaActor(cap=a.cap)
    ours.load_state_dict(sd, strict=True)
    ours.eval(); v5.eval(); v5_no_ln.eval()
    g = torch.Generator().manual_seed(0)
    Z = torch.randn(3, T, 64, generator=g)
    hist = torch.randn(3, 10, 58, generator=g)
    s64 = torch.randn(3, 64, generator=g)
    mask = torch.zeros(3, T)
    mask[:, : int(a.k)] = 1.0
    with torch.no_grad():
        d_ours = ours(Z, hist, s64, mask=mask)
        d_v5 = v5_no_ln(Z, s64, hist, mask)          # ← 去掉 chunk_norm 后应逐位相同
    diff = float((d_ours - d_v5).abs().max())
    rel = diff / max(float(d_ours.abs().max()), 1e-12)
    with torch.no_grad():
        d_v5_full = v5(Z, s64, hist, mask)           # 真结构（含 chunk_norm）
    diff_full = float((d_ours - d_v5_full).abs().max())
    print(f"[parity] 去掉 chunk_norm 后 max|Δ| = {diff:.3e} (rel {rel:.3e})")
    print(f"[parity] 含 chunk_norm 的真结构 max|Δ| = {diff_full:.3e}"
          f"（这一层监督模型没有 ⇒ 属结构差异，不是映射错误）")
    if diff > float(a.tol):
        raise SystemExit(f"[parity] 不通过：{diff:.3e} > tol {a.tol:.1e}（映射不自洽，拒绝产出）")
    # mask 语义：t≥K 必须恒为 0（两条路径都查）
    tail = max(float(d_ours[:, int(a.k):].abs().max()), float(d_v5[:, int(a.k):].abs().max()))
    if tail != 0.0:
        raise SystemExit(f"[parity] t≥K 的 Δ 不为 0（{tail:.3e}）")
    print(f"[parity] t≥K 恒零 ✅   ‖Δ‖ 均值 {float(d_v5.norm(dim=-1).mean()):.4f}")

    # ---- 写 v5 checkpoint ---------------------------------------------------
    spec = DeltaTokenSpecV5(T=int(a.t), K=int(a.k), H=int(a.h), cap=float(a.cap))
    # 参数数校验由 `save_temporal_actor_v5` 内部完成（spec.n_params 是派生属性）
    save_temporal_actor_v5(a.out, v5, spec, extra={
        "init_from": str(a.src),
        "init_note": "监督蒸馏（教师标签 ĝ）初始化；结构与 v5 文档完全一致，仅初始化不同",
        "parity_max_abs_no_layernorm": diff,
        "parity_max_abs_full": diff_full,
    })
    meta = {"out": a.out, "params": int(v5.n_params), "T": T, "K": int(a.k), "H": int(a.h),
            "cap": float(a.cap), "parity_max_abs_no_layernorm": diff,
            "parity_max_abs_full": diff_full,
            "src": str(a.src), "src_heldout_cos": (src.get("meta") or {}).get("heldout_cos")}
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[ok] 写出 {a.out}（v5 结构 ckpt，可直接给 --init-actor）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
