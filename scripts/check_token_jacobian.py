#!/usr/bin/env python3
"""量一个**前置判据**：执行器在 token 格点上的 Jacobian 到底准不准。

## 为什么必须先量这个

反传链是 ``reward → WM → 执行器 → token → VLA head``。最后一段 ``∂a/∂token`` 是**执行器的
局部斜率**，而执行器只在 ``q = n/16`` 的格点上训练过：

* 前向喂连续 token ⇒ 这个斜率是 MLP 的**插值外推**，没有任何数据约束；
* 用 ``token_quant=ste`` 把它掰回格点，才是在"它见过的点"上求导。

所以在决定"要不要量化、要不要 dither 重训执行器"之前，先量：

    解析 Jacobian ``∂a/∂τ_j · δ``  ≟  真实差分 ``a(τ + δ·e_j) − a(τ)``

## 两种执行器

===================  ============================================================  ==============
模式                 观测从哪来                                                     额外必填
===================  ============================================================  ==============
``--ckpt``（默认）    ``reference_start`` 从 packed motion 库构造（同帧配对）        ``--packed``
``--sonic-decoder``   HA raw 的 ``decoder_obs (T,994)``，只扰动前 64 维             ``--raw-npz`` 可选
===================  ============================================================  ==============

SONIC 模式为什么不用重建 994 维布局：HA raw npz 逐帧存了真 ``decoder_obs`` / 
``decoder_target_action``，而 release decoder 能把前者复现成后者（1e-5 度）。直接用数据集里的
真观测 ⇒ 天然在分布内。

## 为什么要扫多个 δ

一个 δ 量不出"外推准不准"，因为差分里同时含**曲率**：

* δ = 1 格（0.0625）时，单维扰动就能让动作变化 0.1 量级（动作本身 ~1）——大位移；
* δ 很小时 ``rel`` 应 ⇢ 0（可微函数的一阶展开）。

============= ==========================================================
δ（格距倍数）  判断什么
============= ==========================================================
0.01           **正确性**：解析 Jacobian 本身对不对（cos 应 ≈ 1.000）
0.25 / 0.5     **可用性**：亚格外推（VLA 输出最多偏半格）误差有多大
1.0            **敏感性**：跨一格的真实变化 + 有效 gain（用来定 lr）
============= ==========================================================

判据::

    cos(δ=0.01) < 0.99                 → autograd 路径有问题，先修代码
    cos(δ=0.5) ≥ 0.99 且 rel ≤ 0.10    → 连续喂（token_quant=none）可接受
    cos(δ=0.5) ≥ 0.95 且 rel ≤ 0.30    → 建议 token_quant=ste
    其它                                → 必须 token_quant=ste + 对执行器 dither 重训

## 用法

::

    # ① WBC（SuperMimic）
    python3 scripts/check_token_jacobian.py \\
      --ckpt runs/S1_handover/exp_WBC_exec_haonly_best.pt \\
      --packed runs/isaac_motion/ha_cmd_exec/packed.pt \\
      --frames 8 --dims 16

    # ② SONIC release decoder（ONNX → 可微 torch，逐算子复刻）
    python3 scripts/check_token_jacobian.py \\
      --sonic-decoder /…/gear_sonic_deploy/policy/release/model_decoder.onnx \\
      --frames 8 --dims 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diff_vla_wbc.quant import GRID_STEP, TOKEN_DIM, fsq_grid, is_on_grid  # noqa: E402

DEFAULT_TOKENS_DIR = ROOT / "runs" / "exp_tokens"
DEFAULT_RAW_GLOB = "data/ha_raw/*/*/*.npz"
#: release decoder 的 ONNX（在**工作区**层，不在 comimic 里）
DEFAULT_SONIC_ONNX = (ROOT.parent / "GR00T-WholeBodyControl" / "gear_sonic_deploy"
                      / "policy" / "release" / "model_decoder.onnx")


# --------------------------------------------------------------------------- token / 帧
def _load_token_file(spec: str) -> tuple[torch.Tensor | None, Path | None]:
    path = Path(spec)
    if path.is_dir():
        cands = sorted(path.glob("*.npy"))
        if not cands:
            raise SystemExit(f"{path} 里没有 .npy token 文件")
        path = cands[0]
        print(f"[tokens] 目录里取第一个: {path.name}", flush=True)
    if not path.is_file():
        raise SystemExit(f"token 文件不存在: {path}")
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.shape[-1] < TOKEN_DIM:
        raise SystemExit(f"{path} 最后一维 {arr.shape[-1]} < {TOKEN_DIM}")
    return torch.as_tensor(arr[:, :TOKEN_DIM], dtype=torch.float32), path


def _match_clip(motion, path: Path | None) -> int | None:
    """把 token 文件名匹配回 motion 库里的 clip（空名一律跳过）。"""
    if path is None:
        return None
    key = path.stem
    names = [str(n) for n in getattr(motion, "names", [])]
    for i, n in enumerate(names):
        if n and (n in key or key in n):
            return i
    return None


def _sample_frames(motion, n: int, g: torch.Generator, clip_i: int | None):
    """``clip_i`` 给定时只在该 clip 内抽（token 行号才能对齐）。"""
    starts = motion.clip_start.detach().to("cpu").long()
    lens = motion.clip_len.detach().to("cpu").long()
    if clip_i is not None:
        c0, ln = int(starts[clip_i]), int(lens[clip_i])
        rows = torch.randint(max(ln - 2, 1), (n,), generator=g)
        return c0 + rows, rows
    pick = torch.randint(starts.numel(), (n,), generator=g)
    offs = (torch.rand(n, generator=g) * (lens[pick] - 2).clamp(min=1).float()).long()
    return starts[pick] + offs, None


# --------------------------------------------------------------------------- 执行器（统一成 fn）
@torch.no_grad()
def _apply(fn, tokens):
    return fn(tokens).float()


def _jacobian(fn, tokens) -> torch.Tensor:
    """完整 Jacobian ``(B, K, 64)``：``J[b,k,j] = ∂a[b,k]/∂token[b,j]``。

    **必须逐输出维求导**。第一版脚本用 ``grad_outputs=ones(K)``，拿到的是"对 K 个输出求和后
    的梯度"（标量），拿去和逐维差分比 ⇒ cos 恒为 ~0.2、rel 与 δ 无关（两边不是同一个东西）。
    """
    tk = tokens.detach().clone().requires_grad_(True)
    a = fn(tk).float()
    b_n, k_n = a.shape
    out = torch.zeros(b_n, k_n, tokens.shape[-1], device=a.device, dtype=a.dtype)
    for b in range(b_n):
        for k in range(k_n):
            g_k = torch.autograd.grad(a[b, k], tk, retain_graph=True)[0]   # (B,64)
            out[b, k] = g_k[b]
    return out


# --------------------------------------------------------------------------- SONIC 模式
def _sonic_setup(args, g: torch.Generator):
    from diff_vla_wbc.sonic_decoder import ACTION_DIM, DECODER_IN_DIM, SonicDecoderTorch

    onnx = Path(args.sonic_decoder)
    if not onnx.is_file():
        raise SystemExit(f"decoder onnx 不存在: {onnx}")
    if args.raw_npz:
        path = Path(args.raw_npz)
        if path.is_dir():
            cands = sorted(path.rglob("*.npz"))
            if not cands:
                raise SystemExit(f"{path} 下没有 .npz")
            path = cands[0]
    else:
        cands = sorted(ROOT.glob(DEFAULT_RAW_GLOB))
        if not cands:
            raise SystemExit(f"找不到 HA raw npz（{DEFAULT_RAW_GLOB}）")
        path = cands[0]
    d = np.load(path, allow_pickle=True)
    if "decoder_obs" not in d:
        raise SystemExit(f"{path} 里没有 decoder_obs（不是 HA raw 口径？）")
    obs_all = np.asarray(d["decoder_obs"], dtype=np.float32)
    tgt_all = (np.asarray(d["decoder_target_action"], dtype=np.float32)
               if "decoder_target_action" in d else None)
    if obs_all.shape[-1] != DECODER_IN_DIM:
        raise SystemExit(f"decoder_obs 最后一维 {obs_all.shape[-1]} != {DECODER_IN_DIM}")

    dec = SonicDecoderTorch.from_onnx(onnx).to(args.device)
    idx = torch.randint(obs_all.shape[0], (int(args.frames),), generator=g)
    obs = torch.from_numpy(obs_all[idx.numpy()]).to(args.device)
    tokens = obs[:, :TOKEN_DIM].clone()

    print(f"[sonic] {onnx.name}  参数 {sum(p.numel() for p in dec.parameters()) / 1e6:.2f}M  "
          f"数据 {Path(path).name} rows={obs_all.shape[0]} 取 {idx.tolist()}", flush=True)
    # 口径自检：torch decoder 复现 HA 录制动作（逐关节仿射拟合，同 SONIC_DECODER_FOUND 口径）
    if tgt_all is not None:
        with torch.no_grad():
            out = dec(obs).cpu().numpy()
        tg = tgt_all[idx.numpy()]
        res = []
        for j in range(ACTION_DIM):
            k_, b_ = np.polyfit(out[:, j], tg[:, j], 1)
            res.append(np.abs(tg[:, j] - (k_ * out[:, j] + b_)).mean())
        print(f"[sonic] 与 decoder_target_action 的逐关节仿射残差 = "
              f"{np.degrees(np.mean(res)):.6f} deg（应 ≈1e-5，否则口径不对）", flush=True)
    print(f"[sonic] 数据集 token on_grid={is_on_grid(tokens)}  "
          f"mean|x|={float(tokens.abs().mean()):.4f}", flush=True)

    def fn(tk):
        return dec.act(tk, obs)

    return fn, tokens


# --------------------------------------------------------------------------- WBC 模式
def _wbc_setup(args, g: torch.Generator):
    from diff_vla_wbc.isaac_joint import load_isaac_bundle

    if args.packed:
        bundle = load_isaac_bundle(args.ckpt, packed_motion=args.packed, device=args.device)
    elif args.npz:
        bundle = load_isaac_bundle(args.ckpt, npz=[Path(p) for p in args.npz], device=args.device)
    else:
        raise SystemExit("WBC 模式必须给 --packed 或 --npz（或改用 --sonic-decoder）")
    bundle.policy.eval()
    for p in bundle.policy.parameters():        # 只对 token 求导
        p.requires_grad_(False)
    motion = bundle.motion

    if str(args.tokens).strip().lower() == "none":
        print("[warn] --tokens none：token 全零，**分布外**，结果只能当冒烟。", flush=True)
        frames, _ = _sample_frames(motion, int(args.frames), g, None)
        tokens = torch.zeros(int(args.frames), TOKEN_DIM, device=bundle.device)
        paired = False
    else:
        arr, path = _load_token_file(args.tokens)
        clip_i = _match_clip(motion, path)
        frames, rows = _sample_frames(motion, int(args.frames), g, clip_i)
        if clip_i is not None and rows is not None:
            rows = rows.clamp(max=arr.shape[0] - 1)
            tokens = arr[rows].to(bundle.device)
            paired = True
            print(f"[tokens] {path.name} → clip#{clip_i} 同帧配对 rows={rows.tolist()}", flush=True)
        else:
            idx = torch.randint(arr.shape[0], (int(args.frames),), generator=g)
            tokens = arr[idx].to(bundle.device)
            paired = False
            print("[warn] token 文件名匹配不到 clip ⇒ 观测与 token **不配对**（OOD）。"
                  "数值只能当上限估计。", flush=True)
    start = bundle.reference_start(motion, frames.to(bundle.device))
    obs_hist = start["obs_hist"]
    print(f"[wbc] paired={paired}", flush=True)

    def fn(tk):
        return bundle.act(obs_hist, tk)

    return fn, tokens


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="WBC/joint ckpt（WBC 模式必填）")
    ap.add_argument("--packed", default=None, help="packed motion 库")
    ap.add_argument("--npz", nargs="*", default=None, help="没有 packed 时用 npz 现建（慢）")
    ap.add_argument("--sonic-decoder", default=None, nargs="?", const="__default__",
                    help="切到 SONIC 模式：release decoder 的 model_decoder.onnx"
                         "（不带值时用 release 默认路径）")
    ap.add_argument("--raw-npz", default=None,
                    help="SONIC 模式的 decoder_obs 来源（npz 或目录；默认取 data/ha_raw 第一个）")
    ap.add_argument("--tokens", default=str(DEFAULT_TOKENS_DIR),
                    help="WBC 模式的 token .npy / 目录 / 'none'")
    ap.add_argument("--frames", type=int, default=8, help="batch")
    ap.add_argument("--dims", type=int, default=16, help="测多少个 token 维（随机抽）")
    ap.add_argument("--steps", default="0.01,0.25,0.5,1.0",
                    help="δ 的格距倍数，逗号分隔（默认 0.01,0.25,0.5,1.0）")
    ap.add_argument("--dead-eps", type=float, default=1e-4,
                    help="‖Δa_fd‖ 小于此值视为该维'几乎不动'（梯度是噪声）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="把结果写成 json")
    args = ap.parse_args(argv)

    torch.manual_seed(int(args.seed))
    g = torch.Generator().manual_seed(int(args.seed))
    steps = [float(s) for s in str(args.steps).split(",") if s.strip()]

    if args.sonic_decoder:
        if args.sonic_decoder == "__default__":
            args.sonic_decoder = str(DEFAULT_SONIC_ONNX)
        fn, tokens = _sonic_setup(args, g)
        label = Path(args.sonic_decoder).name
    else:
        if not args.ckpt:
            raise SystemExit("WBC 模式需要 --ckpt（或改用 --sonic-decoder）")
        fn, tokens = _wbc_setup(args, g)
        label = Path(args.ckpt).name

    tokens = fsq_grid(tokens)            # 必须在格点上比："格点 → 邻格"才是真实差分
    if not is_on_grid(tokens):
        raise SystemExit("[jac] 内部错误：tokens 不在格点上")
    dims = torch.randperm(TOKEN_DIM, generator=g)[: int(args.dims)].tolist()
    B = tokens.shape[0]

    print("\n" + "=" * 92)
    print(f"[jac] {label} frames={B} dims={len(dims)} 格距={GRID_STEP:.5f}")
    print("=" * 92)

    a0 = _apply(fn, tokens)
    J = _jacobian(fn, tokens)                                # (B,K,64)
    per_dim: list[dict] = []
    for j in dims:
        gj = J[:, :, j]                                      # (B,K) 解析斜率
        e = torch.zeros_like(tokens)
        e[:, j] = 1.0
        rec = {"dim": int(j), "rows": {}}
        for s in steps:
            delta = float(s) * GRID_STEP
            fd = _apply(fn, tokens + e * delta) - a0          # (B,K)
            jac = gj * delta
            fd_norm = fd.norm(dim=-1)
            rel = (jac - fd).norm(dim=-1) / fd_norm.clamp_min(1e-12)
            cos = torch.nn.functional.cosine_similarity(jac, fd, dim=-1)
            rec["rows"][f"{s:g}"] = {
                "cos_mean": float(cos.mean()), "rel_mean": float(rel.mean()),
                "rel_median": float(rel.median()),
                "gain": float((fd_norm / delta).mean()),
                "dead": float((fd_norm < float(args.dead_eps)).float().mean()),
            }
        per_dim.append(rec)

    print(f"{'δ/格距':>7} {'cos_mean':>9} {'rel_mean':>9} {'rel_med':>8} "
          f"{'gain':>10} {'dead':>6}  用途")
    print("-" * 92)
    agg: dict[str, dict] = {}
    for s in steps:
        key = f"{s:g}"
        v = torch.tensor([[r["rows"][key]["cos_mean"], r["rows"][key]["rel_mean"],
                          r["rows"][key]["rel_median"], r["rows"][key]["gain"],
                          r["rows"][key]["dead"]] for r in per_dim])
        row = {
            "cos_mean": float(v[:, 0].mean()), "rel_mean": float(v[:, 1].mean()),
            "rel_median": float(v[:, 2].median()), "gain": float(v[:, 3].mean()),
            "dead": float(v[:, 4].mean()),
        }
        agg[key] = row
        note = {0.01: "正确性", 0.25: "亚格外推", 0.5: "亚格外推(最坏)", 1.0: "跨一格/敏感性"}.get(
            float(s), "")
        print(f"{s:>7.2f} {row['cos_mean']:>9.4f} {row['rel_mean']:>9.4f} "
              f"{row['rel_median']:>8.4f} {row['gain']:>10.4f} {row['dead']:>6.3f}  {note}")

    tiny = min(steps)
    ref = min(steps, key=lambda x: abs(x - 0.5))
    cos_tiny = agg[f"{tiny:g}"]["cos_mean"]
    cos_sub, rel_sub = agg[f"{ref:g}"]["cos_mean"], agg[f"{ref:g}"]["rel_mean"]
    if cos_tiny < 0.99:
        verdict = (f"BROKEN: cos(δ={tiny:g})={cos_tiny:.4f} < 0.99 ⇒ 解析 Jacobian 本身不对"
                   f"（autograd 断链 / 执行器饱和 / token-obs 不配对），先别谈量化")
    elif cos_sub >= 0.99 and rel_sub <= 0.10:
        verdict = "OK: 亚格外推可信 ⇒ token_quant=none（连续喂）可接受"
    elif cos_sub >= 0.95 and rel_sub <= 0.30:
        verdict = "MARGINAL: 建议 token_quant=ste（前向格点 + 反向恒等）"
    else:
        verdict = "FAIL: 亚格外推不可信 ⇒ 必须 token_quant=ste + 对执行器 dither 重训"

    print("-" * 92)
    print(f"[jac] cos(δ={tiny:g})={cos_tiny:.4f} | cos(δ={ref:g})={cos_sub:.4f} "
          f"rel={rel_sub:.4f} | gain@1格={agg[f'{max(steps):g}']['gain']:.4f}")
    print(f"[jac] 判定: {verdict}")
    print("[jac] 注意：这个数只依赖「执行器 + token 分布 + 观测构造」，与 WM / 奖励 / δ 课程无关；"
          "换执行器或换 token 空间必须重测。")

    if args.out:
        payload = {
            "mode": ("sonic" if args.sonic_decoder else "wbc"),
            "ckpt": str(args.ckpt), "packed": str(args.packed),
            "sonic_decoder": str(args.sonic_decoder), "tokens": str(args.tokens),
            "frames": B, "dims": len(dims), "steps_cells": steps,
            "aggregate": agg, "verdict": verdict, "per_dim": per_dim,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[jac] 已写 {args.out}")
    return 0 if verdict.startswith("OK") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
