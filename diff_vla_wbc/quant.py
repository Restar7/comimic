"""Token 空间的 FSQ 网格量化 + STE（让反传穿过一步"必须存在但不可导"的操作）。

## 为什么需要这个模块

SONIC 的 64 维 token 不是连续量，而是一张 **1/16 的整数格**。口径来自 release encoder
的导出图（逐算子复刻见 ``sonic_encoder.py:151``）::

    q = round(tanh(h + 0.03223739564418793) * 15.515501022338867 - 0.5) / 16
    ⇒  q ∈ {n/16},  n ∈ [-16, 15]
    ⇒  token 值域 [-1.0, 0.9375]，格距 1/16 = 0.0625，每维 32 级

由此得到三条工程事实：

1. **执行器（WBC / SONIC decoder）只在格点上训练过**。HA 数据集里的
   ``g1_motion_token_64`` 是 FSQ 输出，A' 采集时喂的 token 库同样是 FSQ 输出。
2. **VLA 输出的是格点之间的连续值**（``codec.decode`` 是纯仿射，不量化）。于是
   ``∂a/∂token`` 在那些点上**没有任何数据约束**，是 MLP 的插值外推。
3. 想靠"重训执行器 + dither"补上这一点时，如果执行器不可重训（例如 release SONIC
   decoder），就只剩一条自洽的路：**前向喂格点（分布内）+ 反向按恒等传（梯度不断）**。

第 3 条就是本模块的 STE 实现。

## 为什么 STE 在这里是精确的

::

    τ_q   = fsq_grid(τ)                # 前向用这个（格点上）
    τ_use = τ + (τ_q - τ).detach()     # forward: τ_q   backward: 1

    ∂τ_use/∂τ = 1 + ∂[常数]/∂τ = 1     ← 严格恒等，不是近似

所以 reward → WM → 执行器 → τ_use → τ 这条链**一个断点都没有**；变化的只有"执行器在哪个
点求 Jacobian"（从格点之间变成格点上），而那正是它训练过的点。

## 一个必须避开的写错法

FSQ 公式里的 ``tanh`` 作用在 **encoder 内部的连续 embedding h** 上，**不是 token**。
VLA 输出的 token 已经在 ``q`` 空间里，所以这里的量化只能是**绝对网格**::

    q = clamp(round(τ * 16), -16, 15) / 16

把 FSQ 的 tanh 公式套到 decoded token 上会得到完全错误的值（tanh 会把 [-1,0.9375]
压成 [-0.77,0.73]，再乘 15.5 取整，全乱）。

## 三种模式

======== ================================================================== ==========
mode     行为                                                                用途
======== ================================================================== ==========
``none`` 不量化（旧行为）                                                    对照臂 / 执行器可重训时
``ste``  前向格点 + 反向恒等                                                  **推荐**（部署也量化）
``hard`` 硬量化，梯度归零                                                    对照臂：证明"只量化不 STE 学不动"
======== ================================================================== ==========

``hard`` 存在的意义是消融：它应当**训不动**（``∂loss/∂θ ≡ 0``）。如果 ``hard`` 也能降
loss，说明梯度从别的路径漏进来了（例如 trust/BC 项），此时不能拿它当"量化无害"的证据。
"""

from __future__ import annotations

from typing import Any

import torch

# --------------------------------------------------------------------------- 网格常量
#: 与 release encoder 导出图逐项对齐（见模块 docstring）。
FSQ_DIV = 16
LEVEL_MIN = -16
LEVEL_MAX = 15
GRID_STEP = 1.0 / FSQ_DIV          # 0.0625
TOKEN_MIN = LEVEL_MIN * GRID_STEP  # -1.0
TOKEN_MAX = LEVEL_MAX * GRID_STEP  # 0.9375

#: 量化时保留的 token 维数（78 维动作里的前 64 维；手部 14 维不经过 FSQ）。
TOKEN_DIM = 64

MODES = ("none", "ste", "hard")


# --------------------------------------------------------------------------- 硬量化
def fsq_grid(x: torch.Tensor, *, div: int = FSQ_DIV,
             level_min: int = LEVEL_MIN, level_max: int = LEVEL_MAX) -> torch.Tensor:
    """把任意连续值投射到最近的网格点 ``n/div``（并夹到 encoder 的值域内）。

    注意入参已经在 token（``q``）空间里——**不要**先套 ``tanh``（见模块 docstring）。
    返回**不带梯度**的 fp32 值（这是"前向要用的值"，不是要被学习的量）；``round`` 在 torch 里
    的导数恒为 0，保留图只会白占显存，所以这里直接 detach。
    """
    xf = x.to(torch.float32)
    levels = torch.round(xf * float(div)).clamp(float(level_min), float(level_max))
    q = levels / float(div)
    q = q.to(dtype=x.dtype, device=x.device) if x.dtype != torch.float32 else q
    return q.detach()


def ste_quantize(x: torch.Tensor, **kw: Any) -> torch.Tensor:
    """前向取格点值，反向把梯度原样传给 ``x``（straight-through estimator）。

    ``τ_use = τ + (fsq_grid(τ) − τ).detach()`` ⇒ ``∂τ_use/∂τ ≡ 1``。
    """
    q = fsq_grid(x, **kw)
    return x + (q - x).detach()


# --------------------------------------------------------------------------- 统计
def grid_stats(x: torch.Tensor, *, div: int = FSQ_DIV) -> dict[str, float]:
    """量化前的诊断量。**同时是 STE 偏差的度量**（偏差 ≈ 格距 × 局部曲率）。

    返回（全部无梯度）：

    ``dist_mean`` / ``dist_p90`` / ``dist_max``
        到最近格点的距离。值域内 ≤ 半格 (0.03125)；长期贴着半格 = 在格点边界来回磨。
        超过半格只可能来自越界值（被 ``fsq_grid`` 夹住了），对应 ``out_range > 0``。
    ``near_edge``
        距离 > 四分之一格的元素占比（"快跨界了"的比例）。健康值取决于 lr，看趋势。
    ``out_range``
        落在 encoder 值域 ``[-1, 0.9375]`` 之外的占比（VLA 外推）。
    ``cell``
        格距常数（便于日志里直接换算成 mm/尺度）。
    """
    with torch.no_grad():
        xf = x.detach().to(torch.float32)
        q = fsq_grid(xf, div=div)
        d = (xf - q).abs()
        step = 1.0 / float(div)
        return {
            "dist_mean": float(d.mean()),
            "dist_p90": float(d.flatten().quantile(0.9)) if d.numel() else 0.0,
            "dist_max": float(d.max()) if d.numel() else 0.0,
            "near_edge": float((d > 0.25 * step).float().mean()),
            "out_range": float(((xf < TOKEN_MIN) | (xf > TOKEN_MAX)).float().mean()),
            "cell": step,
        }


def quantize_np(x, mode: str = "none", *, div: int = FSQ_DIV,
                level_min: int = LEVEL_MIN, level_max: int = LEVEL_MAX):
    """部署侧（无梯度）numpy 版本：返回 ``(量化后, 量化前平均偏离度)``。

    为什么单独一份 numpy：闭环服务里不该为了 40×64 个数把 torch 张量搬来搬去。
    但它必须和 :func:`fsq_grid` **逐位一致**——`tests/test_quant.py` 里用随机张量对拍
    （``np.rint`` 与 ``torch.round`` 都是 round-half-to-even，所以能对齐）。

    ``mode`` 只区分 ``none`` 与非 ``none``：推理没有梯度，``ste``/``hard`` 的前向完全一样。
    """
    import numpy as np

    a = np.asarray(x, dtype=np.float32)
    if str(mode or "none").strip().lower() in ("", "none", "off", "false"):
        return a, 0.0
    levels = np.clip(np.rint(a * float(div)), float(level_min), float(level_max))
    q = (levels / float(div)).astype(np.float32)
    return q, float(np.abs(a - q).mean())


def is_on_grid(x: torch.Tensor, *, div: int = FSQ_DIV, atol: float = 1e-6) -> bool:
    """``x`` 是否已经全部落在网格上（审计/测试用）。"""
    with torch.no_grad():
        xf = x.detach().to(torch.float32)
        return bool(((xf * float(div)) - torch.round(xf * float(div))).abs().max() <= atol)


def grid_alignment(lo: torch.Tensor, hi: torch.Tensor, *, div: int = FSQ_DIV) -> dict[str, float]:
    """审计 codec 的原始值域是否与网格对齐。

    ``codec.decode`` 把归一化空间的 ``[-1, 1]`` 映射到 ``[lo, hi]``（= 数据集 min/max）。
    如果 ``lo``/``hi`` 恰好是 ``n/div``，那"标签 token → 归一化 → decode"这条往返会把格点
    原样还回来；否则 decode 之后的标签本身就不在格点上，量化会引入额外误差。

    返回 ``{"lo_off", "hi_off", "lo_ok", "hi_ok"}``（``_off`` 单位是格距）。
    """
    with torch.no_grad():
        step = 1.0 / float(div)
        lo_off = float(((lo.detach().to(torch.float32) / step)
                        - torch.round(lo.detach().to(torch.float32) / step)).abs().max())
        hi_off = float(((hi.detach().to(torch.float32) / step)
                        - torch.round(hi.detach().to(torch.float32) / step)).abs().max())
        return {
            "lo_off": lo_off,
            "hi_off": hi_off,
            "lo_ok": float(lo_off <= 1e-3),
            "hi_ok": float(hi_off <= 1e-3),
        }


# --------------------------------------------------------------------------- 部署侧口径
#: 部署侧允许的量化模式（推理没有梯度，``ste`` 与 ``hard`` 的前向完全一样）
DEPLOY_MODES = ("none", "ste", "hard")
#: "是否量化" 的等价类：训练 ``ste`` 只能配部署 ``ste``/``hard``
_QUANTIZED = ("ste", "hard")


def resolve_deploy_mode(value) -> str:
    """校验并归一化部署侧口径。

    argparse **不校验字符串默认值**是否在 choices 里（只有命令行显式给的值才校验），
    所以 ``TOKEN_QUANT=bogus`` 这种环境变量会静默溜进来 —— 服务端必须显式挡一次。
    """
    mode = str(value or "none").strip().lower()
    if mode not in DEPLOY_MODES:
        raise ValueError(
            f"[token-quant] 未知取值 {value!r}（可选 {'/'.join(DEPLOY_MODES)}，"
            f"来自 --token-quant 或环境变量 TOKEN_QUANT）")
    return mode


def check_meta_mode(ckpt_dir, want, *, key: str = "token_quant",
                    allow_env: str = "ALLOW_TOKEN_QUANT_MISMATCH", log=print) -> str:
    """读 ``vla_meta.json[key]`` 与部署口径比对，不一致直接 ``SystemExit``。

    为什么必须校验：``token_quant=ste`` 时**训练前向喂的是格点值**。部署侧若还喂连续值，
    学到的补偿就是对着一个不存在的执行器学的（``docs/TOKEN_QUANT_STE_20261104.md`` §4）。
    旧 ckpt 没有这个键 ⇒ 按 ``none`` 处理并告警（不挡老线）。

    放行开关（``allow_env=1``）留给 A/B。
    """
    import json
    import os
    from pathlib import Path as _Path

    want = resolve_deploy_mode(want)
    meta_path = _Path(ckpt_dir) / "vla_meta.json"
    train_q = None
    if meta_path.exists():
        try:
            train_q = json.loads(meta_path.read_text()).get(key)
        except Exception:  # noqa: BLE001 - 坏 meta 不挡启动，按旧口径处理
            train_q = None
    if train_q in (None, ""):
        log(f"[token-quant][WARN] {meta_path} 没有 {key}：按 none 处理（旧 ckpt 口径）")
        return "none"
    train_q = str(train_q).lower()
    if (train_q in _QUANTIZED) != (want in _QUANTIZED):
        if os.environ.get(allow_env) != "1":
            raise SystemExit(
                f"[token-quant] 训练用 {key}={train_q!r}，服务端 --token-quant {want!r}；"
                f"二者必须一致（A/B 才允许 {allow_env}=1）")
        log("[token-quant][WARN] 口径不一致但已放行（A/B）")
    return train_q


# --------------------------------------------------------------------------- 入口
def apply_token_quant(x: torch.Tensor, mode: str | None) -> tuple[torch.Tensor, dict[str, float] | None]:
    """按 ``mode`` 处理 token，返回 ``(token, stats)``；``stats`` 只在量化时给。

    这个函数是训练器里**唯一**的量化入口，``x`` 必须是 token 空间
    （``codec.decode(...)[..., :64]`` 之后的 64 维），**不是**归一化空间。
    """
    key = (mode or "none").strip().lower()
    if key in ("", "none", "off", "false"):
        return x, None
    if key == "ste":
        return ste_quantize(x), grid_stats(x)
    if key == "hard":
        return fsq_grid(x), grid_stats(x)
    raise ValueError(f"unknown token_quant mode {mode!r}; choose from {MODES}")


def quant_available() -> bool:
    """留给调用方做能力探测（恒 True；保留以对齐其它模块的 API 形状）。"""
    return True


if __name__ == "__main__":  # pragma: no cover - 手工自检
    t = torch.tensor([[-1.0, -0.031, 0.0, 0.031, 0.0311, 0.9375, 0.99, 1.2]],
                     requires_grad=True)
    print("grid step :", GRID_STEP)
    print("range     :", TOKEN_MIN, "→", TOKEN_MAX)
    print("fsq_grid  :", fsq_grid(t).tolist())
    print("on grid   :", is_on_grid(fsq_grid(t)))
    y = ste_quantize(t)
    print("ste fwd   :", y.tolist())
    y.sum().backward()
    print("ste bwd   : ∂y/∂x 应全为 1 →", t.grad.tolist())
    print("stats     :", grid_stats(t))
    print("align     :", grid_alignment(torch.tensor([-1.0]), torch.tensor([0.9375])))
