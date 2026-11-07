#!/usr/bin/env python3
"""在线梯度修正（**推理期**，不是训练）：给定当前状态与参考，算出
``ĝ = ∂J/∂a``（J = H 步 WM 想象回报，与训练同一个 ``bundle.reward``），
把它当作**这一步动作的修正方向**直接用掉。

为什么是"用"而不是"学"
------------------------
2026-11-06 实测（`logs/truth_probe*.log`）：沿 ĝ 走 δ=0.03 rad，
**真物理里 H=2 96/96 变好、H=20 63/64 变好**（回报 +9% / +14%），
但 WM **高估幅度 ~1.8×**（预测 +0.128 vs 实测 +0.059）。

⇒ 方向可信，**但必须打折**（``eta`` 相对 δ=0.03 再乘 0.5 左右），且只走一小步。

与 `v3_step.py` 的 `J1_GRAD_PROBE` 的不同：那个求 ``∂J/∂A``（整块动作），
这里求**同形状的东西**但用在**一个状态的一次前向**上，不做跨窗口求和。

API
---
    from diff_vla_wbc.online_grad import OnlineGradCorrector
    corr = OnlineGradCorrector(wbc_ckpt=..., wm_ckpt=..., device="cuda:0")
    g = corr.grad_action(state71, ref71, action_norm, horizon=20)     # (29,)
    a_new = corr.apply(action_norm, g, eta=0.01)                      # (29,)

* ``state71`` = ``[p(3), quat_wxyz(4), q(29), v_b(3), w_b(3), qd(29)]``（canonical/standard 序）
* ``ref71``   同布局（参考状态）
* ``action_norm`` = ``(q_target − default_q)/action_scale``（standard 序，与 WM 训练口径一致）
"""
from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


class _OneRefMotion:
    """把"一条参考轨迹"伪装成 motion 库，让 ``bundle.reward`` 用 ``FrameRef(t)`` 直接吃。

    `IsaacJointBundle.reward` 只通过 ``reference.f`` 去 ``self.motion`` 取参考，
    所以把一个 (H,71) 的张量包成这个对象就够了，不需要真的运动库。
    """

    def __init__(self, ref71: torch.Tensor, joint_slice: slice = slice(7, 36)):
        self.ref = ref71
        self._js = joint_slice

    def ref_state_at(self, frames: torch.Tensor) -> torch.Tensor:
        idx = frames.to(self.ref.device).long().clamp(0, self.ref.shape[0] - 1)
        return self.ref[idx]

    def ref_joint_pos(self, frames: torch.Tensor) -> torch.Tensor:
        idx = frames.to(self.ref.device).long().clamp(0, self.ref.shape[0] - 1)
        return self.ref[idx][:, self._js]


class OnlineGradCorrector:
    """一次构造，长期复用；所有方法都 fail-safe（出错就返回 0 修正）。"""

    def __init__(self, *, wbc_ckpt: str, wm_ckpt: str, device: str = "cuda:0",
                 packed_motion: str | None = None, npz_glob: str | None = None,
                 verbose: bool = True):
        from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415

        self.device = torch.device(device)
        # ★ 闭环里 Isaac 进程的 cwd 不是仓库根 ⇒ 一律先解析成绝对路径
        def _abs(x: str) -> str:
            q = Path(x)
            return str(q if q.is_absolute() else (ROOT / q))
        wbc_ckpt = _abs(wbc_ckpt)
        wm_ckpt = _abs(wm_ckpt) if wm_ckpt else wm_ckpt
        if packed_motion:
            packed_motion = _abs(packed_motion)
        npz = sorted(Path(ROOT / (npz_glob or "data/npz/ha_tokens")).glob("*.npz"))
        self.bundle = load_isaac_bundle(
            wbc_ckpt, device=self.device, npz=npz,
            packed_motion=packed_motion or str(ROOT / "runs/isaac_motion/ha_cmd/packed.pt"),
            obs_mode=None, hist=None, token_dim=64,
        )
        if wm_ckpt:
            _wd = torch.load(wm_ckpt, map_location="cpu", weights_only=False)
            self.bundle.wm.load_state_dict(_wd.get("wm", _wd), strict=True)
        self.default_q = self.bundle.default_joint_pos.float().reshape(-1).to(self.device)
        _sc = self.bundle.action_scale.float().reshape(-1).to(self.device)
        if _sc.numel() == 1:
            _sc = _sc.expand_as(self.default_q)
        self.action_scale = _sc
        self._orig_motion = getattr(self.bundle, "motion", None)
        if verbose:
            print(f"[online-grad] bundle ok  default_q[:3]={self.default_q[:3].tolist()} "
                  f"action_scale mean={float(self.action_scale.mean()):.3f}", flush=True)

    # ------------------------------------------------------------------ 核心
    def grad_action(self, state71: torch.Tensor, ref71: torch.Tensor,
                    action_norm: torch.Tensor, horizon: int = 20) -> torch.Tensor:
        """``∂J/∂a`` 对**整块**动作求和后的 (29,) 向量；同时返回标量 J。"""
        from diff_vla_wbc.isaac_joint import FrameRef, default_xi  # noqa: PLC0415

        s0 = state71.to(self.device).float().reshape(-1)[None]
        ref = ref71.to(self.device).float()
        if ref.ndim == 1:
            ref = ref[None]
        H = max(1, min(int(horizon), int(ref.shape[0])))
        ref = ref[:H].expand(H, -1) if ref.shape[0] < H else ref[:H]
        a0 = action_norm.to(self.device).float().reshape(-1)[None].expand(H, -1).clone()

        self.bundle.motion = _OneRefMotion(ref)
        try:
            x = default_xi(1, self.device)
            a = a0.clone().requires_grad_(True)
            s = s0
            J = torch.zeros((), device=self.device)
            prev = a[0]
            for t in range(H):
                s = self.bundle.step(s, a[t][None], x, step_idx=t + 1)
                r = self.bundle.reward(s, a[t][None], prev[None],
                                       FrameRef(torch.tensor([t], device=self.device)))
                J = J + r["total"].sum()
                prev = a[t]
            g = torch.autograd.grad(J, a)[0].sum(0)
            return g.detach(), float(J.detach())
        finally:
            if self._orig_motion is not None:
                self.bundle.motion = self._orig_motion

    # ------------------------------------------------------------------ 施加
    def apply(self, action_norm: torch.Tensor, g: torch.Tensor, *, eta: float,
              cap: float = 0.0, lo: float | None = None,
              hi: float | None = None) -> torch.Tensor:
        """``a + eta·unit(g)``，可选逐维 cap 与动作范围裁剪。"""
        a = action_norm.to(self.device).float().reshape(-1).clone()
        if eta <= 0 or g is None:
            return a
        rms = float(g.pow(2).mean().sqrt())
        if not (rms > 0) or rms != rms:      # 0 或 NaN
            return a
        step = float(eta) * (g / rms)
        if cap and cap > 0:
            step = step.clamp(-float(cap), float(cap))
        out = a + step
        if lo is not None and hi is not None:
            out = out.clamp(float(lo), float(hi))
        return out

    def q_target_from_norm(self, action_norm: torch.Tensor) -> torch.Tensor:
        """归一化动作 → 原始关节目标（给只吃 q_target 的执行器用）。"""
        return self.default_q + self.action_scale * action_norm.to(self.device).float().reshape(-1)
