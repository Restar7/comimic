#!/usr/bin/env python3
"""A10 回归测试：`eval_wm_multistep` 的旋转误差公式必须能回读真实角度。

历史 bug：`acos(tr)`（应该是 `acos((tr-1)/2)`）⇒ 任何小角度都被算成 0.0°，
于是"姿态预测很好"这个结论建立在错指标上（三个模型 k=1..20 全是 0.0）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_wm_multistep import _quat_to_mat  # noqa: E402


def yaw_q(deg: float) -> T.Tensor:
    h = math.radians(deg) / 2
    return T.tensor([[math.cos(h), 0.0, 0.0, math.sin(h)]], dtype=T.float64)


def rot_err_deg(qa: T.Tensor, qb: T.Tensor) -> float:
    Rp, Rt = _quat_to_mat(qa), _quat_to_mat(qb)
    tr = (Rp.transpose(-1, -2) @ Rt).diagonal(dim1=-2, dim2=-1).sum(-1)
    return math.degrees(math.acos(float(((tr - 1.0) * 0.5).clamp(-1.0, 1.0))))


FAIL = []
print("修正后的公式必须回读原角度：")
for deg in (0.0, 0.5, 5.0, 10.0, 37.1, 90.0, 143.4, 179.0):
    got = rot_err_deg(yaw_q(0.0), yaw_q(deg))
    ok = abs(got - deg) < 1e-6
    print(f"  {'✅' if ok else '❌'} 纯 yaw 差 {deg:6.1f}° → {got:8.4f}°")
    if not ok:
        FAIL.append(deg)

print("\n对照：旧公式 `acos(tr)` 的错误行为（必须把小角度全吃掉）")
for deg in (0.5, 10.0, 37.1):
    Rp, Rt = _quat_to_mat(yaw_q(0.0)), _quat_to_mat(yaw_q(deg))
    tr = (Rp.transpose(-1, -2) @ Rt).diagonal(dim1=-2, dim2=-1).sum(-1)
    old = math.degrees(math.acos(float(tr.clamp(-1, 1).min())))
    print(f"  旧公式 {deg:6.1f}° → {old:.4f}°  (这正是 rot_err 恒为 0 的原因)")

print()
if FAIL:
    print(f"❌ {len(FAIL)} 项失败")
    raise SystemExit(1)
print("✅ 全部通过")
