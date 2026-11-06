#!/usr/bin/env python3
"""e2：hinge 尾部"越过 cap 仍有梯度"的回归测试（旧行为 clamp 会在远场把梯度变成 0）。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diff_vla_wbc.reward_shape import ROOT_POS_CAP, ROOT_POS_THR, hinge_tail  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(("  ✅ " if cond else "  ❌ ") + name + ("" if cond else f" {detail}"))
    if not cond:
        FAIL.append(name)


def curve(cap_slope: float, xs):
    """给定 dpos，返回 tail 与 d(tail)/d(dpos)。"""
    out, grads = [], []
    for x in xs:
        rob = torch.zeros(1, 7, dtype=torch.float64)
        ref = torch.zeros(1, 7, dtype=torch.float64)
        ref[0, 0] = float(x)          # 只差位置
        ref[0, 3] = 1.0
        rob[0, 3] = 1.0
        rob.requires_grad_(True)
        t, _ = hinge_tail(rob, ref, cap_slope=cap_slope)
        g, = torch.autograd.grad(t.sum(), rob)
        out.append(float(t)); grads.append(float(-g[0, 0]))   # 惩罚随 dpos 增大而增大
    return out, grads


xs = [0.2, 0.5, 1.0, 1.35, 1.8, 2.5]
old_v, old_g = curve(0.0, xs)
new_v, new_g = curve(0.5, xs)

print("[1] slope=0 必须与旧行为一致（远场梯度为 0）")
check("dpos=1.8/2.5 处梯度 == 0", abs(old_g[4]) < 1e-12 and abs(old_g[5]) < 1e-12,
      f"g={old_g[4]:.3e},{old_g[5]:.3e}")
check("dpos=1.8 与 1.35 的惩罚相等（平台）",
      abs(old_v[4] - old_v[3]) < 1e-12, f"{old_v[3]:.4f} vs {old_v[4]:.4f}")

print("[2] slope=0.5：越过 cap 仍有半速梯度，且惩罚单调")
check("远场梯度 > 0", new_g[4] > 0 and new_g[5] > 0, f"g={new_g[4]:.3f},{new_g[5]:.3f}")
check("远场斜率 = 0.5（线性延续）", abs(new_g[5] - 0.5) < 1e-9, f"g={new_g[5]:.6f}")
check("cap 之前与旧行为逐位相同", all(abs(a - b) < 1e-12 for a, b in zip(old_v[:4], new_v[:4])),
      f"{old_v[:4]} vs {new_v[:4]}")
check("单调不减", all(new_v[i] <= new_v[i + 1] + 1e-12 for i in range(len(new_v) - 1)), str(new_v))
check("cap 之后仍严格递增", new_v[4] > new_v[3] + 1e-9, f"{new_v[3]:.4f} vs {new_v[4]:.4f}")

print("[3] 阈值以下不该有惩罚（不能把'跟得好'也罚）")
check(f"dpos<{ROOT_POS_THR} 时 tail==0", abs(new_v[0]) < 1e-12, f"{new_v[0]:.3e}")

print()
if FAIL:
    print(f"❌ {len(FAIL)} 项失败")
    raise SystemExit(1)
print("✅ 全部通过")
