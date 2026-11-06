#!/usr/bin/env python3
"""``diff_vla_wbc/state_extra.py`` 的单元测试（46 → 52 维的"参考相对姿态误差"）。

跑法：``python3 scripts/test_state_extra.py``（不需要 GPU、不加载模型，秒级）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diff_vla_wbc.state_extra import (  # noqa: E402
    STATE_EXTRA_DIM,
    STATE_EXTRA_SLICE,
    abs6_from_quat,
    mat_to_rot6d_wire,
    rotlocal6,
    mat_to_rot6d,
    quat_to_mat,
    rel_ori6,
    write_state_extra,
    yaw_error_from_rel6,
)

FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name} {detail}")
        FAIL.append(name)


def yaw_quat(deg: float) -> torch.Tensor:
    """绕 z 转 deg 度的四元数（wxyz）。"""
    h = math.radians(deg) * 0.5
    return torch.tensor([[math.cos(h), 0.0, 0.0, math.sin(h)]], dtype=torch.float64)


def main() -> int:
    I = torch.eye(3, dtype=torch.float64)
    print("[1] 基本恒等")
    r6 = mat_to_rot6d(I.unsqueeze(0))
    check("R=I → [1,0,0,0,1,0]", torch.allclose(r6, torch.tensor([[1.0, 0, 0, 0, 1, 0]], dtype=torch.float64)))

    print("[2] 相对姿态：机器人 yaw=0，参考 yaw=+30°")
    q_r, q_f = yaw_quat(0.0), yaw_quat(30.0)
    rel = rel_ori6(q_r, q_f)
    err = math.degrees(float(yaw_error_from_rel6(rel)))
    check("Δyaw 恢复 = +30°", abs(err - 30.0) < 1e-6, f"got {err:.6f}")
    c, s = float(rel[0, 0]), float(rel[0, 1])
    check("前两行是 Rz(30°) 结构 [cos,−sin,0,sin,cos,0]",
          abs(c - math.cos(math.radians(30))) < 1e-9 and abs(s + math.sin(math.radians(30))) < 1e-9,
          f"c={c:.6f} s={s:.6f}")

    print("[3] 方向性：机器人超前参考 ⇒ 负值")
    rel2 = rel_ori6(yaw_quat(45.0), yaw_quat(15.0))
    err2 = math.degrees(float(yaw_error_from_rel6(rel2)))
    check("Δyaw = −30°", abs(err2 + 30.0) < 1e-6, f"got {err2:.6f}")

    print("[4] 6D → 旋转可重建（Gram-Schmidt），误差 < 1e-9")
    torch.manual_seed(0)
    qr = torch.randn(64, 4, dtype=torch.float64)
    qf = torch.randn(64, 4, dtype=torch.float64)
    rel = rel_ori6(qr, qf)
    a, b = rel[..., 0:3], rel[..., 3:6]
    e1 = a / a.norm(dim=-1, keepdim=True)
    b2 = b - (e1 * b).sum(-1, keepdim=True) * e1
    e2 = b2 / b2.norm(dim=-1, keepdim=True)
    e3 = torch.cross(e1, e2, dim=-1)
    R_rec = torch.stack((e1, e2, e3), dim=-2)
    R_true = quat_to_mat(qr).transpose(-1, -2) @ quat_to_mat(qf)
    d = (R_rec - R_true).abs().max().item()
    check("重建最大误差 < 1e-9", d < 1e-9, f"max|Δ|={d:.3e}")

    print("[5] 写槽位")
    st = torch.zeros(4, 1, 132)
    out = write_state_extra(st, torch.arange(6.0).repeat(4, 1))
    check("46:52 被写入", torch.allclose(out[0, 0, 46:52], torch.arange(6.0)))
    check("45 与 52 仍是 0", float(out[0, 0, 45]) == 0.0 and float(out[0, 0, 52]) == 0.0)
    check("原张量未被就地改动", float(st.abs().sum()) == 0.0)
    flat = write_state_extra(torch.zeros(2, 132), torch.ones(2, 6))
    check("(B,132) 也支持", float(flat[1, 46:52].sum()) == 6.0)

    print("[6] 槽位常量自洽")
    a, b = STATE_EXTRA_SLICE
    check("(46,52) 且宽度 6", (a, b) == (46, 52) and b - a == STATE_EXTRA_DIM)

    print("[7] 梯度有限（ace 恒等处不能炸）")
    qr = yaw_quat(0.0).clone().requires_grad_(True)
    r = rel_ori6(qr, yaw_quat(0.0))
    loss = r.sum() + yaw_error_from_rel6(r).sum()
    loss.backward()
    g = qr.grad
    check("梯度全是有限值", torch.isfinite(g).all().item(), str(g))

    print("[8] abs6 必须与**线上** rot6d 逐位一致（列块口径 m[:, :2].reshape(6)）")
    try:
        from scipy.spatial.transform import Rotation as _R
        import numpy as _np
        for deg in (0.0, 45.0, -30.0, 120.0):
            q = yaw_quat(deg)[0].numpy()
            m = _R.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            wire = m[:, :2].reshape(6).astype(_np.float32)
            mine = abs6_from_quat(torch.tensor(q, dtype=torch.float64)[None])[0].numpy()
            check(f"yaw={deg:g}° 逐位一致", bool(_np.allclose(wire, mine, atol=1e-6)),
                  f"{_np.round(wire, 5).tolist()} vs {_np.round(mine, 5).tolist()}")
        # 行口径与列口径**必须不同**（否则说明我们搞错了哪个是线上口径）
        q = yaw_quat(45.0)
        row = mat_to_rot6d(quat_to_mat(q))[0]
        col = mat_to_rot6d_wire(quat_to_mat(q))[0]
        check("行口径 ≠ 列口径（提醒文档）", not torch.allclose(row, col),
              f"row={row.tolist()} col={col.tolist()}")
    except ImportError:
        print("  (跳过：环境里没有 scipy)")

    print("[9] rotlocal6 必须与**线上协议**逐位一致（rot6d(conj(heading(q0))·q_now)）")
    try:
        import numpy as _np
        _sys_path = str(Path(__file__).resolve().parents[1] / "third_party/HumanoidArena/isaaclab_twist2_g1")
        sys.path.insert(0, _sys_path)
        from action_provider.vla_robot_current_local_runtime_v3 import (
            build_vla_rotlocal_v3_observation_state)
        for q0d, q1d in ((-88.0, -63.0), (-100.0, -100.0), (30.0, 95.0), (170.0, -170.0)):
            q0 = yaw_quat(q0d)[0].numpy(); q1 = yaw_quat(q1d)[0].numpy()
            ref = build_vla_rotlocal_v3_observation_state(
                initial_robot_orientation_wxyz=q0, root_orientation_wxyz=q1,
                joint_pos_canonical_29=_np.zeros(29, _np.float32),
                joint_vel_canonical_29=_np.zeros(29, _np.float32))[:6]
            mine = rotlocal6(torch.tensor(q0)[None], torch.tensor(q1)[None])[0].numpy()
            check(f"q0={q0d:g}° q1={q1d:g}° 逐位一致", bool(_np.allclose(ref, mine, atol=1e-6)),
                  f"{_np.round(ref,5).tolist()} vs {_np.round(mine,5).tolist()}")
        check("开局(未转动)恒为 identity",
              torch.allclose(rotlocal6(yaw_quat(-77.0), yaw_quat(-77.0)),
                             torch.tensor([[1.0, 0, 0, 1, 0, 0]], dtype=torch.float64), atol=1e-9))
    except ImportError:
        print("  (跳过：找不到第三方实现)")

    print()
    if FAIL:
        print(f"❌ {len(FAIL)} 项失败: {FAIL}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
