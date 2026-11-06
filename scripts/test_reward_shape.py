#!/usr/bin/env python3
"""`diff_vla_wbc/reward_shape.py` 的自检（不需要 Isaac / GPU，纯 CPU 数学）。

跑法：`python3 scripts/test_reward_shape.py`，全过退出码 0。
"""
from __future__ import annotations

import math
import sys

import torch

sys.path.insert(0, ".")
from diff_vla_wbc.reward_shape import (  # noqa: E402
    hinge_tail, ori_split_reward, quat_to_up, quat_to_yaw, tilt_yaw_yawrate_err, wrap_pi,
    _ramp_then_slope,
)

OK = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global OK, FAIL
    if cond:
        OK += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {detail}")


def quat_from_yaw(yaw: float) -> torch.Tensor:
    return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def quat_from_rollpitch(roll: float, pitch: float) -> torch.Tensor:
    """yaw=0 的 roll/pitch 四元数（wxyz）。"""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return torch.tensor([cr * cp, sr * cp, cr * sp, -sr * sp])


def make_state(quat: torch.Tensor, *, pos=(0.0, 0.0, 0.8), wz: float = 0.0,
               p_xyz=None) -> torch.Tensor:
    s = torch.zeros(71)
    p = torch.tensor(p_xyz if p_xyz is not None else pos, dtype=torch.float32)
    s[0:3] = p
    s[3:7] = quat
    s[39:42] = torch.tensor([0.0, 0.0, wz])
    return s


def main() -> int:
    print("[1] 四元数工具")
    q0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    check("单位四元数 up=(0,0,1)", torch.allclose(quat_to_up(q0), torch.tensor([0.0, 0.0, 1.0]), atol=1e-6))
    check("单位四元数 yaw=0", abs(float(quat_to_yaw(q0))) < 1e-6)
    for y in (0.3, -1.2, 2.9, -3.0):
        check(f"yaw={y:+.2f} 复原", abs(float(quat_to_yaw(quat_from_yaw(y))) - y) < 1e-5)
    check("wrap_pi(3.5)≈-2.78", abs(float(wrap_pi(torch.tensor(3.5))) + (2 * math.pi - 3.5)) < 1e-6)

    print("[2] 同一姿态 → 三项误差全 0")
    ref = make_state(quat_from_rollpitch(0.3, -0.2))          # 有点 roll/pitch
    rob = ref.clone()
    e = tilt_yaw_yawrate_err(rob.unsqueeze(0), ref.unsqueeze(0))
    check("tilt2=0", float(e["tilt2"]) < 1e-8)
    check("yaw2=0", float(e["yaw2"]) < 1e-8)
    check("yawrate2=0", float(e["yawrate2"]) < 1e-8)

    print("[3] 纯 yaw 偏移：只影响 yaw，不影响 tilt")
    root = quat_from_rollpitch(0.3, -0.2)
    for dy in (10.0, 30.0, 90.0):
        qrot = quat_from_yaw(math.radians(dy))
        # 世界系左乘一个 yaw 旋转：w' = q_yaw ⊗ w
        w1, x1, y1, z1 = qrot
        w2, x2, y2, z2 = root
        q = torch.tensor([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                          w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                          w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
        rob = make_state(q)
        e = tilt_yaw_yawrate_err(rob.unsqueeze(0), ref.unsqueeze(0))
        check(f"dyaw={dy:>4.0f}° → tilt2≈0", float(e["tilt2"]) < 1e-4, f"tilt2={float(e['tilt2']):.2e}")
        check(f"dyaw={dy:>4.0f}° → yaw2≈dy²", abs(float(e["yaw2"]) - math.radians(dy) ** 2) < 1e-3,
              f"yaw2={float(e['yaw2']):.4f} 期望={math.radians(dy) ** 2:.4f}")

    print("[4] 纯 roll/pitch 偏移：只影响 tilt")
    rob = make_state(quat_from_rollpitch(0.3 + math.radians(15), -0.2))
    e = tilt_yaw_yawrate_err(rob.unsqueeze(0), ref.unsqueeze(0))
    check("tilt2>0", float(e["tilt2"]) > 1e-4)
    check("yaw2≈0", float(e["yaw2"]) < 1e-4, f"yaw2={float(e['yaw2']):.2e}")

    print("[5] 拆项奖励：完美跟踪 = w_tilt+w_yaw+w_yawrate")
    rob = ref.clone()
    r, _ = ori_split_reward(rob.unsqueeze(0), ref.unsqueeze(0))
    # 跟着常量走（原来硬编码 1.5，权重一改就误报）
    import diff_vla_wbc.reward_shape as _RS
    _exp_perfect = _RS.W_TILT + _RS.W_YAW + _RS.W_YAWRATE
    check(f"完美跟踪 r={_exp_perfect:.3f}", abs(float(r) - _exp_perfect) < 1e-5, f"r={float(r):.5f}")

    print("[6] yaw 项必须活着（A1：σ 0.25 管近场，远场交给 hinge λ）")
    q = quat_from_yaw(math.radians(20))
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = ref[3:7]
    qr = torch.tensor([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                       w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                       w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                       w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
    rob = make_state(qr)
    r, _ = ori_split_reward(rob.unsqueeze(0), ref.unsqueeze(0))
    # 手算口径跟着**当前常量**走（原来硬编码 0.2/0.25/0.3，改权重后必然对不上）
    import diff_vla_wbc.reward_shape as _RS
    _hinge20 = _RS.YAW_HINGE_LAMBDA * max(0.0, math.radians(20) - math.radians(_RS.YAW_HINGE_THR_DEG))
    expect = (_RS.W_TILT + _RS.W_YAW * math.exp(-(math.radians(20)) ** 2 / _RS.YAW_STD ** 2)
              + _RS.W_YAWRATE - _hinge20)
    check("dyaw=20° 的 r 与手算一致(含 hinge)", abs(float(r) - expect) < 1e-4,
          f"r={float(r):.5f} 期望={expect:.5f}")

    # ---- P1 的意图必须被测试钉住，否则下一个人改回去没人知道 ----
    check("W_YAWRATE 已关（实测 corr(w_z)=0.005）", _RS.W_YAWRATE == 0.0,
          f"W_YAWRATE={_RS.W_YAWRATE}")
    check("tilt 权重已下调 (<=0.5)", _RS.W_TILT <= 0.5, f"W_TILT={_RS.W_TILT}")
    check("yaw 权重 >= tilt（最重要的量）", _RS.W_YAW >= _RS.W_TILT,
          f"W_YAW={_RS.W_YAW} vs W_TILT={_RS.W_TILT}")
    # A1 的核心不是"exp 在远处还很大"，而是"**远处也必须有梯度**"：
    #   exp 在 37.1° 处已经饱和（1.2e-3），信号由 hinge 承担 —— 所以验 hinge 的贡献与梯度。
    check("σ_yaw 保持紧凑 0.25（近场精度不丢）", abs(_RS.YAW_STD - 0.25) < 1e-9,
          f"YAW_STD={_RS.YAW_STD}")
    _h37 = _RS.YAW_HINGE_LAMBDA * max(0.0, math.radians(37.1) - math.radians(_RS.YAW_HINGE_THR_DEG))
    check("hinge 在 37.1° 处贡献 >= 0.05（活着）", _h37 >= 0.05, f"贡献={_h37:.4f}")
    # 梯度检查：对 |dyaw| 求导，远超 thr 之后必须恒为 λ（不是 0）
    _a = torch.tensor([math.radians(37.1), math.radians(143.4)], dtype=torch.float64, requires_grad=True)
    _cap = math.radians(_RS.YAW_HINGE_CAP_DEG)
    _hinge = _ramp_then_slope(torch.relu(_a.abs() - math.radians(_RS.YAW_HINGE_THR_DEG)),
                              _cap, _RS.YAW_HINGE_SLOPE)
    _g, = torch.autograd.grad(_hinge.sum(), _a)
    check("hinge 在 37.1°/143.4° 处梯度 > 0（远场不饱和）",
          float(_g[0]) > 0 and float(_g[1]) > 0, f"g={_g.tolist()}")
    check("hinge 梯度有限", bool(torch.isfinite(_g).all()), f"g={_g.tolist()}")

    print("[7] 尾部 hinge：阈值内恒 0，超出线性、有上限")
    # 位置
    for d, expect_zero in ((0.20, True), (0.35, True), (0.50, False), (2.0, False)):
        rob = make_state(q0, p_xyz=(d, 0.0, 0.8))
        ref0 = make_state(q0, p_xyz=(0.0, 0.0, 0.8))
        t, terms = hinge_tail(rob.unsqueeze(0), ref0.unsqueeze(0))
        if expect_zero:
            check(f"‖Δp‖={d:.2f} m 尾=0", float(terms["root_pos_tail"]) < 1e-6)
        else:
            check(f"‖Δp‖={d:.2f} m 尾>0", float(terms["root_pos_tail"]) > 1e-6,
                  f"={float(terms['root_pos_tail']):.3f}")
    rob = make_state(q0, p_xyz=(3.0, 0.0, 0.8))
    ref0 = make_state(q0, p_xyz=(0.0, 0.0, 0.8))
    _, terms = hinge_tail(rob.unsqueeze(0), ref0.unsqueeze(0))
    check("位置尾被 cap 到 1.0", abs(float(terms["root_pos_tail"]) - 1.0) < 1e-3)

    # 倾斜
    for deg, expect_zero in ((10.0, True), (20.0, True), (30.0, False), (80.0, False)):
        rob = make_state(quat_from_rollpitch(math.radians(deg), 0.0))
        ref0 = make_state(q0)
        _, terms = hinge_tail(rob.unsqueeze(0), ref0.unsqueeze(0))
        if expect_zero:
            check(f"tilt={deg:.0f}° 尾=0", float(terms["root_tilt_tail"]) < 1e-6)
        else:
            check(f"tilt={deg:.0f}° 尾>0", float(terms["root_tilt_tail"]) > 1e-6,
                  f"={math.degrees(float(terms['root_tilt_tail'])):.1f}°")
    rob = make_state(quat_from_rollpitch(math.radians(89.0), 0.0))
    _, terms = hinge_tail(rob.unsqueeze(0), ref0.unsqueeze(0))
    check("倾斜尾被 cap 到 57°", abs(math.degrees(float(terms["root_tilt_tail"])) - 57.0) < 1.0)

    print("[8] 批量一致性（B=8，逐条与单条一致）")
    B = 8
    refs = torch.stack([make_state(quat_from_rollpitch(0.1 * i, -0.05 * i)) for i in range(B)])
    robs = torch.stack([make_state(quat_from_rollpitch(0.1 * i + 0.2, -0.05 * i),
                                   p_xyz=(0.01 * i, 0.0, 0.8)) for i in range(B)])
    rb, _ = ori_split_reward(robs, refs)
    single = [float(ori_split_reward(robs[i:i + 1], refs[i:i + 1])[0]) for i in range(B)]
    check("批量与逐条一致", torch.allclose(rb, torch.tensor(single), atol=1e-6))

    print("[9] **梯度**在完美跟踪处必须有限（历史 bug：acos 的导数在 x→1 发散 ⇒ NaN 训练）")
    rob = ref.clone().unsqueeze(0).requires_grad_(True)
    r, terms = ori_split_reward(rob, ref.unsqueeze(0))
    tot = r.sum() + terms["tilt2"].sum() * 0.0  # 只用 r
    # 再用 hinge_tail 也走一遍
    t, _ = hinge_tail(rob, ref.unsqueeze(0))
    (r.sum() + t.sum()).backward()
    g = rob.grad
    check("完美跟踪: 奖励梯度全有限", bool(torch.isfinite(g).all()),
          f"含 {int((~torch.isfinite(g)).sum())} 个非有限值")
    check("完美跟踪: 奖励梯度≈0", float(g.abs().max()) < 1e-4, f"max|g|={float(g.abs().max()):.3e}")

    # 略微偏离时梯度也应有限
    rob2 = (ref + 1e-3 * torch.randn_like(ref)).unsqueeze(0).requires_grad_(True)
    r2, _ = ori_split_reward(rob2, ref.unsqueeze(0))
    t2, _ = hinge_tail(rob2, ref.unsqueeze(0))
    (r2.sum() + t2.sum()).backward()
    check("轻微偏离: 梯度全有限", bool(torch.isfinite(rob2.grad).all()))

    print(f"\n{'✅ 全部通过' if FAIL == 0 else '❌ 有失败'}：{OK} 过 / {FAIL} 挂")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
