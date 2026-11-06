#!/usr/bin/env python3
"""官方 kinematic planner (`planner_sonic.onnx`) 的冒烟 + 语义探针。

为什么单独写这个（2026-10-30）：
  官方 GEAR-SONIC 的**训练参考轨迹**（含 root 位置）来自 `planner/target_vel/V2/planner_sonic.onnx`，
  而我们从 LeRobot 日志拼出来的 `ft318_compact.pt` 里 root x/y 恒为 0 —— 这是"走路"这件事
  唯一的缺口。要补它，第一步必须先确认这个 ONNX 在本机跑得起来、输入输出对齐、
  而且**同一命令下能产出真的位移**。

接口（`docs/source/references/planner_onnx.md`）：
    context_mujoco_qpos      [1, 4, 36]   float32  = 4 帧 [root_pos(3), root_quat(4,wxyz), dof(29)]
    target_vel               [1]          float32   <=0 用 mode 默认速度，>0 覆盖（m/s）
    mode                     [1]          int64     0 idle / 1 slowWalk / 2 walk / 3 run / 4 squat ...
    movement_direction       [1, 3]       float32   world 系，只用 x/y
    facing_direction         [1, 3]       float32   world 系，atan2(y,x) 给目标 yaw
    random_seed              [1]          int64
    has_specific_target      [1, 1]       int64
    specific_target_positions[1, 4, 3]    float32
    specific_target_headings [1, 4]       float32   绝对 yaw（rad）
    allowed_pred_num_tokens  [1, 11]      int64     二值掩码，index i == min_tokens(6)+i
    height                   [1]          float32   <0 关闭
输出：
    mujoco_qpos              [1, 64, 36]  float32   30 Hz，**未截断**，只前 num_pred_frames 帧有效
    num_pred_frames          [1]          int64     = num_tokens * 4（24..64）

用法：
    PY=.venv/bin/python   # 需要 onnxruntime
    $PY scripts/probe_planner.py --check        # 只验 I/O 名与形状
    $PY scripts/probe_planner.py --smoke        # 站姿 + 前进 0.4 m/s，跑一次
    $PY scripts/probe_planner.py --sweep        # 几个 mode × 速度，比较"走多远/朝哪"
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANNER = (
    ROOT.parent / "GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
)

# G1 29dof 默认站姿（MuJoCo 树序）：膝关节略弯，脚踝补偿
DEFAULT_DOF = np.array([
    0.0, 0.0, 0.0, 0.42, -0.42, 0.0,      # left leg: hip_p, hip_r, hip_y, knee, ank_p, ank_r
    0.0, 0.0, 0.0, 0.42, -0.42, 0.0,      # right leg
    0.0, 0.0, 0.0,                        # waist yaw/roll/pitch
    0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0,    # left arm: shoulder p/r/y, elbow, wrist r/p/y
    0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0,    # right arm
], dtype=np.float32)
STAND_HEIGHT = 0.75


def build_inputs(ctx: np.ndarray, *, mode: int = 2, target_vel: float = 0.4,
                 move=(1.0, 0.0, 0.0), face=(1.0, 0.0, 0.0), height: float = -1.0,
                 seed: int = 0, allowed=None, targets=None, headings=None) -> dict:
    ctx = ctx.astype(np.float32)
    assert ctx.shape == (4, 36), ctx.shape
    n_tok = 11
    return {
        "context_mujoco_qpos": ctx[None],
        "target_vel": np.array([target_vel], np.float32),
        "mode": np.array([mode], np.int64),
        "movement_direction": np.array([move], np.float32),
        "facing_direction": np.array([face], np.float32),
        "random_seed": np.array([seed], np.int64),
        "has_specific_target": np.array([[0 if targets is None else 1]], np.int64),
        "specific_target_positions": (np.zeros((1, 4, 3), np.float32) if targets is None
                                      else np.asarray(targets, np.float32)[None]),
        "specific_target_headings": (np.zeros((1, 4), np.float32) if headings is None
                                     else np.asarray(headings, np.float32)[None]),
        "allowed_pred_num_tokens": (np.ones((1, n_tok), np.int64) if allowed is None
                                    else np.asarray(allowed, np.int64)[None]),
        "height": np.array([height], np.float32),
    }


def stand_context(height: float = STAND_HEIGHT) -> np.ndarray:
    """4 帧静止站姿：root 在 (0,0,height)，朝向单位四元数。"""
    ctx = np.zeros((4, 36), np.float32)
    ctx[:, 0:3] = (0.0, 0.0, height)
    ctx[:, 3:7] = (1.0, 0.0, 0.0, 0.0)
    ctx[:, 7:36] = DEFAULT_DOF
    return ctx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", type=Path, default=DEFAULT_PLANNER)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    if not a.planner.exists():
        raise SystemExit(f"[fail] 没有 planner: {a.planner}")

    import onnxruntime as ort

    t0 = time.time()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(a.planner), so, providers=["CPUExecutionProvider"])
    print(f"[load] {a.planner.name} {a.planner.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.1f}s")
    print("[in ] " + ", ".join(f"{i.name}{list(i.shape)}" for i in sess.get_inputs()))
    print("[out] " + ", ".join(f"{o.name}{list(o.shape)}" for o in sess.get_outputs()))
    if a.check:
        return 0

    def run(inputs):
        assert set(inputs) == {i.name for i in sess.get_inputs()}, \
            f"输入名不匹配: {sorted(inputs)} vs {[i.name for i in sess.get_inputs()]}"
        t = time.time()
        out = sess.run(None, inputs)
        dt = time.time() - t
        qpos, nf = out[0], out[1]
        n = int(np.asarray(nf).reshape(-1)[0])
        return qpos[0, :n], n, dt

    if a.smoke:
        for name, kw in (
            ("idle  (mode 0, v=0)", dict(mode=0, target_vel=-1.0, move=(0, 0, 0), face=(1, 0, 0))),
            ("walk  (mode 2, v=0.4 前进)", dict(mode=2, target_vel=0.4)),
            ("walk  (mode 2, v=-1 默认)", dict(mode=2, target_vel=-1.0)),
            ("run   (mode 3, v=1.2)", dict(mode=3, target_vel=1.2)),
        ):
            q, n, dt = run(build_inputs(stand_context(), **kw))
            p = q[:, 0:3]
            travel = np.linalg.norm(p[-1, :2] - p[0, :2])
            dur = n / 30.0
            print(f"\n[{name}] frames={n} ({dur:.2f}s @30Hz) {dt * 1e3:.0f}ms")
            print(f"  root x: {p[0,0]:+.3f} → {p[-1,0]:+.3f}   y: {p[0,1]:+.3f} → {p[-1,1]:+.3f}   z: "
                  f"[{p[:,2].min():.3f},{p[:,2].max():.3f}]")
            print(f"  水平位移 {travel * 100:.1f} cm ⇒ 均速 {travel / dur:.3f} m/s")
            print(f"  qpos 有限: {np.isfinite(q).all()}  dof 范围 [{q[:,7:36].min():+.2f},{q[:,7:36].max():+.2f}]")
            print(f"  yaw 起/末: {np.degrees(np.arctan2(2*(q[0,3]*q[0,6]+q[0,4]*q[0,5]), 1-2*(q[0,5]**2+q[0,6]**2))):+.1f}°"
                  f" / {np.degrees(np.arctan2(2*(q[-1,3]*q[-1,6]+q[-1,4]*q[-1,5]), 1-2*(q[-1,5]**2+q[-1,6]**2))):+.1f}°")
        return 0

    if a.sweep:
        ctx = stand_context()
        print(f"\n{'mode':>5}{'v_req':>7}{'frames':>8}{'位移cm':>9}{'均速m/s':>9}{'yaw末°':>9}  ms")
        for mode in (1, 2, 3):
            for v in (-1.0, 0.3, 0.6):
                q, n, dt = run(build_inputs(ctx, mode=mode, target_vel=v))
                p = q[:, 0:3]
                travel = float(np.linalg.norm(p[-1, :2] - p[0, :2]))
                dur = n / 30.0
                w, x, y, z = q[-1, 3:7]
                yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                print(f"{mode:>5}{v:>7.1f}{n:>8}{travel * 100:>9.1f}{travel / dur:>9.3f}{yaw:>9.1f}  {dt * 1e3:.0f}")
        # 转向：movement 与 facing 不一致（横着走 / 原地转）
        print("\n转向测试 (movement=(0,0,0) → 原地按 facing 转):")
        for face in ((1, 0, 0), (0, 1, 0), (-1, 0, 0)):
            q, n, dt = run(build_inputs(ctx, mode=2, target_vel=-1.0, move=(0, 0, 0), face=face))
            w, x, y, z = q[-1, 3:7]
            yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            print(f"  face={face} → yaw末={yaw:+.1f}°  位移={np.linalg.norm(q[-1,0:2]-q[0,0:2])*100:.1f}cm  ({dt*1e3:.0f}ms)")
        return 0

    print("[hint] 用 --check / --smoke / --sweep 之一")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
