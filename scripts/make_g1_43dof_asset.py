#!/usr/bin/env python3
"""Generate the **43-dof, position-servo** G1 asset used by both W1 fine-tuning and the closed loop.

Why (2026-10-29, user's question "微调到底带不带手"):
  * W1 (SuperMimic HA fine-tune) was trained on mjlab's ``unitree_g1`` = **29 joints, no hands**;
  * the closed loop runs ``scene_43dof*.xml`` = **44 joints (29 body + 14 fingers)**, with
    hand torque motors and an externally written PD;
  * so the WBC had never seen the hand links' mass/inertia, and the two sides used different
    actuator models. This script removes that asymmetry by emitting a robot-only asset with
    **43 position actuators** whose gains are exactly the ones mjlab trains with.

Gain source (same numbers on both sides, verified):
  ``g1_sim.KP/KD`` in the closed loop are already computed as
  ``kp = armature*ω²``, ``kd = 2ζ*armature*ω`` (``ω = 10 Hz*2π``, ``ζ = 2``) with the ×2
  parallel-linkage groups (waist pitch/roll + both ankles) — identical to mjlab's
  ``STIFFNESS_*/DAMPING_*`` and ``G1_ACTION_SCALE = 0.25*effort/kp``.

Usage:
    python3 scripts/make_g1_43dof_asset.py                 # writes assets/g1_43dof_position.xml
    python3 scripts/make_g1_43dof_asset.py --check          # also re-loads it and prints nu/nq
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 真正的场景/网格在 SONIC_DATA（closed_loop 的 DEFAULT_SCENES 也指这里）；
# 仓里的 groot_repro/scenes 只是残本（meshes/ 下只有 ycb），加载会找不到 STL。
SCENES = Path(
    "/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/SONIC_DATA/deployment/"
    "gear_sonic/data/robot_model/model_data/g1"
)
SRC = SCENES / "g1_29dof_with_hand.xml"
# 必须和源 xml 同目录：里面用相对路径引用 meshes/（挪走就找不到 STL）
OUT = SCENES / "g1_43dof_position.xml"

# ---- mjlab 的四组执行器（数值抄自 mjlab/asset_zoo/robots/unitree_g1/g1_constants.py）----
GROUPS = {
    "5020": dict(kp=14.3, kv=0.91, effort=88.0),
    "7520_14": dict(kp=40.2, kv=2.56, effort=88.0),
    "7520_22": dict(kp=99.1, kv=6.31, effort=139.0),
    "4010": dict(kp=16.8, kv=1.07, effort=25.0),
}
# 并联四连杆（腰 pitch/roll + 双踝）：两枚 5020 之和
GROUPS["2x5020"] = dict(kp=28.6, kv=1.82, effort=176.0)

# 关节名 -> 组（正则，与 mjlab 的 target_names_expr 一一对应）
RULES: list[tuple[str, str]] = [
    (r".*_elbow_joint$", "5020"),
    (r".*_shoulder_pitch_joint$", "5020"),
    (r".*_shoulder_roll_joint$", "5020"),
    (r".*_shoulder_yaw_joint$", "5020"),
    (r".*_wrist_roll_joint$", "5020"),
    (r".*_hip_pitch_joint$", "7520_14"),
    (r".*_hip_yaw_joint$", "7520_14"),
    (r"waist_yaw_joint$", "7520_14"),
    (r".*_hip_roll_joint$", "7520_22"),
    (r".*_knee_joint$", "7520_22"),
    (r".*_wrist_pitch_joint$", "4010"),
    (r".*_wrist_yaw_joint$", "4010"),
    (r"waist_pitch_joint$", "2x5020"),
    (r"waist_roll_joint$", "2x5020"),
    (r".*_ankle_pitch_joint$", "2x5020"),
    (r".*_ankle_roll_joint$", "2x5020"),
]
# 手：与闭环现有的手部 PD 完全一致（`closed_loop_vla_supermimic.py` 里 2.0 / 0.05 / ±2.4）
HAND = dict(kp=2.0, kv=0.05, effort=2.4)


def group_of(joint: str) -> dict:
    if "_hand_" in joint:
        return HAND
    for pat, name in RULES:
        if re.fullmatch(pat, joint):
            return GROUPS[name]
    raise SystemExit(f"[fail] 没有增益分组匹配 {joint}")


def name_geoms(text: str) -> str:
    """给所有 geom 起名，符合 mjlab 的约定（它按名字过滤碰撞体）。

    mjlab 的 `FULL_COLLISION` 只认 ``.*_collision``，脚部又要求 ``(left|right)_foot[1-7]_collision``；
    而 SONIC 的 xml 里 103 个 geom **全是匿名**的，直接喂 mjlab 会在
    `resolve_matching_names` 上 `Not all regular expressions are matched`（2026-10-29 实测）。
    命名规则：
      * 视觉网格（contype=conaffinity=0）→ ``{body}_visual{i}``
      * 脚（``*_ankle_roll_link``）下的接触点   → ``{left|right}_foot{k}_collision``
      * 其余接触体                             → ``{body}_collision``（多个则加序号）
    """
    import re as _re

    tag = _re.compile(r'<body\s+name="[^"]+"[^>]*>|</body>|<geom\b[^>]*?/>', _re.S)
    out, stack = [], []
    counters: dict[str, int] = {}
    pos = 0
    for m in tag.finditer(text):
        out.append(text[pos:m.start()])
        pos = m.end()
        tok = m.group(0)
        if tok.startswith("<body"):
            stack.append(_re.search(r'name="([^"]+)"', tok).group(1))
            out.append(tok)
        elif tok == "</body>":
            stack.pop()
            out.append(tok)
        else:
            body = stack[-1]
            if ' name="' in tok:
                out.append(tok)
                continue
            visual = ('contype="0"' in tok) and ('conaffinity="0"' in tok)
            if visual:
                key = f"{body}_visual"
            elif body.endswith("_ankle_roll_link"):
                key = f"{'left' if body.startswith('left') else 'right'}_foot"
            else:
                key = f"{body}_collision"
            counters[key] = counters.get(key, 0) + 1
            i = counters[key]
            if key.endswith("_foot"):
                name = f"{key}{i}_collision"
            elif key.endswith("_visual"):
                name = f"{key}{i}"
            else:
                name = key if i == 1 else f"{body}_collision{i}"
            out.append(tok.replace("<geom", f'<geom name="{name}"', 1))
    out.append(text[pos:])
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="生成后用 mujoco 重新加载校验")
    a = ap.parse_args()

    text = SRC.read_text()
    count = {"body": 0, "hand": 0}

    def repl(m: re.Match) -> str:
        name, joint, ctrl = m.group(1), m.group(2), m.group(3)
        g = group_of(joint)
        lo, hi = ctrl.split()
        # ctrlrange 是关节角范围（40 关节）/手部力矩范围；position 版要分成 ctrl/for
        if "_hand_" in joint:
            ctrl_range, force_range = f"{lo} {hi}", f"{-g['effort']:.2f} {g['effort']:.2f}"
            count["hand"] += 1
        else:
            ctrl_range, force_range = f"{lo} {hi}", f"{-g['effort']:.0f} {g['effort']:.0f}"
            count["body"] += 1
        return (
            f'<position name="{name}" joint="{joint}" kp="{g["kp"]:g}" kv="{g["kv"]:g}" '
            f'ctrlrange="{ctrl_range}" forcerange="{force_range}" />'
        )

    pattern = re.compile(r'<motor\s+name="([^"]+)"\s+joint="([^"]+)"\s+ctrlrange="([^"]+)"\s*/>')
    new, n = pattern.subn(repl, text)
    if n != 43:
        raise SystemExit(f"[fail] 期望替换 43 个 motor，实际 {n}")

    new = name_geoms(new)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = (
        '<!-- 自动生成：scripts/make_g1_43dof_asset.py\n'
        '     43 个 <position> 执行器：29 身体用 mjlab 的四组 kp/kv（并联腰/踝 ×2），\n'
        '     14 手指 kp=2 kv=0.05（与闭环手部 PD 相同）。robot-only，无场景物体。 -->\n'
    )
    OUT.write_text(header + new)
    print(f"[ok] {OUT}  body_act={count['body']} hand_act={count['hand']}")

    # 训练侧（mjlab）要的是**不带 actuator 的 spec**：actuator 由 mjlab 的
    # BuiltinPositionActuatorCfg 现场建（否则 xml 里的 43 个 + cfg 的 43 个 = 86 个）。
    noact = re.sub(r"\n\s*<actuator>.*?</actuator>", "", new, flags=re.S)
    if len(re.findall(r"<position ", noact)) != 0:
        raise SystemExit("[fail] 去 actuator 失败")
    NACT = OUT.with_name("g1_43dof_noact.xml")
    NACT.write_text(header + noact)
    print(f"[ok] {NACT}  （训练侧用：actuator 由 mjlab cfg 建）")

    if a.check:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(OUT))
        npos = sum(1 for i in range(model.nu) if model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED)
        print(f"[check] nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} "
              f"fixedgain_actuators={npos} timestep={model.opt.timestep}")
        print(f"[check] 关节数 = {model.nq - 7}（应 43：29 身体 + 14 手）")
        for i in range(3):
            print(f"[check] act[{i}] {model.actuator(i).name}: "
                  f"gaintype={model.actuator_gaintype[i]} biastype={model.actuator_biastype[i]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
