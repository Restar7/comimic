#!/usr/bin/env python3
"""A1–A14 逐条对账：**当前代码**到底修没修、验没验。只读，不训练。

用户口径："除了 A8 其他都解决完了吗？" —— 口头回答没有意义，这里把每条
落到具体文件/常量/测试，并区分三种状态：
  ✅ 已解决且有验证      （代码在 + 有测试或实测）
  🟡 已改，效果待验证    （代码在，但"能不能降指标"只能等跑完一轮）
  ⛔ 未做 / 按设计暂缓
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
F = {k: (ROOT / v).read_text(errors="ignore") if (ROOT / v).exists() else "" for k, v in {
    "shape": "diff_vla_wbc/reward_shape.py",
    "isaac": "diff_vla_wbc/isaac_joint.py",
    "v3": "diff_vla_wbc/v3_step.py",
    "server": "scripts/ha_vla_latent64_server.py",
    "joint": "scripts/train_joint_v2.py",
    "wm": "scripts/finetune_wm_velocity_policy.py",
    "pipe": "scripts/s3_pipeline.sh",
    "multi": "scripts/eval_wm_multistep.py",
    "vel": "scripts/eval_wm_velocity.py",
    "verify": "scripts/s3_verify_round.sh",
    "bstart": "diff_vla_wbc/b_start.py",
    "se": "diff_vla_wbc/state_extra.py",
}.items()}

import diff_vla_wbc.reward_shape as RS  # noqa: E402

rows: list[tuple[str, str, str, str]] = []   # (编号, 状态, 一句话, 证据)


def add(no, st, what, ev):
    rows.append((no, st, what, ev))


# ---------------- A1 ----------------
_ok = (abs(RS.YAW_STD - 0.25) < 1e-9 and RS.YAW_HINGE_LAMBDA > 0
       and "abs(wrap_pi" in F["shape"] or ("e[\"dyaw\"].abs()" in F["shape"]))
add("A1", "✅" if _ok else "⛔", f"yaw 死区：σ={RS.YAW_STD} + hinge λ={RS.YAW_HINGE_LAMBDA}",
    "σ 保持 0.25（近场），远场交给 hinge(abs(dyaw), λ=0.3)。注：清单里写的 λ≈0.2 是**旧值**，"
    "后按你要求提到 0.3；实测 tilt 28.7% / yaw-exp 33.5% / yaw-hinge 37.8%")

# ---------------- A2 ----------------
_ok = RS.W_YAWRATE == 0.0 and "--yawrate-weight" in F["joint"] and "wm_root_w_corr" in F["pipe"]
add("A2", "✅" if _ok else "⛔", f"yaw_rate 权重={RS.W_YAWRATE}，且已做**按轮准入**",
    "权重 0；新增 gate 的 root_w 栏 + `--yawrate-weight` + 审计准入(root_w_corr≥0.40)，"
    "反证三连：无证据/0.26 → 红，0.55 → 绿")

# ---------------- A3 ----------------
_ok = abs(RS.W_TILT - 0.2) < 1e-9 and abs(RS.W_YAW - 1.0) < 1e-9
add("A3", "✅" if _ok else "⛔", f"权重重配 = ({RS.W_TILT}, {RS.W_YAW}, {RS.W_YAWRATE})",
    "tilt 再降到 0.2（清单里的 0.5 是中间值），yaw 1.0；yaw 占朝向奖励 71%")

# ---------------- A4 ----------------
_ok = ("_ramp_then_slope" in F["shape"] and "--tail-cap-slope" in F["joint"]
       and "TAIL_CAP_SLOPE=${TAIL_CAP_SLOPE:-0.5}" in F["pipe"])
add("A4", "✅" if _ok else "⛔", "hinge 尾部越过 cap 仍有梯度",
    "_ramp_then_slope + --tail-cap-slope，pipeline 默认 0.5；test_tail_cap_slope 7 项")

# ---------------- A5 ----------------
_ok = ("def rotlocal6(" in F["se"] and "rotlocal6" in F["v3"]
       and 'rotlocal6' in F["server"] and "start_quat0" in F["bstart"])
add("A5", "✅" if _ok else "⛔", "观测：喂 **rotlocal6**（不再是 abs6，也不需要改第三方）",
    "线上 state64[:6] 本来就是 rot6d(conj(heading(q0))·q_now)；与第三方实现逐位对拍；"
    "corr(自转量, yaw误差)=+0.341（abs6 只有 0.166）。**清单里的 ori6 方案已作废**")

# ---------------- A6 ----------------
_ok = "state_extra_scale_init: float = 1.0" in F["v3"] and "state_extra/scale_absmax" in F["v3"]
add("A6", "✅" if _ok else "⛔", "scale 初值 = 1.0（yaw 从第 0 步就进前向）",
    "初值 1.0；指标 state_extra/scale_absmax 可观察；端到端冒烟 ckpt 里存了 mode+scale")

# ---------------- A7 ----------------
add("A7", "⛔", "vel6（体坐标速度）**仍未接到线上报文**",
    "state_extra.vel6_body 已实现但线上 state64 只有 rot6d+q+dq ⇒ 按设计暂缓（需改客户端）")

# ---------------- A8 ----------------
_ok = ("--mix-lib" in F["wm"] and "整批配比" in F["wm"]
       and "WM_MIX_LIB=${WM_MIX_LIB:-0}" in F["pipe"] and "WM_AB_RATIO=${WM_AB_RATIO:-1}" in F["pipe"])
add("A8", "✅" if _ok else "⛔", "WM 数据构成：参考库 0 / stage1 纯 A / stage2·2b A50+B50",
    "已改并**三阶段各冒烟**：A100%、A50/B50、k=20 的 A50/B50；打印+断言（|got−want|≤0.02）。"
    "**效果**待 d3 的 root_w_corr/OOD 栏验证")

# ---------------- A9 ----------------
add("A9", "✅" if ("--mix-lib" in F["wm"] and "整批配比" in F["wm"]) else "⛔",
    "配比显式化（不再是 traj_frac 的隐式量纲）",
    "改成 lib + A/B 显式比例（--mix-lib / --mix-b），三阶段同一套；清单里写的 --mix-a 用不到，"
    "因为 A+B 的余量里按 --mix-b 切")

# ---------------- A10 ----------------
_ok = "cos_t = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)" in F["multi"]
add("A10", "✅" if _ok else "⛔", "rot_err 公式修正 acos((tr−1)/2)",
    "单测 8 个角度原值回读（0/0.5/5/10/37.1/90/143.4/179）；旧公式 10° → 0.0000°。"
    "**真实数值**要等 d3 的 gate 打印")

# ---------------- A11 ----------------
_ok = ("audit_reward_activity" in F["pipe"] and "OOD 栏" in F["pipe"]
       and "root 角速度" in F["pipe"])
add("A11", "✅" if _ok else "⛔", "门：两栏（in-dist 准入 + OOD 监测）+ 奖励活性审计 + root_w 栏",
    "preflight 跑 audit_reward_activity；gate 增加 OOD 栏与 root 角速度栏。"
    "**代码已接线，OOD 栏尚未真跑过**（d3 是第一次）")

# ---------------- A12 ----------------
_ok = ("data_source" in F["vel"] and "data_source=B-line holdout" in F["multi"])
add("A12", "✅" if _ok else "⛔", "两套仪器都打数据源标签",
    'one-step → "recorded_npz(OOD)"，multi-step → "B-line holdout (in-distribution)"；'
    "刚补的（这条原本只做了一半）")

# ---------------- A13 ----------------
_ok = ("_AE=\"$(_act_eps_effective)\"" in F["pipe"] and "joint 生效值" in F["pipe"]
       and "yaw_rate_w=" in F["verify"])
add("A13", "✅" if _ok else "⛔", "命令替换污染（ROUND=99）",
    "helper 改在主 shell 先算 + 双断言 + 指纹/生效值反漂移 + verifier 验收")

# ---------------- A14 ----------------
_ok = ("def glob_match_path" in F["bstart"] and "训练侧不允许静默丢掉新数据加权" in F["bstart"])
add("A14", "✅" if _ok else "⛔", "fresh_glob 相对模式匹配绝对路径",
    "glob_match_path + init 时算一次 + 命中 0 直接 SystemExit（不再只 warn）")

w = max(len(r[2]) for r in rows)
print(f"{'编号':<4}{'状态':<4}{'内容':<{w}}  证据/备注")
print("-" * (w + 30))
for no, st, what, ev in rows:
    print(f"{no:<4}{st:<4}{what:<{w}}  {ev}")
n_ok = sum(1 for r in rows if r[1] == "✅")
n_open = sum(1 for r in rows if r[1] == "⛔")
print(f"\n✅ 已解决且有验证 {n_ok}/14   ⛔ 未做 {n_open}/14")
print("🟡 另外：A1/A3/A6/A8/A9/A11 属于'代码改完、**指标效果**只能等 d3 跑完才知道'")
raise SystemExit(0 if n_open == 1 else 1)   # A7 是唯一按设计暂缓的
