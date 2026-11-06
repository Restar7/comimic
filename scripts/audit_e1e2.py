#!/usr/bin/env python3
"""e1/e2 落地的**逐项静态审查**（只读文件，不加载模型、不训练）。

为什么要它：用户反复要求"确保 e1/e2 全部改完"。口头保证没有意义，这个脚本把
每一项改动**定位到具体文件的具体行**，缺了就报红——包括那些"配置写了但代码没读"的
静默失效（d1–d3 的"新数据×3"就是这么废掉三轮的）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "state_extra": ROOT / "diff_vla_wbc/state_extra.py",
    "v3": ROOT / "diff_vla_wbc/v3_step.py",
    "joint": ROOT / "scripts/train_joint_v2.py",
    "server": ROOT / "scripts/ha_vla_latent64_server.py",
    "pipe": ROOT / "scripts/s3_pipeline.sh",
    "launch": ROOT / "scripts/launch_joint_ha_v2.sh",
    "shape": ROOT / "diff_vla_wbc/reward_shape.py",
    "isaac": ROOT / "diff_vla_wbc/isaac_joint.py",
    "verify": ROOT / "scripts/s3_verify_round.sh",
    "extras": ROOT / "diff_vla_wbc/b_start.py",
}

# (分组, 说明, 文件, 正则)
ITEMS = [
    # ---------------- e1：观测扩维 ----------------
    ("e1", "6 维切片常量 (46,52)", "state_extra", r"STATE_EXTRA_SLICE = \(46, 52\)"),
    ("e1", "线上口径 rot6d（列块）", "state_extra", r"def mat_to_rot6d_wire"),
    ("e1", "abs6 = rot6d(R)", "state_extra", r"def abs6_from_quat"),
    ("e1", "rotlocal6（线上协议同款）", "state_extra", r"def rotlocal6\("),
    ("e1", "rotlocal6 接入 b_start（q0）", "extras", r'"q0": S\[0, 3:7\]'),
    ("e1", "rotlocal6 接入 trainer", "v3", r'extra6_for\("rotlocal6"'),
    ("e1", "server rotlocal6 分支", "server", r'_mode in \("abs6", "rotlocal6"\)'),
    ("e1", "pipeline 默认 rotlocal6", "pipe", r"STATE_EXTRA=\$\{STATE_EXTRA:-rotlocal6\}"),
    ("e1", "extra6_for 分派 abs6/vel6/ori6", "state_extra", r'if m == "abs6"'),
    ("e1", "V3Config.state_extra", "v3", r"state_extra: str = \"none\""),
    ("e1", "V3Config 缩放初值=1（P3）", "v3", r"state_extra_scale_init: float = 1.0"),
    ("e1", "缩放参数 + 进优化器", "v3", r"self\.optimizer\.add_param_group"),
    ("e1", "写 46:52（训练/评测同一路）", "v3", r"def _patch_state_extra"),
    ("e1", "abs6 分支走线上编码", "v3", r'extra6_for\("abs6"'),
    ("e1", "ckpt 保存缩放", "v3", r'"state_extra_scale"'),
    ("e1", "CLI --state-extra", "joint", r'"--state-extra"'),
    ("e1", "从 --init-vla 恢复缩放", "joint", r"_state_extra_scale_ckpt"),
    ("e1", "server 读 ckpt 缩放", "server", r"self\.state_extra_scale = None"),
    ("e1", "server 写 46:52 (rotlocal6)", "server", r'_mode in \("abs6", "rotlocal6"\)'),
    ("e1", "server 拒收缺 6 维（vel6）", "server", r"但请求里没有"),
    ("e1", "server / 握手暴露口径", "server", r'"state_extra": str\(getattr'),
    ("e1", "pipeline STATE_EXTRA 进链", "pipe", r'STATE_EXTRA="\$STATE_EXTRA"'),
    ("e1", "pipeline 指纹打印", "pipe", r"观测扩维"),
    ("e1", "launcher 透传 --state-extra", "launch", r'--state-extra "\$STATE_EXTRA"'),
    # ---------------- e2：奖励改造 ----------------
    ("e2", "cap 后保留斜率", "shape", r"def _ramp_then_slope"),
    ("e2", "hinge_tail 支持 cap_slope", "shape", r"cap_slope: float = 0\.0"),
    ("e2", "reward() 传 tail_cap_slope", "isaac", r'tail_cap_slope", 0\.0'),
    ("e2", "CLI --tail-cap-slope", "joint", r'"--tail-cap-slope"'),
    ("e2", "bundle.tail_cap_slope 赋值", "joint", r"bundle\.tail_cap_slope = "),
    ("e2", "CLI --adv-baseline", "joint", r'"--adv-baseline"'),
    ("e2", "CLI --adv-tau", "joint", r'"--adv-tau"'),
    ("e2", "V3Config.adv_baseline", "v3", r"adv_baseline: str = \"none\""),
    ("e2", "配对基线实现（执行 token）", "v3", r"_roll_exec = imagine_with_tokens"),
    ("e2", "有界 tanh 目标", "v3", r"torch\.tanh\(\(adv \+ boot\) / _tau\)"),
    ("e2", "adv 统计进指标", "v3", r'"adv/std_ratio"'),
    ("e2", "pipeline 三开关进链", "pipe", r'ADV_BASELINE="\$ADV_BASELINE"'),
    ("e2", "pipeline 指纹打印", "pipe", r"奖励改造"),
    ("e2", "launcher 透传三参数", "launch", r'--adv-baseline "\$ADV_BASELINE"'),
    # ---------------- 验收 ----------------
    ("验收", "verifier 验 e1 缩放", "verify", r"但 ckpt 里没有 state_extra_scale"),
    ("验收", "verifier 验评测口径一致", "verify", r"无法证明评测也用了"),
    ("验收", "verifier 验 adv 真跑了", "verify", r"但 train_metrics\.jsonl 里没有 adv/std_ratio"),
]
TESTS = [ROOT / "scripts/test_state_extra.py", ROOT / "scripts/test_tail_cap_slope.py",
         ROOT / "scripts/test_reward_shape.py"]


def main() -> int:
    text = {}
    for k, f in FILES.items():
        text[k] = f.read_text(errors="ignore") if f.exists() else ""
    rows, bad = [], []
    for grp, desc, key, pat in ITEMS:
        hit = re.search(pat, text[key]) is not None
        rows.append((grp, desc, FILES[key].name, hit))
        if not hit:
            bad.append(f"{desc}（{FILES[key].name} 里找不到 /{pat}/）")
    for t in TESTS:
        ok = t.exists()
        rows.append(("验收", f"测试文件 {t.name}", t.name, ok))
        if not ok:
            bad.append(f"缺测试文件 {t.name}")

    w = max(len(r[1]) for r in rows)
    grp_now = None
    for grp, desc, fn, hit in rows:
        if grp != grp_now:
            print(f"\n== {grp} ==")
            grp_now = grp
        print(f"  {'✅' if hit else '❌'} {desc:<{w}}  [{fn}]")
    n_ok = sum(1 for r in rows if r[3])
    print(f"\n合计 {n_ok}/{len(rows)} 项就位")
    if bad:
        print(f"\n❌ {len(bad)} 项缺失：")
        for b in bad:
            print(f"   - {b}")
        return 1
    print("✅ e1/e2 静态审查全过（每一项都能指到具体文件与表达式）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
