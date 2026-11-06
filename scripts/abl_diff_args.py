#!/usr/bin/env python3
"""**消融的"只翻一个旋钮"机械校验**：比较两次训练的 args.json，列出差异键。

为什么必须有它：本项目最大的教训就是"一次改 5 样 ⇒ 提升了也归因不了"
（d3 就是 5 个旋钮一起动，AB1–AB4 全是为了补这个洞）。而消融的难点不在跑，
在**你以为只翻了一个**——lambda_r 曾经在 stage_joint 里写死 0.3，
AB3 会静默地变成"完整方法再跑一遍"，看上去一切正常、结论却是假的。

用法：
  # 只看差在哪（不判对错）
  python3 scripts/abl_diff_args.py runs/S3_loop/d3_joint/train_main/args.json \
                                    runs/S3_loop/abl1_joint/train_main/args.json

  # 断言"只允许 act_weight 不同"（多一个/少一个都算失败）
  python3 scripts/abl_diff_args.py <base> <target> --expect act_weight

退出码：0=符合期望；1=出现了期望之外的差异（**这条消融作废**）；2=期望的键没变。
"""
from __future__ import annotations

import argparse
import json
import sys

# 与"方法组件"无关、每次跑都会变的键（目录/设备/端口）。其余任何差异都要人看。
IGNORE = {"out", "device", "port_tokens", "time_budget_minutes", "eval_out", "gate_out"}


def load(p: str) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return str(a) == str(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="参考臂 args.json（消融用 d3 = 完整方法）")
    ap.add_argument("target", help="消融臂 args.json")
    ap.add_argument("--expect", default="", help="允许不同的键，逗号分隔（如 act_weight）")
    a = ap.parse_args()

    import os

    if not (os.path.isfile(a.base) and os.path.isfile(a.target)):
        print(f"[abl-diff] ❌ 文件不存在: {a.base if not os.path.isfile(a.base) else a.target}")
        return 2

    base, tgt = load(a.base), load(a.target)
    keys = (set(base) | set(tgt)) - IGNORE
    expect = {k.strip() for k in a.expect.split(",") if k.strip()}

    diff, missing = [], []
    for k in sorted(keys):
        if k not in base or k not in tgt:
            missing.append(k)
            continue
        if not same(base[k], tgt[k]):
            diff.append((k, base[k], tgt[k]))

    print(f"[abl-diff] base  = {a.base}")
    print(f"[abl-diff] target= {a.target}")
    if missing:
        print(f"[abl-diff] ⚠ 只在一边存在的键: {missing}")
    if not diff:
        print("[abl-diff] 没有任何差异 —— 这条消融等于把完整方法又跑了一遍")
    for k, x, y in diff:
        mark = "✅" if k in expect else "❌"
        print(f"  {mark} {k}: {x} → {y}")

    got = {k for k, _, _ in diff}
    if expect and got == expect:
        print(f"[abl-diff] ✅ 只翻了期望的旋钮: {sorted(got)}")
        return 0
    if expect and not (got & expect):
        print(f"[abl-diff] ❌ 期望的键没变（{'/'.join(sorted(expect))}）⇒ 这个开关没接通")
        return 2
    if expect:
        extra = got - expect
        absent = expect - got
        if extra:
            print(f"[abl-diff] ❌ 期望之外还动了: {sorted(extra)} ⇒ 这条消融作废")
        if absent:
            print(f"[abl-diff] ❌ 期望的键没变: {sorted(absent)} ⇒ 开关没接通")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
