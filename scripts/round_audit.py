#!/usr/bin/env python3
"""**逐轮审计**：从日志里把每一轮真实生效的配置挖出来，和"应该是什么"逐项对照。

为什么需要（2026-10-31）
------------------------
多轮闭环里最危险的不是崩溃，而是**静默漂移**：某一轮的 B 配比悄悄从 25% 变成 19%、
A 线文件数变了却没跟着改、新数据权重没生效……跑完照样出结果，只是结果不可比。

这个脚本只读日志、不跑任何训练，把每轮的：
  * A 线文件数 / 行数
  * 阶段2 实际 B 占比（设计值 vs 实测）
  * 阶段2b 的 k / 步数
  * VLA 训练步数 / H / 冻结 WBC / 奖励配置 / 信任域 / 新数据权重
  * 最终评测：候选 vs π0 的配对差、t、胜率、p90
列成一张表，任何一项偏离预期都会标红。

用法：
    python3 scripts/round_audit.py                     # 审计所有有评测结果的轮次
    python3 scripts/round_audit.py --rounds d1 d2 d2b
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECT = {
    "wm_steps_a": 3000,
    "wm_steps_mix": 3000,
    "wm_steps_2b": 1500,
    "wm_k": 20,
    "vla_steps": 800,
    "H": 8,
    "train_wbc": False,
    "tail_weight": 2.0,
    "root_ori_mode": "split",
    "trust_center": "self",
}
B_SHARE_MIN, B_SHARE_MAX = 0.22, 0.30


def _grep(path: Path, pat: str):
    try:
        return re.findall(pat, path.read_text(encoding="utf-8", errors="ignore"))
    except FileNotFoundError:
        return []


def audit_round(r: str) -> dict | None:
    pd = ROOT / "runs/S3_pipeline" / r
    out: dict = {"round": r}
    # A 线
    p1 = pd / "4a_wm_p1.log"
    m = _grep(p1, r"\[a-line\] 加载 (\d+) 条")
    out["a_traj"] = int(m[-1]) if m else None
    # B 占比（阶段2）
    m = _grep(pd / "4b_wm_p2.log", r"\[traj\] 合计 (\d+) 条轨迹（B=(\d+) A=(\d+)）")
    if m:
        tot, b, a = (int(x) for x in m[-1])
        out["b_share"] = b / max(1, tot)
        out["b_n"], out["a_n"] = b, a
    # k / 步数（阶段2b）
    m = _grep(pd / "4_wm_refresh.log", r"\[loss\] K=(\d+)")
    out["wm_k"] = int(m[-1]) if m else None
    steps = _grep(pd / "4_wm_refresh.log", r"^\s+\[\s*(\d+)\] loss=")
    out["wm_steps_2b"] = int(steps[-1]) if steps else None
    # VLA 训练 args
    aj = ROOT / "runs/S3_loop" / f"{r}_joint/train_main/args.json"
    if not aj.exists():
        aj = ROOT / "runs/S3_loop" / r / "train_main/args.json"
    if aj.exists():
        a = json.load(open(aj))
        _src = str(a.get("init_vla") or "")
        out["init_src"] = (Path(_src).parents[2].name if "S3_loop" in _src else "base(π0)")
        out.update(vla_steps=a.get("steps"), H=a.get("horizon"),
                   train_wbc=a.get("train_wbc"), tail_weight=a.get("tail_weight"),
                   root_ori_mode=a.get("root_ori_mode"),
                   trust_center=a.get("trust_center"), act_eps=a.get("act_eps"),
                   posture_weight=a.get("posture_weight"))
        tm = aj.parent / "train_metrics.jsonl"
        rows = [json.loads(l) for l in tm.open()] if tm.exists() else []
        if rows:
            out["act_rmse_end"] = rows[-1].get("act/rmse")
            out["trust_rmse_end"] = rows[-1].get("trust/rmse")
    # 评测
    for tag in (f"pi0_vs_{r}_avg4", f"pi0_vs_{r}_ablation"):
        sd = ROOT / "runs/S3_loop/sweeps" / tag / "summary.json"
        if sd.exists():
            d = json.load(open(sd))
            m_, c_ = d["main"], d["ctrl"]
            dif = [y - x for x, y in zip(m_["per_seed"], c_["per_seed"])]
            if not dif:
                continue
            mu = st.mean(dif)
            sem = st.stdev(dif) / len(dif) ** 0.5 if len(dif) > 1 else float("nan")
            out.update(cand_mean=c_["root_mean"], pi0_mean=m_["root_mean"],
                       paired=mu, t=mu / sem if sem else None,
                       better=sum(1 for x in dif if x < 0), n=len(dif),
                       cand_p90=sorted(c_["per_seed"])[int(len(dif) * 0.9)],
                       pi0_p90=sorted(m_["per_seed"])[int(len(dif) * 0.9)])
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs="*", default=None)
    args = ap.parse_args()
    if args.rounds:
        rounds = args.rounds
    else:
        rounds = sorted(p.parent.name for p in (ROOT / "runs/S3_pipeline").glob("*/state.json"))
    rows = [r for r in (audit_round(x) for x in rounds) if r]

    print("=" * 100)
    print("=" * 108)
    hdr = ["轮次", "起点", "A线轨迹", "B占比", "k", "2b步数", "VLA步", "H",
           "冻结WBC", "tail", "ori", "trust", "act_rmse", "trust_rmse"]
    print("".join(f"{h:>9}" for h in hdr))
    print("-" * 108)
    bad = []
    for r in rows:
        bs = r.get("b_share")
        flag = ""
        if bs is not None and not (B_SHARE_MIN <= bs <= B_SHARE_MAX):
            flag = " ❌配比"
            bad.append(f"{r['round']}: B 占比 {bs*100:.1f}% 超出 {B_SHARE_MIN*100:.0f}~{B_SHARE_MAX*100:.0f}%")
        for k, want in (("wm_k", EXPECT["wm_k"]), ("vla_steps", EXPECT["vla_steps"]),
                        ("H", EXPECT["H"]), ("tail_weight", EXPECT["tail_weight"]),
                        ("root_ori_mode", EXPECT["root_ori_mode"]),
                        ("trust_center", EXPECT["trust_center"]),
                        ("train_wbc", EXPECT["train_wbc"])):
            got = r.get(k)
            if got is not None and got != want:
                flag += f" ❌{k}={got}"
                bad.append(f"{r['round']}: {k}={got} 期望 {want}")
        # 链路完整性：除了第一条有记录的轮次，起点都不应该是 base(π0)
        if r.get("init_src") == "base(π0)" and r is not rows[0]:
            flag += " ❌链路断(从π0起)"
            bad.append(f"{r['round']}: init_src=base(π0)，链式继承断了")
        if r.get("posture_weight") not in (None, 0, 0.0):
            flag += " ❌posture"
            bad.append(f"{r['round']}: posture_weight={r['posture_weight']} 应为 0")

        cells = [
            str(r["round"]),
            str(r.get("init_src") or "—"),
            str(r.get("a_traj") or "—"),
            (f"{bs*100:.1f}%" if bs is not None else "—"),
            str(r.get("wm_k") or "—"),
            str(r.get("wm_steps_2b") or "—"),
            str(r.get("vla_steps") or "—"),
            str(r.get("H") or "—"),
            str(r.get("train_wbc")),
            str(r.get("tail_weight") or "—"),
            str(r.get("root_ori_mode") or "—"),
            str(r.get("trust_center") or "—"),
            (f"{r['act_rmse_end']:.3f}" if r.get("act_rmse_end") else "—"),
            (f"{r['trust_rmse_end']:.3f}" if r.get("trust_rmse_end") else "—"),
        ]
        print("".join(f"{c:>9}" for c in cells) + flag)

    print("-" * 108)
    hdr2 = ["轮次", "候选均值", "π0均值", "配对差", "t", "胜率", "候选p90", "π0 p90"]
    print("".join(f"{h:>10}" for h in hdr2))

    for r in rows:
        if r.get("paired") is None:
            continue
        cells = [
            str(r["round"]),
            f"{r['cand_mean']:.3f}",
            f"{r['pi0_mean']:.3f}",
            f"{r['paired']:+.3f}",
            (f"{r['t']:+.2f}" if r.get("t") is not None else "—"),
            str(r["better"]) + "/" + str(r["n"]),
            f"{r['cand_p90']:.3f}",
            f"{r['pi0_p90']:.3f}",
        ]
        print("".join(f"{c:>10}" for c in cells))

    print()
    if bad:
        print(f"❌ {len(bad)} 处偏离预期：")
        for x in bad:
            print("   -", x)
        return 1
    print("✅ 所有轮次的配置与设计一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
