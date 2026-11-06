#!/usr/bin/env python3
"""B 线 episode 的**健康过滤 + 硬门**（2026-10-29 停链后的修复项之一）。

为什么必须有它：R3..R6 四轮的策略在 80 集闭环上摔倒率 2/80 → 4/80、平均奖励
0.0859(π0) 单调掉到 0.0743(π5)。而**坏 episode 会顺着 `wm_mix`（B 线占 45%）
喂给 WM**，让"摔"变成分布的一部分，越训越差。

更严重的是**判据字段用错了**：`eval.json` 里的 `fell` 是**单集** eval（同任务同起点
跑一条 600 tick），那条恰好不摔就一直是 False；真正的摔倒记录在每条 dump 的
``meta['fell']`` 里。本模块只认后者。

用法（两种）：
    # 1) 打印健康报告 + 门判定（退出码 1 = 不过）
    python3 scripts/bline_health.py --dump 'runs/j1_scene2/round_5/bline/b_*.pt'

    # 2) 作为库调用，拿到"可用 episode 清单"
    from scripts.bline_health import select_dumps
    good, rep = select_dumps(files)
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 默认阈值：宁可少用几条，也不要把摔的集喂进去
# 2026-10-30：参考库换成 ft318_compact_v3（根轨迹是真的、会走）之后，pos 误差的分布形状
# 和 v1（参考恒为原地）**不一样**，所以这两个门限必须用 v3 的第一批采集重新推导。
# 先用环境变量临时覆盖、看统计量，再把推导出的值写回这里（不要凭感觉沿用 15 cm）。
MAX_TILT_DEG = float(os.environ.get("BHEALTH_MAX_TILT_DEG", 45.0))
MAX_POS_P50_CM = float(os.environ.get("BHEALTH_MAX_POS_P50_CM", 15.0))
REQUIRE_NO_FELL = True


def _recompute_fell(d, meta) -> bool:
    """按新判据重算「是否摔倒」：倾角>60° 或（z<0.30 且倾角>30°）。

    修正旧 dump 的 meta（"min_z < 0.45" 会把 task4 的深蹲误判成摔倒）。
    """
    try:
        import math  # noqa: PLC0415

        import torch  # noqa: PLC0415

        S = d["states"].float()
        q = S[:, 3:7]
        w, x, y, z = q.unbind(-1)
        up_z = 1.0 - 2.0 * (x * x + y * y)
        tilt = torch.acos(up_z.clamp(-1.0, 1.0)) * 180.0 / math.pi
        return bool(float(tilt.max()) > 60.0
                    or (float(S[:, 2].min()) < 0.30 and float(tilt.max()) > 30.0))
    except Exception:  # noqa: BLE001
        return bool(meta.get("fell", False))


def episode_stats(path: str) -> dict:
    """单条 dump 的健康指标（只读 meta + rewards + states，不碰库）。"""
    import torch  # noqa: PLC0415

    d = torch.load(path, map_location="cpu", weights_only=False)
    meta = d.get("meta", {}) or {}
    return {
        "path": path,
        # 2026-10-31：**按 dump 里的 states 重算**，不信 meta —— 因为旧 dump 的 meta 是用
        # "min_z < 0.45" 判的，会把深蹲（task4 捡地上铝罐，骨盆正常就 0.44）误判成摔倒。
        "fell": _recompute_fell(d, meta),
        "max_tilt_deg": float(meta.get("max_tilt_deg", 0.0) or 0.0),
        "reward": float(d["rewards"].mean()) if "rewards" in d else float("nan"),
        "task": str(meta.get("task", "")),
    }


def select_dumps(files, *, max_tilt_deg: float = MAX_TILT_DEG,
                 max_pos_p50_cm: float = MAX_POS_P50_CM,
                 require_no_fell: bool = REQUIRE_NO_FELL,
                 root_pos_p50_cm: dict | None = None):
    """返回 (good_files, report)。``root_pos_p50_cm`` 可传入预先算好的漂移（省一次库加载）。"""
    stats = [episode_stats(f) for f in files]
    good, kept, dropped = [], [], []
    for s in stats:
        why = None
        if require_no_fell and s["fell"]:
            why = "fell"
        elif s["max_tilt_deg"] > max_tilt_deg:
            why = f"tilt {s['max_tilt_deg']:.1f}>{max_tilt_deg:.0f}"
        elif root_pos_p50_cm is not None and root_pos_p50_cm.get(s["path"], 0.0) > max_pos_p50_cm:
            why = f"pos_p50 {root_pos_p50_cm[s['path']]:.1f}>{max_pos_p50_cm:.0f}cm"
        if why:
            dropped.append({**s, "why": why})
        else:
            good.append(s["path"])
            kept.append(s)
    rw = [s["reward"] for s in kept if s["reward"] == s["reward"]]
    rep = {
        "n_total": len(stats), "n_good": len(good), "n_dropped": len(dropped),
        "fell_total": sum(1 for s in stats if s["fell"]),
        "reward_mean": st.mean(rw) if rw else float("nan"),
        "reward_se": (st.stdev(rw) / len(rw) ** 0.5) if len(rw) > 1 else float("nan"),
        "max_tilt_max": max((s["max_tilt_deg"] for s in stats), default=0.0),
        "dropped": dropped[:20],
        "good": good,
    }
    return good, rep


def gate(report: dict, *, max_fell: int = 0, min_reward: float | None = None) -> tuple[bool, str]:
    """**每轮硬门**：摔倒数、平均奖励。返回 (ok, reason)。"""
    if report["fell_total"] > max_fell:
        return False, f"摔倒 {report['fell_total']}/{report['n_total']} > {max_fell}"
    if min_reward is not None and report["reward_mean"] < min_reward:
        return False, (f"80 集奖励 {report['reward_mean']:.4f} < 基线 {min_reward:.4f}"
                       f"（差 {report['reward_mean'] - min_reward:+.4f}）")
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="glob：B 线 dump")
    ap.add_argument("--max-tilt", type=float, default=MAX_TILT_DEG)
    ap.add_argument("--max-fell", type=int, default=0, help="允许的摔倒集数（默认 0）")
    ap.add_argument("--min-reward", type=float, default=None,
                    help="低于此值判失败；建议填 π0 基线（0.0859）")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json", action="store_true", help="只输出 json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.dump))
    if not files:
        raise SystemExit(f"[fail] 没匹配到：{args.dump}")
    good, rep = select_dumps(files, max_tilt_deg=args.max_tilt)
    ok, why = gate(rep, max_fell=args.max_fell, min_reward=args.min_reward)
    rep["gate_ok"], rep["gate_reason"] = ok, why
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    if args.json:
        print(json.dumps({k: v for k, v in rep.items() if k != "good"}, indent=1, ensure_ascii=False))
    else:
        print(f"[bline-health] {args.dump}")
        print(f"  集数 {rep['n_total']}（可用 {rep['n_good']}，剔除 {rep['n_dropped']}）"
              f" | 摔倒 {rep['fell_total']} | 最大倾角 {rep['max_tilt_max']:.1f}°")
        print(f"  可用集平均奖励 {rep['reward_mean']:.4f} ± {rep['reward_se']:.4f}")
        for d in rep["dropped"][:8]:
            print(f"  剔除 {Path(d['path']).name}: {d['why']}")
        print(f"  门判定: {'PASS' if ok else 'FAIL'} —— {why}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
