#!/usr/bin/env python3
"""配对闭环 A/B 判据（8 对起）：真机漂移 + 任务进度 + 站桩时刻，一次给全。

为什么单独一个脚本
------------------
`judge_ft_vs_base.py` 只看漂移 + 训练指标；`eval_closedloop_{drift,progress}.py` 只看单臂。
实际判 A/B 需要**同 seed 配对**的三件事一起看，而且必须带**符号检验**——
4 个 seed 全变好也可能是抽签（本轮 `--sample-avg 4` 就是这样：4 对 4/4 变好，
补到 8 对后变成 4/8，均值和方差都变差）。

用法::

    python3 scripts/judge_paired_closedloop.py --a-tag base --b-tag base_sa4 --seeds 0 1 2 3 4 5 6 7
    python3 scripts/judge_paired_closedloop.py --a-tag base --b-tag ftv10 --seeds 0 1 2 3 \\
        --metric pos_p50        # 只看漂移

判据（写死，不许事后改）
------------------------
* 漂移：B 更好 **≥6/8 对** 且配对差均值为负；
* 进度：B 更好 **≥6/8 对** 且配对差均值为正；
* 任何一项不过 ⇒ 该臂不可进交付物。
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

MIN_WINS_FRAC = 0.75      # 8 对 → 6 对


def sign_test_p(wins: int, n: int) -> float:
    """双侧符号检验（H0: 各 50%）。n 小，用精确二项。"""
    if n == 0:
        return float("nan")
    k = max(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-tag", required=True, help="基线臂前缀（目录 a_tag_s{seed}）")
    ap.add_argument("--b-tag", required=True, help="对照臂前缀")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--root", type=Path, default=ROOT / "runs/closedloop_r3")
    ap.add_argument("--drift-packed", type=Path,
                    default=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt")
    ap.add_argument("--metric", default="pos_p50", choices=["pos_p50", "ori_p50"],
                    help="漂移指标（eval_closedloop_drift.analyse 的键）")
    ap.add_argument("--stall-speed", type=float, default=0.06)
    ap.add_argument("--until", type=int, default=0,
                    help="只统计前 N 步的漂移（0=全部；用来只看接近段）")
    a = ap.parse_args()

    import torch  # noqa: PLC0415

    from analyze_closedloop_stall import analyse as stall_analyse, demo_paths  # noqa: PLC0415
    from eval_closedloop_drift import analyse as drift_analyse  # noqa: PLC0415

    ref = torch.load(str(a.drift_packed), map_location="cpu", weights_only=False)
    ref = torch.as_tensor(ref["ref"]).float().numpy()
    demo = demo_paths()

    rows = []
    for s in a.seeds:
        da, db = a.root / f"{a.a_tag}_s{s}", a.root / f"{a.b_tag}_s{s}"
        if not (da / "dump/steps.jsonl").exists() or not (db / "dump/steps.jsonl").exists():
            print(f"[pair] seed {s}: 缺数据（A={'有' if (da/'dump/steps.jsonl').exists() else '无'} "
                  f"B={'有' if (db/'dump/steps.jsonl').exists() else '无'}）")
            continue
        ma = dict(drift_analyse(da, ref, until=(a.until or None)))
        mb = dict(drift_analyse(db, ref, until=(a.until or None)))
        sa = stall_analyse(da, demo, a.stall_speed)
        sb = stall_analyse(db, demo, a.stall_speed)
        rows.append(dict(seed=s, a=ma, b=mb, sa=sa, sb=sb))

    if len(rows) < 2:
        print("[pair] ❌ 可配对 seed < 2")
        return 2
    n = len(rows)
    print(f"[pair] {a.a_tag} vs {a.b_tag}：{n} 对（同 seed = 同 reset）\n")
    hdr = (f"{'seed':>4}  {'A drift':>8} {'B drift':>8} {'Δdrift':>8}   "
           f"{'A prog':>7} {'B prog':>7} {'Δprog':>7}   {'A stall':>8} {'B stall':>8}")
    print(hdr)
    print("-" * len(hdr))
    ddrift, dprog = [], []
    for r in rows:
        d = r["b"][a.metric] - r["a"][a.metric]
        p = r["sb"]["prog_end"] - r["sa"]["prog_end"]
        ddrift.append(d)
        dprog.append(p)
        print(f"{r['seed']:>4}  {r['a'][a.metric]*100:8.2f} {r['b'][a.metric]*100:8.2f} "
              f"{d*100:+8.2f}   {r['sa']['prog_end']:7.3f} {r['sb']['prog_end']:7.3f} "
              f"{p:+7.3f}   {r['sa']['stall_s']:8.2f} {r['sb']['stall_s']:8.2f}")
    ddrift, dprog = np.asarray(ddrift), np.asarray(dprog)

    # 漂移（越小越好）
    wins_d = int((ddrift < 0).sum())
    need = max(2, math.ceil(MIN_WINS_FRAC * n))
    pd_ = sign_test_p(wins_d, n)
    # 进度（越大越好）
    wins_p = int((dprog > 0).sum())
    pp_ = sign_test_p(wins_p, n)
    print(f"\n[drift {a.metric}] B 更好 {wins_d}/{n}（判据 ≥{need}）；配对差 "
          f"{ddrift.mean()*100:+.2f} cm ± {ddrift.std(ddof=1)*100:.2f}；符号检验 p={pd_:.3f}")
    print(f"[prog  prog_end] B 更好 {wins_p}/{n}（判据 ≥{need}）；配对差 "
          f"{dprog.mean():+.3f} ± {dprog.std(ddof=1):.3f}；符号检验 p={pp_:.3f}")
    sd_a = float(np.std([r["sa"]["prog_end"] for r in rows], ddof=1))
    sd_b = float(np.std([r["sb"]["prog_end"] for r in rows], ddof=1))
    print(f"[prog  方差] A sd {sd_a:.3f} → B sd {sd_b:.3f}（{sd_b/sd_a:.2f}×）")

    ok_d = wins_d >= need and ddrift.mean() < 0
    ok_p = wins_p >= need and dprog.mean() > 0
    print(f"\n[判定] 漂移改善 {'✅' if ok_d else '❌'}   进度改善 {'✅' if ok_p else '❌'}")
    print("[判定] " + ("✅ B 臂成立，可进交付物" if (ok_d or ok_p) else
                       "❌ B 臂不成立（两项都没过）"))
    return 0 if (ok_d or ok_p) else 3


if __name__ == "__main__":
    raise SystemExit(main())
