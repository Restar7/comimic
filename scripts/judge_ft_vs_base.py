#!/usr/bin/env python3
"""**预设判据**：联合微调到底有没有真收益（2026-11-05 事先写死，禁止事后改）。

判据（两项都要满足，任一不满足就是"不成立"）
-------------------------------------------
1. **动作在可信域内**：训练末尾的 `act/rmse` 落在 **[0.05, 0.15]**
   （≈1.2–3.8°/关节）。低于 0.05 = 被钉死在 base 上（实测 v6：0.013，想象回报完全不涨）；
   高于 0.15 = 动作已经偏得执行器跟不上（实测 v5：0.80 ≈ 20°/关节）。
2. **配对闭环更好**：同 seed 配对，`ft` 的**真机位置漂移 p50** 比 `base` 小的对数 **≥6/8**，
   且配对差均值为负（漂移越小越好）。

为什么必须配对
--------------
闭环 seed 间方差极大：base 三个 seed 的位置漂移 4.78 / 33.81 / 31.05 cm，而组间差异只有 ~6 cm。
**同 seed = 同 reset**，配对差能把方差压掉一大截；非配对比较分辨不出 ±20%。

用法::
    python3 scripts/judge_ft_vs_base.py --ft runs/joint_v8 --ft-tag ftv8 \
        --base-tag base --seeds 0 1 2 3 4 5 6 7
退码：0=成立 / 3=不成立 / 2=数据不全
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

ACT_LO, ACT_HI = 0.05, 0.15
MIN_WINS = 6  # 8 对里至少 6 对（对数不足 8 时按 75% 取整，下限 2）


def train_act_drift(run: Path, tail: int = 30) -> float | None:
    p = run / "train_metrics.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    v = [r["act/rmse"] for r in rows[-tail:] if r.get("act/rmse") is not None]
    return float(np.mean(v)) if v else None


def load_ref(packed: Path):
    import torch  # noqa: PLC0415
    pk = torch.load(str(packed), map_location="cpu", weights_only=False)
    return torch.as_tensor(pk["ref"]).float().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", type=Path, required=True, help="FT 的 run 目录（读 train_metrics）")
    ap.add_argument("--ft-tag", required=True, help="闭环目录前缀，如 ftv8 → ftv8_s0")
    ap.add_argument("--base-tag", default="base")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--packed", type=Path,
                    default=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt")
    ap.add_argument("--root", type=Path, default=ROOT / "runs/closedloop_r3")
    a = ap.parse_args()

    from eval_closedloop_drift import analyse  # noqa: PLC0415

    ref = load_ref(a.packed)
    act = train_act_drift(a.ft)
    print(f"[judge] FT run={a.ft}  训练末尾 act/rmse(近30均) = "
          f"{'<缺>' if act is None else f'{act:.4f}'}  目标区 [{ACT_LO}, {ACT_HI}]")

    rows = []
    for s in a.seeds:
        b = a.root / f"{a.base_tag}_s{s}"
        f = a.root / f"{a.ft_tag}_s{s}"
        mb = analyse(b, ref) if (b / "dump" / "steps.jsonl").exists() else None
        mf = analyse(f, ref) if (f / "dump" / "steps.jsonl").exists() else None
        if mb is None or mf is None:
            print(f"[judge] seed {s}: 缺数据（base={'有' if mb else '无'} ft={'有' if mf else '无'}）")
            continue
        rows.append((s, mb, mf))

    if len(rows) < 2:
        print("[judge] ❌ 可配对的 seed 不足 2 对，无法判定")
        return 2

    print(f"\n[judge] 配对结果（位置漂移 p50，越小越好；单位 cm）")
    print(f"  seed   base     ft      配对差(ft-base)   谁更好")
    diffs = []
    for s, mb, mf in rows:
        d = mf["pos_p50"] - mb["pos_p50"]
        diffs.append(d)
        print(f"  {s:>4}  {mb['pos_p50']*100:6.2f}  {mf['pos_p50']*100:6.2f}   "
              f"{d*100:+8.2f}        {'ft' if d < 0 else 'base'}")
    diffs = np.asarray(diffs)
    wins = int((diffs < 0).sum())
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if len(diffs) > 1 else float("nan")
    se = sd / np.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
    print(f"\n[judge] 配对差：均值 {mean*100:+.2f} cm  sd {sd*100:.2f}  se {se*100:.2f}  "
          f"ft 更好 {wins}/{len(diffs)} 对")

    # ---- 两项判据 ----
    min_wins = MIN_WINS if len(diffs) >= 8 else max(2, int(np.ceil(0.75 * len(diffs))))
    g1 = act is not None and ACT_LO <= act <= ACT_HI
    g2 = wins >= min_wins and mean < 0
    print(f"\n[judge] 判据① 动作在可信域 [{ACT_LO},{ACT_HI}]：{'✅' if g1 else '❌'}"
          f"（act/rmse={'?' if act is None else f'{act:.4f}'}）")
    print(f"[judge] 判据② 配对 ≥{min_wins}/{len(diffs)} 更好且均值为负：{'✅' if g2 else '❌'}"
          f"（{wins}/{len(diffs)}，均值 {mean*100:+.2f} cm）")

    if g1 and g2:
        print("\n[judge] ✅ **联合微调成立**：动作在可信域内，且配对闭环真机漂移显著更小。")
        return 0
    print("\n[judge] ❌ **联合微调不成立**（判据事先写死，不许事后解释）。")
    if not g1:
        print("        动作不在可信域：<0.05 是被钉死（想象回报不涨），>0.15 是偏得执行器跟不上。")
    if not g2:
        print("        配对闭环没有一致的改善 ⇒ 想象里的收益没有落到真机。")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
