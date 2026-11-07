#!/usr/bin/env python3
"""P1 轮的关键指标看板：从训练日志里抽 G1–G4 四个门要的数。

用法：
    python3 scripts/watch_p1.py --run joint_p1            # 打印轨迹 + 门判定
    python3 scripts/watch_p1.py --run joint_p1 --tail 5   # 只看最后 5 条

判据（先写死）：
  G1 选择性   pred_resid_ratio 明显 >0，且 loss/action_target < 0.7×tgt_resid_norm²
  G2 动作健康 act/rmse ∈ [0.05, 0.15]（<0.02 钉死 / >0.5 出流形）；token 位移只作参考
  G3 critic   critic/v_ev > 0.90
  G4 自举语义  critic/v_adv_H_raw 不恒正（>0 的比例 < 80%）
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TOKEN_RMS = 0.11          # 服务端实测 tok_absmean≈0.11（40×64 的 RMS 同量级）


def rows_from(path: Path):
    out = []
    for line in path.open():
        if not line.startswith("[v2 "):
            continue
        i = line.find("{")
        if i < 0:
            continue
        try:
            out.append(json.loads(line[i:]))
        except json.JSONDecodeError:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="joint_p1")
    ap.add_argument("--log", default="")
    ap.add_argument("--tail", type=int, default=12)
    a = ap.parse_args()
    path = Path(a.log) if a.log else Path("logs") / f"{a.run}.log"
    if not path.exists():
        print(f"(还没有日志：{path})"); return 0
    R = rows_from(path)
    if not R:
        print(f"(日志里还没有 [v2 ...] 行：{path})"); return 0

    cols = [
        ("act/rmse", "act"), ("trust/rmse", "tokΔ"), ("critic/v_ev", "v_ev"),
        ("action_target/pred_resid_ratio", "predRes"), ("action_target/tgt_resid_ratio", "tgtRes"),
        ("loss/action_target", "L_at"), ("critic/v_adv_H_raw", "advH"),
        ("rollout/reward", "rew"), ("action_var/ratio", "varRat"), ("diverge/pos_t10", "dv10"), ("diverge/pos_t20", "dv20"), ("s_per_step", "s/step"),
    ]
    print(f"{'step':>5s} " + " ".join(f"{c[1]:>8s}" for c in cols))
    for r in R[-a.tail:]:
        print(f"{int(r.get('step', 0)):5d} " + " ".join(
            f"{r.get(c[0], float('nan')):8.4f}" if isinstance(r.get(c[0]), (int, float)) else f"{'-':>8s}"
            for c in cols))

    last = R[-min(len(R), max(5, len(R) // 5)):]
    def avg(k):
        v = [r[k] for r in last if isinstance(r.get(k), (int, float))]
        return sum(v) / len(v) if v else float("nan")

    tok_ratio = avg("trust/rmse") / TOKEN_RMS
    tgt_res2 = avg("action_target/tgt_resid_norm") ** 2
    loss_at = avg("loss/action_target")
    v_ev = avg("critic/v_ev")
    advH = [r["critic/v_adv_H_raw"] for r in last if isinstance(r.get("critic/v_adv_H_raw"), (int, float))]
    pos_frac = sum(1 for x in advH if x > 0) / len(advH) if advH else float("nan")

    print("\n=== 最近 {} 步均值 ===".format(len(last)))
    print(f"  G1 选择性   pred_resid_ratio={avg('action_target/pred_resid_ratio'):.3f} "
          f"(tgt={avg('action_target/tgt_resid_ratio'):.3f})   "
          f"loss_at={loss_at:.5f} vs tgt_res²={tgt_res2:.5f}  "
          f"→ {'✅' if loss_at < 0.7 * tgt_res2 else '❌'}")
    act = avg("act/rmse")
    print(f"  G2 动作健康 act/rmse={act:.4f}（目标区 0.05–0.15）"
          f"  → {'✅' if 0.05 <= act <= 0.15 else '❌'}")
    print(f"     （参考）token 位移 trust/rmse={avg('trust/rmse'):.4f} "
          f"/ token_rms≈{TOKEN_RMS} = {tok_ratio*100:.1f}%  ← v13 实测已 59%，不是瓶颈")
    print(f"  G3 critic   v_ev={v_ev:.4f}  → {'✅' if v_ev > 0.90 else '❌'}")
    print(f"  G4 自举语义  v_adv_H_raw>0 占比={pos_frac*100:.0f}%  "
          f"→ {'✅' if (pos_frac == pos_frac and pos_frac < 0.80) else '❌'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
