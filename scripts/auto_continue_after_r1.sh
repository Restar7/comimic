#!/usr/bin/env bash
# 等 R1 的 9/9 评测完成 → 自动评估（六维漂移 + mean_reward + 训练后 WM 门）
# → 达标则自动启动 R2..R11（10 轮，累积：VLA 与 WBC/WM 都用上一轮的产物）
set -uo pipefail
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

OUT=runs/j1_scene
RD=$OUT/round_1
LOG=$(ls -t logs/j1_rounds_2026*.log | head -1)
echo "[watch] 等 R1 评测完成 … log=$LOG"
for i in $(seq 1 240); do   # 最多等 4 h
  if grep -q "\[eval\] round 1" "$LOG" 2>/dev/null; then break; fi
  sleep 30
done
if ! grep -q "\[eval\] round 1" "$LOG" 2>/dev/null; then
  echo "[watch][fail] 4 h 内没等到 R1 评测"; exit 1
fi
echo "[watch] R1 评测完成，开始评估"

python3 scripts/eval_realized_drift.py --dump "$RD/eval_bline.pt" \
  --out "$RD/drift_eval_r1.json" > "$RD/drift_eval_r1.txt" 2>&1
python3 - "$RD" <<'PY' | tee "$RD/eval_verdict.txt"
import json, sys
from pathlib import Path
rd = Path(sys.argv[1])
drift = json.loads((rd / "drift_eval_r1.json").read_text())
mean_reward = None
try:
    import re
    line = [l for l in open(rd / "eval.json") if "mean_reward" in l or "reward" in l]
except Exception:
    pass
# rounds.csv 最后一行 = 本轮 mean_reward
try:
    row = open(rd.parent / "rounds.csv").read().strip().splitlines()[-1].split(",")
    mean_reward = float(row[1])
except Exception:
    pass
after = None
p = rd / "gate_wm_bline_after.json"
if p.exists():
    d = json.loads(p.read_text()); after = (d["qd_corr"], d["qd_ratio"], d["pass"])

cond = []
rp, ro, jr = drift["root_pos_m"]["p50"], drift["root_ori_deg"]["p50"], drift["joint_rad"]["p50"]
cond.append(("root_pos p50 < 10 cm", rp < 0.10, f"{rp*100:.2f} cm"))
cond.append(("root_ori p50 < 15°", ro < 15.0, f"{ro:.2f}°"))
cond.append(("joint_rad p50 < 0.35 rad", jr < 0.35, f"{jr:.3f} rad"))
if mean_reward is not None:
    cond.append(("mean_reward > 0.06", mean_reward > 0.06, f"{mean_reward:.4f}"))
if after is not None:
    cond.append(("WM after: corr>=0.95 & ratio<=1.15", after[0] >= 0.95 and after[1] <= 1.15,
                 f"corr {after[0]:.4f} ratio {after[1]:.4f}"))
ok = all(c[1] for c in cond)
print("== R1 verdict ==")
for name, passed, val in cond:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name:42s} 实值 {val}")
print(f"== overall: {'PASS ⇒ 自动启动 R2..R11（10 轮）' if ok else 'FAIL ⇒ 不自动续跑，等人看'} ==")
sys.exit(0 if ok else 3)
PY
rc=$?

if [[ $rc -ne 0 ]]; then
  echo "[watch] 判定未通过 ⇒ 不自动续跑（结果见 $RD/eval_verdict.txt）"; exit 0
fi

echo "[watch] 通过 ⇒ 启动 R2..R11（10 轮，累积）"
setsid nohup env OUT=$PWD/runs/j1_scene START_ROUND=2 ROUNDS=10 \
  NGPU=4 GPUS=0,1,2,3 J1_BATCH=16 J1_LORA=64 J1_H=8 \
  SM_SCENE_TASKS="task1 task2 task3 task4" SCENE_ENVS=256 SCENE_STEPS=1000 \
  BLINE_JOBS=4 BLINE_EPISODES=20 BLINE_TICKS=600 \
  WM_REFRESH_EVERY=0 WM_PRETRAIN_STEPS=8000 WM_BLINE_RATIO=0.45 \
  CRITIC_MODE=on CRITIC_SOURCE=imagined TRAIN_WBC=0 \
  VLA_CKPT=$PWD/runs/j1_scene/ckpt_vla_r1 \
  CUR_WBC=$PWD/runs/j1_scene/round_1/train/joint_final.pt \
  bash scripts/launch_j1_rounds.sh > logs/j1_r2_r11_launch.log 2>&1 < /dev/null &
disown
echo "[watch] 已后台启动，日志 logs/j1_r2_r11_launch.log"
