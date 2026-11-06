#!/usr/bin/env bash
# 用一个给定的策略（VLA ckpt + joint ckpt）采 80 集闭环，用来做 80 集判据。
#
# 为什么需要独立脚本：`run_j1_rounds.sh` 的 2/9 是绑在"轮次"上的，而奖励实验常常
# 需要**同一个策略换奖励再采一遍**（或给某个实验臂补采 80 集）。这里只做一件事：
#   指定起点 seed → 4 任务 × 20 集 × 600 tick → 输出 dump + 一个健康报告。
#
# 用法：
#   bash scripts/collect_bline_arm.sh <out_dir> <joint_ckpt> <vla_ckpt> [seed] [jobs]
set -euo pipefail
OUT=${1:?用法: collect_bline_arm.sh <out_dir> <joint_ckpt> <vla_ckpt> [seed] [jobs]}
JOINT=${2:?缺 joint_ckpt}
VLA=${3:?缺 vla_ckpt}
SEED=${4:-2}
JOBS=${5:-4}
TICKS=${BLINE_TICKS:-600}
EPISODES=${BLINE_EPISODES:-20}
TASKS=${BLINE_TASKS:-"task1 task2 task3 task4"}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LIB=$REPO/data/library/ft318_compact.pt
cd "$REPO"
# PYTHONPATH 必须和 run_j1_rounds.sh 完全一致（gr00t_deps 提供 tyro；
# 少了它 `closed_loop_vla_supermimic.py` 会 ModuleNotFoundError: tyro）
export PYTHONPATH=$REPO:$REPO/Isaac-GR00T:$REPO/third_party/gr00t_deps:$REPO/third_party/gr00t_deps/_noflash
mkdir -p "$OUT"
_REF=$(grep -m1 "^export PYTHONPATH" "$REPO/scripts/run_j1_rounds.sh" | sed "s/^export PYTHONPATH=//; s#\$COMIMIC#$REPO#g")
if [[ -n "$_REF" && "$_REF" != "$PYTHONPATH" ]]; then
  echo "[arm][warn] PYTHONPATH 与 runner 不一致 ⇒ 采用 runner 的"; export PYTHONPATH="$_REF"
fi
echo "[arm] out=$OUT seed=$SEED tasks='$TASKS' x $EPISODES ep x $TICKS tick jobs=$JOBS"
_n=0
for T in $TASKS; do
  START_ARGS=$(PYTHONPATH=$REPO python3 scripts/bline_starts.py --library "$LIB" --task "$T" \
    --episodes "$EPISODES" --ticks "$TICKS" --seed "$SEED")
  STARTS=($START_ARGS)   # 不能 read -a：它只读第一行
  NS=${#STARTS[@]}
  [[ $NS -gt 0 ]] || { echo "[fail] $T 没有起点"; exit 3; }
  echo "[arm] $T 起点($NS): ${STARTS[0]} … ${STARTS[$((NS-1))]}"
  for E in $(seq 1 "$EPISODES"); do
    F=${STARTS[$(( (E - 1) % NS ))]}
    D="$OUT/b_${T}_f${F}_e${E}.pt"
    if [[ -s "$D" ]]; then echo "[arm] $T e$E 已存在 ⇒ 跳过"; continue; fi
    G=$(( _n % JOBS ))
    python3 scripts/closed_loop_vla_supermimic.py \
      --vla-ckpt "$VLA" --joint-ckpt "$JOINT" --task "$T" \
      --steps "$TICKS" --device "cuda:$G" --no-video \
      --replay-lib "$LIB" --start-frame "$F" --arm vla --dump "$D" \
      > "$OUT/log_${T}_e${E}.log" 2>&1 &
    _n=$((_n+1))
    if (( _n % JOBS == 0 )); then wait || true; fi
  done
done
wait || true
echo "[arm] $(ls "$OUT"/b_*.pt 2>/dev/null | wc -l) 条 → $OUT"
python3 scripts/bline_health.py --dump "$OUT/b_*.pt" --out "$OUT/health.json" || true
