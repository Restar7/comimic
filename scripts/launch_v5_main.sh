#!/usr/bin/env bash
# v5 主训练启动器（规范 §0-A / §8 M2）：**2×4090 各跑一个 seed**。
#
# 为什么不用 DDP：瓶颈是 CPU 侧 op 派发（规范 §0）⇒ DDP 加速比只有 1.3~1.6×，
# 两个独立进程跑两个 seed 更快、更简单，且各自 `--out` 独立、`updates` 计数不会错。
#
# 用法::
#   nohup setsid bash scripts/launch_v5_main.sh > runs/temporal_ac_bptt_v5/main/launch.log 2>&1 &
set -euo pipefail
WT=${WT:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03}
# shellcheck disable=SC1091
source "$WT/scripts/harness03_env.sh"
# ★ 输出放 **worktree** 的 runs/（gitignored），不往主仓库写任何东西
OUT=${OUT:-$WT/runs/temporal_ac_bptt_v5}
SEEDS=${SEEDS:-"0 1"}
UPDATES=${UPDATES:-1000}
BATCH=${BATCH:-256}
HORIZON=${HORIZON:-20}
VLA_BATCH=${VLA_BATCH:-128}
mkdir -p "$OUT/main"
cd "$WT"
echo "[launch] WT=$WT OUT=$OUT seeds='$SEEDS' updates=$UPDATES batch=$BATCH H=$HORIZON"
echo "[launch] git=$(git rev-parse --short HEAD) $(git rev-parse --abbrev-ref HEAD)"
pids=()
gpu=0
for s in $SEEDS; do
  log="$OUT/main/seed${s}.log"
  echo "[launch] seed=$s gpu=$gpu -> $OUT/main/seed${s}"
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/train_temporal_ac_bptt_v5.py \
      --out "$OUT/main/seed${s}" --seed "$s" --device cuda:0 \
      --updates "$UPDATES" --batch "$BATCH" --horizon "$HORIZON" --beta 0 \
      --on-the-fly 1 --eval-set "$OUT/eval_set" --vla-batch "$VLA_BATCH" \
      > "$log" 2>&1 &
  pids+=($!)
  gpu=$((gpu + 1))
done
echo "[launch] pids=${pids[*]}"
code=0
for p in "${pids[@]}"; do
  wait "$p" || code=$?
done
echo "[launch] all done code=$code"
exit $code
