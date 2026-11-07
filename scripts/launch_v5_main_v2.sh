#!/usr/bin/env bash
# v5 主训练 v2（修复版，2×4090 各一个 seed）：
#   ★ 同观测动作 trust（规范原意，Pilot 证明这是决定性修复）
#   ★ 未 cap 范数 hinge（κ=2, λ_raw=1e-3）把 ‖u‖ 按在 ~2c
#   ★ 训练采样只取评测集的 **train clips**（防 val/test 泄漏）
#   ★ cap 固定 0.08（可学习性试验；不做 0.04→0.08 warmup）
#   ★ 每 25 步做条件化门禁（Δ 方差 + shuffle 敏感性）
set -euo pipefail
WT=${WT:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03}
# shellcheck disable=SC1091
source "$WT/scripts/harness03_env.sh"
OUT=${OUT:-$WT/runs/temporal_ac_bptt_v5/main_v2}
SEEDS=${SEEDS:-"0 1"}
UPDATES=${UPDATES:-1000}
BATCH=${BATCH:-256}
HORIZON=${HORIZON:-20}
VLA_BATCH=${VLA_BATCH:-128}
EVAL_SET=${EVAL_SET:-$WT/runs/temporal_ac_bptt_v5/eval_set}
mkdir -p "$OUT"
cd "$WT"
echo "[launch-v2] OUT=$OUT seeds='$SEEDS' updates=$UPDATES batch=$BATCH H=$HORIZON eval=$EVAL_SET"
echo "[launch-v2] git=$(git rev-parse --short HEAD) $(git rev-parse --abbrev-ref HEAD)"
pids=(); gpu=0
for s in $SEEDS; do
  log="$OUT/seed${s}.log"
  echo "[launch-v2] seed=$s gpu=$gpu -> $OUT/seed${s}"
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/train_temporal_ac_bptt_v5.py \
      --out "$OUT/seed${s}" --seed "$s" --device cuda:0 \
      --updates "$UPDATES" --batch "$BATCH" --horizon "$HORIZON" --beta 0 \
      --on-the-fly 1 --eval-set "$EVAL_SET" --vla-batch "$VLA_BATCH" \
      --trust-mode same_obs --lambda-raw 1e-3 --raw-norm-kappa 2 \
      --cap-warmup-updates 0 --cond-check-every 25 \
      > "$log" 2>&1 &
  pids+=($!)
  gpu=$((gpu + 1))
done
echo "[launch-v2] pids=${pids[*]}"
code=0
for p in "${pids[@]}"; do wait "$p" || code=$?; done
echo "[launch-v2] all done code=$code"
exit $code
