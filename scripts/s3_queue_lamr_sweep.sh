#!/usr/bin/env bash
# S3 λ_r 扫描队列：每张卡顺序跑两格，两卡并行。
#   GPU0: lambda_r = 0（BC 对照） → 0.1
#   GPU1: lambda_r = 0.3        → 1.0
# 执行器固定 = C6h（WBC_A20000_wmvel_cmd：C2 策略 + 按 ha_cmd 域重标的 WM）
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
WBC=${WBC:-$COMIMIC/runs/S1_handover/WBC_A20000_wmvel_cmd.pt}
VLM=${VLM:-$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}
STEPS=${STEPS:-1000}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-4}
H=${H:-16}
K=${K:-4}
EVAL_EVERY=${EVAL_EVERY:-200}
EVAL_BATCHES=${EVAL_BATCHES:-16}
SAVE_EVERY=${SAVE_EVERY:-500}
TAG=${TAG:-s3c6h}

[[ -f "$WBC" ]] || { echo "[queue] missing WBC $WBC"; exit 1; }
[[ -d "$VLM" ]] || { echo "[queue] missing VLM $VLM"; exit 1; }
mkdir -p "$COMIMIC/logs"

run_one() {   # $1 = gpu, $2 = lambda_r
  local gpu=$1 lam=$2
  local run="${TAG}_lam${lam}"
  echo "[queue] $(date +%T) gpu=$gpu lambda_r=$lam run=$run"
  WBC_CKPT="$WBC" VLM_CKPT="$VLM" RUN="$run" GPU="$gpu" STEPS="$STEPS" \
  BATCH="$BATCH" ACCUM="$ACCUM" H="$H" K="$K" LAMBDA_R="$lam" \
  EVAL_EVERY="$EVAL_EVERY" EVAL_BATCHES="$EVAL_BATCHES" SAVE_EVERY="$SAVE_EVERY" LOG_EVERY=20 \
    bash "$COMIMIC/scripts/launch_joint_ha_v2.sh" > "$COMIMIC/logs/${run}.launcher.log" 2>&1
  echo "[queue] $(date +%T) done gpu=$gpu lambda_r=$lam rc=$?"
}

run_one 0 0 &
P0=$!
run_one 1 0.3 &
P1=$!
wait $P0 $P1

run_one 0 0.1 &
P2=$!
run_one 1 1.0 &
P3=$!
wait $P2 $P3

echo "[queue] $(date +%T) sweep done"
