#!/usr/bin/env bash
# S3 A0/A1 对照臂（两卡并行，唯一差异 = lambda_r）。
#
#   A0: lambda_r = 0     纯 hand-BC + 先验锚定（奖励通道关掉）
#   A1: lambda_r = 0.1   + 奖励项（状态/速度/存活，WBC+WM 想象里反传）
#
# 两臂逐字相同的口径（计划文档 §12 / 审计脚本 33 项）：
#   H=20（部署重规划周期）· chunk 40 · K=4 · codec.decode · 标签=hold@ha_cmd
#   WBC=C6h（两侧同一个 md5）· burn-in 10 · DR 0.25/0.45/0.30 · critic γ^20 自举
#   起点=真录制状态 · WM 每 50 步真转移保鲜 · 评测用固定 WM 快照
#
# 用法：
#   bash scripts/s3_launch_A0A1.sh            # 起两臂（500 步）
#   STEPS=20 bash scripts/s3_launch_A0A1.sh   # 冒烟
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC" || exit 1

H=${H:-20}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-8}
STEPS=${STEPS:-500}
EVAL_EVERY=${EVAL_EVERY:-100}
EVAL_BATCHES=${EVAL_BATCHES:-4}
SAVE_EVERY=${SAVE_EVERY:-250}
LAMBDA_HAND=${LAMBDA_HAND:-0.2}
ANCHOR=${ANCHOR:-0.05}
PUSH_STD=${PUSH_STD:-0.03}
PUSH_ANG=${PUSH_ANG:-0.05}
WM_REFRESH_EVERY=${WM_REFRESH_EVERY:-50}
TAG=${TAG:-A0A1_h20}

WBC=${WBC:-$COMIMIC/runs/S1_handover/WBC_A20000_wmvel_cmd.pt}
VLM=${VLM:-$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}

# ---------------------------------------------------------------- 0) 口径闸门
echo "[A0A1] $(date +%T) 口径审计（fail-closed）"
if ! python3 scripts/s3_audit_config.py --horizon "$H" --lambda-r 0.1; then
  echo "[A0A1] ❌ 审计未通过，拒绝启动"
  exit 1
fi

mkdir -p logs
declare -a PIDS=()

launch_arm() {   # $1=gpu  $2=lambda_r  $3=run name
  local gpu=$1 lam=$2 run=$3
  echo "[A0A1] $(date +%T) gpu=$gpu lambda_r=$lam run=$run"
  WBC_CKPT="$WBC" VLM_CKPT="$VLM" RUN="$run" GPU="$gpu" \
  STEPS="$STEPS" BATCH="$BATCH" ACCUM="$ACCUM" H="$H" K=4 \
  LAMBDA_R="$lam" LAMBDA_BC=1.0 LAMBDA_A=0.0 LAMBDA_S=0.0 \
  BURNIN=10 EVAL_EVERY="$EVAL_EVERY" EVAL_BATCHES="$EVAL_BATCHES" \
  SAVE_EVERY="$SAVE_EVERY" LOG_EVERY=20 \
  EXTRA="--v3 --lambda-hand $LAMBDA_HAND --critic \
         --start-source recorded --wm-refresh-every $WM_REFRESH_EVERY \
         --anchor-weight $ANCHOR --push-std $PUSH_STD --push-ang $PUSH_ANG \
         --grad-log-every 20 --clip-head 10" \
    bash scripts/launch_joint_ha_v2.sh
}

launch_arm 0 0.0 "${TAG}_A0" > "logs/${TAG}_A0.launcher.log" 2>&1 &
PIDS+=($!)
launch_arm 1 0.1 "${TAG}_A1" > "logs/${TAG}_A1.launcher.log" 2>&1 &
PIDS+=($!)

echo "[A0A1] 已起 ${#PIDS[@]} 臂：${PIDS[*]}"
echo "[A0A1] 日志: logs/${TAG}_A0.log   logs/${TAG}_A1.log"
echo "[A0A1] 看进度: tail -f logs/${TAG}_A1.log | grep -E 'step |eval|AUDIT'"
wait
echo "[A0A1] $(date +%T) 两臂结束"
