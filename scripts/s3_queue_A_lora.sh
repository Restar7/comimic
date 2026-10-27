#!/usr/bin/env bash
# 正式 A 方案（联合微调）：LoRA r=32 + bf16 + 有效 batch 64 + H=20（= 部署口径）
# 与之前唯一的实质差别：可训参数 1.62B → 33.5M(2.0%)，于是 batch 4 → 32（×accum 2 = 64）。
set -uo pipefail
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC" || exit 1
STEPS=${STEPS:-300}
echo "[A] $(date +%T) 起正式 A 方案（LoRA r=32, bf16, batch 32 x accum 2, H=20）"
GPU=0 RUN=s3A_lora32 STEPS="$STEPS" H=20 BATCH=32 ACCUM=2 LAMBDA_R=0.1 LAMBDA_HAND=0 \
ANCHOR=0 ACT_EPS=0.60 ACT_WEIGHT=1.0 REWARD_VEL_SCALE=1.0 CRITIC_SOURCE=real \
CRITIC_REAL="$COMIMIC/runs/wm_real/real_returns.pt" WM_REFRESH_EVERY=0 DR_XY=0.10 DR_YAW=0.125 \
SAVE_EVERY=100 EVAL_EVERY=50 EVAL_BATCHES=16 EVAL_HORIZON=20 LORA_RANK=32 AMP=bf16 \
  bash scripts/s3_launch_one.sh > logs/s3A_lora32.launcher.log 2>&1
echo "[A] $(date +%T) 结束 rc=$?"
