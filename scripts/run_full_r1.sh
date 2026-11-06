#!/usr/bin/env bash
# R1 of the full-length 5-round plan: base VLA, 180 full clips, act 1.0, max_err 0.5.
set -uo pipefail
COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1
ROUND=${ROUND:-901}
B=${B:-$COMIMIC/runs/B_$ROUND}
[[ -d "$B" ]] || { echo "[fail] missing B dir $B; collect first"; exit 1; }
N=$(find "$B" -name '*.pt' -path '*/dump/*' | wc -l)
[[ "$N" == "180" ]] || { echo "[fail] B dumps=$N, expected 180"; exit 1; }
TASKS=${TASKS:-"DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi"}
export TASKS ROUND
export COLLECT_EPISODES=30
export B_GLOB="$B/*/dump/*.pt"
export B_FRESH_GLOB="$B/*/dump/*.pt"
export B_FRESH_PIN=1
export B_FRESH_WEIGHT=3.0
export INIT_VLA=""
export SM_CKPT="$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt"
export A_ORDER=declared
export WM_AB_RATIO=1
export H=20
export STEPS=200
export BATCH=64
export ACCUM=1
export DDP=1
export NPROC=8
export ACT_WEIGHT=1.0
export ACT_WEIGHT_SCHEDULE="901:1.0"
export ACT_EPS=0.6
export LAMBDA_R=0.3
export B_START_MAX_ERR=0.5
export TAIL_WEIGHT=1.5
export TAIL_CAP_SLOPE=0.5
export YAW_HINGE_LAMBDA=0
export YAW_RATE_WEIGHT_OVERRIDE=0
export REWARD_ROOT_POS_STD=0.2
export REWARD_ROOT_ORI_STD=0.25
export REWARD_ROOT_WEIGHT=1.0
export ROOT_ORI_MODE=split
export TRAIN_ARMS=main
export CRITIC_ON_MAIN=1
export CRITIC_GRAD_STATE_MAIN=1
export CRITIC_STATE_GRAD_SCALE=0.1
export EVAL_STEPS=2080
export RUN_TAG=full_r1
export RUN_LABEL=full_r1
export MASTER_PORT=29620
bash scripts/s3_pipeline.sh all
