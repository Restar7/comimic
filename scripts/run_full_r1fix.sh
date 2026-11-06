#!/usr/bin/env bash
# R1-fix: E_start=1.0, recalibrated hinge (B_901), joint-only, WM reused.
# Use existing B_901; do NOT recollect; do NOT refresh WM.
set -uo pipefail
COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1
B=${B:-$COMIMIC/runs/B_901}
[[ -d "$B" ]] || { echo "[fail] missing B dir $B"; exit 1; }
N=$(find "$B" -name '*.pt' -path '*/dump/*' | wc -l)
[[ "$N" == "180" ]] || { echo "[fail] B dumps=$N, expected 180"; exit 1; }
INIT=${INIT_VLA:-}
[[ -z "$INIT" || -f "$INIT" ]] || { echo "[fail] missing INIT_VLA=$INIT"; exit 1; }

export ROUND=${ROUND:-911}
export RUN_LABEL=${RUN_LABEL:-901fix_joint}
export RUN_TAG=${RUN_TAG:-901fix}
export TASKS=${TASKS:-"DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi"}
export B_GLOB=${B_GLOB:-"$B/*/dump/*.pt"}
export B_FRESH_GLOB=${B_FRESH_GLOB:-"$B/*/dump/*.pt"}
export B_FRESH_WEIGHT=${B_FRESH_WEIGHT:-3.0}
export B_START_MAX_ERR=${B_START_MAX_ERR-1.0}
export B_SELF_ERR_MAX=${B_SELF_ERR_MAX:-}
export INIT_VLA="${INIT:-}"
export WBC_CKPT="$COMIMIC/runs/S1_handover/wm_901.pt"
export SM_CKPT="$WBC_CKPT"
export A_ORDER=declared
export H=20
export STEPS=${STEPS:-200}
export BATCH=64
export ACCUM=1
export DDP=1
export NPROC=8
export TRAIN_ARMS=main
export MASTER_PORT=${MASTER_PORT:-29630}
export ACT_WEIGHT=${ACT_WEIGHT:-1.0}
export ACT_WEIGHT_SCHEDULE=${ACT_WEIGHT_SCHEDULE:-"${ROUND}:${ACT_WEIGHT}"}
export ACT_EPS=0.6
export HYBRID_ERR=${HYBRID_ERR:-0.5}
export LAMBDA_R=0.3
export TAIL_WEIGHT=${TAIL_WEIGHT:-1.0}
export TAIL_POS_THR=${TAIL_POS_THR:-0.401}
export TAIL_POS_CAP=${TAIL_POS_CAP:-0.971}
export TAIL_CAP_SLOPE=0.5
export YAW_HINGE_LAMBDA=0
export YAW_RATE_WEIGHT=0
export REWARD_ROOT_POS_STD=0.2
export REWARD_ROOT_ORI_STD=0.25
export REWARD_ROOT_WEIGHT=1.0
export ROOT_ORI_MODE=split
export REWARD_MODE=saturating
export REWARD_VEL_SCALE=0
export POSTURE_WEIGHT=0
export STATE_EXTRA=rotlocal6
export ADV_BASELINE=self
export ADV_TAU=0.02
export TRUST_CENTER=${TRUST_CENTER:-self}
export TRUST_EPS=0.2
export TRUST_WEIGHT=1.0
export LAMBDA_BC=0
export TOKEN_BC=0
export LAMBDA_HAND=0
export WEIGHT_DECAY=1e-4
export OBS_NOISE=0
export FM_LR=3e-5
export LORA_ALPHA=64
export LAMBDA_HAND=0
export TOKEN_BC_SOURCE=ref
export TOKEN_BC_SPAN=0
export CRITIC_ON=1
export CRITIC_SOURCE=imagined
export CRITIC_GRAD_STATE=1
export CRITIC_STATE_GRAD_SCALE=0.1
export CRITIC_ON_MAIN=1
export CRITIC_GRAD_STATE_MAIN=1
export TRAIN_WBC_MAIN=0
export WBC_UNFREEZE=all
export WBC_KL_ANCHOR=0
export VLA_KL_ANCHOR=0
export EVAL_EVERY=100
export EVAL_BATCHES=8
export SAVE_EVERY=50
export SAVE_VLA_EVERY=50
export LOG_EVERY=10
export GRAD_LOG_EVERY=100
export B_SELF_TARGETS="$COMIMIC/runs/wm_real/self_targets_all.pt"
export CRITIC_REAL="$COMIMIC/runs/wm_real/real_transitions_v2b.pt"

bash scripts/s3_loop.sh train
