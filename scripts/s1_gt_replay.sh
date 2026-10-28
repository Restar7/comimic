#!/usr/bin/env bash
# 在 HA 闭环里**原样重放 gt_2080 那个场景**（DoubleDesk / 2080 步 / GT token / seed 0），
# 用来回答"带手续训之后还会不会在 ~318 步摔"。
#
#   bash scripts/s1_gt_replay.sh                                   # 默认 ckpt_024000（带手训练产物）
#   SM_CKPT=runs/S1_handover/WBC_A20000_wmvel_cmd.pt bash scripts/s1_gt_replay.sh   # 对照：C6h
#
# 依赖（都已就位）：
#   /tmp/gt_tokens.npy (2080,64)  = data/npz/ha_tokens/HOI_double_desk__tw__ep000000.npz 的 g1_motion_token_64
#   /tmp/gt_hands.npy  (2080,14)  = 同 npz 的 g1_hand_joints_14
# 这条链路不需要 VLA 服务、不需要相机（npy 源）。
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

SM_CKPT=${SM_CKPT:-$C/trackerlab-supermimic/logs/wbc_hand_hold_x20c/ckpt_024000.pt}
LABEL=${LABEL:-gt_hand_24000}

TASK=DoubleDesk \
MAX_STEPS=${MAX_STEPS:-2080} \
SEED=${SEED:-0} REPEAT=${REPEAT:-0} \
SM_CKPT=$SM_CKPT \
SM_TOKEN_SOURCE=npy \
SM_TOKEN_NPY=${SM_TOKEN_NPY:-/tmp/gt_tokens.npy} \
SM_HANDS_NPY=${SM_HANDS_NPY:-/tmp/gt_hands.npy} \
MODEL_LABEL=$LABEL \
ISAAC_GPU=${ISAAC_GPU:-0} \
OUT_ROOT=${OUT_ROOT:-$C/runs/ha_isaac_$LABEL} \
  bash $C/scripts/ha_isaac_closedloop.sh
