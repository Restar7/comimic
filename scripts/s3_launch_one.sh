#!/usr/bin/env bash
# S3 单臂启动器：口径审计（fail-closed）→ 带完整口径的一个训练臂。
#
# 刻意把每个 knob 做成环境变量，这样"两臂只差一个变量"可以逐字对照：
#   GPU / RUN / LAMBDA_R / DR_XY / DR_YAW / TRUST_EPS / TRUST_WEIGHT / STEPS / H
#
# 例：
#   GPU=0 RUN=s3h20_dr05_t LAMBDA_R=0.1 DR_XY=0.05 DR_YAW=0.06 bash scripts/s3_launch_one.sh
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC" || exit 1

GPU=${GPU:-0}
RUN=${RUN:-s3h20_arm}
STEPS=${STEPS:-300}
H=${H:-20}
LAMBDA_R=${LAMBDA_R:-0.1}
LAMBDA_HAND=${LAMBDA_HAND:-0.2}
ANCHOR=${ANCHOR:-0.05}
WBC=${WBC:-$COMIMIC/runs/S1_handover/WBC_A20000_wmvel_cmd.pt}
VLM=${VLM:-$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}
# --- DR（起点扰动；之前 DR_XY=0.20 是拍脑袋定的，这里按阶梯测）---
DR_XY=${DR_XY:-0.20}
DR_YAW=${DR_YAW:-0.25}
DR_DQ=${DR_DQ:-0.05}
DR_DQD=${DR_DQD:-0.30}
DR_P_NOM=${DR_P_NOM:-0.25}
DR_P_MID=${DR_P_MID:-0.45}
PUSH_STD=${PUSH_STD:-0.03}
PUSH_ANG=${PUSH_ANG:-0.05}
# --- 约束：动作空间 trust region（中心 = 标签 token 的 rollout）---
ACT_EPS=${ACT_EPS:-0.30}
ACT_WEIGHT=${ACT_WEIGHT:-1.0}
# token 空间旧约束（已被动作空间取代，默认关）
TRUST_EPS=${TRUST_EPS:-0}
TRUST_WEIGHT=${TRUST_WEIGHT:-0}
# --- 奖励 / critic 口径（v4）---
REWARD_MODE=${REWARD_MODE:-saturating}
REWARD_VEL_SCALE=${REWARD_VEL_SCALE:-0}      # 0 = 扣掉 WM 不可信的速度两项
CRITIC_SOURCE=${CRITIC_SOURCE:-real}         # real = 真录制轨迹的 return-to-go（grounded）
CRITIC_REAL=${CRITIC_REAL:-$COMIMIC/runs/wm_real/real_returns_vs0.pt}
# --- 动作头 LoRA + 混合精度（对齐 SuperMimic 的"大 batch"前提）---
LORA_RANK=${LORA_RANK:-32}                   # 0 = 训全量 1.62B（batch 只能 4）
AMP=${AMP:-bf16}                             # GR00T ckpt 原生 dtype=bfloat16
# --- 规模 ---
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-8}
EVAL_EVERY=${EVAL_EVERY:-100}
EVAL_BATCHES=${EVAL_BATCHES:-16}          # ×batch = 窗口数（16×4 = 64）
EVAL_HORIZON=${EVAL_HORIZON:-20}          # 部署口径；与训练视界解耦
SAVE_EVERY=${SAVE_EVERY:-150}
WM_REFRESH_EVERY=${WM_REFRESH_EVERY:-50}

mkdir -p logs runs
echo "[one] $(date +%T) gpu=$GPU run=$RUN lam_r=$LAMBDA_R dr=($DR_XY,$DR_YAW) trust=($TRUST_EPS,$TRUST_WEIGHT) steps=$STEPS"

_HDEV=""
[[ "$H" != "20" ]] && _HDEV="--allow-horizon-deviation"
python3 scripts/s3_audit_config.py --horizon "$H" --lambda-r "$LAMBDA_R" $_HDEV || {
  echo "[one] ❌ 口径审计未通过，拒绝启动 $RUN"; exit 1; }

WBC_CKPT="$WBC" VLM_CKPT="$VLM" RUN="$RUN" GPU="$GPU" \
STEPS="$STEPS" BATCH="$BATCH" ACCUM="$ACCUM" H="$H" K=4 \
LAMBDA_R="$LAMBDA_R" LAMBDA_BC=1.0 LAMBDA_A=0.0 LAMBDA_S=0.0 \
BURNIN=10 EVAL_EVERY="$EVAL_EVERY" EVAL_BATCHES="$EVAL_BATCHES" \
SAVE_EVERY="$SAVE_EVERY" LOG_EVERY=20 \
EXTRA="--v3 --lambda-hand $LAMBDA_HAND --critic \
       --start-source recorded --wm-refresh-every $WM_REFRESH_EVERY \
       --anchor-weight $ANCHOR --push-std $PUSH_STD --push-ang $PUSH_ANG \
       --dr-xy-max $DR_XY --dr-yaw-max $DR_YAW --dr-dq-max $DR_DQ --dr-dqd-max $DR_DQD \
       --dr-p-nominal $DR_P_NOM --dr-p-mid $DR_P_MID \
       --trust-eps $TRUST_EPS --trust-weight $TRUST_WEIGHT \
       --act-eps $ACT_EPS --act-weight $ACT_WEIGHT \
       --reward-mode $REWARD_MODE --reward-vel-scale $REWARD_VEL_SCALE \
       --critic-source $CRITIC_SOURCE --critic-real $CRITIC_REAL \
       --grad-log-every 20 --clip-head 10 \
       --eval-horizon $EVAL_HORIZON \
       --lora-rank $LORA_RANK --amp $AMP" \
  bash scripts/launch_joint_ha_v2.sh
