#!/usr/bin/env bash
# 新线 joint 微调（**直连** launch_joint_ha_v2.sh）
#
# 为什么不走 s3_pipeline.sh joint：
#   旧 pipeline 的 joint 阶段有 critic gate、双 arm(main/ctrl)、B 线 fresh_glob，
#   会覆盖新线的 REWARD_MODE/CRITIC_ON/ACT_WEIGHT，并把起点拉回 B。
#   新线必须单臂、参考起点、无 gate —— 所以直连 launcher。
#
# 用法：
#   ROUND=r1 RUN_LABEL=r1 STEPS=400 BATCH=8 ACCUM=16 \
#   SM_CKPT=$PWD/runs/S1_handover/exp_WBC_exec_haonly_best.pt \
#   bash scripts/run_refstart_joint.sh
#
# 冒烟：
#   ROUND=smoke RUN_LABEL=smoke STEPS=2 BATCH=2 ACCUM=1 \
#   EVAL_EVERY=0 SAVE_EVERY=0 ... bash scripts/run_refstart_joint.sh
set -uo pipefail
COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1
: "${ROUND:?ROUND required}" "${RUN_LABEL:?RUN_LABEL required}" "${STEPS:?STEPS required}"

SM_CKPT=${SM_CKPT:-$COMIMIC/runs/S1_handover/exp_WBC_exec_haonly_best.pt}
MOTION_PACKED=${MOTION_PACKED:-$COMIMIC/runs/isaac_motion/ha_cmd_exec/packed.pt}
VLM_CKPT=${VLM_CKPT:-$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}
VLM_PATH=${VLM_PATH:-$COMIMIC/groot_repro/nvidia/Cosmos-Reason2-2B-weights}
MOTION=${MOTION:-$COMIMIC/data/npz/ha_tokens}
INIT_VLA=${INIT_VLA:-}
WM_CKPT=${WM_CKPT:-}
BATCH=${BATCH:-8}
ACCUM=${ACCUM:-16}
H=${H:-20}
K=${K:-4}
BURNIN=${BURNIN:-10}
GAMMA=${GAMMA:-1.0}
REWARD_MODE=${REWARD_MODE:-huber_err}
ROOT_ORI_MODE=${ROOT_ORI_MODE:-split_huber}
REWARD_DELTA_POS=${REWARD_DELTA_POS:-0.05}
REWARD_DELTA_ORI=${REWARD_DELTA_ORI:-0.05}
REWARD_VEL_SCALE=${REWARD_VEL_SCALE:-0}
TAIL_WEIGHT=${TAIL_WEIGHT:-0}
POSTURE_WEIGHT=${POSTURE_WEIGHT:-0}
LAMBDA_R=${LAMBDA_R:-0.3}
CRITIC_ON=${CRITIC_ON:-1}
TRUST_CENTER=${TRUST_CENTER:-label}
TRUST_EPS=${TRUST_EPS:-0.2}
TRUST_WEIGHT=${TRUST_WEIGHT:-0}
# 2026-11-04：0.6 -> 0.30。依据：本文件之外的标定记录（commit 9e8c4cf）是
#   "base policy's own deviation 0.21, eps=0.30 (~1.4x), frac_out 0.15-0.30"；
#   0.6 是后来无依据地放松的，等于允许动作偏离到 A'(σ=0.05) 覆盖半径的 12 倍。
#   目标压到 mm 级时必须让它真的"咬住"，否则它既无约束力又掩盖真实违约。
# 2026-11-04（用户口径）：0.30 -> 0.10。起点永远从参考出发，20 tick 内偏差只有几 cm，
#   所以信任域应远小于 base policy 自身偏差（0.21，其中大半还是 FM 采样噪声）。
#   0.10 ≈ A'(σ=0.05) 的 2 倍；预期 frac_out≈1.0（它本来就该一直咬住）。
#   观察点：act/rms 若长期贴在 0.10（被钉住）= 奖励没空间，此时调 ACT_WEIGHT 而不是 eps。
ACT_EPS=${ACT_EPS:-0.10}
ACT_W=${ACT_W:-1.0}
START_SAMPLER=${START_SAMPLER:-phase}
START_WEIGHTS=${START_WEIGHTS:-$COMIMIC/runs/wm_real/exp_start_weights.pt}
START_AUDIT=${START_AUDIT:-1}
EVAL_EVERY=${EVAL_EVERY:-0}
SAVE_EVERY=${SAVE_EVERY:-0}
LOG_EVERY=${LOG_EVERY:-1}
GPU=${GPU:-0}
DDP=${DDP:-0}
NPROC=${NPROC:-1}
MASTER_PORT=${MASTER_PORT:-$((29800 + RANDOM % 150))}
OUT=${OUT:-$COMIMIC/runs/S3_loop/exp_ref_${RUN_LABEL}/train_main}

echo "[refstart] round=$ROUND label=$RUN_LABEL steps=$STEPS batch=$BATCH accum=$ACCUM"
echo "[refstart] wbc=$SM_CKPT"
echo "[refstart] packed=$MOTION_PACKED  start=reference sampler=$START_SAMPLER"
echo "[refstart] reward=$REWARD_MODE δ=$REWARD_DELTA_POS/$REWARD_DELTA_ORI ori=$ROOT_ORI_MODE gamma=$GAMMA"
echo "[refstart] critic=$CRITIC_ON trust=$TRUST_WEIGHT act=$ACT_W/$ACT_EPS out=$OUT"

export COMIMIC ROUND RUN="exp_ref_$RUN_LABEL" OUT
export WBC_CKPT="$SM_CKPT" VLM_CKPT VLM_PATH MOTION MOTION_PACKED
export STEPS BATCH ACCUM DDP NPROC MASTER_PORT H K BURNIN GPU
export V3=1 LAMBDA_R LAMBDA_BC=0 LAMBDA_S=0 LAMBDA_A=0 LAMBDA_HAND=0
export REWARD_MODE ROOT_ORI_MODE REWARD_DELTA_POS REWARD_DELTA_ORI REWARD_VEL_SCALE
export TAIL_WEIGHT TAIL_POS_THR="${TAIL_POS_THR:-0.564}" TAIL_POS_CAP="${TAIL_POS_CAP:-1.327}"
export POSTURE_WEIGHT GAMMA
# 2026-11-04：**默认仍是 imagined**（METHOD_LOCK_H20_20261101 定的口径：imagined GAE、
#   无 prefit、与策略一起在线更新 ⇒ 自洽、不依赖任何离线产物）。
#   这里只做两件事：①把硬编码改成可覆盖（`real` 作为可选臂）；
#   ②**启用 real 时会做口径检查**（原来那段检查是死代码，见 train_joint_v2 的注释）。
export CRITIC_ON CRITIC_SOURCE=${CRITIC_SOURCE:-imagined} CRITIC_GRAD_STATE=1 CRITIC_ON_MAIN=1 CRITIC_GRAD_STATE_MAIN=1
export CRITIC_STATE_GRAD_SCALE=0.1 CRITIC_REL_CLIP=5.0
export TRUST_CENTER TRUST_EPS TRUST_WEIGHT
export ACT_EPS ACT_WEIGHT="$ACT_W"
# 2026-11-04：**默认仍是 none**（plan 口径）。新增可选臂 `label`（配对基线，不需 B dump）。
#   注意：pathwise 梯度下减一个 detach 常数**不改变梯度** ⇒ 只有配 ADV_TAU>0 的
#   tanh 才生效；代价是每步多一次 no_grad 的 20 步 rollout。
export ADV_BASELINE=${ADV_BASELINE:-none} ADV_TAU=${ADV_TAU:-0.02} STATE_EXTRA=rotlocal6
export START_FROM_B=""
export START_SAMPLER START_WEIGHTS START_AUDIT
export ALLOW_LEGACY_REWARD=1 WM_CKPT INIT_VLA
export EVAL_EVERY SAVE_EVERY LOG_EVERY EVAL_BATCHES=8
export WEIGHT_DECAY=1e-4 OBS_NOISE=0 FM_LR=3e-5 LORA_ALPHA=64
export TOKEN_BC=0 TRAIN_WBC=0 WBC_UNFREEZE=all WBC_KL_ANCHOR=0 VLA_KL_ANCHOR=0
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD="${SUPERMIMIC_ROBOT_USD:-$COMIMIC/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd}"

bash scripts/launch_joint_ha_v2.sh
