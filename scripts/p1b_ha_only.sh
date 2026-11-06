#!/usr/bin/env bash
# HA-only 训练（不用 bones_seed）：两个变体
#   VARIANT=ft      从 WBC_A24000_wmvel_v6 热启动，冻结 WM，4000 步（推荐：只换"步态统计"，不动先验）
#   VARIANT=scratch 完全从 0 训（用户原始想法；HA 只有 3.5 h，对照组用）
#
# 数据：runs/isaac_motion/ha_cmd_exec/packed_x20.pt = 699 clips ×20 = 13,980 clips / 12.68 M 帧
#       （HA-only；同一 clip 重复 20 次 = 给优化器更多步数，不引入外部数据）
# token：命令流库 ha_cmd（部署口径；名字去掉 __HA_CMDr 后缀后与 HA clip 一一对应）
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
R=$C/trackerlab-supermimic
cd "$R" || exit 1

VARIANT=${VARIANT:-ft}
TAG=${TAG:-haonly_$VARIANT}
LIB=${LIB:-$C/runs/isaac_motion/ha_cmd_exec/packed_x20.pt}
TOKLIB=${TOKLIB:-$C/runs/isaac_motion/ha_cmd/packed.pt}
ITER=${ITER:-28000}
NPROC=${NPROC:-2}
ENVS=${ENVS:-8192}
PORT=${PORT:-29940}
COMPILE=${COMPILE:-1}
LOG="$R/logs/exp_wbc_exec_$TAG"
mkdir -p "$C/logs/p1b"

case "$VARIANT" in
  ft)
    RESUME=${RESUME_CKPT:-$C/runs/S1_handover/WBC_A24000_wmvel_v6.pt}
    LR=${FORCE_LR:-2e-05}; FREEZE_WM=${FORCE_FREEZE_WM:-0}; NO_CRITIC=${FORCE_NO_CRITIC:-1}; WM_ONLY=${FORCE_WM_ONLY:-0}; CRITIC_WARMUP=${FORCE_CRITIC_WARMUP:-0}; CBS=${FORCE_CBS:-1.0} ;;
  scratch)
    RESUME=""; LR=2e-04; FREEZE_WM=0; NO_CRITIC=0; WM_ONLY=500; CRITIC_WARMUP=0; CBS=1.0 ;;
  *) echo "VARIANT must be ft|scratch"; exit 1 ;;
esac

[[ -f "$LIB" && -f "$TOKLIB" ]] || { echo "FATAL: library missing"; exit 1; }
[[ -e "$LOG" ]] && { echo "FATAL: $LOG exists"; exit 1; }

echo "[haonly] variant=$VARIANT iters->$ITER lib=$LIB"
echo "[haonly] token_lib=$TOKLIB resume=${RESUME:-<from scratch>} lr=$LR freeze_wm=$FREEZE_WM"
echo "[haonly] log=$LOG"

# 等 GPU 空下来（别和 G1 评测抢）
for _ in $(seq 1 240); do
  pgrep -f sim_eval_vla >/dev/null || break
  sleep 20
done
sleep 10

ROBOT_USD=$C/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd \
RESUME=$RESUME \
SM_TOKEN_LIB_TRAIN=$TOKLIB \
OBS_MODE=${OBS:-paper_obs93} FREEZE_WM=$FREEZE_WM NO_CRITIC=$NO_CRITIC WM_ONLY_ITERS=$WM_ONLY CRITIC_WARMUP=$CRITIC_WARMUP CRITIC_BOOTSTRAP_SCALE=$CBS LR=$LR \
MODE=train NPROC=$NPROC ENVS_PER_RANK=$ENVS MAX_EPISODE_STEPS=0 ITERATIONS=$ITER \
LIBRARY=$LIB INTERFACE=token COMPILE=$COMPILE HA_ALIGN=1 \
CKPT_EVERY=500 PORT_BASE=$PORT LOG_DIR=$LOG \
  bash scripts/rls/supermimic/train_bones_seed.sh > "$C/logs/p1b/exp_wbc_exec_$TAG.launcher.log" 2>&1
rc=$?
echo "[haonly] train exit=$rc"
CK=$LOG/ckpt_0${ITER}.pt
if [[ $rc -eq 0 && -f "$CK" ]]; then
  cp "$CK" "$C/runs/S1_handover/exp_WBC_exec_$TAG.pt"
  echo "[haonly] deliverable -> runs/S1_handover/exp_WBC_exec_$TAG.pt"
fi
