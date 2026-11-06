#!/usr/bin/env bash
# S3 采集**一臂**（locked 口径 2026-10-28）。
#
# 锁定项（不再由调用方决定）：
#   * plant      = ckpt 的 robot_spec.usd（HA dex3，43 DoF，手指锁 0）
#   * WBC        = runs/S1_handover/WBC_A24000_wmvel_cmd.pt
#   * 起点        = 库里 START_FRAME 那一行（root + 关节 + qd 全写），SM_INIT_FROM_LIB=1
#   * 手          = SM_HAND=1（2026-11-02 起默认打开：闭环一律用 VLA 的 14 维手部输出；
#                    旧 B 数据是手关时采的，回放它们要显式 SM_HAND=0）
#   * 重规划      = HA_VLA_EXEC_STEPS=20
#   * 录像        = 关（采集时纯开销）
#
# 用法：
#   ARM=gt  TASK=DoubleDesk SEED=0 START_FRAME=0    OUT=runs/B_r1/gt_DD_f0   bash scripts/s3_collect_arm.sh
#   ARM=vla TASK=DoubleDesk SEED=0 START_FRAME=2080 OUT=runs/B_r1/vla_DD_f2080 bash scripts/s3_collect_arm.sh
#
# 注意：ARM=vla 需要先起 server（scripts/start_vla_server.sh）；ARM=gt 走 online，不需要 server。
set -uo pipefail

COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
ARM=${ARM:-vla}
TASK=${TASK:-DoubleDesk}
SEED=${SEED:-0}
START_FRAME=${START_FRAME:-0}
# reset 扰动（SuperMimic 的 reset noise 同款量级）：xy 0.10m / yaw 0.25rad / q 0.05rad / qd 0.30
RESET_SEED=${RESET_SEED:-0}
RESET_NOISE=${RESET_NOISE:-1}
RESET_SCALE=${RESET_SCALE:-1.0}
MAX_STEPS=${MAX_STEPS:-300}
GPU=${GPU:-0}
SERVER_URL=${SERVER_URL:-http://127.0.0.1:18080}
# START_FRAME=random：按任务随机挑一条 clip + 一个帧（SuperMimic 的随机起点）
if [[ "$START_FRAME" == "random" ]]; then
  START_FRAME=$(python3 "$COMIMIC/scripts/sample_start_row.py" \
      --packed "$COMIMIC/runs/isaac_motion/ha_cmd/packed.pt" \
      --task "$TASK" --seed "$RESET_SEED" --margin $((MAX_STEPS + 80)))
  echo "[collect] random start -> row=$START_FRAME (task=$TASK seed=$RESET_SEED)"
fi
OUT=${OUT:-$COMIMIC/runs/B_r1/${ARM}_${TASK}_f${START_FRAME}_r${RESET_SEED}}

case "$ARM" in
  gt)  SRC=online ;;
  vla) SRC=http ;;
  *) echo "[collect] ARM 只能是 gt|vla，收到 $ARM" >&2; exit 2 ;;
esac
if [[ "$ARM" == "vla" ]] && ! (echo > /dev/tcp/127.0.0.1/${SERVER_URL##*:}) 2>/dev/null; then
  echo "[collect] VLA 臂需要 server：先 bash scripts/start_vla_server.sh" >&2; exit 3
fi
if [[ ! -f "$COMIMIC/runs/isaac_motion/ha_cmd/packed.pt" ]]; then
  echo "[collect] 缺 ha_cmd 库" >&2; exit 4
fi

export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$COMIMIC/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd}

echo "[collect] arm=$ARM task=$TASK seed=$SEED start_frame=$START_FRAME steps=$MAX_STEPS"
echo "[collect] src=$SRC server=$SERVER_URL gpu=$GPU out=$OUT"
echo "[collect] usd=$SUPERMIMIC_ROBOT_USD hand=0 init_from_lib=1 exec_steps=20 video=0"
echo "[collect] reset_noise=$RESET_NOISE seed=$RESET_SEED scale=$RESET_SCALE"
mkdir -p "$OUT/dump"

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  TASK="$TASK" SEED="$SEED" MAX_STEPS="$MAX_STEPS" ISAAC_GPU="$GPU" \
  MODEL_LABEL="s3_${ARM}" OUT_ROOT="$OUT/runs" \
  SM_CKPT="${SM_CKPT:-$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_cmd.pt}" \
  SM_TOKEN_SOURCE="$SRC" SERVER_URL="$SERVER_URL" \
  SM_TOKEN_START="$START_FRAME" SM_START_FRAME="$START_FRAME" SM_INIT_FROM_LIB=1 \
  SM_HAND=${SM_HAND:-1} RECORD_VIDEO=${RECORD_VIDEO:-0} HA_VLA_EXEC_STEPS=${EXEC_STEPS:-20} \
  SM_RESET_NOISE="$RESET_NOISE" SM_RESET_SEED="$RESET_SEED" SM_RESET_SCALE="$RESET_SCALE" \
  SM_DUMP_STATES="$OUT/dump/${TASK}_${ARM}" \
  bash "$COMIMIC/scripts/ha_isaac_closedloop.sh"
