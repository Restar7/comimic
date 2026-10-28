#!/usr/bin/env bash
# 真物理在线回放（2026-10-28 口径锁定版）
#
# 与旧 `isaac_replay_2arm.sh` 的区别（**旧版的数全部作废**）：
#   1. token **不用预计算 npy**，改用 `SM_TOKEN_SOURCE=online`：每步用参考库（命令流）
#      现算 hold token，anchor = **机器人实时 root quat**。冻 anchor 会让机器人按错的世界
#      方向走（实测漂 3.49 m），这是旧三臂表作废的唯一原因。
#   2. 带 `SM_RESET_NOISE=1`：按 seed 扰动 reset 起点（xy 0.10 m / yaw 0.25 rad /
#      关节 0.05 rad / 关节速度 0.30）。没有它，不同 seed 逐位相同（max|Δa|=0.0），n 永远=1。
#   3. 落盘多了 `q_target_applied`（真正下发的 PD 目标，clip 后）、`tokens`（每步喂的 64 维）
#      和完整采集参数 meta（见 provider `_flush_state_dump`）。
#
# 用法：
#   TASK=DoubleDesk SEEDS="0 1 2" MAX_STEPS=300 STAMP=demo bash scripts/isaac_replay_online.sh
#   # 只跑冒烟：
#   TASK=DoubleDesk SEEDS="0" MAX_STEPS=60 STAMP=smoke bash scripts/isaac_replay_online.sh
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
TASKS=${TASKS:-DoubleDesk}
SEEDS=${SEEDS:-"0 1 2"}
MAX_STEPS=${MAX_STEPS:-300}
ISAAC_GPU=${ISAAC_GPU:-0}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$COMIMIC/runs/isaac_online_$STAMP}
SM_CKPT=${SM_CKPT:-$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_cmd.pt}
LIB=${LIB:-$COMIMIC/runs/isaac_motion/ha_cmd/packed.pt}
RESET_SCALE=${RESET_SCALE:-1.0}
TOKEN_START=${TOKEN_START:-0}
TOKEN_STRIDE=${TOKEN_STRIDE:-1}

mkdir -p "$OUT/dump"
echo "[online] tasks=[$TASKS] seeds=[$SEEDS] steps=$MAX_STEPS gpu=$ISAAC_GPU"
echo "[online] ckpt=$SM_CKPT"
echo "[online] lib=$LIB start=$TOKEN_START stride=$TOKEN_STRIDE reset_noise_scale=$RESET_SCALE"
echo "[online] out=$OUT"

clip_of() {
  case "$1" in
    DoubleDesk|doubledesk|HOI_double_desk) echo "HOI_double_desk__tw__ep000000" ;;
    PPBox|ppbox|HOI_pp_box)                echo "HOI_pp_box__yb__ep000000" ;;
    *) return 1 ;;
  esac
}

n_ok=0; n_bad=0
for task in $TASKS; do
  clip=$(clip_of "$task") || { echo "[online] SKIP 未知 task=$task"; continue; }
  hands="$COMIMIC/runs/arm_tokens/$clip/hands.npy"
  for seed in $SEEDS; do
    run="online_${task}_s${seed}"
    log="$OUT/$run.log"
    echo "[online] $(date +%T) $task seed=$seed"
    # 关键：清掉系统 python 的 torch 库路径，否则 Isaac 的 carb 子进程 fork 报 EAGAIN
    env -u LD_LIBRARY_PATH -u LD_PRELOAD \
      TASK=$task SEED=$seed MAX_STEPS=$MAX_STEPS ISAAC_GPU=$ISAAC_GPU \
      MODEL_LABEL=online OUT_ROOT=$OUT/runs SM_CKPT=$SM_CKPT \
      SM_TOKEN_SOURCE=online SM_TOKEN_LIB=$LIB \
      SM_TOKEN_START=$TOKEN_START SM_TOKEN_STRIDE=$TOKEN_STRIDE \
      SM_HANDS_NPY=$hands \
      SM_RESET_NOISE=1 SM_RESET_SEED=$seed SM_RESET_SCALE=$RESET_SCALE \
      SM_DUMP_STATES="$OUT/dump/${task}_online_s${seed}" \
      HA_VLA_EXEC_STEPS=1 \
      bash "$COMIMIC/scripts/ha_isaac_closedloop.sh" > "$log" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]]; then n_ok=$((n_ok+1)); else n_bad=$((n_bad+1)); echo "[online] rc=$rc 见 $log"; fi
    echo "[online] $(date +%T) done rc=$rc"
  done
done
echo "[online] 完成 ok=$n_ok bad=$n_bad"
