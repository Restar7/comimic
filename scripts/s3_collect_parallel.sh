#!/usr/bin/env bash
# S3 并行采集：**4 个 Isaac 进程 × 六类任务**（默认 vla 臂），共用一个 VLA server。
#
# 用法：
#   # 默认：6 个任务 × 1 seed × start=0，4 并发，2 张卡轮转
#   bash scripts/s3_collect_parallel.sh
#
#   # 公平对照：GT 和 VLA 各跑一遍 DoubleDesk（同 seed / 同起点）
#   ARMS="gt vla" TASKS="DoubleDesk" bash scripts/s3_collect_parallel.sh
#
#   # 每个任务 3 条 episode（不同 reset 扰动；不用改 env seed）
#   EPISODES=3 bash scripts/s3_collect_parallel.sh
#
# 环境变量：
#   TASKS        任务列表（空格分隔；默认六类全跑）
#   SEED         环境 seed（**固定 0**；SuperMimic 的多样性来自 reset 扰动，不是 env seed）
#   EPISODES     每个任务跑几条 episode（每条用不同的 SM_RESET_SEED，默认 1）
#   START_FRAMES 起始库行号；默认 "random" = 按任务随机 clip + 随机帧（SuperMimic 同款）
#   ARMS         vla / gt / "gt vla"（默认 vla；gt 走 online、不需要 server）
#   JOBS         并发进程数（默认 4）
#   NUM_GPUS     轮转的 GPU 数（默认 2）
#   MAX_STEPS    每条步数（默认 300）
#
# 产物：runs/B_r1/<arm>_<task>_s<seed>_f<frame>/{dump,runs,scored.pt}
set -uo pipefail

COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
TASKS=${TASKS:-"DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi"}
SEED=${SEED:-0}
EPISODES=${EPISODES:-3}
START_FRAMES=${START_FRAMES:-"random"}
ARMS=${ARMS:-"vla"}
JOBS=${JOBS:-4}
NUM_GPUS=${NUM_GPUS:-2}
MAX_STEPS=${MAX_STEPS:-600}
ROOT=${ROOT:-$COMIMIC/runs/B_r1}
LOGD=${LOGD:-$COMIMIC/logs/B_r1}
SERVER_URL=${SERVER_URL:-http://127.0.0.1:18080}

mkdir -p "$ROOT" "$LOGD"

if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "[parallel] NUM_GPUS 必须 >= 1（收到 $NUM_GPUS）" >&2; exit 2
fi
if [[ "$JOBS" -gt "$NUM_GPUS" ]]; then
  echo "[parallel] WARN: JOBS=$JOBS > NUM_GPUS=$NUM_GPUS ⇒ 同一张卡会起多个 Isaac + server，容易 OOM。" \
       "建议 JOBS=NUM_GPUS，或把 server 放到另一张卡。" >&2
fi

if [[ "$ARMS" == *vla* ]]; then
  if ! (echo > /dev/tcp/127.0.0.1/${SERVER_URL##*:}) 2>/dev/null; then
    echo "[parallel] VLA 臂需要 server：先 bash scripts/start_vla_server.sh" >&2
    exit 3
  fi
fi

# ---- 组装任务清单 ------------------------------------------------------------
SPECS=()
for arm in $ARMS; do
  for task in $TASKS; do
    for f in $START_FRAMES; do
      for e in $(seq 0 $((EPISODES-1))); do
        SPECS+=("$arm|$task|$e|$f")
      done
    done
  done
done
echo "[parallel] 共 ${#SPECS[@]} 条，JOBS=$JOBS，GPU 轮转 0..$((NUM_GPUS-1))"
echo "[parallel] tasks: $TASKS"
echo "[parallel] arms: $ARMS   seed: $SEED   episodes/task: $EPISODES   start_frames: $START_FRAMES   steps: $MAX_STEPS"

# ---- 按批并发（每批 <= JOBS），批内各进程落自己的 log ----------------------
i=0; batch=0
run_batch() {
  local -a pids=() names=() outs=() logs=()
  local idx=$1; shift
  local -a items=("$@")
  for spec in "${items[@]}"; do
    IFS='|' read -r arm task ep f <<< "$spec"
    if [[ "$f" == "random" ]]; then
      f=$(python3 "$COMIMIC/scripts/sample_start_row.py" \
            --packed "$COMIMIC/runs/isaac_motion/ha_cmd/packed.pt" \
            --task "$task" --seed "$ep" --margin $((MAX_STEPS + 80)))
    fi
    out="$ROOT/${arm}_${task}_f${f}_r${ep}"
    log="$LOGD/${arm}_${task}_f${f}_r${ep}.log"
    echo "[parallel]   launch $arm/$task start=$f reset_seed=$ep gpu=$(( idx % NUM_GPUS ))"
    ARM="$arm" TASK="$task" SEED="$SEED" RESET_SEED="$ep" START_FRAME="$f" \
      GPU="$(( idx % NUM_GPUS ))" MAX_STEPS="$MAX_STEPS" \
      OUT="$out" bash "$COMIMIC/scripts/s3_collect_arm.sh" > "$log" 2>&1 &
    pids+=($!); names+=("$arm/$task/f$f/r$ep"); outs+=("$out"); logs+=("$log")
    idx=$((idx+1))
  done
  local k=0
  for pid in "${pids[@]}"; do
    wait "$pid"; rc=$?
    n=$(find "${outs[$k]}/dump" -name '*.pt' 2>/dev/null | wc -l)
    w=$(grep -c "Warp CUDA error" "${logs[$k]}" 2>/dev/null || true)
    echo "[parallel]   done ${names[$k]} rc=$rc dumps=$n warp_err=${w:-0}"
    k=$((k+1))
  done
}

total=${#SPECS[@]}
for ((start=0; start<total; start+=JOBS)); do
  batch=$((batch+1))
  chunk=("${SPECS[@]:start:JOBS}")
  echo "[parallel] ===== batch $batch (${#chunk[@]} 条) ====="
  run_batch "$start" "${chunk[@]}"
done

echo "[parallel] ===== 汇总 ====="
for arm in $ARMS; do
  for task in $TASKS; do
    for f in $START_FRAMES; do
      for e in $(seq 0 $((EPISODES-1))); do
        ff=$f
        if [[ "$ff" == "random" ]]; then
          ff=$(python3 "$COMIMIC/scripts/sample_start_row.py" \
                --packed "$COMIMIC/runs/isaac_motion/ha_cmd/packed.pt" \
                --task "$task" --seed "$e" --margin $((MAX_STEPS + 80)))
        fi
        out="$ROOT/${arm}_${task}_f${ff}_r${e}"
        n=$(find "$out/dump" -name '*.pt' 2>/dev/null | wc -l)
        printf '  %-28s dumps=%s\n' "${arm}_${task}_f${ff}_r${e}" "$n"
      done
    done
  done
done
