#!/usr/bin/env bash
# 一次**两卡并行**闭环评测：GPU0 跑 HEAD_MAIN，GPU1 跑 HEAD_CTRL。
# head 传 `base` = 不挂 head（官方底模 π₀），用于"同一台机器同一口径"的对照。
#
# 用法：
#   TAG=pi0_vs_pi2 START_FRAME=12345 RESET_SEED=5 HEAD_MAIN=base \
#     HEAD_CTRL=$PWD/runs/S3_loop/r2/train_ctrl/vla_000150.pt bash scripts/s3_eval_wave.sh
#
# 产物：$OUT_ROOT/<TAG>/{main,ctrl}/dump/*.pt  + drift json/txt
set -uo pipefail

COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1
export COMIMIC

TAG=${TAG:-wave}
TASK=${TASK:-DoubleDesk}
START_FRAME=${START_FRAME:-0}
RESET_SEED=${RESET_SEED:-0}
MAX_STEPS=${MAX_STEPS:-300}
SM_CKPT=${SM_CKPT:-$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt}
HEAD_MAIN=${HEAD_MAIN:-base}
HEAD_CTRL=${HEAD_CTRL:-base}
OUT_ROOT=${OUT_ROOT:-$COMIMIC/runs/S3_loop/waves}
OUT=$OUT_ROOT/$TAG
PORT_MAIN=${PORT_MAIN:-18100}
PORT_CTRL=${PORT_CTRL:-18101}
LORA_RANK=32 LORA_TARGETS=${LORA_TARGETS:-attn1,ff} LORA_ALSO_TRAIN=${LORA_ALSO_TRAIN:-proj_out}

mkdir -p "$OUT" "$COMIMIC/logs"

say() { echo "[$(date +%H:%M:%S)] [wave $TAG] $*"; }

pids_on_port() {
  local port=$1 pid cl
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
    cl=$( { tr '\0' ' ' < "$d/cmdline"; } 2>/dev/null ) || continue
    case "$cl" in *ha_vla_latent64_server*"--port $port"*) echo "$pid" ;; esac
  done
}

kill_port() { local p; for p in $(pids_on_port "$1"); do kill -9 "$p" 2>/dev/null; done; }
# 收尾用：不管端口，清掉所有我们家的 server（2026-10-28：wave 结束时漏杀过，白白占 10 GB*2）
kill_all_servers() {
  local pid cl
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
    cl=$( { tr '\0' ' ' < "$d/cmdline"; } 2>/dev/null ) || continue
    case "$cl" in *ha_vla_latent64_server*) kill -9 "$pid" 2>/dev/null ;; esac
  done
}

start_srv() {  # port gpu head tag
  local port=$1 gpu=$2 head=$3 tag=$4
  kill_port "$port"
  local hc=""
  [[ "$head" != "base" ]] && hc="$head"
  say "server $tag port=$port gpu=$gpu head=$(basename "${hc:-base}")"
  PORT=$port GPU=$gpu HEAD_CKPT="$hc" LOG_DIR="$OUT/vla_in_$tag" \
    LORA_RANK=$LORA_RANK LORA_TARGETS=$LORA_TARGETS LORA_ALSO_TRAIN=$LORA_ALSO_TRAIN \
    bash "$COMIMIC/scripts/start_vla_server.sh" > "$COMIMIC/logs/wave_${TAG}_srv_$tag.log" 2>&1 &
  echo $!
}

wait_srv() {  # port head
  local port=$1 head=$2 t0=$SECONDS
  while (( SECONDS - t0 < 600 )); do
    if (echo > "/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
      say "server :$port 就绪（$((SECONDS-t0))s）"; return 0
    fi
    sleep 3
  done
  say "[FAIL] server :$port 没起来"; return 1
}

P0=$(start_srv "$PORT_MAIN" 0 "$HEAD_MAIN" main)
P1=$(start_srv "$PORT_CTRL" 1 "$HEAD_CTRL" ctrl)
wait_srv "$PORT_MAIN" "$HEAD_MAIN" || { kill "$P0" "$P1" 2>/dev/null; exit 1; }
wait_srv "$PORT_CTRL" "$HEAD_CTRL" || { kill "$P0" "$P1" 2>/dev/null; exit 1; }

say "eval task=$TASK row=$START_FRAME seed=$RESET_SEED steps=$MAX_STEPS"
ARM=vla TASK="$TASK" SEED=0 START_FRAME="$START_FRAME" RESET_SEED="$RESET_SEED" \
  MAX_STEPS="$MAX_STEPS" GPU=0 SERVER_URL="http://127.0.0.1:$PORT_MAIN" SM_CKPT="$SM_CKPT" \
  OUT="$OUT/main" bash "$COMIMIC/scripts/s3_collect_arm.sh" \
  > "$COMIMIC/logs/wave_${TAG}_main.log" 2>&1 &
A=$!
ARM=vla TASK="$TASK" SEED=0 START_FRAME="$START_FRAME" RESET_SEED="$RESET_SEED" \
  MAX_STEPS="$MAX_STEPS" GPU=1 SERVER_URL="http://127.0.0.1:$PORT_CTRL" SM_CKPT="$SM_CKPT" \
  OUT="$OUT/ctrl" bash "$COMIMIC/scripts/s3_collect_arm.sh" \
  > "$COMIMIC/logs/wave_${TAG}_ctrl.log" 2>&1 &
B=$!
wait "$A"; ra=$?
wait "$B"; rb=$?
kill "$P0" "$P1" 2>/dev/null; kill_port "$PORT_MAIN"; kill_port "$PORT_CTRL"; kill_all_servers
say "done rc main=$ra ctrl=$rb"

PYTHONPATH=. python3 "$COMIMIC/scripts/plot_drift.py" \
  $(compgen -G "$OUT/main/dump/*.pt" | head -1 | sed 's/^/--dump /') \
  $(compgen -G "$OUT/ctrl/dump/*.pt" | head -1 | sed 's/^/--dump /') \
  --out "$OUT/drift.png" --title "wave $TAG (row=$START_FRAME seed=$RESET_SEED)" 2>&1 | tee "$OUT/drift.txt"
