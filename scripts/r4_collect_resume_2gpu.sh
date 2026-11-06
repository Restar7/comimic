#!/usr/bin/env bash
# R4（round 904）剩余 fresh 采集，专为 2×4090 48G 环境。
# 语义与 run_full_round_auto.sh 的采集块**完全一致**，只把 --jobs/--num-gpus 从 8 降到 2。
# 断点续采：collect_full_episodes.py 只跳过 dump/ 里已有 *.pt 的目录（空壳会被重采）。
set -uo pipefail
COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1

ROUND=${ROUND:-904}
N=${N:-4}
SEED=${SEED:-4}
JOBS=${JOBS:-2}
GPUS=${GPUS:-2}
PREV_HEAD=${PREV_HEAD:-$COMIMIC/runs/S3_loop/903_joint/train_main/vla_final.pt}
WEIGHTS=${WEIGHTS:-$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt}
TARGET=${TARGET:-84}

LOGD=logs/auto_ladder
mkdir -p "$LOGD"
say(){ echo "[r4c $(date '+%H:%M:%S')] $*"; }

[[ -s "$PREV_HEAD" ]] || { say "[FATAL] missing PREV_HEAD=$PREV_HEAD"; exit 1; }
[[ -s "$WEIGHTS" ]]   || { say "[FATAL] missing WBC=$WEIGHTS"; exit 1; }
bash "$COMIMIC/scripts/check_isaac_driver.sh" || { say "[FATAL] Isaac 驱动不兼容（见上），拒绝采集"; exit 9; }

NCOL0=$(find "runs/B_$ROUND" -path '*/dump/*.pt' 2>/dev/null | wc -l)
say "start: have $NCOL0/$TARGET dumps; jobs=$JOBS gpus=$GPUS seed=$SEED head=$(basename "$(dirname "$(dirname "$PREV_HEAD")")")/$(basename "$PREV_HEAD")"
if (( NCOL0 >= TARGET )); then say "already complete; nothing to do"; exit 0; fi

PORT=$((18182 + N * 2))
SRVLOG=$LOGD/vla_server_${PORT}.log
# 清掉可能残留的旧 server（只匹配本端口）
pkill -f "[h]a_vla_latent64_server.py.*--port $PORT" 2>/dev/null

say "starting VLA server port=$PORT (log=$SRVLOG)"
PORT=$PORT GPU=0 CKPT="$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000" \
  HEAD_CKPT="$PREV_HEAD" LORA_RANK=32 LORA_TARGETS="attn1,ff" LORA_ALSO_TRAIN=proj_out \
  SRV_SEED=0 LOG="$SRVLOG" bash scripts/start_vla_server.sh > "$SRVLOG" 2>&1 &
srv=$!
ok=0
for i in $(seq 1 90); do
  if grep -q "listening on http" "$SRVLOG" 2>/dev/null; then ok=1; break; fi
  if ! kill -0 "$srv" 2>/dev/null; then break; fi
  sleep 4
done
if (( ok != 1 )); then
  say "[FATAL] VLA server did not start (see $SRVLOG)"
  kill "$srv" 2>/dev/null
  exit 1
fi
say "server ready; collecting remaining $(( TARGET - NCOL0 )) episodes"

python3 scripts/collect_full_episodes.py --round "$ROUND" --episodes 14 --seed "$SEED" \
  --server-url "http://127.0.0.1:$PORT" --jobs "$JOBS" --num-gpus "$GPUS" \
  --sm-ckpt "$WEIGHTS"
rc=$?

kill "$srv" 2>/dev/null
pkill -f "[h]a_vla_latent64_server.py.*--port $PORT" 2>/dev/null
wait "$srv" 2>/dev/null

NCOL=$(find "runs/B_$ROUND" -path '*/dump/*.pt' 2>/dev/null | wc -l)
say "collection rc=$rc; dumps=$NCOL/$TARGET"
(( NCOL >= TARGET )) || { say "[FATAL] still short of $TARGET"; exit 1; }
say "DONE"
