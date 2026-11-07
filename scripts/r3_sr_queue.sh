#!/usr/bin/env bash
# v13 vs base 成功率实验（VisionNavi / SitSofa），无视频，**串行排队 + 每卡只放一个 Isaac**。
#
# 关键（之前踩过的坑，必须保留）：
#   1) HA_ALLOW_SONIC / HA_GMT_BACKEND / HA_ACTION_SOURCE 只有 r3_closedloop_ab.sh 会导出，
#      ha_isaac_closedloop.sh **不会**。少了这三个 → 落到 [SuperMimic] 执行器，
#      v13 看起来"变差了"（其实是换了控制器）。这里显式导出。
#   2) 一张卡塞 2 个 Isaac 会 OOM（1 个 Isaac ≈ 20 GB + server 6 GB，2 个直接冲 48 GB）。
#      ⇒ 每卡 1 个 Isaac，用队列让两卡都不空转。
#   3) --episode_batch_json 第 2 集起会 sim_error（int(None)）⇒ 每集一个独立 Isaac 进程。
#   4) 录像开关**不影响**成败（已实测：同 seed 带视频 success@1602、不带 success@2838、
#      每步 render 不编码 两次都 timeout@3000 ⇒ 差异是 run-to-run 波动）。所以用最便宜的
#      "不录像" 口径；渲染节拍不是指标。
#
# 用法：SEEDS="0 1 2 3" MAX_STEPS=3000 WAVE=sr2 bash scripts/r3_sr_queue.sh
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$C" || exit 1

SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
MAX_STEPS=${MAX_STEPS:-3000}
WAVE=${WAVE:-sr2}
HEAD_V13=${HEAD_V13:-runs/joint_v13/vla_final.pt}
CKPT=${CKPT:-runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}
BASE=$C/runs/closedloop_sr/$WAVE
mkdir -p "$BASE"

# ---- 执行器口径（缺一不可）---------------------------------------------------
export HA_ALLOW_SONIC=1
export HA_GMT_BACKEND=sonic
export HA_ACTION_SOURCE=sonic_wholebody
export SONIC_VLA_ACTION_FORMAT=latent64
export SUPERMIMIC_HA_ALIGN=1
export HA_VLA_EXEC_STEPS=20
unset SONIC_DEBUG_DUMP

SRV_PY="$C:$C/Isaac-GR00T:$C/third_party/py312_shims:$C/third_party/gr00t_deps:$C/third_party/gr00t_deps/_noflash"
start_server() {  # $1=head|none $2=gpu $3=port $4=dir
  local head=$1 gpu=$2 port=$3 dir=$4
  mkdir -p "$dir"
  local args=()
  [[ "$head" != "none" ]] && args=(--head-ckpt "$head")
  setsid nohup env PYTHONPATH="$SRV_PY" python3 scripts/ha_vla_latent64_server.py \
    --ckpt "$CKPT" "${args[@]}" --port "$port" --device "cuda:$gpu" --seed 0 \
    > "$dir/server.log" 2>&1 &
  echo $!
}
wait_ready() {  # $1=pid $2=log $3=port
  for _ in $(seq 1 150); do
    grep -q "listening on" "$2" 2>/dev/null && return 0
    kill -0 "$1" 2>/dev/null || { echo "[sr] server :$3 崩了" >&2; tail -10 "$2" >&2; return 1; }
    sleep 2
  done
  echo "[sr] server :$3 超时" >&2; return 1
}

P_V13=$(start_server "$HEAD_V13" 0 18311 "$BASE/_srv_v13")
P_BASE=$(start_server none        1 18313 "$BASE/_srv_base")
wait_ready "$P_V13"  "$BASE/_srv_v13/server.log"  18311 || exit 3
wait_ready "$P_BASE" "$BASE/_srv_base/server.log" 18313 || exit 3
echo "[sr] servers ready: v13@18311(gpu0) base@18313(gpu1)  wave=$WAVE seeds='$SEEDS' max_steps=$MAX_STEPS"

U_V13=http://127.0.0.1:18311
U_BASE=http://127.0.0.1:18313

# ---- 任务队列 ---------------------------------------------------------------
QUEUE=$BASE/jobs.txt
DONE=$BASE/done.txt
: > "$QUEUE"; : > "$DONE"
for s in $SEEDS; do
  echo "sits_v13|$s|SitSofa|$HEAD_V13|$U_V13"     >> "$QUEUE"
  echo "vn_v13|$s|VisionNavi|$HEAD_V13|$U_V13"    >> "$QUEUE"
  echo "sits_base|$s|SitSofa|none|$U_BASE"        >> "$QUEUE"
  echo "vn_base|$s|VisionNavi|none|$U_BASE"       >> "$QUEUE"
done
echo "[sr] queue: $(wc -l < "$QUEUE") jobs"

LOCK=$BASE/.claim.lock
claim() {
  while :; do
    if mkdir "$LOCK" 2>/dev/null; then
      local line
      line=$(head -1 "$QUEUE")
      if [[ -z "$line" ]]; then rmdir "$LOCK"; return 1; fi
      tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
      rmdir "$LOCK"
      echo "$line"
      return 0
    fi
    sleep 2
  done
}

worker() {  # $1=gpu
  local gpu=$1 job cell seed task head url cdir
  while job=$(claim) && [[ -n "$job" ]]; do
    IFS='|' read -r cell seed task head url <<< "$job"
    cdir=$BASE/$cell
    mkdir -p "$cdir"
    echo "[w$gpu] $(date +%T) start $cell seed=$seed task=$task" | tee -a "$BASE/progress.log"
    TASK="$task" SEED="$seed" REPEAT=0 MAX_STEPS="$MAX_STEPS" ISAAC_GPU="$gpu" \
      SERVER_URL="$url" OUT_ROOT="$cdir/seed$seed" MODEL_LABEL="${cell##*_}" \
      RECORD_VIDEO=1 \
      EXTRA_ARGS="--step_log_every_n 500" \
      bash scripts/ha_isaac_closedloop.sh >> "$cdir/isaac_seed$seed.log" 2>&1
    local rc=$?
    local R
    R=$(ls -dt "$cdir/seed$seed"/*/result.json 2>/dev/null | head -1)
    if [[ -n "$R" && -f "$R" ]]; then
      cp -f "$R" "$cdir/ep_seed$seed.json"
      local prov=other
      grep -q "\[SonicActionProvider\]" "$cdir/isaac_seed$seed.log" && prov=sonic
      python3 - "$cdir/ep_seed$seed.json" "$prov" <<'PY'
import json, sys
p, prov = sys.argv[1], sys.argv[2]
d = json.load(open(p)); d["provider"] = prov
json.dump(d, open(p, "w"), indent=1)
PY
      echo "[w$gpu] $(date +%T) done  $cell seed=$seed rc=$rc prov=$prov $(python3 -c "
import json;d=json.load(open('$cdir/ep_seed$seed.json'))
print('success=%s reason=%s steps=%s dur=%.0fs' % (d.get('success'), d.get('failure_reason'), d.get('episode_steps'), float(d.get('duration_sec') or 0)))")" | tee -a "$BASE/progress.log"
    else
      echo "{\"seed\": $seed, \"success\": false, \"failure_reason\": \"no_result\", \"episode_steps\": 0, \"max_steps\": $MAX_STEPS, \"duration_sec\": 0, \"provider\": \"none\"}" > "$cdir/ep_seed$seed.json"
      echo "[w$gpu] $(date +%T) FAIL  $cell seed=$seed rc=$rc 无 result.json" | tee -a "$BASE/progress.log"
    fi
    echo "$cell|$seed|$rc" >> "$DONE"
  done
}

worker 0 &
W0=$!
worker 1 &
W1=$!
wait $W0 $W1

kill "$P_V13" "$P_BASE" 2>/dev/null; sleep 1; kill -9 "$P_V13" "$P_BASE" 2>/dev/null

python3 scripts/r3_sr_summary.py --dir "$BASE" | tee "$BASE/SUMMARY.txt"
echo "[sr] done $(date +%F' '%T)"
