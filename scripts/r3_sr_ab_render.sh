#!/usr/bin/env bash
# 判定实验：不录像，但每个控制步都 sim.render()（EVAL_FORCE_RENDER_EVERY_STEP=1）。
#
# 要回答的问题：不录像时 VLA 读到的图到底晚不晚？
#   参照（同一任务/同一 seed/同一 object seed）：
#     带视频（record_video_every_n=1，每步 render） vn_v13 seed0 → success @1602
#     不带视频（record 0，每 20 步才 render）        vn_v13 seed0 → success @2838
#   本轮：不录像 + 每步 render。若回到 ~1600 ⇒ 是"帧滞后"的锅（render 节拍才是指标）；
#         若还是 ~2800 ⇒ 与 render 节拍无关，是普通 run-to-run 波动。
#
# 用法：TASK=VisionNavi HEAD=runs/joint_v13/vla_final.pt SEED=0 bash scripts/r3_sr_ab_render.sh
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$C" || exit 1

TASK=${TASK:-VisionNavi}
HEAD=${HEAD:-runs/joint_v13/vla_final.pt}
SEED=${SEED:-0}
MAX_STEPS=${MAX_STEPS:-3000}
OUT=${OUT:-$C/runs/closedloop_sr/ab_render}
CKPT=${CKPT:-runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}
mkdir -p "$OUT"

export HA_ALLOW_SONIC=1 HA_GMT_BACKEND=sonic HA_ACTION_SOURCE=sonic_wholebody
export SONIC_VLA_ACTION_FORMAT=latent64 SUPERMIMIC_HA_ALIGN=1 HA_VLA_EXEC_STEPS=20
export EVAL_FORCE_RENDER_EVERY_STEP=1
unset SONIC_DEBUG_DUMP
export HA_PIN_BYPASS=${HA_PIN_BYPASS:-1}   # 本实验改了 sim_eval_vla.py（每步 render）→ 契约锁需 bypass，属官方 A/B 通道

SRV_PY="$C:$C/Isaac-GR00T:$C/third_party/py312_shims:$C/third_party/gr00t_deps:$C/third_party/gr00t_deps/_noflash"
start_srv() {  # $1=gpu $2=port $3=dir
  mkdir -p "$3"
  setsid nohup env PYTHONPATH="$SRV_PY" python3 scripts/ha_vla_latent64_server.py \
    --ckpt "$CKPT" --head-ckpt "$HEAD" --port "$2" --device "cuda:$1" --seed 0 \
    > "$3/server.log" 2>&1 &
  echo $!
}
wait_srv() { for _ in $(seq 1 150); do grep -q "listening on" "$2" 2>/dev/null && return 0; sleep 2; done; return 1; }

echo "[ab] render-every-step: task=$TASK head=$HEAD seed=$SEED steps=$MAX_STEPS"
P0=$(start_srv 0 18411 "$OUT/_srv_gpu0")
P1=$(start_srv 1 18412 "$OUT/_srv_gpu1")
wait_srv 0 "$OUT/_srv_gpu0/server.log" || { echo "[ab] srv0 失败" >&2; exit 3; }
wait_srv 1 "$OUT/_srv_gpu1/server.log" || { echo "[ab] srv1 失败" >&2; exit 3; }
echo "[ab] servers ready"

run_one() {  # $1=gpu $2=srvpid $3=port $4=tag
  local gpu=$1 srv=$2 port=$3 tag=$4
  mkdir -p "$OUT/$tag"
  TASK="$TASK" SEED="$SEED" REPEAT=0 MAX_STEPS="$MAX_STEPS" ISAAC_GPU="$gpu" \
    SERVER_URL="http://127.0.0.1:$port" OUT_ROOT="$OUT/$tag" MODEL_LABEL=v13_render \
    RECORD_VIDEO=0 EXTRA_ARGS="--step_log_every_n 500" \
    bash scripts/ha_isaac_closedloop.sh > "$OUT/$tag/isaac.log" 2>&1
  local rc=$?
  local R; R=$(ls -dt "$OUT/$tag"/*/result.json 2>/dev/null | head -1)
  if [[ -f "$R" ]]; then
    cp -f "$R" "$OUT/$tag/ep.json"
    echo "[ab] $tag rc=$rc $(python3 -c "
import json;d=json.load(open('$R'))
print('success=%s reason=%s steps=%s dur=%.0fs provider=%s' % (d.get('success'), d.get('failure_reason'), d.get('episode_steps'), float(d.get('duration_sec') or 0), 'sonic' if 'SonicActionProvider' in open('$OUT/$tag/isaac.log',errors='ignore').read() else 'OTHER'))")"
  else
    echo "[ab] $tag rc=$rc 无 result.json"
  fi
}

run_one 0 "$P0" 18411 "gpu0_seed$SEED" &
A=$!
run_one 1 "$P1" 18412 "gpu1_seed${SEED}_rep" &
B=$!
wait $A $B

kill "$P0" "$P1" 2>/dev/null; sleep 1; kill -9 "$P0" "$P1" 2>/dev/null
echo "[ab] done $(date +%F' '%T)"
