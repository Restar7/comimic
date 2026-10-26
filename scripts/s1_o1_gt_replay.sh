#!/usr/bin/env bash
# =============================================================================
# O1 门禁（S1.3）：把**录制动作的 GT token** 回放进 HA 闭环，测执行器上限。
# 2026-10-26 口径：默认 TOKEN_SPACE=label（npz 的 g1_motion_token_64）。端口侧车已被 S0.4
# 证明是错装配，只有老端口空间 WBC（ckpt_016000*）才用 TOKEN_SPACE=port 当参照，且不进 A1 表。
#
#   执行器 = SM_CKPT（WBC），VLA = scripts/ha_port_token_server.py（回放 GT token）
#   ⇒ 闭环还摔 = 执行器 / 场景的问题；能站能走 = 执行器没问题，瓶颈在 VLA。
#
# 用法：
#   bash scripts/s1_o1_gt_replay.sh                        # PPBox + DoubleDesk, seeds 0 1 2
#   TASKS=PPBox SEEDS="0" MAX_STEPS=120 bash scripts/s1_o1_gt_replay.sh   # 冒烟
#
# 产物：runs/S1_closedloop/O1/<Task>__seed<N>__repeat0__<ts>/result.json + 视频
# =============================================================================
set -uo pipefail
COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}

SM_CKPT=${SM_CKPT:-$COMIMIC/trackerlab-supermimic/logs/wbc_mix2_obs93/ckpt_016000_wmvel.pt}
OUT_ROOT=${OUT_ROOT:-$COMIMIC/runs/S1_closedloop/O1}
TASKS=${TASKS:-"PPBox DoubleDesk"}
SEEDS=${SEEDS:-"0 1 2"}
MAX_STEPS=${MAX_STEPS:-300}
PORT=${PORT:-18082}
ISAAC_GPU=${ISAAC_GPU:-0}
CLIP_PPBox=${CLIP_PPBox:-HOI_pp_box__yb__ep000000}
CLIP_DoubleDesk=${CLIP_DoubleDesk:-HOI_double_desk__tw__ep000000}
TOKEN_SPACE=${TOKEN_SPACE:-label}
PY=${PY:-$COMIMIC/third_party/trackerlab_env/bin/python}

mkdir -p "$OUT_ROOT"
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "O1: ckpt=$SM_CKPT token_space=$TOKEN_SPACE"
say "O1: tasks=$TASKS seeds=$SEEDS max_steps=$MAX_STEPS gpu=$ISAAC_GPU out=$OUT_ROOT"

for T in $TASKS; do
  var="CLIP_$T"; CLIP=${!var:-}
  [[ -z "$CLIP" ]] && { say "WARN: 未知任务 $T（没有 CLIP_$T）"; continue; }
  say "=== [$T] clip=$CLIP"
  "$PY" "$COMIMIC/scripts/ha_port_token_server.py" --clip "$CLIP" --token-space "$TOKEN_SPACE" --port "$PORT" \
      > "$OUT_ROOT/server_$T.log" 2>&1 &
  SRV=$!
  sleep 3
  if ! curl -sf "http://127.0.0.1:$PORT/" >/dev/null; then
    say "FAIL: port-token server 没起来（看 $OUT_ROOT/server_$T.log）"; kill $SRV 2>/dev/null; continue
  fi
  for S in $SEEDS; do
    curl -s -X POST "http://127.0.0.1:$PORT/reset" -d '{}' >/dev/null 2>&1 || true
    say "[$T seed=$S] 开始"
    TASK=$T SEED=$S REPEAT=0 MAX_STEPS=$MAX_STEPS \
      SM_CKPT="$SM_CKPT" SM_TOKEN_SOURCE=http SERVER_URL="http://127.0.0.1:$PORT" \
      ISAAC_GPU=$ISAAC_GPU OUT_ROOT="$OUT_ROOT" \
      bash "$COMIMIC/scripts/ha_isaac_closedloop.sh" > "$OUT_ROOT/${T}_seed${S}.log" 2>&1
    R="$(ls -dt "$OUT_ROOT/${T}__seed${S}__repeat0__"* 2>/dev/null | head -1)/result.json"
    if [[ -f "$R" ]]; then
      "$PY" - "$R" "$T" "$S" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"[O1] {sys.argv[2]:<11} seed={sys.argv[3]} steps={d.get('episode_steps'):>4} "
      f"failure={str(d.get('failure_reason')):<9} success={d.get('success')} "
      f"max_reward={float(d.get('max_reward') or 0):.3f}", flush=True)
PY
    else
      say "[O1] $T seed=$S: 没有 result.json（看 $OUT_ROOT/${T}_seed${S}.log）"
    fi
  done
  kill $SRV 2>/dev/null
  wait $SRV 2>/dev/null
done
say "O1 done -> $OUT_ROOT"
