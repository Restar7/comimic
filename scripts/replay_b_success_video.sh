#!/usr/bin/env bash
# 开环回放 B 线成功 episode 并录像（2026-11-02）
#
# 做法：把采集 dump 里的 (T,64) token 流原样喂给冻结 WBC（SM_TOKEN_SOURCE=npy），
# 起点/关节/reset 噪声全部按采集时那一集复现（SM_START_FRAME / SM_RESET_SEED），
# 因此画出来的是**当时那条成功轨迹**，不重新采样、不受 VLA 随机性影响。
#
# 前置：runs/B_success/replay/index.json 已由导出脚本生成（含 task/start_frame/reset_seed/max_steps）。
#
# 用法：
#   bash scripts/replay_b_success_video.sh Boxing_B901_299
#   bash scripts/replay_b_success_video.sh Boxing_B901_299 OpenDoor_B901_427 SitSofa_B901_710
#
# 产物：
#   runs/B_success/replay/<name>/isaac/<task>__seed0__repeat0__<stamp>/videos_{success,failure}/*.mp4
#   runs/B_success/videos/<name>__<reason>.mp4   （拷贝，方便直接看）
set -uo pipefail
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC" || exit 1

INDEX=$COMIMIC/runs/B_success/replay/index.json
[[ -f "$INDEX" ]] || { echo "缺少 $INDEX，先跑导出脚本" >&2; exit 2; }
[[ $# -ge 1 ]] || { echo "用法: bash scripts/replay_b_success_video.sh <name> [name...]" >&2; exit 2; }

mkdir -p "$COMIMIC/runs/B_success/videos"
# 2026-11-02 本机踩坑：/etc/vulkan/icd.d/nvidia_icd.json 是 0 字节，Isaac 报
# VkResult: ERROR_INCOMPATIBLE_DRIVER。用 /usr/share 下那份（指向 libGLX_nvidia.so.0）才行。
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/xdg}
mkdir -p "$XDG_RUNTIME_DIR"
n_ok=0; n_bad=0
for NAME in "$@"; do
  field() { python3 -c "import json;print(json.load(open('$INDEX'))['$NAME']['$1'])"; }
  TASK=$(field task); ROW=$(field start_frame); RESET=$(field reset_seed)
  T=$(field max_steps); TOK=$(field tokens)
  # ha_isaac_closedloop.sh 会 cd 到 PROJECT_ROOT，token 路径必须是绝对路径
  [[ "$TOK" = /* ]] || TOK=$COMIMIC/$TOK
  OUT=$COMIMIC/runs/B_success/replay/$NAME/isaac
  rm -rf "$OUT"; mkdir -p "$OUT"
  LOG=$COMIMIC/runs/B_success/replay/$NAME/replay.log

  echo "[replay] $(date +%T) $NAME task=$TASK row=$ROW reset=$RESET T=$T gpu=0 out=$OUT"
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    TASK="$TASK" SEED=0 REPEAT=0 MAX_STEPS="$T" ISAAC_GPU=0 \
    MODEL_LABEL="$NAME" OUT_ROOT="$OUT" \
    SM_CKPT="$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt" \
    SM_TOKEN_SOURCE=npy SM_TOKEN_NPY="$TOK" \
    SM_START_FRAME="$ROW" SM_INIT_FROM_LIB=1 \
    SM_RESET_NOISE=1 SM_RESET_SEED="$RESET" SM_RESET_SCALE=1.0 \
    SM_HAND=0 HA_VLA_EXEC_STEPS=20 RECORD_VIDEO=1 \
    EXTRA_ARGS="--step_log_every_n 50" \
    bash "$COMIMIC/scripts/ha_isaac_closedloop.sh" > "$LOG" 2>&1
  rc=$?
  mp4=$(find "$OUT" -name '*.mp4' 2>/dev/null | head -1)
  if [[ $rc -eq 0 && -n "$mp4" ]]; then
    reason=$(basename "$mp4" .mp4); reason=${reason##*__}
    cp -f "$mp4" "$COMIMIC/runs/B_success/videos/${NAME}__${reason}.mp4"
    n_ok=$((n_ok+1)); echo "[replay] $(date +%T) OK  -> runs/B_success/videos/${NAME}__${reason}.mp4"
  else
    n_bad=$((n_bad+1)); echo "[replay] $(date +%T) FAIL rc=$rc mp4='${mp4:-none}' 日志: $LOG"
    tail -5 "$LOG" | sed 's/^/[replay]   /'
  fi
done
echo "[replay] 完成 ok=$n_ok bad=$n_bad"
[[ $n_bad -eq 0 ]]
