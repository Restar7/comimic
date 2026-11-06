#!/usr/bin/env bash
# 每卡 batch / accum 速度对照（2026-11-05）
#   目的：全局 batch 固定 12（= 3 卡 × 每卡 batch × accum），只改 microbatch 切法，
#   看每步墙钟掉多少。语义等价：v3_step 里 `(loss/accum).backward()`，loss 是 batch 内 mean。
#   输出：runs/_speed_p{PER}a{ACC}.log（含 J1_STEP_PROFILE=1 的分段耗时）
#
# ⚠️ 2026-11-05 踩坑：launch_joint_ha_v2.sh 是 `set -euo pipefail`，而它的 `_ARGS=()` 里
#   有一句 `$([[ "$TERM_OFF" == "1" ]] && echo --term-off)`。TERM_OFF≠1 时该替换返回 1，
#   在 set -e 下整个 launch 立刻静默退出（rc=1、零报错）。已修成 `|| true`；这里仍显式给
#   TERM_OFF=1（round-5 口径要求 survival≡1）。
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$C" || exit 1
PORTS=(29611 29612 29613 29614)

( while true; do
    printf '%s ' "$(date +%H:%M:%S)"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' '|'
    echo
    sleep 2
  done ) > runs/_sweep_mem.log 2>&1 &
MEM=$!
trap 'kill $MEM 2>/dev/null' EXIT

i=0
for cfg in "1 4" "2 2" "4 1" "8 1"; do
  set -- $cfg; PER=$1; ACC=$2
  BATCH=$((PER * 3))
  LOG=runs/_speed_p${PER}a${ACC}.log
  RUNDIR=runs/speed_p${PER}a${ACC}
  rm -rf "$RUNDIR"
  echo "=== [sweep] per-card=$PER accum=$ACC global=$((PER*3*ACC)) t=$(date +%H:%M:%S)" \
    | tee -a runs/_sweep_summary.log
  BATCH=$BATCH ACCUM=$ACC STEPS=6 SAVE_EVERY=0 EVAL_EVERY=0 TERM_OFF=1 \
  RUN=speed_p${PER}a${ACC} DDP=1 NPROC=3 DDP_GPUS=0,2,3 \
  MASTER_PORT=${PORTS[$i]} J1_STEP_PROFILE=1 \
  timeout 900 bash scripts/run_joint_ft_round5.sh > "$LOG" 2>&1
  rc=$?
  # 硬门：日志里必须真的出现 s_per_step，否则这条对照无效（别把"没跑起来"当成"很快"）
  if grep -q "s_per_step" "$LOG"; then
    echo "[sweep] per-card=$PER accum=$ACC rc=$rc OK" | tee -a runs/_sweep_summary.log
    grep -E "s_per_step" "$LOG" | tail -2 | tee -a runs/_sweep_summary.log
    grep -E "prof-step" "$LOG" | tail -2 | tee -a runs/_sweep_summary.log
  else
    echo "[sweep] per-card=$PER accum=$ACC rc=$rc **INVALID（没有跑到训练，见 $LOG 尾部）**" \
      | tee -a runs/_sweep_summary.log
    tail -5 "$LOG" | tee -a runs/_sweep_summary.log
  fi
  i=$((i + 1))
done
echo "=== [sweep] done" | tee -a runs/_sweep_summary.log
