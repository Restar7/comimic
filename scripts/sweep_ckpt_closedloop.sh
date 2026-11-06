#!/usr/bin/env bash
# 检查点闭环扫描（2026-11-05）
#
# 为什么需要：联合微调的"想象指标"已被证明不可信（round-3 崩前 win 率 0.88 是全场最好看）。
# 而且 round-4（唯一 ≈base 的一轮）只跑了 **100 次优化器更新**，round-3 跑了 1000 次就崩
# ⇒ 动作漂移是"更新次数"的函数，最优点很可能在早期检查点上。
# 所以固定 seed / 固定 MAX_STEPS，把 base 与各 step 的 head 一起扫，看悬崖在哪。
#
# 用法::
#   GPU_LIST="0 2 3" SEED=0 MAX_STEPS=1000 bash scripts/sweep_ckpt_closedloop.sh \
#       base:=none \
#       f100:=runs/joint_v5/vla_000100.pt \
#       f200:=runs/joint_v5/vla_000200.pt
#
# 每卡同时只跑一条（服务端与 Isaac 同卡，SRV_GPU 默认跟随 GPU）。
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$C" || exit 1
SEED=${SEED:-0}
MAX_STEPS=${MAX_STEPS:-1000}
TASK=${TASK:-DoubleDesk}
GPU_LIST=(${GPU_LIST:-0 2 3})
SUFFIX=${SUFFIX:-s${SEED}}
read -r -a PAIRS <<< "$*"
[[ ${#PAIRS[@]} -gt 0 ]] || { echo "[sweep-cl] 需要至少一个 tag:=path" >&2; exit 2; }

LOG=runs/_sweep_closedloop_${SUFFIX}.log
: > "$LOG"
say() { echo "[sweep-cl $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "seed=$SEED max_steps=$MAX_STEPS task=$TASK gpus=${GPU_LIST[*]} arms=${#PAIRS[@]}"

i=0
for pair in "${PAIRS[@]}"; do
  gpu=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}
  tag=${pair%%:=*}
  head=${pair#*:=}
  # 每轮最多并发 ${#GPU_LIST[@]} 条；这一批跑满就等
  if (( i >= ${#GPU_LIST[@]} )); then
    say "等待上一批 3 条结束…"
    wait
    say "上一批结束"
  fi
  port=$((18100 + (i % ${#GPU_LIST[@]}) * 7))
  OUTTAG="${tag}_${SUFFIX}"
  say "启动 $OUTTAG  head=$head  gpu=$gpu port=$port"
  TAG="$OUTTAG" HEAD="$head" GPU="$gpu" PORT="$port" SEED="$SEED" \
    MAX_STEPS="$MAX_STEPS" TASK="$TASK" SRV_EXTRA="--hand-from-base" \
    setsid nohup bash scripts/r3_closedloop_ab.sh \
      >> "runs/_cl_${OUTTAG}.log" 2>&1 < /dev/null &
  i=$((i + 1))
done
wait
say "全部结束。判读："
say "  python3 scripts/eval_closedloop_progress.py $(for p in "${PAIRS[@]}"; do printf '%s:=runs/closedloop_r3/%s_%s ' "${p%%:=*}" "${p%%:=*}" "$SUFFIX"; done)"
