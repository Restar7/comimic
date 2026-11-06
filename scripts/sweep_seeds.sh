#!/usr/bin/env bash
# 同一个 head、多个 seed 的闭环扫描（2026-11-05）
#
# 为什么要配对：闭环的 seed 间方差极大（base 三个 seed 的位置漂移 4.78 / 33.81 / 31.05 cm），
# 而组间差异往往只有 6 cm ⇒ 非配对比较根本分辨不出。**同 seed = 同 reset = 可配对**，
# 配对差能把方差压掉一大截，8 对就能判出 ±10% 的差异。
#
# 用法::
#   TAG=ftv8 HEAD=runs/joint_v8/vla_final.pt SEEDS="0 1 2 3 4 5 6 7" \
#     GPU_LIST="0 2 3" MAX_STEPS=1000 bash scripts/sweep_seeds.sh
#   不加 HEAD 就是 base 臂：TAG=base2 SEEDS="3 4 5 6 7" bash scripts/sweep_seeds.sh
#
# 注意：每条同时起一个 VLA 服务端（与 Isaac 同卡，见 r3_closedloop_ab.sh 的 SRV_GPU）。
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$C" || exit 1
TAG=${TAG:?TAG required}
HEAD=${HEAD:-none}
SEEDS=${SEEDS:-0 1 2}
MAX_STEPS=${MAX_STEPS:-1000}
TASK=${TASK:-DoubleDesk}
GPU_LIST=(${GPU_LIST:-0 2 3})
SRV_EXTRA=${SRV_EXTRA:---hand-from-base}

LOG=runs/_sweep_seeds_${TAG}.log
# ★ 2026-11-06：不再 `: >` 清空历史日志（同一个 TAG 跑第二批会在旧日志上覆盖，
#   事后判读时容易把两批的退出码/报错看混）。改成追加 + 分隔线。
{ echo; echo "==================== $(date -u +%FT%TZ) ===================="; } >> "$LOG"
say() { echo "[seeds $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "tag=$TAG head=$HEAD seeds=[$SEEDS] max_steps=$MAX_STEPS gpus=${GPU_LIST[*]}"

# ★ 2026-11-06：端口原来按槽位写死（18200 + slot*11）⇒ 两个 sweep 同时跑就撞端口，
#   先在 ICP 层报 "Address already in use"，表现为“上一条没起来”。改成**从基端口往上找空闲端口**。
pick_port() {
  local p=$1
  for _ in $(seq 1 40); do
    if ! (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then echo "$p"; return 0; fi
    exec 3>&- 2>/dev/null || true
    p=$((p + 1))
  done
  echo "$p"
}

i=0
for SEED in $SEEDS; do
  n=${#GPU_LIST[@]}
  if (( i >= n )); then
    say "等上一批 $n 条…"; wait; say "上一批结束"
  fi
  gpu=${GPU_LIST[$((i % n))]}
  port=$(pick_port $((18200 + (i % n) * 11)))
  out="${TAG}_s${SEED}"
  say "启动 $out head=$HEAD gpu=$gpu port=$port"
  TAG="$out" HEAD="$HEAD" GPU="$gpu" PORT="$port" SEED="$SEED" \
    MAX_STEPS="$MAX_STEPS" TASK="$TASK" SRV_EXTRA="$SRV_EXTRA" \
    setsid nohup bash scripts/r3_closedloop_ab.sh > "runs/_cl_${out}.log" 2>&1 < /dev/null &
  i=$((i + 1))
done
wait
say "全部结束"
say "判读："
say "  python3 scripts/eval_closedloop_progress.py --runs $(for s in $SEEDS; do printf '%s:=runs/closedloop_r3/%s_s%s ' "$TAG" "$TAG" "$s"; done)"
say "  python3 scripts/eval_closedloop_drift.py --runs $(for s in $SEEDS; do printf '%s:=runs/closedloop_r3/%s_s%s ' "$TAG" "$TAG" "$s"; done)"
