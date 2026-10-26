#!/usr/bin/env bash
# S3 "裁判几步之内可信" 阶梯：固定同一个墙约束，只变想象步数 H。
#
#   动机：用户的核心疑问是"咋整都偏"——一直漂说明奖励梯度指向的不是"真做得更好"，
#   而是"让 WM 以为更好"。而 WM 的误差随步数累积，所以 H 越小裁判越可信。
#   这一轮不是调参，是**判定这个裁判到底能不能用**：
#     H=2（几乎一步信用）与 H=5，同一个 trust(0.12,1.0)、同一个 DR 0.10、λ_r=0.1。
#
#   读法：
#     - 两档都 tok_mse 稳、ret_vla 上升   → 裁判在小 H 下可信，之后逐步拉长 H
#     - 两档都漂                          → WM 连 2 步都不可信 ⇒ 没有仿真器这条路走不通
#     - H=2 稳、H=5 漂                    → 有效视界在 2~5 步之间，按该视界训练
#
# 用法：setsid nohup bash scripts/s3_queue_horizon_ladder.sh > logs/h_ladder.queue.log 2>&1 < /dev/null &
set -uo pipefail
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC" || exit 1

STEPS=${STEPS:-300}
DR_XY=${DR_XY:-0.10}
DR_YAW=${DR_YAW:-0.125}
TRUST_EPS=${TRUST_EPS:-0.12}
TRUST_WEIGHT=${TRUST_WEIGHT:-1.0}
LAMBDA_R=${LAMBDA_R:-0.1}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-8}
TAG=${TAG:-s3hladder}

echo "[hladder] $(date +%T) 等待 GPU 空闲"
while pgrep -f "[t]rain_joint_v2.py --motion" >/dev/null 2>&1; do sleep 15; done
echo "[hladder] $(date +%T) 开始"

run_arm() {   # $1=gpu $2=run $3=H
  echo "[hladder] $(date +%T) start gpu=$1 run=$2 H=$3"
  GPU="$1" RUN="$2" STEPS="$STEPS" H="$3" LAMBDA_R="$LAMBDA_R" \
  DR_XY="$DR_XY" DR_YAW="$DR_YAW" ANCHOR=0 \
  TRUST_EPS="$TRUST_EPS" TRUST_WEIGHT="$TRUST_WEIGHT" \
  BATCH="$BATCH" ACCUM="$ACCUM" \
    bash scripts/s3_launch_one.sh > "logs/$2.launcher.log" 2>&1
  echo "[hladder] $(date +%T) done gpu=$1 run=$2 rc=$?"
}

run_arm 0 "${TAG}_H2"  2 &
P0=$!
run_arm 1 "${TAG}_H5"  5 &
P1=$!
wait $P0 $P1
echo "[hladder] $(date +%T) 结束：runs/${TAG}_H2, runs/${TAG}_H5"
