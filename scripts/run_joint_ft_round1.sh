#!/usr/bin/env bash
# 联合微调（第 1 轮，2026-11-03）
#   H=20（= 部署重规划周期）、每段重置的 A' 数据、δ_pos=0.10
#   依赖：runs/a_prime_seg20/filtered.pt（A' 转移）+ real_returns.pt（critic 真回报）
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
DIR=${DIR:-$C/runs/a_prime_seg20}
cd "$C" || exit 1
for f in "$DIR/filtered.pt" "$DIR/real_returns.pt"; do
  [[ -f "$f" ]] || { echo "[joint] 缺 $f，先跑 run_a_prime_pipeline.sh" >&2; exit 2; }
done
export V3=1
export H=20
export REWARD_DELTA_POS=${REWARD_DELTA_POS:-0.10}
export REWARD_DELTA_ORI=${REWARD_DELTA_ORI:-0.05}
export CRITIC_ON=${CRITIC_ON:-1}
export CRITIC_REAL=${CRITIC_REAL:-$DIR/critic_real.pt}
export EXTRA="${EXTRA:---real-transitions $DIR/filtered.pt}"
export RUN=${RUN:-joint_v3_round1}
export STEPS=${STEPS:-300}
export DDP=${DDP:-1}
export NPROC=${NPROC:-4}
export DDP_GPUS=${DDP_GPUS:-0,1,2,3}
export BATCH=${BATCH:-4}
export ACCUM=${ACCUM:-4}
echo "[joint] RUN=$RUN STEPS=$STEPS H=$H delta_pos=$REWARD_DELTA_POS critic=$CRITIC_ON"
echo "[joint] real_transitions=$DIR/filtered.pt  critic_real=$CRITIC_REAL"
exec bash scripts/launch_joint_ha_v2.sh
