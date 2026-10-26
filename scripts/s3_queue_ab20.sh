#!/usr/bin/env bash
# S3 干净 A/B（H=20 部署口径，单一变量 = 奖励项开关）。
#
# 前提修正（相对上一轮）：
#   1. H 回到 **20**（部署重规划周期），不再做 H 阶梯；
#   2. 约束只留**一个**：trust(eps=0.12, w=1.0)，anchor=0（上一轮 eps=0.15 时 frac_out 全程 0，墙没生效）；
#   3. DR 固定 0.10（5 vs 10 cm 实测无差别，选中间档）；
#   4. 两臂**只差 λ_r**：GPU0 = 0.1、GPU1 = 0.0。
#
# 判据（先写死）：
#   - GPU1(λ_r=0) 是"没有奖励梯度"的对照：tok_mse 必须 ≈1×；
#   - GPU0(λ_r=0.1) 看三件事：tok_mse 是否 ≤1.5×、trust/frac_out 是否 >0（墙是否真在顶）、
#     ret_vla 是否高于对照组且不超过 ret_gt 太多（超过 GT 就是 hacking 的信号）。
set -uo pipefail
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC" || exit 1

STEPS=${STEPS:-300}
H=${H:-20}
DR_XY=${DR_XY:-0.10}
DR_YAW=${DR_YAW:-0.125}
TRUST_EPS=${TRUST_EPS:-0.12}
TRUST_WEIGHT=${TRUST_WEIGHT:-1.0}
TAG=${TAG:-s3h20ab}

echo "[ab20] $(date +%T) 等待 GPU 空闲"
while pgrep -f "[t]rain_joint_v2.py --motion" >/dev/null 2>&1; do sleep 15; done
echo "[ab20] $(date +%T) 开始"

run_arm() {   # $1=gpu $2=run $3=lambda_r
  echo "[ab20] $(date +%T) start gpu=$1 run=$2 lam_r=$3"
  GPU="$1" RUN="$2" STEPS="$STEPS" H="$H" LAMBDA_R="$3" \
  DR_XY="$DR_XY" DR_YAW="$DR_YAW" ANCHOR=0 \
  TRUST_EPS="$TRUST_EPS" TRUST_WEIGHT="$TRUST_WEIGHT" \
    bash scripts/s3_launch_one.sh > "logs/$2.launcher.log" 2>&1
  echo "[ab20] $(date +%T) done gpu=$1 run=$2 rc=$?"
}

run_arm 0 "${TAG}_lam010" 0.1 &
P0=$!
run_arm 1 "${TAG}_lam000" 0.0 &
P1=$!
wait $P0 $P1
echo "[ab20] $(date +%T) 结束：runs/${TAG}_lam010, runs/${TAG}_lam000"
