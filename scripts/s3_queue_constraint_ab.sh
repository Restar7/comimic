#!/usr/bin/env bash
# S3 约束形状对撞（每臂**只有一种**约束，不是两个都开）。
#
#   墙   : trust(eps=0.15, w=1.0)，anchor = 0     —— 球内 0 力，球外二次增长
#   弹簧 : λ_a = 4.0，trust = 0                    —— 全程二次拉力（r=0.3 处 ≈ 奖励项的 0.9×）
#
# 两臂其它全同：DR 0.10 m / λ_r 0.1 / H 20 / K 4 / batch 4 × accum 8 / 300 步。
# 判据：tok_mse ≤1.5× 且 ret_vla 上升 者胜；看 trust/frac_out 判断墙是否被顶到边界。
#
# 会**等**已有训练结束再起（GPU 空闲检测），所以可以在上一轮还没跑完时就挂上去。
set -uo pipefail
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC" || exit 1

STEPS=${STEPS:-300}
DR_XY=${DR_XY:-0.10}
DR_YAW=${DR_YAW:-0.125}
LAMBDA_R=${LAMBDA_R:-0.1}
TAG=${TAG:-s3h20}

echo "[ab] $(date +%T) 等待 GPU 空闲（现有训练退出后自动开始）"
while pgrep -f "[t]rain_joint_v2.py --motion" >/dev/null 2>&1; do sleep 20; done
echo "[ab] $(date +%T) GPU 空闲，开始对撞"

run_arm() {   # $1=gpu $2=run $3=ANCHOR $4=TRUST_EPS $5=TRUST_WEIGHT
  echo "[ab] $(date +%T) start gpu=$1 run=$2 anchor=$3 trust=($4,$5)"
  GPU="$1" RUN="$2" STEPS="$STEPS" LAMBDA_R="$LAMBDA_R" \
  DR_XY="$DR_XY" DR_YAW="$DR_YAW" \
  ANCHOR="$3" TRUST_EPS="$4" TRUST_WEIGHT="$5" \
    bash scripts/s3_launch_one.sh > "logs/$2.launcher.log" 2>&1
  echo "[ab] $(date +%T) done gpu=$1 run=$2 rc=$?"
}

# 墙收紧到 eps=0.12：上一轮 eps=0.15 时 trust/frac_out 全程为 0（墙一次都没被触发），
# 而 token 的 RMSE 只走到 0.114 就把解码后的 tok_mse 顶到 ×2.9 —— 说明 0.15 太松。
# 底模自身 RMSE ≈ 0.0995，取 0.12 = 允许 1.2× 的偏差。
run_arm 0 "${TAG}_dr10_wall12"  0    0.12 1.0   &
P0=$!
run_arm 1 "${TAG}_dr10_spring4" 4.0  0    0     &
P1=$!
wait $P0 $P1

echo "[ab] $(date +%T) 对撞结束。"
echo "[ab] 墙:   runs/${TAG}_dr10_wall/eval_curve.jsonl"
echo "[ab] 弹簧: runs/${TAG}_dr10_spring/eval_curve.jsonl"
