#!/usr/bin/env bash
# S3 DR 阶梯（bench）：把 DR 从"拍脑袋 20 cm"改成**测出来**的档位。
#
# 背景：奖励是跟踪奖励。状态不偏离 ⇒ 奖励接近饱和 ⇒ 梯度里只剩模型误差方向（纯作弊）；
# 状态偏离太大 ⇒ WBC/WM 进入外推区，裁判开始瞎猜。所以 DR 既不能是 0，也不能随便大。
# 这一轮用**同一个 trust region** 扫两档，看哪一档还能"token 不漂、回报上升"。
#
#   R1: GPU0 = DR 5 cm   GPU1 = DR 10 cm   （λ_r=0.1, trust(0.15,1.0), 300 步）
#   R2: 取 R1 里没崩的那档，跑 λ_r=0 vs λ_r=0.1 的真 A/B（长跑）
#
# 用法：
#   setsid nohup bash scripts/s3_queue_dr_ladder.sh > logs/dr_ladder.queue.log 2>&1 < /dev/null &
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC" || exit 1

STEPS=${STEPS:-300}
TRUST_EPS=${TRUST_EPS:-0.15}
TRUST_WEIGHT=${TRUST_WEIGHT:-1.0}
LAMBDA_R=${LAMBDA_R:-0.1}
TAG=${TAG:-s3h20}

run_arm() {   # $1=gpu $2=run $3=dr_xy $4=dr_yaw
  echo "[ladder] $(date +%T) start gpu=$1 run=$2 dr=($3,$4)"
  GPU="$1" RUN="$2" STEPS="$STEPS" LAMBDA_R="$LAMBDA_R" \
  DR_XY="$3" DR_YAW="$4" TRUST_EPS="$TRUST_EPS" TRUST_WEIGHT="$TRUST_WEIGHT" \
    bash scripts/s3_launch_one.sh > "logs/$2.launcher.log" 2>&1
  echo "[ladder] $(date +%T) done gpu=$1 run=$2 rc=$?"
}

run_arm 0 "${TAG}_dr05_t" 0.05 0.0625 &
P0=$!
run_arm 1 "${TAG}_dr10_t" 0.10 0.125 &
P1=$!
wait $P0 $P1

echo "[ladder] $(date +%T) R1 完成：看 logs/${TAG}_dr05_t.log / logs/${TAG}_dr10_t.log 的 step100 读"
echo "[ladder] 判据：tok_mse ≤1.5× 且 ret_vla 上升 → 该档可进长跑；否则降档"
