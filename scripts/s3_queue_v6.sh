#!/usr/bin/env bash
# S3 v6：**复用已标定好的 WM（C6h 自带），速度奖励全开（1.0），手部监督关掉**。
#
# 与 v4/v5 的两个关键纠正（都是我前一轮判断错的地方）：
#   1. 速度奖励不该关。C6h 的 WM 是按 `eval_wm_velocity.py` 口径**标定到 qd_corr 0.906**
#      的（`runs/S1_ctrl/wmvel_A20000.log`：0.126→0.891→0.906，800 步标定）。
#      我之前引用的 0.545/0.70 是 `eval_wm_real` 在**另一批数据**（ha_transitions）上的读数，
#      两个仪器不能混用，据此关掉速度项是错的。
#   2. 不再做 in-loop WM 保鲜（`wm_refresh_every=0`）：那个刷新正是按 `eval_wm_real` 口径做的，
#      会把已经标定好的 WM 往另一个数据分布上拽，KL 锚定只限制幅度、不保证方向。
#      ⇒ **直接复用 C6h 里那个标定好的 WM**。
#
# 其余沿用 v4/v5（hand-off，body-only）：
#   H=5 可微视界 · grounded critic（真录制 return-to-go） · 动作空间约束（中心=标签 token 的 rollout）
#
# 两臂：act_eps 0.30 / 0.60（约束松紧对撞）；对照 = step 0 评测（= 底模本身）。
set -uo pipefail
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC" || exit 1

STEPS=${STEPS:-1000}
H=${H:-5}
DR_XY=${DR_XY:-0.10}
DR_YAW=${DR_YAW:-0.125}
TAG=${TAG:-s3v6}

echo "[v6] $(date +%T) 等待 GPU 空闲"
while pgrep -f "[t]rain_joint_v2.py --motion" >/dev/null 2>&1; do sleep 15; done
echo "[v6] $(date +%T) 开始（vel_scale=1.0, wm_refresh=0, lambda_hand=0）"

run_arm() {   # $1=gpu $2=run $3=act_eps
  echo "[v6] $(date +%T) start gpu=$1 run=$2 act_eps=$3"
  GPU="$1" RUN="$2" STEPS="$STEPS" H="$H" LAMBDA_R=0.1 LAMBDA_HAND=0 \
  DR_XY="$DR_XY" DR_YAW="$DR_YAW" ANCHOR=0 ACT_EPS="$3" ACT_WEIGHT=1.0 \
  REWARD_VEL_SCALE=1.0 CRITIC_SOURCE=real \
  CRITIC_REAL="$COMIMIC/runs/wm_real/real_returns.pt" \
  WM_REFRESH_EVERY=0 \
  SAVE_EVERY=200 EVAL_EVERY=100 EVAL_BATCHES=4 \
    bash scripts/s3_launch_one.sh > "logs/$2.launcher.log" 2>&1
  echo "[v6] $(date +%T) done gpu=$1 run=$2 rc=$?"
}

run_arm 0 "${TAG}_e30" 0.30 &
P0=$!
run_arm 1 "${TAG}_e60" 0.60 &
P1=$!
wait $P0 $P1
echo "[v6] $(date +%T) 结束：runs/${TAG}_e30, runs/${TAG}_e60"
