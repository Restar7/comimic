#!/usr/bin/env bash
# S3 v4 正式配方（A 方案落地）：真正端到端的梯度，但换掉三个不可信成分。
#
#   ① 可微视界 H=5（WM 准的地方），长期信用交给**真数据**训练的 critic（γ^H·V(s_H)）
#   ② 奖励只用 WM 可信的通道：root 位置/姿态 + 关节位置，**扣掉速度两项**（qd_corr≈0.64）
#   ③ 约束加在**动作空间**：中心 = 用标签 token 跑同一段 rollout 的动作序列，eps=0.30
#      （≈1.4× 底模自身动作偏差 0.21），取代 token 空间的 RMSE 约束与 λ_a 弹簧
#
# 两臂只差奖励开关：
#   GPU0 `v4_lam010`：λ_r=0.1（奖励梯度开）
#   GPU1 `v4_lam000`：λ_r=0  （对照：没有奖励梯度时 token 应纹丝不动）
#
# 判据（先写死，别事后解释）：
#   * 对照组 tok_mse 必须 ≈1×（否则说明有别的路径在动 token）；
#   * 实验组：critic/v_ev 应 >0.3（grounded critic 学到了）；act/frac_out 应在 0.1–0.5
#     （约束在起作用但没把人压死）；tok_mse ≤1.5×；ret_vla 高于对照组**且不超过 ret_gt 太多**；
#   * 任何一条不满足 → 按计划走下一步（换 Isaac 当裁判），不再调约束。
set -uo pipefail
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC" || exit 1

STEPS=${STEPS:-1000}
H=${H:-5}
DR_XY=${DR_XY:-0.10}
DR_YAW=${DR_YAW:-0.125}
TAG=${TAG:-s3v4}

echo "[v4] $(date +%T) 等待 GPU 空闲"
while pgrep -f "[t]rain_joint_v2.py --motion" >/dev/null 2>&1; do sleep 15; done
echo "[v4] $(date +%T) 开始（H=$H steps=$STEPS）"

run_arm() {   # $1=gpu $2=run $3=lambda_r
  echo "[v4] $(date +%T) start gpu=$1 run=$2 lam_r=$3"
  GPU="$1" RUN="$2" STEPS="$STEPS" H="$H" LAMBDA_R="$3" \
  DR_XY="$DR_XY" DR_YAW="$DR_YAW" ANCHOR=0 \
  SAVE_EVERY=200 EVAL_EVERY=100 EVAL_BATCHES=4 \
    bash scripts/s3_launch_one.sh > "logs/$2.launcher.log" 2>&1
  echo "[v4] $(date +%T) done gpu=$1 run=$2 rc=$?"
}

run_arm 0 "${TAG}_lam010" 0.1 &
P0=$!
run_arm 1 "${TAG}_lam000" 0.0 &
P1=$!
wait $P0 $P1
echo "[v4] $(date +%T) 结束：runs/${TAG}_lam010, runs/${TAG}_lam000"
