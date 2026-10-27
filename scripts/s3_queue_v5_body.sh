#!/usr/bin/env bash
# S3 v5「全身」实验：**手部监督关掉**（λ_hand=0），只让全身奖励梯度训练 token。
#
# 起因（v4 实测，决定性）：
#   λ_r=0.1 与 λ_r=0 两臂跑满 1000 步，**逐位相同**（tok_mse 0.00598 vs 0.00630、
#   ret_vla +0.0659 vs +0.0659、shift ×1.18 vs ×1.20）——
#   ⇒ 奖励分支对 token 的贡献是 0；
#   ⇒ 而 token 仍然漂到 ×1.3~1.4，来源只能是**共享 head 上的手部 BC**（λ_hand=0.2）。
#   而部署时手部根本不过 WBC（provider 里走独立的 `_hand_queue` 14 维通道），
#   所以手部 BC 与"全身奖励反传"是两条互不相干的通路，只共享 head 的 trunk。
#
# 这一轮：
#   GPU0 `v5_body_e30`：λ_hand=0, λ_r=0.1, act_eps=0.30（与 v4 同宽，看是否被勒死）
#   GPU1 `v5_body_e60`：λ_hand=0, λ_r=0.1, act_eps=0.60（放宽一倍，看有没有空间）
#   对照 = step 0 的评测（λ_hand=0/λ_r=0 时不训练，等价于底模本身）
#
# 判据：
#   * 若两臂 ret_vla 都不动（与 step0 的 +0.0664 相同）⇒ **有效域内没有可学信号**
#     ⇒ 按计划换 Isaac 当裁判（best-of-K / PPO），不再调约束；
#   * 若 e60 动而 e30 不动 ⇒ 约束是瓶颈，用 e60 做长跑并盯 tok_mse；
#   * 若都动且 tok_mse 稳 ⇒ 终于拿到"全身奖励有效"的第一个正证据。
set -uo pipefail
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC" || exit 1

STEPS=${STEPS:-1000}
H=${H:-5}
DR_XY=${DR_XY:-0.10}
DR_YAW=${DR_YAW:-0.125}
TAG=${TAG:-s3v5}

echo "[v5] $(date +%T) 等待 GPU 空闲"
while pgrep -f "[t]rain_joint_v2.py --motion" >/dev/null 2>&1; do sleep 15; done
echo "[v5] $(date +%T) 开始（λ_hand=0，body-only）"

run_arm() {   # $1=gpu $2=run $3=act_eps
  echo "[v5] $(date +%T) start gpu=$1 run=$2 act_eps=$3"
  GPU="$1" RUN="$2" STEPS="$STEPS" H="$H" LAMBDA_R=0.1 LAMBDA_HAND=0 \
  DR_XY="$DR_XY" DR_YAW="$DR_YAW" ANCHOR=0 ACT_EPS="$3" ACT_WEIGHT=1.0 \
  SAVE_EVERY=200 EVAL_EVERY=100 EVAL_BATCHES=4 \
    bash scripts/s3_launch_one.sh > "logs/$2.launcher.log" 2>&1
  echo "[v5] $(date +%T) done gpu=$1 run=$2 rc=$?"
}

run_arm 0 "${TAG}_body_e30" 0.30 &
P0=$!
run_arm 1 "${TAG}_body_e60" 0.60 &
P1=$!
wait $P0 $P1
echo "[v5] $(date +%T) 结束：runs/${TAG}_body_e30, runs/${TAG}_body_e60"
