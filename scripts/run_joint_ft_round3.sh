#!/usr/bin/env bash
# 联合微调 第 3 轮（2026-11-04，**A 臂：λ_bc=0**）
#
# 与第 1/2 轮的本质区别：**裁判（WM）第一次是真的准**
#   旧 WM：20 步 root 漂移 想象/真实 = 4.18 倍，corr 0.205
#   新 WM：0.99 倍，corr 0.988，终点误差 1.45 cm（runs/a_prime_sonic_leash/wm_ssm_s0_more.pt）
# 因此这一轮才第一次有意义地检验"奖励项能不能推动策略"。
#
# A 臂 = 无 token BC（λ_bc=0）：token 64 维只由 20 步 imagined 回报 + act 信任域 + 预训练锚约束；
#         手部仍由 hand-only FM 监督（λ_hand=0.05，掩码只有 64:78）。
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
DIR=${DIR:-$C/runs/a_prime_sonic_leash}
cd "$C" || exit 1
for f in "$DIR/filtered_std.pt"; do
  [[ -f "$f" ]] || { echo "[r3] 缺 $f" >&2; exit 2; }
done

export V3=1 H=20
# ---- 裁判：本次验收通过的 WM（20 步 imag/real = 0.99）----
export WM_CKPT=${WM_CKPT:-$DIR/wm_ssm_s0_more.pt}
# ---- 数据/起点：A' 真实漂移状态，|ξ| ≤ 10 cm 封顶 ----
export EXTRA="${EXTRA:---real-transitions $DIR/filtered_std.pt}"
export START_SOURCE=recorded START_MAX_DRIFT=0.10
# ---- 奖励口径：huber_err + split_huber + 死区 0.30（>30cm 不硬拉回参考）----
export REWARD_MODE=huber_err ROOT_ORI_MODE=split_huber
export REWARD_DELTA_POS=0.05 REWARD_DELTA_ORI=0.05 REWARD_D_MAX=0.30
export REWARD_VEL_SCALE=0
# ---- 损失权重（A 臂：λ_bc=0）----
# ★ 奖励量级换算：saturating 的每步奖励是 O(1)，而 huber_err（δ=0.05）只有 O(1e-4~1e-3)
#   ⇒ 同一份 λ_r 下奖励项会缩水 3~4 个数量级。这里把 λ_r 抬到 5（等价"旧口径的 0.3"量级），
#   并以 "λ_r × |return| 与 loss/act 同量级" 作为调参判据（监控 loss/reward_norm 与 loss/act）。
export LAMBDA_R=${LAMBDA_R:-5.0} LAMBDA_BC=0.0 LAMBDA_HAND=0.05 TOKEN_BC=0
export ACT_EPS=0.50 ACT_WEIGHT=0.20
export GAMMA=0.95
# ---- critic 本轮关（纯 imagined 20 步回报）----
export CRITIC_ON=0
# ---- 预训练锚（防遗忘）：贴"关掉 LoRA 的预训练策略" ----
export TRUST_CENTER=base
export TRUST_WEIGHT=${TRUST_WEIGHT:-0.05}
export TRUST_EPS=${TRUST_EPS:-0.15}
export ADV_BASELINE=none ADV_TAU=0
# ---- 规模 ----
export RUN=${RUN:-joint_v3_round3A} STEPS=${STEPS:-3000}
export EVAL_EVERY=${EVAL_EVERY:-100} SAVE_EVERY=${SAVE_EVERY:-500}
export DDP=1 NPROC=${NPROC:-4} DDP_GPUS=${DDP_GPUS:-0,1,2,3}
echo "[r3A] WM=$WM_CKPT"
echo "[r3A] λ_bc=$LAMBDA_BC λ_r=$LAMBDA_R λ_hand=$LAMBDA_HAND act=$ACT_WEIGHT/$ACT_EPS γ=$GAMMA"
echo "[r3A] reward=$REWARD_MODE/$ROOT_ORI_MODE δ=$REWARD_DELTA_POS dmax=$REWARD_D_MAX start=$START_SOURCE/$START_MAX_DRIFT"
exec bash scripts/launch_joint_ha_v2.sh
