#!/usr/bin/env bash
# P1 轮：在 v13 基础上只改"目标/方差/起点/视界"，**不改输入、不改架构**。
#
#   ✅ 开 on-policy critic（SuperMimic 同款，现场拟合，无需 ckpt）
#   ✅ HA_AT_CENTER=1（去掉"学常向量"的退化解）
#   ✅ 起点集合 leash → **exc**（`all_transitions.pt`，529,131 行）；cap 0.05 → 0.10（保留 79%）
#   ✅ 有效 batch 24 → 48（单卡 BATCH=8 × ACCUM=6）
#   ✅ BOOT_ONE_SIDED（rel := min(rel,0)）：R2 实测 rel 恒正 = 在给"偏离"发钱
#   ✅ H 保持 20：**exc 窗口上**误差 p50 = 0.26/0.39/0.49/0.82 cm（k=5/8/10/20）
#      ⇒ 误差随 k 亚线性（k^0.83）⇒ SNR 随 k 变好（1.18 → 1.50）⇒ H=20 最优
#      （leash 上是超线性 k^1.34，那是分布外的症状，不能拿来定 H）
#   ⛔ action_target_steps 保持 20（= v13 原值，与 H 一致；"答案偏差"这件事先交给实测
#      diverge/pos_t* 去判断，不再拍脑袋改成 6）
#   ⛔ 不动：γ=0.99、state 输入、LoRA rank
#
# 注意（口径坑）：round5 里的 `export GAMMA=0.95` **是死变量**——launch_joint_ha_v2.sh 不传 --gamma，
#   实际 γ 取 train 默认 0.99。args.json 记的也是 0.99。handover 文档写 0.95 是错的。
#
# 用法：
#   bash scripts/run_p1_round.sh                 # 从 v13 续训，300 步
#   STEPS=200 RUN=joint_p1b bash scripts/run_p1_round.sh
set -uo pipefail
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic || exit 1

export DIR=runs/a_prime_sonic_exc        # 本轮起点集合 = exc
# ★ 必须显式设：round5 的默认是 $DIR/wm_exc.pt，而所有验证/探针用的是 wm_gain2.pt
export WM_CKPT=${WM_CKPT:-runs/a_prime_sonic_exc/wm_gain2.pt}
export TERM_OFF=${TERM_OFF:-1}          # v13 也是 1；关掉软终止头（它在新 WM 上误触发，会砍半回报）
export RUN=${RUN:-joint_p1}
export STEPS=${STEPS:-300}
export NPROC=${NPROC:-1}
export DDP=${DDP:-0}
export DDP_GPUS=${DDP_GPUS:-0}
export BATCH=${BATCH:-8}                  # 单卡内存与 3 卡×8 同量级
export ACCUM=${ACCUM:-6}                 # 有效 batch = 48

# ---- 本轮改动（共 6 项）----
export CRITIC_ON=${CRITIC_ON:-1}
export CRITIC_SOURCE=${CRITIC_SOURCE:-imagined}
export HA_AT_CENTER=${HA_AT_CENTER:-1}
export START_MAX_DRIFT=${START_MAX_DRIFT:-0.10}   # exc 的 p90=12.5cm，0.10 保留 ~85%

# ---- 其余与 v13 完全一致 ----
export ACT_PENALTY=${ACT_PENALTY:-l2}
export ACT_EPS=${ACT_EPS:-0.30}
export ACT_WEIGHT=${ACT_WEIGHT:-0.2}
export TRUST_WEIGHT=${TRUST_WEIGHT:-0.2}
export TRUST_EPS=${TRUST_EPS:-0.05}
export REWARD_HORIZON=${REWARD_HORIZON:-0}
# ★ 必须显式设：launcher 里 ACTION_TARGET_WEIGHT 默认是 0（整项关掉），round5 不设
export ACTION_TARGET_WEIGHT=${ACTION_TARGET_WEIGHT:-1.0}
export ACTION_TARGET_ETA=${ACTION_TARGET_ETA:-0.15}
export ACTION_TARGET_STEPS=${ACTION_TARGET_STEPS:-20}  # 与 v13/H 一致；是否该缩到 6 看 diverge/pos_t* 实测
export H=${H:-20}                                      # exc 上误差亚线性，H 越大 SNR 越好（见头部注释）
export SAVE_EVERY=${SAVE_EVERY:-50}
export EVAL_EVERY=${EVAL_EVERY:-50}

# ★ 2026-11-06：起点集合从 leash 换成 exc —— 两条判据都指向它：
#   ① 起点偏差 p50 5.85cm / p90 12.5cm（leash 是 3.93/8.24），更接近部署实测 4.9–17.4cm；
#   ② WM 在 exc 上 20 步误差 0.82cm（leash 上 2.11cm）⇒ SNR 1.5 vs 0.58。
#   合并产物：runs/a_prime_sonic_exc/all_transitions.pt（50 分片，529,131 行）
export EXTRA="--real-transitions runs/a_prime_sonic_exc/all_transitions.pt --boot-one-sided"
export INIT_VLA=runs/joint_v13/vla_final.pt

echo "[p1] RUN=$RUN STEPS=$STEPS NPROC=$NPROC BATCH=$BATCH ACCUM=$ACCUM (有效 $((BATCH*ACCUM)))"
echo "[p1] critic=$CRITIC_ON/$CRITIC_SOURCE  center=$HA_AT_CENTER  start_max_drift=$START_MAX_DRIFT"
echo "[p1] init-vla=$INIT_VLA"

exec bash scripts/run_joint_ft_round5.sh
