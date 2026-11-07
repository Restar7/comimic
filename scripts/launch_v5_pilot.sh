#!/usr/bin/env bash
# v5 Pilot（**cached_pool** 口径，只为分离因素；报告必须标 data_mode=cached_pool，不得与主训练混表）
#
# 目的（按审阅意见）：固定真实 VLA chunk（不重采），跑 2×2 + 常数模板 的对照：
#   p0_old   : 旧动作 trust（两轨迹差） + 无 raw 范数约束         → 复现"坍缩"
#   p1_trust : 同观测动作 trust        + 无 raw 范数约束
#   p2_cap   : 旧动作 trust            + raw 范数 hinge（κ=2）
#   p3_both  : 同观测动作 trust        + raw 范数 hinge（κ=2）
#   p4_postpl: 逐帧常量模板（不看输入） → "学成常量"的防伪对照
#
# 固定项：cap=0.08 固定（**不做 warmup**，按审阅意见先做可学习性试验）、batch=256、H=20、
#        updates=200、λ_act=0.010、λ_smooth=0.002、λ_anchor=0.001、λ_raw=1e-3。
set -euo pipefail
WT=${WT:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03}
# shellcheck disable=SC1091
source "$WT/scripts/harness03_env.sh"
BIG=${BIG:-$COMIMIC_MAIN/runs/harness_03_bptt/big}
OUT=${OUT:-$WT/runs/temporal_ac_bptt_v5/pilot}
UPDATES=${UPDATES:-200}
mkdir -p "$OUT"
cd "$WT"
COMMON=(--windows "$BIG/windows.pt" --vla-tokens "$BIG/vla_tokens.pt"
        --updates "$UPDATES" --batch 256 --horizon 20 --eval-batch 48
        --eval-every 50 --cond-check-every 25 --log-every 10
        --cap-warmup-updates 0 --device cuda:0)

launch() {  # name gpu extra...
  local name=$1 gpu=$2; shift 2
  echo "[pilot] $name -> gpu$gpu  $*"
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/train_temporal_ac_bptt_v5.py \
      --out "$OUT/$name" "${COMMON[@]}" "$@" > "$OUT/$name.log" 2>&1 &
}

launch p0_old     0 --trust-mode traj     --lambda-raw 0
launch p1_trust   0 --trust-mode same_obs --lambda-raw 0
launch p2_cap     0 --trust-mode traj     --lambda-raw 1e-3
launch p3_both    1 --trust-mode same_obs --lambda-raw 1e-3
launch p4_postpl  1 --actor-kind pos_template --trust-mode same_obs --lambda-raw 0

echo "[pilot] waiting..."
code=0
for p in $(jobs -p); do wait "$p" || code=$?
done
echo "[pilot] done code=$code"
for d in "$OUT"/p*; do
  [ -f "$d/test_once.json" ] && echo "--- $(basename "$d")" && cat "$d/test_once.json"
done
exit $code
