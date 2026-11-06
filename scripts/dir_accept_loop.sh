#!/usr/bin/env bash
# 方向验收闭环（2026-11-05 定案，2026-11-06 从 /tmp 搬进 scripts/）：
#   闭环跑一条 → 取 FT 相对 base 的实际动作位移方向 → 真机 ±δ 各滚 20 步 → 打分
#
# 为什么要有它：漂移/奖励/想象的指标都能被"薅"，但"策略实现出来的动作方向在真物理里
# 是不是那条好方向"没法被薅。判据：≥65% 同号 = 方向对了；<50% = 方向仍错。
#
# 用法：
#   bash scripts/dir_accept_loop.sh <FT_TAG> <FT_HEAD_CKPT> [GPU] [PORT]
# 产物：
#   runs/closedloop_r3/<FT_TAG>_s0/   闭环 dump（方向从这里取）
#   runs/dir_accept/job_<FT_TAG>.pt   真机探针 job（±δ 两串动作）
#   runs/dir_accept/roll_<FT_TAG>.pt  真机滚出来的状态
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$C" || exit 1
TAG=${1:?usage: dir_accept_loop.sh FT_TAG FT_HEAD [GPU] [PORT]}
HEAD=${2:?}
GPU=${3:-0}
PORT=${4:-18300}
WORK=${WORK:-$C/runs/dir_accept}
mkdir -p "$WORK"
export PYTHONPATH=$C SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=$C/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd

echo "=== [dir] 1) 闭环一条 $(date +%H:%M:%S)"
TAG="${TAG}_s0" HEAD="$HEAD" GPU="$GPU" PORT="$PORT" SEED=0 MAX_STEPS=1000 \
  SRV_EXTRA="--hand-from-base" bash scripts/r3_closedloop_ab.sh \
  > "runs/_cl_dir_${TAG}.log" 2>&1 || true
echo "=== [dir] 2) 取方向 $(date +%H:%M:%S)"
python3 scripts/dir_accept_build_job.py base_s0 "${TAG}_s0" "$WORK/job_${TAG}.pt" 2>&1 | tail -3
echo "=== [dir] 3) 真机 ±δ $(date +%H:%M:%S)"
bash scripts/dir_accept_run_probe.sh "$WORK/job_${TAG}.pt" "$WORK/roll_${TAG}.pt" 2>&1 | tail -2
echo "=== [dir] 4) 打分 $(date +%H:%M:%S)"
python3 scripts/dir_accept_score.py "$WORK/job_${TAG}.pt" "$WORK/roll_${TAG}.pt" 2>&1 | tail -6
