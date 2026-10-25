#!/usr/bin/env bash
# =============================================================================
# 2026-10-25：从 step 2000 带 token anchor 续训到 4000（TOKEN_ANCHOR.md §6.2 的 B 方案，扩到 4000）
#
# 卡数安排（本容器 comimic4，4 × 49 GB）：
#   卡 0,1 = chain1（A2 四臂 → A5 两臂，跑完 ≈09:15）
#   卡 2,3 = chain2（A3-BPTT 120 min，跑完 ≈07:56）
#   → 本脚本**等 chain2 结束后**用**单卡（默认卡 2）**跑 anchor 续训；卡 3 留空给 A1 的 Isaac 闭环
#     （Isaac Sim 需要独占一张卡，之前 A1 base 臂就是因为没卡被 Terminated）。
#
# 为什么单卡 / NGPU=1：
#   主线 A1（runs/joint_ha_a1）的 args.json 实测 `_world_size=1 / batch_size=2 / prefetch_vla=True /
#   state_vel_weight=1.0`。本跑**完全同形**（唯一差别 = --token-anchor-weight 1.0），
#   这样 3000/4000 两步的 token_mse / rollout_reward 与主线曲线是**单变量**可比。
#
# 产物隔离（绝不覆盖主线）：
#   runs/joint_ha_anchor_2000to4000/{joint,vla}_00{3000,4000}.pt
#   logs/joint_ha_anchor_2000to4000.log
#   2000 的两个存档用**硬链接**放进新目录（RESUME_STEP 要求它们位于 $OUT 下），
#   这样即使主线目录里的文件被归档/移动，本跑仍能续训。
#
# 用法：
#   setsid nohup bash scripts/launch_anchor_1025.sh > logs/anchor_queue.out 2>&1 &
#   TAG=xxx GPU=3 ANCHOR=3.0 STEPS=4000 setsid nohup bash scripts/launch_anchor_1025.sh ...
# =============================================================================
set -uo pipefail

REPO=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
SRC="$REPO/runs/joint_ha_a1"
TAG="${TAG:-joint_ha_anchor_2000to4000}"
GPU="${GPU:-2}"
VEL="${VEL:-1.0}"
ANCHOR="${ANCHOR:-1.0}"
RESUME_STEP="${RESUME_STEP:-2000}"
STEPS="${STEPS:-4000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
PORT="${PORT:-29672}"
OUT="$REPO/runs/$TAG"
LOGW="$REPO/logs/${TAG}.queue.log"
mkdir -p "$OUT" "$REPO/logs"

say () { echo "[anchor $(date -Is)] $*" | tee -a "$LOGW"; }

step6=$(printf '%06d' "$RESUME_STEP")
for f in "joint_${step6}.pt" "vla_${step6}.pt"; do
  [[ -f "$SRC/$f" ]] || { say "ABORT 缺续训存档 $SRC/$f"; exit 1; }
  if [[ ! -e "$OUT/$f" ]]; then
    ln "$SRC/$f" "$OUT/$f" 2>/dev/null || cp -n "$SRC/$f" "$OUT/$f" || { say "ABORT 无法放置 $f"; exit 1; }
  fi
done
say "resume artifacts ready in $OUT (hardlink->$SRC)"

if pgrep -f "launch_joint_ha.sh" > /dev/null 2>&1; then
  say "WARN 已有 launch_joint_ha.sh 在跑（可能是主线 A1）；本跑仍会启动，只写自己的 OUT 目录"
fi

say "waiting for chain2 (A3-BPTT on GPUs 2,3) to finish before taking GPU $GPU"
while pgrep -f "run_abl_1025_queue.sh chain2" > /dev/null 2>&1; do sleep 60; done
sleep 20
say "chain2 gone; launching on GPU $GPU (NGPU=1, batch 2, prefetch on, vel=$VEL, anchor=$ANCHOR, steps=$STEPS, eval_every=$EVAL_EVERY)"

RUN="$TAG" OUT="$OUT" \
  RESUME_STEP="$RESUME_STEP" STEPS="$STEPS" NGPU=1 PORT="$PORT" \
  CUDA_VISIBLE_DEVICES="$GPU" \
  STATE_VEL_WEIGHT="$VEL" TOKEN_ANCHOR_WEIGHT="$ANCHOR" PREFETCH_VLA=1 \
  EVAL_EVERY="$EVAL_EVERY" \
  setsid nohup bash "$REPO/scripts/launch_joint_ha.sh" > "$REPO/logs/${TAG}.launcher.log" 2>&1 &
say "started (pid=$!) log=$REPO/logs/${TAG}.launcher.log"
say "watch: grep -E '\\[anchor\\]|eval ' $REPO/logs/${TAG}.log"
