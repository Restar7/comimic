#!/usr/bin/env bash
# =============================================================================
# 2026-10-25 4 卡排期（本容器 comimic4：4 × 49 GB，卡 0,1,2,3 全空）
#
# 口径（与 docs/HANDOVER_MAINLINE_AND_ABLATIONS_20261025.md 一致，别改）：
#   SM_CKPT = trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt
#   数据 = data/npz/ha_tokens（纯 HA 699），packed = runs/isaac_motion/ha_tokens
#   --state-vel-weight 0 / a=Δq / 端口 token 条件 / VLM backbone 冻结
#   每臂固定墙钟：A2 30 min、A3 120 min、A5 40 min（同臂内可比）
#
#   chain1（卡 0,1）：A2 四后端 pathwise → checkpointed → truncated → adjoint
#                    然后 A5 两臂 head_only → head_plus_wbc
#   chain2（卡 2,3）：A3 BPTT 臂 120 min（PPO 臂已在 05:29 起跑，两臂同墙钟）
#
# 与旧结果的隔离（§8 旧口径作废、不覆盖旧目录）：
#   A2 -> runs/ablations/A2_fm_backend_v2_physint/<backend>
#   A5 -> runs/ablations/A5_only_head_v2_physint/<arm>
#   A3 -> runs/ablations/A3_bptt_vs_ppo/bptt   （与在跑的 ppo 同目录成对）
#   日志 -> logs/run_1025_v2/
#
# 用法：
#   setsid nohup bash scripts/run_abl_1025_queue.sh chain1 > logs/run_1025_v2/chain1.out 2>&1 &
#   setsid nohup bash scripts/run_abl_1025_queue.sh chain2 > logs/run_1025_v2/chain2.out 2>&1 &
#   RETRIES=1：每臂失败只重试 1 次（默认 2 次总尝试）
# =============================================================================
set -uo pipefail

REPO=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
CHAIN="${1:-}"
case "$CHAIN" in
  chain1|chain2) ;;
  *) echo "usage: bash scripts/run_abl_1025_queue.sh chain1|chain2"; exit 2;;
esac

OUT_A2="$REPO/runs/ablations/A2_fm_backend_v2_physint"
OUT_A5="$REPO/runs/ablations/A5_only_head_v2_physint"
OUT_A3="$REPO/runs/ablations"
LOGD="$REPO/logs/run_1025_v2"
QLOG="$LOGD/queue_$CHAIN.log"
RETRIES="${RETRIES:-2}"
mkdir -p "$LOGD" "$OUT_A2" "$OUT_A5"

say () { echo "[queue $CHAIN] $(date -Is) $*" | tee -a "$QLOG"; }

run_with_retry () {   # run_with_retry <tag> <cmd...>
  local tag="$1"; shift
  local n=1
  while (( n <= RETRIES )); do
    say "START $tag (attempt $n/$RETRIES)"
    if "$@" ; then say "OK    $tag (attempt $n)"; return 0; fi
    say "FAIL  $tag (attempt $n)"
    (( n < RETRIES )) && sleep 30
    n=$((n+1))
  done
  say "GIVEUP $tag (after $RETRIES attempts)"
  return 1
}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false

if [[ "$CHAIN" == "chain1" ]]; then
  # ---------------------------------------------------------------- 卡 0,1: A2 + A5
  export CUDA_VISIBLE_DEVICES=0,1
  say "chain1 on GPUs 0,1 : A2(4x30min) -> A5(2x40min)"
  for b in pathwise checkpointed truncated adjoint; do
    run_with_retry "a2_$b" \
      env CUDA_VISIBLE_DEVICES=0,1 OUT_ROOT="$OUT_A2" LOG_DIR="$LOGD" \
          BACKENDS="$b" MINUTES=30 BATCH=2 HORIZON=8 \
          bash "$REPO/scripts/launch_a2_fm.sh" \
      || { say "A2 中断于 $b，跳过后续 A2 臂，直接进 A5"; break; }
  done
  for a in head_only head_plus_wbc; do
    run_with_retry "a5_$a" \
      env CUDA_VISIBLE_DEVICES=0,1 OUT_ROOT="$OUT_A5" LOG_DIR="$LOGD" \
          ARMS="$a" MINUTES=40 BATCH=2 HORIZON=8 \
          bash "$REPO/scripts/launch_a5_only_head.sh" \
      || { say "A5 中断于 $a"; break; }
  done
  say "chain1 DONE"
else
  # ---------------------------------------------------------------- 卡 2,3: A3 BPTT
  export CUDA_VISIBLE_DEVICES=2,3
  say "chain2 on GPUs 2,3 : A3 BPTT 120 min"
  run_with_retry "a3_bptt" \
    env CUDA_VISIBLE_DEVICES=2,3 OUT_ROOT="$OUT_A3" LOG_DIR="$LOGD" \
        MODE=bptt MINUTES=120 BATCH=2 HORIZON=8 \
        bash "$REPO/scripts/launch_a3_a5.sh" \
    || say "A3 BPTT 失败（2 次尝试）"
  say "chain2 DONE"
fi
