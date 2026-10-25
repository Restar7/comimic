#!/usr/bin/env bash
# =============================================================================
# A 线（从 0 就带 token 锚定）接力：等卡 2 上的 0→2000 跑完 → merge 出可加载 ckpt →
# 继续 2000→8000（带同一个锚定 recipe + 平台哨兵），卡 2 不空转。
#
# 设计要点（与 B 续训保持一致，便于并排比较）：
#   * 同 recipe：λ=1.0 / vel=1.0 / NGPU=1 / batch 2 / prefetch on / EVAL_EVERY=250
#   * 同"到平台为止"：STEPS=8000 只是上限，真正的停止条件是 plateau_guard 的判据
#     （最近 3 个 eval 点不再把最好 token_mse 压低 ≥0.002，且 reward 极差 ≤0.0025）
#   * MERGED_CKPT 显式给（否则默认 ckpt_vla_ha_ft2000 是最早主线那份、内容不对）
#
# 用法：setsid nohup bash scripts/queue_anchor_from0_phase2.sh > logs/anchor_from0_phase2.out 2>&1 &
# =============================================================================
set -uo pipefail
REPO=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
SRC="$REPO/runs/joint_ha_anchor_from0"
TAG=joint_ha_anchor_from0_2000to8000
OUT="$REPO/runs/$TAG"
MERGED="$REPO/runs/ckpt_vla_ha_anchor2000_from0"
LOGD="$REPO/logs"
QLOG="$LOGD/$TAG.queue.log"
PIDFILE="/tmp/${TAG}.pid"
mkdir -p "$OUT" "$LOGD"
say () { echo "[phase2 $(date -Is)] $*" | tee -a "$QLOG"; }

say "等待卡 2 上的 0→2000 跑完（$SRC/vla_final.pt）"
while true; do
  if [[ -f "$SRC/vla_final.pt" || -f "$SRC/vla_002000.pt" ]]; then break; fi
  if ! pgrep -f -- "--out $SRC --prefetch-vla" > /dev/null 2>&1; then
    say "[warn] 进程已不在且没有 2000 存档；放弃接力"
    exit 1
  fi
  sleep 60
done
sleep 20
for f in joint_002000.pt vla_002000.pt; do
  [[ -f "$SRC/$f" ]] || { say "ABORT 缺 $SRC/$f"; exit 1; }
  [[ -e "$OUT/$f" ]] || ln "$SRC/$f" "$OUT/$f" 2>/dev/null || cp -n "$SRC/$f" "$OUT/$f"
done
say "2000 存档就绪；merge vla_002000 -> $MERGED"
/usr/bin/python3 "$REPO/scripts/export_vla_ckpt.py" --base "$REPO/runs/ckpt_vla_ha_8000_local" \
  --vla "$SRC/vla_002000.pt" --out "$MERGED" --force 2>&1 | tail -2 | tee -a "$QLOG" || { say "ABORT merge 失败"; exit 1; }

say "起跑 2000→8000（上限），卡 2"
RUN="$TAG" OUT="$OUT" RESUME_STEP=2000 STEPS=8000 NGPU=1 PORT=29676 \
  CUDA_VISIBLE_DEVICES=2 MERGED_CKPT="$MERGED" \
  STATE_VEL_WEIGHT=1.0 TOKEN_ANCHOR_WEIGHT=1.0 PREFETCH_VLA=1 EVAL_EVERY=250 \
  setsid nohup bash "$REPO/scripts/launch_joint_ha.sh" > "$LOGD/$TAG.launcher.log" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
sleep 5
say "pid=$PID -> $PIDFILE；启动平台哨兵"
setsid nohup bash "$REPO/scripts/plateau_guard.sh" "$PID" \
  "$OUT/eval_curve.jsonl" "$LOGD/$TAG.guard.log" > /dev/null 2>&1 &
say "done（A 线 2000→平台 已在跑；B 线 4000→平台 在卡 3 并行）"
