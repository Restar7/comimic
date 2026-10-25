#!/usr/bin/env bash
# 等 A2 四臂（scripts/launch_a2_fm.sh）全部跑完 → 自动出报告 + 写 FINALIZED 标记。
#
# 用法（先起训练，再起它；训练结束后自动接力）：
#   setsid nohup bash scripts/launch_a2_fm.sh   > logs/a2_master.log   2>&1 &
#   setsid nohup bash scripts/finalize_a2_fm.sh > logs/a2_finalize.log 2>&1 &
#
# 只依赖文件系统：不 import torch，不占卡。
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC"
MASTER_LOG="${MASTER_LOG:-logs/a2_master.log}"
ROOT="${ROOT:-runs/ablations/A2_fm_backend}"

echo "[finalize] waiting for the A2 series ($(date -Is))"
echo "[finalize] master log: $MASTER_LOG"
sleep 60   # 给 launcher 一点启动时间，避免刚起就判"已结束"

CRASHED=0
while true; do
  if grep -q "^\[done\] backends=" "$MASTER_LOG" 2>/dev/null; then
    break
  fi
  if ! pgrep -f "launch_a2_fm.sh" > /dev/null 2>&1; then
    echo "[finalize] launcher 进程已不在，但 master log 没有 [done] —— 视为异常退出，仍出报告"
    CRASHED=1
    break
  fi
  sleep 30
done
echo "[finalize] series finished ($(date -Is)) crashed=$CRASHED"

/usr/bin/python3 scripts/a2_fm_report.py --root "$ROOT" --out docs/ABLATION_A2_RUN.md 2>&1 | tail -5

FINAL="$ROOT/A2_FINALIZED.txt"
{
  echo "finalized_at=$(date -Is)"
  echo "crashed=$CRASHED"
  echo "code_snapshot=runs/_code_a2_20261024"
  echo "code_manifest_sha256=$(sha256sum runs/_code_a2_20261024/CODE_MANIFEST.sha256 2>/dev/null | cut -d' ' -f1)"
  echo "master_log=$MASTER_LOG"
  echo "--- launcher markers ---"
  grep -E "^\[(gate|config|arm|done)" "$MASTER_LOG" 2>/dev/null | tail -20
  echo "--- per arm ---"
  for f in "$ROOT"/pathwise "$ROOT"/checkpointed "$ROOT"/truncated "$ROOT"/adjoint; do
    [[ -d "$f" ]] || continue
    step=$(tail -1 "$f/eval_curve.jsonl" 2>/dev/null | /usr/bin/python3 -c 'import json,sys
try: print(json.loads(sys.stdin.read()).get("step","?"))
except Exception: print("?")' 2>/dev/null)
    printf '%s evals=%s last_step=%s\n' "$(basename "$f")" \
      "$(wc -l < "$f/eval_curve.jsonl" 2>/dev/null || echo 0)" "${step:-?}"
  done
  echo "--- artifacts ---"
  ls -lah "$ROOT"/*/ 2>/dev/null | head -40
} > "$FINAL"

echo "[finalize] done -> $FINAL"
cat "$FINAL"
