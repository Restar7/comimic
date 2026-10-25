#!/usr/bin/env bash
# 等 A5 两臂（scripts/launch_a5_only_head.sh）跑完 → 冻结审计 + 报告 + FINALIZED 标记。
#
# 用法（先起训练，再起它）：
#   setsid nohup bash runs/_code_a5_20261024/scripts/launch_a5_only_head.sh > logs/a5_master.log 2>&1 &
#   setsid nohup bash runs/_code_a5_20261024/scripts/finalize_a5.sh            > logs/a5_finalize.log 2>&1 &
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC"
MASTER_LOG="${MASTER_LOG:-logs/a5_master.log}"
ROOT="${ROOT:-runs/ablations/A5_only_head}"

echo "[finalize] waiting for the A5 series ($(date -Is))"
sleep 60
CRASHED=0
while true; do
  if grep -q "^\[done\] arms=" "$MASTER_LOG" 2>/dev/null; then break; fi
  if ! pgrep -f "launch_a5_only_head.sh" > /dev/null 2>&1; then
    echo "[finalize] launcher 已不在但没有 [done] —— 视为异常退出，仍出报告"; CRASHED=1; break
  fi
  sleep 30
done
echo "[finalize] series finished ($(date -Is)) crashed=$CRASHED"

/usr/bin/python3 scripts/a5_report.py --root "$ROOT" --out docs/ABLATION_A5_RUN.md 2>&1 | tail -6

FINAL="$ROOT/A5_FINALIZED.txt"
{
  echo "finalized_at=$(date -Is)"
  echo "crashed=$CRASHED"
  echo "code_snapshot=runs/_code_a5_20261024"
  echo "code_manifest_sha256=$(sha256sum runs/_code_a5_20261024/CODE_MANIFEST.sha256 2>/dev/null | cut -d' ' -f1)"
  echo "--- launcher markers ---"
  grep -E "^\[(gate|config|arm|done)" "$MASTER_LOG" 2>/dev/null | tail -12
  echo "--- per arm ---"
  for a in head_only head_plus_wbc; do
    [[ -d "$ROOT/$a" ]] || continue
    step=$(tail -1 "$ROOT/$a/eval_curve.jsonl" 2>/dev/null | /usr/bin/python3 -c 'import json,sys
try: print(json.loads(sys.stdin.read()).get("step","?"))
except Exception: print("?")' 2>/dev/null)
    printf '%s evals=%s last_step=%s joint=%s\n' "$a" \
      "$(wc -l < "$ROOT/$a/eval_curve.jsonl" 2>/dev/null || echo 0)" "${step:-?}" \
      "$(ls -s --block-size=1M "$ROOT/$a/joint_latest.pt" 2>/dev/null | cut -d' ' -f1)MB"
  done
} > "$FINAL"

echo "[finalize] done -> $FINAL"
cat "$FINAL"
