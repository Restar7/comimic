#!/usr/bin/env bash
# 等 A3/A5 全部跑完 → 自动出报告（含 A5 冻结审计 + 曲线）→ 写 FINALIZED 标记。
# 用法：setsid nohup bash scripts/finalize_a3_a5.sh > logs/a3a5_finalize.log 2>&1 &
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC"

echo "[finalize] waiting for the A3/A5 series to finish ($(date -Is))"
while pgrep -f "launch_a3_a5.sh" > /dev/null 2>&1; do
  sleep 60
done
echo "[finalize] series finished ($(date -Is))"

/usr/bin/python3 scripts/a3_a5_report.py --audit-a5 \
  --out docs/ABLATION_A3_A5_CARDS23.md 2>&1 | tail -20

{
  echo "finalized_at=$(date -Is)"
  echo "code_snapshot=runs/_code_a3a5_20261024"
  echo "code_manifest_sha256=$(sha256sum runs/_code_a3a5_20261024/CODE_MANIFEST.sha256 | cut -d' ' -f1)"
  for f in runs/ablations/A3_bptt_vs_ppo/bptt runs/ablations/A3_bptt_vs_ppo/ppo \
           runs/ablations/A5_only_head/head_only runs/ablations/A5_only_head/head_plus_wbc; do
    printf '%s evals=%s steps=%s\n' "$f" \
      "$(wc -l < "$f/eval_curve.jsonl" 2>/dev/null || echo 0)" \
      "$(tail -1 "$f/eval_curve.jsonl" 2>/dev/null | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("step","?"))' 2>/dev/null || echo '?')"
  done
} > runs/ablations/FINALIZED.txt

echo "[finalize] done -> runs/ablations/FINALIZED.txt"
cat runs/ablations/FINALIZED.txt
