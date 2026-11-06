#!/usr/bin/env bash
# Automatic R2 -> R5 ladder.  It waits for the R1-fix v2 gate supervisor to
# write 901fix_best.txt, then runs each round with the v2 calibrated table.
# Any failed round stops the ladder; the failure marker is left for the agent.
set -uo pipefail
COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1
LOG=logs/auto_ladder_r2r5.log
exec > >(tee -a "$LOG") 2>&1
say(){ echo "[ladder $(date '+%F %T')] $*"; }

BEST=${BEST:-runs/S3_loop/901fix_best.txt}
FAIL=${FAIL:-runs/S3_loop/901fix_gate_fail.txt}
for i in $(seq 1 300); do
  [[ -s "$BEST" ]] && break
  if [[ -s "$FAIL" ]]; then
    say "R1-fix gate failed: $(cat "$FAIL") -> ladder stops"
    exit 1
  fi
  sleep 20
done
[[ -s "$BEST" ]] || { say "timeout waiting for R1-fix best head"; exit 1; }
PREV_HEAD=$(head -1 "$BEST")
say "R1-fix best head = $PREV_HEAD"

# N, E_START, e_h, lambda, cap, act_weight, steps, eval_n, seed
ROUNDS=(
  "2 1.2 0.485 0.2474 1.165 0.8 200 32 2"
  "3 1.4 0.564 0.8000 1.327 0.6 200 32 3"
  "4 1.6 0.632 0.7000 1.490 0.5 200 32 4"
  "5 1.8 0.679 0.6500 1.637 0.5 300 48 5"
)
PREV_ROUND=901
for row in "${ROUNDS[@]}"; do
  read -r N E EH LAM CAP ACTW STEPS EVALN SEED <<< "$row"
  ROUND=$((900 + N))
  say "===== start R$N (round=$ROUND) E=$E e_h=$EH lam=$LAM cap=$CAP act=$ACTW steps=$STEPS eval=$EVALN ====="
  ROUND="$ROUND" N="$N" PREV_ROUND="$PREV_ROUND" PREV_HEAD="$PREV_HEAD" \
  E_START="$E" EH="$EH" LAM="$LAM" CAP="$CAP" ACT_W="$ACTW" STEPS="$STEPS" EVAL_N="$EVALN" SEED="$SEED" \
  bash scripts/run_full_round_auto.sh
  rc=$?
  if [[ $rc != 0 ]]; then
    say "R$N FAILED (rc=$rc) -> ladder stops; fix this round before continuing"
    exit $rc
  fi
  NEXT=$(cat "runs/S3_loop/auto_round${ROUND}_head.txt" 2>/dev/null || true)
  [[ -s "$NEXT" ]] || { say "R$N produced no head pointer"; exit 1; }
  PREV_HEAD=$NEXT
  PREV_ROUND=$ROUND
  say "R$N PASS; next start head=$PREV_HEAD"
done
say "ALL DONE: R2-R5 complete"
echo "all_done" > runs/S3_loop/auto_ladder_done.txt
