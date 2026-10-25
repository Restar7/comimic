#!/usr/bin/env bash
# Wait for the pure-HA joint finetune to finish, then run the A1 IsaacLab ablation.
#
# This is the unattended hand-off used for the card-6 A1 experiment:
#   train_joint.py (pure HA, 4000 steps)  ->  vla_004000.pt
#   ->  export_vla_ckpt.py (merge into the same HA base)
#   ->  run_a1_ha_isaac.sh (base vs ft, HumanoidArena 6 tasks)
#
# It never touches mixed/ft318/MuJoCo (guards live in the entry points).
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
ROOT=$COMIMIC
cd "$ROOT"

TRAIN_PID=${TRAIN_PID:-1926047}
TRAIN_VLA=${TRAIN_VLA:-$ROOT/runs/joint_ha_a1/vla_004000.pt}
SM_CKPT=${SM_CKPT:-$ROOT/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500.pt}
SEEDS=${SEEDS:-"0 1 2"}
REPEATS=${REPEATS:-1}
MAX_STEPS=${MAX_STEPS:-300}
OUT=${OUT:-$ROOT/runs/ablations/A1_joint_finetune_ha}

say() { echo "[$(date +%H:%M:%S)] $*"; }
say "waiting for training pid=$TRAIN_PID and $TRAIN_VLA"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
say "training process exited"
if [[ ! -f "$TRAIN_VLA" ]]; then
  latest=$(ls -t "$ROOT"/runs/joint_ha_a1/vla_*.pt 2>/dev/null | head -1 || true)
  [[ -n "$latest" ]] || { say "[fail] training exited without any vla_*.pt"; exit 1; }
  say "[warn] $TRAIN_VLA missing; fallback to $latest"
  TRAIN_VLA=$latest
fi
# give the training process time to flush its last eval/save and release GPU memory
sleep 30
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
say "starting A1: update=$TRAIN_VLA seeds=[$SEEDS] repeats=$REPEATS max_steps=$MAX_STEPS"
JOINT_VLA="$TRAIN_VLA" SEEDS="$SEEDS" REPEATS="$REPEATS" MAX_STEPS="$MAX_STEPS" \
  SM_CKPT="$SM_CKPT" OUT="$OUT" \
  bash "$ROOT/scripts/run_a1_ha_isaac.sh"
rc=$?
say "A1 finished rc=$rc -> $OUT/SUMMARY.md"
exit "$rc"
