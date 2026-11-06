#!/usr/bin/env bash
# Wait for R1 pipeline to finish, then run the official base vs R1 full-length eval.
set -uo pipefail
COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1
LOG=logs/full_r1_eval.log
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T') wait R1 pipeline ==="
while pgrep -f "bash scripts/run_full_r1.sh" >/dev/null; do sleep 30; done
[[ -f runs/S3_loop/901_joint/train_main/vla_final.pt ]] || { echo "[fail] no R1 vla_final"; exit 1; }
echo "=== $(date '+%F %T') start base vs R1 full eval ==="
HEAD_MAIN=base \
HEAD_CTRL="$COMIMIC/runs/S3_loop/901_joint/train_main/vla_final.pt" \
SM_CKPT_MAIN="$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt" \
SM_CKPT_CTRL="$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt" \
LABEL_MAIN=base LABEL_CTRL=full_r1 \
TAG_PREFIX=base_vs_full_r1 \
START_FRAME=0 MAX_STEPS=2080 SEED_TOTAL=32 N_SWEEPS=4 BASE_PORT=18250 GPU_OFFSET=0 \
bash scripts/eval_h20_8gpu.sh
rc=$?
echo "=== $(date '+%F %T') eval rc=$rc ==="
exit $rc
