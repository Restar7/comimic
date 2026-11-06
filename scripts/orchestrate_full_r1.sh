#!/usr/bin/env bash
# Wait for the 180 full R1 dumps, then run R1 pipeline + official full-length eval.
set -uo pipefail
COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1
B=runs/B_full_r1
LOG=logs/full_r1_orchestrator.log
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T') orchestrator R1 start ==="
while :; do
  n=$(find "$B" -name '*.pt' -path '*/dump/*' 2>/dev/null | wc -l)
  echo "[$(date '+%T')] dumps=$n/180"
  [[ "$n" -ge 180 ]] && break
  if ! pgrep -f "collect_full_episodes.py --round full_r1" >/dev/null; then
    echo "[fail] collector died with only $n dumps"; exit 1
  fi
  sleep 60
done
# wait for writer to finish
while pgrep -f "collect_full_episodes.py --round full_r1" >/dev/null; do sleep 10; done
python3 - "$B" <<'PY' || exit 1
import sys,glob,torch,os
B=sys.argv[1]; files=glob.glob(B+'/*/dump/*.pt')
assert len(files)==180, len(files)
for f in files:
    d=torch.load(f,map_location='cpu',weights_only=False)
    S=d['states']; F=d['frames']
    if S.shape[0] != F.numel():
        print('BAD',f,S.shape,F.numel()); sys.exit(1)
print('[orchestrator] 180 dumps verified')
PY
# free base server / GPU
pkill -f "ha_vla_latent64_server.py.*--port 18180" 2>/dev/null || true
pkill -f "start_vla_server.sh" 2>/dev/null || true
sleep 8
[[ -d runs/B_901 ]] && { echo "[fail] runs/B_901 already exists"; exit 1; }
mv "$B" runs/B_901
echo "[orchestrator] B_901 ready; starting R1 pipeline"
bash scripts/run_full_r1.sh > logs/full_r1_pipeline.log 2>&1
rc=$?
if [[ $rc != 0 ]]; then echo "[fail] R1 pipeline rc=$rc"; exit $rc; fi
echo "[orchestrator] R1 pipeline done; official full-length eval"
HEAD_MAIN=base \
HEAD_CTRL="$COMIMIC/runs/S3_loop/901_joint/train_main/vla_final.pt" \
SM_CKPT_MAIN="$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt" \
SM_CKPT_CTRL="$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt" \
LABEL_MAIN=base LABEL_CTRL=full_r1 \
TAG_PREFIX=base_vs_full_r1 \
START_FRAME=0 MAX_STEPS=2080 SEED_TOTAL=32 N_SWEEPS=4 BASE_PORT=18250 GPU_OFFSET=0 \
bash scripts/eval_h20_8gpu.sh > logs/full_r1_eval.log 2>&1
rc=$?
if [[ $rc != 0 ]]; then echo "[fail] R1 eval rc=$rc"; exit $rc; fi
echo "[orchestrator] R1 eval done: runs/S3_loop/sweeps/base_vs_full_r1_merged.json"
echo "=== $(date '+%F %T') orchestrator R1 done ==="
