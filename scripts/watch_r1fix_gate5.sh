#!/usr/bin/env bash
# Supervisor for the R1-fix v2 gate.
# Waits for 901fix_joint training, then runs:
#   16-pair base-vs-R1fix screen -> if promising, 32-pair formal R1 gate.
# Writes runs/S3_loop/901fix5_best.txt on pass, 901fix5_gate_fail.txt on fail.
set -uo pipefail
COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1
TAG=901fix5_joint
HEAD=$COMIMIC/runs/S3_loop/$TAG/train_main/vla_final.pt
LOG=logs/r1fix5_gate_supervisor.log
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T') supervisor start; waiting for $HEAD ==="

# Wait for training process and final head
for i in $(seq 1 180); do
  if [[ -s "$HEAD" ]] && ! pgrep -f "train_joint_v2.py .*$TAG/train_main" >/dev/null 2>&1; then
    break
  fi
  sleep 20
done
if [[ ! -s "$HEAD" ]]; then
  echo "[fail] no $HEAD after wait"
  echo "no_head" > runs/S3_loop/901fix5_gate_fail.txt
  exit 1
fi
echo "=== $(date '+%F %T') head ready: $HEAD ==="
sleep 5

run_eval () {
  local n=$1 tag=$2 port=$3
  HEAD_MAIN=base HEAD_CTRL="$HEAD" \
  SM_CKPT_MAIN="$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt" \
  SM_CKPT_CTRL="$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt" \
  LABEL_MAIN=base LABEL_CTRL=r1fix5 \
  TAG_PREFIX="$tag" START_FRAME=0 MAX_STEPS=2080 \
  SEED_TOTAL="$n" N_SWEEPS=4 BASE_PORT="$port" GPU_OFFSET=0 \
  bash scripts/eval_h20_8gpu.sh
}

echo "=== $(date '+%F %T') 16-pair screen ==="
run_eval 16 base_vs_r1fix5_16 18520
rc16=$?
echo "[supervisor] 16-pair rc=$rc16"
M16=runs/S3_loop/sweeps/base_vs_r1fix5_16_merged.json
if [[ $rc16 != 0 || ! -f "$M16" ]]; then
  echo "[fail] 16-pair eval missing/failed"
  echo "eval16_failed" > runs/S3_loop/901fix5_gate_fail.txt
  exit 1
fi
read -r m16 b16 ci16 < <(python3 - "$M16" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); p=d["paired"]
print(p["mean"], p["better"], p["ci95_boot"][1])
PY
)
echo "[supervisor] 16-pair mean=$m16 better=$b16 ci_hi=$ci16"
# Screen: candidate must at least not be worse on the mean and win >=9/16.
python3 - "$m16" "$b16" <<'PY' || { echo "screen_failed" > runs/S3_loop/901fix5_gate_fail.txt; exit 1; }
import sys
m=float(sys.argv[1]); b=int(sys.argv[2])
ok = (m < 0) and (b >= 9)
print(f"[screen] mean={m:+.4f} better={b}/16 -> {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
PY

echo "=== $(date '+%F %T') 32-pair formal gate ==="
run_eval 32 base_vs_r1fix5_32 18540
rc32=$?
echo "[supervisor] 32-pair rc=$rc32"
M32=runs/S3_loop/sweeps/base_vs_r1fix5_32_merged.json
if [[ $rc32 != 0 || ! -f "$M32" ]]; then
  echo "[fail] 32-pair eval missing/failed"
  echo "eval32_failed" > runs/S3_loop/901fix5_gate_fail.txt
  exit 1
fi
python3 - "$M32" "$HEAD" <<'PY'
import json, sys, pathlib
d=json.load(open(sys.argv[1])); p=d["paired"]
m, lo, hi, win, n = p["mean"], p["ci95_boot"][0], p["ci95_boot"][1], p["better"], p["n"]
ok = (m < 0) and (hi < 0) and (win >= n * 0.6)
print(f"[gate] mean={m:+.4f} CI=[{lo:+.4f},{hi:+.4f}] better={win}/{n} -> {'PASS' if ok else 'FAIL'}")
pathlib.Path("runs/S3_loop/901fix5_best.txt").write_text(sys.argv[2] + "\n")
if not ok:
    pathlib.Path("runs/S3_loop/901fix5_gate_fail.txt").write_text(
        f"gate_failed mean={m} ci=[{lo},{hi}] better={win}/{n}\n")
raise SystemExit(0 if ok else 1)
PY
rc=$?
if [[ $rc == 0 ]]; then
  echo "=== $(date '+%F %T') R1-fix v2 GATE PASS ==="
  exit 0
fi
echo "=== $(date '+%F %T') R1-fix v2 GATE FAIL ==="
exit 1
