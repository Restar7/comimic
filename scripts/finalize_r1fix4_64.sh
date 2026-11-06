#!/usr/bin/env bash
# Wait for the 4 extra 64x sweeps, merge with the existing 32full sweeps,
# apply the R1 gate, and (on pass) start the R2-R5 ladder.
set -uo pipefail
COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1
LOG=logs/finalize_r1fix4_64.log
exec > >(tee -a "$LOG") 2>&1
say(){ echo "[finalize64 $(date '+%F %T')] $*"; }

SW=(runs/S3_loop/sweeps/base_vs_r1fix4_32full_s0 runs/S3_loop/sweeps/base_vs_r1fix4_32full_s1 runs/S3_loop/sweeps/base_vs_r1fix4_32full_s2 runs/S3_loop/sweeps/base_vs_r1fix4_32full_s3
    runs/S3_loop/sweeps/base_vs_r1fix4_64x_s0 runs/S3_loop/sweeps/base_vs_r1fix4_64x_s1 runs/S3_loop/sweeps/base_vs_r1fix4_64x_s2 runs/S3_loop/sweeps/base_vs_r1fix4_64x_s3)
for i in $(seq 1 180); do
  ok=1
  for d in "${SW[@]}"; do [[ -f "$d/summary.json" ]] || { ok=0; break; }; done
  (( ok == 1 )) && break
  sleep 20
done
for d in "${SW[@]}"; do
  [[ -f "$d/summary.json" ]] || { say "missing $d/summary.json -> fail"; echo "missing_summary" > runs/S3_loop/901fix4_gate_fail.txt; exit 1; }
done
M=runs/S3_loop/sweeps/base_vs_r1fix4_64merged.json
say "merging 8 sweeps -> $M"
python3 scripts/merge_eval_sweeps.py "${SW[@]}" --out "$M" --label r1fix4_minus_base || { echo "merge_failed" > runs/S3_loop/901fix4_gate_fail.txt; exit 1; }
python3 - "$M" <<'PY' || { echo "gate_failed" > runs/S3_loop/901fix4_gate_fail.txt; exit 1; }
import json,sys
d=json.load(open(sys.argv[1])); p=d["paired"]
m,lo,hi,win,n=p["mean"],p["ci95_boot"][0],p["ci95_boot"][1],p["better"],p["n"]
med=p.get("median")
ok=(m<0 and hi<0 and win>=0.5*n)
print(f"[gate64] mean={m:+.4f} med={med} CI=[{lo:+.4f},{hi:+.4f}] better={win}/{n} -> {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
PY
H=$COMIMIC/runs/S3_loop/901fix4_partial/train_main/vla_final.pt
[[ -s "$H" ]] || { echo "missing_head" > runs/S3_loop/901fix4_gate_fail.txt; exit 1; }
echo "$H" > runs/S3_loop/901fix4_best.txt
rm -f runs/S3_loop/901fix4_gate_fail.txt
say "R1=fix4 PASS; best=$H; starting R2-R5 ladder"
BEST=runs/S3_loop/901fix4_best.txt FAIL=runs/S3_loop/901fix4_gate_fail.txt bash scripts/auto_ladder_r2r5.sh
