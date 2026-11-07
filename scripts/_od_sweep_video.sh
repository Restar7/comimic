#!/usr/bin/env bash
# OpenDoor 成功率配对扫描（**带录像**）：seed 0-5 × {v13, base}，MAX_STEPS=1500。
# 与之前 od_sweep 的唯一区别：RECORD_VIDEO=1（更接近真机连续相机）。
set -uo pipefail
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic || exit 1
OUT=runs/opendoor_sweep_video.csv
if [[ ! -f $OUT ]]; then echo "seed,arm,success,steps,sec,disp_m,at10s" > $OUT; fi

for s in 0 1 2 3 4 5; do
  for arm in v13 base; do
    if [[ "$arm" == "v13" ]]; then H=runs/joint_v13/vla_final.pt; else H=none; fi
    if [[ "$arm" == "v13" ]]; then port=$((18500 + s * 2)); else port=$((18501 + s * 2)); fi
    TAG=odv_${arm}_s${s}
    echo "[sweepv] $(date +%H:%M:%S) seed=$s arm=$arm port=$port" | tee -a runs/_odv_sweep.log
    TASK=OpenDoor TAG=$TAG HEAD=$H GPU=0 PORT=$port SEED=$s MAX_STEPS=1500 RECORD_VIDEO=1 \
      bash scripts/r3_closedloop_ab.sh > "runs/_${TAG}.log" 2>&1
    python3 - "$s" "$arm" "runs/closedloop_r3/$TAG" >> "$OUT" <<'PY'
import glob, json, os, sys
seed, arm, d = sys.argv[1], sys.argv[2], sys.argv[3]
r = glob.glob(os.path.join(d, 'OpenDoor__*', 'result.json'))
if not r:
    print(f"{seed},{arm},NA,NA,NA,NA,NA"); raise SystemExit
j = json.load(open(r[0]))
disp = at10 = float('nan')
try:
    import numpy as np
    st = [json.loads(l) for l in open(os.path.join(d, 'dump', 'steps.jsonl'))]
    P = np.array([x['root_pos'] for x in st])
    disp = float(np.linalg.norm(P[-1, :2] - P[0, :2]))
    at10 = float(np.linalg.norm(P[min(500, len(P) - 1), :2] - P[0, :2]))
except Exception:
    pass
print(f"{seed},{arm},{int(bool(j.get('success')))},{j.get('episode_steps')},"
      f"{float(j.get('episode_steps', 0))/50:.1f},{disp:.3f},{at10:.3f}")
PY
    tail -1 "$OUT" | tee -a runs/_odv_sweep.log
  done
done
echo "[sweepv] $(date +%H:%M:%S) DONE" | tee -a runs/_odv_sweep.log
