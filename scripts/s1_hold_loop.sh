#!/usr/bin/env bash
# A'' 臂：hold + lr1e-5 连续训练，每 500 步一次官方 HA-699 抽检；按 norm_return 判平台
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
R=$C/trackerlab-supermimic; cd $R
export SONIC_ENCODER_ONNX=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../third_party/trackerlab_env/bin/python
RESUME_0=$1; START=$2; END=$3; PORT=${4:-29770}
prev=$RESUME_0; it=$START; best=0; stale=0
while [ "$it" -lt "$END" ]; do
  nxt=$((it+500)); LOG=logs/wbc_hold_loop_$nxt
  RESUME=$prev OBS_MODE=paper_obs93 FREEZE_WM=0 NO_CRITIC=1 WM_ONLY_ITERS=0 LR=1e-5 \
  MODE=train NPROC=2 ENVS_PER_RANK=8192 ITERATIONS=$nxt MAX_EPISODE_STEPS=0 \
  LIBRARY=$R/data/bones_seed/bones_seed_plus_ha_cmd_x20.pt INTERFACE=token COMPILE=1 HA_ALIGN=1 \
  CKPT_EVERY=250 PORT_BASE=$PORT LOG_DIR=$LOG bash scripts/rls/supermimic/train_bones_seed.sh \
    > $C/logs/s1/wbc_hold_loop_$nxt.log 2>&1 || { echo "[loop] train $nxt FAILED"; break; }
  CK=$LOG/ckpt_0$nxt.pt; [ -f "$CK" ] || { echo "[loop] no ckpt $nxt"; break; }
  $PY scripts/rls/supermimic/evaluate.py --run_dir $LOG --ckpt "$CK" \
    --motion_packed $C/runs/isaac_motion/ha_tokens/packed.pt --device cuda:0 --kit_args=--allow-root \
    > $C/logs/s1/eval_loop_$nxt.log 2>&1
  NR=$($PY -c "import json;d=json.load(open('$LOG/ckpt_0${nxt}_eval.json'));print(round(d['norm_return'],4),d['n_failed'])" 2>/dev/null || echo "NA NA")
  cp $LOG/ckpt_0${nxt}_eval.json $C/runs/S1_gates/loop_HA699_$nxt.json 2>/dev/null
  echo "[loop] it=$nxt norm_return/failed = $NR (baseline 0.7094/0)" | tee -a $C/runs/S1_gates/LOOP_SUMMARY.txt
  NEW=$(echo $NR | cut -d' ' -f1)
  if awk "BEGIN{exit !($NEW > $best + 0.005)}"; then best=$NEW; stale=0; else stale=$((stale+1)); fi
  [ "$stale" -ge 3 ] && { echo "[loop] PLATEAU at $nxt (best=$best)" | tee -a $C/runs/S1_gates/LOOP_SUMMARY.txt; break; }
  prev=$LOG/last.pt; it=$nxt; PORT=$((PORT+1))
done
echo "[loop] DONE" | tee -a $C/runs/S1_gates/LOOP_SUMMARY.txt
