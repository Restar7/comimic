#!/usr/bin/env bash
# 用法: run_probe.sh JOB OUT
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$C" || exit 1
source scripts/isaac_env.sh
OMP_NUM_THREADS=8 timeout 2400 "$ISAAC_VENV/bin/python" -u scripts/isaac_probe_action_rollout.py \
  --job "$1" --out "$2" --wbc-ckpt runs/S1_handover/WBC_A24000_wmvel_cmd.pt \
  --num-envs 48 --device cuda:0 --headless
