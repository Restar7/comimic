#!/usr/bin/env bash
# 关节序修正 A/B：回放录制目标（SONIC 序 vs ckpt 序）
set -u
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PROJECT_ROOT=$C/third_party/HumanoidArena/isaaclab_twist2_g1
export PYTHONPATH="$C/third_party/unitree_sdk2_python:$PROJECT_ROOT:$C:$C/trackerlab-supermimic/source/trackerTask:$C/trackerlab-supermimic/source/trackerLab:${PYTHONPATH:-}"
export SUPERMIMIC_HA_ALIGN=1
export SM_DUMP_JOINTS=1
MODE=${1:-sonic}
OUT=${2:-/tmp/replay_$MODE.log}
NPZ=$C/data/ha_raw/HOI_double_desk/tw/Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody_sonic_1776268633456387.npz
cd $C || exit 1
SM_REPLAY_IDS=$MODE $C/third_party/isaacsim_env/bin/python -u scripts/isaac_a_collect_batch.py \
  --enable_cameras --headless \
  --wbc-ckpt runs/S1_handover/exp_WBC_exec_haonly_best.pt \
  --packed runs/isaac_motion/ha_cmd_exec/packed.pt \
  --tokens-dir runs/exp_tokens --token-source npy \
  --num-envs 1 --steps 600 --start-mode first --clip-index 0 --warmup 0 \
  --executor sonic --replay-target-npz "$NPZ" \
  --ha-task Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody \
  --ha-yaml tasks/common_test_config/base_test/doubledesk_sonic_test.yaml \
  --seed 0 --out runs/calib/replay_$MODE.pt > "$OUT" 2>&1
echo "exit=$? log=$OUT"
