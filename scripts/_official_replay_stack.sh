#!/usr/bin/env bash
# 跑官方 sim_main replay，卡住时用 SIGABRT 打 Python 栈（PYTHONFAULTHANDLER=1）
set -u
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
PROJ=$C/third_party/HumanoidArena/isaaclab_twist2_g1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PYTHONFAULTHANDLER=1
export PROJECT_ROOT=$PROJ
export PYTHONPATH="$C/third_party/unitree_sdk2_python:$PROJ:$C:$C/trackerlab-supermimic/source/trackerTask:$C/trackerlab-supermimic/source/trackerLab"
export SUPERMIMIC_HA_ALIGN=1
P=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release
NPZ=$C/data/ha_raw/HOI_double_desk/tw/Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody_sonic_1776268633456387.npz
LOG=/tmp/off3.log
cd "$PROJ" || exit 1
PY=$C/third_party/isaacsim_env/bin/python
$PY -u sim_main.py \
  --device cuda:0 \
  --env_config_yaml tasks/common_env_config/doubledesk_sonic.yaml \
  --task Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody \
  --robot_type g129 --input_source replay --gmt_backend sonic \
  --sonic_encoder_path "$P/model_encoder.onnx" \
  --sonic_decoder_path "$P/model_decoder.onnx" \
  --replay_file "$NPZ" --replay_mode direct_replay \
  --image_transport zmq --image_fps 30 --image_zmq_port 5555 \
  --seed 42 --enable_cameras --headless --exit_when_replay_complete \
  > "$LOG" 2>&1 &
PID=$!
echo "pid=$PID"
for i in $(seq 1 60); do
  sleep 5
  if grep -q "SonicActionProvider.*POSE mode ready" "$LOG" 2>/dev/null; then
    echo "provider ready at ~$((i*5))s"; break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "process exited at ~$((i*5))s"; break; fi
done
if kill -0 "$PID" 2>/dev/null; then
  echo "--- still alive, dumping stack via SIGABRT at ~$((i*5))s ---"
  kill -ABRT "$PID"
  sleep 8
fi
wait "$PID" 2>/dev/null
echo "=== markers ==="
grep -n "create action provider\|POSE mode ready\|Replay joint err\|Started\|Traceback\|File \"" "$LOG" | tail -40
