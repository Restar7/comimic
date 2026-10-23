#!/usr/bin/env bash
# Bring up the SONIC MuJoCo stack for PICO data collection (x86 workstation).
#
#   sim  : MuJoCo + ego_view publisher on tcp://0.0.0.0:5555
#   ctrl : SONIC controller, reads /dev/shm state, publishes g1_debug on 5557
#
# Both processes use CycloneDDS over loopback with an explicit unicast peer,
# because loopback is not multicast-capable and the vendored CycloneDDS crashes
# when asked to exchange type objects over a multicast interface.
set -uo pipefail

REPO=/home/wusichen/GR00T-WholeBodyControl
DEPLOY=$REPO/gear_sonic_deploy
SIMPY=/home/wusichen/groot_repro/simvenv/bin/python
MODELS=/home/wusichen/groot_repro/sonic_models
LOGDIR=/tmp/sonic_stack
mkdir -p "$LOGDIR"

export CYCLONEDDS_URI=file:///tmp/cyclone.xml
export MUJOCO_GL=glfw
export DISPLAY=${DISPLAY:-:1}
export PYTHONPATH=$REPO:$REPO/external_dependencies/unitree_sdk2_python

stop_all() {
  for f in "$LOGDIR"/*.pid; do
    [ -f "$f" ] || continue
    kill -9 "$(cat "$f")" 2>/dev/null
    rm -f "$f"
  done
  # image publisher child of a dead sim can hold 5555
  local holder
  holder=$(ss -ltnp 2>/dev/null | awk '/:5555 /{print $NF}' | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "${holder:-}" ] && kill -9 "$holder" 2>/dev/null
  sleep 2
}

start_sim() {
  setsid nohup "$SIMPY" "$REPO/gear_sonic/scripts/run_sim_loop.py" \
    --enable-image-publish --enable-offscreen --no-enable-onscreen \
    --camera-port 5555 > "$LOGDIR/sim.log" 2>&1 < /dev/null &
  echo $! > "$LOGDIR/sim.pid"
  sleep 12
}

start_ctrl() {
  ( cd "$DEPLOY" && \
    LD_LIBRARY_PATH=/home/wusichen/groot_repro/trt_root/lib:/tmp/ort/onnxruntime-linux-x64-1.19.2/lib \
    setsid nohup ./target/release/g1_deploy_onnx_ref lo \
      "$MODELS/model_decoder.onnx" ./reference/example/ \
      --obs-config "$MODELS/observation_config.yaml" \
      --encoder-file "$MODELS/model_encoder.onnx" \
      --planner-file "$MODELS/planner/target_vel/V2/planner_sonic.onnx" \
      --input-type zmq_manager --output-type zmq --zmq-host localhost \
      --disable-crc-check > "$LOGDIR/ctrl.log" 2>&1 < /dev/null &
    echo $! > "$LOGDIR/ctrl.pid" )
  sleep 20
}

case "${1:-start}" in
  stop) stop_all ;;
  start)
    stop_all
    start_sim
    start_ctrl
    echo "--- sim ---";  tail -3 "$LOGDIR/sim.log"
    echo "--- ctrl ---"; tail -4 "$LOGDIR/ctrl.log"
    echo "--- ports ---"; ss -ltn 2>/dev/null | grep -E '5555|5556|5557' || true
    ;;
esac
