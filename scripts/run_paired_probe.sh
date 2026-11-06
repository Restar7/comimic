#!/usr/bin/env bash
# **配对探针采集**（2026-11-05）：同一个状态 → 64 个不同动作 → 64 个不同下一状态
#
# 为什么需要它（实测依据）
#   A' 里"同状态、不同动作"的配对只有 **2 / 60,000**（状态最近邻距离 p50 = 4.86）
#   ⇒ **动作增益 ∂s'/∂a 在 A' 里根本不可辨识**，这是"WM 动作通道不因果"的根源。
#   本脚本用 `--plan-repeat`（所有 env 共用同一窗口）+ `--segment 20`：
#   每段重置后 64 个 env 状态**完全相同**，动作只因噪声不同 ⇒ 天然真配对。
#
# 用法：
#   bash scripts/run_paired_probe.sh                 # 默认 3 任务 × 3 波
#   SMOKE=1 bash scripts/run_paired_probe.sh         # 8 env × 60 步（验证链路）
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$C" || exit 1
OUT_DIR=${OUT_DIR:-$C/runs/paired_probe}
ENVS=${ENVS:-64}
STEPS=${STEPS:-780}
SEG=${SEG:-20}
NOISE=${NOISE:-0.25}
KREP=${KREP:-4}
GPUS=${GPUS:-"0 1 2 3"}
WAVES=${WAVES:-3}
TASKS=${TASKS:-"DoubleDesk PPBox Boxing"}
DECODER=${DECODER:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx}

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PROJECT_ROOT=$C/third_party/HumanoidArena/isaaclab_twist2_g1
export PYTHONPATH="$C/third_party/unitree_sdk2_python:$PROJECT_ROOT:$C:$C/trackerlab-supermimic/source/trackerTask:$C/trackerlab-supermimic/source/trackerLab:${PYTHONPATH:-}"
export SUPERMIMIC_HA_ALIGN=1 HA_ALLOW_SONIC=1
mkdir -p "$OUT_DIR"

task_spec() {
  case "$1" in
    DoubleDesk) echo "0 100 Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody tasks/common_test_config/base_test/doubledesk_sonic_test.yaml" ;;
    PPBox)      echo "199 299 Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby tasks/common_test_config/base_test/pp_box_sonic_test.yaml" ;;
    Boxing)     echo "299 399 Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody tasks/common_test_config/base_test/boxing_sonic_test.yaml" ;;
    OpenDoor)   echo "399 499 Isaac-Move-Open-Door-G129-Dex3-Wholebody tasks/common_test_config/base_test/open_door_sonic_test.yaml" ;;
    SitSofa)    echo "499 599 Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody tasks/common_test_config/base_test/sit_sofa_sonic_test.yaml" ;;
    VisionNavi) echo "599 699 Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody tasks/common_test_config/base_test/vision_navi_sonic_test.yaml" ;;
    *)          echo "SKIP" ;;
  esac
}

run_one() {  # gpu offset tag cs ce tid yml
  local gpu=$1 off=$2 tag=$3 cs=$4 ce=$5 tid=$6 yml=$7
  local outp="$OUT_DIR/${tag}_off${off}.pt"
  [[ -f "$outp" && "${FORCE:-0}" != "1" ]] && { echo "[probe $tag off=$off] 已存在，跳过"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout 900 "$C/third_party/isaacsim_env/bin/python" -u \
    scripts/isaac_a_collect_batch.py \
    --enable_cameras --headless --executor sonic --sonic-decoder "$DECODER" \
    --wbc-ckpt runs/S1_handover/exp_WBC_exec_haonly_best.pt \
    --packed runs/isaac_motion/ha_cmd_exec/packed.pt \
    --tokens-dir runs/exp_tokens --token-source npy \
    --steps "$STEPS" --len-min 260 --segment "$SEG" --start-mode windows \
    --reset-max-drift 0 --exec-steps 1 \
    --clip-start "$cs" --clip-end "$ce" \
    --num-envs "$ENVS" --plan-offset "$off" --plan-repeat-k "$KREP" --seed 0 \
    --action-noise-std "$NOISE" --perturb-hold 1 \
    --ha-task "$tid" --ha-yaml "$yml" --log-every 1000 \
    --out "$outp" > "$OUT_DIR/${tag}_off${off}.log" 2>&1
  if [[ -f "$outp" ]]; then
    echo "[probe $tag gpu$gpu off=$off] 产物 OK"
  else
    echo "[probe $tag gpu$gpu off=$off] **没有产物**" >&2
  fi
  return 0
}

if [[ "${SMOKE:-0}" == "1" ]]; then
  ENVS=8 STEPS=60
  read -r CS CE TID YML <<< "$(task_spec DoubleDesk)"
  run_one "${GPUS%% *}" 0 "smoke" "$CS" "$CE" "$TID" "$YML"
  exit 0
fi

pids=()
for T in $TASKS; do
  spec=$(task_spec "$T")
  if [[ "$spec" == "SKIP" ]]; then echo "[skip] $T"; continue; fi
  read -r CS CE TID YML <<< "$spec"
  # 每波 4 个 launch，offset 取不同窗口（同一窗口只采一次就够，多波是为了覆盖更多状态）
  for w in $(seq 0 $(( WAVES - 1 ))); do
    base=$(( w * 4 * (ENVS / KREP) ))
    i=0
    for g in $GPUS; do
      off=$(( base + i ))
      run_one "$g" "$off" "$T" "$CS" "$CE" "$TID" "$YML" &
      pids+=($!)
      i=$(( i + 1 ))
    done
    wait
  done
done
wait
echo "[probe] 完成 → $OUT_DIR"
