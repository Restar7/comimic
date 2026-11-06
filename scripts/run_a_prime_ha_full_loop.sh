#!/usr/bin/env bash
# 完整闭环推理（不分段、全长）：6 个 HA 任务族 × 100 条 clip，16 env/进程，4 卡分片。
#
# 每个 (任务, 波次) = 一个进程：从该族第 wave*16 条 clip 开始，取 16 条，跑满该族最长长度。
# 波次间并行在 4 张卡上（每卡 16 env，带相机时这是实测上限）。
set -uo pipefail
C=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$C" || exit 1
H=$C/third_party/HumanoidArena/isaaclab_twist2_g1
source "$C/scripts/isaac_env.sh" >/dev/null 2>&1
export PROJECT_ROOT=$H
export SUPERMIMIC_HA_ALIGN=1
export PYTHONPATH="$C/third_party/unitree_sdk2_python:$H:$C:$C/trackerlab-supermimic/source/trackerTask:$C/trackerlab-supermimic/source/trackerLab:${PYTHONPATH:-}"

WBC=${WBC:-$C/runs/S1_handover/exp_WBC_exec_haonly_best.pt}
PACKED=${PACKED:-$C/runs/isaac_motion/ha_cmd_exec/packed.pt}
NOISE=${NOISE:-0.05}
ENVS=${ENVS:-16}
OUT=${OUT:-$C/runs/wm_real/exp_a_prime_ha}
mkdir -p "$OUT"

# 任务: 起始索引 步数 yaml
TASKS=(
  "DoubleDesk|Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody|0|2080|doubledesk_sonic_test.yaml"
  "PPBox|Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby|199|1917|pp_box_sonic_test.yaml"
  "Boxing|Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody|299|1564|boxing_sonic_test.yaml"
  "OpenDoor|Isaac-Move-Open-Door-G129-Dex3-Wholebody|399|2069|open_door_sonic_test.yaml"
  "SitSofa|Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody|499|1087|sit_sofa_sonic_test.yaml"
  "VisionNavi|Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody|599|3343|vision_navi_sonic_test.yaml"
)

run_one() {  # task taskid base steps yaml wave gpu
  local name=$1 tid=$2 base=$3 steps=$4 yaml=$5 wave=$6 gpu=$7
  local start=$(( base + wave * ENVS ))
  local tag="${name}_w${wave}"
  local out="$OUT/${tag}.pt"
  [[ -s "$out" ]] && { echo "[full] skip $tag"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu "$ISAAC_VENV/bin/python" -u scripts/isaac_a_collect_batch.py \
    --enable_cameras --wbc-ckpt "$WBC" --packed "$PACKED" --tokens-dir "$C/runs/exp_tokens" \
    --num-envs "$ENVS" --steps "$steps" --clip-start "$start" --start-mode first --headless \
    --log-every 2000 --token-source npy --executor wbc \
    --action-noise-std "$NOISE" --perturb-hold 20 \
    --ha-task "$tid" --ha-yaml "tasks/common_test_config/base_test/$yaml" \
    --seed "$wave" --out "$out" > "$OUT/${tag}.log" 2>&1
  echo "[full] $tag gpu=$gpu rc=$? $(ls -la "$out" 2>/dev/null | awk '{print $5}')"
}

i=0
pids=()
for spec in "${TASKS[@]}"; do
  IFS='|' read -r name tid base steps yaml <<<"$spec"
  for wave in 0 1 2 3 4 5 6; do
    run_one "$name" "$tid" "$base" "$steps" "$yaml" "$wave" "$(( i % 4 ))" &
    pids+=($!)
    i=$((i+1))
    [[ $(( i % 4 )) -eq 0 ]] && wait
  done
done
wait
echo "[full] 全部完成，产物在 $OUT"
