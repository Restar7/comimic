#!/usr/bin/env bash
# 全量 A' 采集（2026-11-03 修正版：**按任务分组**）
#
# 关键修正：699 条 clip 分属 7 个任务，必须各自在**自己的场景**里跑。
#   之前把全部 clip 塞进 DoubleDesk 场景 => 非本任务片段段内漂移 p50 = 23.7cm（爆表）。
#   本 checkout 里 GrapCup 任务缺失（未注册 + 无资产）=> 跳过 99 条。
#
# 每条 clip：windows 模式（含尾窗口，全长覆盖）+ --segment 20（每 20 步重置到参考）
#            + --action-noise-std 0.05（环内保持 20 步）
#
# 用法：
#   bash scripts/run_a_prime_full_seg20.sh
#   TASKS="DoubleDesk" bash scripts/run_a_prime_full_seg20.sh
#   SMOKE=1 TASKS="DoubleDesk" bash scripts/run_a_prime_full_seg20.sh
set -uo pipefail

C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
OUT_DIR=${OUT_DIR:-$C/runs/a_prime_seg20}
WINDOW=${WINDOW:-270}
SEG=${SEG:-20}
NOISE=${NOISE:-0.05}
HOLD=${HOLD:-20}
PER=${PER:-16}
WPG=${WPG:-1}          # 每个 GPU 几个 worker（2×8env 与 16env 显存相近，吞吐翻倍）
GPUS=${GPUS:-"0 1 2 3"}
TASKS=${TASKS:-"DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi"}

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PROJECT_ROOT=$C/third_party/HumanoidArena/isaaclab_twist2_g1
export PYTHONPATH="$C/third_party/unitree_sdk2_python:$PROJECT_ROOT:$C:$C/trackerlab-supermimic/source/trackerTask:$C/trackerlab-supermimic/source/trackerLab:${PYTHONPATH:-}"
export SUPERMIMIC_HA_ALIGN=1 HA_ALLOW_SONIC=1
mkdir -p "$OUT_DIR"
cd "$C" || exit 1

task_spec() {
  case "$1" in
    DoubleDesk)   echo "0 100 Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody tasks/common_test_config/base_test/doubledesk_sonic_test.yaml" ;;
    GrapCup)      echo "SKIP" ;;
    PPBox)        echo "199 299 Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby tasks/common_test_config/base_test/pp_box_sonic_test.yaml" ;;
    Boxing)       echo "299 399 Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody tasks/common_test_config/base_test/boxing_sonic_test.yaml" ;;
    OpenDoor)     echo "399 499 Isaac-Move-Open-Door-G129-Dex3-Wholebody tasks/common_test_config/base_test/open_door_sonic_test.yaml" ;;
    SitSofa)      echo "499 599 Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody tasks/common_test_config/base_test/sit_sofa_sonic_test.yaml" ;;
    VisionNavi)   echo "599 699 Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody tasks/common_test_config/base_test/vision_navi_sonic_test.yaml" ;;
    *)            echo "SKIP" ;;
  esac
}

n_windows() {
  python3 - "$C" "$WINDOW" "$1" "$2" <<'PY'
import sys, torch
c, win, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
pk = torch.load(f"{c}/runs/isaac_motion/ha_cmd_exec/packed.pt", map_location="cpu", weights_only=False)
tot = 0
for L in list(pk["lengths"])[a:b]:
    L = int(L)
    tot += len(range(0, max(1, L - win), win)) + (1 if L > win else 0)
print(tot)
PY
}

run_one() {
  local gpu=$1 off=$2 cs=$3 ce=$4 tid=$5 yml=$6 tag=$7
  local outp="$OUT_DIR/${tag}_off${off}.pt"
  if [[ -f "$outp" && "${FORCE:-0}" != "1" ]]; then
    echo "[$tag gpu$gpu] offset=$off 已有产物，跳过"; return 0
  fi
  CUDA_VISIBLE_DEVICES=$gpu timeout 300 "$C/third_party/isaacsim_env/bin/python" -u \
    scripts/isaac_a_collect_batch.py \
    --enable_cameras --headless \
    --wbc-ckpt runs/S1_handover/exp_WBC_exec_haonly_best.pt \
    --packed runs/isaac_motion/ha_cmd_exec/packed.pt \
    --tokens-dir runs/exp_tokens --token-source npy \
    --steps "$WINDOW" --segment "$SEG" --start-mode windows \
    --clip-start "$cs" --clip-end "$ce" \
    --num-envs "$PER" --plan-offset "$off" --seed "$off" \
    --executor wbc --action-noise-std "$NOISE" --perturb-hold "$HOLD" \
    --ha-task "$tid" --ha-yaml "$yml" --log-every 100 \
    --out "$outp" > "$OUT_DIR/${tag}_off${off}.log" 2>&1
  local rc=$?
  echo "[$tag gpu$gpu] offset=$off done rc=$rc"
  return 0
}

for T in $TASKS; do
  spec=$(task_spec "$T")
  if [[ "$spec" == "SKIP" ]]; then echo "[skip] $T（本 checkout 不支持）"; continue; fi
  read -r CS CE TASK_ID TASK_YAML <<< "$spec"
  TOTAL=$(n_windows "$CS" "$CE")
  echo "=== [task] $T clips=[$CS,$CE) task=$TASK_ID windows=$TOTAL ==="

  if [[ "${SMOKE:-0}" == "1" ]]; then
    PER=2
    run_one "${GPUS%% *}" 0 "$CS" "$CE" "$TASK_ID" "$TASK_YAML" "smoke_${T}"
    continue
  fi

  pids=()
  _ngpu=$(echo $GPUS | wc -w)
  _stride=$((_ngpu * WPG * PER))
  for g in $GPUS; do
    for w in $(seq 0 $((WPG - 1))); do
      (
        off=$(( (g * WPG + w) * PER ))
        while [[ $off -lt $TOTAL ]]; do
          if [[ $((off + PER)) -gt $TOTAL ]]; then break; fi
          run_one "$g" "$off" "$CS" "$CE" "$TASK_ID" "$TASK_YAML" "$T"
          off=$((off + _stride))
        done
      ) &
      pids+=($!)
    done
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "=== [task] $T 完成 ==="
done
echo "[driver] 全部任务完成"
ls -1 "$OUT_DIR"/*.pt 2>/dev/null | wc -l
