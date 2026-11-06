#!/usr/bin/env bash
# A' 采集的**覆盖率补洞**（2026-11-03）
#
# 背景：`run_a_prime_full_seg20.sh` 的窗口分片循环为了简单写成
#     while off < TOTAL; do  if off + PER > TOTAL: break;  ... done
# 于是每个任务**最后不足一批**的窗口永远不会被采（实测 6 个任务共漏 37 个窗口）。
# 另外 VisionNavi 的 off192 / off224 两个满批分片文件整个缺失（驱动被 timeout 打断后
# 没补跑，resume 逻辑只看文件是否存在）。
#
# 合计只采到 583/600 条 clip。本脚本把缺口补齐到 600/600：
#   1) 每个任务跑一次"尾批"（--allow-partial，env 数 = 实际窗口数）
#   2) 显式补跑 VisionNavi off192 / off224
#
# 用法：
#   bash scripts/run_a_prime_tail.sh              # 6 个尾批 + 2 个满批 = 8 个分片
#   TAIL_ONLY=1 bash scripts/run_a_prime_tail.sh  # 只跑尾批
set -uo pipefail

C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
OUT_DIR=${OUT_DIR:-$C/runs/a_prime_seg20}
WINDOW=${WINDOW:-270}
SEG=${SEG:-20}
NOISE=${NOISE:-0.05}
HOLD=${HOLD:-20}
PER=${PER:-16}
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
    DoubleDesk) echo "0 100 Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody tasks/common_test_config/base_test/doubledesk_sonic_test.yaml" ;;
    PPBox)      echo "199 299 Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby tasks/common_test_config/base_test/pp_box_sonic_test.yaml" ;;
    Boxing)     echo "299 399 Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody tasks/common_test_config/base_test/boxing_sonic_test.yaml" ;;
    OpenDoor)   echo "399 499 Isaac-Move-Open-Door-G129-Dex3-Wholebody tasks/common_test_config/base_test/open_door_sonic_test.yaml" ;;
    SitSofa)    echo "499 599 Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody tasks/common_test_config/base_test/sit_sofa_sonic_test.yaml" ;;
    VisionNavi) echo "599 699 Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody tasks/common_test_config/base_test/vision_navi_sonic_test.yaml" ;;
    *)          echo "SKIP" ;;
  esac
}

# 全局窗口列表长度（与 isaac_a_collect_batch.py 的 windows 枚举**逐位一致**）
n_windows() {
  python3 - "$C" "$WINDOW" "$1" "$2" <<'PY'
import sys, torch
c, steps, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
pk = torch.load(f"{c}/runs/isaac_motion/ha_cmd_exec/packed.pt", map_location="cpu", weights_only=False)
tot = 0
for L in list(pk["lengths"])[a:b]:
    L = int(L)
    if L < steps + 2:
        continue
    last = -1
    for st in range(0, max(1, L - steps), steps):
        tot += 1
        last = st
    if L > steps and last + steps < L:
        tot += 1
print(tot)
PY
}

run_one() {  # gpu off n_envs cs ce tid yml tag extra
  local gpu=$1 off=$2 nenv=$3 cs=$4 ce=$5 tid=$6 yml=$7 tag=$8 extra=${9:-}
  local outp="$OUT_DIR/${tag}_off${off}.pt"
  if [[ -f "$outp" && "${FORCE:-0}" != "1" ]]; then
    echo "[$tag gpu$gpu] offset=$off 已有产物，跳过"; return 0
  fi
  CUDA_VISIBLE_DEVICES=$gpu timeout 420 "$C/third_party/isaacsim_env/bin/python" -u \
    scripts/isaac_a_collect_batch.py \
    --enable_cameras --headless \
    --wbc-ckpt runs/S1_handover/exp_WBC_exec_haonly_best.pt \
    --packed runs/isaac_motion/ha_cmd_exec/packed.pt \
    --tokens-dir runs/exp_tokens --token-source npy \
    --steps "$WINDOW" --segment "$SEG" --start-mode windows \
    --clip-start "$cs" --clip-end "$ce" \
    --num-envs "$nenv" --plan-offset "$off" --seed "$off" $extra \
    --executor wbc --action-noise-std "$NOISE" --perturb-hold "$HOLD" \
    --ha-task "$tid" --ha-yaml "$yml" --log-every 100 \
    --out "$outp" > "$OUT_DIR/${tag}_off${off}.log" 2>&1
  local rc=$?
  echo "[$tag gpu$gpu] offset=$off nenv=$nenv done rc=$rc"
  return 0
}

# 先把所有要跑的分片算出来，然后按 GPU 并发（每卡一个进程）
JOBS=()   # 每项: gpu|off|nenv|cs|ce|task_id|yaml|tag|extra
for T in $TASKS; do
  spec=$(task_spec "$T")
  if [[ "$spec" == "SKIP" ]]; then echo "[skip] $T"; continue; fi
  read -r CS CE TASK_ID TASK_YAML <<< "$spec"
  TOTAL=$(n_windows "$CS" "$CE")
  FULL=$(( (TOTAL / PER) * PER ))
  TAIL=$(( TOTAL - FULL ))
  echo "=== [tail] $T TOTAL=$TOTAL FULL=$FULL TAIL=$TAIL ==="
  if [[ "$TAIL" -gt 0 ]]; then
    JOBS+=("|$FULL|$TAIL|$CS|$CE|$TASK_ID|$TASK_YAML|tail_${T}|---allow-partial")
  fi
done

if [[ "${TAIL_ONLY:-0}" != "1" ]]; then
  # 整个丢失的满批分片（VisionNavi 32 个窗口，覆盖 clip 640/641/647/648 等）
  for off in 192 224; do
    JOBS+=("|$off|$PER|599|699|Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody|tasks/common_test_config/base_test/vision_navi_sonic_test.yaml|VisionNavi|---allow-partial")
  done
fi

echo "[tail] 共 ${#JOBS[@]} 个分片，${GPUS} 并发"
_i=0
for J in "${JOBS[@]}"; do
  IFS='|' read -r _g off nenv cs ce tid yml tag extra <<< "$J"
  gpu=$(echo $GPUS | cut -d' ' -f$(( (_i % 4) + 1 )))
  _i=$((_i + 1))
  run_one "$gpu" "$off" "$nenv" "$cs" "$ce" "$tid" "$yml" "$tag" "$extra" &
  # 每 4 个（=每张卡一个）等一轮
  if [[ $((_i % 4)) -eq 0 ]]; then wait; fi
done
wait
echo "[done] 覆盖率补洞完成（产物在 $OUT_DIR/tail_*_off*.pt 与 VisionNavi_off{192,224}.pt）"
