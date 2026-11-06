#!/usr/bin/env bash
# A' 采集（**SONIC decoder 执行器版**，2026-11-04）
#
# 为什么重采：2026-11-03 那批 A' 用的是 `--executor wbc`（我们自己的 SuperMimic WBC），
# 而录制的参考轨迹是 **SONIC release decoder** 产出的。用 SuperMimic 去追它，
# 等于拿两个不同执行器互相比 —— 实测段内 20 步漂移 7.54cm(p50)/22.4(p90)、
# 关节 0.139 rad，而且这笔"误差"里混着执行器差异，不能当动力学学。
#
# 2026-11-04 把 SONIC decoder 闭环的两个真 bug 修掉后（见
# docs/A_PRIME_FULL_SEG20_20261103.md §13）：
#   * obs 的 q/qd 块必须按 **SONIC 序** 读（`idx_s`），不是 standard 序
#   * 动作目标必须按 `idx_s` 落位（`SONIC_DEFAULT_POS + raw*G1_ACTION_SCALE_ISAACLAB`）
#   * 段重置也按 `idx_s` 写（原来按 `idx_t` ⇒ 每 20 步瞬移到错位姿态）
# 同一个 20 步窗口的实测对比：
#     SuperMimic: 7.54cm / 22.4(p90) / 0.139 rad
#     SONIC     : 2.75cm /  6.3(p90) / 0.070 rad   (σ=0.05 也在)
# 并且 120 步不摔（z 0.78、up 0.998）。
#
# 口径：全长覆盖（每 270 帧一个进程块，连续铺满 [0,L)，含尾窗口）
#       + 每 20 步重置回参考（teacher forcing）
#       + 动作噪声 σ=0.05、每 20 步重抽一次（perturb_hold=20）
#
# 用法：
#   bash scripts/run_a_prime_sonic_seg20.sh                     # 6 个任务全跑
#   TASKS="DoubleDesk" bash scripts/run_a_prime_sonic_seg20.sh  # 单任务
#   SMOKE=1 TASKS="DoubleDesk" bash scripts/run_a_prime_sonic_seg20.sh
set -uo pipefail

C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
OUT_DIR=${OUT_DIR:-$C/runs/a_prime_sonic_leash}
# ★ 2026-11-04：缰绳式重置（|ξ|>δ 才拉回参考）；0 = 退回固定 --segment 口径
RESET_MAX_DRIFT=${RESET_MAX_DRIFT:-0.10}
# WINDOW = 窗口枚举跨度（覆盖粒度）；STEPS = 一个进程连续跑多少步（= WINDOW 的整数倍）
# 实测：启动 ~60 s、仿真 ~0.05 s/步 ⇒ 让一个进程连跑 3 个窗口（STEPS=780）能把
# 启动开销摊薄 3 倍，而覆盖粒度仍是 260（所有 clip 都能覆盖）。
WINDOW=${WINDOW:-260}      # 最小可用 clip 长度（--len-min），不是枚举跨度
STEPS=${STEPS:-780}        # 一个进程连跑多少步（= 枚举跨度，780=3 个 260 窗口）
# leash 模式下 SEG 必须为 0（否则固定段重置会优先触发）
SEG=${SEG:-0}
NOISE=${NOISE:-0.05}
HOLD=${HOLD:-20}
PER=${PER:-16}
WPG=${WPG:-1}
GPUS=${GPUS:-"0 1 2 3"}
TASKS=${TASKS:-"DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi"}
DECODER=${DECODER:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx}

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PROJECT_ROOT=$C/third_party/HumanoidArena/isaaclab_twist2_g1
export PYTHONPATH="$C/third_party/unitree_sdk2_python:$PROJECT_ROOT:$C:$C/trackerlab-supermimic/source/trackerTask:$C/trackerlab-supermimic/source/trackerLab:${PYTHONPATH:-}"
export SUPERMIMIC_HA_ALIGN=1 HA_ALLOW_SONIC=1
# ★ 2026-11-04 bug①：`HA_VLA_EXEC_STEPS=20` 的含义是"每 20 步重新规划"，
#   不是"token 保持 20 步"（部署侧每个控制步 pop 一个 chunk 元素）。录制数据里
#   `encoder_latent` 逐帧都在变（相邻帧差 0.0045、跨 20 帧差 0.058）⇒ 采集必须
#   `--exec-steps 1`（每帧 token）。实测段末漂移 2.57cm → 1.95cm。
mkdir -p "$OUT_DIR"
cd "$C" || exit 1

[[ -f "$DECODER" ]] || { echo "[fail] 找不到 SONIC decoder: $DECODER" >&2; exit 3; }

task_spec() {
  case "$1" in
    DoubleDesk)   echo "0 100 Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody tasks/common_test_config/base_test/doubledesk_sonic_test.yaml" ;;
    PPBox)        echo "199 299 Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby tasks/common_test_config/base_test/pp_box_sonic_test.yaml" ;;
    Boxing)       echo "299 399 Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody tasks/common_test_config/base_test/boxing_sonic_test.yaml" ;;
    OpenDoor)     echo "399 499 Isaac-Move-Open-Door-G129-Dex3-Wholebody tasks/common_test_config/base_test/open_door_sonic_test.yaml" ;;
    SitSofa)      echo "499 599 Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody tasks/common_test_config/base_test/sit_sofa_sonic_test.yaml" ;;
    VisionNavi)   echo "599 699 Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody tasks/common_test_config/base_test/vision_navi_sonic_test.yaml" ;;
    *)            echo "SKIP" ;;
  esac
}

# 与 isaac_a_collect_batch.py 的 windows 枚举**逐位一致**
n_windows() {
  python3 - "$C" "$STEPS" "$WINDOW" "$1" "$2" <<'PY'
import sys, torch
c, steps, lmin, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
pk = torch.load(f"{c}/runs/isaac_motion/ha_cmd_exec/packed.pt", map_location="cpu", weights_only=False)
tot = 0
for L in list(pk["lengths"])[a:b]:
    L = int(L)
    if L < max(20, lmin):
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

run_one() {  # gpu off nenv cs ce tid yml tag extra
  local gpu=$1 off=$2 nenv=$3 cs=$4 ce=$5 tid=$6 yml=$7 tag=$8 extra=${9:-}
  local outp="$OUT_DIR/${tag}_off${off}.pt"
  # ★ 2026-11-04：rc=0 **不代表成功**（并行 Isaac 曾经 rc=0 但没产物）。产物存在才算。
  if [[ -f "$outp" && "${FORCE:-0}" != "1" ]]; then
    echo "[$tag gpu$gpu] offset=$off 已有产物，跳过"; return 0
  fi
  CUDA_VISIBLE_DEVICES=$gpu timeout 900 "$C/third_party/isaacsim_env/bin/python" -u \
    scripts/isaac_a_collect_batch.py \
    --enable_cameras --headless --executor sonic --sonic-decoder "$DECODER" \
    --wbc-ckpt runs/S1_handover/exp_WBC_exec_haonly_best.pt \
    --packed runs/isaac_motion/ha_cmd_exec/packed.pt \
    --tokens-dir runs/exp_tokens --token-source npy \
    --steps "$STEPS" --len-min "$WINDOW" --segment "$SEG" --start-mode windows \
    --reset-max-drift "${RESET_MAX_DRIFT:-0.10}" \
    --exec-steps 1 \
    --clip-start "$cs" --clip-end "$ce" \
    --num-envs "$nenv" --plan-offset "$off" --seed "$off" $extra \
    --action-noise-std "$NOISE" --perturb-hold "$HOLD" \
    --ha-task "$tid" --ha-yaml "$yml" --log-every 100 \
    --out "$outp" > "$OUT_DIR/${tag}_off${off}.log" 2>&1
  local rc=$?
  if [[ -f "$outp" ]]; then
    echo "[$tag gpu$gpu] offset=$off nenv=$nenv rc=$rc 产物 OK"
  else
    echo "[$tag gpu$gpu] offset=$off nenv=$nenv rc=$rc **没有产物**（失败）"
  fi
  return 0
}

all_offsets() {  # TOTAL -> 所有分片起点（含尾批）
  local total=$1 stride=$(( $(echo $GPUS | wc -w) * WPG * PER ))
  local off=0
  while [[ $off -lt $total ]]; do echo $off; off=$((off + stride)); done
}

for T in $TASKS; do
  spec=$(task_spec "$T")
  if [[ "$spec" == "SKIP" ]]; then echo "[skip] $T"; continue; fi
  read -r CS CE TASK_ID TASK_YAML <<< "$spec"
  TOTAL=$(n_windows "$CS" "$CE")
  FULL=$(( (TOTAL / PER) * PER ))
  TAIL=$(( TOTAL - FULL ))
  echo "=== [task] $T clips=[$CS,$CE) windows=$TOTAL (尾批 $TAIL) ==="

  if [[ "${SMOKE:-0}" == "1" ]]; then
    run_one "${GPUS%% *}" 0 2 "$CS" "$CE" "$TASK_ID" "$TASK_YAML" "smoke_${T}" "--allow-partial"
    continue
  fi

  # 满批分片：每张卡一个进程，4 卡并发；同一波内不同 GPU 各跑一个 offset
  off=0
  while [[ $((off + PER)) -le $TOTAL ]]; do
    pids=()
    for g in $GPUS; do
      for w in $(seq 0 $((WPG - 1))); do
        _o=$(( off + (g + w * 4) * PER ))
        if [[ $((_o + PER)) -le $TOTAL ]]; then
          run_one "$g" "$_o" "$PER" "$CS" "$CE" "$TASK_ID" "$TASK_YAML" "$T" "" &
          pids+=($!)
        fi
      done
    done
    wait
    off=$(( off + $(echo $GPUS | wc -w) * WPG * PER ))
  done
  # 尾批（不足一批的窗口，用 --allow-partial）
  if [[ "$TAIL" -gt 0 ]]; then
    run_one "${GPUS%% *}" "$FULL" "$TAIL" "$CS" "$CE" "$TASK_ID" "$TASK_YAML" "$T" "--allow-partial"
  fi
  echo "=== [task] $T 完成 ==="
done
echo "[done] SONIC-decoder A' 采集结束 -> $OUT_DIR"
