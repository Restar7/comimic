#!/usr/bin/env bash
# 四任务闭环推理（全部锁定在 comimic 内，不依赖 /tmp）
#
#   阶段 1：task4 冒烟（2 chunks，验证链路 + 出 mp4）
#   阶段 2：task1..task4 各 40 chunks（= 32 s 视频，runbook 口径）
#
# 用法：
#   bash closed_loop/run_tasks.sh            # 冒烟 + 四任务
#   STAGE=smoke  bash closed_loop/run_tasks.sh
#   STAGE=tasks  bash closed_loop/run_tasks.sh
#   TASKS="task4" CHUNKS=40 bash closed_loop/run_tasks.sh   # 只跑某一任务
#
# 环境要求（都已就位）：
#   - closed_loop/code/mosaic_n17/tools/closed_loop_mujoco.py   （官方实现）
#   - closed_loop/assets/{scenes,release,repro}                  （软链）
#   - closed_loop/bin/ffmpeg                                     （imageio-ffmpeg 静态二进制）
#   - runs/ckpt_vla20k                                           （VLA 离线视图）
#   - third_party/gr00t_deps 里的 onnxruntime / mujoco 3.10
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
DEPS=$COMIMIC/third_party/gr00t_deps
CL=$COMIMIC/closed_loop
OUT=$COMIMIC/runs/closed_loop
CKPT=${CKPT:-$COMIMIC/runs/ckpt_vla20k}
DEVICE=${DEVICE:-cuda:0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}   # 进程内只看到一张卡 → cuda:0 就是物理 GPU1
GPU_GIB=${GPU_GIB:-40GiB}
STAGE=${STAGE:-all}
CHUNKS=${CHUNKS:-40}
SMOKE_CHUNKS=${SMOKE_CHUNKS:-2}
TASKS=${TASKS:-"task1 task2 task3 task4"}

mkdir -p "$OUT"
STATUS=$OUT/STATUS.txt
: > "$STATUS"

export PATH=$CL/bin:$PATH                       # ffmpeg shim
export PYTHONPATH=$CL/code:$COMIMIC/Isaac-GR00T:$CL/assets/repro:$DEPS
export MUJOCO_GL=egl
export HF_HOME=/inspire/hdd/global_user/liumengfan-253108110079/cache/huggingface
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=8

run_one() {  # $1=task $2=chunks $3=tag
  local task=$1 chunks=$2 tag=$3
  local dir="$OUT/${task}_${tag}"
  local vid="$OUT/${task}_${tag}.mp4"
  # 每个任务用自己录制的 npz 标定 open/grasp 两个手部姿态点（趋势二值化用）
  local calib
  calib=$(ls "$COMIMIC"/data/npz/g1_tasks/${task}_*.npz 2>/dev/null | head -1)
  local hand_args=(--hand-mode trend)
  [ -n "$calib" ] && hand_args+=(--hand-calib-npz "$calib")
  echo "[$(date +%T)] START $task chunks=$chunks calib=$(basename "${calib:-none}") -> $vid" | tee -a "$STATUS"
  /usr/bin/python3 "$CL/code/mosaic_n17/tools/closed_loop_mujoco.py" \
    --task "$task" --ckpt "$CKPT" --chunks "$chunks" --execute-steps 20 \
    --device "$DEVICE" --gpu-gib "$GPU_GIB" \
    --scenes-dir "$CL/assets/scenes" \
    --release-dir "$CL/assets/release" \
    --repro-dir "$CL/assets/repro" \
    "${hand_args[@]}" --log-hands \
    --out-dir "$dir" --video "$vid" > "$OUT/${task}_${tag}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -s "$vid" ]; then
    echo "[$(date +%T)] OK    $task chunks=$chunks  $(du -h "$vid" | cut -f1)  $vid" | tee -a "$STATUS"
  else
    echo "[$(date +%T)] FAIL  $task chunks=$chunks rc=$rc  see $OUT/${task}_${tag}.log" | tee -a "$STATUS"
  fi
}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "smoke" ]; then
  run_one task4 "$SMOKE_CHUNKS" smoke
  if ! grep -q "OK    task4" "$STATUS"; then
    echo "[$(date +%T)] smoke 失败，停止后续任务（先修环境）" | tee -a "$STATUS"
    exit 1
  fi
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "tasks" ]; then
  for t in $TASKS; do
    run_one "$t" "$CHUNKS" 40ch
  done
fi

echo "[$(date +%T)] ALL DONE" | tee -a "$STATUS"
