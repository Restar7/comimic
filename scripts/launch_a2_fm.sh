#!/usr/bin/env bash
# A2 · FM 反传方法（pathwise / checkpointed / truncated / adjoint）—— 第 1、2 卡（本容器 wscoli2）的正式启动器。
#
# 与 docs/ABLATION_ALLOCATION.md §2 的 A2 对齐，但按 2026-10-24 锁定的口径改造（那节命令已过期）：
#   1. 数据纯 HA：`data/npz/ha_tokens`（699 clips / 634,016 frames），packed 库
#      `runs/isaac_motion/ha_tokens`；mixed / ft318 被硬门禁拒绝（docs/DATA_POLICY_HA_ONLY.md）。
#   2. SuperMimic = Isaac Lab/PhysX 的 `{pi,wm}` ckpt（`runs/OFFICIAL_SUPERMIMIC.json` 写死 010500）。
#   3. 环境与 A3/A5 相同：系统 python（torch 2.8 + CUDA）+ gr00t 私有依赖 + 假 flash_attn（fp32 走 sdpa）。
#   4. 同墙钟 `--time-budget-minutes 30`/臂，同一 batch/horizon/辅助损失；只有 `--fm-grad-backend` 变。
#   5. VLM backbone 冻结（不加 `--train-vlm`）：allocation 明确 adjoint 会 detach backbone 特征，
#      这个消融只比较"梯度怎么穿过 FM 的 K 步 Euler 链"，不能把 train-vlm 混进来。
#
# 用法：
#   bash scripts/launch_a2_fm.sh                                  # 全跑 4 臂 × 30 min ≈ 2 h
#   BACKENDS="pathwise adjoint" bash scripts/launch_a2_fm.sh       # 只跑指定臂
#   STEPS=2 MINUTES=0 SMOKE=1 bash scripts/launch_a2_fm.sh        # 冒烟（2 步/臂，写 smoke 目录）
#   CODE_ROOT=runs/_code_a2_20261024 bash scripts/launch_a2_fm.sh # 用代码快照跑（冻结代码）
set -uo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
SM_CKPT="${SM_CKPT:-$COMIMIC/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt}"
VLM_PATH="${VLM_PATH:-$COMIMIC/groot_repro/nvidia/Cosmos-Reason2-2B-weights}"
VLM_CKPT="${VLM_CKPT:-$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}"
MOTION="${MOTION:-$COMIMIC/data/npz/ha_tokens}"                  # 纯 HA（699 条）
PACKED_DIR="${PACKED_DIR:-$COMIMIC/runs/isaac_motion/ha_tokens}" # Isaac(trackerlab) 打包运动库，HA-only
OUT_ROOT="${OUT_ROOT:-$COMIMIC/runs/ablations/A2_fm_backend}"
BACKENDS="${BACKENDS:-pathwise checkpointed truncated adjoint}"
STEPS="${STEPS:-1000000}"
MINUTES="${MINUTES:-30}"             # 每臂墙钟（allocation：4 臂 × 30 min）
BATCH="${BATCH:-2}"                  # 每 rank 的 batch（2 卡 → 全局 4；A3 同栈实测峰值 ~46 GB）
HORIZON="${HORIZON:-8}"
EVAL_EVERY="${EVAL_EVERY:-300}"
LOG_EVERY="${LOG_EVERY:-50}"
PORT="${PORT:-29646}"
LOG_DIR="${LOG_DIR:-$COMIMIC/logs}"
SMOKE="${SMOKE:-0}"

if [[ "$SMOKE" == "1" ]]; then
  OUT_ROOT="$COMIMIC/runs/ablations/smoke_a2_fm"
  LOG_DIR="$COMIMIC/logs_smoke"
  PORT="${PORT:-29656}"
fi
mkdir -p "$LOG_DIR"

for p in "$SM_CKPT" "$VLM_CKPT" "$VLM_PATH" "$MOTION" "$PACKED_DIR"; do
  [[ -e "$p" ]] || { echo "[fail] missing $p"; exit 1; }
done
[[ -f "$PACKED_DIR/packed.pt" ]] || { echo "[fail] missing $PACKED_DIR/packed.pt"; exit 1; }

# ---- 硬门禁 1：数据只允许纯 HA，永远不碰 ft318 / mixed（用户口径，写死）----------
for v in MOTION PACKED_DIR; do
  case "${!v}" in
    *ft318*|*mixed*|*Mixed*)
      echo "[fail] $v=${!v} 命中 ft318/mixed；消融只允许纯 HA（data/npz/ha_tokens）"
      exit 2;;
  esac
done
case "$(basename "$MOTION")" in
  ha_tokens|ha_sessions) ;;
  *) echo "[warn] MOTION 目录名不是 ha_tokens/ha_sessions：$MOTION（确认它确实是纯 HA 数据）";;
esac

# ---- 硬门禁 2：SuperMimic 必须是 Isaac Lab 训出来的 {pi,wm} 格式，不许用 mjlab 旧链路 ----
if ! /usr/bin/python3 - "$SM_CKPT" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert "pi" in ck and "wm" in ck and "policy" not in ck, (
    "not an Isaac/trackerlab checkpoint; refusing the mjlab (MuJoCo) chain")
print(f"[gate] Isaac WBC ok: iteration={ck.get('iteration')}, "
      f"obs_mode={(ck.get('args') or {}).get('obs_mode')}, interface={(ck.get('args') or {}).get('interface')}")
PY
then
  echo "[fail] $SM_CKPT 不是 Isaac/trackerlab 的 {pi,wm} ckpt —— 拒绝在旧 mjlab 链路上跑消融"
  exit 3
fi

# ---- 硬门禁 3：必须恰好用上本容器的 2 张卡 -------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NGPU=$(/usr/bin/python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
if [[ "$NGPU" != "2" ]]; then
  echo "[fail] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES 下 torch 只看到 $NGPU 张卡（需要 2 张）"
  exit 4
fi
echo "[gate] cards ok: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, 2 GPUs"

# ---- 硬门禁 4：backend 名字必须合法（拼错时不要静默跑成 pathwise）----------------
for b in $BACKENDS; do
  case "$b" in
    pathwise|checkpointed|truncated|adjoint) ;;
    qvgm_lite) echo "[fail] qvgm_lite 在真实栈不提供（需要 chunk critic），只在 ablation_fm_grad/ 的 toy 里有"; exit 5;;
    *) echo "[fail] 未知 --fm-grad-backend: $b"; exit 5;;
  esac
done

# GR00T 私有依赖 + 假 flash_attn（迫使 transformers 回落 sdpa；fp32 VLA 只吃 sdpa）
export PYTHONPATH="$CODE_ROOT:$COMIMIC/third_party/gr00t_deps:$COMIMIC/third_party/gr00t_deps/_noflash${PYTHONPATH:+:$PYTHONPATH}"
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

cd "$CODE_ROOT"
echo "[code] running from $CODE_ROOT"
echo "[config] MOTION=$MOTION"
echo "[config] PACKED_DIR=$PACKED_DIR"
echo "[config] SM_CKPT=$SM_CKPT"
echo "[config] VLM_CKPT=$VLM_CKPT"
echo "[config] out=$OUT_ROOT  batch=$BATCH horizon=$HORIZON  minutes=$MINUTES steps=$STEPS"
echo "[config] backends=$BACKENDS  (VLM frozen: no --train-vlm)"
echo "[code-manifest] $CODE_ROOT/CODE_MANIFEST.sha256"
[[ -f "$CODE_ROOT/CODE_MANIFEST.sha256" ]] && \
  (cd "$CODE_ROOT" && sha256sum -c CODE_MANIFEST.sha256 2>&1 | tail -3) || true

# ---- 单实例保护：同一 launcher 不允许并行（避免两个进程写同一 eval_curve / 抢同一对卡）----
mkdir -p "$COMIMIC/runs/ablations"
LOCK_FILE="${LOCK_FILE:-$COMIMIC/runs/ablations/.a2_fm.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[fail] 另一个 launch_a2_fm.sh 正在跑（lock=$LOCK_FILE）；确认要重开先 kill 旧进程"
  exit 7
fi
echo "[gate] lock ok: $LOCK_FILE"

# ---- 显存采样（10 s 一次，用于事后核对是否顶到 48 GB）-----------------------------
GPU_CSV="$OUT_ROOT/gpu_mem.csv"
mkdir -p "$OUT_ROOT"
touch "$OUT_ROOT/.sampling"
(
  echo "timestamp, index, memory.used [MiB], utilization.gpu [%]"
  while [[ -f "$OUT_ROOT/.sampling" ]]; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader
    sleep 10
  done
) > "$GPU_CSV" 2>/dev/null &
GPUSAMPLER_PID=$!
cleanup () {
  rm -f "$OUT_ROOT/.sampling"
  kill "$GPUSAMPLER_PID" 2>/dev/null || true
}
trap cleanup EXIT

declare -a FAILED=()
run_arm () {                 # run_arm <backend> <minutes> <eval_every>
  local b="$1" minutes="$2" eval_every="$3"
  local out="$OUT_ROOT/$b"
  mkdir -p "$out"
  echo "==================================================================" | tee -a "$LOG_DIR/a2.launcher.log"
  echo "[arm a2_$b] start $(date -Is) minutes=$minutes out=$out" | tee -a "$LOG_DIR/a2.launcher.log"
  echo "==================================================================" | tee -a "$LOG_DIR/a2.launcher.log"
  {
    echo "[arm a2_$b] requested $(date -Is) --fm-grad-backend=$b batch=$BATCH horizon=$HORIZON"
    echo "[arm a2_$b] ckpt=$SM_CKPT"
    echo "[arm a2_$b] motion=$MOTION packed=$PACKED_DIR"
    echo "[arm a2_$b] code=$CODE_ROOT $(sha256sum scripts/train_joint.py | cut -d' ' -f1)"
  } >> "$LOG_DIR/a2_$b.log"
  /usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port "$PORT" \
    scripts/train_joint.py \
      --motion "$MOTION" --motion-packed-dir "$PACKED_DIR" --supermimic-ckpt "$SM_CKPT" \
      --vlm --vlm-path "$VLM_PATH" --vlm-ckpt "$VLM_CKPT" --vlm-dataset --vlm-steps 2 \
      --batch-size "$BATCH" --horizon "$HORIZON" --steps "$STEPS" \
      --fm-grad-backend "$b" \
      --hand-loss-weight 0.2 --hand-grasp-boost 5 --state-vel-weight 0 \
      --time-budget-minutes "$minutes" \
      --eval-every "$eval_every" --eval-batches 2 \
      --eval-out "$out/eval_curve.jsonl" \
      --log-every "$LOG_EVERY" --save --save-every 0 \
      --out "$out" 2>&1 | tee -a "$LOG_DIR/a2_$b.log"
  local rc="${PIPESTATUS[0]}"
  echo "[arm a2_$b] rc=$rc finished $(date -Is)" | tee -a "$LOG_DIR/a2.launcher.log"
  [[ "$rc" == "0" ]] || FAILED+=("$b:$rc")
  # 每个臂换个端口，避免上一臂的 TIME_WAIT 影响下一臂
  PORT=$((PORT + 1))
}

for b in $BACKENDS; do
  run_arm "$b" "$MINUTES" "$EVAL_EVERY"
done

echo "[done] backends=$BACKENDS failed=${FAILED[*]:-none} $(date -Is)"
[[ ${#FAILED[@]} -eq 0 ]] || exit 6
