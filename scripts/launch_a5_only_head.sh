#!/usr/bin/env bash
# A5 · 冻结 SuperMimic 只训 head（head_only）vs head+WBC —— 第 1、2 卡（本容器 wscoli2）的正式启动器。
#
# 与 docs/ABLATION_ALLOCATION.md §A5 对齐，口径与 A2 完全一致（纯 HA + Isaac {pi,wm} 底模 +
# 冻结 VLM backbone），唯一变量 = 是否同时更新 WBC policy：
#   head_only      : --policy-lr 0 --adapter-lr 0     （只更新 VLA action head）
#   head_plus_wbc  : 默认 --policy-lr 1e-5           （head + policy 一起更新）
#
# 用法：
#   bash scripts/launch_a5_only_head.sh                       # 两臂 × 40 min ≈ 1.5 h
#   MINUTES=40 ARMS="head_only" bash scripts/launch_a5_only_head.sh
#   STEPS=2 MINUTES=0 SMOKE=1 bash scripts/launch_a5_only_head.sh   # 冒烟
#   CODE_ROOT=runs/_code_a5_20261024 bash scripts/launch_a5_only_head.sh
set -uo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
SM_CKPT="${SM_CKPT:-$COMIMIC/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt}"
VLM_PATH="${VLM_PATH:-$COMIMIC/groot_repro/nvidia/Cosmos-Reason2-2B-weights}"
VLM_CKPT="${VLM_CKPT:-$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}"
MOTION="${MOTION:-$COMIMIC/data/npz/ha_tokens}"                  # 纯 HA（699 条）
PACKED_DIR="${PACKED_DIR:-$COMIMIC/runs/isaac_motion/ha_tokens}" # Isaac 打包运动库，HA-only
OUT_ROOT="${OUT_ROOT:-$COMIMIC/runs/ablations/A5_only_head}"
ARMS="${ARMS:-head_only head_plus_wbc}"
STEPS="${STEPS:-1000000}"
MINUTES="${MINUTES:-40}"
BATCH="${BATCH:-2}"
HORIZON="${HORIZON:-8}"
EVAL_EVERY="${EVAL_EVERY:-300}"
LOG_EVERY="${LOG_EVERY:-50}"
PORT="${PORT:-29651}"
LOG_DIR="${LOG_DIR:-$COMIMIC/logs}"
SMOKE="${SMOKE:-0}"

if [[ "$SMOKE" == "1" ]]; then
  OUT_ROOT="$COMIMIC/runs/ablations/smoke_a5"
  LOG_DIR="$COMIMIC/logs_smoke"
  PORT="${PORT:-29661}"
fi
mkdir -p "$LOG_DIR"

for p in "$SM_CKPT" "$VLM_CKPT" "$VLM_PATH" "$MOTION" "$PACKED_DIR"; do
  [[ -e "$p" ]] || { echo "[fail] missing $p"; exit 1; }
done
[[ -f "$PACKED_DIR/packed.pt" ]] || { echo "[fail] missing $PACKED_DIR/packed.pt"; exit 1; }

# ---- 硬门禁 1：纯 HA，不许 ft318/mixed -----------------------------------------
for v in MOTION PACKED_DIR; do
  case "${!v}" in
    *ft318*|*mixed*|*Mixed*)
      echo "[fail] $v=${!v} 命中 ft318/mixed；消融只允许纯 HA（data/npz/ha_tokens）"; exit 2;;
  esac
done

# ---- 硬门禁 2：SuperMimic 必须是 Isaac {pi,wm} -----------------------------------
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

# ---- 硬门禁 3：恰好 2 张卡 ------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NGPU=$(/usr/bin/python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
if [[ "$NGPU" != "2" ]]; then
  echo "[fail] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES 下 torch 只看到 $NGPU 张卡（需要 2 张）"; exit 4
fi
echo "[gate] cards ok: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, 2 GPUs"

# ---- 硬门禁 4：臂名合法 ---------------------------------------------------------
for a in $ARMS; do
  case "$a" in
    head_only|head_plus_wbc) ;;
    *) echo "[fail] 未知 A5 臂: $a（只允许 head_only / head_plus_wbc）"; exit 5;;
  esac
done

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
echo "[config] out=$OUT_ROOT  batch=$BATCH horizon=$HORIZON minutes=$MINUTES steps=$STEPS"
echo "[config] arms=$ARMS  (VLM frozen: no --train-vlm)"
[[ -f "$CODE_ROOT/CODE_MANIFEST.sha256" ]] && \
  (cd "$CODE_ROOT" && sha256sum -c CODE_MANIFEST.sha256 2>&1 | tail -2) || true

# ---- 单实例保护 ----------------------------------------------------------------
mkdir -p "$COMIMIC/runs/ablations"
LOCK_FILE="${LOCK_FILE:-$COMIMIC/runs/ablations/.a5_fm.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[fail] 另一个 launch_a5_only_head.sh 正在跑（lock=$LOCK_FILE）；确认要重开先 kill 旧进程"; exit 7
fi
echo "[gate] lock ok: $LOCK_FILE"

# ---- 显存采样 ------------------------------------------------------------------
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
cleanup () { rm -f "$OUT_ROOT/.sampling"; kill "$GPUSAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT

declare -a FAILED=()
run_arm () {                 # run_arm <arm> <minutes> <eval_every> <extra...>
  local a="$1" minutes="$2" eval_every="$3"; shift 3
  local out="$OUT_ROOT/$a"
  mkdir -p "$out"
  echo "==================================================================" | tee -a "$LOG_DIR/a5.launcher.log"
  echo "[arm a5_$a] start $(date -Is) minutes=$minutes out=$out" | tee -a "$LOG_DIR/a5.launcher.log"
  echo "==================================================================" | tee -a "$LOG_DIR/a5.launcher.log"
  {
    echo "[arm a5_$a] requested $(date -Is) --policy-lr/--adapter-lr: $*"
    echo "[arm a5_$a] ckpt=$SM_CKPT"
    echo "[arm a5_$a] motion=$MOTION packed=$PACKED_DIR"
  } >> "$LOG_DIR/a5_$a.log"
  /usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port "$PORT" \
    scripts/train_joint.py \
      --motion "$MOTION" --motion-packed-dir "$PACKED_DIR" --supermimic-ckpt "$SM_CKPT" \
      --vlm --vlm-path "$VLM_PATH" --vlm-ckpt "$VLM_CKPT" --vlm-dataset --vlm-steps 2 \
      --batch-size "$BATCH" --horizon "$HORIZON" --steps "$STEPS" \
      --hand-loss-weight 0.2 --hand-grasp-boost 5 --state-vel-weight 0 \
      --time-budget-minutes "$minutes" \
      --eval-every "$eval_every" --eval-batches 2 \
      --eval-out "$out/eval_curve.jsonl" \
      --log-every "$LOG_EVERY" --save --save-every 0 \
      --out "$out" "$@" 2>&1 | tee -a "$LOG_DIR/a5_$a.log"
  local rc="${PIPESTATUS[0]}"
  echo "[arm a5_$a] rc=$rc finished $(date -Is)" | tee -a "$LOG_DIR/a5.launcher.log"
  [[ "$rc" == "0" ]] || FAILED+=("$a:$rc")
  PORT=$((PORT + 1))
}

for a in $ARMS; do
  case "$a" in
    head_only)     run_arm "$a" "$MINUTES" "$EVAL_EVERY" --policy-lr 0 --adapter-lr 0 ;;
    head_plus_wbc) run_arm "$a" "$MINUTES" "$EVAL_EVERY" ;;
  esac
done

echo "[done] arms=$ARMS failed=${FAILED[*]:-none} $(date -Is)"
[[ ${#FAILED[@]} -eq 0 ]] || exit 6
