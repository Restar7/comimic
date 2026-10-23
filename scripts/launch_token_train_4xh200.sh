#!/usr/bin/env bash
# 4×H200 离线启动脚本：token 条件化的 SuperMimic BONES-SEED 预训练。
#
# 前提（离线机器，不能联网）：
#   1. 与当前容器共享同一个挂载（仓路径 /venv 绝对路径一致），否则用 REPO/LIBRARY/ENCODER 覆盖；
#   2. 已就位：third_party/SuperMimic-Reproduced/.venv（torch 2.14.1+cu126 / warp 1.17 / mjlab 1.6）
#      + data/library/bones_seed_compact.pt + data/assets/model_encoder.onnx；
#   3. 4 张 H200 可见（nvidia-smi -L），驱动支持 CUDA 12.x。
#
# 用法：
#   bash scripts/launch_token_train_4xh200.sh                 # 4 卡，320k 迭代
#   NPROC=2 ITER=20000 bash scripts/launch_token_train_4xh200.sh
#   bash scripts/launch_token_train_4xh200.sh --resume logs/<run>/ckpt_latest.pt
#
# 环境变量：REPO SUPERMIMIC NPROC ITER ENVS BATCH PORT RUN LIBRARY ENCODER OMP_NUM_THREADS
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/.." && pwd)}"
SUPERMIMIC="${SUPERMIMIC:-$REPO_ROOT/third_party/SuperMimic-Reproduced}"

NPROC="${NPROC:-4}"
ITER="${ITER:-320000}"

# 利用率档位：paper=论文口径（低显存、低利用率）；high=更吃满 GPU；max=尽量吃满显存
# 显存参考（每卡 141 GB）：paper ~25 GB / high ~55 GB / max ~90-110 GB
PROFILE="${PROFILE:-high}"
case "$PROFILE" in
  paper) DEF_ENVS=4096;  DEF_BATCH=4096; DEF_WMB=2048 ;;
  high)  DEF_ENVS=8192;  DEF_BATCH=8192; DEF_WMB=4096 ;;
  max)   DEF_ENVS=16384; DEF_BATCH=8192; DEF_WMB=8192 ;;
  *) echo "ERROR: PROFILE 只能是 paper|high|max（当前 $PROFILE）" >&2; exit 1 ;;
esac
ENVS="${ENVS:-$DEF_ENVS}"
BATCH="${BATCH:-$DEF_BATCH}"
WMB="${WMB:-$DEF_WMB}"
PORT="${PORT:-29500}"
RUN="${RUN:-e9_bones_seed_token_h200x${NPROC}}"
LIBRARY="${LIBRARY:-$SUPERMIMIC/data/library/bones_seed_compact.pt}"
ENCODER="${ENCODER:-$SUPERMIMIC/data/assets/model_encoder.onnx}"
CKPT_INTERVAL="${CKPT_INTERVAL:-4000}"
# UTD 默认 0.4（论文）。只有「每卡跑多个进程」这种吞吐实验才需要改。
UTD="${UTD:-0.4}"
# LR 为空 = 用配置文件默认（基学习率 2e-4，训练脚本会再按 sqrt(world) 放大）。
# 多进程实验时用它把「基 lr 预除」，让放大后落回论文的 sqrt(2)*2e-4。
LR="${LR:-}"

fail() { echo "ERROR: $*" >&2; exit 1; }

# ---- 离线前置检查（不发起任何网络请求） ----
[[ -d "$SUPERMIMIC" ]] || fail "找不到 $SUPERMIMIC"
[[ -f "$SUPERMIMIC/.venv/bin/python" ]] || fail "找不到 venv: $SUPERMIMIC/.venv（离线环境需预装好）"
[[ -f "$LIBRARY" ]] || fail "找不到紧凑库: $LIBRARY"
[[ -f "$ENCODER" ]] || fail "找不到 token encoder: $ENCODER"

n_gpu=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
[[ "$n_gpu" -ge "$NPROC" ]] || fail "需要 $NPROC 张卡，检测到 $n_gpu 张（nvidia-smi -L）"

# ---- 守卫 1：同一台机器上已经有训练在占卡时直接报错（避免第二个进程抢显存/端口、双双变慢）----
busy_gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 > 8000 {c++} END {print c+0}')
if [[ "${busy_gpu:-0}" -gt 0 && "${FORCE:-0}" != "1" ]]; then
  echo "ERROR: 检测到 $busy_gpu 张卡的显存已被占用 >8 GB，可能已有训练在跑：" >&2
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >&2 || true
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >&2 || true
  fail "先停掉那条（pgrep -af 'supermimic.scripts.train' → kill <PID>），或 FORCE=1 强制并行"
fi

# ---- 守卫 2：端口自动避让（离线机器撞端口会静默失败，最常见的假死原因）----
port_free() {
  "$SUPERMIMIC/.venv/bin/python" -c "
import socket, sys
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sys.exit(0 if s.connect_ex(('127.0.0.1', $1)) != 0 else 1)
" 2>/dev/null
}
if ! port_free "$PORT"; then
  orig="$PORT"
  for _ in $(seq 1 40); do
    PORT=$((PORT + 1))
    port_free "$PORT" && break
  done
  echo "[launch] 端口 $orig 被占用 → 自动改用 $PORT" >&2
fi
[[ "$PORT" =~ ^[0-9]+$ ]] || fail "端口解析失败: $PORT"

GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ',' -)
echo "[launch] gpus=$n_gpu ($GPU_NAMES) nproc=$NPROC profile=$PROFILE envs/rank=$ENVS batch/rank=$BATCH wm-batch=$WMB iters=$ITER"
echo "[launch] run=$RUN"
echo "[launch] library=$LIBRARY"
echo "[launch] encoder=$ENCODER"

export SONIC_ENCODER_ONNX="$ENCODER"
export SM_TF32=1            # Hopper/Ada TF32：policy 的 BPTT 更新快 1.5-2x
export PYTHONPATH=
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 无网机器常常缺 libEGL.so.1，import mujoco 会炸
#   AttributeError: 'NoneType' object has no attribute 'eglQueryString'
# 训练完全不需要渲染，一劳永逸关掉 GL（要在本机可视化就把 MUJOCO_GL 覆盖成空/egl）。
export MUJOCO_GL="${MUJOCO_GL-disable}"

cd "$SUPERMIMIC"

# 启动前再确认 warp 能看到全部卡（warp 版本不对会静默退化到 CPU）
"$SUPERMIMIC/.venv/bin/python" - <<'PY'
import os, torch, warp as wp
wp.init()
devs = [d for d in wp.get_devices() if getattr(d, "is_cuda", False)]
n = int(os.environ.get("NPROC", "4"))
print(f"[launch] torch={torch.__version__} cuda_visible={torch.cuda.device_count()} warp_cuda={len(devs)}")
if len(devs) < n:
    raise SystemExit(
        f"warp 只看到 {len(devs)} 张 CUDA 卡（需要 {n}）；检查驱动/warp 版本（本环境要求 warp-lang 1.17.0）"
    )
PY

LR_ARGS=()
if [[ -n "$LR" ]]; then
  LR_ARGS=(--wm-trainer.lr "$LR" --policy-trainer.lr "$LR" --policy-trainer.critic-lr "$LR")
  echo "[launch] lr override base=$LR (训练中再 x sqrt(world))"
fi

echo "[launch] torchrun --nproc_per_node $NPROC ... utd=$UTD"
# 必须用 venv 里的 torchrun：/usr/local/bin/torchrun 是 `#!/usr/bin/python`（系统 python），
# 直接 `exec torchrun` 会拿系统 python 启动，报 No module named supermimic。
exec "$SUPERMIMIC/.venv/bin/torchrun" --nproc_per_node "$NPROC" --master_port "$PORT" \
  -m supermimic.scripts.train \
  --run-name "$RUN" \
  --seed 0 \
  --data.library "$LIBRARY" \
  --policy.token-dim 64 \
  --env.num-envs "$ENVS" \
  --policy-trainer.batch-size "$BATCH" \
  --policy-trainer.horizon 32 \
  --wm-trainer.batch-size "$WMB" \
  --schedule.total-iterations "$ITER" \
  --schedule.updates-per-env-step "$UTD" \
  --schedule.checkpoint-interval "$CKPT_INTERVAL" \
  "${LR_ARGS[@]}" \
  "$@"
