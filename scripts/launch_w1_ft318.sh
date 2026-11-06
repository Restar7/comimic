#!/usr/bin/env bash
# W1 · 318 数据集在 SuperMimic（mjlab/MuJoCo）上续训 20k（H200×2）
#
# 为什么不直接手打长命令：`\` 续行后面**不能**跟行内注释（`\` + 空格会转义空格而不是换行，
# 命令会被截断，剩下的参数被当成 shell 命令执行 —— 典型症状是日志里只有
# `bash: --env.num-envs: command not found`）。本脚本把参数放进数组，彻底避免这类事故。
#
# 用法：
#   bash scripts/launch_w1_ft318.sh              # 正式：2×H200，20k 步，后台
#   DRY=1 bash scripts/launch_w1_ft318.sh        # 只打印将要执行的命令，不启动
#   SMOKE=1 bash scripts/launch_w1_ft318.sh      # 冒烟：单卡 200 步，前台带 tee
#
# 可覆盖的环境变量：
#   RUN STEPS ENVS LR GPUS NPROC CKPT_EVERY LOG_EVERY PORT RESUME
set -euo pipefail

COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
REPO=$COMIMIC/third_party/SuperMimic-Reproduced
PY=$REPO/.venv/bin/python
TORCHRUN=$REPO/.venv/bin/torchrun

# 可用 LIB=... 覆盖（2026-10-30：参考补齐根轨迹后换成 ft318_compact_v2.pt）
LIB=${LIB:-$COMIMIC/data/library/ft318_compact.pt}
TOK=$COMIMIC/data/library/ft318_tokens.pt

RUN=${RUN:-wbc_ft318_mjlab}
# 43dof = 带手资产（A 步）；空 = 旧 29-dof 行为
STEPS=${STEPS:-20000}
ENVS=${ENVS:-4096}
LR=${LR:-1.4e-5}
NPROC=${NPROC:-2}
GPUS=${GPUS:-0,1}
CKPT_EVERY=${CKPT_EVERY:-1000}
LOG_EVERY=${LOG_EVERY:-200}
PORT=${PORT:-29640}
RESUME=${RESUME:-$COMIMIC/runs/wbc_resume/e9_0024000_ft318.pt}
# 续训时**必须**把 warmup 抬到"当前 iteration + 几十"：replay buffer 是空的，
# 而 loop 在第一轮就要更新 WM ⇒ `replay buffer has no sampleable rows yet`（2026-10-29 实测）。
# 例：从 ckpt_0014000 续训 ⇒ WARMUP=14040。
WARMUP=${WARMUP:-40}
# 续训时把 lr 余弦窗口单独放大（否则一开跑就掉进 base*0.03 的平底）
LR_TOTAL=${LR_TOTAL:-}
# 2026-10-30：短 episode（官方 σ=0.3 的位置项需要"误差小、频繁 reset 回参考"）；
# 以及 speed 加权采样（318 里 72% 原地 ⇒ 均匀采样会把走路能力抹掉）。
EPLEN=${EPLEN:-10.0}
SAMPLING=${SAMPLING:-return}
SPEED_WMAX=${SPEED_WMAX:-2.5}
# 2026-10-30：锚到基座策略（防遗忘）。0=关。经验起点 1e-3。
ANCHOR_L2=${ANCHOR_L2:-0.0}

# ------------------------------------------------------------------ 冒烟模式
SMOKE=${SMOKE:-0}
if [[ $SMOKE == 1 ]]; then
  RUN=${RUN_OVERRIDE:-smoke_ft318}
  STEPS=200
  ENVS=512
  NPROC=1
  GPUS=${GPUS%%,*}
  CKPT_EVERY=200
  LOG_EVERY=20
  BATCH=(--wm-trainer.batch-size 256 --wm-trainer.batches-per-update 2
         --policy-trainer.batch-size 512)
else
  BATCH=(--wm-trainer.batch-size 2048 --wm-trainer.batches-per-update 10)
fi

# ------------------------------------------------------------------ 前置检查
for f in "$LIB" "$TOK" "$RESUME"; do
  [[ -f $f ]] || { echo "[fail] 缺少产物: $f" >&2; exit 1; }
done
[[ -x $PY ]] || { echo "[fail] 解释器不存在: $PY" >&2; exit 1; }
[[ -x $TORCHRUN ]] || { echo "[fail] torchrun 不存在: $TORCHRUN" >&2; exit 1; }
NGPU=$(python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
if [[ $NGPU -lt $NPROC ]]; then
  echo "[fail] 需要 $NPROC 张卡，只看到 $NGPU 张" >&2; exit 1
fi
mkdir -p "$COMIMIC/logs"

# ------------------------------------------------------------------ 参数（数组，无续行陷阱）
ARGS=(
  --run-name    "$RUN"
  --log-root    logs
  --data.library "$LIB"
  --data.tokens  "$TOK"
  --resume       "$RESUME"
  --env.num-envs "$ENVS"
  --env.action-noise-std 0.1
  --env.no-reset-noise
  --env.episode-length-s "$EPLEN"
  --env.sampling-mode "$SAMPLING"
  --policy.token-dim 64
  --policy-trainer.anchor-l2 "$ANCHOR_L2"
  --schedule.total-iterations    "$STEPS"
  --schedule.checkpoint-interval "$CKPT_EVERY"
  --schedule.log-interval        "$LOG_EVERY"
  --schedule.warmup-iterations   "$WARMUP"
)
if [[ -n $LR_TOTAL ]]; then ARGS+=(--schedule.lr-total-iterations "$LR_TOTAL"); fi
ARGS+=(
  "${BATCH[@]}"
  --wm-trainer.lr "$LR"
  --policy-trainer.lr "$LR"
)

cd "$REPO"
LOG=$COMIMIC/logs/$RUN.log

if [[ ${DRY:-0} == 1 ]]; then
  echo "# 将在 $REPO 下执行（$NPROC 卡，GPUS=$GPUS）："
  printf 'CUDA_VISIBLE_DEVICES=%s MUJOCO_GL=disable OMP_NUM_THREADS=8 \\\n  %q --nproc_per_node %s --master_port %s \\\n    -m supermimic.scripts.train' \
    "$GPUS" "$TORCHRUN" "$NPROC" "$PORT"
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

if [[ $SMOKE == 1 ]]; then
  echo "[smoke] 单卡前台，$STEPS 步，日志 tee 到 $LOG"
  CUDA_VISIBLE_DEVICES="$GPUS" MUJOCO_GL=disable OMP_NUM_THREADS=8 \
    "$PY" -m supermimic.scripts.train "${ARGS[@]}" 2>&1 | tee "$LOG"
  exit ${PIPESTATUS[0]}
fi

echo "[run] $RUN | ${NPROC}×GPU($GPUS) | $ENVS envs | $STEPS steps | eplen=$EPLEN sampling=$SAMPLING anchor_l2=$ANCHOR_L2 | lr=$LR(有效≈$(python3 -c "print(round($LR*$NPROC**0.5,2))"))"
echo "[run] resume = $RESUME"
setsid nohup env \
  CUDA_VISIBLE_DEVICES="$GPUS" \
  MUJOCO_GL=disable \
  SM_G1_ASSET="${SM_G1_ASSET:-}" \
  SM_G1_XML="${SM_G1_XML:-}" \
  OMP_NUM_THREADS=8 \
  "$TORCHRUN" --nproc_per_node "$NPROC" --master_port "$PORT" \
    -m supermimic.scripts.train "${ARGS[@]}" \
  > "$LOG" 2>&1 < /dev/null &

PID=$!
echo "[run] pid=$PID  log=$LOG"
echo "[run] 看进度: tail -f $LOG"
echo "[run] 期望 3 行: '[tokens] recorded dataset tokens' / 'params: world model' / 'resumed from ... at iteration 0'"
