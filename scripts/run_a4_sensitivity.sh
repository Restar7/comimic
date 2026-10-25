#!/usr/bin/env bash
# A4 敏感性复查：WM 在线更新的 **batch 大小** 是否会改变结论。
#
# 动机：A4 v3 的 wm_live 臂每 50 步用 8 个参考窗口拟一次 WM（因为每个更新要和 2.5e9 参数的
# VLA 梯度步抢 2 s 预算），而 Isaac 自己每轮用 10×2048 个窗口。若 live 臂"看起来没差别"，
# 必须先排除"更新太稀疏/太吵"这个解释。这里用同一条冻结代码路径、同一份数据，只把
# batch 从 8 提到 256、频率从 50 提到 25，跑一格 20 min 做对照。
#
# 用法：
#   setsid nohup bash scripts/run_a4_sensitivity.sh > logs/a4_sens.log 2>&1 &
set -uo pipefail
REPO=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
CODE_ROOT="$REPO/runs/_code_a4_20261025"     # 与 A4 v3 完全同一份冻结代码
cd "$CODE_ROOT"
SM_CKPT="$REPO/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt"
MOTION="$REPO/data/npz/ha_tokens"
PACKED="$REPO/runs/isaac_motion/ha_tokens/packed.pt"
VLM_PATH="$REPO/groot_repro/nvidia/Cosmos-Reason2-2B-weights"
VLM_CKPT="$REPO/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000"
OUT="$REPO/runs/ablations/A4_sensitivity/wm_live_H8_b256"
LOG="$REPO/logs/a4_sensitivity/wm_live_H8_b256.log"
MEM="$REPO/logs/a4_sensitivity/wm_live_H8_b256.mem"
MINUTES=${MINUTES:-20}

export COMIMIC_ACTION_SCALE=one COMIMIC_REWARD_ACTION_SCALE=spec
export PYTHONPATH="$CODE_ROOT:$REPO/third_party/gr00t_deps:$REPO/third_party/gr00t_deps/_noflash${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$(dirname "$LOG")" "$OUT"

if pgrep -f "scripts/train_joint.py" > /dev/null; then
  echo "[sens][ABORT] 已有 train_joint.py 在跑"; exit 1
fi
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
[[ "${free_mb:-0}" -ge 42000 ]] || { echo "[sens][ABORT] 可用显存 ${free_mb} MiB < 42000"; exit 1; }
( cd "$CODE_ROOT" && sha256sum -c CODE_MANIFEST.sha256 --quiet ) || { echo "[sens][ABORT] 快照代码被改动"; exit 1; }

( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"; sleep 10; done > "$MEM" ) &
POLLER=$!
CUDA_VISIBLE_DEVICES=0,1 timeout $(( MINUTES * 60 + 900 ))s \
  /usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29891 \
    "$CODE_ROOT/scripts/train_joint.py" \
      --motion "$MOTION" --motion-packed "$PACKED" --supermimic-ckpt "$SM_CKPT" \
      --vlm --vlm-path "$VLM_PATH" --vlm-ckpt "$VLM_CKPT" --vlm-dataset --vlm-steps 2 \
      --batch-size 1 --horizon 8 --steps 1000000 --time-budget-minutes "$MINUTES" \
      --hand-loss-weight 0.2 --hand-grasp-boost 5 --state-vel-weight 0.05 \
      --wm-update-every 25 --wm-update-warmup 50 --wm-update-batch 256 --wm-update-window 8 \
      --eval-every 100 --eval-batches 6 --eval-out "$OUT/eval_curve.jsonl" \
      --save --save-every 0 --log-every 20 --seed 0 \
      --out "$OUT" > "$LOG" 2>&1
STATUS=$?
kill "$POLLER" 2>/dev/null
echo "[sens] exit=$STATUS  log=$LOG"
