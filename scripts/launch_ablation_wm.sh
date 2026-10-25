#!/usr/bin/env bash
# 消融：联合微调里世界模型要不要在线更新（--wm-update-every 0 vs N）。
#
# 问题：当前所有主线实验都冻结世界模型（`freeze_world_model`），所以 reward 是一个
# **固定代理**：它的系统性偏差永远不会被修正。之前测到 step 2000 时 token_mse 回涨
# 19% 而 rollout_reward 只涨 0.001，正是"policy/VLA 迎合固定代理"的特征。
#
# 这个消融就是回答：把 WM 也用真实转移数据在线拟合（有监督，不经策略梯度），
# 能不能让 reward 更贴近真值、把 token 保真度稳住？
#
# 臂（除了 wm-update-every，其余参数、种子、数据、warm start 完全相同）：
#   A  frozen   --wm-update-every 0     ← 当前主线口径
#   B  online1  --wm-update-every 1     ← 每步更新一次 WM
#   C  online10 --wm-update-every 10    ← 每 10 步更新一次（省时间）
#
# 注意：`train_world_model` 只做有监督拟合（window_loss），不在 BPTT 图里；每次更新
# 结束立刻重新冻结，所以 reward 回路里 WM 依旧不接收策略梯度。
#
# 用法：
#   bash scripts/launch_ablation_wm.sh                 # 依次跑 A / B 两臂
#   ARMS="A B C" STEPS=1000 bash scripts/launch_ablation_wm.sh
#   ARMS="B" STEPS=20 bash scripts/launch_ablation_wm.sh   # 冒烟
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
REPO="$COMIMIC/third_party/SuperMimic-Reproduced"
SM_CKPT="${SM_CKPT:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500.pt}"   # 写死：Isaac Lab/PhysX 版（见 runs/OFFICIAL_SUPERMIMIC.json）
VLM_PATH="$COMIMIC/groot_repro/nvidia/Cosmos-Reason2-2B-weights"
# 官方微调产物：与 launch_joint_ha.sh 同一个（runs/OFFICIAL_CKPT.json），消融臂共用。
VLM_CKPT="${VLM_CKPT:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}"
MOTION="${MOTION:-$COMIMIC/data/npz/ha_tokens}"
ARMS="${ARMS:-A B}"
STEPS="${STEPS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-250}"
PORT="${PORT:-29655}"

declare -A EVERY=( [A]=0 [B]=1 [C]=10 )
declare -A NAME=( [A]=abl_wm_frozen [B]=abl_wm_online1 [C]=abl_wm_online10 )

# 联合微调用系统 python3（SuperMimic venv 的 torchrun 与 mjlab/tyro 不兼容）。
# 两段 PYTHONPATH 都不能少：
#   gr00t_deps          -> tyro / transformers / onnxruntime 等 GR00T 私有依赖
#   gr00t_deps/_noflash -> 假 flash_attn（import 即抛错），迫使 transformers 回落 sdpa；
#                          反传微调里 VLM 输入是 fp32，真 FlashAttention 只收 fp16/bf16 会直接崩
export PYTHONPATH="$COMIMIC:$COMIMIC/Isaac-GR00T:$COMIMIC/third_party/gr00t_deps/_noflash${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$COMIMIC/logs" "$COMIMIC/runs"
case "$MOTION" in
  *mixed*|*ft318*|*g1_tasks*) echo "[fail] 只允许纯 HA 数据 (data/npz/ha_tokens)，拒绝: $MOTION"; exit 1 ;;
esac
case "$SM_CKPT" in
  *bones_seed_token_ha_v2*) ;;
  *) echo "[fail] WBC 只允许纯 HA 的 trackerlab ckpt，拒绝: $SM_CKPT"; exit 1 ;;
esac
for arm in $ARMS; do
  run="${NAME[$arm]}"
  out="$COMIMIC/runs/$run"
  log="$COMIMIC/logs/$run.log"
  echo "=== arm $arm: $run (wm-update-every=${EVERY[$arm]}) steps=$STEPS -> $out"
  python3 -m torch.distributed.run --nproc_per_node "${NGPU:-1}" --master_port "$PORT" \
    "$COMIMIC/scripts/train_joint.py" \
      --motion "$MOTION" \
      --supermimic-ckpt "$SM_CKPT" \
      --vlm --vlm-path "$VLM_PATH" --vlm-ckpt "$VLM_CKPT" \
      --vlm-dataset --vlm-steps 2 \
      --batch-size 2 --horizon 8 --steps "$STEPS" --log-every 50 \
      --eval-every "$EVAL_EVERY" --eval-batches 2 --eval-out "$out/eval_curve.jsonl" \
      --hand-loss-weight 0.2 --hand-grasp-boost 5 \
      --wm-update-every "${EVERY[$arm]}" \
      --save --save-every 500 --save-vla-every 5000 \
      --out "$out" 2>&1 | tee -a "$log"
  status=${PIPESTATUS[0]}
  echo "=== arm $arm finished with exit $status"
  [[ "$status" -ne 0 ]] && echo "[warn] arm $arm failed, continuing" 
done

echo
echo "=== eval 曲线对比 ==="
for arm in $ARMS; do
  f="$COMIMIC/runs/${NAME[$arm]}/eval_curve.jsonl"
  echo "--- $arm (${NAME[$arm]}) ---"
  [[ -f "$f" ]] && cat "$f" || echo "  (无)"
done
