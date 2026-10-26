#!/usr/bin/env bash
# Our differentiable joint finetune (stage (2)) on the HumanoidArena token data.
#
# Loss chain: VLA head -> 64-D motion token (+ 7+7 hands) -> SuperMimic policy -> frozen
# world model -> analytic tracking reward -> gradient back into the VLA head, plus the
# hand auxiliary loss.  Same trainer as runs/joint_sm24k_hand, only the data and the
# warm-start checkpoint differ:
#
#   --motion     data/npz/ha_tokens            <- scripts/prepare_ha_tokens_npz.py
#   --vlm-ckpt   runs/groot_sonic_vla_20k_ha   <- scripts/launch_ha_official_20k.sh
#
# Usage:
#   bash scripts/launch_joint_ha.sh
#   STEPS=500 bash scripts/launch_joint_ha.sh            # smoke
#   VLM_CKPT=/abs/path/to/别的ckpt \
#     bash scripts/launch_joint_ha.sh                    # 纯 HA 默认口径
set -euo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD="$COMIMIC/trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"
REPO="$COMIMIC/third_party/SuperMimic-Reproduced"
SM_CKPT="${SM_CKPT:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt}"   # 写死：Isaac Lab/PhysX 版（见 runs/OFFICIAL_SUPERMIMIC.json）
VLM_PATH="$COMIMIC/groot_repro/nvidia/Cosmos-Reason2-2B-weights"
# 官方微调产物：写死为 runs/OFFICIAL_CKPT.json 里那一个（当前 = lr1e-5/checkpoint-8000）。
# 想覆盖只用显式 VLM_CKPT=... 环境变量，默认值不再指向任何临时实验。
VLM_CKPT="${VLM_CKPT:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}"
MOTION="${MOTION:-$COMIMIC/data/npz/ha_tokens}"
RUN="${RUN:-joint_ha_a1}"
OUT="${OUT:-$COMIMIC/runs/$RUN}"
STEPS="${STEPS:-4000}"

for p in "$SM_CKPT" "$VLM_CKPT" "$VLM_PATH"; do
  [[ -e "$p" ]] || { echo "[fail] missing $p"; exit 1; }
done
[[ -d "$MOTION" ]] || { echo "[fail] missing $MOTION (run scripts/prepare_ha_tokens_npz.py)"; exit 1; }
case "$MOTION" in
  *mixed*|*ft318*|*g1_tasks*) echo "[fail] 只允许纯 HA 数据 (data/npz/ha_tokens)，拒绝: $MOTION"; exit 1 ;;
esac
case "$SM_CKPT" in
  *bones_seed_token_ha_v2*) ;;
  *) echo "[fail] WBC 只允许纯 HA 的 trackerlab ckpt，拒绝: $SM_CKPT"; exit 1 ;;
esac

# ------------------------------------------------------------------ 断点续训（可选）
# RESUME_STEP=2000 时：自动把该步的 vla_XXXXXX.pt 合并成可加载 ckpt，并从
# joint_XXXXXX.pt / vla_XXXXXX.pt 恢复（--steps 仍是**总步数**）。
RESUME_ARGS=()
if [[ -n "${RESUME_STEP:-}" ]]; then
  _step=$(printf '%06d' "$RESUME_STEP")
  _joint="$OUT/joint_${_step}.pt"
  _vla="$OUT/vla_${_step}.pt"
  [[ -f "$_joint" && -f "$_vla" ]] || { echo "[fail] 缺续训存档: $_joint / $_vla"; exit 1; }
  _merged="${MERGED_CKPT:-$COMIMIC/runs/ckpt_vla_ha_ft${RESUME_STEP}}"
  if [[ ! -f "$_merged/model-00001-of-00002.safetensors" ]]; then
    python3 "$COMIMIC/scripts/export_vla_ckpt.py" --base "$VLM_CKPT" --vla "$_vla" \
      --out "$_merged" --force || { echo "[fail] vla 合并失败"; exit 1; }
  fi
  RESUME_ARGS=(--vlm-ckpt "$_merged" --resume-joint "$_joint" --resume-vla "$_vla")
  echo "[resume] from step $RESUME_STEP: joint=$_joint vla=$_vla merged=$_merged"
fi

mkdir -p "$OUT" "$COMIMIC/logs"
LOG="$COMIMIC/logs/$RUN.log"

# 联合微调用系统 python3（SuperMimic venv 的 torchrun 与 mjlab/tyro 不兼容）。
# 两段 PYTHONPATH 都不能少：
#   gr00t_deps          -> tyro / transformers / onnxruntime 等 GR00T 私有依赖
#   gr00t_deps/_noflash -> 假 flash_attn（import 即抛错），迫使 transformers 回落 sdpa；
#                          反传微调里 VLM 输入是 fp32，真 FlashAttention 只收 fp16/bf16 会直接崩
export PYTHONPATH="$COMIMIC:$COMIMIC/Isaac-GR00T:$COMIMIC/third_party/gr00t_deps:$COMIMIC/third_party/gr00t_deps/_noflash${PYTHONPATH:+:$PYTHONPATH}"
cd "$COMIMIC"
exec python3 -m torch.distributed.run \
  --nproc_per_node "${NGPU:-1}" --master_port "${PORT:-29633}" \
  scripts/train_joint.py \
    --motion "$MOTION" \
    --supermimic-ckpt "$SM_CKPT" \
    --vlm --vlm-path "$VLM_PATH" --vlm-ckpt "$VLM_CKPT" \
    --vlm-dataset --vlm-steps 2 \
    --batch-size 2 --horizon 8 --steps "$STEPS" --log-every 100 \
    --eval-every "${EVAL_EVERY:-1000}" --eval-batches 2 \
    --eval-out "$OUT/eval_curve.jsonl" \
    --hand-loss-weight 0.2 --hand-grasp-boost 5 \
    --state-vel-weight "${STATE_VEL_WEIGHT:-0}" \
    --token-anchor-weight "${TOKEN_ANCHOR_WEIGHT:-0}" \
    --save --save-every 1000 --save-vla-every 1000 \
    --out "$OUT" "${RESUME_ARGS[@]}" \
    $([[ "${PREFETCH_VLA:-1}" == "1" ]] && echo --prefetch-vla || echo --no-prefetch-vla) \
  2>&1 | tee -a "$LOG"
