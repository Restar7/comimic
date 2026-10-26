#!/usr/bin/env bash
# Official GR00T-N1.7-3B finetune on the HumanoidArena raw SONIC data (stage O).
#
# This is the *same* recipe that produced our current 20 000-step checkpoint, with the
# dataset root swapped for the converted arena sessions.  The head is UNITREE_G1_SONIC,
# i.e. 64-D motion token + 7+7 hand joints -- no head swap is needed afterwards, our
# joint finetune (scripts/launch_joint_ha.sh) loads checkpoint-20000 directly.
#
# Prereqs (all verified on this box):
#   - data/ha_sessions/roots.txt      <- scripts/prepare_ha_sessions.py
#   - $HF cache holds nvidia/GR00T-N1.7-3B (6.5 GB) and nvidia/Cosmos-Reason2-2B (4.6 GB)
#
# Reference cost: 271 episodes / 405k frames took ~4.6 h on 1x4090; we have
# ~699 episodes / ~600k frames and the same 20 000 steps, so 2 GPUs -> ~2.5 h.
#
# Usage:
#   bash scripts/launch_ha_official_20k.sh                 # 20000 steps, 2 GPUs
#   MAX_STEPS=100 RUN=smoke_ha bash scripts/launch_ha_official_20k.sh
set -euo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
FLOWBODY=/inspire/hdd/global_user/liumengfan-253108110079/flowbody/FlowBody
GR00T="$FLOWBODY/Isaac-GR00T"
ROOTS="${ROOTS:-$COMIMIC/data/ha_sessions/roots.txt}"
RUN="${RUN:-groot_sonic_vla_20k_ha}"
OUT="${OUT:-$COMIMIC/runs/$RUN}"

# BASE 可以是 HF repo id，也可以是本地 checkpoint 目录（AutoModel.from_pretrained 两者都吃）：
# 续训时指向上一轮的 checkpoint 目录即可（不要用 --resume-from-checkpoint：那会恢复
# scheduler，LR 会停在 ~0，等于没空间）。
BASE="${BASE:-nvidia/GR00T-N1.7-3B}"
MAX_STEPS="${MAX_STEPS:-20000}"
NUM_GPUS="${NUM_GPUS:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"

[[ -s "$ROOTS" ]] || { echo "[fail] missing $ROOTS (run scripts/prepare_ha_sessions.py)"; exit 1; }
[[ -d "$GR00T/.venv" ]] || { echo "[fail] missing $GR00T/.venv"; exit 1; }

export PATH="$GR00T/.venv/bin:$PATH"
export VIRTUAL_ENV="$GR00T/.venv"
# ---- 执行器口径（与 WBC/闭环一致；VLA 训练本身不用，但下游联合微调/评测会读）------
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$COMIMIC/trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd}

export HF_HOME=/inspire/hdd/global_user/liumengfan-253108110079/cache/huggingface
# NOTE: keep HF_HUB_OFFLINE unset. With it set, transformers tries model_info() and dies;
# the cache is complete, so a metadata lookup is harmless (see flowbody docs 2026-10-23).
export TRANSFORMERS_OFFLINE=0
export NO_ALBUMENTATIONS_UPDATE=1
export USE_WANDB=0
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT" "$FLOWBODY/logs"
LOG="${LOG:-$FLOWBODY/logs/${RUN}.log}"
echo "[info] roots=$(tr ':' '\n' < "$ROOTS" | wc -l) steps=$MAX_STEPS gpus=$NUM_GPUS out=$OUT" | tee -a "$LOG"

cd "$GR00T"
exec env \
  NUM_GPUS="$NUM_GPUS" \
  GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
  MAX_STEPS="$MAX_STEPS" \
  SAVE_STEPS="$SAVE_STEPS" \
  DATALOADER_NUM_WORKERS="$DATALOADER_NUM_WORKERS" \
  SHARD_SIZE=1024 \
  NUM_SHARDS_PER_EPOCH=100000 \
  EPISODE_SAMPLING_RATE="$EPISODE_SAMPLING_RATE" \
  bash examples/finetune.sh \
    --base-model-path "$BASE" \
    --dataset-path "$(cat "$ROOTS")" \
    --embodiment-tag UNITREE_G1_SONIC \
    --output-dir "$OUT" 2>&1 | tee -a "$LOG"
