#!/usr/bin/env bash
# S2.2 — 官方 GR00T-N1.7 微调，数据 = **端口 token 重标注**后的 HA 会话树。
#
# 与 scripts/launch_ha_official_20k.sh 的差别（只差这三件事，其它配方相同）：
#   1. 数据根 = data/ha_sessions_port/roots_train.txt（action.motion_token 已换成端口 token）
#   2. GR00T 代码 = comimic/Isaac-GR00T（PyAV 视频回退 + OmegaConf 结构化配置补丁），
#      重依赖（torch 2.9 / deepspeed 0.17.6 / transformers 4.57.3）来自 flowbody venv 的
#      site-packages（该 venv 的 python 已用 `uv python install 3.12` 恢复）；
#   3. 直接调 gr00t/experiment/launch_finetune.py（comimic 的 examples/finetune.sh 把
#      --learning_rate 写死 1e-4，这个 launcher 显式给 1e-5）。
#
# 设计依据：docs/S2_BASE_ALIGNED_PLAN_20261026.md（F3/F5/D4）。
#
# 用法：
#   MAX_STEPS=2 RUN=smoke_port bash scripts/launch_ha_port_6000.sh      # 冒烟
#   bash scripts/launch_ha_port_6000.sh                                  # 6000 步正式
set -euo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
VENV=/inspire/hdd/global_user/liumengfan-253108110079/flowbody/FlowBody/Isaac-GR00T/.venv
DEPS="$COMIMIC/third_party/gr00t_deps"
GR00T="$COMIMIC/Isaac-GR00T"

BASE="${BASE:-$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}"
ROOTS="${ROOTS:-$COMIMIC/data/ha_sessions_port/roots_train.txt}"
RUN="${RUN:-groot_sonic_vla_ha_port}"
OUT="${OUT:-$COMIMIC/runs/$RUN}"
LOG="${LOG:-$COMIMIC/logs/$RUN.log}"

MAX_STEPS="${MAX_STEPS:-6000}"
NUM_GPUS="${NUM_GPUS:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
MASTER_PORT="${MASTER_PORT:-29617}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"

# 相对路径统一按 COMIMIC 解析（脚本会 cd 到 GR00T，相对路径会漂）
for _v in BASE ROOTS OUT LOG; do
  eval "_p=\${$_v}"
  [[ "$_p" = /* ]] || eval "$_v=\"\$COMIMIC/\$_p\""
done
unset _v _p

for p in "$BASE" "$ROOTS" "$GR00T/gr00t/experiment/launch_finetune.py" "$VENV/lib/python3.12/site-packages"; do
  [[ -e "$p" ]] || { echo "[fail] missing $p" >&2; exit 1; }
done
[[ -s "$ROOTS" ]] || { echo "[fail] empty roots file $ROOTS" >&2; exit 1; }
case "$ROOTS" in
  *ha_sessions_port/roots_train.txt) ;;
  *) echo "[fail] 本 launcher 只接受端口重标注的数据根（$ROOTS）" >&2; exit 1 ;;
esac
# 端口数据不能有 label 空间残留：抽查 roots 里的第一个 session stats
python3 - "$ROOTS" <<'PY' || exit 1
import json, sys
from pathlib import Path
roots = [p for p in Path(sys.argv[1]).read_text().strip().split(":") if p]
st = json.loads((Path(roots[0]) / "meta" / "stats.json").read_text())["action.motion_token"]
q01, q99 = st["q01"][:2], st["q99"][:2]
ok = all(abs(v - round(v * 16) / 16) < 1e-9 for v in q01 + q99)
print(f"[audit][data] roots={len(roots)} q01[:2]={q01} q99[:2]={q99} quantized={ok}")
PY

export PYTHONPATH="$GR00T:$VENV/lib/python3.12/site-packages:$DEPS:$DEPS/_noflash${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/inspire/hdd/global_user/liumengfan-253108110079/cache/huggingface
export TRANSFORMERS_OFFLINE=0
export NO_ALBUMENTATIONS_UPDATE=1
export USE_WANDB=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT" "$(dirname "$LOG")"
{
  echo "=== [AUDIT][s2] $(date -Is) run=$RUN"
  echo "    base=$BASE"
  echo "    roots=$ROOTS"
  echo "    out=$OUT log=$LOG"
  echo "    max_steps=$MAX_STEPS gpus=$NUM_GPUS global_batch=$GLOBAL_BATCH_SIZE "
  echo "    lr=$LEARNING_RATE warmup=$WARMUP_RATIO wd=$WEIGHT_DECAY save=$SAVE_STEPS limit=$SAVE_TOTAL_LIMIT"
  echo "    episode_sampling_rate=$EPISODE_SAMPLING_RATE shard=$SHARD_SIZE shards_per_epoch=$NUM_SHARDS_PER_EPOCH"
  echo "    gr00t=$GR00T (comimic 补丁版)  deps=venv(site-packages)+gr00t_deps"
  echo "    git_head=$(git -C "$COMIMIC" rev-parse HEAD 2>/dev/null || echo unknown)"
} | tee -a "$LOG"

cd "$GR00T"
exec python3 -m torch.distributed.run \
  --nproc_per_node "$NUM_GPUS" --master_port "$MASTER_PORT" \
  gr00t/experiment/launch_finetune.py \
    --base_model_path "$BASE" \
    --dataset_path "$(cat "$ROOTS")" \
    --embodiment_tag UNITREE_G1_SONIC \
    --output_dir "$OUT" \
    --num_gpus "$NUM_GPUS" \
    --max_steps "$MAX_STEPS" \
    --global_batch_size "$GLOBAL_BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_ratio "$WARMUP_RATIO" \
    --weight_decay "$WEIGHT_DECAY" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --episode_sampling_rate "$EPISODE_SAMPLING_RATE" \
    --shard_size "$SHARD_SIZE" \
    --num_shards_per_epoch "$NUM_SHARDS_PER_EPOCH" \
    "$@" 2>&1 | tee -a "$LOG"
