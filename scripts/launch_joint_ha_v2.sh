#!/usr/bin/env bash
# 联合微调 v2（S3 阶段 C）启动器。
#
# 用法：
#   LAMBDA_R=0    RUN=s3_lamr000 bash scripts/launch_joint_ha_v2.sh   # 对照臂（只 BC）
#   LAMBDA_R=0.3  RUN=s3_lamr030 bash scripts/launch_joint_ha_v2.sh
#   STEPS=3 LAMBDA_R=0.3 RUN=s3_smoke bash scripts/launch_joint_ha_v2.sh   # smoke
#
# 关键环境变量：
#   WBC_CKPT  WBC_v2（默认 wbc_mix2_obs93/ckpt_016000_spec.pt，obs93 + 混合库共训）
#   VLM_CKPT  起始 VLA（默认官方 checkpoint-8000；S2 的 base_aligned 出来后换成它）
#   BATCH / ACCUM / H / K   batch-size / grad-accum / 想象步数 / FM 步数
#   LAMBDA_R  奖励项权重（扫 0 / 0.1 / 0.3 / 1.0）
set -euo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$COMIMIC/trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd}

WBC_CKPT=${WBC_CKPT:-$COMIMIC/trackerlab-supermimic/logs/wbc_mix2_obs93/ckpt_016000_spec.pt}
VLM_CKPT=${VLM_CKPT:-$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}
VLM_PATH=${VLM_PATH:-$COMIMIC/groot_repro/nvidia/Cosmos-Reason2-2B-weights}
MOTION=${MOTION:-$COMIMIC/data/npz/ha_tokens}
PORT_DIR=${PORT_DIR:-$COMIMIC/data/port_tokens/ha_tokens}

RUN=${RUN:-joint_v2_lamr030}
OUT=${OUT:-$COMIMIC/runs/$RUN}
STEPS=${STEPS:-2000}
LAMBDA_R=${LAMBDA_R:-0.3}
LAMBDA_BC=${LAMBDA_BC:-1.0}
LAMBDA_S=${LAMBDA_S:-0}
LAMBDA_A=${LAMBDA_A:-0}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-4}
H=${H:-20}
K=${K:-4}
BURNIN=${BURNIN:-10}
EVAL_EVERY=${EVAL_EVERY:-250}
EVAL_BATCHES=${EVAL_BATCHES:-16}
SAVE_EVERY=${SAVE_EVERY:-500}
GPU=${GPU:-0}
BACKEND=${BACKEND:-pathwise}
EXTRA=${EXTRA:-}

for p in "$WBC_CKPT" "$VLM_CKPT" "$VLM_PATH" "$MOTION" "$PORT_DIR"; do
  [[ -e "$p" ]] || { echo "[fail] missing $p"; exit 1; }
done
case "$MOTION" in
  *mixed*|*ft318*) echo "[fail] 只允许纯 HA 数据，拒绝: $MOTION"; exit 1 ;;
esac

mkdir -p "$OUT" "$COMIMIC/logs"
LOG="$COMIMIC/logs/$RUN.log"
echo "[launch-v2] out=$OUT lambda_r=$LAMBDA_R H=$H K=$K batch=$BATCH accum=$ACCUM steps=$STEPS"
echo "[launch-v2] wbc=$WBC_CKPT"
echo "[launch-v2] vlm=$VLM_CKPT"

export PYTHONPATH="$COMIMIC:$COMIMIC/Isaac-GR00T:$COMIMIC/third_party/gr00t_deps:$COMIMIC/third_party/gr00t_deps/_noflash${PYTHONPATH:+:$PYTHONPATH}"
export NO_ALBUMENTATIONS_UPDATE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

exec python3 "$COMIMIC/scripts/train_joint_v2.py" \
  --motion "$MOTION" --port-tokens "$PORT_DIR" \
  --wbc-ckpt "$WBC_CKPT" --vlm-ckpt "$VLM_CKPT" --vlm-path "$VLM_PATH" \
  --device "cuda:$GPU" --out "$OUT" \
  --steps "$STEPS" --batch-size "$BATCH" --grad-accum "$ACCUM" \
  --horizon "$H" --fm-steps "$K" --burnin-steps "$BURNIN" \
  --lambda-r "$LAMBDA_R" --lambda-bc "$LAMBDA_BC" \
  --lambda-s "$LAMBDA_S" --lambda-a "$LAMBDA_A" \
  --fm-grad-backend "$BACKEND" \
  --eval-every "$EVAL_EVERY" --eval-batches "$EVAL_BATCHES" \
  --eval-out "$OUT/eval_curve.jsonl" \
  --save-every "$SAVE_EVERY" --save-vla-every "$SAVE_EVERY" \
  --log-every "${LOG_EVERY:-10}" $EXTRA \
  2>&1 | tee -a "$LOG"
