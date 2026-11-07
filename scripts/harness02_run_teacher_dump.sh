#!/usr/bin/env bash
# harness02：用**生产训练入口**跑几步，把 Teacher 标签/窗口原样落盘（J1_TEACHER_DUMP）。
#
# 为什么不在这个脚本里重写训练循环：只有走 train_joint_v2.py 才能保证
#   batch 的 state/reference/xi/prev_action/history = 生产分布（burn-in + DR + recorded start），
#   token chunk = 同一次冻结 VLA 前向的输出，Teacher 的 J = 主训练同一个 discounted_return。
#
# 口径与 runs/gt7fix 逐项一致（唯一差别：batch 32、单卡、8 步、grad_target_n=batch）。
# 环境变量：
#   RUN=dump_h20 STEP_N=8 BATCH=32 GPU=1
#   GRAD_TARGET_COMPARE_OLD=1 J1_TEACHER_EXTRA_H=5 J1_TEACHER_DUMP=<dir>
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export COMIMIC="$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/deps_env.sh"
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$ROOT/trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd}

RUN=${RUN:-dump_h20}
OUT=${OUT:-$ROOT/runs/harness_02_teacher/$RUN}
STEPS=${STEPS:-8}
BATCH=${BATCH:-32}
GPU=${GPU:-1}
H=${H:-20}
GT_HORIZON=${GT_HORIZON:-20}
SEED=${SEED:-0}
INIT_VLA=${INIT_VLA:-}
LAMBDA_R=${LAMBDA_R:-0}
ACT_WEIGHT=${ACT_WEIGHT:-0.15}
ACT_EPS=${ACT_EPS:-0.3}
TRUST_EPS=${TRUST_EPS:-0.05}
TRUST_WEIGHT=${TRUST_WEIGHT:-0.2}
VAR_WEIGHT=${VAR_WEIGHT:-0}
GT_WEIGHT=${GT_WEIGHT:-1.0}
GT_N=${GT_N:-$BATCH}
NO_CRITIC=${NO_CRITIC:-0}
WINDOW_NORM=${WINDOW_NORM:-0}
_DT_EXTRA=()
[[ "$WINDOW_NORM" == "1" ]] && _DT_EXTRA+=(--window-norm)
_CRITIC_ARG=(--critic --critic-source imagined)
[[ "$NO_CRITIC" == "1" ]] && _CRITIC_ARG=(--no-critic)

WBC_CKPT=$ROOT/runs/S1_handover/WBC_A24000_wmvel_cmd.pt
VLM_CKPT=$ROOT/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000
VLM_PATH=$ROOT/groot_repro/nvidia/Cosmos-Reason2-2B-weights
WM_CKPT=$ROOT/runs/a_prime_sonic_exc/wm_gain2.pt
PACKED=$ROOT/runs/isaac_motion/ha_cmd/packed.pt
TRANS=$ROOT/runs/a_prime_sonic_exc/all_transitions.pt
MOTION=$ROOT/data/npz/ha_tokens

for p in "$WBC_CKPT" "$VLM_CKPT" "$VLM_PATH" "$WM_CKPT" "$PACKED" "$TRANS" "$MOTION"; do
  [[ -e "$p" ]] || { echo "[fail] missing $p"; exit 1; }
done
mkdir -p "$OUT"

_INIT_ARG=()
[[ -n "$INIT_VLA" ]] && _INIT_ARG=(--init-vla "$INIT_VLA")

set -x
python3 "$ROOT/scripts/train_joint_v2.py" \
  --motion "$MOTION" --motion-packed "$PACKED" \
  --wbc-ckpt "$WBC_CKPT" --vlm-ckpt "$VLM_CKPT" --vlm-path "$VLM_PATH" --wm-ckpt "$WM_CKPT" \
  --real-transitions "$TRANS" \
  --device "cuda:$GPU" --out "$OUT" \
  --steps "$STEPS" --batch-size "$BATCH" --grad-accum 1 --seed "$SEED" \
  --horizon "$H" --fm-steps 4 --burnin-steps 10 \
  --lambda-r "$LAMBDA_R" --lambda-bc 0 --lambda-s 0 --lambda-a 0 \
  --bc-token-space label --v3 \
  --trust-center base --trust-eps "$TRUST_EPS" --trust-weight "$TRUST_WEIGHT" \
  --act-eps "$ACT_EPS" --act-weight "$ACT_WEIGHT" --act-penalty hinge --reward-horizon 0 \
  --var-weight "$VAR_WEIGHT" "${_DT_EXTRA[@]}" "${_CRITIC_ARG[@]}" \
  --delta-token --delta-token-hidden 1024 --delta-token-depth 2 \
  --delta-token-init-std 4e-4 --delta-token-lr 5e-4 --delta-token-cap 0 \
  --delta-token-fullstate --freeze-vla \
  --grad-target-weight "$GT_WEIGHT" --grad-target-n "$GT_N" --grad-target-eta 0.08 \
  --grad-target-horizon "$GT_HORIZON" \
  --executor sonic --start-source recorded --start-max-drift 0.10 \
  --reward-mode saturating --huber-delta 0.02 --huber-cap 0.15 --reward-d-max 0.30 \
  --root-ori-mode split --tail-weight 2.0 --yaw-hinge-lambda 0.3 --term-off \
  --amp bf16 --eval-every 0 --eval-batches 1 --save-every 0 --save-vla-every 0 \
  --log-every 1 --fm-grad-backend pathwise --token-quant none \
  --weight-decay 1e-4 --obs-noise 0 --lambda-hand 0 --token-bc-weight 0 \
  "${_INIT_ARG[@]}" \
  "$@"
