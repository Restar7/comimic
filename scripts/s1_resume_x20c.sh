#!/usr/bin/env bash
# 从 SuperMimic "HA + hold" 线的 20000（logs/wbc_hold_x20b/last.pt）续训 WBC。
#
#   MODE=smoke bash scripts/s1_resume_x20c.sh      # 50 步连续性冒烟
#   MODE=full  bash scripts/s1_resume_x20c.sh      # 正式续训到 TARGET（默认 24000）
#
# 带手资产（A 方案：手指锁定张开，43 DoF = 29 身体 + 2x7 手指）：
#   ROBOT_USD=$PWD/../third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd \
#   TAG=hand MODE=full bash scripts/s1_resume_x20c.sh
#   → 策略动作仍是 29 维、观测仍是 93 维，手指由 `HAS_HANDS` 自动识别的独立 PD 组锁在 0（张开），
#     `HA_ALIGN` 自动开；网络形状不变，所以 --resume 直接续，不用从头训。
#
# 配方 = 20000 之前那一轮的同配方（hold 前端 / obs93 / no_critic / lr 2e-5 余弦 / 2×8192）。
# 训完仍要跑一次 800 步 ha_cmd 域 WM 标定，才能得到可部署的 "C6h′"：
#   python3 scripts/finetune_wm_velocity_policy.py --ckpt <新 ckpt> --out runs/S1_handover/WBC_A<it>_wmvel_cmd.pt \
#     --packed runs/isaac_motion/ha_cmd/packed.pt --token-space label --steps 800 --vel-lambda 5
#
# 依据：docs/RESUME_HA_HOLD_20000_20261027.md
set -uo pipefail

C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
R=$C/trackerlab-supermimic
cd "$R" || exit 1

export SONIC_ENCODER_ONNX=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODE=${MODE:-smoke}
TAG=${TAG:-}                     # 非空则 log_dir 后缀，如 TAG=hand → logs/wbc_hand_x20c
ROBOT_USD=${ROBOT_USD:-}         # 空 = 默认无手 29DoF 资产；给 dex3 路径 = 带手（手指锁定张开）
RESUME_CKPT=${RESUME_CKPT:-$R/logs/wbc_hold_x20b/last.pt}
TARGET=${TARGET:-24000}
PORT_BASE=${PORT_BASE:-29810}
NPROC=${NPROC:-2}
ENVS_PER_RANK=${ENVS_PER_RANK:-8192}
LR=${LR:-2e-5}
# B 方案（原配方）：WM 照常训练。A 方案（保 C6h 标定）需先拼 wm 件并置 FREEZE_WM=1。
FREEZE_WM=${FREEZE_WM:-0}
CKPT_EVERY=${CKPT_EVERY:-250}

BASE=wbc_${TAG:+${TAG}_}hold_x20c
case "$MODE" in
  smoke) ITER=20050; LOG_DIR=$R/logs/${BASE}_smoke; LAUNCH=$C/logs/s1/${BASE}_smoke.launcher.log ;;
  full)  ITER=$TARGET; LOG_DIR=$R/logs/$BASE;       LAUNCH=$C/logs/s1/${BASE}.launcher.log ;;
  *) echo "MODE must be smoke|full"; exit 1 ;;
esac

if [[ ! -f "$RESUME_CKPT" ]]; then echo "FATAL: resume ckpt missing: $RESUME_CKPT"; exit 1; fi
if [[ -e "$LOG_DIR" ]]; then echo "FATAL: $LOG_DIR 已存在（resume 必须用新 log_dir）"; exit 1; fi
[[ "$ITER" -gt 20000 ]] || { echo "FATAL: ITERATIONS 必须 > 20000（是绝对目标值）"; exit 1; }
if [[ -n "$ROBOT_USD" && ! -f "$ROBOT_USD" ]]; then echo "FATAL: ROBOT_USD 不存在: $ROBOT_USD"; exit 1; fi

mkdir -p "$C/logs/s1"
echo "[resume] mode=$MODE tag=${TAG:-none} resume=$RESUME_CKPT -> iter $ITER | freeze_wm=$FREEZE_WM lr=$LR port=$PORT_BASE"
echo "[resume] robot_usd=${ROBOT_USD:-<default handless 29DoF>}"
echo "[resume] log_dir=$LOG_DIR launcher_log=$LAUNCH"

ROBOT_USD=$ROBOT_USD \
RESUME=$RESUME_CKPT \
OBS_MODE=paper_obs93 FREEZE_WM=$FREEZE_WM NO_CRITIC=1 WM_ONLY_ITERS=0 LR=$LR \
MODE=train NPROC=$NPROC ENVS_PER_RANK=$ENVS_PER_RANK MAX_EPISODE_STEPS=0 \
ITERATIONS=$ITER \
LIBRARY=$R/data/bones_seed/bones_seed_plus_ha_cmd_x20.pt INTERFACE=token COMPILE=1 HA_ALIGN=1 \
CKPT_EVERY=$CKPT_EVERY PORT_BASE=$PORT_BASE LOG_DIR=$LOG_DIR \
  bash scripts/rls/supermimic/train_bones_seed.sh > "$LAUNCH" 2>&1
rc=$?
echo "[resume] train.py exit=$rc"
if [[ -f "$LOG_DIR/run_summary.json" ]]; then cat "$LOG_DIR/run_summary.json"; fi
exit $rc
