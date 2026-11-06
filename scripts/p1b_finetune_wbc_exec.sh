#!/usr/bin/env bash
# P1b（2 卡容器）：用**执行参考库**训练库微调 WBC，产出 runs/S1_handover/exp_WBC_exec.pt
#
# 文档：docs/PLAN_REFSTART_WM_PERTURB_20261102.md §6.2
#
# 为什么是 warm start（RESUME）而不是从 0 训 4000 步：
#   新库 143,923 clips 里 79% 的帧（bones_seed 47.5 M）与旧库**逐位相同**，只有 HA 那 21%
#   从"命令 q"换成"执行 q"。4000 步从 0 训连 bones 步态都学不回来，G1（<0.1 m / 0 摔倒）
#   必挂。"微调 WBC" 本来就该接着 C6h 线（WBC_A24000_wmvel_v6.pt，iteration 24000）往下走。
#
#   ITERATIONS=28000 是**绝对目标步数**（resume 语义），即 24000 → 28000，共 4000 步。
#   ckpt 形状 = ckpt_028000.pt（pi/wm/robot_spec/args/provenance），与 WBC_A24000_wmvel_v6.pt 同构。
#
# 用法：
#   MODE=smoke bash scripts/p1b_finetune_wbc_exec.sh     # 24000 -> 24020，20 步验配置
#   MODE=full  bash scripts/p1b_finetune_wbc_exec.sh     # 24000 -> 28000，4000 步（约 1.5 h）
#
# 可覆盖：NPROC ENVS_PER_RANK LR CKPT_EVERY FREEZE_WM COMPILE RESUME TAG
set -uo pipefail

C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
R=$C/trackerlab-supermimic
cd "$R" || exit 1

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SONIC_ENCODER_ONNX=${SONIC_ENCODER_ONNX:-$C/../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

MODE=${MODE:-smoke}
TAG=${TAG:-}
NPROC=${NPROC:-2}
ENVS_PER_RANK=${ENVS_PER_RANK:-4096}
LR=${LR:-2e-05}
CKPT_EVERY=${CKPT_EVERY:-500}
COMPILE=${COMPILE:-0}
FREEZE_WM=${FREEZE_WM:-0}
INTERFACE=${INTERFACE:-token}
OBS_MODE=${OBS_MODE:-paper_obs93}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-0}
PORT_BASE=${PORT_BASE:-29920}
LIBRARY=${LIBRARY:-$R/data/bones_seed/bones_seed_plus_ha_exec_x20.pt}
RESUME=${RESUME:-$C/runs/S1_handover/WBC_A24000_wmvel_v6.pt}
ROBOT_USD=${ROBOT_USD:-$C/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd}
OUT_CKPT=${OUT_CKPT:-$C/runs/S1_handover/exp_WBC_exec.pt}
# 非空 = 双库 token：ref 用 LIBRARY（执行参考），token 仍由这个**命令流**库在线算。
# 部署时 VLA 输出的是命令空间 label，所以这条才是"能接 VLA"的配方。
TOKEN_LIB_TRAIN=${TOKEN_LIB_TRAIN:-}

BASE=exp_wbc_exec${TAG:+_$TAG}
case "$MODE" in
  smoke) ITER=24020; LOG_DIR=${LOG_DIR:-$R/logs/${BASE}_smoke}; LAUNCH=$C/logs/p1b/${BASE}_smoke.launcher.log
         OUT_CKPT=${OUT_CKPT_OVERRIDE:-$C/runs/S1_handover/exp_WBC_exec_smoke24020.pt};;
  full)  ITER=28000; LOG_DIR=${LOG_DIR:-$R/logs/$BASE};            LAUNCH=$C/logs/p1b/${BASE}.launcher.log;;
  *) echo "MODE must be smoke|full"; exit 1;;
esac
mkdir -p "$C/logs/p1b"

for f in "$LIBRARY" "$RESUME" "$ROBOT_USD" "$SONIC_ENCODER_ONNX" ${TOKEN_LIB_TRAIN:+"$TOKEN_LIB_TRAIN"}; do
  [[ -f "$f" ]] || { echo "FATAL: missing $f"; exit 1; }
done
if [[ -e "$LOG_DIR" ]]; then echo "FATAL: $LOG_DIR 已存在（resume 必须用新 log_dir）"; exit 1; fi

echo "[p1b] mode=$MODE iter 24000 -> $ITER | nproc=$NPROC envs/rank=$ENVS_PER_RANK lr=$LR freeze_wm=$FREEZE_WM compile=$COMPILE"
echo "[p1b] library=$LIBRARY"
echo "[p1b] resume=$RESUME"
echo "[p1b] token_lib_train=${TOKEN_LIB_TRAIN:-<same as library (exec tokens)>}"
echo "[p1b] log_dir=$LOG_DIR launcher=$LAUNCH out_ckpt=$OUT_CKPT"

ROBOT_USD=$ROBOT_USD \
RESUME=$RESUME \
SM_TOKEN_LIB_TRAIN=$TOKEN_LIB_TRAIN \
OBS_MODE=$OBS_MODE FREEZE_WM=$FREEZE_WM NO_CRITIC=1 WM_ONLY_ITERS=0 LR=$LR \
MODE=train NPROC=$NPROC ENVS_PER_RANK=$ENVS_PER_RANK MAX_EPISODE_STEPS=$MAX_EPISODE_STEPS \
ITERATIONS=$ITER \
LIBRARY=$LIBRARY INTERFACE=$INTERFACE COMPILE=$COMPILE HA_ALIGN=1 \
CKPT_EVERY=$CKPT_EVERY PORT_BASE=$PORT_BASE LOG_DIR=$LOG_DIR \
  bash scripts/rls/supermimic/train_bones_seed.sh > "$LAUNCH" 2>&1
rc=$?
echo "[p1b] train.py exit=$rc"
[[ -f "$LOG_DIR/run_summary.json" ]] && cat "$LOG_DIR/run_summary.json"

CK=$LOG_DIR/ckpt_0${ITER}.pt
if [[ $rc -eq 0 && -f "$CK" ]]; then
  mkdir -p "$(dirname "$OUT_CKPT")"
  cp "$CK" "$OUT_CKPT"
  echo "[p1b] deliverable -> $OUT_CKPT ($(du -h "$OUT_CKPT" | cut -f1))"
  python3 - "$OUT_CKPT" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("[p1b] ckpt keys:", sorted(ck.keys()))
print("[p1b] iteration:", ck.get("iteration"), "| robot_spec.usd:", ck.get("robot_spec", {}).get("usd"))
print("[p1b] provenance:", {k: ck.get("provenance", {}).get(k) for k in
                            ("library", "obs_mode", "interface", "iteration", "lr", "freeze_wm")})
PY
else
  echo "[p1b] ckpt 未产出（rc=$rc）；见 $LAUNCH"
  exit 1
fi
