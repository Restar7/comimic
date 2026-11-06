#!/usr/bin/env bash
# ============================================================================
# **锁定方案**启动器（2026-11-01 起）：H=20 + λ_r=0.3 + act_weight=0.5
#
# 依据与判据：docs/METHOD_LOCK_H20_20261101.md
#   * 200 次 optimizer updates × 有效 batch 64（与 d5jT/d5jR/d5jG 严格等价）
#   * 8 卡 DDP：NPROC=8, BATCH=64, ACCUM=1（每卡 8）
#   * 评测：4 条 sweep 占满 8 卡，默认 32 集配对（论文口径用 EVAL_SEEDS=48）
#
# 用法：
#   # 消融轮：固定 WM，只验证策略（当前默认）
#   ROUND=h20r2 MODE=ablation \
#     INIT_VLA=$PWD/runs/S3_loop/h20_8g_b_joint/train_main/vla_final.pt \
#     bash scripts/run_locked_h20_round.sh
#
#   # 完整轮：用当前策略采 on-policy 数据 → 刷 WM → joint → eval
#   ROUND=h20r3 MODE=full COLLECT_EPISODES=7 \
#     INIT_VLA=$PWD/runs/S3_loop/h20_8g_b_joint/train_main/vla_final.pt \
#     bash scripts/run_locked_h20_round.sh
# ============================================================================
set -uo pipefail

COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1

ROUND=${ROUND:?必须给 ROUND（例如 h20r2）}
MODE=${MODE:-ablation}
INIT_VLA=${INIT_VLA:-$COMIMIC/runs/S3_loop/h20_8g_b_joint/train_main/vla_final.pt}
WM_PIN=${WM_PIN:-$COMIMIC/runs/S1_handover/wm_d5_k20.pt}
NPROC=${NPROC:-8}
BATCH=${BATCH:-64}
ACCUM=${ACCUM:-1}
STEPS=${STEPS:-200}
EVAL_SEEDS=${EVAL_SEEDS:-32}
# ★ 2026-11-01（用户口径）：**每轮的主对照是上一轮最优，不是 d2b/π0。**
#   优先级：显式 BASELINE_HEAD > PREV_ROUND > 最近一个 *_joint/train_main/vla_final.pt（排除本轮）。
#   d2b/π0 只作为锚点/ sanity check，不再当主读数。
if [[ -z "${BASELINE_HEAD:-}" ]]; then
  if [[ -n "${PREV_ROUND:-}" ]]; then
    BASELINE_HEAD="$COMIMIC/runs/S3_loop/${PREV_ROUND}_joint/train_main/vla_final.pt"
    BASELINE_LABEL=${BASELINE_LABEL:-$PREV_ROUND}
  else
    BASELINE_HEAD=$(ls -t "$COMIMIC"/runs/S3_loop/*_joint/train_main/vla_final.pt 2>/dev/null \
      | grep -vE "/${ROUND}_joint/|_smoke_joint/|_wbc_kl_joint/|_wbc_joint/|_jointwbc_joint/" | head -1)
    [[ -n "$BASELINE_HEAD" ]] || {
      echo "[fail] 找不到上一轮 vla_final.pt；请给 BASELINE_HEAD 或 PREV_ROUND"; exit 1; }
    _prev=$(basename "$(dirname "$(dirname "$BASELINE_HEAD")")")   # 例如 h20r2_joint
    BASELINE_LABEL=${BASELINE_LABEL:-${_prev%_joint}}
  fi
fi
BASELINE_LABEL=${BASELINE_LABEL:-prev}
TAG_PREFIX=${TAG_PREFIX:-${BASELINE_LABEL}_vs_${ROUND}}
MASTER_PORT=${MASTER_PORT:-29600}
BASE_PORT=${BASE_PORT:-18150}
RUN_TAG=${RUN_TAG:-$ROUND}
B_GLOB=${B_GLOB:-"runs/B_*/vla_*_f*_r*/dump/*.pt"}
DRY_RUN=${DRY_RUN:-0}
# ★ 2026-11-01：stage 8.5（P2）已接进启动器；支持两条路：
#   WBC_MODE=plain    → Path A：VLA 冻结（FM_LR=0），只训 WBC，无 KL；
#   WBC_MODE=joint_kl → Path B：VLA head + WBC 联合，WBC action KL（VLA token KL 可选）。
#   MODE=full 默认开 WITH_WBC=1；MODE=p2 只跑 8.5（用于同一轮的 A/B 两路并行）。
if [[ "$MODE" == "full" || "$MODE" == "p2" ]]; then
  WITH_WBC=${WITH_WBC:-1}
else
  WITH_WBC=${WITH_WBC:-0}
fi
WBC_MODE=${WBC_MODE:-plain}
case "$WBC_MODE" in plain|joint_kl) ;; *) echo "[fail] WBC_MODE 只允许 plain|joint_kl"; exit 1 ;; esac
WBC_STEPS=${WBC_STEPS:-200}
WBC_LR=${WBC_LR:-1e-6}
WBC_MASTER_PORT=${WBC_MASTER_PORT:-29611}
# P2 两条路并行时：A 用 0-3、B 用 4-7；各自 MASTER_PORT 不同，评测可先 SKIP 后串行。
P2_NPROC=${P2_NPROC:-8}
P2_DDP_GPUS=${P2_DDP_GPUS:-}
SKIP_P2_EVAL=${SKIP_P2_EVAL:-0}
P2_BASE_PORT=${P2_BASE_PORT:-$((BASE_PORT + 40))}
P2_GPU_OFFSET=${P2_GPU_OFFSET:-0}
P2_SWEEPS=${P2_SWEEPS:-4}
JOINT_FM_LR=${JOINT_FM_LR:-1e-5}
JOINT_WBC_LR=${JOINT_WBC_LR:-5e-7}
WBC_KL_ANCHOR=${WBC_KL_ANCHOR:-0.5}
VLA_KL_ANCHOR=${VLA_KL_ANCHOR:-0}

[[ -f "$INIT_VLA" ]] || { echo "[fail] INIT_VLA 不存在: $INIT_VLA"; exit 1; }
[[ -f "$BASELINE_HEAD" ]] || { echo "[fail] BASELINE_HEAD 不存在: $BASELINE_HEAD"; exit 1; }
(( EVAL_SEEDS % 4 == 0 )) || { echo "[fail] EVAL_SEEDS 必须能被 4 整除（4 条 sweep）"; exit 1; }
case "$MODE" in ablation|full|p2) ;; *) echo "[fail] MODE 只允许 ablation|full|p2"; exit 1 ;; esac
if [[ "$MODE" == "ablation" ]]; then
  [[ -f "$WM_PIN" ]] || { echo "[fail] WM_PIN 不存在: $WM_PIN"; exit 1; }
elif [[ "$MODE" == "p2" ]]; then
  :   # P2-only：VLA/WM 从已有轮次读取；INIT_VLA/WM_PIN 只作后备
elif [[ "$DRY_RUN" == "1" ]]; then
  echo "[fail] 完整轮的 DRY_RUN 校验尚未接线；请用 MODE=ablation DRY_RUN=1"
  exit 1
fi

echo "======================================================================"
echo "[locked] ROUND=$ROUND MODE=$MODE"
echo "[locked] H=20 λ_r=0.3 act_w=0.5 act_eps=0.6  updates=$STEPS  batch=$BATCH×accum=$ACCUM  NPROC=$NPROC"
echo "[locked] init=$INIT_VLA"
echo "[locked] baseline=$BASELINE_LABEL ($BASELINE_HEAD)"
echo "[locked] eval=${EVAL_SEEDS} 集配对，4 条 sweep"
echo "======================================================================"

COMMON_ENV=(
  H=20 LAMBDA_R=0.3 ACT_WEIGHT=0.5 ACT_EPS=0.6
  FM_LR=3e-5 YAW_HINGE_LAMBDA=0
  REWARD_ROOT_POS_STD=0.2 REWARD_ROOT_ORI_STD=0.25 REWARD_ROOT_WEIGHT=1.0
  CRITIC_ON_MAIN=1 CRITIC_GRAD_STATE_MAIN=1 CRITIC_STATE_GRAD_SCALE=0.1
  TRAIN_ARMS=main STEPS="$STEPS" BATCH="$BATCH" ACCUM="$ACCUM"
  INIT_VLA="$INIT_VLA" RUN_TAG="$RUN_TAG" RUN_LABEL="$ROUND"
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
)

if [[ "$MODE" == "ablation" ]]; then
  echo "[locked] 预生成 self 目标（只算一次，两条并行臂也安全）"
  PYTHONPATH=. python3 "$COMIMIC/scripts/build_b_self_targets.py" \
    --dump "$B_GLOB" --ref "$COMIMIC/runs/isaac_motion/ha_cmd/packed.pt" \
    --out "$COMIMIC/runs/wm_real/self_targets_all.pt" --horizon 20 \
    > "$COMIMIC/logs/${ROUND}_self_targets.log" 2>&1 \
    || { echo "[fail] self 目标生成失败（看 logs/${ROUND}_self_targets.log）"; exit 1; }

  _DDP_GPUS_ARG=()
  if (( NPROC < 8 )); then
    _DDP_GPUS_ARG=(DDP_GPUS="$(seq -s, 0 $((NPROC - 1)))")
  fi
  env "${COMMON_ENV[@]}" "${_DDP_GPUS_ARG[@]}" \
    ROUND="$ROUND" ABL_ROUND=1 DDP=1 NPROC="$NPROC" MASTER_PORT="$MASTER_PORT" \
    WM_PIN="$WM_PIN" SKIP_SELF_TARGETS=1 DRY_RUN="$DRY_RUN" \
    B_FRESH_GLOB="$COMIMIC/runs/B_d3/*/dump/*.pt" B_FRESH_PIN=1 \
    bash "$COMIMIC/scripts/s3_pipeline.sh" joint \
    2>&1 | tee "$COMIMIC/logs/${ROUND}_joint_driver.log" | tail -6
  [[ "${PIPESTATUS[0]}" == "0" ]] || { echo "[fail] joint 失败（看 logs/${ROUND}_joint_driver.log）"; exit 1; }
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[locked] DRY_RUN ✅ 环境链/断言校验通过（未训练、未评测）"
    exit 0
  fi
elif [[ "$MODE" == "full" ]]; then
  # 完整轮：collect → score → transitions → wm_refresh → wm_gate → critic → joint → eval → report
  # ★ CURRICULUM=1：不冻结锚点课程，按 ACT_WEIGHT_SCHEDULE / 上轮 frac_out 决定 act_weight。
  #   默认 CURRICULUM=0（ABL_ROUND=1，act_weight 固定在 COMMON_ENV 的 0.5）。
  _ABL_ENV=(ABL_ROUND=1)
  [[ "${CURRICULUM:-0}" == "1" ]] && _ABL_ENV=()
  _BASE_WBC="${BASE_WM:-$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt}"
  env "${COMMON_ENV[@]}" "${_ABL_ENV[@]}" \
    ROUND="$ROUND" \
    DDP=1 NPROC="${NPROC:-8}" MASTER_PORT="${MASTER_PORT:-29600}" \
    SM_CKPT="${SM_CKPT:-$_BASE_WBC}" \
    ACT_WEIGHT_SCHEDULE="${ACT_WEIGHT_SCHEDULE:-}" \
    YAW_RATE_WEIGHT_OVERRIDE="${YAW_RATE_WEIGHT_OVERRIDE:-}" \
    HEAD_COLLECT="$INIT_VLA" \
    COLLECT_JOBS="${COLLECT_JOBS:-8}" COLLECT_NUM_GPUS="${COLLECT_NUM_GPUS:-8}" \
    COLLECT_EPISODES="${COLLECT_EPISODES:-7}" \
    B_GLOB="$B_GLOB" \
    bash "$COMIMIC/scripts/s3_pipeline.sh" all \
    2>&1 | tee "$COMIMIC/logs/${ROUND}_full_driver.log" | tail -8
  [[ "${PIPESTATUS[0]}" == "0" ]] || { echo "[fail] 完整轮失败（看 logs/${ROUND}_full_driver.log）"; exit 1; }
else
  echo "[locked] MODE=p2：跳过主轮训练/主评测，只跑 stage 8.5（$WBC_MODE）"
fi

if [[ "$MODE" != "p2" ]]; then
HEAD_CTRL="$COMIMIC/runs/S3_loop/${ROUND}_joint/train_main/vla_final.pt"
[[ -f "$HEAD_CTRL" ]] || { echo "[fail] 没有产出 $HEAD_CTRL"; exit 1; }

echo "[locked] === 闭环评测：${BASELINE_LABEL} vs ${ROUND}，${EVAL_SEEDS} 集 ==="
HEAD_MAIN="$BASELINE_HEAD" HEAD_CTRL="$HEAD_CTRL" \
  SM_CKPT_MAIN="${EVAL_WBC_MAIN:-${BASE_WM:-$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt}}" \
  SM_CKPT_CTRL="${EVAL_WBC_CTRL:-${BASE_WM:-$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt}}" \
  TAG_PREFIX="$TAG_PREFIX" LABEL_MAIN="$BASELINE_LABEL" LABEL_CTRL="$ROUND" \
  SEED_TOTAL="$EVAL_SEEDS" N_SWEEPS=4 BASE_PORT="$BASE_PORT" GPU_OFFSET=0 \
  bash "$COMIMIC/scripts/eval_h20_8gpu.sh" \
  2>&1 | tee "$COMIMIC/logs/${ROUND}_eval_driver.log" | tail -20
[[ "${PIPESTATUS[0]}" == "0" ]] || { echo "[fail] 评测失败（看 logs/${ROUND}_eval_driver.log）"; exit 1; }
fi

if [[ "$WITH_WBC" == "1" ]]; then
  P2_VLA="$COMIMIC/runs/S3_loop/${ROUND}_joint/train_main/vla_final.pt"
  [[ -f "$P2_VLA" ]] || { echo "[fail] stage 8.5 缺 VLA: $P2_VLA"; exit 1; }
  # 两条路：plain = Path A（VLA 冻结、WBC-only、无 KL）；joint_kl = Path B（VLA+WBC 联合 + KL）。
  if [[ "$WBC_MODE" == "joint_kl" ]]; then
    P2_ROUND="${ROUND}_wbc_kl"
    P2_FM_LR="$JOINT_FM_LR"
    P2_WBC_LR="$JOINT_WBC_LR"
    P2_UNFREEZE="dec"
    P2_WBC_KL="$WBC_KL_ANCHOR"
    P2_VLA_KL="$VLA_KL_ANCHOR"
  else
    P2_ROUND="${ROUND}_wbc"
    P2_FM_LR=0
    P2_WBC_LR="$WBC_LR"
    P2_UNFREEZE="all"
    P2_WBC_KL=0
    P2_VLA_KL=0
  fi
  P2_JOINT="$COMIMIC/runs/S3_loop/${P2_ROUND}_joint/train_main/joint_final.pt"

  # WM：有本轮 state.gate 就用刷新 WM；否则用 WM_PIN（消融/手动）。
  P2_WM=""
  if [[ -f "$COMIMIC/runs/S3_pipeline/$ROUND/state.json" ]]; then
    P2_WM=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['stages']['gate']['artifact'])" \
      "$COMIMIC/runs/S3_pipeline/$ROUND/state.json" 2>/dev/null || true)
  fi
  [[ -n "$P2_WM" ]] || P2_WM="$WM_PIN"
  [[ -n "$P2_WM" && -f "$P2_WM" ]] || { echo "[fail] stage 8.5 缺 WM（state.gate 或 WM_PIN）"; exit 1; }

  if [[ -d "$COMIMIC/runs/B_${ROUND}" ]]; then
    P2_FRESH="$COMIMIC/runs/B_${ROUND}/*/dump/*.pt"
  else
    P2_FRESH="$COMIMIC/runs/B_d3/*/dump/*.pt"
  fi
  BASE_EXECUTOR=${EXECUTOR:-${BASE_WM:-${WM_PIN:-$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt}}}
  echo "[locked] === stage 8.5 [$WBC_MODE] $P2_ROUND（$WBC_STEPS 步；FM_LR=$P2_FM_LR WBC_LR=$P2_WBC_LR；unfreeze=$P2_UNFREEZE；kl=$P2_WBC_KL/$P2_VLA_KL）==="
  _P2_DDP_ARGS=()
  [[ -n "$P2_DDP_GPUS" ]] && _P2_DDP_ARGS=(DDP_GPUS="$P2_DDP_GPUS")
  env ROUND="$P2_ROUND" ABL_ROUND=1 H=20 DDP=1 NPROC="$P2_NPROC" "${_P2_DDP_ARGS[@]}" MASTER_PORT="$WBC_MASTER_PORT" \
    WM_PIN="$P2_WM" INIT_VLA="$P2_VLA" \
    STEPS="$WBC_STEPS" EVAL_EVERY=0 FM_LR="$P2_FM_LR" \
    ACT_EPS=0.6 ACT_WEIGHT=0.5 LAMBDA_R=0.3 YAW_HINGE_LAMBDA=0 \
    REWARD_ROOT_POS_STD=0.2 REWARD_ROOT_ORI_STD=0.25 REWARD_ROOT_WEIGHT=1.0 \
    CRITIC_ON_MAIN=1 CRITIC_GRAD_STATE_MAIN=1 CRITIC_STATE_GRAD_SCALE=0.1 \
    TRAIN_WBC_MAIN=1 WBC_LR="$P2_WBC_LR" \
    WBC_UNFREEZE="$P2_UNFREEZE" WBC_KL_ANCHOR="$P2_WBC_KL" VLA_KL_ANCHOR="$P2_VLA_KL" \
    TRAIN_ARMS=main \
    RUN_TAG="$P2_ROUND" RUN_LABEL="$P2_ROUND" \
    B_GLOB="runs/B_*/vla_*_f*_r*/dump/*.pt" \
    B_FRESH_GLOB="$P2_FRESH" B_FRESH_PIN=1 SKIP_SELF_TARGETS=1 \
    BATCH=64 ACCUM=1 \
    bash "$COMIMIC/scripts/s3_pipeline.sh" joint \
    2>&1 | tee "$COMIMIC/logs/${P2_ROUND}_driver.log" | tail -6
  [[ "${PIPESTATUS[0]}" == "0" ]] || { echo "[fail] stage 8.5 训练失败（看 logs/${P2_ROUND}_driver.log）"; exit 1; }
  [[ -f "$P2_JOINT" ]] || { echo "[fail] stage 8.5 没产出 $P2_JOINT"; exit 1; }

  if [[ "$SKIP_P2_EVAL" == "1" ]]; then
    echo "[locked] SKIP_P2_EVAL=1 ⇒ 只训练、不评测：$P2_JOINT"
  else
  P2_TAG="${P2_ROUND}_vs_base"
  echo "[locked] === stage 8.5 评测：同一 VLA，base vs v_new（${EVAL_SEEDS} 集）==="
  HEAD_MAIN="$P2_VLA" HEAD_CTRL="$P2_VLA" \
    SM_CKPT_MAIN="$BASE_EXECUTOR" SM_CKPT_CTRL="$P2_JOINT" \
    TAG_PREFIX="$P2_TAG" LABEL_MAIN="${ROUND}_base" LABEL_CTRL="${P2_ROUND}" \
    SEED_TOTAL="$EVAL_SEEDS" N_SWEEPS="$P2_SWEEPS" BASE_PORT="$P2_BASE_PORT" GPU_OFFSET="$P2_GPU_OFFSET" \
    bash "$COMIMIC/scripts/eval_h20_8gpu.sh" \
    2>&1 | tee "$COMIMIC/logs/${P2_TAG}_eval.log" | tail -12
  if [[ "${PIPESTATUS[0]}" != "0" ]]; then
    echo "[locked][warn] stage 8.5 评测失败 ⇒ 沿用 base 执行器"
  fi
  P2_WINNER="$BASE_EXECUTOR"
  P2_WIN_LABEL="${ROUND}_base"
  if [[ -f "$COMIMIC/runs/S3_loop/sweeps/${P2_TAG}_merged.json" ]]; then
    _win=$(python3 - "$COMIMIC/runs/S3_loop/sweeps/${P2_TAG}_merged.json" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1]))
p = d["paired"]
# 保守判据：均值和中位数都要 < 0 才换执行器；只赢尾部（中位≈0）不换。
print("ctrl" if p["mean"] < 0 and p["median"] < 0 else "main")
PY
)
    [[ "$_win" == "ctrl" ]] && { P2_WINNER="$P2_JOINT"; P2_WIN_LABEL="$P2_ROUND"; }
  fi
  echo "[locked] stage 8.5 胜出执行器：$P2_WIN_LABEL"

  # Path B（联合）才需要再比策略；Path A 的 VLA 逐位未动，跳过省时间。
  if python3 -c "import sys; sys.exit(0 if float('$P2_FM_LR') > 0 else 1)"; then
    echo "[locked] === 策略评测：${BASELINE_LABEL} vs ${ROUND}（执行器 $P2_WIN_LABEL）==="
    HEAD_MAIN="$BASELINE_HEAD" HEAD_CTRL="$P2_VLA" \
      SM_CKPT_MAIN="$P2_WINNER" SM_CKPT_CTRL="$P2_WINNER" \
      TAG_PREFIX="${ROUND}_policy_vs_${BASELINE_LABEL}" \
      LABEL_MAIN="$BASELINE_LABEL" LABEL_CTRL="$ROUND" \
      SEED_TOTAL="$EVAL_SEEDS" N_SWEEPS="$P2_SWEEPS" BASE_PORT="$((P2_BASE_PORT + 20))" GPU_OFFSET="$P2_GPU_OFFSET" \
      bash "$COMIMIC/scripts/eval_h20_8gpu.sh" \
      2>&1 | tee "$COMIMIC/logs/${ROUND}_policy_eval.log" | tail -12
    [[ "${PIPESTATUS[0]}" == "0" ]] || echo "[locked][warn] 策略评测失败（看 logs/${ROUND}_policy_eval.log）"
  else
    echo "[locked] 跳过策略评测（Path A：VLA 逐位未动，主评测已是策略对比）"
  fi
  fi   # SKIP_P2_EVAL
fi

echo "[locked] ✅ 完成"
if [[ "$MODE" != "p2" ]]; then
  echo "[locked]   主评测: runs/S3_loop/sweeps/${TAG_PREFIX}_merged.json"
fi
if [[ "$WITH_WBC" == "1" ]]; then
  echo "[locked]   stage8.5 ckpt: runs/S3_loop/${P2_ROUND}_joint/train_main/joint_final.pt"
  if [[ "$SKIP_P2_EVAL" != "1" ]]; then
    echo "[locked]   stage8.5 eval: runs/S3_loop/sweeps/${P2_ROUND}_vs_base_merged.json"
    if python3 -c "import sys; sys.exit(0 if float('$P2_FM_LR') > 0 else 1)"; then
      echo "[locked]   策略评测: runs/S3_loop/sweeps/${ROUND}_policy_vs_${BASELINE_LABEL}_merged.json"
    fi
  fi
fi
