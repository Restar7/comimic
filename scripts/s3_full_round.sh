#!/usr/bin/env bash
# S3 **完整一轮**（严格按 docs/RUNBOOK 的顺序，一个环节都不省）：
#
#   1. 采集     两卡并行，用当前策略（可传 base=底模）采 N 集，server 带 --sample-avg
#   2. 打分     compute_collect_reward（与训练同一个 reward）
#   3. 真转移   build_real_transitions_v2 --extra-dumps（B 线 on-policy 转移）
#   4. 刷新 WM  finetune_wm_velocity_policy（用本轮新采的数据）
#   5. WM 门    eval_wm_multistep（**留出一个任务**做干净留出）
#   6. 联合微调  V3=1，WBC_CKPT = **刚刷新的 WM**，奖励项开着让 WM 当裁判
#
# 用法：
#   LABEL=F1 bash scripts/s3_full_round.sh              # 从底模起跑完整一轮
#   LABEL=F2 HEAD_COLLECT=runs/.../vla_000150.pt bash scripts/s3_full_round.sh
#
# 关键环境变量：
#   LABEL          本轮标签（目录 runs/S3_full/$LABEL，采集落 runs/B_$LABEL）
#   HEAD_COLLECT   采集时 server 挂的 head（base=底模）
#   BASE_WM        WM 刷新起点（默认文档指定的 A24000_wmvel_cmd）
#   HOLD_TASK      WM 门的留出任务（默认 SitSofa，只用它做留出，不进 WM 训练）
#   STEPS          联合微调步数（默认 150）
#   LAMBDA_R_MAIN / LAMBDA_R_CTRL   主臂 / 对照臂奖励权重（默认 0.3 / 0.0）
#   TRUST_CENTER / ACT_EPS          默认 prev / 0.6（= 文档 §5.3 第 1 轮口径）
set -uo pipefail

COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1
export COMIMIC

LABEL=${LABEL:-F1}
HEAD_COLLECT=${HEAD_COLLECT:-base}
BASE_WM=${BASE_WM:-$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_cmd.pt}
HOLD_TASK=${HOLD_TASK:-SitSofa}
STEPS=${STEPS:-150}
TRUST_CENTER=${TRUST_CENTER:-prev}
ACT_EPS=${ACT_EPS:-0.6}
LAMBDA_R_MAIN=${LAMBDA_R_MAIN:-0.3}
LAMBDA_R_CTRL=${LAMBDA_R_CTRL:-0.0}
FM_LR=${FM_LR:-1e-4}
TASKS=${TASKS:-"DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi"}
COLLECT_STEPS=${COLLECT_STEPS:-600}
SRV_EXTRA=${SRV_EXTRA:---sample-avg 4}

OUT=$COMIMIC/runs/S3_full/$LABEL
BDIR=$COMIMIC/runs/B_$LABEL
LOGD=$COMIMIC/logs/S3_full_$LABEL
mkdir -p "$OUT" "$BDIR" "$LOGD"

# 2026-10-29：闭环/打分/转移/WM 全都要这两个口径，否则 default_q 对不上会直接失败
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$COMIMIC/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd}

say() { echo "[$(date +%H:%M:%S)] [full $LABEL] $*"; }
die() { say "[FAIL] $*"; exit 1; }

say "=========== 完整一轮 $LABEL ==========="
say "采集 head=${HEAD_COLLECT} · WM 起点=$(basename "$BASE_WM") · 留出任务=$HOLD_TASK"
say "联合微调 center=$TRUST_CENTER act_eps=$ACT_EPS λ_R=$LAMBDA_R_MAIN/$LAMBDA_R_CTRL steps=$STEPS TRAIN_WBC=${TRAIN_WBC_MAIN:-1}"

# ---------------------------------------------------------------- 1. 采集
say "--- 1/6 采集（两卡并行，$COLLECT_STEPS 步 × $(echo $TASKS | wc -w)）---"
ROUND=99 RUN_LABEL=${LABEL}_collect NEXT_ROUND=0 \
  SRV_EXTRA_VAL="$SRV_EXTRA" HEAD_COLLECT="$HEAD_COLLECT" B_NEXT_DIR="$BDIR" \
  TASKS="$TASKS" COLLECT_STEPS="$COLLECT_STEPS" \
  bash "$COMIMIC/scripts/s3_loop.sh" collect 2>&1 | tee "$LOGD/collect.log" | tail -4
n=$(find "$BDIR" -name '*.pt' -path '*/dump/*' | wc -l)
say "采集完成 dumps=$n"
(( n >= 3 )) || die "采集不足（$n 集）"

# ---------------------------------------------------------------- 2. 打分
say "--- 2/6 打分 ---"
i=0
for d in "$BDIR"/vla_*/dump/*.pt; do
  dd=$(dirname "$d")
  PYTHONPATH=. SUPERMIMIC_HA_ALIGN=1 python3 "$COMIMIC/scripts/compute_collect_reward.py" \
    --in "$d" --out "$dd/../scored.pt" --wbc-ckpt "$BASE_WM" --device "cuda:$((i % 2))" \
    >> "$LOGD/score.log" 2>&1 || say "[warn] 打分失败 $d"
  i=$((i+1))
done
say "打分完成 $(ls "$BDIR"/vla_*/scored.pt 2>/dev/null | wc -l) 个"

# ---------------------------------------------------------------- 3. 真转移
say "--- 3/6 重建真转移 ---"
PYTHONPATH=. python3 "$COMIMIC/scripts/build_real_transitions_v2.py" \
  --wbc-ckpt "$BASE_WM" \
  --extra-dumps "$BDIR/vla_*_f*_r*/dump/*.pt" \
  --out "$COMIMIC/runs/wm_real/rt_v2_$LABEL.pt" --device cuda:0 \
  > "$LOGD/transitions.log" 2>&1 || die "真转移失败（看 $LOGD/transitions.log）"
say "真转移完成 $(tail -2 "$LOGD/transitions.log" | head -1)"

# ---------------------------------------------------------------- 4. 刷新 WM
say "--- 4/6 刷新 WM（留出 $HOLD_TASK）---"
TRAIN_GLOB=$(ls -d "$BDIR"/vla_*_f*_r* 2>/dev/null | grep -v "vla_${HOLD_TASK}_" | tr '\n' ' ')
VAL_GLOB=$(ls -d "$BDIR"/vla_${HOLD_TASK}_* 2>/dev/null | head -1)
[[ -n "$TRAIN_GLOB" && -n "$VAL_GLOB" ]] || die "WM 训练/留出划分失败"
PYTHONPATH=. SUPERMIMIC_HA_ALIGN=1 python3 "$COMIMIC/scripts/finetune_wm_velocity_policy.py" \
  --ckpt "$BASE_WM" --out "$COMIMIC/runs/S1_handover/wm_${LABEL}.pt" \
  --packed "$COMIMIC/runs/isaac_motion/ha_cmd/packed.pt" --token-space label \
  --rollout-steps 3 --steps 150 --batch 256 --lr 3e-5 --vel-lambda 5 \
  --traj-dump "$TRAIN_GLOB/dump/*.pt" --traj-frac 0.2 --traj-val-frac 0.25 \
  --val-dump "$VAL_GLOB/dump/*.pt" \
  --val-horizon 20 --val-every 50 --val-max-starts 32 --device cuda:0 \
  > "$LOGD/wm_refresh.log" 2>&1 || die "WM 刷新失败（看 $LOGD/wm_refresh.log）"
NEW_WM="$COMIMIC/runs/S1_handover/wm_${LABEL}.pt"
[[ -f "$NEW_WM" ]] || die "WM 刷新没产出 $NEW_WM"
say "WM 刷新完成：$(grep -E '^\[early-stop\]' "$LOGD/wm_refresh.log" | tail -1)"

# ---------------------------------------------------------------- 5. WM 门
say "--- 5/6 WM 门（留出 $HOLD_TASK 上比较 base vs 新）---"
PYTHONPATH=. SUPERMIMIC_HA_ALIGN=1 python3 "$COMIMIC/scripts/eval_wm_multistep.py" \
  --ckpt "$BASE_WM" "$NEW_WM" \
  --dump "$VAL_GLOB/dump/*.pt" --horizon 20 --stride 20 --device cuda:1 \
  > "$LOGD/wm_gate.log" 2>&1 || die "WM 门失败（看 $LOGD/wm_gate.log）"
tail -6 "$LOGD/wm_gate.log" | tee -a "$LOGD/summary.txt"
if grep -q "PASS" "$LOGD/wm_gate.log"; then
  say "WM 门 PASS ⇒ 用新 WM 当裁判"
else
  say "WM 门 FAIL ⇒ 回退用 base WM 当裁判"
  NEW_WM="$BASE_WM"
fi

# ---------------------------------------------------------------- 6. 联合微调
say "--- 6/6 联合微调（WM 裁判 = $(basename "$NEW_WM")，奖励项开）---"
ROUND=99 RUN_LABEL=${LABEL}_joint \
  STEPS="$STEPS" FM_LR="$FM_LR" \
  TRUST_CENTER="$TRUST_CENTER" ACT_EPS="$ACT_EPS" TRUST_EPS=0 TRUST_WEIGHT=0 \
  TOKEN_BC=0 LAMBDA_BC=0 LAMBDA_R_MAIN="$LAMBDA_R_MAIN" LAMBDA_R_CTRL="$LAMBDA_R_CTRL" \
  TRAIN_WBC_MAIN=${TRAIN_WBC_MAIN:-1} TRAIN_WBC_CTRL=${TRAIN_WBC_CTRL:-0} WBC_LR=${WBC_LR:-1e-6} \
  WBC_CKPT="$NEW_WM" B_GLOB="$BDIR/vla_*_f*_r*/dump/*.pt" \
  bash "$COMIMIC/scripts/s3_loop.sh" train 2>&1 | tee "$LOGD/joint.log" | tail -4

cat >> "$LOGD/summary.txt" <<EOF
LABEL=$LABEL
collect_dumps=$n
wm=$NEW_WM
joint_out=$COMIMIC/runs/S3_loop/${LABEL}_joint
next: TAG=pi0_vs_${LABEL} SEEDS="0 1 2 3 4 5" START_FRAME=0 HEAD_MAIN=base \\
  HEAD_CTRL=runs/S3_loop/${LABEL}_joint/train_main/vla_final.pt \\
  SM_CKPT_MAIN=$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt \\
  SM_CKPT_CTRL=runs/S3_loop/${LABEL}_joint/train_main/joint_final.pt \\
  EXTRA_MAIN="--sample-avg 4" EXTRA_CTRL="--sample-avg 4" bash scripts/s3_eval_sweep.sh
EOF

say "=========== 一轮完成，报告：$LOGD/summary.txt ==========="
