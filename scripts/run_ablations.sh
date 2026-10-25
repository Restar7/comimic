#!/usr/bin/env bash
# 三个消融实验的统一入口（协议见 docs/ABLATIONS.md）
#
#   MODE=1 bash scripts/run_ablations.sh     # 消融1：有/无反传微调 → 成功率
#   MODE=2 bash scripts/run_ablations.sh     # 消融2：FM 反传方法（4 个后端）
#   MODE=3 bash scripts/run_ablations.sh     # 消融3：BPTT vs PPO（固定墙钟）
#   MODE=all bash scripts/run_ablations.sh   # 顺序跑完（会占卡很多小时）
#   DRY=1 MODE=all bash scripts/run_ablations.sh   # 只打印将执行的命令
#
# 默认假设（都能用环境变量覆盖）：
#   CKPT_BASE=runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000   基线：官方 HA VLA
#   CKPT_FT=runs/ckpt_vla_ha_ft4000         处理组：纯 HA 联合微调合并回来的 checkpoint
#   MOTION=data/npz/ha_tokens               699 episodes 纯 HA npz（mixed/ft318 永久禁用）
#   SM_CKPT=.../bones_seed_token_ha_v2/ckpt_013500.pt  纯 HA IsaacLab WBC
#   VLM_CKPT=$CKPT_BASE                     GR00T checkpoint（processor/统计量都从它来）
#
# 硬约束：只走纯 HA + IsaacLab/PhysX；mixed/ft318/MuJoCo 一律拒绝（docs/DATA_POLICY_HA_ONLY.md）。
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC"

MODE=${MODE:-all}
DRY=${DRY:-0}
PY=${PY:-python3}

CKPT_BASE=${CKPT_BASE:-$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}
CKPT_FT=${CKPT_FT:-$COMIMIC/runs/ckpt_vla_ha_ft4000}
MOTION=${MOTION:-$COMIMIC/data/npz/ha_tokens}
SM_CKPT=${SM_CKPT:-$COMIMIC/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500.pt}
VLM_PATH=${VLM_PATH:-$COMIMIC/groot_repro/nvidia/Cosmos-Reason2-2B-weights}
VLM_CKPT=${VLM_CKPT:-$CKPT_BASE}
OUT_ROOT=${OUT_ROOT:-$COMIMIC/runs/ablations}

SEEDS=${SEEDS:-"0 1 2 3 4"}
CHUNKS=${CHUNKS:-40}
AB2_BACKENDS=${AB2_BACKENDS:-"pathwise checkpointed truncated adjoint"}
AB2_MIN=${AB2_MIN:-30}     # 每个后端固定墙钟分钟
AB3_MIN=${AB3_MIN:-120}    # 每个 arm 固定墙钟分钟（默认 2 h，见 docs/ABLATIONS.md）
NGPU=${NGPU:-1}

export PYTHONPATH="$COMIMIC:$COMIMIC/Isaac-GR00T:$COMIMIC/third_party/gr00t_deps/_noflash${PYTHONPATH:+:$PYTHONPATH}"
case "$MOTION" in
  *mixed*|*ft318*|*g1_tasks*) echo "[fail] 只允许纯 HA 数据 (data/npz/ha_tokens)，拒绝: $MOTION"; exit 1 ;;
esac
case "$SM_CKPT" in
  *bones_seed_token_ha_v2*) ;;
  *) echo "[fail] WBC 只允许纯 HA 的 trackerlab ckpt，拒绝: $SM_CKPT"; exit 1 ;;
esac
case "$CKPT_FT" in
  *mixed*|*ft318*|*ckpt_vla_ft4000) echo "[fail] 处理臂只允许纯 HA 联合微调产物，拒绝: $CKPT_FT"; exit 1 ;;
esac

mkdir -p "$OUT_ROOT"
stamp() { date +%H:%M:%S; }
say() { echo "[$(stamp)] $*"; }
run() {
  if [ "$DRY" != "0" ]; then echo "[dry] $*"; else "$@"; fi
}

need() { [[ -e "$1" ]] || { echo "[fail] missing $1"; exit 1; }; }

ablation1() {
  say "=== 消融1：有/无联合微调（纯 HA + IsaacLab/HumanoidArena）==="
  # 旧 MuJoCo / mixed-ft318 路径永久禁用；正式口径只走 run_a1_ha_isaac.sh。
  SEEDS="$SEEDS" REPEATS="${REPEATS:-1}" MAX_STEPS="${MAX_STEPS:-300}" \
    OUT="$OUT_ROOT/A1_joint_finetune_ha" \
    run bash scripts/run_a1_ha_isaac.sh
  say "结果：$OUT_ROOT/A1_joint_finetune_ha/SUMMARY.md"
}

ablation2() {
  say "=== 消融2：FM 反传后端（同一数据/seed/步数预算）==="
  need "$MOTION"; need "$SM_CKPT"; need "$VLM_CKPT"
  for backend in $AB2_BACKENDS; do
    local out="$OUT_ROOT/a2_fmbackends/$backend"
    mkdir -p "$out"
    say "-> $backend (${AB2_MIN} min)"
    run "$PY" -m torch.distributed.run --nproc_per_node "${NGPU:-1}" --master_port 29641 \
      scripts/train_joint.py \
        --motion "$MOTION" --supermimic-ckpt "$SM_CKPT" \
        --vlm --vlm-path "$VLM_PATH" --vlm-ckpt "$VLM_CKPT" --vlm-dataset \
        --vlm-steps 2 --batch-size 2 --horizon 8 --steps 1000000 \
        --fm-grad-backend "$backend" \
        --time-budget-minutes "$AB2_MIN" \
        --log-every 50 --eval-every 300 --eval-batches 2 \
        --eval-out "$out/eval_curve.jsonl" \
        --hand-loss-weight 0.2 --hand-grasp-boost 5 \
        --save-every 300 --save-vla-every 0 \
        --out "$out"
  done
  say "结果：$OUT_ROOT/a2_fmbackends/*/eval_curve.jsonl"
}

ablation3() {
  say "=== 消融3：BPTT vs PPO，每个 arm 固定 ${AB3_MIN} min ==="
  need "$MOTION"; need "$SM_CKPT"; need "$VLM_CKPT"

  say "-> BPTT（精确反传，train_joint.py）"
  run "$PY" -m torch.distributed.run --nproc_per_node "${NGPU:-1}" --master_port 29642 \
    scripts/train_joint.py \
      --motion "$MOTION" --supermimic-ckpt "$SM_CKPT" \
      --vlm --vlm-path "$VLM_PATH" --vlm-ckpt "$VLM_CKPT" --vlm-dataset \
      --vlm-steps 2 --batch-size 2 --horizon 8 --steps 1000000 \
      --fm-grad-backend pathwise --time-budget-minutes "$AB3_MIN" \
      --log-every 50 --eval-every 300 --eval-batches 2 \
      --eval-out "$OUT_ROOT/a3_bptt_vs_ppo/bptt/eval_curve.jsonl" \
      --hand-loss-weight 0.2 --hand-grasp-boost 5 \
      --save-every 300 --save-vla-every 0 \
      --out "$OUT_ROOT/a3_bptt_vs_ppo/bptt"

  say "-> PPO（同一 reward，score-function 梯度）"
  run "$PY" -m torch.distributed.run --nproc_per_node "${NGPU:-1}" --master_port 29643 \
    scripts/train_ppo.py \
      --motion "$MOTION" --supermimic-ckpt "$SM_CKPT" \
      --vlm --vlm-path "$VLM_PATH" --vlm-ckpt "$VLM_CKPT" \
      --vlm-steps 2 --batch-size 2 --horizon 8 \
      --ppo-epochs 4 --sigma-init 0.25 \
      --time-budget-minutes "$AB3_MIN" \
      --log-every 10 --eval-every 100 --eval-batches 2 \
      --eval-out "$OUT_ROOT/a3_bptt_vs_ppo/ppo/eval_curve.jsonl" \
      --hand-loss-weight 0.2 --hand-grasp-boost 5 \
      --save-vla-every 0 \
      --out "$OUT_ROOT/a3_bptt_vs_ppo/ppo"

  say "结果：$OUT_ROOT/a3_bptt_vs_ppo/{bptt,ppo}/eval_curve.jsonl"
  say "汇总： python3 scripts/ablation_report.py --root $OUT_ROOT"
}

case "$MODE" in
  1) ablation1 ;;
  2) ablation2 ;;
  3) ablation3 ;;
  all) ablation1; ablation2; ablation3 ;;
  *) echo "MODE must be 1|2|3|all"; exit 1 ;;
esac
say "done (MODE=$MODE)"
