#!/usr/bin/env bash
# ============================================================================
# 全任务 × 长视频对照：**未微调基线（base + v6）** vs **我们的系统（h20r2 + v7）**
# ============================================================================
# 目的：给人看的定性对照（不是测量）。每个任务、每个臂，各录一条 60 s 的视频。
#
# 口径（与 docs/HANDOVER_2GPU_EVAL_20261101.md §5 一致，只有两个偏离）：
#   * MAX_STEPS 300 → **3000**（= 60 s @ 50 Hz）——**只用于演示，不用于测量**；
#   * 执行器**按臂不同**（baseline 用 v6、ours 用 v7），因为这是"系统对系统"的演示。
#     若要"只翻策略"的干净对照，把 exec_ours 也设成 v6。
#
# 用法：
#   TASKS="DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi FootballSingle" \
#   bash scripts/record_demo_all_tasks.sh
#   单任务调试：TASKS=DoubleDesk MAX_STEPS=300 bash scripts/record_demo_all_tasks.sh
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD

export COMIMIC=$ROOT
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$ROOT/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd}

TASKS=${TASKS:-"DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi FootballSingle"}
SEEDS=${SEEDS:-"0"}
MAX_STEPS=${MAX_STEPS:-3000}          # 60 s @ 50 Hz（演示口径）
START_FRAME=${START_FRAME:-0}
RECORD_VIDEO=${RECORD_VIDEO:-1}       # 每集都录
BASE_PORT=${BASE_PORT:-18200}
STAMP=${STAMP:-$(date +%m%d_%H%M)}
OUTROOT=${OUTROOT:-$ROOT/runs/DEMO_ALLTASKS}
mkdir -p "$OUTROOT"

HEAD_BASE=${HEAD_BASE:-base}                                   # 未微调（无 head delta）
EXEC_BASE=${EXEC_BASE:-$ROOT/runs/DEPLOY_2GPU/exec_fallback_v6.pt}
HEAD_OURS=${HEAD_OURS:-$ROOT/runs/DEPLOY_2GPU/policy_primary_h20r2.pt}
EXEC_OURS=${EXEC_OURS:-$ROOT/runs/DEPLOY_2GPU/exec_primary_v7.pt}

say() { echo "[$(date +%H:%M:%S)] [demo] $*"; }
say "tasks=$TASKS  steps=$MAX_STEPS (${MAX_STEPS} ticks ≈ $((MAX_STEPS / 50)) s)  seeds=$SEEDS"
say "baseline = $HEAD_BASE + $(basename "$EXEC_BASE")"
say "ours     = $(basename "$HEAD_OURS") + $(basename "$EXEC_OURS")"

i=0
for T in $TASKS; do
  P0=$((BASE_PORT + i * 2)); P1=$((BASE_PORT + i * 2 + 1))
  TAG="${T}_base_vs_ours_${STAMP}"
  say "===== $T （端口 $P0/$P1，tag=$TAG）====="
  TASK="$T" TAG="$TAG" SEEDS="$SEEDS" START_FRAME="$START_FRAME" MAX_STEPS="$MAX_STEPS" \
    RECORD_VIDEO="$RECORD_VIDEO" \
    HEAD_MAIN="$HEAD_BASE" HEAD_CTRL="$HEAD_OURS" \
    SM_CKPT_MAIN="$EXEC_BASE" SM_CKPT_CTRL="$EXEC_OURS" \
    LABEL_MAIN=baseline LABEL_CTRL=ours \
    EXTRA_MAIN="--sample-avg 4" EXTRA_CTRL="--sample-avg 4" \
    PORT_MAIN="$P0" PORT_CTRL="$P1" GPU_MAIN=0 GPU_CTRL=1 SRV_SEED=0 \
    OUT_ROOT="$OUTROOT" \
    bash scripts/s3_eval_sweep.sh > "$OUTROOT/${T}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] [demo] $T 结束 rc=$rc"
  tail -3 "$OUTROOT/${T}.log" | sed 's/^/    /'
  i=$((i + 1))
done

say "全部结束。汇总："
for T in $TASKS; do
  python3 - "$OUTROOT/${T}_base_vs_ours_${STAMP}/summary.json" "$T" <<'PY'
import json, sys, os
p, t = sys.argv[1], sys.argv[2]
if not os.path.exists(p):
    print(f"  {t:14s} (无 summary)"); raise SystemExit
d = json.load(open(p))
m, c = d.get("main", {}), d.get("ctrl", {})
print(f"  {t:14s} baseline={m.get('root_mean',float('nan')):.3f}m  ours={c.get('root_mean',float('nan')):.3f}m"
      f"  配对差={((c.get('root_mean',0)-m.get('root_mean',0))*100):+.1f}cm  (n={c.get('n')})")
PY
done
say "视频在：$OUTROOT/<TASK>_base_vs_ours_${STAMP}/{main,ctrl}_r*/runs/*/videos_*/"
