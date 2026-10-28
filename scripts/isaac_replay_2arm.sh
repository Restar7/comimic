#!/usr/bin/env bash
# S3 真物理裁判（回放版）：同一 clip、同一 seed，两臂 **只差 token 流**。
#
#   gt      = npz 的 g1_motion_token_64（训练标签）
#   oracle  = 在 WM 里对 token 做梯度上升得到的（docs 里 gain_rel +20%）
#
# 判据（见 docs/S3_REAL_JUDGE_PLAN_20261027.md §6.1）：
#   oracle 在**真物理奖励/存活率**上显著优于 gt ⇒ WM 的排序有真信号；
#   打平 ⇒ "用 WM 当环境"这条路没有可判别的收益。
#
# 前置：
#   1) runs/arm_tokens/<clip>/{gt,oracle}.npy + hands.npy 已生成
#      （python3 scripts/dump_arm_tokens.py --arms gt oracle --clips ...）
#   2) patch 已打（patches/sm_token_npy.md）
#   3) 不设 CUDA_VISIBLE_DEVICES（carb 不认）；Isaac 用 ISAAC_GPU 选卡
#
# 用法：
#   bash scripts/isaac_replay_2arm.sh                       # 2 任务 × 3 seed × 2 臂
#   TASKS="DoubleDesk" SEEDS="0" MAX_STEPS=60 STAMP=x bash scripts/isaac_replay_2arm.sh   # smoke
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC" || exit 1

export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$COMIMIC/trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd}
# ---- 本机渲染三开关（见 patches/isaac_render_recipe.md）：缺一则 Isaac 起不来或相机不可用 ----
unset CUDA_VISIBLE_DEVICES
unset LD_PRELOAD
export DISPLAY=${DISPLAY:-:99}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/xdg}
export TERM=xterm
export MESA_GL_VERSION_OVERRIDE=4.6 MESA_GLSL_VERSION_OVERRIDE=460
mkdir -p "$XDG_RUNTIME_DIR"

TASKS=${TASKS:-"DoubleDesk PPBox"}
SEEDS=${SEEDS:-"0 1 2"}
ARMS=${ARMS:-"gt oracle"}
MAX_STEPS=${MAX_STEPS:-300}
ISAAC_GPU=${ISAAC_GPU:-0}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$COMIMIC/runs/isaac_replay_$STAMP}
mkdir -p "$OUT"

clip_of() {   # task -> clip（PPBox 的中缀是 __yb__，不是 __tw__）
  case "$1" in
    DoubleDesk|doubledesk|HOI_double_desk) echo "HOI_double_desk__tw__ep000000" ;;
    PPBox|ppbox|HOI_pp_box)                echo "HOI_pp_box__yb__ep000000" ;;
    *) return 1 ;;
  esac
}

echo "[replay] tasks=[$TASKS] seeds=[$SEEDS] arms=[$ARMS] max_steps=$MAX_STEPS gpu=$ISAAC_GPU"
echo "[replay] out=$OUT"

n_ok=0; n_bad=0
for task in $TASKS; do
  clip=$(clip_of "$task") || { echo "[replay] SKIP 未知 task=$task"; continue; }
  for arm in $ARMS; do
    tok="$COMIMIC/runs/arm_tokens/$clip/$arm.npy"
    hands="$COMIMIC/runs/arm_tokens/$clip/hands.npy"
    if [[ ! -f "$tok" ]]; then echo "[replay] SKIP $clip/$arm（缺 $tok）"; continue; fi
    for seed in $SEEDS; do
      run="replay_${task}_${arm}_s${seed}"
      log="$OUT/$run.log"
      echo "[replay] $(date +%T) $task/$arm/seed=$seed"
      mkdir -p "$OUT/dump"
      TASK=$task SEED=$seed MAX_STEPS=$MAX_STEPS ISAAC_GPU=$ISAAC_GPU \
      MODEL_LABEL=$arm OUT_ROOT=$OUT/runs \
      SM_TOKEN_SOURCE=npy SM_TOKEN_NPY="$tok" SM_HANDS_NPY="$hands" \
      SM_DUMP_STATES="$OUT/dump/${task}_${arm}_s${seed}" \
      HA_VLA_EXEC_STEPS=${HA_VLA_EXEC_STEPS:-20} \
        bash scripts/ha_isaac_closedloop.sh > "$log" 2>&1
      rc=$?
      if [[ $rc -eq 0 ]]; then n_ok=$((n_ok+1)); else n_bad=$((n_bad+1)); echo "[replay] rc=$rc 见 $log"; fi
      echo "[replay] $(date +%T) done rc=$rc"
    done
  done
done

echo "[replay] 完成 ok=$n_ok bad=$n_bad"
python3 - "$OUT" <<'PY'
import json, sys, pathlib, collections, statistics
out = pathlib.Path(sys.argv[1])
rows = []
for f in sorted(out.glob("runs/**/result.json")):
    try: rows.append(json.loads(f.read_text()))
    except Exception: pass
agg = collections.defaultdict(lambda: {"n":0,"ok":0,"steps":[],"rew":[],"fall":0})
for d in rows:
    k = (str(d.get("model_label") or "?"), str(d.get("task","?")))
    a = agg[k]; a["n"] += 1; a["ok"] += int(bool(d.get("success")))
    a["steps"].append(int(d.get("episode_steps") or 0))
    a["rew"].append(float(d.get("max_reward") or d.get("final_reward") or 0.0))
    if str(d.get("failure_reason","")).lower().startswith("fall"): a["fall"] += 1
lines = ["# 真物理回放两臂（GT vs ORACLE）", "",
         "| arm | task | n | success | fall | avg_steps | avg_max_reward |",
         "|---|---|---|---|---|---|---|"]
for (arm, task), a in sorted(agg.items()):
    lines.append(f"| {arm} | {task} | {a['n']} | {a['ok']} | {a['fall']} | "
                 f"{statistics.fmean(a['steps']):.1f} | {statistics.fmean(a['rew']):+.4f} |")
(out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
echo "[replay] 表 -> $OUT/SUMMARY.md"
