#!/usr/bin/env bash
# L1 官方线 —— **官方微调 ckpt + SONIC release decoder**，在 **HA_ALIGN=1 执行器口径**下跑
# HA 的 7 个官方闭环任务。
#
# 口径（与主线/消融一致，别改）：
#   * 7 任务 = DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi FootballSingle
#     （官方 7 类；HOI_grap_cup 在本 checkout 没注册/没资产，官方集里也没有）
#   * 每个任务 EPISODES 集，seed = 0..N-1，max_steps=300，persistent-sim（ha_fast_eval.sh）
#   * pin 契约（场景/资产/init pose/decoder=release）+ 同一个 instruction 表
#   * VLA 服务：官方 ckpt（默认 runs/ckpt_vla_ha_8000_local → checkpoint-8000），
#     --decode codec --fm-steps 4 --horizon 40，latent64 输出
#
# 与"官方 SONIC 口径"的唯一区别（本脚本显式打开）：
#   SONIC_HA_ALIGN=1  →  decoder target = HA_default_q + HA_action_scale * a
#                        default pose / kp / kd 也换成 trackerTask 的 HA_ALIGN 表
#                        （详见 action_provider_sonic.py 顶部 SONIC_HA_ALIGN 说明）
#   对照臂（官方 deploy 口径）不设 SONIC_HA_ALIGN 即可；两组数字**不得混表**。
#
# 用法：
#   SERVER_URL=http://127.0.0.1:18100 bash scripts/run_L1_sonic_ha_align.sh
#   TASKS="DoubleDesk" EPISODES=1 OUT=runs/L1_sonic_ha_align_smoke bash scripts/run_L1_sonic_ha_align.sh
#   RESUME=1 ...   # 已有 summary.json 的任务跳过
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
export COMIMIC

# ---- 执行器口径：HA_ALIGN=1（与 WBC/主线同口径）+ SONIC_HA_ALIGN=1 ----------
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$COMIMIC/trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd}
export SONIC_HA_ALIGN=1
export HA_ALLOW_SONIC=1
# VLA 返回 latent64_chunk(40x64)；每 20 步重规划（receding horizon）
export SONIC_VLA_ACTION_FORMAT=${SONIC_VLA_ACTION_FORMAT:-latent64}
export HA_VLA_EXEC_STEPS=${HA_VLA_EXEC_STEPS:-20}
# obs_mode 跟随 ckpt（WBC 侧参数；sonic 路线不用，但 CLI 要一致）
export SM_OBS_MODE=${SM_OBS_MODE:-auto}
export SM_TOKEN_SOURCE=${SM_TOKEN_SOURCE:-http}

TASKS=${TASKS:-"DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi FootballSingle"}
EPISODES=${EPISODES:-10}
MAX_STEPS=${MAX_STEPS:-300}
PORT=${PORT:-18100}
SERVER_URL=${SERVER_URL:-http://127.0.0.1:$PORT}
ISAAC_GPU=${ISAAC_GPU:-0}
OUT=${OUT:-$COMIMIC/runs/L1_sonic_ha_align}
ROUTE=${ROUTE:-L1_official_sonic_ha_align}
SM_CKPT=${SM_CKPT:-$COMIMIC/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt}
SONIC_DIR=${SONIC_DIR:-$COMIMIC/../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release}
RESUME=${RESUME:-0}

mkdir -p "$OUT"
LOG=$OUT/driver.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "================ L1 / SONIC decoder / HA_ALIGN=1 ================"
say "out=$OUT tasks=[$TASKS] episodes/task=$EPISODES max_steps=$MAX_STEPS"
say "server=$SERVER_URL isaac_gpu=$ISAAC_GPU decoder=$SONIC_DIR/model_decoder.onnx"
say "SONIC_HA_ALIGN=$SONIC_HA_ALIGN SUPERMIMIC_HA_ALIGN=$SUPERMIMIC_HA_ALIGN"
say "usd=$SUPERMIMIC_ROBOT_USD"

[[ -f "$SONIC_DIR/model_decoder.onnx" ]] || { say "[FATAL] 缺 $SONIC_DIR/model_decoder.onnx"; exit 3; }
curl -sf --max-time 5 "$SERVER_URL/" >/dev/null || { say "[FATAL] VLA server 不可达: $SERVER_URL"; exit 5; }
python3 "$COMIMIC/scripts/pin_ha_closedloop.py" --check --quiet || { say "[FATAL] pin 契约校验失败"; exit 9; }

# ---- 单任务：持久仿真；失败回退逐集（与 queue_two_routes.sh 同一策略）---------
run_one_task() {  # $1 = task
  local task=$1 tdir=$OUT/$1
  say "[$task] 开始（$EPISODES 集，max_steps=$MAX_STEPS）"
  if ROUTE="$ROUTE" TASK_KEY="$task" EPISODES="$EPISODES" MAX_STEPS="$MAX_STEPS" \
       PORT="$PORT" ISAAC_GPU="$ISAAC_GPU" SM_CKPT="$SM_CKPT" SONIC_DIR="$SONIC_DIR" \
       OUT="$tdir" bash "$COMIMIC/scripts/ha_fast_eval.sh" >> "$LOG" 2>&1; then
    return 0
  fi
  say "[$task] 持久仿真失败，回退逐集模式"
  local s
  for s in $(seq 0 $((EPISODES - 1))); do
    TASK="$task" SEED="$s" REPEAT=0 MAX_STEPS="$MAX_STEPS" MODEL_LABEL="$ROUTE" \
      SERVER_URL="$SERVER_URL" ISAAC_GPU="$ISAAC_GPU" SM_CKPT="$SM_CKPT" \
      SONIC_DIR="$SONIC_DIR" OUT_ROOT="$tdir/fallback" \
      bash "$COMIMIC/scripts/ha_isaac_closedloop.sh" >> "$LOG" 2>&1 \
      || say "[$task] seed=$s 回退也失败"
  done
  python3 - "$tdir" "$task" "$ROUTE" <<'PY' >> "$LOG" 2>&1
import json, sys
from pathlib import Path
tdir, task, label = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = []
for f in sorted(tdir.glob("**/result.json")):
    try:
        rows.append(json.loads(f.read_text()))
    except Exception:
        pass
ok = sum(1 for r in rows if r.get("success"))
steps = [int(r.get("episode_steps") or 0) for r in rows]
rew = [float(r.get("max_reward") or 0.0) for r in rows]
(tdir / "summary.json").write_text(json.dumps({
    "rows": rows,
    "per_task": [{"variant": label, "task": task, "episodes": len(rows), "success": ok,
                  "success_rate": round(ok / len(rows), 4) if rows else 0.0,
                  "avg_steps": round(sum(steps) / len(steps), 1) if steps else 0.0,
                  "avg_max_reward": round(sum(rew) / len(rew), 4) if rew else 0.0}],
    "per_variant": {label: {"success": ok, "episodes": len(rows),
                            "success_rate": round(ok / len(rows), 4) if rows else 0.0}},
}, indent=2) + "\n")
print(f"[fallback-agg] {tdir}: {ok}/{len(rows)}")
PY
  [[ -f "$tdir/summary.json" ]]
}

# ---- 主循环 ------------------------------------------------------------------
ok_tasks=0; bad_tasks=0
for task in $TASKS; do
  tdir=$OUT/$task
  if [[ "$RESUME" == "1" && -f "$tdir/summary.json" ]]; then
    say "[$task] 已有 summary.json，跳过（RESUME=1）"; ok_tasks=$((ok_tasks + 1)); continue
  fi
  if run_one_task "$task"; then
    ok_tasks=$((ok_tasks + 1))
  else
    bad_tasks=$((bad_tasks + 1)); say "[$task] 失败（见 $tdir/sim.log）"
  fi
  # 口径校验：必须出现 HA_ALIGN=1 审计行 + 正确 provider
  if grep -q "\[SonicActionProvider\]\[ALIGN\] HA_ALIGN=1" "$tdir/sim.log" 2>/dev/null; then
    say "[$task] ✅ 执行器审计：SONIC_HA_ALIGN=1 生效"
  else
    say "[$task] ⚠️ 执行器审计：sim.log 里没有 [ALIGN] HA_ALIGN=1 行，结果不可当 HA_ALIGN 臂"
  fi
done
say "任务完成 ok=$ok_tasks bad=$bad_tasks"

# ---- 汇总 --------------------------------------------------------------------
python3 - "$OUT" "$ROUTE" <<'PY' | tee -a "$LOG"
import json, sys, collections
from pathlib import Path
root, label = Path(sys.argv[1]), sys.argv[2]
rows, per_task = [], []
for f in sorted(root.glob("*/summary.json")):
    d = json.loads(f.read_text())
    rows.extend(d.get("rows") or [])
    per_task.extend(d.get("per_task") or [])
ok = sum(1 for r in rows if r.get("success"))
(root / "summary.json").write_text(json.dumps({
    "rows": rows, "per_task": per_task,
    "per_variant": {label: {"success": ok, "episodes": len(rows),
                            "success_rate": round(ok / len(rows), 4) if rows else 0.0}},
}, indent=2) + "\n")
lines = [f"# L1 官方线（SONIC release decoder）— HA_ALIGN=1 执行器",
         "",
         f"- 协议：7 任务 × {len(rows)} 集（实际），max_steps=300，persistent-sim，pin 契约",
         f"- 总数：**{ok}/{len(rows)} = {(ok / len(rows) if rows else 0):.1%}**", "",
         "| task | episodes | success | rate | avg_steps | avg_max_reward |",
         "|---|---|---|---|---|---|"]
for r in per_task:
    lines.append(f"| {r['task']} | {r['episodes']} | {r['success']} | {r['success_rate']:.1%} "
                 f"| {r['avg_steps']} | {r['avg_max_reward']} |")
modes = collections.Counter("success" if r.get("success") else str(r.get("failure_reason")) for r in rows)
lines += ["", "失败原因：" + ", ".join(f"{k}={v}" for k, v in sorted(modes.items()))]
(root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"[L1] summary -> {root}/SUMMARY.md")
PY

touch "$OUT/DONE"
say "================ L1 finished: $OUT/SUMMARY.md ================"
