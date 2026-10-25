#!/usr/bin/env bash
# 排队：等 joint_ha_a1 的 4000 步联合微调**训练跑完** → 按顺序跑两条闭环 → 出对比表。
#
# 顺序（锁死）：
#   Phase 0  wait : 等 train_joint.py(runs/joint_ha_a1) 退出 + vla_004000.pt 落盘稳定 + 显存空闲
#   Phase 1  L1   : 官方微调 VLA(checkpoint-8000) + **SONIC release decoder**
#   Phase 2  L2   : 联合微调 VLA(vla_004000)    + **SuperMimic WBC**
#   Phase 3  cmp  : runs/compare_two_routes/COMPARE.md
#
# 规模：**7 个任务 × 每个 10 集**，逐任务一个持久仿真进程（`--episode_batch_json`），
#       比逐集重启 Isaac 少付 9 次启动开销。
#
# 用法（现在只 arm，不启动仿真）：
#   setsid nohup bash scripts/queue_two_routes.sh > runs/compare_two_routes/queue.nohup 2>&1 &
#
# 可调：TASKS / EPISODES / MAX_STEPS / PORT / GPU / RESUME=1（已有 summary 就跳过）
set -uo pipefail

ROOT=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
OUT=${OUT:-$ROOT/runs/compare_two_routes}
TRAIN_OUT=$ROOT/runs/joint_ha_a1
JOINT_VLA=${JOINT_VLA:-$TRAIN_OUT/vla_004000.pt}
BASE_CKPT=${BASE_CKPT:-$ROOT/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}
SM_CKPT=${SM_CKPT:-$ROOT/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt}
SONIC_DIR=${SONIC_DIR:-$ROOT/../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release}

# 7 个任务：6 个有录制数据的 + FootballSingle（本 checkout 里注册齐全、yaml 齐全）。
# HOI_grap_cup（ArtVIP-Livingroom-GrapCup）在本 checkout 缺任务注册 + 缺资产，跑不了，
# 见 docs/HANDOVER_ISAAC_CLOSEDLOOP.md §10.7。
TASKS=${TASKS:-"DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi FootballSingle"}
# 口径说明（写进报告，别混）：
#   * 官方数据集 SII-yizhi/HumanoidArenaV3.1_SONIC40 正好 7 个目录：
#     HOI_double_desk / HOI_football / HOI_pp_box / HSI_boxing / HSI_open_door /
#     HSI_sit_sofa / HSI_vision_navi —— 就是上面这 7 个，**官方集里没有 grap_cup**。
#   * 但我们本地 699 条训练数据（data/npz/ha_tokens）的第 7 类是 HOI_grap_cup（99 条），
#     **没有 football**。所以 FootballSingle 对这两条路线都是 OOD，只能看相对差，别看绝对值。
EPISODES=${EPISODES:-10}
MAX_STEPS=${MAX_STEPS:-300}
PORT=${PORT:-18080}
VLA_GPU=${VLA_GPU:-0}
ISAAC_GPU=${ISAAC_GPU:-0}
FM_STEPS=${FM_STEPS:-4}
GPU_FREE_MIN=${GPU_FREE_MIN:-30000}
WAIT_TRAIN_MAX_H=${WAIT_TRAIN_MAX_H:-12}
RESUME=${RESUME:-0}

mkdir -p "$OUT"
LOG=$OUT/queue.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { say "[FATAL] $*" >&2; exit 1; }

say "================ queue armed ================"
say "out=$OUT"
say "joint_vla=$JOINT_VLA  base=$BASE_CKPT  wbc=$SM_CKPT"
say "sonic_release=$SONIC_DIR"
say "tasks=[$TASKS]  episodes/task=$EPISODES  max_steps=$MAX_STEPS"
say "gpu_free_min=${GPU_FREE_MIN}MiB  resume=$RESUME"

# ---------------------------------------------------------------- 工具
training_alive() { pgrep -af 'scripts/train_joint\.py' 2>/dev/null | grep -q 'joint_ha_a1'; }
sim_alive()      { pgrep -af 'sim_eval_vla\.py' >/dev/null 2>&1; }
train_any_alive() { pgrep -af 'scripts/train_joint\.py' >/dev/null 2>&1; }
# 别的编排/启动器也在抢这张卡：A1 接力、批量、以及 launch_joint_ha.sh（今天 14:00 它重启过训练）
others_alive()   { pgrep -af 'a1_pipeline_after_train\.sh|run_a1_ha_isaac\.sh|ha_isaac_batch\.sh|launch_joint_ha\.sh' >/dev/null 2>&1; }
gpu_free_mib()   { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

SERVER_PID=""
kill_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  kill "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}
trap 'kill_server' EXIT

start_server() {  # $1 = ckpt dir, $2 = tag
  local ckpt=$1 tag=$2
  [[ -d "$ckpt" ]] || die "server ckpt 不存在: $ckpt"
  local log="$OUT/$tag/server.log"
  mkdir -p "$OUT/$tag"
  PYTHONPATH=$ROOT/third_party/gr00t_deps/_noflash \
    python3 "$ROOT/scripts/ha_vla_latent64_server.py" \
      --ckpt "$ckpt" --port "$PORT" --device "cuda:$VLA_GPU" --bf16 \
      --horizon 40 --fm-steps "$FM_STEPS" > "$log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 180); do
    curl -sf --max-time 3 "http://127.0.0.1:$PORT/" >/dev/null && { say "server ready: $tag pid=$SERVER_PID"; return 0; }
    sleep 2
  done
  die "VLA server 300s 内未就绪（$tag），见 $log"
}

# 单任务：先走持久仿真；失败就回退到逐集模式（已知可用的老路径）
run_one_task() {  # $1 = tag, $2 = label, $3 = task, $4 = extra env
  local tag=$1 label=$2 task=$3 extra=$4 tdir=$OUT/$1/$3
  if env $extra ROUTE="$label" TASK_KEY="$task" EPISODES="$EPISODES" MAX_STEPS="$MAX_STEPS" \
       PORT="$PORT" ISAAC_GPU="$ISAAC_GPU" SM_CKPT="$SM_CKPT" SONIC_DIR="$SONIC_DIR" \
       OUT="$tdir" bash "$ROOT/scripts/ha_fast_eval.sh" >> "$LOG" 2>&1; then
    return 0
  fi
  say "[$tag] $task 持久仿真失败，回退逐集模式（$EPISODES 次 sim_eval，慢但稳）"
  local s
  for s in $(seq 0 $((EPISODES-1))); do
    env $extra SONIC_DIR="$SONIC_DIR" SM_CKPT="$SM_CKPT" \
      TASK="$task" SEED="$s" REPEAT=0 MAX_STEPS="$MAX_STEPS" MODEL_LABEL="$label" \
      SERVER_URL="http://127.0.0.1:$PORT" ISAAC_GPU="$ISAAC_GPU" \
      OUT_ROOT="$tdir/fallback" bash "$ROOT/scripts/ha_isaac_closedloop.sh" >> "$LOG" 2>&1 \
      || say "[$tag] $task seed=$s 回退也失败"
  done
  python3 - "$tdir" "$task" "$label" <<'PY'
import json, sys
from pathlib import Path
tdir, task, label = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = []
for f in sorted(tdir.glob("**/result.json")):
    try: rows.append(json.loads(f.read_text()))
    except Exception: pass
ok = sum(1 for r in rows if r.get("success"))
steps = [int(r.get("episode_steps") or 0) for r in rows]
rew = [float(r.get("max_reward") or 0.0) for r in rows]
(tdir / "summary.json").write_text(json.dumps({
    "rows": rows,
    "per_task": [{"variant": label, "task": task, "episodes": len(rows), "success": ok,
                  "success_rate": round(ok/len(rows), 4) if rows else 0.0,
                  "avg_steps": round(sum(steps)/len(steps), 1) if steps else 0.0,
                  "avg_max_reward": round(sum(rew)/len(rew), 4) if rew else 0.0}],
    "per_variant": {label: {"success": ok, "episodes": len(rows),
                            "success_rate": round(ok/len(rows), 4) if rows else 0.0}},
}, indent=2) + "\n")
print(f"[fallback-agg] {tdir}: {ok}/{len(rows)}")
PY
  [[ -f "$tdir/summary.json" ]]
}

# 一个路线跑 7 个任务，每个 EPISODES 集；单任务失败不拖垮整条路线
run_route() {  # $1 = tag (L1/L2), $2 = label, $3 = 额外 env（HA_ALLOW_SONIC 等，字符串）
  local tag=$1 label=$2 extra=${3:-}
  local ok_tasks=0 bad_tasks=0
  for task in $TASKS; do
    local tdir=$OUT/$tag/$task
    if [[ "$RESUME" == "1" && -f "$tdir/summary.json" ]]; then
      say "[$tag] $task 已有 summary.json，跳过（RESUME=1）"; ok_tasks=$((ok_tasks+1)); continue
    fi
    say "[$tag] $task 开始（$EPISODES 集，max_steps=$MAX_STEPS）"
    if run_one_task "$tag" "$label" "$task" "$extra"; then
      ok_tasks=$((ok_tasks+1))
    else
      bad_tasks=$((bad_tasks+1)); say "[$tag] $task 失败（见 $tdir/sim.log），继续下一个任务"
    fi
  done
  say "[$tag] 任务完成 ok=$ok_tasks bad=$bad_tasks"
}

# 把每个任务目录的 summary.json 合到路线级，compare_two_routes.py 直接吃
merge_route() {  # $1 = tag, $2 = label
  python3 - "$OUT/$1" "$2" <<'PY'
import json, sys
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
print(f"[merge] {root}: {ok}/{len(rows)}")
PY
}

check_provider() {  # $1 = tag, $2 = 期望 provider 关键字
  local n
  n=$(grep -rl "$2" "$OUT/$1" --include='*.log' 2>/dev/null | wc -l)
  if [[ "$n" -gt 0 ]]; then say "[ok] $1 执行器校验：$n 个日志出现 '$2'"
  else say "[ERROR] $1 执行器校验失败：没有日志出现 '$2' —— 可能跑错执行器，结果不可用！"; fi
}

# ---------------------------------------------------------------- Phase 0
say "---- Phase 0: 等训练跑完 ----"
WAIT_START=$(date +%s)
while training_alive; do
  (( ( $(date +%s) - WAIT_START ) > WAIT_TRAIN_MAX_H * 3600 )) && die "等 ${WAIT_TRAIN_MAX_H}h 训练还在跑，放弃"
  say "训练中… gpu_free=$(gpu_free_mib)MiB"
  sleep 60
done
say "训练进程已退出"

for _ in $(seq 1 60); do
  [[ -f "$JOINT_VLA" ]] && break
  say "等 $JOINT_VLA 出现…"; sleep 30
done
[[ -f "$JOINT_VLA" ]] || die "训练结束了但 $JOINT_VLA 不存在"
S1=$(stat -c %s "$JOINT_VLA"); sleep 20; S2=$(stat -c %s "$JOINT_VLA")
[[ "$S1" == "$S2" && "$S1" -gt 1000000000 ]] || die "vla_004000.pt 大小异常/还在写：$S1 -> $S2"
say "joint vla ok: $JOINT_VLA ($((S2/1024/1024)) MiB)"

FREE_OK=0
while true; do
  sim_alive && { say "另有 sim_eval_vla 在跑，等它结束…"; sleep 60; FREE_OK=0; continue; }
  others_alive && { say "别的编排/启动器还在跑，等它结束…"; sleep 60; FREE_OK=0; continue; }
  # 训练被重启过（14:00 那次）：vla_004000 出现后再有人起训练，也要等它跑完，别抢卡
  train_any_alive && { say "又有 train_joint.py 在跑，等它结束…"; sleep 120; FREE_OK=0; continue; }
  free=$(gpu_free_mib)
  if [[ -n "$free" && "$free" -ge "$GPU_FREE_MIN" ]]; then
    FREE_OK=$((FREE_OK+1)); say "显存空闲 ${free}MiB（$FREE_OK/3）"
    (( FREE_OK >= 3 )) && break
  else
    FREE_OK=0; say "显存空闲 ${free}MiB < ${GPU_FREE_MIN}MiB，继续等…"
  fi
  sleep 30
done
say "显存就绪，开始跑闭环"

# ---------------------------------------------------------------- Phase 1 L1
say "---- Phase 1: L1 = 官方微调 VLA + SONIC release decoder ----"
[[ -f "$SONIC_DIR/model_decoder.onnx" ]] || die "SONIC decoder 不在 $SONIC_DIR"
if [[ "$RESUME" == "1" && -f "$OUT/L1/summary.json" ]]; then
  say "L1 已有 summary.json，跳过（RESUME=1）"
else
  start_server "$BASE_CKPT" L1
  run_route L1 "L1_official_sonic" "HA_ALLOW_SONIC=1"
  kill_server
  merge_route L1 "L1_official_sonic"
  check_provider L1 "SonicActionProvider"
fi

# ---------------------------------------------------------------- Phase 2 L2
say "---- Phase 2: L2 = 联合微调 vla_004000 + SuperMimic WBC ----"
if [[ "$RESUME" == "1" && -f "$OUT/L2/summary.json" ]]; then
  say "L2 已有 summary.json，跳过（RESUME=1）"
else
  MERGED=$OUT/ckpt_vla_joint4000
  if [[ ! -f "$MERGED/vla_merge_report.json" ]]; then
    say "merge $JOINT_VLA -> $MERGED"
    python3 "$ROOT/scripts/export_vla_ckpt.py" \
      --base "$BASE_CKPT" --vla "$JOINT_VLA" --out "$MERGED" --force \
      >> "$OUT/merge.log" 2>&1 || die "merge 失败，见 $OUT/merge.log"
  fi
  start_server "$MERGED" L2
  run_route L2 "L2_joint4000_supermimic" ""
  kill_server
  merge_route L2 "L2_joint4000_supermimic"
  check_provider L2 "SuperMimic"
fi

# ---------------------------------------------------------------- Phase 3
say "---- Phase 3: 两条路线对比 ----"
python3 "$ROOT/scripts/compare_two_routes.py" \
  --l1 "$OUT/L1" --l2 "$OUT/L2" --out "$OUT/COMPARE.md" >> "$LOG" 2>&1 \
  || say "[warn] compare 失败，两份 summary.json 仍在"

touch "$OUT/DONE"
say "================ queue finished: $OUT/COMPARE.md ================"
