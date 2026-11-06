#!/usr/bin/env bash
# ============================================================================
# 多轮自动闭环驱动（2026-10-29）：按**同一套配置**连续跑 N 轮完整 10 阶段流程。
#
# 每轮做三件事，任何一步不过就**立刻停**（不带着坏口径往下跑）：
#   1. `s3_pipeline.sh all` —— 10 阶段：a_collect → collect → score → transitions
#      → wm_refresh → wm_gate → critic_gate → joint → eval → report
#   2. `s3_verify_round.sh` —— 独立验收（阶段齐全 + 5 条一致性断言）
#   3. 记一行到 runs/S3_pipeline/AUTO_ROUNDS.md
#
# 与单人手动跑的唯一差别（是为了让"闭环"名副其实，其余配置逐字相同）：
#   * `HEAD_COLLECT` = **上一轮的候选 head**（第 1 轮用 base π₀）——B 线数据必须是
#     "当前策略自己的漂移"，否则每轮都在采 π₀ 的数据，轮次之间只是数据累积、不构成闭环。
#   * 训练起点仍然固定为官方底模（`INIT_VLA` 不设），保持每轮可与 π₀ 干净对比；
#     若把训练也链式继承，一round 崩就会污染后续所有轮次，且无法归因。
#
# 用法：
#   START=6 N=10 bash scripts/s3_rounds_auto.sh          # big6 … big15
#   PREFIX=loop START=1 N=3 bash scripts/s3_rounds_auto.sh
#
# 配置（与 big5 完全一致，可按需覆盖）：
#   STEPS=200 BATCH=16 ACCUM=4 H=8 B_START_MAX_ERR=0.3
#   TRAIN_WBC_MAIN=0 TRAIN_WBC_CTRL=0
#   REWARD_ROOT_WEIGHT=1.0 REWARD_ROOT_POS_STD=0.2 REWARD_ROOT_ORI_STD=0.25
#   EVAL_MODE=pi0 A_ENVS=512 A_STEPS=300 A_WAVES=4
# ============================================================================
set -uo pipefail

COMIMIC=${COMIMIC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$COMIMIC" || exit 1

PREFIX=${PREFIX:-big}
START=${START:-6}
N=${N:-10}

STEPS=${STEPS:-200}
BATCH=${BATCH:-16}
ACCUM=${ACCUM:-4}
H=${H:-8}
B_START_MAX_ERR=${B_START_MAX_ERR:-0.3}
TRAIN_WBC_MAIN=${TRAIN_WBC_MAIN:-0}
TRAIN_WBC_CTRL=${TRAIN_WBC_CTRL:-0}
REWARD_ROOT_WEIGHT=${REWARD_ROOT_WEIGHT:-1.0}
REWARD_ROOT_POS_STD=${REWARD_ROOT_POS_STD:-0.2}
REWARD_ROOT_ORI_STD=${REWARD_ROOT_ORI_STD:-0.25}
EVAL_MODE=${EVAL_MODE:-pi0}
A_ENVS=${A_ENVS:-512}
A_STEPS=${A_STEPS:-300}
A_WAVES=${A_WAVES:-4}

LOG=$COMIMIC/runs/S3_pipeline/auto_rounds.log
TABLE=$COMIMIC/runs/S3_pipeline/AUTO_ROUNDS.md
mkdir -p "$COMIMIC/runs/S3_pipeline" "$COMIMIC/logs"
[[ -f "$TABLE" ]] || cat > "$TABLE" <<'HDR'
# 自动多轮闭环记录

| 轮次 | 候选均值 | π₀ 均值 | 配对差 | t | 更好 | 验收 |
|---|---|---|---|---|---|---|
HDR

say() { echo "[$(date '+%m-%d %H:%M:%S')] [auto] $*" | tee -a "$LOG"; }

# 上一轮候选 head（第 1 轮用 base）
prev_head_base() {
  local r=$1
  local h=$COMIMIC/runs/S3_loop/${r}_joint/train_main/vla_final.pt
  [[ -f "$h" ]] && echo "$h" || echo "base"
}

overall_ok=1
done_n=0
for k in $(seq 0 $((N-1))); do
  R=${PREFIX}$((START+k))
  if [[ -f "$COMIMIC/runs/S3_pipeline/$R/state.json" ]] \
     && python3 -c "
import json,sys
d=json.load(open('$COMIMIC/runs/S3_pipeline/$R/state.json'))
s=d.get('stages',{})
need=['a_collect','collect','score','transitions','wm','gate','critic','joint','eval','report']
sys.exit(0 if all(k in s for k in need) else 1)" 2>/dev/null; then
    say "轮次 $R 已完成过（10 阶段齐全）⇒ 跳过"
    continue
  fi

  PREV=${PREFIX}$((START+k-1))
  # ★ 2026-10-30：首轮可以用 PREV_HEAD_OVERRIDE 显式指定起点（例如从 d2b 这种非标准轮名续接）
  if [[ $k -eq 0 && -n "${PREV_HEAD_OVERRIDE:-}" ]]; then
    HC="${PREV_HEAD_OVERRIDE}"
    say "首轮起点由 PREV_HEAD_OVERRIDE 指定：$HC"
  else
    HC=$(prev_head_base "$PREV")
  fi
  [[ "$HC" == "base" ]] && say "轮次 $R 采集策略 = base(π₀)（上一轮 $PREV 没有 head）" \
                         || say "轮次 $R 采集策略 = $HC"

  # ★ 2026-10-30：B 线**保持全池累积**（= d1 拿到 −8.4 cm 时的配置）。
  #   不做"只用本轮 on-policy"的切换：①d1 就是用全池跑的，换掉会让 9 轮不可比；
  #   ②本轮数据要等 stage_collect 之后才存在，在轮次开始时判断必然看到 0 集。
  #   （mjlab 用本轮 bline 是因为它链式继承 + 数据量小；我们先保住可比性。）
  say "===== 轮次 $R 开始（$((k+1))/$N）====="
  if ! env ROUND="$R" STEPS="$STEPS" BATCH="$BATCH" ACCUM="$ACCUM" H="$H" \
        B_START_MAX_ERR="$B_START_MAX_ERR" \
        TRAIN_WBC_MAIN="$TRAIN_WBC_MAIN" TRAIN_WBC_CTRL="$TRAIN_WBC_CTRL" \
        REWARD_ROOT_WEIGHT="$REWARD_ROOT_WEIGHT" \
        REWARD_ROOT_POS_STD="$REWARD_ROOT_POS_STD" \
        REWARD_ROOT_ORI_STD="$REWARD_ROOT_ORI_STD" \
        EVAL_MODE="$EVAL_MODE" HEAD_COLLECT="$HC" \
        INIT_VLA="$([[ "$HC" == "base" ]] && echo "" || echo "$HC")" \
        A_ENVS="$A_ENVS" A_STEPS="$A_STEPS" A_WAVES="$A_WAVES" \
        bash "$COMIMIC/scripts/s3_pipeline.sh" all \
        > "$COMIMIC/runs/S3_pipeline/${R}_driver.log" 2>&1; then
    say "❌ 轮次 $R 的流程失败（看 runs/S3_pipeline/${R}_driver.log）"
    overall_ok=0
    break
  fi

  if ! ROUND="$R" bash "$COMIMIC/scripts/s3_verify_round.sh" \
        > "$COMIMIC/runs/S3_pipeline/${R}_verify.log" 2>&1; then
    say "❌ 轮次 $R 验收不合格（看 runs/S3_pipeline/${R}_verify.log）"
    tail -6 "$COMIMIC/runs/S3_pipeline/${R}_verify.log" | tee -a "$LOG"
    overall_ok=0
    break
  fi
  say "轮次 $R 验收 ✅"

  # 记一行结果
  python3 - "$COMIMIC/runs/S3_loop/sweeps/pi0_vs_${R}_avg4/summary.json" "$R" "$TABLE" <<'PY'
import json, statistics as st, sys
sd, r, table = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(sd))
    m, c = d["main"], d["ctrl"]
    dif = [y - x for x, y in zip(m["per_seed"], c["per_seed"])]
    mu, sdd = st.mean(dif), st.stdev(dif)
    sem = sdd / len(dif) ** 0.5
    row = (f"| {r} | {c['root_mean']:.3f} | {m['root_mean']:.3f} | {mu:+.3f} | "
           f"{mu/sem:+.2f} | {sum(1 for x in dif if x < 0)}/{len(dif)} | ✅ |")
except Exception as e:
    row = f"| {r} | — | — | — | — | — | 读取失败({e}) |"
open(table, "a").write(row + "\n")
print(row)
PY
  done_n=$((done_n+1))
done

if [[ "$overall_ok" == "1" ]]; then
  say "全部完成：本次跑了 $done_n 轮成功（共请求 $N 轮）"
else
  say "已停止：本次跑了 $done_n 轮，后续因失败中止"
fi
say "汇总表：$TABLE"
tail -8 "$TABLE" | tee -a "$LOG"
