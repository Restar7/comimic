#!/usr/bin/env bash
# ============================================================================
# 多轮闭环**守护进程**：反复拉起 `s3_rounds_auto.sh`，直到全部轮次完成。
#
# 为什么需要（2026-10-29 实测）：这台是共享集群（inspire），长的后台会话会被**外部 SIGKILL**
# —— 两臂训练日志干净地停在 `[saved] vla_000100.pt`，没有 traceback，驱动进程也一起消失。
# 单轮（~40 min）能活，10 轮（十几小时）会被砍。所以必须可重启 + 从断点续跑：
#   * `s3_rounds_auto.sh` 会跳过 state.json 里 10 阶段齐全的轮次；
#   * 每个阶段自己也有 `st_need` + 跳过逻辑（采集/A 线已存在就不再跑）。
#
# 用法：
#   START=6 N=10 bash scripts/s3_rounds_supervisor.sh          # 前台守护
#   setsid nohup env START=6 N=10 bash scripts/s3_rounds_supervisor.sh > /tmp/sup.log 2>&1 &
#
# 环境变量：
#   MAX_RETRY=40      最多重启多少次（每次都会推进至少一个阶段）
#   RETRY_WAIT=60     重启前等多久（秒）
# ============================================================================
set -uo pipefail

COMIMIC=${COMIMIC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$COMIMIC" || exit 1

START=${START:-6}
N=${N:-10}
PREFIX=${PREFIX:-big}
MAX_RETRY=${MAX_RETRY:-40}
RETRY_WAIT=${RETRY_WAIT:-60}

LOG=$COMIMIC/runs/S3_pipeline/supervisor.log
mkdir -p "$COMIMIC/runs/S3_pipeline"
say() { echo "[$(date '+%m-%d %H:%M:%S')] [sup] $*" | tee -a "$LOG"; }

# 所有轮次是否都已完成（state.json 里 10 阶段齐全）
all_done() {
  local k R
  for k in $(seq 0 $((N-1))); do
    R=${PREFIX}$((START+k))
    python3 - "$COMIMIC/runs/S3_pipeline/$R/state.json" <<'PY' || return 1
import json, sys
need = ["a_collect", "collect", "score", "transitions", "wm", "gate",
        "critic", "joint", "eval", "report"]
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(1)
s = d.get("stages", {})
raise SystemExit(0 if all(k in s for k in need) else 1)
PY
  done
  return 0
}

say "守护启动：${PREFIX}${START}…${PREFIX}$((START+N-1))（最多重启 $MAX_RETRY 次）"
i=0
while (( i < MAX_RETRY )); do
  if all_done; then
    say "全部 $N 轮已完成 ✅ 守护退出"
    exit 0
  fi
  i=$((i+1))
  say "第 $i 次启动 s3_rounds_auto.sh"
  START="$START" N="$N" PREFIX="$PREFIX" bash "$COMIMIC/scripts/s3_rounds_auto.sh" >> "$LOG" 2>&1
  rc=$?
  if (( rc == 0 )) && all_done; then
    say "s3_rounds_auto.sh 正常结束 rc=0，且全部轮次完成 ✅"
    exit 0
  fi
  say "s3_rounds_auto.sh 退出 rc=$rc（疑似被外部杀掉）；${RETRY_WAIT}s 后从断点续跑"
  sleep "$RETRY_WAIT"
done

say "达到重启上限 $MAX_RETRY，守护退出（还有轮次未完成）"
exit 1
