#!/usr/bin/env bash
# =============================================================================
# 平台哨兵：盯一条 anchor 续训的 eval_curve.jsonl，判定"到平台期"就把该 run 停掉。
#
# 判据（两条同时满足才停，避免把 batch 噪声当平台）：
#   1) 最近 3 个 eval 点，都没有把"历史最好 token_mse"再压低 >= IMPROVE（默认 0.002）；
#   2) 最近 3 个 eval 点的 rollout_reward 极差 <= RANGE（默认 0.0025）。
# 即"训练目标（reward）已饱和 + 保真项（token）不再有实质进步"。
#
# 只杀指定进程组：setsid 起的训练有自己的 session（pgid = 脚本 pid），
# `kill -TERM -- -PID` 精确杀掉它和它的 rank，不会误伤同容器别的训练。
#
# 用法：
#   setsid nohup bash scripts/plateau_guard.sh <PID> <eval_curve.jsonl> <guard_log> &
# 环境变量：IMPROVE(0.002) RANGE(0.0025) INTERVAL(60)
# =============================================================================
set -uo pipefail

PID="${1:-}"
CURVE="${2:-}"
GUARDLOG="${3:-}"
if [[ -z "$PID" || -z "$CURVE" || -z "$GUARDLOG" ]]; then
  echo "usage: bash scripts/plateau_guard.sh <PID> <eval_curve.jsonl> <guard_log>"; exit 2
fi
IMPROVE="${IMPROVE:-0.002}"
RANGE="${RANGE:-0.0025}"
INTERVAL="${INTERVAL:-60}"
mkdir -p "$(dirname "$GUARDLOG")"
say () { echo "[guard $(date -Is)] $*" >> "$GUARDLOG"; }

say "start: pid=$PID curve=$CURVE improve>=$IMPROVE range<=$RANGE"
sleep 120   # 给训练启动/首次 eval 留时间

while true; do
  if ! kill -0 "$PID" 2>/dev/null; then
    say "训练进程 $PID 已退出（到 STEPS 上限或出错）→ 哨兵结束"
    exit 0
  fi
  if [[ -f "$CURVE" ]]; then
    VERDICT=$(
      /usr/bin/python3 - "$CURVE" "$IMPROVE" "$RANGE" <<'PY'
import json, sys
pts = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
imp, rng = float(sys.argv[2]), float(sys.argv[3])
if len(pts) < 4:
    print("wait")
    raise SystemExit
best_before = min(p["token_mse"] for p in pts[:-3])
last3 = pts[-3:]
token_ok = all(p["token_mse"] > best_before - imp for p in last3)
rs = [p["rollout_reward"] for p in last3]
rew_ok = (max(rs) - min(rs)) <= rng
print("plateau" if (token_ok and rew_ok) else "go")
PY
    )
    if [[ "$VERDICT" == "plateau" ]]; then
      say "判定【平台期】；最后 3 个 eval 点："
      tail -n 3 "$CURVE" >> "$GUARDLOG"
      kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
      sleep 30
      kill -KILL -- "-$PID" 2>/dev/null || true
      say "已停 pid=$PID（最近一次 save 的 ckpt 留在 out 目录）"
      exit 0
    fi
  fi
  sleep "$INTERVAL"
done
