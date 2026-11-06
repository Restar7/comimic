#!/usr/bin/env bash
# **安全启动器**：把多轮闭环挂到 setsid+nohup 上，关终端/Ctrl-C 不会杀掉它。
#
# 为什么要这个：2026-10-28 18:19 那次就是这样死的——
#   采集跑完了（1.19 GB transitions）、WM 门也写了 json、B 线刚出第一条，
#   然后 18:19:48 全部消失：**没有 traceback、没有 OOM 行**，
#   同一时刻另一路（logs/r1_main.log）也停在 step 190/300。
#   ⇒ 是终端关闭/ Ctrl-C / SIGTERM 把**同一个终端的进程组**一起杀了。
#
# 用法：
#   bash scripts/launch_j1_rounds.sh                 # 后台起，立刻返回
#   ROUNDS=6 REWARD_STEPS=800 bash scripts/launch_j1_rounds.sh
#   DRY=1 bash scripts/launch_j1_rounds.sh           # 只打印将执行的命令
set -euo pipefail

COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC"

ROUNDS=${ROUNDS:-6}
REWARD_STEPS=${REWARD_STEPS:-800}
NGPU=${NGPU:-2}
GPUS=${GPUS:-0,1}
J1_BATCH=${J1_BATCH:-64}
J1_LORA=${J1_LORA:-64}
ENVS=${ENVS:-1024}
STEPS=${STEPS:-1000}
OUT=${OUT:-$COMIMIC/runs/j1_rounds}
LOG=${LOG:-$COMIMIC/logs/j1_rounds_$(date +%Y%m%d_%H%M%S).log}
# 下面这些也必须在这里先落成 shell 变量：`set -u` 下 CMD 里的 ${X:-def} 只作用于子进程环境，
# 本 shell 依旧认为 X 未定义 —— 第 47 行 echo 一读就 "unbound variable" 直接退出（2026-10-29 03:42 就这么死的）。
SMOKE=${SMOKE:-0}
GATES=${GATES:-}
PATIENCE=${PATIENCE:-2}
MIN_GAIN=${MIN_GAIN:-0.01}
TASK=${TASK:-task1}
BLINE_TASKS=${BLINE_TASKS:-task1 task2 task3 task4}
BLINE_EPISODES=${BLINE_EPISODES:-20}
BLINE_EVAL_SET=${BLINE_EVAL_SET:-1}
BLINE_EVAL_SEED=${BLINE_EVAL_SEED:-999}
BLINE_EVAL_PER_TASK=${BLINE_EVAL_PER_TASK:-5}
BLINE_TICKS=${BLINE_TICKS:-600}
BLINE_JOBS=${BLINE_JOBS:-2}
BLINE_STARTS=${BLINE_STARTS:-}
TRAIN_WBC=${TRAIN_WBC:-1}
WBC_LR=${WBC_LR:-1e-6}
J1_H=${J1_H:-20}
J1_FMLR=${J1_FMLR:-1e-4}
REUSE=${REUSE:-0}
START_ROUND=${START_ROUND:-1}
CRITIC_MODE=${CRITIC_MODE:-auto}
CRITIC_SOURCE=${CRITIC_SOURCE:-real}
WM_BLINE_RATIO=${WM_BLINE_RATIO:-0.45}
J1_LAMBDA_BC=${J1_LAMBDA_BC:-0}
BLINE_STARTS_MODE=${BLINE_STARTS_MODE:-auto}
WM_REFRESH_EVERY=${WM_REFRESH_EVERY:-500}
WM_REFRESH_STEPS=${WM_REFRESH_STEPS:-200}
WM_PRETRAIN_STEPS=${WM_PRETRAIN_STEPS:-3000}
SM_SCENE_TASKS=${SM_SCENE_TASKS:-task1 task2 task3 task4}
SCENE_ENVS=${SCENE_ENVS:-256}
SCENE_STEPS=${SCENE_STEPS:-1000}

CMD=(env
  ROUNDS="$ROUNDS" REWARD_STEPS="$REWARD_STEPS" NGPU="$NGPU" GPUS="$GPUS"
  J1_BATCH="$J1_BATCH" J1_LORA="$J1_LORA" ENVS="$ENVS" STEPS="$STEPS" OUT="$OUT" \
  SMOKE="$SMOKE" GATES="$GATES" PATIENCE="$PATIENCE" MIN_GAIN="$MIN_GAIN" \
  TASK="$TASK" \
  BLINE_TASKS="$BLINE_TASKS" BLINE_EPISODES="$BLINE_EPISODES" \
  BLINE_TICKS="$BLINE_TICKS" BLINE_JOBS="$BLINE_JOBS" BLINE_STARTS="$BLINE_STARTS" \
  TRAIN_WBC="$TRAIN_WBC" WBC_LR="$WBC_LR" J1_H="$J1_H" J1_FMLR="$J1_FMLR" REUSE="$REUSE" START_ROUND="$START_ROUND" \
  WM_REFRESH_EVERY="$WM_REFRESH_EVERY" WM_REFRESH_STEPS="$WM_REFRESH_STEPS" WM_PRETRAIN_STEPS="$WM_PRETRAIN_STEPS" SM_SCENE_TASKS="$SM_SCENE_TASKS" SCENE_ENVS="$SCENE_ENVS" SCENE_STEPS="$SCENE_STEPS" \
  BLINE_STARTS_MODE="$BLINE_STARTS_MODE" CRITIC_MODE="$CRITIC_MODE" CRITIC_SOURCE="$CRITIC_SOURCE" WM_BLINE_RATIO="$WM_BLINE_RATIO" RESUME_FROM_STEP="${RESUME_FROM_STEP:-1}" BLINE_EVAL_SET="${BLINE_EVAL_SET:-1}" BLINE_EVAL_SEED="${BLINE_EVAL_SEED:-999}" BLINE_EVAL_PER_TASK="${BLINE_EVAL_PER_TASK:-5}" J1_LAMBDA_BC="$J1_LAMBDA_BC"
  bash scripts/run_j1_rounds.sh)

if [[ ${DRY:-0} == 1 ]]; then
  printf '[dry] '; printf '%q ' "${CMD[@]}"; printf '\n> %s\n' "$LOG"
  exit 0
fi

mkdir -p "$(dirname "$LOG")" "$OUT"
echo "[launch] rounds=$ROUNDS (start=$START_ROUND) steps/round=$REWARD_STEPS gpus=$GPUS log=$LOG"
echo "[launch] 采集 $(echo $BLINE_TASKS | wc -w) 任务 x ${BLINE_EPISODES} 集 x ${BLINE_TICKS} tick (arm=vla, jobs=$BLINE_JOBS) | WBC ${TRAIN_WBC} | batch=$J1_BATCH lora=$J1_LORA H=$J1_H （batch/envs 会被 run_j1_rounds.sh 按显存再收一次，见下方 [batch-policy]）"
setsid nohup "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "[launch] pid=$PID"
disown || true
sleep 20

echo
echo "==================== 启动自检（20 s） ===================="
if ! kill -0 "$PID" 2>/dev/null; then
  echo "[FAIL] 进程已退出 —— 看日志：$LOG"; tail -20 "$LOG"; exit 1
fi
echo "[ok] 进程存活 (pid=$PID)"
echo "--- 日志前 12 行 ---"; head -12 "$LOG"
echo "--- GPU ---"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -4
cat <<EOF

接下来（3 分钟内必须看到，否则说明卡住）：
  1) 采集：$OUT/round_1/transitions.pt 开始变大
  2) WM 门：$OUT/round_1/gate_wm_qd.json 出现（里面 pass=true）
  3) B 线：$OUT/round_1/bline/b_f*.pt 出现
盯进度：
  tail -f $LOG
  watch -n 30 'ls -la $OUT/round_1/*.pt $OUT/round_1/*.json 2>/dev/null; tail -3 $OUT/rounds.csv 2>/dev/null'
EOF
