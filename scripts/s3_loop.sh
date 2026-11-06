#!/usr/bin/env bash
# S3 **多轮闭环** driver（2×4090）：每一步都**同时用满两张卡**。
#
#   stage=train   GPU0 = 主臂(λ_R=0.3)      GPU1 = 对照臂(λ_R=0)      ← 两卡并行训练
#   stage=eval    GPU0 = 主臂 head + Isaac   GPU1 = 对照臂 head + Isaac ← 两卡并行闭环
#   stage=collect GPU0 = server + Isaac     GPU1 = Isaac              ← 两卡并行采集
#   stage=report  CPU + 两卡打分/画图
#
# 用法：
#   ROUND=1 bash scripts/s3_loop.sh train      # 第 1 轮：从官方底模 π_0 起
#   ROUND=1 bash scripts/s3_loop.sh eval       # 两卡并行闭环（DoubleDesk row0/seed0/300 步）
#   ROUND=1 bash scripts/s3_loop.sh collect    # 用胜出的 head 采 B_r2
#   ROUND=1 bash scripts/s3_loop.sh report     # 打分 + 漂移图 + round json
#
#   ROUND=2 INIT_VLA=runs/S3_loop/r1/train_main/vla_000300.pt B_GLOB='runs/B_r2/vla_*_f*_r*/dump/*.pt' \
#     bash scripts/s3_loop.sh train
#
# 关键环境变量：
#   ROUND        轮次（1/2/3…）；决定目录、B 数据、INIT_VLA 的默认值
#   STEPS        训练步数（默认 300）
#   INIT_VLA     本轮 π_k 起点（上一轮的 vla_*.pt）；ROUND=1 留空 = 官方底模
#   B_GLOB       本轮 `--start-from-b` 的 glob（默认按 ROUND 推）
#   WBC_CKPT     WM/WBC（默认 v6，已过 G0-b）
#   LAMBDA_R_MAIN / LAMBDA_R_CTRL   主臂 / 对照臂的奖励权重（0.3 / 0.0）
#   EVAL_STEPS   闭环验收步数（默认 300，与 0.200 m 基线可比）
#   E_ARM        collect 用哪条臂的 head（main / ctrl，默认 main）
#   COLLECT_STEPS / COLLECT_EPISODES  采集步数 / 每任务集数（600 / 1）
set -uo pipefail

COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1
export COMIMIC

ROUND=${ROUND:-1}
RUNS=$COMIMIC/runs/S3_loop
# 2026-10-29：目录名可以单独指定（ROUND 必须是数字，用于算 INIT_VLA 默认值）。
# 例：从基线重开一条干净序列 -> ROUND=11 RUN_LABEL=b1
RUN_LABEL=${RUN_LABEL:-r$ROUND}
RD=$RUNS/$RUN_LABEL
STEPS=${STEPS:-300}
WBC_CKPT=${WBC_CKPT:-$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt}
BASE_CKPT=${BASE_CKPT:-$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}
LAMBDA_R_MAIN=${LAMBDA_R_MAIN:-0.3}
LAMBDA_R_CTRL=${LAMBDA_R_CTRL:-0.0}
LAMBDA_BC=${LAMBDA_BC:-0}
TOKEN_BC=${TOKEN_BC:-0}         # v3 的 token BC 权重（0=旧行为）
TOKEN_BC_SOURCE=${TOKEN_BC_SOURCE:-ref}   # ref=参考 token / self=策略自己的 token
TOKEN_BC_SPAN=${TOKEN_BC_SPAN:-0}         # self 时必须 20（只有前 20 步属于那次观测）
B_SELF_TARGETS=${B_SELF_TARGETS:-}        # build_b_self_targets.py 的产物
B_SELF_ERR_MAX=${B_SELF_ERR_MAX:-}        # 只保留实际漂移 ≤ 该值的时刻
B_START_MAX_ERR=${B_START_MAX_ERR:-}      # 只保留瞬时误差 ≤ 该值的起点（奖励核有梯度）
TOKEN_BC_MAIN=${TOKEN_BC_MAIN:-}          # 每臂覆盖（留空=用 TOKEN_BC）
TOKEN_BC_CTRL=${TOKEN_BC_CTRL:-}
FM_LR=${FM_LR:-1e-4}            # head(LoRA) 学习率
BATCH=${BATCH:-4}                 # micro-batch（2026-10-29：数据涨到 42.5k 转移后可以开大）
ACCUM=${ACCUM:-8}                 # 梯度累积；有效 batch = BATCH*ACCUM
LORA_ALPHA=${LORA_ALPHA:-}        # LoRA 缩放（空=默认=rank）；放大 alpha 等于放大同样权重的 ΔW
# 2026-10-29 审计：这三项原来**硬编码**（lambda-hand 恒 0.2），导致"说改了其实没改"。
# 现在默认按审计结论走：手部 0（部署 SM_HAND=0 用不到）、加一点正则。
LAMBDA_HAND=${LAMBDA_HAND:-0}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
OBS_NOISE=${OBS_NOISE:-0}   # WM 当裁判时必须 0：obs_noise 会给"想象"注入噪声 ⇒ 回报变噪
REWARD_MODE=${REWARD_MODE:-saturating}
POSTURE_WEIGHT=${POSTURE_WEIGHT:-0.0}
REWARD_VEL_SCALE=${REWARD_VEL_SCALE:-1.0}
REWARD_ROOT_WEIGHT=${REWARD_ROOT_WEIGHT:-}
REWARD_ROOT_POS_STD=${REWARD_ROOT_POS_STD:-}
REWARD_ROOT_ORI_STD=${REWARD_ROOT_ORI_STD:-}
ROOT_ORI_MODE=${ROOT_ORI_MODE:-split}
TAIL_WEIGHT=${TAIL_WEIGHT:-2.0}
TAIL_POS_THR=${TAIL_POS_THR:-}            # 根位置 hinge e_h（空=reward_shape.ROOT_POS_THR）
TAIL_POS_CAP=${TAIL_POS_CAP:-}            # 根位置 hinge cap（空=reward_shape.ROOT_POS_CAP）
B_FRESH_GLOB=${B_FRESH_GLOB:-}
B_FRESH_WEIGHT=${B_FRESH_WEIGHT:-3.0}
H=${H:-20}                                # 想象步数（WM 误差随 H 迅速变大：H=5 时 root 1.1cm / H=20 时 6.9cm）
CRITIC_ON=${CRITIC_ON:-0}
CRITIC_SOURCE=${CRITIC_SOURCE:-imagined}
CRITIC_REAL=${CRITIC_REAL:-}
CRITIC_CKPT=${CRITIC_CKPT:-}
CRITIC_USE_DIFF=${CRITIC_USE_DIFF:-0}
GRAD_LOG_EVERY=${GRAD_LOG_EVERY:-0}
CRITIC_GRAD_STATE=${CRITIC_GRAD_STATE:-0}  # 1 = critic 对 s_H 保留梯度（SHAC 式，延长视野）
# ★ 2026-10-29：critic 消融必须**同会话配对**（两卡同时、同种子）才判得了，
#   所以要能按臂开关。留空 = 继承全局 CRITIC_ON / CRITIC_GRAD_STATE。
CRITIC_ON_MAIN=${CRITIC_ON_MAIN:-}
CRITIC_ON_CTRL=${CRITIC_ON_CTRL:-}
# ★ 2026-11-01：长视界梯度（穿过 critic 到状态的那一路）的缩放系数。
#   原来硬编码 0.1（= 削 10 倍）；`probe_wm_gradient_truth` 已证明 WM 梯度方向可信（98–100%），
#   所以这个"怕梯度爆炸"的保险应该按需放开。1.0 = 不削。
CRITIC_STATE_GRAD_SCALE=${CRITIC_STATE_GRAD_SCALE:-0.1}
CRITIC_GRAD_STATE_MAIN=${CRITIC_GRAD_STATE_MAIN:-}
CRITIC_GRAD_STATE_CTRL=${CRITIC_GRAD_STATE_CTRL:-}
# 两条臂可以跑**不同剂量**（剂量梯度实验）：留空则两条臂都用 FM_LR / STEPS
FM_LR_MAIN=${FM_LR_MAIN:-}
FM_LR_CTRL=${FM_LR_CTRL:-}
STEPS_MAIN=${STEPS_MAIN:-}
STEPS_CTRL=${STEPS_CTRL:-}
# 2026-10-29：解冻执行器（WBC）。默认两条臂都关；可每臂单独开。
TRAIN_WBC_MAIN=${TRAIN_WBC_MAIN:-0}
TRAIN_WBC_CTRL=${TRAIN_WBC_CTRL:-0}
WBC_LR=${WBC_LR:-1e-6}
WBC_LR_MAIN=${WBC_LR_MAIN:-}   # 每臂覆盖（空=用 WBC_LR）
WBC_LR_CTRL=${WBC_LR_CTRL:-}
TRUST_CENTER=${TRUST_CENTER:-self}
ACT_EPS=${ACT_EPS:-0.6}
TRUST_EPS_N=${TRUST_EPS:-0}          # token-space hinge 容差（0=关）
TRUST_WEIGHT=${TRUST_WEIGHT:-0}
EVAL_STEPS=${EVAL_STEPS:-300}
EVAL_EVERY=${EVAL_EVERY:-100}
E_ARM=${E_ARM:-main}
NEXT_ROUND=${NEXT_ROUND:-}   # 采集落盘到 B_r??；默认 ROUND+1
HEAD_COLLECT=${HEAD_COLLECT:-}
HEAD_MAIN=${HEAD_MAIN:-}
HEAD_CTRL=${HEAD_CTRL:-}
EVAL_TAG=${EVAL_TAG:-}
COLLECT_STEPS=${COLLECT_STEPS:-600}
COLLECT_EPISODES=${COLLECT_EPISODES:-1}
TASKS=${TASKS:-"DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi"}

# 默认：第 1 轮从官方底模起、用 B_r1；第 k>=2 轮从上一轮 head 起、用 B_rk
INIT_VLA=${INIT_VLA:-}
if [[ -z "$INIT_VLA" && "$ROUND" -gt 1 ]]; then
  for c in "$RUNS/r$((ROUND-1))/train_main/vla_$(printf %06d "$STEPS").pt" \
           "$RUNS/r$((ROUND-1))/train_main/vla_final.pt"; do
    [[ -f "$c" ]] && { INIT_VLA=$c; break; }
  done
fi
B_GLOB=${B_GLOB:-}
if [[ -z "$B_GLOB" ]]; then
  src=$(( ROUND == 1 ? 1 : ROUND ))
  B_GLOB="runs/B_r$src/vla_*_f*_r*/dump/*.pt"
fi
mkdir -p "$RUNS" "$RD" "$COMIMIC/logs"

PORT_MAIN=${PORT_MAIN:-18090}
PORT_CTRL=${PORT_CTRL:-18091}
PORT_COLLECT=${PORT_COLLECT:-18092}
# 部署端 server 额外参数（如 "--sample-avg 4"）；采集/评测的 server 都会带上。
SRV_EXTRA_VAL=${SRV_EXTRA_VAL:-}

stamp() { date +%H:%M:%S; }
say()   { echo "[$(stamp)] [loop r$ROUND] $*"; }

# ------------------------------------------------------------------ server 管理
# 2026-10-28 踩过：上一阶段的 server 没退干净，`wait_server` 看到端口已开就立刻放行，
# 结果 Isaac 打到的是**旧 head**，scan/eval 全串台（日志里 "就绪（0s）" 就是征兆）。
# 所以这里不按 pid 猜，直接按 /proc 里的 cmdline 精确匹配端口，并且**校验 head 路径**。
pids_on_port() {  # port -> 该端口上我们 server 的 pid 列表
  local port=$1 pid cl
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
    cl=$( { tr '\0' ' ' < "$d/cmdline"; } 2>/dev/null ) || continue
    case "$cl" in
      *ha_vla_latent64_server*"--port $port"*) echo "$pid" ;;
    esac
  done
}

verify_server_head() {  # port head
  local port=$1 head=$2 pid cl
  for pid in $(pids_on_port "$port"); do
    cl=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
    case "$cl" in *"--head-ckpt $head"*) return 0 ;; esac
  done
  return 1
}

kill_servers() {  # [ports...] 默认清三个固定端口
  local -a ports=("$@")
  (( ${#ports[@]} )) || ports=("$PORT_MAIN" "$PORT_CTRL" "$PORT_COLLECT")
  local port pids i busy
  for p in "${_SPIDS[@]:-}"; do
    [[ -n "${p:-}" ]] && kill "$p" 2>/dev/null
  done
  for port in "${ports[@]}"; do
    pids=$(pids_on_port "$port")
    [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null
  done
  for i in $(seq 1 40); do
    busy=0
    for port in "${ports[@]}"; do
      [[ -n "$(pids_on_port "$port")" ]] && busy=1
    done
    (( busy == 0 )) && { _SPIDS=(); return 0; }
    sleep 1
  done
  say "[FAIL] 端口 ${ports[*]} 上还有 server 没退出"; _SPIDS=(); return 1
}

_SPIDS=()
start_server() {  # port gpu head_ckpt logdir tag
  local port=$1 gpu=$2 head=$3 logdir=$4 tag=$5
  mkdir -p "$logdir"
  say "起 server tag=$tag port=$port gpu=$gpu head=$(basename "$head" 2>/dev/null || echo base)"
  PORT=$port GPU=$gpu HEAD_CKPT="$head" LOG_DIR="$logdir" \
    LORA_RANK=32 LORA_TARGETS="attn1,ff" LORA_ALSO_TRAIN="proj_out" SRV_EXTRA="$SRV_EXTRA_VAL" \
    bash "$COMIMIC/scripts/start_vla_server.sh" \
    > "$COMIMIC/logs/s3_loop_r${ROUND}_srv_${tag}.log" 2>&1 &
  _SPIDS+=("$!")
}

wait_server() {  # port head [timeout_s]
  local port=$1 head=${2:-} tmo=${3:-600} t0=$SECONDS
  while (( SECONDS - t0 < tmo )); do
    if (echo > "/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
      if [[ -z "$head" ]] || verify_server_head "$port" "$head"; then
        say "server :$port 就绪（$((SECONDS-t0))s，head=$(basename "${head:-<base>}")）"
        return 0
      fi
      say "[FAIL] :$port 上跑的不是 $(basename "$head")（旧 server 没退）"
      return 1
    fi
    sleep 3
  done
  say "[FAIL] server :$port ${tmo}s 内没起来（看 logs/s3_loop_r${ROUND}_srv_*.log）"
  return 1
}

pick_head() {  # dir
  local d=$1 c
  c="$d/vla_$(printf %06d "$STEPS").pt"
  [[ -f "$c" ]] && { echo "$c"; return 0; }
  c="$d/vla_final.pt"
  [[ -f "$c" ]] && { echo "$c"; return 0; }
  return 1
}

metrics_tail() {  # dir -> 最后一行 metrics 的摘要
  local f=$1/train_metrics.jsonl
  [[ -f "$f" ]] || { echo "(no metrics)"; return; }
  python3 - "$f" <<'PY'
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if not rows: print("(empty)"); raise SystemExit
r=rows[-1]
print(f"step={r.get('step')} s/step={r.get('s_per_step',0):.2f} "
      f"rollout_reward={r.get('rollout/reward',float('nan')):.5f} "
      f"codec_oor={r.get('codec/out_of_range',float('nan')):.4f} "
      f"act_frac_out={r.get('act/frac_out',float('nan')):.3f} "
      f"nan={(str(r.get('loss'))=='nan')}")
PY
}

# ------------------------------------------------------------------ stage: train
stage_train() {
  say "===== TRAIN（两卡并行）====="
  say "π_k 起点 = ${INIT_VLA:-<官方底模>}"
  say "B 数据   = $B_GLOB"
  local extra="--amp bf16 --lora-rank 32 --fm-lr $FM_LR --no-dr --vla-state-from-burnin --act-hinge-cap 2"
  [[ "$GRAD_LOG_EVERY" != "0" ]] && extra="$extra --grad-log-every $GRAD_LOG_EVERY"
  [[ -n "$LORA_ALPHA" ]] && extra="$extra --lora-alpha $LORA_ALPHA"
  # ★ critic 相关一律**按臂**拼接（见下方 arm_extra），全局那份已删除，
  #   否则关掉 critic 的那条臂还是会拿到 --critic-grad-through-state，消融就废了。
  # token 空间 hinge 是 v3 里**唯一**能把 head 拉向标签的信号（v3 没有 token BC）；
  # trust_center=prev 会把它强制关掉（见 v3_step.py:136），所以只有 label 中心才生效。
  if [[ "$TRUST_EPS_N" != "0" && -n "$TRUST_WEIGHT" ]]; then
    extra="$extra --trust-eps $TRUST_EPS_N --trust-weight $TRUST_WEIGHT"
  fi
  local -a init_arg=()
  [[ -n "$INIT_VLA" ]] && init_arg=(INIT_VLA="$INIT_VLA")

  # ★ 2026-10-31：允许只训主线（默认仍两臂）。原来两臂配置完全一样 = 纯重复，白占一张卡。
  for arm in ${TRAIN_ARMS:-main ctrl}; do
    # ★ 2026-10-31（消融）：允许显式指定卡片（默认 main→0 / ctrl→1）。
    #   用途：两张卡同时跑**两个不同的消融臂**（各自 TRAIN_ARMS=main，GPU_MAIN 分别 0/1），
    #   4 条消融的时间从「4×30min」压到「2×30min」，且互相不共享显存。
    local lr=$LAMBDA_R_MAIN gpu=${GPU_MAIN:-0}
    [[ "$arm" == "ctrl" ]] && { lr=$LAMBDA_R_CTRL; gpu=${GPU_CTRL:-1}; }
    # 每臂剂量覆盖（空 = 用全局）
    local arm_lr=$FM_LR arm_steps=$STEPS arm_wbc=0 arm_tbc=$TOKEN_BC arm_wl=$WBC_LR
    if [[ "$arm" == "main" ]]; then
      [[ "$TRAIN_WBC_MAIN" == "1" ]] && arm_wbc=1
    else
      [[ "$TRAIN_WBC_CTRL" == "1" ]] && arm_wbc=1
    fi
    if [[ "$arm" == "main" ]]; then
      [[ -n "$FM_LR_MAIN" ]] && arm_lr=$FM_LR_MAIN
      [[ -n "$STEPS_MAIN" ]] && arm_steps=$STEPS_MAIN
      [[ -n "$TOKEN_BC_MAIN" ]] && arm_tbc=$TOKEN_BC_MAIN
      [[ -n "$WBC_LR_MAIN" ]] && arm_wl=$WBC_LR_MAIN
    else
      [[ -n "$FM_LR_CTRL" ]] && arm_lr=$FM_LR_CTRL
      [[ -n "$STEPS_CTRL" ]] && arm_steps=$STEPS_CTRL
      [[ -n "$TOKEN_BC_CTRL" ]] && arm_tbc=$TOKEN_BC_CTRL
      [[ -n "$WBC_LR_CTRL" ]] && arm_wl=$WBC_LR_CTRL
    fi
    # ★ critic 按臂（消融）：留空则继承全局
    local arm_critic=$CRITIC_ON arm_cgs=$CRITIC_GRAD_STATE
    if [[ "$arm" == "main" ]]; then
      [[ -n "$CRITIC_ON_MAIN" ]] && arm_critic=$CRITIC_ON_MAIN
      [[ -n "$CRITIC_GRAD_STATE_MAIN" ]] && arm_cgs=$CRITIC_GRAD_STATE_MAIN
    else
      [[ -n "$CRITIC_ON_CTRL" ]] && arm_critic=$CRITIC_ON_CTRL
      [[ -n "$CRITIC_GRAD_STATE_CTRL" ]] && arm_cgs=$CRITIC_GRAD_STATE_CTRL
    fi
    local arm_extra="${extra/--fm-lr $FM_LR/--fm-lr $arm_lr}"
    [[ "$arm_cgs" == "1" ]] && arm_extra="$arm_extra --critic-grad-through-state --critic-state-grad-scale $CRITIC_STATE_GRAD_SCALE --critic-rel-clip 5"
    local out=$RD/train_$arm
    rm -rf "$out"
    say "launch arm=$arm gpu=$gpu lambda_r=$lr fm_lr=$arm_lr steps=$arm_steps token_bc=$TOKEN_BC train_wbc=$arm_wbc(wbc_lr=$arm_wl) critic=$arm_critic(grad_state=$arm_cgs) center=$TRUST_CENTER act_eps=$ACT_EPS act_w=${ACT_WEIGHT:-1.0} yaw_hinge_lambda=${YAW_HINGE_LAMBDA:-0.3} out=$out"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      say "[dry-run] 不启动：arm=$arm gpu=$gpu out=$out"
      say "[dry-run] env: LAMBDA_R=$lr ACT_EPS=$ACT_EPS ACT_WEIGHT=${ACT_WEIGHT:-1.0} STATE_EXTRA=${STATE_EXTRA:-none} ADV_BASELINE=${ADV_BASELINE:-none} YAW_HINGE_LAMBDA=${YAW_HINGE_LAMBDA:-0.3}"
      continue
    fi
    # 2026-11-02：参考起点开关（START_FROM=reference ⇒ 不传 --start-from-b）
    if [[ "${START_FROM:-b}" == "reference" ]]; then _SFB=""; else _SFB="$B_GLOB"; fi
    env V3=1 TRUST_CENTER="$TRUST_CENTER" START_FROM_B="$_SFB" MOTION_PACKED="${MOTION_PACKED:-}" HUBER_DELTA="${HUBER_DELTA:-0.02}" HUBER_CAP="${HUBER_CAP:-0.15}" GAMMA="${GAMMA:-0.99}" START_SAMPLER="${START_SAMPLER:-uniform}" START_WEIGHTS="${START_WEIGHTS:-}" START_AUDIT="${START_AUDIT:-0}" ALLOW_LEGACY_REWARD="${ALLOW_LEGACY_REWARD:-0}" REWARD_BODY_POS_STD="${REWARD_BODY_POS_STD:-}" REWARD_BODY_ORI_STD="${REWARD_BODY_ORI_STD:-}" REWARD_DELTA_POS="${REWARD_DELTA_POS:-0.05}" REWARD_DELTA_ORI="${REWARD_DELTA_ORI:-0.05}" WM_CKPT="${WM_CKPT:-}" \
      ACT_EPS="$ACT_EPS" ACT_WEIGHT="${ACT_WEIGHT:-1.0}" LAMBDA_R="$lr" LAMBDA_BC="$LAMBDA_BC" \
      YAW_HINGE_LAMBDA="${YAW_HINGE_LAMBDA:-0.3}" \
      RUN=${RUN_TAG:-r${ROUND}_${arm}} OUT="$out" BATCH="$BATCH" ACCUM="$ACCUM" H="$H" K=4 STEPS="$arm_steps" \
      EVAL_EVERY="$EVAL_EVERY" EVAL_BATCHES=8 SAVE_EVERY=50 TEE=0 GPU="$gpu" TOKEN_BC="$arm_tbc" \
      WBC_CKPT="$WBC_CKPT" EXTRA="$arm_extra" \
      TRAIN_WBC="$arm_wbc" WBC_LR="$arm_wl" \
      WBC_UNFREEZE="${WBC_UNFREEZE:-all}" \
      WBC_KL_ANCHOR="${WBC_KL_ANCHOR:-0}" VLA_KL_ANCHOR="${VLA_KL_ANCHOR:-0}" \
      LAMBDA_HAND="$LAMBDA_HAND" WEIGHT_DECAY="$WEIGHT_DECAY" OBS_NOISE="$OBS_NOISE" TOKEN_BC_SPAN="$TOKEN_BC_SPAN" \
      REWARD_MODE="$REWARD_MODE" POSTURE_WEIGHT="$POSTURE_WEIGHT" REWARD_VEL_SCALE="$REWARD_VEL_SCALE" \
      REWARD_ROOT_WEIGHT="$REWARD_ROOT_WEIGHT" REWARD_ROOT_POS_STD="$REWARD_ROOT_POS_STD" \
      REWARD_ROOT_ORI_STD="$REWARD_ROOT_ORI_STD" \
      ROOT_ORI_MODE="$ROOT_ORI_MODE" TAIL_WEIGHT="$TAIL_WEIGHT" \
      TAIL_POS_THR="$TAIL_POS_THR" TAIL_POS_CAP="$TAIL_POS_CAP" \
      B_FRESH_GLOB="$B_FRESH_GLOB" B_FRESH_WEIGHT="$B_FRESH_WEIGHT" \
      B_START_MAX_ERR="$B_START_MAX_ERR" \
      CRITIC_ON="$arm_critic" CRITIC_SOURCE="$CRITIC_SOURCE" CRITIC_REAL="$CRITIC_REAL" \
      CRITIC_CKPT="$CRITIC_CKPT" CRITIC_USE_DIFF="$CRITIC_USE_DIFF" \
      TOKEN_BC_SOURCE="$TOKEN_BC_SOURCE" B_SELF_TARGETS="$B_SELF_TARGETS" \
      B_SELF_ERR_MAX="$B_SELF_ERR_MAX" "${init_arg[@]}" \
      bash "$COMIMIC/scripts/launch_joint_ha_v2.sh" \
      > "$COMIMIC/logs/s3_loop_${RUN_TAG:-r${ROUND}_${arm}}.log" 2>&1 &
  done
  wait
  if [[ "${DRY_RUN:-0}" == "1" ]]; then say "[dry-run] 训练未启动，跳过产物检查"; return 0; fi
  local ok=1
  for arm in ${TRAIN_ARMS:-main ctrl}; do
    local h
    if h=$(pick_head "$RD/train_$arm"); then
      say "arm=$arm head=$(basename "$h")  $(metrics_tail "$RD/train_$arm")"
    else
      say "[FAIL] arm=$arm 没有 vla_*.pt（看 logs/s3_loop_r${ROUND}_${arm}.log）"; ok=0
    fi
  done
  return $((1-ok))
}

# ------------------------------------------------------------------ stage: eval
stage_eval() {
  say "===== EVAL（两卡并行闭环，DoubleDesk row0/seed0/${EVAL_STEPS}步）====="
  local hm hc
  hm=${HEAD_MAIN:-$(pick_head "$RD/train_main")} || { say "[FAIL] main 没 head"; return 1; }
  hc=${HEAD_CTRL:-$(pick_head "$RD/train_ctrl")} || { say "[FAIL] ctrl 没 head"; return 1; }
  [[ -f "$hm" ]] || { say "[FAIL] HEAD_MAIN 不存在: $hm"; return 1; }
  [[ -f "$hc" ]] || { say "[FAIL] HEAD_CTRL 不存在: $hc"; return 1; }
  say "head_main=$(basename "$hm") head_ctrl=$(basename "$hc") tag=${EVAL_TAG:-<none>}"
  kill_servers "$PORT_MAIN" "$PORT_CTRL" || return 1     # 先清残兵
  start_server "$PORT_MAIN" 0 "$hm" "$RD/vla_in_main"  main >/dev/null
  start_server "$PORT_CTRL" 1 "$hc" "$RD/vla_in_ctrl"  ctrl >/dev/null
  wait_server "$PORT_MAIN" "$hm" || { kill_servers; return 1; }
  wait_server "$PORT_CTRL" "$hc" || { kill_servers; return 1; }

  local -a pids=()
  for arm in main ctrl; do
    local gpu=0 port=$PORT_MAIN
    [[ "$arm" == "ctrl" ]] && { gpu=1; port=$PORT_CTRL; }
    say "eval arm=$arm gpu=$gpu server=:$port"
    ARM=vla TASK=DoubleDesk SEED=0 START_FRAME=0 RESET_SEED=0 MAX_STEPS="$EVAL_STEPS" \
      GPU="$gpu" SERVER_URL="http://127.0.0.1:$port" SM_CKPT="$WBC_CKPT" \
      OUT="$RD/eval_${arm}${EVAL_TAG:+_$EVAL_TAG}" \
      bash "$COMIMIC/scripts/s3_collect_arm.sh" \
      > "$COMIMIC/logs/s3_loop_r${ROUND}_eval_${arm}.log" 2>&1 &
    pids+=($!)
  done
  wait
  kill_servers
  local ok=1
  for arm in main ctrl; do
    local n
    n=$(find "$RD/eval_${arm}${EVAL_TAG:+_$EVAL_TAG}/dump" -name '*.pt' 2>/dev/null | wc -l)
    say "eval arm=$arm dumps=$n"
    [[ "$n" -ge 1 ]] || ok=0
  done
  return $((1-ok))
}

# ------------------------------------------------------------------ stage: collect
stage_collect() {
  local next=${NEXT_ROUND:-$((ROUND+1))}
  local bdir=${B_NEXT_DIR:-$COMIMIC/runs/B_r$next}
  say "===== COLLECT（两卡并行，用 $E_ARM 臂的 head 采 B_r$next）====="
  local h
  h=${HEAD_COLLECT:-$(pick_head "$RD/train_$E_ARM")} || { say "[FAIL] $E_ARM 没 head"; return 1; }
  if [[ "$h" != "base" && ! -f "$h" ]]; then say "[FAIL] HEAD_COLLECT 不存在: $h"; return 1; fi
  [[ "$h" == "base" ]] && h=""    # 底模：server 不挂 head
  say "collect 用 head=$(basename "${h:-base}")"
  kill_servers "$PORT_COLLECT" || return 1
  start_server "$PORT_COLLECT" 0 "$h" "$COMIMIC/runs/B_r${next}/vla_in" head >/dev/null
  wait_server "$PORT_COLLECT" "$h" || { kill_servers; return 1; }
  mkdir -p "$bdir" "$COMIMIC/logs/B_r$next"
  TASKS="$TASKS" EPISODES="$COLLECT_EPISODES" ARMS=vla \
    JOBS="${COLLECT_JOBS:-2}" NUM_GPUS="${COLLECT_NUM_GPUS:-2}" \
    MAX_STEPS="$COLLECT_STEPS" START_FRAMES=random \
    ROOT="$bdir" LOGD="$COMIMIC/logs/B_r$next" \
    SERVER_URL="http://127.0.0.1:$PORT_COLLECT" \
    bash "$COMIMIC/scripts/s3_collect_parallel.sh" 2>&1 \
    | tee "$COMIMIC/logs/s3_loop_r${ROUND}_collect.log"
  local rc=${PIPESTATUS[0]}
  kill_servers
  local n
  n=$(find "$bdir" -name '*.pt' -path '*/dump/*' 2>/dev/null | wc -l)
  say "collect 完成 rc=$rc dumps=$n → $bdir"
  return $rc
}

# ------------------------------------------------------------------ stage: report
stage_report() {
  say "===== REPORT（打分 + 漂移图 + json）====="
  export PYTHONPATH="$COMIMIC${PYTHONPATH:+:$PYTHONPATH}"
  export SUPERMIMIC_HA_ALIGN=1
  local -a dumps=() labels=()
  # 基线（π_0）与 GT，与 0.200 / 0.082 m 那张表同口径
  [ -f "$COMIMIC/runs/B_r1/vla_DoubleDesk_f0_r0/dump/DoubleDesk_vla_ep000.pt" ] && \
    dumps+=("$COMIMIC/runs/B_r1/vla_DoubleDesk_f0_r0/dump/DoubleDesk_vla_ep000.pt")
  [ -f "$COMIMIC/runs/B_r1/gt_DoubleDesk_f0_r0/dump/DoubleDesk_gt_ep000.pt" ] && \
    dumps+=("$COMIMIC/runs/B_r1/gt_DoubleDesk_f0_r0/dump/DoubleDesk_gt_ep000.pt")
  for arm in main ctrl; do
    local f
    f=$(ls "$RD/eval_$arm"/dump/*.pt 2>/dev/null | head -1)
    [[ -n "$f" ]] && dumps+=("$f")
  done
  if (( ${#dumps[@]} == 0 )); then say "[FAIL] 没有可画的 dump"; return 1; fi

  python3 "$COMIMIC/scripts/plot_drift.py" \
    $(for d in "${dumps[@]}"; do printf -- '--dump %s ' "$d"; done) \
    --out "$RD/drift_r${ROUND}.png" --title "S3 round$ROUND: GT / pi0 / ${E_ARM}" \
    2>&1 | tee "$RD/drift_r${ROUND}.txt"

  # 打分（给奖励列；与训练同一 reward）
  for arm in main ctrl; do
    local d=$RD/eval_$arm
    if compgen -G "$d/dump/*.pt" >/dev/null; then
      python3 "$COMIMIC/scripts/compute_collect_reward.py" \
        --in "$d"/dump/*.pt --out "$d/scored.pt" --wbc-ckpt "$WBC_CKPT" --device cuda:0 \
        >> "$RD/scored.log" 2>&1 && say "scored arm=$arm"
    fi
  done
  say "报告 → $RD/drift_r${ROUND}.png / .txt"
}

# ------------------------------------------------------------------ dispatch
case "${1:-}" in
  train)   stage_train ;;
  eval)    stage_eval ;;
  collect) stage_collect ;;
  report)  stage_report ;;
  *) echo "用法: ROUND=N bash scripts/s3_loop.sh {train|eval|collect|report}" >&2; exit 2 ;;
esac
