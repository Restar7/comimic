#!/usr/bin/env bash
# **自动多轮闭环**（SuperMimic 式：采集 → 刷 WM → 训 critic → 联合微调 → 闭环验收 → 再来一轮）
#
# 每一轮 r：
#   1) 大规模仿真采集（向量化）           → round_r/transitions.pt（真转移 + 奖励 + ret）
#   2) WM 验收门（比仿真真值）            → 不过就停
#   3) B 线回放采集（frames=真库行号）    → round_r/bline/*.pt（第 1 轮 arm=gt，第 2 轮起 arm=vla）
#   4) 五道门禁                            → 不过就停
#   5) critic 拟合（真 return-to-go）      → v_ev>0.30 ⇒ **下一轮**开 critic；否则继续关
#   6) 联合微调（λ_bc=0 / hand 0.2 / rl 0.3 / H=20 / act 0.6）→ round_r/train/
#   7) 合并 head → 闭环评测（同一任务同起点） + 8-clip benchmark
#   8) 连续 2 轮 reward 不涨 ⇒ 停（对应 S3 的 G5 止损）
#
# 用法：
#   ROUNDS=3 bash scripts/run_j1_rounds.sh                 # 自动跑 3 轮
#   ROUNDS=5 PATIENCE=2 bash scripts/run_j1_rounds.sh
#   DRY=1 bash scripts/run_j1_rounds.sh                    # 只打印每轮要执行的命令
#   ROUNDS=1 REWARD_STEPS=800 bash scripts/run_j1_rounds.sh
set -euo pipefail

COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
REPO=$COMIMIC/third_party/SuperMimic-Reproduced
VENV=$REPO/.venv/bin/python
TORCHRUN=$REPO/.venv/bin/torchrun
PY=python3

# ---------------- 锁定项（与 docs/HANDOVER_4090_COMMANDS_20261028.md 对齐）----------------
# 2026-10-30：318 微调重做（干净基座 + v3 参考 + 官方奖励口径）。
# 训练产出在 logs/wbc_ft318_v3pace/，按 oracle 路径比 0.9-1.1 + tilt<10° 挑步数后钉死这里。
# 2026-10-30 定版：v10（去位置项 + speed 加权 + 官方其余 13 项），
# 按关节 p50 全曲线最优取 1k（0.128 rad；基座 0.204，20k 0.168）。
W1_CKPT=${W1_CKPT:-$REPO/logs/wbc_ft318_v10/ckpt_0001000.pt}
VLA_CKPT=${VLA_CKPT:-/inspire/hdd/global_user/liumengfan-253108110079/flowbody/FlowBody/runs/groot_sonic_vla_20k_frozen271_20261023/checkpoint-20000}
VLM_PATH=${VLM_PATH:-$COMIMIC/groot_repro/nvidia/Cosmos-Reason2-2B-weights}
# 参考库：v3 = 根轨迹(planner 重建) + 配速校正(路径/指令=1.00)；z/关节/朝向与 v1 逐位一致
LIB=${LIB:-$COMIMIC/data/library/ft318_compact_v8.pt}
TOK=$COMIMIC/data/library/ft318_tokens.pt
# npz 参考：必须与 LIB 同源（v1 的 npz 根位置恒 0，会与 v3 打架）
MOTION=${MOTION:-$COMIMIC/data/npz/g1_tasks_v8}
OUT=${OUT:-$COMIMIC/runs/j1_rounds}
TASK=${TASK:-task1}
FROM_FRAME=${FROM_FRAME:-100}     # 闭环验收/回放采集的起始库行号（固定，跨轮可比）
BLINE_TASKS=${BLINE_TASKS:-task1 task2 task3 task4}
BLINE_EPISODES=${BLINE_EPISODES:-20}   # 每任务集数
# 80 集健康门（2026-10-29）：摔倒数默认 0；平均奖励下限建议填 π0（ori 0.25 尺子）的 0.0859 略低
# state 项（2026-10-30）：把**误差直接反传**（mse(pos)+角²+mse(q)+w·mse(qd)）穿过可微 WM。
# lambda_s 在 v3 里原本**空转**（只在 v2 的 train_step 被读），现已接进 v3。
# 注意量纲：loss_s ~1e-5，而 lambda_r*loss_r ~0.2 ⇒ lambda_s 需要 O(1e3) 才有效，
# 所以这里默认 0（关），实验时按 measured `loss/state` 反算填 J1_LAMBDA_S。
J1_LAMBDA_S=${J1_LAMBDA_S:-0}
J1_STATE_VEL=${J1_STATE_VEL:-0.05}
# 固定起点评测集（2026-10-30）：训练用的 B 线保持 seed=$R（多样性），
# 但另跑一套**固定 seed**的小评测集 ⇒ 每轮在同一批起点上量，跨轮可直接比。
# 为什么需要：seed 每轮变会让"起点不同"的噪声掩盖早期退化（R3..R6 那次就是这样漏看的）。
BLINE_EVAL_SET=${BLINE_EVAL_SET:-1}
BLINE_EVAL_SEED=${BLINE_EVAL_SEED:-999}
BLINE_EVAL_PER_TASK=${BLINE_EVAL_PER_TASK:-5}
BLINE_HEALTH_GATE=${BLINE_HEALTH_GATE:-1}
BLINE_MIN_REWARD=${BLINE_MIN_REWARD:-0.0800}
BLINE_JOBS=${BLINE_JOBS:-2}            # 并发（= GPU 数）
BLINE_STARTS=${BLINE_STARTS:-100 2000 9000 20000 40000 80000 120000 160000 200000 240000}
# auto = 起点按**该任务自己的库行区间**均匀取（20 集铺满整个任务），这是 2026-10-29 修的真 bug：
# 库是 4 个任务首尾相接的（task1 [0,183567) / task2 [183567,275453) / task3 [275453,336100) /
# task4 [336100,473876)），而 --start-frame 是全局行号、跟 --task 无关 ⇒ 旧写法让 task3/task4
# 一集都落在自己区间里（全从 task_1 的动作中起跑）。explicit = 用上面的 BLINE_STARTS 原样。
BLINE_STARTS_MODE=${BLINE_STARTS_MODE:-auto}
BLINE_TICKS=${BLINE_TICKS:-600}
EVAL_TICKS=${EVAL_TICKS:-600}

ROUNDS=${ROUNDS:-6}          # 跑几轮（配合 START_ROUND 可以从第 N 轮接续）
START_ROUND=${START_ROUND:-1}   # 从第几轮开始编号（中断后接续用，见 docs §1.1）
WM_REFRESH_EVERY=${WM_REFRESH_EVERY:-0}   # 训练**中**保鲜间隔（默认 0：改用下面的训练前训）
WM_PRETRAIN_STEPS=${WM_PRETRAIN_STEPS:-3000}
# 第一个采集改成**带场景**（2026-10-29 用户口径）：4 个任务各采一份再合并。
# 空场景 SM_SCENE_TASKS="" 时退回原来的"平地无物体"单次采集。
SM_SCENE_TASKS=${SM_SCENE_TASKS:-task1 task2 task3 task4}
SCENE_ENVS=${SCENE_ENVS:-256}
SCENE_STEPS=${SCENE_STEPS:-1000}   # 训练**前**用真转移（含25%B线）把 WM 训稳，之后冻结
WM_REFRESH_STEPS=${WM_REFRESH_STEPS:-200}
PATIENCE=${PATIENCE:-2}          # 连续几轮不涨就停（涨的定义：闭环 mean reward 提升 >1%）

ENVS=${ENVS:-1024}          # 采集只需 ~1M 条；4096×4000 会产 1640 万条/19 GB，过量
STEPS=${STEPS:-1000}        # 每个 env 推进的控制 tick 数（并行推进，不是串行 4000 步）
CAP=${CAP:-1000}
REWARD_STEPS=${REWARD_STEPS:-2000}
J1_BATCH=${J1_BATCH:-64}
J1_LORA=${J1_LORA:-64}
J1_H=${J1_H:-20}
J1_FMLR=${J1_FMLR:-1e-4}
J1_LAMBDA_BC=${J1_LAMBDA_BC:-0}   # >0 时给 token 一个小 BC 正则，压 codec/out_of_range
NGPU=${NGPU:-2}
GPUS=${GPUS:-0,1}
AMP=${AMP:-bf16}
SMOKE=${SMOKE:-0}
DRY=${DRY:-0}
TRAIN_WBC=${TRAIN_WBC:-0}    # 2026-10-29 实测：解冻 WBC 800 步后同 head 闭环 root_pos 4.2cm→89cm（21×）
                             # ⇒ 默认改 0（只微调 head）。要复现消融就显式 TRAIN_WBC=1
WBC_LR=${WBC_LR:-1e-6}
MIN_GAIN=${MIN_GAIN:-0.01}   # 闭环 mean reward 相对提升超过 1% 才算"涨"
# 门禁策略：big=严格（不过就停）；small=auto（GPU 太小、门禁 OOM 时降级继续，并记进 round 目录）
# 注意：这里不能用 `| head -1`。`set -o pipefail` 下 nvidia-smi 写第 2..4 张卡时收 SIGPIPE（141），
# 管道判失败 ⇒ 触发 `|| echo 0`，VRAM_MB 变成 "49140\n0"，后面所有 [[ -gt ]] 全部语法报错、自动降级失效。
# 全读进来再切第一行，才是没有 SIGPIPE 的写法。
VRAM_ALL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)
VRAM_MB=${VRAM_ALL%%$'\n'*}
VRAM_MB=${VRAM_MB//[^0-9]/}
VRAM_MB=${VRAM_MB:-0}
GATES=${GATES:-$([[ ${VRAM_MB:-0} -ge 100000 ]] && echo strict || echo auto)}
if [[ "$GATES" == "auto" && ${VRAM_MB:-0} -lt 100000 ]]; then
  echo "[gate-policy] 显存 ${VRAM_MB} MiB < 100 GiB ⇒ GATES=auto（门禁 OOM 时降级继续，标记到 round 目录）"
fi

if [[ $SMOKE == 1 ]]; then
  ENVS=64; STEPS=60; CAP=60; REWARD_STEPS=4; J1_BATCH=2; J1_H=4; J1_LORA=16
  BLINE_TICKS=60; EVAL_TICKS=20; BLINE_STARTS="100"; ROUNDS=1
fi

# batch 上限按卡自动收（4090 48G 单卡 16 是实测天花板，32 必 OOM；H200 141G 能上 64）。
# 不设这条的话在 4090 上照 H200 口径跑，J1 第 12 步左右 OOM 白等一小时。
if [[ ${VRAM_MB:-0} -gt 0 && ${VRAM_MB:-0} -lt 100000 && $J1_BATCH -gt 16 ]]; then
  echo "[batch-policy] 显存 ${VRAM_MB} MiB < 100 GiB ⇒ J1_BATCH $J1_BATCH → 16（4090 单卡上限）"
  J1_BATCH=16
fi
if [[ ${VRAM_MB:-0} -gt 0 && ${VRAM_MB:-0} -lt 100000 && $ENVS -gt 512 ]]; then
  echo "[env-policy] 显存 ${VRAM_MB} MiB < 100 GiB ⇒ ENVS $ENVS → 512"
  ENVS=512; [[ ${CAP:-0} -gt 512 ]] && CAP=512
fi

export PYTHONPATH=$COMIMIC:$COMIMIC/Isaac-GR00T:$COMIMIC/third_party/gr00t_deps:$COMIMIC/third_party/gr00t_deps/_noflash
export MUJOCO_GL=disable

# HF_HOME 必须在**父进程**里导出，而不是在 python 里 os.environ.setdefault：
# third_party/gr00t_deps/sitecustomize.py 在解释器启动阶段就 import transformers → huggingface_hub，
# 那时 HF_HUB_CACHE 已经定死成 ~/.cache/huggingface/hub，之后再改环境变量完全无效。
# 2026-10-29 03:45 的 B 线就是因此去连 huggingface.co 拿 nvidia/Cosmos-Reason2-2B，
# 401 gated 直接挂了 4 集（本地 4.88 GB 权重其实一直躺在下面这个 cache 里）。
export HF_HOME=${HF_HOME:-/inspire/hdd/global_user/liumengfan-253108110079/cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
mkdir -p "$OUT"

say() { printf '\n\033[1m===== %s =====\033[0m\n' "$*"; }
run() { if [[ $DRY == 1 ]]; then echo "[dry] $*"; else "$@"; fi; }
need() { [[ -e "$1" ]] || { echo "[fail] 缺 $1"; exit 1; }; }

need "$W1_CKPT"; need "$VLA_CKPT"; need "$LIB"; need "$TOK"; need "$MOTION"

# WBC 解冻（用户口径 10-29：WBC 要开）⇒ 解冻后 B_freeze 门必然 FAIL，必须跳过
WBC_ARGS=(); [[ "$TRAIN_WBC" == "1" ]] && WBC_ARGS=(--train-wbc --wbc-lr "$WBC_LR")
GATE_LIST="A_codec C_burnin C2_token_contrast D_fm_bc"
[[ "$TRAIN_WBC" == "1" ]] || GATE_LIST="A_codec B_freeze $GATE_LIST"

WM_ARGS=(--wm-refresh-every "$WM_REFRESH_EVERY" --wm-refresh-steps "$WM_REFRESH_STEPS")
[[ "$WM_REFRESH_EVERY" == 0 ]] && WM_ARGS=(--wm-refresh-every 0 --wm-refresh-steps 0)
# critic 策略：auto=按 fit 的 v_ev 门自动开/关；on=**总是开**（fit 过不过都用它的 ckpt）；off=总是关
# 用户口径 2026-10-29：H=5 + critic 必须开（短 horizon 靠 value bootstrap 做信用分配）
CRITIC_MODE=${CRITIC_MODE:-auto}
# critic 目标来源：real = 监督回归真实 return-to-go（v_ev 只有 0.09–0.15，因为输入里没有 clip 进度/
#   身份，ret 主要由"还剩多少帧"决定 ⇒ 结构性不可学）；imagined = 在想象 rollout 上做 **GAE**
#   （短 horizon + bootstrap，方差小得多，不依赖 clip 身份）⇒ 想要 critic 真能用就用这个。
CRITIC_SOURCE=${CRITIC_SOURCE:-real}
# 2026-10-30：critic 目标类型。mc=整集 return-to-go（默认/推荐）；
#   gae=真轨迹 TD(λ)。R1 曾误用 CRITIC_SOURCE=imagined（8 步窗），已改为钉死 real。
CRITIC_TARGET=${CRITIC_TARGET:-mc}
CRITIC_RET_FILE=""
CRITIC_RET_ARGS=()
CRITIC_ARGS=(--no-critic)
CRITIC_NOTE="off"
BEST_R=""; BEST_ROUND=""; NO_GAIN=0
CUR_VLA=$VLA_CKPT
# 逐轮累积：第 1 轮从 W1 基线出发，之后每轮都用**上一轮训练产出**的 joint_final.pt（policy+wm）。
# 这样 WM 保鲜（--wm-refresh）的成果不会丢；B 线采集 / WM 门 / 评测 / 训练全部用同一个 CUR_WBC。
CUR_WBC=${CUR_WBC:-$W1_CKPT}

for R in $(seq "$START_ROUND" "$((START_ROUND + ROUNDS - 1))"); do
  RD=$OUT/round_$R
  mkdir -p "$RD"
  say "ROUND $R/$ROUNDS   (VLA=$CUR_VLA, critic=$CRITIC_NOTE)"

  # LAUNCH 必须在 RESUME_FROM_STEP 守卫**之外**定义（2026-10-30：跳过 1–6 步时
  # 原来它跟门禁一起被跳过 ⇒ LAUNCH 为空 ⇒ bash 直接 exec .py ⇒ Permission denied）
  LAUNCH=("$PY"); [[ $NGPU -gt 1 ]] && # 必须用**系统 python** 起 torchrun：venv 的 torchrun 会把 .venv/site-packages 放进
  # sys.path ⇒ `import mjlab` 命中**真 mjlab**（它要新版 tyro），与 gr00t_deps 的 tyro 0.9.17 冲突，
  # 报 `module 'tyro.conf' has no attribute 'UsePythonSyntaxForLiteralCollections'`（2026-10-28 21:18 实测）
  LAUNCH=("$PY" -m torch.distributed.run --nproc_per_node "$NGPU" --master_port "${PORT:-29680}")
  # ---- RESUME_FROM_STEP（2026-10-30）：奖励实验里步骤 1–6 与奖励**完全无关**
  #      闭环只把 reward 写进 rec_rewards、从不进控制回路 ⇒ 轨迹与奖励无关；
  #      WM 也只吃 states/actions/next_states ⇒ 可复用已验证的 wm_pre.pt 直接跳到 7/9。
  #      用法：RESUME_FROM_STEP=7，且 round 目录里已备好 bline/ transitions.pt
  #            bline_transitions.pt wm_mix.pt wm_pre.pt self_targets.pt gate_wm_*.json
  _SKIP_TO=${RESUME_FROM_STEP:-1}
  if (( _SKIP_TO <= 4 )); then
  # ---------------- 1) 仿真采集 ----------------
  # REUSE=1：round_N/transitions.pt 已存在（且大于 100 MB）就跳过采集 + WM 门。
  # 只在你**已经采过**、只是后面某步崩了要重跑时用；WM 门结果沿用同目录 gate_wm_qd.json。
  # WBC 冻结 ⇒ 仿真采集（collect_mjlab 只跑 WBC，不含 VLA）每轮等价，第 1 轮真采、之后硬链接复用。
  if [[ "$TRAIN_WBC" == "0" && "$R" -gt "$START_ROUND" && -s "$OUT/round_$START_ROUND/transitions.pt" ]]; then
    ln -f "$OUT/round_$START_ROUND/transitions.pt" "$RD/transitions.pt" 2>/dev/null \
      || cp -f "$OUT/round_$START_ROUND/transitions.pt" "$RD/transitions.pt"
    say "R$R · 1/9 采集：复用 round_$START_ROUND 的 transitions.pt（WBC 冻结 ⇒ 数据等价）"
  elif [[ "${REUSE:-0}" == 1 && -s "$RD/transitions.pt" && $(stat -c%s "$RD/transitions.pt" 2>/dev/null || echo 0) -gt 100000000 ]]; then
    say "R$R · 1/9 采集：REUSE=1 ⇒ 复用 $(du -h "$RD/transitions.pt" | cut -f1) transitions.pt，跳过"
  else
  COLLECT_PID=""
  if [[ -n "$SM_SCENE_TASKS" ]]; then
    say "R$R · 1/9 采集（**带场景**：${SM_SCENE_TASKS} × ${SCENE_ENVS} envs × ${SCENE_STEPS} 步；与 2/9 B线并行）"
    (
    set +e
    cd "$REPO"
    for _T in $SM_SCENE_TASKS; do
      if [[ -s "$RD/transitions_${_T}.pt" && $(stat -c%s "$RD/transitions_${_T}.pt" 2>/dev/null || echo 0) -gt 100000000 ]]; then
        echo "[collect] $_T 已存在（$(du -h "$RD/transitions_${_T}.pt" | cut -f1)）⇒ 跳过"
        continue
      fi
      echo "[collect] $_T"
      run env -u PYTHONPATH SM_G1_ASSET=43dof SM_SCENE_TASK="$_T" \
        CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$VENV" scripts/collect_mjlab.py \
        --ckpt "$W1_CKPT" --library "$LIB" --tokens "$TOK" \
        --num-envs "$SCENE_ENVS" --steps "$SCENE_STEPS" --replay-capacity "$SCENE_STEPS" \
        --action-noise-std 0.1 --out "$RD/transitions_${_T}.pt"
    done
    cd "$COMIMIC"
    run "$PY" scripts/concat_transitions.py --inputs "$RD/transitions_task*.pt" \
      --out "$RD/transitions.pt" || { echo "[fail] 场景采集合并失败"; exit 7; }
    ) &
    COLLECT_PID=$!
    echo "[collect] 后台采集 pid=$COLLECT_PID（与 B 线并行，省 8–14 min）"
  else
    say "R$R · 1/9 采集（envs=$ENVS steps=$STEPS）"
    cd "$REPO"
    run env -u PYTHONPATH CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$VENV" scripts/collect_mjlab.py \
      --ckpt "$W1_CKPT" --library "$LIB" --tokens "$TOK" \
      --num-envs "$ENVS" --steps "$STEPS" --replay-capacity "$CAP" \
      --action-noise-std 0.1 --out "$RD/transitions.pt"
    cd "$COMIMIC"
  fi
  fi

  # ---------------- 2) B 线回放采集（提前：本轮数据要喂本轮的 WM）----------------
  # 任务 × 集数；**恒 arm=vla**
  # 为什么 round1 也用 vla：`trust_center=self` 的中心是"采集时策略真正下发的 token"，
  # 只有 vla 臂才会写进 dump 的 `tokens`（gt 臂写的是录制 token，等于回到 label 口径）。
  say "R$R · 2/9 B 线回放采集（arm=vla · ${BLINE_TASKS} × ${BLINE_EPISODES} 集 × ${BLINE_TICKS} tick）"
  mkdir -p "$RD/bline"
  _WANT=$(( $(echo $BLINE_TASKS | wc -w) * BLINE_EPISODES ))
  # ⚠️ pipefail + `set -e`：目录里没有 dump 时 `ls` 返回 2 ⇒ 整条赋值命令失败 ⇒ 脚本静默退出。
  #    2026-10-29 13:00 就是这么死的（B 线一步没跑、日志无报错）。必须 `|| true` 兜住。
  _HAVE=$(ls "$RD"/bline/b_*.pt 2>/dev/null | wc -l || true)
  if [[ "${REUSE_BLINE:-0}" == 1 && $_HAVE -ge $_WANT ]]; then
    echo "[bline] REUSE_BLINE=1 ⇒ 已有 $_HAVE/$_WANT 条，跳过采集"
  else
  _n=0
  for T in $BLINE_TASKS; do
    if [[ "$BLINE_STARTS_MODE" == "auto" ]]; then
      # --seed "$R"：每轮换一批（分层随机，仍覆盖任务全区间）；想看严格跨轮可比就设 BLINE_SEED=0
      _TASK_STARTS=($("$PY" scripts/bline_starts.py --library "$LIB" --task "$T" \
        --episodes "$BLINE_EPISODES" --ticks "$BLINE_TICKS" --seed "${BLINE_SEED:-$R}"))
    else
      _TASK_STARTS=($BLINE_STARTS)
    fi
    _NS=${#_TASK_STARTS[@]}
    [[ $_NS -gt 0 ]] || { echo "[fail] $T 没有可用起点（BLINE_STARTS_MODE=$BLINE_STARTS_MODE）"; exit 3; }
    echo "[bline] $T 起点(${_NS}): ${_TASK_STARTS[0]} … ${_TASK_STARTS[$((_NS-1))]}"
    for E in $(seq 1 "$BLINE_EPISODES"); do
      F=${_TASK_STARTS[$(( (E - 1) % _NS ))]}
      # **断点续采**（2026-10-30）：文件名是确定性的（task+start_frame+ep），已有就跳过。
      # 动机：02:15 那次链条被外部杀掉，28/80 条已完成却要全部重采（白丢 45 min）。
      if [[ -s "$RD/bline/b_${T}_f${F}_e${E}.pt" ]]; then
        echo "[bline] $T e$E 已存在 ⇒ 跳过"
        continue
      fi
      GPU_FOR=$(( _n % BLINE_JOBS ))
      run "$PY" scripts/closed_loop_vla_supermimic.py \
        --vla-ckpt "$CUR_VLA" --joint-ckpt "$CUR_WBC" --task "$T" \
        --steps "$BLINE_TICKS" --device "cuda:$GPU_FOR" --no-video \
        --replay-lib "$LIB" --start-frame "$F" --arm vla \
        --dump "$RD/bline/b_${T}_f${F}_e${E}.pt" \
        > "$RD/bline/log_${T}_e${E}.log" 2>&1 &
      _n=$((_n+1))
      if (( _n % BLINE_JOBS == 0 )); then wait || true; fi
    done
  done
  # wait 必须 `|| true`：set -e 下任何一个子进程非零都会把整条链带走（2026-10-30 02:15 事故）
  wait || true
  fi
  echo "[bline] $(ls "$RD"/bline/b_*.pt 2>/dev/null | wc -l) 条 → $RD/bline"

  # ---------------- 2.5) 80 集健康门（2026-10-29 新增，必须在这里）----------------
  # 为什么必须：R3..R6 四轮的策略在 80 集闭环上摔倒 2→4/80、平均奖励 0.0859→0.0743，
  # 而 `eval.json` 里的 `fell` 是**单集** eval（同任务同起点一条 600 tick），恰好不摔
  # 就一直是 False ⇒ 我们漏看了四个小时。真正的摔倒记录在每条 dump 的 `meta['fell']`。
  # 判据只用 80 集：摔倒数 + 平均奖励。不过就停，别把坏策略喂给 WM。
  if [[ "${BLINE_HEALTH_GATE:-1}" == "1" ]]; then
    _HG=(--dump "$RD/bline/b_*.pt" --out "$RD/bline_health.json")
    if [[ -n "${BLINE_MIN_REWARD:-}" ]]; then _HG+=(--min-reward "$BLINE_MIN_REWARD"); fi
    if ! run "$PY" scripts/bline_health.py "${_HG[@]}"; then
      echo "[gate] 80 集健康门未过（摔倒或奖励塌）⇒ 停。改奖励/数据，不要继续训。"; exit 8
    fi
  fi
  # trust_center=self 需要"采集时真值"：给每条 dump 打上 (self token, 实际漂移)
  run "$PY" scripts/build_b_self_targets.py --dump "$RD/bline/b_*.pt" \
    --ref "$LIB" --out "$RD/self_targets.pt" --horizon "$J1_H" \
    || echo "[warn] self_targets 构建失败（trust_center=self 会退化成报错，检查 dump）"

  # ---------------- 3) B 线 → transitions 格式，并与仿真采集混合成 WM 训练集 ----------------
  say "R$R · 3/9 混合 WM 训练集（B线占比 ${WM_BLINE_RATIO:-0.25}）"
  run "$PY" scripts/bline_to_transitions.py --dump "$RD/bline/b_*.pt" \
    --library "$LIB" --out "$RD/bline_transitions.pt" \
    || { echo "[fail] B 线转 transitions 失败"; exit 5; }
  run "$PY" scripts/make_wm_mix.py --sim "$RD/transitions.pt" \
    --bline "$RD/bline_transitions.pt" --ratio "${WM_BLINE_RATIO:-0.25}" \
    --out "$RD/wm_mix.pt" || { echo "[fail] 混合 WM 数据集失败"; exit 5; }

  # ---------------- 4) 训练前把 WM 训稳（用当轮 wm_mix，含 25% B 线）----------------
  say "R$R · 4/9 训练前训 WM（${WM_PRETRAIN_STEPS} 步，数据=sim+B线混合）"
  run "$PY" scripts/pretrain_wm.py --joint-ckpt "$CUR_WBC" --transitions "$RD/wm_mix.pt" \
    --steps "$WM_PRETRAIN_STEPS" --out "$RD/wm_pre.pt" \
    || { echo "[fail] WM 预训练失败"; exit 6; }
  CUR_WBC=$RD/wm_pre.pt        # 之后的门/门禁/critic/训练/评测都用这个"训过的 WM"
  echo "[wm-pre] 本轮后续统一使用 $CUR_WBC"

  # ---------------- 5) WM 双分布验收门（sim + B 线部署分布）----------------
  say "R$R · 5/9 WM 验收门（sim 分布 + B线 部署分布；用刚训过的 WM）"
  if ! run "$PY" scripts/probe_wm_qd_corr.py --ckpt "$CUR_WBC" --motion "$MOTION" --tokens "$TOK" \
        --transitions "$RD/transitions.pt" --n 256 --out "$RD/gate_wm_sim.json"; then
    echo "[gate] WM 在 sim 分布未达标 ⇒ 停（不要带着坏 WM 训练）"; exit 2
  fi
  # 部署分布（B 线：VLA 在环 + 任务场景）。第 1 轮必然低（WM 还没在 B 线上训过）⇒ 只记录。
  if ! run "$PY" scripts/probe_wm_qd_corr.py --ckpt "$CUR_WBC" --motion "$MOTION" --tokens "$TOK" \
        --transitions "$RD/bline_transitions.pt" --n 256 --out "$RD/gate_wm_bline.json"; then
    echo "[gate][warn] WM 在部署分布(B线)上未达标（首轮属正常：WM 还没见过接触/物体）"
  fi

  # ---------------- 5) 门禁 ----------------
  say "R$R · 6/9 五道门禁"
  # 一道门一个进程：freeze 之后显存收不回来，同进程里跑 burn-in 必 OOM（4090 实测 45/46 GB）
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  for G in $GATE_LIST; do
    echo "[gate] $G"
    run "$PY" scripts/train_joint_mjlab.py \
      --wbc-ckpt "$CUR_WBC" --motion "$MOTION" --tokens "$TOK" \
      --vlm-ckpt "$CUR_VLA" --vlm-path "$VLM_PATH" \
      --out $RD/gates --gate-only --gate-subset "$G" --gate-batches 4 \
      --lora-rank "$J1_LORA" --amp "$AMP" --device cuda || {
      if [[ "$GATES" == "strict" ]]; then echo "[gate] $G FAIL（strict ⇒ 停）"; exit 3; fi
      echo "[gate][warn] $G 未过/OOM ⇒ 记 'skipped' 并继续（GATES=auto）"
      echo "$G" >> "$RD/gates_skipped.txt"
    }
  done

  else
    CUR_WBC=$RD/wm_pre.pt
    echo "[resume] RESUME_FROM_STEP=$_SKIP_TO ⇒ 跳过 1/9–6/9，直接 7/9；CUR_WBC=$CUR_WBC"
    for _need in wm_pre.pt wm_mix.pt bline_transitions.pt self_targets.pt transitions.pt; do
      [[ -s "$RD/$_need" ]] || { echo "[fail] resume 需要 $RD/$_need"; exit 9; }
    done
    [[ -d "$RD/bline" ]] || { echo "[fail] resume 需要 $RD/bline/"; exit 9; }
  fi

  # ---------------- 5) critic 拟合（决定**下一轮**要不要开） ----------------
  say "R$R · 7/9 critic 拟合（门：留出 v_ev > 0.30；DRY 下默认当它过）"
  if [[ "$CRITIC_SOURCE" == "imagined" && "$CRITIC_MODE" != "off" ]]; then
    # GAE 路线：不需要 prefit，critic 在训练里用想象 rollout 的 GAE 目标更新（方差小）
    CRITIC_ARGS=(--critic --critic-source imagined)
    CRITIC_NOTE="on(GAE)"
    echo "[critic] 走 GAE（imagined 源，无 prefit）"
  else
  _rc=0
  _TGT_ARGS=(--critic-target "$CRITIC_TARGET")
  if [[ "$CRITIC_TARGET" == "gae" ]]; then
    CRITIC_RET_FILE="$RD/critic_targets_gae.pt"
    _TGT_ARGS+=(--targets-out "$CRITIC_RET_FILE")
  fi
  run "$PY" scripts/fit_critic_real.py --transitions "$RD/transitions.pt" \
      --out "$RD/critic.pt" --center linear --epochs 12 --device cuda:0 \
      "${_TGT_ARGS[@]}" || _rc=$?
  if [[ -n "$CRITIC_RET_FILE" && -s "$CRITIC_RET_FILE" ]]; then
    CRITIC_RET_ARGS=(--critic-ret "$CRITIC_RET_FILE")
  fi
  echo "[critic] target=$CRITIC_TARGET${CRITIC_RET_FILE:+ (ret 覆盖文件 $CRITIC_RET_FILE)}"
  if [[ -s "$RD/critic.pt" ]]; then
    CRITIC_ARGS=(--critic --critic-source real --critic-real "$RD/transitions.pt" --critic-ckpt "$RD/critic.pt")
    if [[ $_rc -eq 0 ]]; then CRITIC_NOTE="on(v_ev>0.30)"; else CRITIC_NOTE="on(forced,v_ev<=0.30)"; fi
    if [[ "$CRITIC_MODE" == "off" ]]; then CRITIC_ARGS=(--no-critic); CRITIC_NOTE="off(forced)"; fi
  elif [[ "$CRITIC_MODE" == "on" ]]; then
    echo "[critic] CRITIC_MODE=on 但 fit 没产出 $RD/critic.pt ⇒ 停"; exit 4
  else
    CRITIC_ARGS=(--no-critic); CRITIC_NOTE="off(no ckpt)"
  fi
  fi
  echo "[critic] 本轮 critic=$CRITIC_NOTE（fit 的退出码 0/3 就是门）"

  # ---------------- 6) 联合微调 ----------------
  say "R$R · 8/9 联合微调 ${REWARD_STEPS} 步（token 不做 BC · hand 关 · rl 0.3 · critic=$CRITIC_NOTE · center=self · WBC 解冻）"
  BL=$(ls "$RD"/bline/b_*.pt 2>/dev/null | tr '\n' ' ' || true)
  BSTART=(); [[ -n "$BL" ]] && BSTART=(--start-from-b "$RD/bline/b_*.pt" --b-start-order standard --b-start-nonstrict)
  run "${LAUNCH[@]}" scripts/train_joint_mjlab.py \
    --wbc-ckpt "$CUR_WBC" --motion "$MOTION" --tokens "$TOK" \
    --vlm-ckpt "$CUR_VLA" --vlm-path "$VLM_PATH" \
    --out "$RD/train" --steps "$REWARD_STEPS" --v3 \
    --lambda-bc "$J1_LAMBDA_BC" --lambda-hand 0 --lambda-r 0.3 \
    --lambda-s "$J1_LAMBDA_S" --state-vel-weight "$J1_STATE_VEL" \
    --fm-lr "$J1_FMLR" "${CRITIC_ARGS[@]}" "${CRITIC_RET_ARGS[@]}" \
    --act-eps 0.6 --act-weight 1.0 --act-hinge-cap 2 \
    --vla-state-from-burnin ${WBC_ARGS[@]} \
    --real-transitions "$RD/wm_mix.pt" --start-source reference \
    "${BSTART[@]}" --b-self-targets "$RD/self_targets.pt" \
    ${WM_ARGS[@]} --wm-pretrain-steps 0 \
    --anchor-weight 0 --trust-eps 0.2 --trust-weight 1.0 \
  --trust-center self \
    --batch-size "$J1_BATCH" --grad-accum 1 --horizon "$J1_H" \
    --lora-rank "$J1_LORA" --amp "$AMP" --device cuda \
    --save-every 500 --save-vla-every 500 --log-every 10

  # ---------------- 7) 合并 head + 闭环验收 + benchmark ----------------
  say "R$R · 9/9 闭环验收"
  NEXT_VLA=$OUT/ckpt_vla_r$R
  run "$PY" scripts/export_vla_ckpt.py --base "$CUR_VLA" --vla "$RD/train/vla_final.pt" \
    --out "$NEXT_VLA" --force
  run "$PY" scripts/closed_loop_vla_supermimic.py \
    --vla-ckpt "$NEXT_VLA" --joint-ckpt "$CUR_WBC" --task "$TASK" \
    --steps "$EVAL_TICKS" --device cuda:0 --no-video \
    --replay-lib "$LIB" --start-frame "$FROM_FRAME" --arm vla \
    --dump "$RD/eval_bline.pt" --out-json "$RD/eval.json" || echo "[warn] 闭环验收失败"

  # 数值判据：闭环 dump 里的逐步奖励均值（真物理/同一任务同起点，跨轮可比）
  MET=$(python3 - "$RD/eval_bline.pt" "$RD/eval.json" <<'EOF' 2>/dev/null || echo "nan 0 0 0"
import json,sys,torch
d=torch.load(sys.argv[1], map_location='cpu', weights_only=False)
r=d['rewards'].float()
m=float(r.mean()); n=int(r.numel())
try:
    j=json.load(open(sys.argv[2])); fell=bool(j.get('fell')); tilt=float(j.get('max_tilt_deg',0)); z=float(j.get('final_z',0))
except Exception:
    fell=False; tilt=0.0; z=0.0
print(f"{m:.6f} {int(fell)} {tilt:.1f} {z:.3f}")
EOF
)
  set -- $MET; MR=$1; FELL=$2; TILT=$3; ZZ=$4
  echo "[eval] round $R: mean_reward=$MR fell=$FELL tilt=${TILT}° z=$ZZ"

  # ---------------- 9.5) 固定起点评测集（跨轮同起点可直接比）----------------
  if [[ "${BLINE_EVAL_SET:-1}" == "1" ]]; then
    say "R$R · 9.5/10 固定起点评测集（seed=$BLINE_EVAL_SEED × ${BLINE_EVAL_PER_TASK} 集/任务）"
    BLINE_EPISODES="$BLINE_EVAL_PER_TASK" BLINE_TICKS="$BLINE_TICKS" \
      run bash scripts/collect_bline_arm.sh "$RD/eval_set" "$CUR_WBC" "$NEXT_VLA" \
        "$BLINE_EVAL_SEED" "$BLINE_JOBS" > "$RD/eval_set.log" 2>&1 \
      || echo "[warn] 固定评测集采集失败（看 $RD/eval_set.log）"
    if [[ -s "$RD/eval_set/health.json" ]]; then
      run "$PY" - "$RD/eval_set/health.json" <<'PYEOF' || true
import json,sys
d=json.load(open(sys.argv[1]))
print(f"[eval-set] 集数 {d['n_total']} | 摔倒 {d['fell_total']} | 最大倾角 {d['max_tilt_max']:.1f}° "
      f"| 平均奖励 {d['reward_mean']:.4f}±{d['reward_se']:.4f}")
PYEOF
    fi
  fi
  printf '%s,%s,%s,%s,%s\n' "$R" "$MR" "$FELL" "$TILT" "$ZZ" >> "$OUT/rounds.csv"

  # 本轮训练后的 WM 在**部署分布**上的表现（与 4/9 的门对比 = 混合训练是否有效的直接证据）
  if [[ -s "$RD/train/joint_final.pt" ]]; then
    run "$PY" scripts/probe_wm_qd_corr.py --ckpt "$RD/train/joint_final.pt" \
      --motion "$MOTION" --tokens "$TOK" --transitions "$RD/bline_transitions.pt" \
      --n 256 --out "$RD/gate_wm_bline_after.json" \
      && echo "[wm-after] 见 $RD/gate_wm_bline_after.json（对比 4/9 的 gate_wm_bline.json）"
  fi

  # 判据：mean reward 相对最佳提升 > MIN_GAIN 算"涨"；否则记一次，连续 PATIENCE 次就停
  GAIN=$(python3 - "$MR" "${BEST_R:-nan}" "$MIN_GAIN" <<'EOF'
import sys, math
m=float(sys.argv[1]); b=sys.argv[2]
try: b=float(b)
except Exception: print("first"); raise SystemExit
if not (math.isfinite(m) and math.isfinite(b)): print("nan"); raise SystemExit
print("up" if (m-b) > abs(b)*float(sys.argv[3]) else "flat")
EOF
)
  echo "[loop] round $R 判定: $GAIN (best=${BEST_R:-—} → now=$MR)"
  if [[ "$GAIN" == "up" || "$GAIN" == "first" ]]; then
    NO_GAIN=0; BEST_R=$MR; BEST_ROUND=$R
  elif [[ "$GAIN" == "nan" ]]; then
    echo "[loop][warn] 本轮闭环没有有效数值（评估失败），不计入早停"   # 瞬时失败不该终止多轮
  else
    NO_GAIN=$((NO_GAIN+1))
    echo "[loop] 无改善 $NO_GAIN/$PATIENCE"
  fi
  if [[ $NO_GAIN -ge $PATIENCE ]]; then
    echo "[loop] 连续 $PATIENCE 轮无改善 ⇒ 停（第 $R 轮；最佳第 ${BEST_ROUND:-?} 轮，mean=$BEST_R）"
    break
  fi
  CUR_VLA=$NEXT_VLA
  # 本轮训练出的 policy+wm 接给下一轮（WM 保鲜成果累积、与 VLA head 同步演进）
  if [[ -s "$RD/train/joint_final.pt" && $R -lt $((START_ROUND + ROUNDS - 1)) ]]; then
    CUR_WBC=$RD/train/joint_final.pt
    echo "[loop] 下一轮 WBC/WM = $CUR_WBC（累积）"
  fi
done

say "完成"
cat <<EOF
轮次产物：$OUT/round_*/
每轮：transitions.pt / gate_wm_qd.json / bline/*.pt / gates/gates.json / critic.pt / train/{joint,vla}_final.pt
合并后的 VLA：$OUT/ckpt_vla_r*/  （最后一轮 = 可直接部署的 ckpt）
EOF
