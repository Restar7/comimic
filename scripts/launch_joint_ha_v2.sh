#!/usr/bin/env bash
# 联合微调 v2（S3 阶段 C）启动器。
#
# 用法：
#   LAMBDA_R=0    RUN=s3_lamr000 bash scripts/launch_joint_ha_v2.sh   # 对照臂（只 BC）
#   LAMBDA_R=0.3  RUN=s3_lamr030 bash scripts/launch_joint_ha_v2.sh
#   STEPS=3 LAMBDA_R=0.3 RUN=s3_smoke bash scripts/launch_joint_ha_v2.sh   # smoke
#
#   # 第 1 轮闭环（评审 §2/§3）：起点 = B 线重规划状态，中心 = 上一轮 π_k，无标签
#   TRUST_CENTER=prev START_FROM_B='runs/B_r1_*/dump/*.pt' ACT_EPS=0.6 ACT_WEIGHT=1.0 \
#   LAMBDA_R=0.3 LAMBDA_BC=0 EXTRA="--amp bf16 --lora-rank 32 --fm-lr 1e-4 --no-dr \
#     --vla-state-from-burnin --no-critic --act-hinge-cap 2 --lambda-aref 0" \
#   RUN=s3_r1 bash scripts/launch_joint_ha_v2.sh
#
# 关键环境变量：
#   WBC_CKPT  WBC（默认 A24000 = 带手 plant + ha_cmd 域重标的 C6h′；locked 口径）
#   VLM_CKPT  起始 VLA（默认官方 checkpoint-8000）
#   BATCH / ACCUM / H / K   batch-size / grad-accum / 想象步数 / FM 步数
#   LAMBDA_R  奖励项权重（扫 0 / 0.1 / 0.3 / 1.0）
#   TRUST_CENTER  label / base / prev（prev = 本轮起点 π_k，推荐）
#   START_FROM_B  B 线 dump 的 glob；给了它就不做 burn-in、不叠 DR
#   PORT_DIR  仅当 WBC/BC 用端口空间时才需要（默认空 ⇒ 不传 --port-tokens）
set -euo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$COMIMIC/trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd}

# locked 口径：WBC = A24000（带手 plant 续训 + ha_cmd 域重标），不是旧的 wbc_mix2。
WBC_CKPT=${WBC_CKPT:-$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_cmd.pt}
VLM_CKPT=${VLM_CKPT:-$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}
VLM_PATH=${VLM_PATH:-$COMIMIC/groot_repro/nvidia/Cosmos-Reason2-2B-weights}
MOTION=${MOTION:-$COMIMIC/data/npz/ha_tokens}
# 端口侧车只在 port token 空间才用；默认空，避免"看起来在用 port tokens"的误读。
PORT_DIR=${PORT_DIR:-}

RUN=${RUN:-joint_v2_lamr030}
OUT=${OUT:-$COMIMIC/runs/$RUN}
STEPS=${STEPS:-2000}
LAMBDA_R=${LAMBDA_R:-0.3}
LAMBDA_BC=${LAMBDA_BC:-1.0}
LAMBDA_S=${LAMBDA_S:-0}
LAMBDA_A=${LAMBDA_A:-0}
BATCH=${BATCH:-4}
# ★ 2026-10-31：DDP（数据并行）。`DDP=1` 时用 torchrun 起 NPROC 个进程，
#   **每卡 batch = BATCH / NPROC**，于是全局 batch 与单卡完全一致（语义等价），
#   而每步墙钟时间 ≈ 1/NPROC。梯度靠 `--sync-gradients`（all_reduce）合并。
DDP=${DDP:-0}
NPROC=${NPROC:-2}
MASTER_PORT=${MASTER_PORT:-29511}
if [[ "$DDP" == "1" ]]; then
  _per=$(( BATCH / NPROC )); (( _per < 1 )) && _per=1
  echo "[launch-v2][ddp] nproc=$NPROC 每卡batch=$_per（全局 $((_per*NPROC))）port=$MASTER_PORT"
  BATCH=$_per
fi
_SYNC_ARG=()
[[ "$DDP" == "1" ]] && _SYNC_ARG=(--sync-gradients)
ACCUM=${ACCUM:-4}
H=${H:-20}
K=${K:-4}
BURNIN=${BURNIN:-10}
EVAL_EVERY=${EVAL_EVERY:-250}
EVAL_BATCHES=${EVAL_BATCHES:-16}
SAVE_EVERY=${SAVE_EVERY:-500}
GPU=${GPU:-0}
BACKEND=${BACKEND:-pathwise}
V3=${V3:-0}
TRAIN_WBC=${TRAIN_WBC:-0}
# critic 默认关（trainer 的 --critic 默认是开，必须显式 --no-critic，否则会带随机 critic）
CRITIC_ON=${CRITIC_ON:-0}
CRITIC_SOURCE=${CRITIC_SOURCE:-real}
CRITIC_REAL=${CRITIC_REAL:-$COMIMIC/runs/wm_real/real_transitions_v2b.pt}
CRITIC_CKPT=${CRITIC_CKPT:-}
CRITIC_USE_DIFF=${CRITIC_USE_DIFF:-0}
WBC_LR=${WBC_LR:-1e-5}
# ★ 2026-11-01（Path B）：WBC 解冻范围 + 双 KL 锚；默认全关，Path A 行为不变。
WBC_UNFREEZE=${WBC_UNFREEZE:-all}
WBC_KL_ANCHOR=${WBC_KL_ANCHOR:-0}
VLA_KL_ANCHOR=${VLA_KL_ANCHOR:-0}
TRUST_CENTER=${TRUST_CENTER:-self}
START_FROM_B=${START_FROM_B:-}
MOTION_PACKED=${MOTION_PACKED:-}
HUBER_DELTA=${HUBER_DELTA:-0.02}
HUBER_CAP=${HUBER_CAP:-0.15}
GAMMA=${GAMMA:-0.99}
START_SAMPLER=${START_SAMPLER:-uniform}
START_WEIGHTS=${START_WEIGHTS:-}
START_AUDIT=${START_AUDIT:-0}
ALLOW_LEGACY_REWARD=${ALLOW_LEGACY_REWARD:-0}
REWARD_BODY_POS_STD=${REWARD_BODY_POS_STD:-}
REWARD_BODY_ORI_STD=${REWARD_BODY_ORI_STD:-}
REWARD_DELTA_POS=${REWARD_DELTA_POS:-0.05}
REWARD_DELTA_VEL=${REWARD_DELTA_VEL:-0.05}  # huber_err 速度项 δ
REWARD_DELTA_ORI=${REWARD_DELTA_ORI:-0.05}
WM_CKPT=${WM_CKPT:-}
# 多轮闭环：上一轮的 vla_*.pt（= 本轮 π_k 起点）。空 = 官方底模（第 1 轮）。
INIT_VLA=${INIT_VLA:-}
# 2026-10-28：v3 的 token BC 权重（用 B dump 的 token 当标签）。0 = 旧行为。
TOKEN_BC=${TOKEN_BC:-0}
# 2026-10-29：用采集时的真实结果做监督
B_SELF_TARGETS=${B_SELF_TARGETS:-}
B_SELF_ERR_MAX=${B_SELF_ERR_MAX:-}
B_START_MAX_ERR=${B_START_MAX_ERR:-}   # 瞬时误差筛选（奖励核有梯度的区间）
TOKEN_BC_SOURCE=${TOKEN_BC_SOURCE:-ref}
TOKEN_BC_SPAN=${TOKEN_BC_SPAN:-0}
# 2026-11-04：token 空间的 FSQ 网格量化（见 diff_vla_wbc/quant.py）。
#   none = 连续喂（旧行为；执行器可重训 + dither 时用）
#   ste  = 前向喂格点（执行器训练分布内）+ 反向恒等传（推荐；**部署必须同步量化**）
#   hard = 硬量化、梯度归零（消融对照：应当"训不动"，若仍降 loss 说明梯度从 trust/BC 漏进来）
TOKEN_QUANT=${TOKEN_QUANT:-none}
# 2026-10-29：这三项原来是**硬编码**在 driver 的 EXTRA 里（lambda-hand 恒 0.2），
# 导致"说改了其实没改"。现在改成环境变量，driver 默认传 0 / 1e-4 / 0.1。
LAMBDA_HAND=${LAMBDA_HAND:-0}      # 部署 SM_HAND=0，手部训练用不到；实测占 78% 梯度
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}  # 小数据（~200-400 起点）抗过拟合
OBS_NOISE=${OBS_NOISE:-0}          # WM 当裁判时必须 0（否则回报噪声）
REWARD_MODE=${REWARD_MODE:-saturating}   # 用 WM 当裁判时必须 linear（否则大误差处梯度=0）
POSTURE_WEIGHT=${POSTURE_WEIGHT:-0.0}    # 摔倒/歪斜的连续惩罚
REWARD_VEL_SCALE=${REWARD_VEL_SCALE:-1.0}
TERM_OFF=${TERM_OFF:-0}   # 1 = 关掉存活折扣（新 WM 的 soft_termination 会误触发） # WM 速度通道最不准（qd_corr≈0.64），可设 0
# ★ 2026-11-06：改成**显式数组**，不再用 `$( [[ ... ]] && echo ... )` 命令替换（实测会被吞，
#   导致 term_off 没传进训练器、survival<1、回报被砍半）。我们所有轮次都想要 --term-off。
VAR_WEIGHT=${VAR_WEIGHT:-0}
TERM_OFF_ARG=""
[[ "$TERM_OFF" == "1" ]] && TERM_OFF_ARG="--term-off"
# ★ 2026-11-06：**命令层残差**（DELTA_TOKEN=1）—— 与 LoRA/state_extra/var 并行的第四条路。
#   默认 0 ⇒ 不传任何参数，现有行为逐位不变。
DELTA_TOKEN=${DELTA_TOKEN:-0}
DELTA_TOKEN_HIDDEN=${DELTA_TOKEN_HIDDEN:-1024}  # 99→1024→1024→64 = 1,217,600 参数
DELTA_TOKEN_INIT_STD=${DELTA_TOKEN_INIT_STD:-4e-4}
DELTA_TOKEN_LR=${DELTA_TOKEN_LR:-5e-4}   # ★ 独立小 MLP 的 lr，别用 fm_lr(1e-5)
DELTA_TOKEN_DEPTH=${DELTA_TOKEN_DEPTH:-2}
DELTA_TOKEN_CAP=${DELTA_TOKEN_CAP:-0}
FREEZE_VLA=${FREEZE_VLA:-0}
_DT_ARG=()
DELTA_TOKEN_FULLSTATE=${DELTA_TOKEN_FULLSTATE:-0}
GRAD_TARGET_WEIGHT=${GRAD_TARGET_WEIGHT:-0}
GRAD_TARGET_N=${GRAD_TARGET_N:-16}
GRAD_TARGET_ETA=${GRAD_TARGET_ETA:-0.08}
GRAD_TARGET_HORIZON=${GRAD_TARGET_HORIZON:-0}
WINDOW_NORM=${WINDOW_NORM:-0}
[[ "$DELTA_TOKEN" == "1" ]] && _DT_ARG=(--delta-token
    --delta-token-hidden "$DELTA_TOKEN_HIDDEN" --delta-token-depth "$DELTA_TOKEN_DEPTH"
    --delta-token-init-std "$DELTA_TOKEN_INIT_STD" --delta-token-lr "$DELTA_TOKEN_LR"
    --delta-token-cap "$DELTA_TOKEN_CAP"
    --grad-target-weight "$GRAD_TARGET_WEIGHT" --grad-target-n "$GRAD_TARGET_N"
    --grad-target-eta "$GRAD_TARGET_ETA" --grad-target-horizon "$GRAD_TARGET_HORIZON" \
    $([[ "$WINDOW_NORM" == "1" ]] && printf '%s' '--window-norm')
    $([[ "$DELTA_TOKEN_FULLSTATE" == "1" ]] && printf '%s' '--delta-token-fullstate'))
[[ "$FREEZE_VLA" == "1" ]] && _DT_ARG+=(--freeze-vla)
REWARD_ROOT_WEIGHT=${REWARD_ROOT_WEIGHT:-}   # root 奖励权重覆盖（我们 ckpt 0.5 / mjlab 1.0）
REWARD_ROOT_POS_STD=${REWARD_ROOT_POS_STD:-} # root_pos_std 覆盖（我们 0.3 / mjlab 0.2）
REWARD_ROOT_ORI_STD=${REWARD_ROOT_ORI_STD:-} # root_ori_std 覆盖（我们 0.4 / mjlab 0.25）
ROOT_ORI_MODE=${ROOT_ORI_MODE:-split}       # mono=单块 root ori；split=tilt/yaw/yaw_rate 拆项
TAIL_CAP_SLOPE=${TAIL_CAP_SLOPE:-0.0}     # e2：越过 hinge cap 后保留的斜率
TAIL_POS_THR=${TAIL_POS_THR:-}            # 根位置 hinge e_h（空=reward_shape.ROOT_POS_THR）
TAIL_POS_CAP=${TAIL_POS_CAP:-}            # 根位置 hinge cap（空=reward_shape.ROOT_POS_CAP）
STATE_EXTRA=${STATE_EXTRA:-none}          # e1：none|abs6|vel6|ori6
# 2026-11-04：**默认仍是 none**（plan / PLAN_REFSTART 的口径）。
#   新增的 `label` 是可选臂（配对基线，不需要 B dump）；启用后 ADV_TAU 才有意义。
# ★★ 2026-11-04 **微调起点口径**（必须与 leash 采集成对，否则起点退化为"纯参考 + 假 DR"）：
#   recorded = 用 A' 那一帧的**真实状态**（物理自洽、覆盖 ξ∈[0,δ]）；reference = 旧行为。
START_SOURCE=${START_SOURCE:-recorded}
START_MAX_DRIFT=${START_MAX_DRIFT:-0.10}     # 起点 |ξ| 上限(m)；0=不封顶
# 奖励死区：d>d_max 之后不硬拉回参考（state46 里没有 root 位置，大漂移看不见）
REWARD_D_MAX=${REWARD_D_MAX:-0.30}
ADV_BASELINE=${ADV_BASELINE:-none}        # e2：none|self|label
ADV_TAU=${ADV_TAU:-0.02}                  # e2：advantage 有界温度
YAW_RATE_WEIGHT=${YAW_RATE_WEIGHT:-}       # yaw_rate 奖励权重（空=用常量）
YAW_HINGE_LAMBDA=${YAW_HINGE_LAMBDA:-0.3} # AB6：yaw 远场 hinge 的 λ（0=关）
TAIL_WEIGHT=${TAIL_WEIGHT:-2.0}           # hinge 尾部权重（0=关；mjlab 用 2.0）
B_FRESH_GLOB=${B_FRESH_GLOB:-}            # 新一轮 B 线 glob（加权采样用）
B_FRESH_WEIGHT=${B_FRESH_WEIGHT:-3.0}
START_FROM_B_STRIDE=${START_FROM_B_STRIDE:-20}
# 2026-11-04：0.6 -> 0.30（回到标定值：base policy 自身偏差 0.21 的 ~1.4x）。
#   见 run_refstart_joint.sh 的同类说明 / docs/JOINT_MULTIROUND_PLAN_20261103.md §14.2。
# 2026-11-04（用户口径）：0.30 -> 0.10。起点永远从参考出发，20 tick 内偏差只有几 cm，
#   信任域应远小于 base policy 自身偏差（0.21，其中大半还是 FM 采样噪声）。
#   0.10 ≈ A'(σ=0.05) 的 2 倍；预期 frac_out≈1.0（它本来就该一直咬住）。
#   观察点：act/rms 若长期贴在 0.10（被钉住）= 奖励没空间，调 ACT_WEIGHT 而不是 eps。
ACT_EPS=${ACT_EPS:-0.10}
HYBRID_ERR=${HYBRID_ERR:-0.5}
# 2026-11-04（用户口径）：默认 0 -> 1.0。信任域收紧后它才真正起作用，从 1.0 起调。
ACT_WEIGHT=${ACT_WEIGHT:-1.0}
# ★ 2026-11-05：惩罚形状。`hinge` 在 eps 以内**梯度恰为 0** ⇒ 奖励可以零阻力把动作推出去
#   （round-5 实测滑到 act/rmse=0.80 ≈ 每关节 20°，而 eps 是 0.30）。
#   `l2` = (r/eps)²：从第一步就参与平衡；此时 **eps 只是尺度，真正的旋钮是 ACT_WEIGHT**。
#   粗算平衡点 r* ≈ 1.1·eps²/weight（见下面 banner 的提示）。
ACT_PENALTY=${ACT_PENALTY:-hinge}
# ★ 2026-11-05：奖励只取想象的前 k 步（0=满 H）。WM 单步准、多步靠积分。
REWARD_HORIZON=${REWARD_HORIZON:-0}
# ★ 2026-11-05（P1）：把更新搬到动作空间（WM 梯度 → 目标动作 → 回归）
ACTION_TARGET_WEIGHT=${ACTION_TARGET_WEIGHT:-0}
ACTION_TARGET_ETA=${ACTION_TARGET_ETA:-0.15}
ACTION_TARGET_STEPS=${ACTION_TARGET_STEPS:-20}
EXTRA=${EXTRA:-}

for p in "$WBC_CKPT" "$VLM_CKPT" "$VLM_PATH" "$MOTION"; do
  [[ -e "$p" ]] || { echo "[fail] missing $p"; exit 1; }
done
if [[ -n "$PORT_DIR" ]]; then
  [[ -e "$PORT_DIR" ]] || { echo "[fail] missing $PORT_DIR"; exit 1; }
fi
if [[ "$TRUST_CENTER" == "prev" && "$EXTRA" != *"--lora-rank"* ]]; then
  echo "[fail] TRUST_CENTER=prev 需要 LoRA（EXTRA 里加 --lora-rank 32）"
  exit 1
fi
if [[ -n "$START_FROM_B" ]] && ! compgen -G "$START_FROM_B" > /dev/null; then
  echo "[fail] START_FROM_B 没匹配到文件: $START_FROM_B"
  exit 1
fi
case "$MOTION" in
  *mixed*|*ft318*) echo "[fail] 只允许纯 HA 数据，拒绝: $MOTION"; exit 1 ;;
esac

mkdir -p "$OUT" "$COMIMIC/logs"
LOG="$COMIMIC/logs/$RUN.log"
echo "[launch-v2] out=$OUT lambda_r=$LAMBDA_R H=$H K=$K batch=$BATCH accum=$ACCUM steps=$STEPS term_off=$TERM_OFF"
echo "[launch-v2] wbc=$WBC_CKPT"
echo "[launch-v2] vlm=$VLM_CKPT"
if [[ "$V3" != "1" && ( "$TRUST_CENTER" != "label" || "$ACT_EPS" != "0" || -n "$START_FROM_B" ) ]]; then
  echo "[launch-v2] WARN: V3=0 ⇒ --trust-center/--act-*/--v3 相关项**不生效**；这不是完整 v3 训练"
fi
if [[ "$TRAIN_WBC" == "1" && "$ACT_EPS" == "0" ]]; then
  echo "[launch-v2] WARN: TRAIN_WBC=1 但 ACT_EPS=0 ⇒ 执行器没有信任域，等于在 WM 里裸训 WBC"
fi
echo "[launch-v2] v3=$V3 train_wbc=$TRAIN_WBC wbc_lr=$WBC_LR wbc_unfreeze=$WBC_UNFREEZE wbc_kl=$WBC_KL_ANCHOR vla_kl=$VLA_KL_ANCHOR trust_center=$TRUST_CENTER act_eps=$ACT_EPS act_weight=$ACT_WEIGHT act_shape=$ACT_PENALTY reward_horizon=$REWARD_HORIZON"
echo "[launch-v2] critic=$CRITIC_ON source=$CRITIC_SOURCE ckpt=${CRITIC_CKPT:-<none>} use_diff=$CRITIC_USE_DIFF"
echo "[launch-v2] start_from_b=${START_FROM_B:-<none>} (stride=$START_FROM_B_STRIDE)"
echo "[launch-v2] init_vla=${INIT_VLA:-<base 底模>}"
echo "[launch-v2] lambda_hand=$LAMBDA_HAND wd=$WEIGHT_DECAY obs_noise=$OBS_NOISE token_bc=$TOKEN_BC (source=$TOKEN_BC_SOURCE) self_targets=${B_SELF_TARGETS:-<none>} err_max=${B_SELF_ERR_MAX:-<none>}"
echo "[launch-v2] port_dir=${PORT_DIR:-<unused, bc=label>}"
echo "[launch-v2] token_quant=$TOKEN_QUANT (none=连续 / ste=格点前向+恒等反向 / hard=硬量化消融)"

# P0-1（2026-10-28 换机）：必须先探测，不能盲目把 third_party 的 cp311 副本放前面，
# 否则 py3.12 下 albumentations -> skimage.measure.label 会因为 _ccomp 加载失败而崩。
source "$COMIMIC/scripts/deps_env.sh"

_PORT_ARG=()
[[ -n "$PORT_DIR" ]] && _PORT_ARG=(--port-tokens "$PORT_DIR")
_B_ARG=()
[[ -n "$START_FROM_B" ]] && _B_ARG=(--start-from-b "$START_FROM_B" --start-from-b-stride "$START_FROM_B_STRIDE")
_MP_ARG=(); [[ -n "$MOTION_PACKED" ]] && _MP_ARG=(--motion-packed "$MOTION_PACKED")
_SS_ARG=(); [[ -n "$START_SAMPLER" ]] && _SS_ARG=(--start-sampler "$START_SAMPLER")
_SW_ARG=(); [[ -n "$START_WEIGHTS" ]] && _SW_ARG=(--start-weights "$START_WEIGHTS")
_SA_ARG=(); [[ "$START_AUDIT" == "1" ]] && _SA_ARG=(--start-audit)
_AL_ARG=(); [[ "$ALLOW_LEGACY_REWARD" == "1" ]] && _AL_ARG=(--allow-legacy-reward)
_WM_ARG=(); [[ -n "$WM_CKPT" ]] && _WM_ARG=(--wm-ckpt "$WM_CKPT")
_INIT_ARG=()
[[ -n "$INIT_VLA" ]] && _INIT_ARG=(--init-vla "$INIT_VLA")
_V3_ARG=()
[[ "$V3" == "1" ]] && _V3_ARG=(--v3)
_WBC_ARG=()
[[ "$TRAIN_WBC" == "1" ]] && _WBC_ARG=(--train-wbc --wbc-lr "$WBC_LR"
  --wbc-unfreeze "$WBC_UNFREEZE" --wbc-kl-anchor "$WBC_KL_ANCHOR" --vla-kl-anchor "$VLA_KL_ANCHOR")
_CRITIC_ARG=(--no-critic)
if [[ "$CRITIC_ON" == "1" ]]; then
  _CRITIC_ARG=(--critic --critic-source "$CRITIC_SOURCE")
  [[ -n "$CRITIC_REAL" ]] && _CRITIC_ARG+=(--critic-real "$CRITIC_REAL")
  [[ -n "$CRITIC_CKPT" ]] && _CRITIC_ARG+=(--critic-ckpt "$CRITIC_CKPT")
  [[ "$CRITIC_USE_DIFF" == "1" ]] && _CRITIC_ARG+=(--critic-use-diff)
fi

# 2026-10-28 P0-2：**不要**无条件 `| tee`。tee 被 SIGPIPE 弄死时训练会跟着断
# （runs/s3_r1 就断在 step 58，log 冻在 step 50）。TEE=1 只在交互式观察时需要；
# 后台跑（driver/setsid）用 TEE=0 直接落文件，tee 死不了训练。
_TEE=${TEE:-1}
_ARGS=(
  "$COMIMIC/scripts/train_joint_v2.py"
  --motion "$MOTION" "${_PORT_ARG[@]}" "${_MP_ARG[@]}" "${_SS_ARG[@]}" "${_SW_ARG[@]}" "${_SA_ARG[@]}"
  --wbc-ckpt "$WBC_CKPT" --vlm-ckpt "$VLM_CKPT" --vlm-path "$VLM_PATH" "${_WM_ARG[@]}"
  --device "cuda:$GPU" --out "$OUT"
  --steps "$STEPS" --batch-size "$BATCH" --grad-accum "$ACCUM" \
  "${_SYNC_ARG[@]}"
  --horizon "$H" --fm-steps "$K" --burnin-steps "$BURNIN"
  --lambda-r "$LAMBDA_R" --lambda-bc "$LAMBDA_BC"
  --lambda-s "$LAMBDA_S" --lambda-a "$LAMBDA_A"
  --bc-token-space label --trust-center "$TRUST_CENTER" \
  --trust-eps "${TRUST_EPS:-0.0}" --trust-weight "${TRUST_WEIGHT:-0.0}" \
  --hybrid-err "$HYBRID_ERR" $_V3_ARG "${_WBC_ARG[@]}" "${_CRITIC_ARG[@]}"
  --act-eps "$ACT_EPS" --act-weight "$ACT_WEIGHT" --act-penalty "$ACT_PENALTY" --reward-horizon "$REWARD_HORIZON" \
  --action-target-weight "$ACTION_TARGET_WEIGHT" --action-target-eta "$ACTION_TARGET_ETA" --action-target-steps "$ACTION_TARGET_STEPS"
  "${_B_ARG[@]}" "${_INIT_ARG[@]}"
  --fm-grad-backend "$BACKEND"
  --token-quant "$TOKEN_QUANT"
  --eval-every "$EVAL_EVERY" --eval-batches "$EVAL_BATCHES"
  --eval-out "$OUT/eval_curve.jsonl"
  --save-every "$SAVE_EVERY" --save-vla-every "$SAVE_EVERY"
  --log-every "${LOG_EVERY:-10}" --token-bc-weight "$TOKEN_BC" \
  --lambda-hand "$LAMBDA_HAND" --weight-decay "$WEIGHT_DECAY" --obs-noise "$OBS_NOISE" \
  --reward-mode "$REWARD_MODE" --huber-delta "$HUBER_DELTA" --huber-cap "$HUBER_CAP" \
  --reward-delta-pos "$REWARD_DELTA_POS" --reward-delta-ori "$REWARD_DELTA_ORI" \
  --reward-delta-vel "$REWARD_DELTA_VEL" \
  ${TERM_OFF_ARG:-} ${VAR_WEIGHT:+--var-weight "$VAR_WEIGHT"} \
  --posture-weight "$POSTURE_WEIGHT" --reward-vel-scale "$REWARD_VEL_SCALE" \
  ${REWARD_ROOT_WEIGHT:+--reward-root-weight "$REWARD_ROOT_WEIGHT"} \
  ${REWARD_ROOT_POS_STD:+--reward-root-pos-std "$REWARD_ROOT_POS_STD"} \
  ${REWARD_ROOT_ORI_STD:+--reward-root-ori-std "$REWARD_ROOT_ORI_STD"} \
  ${REWARD_BODY_POS_STD:+--reward-body-pos-std "$REWARD_BODY_POS_STD"} \
  ${REWARD_BODY_ORI_STD:+--reward-body-ori-std "$REWARD_BODY_ORI_STD"} \
  "${_AL_ARG[@]}" \
  --root-ori-mode "$ROOT_ORI_MODE" --tail-weight "$TAIL_WEIGHT" \
  ${TAIL_POS_THR:+--tail-pos-thr "$TAIL_POS_THR"} \
  ${TAIL_POS_CAP:+--tail-pos-cap "$TAIL_POS_CAP"} \
  --tail-cap-slope "$TAIL_CAP_SLOPE" --state-extra "$STATE_EXTRA" \
  --adv-baseline "$ADV_BASELINE" --adv-tau "$ADV_TAU" \
  "${_DT_ARG[@]}" \
  --executor "${EXECUTOR:-sonic}" ${SONIC_DECODER:+--sonic-decoder "$SONIC_DECODER"} \
  --start-source "$START_SOURCE" --start-max-drift "$START_MAX_DRIFT" \
  --reward-d-max "$REWARD_D_MAX" \
  ${YAW_RATE_WEIGHT:+--yawrate-weight "$YAW_RATE_WEIGHT"} \
  --yaw-hinge-lambda "$YAW_HINGE_LAMBDA" \
  ${B_FRESH_GLOB:+--b-fresh-glob "$B_FRESH_GLOB"} --b-fresh-weight "$B_FRESH_WEIGHT" \
  --token-bc-source "$TOKEN_BC_SOURCE" --token-bc-span "$TOKEN_BC_SPAN" ${B_SELF_TARGETS:+--b-self-targets "$B_SELF_TARGETS"} \
  ${B_SELF_ERR_MAX:+--b-self-err-max "$B_SELF_ERR_MAX"} ${B_START_MAX_ERR:+--b-start-max-err "$B_START_MAX_ERR"} $EXTRA
)
# ---- 铁律 5：启动断言（不许落到"默认值坑"）--------------------------------
_py=python3
"$_py" - "$OUT" "$WM_CKPT" "$START_SOURCE" "$REWARD_MODE" "$ACT_EPS" "$ACT_WEIGHT" "$ADV_BASELINE" "$ACT_PENALTY" <<'PYCHK' || exit 9
import sys
from pathlib import Path
out, wm, ss, rm, ae, aw, ab, _shape_in = (list(sys.argv[1:9]) + ["hinge"] * 8)[:8]
bad = []
if wm and not Path(wm).exists():
    bad.append(f"WM_CKPT 不存在: {wm}")
_shape = str(_shape_in or "hinge").lower()
if _shape == "l2":
    # L2 里 eps 只是尺度，**有效刹车强度 = weight / eps²**；平衡点 r* ≈ P·eps²/(2·weight)，
    # 其中 P 是奖励拉力（随奖励视界、λ_r 变），**启动时无法预知** ⇒ 这里只做健全性检查，
    # 真正的判据是跑起来之后观测到的 act/rmse（见 scripts/check_ft_run.py）。
    # 2026-11-05 实测标定：weight=1.0 → r*≈0.013（钉死）；weight=0.1 → 目标 ≈0.13。
    if float(aw) <= 0.0:
        bad.append(f"ACT_PENALTY=l2 但 ACT_WEIGHT={aw} ⇒ 没有刹车")
    if float(ae) <= 0.0:
        bad.append(f"ACT_PENALTY=l2 但 ACT_EPS={ae} ⇒ 没有尺度")
else:
    if float(ae) < 0.20:
        bad.append(f"ACT_EPS={ae} 太小（H=20 的轨迹差量级约 0.4~0.6；过小会让 hinge 全部出界）")
if not wm:
    bad.append("WM_CKPT 为空 ⇒ 会退回 WBC 自带的旧 WM（这不是我们验收过的那份）")
print(f"[launch-v2][assert] start_source={ss} reward={rm} act=({ae},{aw},{_shape}) adv_baseline={ab} wm={wm}", flush=True)
if bad:
    print("[launch-v2][assert] ❌ " + "; ".join(bad), file=sys.stderr)
    raise SystemExit(1)
PYCHK
echo "[launch-v2] act 惩罚：shape=$ACT_PENALTY eps=$ACT_EPS w=$ACT_WEIGHT"
echo "[launch-v2] 标定（2026-11-05 实测，不是公式）：同一套奖励（只算前 5 步）下，"
echo "[launch-v2]   l2 weight=1.0 ⇒ 实测 act/rmse≈0.013（被钉死，想象回报完全不涨）"
echo "[launch-v2]   l2 weight=0.1 ⇒ 目标 0.10~0.15（≈2.5~3.8°/关节）"
echo "[launch-v2]   判据看跑起来之后的 act/rmse：<0.02 钉死 / >0.5 出流形 / 0.05~0.15 才是目标区"
echo "[launch-v2] tee=$_TEE log=$LOG"
_LAUNCH=(python3)
if [[ "$DDP" == "1" ]]; then
  # ★ 2026-11-01（8 卡并行两臂）：torchrun 的 LOCAL_RANK 永远从 0 开始，
  #   不设 CUDA_VISIBLE_DEVICES 时两条 4 卡 DDP 都会落在物理 GPU 0-3 上。
  #   DDP_GPUS="0,1,2,3" / "4,5,6,7" 把每条臂钉到自己那四张卡。
  if [[ -n "${DDP_GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$DDP_GPUS"
    echo "[launch-v2][ddp] CUDA_VISIBLE_DEVICES=$DDP_GPUS"
  fi
  _LAUNCH=(torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" --no_python python3)
fi
# ★ 2026-11-06：DRY_RUN=1 ⇒ 只打印最终命令行然后退出（不占 GPU、不建 run 目录副作用）。
#   用来核对"命令行选哪条路"是否真的传到了训练器：
#     DRY_RUN=1 DELTA_TOKEN=1 FREEZE_VLA=1 bash scripts/run_p1_round_b.sh
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[launch-v2][dry-run] 最终命令："
  printf '  %q' "${_LAUNCH[@]}" "${_ARGS[@]}"; echo
  echo "[launch-v2][dry-run] 线路开关："
  echo "  DELTA_TOKEN=${DELTA_TOKEN:-0} (hidden=${DELTA_TOKEN_HIDDEN:-512} depth=${DELTA_TOKEN_DEPTH:-2} cap=${DELTA_TOKEN_CAP:-0})"
  echo "  FREEZE_VLA=${FREEZE_VLA:-0}   VAR_WEIGHT=${VAR_WEIGHT:-0}   STATE_EXTRA=${STATE_EXTRA:-none}"
  echo "  ACTION_TARGET_WEIGHT=${ACTION_TARGET_WEIGHT:-0} ACT_WEIGHT=${ACT_WEIGHT:-0} ACT_EPS=${ACT_EPS:-0}"
  echo "  WM_CKPT=${WM_CKPT:-<default>}  TERM_OFF=${TERM_OFF:-0}  LORA_RANK=${LORA_RANK:-32}"
  [[ "${FREEZE_VLA:-0}" == "1" ]] && echo "  ↑ FREEZE_VLA=1 会在训练器里把 lora_rank 强制改成 0（VLA 全冻）"
  true
  exit 0
fi
if [[ "$_TEE" == "1" ]]; then
  exec "${_LAUNCH[@]}" "${_ARGS[@]}" 2>&1 | tee -a "$LOG"
else
  exec "${_LAUNCH[@]}" "${_ARGS[@]}" >> "$LOG" 2>&1
fi
