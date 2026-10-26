#!/usr/bin/env bash
# A1 消融（有无联合微调）在 **arena/Isaac** 里的两臂启动器。
#
# 核心设计（2026-10-24 确认）：
#   臂 A「官方 20k 基线」   → GR00T N1.7 + 官方下游执行器（SONIC decoder）
#                              = HA 的 sonic_wholebody 路线 → 需要 HA_ALLOW_SONIC=1
#   臂 B「联合微调后」      → 我们的 SuperMimic WBC（本仓库项目默认）
#                              = supermimic_wholebody 路线
#
# 为什么两条臂的执行器不同：官方 20k 是**为 SONIC decoder 训的**，联合微调是**为 SuperMimic
# WBC 训的**。用各自的部署执行器评测，才是各自的真实部署表现（这也是「训练=部署一致」的定义）。
#
# ⚠️ 论文里必须写清：A1 比的是「整条系统」（VLA + 其部署执行器），不是「单看 VLA 权重」。
#    若要分离两个因素，见 §「2×2 交叉」里的四格。
#
# 用法：
#   bash scripts/run_a1_arena.sh ARM=base TASK=HOI_pp_box SEEDS="0 1 2"     # SONIC 执行器
#   bash scripts/run_a1_arena.sh ARM=ft   TASK=HOI_pp_box SEEDS="0 1 2"     # SuperMimic 执行器
#   DRY=1 bash scripts/run_a1_arena.sh ARM=ft TASK=HSI_open_door             # 只打印
#
# 依赖（缺一不可）：
#   * B1：arena 7 任务场景包（Google Drive 的 ha_assets.zip）已解压到
#         third_party/HumanoidArena/isaaclab_twist2_g1/assets
#   * B3：VLA HTTP server（scripts/ha_vla_server.py）在跑，返回 latent64
#   * B4：action_provider_supermimic.py（已在仓库里）
set -uo pipefail

# 允许写成位置参数：`bash run_a1_arena.sh ARM=ft TASK=HOI_pp_box SEEDS="0 1 2"`
# （等价于把它放在命令前面当环境变量）
for _arg in "$@"; do
  case "$_arg" in
    *=*) export "$_arg" ;;
    *)   echo "[warn] 忽略非 KEY=VALUE 参数: $_arg" >&2 ;;
  esac
done

COMIMIC=$(cd "$(dirname "$0")/.." && pwd)
HA=$COMIMIC/third_party/HumanoidArena/isaaclab_twist2_g1
EVAL=$HA/script/eval_scripts/sonic

ARM=${ARM:-ft}                       # base | ft
TASK=${TASK:-HOI_pp_box}
SEEDS=${SEEDS:-"0 1 2 3 4"}
DRY=${DRY:-0}

# 臂的 checkpoint 与执行器
case "$ARM" in
  base)
    VLA_CKPT=${VLA_CKPT:-$COMIMIC/runs/ckpt_vla20k}            # 官方 20k（未联合微调）
    ACTION_SOURCE=sonic_wholebody                              # 官方下游执行器
    NEED_SONIC=1
    ;;
  ft)
    VLA_CKPT=${VLA_CKPT:-$COMIMIC/runs/ckpt_vla_ft4000}        # 联合微调 4000 步
    ACTION_SOURCE=supermimic_wholebody                         # 我们的 WBC
    NEED_SONIC=0
    ;;
  *) echo "ARM must be base|ft"; exit 1 ;;
esac

OUT=${OUT:-$COMIMIC/runs/ablations/A1_joint_finetune/$ARM/$TASK}
SERVER_URL=${SERVER_URL:-http://127.0.0.1:18080}
SM_CKPT=${SM_CKPT:-$COMIMIC/trackerlab-supermimic/logs/bones_seed_token_2x8192_isaac50/last.pt}
SM_OBS_MODE=${SM_OBS_MODE:-auto}   # auto = 跟随 ckpt 的 args.obs_mode（S1.3/F2）
PORT_BASE=${PORT_BASE:-18080}

say() { echo "[$(date +%H:%M:%S)] $*"; }

[[ -d "$HA/assets/objects" ]] || { echo "[fail] 缺 arena 场景资产（B1）：$HA/assets"; exit 1; }
[[ -d "$COMIMIC/$VLA_CKPT" || -d "$VLA_CKPT" ]] || { echo "[fail] 缺 VLA checkpoint：$VLA_CKPT"; exit 1; }
if [[ "$NEED_SONIC" == "1" ]]; then
  say "arm=base  -> 执行器 = SONIC decoder（官方下游路线，需 HA_ALLOW_SONIC=1）"
else
  say "arm=ft    -> 执行器 = SuperMimic WBC（$SM_CKPT, obs_mode=$SM_OBS_MODE）"
fi

for seed in $SEEDS; do
  mkdir -p "$OUT/seed_$seed"
  cmd=(python "$EVAL/sim_eval_vla.py"
       --task "$TASK" --seed "$seed"
       --lerobot_server_url "$SERVER_URL"
       --result_json "$OUT/seed_$seed/result.json"
       --success_video_dir "$OUT/seed_$seed/videos"
       --model_label "$ARM"
       --eval_model_path "$VLA_CKPT")
  if [[ "$NEED_SONIC" == "0" ]]; then
    cmd+=(--sm_ckpt "$SM_CKPT" --sm_obs_mode "$SM_OBS_MODE" --sm_token_source http)
  fi

  if [[ "$DRY" == "1" ]]; then
    echo "[dry] HA_ALLOW_SONIC=$NEED_SONIC HA_ACTION_SOURCE=$ACTION_SOURCE ${cmd[*]}"
    continue
  fi

  say "seed=$seed -> $OUT/seed_$seed"
  ( cd "$HA" && \
    HA_ALLOW_SONIC="$NEED_SONIC" HA_ACTION_SOURCE="$ACTION_SOURCE" \
    HA_GMT_BACKEND=$([[ "$NEED_SONIC" == "1" ]] && echo sonic || echo supermimic) \
    PYTHONPATH="$HA:$EVAL:$PYTHONPATH" "${cmd[@]}" ) 2>&1 | tee "$OUT/seed_$seed/run.log"
done

say "done: ARM=$ARM TASK=$TASK -> $OUT"
say "汇总： python3 $COMIMIC/scripts/ablation_report.py --root $COMIMIC/runs/ablations"
