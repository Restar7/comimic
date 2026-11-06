#!/usr/bin/env bash
# ============================================================================
# GRU vs MLP 的 **WM 精度**消融（只跑 WM，不碰 collect / gate / joint / closed-loop）
# ============================================================================
# 目的：回答一个问题——给 WM 加一层 GRU 记忆，能不能把 k 步预测误差压下去？
#       （交接文档判据：k=1 ≤0.2 cm 且 k=20 ≤3 cm；基线 wm_d3 = 0.1 cm / 6.5 cm）
#
# 设计：**唯一变量是 --arch**。数据、阶段划分、配比、步数、lr、batch 全部照抄 d3：
#       阶段1  纯 A 线（WBC-only 真物理），K=1，3000 步，batch 256，lr 1e-4
#       阶段2  A:B = 103:103（每文件 103 条 A + 822 条 B 里抽一半），K=1，3000 步，lr 3e-5
#       阶段2b A+B 混合，K=20，batch 128，lr 1e-5，1500 步，早停看环回留出 7 条
#   ⇒ 与 runs/S3_pipeline/d3 的 4a/4b/4_wm_refresh 三条命令逐参数对齐。
#
#   第二版（2b 记忆配方）复用同一份阶段1/2 产物，只把 2b 换成 lr 3e-5 / 3000 步：
#   因为阶段1/2 是 K=1，GRU 的循环矩阵 weight_hh 拿不到梯度（已在
#   scripts/test_wm_arch.py 里断言过），记忆**只能在 2b 学会**，所以给它一次更充分的预算。
#
# 用法：
#   ROUND=d4gru ARCH=gru GPU=0 bash scripts/run_wm_gru_ablation.sh stage12
#   ROUND=d4gru ARCH=gru GPU=0 bash scripts/run_wm_gru_ablation.sh 2b a 1500 1e-5
#   ROUND=d4gru ARCH=gru GPU=1 bash scripts/run_wm_gru_ablation.sh 2b b 3000 3e-5
#   ROUND=d4gru bash scripts/run_wm_gru_ablation.sh eval
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$ROOT/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd}
[ -f "$SUPERMIMIC_ROBOT_USD" ] || { echo "[abort] ROBOT_USD 不存在：$SUPERMIMIC_ROBOT_USD"; exit 1; }

ROUND=${ROUND:-d4gru}
ARCH=${ARCH:-gru}
GPU=${GPU:-0}
CLIP=${CLIP:-1.0}
K1=${K1:-1}      # 阶段1 的多步展开步数（1=d3 配方；>1 让 GRU 的 weight_hh 在阶段1 就拿到梯度）
K2=${K2:-1}      # 阶段2 的多步展开步数
A_ORDER=${A_ORDER:-ha_v31}   # ★ A 线关节存储顺序：ha_v31=沿用旧行为（**实测是错的**，会把 A 线换序）；
                             #   declared=A 线 meta 自称 STANDARD（实测 corr 0.698，换序后 0.034）⇒ 不换序
VAL_EVERY=${VAL_EVERY:-100}  # 2b 的早停间隔；0=不早停（存**最后一步**的权重，位置/姿态更好）

PD=$ROOT/runs/S3_pipeline/$ROUND
mkdir -p "$PD"

BASE_WM=${BASE_WM:-$ROOT/runs/S1_handover/WBC_A24000_wmvel_v6.pt}
PACKED=$ROOT/runs/isaac_motion/ha_cmd/packed.pt
A_GLOB="$ROOT/runs/wm_real/ac_w*.pt"
# ★ 训练数据必须与 d3 **同一批**（822 条 B 轨迹 + 7 条同分布留出），否则不可比。
TRAJ=$ROOT/runs/wm_real/traj_d3_train
VAL=$ROOT/runs/wm_real/traj_d3_val
# 阶段2/2b 的 A 线每文件取多少条：d3 是 round(1 × 822 / 8) = 103
A_ENVS_2=${A_ENVS_2:-103}
MIX_B=0.5
TRAJ_FRAC=0.6
TRAJ_VAL_FRAC=0.25

hr() { echo "────────────────────────────────────────────────────────────"; }
say() { echo "[$(date +%H:%M:%S)] $*"; }

need_files() {  # 前置检查
  local n_a n_b n_v
  n_a=$(ls $A_GLOB 2>/dev/null | wc -l)
  n_b=$(ls "$TRAJ"/*.pt 2>/dev/null | wc -l)
  n_v=$(ls "$VAL"/*.pt 2>/dev/null | wc -l)
  [[ $n_a -ge 1 ]] || { echo "[abort] 没有 A 线 $A_GLOB（先跑 a_collect）"; return 1; }
  [[ $n_b -eq 822 ]] || echo "[warn] B 轨迹 $n_b 条（d3 是 822，若不是同一批则不可比）"
  [[ $n_v -ge 1 ]] || { echo "[abort] 没有环回留出 $VAL"; return 1; }
  say "A 线 $n_a 个文件 / B 轨迹 $n_b 条 / 环回留出 $n_v 条 / arch=$ARCH / GPU=$GPU"
}

# --------------------------------------------------------------------- 阶段1
stage1() {
  local out=$ROOT/runs/S1_handover/wm_${ROUND}_a.pt
  hr; say "阶段1（纯 A 线，K=1，3000 步，lr 1e-4，arch=$ARCH）→ $out"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python3 scripts/finetune_wm_velocity_policy.py \
    --ckpt "$BASE_WM" --out "$out" --arch "$ARCH" --clip-norm "$CLIP" \
    --packed "$PACKED" --token-space label \
    --rollout-steps "$K1" --steps 3000 --batch 256 --lr 1e-4 \
    --vel-lambda 5 --w-lambda 3.0 --vel-scale-lambda 1.0 \
    --mix-lib 0 --mix-b 0.0 \
    --traj-dump "" --traj-frac 4.0 --traj-val-frac "$TRAJ_VAL_FRAC" \
    --a-line "$A_GLOB" --a-line-envs 512 --a-order "$A_ORDER" \
    --val-horizon 0 --device cuda:0 > "$PD/4a_wm_p1.log" 2>&1
  local rc=$?
  tail -3 "$PD/4a_wm_p1.log"
  [[ $rc == 0 && -s "$out" ]] || { echo "[die] 阶段1失败 rc=$rc（看 $PD/4a_wm_p1.log）"; return 1; }
}

# --------------------------------------------------------------------- 阶段2
stage2() {
  local a=$ROOT/runs/S1_handover/wm_${ROUND}_a.pt
  local out=$ROOT/runs/S1_handover/wm_${ROUND}_p2.pt
  [[ -s "$a" ]] || { echo "[abort] 阶段1产物不存在：$a"; return 1; }
  hr; say "阶段2（A+B 混合，K=1，3000 步，lr 3e-5，arch=$ARCH）→ $out"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python3 scripts/finetune_wm_velocity_policy.py \
    --ckpt "$a" --out "$out" --arch "$ARCH" --clip-norm "$CLIP" \
    --packed "$PACKED" --token-space label \
    --rollout-steps "$K2" --steps 3000 --batch 256 --lr 3e-5 \
    --vel-lambda 5 --w-lambda 3.0 --vel-scale-lambda 1.0 \
    --mix-lib 0 --mix-b "$MIX_B" \
    --traj-dump "$TRAJ/*.pt" --traj-frac "$TRAJ_FRAC" --traj-val-frac "$TRAJ_VAL_FRAC" \
    --a-line "$A_GLOB" --a-line-envs "$A_ENVS_2" --a-order "$A_ORDER" \
    --val-horizon 0 --device cuda:0 > "$PD/4b_wm_p2.log" 2>&1
  local rc=$?
  grep -E "\[mix\] 整批配比" "$PD/4b_wm_p2.log" | tail -1
  tail -2 "$PD/4b_wm_p2.log"
  [[ $rc == 0 && -s "$out" ]] || { echo "[die] 阶段2失败 rc=$rc（看 $PD/4b_wm_p2.log）"; return 1; }
  # 配比断言（与管线同一口径：B 占 A+B 的 50% ±20%）
  python3 - "$PD/4b_wm_p2.log" <<'PYR'
import re, sys
txt = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
m = re.findall(r"\[traj\] 合计 (\d+) 条轨迹（B=(\d+) A=(\d+)）", txt)
tot, b, a = (int(x) for x in m[-1])
share = b / max(1, tot)
print(f"[ratio] 阶段2 实际 B={b} A={a} ⇒ B 占比 {share*100:.1f}%（设计 50%）")
raise SystemExit(0 if 0.40 <= share <= 0.60 else 1)
PYR
  [[ $? == 0 ]] || { echo "[die] 阶段2 的 A/B 配比不符（见上）"; return 1; }
}

# --------------------------------------------------------------------- 阶段2b
stage2b() {  # $1=tag $2=steps $3=lr
  local tag=${1:?tag} steps=${2:-1500} lr=${3:-1e-5}
  local p2=$ROOT/runs/S1_handover/wm_${ROUND}_p2.pt
  local out=$ROOT/runs/S1_handover/wm_${ROUND}_${tag}.pt
  [[ -s "$p2" ]] || { echo "[abort] 阶段2产物不存在：$p2"; return 1; }
  hr; say "阶段2b[$tag]（K=20 自回归，$steps 步，lr $lr，arch=$ARCH，GPU=$GPU）→ $out"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python3 scripts/finetune_wm_velocity_policy.py \
    --ckpt "$p2" --out "$out" --arch "$ARCH" --clip-norm "$CLIP" \
    --packed "$PACKED" --token-space label \
    --rollout-steps 20 --steps "$steps" --batch 128 --lr "$lr" \
    --vel-lambda 5 --w-lambda 3.0 --vel-scale-lambda 1.0 \
    --mix-lib 0 --mix-b "$MIX_B" \
    --traj-dump "$TRAJ/*.pt" --traj-frac "$TRAJ_FRAC" --traj-val-frac "$TRAJ_VAL_FRAC" \
    --a-line "$A_GLOB" --a-line-envs "$A_ENVS_2" --a-order "$A_ORDER" \
    --val-dump "$VAL/*.pt" --val-horizon 20 --val-every "$VAL_EVERY" --val-max-starts 16 --val-max-traj 16 \
    --device cuda:0 > "$PD/4_2b_$tag.log" 2>&1
  local rc=$?
  grep -E "\[before val\]|\[after val\]|\[early-stop\]|\[saved\]" "$PD/4_2b_$tag.log" | tail -4
  [[ $rc == 0 && -s "$out" ]] || { echo "[die] 阶段2b[$tag] 失败 rc=$rc（看 $PD/4_2b_$tag.log）"; return 1; }
  say "阶段2b[$tag] ✅ $out"
}

# --------------------------------------------------------------------- 评测
do_eval() {  # 剩下的参数是要评估的 ckpt
  local ckpts=("$@")
  [[ ${#ckpts[@]} -gt 0 ]] || ckpts=("$ROOT/runs/S1_handover/wm_${ROUND}.pt")
  hr; say "评测（同一留出集 runs/wm_real/traj_wmuse2_val，134 条；k=1/5/10/20）"
  local abs=()
  for c in "${ckpts[@]}"; do abs+=("$(readlink -f "$c")"); done
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python3 scripts/eval_wm_multistep.py \
    --ckpt "${abs[@]}" \
    --dump "$ROOT/runs/wm_real/traj_wmuse2_val/*.pt" \
    --horizon 20 --stride 20 --device cuda:0 \
    --out "$PD/eval_wm_${ROUND}.json" 2>&1 | grep -vE "^\[AUDIT|^\[bundle\] Isaac"
  say "评测日志：$PD/eval_wm_${ROUND}.json"
}

case "${1:-}" in
  stage12) need_files || exit 1; stage1 || exit 1; stage2 || exit 1; say "stage12 ✅" ;;
  stage1)  need_files || exit 1; stage1 || exit 1 ;;
  stage2)  need_files || exit 1; stage2 || exit 1 ;;
  2b)      shift; need_files || exit 1; stage2b "$@" || exit 1 ;;
  eval)    shift; do_eval "$@" ;;
  *) echo "用法: ROUND=.. ARCH=.. GPU=.. bash $0 {stage1|stage2|stage12|2b <tag> <steps> <lr>|eval [ckpts...]}"; exit 2 ;;
esac
