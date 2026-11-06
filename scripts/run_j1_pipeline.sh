#!/usr/bin/env bash
# 一条命令跑完 W1 之后的**全部**流程（循环第 1 轮）：
#
#   ① 大规模采集（mjlab env 内，向量化）   → runs/wm_real/mjlab_transitions.pt
#   ② WM 验收门（正确口径：比仿真真值）    → runs/gates/wm_qd_corr.json
#   ③ J1 五道门禁（codec/冻结/burn-in/token 对比度/FM-BC） → runs/joint_mjlab_ft318/gates.json
#   ④ BC warmup（λ_r=0，默认 800 步）      → joint_/vla_ ckpt
#   ⑤ 阶段 B（+奖励 + critic + WM 保鲜）   → joint_/vla_ ckpt
#   ⑥ 环境内评测（318 库 benchmark）       → runs/joint_mjlab_ft318/bench*.json
#
# 用法：
#   bash scripts/run_j1_pipeline.sh                 # 全流程
#   W1_CKPT=.../ckpt_0007000.pt bash scripts/run_j1_pipeline.sh
#   SKIP_COLLECT=1 bash scripts/run_j1_pipeline.sh  # 复用已有 transitions
#   SMOKE=1 bash scripts/run_j1_pipeline.sh         # 每步都缩到极小，只验证链路
#
# 每一步失败都会**立刻停**（set -e + 门禁非 0 退出），不会带着坏口径往下跑。
set -euo pipefail

COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
REPO=$COMIMIC/third_party/SuperMimic-Reproduced
VENV=$REPO/.venv/bin/python
TORCHRUN=$REPO/.venv/bin/torchrun
PY=python3

RUN=${RUN:-wbc_ft318_mjlab}
W1_CKPT=${W1_CKPT:-$REPO/logs/$RUN/ckpt_0007000.pt}
VLA_CKPT=${VLA_CKPT:-/inspire/hdd/global_user/liumengfan-253108110079/flowbody/FlowBody/runs/groot_sonic_vla_20k_frozen271_20261023/checkpoint-20000}
VLM_PATH=${VLM_PATH:-$COMIMIC/groot_repro/nvidia/Cosmos-Reason2-2B-weights}
LIB=$COMIMIC/data/library/ft318_compact.pt
TOK=$COMIMIC/data/library/ft318_tokens.pt
MOTION=$COMIMIC/data/npz/g1_tasks
OUT=${OUT:-$COMIMIC/runs/joint_mjlab_ft318}
TRANS=${TRANS:-$COMIMIC/runs/wm_real/mjlab_transitions.pt}
BLINE=${BLINE:-$COMIMIC/runs/bline}
BLINE_TICKS=${BLINE_TICKS:-600}
BLINE_STARTS=${BLINE_STARTS:-100 2000 9000 20000 40000}   # 5 个不同起点（库行号）
GATES=$OUT/gates_wm_qd.json

ENVS=${ENVS:-1024}          # 采集只需 ~1M 条；4096×4000 会产 1640 万条/19 GB，过量          # 采集并行 env 数（H200 4096 很轻）
STEPS=${STEPS:-1000}        # 每个 env 推进的控制 tick 数（并行推进，不是串行 4000 步）        # 每个 env 推进步数
CAP=${CAP:-1000}            # 环形缓冲长度
BC_STEPS=${BC_STEPS:-800}
REWARD_STEPS=${REWARD_STEPS:-2000}
J1_BATCH=${J1_BATCH:-64}    # H200 用 64（每卡）；4090 请 16
J1_LORA=${J1_LORA:-64}
J1_H=${J1_H:-20}
NGPU=${NGPU:-2}             # H200 双卡；1 = 单卡
GPU_LIST=${GPU_LIST:-0,1}
AMP=${AMP:-bf16}
J1_FMLR=${J1_FMLR:-1e-4}   # 与 docs/HANDOVER_4090_COMMANDS_20261028.md §5.3 一致

if [[ ${SMOKE:-0} == 1 ]]; then
  ENVS=64; STEPS=60; CAP=60; BC_STEPS=4; REWARD_STEPS=4; J1_BATCH=2; J1_H=4; J1_LORA=16; GATE_BATCHES=1
fi

export PYTHONPATH=$COMIMIC:$COMIMIC/Isaac-GR00T:$COMIMIC/third_party/gr00t_deps:$COMIMIC/third_party/gr00t_deps/_noflash
export MUJOCO_GL=disable
mkdir -p "$OUT" "$(dirname "$TRANS")" "$(dirname "$GATES")"

step() { printf '\n\033[1m===== %s =====\033[0m\n' "$*"; }
check() { [[ -e "$1" ]] || { echo "[fail] 缺 $1"; exit 1; }; }   # ckpt 可能是目录（HF 格式）

step "0/5 前置检查"
for f in "$W1_CKPT" "$VLA_CKPT" "$LIB" "$TOK"; do check "$f"; done
check "$MOTION"
echo "W1_CKPT=$W1_CKPT"
echo "VLA_CKPT=$VLA_CKPT"
echo "OUT=$OUT  NGPU=$NGPU  GPU_LIST=$GPU_LIST"

# ---------------------------------------------------------------- ① 采集
if [[ ${SKIP_COLLECT:-0} == 1 ]]; then
  step "1/5 采集（跳过，复用 $TRANS）"
  check "$TRANS"
else
  step "1/5 大规模采集：envs=$ENVS steps=$STEPS capacity=$CAP"
  cd "$REPO"
  CUDA_VISIBLE_DEVICES=${GPU_LIST%%,*} env -u PYTHONPATH "$VENV" scripts/collect_mjlab.py \
    --ckpt "$W1_CKPT" --library "$LIB" --tokens "$TOK" \
    --num-envs "$ENVS" --steps "$STEPS" --replay-capacity "$CAP" \
    --action-noise-std 0.1 --out "$TRANS"
fi

# ---------------------------------------------------------------- ② WM 门（正确口径）
step "2/5 WM 验收门（--transitions：比仿真真值）"
cd "$COMIMIC"
"$PY" scripts/probe_wm_qd_corr.py --ckpt "$W1_CKPT" --motion "$MOTION" --tokens "$TOK" \
  --transitions "$TRANS" --n 256 --out "$GATES" || {
    echo "[gate] WM 速度通道未达标 —— 停在这里（不要带着坏 WM 进 J1）"; exit 2; }

# ---------------------------------------------------------------- ③ J1 门禁
step "2b/5 B 线回放采集（frames = 真库行号；给 b_start / critic / 奖励）"
mkdir -p "$BLINE"
for F in $BLINE_STARTS; do
  CUDA_VISIBLE_DEVICES=${GPU_LIST%%,*} "$PY" scripts/closed_loop_vla_supermimic.py \
    --vla-ckpt "$VLA_CKPT" --joint-ckpt "$W1_CKPT" --task task1 \
    --steps "$BLINE_TICKS" --device cuda:0 --no-video \
    --replay-lib "$LIB" --start-frame "$F" --arm gt --tokens "$TOK" \
    --dump "$BLINE/gt_f${F}.pt" || echo "[warn] B 线 f${F} 失败（跳过）"
done
echo "[bline] $(ls "$BLINE"/gt_f*.pt 2>/dev/null | wc -l) 条 -> $BLINE"

step "3/5 J1 五道门禁（不训练）"
# 一道门一个进程：freeze 之后显存收不回来，同进程里跑 burn-in 必 OOM（4090 实测 45/46 GB）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for G in A_codec B_freeze C_burnin C2_token_contrast D_fm_bc; do
  echo "[gate] $G"
  run "$PY" scripts/train_joint_mjlab.py \
    --wbc-ckpt "$W1_CKPT" --motion "$MOTION" --tokens "$TOK" \
    --vlm-ckpt "$VLA_CKPT" --vlm-path "$VLM_PATH" \
    --out $OUT/gates --gate-only --gate-subset "$G" --gate-batches 4 \
    --lora-rank "$J1_LORA" --amp "$AMP" --device cuda || { echo "[gate] $G FAIL"; exit 3; }
done

# ---------------------------------------------------------------- ④ 联合微调（Isaac v3 口径）
# 口径锁定（对齐 docs/HANDOVER_4090_COMMANDS_20261028.md §5.3）：
#   lambda_bc=0（token 不做 BC）/ lambda_hand=0.2（hand-only 掩码）/ lambda_r=0.3
#   --no-critic（critic v_ev 为负，第 1 轮关掉）/ act 信任域 0.6/1.0 / hinge cap 2
#   --vla-state-from-burnin（A1：把 burn-in 末状态写回 VLA）/ H=20（部署重规划周期）
BASE_ARGS=(
  --wbc-ckpt "$W1_CKPT" --motion "$MOTION" --tokens "$TOK"
  --vlm-ckpt "$VLA_CKPT" --vlm-path "$VLM_PATH"
  --batch-size "$J1_BATCH" --grad-accum 1 --horizon "$J1_H"
  --lora-rank "$J1_LORA" --amp "$AMP" --device cuda --fm-lr "$J1_FMLR"
)
if [[ $NGPU -gt 1 ]]; then
  # 必须用**系统 python** 起 torchrun：venv 的 torchrun 会把 .venv/site-packages 放进
  # sys.path ⇒ `import mjlab` 命中**真 mjlab**（它要新版 tyro），与 gr00t_deps 的 tyro 0.9.17 冲突，
  # 报 `module 'tyro.conf' has no attribute 'UsePythonSyntaxForLiteralCollections'`（2026-10-28 21:18 实测）
  LAUNCH=("$PY" -m torch.distributed.run --nproc_per_node "$NGPU" --master_port "${PORT:-29670}")
else
  LAUNCH=("$PY")
fi

step "4/5 联合微调：${REWARD_STEPS} 步（token 不做 BC · hand 0.2 · rl 0.3 · no-critic · H=${J1_H}）"
"${LAUNCH[@]}" scripts/train_joint_mjlab.py "${BASE_ARGS[@]}" \
  --out "$OUT/rl" --steps "$REWARD_STEPS" --v3 \
  --lambda-bc 0 --lambda-hand 0 --lambda-r 0.3   # hand 关（用户口径 10-29） \
  --no-critic --act-eps 0.6 --act-weight 1.0 --act-hinge-cap 2 \
  --vla-state-from-burnin --train-wbc --wbc-lr 1e-6 \
  --real-transitions "$TRANS" --start-source reference \
  --start-from-b "$BLINE/gt_f*.pt" --b-start-order standard --b-start-nonstrict \
  --wm-refresh-every 500 --wm-refresh-steps 200 \
  --anchor-weight 0 --trust-eps 0.2 --trust-weight 1.0   # 不锚定；token 信任域=开，中心=self \
  --trust-center self \
  --save-every 500 --save-vla-every 500 --log-every 10

# ---------------------------------------------------------------- ⑥ 评测（闭环，评的是 VLA head）
step "5/5 合并阶段 B 的 head → 闭环评测（MuJoCo，600 tick）"
RL_CKPT=$OUT/ckpt_rl
"$PY" scripts/export_vla_ckpt.py --base "$VLA_CKPT" --vla "$OUT/rl/vla_final.pt" \
  --out "$RL_CKPT" --force
for K in 1; do
  CUDA_VISIBLE_DEVICES=${GPU_LIST%%,*} "$PY" scripts/closed_loop_vla_supermimic.py \
    --vla-ckpt "$RL_CKPT" --joint-ckpt "$W1_CKPT" \
    --task task1 --steps 600 --out-video "$OUT/cl_task1.mp4" \
    --out-json "$OUT/cl_task1.json" --device cuda:0 \
    --dump "$OUT/cl_task1_dump.pt" || echo "[warn] 闭环 task1 失败（跳过，不影响训练产物）"
done

step "完成"
cat <<EOF
产物：
  采集        $TRANS
  WM 门       $GATES
  J1 门禁     $OUT/gates.json
  阶段 B      $OUT/rl/{joint_final,vla_final}.pt
  评测        $OUT/bench_*.json
下一轮（循环）：把 $OUT/rl/vla_final.pt 合并进 VLA ckpt，再跑本脚本
  python3 scripts/export_vla_ckpt.py --base "$VLA_CKPT" --vla "$OUT/rl/vla_final.pt" --out runs/ckpt_vla_j1_r1
  VLA_CKPT=runs/ckpt_vla_j1_r1 bash scripts/run_j1_pipeline.sh
EOF
