#!/usr/bin/env bash
# A3 (BPTT vs PPO) + A5 (head-only vs head+WBC) —— 卡 2,3（本容器 2×4090）的正式启动器。
#
# 与 docs/ABLATION_ALLOCATION.md §2 的命令逐字对应，只多了三件工程上必须的事：
#   1. PYTHONPATH 里必须带 gr00t_deps + _noflash，否则 GR00T action head import 不了
#      （tyro 缺失），或者 fp32 VLA 前向撞上 FlashAttention 的 dtype 限制；
#   2. SuperMimic 用 runs/OFFICIAL_SUPERMIMIC.json 里写死的 Isaac ckpt：
#      `diff_vla_wbc/supermimic_backend.py::from_checkpoint` 自动识别 {pi,wm} 格式并转
#      TrackerLabBundle（policy/WM 是 Isaac 权重，FK/奖励/观测/终止头用解析层）；
#   3. 每个臂结束时都落盘（joint_latest.pt / vla_final.pt），A5 的「是否真的冻结」审计
#      才有东西可比，A3 的两份 vla 也能直接接 A1 的成功率流程。
#
# 用法：
#   bash scripts/launch_a3_a5.sh                                   # 全跑（120+120+40+40 min）
#   MODE=bptt bash scripts/launch_a3_a5.sh                         # 只跑 BPTT 臂
#   STEPS=3 MINUTES=0 MODE=bptt bash scripts/launch_a3_a5.sh       # 冒烟（3 步就停）
#   MODE=a5 A5_MINUTES=40 bash scripts/launch_a3_a5.sh
set -euo pipefail

# CODE_ROOT 默认指向本脚本所在代码树；用快照跑时传 CODE_ROOT=<snapshot> 即可冻结代码。
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
SM_CKPT="${SM_CKPT:-$COMIMIC/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt}"
VLM_PATH="${VLM_PATH:-$COMIMIC/groot_repro/nvidia/Cosmos-Reason2-2B-weights}"
VLM_CKPT="${VLM_CKPT:-$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}"
MOTION="${MOTION:-$COMIMIC/data/npz/ha_tokens}"          # 纯 HA（699 条 / 700 npz），禁止 ft318/混合
PACKED_DIR="${PACKED_DIR:-$COMIMIC/runs/isaac_motion/ha_tokens}"   # Isaac(trackerlab) 打包运动库，HA-only
OUT_ROOT="${OUT_ROOT:-$COMIMIC/runs/ablations}"
MODE="${MODE:-all}"                  # all | bptt | ppo | a5
STEPS="${STEPS:-1000000}"
MINUTES="${MINUTES:-120}"            # A3 每臂墙钟
A5_MINUTES="${A5_MINUTES:-40}"       # A5 每臂墙钟
EVAL_EVERY="${EVAL_EVERY:-300}"      # BPTT / A5
PPO_EVAL_EVERY="${PPO_EVAL_EVERY:-100}"
PORT="${PORT:-29642}"
BATCH="${BATCH:-2}"                  # 每 rank 的 batch（2 卡 → 全局 4）
HORIZON="${HORIZON:-8}"
LOG_EVERY="${LOG_EVERY:-50}"
LOG_DIR="${LOG_DIR:-$COMIMIC/logs}"

for p in "$SM_CKPT" "$VLM_CKPT" "$VLM_PATH" "$MOTION"; do
  [[ -e "$p" ]] || { echo "[fail] missing $p"; exit 1; }
done

# ---- 硬门禁 1：数据只允许纯 HA，永远不碰 ft318 / mixed（用户口径，写死）----------
case "$MOTION" in
  *ft318*|*mixed*|*Mixed*)
    echo "[fail] MOTION=$MOTION 命中 ft318/mixed；消融只允许纯 HA（data/npz/ha_tokens）"
    exit 2;;
esac
case "$(basename "$MOTION")" in
  ha_tokens|ha_sessions) ;;
  *) echo "[warn] MOTION 目录名不是 ha_tokens/ha_sessions：$MOTION（确认它确实是纯 HA 数据）";;
esac
case "$PACKED_DIR" in
  *ft318*|*mixed*|*Mixed*)
    echo "[fail] PACKED_DIR=$PACKED_DIR 命中 ft318/mixed；Isaac 打包运动库也只允许纯 HA"
    exit 2;;
esac

# ---- 硬门禁 2：SuperMimic 必须是 Isaac Lab 训出来的 {pi,wm} 格式，不许用 mjlab 旧链路 ----
if ! /usr/bin/python3 - "$SM_CKPT" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert "pi" in ck and "wm" in ck and "policy" not in ck, (
    "not an Isaac/trackerlab checkpoint; refusing the mjlab (MuJoCo) chain")
print(f"[gate] Isaac WBC ok: iteration={ck.get('iteration')}, "
      f"obs_mode={(ck.get('args') or {}).get('obs_mode')}, interface={(ck.get('args') or {}).get('interface')}")
PY
then
  echo "[fail] $SM_CKPT 不是 Isaac/trackerlab 的 {pi,wm} ckpt —— 拒绝在旧 mjlab 链路上跑消融"
  exit 3
fi

# GR00T 私有依赖 + 假 flash_attn（迫使 transformers 回落 sdpa；fp32 VLA 只吃 sdpa）
export PYTHONPATH="$CODE_ROOT:$COMIMIC/third_party/gr00t_deps:$COMIMIC/third_party/gr00t_deps/_noflash${PYTHONPATH:+:$PYTHONPATH}"
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
mkdir -p "$LOG_DIR"
cd "$CODE_ROOT"
echo "[code] running from $CODE_ROOT"

run_arm () {                 # run_arm <tag> <script> <outdir> <minutes> <eval_every> <extra...>
  local tag="$1" script="$2" out="$3" minutes="$4" eval_every="$5"; shift 5
  mkdir -p "$out"
  echo "==================================================================" | tee -a "$LOG_DIR/$tag.launcher.log"
  echo "[arm $tag] start $(date -Is) minutes=$minutes script=$script out=$out" | tee -a "$LOG_DIR/$tag.launcher.log"
  echo "==================================================================" | tee -a "$LOG_DIR/$tag.launcher.log"
  /usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port "$PORT" \
    "$script" \
      --motion "$MOTION" --motion-packed-dir "$PACKED_DIR" --supermimic-ckpt "$SM_CKPT" \
      --vlm --vlm-path "$VLM_PATH" --vlm-ckpt "$VLM_CKPT" \
      --batch-size "$BATCH" --horizon "$HORIZON" --steps "$STEPS" \
      --hand-loss-weight 0.2 --time-budget-minutes "$minutes" \
      --eval-every "$eval_every" --eval-batches 2 \
      --eval-out "$out/eval_curve.jsonl" \
      --log-every "$LOG_EVERY" \
      --out "$out" \
      "$@" 2>&1 | tee -a "$LOG_DIR/$tag.log"
  echo "[arm $tag] finished $(date -Is)" | tee -a "$LOG_DIR/$tag.launcher.log"
}

# ------------------------------------------------------------------ A3: BPTT arm
if [[ "$MODE" == "all" || "$MODE" == "bptt" ]]; then
  run_arm a3_bptt scripts/train_joint.py "$OUT_ROOT/A3_bptt_vs_ppo/bptt" "$MINUTES" "$EVAL_EVERY" \
    --vlm-dataset --vlm-steps 2 --fm-grad-backend pathwise --hand-grasp-boost 5 --state-vel-weight 0 \
    --save --save-every 0
fi

# ------------------------------------------------------------------ A3: PPO arm
if [[ "$MODE" == "all" || "$MODE" == "ppo" ]]; then
  # --no-prefetch-vla: the background prefetch thread decodes ego-view frames with PyAV from a
  # second thread and segfaults inside libavcodec (SIGSEGV, 2026-10-25 04:54, rank 0).  Prefetch
  # is numerically identical to the in-line path (scripts/check_prefetch_parity.py), so turning it
  # off costs a little speed and changes nothing else.
  run_arm a3_ppo scripts/train_ppo.py "$OUT_ROOT/A3_bptt_vs_ppo/ppo" "$MINUTES" "$PPO_EVAL_EVERY" \
    --vlm-steps 2 --ppo-epochs 4 --sigma-init 0.25 --hand-grasp-boost 5 --no-prefetch-vla
fi

# ------------------------------------------------------------------ A5: head-only vs head+WBC
if [[ "$MODE" == "all" || "$MODE" == "a5" ]]; then
  run_arm a5_head_only scripts/train_joint.py "$OUT_ROOT/A5_only_head/head_only" "$A5_MINUTES" "$EVAL_EVERY" \
    --vlm-dataset --vlm-steps 2 --policy-lr 0 --adapter-lr 0 --hand-grasp-boost 5 --state-vel-weight 0 \
    --save --save-every 0
  run_arm a5_head_plus_wbc scripts/train_joint.py "$OUT_ROOT/A5_only_head/head_plus_wbc" "$A5_MINUTES" "$EVAL_EVERY" \
    --vlm-dataset --vlm-steps 2 --hand-grasp-boost 5 --state-vel-weight 0 \
    --save --save-every 0
fi

echo "[done] mode=$MODE $(date -Is)"
