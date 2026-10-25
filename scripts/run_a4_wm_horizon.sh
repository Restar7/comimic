#!/usr/bin/env bash
# =============================================================================
# A4 消融（本机 = A4 的 **4、5 号卡**，2× RTX 4090 48G）
#
#   变量 1：世界模型要不要在线更新
#       wm_frozen : --wm-update-every 0                        （主线口径）
#       wm_live   : --wm-update-every 50 --wm-update-warmup 100 --wm-update-batch 8
#   变量 2：imagination horizon  H ∈ {8,16,32,40}
#       40 是硬上界：VLA 一次吐 40 帧 token chunk，imagine_with_tokens 用
#       tokens[:, min(t, T-1)]，t>40 只是重复最后一帧。
#
# 全部写死，不允许任何覆盖（脚本内直接拒绝 mixed / ft318 数据集）：
#   * WBC + 世界模型 + 99 维观测 + 奖励 + 软终止 + 可微 FK + 参考运动库
#     => 全部来自 trackerlab-supermimic（Isaac Lab / PhysX 实现）
#     => mjlab(third_party/SuperMimic-Reproduced) 的权重/解析头/MuJoCo 栈一律不参与
#   * 数据 => 纯 HA：data/npz/ha_tokens（699 clips / 634,016 frames，与官方 VLA 微调同源）
#   * 固定墙钟 40 min/格（跨臂可比），--steps 给极大值由墙钟截断
#
# 用法（无参数；改时长请编辑下面的 MINUTES）：
#   setsid nohup bash scripts/run_a4_wm_horizon.sh > logs/a4_wm_horizon.log 2>&1 &
# =============================================================================
set -uo pipefail
# NOTE(2026-10-25): 定义性 A4 运行已改为从代码快照跑：
#   bash runs/_code_a4_20261025/scripts/run_a4_v3.sh
# 本脚本保留为对照；两者都会在检测到已有 train_joint 时拒绝启动。

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC"

# ---- 写死：本机就是 4、5 号卡 ------------------------------------------------
GPUS=0,1
# ---- 写死：Isaac Lab / PhysX 版 SuperMimic（runs/OFFICIAL_SUPERMIMIC.json）----
SM_CKPT="$COMIMIC/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt"
# ---- 写死：纯 HA 数据（**禁止** mixed / ft318）-------------------------------
MOTION="$COMIMIC/data/npz/ha_tokens"
PACKED="$COMIMIC/runs/isaac_motion/ha_tokens/packed.pt"
# ---- 写死：官方 VLA 微调产物（runs/OFFICIAL_CKPT.json）----------------------
VLM_PATH="$COMIMIC/groot_repro/nvidia/Cosmos-Reason2-2B-weights"
VLM_CKPT="$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000"

OUTROOT="$COMIMIC/runs/ablations/A4_wm_horizon"
LOGDIR="$COMIMIC/logs/a4"
MINUTES=30                 # 每格墙钟（实测 reward 在 ~12 min 就进平台，30 min 给 2.5x 平台期）
EVAL_BATCHES=6             # 每个评测点固定 6 个窗口（旧版 2 个 → 曲线噪声过大）
BATCH=1                    # 两卡各 1（全局 batch 2）。**全网格统一**：
                           # 实测 bs=2 时单卡峰值 ~46/48 GB（1.62B 头部的 fp32 优化器态
                           # + 梯度 + all-reduce 缓冲），H 越大越险；bs=1 把峰值压到 ~40 GB
                           # 以下，且六格 batch 一致 → H 与 WM 两个变量的对比干净。
PID_BASE=29741

case "$MOTION" in
  *mixed*|*ft318*) echo "[fail] 数据集被永久禁用：$MOTION（只允许纯 HA）"; exit 1;;
esac

export PYTHONPATH="$COMIMIC/third_party/gr00t_deps:$COMIMIC/third_party/gr00t_deps/_noflash${PYTHONPATH:+:$PYTHONPATH}"
# 40 min 长跑 + 大优化器态：expandable_segments 抑制碎片化导致的假 OOM
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$OUTROOT" "$LOGDIR"
# 单实例门禁：已经有训练进程时拒绝启动（防止两路 agent 各自起一格抢同一对卡）
if pgrep -f "scripts/train_joint.py" > /dev/null; then
  echo "[A4][ABORT] 已有 train_joint.py 在跑，拒绝重复启动"; exit 1
fi

for p in "$SM_CKPT" "$MOTION" "$VLM_CKPT" "$VLM_PATH"; do
  [[ -e "$p" ]] || { echo "[fail] missing $p"; exit 1; }
done
[[ -e "$PACKED" ]] || /usr/bin/python3 scripts/build_isaac_motion_packed.py --npz "$MOTION" --out "$PACKED" || exit 1

# 每格的 (名字, wm 参数, horizon, eval 间隔)：eval 间隔按实测单步时间选，
# 目标是每个 40 min 窗口 ~10 个评测点（间隔固定写死，跨格一致的口径是"每 ~3.5 min 一次"）。
run_cell() {
  local name="$1" wm_args="$2" H="$3" EVAL_EVERY="$4"
  local out="$OUTROOT/$name" log="$LOGDIR/$name.log" mem="$LOGDIR/$name.mem"
  echo "=============================================================================="
  echo "[A4] $name  H=$H  batch=$BATCH  minutes=$MINUTES  eval_every=$EVAL_EVERY"
  echo "[A4] wm: $wm_args"
  echo "[A4] out=$out"
  mkdir -p "$out"
  : > "$log"
  # 峰值显存记录（独立轮询）
  ( while true; do
      echo "$(date +%s) $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
      sleep 10
    done > "$mem" ) &
  local poller=$!
  # 硬超时兜底：墙钟 + 15 min（模型加载 / 6 GB vla_final 落盘 / 万一的 NCCL 退出卡顿）。
  # 有它就没有"一挂一整晚"的可能。
  CUDA_VISIBLE_DEVICES=$GPUS timeout $(( MINUTES * 60 + 900 ))s \
    /usr/bin/python3 -m torch.distributed.run \
      --nproc_per_node 2 --master_port "$PID_BASE" \
      scripts/train_joint.py \
        --motion "$MOTION" --motion-packed "$PACKED" \
        --vlm --vlm-path "$VLM_PATH" --vlm-ckpt "$VLM_CKPT" \
        --vlm-dataset --vlm-steps 2 --state-vel-weight 0 \
        --batch-size "$BATCH" --horizon "$H" --steps 1000000 \
        --time-budget-minutes "$MINUTES" \
        --hand-loss-weight 0.2 --hand-grasp-boost 5 \
        $wm_args \
        --eval-every "$EVAL_EVERY" --eval-batches "$EVAL_BATCHES" \
        --eval-out "$out/eval_curve.jsonl" \
        --save --save-every 0 --log-every 20 \
        --seed 0 \
        --out "$out" >> "$log" 2>&1
  local status=$?
  kill "$poller" 2>/dev/null
  local peak
  peak=$(awk '{for(i=2;i<=NF;i+=2) if($(i+1)+0>m) m=$(i+1)+0} END{print m+0}' "$mem" 2>/dev/null)
  echo "[A4] $name done: exit=$status  peak_mem=${peak} MiB  points=$(wc -l < "$out/eval_curve.jsonl" 2>/dev/null || echo 0)"
  PID_BASE=$((PID_BASE + 1))
  if [[ "$status" -ne 0 ]]; then
    echo "[A4][ERROR] $name exited $status — 停止后续格子（先修，别刷噪声）"
    tail -40 "$log"
    return 1
  fi
  return 0
}

WM_FROZEN="--wm-update-every 0"
WM_LIVE="--wm-update-every 50 --wm-update-warmup 100 --wm-update-batch 8 --wm-update-window 8"

# 网格（顺序：先冻结四格，再在线更新两格）
run_cell wm_frozen_H8  "$WM_FROZEN"  8 100  || exit 1
run_cell wm_frozen_H16 "$WM_FROZEN" 16  80  || exit 1
run_cell wm_frozen_H32 "$WM_FROZEN" 32  50  || exit 1
run_cell wm_frozen_H40 "$WM_FROZEN" 40  40  || exit 1
# ⚠️ 2026-10-25：wm_live 两格**暂停**。原因：joint 侧的 WM 更新用的是手算 Δq 参考动作
# （`IsaacJointBundle.sample_windows`），与 policy 的实际动作分布不一致，会把
# `ckpt_010500_physint.pt` 的速度标定洗掉（实测 300 iter: qd_corr 0.90→0.095）。
# 等 `sample_windows` 改成用 `pi(obs, port_token)` + 加参考速度辅助项后再恢复。
if [[ "${SKIP_WM_LIVE:-1}" == "1" ]]; then
  echo "[A4] SKIP_WM_LIVE=1 → 只跑 wm_frozen 四格（wm_live 暂停，见 docs/HANDOVER_MAINLINE_AND_ABLATIONS_20261025.md §4）"
else
  run_cell wm_live_H8    "$WM_LIVE"    8 100  || exit 1
  run_cell wm_live_H40   "$WM_LIVE"   40  40  || exit 1
fi

echo
echo "[A4] all cells finished; summarising"
/usr/bin/python3 scripts/summarize_a4.py --root "$OUTROOT" | tee "$LOGDIR/A4_SUMMARY.md"
