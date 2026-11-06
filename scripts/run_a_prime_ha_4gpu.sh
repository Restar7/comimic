#!/usr/bin/env bash
# 全量 A'（**HA 任务场景**，4×4090）：官方闭环 + L1 动作噪声 + L2 起点扰动 + 状态 dump。
#
# 为什么改到 HA 场景：裸地面场景（isaac_a_collect_batch.py 的自建场景）既没有 HA 的
# 接触材质，也没有桌子/物体，回放出来的东西和部署不是同一个系统。官方闭环
# （sim_eval_vla.py + HumanoidArena 任务 cfg）才是唯一口径 —— 那里的 physics_material
# 就是 static/dynamic=1.0 + combine=max。
#
# 产物：runs/exp_g1_<TAG>/dump/<TASK>__clip<i>_<arm>_ep000.pt
#       每个 dump = {states(71), actions(29), frames, q_target_applied, tokens}
#
# 用法：
#   SEEDS="0" JOBS=3 bash scripts/run_a_prime_ha_4gpu.sh
set -uo pipefail
C=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$C" || exit 1

SEEDS=${SEEDS:-"0"}
JOBS=${JOBS:-3}                 # 每卡并发 Isaac 进程（每个进程一条 clip；启动 ~90 s 是主要成本）
WBC_CKPT=${WBC_CKPT:-$C/runs/S1_handover/exp_WBC_exec_haonly_best.pt}
PACKED=${PACKED:-$C/runs/isaac_motion/ha_cmd_exec/packed.pt}
TOKEN_LIB=${TOKEN_LIB:-$C/runs/isaac_motion/ha_cmd/packed.pt}
NOISE_STD=${NOISE_STD:-0.05}
NOISE_HOLD=${NOISE_HOLD:-20}
TAGBASE=${TAGBASE:-haprime}
CLIP_LIMIT=${CLIP_LIMIT:-0}     # >0 = 只跑前 N 条（调试）

# ---- 选出 6 个 HA 任务族里、有 token npy 的 clip ----
mapfile -t CLIPS < <(python3 - "$PACKED" "$C/runs/exp_tokens" "$CLIP_LIMIT" <<'PY'
import sys, torch
from pathlib import Path
packed, tokdir, lim = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
d = torch.load(packed, map_location="cpu", weights_only=False)
keep = ("HOI_double_desk", "HOI_pp_box", "HSI_boxing", "HSI_open_door", "HSI_sit_sofa", "HSI_vision_navi")
idx = [i for i, n in enumerate(d["names"])
       if n.startswith(keep) and (tokdir / f"{n}.npy").exists()]
if lim > 0:
    idx = idx[:lim]
print("\n".join(str(i) for i in idx))
PY
)
N=${#CLIPS[@]}
echo "[haprime] clip 总数 $N  seeds=[$SEEDS] jobs/card=$JOBS noise=$NOISE_STD hold=$NOISE_HOLD"
[[ "$N" -gt 0 ]] || { echo "[haprime] 没选到 clip"; exit 1; }

NSHARD=$((4 * JOBS))
run_one() {  # gpu seed shard_idx
  local gpu=$1 seed=$2 sh=$3
  local list="" i
  for ((i = sh; i < N; i += NSHARD)); do list+="${CLIPS[i]},"; done
  list=${list%,}
  local tag="${TAGBASE}_s${seed}_sh${sh}"
  echo "[haprime] GPU$gpu seed=$seed shard$sh -> $(echo "$list" | tr ',' '\n' | wc -l) clip"
  SM_ACTION_NOISE_STD="$NOISE_STD" SM_NOISE_HOLD="$NOISE_HOLD" \
  SM_RESET_SEED="$seed" SM_RESET_SCALE=1.0 \
  HA_PIN_BYPASS=1 \
  SM_CKPT="$WBC_CKPT" SM_PACKED="$PACKED" TOKEN_LIB="$TOKEN_LIB" \
  CLIPS="$list" ARMS="online" ISAAC_GPU="$gpu" TAG="$tag" \
    bash scripts/p1c_g1_replay.sh > "logs/${tag}.log" 2>&1
  echo "[haprime] GPU$gpu seed=$seed shard$sh done rc=$?"
}

echo "[haprime] 并发 = $NSHARD（4 卡 × $JOBS）"
pids=()
for seed in $SEEDS; do
  for ((sh = 0; sh < NSHARD; sh++)); do
    run_one "$((sh % 4))" "$seed" "$sh" &
    pids+=($!)
  done
done
wait
echo "[haprime] 全部完成"
