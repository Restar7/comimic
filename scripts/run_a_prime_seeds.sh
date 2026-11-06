#!/usr/bin/env bash
# A' 多 seed 生成：同一条 clip、同一组 L1/L2 噪声参数，跑 N 个 seed 并合并（2026-11-02）
#
# 为什么不是 1 个 seed：单 seed 只覆盖一组扰动；A' 要给 WM 覆盖"参考附近的动力学"，
# 至少 4 组独立噪声（G3a 的 1→2→4 曲线再决定是否加到 8）。
#
# 用法：
#   TASK=DoubleDesk CLIP=HOI_double_desk__tw__ep1776268633456387 \
#   TOKEN_NPY=runs/exp_tokens/HOI_double_desk__tw__ep1776268633456387.npy \
#   WBC=runs/S1_handover/exp_WBC_exec.pt SEEDS="0 1 2 3" STEPS=2080 \
#   bash scripts/run_a_prime_seeds.sh
set -uo pipefail
COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1

CLIP=${CLIP:?CLIP required}
TOKEN_NPY=${TOKEN_NPY:?TOKEN_NPY required}
WBC=${WBC:-$COMIMIC/runs/S1_handover/exp_WBC_exec_cmd.pt}
PACKED=${PACKED:-$COMIMIC/runs/isaac_motion/ha_cmd_exec/packed.pt}
SEEDS=${SEEDS:-"0 1 2 3"}
STEPS=${STEPS:-2080}
START_FRAME=${START_FRAME:-0}
NUM_ENVS=${NUM_ENVS:-8}
COLLECT_TIMEOUT=${COLLECT_TIMEOUT:-3600}
OUT_DIR=${OUT_DIR:-$COMIMIC/runs/wm_real/exp_a_prime/$CLIP}
# 2026-11-04：噪声改成**逐关节（弧度）**口径 + hold 抖动。
#   NOISE_RAD=0.014 ≈ 旧口径的中位等效（0.05 × action_scale 中位 0.275）。
#   NOISE_RAD>0 时 NOISE_STD 被忽略。
NOISE_RAD=${NOISE_RAD:-0.014}
NOISE_STD=${NOISE_STD:-0.05}
HOLD=${HOLD:-20}
HOLD_JITTER=${HOLD_JITTER:-5}
SP_XY=${SP_XY:-0.10}
SP_YAW=${SP_YAW:-0.10}
SP_DQ=${SP_DQ:-0.02}

mkdir -p "$OUT_DIR"
echo "[a'] clip=$CLIP seeds=[$SEEDS] steps=$STEPS start=$START_FRAME"
echo "[a'] wbc=$WBC token=$TOKEN_NPY packed=$PACKED"
echo "[a'] L1 rad=$NOISE_RAD (std=$NOISE_STD 仅当 rad=0) hold=$HOLD±$HOLD_JITTER | L2 xy=$SP_XY yaw=$SP_YAW dq=$SP_DQ"
echo "[a'] out=$OUT_DIR"

n_ok=0; n_bad=0
for s in $SEEDS; do
  log="$OUT_DIR/seed${s}.log"
  echo "[a'] $(date +%T) seed=$s"
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    SUPERMIMIC_HA_ALIGN=1 \
    SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$COMIMIC/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd} \
    timeout "$COLLECT_TIMEOUT" python3 scripts/isaac_collect_perturbed.py \
      --clip "$CLIP" --token-npy "$TOKEN_NPY" --wbc-ckpt "$WBC" --packed "$PACKED" \
      --num-envs "$NUM_ENVS" --steps "$STEPS" --start-frame "$START_FRAME" \
      --seed "$s" --action-noise-rad "$NOISE_RAD" --action-noise-std "$NOISE_STD" \
      --perturb-hold "$HOLD" --perturb-hold-jitter "$HOLD_JITTER" \
      --start-perturb-xy "$SP_XY" --start-perturb-yaw "$SP_YAW" --start-perturb-dq "$SP_DQ" \
      --out "$OUT_DIR/seed${s}.pt" > "$log" 2>&1
  rc=$?
  # 以**输出文件**为准：Isaac 退出偶发挂起（rc≠0）但数据已落盘时仍算成功
  if [[ -s "$OUT_DIR/seed${s}.pt" ]]; then
    n_ok=$((n_ok+1)); echo "[a'] seed=$s OK (rc=$rc, 以输出文件为准)"
  else
    n_bad=$((n_bad+1)); echo "[a'] seed=$s FAIL rc=$rc (log $log)"; tail -3 "$log" | sed "s/^/[a']   /"
  fi
done

# 合并（只合并成功且在 OUT_DIR 下的 seed*.pt）
python3 - "$OUT_DIR" <<'PY'
import sys, glob, torch
from pathlib import Path
out_dir = Path(sys.argv[1])
files = sorted(glob.glob(str(out_dir / "seed*.pt")))
if not files:
    raise SystemExit("[a'] no seed file to merge")
cat = {k: [] for k in ("states", "actions", "next_states", "frames")}
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    for k in cat:
        cat[k].append(d[k])
merged = {k: torch.cat(v, dim=0) for k, v in cat.items()}
merged["num_seeds"] = len(files)
merged["seeds"] = [Path(f).stem for f in files]
merged["clip"] = torch.load(files[0], map_location="cpu", weights_only=False).get("clip")
merged["meta"] = {"source": "run_a_prime_seeds.sh", "seeds": len(files)}
dst = out_dir / "merged.pt"
torch.save(merged, str(dst))
print(f"[a'] merged {len(files)} seeds -> {dst} rows={merged['states'].shape[0]}")
PY

echo "[a'] done ok=$n_ok bad=$n_bad"
[[ $n_bad -eq 0 ]]
