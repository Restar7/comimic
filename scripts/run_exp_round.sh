#!/usr/bin/env bash
# 新线自动整轮（并行；不覆盖旧 B 采集线）
#
# 阶段：aprime（A'） → merge → reward → wm → joint → eval
# 产物全部写 exp_ 前缀：
#   runs/wm_real/exp_round_<ROUND>/...
#   runs/S3_loop/exp_ref_<ROUND>/...
#
# 用法（先 dry-run 看命令）：
#   ROUND=r1 WBC_CKPT=runs/S1_handover/exp_WBC_exec.pt SEEDS="0 1" STEPS=400 \
#   DRY_RUN=1 bash scripts/run_exp_round.sh
#
# 冒烟（只 1 clip × 1 seed × 20 步）：
#   ROUND=smoke WBC_CKPT=runs/S1_handover/exp_WBC_exec.pt SEEDS="0" A_STEPS=20 \
#   MAX_CLIPS=1 STAGES="aprime,merge,reward" bash scripts/run_exp_round.sh
#
# 正式（G1 过、换 two-lib ckpt 后）：
#   ROUND=r1 WBC_CKPT=runs/S1_handover/exp_WBC_exec_cmd.pt SEEDS="0 1 2 3" \
#   STEPS=400 ACCUM=16 STAGES="" bash scripts/run_exp_round.sh
set -uo pipefail
COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1

ROUND=${ROUND:?ROUND required (e.g. r1)}
WBC_CKPT=${WBC_CKPT:-$COMIMIC/runs/S1_handover/exp_WBC_exec.pt}
PACKED=${PACKED:-$COMIMIC/runs/isaac_motion/ha_cmd_exec/packed.pt}
TOKENS_DIR=${TOKENS_DIR:-$COMIMIC/runs/exp_tokens}
SEEDS=${SEEDS:-"0 1"}
A_STEPS=${A_STEPS:-0}                 # 0 = 用 clip 长度
NUM_ENVS=${NUM_ENVS:-8}
MAX_CLIPS=${MAX_CLIPS:-0}             # 0 = 所有有 token npy 的 clip
DELTA_POS=${DELTA_POS:-0.05}
DELTA_ORI=${DELTA_ORI:-0.05}
JOINT_STEPS=${JOINT_STEPS:-400}
ACCUM=${ACCUM:-16}
BATCH=${BATCH:-8}
B_GLOB=${B_GLOB:-}                   # R2+：B 的 dump glob；R1 留空
STAGES=${STAGES:-"aprime,merge,reward,wm,joint,eval"}
[[ -z "$STAGES" ]] && STAGES="aprime,merge,reward,wm,joint,eval"  # 空 = 全跑
DRY_RUN=${DRY_RUN:-0}

WORK=$COMIMIC/runs/wm_real/exp_round_${ROUND}
mkdir -p "$WORK" "$WORK/a_prime" "$WORK/logs"
echo "[round] ROUND=$ROUND WBC=$WBC_CKPT"
echo "[round] PACKED=$PACKED  SEEDS=[$SEEDS] A_STEPS=$A_STEPS NUM_ENVS=$NUM_ENVS"
echo "[round] δ_pos=$DELTA_POS δ_ori=$DELTA_ORI  joint_steps=$JOINT_STEPS ACCUM=$ACCUM"
echo "[round] stages=[$STAGES] DRY_RUN=$DRY_RUN  work=$WORK"

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] $*"
    return 0
  fi
  echo "[run] $*"
  "$@"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[fail] rc=$rc: $*" >&2
    exit $rc
  fi
}

want() { [[ ",$STAGES," == *",$1,"* ]]; }

# ---------------------------------------------------------------- clips
CLIPS_FILE=$WORK/clips.txt
python3 - "$PACKED" "$TOKENS_DIR" "$CLIPS_FILE" "$MAX_CLIPS" <<'PY'
import sys, torch
from pathlib import Path
packed, tokens_dir, out, max_clips = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
p = torch.load(packed, map_location="cpu", weights_only=False)
rows = []
for n, L in zip(p["names"], p["lengths"]):
    n = str(n)
    if (tokens_dir / f"{n}.npy").exists():
        rows.append((n, int(L)))
if max_clips > 0:
    rows = rows[:max_clips]
out.write_text("\n".join(f"{n}\t{L}" for n, L in rows) + ("\n" if rows else ""))
print(f"[round] clips with token npy: {len(rows)}")
PY

# ---------------------------------------------------------------- 1) A'
if want aprime; then
  echo "[stage] aprime"
  while IFS=$'\t' read -r CLIP LEN; do
    [[ -z "${CLIP:-}" ]] && continue
    ST=$A_STEPS
    [[ "$ST" -le 0 ]] && ST=$LEN
    TOK=$TOKENS_DIR/$CLIP.npy
    echo "[aprime] $CLIP len=$LEN steps=$ST"
    run env SEEDS="$SEEDS" STEPS="$ST" NUM_ENVS="$NUM_ENVS" \
      CLIP="$CLIP" TOKEN_NPY="$TOK" WBC="$WBC_CKPT" PACKED="$PACKED" \
      OUT_DIR="$WORK/a_prime/$CLIP" \
      bash scripts/run_a_prime_seeds.sh
  done < "$CLIPS_FILE"
fi

# ---------------------------------------------------------------- 2) merge
A_MERGED=$WORK/a_prime.pt
if want merge; then
  echo "[stage] merge -> $A_MERGED"
  if [[ "$DRY_RUN" != "1" ]]; then
    python3 - "$WORK/a_prime" "$A_MERGED" <<'PY'
import sys, glob, torch
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(glob.glob(str(src / "*" / "merged.pt")))
if not files:
    raise SystemExit(f"[merge] no merged.pt under {src}")
cat = {k: [] for k in ("states", "actions", "next_states", "frames")}
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    for k in cat:
        cat[k].append(d[k])
out = {k: torch.cat(v, 0) for k, v in cat.items()}
out["clips"] = [Path(f).parent.name for f in files]
torch.save(out, str(dst))
print(f"[merge] {len(files)} clips -> {dst} rows={out['states'].shape[0]}")
PY
  fi
fi

# ---------------------------------------------------------------- 3) reward
A_REW=$WORK/a_prime_reward.pt
if want reward; then
  echo "[stage] reward -> $A_REW"
  run env SUPERMIMIC_HA_ALIGN=1 \
    SUPERMIMIC_ROBOT_USD="$COMIMIC/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd" \
    python3 scripts/compute_collect_reward.py \
    --in "$A_MERGED" --out "$A_REW" --wbc-ckpt "$WBC_CKPT" \
    --reward-mode huber_err --reward-delta-pos "$DELTA_POS" --reward-delta-ori "$DELTA_ORI" \
    --packed "$PACKED" --device cuda:0
fi

# ---------------------------------------------------------------- 4) WM
WM_OUT=$COMIMIC/runs/S1_handover/exp_wm_${ROUND}.pt
WM_MIX=$WORK/wm_mix.pt
if want wm; then
  echo "[stage] wm -> $WM_OUT"
  if [[ -n "$B_GLOB" ]]; then
    run python3 scripts/make_wm_mix.py --sim "$A_REW" --bline "$B_GLOB" --ratio 0.25 --out "$WM_MIX"
    WM_IN="$WM_MIX"
  else
    # 2026-11-03：R1 无 B ⇒ 直接用 A'（make_wm_mix 强制 --bline，且 take() 会丢 clips）；
    # 一阶段 6000 步足够；roll 阶段等 R2 有 clips/episodes 再开。
    WM_IN="$A_REW"
  fi
  run env SUPERMIMIC_HA_ALIGN=1 \
    SUPERMIMIC_ROBOT_USD="$COMIMIC/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd" \
    python3 scripts/pretrain_wm.py --joint-ckpt "$WBC_CKPT" \
    --transitions "$WM_IN" --out "$WM_OUT" --steps 6000 --stage1-steps 0 --roll-steps 0 --roll-k 0
fi

# ---------------------------------------------------------------- 5) joint
if want joint; then
  echo "[stage] joint"
  run env \
    ROUND="${ROUND#r}" RUN_LABEL="${ROUND}" STEPS="$JOINT_STEPS" \
    BATCH="$BATCH" ACCUM="$ACCUM" NPROC=1 \
    INIT_VLA="" SM_CKPT="$WBC_CKPT" MOTION_PACKED="$PACKED" WM_CKPT="$WM_OUT" \
    ADV_BASELINE=none CRITIC_ON=1 GAMMA=1.0 \
    REWARD_MODE=huber_err ROOT_ORI_MODE=split_huber \
    REWARD_DELTA_POS="$DELTA_POS" REWARD_DELTA_ORI="$DELTA_ORI" \
    TRUST_WEIGHT=0 ACT_WEIGHT=0.2 EVAL_EVERY="${EVAL_EVERY:-50}" SAVE_EVERY="${SAVE_EVERY:-50}" \
    START_SAMPLER=phase START_WEIGHTS="$COMIMIC/runs/wm_real/exp_start_weights.pt" START_AUDIT=1 \
    bash scripts/run_refstart_joint.sh
fi

# ---------------------------------------------------------------- 6) eval
if want eval; then
  echo "[stage] eval"
  SWEEP=$(ls -dt "$COMIMIC"/runs/S3_loop/sweeps/exp_ref_${ROUND}* 2>/dev/null | head -1 || true)
  if [[ -n "${SWEEP:-}" ]]; then
    run python3 scripts/eval_drift_curve.py --sweep "$SWEEP" --packed "$PACKED" \
      --out "$WORK/drift_${ROUND}.json"
  else
    echo "[eval] 没有找到 sweep: runs/S3_loop/sweeps/exp_ref_${ROUND}*（joint 阶段可能还没跑/没产 sweep）"
  fi
fi

echo "[round] done ROUND=$ROUND  work=$WORK"
