#!/usr/bin/env bash
# 8 卡并行闭环评测：把 SEED_TOTAL 集拆成 N_SWEEPS 条独立 sweep 同时跑（每条 2 张卡），
# 每条内部仍然「同 server seed、同 reset seed、main/ctrl 成对」，最后用
# scripts/merge_eval_sweeps.py 合并逐集配对差。
#
# 为什么要拆：单条 sweep 只吃 2 张卡、12 集 ⇒ 配对 95%CI ≈ ±1.3 cm；8 卡拆 4 条
# 可以 32 集在 ~15 min 内跑完，CI 收到 ~±0.8 cm，且不改变任何单条 sweep 的配对语义。
#
# 用法（默认 d2b → h20_8g，32 集）：
#   HEAD_CTRL=$PWD/runs/S3_loop/h20_8g_joint/train_main/vla_final.pt \
#   TAG_PREFIX=d2b_vs_h20_8g bash scripts/eval_h20_8gpu.sh
#
#   HEAD_CTRL=... TAG_PREFIX=d2b_vs_h20_lamr012 SEED_TOTAL=32 N_SWEEPS=4 \
#     bash scripts/eval_h20_8gpu.sh
set -uo pipefail

COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1

HEAD_MAIN=${HEAD_MAIN:-$COMIMIC/runs/S3_loop/d2b/train_main/vla_final.pt}
HEAD_CTRL=${HEAD_CTRL:?必须给 HEAD_CTRL}
LABEL_MAIN=${LABEL_MAIN:-d2b}
LABEL_CTRL=${LABEL_CTRL:-$(basename "$(dirname "$(dirname "$HEAD_CTRL")")")}
SM_CKPT=${SM_CKPT:-$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt}
SM_CKPT_MAIN=${SM_CKPT_MAIN:-$SM_CKPT}
SM_CKPT_CTRL=${SM_CKPT_CTRL:-$SM_CKPT}
TAG_PREFIX=${TAG_PREFIX:?必须给 TAG_PREFIX}
SEED_TOTAL=${SEED_TOTAL:-32}
N_SWEEPS=${N_SWEEPS:-4}
BASE_PORT=${BASE_PORT:-18150}
GPU_OFFSET=${GPU_OFFSET:-0}   # 两条臂同时评测时，第二条用 4（占 4,5,6,7 卡）
START_FRAME=${START_FRAME:-0}
MAX_STEPS=${MAX_STEPS:-300}
EXTRA=${EXTRA:---sample-avg 4}
LOGD=$COMIMIC/logs/eval_${TAG_PREFIX}
mkdir -p "$LOGD"

[[ "$HEAD_MAIN" == "base" || -f "$HEAD_MAIN" ]] || { echo "[fail] 缺 HEAD_MAIN=$HEAD_MAIN"; exit 1; }
[[ "$HEAD_CTRL" == "base" || -f "$HEAD_CTRL" ]] || { echo "[fail] 缺 HEAD_CTRL=$HEAD_CTRL"; exit 1; }
(( N_SWEEPS >= 1 && N_SWEEPS <= 4 )) || { echo "[fail] N_SWEEPS 只允许 1..4（每条 2 张卡）"; exit 1; }
(( SEED_TOTAL % N_SWEEPS == 0 )) || { echo "[fail] SEED_TOTAL 必须能被 N_SWEEPS 整除"; exit 1; }

PER=$(( SEED_TOTAL / N_SWEEPS ))
echo "[eval8] head_main=$HEAD_MAIN"
echo "[eval8] head_ctrl=$HEAD_CTRL"
echo "[eval8] ${N_SWEEPS} 条 sweep 并行，每条 ${PER} 集，共 ${SEED_TOTAL} 集；每集 ${MAX_STEPS} 步"

pids=()
tags=()
for ((i = 0; i < N_SWEEPS; i++)); do
  lo=$(( i * PER ))
  hi=$(( lo + PER - 1 ))
  seeds=$(seq -s ' ' "$lo" "$hi")
  gpu_main=$(( GPU_OFFSET + i * 2 ))
  gpu_ctrl=$(( GPU_OFFSET + i * 2 + 1 ))
  port_main=$(( BASE_PORT + i * 2 ))
  port_ctrl=$(( port_main + 1 ))
  tag="${TAG_PREFIX}_s${i}"
  tags+=("$tag")
  echo "[eval8] sweep#$i tag=$tag seeds=[$lo..$hi] gpu=$gpu_main/$gpu_ctrl port=$port_main/$port_ctrl"
  TAG="$tag" SEEDS="$seeds" START_FRAME="$START_FRAME" MAX_STEPS="$MAX_STEPS" \
    HEAD_MAIN="$HEAD_MAIN" HEAD_CTRL="$HEAD_CTRL" \
    SM_CKPT_MAIN="$SM_CKPT_MAIN" SM_CKPT_CTRL="$SM_CKPT_CTRL" \
    LABEL_MAIN="$LABEL_MAIN" LABEL_CTRL="$LABEL_CTRL" \
    EXTRA_MAIN="$EXTRA" EXTRA_CTRL="$EXTRA" \
    PORT_MAIN="$port_main" PORT_CTRL="$port_ctrl" \
    GPU_MAIN="$gpu_main" GPU_CTRL="$gpu_ctrl" SRV_SEED=0 \
    bash "$COMIMIC/scripts/s3_eval_sweep.sh" > "$LOGD/${tag}.log" 2>&1 &
  pids+=($!)
done

rc_all=0
for idx in "${!pids[@]}"; do
  wait "${pids[$idx]}"; rc=$?
  echo "[eval8] sweep#${idx} (${tags[$idx]}) rc=$rc"
  [[ $rc == 0 ]] || rc_all=1
done

outs=()
for tag in "${tags[@]}"; do
  [[ -f "$COMIMIC/runs/S3_loop/sweeps/$tag/summary.json" ]] && outs+=("$COMIMIC/runs/S3_loop/sweeps/$tag")
done
if (( ${#outs[@]} == 0 )); then
  echo "[eval8][fail] 没有任何 sweep 产出 summary.json（看 $LOGD/*.log）"
  exit 1
fi
echo "[eval8] 合并 ${#outs[@]}/${N_SWEEPS} 条 sweep…"
PYTHONPATH=. python3 "$COMIMIC/scripts/merge_eval_sweeps.py" "${outs[@]}" \
  --label "${LABEL_CTRL}_minus_${LABEL_MAIN}" \
  --out "$COMIMIC/runs/S3_loop/sweeps/${TAG_PREFIX}_merged.json"
exit $rc_all
