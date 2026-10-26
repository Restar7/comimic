#!/usr/bin/env bash
# oracle-token 三臂闭环编排器（GT / ORACLE / VLA）。
#
# 一个任务族 + 一条 clip → 起三个 token 服务（同协议、同端口族）→ 跑 ha_isaac_batch.sh → 出汇总。
# 判读见 docs/HANDOVER_ORACLE_TOKEN_CLOSEDLOOP_4GPU_20261026.md §4。
#
# 前置：
#   * 本机 Isaac 可用（Isaac Sim 5.0 + IsaacLab 2.2，scripts/isaac_env.sh 能 import 三件套）
#   * runs/oracle_tokens/<clip>.npy 已由 scripts/build_oracle_tokens.py 产出
#   * VLA 权重 runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000
#
# 用法：
#   TASK=DoubleDesk CLIP=HOI_double_desk__tw__ep000000 SEEDS="0 1 2" \
#     bash scripts/launch_oracle_3arm.sh
#
#   # 先核对 --consume（交接文档 §3 步骤 0）：两档各跑一次短程，看 [replay] infer 行
#   TASK=DoubleDesk CLIP=HOI_double_desk__tw__ep000000 SEEDS="0" CONSUME=40 \
#     MAX_STEPS=20 MODE=probe bash scripts/launch_oracle_3arm.sh
#
# 关键变量：
#   TASK        DoubleDesk | PPBox | Boxing | OpenDoor | SitSofa | VisionNavi
#   CLIP        npz stem（GT/ORACLE 臂都从这条 clip 出发，手部三臂必须一致）
#   SEEDS       "0 1 2"
#   REPEATS     默认 1
#   MAX_STEPS   默认 300
#   CONSUME     默认 12（provider 每 12 步重取；若实测该用 40，改成 40）
#   ISAAC_GPU   默认 0
#   OUT         默认 runs/oracle_3arm_<stamp>
#   SKIP_VLA=1  只跑 GT vs ORACLE（省掉 VLA 服务与 6.5GB 权重加载）
set -uo pipefail

COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1

TASK=${TASK:-DoubleDesk}
CLIP=${CLIP:-HOI_double_desk__tw__ep000000}
SEEDS=${SEEDS:-"0 1 2"}
REPEATS=${REPEATS:-1}
MAX_STEPS=${MAX_STEPS:-300}
CONSUME=${CONSUME:-12}
ISAAC_GPU=${ISAAC_GPU:-0}
MODE=${MODE:-full}
SKIP_VLA=${SKIP_VLA:-0}
VLM_CKPT=${VLM_CKPT:-$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}
NPZ=data/npz/ha_tokens/${CLIP}.npz
ORACLE=runs/oracle_tokens/${CLIP}.npy
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-$COMIMIC/runs/oracle_3arm_${TASK}_${STAMP}}
P_GT=${P_GT:-18081}; P_OR=${P_OR:-18082}; P_VLA=${P_VLA:-18080}

mkdir -p "$OUT" logs
echo "[3arm] task=$TASK clip=$CLIP seeds=[$SEEDS] repeats=$REPEATS steps=$MAX_STEPS consume=$CONSUME mode=$MODE"

[[ -f $NPZ ]] || { echo "[3arm][fail] 缺 $NPZ" >&2; exit 1; }
if [[ $MODE == full ]]; then
  [[ -f $ORACLE ]] || { echo "[3arm][fail] 缺 $ORACLE（先跑 build_oracle_tokens.py）" >&2; exit 1; }
fi

SRV_PIDS=()
cleanup() {  # 只杀本脚本起的服务：记 PID，别用 pkill -f（会误伤同名调用者）
  for pid in "${SRV_PIDS[@]:-}"; do
    [[ -n "${pid:-}" ]] && kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

wait_ok() {  # $1=port $2=name
  for _ in $(seq 1 90); do
    if curl -sf --max-time 2 "http://127.0.0.1:$1/" >/dev/null; then echo "[3arm] $2 up on :$1"; return 0; fi
    sleep 1
  done
  echo "[3arm][fail] $2 没起来 (: $1)" >&2; return 1
}

# ---- 臂 1：GT（录制 token + 录制手）-----------------------------------------
python3 scripts/ha_replay_token_server.py --npz-tokens "$NPZ" --key g1_motion_token_64 \
  --hands-npz "$NPZ" --port "$P_GT" --consume "$CONSUME" > "logs/3arm_gt_${STAMP}.log" 2>&1 &
SRV_PIDS+=($!)
wait_ok "$P_GT" gt || exit 1

VARIANTS="gt=http://127.0.0.1:$P_GT"

# ---- 臂 2：ORACLE（优化后 token + **同一套录制手**）-------------------------
if [[ $MODE == full ]]; then
  python3 scripts/ha_replay_token_server.py --tokens "$ORACLE" \
    --hands-npz "$NPZ" --port "$P_OR" --consume "$CONSUME" > "logs/3arm_oracle_${STAMP}.log" 2>&1 &
  SRV_PIDS+=($!)
  wait_ok "$P_OR" oracle || exit 1
  VARIANTS="$VARIANTS,oracle=http://127.0.0.1:$P_OR"
fi

# ---- 臂 3：VLA（官方 ckpt，decode codec，手部来自模型输出）--------------------
if [[ $SKIP_VLA != 1 && $MODE == full ]]; then
  export PYTHONPATH="$COMIMIC:$COMIMIC/Isaac-GR00T:$COMIMIC/third_party/gr00t_deps:$COMIMIC/third_party/gr00t_deps/_noflash${PYTHONPATH:+:$PYTHONPATH}"
  export NO_ALBUMENTATIONS_UPDATE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 SUPERMIMIC_HA_ALIGN=1
  python3 scripts/ha_vla_latent64_server.py --ckpt "$VLM_CKPT" --port "$P_VLA" \
    --bf16 --fm-steps 4 --decode codec > "logs/3arm_vla_${STAMP}.log" 2>&1 &
  SRV_PIDS+=($!)
  wait_ok "$P_VLA" vla || echo "[3arm][warn] VLA 服务没起来，只跑 gt/oracle"
  curl -sf --max-time 2 "http://127.0.0.1:$P_VLA/" >/dev/null && \
    VARIANTS="$VARIANTS,vla=http://127.0.0.1:$P_VLA"
fi

echo "[3arm] VARIANTS=$VARIANTS"
echo "[3arm] 若 GT 服务日志里 [replay] infer 的 rows 增长快于仿真步，把 CONSUME 从 $CONSUME 改成反比档位"

# ---- 跑批量闭环 --------------------------------------------------------------
TASKS="$TASK" SEEDS="$SEEDS" REPEATS="$REPEATS" MAX_STEPS="$MAX_STEPS" \
  VARIANTS="$VARIANTS" ISAAC_GPU="$ISAAC_GPU" OUT="$OUT" \
  bash scripts/ha_isaac_batch.sh 2>&1 | tee "$OUT/batch.log"
rc=${PIPESTATUS[0]}

echo "[3arm] done rc=$rc out=$OUT"
[[ -f $OUT/SUMMARY.md ]] && sed -n '1,60p' "$OUT/SUMMARY.md"

# 0 集 = Isaac 没跑起来（本机 5.0 栈在 595 驱动上是段错误），别把空表当"任务失败"
EP=$(python3 - "$OUT" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "summary.json"
try:
    print(len(json.loads(p.read_text()).get("rows", [])))
except Exception:
    print(0)
PY
)
if [[ "$EP" == "0" ]]; then
  echo "[3arm][fail] 一个 episode 都没跑出来 —— Isaac 大概率没起来（不是任务失败）。" >&2
  echo "[3arm][fail] 先单独验证：bash scripts/isaac_env.sh check 三项 OK，且最小 SimulationApp 能起；日志见 $OUT/runs/*.log" >&2
  rc=1
fi
echo "[3arm] 回传：$OUT/summary.json  $OUT/SUMMARY.md  $OUT/runs/*/result.json  以及 logs/3arm_*.log 里的 [replay] infer 行"
exit $rc
exit $rc
