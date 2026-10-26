#!/usr/bin/env bash
# S3.7：mini 闭环（按闭环挑 ckpt）。
#
# 把一个训练产物 `vla_XXXXXX.pt` 合并回 ckpt 目录，起 VLA 服务，跑
# PPBox + DoubleDesk 各 1 seed × MAX_STEPS 步，出 result.json / SUMMARY.md。
#
# 用法：
#   VLA=runs/joint_v2_lamr030/vla_001000.pt bash scripts/mini_closedloop_v2.sh
#   VLA=... WITH_BASE=1 bash scripts/mini_closedloop_v2.sh   # 顺带跑起始 ckpt 作参照
#   VLA=... TASKS="PPBox" SEEDS="0 1" MAX_STEPS=300 bash scripts/mini_closedloop_v2.sh
#
# 关键环境变量：
#   BASE        起始 VLA ckpt（默认官方 checkpoint-8000；必须与训练时同一份）
#   K           fm_steps（必须等于训练时的 --fm-steps；服务端会校验 vla_meta.json）
#   DECODE      codec（v2 产物必须 codec）/ none（v1 旧产物）
#   VLA_GPU     服务端 GPU（默认 1）；ISAAC_GPU  Isaac 的 GPU（默认 0）
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
ROOT=$COMIMIC
VLA=${VLA:?需要 VLA=runs/<run>/vla_XXXXXX.pt}
BASE=${BASE:-$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}
WBC_CKPT=${WBC_CKPT:-$COMIMIC/trackerlab-supermimic/logs/wbc_mix2_obs93/ckpt_016000_spec.pt}
K=${K:-4}
DECODE=${DECODE:-codec}
TASKS=${TASKS:-"PPBox DoubleDesk"}
SEEDS=${SEEDS:-"0"}
MAX_STEPS=${MAX_STEPS:-300}
VLA_GPU=${VLA_GPU:-1}
ISAAC_GPU=${ISAAC_GPU:-0}
PORT=${PORT:-18085}
WITH_BASE=${WITH_BASE:-0}

[[ -f "$VLA" ]] || { echo "[fail] missing VLA $VLA"; exit 1; }
[[ -d "$BASE" ]] || { echo "[fail] missing BASE $BASE"; exit 1; }
[[ -e "$WBC_CKPT" ]] || { echo "[fail] missing WBC $WBC_CKPT"; exit 1; }

TAG=$(basename "$(dirname "$VLA")")__$(basename "$VLA" .pt)
MERGED=${MERGED:-$COMIMIC/runs/ckpt_$TAG}
OUT=${OUT:-$COMIMIC/runs/mini_closedloop/$TAG}
mkdir -p "$OUT"

echo "[mini] vla=$VLA"
echo "[mini] base=$BASE"
echo "[mini] merged=$MERGED  out=$OUT  K=$K decode=$DECODE"

python3 "$COMIMIC/scripts/export_vla_ckpt.py" --base "$BASE" --vla "$VLA" \
  --out "$MERGED" --force || { echo "[fail] 合并失败"; exit 2; }

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

export PYTHONPATH=$ROOT/third_party/gr00t_deps/_noflash
export NO_ALBUMENTATIONS_UPDATE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
python3 "$ROOT/scripts/ha_vla_latent64_server.py" \
  --ckpt "$MERGED" --port "$PORT" --device "cuda:$VLA_GPU" --bf16 \
  --horizon 40 --fm-steps "$K" --decode "$DECODE" \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
ok=0
for _ in $(seq 1 150); do
  if curl -sf --max-time 3 "http://127.0.0.1:$PORT/" >/dev/null; then ok=1; break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[fail] server 挂了，见 $OUT/server.log"; exit 3; }
  sleep 2
done
[[ "$ok" == "1" ]] || { echo "[fail] server 300s 未就绪，见 $OUT/server.log"; exit 3; }
echo "[mini] server ready on :$PORT"

VARIANTS="v2=http://127.0.0.1:$PORT"
if [[ "$WITH_BASE" == "1" ]]; then
  python3 "$ROOT/scripts/ha_vla_latent64_server.py" \
    --ckpt "$BASE" --port "$((PORT + 1))" --device "cuda:$VLA_GPU" --bf16 \
    --horizon 40 --fm-steps "$K" --decode "$DECODE" \
    > "$OUT/server_base.log" 2>&1 &
  BASE_PID=$!
  for _ in $(seq 1 150); do
    curl -sf --max-time 3 "http://127.0.0.1:$((PORT + 1))/" >/dev/null && break
    sleep 2
  done
  VARIANTS="base=http://127.0.0.1:$((PORT + 1)),$VARIANTS"
  echo "[mini] base server ready on :$((PORT + 1)) (pid $BASE_PID)"
fi

SM_CKPT="$WBC_CKPT" ISAAC_GPU="$ISAAC_GPU" VARIANTS="$VARIANTS" \
TASKS="$TASKS" SEEDS="$SEEDS" REPEATS=1 MAX_STEPS="$MAX_STEPS" OUT="$OUT/isaac" \
  bash "$ROOT/scripts/ha_isaac_batch.sh" || echo "[warn] batch 非零返回（逐集结果仍会汇总）"

echo "[mini] 结果: $OUT/isaac/SUMMARY.md"
cat "$OUT/isaac/SUMMARY.md" 2>/dev/null | head -30
python3 - "$OUT" "$VLA" "$BASE" "$WBC_CKPT" "$K" "$DECODE" <<'PY'
import json, sys, pathlib
out, vla, base, wbc, k, decode = sys.argv[1:7]
out = pathlib.Path(out)
summ = out / "isaac" / "summary.json"
d = json.loads(summ.read_text()) if summ.exists() else {}
(out / "pick.json").write_text(json.dumps({
    "vla": str(vla), "base_ckpt": str(base), "wbc_ckpt": str(wbc),
    "fm_steps": int(k), "decode": decode,
    "per_task": d.get("per_task", []), "per_variant": d.get("per_variant", {}),
}, indent=2))
print(f"[mini] pick.json -> {out / 'pick.json'}")
PY
