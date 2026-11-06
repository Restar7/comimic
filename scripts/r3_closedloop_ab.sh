#!/usr/bin/env bash
# Round-3 闭环 A/B：同一个官方链路（SONIC release decoder 执行器），只换 action head。
#
#   FT   = checkpoint-8000 + runs/joint_v3_round3A/vla_000250.pt（round-3 arm A，λ_bc=0）
#   BASE = checkpoint-8000 原始（未微调），其余一切相同
#
# 目的：判定 round-3 的 "reward_norm −66% / act/rmse 0.02→0.87" 到底是真学会了，
#       还是把 WM 薅了（动作跑到训练流形外）。判据只看闭环：官方 success / 摔不摔 / ξ。
#
# 用法：TAG=ft HEAD=runs/joint_v3_round3A/vla_000250.pt GPU=2 PORT=18081 bash scripts/r3_closedloop_ab.sh
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
TAG=${TAG:?TAG required}
HEAD=${HEAD:-none}
GPU=${GPU:?GPU required}
PORT=${PORT:?PORT required}
MAX_STEPS=${MAX_STEPS:-600}
TASK=${TASK:-DoubleDesk}
SEED=${SEED:-0}
# ★ 2026-11-05：服务端卡号原来写死 `cuda:1`。本机 GPU1 被**别的容器**占了 43.5/48 GB，
#   2B VLA 服务端放不进去（只剩 ~5.6 GB）⇒ 默认改成**和 Isaac 同卡**（Isaac 约 10 GB +
#   服务端约 6 GB ≪ 48 GB）。要分卡时显式给 SRV_GPU。
SRV_GPU=${SRV_GPU:-$GPU}

OUT=$COMIMIC/runs/closedloop_r3/$TAG
mkdir -p "$OUT"
SLOG=$OUT/server.log
DLOG=$OUT/isaac.log
DUMP=$OUT/dump
rm -rf "$DUMP"

cd "$COMIMIC"

# ---- 1) VLA 服务 -------------------------------------------------------------
HEAD_ARGS=()
if [[ "$HEAD" != "none" ]]; then
  [[ -f "$HEAD" ]] || { echo "[ab] head ckpt 不存在: $HEAD" >&2; exit 2; }
  HEAD_ARGS=(--head-ckpt "$HEAD")
fi

# GR00T 的依赖是 vendored 的（diffusers/tyro/transformers 在 third_party/gr00t_deps），
# 系统 python3 没有 diffusers；**只给 server 这一条进程加**，别污染 Isaac 那条
# （isaac_env.sh 会自己 prepend venv，混进来会串包）。
SRV_PY="$COMIMIC:$COMIMIC/Isaac-GR00T:$COMIMIC/third_party/gr00t_deps:$COMIMIC/third_party/gr00t_deps/_noflash"
# SRV_CMD 覆盖：用来把闭环接到**真值 token 服务**（scripts/ha_gt_token_server.py），
# 以验证"执行器 + 部署接线"本身是对的（机器人应能复现录制里的 1.17 m/12 s 行走）。
if [[ -n "${SRV_CMD:-}" ]]; then
  setsid nohup env PYTHONPATH="$SRV_PY" ${SRV_CMD} > "$SLOG" 2>&1 &
else
  setsid nohup env PYTHONPATH="$SRV_PY" \
    python3 scripts/ha_vla_latent64_server.py \
    --ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
    "${HEAD_ARGS[@]}" \
    --port "$PORT" --device "cuda:$SRV_GPU" --seed "$SEED" \
    ${SRV_EXTRA:-} \
    > "$SLOG" 2>&1 &
fi
SPID=$!
echo "[ab] server pid=$SPID port=$PORT head=${HEAD} isaac_gpu=$GPU srv_gpu=$SRV_GPU seed=$SEED max_steps=$MAX_STEPS"

# 等 "listening on"（2B 底模加载约 60–120 s）
for i in $(seq 1 120); do
  if grep -q "listening on" "$SLOG" 2>/dev/null; then break; fi
  if ! kill -0 "$SPID" 2>/dev/null; then
    echo "[ab] server 启动失败：" >&2; tail -30 "$SLOG" >&2; exit 3
  fi
  sleep 2
done
grep -q "listening on" "$SLOG" || { echo "[ab] server 120s 未就绪" >&2; tail -30 "$SLOG" >&2; kill "$SPID" 2>/dev/null; exit 3; }
echo "[ab] server ready"; grep -E "\[server\]\[head\]|state-extra" "$SLOG" | tail -4

# ---- 2) 官方闭环（SONIC 执行器）----------------------------------------------
export HA_ALLOW_SONIC=1
export HA_GMT_BACKEND=sonic
export HA_ACTION_SOURCE=sonic_wholebody
export SONIC_VLA_ACTION_FORMAT=latent64
export SUPERMIMIC_HA_ALIGN=1
export HA_VLA_EXEC_STEPS=20          # receding horizon = 训练 H=20
export SONIC_DEBUG_DUMP=$DUMP
T0=$SECONDS
TASK=$TASK MAX_STEPS=$MAX_STEPS SEED=$SEED REPEAT=0 ISAAC_GPU=$GPU \
  SERVER_URL=http://127.0.0.1:$PORT OUT_ROOT=$OUT RECORD_VIDEO=0 \
  EXTRA_ARGS="--step_log_every_n 20" \
  bash scripts/ha_isaac_closedloop.sh > "$DLOG" 2>&1
RC=$?
echo "[ab] isaac rc=$RC  elapsed=$((SECONDS-T0))s"

kill "$SPID" 2>/dev/null
sleep 1
kill -9 "$SPID" 2>/dev/null

# ---- 3) 结果 -----------------------------------------------------------------
RES=$(ls -dt "$OUT"/${TASK}__seed${SEED}__repeat0__* 2>/dev/null | head -1)
if [[ -n "$RES" && -f "$RES/result.json" ]]; then
  echo "[ab] result: $RES/result.json"
  python3 - "$RES/result.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for k in ("success","failure_reason","episode_steps","max_steps","duration_sec"):
    print(f"  {k}: {d.get(k)}")
PY
else
  echo "[ab] 没有 result.json（rc=$RC）" >&2; tail -40 "$DLOG" >&2; exit 4
fi

if [[ -f "$DUMP/steps.jsonl" ]]; then
  echo "[ab] dump frames: $(wc -l < "$DUMP/steps.jsonl")"
else
  echo "[ab] WARN 没有 steps.jsonl" >&2
fi
