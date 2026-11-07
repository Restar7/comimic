#!/usr/bin/env bash
# 成功率实验的**一个格**（task × head）：server 起一次（跨 seed 复用），每个 seed 起一个
# 独立 Isaac 进程（= 官方单集路径，最稳）。
#
# 为什么不用 --episode_batch_json：实测第 2 集会在 warm-start 后抛
#   `int() argument must be ... not 'NoneType'` → failure_reason=sim_error，
#   每格只有第 1 集有效。改成一集一进程 + server 预热，代价只是每集 ~60 s Isaac 启动。
#
# 环境变量：
#   TASK=VisionNavi|SitSofa          （ha_isaac_closedloop.sh 的别名，自带 yaml+instruction）
#   HEAD=runs/joint_v13/vla_final.pt 或 none（= 未微调底模）
#   GPU=0|1  PORT=18xxx  CELLDIR=...  SEEDS="0 1 2"  MAX_STEPS=3000  LABEL=...
set -uo pipefail

# ★ 2026-11-07（会话 01）：
#   * 现状（实测）：导出 `HA_ALLOW_SONIC` 的是**调用方**队列脚本
#     （`r3_sr_queue.sh:29-31` 等 13 个），正式 eval 一直走 SONIC
#     （`runs/closedloop_r3` 88/88 日志是 [SonicActionProvider]）。
#     **但本脚本自己不导出** ⇒ 单独调用 + 干净环境会静默落到 SuperMimic
#     （`runs/closedloop_sr/sr1_WRONG_ACTION_SOURCE_supermimic/` 就是那次的负对照）。
#   * `SONIC=1` ⇒ 由本脚本显式导出三件套，不再依赖调用方。
#   * head ckpt 里有 delta_token_state ⇒ Δtoken 已启用 ⇒ 必须 SONIC（除非显式
#     ALLOW_DELTA_WITH_SUPERMIMIC=1 做对照），否则 audit_delta_required 会非零退出。
SONIC=${SONIC:-0}
if [[ "$SONIC" == "1" ]]; then
  export HA_ALLOW_SONIC=1
  export HA_GMT_BACKEND=${HA_GMT_BACKEND:-sonic}
  export HA_ACTION_SOURCE=${HA_ACTION_SOURCE:-sonic_wholebody}
fi
EXPECT_PROVIDER=${EXPECT_PROVIDER:-$([[ "$SONIC" == "1" ]] && echo sonic || echo any)}

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
TASK=${TASK:?}; HEAD=${HEAD:-none}
GPU=${GPU:?}; PORT=${PORT:?}; CELLDIR=${CELLDIR:?}
SEEDS=${SEEDS:?}; MAX_STEPS=${MAX_STEPS:-3000}; LABEL=${LABEL:-$TASK}
CKPT=${CKPT:-runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}

mkdir -p "$CELLDIR"
LOG=$CELLDIR/cell.log
cd "$COMIMIC" || exit 1
source "$COMIMIC/scripts/_deploy_audit.sh"
audit_init "${AUDIT_DIR:-$CELLDIR/audit}"
audit_kv "script" "r3_sr_cell.sh"
audit_kv "task" "$TASK"; audit_kv "head" "$HEAD"; audit_kv "gpu" "$GPU"; audit_kv "port" "$PORT"
audit_kv "sonic_requested" "$SONIC"
audit_kv "ha_allow_sonic" "${HA_ALLOW_SONIC:-0}"
audit_kv "expected_provider" "$EXPECT_PROVIDER"
audit_file "vla_ckpt_dir" "$CKPT"
audit_file "head_ckpt" "$HEAD"
audit_file "sonic_decoder" "${SONIC_DIR:-$COMIMIC/../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release}/model_decoder.onnx"
audit_file "ref_lib" "${SM_TOKEN_LIB:-$COMIMIC/runs/isaac_motion/ha_cmd/packed.pt}"
audit_python python3
audit_delta_required "$HEAD" || { echo "[cell][FATAL] Δtoken 与执行器不匹配（见上）" | tee -a "$LOG"; exit 2; }
echo "[cell] start $(date +%F' '%T) task=$TASK head=$HEAD gpu=$GPU port=$PORT seeds='$SEEDS' max_steps=$MAX_STEPS" | tee "$LOG"

# ---- server（head==none 时就是未微调底模），整格复用 -------------------------
HEAD_ARGS=()
if [[ "$HEAD" != "none" ]]; then
  [[ -f "$HEAD" ]] || { echo "[cell] head 不存在: $HEAD" >&2; exit 2; }
  HEAD_ARGS=(--head-ckpt "$HEAD")
fi
SRV_PY="$COMIMIC:$COMIMIC/Isaac-GR00T:$COMIMIC/third_party/py312_shims:$COMIMIC/third_party/gr00t_deps:$COMIMIC/third_party/gr00t_deps/_noflash"
setsid nohup env PYTHONPATH="$SRV_PY" \
  python3 scripts/ha_vla_latent64_server.py \
  --ckpt "$CKPT" "${HEAD_ARGS[@]}" \
  --port "$PORT" --device "cuda:$GPU" --seed 0 \
  > "$CELLDIR/server.log" 2>&1 &
SPID=$!
for _ in $(seq 1 150); do
  grep -q "listening on" "$CELLDIR/server.log" 2>/dev/null && break
  kill -0 "$SPID" 2>/dev/null || { echo "[cell] server 崩了" | tee -a "$LOG"; tail -15 "$CELLDIR/server.log" >&2; exit 3; }
  sleep 2
done
grep -q "listening on" "$CELLDIR/server.log" 2>/dev/null || { echo "[cell] server 超时" | tee -a "$LOG"; kill "$SPID" 2>/dev/null; exit 3; }
echo "[cell] server ready" | tee -a "$LOG"
_DT=$(grep -m1 "\[server\]\[delta-token\] loaded" "$CELLDIR/server.log" 2>/dev/null || true)
if [[ -n "$_DT" ]]; then
  echo "[cell][delta-token] ${_DT#*] }" | tee -a "$LOG"
  audit_kv "delta_token_enabled" "1"
  audit_kv "delta_token_detail" "$_DT"
elif grep -q "\[server\]\[delta-token\] ckpt 里没有" "$CELLDIR/server.log" 2>/dev/null; then
  echo "[cell][delta-token] 未启用（head ckpt 里没有 delta_token_state）" | tee -a "$LOG"
  audit_kv "delta_token_enabled" "0"
else
  echo "[cell][delta-token][WARN] server.log 里没有 delta-token 结论行（可能加载失败）" | tee -a "$LOG"
  audit_kv "delta_token_enabled" "UNKNOWN"
fi
# 服务端自己记录的口径（feat_dim/cap/token_dim/ckpt sha）
grep -m20 "\[audit\]\[server\]" "$CELLDIR/server.log" 2>/dev/null | sed 's/^/[cell] /' | tee -a "$LOG" || true

# ---- 逐 seed 跑（每 seed 一个 Isaac 进程，server 不复位不重启）----------------
for s in $SEEDS; do
  echo "[cell] seed=$s start $(date +%T)" | tee -a "$LOG"
  TASK="$TASK" SEED="$s" REPEAT=0 MAX_STEPS="$MAX_STEPS" ISAAC_GPU="$GPU" \
    SERVER_URL="http://127.0.0.1:$PORT" OUT_ROOT="$CELLDIR/seed$s" MODEL_LABEL="$LABEL" \
    RECORD_VIDEO=${RECORD_VIDEO:-0} \
    EXTRA_ARGS="--step_log_every_n 500" \
    bash scripts/ha_isaac_closedloop.sh >> "$CELLDIR/isaac_seed$s.log" 2>&1
  rc=$?
  audit_provider_check "$CELLDIR/isaac_seed$s.log" "$EXPECT_PROVIDER"
  prc=$?
  if [[ "$prc" -ne 0 ]]; then
    echo "[cell][FAIL] seed=$s provider 核对失败（见 isaac_seed$s.log）" | tee -a "$LOG"
    printf '{"seed": %s, "success": false, "failure_reason": "provider_mismatch", "episode_steps": 0, "max_steps": %s, "duration_sec": 0}\n' "$s" "$MAX_STEPS" > "$CELLDIR/ep_seed$s.json"
    continue
  fi
  R=$(ls -dt "$CELLDIR/seed$s"/*/result.json 2>/dev/null | head -1)
  if [[ -n "$R" && -f "$R" ]]; then
    cp -f "$R" "$CELLDIR/ep_seed$s.json"
    SUCC=$(python3 -c "import json;d=json.load(open('$R'));print(d.get('success'),d.get('failure_reason'),d.get('episode_steps'),round(float(d.get('duration_sec') or 0),1))" 2>/dev/null)
    echo "[cell] seed=$s rc=$rc -> $SUCC" | tee -a "$LOG"
  else
    echo "[cell] seed=$s rc=$rc -> 没有 result.json（见 isaac_seed$s.log）" | tee -a "$LOG"
    echo "{\"seed\": $s, \"success\": false, \"failure_reason\": \"no_result\", \"episode_steps\": 0, \"max_steps\": $MAX_STEPS, \"duration_sec\": 0}" > "$CELLDIR/ep_seed$s.json"
  fi
done

kill "$SPID" 2>/dev/null; sleep 1; kill -9 "$SPID" 2>/dev/null

python3 - "$CELLDIR" <<'PY' | tee -a "$LOG"
import json, glob, sys
d = sys.argv[1]
rows = [json.load(open(p)) for p in sorted(glob.glob(f"{d}/ep_seed*.json"))]
ok = sum(1 for r in rows if r.get("success"))
print(f"[cell] episodes={len(rows)} success={ok}")
for r in rows:
    print(f"  seed={r.get('seed')} steps={r.get('episode_steps')} reason={r.get('failure_reason')} dur={round(float(r.get('duration_sec') or 0),1)}s")
PY
echo "[cell] done $(date +%F' '%T)" | tee -a "$LOG"
