#!/usr/bin/env bash
# One automatic full-length round: collect 84 fresh episodes -> WM refresh ->
# joint -> official 32/48-pair eval.  Used by auto_ladder_r2r5.sh.
# All curriculum values are explicit env; no hidden defaults.
set -uo pipefail
COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1
: "${ROUND:?ROUND required}" "${N:?N required}" "${PREV_ROUND:?}" "${PREV_HEAD:?}"
: "${E_START:?}" "${EH:?}" "${LAM:?}" "${CAP:?}" "${ACT_W:?}" "${STEPS:?}" "${EVAL_N:?}" "${SEED:?}"
LOGD=logs/auto_ladder
mkdir -p "$LOGD"
LOG=$LOGD/r${N}_round${ROUND}.log
exec > >(tee -a "$LOG") 2>&1
say(){ echo "[auto r$N $(date '+%H:%M:%S')] $*"; }
die(){ say "[FATAL] $*"; echo "round_${ROUND}_failed: $*" > "runs/S3_loop/auto_ladder_fail.txt"; exit 1; }

B=$COMIMIC/runs/B_$ROUND
B_LAST=$N
BGLOB="$COMIMIC/runs/B_90[1-$B_LAST]/*/dump/*.pt"
BFRESH="$B/*/dump/*.pt"
CUR=$COMIMIC/runs/S3_loop/${ROUND}_joint/train_main/vla_final.pt
say "start round=$ROUND N=$N E=$E_START e_h=$EH lam=$LAM cap=$CAP act=$ACT_W steps=$STEPS eval=$EVAL_N"
[[ -s "$PREV_HEAD" ]] || die "missing PREV_HEAD=$PREV_HEAD"
[[ -f "$COMIMIC/runs/S1_handover/wm_901_a.pt" ]] || die "missing wm_901_a.pt"
[[ -f "$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt" ]] || die "missing v6 WBC"

# ---------------- 0) 驱动预检（2026-11-02 加）----------------
# 宿主驱动 >575 时 Isaac 会静默退回 CPU 物理（不报错、慢 10 倍、动力学口径不对）。
# 这一轮既有采集又有评测，两个都要 Isaac，所以在最前面拦住，别浪费一小时。
bash "$COMIMIC/scripts/check_isaac_driver.sh" || die "Isaac 驱动不兼容（见上）；换节点或 ALLOW_UNKNOWN_DRIVER=1"

# ---------------- 1) collect 84 fresh, if not already complete ----------------
NCOL=$(find "$B" -path '*/dump/*.pt' 2>/dev/null | wc -l)
if (( NCOL < 84 )); then
  PORT=$((18182 + N * 2))
  SRVLOG=$LOGD/vla_server_${PORT}.log
  say "start VLA server port=$PORT head=$PREV_HEAD"
  PORT=$PORT GPU=0 CKPT="$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000" \
    HEAD_CKPT="$PREV_HEAD" LORA_RANK=32 LORA_TARGETS="attn1,ff" LORA_ALSO_TRAIN=proj_out \
    SRV_SEED=0 LOG="$SRVLOG" bash scripts/start_vla_server.sh > "$SRVLOG" 2>&1 &
  srv=$!
  ok=0
  for i in $(seq 1 90); do
    if grep -q "listening on http" "$SRVLOG" 2>/dev/null; then ok=1; break; fi
    if ! kill -0 "$srv" 2>/dev/null; then break; fi
    sleep 4
  done
  (( ok == 1 )) || { kill "$srv" 2>/dev/null; die "VLA server did not start (see $SRVLOG)"; }
  say "VLA server ready; collecting 84 episodes"
  python3 scripts/collect_full_episodes.py --round "$ROUND" --episodes 14 --seed "$SEED" \
    --server-url "http://127.0.0.1:$PORT" --jobs "${CJOBS:-8}" --num-gpus "${CGPS:-8}" \
    --sm-ckpt "$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt" \
    || { kill "$srv" 2>/dev/null; pkill -f "[h]a_vla_latent64_server.py.*--port $PORT" 2>/dev/null; die "collection failed"; }
  kill "$srv" 2>/dev/null
  pkill -f "[h]a_vla_latent64_server.py.*--port $PORT" 2>/dev/null
  wait "$srv" 2>/dev/null
  sleep 3
fi
NCOL=$(find "$B" -path '*/dump/*.pt' 2>/dev/null | wc -l)
(( NCOL >= 84 )) || die "B_$ROUND has only $NCOL dumps (need 84)"
say "B_$ROUND ready: $NCOL dumps"

# ---------------- 2) WM stage1 reuse ----------------
cp -n "$COMIMIC/runs/S1_handover/wm_901_a.pt" "$COMIMIC/runs/S1_handover/wm_${ROUND}_a.pt" 2>/dev/null || true
[[ -s "$COMIMIC/runs/S1_handover/wm_${ROUND}_a.pt" ]] || die "missing wm_${ROUND}_a.pt"

# ---------------- 3) pipeline: collect/score/transitions/WM/gate/critic/joint ----------------
say "pipeline stages collect score transitions wm_refresh wm_gate critic_gate joint"
run_stage() {
  WM_SKIP_P1=1 \
  ROUND="$ROUND" B_GLOB="$BGLOB" B_FRESH_GLOB="$BFRESH" B_FRESH_WEIGHT=3.0 B_FRESH_PIN=0 \
  INIT_VLA="$PREV_HEAD" SM_CKPT="$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt" \
  A_ORDER=declared H=20 STEPS="$STEPS" BATCH="${JBATCH:-64}" ACCUM="${JACCUM:-1}" DDP=1 NPROC="${JNPROC:-8}" TRAIN_ARMS=main \
  MASTER_PORT=$((29640 + N * 10)) \
  ACT_WEIGHT="$ACT_W" ACT_WEIGHT_SCHEDULE="${ROUND}:${ACT_W}" ACT_EPS=0.6 LAMBDA_R=0.3 \
  B_START_MAX_ERR="$E_START" B_SELF_ERR_MAX="" \
  TAIL_WEIGHT="$LAM" TAIL_POS_THR="$EH" TAIL_POS_CAP="$CAP" TAIL_CAP_SLOPE=0.5 \
  YAW_HINGE_LAMBDA=0 YAW_RATE_WEIGHT_OVERRIDE=0 \
  REWARD_ROOT_POS_STD=0.2 REWARD_ROOT_ORI_STD=0.25 REWARD_ROOT_WEIGHT=1.0 \
  ROOT_ORI_MODE=split REWARD_MODE=saturating REWARD_VEL_SCALE=0 POSTURE_WEIGHT=0 \
  STATE_EXTRA=rotlocal6 ADV_BASELINE=self ADV_TAU=0.02 TRUST_CENTER=label TRUST_EPS=0.2 TRUST_WEIGHT=1.0 \
  LAMBDA_BC=0 TOKEN_BC=0 LAMBDA_HAND=0 WEIGHT_DECAY=1e-4 OBS_NOISE=0 FM_LR=3e-5 LORA_ALPHA=64 \
  TOKEN_BC_SOURCE=ref TOKEN_BC_SPAN=0 \
  CRITIC_ON=1 CRITIC_SOURCE=imagined CRITIC_GRAD_STATE=1 CRITIC_ON_MAIN=1 CRITIC_GRAD_STATE_MAIN=1 \
  CRITIC_STATE_GRAD_SCALE=0.1 TRAIN_WBC_MAIN=0 WBC_UNFREEZE=all WBC_KL_ANCHOR=0 VLA_KL_ANCHOR=0 \
  B_SELF_TARGETS="$COMIMIC/runs/wm_real/self_targets_all.pt" \
  CRITIC_REAL="$COMIMIC/runs/wm_real/real_transitions_v2b.pt" \
  EVAL_STEPS=2080 COLLECT_EPISODES=14 RUN_TAG="full_r$N" RUN_LABEL="full_r$N" \
  bash scripts/s3_pipeline.sh "$1" || die "pipeline stage $1 failed"
}
for st in collect score transitions wm_refresh wm_gate critic_gate joint; do
  say "pipeline stage: $st"
  run_stage "$st"
done
[[ -s "$CUR" ]] || die "missing current head $CUR"
say "joint done: $CUR"

# ---------------- 4) official main eval: previous vs current ----------------
TAG="full_r${N}_vs_r${PREV_ROUND}"
say "eval main previous vs current: $TAG n=$EVAL_N"
HEAD_MAIN="$PREV_HEAD" HEAD_CTRL="$CUR" \
SM_CKPT_MAIN="$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt" \
SM_CKPT_CTRL="$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt" \
LABEL_MAIN="r${PREV_ROUND}" LABEL_CTRL="r${ROUND}" \
TAG_PREFIX="$TAG" START_FRAME=0 MAX_STEPS=2080 SEED_TOTAL="$EVAL_N" N_SWEEPS="${N_SWEEPS:-4}" \
BASE_PORT=$((18500 + N * 10)) GPU_OFFSET=0 \
bash scripts/eval_h20_8gpu.sh || die "main eval failed"
M="runs/S3_loop/sweeps/${TAG}_merged.json"
[[ -f "$M" ]] || die "missing main merge $M"
python3 - "$M" "$ROUND" <<'PY' || die "main gate failed"
import json,sys,pathlib
d=json.load(open(sys.argv[1])); p=d["paired"]
m, lo, hi, win, n = p["mean"], p["ci95_boot"][0], p["ci95_boot"][1], p["better"], p["n"]
med = p.get("median")
fm = d["main"].get("fell"); fc = d["ctrl"].get("fell")
# 2026-11-01: full-length root mean is underpowered at n=32 (SEM~6cm).
# Directional+safety gate: mean & median negative, >=55% wins, falls not worse.
ok = (m < 0) and (med is not None and med < 0) and (win >= n * 0.55) and (fc is not None and fm is not None and fc <= fm)
print(f"[main-gate] mean={m:+.4f} med={med:+.4f} CI=[{lo:+.4f},{hi:+.4f}] better={win}/{n} fell={fm}->{fc} -> {'PASS' if ok else 'FAIL'}")
if fc and fc > 0:
    print(f"[main-gate][STRICT] 候选摔了 {fc} 次 ⇒ 严格安全门 falls==0 未过（本门 fc<=fm 允许）")
if not ok:
    pathlib.Path(f"runs/S3_loop/round_{sys.argv[2]}_gate_fail.txt").write_text(
        f"main_gate mean={m} ci=[{lo},{hi}] better={win}/{n}\n")
raise SystemExit(0 if ok else 1)
PY
say "main gate PASS"

# ---------------- 5) base anchor (non-blocking) ----------------
BTAG="base_vs_full_r${N}"
say "eval base anchor: $BTAG n=$EVAL_N"
HEAD_MAIN=base HEAD_CTRL="$CUR" \
SM_CKPT_MAIN="$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt" \
SM_CKPT_CTRL="$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt" \
LABEL_MAIN=base LABEL_CTRL="r${ROUND}" \
TAG_PREFIX="$BTAG" START_FRAME=0 MAX_STEPS=2080 SEED_TOTAL="${ANCHOR_N:-$EVAL_N}" N_SWEEPS="${ANCHOR_SWEEPS:-${N_SWEEPS:-4}}" \
BASE_PORT=$((18600 + N * 10)) GPU_OFFSET=0 \
bash scripts/eval_h20_8gpu.sh || say "base anchor eval failed (non-blocking)"
say "round $ROUND DONE; head=$CUR"
echo "$CUR" > runs/S3_loop/auto_round${ROUND}_head.txt
