#!/usr/bin/env bash
# v13 单任务闭环（自包含，不依赖 scripts/r3_closedloop_ab.sh）。
#
# 口径 100% 复刻 runs/closedloop_r3/od_v13/OpenDoor__seed0__repeat0__20261106_123945：
#   底模 ckpt = runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000
#   action head = runs/joint_v13/vla_final.pt
#   server 全默认（fm_steps=4 sample_avg=1 decode=codec token_quant=none state_layout=legacy
#                  state_order=ha_v31 hand_from_base=False hand_source=feedback lead_beta=0.0
#                  horizon=40 lora_rank=32 seed=0）
#   闭环 SONIC release 执行器 + HA_VLA_EXEC_STEPS=20（= 训练 H=20 的 receding horizon）
#   RECORD_VIDEO=1（record_video_every_n=1 → h264 30fps mp4）
#
# ★ 为什么要自包含：r3_closedloop_ab.sh 里的 SRV_PY 被别的会话反复塞进
#   third_party/skimage_py311（cp311 扩展），本机 python3=3.12 ⇒ server 必崩
#   ModuleNotFoundError: No module named 'skimage.measure._ccomp'。
#   这里把 PYTHONPATH 写死成本机 python3.12 能用的那一份（= od_v13 当初的已提交版本）。
#
# 用法：TASK=SitSofa GPU=1 PORT=18212 TAG=sits_v13_t60 MAX_STEPS=3000 bash scripts/r3_v13_task.sh
#   MAX_STEPS=600  → 12 s 仿真（视频 20.03 s）
#   MAX_STEPS=3000 → 60 s 仿真（仓库 "t60" 口径；原始视频 100.03 s，_realtime 60 s）
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
TASK=${TASK:?TASK required}
GPU=${GPU:?GPU required}
PORT=${PORT:?PORT required}
TAG=${TAG:?TAG required}
HEAD=${HEAD:-runs/joint_v13/vla_final.pt}
CKPT=${CKPT:-runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}
MAX_STEPS=${MAX_STEPS:-600}
SEED=${SEED:-0}
RECORD_VIDEO=${RECORD_VIDEO:-1}

OUT=$COMIMIC/runs/closedloop_r3/$TAG
mkdir -p "$OUT"
SLOG=$OUT/server.log
DLOG=$OUT/isaac.log
DUMP=$OUT/dump
rm -rf "$DUMP"
cd "$COMIMIC" || exit 1

SIMPLE=$(( MAX_STEPS * 2 / 100 ))   # 0.02 s/step → 仿真秒数
echo "[v13] ===== start $(date +%F' '%T) task=$TASK gpu=$GPU port=$PORT tag=$TAG steps=$MAX_STEPS (${SIMPLE}s sim) =====" | tee "$OUT/driver.log"

# ---- 1) VLA 服务 -------------------------------------------------------------
[[ -f "$HEAD" ]] || { echo "[v13] head ckpt 不存在: $HEAD" >&2; exit 2; }
[[ -d "$CKPT" ]] || { echo "[v13] base ckpt 不存在: $CKPT" >&2; exit 2; }

# 只给 server 这一条进程加 vendored 依赖（Isaac 那条由 isaac_env.sh 自己 prepend venv）。
# ★ py312_shims 必须排在 gr00t_deps 前面：2026-11-06 05:06 有人把 **cp311** 的 skimage
#   拷进了 third_party/gr00t_deps/skimage（.so 是 cpython-311），本机 python3=3.12 会
#   ModuleNotFoundError: skimage.measure._ccomp。py312_shims/skimage 指向系统那份 0.26.0
#   的 cp312 构建，遮住坏的即可。
SRV_PY="$COMIMIC:$COMIMIC/Isaac-GR00T:$COMIMIC/third_party/py312_shims:$COMIMIC/third_party/gr00t_deps:$COMIMIC/third_party/gr00t_deps/_noflash"
setsid nohup env PYTHONPATH="$SRV_PY" \
  python3 scripts/ha_vla_latent64_server.py \
  --ckpt "$CKPT" --head-ckpt "$HEAD" \
  --port "$PORT" --device "cuda:$GPU" --seed "$SEED" \
  > "$SLOG" 2>&1 &
SPID=$!
echo "[v13] server pid=$SPID port=$PORT isaac_gpu=$GPU srv_gpu=$GPU" | tee -a "$OUT/driver.log"

for _ in $(seq 1 150); do
  grep -q "listening on" "$SLOG" 2>/dev/null && break
  if ! kill -0 "$SPID" 2>/dev/null; then
    echo "[v13] server 启动失败：" >&2; tail -20 "$SLOG" >&2; exit 3
  fi
  sleep 2
done
if ! grep -q "listening on" "$SLOG" 2>/dev/null; then
  echo "[v13] server 300s 未就绪" >&2; tail -20 "$SLOG" >&2
  kill "$SPID" 2>/dev/null; exit 3
fi
echo "[v13] server ready" | tee -a "$OUT/driver.log"
grep -E "\[server\]\[cfg\]|\[server\]\[head\]" "$SLOG" | tee -a "$OUT/driver.log"

# ---- 2) 官方闭环（SONIC 执行器）----------------------------------------------
export HA_ALLOW_SONIC=1
export HA_GMT_BACKEND=sonic
export HA_ACTION_SOURCE=sonic_wholebody
export SONIC_VLA_ACTION_FORMAT=latent64
export SUPERMIMIC_HA_ALIGN=1
export HA_VLA_EXEC_STEPS=20
export SONIC_DEBUG_DUMP=$DUMP
T0=$SECONDS
TASK=$TASK MAX_STEPS=$MAX_STEPS SEED=$SEED REPEAT=0 ISAAC_GPU=$GPU \
  SERVER_URL=http://127.0.0.1:$PORT OUT_ROOT=$OUT RECORD_VIDEO=$RECORD_VIDEO \
  EXTRA_ARGS="--step_log_every_n 20" \
  bash scripts/ha_isaac_closedloop.sh > "$DLOG" 2>&1
RC=$?
echo "[v13] isaac rc=$RC elapsed=$((SECONDS-T0))s" | tee -a "$OUT/driver.log"

kill "$SPID" 2>/dev/null; sleep 1; kill -9 "$SPID" 2>/dev/null

# ---- 3) 结果 + 视频 ----------------------------------------------------------
RES=$(ls -dt "$OUT"/${TASK}__seed${SEED}__repeat0__* 2>/dev/null | head -1)
if [[ -z "$RES" || ! -f "$RES/result.json" ]]; then
  echo "[v13] FAIL 没有 result.json（rc=$RC），见 $DLOG" | tee -a "$OUT/driver.log"
  tail -30 "$DLOG" >&2
  exit 4
fi
echo "[v13] result.json: $RES/result.json" | tee -a "$OUT/driver.log"
python3 - "$RES/result.json" <<'PY' | tee -a "$OUT/driver.log"
import json,sys
d=json.load(open(sys.argv[1]))
for k in ("task","success","failure_reason","episode_steps","max_steps","duration_sec"):
    print(f"  {k}: {d.get(k)}")
PY

SRC=$(python3 -c "import json;print(json.load(open('$RES/result.json')).get('video_path') or '')")
BASE=$(echo "$TASK" | tr 'A-Z' 'a-z')
TLEN=""; [[ "$MAX_STEPS" != "600" ]] && TLEN="_t$((MAX_STEPS/50))"
STEM="${BASE}_v13_seed0${TLEN}"
if [[ -z "$SRC" || ! -f "$SRC" ]]; then
  echo "[v13] WARN 没找到 mp4（video_path='$SRC'）" | tee -a "$OUT/driver.log"
else
  cp -f "$SRC" "$OUT/${STEM}.mp4"
  echo "[v13] video -> $OUT/${STEM}.mp4" | tee -a "$OUT/driver.log"
  FF=""
  command -v ffmpeg >/dev/null 2>&1 && FF=$(command -v ffmpeg)
  [[ -z "$FF" && -x "$COMIMIC/closed_loop/bin/ffmpeg" ]] && FF=$COMIMIC/closed_loop/bin/ffmpeg
  if [[ -n "$FF" ]]; then
    PLAY=$("$FF" -i "$SRC" 2>&1 | grep -o 'Duration: [0-9:.]*' | head -1 | awk '{print $2}')
    if [[ -n "$PLAY" ]]; then
      PLAY_SEC=$(python3 -c "
h,m,s='$PLAY'.split(':')
print(float(h)*3600+float(m)*60+float(s))")
      # _realtime = 仿真时间速度（1 s 仿真 = 1 s 视频）；录制本身是 30fps 定速快放
      FACTOR=$(python3 -c "print(f'{${SIMPLE}/max($PLAY_SEC,1e-6):.4f}')")
      "$FF" -y -v error -i "$SRC" -filter:v "setpts=${FACTOR}*PTS" -r 30 \
        -an -c:v libx264 -preset veryfast -crf 23 "$OUT/${STEM}_realtime.mp4" \
        && echo "[v13] realtime(${SIMPLE}s sim) -> $OUT/${STEM}_realtime.mp4 (raw=${PLAY_SEC}s, factor=${FACTOR})" | tee -a "$OUT/driver.log"
    fi
  fi
fi
echo "[v13] ===== done $(date +%F' '%T) task=$TASK tag=$TAG =====" | tee -a "$OUT/driver.log"
