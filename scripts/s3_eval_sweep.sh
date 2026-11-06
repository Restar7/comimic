#!/usr/bin/env bash
# 两卡并行**多种子**闭环扫描：server 只起一次，GPU0 跑 HEAD_MAIN、GPU1 跑 HEAD_CTRL，
# 对 SEEDS 里每个 reset 种子各跑一集（同一行号、同一步数），最后给出每臂的均值/标准差。
#
# 为什么必须这样做：闭环里 VLA 的 flow-matching 采样是**随机的**，同一配置单集重复
# 会在 0.13–0.42 m 之间跳（2026-10-28 实测 π₀ 三次：0.130 / 0.163 / 0.421）。
# 所以 n=1 的比较（包括文档里 GT 0.082 vs VLA 0.200）不能用来判优劣。
#
# 用法：
#   SEEDS="0 1 2 3 4 5" TAG=pi0_vs_pi3main START_FRAME=0 \
#     HEAD_MAIN=base HEAD_CTRL=$PWD/runs/S3_loop/r3/train_main/vla_000150.pt \
#     bash scripts/s3_eval_sweep.sh
set -uo pipefail

COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1
export COMIMIC

TAG=${TAG:-sweep}
TASK=${TASK:-DoubleDesk}
START_FRAME=${START_FRAME:-0}
SEEDS=${SEEDS:-"0 1 2 3 4 5"}
MAX_STEPS=${MAX_STEPS:-300}
SM_CKPT=${SM_CKPT:-$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt}
# 2026-10-29：`TRAIN_WBC=1` 训过的臂，执行器在它的 `joint_*.pt` 里，必须单独指定。
SM_CKPT_MAIN=${SM_CKPT_MAIN:-$SM_CKPT}
SM_CKPT_CTRL=${SM_CKPT_CTRL:-$SM_CKPT}
HEAD_MAIN=${HEAD_MAIN:-base}
HEAD_CTRL=${HEAD_CTRL:-base}
OUT_ROOT=${OUT_ROOT:-$COMIMIC/runs/S3_loop/sweeps}
OUT=$OUT_ROOT/$TAG
PORT_MAIN=${PORT_MAIN:-18100}
PORT_CTRL=${PORT_CTRL:-18101}
LORA_TARGETS=${LORA_TARGETS:-attn1,ff}
# 2026-10-28：每个槽位可以带自己的 server 参数（如 "EXTRA_CTRL=--sample-avg 4"）。
EXTRA_MAIN=${EXTRA_MAIN:-}
EXTRA_CTRL=${EXTRA_CTRL:-}
# 2026-10-29：重规划间隔（开环盲区）。20=文档口径；10=盲区减半
EXEC_STEPS_MAIN=${EXEC_STEPS_MAIN:-20}
EXEC_STEPS_CTRL=${EXEC_STEPS_CTRL:-20}

mkdir -p "$OUT" "$COMIMIC/logs"
say() { echo "[$(date +%H:%M:%S)] [sweep $TAG] $*"; }

pids_on_port() {
  local port=$1 pid cl
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
    cl=$( { tr '\0' ' ' < "$d/cmdline"; } 2>/dev/null ) || continue
    case "$cl" in *ha_vla_latent64_server*"--port $port"*) echo "$pid" ;; esac
  done
}
kill_all_servers() {
  # ★ 2026-11-01：只在**本 sweep 自己的端口**上清残兵。原来按进程名全杀，
  #   8 卡机器上同时开 4 条 sweep 时，先结束的那条会把别人正在用的 server 杀掉
  #   ⇒ 其余 sweep 静默失败、summary 缺集。这里收紧到 PORT_MAIN/PORT_CTRL。
  local pid cl
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
    cl=$( { tr '\0' ' ' < "$d/cmdline"; } 2>/dev/null ) || continue
    case "$cl" in
      *ha_vla_latent64_server*"--port $PORT_MAIN"*) kill -9 "$pid" 2>/dev/null ;;
      *ha_vla_latent64_server*"--port $PORT_CTRL"*) kill -9 "$pid" 2>/dev/null ;;
    esac
  done
}
start_srv() {  # port gpu head tag
  local port=$1 gpu=$2 head=$3 tag=$4 hc=""
  [[ "$head" != "base" ]] && hc="$head"
  for p in $(pids_on_port "$port"); do kill -9 "$p" 2>/dev/null; done
  say "server $tag port=$port gpu=$gpu head=$(basename "${hc:-base}")"
  local extra=""
  [[ "$tag" == "main" ]] && extra="$EXTRA_MAIN"
  [[ "$tag" == "ctrl" ]] && extra="$EXTRA_CTRL"
  PORT=$port GPU=$gpu HEAD_CKPT="$hc" LOG_DIR="$OUT/vla_in_$tag" SRV_EXTRA="$extra" \
    SRV_SEED="${SRV_SEED:-0}" \
    LORA_RANK=32 LORA_TARGETS="$LORA_TARGETS" LORA_ALSO_TRAIN=proj_out \
    bash "$COMIMIC/scripts/start_vla_server.sh" > "$COMIMIC/logs/sweep_${TAG}_srv_$tag.log" 2>&1 &
  echo $!
}
wait_srv() {
  local port=$1 t0=$SECONDS
  while (( SECONDS - t0 < 600 )); do
    (echo > "/dev/tcp/127.0.0.1/$port") 2>/dev/null && { say "server :$port 就绪（$((SECONDS-t0))s）"; return 0; }
    sleep 3
  done
  say "[FAIL] server :$port 没起来"; return 1
}

# 2026-10-28 22:50：**必须把槽位与 head 的对应关系落盘**。
# 之前 `summary.json` 只有 main/ctrl 两个槽位名，而 main=π₀、ctrl=候选 这种排布
# 极易被读反（本次就据此写过错的结论）。从此 summary 里直接带 head 路径和简称。
python3 - "$OUT" "$HEAD_MAIN" "$HEAD_CTRL" "$EXTRA_MAIN" "$EXTRA_CTRL" "$SM_CKPT_MAIN" "$SM_CKPT_CTRL" "${LABEL_MAIN:-}" "${LABEL_CTRL:-}" <<'PY2'
import json, sys
from pathlib import Path
out, hm, hc = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
em = sys.argv[4] if len(sys.argv) > 4 else ""
ec = sys.argv[5] if len(sys.argv) > 5 else ""
wm = sys.argv[6] if len(sys.argv) > 6 else ""
wc = sys.argv[7] if len(sys.argv) > 7 else ""
# 2026-10-29：允许显式标签（消融实验必须能一眼看出哪条臂做了什么）
lab_m = sys.argv[8] if len(sys.argv) > 8 else ""
lab_c = sys.argv[9] if len(sys.argv) > 9 else ""
def short(h):
    # 2026-10-28：原来只拼到 "r7/vla_000150.pt"，两条臂同名分不开。
    # 现在拼成 "r7/train_main" / "r7/train_ctrl"，一眼能看出是哪条臂。
    if h == "base":
        return "base(pi0)"
    q = Path(h)
    return f"{q.parent.parent.name}/{q.parent.name}"
(out / "heads.json").write_text(json.dumps(
    {"main": {"head": hm, "label": (lab_m or short(hm)), "extra": em, "wbc": wm},
     "ctrl": {"head": hc, "label": (lab_c or short(hc)), "extra": ec, "wbc": wc}}, indent=1, ensure_ascii=False))
print(f"[sweep] server extra: main=[{em}] ctrl=[{ec}]")
print(f"[sweep] 槽位: main={short(hm)}  ctrl={short(hc)}")
PY2

# ★ 2026-11-01：槽位 GPU 可配（默认 0/1）。用来在**一卡被训练占着**时，把两个槽位都放到另一张卡。
GPU_MAIN=${GPU_MAIN:-0}
GPU_CTRL=${GPU_CTRL:-1}
P0=$(start_srv "$PORT_MAIN" "$GPU_MAIN" "$HEAD_MAIN" main)
P1=$(start_srv "$PORT_CTRL" "$GPU_CTRL" "$HEAD_CTRL" ctrl)
wait_srv "$PORT_MAIN" || { kill "$P0" "$P1" 2>/dev/null; exit 1; }
wait_srv "$PORT_CTRL" || { kill "$P0" "$P1" 2>/dev/null; exit 1; }

for s in $SEEDS; do
  say "seed=$s row=$START_FRAME"
  ARM=vla TASK="$TASK" SEED=0 START_FRAME="$START_FRAME" RESET_SEED="$s" MAX_STEPS="$MAX_STEPS" \
    GPU=$GPU_MAIN SERVER_URL="http://127.0.0.1:$PORT_MAIN" SM_CKPT="$SM_CKPT_MAIN" EXEC_STEPS="$EXEC_STEPS_MAIN" \
    OUT="$OUT/main_r$s" bash "$COMIMIC/scripts/s3_collect_arm.sh" \
    > "$COMIMIC/logs/sweep_${TAG}_main_r$s.log" 2>&1 &
  A=$!
  ARM=vla TASK="$TASK" SEED=0 START_FRAME="$START_FRAME" RESET_SEED="$s" MAX_STEPS="$MAX_STEPS" \
    GPU=$GPU_CTRL SERVER_URL="http://127.0.0.1:$PORT_CTRL" SM_CKPT="$SM_CKPT_CTRL" EXEC_STEPS="$EXEC_STEPS_CTRL" \
    OUT="$OUT/ctrl_r$s" bash "$COMIMIC/scripts/s3_collect_arm.sh" \
    > "$COMIMIC/logs/sweep_${TAG}_ctrl_r$s.log" 2>&1 &
  B=$!
  wait "$A"; echo "[sweep] main seed=$s rc=$?"
  wait "$B"; echo "[sweep] ctrl seed=$s rc=$?"
done

kill "$P0" "$P1" 2>/dev/null
kill_all_servers
say "全部结束，汇总中"

PYTHONPATH=. python3 - "$OUT" <<'PY'
import glob, json, sys
from pathlib import Path
import torch
out = Path(sys.argv[1])
lib = torch.load(str(Path(__import__('os').environ.get('COMIMIC', '.')) /
                     'runs/isaac_motion/ha_cmd/packed.pt'),
                 map_location='cpu', weights_only=False)['ref'].float()
def one(p):
    d = torch.load(str(p), map_location='cpu', weights_only=False)
    S = d['states'].float(); fr = d['frames'].long().clamp(0, int(lib.shape[0]) - 1)
    e = (S[:, :3] - lib[fr][:, :3]).norm(dim=-1)
    j = ((S[:, 7:36] - lib[fr][:, 7:36]) ** 2).mean(-1).sqrt()
    w, x, y, z = S[:, 3:7].unbind(-1)
    tilt = torch.rad2deg(torch.acos((1 - 2 * (x * x + y * y)).clamp(-1, 1)))
    return float(e.mean()), float(j.mean()), float(tilt.max())
rows = {}
for arm in ('main', 'ctrl'):
    vals = []
    for p in sorted(glob.glob(str(out / f'{arm}_r*' / 'dump/*.pt'))):
        vals.append(one(Path(p)))
    if not vals:
        continue
    n = len(vals)
    m = sum(v[0] for v in vals) / n
    sd = (sum((v[0] - m) ** 2 for v in vals) / max(1, n - 1)) ** .5
    rows[arm] = dict(n=n, root_mean=m, root_sd=sd,
                     joint=sum(v[1] for v in vals) / n,
                     tilt=sum(v[2] for v in vals) / n,
                     per_seed=[round(v[0], 3) for v in vals],
                     per_tilt_max=[round(v[2], 1) for v in vals])
heads = json.loads((out / "heads.json").read_text()) if (out / "heads.json").exists() else {}
for k in ("main", "ctrl"):
    if k in rows and k in heads:
        rows[k]["label"] = heads[k]["label"]
        rows[k]["head"] = heads[k]["head"]
print(json.dumps(rows, indent=1, ensure_ascii=False))
(out / 'summary.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False))
if 'main' in rows and 'ctrl' in rows:
    lm = rows['main'].get('label', 'main'); lc = rows['ctrl'].get('label', 'ctrl')
    d = rows['ctrl']['root_mean'] - rows['main']['root_mean']
    print(f"[sweep] {lc} − {lm} = {d:+.3f} m  "
          f"({lm} {rows['main']['root_mean']:.3f}±{rows['main']['root_sd']:.3f} vs "
          f"{lc} {rows['ctrl']['root_mean']:.3f}±{rows['ctrl']['root_sd']:.3f})")

    # ---- 2026-10-30：按 mjlab §11 的评测方法论补三项 -------------------------
    # (1) 配对差的自助法 95% CI（250 次重采样）——单点 ±sd 在小样本上不可信；
    # (2) 中位数与 p90：纯 exp 奖励会让"中位变好、尾部变差"，只盯均值会看不见；
    # (3) 摔倒率：mjlab 的 `meta['fell']`，这里用 tilt.max() > 45° 近似一集是否翻过。
    import random as _rnd
    a_seed = rows['main']['per_seed']; b_seed = rows['ctrl']['per_seed']
    dif = [y - x for x, y in zip(a_seed, b_seed)]
    _rnd.seed(0)
    boots = []
    for _ in range(250):
        samp = [_rnd.choice(dif) for _ in dif]
        boots.append(sum(samp) / len(samp))
    boots.sort()
    lo, hi = boots[6], boots[243]          # ~95%
    mu = sum(dif) / len(dif)
    sd = (sum((x - mu) ** 2 for x in dif) / max(1, len(dif) - 1)) ** .5
    sem = sd / len(dif) ** .5
    rows['main']['paired_vs_ctrl'] = dict(
        mean=round(mu, 4), sem=round(sem, 4), ci95_boot=[round(lo, 4), round(hi, 4)],
        better=int(sum(1 for x in dif if x < 0)), n=len(dif),
        median=round(sorted(b_seed)[len(b_seed) // 2] - sorted(a_seed)[len(a_seed) // 2], 4),
        p90_main=round(sorted(a_seed)[int(len(a_seed) * 0.9)], 4),
        p90_ctrl=round(sorted(b_seed)[int(len(b_seed) * 0.9)], 4),
        fell_main=int(sum(1 for v in rows['main'].get('per_tilt_max', []) if v > 45.0)),
        fell_ctrl=int(sum(1 for v in rows['ctrl'].get('per_tilt_max', []) if v > 45.0)),
    )
    (out / 'summary.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    pv = rows['main']['paired_vs_ctrl']
    print(f"[sweep] 配对差 {mu:+.4f} m  sem={sem:.4f}  自助 95%CI=[{lo:+.4f}, {hi:+.4f}]  "
          f"候选更好 {pv['better']}/{pv['n']}")
    print(f"[sweep] 中位差 {pv['median']:+.4f} m   p90: {lm} {pv['p90_main']:.3f} vs "
          f"{lc} {pv['p90_ctrl']:.3f}   翻倒(tilt>45°) {pv['fell_main']}/{pv['fell_ctrl']}")
    if lo <= 0.0 <= hi:
        print("[sweep] ⇒ CI 跨 0：本场两臂不可分辨（别据此判优劣）")
PY
