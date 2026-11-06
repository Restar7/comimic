#!/usr/bin/env bash
# 全量 A'（4×4090，全长覆盖 + 噪声）：按 clip 长度分桶，4 张卡排队跑，最后合并。
#
# 为什么要分桶：一个 launch 里所有 env 共用同一个 `--steps`，而 clip 长度从 271 到 10000 帧。
# 不分桶就得按 10000 步跑，浪费 11×。分桶后短 clip 只跑自己的长度（越界行由掩码丢掉）。
#
# 用法：
#   SEEDS="0 1 2 3" bash scripts/run_a_prime_full_4gpu.sh
# 关键旋钮：
#   WBC_CKPT   默认 HA-only 28k（= 当前选定）
#   NOISE_STD  默认 0.05（已校准：800 步全站立）
#   TOKEN_SRC  默认 online（实时算 token，= 训练/部署同口径）；npy 是开环对照
set -uo pipefail
C=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$C" || exit 1

ROUND=${ROUND:-a_full}
SEEDS=${SEEDS:-"0 1 2 3"}
WBC_CKPT=${WBC_CKPT:-$C/runs/S1_handover/exp_WBC_exec_haonly_best.pt}
PACKED=${PACKED:-$C/runs/isaac_motion/ha_cmd_exec/packed.pt}
TOKEN_LIB=${TOKEN_LIB:-$C/runs/isaac_motion/ha_cmd/packed.pt}
TOKENS_DIR=${TOKENS_DIR:-$C/runs/exp_tokens}
TOKEN_SRC=${TOKEN_SRC:-online}
NOISE_STD=${NOISE_STD:-0.05}
HOLD=${HOLD:-20}
SP_XY=${SP_XY:-0.0}; SP_YAW=${SP_YAW:-0.0}; SP_DQ=${SP_DQ:-0.0}
BASE=${BASE:-$C/runs/wm_real/exp_a_prime_full/$ROUND}
JOBS_GPU=${JOBS_GPU:-1}          # 每卡几个进程（显存不是瓶颈，CPU 也够，先 1）
mkdir -p "$BASE/logs"

source "$C/scripts/isaac_env.sh" >/dev/null 2>&1
export PYTHONPATH="$C:$C/trackerlab-supermimic/source/trackerTask:$C/trackerlab-supermimic/source/trackerLab:${PYTHONPATH:-}"
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}

# ---- 分桶（由 packed 的长度算出来，写进 jobs.txt：seed bucket len_min len_max steps nclip）----
python3 - "$PACKED" "$TOKENS_DIR" "$BASE/jobs.txt" <<'PY'
import sys, torch
from pathlib import Path
packed, tokdir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
d = torch.load(packed, map_location="cpu", weights_only=False)
names = list(d["names"]); L = [int(x) for x in d["lengths"]]
ok = [(n, l) for n, l in zip(names, L) if (tokdir / f"{n}.npy").exists()]
edges = [(0, 600), (600, 1200), (1200, 3000), (3000, 10**9)]
buckets = []
for lo, hi in edges:
    sel = [l for _, l in ok if lo < l <= hi]
    if not sel:
        continue
    buckets.append((lo, hi, max(sel), len(sel)))
print("[full] 分桶（长度区间 / steps / clip 数）:")
for lo, hi, st, n in buckets:
    print(f"   ({lo:5d},{hi if hi < 10**9 else 'inf'}]  steps={st:6d}  clips={n:4d}  行≈{st*n}")
out.write_text("\n".join(f"{i} {lo} {hi} {st} {n}" for i, (lo, hi, st, n) in enumerate(buckets)) + "\n")
PY

# ---- 作业队列：按 (steps 大的先跑) 排序，4 卡轮流认领 ----
: > "$BASE/job_status.txt"
run_one() {  # seed lo hi steps nclip gpu
  local seed=$1 lo=$2 hi=$3 steps=$4 nclip=$5 gpu=$6
  local tag="s${seed}_b${lo}-${hi}"
  local out="$BASE/${tag}.pt"
  local log="$BASE/logs/${tag}.log"
  if [[ -s "$out" ]]; then echo "[full] skip $tag（已存在）"; return 0; fi
  CUDA_VISIBLE_DEVICES=$gpu "$ISAAC_VENV/bin/python" -u scripts/isaac_a_collect_batch.py \
    --wbc-ckpt "$WBC_CKPT" --packed "$PACKED" --tokens-dir "$TOKENS_DIR" \
    --num-envs "$nclip" --steps "$steps" --start-mode first --headless --log-every 200 \
    --token-source "$TOKEN_SRC" --token-lib "$TOKEN_LIB" \
    --action-noise-std "$NOISE_STD" --perturb-hold "$HOLD" \
    --start-perturb-xy "$SP_XY" --start-perturb-yaw "$SP_YAW" --start-perturb-dq "$SP_DQ" \
    --len-min "$lo" --len-max "$hi" --seed "$seed" --out "$out" > "$log" 2>&1
  local rc=$?
  echo "[full] $tag gpu=$gpu rc=$rc rows=$(python3 -c "
import torch;d=torch.load('$out',map_location='cpu',weights_only=False);print(d['states'].shape[0])" 2>/dev/null || echo NA)"
}

SLOTS=$(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) * JOBS_GPU ))
echo "[full] seeds=[$SEEDS] noise=$NOISE_STD token=$TOKEN_SRC slots=$SLOTS out=$BASE"

# 作业表：seed × bucket，按 steps 降序（长的先跑，避免最后被一条 10000 帧的拖住）
mapfile -t JOBS < <(awk '{print $1, $2, $3, $4, $5}' "$BASE/jobs.txt" | while read -r b lo hi st n; do
  for s in $SEEDS; do echo "$st $s $lo $hi $st $n"; done; done | sort -rn)
echo "[full] 作业数 ${#JOBS[@]}"

i=0
for j in "${JOBS[@]}"; do
  read -r _st seed lo hi st n <<<"$j"
  gpu=$(( i % SLOTS ))
  run_one "$seed" "$lo" "$hi" "$st" "$n" "$gpu" &
  i=$((i+1))
  [[ $(( i % SLOTS )) -eq 0 ]] && wait
done
wait
echo "[full] 全部作业完成"

# ---- 合并 ----
python3 - "$BASE" "$BASE/a_prime.pt" <<'PY'
import sys, glob, torch
from pathlib import Path
base, dst = Path(sys.argv[1]), sys.argv[2]
files = sorted(glob.glob(str(base / "*.pt")))
files = [f for f in files if not f.endswith("a_prime.pt")]
if not files:
    raise SystemExit("[full] 没有可合并的分片")
cat = {k: [] for k in ("states", "actions", "next_states", "frames", "clip_ids")}
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    for k in cat:
        cat[k].append(d[k])
m = {k: torch.cat(v, 0) for k, v in cat.items()}
m["meta"] = {"source": "run_a_prime_full_4gpu.sh", "files": [Path(f).name for f in files]}
torch.save(m, dst)
F = m["frames"]
print(f"[full] merged {len(files)} 片 -> {dst} rows={m['states'].shape[0]} "
      f"frames=({int(F.min())},{int(F.max())}) clips={len(m['clip_ids'].unique())}")
PY
