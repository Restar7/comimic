#!/usr/bin/env bash
# 4×4090：A' 全量（699 clip）4 卡分片并行 + 合并
#
# 每张卡一个 shard（SHARD=i/NSHARD=4），每卡内部 JOBS 个 Isaac 进程。
# 用法：
#   ROUND=a_full SEEDS="0 1" JOBS=6 \
#   WBC_CKPT=$PWD/runs/S1_handover/exp_WBC_exec_haonly_best.pt \
#   bash scripts/run_a_prime_4gpu.sh
#
# 时间（估算）：699 clip × 2 seed × 1 env，4 卡 × 6 进程 ≈ 4–6 h。
set -uo pipefail
COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1

ROUND=${ROUND:-a_full}
SEEDS=${SEEDS:-"0 1"}
NUM_ENVS=${NUM_ENVS:-1}
JOBS=${JOBS:-6}            # 每卡并发 Isaac 进程
SHARDS=${SHARDS:-4}
WBC_CKPT=${WBC_CKPT:-$COMIMIC/runs/S1_handover/exp_WBC_exec_haonly_best.pt}
PACKED=${PACKED:-$COMIMIC/runs/isaac_motion/ha_cmd_exec/packed.pt}
TOKENS_DIR=${TOKENS_DIR:-$COMIMIC/runs/exp_tokens}
BASE=${BASE:-$COMIMIC/runs/wm_real/exp_a_prime_all/$ROUND}
mkdir -p "$BASE"
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}

echo "[a4] round=$ROUND seeds=[$SEEDS] envs=$NUM_ENVS jobs/gpu=$JOBS shards=$SHARDS"
echo "[a4] wbc=$WBC_CKPT packed=$PACKED base=$BASE"

pids=()
for i in $(seq 0 $((SHARDS - 1))); do
  echo "[a4] launch shard $i on GPU $i"
  CUDA_VISIBLE_DEVICES=$i \
  ROUND=$ROUND SHARD=$i NSHARD=$SHARDS JOBS=$JOBS SEEDS="$SEEDS" NUM_ENVS="$NUM_ENVS" \
  WBC_CKPT="$WBC_CKPT" PACKED="$PACKED" TOKENS_DIR="$TOKENS_DIR" \
  OUT_ROOT="$BASE/shard$i" \
  nohup bash scripts/run_a_prime_all.sh > "$BASE/shard$i.log" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done
echo "[a4] shards done fail=$fail"

python3 - "$BASE" <<'PY'
import sys, glob, torch
from pathlib import Path
base = Path(sys.argv[1])
files = sorted(glob.glob(str(base / "shard*" / "a_prime.pt")))
if not files:
    raise SystemExit(f"[a4] no shard a_prime.pt under {base}")
cat = {k: [] for k in ("states", "actions", "next_states", "frames")}
clips = []
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    for k in cat:
        cat[k].append(d[k])
    clips += list(d.get("clips") or [])
out = {k: torch.cat(v, 0) for k, v in cat.items()}
out["clips"] = clips
dst = base / "a_prime.pt"
torch.save(out, str(dst))
print(f"[a4] merged {len(files)} shards / {len(clips)} clips -> {dst} rows={out['states'].shape[0]}")
PY
echo "[a4] done -> $BASE/a_prime.pt"
