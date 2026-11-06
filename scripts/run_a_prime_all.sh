#!/usr/bin/env bash
# 699 条 clip 的 A' 全量并行采集（每条 clip 一个 Isaac 进程；输出 exp_ 前缀）
#
# 用法：
#   ROUND=a_full SEEDS="0 1" NUM_ENVS=1 JOBS=6 \
#   WBC_CKPT=$PWD/runs/S1_handover/exp_WBC_exec_haonly_best.pt \
#   bash scripts/run_a_prime_all.sh
#
# 说明：
#   - 每条 clip 独立进程（Isaac 每进程 ~4–6 GB；JOBS=6 约 30 GB，1×4090 48GB 可跑）；
#   - NUM_ENVS=1：一个进程只跑 1 条轨迹；噪声多样性靠 SEEDS（多 seed）× 多进程（多 clip）；
#   - 如果要每 clip 8 个 env 的噪声副本，NUM_ENVS=8，但 JOBS 要降到 1–2；
#   - 产物：runs/wm_real/exp_a_prime_all/<ROUND>/a_prime/<clip>/merged.pt
#           全局合并 a_prime.pt（给 compute_collect_reward）
set -uo pipefail
COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1

ROUND=${ROUND:-a_full}
SEEDS=${SEEDS:-"0 1"}
NUM_ENVS=${NUM_ENVS:-1}
JOBS=${JOBS:-6}
SHARD=${SHARD:-0}
NSHARD=${NSHARD:-1}
WBC_CKPT=${WBC_CKPT:-$COMIMIC/runs/S1_handover/exp_WBC_exec_haonly_best.pt}
PACKED=${PACKED:-$COMIMIC/runs/isaac_motion/ha_cmd_exec/packed.pt}
TOKENS_DIR=${TOKENS_DIR:-$COMIMIC/runs/exp_tokens}
OUT_ROOT=${OUT_ROOT:-$COMIMIC/runs/wm_real/exp_a_prime_all/$ROUND}
CLIPS_FILE=$OUT_ROOT/clips.txt
mkdir -p "$OUT_ROOT/a_prime" "$OUT_ROOT/logs"

echo "[a-all] round=$ROUND seeds=[$SEEDS] envs=$NUM_ENVS jobs=$JOBS shard=$SHARD/$NSHARD"
echo "[a-all] wbc=$WBC_CKPT packed=$PACKED"
echo "[a-all] out=$OUT_ROOT"

python3 - "$PACKED" "$TOKENS_DIR" "$CLIPS_FILE" <<'PY'
import sys, torch
from pathlib import Path
packed, tokdir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
p = torch.load(packed, map_location="cpu", weights_only=False)
rows = [(str(n), int(L)) for n, L in zip(p["names"], p["lengths"]) if (tokdir / f"{n}.npy").exists()]
import os
_sh, _ns = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARD", "1"))
if _ns > 1:
    rows = rows[_sh::_ns]
out.write_text("\n".join(f"{n}\t{L}" for n, L in rows) + "\n")
print(f"[a-all] clips with token npy: {len(rows)}")
PY

export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}
export COMIMIC

# xargs：每条 clip 一个进程；run_a_prime_seeds.sh 内部再按 SEEDS 串行
export SEEDS NUM_ENVS WBC_CKPT PACKED TOKENS_DIR OUT_ROOT
cat "$CLIPS_FILE" | xargs -P "$JOBS" -n 2 bash -c '
  CLIP="$1"; LEN="$2"
  [ -z "$CLIP" ] && exit 0
  OUT="$OUT_ROOT/a_prime/$CLIP"; LOG="$OUT_ROOT/logs/$CLIP.log"
  mkdir -p "$OUT"
  SEEDS="$SEEDS" STEPS="$LEN" NUM_ENVS="$NUM_ENVS" \
    CLIP="$CLIP" TOKEN_NPY="$TOKENS_DIR/$CLIP.npy" \
    WBC="$WBC_CKPT" PACKED="$PACKED" OUT_DIR="$OUT" \
    bash "$COMIMIC/scripts/run_a_prime_seeds.sh" > "$LOG" 2>&1
  rc=$?
  echo "[a-all] $CLIP rc=$rc"
' _
rc=$?
echo "[a-all] xargs rc=$rc"

echo "[a-all] 全局合并 -> $OUT_ROOT/a_prime.pt"
python3 - "$OUT_ROOT/a_prime" "$OUT_ROOT/a_prime.pt" <<'PY'
import sys, glob, torch
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(glob.glob(str(src / "*" / "merged.pt")))
if not files:
    raise SystemExit(f"[a-all] no merged.pt under {src}")
cat = {k: [] for k in ("states", "actions", "next_states", "frames")}
clips = []
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    for k in cat:
        cat[k].append(d[k])
    clips.append(Path(f).parent.name)
out = {k: torch.cat(v, 0) for k, v in cat.items()}
out["clips"] = clips
torch.save(out, str(dst))
print(f"[a-all] merged {len(files)} clips -> {dst} rows={out['states'].shape[0]}")
PY
echo "[a-all] done -> $OUT_ROOT"
