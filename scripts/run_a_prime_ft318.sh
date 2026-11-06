#!/bin/bash
# Full-coverage A' collection on the 318 dataset (segmented teacher-forced, SONIC executor).
#
#   docs/MUJOCO_SONIC_JOINTFT_PLAN_20261104.md §6
#
# 每个子数据集切成若干 episode 分片并行跑（单进程 ~350–500 控制步/s），最后 merge 成一份
# 转移文件，格式与 `build_real_transitions_mjlab.py` 一致（states/next_states/actions/frames）。
#
# 用法:
#   CHUNK=16 SEGMENTS=64 JOBS=16 bash scripts/run_a_prime_ft318.sh
#   SEGMENTS=4 EPISODES=1  bash scripts/run_a_prime_ft318.sh   # 冒烟
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT/third_party/SuperMimic-Reproduced/.venv/bin/python}"
OUTDIR="${OUTDIR:-$ROOT/runs/wm_real/ft318_seg}"
CHUNK="${CHUNK:-16}"          # 每个分片多少集
SEGMENTS="${SEGMENTS:-64}"    # 每集采几段（计划口径 64）
SEGLEN="${SEGLEN:-20}"        # 段长 = joint FT 的 H
GAINS="${GAINS:-ha}"
NOISE_RAD="${NOISE_RAD:-0.014}"
PERTURB_XY="${PERTURB_XY:-0.05}"
MAXJOBS="${MAXJOBS:-24}"
SEED="${SEED:-0}"

mkdir -p "$OUTDIR"
echo "[a'] out=$OUTDIR chunk=$CHUNK segments=$SEGMENTS seglen=$SEGLEN gains=$GAINS noise=$NOISE_RAD perturb=$PERTURB_XY"

running=0
for set_idx in 0 1 2 3 4 5 6 7 8; do
  n_eps=$("$PY" - "$set_idx" <<'PY'
import json, sys
from pathlib import Path
m = json.loads(Path("data/sonic_datasets.json").read_text())
a = m["sets"][m["active"]]
d = a["datasets"][int(sys.argv[1])]
root = Path(a["root"])
print(len(list((root / d["path"] / "data" / "chunk-000").glob("*.parquet"))))
PY
)
  start=0
  while [ "$start" -lt "$n_eps" ]; do
    end=$(( start + CHUNK - 1 ))
    [ "$end" -ge "$n_eps" ] && end=$(( n_eps - 1 ))
    eps=$("$PY" -c "print(','.join(str(i) for i in range($start,$end+1)))")
    log="$OUTDIR/set${set_idx}_ep${start}-${end}.log"
    out="$OUTDIR/set${set_idx}_ep${start}-${end}.pt"
    if [ -s "$out" ]; then
      echo "[skip] $out"
    else
      "$PY" "$ROOT/scripts/collect_seg_ft318.py" \
        --set-index "$set_idx" --episodes "$eps" \
        --segments "$SEGMENTS" --seg-len "$SEGLEN" --gains "$GAINS" \
        --action-noise-rad "$NOISE_RAD" --start-perturb-xy "$PERTURB_XY" \
        --seed $(( SEED + set_idx * 100 + start )) \
        --out "$out" > "$log" 2>&1 &
      running=$(( running + 1 ))
      if [ "$running" -ge "$MAXJOBS" ]; then wait -n; running=$(( running - 1 )); fi
    fi
    start=$(( end + 1 ))
  done
done
wait
echo "[a'] all shards done"

"$PY" - "$OUTDIR" <<'PY'
import sys, torch
from pathlib import Path

d = Path(sys.argv[1])
parts = sorted(d.glob("set*_ep*.pt"))
ps = [torch.load(p, weights_only=False) for p in parts]
ps = [p for p in ps if p["states"].shape[0] > 0]
states = torch.cat([p["states"] for p in ps])
next_states = torch.cat([p["next_states"] for p in ps])
actions = torch.cat([p["actions"] for p in ps])
frames = torch.cat([p["frames"] for p in ps])
eps = torch.cat([
    torch.full((p["states"].shape[0],), int(p["meta"]["episodes"].split(",")[0]), dtype=torch.long)
    for p in ps
])
meta = {"source": "ft318_seg_sonic_merged", "shards": len(ps), "n": int(states.shape[0]),
        "segment_len": ps[0]["meta"]["segment_len"] if ps else None,
        "action_noise_rad": ps[0]["meta"]["action_noise_rad"] if ps else None,
        "start_perturb_xy": ps[0]["meta"]["start_perturb_xy"] if ps else None}
out = d / "a_prime_ft318.pt"
torch.save({"states": states, "next_states": next_states, "actions": actions,
            "frames": frames, "episodes": eps, "meta": meta}, str(out))
print(f"[merge] {len(ps)} shards -> {out}  n={states.shape[0]}  "
      f"states={tuple(states.shape)} actions={tuple(actions.shape)}")
PY
