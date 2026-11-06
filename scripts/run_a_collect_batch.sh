#!/usr/bin/env bash
# ============================================================================
# A 线批量采集启动器：**只跑冻结 WBC、不进 VLA**，在 Isaac 真物理里向量化走，
# 产出 ~10^6 条真实 (s, a, s', 全局库行号) —— 补上 mjlab 那条线的「第 1 步」。
#
# 为什么必须用 $ISAAC_VENV/bin/python（不要 isaaclab.sh）：TERM=dumb 下 isaaclab.sh 会失败；
# 为什么 unset CUDA_VISIBLE_DEVICES：不改可见性，让 Isaac 自己选卡（见 run_collect_onpolicy.sh）。
#
# 用法：
#   bash scripts/run_a_collect_batch.sh                 # 默认 4 波 × 512 env × 300 步 ≈ 61 万条
#   ENVS=1024 STEPS=300 WAVES=3 bash scripts/run_a_collect_batch.sh   # ≈ 92 万条
#   SMOKE=1 bash scripts/run_a_collect_batch.sh          # 16 env × 20 步，验证链路
#   SEED=1 WAVES=1 bash scripts/run_a_collect_batch.sh   # 换种子只跑一波
#
# 产出：runs/wm_real/a_w{0..WAVES-1}.pt（每个文件 ENVS×STEPS 条）
# 奖励：跑完用系统 python 补
#   python3 scripts/compute_collect_reward.py --in 'runs/wm_real/a_w*.pt' \
#       --out runs/wm_real/a_rew.pt
# ============================================================================
set -uo pipefail

COMIMIC=${COMIMIC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$COMIMIC" || exit 1

ENVS=${ENVS:-512}
STEPS=${STEPS:-300}
WAVES=${WAVES:-4}
WAVE0=${WAVE0:-0}          # 起始波号（避免覆盖已有的 a_w*.pt）
SEED=${SEED:-20261029}
START_MODE=${START_MODE:-uniform}
WARMUP=${WARMUP:-20}
OUTDIR=${OUTDIR:-$COMIMIC/runs/wm_real}
WBC=${WBC:-$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt}
MAX_CLIPS=${MAX_CLIPS:-0}

if [[ "${SMOKE:-0}" == "1" ]]; then
  ENVS=16; STEPS=20; WAVES=1; WARMUP=5
fi

mkdir -p "$OUTDIR" "$COMIMIC/logs"
say() { echo "[$(date +%H:%M:%S)] [a-line] $*"; }

source scripts/isaac_env.sh >/dev/null 2>&1
unset CUDA_VISIBLE_DEVICES
export TERM=xterm
export PYTHONPATH="$COMIMIC:$COMIMIC/trackerlab-supermimic/source/trackerTask:$COMIMIC/trackerlab-supermimic/source/trackerLab:${PYTHONPATH:-}"

[[ -x "$ISAAC_VENV/bin/python" ]] || { echo "❌ 找不到 \$ISAAC_VENV/bin/python（$ISAAC_VENV）" >&2; exit 1; }

say "python=$ISAAC_VENV/bin/python  envs=$ENVS steps=$STEPS waves=$WAVES ⇒ 预计 $((ENVS*STEPS*WAVES)) 条"
ok=0
for w in $(seq "$WAVE0" $((WAVE0+WAVES-1))); do
  out=$(printf '%s/a_w%d.pt' "$OUTDIR" "$w")
  seed_w=$((SEED + w))
  say "wave $w  seed=$seed_w out=$out"
  "$ISAAC_VENV/bin/python" -u scripts/isaac_a_collect_batch.py \
      --wbc-ckpt "$WBC" --num-envs "$ENVS" --steps "$STEPS" \
      --start-mode "$START_MODE" --seed "$seed_w" --warmup "$WARMUP" \
      --max-clips "$MAX_CLIPS" --log-every 100 --headless \
      --out "$out" 2>&1 | tee "$COMIMIC/logs/a_collect_w$w.log" | tail -3
  rc=${PIPESTATUS[0]}
  if [[ "$rc" != "0" || ! -s "$out" ]]; then
    say "❌ wave $w 失败 rc=$rc（看 logs/a_collect_w$w.log）"
    exit 1
  fi
  ok=$((ok+1))
done

say "完成 $ok/$WAVES 波"
"$ISAAC_VENV/bin/python" - "$OUTDIR" "$WAVES" <<'PY'
import sys, torch
from pathlib import Path
d, n = Path(sys.argv[1]), int(sys.argv[2])
tot = 0
for w in range(n):
    p = d / f"a_w{w}.pt"
    if not p.exists():
        continue
    x = torch.load(str(p), map_location="cpu", weights_only=False)
    k = int(x["states"].shape[0])
    tot += k
    print(f"  {p.name}: n={k} clips={x['meta'].get('n_clips_used')} "
          f"frames[{int(x['frames'].min())},{int(x['frames'].max())}]")
print(f"  合计 {tot} 条真实转移")
PY
say "下一步：python3 scripts/compute_collect_reward.py --in '$OUTDIR/a_w*.pt' --out $OUTDIR/a_rew.pt"
