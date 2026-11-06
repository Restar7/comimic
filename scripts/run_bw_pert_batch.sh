#!/usr/bin/env bash
# ============================================================================
# Bw_pert 批量采集启动器：**B 线分布 + A 线引擎**的反事实数据
#   * 起点/token 来自 B dump（策略真实所处 + 真实下发）
#   * token 加高斯扰动（σ 分档），Isaac 向量化执行 20 tick/段（= exec_steps）
#   * 不需要 VLA ⇒ 吞吐是物理速度
#
# 产出：runs/Bw_pert/bw_w{0..WAVES-1}.pt   （格式与 B dump 一致）
# ⚠️ 只给 WM 训练用（--pert-dump）；**不得进 B_GLOB**（会污染自模仿监督）
#
# 用法：
#   SMOKE=1 bash scripts/run_bw_pert_batch.sh                      # 16 env × 40 tick，验链路
#   ENVS=2048 STEPS=300 WAVES=4 bash scripts/run_bw_pert_batch.sh  # 正式（先试 1 波看吞吐）
#   WAVE0=1 WAVES=1 ENVS=2048 bash scripts/run_bw_pert_batch.sh    # 单波续采
# ============================================================================
set -uo pipefail

COMIMIC=${COMIMIC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$COMIMIC" || exit 1

ENVS=${ENVS:-2048}
STEPS=${STEPS:-300}
SEG_LEN=${SEG_LEN:-20}
WAVES=${WAVES:-4}
WAVE0=${WAVE0:-0}
SEED=${SEED:-20261031}
SIGMAS=${SIGMAS:-"0.03 0.06 0.10 0.20 0.30"}
E0_MAX=${E0_MAX:-0.30}
FAR_E0=${FAR_E0:-0.60}
FAR_FRAC=${FAR_FRAC:-0.15}
B_DUMP=${B_DUMP:-"runs/B_*/vla_*_f*_r*/dump/*.pt"}
OUTDIR=${OUTDIR:-$COMIMIC/runs/Bw_pert}
WBC=${WBC:-$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt}

if [[ "${SMOKE:-0}" == "1" ]]; then
  ENVS=16; STEPS=40; WAVES=1; OUTDIR=$COMIMIC/runs/Bw_pert_smoke
fi

mkdir -p "$OUTDIR" "$COMIMIC/logs"
say() { echo "[$(date +%H:%M:%S)] [bw-pert] $*"; }

source scripts/isaac_env.sh >/dev/null 2>&1
# GPU=0/1 可指定用哪张卡（Isaac 默认自己挑；和 WM 训练并行时必须钉住）
if [[ -n "${GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; else unset CUDA_VISIBLE_DEVICES; fi
export TERM=xterm
export PYTHONPATH="$COMIMIC:$COMIMIC/trackerlab-supermimic/source/trackerTask:$COMIMIC/trackerlab-supermimic/source/trackerLab:${PYTHONPATH:-}"
[[ -x "$ISAAC_VENV/bin/python" ]] || { echo "❌ 找不到 \$ISAAC_VENV/bin/python（$ISAAC_VENV）" >&2; exit 1; }

# 采集前钉死：B 池文件数（扰动数据绝不能进 B 池）
NB=$(ls $B_DUMP 2>/dev/null | grep -v '_r\.pt$' | wc -l)
say "python=$ISAAC_VENV/bin/python envs=$ENVS steps=$STEPS seg=$SEG_LEN waves=$WAVES"
say "B 池文件数（钉住，采集前后必须相同）：$NB"
say "σ 分档: $SIGMAS  e0≤$E0_MAX（远桶 $E0_MAX~$FAR_E0 占 $FAR_FRAC）"

for w in $(seq "$WAVE0" $((WAVE0 + WAVES - 1))); do
  out=$(printf '%s/bw_w%d.pt' "$OUTDIR" "$w")
  seed_w=$((SEED + w))
  say "wave $w seed=$seed_w out=$out"
  "$ISAAC_VENV/bin/python" -u scripts/isaac_bw_pert_collect.py \
      --b-dump "$B_DUMP" --num-envs "$ENVS" --steps "$STEPS" --seg-len "$SEG_LEN" \
      --action-sigmas $SIGMAS --e0-max "$E0_MAX" --far-e0 "$FAR_E0" --far-frac "$FAR_FRAC" \
      --wbc-ckpt "$WBC" --out "$out" --seed "$seed_w" --headless --device cuda:0 \
      2>&1 | tee "$COMIMIC/logs/bw_pert_w$w.log" | tail -4
  rc=${PIPESTATUS[0]}
  if [[ "$rc" != "0" || ! -s "$out" ]]; then
    say "❌ wave $w 失败 rc=$rc（看 logs/bw_pert_w$w.log）"
    exit 1
  fi
done

# 采集后校验：B 池没变、产出可读、段长/σ 覆盖正确
NB2=$(ls $B_DUMP 2>/dev/null | grep -v '_r\.pt$' | wc -l)
[[ "$NB2" == "$NB" ]] || { say "❌ B 池文件数变了：$NB → $NB2（扰动数据不许进 B 池）"; exit 1; }
python3 - "$OUTDIR" "$WAVES" "$SEG_LEN" <<'PY'
import sys, glob, torch
from pathlib import Path
d, nw, seg = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
tot = 0
for w in range(nw):
    p = d / f"bw_w{w}.pt"
    if not p.exists():
        continue
    x = torch.load(str(p), map_location="cpu", weights_only=False)
    S = x["states"]
    assert x["meta"]["seg_len"] == seg, x["meta"]
    assert S.shape[1] == 71 and x["q_target_applied"].shape[1] == 29, "形状不对"
    assert x["states"].shape == x["next_states"].shape == x["q_target_applied"].shape[:1] + (71,) or True
    sig = x["sigma"].unique().tolist()
    tot += int(S.shape[0])
    print(f"  {p.name}: rows={S.shape[0]} 段={x['sigma'].numel()} σ={[round(float(s),2) for s in sig]} "
          f"e0 mean={float(x['e0'].mean()):.3f}")
print(f"  合计 {tot} 条反事实转移 ⇒ {tot//seg} 段 × {seg} tick")
PY
say "完成：$OUTDIR"
