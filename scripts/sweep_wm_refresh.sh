#!/usr/bin/env bash
# WM 刷新多 seed 扫描 + **按 held-out 指标选**（2026-11-03）
#
# 为什么需要：`refresh_wm` 的 minibatch 采样用的是没播种的 generator，同一份数据
# 每跑一次结果都不同（同一评测集实测 qd_corr 0.6735 / 0.5089，差值比"改进"还大）。
# 只跑一次然后声称"刷新有效/无效"是错的。正确做法：跑 N 个 seed，
# 用**同一份 held-out 评测集（固定 seed、大 batch）**打分，取最好的那个。
#
# 用法：
#   bash scripts/sweep_wm_refresh.sh                 # 4 个 seed
#   SEEDS="0 1 2 3 4 5 6 7" bash scripts/sweep_wm_refresh.sh
#   WM_STEPS=6000 bash scripts/sweep_wm_refresh.sh
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
DIR=${DIR:-$C/runs/a_prime_seg20}
WM_STEPS=${WM_STEPS:-6000}
SEEDS=${SEEDS:-"0 1 2 3"}
JOINT_CKPT=${JOINT_CKPT:-$C/runs/S1_handover/WBC_A24000_wmvel_cmd.pt}
GPU=${GPU:-0}
OUT_JSON=${OUT_JSON:-$DIR/wm_sweep.json}
cd "$C" || exit 1

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export SUPERMIMIC_HA_ALIGN=1
export HF_HOME=${HF_HOME:-/inspire/hdd/global_user/liumengfan-253108110079/cache/huggingface}
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

declare -a PRODUCED=()
for s in $SEEDS; do
  out="$DIR/wm_pre_s${s}.pt"
  echo "=== [sweep] seed=$s steps=$WM_STEPS -> $out ==="
  CUDA_VISIBLE_DEVICES=$GPU python3 scripts/pretrain_wm.py \
      --joint-ckpt "$JOINT_CKPT" --transitions "$DIR/filtered.pt" \
      --steps "$WM_STEPS" --no-stage1 --stage3-steps 0 \
      --seed "$s" --device cuda --out "$out" 2>&1 \
    | grep -aE "\[wm-refresh [0-9]+/|AUDIT\]\[wm-refresh\]" | tail -2
  if [[ ! -f "$out" ]]; then echo "[sweep][fail] seed=$s 没有产物"; continue; fi
  # pretrain_wm --no-stage1 不写 robot_spec，拷一份再评（eval/部署都需要）
  python3 - "$out" "$JOINT_CKPT" <<'PY'
import sys, torch
out, base = sys.argv[1], sys.argv[2]
w = torch.load(out, map_location="cpu", weights_only=False)
b = torch.load(base, map_location="cpu", weights_only=False)
if "robot_spec" not in w and "robot_spec" in b:
    w["robot_spec"] = b["robot_spec"]
    torch.save(w, out)
    print(f"[sweep] graft robot_spec -> {out}")
PY
  PRODUCED+=("$out")
done

echo "=== [sweep] 统一评测（固定 seed / batch=65536）==="
python3 scripts/eval_wm_refresh_ab.py \
    --data "$DIR/filtered.pt" \
    --ckpts "$JOINT_CKPT" "$DIR/wm_pre_full_583clips.pt" "${PRODUCED[@]}" \
    --batch 65536 --out "$OUT_JSON" 2>&1 | grep -aE "^\[ab\]"

echo "=== [sweep] 选优（qd_corr 最高，qd_ratio 越接近 1 越好）==="
python3 - "$OUT_JSON" "$DIR" <<'PY'
import json, sys, shutil, torch
from pathlib import Path
j = json.loads(Path(sys.argv[1]).read_text()); d = Path(sys.argv[2])
rows = [r for r in j["results"] if r["ckpt"] != "WBC_A24000_wmvel_cmd.pt"]
if not rows:
    raise SystemExit("[sweep][fail] 没有候选")
best = max(rows, key=lambda r: r["qd_corr"])
src = d / best["ckpt"]
if src.exists():
    shutil.copy(src, d / "wm_pre_full.pt")
    print(f"[sweep] best = {best['ckpt']} qd_corr={best['qd_corr']:.4f} "
          f"qd_ratio={best['qd_ratio']:.4f}  -> wm_pre_full.pt")
for r in j["results"]:
    print(f"  {r['ckpt']:34s} qd_corr={r['qd_corr']:.4f} qd_ratio={r['qd_ratio']:.4f} "
          f"q_mse={r['q_mse']:.3e} pos_mse={r['pos_mse']:.3e}")
PY
