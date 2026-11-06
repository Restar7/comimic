#!/usr/bin/env bash
# P1c（2 卡容器）G1：G1 参考回放验收 —— 新 WBC × 执行参考库，跑完整条 clip，算 root/关节误差与摔倒。
#
# 文档：docs/PLAN_REFSTART_WM_PERTURB_20261102.md §6.3
# 基线（同一条 DoubleDesk clip，旧 WBC × ha_cmd，online token 臂）：root xy 0.321 m、1327 步摔
#   —— 用 scripts/eval_g1_replay.py 在同一口径下复现过。
#
# 两臂：
#   online : token 由**参考库行 + 实时机器人 root quat**在线算（hold 窗口）→ 与训练完全同一条路，
#            也是上面 0.32 m 基线的口径（G1 主判据）。
#   npy    : token = 录制时的命令流 label（encoder_latent / g1_motion_token_64）→ **部署口径**，
#            anchor 冻在录制时的机器人朝向。
#
# 用法：
#   CLIPS=0 ARMS=online bash scripts/p1c_g1_replay.sh            # 单条 DoubleDesk 冒烟（~4 min）
#   CLIPS=0,1,2 ARMS="online npy" bash scripts/p1c_g1_replay.sh  # 多条 × 两臂
#   TASKS="DoubleDesk PPBox Boxing" bash scripts/p1c_g1_replay.sh
#
# 环境变量：SM_CKPT（默认 runs/S1_handover/exp_WBC_exec.pt）、SM_PACKED、ISAAC_GPU、RECORD_VIDEO、
#          MAX_CLIP_STEPS（0=整条）、TAG（结果目录后缀）
set -uo pipefail

C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$C" || exit 1

SM_CKPT=${SM_CKPT:-$C/runs/S1_handover/exp_WBC_exec.pt}
SM_PACKED=${SM_PACKED:-$C/runs/isaac_motion/ha_cmd_exec/packed.pt}
ARMS=${ARMS:-"online npy"}
CLIPS=${CLIPS:-0}
TASKS=${TASKS:-}
ISAAC_GPU=${ISAAC_GPU:-0}
RECORD_VIDEO=${RECORD_VIDEO:-0}
MAX_CLIP_STEPS=${MAX_CLIP_STEPS:-0}
TAG=${TAG:-}
OUT=${OUT:-$C/runs/exp_g1${TAG:+_$TAG}}
TOKEN_DIR=${TOKEN_DIR:-$C/runs/exp_tokens}
HAND=${SM_HAND:-1}
# online 臂的 token 库：默认 = 参考库；双库 ckpt（ref=执行、token=命令）要显式给 ha_cmd。
TOKEN_LIB=${TOKEN_LIB:-$SM_PACKED}

[[ -f "$SM_CKPT" ]] || { echo "FATAL: SM_CKPT 不存在: $SM_CKPT"; exit 1; }
[[ -f "$SM_PACKED" ]] || { echo "FATAL: SM_PACKED 不存在: $SM_PACKED"; exit 1; }
[[ -f "$TOKEN_LIB" ]] || { echo "FATAL: TOKEN_LIB 不存在: $TOKEN_LIB"; exit 1; }
mkdir -p "$OUT/dump" "$OUT/isaac" "$TOKEN_DIR" "$C/logs/p1c"

# clip 名前缀 -> HA 闭环任务名
task_of() {
  case "$1" in
    HOI_double_desk*) echo DoubleDesk ;;
    HOI_pp_box*)      echo PPBox ;;
    HSI_boxing*)      echo Boxing ;;
    HSI_open_door*)   echo OpenDoor ;;
    HSI_sit_sofa*)    echo SitSofa ;;
    HSI_vision_navi*) echo VisionNavi ;;
    *) echo "" ;;
  esac
}

# 选 clip：--clips 索引或 --tasks（每个任务取第一条）
mapfile -t PICK < <(python3 - "$SM_PACKED" "$CLIPS" "$TASKS" <<'PY'
import sys, torch
packed, clips, tasks = sys.argv[1], sys.argv[2], sys.argv[3]
d = torch.load(packed, map_location="cpu", weights_only=False)
names = list(d["names"]); starts = [int(x) for x in d["starts"]]; lens = [int(x) for x in d["lengths"]]
idx = []
if tasks.strip():
    want = ["HOI_double_desk" if t == "DoubleDesk" else
            "HOI_pp_box" if t == "PPBox" else
            "HSI_boxing" if t == "Boxing" else
            "HSI_open_door" if t == "OpenDoor" else
            "HSI_sit_sofa" if t == "SitSofa" else
            "HSI_vision_navi" if t == "VisionNavi" else t for t in tasks.split()]
    for w in want:
        idx.append(next(i for i, n in enumerate(names) if n.startswith(w)))
else:
    idx = [int(x) for x in clips.split(",") if x.strip()]
for i in idx:
    print(f"{i}\t{names[i]}\t{starts[i]}\t{lens[i]}")
PY
)
[[ ${#PICK[@]} -gt 0 ]] || { echo "FATAL: 没选中 clip"; exit 1; }

echo "[p1c] ckpt=$SM_CKPT"
echo "[p1c] packed=$SM_PACKED  token_lib=$TOKEN_LIB"
echo "[p1c] arms='$ARMS' clips=${#PICK[@]} out=$OUT"

summaries=()
for row in "${PICK[@]}"; do
  IFS=$'\t' read -r IDX CLIP START LEN <<<"$row"
  TASK=$(task_of "$CLIP")
  if [[ -z "$TASK" ]]; then echo "[p1c] skip $CLIP（本 checkout 没有对应任务/资产）"; continue; fi
  STEPS=$LEN
  if [[ "$MAX_CLIP_STEPS" != "0" && "$MAX_CLIP_STEPS" -lt "$LEN" ]]; then STEPS=$MAX_CLIP_STEPS; fi

  # label token（命令流 encoder_latent）—— npy 臂用；与 data/npz ha_tokens 的 g1_motion_token_64 逐位相同
  NPY=$TOKEN_DIR/$CLIP.npy
  if [[ ! -f "$NPY" ]]; then
    python3 - "$CLIP" "$NPY" <<'PY'
import sys, numpy as np
from pathlib import Path
clip, out = sys.argv[1], sys.argv[2]
raw = sorted(Path("data/ha_raw").glob("*/*/*.npz"))
hit = [p for p in raw if p.stem.rsplit("_", 1)[-1] == clip.rsplit("ep", 1)[-1] and p.parent.parent.name == clip.split("__")[0]]
if not hit:
    raise SystemExit(f"no raw npz for {clip}")
z = np.load(hit[0], allow_pickle=True)
tok = np.asarray(z["encoder_latent"], dtype=np.float32)
np.save(out, tok)
print(f"[p1c] label tokens {out} {tok.shape}")
PY
  fi

  for ARM in $ARMS; do
    RUN_ID=${TASK}__clip${IDX}_${ARM}
    RES_DIR=$OUT/isaac/$RUN_ID
    mkdir -p "$RES_DIR"
    echo "==================== $RUN_ID ($CLIP) start=$START steps=$STEPS arm=$ARM"
    if [[ -f "$RES_DIR/result.json" ]]; then echo "[p1c] skip（已有 result.json）"; else
      DUMP_PREFIX=$OUT/dump/${RUN_ID}
      # ★ 2026-11-03 真 bug：原来**没有把 TASK 传下去** ⇒ ha_isaac_closedloop.sh 用默认
      #   TASK=DoubleDesk，于是**所有 clip（VisionNavi/Boxing/PPBox/OpenDoor/SitSofa）都被
      #   放在 DoubleDesk 场景里跑**。场景/物体全错，误差自然大，replay 也会摔。
      TASK="$TASK" \
      SM_CKPT="$SM_CKPT" \
      SM_TOKEN_SOURCE="$ARM" \
      SM_PACKED="$SM_PACKED" \
      SM_TOKEN_LIB="$TOKEN_LIB" \
      SM_TOKEN_START="$START" \
      SM_TOKEN_NPY="$NPY" \
      SM_START_FRAME="$START" \
      SM_INIT_FROM_LIB=1 \
      SM_HAND="$HAND" \
      MAX_STEPS="$STEPS" \
      RECORD_VIDEO="$RECORD_VIDEO" \
      ISAAC_GPU="$ISAAC_GPU" \
      OUT_ROOT="$RES_DIR" \
      SM_DUMP_STATES="$DUMP_PREFIX" \
        bash scripts/ha_isaac_closedloop.sh > "$OUT/logs_${RUN_ID}.log" 2>&1
      echo "[p1c] closedloop exit=$? (log $OUT/logs_${RUN_ID}.log)"
    fi
    python3 scripts/eval_g1_replay.py \
      --dump "$OUT/dump/${RUN_ID}_ep*.pt" \
      --packed "$SM_PACKED" \
      --start-row "$START" \
      --result "$RES_DIR/*/result.json" \
      --tag "$RUN_ID" \
      --out "$OUT/metrics_${RUN_ID}.json" | tee -a "$OUT/metrics.txt"
  done
done

# ---- 汇总 ---------------------------------------------------------------
python3 - "$OUT" "$SM_CKPT" "$SM_PACKED" <<'PY'
import json, sys, glob
from pathlib import Path
out, ckpt, packed = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
runs = []
for f in sorted(out.glob("metrics_*.json")):
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    for r in d.get("reports", []):
        r["run_id"] = f.stem.replace("metrics_", "")
        runs.append(r)
verdicts = {}
for r in runs:
    verdicts[r["run_id"]] = ("PASS" if all(v is not False for v in r["gates"].values()) else "FAIL")
report = {
    "gate": {"root_xy_mean_m": 0.10, "joint_rmse_mean_rad": 0.20, "falls": 0, "walked_full_clip": True},
    "baseline_old_wbc_ha_cmd_online": {"clip": "HOI_double_desk__tw__ep1776268633456387",
                                       "root_xy_err_m": 0.321, "joint_rmse_rad": 0.420,
                                       "steps": 1327, "fall": True},
    "ckpt": ckpt, "packed": packed,
    "runs": runs, "verdicts": verdicts,
}
(out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
print("\n[p1c] ===== G1 summary =====")
for r in runs:
    print(f"  {r['run_id']:<26} steps {r['steps']:>5}/{r['clip_len']:<5} "
          f"root_xy {r['root_xy_err_m']['mean']:.3f} m  joint_rmse {r['joint_rmse_rad']['mean']:.3f} rad  "
          f"fall={r['fall']}  -> {verdicts[r['run_id']]}")
print(f"[p1c] report -> {out / 'report.json'}")
PY
