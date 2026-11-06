#!/usr/bin/env bash
# A' 重采（2026-11-05）：**逐步独立动作扰动**，把 WM 里的"动作=意图提示"打掉。
#
# 与旧数据（runs/a_prime_sonic_leash）的**唯一差别**
#   action_noise_std 0.05 → 0.15      （幅度要对上伪方向的范数 ~0.18）
#   perturb_hold     20   → 1         （噪声从"20 步常数"变成**白噪声**）
#   leash δ          0.10 → 0.15      （噪声更大，放宽重置阈值保覆盖）
# 其余一切不变：同一个 SONIC release decoder、`--exec-steps 1`、同一套契约门。
#
# 为什么 hold=1 是关键（docs/WHY_FT_DEGRADED_20261104.md §10）：
#   旧噪声在 20 步内是常数 ⇒ 它和"意图偏移"一样光滑 ⇒ WM 能把它当意图吸收掉，伪通道没被破坏。
#   白噪声逐帧跳变、无法被"光滑意图"解释，而 s' 又确实随它变化 ⇒ WM 只能把 a 当因果控制量。
#
# 用法（两组并行，墙钟 ≈13 min；单组串行 ≈24 min）：
#   bash scripts/run_a_prime_sonic_exc.sh            # 默认两组并行
#   PARALLEL=0 bash scripts/run_a_prime_sonic_exc.sh # 串行（旧行为）
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$C" || exit 1

export OUT_DIR=${OUT_DIR:-$C/runs/a_prime_sonic_exc}
export NOISE=${NOISE:-0.15}
export HOLD=${HOLD:-1}
export RESET_MAX_DRIFT=${RESET_MAX_DRIFT:-0.15}
export WINDOW=${WINDOW:-260}
export STEPS=${STEPS:-780}
export PER=${PER:-16}
export SEG=0
PARALLEL=${PARALLEL:-1}

echo "[exc] OUT_DIR=$OUT_DIR  noise=$NOISE  hold=$HOLD  leash=$RESET_MAX_DRIFT  parallel=$PARALLEL"
mkdir -p "$OUT_DIR"

if [[ "$PARALLEL" == "1" ]]; then
  # 两组并行：各 2 张卡、各 3 个任务。驱动脚本内部按任务串行，所以这样能省掉一半冷启动空档。
  TASKS="DoubleDesk PPBox Boxing"    GPUS="0 1" bash scripts/run_a_prime_sonic_seg20.sh \
      > "$OUT_DIR/groupA.log" 2>&1 &
  P1=$!
  TASKS="OpenDoor SitSofa VisionNavi" GPUS="2 3" bash scripts/run_a_prime_sonic_seg20.sh \
      > "$OUT_DIR/groupB.log" 2>&1 &
  P2=$!
  echo "[exc] groupA pid=$P1  groupB pid=$P2"
  wait $P1; RA=$?
  wait $P2; RB=$?
  echo "[exc] groupA rc=$RA  groupB rc=$RB"
  [[ "$RA" != "0" || "$RB" != "0" ]] && exit 4
else
  bash scripts/run_a_prime_sonic_seg20.sh > "$OUT_DIR/all.log" 2>&1 || exit 4
fi

# ---- 产物校验（rc=0 不代表成功，必须看文件 + meta）-----------------------------
python3 - "$OUT_DIR" <<'PY'
import glob, sys, torch
d = sys.argv[1]
files = sorted(glob.glob(f"{d}/*_off*.pt"))
if not files:
    print(f"[exc][FAIL] {d} 没有产物"); sys.exit(1)
rows = 0
for f in files:
    z = torch.load(f, map_location="cpu", weights_only=False)
    m = z.get("meta", {})
    rows += int(z["states"].shape[0])
    bad = []
    if float(m.get("action_noise_std", -1)) <= 0: bad.append("noise")
    if int(m.get("perturb_hold", -1)) != 1: bad.append(f"hold={m.get('perturb_hold')}")
    if abs(float(m.get("reset_max_drift", -1)) - 0.15) > 1e-6: bad.append("leash")
    if bad:
        print(f"[exc][FAIL] {f} meta 不符：{bad}"); sys.exit(1)
print(f"[exc][OK] {len(files)} 个 shard，共 {rows} 条转移，meta 全部 = (noise>0, hold=1, leash=0.15)")
PY
[[ $? -ne 0 ]] && exit 5
echo "[exc] 完成 → $OUT_DIR"
