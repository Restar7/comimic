#!/usr/bin/env bash
# A1（有无联合微调）— **纯 HA + IsaacLab/HumanoidArena** 正式口径。
#
# ⚠️ 两条路线（写死，见 docs/DATA_POLICY_HA_ONLY.md §0.5）：
#   L1 官方线 : 官方监督 VLA → SONIC decoder → 动作
#               （decoder = GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx，
#                 真权重；runs/sonic_decoder_final.pt 是蒸馏替代，仅离线对照 —— 见 docs/SONIC_DECODER_FOUND.md）
#   L2 联合线 : 联合微调 VLA → SuperMimic WBC → 动作
# ⚠️ 联合微调的 VLA **只连 SuperMimic WBC**，配 SONIC decoder 是错的（token 空间不同，从未这样设计）。
# 本脚本是 **L2 内部对比**：base=官方监督 VLA(走 WBC) vs ft=联合微调 VLA(走 WBC)，两臂同一执行器。
# L1 vs L2 的 headline 对比需要另跑一个 L1 臂（官方 VLA + decoder），不能和 L2 混。
#
# 这条脚本只做一件事：用两个**同一底模、只在 action head 上不同**的 VLA 检查点，
# 在 HumanoidArena 的 6 个任务上跑 Isaac Sim 闭环，输出成功率 / 未摔倒率 / 存活步数 /
# max_reward，以及配对 McNemar 检验。
#
# 硬约束（见 docs/DATA_POLICY_HA_ONLY.md）：
#   * 数据只允许纯 HA：data/npz/ha_tokens（699 eps）；
#   * WBC 只允许 trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_*.pt；
#   * 处理臂必须由 **纯 HA** 的联合微调产出（runs/joint_ha_a1），
#     且 merge base 必须就是对照组底模；mixed/ft318/MuJoCo 一律拒绝。
#
# 用法：
#   bash scripts/run_a1_ha_isaac.sh
#   SEEDS="0 1 2" REPEATS=1 MAX_STEPS=300 bash scripts/run_a1_ha_isaac.sh
#   TASKS="DoubleDesk PPBox" SEEDS="0" bash scripts/run_a1_ha_isaac.sh   # 冒烟
#
# 前置：
#   * 训练已完成：runs/joint_ha_a1/vla_004000.pt
#   * GPU0 空闲（训练/评测不能同时跑）
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

# ---- LOCKED 2026-10-25：闭环必须与 WBC 训练同口径（HA_ALIGN=1）---------------------
# 缺这两个变量 -> action_scale 退回 BeyondMimic（hip_pitch 放大 2.85x、腕缩小 10x）-> 必摔。
# 见 docs/HANDOVER_CLOSEDLOOP_ACTION_SCALE_20261025.md；A/B 才允许 SM_ALLOW_SCALE_MISMATCH=1。
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$COMIMIC/trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd}
ROOT=$COMIMIC

# checkpoint-8000 的本地路径视图（权重/统计量/processor 都逐字节相同，只把 model_name 指到本地，
# 避免离线环境去访问 HF API）。合并处理臂也必须用这同一个视图作 base。
BASE_CKPT=${BASE_CKPT:-$ROOT/runs/ckpt_vla_ha_8000_local}
FT_CKPT=${FT_CKPT:-$ROOT/runs/ckpt_vla_ha_ft4000}
JOINT_VLA=${JOINT_VLA:-$ROOT/runs/joint_ha_a1/vla_004000.pt}
SM_CKPT=${SM_CKPT:-$ROOT/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt}
TASKS=${TASKS:-"DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi"}
SEEDS=${SEEDS:-"0 1 2"}
REPEATS=${REPEATS:-1}
MAX_STEPS=${MAX_STEPS:-300}
OUT=${OUT:-$ROOT/runs/ablations/A1_joint_finetune_ha}
VLA_GPU=${VLA_GPU:-0}
ISAAC_GPU=${ISAAC_GPU:-0}
FM_STEPS=${FM_STEPS:-4}
PORTS=(${PORTS:-18080 18081})

say() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { echo "[fail] $*" >&2; exit 1; }

[[ -d "$BASE_CKPT" ]] || fail "对照组底模不存在: $BASE_CKPT"
[[ -f "$SM_CKPT" ]] || fail "WBC ckpt 不存在: $SM_CKPT"
case "$SM_CKPT" in
  *bones_seed_token_ha_v2*) ;;
  *) fail "WBC 必须是纯 HA 的 bones_seed_token_ha_v2/ckpt_*.pt，当前: $SM_CKPT" ;;
esac
case "$JOINT_VLA" in
  *mixed*|*ft318*|*mujoco*) fail "处理臂来自禁用数据集: $JOINT_VLA" ;;
esac

# ---- 处理臂：纯 HA 联合微调产物合并回**同一**底模 -------------------------
if [[ ! -d "$FT_CKPT" ]]; then
  [[ -f "$JOINT_VLA" ]] || fail "处理臂 ckpt 未生成: $JOINT_VLA（先跑 scripts/launch_joint_ha.sh / 4000 步纯 HA 训练）"
  mkdir -p "$(dirname "$FT_CKPT")"
  say "merge $JOINT_VLA -> $FT_CKPT (base=$BASE_CKPT)"
  python3 "$ROOT/scripts/export_vla_ckpt.py" --base "$BASE_CKPT" --vla "$JOINT_VLA" \
    --out "$FT_CKPT" --force || fail "导出处理臂 checkpoint 失败"
fi
[[ -f "$FT_CKPT/vla_merge_report.json" ]] || fail "缺 $FT_CKPT/vla_merge_report.json（不是合法合并产物）"
python3 - "$BASE_CKPT" "$FT_CKPT" "$JOINT_VLA" <<'PY' || exit 1
import json, sys
from pathlib import Path
base, ft, vla = (Path(p).resolve() for p in sys.argv[1:4])
d = json.loads((ft / "vla_merge_report.json").read_text())
rb, ru = Path(d["base"]).resolve(), Path(d["update"]).resolve()
if rb != base:
    raise SystemExit(f"[fail] 处理臂 merge base={rb} != 对照组 {base}；A1 必须同底模")
if ru != vla:
    raise SystemExit(f"[fail] 处理臂 update={ru} != {vla}")
text = str(ru).lower()
if "mixed" in text or "ft318" in text:
    raise SystemExit(f"[fail] 处理臂来自禁用数据: {ru}")
if d.get("num_tensors") != 537:
    raise SystemExit(f"[fail] merge 覆盖 {d.get('num_tensors')} 个张量，期望 537")
print(f"[ok] 处理臂审计: base={rb} update={ru} tensors={d['num_tensors']}")
PY

mkdir -p "$OUT"
python3 - "$OUT" "$BASE_CKPT" "$FT_CKPT" "$SM_CKPT" "$TASKS" "$SEEDS" "$REPEATS" "$MAX_STEPS" <<'PY'
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

out, base, ft, sm, tasks, seeds, repeats, steps = sys.argv[1:9]
out = Path(out)
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
def small(p):
    p = Path(p)
    info = {"path": str(p.resolve())}
    if p.is_file():
        info["bytes"] = p.stat().st_size
        if p.stat().st_size < 200_000_000:
            info["sha256"] = sha(p)
    return info
meta = {
    "kind": "A1 joint-finetune ablation, pure HA + IsaacLab/HumanoidArena",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "git_head": subprocess.run(["git", "-C", str(out.parents[2]), "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip() or "?",
    "base_ckpt": str(Path(base).resolve()),
    "ft_ckpt": str(Path(ft).resolve()),
    "wbc_ckpt": small(sm),
    "base_config": small(Path(base) / "config.json"),
    "base_index": small(Path(base) / "model.safetensors.index.json"),
    "ft_merge_report": str((Path(ft) / "vla_merge_report.json").resolve()),
    "protocol": {"tasks": tasks, "seeds": seeds, "repeats": int(repeats),
                 "max_steps": int(steps)},
    "policy": "docs/DATA_POLICY_HA_ONLY.md (mixed/ft318/MuJoCo forbidden)",
}
(out / "manifest.json").write_text(json.dumps(meta, indent=2))
print(f"[manifest] {out / 'manifest.json'}")
PY
say "A1 (pure HA + IsaacLab)  root=$OUT"
say "base=$BASE_CKPT"
say "ft=$FT_CKPT"
say "wbc=$SM_CKPT"
say "tasks=[$TASKS] seeds=[$SEEDS] repeats=$REPEATS max_steps=$MAX_STEPS"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

run_arm() {  # $1 = arm name, $2 = ckpt, $3 = port
  local arm=$1 ckpt=$2 port=$3
  local adir="$OUT/$arm"
  mkdir -p "$adir"
  [[ -d "$ckpt" ]] || fail "arm=$arm ckpt 不存在: $ckpt"
  say "arm=$arm ckpt=$ckpt port=$port"
  # 只放 _noflash：gr00t_deps 里那份 onnxruntime/skimage 是 cp312 的部分拷贝，
  # 会 shadow conda 3.11 的可用包（历史 server 就是用这个最小 PYTHONPATH 跑通的）。
  PYTHONPATH=$ROOT/third_party/gr00t_deps/_noflash \
    python3 "$ROOT/scripts/ha_vla_latent64_server.py" \
      --ckpt "$ckpt" --port "$port" --device "cuda:$VLA_GPU" --bf16 \
      --horizon 40 --fm-steps "$FM_STEPS" \
      > "$adir/server.log" 2>&1 &
  SERVER_PID=$!
  local ok=0
  for _ in $(seq 1 150); do
    if curl -sf --max-time 3 "http://127.0.0.1:$port/" >/dev/null; then ok=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then fail "VLA server 挂了，见 $adir/server.log"; fi
    sleep 2
  done
  [[ "$ok" == "1" ]] || fail "VLA server 300s 内未就绪，见 $adir/server.log"
  say "arm=$arm server ready"

  SM_CKPT="$SM_CKPT" ISAAC_GPU="$ISAAC_GPU" \
  VARIANTS="$arm=http://127.0.0.1:$port" \
  TASKS="$TASKS" SEEDS="$SEEDS" REPEATS="$REPEATS" MAX_STEPS="$MAX_STEPS" \
  OUT="$adir" \
    bash "$ROOT/scripts/ha_isaac_batch.sh" || say "[warn] arm=$arm batch 返回非零（逐集结果仍会汇总）"

  cleanup
  SERVER_PID=""
  say "arm=$arm done -> $adir"
}

run_arm base "$BASE_CKPT" "${PORTS[0]}"
run_arm ft   "$FT_CKPT"   "${PORTS[1]}"

say "汇总 A1 metrics"
python3 "$ROOT/scripts/a1_report_ha.py" --root "$OUT"
say "done: $OUT/SUMMARY.md"
