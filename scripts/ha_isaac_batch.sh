#!/usr/bin/env bash
# 批量闭环：多任务 × 多 seed × 多 repeat，跑完直接出成功率表。
# 消融实验就用这个：每个 variant 起一个自己的 VLA 服务（不同 ckpt / 不同消融模块），
# 用 VARIANTS="name=url,name=url" 传进来，其余不变。
#
# 用法：
#   bash scripts/ha_isaac_batch.sh                      # 6 个任务 × seed0 × 1 次
#   TASKS="DoubleDesk PPBox" SEEDS="0 1 2" REPEATS=2 bash scripts/ha_isaac_batch.sh
#   VARIANTS="base=http://127.0.0.1:18080,abl=http://127.0.0.1:18081" \
#     TASKS="DoubleDesk" SEEDS="0 1 2" REPEATS=5 bash scripts/ha_isaac_batch.sh
#
# 前置条件（脚本不负责起服务，避免和训练抢卡）：
#   每个 variant 的 VLA 服务已经在对应端口上 {"ok": true}
#   Isaac 用 GPU0，VLA 服务建议放 GPU1（ISAAC_GPU 可改）
#
# 产出：$OUT_ROOT/../ha_isaac_batch_<stamp>/{runs/*, summary.json, SUMMARY.md}
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

# ---- LOCKED 2026-10-25：闭环必须与 WBC 训练同口径（HA_ALIGN=1）---------------------
# 缺这两个变量 -> action_scale 退回 BeyondMimic（hip_pitch 放大 2.85x、腕缩小 10x）-> 必摔。
# 见 docs/HANDOVER_CLOSEDLOOP_ACTION_SCALE_20261025.md；A/B 才允许 SM_ALLOW_SCALE_MISMATCH=1。
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$COMIMIC/trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd}
TASKS=${TASKS:-"DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi"}
SEEDS=${SEEDS:-"0"}
REPEATS=${REPEATS:-1}
MAX_STEPS=${MAX_STEPS:-300}
VARIANTS=${VARIANTS:-"base=http://127.0.0.1:18080"}
ISAAC_GPU=${ISAAC_GPU:-0}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-$COMIMIC/runs/ha_isaac_batch_$STAMP}
mkdir -p "$OUT/runs"

echo "[batch] variants=$VARIANTS"
echo "[batch] tasks=$TASKS seeds=[$SEEDS] repeats=$REPEATS max_steps=$MAX_STEPS"
echo "[batch] out=$OUT"

# ---- 1) 逐集跑 -------------------------------------------------------------
IFS=',' read -ra VSPECS <<< "$VARIANTS"
for vspec in "${VSPECS[@]}"; do
  VNAME=${vspec%%=*}
  VURL=${vspec#*=}
  # 服务活着才跑，避免把"服务没起"记成"任务失败"
  if ! curl -sf --max-time 5 "$VURL/" >/dev/null; then
    echo "[batch] SKIP variant=$VNAME ($VURL not reachable)" || true
    continue
  fi
  for task in $TASKS; do
    for seed in $SEEDS; do
      for ((r = 0; r < REPEATS; r++)); do
        echo "[batch] $(date +%T) run variant=$VNAME task=$task seed=$seed repeat=$r"
        TASK=$task SEED=$seed REPEAT=$r MAX_STEPS=$MAX_STEPS \
          SERVER_URL=$VURL ISAAC_GPU=$ISAAC_GPU MODEL_LABEL=$VNAME \
          OUT_ROOT=$OUT/runs \
          bash "$COMIMIC/scripts/ha_isaac_closedloop.sh" \
          >> "$OUT/runs/${VNAME}__${task}__seed${seed}__repeat${r}.log" 2>&1
        rc=$?
        echo "[batch] $(date +%T) done rc=$rc variant=$VNAME task=$task seed=$seed repeat=$r"
      done
    done
  done
done

# ---- 2) 汇总 ---------------------------------------------------------------
python3 - "$OUT" <<'PY'
import json, sys, pathlib, collections
out = pathlib.Path(sys.argv[1])
rows = []
for f in sorted(out.glob("runs/**/result.json")):
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    rows.append(d)

agg = collections.defaultdict(lambda: {"n": 0, "ok": 0, "steps": [], "rewards": []})
for d in rows:
    task = str(d.get("task", "?"))
    label = str(d.get("model_label") or "base")
    key = (label, task)
    a = agg[key]
    a["n"] += 1
    a["ok"] += int(bool(d.get("success")))
    a["steps"].append(int(d.get("episode_steps") or 0))
    a["rewards"].append(float(d.get("max_reward") or 0.0))

summary = []
for (label, task), a in sorted(agg.items()):
    rate = a["ok"] / a["n"] if a["n"] else 0.0
    avg_steps = sum(a["steps"]) / len(a["steps"]) if a["steps"] else 0.0
    summary.append({"variant": label, "task": task, "episodes": a["n"],
                    "success": a["ok"], "success_rate": round(rate, 4),
                    "avg_steps": round(avg_steps, 1),
                    "avg_max_reward": round(sum(a["rewards"]) / len(a["rewards"]), 4)})

# 每个 variant 的总体成功率（跨任务）
per_variant = collections.defaultdict(lambda: [0, 0])
for s in summary:
    per_variant[s["variant"]][0] += s["success"]
    per_variant[s["variant"]][1] += s["episodes"]

(out / "summary.json").write_text(json.dumps(
    {"rows": rows, "per_task": summary,
     "per_variant": {k: {"success": v[0], "episodes": v[1],
                         "success_rate": round(v[0] / v[1], 4) if v[1] else 0.0}
                     for k, v in per_variant.items()}}, indent=2))

lines = ["# 闭环批量结果", "",
         f"episodes: {len(rows)}", "",
         "| variant | task | episodes | success | rate | avg_steps |",
         "|---|---|---|---|---|---|"]
for s in summary:
    lines.append(f"| {s['variant']} | {s['task']} | {s['episodes']} | {s['success']} "
                 f"| {s['success_rate']:.1%} | {s['avg_steps']} |")
lines += ["", "## 每个 variant 总体", "", "| variant | success | episodes | rate |",
          "|---|---|---|---|"]
for k, v in sorted(per_variant.items()):
    lines.append(f"| {k} | {v[0]} | {v[1]} | {v[0] / v[1]:.1%} |" if v[1] else f"| {k} | 0 | 0 | - |")
(out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n[batch] summary -> {out}/SUMMARY.md")
PY
