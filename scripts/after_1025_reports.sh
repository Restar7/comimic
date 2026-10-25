#!/usr/bin/env bash
# 等 chain1/chain2 跑完 → 自动出三份报告 + 一份机器可读汇总，供写"结论"用。
# 用法：setsid nohup bash scripts/after_1025_reports.sh > logs/run_1025_v2/after_reports.log 2>&1 &
set -uo pipefail
REPO=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
LOGD="$REPO/logs/run_1025_v2"
SM="$REPO/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt"
mkdir -p "$LOGD"
cd "$REPO"

echo "[after] waiting for chain1/chain2 $(date -Is)"
while pgrep -f "run_abl_1025_queue.sh chain" > /dev/null 2>&1; do sleep 60; done
echo "[after] chains finished $(date -Is)"

/usr/bin/python3 scripts/a2_fm_report.py \
  --root runs/ablations/A2_fm_backend_v2_physint \
  --log-dir logs/run_1025_v2 \
  --master-log logs/run_1025_v2/chain1.out \
  --out docs/ABLATION_A2_RUN_PHYSINT_20261025.md \
  && echo "[after] a2 report ok" || echo "[after] a2 report FAILED"

/usr/bin/python3 scripts/a5_report.py \
  --root runs/ablations/A5_only_head_v2_physint \
  --log-dir logs/run_1025_v2 \
  --sm-ckpt "$SM" \
  --out docs/ABLATION_A5_RUN_PHYSINT_20261025.md \
  && echo "[after] a5 report ok" || echo "[after] a5 report FAILED"

/usr/bin/python3 scripts/a3_a5_report.py \
  --root runs/ablations --logs logs/run_1025_v2 --no-plots --sm-ckpt "$SM" \
  --out docs/ABLATION_A3_RUN_PHYSINT_20261025.md \
  && echo "[after] a3 report ok" || echo "[after] a3 report FAILED (按需手写)"

/usr/bin/python3 - "$REPO" <<'PY'
import json, sys
from pathlib import Path
repo = Path(sys.argv[1])
arms = {
    "A2": {b: repo / "runs/ablations/A2_fm_backend_v2_physint" / b
           for b in ("pathwise", "checkpointed", "truncated", "adjoint")},
    "A3": {a: repo / "runs/ablations/A3_bptt_vs_ppo" / a for a in ("bptt", "ppo")},
    "A5": {a: repo / "runs/ablations/A5_only_head_v2_physint" / a
           for a in ("head_only", "head_plus_wbc")},
    "A4v6": {c: repo / "runs/ablations/A4_v6_fixframe" / c
             for c in ("wm_frozen_H8", "wm_frozen_H16", "wm_frozen_H32", "wm_frozen_H40")},
}
out = {}
for grp, ds in arms.items():
    out[grp] = {}
    for name, d in ds.items():
        cur = d / "eval_curve.jsonl"
        pts = []
        if cur.exists():
            for line in cur.read_text().splitlines():
                line = line.strip()
                if line:
                    try: pts.append(json.loads(line))
                    except Exception: pass
        keys = ("step", "rollout_reward", "token_mse", "state_term",
                "state_pos", "state_ori", "state_q", "state_vel", "hand_mse")
        out[grp][name] = {
            "dir": str(d.relative_to(repo)),
            "n_evals": len(pts),
            "last_step": pts[-1].get("step") if pts else None,
            "traj": [{k: p.get(k) for k in keys if k in p} for p in pts],
        }
p = repo / "runs/ablations/SUMMARY_1025_v2.json"
p.write_text(json.dumps(out, indent=2))
print(f"[after] summary -> {p}")

rows = []
for grp, ds in out.items():
    for name, v in ds.items():
        traj = v["traj"]
        if not traj:
            rows.append(f"| {grp} | {name} | — | — | — |")
            continue
        first, last = traj[0], traj[-1]
        rows.append("| {g} | {n} | {s} | {r0:+.4f} → {r1:+.4f} | {t} |".format(
            g=grp, n=name, s=last.get("step"),
            r0=first.get("rollout_reward") or 0.0, r1=last.get("rollout_reward") or 0.0,
            t=last.get("token_mse")))
md = "| 组 | 臂 | last step | rollout_reward 首→末 | token_mse(last) |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
(repo / "runs/ablations/SUMMARY_1025_v2.md").write_text(md)
print(md)
PY

echo "[after] done $(date -Is)"
