#!/usr/bin/env python3
"""给 scripts/s3_pipeline.sh 插入第 6 阶段 critic_gate（把 8 阶段变成 9 阶段）。

只做**机械替换**，每处都断言唯一命中；重复跑会报错（幂等检查靠 critic_gate 已存在）。
"""
from pathlib import Path

p = Path("scripts/s3_pipeline.sh")
s = p.read_text()

if "stage_critic()" in s:
    raise SystemExit("[patch] 已经有 stage_critic，先还原再打")

# ---- 1) 参数 ----
a = 'WM_LR=${WM_LR:-1e-5}\n'
assert s.count(a) == 1, s.count(a)
s = s.replace(a, a + 'CRITIC_VEV_MIN=${CRITIC_VEV_MIN:-0.30}   # 留出 v_ev 低于这个就不许开 critic（mjlab run_j1_rounds.sh 第 7 步的口径）\n')

# ---- 2) 阶段编号 8 → 9（critic 插在后面，所以 6~8 顺延）----
for old, new in (("8/8", "9/9"), ("7/8", "8/9"), ("6/8", "7/9"),
                 ("5/8", "5/9"), ("4/8", "4/9"), ("3/8", "3/9"),
                 ("2/8", "2/9"), ("1/8", "1/9")):
    s = s.replace(f"--- {old} ", f"--- {new} ")

# ---- 3) critic_gate 阶段本体，插在 stage_gate 之后 ----
anchor = "# ============================================================ 6. joint"
assert s.count(anchor) == 1
block = '''# ============================================================ 6. critic 门
stage_critic() {
  local wm; wm=$(st_need gate)
  local tr; tr=$(st_need transitions)
  say "--- 6/9 critic 拟合 + 门（留出 v_ev > $CRITIC_VEV_MIN 才允许开裁判）---"
  PYTHONPATH=. python3 "$COMIMIC/scripts/fit_critic_real.py" \\
    --transitions "$tr" --out "$COMIMIC/runs/wm_real/critic_b_$ROUND.pt" \\
    --center linear --epochs 12 --device cuda:0 > "$PD/6_critic.log" 2>&1 \\
    || die "critic 拟合失败（看 $PD/6_critic.log）"
  local line vev
  line=$(grep -E "^\\[critic\\] best epoch=.* val v_ev=" "$PD/6_critic.log" | tail -1)
  [[ -n "$line" ]] || die "critic 日志里找不到 best epoch 的 v_ev（看 $PD/6_critic.log）"
  vev=$(printf '%s' "$line" | grep -oE "val v_ev=[-+]?[0-9.]+" | cut -d= -f2)
  say "critic 留出 v_ev=$vev（门槛 $CRITIC_VEV_MIN）"
  python3 - "$PD" "$vev" "$CRITIC_VEV_MIN" "$tr" <<'PY'
import json, sys
from pathlib import Path
pd, vev, thr, tr = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
allow = vev > thr
json.dump({"v_ev": vev, "threshold": thr, "allow": allow, "transitions": tr,
           "rule": "v_ev <= threshold ⇒ 强制关 critic（mjlab run_j1_rounds.sh 第 7 步）"},
          open(pd / "critic_gate.json", "w"), indent=1, ensure_ascii=False)
print(f"[critic-gate] {'PASS ⇒ 本轮开 critic' if allow else 'FAIL ⇒ 本轮强制关 critic（裁判不可信）'}")
PY
  [[ $? == 0 ]] || die "critic 门写入失败"
  tail -2 "$PD/6_critic.log" | tee -a "$PD/summary.txt"
  say "critic_gate ✅ $PD/critic_gate.json"
  st_mark critic "$PD/critic_gate.json"
}

'''
s = s.replace(anchor, block + anchor)

# ---- 4) stage_joint 先过 critic 门，并由门决定是否开 critic ----
a = '''stage_joint() {
  local wm; wm=$(st_need gate)'''
assert s.count(a) == 1
b = '''stage_joint() {
  local wm; wm=$(st_need gate)
  local cg; cg=$(st_need critic)      # 门决定开不开裁判，不许绕过
  local cg_allow
  cg_allow=$(python3 -c "import json,sys;print(1 if json.load(open(sys.argv[1]))['allow'] else 0)" "$cg")
  local c1=$CRITIC_ON_MAIN c0=$CRITIC_ON_CTRL g1=$CRITIC_GRAD_STATE_MAIN g0=$CRITIC_GRAD_STATE_CTRL
  if [[ "$cg_allow" != "1" ]]; then
    c1=0; c0=0; g1=0; g0=0
    say "⚠ critic 门未通过 ⇒ 两臂强制 critic=OFF（只走奖励项）"
  fi'''
s = s.replace(a, b)

a = '''    CRITIC_ON_MAIN="${CRITIC_ON_MAIN:-}" CRITIC_ON_CTRL="${CRITIC_ON_CTRL:-}" \\
    CRITIC_GRAD_STATE_MAIN="${CRITIC_GRAD_STATE_MAIN:-}" \\
    CRITIC_GRAD_STATE_CTRL="${CRITIC_GRAD_STATE_CTRL:-}" GRAD_LOG_EVERY=100 \\'''
assert s.count(a) == 1
b = '''    CRITIC_ON_MAIN="$c1" CRITIC_ON_CTRL="$c0" \\
    CRITIC_GRAD_STATE_MAIN="$g1" CRITIC_GRAD_STATE_CTRL="$g0" GRAD_LOG_EVERY=100 \\'''
s = s.replace(a, b)

# ---- 5) all / 用法 ----
a = '''    stage_collect; stage_score; stage_transitions; stage_wm
    stage_gate;    stage_joint; stage_eval;      stage_report'''
assert s.count(a) == 1
b = '''    stage_collect; stage_score; stage_transitions; stage_wm
    stage_gate;    stage_critic; stage_joint;      stage_eval; stage_report'''
s = s.replace(a, b)

a = "  wm_gate)     stage_gate ;;\n"
assert s.count(a) == 1
s = s.replace(a, a + "  critic_gate) stage_critic ;;\n")

a = "collect|score|transitions|wm_refresh|wm_gate|joint|eval|report|all"
assert s.count(a) == 1
s = s.replace(a, "collect|score|transitions|wm_refresh|wm_gate|critic_gate|joint|eval|report|all")

p.write_text(s)
print("[patch] ok：s3_pipeline.sh 现在是 9 阶段")
