#!/usr/bin/env python3
"""给 scripts/s3_pipeline.sh 插入第 1 阶段 `a_collect`（A 线：WBC-only、Isaac 真物理、~10^6 条）。

把流程从 9 阶段变 10 阶段：
    a_collect → collect → score → transitions → wm_refresh → wm_gate
              → critic_gate → joint → eval → report

只做机械替换，每处都断言唯一命中；已打过会报错（幂等检查靠 stage_a_collect 已存在）。
"""
from pathlib import Path

p = Path("scripts/s3_pipeline.sh")
s = p.read_text()
if "stage_a_collect()" in s:
    raise SystemExit("[patch] 已经有 stage_a_collect，先还原再打")

# ---------------------------------------------------------------- 1) 参数
a = "CRITIC_VEV_MIN=${CRITIC_VEV_MIN:-0.30}"
assert s.count(a) == 1
i = s.index(a)
s = s[:i] + (
    '# A 线（mjlab 第 1 步的等价物）：只跑冻结 WBC、在 Isaac 真物理里向量化走，产真转移。\n'
    '# 为什么需要：我们原来 95% 的 WM 训练数据是**运动学参考**（packed.ref + 录制动作），\n'
    '# 实测 qd_corr 只有 0.743、qd_ratio 0.948（系统性低估速度 5%）；A 线补的是真动力学。\n'
    'A_ENVS=${A_ENVS:-512}                 # 每波并行 env 数\n'
    'A_STEPS=${A_STEPS:-300}               # 每 env 推进的控制步数\n'
    'A_WAVES=${A_WAVES:-4}                 # 波数（总量 = A_ENVS*A_STEPS*A_WAVES）\n'
    'A_MIN_ROWS=${A_MIN_ROWS:-200000}      # 少于这个数就不许往下走\n'
    'WM_ALINE_FRAC=${WM_ALINE_FRAC:-0.6}   # WM 刷新里轨迹窗口的占比（A/B 线合计）\n'
) + s[i:]

# ---------------------------------------------------------------- 2) 阶段编号 9 -> 10
for old, new in (("6/9", "7/10"), ("7/9", "8/10"), ("8/9", "9/10"), ("9/9", "10/10"),
                 ("5/9", "6/10"), ("4/9", "5/10"), ("3/9", "4/10"),
                 ("2/9", "3/10"), ("1/9", "2/10")):
    s = s.replace(f"--- {old} ", f"--- {new} ")

# ---------------------------------------------------------------- 3) a_collect 阶段本体
anchor = "# ============================================================ 1. collect"
assert s.count(anchor) == 1
block = '''# ============================================================ 0. A 线采集
stage_a_collect() {
  local outdir=$COMIMIC/runs/wm_real
  local n_total rows
  rows=$(python3 - "$outdir" <<'PYA0'
import glob, sys, torch
n = 0
for f in sorted(glob.glob(f"{sys.argv[1]}/a_w*.pt")):
    try:
        n += int(torch.load(f, map_location="cpu", weights_only=False)["states"].shape[0])
    except Exception:
        pass
print(n)
PYA0
)
  n_total=$rows
  if [[ "${n_total:-0}" -ge "$A_MIN_ROWS" ]]; then
    say "--- 1/10 A 线采集：已有 $(ls "$outdir"/a_w*.pt | wc -l) 波 / $n_total 条，跳过 ---"
    st_mark a_collect "$outdir"
    return 0
  fi
  say "--- 1/10 A 线采集（冻结 WBC · Isaac 真物理 · $A_ENVS env × $A_STEPS 步 × $A_WAVES 波）---"
  ENVS="$A_ENVS" STEPS="$A_STEPS" WAVES="$A_WAVES" OUTDIR="$outdir" \\
    bash "$COMIMIC/scripts/run_a_collect_batch.sh" 2>&1 | tee "$PD/0_a_collect.log" | tail -4
  [[ "${PIPESTATUS[0]}" == "0" ]] || die "A 线采集失败（看 $PD/0_a_collect.log）"
  n_total=$(ls "$outdir"/a_w*.pt 2>/dev/null | wc -l)
  local rows
  rows=$(python3 - "$outdir" <<'PYA'
import glob, sys, torch
n = 0
for f in sorted(glob.glob(f"{sys.argv[1]}/a_w*.pt")):
    n += int(torch.load(f, map_location="cpu", weights_only=False)["states"].shape[0])
print(n)
PYA
)
  say "A 线共 $n_total 个文件 / $rows 条真实转移"
  (( rows >= A_MIN_ROWS )) || die "A 线只有 $rows 条（< $A_MIN_ROWS）⇒ 不够训 WM，停止"
  # 奖励/ret 用系统 python 补（Isaac venv 里 import 训练栈会炸）
  say "补算 A 线奖励..."
  SUPERMIMIC_HA_ALIGN=1 PYTHONPATH=. python3 "$COMIMIC/scripts/compute_collect_reward.py" \\
    --in "$outdir"/a_w*.pt --out "$outdir/a_rew.pt" > "$PD/0_a_reward.log" 2>&1 \\
    || die "A 线奖励补算失败（看 $PD/0_a_reward.log）"
  tail -2 "$PD/0_a_reward.log"
  st_mark a_collect "$outdir"
  say "a_collect ✅ $rows 条（奖励 $outdir/a_rew.pt）"
}

'''
s = s.replace(anchor, block + anchor)

# ---------------------------------------------------------------- 4) WM 刷新吃 A 线
a = '''    --traj-dump "$tr/*.pt" --traj-frac 0.2 --traj-val-frac 0.25 \\'''
assert s.count(a) == 1, s.count(a)
b = '''    --traj-dump "$tr/*.pt" --traj-frac "$WM_ALINE_FRAC" --traj-val-frac 0.25 \\
    --a-line "$COMIMIC/runs/wm_real/a_w*.pt" --a-line-envs 256 \\'''
s = s.replace(a, b)

# ---------------------------------------------------------------- 5) all / 用法 / 版本
a = '''    stage_collect; stage_score; stage_transitions; stage_wm'''
assert s.count(a) == 1
s = s.replace(a, '''    stage_a_collect
    stage_collect; stage_score; stage_transitions; stage_wm''')

a = "  collect)     stage_collect ;;\n"
assert s.count(a) == 1
s = s.replace(a, "  a_collect)   stage_a_collect ;;\n" + a)

a = "collect|score|transitions|wm_refresh|wm_gate|critic_gate|joint|eval|report|all"
assert s.count(a) == 1
s = s.replace(a, "a_collect|collect|score|transitions|wm_refresh|wm_gate|critic_gate|joint|eval|report|all")

a = 'd["pipeline_version"] = "v9"'
assert s.count(a) == 1
s = s.replace(a, 'd["pipeline_version"] = "v10"')

p.write_text(s)
print("[patch] ok：s3_pipeline.sh 现在是 10 阶段（a_collect 打头）")
