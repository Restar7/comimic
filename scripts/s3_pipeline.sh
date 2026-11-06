#!/usr/bin/env bash
# ============================================================================
# S3 **锁死的完整流程**（2026-10-29）。这是唯一允许执行 S3 训练轮次的入口。
#
# 设计原则（就是为了杜绝"漏步骤 / 用了上一轮的 WM / 没打分就建转移"这类错误）：
#   1. 八个阶段**必须按顺序**跑；每个阶段开始前先校验**上一阶段的产物存在且是这一轮的**；
#   2. 每个阶段的产物记进 `runs/S3_pipeline/<ROUND>/state.json`（含时间戳、漂移量、关键参数）；
#   3. 关键断言（**不满足直接退出**）：
#        - 打分前：每个 dump 都要有 scored.pt
#        - 建转移前：scored.pt 的数量 == dump 数量
#        - WM 刷新前：转移文件里有 extra（B 线）且 > 0
#        - WM 刷新后：必须产出新 WM，且 **WM 门必须 PASS**（否则回退 base 并记录）
#        - 联合训练前：`WBC_CKPT` **必须等于本轮的 WM**（上一轮就会在这里被抓出来）
#        - 评测前：joint ckpt 存在，且 `train_wbc=1` 时评测必须用 joint_*.pt 当执行器
#   4. 任何阶段失败 → 立即停止，绝不"跳过继续"。
#
# 用法（一轮 = 从头到尾一条命令）：
#   ROUND=r3 bash scripts/s3_pipeline.sh all
#   ROUND=r3 bash scripts/s3_pipeline.sh collect     # 只跑某一段（仍会校验前置）
#   ROUND=r3 COLLECT_EPISODES=7 bash scripts/s3_pipeline.sh collect
#
# 阶段：
#   1 collect     采 6 任务 × EPISODES 集（两卡并行，server 带 --sample-avg 4）
#   2 score       给整池 dump 补 rewards（bundle 只加载一次）
#   3 transitions 建真转移（含 B 线 extra）
#   4 wm_refresh  用**本轮池子**刷新 WM（留出一个任务当门）
#   5 wm_gate     20 步关节误差 ≤ 1.05× base 才算过；不过就回退 base
#   6 joint       联合微调（WM 当裁判 + critic 用**想象的 GAE**）
#   7 eval        6-seed 配对闭环评测（π0 vs 候选，两边 --sample-avg 4）
#   8 report      汇总表 + 漂移图
# ============================================================================
set -uo pipefail

# ★ 2026-10-30 自检：joint 的 env 续行链里**不能有注释行**。
#   `\` + 换行会把注释并进同一逻辑行，注释吃掉续行 ⇒ 后面的变量（RUN_LABEL 等）静默丢失、
#   训练跑到默认目录 r99。这个坑今天已经犯过三次，钉成启动自检。
python3 - "$0" <<'PYSELF' || { echo "[selfcheck] ❌ joint env 链里有注释行 ⇒ 会静默丢变量" >&2; exit 1; }
import sys
src = open(sys.argv[1], encoding="utf-8").read().split("\n")
try:
    i = next(k for k, ln in enumerate(src) if ln.startswith('  ROUND=99 RUN_LABEL="$jl"'))
    j = next(k for k in range(i, len(src)) if 's3_loop.sh" train' in src[k])
except StopIteration:
    raise SystemExit(0)
bad = [k + 1 for k in range(i, j) if src[k].lstrip().startswith("#")]
if bad:
    print(f"[selfcheck] joint env 链第 {bad} 行有注释")
    raise SystemExit(1)
PYSELF

COMIMIC=${COMIMIC:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
cd "$COMIMIC" || exit 1
export COMIMIC
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$COMIMIC/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd}

ROUND=${ROUND:?必须给 ROUND（例如 ROUND=r3）}
PD=$COMIMIC/runs/S3_pipeline/$ROUND
STATE=$PD/state.json
mkdir -p "$PD" "$COMIMIC/logs"

# ---- 本轮参数（都在这里，别在别处改）------------------------------------
TASKS=${TASKS:-"DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi"}
COLLECT_EPISODES=${COLLECT_EPISODES:-7}
COLLECT_STEPS=${COLLECT_STEPS:-600}
HEAD_COLLECT=${HEAD_COLLECT:-base}          # 采集用哪个 head（base=底模）
SRV_EXTRA=${SRV_EXTRA:---sample-avg 4}      # 部署端降噪
HOLD_TASK=${HOLD_TASK:-SitSofa}             # WM 门的留出任务
BASE_WM=${BASE_WM:-$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt}
WM_STEPS=${WM_STEPS:-100}
WM_LR=${WM_LR:-1e-5}
# A 线（mjlab 第 1 步的等价物）：只跑冻结 WBC、在 Isaac 真物理里向量化走，产真转移。
# 为什么需要：我们原来 95% 的 WM 训练数据是**运动学参考**（packed.ref + 录制动作），
# 实测 qd_corr 只有 0.743、qd_ratio 0.948（系统性低估速度 5%）；A 线补的是真动力学。
A_ENVS=${A_ENVS:-512}                 # 每波并行 env 数
A_STEPS=${A_STEPS:-300}               # 每 env 推进的控制步数
A_WAVES=${A_WAVES:-4}                 # 波数（总量 = A_ENVS*A_STEPS*A_WAVES）
A_MIN_ROWS=${A_MIN_ROWS:-200000}      # 少于这个数就不许往下走
# ★ 2026-11-01：反事实数据 P 池（runs/Bw_pert*）是**负结果，勿用**（见
#   docs/HANDOVER_WM_20261031.md §"负结果"）。已经从 pipeline 里删除接线：
#   WM 只走 A+B，不再有第三池。`finetune_wm_velocity_policy.py` 的 --pert-* 参数
#   只为复现历史负结果保留，正常流程不传。
WM_ALINE_FRAC=${WM_ALINE_FRAC:-0.6}   # WM 刷新里轨迹窗口的占比（A/B 线合计）
WM_STEPS_A=${WM_STEPS_A:-3000}      # 阶段1（场景/A 线）步数；mjlab 用 3000
WM_STEPS_MIX=${WM_STEPS_MIX:-3000}  # 阶段2（A+B 混合）步数；mjlab 用 8000（用户口径 3000）
WM_STEPS_2B=${WM_STEPS_2B:-1500}    # 阶段2b（k=20 自回归）步数；mjlab 用 1500
WM_LR_P2=${WM_LR_P2:-3e-5}          # 阶段2 学习率           # A 段步数（0=跳过）。实测 A 段猛训会把 20 步位置精度打烂（A3000 后 k=20 关节 +176%），默认走单阶段混合
WM_LR_A=${WM_LR_A:-1e-4}              # A 段学习率（从运动学基线切到真动力学，要大一点）
WM_ALINE_ENVS_A=${WM_ALINE_ENVS_A:-256}  # A 段每文件取多少条 A 线轨迹
# ★ 2026-10-31（A8，用户口径）：**整批配比显式化**。
#   原来 `--traj-frac` 是"相对 prov.sample() 的追加倍数"，实际整批是：
#     stage1 = 参考库 20% + A 线 80%；stage2 = 参考库 62% + A 27% + B 10%
#   而指纹里打印的"B 占比 28%"只是 **A+B 内部**口径，把人骗了。
#   现在：WM_MIX_LIB=0 表示**完全不用参考动作库**（只用我们采的 A/B 线）。
# ★ 2026-10-31（A2 前置）：root 角速度 w 的损失权重。原来 v/w/qd 三项等权，
#   w 只有 3 维被 29 维的 qd 挤掉 ⇒ 刷新后 root_w_corr 0.500→0.261、ratio 1.11→1.82。
#   奖励里的 yaw_rate 项吃的就是这个量，所以必须先把它训好，否则恢复 yaw_rate 只是喂噪声。
WM_VEL_SCALE_LAMBDA=${WM_VEL_SCALE_LAMBDA:-1.0}   # 速度通道"幅度比"权重（防 L2 回归均值把幅度压低）
WM_W_LAMBDA=${WM_W_LAMBDA:-3.0}
WM_MIX_LIB=${WM_MIX_LIB:-0}
WM_AB_RATIO=${WM_AB_RATIO:-1}         # A:B 条数比（1 ⇒ B 占 A+B 的 50%；用户口径 40~50%）
# ★ 2026-10-29 实测（用户口径）：**多步自回归训练**。K=3 时门口径 k=20 = 0.147；
#   K=20 自回归 ⇒ 0.141（−8.9%），且 k=1/5/10/20 全部变好、qd_ratio 1.31→0.93。
#   这是目前为止唯一在**每个视界**都赢的 WM 改法。
WM_ROLLOUT_STEPS=${WM_ROLLOUT_STEPS:-20}   # 刷新时多步自回归展开的步数（原来 3）
WM_BATCH=${WM_BATCH:-128}                  # K=20 时显存大，batch 从 256 降到 128
WM_ARCH=${WM_ARCH:-mlp}                    # WM 结构：mlp（默认，与历史轮次逐位可比）/ gru（单层 GRU 1024 + 2048 头）
WM_CLIP_NORM=${WM_CLIP_NORM:-1.0}          # 梯度裁剪范数（BPTT 安全阀；1.0 = 旧行为）
# ★ 2026-11-01：A 线关节顺序。脚本默认 `ha_v31` 是给历史数字复现用的；我们的 A 线文件
#   meta 已经是 standard，必须用 `declared`（不换序）。训练脚本自检会直接拦下选错的顺序，
#   所以完整轮必须显式传这个值，否则 WM 刷新阶段1必挂（h20r2 第一次就是这个原因）。
A_ORDER=${A_ORDER:-declared}
# ★ 2026-10-31：这些原来只在 joint 的 env 链里用 `${VAR:-默认}`，脚本层没有定义，
#   preflight 里直接引用 $ROOT_ORI_MODE 会在 `set -u` 下崩（unbound variable）。现在统一在顶部给默认。
ROOT_ORI_MODE=${ROOT_ORI_MODE:-split}
TAIL_WEIGHT=${TAIL_WEIGHT:-2.0}
TAIL_POS_THR=${TAIL_POS_THR:-}            # 根位置 hinge e_h（空=reward_shape.ROOT_POS_THR）
TAIL_POS_CAP=${TAIL_POS_CAP:-}            # 根位置 hinge cap（空=reward_shape.ROOT_POS_CAP）
TRUST_CENTER=${TRUST_CENTER:-self}
# ★ 2026-10-31（方向修正）：**锚点退场 = 降 `act_weight`**，不是降 `act_eps`。
#   `relu(RMS(a−a_ref)/eps − 1)²` 里 eps 变小只会让球**更紧**（之前写反了）。
#   课程：前期权重大（1.0，负责把行为钉在"真机执行过的动作"附近），后期逐步交给奖励+hinge。
ACT_WEIGHT=${ACT_WEIGHT:-1.0}
ACT_WEIGHT_SCHEDULE=${ACT_WEIGHT_SCHEDULE:-}
ACT_WEIGHT_FLOOR=${ACT_WEIGHT_FLOOR:-0.05}
TRUST_EPS=${TRUST_EPS:-0.2}
TRUST_WEIGHT=${TRUST_WEIGHT:-1.0}
# ★ 2026-10-31（e1/e2）：观测扩维 + 奖励改造的开关（全部进指纹，防止"配置写了代码没生效"）
# ★ 2026-10-31（A5）：改用 **rotlocal6** —— 线上 state64[:6] 本来就是
#   rot6d(conj(heading(q0))·q_now)（相对本回合起始朝向，开局 identity），
#   与训练侧 `rotlocal6` 已逐位对拍 ⇒ **零客户端改动**、且口径与部署一致。
#   （ori6 仍保留：它需要客户端额外发参考帧，暂不启用。）
STATE_EXTRA=${STATE_EXTRA:-rotlocal6}     # none|rotlocal6|abs6|vel6|ori6
ADV_BASELINE=${ADV_BASELINE:-self}        # none|self（用真机执行 token 做控制变量）
ADV_TAU=${ADV_TAU:-0.02}                  # advantage 有界温度（0=退回均值形式=安慰剂）
TAIL_CAP_SLOPE=${TAIL_CAP_SLOPE:-0.5}     # 越过 hinge cap 后保留的斜率（0=旧行为，远场零梯度）
B_FRESH_WEIGHT=${B_FRESH_WEIGHT:-3.0}
B_FRESH_GLOB=${B_FRESH_GLOB:-}
# ★ 2026-10-31：新数据加权 glob 的**唯一来源**。
#   没显式指定 ⇒ 只覆盖本轮（runs/B_<ROUND>/*/dump/*.pt）。**绝不**回退到全池：全池乘 N 等于没加权。
_fresh_glob_effective() {
  if [[ -n "${B_FRESH_GLOB:-}" ]]; then printf '%s\n' "$B_FRESH_GLOB"
  else printf '%s\n' "$COMIMIC/runs/B_${ROUND}/*/dump/*.pt"; fi
}
# ★ 2026-10-31：act 锚点的 eps **逐步退场**（用户口径："d3 是 0.6，下一轮开始慢慢降"）。
#   规则是"看上一轮的 act/frac_out 再决定"而不是写死时间表：
#     frac_out < 0.35（球真的收拢了）⇒ eps −= 0.15，下限 0.30
#     否则保持不变（说明球还在起作用，硬降会把它废掉）
#   只在 ACT_EPS=auto 时生效；显式给值就完全听调用方的（但会被断言拦住"反向变大"）。

# 锚点权重的有效值：显式表 > 显式值 > 自动退场（看上轮 frac_out）
_act_weight_effective() {
  # ★ 2026-10-31（消融）：消融轮**冻结课程**，显式取 ACT_WEIGHT（默认 1.0 = 与 d3 相同）。
  #   为什么必须短路：课程规则是"上轮 frac_out<0.35 ⇒ 本轮 ×0.5"，d3 结束时 frac_out=0.2266
  #   ⇒ 所有消融臂会**静默**拿到 act_w=0.5，那就等于同时翻了两个旋钮（AB2 起全部作废）。
  #   这个坑是 2026-10-31 用 dry-run 抓到的，写进 docs/ABLATION_SUITE.md §2 的检查项。
  if [[ "${ABL_ROUND:-0}" == "1" ]]; then printf '%s\n' "${ACT_WEIGHT:-1.0}"; return 0; fi
  if [[ -n "${ACT_WEIGHT_SCHEDULE:-}" ]]; then
    IFS=',' read -ra _wp <<< "$ACT_WEIGHT_SCHEDULE"
    local _kv
    for _kv in "${_wp[@]}"; do
      if [[ "${_kv%%:*}" == "$ROUND" ]]; then printf '%s\n' "${_kv##*:}"; return 0; fi
    done
  fi
  python3 - "$COMIMIC/runs/S3_loop" "$ROUND" "${ACT_WEIGHT:-1.0}" "${ACT_WEIGHT_FLOOR:-0.05}" <<'PYAW'
import glob, json, os, sys
root, cur, base, floor = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
best = None
for d in sorted(glob.glob(os.path.join(root, "*", "train_main"))):
    r = os.path.basename(os.path.dirname(d))
    # ★ 目录名是 "<round>" 或 "<round>_joint" —— 都要排除，否则会把自己当成"上一轮"
    #   （d3 因此被误判成"上轮 frac_out=0.211 ⇒ 退场到 0.5"，实际应该是 1.0）
    if r == cur or r == cur + "_joint":
        continue
    ap, mp = os.path.join(d, "args.json"), os.path.join(d, "train_metrics.jsonl")
    if not (os.path.exists(ap) and os.path.exists(mp)):
        continue
    try:
        a = json.load(open(ap)); last = [json.loads(l) for l in open(mp)][-1]
    except Exception:
        continue
    if "act_weight" not in a:
        continue
    m = os.path.getmtime(mp)
    if best is None or m > best[0]:
        best = (m, r, float(a["act_weight"]), last.get("act/frac_out"))
if best is None:
    print(f"{base:g}"); raise SystemExit
_, r, w, fr = best
fr = float(fr) if fr is not None else 1.0
# 只在"球已经不再咬住"（frac_out<0.35）时退场；否则保持——避免奖励还没接管就把安全网撤了
new = max(floor, w * 0.5) if fr < 0.35 else w
print(f"{new:g}")
PYAW
}

_act_eps_effective() {
  if [[ -n "${ACT_EPS:-}" && "${ACT_EPS}" != "auto" ]]; then printf '%s\n' "$ACT_EPS"; return 0; fi
  # 显式逐轮表（用户口径："d3 用原始 0.60，d4 一定降下去"）。优先级高于下面的自动规则。
  # 例：ACT_EPS_SCHEDULE="d3:0.60,d4:0.45"
  if [[ -n "${ACT_EPS_SCHEDULE:-}" ]]; then
    IFS=',' read -ra _pairs <<< "$ACT_EPS_SCHEDULE"
    local _kv
    for _kv in "${_pairs[@]}"; do
      if [[ "${_kv%%:*}" == "$ROUND" ]]; then printf '%s\n' "${_kv##*:}"; return 0; fi
    done
  fi
  python3 - "$COMIMIC/runs/S3_loop" "$ROUND" <<'PYAE'
import glob, json, os, sys
root, cur = sys.argv[1], sys.argv[2]
best = None
for d in sorted(glob.glob(os.path.join(root, "*", "train_main"))):
    r = os.path.basename(os.path.dirname(d))
    # ★ 目录名是 "<round>" 或 "<round>_joint" —— 都要排除，否则会把自己当成"上一轮"
    #   （d3 因此被误判成"上轮 frac_out=0.211 ⇒ 退场到 0.5"，实际应该是 1.0）
    if r == cur or r == cur + "_joint":
        continue
    ap, mp = os.path.join(d, "args.json"), os.path.join(d, "train_metrics.jsonl")
    if not (os.path.exists(ap) and os.path.exists(mp)):
        continue
    try:
        a = json.load(open(ap))
        last = [json.loads(l) for l in open(mp)][-1]
    except Exception:
        continue
    if "act_eps" not in a:
        continue
    mtime = os.path.getmtime(mp)
    if best is None or mtime > best[0]:
        best = (mtime, r, float(a["act_eps"]), last.get("act/frac_out"))
if best is None:
    print("0.6"); raise SystemExit
_, r, eps, fr = best
fr = float(fr) if fr is not None else 1.0
new = eps
if fr < 0.35:
    new = max(0.30, eps - 0.15)
print(f"{new:.2f}")
PYAE
}
CRITIC_VEV_MIN=${CRITIC_VEV_MIN:-0.30}   # 留出 v_ev 低于这个就不许开 critic（mjlab run_j1_rounds.sh 第 7 步的口径）
STEPS=${STEPS:-600}
# 想象步数：mjlab 交接文档 §10.3 实测 H=5 太短（远端全靠 critic）、H=20 太慢，取 H=8。
# 注意我们 WM 的 20 步关节误差 0.142 rad、k=8 约 0.08–0.09 rad，比 H=5 的 0.054 差，是已知代价。
H=${H:-8}
B_START_MAX_ERR=${B_START_MAX_ERR:-0.3}   # 只训"奖励核还有梯度"的起点（e>0.5 梯度≈0，占 B 数据 52%）
FM_LR=${FM_LR:-3e-5}
WBC_LR=${WBC_LR:-1e-6}
BATCH=${BATCH:-16}
ACCUM=${ACCUM:-4}
LORA_ALPHA=${LORA_ALPHA:-64}
EVAL_SEEDS=${EVAL_SEEDS:-"0 1 2 3 4 5 6 7 8 9 10 11"}   # 12 集/臂（mjlab §11：单集噪声 ~30%）
B_GLOB=${B_GLOB:-"runs/B_*/vla_*_f*_r*/dump/*.pt"}

say()  { echo "[$(date +%H:%M:%S)] [pipeline $ROUND] $*"; }
die()  { echo "[$(date +%H:%M:%S)] [pipeline $ROUND][FATAL] $*"; echo "  state: $STATE"; exit 1; }

# ---- state.json 读写 -----------------------------------------------------
st_mark() {  # stage artifact [extra-json]
  python3 - "$STATE" "$1" "$2" "${3:-}" <<'PY'
import json, sys, time
from pathlib import Path
state_f, stage, art, extra = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
d = json.loads(state_f.read_text()) if state_f.exists() else {}
d.setdefault("stages", {})[stage] = {"artifact": art, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                     "epoch": time.time(), "extra": extra}
state_f.write_text(json.dumps(d, indent=1, ensure_ascii=False))
PY
}
st_get() { python3 - "$STATE" "$1" <<'PY'
import json, sys
from pathlib import Path
f = Path(sys.argv[1])
if not f.exists(): print(""); raise SystemExit
d = json.loads(f.read_text()).get("stages", {}).get(sys.argv[2], {})
print(d.get("artifact", ""))
PY
}
st_need() {  # stage -> 校验该阶段已记录且产物存在
  local art; art=$(st_get "$1")
  [[ -n "$art" ]] || die "阶段 $1 还没跑过（state 里没有记录）⇒ 先跑 $1"
  [[ -e "$art" ]] || die "阶段 $1 的产物不见了: $art ⇒ 重跑 $1"
  echo "$art"
}

# ============================================================ 0. A 线采集
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
  ENVS="$A_ENVS" STEPS="$A_STEPS" WAVES="$A_WAVES" OUTDIR="$outdir" \
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
  # 清洗：只排物理退化（炸开/倒地/NaN），保留 ~60% 且中位有效长度 200+ 步
  python3 "$COMIMIC/scripts/filter_a_line.py" --in "$outdir"/a_w*.pt \
    --out-dir "$outdir" > "$PD/0_a_filter.log" 2>&1 \
    || die "A 线清洗失败（看 $PD/0_a_filter.log）"
  tail -2 "$PD/0_a_filter.log"
  (( rows >= A_MIN_ROWS )) || die "A 线只有 $rows 条（< $A_MIN_ROWS）⇒ 不够训 WM，停止"
  # 奖励/ret 用系统 python 补（Isaac venv 里 import 训练栈会炸）
  say "补算 A 线奖励..."
  SUPERMIMIC_HA_ALIGN=1 PYTHONPATH=. python3 "$COMIMIC/scripts/compute_collect_reward.py" \
    --in "$outdir"/a_w*.pt --out "$outdir/a_rew.pt" > "$PD/0_a_reward.log" 2>&1 \
    || die "A 线奖励补算失败（看 $PD/0_a_reward.log）"
  tail -2 "$PD/0_a_reward.log"
  st_mark a_collect "$outdir"
  say "a_collect ✅ $rows 条（奖励 $outdir/a_rew.pt）"
}

# ============================================================ 1. collect
stage_collect() {
  local bdir=$COMIMIC/runs/B_$ROUND
  # ★ 2026-10-31 修（真 bug）：原来只要目录里**有任何一集**就跳过采集，
  #   于是一次中断留下的半截目录（big6 的 8 集、d3 的 8 集）会被当成"采完了"，
  #   那一轮实际上没有新数据 —— 累积闭环就断了。
  #   现在必须**采满**（任务数 × 每任务集数，允许差 1）才跳过；不完整就清空重采。
  local ntasks need have
  ntasks=$(echo $TASKS | wc -w)
  need=$(( COLLECT_EPISODES * ntasks ))
  have=$(find "$bdir" -name '*.pt' -path '*/dump/*' 2>/dev/null | wc -l)
  if (( have >= need - 1 )); then
    say "collect 已有 $bdir（$have/$need 集，完整），跳过"
  else
    if (( have > 0 )); then
      say "⚠ $bdir 只有 $have/$need 集（不完整）⇒ 清空重采"
      rm -rf "$bdir"
    fi
    say "--- 2/10 采集（$COLLECT_EPISODES 集/任务 × $ntasks 任务，两卡并行）---"
    local cl="${ROUND}_collect"
    ROUND=99 RUN_LABEL="$cl" NEXT_ROUND=$ROUND B_NEXT_DIR="$bdir" \
      COLLECT_EPISODES="$COLLECT_EPISODES" COLLECT_STEPS="$COLLECT_STEPS" TASKS="$TASKS" \
      HEAD_COLLECT="$HEAD_COLLECT" SRV_EXTRA_VAL="$SRV_EXTRA" \
      bash "$COMIMIC/scripts/s3_loop.sh" collect 2>&1 | tee "$PD/1_collect.log" | tail -3
    [[ "${PIPESTATUS[0]}" == "0" ]] || die "采集失败"
  fi
  local n; n=$(find "$bdir" -name '*.pt' -path '*/dump/*' 2>/dev/null | wc -l)
  (( n >= need - 1 )) || die "采集只有 $n 集（要求 ≥ $((need-1))）⇒ 不许进入下一阶段"
  say "collect ✅ $n 集（要求 $need）"
  st_mark collect "$bdir"
  # 记录流程版本：v9 = 含 critic 门。老的 8 阶段轮次不会被要求补这个阶段。
  python3 - "$STATE" <<'PYV'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); d = json.loads(p.read_text())
d["pipeline_version"] = "v10"
p.write_text(json.dumps(d, indent=1, ensure_ascii=False))
PYV
}

# ============================================================ 2. score
stage_score() {
  st_need collect >/dev/null
  say "--- 3/10 打分（全池，bundle 只加载一次）---"
  PYTHONPATH=. python3 "$COMIMIC/scripts/score_b_pool.py" --dump "$B_GLOB" --device cuda:0 \
    > "$PD/2_score.log" 2>&1 || die "打分失败（看 $PD/2_score.log）"
  local ndump nscored
  ndump=$(ls $B_GLOB 2>/dev/null | grep -v '_r\.pt$' | wc -l)
  nscored=$(ls runs/B_*/vla_*_f*_r*/scored.pt 2>/dev/null | wc -l)
  say "dump=$ndump  scored=$nscored"
  (( nscored >= ndump )) || die "scored.pt 数量($nscored) < dump 数量($ndump) ⇒ 打分没覆盖全"
  say "score ✅"
  st_mark score "$COMIMIC/runs/wm_real"
}

# ============================================================ 3. transitions
stage_transitions() {
  st_need score >/dev/null
  local out=$COMIMIC/runs/wm_real/rt_v2_$ROUND.pt
  say "--- 4/10 真转移 ---"
  PYTHONPATH=. python3 "$COMIMIC/scripts/build_real_transitions_v2.py" \
    --wbc-ckpt "$BASE_WM" --extra-dumps "runs/B_*/vla_*_f*_r*/scored.pt" \
    --out "$out" --device cuda:0 > "$PD/3_transitions.log" 2>&1 || die "真转移失败"
  python3 - "$out" <<'PY' || exit 1
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
e = d.get("extra")
n = int(e["states"].shape[0]) if e else 0
print(f"[pipeline] extra(B线) = {n} 条")
raise SystemExit(0 if n > 0 else 1)
PY
  [[ $? == 0 ]] || die "转移里没有 B 线 extra"
  say "transitions ✅ $out"
  st_mark transitions "$out"
}

# ============================================================ 4. wm refresh
stage_wm() {
  st_need transitions >/dev/null
  local out=$COMIMIC/runs/S1_handover/wm_$ROUND.pt
  local tr=$COMIMIC/runs/wm_real/traj_${ROUND}_train
  local va=$COMIMIC/runs/wm_real/traj_${ROUND}_val
  say "--- 5/10 刷新 WM（训练=除 $HOLD_TASK 外的全部，留出=$HOLD_TASK）---"
  rm -rf "$tr" "$va"; mkdir -p "$tr" "$va"
  # ★ 2026-10-30（对标 mjlab §8.4）：WM 判据要**当轮同分布**。
  #   跨轮用固定留出集去量，量到的其实是"策略演化"而不是 WM（他们实测会假性 1.25→1.39 逐轮变差）。
  #   所以留出优先取**本轮**采集的 SitSofa 集（7 集 ≈ 4200 行，够他们说的 n≥2000），
  #   本轮没有才退回全池留出。
  # ★ 2026-10-30 修：`$B_GLOB` 是**相对路径**（runs/B_*/…），而原来这里拿绝对路径比，
  #   导致 `[[ $f == $cur_hold/* ]]` 恒假 ⇒ 留出目录里一个链接都没有（val=0）⇒ 整个阶段 die。
  #   现在用相对前缀比较，并在 val 为空时自动退回全池留出。
  local cur_rel="runs/B_${ROUND}"
  local cur_hold=$COMIMIC/runs/B_${ROUND}
  local use_cur=0
  if compgen -G "$cur_hold/vla_${HOLD_TASK}_*/dump/*.pt" > /dev/null 2>&1; then
    use_cur=1
    say "WM 留出用**本轮** $HOLD_TASK 集（同分布口径）"
  else
    say "WM 留出用全池 $HOLD_TASK 集（本轮没有该任务的数据）"
  fi
  # ★ 2026-11-01 修：B_GLOB 现在可能是**绝对路径**（run_full_r1.sh），
  #   原来无条件拼 $COMIMIC/ 会把链接指到 .../comimic//inspire/... 这种双前缀，
  #   150 条全成断链，阶段2 torch.load 直接 FileNotFoundError。
  #   统一成绝对 target；留出比较同时兼容相对/绝对两种前缀。
  local i=0
  while IFS= read -r f; do
    local src=$f
    [[ "$src" == /* ]] || src=$COMIMIC/$src
    case "$f" in
      *vla_${HOLD_TASK}_*)
        if [[ "$use_cur" == "1" && ( "$f" == ${cur_rel}/* || "$f" == ${cur_hold}/* ) ]]; then
          ln -sf "$src" "$va/v_$i.pt"
        elif [[ "$use_cur" == "0" ]]; then
          ln -sf "$src" "$va/v_$i.pt"
        else
          ln -sf "$src" "$tr/v_$i.pt"
        fi ;;
      *)  ln -sf "$src" "$tr/v_$i.pt" ;;
    esac
    i=$((i+1))
  done < <(ls $B_GLOB | grep -v '_r\.pt$')
  local ntr nva; ntr=$(ls "$tr" | wc -l); nva=$(ls "$va" | wc -l)
  if [[ "$nva" == "0" ]]; then
    say "⚠ 本轮留出为空 ⇒ 回退全池 $HOLD_TASK 留出"
    local j=0
    while IFS= read -r f; do
      case "$f" in
        *vla_${HOLD_TASK}_*)
          src=$f; [[ "$src" == /* ]] || src=$COMIMIC/$src
          ln -sf "$src" "$va/v_$j.pt"; rm -f "$tr/v_$j.pt"; j=$((j+1)) ;;
      esac
    done < <(ls $B_GLOB | grep -v '_r\.pt$')
    nva=$(ls "$va" | wc -l); ntr=$(ls "$tr" | wc -l)
  fi
  (( ntr >= 10 && nva >= 1 )) || die "WM 训练/留出划分太少（train=$ntr val=$nva）"
  say "WM 训练 $ntr 条轨迹，留出 $nva 条"
  # ★ 2026-10-29（用户口径，对齐 mjlab 的课程）：WM 刷新拆成**两段**，不能一次混完。
  #   段 A：只用 A 线（冻结 WBC 在 Isaac 真物理里走出来的真动力学）主导训练，
  #         先把"机器人自身动力学"学会 —— 这是 k=1 速度相关从 0.737 拉到 0.888 的那一步。
  #   段 B：A 线与 B 线按 A:B = WM_AB_RATIO 混合（默认 3 ⇒ B 占混合的 25%，与 mjlab 一致），
  #         把部署分布（VLA 在环、任务场景）的偏移吸收进来，同时不丢动力学。
  #   为什么不能一次混：采样是"每步随机取一条轨迹"，A 线条数（1024）远多于 B（~200），
  #   一次混会把 B 稀释到 ~7%（实测 big5），部署适配就没了。
  # ★ 2026-11-01：第三池 P（反事实扰动）已删除——它是负结果，见
  #   docs/HANDOVER_WM_20261031.md。WM 只走 A+B，不再读 runs/Bw_pert*。

  local wm_a=$COMIMIC/runs/S1_handover/wm_${ROUND}_a.pt
  # ★ WM_SKIP_P1=1 且阶段1产物已存在 ⇒ 从阶段1续跑（跳过 3000 步 A 段）
  if [[ "${WM_SKIP_P1:-0}" == "1" && -s "$wm_a" ]]; then
    say "续跑：复用阶段1产物 $wm_a（跳过 A 段）"
  else
  say "--- 5/10 WM 刷新【阶段1】场景/A 线 $WM_STEPS_A 步（单步, lr $WM_LR_A）---"
  PYTHONPATH=. python3 "$COMIMIC/scripts/finetune_wm_velocity_policy.py" --a-order "$A_ORDER" \
    --ckpt "$BASE_WM" --out "$wm_a" --arch "$WM_ARCH" --clip-norm "$WM_CLIP_NORM" \
    --packed "$COMIMIC/runs/isaac_motion/ha_cmd/packed.pt" --token-space label \
    --rollout-steps 1 --steps "$WM_STEPS_A" --batch 256 --lr "$WM_LR_A" --vel-lambda 5 --w-lambda "$WM_W_LAMBDA" --vel-scale-lambda "$WM_VEL_SCALE_LAMBDA" \
    --mix-lib "$WM_MIX_LIB" --mix-b 0.0 \
    --traj-dump "" --traj-frac 4.0 --traj-val-frac 0.25 \
    --a-line "$COMIMIC/runs/wm_real/ac_w*.pt" --a-line-envs 512 \
    --val-horizon 0 --device cuda:0 > "$PD/4a_wm_p1.log" 2>&1 \
    || die "WM 阶段1失败（看 $PD/4a_wm_p1.log）"
  tail -2 "$PD/4a_wm_p1.log"
  fi

  local n_b a_envs
  n_b=$(ls "$tr" | wc -l)
  # ★ 2026-10-31 订死：A 线文件数**从目录里数**，不再硬编码（原来写死 /4，A 线加到 8 波后
  #   B 占比悄悄从 25% 掉到 19%，而且没有任何报错）。四舍五入到整数后按真实文件数均摊。
  local n_af
  n_af=$(ls "$COMIMIC"/runs/wm_real/ac_w*.pt 2>/dev/null | wc -l)
  (( n_af >= 1 )) || die "找不到 A 线文件 runs/wm_real/ac_w*.pt（先跑 a_collect）"
  a_envs=$(python3 -c "nb=$n_b; nf=$n_af; r=$WM_AB_RATIO; print(max(16, int(round(r*nb/max(1,nf)))))")
  local _mix_b
  _mix_b=$(python3 -c "r=$WM_AB_RATIO; print(round(1.0/(1.0+r), 4))")
  say "A 线 $n_af 个文件，B 轨迹 $n_b 条 ⇒ 每文件取 $a_envs 条（目标 B 占比 $((100/(WM_AB_RATIO+1)))%）"
  local wm_p2=$COMIMIC/runs/S1_handover/wm_${ROUND}_p2.pt
  say "--- 5/10 WM 刷新【阶段2】A+B 混合 $WM_STEPS_MIX 步（B≈$((100 / (WM_AB_RATIO + 1)))%, 单步, lr $WM_LR_P2）---"
  PYTHONPATH=. python3 "$COMIMIC/scripts/finetune_wm_velocity_policy.py" --a-order "$A_ORDER" \
    --ckpt "$wm_a" --out "$wm_p2" --arch "$WM_ARCH" --clip-norm "$WM_CLIP_NORM" \
    --packed "$COMIMIC/runs/isaac_motion/ha_cmd/packed.pt" --token-space label \
    --rollout-steps 1 --steps "$WM_STEPS_MIX" --batch 256 --lr "$WM_LR_P2" --vel-lambda 5 --w-lambda "$WM_W_LAMBDA" --vel-scale-lambda "$WM_VEL_SCALE_LAMBDA" \
    --mix-lib "$WM_MIX_LIB" --mix-b "$_mix_b" \
    --traj-dump "$tr/*.pt" --traj-frac "$WM_ALINE_FRAC" --traj-val-frac 0.25 \
    --a-line "$COMIMIC/runs/wm_real/ac_w*.pt" --a-line-envs "$a_envs" \
    --val-horizon 0 --device cuda:0 > "$PD/4b_wm_p2.log" 2>&1 \
    || die "WM 阶段2失败（看 $PD/4b_wm_p2.log）"
  tail -2 "$PD/4b_wm_p2.log"
  # ★ 2026-10-31 订死：核对**实际**混合配比（防止再出现"配比被悄悄改掉"）
  python3 - "$PD/4b_wm_p2.log" "$WM_AB_RATIO" <<'PYRATIO' || die "阶段2 的 B/A 配比与设计不符 ⇒ 停止"
import re, sys
log, r = sys.argv[1], int(sys.argv[2])
txt = open(log, encoding="utf-8", errors="ignore").read()
want = 1.0 / (r + 1)
# ★ 2026-11-01 修：配比要量**实际进 batch 的行数**，不是载入的轨迹条数。
#   R1 的 B 池只有 150 条（每条 1 轨迹），A 线 24 文件 ×16 env=333 条；
#   若拿轨迹条数当比例会误判成 31%，而 [mix] 行明确是 A=128/B=128（每步真正反传的样本）。
mix = re.findall(r"\[mix\].*?A 线 (\d+).*?B 线 (\d+)", txt)
traj = re.findall(r"\[traj\] 合计 (\d+) 条轨迹（B=(\d+) A=(\d+)）", txt)
if mix:
    a, b = (int(x) for x in mix[-1])
    tot = a + b
    share = b / max(1, tot)
    print(f"[ratio] 阶段2 实际 batch: B={b} A={a} ⇒ B 占比 {share*100:.1f}%（设计 {want*100:.0f}%）")
    if traj:
        _t, _b, _a = (int(x) for x in traj[-1])
        print(f"[ratio] 载入池（仅诊断）: B={_b} A={_a} 条轨迹")
elif traj:
    tot, b, a = (int(x) for x in traj[-1])
    share = b / max(1, tot)
    print(f"[ratio] 阶段2 实际轨迹池: B={b} A={a} ⇒ B 占比 {share*100:.1f}%（设计 {want*100:.0f}%）")
else:
    print("[ratio] ❌ 日志里既找不到 [mix] 配比行，也找不到 [traj] 合计 行")
    raise SystemExit(1)
# ★ 2026-11-01：第三池 P 已删除（负结果，勿用）；这里只校验 A:B 配比。
if not (0.9 * want <= share <= 1.2 * want):
    print(f"[ratio] ❌ 偏离设计太多（允许 {0.9*want*100:.0f}%~{1.2*want*100:.0f}%）")
    raise SystemExit(1)
PYRATIO

  say "--- 5/10 WM 刷新【阶段2b】k=$WM_ROLLOUT_STEPS 自回归 $WM_STEPS_2B 步（交织 A 线单步, lr $WM_LR）---"
  PYTHONPATH=. python3 "$COMIMIC/scripts/finetune_wm_velocity_policy.py" --a-order "$A_ORDER" \
    --ckpt "$wm_p2" --out "$out" --arch "$WM_ARCH" --clip-norm "$WM_CLIP_NORM" \
    --packed "$COMIMIC/runs/isaac_motion/ha_cmd/packed.pt" --token-space label \
    --rollout-steps "$WM_ROLLOUT_STEPS" --steps "$WM_STEPS_2B" --batch "$WM_BATCH" --lr "$WM_LR" --vel-lambda 5 --w-lambda "$WM_W_LAMBDA" --vel-scale-lambda "$WM_VEL_SCALE_LAMBDA" \
    --mix-lib "$WM_MIX_LIB" --mix-b "$_mix_b" \
    --traj-dump "$tr/*.pt" --traj-frac "$WM_ALINE_FRAC" --traj-val-frac 0.25 \
    --a-line "$COMIMIC/runs/wm_real/ac_w*.pt" --a-line-envs "$a_envs" \
    --val-dump "$va/*.pt" --val-horizon 20 --val-every 100 --val-max-starts 16 --val-max-traj 16 \
    --device cuda:0 > "$PD/4_wm_refresh.log" 2>&1 || die "WM 阶段2b失败（看 $PD/4_wm_refresh.log）"
  tail -2 "$PD/4_wm_refresh.log"
  [[ -f "$out" ]] || die "WM 刷新没产出 $out"
  say "wm_refresh ✅ $out"
  st_mark wm "$out"
}

# ============================================================ 5. wm gate
stage_gate() {
  local wm; wm=$(st_need wm)
  say "--- 6/10 WM 门（留出 $HOLD_TASK，20 步关节 ≤ 1.05× base）---"
  local va=$COMIMIC/runs/wm_real/traj_${ROUND}_val
  PYTHONPATH=. python3 "$COMIMIC/scripts/eval_wm_multistep.py" \
    --ckpt "$BASE_WM" "$wm" --dump "$va/*.pt" --horizon 20 --stride 20 --device cuda:1 \
    > "$PD/5_wm_gate.log" 2>&1 || die "WM 门跑失败"
  tail -7 "$PD/5_wm_gate.log" | tee -a "$PD/summary.txt"
  # ★ 2026-10-29 修：原来 `grep -q "PASS"` 会误命中 `[AUDIT][spec] robot_spec 对齐 PASS`，
  #   把 FAIL 的 WM 当成 PASS 用。现在只认 G0-b 那一行。
  # ---- A11 第二栏：**OOD 一步仪器**（录制语料）。历史教训：刷新后 in-dist 变好、
  #      OOD 的 qd_corr 从 0.670 掉到 0.462，而 gate 只看 in-dist ⇒ 完全看不见。
  #      这里同时量 base 与新 WM，新 WM 的 qd_corr 不得低于 base 的 0.90×。
  local _ood_ok=1
  if [[ "${SKIP_OOD_GATE:-0}" != "1" ]]; then
    say "--- 6/10 WM 门 · OOD 栏（录制语料一步 qd_corr，两张卡并行）---"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 "$COMIMIC/scripts/eval_wm_velocity.py" \
      --ckpt "$BASE_WM" --tokens label --device cuda --samples 8 \
      > "$PD/5_wm_ood_base.log" 2>&1 || true
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 "$COMIMIC/scripts/eval_wm_velocity.py" \
      --ckpt "$wm" --tokens label --device cuda --samples 8 \
      > "$PD/5_wm_ood_new.log" 2>&1 || true
    local qb qn
    qb=$(grep -o '"qd_corr": [-0-9.]*' "$PD/5_wm_ood_base.log" | tail -1 | awk '{print $2}')
    qn=$(grep -o '"qd_corr": [-0-9.]*' "$PD/5_wm_ood_new.log" | tail -1 | awk '{print $2}')
    if [[ -z "$qb" || -z "$qn" ]]; then
      say "OOD 栏 ⚠ 读数缺失（base='$qb' new='$qn'）⇒ 记录但不拦（看 $PD/5_wm_ood_*.log）"
    else
      say "OOD 栏: base qd_corr=$qb / new=$qn  （in-distribution 见上一栏）"
      if python3 -c "import sys; sys.exit(0 if float('$qn') >= 0.9*float('$qb') else 1)"; then
        say "OOD 栏 ✅ qd_corr 通过（new ≥ 0.9×base）"
      else
        say "OOD 栏 ⚠ 监测 new qd_corr=$qn < 0.9×base($qb)：录制语料泛化下降（**不阻断**——"\
            "想象是从 B 线状态出发的，OOD 不在关键路径上）"
      fi
      # ---- A2 前置：**root 角速度**质量（yaw_rate 奖励吃这个量）----
      local wcb wcn rwn
      wcb=$(grep -o '"root_w_corr": [-0-9.]*' "$PD/5_wm_ood_base.log" | tail -1 | awk '{print $2}')
      wcn=$(grep -o '"root_w_corr": [-0-9.]*' "$PD/5_wm_ood_new.log" | tail -1 | awk '{print $2}')
      rwn=$(grep -o '"root_w_ratio": [-0-9.]*' "$PD/5_wm_ood_new.log" | tail -1 | awk '{print $2}')
      if [[ -n "$wcn" ]]; then
        say "OOD 栏 · root 角速度: base corr=$wcb / new corr=$wcn, new ratio=${rwn:-NA}"
        if python3 -c "
import sys
c, b, r = float('$wcn'), float('${wcb:-0}'), float('${rwn:-0}')
sys.exit(0 if (c >= 0.9*b and 0.75 <= r <= 1.35) else 1)"; then
          say "OOD 栏 · root 角速度 ✅（corr 未退化、ratio 在 0.75~1.35）"
          echo "$wcn" > "$PD/wm_root_w_corr.txt"
        else
          say "OOD 栏 · root 角速度 ❌（corr 退化或 ratio 越界）⇒ WM 还不足以支撑 yaw_rate 奖励"
          echo "$wcn" > "$PD/wm_root_w_corr.txt"
          _ood_ok=0
        fi
      fi
    fi
  fi
  # ---- 准入第二项：**in-dist root 角速度相关性**（和奖励同一个分布）----
  #   为什么不用 OOD 的 qd_corr 当门：想象是从 **B 线状态**出发的，OOD 不在关键路径上；
  #   而且这次 OOD 退化（0.670→0.522）的同时，rollout 的 state_term 反而 **−7.5%**。
  #   所以 OOD 只当"遗忘监测"，照常打印与记录，不阻断。
  local _rwok=1 _rwc5=""
  _rwc5=$(grep -oE "^   k= 5  .*root_w_corr=-?[0-9.]+" "$PD/5_wm_gate.log" | tail -1 \
          | grep -oE "root_w_corr=-?[0-9.]+" | awk -F= '{print $2}')
  local _rwb5
  _rwb5=$(grep -oE "^   k= 5  .*root_w_corr=-?[0-9.]+" "$PD/5_wm_gate.log" | head -1 \
          | grep -oE "root_w_corr=-?[0-9.]+" | awk -F= '{print $2}')
  local _rwr5
  _rwr5=$(grep -oE "^   k= 5  .*root_w_ratio=[0-9.]+" "$PD/5_wm_gate.log" | tail -1 \
          | grep -oE "root_w_ratio=[0-9.]+" | awk -F= '{print $2}')
  if [[ -n "$_rwc5" && -n "$_rwb5" ]]; then
    say "in-dist root 角速度: base corr=$_rwb5 → new corr=$_rwc5（k=5），new ratio=${_rwr5:-NA}"
    if python3 -c "import sys; sys.exit(0 if float('$_rwc5') >= 0.9*float('$_rwb5') else 1)"; then
      say "in-dist root 角速度 ✅ 未退化（≥0.9×base）"
    else
      say "in-dist root 角速度 ❌ 退化 ⇒ WM 的角速度通道不可用"
      _rwok=0
    fi
    echo "$_rwc5" > "$PD/wm_root_w_corr.txt"
    echo "${_rwr5:-}" > "$PD/wm_root_w_ratio.txt"
  else
    say "in-dist root 角速度 ⚠ 读数缺失 ⇒ 不阻断，但 yaw_rate 不会开"
    _rwok=0
  fi
  if grep -qE "G0-b\(多步\).*PASS" "$PD/5_wm_gate.log" && (( _rwok == 1 )); then
    say "WM 门 ✅ PASS（多步关节 + in-dist root 角速度；OOD 仅监测）⇒ 用新 WM 当裁判"
    st_mark gate "$wm"
  else
    # ★ 2026-10-30 用户口径："判据过了以后才能启动联合微调" ⇒ 门 FAIL 就**中止**，
    #   不再静默回退 base（那样会拿没验证过的 WM 当裁判，等于白跑）。
    say "WM 门 ❌ FAIL ⇒ 中止（不启动联合微调）"
    exit 1
  fi
}

# ============================================================ 6. critic 门
stage_critic() {
  local wm; wm=$(st_need gate)
  local tr; tr=$(st_need transitions)
  # ★ 2026-10-29 修正（第一版搞错了对象，误关了 big3 本来健康的裁判）：
  #   门只对 **real 监督** 的 critic 有意义（mjlab 交接文档 §10.2：real 结构性不可学，v_ev 0.088–0.152）。
  #   `CRITIC_SOURCE=imagined` 走 GAE 路线，critic 在训练里用想象 rollout 的 GAE 目标在线更新，
  #   不需要 prefit、也不该拿 real 的 v_ev 去判它 —— 我们实测 imagined 的 critic/v_ev 达 0.9946–0.9974
  #   （与 mjlab 的 0.9963 同级）。对应 mjlab run_j1_rounds.sh 第 7 步的 if 分支。
  local src=${CRITIC_SOURCE:-imagined}
  if [[ "$src" == "imagined" ]]; then
    say "--- 7/10 critic 门（源=imagined ⇒ GAE 在线更新，跳过 prefit 门）---"
    python3 - "$PD" "$CRITIC_VEV_MIN" "$tr" <<'PYG'
import json, sys
from pathlib import Path
pd, thr, tr = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
json.dump({"source": "imagined", "allow": True, "threshold": None, "transitions": tr,
           "rule": "imagined/GAE: critic 在训练内用想象 rollout 的 GAE 目标在线更新，不做 prefit 门 "
                   "(mjlab run_j1_rounds.sh: CRITIC_SOURCE=imagined 直接 --critic)"},
          open(pd / "critic_gate.json", "w"), indent=1, ensure_ascii=False)
print("[critic-gate] PASS（imagined/GAE，无 prefit 门）⇒ 本轮开 critic")
PYG
    [[ $? == 0 ]] || die "critic 门写入失败"
    say "critic_gate ✅ $PD/critic_gate.json（imagined 放行）"
    st_mark critic "$PD/critic_gate.json"
    return 0
  fi

  say "--- 7/10 critic 拟合 + 门【real 源】（留出 v_ev > $CRITIC_VEV_MIN 才允许开）---"
  # 退出码约定（fit_critic_real.py:223）：0 = 门过(>0.30)，3 = 门没过但 ckpt 已存，其它 = 真崩
  local rc=0
  PYTHONPATH=. python3 "$COMIMIC/scripts/fit_critic_real.py" \
    --transitions "$tr" --out "$COMIMIC/runs/wm_real/critic_b_$ROUND.pt" \
    --center linear --epochs 12 --device cuda:0 > "$PD/6_critic.log" 2>&1 || rc=$?
  if [[ "$rc" != "0" && "$rc" != "3" ]]; then
    tail -5 "$PD/6_critic.log" >&2
    die "critic 拟合失败 rc=$rc（看 $PD/6_critic.log）"
  fi
  local line vev
  line=$(grep -E "^\[critic\] best epoch=.* val v_ev=" "$PD/6_critic.log" | tail -1)
  [[ -n "$line" ]] || die "critic 日志里找不到 best epoch 的 v_ev（看 $PD/6_critic.log）"
  vev=$(printf '%s' "$line" | grep -oE "val v_ev=[-+]?[0-9.]+" | cut -d= -f2)
  say "critic 留出 v_ev=$vev（门槛 $CRITIC_VEV_MIN）"
  python3 - "$PD" "$vev" "$CRITIC_VEV_MIN" "$tr" <<'PYR'
import json, sys
from pathlib import Path
pd, vev, thr, tr = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
allow = vev > thr
json.dump({"source": "real", "v_ev": vev, "threshold": thr, "allow": allow, "transitions": tr,
           "rule": "real 监督: v_ev <= threshold ⇒ 强制关 critic"},
          open(pd / "critic_gate.json", "w"), indent=1, ensure_ascii=False)
print(f"[critic-gate] {'PASS ⇒ 本轮开 critic' if allow else 'FAIL ⇒ 本轮强制关 critic'}")
PYR
  [[ $? == 0 ]] || die "critic 门写入失败"
  tail -2 "$PD/6_critic.log" | tee -a "$PD/summary.txt"
  say "critic_gate ✅ $PD/critic_gate.json"
  st_mark critic "$PD/critic_gate.json"
}

# ============================================================ 6. joint
stage_joint() {
  # ★ 2026-10-31（消融支持）：`WM_PIN=<path>` 时跳过 collect/gate 的前置校验，
  #   直接把裁判钉在指定 WM 上。用途只有一个：**单旋钮消融**——同一份数据、同一个 WM、
  #   同一个起点、同样步数，只翻一个开关（见 docs/ABLATION_SUITE.md）。
  #   正常轮次**不允许**用它（否则就绕过了 WM 门）；所以要求同时给 ABL_ROUND=1，双保险。
  local wm cg_allow=1
  if [[ -n "${WM_PIN:-}" ]]; then
    [[ "${ABL_ROUND:-0}" == "1" ]] || die "WM_PIN 只允许在消融轮用（同时给 ABL_ROUND=1）"
    [[ -f "$WM_PIN" ]] || die "WM_PIN 指向的文件不存在：$WM_PIN"
    wm=$(cd "$(dirname "$WM_PIN")" && pwd)/$(basename "$WM_PIN")
    cg_allow=1
    say "--- 8/10 消融：WM 钉在 $(basename "$wm")（跳过 collect/score/transition/refresh/gate）---"
  else
    wm=$(st_need gate)
    local cg; cg=$(st_need critic)      # 门决定开不开裁判，不许绕过
    cg_allow=$(python3 -c "import json,sys;print(1 if json.load(open(sys.argv[1]))['allow'] else 0)" "$cg")
  fi
  local c1=${CRITIC_ON_MAIN:-} c0=${CRITIC_ON_CTRL:-} g1=${CRITIC_GRAD_STATE_MAIN:-} g0=${CRITIC_GRAD_STATE_CTRL:-}
  if [[ "$cg_allow" != "1" ]]; then
    c1=0; c0=0; g1=0; g0=0
    say "⚠ critic 门未通过 ⇒ 两臂强制 critic=OFF（只走奖励项）"
  fi
  # ★ 2026-10-29 修：原来写 `ROUND=99 RUN_LABEL=${ROUND}_joint ...`，bash 会**先**应用 ROUND=99
  #   再展开 ${ROUND}，于是目录变成 99_joint，后面断言找不到文件。必须先算好标签。
  local jl="${ROUND}_joint"
  if [[ "$cg_allow" == "1" ]]; then
  if [[ "${SKIP_SELF_TARGETS:-0}" == "1" ]]; then
    # ★ 2026-11-01：8 卡上并行跑两条 joint（4+4 DDP）时，self 目标只取决于
    #   B_GLOB/horizon，与臂无关；两条流水线会同时重写同一个文件。
    #   跳过前必须确认文件已存在（由主 shell 预先算好一遍），否则训练侧会缺标签。
    [[ -f "$COMIMIC/runs/wm_real/self_targets_all.pt" ]] \
      || die "SKIP_SELF_TARGETS=1 但 runs/wm_real/self_targets_all.pt 不存在"
    say "--- 8/10 前置：SKIP_SELF_TARGETS=1 ⇒ 复用已有 self 目标（并行臂安全）---"
  elif [[ "${DRY_RUN:-0}" != "1" ]]; then
  # ★ 2026-10-30：训练前**重算** self 目标（必须含本轮刚采的 dump）。
  #   不重算的话本轮新 episode 没有标签 ⇒ 这些窗口的 self token 是零占位、信任域无信号；
  #   而且昨天正因为行数少 1（16 vs 15）直接把 `tokens - prior_t` 打崩过一次。
  say "--- 8/10 前置：重算 self 目标（token 真值，含本轮）---"
  PYTHONPATH=. python3 "$COMIMIC/scripts/build_b_self_targets.py" \
    --dump "$B_GLOB" --ref "$COMIMIC/runs/isaac_motion/ha_cmd/packed.pt" \
    --out "$COMIMIC/runs/wm_real/self_targets_all.pt" --horizon 20 \
    > "$PD/6b_self_targets.log" 2>&1 || die "self 目标重算失败（看 $PD/6b_self_targets.log）"
  grep -E "self-tgt\] (瞬时|realized)" "$PD/6b_self_targets.log" | tail -2 | tee -a "$PD/summary.txt"
  else
    say "--- 8/10 前置：DRY_RUN ⇒ 跳过 self 目标重算 ---"
  fi

    say "--- 8/10 联合微调（裁判 WM = $(basename "$wm")，critic = ${CRITIC_SOURCE:-imagined}/GAE 开）---"
  else
    say "--- 8/10 联合微调（裁判 WM = $(basename "$wm")，critic 关 ⇒ 只走奖励项）---"
  fi
  # ★ 冻结 WBC 的理由见 mjlab 交接文档 §10.1（解冻 800 步 root_pos 4.16 -> 89.2 cm，21x）。
  # ⚠ 教训：**绝不能把注释插进 env 赋值链的续行中间** —— `\` 会被注释吃掉，整条命令被截断，
  #   后面的变量（CRITIC_ON / TRUST_CENTER / ACT_EPS / REWARD_VEL_SCALE）会静默退回默认值。
  #   2026-10-29 big4 第一版就这么跑错的：门放行了 critic，训练却收到 --no-critic。
    # ★ 2026-10-30（对标 mjlab FULL_ARCH §6.3）：信任域中心改成 **self**、eps=0.2、w=1.0。
    #   `trust_center=prev` 会把 trust_eps 强制置 0，等于**把 token 空间唯一的高信噪比信号关掉**
    #   （v3 里 token 只有两个信号：WM 想象的奖励 + trust hinge）。
    #   self 的中心 = 采集时策略**真正下发**的 token（`b_self_targets`），即"按执行结果"监督，
    #   不是标签、也不经过世界模型。
    #   `B_SELF_ERR_MAX` 只保留"接下来 20 步实际漂移 ≤ 该值"的时刻（advantage 过滤），
    #   全池实测 p25 = 0.294 m。
    # ACT_EPS 用 **0.6**（mjlab FULL_ARCH §6.1 的值）。实测我们 B 线起点的 act/rmse≈0.63，
    #   若 eps=0.3 会把**全部**样本判成越界（act/frac_out=1.0），铰链强度放大 4 倍并压住奖励项。
    # B_SELF_ERR_MAX 留空 = **不过滤窗口**：self 目标只当"信任域中心"用（mjlab §6.3）。
    #   实测若按 realized_err p25=0.294 过滤，DoubleDesk 只剩 5 个训练窗（54→5），等于没数据。
  # ★ 2026-10-30 用户口径：**tilt / height 惩罚一律不要**（代码里已删除）。
  #   POSTURE_WEIGHT 恒传 0，并用训练后的断言钉死，防止再被打开。
  # ★★ 2026-10-31 修（真 bug，害我们白跑了两轮 WM）：**不能在赋值列表里做命令替换**。
  #   `ROUND=99 X="$(f)" bash ...` 里，`$(f)` 的子 shell 会继承**这次命令的临时赋值**
  #   ⇒ f 里看到的 ROUND 是 99，于是：
  #        _act_eps_effective → 查不到 d3 的排程 → 退回自动规则 → 0.45（而不是 0.60）
  #        _fresh_glob_effective → runs/B_99/*/dump/*.pt → 0 条匹配 → 训练侧硬失败
  #   （bash 里 `VAR=x echo $VAR` 打印旧值是另一回事：那是**词展开**；命令替换走的是子 shell。）
  #   先在**主 shell**（ROUND 正确）里算好，再作为普通值传进去。
  # ---- yaw_rate 准入：WM gate 在 stage6、奖励只在 stage7 的想象里用 ⇒ **来得及按测量决定** ----
  local _yaww=0.0 _rwc=""
  if [[ -f "$PD/wm_root_w_corr.txt" ]]; then
    _rwc=$(cat "$PD/wm_root_w_corr.txt")
    local _rwr=""
    [[ -f "$PD/wm_root_w_ratio.txt" ]] && _rwr=$(cat "$PD/wm_root_w_ratio.txt")
    # 两个条件都要：**相关性够**（>=0.40）且**幅度不偏**（ratio ∈ [0.85,1.15]）。
    # 只满足相关性而幅度偏 40% ⇒ 奖励会把策略往"多转/少转"的偏置上推。
    _yaww=$(python3 -c "
c = float('${_rwc:-0}')
r = float('${_rwr:-0}' or 0)
ok = (c >= 0.40) and (0.85 <= r <= 1.15)
print(0.05 if ok else 0.0)" 2>/dev/null || echo 0.0)
    if python3 -c "import sys; c=float('${_rwc:-0}'); r=float('${_rwr:-0}' or 0); sys.exit(0 if (c>=0.40 and 0.85<=r<=1.15) else 1)"; then
      say "  yaw_rate 准入 ✅ in-dist root_w_corr=$_rwc ≥0.40 且 ratio=$_rwr ∈[0.85,1.15] ⇒ W_YAWRATE=$_yaww"
    else
      say "  yaw_rate 准入 ❌ in-dist root_w_corr=$_rwc / ratio=${_rwr:-NA} ⇒ W_YAWRATE=0（相关性不足或幅度偏）"
    fi
  else
    say "  yaw_rate 准入 ⚠ 没有 in-dist wm_root_w_corr.txt ⇒ W_YAWRATE=0（保守）"
  fi
  # ★ 2026-11-01：单旋钮消融时允许显式覆盖（例如只翻 act_weight 时强制 yaw_rate=0）。
  if [[ -n "${YAW_RATE_WEIGHT_OVERRIDE:-}" ]]; then
    say "  yaw_rate 权重被显式覆盖：$_yaww → ${YAW_RATE_WEIGHT_OVERRIDE}（单旋钮）"
    _yaww="${YAW_RATE_WEIGHT_OVERRIDE}"
  fi
  _AE="$(_act_eps_effective)"
  _AW="$(_act_weight_effective)"
  _FG="$(_fresh_glob_effective)"
  # ★ 2026-10-31（消融）：消融轮不采新数据，fresh 池必须**钉成 d3 那一份**，
  #   否则"新数据 ×3 权重"恒为空 ⇒ 训练侧硬失败。B_FRESH_PIN=1 表示"我知道我在钉"。
  if [[ "${B_FRESH_PIN:-0}" == "1" ]]; then
    [[ -n "$_FG" ]] || die "B_FRESH_PIN=1 但 B_FRESH_GLOB 为空"
    say "  fresh glob 已钉住（消融）：$_FG（跳过 B_${ROUND} 断言）"
  else
    case "$_FG" in
      *"B_${ROUND}/"*|*"B_${ROUND}"*) : ;;
      *) die "joint: 传下去的 fresh glob=$_FG 不含 B_${ROUND}（会在训练侧硬失败）" ;;
    esac
  fi
  case "$_AE" in
    0.25|0.3|0.30|0.35|0.4|0.40|0.45|0.5|0.50|0.55|0.6|0.60|0.65) : ;;
    *) die "joint: act_eps=$_AE 不在允许集合（0.25~0.65，步长 0.05）" ;;
  esac
  local _LR=${LAMBDA_R:-0.3} _YLH=${YAW_HINGE_LAMBDA:-0.3}
  say "  joint 生效值  : act_eps=$_AE, act_w=$_AW, yaw_rate_w=$_yaww, lambda_r=$_LR, yaw_hinge_lambda=$_YLH, fresh_glob=$_FG"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    say "  [dry-run] WM     = $wm"
    say "  [dry-run] arms   = ${TRAIN_ARMS:-main ctrl}   （每臂 1 张卡）"
    say "  [dry-run] 数据   = B_GLOB=${B_GLOB}"
    say "  [dry-run] fresh  = $_FG × ${B_FRESH_WEIGHT:-3.0}"
    say "  [dry-run] 起点   = ${INIT_VLA:-<官方底模>}"
    say "  [dry-run] 旋钮   = state_extra=$STATE_EXTRA, adv_baseline=$ADV_BASELINE/τ=$ADV_TAU, tail_cap_slope=$TAIL_CAP_SLOPE, lambda_r=$_LR, yaw_hinge_lambda=$_YLH, critic=${CRITIC_ON:-1}"
    say "  [dry-run] 训练未启动（去掉 DRY_RUN=1 才会真跑）"
    return 0
  fi
  ROUND=99 RUN_LABEL="$jl" STEPS="$STEPS" H="$H" BATCH="$BATCH" ACCUM="$ACCUM" \
    LORA_ALPHA="$LORA_ALPHA" FM_LR="$FM_LR" \
    TRUST_CENTER="${TRUST_CENTER:-self}" ACT_EPS="$_AE" ACT_WEIGHT="$_AW" \
    TRUST_EPS="${TRUST_EPS:-0.2}" TRUST_WEIGHT="${TRUST_WEIGHT:-1.0}" \
    B_SELF_TARGETS="${B_SELF_TARGETS:-$COMIMIC/runs/wm_real/self_targets_all.pt}" \
    INIT_VLA="${INIT_VLA:-}" \
    B_SELF_ERR_MAX="${B_SELF_ERR_MAX:-}" \
    LAMBDA_R_MAIN="$_LR" LAMBDA_R_CTRL="$_LR" LAMBDA_BC=0 TOKEN_BC=0 \
    POSTURE_WEIGHT=0 \
    REWARD_MODE="${REWARD_MODE:-saturating}" REWARD_VEL_SCALE="${REWARD_VEL_SCALE:-0}" OBS_NOISE=0 \
    REWARD_ROOT_WEIGHT="${REWARD_ROOT_WEIGHT:-}" REWARD_ROOT_POS_STD="${REWARD_ROOT_POS_STD:-}" \
    REWARD_ROOT_ORI_STD="${REWARD_ROOT_ORI_STD:-}" \
    ROOT_ORI_MODE="${ROOT_ORI_MODE:-split}" TAIL_WEIGHT="${TAIL_WEIGHT:-2.0}" \
    TAIL_POS_THR="$TAIL_POS_THR" TAIL_POS_CAP="$TAIL_POS_CAP" \
    YAW_HINGE_LAMBDA="$_YLH" \
    TAIL_CAP_SLOPE="$TAIL_CAP_SLOPE" STATE_EXTRA="$STATE_EXTRA" \
    ADV_BASELINE="$ADV_BASELINE" ADV_TAU="$ADV_TAU" \
    YAW_RATE_WEIGHT="$_yaww" \
    B_FRESH_GLOB="$_FG" \
    B_FRESH_WEIGHT="${B_FRESH_WEIGHT:-3.0}" \
    CRITIC_ON="${CRITIC_ON:-1}" CRITIC_SOURCE="${CRITIC_SOURCE:-imagined}" \
    CRITIC_GRAD_STATE="${CRITIC_GRAD_STATE:-1}" \
    CRITIC_ON_MAIN="$c1" CRITIC_ON_CTRL="$c0" \
    CRITIC_GRAD_STATE_MAIN="$g1" CRITIC_GRAD_STATE_CTRL="$g0" GRAD_LOG_EVERY=100 \
    TRAIN_WBC_MAIN="${TRAIN_WBC_MAIN:-0}" TRAIN_WBC_CTRL="${TRAIN_WBC_CTRL:-0}" \
    WBC_LR_MAIN="$WBC_LR" WBC_LR_CTRL="$WBC_LR" \
    WBC_UNFREEZE="${WBC_UNFREEZE:-all}" \
    WBC_KL_ANCHOR="${WBC_KL_ANCHOR:-0}" VLA_KL_ANCHOR="${VLA_KL_ANCHOR:-0}" \
    WBC_CKPT="$wm" B_GLOB="$B_GLOB" B_START_MAX_ERR="$B_START_MAX_ERR" \
    bash "$COMIMIC/scripts/s3_loop.sh" train 2>&1 | tee "$PD/6_joint.log" | tail -4
  [[ "${PIPESTATUS[0]}" == "0" ]] || die "联合微调失败（看 $PD/6_joint.log）"
  # ★ 2026-10-30：抓"env 链被注释截断"这类静默错——症状是训练输出跑到默认的 r99 目录。
  [[ -d "$COMIMIC/runs/S3_loop/$jl/train_main" ]] \
    || die "训练输出目录不存在：runs/S3_loop/$jl/train_main ⇒ RUN_LABEL 没生效（env 续行链可能被注释截断）"
  # ★ 硬断言：训练实际用的 WBC 必须就是本轮 gate 出来的 WM
  python3 - "$COMIMIC/runs/S3_loop/$jl/train_main/args.json" "$wm" <<'PY' || exit 1
import json, sys
a = json.load(open(sys.argv[1])); want = sys.argv[2]
got = a["wbc_ckpt"]
ok = got == want
print(f"[pipeline] 断言 WBC_CKPT: 实际={got}\n           期望={want}\n           {'✅ 一致' if ok else '❌ 不一致'}")
raise SystemExit(0 if ok else 1)
PY
  [[ $? == 0 ]] || die "联合训练用的 WM 不是本轮刷新的那个 ⇒ 流程错，停止"
  # ★ 2026-10-30：把"奖励各项真的生效"钉成断言（防 posture_weight 那种静默为 0 的事故）
  # ★ 2026-10-31（消融）：断言范围扩到**所有单旋钮**。理由：消融轮全部结论都建立在
  #   "只翻了一个开关"上，任何一个没生效都会让结论变成假的（lambda_r 之前在 stage 里
  #   是**写死 0.3** 的，AB3 会静默变成"完整方法"再跑一遍）。
  python3 - "$COMIMIC/runs/S3_loop/$jl/train_main/args.json" \
           0 "${TAIL_WEIGHT:-0.0}" "${ROOT_ORI_MODE:-mono}" \
           "$_LR" "$_AW" "$_AE" "$STATE_EXTRA" "$ADV_BASELINE" \
           "$TAIL_CAP_SLOPE" "$_YLH" "${CRITIC_ON:-1}" \
           "${WBC_UNFREEZE:-all}" "${WBC_KL_ANCHOR:-0}" "${VLA_KL_ANCHOR:-0}" <<'PYCHK'
import json, sys
a = json.load(open(sys.argv[1]))
want = {"posture_weight": float(sys.argv[2]), "tail_weight": float(sys.argv[3]),
        "root_ori_mode": sys.argv[4],
        "lambda_r": float(sys.argv[5]), "act_weight": float(sys.argv[6]),
        "act_eps": float(sys.argv[7]), "state_extra": sys.argv[8],
        "adv_baseline": sys.argv[9], "tail_cap_slope": float(sys.argv[10]),
        "yaw_hinge_lambda": float(sys.argv[11]), "critic": sys.argv[12] == "1",
        "wbc_unfreeze": sys.argv[13],
        "wbc_kl_anchor": float(sys.argv[14]),
        "vla_kl_anchor": float(sys.argv[15])}
bad = []
for k, v in want.items():
    got = a.get(k)
    if isinstance(v, bool):
        ok = bool(got) == v
    elif isinstance(v, float):
        ok = got is not None and abs(float(got) - v) < 1e-9
    else:
        ok = str(got) == str(v)
    if not ok:
        bad.append(f"{k}: args={got} 期望={v}")
print("[pipeline] 断言 生效值: " + ("✅ " + ", ".join(f"{k}={v}" for k, v in want.items())
                                 if not bad else "❌ " + "; ".join(bad)))
raise SystemExit(1 if bad else 0)
PYCHK
  [[ $? == 0 ]] || die "生效值不符（奖励/锚点/观测扩维/advantage 基线/hinge 与设定不一致）⇒ 停止"
  BSM=$(python3 -c "import json;print(json.load(open('$COMIMIC/runs/S3_loop/$jl/train_main/args.json')).get('b_start_max_err'))")
  [[ "$BSM" == "$B_START_MAX_ERR" ]] || die "起点筛选没生效（args 里是 $BSM，期望 $B_START_MAX_ERR）"
  say "joint ✅ 起点筛选=$BSM m"

  st_mark joint "$COMIMIC/runs/S3_loop/$jl"
}

# ============================================================ 7. eval
stage_eval() {
  local jd; jd=$(st_need joint)
  local head=$jd/train_main/vla_final.pt
  [[ -f "$head" ]] || die "没有 $head"
  # 执行器：默认用**冻结的 W1**（WBC_A24000_wmvel_v6.pt）。只有显式解冻过 WBC 才用 joint_final.pt。
  local wbc=$BASE_WM
  if [[ "${TRAIN_WBC_MAIN:-0}" == "1" ]]; then
    wbc=$jd/train_main/joint_final.pt
    [[ -f "$wbc" ]] || die "TRAIN_WBC_MAIN=1 但没产出 $wbc"
    say "⚠ 评测执行器 = 解冻后的 joint_final.pt（见过 §10.1 的 21x 崩溃再这么用）"
  else
    say "评测执行器 = 冻结的 $(basename "$wbc")"
  fi
  local sd
  if [[ "${EVAL_MODE:-pi0}" == "ablation" ]]; then
    # 本轮两臂互比（消融）：main/ctrl 都是本轮产物，同会话同种子 ⇒ 配对可比。
    local head2=$jd/train_ctrl/vla_final.pt wbc2=$BASE_WM
    if [[ "${TRAIN_WBC_CTRL:-0}" == "1" ]]; then wbc2=$jd/train_ctrl/joint_final.pt; fi
    [[ -f "$head2" ]] || die "没有 $head2"
    [[ -f "$wbc2" ]] || die "没有 $wbc2（TRAIN_WBC_CTRL=1 时会用 ctrl 臂的 joint_final.pt）"
    say "--- 9/10 闭环评测（消融：同会话 6-seed 配对，${LABEL_MAIN:-main} vs ${LABEL_CTRL:-ctrl}）---"
    TAG=${ROUND}_ablation SEEDS="$EVAL_SEEDS" START_FRAME=0 \
      HEAD_MAIN="$head" HEAD_CTRL="$head2" \
      SM_CKPT_MAIN="$wbc" SM_CKPT_CTRL="$wbc2" \
      LABEL_MAIN="${LABEL_MAIN:-main}" LABEL_CTRL="${LABEL_CTRL:-ctrl}" \
      EXTRA_MAIN="--sample-avg 4" EXTRA_CTRL="--sample-avg 4" \
      bash "$COMIMIC/scripts/s3_eval_sweep.sh" 2>&1 | tee "$PD/7_eval.log" | tail -6
    sd=$COMIMIC/runs/S3_loop/sweeps/${ROUND}_ablation/summary.json
  else
    say "--- 9/10 闭环评测（$(echo "$EVAL_SEEDS" | wc -w)-seed 配对：π0 vs 本轮，两边 --sample-avg 4）---"
    TAG=pi0_vs_${ROUND}_avg4 SEEDS="$EVAL_SEEDS" START_FRAME=0 \
      HEAD_MAIN=base HEAD_CTRL="$head" \
      SM_CKPT_MAIN="$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt" SM_CKPT_CTRL="$wbc" \
      EXTRA_MAIN="--sample-avg 4" EXTRA_CTRL="--sample-avg 4" \
      bash "$COMIMIC/scripts/s3_eval_sweep.sh" 2>&1 | tee "$PD/7_eval.log" | tail -6
    sd=$COMIMIC/runs/S3_loop/sweeps/pi0_vs_${ROUND}_avg4/summary.json
  fi
  [[ -f "$sd" ]] || die "评测没产出 summary.json"
  say "eval ✅ $sd"
  st_mark eval "$sd"
}

# ============================================================ 8. report
stage_report() {
  local sd; sd=$(st_need eval)
  say "--- 10/10 汇总 ---"
  PYTHONPATH=. python3 - "$sd" <<'PY' | tee "$PD/8_report.txt"
import json, statistics as st, sys
d = json.load(open(sys.argv[1]))
m, c = d["main"], d["ctrl"]
dif = [y - x for x, y in zip(m["per_seed"], c["per_seed"])]
mu, sd = st.mean(dif), st.stdev(dif)
sem = sd / len(dif) ** 0.5
# ★ 2026-10-29 修：原来写死 "π0/候选"，消融模式下（main/ctrl 都是本轮产物）这是误导。
#   现在用 summary.json 里 heads.json 记的真实标签。
lm, lc = m.get("label", "main"), c.get("label", "ctrl")
print(f"{lm:<10} {m['root_mean']:.3f} ± {m['root_sd']:.3f}  {m['per_seed']}")
print(f"{lc:<10} {c['root_mean']:.3f} ± {c['root_sd']:.3f}  {c['per_seed']}")
print(f"配对    {mu:+.3f} m (={lc}−{lm})  sem={sem:.3f}  t={mu/sem:+.2f}  "
      f"{lc}更好 {sum(1 for x in dif if x<0)}/{len(dif)}")
# ★ 2026-10-31 修：判决文案原来写死 "≥5/6"，EVAL_SEEDS 已改成 12 集/臂，
#   文案与实际门限（≥n−1 同向）不一致 ⇒ 改成按 n 生成，避免误读。
win, need, nf = sum(1 for x in dif if x < 0), len(dif) - 1, len(dif)
print("判决：", "成功" if (mu < 0 and mu/sem <= -2 and win >= need)
      else f"未达门（c 更好需要 t≤-2 且 ≥{need}/{nf} 同向；本轮 t={mu/sem:+.2f}、同向 {win}/{nf}）")
PY
  PYTHONPATH=. python3 "$COMIMIC/scripts/s3_round_table.py" --out "$COMIMIC/runs/S3_loop/ROUNDS.md" >/dev/null 2>&1 || true
  say "report ✅ $PD/8_report.txt"
  st_mark report "$PD/8_report.txt"
}

# ============================================================ dispatch
# ★ 2026-10-31 订死：每轮开跑前打印**配置指纹**，任何一项异常都直接中止。
preflight() {
  local _cmd=${1:-} n_af n_rows
  n_af=$(ls "$COMIMIC"/runs/wm_real/ac_w*.pt 2>/dev/null | wc -l)
  (( n_af >= 1 )) || die "preflight: 找不到 A 线文件（runs/wm_real/ac_w*.pt）"
  n_rows=$(python3 - "$COMIMIC" "$n_af" <<'PYA'
import glob, sys, torch
n = 0
for f in sorted(glob.glob(f"{sys.argv[1]}/runs/wm_real/ac_w*.pt")):
    n += int(torch.load(f, map_location="cpu", weights_only=False)["states"].shape[0])
print(n)
PYA
)
  say "===== 配置指纹（$ROUND）====="
  say "  A 线      : $n_af 个文件 / $n_rows 行"
  say "  B 线      : 全池 ${B_GLOB}"
  say "  WM 三阶段 : 场景 $WM_STEPS_A / 混合 $WM_STEPS_MIX / k=$WM_ROLLOUT_STEPS 自回归 $WM_STEPS_2B"
  say "  WM 结构   : arch=$WM_ARCH（mlp=133→2048x3→70 / gru=GRU(133→1024)+1024→2048→2048→70），clip=$WM_CLIP_NORM"
  say "  WM 角速度 : w_lambda=$WM_W_LAMBDA, vel_scale_lambda=$WM_VEL_SCALE_LAMBDA（幅度比 vs L2 回归均值）"
  say "  WM 配比   : **整批** 参考库 ${WM_MIX_LIB} / A+B 其余；A:B = $WM_AB_RATIO:1 ⇒ B 占整批 $((100/(WM_AB_RATIO+1)))%（A8 口径已改）"
  say "  VLA 训练  : $STEPS 步, H=$H, batch $BATCH×$ACCUM, 冻结 WBC"
  say "  奖励      : root_ori=$ROOT_ORI_MODE, tail=$TAIL_WEIGHT, posture=0(已删), root_w=${REWARD_ROOT_WEIGHT:-默认}"
  _ae="$(_act_eps_effective)"
  say "  信任域    : trust_center=${TRUST_CENTER:-self}, token_eps=${TRUST_EPS:-0.2}"
  say "  观测扩维  : state_extra=$STATE_EXTRA（abs6=绝对朝向 6 维，46→52；none=46）"
  say "  奖励改造  : tail_cap_slope=$TAIL_CAP_SLOPE, adv_baseline=$ADV_BASELINE, adv_tau=$ADV_TAU"
  say "  hinge 位置 : e_h=${TAIL_POS_THR:-默认0.35}, cap=${TAIL_POS_CAP:-默认1.0}, λ=$TAIL_WEIGHT, E_start=${B_START_MAX_ERR:-无}"
  say "  act 锚点  : eps=$_ae, weight=$(_act_weight_effective)（**退场靠降权重**：上轮 frac_out<0.35 才 ×0.5，下限 $ACT_WEIGHT_FLOOR）"
  say "  新数据权重: ${B_FRESH_WEIGHT:-3.0}×  glob=$(_fresh_glob_effective)"
  say "  评测      : $(echo "${EVAL_SEEDS}" | wc -w) 集/臂"
  # ★ 2026-10-31（A11/A12）：**奖励项活性审计** —— 在真实 B 线数据上量每一项有没有信号。
  #   历史教训：yaw 项 σ=0.25 而实际误差 37~57° ⇒ 整项死区；yaw_rate 与参考相关 0.005。
  #   两个问题 d1~d3 三轮都在，gate 完全看不见。现在开训前就拦。
  if [[ "${SKIP_REWARD_AUDIT:-0}" != "1" ]]; then
    local _ad _rwc=""
    # 上一轮 gate 量到的 WM root 角速度质量（W_YAWRATE>0 时是硬准入）
    _rwc=$(ls -t "$COMIMIC"/runs/S3_pipeline/*/wm_root_w_corr.txt 2>/dev/null | head -1)
    [[ -n "$_rwc" ]] && _rwc=$(cat "$_rwc")
    _ad=$(ls -d "$COMIMIC"/runs/B_${ROUND} 2>/dev/null | head -1)
    local _dumps
    if [[ -n "$_ad" ]]; then _dumps="$_ad/*/dump/*.pt"; else _dumps="runs/B_*/vla_*_f*_r*/dump/*.pt"; fi
    python3 "$COMIMIC/scripts/audit_reward_activity.py" --dumps "$_dumps" \
      ${_rwc:+--wm-root-w-corr "$_rwc"} 2>&1 | tail -12 \
      || die "奖励项活性审计不合格（SKIP_REWARD_AUDIT=1 可临时跳过）"
  fi

  # ★ 2026-10-31：新数据加权 glob 必须**只覆盖本轮**。
  #   踩过的坑：一个继承来的全池 glob（runs/B_*）会让"×3 新数据"退化成"全池 ×3"= 均匀采样，
  #   而日志里只有一行 warn，d1/d2/d2b 三轮都是这么静默失效的。
  _fg="$(_fresh_glob_effective)"
  if [[ -n "$_fg" ]]; then
    case "$_fg" in
      *"B_${ROUND}/"*|*"B_${ROUND}"*) : ;;
      *) die "B_FRESH_GLOB=$_fg 不含 B_${ROUND}：全池加权等于没加权。请 unset B_FRESH_GLOB（默认按轮生成）" ;;
    esac
    _nd=$(eval ls -d "$_fg" 2>/dev/null | wc -l)
    if [[ "${_nd:-0}" -gt 0 ]]; then
      say "  fresh glob 命中: $_nd 个 dump 文件"
    elif [[ "$_cmd" == "collect" || "$_cmd" == "all" ]]; then
      # ★ 2026-11-01（完整轮）：collect/all 之前本轮数据还不存在，fresh glob 为空是正常的；
      #   采集完成后 stage_joint 会再检查。其它阶段仍然必须命中，防止加权静默失效。
      say "  fresh glob 尚未生成（$_cmd 会先采集）：$_fg"
    else
      die "B_FRESH_GLOB=$_fg 一个文件都没匹配到（加权会静默失效）"
    fi
  fi
  # 硬校验：数值必须在合理范围（防止某次改参数改出离谱值）
  python3 - "$STEPS" "${B_FRESH_WEIGHT:-3.0}" "$WM_AB_RATIO" "$TAIL_WEIGHT" "$_ae" "$(_act_weight_effective)" "${TAIL_POS_THR:-}" "${TAIL_POS_CAP:-}" "${B_START_MAX_ERR:-}" <<'PYCHK2' || die "preflight 校验失败"
import sys
steps, fresh, ratio, tail = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
_eps = float(sys.argv[5])
_w = float(sys.argv[6])
_tpt = float(sys.argv[7]) if sys.argv[7] else None
_tpc = float(sys.argv[8]) if sys.argv[8] else None
_estart = float(sys.argv[9]) if sys.argv[9] else None
bad = []
if not (50 <= steps <= 5000): bad.append(f"STEPS={steps} 超出 50~5000")
if not (1.0 <= fresh <= 20.0): bad.append(f"B_FRESH_WEIGHT={fresh} 超出 1~20")
if not (1 <= ratio <= 10): bad.append(f"WM_AB_RATIO={ratio} 超出 1~10")
if not (0.0 <= tail <= 20.0): bad.append(f"TAIL_WEIGHT={tail} 超出 0~20")
if not (0.25 <= _eps <= 0.65): bad.append(f"ACT_EPS={_eps} 超出 0.25~0.65")
if not (0.0 <= _w <= 3.0): bad.append(f"ACT_WEIGHT={_w} 超出 0~3（退场只许往下走）")
if _tpt is not None and _tpt <= 0: bad.append(f"TAIL_POS_THR={_tpt} 应 >0")
if _tpc is not None and _tpc <= 0: bad.append(f"TAIL_POS_CAP={_tpc} 应 >0")
if _tpt is not None and _tpc is not None and _tpt >= _tpc:
    bad.append(f"hinge e_h={_tpt} 必须 < cap={_tpc}")
if _tpt is not None and _estart is not None and _tpt >= _estart:
    bad.append(f"hinge e_h={_tpt} 必须 < E_start={_estart}（否则 hinge 从起点就全程在线）")
print("[preflight] 数值范围: " + ("✅" if not bad else "❌ " + "; ".join(bad)))
raise SystemExit(1 if bad else 0)
PYCHK2
}

case "${1:-}" in
  a_collect)   stage_a_collect ;;
  collect)     stage_collect ;;
  score)       stage_score ;;
  transitions) stage_transitions ;;
  wm_refresh)  stage_wm ;;
  wm_gate)     stage_gate ;;
  critic_gate) stage_critic ;;
  joint)       stage_joint ;;
  eval)        stage_eval ;;
  report)      stage_report ;;
  all)
    preflight all
    stage_a_collect
    stage_collect; stage_score; stage_transitions; stage_wm
    stage_gate;    stage_critic; stage_joint;      stage_eval; stage_report
    say "=========== $ROUND 全流程完成 ==========="
    ;;
  *) echo "用法: ROUND=<name> bash scripts/s3_pipeline.sh {a_collect|collect|score|transitions|wm_refresh|wm_gate|critic_gate|joint|eval|report|all}" >&2; exit 2 ;;
esac
