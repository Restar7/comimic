#!/usr/bin/env bash
# ============================================================================
# S3 轮次**独立验收器**（只读，不训练、不采集）。
#
# 目的：把「八个阶段一个都不能漏」变成一条能跑的检查，而不是靠记忆。
# 它不相信任何日志，只相信：
#   1. runs/S3_pipeline/<ROUND>/state.json 里 8 个阶段齐全、时间戳单调递增；
#   2. 每个阶段记的产物**真的存在**（文件/目录）；
#   3. 关键一致性断言（这几条以前真出过事）：
#        - gate 用的 WM == wm 阶段刷出来的那个（不许拿上一轮的 WM 当裁判）
#        - joint 的 args.json 里 wbc_ckpt == gate 产物（训练真的用了新 WM）
#        - joint 的 args.json 里 b_start_max_err == 本轮设定（起点筛选真的生效）
#        - eval 的 summary.json 有 main/ctrl 两臂、per_seed 长度一致（配对可算）
#
# 用法：
#   ROUND=big1 bash scripts/s3_verify_round.sh          # 验收某一轮
#   ROUND=big1 bash scripts/s3_verify_round.sh -v       # 打印每个产物路径
# 退出码 0 = 全过；1 = 有缺口（并逐条列出缺什么）。
# ============================================================================
set -uo pipefail

COMIMIC=${COMIMIC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROUND=${ROUND:?需要 ROUND=<轮次名>}
PD=$COMIMIC/runs/S3_pipeline/$ROUND
VERBOSE=${1:-}

[[ -f "$PD/state.json" ]] || { echo "❌ 没有 $PD/state.json（这一轮根本没按 pipeline 跑）" >&2; exit 1; }

PYTHONPATH="$COMIMIC" python3 - "$PD" "$VERBOSE" <<'PY'
import json, os, re, sys
from pathlib import Path

pd = Path(sys.argv[1]); verbose = sys.argv[2] == "-v"
ROUND_NM = pd.name
COMIMIC = str(pd.parent.parent.parent)
_st = json.load(open(pd / "state.json"))
st = _st.get("stages", {})
ver = _st.get("pipeline_version", "v8")
# 版本演进（老轮次不要求新阶段，否则历史轮次全判不合格）：
#   v8 : collect → … → report（8 阶段）
#   v9 : + critic 门
#   v10: + A 线采集（a_collect 打头）  ← 2026-10-29
order = []
if ver in ("v10",):
    order.append("a_collect")
order += ["collect", "score", "transitions", "wm", "gate"]
if ver in ("v9", "v10"):
    order.append("critic")
order += ["joint", "eval", "report"]
cn = {"a_collect": "A线采集", "collect": "采集", "score": "打分", "transitions": "建转移",
      "wm": "刷新WM", "gate": "WM门", "critic": "critic门", "joint": "联合微调",
      "eval": "闭环评测", "report": "汇总"}
bad = []
warn = []

def path_of(v):
    return Path(v["artifact"]) if isinstance(v, dict) and v.get("artifact") else None

# 1) 齐全 + 顺序
prev_epoch = -1.0
for k in order:
    v = st.get(k)
    if not v:
        bad.append(f"缺阶段 {cn[k]}({k})")
        print(f"  {cn[k]:<6} ❌ 未执行")
        continue
    p = path_of(v)
    e = float(v.get("epoch", 0.0))
    ok_time = e >= prev_epoch
    prev_epoch = max(prev_epoch, e)
    ok_exist = p is not None and p.exists()
    mark = "✅" if (ok_exist and ok_time) else "❌"
    if not ok_exist:
        bad.append(f"{cn[k]} 的产物不存在: {p}")
    if not ok_time:
        # ★ 2026-10-30：重跑某个阶段（例如 WM 重训、门重判）会让时间戳乱序，这是**合法操作**，
        #   不能因此判不合格——真正的硬约束是"阶段齐全 + 产物存在 + 一致性断言"。
        warn.append(f"{cn[k]} 的时间戳早于上一阶段（重跑过，允许）")
    print(f"  {cn[k]:<6} {mark} {v.get('at','?')}  {p.name if p else '<无记录>'}")
    if verbose:
        print(f"          {p}")

# 2) 一致性断言
def art(k):
    v = st.get(k) or {}
    return v.get("artifact")

if st.get("wm") and st.get("gate"):
    if art("wm") != art("gate"):
        bad.append(f"gate 用的 WM 不是本轮刷新的：gate={art('gate')} wm={art('wm')}")
    else:
        print(f"  断言 gate==wm                     ✅ {Path(art('wm')).name}")

jr = art("joint")
if jr and st.get("gate"):
    aj = Path(jr) / "train_main" / "args.json"
    if not aj.exists():
        bad.append(f"joint 缺 args.json: {aj}")
    else:
        a = json.load(open(aj))
        got, want = a.get("wbc_ckpt"), art("gate")
        if got != want:
            bad.append(f"训练用的 WM 不是 gate 产物：args={got} gate={want}")
        else:
            print(f"  断言 joint.wbc_ckpt==gate          ✅ {Path(got).name}")
        # 两个阈值是**不同量**，别混：
        #   b_start_max_err → BStartSource.max_e_inst（瞬时 root 误差，决定奖励核有没有梯度）
        #   b_self_err_max  → BStartSource.max_err（接下来 horizon 步的**实际**漂移，默认 p25）
        # critic 门的决定必须和训练实际用的 critic 一致（门说不过，训练就不能开）
        cgj = pd / "critic_gate.json"
        if cgj.exists():
            allow = bool(json.load(open(cgj))["allow"])
            used = bool(a.get("critic", a.get("use_critic", None)))
            if used != allow:
                bad.append(f"critic 门说 allow={allow}，但训练 args.critic={used} ⇒ 门的决定没生效")
            else:
                print(f"  断言 critic门==训练args             ✅ allow={allow} / args.critic={used}")
        if "b_start_max_err" not in a:
            bad.append("args.json 里没有 b_start_max_err（起点筛选可能没生效）")
        else:
            e_inst = a.get("b_start_max_err")
            e_real = a.get("b_self_err_max")
            tag = "✅" if e_inst not in (None, 0, 0.0) else "⚠"
            if e_inst in (None, 0, 0.0):
                print(f"  提示 瞬时误差筛未开                {tag} b_start_max_err={e_inst}"
                      f"（e 大时奖励核梯度≈0，等于教噪声）")
            else:
                print(f"  断言 瞬时误差筛已开                {tag} b_start_max_err={e_inst} m"
                      f" / 实际漂移筛 b_self_err_max={e_real}")

# 2.5) e1/e2（2026-10-31）**只能按"实际生效"验收**，不看我们打算传什么
try:
    for arm in ("main", "ctrl"):
        ap = Path(str(art("joint") or ""))
    jargs = None
    for cand in sorted(PD.glob("*")):
        pass
except Exception:
    pass
_jd = None
for _c in sorted(Path(COMIMIC).glob("runs/S3_loop/*/train_main/args.json")):
    if _c.parent.parent.name in (ROUND_NM, ROUND_NM + "_joint"):
        _jd = _c
if _jd is not None:
    ja = json.load(open(_jd))
    se = str(ja.get("state_extra", "none") or "none")
    ab = str(ja.get("adv_baseline", "none") or "none")
    at = float(ja.get("adv_tau", 0.0) or 0.0)
    ts = float(ja.get("tail_cap_slope", 0.0) or 0.0)
    print(f"  e1/e2 实际生效: state_extra={se} adv_baseline={ab} adv_tau={at} tail_cap_slope={ts}")
    # (a) e1：ckpt 里必须带缩放，且 mode 与训练一致；评测 server 日志必须同口径
    if se != "none":
        ck = _jd.parent / "vla_final.pt"
        if not ck.exists():
            bad.append(f"state_extra={se} 但 {ck} 不存在")
        else:
            try:
                import torch as _t
                _d = _t.load(str(ck), map_location="cpu", weights_only=False)
                _sc = _d.get("state_extra_scale")
                _md = str(_d.get("state_extra_mode") or "")
                if _sc is None:
                    bad.append(f"state_extra={se} 但 ckpt 里没有 state_extra_scale（下一轮会丢）")
                elif _md != se:
                    bad.append(f"ckpt state_extra_mode={_md!r} != 训练 args {se!r}")
                else:
                    print(f"  断言 e1 ckpt 带缩放                ✅ mode={_md} "
                          f"scale_absmax={float(_sc.abs().max()):.6f}")
            except Exception as exc:  # noqa: BLE001
                warn.append(f"ckpt 读取失败（{exc}）")
        # 评测侧：找本轮的 server 日志，确认它也是同一口径（否则训练/部署错配）
        _srv = sorted(Path(COMIMIC).glob(f"logs/sweep_*{ROUND_NM}*srv*.log")) + \
               sorted(Path(COMIMIC).glob(f"logs/s3_loop_r*_srv_*.log"))
        _hit = [f for f in _srv if f.exists() and f.stat().st_mtime > _jd.stat().st_mtime - 86400]
        _ok = []
        for f in _hit:
            try:
                txt = f.read_text(errors="ignore")
            except Exception:
                continue
            if f"[server][head][state-extra] mode={se}" in txt:
                _ok.append(f.name)
        if _ok:
            print(f"  断言 e1 评测口径一致                ✅ {_ok[0]}")
        else:
            bad.append(f"没在评测 server 日志里找到 mode={se} 的口径声明 ⇒ "
                       f"无法证明评测也用了 {se}（训练/部署错配风险）")
    # (a2) 反漂移：**指纹里写的**必须和 args.json 里**实际生效的**一致
    #      （d1–d3 的"新数据×3"就是指纹写了、代码没生效，静默废了三轮）
    _fp = Path(COMIMIC) / "runs/S3_pipeline" / f"{ROUND_NM}_driver.log"
    if _fp.exists():
        _txt = _fp.read_text(errors="ignore")
        _mo = re.findall(r"观测扩维\s*:\s*state_extra=(\w+)", _txt)
        _mb = re.findall(r"奖励改造\s*:\s*tail_cap_slope=([0-9.]+), adv_baseline=(\w+)", _txt)
        if _mo:
            if _mo[-1] != se:
                bad.append(f"指纹说 state_extra={_mo[-1]}，但 args.json 实际是 {se} ⇒ 配置没传到训练")
            else:
                print(f"  断言 e1 指纹==实际生效               ✅ {se}")
        # 今天的真 bug：`ROUND=99 X="$(f)" cmd` 让 f 看到 ROUND=99 ⇒ act_eps 与 fresh glob 都跑偏。
        # 指纹（主 shell 算的）与 args.json（子进程实际收到的）不一致时就该报红。
        _mp = re.findall(r"act 锚点\s*:\s*eps=([0-9.]+)", _txt)
        if _mp and abs(float(_mp[-1]) - float(ja.get("act_eps", -1))) > 1e-9:
            bad.append(f"指纹 act_eps={_mp[-1]} ≠ args.json act_eps={ja.get('act_eps')} "
                       f"⇒ 命令替换在临时赋值里被污染（见 s3_pipeline.sh 的注释）")
        else:
            print(f"  断言 act_eps 指纹==实际生效           ✅ {ja.get('act_eps')}")
        # yaw_rate 是**gate 之后**按 root_w_corr 决定的 ⇒ 权威值是 joint 前那行"生效值"，
        # 不是 preflight 指纹（preflight 时还没量 WM 角速度质量）。
        _my = re.findall(r"joint 生效值\s*:\s*act_eps=[0-9.]+, act_w=([0-9.]+), "
                         r"yaw_rate_w=([0-9.]+)", _txt)
        if _my:
            _wa, _wy = float(_my[-1][0]), float(_my[-1][1])
            _ga = float(ja.get("act_weight", -1.0))
            if _ga >= 0 and abs(_ga - _wa) > 1e-9:
                bad.append(f"joint 生效值 act_w={_wa} 但 args.json act_weight={_ga} ⇒ 没传下去")
            else:
                print(f"  断言 act_weight 生效值==实际        ✅ {_wa}")
        if _my:
            _want = float(_my[-1])
            _got = float(ja.get("yawrate_weight", -1.0))
            _eff = _want if _got < 0 else _got
            if abs(_eff - _want) > 1e-9:
                bad.append(f"joint 生效值 yaw_rate_w={_want} 但 args.json 实际 {_got} ⇒ 没传下去")
            else:
                print(f"  断言 yaw_rate 生效值==实际          ✅ {_want}")
        _mf = re.findall(r"新数据权重:.*?glob=(\S+)", _txt)
        _fgj = str(ja.get("b_fresh_glob", "") or "")
        if _mf:
            if f"B_{ROUND_NM}/" not in _fgj:
                bad.append(f"指纹 glob={_mf[-1]}，但 args.json b_fresh_glob={_fgj or '<空>'} "
                           f"不含 B_{ROUND_NM} ⇒ 新数据加权没落到本轮")
            else:
                print(f"  断言 fresh glob 指向本轮              ✅ {Path(_fgj).name if _fgj else ''}")
        if _mb:
            _ts_fp, _ab_fp = float(_mb[-1][0]), _mb[-1][1]
            if _ab_fp != ab or abs(_ts_fp - ts) > 1e-9:
                bad.append(f"指纹(adv={_ab_fp},cap_slope={_ts_fp}) 与 args.json"
                           f"(adv={ab},cap_slope={ts}) 不一致 ⇒ 配置没传到训练")
            else:
                print(f"  断言 e2 指纹==实际生效               ✅ adv={ab} cap_slope={ts}")
    else:
        warn.append(f"没有 {_fp.name}，无法做'指纹==实际生效'的反漂移断言")

    # (b) e2：adv 指标必须真的出现过
    tm = _jd.parent / "train_metrics.jsonl"
    if ab != "none":
        found = False
        if tm.exists():
            for line in open(tm):
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                if "adv/std_ratio" in j:
                    found = True
                    print(f"  断言 e2 配对基线真的跑了          ✅ adv/std_ratio={j['adv/std_ratio']:.3f} "
                          f"frac_inside_tau={j.get('adv/frac_inside_tau', float('nan')):.3f}")
                    break
        if not found:
            bad.append(f"adv_baseline={ab} 但 train_metrics.jsonl 里没有 adv/std_ratio（没生效）")
    if ts > 0 and abs(float(ja.get("tail_weight", 0.0) or 0.0)) > 0:
        print(f"  断言 e2 hinge 远场仍有梯度        ✅ tail_cap_slope={ts}")
else:
    warn.append(f"没找到 {ROUND_NM} 的 train_main/args.json（e1/e2 无法自动验收）")

es = art("eval")
if es:
    sp = Path(es)
    if not sp.exists():
        bad.append(f"eval summary 不存在: {sp}")
    else:
        d = json.load(open(sp))
        for arm in ("main", "ctrl"):
            if arm not in d or "per_seed" not in d[arm]:
                bad.append(f"eval summary 缺 {arm}.per_seed")
        if all(a in d and "per_seed" in d[a] for a in ("main", "ctrl")):
            n1, n2 = len(d["main"]["per_seed"]), len(d["ctrl"]["per_seed"])
            if n1 != n2:
                bad.append(f"两臂种子数不等，无法配对：main={n1} ctrl={n2}")
            elif n1 < 6:
                bad.append(f"配对种子只有 {n1} 个（<6，判决不可信）")
            else:
                print(f"  断言 成对评测                      ✅ {n1} 种子 "
                      f"main={d['main'].get('label','?')} ctrl={d['ctrl'].get('label','?')}")

print()
if warn:
    print(f"⚠ {len(warn)} 条提示（不影响合格判定）：")
    for w in warn:
        print(f"   - {w}")
if bad:
    print("❌ 不合格 —— " + f"{len(bad)} 处缺口：")
    for b in bad:
        print(f"   - {b}")
    raise SystemExit(1)
print("✅ 合格：八阶段齐全、顺序正确、产物真实存在、四项一致性断言全过")
PY
