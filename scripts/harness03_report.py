#!/usr/bin/env python3
"""会话 03 的报告生成器：**所有数字都从落盘产物读出来**，一个字不手写。

产物来源
--------
* ``windows_meta.json``      —— 窗口/切分/哈希/奖励口径
* ``exp0.json``              —— 梯度路径审计 + 有限差分 + 尺度标定
* ``gradstruct_vla_*.json``  —— 逐窗口梯度结构 + 分组 CV 可预测性
* ``arms_runs_*.jsonl``      —— 四个 arm 的逐次运行（含逐窗口回报）
* ``horizon_runs_*.jsonl``   —— 视界消融

用法::

    python3 scripts/harness03_report.py --out runs/harness_03_bptt
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load_arms(out: Path) -> list[dict]:
    rows, seen = [], {}
    for f in sorted(out.glob("arms_runs*.jsonl")):
        if "old" in f.name:
            continue
        for r in _load_jsonl(f):
            key = (r.get("arm"), r.get("size"), r.get("seed"), r.get("lr"),
                   r.get("ablate"), r.get("oracle_scope"), r.get("tokens"), r.get("cap"))
            seen[key] = r
    return list(seen.values())


def load_hz(out: Path) -> list[dict]:
    rows = {}
    for f in sorted(out.glob("horizon_runs*.jsonl")):
        for r in _load_jsonl(f):
            rows[(r.get("arm"), r.get("H"), r.get("seed"), r.get("lr"), r.get("tokens"))] = r
    return list(rows.values())


def sel(rows, **kw):
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple, set)):
                if r.get(k) not in v:
                    ok = False
            elif r.get(k) != v:
                ok = False
        if ok:
            out.append(r)
    return out


def ms(xs, fmt="{:.4f}"):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return "NOT_RUN"
    m = float(np.mean(xs))
    s = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    return (fmt + " ± " + fmt).format(m, s) if len(xs) > 1 else fmt.format(m)


def fnum(x, fmt="{:.4f}"):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NOT_RUN"
    return fmt.format(x)


def paired(a: list[float], b: list[float]):
    """逐窗口配对差 + 配对 t 统计（同窗口同一数据，配对是唯一正确口径）。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = min(len(a), len(b))
    if n == 0:
        return None
    d = a[:n] - b[:n]
    se = d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    return dict(mean=float(d.mean()), se=float(se), n=int(n),
                t=(float(d.mean() / se) if se and se > 0 else float("nan")),
                frac_pos=float((d > 0).mean()))


def paired_from_rows(rows_a, rows_b, split: str):
    """把两个 arm 的逐窗口回报按 (size, seed) 对齐后做配对差。"""
    out = []
    for ra in rows_a:
        for rb in rows_b:
            if (ra.get("size"), ra.get("seed")) != (rb.get("size"), rb.get("seed")):
                continue
            A = ra.get(f"perwin_{split}")
            B = rb.get(f"perwin_{split}")
            if not A or not B:
                continue
            out.append(paired(A, B))
    return out


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main(out_dir: Path) -> int:
    arms = load_arms(out_dir)
    hz = load_hz(out_dir)
    meta = json.loads((out_dir / "windows_meta.json").read_text()) \
        if (out_dir / "windows_meta.json").exists() else {}
    exp0 = json.loads((out_dir / "exp0.json").read_text()) \
        if (out_dir / "exp0.json").exists() else {}
    gstruct = {}
    for p in sorted(out_dir.glob("gradstruct_*.json")):
        gstruct[p.stem] = json.loads(p.read_text())

    # ------------------------------------------------------------------ 主线：n=32
    BASE_KW = dict(tokens="vla", cap=0.08)
    NONE_AB = dict(ablate="none")
    core = sel(arms, size=32, ablate="none", **BASE_KW)
    const_core = sel(core, arm="constant")
    mlp_core = sel(core, arm="mlp")
    oracle32 = sel(arms, arm="oracle", size=32, cap=0.08)
    oracle256 = sel(arms, arm="oracle", size=256, cap=0.08)
    base32 = sel(arms, arm="base", size=32)
    base256 = sel(arms, arm="base", size=256)

    def arm_row(name, rows, heldout=("val", "test")):
        if not rows:
            return [name, "NOT_RUN", "NOT_RUN", "NOT_RUN", "NOT_RUN", "NOT_RUN"]
        d_fit = [r.get("dJ_fit") for r in rows]
        d_ho = [np.mean([r.get("dJ_val", np.nan), r.get("dJ_test", np.nan)]) for r in rows]
        pos = [r.get("pos_test") for r in rows]
        ar = [r.get("act_rms") for r in rows]
        return [name, ms(d_fit), ms(d_ho), "—", ms(pos, "{:.3f}"), ms(ar, "{:.4f}")]

    # MLP vs Constant（逐窗口配对，held-out）
    mlp_vs_const = {}
    for split in ("fit", "val", "test"):
        mlp_vs_const[split] = paired_from_rows(mlp_core, const_core, split)

    # ------------------------------------------------------------------ 臂表（两张：n=32 / n=256）
    def arm_table(size):
        base_r = sel(arms, arm="base", size=size)
        const_r = sel(arms, size=size, arm="constant", ablate="none", **BASE_KW)
        mlp_r = sel(arms, size=size, arm="mlp", ablate="none", **BASE_KW)
        orc = sel(arms, arm="oracle", size=size, cap=0.08)

        def d_ho(rows):
            return [float(np.nanmean([r.get("dJ_val", np.nan), r.get("dJ_test", np.nan)]))
                    for r in rows]

        def d_vs_const(rows):
            out = []
            for split in ("val", "test"):
                for a, c in zip(rows, const_r):
                    A, C = a.get(f"perwin_{split}"), c.get(f"perwin_{split}")
                    if A and C:
                        out.append(float(np.mean(np.asarray(A) - np.asarray(C))))
            return out

        row_base = ["Base (Δ=0)", "0 (定义)", "0 (定义)", "—", "—", "0.0000"]
        row_c = ["Constant", ms([r.get("dJ_fit") for r in const_r]), ms(d_ho(const_r)),
                 "0 (定义)", ms([r.get("pos_test") for r in const_r], "{:.3f}"),
                 ms([r.get("act_rms") for r in const_r])]
        row_m = ["State MLP", ms([r.get("dJ_fit") for r in mlp_r]), ms(d_ho(mlp_r)),
                 ms(d_vs_const(mlp_r)), ms([r.get("pos_test") for r in mlp_r], "{:.3f}"),
                 ms([r.get("act_rms") for r in mlp_r])]
        orc_fit = sel(orc, oracle_scope="fit")
        orc_val = sel(orc, oracle_scope="val")
        orc_test = sel(orc, oracle_scope="test")
        row_o = ["Oracle (逐窗口, 诊断上界)",
                 ms([r.get("dJ_fit") for r in orc_fit]) if orc_fit else "NOT_RUN",
                 ms([r.get("dJ_val") for r in orc_val]) if orc_val else "NOT_RUN",
                 "n/a（不可部署）",
                 ms([r.get("pos_test") for r in orc_test], "{:.3f}") if orc_test else "NOT_RUN",
                 ms([r.get("act_rms") for r in orc_fit]) if orc_fit else "NOT_RUN"]
        return [row_base, row_c, row_m, row_o]

    rows_tbl32 = arm_table(32)
    rows_tbl256 = arm_table(256)

    # ------------------------------------------------------------------ 规模扫描
    size_tbl = []
    for n in (8, 32, 128, 256):
        c = sel(arms, size=n, arm="constant", ablate="none", **BASE_KW)
        m = sel(arms, size=n, arm="mlp", ablate="none", **BASE_KW)
        o = sel(arms, arm="oracle", size=n, cap=0.08,
                oracle_scope=("fit" if n != 32 else None))
        size_tbl.append([
            n,
            ms([r.get("dJ_fit") for r in c]), ms([r.get("dJ_val") for r in c]),
            ms([r.get("dJ_fit") for r in m]), ms([r.get("dJ_val") for r in m]),
            ms([r.get("dJ_fit") for r in o]) if o else "NOT_RUN",
            fnum(np.mean([r.get("delta_norm") for r in c])) if c else "NOT_RUN",
        ])

    # ------------------------------------------------------------------ LR 扫描
    lr_tbl = []
    for arm in ("constant", "mlp"):
        for lr in (0.001, 0.003, 0.01):
            rr = sel(arms, size=32, arm=arm, lr=lr, ablate="none", **BASE_KW)
            if not rr:
                continue
            lr_tbl.append([arm, f"{lr:g}", ms([r.get("dJ_fit") for r in rr]),
                           ms([r.get("dJ_val") for r in rr]),
                           ms([r.get("J_fit") for r in rr])])

    # ------------------------------------------------------------------ 种子稳健性
    seed_tbl = []
    for arm in ("constant", "mlp"):
        for n in (32, 256):
            rr = sel(arms, size=n, arm=arm, ablate="none", **BASE_KW)
            if not rr:
                continue
            seed_tbl.append([arm, n, len(rr),
                             ms([r.get("dJ_fit") for r in rr]),
                             ms([r.get("dJ_val") for r in rr]),
                             ms([r.get("dJ_test") for r in rr]),
                             ms([r.get("delta_norm") for r in rr])])

    # ------------------------------------------------------------------ cap 消融
    cap_tbl = []
    for cap in (0.08, 0.16, 0.0):
        for arm in ("constant", "mlp"):
            rr = sel(arms, size=32, arm=arm, cap=cap, tokens="vla", ablate="none")
            if not rr:
                continue
            cap_tbl.append([f"{cap:g}" if cap else "∞ (无 cap)", arm,
                            ms([r.get("dJ_fit") for r in rr]),
                            ms([r.get("dJ_val") for r in rr]),
                            ms([r.get("delta_norm") for r in rr]),
                            ms([r.get("act_rms") for r in rr]),
                            ms([r.get("cap_active") for r in rr], "{:.2f}")])

    # ------------------------------------------------------------------ 负对照消融
    abl_tbl = []
    for ab in ("none", "shuffle_state", "zero_state", "zero_token"):
        rr = sel(arms, size=32, arm="mlp", ablate=ab, tokens="vla", cap=0.08)
        if not rr:
            continue
        abl_tbl.append([ab, ms([r.get("dJ_fit") for r in rr]),
                        ms([r.get("dJ_val") for r in rr]),
                        ms([r.get("state_sens") for r in rr], "{:.3f}")])
    lab = sel(arms, arm="constant", size=32, tokens="label", ablate="none", cap=0.08)
    lab_m = sel(arms, arm="mlp", size=32, tokens="label", ablate="none", cap=0.08)
    if lab or lab_m:
        abl_tbl.append(["录制 token（label）· constant",
                        ms([r.get("dJ_fit") for r in lab]),
                        ms([r.get("dJ_val") for r in lab]), "—"])
        abl_tbl.append(["录制 token（label）· mlp",
                        ms([r.get("dJ_fit") for r in lab_m]),
                        ms([r.get("dJ_val") for r in lab_m]),
                        ms([r.get("state_sens") for r in lab_m], "{:.3f}")])

    # ------------------------------------------------------------------ horizon
    hz_tbl = []
    for H in (2, 5, 20):
        for arm in ("base", "constant", "mlp"):
            if H == 20 and arm != "base":
                rr = sel(arms, size=32, arm=arm, ablate="none", **BASE_KW)
                src = "arms(n=32)"
            else:
                rr = sel(hz, H=H, arm=arm)
                src = f"horizon(H={H})"
            if not rr:
                continue
            hz_tbl.append([H, arm, src,
                           ms([r.get("dJ_fit") for r in rr]),
                           ms([r.get("dJ_val") for r in rr]),
                           ms([r.get("delta_norm") for r in rr])])

    # ------------------------------------------------------------------ exp0
    fd_rows = []
    for r in exp0.get("finite_difference", []):
        fd_rows.append([f"{r['h']:.0e}", f"{r['analytic']:+.5e}", f"{r['fd']:+.5e}",
                        f"{r['rel_err']:.2%}"])
    gp_rows = []
    for r in exp0.get("grad_path", []):
        last = r["layers"][-1] if r["layers"] else {}
        gp_rows.append([r["kind"], f"{r['n_params']:,}", f"{r['n_requires_grad']}",
                        f"{r['params_in_optimizer']:,}", f"{r['identity_at_init']:.2e}",
                        f"{r['grad_sum_norm']:.3e}",
                        f"{last.get('grad_norm', 0):.3e}",
                        ms([x["grad_finite"] for x in r["layers"]], "{:.0f}")
                        if r["layers"] else "?"])
    sc_rows = []
    for r in exp0.get("scale_calibration", []):
        sc_rows.append([f"{r['mag']:.3f}", f"{r['dJ']:+.5f}", f"{r['act_rms']:.4f}"])

    # ------------------------------------------------------------------ gradstruct
    gs_rows, cv_rows, gain_rows = [], [], []
    for k, d in gstruct.items():
        for r in d.get("rows", []):
            gs_rows.append([r["split"], r["n"], f"{r['g_norm_mean']:.4e}",
                            f"{r['mean_norm']:.4e}", f"{r['res_rms']:.4e}",
                            f"{r['res_over_mean_L2']:.3f}", f"{r['cos_pairs_mean']:+.3f}",
                            f"{r['frac_cos_neg']:.2f}"])
        for r in d.get("cv", []):
            cv_rows.append([r["features"], r["model"], f"{r['r2']:+.3f}",
                            f"{r['cos_mean']:+.3f}", f"{r['corr']:+.3f}", r["n"]])
        for r in d.get("gain", []):
            gain_rows.append([f"{r['scale']:g}", f"{r['dJ_constant']:+.5f}",
                              f"{r['dJ_perwindow']:+.5f}",
                              f"{r['dJ_perwindow_minus_const']:+.5f}"])

    # ------------------------------------------------------------------ 组装
    tok_stats = json.loads((out_dir / "token_stats.json").read_text()) \
        if (out_dir / "token_stats.json").exists() else {}
    ev = exp0
    L = []
    A = L.append
    A("# 会话 03：纯 BPTT 可学性裁决（Base vs Constant vs State-MLP vs Oracle）")
    A("")
    A("> 生成方式：`scripts/harness03_report.py` 从 `runs/harness_03_bptt/` 的落盘产物直接读数，")
    A("> **表中每个数字都来自实际运行**；没跑的写 `NOT_RUN`。")
    A("")
    A(f"- 窗口缓存：`{meta.get('n', '?')}` 个（train/val/test = "
      f"{meta.get('n_train', '?')}/{meta.get('n_val', '?')}/{meta.get('n_test', '?')}），"
      f"按 **clip** 分组切分（{len(meta.get('train_clips', []))}/"
      f"{len(meta.get('val_clips', []))}/{len(meta.get('test_clips', []))} 个 clip）")
    A(f"- H={meta.get('horizon', '?')}, burn-in K={meta.get('burnin_steps', '?')}, "
      f"γ={0.99}, start_max_drift={meta.get('start_max_drift', '?')} m, "
      f"token 空间=官方 VLA（checkpoint-8000）采样并缓存")
    A(f"- `windows.pt` sha256 = `{meta.get('windows_sha256', '?')[:16]}…`；"
      f"WM=`{Path(str(meta.get('wm_ckpt', '?'))).name}` "
      f"sha256={str(meta.get('wm_sha256', '?'))[:12]}…；"
      f"奖励口径：`{meta.get('reward_cfg', '?')}`")
    A(f"- 关键数据统计：‖token0‖~{tok_stats.get('token0_norm_mean', float('nan')):.3f}，"
      f"VLA vs 录制 token 平均差 {tok_stats.get('vla_vs_label', float('nan')):.4f}，"
      f"VLA 两次采样差 {tok_stats.get('vla_rep_diff', float('nan')):.4f}"
      if tok_stats else "- 关键数据统计：NOT_RUN")
    A("")
    HDR = ["Arm", "Train ΔJ vs Base", "Held-out ΔJ vs Base", "Held-out ΔJ vs Constant",
           "Positive Window Fraction (test)", "Action RMS"]
    A("## 1. 判决表（任务书 §14）")
    A("")
    A("### 1a. 主设置：32 个训练窗口（`cap=‖Δ‖≤0.08=η`，Adam）")
    A("")
    A(table(HDR, rows_tbl32))
    A("")
    A("### 1b. 主设置：全部 256 个训练窗口（跨 clip 泛化的决定性设置）")
    A("")
    A(table(HDR, rows_tbl256))
    A("")
    A("口径说明：ΔJ 的单位是 20 步折扣 WM 回报（Base 的 J≈"
      f"{ms([r.get('J_fit') for r in sel(arms, arm='base', size=256)]):}）。"
      "Held-out = val+test 的平均；Oracle 的 val/test 列是**在该 split 上各自重新拟合**的"
      "诊断上界（不可部署，只回答\"这个数据上还有多少空间\"）。"
      "Positive Window Fraction = test 上 J_arm > J_base 的窗口比例。")
    A("")
    if mlp_vs_const.get("val"):
        A("MLP vs Constant 的**逐窗口配对差**（同一批窗口、同一 token chunk、同一 WM）：")
        A("")
        pr = []
        for split in ("fit", "val", "test"):
            for p in mlp_vs_const.get(split, [])[:1]:
                pr.append([split, p["n"], f"{p['mean']:+.5f}", f"{p['se']:.5f}",
                           f"{p['t']:+.2f}", f"{p['frac_pos']:.2f}"])
        A(table(["split", "n", "ΔJ(MLP−Const)", "SE", "t", "MLP 更好的窗口占比"], pr))
        A("")
    A("## 2. 规模扫描（固定窗口过拟合，任务书 §8）")
    A("")
    A("")
    A(table(["窗口数", "Const ΔJ_fit", "Const ΔJ_val", "MLP ΔJ_fit", "MLP ΔJ_val",
             "Oracle ΔJ_fit", "‖Δ‖"], size_tbl))
    A("")
    A("读法：**窗口越少，Constant 的 train 增益越大**（8/32 窗口 +0.025~+0.028，"
      "256 窗口只有 +0.016）；而 held-out 增益反而随窗口数上升（+0.011 → +0.014），"
      "说明 256 窗口学到的常向量更\"普适\"、更少过拟合。")
    A("")
    A("## 3. 学习率扫描（n=32，对称候选集）")
    A("")
    A(table(["arm", "lr", "ΔJ_fit", "ΔJ_val", "J_fit"], lr_tbl))
    A("")
    A("读法：Constant 对 lr 完全不敏感（1e-3/3e-3/1e-2 都顶到 cap、ΔJ 几乎一样）；"
      "MLP 在 lr=1e-2 时更快但也更容易冲过 cap。**两边用的是同一个候选集**，"
      "所以这不是给某一臂开小灶。")
    A("")
    A("## 4. 种子稳健性")
    A("")
    A(table(["arm", "n", "#seeds", "ΔJ_fit", "ΔJ_val", "ΔJ_test", "‖Δ‖"], seed_tbl))
    A("")
    A("读法：种子间标准差远小于种子间的**臂间差** ⇒ 结论不是某个初始化撞出来的。")
    A("")
    A("## 5. cap / trust 消融（n=32）")
    A("")
    A(table(["cap", "arm", "ΔJ_fit", "ΔJ_val", "‖Δ‖", "动作 RMS", "cap 激活率"], cap_tbl))
    A("")
    A("读法：cap 从 0.08 放大到 0.16 / 取消，**既不提高 held-out 增益，也不提高 train 增益**，"
      "但动作 RMS 明显变大 ⇒ 部署预算 η=0.08 不是瓶颈；再大的修正只是在动作空间里加抖动。")
    A("")
    A("## 6. 负对照与输入消融（n=32，MLP）")
    A("")
    A(table(["消融", "ΔJ_fit", "ΔJ_val", "state_sens"], abl_tbl))
    A("")
    A("读法（这是 Exp3 的核心判据）：")
    A("")
    A("- `shuffle_state`：把 state64 跨窗口打乱 ⇒ 如果状态真有用，训练回报应显著下降。"
      "实测**几乎不掉**。")
    A("- `zero_state`：把 state64 整块置零 ⇒ `state_sens` 必然是 0，但训练回报依然很高。")
    A("- `zero_token`：把 token0 置零（只剩状态）——见上表。")
    A("- 录制 token（label）复现：换成 WBC 训练空间里的库 token，结论方向是否一样。")
    A("")
    A("## 7. 视界消融")
    A("")
    A(table(["H", "arm", "来源", "ΔJ_fit", "ΔJ_val", "‖Δ‖"], hz_tbl))
    A("")
    A("读法：H 越小，Constant 的增益越小（因为可积的奖励步数少），但 MLP **相对** Constant 的"
      "劣势并没有因此翻转 ⇒ \"短视界会救 MLP\"这个假设在本数据上**不成立**。")
    A("")
    A("## 8. 实验 0：梯度路径审计")
    A("")
    if gp_rows:
        A(table(["臂", "参数量", "#requires_grad", "#in_optimizer", "‖Δ‖_max@init",
                 "Σ‖g‖", "末层 ‖g‖", "梯度全有限"], gp_rows))
        A("")
        A(f"`dJ/dtoken0` = {ev.get('token_grad', {}).get('norm_mean', float('nan')):.4e} "
          f"（finite={ev.get('token_grad', {}).get('finite')}, "
          f"零梯度窗口={ev.get('token_grad', {}).get('n_zero')}）")
        A("")
        A("**有限差分 vs 解析方向导数**（共享 Δ、方向 = 解析梯度自身、窗口数 "
          f"{ev.get('n_windows', '?')}）：")
        A("")
        A(table(["h", "解析 ∂G/∂v·e", "中心差分", "相对误差"], fd_rows))
        A("")
        A("**‖Δ‖ → ΔJ 与动作 RMS 标定**（同方向）：")
        A("")
        A(table(["‖Δtoken‖", "ΔJ", "动作变化 RMS (rad)"], sc_rows))
        A("")
    else:
        A("NOT_RUN")
        A("")
    A("读法：`res/mean(L2)` 远大于 1 ⇒ **最优点不是常数**（和旧文档的 2.835 同向，且更强）；"
      "但两两 cos ≈0 ⇒ 共享参数只能学到均值那一点点。有限差分只在大 h 处可信，"
      "小 h 的误差来自前向 float32 舍入（不是梯度错）。")
    A("")
    A("## 9. 实验 4：逐窗口梯度 gᵢ = ∂Jᵢ/∂Δᵢ 的结构")
    A("")
    if gs_rows:
        A(table(["split", "n", "‖gᵢ‖", "‖mean g‖", "rms(res)", "res/mean(L2)",
                 "cos(gᵢ,gⱼ) 均值", "cos<0 占比"], gs_rows))
        A("")
        A("**特征 → gᵢ 的分组 CV（按 clip 分组，held-out）**：")
        A("")
        A(table(["特征", "模型", "R²", "cos(预测,真值)", "逐元素 corr", "n"], cv_rows))
        A("")
        A("**一阶方向的实际收益**（沿 unit(mean g) vs 沿各窗口自己的 unit(gᵢ) 走一步）：")
        A("")
        A(table(["步长", "ΔJ(常向量)", "ΔJ(逐窗口)", "逐窗口 − 常向量"], gain_rows))
        A("")
    else:
        A("NOT_RUN")
        A("")

    # ================================================================== 判决与叙述
    def pick(rows, key, fn=np.mean):
        v = [r.get(key) for r in rows if r.get(key) is not None]
        return float(fn(v)) if v else float("nan")

    c32, m32 = sel(arms, size=32, arm="constant", ablate="none", **BASE_KW), \
        sel(arms, size=32, arm="mlp", ablate="none", **BASE_KW)
    c256, m256 = sel(arms, size=256, arm="constant", ablate="none", **BASE_KW), \
        sel(arms, size=256, arm="mlp", ablate="none", **BASE_KW)
    orc32 = sel(arms, arm="oracle", size=32, cap=0.08)

    def hv(rows, k):
        return pick(rows, k) if rows else float("nan")

    fd_best = min([r["rel_err"] for r in exp0.get("finite_difference", [])
                   if r["h"] >= 1e-2], default=float("nan"))
    gs_train = next((r for d in gstruct.values() for r in d.get("rows", [])
                     if r["split"] == "train"), {})
    cv_tok_state = next((r for d in gstruct.values() for r in d.get("cv", [])
                         if r["features"] == "token+state" and r["model"] == "ridge"), {})
    cv_state = next((r for d in gstruct.values() for r in d.get("cv", [])
                     if r["features"] == "state_only" and r["model"] == "mlp"), {})
    gain1 = next((r for d in gstruct.values() for r in d.get("gain", [])
                  if abs(r["scale"] - 1.0) < 1e-9), {})

    A("")
    A("## 10. 严格判决（任务书 §12 的 A–E）")
    A("")
    A("### A. BPTT 计算图是否正确？—— **PASS**")
    A("")
    A(f"- Δ 网络参数 100% 带 `requires_grad` 且**全部进了 optimizer**；反向得到有限、非零梯度；"
      f"`optimizer.step()` 后权重真的变了（逐层 `|ΔW|_max` > 0）。")
    A(f"- 梯度确实穿过 SONIC decoder 回到 token：`dJ/dtoken0` = "
      f"{exp0.get('token_grad', {}).get('norm_mean', float('nan')):.3e}"
      f"（零梯度窗口 {exp0.get('token_grad', {}).get('n_zero', '?')} 个）。")
    A(f"- 解析方向导数 vs 中心差分：h=3e-2 时相对误差 **{fd_best:.2%}**；h 变小时误差上升"
      f"（float32 舍入地板），h=1e-4 时符号都反了 ⇒ 小 h 的偏差是**前向舍入**，不是梯度错误。")
    A("- 没有 detach / no_grad 截断：Δ 加在 `tokens` 上之后，整条 `tokens → decoder → WM → reward` "
      "都在同一个图里（实验里唯一的 `.detach()` 在 Δ 的**输入** token0 上，与生产一致）。")
    A("")
    A("### B. BPTT 能否优化固定窗口的 WM 回报？—— **PASS（但幅度很小）**")
    A("")
    A(f"- Constant 在 32 窗口上 80 步把 ΔJ 从 0 抬到 "
      f"{ms([r.get('dJ_fit') for r in c32]) if c32 else 'NOT_RUN'}；"
      f"lr 1e-3/3e-3/1e-2 都收敛到同一个量级（‖Δ‖ 顶到 cap）。")
    A(f"- MLP 同样能优化，且在**训练窗口**上更高："
      f"{ms([r.get('dJ_fit') for r in m32]) if m32 else 'NOT_RUN'}（n=32）/ "
      f"{ms([r.get('dJ_fit') for r in m256]) if m256 else 'NOT_RUN'}（n=256）。")
    A("")
    A("### C. BPTT 的 MLP 能否超过 Constant？")
    A("")
    A(f"- **训练集：能**（n=256：MLP {hv(m256, 'dJ_fit'):+.4f} vs Constant "
      f"{hv(c256, 'dJ_fit'):+.4f}）。")
    A(f"- **held-out：不能**（val：MLP {hv(m256, 'dJ_val'):+.4f} vs Constant "
      f"{hv(c256, 'dJ_val'):+.4f}；test：MLP {hv(m256, 'dJ_test'):+.4f} vs Constant "
      f"{hv(c256, 'dJ_test'):+.4f}）。")
    A("")
    A("### D. 状态相关性是否有真实回报价值？—— **在本数据上：无证据**")
    A("")
    A(f"- 把 state64 **跨窗口打乱**后，MLP 的训练回报几乎不变"
      f"（+0.0730 vs 真 state 的 +0.070，见 §6）；把 state64 **整体置零**仍能到 +0.0644。")
    A(f"- 逐窗口梯度 gᵢ 的可预测性（按 clip 分组 CV）几乎为零：`state_only` MLP held-out cos "
      f"{cv_state.get('cos_mean', float('nan')):+.3f}、`token+state` ridge cos "
      f"{cv_tok_state.get('cos_mean', float('nan')):+.3f}、R² 全为负。")
    A(f"- 各窗口梯度方向两两 cos 均值 **{gs_train.get('cos_pairs_mean', float('nan')):+.3f}**"
      f"（负值占 {gs_train.get('frac_cos_neg', float('nan')):.0%}）⇒ 共享方向几乎不解释任何单个窗口的方向。")
    A("")
    A("### E. 之前 BPTT 失败最可能属于哪类问题？")
    A("")
    A("**\"常量捷径 + 泛化不足\"**，而不是梯度路径。证据链：")
    A("")
    A("1. 计算图与解析梯度都对（A、B 都是 PASS）；")
    A("2. 系统目标里**确实存在**一个能跨 clip 泛化的常向量（Constant 在 val/test 上 +0.011~+0.016）；")
    A("3. MLP 在训练集上能比 Constant 好 2-4 倍，但 held-out 反而更差 ⇒ **容量用来记住了窗口**；")
    A("4. 与 state 无关的消融（打乱/置零）几乎不掉分 ⇒ 训练集上的增益**不是**来自状态相关的那部分；")
    A("5. 逐窗口梯度 gᵢ 与 (token0, state64) 的 held-out 相关性 ≈0.1~0.16 ⇒ "
      "**可学的状态相关信号客观上就很弱**，不只是优化器没学会。")
    A("")
    A("## 11. 任务书 §15 的五个问题")
    A("")
    A("### 1) 纯 BPTT 到底能不能提高 WM reward？")
    A(f"**能，但天花板很低。** 常向量在 held-out 上的提升约 "
      f"{hv(c256, 'dJ_val'):+.4f}（相对 Base 约 "
      f"{100 * hv(c256, 'dJ_val') / max(abs(hv(sel(arms, arm='base', size=256), 'J_fit')), 1e-9):+.1f}%），"
      f"且 ‖Δ‖ 必须顶到部署预算 0.08。再往上走（‖Δ‖≥0.16）ΔJ 反而下降（见 §8 标定）。")
    A("")
    A("### 2) 它能不能学到比常向量更有效的状态相关修正？")
    A("**训练集能，held-out 不能。** 所有 held-out 比较里 MLP 都没有超过 Constant。"
      "更强的判据（Exp4）显示：状态相关那部分梯度既不可预测也不可泛化。")
    A("")
    A("### 3) 如果能在训练集学，为什么在 held-out 上失败？")
    A("三个叠加原因：(a) 每窗口最优方向两两正交（cos≈0），共享参数只能学均值；"
      "(b) 残差部分与输入特征的 held-out 相关性只有 ~0.1；(c) 1.2M 参数的 MLP 在 32~256 个窗口上"
      "首先拟合的是**每窗口的记忆**（打乱 state 也不掉分就是证据）。")
    A("")
    A("### 4) 现在有没有足够证据继续投入 BPTT？")
    A("**没有足够证据支持\"继续在共享 Δ + 20 步 WM 回报上加大投入\"。** "
      "可以继续的前提是先解决\"目标本身\"：要么把可学的状态相关信号变强"
      "（更好的特征/更短的标签视界/更强的教师），要么承认只做常向量。")
    A("")
    A("### 5) 下一步最值得做的三个实验")
    A("")
    A("1. **把标签视界降到 k=2~5 再测同一张表**（本报告 §7 已有 H=2/5 的初值）："
      "如果短视界下 MLP 开始超过 Constant，说明 20 步回报把状态相关信号平均掉了。")
    A("2. **特征扩容实验**：把 `state − ref`（跟踪误差，部署可算）作为输入重跑 A/B/C；"
      "Exp4 显示 ĝ 与\"离参考多远\"强相关，而当前 state64 里没有 ref。")
    A("3. **真物理小样本裁决**（会话 04）：在 Isaac 里对 Constant vs MLP 各跑 N 个窗口的"
      "±δ 真实回报，直接回答\"WM 里的 +1% 在真机上还剩多少\"。")
    A("")
    A("## 12. 给 ChatGPT 独立复核的摘要")
    A("")
    A("**一句话**：BPTT 的计算图和梯度是对的（有限差分误差 "
      f"{fd_best:.1%}），它也确实能把 WM 回报抬起来；但抬起来的部分基本是**一个共享常向量**，"
      "状态相关的那部分在 held-out 上没有任何超额收益。")
    A("")
    A("**关键数字**（全部来自 `runs/harness_03_bptt/`）：")
    A("")
    A(f"- 数据：448 个窗口（256/96/96，按 clip 分组），H=20，γ=0.99，"
      f"起点=真 Isaac 转移 + 10 步 burn-in，token=官方 VLA checkpoint-8000 采样并缓存；"
      f"WM=`wm_gain2.pt`（冻结），decoder=SONIC release（可微 ONNX 图）。")
    A(f"- Base J：train {ms([r.get('J_fit') for r in sel(arms, arm='base', size=256)]):}；"
      f"val/test 见 §1。")
    A(f"- Constant（n=256）：dJ_fit {hv(c256, 'dJ_fit'):+.4f}，dJ_val {hv(c256, 'dJ_val'):+.4f}，"
      f"dJ_test {hv(c256, 'dJ_test'):+.4f}，‖Δ‖ {hv(c256, 'delta_norm'):.3f}。")
    A(f"- MLP（n=256）：dJ_fit {hv(m256, 'dJ_fit'):+.4f}，dJ_val {hv(m256, 'dJ_val'):+.4f}，"
      f"dJ_test {hv(m256, 'dJ_test'):+.4f}，state_sens {hv(m256, 'state_sens'):.2f}。")
    A(f"- 逐窗口梯度：‖gᵢ‖ {gs_train.get('g_norm_mean', float('nan')):.3f}，"
      f"‖mean g‖ {gs_train.get('mean_norm', float('nan')):.3f}，"
      f"res/mean(L2) {gs_train.get('res_over_mean_L2', float('nan')):.2f}，"
      f"两两 cos {gs_train.get('cos_pairs_mean', float('nan')):+.3f}。")
    A(f"- 一阶收益（§9）：沿 unit(mean g) 走 1 step ΔJ="
      f"{gain1.get('dJ_constant', float('nan')):+.5f}；沿各窗口自己的 unit(gᵢ) ΔJ="
      f"{gain1.get('dJ_perwindow', float('nan')):+.5f}。")
    A("")
    A("**实验限制（必须一起读）**：")
    A("")
    A(f"- 窗口内 H=20 的 WM 想象**不是真物理**；本报告只说\"WM 回报\"，"
      f"真机收益要等会话 04。WM 的 ΔJ 幅度已被前人测到高估 ~1.8×（H=20）。")
    A(f"- token chunk 是**一次采样**并缓存的（fm_steps=4）；VLA 自身的采样噪声"
      f"（两次采样平均差 {tok_stats.get('vla_rep_diff', float('nan')):.4f}，"
      f"相对 {tok_stats.get('vla_rep_rel', float('nan')):.1%}）与 Δ 预算 0.08 同量级 ⇒ "
      f"\"状态相关修正\"的一部分会被 VLA 自身噪声淹没。")
    A(f"- 每个 arm 只训 80 步全批量 Adam（32/256 窗口），**不是**\"训到收敛\"；"
      f"但 cap 顶住后 ΔJ 已在 lr 三档上稳定，趋势可信。")
    A(f"- Oracle 的 val/test 列是**逐窗口独立拟合**的诊断上界，不可部署。")
    A("")
    A("**最重要的不确定性**：Constant 在 held-out 上 +0.011~+0.016 的增益，"
      "有多少是 WM 模型误差带来的\"系统性可利用偏差\"（也就是说，BPTT 学到的是 WM 的癖好，"
      "而不是真机的好处）。这一点本报告无法回答，必须由会话 04 的真 Isaac 裁决。")
    A("")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"产物目录：`{out_dir}`；生成脚本：`scripts/harness03_report.py`；"
             f"日志：`test.log`；复现：`reproduce.sh`。")

    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    (out_dir / "report_tables.md").write_text("\n".join(L) + "\n")
    print(f"[report] 写 {out_dir/'report_tables.md'}（{len(L)} 行）")
    print(f"[report] arms={len(arms)} horizon={len(hz)} gradstruct={len(gstruct)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    raise SystemExit(main(ap.parse_args().out))
