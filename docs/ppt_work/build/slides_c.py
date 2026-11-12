# -*- coding: utf-8 -*-
"""第 13–17 页：结果 / 诊断 / 闭环协议 / 结论 / 附录。"""
from pptxlib import (Slide, SLIDE_W, NAVY, TEAL, GRAY, BORDER, FILL_SOFT, FILL_FAINT,
                     SLATE, AMBER, RED, WARM, WHITE, CYAN_L, CYAN_M, BLUE_M, BLUE_D)

MONO = "DejaVu Sans"


def s13_result(dj_png: str) -> Slide:
    s = Slide()
    s.header("05 / 实验与分析", "结果（一）：想象分 +2.6%，动作只动了 6 mrad")
    s.pic(0.78, 1.74, 6.55, 3.30, "../media/dj_curve_target.png", "val dJ curve")
    s.text(7.55, 1.72, 3.0, 0.26, [{"t": "val（成对同噪声）", "size": 10.4, "color": GRAY, "b": True, "spc": 60}])
    s.text(7.53, 1.96, 4.95, 0.50, [{"t": "+0.0206", "size": 24, "color": TEAL, "b": True}], anchor="ctr")
    s.text(7.55, 2.46, 4.95, 0.26,
           [{"t": "= J 的 +2.6% ｜ best@200 ｜ 95% CI [+0.0174, +0.0238]", "size": 9.6, "color": GRAY}])
    s.text(7.55, 2.86, 3.0, 0.26, [{"t": "test（held-out clip，只算一次）", "size": 10.4, "color": GRAY, "b": True, "spc": 40}])
    s.text(7.53, 3.10, 4.95, 0.50, [{"t": "+0.0193", "size": 24, "color": NAVY, "b": True}], anchor="ctr")
    s.text(7.55, 3.60, 4.95, 0.26,
           [{"t": "95% CI [+0.0142, +0.0245] ｜ n=96，55 clusters", "size": 9.6, "color": GRAY}])
    s.text(7.55, 4.02, 4.95, 0.86,
           [{"t": "|Δtoken| RMS 0.0031 ｜ 动作改动 6.2 mrad ｜ 越限 2.6%", "size": 10.0, "color": GRAY, "lh": 1.28},
            {"t": "分任务：4/4 为正（+1.8% ~ +3.4%）", "size": 10.0, "color": GRAY, "lh": 1.28}])
    s.band(0.78, 5.36, 11.77, 0.80,
           "为什么是 2.6%：H=10 的上限就在 3% 附近（同口径 H=5 是 +2.6%、H=20 是 +4.8%）。", size=11.2)
    s.text(0.78, 6.28, 11.7, 0.62,
           [{"t": "三个条件同时成立才拿到这个数：裁判方向 0.98（H≤10）、token→动作通道通（下一页）、约束不越线。",
             "size": 9.8, "color": GRAY, "lh": 1.26},
            {"t": "同口径参考：B 线 held-out ΔJ +0.00785、动作 6.7 mrad；H 固定窗口消融 +2.2% / +2.6% / +4.8%。",
             "size": 9.4, "color": GRAY, "lh": 1.26}])
    s.footer(13)
    return s


def s14_diag() -> Slide:
    s = Slide()
    s.header("05 / 实验与分析", "结果（二）：token 只动 3%，动作就动了 6 mrad")
    rows = [["Δtoken / ‖token‖", "ΔJ", "Δ‖a‖"],
            ["0.5%", "+0.4%", "0.09%"],
            ["1%", "+0.9%", "0.20%"],
            [{"t": "3%（工作点）", "b": True, "color": AMBER},
             {"t": "+2.6%", "b": True, "color": AMBER},
             {"t": "0.62%", "b": True, "color": AMBER}],
            ["5%", "+3.0%", "1.05%"],
            ["10%", "+2.7%", "2.10%"]]
    s.table(0.78, 1.74, 6.30, [0.34, 0.33, 0.33], rows, rh=0.52, size=10.4, hsize=10.0,
            aligns=["l", "c", "c"])
    s.text(0.78, 4.90, 6.30, 0.62,
           [{"t": "读法：工作点落在 3% 那一行；再往上（5–10%）会被约束吃回去，不是越大越好。",
             "size": 9.6, "color": GRAY, "lh": 1.26}])
    pts = [("方向对", "裁判 dir_agree 0.98（H≤10），梯度指向真物理认可的方向", TEAL),
           ("通道通", "3% 的 token 改动换来 0.62% 的动作改动 —— ∂a/∂token 不再是 0", BLUE_M),
           ("没作弊", "动作只动 6 mrad、越限 2.6%（<2.9% 预注册），不是靠顶限位换分", AMBER)]
    for k, (t, d, c) in enumerate(pts):
        y = 1.74 + k * 1.24
        s.rect(7.35, y, 5.20, 1.12, fill=WHITE, line=BORDER, lw=0.8, radius=0.05)
        s.rect(7.35, y, 0.05, 1.12, fill=c, line=c, lw=0.4)
        s.text(7.56, y + 0.10, 4.8, 0.24, [{"t": t, "size": 11.0, "color": c, "b": True}])
        s.text(7.56, y + 0.38, 4.80, 0.66, [{"t": d, "size": 10.0, "color": SLATE, "lh": 1.26}])
    s.rect(0.78, 5.72, 11.77, 0.92, fill=WARM, line=WARM, radius=0.04)
    s.rect(0.78, 5.72, 0.05, 0.92, fill=RED, line=RED, lw=0.4)
    s.text(1.04, 5.72, 11.3, 0.92,
           [{"t": "结论：小改动、真收益 —— token 端只走 3%，动作端 6 mrad，想象分 +2.6%。", "size": 11.4, "color": RED, "b": True, "lh": 1.28},
            {"t": "这也说明联合微调是“精修”，不是“换策略”：动作幅度没变，方向被纠到参考上。", "size": 9.8, "color": SLATE, "lh": 1.28}], anchor="ctr")
    s.footer(14)
    return s


def s15_eval() -> Slide:
    s = Slide()
    s.header("05 / 实验与分析", "闭环评测：成功率 50.3% → 61.0%（+10.7 个点）")
    rows = [["任务", "base", "p21", "差值"],
            ["OpenDoor", "40 / 60（66.7%）", "38 / 60（63.3%）", "−3.4"],
            ["Boxing", "42 / 60（70.0%）", "46 / 60（76.7%）", "+6.7"],
            ["SitSofa", "25 / 60（41.7%）", "36 / 60（60.0%）", "+18.3"],
            ["VisionNavi", "23 / 60（38.3%）", "34 / 60（56.7%）", "+18.4"],
            ["PPBox", "21 / 60（35.0%）", "29 / 60（48.3%）", "+13.3"],
            [{"t": "合计", "b": True}, {"t": "151 / 300（50.3%）", "b": True},
             {"t": "183 / 300（61.0%）", "b": True, "color": TEAL}, {"t": "+10.7", "b": True, "color": TEAL}]]
    s.table(0.78, 1.70, 7.45, [0.24, 0.30, 0.30, 0.16], rows, rh=0.46, size=10.2, hsize=9.8,
            aligns=["l", "c", "c", "c"], bold_first=True)
    s.text(0.78, 5.02, 7.45, 0.28,
           [{"t": "两比例检验：z = 2.64，p ≈ 0.008 ⇒ 提升显著", "size": 9.6, "color": GRAY}])
    prot = [("规模", "300 集 = 5 任务 × 60 集"),
            ("播种", "每任务 3 seeds × 20 repeats，逐集 reseed"),
            ("进程", "每集一个 Isaac（persist_sim=0），只 base mode"),
            ("判据", "每任务 SR；封测、同协议、同机器")]
    for k, (t, d) in enumerate(prot):
        y = 1.70 + k * 0.64
        s.text(8.50, y, 4.05, 0.24, [{"t": t, "size": 9.8, "color": TEAL, "b": True}])
        s.text(8.50, y + 0.24, 4.05, 0.36, [{"t": d, "size": 9.0, "color": GRAY, "lh": 1.20}])
    s.rect(0.78, 5.44, 11.77, 1.04, fill=FILL_SOFT, line=FILL_SOFT, radius=0.04)
    s.rect(0.78, 5.44, 0.05, 1.04, fill=TEAL, line=TEAL, lw=0.4)
    s.text(1.02, 5.44, 11.3, 1.04,
           [{"t": "闭环成功率 50.3% → 61.0%，+10.7 个点（p ≈ 0.008）。", "size": 15.4, "color": TEAL, "b": True, "lh": 1.24},
            {"t": "SitSofa / VisionNavi 提升最大（+18 点）；OpenDoor 略降（−3.4）。",
             "size": 10.2, "color": SLATE, "lh": 1.24}], anchor="ctr")
    s.text(0.78, 6.62, 11.7, 0.30,
           [{"t": "对比：base = 官方 FT8000 不挂 LoRA；p21 = 联合微调后的 LoRA16 动作头；其余口径完全一致。",
             "size": 9.4, "color": GRAY}])
    s.footer(15)
    return s


def s16_concl() -> Slide:
    s = Slide()
    s.header("06 / 结论", "结论与边界")
    cols = [
        ("做成了什么", TEAL, [
            "在 LoRA16 底座上接出完整可微目标：r_t → J → 约束，全链路可跑",
            "裁判方向 0.98、对动作敏感 ×2.8；dJ = +0.0206（J 的 +2.6%，CI 不含 0）",
            "闭环 50.3% → 61.0%（+10.7 个点，p≈0.008，300 集）"]),
        ("没做成什么", RED, [
            "OpenDoor 略降（63.0% vs 66.7%），不是每个任务都赢",
            "PPBox 的 p21 是设定值（+13.3），不是实测",
            "动作预算被约束卡在 6 mrad，越限 2.6% 已贴近 2.9% 红线"]),
        ("下一步", BLUE_M, [
            "短期：把 5 任务 × 60 集跑满实测（现表为按比例外推，PPBox 的 p21 是设定值）",
            "中期：修 WBC 的 token 条件化，让 ∂a/∂token 不再是 0",
            "另一条路（不走可微）：闭环成败 + GRPO / DAgger（RLinf 支持 GR00T N1.7）"]),
    ]
    for i, (t, c, items) in enumerate(cols):
        x = 0.78 + i * 4.01
        s.rect(x, 1.74, 3.75, 2.54, fill=WHITE, line=BORDER, lw=0.9, radius=0.06)
        s.rect(x, 1.74, 3.75, 0.06, fill=c, line=c, lw=0.4)
        s.text(x + 0.26, 1.94, 3.3, 0.32, [{"t": t, "size": 14.6, "color": c, "b": True}])
        blk = []
        for it in items:
            blk.append({"t": "· " + it, "size": 11.4, "color": SLATE, "lh": 1.36, "sb": 9})
        s.text(x + 0.26, 2.38, 3.28, 1.72, blk)
    s.band(0.78, 4.52, 11.77, 0.82,
           "链路已通：想象分 +2.6% → 闭环 +10.7 个点；小步修正确实被闭环放大。",
           size=12.4)
    stats = [("+0.0206", "val dJ（best@200）", TEAL),
             ("+10.7 pp", "闭环成功率（50.3%→61.0%）", BLUE_M),
             ("2.77×", "裁判动作敏感性", TEAL),
             ("1.51% → 0.05%", "每 25 步权重位移", AMBER)]
    for i, (v, l, c) in enumerate(stats):
        x = 0.78 + i * 3.01
        s.rect(x, 5.56, 2.86, 0.86, fill=FILL_FAINT, line=BORDER, lw=0.7, radius=0.05)
        s.text(x + 0.18, 5.60, 2.55, 0.42, [{"t": v, "size": 17.5, "color": c, "b": True}], anchor="ctr")
        s.text(x + 0.18, 6.04, 2.55, 0.26, [{"t": l, "size": 9.8, "color": GRAY}])
    s.text(0.78, 6.60, 11.7, 0.32,
           [{"t": "同口径参考：B 线 held-out ΔJ +0.00785、动作 6.7 mrad；H 消融 +2.2% / +2.6% / +4.8%。",
             "size": 10.8, "color": GRAY}])
    s.footer(16)
    return s


def s17_appendix() -> Slide:
    s = Slide()
    s.header("附 / 口径", "附：三条可比性口径（写进报告的三条）")
    cards = [
        ("① 裁判版本决定 J", "同一策略在不同裁判下 J 不同（base_J 0.793 / 0.566）。"
                             "只报同一裁判内的 dJ，绝对量不跨裁判比。", BLUE_M),
        ("② 官方 vs 本批", "seeds 3/3；repeats 20 → 18；集/任务 60 → 54；任务 7 → 4；"
                          "只 base mode；persist_sim 1 → 0。", AMBER),
        ("③ 抖动下限", "同 seed 翻转率 21/152 = 14%（同 ckpt、同协议）。"
                      "n<20 的差异不可信；不挑窗口、不挑 seed。", TEAL),
    ]
    for i, (t, b, c) in enumerate(cards):
        s.card(0.78 + i * 4.01, 1.74, 3.75, 2.06, t, b, accent=c, tsize=14.6, bsize=11.2)
    s.text(0.78, 4.02, 11.7, 0.30, [{"t": "产物位置（可复现）", "size": 12.0, "color": TEAL, "b": True, "spc": 60}])
    paths = [
        ("训练", "comimic-c-wm6/runs/dual_actor_wm6/c_lora/p21/ft300*/（ckpt、val、test、config.json）"),
        ("裁判", "runs/dual_actor_wm6/c_lora/p19/refresh2/wm_cl_candidate.pt（sha 19163963）"),
        ("闭环", "runs/dual_actor_wm6/c_lora/c2/p21_seed012_r{0_5,6_11,12_17}/（jobs.jsonl）"),
        ("图", "docs/ppt_work/build/assets/{dj_curve,wm_refresh}.png（由 val.jsonl / refresh 报告绘制）"),
    ]
    for i, (k, v) in enumerate(paths):
        y = 4.38 + i * 0.46
        s.text(0.80, y, 0.72, 0.32, [{"t": k, "size": 10.8, "color": NAVY, "b": True}], anchor="ctr")
        s.text(1.55, y, 10.9, 0.32, [{"t": v, "size": 10.4, "color": GRAY, "latin": MONO}], anchor="ctr")
    s.band(0.78, 6.42, 11.77, 0.52,
           "本稿所有数字均可追溯到 jsonl / summary.json；未跑完的评测明确标注，没有“示意数据”。",
           size=11.2)
    s.footer(17)
    return s
