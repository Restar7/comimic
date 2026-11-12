# -*- coding: utf-8 -*-
"""第 1–6 页：封面 / 背景 / 框架 / 思想 / 数据 / WM。"""
from pptxlib import (Slide, SLIDE_W, SLIDE_H, NAVY, TEAL, GRAY, BORDER, FILL_SOFT, FILL_FAINT,
                     SLATE, AMBER, RED, WARM, WHITE, CYAN_L, BLUE_M, BLUE_D)

FONT = "Noto Sans CJK SC"
MONO = "DejaVu Sans"


def s01_cover() -> Slide:
    s = Slide()
    s.rect(0, 0, SLIDE_W, 0.115, fill=NAVY, line=NAVY)
    s.rect(0, 0.115, 4.45, SLIDE_H - 0.115, fill=NAVY, line=NAVY)
    s.text(0.95, 1.32, 3.2, 0.24,
           [{"t": "ROBOTICS · COURSE DESIGN", "size": 10.8, "color": "84DBE2", "b": True, "spc": 120}],
           anchor="ctr")
    s.text(0.92, 1.74, 3.45, 0.86, [{"t": "COMIMIC", "size": 42, "color": WHITE, "b": True}], anchor="ctr")
    s.rect(0.95, 2.72, 0.95, 0.035, fill="84DBE2", line="84DBE2")
    s.text(0.95, 2.95, 3.2, 1.05,
           [{"t": "面向人形全身执行误差的", "size": 17.5, "color": WHITE, "b": True, "lh": 1.30},
            {"t": "可微 VLA–WBC 联合优化", "size": 17.5, "color": WHITE, "b": True, "lh": 1.30}])
    s.text(0.95, 4.28, 3.2, 1.5,
           [{"t": "方法口径", "size": 10.5, "color": "84DBE2", "b": True, "spc": 80},
            {"t": "底座：LoRA16 动作头", "size": 12.6, "color": WHITE, "sb": 4},
            {"t": "+ A′ 世界模型（可微裁判）", "size": 12.6, "color": WHITE},
            {"t": "+ BPTT 联合微调（训练侧）", "size": 12.6, "color": WHITE}])
    s.text(0.95, 6.55, 3.4, 0.3,
           [{"t": "课程设计汇报 · 约 10 分钟", "size": 10.5, "color": "B4DEE7"}])

    s.text(5.05, 1.55, 7.6, 0.42,
           [{"t": "GR00T N1.7  +  SONIC  +  A′ World Model  +  BPTT", "size": 15.5, "color": TEAL, "b": True}],
           anchor="ctr")
    s.text(5.05, 2.15, 7.5, 1.5,
           [{"t": "从真实执行后的 Root / Body 偏差，", "size": 25, "color": NAVY, "b": True, "lh": 1.28},
            {"t": "反传到上层运动决策", "size": 25, "color": NAVY, "b": True, "lh": 1.28}])
    s.rect(5.05, 3.95, 7.3, 0.02, fill=BORDER, line=BORDER)
    s.text(5.05, 4.15, 7.3, 0.95,
           [{"t": "方法底座是 LoRA16 动作头（主干冻结）；", "size": 13.6, "color": GRAY, "lh": 1.45},
            {"t": "目标函数与训练协议见第 7–10 页（A′ WM 当可微裁判 + BPTT）。", "size": 13.6, "color": GRAY, "lh": 1.45}])
    s.text(5.05, 5.35, 7.3, 0.9,
           [{"t": "数字分三类标注：实测 / 同口径参考 / 目标量级；", "size": 12.2, "color": GRAY},
            {"t": "未跑完的评测明确标“进行中”，不画示意曲线。", "size": 12.2, "color": GRAY}])
    s.rect(5.05, 6.42, 7.3, 0.5, fill=FILL_SOFT, line=FILL_SOFT, radius=0.03)
    s.text(5.28, 6.42, 6.9, 0.5,
           [{"t": "汇报人／学号：__________________________", "size": 12, "color": NAVY}],
           anchor="ctr")
    s.text(11.6, 0.32, 1.2, 0.2, [{"t": "2026 · 11", "size": 9.5, "color": GRAY}], align="r")
    return s


def s02_problem() -> Slide:
    s = Slide()
    s.header("01 / 研究背景与目标", "研究问题：VLA 到“真正执行”之间隔着什么？")
    cards = [
        ("① 标签空间", "传统操作 VLA",
         "模型输出动作后直接对接机械臂控制接口，损失容易对齐动作标签。\n训练信号干净、链路也短。",
         "⇒ 损失只看得见动作，看不见结果"),
        ("② 物理执行", "人形全身 VLA",
         "运动意图要先过 SONIC WBC（994D → 29D）、PD 控制与接触动力学，才落到关节目标。",
         "⇒ 执行链越长，标签与结果越脱节"),
        ("③ 训练信号", "梯度在仿真处阻断",
         "真实 root / body 误差可观测，但 Isaac 不可微，回传不到 VLA 的动作头。",
         "⇒ 有误差、没梯度，只能靠调参"),
    ]
    for i, (tag, t, b, f) in enumerate(cards):
        x = 0.78 + i * 4.01
        s.card(x, 1.72, 3.75, 2.30, t, b, tag=tag, tsize=17.5, bsize=12.4, foot=f)
    s.band(0.78, 4.28, 11.77, 0.95,
           "研究命题：把“执行后的 Root / Body 偏差”变成可微训练信号。")
    s.text(0.78, 5.42, 5.7, 0.32, [{"t": "本文的走法", "size": 11.5, "color": TEAL, "b": True, "spc": 60}])
    s.text(0.78, 5.76, 5.6, 1.1,
           [{"t": "底座：LoRA16 动作头（主干冻结）", "size": 12.6, "color": GRAY, "lh": 1.40},
            {"t": "目标：A′ 可微世界模型当裁判，BPTT 回传到 LoRA", "size": 12.6, "color": GRAY, "lh": 1.40},
            {"t": "验收：真实 Isaac 闭环成功率（不可微，只做终点判定）", "size": 12.6, "color": GRAY, "lh": 1.40}])
    s.text(6.68, 5.42, 5.8, 0.32, [{"t": "本文不做什么", "size": 11.5, "color": AMBER, "b": True, "spc": 60}])
    s.text(6.68, 5.76, 5.9, 1.1,
           [{"t": "不替换线上 SONIC，不改控制接口与部署频率", "size": 12.6, "color": GRAY, "lh": 1.40},
            {"t": "不声称 WM 回报 ↑ 等于闭环成功 ↑（实测会打脸）", "size": 12.6, "color": GRAY, "lh": 1.40},
            {"t": "不把半程评测当结论（见第 14 页的 98/216）", "size": 12.6, "color": GRAY, "lh": 1.40}])
    s.footer(2)
    return s


def s03_system() -> Slide:
    s = Slide()
    s.header("02 / 系统框架", "官方执行链，以及我们只改动的两处")
    items = [
        ("输入", "RGB + 指令\n机器人状态"),
        ("GR00T N1.7", "Cosmos-Reason2-2B\nVLA 主干"),
        ("Motion Chunk", "40 帧 × 64 维\n0.8 s 运动意图"),
        ("SONIC", "994D → 29D\nq_target"),
        ("PD / Isaac", "物理执行\n接触动力学"),
    ]
    s.arrow_flow(items, 0.78, 1.78, 11.77, 1.24, gap=0.20, tsize=13.4, bsize=11.2)
    s.card(0.78, 3.30, 5.78, 1.34, "改动 ①：只在训练时换状态传播",
           "把不可微的 Isaac 换成 A′ 可微世界模型；部署时仍是原 SONIC + 原控制接口。",
           accent=TEAL, tsize=15.0, bsize=12.0)
    s.card(6.72, 3.30, 5.83, 1.34, "改动 ②：只在动作头挂可训参数",
           "LoRA16（attn1 / attn2 / ff）+ proj_out，主干与 VLM 全程冻结。",
           accent=BLUE_M, tsize=15.0, bsize=12.0)
    s.band(0.78, 4.92, 11.77, 1.06,
           "数据契约：token 64 + 10 帧历史（93×10）= 994D → 29D q_target；50 Hz 控制，"
           "chunk 40 帧（0.8 s），每 20 帧（0.4 s）重规划。", size=11.8)
    s.text(0.78, 6.18, 11.7, 0.36,
           [{"t": "部署侧一行都不改 —— “联合微调”改的只有训练侧的状态传播与动作头。",
             "size": 11.8, "color": GRAY}])
    s.footer(3)
    return s


def s04_idea() -> Slide:
    s = Slide()
    s.header("02 / 研究思想", "借鉴 SuperMimic：用可微世界模型提供梯度")
    s.card(0.78, 1.74, 5.78, 2.80, "不可微真实仿真 · 验收路径",
           "GR00T → SONIC → Isaac\n"
           "拿到 s_{t+1} 与 root / body 误差\n"
           "但 Jacobian 不可得：梯度到 q_target 就断\n"
           "只能当“终点判定”，不能当训练信号",
           accent=BLUE_M, tsize=16.5, bsize=12.6, lh=1.52,
           foot="⇒ 能验收，不能训练")
    s.card(6.72, 1.74, 5.83, 2.80, "可微替代模型 · 训练路径",
           "用真实转移训练 Fω（A′ 世界模型）\n"
           "Torch SONIC + Fω 做想象 rollout\n"
           "BPTT：∂J/∂token 一路回到动作头\n"
           "可以反复求导，成本与采样解耦",
           accent=TEAL, tsize=16.5, bsize=12.6, lh=1.52,
           foot="⇒ 能训练，但要为 Fω 的保真度负责")
    s.band(0.78, 4.76, 11.77, 0.86, "关键：不替换线上 SONIC；训练时只替换不可微的 Isaac 状态传播。")
    s.text(0.78, 5.86, 11.7, 0.94,
           [{"t": "推论：这套方法的收益上限由两个量决定 ——", "size": 12.6, "color": NAVY, "b": True},
            {"t": "① Fω 的准确度（含“对动作是否敏感”）；② ∂a/∂token（WBC 到底看不看 token）。"
                  "后者是本项目最后撞上的墙（第 13 页）。", "size": 12.4, "color": GRAY, "lh": 1.34}])
    s.footer(4)
    return s


def s05_data() -> Slide:
    s = Slide()
    s.header("03 / A′ 数据与世界模型", "A′ 数据集构成：扰动回放为主体 + 少量 VLA 闭环")
    # --- 两个来源 ---
    s.card(0.78, 1.70, 6.55, 2.40, "扰动回放：WBC-only 在 Isaac 里跑", accent=TEAL,
           tag="主体 ｜ 529,131 条 / 50 shards", tsize=15.6, bsize=11.2, lh=1.34,
           body="参考库 6 任务 clips，随机取窗口（每 shard 15 段）\n"
                "只执行 q_target、不接 VLA；动作加 σ=0.15 噪声激励\n"
                "漂移 >0.15 m 复位；50 Hz 逐 tick 记 (s_t, a_t, s_{t+1})",
           foot="覆盖广、可复现；但只有“执行器在参考轨迹附近”的分布")
    s.card(7.55, 1.70, 5.00, 2.40, "VLA 闭环推理落下的 tick", accent=AMBER,
           tag="补充 ｜ 少量", tsize=15.6, bsize=11.2, lh=1.34,
           body="策略在 Isaac 闭环里跑时的逐 tick 记录\n"
                "7,552 tick → 366 段 → 5,040 行（≈ A′ 的 1%）\n"
                "OpenDoor × A_delta，12 个进程",
           foot="补部署分布：|a| 0.29→0.81，每 tick 只走 2.58 mm")
    # --- 为什么这么配 ---
    s.text(0.78, 4.22, 8.0, 0.26, [{"t": "为什么这么配（使用原因）", "size": 11.4, "color": TEAL, "b": True, "spc": 60}])
    reasons = [
        ("先要覆盖与可复现", "参考库每段动作都能回放，起点/窗口可枚举，重跑一致。"),
        ("闭环补的是分布差", "发大指令、机器人跟不上 —— 这种动力学回放里一条都没有。"),
        ("只当少量增量", "只有 1 任务 1 策略，只能当增量；量大了分布会被它主导。"),
    ]
    for i, (t, b) in enumerate(reasons):
        x = 0.78 + i * 4.01
        s.rect(x, 4.52, 3.75, 1.14, fill=WHITE, line=BORDER, lw=0.8, radius=0.05)
        s.rect(x, 4.52, 0.05, 1.14, fill=BLUE_M, line=BLUE_M, lw=0.4)
        s.text(x + 0.24, 4.60, 3.36, 0.26, [{"t": t, "size": 12.2, "color": NAVY, "b": True}])
        s.text(x + 0.24, 4.90, 3.36, 0.68, [{"t": b, "size": 10.4, "color": GRAY, "lh": 1.28}])
    # --- 用法 ---
    s.rect(0.78, 5.82, 11.77, 0.62, fill=FILL_SOFT, line=FILL_SOFT, radius=0.04)
    s.rect(0.78, 5.82, 0.05, 0.62, fill=TEAL, line=TEAL, lw=0.4)
    s.text(1.02, 5.82, 11.3, 0.62,
           [{"t": "用法：A′（回放）为主体，混入少量闭环（≈1%）—— 量小，采样时上采样到 batch 内约一半，免得被淹没",
             "size": 11.4, "color": NAVY}], anchor="ctr")
    s.text(0.78, 6.56, 11.7, 0.34,
           [{"t": "一条转移 = states / next_states (N,71) = [root_pos 3 | quat 4 | q 29 | v 3 | ω 3 | qd 29] ｜ "
                  "actions (N,29) = (q_cmd − dq)/sc ｜ 奖励事后按 reward_cfg 补算",
             "size": 9.8, "color": GRAY, "latin": MONO}])
    s.footer(5)
    return s


def s06_wm_train() -> Slide:
    s = Slide()
    s.header("03 / A′ 数据与世界模型", "WM 怎么训：双数据、两阶段、三项 loss")
    rows = [["项", "值"],
            ["数据", "A′ 92,785 个 20 步窗口 ｜ 闭环 820,384 条转移"],
            ["stage1（1000 步）", "单步：A′ 70% + 闭环 30%"],
            ["stage2（2000 步，k=20）", "多步：A′ 70% + 闭环窗口 30%"],
            ["loss", "ssm + 2.0·inc + 2.0·gain"],
            ["xi", "default_xi（摩擦 0.75）全程"],
            ["存盘", "每 500 步一个 ckpt"]]
    s.table(0.78, 1.68, 6.80, [0.31, 0.69], rows, rh=0.42, size=9.8, hsize=9.8,
            aligns=["l", "l"], bold_first=True)
    notes = [
        ("为什么两个阶段都混", "单步先学“这一步准”，多步再学“20 帧不漂”；闭环只占 30%，防止过拟合到单一分布。"),
        ("为什么 stage2 用 k=20", "部署一个执行窗口就是 20 帧；多步监督让 10 步误差 0.104 → 0.093 rad（−10.8%）。"),
        ("为什么 xi 用 default_xi", "数据是在摩擦 0.75 下采的，训练与评测必须同口径，否则误差不可比。"),
    ]
    for k, (t, d) in enumerate(notes):
        y = 4.74 + k * 0.62
        s.text(0.78, y, 6.80, 0.22, [{"t": t, "size": 9.8, "color": TEAL, "b": True}])
        s.text(0.78, y + 0.24, 6.80, 0.40, [{"t": d, "size": 9.0, "color": GRAY, "lh": 1.20}])
    # 右侧三项 loss：公式 + 为什么
    cards = [
        ("ssm（主项，权重 1）", "L_ssm = mean[ ((ŝ′ − s′) / σ)² ]，σ = max(std, ε)",
         "71 维状态按各自 σ 归一后一起优化；不归一化时位置（0.5 m 量级）会把关节、速度全部淹没。", TEAL),
        ("inc（×2.0，动作可归因）", "L_inc = Σ_b ‖(Δŝ′ − Δs′) / std(Δs′)‖²，Δ = s′ − s",
         "动作对一步位置的贡献只有毫米级，按 0.5 m 归一后在损失里只剩 2.6e−6 —— 等于没训；换成“一步增量”的尺度才看得见，而增量里只有动作能解释。", BLUE_M),
        ("gain（×2.0，配对探针）", "L_gain = mean_b [ (Δ_pred − Δ_true)² / std_b² ]\nΔ_pred = ŝ′(s,a_k) − ŝ′(s,a₁)，Δ_true 同式",
         "A′ 里“同状态、不同动作”的配对只有 2/60000，∂s′/∂a 完全没被约束：自由优化能把误差降 89%，跨模型只转移 8%（伪方向）。配对探针直接观测动作差，把它钉住。", AMBER),
    ]
    for k, (t, f, w, c) in enumerate(cards):
        y = 1.68 + k * 1.56
        s.rect(7.85, y, 4.70, 1.46, fill=WHITE, line=BORDER, lw=0.8, radius=0.05)
        s.rect(7.85, y, 0.05, 1.46, fill=c, line=c, lw=0.4)
        s.text(8.06, y + 0.06, 4.30, 0.22, [{"t": t, "size": 10.0, "color": c, "b": True}])
        s.text(8.06, y + 0.30, 4.32, 0.34, [{"t": f, "size": 8.8, "color": SLATE, "latin": MONO, "lh": 1.16}])
        s.text(8.06, y + 0.68, 4.32, 0.74, [{"t": w, "size": 8.6, "color": GRAY, "lh": 1.20}])
    s.footer(6)
    return s


def s07_wm_verify() -> Slide:
    s = Slide()
    s.header("03 / A′ 数据与世界模型", "WM 训练效果：先看误差，再看方向")
    rows = [["判据", "含义", "阈值", "最佳 WM"],
            ["20 步端点误差", "想象终点 vs 真实终点（喂录制动作）", "报告", "**0.89 cm**"],
            ["20 步 root 漂移", "20 帧后 root 与参考的累计偏差", "≤1.05× base", "**4.48 cm**"],
            ["dir_agree", "WM 的梯度方向被真物理认可的比例", "≥0.90", "**0.979**"],
            ["corr_dJ", "WM 预测的 ΔJ 与真物理 ΔJ 的相关", "≥0.28", "**+0.339**"]]
    rows = [[c.replace("**", "") if isinstance(c, str) else c for c in r] for r in rows]
    s.table(0.78, 1.68, 7.55, [0.21, 0.40, 0.17, 0.22], rows, rh=0.56, size=9.8, hsize=9.6,
            aligns=["l", "l", "c", "c"], bold_first=True)
    s.text(0.78, 4.62, 7.55, 0.44,
           [{"t": "同一版 WM 的其它口径：ratio 0.98–0.99、单步方向余弦 0.97–0.98、同号率 98–99%。",
             "size": 9.4, "color": GRAY, "lh": 1.22},
            {"t": "BPTT 只走 10 步（H=20 时方向崩到 0.46，见右下表）。", "size": 9.4, "color": GRAY, "lh": 1.22}])
    # 右：dir_agree vs H
    s.text(8.60, 1.62, 3.95, 0.26, [{"t": "dir_agree 随视界的变化", "size": 10.0, "color": TEAL, "b": True}])
    hrows = [["H", "dir_agree"], ["2", "0.927"], ["5", "0.990"], ["10", "0.979"],
             [{"t": "20", "b": True, "color": RED}, {"t": "0.458", "b": True, "color": RED}]]
    s.table(8.60, 1.88, 3.95, [0.35, 0.65], hrows, rh=0.46, size=10.4, hsize=9.8,
            aligns=["c", "c"])
    s.text(8.60, 4.28, 3.98, 0.80,
           [{"t": "H≤10 方向可信（0.97–0.99），H=20 掉到 0.46 —— 等于抛硬币。", "size": 9.2, "color": NAVY, "lh": 1.22},
            {"t": "所以监督覆盖 20 步，反传只走 10 步。", "size": 9.2, "color": GRAY, "lh": 1.22}])
    s.rect(0.78, 5.28, 11.77, 0.74, fill=WARM, line=WARM, radius=0.04)
    s.rect(0.78, 5.28, 0.05, 0.74, fill=AMBER, line=AMBER, lw=0.4)
    s.text(1.02, 5.28, 11.3, 0.74,
           [{"t": "选型只看 dir_agree：单步误差小可能是“拟合得好、方向不对”，而反传吃的就是方向。",
             "size": 10.2, "color": RED}], anchor="ctr")
    s.rect(0.78, 6.12, 11.77, 0.70, fill=FILL_SOFT, line=FILL_SOFT, radius=0.04)
    s.rect(0.78, 6.12, 0.05, 0.70, fill=TEAL, line=TEAL, lw=0.4)
    s.text(1.02, 6.12, 11.3, 0.70,
           [{"t": "两道门：G1 方向 = dir_agree ≥0.90 且 corr_dJ ≥0.28；G2 精度 = 录制 ≤0.20、闭环 ≤0.30。",
             "size": 9.6, "color": NAVY}], anchor="ctr")
    s.footer(7)
    return s
