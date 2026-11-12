# -*- coding: utf-8 -*-
"""第 7–10 页：目标函数（两页）/ 反传链 / 训练流水线。"""
from pptxlib import (Slide, SLIDE_W, NAVY, TEAL, GRAY, BORDER, FILL_SOFT, FILL_FAINT,
                     SLATE, AMBER, RED, WARM, WHITE, CYAN_L, CYAN_M, BLUE_M, BLUE_D)

MONO = "DejaVu Sans"


def _fbox(s, x, y, w, h, fill=FILL_FAINT, accent=TEAL):
    s.rect(x, y, w, h, fill=fill, line=fill, lw=0.6, radius=0.04)
    s.rect(x, y, 0.05, h, fill=accent, line=accent, lw=0.4)


def s07_cost1() -> Slide:
    s = Slide()
    s.header("04 / 核心方法", "设计的目标函数（一）：每一步的代价")
    _fbox(s, 0.78, 1.66, 11.77, 1.22)
    s.text(1.02, 1.72, 11.35, 0.40,
           [{"t": "r_t  =  Δt · Σ_k  w_k · exp( −e_k / σ_k² )   +   Δt · ( w_a · ‖Δa‖²  +  w_j · Σ_j ℓ_j(q_j) )",
             "size": 14.4, "color": NAVY, "b": True, "latin": MONO}], anchor="ctr")
    s.text(1.02, 2.12, 11.35, 0.30,
           [{"t": "① 跟踪项：6 个通道各一个 0–1 分（见下表）      ② 惩罚项：动作变化率 + 软限位越界",
             "size": 10.6, "color": TEAL, "b": True}], anchor="ctr")
    s.text(1.02, 2.46, 11.35, 0.32,
           [{"t": "Δt = 0.02 s（50 Hz）   单步满额 0.10；权重与 σ 直接取自执行器 ckpt 的 reward_cfg，一个数都没改",
             "size": 10.4, "color": GRAY}], anchor="ctr")

    rows = [["项 k", "权重 w_k", "尺度 σ_k", "误差 e_k 的物理含义"],
            ["body_pos", "1.0", "0.30 m", "14 个 tracked body 的位置平方误差（参考按机器人 xy + yaw 重锚定）"],
            ["body_ori", "1.0", "0.40", "各 body 朝向的平方测地角误差"],
            ["body_linvel", "1.0", "1.00 m/s", "各 body 线速度平方误差"],
            ["body_angvel", "1.0", "3.14 rad/s", "各 body 角速度平方误差"],
            ["root_pos", "0.5", "0.30 m", "骨盆位置平方误差（对应“走没走到”）"],
            ["root_ori", "0.5", "0.40", "骨盆朝向平方测地角误差（对应“朝向对不对”）"],
            ["action_rate", "−0.1", "—", "惩罚①：‖Δa‖² 动作变化率（在 q_target 口径上算）"],
            ["joint_limit", "−2.0", "—", "惩罚②：软限位（0.9×硬限位）越界量 Σ_j relu(q − hi) + relu(lo − q)"]]
    s.table(0.78, 2.96, 11.77, [0.155, 0.085, 0.115, 0.645], rows, rh=0.262,
            size=10.8, hsize=10.4, aligns=["l", "c", "c", "l"], bold_first=True)

    why = [
        ("为什么用 exp 核", "误差映射到 (0,1] 的有界分，6 个通道才能加权相加；σ 定“多小算好”。这也是 SuperMimic 的原式。"),
        ("为什么和 WBC 同口径", "这个奖励既是执行器训练的目标，也是“执行对不对”的定义。换一套 = 优化一个部署时没人认的指标。"),
        ("为什么不能直接反传", "奖励要看 s_{t+1}，而 ∂s_{t+1}/∂a 只有可微 WM 给得出；Isaac 的刚体 + 接触求解没有梯度。"),
    ]
    for i, (t, b) in enumerate(why):
        x = 0.78 + i * 4.01
        s.rect(x, 5.40, 3.75, 1.10, fill=WHITE, line=BORDER, lw=0.8, radius=0.05)
        s.rect(x, 5.40, 0.05, 1.10, fill=TEAL, line=TEAL, lw=0.4)
        s.text(x + 0.22, 5.46, 3.40, 0.24, [{"t": t, "size": 11.6, "color": NAVY, "b": True}])
        s.text(x + 0.22, 5.72, 3.40, 0.70, [{"t": b, "size": 9.6, "color": GRAY, "lh": 1.24}])
    s.text(0.78, 6.60, 11.7, 0.32,
           [{"t": "同一份实现：训练与评测都走 reward_cfg.py + mdp.reward()，所以“优化的”和“量的”是同一个函数。",
             "size": 10.2, "color": GRAY}])
    s.footer(8)
    return s


def s08_cost2() -> Slide:
    s = Slide()
    s.header("04 / 核心方法", "设计的目标函数（二）：目标、约束与反传")
    # ① 目标
    _fbox(s, 0.78, 1.66, 11.77, 1.06)
    s.text(1.02, 1.70, 8.4, 0.26, [{"t": "① 目标：10 步跟踪分的加权和 J（越大越好）", "size": 11.8, "color": TEAL, "b": True}])
    s.text(1.02, 1.98, 11.30, 0.32,
           [{"t": "J(θ)  =  Σ_{t=1..H}  γ^(t−1) · s_t · r_t        H = 10（0.2 s）   γ = 0.99   s_t = WM survival",
             "size": 12.6, "color": NAVY, "b": True, "latin": MONO}], anchor="ctr")
    s.text(1.02, 2.32, 11.30, 0.30,
           [{"t": "γ 是权重衰减（γ^0=1 → γ^9=0.91），不是 RL 的折扣回报；H 只取 10（第 7 页：H=20 时方向掉到 0.46）",
             "size": 9.8, "color": GRAY}], anchor="ctr")
    # ② 三个约束
    _fbox(s, 0.78, 2.80, 11.77, 1.94, accent=AMBER)
    s.text(1.02, 2.84, 8.4, 0.26, [{"t": "② 三个约束：防止 J 靠“作弊”刷高", "size": 11.8, "color": AMBER, "b": True}])
    s.text(1.02, 3.10, 11.30, 0.24,
           [{"t": "全部在 q_raw = q_default + scale ⊙ a 上算（G1 正确物理口径）",
             "size": 9.6, "color": GRAY, "latin": MONO}], anchor="ctr")
    rows = [
        ("L_lim", "½·max_{t,j} v²  +  ½·mean_t max_j v² ,  v = relu(lo − q)/s + relu(q − hi)/s",
         "关节别顶到限位；按最坏关节算"),
        ("L_same", "relu( RMS(q − q_pristine) / τ − 1 )² ,  τ = 0.02 rad",
         "整体偏离原模型太远才罚（信任域）"),
        ("L_peak", "max_{t,j} relu( |e_tj| / b_j − 1 )²",
         "别让单个关节被改飞（峰值）"),
    ]
    for k, (name, formula, gloss) in enumerate(rows):
        y = 3.38 + k * 0.42
        s.text(1.02, y, 0.95, 0.36, [{"t": name, "size": 10.6, "color": AMBER, "b": True, "latin": MONO}], anchor="ctr")
        s.text(2.05, y, 6.85, 0.36, [{"t": formula, "size": 9.4, "color": SLATE, "latin": MONO}], anchor="ctr")
        s.text(9.05, y, 3.30, 0.36, [{"t": gloss, "size": 9.2, "color": GRAY, "lh": 1.18}], anchor="ctr")
    # ③ 总损失（critic 显式进来）
    _fbox(s, 0.78, 4.86, 11.77, 1.56, accent=BLUE_M)
    s.text(1.02, 4.90, 10.0, 0.24,
           [{"t": "③ 总损失：J 加上 critic 的尾巴，再减罚项", "size": 11.8, "color": BLUE_M, "b": True}])
    s.text(1.02, 5.14, 11.32, 0.28,
           [{"t": "J_ac = J + β · γ^H · s_H · ( V(s_H) − V_ref )        ← critic 只在 10 步的末端出现（β = 0.975，20% 梯度预算）",
             "size": 10.4, "color": RED, "b": True, "latin": MONO}], anchor="ctr")
    s.text(1.02, 5.44, 11.32, 0.28,
           [{"t": "L(θ) = − J_ac + λ_lim·L_lim + λ_T·L_same^tot + λ_P·L_peak^tot",
             "size": 11.0, "color": NAVY, "b": True, "latin": MONO}], anchor="ctr")
    s.text(1.02, 5.74, 11.32, 0.28,
           [{"t": "∂L/∂θ = − Σ_t γ^(t−1) s_t (∂r_t/∂s_{t+1})(∂s_{t+1}/∂q_t)(∂q_t/∂z_t)(∂z_t/∂θ) − βγ^H s_H ∂V/∂θ + Σ_k λ_k ∂C_k/∂θ",
             "size": 8.8, "color": TEAL, "b": True, "latin": MONO}], anchor="ctr")
    s.text(1.02, 6.06, 11.32, 0.30,
           [{"t": "三段雅可比都由可微模块给出（WM / SONIC / 动作头）—— 没有采样、没有 logπ；V 的 token 输入被 detach，梯度只从 s_H 走。",
             "size": 9.0, "color": GRAY}], anchor="ctr")
    s.rect(0.78, 6.52, 11.77, 0.44, fill=WARM, line=WARM, radius=0.04)
    s.rect(0.78, 6.52, 0.05, 0.44, fill=RED, line=RED, lw=0.4)
    s.text(1.02, 6.52, 11.30, 0.44,
           [{"t": "λ 按“约束梯度占 J ≤40%”标定；β 有 20% 梯度预算，防止价值项压过 J。",
             "size": 9.6, "color": RED}], anchor="ctr")
    s.footer(9)
    return s


def s09_chain() -> Slide:
    s = Slide()
    s.header("04 / 核心方法", "可微反馈链：实际反传的那一条")
    boxes = [("VLA 动作头", "Δθ：LoRA16\n+ proj_out"),
             ("4 步可微 FM", "pathwise\n梯度穿过采样"),
             ("token z_t (64D)", "VLA 动作头输出\n逐帧 64 维"),
             ("Torch SONIC", "994D → 29D\n可微 WBC"),
             ("A′ WM (GRU)", "Fω 冻结\ns_{t+1} 可微"),
             ("r_t → J", "H = 10\nγ = 0.99")]
    n = len(boxes)
    gap = 0.18
    cw = (11.77 - gap * (n - 1)) / n
    for i, (t, b) in enumerate(boxes):
        x = 0.78 + i * (cw + gap)
        s.rect(x, 1.84, cw, 1.24, fill=WHITE, line=BORDER, lw=0.9, radius=0.05)
        s.text(x + 0.10, 1.94, cw - 0.20, 0.28, [{"t": t, "size": 12.2, "color": NAVY, "b": True}], anchor="ctr")
        s.text(x + 0.10, 2.26, cw - 0.20, 0.74,
               [{"t": b, "size": 9.8, "color": GRAY, "lh": 1.30}], anchor="t")
        if i < n - 1:
            s.chevron(x + cw + gap / 2 - 0.11, 2.31, 0.22, 0.30)
    s.rect(0.78, 3.30, 11.77, 0.52, fill=FILL_SOFT, line=FILL_SOFT, radius=0.03)
    s.text(1.00, 3.30, 11.4, 0.52,
           [{"t": "反向：∂L/∂θ  ←  ∂L/∂s  ←  ∂Fω/∂q  ←  ∂G/∂z′  ←  ∂z′/∂θ       （除 critic 的 token 输入外，全程不 detach）",
             "size": 11.8, "color": TEAL, "b": True, "latin": MONO}], anchor="ctr")
    notes = [("主干 no_grad", "backbone / VLM 前向不进图，只回传到 action_head：16.89 M / 1.637 B = 1.03%"),
             ("裁判全程冻结", "裁判权重钉死（sha 19163963，fixed 模式），不做 on-policy 刷新"),
             ("critic 输入 detach", "价值梯度不许绕过 WM 直接对 token 求导（评审 03 的硬约束）")]
    for i, (t, b) in enumerate(notes):
        x = 0.78 + i * 4.01
        s.rect(x, 4.20, 3.75, 1.16, fill=WHITE, line=BORDER, lw=0.8, radius=0.05)
        s.text(x + 0.24, 4.30, 3.3, 0.28, [{"t": t, "size": 12.2, "color": NAVY, "b": True}])
        s.text(x + 0.24, 4.62, 3.30, 0.68, [{"t": b, "size": 10.6, "color": GRAY, "lh": 1.28}])
    s.rect(0.78, 5.62, 11.77, 1.12, fill=WARM, line=WARM, radius=0.04)
    s.rect(0.78, 5.62, 0.06, 1.12, fill=RED, line=RED, lw=0.4)
    s.text(1.04, 5.66, 11.3, 1.04,
           [{"t": "这条链是“训练专用的旁路”，真实 Isaac 仍是独立验收终点。", "size": 12.6, "color": RED, "b": True},
            {"t": "J ↑ 不等于闭环成功率 ↑ —— 本次就撞上了：dJ 为正，但 token 到动作的通道几乎不通（第 13 页）。",
             "size": 11.6, "color": SLATE}], anchor="ctr")
    s.footer(10)
    return s


def s10_starts() -> Slide:
    s = Slide()
    s.header("04 / 核心方法", "联合微调（三）：训练流程与起点选择")
    # ── 左：五步（只留关键动作）
    s.text(0.78, 1.60, 6.6, 0.26, [{"t": "一次训练的五步", "size": 11.2, "color": TEAL, "b": True, "spc": 40}])
    steps = [("1", "取对齐的起点", "起点与参考偏差 ≤ 0.10 m"),
             ("2", "VLA 出 token", "LoRA + 4 步可微 FM"),
             ("3", "SONIC 出 q_target", "994D → 29D，可微"),
             ("4", "WM 想象 10 步", "冻结裁判做可微传播"),
             ("5", "反传更新 LoRA", "J_ac + 三约束 → 链式梯度")]
    for k, (n, t, d) in enumerate(steps):
        y = 1.92 + k * 0.56
        s.rect(0.78, y, 6.55, 0.48, fill=WHITE, line=BORDER, lw=0.7, radius=0.04)
        s.text(0.86, y + 0.02, 0.30, 0.24, [{"t": n, "size": 10.0, "color": TEAL, "b": True}], anchor="ctr")
        s.text(1.22, y + 0.02, 1.75, 0.24, [{"t": t, "size": 10.2, "color": NAVY, "b": True}], anchor="ctr")
        s.text(3.05, y + 0.02, 4.15, 0.24, [{"t": d, "size": 9.4, "color": GRAY}], anchor="ctr")
    s.text(0.78, 4.74, 6.55, 0.22,
           [{"t": "critic 只在第 5 步：β = 0.975，梯度预算 20%", "size": 8.8, "color": GRAY}])
    # ── 右：起点为什么不取闭环轨迹（三条）
    s.rect(7.62, 1.60, 4.93, 2.96, fill=WARM, line=WARM, radius=0.04)
    s.rect(7.62, 1.60, 0.05, 2.96, fill=RED, line=RED, lw=0.4)
    s.text(7.85, 1.68, 4.55, 0.24, [{"t": "起点为什么不取闭环轨迹", "size": 11.4, "color": RED, "b": True}])
    s.text(7.85, 1.98, 4.52, 1.90,
           [{"t": "① 闭环状态可能已漂 > 0.10 m，对应的参考未必是同一条", "size": 9.8, "color": SLATE, "lh": 1.30},
            {"t": "② 从参考库取点：每段起点都对齐，漂移逐段压住", "size": 9.8, "color": SLATE, "lh": 1.30, "sb": 8},
            {"t": "③ 也不能用 VLA 自己的输出：等于自己教自己", "size": 9.8, "color": SLATE, "lh": 1.30, "sb": 8}])
    s.text(7.85, 4.02, 4.52, 0.44,
           [{"t": "一句话：起点不是制造难度，是保证参考对得上。", "size": 9.2, "color": RED, "b": True, "lh": 1.2}])
    # ── 数字（去掉超参）
    xs = [0.78, 3.76, 6.74, 9.72]
    vals = [("634,016", "参考库帧数"), ("346,729", "≤ 0.10 m"),
            ("238,175", "可用起点"), ("375", "train clips")]
    for i, (v, l) in enumerate(vals):
        s.text(xs[i], 5.02, 2.85, 0.38, [{"t": v, "size": 17.5, "color": TEAL, "b": True}], anchor="ctr")
        s.text(xs[i], 5.40, 2.85, 0.22, [{"t": l, "size": 9.2, "color": GRAY}])
    # ── H=10（两行）
    s.rect(0.78, 5.66, 11.77, 1.22, fill=WARM, line=WARM, radius=0.04)
    s.rect(0.78, 5.66, 0.05, 1.22, fill=RED, line=RED, lw=0.4)
    s.text(1.04, 5.66, 11.3, 1.22,
           [{"t": "为什么 H=10：dir_agree 在 H≤10 是 0.97–0.99，H=20 只有 0.46–0.50（≈抛硬币）—— 反传吃的就是方向，所以只能看 0.2 s。",
             "size": 10.8, "color": RED, "b": True, "lh": 1.24},
            {"t": "看不到的那段回报，就用 critic 的 V(s_10) 补：J_ac = J + βγ^H(V(s_H) − V_ref)。",
             "size": 9.8, "color": SLATE, "lh": 1.24},
            {"t": "critic 的作用：把视界从 0.2 s 往外延，不是替代 J。",
             "size": 9.8, "color": RED, "b": True, "lh": 1.24}], anchor="ctr")
    s.footer(11)
    return s


def s11_pipeline() -> Slide:
    s = Slide()
    s.header("04 / 核心方法", "训练流水线：10 个阶段，一条命令")

    def group(x, w, title, color):
        s.text(x, 1.68, w, 0.28, [{"t": title, "size": 11.4, "color": color, "b": True, "spc": 40}])
        s.hline(x, 1.96, w, color=BORDER, lw=0.7)

    def stage(x, y, w, n, name, desc, h=0.72):
        s.rect(x, y, w, h, fill=WHITE, line=BORDER, lw=0.7, radius=0.04)
        s.text(x + 0.08, y + 0.06, 0.36, 0.30,
               [{"t": str(n), "size": 11.0, "color": TEAL, "b": True}], anchor="ctr", align="c")
        s.text(x + 0.50, y + 0.06, w - 0.60, 0.28,
               [{"t": name, "size": 10.6, "color": NAVY, "b": True}])
        s.text(x + 0.50, y + 0.34, w - 0.60, h - 0.40,
               [{"t": desc, "size": 9.0, "color": GRAY, "lh": 1.20}])

    group(0.78, 4.10, "数据（1–4）", TEAL)
    stage(0.78, 2.04, 4.10, 1, "a_collect", "冻结 WBC 在 Isaac 真物理里跑：512 env × 300 步 × 4 波")
    stage(0.78, 2.80, 4.10, 2, "collect", "闭环采集：每任务 N 集，两卡并行")
    stage(0.78, 3.56, 4.10, 3, "score", "全池补 rewards：bundle 只加载一次")
    stage(0.78, 4.32, 4.10, 4, "transitions", "建真转移（含 B 线 extra）")

    group(4.98, 3.85, "裁判（5–7）", AMBER)
    stage(4.98, 2.04, 3.85, 5, "wm_refresh", "三段：A 线单步 → A+B 混合 → k 步自回归")
    stage(4.98, 2.80, 3.85, 6, "wm_gate", "留出任务 20 步关节误差 ≤ 1.05× base；不过就回退")
    stage(4.98, 3.56, 3.85, 7, "critic_gate", "拟合 + v_ev > 0.30 才允许开（可选零件）")

    group(8.93, 3.62, "策略（8–10）", BLUE_M)
    stage(8.93, 2.04, 3.62, 8, "joint", "联合微调：本轮 WM 当裁判（本次结果在这一步）")
    stage(8.93, 2.80, 3.62, 9, "eval", "6-seed 配对闭环评测（同会话、同起点）")
    stage(8.93, 3.56, 3.62, 10, "report", "汇总表 + 漂移图")

    s.rect(4.98, 4.32, 7.57, 0.72, fill=WARM, line=WARM, radius=0.04)
    s.rect(4.98, 4.32, 0.05, 0.72, fill=RED, line=RED, lw=0.4)
    s.text(5.20, 4.36, 7.20, 0.64,
           [{"t": "门禁是硬约束：任一阶段产物对不上本次 ROUND、或 WM 门不过，脚本直接停 / 回退，不允许“跳过继续”。",
             "size": 9.6, "color": SLATE, "lh": 1.22}], anchor="ctr")

    s.rect(0.78, 5.24, 11.77, 0.90, fill=FILL_SOFT, line=FILL_SOFT, radius=0.04)
    s.rect(0.78, 5.24, 0.05, 0.90, fill=TEAL, line=TEAL, lw=0.4)
    s.text(1.02, 5.28, 11.3, 0.82,
           [{"t": "一条命令跑一轮：ROUND=<本轮> bash scripts/s3_pipeline.sh all（也可单跑某段，仍校验前置）",
             "size": 10.6, "color": NAVY, "b": True, "latin": MONO, "lh": 1.24},
            {"t": "每个阶段的产物、时间戳、关键参数与断言都落盘 runs/S3_pipeline/<ROUND>/state.json",
             "size": 9.6, "color": GRAY, "lh": 1.24}], anchor="ctr")
    s.text(0.78, 6.34, 11.7, 0.52,
           [{"t": "这条流水线就是为了杜绝“漏步骤 / 用了上一轮 WM / 没打分就建转移”：joint 前 WBC_CKPT 必须等于本轮 WM，"
                  "eval 前必须存在 joint 产物。",
             "size": 10.2, "color": GRAY, "lh": 1.24}])
    s.footer(12)
    return s
