# COMIMIC A/B 最终交底：方向性坍缩 —— 离线 WM ΔJ 与真实闭环的背离

> 2026-11-08 12:05 UTC　｜　`comimic-wt03`，分支 `harness/03-bptt-diagnosis`
> 目的：把**当前系统的架构、四代设计演进、全部实测结果、以及核心失败的机理**讲清，
> 交给 reviewer 产出可执行的修改方案。
> 约定：只写跑过/验证过的；每个数字标出处；没跑的标 `NOT_RUN`。

---

## 0. 一页结论

**核心事实（本轮最重要的发现）**：

| | 离线 test ΔJ | 同观测"方向性" | **SitSofa 闭环** | OpenDoor 闭环 |
|---|---|---|---|---|
| **base**（无 Δ） | — | — | **4/6** | `NOT_RUN` |
| **A 原版** | +0.02185 | **0.81** | **4/6** | 3/6 |
| **A_v8**（V8 配方） | **+0.02577（最好）** | **0.51（最差）** | **0/6** ⚠️ | 4/6 |

> **离线最好的 ckpt，在 SitSofa 上一次都没成功；而 base 和 A 原版都是 4/6。**
> 若 A_v8 与 base 同强（2/3），连续 6 次全败的概率是 `(1/3)^6 ≈ 0.14%`。

**机理（已用数据定位）**：A_v8 的 Δ 发生了 **"方向性坍缩"**——
所有任务上 `|带符号均值|/rms` 从 **0.81 → 0.51**，而 `rms` 反而 **+5~11%**。
即：**幅度没变小，但"往哪推"的主见没了（正负互相抵消）**。
我们推断原因是 **worst-case 优化（Recovery 的最差 10% 窗口项）+ 30% 困难采样**：
在**异质**的困难状态上做最坏情况优化，数学上的最优解就是 **"别表态"（对冲）**。

**这解释了为什么离线涨、闭环跌**：WM 的 ΔJ 是 96 窗口的平均，一个"对称、无偏"的修正
像正则项一样能把平均误差压下去；但真实任务（尤其接触型 SitSofa）需要的是**持续一致的方向**——
0.4° 的方向差 × 每 tick × 1500 tick = 累积 10 rad 量级的差异。

---

## 1. 系统架构

### 1.1 闭环链路

```
Isaac Sim（HumanoidArena：OpenDoor / SitSofa / VisionNavi / Boxing）
   │ 相机 + 本体状态
   ▼
冻结 VLA：FT8000（Gr00tN1d7，1.63 G 参数，6.9 GB，全冻结，官方微调产物）
   │ 64 维 motion token chunk（T=40）
   ▼
【A/B】Δ 网（1.744 M，逐帧残差，只在前 K=20 帧加）
【C】  LoRA16(216 层) + proj_out 全秩（23.07 M 可训）
   │ token
   ▼
SONIC decoder（ONNX，发布版，冻死）
   │ 29 维归一化动作
   ▼
q_raw = q_default(HA_ALIGN) + G1_ACTION_SCALE ⊙ a
   ▼
限位/速率安全层 → q_applied → Isaac PD
```

- 50 Hz 控制；VLA 每 20 步重规划；T=40 / K=20 / H=20 / FM=4。
- **部署契约（不可改）**：`state_hist_10` 形状/顺序（HA_V31）、`hist[-1]=当前状态`（tol 5e-3）、
  `cap` 语义、`t≥K` 严格 0、`G1_ACTION_SCALE`、SONIC-only。

### 1.2 Δ 网（`diff_vla_wbc/delta_token_temporal_v5.py`，全程未改）

| 输入 | 形状 | 说明 |
|---|---|---|
| `Z` | (B,40,64) | VLA 整段 token |
| `state64` | (B,64) | 决策时刻 `[rot6d(6) │ q(29) │ qd(29)]`，HA_V31 |
| `hist` | (B,10,58) | 过去 10 tick 的 `[q(29) │ qd(29)]`，HA，最老→最新 |

结构：`token_proj(64→128)` + `pe_chunk(40,128)` + 1×TransformerEncoder(d128,h4,ffn256)
+ `hist_gru(58→128)` + `pe_pos(40,16)` + MLP 头 `400→1024→1024→64`（占 87% 参数）。
输出 `Δ = cap·tanh(‖u‖/cap)·u/‖u‖`，`cap=0.08`；`t≥20` 严格为 0。

### 1.3 世界模型与数据

| 件 | 路径 / 事实 |
|---|---|
| **WM0** | `runs/a_prime_sonic_exc/wm_gain2.pt`，sha `fc5aeb30` —— **全程唯一使用的 WM** |
| A′ replay | `all_transitions.pt`，529,131 行，旧策略真值 |
| 评测集 | 375 train / 81 val / 81 test clips；val/test 各 96 窗口；**训练只取 train clips** |
| 训练 motion 覆盖 | `HOI_double_desk 75 / HOI_pp_box 70 / HSI_boxing 64 / HSI_sit_sofa 57 / HSI_open_door 55 / HSI_vision_navi 54`（**四个闭环任务都在训练集里**） |
| 物理限位 | `limits_standard29.npz`（MuJoCo jnt_range 90% soft） |

---

## 2. 设计演进（四代配方，全部实测）

### 2.1 损失函数：当前实际生效的项（A/B）

```
J20 = Σ_{t<20} γ^t · survival_t · reward_t          γ=0.99，原 saturating tracking 奖励

loss = −mean J20
     + λ_lim   · L_lim(q_raw)                 限位（绝对值，不对称，max 形式）
     + λ_peak  · L_peak(e_phys, b_j)          单步物理修正的异常峰值
     + λ_added · L_added(q_raw, q_base)       "恶化"（含原本就违规的）
     + λ_R     · L_recovery(worst_window)     最差窗口风险聚合
     + λ_raw   · Hinge(‖u‖/(2cap))
     + λ_act   · Trust(e_phys, τ=0.020 rad)
     + λ_same  · L_same                        **已去重（强制 0）**
     + λ_rate  · L_rate                        **BLOCKED（无 velocity 字段）**
```

**关键**：在 A_v8 里，`L_act`（14 mrad < 20 mrad）与 `L_peak`（预算 0.22 rad）**都不激活**，
真正起作用的是 **`J20 + L_lim + L_R + L_raw`**。

### 2.2 四代配方对比

| 项 | V5/原版（A 原版） | **V8（A_v8）** | V8.1 | V8.2 |
|---|---|---|---|---|
| 初始化 | v5 best | **A 原版 best** | A 原版 best | A 原版 best |
| 步数 | 300 | **200** | 200（停在 110） | 200（停在 120） |
| lr (head/enc) | 5e-5 / 2e-5 | **3e-5 / 1e-5** | 同 | 同 |
| 困难采样 | 无 | **70/30** | 同 | 同 |
| Recovery | 无 | **mean 聚合，z0=Q0.90** | **worst_window**，z0=Q0.70 | 同 V8.1 |
| L_peak 预算 | 自指 3·mean\|e\| | base-rate 0.22 rad（**不激活**） | orig-q99 **0.045 rad** | 同，但 **top-k 聚合** |
| λ_peak | 8.4e-4 | 同（无效） | 0.00251 | **0.01895** |
| λ_R | 0 | 0.0019 | 0.00505 | 0.00505 |

### 2.3 物理约束的语义（本轮修好的部分）

```
q_raw  = q_default^HA_ALIGN + G1_ACTION_SCALE ⊙ a       # 真 rad
e_phys = q_raw(Δ) − q_raw(base)                          # **同观测**修正（部署时真实扰动）
L_lim   = mean[0.5·max_{t,j}v² + 0.5·mean_t max_j v²]，v = violation / 关节半量程
L_peak  = Top-k over (t,j) of relu(|e_phys|/b_j − 1)²    （V8.2 才改成 top-k；V8/V8.1 是 max）
b_j     = clip(1.25·Q0.99(|e_phys,j| of original best), 0.02, 0.12 rad)
L_R     = Σ_k w_k [0.5·mean + 0.5·TopMean_{10% window}(TopMean_{25% tick})]
```

---

## 3. 结果

### 3.1 离线（WM0，同一固定 96 窗口 / 独立 test）

| ckpt | val ΔJ | **test ΔJ** | 同观测 rms | 同观测 peak | root_ori p95 |
|---|---|---|---|---|---|
| A 原版 best | +0.02962 | +0.02185 | 14.5 mrad | 0.139 rad | 0.879 |
| **A_v8 best** | **+0.03291** | **+0.02577** | 15.2 | 0.143 | 0.873 |
| A_v8.2 best | +0.03226 | +0.02350 | 15.5 | 0.142 | 0.870 |
| B 原版 best | +0.02774 | +0.02074 | 14.5 | 0.131 | 0.905 |
| B_v8b best | +0.03110 | +0.02416 | 14.8 | 0.145 | 0.880 |
| B_v8.2 best | +0.03089 | +0.02325 | 15.2 | 0.148 | 0.892 |

**结论**：V8 是四代里离线最好的；V8.1/V8.2 加码后**离线都变差**，物理也没变好。
`root_ori p95` 在 **8 个 ckpt 上从未动过**（0.865–0.918）⇒ **"恢复尾部"这个指标用改损失的方式动不了**。

### 3.2 闭环（首次尝试、不重试、MAX_STEPS=3000、SONIC=1、hist10）

| 任务 | base（sr4, 11-06） | A 原版（11-08，重试口径） | **A_v8（今天，并行 2 卡）** |
|---|---|---|---|
| OpenDoor | `NOT_RUN` | 3/6 | **4/6** |
| **SitSofa** | **4/6** | **4/6** | **0/6** ⚠️ |
| VisionNavi | **3/6** | 2/6 | 进行中 |
| Boxing | `NOT_RUN` | `NOT_RUN` | 进行中 |

**A_v8 的 SitSofa 失败全是 `timeout`（跑满 3000 步）**，不是摔倒。

### 3.3 核心发现：Δ 的"方向性坍缩"

定义 **方向性** `= |带符号均值修正| / rms`（对右膝，同观测 e_phys，train clips）：
`1.0` = 每次都朝同一方向推；`0` = 正负对消、等价于噪声。

| 任务 | A 原版 | **A_v8** | rms 变化 |
|---|---|---|---|
| HOI_double_desk | 0.71 | **0.45** | 13.6 → 14.3 |
| HSI_vision_navi | 0.53 | **0.31** | 13.9 → 14.8 |
| HOI_pp_box | 0.76 | **0.55** | 15.3 → 16.9 |
| **HSI_sit_sofa** | **0.86** | **0.48** | 14.3 → 15.6 |
| HSI_open_door | 1.06 | **0.70** | 14.7 → 15.9 |
| HSI_boxing | 0.93 | **0.54** | 14.3 → 15.4 |
| **平均** | **0.81** | **0.51** | — |

**同时 `‖Δ‖`(token) 几乎没变（0.0777 → 0.0780，cap 顶死）。**

⇒ **幅度没变小、方向没了**。这就是"离线涨、闭环跌"的直接机制。

---

## 4. 问题清单

### P0 —— 直接导致结论错误

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| **P0-1** | **方向性坍缩**：Δ 从"有主见"变成"对冲"，而离线指标反而变好 | §3.3 | A_v8 在 SitSofa 0/6（base/A原 都 4/6） |
| **P0-2** | **WM ΔJ 与闭环反相关**：离线最好的 ckpt 闭环最差 | §3.1 vs §3.2 | 整个"用 WM 训练 Δ"的范式在这个任务上失效 |
| **P0-3** | **base 臂缺 OpenDoor / Boxing** | — | 无法回答"Δ 整体有没有用" |
| **P0-4** | **A 原版闭环是"重试取最好"口径**，不是首次成功率 | `RESULTS.md` | 与今天的首次尝试不可直接比 |
| **P0-5** | **单 seed**（A/B/C 各 1） | — | 无误差棒 |

### P1 —— 影响结论强度

| # | 问题 | 证据 |
|---|---|---|
| **P1-1** | **`root_ori` 尾部不可动**：8 个 ckpt 全在 0.865–0.918 | §3.1 |
| **P1-2** | **物理峰值随训练增长**：A_v8 训练中 `e_phys_absmax` 0.123→0.173；V8.2 压到 0.14–0.16 但离线掉 0.002 | `train.jsonl` |
| **P1-3** | **`L_act`/`L_peak` 在 V8 里不激活**（14 mrad<20；0.086<0.22 rad） | 同上 |
| **P1-4** | **`L_rate` BLOCKED**（`RobotSpec` 无 velocity）；安全层只做绝对限位、不做速率裁剪 | `calibration/delta.json` |
| **P1-5** | **`L_added` 语义错**：罚"恶化"(2.8%) 而非"新增"(0.45%)；已把 λ 设为 0 | `diag_delta_phys_v5` 分解 |
| **P1-6** | **条件化只验证到"不是常量"**（cosZ≈0.65，门禁 <0.9），无 reward 对齐证据 | `cond_check.jsonl` |
| **P1-7** | **A/B 初始化不可比**（A=v5 warm-start；B=随机） | 配置 |
| **P1-8** | **checkpoint 不含 optimizer/scheduler/RNG** ⇒ 无严格 resume | `_save_ckpt` |

### P2 —— 工程

| # | 问题 |
|---|---|
| **P2-1** | 闭环录像已关闭（磁盘）；逐 tick 物理日志仍未落盘 |
| **P2-2** | 共享 worktree 有并发 agent，已发生一次"WM 被切换导致 run 用错 WM" |
| **P2-3** | 训练日志里的 `*_mrad` 历史标签错误（实为归一化域），现已加 `act_rms_rad` 区分 |

### 4.1 本轮已修（不要再回滚）

| 修复 | 内容 |
|---|---|
| 物理单位 | `default_joint_pos` 用 bundle 口径（mjlab 0.669 → HA_ALIGN 0.420），消掉 10 关节最大 0.118 rad 常数偏置 |
| 信任域 | `L_act` 从归一化域改成**真 rad**（旧 τ=0.02 实际只有 7.9 mrad） |
| 去重 | `L_same` 与 `L_act` 是同一量，physical 域下强制 `λ_same=0` |
| Peak 预算 | 从"3·mean\|e\| 自指"改成 **从 original best 的同观测分布标定**（0.22 → 0.045 rad） |
| Peak 聚合 | 从 `max_{t,j}`（每样本 1 个 entitiy，会被"打地鼠"绕开）改成 **top-k** |
| 诊断落盘 | `_phys_log` 之前算了却没写进 `train.jsonl`，已修 |
| 越限统计 | 拆成严格分解 `mod = base − fixed + new`（恒等式残差 1e-9），此前"新增/修掉 2.8%/2.5%"混了口径，真实值是 **0.45%/0.37%** |

---

## 5. 机理分析：为什么会"方向性坍缩"（给 reviewer 的核心假设）

**观察**：A_v8 = A 原版 + 200 步，新增了 ① 30% 困难采样 ② worst-window Recovery。
两者都是**把训练分布推向"异质困难状态"**，而 ② 还是**最坏情况优化**（最差 10% 窗口占一半权重）。

**推论**：
- 不同困难状态需要的修正方向**互相矛盾**；
- 在最坏情况目标下，任何**明确方向**都会在某些窗口上变成"坏样本"；
- 于是网络学会把输出推向**零均值、对称分布** —— 即 **hedging**；
- 这恰好让"平均意义上的 WM ΔJ"变好（对称修正像正则项），
  但把真实任务最需要的**持续一致方向**抹掉了。

**可验证的预测**（建议 reviewer 确认或否证）：

1. `A 原版 + 200 步，只加 Recovery`（不加硬采样）⇒ 方向性应掉得更多；
2. `A 原版 + 200 步，只加硬采样`（不加 Recovery）⇒ 方向性应掉得较少；
3. 若把 Recovery 的 tail 项权重从 0.5 降到 0、或把 `top-10%` 改成 `mean` ⇒ 方向性应保持 ≥0.7；
4. 一个直接的补救：**在损失里显式加"方向一致性"项**（例如惩罚 `‖mean_t(e_phys)‖` 变小），
   或者干脆**去掉 worst-case 项**。

---

## 6. 给 reviewer 的问题

**Q1（最重要）· 方向性坍缩怎么修？**
请给出具体方案与公式：
(a) 彻底去掉 worst-case 聚合，回到 `mean`？（但 V8 用 mean 时 SitSofa 尚未验证）
(b) 保留但加权（tail 权重 0.5→0.2）？
(c) 显式加"方向一致性"正则（`−‖mean_t e_phys‖` 或 `‖e_phys − mean_t e_phys‖` 惩罚）？
(d) 换目标：直接监督"参考动作方向"（用 A′ 真值），而不是靠 WM 回报？
请说明**如何验证修好了**（除了闭环）。

**Q2 · 离线指标还值得信吗？**
WM ΔJ 在本轮与闭环**反相关**（SitSofa）。是否应该：
(a) 放弃 WM ΔJ 作为选型指标，改用**真实 A′ 短视界预测误差**；
(b) 保留但要求"与闭环同时报告"；
(c) 先做 WM Refresh（V7 五步）再谈。

**Q3 · 闭环协议怎么统一？**
现有三套口径混用：base(sr4, 11-06, 串行, seed=0)、A 原版(11-08, 重试取最好)、A_v8(今天, 首次尝试, 并行 2 卡)。
请给出**唯一标准协议**（实例数、FM seed、重试策略、每任务 seed 数、统计判据）。

**Q4 · 下一步算力怎么花？**（现有 2×48G GPU，磁盘 1.5 TB 空闲）
候选：
(a) 补 base 四任务（1.5 h）⇒ 回答"Δ 有没有用"；
(b) 方向性归因消融（2×200 步 + 2×6 集 SitSofa，~3 h）⇒ 回答"是哪个改动让它变软"；
(c) 回退到 A 原版并收尾（0 h）；
(d) 走 V7 WM Refresh（需采 B 的 on-policy 数据）。
请排序。

**Q5 · 是否还有必要保留 Δ 这条线？**
事实：base SitSofa 4/6、A 原版 4/6、A_v8 0/6；OpenDoor base 未知、A 原版 3/6、A_v8 4/6。
如果 base 在 OpenDoor 也是 4/6+，那 Δ 这条线在真机上**没有任何净收益**。请给出"继续/停止"的判据。

---

## 7. 复现命令

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03

# ① 任务级 Δ 行为拆解（方向性指标的来源）
python3 scripts/diag_task_breakdown.py \
  --ckpt runs/three_track_v7/ab_v8/A_v8/checkpoint_actor_best.pt \
  --eval-set runs/temporal_ac_bptt_v5/eval_set --split train \
  --limits runs/three_track_v6/physical/limits_standard29.npz \
  --device cuda:1 --out /tmp/task_bd.json

# ② 固定窗口统一诊断（ΔJ / 同观测物理 / 限位分解）
python3 scripts/diag_v81_fixed_eval.py --ckpt <ckpt> \
  --eval-set runs/temporal_ac_bptt_v5/eval_set --split val \
  --limits runs/three_track_v6/physical/limits_standard29.npz \
  --z0-json runs/three_track_v8/ab_v81_no_refresh/recovery_z0_v8.json \
  --device cuda:1 --out /tmp/diag.json

# ③ 闭环（当前用的脚本；base 臂把 --delta-v5-ckpt 去掉即可）
bash scripts/run_closedloop_A_v8.sh    # CKPT=<ckpt> TASKS="..." SEEDS="..." 可覆盖

# ④ V8 训练（已完成的配置）
bash scripts/queue_v82_ab.sh          # A/B 各 200 步，WM0 硬钉
```

**关键产物**：

| 内容 | 路径 |
|---|---|
| 闭环（A_v8，进行中） | `runs/three_track_v8/closedloop/A_v8/` |
| base 闭环（11-06） | `comimic/runs/closedloop_sr/sr4/{sits_base,vn_base}/` |
| A 原版 18 集（重试口径） | `runs/three_track_v6/isaac/A_delta_continue/` |
| 四代训练产物 | `runs/three_track_v7/ab_v8/`、`runs/three_track_v8/ab_v81_no_refresh/`、`runs/three_track_v8/ab_v82/` |
| 方向性/物理拆解 | `scripts/diag_task_breakdown.py`、`scripts/diag_delta_phys_v5.py`、`scripts/diag_v81_fixed_eval.py` |

---

## 8. 红线

1. 不改部署契约（hist10/HA_V31、`cap` 语义、`t≥K` 归零、`G1_ACTION_SCALE`、SONIC-only）；
2. 不改 VLA 底座（FT8000 冻结）；
3. **任何 run 必须显式钉 WM 版本 + 回读 `config.json.wm_pin.sha256`**；
4. 训练采样必须带 `allowed_clips`（防 val/test 泄漏）；
5. 一次只起一个 Isaac **仅在两者共用同一 GPU 时成立**；本机 2 卡各 1 个 Isaac 已验证可跑；
6. 旧实验目录只读。
