# C-V8 交付：训练结果、世界模型（WM）证据与未验证猜测

> **提交对象**：外部评审（GPT）。本文件**自包含**：不需要看任何历史对话或代码仓库。
> **写于**：2026-11-08 18:50（闭环评测正在并行跑，§7 是写稿时的快照；跑完后会补全 RESULTS.md）。
> **作者口径**：全部数字来自真实落盘日志/JSON；凡是没有落盘、只是推理或猜测的，**逐条标注**。
>
> 证据等级约定：
> * **【FACT】** — 有文件/日志，给出路径，可逐位复核；
> * **【INFER】** — 从 FACT 推出的结论，推理链写在条目里；
> * **【GUESS】** — **未验证的猜测**，每条都写"如何证伪"。
>
> 口径警告（很重要）：本文同时出现 **val / test**、**WM0 / WM1**、**mean / worst_window 聚合**、
> **280 步 / 400 步** 四种不同口径的 run。**它们不能拼成一条曲线**，每张表都标了口径。

---

## 0. 三十秒速读

| 问题 | 回答 | 等级 |
|---|---|---|
| C-V8 是什么？ | 冻结 1.64B 参数的 FT8000 VLA，只训 **1.03%** 的 LoRA16 参数（218 层），目标函数是**冻结世界模型上的 20 步想象回报 J20** + 6 项物理/长尾正则 | FACT |
| 离线结果？ | 独立 test **ΔJ = +0.01640**，95% CI **[+0.01217, +0.02035]**（55 个 clip 簇），相对基座 **+1.041%**（CI 0.772%–1.291%） | FACT |
| 这个收益大不大？ | J 的理论上限是 **1.821**（6 个核全满）；基座 1.5757 = 86.5%，训练后 1.5921 = 87.4% ⇒ **拿走了剩余 13.5% headroom 中的 6.7%** | INFER |
| 物理安全？ | 越限 2.59%（记录值）→ **2.53%**（目标 <2.45%，**未达**）；四个主要关节都在降，但 j4/j10 几乎不动 | FACT |
| 为什么 j4/j10 压不下去？ | 原始 FT8000 自己就在 j4 越限 **25.78%**、j10 **22.55%**；且 j4/j10 的"超限量"p99 高达 **1.76/1.66 rad（≈100°/95°）** —— 物理上不可能是真越限，**更像限位定义/关节约定问题** | FACT + 【GUESS H3】 |
| WM 准不准？ | **三种口径三个答案**：录制轨迹保真 ratio **1.02（PASS）**；任意真实转移 20 步 root 误差 **4.59 cm**；旧文档的 0.89 cm **今天复现不出来** | FACT |
| WM 刷新有用吗？ | 对新策略分布预测 **31.8 → 11.0 cm（−65.5%）**，但只换来 Actor **+9.5%** 的收益（且该对比混杂了步数/聚合方式） | FACT + 混杂警告 |
| 真机闭环？ | 之前 **BLOCKED**（环境 bug），**已修复并在跑**：V8-C 最新 adapter × 4 任务（Boxing/SitSofa/VisionNavi/OpenDoor）× 6 seeds，双卡并行 | FACT（§7 有快照） |
| 最大未验证项？ | ① WM 的**梯度方向**是否正确（`dir_agree` 从未测过）；② 单 seed；③ J20 收益是否能在真机复现 | FACT |

---

## 1. 任务与目标

### 1.1 C 线要做什么

基座是 **FT8000**（GR00T-N1.7 风格 VLA，1,637,293,184 参数，fp32，`runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`），
输入"前视图像 + 64 维状态 + 指令"，输出 40×64 的 **SONIC motion token**；token 经**冻结的 SONIC**
变成 29 维关节目标 `q_raw`，再由冻结的世界模型（WM）想象 20 步，得到可微回报 **J20**。
C 线只微调 VLA 的 action-head（LoRA），目的是在**不破坏基座**的前提下提升 J20。

### 1.2 V8-C 任务书要求（本次交付对应）

1. **长尾 Recovery**：把"误差超过 z0 的部分"用 Huber 惩罚，聚合方式用 **worst-window**（不是均值）；
2. **物理 Loss 修复**：`L_lim / L_added / L_margin / L_peak` 四项，λ 用**梯度均衡**标定；
3. **困难起点采样**：30% 的 batch 用"root/heading/速度误差 top-20%"的起点；
4. **正式训练 400 步**（实际停在 280，见 §3.5）；
5. **独立 test**（只评一次）；
6. **真实 SONIC–Isaac 闭环**（V8-C 的最终验收项，§7）。

---

## 2. 设置（可复现）

### 2.1 模型与可训练参数 【FACT】

```
Frozen FT8000 (1,637,293,184 params, fp32)
  + LoRA16 注入 218 层：
      216 个 attn1/attn2/ff + 2 个 proj_out
      原全秩权重**全部冻结**（proj_out 基底 proj_base_frozen=True）
  可训 = 16,891,904 = 主干 16,777,216 + proj_out 114,688（占全模型 1.032%）
  rank=16, alpha=32 ⇒ scale=2.0
  零 LoRA 恒等性实测：max|Δtoken| = 0.0（与 pristine 逐位一致）
```

### 2.2 数据与评测集 【FACT】

* 数据来源：HumanoidArena 6 类任务的真实录制（`runs/isaac_motion` 转成的 `eval_set`，共 **634,016 帧**）；
  切分按 **clip 级**（同一 clip 不会跨 split），`split_seed=20261107`。
* 评测集 `runs/temporal_ac_bptt_v5/eval_set`：**448 个 20 步窗口**
  （train 256 / val 96 / test 96），H=20，burnin=10，起点最大漂移 0.1 m。
* clip 数：train 375 / val 81 / test 81；6 个任务族分布（train）：
  double_desk 75、pp_box 70、vision_navi 54、open_door 55、sit_sofa 57、boxing 64。
* **val 用 96 窗口（54 个 episode 簇）**，**test 用 96 窗口（55 簇）**；
  CI 是 **episode 聚类 bootstrap**（不是窗口独立同分布）。

### 2.3 目标函数 J20 —— 它是"想象回报"，不是真机回报 【FACT】

```
J20 = Σ_{t=0}^{19} γ^t · survival_t · r_t,   γ = 0.99
r_t = dt · Σ_{6 项} w · exp(−e / std²),      dt = 0.02
     (body_pos, body_ori, body_linvel, body_angvel, root_pos, root_ori)
     权重 1,1,1,1,0.5,0.5；std 0.3,0.4,1.0,3.14,0.3,0.4
```

* **理论上限**：每步最大奖励 = (4×1.0 + 2×0.5)×0.02 = **0.1**；Σγ^t(t<20)=18.209 ⇒ **J_max = 1.8209**。
* 基座 test **1.5757 = 86.5% of J_max**；训练后 **1.5921 = 87.4%**。
* ⇒ **ΔJ = +0.0164 相当于"剩余 13.5% headroom 中的 6.7%"**。这个换算对判断"1% 到底算不算大"很关键。
* `survival_t` 是 WM 的 soft-termination；本轮 `term_off=True`（关掉，避免 WM 误触发把 20 步回报砍半）。
* **重要**：J20 由 **WM 的想象动力学**算出来。WM 的误差会直接进入 J 和梯度（见 §4、§5-H1/H2）。

### 2.4 损失与标定 【FACT】

```
L = −J20
  + λ_R      · L_recovery(worst_window)     λ_R      = 2.223e-3
  + λ_lim    · L_abs(q_raw)                 λ_lim    = 3.068e-2
  + λ_added  · L_added(v_full − v_pristine) λ_added  = 6.78179
  + λ_margin · L_margin(κ=0.05)             λ_margin = 0.2201
  + λ_peak   · L_peak(e_same, b_j)          λ_peak   = 1.785e-5
  + λ_trust  · L_same(e_same, τ=0.03)       λ_trust  = 0.01（未标定，回退旧值）
```

各项含义：

| 项 | 约束什么 | 形式 |
|---|---|---|
| `L_recovery` | 跟踪误差的**尾部**（超过 z0 的部分） | 逐 tick Huber `relu(z−z0)`，聚合 = 窗口内最差 25% tick + batch 最差 10% 窗口 |
| `L_lim` | 裁剪前 `q_raw` 的**绝对**越限 | 不对称上下限、按关节尺度归一化，`max` 形式（不被 28 个正常关节稀释） |
| `L_added` | **相对 pristine 变差**的越限量 | `E[relu(v_new − v_base)²]`，`q_base` detach |
| `L_margin` | 距限位太近（κ=0.05 的软边际） | 双段 hinge |
| `L_peak` | 单关节**额外修正**的峰值（不是 29 维 RMS） | `max_{t,j} relu(|e|/b_j − 1)²` |
| `L_same` | 同观测下的整体 trust（修正幅度 RMS） | `relu(rms/τ − 1)²` |

**标定（全部 Train-only，训练前冻结）**：

| 量 | 值 | 方法 |
|---|---|---|
| Recovery `z0` | root_pos **0.0992 m** / root_ori **0.2178 rad** / linvel **0.2428** / angvel **0.3046** | Train 窗口 Q0.7 |
| Peak 预算 `b_j` | 29 个值，mean **0.0486**、range [0.020, 0.120]、**21 个不同值**；j4=j10=**0.12**、j5=0.0396 | `1.25 × Q0.99(|Δq|)`，在**旧 C1 adapter**（有修正的参照）上标定（zero-LoRA 会全退化到下限） |
| λ | 见上式 | **梯度均衡**：λ_k = ‖g_J‖/‖g_k‖ × 占比 |

**标定时的梯度范数（对 LoRA 参数）与 cos(g_k, g_J)**：

| 项 | ‖g‖ | cos with J | λ | λ·‖g‖（有效梯度） |
|---|---|---|---|---|
| J | 0.0962 | +1 | 1 | 0.0962 |
| recovery | 12.99 | **+0.214** | 2.223e-3 | 0.0289 |
| abs(L_lim) | 1.255 | **+0.321** | 3.068e-2 | 0.0385 |
| added | 2.13e-3 | **−0.371**（与 J 反向） | 6.782 | 0.0144 |
| margin | 3.28e-2 | +0.264 | 0.2201 | 0.0072 |
| peak | 1078 | −0.056 | 1.785e-5 | 0.0192 |
| trust | 0 | — | — | 0（标定无效，回退 0.01） |

> 【INFER】物理项的"有效梯度"合计 ≈ **0.108**（范数和，方向未对齐 ⇒ 上界），是 J 的 0.096 的 **1.12 倍** ⇒ 这不是"小正则"，
> 而是**近似等权**的多目标优化。若 J 与物理目标冲突（`L_added` 的 cos 已经是负的），
> 收益会被摊薄。这一点值得评审重点关注。

### 2.5 训练超参 【FACT】

batch 32，vla_batch 32，4 步 Euler Flow-Matching（可微，checkpointed 反传），
lr 3e-5（LoRA 与 proj 两组同值）→ cosine 到 5e-6，warmup 20，grad clip 1.0，weight decay 1e-4，
seed 0，**7.7–8.0 s/步**（单卡 4090），280 步 ≈ 47 分钟。

困难采样：`hard_start_ratio=0.30`，难度= root/heading/vel 误差的 top-20%（quantile 0.8）；
支持域门禁：阈值 **15.316**（旧 A′ train 的 99 分位），通过 **233,366 / 235,280 = 99.19%**。

---

## 3. 结果

### 3.1 训练曲线（WM0 固定，val 96 窗口，成对同噪声 + 聚类 CI）【FACT】

| it | ΔJ | 95% CI（54 簇） | 越限率 | token Δrms |
|---|---|---|---|---|
| 50 | +0.01194 | [+0.00916, +0.01503] | 2.67% | 0.00669 |
| **100** | **+0.01727** | [+0.01280, +0.02221] | **2.56%** | 0.00910 |
| 150 | +0.01470 | [+0.01094, +0.01867] | 2.66% | 0.00975 |
| 200 | +0.01669 | [+0.01255, +0.02084] | 2.61% | 0.00998 |
| **250** | **+0.01748（Best）** | [+0.01349, +0.02147] | 2.62% | 0.01024 |

* 所有 CI 都不含 0；从 it=100 起进入平台（见 §3.5）。
* 条件检查（`cond_check.jsonl`，每 25 步，16 窗口）：**全部 passed**；
  `dtok_cos_across_noise` 0.89–0.97（同一观测、不同 flow 噪声下 token 位移方向一致），
  `dtok_std_across_windows` 0.0022–0.0063。

### 3.2 独立 Test（一次性，Best = it=250）【FACT】

| 指标 | 值 |
|---|---|
| J / J_base | **1.59213 / 1.57573** |
| **ΔJ** | **+0.01640**，95% CI **[+0.01217, +0.02035]**（55 簇，n=96） |
| 相对基座 | **+1.041%**（CI **0.772% – 1.291%**） |
| 越限率 | **2.527%**（pristine 同 split 记录值 2.59%） |
| `q_raw_absmax` | **3.843**（pristine 3.580） |
| token Δrms / absmax | 0.01035 / 0.0720 |
| 最差关节 | **j4 24.95% / j10 21.88% / j5 11.82% / j11 8.44% / j3 1.77%** |

> 备注：`J_cached_token = 1.57576` 与 `J_base = 1.57573` 一致（<3e-5），说明"缓存 token 的路由"
> 没有 bug，基座复算可信。

### 3.3 四路归因（同一个 checkpoint，无梯度，同一 flow 噪声）【FACT】

| 组合 | test J20 | 相对 pristine | token 位移 rms | rms / full |
|---|---|---|---|---|
| pristine（LoRA 全关） | 1.57060 | — | — | — |
| **仅主干 LoRA**（attn1/attn2/ff） | **1.58615** | **+0.01555（90%）** | **0.008335** | **80.3%** |
| 仅 proj_out LoRA | 1.57317 | +0.00257（15%） | 0.002736 | 26.4% |
| 全开 | 1.58788 | +0.01728（100%） | 0.010376 | 100% |

【INFER】两种归因说法要分开看：
* **按 J**：主干解释 **90%** 的收益，proj 15%（两者之和 >100% ⇒ 有重叠/次可加）；
* **按 token rms**：主干 0.00834 = 全开的 80%，proj 0.00274 = 26%（rms 不可加，所以两项之和 ≠ 100%）。
* 历史对照：旧 C1（proj_out 用**全秩**而不是 LoRA）的 **proj 组梯度范数占 87%**（3.05e-2 vs 主干 3.2e-3）。
  本轮的"结构反转"（主干主导）成立。

### 3.4 逐关节与物理 【FACT】

| 关节 | **pristine（原始 FT8000）** | **V8-C 训练后** | 相对改善 | 超限量 p99（rad） |
|---|---|---|---|---|
| **j4** | **25.78%** | 24.95% | −3.2% | **1.756** |
| **j10** | **22.55%** | 21.88% | −3.0% | **1.663** |
| j5 | 13.33% | 11.82% | −11.3% | 1.501 |
| j11 | 9.64% | 8.44% | −12.4% | 1.329 |
| j3 | 1.77% | 1.77% | 0 | 0.040 |
| 全局（tick×关节） | 2.59% | **2.53%** | −2.3% | — |

> 口径：2.53% = 所有 (tick×关节) 组合里越限的比例；24.95% = **j4 这一个关节**在多少 tick 里越限。
> 两者不矛盾。
>
> **关键观察**：j4/j10 的"超限量 p99"是 **1.76 / 1.66 rad ≈ 100° / 95°**，最大值 3.68 rad。
> 一个 29 自由度人形机器人在正常动作里不可能"超过硬限位 100 度还在跑"——
> 这强烈提示 **限位定义/关节符号或零位约定与参考动作不一致**（见 H3）。

### 3.5 为什么停在 280（平台化证据）【FACT】

| 指标 | it=100 | 150 | 200 | 250 | 趋势 |
|---|---|---|---|---|---|
| ΔJ | +0.01727 | +0.01470 | +0.01669 | +0.01748 | 无（+0.0165 ± 0.0014） |
| 越限率 | 2.56% | 2.66% | 2.61% | 2.62% | 无（±0.05pp） |
| j4 | 22.6% | 23.1% | 22.8% | 22.8% | 无（±0.3pp） |
| j10 | 19.9% | 20.4% | 20.3% | 20.5% | 无（±0.2pp） |

此时 LR 已从 3e-5 余弦衰减到 ~1.07e-5（剩余 120 步只是低 LR 精修），用户决定停在 280
（`stopped.json`：`{"stopped_by":"user","at_update":280,"reason":"plateau since it=100"}`）。

### 3.6 与历史 run 的对照（**口径不同，不能拼曲线**）【FACT】

| Run | WM | 步数 | 聚合 | val ΔJ | test ΔJ | test 越限 | 备注 |
|---|---|---|---|---|---|---|---|
| 旧 C1 `pilot200` | WM0 | 200 | mean | +0.00607 | +0.00568 [0.00412,0.00712] | 2.51% | proj_out **全秩**、单重 trust、无 recovery |
| `actor_seed0` | **WM1** | 400 | mean | +0.01838 | **+0.01796** [0.01180,0.02696] | 2.60% | 换了 WM |
| `actor_seed0_wm0_stopped_at190` | WM0 | ~190（val 记在 it=200） | mean | +0.01631 | — | 2.57% | 无困难采样，中途停 |
| **`actor_wm0_p2p4`（本文）** | **WM0** | **280** | **worst_window** | **+0.01748** | **+0.01640** | **2.53%** | P2/P4 最终配方 |

> 【INFER】"WM1 只多 +9.5%"（0.01796 vs 0.01640）这个对比 **不是干净的 A/B**：
> 步数 400 vs 280、聚合 mean vs worst_window、还有 LR 调度终点不同。见 H6。

---

## 4. 世界模型（WM）：全部证据

### 4.1 WM 是什么 【FACT】

* 结构：**MLP 133 → 2048 → 2048 → 70**，8,804,352 参数；`s_{k+1} = s_k + dt·f(s_k, a_k)`，dt=0.02（50 Hz）。
* 输入 133 = **71 维状态 + 29 维动作 + 33 维 ξ**（训练时 ξ=zeros）；输出 70（积分成下一状态 71 维）。
* 训练数据：A′ 真转移 **529,131 行**（`runs/a_prime_sonic_exc/all_transitions.pt`）。
* 版本：**WM0 = `wm_gain2.pt`，sha256 `fc5aeb30e987f016…`，35.3 MB，mtime 2026-11-05 15:54**。
  本轮 C 的训练**全程冻结 WM0**（`--wm-refresh-mode fixed --strict-wm-hash fc5aeb30e987f016`）。
* WM 的**作用**：它同时是 (a) 训练目标的动力学、(b) 评测裁判、(c) 物理约束的观察器。
  **它的误差会直接进入 J 和梯度**。

### 4.2 保真度：三种口径、三个答案（这是本节最重要的表）【FACT】

| # | 协议 | 数据/分布 | 20 步 root 误差 | 结论 |
|---|---|---|---|---|
| A | `probe_wm_multistep.py`：256 个**策划好的 20 步窗口**，**喂录制动作**（不做闭环） | A′ 的稳定窗口 | 想象 **6.93 cm** vs 真实 **6.81 cm**，**ratio 1.018**；端点 p50 **2.11 cm** / p90 **5.93 cm**；corr **0.929** | **PASS**（判据 ratio ∈ [0.8,1.2]） |
| B | `diagnose_wm_shift_v7.py`：任意真实转移，**逐 tick 均值** | 旧 A′ held-out（128 起点） | h=1 **2.80** → h=5 **3.00** → h=10 **3.55** → h=20 **4.59 cm**（p95 18.5 cm） | 部署会遇到的高动态分布上，**2–5 cm** |
| C | 旧文档（2026-11-05）记的"端点保真" | — | **0.89 cm** | **今天复现不出来** |

**关于 0.89 cm 的争议（必须让评审知道）**：

* `docs/CURRENT_APPROACH_FOR_REVIEW_20261105.md` 与 `docs/HANDOVER_20261105.md` 都写了 0.89 cm；
* 今天用同一个 `probe_wm_multistep.py` 在当前 `wm_gain2.pt` 上实测是 **端点 p50 = 2.11 cm**；
* `wm_gain2.pt` 的 mtime 是 **2026-11-05 15:54**（比那些文档晚），且历史上有过一次哈希争议
  （`c74060f1` vs `fc5aeb30`）⇒ **【INFER】0.89 属于"重导出之前"的旧权重**；
* 影响：任何用 "0.89 cm" 论证 "WM 误差远小于信号" 的旧结论 **都不再成立**，应改用 2.11 cm（A）与 4.59 cm（B）。

**多步误差的 h=1→h=20 增长**：2.80 → 4.59 cm（×1.64）。若是纯随机游走应 ×√20≈4.5，
而实际只 ×1.64 ⇒ **【INFER】误差以"系统性偏差"为主，不是随机噪声**（漂移慢但方向一致）。

### 4.3 WM1 刷新：数据、训练、门禁 【FACT】

**动机**：V7 任务书要求"用策略在环的真 Isaac 数据刷新 WM"。

| 项 | 值 |
|---|---|
| 新数据 | **366 段 / 7,552 tick / 12 episodes** 真 Isaac 逐 tick 采集（A 线策略在环） |
| 旧数据 | A′ 529,131 行 |
| 混合 | 70% 旧 / 30% 新（episode 防泄漏、动作域门禁 ±0.25） |
| 训练 | **150 updates**，total **942 s（6.28 s/step）**；val 三次：single **2.196 → 1.464 → 1.204**，kstep **29.44 → 19.36 → 15.82** |
| 产物 | `wm_c1_v001_candidate`（sha `81d24cafbd4b0953`）→ 发布为 `wm_c1_v001_published` |

**门禁结果（`wm_refresh/gate_publish/gate_report.json`）**：

| 判据 | 结果 |
|---|---|
| finite / action_contract / no_leakage | ✅ |
| 新 on-policy 20 步 root 相对改善 | ✅ **0.655（−65.5%）**，CI **[0.676, 0.762]** —— **但只有 4 个 episode / 96 窗口** |
| 旧 A′ 20 步不退步 | ✅（点估计 **+0.45%**：4.500 → 4.480 cm） |
| **`dir_agree`（梯度方向探针）** | ❌ **MISSING ⇒ 用 override 发布** |

> **两个必须说清的细节**：
> 1. 旧 A′ 的"不退步"是**点估计**；同一份报告里的**成对 bootstrap CI 是 [−0.63%, −0.14%]**，
>    即 **candidate 在配对意义上其实略差**。"无遗忘"这个说法 **证据不足**。
> 2. 新 on-policy 的 −65.5% **只有 4 个 episode**，CI 的簇数是 4 ⇒ 置信区间本身不可靠，
>    存在"记住了这 4 集"的过拟合风险（H7）。

**新 on-policy 详细数字**（4 episodes / 96 windows）：

| 指标 | WM0 | WM1 | 相对 |
|---|---|---|---|
| root 20 步误差 | **31.80 cm** | **10.97 cm** | **−65.5%** |
| heading 误差 | 0.3856 rad | 0.0503 rad | −87% |
| q MAE | 0.5131 rad | 0.1732 rad | −66% |

### 4.4 WM1 在"任意真实转移"上几乎没变（重要反证）【FACT】

同一套 `diagnose_wm_shift_v7.py`、同一批 128 个起点：

| 指标 | WM0 | WM1 | 变化 |
|---|---|---|---|
| root@20 步 | **4.59 cm** | **4.56 cm** | −0.7% |
| ∑ssm@20 | 89.13 | **101.30** | **+13.6%（更差）** |
| q MAE@20 | 0.0162 rad | **0.0214 rad** | **+32%（更差）** |
| heading@20 | 0.0160 rad | 0.0184 rad | +15%（更差） |

【INFER】刷新让 WM **专精于新策略分布**（−65.5%），但在**旧分布/一般分布上反而变钝**。
它不是"更准的世界模型"，而是"更贴近当前策略的世界模型"。
**这直接影响"WM refresh 能不能作为通用杠杆"的判断。**

### 4.5 刷新对 Actor 的收益（混杂警告）【FACT + INFER】

* WM0 run（280 步，worst_window）：test **+0.01640**
* WM1 run（400 步，mean）：test **+0.01796**
* 名义差 **+9.5%**。

【INFER】这个数字 **不能** 直接读成"WM 刷新的价值"：两条 run 在 **步数（280 vs 400）**、
**recovery 聚合（worst_window vs mean）**、**LR 终点** 三处都不同。要干净回答，
必须**同配方只换 WM**（未做，见 §6）。

### 4.6 WM 作为"裁判"的已知失效记录（负结果，避免重复踩坑）【FACT】

| 尝试 | 结果 |
|---|---|
| **GRU 版 WM**（1 层 1024 + 3 层 2048 头，参数只比 MLP 多 13.5%，带记忆） | **更差**：最好的一只 k=20 是 **7.3 cm（比 MLP 基线差 28%）**；另一组同参数对比里 k=20 从 9.4 变 9.7 cm ⇒ **隐状态里没有可用信息**，不是接线问题 |
| 4 次"补反事实数据"（token/动作扰动，σ 0.02–0.30，占比 8–25%） | **全部失败**：k=20 从 6.5 cm 恶化到 12–65 cm |

⇒ 【INFER】WM 的瓶颈**不在记忆**（GRU 更差）、**不在训练数据覆盖**（4 种补法全失败）；
容量方面只验证过"GRU 头 +13.5% 参数"这一档，**没有系统做过加宽 MLP 的消融**（缺口）。
而在 **它对"动作变化"的响应本身**（即梯度方向的正确性，见 H2）。

---

## 5. 我们的解释与猜测（分条标注，附证伪方法）

> 这一节是**给评审重点攻击的**。每条都写：内容 → 支持证据 → 反证/薄弱处 → 如何证伪。

### H1：+1.04% 的 J 收益里，有一部分是在"拟合 WM 误差"【GUESS】

* **内容**：J20 由 WM 计算。WM 在部署分布上的 20 步 root 误差是 **4.59 cm**。
  奖励核 `root_pos_std = 0.3 m`，则 4.6 cm 误差对应核内 `z = (0.046/0.3)² ≈ 0.023` ⇒
  **单步奖励误差 ≈2.3%**；而全程 ΔJ/J ≈ **1.04%**。**两者同量级**。
  即：策略完全可能通过"把状态推到 WM 以为更好、实际没那么好"的区域来拿到这 1%。
* **支持证据**：WM 误差与收益同阶；`term_off=True` 说明 WM 的 survival 本身已被判定不可信
  （会误触发，把回报砍半）；h=1 的误差 2.80 cm 就已是"每步约 0.9%"量级。
* **反证/薄弱**：在**评测集分布**（稳定的 20 步窗口）上，probe 的 ratio 只有 **1.02**，
  也就是"沿录制车道"WM 是准的；而且 test 是**独立 clip**，收益可复现，不是单集过拟合。
* **如何证伪**：用**同一策略**在 3 个不同 seed/数据刷新的 WM 上评 J20。
  若 ΔJ 在 ±0.004 以上漂移，说明评测对 WM 版本敏感，1% 不可靠。
  （更硬的做法：真机/Isaac 跑少量窗口的**真实 J20**，与 WM 的 J20 对比。）

### H2：瓶颈是 WM 的**梯度方向**，不是预测精度【GUESS，但证据较多】

* **内容**：WM1 把新分布的预测误差降了 65.5%，Actor 收益只多 9.5%（且混杂）。
  ⇒ "预测更准"没有等比例转化为"梯度更有用"。
* **支持证据**：`dir_agree` **从未测过**（门禁 override 发布）；
  GRU/反事实数据都失败 ⇒ 不是记忆、也不是数据覆盖（容量只验证过 GRU 那一档）；`L_added` 与 `g_J` 的 cos 是 **−0.371**（目标之间互相拉扯）。
* **反证/薄弱**：+9.5% 的对比本身混杂（H6）；4 个 episode 的新数据太少。
* **如何证伪**：跑 `scripts/probe_wm_gradient_truth.py`（ΔJ_WM vs ΔJ_Isaac，perturb 动作后各滚 20 步），
  直接测 `dir_agree`。这是**成本最低、信息量最大**的下一步。

### H3：j4/j10 的"越限"主要是**限位定义/关节约定**问题，不是策略真的越限【GUESS，证据强】

* **内容**：`limits_standard29.npz` 与参考动作/`q_raw` 的关节符号或零位约定不一致，
  导致 j4/j10 被"假越限"，真正的物理越限率远低于 2.53%。
* **支持证据**：① 原始 FT8000 自己就 25.78%/22.55%；② 超限量 p99 = **1.76/1.66 rad（≈100°/95°）**，
  物理上不可能是真的；③ 四个 checkpoint 上 j4/j10 的比例几乎不动（±0.3pp）——
  训练改不动的东西，通常是**定义**而不是**行为**。
* **反证/薄弱**：也可能真的是参考动作就要求超出 G1 硬限位（数据集本身不规范）。
* **如何证伪**（半天）：取 20 个 test 窗口，把 `q_raw`、`q_min/q_max`、参考动作的 q 画在同一张图上，
  检查符号/零位；若是约定问题，把 `q` 变换到同一约定后越限率应该塌到 <0.5%。

### H4：token 修正已经"饱和"，且近一半是采样噪声【GUESS，证据中等】

* **内容**：`dtok_rms` 从 it=100 的 0.0091 只涨到 0.0102（+12%），J 不再涨；
  而 `cond_check` 的 `dtok_std_across_windows`（同一观测、不同噪声之间的 token 位移标准差）
  在 it=250 是 **0.00459**，相对 `dtok_rms` 0.0102 是 **45%**。
  ⇒ 一半的"动作修正"其实是 4 步 flow-matching 的采样噪声，不是策略信号。
* **支持证据**：上表；`dtok_cos_across_noise` 只有 0.93–0.97（不是 1.0）。
* **反证/薄弱**：`cos` 0.93+ 也说明方向是稳定的；噪声大不一定意味着信号被淹没。
* **如何证伪**：把 `vla_batch` 或 `--sample-avg` 加倍再评同一个 checkpoint；
  若 ΔJ 明显上升，说明评测本身有采样噪声；若不变，说明噪声与收益无关。

### H5：J20 的 headroom 只剩 13.5%，"1%"可能已是不小的份额【INFER】

* **内容**：J_max = 1.821（六核全满）。基座 1.5757（86.5%），训练后 1.5921（87.4%）。
  ΔJ = 0.0164 = **剩余 headroom（0.245）的 6.7%**。
* **支持证据**：奖励核与权重的定义（§2.3）。
* **反证**：这个"上限"假设六项核同时为 1，实际跟踪任务里 body 速度/姿态核不可能全满，
  真实可达上限低于 1.821 ⇒ 6.7% 可能被高估。
* **如何证伪**：用**真机/Isaac 的参考动作**跑一遍"完美跟踪"（直接喂参考 q），测 J 的实际上限。

### H6：所有结论都建立在 **单 seed** 上，方差未知【FACT + GUESS】

* 【FACT】C 的所有 run 都是 seed 0；test 的 CI 只覆盖"窗口/簇"的统计误差，**不覆盖训练随机性**。
* 【GUESS】**seed 间的 ΔJ 波动完全未知**（没有第二个 seed 可估）。若它与 +1% 同量级，则本文所有"提升"都只能算单例。
* **如何证伪**：同配方跑 seed 1/2（各 ~47 分钟），看 ΔJ 的 seed 方差。

### H7：WM1 的 −65.5% 有"记住这 4 集"的成分【GUESS，证据中等】

* **内容**：新 on-policy 只有 12 episodes（其中 4 个进 val），96 个窗口；
  CI 的簇数 = **4**。刷新 150 updates 完全可以把这 4 集拟合得很好。
* **支持证据**：旧分布上 ∑ssm 反而 +13.6%、q MAE +32%（**过拟合的特征**）；
  gate 的 `dir_agree` 缺失，无法排除"只是背下来了"。
* **反证**：old A′ 点估计没有变差（+0.45%），说明没有灾难性遗忘。
* **如何证伪**：再采 12 集新数据，**6 集训练、6 集只做验证**，重跑刷新与门禁；若改善从 −65.5% 掉到 <−20%，即证实过拟合。

### H8：`L_added` 与 J 目标冲突，可能在不必要地压低收益【INFER】

* **内容**：`g_added` 与 `g_J` 的 cos = **−0.371**（唯一一个显著负相关的物理项）；
  它的 λ=6.78 是"反解"出来的（因为梯度只有 2.1e-3），有效梯度 0.0144 ≈ J 的 15%。
  实际 `L_added ≈ 0.0006`（几乎不激活）⇒ **代价付了，收益没拿到**。
* **支持证据**：标定表与训练末期的 `L_added` 值。
* **如何证伪**：关掉 `L_added` 重跑 280 步，看 ΔJ 与越限率是否同时变好。

### H9：`q_raw_absmax` 从 3.58 → 3.84 说明策略在**动作域边缘**工作【INFER】

* 【FACT】test 的 `q_raw_absmax`：pristine 3.580 → 训练后 3.843（+7.3%）。
* 【INFER】SONIC 的动作域是训练时见过的分布；超出越多，真机越可能出现
  "PD 跟不上 / 抖动 / 饱和"这类**离线指标看不见**的问题。
* **如何证伪**：闭环里统计 `q_raw` 的绝对最大值与饱和比例，与离线对比。

---

## 6. 未验证项与已知缺口（评审请把这些当作"证据边界"）

| # | 缺口 | 状态 | 影响 |
|---|---|---|---|
| 1 | **WM 梯度方向 `dir_agree`** | **NOT_RUN**（门禁 override 发布） | 无法证明"WM refresh 让梯度更好" |
| 2 | 同配方只换 WM（WM0 vs WM1 干净 A/B） | **NOT_RUN** | "+9.5%" 是混杂估计 |
| 3 | 多 seed（seed 1/2） | **NOT_RUN** | 训练方差未知 |
| 4 | H=40 / 更长视界 | **NOT_RUN** | 漂移是秒级现象，H=20（0.4 s）可能看不见 |
| 5 | `L_rate` | **BLOCKED** | RobotSpec 无 velocity 字段，速度约束未做 |
| 6 | 固定窗口尾部诊断（z_p50/p95/p99 跨 ckpt） | **NOT_RUN**（C 版） | 只有均值，尾部是否变好未知 |
| 7 | pristine 越限率 2.59% 的落盘 | **PROVENANCE 弱** | 该数字来自文档记录，未在 `test_best/test.json` 落盘 |
| 8 | 严格 resume 的真实演练 | **未验证** | 代码已接线（含指纹校验），没真跑过 |
| 9 | 消融（关掉 recovery / 关掉 hard sampling / 关掉 proj LoRA） | **NOT_RUN** | 无法归因到单个组件 |
| 10 | 真机/Isaac 的**真实 J20** | **NOT_RUN** | 所有 J 都是 WM 想象的 |

---

## 7. 真实 SONIC–Isaac 闭环（新增，进行中）

### 7.1 之前为什么 BLOCKED，怎么修的 【FACT】

* 现象：4 次尝试全部在 ~50 s 失败：`[SonicActionProvider] onnxruntime is not available`。
* 根因：评测脚本 source 了 `harness03_env.sh`，把 `third_party/gr00t_deps` 放进了 `PYTHONPATH`；
  那份 `onnxruntime` 缺 `capi/onnxruntime_pybind11_state*.so`，**遮住了 Isaac venv 自带的 1.22.1**。
* 修法：**server 的环境只在子 shell 里生效**；**Isaac 用 `env -u PYTHONPATH` 干净启动**，
  由 `ha_isaac_closedloop.sh` 自己 source `isaac_env.sh`。
* 验证：干净环境下 `import onnxruntime` → 1.22.1 OK；冒烟 1 集 **Boxing seed0 成功（370 步，104 s）**。

### 7.2 正在跑的配置 【FACT】

| 项 | 值 |
|---|---|
| Adapter | **`actor_wm0_p2p4/checkpoint_lora_best.pt`**（it=250, test ΔJ +0.01640, sha256 `ac85a95c29c5bc13…`） |
| 任务 | **Boxing / SitSofa / VisionNavi / OpenDoor**（只跑 V8-C，**不跑 base、不跑旧 C1**） |
| seeds | 0–5（每任务 6 集，共 **24 集**） |
| 并行 | **双卡两条流水线**：卡0 = Boxing+VisionNavi，卡1 = SitSofa+OpenDoor；每卡 1 个 server + 1 个 Isaac |
| MAX_STEPS | 3000（控制频率实测 3.6–3.9 Hz ⇒ 满 3000 步约 13–14 min/集） |
| 产物 | `runs/three_track_v8/c_tracking_recovery/closedloop_v8c_final/`（`run.log`、`pipe{0,1}/`、`RESULTS.md`） |

### 7.3 写稿时快照（2026-11-08 18:50）【FACT】

| 任务 | seed | 结果 | 步数 | 时长 |
|---|---|---|---|---|
| Boxing | 0 | **success=True** | 370 | 104 s |
| Boxing | 1 | **success=True** | 1622 | 433 s |
| Boxing | 2 | 进行中 | — | — |
| SitSofa | 0 | 进行中（已 >8 min，属长 episode） | — | — |

> 两条流水线互不阻塞；每集 = Isaac 启动 ~60 s + 仿真（实测 3.7–3.9 步/s，满 3000 步约 13 min）。

> 这一节会在跑完后补全（成功率、步数、失败原因分布、视频）。
> 在此之前的任何"真机提升"结论都 **不成立**。

---

## 8. 给评审的 10 个问题（请明确回答）

1. **+1.04% 的可信度**：在 WM 20 步误差 4.59 cm、reward std 0.3 m 的前提下，
   这个 ΔJ 有多少概率是"拟合 WM 误差"？我们还需要哪些**最小**实验才能排除？
2. **J20 的 headroom**：J_max=1.821、基座 86.5% 的换算方式是否合理？
   若不合理，你会怎么定义"还有多少空间"？
3. **多目标 λ**：物理项有效梯度（范数和）≈ J 的 1.12 倍（§2.4）。这种"等权"是否合理？
   `L_added` 与 J 的 cos=−0.37 却在训练中几乎不激活（L≈6e-4），是否应该直接删掉？
4. **四路归因的口径**：按 J 是"主干 90% / proj 15%"，按 token rms 是"80% / 26%"，两者都不可加。
   你会用哪个作为"结构是否正确"的判据？还是两个都不够？
5. **WM refresh 的正确期望**：预测 −65.5% → Actor +9.5%（混杂）。
   你认为这个转化率正常吗？还是必须先补 `dir_agree`？如果只能做一个实验，你选哪个？
6. **WM1 在旧分布上变差**（∑ssm +13.6%、q MAE +32%）但新分布大改善 —— 这算"专精"还是"过拟合"？
   4 个 episode 的验证集能不能支撑发布？
7. **j4/j10 的 1.76 rad 超限量**：你同意"这是限位定义/关节约定问题"吗？
   如果是，正确的处理是修 `limits_standard29.npz`，还是从数据里剔除这些动作？
8. **单 seed + 96 窗口**：在这样的样本量下，CI [+0.0122,+0.0204] 是否被低估
   （它没覆盖训练随机性）？你会要求几个 seed 才认这个结论？
9. **下一步的性价比排序**：在"补 dir_agree / seed1 / H=40 / 真机 J20 / 限位审计 / 闭环 24 集"里，
   你会怎么排？为什么？
10. **如果要写论文/对外汇报**：以现有证据，哪三句话是**可以写**的，哪三句是**不能写**的？

---

## 9. 证据索引（复核用，全部是绝对路径下的相对位置）

```
WT = /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
MAIN = /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

# 最终训练 run
$WT/runs/three_track_v8/c_tracking_recovery/actor_wm0_p2p4/
  ├ config.json                # 全部超参 + lora 结构 + eval_meta（split/clip 列表）
  ├ train.jsonl / train.log    # 每步 loss、各项 L、梯度/更新/token 诊断
  ├ val.jsonl                  # 5 个评测点（ΔJ、CI、越限、逐关节）
  ├ cond_check.jsonl           # 条件检查
  ├ attribution.jsonl          # 四路归因（val）
  ├ physical_report.jsonl      # 逐关节越限
  ├ stopped.json               # 停在 280 的原因
  ├ checkpoint_lora_best.pt    # it=250（test 用的就是它）
  └ test_best/test.json        # ★ 独立 test：ΔJ/CI/逐关节/四路归因

# 标定
$WT/runs/three_track_v8/c_tracking_recovery/calibration/
  ├ physical_weights_v8c.json  # λ + 标定梯度范数/cos
  ├ recovery_z0.json           # Q70
  └ peak_budget_self_q99.json  # 29 个 b_j

# WM
$MAIN/runs/a_prime_sonic_exc/wm_gain2.pt                  # WM0, sha fc5aeb30…
$WT/runs/three_track_v7/wm_versions/{wm_v000,wm_c1_v001_published,active.json}
$WT/runs/three_track_v8/c_tracking_recovery/probe_wm0_multistep/probe.json   # 口径 A
$WT/runs/three_track_v8/c_tracking_recovery/diag_wm0_same/diagnostics.json    # 口径 B（WM0）
$WT/runs/three_track_v8/c_tracking_recovery/diag_wm1_same/diagnostics.json    # 口径 B（WM1）
$WT/runs/three_track_v8/c_tracking_recovery/wm_refresh/gate_publish/gate_report.json  # 门禁
$WT/runs/three_track_v8/c_tracking_recovery/wm_refresh/refresh.log            # 刷新训练

# 历史对照
$WT/runs/three_track_v7/c1_lora_proj/pilot200/             # 旧 C1（proj 全秩）
$WT/runs/three_track_v8/c_tracking_recovery/actor_seed0/   # WM1 400 步
$WT/runs/three_track_v8/c_tracking_recovery/actor_seed0_wm0_stopped_at190/

# 闭环（进行中）
$WT/runs/three_track_v8/c_tracking_recovery/closedloop_v8c_final/
  ├ run.log  ├ pipe0/{server.log,runs/,audit/}  ├ pipe1/...  └ RESULTS.md（跑完生成）

# 相关文档
$MAIN/docs/C_STATUS_V8C_FINAL_20261108.md          # 上一版状态（不含闭环）
$MAIN/docs/HANDOVER_FULL_SESSION_20261108.md       # 全会话交接（含踩坑表）
$WT/docs/V7_PROGRESS.md                            # WM 刷新/门禁/诊断的实施记录
$MAIN/docs/HANDOVER_WM_20261031.md                 # WM 结构/GRU 负结果
$MAIN/docs/CURRENT_APPROACH_FOR_REVIEW_20261105.md # 0.89 cm 的来源（已过时）
```

---

## 10. 术语与口径表

| 术语 | 含义 |
|---|---|
| **J20** | 冻结 WM 上 20 步想象回报，Σγ^t·survival·r_t，γ=0.99；**不是真机回报** |
| **ΔJ** | J(adapter) − J(pristine)，**同一 batch、同一 flow 噪声**成对计算 |
| **CI** | episode 聚类 bootstrap 95% 区间（val 54 簇 / test 55 簇） |
| **pristine** | 关掉全部 LoRA 的 FT8000（= 基座） |
| **WM0 / WM1** | `wm_gain2.pt`(fc5aeb30) / `wm_c1_v001_published`(81d24caf) |
| **worst_window** | Recovery 聚合：窗口内最差 25% tick + batch 最差 10% 窗口 |
| **dtok_rms** | adapter 相对 pristine 的 token 位移 RMS（64 维，40 帧） |
| **越限率** | `relu(lo−q)+relu(q−hi) > 0` 的 (tick×关节) 比例 |
| **超限量 p99** | 越限幅度（rad）的 99 分位；j4 的 1.76 rad ≈ 100° |
