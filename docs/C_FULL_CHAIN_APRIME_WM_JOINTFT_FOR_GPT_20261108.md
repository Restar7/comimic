# 从 A′ 到 WM 到联合微调再到闭环：C 线全链路、结论、以及我们踩过的错误

> **提交对象**：外部评审（GPT）。本文件自包含，不需要仓库，也不需要历史对话。
> **写于**：2026-11-08 21:55。
> **目的**：把整条链路——**A′ 怎么造 → WM 怎么训 → WM 怎么一步步被用来做联合微调 → C 得到了什么 →
> 闭环为什么是现在这样**——一次讲清，并把我们**下错过的结论**与**未验证的猜测**逐条标出来。
>
> 证据等级：【FACT】有落盘文件可复核；【INFER】从 FACT 推出，推理链写明；【GUESS】未验证，附证伪方法。
>
> **一句话结论**：**WM 没有训坏（方向真物理验证过、覆盖也够），C 的离线增益也是真的（+1.04%）；
> 但"0.4 秒的想象跟踪奖励"和"12–54 秒的任务成功"之间没有已验证的传导链，而闭环失败形态
> （11/12 是 timeout、只有 1 次摔倒）正好落在这条断链上。**

---

## 0. 三十秒速读

| 问题 | 答案 | 等级 |
|---|---|---|
| 整条链有几层？ | 4 层：**参考动作库 → A′（真转移）→ WM（冻结的"可微 plant"）→ VLA+LoRA（唯一可训）** | FACT |
| A′ 是什么？ | 529,131 条 `(s_t, a_t, s_{t+1})` 真转移，来自 699 条参考轨迹、634,016 帧 | FACT |
| WM 是什么？ | MLP **133→2048→2048→70**（8.8M 参数），`s_{k+1}=s_k+dt·f(s_k,a_k)`，dt=0.02，输入=71 状态+29 动作+33 ξ | FACT |
| WM 怎么用？ | 冻结。联合微调时：token → 冻结 SONIC policy → action → **WM 滚 20 步** → 与参考算 reward → J20 → 反传 | FACT |
| WM 准吗？ | 三种口径：录制轨迹保真 **ratio 1.018（PASS，端点 p50 2.11 cm）**；任意真转移 20 步 **4.59 cm**；旧文档 0.89 cm 复现不出来 | FACT |
| WM 方向对吗？ | **对**。两次真物理实验：48 窗口 **69% 同号、真实漂移 −1.23 cm**；64 窗口 H=20 **98.4% 同号** | FACT |
| WM 幅度呢？ | **高估**：`|ΔJ_WM|/|ΔJ_true|` = **1.79×（中位）/ 2.17×（均值）** | FACT |
| A′ 覆盖够广吗？ | **够**。真实闭环状态到 A′ 的最近邻距离：p50 7.26 vs A′ 自身 4.41，**只有 0.5% 超出 A′ 99 分位阈值** | FACT |
| C 的离线成绩？ | test **ΔJ +0.01640（+1.04%，CI 0.77–1.29%）**，是旧配方 2.9 倍；占 J 剩余 headroom 的 6.7% | FACT |
| C 的闭环成绩？ | **12/24（50%）**：Boxing 5/6、VisionNavi 3/6、OpenDoor 2/6、SitSofa 2/6 | FACT |
| 失败长什么样？ | **12 次失败里 11 次 timeout（跑满 3000 步）、只有 1 次摔倒** | FACT |
| 和 base 比？ | **同场景配对后无显著差异**：OpenDoor 4:0 全倒向 base（McNemar p=0.125），SitSofa 2:1、VisionNavi 0:1（p=1.0） | FACT |
| 最可能的问题？ | ① 目标—指标错配（J20=0.4 s 跟踪 ≠ 12–54 s 任务）；② WM 幅度 1.8× 累积；③ 统计功效不足 | 【INFER】 |

---

## 1. 整体架构（一张图）

```
┌──────────────────────────────────────────────────────────────────────┐
│ L0  参考动作库（699 条 / 634,016 帧）                                  │
│     runs/isaac_motion/ha_cmd[_exec]/packed.pt                        │
│     每帧: root pos(3) + quat(4) + v(3) + w(3) + q(29) + qd(29) = 71   │
└───────────────┬──────────────────────────────────────────────────────┘
                │ 真 Isaac rollout 采集（SONIC decoder 执行 + 真物理）
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ L1  A′ 真转移库  runs/a_prime_sonic_exc/all_transitions.pt            │
│     (s_t, a_t, s_{t+1})  529,131 条                                   │
│     states/next_states (N,71) · actions (N,29) · frames · clip_ids    │
└───────────────┬──────────────────────────────────────────────────────┘
                │ 监督学习（单步 + k=20 多步 + 增益约束）
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ L2  WM（世界模型，**冻结**）  runs/a_prime_sonic_exc/wm_gain2.pt       │
│     MLP 133→2048→2048→70，dt=0.02，in/out_norm                        │
│     s_{k+1} = s_k + dt·f(s_k, a_k)                                    │
│     作用：**可微的 plant**，让"动作好坏"能反传到 VLA 参数              │
└───────────────┬──────────────────────────────────────────────────────┘
                │ 联合微调：J20 = 20 步想象回报（见 §4）
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ L3  VLA（FT8000）+ **LoRA16（唯一可训，1.03%）**                       │
│     image + state64 + instruction → 4 步 Flow Matching → token[40,64] │
└───────────────┬──────────────────────────────────────────────────────┘
                │ token → 冻结 SONIC policy → action29 → q_raw
                ▼
       （部署时）真实 Isaac：SONIC ONNX decoder → PD → 真物理 → 成功率
```

**每一层的冻结/可训状态**：

| 层 | 是否训练 | 规模 |
|---|---|---|
| 参考动作库 | 只读数据 | 699 条 / 634k 帧 |
| A′ | 只读数据 | 529,131 条 |
| **WM** | **冻结**（`--strict-wm-hash fc5aeb30…`） | 8.8M 参数 |
| SONIC policy（token→action） | 冻结 | WBC ckpt `WBC_A24000_wmvel_cmd.pt`（54.5 MB） |
| **VLA LoRA** | **可训** | 16,891,904（全模型 1.032%） |
| VLA 主体（FT8000） | 冻结 | 1,637,293,184 |

---

## 2. A′：真转移库是怎么造的

### 2.1 参考动作库【FACT】

* `packed.pt` 里 7 个任务族各约 100 条轨迹：**699 条 / 634,016 帧**
  （double_desk 100、pp_box 100、boxing 100、open_door 100、sit_sofa 100、vision_navi 100、grap_cup 99）。
* 每条轨迹命名如 `HSI_open_door__yb__ep1776527240815310`（任务 + 操作者 + episode）。
* 每帧 71 维：`[root_pos(3), quat(4), linvel(3), angvel(3), q(29), qd(29)]`。

### 2.2 A′ 的字段与语义【FACT】

`runs/a_prime_sonic_exc/all_transitions.pt`，keys = `states / actions / next_states / frames / clip_ids / meta`。

| 字段 | 形状 | 语义 |
|---|---|---|
| `states` | (529131, 71) | 真实状态 |
| `actions` | (529131, 29) | **`(q_cmd − default_q) / action_scale`** —— 是"命令动作"，不是 env 的 43 维动作、也不是 decoder 归一化输出 |
| `next_states` | (529131, 71) | 下一真实状态（含跟踪滞后） |
| `frames` | (529131,) | 在参考库里的全局帧号（→ 用来取参考、算漂移、算 reward） |
| `clip_ids` | (529131,) | 属于哪条参考轨迹 |

> 动作语义是**最容易搞错的地方**：`build_real_transitions.py` 明确写 `a_t = (q_cmd(t+1) − default_q)/action_scale`。
> 早期把 `q_target_applied` 或 43 维 env action 塞进去，会让 WM 学到一个错误的动力学（§9 错误清单 E4）。

### 2.3 A′ 的规模与分布【FACT】

* 529,131 条，来自 634,016 帧参考库（同一批 rollout）。
* **相对参考的 root 漂移**（`||state[0:3] − ref[frame][0:3]||`）：

| p10 | **p50** | **p90** | p99 | ≤0.1 m |
|---|---|---|---|---|
| 0.012 m | **0.056 m** | **0.122 m** | 0.146 m | **79.0%** |

### 2.4 为什么要造 A′【INFER】

WM 必须是"真实世界的可微替身"。用 HA 录制数据造 `(s_t, a_t, s_{t+1})`
= "部署时执行器给这个动作，机器人实际怎么动"（含跟踪滞后），
这样 WM 学到的动力学才和部署一致。**不需要仿真器**（`build_real_transitions.py` 的动机原话）。

---

## 3. WM：结构与训练

### 3.1 结构【FACT】

```
输入 133 = 71（状态） + 29（动作） + 33（ξ，训练时恒为 0）
MLP: 133 → 2048 → LN → SiLU → 2048 → LN → SiLU → 70
积分: s_{k+1} = s_k + dt · f(s_k, a_k),  dt = 0.02（50 Hz）
参数: 2D 权重 8,804,352；带 in_norm / out_norm（mean/var）
```

### 3.2 训练：两阶段 + **增益约束**【FACT】

`wm_gain2.pt`（35.3 MB，mtime **2026-11-05 15:54**，sha256 `fc5aeb30…`）内容：

```
wm           : 上面的 MLP + in/out_norm
stage1 = 800 : 单步样本 + L_gain（配对探针）
stage2 = 800 : k=20 多步窗口 + 增量项 + L_gain（必须保留）
gain_weight = 2.0
gain_pairs  = runs/paired_probe2/gain_pairs.pt（**7,456 组**"同状态 × 4 个不同动作"的真配对）
```

**为什么要 `L_gain`**：A′ 单步数据对"动作增益"约束很弱 ⇒ 阶段 2 会把阶段 1 修好的增益拉回旧解。
`build_gain_pairs.py` 用**同状态、不同动作**的配对探针（状态两两距离 <1e-5、动作 std >0.05）造出 32 组，
专门约束 `∂s'/∂a` 的**幅度**（`gain_pairs.pt`: `s (7456,71)`、`a (7456,4,29)`、`s1 (7456,4,71)`）。

### 3.3 损失口径【FACT】

* 分块 SSM：`pos / rot / linvel / angvel / q / qd` 六块，各用 **WM 自己的 `out_norm.std() + ERR_FLOOR`** 归一化（不是等权 MSE）；
* 多步：**不做 teacher forcing**（喂模型自己的输出），且必须传 `step_idx`（GRU 隐状态/逐 step 标定）；
* 绕过 `bundle.step` 的 `std→ha` 换序，走 `bundle.base.step`（否则同一个张量被置换两次）。

### 3.4 负结果（别重复踩）【FACT】

| 尝试 | 结果 |
|---|---|
| **GRU 版 WM**（1 层 1024 + 3 层 2048 头，参数 +13.5%，带记忆） | **更差**：最好一只 k=20 = 7.3 cm（比 MLP 基线差 28%）；另一组 9.4 → 9.7 cm ⇒ 隐状态里没有可用信息 |
| 4 次"补反事实数据"（token/动作扰动，σ 0.02–0.30，占比 8–25%） | **全部失败**：k=20 从 6.5 cm 恶化到 12–65 cm |

### 3.5 保真度：三种口径、三个答案【FACT】

| # | 协议 | 20 步 root 误差 | 结论 |
|---|---|---|---|
| A | `probe_wm_multistep.py`：256 个策划窗口，**喂录制动作** | 想象 **6.93 cm** vs 真实 **6.81 cm**，**ratio 1.018**；端点 p50 **2.11** / p90 **5.93 cm**；corr **0.929** | **PASS**（判据 ratio∈[0.8,1.2]） |
| B | `diagnose_wm_shift_v7.py`：任意真实转移，逐 tick 均值 | h=1 **2.80** → h=5 3.00 → h=10 3.55 → **h=20 4.59 cm**（p95 18.5 cm） | 高动态分布 2–5 cm |
| C | 旧文档（2026-11-05）的"端点保真" | **0.89 cm** | **今天复现不出来**（权重重导出/哈希争议；mtime 11-05 15:54） |

---

## 4. WM 怎么被一步步用来做联合微调（**这是本文最该看懂的一节**）

### 4.0 训练窗口【FACT】

`runs/temporal_ac_bptt_v5/eval_set`：**448 个窗口**（train 256 / val 96 / test 96），H=20，burnin=10。
每个窗口固定存了：`frames`（起点帧号）、`clip_id`（哪条参考）、**`refs_f = frames+1..20`（同一条参考的后续 20 帧）**、
`state`(71)、`history`(10×93)、`prev_action`(29)、`tokens_label`(40,64)。

* **448/448 个窗口的 20 个参考帧全部落在同一条 clip 内**（0 个跨边界，已逐条验证）；
* 起点筛选：真实采到过、`root 漂移 ≤ 0.1 m`；**之后 burn-in 10 步带扰动**（`xy=0.05, yaw=0.10, dq=0.05, dqd=0.30`）；
* 实际起点分布：p50 **0.076 m**、p90 **0.164 m**（比 0.1 宽）。

### 4.1 一次训练迭代，逐步发生了什么

```
① 采 batch（32 个窗口，70% 普通 + 30% 困难起点；支持域门禁 99.2%）
        │
② VLA 前向：image + state64 + instruction
   → 4 步 Euler Flow Matching（从随机噪声 x_T 出发，可微）
   → token [B, 40, 64]                     ← 这是**唯一**与 LoRA 有关的东西
        │
③ frozen SONIC policy（bundle.policy，来自 WBC ckpt）：
   token[40,64] + obs_hist + 参考帧 → action [B,29]
        │
④ 部署口径：q_raw = q_default + G1_ACTION_SCALE ⊙ a
        │
⑤ frozen WM：s_{k+1} = s_k + dt·f(s_k, a_k)，滚 H=20 步
        │
⑥ 每步算 reward：r_t = Σ_6 w·exp(−e/std²) × dt
   e = {body_pos, body_ori, body_linvel, body_angvel, root_pos, root_ori}
   参考来自 refs_f（同一条 clip 的后续 20 帧）
        │
⑦ J20 = Σ_{t<20} γ^t · survival_t · r_t,  γ=0.99, survival≡1（term_off=1）
        │
⑧ L = −J20 + λ_R·L_recovery + λ_lim·L_lim + λ_added·L_added
        + λ_margin·L_margin + λ_peak·L_peak + λ_trust·L_same
        │
⑨ 反传：穿过 20 步 WM → action → frozen SONIC policy → token → 4 步 FM → LoRA
        │
⑩ AdamW 更新 LoRA（lr 3e-5 → cosine 5e-6，grad clip 1.0）
```

**关键点**：WM 是**可微的 plant**（不是仿真器、不是 oracle）。所以 J20 的梯度会穿过 WM 回到 LoRA。
**WM 的误差和偏差，会直接变成策略的学习信号。**

### 4.2 评测（val/test）【FACT】

* 成对同噪声：adapter 与 pristine 用**同一个 flow 噪声**，算 ΔJ；
* **episode 聚类 bootstrap CI**（val 54 簇 / test 55 簇）；
* 四路归因：pristine / 仅主干 / 仅 proj_out / 全开；
* 逐关节越限与超限量。

### 4.3 物理项与标定【FACT】

| 项 | 约束 | λ |
|---|---|---|
| `L_recovery` | 跟踪误差尾部（worst-window：窗口内最差 25% tick + batch 最差 10% 窗口） | 2.223e-3 |
| `L_lim` | 裁剪前 `q_raw` 绝对越限（不对称限位、按关节尺度、max 形式） | 3.068e-2 |
| `L_added` | 相对 pristine **变差**的越限量（`q_base` detached） | 6.78179 |
| `L_margin` | 距限位太近（κ=0.05） | 0.2201 |
| `L_peak` | 单关节额外修正峰值（预算 `b_j`，29 个值） | 1.785e-5 |
| `L_same` | 同观测整体 trust（τ=0.03） | 0.01（未标定，回退旧值） |

λ 用**梯度均衡**标定（Train-only，训练前冻结）：各物理项的有效梯度（λ·‖g‖）
分别 0.0289 / 0.0385 / 0.0144 / 0.0072 / 0.0192，**合计 ≈0.108 ≈ J（0.096）的 1.12 倍**。

---

## 5. C 的结果（V8-C 最终配方）

### 5.1 配置【FACT】

```
Frozen FT8000 (1,637,293,184, fp32) + LoRA16 注入 218 层
  = 216 个 attn1/attn2/ff + 2 个 proj_out（原全秩权重冻结）
  可训 16,891,904（1.032%）= 主干 16,777,216 + proj_out 114,688
  rank=16, alpha=32（scale=2.0）；零 LoRA 恒等性 max|Δtoken| = 0.0
batch 32，lr 3e-5（两组同值）→ cosine 5e-6，warmup 20，grad clip 1.0
280 步（停在平台），7.7–8.0 s/步
```

### 5.2 val 曲线（96 窗口，54 簇）【FACT】

| it | ΔJ | 95% CI | 越限 | token Δrms |
|---|---|---|---|---|
| 50 | +0.01194 | [+0.00916, +0.01503] | 2.67% | 0.0067 |
| **100** | **+0.01727** | [+0.01280, +0.02221] | 2.56% | 0.0091 |
| 150 | +0.01470 | [+0.01094, +0.01867] | 2.66% | 0.0098 |
| 200 | +0.01669 | [+0.01255, +0.02084] | 2.61% | 0.0100 |
| **250（Best）** | **+0.01748** | [+0.01349, +0.02147] | 2.62% | 0.0102 |

### 5.3 独立 test（Best=it=250）【FACT】

| 指标 | 值 |
|---|---|
| J / J_base | 1.59213 / 1.57573 |
| **ΔJ** | **+0.01640**，CI **[+0.01217, +0.02035]**（55 簇） |
| 相对 | **+1.041%**（CI 0.772–1.291%） |
| 越限 | **2.527%**（pristine 记录值 2.59%） |
| `q_raw_absmax` | 3.843（pristine 3.580） |

**J 的量级换算**：六核全满时 J_max = (4×1.0+2×0.5)×0.02×Σγ^t = **1.8209**；
基座 1.5757 = **86.5%**，训练后 1.5921 = **87.4%**
⇒ **ΔJ 拿走了剩余 headroom（0.245）的 6.7%**。

### 5.4 四路归因（同一 checkpoint，无梯度）【FACT】

| 组合 | test J20 | 相对 pristine | token rms |
|---|---|---|---|
| pristine | 1.57060 | — | — |
| **仅主干 LoRA** | **1.58615** | **+0.01555（90%）** | **0.00834（= 全开的 80%）** |
| 仅 proj_out | 1.57317 | +0.00257（15%） | 0.00274（全开的 26%） |
| 全开 | 1.58788 | +0.01728 | 0.01038 |

> 归因有两种说法且都**不可加**：按 J 是"主干 90% / proj 15%"，按 token rms 是"80% / 26%"。
> 历史对照：旧 C1（proj_out **全秩**）的 proj 组梯度范数占 87%。本轮"主干主导"的结构反转成立。

### 5.5 逐关节与物理【FACT】

| 关节 | pristine | 训练后 | 相对改善 | 超限量 p99 |
|---|---|---|---|---|
| **j4** | **25.78%** | 24.95% | −3.2% | **1.756 rad（≈100°）** |
| **j10** | **22.55%** | 21.88% | −3.0% | **1.663 rad（≈95°）** |
| j5 | 13.33% | 11.82% | −11.3% | 1.501 |
| j11 | 9.64% | 8.44% | −12.4% | 1.329 |
| 全局 | 2.59% | **2.53%** | −2.3% | — |

> **超限量 p99 达到 1.76 rad** —— 物理上不可能是真越限，**强烈提示限位定义/关节约定问题**（见 H3）。

### 5.6 平台化【FACT】

it=100 之后 ΔJ 在 +0.0165 ± 0.0014、越限在 2.61% ± 0.05、j4 在 22.8% ± 0.3 上**无趋势**；
LR 已从 3e-5 余弦降到 ~1.07e-5，用户决定停在 280（`stopped.json`）。

---

## 6. 闭环：24 集全量结果与失败解剖

### 6.1 配置【FACT】

* Adapter：`actor_wm0_p2p4/checkpoint_lora_best.pt`（it=250，test ΔJ +0.01640，sha256 `ac85a95c…`）
* **只跑 V8-C**，4 任务 × 6 seeds = 24 集，`MAX_STEPS=3000`，双卡两条流水线，录像全开
* 真实 SONIC–Isaac（release decoder sha `c7241a12`，WBC `2390dc29`，CPU ONNX EP）
* 墙钟 1 h 57 min；总仿真 3.21 h；控制频率 3.7–4.7 步/s

### 6.2 结果【FACT】

| 任务 | n | 成功 | 成功率 | 平均步数 | 终止原因 |
|---|---|---|---|---|---|
| **Boxing** | 6 | **5** | **83%** | 998 | success×5, timeout×1 |
| **VisionNavi** | 6 | **3** | **50%** | 2352 | success×3, timeout×3 |
| **OpenDoor** | 6 | **2** | **33%** | 2302 | success×2, timeout×4 |
| **SitSofa** | 6 | **2** | **33%** | 2519 | success×2, timeout×3, fall×1 |
| **合计** | **24** | **12** | **50.0%** | — | **timeout×11, fall×1** |

* 成功步数：中位 **1068**，范围 180–2708；
* 失败步数：11 个 3000 + 1 个 2679（摔倒）。

### 6.3 失败形态：**11/12 是"站着不动/无效运动"**【FACT】

| 失败类型 | 次数 | 说明 |
|---|---|---|
| **timeout** | **11** | 跑满 3000 步仍未触发成功判定；**机器人没倒** |
| fall | 1 | SitSofa seed0，第 2679 步，`tilt_deg=90.5`（完全躺平） |

视频抽帧（SitSofa seed0）：第 0 步站在房间；**400–2000 步沙发距离几乎没变**；2400 步才靠近；
2600 步地平线开始倾斜；2670 步画面横 90°（摔倒）。
⇒ 这集是"**走了 2600 步才靠近沙发，然后在坐下前翻倒**"。

### 6.4 与 base 的**同场景**对照【FACT】

**场景是逐位配对的**：`result.json` 的 `episode_object_seed` 在 base 与 V8-C 之间**逐 (task,seed) 完全相同**
（OpenDoor seed0–5 = `11400714785074694791`…`…4790`，`source=env_seed`）。

| 任务 | base（`runs/ab36/base`，11-07） | V8-C（今天） | 不一致对（base 赢 / V8-C 赢） | McNemar 精确 p |
|---|---|---|---|---|
| OpenDoor | **6/6** | 2/6 | **4 / 0** | **0.125** |
| SitSofa | 3/6 | 2/6 | 2 / 1 | 1.000 |
| VisionNavi | 2/6 | **3/6** | 0 / 1 | 1.000 |

**读法**：OpenDoor 是唯一有一致方向的任务——**4 个不一致对全部是"base 成功、V8-C 失败"，零反向**；
SitSofa/VisionNavi 是双向噪声。

**⚠️ 但 base 是另一套环境**：python 3.12.3 + torch **2.8.0a0**（11-07），V8-C 是 python 3.11 + torch **2.7.0**；
server 采样 seed 也不同（老 `--seed 0` 固定 vs 今天默认 `--seed -1` 随机）。
**Isaac 侧（SONIC ONNX/WBC/pin）逐字相同**。
⇒ 这不是同环境配对，**必须补一个同环境 base 臂**才能定性。

---

## 7. WM 分析（关键问答）

### 7.1 WM 方向：**对**，有两次真物理实验【FACT】

**实验 ①（2026-11-05，`CURRENT_APPROACH_FOR_REVIEW_20261105.md` §12.2）**
48 个窗口、δ=0.15 SONIC 单位（≈3.3°/关节）、滚 20 步，**真物理（Isaac）结算**：

| 方向 | 真物理里"+"更好 | 真实漂移变化 |
|---|---|---|
| **∂J_WM/∂a** | **33/48 = 69%** | **−1.23 cm（变好）** |
| FT 实际造成的动作位移（0.17 单位 ≈4.3°） | 17/48 = 35% | −0.11 cm（没变好） |

原话：**"WM 没训错：它的 20 步梯度方向在真机上是有效的（69% 同号、漂移 −1.23 cm、信号/误差 ≈1.4）"**；
**"错在把方向落实成参数：FT 实际走的方向，不是动作空间里的上升方向"**。

**实验 ②（2026-11-07，`REVIEW_TWO_ROUTES_BPTT_VS_DISTILL_20261107.md` §3.2）**
64 窗口、`a0 ± δ·ĝ`（δ=0.03 rad）、H=20：

| 视界 | 窗口 | 符号一致率 | 真物理变好比例 | J_true | J_WM |
|---|---|---|---|---|---|
| H=2 | 96 | 1.000 | **1.000** | +0.0076 | +0.0087 |
| H=20 | 64 | 0.984 | **0.984** | **+0.0587** | **+0.1277** |

### 7.2 WM 幅度：**高估 ~1.8×**【FACT】

`|ΔJ_WM| / |ΔJ_true|` = **1.79×（中位）/ 2.17×（均值）** ⇒ 当时取 η=0.08 ≈ 0.15/1.8。

**【INFER】这条对 C 很关键**：C 的 LoRA 是**每步都施加** token 偏移（rms 0.010），不是一次性 δ 扰动。
如果策略沿 WM 方向但幅度按 1.8× 过冲，**3000 步的累积会把"局部最优"变成"系统性偏离"**。

### 7.3 WM 的覆盖：**够广**（这条推翻了我们之前的一个错误指标）【FACT】

**错误指标**：我们曾用"距自己那条参考的漂移 p99=14.6 cm"当作 WM 的覆盖边界。
**这是错的**——WM 学的是 `f(state, action)`，**它不知道参考是哪条**；离这条参考远，可能离另一条很近。

**正确的检验**（用真实闭环 dump 做最近邻）：
* 参考集：A′ 的 529,131 个状态，取 `q/qd`（58 维），按 A′ 自身标准化，随机抽 10 万；
* 查询：① A′ 自己另外 2000 个状态；② **base 跑 OpenDoor 的真实闭环轨迹 416 个状态**（`ab36/base/OpenDoor/dump/chunks.jsonl`）。

| 查询集 | NN p50 | p90 | p99 |
|---|---|---|---|
| A′ 自身（基准） | **4.41** | 6.35 | 11.88 |
| **闭环真实状态** | **7.26** | 9.34 | 11.29 |

**闭环状态超过 A′ 自身 99 分位阈值的比例 = 0.5%** ⇒ **闭环到过的状态基本都在 A′ 支持域内**。
（限制：dump 只有 `rot6+q+qd`，**没有绝对位置**，所以空间覆盖未验证。）

### 7.4 WM 刷新（WM1）：有效但局限【FACT】

| 项 | 结果 |
|---|---|
| 新数据 | 366 段 / 7,552 tick / 12 episodes 真 Isaac 逐 tick |
| 训练 | 150 updates，942 s；val single 2.196→1.204，kstep 29.44→15.82 |
| 门禁 | 新分布 root **31.80 → 10.97 cm（−65.5%，CI [0.676, 0.762]）**；旧 A′ 4.500→4.480（点估计 +0.45%） |
| 但 | ① 只有 **4 个 episode** 进验证；② 旧 A′ 的**成对 bootstrap CI = [−0.63%, −0.14%]（略变差）**；③ **`dir_agree` 未跑 → override 发布** |
| 对 Actor | WM1 那条 test **+0.01796** vs WM0 本条 **+0.01640**（+9.5%）——**步数 400 vs 280、聚合 mean vs worst_window，不是干净 A/B** |
| 在任意真转移上 | WM1 **4.56 cm** vs WM0 4.59（−0.7%），但 ∑ssm **+13.6% 更差**、q MAE **+32% 更差** |

### 7.5 WM 的视界：**只验证到 20 步 = 0.4 秒**【FACT】

* 保真、方向、幅度——**全部是在 H=20（0.4 s）上测的**；
* 闭环 episode 是 600–2700 步 = **12–54 秒**；
* **没有任何实验验证过 0.4 s 之外的 WM 信号**。

---

## 8. 我们的猜测（分条，附证伪方法）

> 这节是**给评审重点攻击的**。

### H1：目标—指标错配（最可能）【INFER，证据强】

* **内容**：J20 是 20 步（0.4 s）的跟踪积分奖励；任务成功是 12–54 s 的语义完成。
  优化前者不保证后者。
* **支持证据**：11/12 失败是 timeout（站得稳、不推进）；J20 里"稳定"项接近满分；
  基座已 86.5% of J_max；C 把剩余 headroom 只吃掉 6.7%。
* **反证**：J20 的 6 个核里包含 root/body 位置，理论上也奖励"靠近参考"；如果参考动作本身代表任务，
  优化跟踪应该有帮助。
* **证伪**：给闭环加**任务进度**观测（`prog_end`/`d_path`），看 timeout 的集是"走错"还是"只差一点"。

### H2：WM 幅度 1.8× 在 3000 步里累积成系统性偏离【GUESS，证据中等】

* **支持**：1.79× 是实测；2026-11-05 已有先例（FT 的位移方向对、真机漂移 +4.5 cm，8 对里 2 对变好）。
* **反证**：C 的 token 偏移量很小（rms 0.010，动作量级约 1），1.8× 只放大到 0.018；
  而且很多步的偏差方向会互相抵消。
* **证伪**：LoRA 幅度扫描（scale 0 / 1.0 / 2.0），同 seed 配对看成功率。

### H3：j4/j10 的越限主要是**限位定义/关节约定**问题【GUESS，证据强】

* **支持**：超限量 p99 = **1.76/1.66 rad（≈100°/95°）**，物理上不可能是真越限；
  原始 FT8000 自己就 25.78%/22.55%；四个 checkpoint 上比例几乎不动（±0.3pp）。
* **反证**：也可能是参考动作本身超出 G1 硬限位（数据集不规范）。
* **证伪**（半天）：取 20 个窗口，把 `q_raw`、`q_min/q_max`、参考 q 画在一起，检查符号/零位。

### H4：token 修正已饱和，近一半是采样噪声【GUESS，证据中等】

* **支持**：`dtok_rms` 从 it=100 的 0.0091 只涨到 0.0102（+12%）后 J 不再涨；
  `cond_check` 的 `dtok_std_across_windows` 在 it=250 是 **0.00459 = 45% of 0.0102**；
  `dtok_cos_across_noise` 只有 0.93–0.97。
* **反证**：cos 0.93+ 说明方向稳定；噪声大不等于信号被淹没。
* **证伪**：`vla_batch` 或 `sample_avg` 加倍后重评同一 checkpoint。

### H5：J20 的 headroom 只剩 13.5%，1% 已是不小的份额【INFER】

* J_max = 1.821；基座 86.5%、训练后 87.4% ⇒ ΔJ = 剩余 headroom 的 6.7%。
* **反证**：这个上限假设六个核同时为 1，实际跟踪任务里 body 速度/姿态核不可能全满。
* **证伪**：用"完美跟踪"（直接喂参考 q）测 J 的实际上限。

### H6：单 seed + n=6/任务，统计功效严重不足【FACT + INFER】

* **FACT**：所有 C run 都是 seed 0；test CI 只覆盖窗口/簇，不覆盖训练随机性。
* **INFER**：n=6 时 Wilson CI 是 5/6→[44%,97%]、2/6→[10%,70%]，**互相重叠**；
  要把 50% vs 83% 测到显著（80% power）大概需要 **20–30 个 seed/任务**。
* **证伪**：同配方跑 seed 1/2（各 47 min）。

### H7：WM1 的 −65.5% 有"记住这 4 集"的成分【GUESS，证据中等】

* **支持**：只有 4 个 episode / 96 窗口进验证；旧分布上 ∑ssm +13.6%、q MAE +32%（过拟合特征）；
  `dir_agree` 缺失。
* **反证**：旧 A′ 点估计没变差（+0.45%）。
* **证伪**：再采 12 集，6 训 6 验，重跑刷新与门禁。

### H8：环境差异（torch 2.8 vs 2.7）足以解释 OpenDoor 的 6/6→2/6【GUESS】

* **支持**：4 步 FM 采样对 kernel/TF32/RNG 敏感，闭环是反馈回路会放大。
* **反证**：VisionNavi 今天 **3/6 > base 2/6**——如果环境系统性拖累，不该出现反向。
* **证伪**：**同环境 base 臂**（正在采：PPBox/DoubleDesk/Boxing；OpenDoor/SitSofa/VisionNavi 还没补）。

### H9：策略学到的是"参考附近的微调"，不是"从任意场景推进任务"【GUESS，证据较强】

* **支持**：训练起点被限制在 ≤0.1 m 且只做 10 步 burn-in 扰动；闭环起点是任意场景；
  闭环失败形态是"不推进"而不是"摔"。
* **反证**：VLA 输入包含图像，理论上能"看图决定方向"，不一定需要参考。
* **证伪**：**放大训练起点漂移**（`start_max_drift` 0.1 → 0.3）重训，看闭环是否改善。

### H10：`L_added` 与 J 目标冲突且几乎不激活【INFER】

* `g_added` 与 `g_J` 的 cos = **−0.371**（唯一显著负相关）；λ=6.78（反解出来），有效梯度 0.0144 ≈ J 的 15%；
  训练末期 `L_added ≈ 6e-4`（几乎不激活）⇒ **代价付了、收益没拿到**。
* **证伪**：关掉 `L_added` 重跑 280 步，看 ΔJ 与越限是否同时变好。

---

## 9. 我们下错/差点下错的结论（**错误清单**）

| # | 错误 | 错在哪 | 现状 |
|---|---|---|---|
| **E1** | 用"距**自己**参考的漂移 p99=14.6 cm"当作 WM 覆盖边界 | WM 是 reference-agnostic 的动力学模型；该指标测的是数据分布，不是覆盖度 | **已修正**：NN 检验显示闭环状态 99.5% 在 A′ 支持域内（§7.3） |
| **E2** | 把"WM 方向验证 69%/98.4%"当成"策略闭环会变好" | 中间有断链：单次 δ 扰动 → 每步持续偏移 → 长程累积，三者不是一回事 | 已拆成 §7.1/7.2/7.5 三段 |
| **E3** | 用**未配对** Fisher 检验说"OpenDoor p=0.061 更差" | 场景其实是配对的（`episode_object_seed` 完全一致），应做 McNemar | **已修正**：配对后 4:0，p=0.125 |
| **E4** | 早期用 `q_target_applied` / 43 维 env action 当 WM 动作 | 动作合同错（应为 `(q_cmd−default)/scale`） | 已由 `assert_action_domain` fail-closed 拦住 |
| **E5** | 用 600 步预算判"策略不会做任务" | v13 时代踩过：OpenDoor 只需 855 步，600 步必然 timeout | 已改 3000 步；本轮仍 11/12 是 timeout |
| **E6** | 把 WM1 门禁的 **override 发布**当成"已验证" | `dir_agree` 从未跑；override 明确写着"不代表梯度方向已验证" | 已在文档中标注 |
| **E7** | 引用旧文档的 WM 保真 **0.89 cm** | 权重重导出后复现不出来（今天 2.11 cm 端点） | 已改用 2.11 / 4.59 cm |
| **E8** | 用"OT 保真 ratio 1.018 PASS"论证"WM 可以当训练目标" | 该协议喂的是**录制动作**、走的是录制车道；它证明"沿车道准"，不证明"横向（动作方向）准" | 已补充 §7.1 的方向证据 |

---

## 10. 证据边界（请当作硬约束）

| # | 缺口 | 状态 |
|---|---|---|
| 1 | **同环境 base 臂**（今天这套 python/torch/seed 口径） | **PARTIAL**：PPBox 已采完，DoubleDesk/Boxing 在采；OpenDoor/SitSofa/VisionNavi 还是 11-07 的老环境 |
| 2 | 闭环轨迹 dump（进度 `prog_end` / `d_path`） | 本轮 V8-C **没开**（ab36 base 有 dump） |
| 3 | WM 在 H>20 的保真/方向 | **NOT_RUN** |
| 4 | LoRA 幅度扫描（scale 0/1.0/2.0） | **NOT_RUN** |
| 5 | `dir_agree` 梯度探针（WM1 门禁） | **NOT_RUN**（override 发布） |
| 6 | 训练多 seed | **NOT_RUN** |
| 7 | 任务级 progress reward | **未实现** |
| 8 | 绝对位置维度的覆盖检验 | **无法做**（dump 无 root pos） |
| 9 | 动作域覆盖（token→action 后） | **NOT_RUN**（需 ONNX decode） |
| 10 | pristine 越限率 2.59% 的落盘 | **PROVENANCE 弱**（文档记录，未落 JSON） |

---

## 11. 下一步实验（按性价比）

| 优先级 | 实验 | 成本 | 能证伪/回答 |
|---|---|---|---|
| **P0** | **同环境 base 臂**（补 OpenDoor/SitSofa/VisionNavi 18 集） | ~2.5 h（双卡） | H8；也是所有对比的地基 |
| **P0** | **LoRA 幅度扫描** scale ∈ {0, 1.0, 2.0} × OpenDoor 6 seeds | ~2.5 h | H2；scale=0 顺便就是同环境 base |
| **P1** | **开 dump 重跑 6–8 个失败集**，算 `prog_end/d_path` | ~1.5 h | H1/H9：是"走错路"还是"只差一点" |
| **P1** | **H=40/H=100 想象回报**（同 checkpoint 重评） | 半天 | H1/H5：0.4 s 视界是否瓶颈 |
| **P1** | **放大训练起点漂移**（`start_max_drift` 0.1→0.3）重训 280 步 | ~1 h 训练 | H9 |
| **P2** | 任务级 progress reward 替代/补充 J20 | 2–3 天 | H1 根治 |
| **P2** | 多 seed 重训 | 2×47 min | H6 |
| **P2** | 限位定义审计（j4/j10） | 半天 | H3 |

---

## 12. 给评审的 10 个问题

1. **链路诊断**：11/12 失败是 timeout、只有 1 次摔倒，是否足以支持"策略稳定但不推进任务"？
   还有什么替代解释（成功判定太严？参考动作不可完成？3000 步预算不合理？）
2. **断点位置**：WM 方向已验证、覆盖也够、幅度高估 1.8×——你认为"0.4 s → 50 s"这条链断在哪？
   (a) 视界外推 (b) 幅度累积 (c) token→FM→动作的雅可比 (d) 指标本身
3. **J20 该换吗**？如果换，你会用什么任务级信号（progress / success / sparse）构造可微目标？
4. **统计设计**：单 seed + n=6/任务，下一轮最少需要多少 seed/episode 才能得到可判定的结论？
   配对设计（同场景、同 seed）是否比堆数量更划算？
5. **环境混杂**：torch 2.8 vs 2.7（Isaac 侧完全相同）能否解释 OpenDoor 6/6→2/6？
   你会怎么花最少的算力排除它？
6. **幅度扫描**：把 LoRA delta scale 从 2.0 降到 1.0/0.5，是"合理工程修正"还是"掩盖问题"？
7. **`dir_agree`**：在只有 12 episodes 的新数据上，这个探针有意义吗？要先采多少 on-policy 数据？
8. **WM 覆盖 vs 动作覆盖**：我们证明了"状态"在 A′ 支持域内（0.5% 超阈）。
   但 `(state, action)` 联合覆盖没测。你认为需要测吗？怎么测最便宜？
9. **限位问题**：j4/j10 超限量 p99 ≈ 1.76 rad（100°）。你同意这是"限位定义/关节约定"问题吗？
   正确做法是修 `limits_standard29.npz`，还是把参考动作里超限的段剔掉？
10. **对外表述**：以现有证据，哪三句话**可以写**、哪三句**不能写**？

---

## 13. 证据索引

```
WT   = /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
MAIN = /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

# L0 参考动作库
$MAIN/runs/isaac_motion/ha_cmd/packed.pt            # 699 clips / 634,016 帧
$MAIN/runs/isaac_motion/ha_cmd_exec/packed.pt       # 漂移口径用的同一套参考

# L1 A′
$MAIN/runs/a_prime_sonic_exc/all_transitions.pt     # 529,131 条
$MAIN/scripts/build_real_transitions.py             # 动作语义（q_cmd−default)/scale

# L2 WM
$MAIN/runs/a_prime_sonic_exc/wm_gain2.pt            # WM0，sha fc5aeb30…，stage1/2=800，gain_weight=2.0
$MAIN/scripts/finetune_wm_gain.py                   # 两阶段 + L_gain
$MAIN/scripts/build_gain_pairs.py                   # 32 组"同状态×不同动作"配对
$WT/runs/three_track_v8/c_tracking_recovery/probe_wm0_multistep/probe.json      # 保真口径 A
$WT/runs/three_track_v8/c_tracking_recovery/diag_wm0_same/diagnostics.json      # 保真口径 B (WM0)
$WT/runs/three_track_v8/c_tracking_recovery/diag_wm1_same/diagnostics.json      # 保真口径 B (WM1)
$WT/runs/three_track_v8/c_tracking_recovery/wm_refresh/gate_publish/gate_report.json
$MAIN/runs/probe_truth/exc_dump.pt ; $MAIN/logs/truth_probe_h20.log             # 方向/幅度真物理实验

# L3 C 训练
$WT/runs/three_track_v8/c_tracking_recovery/actor_wm0_p2p4/
  ├ config.json / train.jsonl / val.jsonl / cond_check.jsonl
  ├ attribution.jsonl / physical_report.jsonl / stopped.json
  ├ checkpoint_lora_best.pt（it=250）
  └ test_best/test.json                             # ΔJ +0.01640

# 闭环
$WT/runs/three_track_v8/c_tracking_recovery/closedloop_v8c_final/{run.log,RESULTS.md,summary.json,pipe0,pipe1}
$MAIN/runs/ab36/base/{OpenDoor,SitSofa,VisionNavi}/seed*/**/result.json   # base 同场景对照
$MAIN/runs/ab36/base/OpenDoor/dump/chunks.jsonl                          # 真实闭环状态（NN 覆盖检验）
$MAIN/runs/ab36/base/_eval_20261108_base/                                # 同环境 base 补充采集（进行中）

# 相关文档
$MAIN/docs/C_V8_FOR_GPT_REVIEW_20261108.md
$MAIN/docs/C_V8_WM_SIGNAL_AND_CLOSEDLOOP_FAILURE_20261108.md
$MAIN/docs/CURRENT_APPROACH_FOR_REVIEW_20261105.md      # 69% / −1.23 cm
$MAIN/docs/REVIEW_TWO_ROUTES_BPTT_VS_DISTILL_20261107.md # 98.4% / 1.79×
$WT/docs/V7_PROGRESS.md
```

---

## 14. 术语

| 术语 | 含义 |
|---|---|
| **A′** | 真转移库 `(s_t,a_t,s_{t+1})`，动作 = `(q_cmd−default)/action_scale` |
| **WM** | 世界模型，MLP 133→70，dt=0.02；冻结；可微 plant |
| **ξ** | WM 输入的 33 维扰动项，训练时恒为 0 |
| **J20** | WM 上 20 步想象回报，γ=0.99，6 个 exp 跟踪核；**不是真机回报** |
| **ΔJ** | adapter 与 pristine 在**同一 batch、同一 flow 噪声**下的成对差 |
| **pristine** | 关掉全部 LoRA 的 FT8000（= 基座） |
| **timeout / fall** | 跑满 3000 步未成功 / 姿态终止 |
| **McNemar** | 同场景配对检验，只看"一边成功一边失败"的对 |
| **NN 覆盖** | 查询状态到 A′ 状态集的最近邻距离（标准化 q/qd 空间） |
