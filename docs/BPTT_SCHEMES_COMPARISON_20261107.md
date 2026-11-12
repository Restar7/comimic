# BPTT 方案全集与对比（供评审敲定后重训）

> 写于 2026-11-07 08:10。目的：把**历史上所有 BPTT 方案**、**本轮新做的全部方案**、
> **统一的评测口径**、**每个方案的实测结论**摊在一份文件里，供外部评审裁定"重训用哪一个方案"。
> 所有数字都来自 `runs/harness_03_bptt/` 或历史 run 目录，没跑的一律标 `NOT_RUN / 未定`。
>
> 阅读顺序：§1 目标 → §2 统一口径 → §3 方案清单（A–G）→ §4 统一结果表 →
> §5 机理证据 → §6 结论 → §7 待解矛盾 → §8 **候选方案（请裁定）** → §9 复现

---

## 1. 目标与不变量

部署链路（不变）：

```
相机 ─► VLA(GR00T Cosmos-Reason2-2B + 1.09B DiT, checkpoint-8000, 冻结)
      ─► 64 维 motion token chunk (T=40 帧)
      ─► token'_t = token_t + Δ_t        ← 唯一的改动点（服务端）
      ─► SONIC release decoder(可微) ─► 29 维关节目标 ─► PD ⊂ Isaac
```

**不变量**：VLA 冻结；WM 冻结；decoder 冻结；奖励函数固定；只训 Δ。
**唯一的问题**：Δ 应该怎么参数化、用什么损失、怎样采样，才能让闭环成功率超过 base（58.3%）。

---

## 2. 统一口径（本轮所有新方案共用；历史方案不同处会单独标注）

| 项 | 取值 | 来源 |
|---|---|---|
| 起点数据 | `runs/a_prime_sonic_exc/all_transitions.pt`（529,131 条真 Isaac 转移） | 历史 A' 采集 |
| 起点筛选 | 只取**真采到过**且漂移 ≤ **0.10 m** 的帧（346,729 帧可用） | 与 bptt1 的 `START_MAX_DRIFT=0.10` 同 |
| burn-in | 10 步（`sample_burnin`，扰动 xy 5cm / yaw 0.1 / dq 0.05 / dqd 0.3） | 与 bptt1 同一条代码路径 |
| 参考帧 | `teacher_references`，从 f+1 开始取 H 帧 | 同上 |
| token chunk | **官方 VLA（checkpoint-8000）在窗口自己那帧上采样并缓存**（`fm_steps=4`） | 本轮新增：为了让 BPTT 便宜/可复现 |
| WM | `runs/a_prime_sonic_exc/wm_gain2.pt`（冻结，sha256 `fc5aeb30e987…`） | 与 bptt1 同 |
| 执行器 | SONIC release decoder（ONNX→torch 图，可微） | 与部署同一个 |
| 奖励 | `saturating, vel_scale=1, huber 0.02/0.15, d_max=0.3, root_ori=split, tail_w=2, yaw_hinge=0.3, **term_off=1**` | 直接读 `runs/bptt1/args.json` |
| 回报 | `J_i = Σ_{t<H} γ^t r_t`，**H=20, γ=0.99, survival≡1** | 与 bptt1 同 |
| 数据切分 | **按 clip 分组**：train/val/test = 375/81/81 个 clip（6 类任务各覆盖） | 防相邻帧泄漏 |
| 窗口数 | 固定窗口实验：8/32/128/256；population 实验：池 **3,072**（train） | |
| 幅度预算 | `apply_norm_cap(Δ, 0.08)`（tanh 软上限，**与部署同一个函数**） | 0.08 = η = 0.15/1.8 |
| 指标 | 逐窗口 J、ΔJ vs Base、**逐窗口配对差**、positive window fraction、动作变化 RMS、‖Δ‖、cap 激活率 | |

**数据侧两个硬事实**（影响可学性上限，务必一起读）：

- `‖token_0‖ ≈ 1.337`；**VLA token 与录制库 token 平均绝对差 0.0717** —— 与整个 Δ 预算 0.08 同量级。
- **VLA 自身两次采样**平均绝对差 **0.0328（相对 24.1%）** —— Δ 想学的修正里，一部分会被 VLA 采样噪声淹没。

---

## 3. 方案清单

### A. 历史方案（已判定失败，仅作背景）

| 方案 | 结构 | 损失 | 窗口量 | 结果 |
|---|---|---|---|---|
| `v10/v11/v13` | LoRA rank 32（改 VLA 头部） | λ_r·reward + BC 等 | bs 8，100–500 步 ⇒ **≈1,000–4,000 个不同窗口** | 学到**常向量**；跨臂 `cos(v10,v11)=0.980`；v13 闭环 30.8% < base 58.3% |
| `p1b/p1c` | state_extra + **反常量项** 0.05 | +var_weight | bs 8, 300 步 ≈2,400 | 输出方差上去了，WM 回报只 +2.9%，残差/均值 仍 0.164~0.170 |
| `p1d` | Δtoken，λ_r=5，act 0.3 | +动作 hinge | bs 19, 139 步 | WM 回报 **0 改善**（0.9196→0.9196） |
| `p1e` | Δtoken，λ_r=15 | +动作 hinge | bs 19, 300 步 | 回报 +4.0%，但 ‖Δ‖→**0.381（4.7η）**，动作过冲，不可部署 |

**共同形态**：回报几乎不动，或靠超预算的 Δ 换一点点回报。

### B. `bptt1`（历史最后一个 BPTT 主臂，**未跑完**）

完整配方（`runs/bptt1/args.json` + `v3_config.json`）：

```
STEPS=150（实际 59 步后被 SIGTERM 杀掉）  NPROC=2 DDP  BATCH=256/rank（全局 512） ACCUM=2
H=20  γ=0.99  TERM_OFF=1  seed=0
LAMBDA_R=2.8  VAR_WEIGHT=0  ACT_WEIGHT=0.15  ACT_EPS=0.30（hinge, 中心=base token 的动作）
TRUST_EPS=0.05  TRUST_WEIGHT=0.2  TRUST_CENTER=base        ← token 空间 trust hinge
CRITIC_ON=1  CRITIC_SOURCE=imagined  boot_one_sided=1      ← critic bootstrap
WINDOW_NORM=1（loss = −mean_w(ret_w/|ret_w|)）              ← 每窗口按自身量级归一化
DELTA_TOKEN=1  DELTA_TOKEN_FULLSTATE=1  DELTA_TOKEN_LR=5e-4  DELTA_TOKEN_INIT_STD=4e-4
DELTA_TOKEN_CAP=0.0（**无幅度上限**）
GRAD_TARGET_WEIGHT=0.2  GRAD_TARGET_N=8  GRAD_TARGET_ETA=0.08  GRAD_TARGET_HORIZON=20
FREEZE_VLA=1  START_SOURCE=recorded  START_MAX_DRIFT=0.10
```

- 参数化：`Δ = f_θ(token_0, state64)`，**广播到整段 chunk**（`d.unsqueeze(1)`），`DeltaTokenNet` 2×1024，末层 std 4e-4。
- 采样：**每步重抽 512 个窗口**（池 = 全部合格帧），59 步 ⇒ **≈30,208 次抽样**（计划 76,800）。
- 实测（59 步）：

| 指标 | 首 10 步 | 末 10 步 |
|---|---|---|
| `loss/reward_raw`（= J，H20 折扣和） | 1.51847 | **1.53543（+1.12%）** |
| `delta_token/norm` | 0.0081 | **0.1523**（=1.9η，**无 cap**） |
| `delta_token/grad_cos`（Δ 与 ĝ 余弦） | +0.003 | 在 **−0.05 ~ +0.27** 之间抖动 |
| `delta_token/grad_norm_ratio`（‖Δ‖/η） | 0.09 | **1.8 ~ 2.3** |
| `delta_token/state_sens` | 1.46 | 0.93 |
| `loss/reward_norm`（window_norm 后的目标） | −0.987 | −0.935（几乎平坦） |

⇒ **WM 回报 +1.12%（≈共享常向量的水平），Δ 却长到 1.9η；且 critic 的 V(x_H) 因 `critic_grad_through_state=false` 对 Actor 无梯度**（见 `docs/TEMPORAL_AC_BPTT_V5_IMPLEMENTATION_PLAN.md` §1.1）。**没有任何闭环评测。**

### C. 本轮方案 1：最小 BPTT + **固定窗口**（过拟合诊断，已完成）

```
参数化：Δ = f(token0, state64)（广播整段）        cap=0.08
损失：L = −mean_i J_i   （无 critic / 无 window_norm / 无 grad_target / 无反常量 / 无 LoRA）
采样：**固定** N 个窗口做全批量 Adam，80 步，lr ∈ {1e-3, 3e-3, 1e-2}
N ∈ {8, 32, 128, 256}
```
（= 任务书 §8 要求的"固定窗口过拟合"实验；**不是** bptt1 的采样方式）

结果见 §4。**结论：能记住，不能迁移。**

### D. 本轮方案 2：最小 BPTT + **population 采样** + 共享 Δ（已完成）

```
参数化：同 C（共享 Δ，广播整段）                  cap=0.08
损失：L = −mean_i J_i
采样：池 3,072 窗口，**每步重抽 256**，250 步 ⇒ 64,000 次抽样
优化：Adam lr 3e-3，全批量(grad clip 不生效)
```
= **bptt1 的采样语义**（每步重抽）但损失最小化。

### E. 本轮方案 3：**per-frame Δ**（每帧一个 Δ，已完成）

```
参数化：Δ_t = f(token_t, state64)  → 输出 (T,64)，T=40（rollout 只用前 20 帧）
        每帧独立 cap；两种预算口径：
          (i) 每帧 ‖Δ_t‖ ≤ 0.08（总预算 = 20×，与共享 Δ **不同**）
          (ii) 每帧 ≤ 0.0179 = 0.08/√20（总预算与共享 Δ 相同）
损失/采样/优化：同 D
```

### F. 本轮方案 4：**per-frame Δ + 整段 chunk 进输入**（当前最优，已完成 2 seed）

```
参数化：Δ_t = f( flatten(chunk 40×64) , token_t , state64 )   → (T,64)
        （另一种更省参数的写法：z = g(chunk) 128 维摘要，再 Δ_t = f(token_t, z, state64)）
每帧 cap 0.08；损失/采样/优化：同 D
```

### G. 本轮方案 5：per-frame + chunk + **历史状态**（口径已定义，未跑完）

```
参数化：Δ_t = f( flatten(chunk) , token_t , state64 , hist_10 ) 
hist_10 = 过去 10 帧的 **WBC 观测块前 58 维** = [q(29) | qd(29)]，共 580 维
```

**口径已逐位验证**：

```
obs93 的 q 块  ==  flat state 的 q(HA_V31) − default     max|Δ| = 0.0
obs93 的 qd 块 ==  flat state 的 qd(HA_V31)              max|Δ| = 0.0
```

⇒ obs93 的 `q/qd` 与 `state64` **同一套关节序（HA_V31）**，只差一个常数偏移。
⇒ **客户端本来就为跑 decoder 每个控制步计算这份 obs**，所以只要请求里多带一个 `(10,58)` 字段，
训练/推理就**逐位同口径**（代价：破"客户端零改动"前提，客户端改动约 20 行）。
**注意不可用**：`a_prev` 块（burn-in 里由参考库推出、非实际下发动作）；`gravity/ang_vel`（估计器管线不同）。

已经跑过的两个"历史"版本（口径不同、都仅供参考）：

| 版本 | hist 特征 | Val ΔJ | Test ΔJ | 动作 RMS |
|---|---|---|---|---|
| hist64（state64 空间 10×64） | HA 序 + rot6d + 绝对值 q | +0.0172 | +0.0313 | 0.0398 |
| （未跑）obs 空间 10×58 | q−default | 未定 | 未定 | — |

---

## 4. 统一结果表（全部实测）

### 4.1 固定窗口（方案 C；每格 `ΔJ`，单位 = 20 步折扣 WM 回报）

| 窗口数 | Constant ΔJ_fit | Constant ΔJ_val | MLP ΔJ_fit | MLP ΔJ_val | MLP ΔJ_test | Oracle ΔJ（自拟合） |
|---|---|---|---|---|---|---|
| 8 | +0.0247 | +0.0111 | NOT_RUN | NOT_RUN | NOT_RUN | — |
| 32 | +0.0281 ± 0.0002 | +0.0111 ± 0.0006 | **+0.0725**（4 配置） | +0.0103 | +0.0028 | fit +0.0734 / val +0.0678 / test +0.0730 |
| 128 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | — |
| 256 | +0.0162 | +0.0156 | **+0.0700** | +0.0114 | +0.0044 | — |
| 3072（池） | 见 4.2 | | | | | |

- Constant 的 lr 不敏感（1e-3/3e-3/1e-2 → fit +0.0269/+0.0281/+0.0282），种子散布 ~0.0006。
- **MLP 训练集 2.6~4.3× Constant，held-out 反而更差** ⇒ 固定窗口 = 背题。

### 4.2 population（方案 D/E/F；池 3,072，每步重抽 256，250 步，cap 0.08）

| 方案 | 参数化 | Val ΔJ | Test ΔJ | ‖Δ‖ | 动作 RMS | 备注 |
|---|---|---|---|---|---|---|
| — | Base | 0 | 0 | 0 | 0 | J: train 1.5061 / val 1.5086 / test 1.4554 |
| D | Constant（64 维共享向量） | +0.0115 | +0.0141 | 0.080 | 0.0172 | **核心基线** |
| D | 共享 Δ MLP（token0+state64） | +0.0229 | +0.0195 | 0.080 | 0.0333 | 1 seed |
| E | per-frame Δ（token_t+state64，每帧 cap 0.08） | +0.0199 | +0.0252 | 0.070 | 0.0354 | 1 seed |
| E(ii) | per-frame，**总预算对齐**（每帧 0.0179） | +0.0067 | +0.0070 | 0.016 | 0.0059 | 预算太小，几乎无效 |
| F | per-frame + **chunk 摘要**（z=128） | +0.0279 | +0.0309 | 0.071 | 0.0330 | 1 seed |
| **F** | **per-frame + chunk 全帧（flat）** | **+0.0271**（2 seed 均值） | **+0.0347**（2 seed: +0.0351/+0.0344） | 0.072 | 0.032 | **当前最优** |
| G | per-frame + chunk + hist64（state64 空间） | +0.0172 | +0.0313 | 0.078 | 0.0398 | 1 seed，口径待换成 obs |
| 诊断上界 | Oracle（每窗口独立最优，共享 Δ） | +0.0678 | +0.0730 | 0.080 | 0.050 | **不可部署**，positive window frac = 1.00 |

### 4.3 固定窗口的负对照与消融（方案 C，N=32，MLP）

| 变体 | ΔJ_fit | ΔJ_val | ΔJ_test | state_sens |
|---|---|---|---|---|
| 真 state64（4 配置均值） | +0.0725 | +0.0103 | +0.0028 | 1.2 |
| **state64 跨窗口打乱** | +0.0730 | +0.0016 | +0.0051 | 1.09 |
| **state64 整块置零** | +0.0644 | +0.0041 | +0.0086 | 0.00 |

⇒ 打乱/置零后**训练集不掉分**（差 ≤12%）⇒ 训练集增益来自"背题"，不是状态。

### 4.4 视界消融（固定窗口 N=32）

| H | Constant ΔJ_fit | Constant ΔJ_val | MLP ΔJ_fit | MLP ΔJ_val |
|---|---|---|---|---|
| 2 | +0.0007 | −0.0000 | +0.0033 | +0.0001 |
| 5 | +0.0028 | +0.0008 | +0.0097 | +0.0007 |
| 20 | +0.0281 | +0.0111 | +0.0725 | +0.0103 |

---

## 5. 机理证据（回答"为什么"）

### 5.1 计算图与解析梯度 —— **PASS**

8 个真窗口、H=20、cap=0.08：

- 参数 100% 带 `requires_grad` 且**全部在 optimizer 里**；反向梯度有限非零；`step()` 后权重真变。
- `dJ/dtoken0 = 0.1825`（finite，**零梯度窗口 0 个**）⇒ 梯度确实穿过 SONIC decoder 回到 token。
- **有限差分 vs 解析方向导数**（共享 Δ，方向 = 解析梯度自身）：

| h | 解析 | 中心差分 | 相对误差 |
|---|---|---|---|
| 1e-1 | +0.4047 | +0.3586 | 11.4% |
| **3e-2** | +0.4047 | +0.3976 | **1.74%** |
| **1e-2** | +0.4047 | +0.4148 | **2.50%** |
| 1e-3 | +0.4047 | +0.3573 | 11.7% |
| 1e-4 | +0.4047 | −0.4792 | 218% |

大 h 有曲率误差、小 h 撞 float32 舍入 ⇒ **1.7~2.5% 的一致性就是"梯度正确"的证据**。

### 5.2 幅度标定（沿同一方向的 ΔJ）

| ‖Δtoken‖ | ΔJ | 动作变化 RMS (rad) |
|---|---|---|
| 0.02 | +0.0076 | 0.0008 |
| 0.04 | +0.0139 | 0.0035 |
| **0.08（=η）** | **+0.0203** | 0.0136 |
| 0.16 | +0.0118 | 0.0414 |
| 0.32 | **−0.0500** | 0.1244 |

⇒ **甜点在 η 附近；无 cap 放任到 0.15（bptt1 的实际值）就已经在下坡上。**
这是"bptt1 无 cap 可能是它 0 改善的原因"的直接线索。

### 5.3 理想修正的结构（逐帧梯度 `g[i,t] = ∂J_i/∂token[i,t]`，n=3,072）

| 成分 | 含义 | 占总梯度能量 |
|---|---|---|
| μ（全局常数方向） | 常向量**能**表达的全部 | **0.1%** |
| a[t]（只随帧号变） | per-frame 但不要输入 | **0.2%** |
| b[i]（窗口特有常数） | 只有逐窗口 Oracle 能表达 | 5.2% |
| **c[i,t]（窗口×帧 交互）** | **只有"看未来 chunk + 状态"的条件化能表达** | **94.5%** |

逐窗口共享梯度：`‖g_i‖=1.127`、`‖mean g‖=0.193`、`res/mean(L2)=5.75`、**两两 cos 仅 +0.043（48% 为负）**。

### 5.4 可预测性（按 clip 分组 CV，held-out）

| 特征 → 目标 | n=256（旧） | **n=3,072** | 备注 |
|---|---|---|---|
| token_only → 共享梯度 | cos 0.128（R² −0.42） | **cos 0.284（R² +0.006）** | n=256 是**小样本伪影** |
| state_only → 共享梯度 | cos 0.141（R² −0.86） | **cos 0.287（R² −0.042）** | |
| token0+state → 共享梯度 | cos 0.108（R² −2.24） | **cos 0.308（R² −0.063）** | |
| chunk 前 5 帧+state → 共享梯度 | — | cos 0.257 / MLP 0.279 | 加未来帧**没有**提升 |
| chunk 全 20 帧+state → 共享梯度 | — | cos 0.163 / MLP 0.263 | 高维反而更差 |
| chunk 均值+state → 共享梯度 | — | cos 0.313 / MLP 0.294 | 略好（说明"摘要"够用） |

**经验规则（本轮验证过）**：`held-out ΔJ ≈ cos × Oracle头寸`
（共享 MLP：0.0229/0.0678 = 0.34 ≈ 它的 cos 0.31；per-frame+chunk：0.0348/0.0730 = 0.48）

---

## 6. 结论

### 已验证（可以当结论用）

1. **BPTT 计算图和解析梯度正确**（有限差分 1.74%）。
2. **纯 BPTT 能超过常向量**，前提是 **population 采样**（每步重抽）：
   - 共享 Δ MLP：test +0.0195 vs Constant +0.0141（1.4×）
   - **per-frame + chunk 全帧：test +0.0347（2 seed）vs Constant +0.0141（2.5×）**，且动作 RMS 0.032 与共享 MLP 的 0.033 相当 ⇒ **不是靠更大的修正换来的**。
3. **固定小窗口训练 = 背题**（train +0.070 / held-out +0.008），**打乱或置零 state 训练集不掉分** ⇒ 小窗口实验不能用来判断"能不能学"。
4. **共享 Δ 有结构性上限**：理想修正 94.5% 的能量在"窗口×帧"交互里，共享 Δ 只能碰 5.3%。
5. **每帧预算不能简单按 √T 分摊**：总预算对齐（每帧 0.0179）时几乎无效（+0.0070）。
6. **Oracle 头寸很大**：同一 ‖Δ‖≤0.08 预算下，逐窗口最优能把 test 抬高 **+0.0730（+5.0%，96/96 窗口全变好）**。

### 未定（必须再测）

1. **bptt1 为什么是 0 改善**（§7 的矛盾）—— 这是重训前必须解释的，否则会重踩。
2. **per-frame + chunk 只有 1~2 个 seed**，且训练曲线在 step 100 后回落 ⇒ **终点敏感**，需要更长训练 + 早停 + held-out 曲线。
3. **池子只有 3,072**（bptt1 是全部 346,729 合格帧）⇒ 扩池（12k/50k）是否还能涨，未定。
4. **历史状态（obs 口径 580 维）** 的边际收益未测；state64 空间那一版是**负收益**（+0.0313 < +0.0347）。
5. **WM → 真机**：历史实测 WM 的 |ΔJ| 高估约 **1.8×**（H=20）⇒ 真机可能只剩 +1.3%。

### 已否证

1. **"加未来帧去算一个共享 Δ"** 没用（cos 不升反降）。
2. **"用逐帧梯度预测状态依赖"在 n=256 下不可预测** 这个说法是**小样本伪影**，已撤回。
3. **"减少反常量正则/换成 FiLM 就能救共享 Δ"** 不成立（结构性 5.3% 上限）。

---

## 7. 待解矛盾（重训前最重要的一件事）

| | 数据/裁判 | 步数 | WM 回报 |
|---|---|---|---|
| `bptt1`（完整配方：critic + window_norm + grad_target + act hinge + trust，**无 cap**） | 同一份真转移、同一个 `wm_gain2`、同一个 decoder | 59（512/步） | **+1.12%**（J 1.5185→1.5354；≈ 一个常向量的水平） |
| 我的最小配方（共享 Δ，**cap 0.08**） | 同上 | 250（256/步） | held-out **+0.0195** |
| 我的 per-frame+chunk（**cap 0.08**） | 同上 | 250 | held-out **+0.0347** |

同一份数据和裁判，一个 0 一个 +3.5%。三个候选解释（都可测）：

| 假设 | 支持证据 | 判别实验 |
|---|---|---|
| **H-a 步数不够** | 我的曲线到 ~100 步才明显（+0.05），且 minibatch J 抖动 ±0.03；bptt1 只 59 步 | 跑 59 步版本 |
| **H-b 无 cap → 过冲** | 标定：ΔJ 在 ‖Δ‖≈0.08 达峰、0.16 下降、0.32 变负；bptt1 的 ‖Δ‖=0.15 | 复刻臂分 cap 开/关 |
| **H-c 辅助项互相抵消** | window_norm 按每窗口 \|ret\| 归一（小回报窗口权重被放大）、grad_target 把 Δ 拽向 η·unit(ĝ)、act hinge 直接压制动作改变 | 复刻臂逐项消融 |

---

## 8. 候选方案（请评审裁定）

> 每个候选都写明：结构 / 损失 / 采样 / 需要改的代码 / 预期 / 风险。
> 我的建议：**B 作主方案，但必须先跑 D 定位 bptt1 的失败项**（否则重训可能重复同样的坑）。

### 候选 A：共享 Δ 最小 BPTT（对照基线）
- 结构：`Δ = f(token0, state64)`，cap 0.08/窗口
- 损失：`L = −mean(J)`；采样 population（≥3,072 池，256/步，500 步）
- 代码：**已实现**（`scripts/diag_bptt_learnability.py --arms mlp`）
- 预期：test ≈ +0.0195，val ≈ +0.0229
- 价值：证明"不是只有 per-frame 能work"，并作为任何新方案的必带基线

### 候选 B：per-frame Δ + chunk 全帧（**当前最优**）
- 结构：`Δ_t = f(flatten(chunk 40×64), token_t, state64)` → (T,64)，**每帧 cap 0.08**
- 损失：`L = −mean(J)`；采样同上；500 步 + 每 25 步 held-out 早停；**3 seeds**
- 代码：**已实现**（`--arms mlp_frame_flat`）
- 预期：test +0.033~+0.036（2 seed 实测 +0.0351/+0.0344），动作 RMS 0.032
- 风险：① 输出维度 T×64、总预算 ×T，**动作 RMS 约 2× 常向量**（需核 `‖Δa‖≤1.5η` 探针）；② 只有 2 seed；③ WM→真机可能打 0.55 折

### 候选 C：B + 历史状态（obs 口径 580 维）
- 结构：B 的输入再加 `hist_10 = 10×[q(29)|qd(29)]`
- 代码：**5 行**（用 `history[:,:,:58]` 代替 hist64）
- 前提：**客户端多传一个字段**（破"客户端零改动"，约 20 行）
- 预期：test ≈ +0.031~+0.038（不确定；state64 空间那版是 +0.0313，**负收益**）
- 建议：**不要作为主方案**，只在 B 定稿后作为一次增量实验

### 候选 D：bptt1 完整配方复刻 + 逐项消融（**诊断，强烈建议先做**）
- 结构：同 bptt1（共享 Δ，cap 0）
- 损失：`λ_r·window_norm(−J) + act hinge + trust hinge + critic bootstrap + grad_target`
- 代码：需要把这几项接进诊断脚本（**约 100 行**），或直接用原 `train_joint_v2.py` 跑但用同一池/评测
- 5 个臂：完整 / −grad_target / −window_norm / +cap / −critic（并行 ≈40 分钟）
- 产出：**到底哪一项让回报归零** —— 这是重训的直接依据

### 候选 E：B + 工程加固
- B + **动作 RMS hinge**（`w=0.15, eps=0.30`，中心=底模令牌的动作）+ cap
- 目的：动作 RMS 0.032 → 压到常数级（0.017），换更保险的部署幅度
- 代价：可能损失一部分 ΔJ；需 A/B

---

## 9. 复现与文件

**代码**（worktree `/…/comimic-wt03`，分支 `harness/03-bptt-diagnosis`）

| 文件 | 作用 |
|---|---|
| `scripts/diag_bptt_learnability.py` | 全部实验：`build / exp0 / arms / gradstruct / gradframe / horizon / report` |
| `scripts/harness03_env.sh` | 运行环境（worktree 代码 + 主仓库只读数据） |
| `scripts/harness03_report.py` | 从落盘产物生成 report（含全部表格） |

**产物**（`runs/harness_03_bptt/`）

| 文件 | 内容 |
|---|---|
| `windows.pt` / `windows_meta.json` | 448 窗口（256/96/96）+ 哈希 + 奖励口径 |
| `vla_tokens.pt` | 官方 VLA 采样的 token chunk（两次采样） |
| `big/windows.pt` / `big/vla_tokens.pt` | 3,264 窗口（3,072 train）population 池 |
| `exp0.json` | 梯度路径审计 + 有限差分 + 幅度标定 |
| `gradstruct_vla_big.json` | n=3,072 的逐窗口梯度结构 + 分组 CV |
| `gradframe_big.json` / `perframe_grad_big.npz` | 逐帧梯度方差分解 + chunk 特征 CV |
| `arms_runs_*.jsonl` | 每一次臂运行的完整记录（含逐窗口回报与训练曲线） |
| `report.md` | 自动生成的报告（表格全部来自落盘数据） |

**复现命令**（摘要）

```bash
source scripts/harness03_env.sh
# ① 建窗口 + 缓存 VLA token
python3 scripts/diag_bptt_learnability.py build --out $OUT --n-train 3072 --n-val 96 --n-test 96
# ② 梯度审计 / 机理
python3 scripts/diag_bptt_learnability.py exp0     --out $OUT --n 8
python3 scripts/diag_bptt_learnability.py gradstruct --out $OUT --n-grad 3072
python3 scripts/diag_bptt_learnability.py gradframe  --out $OUT --n-grad 3072
# ③ population 臂
W=$OUT/big/windows.pt; T=$OUT/big/vla_tokens.pt
python3 scripts/diag_bptt_learnability.py arms --out $OUT --windows $W --vla-tokens $T \
    --sizes 3072 --resample --bs 256 --steps 250 --lrs 3e-3 --arms mlp_frame_flat
# ④ 汇总
python3 scripts/harness03_report.py --out $OUT
```
