# 路线评审材料：Δtoken 的两条训练路线（BPTT vs 蒸馏）

> 写于 2026-11-07 02:15。目的：把**全部设计、结构、参数、实测结果、失败证据**摊在一次，
> 供外部评审判断"该怎么改"。
> 阅读顺序建议：§1 背景 → §2 核心矛盾 → §3 已知事实（全部可复现）→ §4 路线 A → §5 路线 B
> → §6 共同结构 → §7 现在的状态 → §8 我们想被评判的具体问题。

---

## 1. 背景与目标

**部署链路（推理）**：

```
前端相机 ─► VLA（GR00T Cosmos-Reason2-2B + 1.09B DiT 头）
             输入：state46 = [关节29 | 手14 | 重力3] + 一帧 RGB + task 文本
             输出：78 维 = [64 维运动 token | 14 维手]
        ─► 服务端：token' = token + Δtoken_net(token, state64)      ← 我们唯一的改动点
        ─► 执行器（SONIC release decoder, ONNX, CPU）把 token' 解码成 29 维关节目标
        ─► PD 控制 ⊂ Isaac 物理
```

- 64 维 token 是"要做什么运动"的压缩命令；`state64 = [root_rot6d(6) | dof_pos(29) | dof_vel(29)]`。
- **推理不带世界模型**。世界模型（WM）只在**训练**里用来算"理想修正方向"。
- 客户端零改动；Δtoken 一次前向 0.105 ms（实测），相对 2.6 s/chunk 的闭环开销可忽略。

**目标**：闭环成功率。当前基线（各 4~6 episode，seed 0..4）：

| 臂 | SitSofa | VisionNavi | 合并 |
|---|---|---|---|
| base（官方微调 checkpoint-8000） | 4/6 = 66.7% | 3/6 = 50.0% | 7/12 = **58.3%** |
| v13（早期 LoRA 微调） | 2/7 = 28.6% | 2/6 = 33.3% | 4/13 = **30.8%** |

⇒ **本轮所有工作要回答的唯一问题：能不能超过 base 的 58.3%。**

---

## 2. 核心矛盾（一句话）

> **WM 给出的理想修正方向，在真机上是有用的（实测 §3.2）；但把它变成"一个网络的参数"时，
> 跨窗口求和会把每窗口各自的方向平均掉，网络只能学到那个平均值（=常数）。**

这就是早期四种参数化（LoRA / 反常量项 / state_extra / delta-token）**全部收敛到 0.08~0.17 常向量**的原因。

---

## 3. 已知事实（全部实测，附出处）

### 3.1 WM 前向是准的

| 指标 | 数值 | 出处 |
|---|---|---|
| 20 步位置误差（exc 起点） | 0.82 cm | `logs/` + p1b 文档 |
| 1 步误差（`all_transitions.pt`，口径已验证） | pos 0.05 cm / q 0.018 rad | 本文 §5.6 自检 |

### 3.2 真物理 ±δ 检验：WM 梯度方向**可用**，但**幅度高估 ~1.8×**

做法：取 N 个真实窗口 (s0, a0)，用 WM 反传得 `ĝ = ∂J/∂A`，归一化到动作 RMS δ=0.03 rad，
在**真物理（Isaac）**里从同一 s0 分别执行 `a0 ± δ·ĝ` 走 H 步，用**同一个 reward** 结算。

| 视界 | 窗口数 | 符号一致率 | **真物理变好的比例** | ΔJ 相关 | J_true 均值 | J_WM 均值 |
|---|---|---|---|---|---|---|
| H=2 | 96 | 1.000 | **1.000** | +0.964 | +0.0076 | +0.0087 |
| H=20 | 64 | 0.984 | **0.984** | +0.092 | **+0.0587** | **+0.1277** |

- 中位数比：|ΔJ_WM| 0.1053 / |ΔJ_true| 0.0589 = **1.79×**；均值比 **2.17×**。
- 脚本：`scripts/probe_wm_gradient_truth.py plan|score`、`scripts/isaac_probe_action_rollout.py`
- 数据：`runs/probe_truth/exc_dump.pt`；日志 `logs/truth_probe.log`、`logs/truth_probe_h20.log`

⇒ ① 方向可用（H=20 也可用）；② **幅度必须打 ~0.55 折**（我们取 η=0.08 = 0.15/1.8）。

### 3.3 可学性：特征 → ĝ 的 CV 余弦（n=3072，8 折，标准化 + ridge / 2×256 MLP）

**动作空间单步 `g1 = ∂r₀/∂a₀`**

| 特征 | 维度 | ridge R² | ridge cos | MLP cos |
|---|---|---|---|---|
| token | 64 | +0.010 | 0.124 | 0.117 |
| state | 71 | +0.079 | 0.335 | 0.342 |
| act0（当前动作） | 29 | +0.083 | 0.327 | 0.370 |
| **state+act0** | 100 | **+0.195** | **0.525** | **0.567** |
| ref0（参考状态） | 71 | +0.026 | 0.181 | 0.160 |
| all | 270 | +0.202 | 0.543 | 0.506 |

**token 空间单步 `∂r₀/∂token` / 20 步 `∂J₂₀/∂token`**

| 目标 | 特征 | ridge R² | ridge cos | MLP cos |
|---|---|---|---|---|
| ∂r₀/∂token | token | +0.029 | 0.194 | 0.175 |
| | state | +0.119 | 0.393 | 0.418 |
| | **state+act0** | +0.176 | 0.472 | **0.500** |
| ∂J₂₀/∂token | token | −0.022 | 0.206 | 0.236 |
| | state+token | −0.057 | 0.232 | 0.334 |
| | state+actH(651) | −0.419 | 0.215 | 0.351 |

**跨窗口"残差/均值"（L2）**：k=1 **2.835**、k=2 2.519、k=5 1.442、k=20 1.073
⇒ 最优点**不是常数**（任何 k 都不是）。

**样本量敏感性**：n=256 时所有 CV R² 为负（−0.17~−0.99）；n=3072 后转正
⇒ "R² 全 ≤0"是**小样本伪影**，不能当结论。

**结论**：可预测部分 20~35%；**token 本身几乎没信息**（cos 0.12~0.21）；
`state`（含关节位置）与**当前动作**才是主要信息源；`ref` 单独喂也没用。

### 3.4 在线（推理期现场算）的成本 —— 已否决作为交付形态

| 做法（batch=1，含一次 autograd） | 耗时 | 相对 2.6 s/chunk |
|---|---|---|
| H=2 | 231 ms | +9% |
| H=5 | 439 ms | +17% |
| H=10 | 987 ms | +38% |
| **H=20** | **1869 ms** | **+72%** |
| 只前向 H=20 | 88 ms | —— |
| **Δtoken 前向（蒸馏版）** | **0.105 ms** | ≈0% |

⇒ 在线只适合当"离线标签生成器"，不适合上线。

### 3.5 早期失败证据（路线 A 的前科）

| 轮次 | 参数化 | 结果 |
|---|---|---|
| v10/v11/v13 | LoRA 微调 | 学到常向量；跨臂 `cos(v10,v11)=0.980`、`cos(v10,v13)=0.985` |
| p1b/p1c/p1d | 反常量项 0.03/0.1/0.3 + 不同 λ_r/lr | `残差/均值(L2)` = 0.164 / 0.170 / 0.081，**全部≈常数** |
| p1e | lr 5e-4、λ_r=15、act w=0.15 | `state_sens` 型变化上来了，但探针 `‖Δa‖ = 0.724 = 4.83η`（过冲，不可部署），`残差/均值` 仍 0.105 |

---

## 4. 路线 A：BPTT（端到端报酬反传）

### 4.1 设计与代码路径

```
token' = token + Δ_θ(token, state64)
        ↓
WBC/执行器解码 → 动作 a_t
        ↓
WM 走 20 步（可微）→ 每步 reward（同一 reward 函数）→ 回报 J
        ↓
loss = −λ_r · mean_w(J_w) + act 惩罚 + trust 项 (+ 可选 ĝ 辅损)
        ↓
loss.backward()  ⇒ 梯度 **穿过 20 步 WM + 跨窗口求和** 进 Δ 的参数
```

- 入口：`diff_vla_wbc/v3_step.py::train_step`；训练器 `scripts/train_joint_v2.py`；
  launcher `scripts/launch_joint_ha_v2.sh`；env 见 §7。
- 起点集合：`runs/a_prime_sonic_exc/all_transitions.pt`（529,131 行真 Isaac 转移），`START_MAX_DRIFT=0.10`。
- WM：`runs/a_prime_sonic_exc/wm_gain2.pt`（**冻结**，`wm_refresh_every=0`）。
- `FREEZE_VLA=1`：VLA 全部 1,620,515,968 参数冻结，只训 Δ + critic（显存 15 GB/卡，6~7 s/步）。

### 4.2 本轮为治"求和→均值"做的三处改动（都在本次提交里，默认关）

| # | 开关 | 作用 | 现状 |
|---|---|---|---|
| 1 | `DELTA_TOKEN_FULLSTATE=1` | Δ 的输入从 `rot6d+dof_vel`(35) 换成 **部署 state64 全 64 维**（补上 dof_pos 29 维） | 已开 |
| 2 | `WINDOW_NORM=1`（新） | `loss = −mean_w( ret_w / |ret_w|_detach )`：每窗口按自身量级归一化 ⇒ 每窗口影响力相同，削弱大回报窗口主导 | 已开 |
| 3 | `LAMBDA_R` 5 → **2.8** | 平衡点 `‖Δa‖ ∝ λ_r / act_weight` ⇒ 幅度直接降到 ~0.55×（治 1.8× 过冲） | 已开 |

### 4.3 本轮配置（正在跑：`runs/bptt1`，02:07 起）

```
STEPS=150  NPROC=2 DDP=1  BATCH=512(全局) ACCUM=2  → 有效 1024
WM_CKPT=runs/a_prime_sonic_exc/wm_gain2.pt  TERM_OFF=1
LAMBDA_R=2.8  VAR_WEIGHT=0  ACT_WEIGHT=0.15  ACT_EPS=0.30  WINDOW_NORM=1
TRUST_CENTER=base  TRUST_EPS=0.05  TRUST_WEIGHT=0.2  CRITIC_ON=1  CRITIC_SOURCE=imagined
DELTA_TOKEN=1  DELTA_TOKEN_FULLSTATE=1  DELTA_TOKEN_LR=5e-4  DELTA_TOKEN_INIT_STD=4e-4
GRAD_TARGET_WEIGHT=0.2  GRAD_TARGET_N=8  GRAD_TARGET_ETA=0.08  GRAD_TARGET_HORIZON=20   ← ĝ 辅损（弱）
FREEZE_VLA=1  H=20  seed=0
```

### 4.4 本轮早期结果（step 18/150）

| 指标 | 数值 | 老 BPTT（参照） |
|---|---|---|
| `delta_token/norm` | 0.113（末 10 步） | p1e 0.36~0.44（过冲） |
| `delta_token/grad_cos` | 首 10 步 +0.094 → 末 10 步 **+0.142** | ≈0 |
| `delta_token/state_sens` | 1.136（首 10 1.241 → 末 10 1.013） | p1e 1.5 左右 |
| `action_var/ratio` | **NaN**（该指标只在非 delta-token 路径产生；等价物是 `state_sens`） | 0.098~0.17 |

⚠️ 说明：`state_sens = ‖g(tok,s) − g(tok,s_perm)‖ / ‖g(tok,s)‖`（把状态打乱再算一次 Δ）。
它在初始化时就 ≈1.4（因为初始 Δ 很小、被噪声主导），**所以它不是"有没有学状态"的干净指标**；
干净指标是 `grad_cos`（Δ 与 ĝ 的余弦）与探针的 `残差/均值`。

### 4.5 路线 A 的失败机制（为什么它可能永远学不到状态相关的部分）

参数梯度（对一个早期层权重）近似为

```
Σ_w  g_w ⊗ φ(x_w),    g_w = ∂J_w/∂a_w,  φ(x_w) = 该窗口的输入特征
```

- 早期 Δ 几乎不随 x 变化 ⇒ 所有窗口的有效更新方向 ≈ `mean_w(g_w)` ⇒ 一步就跳到"平均方向"；
- 到了平均方向后，状态相关那部分只剩**很小的残差梯度**，优化器容易停在平台（这就是我们四次观测到的常向量）；
- 正则化项（反常量项）只能强制"输出有方差"，不能强制"方差有用"——已实测三档（0.03/0.1/0.3）全部失败。

---

## 5. 路线 B：蒸馏（把 ĝ 当标签）

### 5.1 设计

```
老师（每步一次，detach）：
   token_base（VLA 本次前向输出）──► 解码 ──► WM 20 步 ──► reward ──► J
   ĝ = ∂J/∂token_base                                   （一次 autograd，链长 = 标签视界）
   目标标签 = η · unit(ĝ)，  η = 0.08（= 0.15/1.8，按 §3.2 的幅度高估修正）

学生（Δtoken，见 §6）：
   loss = ‖ Δ(token_base, state64) − η·unit(ĝ) ‖²  (+ 可选弱报酬项 λ_r)
   ⇒ **学生完全不穿 WM，也不做跨窗口求和**
```

- 实现：`diff_vla_wbc/v3_step.py::_grad_target_loss()`；开关 `GRAD_TARGET_WEIGHT/…`。
- 诊断指标：`grad_cos`（Δ 与 ĝ 的余弦，**逐窗口、在当前子批上算**）、
  `grad_norm_ratio`（‖Δ‖/η）、`norm`。

### 5.2 四次迭代（每次都是发现问题→修）

| 跑批 | 标签视界 k | λ_r | 标签算在哪个 token | 结果 | 结论 |
|---|---|---|---|---|---|
| gt4cut | 20 | 5 | 录制 token | step 30：norm 0.152（**1.9η**）、cos 0.06 | 报酬项压过标签 |
| gt5cut | **2** | 5 | 录制 token | step 30 停 | k=2 与 H=20 **不对齐** |
| gt6ddp | 20 | 0 | **录制 token（错）** | cos 峰值 0.11 → 掉到 0.047；norm/η → 1.08 | 标签点错 ⇒ 方向学不到 |
| **gt7fix** | 20 | 0 | **VLA 输出的 token（修好）** | **150 步跑完：cos 全程 +0.081、末 20 步 +0.095、norm/η 0.88** | 修复有效，但 cos 仍低 |

### 5.3 gt7fix 的完整分段结果（150 步，2 卡 DDP，9.9 s/步）

| 步数 | cos | ‖Δ‖ | ‖Δ‖/η |
|---|---|---|---|
| 1–10 | +0.097 | 0.043 | 0.66 |
| 11–20 | +0.024 | 0.053 | 0.70 |
| 21–30 | +0.062 | 0.053 | 0.61 |
| 31–40 | +0.104 | 0.061 | 0.74 |
| 41–50 | +0.076 | 0.057 | 0.69 |
| 51–60 | +0.084 | 0.054 | 0.70 |
| 61–80 | +0.060 | 0.069 | 0.89 |
| 81–96 | +0.103 | 0.070 | 0.88 |
| **全程** | **+0.081** | —— | —— |

产物：`runs/gt7fix/vla_000050.pt / vla_000100.pt / vla_000150.pt / vla_final.pt`

### 5.4 路线 B 现在的瓶颈：**标签太少**

| 项 | 数值 |
|---|---|
| 每步标签数 | 8 / rank × 2 rank = **16** |
| 累计标签（150 步 × 4 次跑批） | ~2,400（单次 150 步 ≈ 2,400） |
| 每个标签的成本 | 一次 20 步 rollout + 反向 ≈ **0.4 s/窗口** |
| 学生容量 | **1,247,296 参数** |

参照：同口径**线性**基线在 n≈1k 时 cos 已有 **0.26**（§3.3）；我们现在 1.2M MLP 只有 **0.10**
⇒ **学生打不过一条直线 ⇒ 是样本不够，不是数据结构上限**（大概率）。

### 5.5 计划中的改进（尚未执行）

| # | 改动 | 依据 | 成本 |
|---|---|---|---|
| 1 | **标签离线一次性生成 + 缓存**：`FREEZE_VLA=1` ⇒ token 固定 ⇒ `(state, token, ĝ)` 是静态数据集；生成 5 万条，训练变成普通监督（0.02 s/步，可多 epoch，可做 train/val 切分） | 现在每步现算，成本限制样本量 | 生成 ~1~1.5 h |
| 2 | 标签量按"学习曲线"决定：1k / 3k / 10k / 30k / 100k 各训一次看 held-out cos 何时平 | 见 §3.3 的样本量敏感性 | 逐步 |
| 3 | **输入补 `state − ref` 误差特征** | ĝ 本质是"离参考差多少→往哪修"；`ref` 单独喂 cos 仅 0.18，但**差量**没试过 | 20 行 |
| 4 | 标签视界 k：20（对齐部署 20 步 exec）vs 5（可学性更好但偏短视） | §3.3 | 一次 A/B |

### 5.6 口径自检（已做）

- `all_transitions.pt` 的 state 已是 standard 序，喂 WM 一步误差 **0.05 cm / 0.018 rad**；
  加置换反而恶化（0.60 cm / 0.40 rad）⇒ **不要置换**。
- 闭环 `dump/steps.jsonl` 的 `q` 与运动库 ref **对不上**（一步 0.415 rad；置换后 1.48 rad）
  ⇒ 用闭环录制数据评估 WM/漂移的历史结论**一律作废**（我们已撤回一组"漂移分解"数据）。
- 运行时参考库：`isaac_replay_online.sh` 的 `LIB` 默认 `runs/isaac_motion/ha_cmd/packed.pt`，
  但用 `frame` 索引两种库都匹配不上（最近邻 0% 命中）⇒ **运行时的参考到底是哪份数据仍未查清**。

---

## 6. 两条路共用的结构（完全相同）

| 项 | 取值 |
|---|---|
| 网络 | `DeltaTokenNet`：`(token 64 ⊕ feats) → Linear → SiLU → Linear → SiLU → Linear(64)` |
| 输入 | 第 1 刀前：`token(64) + [rot6d(6)+dof_vel(29)] = 99`；现在：`token(64) + state64(64) = 128` |
| 隐藏层 | 2 × 1024 |
| 输出 | 64（Δtoken） |
| 参数 | 99 输入：1,217,600；128 输入：**1,247,296** |
| 末层初始化 | `normal(std=4e-4)`（≈零，保证第 0 步与底模接近；`init_std=0` 可得严格恒等） |
| lr | 5e-4（Adam，独立参数组） |
| 幅度约束 | `apply_norm_cap(Δ, cap)`（tanh 软上限；`cap=0` ⇒ 恒等），训练/部署同一函数 |
| 施加点 | 服务端，`token' = token + Δ`，广播到整段 chunk；**客户端零改动** |
| 部署成本 | 0.105 ms/次 |
| 元信息落盘 | ckpt 里存 `delta_token_state/meta(cap)`，服务端按 meta 自动加载 |

---

## 7. 现在的运行状态（02:15）

| 跑批 | 路线 | 状态 | 产物 |
|---|---|---|---|
| `gt7fix` | B 蒸馏（k=20, λ_r=0, 修好的标签） | ✅ 完成 150 步 | `runs/gt7fix/vla_final.pt` |
| `bptt1` | A BPTT（λ_r=2.8 + window_norm + fullstate + 弱 ĝ 辅损 0.2） | 🔄 跑到 step 18/150（约 02:33 完成） | `runs/bptt1/` |
| 早期对照 | —— | 已完成 | `runs/gt4cut`(k20/λ5) `runs/gt5cut`(k2) `runs/gt6ddp`(bug) `runs/p1b~p1e` |
| 真机 ±δ | 判据实验 | ✅ | `logs/truth_probe*.log`、`runs/probe_truth/` |
| 可学性 | 判据实验 | ✅ | `docs/GRAD_OPTIMUM_WHAT_IT_DEPENDS_ON_20261106.md`、`runs/gradtok2/` |

**验收流程（跑完必须做）**：

```bash
# ① 探针 30 s：Δ 的落地量级 / 残差·均值
python3 scripts/probe_ft_landing.py --arms base= new=runs/<run>/vla_final.pt \
    --n-noise 3 --max-req 24 --device cuda:0     # 判据 ‖Δa‖ ≤ 1.5η
# ② 闭环（每臂 2 任务 × 1 seed，约 15 min；必须用 py312 版 runner + sonic env）
TASK=SitSofa HEAD=runs/<run>/vla_final.pt ... bash scripts/r3_sr_cell.sh
# 判据：成功率 ≥ base（SitSofa 66.7% / VisionNavi 50.0%）
```

已知坑：`r3_closedloop_ab.sh` 的 `SRV_PY` 含 cp311 的 skimage（本机 py3.12 会崩）；
`r3_sr_cell.sh` 不自动导出 sonic 栈环境变量（缺了会**静默回落**到 SuperMimic provider）。

---

## 8. 请评审回答的问题（关键）

1. **路线 A（BPTT）还有救吗？**
   在"共享参数 + 跨窗口求和"的前提下，是否存在能让网络学到**状态相关**（而非均值）的标准做法？
   （per-window gradient balancing / PCGrad / 每窗口独立 head + 混合权重 / FiLM 条件化 / 先用蒸馏初始化再 BPTT …）
   还是说这条路在"目标只有 20~35% 可被输入解释"的事实下注定停在均值？

2. **路线 B 的瓶颈判断对不对？**
   学生（1.2M）held-out cos 0.10 vs 线性基线 0.26（n≈1k）——这是"样本不够"还是"标签分布不同"
   （我们的标签是在 **VLA 输出的 token** 上算的，而可学性实验在**数据集 token** 上）？
   应该先加标签量、还是先补特征（`state − ref`）、还是先换标签视界？

3. **标签生成的性价比**：现在 8 窗口/rank/步。是应该（a）离线一次性生成 5 万条缓存起来，
   （b）继续在线生成但加大到 64/rank，（c）用 IoU/重要性采样挑"高价值窗口"先标？
   有没有我们没想到的更省的方案（例如用小模型代替 WM 做 teacher、或用有限差分代替 autograd）？

4. **对齐问题**：标签视界 20（与部署 exec 20 步一致、但可学性差 cos 0.23~0.35）
   vs 5（可学性好、但偏短视）。在"每 chunk 重规划一次"的用法下，哪个更合理？

5. **验收口径**：我们只有 2 任务 × 1~4 seed（每 episode ~500-1000 步）。
   在 base 58.3% 这个量级上，要多少 episode 才能看出 +10 个点的真实提升（power analysis）？
   有没有更便宜的中间指标（例如真机 ±δ 那种"真物理变好比例"）可以先筛？

6. **一个更根本的疑问**：既然 §3.2 证明"沿 ĝ 走 20 步能让真机回报 +14%（63/64 窗口变好）"，
   而蒸馏只能拿到 cos≈0.3 的近似——**是否应该优先把在线修正做成"每 N 个 chunk 跑一次"的
   低频版本**（成本可摊薄），而不是继续蒸馏？如果有更好的折中，请指出。

---

## 9. 附：可复现命令

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

# 路线 A（BPTT，本轮配置）
RUN=bptt1 STEPS=150 BATCH=512 ACCUM=2 NPROC=2 DDP=1 DDP_GPUS=0,1 GPU=0 MASTER_PORT=30071 V3=1 \
 WM_CKPT=runs/a_prime_sonic_exc/wm_gain2.pt TERM_OFF=1 LAMBDA_R=2.8 VAR_WEIGHT=0 \
 ACT_WEIGHT=0.15 ACT_EPS=0.30 WINDOW_NORM=1 TRUST_CENTER=base TRUST_EPS=0.05 TRUST_WEIGHT=0.2 \
 CRITIC_ON=1 CRITIC_SOURCE=imagined DELTA_TOKEN=1 DELTA_TOKEN_FULLSTATE=1 DELTA_TOKEN_LR=5e-4 \
 DELTA_TOKEN_INIT_STD=4e-4 GRAD_TARGET_WEIGHT=0.2 GRAD_TARGET_N=8 GRAD_TARGET_ETA=0.08 \
 GRAD_TARGET_HORIZON=20 FREEZE_VLA=1 START_MAX_DRIFT=0.10 INIT_VLA= \
 EXTRA="--real-transitions runs/a_prime_sonic_exc/all_transitions.pt --boot-one-sided" H=20 \
 bash scripts/launch_joint_ha_v2.sh

# 路线 B（蒸馏，本轮配置：把上面 GRAD_TARGET_WEIGHT 换成 1.0、LAMBDA_R=0、去掉 WINDOW_NORM）

# 离线标签 dump（给"加厚标签"用）
J1_GRAD_DUMP=runs/gradump2 J1_GRAD_DUMP_TOK=1 STEPS=24 BATCH=128 ... （同上其余）
```

**关键代码位置**

| 内容 | 位置 |
|---|---|
| Δtoken 网络 | `diff_vla_wbc/delta_token.py` |
| Δ 施加 + 输入特征 | `diff_vla_wbc/v3_step.py::_delta_token_delta / _delta_token_features` |
| 稠密 ĝ 监督 | `diff_vla_wbc/v3_step.py::_grad_target_loss` |
| 每窗口归一化 | `diff_vla_wbc/v3_step.py::_red`（`window_norm`） |
| BPTT 路径 | `diff_vla_wbc/v3_step.py::train_step`（rollout → reward → backward） |
| 真机 ±δ 判据 | `scripts/probe_wm_gradient_truth.py` + `scripts/isaac_probe_action_rollout.py` |
| 在线修正（备用，未部署） | `diff_vla_wbc/online_grad.py` |
| 交叉验证脚本 | `scripts/probe_grad_mlp.py` |
