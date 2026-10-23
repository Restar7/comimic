# CoMimic 论文修改方案（基于当前代码实现）

> 输入：`main.tex` 工作稿（v0.1，2026-10；用户提供，仓库内暂无源码）。
> 依据：`docs/FINAL_HANDOVER.md`、`docs/SONIC_TOKEN.md`、`docs/SONIC_DATASET_PIN.md`、
> `diff_vla_wbc/`、`third_party/SuperMimic-Reproduced/`、`groot_repro/`。
> 结论先行：**论文当前写的是"adapter 路线"（64→10×73 goal→WBC），而代码实现的是"token 直连路线"（64 token 直接进 WBC，无 adapter、无 goal 空间）。必须整体改写，否则方法与实验对不上。**
> 另外：**"端到端联合训练"这一阶段目前只完成了接口与单侧训练，Stage C 尚未实跑**；因此"first truly end-to-end" 只能作为**有条件的定位**，要等 Stage C 出梯度证据后才能写成结论。

---

## 0. 五个最严重的问题（按优先级）

| # | 问题 | 后果 | 处理 |
|---|---|---|---|
| 1 | **Method 写的是 adapter 路线**（64→10×73→WBC），代码是 **token 直连**（`PolicyCfg.token_dim=64`，`Policy.forward(..., interface="token")`，无 adapter） | 方法、公式、图、实验全部对不上；审稿人一跑代码就发现 | 把 §4.2 整体重写为 "token-native interface"；adapter 降级为 baseline B3 |
| 2 | **维度写错**：goal 73-D / obs 99-D 是 Motphys Uni-SuperMimic 的口径；当前代码是 **goal 68-D / obs 93-D（历史 930）** | 公式 (2)(5)、附录 shape 表、Eq (6) 全部错误 | 统一为 68/93/930；或明确声明用 73/99 的哪个 commit（不能混） |
| 3 | **"端到端"被写成已实现**，但 Stage C（execution loss → WBC → VLA）尚未跑 | 核心 claim 无证据，直接拒稿风险 | 改为"implementation status"分级：Stage A/B 已跑，Stage C 待跑；"first end-to-end" 挂在 Stage C 之后 |
| 4 | **数据数字过时**：BONES 写 142,220 / 288 h；实际保留 **129,943 clips / 47,475,680 帧 / 263.75 h**；VLA 数据 ft318 是 **318 eps / 473,876 帧 / 2.633 h**，任务名也不是 apple/carton/cup/handover | 可复现性与实验协议不成立 | 用 §4 的修正表替换 |
| 5 | **"first whole-body VLA" 表述过宽**：SONIC/GR00T 已是 whole-body VLA 栈 | 被审稿人用 SONIC/GR00T 直接反驳 | 改成"first to backpropagate execution feedback through WBC+dynamics into the flow head"（范围限定、可辩护） |

---

## 1. Idea 定位（用户最关心的部分）

### 1.1 一句话电梯稿（推荐主版本）

> **CoMimic is the first humanoid whole-body VLA that is trained end-to-end through a differentiable whole-body controller and a learned dynamics model — closing the last gap between manipulation VLAs and whole-body VLAs by turning execution feedback into gradients on the action head.**

中文口径：

> CoMimic 是**第一篇真正意义上端到端训练的人形全身 VLA**：执行反馈穿过可微全身控制器与学习到的动力学模型，直接回传到 flow-matching 动作头。它消掉了操作 VLA 与人形全身 VLA 之间的**最后一道 gap**——动作头不再只对"动作标签/token"做模仿，而是对"身体能不能执行"负责。

### 1.2 "最后一道 gap" 到底指什么（必须写清楚，否则显得空）

现有两条路线各自缺一块：

1. **操作 VLA**（π0/GR00T 等）：动作头直接输出关节/末端动作，**执行误差在训练中不可见**；到了人形上，动作是否可跟踪、是否平衡，全靠下游控制器兜底。
2. **人形全身 VLA**（SONIC 等）：VLA 输出 motion token，**冻结的 decoder/WBC** 负责执行；VLA 只对 token 做模仿，token 与"身体实际执行"之间的转换是**单向、不可微、不可学习**的。Uni-SuperMimic 路线则额外插入一个 **goal-space adapter**，仍然是把语义 token 投影到物理 goal，再由冻结 WBC 执行。

CoMimic 的做法：**让 token 本身就是控制接口**（不换空间、不学 adapter、不查表解码），并让 execution loss 穿过 WBC 与动力学 surrogate 回到 VLA。于是"语义动作 → 物理执行"这条链上**不再有不可微的转换层**。

> 这就是"最后一道 gap"的精确含义：不是"人形没有 VLA"（SONIC/GR00T 有），也不是"没人做分层"（很多），而是**没有人把执行反馈接到人形全身 VLA 的动作头上**。

### 1.3 三个贡献 bullet（可直接替换论文 Intro 末尾）

1. **Token-native whole-body interface.** We show that a flow-matching VLA's 64-D motion token can *directly* condition a whole-body controller, eliminating both the hand-designed SONIC decoder and the learned physical-goal adapter used in prior hierarchical stacks; the interface is bit-exact with the deployed tokenizer and runs at 50 Hz.
2. **Execution-aware joint training.** We backpropagate a differentiable execution objective through the controller and a learned dynamics surrogate into the flow-matching action head, while updating the dynamics model from real simulator transitions in an alternating schedule.
3. **Open, reproducible G1 stack + diagnostics.** We release the data pipeline (BONES-SEED pretraining, 318-episode SONIC fine-tuning set), the tokenizer parity checks, and a model-bias / gradient-fidelity protocol that measures when execution gradients help and when they exploit the surrogate.

### 1.4 不能怎么写（审稿人必抓）

- ❌ "the first whole-body VLA" / "the first humanoid VLA" —— SONIC、GR00T 已是。
- ❌ "end-to-end" 不加限定 —— 若 Stage C 未跑，等于虚假声明；若跑了，也建议写 "end-to-end **training**"（部署仍是分层执行）。
- ❌ "removes the need for a WBC" —— 不，WBC 还在，只是**可微、可联合训练**。
- ❌ "differentiable simulator" —— 我们用的是 **learned surrogate**，不是可微物理引擎。
- ✅ 推荐措辞："to our knowledge, the first humanoid whole-body VLA whose action head is trained with gradients that pass through a differentiable controller and a learned dynamics model" + 脚注 "concurrent work may exist; claim scoped to published systems as of Oct 2026."

### 1.5 建议的 Abstract（可直接替换）

```latex
\begin{abstract}
Flow-matching vision-language-action (VLA) policies can generate expressive humanoid motion commands, but existing humanoid stacks train the action generator and the whole-body controller separately: the VLA imitates motion tokens, while a frozen decoder or a learned goal-space adapter absorbs execution error. Execution quality therefore never reaches the action head, leaving a persistent gap between manipulation VLAs, which act directly, and whole-body VLAs, which act only through an opaque interface.
We propose \Method, a framework that closes this gap by making the motion token itself the control interface and by training the VLA end-to-end through a differentiable whole-body controller and a learned dynamics surrogate. A frozen, bit-exact motion tokenizer defines the interface; a token-conditioned controller maps the token and a proprioceptive history to 29 body actions at 50\,Hz; and short imagined rollouts of a learned dynamics model let execution losses be backpropagated through the controller into the flow-matching action head. The world model is fitted on simulator transitions and frozen during policy updates, so execution gradients never differentiate through the rigid-body simulator. Motion-capture pretraining supplies whole-body skills, and a 318-episode humanoid manipulation dataset supplies task-conditioned VLA supervision; a separate 14-D hand channel remains explicit rather than being absorbed into the body model.
We specify the architecture, the alternating training schedule, and a falsifiable evaluation protocol with same-budget baselines, component ablations, and surrogate-gradient validity checks. \textbf{This manuscript is a methods-and-experiments blueprint: no empirical improvement or task success is claimed before the planned experiments are completed.}
\end{abstract}
```

### 1.6 建议的 Intro "gap" 段（替换现在的第 3 段附近）

```latex
Existing humanoid systems resolve the interface problem in one of two ways, and both leave execution outside the action head. SONIC-style stacks learn a shared motion token and a universal decoder, then fine-tune a VLA to predict tokens; the decoder is frozen, so token errors that are easy to imitate but hard to execute are never corrected by task or tracking feedback. Adapter-style stacks insert a learned map from tokens to a physically interpretable goal window and rely on a pretrained goal-conditioned controller; the projection is trained by supervised goal regression, not by whether the resulting motion is executable. In both cases the last conversion between semantics and body remains either hand-designed or supervised-only. \Method removes that conversion: the token is the native input of a differentiable controller, and the controller's execution loss is differentiated back into the flow head. To our knowledge this is the first humanoid whole-body VLA trained with execution gradients, and the first that needs neither a hand-designed decoder nor a goal-space adapter at inference.
```

---

## 2. 与当前代码的逐项对照（必须按这张表改）

| 论文写法 | 当前代码事实 | 证据 | 处理 |
|---|---|---|---|
| VLA 输出 40×64 token + 40×14 hand | ✅ 对 | `groot_repro/repro_infer.py` 契约；`modality.json` | 保留；补"132 维 padded、按 processor 键切片" |
| adapter：64→10×73 goal | ❌ 代码主路径**没有 adapter**；`diff_vla_wbc/adapter.py` 是 40×64→10×**68**，且仅作 fallback | `diff_vla_wbc/adapter.py: GOAL_FEATURE_DIM=68` | 主路径改为 token 直连；adapter 降级 baseline B3 |
| WBC 输入 obs 10×99 + goal 10×73 | ❌ 实际 obs **10×93 = 930**；goal **10×68 = 680**（critic 用） | `world_model/heads.py: OBS_TERMS=(29,29,3,3,29)`；`GOAL_INTERFACE.md` | 统一 93/930、68/680；删除 99/73 |
| 控制器输出 29 维 | ✅ 对 | `Policy` decoder 输出 29 | 保留 |
| 14-D hand 由 `π_hand(c_t,Z_t)` 预测 | ⚠️ VLA 已直接输出 7+7 hand；`HandStream` 直接执行；`hand_head.py` 是可选自建头 | `repro_infer.py`；`diff_vla_wbc/hand_stream.py` | 主路径写"VLA 直出 14-D + HandStream 执行"；自建头降为可选 |
| Stage C 端到端联合训练 | ⚠️ **接口已通，Stage C 未实跑**；已跑的是 VLA 监督微调 + SuperMimic token 条件训练 | `logs/e9_bones_seed_token*`；`docs/FINAL_HANDOVER.md` | 明确"已实现/待跑"；claim 挂后 |
| BONES：142,220 条 / 288 h | ⚠️ 142,220 是**原始**；过滤后 **129,943 clips / 47,475,680 帧 / 263.75 h** | `FINAL_HANDOVER.md` §2.1 | 两个数都写清（raw vs usable） |
| VLA 数据：未写具体 | 应写 **ft318：318 eps / 4 tasks / 473,876 帧 / 2.633 h / 50 Hz** | `data/sonic_datasets.json` | 补进 §5.2 |
| 四任务：apple / carton / cup / handover | ⚠️ ft318 实际任务：meeting_room_milk_basket（89）、office_red_cup_drawer（70）、kitchen_plate_milk_transfer（65）、diningroom_can_low_chair（94） | `data/sonic_datasets.json` | 用真实任务名；若保留 apple/carton 场景，需声明它们是 MuJoCo 场景而非 ft318 数据 |
| 73-D 定义：3+6+3+3+29+29 | ❌ 代码 68-D 定义：关节位置 29 + 关节速度 29 + 重力 3 + 角速度 3 + 线速度 3 + 高度 1 | `GOAL_INTERFACE.md` | 若保留 68，就写 68 的定义；不要用 6D 旋转公式 |
| WBC 预训练：LAFAN1 40 clips + BONES | ⚠️ 本地没有 LAFAN1；BONES 管线已就绪 | `FINAL_HANDOVER.md` | 改为 BONES 为主；LAFAN1 只在有数据时写 |
| 世界模型 K_m=8 | ✅ 对（window=8） | `wm_trainer.py` | 保留 |
| flow steps K∈{4,8} | ✅ 基座 checkpoint `num_inference_timesteps=4`（K=4） | `GR00T-N1.7-3B/config.json`；`gr00t_fm.py` | 写 K=4，消融再加 8/16 |
| 实现环境未写 | torch 2.14.1+cu126 / warp 1.17 / mjlab 1.6 / mujoco 3.16；4090 + H200 脚本 | `FINAL_HANDOVER.md` §1 | 补 reproducibility 段 |

---

## 3. 推荐的最终架构叙述（与代码一致）

### 3.1 一张图的文字描述（给 Fig. 1 改 caption 用）

```text
VLA (GR00T N1.7, frozen VLM + flow action head)
  └─ 40 × {motion_token(64), left_hand(7), right_hand(7)}   @ 50 Hz
        │
        ├── motion_token(64) ──────────────► SuperMimic policy (token-conditioned)
        │                                      input: [token(64) ‖ obs_hist(10×93)]
        │                                      output: 29 body actions
        │                                            │
        │                                            ▼
        │                                   differentiable dynamics surrogate F_φ
        │                                   (71-D robot state, 8-step window)
        │                                            │
        │                                            ▼
        │                                   execution loss → ∂L/∂token → ∂L/∂θ_VLA
        │
        └── hands(7+7) ────────────────────► HandStream (PD, ctrl[22:29], [36:43])
```

关键：**图里不能出现 adapter / physical-goal 转换**（那是 baseline B3）。

### 3.2 Method §4.2 重写（替换整个 adapter 小节）

```latex
\subsection{Token-Native Whole-Body Interface}
Prior hierarchical stacks convert the VLA's motion token into either a decoded
action (SONIC) or a physical goal window (adapter-based). We instead make the
token the native input of the controller. Let $z_t\in\R^{64}$ be the token at
low-level step $t$ and $o_{t-9:t}\in\R^{10\times 93}$ the proprioceptive history
(joint position offset, joint velocity, body angular velocity, projected gravity,
previous action; $10\times 93=930$). The controller computes
\begin{equation}\label{eq:controller}
u_t=\pi_\psi\!\left(z_t,\;o_{t-9:t}\right),\qquad
q_t^{\rm cmd}=q^{\rm default}+\bm\alpha\odot u_t,
\end{equation}
with 29 body actions and SuperMimic's joint-order/actuator-scale contract.
No projection, no goal window, and no decoder are used at inference.
The interface is bit-exact: the frozen tokenizer reproduces the deployed ONNX
encoder to the last bit (offset-corrected, FSQ on a $1/16$ grid), so the VLA's
training targets and the controller's inputs are the same object.
A goal-space adapter $D_\eta:\R^{64}\to\R^{10\times 68}$ is retained only as a
baseline (B3), not as part of \Method.
```

### 3.3 训练分阶段（与代码状态一致）

| Stage | 内容 | 代码状态 |
|---|---|---|
| A | SuperMimic WBC + WM 在 BONES-SEED 上预训练（token 条件，B 模式 token） | ✅ 已跑（4090 20k 曲线健康；H200 脚本就绪） |
| B | VLA 在 ft318 上监督微调（motion_token + hands） | ✅ 既有 GR00T 微调（仓库外，契约已验证） |
| C | execution loss → WBC → VLA 的联合训练 | ⚠️ 接口已通，**未实跑** |
| D | 对象感知扩展 | ❌ 未实现 |

> 论文里必须按这个表写"implementation status"，不能把 C 写成已完成。

### 3.4 与"first end-to-end"的对应关系

- Stage A/B 完成 ⇒ 可以写 **"token-native interface"** 与 **"no adapter/decoder"**（这是已有代码支撑的贡献）；
- Stage C 完成（能证明 `∂L_exec/∂θ_VLA ≠ 0` 且联合训练有增益）⇒ 才能写 **"first end-to-end"**；
- 在此之前，abstract 最后一句保持 "no empirical improvement claimed"，intro 用 "we target / we propose"。

---

## 4. 数值与事实修正表（直接替换论文里的数字）

| 项 | 论文现写 | 改为 |
|---|---|---|
| BONES 原始 | 142,220 / ~288 h | 142,220 CSV（raw），`g1.tar.gz` 23,499,973,647 B |
| BONES 可用 | 未写 | **129,943 clips / 47,475,680 frames / 263.75 h**（50 Hz 重采样后；`bones_seed_compact.pt` 6.28 GiB） |
| VLA 数据 | 未写 | **ft318：4 tasks / 9 sub-datasets / 318 episodes / 473,876 frames / 2.633 h / 50 Hz / 31 cols** |
| 任务名 | apple, carton, cup, handover | meeting_room_milk_basket（89）、office_red_cup_drawer（70）、kitchen_plate_milk_transfer（65）、diningroom_can_low_chair（94） |
| VLA 输出 | 40×64 + 40×14 | 40×{64,7,7}；checkpoint `max_action_dim=132`（padded），按 processor 键切片 |
| WBC obs | 10×99 | **10×93（930）**：`joint_pos(29)+joint_vel(29)+base_ang_vel(3)+projected_gravity(3)+prev_action(29)` |
| WBC goal（critic） | 10×73 | **10×68（680）**（若保留 adapter baseline，adapter 输出也是 68） |
| 控制器输入 | goal + history | **token + history**（主路径）；goal 只给 critic |
| hand | π_hand 预测 | VLA 直出 7+7；`HandStream` 执行（ctrl `22:29` / `36:43`，右手=左手镜像） |
| 世界模型 | 未写维度 | 71-D canonical state；window 8；`nse1/nse8` 监控 |
| 环境 | 未写 | Python 3.12 / torch 2.14.1+cu126 / warp-lang 1.17.0 / mujoco 3.16 / mjlab 1.6.0；单卡 4090 + 4×H200 离线脚本 |

---

## 5. 逐节修改清单

### Abstract
- 删掉 adapter / physical goal / 73-D / 99-D；
- 改为 token-native + execution gradients；
- 保留 "no empirical improvement claimed"；
- 若 Stage C 未跑，不要写 "we demonstrate"。

### Introduction
- 第 2 段：把 "we insert a learnable physical-goal adapter" 改为 "we remove the adapter"；
- 第 3 段：用 §1.6 的 gap 段替换；
- contributions：用 §1.3 的三条替换；
- 加入 "why now" 一句：GR00T 的 token 已足够通用 + SuperMimic 的 WM 已可微 + 二者接口可对齐。

### Related Work
- SONIC：写清它是 "VLA + frozen decoder" 的分层，不是端到端；
- GR00T-N1.7：写清它是 VLA 架构，执行层不在训练图里；
- Uni-SuperMimic：写清它是 goal-conditioned WBC + adapter 路线（73/99 口径），与本文 token-native 不同；
- SuperMimic：可微 WM + BPTT，但**没有 VLA**；
- CoMic / ImMimic：joint motion+task，但不是 flow VLA、不是 humanoid whole-body；
- 加一句 "concurrent work" 免责。

### Problem Formulation
- 把 $G_t\in\R^{10\times73}$ 改为 token $z_t\in\R^{64}$ + obs $o\in\R^{10\times93}$；
- 把 "WBC receives goal" 改为 "controller receives token; critic receives goal during training"；
- 保留 simulator/surrogate 的形式化。

### Method
- §4.1 flow head：保留（**K=4**，GR00T N1.7 基座 `num_inference_timesteps=4`）；
- §4.2：用 §3.2 替换（token-native）；
- §4.3：WBC 写成 token-conditioned；删除 goal 输入；hand 写成 VLA 直出 + HandStream；
- §4.4：WM 保留；补 71-D state、8-step、`nse1/nse8`；
- §4.5：Eq (9) 链式法则改为
  $\partial L/\partial\theta = (\partial L/\partial \hat X)(\partial \hat X/\partial u)(\partial u/\partial z)(\partial z/\partial\theta)$，
  **没有 $\partial G/\partial Z$**；
- §4.6：算法保留，但 Stage B 改成 "VLA supervised fine-tune on ft318 tokens+hands"；Stage C 标注 "this paper evaluates it"（若未跑则写 "we specify"）。

### Experiments
- Baselines 改为：
  - B1 SONIC hierarchy（GR00T + frozen `model_decoder.onnx`）；
  - B2 two-stage token-native（GR00T + frozen SuperMimic，无 BPTT）；
  - B3 adapter/goal-head baseline（64→10×68→SuperMimic）；
  - B4 CoMimic robot-only（execution gradients）；
  - B5 CoMimic object-aware（未实现则明确标注 "not evaluated"）。
- 数据：用 §4 修正表；
- 指标：补 token 重建误差（token MSE / 1/16 网格命中率）、WBC tracking（$e_q,e_p$、fall）、任务成功率、latency、GPU-h；
- 统计：3 seeds × 50 episodes/task（或写实际数）；
- 补 "same VLA checkpoint for all baselines"。

### Results
- 保持模板；
- 新增一段 "Interface fidelity"：tokenizer parity（逐位一致）、50 Hz 闭环、hand 通道；
- 新增 "Implementation status" 小表（A/B/C/D），避免审稿人误读。

### Limitations
补三条：
1. training-time token 由 GT 参考 + 仿真状态现算（B 模式），部署时用 VLA 预测 token，**分布差未量化**；
2. 世界模型是 robot-only，对象/接触梯度需要显式扩展，当前未实现；
3. Stage C 的梯度只对 surrogate 成立，方向一致性需要用 simulator finite-difference 验证（已有协议）。

### Appendix shapes 表
| Signal | 论文现写 | 改为 |
|---|---|---|
| body tokens | [B,40,64] | ✅ 不变 |
| hand chunk | [B,40,14] | ✅ 不变 |
| physical goal | [B,10,73] | [B,10,68]（仅 baseline） |
| decoded goal | [B,40,730] | 删除（主路径无）；baseline 为 [B,40,680] |
| obs history | [B,10,99] | **[B,10,93] → [B,930]** |
| body action | [B,29] | ✅ 不变 |
| token→policy input | — | **[B,64+930=994]** |

---

## 6. 需要补的实验（按优先级）

1. **Stage C 最小可行性**：一个 batch，验证 `∂L_exec/∂θ_VLA` 非零、有限；记录梯度范数；证明"端到端"不是口号。
2. **A/B 解码器对比**：同一段 VLA token，`model_decoder.onnx` vs SuperMimic policy，在 ft318 四个任务上的成功率/跟踪误差。
3. **token 重建**：VLA 预测 token vs 数据 token（MSE、1/16 网格命中率、按任务分解）。
4. **WBC 预训练消融**：BONES full vs subset vs no-pretrain（Table 2）。
5. **模型偏差诊断**：surrogate 方向 vs simulator finite-difference（cosine、符号一致率），至少 3 个状态 × 8 个动作方向。
6. **手部指标**：指尖误差、抓握成功率（需要对象状态）。
7. **对象感知扩展**：先验证 robot-only 无法产生对象梯度，再决定是否实现。

---

## 7. Claim ladder（写论文时的诚实分级）

| 级别 | 可以写 | 前置条件 |
|---|---|---|
| ✅ 现在 | token-native interface；无需 adapter/decoder；tokenizer 逐位一致；数据管线；训练协议 | 代码 + 单测 |
| ✅ 现在 | "we specify / we propose" 的 Stage C 方法 | 无 |
| ⚠️ Stage C 后 | "first end-to-end humanoid whole-body VLA"（加 "to our knowledge"） | 梯度证据 + 联合训练曲线 |
| ⚠️ 实验后 | "execution gradients improve success over two-stage baselines" | 同预算 A/B + 置信区间 |
| ❌ 永远别写 | "first whole-body VLA"；"removes WBC"；"differentiable simulator"；"SOTA"（无对照） | — |

---

## 8. 术语与符号统一

| 符号 | 含义 | 值 |
|---|---|---|
| $z_t$ | motion token | 64 |
| $o_{t-9:t}$ | proprio history | 10×93 = 930 |
| $g_t$ | critic goal | 10×68 = 680 |
| $u_t$ | body action | 29 |
| $h_t$ | hand action | 14（7+7） |
| $L$ | VLA chunk | 40 |
| $H$ | imagined horizon | 32（默认），消融 {4,8,16,32} |
| $K$ | flow steps | 4（基座配置；消融 {4,8,16}） |
| $K_m$ | WM window | 8 |
| $d_z$ | token dim | 64 |

- 全文统一用 "token"，不要 "latent code" / "motion embedding" 混用；
- "adapter" 只出现在 baseline B3；
- "physical goal" 只出现在 B3 与 critic 描述；
- "decoder" 指 SONIC 的 `model_decoder.onnx`（B1）或 SuperMimic policy（本文），首次出现时定义清楚。

---

## 9. 提交前 checklist

- [ ] Abstract/Intro/Method 中不再出现 73/99/730/adapter（除非在 B3 baseline 段）
- [ ] 所有维度与 `FINAL_HANDOVER.md` §2/§3 一致
- [ ] Stage A/B/C/D 状态表已写，未完成项明确标注
- [ ] "first end-to-end" 有脚注限定 + 梯度证据引用
- [ ] BONES 数字区分 raw 142,220 与 usable 129,943/263.75 h
- [ ] ft318 任务名、episode 数、帧数、时长与 `data/sonic_datasets.json` 一致
- [ ] tokenizer parity、FSQ 1/16、offset +1 写进附录
- [ ] hand 通道 ctrl 布局写进附录（22:29 / 36:43，右手镜像）
- [ ] 环境版本（torch/warp/mjlab/mujoco）写进 reproducibility
- [ ] 所有表格由 `results_values.tex` 宏填充，正文无手写数字
- [ ] 没有 "SOTA"、"first whole-body VLA"、"differentiable simulator" 等过度表述

---

## 10. 建议的新标题（可选）

- 现标题：*CoMimic: Differentiable Whole-Body Execution Feedback for Flow-Matching Vision-Language-Action Policies*（可以，偏方法）
- 更贴 idea：
  - *CoMimic: Closing the Last Gap Between Manipulation and Whole-Body VLAs*
  - *CoMimic: End-to-End Humanoid Whole-Body VLA via Differentiable Execution Feedback*
  - *From Tokens to Bodies: End-to-End Training of Humanoid Whole-Body VLAs*

推荐第二个：既点出 "end-to-end humanoid whole-body VLA"，又不与 SONIC/GR00T 的 "first whole-body VLA" 冲突。
