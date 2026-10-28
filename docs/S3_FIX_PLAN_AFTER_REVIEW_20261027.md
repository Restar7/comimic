# S3 联合微调：完整修改意见（评审结合代码核实版，2026-10-27）

> 本文回答三个问题：**① 外部评审的 4 条结构性意见，哪些成立、改什么；② 我们的 collect A/B 各自彻底修哪些；
> ③ 完整改动清单与验收判据。**
> 证据等级：✅ = 已对代码逐行核实（附 `文件:行号`）；🟡 = 需上机确认。

---

## 0. 结论摘要

| 层面 | 判断 |
|---|---|
| 工程 / 口径 | ✅ **没做错**（冻结执行器、codec.decode、K=4、hold/label/ha_cmd、39 项启动审计、64 窗口配对评测）|
| 算法 | ❌ **评审指出的 4 处实质问题全部成立**（① ② ③ ④），已逐条核实到代码行 |
| 我方的评审文档 | ⚠️ **有 3 处要更正**（P2 归因、P6 归因、P4 判据），见 §3 |
| 目标层面 | ⚠️ 评审的核心判断成立：**当前奖励衡量"跟得像不像示范"，不含"任务做没做成"**；可达 headroom 远小于 oracle 天花板 |

**一句话**：**口径没错、算法有 4 处错、文档有 3 处归因错。** 下面的清单按"先便宜、先判生死"排序。

---

## 1. 评审 4 条的核对结论与具体改法

### 1.1 ① VLA 观测不到它要纠正的状态

**代码核实（✅ 全部确认）**

| 事实 | 位置 |
|---|---|
| VLA 输入由**录制帧**构建并缓存 | `scripts/train_joint_v2.py:714` `b.model_inputs = memo_eval.get(b.frames)`；`:745` `prefetch.get(batch.frames, lambda: memo.get(batch.frames))` |
| DR 扰动只改 burn-in 起始 state，与 `model_inputs` 无耦合 | `diff_vla_wbc/burnin.py` 的 `perturb_state(state, ...)` |
| 唯一的观测侧随机化 `_NoisyBuilder` 默认关闭 | `train_joint_v2.py:584`（`args.obs_noise > 0` 才启用，默认 0）|
| 46 维状态 = 腿/腰/臂/手 + 3 维，**不含 root xy** | `diff_vla_wbc/g1_tasks.py:35-46` |

**判断（与评审的差别）**

- 评审："这条路不成立"。**我们的立场：不成立的是"逐窗口最优"，而不是"执行器误差反传"。**
  - 梯度的第一段（`∂J/∂token`）来自 rollout 内的 WBC+WM 真实响应，**与 VLA 能否看到扰动无关** —— 所以"执行器误差反传"这个目标本身是可达的；
  - 第二段（`∂token/∂θ`）受限于 `o → token`：看不到的扰动只能学成**条件期望修正**（一个"状态相关但扰动无关"的预补偿）。
- **可达 headroom 明显小于 oracle 天花板**，这与实测一致（名义 +0.2%、5× 扰动 +3.6%，而 oracle 在知道扰动的条件下 +20~30%）。

**改法（三档，从便宜到彻底）**

| 档 | 动作 | 代价 | 收益 |
|---|---|---|---|
| **A1（推荐先做）** | 把 **burn-in 末状态的本体感觉**（29 关节 + dq + 重力投影 + 姿态）经 processor 写回 VLA 的 state 输入；图像仍用录制帧 | ~50 行（在 `_processed` 里把 `state.*` 用 `batch.start_state` 覆写）| 让"执行器误差"变成**可观测**，梯度从"常数偏置"升级为"状态相关修正" |
| **A2** | 把 DR 的 **xy/yaw 去掉或压到 ≤5 cm / ≤0.06 rad**，只保留关节/倾斜扰动 | 改 DR 配置 | 去掉"结构上不可观测"的那部分信号，避免学成平均 |
| **A3（彻底）** | B 线在线环：VLA 吃**实时**图像 + 实时状态 | 需渲染（已打通，见 §2）| 观测与状态完全一致 |

**验收**：A1 之后重测 "VLA 可观测扰动区间"里的 oracle headroom（🟡 目前没测过这个数）；如果它仍 ≈0.2%，说明连 A1 都救不了，应转 §5 的路线判断。

---

### 1.2 ② critic 自举：没有有用梯度 + 符号隐患

**代码核实（✅ 确认）**

| | 我们 | SuperMimic |
|---|---|---|
| 代码 | `diff_vla_wbc/v3_step.py:174-176`：`with torch.no_grad(): vH = self.critic(states[-1].detach(), ...)`；`boot = γ^H·survival[-1]·vH` | `third_party/SuperMimic-Reproduced/src/supermimic/imagination/policy_trainer.py:72`：`v_H = self.critic(feat[-1], goals[-1], xi_norm, e_root[-1])`（**无 no_grad、无 detach**）|
| 梯度 | **只经 `survival[-1]`** | **穿过 V 回到 s_H**（SHAC 本意）|

- `critic_center=True`（`v3_step.py:116-127`）按 clip 去均值 ⇒ 约一半样本 `V<0` ⇒ 对它们梯度会推 `survival` 往下（`survival≈0.9999`，量级极小但不该存在）。
- critic 的目标是**录制轨迹**的去趋势 return-to-go ⇒ 它主要从 `ref71` 认出"第几帧"，不是评估"偏离有多糟"。
- ⇒ **`critic/v_ev = +0.65~0.81` 只能证明拟合得好，不能证明对策略有用。**

**改法（4 步，全部在 `v3_step.py` + `token_critic.py`）**

| # | 改什么 | 具体 |
|---|---|---|
| **B1** | 让 V 对 `s_H` 保留梯度 | 去掉 `no_grad` 与 `states[-1].detach()`；`boot = γ^H · survival[-1] · V(s_H, ref_H, ξ)`（critic 参数仍在自己的监督损失里更新，不在策略损失里双计） |
| **B2** | 消除符号隐患 | 改 `boot = γ^H · survival[-1] · (V(s_H) − V(s_rec,H))`：保留"偏离有多糟"的语义，且不会出现负值乘 survival |
| **B3** | 覆盖偏离状态 | critic 训练数据 = 录制轨迹 **+** batch 里的 `batch.start_state`（DR 扰动起点）**+** A 线 on-policy 转移 |
| **B4** | 开/关 A/B | 1 小时判定"现在的 critic 是不是空转"（预期无差别）|

**验收**：B4 显示"开 critic 的 `imag_return` 显著高于关 critic"，或 B1-B3 后 `critic/v_ev` 与策略改进同向；否则**直接关掉 critic**（简化主线）。

---

### 1.3 ③ "WM 保鲜有害"的归因：动作标签错，不是 off-policy

**代码核实（✅ 确认，且修复材料就在手边）**

| 事实 | 位置 |
|---|---|
| `ha_transitions` 的动作是**命令反推** | `scripts/build_real_transitions.py:9`：`a_t = (q_cmd(t+1) − default_q)/scale` |
| 我们实测 WBC 实际动作与该参考动作 **RMS≈1.1** | `diff_vla_wbc/v3_step.py` 里对比 `q_cmd` 派生动作的注释 |
| **raw npz 里有真实施加的动作** | `data/ha_raw/*/*/*.npz` 的 132 个字段中含：`decoder_target_action (T,29)`、`decoder_raw_action (T,29)`、`final_body_action_29dof`、`body_effort_target`、`decoder_obs (T,994)` |
| WM 的原生训练域是"WBC 自己的动作"（trackerlab 的 PhysX rollout）| `docs/PLAN_HA_WBC_FINETUNE_20261025.md` §39 |

⇒ 评审的判断成立：**那批转移的 `a` 不是任何控制器真实施加的动作**，`qd_corr 0.752→0.589` 更可能是动作标签错造成的。

**改法（便宜且可能省掉整个 Isaac collect）**

| # | 动作 | 说明 |
|---|---|---|
| **C1** | 用 `decoder_target_action` 重建真转移 `(s_t, a_t, s_{t+1})` | 注意关节顺序（🟡 isaaclab 序 vs MJCF 序）+ 是否需要 `a=(q_target−default_q)/scale` 归一化 |
| **C2** | 用它重做保鲜（600–800 步、lr 1e-4、kl 1e-3），用**同一仪器**（`eval_wm_velocity`，ha_cmd，label，32×32）验收 | 目标 `qd_corr ≥ 0.85`；与现在的 0.752 对比 |
| **C3** | 若 C2 提升 ⇒ **WM 可以不靠 Isaac collect 就标定好**，Isaac collect 降级为最终验证；若仍下降 ⇒ 确认"必须 on-policy"，走 §2-A | — |

**当前配置备注（核实过）**：最近三轮（`s3v6_e60`、`s3v8_e60`、`s3A_lora32`）的 `wm_refresh_every` **都是 0（关）**，日志里 0 行 `wm-refresh`；只有更早的 `s3v4_*`/`s3v5_*`/`A0A1fix_*` 是 50（开）。**"关"这个结论建立在 C1 修好之前的测量上，证据不足，需 C2 重判。**

---

### 1.4 ④ 信任域：先"换中心"，自蒸馏留作后备

**代码核实（✅）**

| 事实 | 位置 |
|---|---|
| 现在的 hinge 中心是**标签 token** 的 rollout | `diff_vla_wbc/v3_step.py`：`roll_ref = imagine_with_tokens(..., tokens=batch.port_tokens.detach(), ...)` |
| 三代约束的结局：MSE 锚太弱（1/300）/ token hinge 从不触发（eps=0.15）/ 动作 hinge 有效（eps 0.3–0.6）| `docs/S3_JOINT_FT_ARCHITECTURE_FOR_REVIEW_20261027.md` P3 |
| v2 用完整 78 维 FM-BC 时 token 是稳的（`token_mse 0.00420→0.00509`、`hand_mse 0.00229→0.00230`）| `docs/S3_JOINT_V2_REPORT_20261026.md` §4.1 表 |

**判断**

- 评审说"token/动作欧氏距离都不是 flow 策略的信任域度量"——**成立**（三种失败模式就是证据）。
- 但**"换中心"与"自蒸馏"不是同一类东西**：

| | 换中心（保留 hinge） | 自蒸馏（加损失项） |
|---|---|---|
| 形状 | **硬墙**（球内 0 力） | **软弹簧**（全程有拉力） |
| 度量空间 | 动作空间 | token 空间 |
| 新超参 | ❌ 无（沿用 eps=0.6） | ✅ 需要（且属于我们被坑过的"软弹簧"类） |
| 改动 | 一行级 | 十几行 |

- **我们的历史证据**：软弹簧（`λ_a·MSE`）在 0.05 时是奖励项的 1/300 ⇒ 报废；硬墙在动作空间是**唯一验证过有效**的机制。
- ⇒ **先换中心；自蒸馏只在出现下列任一判据时才加：**
  1. `act/frac_out ≈ 0` 但 `tok_mse` 明显上涨（动作球内还能跑很远）；
  2. `ret_vla` 涨、`act/rmse` 不大、但生成分布明显变了（"动作球内、底模分布外"）；
  3. 换中心后 `tok_mse` 仍 >2×。

**改法**

| # | 动作 | 具体 |
|---|---|---|
| **D1** | hinge 中心换成**底模自己的输出** | 参照 rollout 的 token = `base_head_tokens`：**把 LoRA 增量临时关掉**（`B` 零初始化 ⇒ 关掉即底模），用同一批输入、同一噪声跑一次 FM 采样；其余 hinge 公式不变 |
| **D2** | （可选，满足上面判据才做）小权重自蒸馏 | 让适配 head 复现 `base_head_tokens`（MSE，同噪声采样）；**不含任何标签** |
| **D3** | 手部 | 部署时手部取**冻结底模 head 的输出**（多一次 4 步 DiT 前向），或恢复 78 维 FM-BC + D2 约束；**不要再用 `λ_hand=0`**（那等于接受抓取必失败）|

**验收**：D1 后 `tok_mse ≤ 1.5×` 且 `act/frac_out ∈ 0.1–0.5`；若 `frac_out≈0` 而 `tok_mse` 上涨 ⇒ 按判据 1 加 D2。

---

## 2. collect 的 A / B：各自彻底修什么

| 问题 | **A：回放式 collect**（已跑通，`scripts/isaac_g1_collect.py`）| **B：在线环**（渲染已打通，未实现）|
|---|---|---|
| ③ WM 动作标签错 | ✅ **彻底**（采 WBC 真实施加的动作）| ✅ 彻底 |
| WM 域不一致（P5）| ✅ **彻底**（我们 WBC 的状态分布）| ✅ 更强（含 VLA 引起的状态）|
| ① VLA 观测不到扰动 | ❌ 完全不解决 | ✅ **彻底**（实时图像 + 实时状态）|
| ② critic / ④ 信任域 | ❌ | ❌（必须改训练代码）|
| "+12.7% 是不是 hack" 可判定 | 🟡 半个（开环、仅名义区间）| ✅ **彻底** |
| 任务"做没做成" | ❌ | ✅（需加物体 + 任务奖励）|

**当前状态**（已核实）

| 项 | 状态 |
|---|---|
| A 采集 | ✅ 16 envs × 300–400 步 / 19–70 s → **4800–6400 条/次**；根位姿已对齐参考帧；`up=1.000` 不摔 |
| A 奖励补算 | ✅ `scripts/compute_collect_reward.py`（系统 python，用训练同一个 `bundle.reward`）|
| WM 重标定 | ✅ `scripts/calibrate_wm_onpolicy.py` 已写，等数据 |
| B 渲染 | ✅ 三开关配方见 `patches/isaac_render_recipe.md`，相机实测 `rgb=(1,64,64,3) mean=10.25` |
| B 本体 | ❌ 未实现；前置缺 `transformers/safetensors/av/regex`（Isaac venv 里）|

**A 的已知边界（必须写进判读）**：回放是**开环 token**（不随观测变化）⇒ **绝对任务成功率不可信**；但
①WM 的动作→状态标定 ②三臂相对比较 都有效。

---

## 3. 我们的评审文档要更正的 3 处

| 编号 | 原写法 | 更正为 |
|---|---|---|
| **P2** | "手部 BC 污染 token" | "**token 64 维无约束**（v3 关掉 token BC）+ 共享 trunk 在动 ⇒ 漂移"；修法是 D1/D2 的分布约束，**不是关手部** |
| **P4** | "`ret_vla > ret_gt` = reward hacking" | "**超过 GT 不等于作弊**（oracle 已证明存在更会跟参考的 token）；`+12.7%/+24%` 属**待真物理证实**" |
| **P6** | "in-loop WM 刷新有害（off-policy 数据）" | "**更可能原因是动作标签错**（命令反推 ≠ 真实施加）；需 C2 重做实验后再定论" |

---

## 4. 完整改动清单（按优先级，含验收）

### P0（决定生死，先做）

| # | 事项 | 文件 / 动作 | 验收 |
|---|---|---|---|
| **P0-1** | **窗口级真物理评测器**（评审 §4.1）| 在 `scripts/isaac_g1_collect.py` 上加：批量 reset 到录制状态 → 10 步 GT burn-in → 切候选 token 20 步 → 用 `bundle.reward` 算真回报；落 `(Δ_real)` | 同一个窗口上 GT 与 oracle 各跑一次，`Δ_real` 可复现（同 seed 逐位相同）|
| **P0-2** | **Δ_imag vs Δ_real 校准实验** | 输入：`s3v8_e60`、λ_r 扫描各臂、oracle、GT、底模；每个窗口同时有 `Δ_imag` 与 `Δ_real`，算相关系数与斜率 | 判读见 §5 |
| **P0-3** | **C1+C2**：用 `decoder_target_action` 重建真转移 → 重做 WM 保鲜 | 新脚本 `scripts/build_real_transitions_v2.py`（复用 `decoder_target_action`）+ `refresh_wm_ckpt.py` | 同一仪器 `qd_corr` 是否 ≥0.752；若 ≥0.85 则 collect 降级 |

### P1（修算法错误）

| # | 事项 | 文件 | 验收 |
|---|---|---|---|
| **P1-1** | **B1+B2**：critic 梯度穿 `s_H` + 改相对形式 | `diff_vla_wbc/v3_step.py:172-176` | `critic/v_ev` 与策略改进同向；无 NaN |
| **P1-2** | **B3**：critic 数据覆盖偏离状态 | `scripts/calibrate…` / batch 里加 `start_state` | `v_ev` 在扰动状态上不崩 |
| **P1-3** | **B4**：critic 开/关 A/B | 两然后个 100 步 run | 差异是否显著；不显著 ⇒ 关掉 |
| **P1-4** | **D1**：hinge 中心换成底模输出 | `v3_step.py` 里 `roll_ref` 的 token 源 | `tok_mse ≤1.5×`、`frac_out ∈0.1–0.5` |
| **P1-5** | **bf16 vs fp32 对拍** | 同 seed、64 窗口 | token 差 ≪0.004；`imag_return` 差 ≪1% |
| **P1-6** | **A1**：把 burn-in 状态写回 VLA 的 state 输入 | `scripts/train_joint_v2.py` 的 `_processed` | 重测 oracle headroom（可观测区间）|

### P2（按判读二选一）

| 分支 | 条件 | 动作 |
|---|---|---|
| 继续梯度路线 | §5 判为"正相关且斜率可接受" | 修完 P1 后长跑；**D2 仅在判据 1–3 出现时加**；手部按 D3 |
| 转 best-of-K + FM-BC | §5 判为"不相关" | 每个窗口采 K=8 候选 → 真物理打分 → 对胜者跑**原生 FM 损失**（自蒸馏/ReST 风格）；先量 K 个候选的**真回报离散度** |

### P3（长期）

- B 线在线环（渲染已通）：VLA 实时出 token + 相机；
- 带物体与任务奖励的场景（真正对齐"任务做成"）；
- 长期方法上参考 flow 策略专用 RL（FPO / ReinFlow / DPPO）。

---

## 5. 判据（P0-2 校准实验的读法，先写死）

| 结果 | 结论 | 下一步 |
|---|---|---|
| `Δ_imag` 与 `Δ_real` **不相关** | WM 梯度路线**判死刑** | 停梯度路线 → best-of-K（P2 分支 2）|
| **正相关但斜率 ≪ 1** | WM 排序有信号、幅度被高估 | 把它当**筛选器**；约束强度按斜率定 |
| 正相关、斜率 ≈1 | WM 在该区间可信 | 修完 P1 继续梯度路线 |

**附带产出**：顺带回答"`s3v8_e60` 的 `+12.7%` 到底是 hack 还是真改进"。

---

## 6. 风险与回滚

| 风险 | 缓解 |
|---|---|
| P0-1 的窗口级评测需要**批量 reset**（当前脚本每次只 reset 一次）| 先用 16 envs × 单窗口起步；确认吞吐后再做批量 |
| C1 的关节顺序/归一化搞错（🟡）| 先做 3 条 clip 的冒烟：用 `decoder_obs (T,994)` 反查动作约定，再全量 |
| B2 的 `V(s_rec,H)` 需要额外一次 critic 前向 | 可忽略（1.4e5 参数）|
| D1 的 `base_head_tokens` 计算 | 把 LoRA `B` 关掉跑一次采样；成本 ≈ 现有参照 rollout |
| 现在正在跑的 A collect 数据不可用 | 已在 `runs/wm_real/onpolicy_gt_*.pt`，随时可重算奖励 |

---

## 7. 附录：本文引用的代码位置（便于逐条核对）

| 事项 | 位置 |
|---|---|
| VLA 输入来自录制帧 | `scripts/train_joint_v2.py:714,745`；`_NoisyBuilder` `:562/584` |
| 46 维状态组成（无 root xy）| `diff_vla_wbc/g1_tasks.py:35-46` |
| DR 扰动在 burn-in | `diff_vla_wbc/burnin.py`（`perturb_state`）|
| critic 自举 detach | `diff_vla_wbc/v3_step.py:172-176` |
| SuperMimic 的 V_H 不 detach | `third_party/SuperMimic-Reproduced/src/supermimic/imagination/policy_trainer.py:72` |
| 命令反推动作 | `scripts/build_real_transitions.py:9` |
| hinge 中心是标签 token | `diff_vla_wbc/v3_step.py`（`roll_ref = imagine_with_tokens(..., tokens=batch.port_tokens.detach(), ...)`）|
| LoRA（`B` 零初始化）| `diff_vla_wbc/lora.py` |
| 真数据 critic 目标 | `scripts/build_real_returns.py` |
| on-policy collect（已跑通）| `scripts/isaac_g1_collect.py` |
| 奖励补算（同一函数）| `scripts/compute_collect_reward.py` |
| WM 重标定 | `scripts/calibrate_wm_onpolicy.py` |
| 渲染三开关配方 | `patches/isaac_render_recipe.md` |
| 启动前 39 项口径审计 | `scripts/s3_audit_config.py` |
