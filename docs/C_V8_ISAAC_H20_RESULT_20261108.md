# C-V8 × 真实 Isaac H20 对照（口径修正版 v2）：**WM 说 +0.0095，真物理 −0.0073**

> **提交对象**：外部评审（GPT）。
> **写于**：2026-11-09 00:10（**v2：修掉两个口径 bug 后重算**）。
> **一句话**：**把"完全相同的 20 个动作"分别喂给 WM 和真实 Isaac：
> WM 预测 C 比 Base 好 +0.00950，真实 Isaac 测得 −0.00733
> （18/24 窗口为负，符号检验 p=0.0227；A/A 噪声底 mean +0.00067 / std 0.00387）。**
>
> 证据等级：【FACT】有落盘 job/roll/report；【INFER】从 FACT 推出；【GUESS】未验证。

---

## 0. 速读

| 问题 | 答案 | 等级 |
|---|---|---|
| 这次测的是什么？ | **同一组 20 个动作**：同一初始状态、同一参考，C 的 token 解码出的动作 vs Base 的，分别喂 WM 与真 Isaac | FACT |
| WM 怎么说？ | **C 更好 +0.00950**（中位 +0.00407，71% 窗口为正） | FACT |
| 真实 Isaac 怎么说？ | **C 更差 −0.00733**（中位 −0.01110，**18/24 = 75% 为负**，符号检验 **p = 0.0227**） | FACT |
| 是噪声吗？ | A/A（两臂同动作）ΔJ_Isaac mean **+0.00067**、std **0.00387** ⇒ 主效应是均值噪声的 **11 倍**、单点 std 的 **1.9 倍** | FACT |
| WM 的排序能力？ | 符号一致率 **46%**，`corr(ΔJ_WM, ΔJ_Isaac) = +0.267` | FACT |
| 分任务？ | OpenDoor −0.00123（≈持平）· SitSofa **−0.00726** · VisionNavi **−0.01350** | FACT |
| **和 v1 的差别？** | v1 的 ΔJ_WM（+0.00680）是**用错动作序**算的（§2.5）；修正后 **+0.00950**，ΔJ_Isaac 从 −0.00532 变 **−0.00733**。**方向不变、幅度更大** | FACT |
| 还缺什么？ | **闭环 H20**（obs 历史随真实状态更新）——本轮动作是**开环解码**的 | FACT |
| 能不能说"WM 动力学彻底错了"？ | **不能**。只能说："对这 24 组开环动作，WM 的收益判断与真物理反号" | 【INFER】 |

---

## 1. 实验设计（可复现）

### 1.1 窗口【FACT】

`runs/temporal_ac_bptt_v5/eval_set` 的 **test split（96 窗口）** → 选 **24 个**：
OpenDoor 8 + SitSofa 8 + VisionNavi 8；每任务内按上一轮方向探针的 `cos_used` **分层**（4 高 + 4 低）。
每个窗口带 `state`(71, ha_v31 序)、`prev_action`(29)、`refs_f`(= frames+1..20，同一条参考)。

### 1.2 两臂的动作怎么来【FACT】

```
token:  同一窗口、同一 flow 噪声，分别采 C（LoRA 开）与 Base（lora_scale=0）
解码:   policy = SONIC release decoder（SonicDecoderTorch.from_onnx）
        —— 与真实 ONNX 对拍 max|Δa| = 1.9e-6（runs/harness_01_deploy/decoder_alignment.json）
动作:   a_t = policy(初始 10 帧 obs 历史, token[:, t]), t = 0..19  → (20, 29) **STANDARD 序**
```

**开环**：20 步都使用**同一个初始 obs 历史**解码，两臂完全一致，**唯一差别是 token**。
（真实部署里 obs 历史会随反馈更新 —— 这是本实验最大的软点，见 §4。）

### 1.3 Isaac 怎么执行【FACT】

`scripts/isaac_probe_action_rollout.py`（Isaac Sim 5.0 + IsaacLab 2.2）：

* 24 个 env 并行；起点 = `states_ha` 写进 Isaac 的根 + 关节；
* 每控制步 `q_target = default_q + a_t · action_scale`（`robot_spec` 从 `WBC_A24000_wmvel_cmd.pt` 读）；
* 4 个物理子步（1/200 s ⇒ 20 ms/控制步）；记录每步后的真实状态；
* **Isaac 里不跑任何 policy**，只执行给定动作序列。执行器是 **SONIC release decoder**（§2.4 逐层确认）。

### 1.4 评分口径【FACT】（★ v2 修正处）

**两边都必须用训练/评测同一个 bundle：`SonicIsaacBundle`**（`d03.load_env`）。
原因（这是 v1 的 bug）：

```python
# diff_vla_wbc/sonic_isaac.py  SonicIsaacBundle.step()
act_wm = action.index_select(-1, self.executor._std_to_ha)   # STANDARD → SONIC/HA 序
self.base.step(state, act_wm, xi, step_idx)
```

WM 是拿 **A′ 语料的动作（SONIC/HA 序）** 训的，而策略输出的是 **STANDARD 序**；
`SonicIsaacBundle.step` 是**唯一**做这个换序的边界。直接调 `load_isaac_bundle(...).step`
会把动作**置换着**喂给 WM（置换幅度 ≈1.24，动作量级只有 0.945）。

v2 的评分：

```
两边都用 SonicIsaacBundle.step / .reward
reward 配置 = 训练口径（runs/bptt1/args.json：term_off=1, d_max=0.3）
未折扣求和（不含 γ^t；term_off=1 ⇒ survival≡1）

ΔJ_WM    = Σ_t r(WM 想象状态_t)[C]    − Σ_t r(WM 想象状态_t)[Base]     ← 喂**同样的 20 个动作**
ΔJ_Isaac = Σ_t r(真实状态_t)[C]       − Σ_t r(真实状态_t)[Base]
```

---

## 2. 结果（v2）

### 2.1 主表

| | ΔJ_WM | ΔJ_Isaac |
|---|---|---|
| **合计（24）** | **+0.00950**（中位 +0.00407，**71% 正**） | **−0.00733**（中位 −0.01110，**25% 正**） |

* **18/24 窗口在真实 Isaac 里变差**，符号检验 **p = 0.0227**；
* 符号一致率 **46%**；`corr(ΔJ_WM, ΔJ_Isaac) = +0.267`。

### 2.2 分任务

| 任务 | n | ΔJ_WM | ΔJ_Isaac | Isaac>0 | 符号一致 |
|---|---|---|---|---|---|
| HSI_open_door | 8 | +0.02053 | **−0.00123** | 38% | 38% |
| HSI_sit_sofa | 8 | +0.00501 | **−0.00726** | 12% | 50% |
| HSI_vision_navi | 8 | +0.00297 | **−0.01350** | 25% | 50% |

⇒ **OpenDoor 基本持平；SitSofa、VisionNavi 明确变差**（与 v1 同结论，幅度更大）。

### 2.3 A/A 噪声对照【FACT】

两臂动作用成完全相同（都取 C 的动作），重跑 Isaac + 评分：

| | ΔJ_WM | ΔJ_Isaac mean | ΔJ_Isaac std | 非零星数 |
|---|---|---|---|---|
| 主实验（C vs Base） | +0.00950 | **−0.00733** | 0.0157 | 18/24 为负 |
| **A/A 对照** | 0（构造） | **+0.00067** | **0.00387** | 5/24 非零（其余逐位为 0） |

⇒ Isaac rollout 有少量非确定性（同动作重跑最多差 ±0.01），但**平均噪声只有 +0.00067**；
主效应 −0.00733 是它的 **11 倍**、是单点 std 的 **1.9 倍**，且 18/24 同号（p=0.0227）。

### 2.4 执行器确认（回应"是不是 SuperMimic"的疑问）【FACT】

| 环节 | 实际用的东西 | 证据 |
|---|---|---|
| 动作解码（训练侧） | **SONIC release decoder** | `d03.load_env` 日志 `executor=SonicExecutor`；`SonicExecutor.__init__` 里 `SonicDecoderTorch.from_onnx(DEFAULT_DECODER_ONNX)` |
| 与部署 ONNX 一致性 | **同一张图，max\|Δa\| = 1.9e-6** | `runs/harness_01_deploy/decoder_alignment.json`（E1/E5） |
| Isaac 执行 | 不做任何 policy 推理，只下发 `q_target = default_q + a·scale` | `isaac_probe_action_rollout.py:139` |
| `WBC_A24000_wmvel_cmd.pt` 的作用 | **只提供 `robot_spec`（default_q / action_scale）** | 同上：`ck["robot_spec"]` |
| SuperMimic 出现在哪 | **只有 reward 定义**（`trackerLab...mdp.supermimic.RewardCfg`）与 **robot spec** | 不在动作链路上 |

### 2.5 ★ v2 修掉的两个口径 bug（重要）

| # | bug | 后果 | 修法 |
|---|---|---|---|
| **A** | v1 评分用 `load_isaac_bundle`（裸 bundle），**没有**走 `SonicIsaacBundle.step` 的 STANDARD→SONIC 换序 | **WM 收到被置换过的动作** ⇒ v1 的 ΔJ_WM=+0.00680 是错的 | 改用 `d03.load_env` 的 `SonicIsaacBundle`；修正后 ΔJ_WM=**+0.00950** |
| **B** | val/test reward 配置不一致（`term_off` 1 vs 0、`d_max` 0.3 vs 0） | 离线数字与训练目标差一项 | 本报告统一用训练口径（`runs/bptt1/args.json`） |

**修完之后结论方向不变、幅度更大**：WM **+0.0095** vs Isaac **−0.0073**。
并且这排除了评审最担心的"WM 与 Isaac 执行了不同动作"的**一半**可能：
两边喂的是**同一批动作数组**（job 里的 `acts_plus`/`acts_minus`），且 WM 收到的是**正确顺序**的同一批动作。

**还剩的一半**：这些动作本身是**开环解码**的（固定初始历史），
真实闭环里 SONIC 每个 tick 都会从**真实状态**重算 obs 历史 ⇒ 动作序列会不同。见 §4 软点 1。


### 2.5.1 动作序的直接验证（本 v2 补做）【FACT】

用**同一批动作**分别按两种口径滚 WM，与真实 Isaac 的每步状态比 RMSE：

| WM 动作序 | 20 步平均 RMSE（vs 真 Isaac） |
|---|---|
| 不换序（v1 的错法） | **1.864** |
| 换到 SONIC/HA（v2，= 训练口径） | **1.790**（更低） |

⇒ **v2 的换序确实更符合 WM 的训练口径**（RMSE −4%）。
但要注意：**状态 RMSE 只差 4%，而 ΔJ_WM 从 +0.00680 变成 +0.00950（+40%）**——
说明"收益"这个量对动作序比"状态预测"敏感得多。
另外 A′ 单步重放也验证了同一件事：A′ 动作**按原序**喂 WM 误差最低（2.592 vs 置换后 3.395/2.837）。

### 2.6 逐窗口（v2）

| idx | task | cos | ΔJ_WM | ΔJ_Isaac |
|---|---|---|---|---|
| 52 | open_door | +0.080 | +0.00311 | −0.00372 |
| 54 | open_door | +0.271 | +0.00014 | −0.00681 |
| 56 | open_door | +0.185 | +0.00218 | −0.00868 |
| 59 | open_door | +0.197 | +0.00339 | −0.01385 |
| 63 | open_door | +0.073 | +0.02723 | −0.01624 |
| 64 | open_door | −0.008 | +0.02030 | **+0.01443** |
| 67 | open_door | +0.178 | +0.06289 | **+0.00752** |
| 68 | open_door | +0.053 | +0.04501 | **+0.01748** |
| 69 | sit_sofa | +0.246 | −0.00225 | −0.01587 |
| 70 | sit_sofa | +0.181 | +0.02384 | −0.00338 |
| 71 | sit_sofa | −0.200 | +0.01067 | −0.03041 |
| 72 | sit_sofa | +0.052 | −0.02948 | −0.01013 |
| 74 | sit_sofa | −0.075 | +0.04673 | −0.01712 |
| 75 | sit_sofa | +0.082 | +0.01986 | −0.00757 |
| 76 | sit_sofa | +0.164 | −0.01435 | **+0.04094** |
| 77 | sit_sofa | +0.114 | −0.01498 | −0.01455 |
| 78 | vision_navi | −0.055 | −0.02569 | −0.02757 |
| 80 | vision_navi | +0.189 | +0.00264 | −0.03360 |
| 82 | vision_navi | +0.335 | +0.02196 | **+0.02587** |
| 85 | vision_navi | −0.081 | −0.00823 | −0.02214 |
| 86 | vision_navi | −0.157 | −0.00153 | −0.01207 |
| 89 | vision_navi | +0.267 | +0.01925 | **+0.00448** |
| 93 | vision_navi | −0.186 | +0.00475 | −0.02292 |
| 94 | vision_navi | +0.398 | +0.01064 | −0.02003 |

* 只有 **6 个**窗口在真实 Isaac 里变好（64/67/68/76/82/89），其中 4 个在 OpenDoor；
* `corr(cos, ΔJ_Isaac) = +0.342`（弱），`corr(cos, ΔJ_WM) = +0.093`（几乎没有）
  ⇒ **cos 不能当"真物理会不会变好"的预测器**。

---

## 3. 结论

### 3.1 直接结论【FACT】

> **对这 24 组动作，WM 的收益判断与真实物理反号**：
> WM 说 C 平均更好（+0.00950，71% 窗口为正），真实 Isaac 说 C 平均更差（−0.00733，75% 窗口为负）。

### 3.2 与之前证据的关系【INFER】

| 之前的测量 | 现在怎么读 |
|---|---|
| `cos(Δtoken, ∂J/∂token) ≈ 0.08`（单点 & 累计一致） | LoRA 的位移 92% 不在 WM 梯度方向；WM 只"看见"那 8% |
| 单步真实 AdamW：ΔJ_WM **+9.5e-5**、76% 窗口为正 | 每步对 **WM 目标** 都是正贡献 ⇒ 训练本身没坏 |
| 累计 ΔJ_WM **+0.0164**（test） | 那 8% 分量的线性累积 |
| **本轮**：ΔJ_Isaac **−0.0073**（同动作） | **WM 对这批动作的收益预测是反的** |

⚠️ v1 报告里"另外 92% 的非梯度分量造成伤害"这个解释**仍未被证明**：
梯度投影比例不能直接分解真实回报损失，必须做**受控动作干预**才能归因（§6 第 3 项）。

### 3.3 与完整闭环的关系【INFER】

C 12/24 vs Base（老环境）11/18、同环境 Boxing 4/6→5/6、OpenDoor 同场景 4:0 倒向 Base
—— 与本轮"短程跟踪本身就变差"一致。

---

## 4. 边界

| # | 软点 | 方向 | 影响 |
|---|---|---|---|
| **1** | **动作是开环解码**（固定初始 obs 历史） | 未知（可能对 C 不利，也可能对两臂都失真） | **最大软点**：不能直接推"部署分布上 WM 也错" |
| 2 | 只测 20 步（0.4 s） | 未知 | 长视界可能反转 |
| 3 | 24 窗口、按 cos 分层、可能 clip 相关 | 抽样 | p=0.0227 只对这批窗口成立 |
| 4 | 只恢复关节+根状态，不是完整物理快照（接触/物体未恢复） | 未知 | 与"严格配对"差一层 |
| 5 | Isaac rollout 有 ±0.01 的非确定性 | 已知 | A/A 已量化（mean +0.00067） |

---

## 5. 能说 / 不能说

| ✅ 可以写 | ❌ 不能写 |
|---|---|
| "对这 24 组开环动作，WM 预测 C 更好而真实 Isaac 更差（−0.00733，75% 窗口为负，p=0.0227，超过 A/A 均值噪声 11 倍）" | "WM0 的动力学彻底错了"（方向在单次小扰动上仍 69%/98.4%） |
| "离线 ΔJ 的正收益在真实物理上没有复现" | "V8-C 一定比 Base 差"（开环 + 小样本） |
| "cos 与真实 ΔJ 相关仅 0.34，不能当预测器" | "92% 非梯度分量是伤害来源"（未做受控干预） |
| "A/A 对照证明该差异不是 Isaac 非确定性" | "20 步结论能外推到完整任务" |

---

## 6. 下一步（按评审建议的顺序）

| 顺序 | 实验 | 成本 | 判读 |
|---|---|---|---|
| **1** | ✅ **同动作重放**（本 v2） | — | 已排除"WM/Isaac 动作不同"——两边是同一批动作 |
| **2** | **闭环 H20**：Isaac 里逐 tick 用**真实 obs 历史**解码（token 回放或在线 server），同初始状态、Base/C 配对 | 半天 | 仍为负 ⇒ WM 在 Actor 分布上可靠性不足；转正 ⇒ 开环是主因 |
| **3** | **受控动作干预**：把 Δtoken 分解成"沿梯度分量 + 正交分量"，分别执行看真实 J | 半天 | 直接检验 v1 那个未证明的解释 |
| **4** | 若 3 证明是 WM 误判 ⇒ **定向 WM Refresh**（A′ 为主 + 少量 Base/C 真实转移，保留 Gain 约束，门禁必须含"真实动作改进方向"） | 1–2 天 | 不能只看预测误差下降 |

**明确不做**：暂不从官方 Base 再训一轮 280 步 C；不升 LoRA rank；不加 token 方向余弦正则
（WM 收益判断不可靠时，提高 WM 上的优化效率只会更有效地利用 WM 误差）。

---

## 7. 给评审的 6 个问题

1. **开环是不是最可能的主因**？如果闭环 H20 转正，你会怎么重新解释本轮 24 窗口的结果？
2. **受控干预怎么做最省**：把 Δtoken 投影成"梯度分量 + 正交分量"各跑 24 窗口够吗？还是用 δ 扫描（0.25×/0.5×/1×）？
3. **WM 的"可靠域"怎么划**？现在是"单次 δ 扰动 69–98% 同号、Actor 累积位移反号"。
   用什么量划这条线（δ 幅度 / cos 阈值 / 动作分布距离）？
4. **闭环 H20 的工程口径**：SONIC 的 obs 历史、前一动作、token 刷新时序要与部署一致。
   你会用 `SM_TOKEN_NPY` 回放还是在线 server？各自的坑？
5. **如果闭环 H20 也是负的**：先定向 WM Refresh，还是先改 Actor 优化（关物理项/J-only/降 update 幅度）？
6. **怎么避免下一次"WM 说好、物理说差"**：把什么做成**强制门禁**（例如任何声称的改进必须先 N 个真实 Isaac 窗口配对验证）？

---

## 8. 证据索引

```
job（动作+参考+起点）: $WT/runs/three_track_v8/c_tracking_recovery/isaac_h20/job.pt
真物理状态          : $WT/runs/three_track_v8/c_tracking_recovery/isaac_h20/roll.pt
A/A job / roll      : $WT/runs/three_track_v8/c_tracking_recovery/isaac_h20/{job_aa.pt,roll_aa.pt}
**v2 评分**         : $WT/runs/three_track_v8/c_tracking_recovery/isaac_h20/report_v2.json
**v2 A/A**          : $WT/runs/three_track_v8/c_tracking_recovery/isaac_h20/report_v2_aa.json
v1（有动作序 bug）  : $WT/runs/three_track_v8/c_tracking_recovery/isaac_h20/report.json

脚本：
  $WT/scripts/probe_c_vs_base_isaac_plan.py     # 选窗口 + 解码两臂动作
  $MAIN/scripts/isaac_probe_action_rollout.py   # 真 Isaac 执行（24 env × 20 步）
  $WT/scripts/probe_isaac_h20_report.py         # 评分（**v2：SonicIsaacBundle + 训练 reward 口径**）

前置证据：
  $WT/runs/three_track_v8/c_tracking_recovery/dir_probe_test_v2/test.json   # cos=0.106
  $WT/runs/three_track_v8/c_tracking_recovery/step_probe_0251/step_test.json # 单步 ΔJ=+9.5e-5
  $MAIN/runs/harness_01_deploy/decoder_alignment.json                      # SONIC 对拍 1.9e-6
```
