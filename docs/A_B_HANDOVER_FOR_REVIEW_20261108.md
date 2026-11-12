# COMIMIC · A/B（Delta-Continue / Delta-Retrain）交底文档 —— 供外部 reviewer 修改设计

> ❌ **已被 `docs/A_B_SITUATION_ARCH_PLAN_20261108.md` 取代**：本文件里的**物理数字、单位、`L_same` 结论**已被三次复测推翻（2026-11-08）。仅供历史留档，不要再据此做决策。

> 版本：2026-11-08
> 用途：把 **A（Delta-Continue）** 与 **B（Delta-Retrain）** 两条线的**结论、设计、架构、问题**一次讲清，
> 交给外部 reviewer（GPT）出"改什么、怎么改"的方案。
> 约定：**只写跑过的东西**，每个数字都注明出处文件；没跑过的明确标 `NOT_RUN`。
> 本文自称"我"的地方是执行方，"你"是 reviewer。

> ⚠️ **2026-11-08 复测增补（必读）**：本文有四处结论已被新实测推翻，见
> **`docs/A_B_ADDENDUM_MEASUREMENTS_20261108.md`**。摘要：
> (1) B 的 test ΔJ 已补 = **+0.02074**（A +0.02185）；
> (2) **A 的物理"−79% 尖峰改善"不成立**（96 窗口实测：全局峰值 2.027→**2.093**，t=14 只降 42%，尖峰移到 t=5/t=19）；
> (3) 训练日志与 `diag_action_budget_v5.py` 里的 **"rad / mrad" 实为归一化动作域**（`bundle.action_scale=ones`），
> 真物理值 = ×`G1_ACTION_SCALE`（均值 0.396）；
> (4) "λ_peak 形同虚设"**错**（加权梯度 0.0605 ≈ 30%×‖∇J‖），真正的 bug 是
> **`L_act` 与 `L_same` 是同一个量、单位却不同（7.9 mrad vs 20 mrad 阈值）⇒ 重复计算**。
> 读 §5.4 / §5.5 / §5.7 / §7.1 时请以增补文档为准。

---

## 0. 一页摘要（先读这页就够跟 reviewer 对话）

### 0.1 我们在做什么

冻结一个官方 VLA（`FT8000`，Gr00tN1d7），不动它一根权重；在它输出的 **64 维 motion token chunk** 上加一个
**1.74 M 参数的逐帧残差网络 Δ**，让"同一个 VLA、同一个观测"下的动作更贴合参考轨迹。
Δ 用**可微世界模型（A′ WM）的 20 步想象回报**做 BPTT 训练，再用**真实 SONIC–Isaac 闭环**做最终裁判。

- **A（Delta-Continue）**：Δ 网从 v5 的 best **warm-start**，只训 300 updates。
- **B（Delta-Retrain）**：Δ 网**随机初始化**、head lr 大 6 倍 / enc lr 大 5 倍、训 1000 updates（**实际停在 960/1000**）。
- （第三条线 C 是 LoRA16 微调 VLA 本体，与本文件无关，只在需要对照时提一句。）

### 0.2 六个必须知道的事实

1. **A 的 checkpoint**：`it=225`，val ΔJ **+0.02962**，test ΔJ **+0.02185**。
   **A 的 18 集真实闭环：9/18**（OpenDoor 3/6、SitSofa 4/6、VisionNavi 2/6）。
2. **B 的 checkpoint**：`it=900`，val ΔJ **+0.02774**；**test ΔJ = +0.02074（已补，见增补 §2）**；
   **没有 last**；**只有 1 个 seed**。
3. **A 已接近该度量的"局部梯度候选值"**：⚠️ *已修正* —— val 集上
   `Δ* = 0.08·∇J/‖∇J‖`（**单步梯度方向 + cap 范数**）给出 **+0.0298**，A 的 **+0.02962 = 该候选值的 99.4%**。
   **但这只是一个一阶候选，不是非线性 20 步目标的证明上界**，所以**不能**推出"A 只剩 0.00018 的提升空间"、
   也**不能**推出"必须放宽 cap"。（reviewer 的这条纠正成立，本文采纳。）
4. **A 与 B 不是干净的对照**：初始化、学习率、warmup、步数**四个变量同时不同**（见 §5.1）。
   更麻烦的是 **B 的 lr/warmup 恰好等于 v5 的原配方** ⇒ **B 实质是"v5 配方 + 物理损失"的重跑**
   （见 §5.2b），所以"B 没超过 A"既不能证明 warm-start 有用，也不能证明它没用。
5. **两个已知病灶一个没修**：`‖Δ‖` 被 `cap=0.08` 顶死（`‖u‖/cap p50 = 2.4`，70% 的帧 >2×cap）；
   条件化只被验证到"不是常量"（`cos_shuffle_z ≈ 0.65`，门禁阈值是 `< 0.9`）。
6. **单位陷阱（本轮新发现）**：训练日志与 `diag_action_budget_v5.py` 里的 `rad/mrad` 其实是
   **归一化动作域**（`bundle.action_scale=ones`）；真物理值 = ×`G1_ACTION_SCALE`（均值 **0.396**）。
   ⇒ 主文档 §5.5 的所有 mrad 数字、以及"A 把 t=14 从 2.07 压到 0.43"这一条，都要按增补文档重读。

### 0.3 我最希望你回答的五个问题

1. **在"A 已接近单步梯度候选值（99.4%，但这不是证明上界）"的前提下，下一刀应该切哪？**
   （目标函数 / 物理约束 / WM / 闭环指标 / 架构）
2. 给我一个**能真正区分 A 与 B 的重跑设计**（变量控制、seed 数、窗口数、显著性与配对口径）。
3. 物理损失那套"按梯度预算反解 λ"的做法有哪几处会骗人？（§4.6 有完整公式和实测数）
4. 闭环只有稀疏 success（9/18），要不要、以及怎样加**稠密信号**来定位"第几 tick 开始偏"？
5. 在**不改部署契约**（§6.3 的红线）的前提下，你要求我改哪几个文件哪几个函数？请给可执行清单 + 验收口径。

---

## 1. 系统背景与三条线路的定位

### 1.1 任务与闭环链路

```
Isaac Sim (HumanoidArena 3 任务) ──相机/本体状态──► 冻结 VLA (FT8000) ──64 维 token chunk──►
  ──► Δ 网（本文件的主角，1.74 M）──► 加回 token ──► SONIC decoder (ONNX, 冻死) ──► 29 维归一化动作
  ──► 物理映射 q_target = q_default + G1_ACTION_SCALE ⊙ a ──► 限位/速率安全层 ──► Isaac PD 执行
```

- 控制频率 **50 Hz**，VLA 每 **20 步**重规划一次（receding horizon，`exec_k=20`）。
- token chunk 长度 **T=40**，只在**前 K=20 帧**（= 会被执行的部分）上加 Δ，`t ≥ 20` 严格置 0。
- VLA 底座：`runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（A/B/C 共用，**一个字节都不训**）。

### 1.2 A/B 在论文叙事里的位置

| 线路 | 一句话 | 当前状态 |
|---|---|---|
| **A** Delta-Continue | Δ 网 warm-start 自 v5 best，只做物理约束微调 | **完成**：WM 指标、物理审计、18 集闭环都有 |
| **B** Delta-Retrain | Δ 网随机初始化重训，验证"不依赖 warm-start 也能得到同样解" | **未完成**：960/1000 被人工停，只有 best ckpt，没有闭环 |
| C LoRA16-FT8000 | 改 VLA 本体（LoRA + proj_out） | 与本文件无关，仅作"另一条路"的对照 |

> A/B 的科学目的：**A 的提升是不是只是"v5 warm-start 的余温"**。
> 如果随机初始化重训（B）也能达到同等水平，说明 Δ 网这个结构本身有效；否则说明收益主要来自 v5。

---

## 2. 冻结件与数据（A/B 共用，reviewer 需要知道这些东西不可动）

| 角色 | 路径 | 关键事实 |
|---|---|---|
| 冻结 VLA | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000` | 6.9 GB，bf16，**全冻结**（1.62 G 参数） |
| VLM | `groot_repro/nvidia/Cosmos-Reason2-2B-weights` | 冻结 |
| 世界模型 WM | `runs/a_prime_sonic_exc/wm_gain2.pt` | 35.3 MB，sha `fc5aeb30e987f016…` |
| 真转移语料 | `runs/a_prime_sonic_exc/all_transitions.pt` | 370 MB，sha `063355356116…` |
| 评测集 | `runs/temporal_ac_bptt_v5/eval_set/` | **375 train / 81 val / 81 test clips**；`n_eligible_frames=338,402`，训练时 `allowed_clips` 过滤到 **235,280** 帧防泄漏 |
| 物理限位 | `runs/three_track_v6/physical/limits_standard29.npz` | 从 trackerlab `RobotSpec.mjcf_path` + MuJoCo `jnt_range` 导 90% soft 限位（knee `[-0.012, 2.659]`、hip_pitch `[-2.309, 2.561]`） |
| 物理 λ | `runs/three_track_v6/calibration/physical_weights_delta.json` | `λ_lim=0.0529`、`λ_peak=8.4e-4`、`λ_same=0.010`（**未激活**）、`λ_rate=0`（**BLOCKED**） |
| SONIC decoder | `GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx` | 40.9 MB，sha `c7241a12…`，**部署与训练同一个** |
| WBC 权重 | `runs/S1_handover/WBC_A24000_wmvel_cmd.pt` | 54.5 MB，sha `2390dc29…` |

**WM 训练时的评测配置**（`runs/bptt1/args.json`，A/B 都吃这套）：
`reward_mode=saturating`（6 个 `exp(-e/std²)` 跟踪核）、`reward_vel_scale=1.0`（速度项开）、
`fm_steps=4`、`burnin_steps=10`、`horizon=20`。

---

## 3. 架构（`diff_vla_wbc/delta_token_temporal_v5.py`）

### 3.1 输入 / 输出契约（全部只用**部署可获得**的量）

| 张量 | 形状 | 含义 |
|---|---|---|
| `Z` | `(B, 40, 64)` | 冻结 VLA 本次生成的**整段** motion token chunk |
| `state64` | `(B, 64)` | **chunk 决策时刻**的观测 `[rot6d(6) \| q(29) \| qd(29)]`，HA_V31 关节序 |
| `hist` | `(B, 10, 58)` | 过去 10 个控制 tick 的 `[q(29) \| qd(29)]`，最老→最新，**HA 序** |
| `mask` | `(B, 40)` | `1` = 该帧要执行且要修正（`t<K`）；`0` = 输出**严格为 0** |
| `Δ` | `(B, 40, 64)` | **逐帧一个**修正量（不是把某个 Δ 广播到整段） |

**关键设计点**（为什么长这样）：

- 保留了 `z_t`（本帧要改的那个 token）、`c_t`（chunk 自注意力上下文）、`PE_pos(t)`（位置编码）三者同时给，
  网络才有能力在"同一 chunk、同一 state、同一 history"下输出**随 t 变化**的修正。
- `hist` 用 **GRU**（1 层，58→128）压成一个向量后广播到所有帧 —— 这样历史只提供"我现在处于什么状态"，
  不提供"第 t 帧该改多少"，避免把历史当帧索引用。
- 输出层 `N(0, 4e-4)` + bias=0 ⇒ **第 0 步 ≈ Base**（不做任何修正），这是 A/B 都能从 v5 平滑起步的前提。

### 3.2 网络结构与参数量（注册值 **1,744,832**）

```
token_proj   : Linear(64 → 128)                                   8,320
pe_chunk     : Parameter(40, 128)                                 5,120
chunk_encoder: TransformerEncoderLayer(d=128, h=4, ffn=256,
               gelu, norm_first, dropout=0)  × 1  ≈              133,000
chunk_norm   : LayerNorm(128)                                       256
hist_gru     : GRU(58 → 128, 1 layer)                            72,192
pe_pos       : Parameter(40, 16)                                    640
head         : Linear(400→1024) + SiLU + Linear(1024→1024) + SiLU
               + Linear(1024→64)                              1,525,824
```

- `d_in = 400 = 64(Z) + 128(c_t) + 128(h) + 64(state64) + 16(PE)`；
- 头占了 **87%** 的参数（1.526 M / 1.745 M）⇒ 这是一个"小上下文编码器 + 大 MLP 头"的结构；
- **`cap=0.08` 是软上限**：`Δ = cap · tanh(u/cap) · u/‖u‖`（相当于把 `‖Δ‖` 限在 0.08、方向由 `u` 定）。

### 3.3 部署端的等价性（已逐位验证）

- 部署侧先按 HA 序减掉 ckpt meta 里的 `hist_default_joint_pos`，再喂 GRU；
- 训练序 vs 部署序 `max|Δ| = **1.49e-8**`；
- `hist[10]`（最后一帧）必须等于 `state64` 的 q/qd，**容差 5e-3，不一致直接抛**（fail-closed）——
  真跑中确实拦下过一次"hist 落后一帧"。

---

## 4. 训练设计（`scripts/train_delta_physical_bptt_v6.py`）

### 4.1 目标函数（完整、可直接抄进 reviewer 的推导）

```
J_i(Δ) = Σ_{t=0}^{H-1} γ^t · s_t(Δ) · r_t(Δ)          γ=0.99, H=20, s=survival, r=WM 跟踪奖励

loss = -mean_i J_i
       + λ_act   · Trust(e_same)        λ_act  = 0.010, τ = 0.020 rad
       + λ_smooth· Smoothness(e_same)   λ_smooth = 0.002
       + λ_anchor· Anchor(Δ)            λ_anchor = 0.001
       + λ_raw   · Hinge(‖u‖/2cap)      λ_raw  = 1e-3, κ = 2.0
       + λ_lim   · L_lim                λ_lim  = 0.0529
       + λ_peak  · L_peak               λ_peak = 8.4e-4
       + λ_same  · L_same               λ_same = 0.010（**实测梯度为 0，未激活**）
       + λ_rate  · L_rate               λ_rate = 0       （**BLOCKED**）
```

- `e_same` 的定义是**同观测**动作差：用**同一份 obs93 + 基础 token** 再解一次动作，只隔离"token 差异"带来的直接动作影响，
  不混入两条 WM 轨迹未来状态不同造成的差异（`trust_mode=same_obs`，已在训练器里拦截 `executor.act` 实现）。
- `L_lim = mean_b[ 0.5·max_{t,j} v² + 0.5·mean_t max_j v² ]`，
  `v = relu(lo-q_raw)/s + relu(q_raw-hi)/s`（**裁剪前**的绝对物理目标、**不对称**上下限、按关节尺度归一化、**max 形式**）。
- `L_peak = mean_b[ max_{t,j} relu(|e_phys|/b_j - 1)² ]`，`b_j = 3 × mean_{t}(|e_phys|)_j`（**自适应预算**，见 §7.3 的质疑）。
- `L_rate` 约束裁剪前目标的 `Δq/dt`，但 `RobotSpec` 没有 velocity 字段 ⇒ **限位 npz 里 `has_rate=False`**。

### 4.2 采样与防泄漏

- **on-the-fly**：每个 update 现抽帧、现跑 burn-in（10 步）再截 20 步窗口；
- `allowed_clips` 只含 **375 个 train clips**（`n_eligible_frames=235,280`），val/test clips 进程内就屏蔽；
- `uniform_over_eligible=true`：在"合法帧"上均匀采样，不做 episode 内加权；
- 扰动：`start_max_drift=0.1`、`perturb_xy=0.05`、`perturb_yaw=0.1`、`perturb_dq=0.05`、`perturb_dqd=0.3`，
  `noise_cycling=0`（不换噪声，同一窗口固定噪声 ⇒ val 评测可复现）。

### 4.3 评测（val / test）

- 固定 96 个 val 窗口、96 个 test 窗口；`ΔJ = mean(J(Δ)) - J_base`，**同一个 base**：
  `J_base(val)=1.50594`、`J_base(test)=1.57581`；
- 该评测的统计不确定度 **SE ≈ 3e-4**（n=96），所以 0.001 级别的差异**在这个集上是可测的**；
- 但**训练轨迹自身的抖动是 ±0.0015~0.002**（见 §5.3 曲线）⇒ 不同 ckpt 之间 0.002 的差要小心解读。

### 4.4 门禁（fail-closed）

- **cond 门禁**（每 25 步）：把 `Z` / `state64` / `hist` 分别打乱后比较 Δ 的余弦相似度；
  `passed = (cos_shuffle_z < 0.9) & (std_across_windows > 1e-4) & (frame_to_frame_std > 1e-4)`。
  ⇒ **这个门禁只证明"Δ 不是常量"，不证明"条件化是有用的"**。
- **history parity**：训练序 vs 部署序 `max|Δ| = 0.0`、`q` 块 vs `state64` `max|Δ| = 0.0`。
- **critic**：`critic_enabled=0`，`status=BLOCKED_BOOTSTRAP_NOT_IDENTIFIABLE`（`K==H==20` ⇒ 没有可自举的 tail）。

### 4.5 checkpoint 契约（这是后面很多麻烦的根源）

`_save_ckpt` 只存**推理所需**：

```
{ delta_token_v5_state, delta_token_v5_spec, delta_token_v5_meta,
  ckpt_sha256, it, dJ_val, cap, data_mode }
```

**没有 optimizer / scheduler / RNG / sampler** ⇒
**严格 resume 不存在**，任何"续跑到 1000"都只能是 warm-start（优化器从 0 重开）。

### 4.6 λ 是怎么来的（"按梯度预算反解"）

`scripts/calibrate_physical_bptt_weights.py` 在 48 个 **train** 窗口上：
1. 算 `‖g_J‖`（本 ckpt 实测 **0.2018**）；
2. 对每个物理项算 `‖g_term‖ / ‖g_J‖`；
3. 按预算 `limit 0.5 / peak 0.3 / same 0.15 / rate 0.1` 反解 λ；
4. **梯度为 0 的项不反解**（`same`、`rate` 在标定时梯度为 0 ⇒ 保留配置默认值）。

标定实测（`runs/three_track_v6/calibration/delta.json`）：

| 项 | loss | ‖g‖ | **‖g‖/‖g_J‖** | 反解 λ | 备注 |
|---|---|---|---|---|---|
| `J` | — | 0.2018 | 1.0 | — | — |
| `lim` | 1.286 | 1.906 | **9.45×** | 0.0529 | 活跃 |
| `peak` | 0.267 | 72.05 | **357×** | **8.4e-4** | 活跃，λ 被压到极小 |
| `same` | 0.0 | 0.0 | 0 | 未反解 | **未激活** |
| `rate` | 0.0 | 0.0 | 0 | 0 | BLOCKED |

物理量标定值：`q_raw ∈ [-1.955, 2.389]`、`e_same_absmax = 0.0928 rad`、`rms_same = 0.01305 rad`。

---

## 5. A 与 B：唯一的差别、结果、以及"不是干净对照"这个硬伤

### 5.1 配置差异（**四个变量同时变**）

| 项 | A（Delta-Continue） | B（Delta-Retrain） |
|---|---|---|
| 初始化 | `--init-delta-checkpoint` = **v5 best**（`temporal_ac_bptt_v5/main_v2/seed0/checkpoint_actor_best.pt`，it=450） | **不传** ⇒ 随机初始化 |
| head lr / enc lr | **5e-5 / 2e-5** | **3e-4 / 1e-4**（大 6 倍 / 5 倍） |
| warmup | 20 | 50 |
| updates | 300 | 1000（**实跑 960**） |
| eval_every | 25 | 50 |
| 其余（batch 256、cap 0.08 固定、λ、horizon/exec_k=20、T/K=40/20、trust=same_obs、λ_raw=1e-3） | 同 | 同 |

> **硬伤**：这四项一起变 ⇒ 现在的"A vs B"**不是**"continue vs retrain"的单变量实验。
> 如果要回答"warm-start 有没有用"，必须补一组 `B' = 随机初始化 + A 的 lr/warmup/steps`
> 和一组 `A' = v5 warm-start + B 的 lr/warmup/steps`（2×2 设计，见 §9 的 Q2）。

### 5.2 结果总表（同一固定 val 96 窗口、同一 `J_base=1.50594`，可直接横比）

| 模型 | 初始化 / lr | steps | **best val ΔJ** | best@it | 末轮 val ΔJ | **test ΔJ** | 文件 sha256（前 16） |
|---|---|---|---|---|---|---|---|
| v5 best（A 的起点） | 随机初始化，3e-4 / 1e-4，warmup 50，**无物理损失** | 530（计划 1000） | +0.02877 | 450 | +0.02802（@500） | — | — |
| **A** | v5 warm-start，5e-5 / 2e-5 | 300 | **+0.02962** | **225** | +0.02823 | **+0.02185** | `5c318a5a029df5c7` |
| **B** | 随机初始化，3e-4 / 1e-4 | **960/1000** | **+0.02774** | **900** | +0.02675 | **+0.02074（已补）** | `60a1c5186eee8200` |

- ckpt 路径：
  - A：`runs/three_track_v6/delta_continue/seed0/checkpoint_actor_best.pt`（6,990,827 B，`it=225`，还另有 `checkpoint_actor_last.pt` @300）
  - B：`runs/three_track_v6/delta_retrain/seed0/checkpoint_actor_best.pt`（6,990,827 B，`it=900`，**没有 last**）
- 两者都在 **`comimic-wt03` worktree**，且文件 sha256 与 `runs/three_track_v7/phase0_freeze/inventory.json` 冻结值一致。

### 5.2b **重要更正：B 的真实身份是"v5 配方的重跑 + 物理损失"**（不是"小 lr 的 A 的对照"）

把三个 run 的配置并排看，会得到一个和"实验意图"不一致的结论：

| 项 | v5 main_v2（A 的起点） | **B**（Delta-Retrain） | A（Delta-Continue） |
|---|---|---|---|
| 初始化 | **随机** | **随机** | **v5 best** |
| head lr / enc lr | **3e-4 / 1e-4** | **3e-4 / 1e-4** | 5e-5 / 2e-5 |
| warmup | **50** | **50** | 20 |
| batch | 256 | 256 | 256 |
| 采样 | on-the-fly | on-the-fly | on-the-fly |
| 物理损失 `L_lim/L_peak` | **无**（v5 训练器） | **有**（v6 训练器） | 有 |
| 计划/实跑 updates | 1000 / **530** | 1000 / **960** | 300 / 300 |
| best val ΔJ | **+0.02877** | **+0.02774** | **+0.02962** |

**这张表的三个推论（请 reviewer 重点看）**：

1. **B 的 lr / warmup / batch / 采样器与 v5 完全相同**，唯一实质差别是"**多了 v6 的物理损失**"
   ⇒ B ≈ **"v5 的配方多跑约 2 倍步数 + 物理正则"** 的重跑；
2. 结果 **B（+0.0277）≈ v5（+0.0288）**，落在 ±0.003 的抖动带内
   ⇒ **在 WM ΔJ 这个指标上，"加物理损失"没有带来可测收益**（甚至名义上略低）；
   所以物理损失的合法性**不能靠 ΔJ 证明**，只能靠 §5.7 的物理指标（而 **B 的物理指标恰好没测**）；
3. 因此 **A 的 +0.0021 提升，目前唯一能归因的变量是"warm-start + 小 lr 精调"**，
   但这不是"因为继续训练"——而是"**因为把 lr 从 3e-4 降到 5e-5，在 v5 的解附近做了局部精修**"。
   这个解释与"A 更贴近 oracle 方向（99.4% vs 96.5%）"是自洽的。

> **想真正回答"warm-start 有没有用"，缺的是下面两个格子（现在一个都没有跑）**：
> `B' = 随机初始化 + 5e-5/2e-5 + 300 步` 与 `A' = v5 warm-start + 3e-4/1e-4 + 960 步`。
> 只有补齐 2×2 才能把"初始化"与"学习率"分开。详见 §9 Q2。

### 5.3 训练曲线（原始数据，reviewer 可以拿去画图/检验）

**A（val，每 25 步）**

| it | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | **225** | 250 | 275 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dJ | +0.02752 | +0.02658 | +0.02762 | +0.02775 | +0.02755 | +0.02879 | +0.02861 | +0.02847 | **+0.02962** | +0.02799 | +0.02833 | +0.02823 |

**B（val，每 50 步）**

| it | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | 650 | 700 | 750 | 800 | 850 | **900** | 950 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dJ | +0.0098 | +0.0106 | +0.0169 | +0.0188 | +0.0200 | +0.0167 | +0.0211 | +0.0224 | +0.0226 | +0.0215 | +0.0211 | +0.0253 | +0.0255 | +0.0268 | +0.0264 | +0.0239 | +0.0258 | **+0.0277** | +0.0268 |

**读法**：
- B 前 300 步在 0.01–0.02 打转（随机初始化要从零学"怎么改动作"），**约 600 步后才进平台**；
- A 是"**起步就在 v5 水平（0.0275）上小幅上探**"；
- 两者的**摆动幅度都在 ≈0.003** 量级，而 A−B 只差 **0.0019** ⇒ **排名不硬**。

### 5.4 三个模型与该"单步梯度候选值"的距离（⚠️ 措辞已修正，见增补 §6）

| 模型 | val ΔJ | 相对候选值（+0.0298）比例 |
|---|---|---|
| v5 best | +0.02877 | 96.5% |
| **A** | **+0.02962** | **99.4%** |
| **B** | +0.02774 | 93.1% |

候选值定义（`scripts/diag_oracle_upper_v5.py`，val 96 窗口）：
把 `Δ_t (t<K)` 当自由变量求 `g = ∂J/∂Δ`，取 `Δ* = cap·g/‖g‖`（逐窗口），量 `J(Δ*) − J0`；
实测 **+0.0298**（p50 +0.0270，**96.9% 的窗口为正**）。

> ⚠️ **必须修正的说法**：这只是"**在当前点沿一阶梯度方向、走满 cap 范数**"的一个候选修正，
> **不是**非线性 20 步 WM 目标下经过证明的最优上界（reviewer 的纠正成立）。
> 因此**不能**推出"A 只剩 0.00018 的提升空间"，也**不能**据此认定"必须放宽 cap"。
> 可以保留的弱结论是：**A 已经非常接近"一阶梯度 + 满 cap"这个解**，
> 说明它在 WM 世界里基本学到了"该往哪边推、推多满"，剩下的是**这个解本身够不够好**（与 cap 无关的部分）。

### 5.5 动作预算（同一个 cap 下的"性价比"）

> ⚠️ **单位更正（2026-11-08 复测）**：下表里所有 `rms / p99 / peak` 的 **"mrad" 标签是错的**。
> 它们来自 `action_budget_stats()`，输入 `e` 是 **`bundle.action_scale = ones` 的归一化动作域**
> （`COMIMIC_ACTION_SCALE` 默认 `one`）。**真物理值 = 该数 × `G1_ACTION_SCALE`（均值 0.396）**。
> 换算后的正确数字见增补文档 §3.2 与 §4.1（例：A 的轨迹级 rms 是 **51.8 mrad** 而不是 100.6 "mrad"，
> 全局峰值是 **2.093 rad** 而不是 5017 "mrad"）。

| 指标 | A@220 | B@900 | B@950 | 说明 |
|---|---|---|---|---|
| `‖Δ‖` | 0.0778 | 0.0774 | 0.0777 | **cap=0.08 顶死** |
| `‖u‖/cap` p50 | 2.48 | 2.36 | 2.42 | tanh 早已饱和 |
| `‖u‖/cap` p95 | 5.09 | — | 4.68 | — |
| `frac(‖u‖>2cap)` | **74.8%** | **70.1%** | 70.1% | — |
| `frac(‖u‖>3cap)` | **27.8%** | **21.3%** | 19.7% | — |
| 轨迹级 `rms`（**归一化域**） | 100.6 | 107.4 | 107.3 | 含状态发散；×0.396 才是 mrad |
| 轨迹级 `p99`（**归一化域**） | 482.6 | **534.0** | 529.2 | B 更"毛" |
| 轨迹级 `peak`（**归一化域**） | 5017 | 4740 | 4464 | 即 1.99 / 1.88 / 1.77 rad |
| **同观测**`rms`（**归一化域**） | 33.3 | 33.4 | 33.2 | 只隔离 Δ 的直接作用；≈13 mrad 物理 |
| **同观测**`p99`（**归一化域**） | 100.4 | 100.7 | 98.7 | — |
| **同观测**`peak`（**归一化域**） | 388.2 | 364.8 | 384.2 | ≈154 / 145 / 152 mrad 物理 |

**读法**：
- `‖Δ‖` 钉死在 cap、`‖u‖` 是 cap 的 2.4 倍 ⇒ **径向导数 `sech²(2.4) ≈ 0.032`**（幅度学不动），
  但**切向导数 `tanh(2.4)/2.4 ≈ 0.41`**（方向还能学）。
  ⇒ 原文"所有梯度被压掉 45–180×"的说法**只对径向成立**；v5 实测的 encoder 梯度小 45–180× 是端到端链路的结果，
  与"切向 0.41"并不矛盾，但**不能混为一谈**（reviewer 的这条纠正成立）；
- 轨迹级 `peak` 与同观测 `peak` **必须分开报**：前者含"两条轨迹跑散"，后者只反映 Δ 本身；
  历史上混用过，出过 `5.89 rad` 的假数（见 §7.6）。

> 数据来源说明：`‖u‖/cap`、`frac`、`‖Δ‖` 取自 `train.jsonl`（每 10 步一条；A 取最接近 best@225 的 **it=220**，
> it=300 为 p50 2.47 / >2c 74.0% / >3c 27.7%）；`rms/p99/peak` 取自 `val.jsonl`（A it=225、B it=900/950）。
> 两处都是"同口径下的 96 窗口 / 训练 batch"统计，差 3e-4 量级属采样噪声。

### 5.6 条件化强度（门禁通过 ≠ 条件化有效）

| | it | cos_shuffle_z | cos_shuffle_state | cos_shuffle_hist | 结论 |
|---|---|---|---|---|---|
| A | 25 | 0.6859 | 0.8117 | 0.8336 | PASS |
| A | 300 | 0.6546 | 0.8058 | 0.8124 | PASS |
| B | **25** | **0.9187** | 0.9012 | 0.9978 | **FAIL**（打乱 token 几乎不改变 Δ） |
| B | 50 | 0.7917 | 0.9372 | 0.9971 | PASS |
| B | 900 | 0.6482 | 0.8284 | 0.8240 | PASS |

- **B 的随机初始化连"用上 token"都要先学 ~50 步**；A 从第一步就 PASS（warm-start 的余温在这里非常直观）；
- 但 0.65 这个数**只说明"打乱输入会让输出变化"**，不说明"变化的方向对回报有利"。
  ⇒ 我们**没有**任何证据证明条件化是 reward-aligned 的（见 §9 Q1）。

### 5.7 A 的物理审计（❌ 已撤回：96 窗口复测不复现）

> **下面这张 24 窗口的表已作废**，保留只为说明"错在哪"。96 窗口、同一 ckpt、真物理口径的复测见
> **增补文档 §4**；新结论是：**A 的全局单关节峰值没有下降（2.027 → 2.093 rad），t=14 只降 42%，
> 尖峰迁移到 t=5/t=19；限位违反率反而从 2.701% 降到 2.640%。**

| 指标（24 窗口，❌ 不可复现） | 原 v5 best | A best | 变化 |
|---|---|---|---|
| Δ 对关节目标全局最大 | 2.065 rad | 1.847 rad | −11% |
| **Δ @ t=14（单关节）** | **2.065 rad** | **0.425 rad** | **−79%** |
| 限位违反率 | 2.74% | 2.90% | 持平 |
| `q_target` 绝对最大 | 4.530 rad | 4.510 rad | 持平 |

**为什么作废**（取证见增补 §4.2）：
1. 不是 ckpt 混用 —— A 的 `last@300` 复测 t=14 = **1.239 rad**、全局 2.093 rad，与 best 一致；
2. 不是单位错 —— v5 的旧值 2.065 与新复测 2.027 只差 2%，说明旧脚本量级是对的；
3. ⇒ 只能解释为**旧审计用了 24 个窗口的子集，恰好避开了 A 的尖峰窗口**。

**可以保留的物理正效应**（96 窗口、拆分越限后的真结论）：

| 指标（96 窗口，rad） | v5 | **A** | **B** |
|---|---|---|---|
| rms | 0.0608 | **0.0524（−14%）** | 0.0554 |
| p99 | 0.2239 | **0.1878（−16%）** | 0.2108 |
| 全局峰值 | 2.027 | 2.093 | **2.064** |
| Δ 新增越限 / Δ 修掉越限 | 1.279% / 1.643% | **1.223% / 1.676%** | 1.254% / 1.697% |

⇒ **物理约束的合法证据 = "rms/p99 降 + 净越限为正（修掉>新增）"，而不是"峰值尖峰"。**
**B 也已经做了物理审计**（原"最大空白"已填），结论：**与 A 基本同级，A 只在 rms/t=14 上略好。**

### 5.8 A 的真实闭环（唯一硬证据）

**协议**（`runs/three_track_v6/isaac/A_delta_continue/`）：

- server：`--delta-v5-ckpt A/best --delta-v5-hist-order ha_v31 --delta-v5-hist-absolute 1`，
  FT8000 + VLM，`--fm-steps 4`，端口 18095，`cuda:0`；
- client：`SONIC=1 SM_VLA_HIST10=1 ISAAC_GPU=1 MAX_STEPS=3000 REPEAT=0 RECORD_VIDEO=1 HEADLESS=1`；
- 规模：**3 任务 × 6 seeds = 18 集**；每 seed 最多 **2 次重试**（= 最多 3 次尝试；OpenDoor seed0 用满 3 次）；
- 判定：`result.json.success`（**sparse，成功才 reward=1.0**），取该 seed 所有 attempt 里最好的一次；
- 耗时：**2 h 27 min**。

**结果：9/18**

| 任务 | 成功 | 明细（成功那几次的步数） |
|---|---|---|
| OpenDoor | **3/6** | seed0 2625 / seed2 804 / seed4 1386 |
| SitSofa | **4/6** | seed1 2181 / seed2 2628 / seed4 2013 / seed5 958 |
| VisionNavi | **2/6** | seed4 2846 / seed5 1436 |

**参照点（不能混用）**：同一套解码链上，**v5 best** 在 OpenDoor 是 **2/2**（929 / 1633 步），
SitSofa seed0 失败（3000 步 timeout，未摔）。⇒ **A best 与 v5 best 是两个不同策略，闭环结果不可互相替代。**

---

## 6. 部署与闭环协议（复现 B 的闭环必须严格照这个来）

### 6.1 server 启动（A 的原始口径）

```bash
python3 <wt03>/scripts/ha_vla_latent64_server.py \
  --ckpt      <main>/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --vlm-path  <main>/groot_repro/nvidia/Cosmos-Reason2-2B-weights \
  --layout-check-npz '<main>/data/npz/ha_tokens/*.npz' \
  --state-order ha_v31 \
  --delta-v5-ckpt <wt03>/runs/three_track_v6/delta_continue/seed0/checkpoint_actor_best.pt \
  --delta-v5-state-order ha_v31 \
  --delta-v5-hist-order  ha_v31 \
  --delta-v5-hist-absolute 1 \
  --fm-steps 4 --port 18095 --device cuda:0
```

server 日志自证：`loaded params=1744832 T/K/H=40/20/20 cap=0.08 state_order=ha_v31 hist_order=ha_v31`，
每 20 次请求打印 `|Δ|mean(t<K)≈0.073–0.079 tail_zero=True`。

**给 B 跑闭环时只需要改三个东西**：`--delta-v5-ckpt` 指向 B、端口换 18097、输出目录换 `B_delta_retrain/`。
B 的 spec 与 A **逐字段相同**，server 无需改代码。

### 6.2 client 调用（Isaac 侧）

```bash
TASK=<OpenDoor|SitSofa|VisionNavi> SEED=<0..5> REPEAT=0 MAX_STEPS=3000 \
ISAAC_GPU=1 SONIC=1 SM_VLA_HIST10=1 \
SERVER_URL=http://127.0.0.1:18095 \
OUT_ROOT=<wt03>/runs/three_track_v6/isaac/<arm>/<task>_seed<k> \
RECORD_VIDEO=1 HEADLESS=1 \
bash <main>/scripts/ha_isaac_closedloop.sh
```

- **一次只能起一个 Isaac**（两个同时起会 `onnxruntime not available` / Isaac 扩展加载失败）；
- server 用 **cuda:0**，Isaac 用 **GPU1**；`ISAAC_GPU` 只影响 Isaac，不改 `CUDA_VISIBLE_DEVICES`；
- `scripts/run_closedloop_C_lora16.sh` 就是这套协议的**工程化版本**（flock 单实例、pgrep 串行化 Isaac、
  2 次重试、自动写 `RESULTS.md`/`summary.json`），给 B 复制它最省事。

### 6.3 红线（改了就没法跟 A 比）

1. **必须 `SONIC=1`**（Supermimic 不是部署路径，旧结论全部作废）；
2. **物理换算必须用 `G1_ACTION_SCALE`**（0.0745–0.5475），`action_scale=1.0` 会造出 5.89 rad 这种假数；
3. `hist10[-1]` 必须写成**当前**机器人状态（provider 在 decode 时才更新 hist，容易落后一帧）；
4. 关节序必须 `ha_v31`，错序会被 fail-closed 拦（真跑中拦下过）；
5. 训练采样必须带 `allowed_clips`（防 val/test 泄漏）；
6. `all_transitions.pt` / `wm_gain2.pt` 的 sha 必须与 `windows_meta.json` 对齐。

---

## 7. 问题清单（分层，带证据和建议）

> 每条标 **P0（阻塞/结论不成立） / P1（影响结论强度） / P2（工程）**。

### 7.1 目标函数层

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| **P0-1** | **A 已接近"单步梯度 + 满 cap"候选值（99.4%）** | §5.4 | 只是"这条一阶方向已经学到"，**不能**推出"没有提升空间"（见增补 §6） |
| **P1-2a** | **`L_act` 与 `L_same` 是同一个量、单位却不同**：`L_act` 用归一化域（τ=0.02 ⇒ 实际 **7.9 mrad**），`L_same` 用 rad（τ=0.02 ⇒ 20 mrad），两者权重都是 0.010 | 增补 §5.1 | **信任域被重复计算**，且真正生效的是"紧 2.5 倍"的那一项 ⇒ 设计意图（20 mrad）从未被执行 |
| **P1-2b** | `cap` 饱和：径向导数 `sech²(2.4)=0.032`、切向 `tanh(2.4)/2.4=0.41` | §5.5 | 饱和**限制幅度**、不限制方向；"条件化学不动"需要单独证据 |
| **P1-3** | ~~`λ_peak` 形同虚设~~ → **错，撤回**：加权梯度 `8.4e-4 × 72.05 = 0.0605` ≈ **30%×‖∇J‖**（`λ_lim` 为 50%） | 增补 §5 | 标定点上它是有效梯度；真正没验证的是**梯度方向、激活比例、裁剪后影响** |
| **P1-4** | **`L_same` 的 hinge 在结构上打不着**：实测 `rms_same = 13.05 mrad < τ=20 mrad` ⇒ 恒为 0 | 增补 §5.1 | 与 P1-2a 合起来看：**两个"信任项"一个太紧一直开、一个太松永远关** |
| **P2-5** | **`L_rate` BLOCKED**：`RobotSpec` 无 velocity 字段 | `calibration/delta.json` | 变化率约束完全缺席；`q_raw` 从 -1.96 跳到 2.39 rad 也没人管 |
| **P1-6** | `L_peak` 的预算是**自适应**的 `b_j = 3·mean_t(\|e_phys\|)_j` | §4.1 | 对"整体缩放"不变 ⇒ 只罚"某关节/某时刻相对突出"，**不限制绝对 rad**（reviewer 的推导正确） |
| **P0-7** | **全部"rad/mrad"日志的标签是错的**（归一化域标成 rad） | 增补 §3 | 任何"物理量"结论在修正前都不可信；§5.7 的作废就是这个原因 |

### 7.2 架构层

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| **P1-7** | **头占了 87% 参数**（1.526 M / 1.745 M），上下文编码器只有 ~133 K | §3.2 | 条件化能力天然受限；`cosZ≈0.65` 可能就是这个结构的**上限**，不是训练不够 |
| **P1-8** | `hist` 只用 **1 层 GRU 压成一个向量再广播** | §3.1 | "历史"只能提供全局状态，不能提供逐帧信息；B 在 it=25 时 `cos_hist=0.9978`（几乎不用历史）说明它学不到 history 的用法 |
| **P2-9** | 输出层 `init_std=4e-4` + bias=0 让第 0 步 ≈ Base | §3.1 | 好处是平滑起步；坏处是**随机初始化时 Δ 必须先"长出来"**，B 前 300 步的低 ΔJ 有相当部分是这个原因 |

### 7.3 优化层

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| **P1-10** | **A/B 不是单变量对照**；且 **B 的 lr/warmup 与 v5 原配方相同** ⇒ B 实质是"v5 配方 + 物理损失"的重跑 | §5.1、§5.2b | "B 没超过 A"既不能证明 warm-start 有用也不能证明无用；必须补 2×2 的两个空格子 |
| **P1-11** | **单 seed**（A 1 个、B 1 个；B 的 seed1 只跑 10 步、seed2 从未创建） | `train.jsonl` 行数 | 给不出均值±方差，任何"A vs B"都缺置信度 |
| **P1-12** | **val ΔJ 的抖动（≈0.003）> A−B 的差（0.0019）** | §5.3 | 用单点 best 比较会被噪声骗 |
| **P2-13** | B 用 **19.1 s/step**，960 步 ≈ 5.1 h；A 300 步 73 min | `train.jsonl`、`summary.json` | 采样（`t_data≈15 s`）是瓶颈，不是优化；想多 seed 必须先解决吞吐 |
| **P0-14** | **B 停在 960/1000 且没有 last ckpt** | `QUEUE_STATUS.txt`、目录 ls | 只能"独立 ckpt 评测"；想续跑只能 warm-start（优化器从 0），**严格 resume 不存在** |

### 7.4 评测层

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| **P0-15** | **闭环只有 sparse success**（9/18），没有稠密信号 | §5.8 | reward=0 ≠ 没进步；无法回答"第几 tick 开始偏" |
| **P1-16** | **闭环方差极大**：v5 best OpenDoor 2/2，A best 3/6 且 seed0 需 3 次尝试 | §5.8 参照点 | 9/18 与 2/2 不能直接比；样本量严重不足 |
| **P1-17** | **没有 base 臂**（不载 Δ 的对照）在 A 的 18 集里 | `RESULTS.md` | "Δ 有用"这件事在闭环上尚无同批对照 |
| **P2-18** | provider **没有逐 tick 落盘** `q_raw/q_applied/q_real/限位/任务进展` | 交付文档 §8 | 失败无法归因 |
| **P1-19** | **B 没有 test 指标** | §5.2 | 只有 val，无法排除 val 选择的偏差 |

### 7.5 物理层

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| **P1-20** | **两个"peak"口径长期混用**：轨迹级 4.7–5.0 rad vs 同观测 0.36–0.39 rad | §5.5 | 历史上出过 `5.89 rad` 假数并进过结论，**必须永远分开报** |
| **P1-21** | **B 完全没做物理审计** | §5.7 | 无法回答"Δ 修尖峰"是结构带来的还是 warm-start 带来的 |
| **P2-22** | 限位违反率 2.74%→2.90% **没有改善**，绝对值主要由 Base 贡献 | §5.7 | `L_lim` 到底有没有用，需要拆成 "Base 违反" 与 "Δ 额外违反" 两项 |

### 7.6 已修正的历史错误（写进文档防止再犯）

| 旧说法 | 现状 |
|---|---|
| 闭环 6/6 全失败 | ❌ **作废**：那 6 个 run 用的是 Supermimic（`SONIC=0`），部署不用 |
| 训练 decoder ≠ 部署 decoder | ❌ **作废**：两边都是同一个 SONIC decoder |
| 单关节尖峰 5.89 rad | ❌ **口径错**：那是 `action_scale=1.0`；正确 `G1_ACTION_SCALE` 下是 **2.07 rad @ t=14** |
| H=16 能消掉晚期爆炸 | ❌ **不成立**：t=14 在 H=16 内 |
| LoRA 没有提升空间 | ❌ 不严谨：oracle 只是"冻结 WM + 残差预算"的局部参考，不是 DiT LoRA 的上界 |
| A 的 val ΔJ 是 +0.02823 | ⚠️ **不精确**：那是 it=300 的末轮；**闭环用的是 best@225 = +0.02962** |
| B 的 best 在 it≈950 | ⚠️ **不精确**：ckpt 里 `it=900`，`dJ_val=0.027736` 与 val.jsonl 的 it=900 精确对齐 |

---

## 8. 未完成项（按性价比排序，reviewer 可以据此排优先级）

| # | 事项 | 成本 | 为什么值得做 |
|---|---|---|---|
| **0** | ~~B 的物理审计~~ | ✅ **已完成** | 96 窗口真物理口径：B 全局峰值 2.064 rad、rms 55.4 mrad，与 A 同级（增补 §4） |
| **0'** | ~~B 的 test ΔJ~~ | ✅ **已完成** | **+0.02074**（A +0.02185），见增补 §2 |
| **0''** | **单位修正**：把所有 `rad/mrad` 日志改成物理口径（或明确标注"归一化域"） | 代码 ~1 h | 不修则任何物理结论都可能重蹈 §5.7 的覆辙（增补 §3） |
| 1 | **合并 `L_act`/`L_same` + `L_peak` 固定预算** | 代码 ~1 h + 标定 10 min | 物理约束是否真的生效，取决于这一条（增补 §7） |
| 1 | **B 的 18 集闭环**（协议完全照 A） | ~2.5 h | 唯一硬证据；与 A 的 9/18 直接配对 |
| 2 | **闭环加 base 臂**（不载 Δ） | +2.5 h | 把"Δ 有用"从"和 v5 比"变成"同批对照" |
| 3 | **A/B 的 2×2 重跑**（随机初始化 × A 超参，warm-start × B 超参） | 4 组 × (1.3–5 h) | 唯一能回答"warm-start 有没有用"的设计 |
| 4 | **多 seed**（A/B 各 3 seeds） | ×3 | 现在全是 n=1，结论没有误差棒 |
| 5 | **cap 松到 0.10/0.12** 或 `λ_raw` 1e-3→1e-2 | ~30 min/组 | 验证"指标吃满"是不是 cap 瓶口 |
| 6 | **enc-lr 单独提到 1e-3** | ~30 min | 验证"条件化偏弱"是不是编码器学得慢 |
| 7 | **provider 逐 tick 落盘** | 代码 ~1 h | 闭环失败能定位到 tick |
| 8 | **WM 刷新（V7）** | 数小时 | A/B 的所有 ΔJ 都建立在**旧 A′ WM** 上；WM 本身误差可能大于 Δ 的收益 |

---

## 9. 给 reviewer 的具体问题（请逐条回答，并给可执行清单）

**Q1（最重要）** 在"A 已接近**单步梯度 + 满 cap 候选值**（99.4%，但这不是证明上界）"的前提下，
下一刀切哪里？请给出你的判断依据，并从以下选项里排序：
(a) 松 `cap` / 改输出的范数参数化（去掉 tanh 饱和）；
(b) 换目标：把 WM 的 saturating 跟踪奖励换成更陡的（`huber_err`）/ 加速度项权重；
(c) 换裁判：把"物理尖峰 + 闭环 success"直接写进训练目标（可微化怎么办？）；
(d) 换 WM：先把 A′ WM 刷新到 V7 版本，再重训 Δ；
(e) 换结构：把 87% 的参数从 MLP 头挪到 chunk 编码器 / 加深 GRU；
(f) 其他。

**Q2** 给我一个**能真正区分 A 与 B 的重跑设计**。已知现状：`A = v5 warm-start + 5e-5/2e-5 + 300 步`，
`B = 随机初始化 + 3e-4/1e-4 + 960 步`，而 `v5 本身 = 随机初始化 + 3e-4/1e-4 + 530 步`。
请写清：
- 你要的 2×2 是哪四组（我判断缺的是 `随机初始化 + 5e-5/2e-5 + 300 步` 与 `v5 warm-start + 3e-4/1e-4 + 960 步`），
  以及**每组的步数要不要对齐**（A 只有 300 步，和 B 的 960 步比 best 是否公平）；
- seed 数、每组 updates、用 val 还是 test 做 model selection；
- **显著性的判定口径**：现在 val 固定集 SE≈3e-4，但训练轨迹自身的抖动≈0.003，
  你建议用哪个当"最小可分辨差"、为什么？要不要改成多窗口×多噪声的配对检验？

**Q3** 审 §4.6 的 λ 反解：
- `λ_peak=8.4e-4` 是不是等于"这项形同虚设"？
- `L_peak` 用 `b_j = 3·mean|e|` 自适应预算，会不会被网络自己抬高？
- `L_same` 标定时梯度为 0 ⇒ 保留默认 λ=0.010，这个"不反解"的处理合理吗？

**Q4** 闭环只有 9/18，你建议怎么加**稠密信号**？
（候选项：参考轨迹跟踪误差、root 漂移、关节限位违反率、tilt、`q_raw` 的 tick 级统计。
需要你给"训练不用、只用于诊断"的指标定义，避免把评测信号又泄漏回训练。）

**Q5** 请给出**代码级改动清单**：文件 / 函数 / 具体改动 / 预期效果 / 验收指标。
约束见 §10；请明确哪些改动**会破坏与 A 的可比性**，必须单独成组跑。

---

## 10. 硬约束（改设计时不许碰）

1. **不改部署契约**：`state_hist_10` 的形状/顺序（HA_V31）、`hist[-1]=当前状态`、`cap` 的语义、
   `t≥K` 输出严格为 0、`Δ` 是 64 维 token 空间残差；
2. **不改 VLA 底座**（FT8000 冻结）；C 线是另一条路，不混进 A/B 的对照；
3. **不动 `G1_ACTION_SCALE` / 限位 npz / `default_joint_pos`** 的口径；
4. **物理换算必须走 `to_wm_action` 唯一入口**（物理→WM 域的转换只允许在那里发生）；
5. **一次只起一个 Isaac**；server 在 cuda:0，Isaac 在 GPU1；
6. **不许用 Supermimic 出结论**（`SONIC=0` 一律作废）；
7. **监督信息不许跨 split**：训练采样必须带 `allowed_clips`；
8. `all_transitions.pt` / `wm_gain2.pt` 的 sha 必须与 `windows_meta.json` 对齐，换 WM 要显式声明。

---

## 11. 复现命令（reviewer 若要自己核数）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
source scripts/harness03_env.sh

# 1) 物理限位（已完成，可复跑）
python3 scripts/export_g1_physical_limits.py

# 2) 物理 λ 标定（已完成；换 ckpt 要重标）
python3 scripts/calibrate_physical_bptt_weights.py --actor delta \
  --delta-checkpoint runs/temporal_ac_bptt_v5/main_v2/seed0/checkpoint_actor_best.pt \
  --limits runs/three_track_v6/physical/limits_standard29.npz \
  --windows 48 --out runs/three_track_v6/calibration/delta.json

# 3) A（已完成）
python3 scripts/train_delta_physical_bptt_v6.py --out runs/three_track_v6/delta_continue/seed0 \
  --on-the-fly 1 --eval-set runs/temporal_ac_bptt_v5/eval_set --updates 300 --batch 256 \
  --init-delta-checkpoint runs/temporal_ac_bptt_v5/main_v2/seed0/checkpoint_actor_best.pt \
  --physical-weights runs/three_track_v6/calibration/physical_weights_delta.json \
  --limits-npz runs/three_track_v6/physical/limits_standard29.npz \
  --head-lr 5e-5 --enc-lr 2e-5 --warmup-updates 20 --lambda-raw 1e-3 --trust-mode same_obs \
  --cap-warmup-updates 0 --eval-every 25 --cond-check-every 25 --seed 0

# 4) B（同脚本，去掉 --init-delta-checkpoint；lr 3e-4/1e-4；warmup 50；1000 updates；eval-every 50）
#    → 见 scripts/queue_three_track_v6_serial.sh

# 5) 离线物理审计（B 还没做）
python3 scripts/diag_action_budget_v5.py \
  --ckpt runs/three_track_v6/delta_retrain/seed0/checkpoint_actor_best.pt \
  --eval-set runs/temporal_ac_bptt_v5/eval_set --split val --device cuda:0

# 6) oracle 上界
python3 scripts/diag_oracle_upper_v5.py --eval-set runs/temporal_ac_bptt_v5/eval_set --split val \
  --ckpt runs/temporal_ac_bptt_v5/main_v2/seed0/checkpoint_actor_best.pt --device cuda:0 \
  --out /tmp/oracle_val.json

# 7) 补 B 的 test ΔJ（空循环 + 收尾 test；产物写在独立目录，别覆盖 B 的 run dir）
python3 scripts/train_delta_physical_bptt_v6.py \
  --out runs/three_track_v6/delta_retrain/seed0_testonly \
  --on-the-fly 1 --eval-set runs/temporal_ac_bptt_v5/eval_set --updates 0 --batch 256 \
  --init-delta-checkpoint runs/three_track_v6/delta_retrain/seed0/checkpoint_actor_best.pt \
  --eval-every 0 --cond-check-every 0 --seed 0 --device cuda:0
```

---

## 12. 文件清单（reviewer 需要看的代码只有这些）

| 文件 | 作用 |
|---|---|
| **`docs/A_B_ADDENDUM_MEASUREMENTS_20261108.md`** | **复测增补：B 的 test、真物理口径、三条撤回的结论（必读）** |
| `diff_vla_wbc/delta_token_temporal_v5.py` | **Δ 网架构**（本文件 §3） |
| `scripts/diag_delta_phys_v5.py` | **新增**：真物理口径审计 + Base/Δ 越限拆分（替代 `diag_action_budget_v5.py`） |
| `scripts/train_delta_physical_bptt_v6.py` | **A/B 训练器**（§4；loss 在 §4.1，采样 §4.2） |
| `diff_vla_wbc/physical_bptt_losses.py` | `L_lim / L_same / L_peak / L_rate / action_stats` |
| `diff_vla_wbc/sonic_physical_action.py` | `q_raw = q_default + G1_ACTION_SCALE ⊙ a`、安全层、`to_wm_action` |
| `scripts/calibrate_physical_bptt_weights.py` | λ 反解（§4.6） |
| `scripts/export_g1_physical_limits.py` | 限位导出 |
| `scripts/diag_oracle_upper_v5.py` | oracle 上界（§5.4） |
| `scripts/diag_action_budget_v5.py` | 物理审计 / `per_step_max_rad`（§5.7） |
| `scripts/ha_vla_latent64_server.py` | 部署 server（§6.1） |
| `scripts/ha_isaac_closedloop.sh` | Isaac 闭包 client（§6.2） |
| `scripts/run_closedloop_C_lora16.sh` | 闭环评测的工程化模板（给 B 用） |
| `runs/three_track_v6/isaac/A_delta_continue/{RESULTS.md,summary.json,eval.log}` | A 的 18 集原始结果 |

---

## 附录 A · 术语表

| 记号 | 含义 |
|---|---|
| `ΔJ` | `mean(J(Δ)) - J_base`，正数=比不加 Δ 好。`J` 为 20 步折扣回报（`γ=0.99`） |
| `J_base` | 冻结 VLA + 无 Δ 时的回报：val 1.50594、test 1.57581 |
| **轨迹级** rms/peak | 两条 WM 想象轨迹（有 Δ / 无 Δ）跑散之后的动作差，含状态发散 |
| **同观测** rms/peak | 同一份 obs 下、只换 token 重新解动作得到的差 —— **只反映 Δ 的直接作用** |
| `‖u‖/cap` | 输出层原始向量范数 / cap，`≈2.4` 说明 tanh 已饱和 |
| cond 门禁 | 打乱 `Z/state/hist` 后 Δ 的余弦相似度，`<0.9` 且非常量即 PASS（**弱判据**） |
| 单步梯度候选值（原称"oracle 上界"） | 逐窗口 `Δ* = cap·g/‖g‖`（`g=∂J/∂Δ`）的 ΔJ，val 上 **+0.0298**；**是一阶候选，不是证明上界** |
| 归一化动作域 | `bundle.action_scale = ones` 时的动作单位；`q = default + a`。**乘 `G1_ACTION_SCALE` 才是 rad** |
| A / B | Delta-Continue（warm-start）/ Delta-Retrain（随机初始化） |

## 附录 B · 直接把这张表贴给 reviewer 的"最小事实包"（2026-11-08 复测版）

```
冻结 VLA：FT8000（不训）           Δ 网：1.744 M，T/K/H=40/20/20，cap=0.08
val 集：96 窗口，J_base=1.50594    test 集：96 窗口，J_base=1.57581
A：warm-start v5，5e-5/2e-5，300 步 → val +0.02962 @225，test +0.02185，闭环 9/18
B：随机初始化，3e-4/1e-4，960/1000 步 → val +0.02774 @900，test +0.02074，闭环 未跑
单步梯度候选值：+0.0298 ⇒ A 99.4%、v5 96.5%、B 93.1%（注：一阶候选，非证明上界）
真物理（96 窗口，rad）：      v5      A       B
    rms                     0.0608  0.0524  0.0554
    p99                     0.2239  0.1878  0.2108
    全局单关节峰值          2.027   2.093   2.064      ← A 没有更好
    t=14 峰值               2.027   1.180   1.232      ← 只降 42%，不是 79%
    限位：base 2.701% → A 2.640% / B 2.696%；
          Δ 修掉的越限(1.68%) > 新增的(1.22%)  ⇒ 净效应为正
单位陷阱：日志里的 "mrad" 是归一化动作域；真 rad = ×G1_ACTION_SCALE（均值 0.396）
两个病灶：‖Δ‖ 钉死 cap（径向导数 0.032 / 切向 0.41）、条件化 cosZ≈0.65（门禁 <0.9）
代码级 bug：L_act(τ=0.02 归一化 ⇒ 实际 7.9 mrad) 与 L_same(τ=0.02 rad ⇒ 20 mrad) 重复；
           实测 rms_same=13.05 mrad ⇒ L_same 恒为 0
最大的设计硬伤：A/B 同时变了 4 个变量；且 B 的 lr == v5 原配方 ⇒ 不能归因到 warm-start
```
