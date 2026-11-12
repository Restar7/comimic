# COMIMIC · A/B/C 全面交底（架构 · 设计 · 问题 · 待定方案）—— 交给 reviewer 出修改方案

> 版本：2026-11-08 07:30 UTC　｜　代码：`comimic-wt03`，分支 `harness/03-bptt-diagnosis`，HEAD ≥ `bf42248`
> 本文目的：把**当前系统怎么搭的、跑到哪、哪里坏了、哪些数字作废了**一次讲清，让 reviewer（GPT）
> 产出**可直接执行的修改方案**（文件/函数/参数/验收口径）。
> 约定：只写**跑过或验证过**的东西；每条问题带证据出处；没跑的标 `NOT_RUN`。

---

## 0. 先回答一个概念问题：什么叫 "on-policy 数据"

系统里有三类数据，**它们的真实性和用途完全不同**：

| 类型 | 是什么 | 有没有真值 | 现在用来干什么 |
|---|---|---|---|
| **A′ replay**（`all_transitions.pt`） | 过去用**旧策略**在真实 Isaac 上跑出来的 `(state, action, next_state)` | ✅ 真值 | 训练 WM0；训练 Actor 时做想象起点 |
| **WM 想象**（imagination） | 把 token 喂给 WM，**推演** 20 步得到的轨迹 | ❌ 无真值（是模型的预测） | Actor 的 BPTT 梯度来源 |
| **on-policy 数据** | 用**当前这一个** A/B/C 策略，在真实 Isaac 里跑一遍，逐步记录 `状态→动作→下一状态` | ✅ 真值，且**分布=当前策略** | 只用于 **WM Refresh**（校准 WM） |

**为什么 WM Refresh 必须要 on-policy 数据**：

1. WM0 是拿**旧策略**的数据训的。现在的 A/B 已经不同了，会走到**旧数据里没出现过的状态**；
2. 在这些新状态上，WM 的预测**没有经过任何真实数据校准** ⇒ 想象出来的回报和梯度**不可信**；
3. 只对旧数据"重加权"（我们上一版做错的事）**不会带来任何新状态的真实标签**，
   WM 在没见过的状态上依旧是错的 ⇒ **不能算是 refresh**。

**采集链路**（必须是真的因果链，不能拿历史状态硬拼）：

```
当前 Actor → token chunk → 正式 SONIC decoder → q_raw → Isaac PD → 下一真实状态 → 记录
```

已有实例：`runs/three_track_v7/buffer/cycle000_A/`
（A 策略、OpenDoor、12 episodes / 7,552 ticks / **366 段 20 步**、`wm_v000`、动作域检查 `frac_outside=0.13%` 通过）。

**B 的 on-policy 数据目前不存在**（= 还没有人用 B 这个 checkpoint 去真实 Isaac 里跑过并落盘），
所以针对 B 的 WM 校准、OOD 检测目前都做不了。

---

## 1. 系统架构

### 1.1 闭环链路

```
Isaac Sim（HumanoidArena 3 任务：OpenDoor / SitSofa / VisionNavi）
   │ 相机 + 本体状态（q,qd,root）
   ▼
冻结 VLA：FT8000（Gr00tN1d7，6.9 GB，全冻结）
   │ 64 维 motion token chunk（T=40）
   ▼
【A/B】Δ 网（1.744 M）  ← 逐帧残差，只在前 K=20 帧加
【C】  LoRA16 + proj_out（23.07 M 可训）← 改 VLA 本体
   │ token
   ▼
SONIC decoder（ONNX，冻死）
   │ 29 维归一化动作（bundle 域：q = default + a）
   ▼
物理映射：q_raw = q_default(HA_ALIGN) + G1_ACTION_SCALE ⊙ a
   │
   ▼
限位/速率安全层 → q_applied → Isaac PD
```

- 控制 50 Hz（dt=0.02）；VLA 每 20 步重规划；T=40 / K=20 / H=20 / FM=4。

### 1.2 A/B 的 Δ 网（`diff_vla_wbc/delta_token_temporal_v5.py`）

| 输入 | 形状 | 说明 |
|---|---|---|
| `Z` | (B,40,64) | VLA 整段 token |
| `state64` | (B,64) | 决策时刻 `[rot6d(6) │ q(29) │ qd(29)]`，**HA_V31** 序 |
| `hist` | (B,10,58) | 过去 10 tick 的 `[q(29) │ qd(29)]`，HA 序，最老→最新 |

结构：`token_proj(64→128)` + `pe_chunk(40,128)` + 1×TransformerEncoder(d=128,h=4,ffn=256)
+ `hist_gru(58→128)` + `pe_pos(40,16)` + MLP 头 `400→1024→1024→64`（占 **87%** 参数）。
输出 `Δ = cap·tanh(‖u‖/cap)·u/‖u‖`，`cap=0.08`；`t≥20` 严格为 0。
注册参数 **1,744,832**；末层 `N(0,4e-4)`+bias=0 ⇒ 第 0 步 ≈ Base。

### 1.3 C 的 LoRA（`diff_vla_wbc/lora_ft8000_loader.py`）

- 底座：同一个 FT8000；`rank=16, alpha=32`；替换 **216 层**（attn1/attn2/ff）+ **`proj_out` 全秩**；
- 可训 **23,072,768**（其中 `proj_out` 6.30 M）⇒ **注意：这不是 LoRA-only**，
  旧文档"LoRA16 边际 +0.00548"是把 proj_out 也算进去的，命名必须改成 **C1 = LoRA16 + full-rank proj_out**。
- 另有 **C2 = 纯 LoRA16**（冻结 proj_out）尚未跑。

### 1.4 世界模型与版本存储

| 件 | 路径 | 状态 |
|---|---|---|
| WM0 | `runs/a_prime_sonic_exc/wm_gain2.pt`（sha `fc5aeb30`） | **A/B 当前使用** |
| A′ replay | `runs/a_prime_sonic_exc/all_transitions.pt`（529,131 行，370 MB） | 旧策略真值 |
| 版本库 | `runs/three_track_v7/wm_versions/{wm_v000, wm_probe_candidate, wm_c1_v001_published, active.json}` | 有原子发布/回滚 |
| **WM1** | `wm_c1_v001_published`（sha `81d24caf`） | ⚠️ **见 §5.1：门禁是 override，梯度探针没跑** |

### 1.5 数据与冻结件

| 件 | 事实 |
|---|---|
| 评测集 | `runs/temporal_ac_bptt_v5/eval_set/`：375 train / 81 val / 81 test clips；`n_eligible_frames=338,402`，训练过滤后 **235,280**；val/test 各 **96 窗口** |
| 物理限位 | `runs/three_track_v6/physical/limits_standard29.npz`（MuJoCo jnt_range 90% soft） |
| 物理 λ | `runs/three_track_v6/calibration/physical_weights_delta.json` |
| peak 固定预算 | `runs/three_track_v7/ab_fix_smoke3/peak_budget.json`（mean 0.2216 rad） |
| recovery z0 | `runs/three_track_v7/ab_v8_smoke/recovery_z0.json`（Q0.90: root_pos 0.645 / ori 2.510 / linvel 1.456 / angvel 3.142） |

---

## 2. 训练目标（**当前实际生效的**，不是文档里的理想版）

### 2.1 A/B（`scripts/train_delta_physical_bptt_v6.py`）

```
J_i = Σ_{t<20} γ^t · survival_t · reward_t          γ=0.99

loss = −mean J
     + λ_lim   · L_lim(q_raw)                  λ_lim = 0.0528(A) / 0.0570(B)   【活跃】
     + λ_added · L_added(q_raw, q_base)        λ_added = 0.053                【数值≈0，见 P1-3】
     + λ_R     · L_recovery(rollout.terms)     λ_R = 0.0019(A) / 0.0031(B)    【活跃，见 P0-2】
     + λ_raw   · Hinge(‖u‖/(2cap))             λ_raw = 1e-3                   【活跃】
     + λ_smooth· Smooth(e_phys)                λ_smooth = 0.002               【≈0】
     + λ_anchor· Anchor(Δ)                     λ_anchor = 0.001
     + λ_act   · Trust(e_phys, τ=0.02 rad)     λ_act = 0.010                  【**不激活**：14 mrad < 20 mrad】
     + λ_same  · L_same(e_phys, τ=0.02)        强制 0（与 L_act 重复，已去重）
     + λ_peak  · L_peak(e_phys, b_j)           λ_peak = 8.4e-4                【**不激活**：0.086 < 0.22 rad】
     + λ_rate  · L_rate                        BLOCKED（RobotSpec 无 velocity 字段）
```

**关键事实**：A/B 现在真正起作用的只有 **`J20 + L_lim + L_recovery + L_raw`**；
`L_act / L_same / L_peak` 都是"护栏"，**不产生塑形梯度**。

### 2.2 C（`scripts/train_sonic_lora16_ft8000_bptt.py`）

`J20 + λ_lim·L_lim + λ_peak·L_peak + λ_same·L_same + λ_margin·L_margin + λ_added·L_added + λ_R·L_recovery`
（λ 见 `checkpoint_lora_best.json` 的 `meta.physical`：lim 0.0369 / peak 2.06e-4 / same 0.01 / rate 0）

### 2.3 Recovery 项的实现（**这是 P0-2 的核心**）

`diff_vla_wbc/tracking_recovery_tail.py`：

```
z_k = e_k / std_k²            （k ∈ {root_pos, root_ori, body_linvel, body_angvel}）
u_k = relu(z_k − z0_k)
φ(u) = u²/(2δ)  (u ≤ δ)  ;  u − δ/2  (u > δ)      δ = 1
L_recovery = E[ Σ_t γ^t Σ_k w_k · φ(u_{k,t}) ]    z0_k 由 **Train 的 Q0.90** 冻结
```

**它"不压尾部"的四个机理**（见 P0-2）：
1. `z0=Q0.90` ⇒ 激活的是 **10–18% 的"刚过线"样本**，极端样本占比很小；
2. Huber 在 `u>δ` 时**梯度恒为 1**（刻意防爆）⇒ **越差不会给越多梯度**；
3. 目标是 **batch 平均**，不是 max / 分位数 ⇒ 极少数最差样本权重极小；
4. **训练分布里几乎没有大误差样本**（`start_max_drift=0.1` + H=20，实测 `z_p95 ≈ 0.8–1.0`）。

---

## 3. 训练/评测协议

### 3.1 采样（A/B）

- **on-the-fly**：每 update 现抽帧 → 现跑 10 步 burn-in → 截 20 步窗口 → 现算 VLA token；
- 只允许 **375 个 train clips**（`allowed_clips` 过滤），val/test clip 一帧不取；
- **困难采样（V8 新增）**：`70% 均匀 + 30% 困难`，
  困难度 = `‖state[:3] − ref[:3]‖ + 0.5·‖rot6x − ref6x‖`，阈值取合格帧的 p70（实测 0.3197）；
- 扰动：`start_max_drift=0.1`、`perturb_xy=0.05`、`perturb_yaw=0.10`、`perturb_dq=0.05`、`perturb_dqd=0.30`。

### 3.2 评测

- 固定 val/test 各 96 窗口；`ΔJ = mean J(Δ) − J_base`；
  **A/B 的 base：val 1.50594 / test 1.57581**；
- 统计不确定度 `SE≈3e-4`；**但训练曲线自身抖动 ≈0.003**（比 SE 大一个量级）。

### 3.3 闭环（唯一硬证据）

- server：`scripts/ha_vla_latent64_server.py`，cuda:0，端口 18095/18096；
- client：`scripts/ha_isaac_closedloop.sh`，`SONIC=1 SM_VLA_HIST10=1 ISAAC_GPU=1 MAX_STEPS=3000`；
- **一次只能起一个 Isaac**（两个同时起会 `onnxruntime not available`）；
- A 的成绩：**3 任务 × 6 seeds = 18 集，9/18**，但**允许每 seed 2 次重试取最好** ⇒ **不是首次成功率**。

---

## 4. 当前结果（已校正单位/口径）

### 4.1 WM 指标（同一 WM0、同一 96 窗口；A/B 可直接横比）

| 模型 | 初始化 | 步数 | val ΔJ（best@it） | test ΔJ |
|---|---|---|---|---|
| A 原 | v5 warm-start，5e-5/2e-5 | 300 | +0.02962 @225 | **+0.02185**（final） |
| **A_v8** | A best@225，3e-5/1e-5 | +200 | **+0.03291 @100** | **+0.02477（best）/ +0.02577（final）** |
| B 原 | 随机，3e-4/1e-4 | 960 | +0.02774 @900 | **+0.02074** |
| **B_v8b** | B best@900，3e-5/1e-5 | +200 | **+0.0301~0.0311**（进行中，it=130） | 待跑完 |
| C1 旧（lr1e-5） | FT8000 | 800 | +0.00548（LoRA 边际） | — |
| C1 新（DDP lr1e-4，bs64） | FT8000 | 300 | +0.00372 @250（**震荡**） | — |

⚠️ **C 的 base 与 A/B 不同**（C: val 1.49194 / test 1.57558；A/B: 1.50594 / 1.57581）
⇒ **C 的 ΔJ 不能与 A/B 直接横比**。

### 4.2 真物理（96 val 窗口，`scripts/diag_delta_phys_v5.py`，已修默认角偏置）

| 指标（rad） | v5 | A 原 | B 原 | A_v8（it=200） |
|---|---|---|---|---|
| 同观测 rms | — | 0.0141 | 0.0142 | 0.0155 |
| 轨迹级 rms | 0.0608 | 0.0524 | 0.0554 | — |
| 轨迹级 p99 | 0.2239 | 0.1878 | 0.2108 | — |
| 全局单关节峰值 | 2.0265 | 2.0928 | 2.0641 | —（`e_phys_absmax` 0.086→**0.214**） |
| 限位违反（base 4.867%） | 4.883% | 4.939% | 4.989% | — |

**严格越限分解**（恒等式 `mod = base − fixed + new`，残差 1e-9）：

| 模型 | 真正修掉 `fixed` | 真正新增 `new` | 净变化 |
|---|---|---|---|
| v5 | 0.431% | 0.447% | +0.016pp |
| A | **0.375%** | **0.447%** | +0.072pp |
| B | 0.370% | 0.492% | +0.122pp |

### 4.3 A_v8 的"代价"（必须一起报）

| 指标 | 首 5 条均值 | 末 5 条均值 | 变化 |
|---|---|---|---|
| `L_recovery` | 20.91 | 14.93 | −28.6% |
| `rec_root_pos_z_p95` | 0.880 | 0.884 | **+0.4%（持平）** |
| `rec_root_ori_z_p95` | 1.688 | 1.919 | **+13.7%（变差）** |
| `rec_body_linvel_z_p95` | 1.570 | 1.539 | −2.0% |
| `act_rms_rad` | 0.0146 | 0.0155 | +6.5% |
| `e_phys_absmax` | 0.136 | **0.214** | **+57.4%** |

⇒ **WM 分数提升是真的，但"恢复能力"没被证明，动作修正反而变大。**

---

## 5. 问题清单（编号，供 reviewer 逐条给方案）

### 5.1 P0 —— 会直接导致结论错误

| # | 问题 | 证据 | 现状 |
|---|---|---|---|
| **P0-1** | **WM1（`wm_c1_v001_published`）是在"梯度探针未跑"的情况下发布的**：`gate.json` 里 `dir_agree.passed=false`、`override.reason=gradient_probe_not_run`。按 V7 §7 这**不算有效发布** | `wm_versions/wm_c1_v001_published/gate.json` | 仍被设为 active；**A/B 已显式钉回 wm_v000** |
| **P0-2** | **`L_recovery` 不压尾部**（四个机理见 §2.3）⇒ `L_recovery −29%` 但 `z_p95` 不动/变差 | §4.3 | 未修；正在跑"起点漂移 0.25"实验 |
| **P0-3** | **WM Refresh 路线错误**：上一版只对**旧 A′ 数据重加权**，没有新真值 ⇒ 违反 V7 §3.1；脚本已标弃用 | `scripts/reweight_replay_for_wm_v8.py` | 需改成 V7 五步流程（§6） |
| **P0-4** | **A/B 的对照不成立**：A 是 warm-start + 5e-5，B 是随机初始化 + 3e-4（且**B 的 lr 恰等于 v5 原配方**）⇒ 无法归因到"初始化" | 配置对比 | 未修；V8 后两条都是 3e-5/1e-5，但**仍是不同初始化** |
| **P0-5** | **单 seed**（A/B/C 各 1 个；B 的 seed1 只 10 步、seed2 从未创建） | `train.jsonl` | 未修 |
| **P0-6** | **checkpoint 不含 optimizer/scheduler/RNG** ⇒ 不存在严格 resume；"B 补 960→1000"只能 warm-start | `_save_ckpt` | 未修（V8 明确定义为新的 warm-start run） |
| **P0-7** | **闭环协议是"重试取最好"**：A 的 9/18 不是首次成功率，不能作为"策略成功率" | `isaac/A_delta_continue/RESULTS.md` | 未重跑 |

### 5.2 P1 —— 影响结论强度 / 造成口径混乱

| # | 问题 | 证据 |
|---|---|---|
| **P1-1** | **`L_added` 的定义与名字不符**：它罚 `relu(v_mod − v_base)²`，即"恶化"（2.8% 样本），而"真正新增越限"只有 **0.45%** ⇒ 它实际是 `L_lim` 的软版本，**不能当"新增越限已约束"的证据** | §4.2 分解 |
| **P1-2** | **`L_act/L_same/L_peak` 三个"物理项"全都不激活**（14 mrad < 20 mrad；0.086 < 0.22 rad）⇒ 物理层现在只剩 `L_lim` | `train.jsonl` |
| **P1-3** | **`λ_added` 不能反解**：96 窗口下 `‖g_added‖≈5e-4`，反解得 **210–323**，会爆；现固定 0.053 | `calib_{A,B}_v8.json` |
| **P1-4** | **`L_rate` BLOCKED**：`RobotSpec` 无 velocity 字段 | `calibration/delta.json` |
| **P1-5** | **日志单位标签历史性错误**：`train.jsonl` 的 `*_mrad` 实为**归一化动作域**（×`G1_ACTION_SCALE` 均值 0.396 才是 rad） | 代码 + 换算验证 |
| **P1-6** | **三个脚本对同一 ckpt 报三种动作预算**（训练器 `eval_split` 用缓存 base / 诊断脚本现算），差 ~30% | 实测 |
| **P1-7** | **条件化只被验证到"不是常量"**（`cos_shuffle_z≈0.65`，门禁阈值 <0.9），**没有 reward 对齐证据** | `cond_check.jsonl` |
| **P1-8** | **C 的 base 与 A/B 不同**（1.49194 vs 1.50594）⇒ C 的 ΔJ 与 A/B 不可比 | `base_eval.json` |
| **P1-9** | **C 的命名错误**：旧文档写 "LoRA16 边际 +0.00548"，但那包含 `proj_out` 全秩 6.30 M；C2（纯 LoRA）未跑 | `checkpoint_lora_best.json` |
| **P1-10** | **C 新跑 lr=1e-4 是震荡的**（0.00023→0.00239→0.00036→0.00144→0.00372→0.00285） | `val.jsonl` |

### 5.3 P2 —— 工程/流程

| # | 问题 |
|---|---|
| **P2-1** | 闭环没有逐 tick 落盘（`q_raw/q_applied/q_real/限位/tilt/进度`）⇒ 失败无法归因 |
| **P2-2** | **共享 worktree 有并发 agent**：本轮已发生一次"WM 被换而 B 训练 pin 到新版本"的事故（已修，见 §7） |
| **P2-3** | `runs/` 未纳入版本管理，checkpoint 与代码 commit 的对应关系靠手写 |
| **P2-4** | `train_wm_refresh_v7.py` 依赖 `trackerLab`（需 `source scripts/harness03_env.sh`），直接 `--help` 会 `ModuleNotFoundError` |

---

## 6. WM Refresh 的正确设计（V7 五步）+ 现有资产

### 6.1 流程（reviewer 需按此给方案）

```
① collect：当前策略在真实 Isaac 采 128 窗口（每窗口 20 步）
② buffer ：Shared Mixed Replay = 70% A′ + 30% on-policy（按 episode 切分；另留 90/10 作对照）
③ train  ：从 WM0 **复制**候选 → 300 updates（lr 1e-5，H∈{1,5,10,20} 多步监督）
④ gate   ：新策略 20 步误差↓（≥5%）、旧 A′ 不遗忘（退化≤5%）、
            **梯度方向探针 dir_agree 不劣于旧 WM**、无 NaN、动作合同一致、无泄漏
⑤ publish：PASS ⇒ 原子发布新版本；FAIL ⇒ REJECTED，Actor 继续用旧 WM
```

**§13 的关键对照**（V7 明确要求，目前完全没做）：

```
同一 Actor ckpt θ0
 ├── Fixed-WM：θ0 + WM0 → 100 updates
 └── Refreshed：θ0 + WM1 → 100 updates
两者在 **两个 WM 上分别评估** + 在**独立真实短轨迹**上评估预测/物理/20 步真实回报
```
只有这样才能区分"WM 真变准了帮了 Actor" vs "换 WM 只是让训练目标数字更好看"。

### 6.2 现有资产（可直接用）

| 件 | 状态 |
|---|---|
| `scripts/collect_wm_refresh_isaac.py` | ✅ 存在（`--policy-id --policy-checkpoint --wm-version --task --seeds --max-steps --horizon --ticks --convert-only --max-segments` 等） |
| `scripts/diagnose_wm_shift_v7.py` | ✅ 存在（`--wm-checkpoint --old-replay --onpolicy-data --horizons --windows --out`） |
| `scripts/train_wm_refresh_v7.py` | ✅ 存在（`--init-version --new-version --old-ratio --new-ratio --updates --lr --mode --k ...`） |
| `scripts/eval_wm_refresh_v7.py` | ✅ 存在（`--wm-old --wm-candidate --gradient-probe-result --allow-prediction-only --skip-gradient --publish --reject-on-fail`） |
| `diff_vla_wbc/wm_replay_buffer.py` | ✅ 存在（`MixedReplay`、`validate_transitions`、`assert_action_domain`、`build_kstep_windows`） |
| `diff_vla_wbc/wm_refresh_gate.py` / `wm_version_store.py` | ✅ 存在（门禁 + 原子版本 + 回滚） |
| **A 的 on-policy 数据** | ✅ `buffer/cycle000_A`（366 段 / OpenDoor / A ckpt / 动作域 PASS） |
| **B 的 on-policy 数据** | ❌ **不存在**，需 `collect_wm_refresh_isaac.py` 采（估 25–60 min） |
| **梯度探针** | ❌ 从未跑过（这正是 WM1 override 的原因） |

---

## 7. 本轮已修/已发生的事（时间线）

| 时间 | 事件 |
|---|---|
| 05:46 | **A/B 物理层 5 个 bug 修复**并单独提交 `941f7a9`（`default_joint_pos` 口径、信任项真 rad 域、`L_same` 去重、`L_peak` 固定预算、物理诊断落盘） |
| 05:51 | A_v8 启动（WM0） |
| 06:29 | ⚠️ 并发会话把 active WM 切到 `wm_c1_v001_published` |
| 06:33 | A_v8 完成（val +0.03291 / test +0.02577）；B_v8 启动 → **pin 到 WM1** ⇒ 作废 |
| 06:49 | 发现事故：kill B_v8 + 两个队列；重写为 `--wm-version wm_v000 --strict-wm-hash fc5aeb30` + **跑完回读打印 OK/WRONG** |
| 06:50 | B_v8b 启动（WM0）；A_v8 best 的 test 在 WM0 上重测 = +0.02477 |
| 07:19 | B_v8b 到 it=130（val +0.0301~0.0311），漂移两组仍在排队 |

**教训**：共享 worktree 里任何"发布新 WM"的动作都会影响**下一个**启动的 run；训练前必须回读 `config.json.wm_pin.sha256`。

---

## 8. 正在跑 / 已排队的实验

| 实验 | 配置 | 目的 | 状态 |
|---|---|---|---|
| `A_v8` | A best + 200 步，drift 0.10，z0 Q0.90 | V8 配方基线 | ✅ 完成 |
| `B_v8b` | B best + 200 步，drift 0.10 | B 上复现 V8 配方 | 🔄 it≈130/200 |
| `A_d25_q90` | A best + 100 步，**drift 0.25**，z0 由新分布冻结 | 让"恢复"真的被练到 | ⏳ 排队 |
| `A_d25_z2` | A best + 100 步，**drift 0.25**，z0 固定 2.0 | 隔离"只加难度"的贡献 | ⏳ 排队 |

全部 pin 在 `wm_v000`（`fc5aeb30`），脚本每次跑完回读校验。

---

## 9. 请 reviewer 回答/给方案的问题（**这是本文的重点**）

### Q1 · Recovery 怎么才能真正压住尾部？（P0-2）
现在 `z0=Q0.90` + Huber(δ=1) + batch 均值 ⇒ 只压"刚过线"的样本。
请给出**具体改法**并说明为什么：抬 `z0` 到 Q0.97/Q0.99？改 δ？改凸尾 `u²`？
改成**尾部加权**（`w ∝ relu(z−z0)`）？还是改成 **CVaR / 分位数目标**？
以及：**如何在不引入爆炸的前提下让梯度随误差增长**？

### Q2 · 起点漂移要多大？（P0-2 的另一半）
`start_max_drift=0.1` 让训练里没有大误差样本。V7 §4.2 要求"漂移状态"起点。
请给出：drift 取多少（0.25？0.4？）、是否需要 **root 横向/航向的定向扰动**、
以及如何保证"仍在 WM 支持范围内"（不能把 OOD 状态当训练起点）。

### Q3 · WM Refresh 的完整落地顺序（P0-1/P0-3）
按 §6 五步，请给出：
1. A（已有数据）与 B（需采集）分别怎么做、先做谁；
2. `eval_wm_refresh_v7.py --gradient-probe-result` 的**探针怎么产生**（谁算 `ΔJ_Isaac`？需要多少真实窗口？）；
3. **`wm_c1_v001_published` 该怎么处置**：回滚到 wm_v000？降级为 candidate 重新过门禁？
4. §13 的 Fixed-vs-Refreshed 对照，用哪个 actor ckpt、多少步、评什么。

### Q4 · A/B 的训练配方是否还要继续加项？（P0-4/P1-2）
现在物理三项都不激活，只有 `L_lim + L_recovery + L_raw` 起作用。
请判断：是**继续加项**，还是**先证明现有项有效**（A_v8 的 test +18% 是否足以停手）？
以及 A/B 是否值得再做 2×2（初始化 × 学习率）。

### Q5 · C 的路线怎么收（P1-8/P1-9/P1-10）
1. C 的 base 与 A/B 不同，**要不要统一评测口径**？
2. **C1（LoRA+proj_out）与 C2（纯 LoRA）** 的对照怎么做、预算多少？
3. 分组 LR（LoRA 1e-4 / proj_out 1e-5）是不是比现在一起训更合理？
4. C 新跑的震荡（§5.2 P1-10）是 lr 太大还是 batch 太小？

### Q6 · 闭环协议怎么定（P0-7）
"重试取最好"不能当成功率。请给出：**首次尝试成功率**为主指标、
重试率/成功率作为辅助、每任务多少 seed 才够、失败如何归因（要哪些逐 tick 字段）。

---

## 10. 红线（改方案时不许碰）

1. 不改部署契约：`state_hist_10` 形状/顺序（HA_V31）、`hist[-1]=当前状态`（tol 5e-3）、
   `cap` 语义、`t≥K` 严格 0、`Δ` 是 64 维 token 残差；
2. 不改 VLA 底座（FT8000 冻结）；C 线不混进 A/B 对照；
3. 物理换算只走 `G1_ACTION_SCALE`（**不许** `action_scale=1.0`）；
4. **一次只起一个 Isaac**；server 在 cuda:0、Isaac 在 GPU1；
5. 不许用 Supermimic 出结论（`SONIC=0` 一律作废）；
6. 训练采样必须带 `allowed_clips`（防 val/test 泄漏）；
7. **任何 Actor run 必须显式钉住 WM 版本 + 回读校验**；
8. 旧实验目录 `runs/three_track_v6/`、`runs/a_prime_sonic_exc/` **只读**。

---

## 11. 复现命令（关键几条）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
source scripts/harness03_env.sh

# A/B 的 V8 训练（已用）
python3 scripts/train_delta_physical_bptt_v6.py --out <OUT> \
  --on-the-fly 1 --eval-set runs/temporal_ac_bptt_v5/eval_set --updates 200 --batch 256 \
  --init-delta-checkpoint <A|B best> \
  --physical-weights runs/three_track_v6/calibration/physical_weights_delta.json \
  --limits-npz runs/three_track_v6/physical/limits_standard29.npz \
  --lambda-lim 0.0528 --lambda-added 0.053 --lambda-recovery 0.0019 \
  --recovery-z0-json runs/three_track_v7/ab_v8_smoke/recovery_z0.json \
  --start-max-drift 0.10 --hard-frac 0.3 \
  --trust-domain physical --peak-budget-mode fixed \
  --peak-budget runs/three_track_v7/ab_fix_smoke3/peak_budget.json \
  --head-lr 3e-5 --enc-lr 1e-5 --warmup-updates 20 \
  --eval-every 25 --cond-check-every 25 --seed 0 --device cuda:0 \
  --wm-version wm_v000 --strict-wm-hash fc5aeb30

# 真物理审计
python3 scripts/diag_delta_phys_v5.py --ckpt <ckpt> --eval-set runs/temporal_ac_bptt_v5/eval_set \
  --split val --device cuda:1 --limits runs/three_track_v6/physical/limits_standard29.npz \
  --out /tmp/phys.json

# λ 标定（含 recovery / added）
python3 scripts/calibrate_physical_bptt_weights.py --actor delta --delta-checkpoint <ckpt> \
  --limits runs/three_track_v6/physical/limits_standard29.npz --windows 96 \
  --peak-budget runs/three_track_v7/ab_fix_smoke3/peak_budget.json \
  --recovery-z0 runs/three_track_v7/ab_v8_smoke/recovery_z0.json \
  --out /tmp/calib.json

# WM Refresh（V7 正确入口；B 需要先 collect）
python3 scripts/collect_wm_refresh_isaac.py --out runs/three_track_v7/buffer/cycle000_B \
  --policy-id B_delta --policy-checkpoint <B best> --wm-version wm_v000 \
  --task OpenDoor --seeds 0 1 2 3 --max-steps 600 --horizon 20
python3 scripts/diagnose_wm_shift_v7.py --wm-checkpoint runs/a_prime_sonic_exc/wm_gain2.pt \
  --onpolicy-data runs/three_track_v7/buffer/cycle000_B --horizons 1,5,10,20 \
  --out runs/three_track_v7/diagnostics/wm0_B
```

---

## 附录 A · 指标口径（防再混）

| 口径 | 定义 | 用途 |
|---|---|---|
| **归一化动作域** | `bundle.action_scale=ones` 时 `q = default + a`；日志里的 "mrad" 通常是这个 | **乘 `G1_ACTION_SCALE`（均值 0.396）才是 rad** |
| **同观测 `e`** | 同一份 obs93 只换 token 解出的动作差 | 部署时的**真实扰动量**（A/B ≈ 14 mrad） |
| **轨迹级 `e`** | 两条想象轨迹跑散后的动作差 | 含行为差异，**不是违规**（可达 2 rad） |
| 单步梯度候选值 | `Δ*=cap·g/‖g‖` 的 ΔJ（val +0.0298） | **一阶候选，不是证明上界** |

## 附录 B · 文件清单

| 文件 | 作用 |
|---|---|
| `diff_vla_wbc/delta_token_temporal_v5.py` | Δ 网架构 |
| `diff_vla_wbc/tracking_recovery_tail.py` | Recovery（`huber_tail` / `recovery_loss` / z0 冻结） |
| `diff_vla_wbc/physical_bptt_losses.py` | `l_lim / l_same / l_peak / l_added_limit / l_margin / l_rate` |
| `diff_vla_wbc/sonic_physical_action.py` | `q_raw = q_default + G1_ACTION_SCALE ⊙ a`、安全层、`to_wm_action` |
| `diff_vla_wbc/wm_replay_buffer.py` / `wm_refresh_gate.py` / `wm_version_store.py` | Mixed replay / 门禁 / 原子版本 |
| `scripts/train_delta_physical_bptt_v6.py` | A/B 训练器 |
| `scripts/train_sonic_lora16_ft8000_bptt.py` | C 训练器 |
| `scripts/calibrate_physical_bptt_weights.py` | λ 反解 |
| `scripts/diag_delta_phys_v5.py` | 真物理审计 + 越限严格分解 |
| `scripts/collect_wm_refresh_isaac.py` / `diagnose_wm_shift_v7.py` / `train_wm_refresh_v7.py` / `eval_wm_refresh_v7.py` | V7 WM Refresh 四件套 |
| `scripts/queue_ab_v8.sh` / `queue_ab_v8_wm0.sh` / `queue_ab_drift_v8.sh` | 本轮队列（含 WM 硬钉 + 回读校验） |
| `runs/three_track_v7/ab_v8/{A_v8,B_v8b,A_d25_*}` | 本轮产物 |
