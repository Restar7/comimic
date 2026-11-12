# COMIMIC · A/B 现状 · 架构 · 问题 · 计划（2026-11-08 修复后交接版）

> 取代：`A_B_HANDOVER_FOR_REVIEW_20261108.md`、`A_B_ADDENDUM_MEASUREMENTS_20261108.md`
> （那两份里的**物理数字、单位、`L_same` 结论**已被本轮三次复测推翻，仅作历史留档。）
> 本文只写**跑过/验证过**的东西；每个数字标出处；没跑的标 `NOT_RUN`。
> 代码 worktree：`comimic-wt03`；产物：`runs/three_track_v7/ab_diagnostics/`、`runs/three_track_v7/ab_fix_smoke*/`

---

## 0. 摘要：现在能不能直接长训？

**不能。** 修完 5 个 bug 之后，正确的下一步是「**100–200 步 pilot + 闭环 base 臂**」，不是长训。三个理由：

1. **修复让物理约束变松，而不是变紧**：真 rad 口径下 A/B 的同观测偏差只有 **14.1 mrad** < τ=20 mrad
   ⇒ `L_act` **从"一直开"变成"基本不开"**；`L_peak` 在固定预算 0.22 rad 下也**恒为 0**。
   ⇒ 修复后训练实际只剩 `J20 + L_lim + L_raw + L_smooth + L_anchor`，**必须先看 pilot 的物理画像有没有变坏**。
2. **WM 指标已经饱和**：A 300 步就到"单步梯度 + 满 cap"候选值的 **99.4%**；B 训 960 步只到 **93.1%**，
   且 it≥600 后 plateau。**加步数不是有效变量**。
3. **唯一未验证的硬变量是闭环**：A 9/18 没有 base 臂、方差极大。这决定"Δ 到底有没有用"，长训回答不了。

> 一句话：**先花 6 小时确认"修完的东西没有变坏 + Δ 在真机上有效"，再决定要不要花 48 小时长训。**

---

## 1. 现状（截至 2026-11-08 04:30 UTC）

### 1.1 三条线

| 线 | 做法 | 状态 | 产物 |
|---|---|---|---|
| **A** Delta-Continue | Δ 网 warm-start 自 v5 best，head 5e-5 / enc 2e-5，300 步 | **完成**：val / test / 物理 / 18 集闭环都有 | `runs/three_track_v6/delta_continue/seed0/` |
| **B** Delta-Retrain | 随机初始化，head 3e-4 / enc 1e-4，1000 步（**实跑 960 被人工停**） | **训练完整、评估完整**（test/物理本轮补齐） | `runs/three_track_v6/delta_retrain/seed0/` |
| C LoRA16 | 改 VLA 本体（LoRA+proj_out） | 与本文件无关；B 之外的另一条路 | — |

### 1.2 关键数字（**单位已校正**：真物理 = 归一化域 × `G1_ACTION_SCALE` 均值 0.396）

**WM 指标**（同一固定 val/test 各 96 窗口；`J_base` val 1.50594 / test 1.57581）

| 模型 | 初始化 | 步数 | val ΔJ（best@it） | test ΔJ |
|---|---|---|---|---|
| v5 best | 随机，3e-4/1e-4 | 530（计划 1000） | +0.02877 @450 | — |
| **A** | v5 warm-start，5e-5/2e-5 | 300 | **+0.02962 @225** | **+0.02185** |
| **B** | 随机，3e-4/1e-4 | 960 | **+0.02774 @900** | **+0.02074** |

- 单步梯度候选值（`Δ*=0.08·∇J/‖∇J‖`）：**+0.0298**（p50 +0.0270，96.9% 窗口为正）⇒ A 99.4%、v5 96.5%、B 93.1%。
  ⚠️ 这是**一阶候选**，不是证明上界。
- A−B：val 差 0.0019、test 差 **0.0011**；两者都 < 训练曲线抖动（≈0.003）、且**单 seed**
  ⇒ **"A 优于 B"不成立**，只能写"相当"。

**真物理**（96 val 窗口，同一脚本 `scripts/diag_delta_phys_v5.py`，已修默认关节角偏置）

| 指标（rad） | v5 | A | B |
|---|---|---|---|
| 同观测 rms | — | **0.0141（14.1 mrad）** | ≈0.0142 |
| 轨迹级 rms | 0.0608 | **0.0524** | 0.0554 |
| 轨迹级 p99 | 0.2239 | **0.1878** | 0.2108 |
| 轨迹级全局峰值 | 2.0265 | 2.0928 | **2.0641** |
| t=14 峰值 | 2.0265 | **1.1801（−42%）** | 1.2324 |
| t=5 峰值 | 1.9606 | 2.0928 | 2.0641 |
| 限位违反（base 4.867%） | 4.883% | 4.939% | 4.989% |
| **Δ 真正修掉**（越限→不越限） | 0.431% | 0.375% | 0.370% |
| **Δ 真正新增**（不越限→越限） | 0.447% | 0.447% | 0.492% |
| 净变化 | +0.016pp | +0.072pp | +0.122pp |
| （参考）"恶化"含原本就违规的 | 2.678% | 2.839% | 2.796% |
| （参考）"改善"含仍违规的 | 2.636% | 2.475% | 2.563% |


> ⚠️ **2026-11-08 二次更正**：上表此前写的"新增/修掉 2.8% / 2.5%"混了两种口径。
> 严格分解（`mod = base − fixed + new`，恒等式残差 ~1e-9）后，**真正新增只有 0.45%、真正修掉只有 0.37%**；
> 那 2.8%/2.5% 里的绝大多数是"**本来就违规、Δ 让它更差/稍好**"，不改变是否违规。
> 所以 Δ 对限位的影响是 **0.45% vs 0.37% 的微小净负**，不是"2.8% 的新增越限"。

**读法（三条，都是本轮新结论）**：
1. **Δ 在真实执行时的单步改动只有 14 mrad rms / 86 mrad 峰值** ⇒ 物理上很温和；
2. 轨迹级"峰值 2.09 rad"是**两条策略本来就会走得不一样**，**不是违规**（同观测只有 86 mrad）；
3. **Δ 对限位是净微负**：新增 2.8% > 修掉 2.5%（修默认角偏置前曾误判为"净正"，见 §3.5）；
   base 自己就有 4.87% 越限，Δ 修不完是结构决定的（64 维 token 残差 → 29 维动作）。

**闭环**（A，`runs/three_track_v6/isaac/A_delta_continue/`，SONIC=1、3 任务 × 6 seeds、MAX_STEPS=3000）

| 任务 | 成功 | 成功步数 |
|---|---|---|
| OpenDoor | 3/6 | 2625 / 804 / 1386 |
| SitSofa | 4/6 | 2181 / 2628 / 2013 / 958 |
| VisionNavi | 2/6 | 2846 / 1436 |
| **合计** | **9/18** | 耗时 2 h 27 min |

- **没有 base 臂**（不载 Δ 的对照）；sparse reward；seed0 曾用满 3 次尝试 ⇒ **方差极大**。
- B 的闭环：**NOT_RUN**。

---

## 2. 架构

### 2.1 链路

```
Isaac(3 任务) ─相机/本体─► 冻结 VLA(FT8000) ─64 维 token chunk(T=40)─►
  ─► Δ 网(1.744 M，逐帧) ─加回 token─► SONIC decoder(ONNX, 冻死) ─29 维归一化动作─►
  ─► q_raw = q_default(HA_ALIGN) + G1_ACTION_SCALE ⊙ a ─► 限位/速率安全层 ─► Isaac PD
```
50 Hz 控制；VLA 每 20 步重规划；**只在 t<K=20 加 Δ**，`t≥20` 严格 0。

### 2.2 Δ 网（`diff_vla_wbc/delta_token_temporal_v5.py`，**不改**）

| 输入 | 形状 | 说明 |
|---|---|---|
| `Z` | (B,40,64) | VLA 整段 token |
| `state64` | (B,64) | 决策时刻 `[rot6d(6) \| q(29) \| qd(29)]`，HA_V31 |
| `hist` | (B,10,58) | 过去 10 tick 的 `[q(29) \| qd(29)]`，HA，最老→最新 |

结构：`token_proj(64→128)` + `pe_chunk(40,128)` + 1 层 TransformerEncoder(d=128,h=4,ffn=256)
+ `hist_gru(58→128)` + `pe_pos(40,16)` + MLP 头 `400→1024→1024→64`（占 87% 参数）。
输出 `Δ = cap·tanh(‖u‖/cap)·u/‖u‖`，`cap=0.08`。
注册参数 **1,744,832**；末层 `N(0,4e-4)`+bias=0 ⇒ 第 0 步 ≈ Base。

### 2.3 训练目标（`scripts/train_delta_physical_bptt_v6.py`，**本轮已改**）

```
J_i  = Σ_{t<H} γ^t · survival_t · r_t         γ=0.99, H=20, r=WM 跟踪奖励
loss = -mean J
     + λ_act  · Trust(e)              λ_act=0.010, τ=0.020 rad   ← **域已改：physical**
     + λ_sm   · Smooth(e)             λ_sm =0.002                ← 同域
     + λ_anc  · Anchor(Δ)             λ_anc=0.001
     + λ_raw  · Hinge(‖u‖/(κ·cap))    λ_raw=1e-3, κ=2.0
     + λ_lim  · L_lim(q_raw)          λ_lim =0.0529  ← **需按新 q_raw 重标**
     + λ_peak · L_peak(e_phys, b_fix) λ_peak=8.4e-4  ← **b 已改固定预算**
     ( λ_same · L_same )              ← **physical 域下默认不启用（与 L_act 重复）**
     ( λ_rate · L_rate )              λ_rate=0，BLOCKED（无 velocity 字段）
```

### 2.4 部署契约（不动）

`state_hist_10` 形状/顺序（HA_V31）、`hist[-1]=当前状态`（tol 5e-3，fail-closed）、
`cap` 语义、`t≥K` 严格 0、`G1_ACTION_SCALE`、SONIC-only、server 用 cuda:0 / Isaac 用 GPU1、
一次只能起一个 Isaac。

---

## 3. 本轮修复（5 个 bug，全部已改 + 冒烟通过）

### 3.1 `L_act` 的单位错（P0）

**病**：`e = q_target_from_action(bundle.dp, bundle.action_scale, a)`，而 `bundle.action_scale = ones`
（`COMIMIC_ACTION_SCALE` 默认 `one`）⇒ `e` 是**归一化动作差**，不是 rad。τ=0.02 实际触发在
**RMS(e_norm) > 0.02 ⇒ 约 7.9 mrad**，而设计意图是 20 mrad。

**改**：新增 `--trust-domain {physical,normalized}`，默认 `physical` ⇒ `e_loss = E_phys`
（`= G1_ACTION_SCALE ⊙ (a_Δ − a_base)`），τ=0.02 才是真的 20 mrad。
`normalized` 保留用于复现旧结果。

### 3.2 `L_same` 与 `L_act` 重复（P0）

**病**：两项是**同一个量**（只差 per-joint 缩放），权重都是 0.010 ⇒ 信任域被罚两次。
**改**：`--trust-domain physical` 且 `--allow-double-trust 0`（默认）时 **`λ_same` 强制为 0**
（`train.jsonl` 里记录 `lambda_same_eff`）；`L_same_phys` 仍作为**诊断量**记录。

> 更正上一版的说法：上一版写的"`L_same` 结构上永远打不着"**不成立**——
> 那只是**标定用的 48 个窗口 + v5 ckpt** 上的数字（13.05 mrad）。实际训练 batch 上 A/B 的
> 同观测 rms 是 **14–41 mrad**（取决于口径），两项**都会开**。真实问题是"重复 + 单位不一致"。

### 3.3 `L_peak` 的预算自指（P0）

**病**：`b_j = 3·mean_t|e_j|` ⇒ 对**整体缩放 k 倍不变**（`|ke|/(3·mean|ke|) = |e|/(3·mean|e|)`），
只罚"某关节/某时刻相对突出"，**不限制绝对 rad**。

**改**：新增 `--peak-budget-mode {base-rate,fixed,self}`，默认 `base-rate`：

```
budget_j = clamp( κ · p90( |a_base[t,j] − a_base[t−1,j]| · G1_ACTION_SCALE_j ), min=floor )
κ = 2.0(default), floor = 0.02 rad, n = 128 个 train-clip 样本
```
只读 **base 策略 + train clips**，与 actor 参数无关 ⇒ 真正外生。
产物 `out/peak_budget.json`（含 `budget_rad` 29 维 + 来源），可复用到标定。
`self` 保留用于复现旧结果；`fixed` 可读外部 json/npz。

**实测预算**（B 初始化、seed0）：mean **0.2216** rad、min 0.0261、max 0.5981。
⇒ 当前 A/B 的 `e_phys_absmax` 只有 **0.086 rad**，**`L_peak` 恒为 0**。
也就是说：旧口径下它一直在"塑造相对形状"，新口径下它变成**只兜底的物理护栏**。
**若想让它继续塑形，必须显式把 budget 调紧（如 50 mrad）——这是一个待决策项，不是 bug。**

### 3.4 物理映射用错了 `default_joint_pos`（P0，本轮新发现）

**病**：物理层的 `_dp` 用了 `default_joint_pos("standard")`（mjlab `KNEES_BENT`：
knee **0.669** / ankle −0.363 / elbow 0.600），而 `q_base_same` 是用
`env.bundle.default_joint_pos`（HA_ALIGN：knee **0.420** / ankle −0.230 / elbow **0.870**）算的。
两者在 **10 个关节**上差 0.15–0.27 rad ⇒

```
E_phys = G1_ACTION_SCALE ⊙ (e_norm − Δdp),   Δdp = bundle_dp − mjlab_dp
max |G1·Δdp| = **0.1184 rad**（左/右肘）、mean 0.0261、**RMS 0.0462 rad**（10 个关节 > 0.01）
```

⇒ 训练里的 `e_phys / q_raw / L_lim / L_peak / L_same` **全部被一个常数偏置污染**。
实测证据：修复前 `act_rms_rad = 41.2 mrad`，修复后 **14.1 mrad**，两者之差（√(41.2²−14.1²)≈38.7 mrad）与偏置 RMS 0.0462 rad 同量级 ✓。

**改**：`_dp = env.bundle.default_joint_pos`（带 29 维断言）。同步修了
`calibrate_physical_bptt_weights.py`（3 处）与 `diag_delta_phys_v5.py`。

> 这条也**改写了限位统计**：修复前 base 违反率 2.701%，修复后 **4.867%**；
> "Δ 净改善限位"的结论**反转为"净微负"**（见 §1.2）。

### 3.5 物理诊断算了却没落盘（P1）

**病**：`_phys_log`（`L_lim/L_peak/L_same/rms_same/q_raw_max/e_phys_absmax`）算完**没有并进 row**
⇒ `train.jsonl` 里从来没有这些字段，事后无法判断物理约束是否生效。

**改**：`row = dict(..., **_phys_log)`，并新增 5 个字段：
`trust_domain / lambda_same_eff / peak_budget_mean / act_rms_rad / trust_active_frac / L_act_eff`。

### 3.6 冒烟验证（`runs/three_track_v7/ab_fix_smoke3/`）

| 检查 | 结果 |
|---|---|
| 固定预算生成 | `mode=base-rate mean=0.2216rad min=0.0261 max=0.5981` ✅ |
| 域切换 | `domain=physical τ=0.02rad double_trust=0` ✅ |
| 物理量 | `act_rms_rad=14.1 mrad`、`trust_active_frac=0.00`、`L_peak=0.0`、`L_lim=1.47`、`q_raw_max=3.136 rad` |
| 端到端不变性 | 从 B best 起、lr=1e-9、2 步 ⇒ **test ΔJ=+0.02080**（B 原值 +0.02074）✅ |
| 旧口径兼容 | `--trust-domain normalized --peak-budget-mode self` 跑通，`λ_same_eff=0.010`、`actRMS=33.7(norm)` ✅ |

### 3.7 重标定冒烟（8 窗口，新定义 + 固定预算）

| 项 | 旧定义（有偏置） | **新定义（已修）** |
|---|---|---|
| `rms_same` | 13.05 mrad | **13.56 mrad**（与训练里的 14.1 一致 ✓） |
| `‖g_lim‖/‖g_J‖` | 9.45× | **3.44×** |
| `λ_lim` 目标 | 0.0529 | **0.1453**（≈2.7 倍） |
| `L_same` | 13.05 < τ=20 ⇒ 未激活 | **13.56 < 20 ⇒ 未激活**（与训练一致） |
| `L_peak` | 自指预算下活跃 | **固定预算 0.22 rad ⇒ 未激活** |
| `L_rate` | BLOCKED | BLOCKED |

⇒ **改对单位之后，`L_act`/`L_same`/`L_peak` 对 A/B 当前策略全都不激活；只有 `L_lim` 活跃，
且它的建议 λ 从 0.0529 涨到 0.1453。** 这正是 §7-Q1 必须由 reviewer 决策的地方。

---

## 4. 问题清单（按优先级；★=本轮新发现）

### 4.1 P0（不修则结论不成立）

| # | 问题 | 证据 | 状态 |
|---|---|---|---|
| **P0-1 ★** | **训练与物理层的 `default_joint_pos` 用了两套**（mjlab 0.669 vs HA_ALIGN 0.420），10 个关节带 0.15–0.27 rad 偏置 | 冒烟 41.2 → 14.1 mrad | **已修**（§3.4） |
| **P0-2 ★** | **`L_act` 单位错**：τ=0.02 作用在归一化域 ⇒ 实际 7.9 mrad（设计 20） | 代码 + 实测 | **已修**（§3.1） |
| **P0-3 ★** | **`L_same` 与 `L_act` 重复**（同一量、两个单位、权重都 0.010） | 代码 | **已修**（§3.2） |
| **P0-4 ★** | **`L_peak` 预算自指** ⇒ 对整体幅度免疫 | 解析 + 代码 | **已修**（§3.3） |
| **P0-5** | **B 的闭环 NOT_RUN，A 的闭环无 base 臂** | §1.2 | 待办 |
| **P0-6 ★** | **λ 需按新定义重标**：`λ_lim` 0.0529 → **0.1453**；`same/peak` 均未激活 | §3.7 | **已冒烟（8 窗口）**，正式标定待办 |
| **P0-7 ★** | **物理层诊断从未落盘** ⇒ 无法判断约束是否生效 | §3.5 | **已修** |

### 4.2 P1（影响结论强度）

| # | 问题 | 证据 |
|---|---|---|
| P1-1 | **单 seed**（A/B 各 1 个；B 的 seed1 只 10 步、seed2 从未创建） | `train.jsonl` |
| P1-2 | **A/B 不是单变量对照**：初始化/lr/warmup/步数四变量同变；且 **B 的 lr 恰等于 v5 原配方** | §1.1、配置对比 |
| P1-3 | **val 抖动（≈0.003）> A−B 差（0.0011–0.0019）** | val 曲线 |
| P1-4 | **`λ_peak` 其实不弱**：加权梯度 8.4e-4×72.05 = **0.0605 ≈ 30%×‖∇J‖**（`λ_lim` 为 50%） | 标定 json |
| P1-5 | **条件化只被验证到"不是常量"**（cosZ≈0.65，门禁 <0.9），没有 reward 对齐证据 | `cond_check.jsonl` |
| P1-6 | **`hist` 只用 1 层 GRU 压成一个向量广播**；B 在 it=25 时 `cos_hist=0.9978`（几乎不用历史） | cond |
| P1-7 | **头占 87% 参数**（1.526/1.745 M），条件化容量可能天然受限 | 结构 |

### 4.3 P2（工程/口径）

| # | 问题 |
|---|---|
| P2-1 | `L_rate` **BLOCKED**（`RobotSpec` 无 velocity 字段） |
| P2-2 | 闭环只有 sparse success，**没有逐 tick 落盘**（`q_raw/q_applied/q_real/限位/tilt/进度`） |
| P2-3 | 三个脚本对同一 ckpt 报三种动作预算（训练器 `eval_split` 用缓存 base / 诊断脚本现算）⇒ 差 ~30%，**跨脚本比较前必须对齐路径** |
| P2-4 | checkpoint **不含 optimizer/scheduler/RNG** ⇒ 不存在严格 resume |
| P2-5 | B 停在 960/1000，**没有 last ckpt** |

---

## 5. 相对上一版交底：撤回清单

| 上一版说法 | 现状 |
|---|---|
| "A 达到 oracle 上界 99.4%，只剩 0.00018 空间，必须松 cap" | ❌ **撤回**：那只是**单步梯度 + 满 cap 的一阶候选**，不是证明上界 |
| "A 把 t=14 尖峰 2.065 → 0.425 rad（−79%），全局 2.065 → 1.847（−11%）" | ❌ **撤回**：96 窗口实测 **2.027 → 1.180（−42%）**，全局 **2.027 → 2.093（略差）**，尖峰迁到 t=5/t=19 |
| "限位违反率 2.74% → 2.90%（略差）" | ❌ **撤回**：正确 default 下 base **4.867%**，A 4.939%、B 4.989%，**Δ 净微负** |
| "`λ_peak=8.4e-4` 形同虚设" | ❌ **撤回**：加权梯度是 ‖∇J‖ 的 30% |
| "`L_same` 因标定梯度为 0 而从未激活" | ❌ **撤回**：它是**与 `L_act` 重复**，且两项在实际 batch 上都会开 |
| "训练日志里的 mrad 是物理量" | ❌ **撤回**：是归一化域，×0.396 才是 rad |
| "B 只有 val，没有 test/物理" | ✅ 已补齐：test +0.02074、物理与 A 同级 |
| "B 没失败" | ✅ **成立**：val/test 都正、物理温和 |

---

## 6. 计划（闸门制，每步都有"不通过就停"）

| # | 事项 | 成本 | 通过标准 | 不通过则 |
|---|---|---|---|---|
| **1** | **重标 λ**（新定义/单位/`q_raw`）：`calibrate_physical_bptt_weights.py --peak-budget <smoke3/peak_budget.json>` | ~10 min | 产出新 `physical_weights_delta_v2.json`；记录加权梯度/cos/激活率 | 定义还有问题，回炉 |
| **2** | **A/B 各 100–200 步修复 pilot**（从现有 best 起，cap 0.08、WM0 不变） | 2 × 35 min | `ΔJ_test` 不掉 >0.002；`e_phys_absmax` ≤ 0.12 rad；`L_lim` 不升；`trust_active_frac`、`L_peak` 行为符合预期 | 修复有害 ⇒ 回滚 |
| **3** | **闭环 base 臂 + B 的 18 集**（协议照 A，加 `--delta-v5-ckpt` 置空作 base） | ~5 h（可串行两轮） | **B ≥ 9/18 且 base 明显更低** ⇒ Δ 有效 | Δ 无 ⇒ **长训无意义**，改走 WM/奖励路线 |
| **4** | **条件化换成 reward 对照**：`G_cond = J(Z+Δ) − J(Z+Δ_shuffled)`（同任务同阶段内置换） | 代码 ~40 min | test 上 `G_cond` 稳定 >0 | 条件化无效 ⇒ 动编码器 |
| **5** | **逐 tick 稠密落盘**（reviewer 那张字段表） | 代码 ~1 h | 能定位"第几 tick 开始偏" | — |
| **6** | 只有 3 通过，才做**长训**（2×2 × 3 seeds × 1000 步） | **≈48 h**（串行；B 19.1 s/step） | 多 seed 均值±方差 + 配对检验 | — |
| **7** | WM 刷新（V7） | 依赖真物理分片（当前 BLOCKED） | 门禁通过才换 | — |

**长训的三个前置条件**（缺一不可）：
1. 闸门 3 证明 Δ 在真机上有效；
2. WM 版本先定死（在 WM0 上长训完再换 WM1 = 全部重做）；
3. 评测协议先定死（val 选模型 / test 报结果 / ≥3 seeds / bootstrap CI）。

---

## 7. 需要 reviewer 决策的 5 个问题

**Q1** 修复后 `L_act`（τ=20 mrad）与 `L_peak`（budget 0.22 rad）对 A/B **都是不激活**的。
你要的"物理约束"目标是
(a) **兜底护栏**（保持现状：只在越界时起作用）？
(b) **塑形项**（把 τ 调到 8–10 mrad、budget 调到 50 mrad，让它们持续给梯度）？
(c) 干脆去掉，只留 `L_lim` + `L_raw`？
请给判据：**你希望 Δ 的同观测动作扰动上限是多少 mrad？**

**Q2** `L_lim` 现在有了正确的 `q_raw`。base 自身违反率 **4.87%**，Δ 只能改动其中一小部分。
你要 (a) 只罚 **Δ 新增**的越限（`relu(v_mod − v_base)`），(b) 继续罚**绝对**越限，还是 (c) 两者都要？
（当前代码是 (b)。）

**Q3** 长训前，**B 是否要补齐 960→1000**？它没有 optimizer 状态，只能 warm-start（优化器从 0）。
你要 (a) 就用 best@900，(b) warm-start 补 40 步再评，还是 (c) 用同样配方重跑 1000 步？

**Q4** 2×2 的**两个空格**（`随机初始化 + 5e-5/2e-5 + 300 步`、`v5 warm-start + 3e-4/1e-4 + 960 步`）
是否值得花 4×1.3–5 h？还是接受"A/B 不可归因"，把算力全押在闭环与 WM 上？

**Q5** 条件化判据 `G_cond` 的**置换策略**：同任务内置换、同阶段（t 相近）内置换、还是随机？
以及 `G_cond` 的门限（例如 test 上 >0.002 且 95% CI 不含 0）？

---

## 8. 复现命令

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
source scripts/harness03_env.sh
O=runs/three_track_v7/ab_diagnostics

# (a) 生成固定 peak 预算（也顺带产出 test 指标；updates=0 不走训练循环）
python3 scripts/train_delta_physical_bptt_v6.py --out $O/b_testonly \
  --on-the-fly 1 --eval-set runs/temporal_ac_bptt_v5/eval_set --updates 0 --batch 256 \
  --init-delta-checkpoint runs/three_track_v6/delta_retrain/seed0/checkpoint_actor_best.pt \
  --eval-every 0 --cond-check-every 0 --seed 0 --device cuda:0

# (b) 真物理审计（已修默认角偏置）
python3 scripts/diag_delta_phys_v5.py \
  --ckpt runs/three_track_v6/delta_continue/seed0/checkpoint_actor_best.pt \
  --eval-set runs/temporal_ac_bptt_v5/eval_set --split val --device cuda:0 \
  --limits runs/three_track_v6/physical/limits_standard29.npz \
  --out $O/A_phys_rad_v2.json

# (c) 重标 λ（用 (a) 的 peak_budget.json）
python3 scripts/calibrate_physical_bptt_weights.py --actor delta \
  --delta-checkpoint runs/three_track_v6/delta_retrain/seed0/checkpoint_actor_best.pt \
  --limits runs/three_track_v6/physical/limits_standard29.npz \
  --windows 48 --peak-budget $O/b_testonly/peak_budget.json \
  --out runs/three_track_v6/calibration/delta_v2.json

# (d) 修复 pilot（100 步，不换 cap、不换 WM）
python3 scripts/train_delta_physical_bptt_v6.py --out runs/three_track_v7/ab_fix_pilot/B_s0 \
  --on-the-fly 1 --eval-set runs/temporal_ac_bptt_v5/eval_set --updates 100 --batch 256 \
  --init-delta-checkpoint runs/three_track_v6/delta_retrain/seed0/checkpoint_actor_best.pt \
  --physical-weights runs/three_track_v6/calibration/physical_weights_delta_v2.json \
  --limits-npz runs/three_track_v6/physical/limits_standard29.npz \
  --trust-domain physical --peak-budget-mode fixed \
  --peak-budget $O/b_testonly/peak_budget.json \
  --head-lr 5e-5 --enc-lr 2e-5 --warmup-updates 20 \
  --eval-every 25 --cond-check-every 25 --seed 0 --device cuda:0
```

---

## 9. 文件清单

| 文件 | 作用 |
|---|---|
| `diff_vla_wbc/delta_token_temporal_v5.py` | Δ 网架构（**不改**） |
| `diff_vla_wbc/physical_bptt_losses.py` | `L_lim/L_same/L_peak/L_rate` 实现（**不改**，问题在调用侧） |
| `scripts/train_delta_physical_bptt_v6.py` | 训练器（**本轮改了 4 处**：域、重复项、固定预算、default_joint_pos、诊断落盘） |
| `scripts/calibrate_physical_bptt_weights.py` | λ 反解（**已修 default_joint_pos + 新增 `--peak-budget`**） |
| `scripts/diag_delta_phys_v5.py` | **新增**：真物理审计 + Base/Δ 越限拆分 |
| `scripts/train_sonic_lora16_ft8000_bptt.py` | C 线训练器（本文件不涉及） |
| `runs/three_track_v7/ab_diagnostics/{b_testonly, v5\|A\|B_phys_rad_v2}` | 本轮全部实测产物 |
| `runs/three_track_v7/ab_fix_smoke3/` | 修复后冒烟 run（含 `peak_budget.json`） |

## 10. 并发与版本（⚠️ 必读）

**`comimic-wt03` 在本轮期间有另一个 agent 会话在并发提交**（11:15–13:13 CST，4 个 commit），
HEAD 已从 `71df44c` 走到 **`7692bc1`（V8-Lite-R: Tracking Reward 长尾替换）**。

| 事项 | 状态 |
|---|---|
| 我本轮前半改动（`--trust-domain` / `--peak-budget-*` / `fit_peak_budget_base_rate` / 预算接线） | 已被对方 commit `7692bc1` **一并提交** |
| 后半改动（`default_joint_pos` 修复、`_phys_log` 落盘、`act_rms_rad`/`trust_active_frac`） | 目前是 **未提交** 的 working-tree 修改（3 个文件 `M`） |
| 对方的 `l_margin`（软限位，`physical_bptt_losses.py` +23 行） | **只接在 C 的 LoRA 训练器**，A/B 训练器未使用 |
| 对方的 `reward_tail_mode`（`isaac_joint.reward`） | **默认 `off` ⇒ 逐位不变**，本轮所有测量不受影响 |

**结论**：本文件的数字在 `7692bc1` + 上述未提交修改下测得，**reward 未变、A/B 链路未受并发改动影响**。
**但**：继续跑 pilot 前建议先 `git add -A && git commit`（或换独立 worktree），否则会和对方的 V8 提交互相踩。

---

## 附录 · 术语

| 记号 | 含义 |
|---|---|
| 归一化动作域 | `bundle.action_scale=ones` 时的单位；×`G1_ACTION_SCALE`（均值 0.396）才是 rad |
| 同观测 `e` | 同一份 obs93 下只换 token 解出的动作差 —— **部署时的真实扰动量** |
| 轨迹级 `e` | 两条想象轨迹（有/无 Δ）跑散后的动作差 —— **含行为差异，不是违规** |
| 单步梯度候选值 | `Δ*=cap·g/‖g‖` 的 ΔJ（+0.0298），**一阶候选，非上界** |
| `b_j` 固定预算 | 由 **base 策略 + train clips** 的逐 tick 变化率标定，与 actor 无关 |
