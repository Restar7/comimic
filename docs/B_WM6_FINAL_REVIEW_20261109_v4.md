# B-WM6 终版评审材料 v4（架构 / 指标 / 全部结果 / 问题 / 结论）

> 生成：2026-11-09 14:05 CST｜容器 2（B 线）｜分支 `experiment/b-wm6-h10-critic`
> worktree：`/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-b-wm6`
> **本文取代 v2/v3 与所有中间文档，自包含。**
> 真物理数字经历过 3 次修正，本文只保留**最终正确值**（见 §9 更正链）。
> git：`d367454` → `c62f1fd` → `b033b3a` → `13d4884` → `139acb9` → `17b0d09` → `bf91e6e`

---

## 0. 一页结论

### 0.1 成立的三件事

| # | 结论 | 证据 |
|---|---|---|
| 1 | **"用冻结 WM 当可微裁判训练 Δ 适配器" 有效** | 真 Isaac 每段 tracking 回报 **+0.69%（J10）/ +1.21%（J20）**，**6/6 任务为正**，胜率 75–79%，**85.4% 窗口胜过同范数随机 Δ** |
| 2 | **工程链路完全打通且可复现** | B0 **27/27 PASS**；β 调度单测 **6/6**；严格 resume **bitwise 0**；跨进程 A/A **恰 0.0** |
| 3 | **critic 在训练所在分布上是准的** | 想象域 `v_ev=+0.686`、corr **0.838**、bias +0.008；在线全程 0.57–0.62 过门限 |

### 0.2 没成立的四件事

| # | 结论 | 证据 |
|---|---|---|
| 4 | **WM6 相对 WM0 没有买到任何东西** | 当裁判训练：真物理 **+0.00450 vs +0.00434**，差 −0.00016，**CI [−0.00111, +0.00078] 含 0**；排序：**1/6 任务占优** |
| 5 | **WM 只能做粗方向，不能做逐窗排序** | 整体 corr **+0.41**，但 pp_box **+0.09**、open_door +0.22、vision_navi **−0.07**；幅度乐观 1.66× |
| 6 | **critic 跨分布不能给绝对值** | 真 Isaac `v_ev=−5.64`、`v_mean=−0.284` vs 真值 **+0.550**、bias −0.834（corr 0.579 保留） |
| 7 | **任务级（闭环）完全没测** | base 8/18（44%），B 臂只有 4 集且是旧 ckpt |

### 0.3 总判定

> **`PASS_WITH_CAVEATS`**
> 机制 PASS（每段 +0.69%，6/6 任务，胜随机 85%）
> critic 训练域 PASS / 跨分布 FAIL
> **WM6 refresh 无增益（这是本轮最重要的负结果）**
> 任务级 **INCONCLUSIVE**（未测）

---

## 1. 三个指标必须分清（这一节是理解全文的前提）

| 指标 | 定义 | 量级 | 能不能当验收 |
|---|---|---|---|
| **① WM 想象 J** | 冻结 WM6 前向 40 步，`J = Σ_{t<H} γ^t·surv_t·r_t` | 0.79–0.84 | ❌ 不能（乐观 1.66×） |
| **② 真 Isaac open-loop J** | 回放固定动作序列，用**真物理状态**算同一个 reward | 0.27–0.68（分任务） | ✅ 可以（就是"每段漂移"） |
| **③ 闭环任务成功率** | `Isaac RGB → FT8000 → ΔV5 → SONIC → Isaac`，每 20 tick 重规划 | 44%（base 8/18） | ✅ 唯一任务级指标（**未测**） |

⚠️ **③ 的奖励是二值的**（代码核实）：

```python
def compute_reward(env):
    reward[success] = 1.0     # compute_success_mask 为真才给 1，否则 0
```

6 个任务全是这个写法 ⇒ **失败回合的 `max_reward=0` 意味着"3000 步里成功条件一次都没成立"**，
不是"差一点"。这直接决定了 §8 的"能否放大"分析。

---

## 2. 架构

### 2.1 两条链

```
部署（闭环）:
  Isaac RGB + 指令 + 本体状态
    → FT8000 (冻结, 无 LoRA) → Z_base [B,40,64]
    → TemporalDeltaActorV5 (1,744,832)
    → SONIC Release (冻结, ONNX)
    → Isaac 真物理
    → 每 20 tick 重规划；hist 逐 tick 更新
  ⚠️ 部署链**不含** WM6 / Critic

训练（可微）:
  同一 Z_base → Z_base + Δ
    → SONIC (冻结) → WM6 (冻结, 8.82M) → 40 步想象
    → J10 + β·γ^10·surv_10·V(y10) → 反传 → Δ
```

### 2.2 Actor（`diff_vla_wbc/delta_token_temporal_v5.py`，**未改动**）

```
输入  Z (B,40,64) | state64 (B,64) | hist10 (B,10,58)

hz   = token_proj(Z) + PE_chunk[:40]                        # 64→128
c_t  = LayerNorm(TransformerEncoder(1 block, d128, 4 head, FFN256, pre-LN))(hz)
h    = GRU(58→128)(hist10).last
x_t  = [ z_t(64) | c_t(128) | h(128) | state64(64) | PE_pos(t)(16) ]     # d_in = 400
u_t  = MLP(400→1024→1024→64, SiLU)                          # 末层 N(0,4e-4), bias=0
Δ_t  = apply_norm_cap(u_t, cap=0.08) · 1[t < delta_k]
```

参数分解：`8,320 + 5,120 + 132,480 + 256 + 72,192 + 640 + 1,525,824 = 1,744,832`
初始化：**从零**，末层 `N(0,4e-4)`（刻意非零，否则上游首步无梯度）；实测 init `|Δ|max=1.85e-3`。

### 2.3 Critic（`ImaginedValueV5`）

```
输入  s71 | ref71 | (s−ref)71 | xi33 | hist10(580) | plan4×64(256) | extra4
in_dim = 1086，hidden 512 × 2，参数 819,713
目标  TD(λ=0.95, γ=0.99) + 末端自举 V̄(y_H)；target EMA 0.995
```

### 2.4 张量合同（B0 逐位验证）

| 接口 | 定义 | 验证 |
|---|---|---|
| WM `state71` | `[root_pos3\|quat_wxyz4\|q_STD29\|v_w3\|w_w3\|qd_STD29]`，dt=0.02 | 与锁一致 |
| WM `action29` | **HA(SONIC) 序**，`(q_target_HA − dq_HA)/sc_HA` | 记录数据逐位复现 **4.2e-6**（640 tick） |
| Δ `state64` | `[rot6d6\|q_STD29\|qd_STD29]`，训练 standard；入站 ha_v31 由 spec 转 | B0 #15 |
| Δ `hist10` | `[q − **g1_default**(29) \| qd(29)]`，standard 序，最老→最新 | B0 #15 逐位 0 |
| SONIC | torch vs `onnx.reference.ReferenceEvaluator` | **4.8e-7** |
| 物理尺度 | `q_target_std = dq_std + sc_std·a_std`，`sc_std` 均值 **0.378852** | 记录器反解 |

⚠️ `hist` 的偏移是 `g1_default_joint_pos()`（hip −0.312 / knee 0.669 / elbow 0.6），
**不是** `bundle.default_joint_pos`（SONIC 口径 −0.2 / 0.42 / 0.87）；elbow 差 **0.27 rad**。
⚠️ `bundle.action_scale` 是**全 1.0**（`action_scale_mode=one`），**不能当 rad 尺度**（会错 2.64×）。

### 2.5 版本锁（B0 重算，与 C 容器一致）

| 对象 | sha256 前缀 |
|---|---|
| WM6 `wm_base6_best_candidate/wm.pt`（s350） | `8edb59cf` |
| FT8000 `checkpoint-8000/config.json` | `7d06d0da` |
| SONIC Release `model_decoder.onnx` | `c7241a12` |
| WBC `WBC_A24000_wmvel_cmd.pt` | `2390dc29` |
| packed motion `ha_cmd/packed.pt` | `3f444d50` |
| eval_set `windows.pt`（256/96/96） | `e2f8586e` |
| **WM0（旧版，用于消融）** | `fc5aeb30` |

资源：GPU0 = 冻结 FT8000（峰值 27.3 GB）；GPU1 = Δ+C+SONIC+WM6 BPTT（~2.4 GB）；
**12.7–14.6 s/update（B=256, H=10）**。GPU UUID 隔离，与 C 容器无重叠。

---

## 3. 全部参数

| 项 | 值 |
|---|---|
| T / K_exec / H_train / delta_k | **40 / 20 / 10 / 10**（`tail_control=mask10`） |
| dt / 控制频率 / 重规划 | 0.02 s / 50 Hz / 每 20 tick |
| γ | 0.99 |
| β_max / warmup / 限速 / 滞回 | 1.0 / **从 `critic_activation_update` 起算 40 步** / **0.05 per update** / on 0.35、off 0.28 |
| 梯度预算 | ‖g_V_scaled‖ ≤ **0.20 × ‖g_J10‖**（实测只用 3–4%） |
| Actor 优化器 | AdamW，head 3e-4 / enc 1e-4，min 3e-5 / 1e-5，wd 1e-4，clip 1.0，50 步 warmup + cosine |
| cap | 0.08（**固定**，`--cap-warmup-updates 0`） |
| λ_raw / κ | **1e-3 / 2.0** |
| λ_action / τ | 0.010 / **0.020（归一化单位 ≈ 7.6 mrad 物理 RMS）** |
| λ_smooth / λ_anchor / λ_limit / λ_peak | 0.002 / 0.001 / **0** / **0** |
| batch / seed | 256 / 0 |
| 起点采样 | `multinomial` over **237,880** 合格帧（漂移 ≤0.10 m、f≥10、仅 train clips），burn-in 10 步 |
| Critic 更新 | 4 次/actor step，batch 2048，lr 3e-4，wd 1e-4，clip 1.0，验证每 5–25 步 |

### 3.1 各 run 实际差异

| run | updates | 裁判 | critic | β | test ΔJ（**WM 自评**） |
|---|---|---|---|---|---|
| `c1` | 100 | WM6 | token scratch | 0→1 | +0.00376 |
| `c2b` | 300 | WM6 | token scratch | 门控后 0 | **+0.00785** |
| `c3_v5` | 300→420 | WM6 | v5 warm | **1（突跳）** | +0.00588 |
| `expA_beta0` | 300→340 | **WM6** | 关 | 0 | +0.00767 |
| `expB_fixed` | 300→340 | WM6 | v5 warm + 预算 | 0→0.975 | **+0.00788** |
| `expB2_histfix` | 300→340 | WM6 | v5 + hist 逐 t | 0→0.975 | +0.00793 |
| `expC_wm0` | 300→340 | **WM0** | 关 | 0 | +0.00756（**WM0 自评，不可与上比**） |

⚠️ 这一列的"test ΔJ"是**各自裁判的想象值**，跨行不可比。唯一可比的判据是 §6 的真 Isaac。

---

## 4. 已定位并修掉的 bug（12 个）

### 4.1 v5 训练器遗留（5）

| # | 原写法 | 后果 | 修法 |
|---|---|---|---|
| 1 | `states[H].detach()` 进 critic | `β·γ^H·V(s_H)` 对 Δ **梯度恒 0** | 不 detach；`FrozenParamCtx` 只关参数 |
| 2 | `refs_f[:, H]` | `IndexError`；critic 要 71 维参考状态 | `teacher_references(...,H+1)[H]` |
| 3 | `adapter.as_dict()`（token 分支 `adapter=None`） | `it%50==0` 崩溃 | token 分支不引用 |
| 4 | `beta` 只在 `it%50==0` 更新 | 前 49 步恒 0、之后一台阶 | 逐 update |
| 5 | checkpoint 缺 critic/target/opt/RNG/β | **无法 resume** | 全量保存 + sha 对账（B0 #25 bitwise） |

### 4.2 本轮新发现（4）

| # | bug | 证据 | 修法 |
|---|---|---|---|
| 6 | **β warmup 用全局 update** | 从 c2b it300 续训 + 换 critic ⇒ `min(1,301/100)=1`，β **一步 0→1** | `BetaController`：激活起算 + slew + 滞回（§5） |
| 7 | **resume 顺序**：`beta/gate_status` 无条件覆盖 | 覆盖在"架构重置"之后，冲掉重置 | 状态改由控制器持有 |
| 8 | **真物理两臂动作逐位相同** | `actions_of()` 忘传 `tokens` | 第一版 "+0.0014 显著" 是假信号 |
| 9 | **同进程顺序跑两臂互相污染** | 同动作末状态差 **6.84**；A/A 偏置 **+0.00143** | 每臂独立进程；实测跨进程 A/A **= 0.0** |

### 4.3 评测配对 bug（本轮最重要，3 个）

| # | bug | 影响 | 修法 |
|---|---|---|---|
| 10 | **Δ 臂用了 base 臂轨迹**（`states_plus` 是 plus/base 臂） | 所有真物理数字变成"base 状态 + Δ 动作"的混合量 | 一律用 `states_minus`；base 也用 minus（同位置 ⇒ 偏移抵消） |
| 11 | **critic 真物理校验视界混用** | 训练含"自举到 ∞"，校验用 20 步截断 MC ⇒ v_ev 被放大到 −4.53 | 统一视界（K20 截断），并放弃"改相对形式"的错误建议 |
| 12 | **hist10 在真实 20 步上陈旧** | 训练侧所有 t 复用 chunk 起点 | `hist10_at()` 逐 t 重建（对称/部署已对齐） |

### 4.4 环境（4）

| 症状 | 根因 | 修法 |
|---|---|---|
| Isaac 卡死在 `Starting the simulation` | `VK_ICD_FILENAMES` 指向不存在的 `/etc/vulkan/icd.d/nvidia_icd.json` | 改指 `/usr/share/vulkan/icd.d/nvidia_icd.json` |
| `[SonicActionProvider] onnxruntime is not available` | `PYTHONPATH` 含坏的 `third_party/gr00t_deps` onnxruntime | `env -u PYTHONPATH -u VIRTUAL_ENV` |
| 找不到 `result.json` | `cd $PROJECT_ROOT` 后相对 `OUT_ROOT` 被重解析 | `OUT` 绝对路径 |
| `wait` 死锁 | 无参 `wait` 等到了 VLA server | 只 `wait` episode PID |

---

## 5. β 调度修复（核心工程贡献）

```
β_sched = β_max · clamp((it − critic_activation_update) / W, 0, 1)
gate    : v_ev ≥ 0.35 ⇒ ok；v_ev < 0.28 ⇒ 不 ok；中间**保持**（滞回，不抖）
β_eff   : 每 update 最多变 0.05（限速）
```

`critic_activation_update` 在「新建 critic / `--critic-init ckpt` / resume 时架构不一致」时重置为当前 it，
**不动 Actor 全局 step**；状态 `{activation, gate_ok, β_eff, β_sched}` 写进 checkpoint。

**单测 6/6**（`audit/audit_beta_resume.json`）：

| # | 检查 | 结果 |
|---|---|---|
| 01 | it=300 换新 critic ⇒ β 起始 ≈0 | **0.0**（旧实现同点 = 1.0） |
| 02 | 之后 10 步不跳到 1 | 0, 0.01, …, 0.10 |
| 03 | 中断再 resume 曲线逐位复现 | **max diff = 0.0**（51 步） |
| 04 | 门禁抖动无突跳 | 每步 ≤ slew；死区保持 PASS |
| 05 | 不隐式改 LR/Scheduler | 源码不含 `.lr`/`optimizer`/`scheduler` |
| 06 | 旧实现门禁一翻满功率 | legacy@301=**1.0** vs 新首步 **0.0** |

**效果**：`expB`（β→0.975）test ΔJ **+0.00788**、动作 6.56 mrad，
与 `expA`（β=0）的 **+0.00767 / 6.65 mrad** 打平 ⇒ **非零价值梯度不再伤 Actor**。
对照 `c3_v5`（β 突跳）：**+0.00588 / 11.58 mrad**。

---

## 6. 全部实验结果

### 6.1 B0 门禁：**27/27 PASS**

`04_action_domain` 4.17e-6 / `09_sonic_onnx` 4.8e-7 / `10_init` 1,744,832 / `11_tail_zero` 恰 0 /
`13_cap` / `15/16_hist` 逐位 / `17/19/20` 三路梯度非零 / `21/22/23` TDλ+EMA+隔离 /
`25_resume` bitwise 0 / `27_full_step` rc=0。

### 6.2 梯度通路审计（P1）：**无 plan 捷径**

| 通路 | ‖g‖ | 占全通路 |
|---|---|---|
| A：只 `s10` 回传 | **0.7462** | 1.000 |
| B：只 plan 特征回传 | **0.0000** | 0.000 |
| C：都开 | 0.7462 | 1.000 |

`cos(A,C)=1.000` ⇒ 唯一通路是 `s10`（`z = tokens + Δ.detach()` 生效）。
⚠️ 口径陷阱：critic 的 `K` 是 **`exec_k=20`**，不是 `spec.K=10`（Δ 生效帧数）。

### 6.3 价值目标审计（P2）：**无重复计数**

TD(λ) 手算 vs 实现 **max diff 0.0**；`J10` 重算 **0.0**；MC 从 t 起算 ✅。
`J_AC = J10 + βγ^H·surv·V(y_H)`，J10 只覆盖 t<H，V 只覆盖 t≥H。

### 6.4 真物理 —— 每段漂移（**核心正结果**）

**协议**：48 窗（6 任务 × 8）/ 28 clips，`job_strat_*`，base 与 Δ 都用 `states_minus`（偏移抵消）。

| 任务 | base J10（Isaac） | **ΔJ10** | **相对** | **ΔJ20** |
|---|---|---|---|---|
| double_desk | 0.6824 | **+0.00288** | +0.42% | +0.00413 |
| pp_box | 0.2713 | **+0.00422** | +1.55% | +0.00897 |
| boxing | 0.6592 | **+0.00458** | +0.70% | +0.01168 |
| open_door | 0.6384 | **+0.00575** | +0.90% | +0.00918 |
| sit_sofa | 0.4440 | **+0.00737** | **+1.66%** | +0.01372 |
| vision_navi | 0.4582 | **+0.00546** | +1.19% | +0.01068 |
| **全体** | — | **+0.00504（win 0.792）** | **+0.89%** | **+0.00973（win 0.771）** |

**规律：base 越弱，Δ 提升越大**（pp_box 0.27 → +1.55%；sit_sofa 0.44 → +1.66%；
double_desk 0.68 → +0.42%）。这正是残差适配器该有的形态。

### 6.5 随机 Δ 对照（**排除"扰动本身有益"**）

| 臂（同范数 0.0684） | ΔJ10 | CI | 胜率 |
|---|---|---|---|
| 随机 Δ vs base | **−0.000373** | [−0.00099, +0.00034] | 0.333 |
| **训练 Δ vs 随机 Δ** | **+0.004185** | **[+0.00292, +0.00569]** | **0.854** |
| 训练 Δ vs 随机 Δ（J20） | **+0.006823** | [+0.00361, +0.01066] | 0.729 |

⇒ **不是"任何小扰动都有益"，训练确实在压低漂移。**

### 6.6 WM 与真实的对应关系

| 任务 | **WM6 corr(J20)** | WM0 corr(J20) | WM6 符号 / 恒正基线 | 幅度比 |
|---|---|---|---|---|
| double_desk | +0.587 | +0.663 | 0.50 / 0.62 | 4.1× |
| pp_box | **+0.091** | +0.163 | 0.75 / 0.75 | 1.2× |
| boxing | +0.676 | +0.807 | 0.62 / 0.75 | 0.9× |
| open_door | +0.215 | +0.451 | 0.75 / 0.75 | 1.9× |
| sit_sofa | **+0.832** | **−0.120** | 0.88 / 1.00 | 1.6× |
| vision_navi | **−0.067** | +0.279 | 0.62 / 0.75 | 1.8× |
| **全体** | **+0.413** | +0.300 | 0.688 / 0.771 | **1.66×** |

**判读**：整体正相关，但 **3/6 任务在 0.1 附近**；且 **WM6 只在 1/6 任务优于 WM0**。

### 6.7 WM6 vs WM0 当裁判（**决定性负结果**）

| 臂 | 真 ΔJ10 | CI | 胜率 | 真 ΔJ20 |
|---|---|---|---|---|
| expA（**WM6**） | **+0.004497** | [+0.00314, +0.00591] | 0.792 | +0.008832 |
| expC（**WM0**） | **+0.004341** | [+0.00270, +0.00606] | 0.688 | +0.008735 |
| **expC − expA** | **−0.000156** | **[−0.001107, +0.000782]** | 0.458 | −0.000098 |

**CI 含 0 ⇒ 不可分辨。** 分任务 4:2，无显著。
⚠️ 且起点 `c2b` 本身就是 WM6 训的（**对 WM6 有利**），仍未赢。

### 6.8 Critic 各版本

| critic | 目标 | 想象域 v_ev | 真物理 v_ev | 真物理 corr |
|---|---|---|---|---|
| TokenCritic | TDλ | **+0.067** | **−6.06** | — |
| ImaginedValueV5（H10+自举） | TDλ | **+0.626** | −0.843（同视界） | 0.536 |
| ImaginedValueV5（K20 截断） | MC(K20) | **+0.658**（dev）/ **+0.686**（vs 截断 MC） | **−5.64** | **0.579** |

真物理分解：`state_shift = −0.741` ≫ `horizon_shift = −0.035`；`v_mean −0.284` vs 真值 `+0.550`。
⇒ **critic 只认识 WM 状态分布**；改视界/hist 都救不了，唯一解是**用真 Isaac 数据训 critic**。

### 6.9 六任务闭环（base 臂，18 集；B 臂未测）

| 任务 | 成功率 | 成功集步数 | 失败原因 |
|---|---|---|---|
| Boxing | 2/3 | 330 / 1218 | timeout 3000 |
| DoubleDesk | **0/3** | — | timeout ×3 |
| OpenDoor | 2/3 | 1206 / 1313 | timeout 3000 |
| PPBox | 1/3 | 1246 | timeout ×2 |
| SitSofa | 2/3 | 1682 / 1626 | timeout 3000 |
| VisionNavi | 1/3 | 855 | timeout ×2 |
| **合计** | **8/18 = 44%** | 330–1682 | **100% timeout，0 摔倒** |

**所有失败 `max_reward = 0.0`**（二值奖励，成功条件一次未触发）。

### 6.10 其它审计

| 项 | 结果 |
|---|---|
| 物理动作（P5） | base decoder 自带越限 **7.58%** joint-step；Δ **新增仅 0.05%**；关节差 RMS 16.2 mrad（真 rad） |
| rot6（P9） | 训练 vs 部署 cos **0.9968**、变化 6.9%；置零 cos 0.9977 ⇒ 非主因 |
| 跨进程 A/A | **恰 0.0**（逐位相同） |
| 进程内换臂偏移 | **+0.00143**（已用"同位置对比"抵消，但仍非零） |

---

## 7. 当前问题清单（分级）

### 7.1 P0：决定能不能宣布成功

#### A. 闭环完全没测（B 臂）
- 现状：base 8/18；B 臂只有 4 集且是 β=0 的旧 ckpt。
- 影响：**任务级结论 = INCONCLUSIVE**；不能对外宣布任何成功率提升。
- 成本：6 任务 × 6 seeds × 2 臂 ≈ 3 h（PAR=2）。

#### B. WM6 refresh 没有可测增益
- 证据：排序 1/6 任务占优；训练收益差 −0.00016（CI 含 0）。
- 影响：**"升级 WM 让策略更好"这条不成立**（至少在本轮配置下）。
- 待办：若要继续投入 refresh，必须用 §6.7 的协议（40 步 + 真物理回放）先验收。

#### C. WM 不能做逐窗排序
- 证据：整体 corr +0.41，但 pp_box +0.09 / vision_navi −0.07；幅度乐观 1.66×。
- 影响：不能拿 WM 做候选筛选/数据过滤；只能当训练期可微裁判。

#### D. critic 跨分布不能给绝对值
- 证据：`v_mean −0.284` vs 真值 `+0.550`，bias −0.834（corr 0.579 保留）。
- 影响：`β·V` 只能在小 β + 梯度预算下用；不能当真物理评估器。
- 唯一解：用真 Isaac 数据训 critic（需采集）。

### 7.2 P1：重要但已有对策

| # | 问题 | 现状 |
|---|---|---|
| E | critic 在线解释度随策略漂移 | expB 40 步内 0.62→0.57，未触发门限；长训练需开 catch-up |
| F | 进程内换臂偏移 +0.00143 | 已用同位置对比抵消；彻底消除需 `--arm` 单臂进程 |
| G | hist10 真物理侧已逐 t，训练侧已修（expB2 验证等价） | 已修 |
| H | rot6 训练/部署参考系不同（cos 0.9968） | 非主因，跨容器统一延后 |
| I | base SONIC decoder 自带 7.58% 越限 | 部署侧安全层兜底；**不得改限位** |

### 7.3 P2：需记录

- `L_lim` 在 Δ≈0 时就 24.9 ⇒ 只能当监控（`--lambda-limit 0`）。
- 不用 `mask10` 时后半 chunk 的 Δ 与前半一样大（B1：0.0731 vs 0.0736）。
- `spec.K`（Δ 生效帧=10）≠ critic `K`（exec_k=20）。
- 单 seed、闭环 3 seeds/任务（统计功效 ±11.7pp）。
- `bundle.action_scale == 1.0` 陷阱（真 rad 尺度是 `sc_std`，均值 0.378852）。

### 7.4 已排除的怀疑（**不要重复投入**）

| 怀疑 | 排除证据 |
|---|---|
| critic 有 plan token 梯度捷径 | plan 通路梯度**恰 0.0**，cos(A,C)=1.000 |
| J_AC 重复计入前 H 步 | TDλ 手算差 0.0、J10 重算 0.0 |
| β warmup 无效 | 确认是 bug，已修，单测 6/6 |
| 真 Isaac 噪声 > 信号 | 跨进程 A/A **恰 0.0** |
| critic 目标视界错是主因 | 改 K20 截断后真物理 v_ev 更差（−4.53 → −5.64） |
| hist10 陈旧是主因 | 逐 t 重建后 actor 指标不变、critic v_ev 无改善 |
| `V(s_H) − V(ref_H)` 能去偏置 | **数学不成立**：减常数不改变 `∇θV(s_H)` |
| rot6 是主因 | cos 0.9968 |
| Δ 引入大量越限 | Δ 新增仅 0.05% |
| WM6 比 WM0 好 | 排序 1/6 占优；训练收益 CI 含 0 |

---

## 8. 关键分析：每段 +0.9%，能不能放大成 +10pp 成功率？

### 8.1 理论机制（支持放大）

1. **阈值效应**：成功是离散条件，若失败多为"擦边"，小幅改善可翻转。
2. **误差累积**：漂移若随步数累积，每段小改善会在长回合里放大。

### 8.2 但本批数据不支持

**① 失败不是擦边（代码 + 数据双证）**

```python
def compute_reward(env):
    reward[success] = 1.0        # 二值：成功条件成立那一步才 1
```

- 6 任务全是二值奖励 ⇒ `max_reward=0` 意味着 **3000 步里成功条件一次都没成立**。
- 成功集只需 **330–1682 步**（预算 3000，余量 45–89%）⇒ 成功有大余量。
- 失败 **100% 是 timeout，0 摔倒**。

**⇒ 分布是双峰（早成功 / 完全不成），不在阈值边缘。0.9% 的跟踪改善无法把"一次未触发"变成"触发一次"。**

**② 统计功效不够**

| 样本量 | 成功率标准误（p≈0.44） | 可分辨差异 |
|---|---|---|
| 18 集/臂 | **≈ 11.7 pp** | ≥ 23 pp |
| 36 集/臂 | ≈ 8.3 pp | ≥ 16 pp |
| 每任务 6 集 | ≈ 20 pp | 无意义 |

**⇒ 即便真提升 10pp，18 集也测不出来。**

**③ 历史反例**

`A_v8`：离线指标变好，**SitSofa 闭环 4/6 → 0/6**（全 timeout）。
⇒ **离线 ΔJ 变好不保证闭环不崩。**

### 8.3 结论与正确做法

| 判断 | 结果 |
|---|---|
| 理论上可能吗 | 可能 |
| 本批数据支持吗 | **不支持**（双峰、失败一次未触发、成功余量大） |
| 现在能测出来吗 | **不能**（±11.7pp） |
| 该看什么指标 | **成功集的 `episode_steps`**（连续、敏感）+ **`success_mask` 逐步命中率**（进度） |

**建议的闭环设计**：
1. 先改评测脚本，记录 **每步 `compute_success_mask`** + 目标距离（~10 行）。
2. 用 base 跑 18 集，看失败回合是"擦边"还是"完全没到"。
3. 若擦边为 0 ⇒ 放弃"成功率放大"假设，只看 `episode_steps`。
4. 若擦边多 ⇒ 再按 6 seeds/任务正式跑。

---

## 9. 更正链（旧结论全部作废）

| 版本 | 真物理 ΔJ10 | 作废原因 |
|---|---|---|
| 最初 | +0.0014 | 两臂动作逐位相同（bug 8） |
| 中途 | +0.00049（0.09%） | Δ 臂用了 base 臂轨迹（bug 10） |
| **现在** | **+0.003813（0.69%）/ 分层 +0.00504（0.89%）** | ✅ |

连带作废的数字：
- "WM 幅度乐观 **24×**" → **1.66×**
- "WM 逐窗 corr **+0.09**（无判别力）" → **+0.41**（但仍 3/6 任务无效）
- "真物理胜率 79.2%（0.09%）" → **75%（0.69%）**
- "critic 真物理 v_ev −4.53" → **−5.64**（口径修正后更差）

---

## 10. 给 GPT 的问题

| # | 问题 | 我的建议 |
|---|---|---|
| **Q1** | 每段 +0.9% 但失败是"一次未触发"，还值不值得跑闭环？ | **值得，但先补进度指标**（§8.3），否则 3 小时只能得到 ±11.7pp 的噪声 |
| **Q2** | WM6 refresh 已被证明无增益，是否停止 refresh 投入？ | **停**；或改用 §6.7 协议做准入验收 |
| **Q3** | critic 要不要用真 Isaac 数据重训？ | 只有 Q1 结果正面才值得 |
| **Q4** | 是否接受"WM 只做粗方向、不做排序"这个定位？ | 建议接受（corr 0.41 足以支撑梯度，不足以支撑筛选） |
| **Q5** | β 上限、catch-up、rot6 等次级项 | 按 §7.2 处理 |
| **Q6** | 版本号命名（B-WM6 / v7 / …） | 待指定 |

---

## 11. 下一步选项

| 选项 | 内容 | 成本 | 产出 |
|---|---|---|---|
| **A（推荐）** | 加进度指标 → base 18 集复跑 → 判断"擦边" | ~40 min | 决定闭环值不值得跑 |
| B | 直接跑 sit_sofa / navi / pp_box × 6 seeds | ~1.5 h | 3 任务的成功率 + steps |
| C | 全 6 任务 × 6 seeds | ~3 h | 唯一完整任务级结论 |
| D | 用真 Isaac 数据训 critic | 采集 + 2 h | critic 认识真状态 |
| E | 停，固化结论 | 0.5 h | 终稿 |

---

## 12. 复现

```bash
WT=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-b-wm6
cd $WT && source scripts/b_env.sh

python3 scripts/b_wm6/b0_tests.py --device cuda:1 --vla-device cuda:0 --batch 32
python3 scripts/b_wm6/test_beta_schedule.py          # 6/6
python3 scripts/b_wm6/baseline_audit.py              # 79 ckpt

# 训练（WM6 裁判）
python3 scripts/train_b_wm6_h10_critic.py --out .../expB_fixed \
  --resume .../c2b/seed0/checkpoint_last.pt --updates 340 \
  --device cuda:1 --vla-device cuda:0 --batch 256 --horizon 10 --exec-k 20 \
  --tail-control mask10 --lambda-raw 1e-3 --raw-norm-kappa 2 --cap-warmup-updates 0 \
  --beta 1.0 --beta-stage-warmup 40 --beta-slew 0.05 --beta-grad-budget 0.20 \
  --critic-enabled 1 --critic-kind v5 --critic-init ckpt --critic-ckpt .../critic_v5_b3.pt \
  --on-the-fly 1 --vla-batch 128 --seed 0 --eval-set runs/temporal_ac_bptt_v5/eval_set

# 训练（WM0 裁判，消融）
python3 scripts/train_b_wm6_h10_critic.py --out .../expC_wm0 --wm-ckpt <WM0.pt> \
  --resume .../c2b/seed0/checkpoint_last.pt --updates 340 --beta 0 --critic-enabled 0 ...

# 真物理（分层 48 窗）
python3 scripts/b_wm6/build_replay_job.py --ckpt ... --out .../job_strat_b_h20.pt \
  --split val --stratified-per-task 8 --horizon 20
bash scripts/b_wm6/run_isaac_roll.sh .../job_strat_b_h20.pt .../roll_strat_b_h20.pt 1
python3 scripts/b_wm6/per_task_stratified.py --device cuda:1
python3 scripts/b_wm6/rescore_p7_arms.py --device cuda:1
python3 scripts/b_wm6/wm0_vs_wm6_judge.py --device cuda:1
```

---

## 13. 产物索引

```
runs/dual_actor_wm6/b_delta/
├── config/{b_wm6_h10_critic.json, b_lock.json}
├── c0/ c1/ c2b/ c3_v5/                        历史 run
├── critic_warm_b1.pt  critic_v5_b3.pt          预训 critic
├── recovery_h10_critic/
│   ├── BASELINE_AUDIT.md                       79 ckpt 全表
│   ├── audit/{audit_beta_resume,baseline_audit}.json
│   ├── critic_grad/audit_critic_gradient_paths.json
│   ├── target_audit/{audit_critic_target,check2_critic_target,check2_critic_target_K20,check2_critic_target_K20arms}.json
│   ├── phys/{audit_physical_action,p7_direction_audit,pair_fix_h20,pair_random_h20,
│   │         pair_corrected_arms,per_task_stratified,wm0_vs_wm6_judge}.json
│   ├── rot6/audit_rot6.json
│   ├── critic_v5_K20.pt
│   ├── expA_beta0/ expB_fixed/ expB2_histfix/ expC_wm0/
│   └── final_report/REPORT.md
├── isaac/{job_*,roll_*,pair_*}.pt/json
└── isaac/final2/{tasks_base,tasks_b}/          闭环（base 18 集 / B 4 集）

comimic/docs/
├── B_WM6_FINAL_REVIEW_20261109_v4.md           ← 本文
├── B_WM6_INDEX.md                              索引
├── B_WM6_CORRECTION_20261109.md                更正详情（含 #1/#2 追加）
├── B_WM6_CHECK1_CHECK2_20261109.md             三检查详细数据
├── B_WM6_RECOVERY_REPORT_20261109.md           P0–P9
├── B_WM6_FINAL_REVIEW_20261109_v3.md           v3（已被本文取代）
└── B_WM6_FULL_REVIEW_20261109_v2.md            v2（作废）
```

---

## 14. 一句话给 GPT

> **"冻结 WM 当可微裁判"这条路是通的（6/6 任务、每段 +0.69~0.89%、85% 胜随机对照），
> 但 (a) WM6 相对 WM0 没有任何可测增益，(b) WM 只能做粗方向不能做排序，
> (c) critic 只认识 WM 分布、真物理只能排序（corr 0.58），(d) 闭环没测且二值奖励让"小幅改善→成功率"很难放大。
> 因此下一步不是继续调 WM/critic，而是**先给闭环加进度指标**，判断失败是"擦边"还是"从未接近"，再决定要不要投 3 小时跑正式闭环。**

---

## 15. 追加（15:45）：**与 C-V8 的同口径对拍**（回答了"两次不是同一口径"的质疑）

### 15.1 为什么之前不可比

C-V8 那次（`C_V8_ISAAC_H20_RESULT_20261108.md`）与我们的协议有 **5 处不同**：
窗口集（test 24 / 3 任务 vs val 48 / 6 任务）、动作生成（**开环固定历史** vs 滚动历史）、
H、Δ 来源（LoRA vs delta actor）、A/A 口径。

⇒ 之前"我们 corr 0.41 比他们 0.267 好"的说法**站不住**。

### 15.2 对齐做法

用 C 的原始 `job.pt` 里的 **24 个窗口索引**，按 C 的**开环解码**（`bundle.act(obs_flat, Z[:,t])`，
固定初始 10 帧历史）重建两个 job：

* `job_cv8ol_base.pt`：两臂都是 base（A/A）
* `job_cv8ol_delta.pt`：plus=Δ(expB)，minus=base

比较用**两臂的第一次臂**（跨进程，无进程内偏移）。评分同 `score_arm` + `bundle.reward`。

### 15.3 结果（同 24 窗 / 同 H20 / 同开环解码 / 同 reward）

| 任务 | n | **C-V8** | **我们（expB）** | 我们的逐窗胜率 |
|---|---|---|---|---|
| HSI_open_door | 8 | −0.00123 | **+0.00147** | 0.500 |
| **HSI_sit_sofa** | 8 | **−0.00726** | **+0.00429** | **0.875** |
| HSI_vision_navi | 8 | −0.01350 | **+0.00840** | **0.875** |
| **全体** | **24** | **−0.00733** | **+0.00472** | **0.750** |

我们的 95% CI（iid bootstrap 8000 次）：**ΔJ20 = +0.00472 [+0.00156, +0.00771]**；ΔJ10 = +0.00376 [+0.00179, +0.00574]。

### 15.4 判读

1. **同一批窗口、同一协议下，C-V8 三个任务全负，我们三个任务全正。**
2. 最关键的两个任务（正是 A_v8 / C-V8 崩掉的地方）：
   * **SitSofa**：C-V8 −0.00726 → 我们 **+0.00429（7/8 窗更好）**
   * **VisionNavi**：C-V8 −0.01350 → 我们 **+0.00840（7/8 窗更好）**
3. ⇒ "WM 与真实同号、非随机"这一条，**现在有了同口径证据**（此前只是跨口径推断）。
4. 但仍有 4 条限制：
   * 仍是**开环每段**，不是闭环任务成功率；
   * 只有 3 个任务（C 的窗口集不含 double_desk / pp_box / boxing）；
   * 24 窗、每任务 8 窗，open_door 胜率恰好 0.50（最弱）；
   * VLA token 采样与 Δ 来源不同（这正是要比较的变量，但 base 动作序列也因此不同）。

### 15.5 复现

```bash
python3 scripts/b_wm6/build_cv8_matched_job.py \
  --ckpt runs/dual_actor_wm6/b_delta/recovery_h10_critic/expB_fixed/checkpoint_last.deploy.pt
bash scripts/b_wm6/run_isaac_roll.sh .../isaac/job_cv8ol_base.pt  .../isaac/roll_cv8ol_base.pt  1
bash scripts/b_wm6/run_isaac_roll.sh .../isaac/job_cv8ol_delta.pt .../isaac/roll_cv8ol_delta.pt 1
# 评分脚本见 §15.3 的命令（score_arm，states_plus 跨进程配对）
```

---

## 16. 追加（22:10）：**六任务真实 VLA 闭环结果**（最终判据）

### 16.1 协议

| 项 | 值 |
|---|---|
| 模型 | `expF_seg_b/checkpoint_last.deploy.pt`，sha `40e2ad2b3f1fe343` |
| 训练 | **从零 scratch** → 100 步 → **每 100 步用当前策略重训 critic** → 300 步 |
| 闭环 | Isaac RGB → FT8000 在线 FM → ΔV5 → SONIC Release → Isaac，K=20 重规划 |
| 任务 / seeds | 6 任务 × **seeds 6/7/8**（与已有 base 18 集**同种子**） |
| 视频 | **18 集全部录制**（`videos_success/` / `videos_failure/`） |
| 并行 | 双卡（Isaac 在 GPU0+GPU1，VLA server 在 GPU1） |
| 输出 | `isaac/final3_d_video/` |

### 16.2 结果

| 任务 | base | **我们** | Δ | base 成功步数 | 我们成功步数 |
|---|---|---|---|---|---|
| PPBox | 1/3 | **0/3** | **−1** | 1246 | — |
| DoubleDesk | 0/3 | 0/3 | 0 | — | — |
| **Boxing** | 2/3 | **3/3** | **+1** | 330 / 1218 | **210 / 331 / 500** |
| OpenDoor | 2/3 | 2/3 | 0 | 1206 / 1313 | 1163 / 1302 |
| **SitSofa** | 2/3 | **3/3** | **+1** | 1626 / 1682 | 870 / 2615 / 2954 |
| VisionNavi | 1/3 | 1/3 | 0 | 855 | 2188 |
| **合计** | **8/18 (44.4%)** | **9/18 (50.0%)** | **+1** | — | — |

### 16.3 三条硬结论

1. **没有坍缩。** A_v8 在 SitSofa 是 **0/6**（base 4/6），我们 **3/3**（base 2/3）。
   这是本轮最重要的一条：**"离线变好、闭环全灭"的失败模式没有重演。**
2. **失败模式完全一致**：18 集里所有失败都是 `timeout 3000` 步、`max_reward=0`，
   **没有摔倒、没有异常动作**。
3. **成功率 +1 集（+5.6pp），统计上不显著**（n=18 时成功率标准误 ≈ 11.7pp）。

### 16.4 更有信息量的：完成步数变了

| 任务 | 方向 | 幅度 |
|---|---|---|
| **Boxing** | **快 2.2×**（774 → 347 步） | 大 |
| OpenDoor | 持平（1259 → 1232） | — |
| **SitSofa** | **慢 30%**（1654 → 2146），但多赢一集 | 中 |
| **VisionNavi** | **慢 2.6×**（855 → 2188） | 大 |
| PPBox | 唯一的成功丢了 | 负 |

**⇒ 每段 +0.9% 的跟踪收益，在闭环里表现为"**策略风格改变**"（有的任务更快、有的更慢），
而不是"成功率整体提升"。**

### 16.5 判定

| 维度 | 判定 |
|---|---|
| 坍缩风险（A_v8 的 0/6） | ✅ **排除**（SitSofa 3/3） |
| 失败模式安全性 | ✅ **无新故障** |
| 任务成功率 | **持平偏正**（+1 集，不显著） |
| 效率 | **任务依赖**（Boxing 快 2.2×，Navi 慢 2.6×） |
| 每段收益 → 任务收益 | ❌ **没有稳定转化** |

**总判定维持 `PASS_WITH_CAVEATS`：**
机制成立、无坍缩、成功率不显著提升、收益表现为任务相关的效率变化。

### 16.6 产物

```
isaac/final3_d_video/tasks_b/<Task>__seed<N>__repeat0__<ts>/
  ├── result.json            success / failure_reason / episode_steps
  └── videos_{success,failure}/*.mp4     ← 18 个视频
isaac/final3_d_video/SUMMARY.json
```
