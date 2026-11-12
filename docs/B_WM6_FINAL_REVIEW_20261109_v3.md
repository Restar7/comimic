# B-WM6 终稿评审材料 v3（问题 / 排查 / 结论 / 参数 全量）

> 生成：2026-11-09 12:50 CST｜容器 2（B 线）｜分支 `experiment/b-wm6-h10-critic`
> worktree：`/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-b-wm6`
> **本文取代 v2（`B_WM6_FULL_REVIEW_20261109_v2.md`）与 Check1/2 文档，自包含。**
> 所有数字可用 §11 复跑。git：`d367454` → `c62f1fd` → `b033b3a` → `13d4884`。


---

## ⚠️ 重要更正（2026-11-09 13:45，优先于本文所有正文）

本文写作时，真物理评测存在一个**配对臂 bug**（Δ 臂用了 base 臂的轨迹 `states_plus`），
导致 §0/§2/§6/§7 中**所有真物理数字作废**。已修正并重算，详见
**`B_WM6_CORRECTION_20261109.md`**。更正后的关键数字：

| 项目 | 本文旧值（作废） | **更正后（以此为准）** |
|---|---|---|
| 真物理 ΔJ10 | +0.00049（0.09%） | **+0.003813（0.69%）**，cluster CI [+0.00255, +0.00530] |
| 真物理 ΔJ20 | +0.00056 | **+0.006720（1.21%）** |
| 逐窗胜率 | 79.2% | **75.0%（J10）/ 66.7%（J20）** |
| 6 任务分层真实收益 | 未测 | **6/6 任务全为正**（+0.0041 ~ +0.0137，J20） |
| 训练 Δ vs **同范数随机 Δ** | 未测 | **+0.004185，85.4% 窗口胜出**（随机 Δ 本身 −0.00037） |
| WM6 vs 真实 corr | +0.093 | **+0.344（J20）/ +0.419（J10）** |
| WM 幅度乐观 | 24.1× | **1.66×（整体）/ 0.9×~4.1×（分任务）** |
| WM6 逐任务排序 | 未测 | **3/6 任务可用**（pp_box 0.09、open_door 0.22、vision_navi −0.07） |
| **WM6 vs WM0 当裁判** | 未测 | **真物理不可分辨**（+0.00450 vs +0.00434，CI 含 0） |

**受影响而作废的本文结论**：§1 的"WM6 逐窗方向判别 FAIL / 幅度校准 FAIL"、
§2.1 的问题 C/D 中的数字、§6.2 的真物理表、§6.5 的 WM 对拍表、§7 的 Check 1 结论。
**仍然成立**：B0 27/27、β 调度修复、梯度预算、critic 训练域正确、critic 跨分布失败、
"六任务闭环未做 ⇒ 任务级 INCONCLUSIVE"。

> 一句话更正：**WM 当裁判是有效的（每段 +0.69%，6/6 任务为正，85% 胜随机对照）；
> 但 WM6 相对 WM0 无论排序还是训练收益都没有可分辨优势。**

---

## 0. 一句话结论

> **工程链路、β 调度、梯度预算、critic 的训练期正确性：全部成立。**
> **真 Isaac 有统计显著的正收益：每段 +0.69%（J10）/+1.21%（J20），6/6 任务为正，胜率 75~79%，跨进程 A/A=0，且 85% 窗口胜过同范数随机 Δ。**
> **但三件事没成立：**
> **(a) WM6 只能做粗方向（整体 corr +0.41），6 个任务里只有 3 个可用，幅度乐观 1.66×；且 WM0 与之无差别（真物理 CI 含 0，排序 1/6 占优）；**
> **(b) critic 只认识 WM 的状态分布（想象域 v_ev 0.686，真物理 v_ev −5.68，v_mean −0.283 vs 真值 +0.545）；**
> **(c) 六任务闭环没跑 ⇒ 任务级结论 INCONCLUSIVE。**

**总判定：`PASS_WITH_CAVEATS`**（机制 PASS / 真物理短程 PASS（幅度 0.09%）/ critic 真物理绝对价值 FAIL / 任务级 INCONCLUSIVE）。

---

## 1. 各子系统状态总表

| 子系统 | 状态 | 判定 | 关键数字 |
|---|---|---|---|
| 架构合规（冻结栈 + H10 + K20 + Δ 只在 t<10） | 完成 | **PASS** | B0 27/27；部署日志 `tail_zero=True` |
| 契约层（state71/action29/state64/hist10/SONIC） | 完成 | **PASS** | 动作反解 4.2e-6；SONIC vs ONNX 4.8e-7；hist 逐位 0 |
| β 调度（warmup/门控/限速/resume） | 完成 | **PASS** | 单测 6/6；resume 曲线 max diff 0.0 |
| 价值梯度预算 | 完成 | **PASS** | 实测占 J10 的 3–4%（上限 20%） |
| H10+Critic 联合训练（非零 β 不伤 Actor） | 完成 | **PASS** | β=0.975；test ΔJ +0.00788（β=0 是 +0.00767）；动作 6.56 vs 6.65 mrad |
| Critic 训练域准确性 | 完成 | **PASS** | 想象域 v_ev **+0.686**、corr 0.838、bias +0.008 |
| Critic 跨分布（真物理） | 完成 | **FAIL** | v_ev **−5.68**、v_mean −0.283 vs +0.545、corr 0.577 |
| WM6 逐窗方向判别 | 完成 | **部分**（3/6 任务） | 整体 corr +0.41；pp_box 0.09 / open_door 0.22 / vision_navi −0.07 |
| WM6 幅度校准 | 完成 | **可用** | 乐观 **1.66×**（分任务 0.9×~4.1×） |
| 真 Isaac 短程收益 | 完成 | **PASS** | ΔJ10 **+0.003813**，cluster CI [+0.00255,+0.00530]，胜率 75%；6/6 任务为正；胜随机 +0.00419 |
| 真 Isaac 可重复性 | 完成 | **PASS** | 跨进程 A/A **恰 0.0** |
| 六任务 VLA 闭环 | **未做** | **INCONCLUSIVE** | Base 8/18；B 仅 4 集（旧 ckpt） |
| WM6 是否需要 refresh | 未决 | **暂不需要** | 见 §2.4 |

---

## 2. 当前问题清单（按严重度）

### 2.1 P0/阻塞：决定"能不能宣布成功"

#### 问题 A：六任务闭环未完成
| | |
|---|---|
| **现象** | 没有任何 expB 的闭环数据。Base 18 集（8/18=44%），B 臂只有 4 集且用的是 β=0 的 `c2b` |
| **证据** | `isaac/final2/SUMMARY.json`；`tasks_b/` 只有 pp_box 3 集 + double_desk 1 集 |
| **影响** | **无法回答"任务上有没有用"**，也无法判断 0.09% 的 tracking 改善是否有任务级意义 |
| **已做** | 闭环驱动 `run_isaac_pair.sh` 已修好 4 个 bug（Vulkan/PYTHONPATH/绝对路径/`wait` 死锁），Base 18 集已产出 |
| **还差** | expB ckpt × 6 任务 × 6 seeds（≈36 集，~3 h，PAR=2） |
| **判定** | **INCONCLUSIVE**（不是 FAIL，是没测） |

#### 问题 B：critic 只认识 WM 的状态分布
| | |
|---|---|
| **现象** | 想象域 v_ev **+0.686**（corr 0.838、bias +0.008）；真物理 v_ev **−5.68**（corr 0.577、bias **−0.828**、v_mean −0.283 vs 真值 +0.545） |
| **证据** | `target_audit/check2_critic_target_K20.json`；分解 `state_shift=−0.741` vs `horizon_shift=−0.035` |
| **根因** | critic 在 WM 状态流形上训练；真 Isaac 状态（接触/滑移/跟踪误差分布不同）是**分布外**，输出系统性偏低，而真值反而更高（0.545 > 0.450） |
| **已排除** | ~~视界不一致~~（改成 K20 截断后 v_ev 从 −4.53 → **−5.68**，没改善）；~~hist10 陈旧~~（逐 t 重建后同样没改善） |
| **影响** | critic **不能**作真物理绝对奖励；只能作**相对排序**（corr 0.577）。训练期使用不受影响（梯度在想象域算） |
| **唯一解法** | **用真 Isaac 数据训 critic**（需采集），改结构/口径无用 |

#### 问题 C：WM6 对 B 动作没有判别力
| | |
|---|---|
| **现象** | 同批 48 窗、同 ckpt、同 reward：WM6 sign agree **0.646**（J20）/**0.708**（J10），但**恒正基线**是 **0.688/0.792**；WM6 说"Δ 有害"的窗口里只对 **40%**（J10 30%）；corr **+0.093** |
| **证据** | `phys/p7_direction_audit.json` |
| **根因** | WM6 输出几乎恒为正（48 窗里 38 窗），sign agreement 只是与"真值 69% 为正"这个先验的重合 |
| **影响** | **WM 不能当比较器/筛选器**；但作为可微裁判给梯度仍有效（见问题 D） |
| **注意** | 这**不是"反号"**：平均方向两边一致（都正），只是幅度乐观 24× + 个体判别无效 |

#### 问题 D：为什么"裁判无效"却"训练有效"？（必须讲清楚，否则结论自相矛盾）
| | |
|---|---|
| **事实** | WM 逐窗排序无效（corr 0.09），但训出来的 Δ 在真物理上胜率 **79.2%**、CI 不含 0 |
| **解释** | 训练用的是**梯度**（`∂J10/∂Δ` 在 batch 上的平均方向），不是"选优"。只要梯度的**平均方向**指向 tracking 改善，训出来的 Δ 就有效；这不需要 WM 能逐窗排序 |
| **边界** | 但这条梯度很"浅"：真收益只有 0.09%，且 WM 说幅度是 0.95%（乐观 24×） |
| **风险** | 若继续加大 WM 优化强度（更大 β、更长训练），可能进入 WM 误差主导区；目前 300 步内未观察到 |

### 2.2 P1：重要但已有对策

#### 问题 E：critic 在线解释度随策略漂移下降
| | |
|---|---|
| **现象** | expB：v_ev 0.6206→0.5689（40 步）；c3_v5（旧 bug 版）0.62→0.25 |
| **现状** | 修复版 40 步内仍 > 0.55，未触发门限；但趋势存在 |
| **已备** | `--critic-catchup-every/--critic-catchup-steps`（actor 冻结追赶），**本轮未启用** |
| **建议** | 长训练（>200 步 @β>0.5）时启用：每 20 步、每次 100 步 |

#### 问题 F：`hist10` 在真实 20 步上仍偏陈旧
| | |
|---|---|
| **现象** | 训练器已改逐 t 重建（`hist10_at`）；但 H=10 想象轨迹漂移小，数值影响可忽略（expB2 与 expB 的 test ΔJ 差 0.00005） |
| **状态** | 代码已修、单测通过（末帧 == `obs(state_t)`） |
| **残留** | `score_pair` 的真物理侧逐 t 重建已实现；其他离线分析脚本若仍用旧口径需注意 |

#### 问题 G：`rot6` 训练/部署参考系不同
| | |
|---|---|
| **现象** | 训练 = 相对 clip 首帧参考；部署 = 相对回合首帧朝向 |
| **量化** | cos **0.9968**、相对变化 6.9%；整块置零 cos 0.9977 / 变化 6.0% |
| **判定** | 非主因，**不置零**（置零会掩盖问题）；彻底统一需改 C 侧 client（跨容器） |
| **风险** | 低 |

#### 问题 H：base SONIC decoder 自带越限
| | |
|---|---|
| **现象** | 7.58% 的 joint-step 里 `q_raw` 超出 MJCF 限位；Δ 只**新增** 0.05% |
| **现状** | 部署侧 `sonic_physical_action` 裁剪兜底；**WM6 看到的仍是未裁剪的归一化命令** |
| **风险** | 训练/部署之间有一层"裁剪"未被建模；但新增占比极小 |
| **禁止** | **不得**改真实机械限位来掩盖 |

### 2.3 P2：次要 / 需记录

| # | 问题 | 状态 |
|---|---|---|
| I | `L_lim` 在 Δ≈0 时就已 24.9（base 自带越限）⇒ 只能当监控（`--lambda-limit 0`） | 已处理 |
| J | 后半 chunk 若不用 `mask10`，Δ 会和无监督段一样大（B1：0.0731 vs 0.0736） | 已用 `mask10` 消除 |
| K | `hist` 的偏移是 `g1_default_joint_pos()`，与 `bundle.default_joint_pos` 差 **0.27 rad**（elbow） | 已加 `assert_hist_state_consistent`（tol 5e-3） |
| L | `spec.K`（Δ 生效帧数=10）≠ critic plan 的 `K`（exec_k=20） | 已在代码/文档标注；训练器用 20 正确 |
| M | 训练 seed 单一（全部 seed 0） | 未做多 seed 方差 |
| N | 六任务闭环每任务只 3 seeds | 统计功效不足 |
| O | `bundle.action_scale == 1.0`（全 1），不是 rad 尺度 | 已全部改用 `sc_std`（均值 0.378852） |

### 2.4 已**排除**的怀疑（不要重复投入）

| 怀疑 | 排除证据 |
|---|---|
| ~~critic 有 plan token 梯度捷径~~ | plan 通路梯度**恰为 0.0**（`z = tokens+Δ.detach()`），`cos(A,C)=1.000` |
| ~~J_AC 重复计入前 10 步奖励~~ | TD(λ) 手算差 **0.0**；J10 重算 **0.0**；MC 从 t 起算 |
| ~~β warmup 无效~~ | 确认是 bug，**已修**；单测 6/6 |
| ~~真 Isaac 噪声 > 信号~~ | 跨进程 A/A **恰 0.0**；旧 +0.00143 是进程内换臂污染 |
| ~~两臂动作相同导致假信号~~ | 已修（`actions_of` 忘传 tokens）；现 `max\|Δa\|=2.68` |
| ~~critic 目标视界错~~ | 改成 K20 截断后真物理 v_ev 反而更差（−4.53 → −5.68） |
| ~~hist10 陈旧是主因~~ | 逐 t 重建后 actor 指标不变、critic v_ev 无改善 |
| ~~`V(s_H) − V(ref_H)` 能修正梯度~~ | **数学上不成立**：减常数不改变 `∇θ V(s_H)`（我上一版的错误，已撤回） |
| ~~rot6 口径是主因~~ | cos 0.9968，变化 6.9% |
| ~~Δ 引入大量越限~~ | Δ 新增越限仅 0.05% |

---

## 3. 架构与合同

### 3.1 两条链

```
部署: Isaac RGB + 指令 + 本体状态 → FT8000(冻结,无LoRA) → Z_base[40,64]
      → ΔV5(1,744,832) → SONIC Release(冻结) → Isaac 真物理
      → 每 20 tick 重规划；history 逐 tick 更新；**不含 WM6/Critic**

训练: 同一 Z_base → Z_base+Δ → SONIC(冻结) → WM6(冻结,8.82M)
      → J10 + β·γ^10·surv_10·V(y10) → Δ
```

### 3.2 Actor（`diff_vla_wbc/delta_token_temporal_v5.py`，未改）

```
Z (B,40,64) | state64 (B,64) | hist10 (B,10,58)
hz   = token_proj(Z) + PE_chunk[:40]                  # 64→128
c_t  = LayerNorm(TransformerEncoder(1 block,d128,4head,FFN256,pre-LN))(hz)
h    = GRU(58→128)(hist10).last
x_t  = [ z_t(64) | c_t(128) | h(128) | state64(64) | PE_pos(t)(16) ]   # d_in=400
u_t  = MLP(400→1024→1024→64, SiLU)                    # 末层 N(0,4e-4), bias=0
Δ_t  = apply_norm_cap(u_t, cap=0.08) · 1[t < delta_k]
```
参数 `8320+5120+132480+256+72192+640+1525824 = 1,744,832`；从零初始化（末层 `N(0,4e-4)`，刻意非零）。

### 3.3 合同（B0 逐位验证）

| 接口 | 定义 | 验证 |
|---|---|---|
| WM `state71` | `[root_pos3\|quat4\|q_STD29\|v3\|w3\|qd_STD29]`，dt=0.02 | 与锁一致 |
| WM `action29` | **HA 序**，`(q_target_HA − dq_HA)/sc_HA` | 记录数据逐位复现 4.2e-6 |
| Δ `state64` | `[rot6d6\|q_STD29\|qd_STD29]` | B0 #15 |
| Δ `hist10` | `[q − g1_default(29) \| qd(29)]`，standard 序，最老→最新 | 逐位 0 |
| SONIC | torch vs ONNX ReferenceEvaluator | **4.8e-7** |
| 物理尺度 | `q_target_std = dq_std + sc_std·a_std`，`sc_std` 均值 **0.378852** | 记录器反解 |

### 3.4 Critic（`ImaginedValueV5`）

```
输入 s71 | ref71 | (s−ref)71 | xi33 | hist10(580) | plan4×64(256) | extra4   → in_dim 1086
网络 hidden 512 ×2，参数 819,713
目标 TD(λ=0.95, γ=0.99) + 末端自举 V̄(y_H)；target EMA 0.995
```

---

## 4. 全部参数

### 4.1 时间轴 / RL

| 项 | 值 |
|---|---|
| T / K_exec / H_train / delta_k | **40 / 20 / 10 / 10**（`tail_control=mask10`） |
| dt / control / 重规划 | 0.02 s / 50 Hz / 每 20 tick |
| γ | 0.99 |
| β_max / warmup / slew / 滞回 | 1.0 / 从 `critic_activation_update` 起算 40–100 步 / **0.05 per update** / on 0.35、off 0.28 |
| 梯度预算 | ‖g_V_scaled‖ ≤ **0.20 × ‖g_J10‖** |
| Critic γ/λ/EMA | 0.99 / 0.95 / 0.995 |
| Critic 更新 | 4 次/actor step，batch 2048，lr 3e-4，wd 1e-4，clip 1.0 |
| Critic 独立验证 | 每 5–25 步，独立 clip 的想象 MC `v_ev`，门限 0.30 |

### 4.2 Actor 优化 / 损失

| 项 | 值 |
|---|---|
| 优化器 | AdamW，head 3e-4 / enc 1e-4，min 3e-5 / 1e-5，wd 1e-4，clip 1.0 |
| warmup / decay | 50 步 linear + cosine |
| cap | 0.08（固定，`--cap-warmup-updates 0`） |
| λ_raw / κ | **1e-3 / 2.0** |
| λ_action / τ | 0.010 / **0.020（归一化单位 ≈ 7.6 mrad 物理 RMS）** |
| λ_smooth / λ_anchor / λ_limit / λ_peak | 0.002 / 0.001 / **0** / **0** |
| batch | 256，seed 0 |

### 4.3 采样 / 数据

| 项 | 值 |
|---|---|
| 起点 | `multinomial(w, B)` over **237,880** 合格帧（漂移 ≤0.10 m、f≥10、f+H+1≤clip_end、仅 train clips） |
| burn-in | 10 步；扰动 xy 0.05 m / yaw 0.10 rad / dq 0.05 rad / dqd 0.30 rad/s |
| VLA | 冻结 FT8000，`fm_steps=4`，每次 update 新噪声，vla_batch 128 |
| eval_set | 256/96/96（train/val/test），训练只用 train clips |

### 4.4 各 run 的实际差异

| run | updates | delta_k | β | critic | λ_raw | cap warmup | 备注 |
|---|---|---|---|---|---|---|---|
| `c1` | 100 | 20 | 0→1 | token scratch | 0 | 100 | B1 |
| `c2b` | 300 | 10 | 门控后 0 | token scratch | 1e-3 | 0 | **B3 正式，actor 最优** |
| `c3_v5` | 300→420 | 10 | **1 突跳** | v5 warm | 1e-3 | 0 | **失败样本** |
| `expA_beta0` | 300→340 | 10 | 0 | 关 | 1e-3 | 0 | P6 对照 |
| **`expB_fixed`** | 300→340 | 10 | **0→0.975** | v5 warm + 预算 | 1e-3 | 0 | **P6 修复候选** |
| `expB2_histfix` | 300→340 | 10 | 0→0.975 | 同上 + hist 逐 t | 1e-3 | 0 | 等价（+0.00793） |

---

## 5. 全部 bug 清单（12 个）

### 5.1 v5 训练器遗留（5 个）

| # | 原写法 | 后果 | 修法 |
|---|---|---|---|
| 1 | `states[H].detach()` 进 critic | `β·γ^H·V(s_H)` 对 Δ **梯度恒 0** | 不 detach；`FrozenParamCtx` 只关参数 |
| 2 | `refs_f[:, H]` | `IndexError`；critic 需要 71 维参考状态 | `teacher_references(...,H+1)[H]` + `ref_state_at()` |
| 3 | `adapter.as_dict()`（token 分支 `adapter=None`） | `it%50==0` 崩溃 | token 分支不引用 adapter |
| 4 | `beta` 只在 `it%50==0` 更新 | 前 49 步恒 0、之后一台阶 | 逐 update |
| 5 | checkpoint 缺 critic/target/opt/RNG/β | **无法 resume** | 全量保存 + sha 对账；B0 #25 验 bitwise |

### 5.2 本轮新发现（4 个）

| # | bug | 证据 | 后果 | 修法 |
|---|---|---|---|---|
| 6 | **β warmup 用全局 update** | `min(1,301/100)=1` ⇒ β 一步 0→1 | c3_v5 test ΔJ +0.00785→+0.00588、动作 6.7→11.6 mrad | `BetaController`：激活起算 + slew + 滞回 |
| 7 | **resume 顺序**：`beta/gate_status` 无条件覆盖 | 覆盖在"架构重置"之后，冲掉重置 | 同上 | 状态改由控制器持有 |
| 8 | **真物理两臂动作逐位相同** | `actions_of()` 忘传 `tokens` | 第一版 "+0.0014 显著" 是假信号 | 传参 + `--dup` A/A |
| 9 | **同进程顺序跑两臂互相污染** | 同动作末状态差 **6.84**；ΔJ10 偏置 +0.00143 | 曾误判 INCONCLUSIVE | **每臂独立进程**；实测 A/A **=0.0** |

### 5.3 环境（3 个）

| 症状 | 根因 | 修法 |
|---|---|---|
| Isaac 卡死在 `Starting the simulation` | `VK_ICD_FILENAMES` 指向不存在的 `/etc/vulkan/icd.d/nvidia_icd.json` ⇒ `vkCreateInstance ERROR_INCOMPATIBLE_DRIVER` ⇒ 无 CUDA GPU ⇒ PhysX 软件回退挂死 | 指向 `/usr/share/vulkan/icd.d/nvidia_icd.json` |
| `[SonicActionProvider] onnxruntime is not available` | `PYTHONPATH` 含坏的 `third_party/gr00t_deps` onnxruntime | `env -u PYTHONPATH -u VIRTUAL_ENV` |
| 找不到 `result.json` | `ha_isaac_closedloop.sh` `cd $PROJECT_ROOT` 后相对 `OUT_ROOT` 被重解析 | `OUT` 绝对路径 |
| （附加）`wait` 死锁 | 无参 `wait` 等同 shell 后台的 VLA server，而 `stop_server` 在其后 | 只 `wait` episode PID |

---

## 6. 全部实验结果

### 6.1 B0 门禁 27/27 PASS

`04_action_domain` 4.17e-6 / `09_sonic_onnx` 4.8e-7 / `10_init` 1,744,832 / `11_tail_zero` 恰 0 /
`13_cap` / `15/16_hist` 逐位 / `17/19/20` 三路梯度非零 / `21/22/23` TDλ+EMA+隔离 /
`25_resume` bitwise 0 / `27_full_step` rc=0。

### 6.2 Actor 各版本（held-out test，独立 clip 96 窗）

| run | β | test ΔJ | same-obs RMS | peak | 备注 |
|---|---|---|---|---|---|
| `c1` | 1.0 | +0.00376 | 13.98 mrad | 74.0 | 101 步 |
| `c2b` | 0 | **+0.00785** | 6.71 mrad | 43.5 | **300 步** |
| `c3_v5` | 1→0 | +0.00588 | 11.58 | — | 失败样本 |
| `expA` | 0 | +0.00767 | 6.65 | — | 对照 |
| **`expB`** | **0.975** | **+0.00788** | **6.56** | — | 修复候选 |
| `expB2` | 0.975 | +0.00793 | 6.57 | — | hist 逐 t，等价 |

逐任务 val ΔJ（c2b）：sit_sofa +0.01314 / pp_box +0.00991 / vision_navi +0.00988 /
open_door +0.00877 / boxing +0.00847 / double_desk +0.00697 —— **6/6 全正**。

### 6.3 Critic 各版本

| critic | 目标 | 想象域 v_ev | 真物理 v_ev | 真物理 corr |
|---|---|---|---|---|
| `critic_warm_b1.pt`（TokenCritic） | TDλ | **+0.067** | **−6.06** | — |
| `critic_v5_b3.pt` | H10+自举 | **+0.626** | −4.53（混口径）/ −0.843（H10 截断同口径） | 0.536 |
| **`critic_v5_K20.pt`** | **K20 截断** | **+0.658**（dev）/ **+0.686**（vs 截断 MC） | **−5.68** | **0.577** |

`critic_v5_K20` 的真物理分解：`state_shift −0.741` ≫ `horizon_shift −0.035`。

### 6.4 真物理（48 窗 × H=20，每臂独立进程）

| 指标 | Base | B | Δ | 95% CI | 胜率 |
|---|---|---|---|---|---|
| **J10** | 0.55514 | 0.55563 | **+0.00049** | iid [+1.08e-4, +1.06e-3] / **cluster [+1.09e-4, +1.05e-3]** | **79.2%** |
| **J20** | — | — | **+0.00056** | [+1.1e-4, +1.19e-3] | 68.8% |
| A/A（跨进程） | — | — | **0.0** | — | — |
| A/A（同进程顺序） | — | — | +0.00143 | — | — |

48 窗来自 **26 个 clip / 3 个任务**（boxing 20、double_desk 17、pp_box 11）。

### 6.5 WM vs Isaac 逐窗方向

| 预测器 | sign agree (J10) | sign agree (J20) | 恒正基线 (J10/J20) | corr (J20) | 幅度比 |
|---|---|---|---|---|---|
| WM0 | 0.667 | 0.604 | 0.792 / 0.688 | +0.023 | 24.3× |
| **WM6** | 0.708 | **0.646** | 0.792 / 0.688 | **+0.093** | **24.1×** |

WM6 混淆矩阵（J20）：`TP 27 / FP 11 / TN 4 / FN 6` ⇒ **说"有害"时准确率 40%**。

### 6.6 六任务闭环（未完成）

| 任务 | Base（seeds 6/7/8） | B（旧 c2b，被叫停） |
|---|---|---|
| PPBox | 1/3 | 0/3 |
| DoubleDesk | 0/3 | 0/1 |
| Boxing | 2/3 | — |
| OpenDoor | 2/3 | — |
| SitSofa | 2/3 | — |
| VisionNavi | 1/3 | — |
| **合计** | **8/18 = 44%** | **0/4** |

部署链确认真生效：`[server][delta-v5] #121 |Δ|mean(t<K)=0.0542 tail_zero=True`。

---

## 7. 三个专项检查（GPT 指定）

### Check 3：P7 测的哪个 ckpt
`job_fix_base_h20.pt` / `job_fix_b_h20.pt` 内嵌
`ckpt = recovery_h10_critic/expB_fixed/checkpoint_last.deploy.pt`、
`ckpt_sha256 = 7eb3d2bbbb2d…`、文件 sha `3bc77a94f6b9…`。
**⇒ P7 的 +0.00049/+0.00056 确实来自 β=0.975 的 H10+Critic Actor。**
（对照 c2b β=0：+0.00047 / 胜率 72.9%。）

### Check 1：WM 方向对拍
见 §6.5。**结论：平均方向一致（都正），逐窗判别无效（< 恒正基线），幅度乐观 24×。**
分任务：boxing sign 0.55、double_desk 0.82（**真 ΔJ≈0，无意义**）、pp_box 0.55。

### Check 2：critic 目标口径
三层差异：① 视界（训练含自举 vs 校验截断）② hist10（陈旧 vs 逐 t）③ plan/ref/γ 一致。
修完①（K20 截断）②（逐 t）后：真物理 v_ev 从 −4.53 → **−5.68**（更差），
`state_shift −0.741` ≫ `horizon_shift −0.035` ⇒ **主导因素是状态分布迁移**。

---

## 8. 已撤回/修正的结论（避免 GPT 被旧文档误导）

| 旧结论 | 修正 | 原因 |
|---|---|---|
| "真物理 ΔJ=+0.0014 显著" | **撤回**，是假信号 | 两臂动作逐位相同（bug 8） |
| "A/A 噪声 +0.00143 > 信号 ⇒ INCONCLUSIVE" | **撤回**，A/A **= 0.0** | 进程内换臂污染（bug 9） |
| "critic 真物理 v_ev = −4.53 ⇒ critic 坏了" | **修正**：−4.53 是口径混用；同口径 −0.843；视界对齐 −5.68 | Check 2 |
| "把 `β·V` 改成 `V(s_H)−V(ref_H)` 去偏置" | **撤回**：减常数不改变 `∇θV(s_H)` | 数学错误（GPT 指出） |
| "WM6 把收益放大 10×" | **修正为 24×**（同窗口同 ckpt 同 reward） | 之前用了不同口径 |
| "B4 的 B 臂 0/4 ⇒ B 线失败" | **撤回**：4 集、旧 ckpt、样本不足 | 统计上无效 |
| "hist10 陈旧是 critic 失败的主因" | **撤回**：逐 t 修复后无改善 | expB2 |

---

## 9. 给 GPT 的问题（更新版）

| # | 问题 | 现状 | 我的建议 |
|---|---|---|---|
| **Q1** | 六任务闭环要不要补（6 seeds × 6 任务） | 唯一缺口 | **补**，~3 h，是唯一能回答"任务上有没有用"的 |
| **Q2** | critic 要不要用真 Isaac 数据重训 | 现在只认识 WM 分布 | 若 Q1 结果正面再说；否则不值 |
| **Q3** | WM6 要不要 refresh | 判别力无效但训练梯度有效 | **暂不**；先补闭环，若闭环无提升再谈 |
| **Q4** | β 上限 | 1.0 已 OK（expB） | 保持；可做 0.5 消融 |
| **Q5** | critic 长训练时要不要开 catch-up | 40 步内未触发 | 长训练（>200 步）时开 |
| **Q6** | rot6 要不要跨容器统一 | cos 0.9968 | 非主因，可延后 |
| **Q7** | 版本号命名（B-WM6 / v7 / …） | 未定 | 请指定，我改标题与判定表 |
| **Q8** | 是否接受 `PASS_WITH_CAVEATS` 作为对外结论 | — | 建议接受；不要写成 PASS |

---

## 10. 下一步选项

| 选项 | 内容 | 成本 | 产出 |
|---|---|---|---|
| **A（推荐）** | 补六任务闭环：expB × 6 任务 × 6 seeds vs Base | ~3 h | 唯一能定 PASS/FAIL 的数据 |
| B | 用真 Isaac 数据训 critic | 采集 + ~2 h | critic 认识真状态；收益仍存疑 |
| C | 停在这里出终稿 | 0.5 h | `PASS_WITH_CAVEATS` + 任务级 INCONCLUSIVE 写死 |
| D | WM6.1 动作敏感度 refresh | 需采集 | 现有证据不支持（判别力问题可能是 WM 本身） |

---

## 11. 复现

```bash
WT=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-b-wm6
cd $WT && source scripts/b_env.sh

# 门禁与单测
python3 scripts/b_wm6/b0_tests.py --device cuda:1 --vla-device cuda:0 --batch 32
python3 scripts/b_wm6/test_beta_schedule.py
python3 scripts/b_wm6/baseline_audit.py

# 审计
python3 scripts/b_wm6/audit_critic_grad.py --actor-ckpt <ckpt> --critic-ckpt <critic> --n 24
python3 scripts/b_wm6/audit_target_phys_rot6.py --actor-ckpt <ckpt>
python3 scripts/b_wm6/p7_direction_audit.py --horizon 20 --device cuda:1
python3 scripts/b_wm6/check2_critic_target.py --ckpt <critic> --eval-horizon 20 --tag K20

# 训练（修复版）
python3 scripts/train_b_wm6_h10_critic.py \
  --out runs/dual_actor_wm6/b_delta/recovery_h10_critic/expB_fixed \
  --resume runs/dual_actor_wm6/b_delta/c2b/seed0/checkpoint_last.pt --updates 340 \
  --device cuda:1 --vla-device cuda:0 --batch 256 --horizon 10 --exec-k 20 \
  --tail-control mask10 --lambda-raw 1e-3 --raw-norm-kappa 2 --cap-warmup-updates 0 \
  --beta 1.0 --beta-stage-warmup 40 --beta-slew 0.05 --beta-gate-on 0.35 --beta-gate-off 0.28 \
  --beta-grad-budget 0.20 --critic-enabled 1 --critic-kind v5 --critic-init ckpt \
  --critic-ckpt runs/dual_actor_wm6/b_delta/critic_v5_b3.pt \
  --critic-v5-hidden 512 --critic-v5-depth 2 --critic-updates 4 --critic-batch 2048 \
  --critic-validate-every 5 --on-the-fly 1 --vla-batch 128 --seed 0 \
  --eval-set runs/temporal_ac_bptt_v5/eval_set --eval-every 10 --log-every 10

# 真物理（每臂独立进程）
bash scripts/b_wm6/run_isaac_roll.sh job_base.pt roll_base.pt 1
bash scripts/b_wm6/run_isaac_roll.sh job_b.pt    roll_b.pt    1
python3 scripts/b_wm6/score_pair.py --job-a ... --roll-a ... --job-b ... --roll-b ... --b-ckpt ...
```

---

## 12. 产物索引

```
runs/dual_actor_wm6/b_delta/
├── config/{b_wm6_h10_critic.json, b_lock.json}
├── c0/ c1/ c2b/ c3_v5/                    历史 run（只读）
├── critic_warm_b1.pt  critic_v5_b3.pt     预训 critic
├── recovery_h10_critic/
│   ├── BASELINE_AUDIT.md                  79 ckpt 全表
│   ├── audit/{audit_beta_resume,baseline_audit}.json
│   ├── critic_grad/audit_critic_gradient_paths.json
│   ├── target_audit/{audit_critic_target,check2_critic_target,check2_critic_target_K20}.json
│   ├── phys/{audit_physical_action,p7_direction_audit,pair_fix_h20}.json
│   ├── rot6/audit_rot6.json
│   ├── critic_v5_K20.pt
│   ├── expA_beta0/  expB_fixed/  expB2_histfix/
│   └── final_report/REPORT.md
├── isaac/{job_*,roll_*,pair_*}.pt/json
└── isaac/final2/{tasks_base,tasks_b}/

comimic/docs/
├── B_WM6_FINAL_REVIEW_20261109_v3.md      ← 本文
├── B_WM6_CHECK1_CHECK2_20261109.md        三检查 + §8 K20
├── B_WM6_FULL_REVIEW_20261109_v2.md       v2（已被本文取代）
├── B_WM6_RECOVERY_REPORT_20261109.md      P0–P9
└── B_WM6_REPORT_20261109.md               正式报告草稿
```

---

## 13. 一句话给 GPT

> **别再修 critic 的口径了**（视界、hist 都修过，真物理 v_ev 反而更差）；
> **也别急着刷 WM6**（WM 的判别力问题可能是它本身的性质）。
> 现在唯一能改变结论的动作是**补六任务闭环**（expB vs Base，6 seeds）。
> 在此之前，对外结论只能是 `PASS_WITH_CAVEATS`：机制成立、真物理小幅正收益、任务级未知。
