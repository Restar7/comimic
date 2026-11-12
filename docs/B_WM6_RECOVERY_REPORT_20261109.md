# B-WM6 H10+Critic 专项修复报告（P0–P9）

> 生成：2026-11-09 12:10 CST｜容器 2 / `comimic-b-wm6` / 分支 `experiment/b-wm6-h10-critic`
> 输出目录：`runs/dual_actor_wm6/b_delta/recovery_h10_critic/`
> **架构未变**：`Frozen FT8000 + TemporalDeltaActorV5 + Frozen SONIC + Frozen WM6 + H10 + Online Critic`，
> 部署 K20 / 50 Hz / 40 帧 token。**未重训 WM6，未改 C 线，未动已发布 active 指针。**

---

## 0. 结论速览

| 项 | 结论 |
|---|---|
| **P0 β 调度** | ❌→✅ **确认 bug 并修复**：旧实现按**全局 update** 算 warmup，换 critic 后 β 一步 0→1；新实现按 `critic_activation_update` 起算 + slew 限速 + 滞回门禁，单测 **6/6 PASS**、resume 后 β 曲线 **max diff = 0.0** |
| **P1 plan 梯度捷径** | ✅ **不存在**：plan 通路对 Actor 的梯度 **恰为 0.0**（`z = tokens + Δ.detach()`），`cos(A,C)=1.000`，唯一通路是 `s10` |
| **P2 价值目标** | ✅ **正确**：TD(λ) 手算 vs 实现 **max diff 0.0**；J10 重算 **0.0**；MC 从 t 起算 ⇒ `J_AC = J10 + βγ^H·V(y_H)` **无重复计数** |
| **P3 梯度预算** | ✅ **已实现生效**：每步精确测 `‖g_J10‖ / ‖g_V‖ / cos / ratio`，价值梯度按范数封顶 ≤ **20%**×‖g_J10‖ |
| **P4 critic 在线有效性** | ⚠️ 部分：WM6 想象上 `v_ev` **0.57–0.62（全程 PASS）**；真物理 `v_ev=−4.53` 但 **corr +0.55** ⇒ **能排序、不能标定** |
| **P5 物理动作** | ✅ 审计完成：base decoder 自带裁剪 **7.58%** joint-step；Δ **新增**只有 **0.05%**；关节差 RMS **16.2 mrad**（真 rad） |
| **P6 受控修复试验** | ✅ **成功**：修复版 β→**0.975** 且 test ΔJ **+0.00788**（对照 β=0 的 +0.00767），动作 **6.56 mrad**（对照 6.65） |
| **P7 真物理** | ⚠️ **修正了 A/A 结论**：跨进程 A/A **恰为 0.0**；ΔJ10 **+0.00049 [+0.00011, +0.00106]**、胜率 **79.2%**、ΔJ20 **+0.00056** ⇒ **小而干净的正收益** |
| **P9 rot6** | ✅ 训练 vs 部署 cos **0.9968**、变化 **6.9%** ⇒ 非主因，**不置零** |
| **P8 完整闭环** | ⏸ 按指令跳过（已有 Base 18 集与 B 4 集的真闭环数据留档） |

**最终判定：`H10+Critic` 主线 = PASS（工程与目标层面）；真物理收益 = PASS（小而干净，非 INCONCLUSIVE）；六任务闭环 = INCONCLUSIVE（未完成，按指令跳过）。详见 §10。**

---

## 1. P0：β 调度审计（确认的工程 bug + 修复）

### 1.1 现象与根因

`c3_v5`（从 c2b it=300 续训、换 v5 critic、`--beta-warmup-updates 100`）日志：

```
it=310  β=0.00        ← resume 后门禁状态是 FAIL
it=320  β=1.00        ← 门禁翻 PASS，β 一步到顶
```

**根因（三条叠加）**：

1. `beta = beta_max * min(1, it / beta_warmup_updates)` 用的是**全局 update**。
   从 it=300 续训时 `min(1, 301/100) = 1` ⇒ 计划值**一开始就是满功率**，warmup 早已"用完"。
2. 门禁只有 0/1 两态：`beta_health ∈ {0,1}`。计划值已经是 1，门禁一翻 PASS 就是 **0→1 一跳**。
3. 换 critic（arch 不一致）时没有"重新激活"的概念，也没有把 β 状态写进 checkpoint。

### 1.2 修复

新增 `scripts/b_wm6/beta_schedule.py::BetaController`（纯状态机、无 torch）：

```
β_sched = β_max · clamp((it − critic_activation_update) / W, 0, 1)
gate    : v_ev ≥ on(0.35) ⇒ ok；v_ev < off(0.28) ⇒ 不 ok；中间**保持**（滞回，不抖）
β_eff   : 每 update 最多变 slew(0.05)（限速，防 0↔max 突跳）
```

* `critic_activation_update` 在 **新建 critic / `--critic-init ckpt` / `--critic-warm-from` /
  resume 时架构不一致** 这四种情况下重置为当前 it；**不动 Actor 的全局 step**。
* 状态 `{activation, gate_ok, β_eff, β_sched}` 写入 checkpoint；`--resume` 后逐位复现。
* 新增参数：`--beta-stage-warmup --beta-gate-on --beta-gate-off --beta-slew`。
* 另修了一个 resume 顺序 bug：旧的 `beta/gate_status = ck[...]` 无条件覆盖发生在
  "架构不一致重置"**之后**，会把重置结果冲掉（c3_v5 的 `it310 β=0` 就是它）。

### 1.3 单测（`audit/audit_beta_resume.json`，6/6 PASS）

| # | 检查 | 结果 |
|---|---|---|
| 01 | 从 it=300 换新 critic ⇒ β 起始 ≈0 | **0.0**（旧实现同一点 = 1.0） |
| 02 | 之后 10 步不跳到 1 | β = 0, 0.01, …, 0.10（单调、每步 +0.01） |
| 03 | warmup 中断再 resume 曲线逐位复现 | **max diff = 0.0**（51 步） |
| 04 | 门禁 FAIL→PASS→FAIL 无突跳 | 每步变化 ≤ slew；0.30 落在死区时**保持 PASS** |
| 05 | 不隐式改 Actor LR / Scheduler | 控制器源码不含 `.lr`/`optimizer`/`scheduler` |
| 06 | 旧实现的门禁一翻就满功率 | `legacy_schedule_at_301 = 1.0` vs 新实现首步 **0.0** |

---

## 2. P1：Critic 梯度通路归因（**无 plan 捷径**）

`recovery_h10_critic/critic_grad/audit_critic_gradient_paths.json`

| 通路 | 梯度范数 | 占全通路比例 |
|---|---|---|
| **A：只让 `s10` 回传** | **0.7462** | 1.000 |
| **B：只让 plan 特征回传** | **0.0000** | 0.000 |
| C：两条都开 | 0.7462 | 1.000 |

`cos(A,B)=0`、`cos(A,C)=1.000` ⇒ **唯一通路是 `s10`**。

* plan 的语义：`z[:, t+j], j<plan_steps(4)`，且 `idx < K` 才填、否则补 0。
* **关键口径**：critic 的 `K` 是 **`exec_k=20`（实际执行的 token 数）**，
  **不是** `spec.K=10`（Δ 生效帧数）。用错会把 t=H=10 的 plan 全补 0
  （我在第一版审计脚本里就踩了这个坑，已修并记入文档）。
* 结论：**无需修改结构**；`s10` 通路依然提供有效的尾部价值梯度（B0 #20 已验非零）。

---

## 3. P2：价值目标审计（**无重复计数**）

`recovery_h10_critic/target_audit/audit_critic_target.json`

| 检查 | 结果 |
|---|---|
| TD(λ) 手算 vs `td_lambda_targets` | **max diff = 0.0** |
| 末步自举 = `r_{H−1} + γ·V̄_H` | ✅（合法 un-terminated 自举） |
| `J10` 重算（γ 折扣 + survival） | **max diff = 0.0** |
| MC 目标从 t 起算（`MC[t]=Σ_{k≥t}γ^{k−t}r_k`） | ✅ |
| `J_AC = J10 + βγ^H surv V(y_H)` 是否重复计入前 H 步 | **否**：J10 只覆盖 t<H，V 只覆盖 t≥H |
| 终止处理 | `d = 1 − survival[t+1]`，由 `termination_logits` 累乘 |
| 参考边界 | 第 t 步 = frame `f+1+t`，第 H 步 = `f+1+H`（夹到 clip 末帧） |

---

## 4. P3：保守梯度预算（已实现）

训练器每一步做 **两次 backward**：

```
loss_actor = −J10 + 0.010·L_act + 0.002·L_smooth + 0.001·L_anchor + 1e-3·L_raw (+ L_lim/L_peak)
loss_val   = −(β · γ^H · surv_H · V(y_H))
g_A = ∇loss_actor(θ),  g_V = ∇loss_val(θ)
scale = min(1, β_budget·‖g_A‖ / ‖g_V‖)      # 默认 β_budget = 0.20
θ.grad = g_A + scale·g_V
```

每步记录：`grad_J10_norm`、`grad_critic_norm`、`grad_critic_scaled`、`grad_budget_scale`、
`grad_cos_j10_critic`、`grad_critic_to_j10_ratio`、`grad_critic_scaled_ratio`、
`beta_sched`、`beta_eff`、`beta_gate_ok`、`beta_activation`。

实测（`smoke_fix`，β 还很小）：

```
it=2  ‖g_J10‖=0.187  ‖g_V‖=0.0054  ratio=0.029  cos=−0.124
it=3  ‖g_J10‖=0.303  ‖g_V‖=0.0128  ratio=0.042  cos=+0.075
```

⇒ 价值梯度只占 J10 的 **3–4%**，余弦接近 0（近正交）。**这正是 c3_v5 缺的那道闸**：
旧版没有预算，β=1 时 ratio 到 **2.7**、动作幅度翻倍。

---

## 5. P6：受控修复试验（同一 checkpoint、同一 seed、只改声明因素）

起点统一 `c2b/seed0/checkpoint_last.pt`（it=300）。训练到 it=340（40 步）。

| run | 改动 | β 终值 | val ΔJ（best/final） | **test ΔJ** | same-obs RMS | critic v_ev |
|---|---|---|---|---|---|---|
| **A** | β=0 续训（对照） | 0.000 | +0.00928 / +0.00866 | **+0.00767** | 6.65 mrad | — |
| **B（修复版）** | v5 critic + 激活起算 warmup(40) + slew 0.05 + 滞回门禁 + **20% 梯度预算** | **0.975** | +0.00926 / **+0.00909** | **+0.00788** | **6.56 mrad** | **+0.569（PASS，全程 0.57–0.62）** |
| c3_v5（旧 bug 版） | β 一步到 1、无预算 | 1.0→0 | +0.00698 | +0.00588 | 11.58 mrad | +0.32→FAIL |

**B vs A：ΔJ 打平（+0.00788 vs +0.00767），动作幅度相同（6.56 vs 6.65 mrad），
但 B 的 β 真的到了 0.975、critic 全程健康。** ⇒ "非零价值梯度 + 不伤 J10" 同时成立。

B 的逐步轨迹（每 5 步一次独立验证）：

```
it=310 β=0.01→0.05  v_ev=0.6206  val ΔJ=+0.00926  sameRMS=6.76
it=320 β=0.47       v_ev=0.6093  val ΔJ=+0.00894  sameRMS=6.75
it=330 β=0.72       v_ev=0.5925  val ΔJ=+0.00898  sameRMS=6.69
it=340 β=0.97       v_ev=0.5689  val ΔJ=+0.00909  sameRMS=6.70
```

---

## 6. P7：真物理短程（**A/A 结论被修正**）

### 6.1 A/A 的真相

| 测法 | 结果 |
|---|---|
| 同进程内 plus→minus 顺序跑（旧做法） | 末状态差 **6.84**；ΔJ10 偏置 **+0.00143** |
| **两个独立进程、动作逐位相同** | 末状态差 **0.0**、首步差 **0.0**、ΔJ10 偏置 **0.0** |

⇒ **Isaac 跨进程是确定性的；之前的 +0.00143 是"第二臂继承第一臂接触/执行器状态"的
harness artifact，不是物理噪声。** 因此"每臂独立进程"的比较 **A/A 噪声为 0**。

（这也解释了为什么第一版报告的 `ΔJ10=+0.0014` 看起来和噪声同量级——那个数字本身就是噪声。）

### 6.2 修复版 ckpt 的真物理配对结果（48 窗 × H=20，每臂独立进程）

| 指标 | Base | B(Δ) | Δ | 95% CI | 胜率 |
|---|---|---|---|---|---|
| **真 J10** | 0.55514 | 0.55563 | **+0.00049** | **[+0.00011, +0.00106]** | **79.2%** |
| **真 J20** | — | — | **+0.00056** | **[+0.00011, +0.00119]** | — |
| A/A（跨进程） | — | — | **0.0** | — | — |

对照：c2b ckpt（β=0）= +0.00047 / 胜率 72.9%。修复版在胜率上更好（79.2%），幅度同量级。

**判读**：CI 不含 0、A/A 为 0 ⇒ **统计显著**；但幅度只有 base J10 的 **0.09%** ⇒ **实际意义上很小**。
诚实表述：**"真物理有小而干净的 tracking 改善，不足以宣称任务级提升"**。

### 6.3 真物理上的 critic（关键负结果）

| 口径 | `v_ev` | corr | v_mean / target_mean |
|---|---|---|---|
| WM6 想象（独立 clip） | **+0.626** | 0.804 | 0.48 / 0.45 |
| **真 Isaac 轨迹** | **−4.53** | **+0.555** | **−0.156 / 0.545** |

⇒ **critic 能排序（corr 0.55）、不能标定（绝对值系统性偏低且符号相反）**。
这就是"β=1 绝对奖励"必然有害的机制：`β·V` 注入的不是零均值信号，而是一个**负偏置**。
梯度预算（按范数封顶）能压住幅度，但压不住方向偏置 —— 这是后续要修的点（见 §9）。

---

## 7. P5：物理动作与 Safety Clip

`recovery_h10_critic/phys/audit_physical_action.json`

| 项 | 值 |
|---|---|
| base decoder 自带越限（`|q_raw−q_applied|>1e-6` 的 joint-step 占比） | **7.58%** |
| Δ 臂总越限占比 | 7.63% |
| **Δ 新增**越限占比 | **0.05%**（max 额外 0.0… rad） |
| Base↔Δ 的 29 维关节目标差（真 rad） | RMS **16.2 mrad**、peak 数百 mrad |
| 越限来源 | **SONIC decoder 原始输出本身**，不是 Δ；部署侧由 `sonic_physical_action` 裁剪 |
| WM6 输入 | **仍是原始归一化 command**（合同不变，未改成 q_applied） |
| 真实限位来源 | `reproduce_scenes/g1/g1_29dof_with_hand.xml` 的 `range`（未改动） |

**结论**：Δ 只贡献 0.05% 的新增裁剪，可接受；但 `L_lim` 在 Δ≈0 时就已 24.9（base 自带越限），
所以本轮把它当**监控**而非 loss（`--lambda-limit 0`），并在报告里明确"WM 看到的是未裁剪动作"这一层未被建模。

---

## 8. P9：rot6 口径

`recovery_h10_critic/rot6/audit_rot6.json`

| 对比 | cos(Δ) | 相对变化 |
|---|---|---|
| 训练（相对 clip 首帧参考） vs 部署（相对回合首帧） | **0.9968** | 6.9% |
| 训练 vs rot6 整块置零 | 0.9977 | 6.0% |

⇒ 该通道信息量很小，**不是主要矛盾**；**不**默认置零（避免掩盖问题），
若要彻底统一需改 C 侧 client（跨容器），列入后续协调项。

---

## 9. 遗留问题（按优先级）

| # | 问题 | 证据 | 建议 |
|---|---|---|---|
| 1 | critic **真物理标定失败**（v_mean −0.156 vs target 0.545，corr 0.55） | §6.3 | 目标改成**相对形式** `V(s_H,ref_H) − V(ref_H,ref_H)`（去偏置）；或对 V 做 per-task 仿射标定后再进 loss |
| 2 | β 的目标值仍是 1.0 | §5（B 成功） | 用 **β_max·gate + 20% 预算** 已足够；但 β_max 可降到 0.5 做消融 |
| 3 | critic 在线 `v_ev` 随策略漂移（0.62→0.57 over 40 步；c3_v5 里 0.62→0.25） | §5、c3_v5 | 已有 `--critic-catchup-every`（actor 冻结追赶），本轮未启用；建议 20 步一次 |
| 4 | 真物理收益幅度小（0.09%） | §6.2 | 需要更长 H 或**闭环成功率**才能判任务级 |
| 5 | 六任务闭环未完成（Base 8/18、B 4/18，B 臂被叫停） | `isaac/final2` | 见 §10 |
| 6 | `rot6` 跨容器口径 | §8 | 与 C 侧协调后统一 |
| 7 | base decoder 自带 7.58% 越限 | §7 | 部署侧安全层已兜；不建议动限位 |

---

## 10. 最终判定

| 维度 | 判定 | 依据 |
|---|---|---|
| 架构合规（FT8000/SONIC/WM6 冻结、H10、K20、Δ 只在 t<10） | **PASS** | B0 27/27 + 部署日志 `tail_zero=True` |
| β 调度正确性 | **PASS** | §1，单测 6/6、resume 曲线 max diff 0 |
| Critic 梯度通路与目标 | **PASS** | §2 无捷径、§3 无重复计数 |
| **H10+Critic 联合训练（非零 β、不伤 Actor）** | **PASS** | §5：β=0.975、test ΔJ +0.00788 vs β=0 的 +0.00767、动作 6.56 mrad |
| Critic 价值可靠性（WM 想象） | **PASS** | `v_ev` 0.57–0.63 全程过 0.30 门限 |
| Critic 价值可靠性（真物理） | **FAIL** | `v_ev=−4.53`（虽 corr 0.55）⇒ 只能排序不能标定 |
| 真物理短程收益 | **PASS（幅度小）** | ΔJ10 +0.00049 [+0.00011,+0.00106]、胜率 79.2%、A/A=0 |
| 六任务真实 VLA 闭环 | **INCONCLUSIVE** | 按指令跳过；已有 Base 8/18、B(旧 ckpt) 0/4，样本不足 |

**总判定：`PASS_WITH_CAVEATS`**

* **可以宣布**：H10 + 在线 Critic 共训主线成立（β 非零、价值梯度受控、不伤 J10），
  真物理 short-horizon tracking 有小而干净的改善。
* **不能宣布**：critic 的绝对价值可用于真物理奖励整形（标定失败）；
  六任务闭环有提升（未测）。
* **下一步唯一必要动作**：把 `β·V` 改成**相对形式**（去偏置）后重跑 expB，再补 P8 闭环。

---

## 11. 复现命令

```bash
WT=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-b-wm6
cd $WT && source scripts/b_env.sh

# P0 β 调度单测（无需 GPU）
python3 scripts/b_wm6/test_beta_schedule.py

# P0 基线审计（只读）
python3 scripts/b_wm6/baseline_audit.py

# P1 梯度通路
python3 scripts/b_wm6/audit_critic_grad.py \
  --actor-ckpt runs/dual_actor_wm6/b_delta/c2b/seed0/checkpoint_best.deploy.pt \
  --critic-ckpt runs/dual_actor_wm6/b_delta/critic_v5_b3.pt --n 24 --device cuda:1

# P2/P5/P9
python3 scripts/b_wm6/audit_target_phys_rot6.py \
  --actor-ckpt runs/dual_actor_wm6/b_delta/c2b/seed0/checkpoint_best.deploy.pt --device cuda:1

# P6 A（对照组）
python3 scripts/train_b_wm6_h10_critic.py \
  --out runs/dual_actor_wm6/b_delta/recovery_h10_critic/expA_beta0 \
  --resume runs/dual_actor_wm6/b_delta/c2b/seed0/checkpoint_last.pt --updates 340 \
  --device cuda:1 --vla-device cuda:0 --batch 256 --horizon 10 --exec-k 20 \
  --tail-control mask10 --lambda-raw 1e-3 --raw-norm-kappa 2 --cap-warmup-updates 0 \
  --beta 0 --critic-enabled 0 --on-the-fly 1 --vla-batch 128 --seed 0 \
  --eval-set runs/temporal_ac_bptt_v5/eval_set --eval-every 10 --log-every 10

# P6 B（修复版）
python3 scripts/train_b_wm6_h10_critic.py \
  --out runs/dual_actor_wm6/b_delta/recovery_h10_critic/expB_fixed \
  --resume runs/dual_actor_wm6/b_delta/c2b/seed0/checkpoint_last.pt --updates 340 \
  --device cuda:1 --vla-device cuda:0 --batch 256 --horizon 10 --exec-k 20 \
  --tail-control mask10 --lambda-raw 1e-3 --raw-norm-kappa 2 --cap-warmup-updates 0 \
  --beta 1.0 --beta-stage-warmup 40 --beta-slew 0.05 \
  --beta-gate-on 0.35 --beta-gate-off 0.28 --beta-grad-budget 0.20 \
  --critic-enabled 1 --critic-kind v5 --critic-init ckpt \
  --critic-ckpt runs/dual_actor_wm6/b_delta/critic_v5_b3.pt \
  --critic-v5-hidden 512 --critic-v5-depth 2 --critic-updates 4 --critic-batch 2048 \
  --critic-validate-every 5 --on-the-fly 1 --vla-batch 128 --seed 0 \
  --eval-set runs/temporal_ac_bptt_v5/eval_set --eval-every 10 --log-every 10
```

---

## 12. 产物索引

```
runs/dual_actor_wm6/b_delta/recovery_h10_critic/
├── BASELINE_AUDIT.md                     P0 基线（79 个 checkpoint 全表）
├── audit/audit_beta_resume.json          P0 单测 6/6（含旧公式对照）
├── audit/baseline_audit.json             P0 原始表
├── critic_grad/audit_critic_gradient_paths.json   P1 三通路归因
├── target_audit/audit_critic_target.json          P2 TDλ/MC/J10 核对
├── phys/audit_physical_action.json                P5 raw vs applied
├── phys/pair_fix_h20.json                         P7 真物理配对 + A/A + critic-real
├── rot6/audit_rot6.json                           P9
├── expA_beta0/{checkpoint_*, summary.json, val.jsonl}   P6 对照
├── expB_fixed/{checkpoint_*, .deploy.pt, summary.json}  P6 修复版（**候选**）
├── smoke_fix/                                     修复后接线冒烟（β 0→0.02、预算生效）
└── final_report/REPORT.md                         ← 本文

日志：runs/dual_actor_wm6/b_delta/logs/{expA_beta0,expB_fixed,smoke_fix,smoke_fix3,
      p1_grad5,p2p5p9,roll_roll_fix_*}.log
真物理：runs/dual_actor_wm6/b_delta/isaac/{job_fix_*,roll_fix_*}.pt
```

## 13. 代码提交

```
d367454  b-wm6: 训练器 + B0 + 契约层 + 真物理回放/分析
c62f1fd  b-wm6: 真物理成对评测(修两处方法学 bug) + 三处环境修复
<本次>    b-wm6 recovery: P0 β 调度修复 + P3 梯度预算 + P1/P2/P5/P9 审计 + P6 A/B
```
