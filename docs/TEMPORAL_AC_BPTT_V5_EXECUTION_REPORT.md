# v5 主线执行报告（worktree `comimic-wt03` / `harness/03-bptt-diagnosis`）

> 规范：`docs/TEMPORAL_AC_BPTT_V5_MAINLINE_SPEC.md`。
> 本文只记录**本机实际做了什么、验证结果、与规范的偏差**；不复述规范。
> 生成时间：2026-11-07 20:5x（训练进行中，见 §6 的实时状态）。

---

## 0. 一句话状态

- **代码全部改在 worktree**（`comimic-wt03`），**主仓库 main 未改任何文件**（§8 有审计命令）。
- 规范 §2 的 4 个部署契约文件已从 `788d1ee` 取来并通过测试；§9 的 6 个新测试全绿；
  §8 M0/M1 完成；§8 M2 的**评测集已建好**、**2 个 seed 的主训练已启动**（on-the-fly）。
- **两处规范/文档级修正**（都是实测发现，很重要）：
  1. **训练侧关节序是 `standard`（MJCF），不是 HA_V31**；部署 `state64` 才是 HA_V31 ⇒
     契约层必须换序（已实现，强口径对拍两条路径 `max|Δ|=0`）。
  2. **hist10 的 `default` 与动作换算的 `default` 是两个不同的常量**
     （`g1_default_joint_pos()` vs `bundle.default_joint_pos`，差 ≈0.27 rad）。
- §5.5.5 的 WM 40 步门槛：B 路线 **NOT_RUN**（`INCONCLUSIVE_LIMITED_DATA`，见 §5）。
- **部署集成仍标 `BLOCKED_ON_SESSION01`**（规范 §2.4：服务端/探针归会话 01）。
- 代码已提交到 worktree 分支：**`befeff5`**（`harness/03-bptt-diagnosis`），**未合入 main**；
  两个 seed 的 `git_revision.txt` 记录了这个 commit + 基线 `a4bd1ff`。
- **真 checkpoint 的部署契约自检通过**（`seed{0,1}/deploy_contract_check.txt`）：
  `params=1,744,832`、`T/K/H=40/20/20`、`‖Δ‖max=0.0712 ≤ 0.08`、`t≥K` 恒 0、
  部署序 vs 直连训练序 `max|diff|=0.0`。

---

## 1. 目录与产物（全部在 worktree，主仓库只读）

| 类别 | 路径 |
|---|---|
| 代码（新） | `diff_vla_wbc/delta_token_temporal_v5.py`（已有，复核）、`delta_token_io_v5.py`、`imagined_value_v5.py`、`action_trust_v5.py` |
| 代码（取） | `diff_vla_wbc/delta_token_io.py`、`scripts/verify_delta_token_deploy.py`、`tests/test_delta_token_io.py`、`tests/test_deploy_audit.py`（来自 `788d1ee`） |
| 代码（改） | `scripts/diag_bptt_learnability.py`（`--uniform-frames`/`--h-tail`、窗口硬约束、`eligible_start_mask`、`FrozenVLASampler`、py3.11 语法修复）、`scripts/train_temporal_ac_bptt_v5.py`（on-the-fly 主训练） |
| 脚本（新） | `scripts/launch_v5_main.sh`、`scripts/check_history_parity_v5.py`、`scripts/verify_v5_deploy_contract.py`、`scripts/wm_horizon_error_v5.py`、`scripts/wm_dump_check_v5.py` |
| 测试（新） | `tests/test_temporal_v5_grad.py`、`test_history_and_token_index.py`、`test_value_td_lambda.py`、`test_actor_tail_value_grad.py`、`test_physical_action_trust.py`、`test_v5_fail_closed.py` |
| 文档（新） | 本文、`docs/TEMPORAL_AC_BPTT_V5_RUNTIME_CONTRACT.md`、`docs/LEGACY_BPTT_CRITIC_AUDIT.md` |
| 评测集 | `runs/temporal_ac_bptt_v5/eval_set/`（`windows.pt` 448 窗口 + `vla_tokens.pt` 2 个实现 + manifest） |
| 训练 run | `runs/temporal_ac_bptt_v5/main/seed{0,1}/`（见 §6） |
| 动作预算诊断 | `runs/temporal_ac_bptt_v5/main/seed0/action_budget_diag.json`（逐关节/逐控制步） |
| WM 测量 | `runs/temporal_ac_bptt_v5/main/wm_horizon_error.{json,csv}`、`wm_dump_check.json` |
| FD 检查 | `runs/temporal_ac_bptt_v5/main/exp0/exp0.json` |

> 输出放在 **worktree 的 `runs/`**（gitignored）里，而不是规范 §8 M2 写的 `$COMIMIC_MAIN/runs/...`，
> 因为用户要求"不要改 main"。除路径外其余命令与规范逐字一致。

---

## 2. §2 部署契约对齐（M0/M1 门禁）

```bash
python3 -m pytest tests/test_delta_token_io.py tests/test_deploy_audit.py -q
# => 25 passed, 14 skipped
python3 scripts/verify_delta_token_deploy.py --help    # 可启动
git status --short | grep -E "v3_step|teacher_probe"   # 空（未迁移主线）
```

v5 全部测试：

```bash
python3 -m pytest tests/test_temporal_v5_grad.py tests/test_value_td_lambda.py \
    tests/test_actor_tail_value_grad.py tests/test_history_and_token_index.py \
    tests/test_physical_action_trust.py tests/test_v5_fail_closed.py -q
# => 45 passed（+ v4 的 25 passed / 14 skipped）
```

**M1 有限差分**（`diag exp0 --fd-at grad --n 8`）：

| h | analytic | FD | rel_err |
|---|---|---|---|
| 1e-2 | 0.3298905 | 0.3193140 | **3.21e-2** |
| 3e-3 | 0.3298905 | 0.3285408 | **4.09e-3** |
| 1e-3 | 0.3298905 | 0.3295541 | 1.02e-3 |
| 3e-4 | 0.3298905 | 0.3302097 | 9.68e-4 |

⇒ 梯度路径正确（h=1e-2 时 3.2%，略高于规范的 3% 门限；h 减小后 4e-3，符合 float32 舍入预期）。
**结论：PASS（带脚注）**。

---

## 3. §3 数据合同（实测）

| 项 | 规范 | 本机实测 | 判定 |
|---|---|---|---|
| 总帧数 | — | 634,016（699 clip） | — |
| 漂移合格帧（`start_max_drift=0.10`，含 `w==0` 前一帧回填原坑） | **346,729** | **346,729** | ✅ 逐位一致 |
| §3.4 窗口约束后（`f≥10`、`f+H≤clip_end`） | — | **338,402**（−2.4%） | ✅ `uniform_over_eligible=true` |
| 评测集 | val/test 各 96 | 448 窗口 = 256/96/96，**uniform** 抽帧，val/test 各自 81 个**独立 clip** | ✅ |
| burn-in | k=10，扰动 5cm/0.1rad/0.05rad/0.3rad/s | 同 | ✅ |
| 抽样 | `multinomial(w,B)` 均匀 | 同（每步现场抽） | ✅ |
| VLA | 现场采样，每步新噪声 | 同（`fm_steps=4`、`vla_batch=128`） | ✅ |

---

## 4. §4 History10 对齐（强口径，本机复跑）

`scripts/check_history_parity_v5.py`（真窗口 N=32，**部署序 HA_V31 → 训练序 standard**）：

```json
{"max_abs_offset_const_across_windows": 5.96e-08,
 "max_abs_q_block_vs_state": 1.19e-07,
 "max_abs_qd_block_vs_state": 0.0,
 "max_abs_hist10_train_vs_deploy_state64_path": 0.0,
 "max_abs_hist10_train_vs_deploy_blocks_path": 0.0,
 "passed": true}
```

两条部署路径（`state64` 序列 / client decoder 两块）都逐位一致 ⇒ **PASS**。

### 4.1 两处**必须写进报告**的修正

**(a) 训练侧关节序 = `standard`，不是 HA_V31。** 规范 §1.3 的逐位测量只能证明
"obs93 的 q/qd 与 state64 的 dof_pos/dof_vel **同序**、差一个 `−default`"，不能区分绝对序。
四处代码证据判定为 STANDARD：
1. `MDP.obs` 直接算 `q − default_q`（`supermimic.py:171`），而
   `bundle.default_joint_pos[:6]=[-0.2,0,0,0.42,-0.23,0]` 的 **knee 在 index 3**（STANDARD 序）；
2. `SonicIsaacBundle.step` 注释："状态的关节序天生是 STANDARD（语料/VLA 观测/FK 全按它）"；
3. `SonicExecutor.obs994` 对 q/qd/last_action 做 `index_select(std_to_ha)` 才喂 SONIC decoder；
4. `delta_token_io`（v4）把训练侧约定写成 `standard`。

⇒ v5 契约：`state_joint_order=ha_v31`（入站声明）→ **内部换到 standard** → 网络；
反向（`q_target`、部署施加）由会话 01 的集成层负责。

**(b) 两个 `default_joint_pos` 不是同一个东西。**
- hist10 的 q 偏移 = `g1_default_joint_pos()` = `[-0.312, 0, 0, 0.669, -0.363, …]`（mjlab `KNEES_BENT`）；
- 动作换算 `q_target = default + scale·a` 的 default = `bundle.default_joint_pos` =
  `[-0.2, 0, 0, 0.42, -0.23, …]`（SONIC/MDP，`SonicIsaacBundle` 构造时换过）。
- 两者差 ≈0.27 rad；混用会让 hist10 整体平移（本机用错时 `max|Δ|=0.27` 假失败）。
- ckpt meta 同时写 `hist_default_joint_pos` 与 `default_joint_pos`/`action_scale`。

---

## 5. §5.5.5 WM 40 步误差（B 路线门槛）

**结论：B 路线 `NOT_RUN`，标 `INCONCLUSIVE_LIMITED_DATA`。**

| 测量 | 口径 | pos p50 @k=20 | pos p50 @k=40 | 判定 |
|---|---|---|---|---|
| `wm_horizon_error_v5.py` | 起点=参考状态，动作=**参考命令代理** | 110.9 mm | 232.7 mm | 代理口径**不可用于绝对阈值** |
| `wm_dump_check_v5.py` | 起点/动作/目标=B dump 连续数据（历史同款） | **21.2 mm**（p90 77.3） | — | 仍 > 15 mm 阈值 |

- **为什么代理口径无效**：`all_transitions.pt` 是稀疏采样（"同 clip 且帧号 +1"的相邻行只占
  **0.05%**，最长连续段 85 行仅 2 段）⇒ 没法回放 40 个连续控制步；用 `q_ref` 当动作会把
  "命令 vs 执行"的差算进 WM 误差里（11 cm vs 2 cm）。
- **自证动作顺序**：dump 两种口径都跑了 —— `actions` 直接喂（STANDARD→`bundle.step` 内部转 HA）
  得 21.2 mm；先换 HA 再喂得 121.2 mm（5.7× 差）⇒ **正确口径 = policy 输出 STANDARD 序**，
  与 `bundle.step` 的注释一致。
- **为什么仍判 NOT_RUN**：21.2 mm > 规范阈值 15 mm，且只有 1 个 dump/clip
  （`DoubleDesk_proven_s0_ep000`，2080 帧，64 起点）⇒ 数据不足，不能据此批准 B。
  A 路线（`K=20`）不受影响。

---

## 6. §8 M2 主训练（已启动）

**评测集**（448 窗口，uniform）：

```bash
OUT=$WT/runs/temporal_ac_bptt_v5
python3 scripts/diag_bptt_learnability.py build --out $OUT/eval_set \
    --n-train 256 --n-val 96 --n-test 96 --batch 64 --vla-batch 4 \
    --uniform-frames 1 --seed 0
```

**主训练**（2×4090 各一个 seed，见 `scripts/launch_v5_main.sh`）：

```bash
OUT=$WT/runs/temporal_ac_bptt_v5
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_temporal_ac_bptt_v5.py \
    --out $OUT/main/seed0 --seed 0 --device cuda:0 --updates 1000 --batch 256 \
    --horizon 20 --beta 0 --on-the-fly 1 --eval-set $OUT/eval_set --vla-batch 128
CUDA_VISIBLE_DEVICES=1 python3 scripts/train_temporal_ac_bptt_v5.py \
    --out $OUT/main/seed1 --seed 1 --device cuda:0 --updates 1000 --batch 256 \
    --horizon 20 --beta 0 --on-the-fly 1 --eval-set $OUT/eval_set --vla-batch 128
```

已确认的锁定口径：`T/K/H = 40/20/20`、`mask t<K`、`cap 0.04→0.08`（100 updates）、
`γ=0.99`、`λ_act/τ/λ_smooth/λ_anchor = 0.010/0.020/0.002/0.001`、
`head 3e-4 / enc 1e-4` + 50 warmup + cosine、`clip 1.0`、`val 每 50`、早停 6、
`critic OFF`（日志写 `BOOTSTRAP_NOT_IDENTIFIABLE(K==H==20)`）。

**启动时间 / 实测速度**（2026-11-07 20:19 启动）：

- ~15.2–15.5 s/step（data≈11 s 占 2/3，rollout≈1.3 s，backward≈2.2 s）；
- 1000 updates ≈ **4.3–4.6 h/seed** ⇒ 预计 **00:50–01:10** 结束
  （规范估 9–10 s/step ⇒ 2.6 h；差的 1.7× 主要是 CPU 侧 burn-in/VLA 图像预处理，
  bptt1 的 `prefetch_vla` 把这段藏起来了，本版先不做预取）；
- 两块 4090 各 ~27.9 GB/49 GB，GPU util 3–100%；进程已 `setsid` 脱离会话，
  不会随本会话结束而终止。

**it=50 的第一次 held-out 评测（实时快照）**：

| seed | val J | base | ΔJ | actRMS | peak |
|---|---|---|---|---|---|
| 0 | 1.50635 | 1.50594 | **+0.00041** | 42.3 mrad | 2018 mrad |
| 1 | 1.50580 | 1.50594 | **−0.00014** | 52.6 mrad | 2156 mrad |

> it=50 还在 `cap` warmup（0.060）内、且只有 50 步，ΔJ 在噪声量级，**不能据此判断成败**；
> 早停规则（连续 6 次无改善）会决定是否提前结束。

**动作预算诊断**（`seed0/checkpoint_actor_best.pt` @ it=50，val 96 窗口）：

| 量 | 值 |
|---|---|
| 逐控制步 RMS | t=0..4: 19/23/25/24/27 mrad；t=19: **81 mrad** |
| 逐控制步 peak | t=0: 0.082 rad；t=19: **2.54 rad**（left_ankle_pitch，相位切换敏感关节） |
| 两个 rollout 的**终点状态差** | p50 **2.8 mm**、p90 5.7 mm、max 22 mm |

⇒ 晚期的"2.5 rad"**不是状态发散**（终点只差 mm 级），而是 decoder 对 token/state 微小差异
的**相位敏感**（踝关节早/晚摆切换）；前 5 步（部署最关心的区间）RMS 19–27 mrad。
`action_budget.csv` 里的 peak 是含晚期步的全局 max，报告时必须带这个分解。

### 6.1 学得动吗？——三个诊断（it=100 时）

**(a) 评测集的可提升上界（oracle，`diag_oracle_upper_v5.py`，val 96 窗口）**

| 量 | 值 |
|---|---|
| 逐窗口 oracle 方向（∂J/∂Δ + cap 范数）ΔJ | **+0.0298**（p50 +0.0270，96.9% 窗口为正） |
| 当前 Actor（it=50 ckpt）ΔJ | +0.00041（**1.4%** 上界） |
| cos(Δ_actor, 逐窗口梯度) | **0.0055** |
| cos(Δ_actor, 窗口平均梯度) | 0.028 |
| "沿平均梯度方向走 cap"的一阶 ΔJ | +0.0078（= **与输入无关的常数解**的上界） |

⇒ 评测集对 Δ **是敏感的**（不是"怎么训都白搭"）；Actor 目前只拿到上界的 1.4%，
连"常数解"的 0.0078 都还没到。

**(b) 梯度本身对不对？——参数级有限差分**（`head.0/head.4/token_proj` 的最大梯度元素）

| 参数 | analytic | FD（h=0.03） | rel_err |
|---|---|---|---|
| `head.0.weight` | −5.194e-4 | −5.205e-4 | 2.3e-3 |
| `head.4.weight` | +6.826e-3 | +6.825e-3 | 2.2e-4 |
| `token_proj.weight` | −2.597e-4 | −2.583e-4 | 5.5e-3 |

⇒ **BPTT 梯度管线正确**（差别只在 float32 精度）。不是"梯度断了/符号反了"。

**(c) 梯度信噪比**（固定评测 token，val 96 窗口拆两半）

| 量 | 值 |
|---|---|
| ‖mean grad(96)‖ | 0.252 |
| mean‖batch grad(8)‖ | 0.573（ratio 0.44） |
| cos(batch8, mean96) | 0.32 |
| **cos(g_half1, g_half2)（各 48 窗口）** | **0.131** |

⇒ **单窗口梯度方向主要被"窗口自身"的噪声主导**；即使 48 个窗口平均，两个独立估计的方向
余弦也只有 0.13。训练里每步还叠加**新的 VLA 流噪声**，信噪比更低。
配合 AdamW（按参数归一化）+ grad clip，每步都走满 $\eta$ 的**噪声方向** ⇒
前 100 步表现为"‖Δ‖ 很快涨到 cap、方向基本随机"，这正是实测（cos 0.03、ΔJ≈0）。

**(d) 真正的主因：正则项的梯度比 J 大 27 倍**（`loss_terms_grad` 在 val 32 窗口上实测）

| 损失项 | 数值（batch mean） | **参数梯度范数** |
|---|---|---|
| `-J.mean()` | −1.475 | **0.369** |
| `0.010·L_act` | 0.0516 | **10.04** |
| `0.002·L_smooth` | 0.0116 | **2.78** |
| `0.001·L_anchor` | 0.00094 | 0.0026 |
| 合计 | — | 12.94 |

- `cos(g_act, g_total) = 0.999`、`cos(g_J, g_total) = 0.418`；
- 再叠加 `grad_clip=1.0`（把总梯度归一化）⇒ **每步只有 0.369/12.94 ≈ 2.8% 的步长在优化 J**，
  其余 97% 在压低动作预算与平滑项。

这完整解释了观测：`actRMS` 从 42 → 13.6 mrad（正则被优化）、`‖Δ‖` 顶在 cap、
`ΔJ≈0`、`cos(Δ, g_mean)=0.03`。**这是 v5 锁定损失权重（0.010/0.002）在本数据上的
梯度尺度失配，不是代码 bug**（损失*数值*看不出问题：0.05 vs 1.48，但*梯度*是 27×）。

**重要补充（it≈165 观测）：这是"早期相位"，不是永久状态。** `L_act` 是 **hinge**：
一旦 `actRMS < τ=20mrad` 它的梯度就**自动归零**。实测 it=160–165 时
`actRMS` 已降到 **9.7–16.3 mrad**、`L_act→0.09–0.74`、**总梯度范数从 10–27 跌到 0.13–0.73**
（此时 `grad_clip=1.0` 已经不再裁剪）⇒ 从这一步起，**J 的梯度（0.369）才真正主导更新**。
所以：前 ~150 步是"把动作预算压回 τ 以内"的相位（正则主导是设计使然），
**it≥165 之后才是"学 J"的相位**；能不能学到东西要看 200–1000 步的 held-out 曲线。

**当轮处理（不改规范锁定配置）**：主训练继续跑到早停/1000 步，作为"锁定口径的基线"；
同时建议下一轮做**明确标注的独立对照臂**（不混表）：

| 臂 | 改动 | 目的 |
|---|---|---|
| R1 | `λ_act 0.010 → 3e-4`、`λ_smooth 0.002 → 3e-5`（其余不变） | 让 ‖g_act‖≈0.1‖g_J‖，检验"是否只是权重尺度问题" |
| R2 | `grad-clip 1.0 → 10`（或按 J 分量单独归一化） | 验证 clip 是否也在放大正则项的影响 |
| R3 | R1 + `cache-tokens`（`data_mode=cached_pool`） | 去掉每步 VLA 噪声，量"梯度噪声"贡献 |

**结论**：v5 的**架构/梯度/契约都没问题**，但"锁定损失权重 + grad clip"在前 ~150 步
**梯度过配**；hinge 一旦满足就自关断，之后 J 才主导。实测 it=200 时 seed0 的
held-out **ΔJ 首次显著**（见下）。

### 6.2 held-out 的统计不确定度（决定"多少 ΔJ 算真信号"）

对 seed0 的 val 96 窗口逐窗口算 ΔJ：

| 量 | 值 |
|---|---|
| 逐窗口 ΔJ 均值 | **+0.001007** |
| 逐窗口 std | 0.00270 |
| **均值 SE**（n=96） | **0.000275** |
| 改善窗口占比 | **70.8%** |
| 中位数 | +0.000598 |

⇒ it=200 的 **ΔJ=+0.00101 是 3.7σ**（不是噪声）；但仍只有 oracle 上界（+0.0298）的 **3.4%**。
（训练 batch 的 J 逐窗口 std=0.237，B=256 的 batch-mean SE=0.0148 ⇒ **训练曲线本身看不出
0.001 级别的改善**，必须靠固定评测集的 96 窗口 × SE=0.0003。）

**监控**：

```bash
tail -f $WT/runs/temporal_ac_bptt_v5/main/seed0.log
tail -f $WT/runs/temporal_ac_bptt_v5/main/seed1.log
cat $WT/runs/temporal_ac_bptt_v5/main/seed0/val.jsonl    # 每 50 步
```

---

## 7. 与规范的偏差清单（全部显式记录，不隐藏）

| # | 偏差 | 原因 | 影响 |
|---|---|---|---|
| 1 | 输出目录在 **worktree** `runs/` 而非 `$COMIMIC_MAIN/runs/` | 用户要求"不要改 main" | 仅路径 |
| 2 | `--vla-batch 128`（规范未规定） | 显存/吞吐；**不影响采样分布** | 无 |
| 3 | VLA 全部参数冻结（含 action head） | `freeze_vlm=True` 只冻 backbone；不冻 head 会在 B=256 时保留 1.09B 参数的计算图（规范/历史都踩过） | 数值不变，只是不建图 |
| 4 | 训练侧关节序 = `standard`（规范 §1.3 写 HA_V31） | 四处代码证据（§4.1a） | 契约层换序；已在 ckpt/契约里声明 |
| 5 | hist default 用 `g1_default_joint_pos()`（规范 §1.3 的 `default_joint_pos` 是动作路径） | 实测（§4.1b） | 已写进 ckpt meta |
| 6 | FD 在 h=1e-2 时 3.2%（规范 ≤3%） | float32；h=3e-3 时 0.4% | 记录为脚注，不阻断 |
| 7 | WM 40 步用 dump 口径（1 clip）而非"每 clip 256 起点" | 只有 1 个 B dump | B 路线 NOT_RUN（保守） |
| 8 | 未做 `prefetch_vla` | 先保证正确性/可复现 | 训练慢 1.7× |
| 9 | `L_act` 用**逐样本** hinge 再取均值（规范 §7.1 的伪代码是**整批** RMS 再 hinge） | 逐样本是更严格的"动作预算"，避免个别窗口越界被批均值稀释 | 更保守，量级相同 |
| 10 | eval 固定用 `cap=0.08`（部署 cap），不用当前 warmup cap | 评测要对齐部署 | 训练前期 eval 的 ‖Δ‖ 略大于训练时的 cap |

---

## 8. 主仓库未被修改（审计）

```bash
cd $COMIMIC_MAIN && git status --short | grep -v '^?? runs/'
# 只剩**本次会话之前就存在**的改动：
#   M docs/HARNESS03_BPTT_VERDICT_20261107.md
#   ?? diff_vla_wbc/delta_token_temporal.py
#   ?? docs/{BPTT_SCHEMES_COMPARISON_20261107,TEMPORAL_AC_BPTT_V5_IMPLEMENTATION_PLAN,TEMPORAL_AC_BPTT_V5_MAINLINE_SPEC}.md
#   ?? scripts/train_distill_{temporal,v3}.py
ls $COMIMIC_MAIN/runs/temporal_ac_bptt_v5   # 不存在（已迁回 worktree）
```

本次新增/修改的**代码与文档全部**在 `comimic-wt03`；`runs/temporal_ac_bptt_v5` 也在 wt03。

---

## 9. 还没做 / 下一步

1. **等待两个 seed 跑完**（~4.6 h）：拿到 `test_once.json` 的 held-out `ΔJ` 与 29D rad 预算；
2. **M3 真 Isaac 配对**（会话 04 的门）：24–64 个独立 clip，同起点/同 Z/同 hist，报告
   `J_Isaac(Δ) − J_Isaac(Base)` + clip 级 bootstrap CI；
3. **M4 集成**：服务端 `delta_from_chunk_full_v5` + `state_hist_10` 字段
   ⇒ **BLOCKED_ON_SESSION01**（规范 §2.4）；回滚方案见 runtime contract §5；
4. **备选 A（`H_actor=12 < K=20`，β>0）**：`imagined_value_v5.py` 已就绪 + 6 个单测；
   Gate 通过后才允许 β>0；
5. **seed2**（第三张卡/排队）与 `--prefetch` 优化，视算力决定。
