# Δtoken 蒸馏 Teacher 修复：结论与证据（会话 02）

> 写给后续接手的人 / 外部评审（包括别的模型）：**这一轮到底证明了什么、没证明什么**。
> 写于 2026-11-07。代码 commit **`1fa87b0`**（Teacher 修复）+ **`9c58c0d`**（DDP 指标聚合 + 本文档），
> 分支 `harness/02-teacher`、worktree `comimic-wt02`，基线 `a4bd1ff`。
> 完整数据、逐项数字、复现脚本在 `runs/harness_02_teacher/`（`runs/` 被 `.gitignore` 排除，不在 commit 里）。
> 本文所有"结论"都带 `PASS / FAIL / NOT_RUN` 标记；**推测**统一放在 §7，不写成结果。

---

## 0. 三句话结论

1. **旧的蒸馏 Teacher 确实写错了**（只拿第 1 个 token 重复 20 步、回报未折扣、异常被静默吞掉），
   但它**不是**"学生 cos 只有 0.10"的主因：在 512 个生产窗口上，新旧 Teacher 的梯度余弦
   均值 0.672 / 中位 0.843。真正致命的是它**在 1/4 的窗口上基本错**（14.8% 的窗口 cos<0.2）。
2. **修好的 Teacher 在数学/数值上是正确的**：共享 Δ 的两种独立求导算法（对 ε 求导 vs 逐步求导再求和）
   在 H=1/2/5/20、Δ=0/0.01/0.08、真实转移与生产 VLA 窗口共 83 组配置上**逐元素相等**
   （cos=1.0、max|Δ|=0、rel_L2=0、符号一致 100%）。有限差分 H≤5 一致到 <1%，H=20 最好 2.7%~3.6%
   （已定位为 20 步世界模型的 FP32/非线性条件数，不是代码错）。
3. **学生"能学"但天花板低**：按 clip 分组的 held-out cos（n=256）H=20 最好 **0.259**、
   H=5 最好 **0.331**；而"不学，直接输出平均方向"的常数解就有 **0.156~0.19**。
   训练集能背到 0.997~1.000（容量不是问题），当年在线训练**只做了 75 次 Adam 更新**才是次因。

---

## 1. cos≈0.10 到底怪谁（按贡献排序）

| # | 原因 | 证据（都在 `runs/harness_02_teacher/`） | 量级 |
|---|---|---|---|
| 1 | **目标本身可解释度低**：H=20 的 reward-only 梯度只有 ~26% 能被"部署可用输入"解释 | `distill_learning_curve.csv`：held-out 常数 0.156 / ridge 0.259 / MLP 0.218 | **主因** |
| 2 | **在线训练量严重不足**：150 micro-step、accum=2、2 卡 ⇒ **75 次 Adam 更新**，每次 32 个标签，共 ~2400 个、基本不重复 | `runs/gt7fix/args.json` + `train_metrics.jsonl`；离线 n=256 能 train cos 0.998 | 次因，但可修 |
| 3 | **标签抖动**：`FREEZE_VLA=1` 但每步重新随机采样 token ⇒ 同一窗口两次采样标签 cos 只有 **0.693/0.708** | `epsgrad/train_metrics.jsonl` 的 `teacher/cos_resample_label` | 与 #4 同量级 |
| 4 | **旧 Teacher 口径错**：token0 重复 H 步 ⇒ 1/4 窗口标签方向不可用 | `teacher_old_vs_new.csv`：mean 0.672 / p10 0.025；`frac(cos<0.2)=14.8%`、`<0=9.4%` | 不是主因，但确实在污染标签 |
| 5 | 旧 Teacher 的"未折扣 sum / 无 survival" | `cos(new_disc,new_undisc)=0.9995` | **可忽略** |

**决定性反证**：把旧 Teacher 的标签当"完美学生"的目标训练，再拿它对**正确目标**评分，
held-out cos = **0.285**（n=256 ridge）；直接学正确标签的学生是 0.259。
⇒ **就算当年 Teacher 完全正确、学生也学满，也只能到 0.26 左右**，这是任务的固有难度，不是实现的锅。

---

## 2. 旧 Teacher 错在哪类窗口（机制）

旧实现等价于：`token'_t = token_0 + Δ`（20 步都假装还在执行 chunk 的第一个 token），
而部署/主训练是 `token'_t = token_t + Δ`（每个 token 各自加同一个 Δ）。

以 chunk 内 token 的时变性 `mean_t‖token_t − token_0‖` 分桶（512 个生产窗口）：

| 桶 | `cos(g_old, g_new)` | `frac(cos<0.2)` |
|---|---|---|
| 低时变（最低 25%） | **0.951** | 0% |
| 高时变（最高 25%） | **0.439** | 29.7% |

相关系数 **−0.446**。⇒ 旧 Teacher 在"这段动作变化小"时几乎正确，在"chunk 内要换动作/换相位"时失效。
这正是把 `token_0` 重复 20 步的必然结果，也解释了为什么平均 cos 看起来"还行"（0.67）但学生学不好。

---

## 3. 修好之后：新 Teacher 的定义与验证

### 3.1 定义（`diff_vla_wbc/teacher_probe.py`，训练与测试共用同一实现）

```
token_t  : 同一次冻结 VLA 前向输出的完整 chunk（T=40；生产 H=20，20 步各用 token_0..19，无 repeat）
ε        : (B,1,64) 共享可微变量
token'_t = token_t + ε
a_t      = WBC(token'_t, obs_t)          # 冻结 SONIC decoder，对 token 可微
s_{t+1}  = WM(s_t, a_t, ξ)               # 冻结世界模型
J        = Σ_t γ^t · survival_t · r_t    # = discounted_return(...)，与 train_step 同一个定义
g_new    = ∂J/∂ε
label    = η · unit(g_new),  η=0.08
L_teacher= Σ_w v_w‖Δ_w − label_w‖² / Σ_w v_w    # v_w = 有限且 ‖g_w‖>0
```

旧实现保留为 `legacy_token0_repeat_v1`，只能 `GRAD_TARGET_TEACHER=legacy` 显式启用；
复跑对照：31/32 窗口 `max|Δg|=0`、cos=1.0（唯一差异窗口是 burn-in RNG 分支不同导致 start_state 不同）。

### 3.2 一致性（最硬的结论）

| 检查 | 结果 | 文件 |
|---|---|---|
| `g_A`（对共享 ε 求导）vs `g_B`（逐步求导再沿 T 求和），真实转移窗口，H=1/2/5/20 × Δ=0/0.01/0.08 × 2 seed = 24 组 | **cos=1.0（1 例 0.99999994）、max|Δ|=0、rel_L2=0、符号 1.0** | `gradient_consistency.csv` |
| 同上，生产 VLA 窗口（13 个 batch）= 59 组 | **全部 cos=1.0、max|Δ|=0** | `gradient_consistency_prod.csv` |
| 有限差分方向导数（第三方，不含 autograd） | H=1/2/5：0.28%~0.92%；H=20：2.7%~3.6% | `gradient_consistency_fd.csv` |

H=20 的 3.6% 残差**已定位**（不是忽略）：rollout 完全确定性（同点重复算 J，ΔJ=0）；
动作空间 FD（绕开 ONNX 执行器）同样只有 ~4%；分项拆解显示残差集中在 20 步复合的
`exp` 奖励核与误差通道（|∂/∂ε|≈226 vs 奖励项 ≈0.5）。结论：这是 **20 步 WM 的 FP32 条件数**，
不是口径/实现错误。**含义**：如果要用 H=20 做大规模蒸馏，标签本身就带这个量级的数值噪声。

---

## 4. 学生的可学性（held-out，按 clip 分组，不是随机打散）

数据：512 个生产窗口（16 步 × 32 窗口，冻结 VLA 真实 token chunk、真实 burn-in/DR 起点），
326 个 clip，train/val/test = 304/102/106；特征 = `token_0(64) ⊕ state64(64)`（部署可用，不加任何线上拿不到的量）；
标准化只用训练集。

| target | n_train | 模型 | train cos | val cos | test cos |
|---|---|---|---|---|---|
| H=20 | 64 | 最优常向量 | 0.230 | 0.135 | 0.130 |
| H=20 | 128 | DeltaTokenNet（生产超参） | 0.999 | 0.183 | 0.209 |
| H=20 | 256 | **ridge(state64)** | 0.534 | **0.285** | **0.259** |
| H=20 | 256 | mlp2x256 | 0.998 | 0.162 | 0.218 |
| H=20 | 256 | DeltaTokenNet（生产超参） | 0.998 | 0.170 | 0.218 |
| H=5 | 256 | **ridge(token+state)** | 0.722 | **0.307** | **0.331** |
| H=5 | 256 | mlp2x256 | 0.998 | 0.277 | 0.333 |

判读：

- **能过拟合**：128~256 窗口，train cos 0.997~1.000 ⇒ 容量够、优化能收敛；
- **held-out 上限 ~0.26（H=20）/ ~0.33（H=5）**，且 **1.25M 大网络在 256 样本上不如 ridge**
  ⇒ 是样本/信噪比问题，不是网络表达力问题；
- **常数解 0.16~0.19** ⇒ 任何"看起来有 cos"的结果必须先减掉这个基线；
- H=5 比 H=20 好学，但**"好学"≠"真机回报更好"**，这需要闭环，本会话 **NOT_RUN**。

---

## 5. reward-only 梯度 vs 完整训练梯度（谁在放大/抵消）

`J1_EPS_GRAD=1` 把施加的 Δ 换成自由变量 ε，对完整训练 loss 逐项求 `dL/dε`
（配置与 BPTT 主线一致：λ_r=2.8、act 0.15/0.3、window_norm、trust 0.05/0.2、critic、gt 0.2）：

| 量 | step 0 | step 1 |
|---|---|---|
| `cos(-dL_total/dε, g_reward)` | **0.9884** | **0.9795** |
| `‖-dL_total/dε‖` / `‖-dL_reward/dε‖` | 0.524 / 0.510 | 0.434 / 0.417 |
| `‖-dL_trust/dε‖`（cos with reward） | 0.050（+0.165） | 0.071（+0.091） |
| `‖-dL_grad_target/dε‖` | 0.0066 | 0.0066 |
| `‖-dL_act/dε‖`（hinge 未激活） | 0.0000 | 0.0000 |
| `composition_residual`（逐项和 vs total） | **0.0** | **0.0** |

⇒ 在这套配置里**没有辅助项在抵消 reward 梯度**；trust 项同向、量级 ~10%；
真正把 g_full 拉离纯奖励的是 critic bootstrap（范数差 ~10%，方向不变）。
符号约定正确：`g_full := -dL/dε` 与奖励上升方向同号。

---

## 6. 顺带修掉/确认的工程问题

| 问题 | 状态 |
|---|---|
| Teacher 异常被 `except: return None` 静默吞掉、训练记成功 | **已修**：默认 strict 直接抛错（smoke 里真实触发过一次并中止，见 `evidence_strict_abort.log`）；`GRAD_TARGET_STRICT=0` 才允许跳过并计数 |
| `grad_cos` 是训练子批指标、不是 held-out | **已澄清**：新增 `grad_cos_scope=0` 标记；held-out 结果见 §4 |
| DDP 下指标没有 all_reduce（只有 rank0 落盘） | **已修**：Teacher 指标用一次 4 元 all_reduce 汇总；2 卡 smoke 里两 rank 的 `grad_cos` 完全相同（−0.005699）、`grad_n_global=16`、`grad_world=2`，而**本 rank** 的 `grad_norm_mean` 不同（1.093 vs 1.228）——正是全局聚合应有的行为 |
| ckpt 元数据缺 `feat_dim`/输入特征版本/Teacher 口径 | **已修**：`delta_token_meta` schema v2（17 字段：feat_dim/token_dim/hidden/depth/cap/input_features(+version)/apply/norm_cap/teacher_objective/teacher_reward/horizon/eta/n/grad_target_weight/trained_steps/git_commit） |
| 历史 ckpt 怎么兼容 | 训练侧提供 `scripts/harness02_check_delta_ckpt.py`：从 `net.0.weight` 反推 `feat_dim=64`，旧 ckpt 标 `legacy_inferred`、新 ckpt `ok`。**部署侧（会话 01）仍需按 schema 分支加载** |
| 旧 Teacher 语义不能被静默改变 | **已保证**：legacy 路径复跑与旧口径 `g_old` 逐位相同（31/32 窗口 max|Δg|=0） |

---

## 7. 推测 / 未验证（明确不写成结果）

- "在线训练不足" 是从 **75 次更新 vs 离线 300 epoch** 推断的，**没有做"在线同预算 vs 离线"的对照实验**（NOT_RUN）。
- "固定 VLA 噪声能提高标签质量" 是合理推断，**没有验证过**（NOT_RUN）。
- H=5 更可学 ⇒ **不**代表闭环回报更好（NOT_RUN）。
- "加 `state−ref` 特征能提高上限" 是评审文档里的假设，本会话**没有测**（NOT_RUN）。
- H=20 的 FD 3.6% 归因于 FP32 条件数是**间接证据**（确定性 + 动作空间对照 + 分项拆解），
  没有做 FP64 重算（WM/ONNX 都是 fp32，成本高）。

---

## 8. 未做（NOT_RUN）清单

| 项 | 原因 |
|---|---|
| 闭环成功率（新/旧 Teacher 各训一个学生） | 任务要求避免大规模训练；GPU 与另一会话共享 |
| 1k / 3k / 10k 学习曲线 | 资源不足；已完成的 64/128/256 足以判断"可学但上限 0.26/0.33" |
| 生产规模（150 步）的 g_full 逐项分解 | 当前来自 2 步诊断跑 |
| 生产规模的重采样标签噪声统计 | 当前 2 步 × 24 窗口 |
| 在线同预算 vs 离线对照 | 需要改训练循环/跑两组，未做 |

---

## 9. 下一步建议（按性价比排序）

1. **先决定 horizon**：如果接受"滚动重规划"的用法，用 **H=5 标签**（可学性 0.33 vs 0.26），
   并在闭环里验证；不要用 H=20 的 cos 去证明 H=20 更好。
2. **把标签生成从"在线每步现算"改成"离线缓存 + 多 epoch"**：`runs/harness_02_teacher/dump_h20/`
   已经是这个形态（512 窗口）。要做 1k~10k 只需重跑 `harness02_run_teacher_dump.sh` 加步数。
3. **压标签抖动**：固定 VLA 采样噪声，或同一窗口采样 k 次取平均标签（先量 k=1/2/4 的 cos 收益）。
4. **加输入特征前先做消融**：`state−ref`、动作 `a_t`（部署端可得）——本会话已验证
   `token_0` 单独几乎无增益（ridge 0.19~0.20），`state64` 才是主力。
5. **学生用线性/小模型 + 强正则起步**：256 样本下 ridge 反超 1.25M MLP；样本上千后再放大网络。

---

## 10. 产物索引（`runs/harness_02_teacher/`）

| 文件 | 内容 |
|---|---|
| `report.md` | 正式报告（12 节，逐项 PASS/FAIL/NOT_RUN） |
| `metrics.json` | 所有关键数值（机器可读）+ verdicts |
| `gradient_consistency.csv` / `_prod.csv` / `_fd.csv` | g_A vs g_B（真实转移 / 生产窗口）+ FD h 曲线 |
| `teacher_old_vs_new.csv` | 新旧 Teacher 余弦分布（512 窗口） |
| `old_label_transfer.csv` | "完美学旧标签"对正确目标的得分 |
| `distill_learning_curve.csv` | 常数/ridge/MLP/DeltaTokenNet × n × H |
| `fd_diagnose.json` / `fd_action_probe.json` | H=20 FD 残差定位 |
| `eps_grad.jsonl` (+ `*.stepNNNN.pt`) | g_full 逐项分解 |
| `legacy_equivalence.json` | legacy Teacher 语义未变的证据 |
| `delta_token_meta_check.json` | ckpt 元数据（新 ok / 旧 legacy_inferred） |
| `dump_h20/` | 512 个生产窗口缓存（token chunk/state64/g/g_old/g_H5/J/refs/history）+ 新 schema ckpt |
| `test.log` | 玩具解析测试 + 全部自检（当前全 PASS） |
| `reproduce.sh` | 一键复现（`SKIP_GPU=0` 才跑 GPU 步骤） |

复现（在 worktree 里）：

```bash
cd <worktree> && COMIMIC=$PWD && source scripts/deps_env.sh
python3 tests/test_teacher_shared_delta.py                 # 秒级
python3 scripts/harness02_gradient_consistency.py --source recorded ...   # ~10 min
RUN=dump_h20 STEPS=16 BATCH=32 GPU=1 J1_TEACHER_DUMP=$PWD/runs/harness_02_teacher/dump_h20 \
  GRAD_TARGET_COMPARE_OLD=1 J1_TEACHER_EXTRA_H=5 bash scripts/harness02_run_teacher_dump.sh
python3 scripts/harness02_distill_learnability.py --dump runs/harness_02_teacher/dump_h20
```
