# COMIMIC · A/B 交底【增补】—— 2026-11-08 复测：B 的 Test、真物理口径、以及三条需要撤回的结论

> ❌ **已被 `docs/A_B_SITUATION_ARCH_PLAN_20261108.md` 取代**：本文件里的**物理数字、单位、`L_same` 结论**已被三次复测推翻（2026-11-08）。仅供历史留档，不要再据此做决策。

> 配套文档：`docs/A_B_HANDOVER_FOR_REVIEW_20261108.md`（主文档）
> 本文件是**复测后的更正**：主文档里关于"oracle 上界""A 的物理改善""mrad 单位""λ_peak"的四处叙述必须按本文更新。
> 所有数字都是本轮在 `comimic-wt03` 上**现跑**的，产物在 `runs/three_track_v7/ab_diagnostics/`。

---

## 0. 三条一句话结论

1. **B 补上了 Test：`ΔJ_test = +0.02074`**（A 是 +0.02185）。⇒ **B 没失败，是个有效正收益 ckpt**；
   A/B 在 val 上差 0.0019、在 test 上差 **0.0011**，两者都在噪声带宽内 ⇒ **"A 优于 B" 不成立**。
2. **A 的物理"卖点"不成立**：全 96 个 val 窗口上，**A 的最坏单关节尖峰没有下降**（v5 2.027 → A **2.093** rad），
   t=14 只降了 **−42%**（不是主文档写的 −79%），尖峰从 t=14 **迁移到 t=5 / t=19**。
   主文档的 `0.425 rad @ t=14` 与 `全局 1.847 rad` **无法复现**（A best 与 A last 都试过）。
3. **两处单位/标签错误**：`diag_action_budget_v5.py` 输出的 "rad" 其实是 **WM 归一化动作域**（`bundle.action_scale = ones`）；
   训练日志里的 `*_mrad` 同样不是 rad。真物理值 = 该数 × `G1_ACTION_SCALE`（均值 **0.396**，即日志被放大 ~2.5×）。

---

## 1. 本轮新增/复跑的东西（都可复现）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
source scripts/harness03_env.sh
O=runs/three_track_v7/ab_diagnostics

# ① B 的独立 Test（空循环 + 收尾 test；1.1 min）
python3 scripts/train_delta_physical_bptt_v6.py --out $O/b_testonly \
  --on-the-fly 1 --eval-set runs/temporal_ac_bptt_v5/eval_set --updates 0 --batch 256 \
  --init-delta-checkpoint runs/three_track_v6/delta_retrain/seed0/checkpoint_actor_best.pt \
  --eval-every 0 --cond-check-every 0 --seed 0 --device cuda:0

# ② 真物理口径审计（本轮新写的脚本，修了单位并拆 Base/Δ 越限；每个 ckpt ~15 s）
python3 scripts/diag_delta_phys_v5.py \
  --ckpt runs/three_track_v6/delta_continue/seed0/checkpoint_actor_best.pt \
  --eval-set runs/temporal_ac_bptt_v5/eval_set --split val --device cuda:0 \
  --limits runs/three_track_v6/physical/limits_standard29.npz \
  --out $O/A_phys_rad_val.json
```

产物清单：

| 文件 | 内容 |
|---|---|
| `ab_diagnostics/b_testonly/test_once.json` | **B 的 test ΔJ = +0.02074**（J=1.59654，base=1.57580，n=96） |
| `ab_diagnostics/b_testonly.log` | 上面这次 run 的完整日志（含 `[AUDIT][spec] action_scale_mode=one`） |
| `ab_diagnostics/{v5,A,B}_phys_rad_val.json` | 三者的真物理审计（96 窗口、同一路径） |
| `ab_diagnostics/{v5,A,B}_phys_rad_val.log` | 人读版 |
| `ab_diagnostics/Alast_phys_rad_val.json` | A 的 `checkpoint_actor_last.pt`（it=300）对照，用于验证 0.425 的来源 |
| `ab_diagnostics/A_phys_val.json`、`B_phys_val.json` | 旧脚本（归一化域）的对照，用于确认两者是同一事件的不同单位 |

---

## 2. 更正一：B 的 Test 结果（主文档 §5.2 的"缺（NOT_RUN）"已补齐）

| 指标 | A（best@225） | **B（best@900）** | 差 |
|---|---|---|---|
| val ΔJ | +0.02962 | +0.02774 | 0.00188 |
| **test ΔJ** | **+0.02185** | **+0.02074** | **0.00111** |
| ‖Δ‖ | 0.0781 | 0.0774 | — |

- 两者 test 都显著为正（base test J=1.57580），**B 不是"训崩了"或"白训"**；
- val 与 test 的排序一致（A 略高），但差值 **0.0011–0.0019** 与训练曲线抖动（≈0.003）同量级，
  且都是 **n=1 seed** ⇒ **只能写"B 与 A 相当"，不能写"A 优于 B"**。

> 这条正好回应 reviewer 的判断："A 有正的独立 test 增益，B 在 val 上接近 A，但物理与 test 未知"。
> 现在 test 已知；**物理也已知，而且方向不利（见 §4）**。

---

## 3. 更正二：`diag_action_budget_v5.py` 的 "rad" 不是 rad（单位链号）

### 3.1 事实

`diff_vla_wbc/isaac_joint.py:296-303`：

```
_scale_mode = os.environ.get("COMIMIC_ACTION_SCALE", "one")   # 默认 one
self.action_scale = ones(29)   if _scale_mode != "spec"
```

本轮所有 run 的审计行都自证：`[AUDIT][spec] ... action_scale_mode=one`。

而 `diag_action_budget_v5.py` 用 `q_target_from_action(bundle.default_joint_pos, bundle.action_scale, a)`：
`ones` ⇒ `q_target = default + a` ⇒ **它算出来的 `e` 是归一化动作差，不是关节弧度**。
真物理换算是部署口径（`sonic_physical_action.py` 文件头）：

```
q_raw  = q_default_standard + G1_ACTION_SCALE ⊙ a_sonic
e_phys = G1_ACTION_SCALE ⊙ (a_Δ − a_base)
```

`G1_ACTION_SCALE` 实测 **min 0.0745 / mean 0.396 / max 0.5475** ⇒ 旧脚本的 "rad" 被**放大约 2.5 倍**。

### 3.2 训练日志的 `*_mrad` 同样不是 mrad

`action_budget_stats()` 只是把 `masked_rms(e)` 乘了 1e3，而 `e` 来自同一条 `bundle` 链路
（`eval_split` 里 `q_target_from_action(..., bundle.action_scale, ...)`）⇒
主文档 §5.5 表里的 `rms 100.6 mrad / peak 5017 mrad / same_obs 33.3 mrad` 全部应读作
**"归一化域数值 ×1e-3"**，乘 0.396 才是真正的 mrad。

**同一批数据在两个脚本里的对照**（A best）：

| | 旧脚本（归一化域） | 新脚本 `diag_delta_phys_v5.py` |
|---|---|---|
| rms | 130.8（标 mrad） | **0.0518 rad = 51.8 mrad** |
| p99 | 480（标 mrad） | **0.189 rad = 189 mrad** |
| global peak | 5245（标 mrad） | **2.093 rad** |
| t=14 | 2691（标 mrad） | **1.239 rad** |

比值 ≈ 0.396 = `G1_ACTION_SCALE` 均值，**完全对上**。⇒ 单位链已闭环，不是随机误差。

> 顺带：主文档 §5.5 里 A 的 `rms 100.6 mrad` 与本轮的 `130.8` 还差 30%，
> 这是**两个评测路径**（训练器 `eval_split` 用缓存 base 动作 vs 诊断脚本现算 base rollout）的差，
> 不是单位问题。**跨脚本比较前必须先对齐路径**，否则不同 ckpt 之间也会引入这种 30% 的假差。

---

## 4. 更正三：A 的物理改善不成立（这是最严重的一条）

### 4.1 三者在 **96 个 val 窗口、同一脚本、真物理口径** 下的对比

| 指标（rad） | v5 best | **A best** | **B best** |
|---|---|---|---|
| rms | 0.0608 | **0.0524** | 0.0554 |
| p99 | 0.2239 | **0.1878** | 0.2108 |
| **全局单关节峰值** | 2.0265 | **2.0928（更差）** | 2.0641 |
| **t=14 单关节峰值** | 2.0265 | **1.1801（−42%）** | 1.2324 |
| t=5 峰值 | 1.9606 | **2.0928（新峰）** | 2.0641 |
| t=19 峰值 | 1.5635 | 1.8394 | 1.6409 |

逐 tick（rad）：

| t | 0 | 1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | **14** | 15 | 16 | 17 | 18 | **19** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v5 | .097 | .121 | .432 | .400 | .562 | **1.961** | 1.283 | 1.214 | 1.120 | 1.075 | 1.187 | 1.114 | 1.442 | 1.338 | **2.026** | 1.596 | 1.919 | 1.413 | 1.252 | 1.564 |
| A | .080 | .170 | .349 | .374 | .630 | **2.093** | 1.718 | .718 | 1.507 | 1.109 | 1.169 | .817 | 1.496 | 1.361 | **1.180** | .854 | .932 | .841 | 1.117 | 1.839 |
| B | .092 | .171 | .343 | .372 | .640 | **2.064** | 1.690 | .959 | 1.489 | 1.076 | .907 | 1.226 | 1.598 | 1.276 | **1.232** | 1.168 | .883 | .967 | 1.046 | 1.641 |

### 4.2 与主文档的冲突（必须撤回的三行）

| 主文档（24 窗口） | 本轮（96 窗口，同一 ckpt） | 判定 |
|---|---|---|
| Δ 全局最大：2.065 → **1.847** rad（−11%） | 2.027 → **2.093** rad（**+3%，略差**） | ❌ **撤回** |
| Δ @ t=14：2.065 → **0.425** rad（**−79%**） | 2.027 → **1.180** rad（**−42%**） | ❌ **撤回**（幅度、方向都不对） |
| 限位违反率：2.74% → **2.90%**（持平/略差） | base **2.701%** → A **2.640%**（**略好**） | ❌ **撤回** |

**为什么会对不上？** 我验证了三种可能：
1. 不是 ckpt 混用：A `last@300` 的 t=14 是 **1.239 rad**、全局 2.093 rad，与 best 几乎一致；
2. 不是单位：v5 的 2.065（旧）与本轮 2.027（新）只差 2%，说明**旧脚本的量级是对的**；
3. ⇒ 旧结论只能来自**一个更小的、恰好避开了 A 的尖峰窗口的子集（24 个）**。
   也就是说：**"−79%" 是窗口抽样挑出来的**，不是策略性质。

### 4.3 但物理约束**确实**有一个真实的正作用（这条要保留）

把限位违反按 reviewer 建议拆开后（`v = (relu(qmin−q)+relu(q−qmax))/s`，96 窗口）：

| | base 违反率 | Δ 修正后 | **Δ 新增违反** | **Δ 修掉的违反** | 新增/修掉 |
|---|---|---|---|---|---|
| v5 | 2.701% | 2.647% | 1.279% | 1.643% | 0.473 |
| **A** | 2.701% | **2.640%** | **1.223%** | **1.676%** | **0.453** |
| **B** | 2.701% | 2.696% | 1.254% | 1.697% | 0.464 |

**读法**（这也是 reviewer 想要的"拆开"）：
- Base 自己就有 2.70% 的越限，**Δ 无法全部修掉**（只能用 64 维 token 残差间接影响 29 维动作）；
- Δ **修掉的越限（1.68%）多于它新增的（1.22%）**，比值 1.37× ⇒ **净效果是正的**；
- 三者差别很小，A 略好于 B（2.640% vs 2.696%），与 ΔJ 的排序一致。

⇒ **物理约束的合法证据应该换成"rms/p99 + 净越限"，而不是"峰值尖峰"。**

---

## 5. 更正四：λ_peak 不是"形同虚设"，但代价是 **L_act 的单位错了**

reviewer 的算术是对的，我先确认：

```
‖∇_θ J‖      = 0.2018
‖∇_θ L_peak‖ = 72.05   ⇒ λ_peak=8.4e-4  ⇒ 加权 0.0605  ≈ 30% × ‖∇J‖
‖∇_θ L_lim‖  = 1.906   ⇒ λ_lim=0.0529   ⇒ 加权 0.1008  ≈ 50% × ‖∇J‖
```

⇒ 主文档 §7.1 的"P1-3：物理约束可能名存实亡"**撤回**：标定点上它是 30% 量级的有效梯度。

### 5.1 但我在核这条时发现了更硬的 bug：**信任域被重复计算，且有一个口径是错的**

`train_delta_physical_bptt_v6.py` 同时加了两项：

```python
# ① L_act：e 是 bundle 域（action_scale=ones）
L_act, _ = physical_action_trust(e_loss, M_surv, tau=a.tau_action)      # λ_act=0.010
# ② L_same：e_phys = G1_ACTION_SCALE ⊙ e
_L_same_p, _ = l_same(_e_phys, float(a.tau_action), _M)                # λ_same=0.010
L_phys += λ_lim*L_lim + λ_peak*L_peak + λ_same*_L_same_p + ...
```

- `e_loss` 与 `_e_phys` 是**同一个量**，只差 per-joint 缩放 `G1_ACTION_SCALE`；
- 两者 **τ 都是 0.02**，但单位不同 ⇒ 触发阈值不同：
  - `L_act` 触发于 `RMS(e_norm) > 0.02` ⇒ 物理上 **≈ 7.9 mrad**；
  - `L_same` 触发于 `RMS(e_phys) > 0.02 rad` ⇒ **20 mrad**。
- 实测（标定产物）：`rms_same = 0.01305 rad = 13.05 mrad`
  ⇒ **13.05 < 20 ⇒ `L_same` 恒为 0（hinge 未越过）**；而 **13.05 > 7.9 ⇒ `L_act` 一直在起作用**。

**结论（与 reviewer 的猜测方向一致，但机理不同）**：
1. `L_same` 不是"因为标定时梯度为 0 才没激活"，而是**它的阈值就比实际误差宽 1.5 倍，结构上永远打不着**；
   主文档 P1-4 的措辞（"λ 保留默认值所以未激活"）**不准确**，应改为"阈值未越过"；
2. `L_act` 的 `tau=0.02` **被误当成 rad 用**，真实信任半径是 **7.9 mrad 而不是 20 mrad** ⇒
   **训练实际施加的信任域比设计意图紧 2.5 倍**；
3. 因此"物理约束到底有没有生效"这件事，**必须先把这两项合并成一个、且统一到 rad**，再谈校准 λ。
   reviewer 说的"排查是否重复计算同一种约束"——**已经确认：是重复的**。

---

## 6. 对 reviewer 十条意见的逐条回应

| reviewer 的判断 | 我的核对结果 | 处理 |
|---|---|---|
| ① oracle ≠ 上界，不能推出"只剩 0.00018" | **同意**。那是**一步梯度方向 + cap 范数**的候选值，不是非线性 20 步最优 | 主文档 §5.4 措辞已修正为"局部梯度候选" |
| ② A/B 谁更好未证明 | **同意**，并已补上 **B 的 test +0.02074** | 主文档 §5.2 已更新 |
| ③ cap 径向饱和但切向未死（0.41） | **同意**。`sech²(2.4)=0.032`、`tanh(2.4)/2.4=0.410` | 主文档已补这条区分 |
| ④ `L_peak` 预算自指 | **同意，代码确认**（`b_j = 3·mean_t(\|e\|)`，整体缩放不变） | 待改：固定预算 |
| ⑤ `L_same` 未激活 / `L_rate` 缺失 | **部分不同意**：`L_same` 是**阈值结构上打不着**，且**与 `L_act` 重复**；`L_rate` 确实 BLOCKED | 见 §5.1 |
| ⑥ λ 小 ≠ 无效 | **同意**，主文档该条撤回 | 见 §5 |
| ⑦ B 先补 Test/物理审计 | **已做完**（1.1 min + 15 s） | 见 §2、§4 |
| ⑧ 闭环只有 sparse success，要稠密日志 | **同意**，未做 | 待办 |
| ⑨ A/B 需 2×2 重跑 | **同意**，且发现 **B 的 lr 与 v5 原配方相同**（主文档 §5.2b） | 待办：补两个空格子 |
| ⑩ 先修损失/评估，再动 cap/网络/WM | **同意**，且本轮的测量进一步支持：**改峰值指标比改网络更必要** | 见 §7 |

**reviewer 没有意识到、但本轮实测出来的两件事**：
1. 他推荐的 `diag_action_budget_v5.py` **单位是错的**（标 "rad" 实为归一化域）——必须用新脚本；
2. 他用来论证"A 物理改善"的前提（A 把 t=14 从 2.07 压到 0.43）**不成立**，
   所以"B 的物理表现是否比 A 差"这个问题的答案很可能是：**两者一样，A 只是把尖峰挪到了 t=5/t=19**。

---

## 7. 修正后的最小执行顺序（替换主文档 §8）

| 顺序 | 事项 | 成本 | 为什么现在做 |
|---|---|---|---|
| **1** | **把 `L_act` 与 `L_same` 合并**为一个 rad 口径的信任项（删掉归一化域那一项），τ 明确用 20 mrad；`L_peak` 换固定预算 | 改 2 个函数 + 1 次标定，~1 h | 这是"物理约束到底有没有用"的前提；不修则后面所有 λ 讨论都是空中楼阁 |
| **2** | **重跑一次标定**，记录 reviewer 要的那 12 个诊断量（加权梯度、cos、激活率、before/after clip） | ~10 min | 现在的标定只记了范数，无法判断方向 |
| **3** | **A/B 各 100 步修复 Pilot**（从现有 best 出发，`fixed-peak` + 合并后的 trust，其余不变） | 2 × 35 min | 不需要重训，也不动 cap |
| **4** | **条件化门禁加 reward 对照**（`G_cond = J(Z+Δ) − J(Z+Δ_shuffled)`，只在同任务同阶段内置换） | 代码 ~40 min | 现在的 cos 门禁只证明"不是常量" |
| **5** | **闭环逐 tick 落盘**（reviewer 列的那张字段表），下一次闭环才能定位失败 | 代码 ~1 h | A 的 9/18 目前无法归因 |
| **6** | B 的 18 集闭环（协议照 A，另加 base 臂） | ~2.5 h ×2 | 现在有了 test，闭环是最后一块拼图 |
| **7** | 2×2 重跑 + 多 seed | 4 组 × 1.3–5 h × seeds | 最后再做，避免在错误的损失上烧算力 |

> 注意顺序变了：reviewer 把"补 B 的 Test/物理"放第 0 位，这已经做完；
> 我把**合并信任项 + 固定 peak 预算**提到第 1 位，因为它会改变后面所有 pilot 的结论。

---

## 8. 逐项复核清单（本轮实际跑过的东西）

| 项 | 命令/文件 | 状态 |
|---|---|---|
| B 的 test ΔJ | `ab_diagnostics/b_testonly/test_once.json` | **PASS +0.02074** |
| A/B/v5 真物理审计（96 窗口） | `scripts/diag_delta_phys_v5.py` + 3 个 json | **PASS** |
| A last@300 对照 | `ab_diagnostics/Alast_phys_rad_val.json` | **PASS**（排除 ckpt 混用） |
| Base/Δ 越限拆分 | 同上 json 的 `limit` 段 | **PASS** |
| 单位链闭环（0.396×） | 旧脚本 vs 新脚本对照 | **PASS** |
| `L_act`/`L_same` 重复与阈值 | 代码 + `rms_same=13.05 mrad` | **确认重复** |
| 旧「−79%」复现 | 96 窗口 / 24 窗口 / best / last | **不可复现 ⇒ 撤回** |
| 闭环 / 2×2 / 多 seed | — | **NOT_RUN** |
