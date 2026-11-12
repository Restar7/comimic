# B-WM6 重要更正：P7 真物理配对用错了臂（结论大幅好转）

> 生成：2026-11-09 13:10 CST｜容器 2 / `comimic-b-wm6`
> **本文更正 `B_WM6_FINAL_REVIEW_20260909_v3.md` 与所有此前文档中的真物理数字。**
> 产物：`recovery_h10_critic/phys/pair_corrected_arms.json`、`phys/p7_direction_audit.json`（已重算）
> 复现：`python3 scripts/b_wm6/rescore_p7_arms.py`

---

## 0. 一句话

> **`score_pair.py` / `p7_direction_audit.py` 在评 Δ 臂时用了 `roll["states_plus"]` ——
> 那是 **base 臂的轨迹**，只把动作换成 Δ 的。而 `bundle.reward(S, A, ...)` **同时依赖状态和动作**，
> 所以之前所有"真物理"数字都是「**base 状态 + Δ 动作**」的混合量。**
> **用正确的 `states_minus` 重算后：每段漂移降低 0.69%（J10）/ 1.21%（J20），
> 比同范数随机 Δ 好 0.42 个百分点、85.4% 窗口胜出；WM 与真实的相关性 0.34–0.42（不是 0.09），
> 幅度只乐观 2×（不是 24×）。**

---

## 1. Bug 本身

### 1.1 错在哪

```python
# 旧（错）
real_base = score_arm(..., roll_fix_base["states_plus"],  job["acts_plus"],  ...)
real_d    = score_arm(..., roll_fix_b["states_plus"],     job["acts_minus"], ...)
#                          ^^^^^^^^^^^^ base 臂的轨迹        ^^^^^^^^^^^^^^^ Δ 的动作
```

滚动的输出里 **两臂都有**：`states_plus` / `states_minus`。
`run_isaac_roll.sh` 每次都跑臂序列，但 `states_plus` 对应的是 **plus 臂（base 动作）**。
三个 job（base / trained / random）的 `states_plus` **逐位相同**（实测 max diff = **0.0**），
而 `states_minus` 互不相同（base 6.84 / trained 8.64 / random 23.3 的臂间差）。

### 1.2 为什么"混合量"不是零

`bundle.reward(S[:,t+1], A[:,t], a_prev, ref)` 里除了跟踪项，还有**动作相关项**（平滑/限位/峰值）。
所以"base 状态 + Δ 动作"会得到一个小而非零的 ΔJ —— 那正好是之前报的 +0.00049。
**它衡量的是"动作项的变化"，不是"跟踪漂移的改善"。**

### 1.3 正确做法

| 臂 | 状态 | 动作 |
|---|---|---|
| base（第一臂） | `states_plus` | `acts_plus` |
| **base'（第二臂）** | `states_minus` | `acts_minus` |
| **Δ（第二臂）** | `states_minus` | `acts_minus` |
| **random（第二臂）** | `states_minus` | `acts_minus` |

`ΔJ = J(Δ.minus) − J(base.minus)`：两者都是**同一进程内的第二臂** ⇒ 进程内偏移**抵消**。
另报 `bias = J(base.minus) − J(base.plus)` 作为该偏移的实测值。

---

## 2. 更正后的数字（48 窗 × H=20，26 clips / 3 任务）

### 2.1 每段漂移的降低（用户口径：降低每段小漂移）

| 对比 | ΔJ10 | 95% CI (iid) | 95% CI (**cluster**) | 胜率 |
|---|---|---|---|---|
| 进程内 A/A 偏移（同动作两臂） | +0.001431 | [+0.00033, +0.00277] | [+0.00032, +0.00272] | 0.312 |
| **Δ vs base（同位置，偏移抵消）** | **+0.003813** | **[+0.00250, +0.00514]** | **[+0.00255, +0.00530]** | **0.750** |
| 随机 Δ vs base（同范数 0.0684） | **−0.000373** | [−0.00101, +0.00029] | [−0.00099, +0.00034] | 0.333 |
| **训练 Δ vs 随机 Δ** | **+0.004185** | **[+0.00282, +0.00559]** | **[+0.00292, +0.00569]** | **0.854** |
| Δ vs base（J20） | **+0.006720** | [+0.00368, +0.00989] | [+0.00371, +0.01044] | 0.667 |
| **训练 Δ vs 随机 Δ（J20）** | **+0.006823** | [+0.00325, +0.01034] | [+0.00361, +0.01066] | **0.729** |

水平：`base+ 0.555137 / base− 0.556568 / Δ 0.560381 / rand 0.556195`（J10）

**关键读法**：
1. Δ 把每段 J10 从 **0.55657 → 0.56038**，即 **+0.69%**；J20 **+1.21%**。
2. **同范数随机 Δ 完全无效甚至略负**（−0.00037，胜率 33%）⇒ **不是"任何小扰动都有益"**。
3. **训练 Δ 比随机 Δ 好 +0.00419，85.4% 的窗口胜出** ⇒ **WM 梯度确实在压低每段漂移**。

### 2.2 分任务（全部为正）

| 任务 | n / clips | 真 ΔJ10 | 真 ΔJ20 | WM6 预测 J20 | WM6 sign |
|---|---|---|---|---|---|
| boxing | 20 / 10 | **+0.00462** | +0.00850 | +0.01217 | 0.700 |
| double_desk | 17 / 9 | **+0.00294** | +0.00417 | +0.01262 | 0.529 |
| pp_box | 11 / 7 | **+0.00368** | +0.00742 | +0.01735 | 0.636 |

### 2.3 WM 校准（之前的"24× / 无判别力"是同一 bug 导致的）

| 预测器 | sign agree (J10) | sign agree (J20) | 恒正基线 | corr (J20) | corr (J10) | **幅度比** |
|---|---|---|---|---|---|---|
| WM0 | — | 0.625 | 0.667 | **+0.419** | — | 2.03× |
| **WM6** | 0.750 | 0.625 | 0.750 / 0.667 | **+0.344** | **+0.419** | **2.01×（J20）/ 3.55×（J10）** |

| 指标 | 旧（错） | **新（对）** |
|---|---|---|
| WM6 vs 真实 corr | +0.093 | **+0.344（J20）/ +0.419（J10）** |
| 幅度乐观 | 24.1× | **2.0×（J20）/ 3.6×（J10）** |
| WM0 vs WM6 | WM6 略好 | **WM0 corr 0.419 ≥ WM6 0.344（J20，打平/略优）** |

⚠️ **新发现：WM0 的排序相关性不低于 WM6。** 之前"WM6 略好"的结论也建立在错误数据上。
这条要单独留意：**WM6 相对 WM0 的优势目前没有在真物理排序上体现出来。**

### 2.4 critic（这条**没变**）

用正确的 `states_minus` 重算后：

| 口径 | v_mean | target_mean | corr | v_ev | bias |
|---|---|---|---|---|---|
| V_imag vs 截断 MC（训练域） | 0.458 | 0.450 | 0.838 | **+0.686** | +0.008 |
| **V_real vs 截断 MC（真 Isaac）** | −0.284 | 0.550 | **0.579** | **−5.643** | −0.834 |

⇒ **critic 跨分布仍然只给排序、不给绝对值。** 这一条与臂配对无关（V 只看状态）。

---

## 3. 更正清单（覆盖此前所有文档）

| 旧结论 | 状态 | 新值 |
|---|---|---|
| 真物理 ΔJ10 = **+0.00049**（0.09%） | ❌ 作废 | **+0.003813（0.69%）** |
| 真物理 ΔJ20 = **+0.00056** | ❌ 作废 | **+0.006720（1.21%）** |
| 逐窗胜率 79.2% | ❌ 作废 | **75.0%（J10）/ 66.7%（J20）** |
| "WM 幅度乐观 **24×**" | ❌ 作废 | **2.0×（J20）/ 3.6×（J10）** |
| "WM 逐窗判别无效（corr +0.09）" | ❌ 作废 | **corr +0.344（J20）/ +0.419（J10），有判别力但弱** |
| "WM6 略优于 WM0" | ⚠️ 动摇 | J20 上 **WM0 0.419 > WM6 0.344** |
| "训练只拿到 0.09%，收益很浅" | ❌ 作废 | **每段漂移降 0.69%/1.21%，且 85% 窗口胜过同范数随机** |
| critic 真物理 v_ev −5.68 | ✅ 成立 | −5.643（不变） |
| 跨进程 A/A = 0.0 | ✅ 成立 | 但那只验证了 plus 臂；**第二臂的进程内偏移 +0.00143 仍存在** |

---

## 4. 对"降低每段小漂移 → 降低整体漂移"假设的意义

**这条假设现在有直接证据了：**

1. 每段（20 tick chunk）的 tracking 回报确实被 Δ 抬高了 **0.69%（J10）/ 1.21%（J20）**；
2. 三个任务**全部为正**；
3. 同样的扰动幅度，**随机 Δ 做不到**（−0.00037），训练 Δ 做得到（+0.0038）；
4. 训练 Δ 相对随机 Δ 的胜率 **85.4%**。

**剩下的缺口只有"累积"这一步没测**：每段 +0.69% 是否真的累积成整体轨迹漂移下降
（而不是被 20 tick 后的重规划抹平）。这**需要闭环**，但闭环只是验证"累积"，不再是验证"有没有用"。

---

## 5. 仍未解决的

| # | 问题 | 状态 |
|---|---|---|
| 1 | 第二臂进程内偏移（+0.00143）只能**抵消**、不能**消除** | 若要彻底干净，需要"每臂一个独立进程"的回放协议（改 `isaac_probe` 支持 `--arm minus`） |
| 2 | critic 跨分布绝对值（v_mean −0.284 vs 0.550） | 未解；只能排序（corr 0.579） |
| 3 | WM0 的排序相关性不低于 WM6 | 新问题；WM6 的收益尚未在真物理排序上体现 |
| 4 | 闭环（累积效应） | 未跑 |
| 5 | 只有 3 个任务 / 26 clips 的 open-loop 覆盖 | val 前 48 窗按 clip 排序所致；需分层采样 |

---

## 6. 复现

```bash
WT=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-b-wm6
cd $WT && source scripts/b_env.sh

# 更正后的配对（含随机 Δ 对照）
python3 scripts/b_wm6/build_replay_job.py --ckpt .../expB_fixed/checkpoint_last.deploy.pt \
  --out .../isaac/job_rand_h20.pt --split val --n 48 --horizon 20 --delta-mode random --delta-seed 0
bash scripts/b_wm6/run_isaac_roll.sh .../job_rand_h20.pt .../roll_rand_h20.pt 1
python3 scripts/b_wm6/rescore_p7_arms.py --device cuda:1

# WM 方向（已修正到 states_minus）
python3 scripts/b_wm6/p7_direction_audit.py --horizon 20 --device cuda:1

# critic（已修正）
python3 scripts/b_wm6/check2_critic_target.py --ckpt .../critic_v5_K20.pt \
  --deploy-ckpt .../expB_fixed/checkpoint_last.deploy.pt --eval-horizon 20 --tag K20arms
```

---

## 7. 追加：#1 分层审计（6 任务 × 8 窗）

之前 48 窗只覆盖 3 个任务（val 前 48 窗按 clip 排序所致）。本次改为**每任务 8 窗、共 48 窗 / 28 clips / 6 任务**，
重跑 Isaac（`job_strat_base_h20.pt` / `job_strat_b_h20.pt`）并按任务统计。

### 7.1 真实收益：**6/6 任务全为正**

| 任务 | n | 真实 ΔJ10 | 真实 ΔJ20 |
|---|---|---|---|
| double_desk | 8 | +0.00214 | **+0.00413** |
| pp_box | 8 | +0.00431 | **+0.00897** |
| boxing | 8 | +0.00618 | **+0.01168** |
| open_door | 8 | +0.00492 | **+0.00918** |
| sit_sofa | 8 | +0.00712 | **+0.01372** |
| vision_navi | 8 | +0.00556 | **+0.01068** |
| **全体** | **48** | **+0.00504（win 0.792）** | **+0.00973（win 0.771）** |

**没有任何一个任务为负。** 每段 J20 平均 **+0.97%**（基值约 0.55~0.60 量级）。

### 7.2 WM 与真实的逐任务相关

| 任务 | **WM6 corr(J20)** | **WM0 corr(J20)** | WM6 符号 / 恒正基线 | 幅度比 |
|---|---|---|---|---|
| double_desk | +0.587 | +0.663 | 0.50 / 0.62 | 4.1× |
| pp_box | **+0.091** | +0.163 | 0.75 / 0.75 | 1.2× |
| boxing | +0.676 | +0.807 | 0.62 / 0.75 | 0.9× |
| open_door | +0.215 | +0.451 | 0.75 / 0.75 | 1.9× |
| sit_sofa | **+0.832** | **−0.120** | 0.88 / 1.00 | 1.6× |
| vision_navi | **−0.067** | +0.279 | 0.62 / 0.75 | 1.8× |
| **全体** | **+0.413** | **+0.300** | 0.688 / 0.771 | **1.66×** |

### 7.3 判定

| 断言 | 结论 |
|---|---|
| 真实收益跨任务一致（全为正） | ✅ **成立**（6/6） |
| WM 平均方向与真实一致 | ✅ 大体成立（WM 预测 6/6 为正） |
| **WM 逐窗排序跨任务一致** | ❌ **不成立**：pp_box 0.091、open_door 0.215、vision_navi −0.067 |
| **WM6 优于 WM0** | ❌ **不成立**：仅 sit_sofa 一个任务占优（0.832 vs −0.120）；其余 5 个 WM0 更好或持平 |
| 幅度校准 | ⚠️ 整体 1.66×，分任务 0.9×~4.1×（double_desk 差 4 倍） |

**关键读法**：
* WM 的"整体 +0.413"是被 **sit_sofa(+0.832)** 和 **boxing(+0.676)** 两个任务抬起来的；
  中位数相关只有 **+0.40**，且有 3 个任务在 0.1 附近徘徊。
* **WM6 refresh 在排序指标上没有买到东西**（1/6 任务占优）。
* 但**训练效果不等于排序能力**：训练用梯度平均方向，实测 6/6 任务真实为正。
  所以「WM6 排序不优于 WM0」与「用 WM6 训出来的 Δ 有效」**可以同时成立**。

### 7.4 复现

```bash
python3 scripts/b_wm6/build_replay_job.py --ckpt .../expB_fixed/checkpoint_last.deploy.pt \
  --out .../isaac/job_strat_base_h20.pt --split val --stratified-per-task 8 --horizon 20 --dup
python3 scripts/b_wm6/build_replay_job.py --ckpt .../expB_fixed/checkpoint_last.deploy.pt \
  --out .../isaac/job_strat_b_h20.pt --split val --stratified-per-task 8 --horizon 20
bash scripts/b_wm6/run_isaac_roll.sh .../job_strat_base_h20.pt .../roll_strat_base_h20.pt 1
bash scripts/b_wm6/run_isaac_roll.sh .../job_strat_b_h20.pt    .../roll_strat_b_h20.pt    1
python3 scripts/b_wm6/per_task_stratified.py --device cuda:1
```

产物：`phys/per_task_stratified.json`、`isaac/{job,roll}_strat_*`、日志 `/tmp/pt.log`。

---

## 8. 追加：#2 裁判对照 —— 用 WM0 vs 用 WM6 训出来的 Actor，真物理有差别吗

### 8.1 协议

| 项 | 设置 |
|---|---|
| 起点 | `c2b/seed0/checkpoint_last.pt`（it=300，**注意它本身是 WM6 训出来的**） |
| 步数 | 40（it=301…340） |
| 其余 | 完全相同：batch 256 / H10 / K20 / mask10 / λ_raw 1e-3 / β=0 / critic off / seed 0 |
| 唯一变量 | `--wm-ckpt`：`expA` 用锁定 **WM6**，`expC` 用 **WM0** |
| 评测 | 真 Isaac，**分层 48 窗（6 任务 × 8）**，base 与两臂都用 `states_minus`（偏移抵消） |

### 8.2 结果

| 臂 | 真 ΔJ10 | 95% CI | 胜率 | 真 ΔJ20 | 胜率 |
|---|---|---|---|---|---|
| **expA（WM6 裁判）** | **+0.004497** | [+0.00314, +0.00591] | 0.792 | **+0.008832** | 0.708 |
| **expC（WM0 裁判）** | **+0.004341** | [+0.00270, +0.00606] | 0.688 | **+0.008735** | 0.729 |
| **expC − expA** | **−0.000156** | **[−0.001107, +0.000782]** | 0.458 | −0.000098 | 0.479 |

水平：`J10 base 0.525580 / expA 0.530077 / expC 0.529921`

**CI 含 0 ⇒ 两个裁判在真物理上不可分辨。**

分任务（ΔJ10 vs base）：

| 任务 | WM6 裁判 | WM0 裁判 | 谁更好 |
|---|---|---|---|
| double_desk | +0.00281 | +0.00132 | WM6 |
| pp_box | +0.00316 | **+0.00550** | WM0 |
| boxing | +0.00353 | +0.00271 | WM6 |
| open_door | +0.00592 | +0.00553 | WM6 |
| sit_sofa | +0.00597 | **+0.00659** | WM0 |
| vision_navi | +0.00559 | +0.00440 | WM6 |
| **合计** | **+0.00450** | **+0.00434** | 4:2，无显著差异 |

### 8.3 判读

1. **WM6 相对 WM0 没有买到训练收益**：差 −0.00016（CI 含 0），4:2 的任务分布也在噪声内。
2. **但两条臂都拿到了 +0.0043~+0.0045（每段 ~0.8%）**，胜率 69–79%
   ⇒ **"用 WM 当可微裁判"这件事本身有效，不是 WM6 的功劳。**
3. **这个对照对 WM6 是有利的**（起点 c2b 就是 WM6 训出来的），
   即便在有利条件下 WM6 也没赢 ⇒ 结论更强。
4. 与 #1 的排序结果一致：**WM6 在排序上 1/6 任务占优，在训练收益上无差异。**
   ⇒ 目前**没有证据**支持"refresh 到 WM6 带来了额外价值"。

### 8.4 复现

```bash
WM0=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03/runs/three_track_v7/wm_versions/wm_v000/wm.pt
python3 scripts/train_b_wm6_h10_critic.py --out .../expC_wm0 --wm-ckpt $WM0 \
  --resume .../c2b/seed0/checkpoint_last.pt --updates 340 --beta 0 --critic-enabled 0 ...
python3 scripts/b_wm6/build_replay_job.py --ckpt .../expC_wm0/checkpoint_last.deploy.pt \
  --out .../isaac/job_expC_wm0_strat.pt --split val --stratified-per-task 8 --horizon 20
bash scripts/b_wm6/run_isaac_roll.sh .../job_expC_wm0_strat.pt .../roll_expC_wm0_strat.pt 1
python3 scripts/b_wm6/wm0_vs_wm6_judge.py
```

产物：`phys/wm0_vs_wm6_judge.json`、`isaac/{job,roll}_exp{A,C}_*_strat.*`、日志 `logs/expC_wm0b.log`。
