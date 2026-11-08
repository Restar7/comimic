# base 采集现状 / WM 训练现状 / A′ 语料现状（给 GPT 的现状清单）

> 写于 2026-11-08 15:25（正在采集中的数字会注明"进行中"）。
> 目的：把**当前手里有什么数据、WM 是怎么训的、训到哪一步**讲清楚，请 GPT 直接给"接下来怎么训 WM"的方案。
> 所有数字都附了可复核路径；仓库根 = `/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic`（工作树 `comimic-wt03`）。

---

## 0. 一句话背景

一个冻结的 VLA（官方微调 `checkpoint-8000`）出 64 维 motion token chunk（T=40），SONIC ONNX decoder 解成关节目标，再叠一个**逐帧 Δ 残差网**（1.74 M，cap=0.08 rad）。训练 Δ 时用**世界模型 WM 当可微裁判**（预测未来状态 → 算代价/奖励），因为真实 Isaac 不可微。**WM 准不准决定了 Δ 学到什么**，所以现在要专门把 WM 训好。

---

## 1. 三个角色与 WM 的位置

| 角色 | 模型 | 参数 | 是否训练 | 作用 |
|---|---|---|---|---|
| VLA | `Gr00tN1d7`（官方微调 `checkpoint-8000`） | 1.63 G | **冻结** | 视觉+语言+状态 → 64 维 token chunk（T=40） |
| SONIC decoder | `policy/release/model_decoder.onnx`（sha `c7241a12…`，40,900,688 B） | — | 冻结 | token → 29 维关节目标（50 Hz） |
| Δ 残差网 | token_proj+Transformer+GRU+MLP | 1.744 M | 训练中 | 在同一观测上加 `a_Δ`（真 rad，cap=0.08） |
| **WM** | `DynamicsMLP`（见 §6.1） | **8.82 M** | **要重训** | 可微替代 Isaac：`s_{t+1}=f(s_t,a_t,ξ)`，给 Δ 的训练提供未来状态/终止/奖励 |

WM 的单步定义（`comimic/third_party/SuperMimic-Reproduced/src/supermimic/world_model/world_model.py`）：

```
feat    = StateSpec.canonical(s_t)            # 71 维状态 → 规范特征
delta   = DynamicsMLP(feat, a_t, normalize(xi))   # 输入 133 维
s_{t+1} = StateSpec.integrate(s_t, delta)     # 输出 70 维增量
```

---

## 2. base（官方 FT8000，无 Δ）闭环采集现状

### 2.1 "确定性"口径

一条 base 记录只有在同时满足下面三条时才算**确定性数据**：

1. VLA = `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（`cell.log` / `deploy_audit.tsv` 里有逐文件 sha256）；
2. SONIC decoder = `policy/release/model_decoder.onnx`，sha256 `c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed`；
3. **没有任何 Δ**：`head_ckpt=MISSING`、`server.log` 里 `delta-v5]` 命中 0 次、客户端 `decoder=release` 契约校验通过。

### 2.2 各任务确定性集数

两个来源都满足 2.1，且互相独立：

| 任务 | `ab36/base`（11-07/11-08） | `closedloop_r3/base6x6`（11-08 本次） | 确定性集数 | 独立 seed |
|---|---|---|---|---|
| PPBox | 6（11-08） | 6 | **12** | 6 |
| DoubleDesk | 6（11-08） | 6 | **12** | 6 |
| Boxing | 6（11-08） | 6 | **12** | 6 |
| OpenDoor | 12（11-07，6 seed×2 遍） | 6 | **18** | 6 |
| SitSofa | 12（11-07，6 seed×2 遍） | 0/4 已完成，seed4/5 进行中 | **12→18** | 6 |
| VisionNavi | 12（11-07，6 seed×2 遍） | 未开始 | **12→18** | 6 |
| 合计 | 54 | 28 已完成 + 8 待采（进行中） | **82→96** | **36** |

**结论：6 个任务每个都有 6 个独立 seed；单任务硬证据 ≥12 集。** 全部 `max_steps=3000`（60 s 任务时间），配置 yaml 自 10-24 未改过（`third_party/HumanoidArena/isaaclab_twist2_g1/tasks/common_test_config/base_test/*_sonic_test.yaml`）。

### 2.3 base 成功率（两套独立采集的复现情况）

| 任务 | ab36/base | base6x6（本次） | 一致性 |
|---|---|---|---|
| PPBox | 2/6（成功 1227、948 步） | 2/6（成功 724、1118 步） | ✅ 比率一致，**成功 seed 不同** |
| DoubleDesk | 0/6 | 0/6 | ✅ 完全一致（全部 3000 步 timeout） |
| Boxing | 4/6（240、248、277、1821 步） | 4/6（176、233、276、298 步） | ✅ 比率一致 |
| OpenDoor | 6/6（974–1923 步） | 5/6（进行中已完 6 集） | ≈ |
| SitSofa | 3/6（487、963、2324 步） | **0/4（进行中）** | ❌ 待观察 |
| VisionNavi | 2/6（1678、701 步） | 未开始 | — |

需要 GPT 注意的三个现象：

1. **同 seed 跨进程不可复现**：PPBox 我这次成功在 seed3/4，ab36 在 seed3/5；Boxing 我这次成功在 seed0/2/4/5，ab36 在 seed0/3/4/5。⇒ 只能按 "n/6" 统计，不能拿单 seed 比。
2. **DoubleDesk 在 base 下 37 集全 0**（含早期单集 25 集）——不是抖动，是系统性做不到，最适合用来验证 Δ 的价值。
3. **SitSofa 三套数字打架**：11-06 sr4 4/6、11-07 ab36 3/6、11-08 本次 0/4（进行中）。yaml 未改，怀疑与录制视频/并发 Isaac/复位随机有关，**在拿到 6/6 之前不要把 SitSofa 当稳定基线**。

### 2.4 目录与复核方式

```
comimic/runs/ab36/base/<Task>/seed<k>/<run>/result.json + videos_{success,failure}/*.mp4
comimic/runs/closedloop_r3/base6x6/<Task>_seed<k>/<run>/result.json + videos_*
comimic/runs/closedloop_sr/sr4/{sits_base,vn_base}/ep_seed<k>.json      # 早期，无 sha 审计
```

汇总脚本（本次新写）：`comimic-wt03/scripts/summarize_closedloop.py`，兼容新旧三种目录布局：

```bash
python scripts/summarize_closedloop.py ../comimic/runs/closedloop_r3/base6x6 ../comimic/runs/ab36/base
```

### 2.5 base 数据的**关键局限**

base 闭环只落了 `result.json` + 视频，**没有逐 tick 的 `(s_t, a_t, s_{t+1})`**。所以这 96 集**不能直接拿去训 WM**，只能当"成功率基线"和"复位/评测配置"的参照。要变成 WM 语料必须用 §5 的 tick 记录器重采。

---

## 3. A′ 旧 WM 语料（WM0 的训练集）

### 3.1 规模与文件

| 项 | 值 |
|---|---|
| 目录 | `comimic-wt03/runs/a_prime_sonic_exc/` |
| shard 数 | 50（`<Task>_off<k>.pt`，6 任务 × offset ∈ {0,16,…,112}） |
| 总转移数 | **529,131** |
| 体积 | 370.6 MB |
| 合并文件 | `all_transitions.pt`（sha256 `418c310b7813968d52063b8fcb01f6c916019b9827c826ca60c9194fde8beb64`） |
| 采集时间 | 2026-11-05（Isaac，`third_party/isaaclab_220`，headless） |

单 shard 字段：

```python
{'states': (N,71) f32, 'actions': (N,29) f32, 'next_states': (N,71) f32,
 'frames': (N,) int64, 'clip_ids': (N,) int64, 'num_envs', 'steps', 'n_dropped_clip_end', 'meta'}
```

### 3.2 结构与时间基（已核实）

- `state71` 布局：`[root_pos(3) | root_quat_wxyz(4) | q(29) | lin_vel(3) | ang_vel(3) | qd(29)]`，与 SONIC/HA 关节序一致；
- `actions` **不是 rad，是归一化动作**：`a = (cmd_q − dq_ckpt) / sc_ckpt`（`scripts/build_real_transitions.py:58,62,100`），**并且加了动作噪声**：`meta.action_noise_std=0.15`、`noise_mode='gauss'`、`perturb_hold=1`。实测该 shard `actions` absmax 9.21、mean|a| 0.88，全 50 shard 合并后 absmax 24.4 —— 都在归一化域，不是弧度；
- 行序是 **(step-major, env-minor)，num_envs=16** ⇒ 时间相邻的行间隔 16 行（我实测 lag=16 时 root 位移中位数 8.5 mm、frames 步长 1；lag=1 时位移 0.83 m，是跨 env 跳变）；
- `meta.exec_steps=1`、`frames_unit='global_library_row'`、`seg_len_median=44`；
- **时间基验证**：|lin_vel| 中位数 0.327 m/s，而相邻帧位移/0.02 = 0.63 m/s（同一量级）；若假设步长 0.4 s 则得 0.03 m/s（差 20 倍）。⇒ **A′ 确实是 50 Hz、dt=0.02 s**。

### 3.3 采集配方（来自 shard meta）

| 键 | 值 | 含义 |
|---|---|---|
| `source` | `a_line_wbc_only` | 只跑 WBC 执行器 + 参考 clip，**不接 VLA** |
| `wbc_ckpt` | `runs/S1_handover/exp_WBC_exec_haonly_best.pt` | 执行器 |
| `start_mode` | `windows` | 从参考库随机窗口起步 |
| `n_clips_used / usable` | 15 / 100（每 shard） | 参考 clip 复用度 |
| `action_noise_std` | 0.15 | 动作高斯噪声（**激励，用于让 WM 见到 off-trajectory**） |
| `reset_max_drift` | 0.15 | 漂移超限就复位 |
| `n_leash_resets` | 189 | 复位次数 |
| `note` | 奖励未计算 | 奖励用 `scripts/compute_collect_reward.py` 事后算 |

### 3.4 A′ 的根本局限（对本次训练最重要）

A′ 是**11-05 之前的旧策略**（`exec_WBC_exec_haonly_best.pt` + 参考 clip），**不是现在的 base，也不是现在的 A/B Δ 策略**。所以：

- WM0 只见过 A′ 的分布；
- 现在的策略（base / A / B / A_v8）都会偏离 A′，而偏离量还没被量化（见 §6.5 的 31.80 cm）。

---

## 4. 新 on-policy 语料（V7 refresh 已经跑过一轮）

### 4.1 `cycle000_A`（唯一一份真正的 on-policy 语料）

路径：`comimic-wt03/runs/three_track_v7/buffer/cycle000_A/`

| 项 | 值 |
|---|---|
| 策略 | `A_delta`，ckpt `runs/three_track_v6/delta_continue/seed0/checkpoint_actor_best.pt` |
| 任务 | **只有 OpenDoor** |
| seeds | 2–11（12 个 Isaac 进程，`episode_seed*.log` 各一份） |
| 采集 | 2026-11-08 12:04–12:59，每进程约 4.6 min |
| ticks | **7,552**（每进程 600–800） |
| segments | **366**（`horizon=20`，不重叠切窗） |
| 体积 | `segments.pt` 每段 `states(20,71) / actions(20,29) / next_states(20,71)`，`dt=0.02` |
| 合同检查 | `action_domain_check`: frac_outside=0.0013（tol 0.25，阈值 0.005）✅ |

另有 `buffer/smoke_aline`（OpenDoor seed0、60 步、2 段，仅冒烟）。

### 4.2 记录方式（可复用）

不改 vendored 源码，用 sitecustomize 注入 hook（`comimic-wt03/scripts/v7_tick_record/sitecustomize.py`）：

```bash
export V7_TICK_DUMP_DIR=$OUT/ticks
export V7_TICK_POLICY_ID=A_delta
export V7_TICK_WM_VERSION=wm_v000
export PYTHONPATH=scripts/v7_tick_record:$PYTHONPATH
bash scripts/ha_isaac_closedloop.sh ...
```

hook 包住 `SonicActionProvider.get_action`，逐 tick 落盘 `state71 / actions(=q_target) / episode_id / frame_count / q_target / raw_action`，分块 `torch.save`（原子写）。

**时间基同样验证过**：新 tick 的 `|lin_vel|` 中位数 0.061 m/s，相邻 tick 位移/0.02 = 0.060 m/s（完全吻合）；若按 0.4 s 则差 20 倍。⇒ dt=0.02 可信，与 A′ 同一时间基。

### 4.3 字段覆盖度：base 采集 vs on-policy tick（重要）

要求每个 tick 至少 10 个字段，实际覆盖如下：

| 字段 | base 采集（ab36/base、base6x6） | on-policy tick（cycle000_A） | 说明 |
|---|---|---|---|
| `state71` (s_t) | ❌ 完全没有 | ✅ `states` | 布局 `[p(3)\|quat_wxyz(4)\|q(29)\|v(3)\|w(3)\|qd(29)]` |
| `action29` | ❌ | ✅ `actions`（**归一化域** `(q_target−dq)/sc`） | 与 A′ 逐位同口径 |
| `next_state71` | ❌ | ✅ 段里 `next_states`（= 下一 tick 的 `states`） | 单 tick 文件里没有，转换时 shift 得到 |
| `q_cmd` | ❌ | ✅ `q_target`（rad，SONIC 序） | decoder 目标 |
| `q_applied` | ❌ | ❌ **没有** | 执行器裁剪/限幅后真正下发的目标没记；只能从 `next_state` 的 q 通道反推近似值 |
| `task_id` | ❌ | △ 只在 chunk 的 `meta.task` 里，非每行 | 多任务混采时必须每行带 |
| `episode_id` | ❌ | ✅（PID 哈希 + `episode_id` 组成 episode 键） | 同进程多 seed 时 `episode_id` 恒为 0，靠 PID 区分 |
| `tick` | ❌ | ✅ `frame_count`（每 tick +1） | 已验证连续无丢帧 |
| `done` | ❌ | ❌ **没有** | 只有 episode 结束的隐含边界，没有终止标签 |
| `dt` | ❌ | △ 段里硬编码 `dt=0.02`，非每行 | 已用速度一致性验证确实是 50 Hz |

**结论：base 的 96 集里，这 10 个字段一个都没有（只存了 `result.json` + 视频）；现有 on-policy tick 有 6 个，缺 `q_applied`、`done`，`task_id`/`dt` 只在块级。**

要补齐，改动点集中在一个文件（`scripts/v7_tick_record/sitecustomize.py` 的 `get_action` 包装）：

1. `q_applied`：在 hook 里记录"传进 sim 之前"的目标；若 Isaac 端有裁剪，需要在 `controller.step()` 的 action 应用处再挂一个 hook；
2. `done`：从 `env` 的终止标志读（`termination`/`reset` 事件），或至少在 `on_env_reset` 时给最后一行打 `done=1`；
3. `task_id`/`dt`：每行写进 row dict（而不是只写 chunk meta），多任务混采时是硬需求；
4. `next_state71`：转换时由下一 tick 的 `states` 得到（现有 `_segments_from_ticks` 已经这么做），单 tick 文件里可省。

### 4.4 这块语料的其它缺口

1. **只有 1 个任务（OpenDoor）+ 1 个策略（A_delta）**，没有 base 臂、没有 PPBox/DoubleDesk/Boxing/SitSofa/VisionNavi；
2. 规模小：7,552 tick ⇒ 366 段 ⇒ 5,040 训练行（vs A′ 529,131 行，约 1%）；
3. episode 键用 (PID 哈希, episode_id)，本份数据 12 个进程 ⇒ 12 个 episode，切分正确；但 `clip_id` 全为 0，**任何依赖 clip_id 的泄漏检查对新数据是空转**；
4. 没有录 `xi`（参考/命令）的完整含义说明，也没有 reward / termination 标签（奖励需事后算）。

---

## 5. WM 本体现状

### 5.1 架构（实测自 ckpt）

```
输入 133 = state canonical(71) + action(29) + xi(33)
net: 133 → 2048 → 2048 → 2048 → 70      (每层 LayerNorm + 激活, Kaiming)
输出 70 维状态增量，经 StateSpec.integrate 得到 s_{t+1}
termination head: 4 → 64 → 64 → 1（输入 = 高度误差、倾角误差及其幅值）
analytic heads（观测/奖励/FK）：无参数
参数量 8.82 M；ckpt 35.3 MB
```

配套统计量存在 ckpt 里：`in_norm.mean/var`(133)、`out_norm.mean/var`(70)、`in_norm.count=2.0e8`（= 训练见到的样本数）。

### 5.2 训练口径

- 损失：**SuperMimic 原版逐维归一化平方误差**（`ssm`：用 WM 自己的 `out_norm.std()` + `ERR_FLOOR`，不是等权 MSE）；
- 监督：单步 `s_{t+1}=WM(s_t,a_t)` + **k 步自回归（k=20，不 teacher forcing）**，实测训练脚本用 `mode=both`；
- 两阶段：ckpt meta 记录 `stage1=800 / stage2=800` 步，另有 `gain_weight=2.0` + `gain_pairs=runs/paired_probe2/gain_pairs.pt`（"动作增益"辅助项，用于修 ∂s/∂a 偏小）；
- 训练脚本：`scripts/pretrain_wm.py`、`scripts/finetune_wm_multistep.py`（只续 k 步损失，`--inc-weight` 可加"动作可归因增量"项）。

### 5.3 版本库与门禁（已实现，可直接复用）

```
comimic-wt03/runs/three_track_v7/wm_versions/
  active.json                     # 指针 + sha + history
  wm_v000/                        # WM0 = a_prime_sonic_exc/wm_gain2.pt 的拷贝
  wm_c1_v001_published/           # WM1（2026-11-08 14:29 发布）
  wm_probe_candidate/ wm_probe_gate/
```

| 版本 | sha256（前 8） | 说明 |
|---|---|---|
| `wm_v000` | `fc5aeb30` | WM0，旧 A′ 训练，A/B 一直用它 |
| `wm_c1_v001_published` | `81d24caf` | **当前 active**，用 A_delta on-policy 数据 70/30 混采续训 150 updates |

门禁 `gate.json`（`wm_c1_v001_published/gate.json`）判据与结果：

| 判据 | 结果 | 值 |
|---|---|---|
| finite | PASS | — |
| action_contract | PASS | 动作域与 A′ 一致 |
| no_leakage | PASS | 但新数据 clip_id 全 0（见 §4.3） |
| new_e20_rel_improve | PASS | **+65.5%**（CI 下界 0.676） |
| old_e20_rel_degrade | PASS | −0.45%（旧分布几乎不掉） |
| **dir_agree（梯度探针）** | **FAIL（OVERRIDE）** | `梯度探针未跑`，靠 `allow_missing_gradient_probe` 放行 |

超参（`train_summary.json`）：`mode=both, k=20, lr=1e-5, batch=256, window_batch=8, updates=150, old:new=0.7:0.3`。

### 5.4 已知数字（全部可复核）

| 指标 | WM0（`fc5aeb30`） | WM1（`81d24caf`） | 出处 |
|---|---|---|---|
| 旧 A′ val，h=1，root 误差 | 0.0280 m | — | `diagnostics/wm0_oldonly/REPORT.md` |
| 旧 A′ val，h=20，root 误差 | **0.0459 m（4.59 cm）** | 0.0448 m（−0.45%） | 同上 + `gate.json` |
| 旧 A′ val，h=20，root p95 | 0.1854 m | — | 同上 |
| 新 on-policy 分布，h=20 root | **0.3180 m（31.80 cm）** | **0.1097 m（10.97 cm）** | 其他会话的门禁报告 |
| 旧 A′ val，h=20，∑ssm | 89.13 | — | REPORT.md |
| 旧 A′ val，h=20，heading | 0.0160 rad | — | REPORT.md |

**读法**：WM0 在自己训练分布上是 4.6 cm（勉强可用），一到现在的策略分布就是 **31.8 cm**（20 步 = 0.4 s 就飘 32 cm，等于完全失效）；WM1 把新分布拉回 11 cm，但**根因（为什么新分布这么离谱）没有解释**，而且梯度方向（dir_agree）从没验证过。

### 5.5 疑点清单（希望 GPT 重点看）

1. **WM 当裁判时，训的是"WM 眼里的好"，不是真机好**。WM0 在新分布 31.8 cm 误差，几乎不可能给出可靠梯度；这可能是 A/B Δ 出现"方向崩塌（direction collapse）"的根因之一。
2. **门禁是自证循环**：新数据只有 366 段且与训练数据同源（同一批 12 个进程），"new_e20 改善 65.5%"是在同一份小数据上量的，没有独立 holdout。
3. **梯度探针（dir_agree）从未跑过**，两次发布都靠 override。⇒ 我们并不知道 WM 的 ∂s/∂a 符号/量级对不对。
4. **新数据只有 OpenDoor 一个任务**，而 PPBox/DoubleDesk/Boxing/SitSofa/VisionNavi 的动力学（推门 vs 搬箱 vs 坐沙发）差别很大。
5. **A′ 数据带 0.15 动作噪声**（激励）；新 on-policy 数据是策略自己的动作（无噪声、分布窄）。混采 70/30 时，"见过的动作范围"其实还是 A′ 决定，但"见到的状态分布"被新数据主导 —— 两者可能不自洽。
6. WM 输入 133 维里 `xi`(33) 的物理含义、是否真的必要，文档没写清；如果 xi 携带了参考轨迹信息，那么 WM 在"没有参考"的场景（如 base 自主探索）会失真。

---

## 6. 现成脚本清单（可直接复用）

| 阶段 | 脚本 | 作用 |
|---|---|---|
| 采集 tick | `scripts/collect_wm_refresh_isaac.py` | Isaac 闭环短 episode → ticks → 20 步 segments + manifest（含 action-domain 合同检查） |
| 采集注入 | `scripts/v7_tick_record/sitecustomize.py` | 不改 vendored 源码的逐 tick 落盘 hook |
| 诊断偏移 | `scripts/diagnose_wm_shift_v7.py` | 多步自回归误差 + OOD 支撑度（NN 距离，阈值 = 旧 A′ val 99 分位 23.542） |
| 混采 | `scripts/make_wm_mix.py`、`diff_vla_wbc/wm_replay_buffer.py` | 70/30 混合、防泄漏、k 步窗口 |
| 训练 | `scripts/train_wm_refresh_v7.py` | 从 active 版本拷贝 → 候选版本训练（单步 / k 步 / both） |
| 评测/门禁 | `scripts/eval_wm_refresh_v7.py`、`diff_vla_wbc/wm_gate*` | 5 条判据 + bootstrap CI |
| 版本发布 | `diff_vla_wbc/wm_version_store.py` | 原子发布/回滚、active.json、history |
| 梯度探针 | `scripts/probe_wm_action_jacobian.py`、`probe_wm_gain_heldout.py`、`probe_wm_gradient_truth.py` | ∂s/∂a 检查（**从未跑**） |

一条最小命令模板（训练一份候选 WM）：

```bash
python3 scripts/train_wm_refresh_v7.py \
  --version-store runs/three_track_v7/wm_versions --new-version wm_v002 \
  --old-replay runs/a_prime_sonic_exc/all_transitions.pt \
  --new-replay  runs/three_track_v7/buffer/<NEW_BUFFER> \
  --old-ratio 0.7 --new-ratio 0.3 --mode both --k 20 \
  --updates 300 --lr 1e-5 --batch 256 --eval-every 50 --device cuda:1
```

---

## 7. 请 GPT 回答的问题

**P0（决定能不能训）**

1. 用 366 段（5,040 行）on-policy + 529,131 行 A′，**混采比该是多少**？70/30 是不是让新数据被淹没（新数据只占 1.2%）？要不要改成按"有效样本数"而不是按 batch 比例？
2. 新数据只有 OpenDoor，**六个任务要不要各采一份**？每任务多少 tick 才够（当前 7.5k）？给一个"够用"的判据（比如某任务 20 步 root 误差降到 X cm）。
3. **续训还是重训**？现在是从 WM0 拷贝续 150 updates。如果 WM0 本身在错误的时间基/状态分布上过拟合，续训是不是把错误也继承了？
4. **时间基/归一化一致性**：A′ 与 on-policy 都是 dt=0.02（已核实），但 A′ 有 0.15 动作噪声、on-policy 没有。`in_norm/out_norm` 要不要按新数据重算？重算会不会破坏旧分布？

**P1（决定训好没有）**

5. **门禁怎么改才不是自证**：现在 new_e20 是在与训练同源的小数据上量的，怎么构造 episode 级独立 holdout？需要多少段？
6. **梯度探针怎么设计**：怎么用可复现的实验证明 WM 的 ∂s/∂a 符号与量级正确（真机 Isaac 有限差分 vs WM 解析梯度）？给一个可执行的协议。
7. 损失口径：`ssm`（逐维归一化平方误差）vs 物理单位加权（root 位置/朝向权重更高）——哪种更适合"当裁判"这个用途？
8. 训练超参建议：`updates / lr / batch / k / mode(single|kstep|both)` 的具体数值，以及**早停判据**（用哪个指标、阈值多少）。
9. 要不要给 on-policy 数据加**激励噪声**（像 A′ 那样 0.15）来扩大动作覆盖？加了会不会破坏"这是真实策略分布"的意义？

**P2（工程边界）**

10. 我们有 2×48 GB GPU，Isaac 单实例约 3.5–4.2 Hz 控制频率。**采 6 任务 × N tick 的时间预算**怎么估最省？
11. base 闭环 96 集（3000 步/集）目前没有 tick 记录。**要不要重采成 tick 版**，还是说 base 数据对训 WM 没有额外价值（因为 Δ 训练时策略是 base+Δ）？

---

## 8. 附录：关键路径与 sha 汇总

```
# 官方 VLA
runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000            (15 文件, 6.92 GB)

# SONIC decoder（唯一正确的一个）
../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx
  sha256 c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed  (40,900,688 B)
# 注意：还有一个 policy/low_latency/model_decoder.onnx（149,763,762 B），不要混用

# base 闭环（确定性）
comimic/runs/ab36/base/<Task>/seed<k>/...                     # 6 任务 × 6 seed
comimic/runs/closedloop_r3/base6x6/<Task>_seed<k>/...         # 本次，进行中

# A′ 旧 WM 语料
comimic-wt03/runs/a_prime_sonic_exc/all_transitions.pt        # 529,131 行
  sha256 418c310b7813968d52063b8fcb01f6c916019b9827c826ca60c9194fde8beb64
comimic-wt03/runs/a_prime_sonic_exc/wm_gain2.pt               # = WM0
  sha256 fc5aeb30e987f016d24a8392af9d0fecb3151e07dc33d49b71f0544f9621ca97

# on-policy 语料
comimic-wt03/runs/three_track_v7/buffer/cycle000_A/           # 366 段 / 7,552 tick / OpenDoor

# WM 版本库
comimic-wt03/runs/three_track_v7/wm_versions/active.json      # 当前 = wm_c1_v001_published
  wm_v000              sha256 fc5aeb30…   （A/B 一直用这个）
  wm_c1_v001_published sha256 81d24caf…   （2026-11-08 14:29 发布, dir_agree OVERRIDE）

# 诊断报告
comimic-wt03/runs/three_track_v7/diagnostics/wm0_oldonly/REPORT.md
comimic-wt03/runs/three_track_v7/wm_versions/wm_c1_v001_published/gate.json
```
