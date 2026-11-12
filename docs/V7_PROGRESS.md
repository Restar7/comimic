# COMIMIC V7 实施进度（Mixed Replay + WM Refresh + A/B/C 稳定性修复）

> 对应任务书：`docs/`（V7 完整修改任务书）。状态口径 **PASS / RUNNING / BLOCKED / NOT_RUN**。
> **只写已经落地的东西**：新 CLI 只有"代码 + 测试 + 真跑过"才标 PASS，否则标 NOT_RUN。
> 旧实验目录 `runs/three_track_v6/`、`runs/a_prime_sonic_exc/` **只读**，V7 产物一律在 `runs/three_track_v7/`。

---

## 0. Phase 状态矩阵

| Phase | 内容 | 状态 | 证据 |
|---|---|---|---|
| **0** | 冻结 A/B/C + WM0 基线（hash/metrics/ckpt 合同） | **PASS** | `runs/three_track_v7/phase0_freeze/{inventory.json,README.md}`；`scripts/freeze_v7_baseline.py` |
| **1a** | Shared Mixed Replay Buffer（episode 切分/比例/动作域/k 步窗口/manifest） | **PASS（代码+单测）** | `diff_vla_wbc/wm_replay_buffer.py`；`tests/test_wm_replay_buffer.py` **12/12** |
| **1b** | 真实 Isaac 短轨迹采集器（A/B/C 各 128 起点） | **PASS（A 线已通，B/C 待换策略）** | `scripts/v7_tick_record/sitecustomize.py`（注入式逐 tick 落盘）+ `scripts/collect_wm_refresh_isaac.py`；A 线已采 12 episodes，动作域 q50=2.918 vs A′ 2.93 |
| **2a** | WM0 分布偏移诊断（E1/E5/E10/E20 + 分块 + 物理单位 + 支撑度） | **PASS（旧 A′ 部分）** | `scripts/diagnose_wm_shift_v7.py`；`runs/three_track_v7/diagnostics/wm0_oldonly/`（new 部分 NOT_RUN，缺新数据） |
| **2b** | WM 梯度真值探针（ΔJ_WM vs ΔJ_Isaac） | **PARTIAL** | 直接复用已有 `scripts/probe_wm_gradient_truth.py`（plan/score 两段 + `scripts/isaac_probe_action_rollout.py`）；**尚未用 V7 数据跑** |
| **3a** | WM 候选增量训练（70/30、H∈{1,5,10,20}、复用 `wm_refresh`） | **PASS（代码+真跑）** | `scripts/train_wm_refresh_v7.py`（3 步真实跑通、candidate 落盘）；`runs/three_track_v7/wm_versions/wm_probe_candidate` |
| **3b** | 发布门禁 + 原子版本/回滚 | **PASS（代码+测试+真跑）** | `diff_vla_wbc/wm_refresh_gate.py`、`wm_version_store.py`、`scripts/eval_wm_refresh_v7.py`；门禁对 probe 候选判 **FAIL**（3 条原因）且 active 不变 |
| **3c** | Actor 侧固定 WM 版本（`--wm-*` / gated 复查 / DDP hash 一致） | **PASS（代码+测试）** | 两个训练器已接线；`test_wm_version_store`（pin/rollback/hash 断言）14 passed |
| **4A** | A 的 raw-norm λ 扫描（0.001/0.003/0.010） | **NOT_RUN** | 训练器已支持 `--lambda-raw`，未跑 |
| **4B** | B 的 960/1000 收尾 | **BLOCKED** | **A/B 的 ckpt 不含 optimizer/scheduler/RNG ⇒ 严格 resume 不可行**（见 §2）；按任务书 §9.1 改为"独立 ckpt 评测" |
| **4C** | C1/C2 拆分 + 分组 LR + 物理损失重标 | **PARTIAL** | C1（LoRA+proj_out）主跑 RUNNING；C2（纯 LoRA）与分组 LR 未跑 |
| **5** | Fixed-WM vs Refreshed-WM 对照 | **NOT_RUN** | 依赖 3a/3b |
| **6** | 周期 Refresh 调度 | **NOT_RUN** | 依赖 3b |

---

## 1. 本轮已完成（代码 + 测试 + 真跑）

### 1.1 Phase 0：基线冻结

```bash
python3 scripts/freeze_v7_baseline.py --out runs/three_track_v7/phase0_freeze
```

产物 `inventory.json`：20 个产物的 `path/bytes/sha256/mtime` + 关键指标（每项带出处），
`README.md` 是人读版。关键 hash（用于后续任何"WM/数据被换过"的告警）：

| 产物 | sha256（前 16） |
|---|---|
| `wm_gain2.pt`（WM0） | `fc5aeb30e987f016` |
| `all_transitions.pt`（A′） | `063355356116…`（整文件） |
| A best | `5c318a5a029d…` |
| C 旧跑 last | `81f88ab089a3…` |
| C 审计 manifest | `7234791ce4de…` |

### 1.2 Phase 1a：Shared Mixed Replay Buffer

`diff_vla_wbc/wm_replay_buffer.py`：

- **旧 A′ 合同完全复用**：`states(71) / actions(29) / next_states(71) / frames / clip_ids`，
  `validate_transitions` 对形状/dtype/finite **fail-closed**（不补零、不猜）；
- **episode/clip 级切分**：用 `sha1(seed:episode)` 分桶，`assert_no_leakage()` 断言
  train/val 的旧 clip 集合与新 episode 集合**各自不相交**；
- **比例混合**：`old_ratio/new_ratio`（70/30、90/10 都测过），实际比例偏差 <3%；
  新数据不够时返回 `new_ratio_shortfall=True`，**不静默伪装成 70/30**；
- **per-policy 平衡**：`policy_weights={"A":1,"B":1,"C":1}`（任务书 §3.1 的"A/B/C 近似均衡"）；
- **动作域合同**：`assert_action_domain` 用旧 A′ 动作分布做 ±0.25 的软边，
  越界直接抛 —— 专门拦"把 `q_target_applied` 当 action"这种换域错误（任务书 §3.2 明令禁止）；
- **k 步窗口**：`build_kstep_windows` 先按 `(clip, frame)` 排序求**连续段**（`all_transitions.pt`
  是多 shard 拼接、行序不保证；直接 reshape 会造出跨 clip 的假窗口），只在同段内取窗；
- **manifest**：旧/新张量 sha256、split 计数、per-policy 计数、采集器段级哈希、
  动作域分位表、`wm0_original_split="UNKNOWN"`（**不假装知道原 WM0 的 split**）。

测试（`tests/test_wm_replay_buffer.py`，12/12 通过）：防泄漏、70/30 与 90/10 比例、
shortfall 不隐藏、动作域拒绝 `q_target` 风格动作、旧数据只读（sha 不变）、
k 步窗口不跨 clip、段存取往返哈希、缺新数据/比例不合法时 fail-closed。

### 1.3 Phase 2a：WM0 分布偏移诊断（旧 A′ 部分）

```bash
python3 scripts/diagnose_wm_shift_v7.py \
  --wm-checkpoint $MAIN/runs/a_prime_sonic_exc/wm_gain2.pt \
  --old-replay     $MAIN/runs/a_prime_sonic_exc/all_transitions.pt \
  --horizons 1,5,10,20 --windows 128 --device cuda:1 \
  --old-ratio 1 --new-ratio 0 \
  --out runs/three_track_v7/diagnostics/wm0_oldonly
```

实现要点（都是"口径不能错"的地方）：

- 误差用 **WM 自己的** `out_norm.std() + ERR_FLOOR` 归一化（`wm_refresh.ssm_wm_loss_parts`），
  分块 pos/rot/linvel/angvel/q/qd，**不是等权 MSE**；
- 滚动**不做 teacher forcing**（喂模型自己的输出），并且 **必须传 `step_idx`**
  （GRU 隐状态生命周期 + `_w_calib_alpha` 逐 step 标定；不传会系统性偏）；
- **绕过 `bundle.step` 的 `std→ha` 换序**，走 `bundle.base.step`：A′ 的 actions 本来就是
  decoder 原始输出（HA/SONIC 序），再换一次就是把同一个张量置换两次（`sonic_isaac.py:step`
  注释里那个"动作通道错位"的 bug）；
- 支撑度：旧 A′ **train** 标准化空间的最近邻距离，阈值取旧 A′ **val 的 99 分位**（不是拍脑袋）。

**旧 A′ val 的 WM0 多步误差（128 起点，`wm_gain2.pt`，sha `fc5aeb30e987f016`）**：

| h | ∑ssm | pos | rot | q | qd | root 位置 (m) | root p95 (m) | heading (rad) | q MAE (rad) | qd MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 67.66 | 37.74 | 22.73 | 2.52 | 0.90 | 0.0280 | 0.1666 | 0.0143 | 0.0183 | 0.5529 |
| 5 | 62.07 | 36.28 | 21.95 | 1.39 | 0.29 | 0.0300 | 0.1662 | 0.0122 | 0.0156 | 0.3434 |
| 10 | 75.32 | 41.86 | 29.28 | 1.58 | 0.23 | 0.0355 | 0.1707 | 0.0156 | 0.0161 | 0.3236 |
| 20 | 89.13 | 56.03 | 29.64 | 1.33 | 0.16 | 0.0459 | 0.1854 | 0.0160 | 0.0162 | 0.2835 |

支撑度：旧 val 自身 median NN 距离 **11.50**，99 分位阈值 **23.54**，超阈 1.05%（自洽）。

> 说明：h=5 的 ∑ssm 低于 h=1 是**速度通道**（qd 0.90→0.29）贡献的，pos/rot 单调上升；
> 这一现象需要在新数据上复核，不在本轮下结论。

---

## 2. 本轮新增的**硬发现**（会改变后续计划）

### 2.1 A/B 的 checkpoint **没有训练状态** ⇒ 严格 resume 不存在

`runs/three_track_v6/delta_continue/seed0/checkpoint_actor_best.pt` 的 payload 只有
`delta_token_v5_state / spec / meta / ckpt_sha256 / it / dJ_val / cap / data_mode`：
**没有 optimizer、没有 scheduler、没有 RNG、没有 sampler**（`_save_ckpt` 只存推理所需）。

⇒ 任何"A/B 续跑到 1000"都只能是 **warm-start（优化器从 0）**。任务书 §9.1 的处理因此改为：
把 B 的 960 步模型当**独立有效 ckpt** 做 Val/Test，并在报告里明确写
`strict_resume=BLOCKED(missing optimizer/scheduler/rng)`。

### 2.2 B seed0 只有 `best`，没有 `last`

B 在 960/1000 被"全停"命令终止时，v6 训练器只在**收尾**写 `checkpoint_actor_last.pt` ⇒
`last` 不存在（`checkpoint_actor_best.pt` 存在，对应最后一次 val 改善的 it≈950）。

### 2.3 C 的新高 LR 跑（RUNNING）并不支持"LR 太小"这一诊断

`train400_lr1e-4_ddp`（DDP×2、有效 bs64、lr 1e-4）到 it=150 的 val ΔJ：
`50:+0.00023 → 100:+0.00239 → 150:+0.00036`（**震荡**），Δtoken 稳定在 0.005 左右。
对比旧跑 lr=1e-5：`50:+0.00204 → 200:+0.00164 → 650:+0.00548`（**单调爬升**）。

⇒ **把 LR 提 10 倍没有带来系统性的 ΔJ 提升**，更像"动得更快但方向更噪"。
本轮的结论应写成："C 的瓶颈是 **WM 对 LoRA 子空间的梯度信噪比**，不是步长"，
而不是"LR 太小"。最终以 400 步收尾曲线为准。

### 2.4 现有"真物理分片"的动作域与 A′ **不兼容**（这条拦住了拿旧数据冒充新数据）

V7 的动作域门禁（旧 A′ 分布 ±0.25 软边）对已有真物理数据逐一检查：

| 数据 | 来源 | q50 \|a\| | 与 A′（q50 2.93）比 | 门禁 |
|---|---|---|---|---|
| `all_transitions.pt`（A′） | SONIC decoder 下发 | 2.93 | — | 基准 |
| `runs/wm_real/a_w*.pt`（24 片 ×15.36 万行） | 冻结 **Supermimic policy**（`a_line_wbc_only`） | **1.55** | 0.53× | **拒绝**（15.36 万行里 24,012 行越界） |
| `runs/wm_real/a_line.pt` | 同上 | **8.73** | 2.98× | 拒绝 |
| `runs/B_*/dump/*.pt` | VLA 在环（MuJoCo） | 2.73 | 0.93× | 接近，但**只有 1 个 dump 文件**、且缺 `next_states` |

⇒ 结论：**不能**拿 `a_w*.pt` 当"新 on-policy 数据"混进 A′ 的 WM（那会把"Supermimic 动作 scale"
当成"S0NIC decoder 动作"学进去，静默污染动力学）。要拿到合同一致的真 Isaac 数据，只能：
`Actor token → 正式 SONIC → q_target → Isaac PD → 下一真实状态`（任务书 §4.2），
而这需要在 Isaac 进程里**逐 tick 落盘**（`state_t, action_t, state_{t+1}, q_raw/q_applied/q_real`），
当前 vendored 栈没有这个 hook（`--recording_save_dir` 只写视频，`result.json` 只有汇总）。

**Phase 1b 已解决**（2026-11-08）：用 `scripts/v7_tick_record/sitecustomize.py` 在
`SonicActionProvider.get_action` 上做 monkeypatch（不改 vendored 源码），逐 tick 落
`state71 + action(WM 域) + q_target`，再由 `collect_wm_refresh_isaac.py` 转 segment。

踩过的 4 个坑（都写进代码注释了）：

1. **动作域**：A′ 的 `actions` 是 `(cmd_q − dq_ckpt)/sc_ckpt`（`build_real_transitions.py:58-106`），
   不是 decoder 归一化输出、也不是 43 维 env action。按同一仿射变换记录后 **q50=2.918 vs A′ 2.93**；
2. `get_action` 返回的是 **43 维**（29 体 + 14 手）⇒ 直接记会 `43 != 29` fail-closed；
3. 短 episode（60 tick）里 chunk=256 永远不 flush ⇒ 默认改成 8，并在 `on_env_reset` 也 flush；
4. **相对路径**：Isaac 进程 cwd 是 `$MAIN`，会让 tick 写进主仓 ⇒ 一律 `resolve()` 成绝对路径；
   另外 **PYTHONPATH 污染**会让 `import onnxruntime` 解析到半残包
   （`ModuleNotFoundError: onnxruntime.capi.onnxruntime_pybind11_state`）⇒ 采集器只注入
   `v7_tick_record` + 主仓，其余交给 harness 自己 export。

---

### 2.5 C（LoRA16-FT8000）在 review 后的修复与结果

按外部 review 把 C 改成 **C1**：主干 LoRA16 + **proj_out 也用 LoRA16**（原全秩权重冻结，
proj 组 = 114,688 参数）+ **双重物理 trust**（总修正对 pristine、边际对 proj-only、同噪声采样）
+ **固定 peak 预算** `b_j=clamp(1σ,0.01,0.10)` + 按 clip 聚类的 bootstrap CI + 分组
梯度/更新/token 诊断。200 步 pilot（`runs/three_track_v7/c1_lora_proj/pilot200`）：

| it | 25 | 50 | 75 | 100 | 125 | **150** | 175 | 200 |
|---|---|---|---|---|---|---|---|---|
| val ΔJ ×1e-3 | 5.04 | 3.85 | 4.60 | 5.20 | 5.24 | **6.07** | 5.75 | 5.65 |
| CI 下界 ×1e-3 | 3.53 | 2.44 | 3.18 | 3.43 | 3.18 | 4.20 | 4.01 | 4.02 |

* **收益 PASS**：test ΔJ **+5.68e-3**，CI **[+4.12, +7.12]e-3**（55 clip 簇）——旧配方要 650 步才到 +5.5e-3；
* **物理 FAIL（本轮目标）**：限位违反率 2.67–2.72%（pristine 2.77%），旧配方 2.45%；
  原因是总修正 trust 每步都在拉回（rms 15–47 mrad > τ=20 mrad）且 `L_lim` 是纯 hinge；
* 详见 `docs/C1_AFTER_REVIEW_PROBLEM_20261108.md`（含 P1–P8 问题、G1–G4 物理目标、5 个候选杠杆）。

---

## 3. 还没做（按任务书顺序，下一步就做这些）

| 顺序 | 事项 | 具体落点 |
|---|---|---|
| 1 | **真实 Isaac 短轨迹采集（Phase 1b）** | `scripts/vla_tick_recorder/`（monkeypatch 逐 tick 落盘）+ `scripts/collect_wm_refresh_isaac.py`；数据转换器已就绪（`make_segments_from_replay_v7.py`） |
| 2 | A 的 raw-norm λ 扫描 / C2 纯 LoRA / 分组 LR（Phase 4A/4C） | 现有训练器加参数即可（见 `docs/TRACK_ISSUES_PARAMS_RESULTS_20261108.md` §7） |
| 3 | 梯度真值探针（Phase 2b，用新数据跑） | 复用 `probe_wm_gradient_truth.py` plan/score + `isaac_probe_action_rollout.py` |
| 4 | Fixed-WM vs Refreshed-WM 对照（Phase 5） | `scripts/compare_actor_wm_versions_v7.py`（未写） |
| 5 | 周期 Refresh 调度（Phase 6） | `scripts/schedule_wm_refresh_v7.py`（未写） |

---

## 4. 本轮的负结果/教训（都是被门禁挡下来的）

1. **动作域门禁真的挡住了一次**：`a_w*.pt`（Supermimic policy 动作）与 A′（SONIC decoder）
   差 1.9×，15.36 万行里 24,012 行越界 ⇒ 拒绝。**如果没有这道门禁，WM 会把错误的动作 scale
   学进去**，而且从 loss 上看不出来（这正是任务书 §3.2 要防的）。
2. **k 步窗口的 episode 分组**：一开始按 `i % n_episodes` 平均分配，导致按 episode 聚类的 CI
   假性变窄（把同一条 20 步轨迹的 tick 当成独立样本）。已改成按**窗口起点行**反查 clip/segment，
   并加了单测 `test_kstep_windows_report_episode_groups_for_ci`。
3. **3 步候选的门禁结果**：`wm_probe_candidate`（3 次 WM 更新）在旧 A′ val 上 root 4.24cm → 4.06cm
   （rel +4.44%）但 **CI [-0.152, +0.136] 跨零** ⇒ 判 FAIL。说明"看起来变好"在小样本下不可信，
   CI 口径确实是必要的。

---

## 5. 复现命令（已落地部分）

```bash
source scripts/harness03_env.sh

# Phase 0
python3 scripts/freeze_v7_baseline.py --out runs/three_track_v7/phase0_freeze

# Phase 1a 单测
python3 -m pytest tests/test_wm_replay_buffer.py -q          # 12 passed

# Phase 2a（旧 A′ 基线）
python3 scripts/diagnose_wm_shift_v7.py \
  --wm-checkpoint $COMIMIC_MAIN/runs/a_prime_sonic_exc/wm_gain2.pt \
  --old-replay     $COMIMIC_MAIN/runs/a_prime_sonic_exc/all_transitions.pt \
  --horizons 1,5,10,20 --windows 128 --device cuda:1 \
  --old-ratio 1 --new-ratio 0 --out runs/three_track_v7/diagnostics/wm0_oldonly
```

> 所有 V7 产物只写 `runs/three_track_v7/`；旧目录一条都没改（Phase 0 只读校验过 sha）。

```bash
# Phase 3：候选 WM（3 步验证链路）+ 门禁（会判 FAIL）
python3 scripts/train_wm_refresh_v7.py --version-store runs/three_track_v7/wm_versions \
  --new-version wm_probe --old-replay $COMIMIC_MAIN/runs/a_prime_sonic_exc/all_transitions.pt \
  --old-ratio 1 --new-ratio 0 --mode both --k 20 --updates 3 --batch 64 --window-batch 8 \
  --device cuda:1 --npz-dir $COMIMIC_MAIN/data/npz/ha_tokens
python3 scripts/eval_wm_refresh_v7.py --version-store runs/three_track_v7/wm_versions \
  --wm-candidate wm_probe_candidate --old-replay $COMIMIC_MAIN/runs/a_prime_sonic_exc/all_transitions.pt \
  --old-ratio 1 --new-ratio 0 --windows 64 --device cuda:1 --skip-gradient \
  --out runs/three_track_v7/wm_versions/wm_probe_gate      # 期望 FAIL（缺新策略证据/梯度探针）

# 真实分片 -> segment buffer（动作域不兼容会被拒，见 §2.4）
python3 scripts/make_segments_from_replay_v7.py \
  --shards $COMIMIC_MAIN/runs/wm_real/a_w0.pt --policy-id A_line_frozenWBC \
  --horizon 20 --out runs/three_track_v7/buffer/aline_cycle000

# V7 全部单测（无 Isaac、无大权重）
python3 -m pytest tests/test_wm_replay_buffer.py tests/test_wm_version_store.py \
  tests/test_wm_refresh_gate.py -q        # 43 passed
```
