# WM6 全链路交接 · 给 GPT 的操作手册（2026-11-08 → 11-09）

> **本文位置**：主仓 `comimic/docs/`（与 `comimic-wt03/docs/` 同一份）。
> **实际工作区**：`/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03`（脚本、`runs/wm_base6_v1/`、`runs/wm_real/` 都在那里；主仓只读权重/资产）。


> 作者：容器 A 接手 Agent（DeepSeek）｜基线文档：`docs/HANDOVER_20261108_MASTER_A_WM6.md`（P0 合同 + 采集器）、`docs/HANDOVER_FULL_SESSION_20261108.md`（C 线 LoRA）
> 本文覆盖：**36 集 base 数据采集 → 合并复核 → WM1 训练 → 门禁 → dir_agree → P5-4 复评 → critic 接入**。
> 所有数字都可用文中命令复跑；产物都在 `runs/wm_base6_v1/` 与 `runs/wm_real/`（worktree 根 `comimic-wt03`）。

---

## 0. 五条铁律（先记住，能省你一天）

1. **合同唯一真相**：`state71` 的 q/qd 用 **STANDARD** 序；`action29` 用 **HA(SONIC) 序**、按 `(q_target_HA − dq_HA)/sc_HA` 归一化。喂 WM 时**原样喂**（不要置换）。详见 §3。
2. **running 脚本绝不原地编辑**（bash 按字节偏移继续读 ⇒ 必崩）。改配置 → 写新文件 → 重启；改库文件 → 只改不被运行中进程 import 的。
3. **环境**：`scripts/isaac_env.sh` 会改 `PYTHONPATH`，把主仓旧包 `diff_vla_wbc`（**缺 `wm_replay_buffer`**）顶到前面。**只在子 shell 里 source 它**（Isaac 子进程用），plan/score/训练用干净 PYTHONPATH：`$WT:$MAIN/trackerlab-.../source/trackerTask:$MAIN/trackerlab-.../source/trackerLab:$MAIN`。
4. **门禁口径不要混用**：主门禁用 `--val-fraction 0.2`（文档口径）；`0.9` 那版的新分布数字可用，但**旧分布数字因 val 与训练 clip 重叠而失效**。旧 A′ 有两个口径（官方 567 窗 vs 稠密 lag-16 40 万窗），**差 12 倍**。
5. **不许发布**：私有店 `runs/wm_base6_v1/wm_candidates/active.json` 与 `runs/three_track_v7/wm_versions/active.json` **都没动**。发布是人工决定。

---

## 1. 一分钟状态

| 项 | 状态 |
|---|---|
| 数据 | **36/36 集**、75,216 tick、逐集独立复核全过；merge **24 train / 6 val / 6 test**（2859/426/454 段）；**3739 段与 Raw 逐位核对 0 坏段** |
| 最佳候选 | `wm_base6_best` = **WM0 + 600 步长训的 s350**，sha `8edb59cf2adddf0a…` |
| 门禁 | **PASS 6/6**（新分布 **+38.59%**、旧官方退化 **+2.72%**、dir_agree **0.9792 ≥ 0.9271**） |
| 已发布？ | **否**（active 指针未动） |
| dir_agree | A′ holdout：**H=2 0.979 / H=5 0.990 / H=10 0.979（WM1）**；C-V8 H=20 只有 0.500 |
| P5-4（反号/抛硬币） | **未通过**：随机 91 窗上两 WM 都≈0.45–0.53（McNemar p=0.18）；**但 J10 优于 J20**（0.583 vs 0.458/0.500） |
| critic | **设计已定稿：H_train=10 + critic 在联合微调内共训（§10.5）**；已接入训练器（`--critic-kind token`，dry-run 通过：H=10/K=20/β=1.0/γ=0.95/`bootstrap_identifiable=True`），**尚未开训** |
| 主要未决 | ① critic 必须在联合微调内共训（A′ 离线 critic 跨分布失效，已证）② base6 无参考帧 ⇒ 收益类判据不能用 ③ 门禁新分布只落在 1 张 val 集上 |

---

## 2. 系统与名词

```
图像+语言+state46 →[VLA Gr00tN1d7 checkpoint-8000]→ 64-d token chunk(T=40)
   →[SONIC release decoder]→ q_target(29,rad,HA序) @50Hz，每 K=20 tick 重规划
   →叠加 Δ(1.744M, cap 0.08 rad) 或 LoRA16
   →[WM: DynamicsMLP 8.82M] 当可微裁判，H 步 BPTT
```

| 名词 | 含义 |
|---|---|
| **WM0** | `runs/three_track_v7/wm_versions/wm_v000/wm.pt`，sha `fc5aeb30…`，冻结参考 |
| **WM1 / `wm_base6_best`** | WM0 在新 base 数据上 600 步 refresh 后的 s350 检查点 |
| **A′** | `runs/a_prime_sonic_exc/`：50 shard、529,131 转移、WBC-only、带 `frames`（全局库行号） |
| **base6** | 本轮 VLA 闭环采集的 36 集（6 任务 × 6 seed），**无参考帧** |
| **e20** | 自回归 20 步的 root 位置误差（m）；门禁主指标 |
| **dir_agree** | 沿 WM 梯度走 ±δ，真物理"变好"的比例（符号一致率） |
| **ΔJ / J** | tracking reward 的 H 步未折扣和（不带 γ、不带 survival） |
| **K / H** | K=每次 chunk 执行步数（=exec_k=20）；H=可微 WM 步数（原 20，建议 10） |
| **v_ev** | critic 的留出集解释度 `1 − MSE/Var(target)`，门限 >0.30 |

---

## 3. P0 动作合同（唯一真相）

```
state71 = [ root_pos(3) | root_quat_wxyz(4) | q_STANDARD(29) | v_w(3) | w_w(3) | qd_STANDARD(29) ]
action29_model = (q_target_HA − dq_HA) / sc_HA           dt = 0.02 s
PERM_HA_TO_STD = [0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28]
PERM_STD_TO_HA = [0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28]
dq_STD = [-0.2,0,0,0.42,-0.23,0, -0.2,0,0,0.42,-0.23,0, 0,0,0, 0.35,0.18,0,0.87,0,0,0, 0.35,-0.18,0,0.87,0,0,0]
sc 均值 0.3789（逐关节从 WBC robot_spec 取，记录器写进 tick meta 的 dq_ha/sc_ha）
```

**独立复核（我自己重跑，不复用旧结论）**：A′ 64 窗口 × 20 步，冻结 WM0：

| 组合 | root e1 | root e20 |
|---|---|---|
| **正确（state 原样 + action 原样）** | **0.298 cm** | **3.79 cm** |
| 状态换序 HA→STD | 0.77 cm | 10.95 cm（2.9×） |
| 状态换序 STD→HA | 0.78 cm | 12.10 cm（3.2×） |
| 动作换序 | 0.65 cm | 18.54 cm（4.9×） |

命令：`python3 scripts/supervise_check_wm0_contract.py --windows 64 --device cuda:1`
产物：`supervision/wm0_contract_check.json`。（文档原表：5.63 / 15.39–18.79 / 12.75 cm，量级一致。）

**为什么必须原样喂**：`bundle.step()` 内部已处理动作/状态序；再把 state 或 action 置换一次就等于换错合同，20 步误差翻 3–5 倍。

---

## 4. 数据

### 4.1 分工与逐集结果（36/36）

| 任务 | 采集方 | 成功/超时 | 备注 |
|---|---|---|---|
| open_door | A | **4 / 2** | |
| sit_sofa | A | **4 / 2** | s5 为 592 tick 短集 |
| vision_navi | A | **4 / 2** | 本会话最好（历史 2–3/6） |
| pp_box | B | **2 / 4** | 与基线 2/6 一致 |
| double_desk | B | **0 / 6** | 与基线 0/6 一致（base 做不到，Δ 的靶子） |
| **boxing** | **A 接手**（B 未采，避免重复） | **0 / 6** | 6 集全 3000 步超时；A 于 17:37 接手、18:02 收尾 |

A/B 两容器合同元数据逐项一致（contract / 两个置换表 / dq_ha / sc_ha / `git_commit=38a2bf0` / decoder sha `c7241a12` / `policy_id=base_ft8000`）。

### 4.2 采集器（唯一入口）

```bash
cd /inspire/.../comimic-wt03
python3 scripts/collect_base_wm_v2.py --tasks boxing --seeds 0,1,2,3,4,5 \
  --max-steps 3000 --sim-gpu 0 --vla-gpu 1 --video off --port 18220 \
  --out-root runs/wm_base6_v1/collector_A
# 另一个 worker 复用 server：加 --reuse-server（同端口）；B 容器用 18222 / collector_B
```
产物：`<root>/<task>/seed<k>/{ticks/ticks_<pid>_<chunk>.pt, episode.log, episode.json, <HA_TASK>__seed<k>__repeat0__<stamp>/result.json}`。
每 tick **15 字段**：`states_before(71) states_after(71) actions_model(29) actions_raw(29) q_target(29) q_applied(29) task_id episode_uid seed tick frame_count dt done done_reason meta{}`。

**坑（已踩）**：
- `tick` 字段是**分块局部序号**（0..15），全局时序只能靠 `frame_count`；任何"按 tick 排序/去重"的下游会压掉数据。
- `result.json` 在**嵌套目录**，所以采集器的 `finalize_ticks()`（找 `seed<k>/result.json`）拿不到 ⇒ `done_reason` 全落 `unknown`。
  我用 `scripts/annotate_episode_outcome.py` 事后回填（只写 `episode.json`，**不改 Raw tick**）。
- 单集墙钟：快速成功 1.5–4 min；3000 步超时 10–12 min；`--video on` 约 +3–4×。

### 4.3 复核

`supervise_wm6_snapshot.py`（独立实现，不复用审计代码）逐集检查：15 字段、`frame_count` 逐 tick +1、`states_after[t]==states_before[t+1]`（实测 0.0）、四元数范数、动作公式 `(q_target−dq_ha)/sc_ha`、`done` 回填、dt 唯一值、速度自洽比、动作域。**A 18/18 + B 18/18 全 ✓**。

---

## 5. 合并 / 切分 / 复核工具链

```bash
bash scripts/run_wm_base6_pipeline.sh merge     # = audit + merge（同时扫 collector_A/B）
python3 scripts/verify_merged_wm6.py --all      # merged↔Raw 逐位核对（关键！）
python3 scripts/wm6_coverage.py                 # 六任务覆盖度（exit 2=不达标）
python3 scripts/annotate_episode_outcome.py     # 回填 done_reason/success
```

- **切分规则**（`merge_base_wm_v2.py`）：每任务按 seed 升序，**最后两个 = test, val**，其余 train；窗口 `stride=H=20`、不跨 episode。
  与 `audit_base_wm_v2.py` 的哈希切分**不是一回事**（审计里的 `split` 字段只作建议，以 merged manifest 为准）。
- **复核内容**：`states/actions/next_states/state_final` 与 Raw 逐位一致、窗口不跨集、窗口内 frame 连续、split 无交叠、与 manifest 一致。**结果 3739/3739，0 坏段**。
- 产物：`runs/wm_base6_v1/merged/{train,val,test}/segments.pt`、`merged/flat/<task>__<split>.pt`、`merged/manifest.json`、`supervision/merged_verify.json`。

---

## 6. WM 训练（P4）

### 6.1 配方（实测有效）

```
init = WM0(fc5aeb30)   A′:base = 80:20   mode=both   k=20   lr=1e-5
batch=256   window_batch=32（kstep 对 wb 不敏感）   eval_every=25   save_every=50
updates=600 → 12 个检查点（35 MB/个）
```
命令：`bash scripts/run_wm6_stage.sh`（环境变量 `VER / LR / UPDATES / WB / SAVE_EVERY / OLD_RATIO`）。

### 6.2 提速：采样池化（**必开**，20×）

`MixedReplay.sample_windows()` 每步都用 `contiguous_runs()` 对 A′ 44 万行做**纯 Python 循环**并重建窗口池 ⇒ 实测 **13.1 s/次**；而 20 步 BPTT 只要 0.66 s。
加 `WM6_WINDOW_CACHE=1`（`diff_vla_wbc/wm_replay_buffer.py`，默认关闭，不影响其它线）后 **0.3 ms/次**，单步 **10.6 s → 0.52 s**，600 步 **≈5.5 min**。

**顺带发现的数据事实**：A′ 能被 `contiguous_runs` 切出的合法 20 步窗口只有 **567 个**（1.3 窗/clip，99,090 个短段、平均 4.3 行），而按真实 `(step-major, env-minor, num_envs=16)` 布局的 **lag=16 轴上有 398,946 个**。
⇒ kstep 项长期只在 567 个窗口上重复采样（这解释了 kstep loss 卡在 ~5.5）。**改进点：加 env-aware(lag=16) 窗口构造器**（尚未做，见 §13）。

### 6.3 检查点与选优（双口径）

`scripts/select_wm6_checkpoint_v2.py`（**必须用 v2**）：

| 口径 | 窗口来源 | 池 | WM0 | s350 |
|---|---|---|---|---|
| **旧-官方**（=门禁） | `contiguous_runs` × clip 哈希 holdout | 567 | 4.60 cm | +2.3% ✅ |
| 旧-稠密 | lag=16 × holdout clips | 40 万候选 | 5.62 cm | **+26.4%** ⚠️ |
| 新分布 | `merged/val` 全段 | 426 | 8.26 cm | **+48.3%** ✅ |

选优规则：**在旧-官方退化 ≤5% 的检查点里取新分布 e20 最小**。v1 选优器用"含训练 clip 的稠密窗口"，曾报"旧退化 +33%"——**已弃用**（与门禁差 12 倍）。

### 6.4 候选清单

| 版本 | 来源 | 新 e20 | 旧-官方 | 旧-稠密 | 门禁 |
|---|---|---|---|---|---|
| `wm_base6_best_candidate` | 600 步 s350，sha `8edb59cf` | 4.27 cm (+48.3%) | +2.3% | +26.4% | **PASS** |
| `wm_base6_150_candidate` | s150（文档 P4 原口径） | 5.40 cm (+34.6%) | +2.9% | +19.8% | 未单独跑 |
| `wm_base6_lr2e6_best_candidate` | lr 2e-6 × 200 步 s200，sha `89f42289` | 5.73 cm (+30.6%) | **−1.6%** | **+9.5%** | **FAIL**（新分布 ci_lo −5.9%） |
| `wm_base6_001_candidate` | 长训终态（s600） | 4.27 cm | +2.5% | +31.8% | 未单独跑 |

---

## 7. 门禁（P5）

命令（**主口径**，带 dir_agree 探针）：

```bash
python3 scripts/eval_wm_refresh_v7.py --version-store runs/wm_base6_v1/wm_candidates \
  --wm-old init_from_wm0 --wm-candidate wm_base6_best_candidate \
  --old-replay runs/a_prime_sonic_exc/all_transitions.pt \
  --onpolicy-data runs/wm_base6_v1/merged/val --old-ratio 0.8 --new-ratio 0.2 \
  --horizons 1,5,10,20 --primary-h 20 --windows 128 --device cuda:1 \
  --gradient-probe-result runs/wm_base6_v1/supervision/gradient_probe.json \
  --out runs/wm_base6_v1/gates/final
```

**`wm_base6_best` 结果（PASS 6/6）**：`gates/wm_base6_best_eval_with_probe/gate.md`

| 判据 | 值 | 阈值 |
|---|---|---|
| finite / action_contract / no_leakage | ok | True |
| new_e20_rel_improve | **+38.59%**（ci_lo +7.36%） | ≥5% 且 ci_lo>0 |
| old_e20_rel_degrade | **+2.72%** | ≤5% |
| dir_agree_new ≥ dir_agree_old | **0.9792 ≥ 0.9271** | ≥ |

逐任务 val（e20 8.26→4.27 cm）：vision_navi +60.0%、double_desk +48.9%、sit_sofa +46.6%、boxing +45.6%、pp_box +39.8%、open_door +38.2%（`supervision/per_task_val.json`）。

### ⚠️ 三个口径陷阱（必须知道）

1. **新分布指标只落在 1 张 val 集上**：`MixedReplay` 内部再按哈希切 20% ⇒ 6 张 val 只剩 1–2 张（日志 `eps=1`）。A 的 +38.59%（ci_lo +7.36%）成立；B 的 +18.9%（ci_lo **−5.9%**）就被判 FAIL——**这是尺子问题，不是模型变差**。
2. **`--val-fraction 0.9` 版**（5 张集）：新分布 **+43.70%**、ci [+0.322,+0.487] 可信；但它的**旧分布数字（+32% "改善"）失效**——0.9 时旧 val 与训练 clip 重叠。
3. **`--skip-gradient` 时门禁必 FAIL**（缺 dir_agree），只能当诊断。

---

## 8. dir_agree（P5-3）：方法 + H 依赖（本轮核心结论之一）

### 8.1 方法（`scripts/probe_wm6_dir_agree.py`）

在 **A′ clip 哈希 holdout** 上取 H 步窗口 `(s0, a0..a_{H-1}, frames)`：
1. WM 侧：翻转 `J_WM = Σ_t bundle.reward(step(s), a_t, a_{t-1}, FrameRef(frames_t))`，得 `g = ∂J_WM/∂a`；
2. 归一化到 `|g|_rms=1`，取 `a± = a0 ± δ·ĝ`（δ=0.03 rad RMS）；
3. **精确换算**成 Isaac 口径：`q_target_ha = a_ha·sc_ha + dq_ha → q_target_std = q_target_ha[PERM_HA_TO_STD] → a_std = (q_target_std − dq_std)/sc_std`；state 反过来 `q_ha = q_std[PERM_STD_TO_HA]`；
4. Isaac `isaac_probe_action_rollout.py` 还原**完整 71 维**（root pose+vel+joint）后各滚 H 步；
5. 用同一个 `bundle.reward` 算 `J_true(±)` → 符号一致率 = dir_agree。

### 8.2 结果（A′ holdout，96 窗口）

| H | WM0 | WM1 | ΔJ_true 均值 | ΔJ_WM（WM0/WM1） |
|---|---|---|---|---|
| 2 | 0.927 | **0.979** | +0.0008 | +0.0110 / +0.0099 |
| **5** | **0.990** | **0.990** | +0.0042 | +0.0301 / +0.0281 |
| **10** | **0.979** | 0.969 | +0.0138 / 0.0130 | +0.0693 / 0.0629 |
| 20（C-V8 口径） | 0.458 | 0.500 | −0.0073 | +0.0095 / +0.0011 |

**结论**：**H≤10 时 WM 梯度方向可信（0.97–0.99），H=20 崩到 ~0.5**；WM1 不差于 WM0（H=2 更好）。

### 8.3 C-V8 那 24 窗的"截断 H"扫描（用现有 roll，无需重跑）

| H | WM0 符号率 | WM1 | 相关（WM0/WM1） |
|---|---|---|---|
| 2 | 0.833 | 0.667 | 0.17 / 0.19 |
| 5 | 0.667 | 0.542 | **0.64 / 0.72** |
| 10 | 0.667 | 0.667 | 0.57 / 0.46 |
| 20 | 0.458 | 0.500 | 0.26 / 0.13 |

⇒ H 降低确实改善相关（0.13→0.72@H5），但那些窗**在 H≤5 时真值差只剩 ±0.0006**（噪声底），所以 0.667 不能当"判别力"。

---

## 9. P5-4（"反号 / 抛硬币"）：**未通过**

### 9.1 原始结论的来历

`runs/three_track_v8/c_tracking_recovery/isaac_h20/`：24 窗 = **每任务按 `cos_used` 取 4 低 + 4 高**（`probe_c_vs_base_isaac_plan.py:19,111-113`）⇒ **刻意挑的极端错位子集**；比较对象是 C adapter 的动作 vs Base 动作；真物理 `dJ_Isaac` 均值 −0.00733，WM 预测 +0.00949，符号一致率 0.458。

### 9.2 随机抽样复评（本轮新增）

用 `--select random --per-task 32 --split-key {test,val}` 把这两分片里属于 3 个任务的窗口**全部**纳入（45 + 46 = 91 窗，H=10）：

| 数据集 | 真值 | WM0 | WM1 |
|---|---|---|---|
| test 45 窗 | ΔJ −0.00232，变好 33%，\|ΔJ\|中位 0.0059 | **0.556** CI[0.400,0.711]，corr +0.358，Spearman +0.378 | **0.444** CI[0.311,0.600]，corr +0.300 |
| val 46 窗 | ΔJ −0.00255，变好 30%，中位 0.0047 | **0.500** CI[0.348,0.652] | **0.457** CI[0.326,0.609] |

- **配对检验（45 窗）**：只有 WM0 对 7 个、只有 WM1 对 2 个 ⇒ **McNemar p=0.180**，两者**无显著差异**；两 WM 的 ΔJ 相关 **+0.705**。
- 去任务偏置后：WM0 0.622 / WM1 0.556；每任务仿射校准（in-sample 上限）：0.733 / 0.644。
- **幅度校准**：过原点最小二乘 `c` = 0.267（WM0）/ 0.568（WM1），MAE 7.6→6.8 mm / 7.1→6.9 mm。**校准对符号率零帮助**（c>0 不改符号），只对 BPTT 权重尺度有用。注意 **H=10 随机窗是"低估"幅度（0.4–0.8×）**，与 H=20 C-V8 的"高估 5–15×"方向相反。
- **置信门限（可用性口径）**：

| 只在 \|c·ΔJ_wm\| > τ 时判断 | WM0 | WM1 |
|---|---|---|
| τ=0.25×中位\|ΔJ_true\| | 覆盖 36% / 精确率 **0.688** | 49% / 0.545 |
| τ=0.50× | 11% / **0.800** | 20% / **0.667** |
| τ=1.00× | — | 4% / **1.000**（2 窗） |

**结论**：WM（不管 0 还是 1）**不能当"任意两串动作谁更好"的比较器**（≈抛硬币）；只能当**带弃权的粗筛**（最自信 11–20% 的窗口精确率 0.67–0.80）。而"WM 梯度方向"这个**本职**指标是 0.97–0.99 —— 两个问题必须分开看。

### 9.3 复现命令

```bash
# 1) 生成 job（随机选窗 + 解码 C/B 两串动作；需 harness03_env：tyro 等）
source scripts/harness03_env.sh
python3 scripts/probe_c_vs_base_isaac_plan.py --select random --per-task 32 --H 10 \
  --split-key test --out runs/wm_base6_v1/dir_agree/p54_job_rand96_h10.pt --device cuda:0
# 2) 真物理 rollout（Isaac；注意保存后会挂住 app.close()，见 §12）
( source scripts/isaac_env.sh; $ISAAC_VENV/bin/python -u scripts/isaac_probe_action_rollout.py \
    --job .../p54_job_rand96_h10.pt --out .../p54_roll_rand96_h10.pt --device cuda:1 --headless )
# 3) 打分 + 校准 + 置信门限
python3 scripts/p54_rank_calib.py --job ... --roll ... --out supervision/p54_calib_test45.json
```

---

## 10. Critic（本轮已接入，未训练）

### 10.1 两套现成结构（别混）

| 结构 | 输入 | 用于 |
|---|---|---|
| `diff_vla_wbc/token_critic.py::TokenCritic` | `V(state71, ref71, xi33)`，~1.1e5 参数 | `fit_critic_real.py` 离线拟合；本轮接入用这个 |
| `diff_vla_wbc/imagined_value_v5.py::ImaginedValueV5` + `CriticFeatureAdapter` | state+ref+diff+xi+hist10+plan(4×64)+extra(4) | `train_temporal_ac_bptt_v5.py` 训练内默认（`--critic-kind v5`） |

两者都注释了同一条纪律：**critic 不吃 token**（价值 = "在这个状态/参考下往后能拿多少"），并且**必须配合 WM 保鲜**（WM 一变，V 就偏）。

### 10.2 离线 critic 的完整管线（已跑通，产物在盘）

```bash
# 1) A′ 逐行奖励（用训练同款 reward 口径）
python3 scripts/compute_collect_reward.py --in runs/a_prime_sonic_exc/*_off*.pt \
  --out runs/a_prime_sonic_exc/rewards_current --from-args ../comimic/runs/bptt1/args.json --device cuda:0
# 2) MC return-to-go + ref_rows（自动推断 stride=16，跳过 5 个连续率过低的分片）
python3 scripts/build_critic_dataset.py --shards runs/a_prime_sonic_exc \
  --rewards runs/a_prime_sonic_exc/rewards_current --packed ../comimic/runs/isaac_motion/ha_cmd/packed.pt \
  --out runs/wm_real/transitions_a_prime.pt --gamma 0.95 --window 20
# 3) 拟合 + 留出集门禁（v_ev > 0.30）
python3 scripts/fit_critic_real.py --transitions runs/wm_real/transitions_a_prime.pt \
  --out runs/wm_real/critic_a_prime.pt --center linear --epochs 12 --device cuda:0
```
结果：**502,636 行 / 1369 段**；reward mean +0.0101 std 0.0416；ret mean +0.193；
**v_ev = +0.3507（留出 n=96,357，门限 0.30 ⇒ PASS）**，`critic_a_prime.pt`（hidden 256 / depth 2 / use_diff=false / γ 0.95 / center=linear）。

### 10.3 复合裁判实测：**离线 critic 跨分布失效（重要负结果）**

`scripts/p54_composite_judge.py`：`J_comp = J10(WM) + γ¹⁰·(V(s₁₀) − V(ref₁₀))`，在 C-V8 24 窗（真值 H=20）上：

| WM | J10 单独 | J10 + critic | J20（现状） |
|---|---|---|---|
| WM0 | **0.583**（corr +0.489） | 0.542（corr +0.271） | 0.458（corr +0.257） |
| WM1 | **0.583**（corr +0.285） | 0.458（corr +0.096） | 0.500（corr +0.132） |

⇒ **加 critic 反而变差**。原因：critic 是在 **A′ 分布**上拟合的（v_ev 0.35 也是 A′ 留出集），C-V8 是闭环 VLA 分布；且目标做了 `--center linear` 去趋势，跨分布后残差主导。
**同时得到一个正面结论：J10 单独用就是三者最好**，再次支持 H=10。

### 10.4 正确用法（本轮已在训练器里接好线）

**结构规则（代码强制）**：`bootstrap_identifiable(K,H) = K > H`；现在 `K = min(exec_k=20, T=40) = 20`。
- `H=20` ⇒ K==H ⇒ **β 强制 0**、日志 `BOOTSTRAP_NOT_IDENTIFIABLE`（这就是 critic 一直关着的原因）；
- `H=10` ⇒ K>H ⇒ 可辨识，critic 负责 10→20 步尾段。

**本轮对 `scripts/train_temporal_ac_bptt_v5.py` 的接线（最小侵入，v5 路径不变）**：

| 新增参数 | 作用 |
|---|---|
| `--critic-kind {v5,token}` | `token` = 用 `TokenCritic(state,ref,xi)`（与 `fit_critic_real.py` 同结构） |
| `--critic-ckpt PATH` | token critic 热启动（如 `runs/wm_real/critic_a_prime.pt`） |
| `--critic-hidden/--critic-depth/--critic-use-diff` | token critic 形状 |
| `--critic-gamma` | **critic 目标与自举共用 γ**（必须与构建 MC 目标时一致；默认 `GAMMA=0.99`，本轮实验用 0.95） |
| `--dry-run` | 只建 bundle/数据/actor/critic、写 `critic_gate.json` 后退出，**不训练** |

自举：`J_ac = J + β·γ^H·survival[H]·V(s_H, ref_H, xi)`；critic 更新：`td_lambda_targets(γ=critic_gamma, λ=0.95)` + EMA(0.995) target。
**dry-run 已通过**：

```
[v5][critic] 热启动 token critic ← runs/wm_real/critic_a_prime.pt
[v5][critic] enabled(kind=token) beta=1.0 params=111,105 K=20 H=10 γ=0.95
[v5][dry-run] 接线自检完成：H=10 K=20 β=1.0 critic=on(token) γ=0.95 bootstrap_identifiable=True ⇒ 不训练，退出
```
（产物 `runs/wm_base6_v1/supervision/…`/`/tmp/v5_dryrun/critic_gate.json`；rc=0）

**正式开训的命令模板（等指令，不自动跑）**：

```bash
source scripts/harness03_env.sh
python3 scripts/train_temporal_ac_bptt_v5.py --out runs/three_track_v8/c_tracking_recovery/critic_h10/seed0 \
  --horizon 10 --exec-k 20 --beta 1.0 --beta-warmup-updates 100 \
  --critic-enabled 1 --critic-kind token --critic-gamma 0.99 \
  --critic-ckpt runs/wm_real/critic_a_prime.pt \
  --on-the-fly 1 --updates 200 --batch 256 --seed 0 --device cuda:1 \
  --eval-set runs/temporal_ac_bptt_v5/eval_set
```
**注意**：离线 critic 的 γ/center 与新目标必须对齐（γ=0.99 就要重建 MC 目标，别沿用 0.95 的 ckpt 直接开训；可以当热启动，但标签口径要一致）。

### 10.5 设计定稿：**H_train = 10 + critic 必须"联合微调内共训"**

#### 10.5.1 为什么 H_train 取 10（三条独立证据）

1. **梯度方向**：`dir_agree` 在 H≤10 时 0.93–0.99，H=20 崩到 0.46–0.50（§8.2）。
2. **模型精度**：同一条 episode 内 20 步 root 误差 WM1 = 4.27 cm；拆到 H=10 只有 **2.10 cm**，H=5 是 1.10 cm（实测逐 H 曲线）。
3. **复合裁判对照**（C-V8 24 窗，真值 H=20）：**J10 单独 = 0.583 / corr +0.489，优于 J20 的 0.458/+0.257，也优于 J10+离线critic**（§10.3）。

⇒ **可微部分只用 10 步**；10→20 步的信用分配交给价值函数（critic）。结构上必须 `K > H`（现 K=min(exec_k=20, T=40)=20），代码已强制。

#### 10.5.2 为什么 critic **必须**在联合微调里共训（而不是离线拟合后冻结）

**（a）机制上：V 是策略相关的。** `V^π(s)` 依赖"当前策略往后会做什么"。微调过程中策略在漂移，离线 V 立刻变成 off-policy 的旧值；而 BPTT 的自举项 `β·γ^H·V(s_H)` 会把这个偏差**放大到动作梯度里**。

**（b）本轮实测（负结果，已复现）：** 用 A′ 语料 + 真奖励离线拟合的 `critic_a_prime.pt`，在**自己的留出集**上 `v_ev = +0.3507`（门限 >0.30，PASS），但拿它给 C-V8 分布做复合裁判，**反而比不用 critic 更差**：

| WM | J10 单独 | J10 + 离线 critic | J20 |
|---|---|---|---|
| WM0 | **0.583**（corr +0.489） | 0.542（corr +0.271） | 0.458（corr +0.257） |
| WM1 | **0.583**（corr +0.285） | 0.458（corr +0.096） | 0.500（corr +0.132） |

**（c）根因量化（24 窗，分解到"绝对值 / 分支差"）：**

| | H=10 | H=20 |
|---|---|---|
| 真物理分支差 \|ΔJ\| 中位 | 0.0060 | 0.0146 |
| 真物理绝对 J 水平 | ≈ **0.166** | ≈ 0.306 |
| WM0 绝对 J 预测 / 分支绝对误差 | 0.517 / 0.353 | 1.080 / 0.789 |
| WM1 绝对 J 预测 / 分支绝对误差 | 0.549 / **0.383** | 1.184 / **0.881** |
| WM0 差分 MAE / 偏置 / SNR / corr | 0.0065 / +0.0044 / 0.92 / +0.574 | 0.0234 / +0.0168 / 0.62 / +0.257 |
| WM1 差分 MAE / 偏置 / SNR / corr | 0.0062 / **+0.0019** / 0.97 / +0.456 | 0.0187 / **+0.0084** / 0.78 / +0.132 |

读法：**分支差只有绝对 J 的 0.4–2%，而绝对误差是它的 20–100 倍** ⇒ 排序只能靠"两条分支的误差是公共模式、相减抵消"。WM1 在这批**分布外动作**上绝对误差更大（0.383 vs 0.353）、更"乐观"（预测 0.549 vs 真 0.166），抵消得更差 ⇒ 相关性掉一半（但偏置小 2.3 倍）。**离线 critic 学的是 A′ 的 V，在这批 VLA 闭环动作上同样不抵消** ⇒ 加到 J10 上只增加噪声。

**（d）仓库既有纪律（照抄即可）**：`token_critic.py` 注释写明"critic 不吃 token""critic 会把 WM 偏差自举进去，必须配合 WM 保鲜"；`build_critic_return_targets.py` 写明"扰动起点上没有真值，唯一干净做法是拿控制器**真走过**的轨迹算 MC return-to-go"。
⇒ **两条硬规矩**：① critic 的标签只能来自**当前策略真实轨迹**；② **WM 每次 refresh 后 critic 必须继续共训/重训**。

#### 10.5.3 标准配方（等指令再跑，不要自动开）

```bash
source scripts/harness03_env.sh
OUT=runs/three_track_v8/c_tracking_recovery/h10_critic/seed0
# 1) 用"当前策略的真轨迹 + 同一 reward 口径"重建 MC 目标（γ 必须与下面 --critic-gamma 一致）
python3 scripts/compute_collect_reward.py --in <该轮 rollout dump>/*_r.pt \
  --out <dump>/rewards_current --from-args ../comimic/runs/bptt1/args.json
python3 scripts/build_critic_dataset.py --shards <dump> --rewards <dump>/rewards_current \
  --packed ../comimic/runs/isaac_motion/ha_cmd/packed.pt \
  --out runs/wm_real/transitions_policy.pt --gamma 0.99 --window 20
python3 scripts/fit_critic_real.py --transitions runs/wm_real/transitions_policy.pt \
  --out runs/wm_real/critic_policy.pt --center linear --epochs 12      # 仅作热启动
# 2) 联合微调：H=10（可微）+ K=20（执行/自举），critic 在线共训 + β 爬坡
python3 scripts/train_temporal_ac_bptt_v5.py --out $OUT \
  --horizon 10 --exec-k 20 \
  --critic-enabled 1 --critic-kind token --critic-ckpt runs/wm_real/critic_policy.pt \
  --critic-gamma 0.99 --beta 1.0 --beta-warmup-updates 100 \
  --critic-lr 3e-4 --critic-updates 4 --critic-batch 2048 \
  --on-the-fly 1 --updates 200 --batch 256 --seed 0 --device cuda:1 \
  --eval-set runs/temporal_ac_bptt_v5/eval_set
# 3) 只验证接线不训练：
#    同上 + --dry-run   ⇒ 写 critic_gate.json（K=20 / H=10 / γ / bootstrap_identifiable）
```

**必须对齐的三件事**（否则 β 放大的是偏差，不是信号）：
| 项 | 要求 |
|---|---|
| γ | `build_critic_dataset --gamma` == `--critic-gamma`（本轮实验用 0.95；换成 0.99 就要重建目标） |
| 去趋势 | 标签 `--center linear` ⇒ 自举用**相对形式** `V(s_H) − V(ref_H)` |
| 分布 | 标签/共训数据 == 策略当前产生的动作分布（不是 A′、不是 base6 的旧采样） |

#### 10.5.4 放行条件（不满足就回退，不许硬上）

1. critic 在**其训练分布的留出 episode** 上 `v_ev > 0.30`；
2. 在 **≥96 窗、H=20 真值**上，`J10 + γ^H(V(s_H)−V(ref_H))` 的**符号率与相关都优于 J10 单独**（带 95% CI，且 CI 不跨 0 的改善）；
3. `dir_agree(H=10) ≥ 0.65`（现有 0.97）不退化；
4. A/A 噪声底不变（同一 job 跑两次的 ΔJ std 不应因 critic 变大）。

⇒ 三条不满足，就**保持 `β=0`（现状）**，把 WM 只当"短程梯度裁判"用，并在报告里写明"收益排序不可用"。

---

## 11. 结论总表（哪些成立 / 哪些被推翻）

| 结论 | 状态 | 证据 |
|---|---|---|
| WM1 状态预测比 WM0 好 ~50%（每个 H 都是） | ✅ 成立 | H1 0.17 vs 0.32 cm … H20 4.27 vs 8.26 cm |
| WM1 梯度方向不差于 WM0（H≤10，0.97–0.99） | ✅ 成立 | A′ holdout 96 窗 |
| H=20 的梯度方向不可信（~0.46–0.50） | ✅ 成立 | A′ H 扫描 + C-V8 |
| **"H 降到 10 就能解决收益排序/反号"** | ❌ **被推翻** | 随机 91 窗符号率仍 0.44–0.56 |
| **"旧 A′ 退化 33%"** | ❌ **被推翻（口径错）** | 官方 holdout 只有 +2.72%；稠密口径 +26% 是另一回事 |
| **"WM 幅度高估 5–15×"** | ⚠️ 只在 H=20/C-V8 成立；**H=10 随机窗反而低估** | c=0.27/0.57 且为收缩 |
| "C-V8 那 24 窗能代表 C adapter 好坏" | ❌ 极端 cos 子集，不可外推 | 抽样方式在 plan 脚本里 |
| 离线 critic（A′）能提升收益排序 | ❌ 反了（0.583→0.54/0.46） | 跨分布 |
| 加 critic 必须联合微调内共训 + K>H | ✅ 结构上强制、实测支持 | `bootstrap_identifiable` + 本轮负结果 |
| base6 的 tick 无参考帧 ⇒ 收益类判据只能 A′ 上算 | ✅ 事实 | `frames` 缺失 |
| **WM1 在"收益排序"上不比 WM0 好（相关性还更差）** | ⚠️ 成立（0.574→0.456 @H=10） | 机制：分支差 0.4–2% × 绝对误差 20–100 倍；WM1 分布外更乐观（J 0.549 vs 真 0.166） |
| **"H=10 + critic 在联合微调内共训"是唯一可行路径** | ✅ 设计定稿（§10.5） | 离线 critic 跨分布实测变差 + `bootstrap_identifiable(K>H)` |

---

## 12. 坑与事故（都是真踩过的）

| # | 现象 | 根因 | 正确做法 |
|---|---|---|---|
| 1 | v3 监督脚本 18:03 突然 `break: only meaningful in a loop` / `syntax error near fi` 退出 | **在 bash 运行时用 python 原地改写同一脚本**，偏移失效 | 改配置写新文件 + 重启；不原地编辑运行中脚本 |
| 2 | `pkill -f "job_old_h5"` 把自己也杀了（SIGTERM） | `pkill -f` 匹配到**本 shell 自己的命令行**（模式串出现在命令里） | 用 `pgrep` 拿 PID 再 `kill`，或模式串拼接避免自匹配（`"isaacsim_env""/bin/python"`） |
| 3 | `ModuleNotFoundError: No module named 'diff_vla_wbc.wm_replay_buffer'` | source `isaac_env.sh`/`harness03_env.sh` 把主仓（旧包）顶到 `PYTHONPATH` 前面 | Isaac 只在**子 shell** source；plan/score/训练用干净 PYTHONPATH（wt03 第一） |
| 4 | `No module named 'tyro'`（载 FT8000 时） | 需要 harness 依赖环境 | `source scripts/harness03_env.sh` 再跑解码类脚本 |
| 5 | Isaac rollout 保存 `roll.pt` 后进程**永不退出**（挂 20+ min） | `app.close()` 卡住 | "输出文件出现 + sleep 6 → kill（先 TERM 再 KILL）"，别无限等 |
| 6 | `d03.load_env()` 报"找不到 motion npz" | 没传 `--bptt-args`（`motion` 路径在里面） | 一律传 `--bptt-args ../comimic/runs/bptt1/args.json` |
| 7 | `MixedReplay` 报"动作越出旧 A′ 域" | `MixedReplay.__init__` 默认 `action_max_outside_frac=0.0`（严格），而训练器 CLI 默认 0.005 | 自写脚本调用时显式传 `action_max_outside_frac=0.005` |
| 8 | 预演训练偶发 `loss=830/57`（gn 878） | 新数据里的极端姿态批次 | 已由 `grad_clip=1.0` 兜住；建议后续加异常段过滤 |
| 9 | `--skip-gradient` 门禁恒 FAIL | 缺 dir_agree | 只在诊断时用；发布必须 `--gradient-probe-result` |
| 10 | 门禁新分布 `eps=1` 的 CI 抖动 | `MixedReplay` 内部再切 20% | 报告里同时给 `--val-fraction 0.9` 的版本；或改进 eval 不用二次切分 |

---

## 13. 给 GPT 的下一步（按优先级，含放行条件）

### 路线 A（推荐先做）：把"尺子"做扎实
1. **P5-4 扩样到 91+ 窗的 H=20 真值**：给已有 `p54_job_rand96_h10.pt` / `..._val_h10.pt` 重跑 **H=20** rollout（约 10 min），然后 `p54_rank_calib.py` 用 H=20 真值出 `J10 / J10+critic / J20` 三口径 + 95% CI。
   **放行条件**：若 `J10` 在 n≥91 上仍 ≥0.60 且 CI 下界 >0.5，才算"短程裁判可用"。
2. **env-aware 窗口构造器**（lag=16，398,946 窗）：修 `contiguous_runs` 的 A′ 适配（**opt-in**，别改默认），重训 WM1（6 分钟/轮），预期 kstep loss 与 20 步精度都能再降。
   **注意**：换构造器后**旧 A′ 基线要重算**（口径变了不能沿用 4.60 cm）。

### 路线 B：critic 正确落地（用户已明确要做）
1. **先用 C 线自己的轨迹重建 MC 目标**（γ 与你准备用于自举的一致；`--center linear`），不要再拿 A′ 的 critic 直接上；
2. `--horizon 10 --critic-enabled 1 --critic-kind token --beta 1.0 --beta-warmup-updates 100 --critic-ckpt <同口径热启动>`；
3. 每轮 log 检查：`critic/v_loss`、`v_mean`、`β` 爬坡、`J_ac` 与 `J` 的差；**critic 的 v_ev 必须在其训练分布留出集 >0.30**；
4. WM 每次 refresh 后 critic 必须**重训或至少继续共训**（否则 V 与 WM 口径漂移）。
**放行条件**：在 ≥91 窗、H=20 真值上，`J10+critic` 的符号率与相关**都**优于 `J10` 单独（否则 critic 无用，回退）。

### 路线 C：P5-5 真闭环 + P6
- 修 `bptt_args` 的 `term_off/d_max` 对拍（离线 reward 与训练目标必须同口径）；
- 真闭环 H=20（可恢复范围 71 维）同动作/参考/reward 对照；
- 过 P5-4/P5-5 才谈发布：`activate` 私有店 active → 再决定是否提升到 `runs/three_track_v7/wm_versions/`。

### 禁止事项
- ❌ `git add -A`（主仓/别的会话在同 worktree 提交；`runs/` 被 gitignore）
- ❌ 改 `three_track_v7/wm_versions/active.json`（发布需人工）
- ❌ 用 H=20 的 WM 做收益排序/决策
- ❌ 在 base6 数据上算 tracking 类判据（无参考帧）
- ❌ 用极端 cos 子集下"C adapter 有没有用"的结论
- ❌ 原地编辑正在运行的 bash/python 脚本

---

## 14. 文件与命令索引

### 14.1 本轮新增/修改的脚本（都在 `comimic-wt03/scripts/`）

| 文件 | 作用 |
|---|---|
| `supervise_wm6_snapshot.py` | 采集期监督快照（进度/ETA/独立合同复核） |
| `wm6_coverage.py` | 六任务覆盖度闸门（`--brief` 给 bash 用） |
| `annotate_episode_outcome.py` | 从嵌套 `result.json` 回填 `done_reason/success`（只写 episode.json） |
| `verify_merged_wm6.py` | merged↔Raw 逐位核对（可 `--all`） |
| `select_wm6_checkpoint_v2.py` | **双口径**检查点选优（官方 holdout 约束 + 稠密报告） |
| `register_wm6_candidate.py` | 把检查点登记成版本（sha 可追溯） |
| `run_wm6_stage.sh` | 单阶段长训 + 选优 + 登记（LR/UPDATES/WB 环境变量） |
| `supervise_wm_base6_v3.sh` | 端到端监督：等采集 → 审计/合并 → 逐位核对 → 覆盖闸门 → 长训 → 选优 → 门禁三份 → `chain_result.json`/`next_action.json` |
| `supervise_check_wm0_contract.py` | 独立重跑"换序代价"表（A′ 64 窗，lag=16 窗口） |
| `probe_wm6_dir_agree.py` | dir_agree：WM 梯度 → 精确换算 → job / score |
| `p54_rank_calib.py` | P5-4 复评：符号率+CI、幅度校准、置信门限精确率 |
| `p54_composite_judge.py` | 复合裁判 `J10 + γ^H·(V(s_H)−V(ref_H))` 评估 |
| `profile_wm6_step.py` | 单步耗时拆解（采样 vs kstep） |
| `eval_wm6_per_task.py` | 六任务逐任务 val（P5-2） |
| `probe_c_vs_base_isaac_plan.py` | **改**：`--select random|stratified`、`--split-key`、`--bptt-args` |
| `probe_isaac_h20_report.py` | **改**：`--wm-ckpt`（同一批窗口换裁判） |
| `train_wm_refresh_v7.py` | **改**：`--save-every/--save-dir` 中间检查点 |
| `train_temporal_ac_bptt_v5.py` | **改**：`--critic-kind/--critic-ckpt/--critic-hidden/--critic-depth/--critic-use-diff/--critic-gamma/--dry-run`，token critic 自举与更新 |
| `diff_vla_wbc/wm_replay_buffer.py` | **改**：`WM6_WINDOW_CACHE=1` 窗口池化（默认关）+ `_window_pool()` |
| `run_wm_base6_pipeline.sh` | **改**：候选 `_candidate` 后缀解析（否则 gate 必 FATAL） |
| `docs/SUPERVISION_WM6_20261108.md` | 本轮监督报告（更偏结论） |
| **本文** | 全链路交接（给 GPT） |

### 14.2 关键产物

```
runs/wm_base6_v1/
  collector_A|B/<task>/seed<k>/…                 36 集原始 tick
  audit/base_audit.json|.md                       审计（0 问题 episode）
  merged/{train,val,test}/segments.pt + manifest  2859/426/454 段，24/6/6 集
  gates/wm_base6_best_eval_with_probe/gate.md     ★ PASS 6/6
  gates/gradient_probe.json                       {dir_agree_old:0.9271, dir_agree_new:0.9792}
  wm_candidates/{init_from_wm0, wm_base6_001, wm_base6_best, wm_base6_150, wm_base6_lr2e6_best}_candidate
  train_ckpts/wm_base6_001/…_s0050..s0600.pt      12 个检查点
  dir_agree/{job,roll,score}_*                     dir_agree 与 P5-4 产物
  supervision/{merged_verify, ckpt_selection_v2_stageA, per_task_val, h_sweep,
               cv8_h_sweep, p54_calib_test45, p54_calib_val46, p54_pairdiag,
               p54_composite_cv8, gradient_probe, chain_result, next_action}.json
runs/wm_real/{transitions_a_prime.pt, critic_a_prime.pt}   critic 数据与权重（v_ev 0.3507）
runs/a_prime_sonic_exc/rewards_current/                   A′ 逐行奖励（429 MB）
```

### 14.3 sha 前缀

| 对象 | sha |
|---|---|
| VLA `config.json` | `7d06d0da` |
| SONIC release decoder | `c7241a12` |
| SONIC low_latency（**禁用**） | `c4ac2e74` |
| A′ `all_transitions.pt` | `418c310b` |
| WM0 `wm_v000` | `fc5aeb30` |
| `wm_base6_best`（s350） | `8edb59cf` |
| `wm_base6_lr2e6_best`（s200） | `89f42289` |
| `wm_c1_v001_published`（INVALID） | `81d24caf` |
| packed motion | `3f444d50` |

---

## 15. 附录：指标定义

- **root e20**：`‖p̂_20 − p_20‖`，p 为 root 位置（m），沿 episode 内窗口取平均。
- **dir_agree**：`mean_{窗口} 1[sign(ΔJ_WM) == sign(ΔJ_true)]`；δ 为动作块 RMS 位移（rad）。
- **ΔJ**：`Σ_{t=0}^{H−1} bundle.reward(s_t, a_t, a_{t−1}, FrameRef(frames_t))["total"]`，不带 γ、不带 survival。
- **v_ev**：`1 − MSE(V, ret)/Var(ret)`（留出 episode）。
- **bootstrap_identifiable(K,H)**：`K > H`；否则尾部区间为 0，`V(s_H)` 不可识别（规范 §5.5.3/§6.1）。
- **旧 A′-官方口径**：`contiguous_runs()` 在 clip 哈希 holdout 行上切窗（567 窗）；**旧 A′-稠密口径**：lag=16 轴、holdout clip 的窗口（40 万候选）。
- **覆盖率/精确率（P5-4）**：只在 `|c·ΔJ_wm| > τ` 的窗口下判定"更好/更差"，`coverage=n_conf/n`、`precision=符号正确的比例`。

---

## 16. 时间线（2026-11-08）

| 时间 | 事件 |
|---|---|
| 16:41 | 接手；发现 v1 监督不等 B、缺六任务闸门、候选 `_candidate` 路径 bug |
| 16:46 | 换 v2 监督；16:47 首次 audit+merge；独立复核合同（A′ 换序代价复现） |
| 17:31 | **A 18/18 完成**（17:12 vision_navi 6/6 收、17:31 sit_sofa 收、server 停） |
| 17:37–18:02 | **A 接手 boxing 6 集**（B 未采，避免重复） |
| 18:02 | 36/36 完成；B 写 `DONE.json` |
| 18:03 | v3 因"热改运行中脚本"崩溃（数据无损）→ 18:05 重启 |
| 18:06 | merge 24/6/6；**3739 段逐位复核 0 坏** |
| 18:11 | 600 步长训完成（5.5 min）；选 s350 |
| 18:12–18:26 | 门禁：先 FAIL（缺 dir_agree），补探针后 **PASS 6/6** |
| 18:19–18:26 | dir_agree 真跑：0.927 / 0.979 |
| 18:45–19:05 | H 扫描（0.93–0.99 @H≤10）+ C-V8 截断扫描 |
| 19:00–19:15 | P5-4 随机 91 窗复评（0.44–0.56）+ McNemar + 校准 |
| 19:12–19:18 | critic 数据（502,636 行）→ 拟合 v_ev 0.3507 → 复合裁判（变差） |
| 19:22 | **token critic 接入训练器 + dry-run 通过（未训练）** |
