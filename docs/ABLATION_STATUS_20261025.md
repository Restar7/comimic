# 消融状态板（2026-10-25，comimic4 / 4×RTX 4090 49 GB）

> **这份文档只回答两个问题：①每个格子现在到底在不在跑；②跑完了结论是什么。**
> 口径以 `docs/HANDOVER_MAINLINE_AND_ABLATIONS_20261025.md` 为唯一来源（本节不复述）：
> 底模 `ckpt_010500_physint.pt`（`physint_fix` 仅在 A4 v6）、端口 token 条件、`a=Δq`、
> `--state-vel-weight 0`、纯 HA `data/npz/ha_tokens`（699/634,016）、VLM backbone 冻结。
> 旧口径（HA 标签 token / `state-vel-weight 0.05` / 旧动作换算）的结果按 §8 全部作废，**不混表**。

---

## 1. 一览（2026-10-25 08:45 UTC 快照）

| 组 | 格子 | 状态 | 在哪 / 产物 |
|---|---|---|---|
| **A1** | 主线联合微调 4000 步（ft 臂） | 🔴 **死了**（06:38:45 被 `SIGTERM` 杀在 step ≈3100；最后落盘 `joint_003000/vla_003000`） | `runs/joint_ha_a1`，`logs/joint_ha_a1.log` |
| **A1** | Isaac 闭环 base vs ft | 🔴 **停**（base 只跑完 5/18 episode 就被 `Terminated`，ft 臂 0） | `runs/ablations/A1_joint_finetune_ha/`，`logs/a1_after_train.log` |
| **A2** | 4 后端 × 30 min | ✅ **四臂全收**（`runs/ablations/A2_fm_backend_v2_physint/`）：`pathwise` 900 步 `+0.0834/0.1149`、`adjoint` 900 步 `+0.0836/0.1213`、`truncated` 900 步 `+0.0829/**0.1740（比起点还差）**`、`checkpointed` 600 步 `+0.0808/0.1472` | 卡 0,1 |
| **A3** | PPO 臂 120 min | 🟢 跑（06:15:03 起第 3 次；**加保护仍在发散** → §4.1） | `runs/ablations/A3_bptt_vs_ppo/ppo` |
| **A3** | BPTT 臂 120 min | ⛔ **用户指令中止在 step 2100**（`+0.0844/0.0928`，无 ckpt）⇒ 该格**不完整**，只能作方向性证据 | `runs/ablations/A3_bptt_vs_ppo/bptt` |
| **A4** | v6 `wm_frozen_H8` | ✅ **已完成**（400/400；末点 `rollout_reward 0.0820 / token_mse 0.1669`） | `runs/ablations/A4_v6_fixframe/wm_frozen_H8` |
| **A4** | v6 `wm_frozen_H16` | ✅ **已完成**（400/400；末点 `0.0778 / 0.1166`） | `runs/ablations/A4_v6_fixframe/wm_frozen_H16` |
| **A4** | v6 `wm_frozen_H32` | ✅ **已完成**（400/400；末点 `0.0734 / 0.1032`） | `runs/ablations/A4_v6_fixframe/wm_frozen_H32` |
| **A4** | v6 `wm_frozen_H40` | ✅ **已完成**（400/400；末点 `+0.0681 / 0.1338`） | `runs/ablations/A4_v6_fixframe/wm_frozen_H40` |
| **A4** | `wm_live_H8/H40` | ⛔ **政策禁止**（§4.1/§11.4：等 BONES-SEED 重训底模） | — |
| **A5** | `head_only` / `head_plus_wbc` × 40 min | 🟢 `head_only` ✅ 900 步（`+0.0796 / token 0.3721`，即**token 严重漂移但 reward 尚可**）；`head_plus_wbc` 08:42 起（≈09:25 收） | 卡 0,1；`runs/ablations/A5_only_head_v2_physint/<arm>` |
| **anchor-B** | 锚定续训 2000→4000（λ=1.0、vel=1.0） | ✅ **已完成**（07:17:27→08:43，单卡 2000 步）：`token_mse 0.0773→**0.0152**`、`reward +0.0816→+0.0823`、`pos/ori/q 0.0040` ⇒ **锚定有效**，详见 `docs/TOKEN_ANCHOR.md` §8 | 卡 2；`runs/joint_ha_anchor_2000to4000/` |
| **anchor-B-cont** | 续训到平台 4000→（上限 8000） | 🟢 跑（卡 3，09:03 起）：`[eval 4000] 0.015151/+0.082262` 与 B 末点逐位一致；**平台哨兵**在盯 | `runs/joint_ha_anchor_4000to8000/` |
| **anchor-A** | **从 0** 带锚定 0→2000 → 接力 2000→（上限 8000） | 🟢 跑（卡 2，08:53 起，09:06 时 step 600）：`eval 500 token 0.1698→**0.0233**、reward +0.0402→+0.0799、kin 0.0685→0.0049、hand 0.6513→0.0090` | `runs/joint_ha_anchor_from0/` → 接力 `runs/joint_ha_anchor_from0_2000to8000/`（`scripts/queue_anchor_from0_phase2.sh` 自动接） |
| 基础设施 | BONES-SEED 重训 32000 iter | 🟢 跑（`ckpt_001000` 已出） | `trackerlab-supermimic/logs/bones_seed_retrain_1025b` |
| 可选 | §1.4 WM A/B（physint vs 残差） | ⚪ 未开（可选） | — |

> 说明：A2 与 A5 在**同一条 2 卡链**上串行（A2 2 h → A5 1.3 h）；A3-BPTT 已按用户指令让位给 anchor；
> 每组内部所有臂共用**同一冻结 WBC + 同一数据 + 同一 batch/horizon/辅助损失**，唯一变量是文档写明的那一个。
> 读日志小坑：`joint_ha_a1.log` 是**块缓冲**写盘，尾部行会滞后几分钟，别把"日志没动"当成掉速/卡死。
> **当前两条 anchor 线并行在两卡**：卡 2 = A（从 0 带锚定）、卡 3 = B-cont（4000 续训）；都用
> `scripts/plateau_guard.sh` 按"平台判据"自动停（见 `docs/TOKEN_ANCHOR.md` §10）。

## 2. 起跑记录（可复现）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
# chain1：卡 0,1 —— A2(4×30 min) → A5(2×40 min)，逐臂重试 2 次
setsid nohup bash scripts/run_abl_1025_queue.sh chain1 > logs/run_1025_v2/chain1.out 2>&1 &
# chain2：卡 2,3 —— A3-BPTT(120 min)，逐臂重试 2 次
setsid nohup bash scripts/run_abl_1025_queue.sh chain2 > logs/run_1025_v2/chain2.out 2>&1 &
```

- 队列总控：`scripts/run_abl_1025_queue.sh`（新文件，只做串行+重试，不改任何训练语义）
- **eval 0 指纹（两链一致，证明同一底模/同一 eval batch）**：
  `token_mse 0.169811 / state_term 0.068489 / rollout_reward 0.040185 / hand_mse 0.65125`
  （与交接文档 §3.2 的起点 0.17 / 0.071 / +0.039 对得上）
- **预取状态**：A2/A5/A3-BPTT 全部 `[prefetch] enabled=False`（`train_joint.py` 默认关；
  只有 `launch_joint_ha.sh` 会传 `--prefetch-vla`）。⇒ 同时满足两件事：
  ① 避开 04:54/04:58 那两次后台 PyAV 线程 SIGSEGV 的崩溃类别；
  ② A3 的 BPTT 臂与在跑的 PPO 臂（`--no-prefetch-vla`）**预处理路径完全对称**，同墙钟可比。
- 旧结果目录**一个都没覆盖**（§8）：A2 写 `A2_fm_backend_v2_physint/`，A5 写 `A5_only_head_v2_physint/`，
  旧的 `A2_fm_backend/`、`A5_only_head/` 原样保留作"旧口径"归档。

## 3. 判读口径（跑完按这个出结论）

| 组 | 主指标 | 次要指标 | 预期/警戒 |
|---|---|---|---|
| A2 | 同 30 min 的 `rollout_reward` 斜率与终值 | `token_mse`、`state_pos/ori/q`、显存峰值、s/step | `truncated` 应更快但有偏（终值更差）；`adjoint` 应 ≈ `pathwise`（明显偏离=实现 bug）；`checkpointed` 应 ≈ `pathwise` 但更省显存 |
| A3 | 同 120 min 的 `rollout_reward` 曲线 | `ppo/sigma`、`ppo/clip_frac`、`token_mse`、存活 | 先看"能不能动"；`ppo/sigma` 塌到下限=策略退化。⚠️ PPO 臂 step 50 的 `rollout_reward` 已 0.0454→0.0063，需盯 |
| A5 | `head_only` vs `head_plus_wbc` 的三项 | 冻结审计 `max|joint_latest.policy − SM_CKPT.policy| == 0` | 若 head_only ≈ head_plus_wbc ⇒ 收益主要来自 VLA head，可省掉 WBC 更新 |

## 4. 结论（跑完后填；按时间追加）

- ⏳ A2：待填（预计 ≈07:55 完成 4 臂）
- ⏳ A3：BPTT 待填（预计 ≈07:55）；PPO 已完成度见 `runs/ablations/A3_bptt_vs_ppo/ppo/eval_curve.jsonl`

### 4.1 A3 · PPO 臂（本容器 = 卡 0,1）—— 结论：**数值发散，不能当作"方法优劣"的证据**

三次尝试，配置逐次收紧，结局一致：

| # | 时间 (UTC) | 配置 | 结局 |
|---|---|---|---|
| 1 | 10-24 23:35 | 旧底模 `ckpt_010500.pt` + `state-vel-weight 0.05` | step 700 起全 NaN，最终 SIGABRT + NCCL 超时 |
| 2 | 10-25 04:52 | `physint` + prefetch 默认开 | 04:54:16 rank0 **SIGSEGV 于 PyAV/libavcodec**（后台预取线程解 ego-view），与数值无关 |
| 3 | 10-25 05:29 | `physint` + `--no-prefetch-vla`（数值等价，见 §2 预取说明） | **数值发散**，见下表；预计 07:29 结束 |

第 3 次实测（`runs/ablations/A3_bptt_vs_ppo/ppo/eval_curve.jsonl` 与 `logs/a3_ppo_physint_v2.log`）：

| step | `state_term` | `rollout_reward` | `ppo/ratio_mean` |
|---|---|---|---|
| 0 | 1.7886 | **+0.0454** | 1.0 |
| 100 | 4.8364 | +0.0096 | ≈1.4e2 |
| 200 | 6.9516 | **+0.0003** | 1.23e10 |
| 250 | — | — | 3.46e10 |

旁证：`loss/policy` 恒为 `-0.141421`（退化饱和）、`ppo/sigma ≈ 0.25`（未塌）、`clip_frac_epoch0 = 0.0`、**无 NaN**。

**机制**：`ratio_mean = π_new/π_old` 在一次更新后即从 1 到 1e2→1e10，说明策略均值一步就被推离采样分布数量级。`diff_vla_wbc/ppo.py` 与 `scripts/train_ppo.py` 中**没有梯度裁剪、没有 advantage 标准化、没有 ratio/非有限值保护**，而策略均值是 2 步 Euler 的流匹配 `FM_θ(obs,z)`，反传链会放大梯度。

**同口径对照**：主线 BPTT（`runs/joint_ha_a1`，physint + vel 0 + WM 冻结）eval0 `state 3.05 / reward +0.0391` → step1000 `0.413 / +0.0819` → step2000 `+0.0816`。

**写进论文的措辞（在补保护重跑之前）**：

> 在钉死的 recipe（physint 底模、端口 token、`a=Δq`、WM 冻结、`ppo-epochs 4 / sigma 0.25 / fm-lr 1e-5`）下，
> PPO 臂数值发散、不可用；BPTT 臂正常收敛。该结果**不足以**支撑"BPTT 优于 PPO"的方法学结论，
> 必须先给 PPO 补上标准保护再跑一次，以区分"方法本身"与"实现缺保护"。

**恢复条件**（约 20 min 代码 + 3 步冒烟 + 120 min 重跑）：

1. `nn.utils.clip_grad_norm_(fm_params, 1.0)`；
2. advantage 每 batch 标准化（减均值 / 除标准差）；
3. 训练循环加 `if not torch.isfinite(loss): skip` 与 `ratio.clamp(0, 10)`。

**已实施保护（2026-10-25 06:15，卡 0,1）**：`diff_vla_wbc/ppo.py` 加了三项，均为标准 PPO 实践、不改策略参数化：

| 项 | 改动 |
|---|---|
| log-ratio 裁剪 | `log_ratio.clamp(-5, 5)`（`PpoConfig.log_ratio_clip`，置 0 可关闭） |
| 梯度裁剪 | 专用 `grad_clip = 1.0`（原来借用 `value_clip = 10.0`，过宽） |
| 非有限值保护 | `grad_norm` 非有限时跳过该步并计数，不污染优化器状态 |

新增诊断 `ppo/log_ratio_absmax` 立刻把机制钉死：**冒烟第 1 步的原始 log-ratio 绝对值已达 2649**
（`exp(2649)` 远超浮点表示），说明"40×64 token 动作 + 标量 σ=0.25 + 全维高斯 logp"下，
一个 batch 内 4 个 Adam epoch 就能把均值推离到不可表示的程度——这正是 `ratio_mean → 1e10` 的来源。
裁剪后 ratio 被限制在 `[e^-5, e^5]`，epoch≥1 的梯度被大幅中和（等效退化为"每 batch 近似单步策略梯度"，
与交接文档 §A3 "PPO 有效 horizon = 1" 的定位一致），但**配置未改**（仍是 `ppo-epochs 4 / sigma 0.25 / fm-lr 1e-5`）。

第三次重跑：`[arm a3_ppo] start 2026-10-25T06:15:03Z`，120 min（预计 08:15 结束），
日志 `logs/a3_ppo_guarded.log`，产物 `runs/ablations/A3_bptt_vs_ppo/ppo/`。

**产物归档**：`runs/ablations/A3_bptt_vs_ppo/ppo/`（第 3 次，在跑）、
`runs/ablations/A3_ppo_sigsegv_20261025_0454/`（第 2 次）、
`runs/ablations/A3_pre_physint_VOID_20261024/`（第 1 次 + 旧口径 BPTT，按 §8 作废）。
- ⏳ A5：待填（预计 ≈09:15 完成 2 臂）

## 5. 两个必须先处理掉的坑（否则 A1 闭环会静默出错）

1. **`runs/ckpt_vla_ha_ft4000` 目前是假货**：`logs/a1_after_train.log` 显示 `vla_004000.pt` 缺失 →
   回退合并了 **`vla_002000.pt`**。主线到 4000 步后必须重新
   `python3 scripts/export_vla_ckpt.py --base runs/ckpt_vla_ha_8000_local --vla runs/joint_ha_a1/vla_004000.pt --out runs/ckpt_vla_ha_ft4000 --force`，
   否则 A1 的 ft 臂其实是 step-2000。
2. **A1 的 WBC 两本文档打架**：`docs/DATA_POLICY_HA_ONLY.md` §4 与 `A1_joint_finetune_ha/manifest.json` 用
   `ckpt_013500.pt`，而 `docs/HANDOVER_MAINLINE_AND_ABLATIONS_20261025.md` §4/§11.2 写 `physint(_fix)`。
   重跑闭环前二选一，**两臂必须同一个**。

## 6. 更新日志

| 时间 (UTC) | 事件 |
|---|---|
| 2026-10-25 05:38 | 盘点：A5/A2/A3-BPTT/A4-H16·32·40/A1 闭环均未开跑 |
| 2026-10-25 05:53:56 | 卡 0,1 起 chain1（A2 4 臂 → A5 2 臂），下 `scripts/run_abl_1025_queue.sh chain1` |
| 2026-10-25 05:53:58 | 卡 2,3 起 chain2（A3-BPTT 120 min） |
| 2026-10-25 05:57 | 两链均过门禁并开始训练；eval 0 指纹一致（见 §2） |
| 2026-10-25 06:05 | 卡 0,1（本容器）追加 **A3-PPO 结论：数值发散**（ratio 1e10、reward 0.0454→0.0003），需补梯度裁剪/advantage 标准化后重跑；详见 §4.1 |
| 2026-10-25 06:21 | 复核：A1 主线 **step 2600/4000**（1.5 s/step，日志块缓冲导致的"停滞"假警报已排除）；A2 `pathwise` 步 750、A3-BPTT 步 750（均 ≈2.0 s/step）；A4 v6 **H8/H16 已完成**、H32 跑到 160/400 |
| 2026-10-25 06:25 | 排期不变：A2 四臂 → A5 两臂（卡 0,1）、A3-BPTT 120 min（卡 2,3）；后处理器 `scripts/after_1025_reports.sh` 已在等，跑完自动出 `docs/ABLATION_A2_RUN_PHYSINT_20261025.md` / `docs/ABLATION_A5_RUN_PHYSINT_20261025.md` / `docs/ABLATION_A3_RUN_PHYSINT_20261025.md` + `runs/ablations/SUMMARY_1025_v2.{json,md}` |
| 2026-10-25 07:04 | **A2 两臂收工**：`pathwise` 900 步（`reward +0.0834 / token 0.1149`）、`checkpointed` 600 步（`+0.0808 / 0.1472`，同 30 min 步数更少＝用步速换显存）；`truncated` 第 1 次 4 s 退出＝**上一臂 launcher 的 `flock` 还没释放**（队列在同一秒起了下一臂），30 s 后重试即成功（同类现象 `checkpointed` 也遇到一次，均已自愈）；`adjoint` 排队。A3-BPTT 步 2050（`+0.0839 / 0.0928`，120 min 预算，不能与 A2 的 30 min 直接比） |
| 2026-10-25 07:04 | A4 v6 **H8/H16/H32 三格已完成**（400 步末点 reward `+0.0820 / +0.0778 / +0.0734`），H40 跑到 280/400（`+0.0681`）⇒ 初步趋势：**同 400 步下 horizon 越长终值越低**，H40 收工后出结论 |
| 2026-10-25 07:04 | **新增 anchor 续训**（用户指令）：`docs/TOKEN_ANCHOR.md` §7，2000→4000、λ=1.0、vel=1.0、`NGPU=1`+batch 2（与主线 `_world_size=1` 同形，单变量可比）；排队器 `scripts/launch_anchor_1025.sh` 等 A3-BPTT 释放**卡 2** 后自动起跑；产物 `runs/joint_ha_anchor_2000to4000/`（**不覆盖 `runs/joint_ha_a1/`**，2000 存档走硬链接）；**卡 3 留空给 A1 的 Isaac 闭环** |
| 2026-10-25 07:15 | **A3-BPTT 已被用户指令中止**（停在 **step 2100**：`reward +0.0844 / token 0.0928 / state 0.0079`，eval 曲线 6 个点、**没有落 ckpt**）。⚠️ 因此 A3 的"BPTT vs PPO 同 120 min"这一格**不完整**，只能当"BPTT 在 120 min 内单调爬升、PPO 发散"的方向性证据，不能作为等墙钟结论；需要正式对比时要重跑满 120 min（约 2 h，卡 2,3） |
| 2026-10-25 07:15 | 卡 2,3 已释放；卡 2 立刻交给 anchor 续训，**卡 3 空闲**（留给 A1 的 Isaac 闭环，Isaac Sim 需独占一张卡） |
| 2026-10-25 07:17 | **anchor 续训正式起跑**（卡 2，单卡，07:17:27）：`[anchor] ON: λ=1.0`、`[resume] joint_002000 (step=2000)`、`[eval 2000] token 0.077289 / reward +0.081576`，与主线同点 `0.077285 / +0.081578` **逐位一致** ⇒ 单变量成立。首跑 07:15 因 `launch_joint_ha.sh` 的 PYTHONPATH 漏 `gr00t_deps`（`No module named 'tyro'`）失败，已修（见 `docs/TOKEN_ANCHOR.md` §7.2）。判据：**3000 步 `token_mse` 能否守住 ≤0.08**（主线同点是 0.0925） |
| 2026-10-25 08:43 | **anchor 跑完 4000 步：锚定有效，全部验收过。** `token_mse 0.0773→0.0193(2500)→0.0181(3000)→0.0183(3500)→0.0152(4000)`（−80%，全程 ≤0.02，**没有 2000→3000 的漂移**）；`rollout_reward +0.0816→+0.0823`（3000 点 +0.08344 比主线同点 +0.08223 还高）；`pos/ori/q = 3.5e-5/0.0010/0.0030`（合计 0.0040，ori 比主线 0.0022 好）；`state_vel 0.418` 守住地板；`hand_mse 0.0114→0.0079`。⇒ 结论与"该不该再跑 λ=3.0/0.3"的答案：**不用**（详见 `docs/TOKEN_ANCHOR.md` §8） |
| 2026-10-25 08:43 | ⚠️ **A1 主线已死**：`logs/joint_ha_a1.log` 末尾 `SignalException: Process … got signal: 15`，06:38:45 被杀在 step ≈3100，最后落盘 = `joint_003000/vla_003000`。⇒ **A1 的 ft 臂要用 4000 步模型时用 anchor 的 `vla_004000.pt`**（merge 命令见 `docs/TOKEN_ANCHOR.md` §8.3），不要再等主线 |
| 2026-10-25 08:45 | **A2 四臂全收（30 min/臂）**：`pathwise` 900 步 `reward +0.0834 / token 0.1149`；`adjoint` 900 步 `+0.0836 / 0.1213`（reward ≈pathwise，token 略差 = detach backbone 的代价）；`truncated` 900 步 `+0.0829 / **0.1740**`（token 比起点 0.1698 还差 ⇒ **有偏，符合预期**）；`checkpointed` 600 步 `+0.0808 / 0.1472`（同墙钟少 1/3 步数，reward 略低 ⇒ 省的显存要用步速买）。A5 `head_only` 已收（900 步，`+0.0796`，**token 0.3721 严重漂移** ⇒ 又一例"reward 饱和后 token 乱跑"，正是 anchor 要解决的问题）；`head_plus_wbc` 08:42:42 起（≈09:25 收） |
| 2026-10-25 08:45 | A4 v6 四格**全部完成**：`H8 +0.0820 / H16 +0.0778 / H32 +0.0734 / H40 +0.0681`（各 400 步，末点 reward）；token_mse 末点 0.1669/0.1166/0.1032/0.1338 ⇒ **同 400 步下 horizon 越长终值越低**（长 H 起步更低、回升更慢） |
| 2026-10-25 08:50 | **查清"token 约束线为什么停"**：停的不是 B（续训），B 是**跑满 2000 步正常退出**（`[done]`、4 个 ckpt 齐全、08:43 收）。停的是 **A 方案（从头 0→2000，`runs/joint_ha_anchor/`）**：06:39 启动、**06:40:02 收到 SIGTERM（15）**，停在加载权重阶段，没有 `eval 0` / 没有 step / 没有 ckpt，也没有 OOM/代码异常；那次解释器还是 `/opt/miniconda3/envs/isaaclab`(py3.11) 而不是 `/usr/bin/python3`(3.12)，且当时 4 卡全满 ⇒ **起在没卡的场景被外部清理**。详见 `docs/TOKEN_ANCHOR.md` §9 |
| 2026-10-25 08:53 | **A 方案已重跑**（卡 2 空闲）：`RUN=joint_ha_anchor_from0`，冷启官方底模，`STEPS=2000 NGPU=1 λ=1.0 vel=1.0 EVAL_EVERY=500`；`[eval 0] token 0.169811 / reward +0.040185` 与 A2/A5 指纹逐位一致，`loss/token_anchor 0.2099`，ETA ≈09:50 ⇒ 给论文一张"从 0 就带锚定"的干净曲线 |
| 2026-10-25 09:03 | **B 续训（续到平台为止）在卡 3 起跑**：`runs/joint_ha_anchor_4000to8000/`（`RESUME_STEP=4000`，上限 `STEPS=8000`，`EVAL_EVERY=250`，λ=1.0/vel=1.0/NGPU=1/batch 2/prefetch on）。核对：`[eval 4000] token 0.015151 / reward +0.082262` 与 B 末点**逐位一致**；基座载入 = `runs/ckpt_vla_ha_anchor4000`（为此给 `launch_joint_ha.sh` 加 `MERGED_CKPT` 覆盖——默认的 `ckpt_vla_ha_ft4000` 里是最早主线的 step-2000 权重，**用它就静默错**）。平台哨兵 `scripts/plateau_guard.sh` 已挂 |
| 2026-10-25 09:04 | **A 线接力队列已挂**（卡 2 不空转）：`scripts/queue_anchor_from0_phase2.sh` 等 0→2000 收工 → merge `ckpt_vla_ha_anchor2000_from0` → 起 2000→8000（同 recipe + 平台哨兵），产物 `runs/joint_ha_anchor_from0_2000to8000/` |
| 2026-10-25 09:06 | **A 线（从 0 带锚定）eval 500 就见效**：`token 0.1698→0.0233`、`reward +0.0402→+0.0799`、`kin 0.0685→0.0049`、`hand 0.6513→0.0090` ⇒ **500 步（≈13 min）就到位，B 花了 2500 步才到同级**；若维持，最终模型用 A 的更省、recipe 更干净。**当前两卡并行**：卡 2=A（step 600/2000，之后自动接 2000→平台）、卡 3=B-cont（4000→平台）、卡 0,1=A5 `head_plus_wbc` |
| 2026-10-25 09:22 | **B（4000→）已由哨兵判定平台并自动停**：4000 之后 token 再没创新低（0.0152→0.0171→0.0165→0.0169，全部 > 0.0152−0.002），reward 极差仅 0.0011 ⇒ **B 的平台点 = 4000 步**，最终产物 = `vla_004000`（已 merge 为 `runs/ckpt_vla_ha_anchor4000`）。hand 仍在降（0.0081→0.0046，辅助项）。日志 `logs/joint_ha_anchor_4000to8000.guard.log`；详见 `docs/TOKEN_ANCHOR.md` §10.6 |
| 2026-10-25 09:24 | **A5 两臂收工**（40 min/臂，`--state-vel-weight 0` 口径，**速度项不可与 anchor 线横比**）：`head_only` 900 步 `token 0.1698→**0.3721**（越训越差 2.2×）、reward +0.0796、kin 0.0066、hand 0.0083`；`head_plus_wbc` 900 步 `token →**0.1161**（比 head_only 好 3.2×）、reward **+0.0834**、kin 0.0072（略差）、hand **0.0235**（更差）` ⇒ **A5 结论：更新 WBC 明显救 token 与 reward，代价是手部变差**；且两臂都远不如带 anchor 的线（0.0152）。卡 0,1 已释放 |
| 2026-10-25 09:25 | **指标体检（回答"有没有指标坏"）**：两条 anchor 线**没有任何指标在变坏**，且无 NaN/异常；B-cont 的 token/reward/kin 是"平"（平台），A 的 6 项全在单调向好。唯一贴地板的是 `state_vel`（B 0.386、A 0.388，文档 §6.6 记录地板≈0.39，是 mocap-FD vs sim 的口径差，不是退化）。**别处确有坏指标**：A5 `head_only` token 0.3721、A3-PPO 整条发散、A2 `truncated` token 0.1740 —— 都不是这两条线 |
