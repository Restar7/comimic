# 消融总账（2026-10-25 09:45 UTC）：跑完没有 / 结论 / 还差什么

> 一句话：**5 个原始消融组里 A2 / A4 / A5 已跑完；A1、A3 没跑完。**
> 另外本轮新增的 **token-anchor 线**：B（2000→4000）已跑完并判定到平台；A（从 0 起）0→2000 已完工、
> 2000→平台段正在跑。
>
> 口径唯一来源：`docs/HANDOVER_MAINLINE_AND_ABLATIONS_20261025.md`（physint 底模 / 端口 token /
> `a=Δq` / WM 冻结 / 纯 HA 699 clips）。旧口径结果按该文 §8 **全部作废**，本表只收新口径。

---

## 0. 一句话结论（按组）

| 组 | 状态 | 结论 |
|---|---|---|
| **A1** 有无联合微调（主结果） | ❌ **没跑完** | ft 臂训练主线死在 step ≈3100；**Isaac 闭环 base 只 5/18 episode、ft 0 episode** ⇒ 成功率没有。目前最好的 4000 步 ft 产物来自 anchor-B（`vla_004000`） |
| **A2** FM 反传 4 后端 | ✅ **跑完** | **pathwise 保真最好**（token 0.1149）；`adjoint` reward 齐平（+0.0836）但 token 略差（0.1213，detach 的代价）；`truncated` **有偏**（token 反升到 0.1740）；`checkpointed` 同墙钟少 1/3 步（省显存换步速） |
| **A3** BPTT vs PPO | ❌ **没跑完** | BPTT 被用户指令中止在 step 2100（无 ckpt）；PPO **三次全失败**（旧口径 NaN→SIGSEGV→加保护后 NCCL **SIGABRT**）⇒ **等墙钟对比不成立**，只能写"PPO 在本 recipe 下数值不可用"的负结果 |
| **A4** WM 冻结 × horizon | ✅ **跑完（4 格）** | **同 400 步下 horizon 越长越差**：reward `H8 +0.0820 > H16 +0.0778 > H32 +0.0734 > H40 +0.0681` ⇒ 用 H8；`wm_live` 两格仍**政策禁止** |
| **A5** 冻结 WBC 只训 head | ✅ **跑完（2 臂）** | **更新 WBC 有用**：`head_plus_wbc` token **0.1161** vs `head_only` **0.3721**（3.2×）、reward +0.0834 vs +0.0796；代价是 hand 0.0235 vs 0.0083、kin 略差 ⇒ 选 `head_plus_wbc` |
| **token-anchor**（本轮新增） | 🟢 B 完成 / A 进行中 | **锚定有效**：B 的 token `0.0773→0.0152（−80%）`、reward 不降（+0.0823）、kin 更好（0.0040）；**B 的平台点 = 4000 步**（哨兵 09:22 判定并停）；**从 0 就锚定更快**：A 到 2000 步 token 已 0.0162 |

---

## 1. 完成度总表

| 组 | 格子 | 状态 | 关键数字（末点） | 产物 |
|---|---|---|---|---|
| A1 | 主线联合微调 4000 步 | ❌ 死（06:38:45 SIGTERM @≈3100） | 最后落盘 step 3000 | `runs/joint_ha_a1/`（`joint/vla_003000.pt`） |
| A1 | Isaac 闭环 base vs ft | ❌ base 5/18、ft 0 | — | `runs/ablations/A1_joint_finetune_ha/base/` |
| A2 | `pathwise` | ✅ 900 步 / 30 min | token 0.1149、reward **+0.0834** | `runs/ablations/A2_fm_backend_v2_physint/pathwise/` |
| A2 | `adjoint` | ✅ 900 步 | token 0.1213、reward **+0.0836** | 同目录 `/adjoint/` |
| A2 | `truncated` | ✅ 900 步 | token **0.1740（＞起点）**、reward +0.0829 | 同目录 `/truncated/` |
| A2 | `checkpointed` | ✅ 600 步 | token 0.1472、reward +0.0808 | 同目录 `/checkpointed/` |
| A3 | BPTT 臂 | ⛔ 中止 @2100（无 ckpt） | token 0.0928、reward +0.0844 | `runs/ablations/A3_bptt_vs_ppo/bptt/` |
| A3 | PPO 臂 | ❌ 3 次全失败（NaN / SIGSEGV / NCCL SIGABRT） | 最好一次 step 500：token 1.66、reward **−0.0069** | `runs/ablations/A3_bptt_vs_ppo/ppo/`、`A3_ppo_sigsegv_20261025_0454/` |
| A4 | `wm_frozen_H8` | ✅ 400 步 | reward **+0.0820**、token 0.1669 | `runs/ablations/A4_v6_fixframe/wm_frozen_H8/` |
| A4 | `wm_frozen_H16` | ✅ 400 步 | +0.0778、0.1166 | 同目录 `/wm_frozen_H16/` |
| A4 | `wm_frozen_H32` | ✅ 400 步 | +0.0734、0.1032 | 同目录 `/wm_frozen_H32/` |
| A4 | `wm_frozen_H40` | ✅ 400 步 | +0.0681、0.1338 | 同目录 `/wm_frozen_H40/` |
| A4 | `wm_live_H8/H40` | ⛔ **政策禁止** | — | 等 BONES-SEED 重训底模（现 8450/32000 iter） |
| A5 | `head_only` | ✅ 900 步（40 min） | token **0.3721**、reward +0.0796、kin 0.0066、hand 0.0083 | `runs/ablations/A5_only_head_v2_physint/head_only/` |
| A5 | `head_plus_wbc` | ✅ 900 步 | token **0.1161**、reward **+0.0834**、kin 0.0072、hand 0.0235 | 同目录 `/head_plus_wbc/` |
| anchor | **B**：2000→4000（λ=1.0, vel=1.0） | ✅ 跑完 + 判定平台 | token **0.0152**、reward **+0.0823**、kin 0.0040 | `runs/joint_ha_anchor_2000to4000/`（`vla_004000`） |
| anchor | B-cont：4000→（上限 8000） | ✅ 跑完（哨兵 09:22 停 @4750） | 4000 之后无新低（0.0171/0.0165/0.0169），reward 极差 0.0011 ⇒ **平台=4000** | `runs/joint_ha_anchor_4000to8000/` |
| anchor | **A**：从 0 → 2000 | ✅ 跑完 | token **0.0162**、reward +0.0817、kin 0.0045、hand 0.0099 | `runs/joint_ha_anchor_from0/` |
| anchor | A-cont：2000→（上限 8000） | 🟢 跑（09:38 起） | — | `runs/joint_ha_anchor_from0_2000to8000/` |

---

## 2. 每组详情

### 2.1 A2 · FM 反传 4 后端（✅ 完成，30 min/臂，卡 0,1，2 卡 DDP）

| backend | 同墙钟步数 | `token_mse`（0.1698→） | `rollout_reward`（0.0402→） | 读法 |
|---|---|---|---|---|
| `pathwise` | **900** | **0.1149** | +0.0834 | 精确 BPTT，保真最好 → **默认选它** |
| `adjoint` | 900 | 0.1213 | **+0.0836** | reward 与 pathwise 齐平，token 略差＝detach backbone 特征的代价 |
| `truncated` | 900 | **0.1740** | +0.0829 | **token 比起点还差** ⇒ 有偏后端，符合预期（廉价但学不到 token 保真） |
| `checkpointed` | **600** | 0.1472 | +0.0808 | 同 30 min 少 1/3 步（重算）＝**省显存要用步速买**；reward 略低 |

- 峰值显存 **47,624 MiB**（49 GB 卡，几乎顶满）；`gpu_mem.csv` 在产物根目录。
- 判读纪律：四臂只有 `--fm-grad-backend` 变；VLM backbone 冻结；同数据/同 batch/同 horizon/同辅助损失。

### 2.2 A4 · WM 冻结 × horizon（✅ 4 格完成，各 400 步定步数，卡 4,5 另一容器）

| H | `rollout_reward`（末点） | `token_mse` | 备注 |
|---|---|---|---|
| **8** | **+0.0820** | 0.1669 | 最好；也最快 |
| 16 | +0.0778 | 0.1166 | 起跑点更低、回升慢 |
| 32 | +0.0734 | 0.1032 | — |
| 40 | +0.0681 | 0.1338 | 最差 |

- 结论：**同 400 步下 horizon 越长终值越低**（每步更贵 + 长链更难优化）⇒ 主实验用 H8。
- `wm_live_H8/H40` **仍未跑，且是政策禁止**（§4.1/§11.4：必须等 BONES-SEED 重训底模；
  该重训现在 **8450/32000 iter**）。

### 2.3 A5 · 冻结 WBC 只训 head（✅ 2 臂完成，40 min/臂，卡 0,1）

| 臂 | 步数 | `token_mse` | `rollout_reward` | kin | `hand_mse` |
|---|---|---|---|---|---|
| `head_only`（`--policy-lr 0`） | 900 | **0.3721** | +0.0796 | 0.0066 | **0.0083** |
| `head_plus_wbc`（默认 lr） | 900 | **0.1161** | **+0.0834** | 0.0072 | 0.0235 |

- 结论：**更新 WBC 明显救 token（3.2×）与 reward**；代价是 hand 变差、kin 略差 ⇒ 选 `head_plus_wbc`。
- 注意：A5 两臂是 `--state-vel-weight 0` 口径，`state_vel` 0.53–0.60 与 anchor 线的 0.38–0.42 **不可横比**。

### 2.4 A3 · BPTT vs PPO（❌ 未完成）

- **BPTT 臂**：跑到 step 2100 被中止（用户让卡给 anchor）。eval 末点 `token 0.0928 / reward +0.0844`，
  **没有落 ckpt**（`--save-every 0`）。
- **PPO 臂**：三次尝试三个结局 ——
  1. 旧口径：step 700 起全 NaN，最终 SIGABRT；
  2. physint + 预取默开：04:54 rank0 **SIGSEGV 于 PyAV/libavcodec**（后台预取线程解 ego-view）；
  3. 加保护（log-ratio 裁剪 / grad clip=1.0 / 非有限值跳过）后：**仍在发散**（step 500 `token 1.66 / reward −0.0069`），
     最终 **NCCL watchdog → SIGABRT**。
- ⇒ **不能写"BPTT 优于 PPO"（等墙钟对比不成立）**，能写的只是负结果：*"该 recipe 下 PPO 数值不可用"*。

### 2.5 token-anchor 线（🟢 本轮新增，不算原 A1–A5）

| 线 | 区间 | token_mse | reward | kin | 状态 |
|---|---|---|---|---|---|
| **B** | 2000→4000 | 0.0773 → **0.0152**（2500 就 0.0193） | +0.0816 → +0.0823 | 0.0040 | ✅ 完成 |
| B-cont | 4000→4750 | 0.0152→0.0171→0.0165→0.0169（无新低） | 0.0823–0.0839（极差 0.0011） | 0.0038–0.0044 | ✅ **平台判定**（09:22 哨兵停） |
| **A** | 0→2000 | 0.1698 → 0.0233(500) → 0.0204(1000) → 0.0207(1500) → **0.0162**(2000) | +0.0402 → +0.0817 | 0.0045 | ✅ 完成 |
| A-cont | 2000→（上限 8000） | — | — | — | 🟢 跑（09:38 起） |

- **锚定的作用已被两条独立证据夹住**：① B（半路带锚定）把漂移按住 —— 同一段窗口里，主线（无锚定）
  `token 0.0773 → 0.0925（↑20%）`，B 是 `0.0773 → 0.0181`；② A（从 0 带锚定）**2000 步就到了
  "半路才开始锚定"要 4000 步才有的水平**（token 0.0162 vs 0.0152、reward 持平）。
- 平台判据（`scripts/plateau_guard.sh`）：最近 3 个 eval 点不再把最好 token 压低 ≥0.002 且 reward 极差 ≤0.0025。
- **可用于 A1 ft 臂的产物**：`runs/ckpt_vla_ha_anchor4000`（由 B 的 `vla_004000` merge，537/537 校验通过）。

---

## 3. 还没跑完的（按对论文的重要性排序）

| # | 缺什么 | 为什么缺 | 补法（命令/入口） | 成本 |
|---|---|---|---|---|
| **1** | **A1 的 Isaac 闭环 base vs ft（成功率）** | 05:15 base 臂 5/18 episode 被 `Terminated`；ft 臂从未开始；且 ft 的 ckpt 当时是错的（`ckpt_vla_ha_ft4000` 里是 step-2000 权重） | 用 `runs/ckpt_vla_ha_anchor4000` 当 ft，`bash scripts/run_a1_ha_isaac.sh`；base 臂续完 13 episode | 1 张卡，≈1.5 h |
| **2** | **A3 的 BPTT 满 120 min**（让 A3 恢复"等墙钟"可比） | 被用户指令中止（让卡给 anchor） | `MODE=bptt MINUTES=120 bash scripts/launch_a3_a5.sh` | 2 卡 × 2 h |
| **3** | **A3 PPO 修复后重跑** | 3 次失败（发散 + NCCL abort）；现有保护不够 | 先补：advantage 标准化、把 `fm-lr` 降到 1e-6、或缩短 steps/epoch；再同墙钟重跑 | 2 卡 × 2 h + 修 |
| **4** | A4 `wm_live_H8/H40` | **政策禁止**（在线更新会把速度标定洗掉） | 等 BONES-SEED 重训（现 8450/32000）→ 用新底模 | 等 ≈10 h |
| **5** | §1.4 的 WM A/B（`physint` vs 残差 `010500`） | 可选，未跑 | 同配置 2×30 min，只换 `--supermimic-ckpt` | 2×30 min |
| **6** | L1 官方线（官方 VLA + SONIC decoder）成功率 | 未跑；A1 headline 的另一半 | 单独一条链，不能与 L2 混 token | ≈1 h（1 卡 + Isaac） |
| 7 | A5 两臂的闭环成功率 | 未跑（训练侧已出结论） | 同 #1 流程，换两条 ckpt | ≈1.5 h |

> 也就是说：**"相关的都跑完了吗"——训练侧的 A2/A4/A5 + anchor 已跑完；A1/A3 没跑完，主要缺的都是"闭环成功率"和"PPO 这条对照"。**

---

## 4. 口径提醒（读表前必看）

1. **旧口径全部作废**（HA 标签 token / `state-vel-weight 0.05` / 旧动作换算 / 旧底模 `ckpt_010500.pt`）：
   `runs/ablations/A2_fm_backend/`、`A5_only_head/`、`A4_pilot_v1/`、`A4_v2_contaminated/`、
   `A3_pre_physint_VOID_20261024/` **只作归档，不进本表**。
2. **本表的 A2/A5 实际底模是 `ckpt_010500_physint.pt`**（启动器默认值；可用 `ps` 命令核对）。
   注意自动生成的 `docs/ABLATION_A3_RUN_PHYSINT_20261025.md` 的"协议"段里 WBC 仍印着 `ckpt_010500.pt`，
   那是**报告模板的旧默认字符串**，不是实际取值。
3. **`eval` 只有 2×2 = 4 个窗口**：±0.001 级别的差异不可解读，只能看量级/趋势。
4. **`reward` / kinematic 已饱和**：anchor-B 的 `+0.0823 / 0.0040` 已经**优于 recorded-token oracle 参照**
   （`+0.0786 / 0.0063`）⇒ 这两项失去分辨力，**不要再用来证明"更好"**；唯一独立指标是闭环成功率。
5. **A1 闭环不要用 `runs/ckpt_vla_ha_ft4000`**：里面是最早主线（07:00 接力）的 **step-2000** 权重，
   名字骗人；要用 `runs/ckpt_vla_ha_anchor4000`。
6. `joint_ha_a1.log` 是**块缓冲**写盘，尾部滞后几分钟，别把"日志没动"当掉速。

---

## 4.1 2026-10-25 11:01 起：**消融暂停，4 卡转 WBC 全 clip 重训**

用户指令：停下当前所有任务，按 `docs/WBC_TRAIN_4GPU_BONES_SEED.md` 在本容器 4×4090 上重训 WBC
（要求"所有 episode 的所有长度都要用到"）。

- 已停：anchor-A 续训（2000→8000）、平台哨兵、A2/A5 队列、后处理器 —— 11:01 前本容器**零训练进程**。
- 已起：`NPROC=4 × ENVS_PER_RANK=8192`、`MAX_EPISODE_STEPS=0`（→ 生效值 `2^31-1`，**只由 clip 结束触发**）、
  `ITERATIONS=32000`、`CKPT_EVERY=500`、`interface=token`、`HA_ALIGN=1`、`compile=1`、bf16。
- 数据覆盖已硬验证：库 **129,943 clips / 47,475,680 帧 / 263.75 h**，每 rank 分片 **32,486 clips / 11,848,839 帧 = 24.96%**，
  长度范围 min 29 / median 288 / max 8,979 帧**全部在跑**。
  旧默认 `MAX_EPISODE_STEPS=500` 的真实影响（**11:30 更正**）：起点是随机 1 秒 bin，所以**帧都见过**；
  被砍的是**连续轨迹**——**16.09%** 的回合起点会撞上 500 步（=10 s 硬顶），38.8% 的帧（属于长于 500 帧的
  21,714 条 clip）只能以"从中间起步的一小段"出现。
- 运行记录与勘误：**`docs/WBC_TRAIN_4GPU_RUN_20261025.md`**（含"文档 §1 的中位 789/均值 907/最长 10000 与库不符"的更正：
  实测 median 288 / mean 365.4 / max 8,979）。
- 影响：`A4 wm_live` 的底模就等这次重训的产物（§3 第 4 项），其余消融的补跑（A1 闭环、A3）在本轮训练结束后再排。
- 注意：另一容器上的 `bones_seed_retrain_1025b`（NPROC=2、**500 步上限**）仍在跑且**无法从本容器停止**；
  结论以本次 4 卡全 clip run 为准。

---

## 5. 数据来源索引（都能直接复核）

| 内容 | 位置 |
|---|---|
| 状态板（滚动更新） | `docs/ABLATION_STATUS_20261025.md` |
| A2 全量报告 | `docs/ABLATION_A2_RUN_PHYSINT_20261025.md` |
| A3 + A5 全量报告 | `docs/ABLATION_A3_RUN_PHYSINT_20261025.md`、`docs/ABLATION_A5_RUN_PHYSINT_20261025.md` |
| 机器可读汇总 | `runs/ablations/SUMMARY_1025_v2.{json,md}` |
| token 锚定（方案/执行/结论/平台判定） | `docs/TOKEN_ANCHOR.md`（§7 起跑、§8 B 结论、§9 A 为什么停过、§10 两线并行+平台判据） |
| 交接口径（写死） | `docs/HANDOVER_MAINLINE_AND_ABLATIONS_20261025.md` |
| 数据/链路政策 | `docs/DATA_POLICY_HA_ONLY.md` |
| 平台哨兵 | `scripts/plateau_guard.sh`（判据与用法见文件头） |
| 队列总控 | `scripts/run_abl_1025_queue.sh`、`scripts/queue_anchor_from0_phase2.sh`、`scripts/after_1025_reports.sh` |
