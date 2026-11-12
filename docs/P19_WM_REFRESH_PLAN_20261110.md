# p19 计划：用六任务闭环 ticks 更新 WM → 从 C 基线联合微调 → 闭环验证

> 拟定于 2026-11-10 18:20 ｜ 执行者：C 线 Agent ｜ **本文只写计划，尚未执行**
> 核心假设：**当前 WM（裁判）不认识"闭环爬行区"，所以策略优化拿不到修复所需的梯度。**
> 做法：起点分布**保持不变**（drift ≤ 0.10），只把 WM 用真实闭环数据补强，再从 **C 基线**做联合微调。

---

## 0. 一句话方案

```
第0步 口径对齐 + WM 诊断（20 min，决定要不要继续）
  ↓ 通过
第1步 把闭环 ticks 转成 WM 的 replay 格式（15 min）
  ↓
第2步 WM 增量训练（旧数据 + 闭环数据混合，300–600 updates，~20 min）
  ↓ Gate W：闭环上单步误差显著下降、录制窗口上不退化
第3步 从 C 基线联合微调（--init-lora + 新 WM，原配方 300 步，~35 min）
  ↓ Gate P：训练健康 + Isaac 开环门
第4步 2 路并发闭环（5 任务 × 16 seeds，c vs 新 ckpt，~4 h）
  ↓ 决策：提升 ≥2 seed 且不全线掉 → 才考虑替换交付物
```

---

## 1. 为什么这件事可能有用（以及它的硬边界）

**可能有用**：WM 是策略的**唯一裁判**。它若在"偏离轨迹后"完全不准，那么无论奖励怎么调，策略得到的信号都是错的。

**硬边界（必须承认）**：

1. 训练起点仍是 `drift ≤ 0.10`（本计划**不扩大**），所以想象轨迹仍从"轨迹附近"出发——WM 变准只改善**裁判质量**，不改变**考场分布**；
2. 闭环状态**没有对应的 VLA token**（ticks 只存状态/动作，没存图像），所以本轮**不能用闭环状态做起点**；
3. 因此本计划的合理预期是：**"同样的考场、更准的裁判" → 策略在近轨迹区域的动作质量提升**，可能转化为更少的爬行坍缩，但**不保证**。

---

## 2. 现有资产（已核对）

| 对象 | 路径 / 数值 |
|---|---|
| 当前 WM | `comimic-wt03/runs/wm_base6_v1/wm_candidates/wm_base6_best_candidate/wm.pt`（sha `8edb59cf…`） |
| WM 增量训练脚本 | `comimic-b-wm6/scripts/train_wm_refresh_v7.py`（混合 replay + 版本库，**不改原 WM 文件**） |
| replay 格式 | `dict(states[71], next_states[71], actions[29], 可选 ep_ids/clip_ids)`，float32，fail-closed 校验 |
| C 基线适配器 | `critic10/lora300_ddp/checkpoint_lora_best.pt`（sha `6942eab9`） |
| 训练脚本能力 | 支持 `--init-lora`（只加载 adapter，不续优化器）与 `--wm-checkpoint` + `--strict-wm-hash` |
| **闭环 ticks 总量** | **≈ 925,216 条转移**（见下表） |

| 任务 | tick 文件 | ≈ 转移条数 |
|---|---|---|
| OpenDoor | 14,721 | 235,536 |
| VisionNavi | 11,298 | 180,768 |
| PPBox | 9,335 | 149,360 |
| SitSofa | 8,387 | 134,192 |
| DoubleDesk | 8,228 | 131,648 |
| Boxing | 5,857 | 93,712 |
| **合计** | **57,826** | **≈ 925,216** |

> 覆盖两种臂（c / base）、两种并发档（2 路 / 4 路）、多个种子——**数据多样性足够**，但要按 episode 做 train/val 切分，避免同集泄漏。

---

## 3. 第 0 步：口径对齐 + WM 诊断（**必须先做，20 min**）

闭环 ticks 是"recorder"口径，未必要和 WM 训练口径一致。**两个必须核对的点**：

| 口径 | ticks 里的现状 | 风险 |
|---|---|---|
| **state 顺序** | q 已转 **STANDARD** 序（`perm_ha_to_std`） | WM 若吃 HA 序，喂错会静默变差 |
| **action 顺序/尺度** | `actions_model = (q_target − dq_ha)/sc_ha`，**HA 序** | WM 的 ACTION_DIM=29，若期望 STANDARD 序则需置换 |

**做法**：抽 2000 条闭环转移，分别按 (a) actions 原样、(b) actions 置换到 STD 序 两种口径，喂当前 WM 算**单步预测误差**（MSE，逐维归一化口径同 `ssm_wm_loss`），取误差更小者作为正确口径。

**同时得到 Gate W 的基线**：同一批转移上，录制窗口（`eval_set` train 窗口）的单步误差作为对照。

| 判据 | 阈值 | 含义 |
|---|---|---|
| 闭环误差 / 录制误差 | **≥ 3×** | **GO**：WM 确实不认识闭环区，补训有空间 |
| | **≤ 1.5×** | **NO-GO**：WM 不是瓶颈，本计划终止（省下后续 5 h） |
| 两种口径误差差 | 明显 | 说明口径错了，**必须先修正转换脚本** |

产出：`runs/dual_actor_wm6/c_lora/p19/diag/wm_error_report.json`

---

## 4. 第 1 步：把 ticks 转成 replay（15 min）

新增脚本 `scripts/c_wm6/ticks_to_wm_replay.py`：

* 输入：`runs/dual_actor_wm6/c_lora/c2/**/ticks/ticks_*.pt`（六任务全量，可按需过滤并发档位）；
* 每条 tick 文件含 16 行：`states_before / states_after / actions_model / frame_count / task_id / seed`；
* 输出：`all_closedloop.pt`，字段 = `states / next_states / actions`（口径按第 0 步结论）+ `ep_ids`（=`{task}:{arm}:s{seed}`）+ `clip_ids`（= task）；
* **抽样策略**：每个 episode 最多取前 3000 tick（全取），总量按任务均衡降采样到 ~40 万条，避免 DoubleDesk/PPBox 的全超时数据淹没其它任务；
* **切分**：`ep_ids` 做确定性哈希 → 80/20 train/val，**同集不跨边**（脚本自带）。

---

## 5. 第 2 步：WM 增量训练（~20 min，2 卡中的 1 卡）

```bash
cd comimic-b-wm6
source scripts/harness03_env.sh
python3 scripts/train_wm_refresh_v7.py \
  --version-store runs/.../wm_versions --new-version wm_p19 \
  --old-replay <录制数据的 all_transitions.pt> \
  --new-replay <第1步产出的 all_closedloop.pt> \
  --old-ratio 0.5 --new-ratio 0.5 \
  --mode single --updates 400 --lr 5e-6 --batch 256 --val-fraction 0.2 \
  --device cuda:2
```

**关键选择与理由**

| 选择 | 理由 |
|---|---|
| `--mode single`（不用 kstep） | 闭环 ticks 是真实转移，kstep 自回归会放大误差；先做最保守的一步监督 |
| `old:new = 0.5:0.5` | 全用闭环数据会忘掉录制分布（而评测/训练窗口都在录制分布上）；50/50 起步，留 90/10 对照空间 |
| `lr 5e-6`（而非 1e-5） | 增量更新，目标是"补盲区"而不是重训 |
| 版本库（`--version-store`） | **不覆盖原 WM**，随时可回滚 |

### Gate W（两条都要过）

| # | 判据 | 阈值 |
|---|---|---|
| 1 | 闭环 val 上单步误差 | 相对旧 WM **下降 ≥30%** |
| 2 | 录制窗口（`eval_set` train）上单步误差 | **不退化**（≤ +5%） |

不过 → 调 `old:new`（0.7:0.3）或 `lr`（2e-6）重试一次；再不过 ⇒ 本计划终止。

---

## 6. 第 3 步：从 C 基线联合微调（~35 min，2 卡 DDP）

**保持原配方不变**，只换裁判与初值：

| 项 | 值 | 备注 |
|---|---|---|
| 初值 | `--init-lora critic10/lora300_ddp/checkpoint_lora_best.pt` | 从交付物出发（不续优化器） |
| 裁判 | `--wm-checkpoint <新 WM>` + `--strict-wm-hash <新 sha>` | 只有在 Gate W 通过后才用 |
| λ | `--lambda-lim 0.0306794`、**无 margin**、无 recovery | 与交付物同配方 |
| 起点 | `--start-max-drift 0.10` | **按你的要求：不改** |
| H / K / batch / lr | `10 / 20 / 16 / 1e-4` | 同交付物 |
| 步数 | **300**（`--lr-total-updates 300`） | 与历史可比 |

输出：`p19/ft300/`（新目录，绝不动 `cmkm300_ddp` 与 `lora300_ddp`）

### Gate P（训练健康，缺一不可）

| # | 判据 | 阈值 |
|---|---|---|
| 1 | val `dJ` | 不塌（≥ 0.0035，且 CI 不是断崖式下降） |
| 2 | 越限率 | ≤ 2.9% |
| 3 | `q_raw_absmax` | ≤ 4.5 |
| 4 | NaN / grad 爆炸 | 无 |

---

## 7. 第 4 步：闭环验证（2 路并发，**5 任务 × 16 seeds**，~4 h）

**对比**：`C 基线` vs `p19 新 ckpt`，同一批 seed，strict 单次尝试（**禁止 best-of**）。

| 任务 | seeds |
|---|---|
| OpenDoor、Boxing、VisionNavi、SitSofa、PPBox | 16 个（本轮先用 50→35 之外的新段，如 20–35） |
| DoubleDesk | 不跑（两臂恒 0） |

**决策规则（先写死）**

| 结果 | 结论 |
|---|---|
| 成功率 ↑ ≥2 seed（McNemar p<0.05）且无任务掉 >1 seed | **成立**，进 30-seed 复核 |
| 持平（|Δ|≤1 seed） | 不成立，但记录"裁判变准≠策略变强"这条负结果 |
| 普遍下降 | 回滚，写负结果 |

---

## 8. GPU 排期（重要，会互相污染）

| 时段 | GPU0/1 | GPU2/3 |
|---|---|---|
| 现在 – 明天 | 正在跑 `ab2way_s35_50`（160 集，约 8–9 h） | **空闲** |
| 第 0–2 步 | 继续跑评测 | **做诊断 + WM 训练（单卡）** |
| 第 3 步（策略 DDP） | **必须让出**：训练占用 2 卡会把评测延迟拉到 2×（已验证会压垮闭环结果） | 同左 |

> **结论**：第 3 步开始前，**先停掉正在跑的评测**（约损失 1 小时残余进度），或等 160 集跑完（明早）再训练。二选一，**不要同时跑**。

---

## 9. 产物与保护

| 项 | 位置 |
|---|---|
| 诊断报告 | `p19/diag/wm_error_report.json` |
| 闭环 replay | `p19/replay/all_closedloop.pt` |
| 新 WM 候选 | `comimic-b-wm6/.../wm_versions/wm_p19/`（**不覆盖原 WM**） |
| 策略 ckpt | `p19/ft300/checkpoint_lora_best.pt` |
| 评测结果 | `c2/p19_vs_c/` |
| **保护** | `critic10/lora300_ddp/checkpoint_lora_best.pt`（交付物）**全程只读**；`wm_base6` 原文件只读 |

---

## 10. 时间线

| 步骤 | 成本 |
|---|---|
| 第 0 步 诊断 | 20 min |
| 第 1 步 转格式 | 15 min |
| 第 2 步 WM 训练 + Gate W | 30 min |
| 第 3 步 策略微调 + Gate P | 45 min |
| **小计（到出 ckpt）** | **~2 h** |
| 第 4 步 闭环 160 集 | ~4 h |
| **总计** | **~6 h** |

---

## 11. 风险清单

| 风险 | 概率 | 处理 |
|---|---|---|
| ticks 的 action 口径与 WM 不一致 | 中 | 第 0 步两种口径都测，取误差小者；必要时写置换 |
| 闭环数据把 WM 拉偏（忘掉录制分布） | 中 | `old:new` 从 0.5 起，Gate W 第 2 条卡住；不过就退回 0.7:0.3 |
| 裁判变准但策略不变强 | **高** | 这是最可能的结果（考场没变）；用第 4 步决策规则如实记录 |
| 训练与评测抢卡导致数据污染 | 高 | 第 8 节：**串行**，不并行 |
| 显存（4090 24 GB / 本机 48 GB） | 低 | WM 训练的 batch 256 可降到 128 |

---

## 12. 开工前需要你确认的三件事

1. **是否接受"不改起点分布"的预期上限**（本计划的核心局限）；
2. **第 3 步前是否停掉正在跑的 160 集评测**（不停会把两边数据都污染）；
3. **WM 训练用哪份 old-replay**（我会用录制数据的 `all_transitions.pt`；若你希望只用六任务相关的录制数据，请指明）。
