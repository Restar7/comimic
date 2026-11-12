# base × repeat 全网格结果（5 任务 × seeds 0/1/2 × repeat 0–17）

> 生成：2026-11-12 19:30 ｜ 执行：base 线 ｜ 上游：`docs/BASE_REPEAT_PAIRED_HANDOFF_20261111.md`、`docs/BASE_REPEAT_PAIRED_RESULT_20261112.md`、`docs/BASE_REPEAT_EXT_PHASE_A_20261112.md`
> **一句话结论：270 格全部落地（0 缺失、0 重跑），base 142/270 = 52.6%。p21 有参考的 178 格里 base 102 vs p21 102，McNemar p=1.000 —— 逐任务、逐 seed、逐 repeat 段全部"分不出"。PPBox 没有 p21 参考，base 24.1%（块间 28%→0%→44%，极不稳定）。**

---

## 0. 速览

| 项 | 值 |
|---|---|
| 规模 | **5 任务 × seeds 0/1/2 × repeat 0–17 = 270 格** |
| 本次新增 | 240 集（Phase A 60 + Phase B 180）＋ 之前 30 集 = 270 |
| base 总成绩 | **142/270 = 52.6%** |
| 可配对格 | **178**（p21 覆盖 OD 50 / BX 54 / SS 36 / VN 38；PPBox 0） |
| 配对结果（预注册口径） | base **102/178** vs p21 **102/178**；都对 78 / 都错 52 / 只 p21 对 24 / 只 base 对 24；**McNemar p=1.000** |
| 敏感性（纳入 recheck2） | p21 110/178 vs base 102/178，p=0.332（仍不显著；增量全是"只重跑失败格"） |
| 校验 | 270/270 `episode_seed` 一致；30 份 `run_meta` 全 `arms=["base"]`/`max_steps=3000`；`[lora16] loaded`=**0**；240/240 `attempt=1`、`rc=0` |
| 耗时 | 01:23:17 → 19:26:07（**18h02m50s**，两卡并行） |
| 机器 | 2 × RTX 4090，每卡 1 个 base server（18080/18081）＋ 1 个 Isaac sim |

---

## 1. 执行

| 阶段 | 内容 | 时间 | 集数 |
|---|---|---|---|
| 前置 | OpenDoor/Boxing r0–4（30 格） | 11-11 23:29 → 11-12 01:14 | 30 |
| Phase A | SitSofa/VisionNavi/PPBox r0–5 + OD/BX r5 | 01:23 → 06:41（5h18m） | 60 |
| Phase B | 5 任务 × seeds × r6–17 | 06:41 → 19:26（12h45m） | 180 |

- 全部用 `--arms base`、`C2_SERVERS=base`、`C2_SERVER_SEED=20261109`、`--max-steps 3000`、`--workers 1`、`--episode-timeout 3600`、`--record-video 0`。
- 调度器 `scripts/c_wm6/base_repeat_ext_supervisor.py`：两卡并行、每 job 6/12 集、独立 out 目录；**断点续跑**（启动时扫描已完成格）；只对"没有产出 result.json"的格重试。
- 实际执行中**没有任何一格需要重试**：240/240 都是 `attempt=1`、`rc=0`。
- 单集时长：成功中位 ~5 min，失败（跑满 3000 步）10–14 min；Phase B 的 12 集 job 耗时 **59.5–151.7 min**（Boxing 最快、PPBox 最慢）。

---

## 2. 校验（fail-closed，全部通过）

| 检查 | 结果 |
|---|---|
| 网格完整性 | 270/270，0 缺失、0 重复 |
| 首跑一次 | 240/240 `attempt=1`；`try2*` 目录 0 个 |
| 进程正常 | 240/240 `rc=0` |
| `episode_seed` | **270/270** 与 `sha256(<Isaac task 名>\|<seed>\|<repeat>)[-4:] & 0x7fffffff` 一致 |
| `run_meta` | 30/30 `arms=["base"]`、`max_steps=3000` |
| LoRA | 30 份 base server 日志 `[server][lora16] loaded` = **0**；命令行无 `--lora16-ckpt` |
| 结果口径 | `success` 字段，失败原因全部 `timeout`（无崩溃/环境异常） |

> 注：`run_meta.lora_sha=b01e423f…` 是 driver 默认字段（h10_critic），本批**未加载**；按 handoff §4 不作为过滤条件。

---

## 3. base 全网格成绩（270 格）

### 3.1 分任务

| 任务 | 成功/格数 | 成功率 | Wilson 95% CI |
|---|---|---|---|
| OpenDoor | 36/54 | 66.7% | [53.4%, 77.8%] |
| Boxing | 38/54 | 70.4% | [57.2%, 80.9%] |
| VisionNavi | 30/54 | 55.6% | [42.4%, 68.0%] |
| SitSofa | 25/54 | 46.3% | [33.7%, 59.4%] |
| PPBox | 13/54 | 24.1% | [14.6%, 36.9%] |
| **合计** | **142/270** | **52.6%** | — |

### 3.2 分 repeat 段（每段 90 格）

| 段 | base | 说明 |
|---|---|---|
| r0–5 | 43/90 = 47.8% | 含全部 5 任务 |
| r6–11 | 43/90 = 47.8% | |
| r12–17 | 56/90 = 62.2% | 明显更高 |

### 3.3 每任务 × repeat 段

| 任务 | r0–5 | r6–11 | r12–17 |
|---|---|---|---|
| OpenDoor | 13/18 | 12/18 | 11/18 |
| Boxing | 13/18 | 12/18 | 13/18 |
| VisionNavi | 7/18 | 9/18 | **14/18** |
| SitSofa | 5/18 | 10/18 | 10/18 |
| PPBox | 5/18 | **0/18** | 8/18 |

### 3.4 分 seed（每 seed 90 格）

| seed | base |
|---|---|
| s0 | 49/90 = 54.4% |
| s1 | 46/90 = 51.1% |
| s2 | 47/90 = 52.2% |

⇒ 三个 seed 基本持平（±1.7pp），没有 seed 偏置。

---

## 4. 与 p21 的配对（178 格，预注册口径）

**p21 口径**：`p21_seed012_{r0_5,r6_11,r12_17}` 原始结果，存在 `p21_recheck_r0_5` 时覆盖（即 handoff 预注册的"第一轮重跑替换后"）。`p21_recheck2_r0_5` 不进主判据。

### 4.1 分任务

| 任务 | n | p21 | base | 都对 | 都错 | 只有 p21 对 | 只有 base 对 | McNemar p |
|---|---|---|---|---|---|---|---|---|
| OpenDoor | 50 | 30 | 33 | 22 | 9 | 8 | 11 | 0.648 |
| Boxing | 54 | 39 | 38 | 33 | 10 | 6 | 5 | 1.000 |
| SitSofa | 36 | 16 | 15 | 11 | 16 | 5 | 4 | 1.000 |
| VisionNavi | 38 | 17 | 16 | 12 | 17 | 5 | 4 | 1.000 |
| PPBox | 0 | — | — | — | — | — | — | — |
| **合计** | **178** | **102** | **102** | **78** | **52** | **24** | **24** | **1.000** |

### 4.2 分 repeat 段（每段相对独立，因为 p21 三个段来自三个容器）

| 段 | n | p21 | base | 只有 p21 对 | 只有 base 对 | McNemar p |
|---|---|---|---|---|---|---|
| r0–5 | 72 | 36 | 38 | 7 | 9 | 0.804 |
| r6–11 | 72 | 43 | 43 | 10 | 10 | 1.000 |
| r12–17 | 34 | 23 | 21 | 7 | 5 | 0.774 |

### 4.3 分 seed

| seed | n | p21 | base | 只有 p21 对 | 只有 base 对 | McNemar p |
|---|---|---|---|---|---|---|
| s0 | 62 | 35 | 37 | 6 | 8 | 0.791 |
| s1 | 60 | 36 | 34 | 9 | 7 | 0.804 |
| s2 | 56 | 31 | 31 | 9 | 9 | 1.000 |

⇒ **三个 seed、三个 repeat 段、四个任务，没有任何一个维度出现显著差异。**

---

## 5. 敏感性：p21 的重跑红利

| p21 口径 | p21 | base | 只有 p21 对 / 只有 base 对 | McNemar p |
|---|---|---|---|---|
| **预注册：原始 + 第一轮重跑替换（主判据）** | 102/178 | 102/178 | 24 / 24 | **1.000** |
| 原始首跑（不替换，仅参考） | 94/178 | 102/178 | 20 / 28 | 0.288 |
| 原始 + 两轮重跑（recheck2 也算） | 110/178 | 102/178 | 30 / 22 | 0.332 |

- 第一轮重跑：17 格覆盖，**8 格 ❌→✅**；第二轮：16 格覆盖，**又 8 格 ❌→✅**；两轮共加 **16 格，零 ❌→❌ 反向**。
- 三个口径全部不显著；主判据 1.000 是完全打平。
- 再次提醒：这是"只重跑失败格"的单边红利（handoff §2.2 禁止的做法）；base 这边是每格一次首跑、失败不重跑。

---

## 6. 无 p21 参考的格（base-only 基线）

| 任务 | 段 | base |
|---|---|---|
| OpenDoor | r12–17 缺 4 格 | 3/4 = 75.0% |
| SitSofa | r12–17 | 10/18 = 55.6% |
| VisionNavi | r14–17 | 14/16 = 87.5% |
| PPBox | 全部 54 格 | 13/54 = 24.1% |

这些格只作 base 基线，不参与任何显著性结论。

---

## 7. 结论与观察

1. **主判据：分不出。** 178 对里 102 vs 102，完全打平（p=1.000）；分任务、分 seed、分 repeat 段也全部 p≥0.65。**没有任何证据表明 p21 优于 base，也没有证据表明 base 优于 p21。**
2. **block 效应是两边共有的**：base 与 p21 都在 r12–17 明显更高（base 47.8%/47.8%/62.2%，p21 50%/59.7%/67.6%）。说明这是**场景/种子段/容器**带来的共同漂移，不是策略差异——所以**必须段内配对**，跨段比较会得出任意结论。
3. **PPBox 是真正的短板**：base 24.1%，且块间极不稳定（r0–5 28%、r6–11 **0/18**、r12–17 44%）。这个任务没有 p21 数据，属于任务级能力问题，调权重/调 LoRA 不太可能解决。
4. **VisionNavi 在 r12–17 反而最强**（14/18），说明任务难度排序本身依赖 repeat 段；单一 aggregate 数字容易误导。
5. **噪声仍然是判据的瓶颈**：p21 两次重跑能在 33 格里翻 16 格（48%），远大于任何"方法提升"的效应量。任何 5–10pp 的改进，用 30–100 对配对都测不出来。

---

## 8. 文件

| 文件 | 说明 |
|---|---|
| `docs/BASE_REPEAT_FULL_GRID_20261112.csv` | **270 格机读表**（task/seed/repeat_idx/p21/p21_steps/p21_source/base_result/base_steps/base_reason） |
| `docs/BASE_REPEAT_PAIRED_RESULT_20261112.md` | 最早的 30 格配对报告（OpenDoor/Boxing r0–4） |
| `docs/BASE_REPEAT_EXT_PHASE_A_20261112.md` | Phase A（5 任务 r0–5，90 格）报告 |
| `docs/BASE_REPEAT_REF_P21_30CELLS_20261111.csv` | 最早的 30 格配对 CSV（已回填） |
| `docs/BASE_REPEAT_EXT_R0_5_20261112.csv` | 90 格 r0–5 机读表 |
| `runs/dual_actor_wm6/c_lora/c2/base_repeat_ext/` | 240 集原始结果 + `summary.json` + `progress.json` + `supervisor.log` + `full_grid_report.json` |
| `runs/dual_actor_wm6/c_lora/c2/base_repeat_s012_r0_4/` | 前置 30 集原始结果 |
| `scripts/c_wm6/base_repeat_ext_supervisor.py` | 断点续跑调度器 |
| `scripts/c_wm6/agg_base_repeat_full.py` | 全网格配对汇总（可复跑） |

### 复跑

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
# 汇总（校验 + 配对 + McNemar + CSV）
python3 scripts/c_wm6/agg_base_repeat_full.py \
  --csv docs/BASE_REPEAT_FULL_GRID_20261112.csv \
  --json runs/dual_actor_wm6/c_lora/c2/base_repeat_ext/full_grid_report.json

# 若还有缺口，重跑调度器会自动补（已完成格跳过）
python3 scripts/c_wm6/base_repeat_ext_supervisor.py --dry-run
```
