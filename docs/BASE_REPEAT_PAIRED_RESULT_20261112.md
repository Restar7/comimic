# base × repeat 协议配对结果（OpenDoor + Boxing，seeds 0/1/2，repeat 0–4）

> 生成：2026-11-12 01:20 ｜ 执行：base 线 ｜ 对应交接：`docs/BASE_REPEAT_PAIRED_HANDOFF_20261111.md`
> **一句话结论：该协议下 base 与 p21 分不出**（30 对里 base 23/30 vs p21 21/30，McNemar 精确检验双侧 **p=0.727**）。
> 方向上是 base 不劣于 p21（不一致格 5 vs 3），但 30 对只能分辨 ≥30pp 的差，当前 6.7pp 远在噪声内——按预注册**不许**改口成"base 更好"。

---

## 0. 30 秒速览

| 项 | 值 |
|---|---|
| 评测对象 | 官方 FT8000 `comimic/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`，**不挂 LoRA** |
| 臂 | `--arms base`（`run_meta.arms = ["base"]`；driver 里只有 base/c 两臂，无 `c_base`） |
| 格数 | **30/30 落地**，0 缺失、0 重复、0 重跑（全部 `attempt=1`、`rc=0`） |
| base 成绩 | **23/30 = 76.7%**（OpenDoor 11/15；Boxing 12/15） |
| p21 成绩（预注册口径） | **21/30 = 70.0%**（OpenDoor 11/15；Boxing 10/15） |
| 四格组合 | 都对 **18** ｜ 都错 **4** ｜ 只有 p21 对 **3** ｜ 只有 base 对 **5** |
| 主判据 | McNemar 精确检验（双侧）**p = 0.727** ⇒ **不显著** |
| 结论 | **该协议下分不出**；不能声称 base 更好或 p21 更好 |
| 校验 | `episode_seed` **30/30** 与 sha256 公式一致；`[lora16] loaded` 计数 **0**；`max_steps=3000`；`arms=["base"]` |
| 耗时 | 2026-11-11 23:29:38 → 2026-11-12 01:14:30（**1h44m52s**，两卡并行） |
| 机器 | 2 × RTX 4090（本机），每卡 = 1 个 base server + 1 个 Isaac sim |

---

## 1. 执行与可复现性

### 1.1 命令（与 handoff §6 完全一致）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh
OUT=runs/dual_actor_wm6/c_lora/c2/base_repeat_s012_r0_4

# card0（GPU0，OpenDoor）
export C2_SERVER_SEED=20261109 C2_SERVERS=base C2_PORT_BASE=18080 C2_GPU_BASE=0 C2_ISAAC_GPU=0
nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks OpenDoor --seeds 0 1 2 --repeat-start 0 --repeats 5 --arms base \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out "$OUT/card0" > "$OUT/card0.log" 2>&1 &

# card1（GPU1，Boxing）
export C2_SERVER_SEED=20261109 C2_SERVERS=base C2_PORT_BASE=18081 C2_GPU_BASE=1 C2_ISAAC_GPU=1
nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks Boxing --seeds 0 1 2 --repeat-start 0 --repeats 5 --arms base \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out "$OUT/card1" > "$OUT/card1.log" 2>&1 &
```

- 代码：worktree `comimic-c-wm6`，HEAD **4513b42**；`run_meta.main_git = bb6c355`。
- server 与 sim 用同一套代码；两臂（若跑 c）唯一差别是 `--lora16-ckpt`，本单不追加。
- 失败集全部是 `timeout`（跑满 3000 步），不是崩溃/环境异常；**没有**任何一格因环境问题重跑。

### 1.2 环境注记（不影响判据口径，但必须记录）

1. **`gpu_utilizer`（flowbody，非本栈）**：20:39 起在两卡各占 ~490 MiB、~15% duty。它启动于 p21 批次中途，而 base 这批**全程**在它下面跑。它不改物理步进，但"机器不干净"，与 handoff §9.1 的理想状态有差。
2. **p21 第二轮重跑（`p21_recheck2_r0_5`）在 base 跑的同时仍在写盘**（最后落盘 23:58），它覆盖本 30 格中的 **1 格**（Boxing s2r3）。按预注册不采用，见 §5 敏感性。
3. base 与 p21（r0–5 批）跑在**同一台机器**、同一 worktree、同一 `C2_SERVER_SEED=20261109`。

---

## 2. 校验（fail-closed 项，全部通过）

| 检查 | 方法 | 结果 |
|---|---|---|
| 集数完整 | `find ... -name result.json \| wc -l` | **30/30**，0 缺失、0 重复 |
| 首跑一次 | `jobs.jsonl` 的 `attempt` | **30/30 都是 attempt=1**；`try2*` 目录数 = 0 |
| 进程正常 | `jobs.jsonl` 的 `rc` | **30/30 rc=0** |
| 种子对齐 | `episode_seed == sha256("<完整 Isaac task 名>\|<seed>\|<repeat>")[-4:] & 0x7fffffff` | **30/30 一致，mismatch=0** |
| base 纯度 | `run_meta.arms` | 两卡均 **`["base"]`** |
| base 纯度 | server 日志 `[server][lora16] loaded` | **0 次**（两卡） |
| base 纯度 | server 命令行 `--lora16-ckpt` | **无**（两卡进程逐个核对） |
| 步数口径 | `run_meta.max_steps` / result.json | 两卡 **3000** |
| 结果口径 | `success` 字段 | 与本报告一致 |

> 注：`run_meta.lora_sha = b01e423f…` 是 driver 的**默认字段**（h10_critic），本单**未加载**；按 handoff §4 不用它做过滤。

---

## 3. 逐格配对表（30 对，主表）

`p21` 列 = handoff 预注册口径（`p21_recheck_r0_5` 覆盖 `p21_seed012_*`，即"重跑替换后"）；`base` 列 = 本单首次尝试。

| 任务 | seed | r | p21 | p21 步数 | base | base 步数 | 组合 | p21 来源 |
|---|---|---|---|---|---|---|---|---|
| OpenDoor | 0 | 0 | ❌ | 3000 | ✅ | 916 | 只有 base 对 | recheck |
| OpenDoor | 0 | 1 | ✅ | 638 | ✅ | 607 | 都对 | original |
| OpenDoor | 0 | 2 | ❌ | 3000 | ❌ | 3000 | 都错 | recheck |
| OpenDoor | 0 | 3 | ✅ | 1084 | ✅ | 888 | 都对 | recheck |
| OpenDoor | 0 | 4 | ✅ | 1696 | ✅ | 1100 | 都对 | original |
| OpenDoor | 1 | 0 | ✅ | 734 | ✅ | 2244 | 都对 | original |
| OpenDoor | 1 | 1 | ❌ | 3000 | ✅ | 2594 | 只有 base 对 | recheck |
| OpenDoor | 1 | 2 | ✅ | 2821 | ✅ | 1264 | 都对 | recheck |
| OpenDoor | 1 | 3 | ✅ | 1538 | ✅ | 1021 | 都对 | original |
| OpenDoor | 1 | 4 | ❌ | 3000 | ❌ | 3000 | 都错 | recheck |
| OpenDoor | 2 | 0 | ✅ | 1239 | ❌ | 3000 | 只有 p21 对 | original |
| OpenDoor | 2 | 1 | ✅ | 1132 | ✅ | 1432 | 都对 | original |
| OpenDoor | 2 | 2 | ✅ | 712 | ✅ | 1752 | 都对 | recheck |
| OpenDoor | 2 | 3 | ✅ | 750 | ❌ | 3000 | 只有 p21 对 | original |
| OpenDoor | 2 | 4 | ✅ | 2691 | ✅ | 1764 | 都对 | original |
| Boxing | 0 | 0 | ✅ | 334 | ✅ | 381 | 都对 | original |
| Boxing | 0 | 1 | ✅ | 1395 | ✅ | 286 | 都对 | original |
| Boxing | 0 | 2 | ❌ | 3000 | ✅ | 1909 | 只有 base 对 | recheck |
| Boxing | 0 | 3 | ❌ | 3000 | ❌ | 3000 | 都错 | recheck |
| Boxing | 0 | 4 | ✅ | 399 | ✅ | 1157 | 都对 | original |
| Boxing | 1 | 0 | ❌ | 3000 | ❌ | 3000 | 都错 | recheck |
| Boxing | 1 | 1 | ✅ | 860 | ❌ | 3000 | 只有 p21 对 | recheck |
| Boxing | 1 | 2 | ✅ | 399 | ✅ | 549 | 都对 | recheck |
| Boxing | 1 | 3 | ✅ | 1694 | ✅ | 528 | 都对 | recheck |
| Boxing | 1 | 4 | ✅ | 257 | ✅ | 360 | 都对 | original |
| Boxing | 2 | 0 | ✅ | 423 | ✅ | 548 | 都对 | recheck |
| Boxing | 2 | 1 | ❌ | 3000 | ✅ | 2520 | 只有 base 对 | recheck |
| Boxing | 2 | 2 | ✅ | 397 | ✅ | 602 | 都对 | original |
| Boxing | 2 | 3 | ❌ | 3000 | ✅ | 2714 | 只有 base 对 | original |
| Boxing | 2 | 4 | ✅ | 667 | ✅ | 1006 | 都对 | original |

> 机读版：`docs/BASE_REPEAT_REF_P21_30CELLS_20261111.csv` 的 `base_result` 列已按上表填好（30/30，无空值）。

---

## 4. 主判据（预注册，跑前写死）

### 4.1 合计

| 统计量 | 值 |
|---|---|
| 配对数 | 30 |
| 都对 | 18 |
| 都错 | 4 |
| 只有 p21 对 | **3** |
| 只有 base 对 | **5** |
| p21 | 21/30 = 70.0% |
| base | 23/30 = 76.7% |
| 差 | +6.7pp（base 高） |
| **McNemar 精确检验（双侧）** | **p = 0.727** ⇒ 不显著 |

### 4.2 分任务（仅描述，按预注册不做显著性结论）

| 任务 | 对 | p21 | base | 只有 p21 对 | 只有 base 对 | McNemar p |
|---|---|---|---|---|---|---|
| OpenDoor | 15 | 11/15 | 11/15 | 2 | 2 | 1.000 |
| Boxing | 15 | 10/15 | 12/15 | 1 | 3 | 0.625 |

### 4.3 采样量限制（必须随结论一起报）

- 30 对只能看出 **≥30 个百分点**的差；本次只差 6.7pp，**注定分不出**。
- 两个比例的 Wilson 95% CI 大幅重叠：base [59.1%, 88.2%]、p21 [52.1%, 83.3%]。
- 按观察到的 8/30 不一致率外推（McNemar，α=0.05、power=0.8）：分辨 6.7pp 需 **≈581 对**；10pp 需 ≈285 对；20pp 需 ≈89 对；30pp 需 ≈47 对。

---

## 5. 敏感性分析

| p21 口径 | p21 | 只有 p21 对 | 只有 base 对 | McNemar p | 说明 |
|---|---|---|---|---|---|
| **预注册：重跑替换后（主判据）** | 21/30 | 3 | 5 | **0.727** | `p21_recheck_r0_5` 覆盖；对 p21 有利 |
| 纳入 `p21_recheck2_r0_5` | 22/30 | 3 | 4 | 1.000 | 该批不在预注册口径内，且跑 base 时仍在写盘 |
| 原始首跑（不替换；仅参考） | 14/30 | 2 | 11 | 0.022 | 把 p21 批次的机器批次效应算进算法，**不能当结论** |

- 主判据只用第一行。第三行说明：p21 那批的"重跑红利"正好是 7 格 ❌→✅；如果当初不重跑，数字会低得多——但这恰是 handoff §2.2 禁止的行为，所以不能反过来当 p21 的真实水平。
- 纳入 recheck2 后 p 从 0.727 变 1.000，结论方向不变（分不出）。**结论对这两项口径选择是稳健的。**

---

## 6. 描述性观察（不作判据）

1. **为什么 base 看起来强（不是 bug）**：base 在同两任务的 `/reseed` 历史率是 OpenDoor 60.0%（18/30）、Boxing 75.7%（28/37），合并 **68.7%**；本单 23/30 = 76.7%，二项检验 P=0.32，**完全在自身历史噪声内**。而"base 50%"那个数含 PPBox 25%、DoubleDesk 0%，是被难任务稀释的均值。
2. **p21 的 70% 反而是高侧**：p21 在 repeat 协议同两任务的历史率只有 **57.0%**（69/121）；30 格里的 21/30 靠重跑替换抬高（原始 14/30 = 46.7%）。所以本次配对里 p21 = 70.0% 已是对它最有利的读数。
3. **共同成功格的步数**：18 个"都对"格里，p21 成功步数中位 **723**、base **947**（p21 更"短"）；但 base 的 23 个成功集整体中位 1021（286–2714）。单集步数不是预注册判据，仅记录。
4. **失败形态**：base 7 个失败、p21 9 个失败，全部是 3000 步 timeout，无崩溃、无环境异常。
5. **时长**：30 集总 episode 时长 201.6 min；单集中位 5.1 min；失败集 9–13 min（跑满 3000 步）。

---

## 7. 结论（按预注册口径）

1. **主判据：该协议下 base 与 p21 分不出**（23/30 vs 21/30，McNemar p=0.727）。按 handoff §2.4，**不写**"base 更好/更差"。
2. **方向性观察（不显著）**：base 不劣于 p21——不一致格 5 vs 3，且 base 的 76.7% 落在它自己的历史区间内。
3. **机制上一致**：p21 的 LoRA 是 WM 预测向微调（rank16、α=32、1.03% 参数、`lora_B_absmean=7.8e-4`），val `dJ=+0.00217`（相对 J≈0.589 仅 **+0.37%**）。没有机制支撑 30pp 级的闭环提升；本次也没观察到。
4. **要把"分不出"变成"能分辨"**：按本次不一致率，需要 **~285 对**才能分辨 10pp、**~581 对**才能分辨 6.7pp。当前 30 对的用途是"排除 p21 大幅领先"，不是"证明 base 更好"。
5. **建议**：该协议下不要再加格；若要做胜负判定，应换更敏感的指标（成功步数/接近成功率）或扩大配对量；p21 若要继续，方向应是 token→动作/DAgger，而不是继续压 dJ。

---

## 8. 交付物清单（handoff §10）

- [x] `base_repeat_s012_r0_4/card0/`、`card1/` 各 15 个 `result.json`（共 30；全部 `attempt=1`、`rc=0`）
- [x] 两份 `run_meta.json`（`arms=["base"]`、`max_steps=3000`、`repeat_start=0/repeats=5`）
- [x] 两份 server 日志（`[server][lora16] loaded` = 0；无 `--lora16-ckpt`）
- [x] `episode_seed` 核对：**mismatch = 0**（30/30）
- [x] §8 的配对汇总表（含四格组合、分任务、McNemar、采样量限制）
- [x] 一句话结论：**该协议下 p21 与 base 分不出**

### 文件

| 文件 | 说明 |
|---|---|
| `runs/dual_actor_wm6/c_lora/c2/base_repeat_s012_r0_4/{card0,card1}/` | 30 集结果 + 两份 run_meta + `card*.log` + `servers/base.log` |
| `runs/dual_actor_wm6/c_lora/c2/base_repeat_s012_r0_4/paired_report.json` | 结构化配对结果（表格 + 统计 + 校验问题列表） |
| `scripts/c_wm6/agg_base_repeat_paired.py` | 校验 + 配对 + McNemar + 回填 CSV 的脚本（可复跑） |
| `docs/BASE_REPEAT_REF_P21_30CELLS_20261111.csv` | `base_result` 列已回填（30/30） |

---

## 附：复跑方式

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
python3 scripts/c_wm6/agg_base_repeat_paired.py \
  --csv /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/docs/BASE_REPEAT_REF_P21_30CELLS_20261111.csv \
  --out runs/dual_actor_wm6/c_lora/c2/base_repeat_s012_r0_4 \
  --write-csv --json runs/dual_actor_wm6/c_lora/c2/base_repeat_s012_r0_4/paired_report.json
# 退出码 0 = 无缺失、无校验问题
```
