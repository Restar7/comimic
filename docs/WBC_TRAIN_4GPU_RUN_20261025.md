# WBC 4×4090 全 clip 训练：运行记录与验证（2026-10-25 11:07 UTC）

> 按 `docs/WBC_TRAIN_4GPU_BONES_SEED.md` 执行。**本容器（comimic4）4 张 4090 全部投入**，
> 其余所有训练/队列/哨兵**已全部停止**（11:01 前已清空：anchor 续训、A2/A5 链、平台哨兵、后处理器）。

## 0. 状态

> **2026-10-25 11:17 已按用户指令停止**：停在 **it 500/32000**，`ckpt_000500.pt` 已保留，
> 4 张卡全部释放（489 MiB）。停止前最后一行曲线：`sim_r 0.458 / ep_len 40.1 / term 0.85`
> （起步阶段，术语注意：这里的 `ep_len 40` 是"随机起点到 clip 结尾"的剩余长度，见 §4.1，
> **不能**与旧 run 的 `ep_len 216` 直接比）。要续训：同命令 + `RESUME=logs/…/ckpt_000500.pt`。

| 项 | 值 |
|---|---|
| 起跑 | **11:01:26**（torchrun 拉起），rank 11:01:45 起加载 |
| 形态 | `NPROC=4` × `ENVS_PER_RANK=8192` = **32,768 envs**，DDP |
| 进度 | 11:17 停在 **it 500/32000**（`ckpt_000500.pt` 已落盘），`sim_r 0.458 / ep_len 40.1 / term 0.85` |
| 速度 | 编译预热后 **≈1.2 it/s** ⇒ 跑满 32000 iter 约 **7.5 h**；显存 **19 → 26.0 GB/卡**（随 replay buffer 涨，预计 30–35 GB 封顶） |
| 日志 | `trackerlab-supermimic/logs/bones_seed_fullclip_4x8192.launcher.log`（4 rank 合并输出） |
| 曲线 | `trackerlab-supermimic/logs/bones_seed_token_fullclip_4x8192/train_log.csv`（每 50 iter 一行） |
| ckpt | 同目录 `ckpt_*.pt`，**每 500 iter** 一份。**按 `sim_r` + `ep_len` 挑，不要用 `last.pt`** |

复现命令（与文档 TL;DR 一致，已按实跑核对）：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic
export SONIC_ENCODER_ONNX=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx
MODE=train NPROC=4 ENVS_PER_RANK=8192 ITERATIONS=32000 \
MAX_EPISODE_STEPS=0 \
LIBRARY=data/bones_seed/bones_seed_compact.pt \
INTERFACE=token COMPILE=1 HA_ALIGN=1 CKPT_EVERY=500 \
PORT_BASE=29660 LOG_DIR=logs/bones_seed_token_fullclip_4x8192 \
setsid nohup bash scripts/rls/supermimic/train_bones_seed.sh \
  > logs/bones_seed_fullclip_4x8192.launcher.log 2>&1 &
```

## 1. "所有 episode、所有长度都用到"——已逐条验证（不是靠文档）

证据 1：**实跑参数**（`logs/bones_seed_token_fullclip_4x8192/args.json` → `arguments`）

| 参数 | 实测值 | 含义 |
|---|---|---|
| `max_episode_steps` | **0** | CLI 传 0 → runner 置为 `2**31-1`，**episode 只由 clip 结束触发** |
| `task.max_episode_steps`（生效值） | **2147483647** | 同上，已确认落到 task cfg 里（`args.json → task`） |
| `num_envs` | 8192 | 每 rank |
| `iterations` | 32000 | — |
| `shard_motion` / `shard_mode` | `True` / `interleave` | 129,943 条 clip 按 rank 交错分片 |
| `motion_packed` | `…/data/bones_seed/bones_seed_compact.pt` | **完整库**（不是子集） |
| `compact_library` / `compile` / `amp` | True / True / bf16 | — |
| `interface` | `token` | 64 维 SONIC token 在线编码（需 `SONIC_ENCODER_ONNX`，已就位） |

证据 2：**库本身**（`meta.json` + 直接读 `bones_seed_compact.pt` 的 `lengths`）

| 项 | 值 |
|---|---|
| clip 数 | **129,943** |
| 总帧数 | **47,475,680**（50 Hz ⇒ **263.75 h**） |
| 长度分布 | min 29 / p10 149 / **median 288** / p90 628 / p99 1,531 / **max 8,979** / mean 365.4 帧 |
| 短于 1 s（50 帧）的 clip | 存在（min 29 帧 = 0.58 s）⇒ 短 clip 也照跑 |

证据 3：**每 rank 分片 = 全库的 1/4**（`logs/…/motion_library.json`）

| 项 | 值 |
|---|---|
| 本 rank clip 数 | **32,486**（×4 ≈ 129,943） |
| 本 rank 帧数 | **11,848,839**（= 全库 **24.96%**） |
| 本 rank 长度范围 | min 29 / max 8,979（**长短都在**，没有按长度过滤） |

⇒ **结论：这一轮会把 129,943 条 clip 的每一条都跑到 clip 自然结束**，不存在"500 步截断"。

### 1.1 数据真实性复核（2026-10-25 11:15，逐项从库文件里读出来，不依赖任何文档）

| 检查 | 结果 | 判定 |
|---|---|---|
| `len(lengths)` / `len(starts)` / `len(names)` | **129,943 / 129,943 / 129,943** | ✅ 三者一一对应 |
| `sum(lengths)` vs `ref.shape[0]` | **47,475,680** vs **47,475,680**（`ref` 是 `(47475680, 71)` fp16） | ✅ 帧数与张量逐位一致 |
| `starts` 是否连续 | `starts[0]=0`，且 `starts[i+1] == starts[i]+lengths[i]` 全部成立、末尾闭合 | ✅ **无空洞、无重叠**（不是抽样拼的） |
| `names` 去重 | **unique = 129,943**（重复 0） | ✅ 每条 clip 都是独立条目，不是同一段重复计数 |
| 长度分布 | min 29 / p10 149 / **median 288** / p90 628 / p99 1,531 / **max 8,979** | ✅ 长尾在（>5000 帧的有 **114** 条） |
| `>500` 帧的 clip | **21,714 条 = 16.7%** | ✅ 与"旧上限截断 16.7% clip"一致 |

**关于"文档 §1 的数字对不上"——已查明原因，不是数据问题**：

```
BONES-SEED（本库，129,943 clips）：median 288 / mean 365.4 / max 8,979
HA 库（data/npz/ha_tokens，699 clips / 634,016 帧）：median 789 / mean 907.0 / max 10,000
                                                        ↑ 文档 §1 写的正是这三个数 ⇒ 贴错了对象
```

⇒ 数据侧**没有任何问题**：129,943 条 clip、263.75 h、无重复、无空洞；HA 的 699 条才是"少"，BONES-SEED 是它的 **186 倍**。

### 1.2 4 个 rank 的分片是互补的（不是各自只跑一小部分）

`interleave` 分片按索引轮转：每个 rank **32,486 条**（4 × 32,486 = 129,944 ≈ 129,943，其中一条 rank 拿到 32,485），
每个 rank 的帧数 **11,848,839 = 全库 24.96%**（实测自 `logs/…/motion_library.json`）。

### 1.3 为什么是 129,943 而不是 npz 源目录里的 142,220 —— 已集合级对账（11:15）

源目录 `third_party/SuperMimic-Reproduced/data/npz/bones_seed/` 有 **142,220** 个 `.npz`，
而库里是 **129,943** 条，差的 **12,277** 条不是丢失，是**按清单主动筛掉的**：

| 对账项 | 结果 |
|---|---|
| 库的 names 集合 == `manifest_filtered.csv` 的 `keep=1` 集合 | **完全相同**（集合差 = 0） |
| manifest 的 142,220 个名字 == npz 目录的 142,220 个文件 | **完全相同** |
| `npz − 库` = 12,277 == manifest 的 `keep=0` 集合 | **完全相同** |
| `keep=1` 的帧数合计 vs 库的帧数 | **47,475,680 vs 47,475,680**（逐位一致） |

剔除理由（`manifest_filtered.csv` 的 `reason` 列，共 12,277 条）：

| 原因 | 条数 | 原因 | 条数 |
|---|---|---|---|
| `name:sitting` | 3,719 | `name:_box_` / `name:box_` | 126 / 96 |
| `name:50cm_box_` | 3,402 | `name:walking_on_edge` | 80 |
| `name:sit_` | 1,576 | `name:push_obstacle` | 72 |
| `name:crutch` | 930 | `name:stair` | 70 |
| `name:chair` | 686 | `name:ladder` | 48 |
| `name:table` | 646 | `ground_penetration`（物理质量） | 46 |
| `name:_sit_` | 190 | `name:_ride_` | 30 |
| `name:climb` | 180 | `off_1m` / `jump_off_50cm` / `fence` | 24 / 24 / 24 |
| `name:jump_over_obstacle_` | 136 | `floating_feet`（物理质量） | 16 |
| `elevated_support` | 132 | `on_1m` / `seat` / `fall_from` | 12 / 8 / 4 |

⇒ **语义**：被剔的是"需要椅子/桌子/箱子/台阶/梯子/拐杖/骑乘/跳越/推挤等道具或环境支撑"的动作
（合计 12,215 条）＋ **62 条物理质量不合格**（46 穿地 + 16 脚悬空）。
这些在**平地、无道具**的 WBC 仿真里无法复现（场景里没有那些实体），属于**设计性剔除**，不是数据缺失。

⇒ 因此本轮训练的"全量"定义为：**manifest `keep=1` 的全部 129,943 条**，每条跑到 clip 自然结束。
若要把道具类 12,277 条也训上，需要**重建库并改筛选**，同时接受"没有实体支撑"的错误动作进入数据——
建议保持现口径，并在论文里写明筛选规则与理由（清单文件可复现）。



## 2. 与旧 run 的差别（这次为什么要重跑）

> ⚠️ **2026-10-25 11:30 更正**：本文件初版写的"旧 500 上限 → 84.1% 帧覆盖 / 15.9% 帧从未训到"**是错的**。
> 那个算法假设**每个 episode 都从 clip 的第 0 帧开始**；实际上起点是 `managers/supermimic.py:35-50`
> 在**整条 clip 铺满的 1 秒 bin 里随机取的**（`compact.py:358-361`：`bin_start = clip_start + b*BIN_FRAMES`，
> 覆盖全 clip），所以**每一帧都能被当成起点访问到**。500 上限砍掉的不是"帧覆盖"，而是**连续轨迹的长度**。

正确口径（按"起点"统计，共 **1,010,553** 个 1 秒 bin）：

| episode 上限 | 起点落在"剩余 > 上限"的比例（= 会被上限砍掉的回合） | 连续轨迹长度上限 |
|---|---|---|
| **500（旧）** | **16.09%** | **10.0 s**（硬顶），其余 83.9% 的回合本来就短于上限 |
| 1200（文档里的保守替代） | 4.78% | 24.0 s |
| **0（本次，无上限）** | **0%** | **到 clip 结束**：随机起点下中位 **202 帧（4.0 s）**、p90 681、p99 3,303、max **8,979（180 s）** |

补充事实：长于 500 帧的 clip 有 **21,714 条（16.7%）**，但它们很长，合计 **18,428,006 帧 = 全库 38.8%**。

⇒ **"长时序没覆盖"的准确含义**：旧训练里模型**从没有连续走过超过 10 s 的同一条轨迹**；
38.8% 的帧（属于长 clip 的那些）只能以"从中间某处起步的一小段"被见到，**从没有"从头连续走到后段"的经验**。
⇒ 另一个反直觉点：**只把上限设成 0 并不会自动变成"长时程训练"**——因为起点随机铺在整条 clip 上，
无上限时随机起点的连续长度中位数只有 **202 帧（4 s）**。要真练长时程，必须同时把**起点分布**改成长 clip 早期优先
（改 `sample_frames` 的 bin 概率），这是下一步而不是本命令能做到的。

### 2.1 数据里到底有没有"长 clip"（回答质疑，逐条列出来）

**10 s 是训练配置（`max_episode_steps=500 × control_dt 0.02 s`），不是数据长度。** 数据侧实测：

| 阈值 | 条数 | 占比 |
|---|---|---|
| > 10 s（500 帧） | **21,714** | **16.71%** |
| > 20 s（1,000 帧） | 3,537 | 2.72% |
| > 30 s（1,500 帧） | 1,354 | 1.04% |
| > 60 s（3,000 帧） | 494 | 0.38% |
| > 120 s（6,000 帧） | 28 | 0.02% |
| **最长** | **8,979 帧 = 179.58 s** | —（**没有超过 180 s 的**） |

最长 10 条（全部是 `dancecards1_CD_*` 舞蹈卡，含 `_M` 镜像对）：

| # | clip | 帧 | 秒 |
|---|---|---|---|
| 1 | `dancecards1_CD_injured_L_leg_001__A008` | 8,979 | 179.58 |
| 2 | 同上 `_M` | 8,979 | 179.58 |
| 3 | `dancecards1_CD_injured_R_leg_001__A008` | 8,888 | 177.76 |
| 5 | `dancecards1_CD_normal_003__A008` | 7,322 | 146.44 |
| 7 | `dancecards1_CD_normal_003__A006` | 7,137 | 142.74 |
| 9 | `dancecards1_CD_normal_002__A008` | 6,814 | 136.28 |

⇒ **判断门槛**：长 clip 集中在舞蹈类、且只占 16.71%（>10 s）/ 1.04%（>30 s）。
如果目标任务是"10 s 内的短程跟踪"，旧 500 上限**完全够用**，本次去上限的收益主要落在舞蹈/长动作那一档；
这条决定了"要不要在长时程上继续投入"，建议按目标场景拍板。


## 3. 文档勘误（`docs/WBC_TRAIN_4GPU_BONES_SEED.md` §1）

该节写的是"clip 长度 中位 **789** 帧、均值 **907**、最长 **10000**"，与库自己不符。实测：

```
min 29  p10 149  median 288  p90 628  p99 1531  max 8979  mean 365.4
（帧数与 meta.json 一致：47,475,680 / 129,943 = 365.4）
```

不影响结论（"旧上限截掉了 16% clip / 16% 帧"成立），但引用具体分位数时请用本文档的数字。

## 3.5 三个"存活/长度"指标的定义与陷阱（**别混，混了会得出反向结论**）

| 指标 | 出处 | 真实定义 | 陷阱 |
|---|---|---|---|
| `ep_len` | WBC 训练日志 | 完成回合的长度（控制步） | **不是"活多久"**。起点是 `managers/supermimic.py:35-50` 在 clip 的某个 50 帧 bin 内**随机**取的，结束条件含"跑到 clip 结尾"（`env.py:343-347`）⇒ `ep_len ≈ 起点到 clip 结尾的剩余长度`（全库平均 365 帧 ⇒ 期望 ≈183 步）。**且起点 bin 的概率随 `bin_returns` 漂移**（`soft_probs=softmax(-25·bin_returns/…))`）⇒ **跨迭代不可横比** |
| `term_frac` | WBC 训练日志 | `sum(term)/sum(done)` = **摔倒回合 / 完成回合**（`runner.py:545`） | 这才是"摔倒率"。参考 run：10.5k = **0.49%**、12.5k 1.04%、17.5k 1.13%、**25.5k 3.56%** ←退化在这里最清楚 |
| `imag_survival` | WBC 训练日志 | 世界模型**想象**里的单步存活 | 不是真实 sim |
| `survival` | 联合微调 eval（`diff_vla_wbc/rollout.py:201-232`） | **8 步想象窗口**的累积存活：`survival *= (1 − prob)`，`prob` 由 `soft_termination`（高度/倾角规则）作用在**想象状态**上 | 8 步内几乎不触发 ⇒ **天然 ≈1**（实测 0.999963 ⇒ 隐含单步 ≈4.6e-6）。**不能当"没摔倒"的证据** |

**结论口径**：
- ✅ 判"会不会倒"用 `term_frac`（真实 sim）或 Isaac/Arena **闭环成功率**；
- ❌ 不要用 `ep_len` 判"倒得多不多"，也不要拿不同 episode 上限/不同迭代的 `ep_len` 直接横比；
- ❌ 不要用 eval 的 `survival≈1` 判"没倒"。

## 4. 怎么盯、怎么选 ckpt

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic
tail -f logs/bones_seed_token_fullclip_4x8192/train_log.csv     # 每 50 iter 一行
# 关键列：sim_r（真实仿真奖励）、ep_len、ep_ret、term
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

- **选 ckpt 判据（文档 §4）**：按 `sim_r` + `ep_len` 峰值挑，不要用 `last.pt`。
  参考 run（旧上限）的峰值在 **12.5k–17.5k iter**，32k 明显退化；本次因为不截断，
  **峰值位置可能后移**，所以 `CKPT_EVERY=500` 保留，按曲线选。
- 早期读数（供对照，别当结论）：`it 50 sim_r 0.469 / ep_len 32.9 / term 0.82`；
  `it 100 sim_r 0.462 / ep_len 37.6 / term 0.83`。

## 5. 这一轮**不能**解决的事（文档 §6，未变）

| 优先级 | 问题 | 现状 |
|---|---|---|
| **P0** | 闭环 reset 用 Arena 初始姿态（TWIST2 −0.2/0.4/−0.2），而训练是参考帧 reset + HA 默认姿态（−0.312/0.669/−0.363） | ❌ 未修 |
| **P0** | obs99 末尾 6 维（参考朝向）部署侧是常量（`SM_REF6D=zeros`），训练侧是 `mat_to_6d(Rᵀ·R_ref)` | ❌ 未修 |
| P1 | eval 侧 Arena 资产驱动的 kp/kd 与训练 `HA_ALIGN=1` 是否逐项一致 | 待核对 |

⇒ 训练做得再对，**闭环摔倒仍可能由 P0 两项造成**。别用本次训练的成功与否去反推 episode 长度的作用。

## 6. 与另一容器那个 2-rank run 的关系

`trackerlab-supermimic/logs/bones_seed_retrain_1025b/`（NPROC=2、**默认 500 步上限**）**现在仍在跑**
（`train_log.csv` 11:06:43 还在写，≈10,900 iter）。它不在本容器，**无法从这里 kill**。
两者产物目录不同、互不覆盖；**结论以本次 4 卡全 clip run 为准**，那个只作"500 步上限"的对照。
