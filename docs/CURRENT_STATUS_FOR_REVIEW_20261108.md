# COMIMIC 当前状态完整说明（给外部 reviewer 的交接文档）

> 写于 2026-11-08 15:25（北京时间）。**这份文档只讲"现在是什么状态、今天做了什么、数字是多少、卡在哪、下一步有哪几种走法"**，
> 不要求读者看过任何历史对话。所有结论都标 `PASS / FAIL / BLOCKED / NOT_RUN`，所有数字都来自真实日志。
> 如果只读三行：**① C 线（LoRA16）今天重训到 340/400 步，val ΔJ = +0.0174（CI 不含 0），是历史最好；② WM 已经离线刷新过一次并发布（wm_v000 → wm_c1_v001）；③ 还缺三样东西：周期刷新的调度器、C 自己的 on-policy 采集、梯度方向探针。**

---

## 0. 一分钟版本（TL;DR）

| 问题 | 答案 |
|---|---|
| 现在在跑什么？ | **C 线正式训练**：FT8000 冻结 + 218 层 LoRA16，400 updates，正在 it=340，预计 ~15:32 结束（8 s/步） |
| 用的哪个世界模型？ | **wm_c1_v001_published**（sha `81d24cafbd4b0953`），整个 400 步**冻结**不变 |
| 成绩如何？ | val ΔJ：50:+0.01233 / 100:+0.01494 / 150:+0.01704 / 200:+0.01735 / 250:+0.01774 / 300:+0.01759，**全部 CI 不含 0**；限位违反率 2.86%→2.77%（pristine 2.77%） |
| 和以前比？ | 旧 C1（WM0、200 步）最好 +0.0061；现在 **+0.0174 ≈ 2.8 倍**（但 WM 也换了，见 §4.3） |
| 还缺什么？ | ① 周期刷新的调度器（V7 §12）② C 自己的 on-policy 采集（现在 Buffer 里只有 A 的数据）③ 梯度方向探针 `dir_agree`（门禁现在是 override 发布） |
| 物理侧改善了吗？ | **没有**。违反率 2.77–2.86%，目标是 <2.45%（挑战 2.30%）。这是当前最大的 FAIL |

---

## 1. 涉及的三份任务书，以及现在按哪份执行

| 文档 | 内容 | 现在的作用 |
|---|---|---|
| `docs/THREE_TRACK_V6_DELIVERY_20261108.md` | 三条线 A/B/C 的原始交付与历史结果 | 背景资料（A/B 已完成、C 的历史数据） |
| **V7 任务书**（用户粘贴） | Mixed Replay Buffer + **周期** WM Refresh（每 100 步检查、按需刷新、双时间尺度）+ A/B/C 修复 | **WM refresh 部分按它执行** |
| **V8-C 任务书**（用户粘贴，最后确认"这个才是正确的"） | 只做 C1：长尾 Recovery + 物理 Loss 修复 + 困难采样 + 正式 400 步 | **除 §5（WM refresh）外全部按它执行**；§5 用 V7 的周期刷新替代 |

**合并口径（已写进 `docs/V8C_V7_MERGED_PLAN_20261108.md`）**：
V8-C 的训练/损失/评测全部保留；WM refresh 换成 V7 的"每 100 步诊断 → 触发才刷 → 门禁 → 边界换版本"。

---

## 2. 系统结构（当前代码实际跑的）

```
Frozen FT8000 (1,637,293,184 参数, fp32)
  ├─ backbone 冻结
  └─ AlternateVLDiT 冻结
       └─ 218 层 LoRA16 注入（216 个 attn1/attn2/ff + 2 个 proj_out）
          可训 = 16,777,216（主干）+ 114,688（proj）= 16,891,904（1.032%）
          原 proj_out 全秩权重：严格冻结
            ↓ 4 步 Euler Flow Matching（可微）
        token [B,40,64]
            ↓ Frozen SONIC（可微执行器）
        q_raw = q_default + G1_ACTION_SCALE ⊙ a   （standard 序，唯一入口）
            ↓ Frozen WM（版本化，见 §4）
        H=20 想象 rollout → J20
            ↓
        只反传到 LoRA 的 A/B 矩阵
```

锁定：T=40 / K=20 / H=20 / FM=4 步 / dt=0.02 / batch 32 / 400 updates / LR 3e-5（主干与 proj 各一组）/ warmup 20 / cosine → 5e-6 / grad clip 1.0。

---

## 3. 今天做了什么（按时间顺序）

| 时间 | 事情 | 结果 |
|---|---|---|
| 11:00–12:35 | **旧 C1 配方**（全秩 proj_out + 单重 trust）200 步 pilot | val ΔJ 最好 **+0.00607**（it=150），test **+0.00568**，CI 全部不含 0；但**物理违反率只有 2.67%**（旧配方 650 步时是 2.45%） |
| 12:36–13:20 | 按 review 改成 **C1 配方**（proj_out 也用 LoRA16 + 双重 trust + 固定 peak 预算 + 分组 LR + 聚类 CI） | 见上一行（就这一版）；proj_out LoRA = **114,688** 参数，与 review 手算完全一致 |
| 13:20–14:20 | 写 V8-C 的东西：**Recovery 长尾**（Huber，不改原 J20）、**Added Limit**、**弱 Margin**、**Train-only 逐关节 peak 预算**、**C 空间 λ 校准脚本** | 7 + 5 项测试全绿；校准出 λ 与 z0（见 §4.2） |
| 13:55–14:29 | **第一次 V8-C 训练**（pin WM0）跑 190 步 | 被主动停掉并**归档**为 `actor_seed0_wm0_stopped_at190`（保留作对照，不删） |
| 14:00–14:29 | **WM 离线 refresh**：70% 旧 A′ + 30% 我们采的真 Isaac A 线数据（366 段 / 7,552 tick），150 WM updates | 候选 `wm_c1_v001_candidate`（sha `81d24caf`） |
| 14:29 | 门禁 6 条 | **PASS**（0 条未通过）→ **publish** `wm_c1_v001_published` |
| 14:29–现在 | **V8-C 正式训练**（pin WM1，从官方 FT8000 全新 zero-LoRA 起） | 现在 it=340，val 曲线见 §4.1 |

### 3.1 旧 C1 与新 V8-C 的关键差别

| 项 | 旧 C1 | **V8-C（现在）** |
|---|---|---|
| proj_out | LoRA16（114,688） | 同（**保留**） |
| 边际 trust（LoRA vs proj-only） | 有（λ_lora 0.0025） | **取消**（λ=0，且跳过 proj-only 支路 ⇒ 8.0 s/步） |
| 总 trust τ | 0.020 rad | **0.030 rad** |
| peak 预算 | `1σ(q_raw_pristine)` clamp[0.01,0.10]（29/29 都顶到 0.10） | **扰动响应 Q90 + floor 0.02** ⇒ 逐关节 11 个不同值，mean 0.0225 |
| 长尾 Recovery | 无 | **有**（Huber δ=1，z0 由 Train Q90 冻结） |
| Added Limit | 无 | **有**（λ=6.78） |
| 弱 Margin | 无 | **有**（κ=0.05，λ=0.220） |
| WM | WM0 | **WM1（wm_c1_v001）** |

---

## 4. 数字

### 4.1 当前 V8-C 训练曲线（`runs/three_track_v8/c_tracking_recovery/actor_seed0`）

| it | **val ΔJ** | 95% CI（按 clip 聚类，54 簇） | 限位违反率 | Δtoken |
|---|---|---|---|---|
| 50 | +0.01233 | [+0.00951, +0.01491] | 2.86% | 0.0061 |
| 100 | +0.01494 | [+0.01143, +0.01830] | 2.84% | 0.0081 |
| 150 | +0.01704 | [+0.01392, +0.02025] | 2.79% | 0.0085 |
| 200 | +0.01735 | [+0.01357, +0.02108] | 2.78% | 0.0087 |
| 250 | +0.01774 | [+0.01413, +0.02120] | 2.77% | 0.0091 |
| 300 | +0.01759 | [+0.01447, +0.02054] | 2.79% | 0.0090 |

* `best` 目前是 it=250（+0.01774）；`checkpoint_lora_best.pt` 已存；
* 训练侧：`Lrec`（Recovery）在 3–58 波动、`Ladd=0.000`（LoRA 未造成额外越限）、`Lmar 0.02–0.05`、`Lpeak_t` 0–0.7；
* 每 50 步保存 `checkpoint_lora_XXXX.pt`（92.9 MB）+ `trainer_state.pt`（135 MB）。

### 4.2 V8-C 的标定值（全部 Train-only，48 窗口，已冻结）

| 项 | 值 |
|---|---|
| `λ_lim`（绝对限位） | **0.03068** |
| `λ_added`（额外越限） | **6.78179**（该项梯度极小 ⇒ λ 大） |
| `λ_margin`（软限位 κ=0.05） | **0.2201** |
| `λ_recovery`（长尾） | **1.913e-3** |
| `λ_peak_total` | **1.8e-5** |
| `λ_trust_total` | **0.01（fallback）** — 校准点 rms_total=12.2 mrad < τ=30 mrad，**INACTIVE**，按 V8-C §3.6 记 INACTIVE 但安全项不关掉 |
| Recovery `z0`（Q90 冻结） | root_pos **0.2059** / root_ori **0.5521** / body_linvel **0.5982** / body_angvel **0.7205** |
| peak 预算 `b_j` | mean **0.0225** rad，range [0.020, 0.0376]，**11 个不同值** |

> **重要事实**：如果按任务书原值 `z0 = 2`，实测 `frac(z > 2) = 0.000`（H=20 想象 rollout 与真实窗口都一样，root_pos p95 只有 0.32）⇒ 长尾项恒为 0，等于没改。所以 z0 用 Train Q90（≈0.21–0.72）。这是**唯一偏离任务书原文**的参数，理由与数据都记录在 `physical_weights_v8c.json`。

### 4.3 WM 刷新结果（V7 流水线）

| 判据 | 数字 | 结论 |
|---|---|---|
| 旧 A′ held-out E20 root | 4.50 cm → **4.48 cm**（rel +0.45%） | PASS（≤5% 门槛） |
| **新 on-policy E20 root** | **31.80 cm → 10.97 cm**（rel **+65.5%**，CI [+0.676, +0.762]，**4 episodes**） | PASS（≥5% 且 CI>0） |
| finite / 动作合同 / 无泄漏 | 全 True（动作域 0.13% 长尾按 0.5% 阈值放行） | PASS |
| **梯度方向 `dir_agree`** | **NOT_RUN** | 门禁记 **override**（不是 PASS） |

* 版本：`wm_v000`（`fc5aeb30e987f016…`）→ **`wm_c1_v001_published`（`81d24cafbd4b0953`）**；
* 训练数据：529,131 行旧 A′ + 366 段（7,552 tick）真 Isaac A 线数据；150 updates，11.4 s/步；
* 训练曲线：single 4.96→**1.20**，kstep 74.3→**15.8**；
* **注意**：新 on-policy 验证只有 **4 个 episode**，统计上偏薄；且这批数据是 **A 的策略**采的，不是 C 的。

---

## 5. 当前代码盘点（哪些有、哪些没有）

### 5.1 已经有、有测试（PASS）

| 能力 | 文件 | 测试数 |
|---|---|---|
| Mixed Replay（70/30、episode 防泄漏、动作域、k 步窗口、manifest） | `diff_vla_wbc/wm_replay_buffer.py` | 14 |
| WM 版本存储（原子 publish/reject/rollback、sha 断言） | `diff_vla_wbc/wm_version_store.py` | 14 |
| 门禁（6 条 + episode 聚类 CI + override） | `diff_vla_wbc/wm_refresh_gate.py` | 12 |
| 候选 WM 训练 | `scripts/train_wm_refresh_v7.py` | 真跑 150 步 |
| 门禁执行（含 `--publish/--allow-prediction-only`） | `scripts/eval_wm_refresh_v7.py` | 真跑 PASS |
| 真 Isaac 逐 tick 采集（注入式，不改 vendored） | `scripts/v7_tick_record/sitecustomize.py` + `scripts/collect_wm_refresh_isaac.py` | 3 |
| WM0 偏移诊断（E1/5/10/20、root/heading、支撑度） | `scripts/diagnose_wm_shift_v7.py` | 真跑 |
| 长尾 Recovery（Huber） | `diff_vla_wbc/tracking_recovery_tail.py` | 7 |
| Added Limit / 弱 Margin | `diff_vla_wbc/physical_bptt_losses.py` | 7 |
| C 空间 λ 校准 + 逐关节 peak 预算 | `scripts/calibrate_c_v8_losses.py` | 真跑 |
| 训练器：C1 配方 / 分组 LR / 三路同噪声 / 聚类 CI / per-group 诊断 / WM pin | `scripts/train_sonic_lora16_ft8000_bptt.py` | 7 |

### 5.2 还没有（本轮要写）

| # | 缺什么 | 后果 | 优先级 |
|---|---|---|---|
| **G1** | `scripts/schedule_wm_refresh_v7.py`（每 100 步诊断 → 触发 → 采集 → WM 训练 → 门禁 → 发布） | 没有"周期"，现在只有一次性刷新 | P0 |
| **G2** | 训练器 `--wm-refresh-mode gated` 的**真实实现**（segment 边界重新加载已发布 WM + 记录 hash；segment 内拒绝换） | 现在参数接受但行为仍是 fixed | P0 |
| **G3** | **C 自己的 on-policy 采集**（用 `checkpoint_lora_XXXX.pt` 起带 LoRA 的 server 去 Isaac 采 20 步） | Buffer 里只有 A 的数据 | P0 |
| **G4** | **梯度真值探针**（`probe_wm_gradient_truth.py` + `isaac_probe_action_rollout.py` → `dir_agree`） | 门禁第 3 条只能 override | P1 |
| **G5** | 困难起点采样 70/30 + Support Gate（V8-C §4） | 训练分布仍偏简单 | P1 |
| **G6** | Fixed-WM vs Refreshed-WM 100 步对照（V7 §13） | 无法证明"刷新帮到 Actor" | P1 |
| **G7** | 严格 resume（optimizer/scheduler/RNG/sampler/update 编号） | `trainer_state.pt` 已存但未验证能否严格恢复 | P2 |
| **G8** | 每 tick 物理落盘（q_raw/q_applied/q_real/qd_real/limit/task_stage） | 诊断与 Buffer 都缺细粒度 | P2 |

---

## 6. 现在卡在哪（真正的阻塞点）

1. **"周期刷新"没接上**：V7 要求每 100 步检查、触发才刷；现在只做了**训练前的一次性刷新**。
   要做周期，必须先有 G1 + G3（否则"用当前 C 策略采数据"这一步做不到）。
2. **C 侧采集要换 server 策略**：现在 GPU1 上的 server 是 A 的 Δ 策略；C 要换成 `--lora16-ckpt checkpoint_lora_0200.pt`。这一步代码支持（`ha_vla_latent64_server.py --lora16-ckpt` 已实现），但**采集编排还没写**。
3. **物理改善不达标**：违反率 2.77%（≈pristine），目标 <2.45%。已加的 Added/Margin/固定 peak 都**没有**把违反率压下去，需要看 per-joint 报告才能判断是"限位约束太软"还是"LoRA 根本没往安全方向走"。
4. **统计偏薄**：WM 门禁的新数据只有 4 个 episode；训练 val 只有 54 个 clip 簇。
5. **一致性提示**：`checkpoint_lora_best.pt` 是 **WM1 口径**；归档的 `actor_seed0_wm0_stopped_at190` 是 **WM0 口径** —— 两者**不能**拼成一条曲线比较。

---

## 7. 接下来可以怎么走（三选一，待定）

### 方案 A：跑完 400 步，再做对照（最省算力）
1. 等 it=400（~15:32）→ 独立 test 一次；
2. 写 G1/G3/G4，做一个完整的 **cycle001**（用 C 的 400 步快照采数据 → 训 WM2 → 门禁）；
3. 用 `checkpoint_lora_0300 或 0400` 做 **Fixed-WM vs Refreshed-WM 各 100 步**对照（V7 §13）→ 这是"刷新是否有效"的唯一硬证据。

### 方案 B：在 it=340 边界停，立刻做 cycle001，然后 resume 到 400
* 优点：更贴合"周期"；缺点：要中断当前 run，且 cycle 一次要 110–145 分钟（采集 75–110 + WM 28 + 门禁 4）。

### 方案 C：先只补"便宜诊断"，不做采集
* 每 100 步算 E20/token/物理（2–3 分钟），记录但不刷新；等 G3 写好后再真正触发。

> 我的建议：**A**（先拿到 V8-C 的完整 400 步结果 + 独立 test），然后用 C 的最终快照做一次完整 cycle + Fixed-vs-Refreshed 对照。
> 这样"V8-C 训练"和"V7 周期刷新"两件事都有干净结论，不会互相污染。

---

## 8. 关键路径速查

```
训练（当前）      runs/three_track_v8/c_tracking_recovery/actor_seed0/
                  ├ train.log / train.jsonl / val.jsonl / cond_check.jsonl
                  ├ checkpoint_lora_{0050..0300}.pt / checkpoint_lora_best.pt
                  ├ trainer_state.pt
                  └ config.json（含 218 层 / 16,891,904 / 参数组）
标定              runs/three_track_v8/c_tracking_recovery/calibration/
                  ├ physical_weights_v8c.json  ├ peak_budget_train_v8.json  └ recovery_z0.json
WM 刷新           runs/three_track_v8/c_tracking_recovery/wm_refresh/
                  ├ refresh.log  └ gate*/（gate.json / gate.md）
WM 版本           runs/three_track_v7/wm_versions/
                  ├ wm_v000/（fc5aeb30…）  ├ wm_c1_v001_candidate/（81d24caf…）
                  └ wm_c1_v001_published/  ← active
A 线真 Isaac 数据  runs/three_track_v7/buffer/cycle000_A/（366 段 / 7,552 tick / 12 episodes）
旧 WM0 训练 run    runs/three_track_v8/c_tracking_recovery/actor_seed0_wm0_stopped_at190/（归档，勿混）
计划文档           docs/V8C_V7_MERGED_PLAN_20261108.md
旧 C1 结果         runs/three_track_v7/c1_lora_proj/pilot200/
```

**复核命令（复制即用）**：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
source scripts/harness03_env.sh
# 当前曲线
cat runs/three_track_v8/c_tracking_recovery/actor_seed0/val.jsonl
# WM 版本
cat runs/three_track_v7/wm_versions/active.json
# 门禁
cat runs/three_track_v8/c_tracking_recovery/wm_refresh/gate*/gate.json
# 标定
cat runs/three_track_v8/c_tracking_recovery/calibration/physical_weights_v8c.json
# 测试（全部离线、不需要 Isaac）
python3 -m pytest tests/test_wm_replay_buffer.py tests/test_wm_version_store.py \
  tests/test_wm_refresh_gate.py tests/test_tracking_reward_tail.py \
  tests/test_tracking_recovery_tail.py tests/test_physical_bptt_losses.py \
  tests/test_lora16_dummy.py -q
```

---

## 9. 需要 reviewer 回答的 5 个问题

1. **物理违反率压不下去**（2.77% ≈ pristine，目标 <2.45%）：是应该继续加 `λ_added/λ_margin`（现在已经 6.78 / 0.22，比其它项大 2 个数量级），还是说明"限位约束"与"ΔJ 提升"在这个目标下真的冲突？有没有更直接的指标/损失（例如 per-joint 违反率加权）？
2. **`z0` 偏离任务书**（任务书写 2.0，实测数据支持 0.21–0.72）：这个偏离是否可接受？还是应该改成"对 z 做归一化后再用 2.0"？
3. **WM 门禁的新数据只有 4 个 episode**：是否应该先扩采到 12+ 再发布？当前是 `override` 发布（梯度探针未跑），风险如何评估？
4. **Fixed-vs-Refreshed 对照的最小可信规模**：100 步 / 1 个 seed 够不够？要不要 2 seed？
5. **周期刷新的成本**：100 步训练 13 分钟，一次刷新 110–145 分钟 ⇒ 是否应该改成"每 200 步检查"或"只在 E20 恶化 >10% 时刷"？

---

## 10. 边界（务必遵守）

* `runs/three_track_v6/`、`runs/a_prime_sonic_exc/` **只读**，今天的产物全在 `runs/three_track_v7/` 与 `runs/three_track_v8/`；
* 本轮 **没有** 做完整任务闭环成功率（C 的 18 对闭环是下一阶段）；
* `ΔJ` 是**冻结 WM 的想象回报**，不是真机成功率；
* **新旧 Reward 不可混比**：`J`（原 J20）与 `J_pristine_base` 分开记录；不同 WM 版本的 ΔJ 也不能拼成一条曲线；
* 任何"LoRA 有效"的说法必须区分：pristine 对比 / LoRA 边际（同 proj）/ token 位移量。
