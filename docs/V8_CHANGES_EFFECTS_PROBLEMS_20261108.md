# V8 改动 · 效果 · 问题（A/B）—— 交给 reviewer 出下一步方案

> 2026-11-08 08:00 UTC　｜　`comimic-wt03`　｜　基线提交 `941f7a9`（物理修复）+ `8abaadb`/`bf42248`（C 线）
> 本文只讲**这一轮（V8）到底改了什么、效果如何、哪里坏了**。全量架构见
> `docs/COMIMIC_FULL_HANDOFF_FOR_GPT_20261108.md`。

---

## 0. 一句话

**除 WM Refresh 外，reviewer 给的 V8 条目全部实现了**（Recovery、困难采样、λ 重标、Added、从 best 继续训）。
**A/B 各训了 200 步，WM 指标提升 11–18%（独立 test 集）**。
但有三件事必须让 reviewer 知道：

1. **提升不是 Recovery 带来的**（恢复项的尾部两个都没压住，B 甚至全面变差）；
2. **最后一组实验（起点漂移 0.25）因为磁盘写满而崩溃**，没拿到结果；
3. **闭环一次都没跑过新 ckpt** —— 所以"真实效果"仍然未知。

---

## 1. V8 清单逐项核对（改了什么 / 没改什么）

| # | reviewer 的 V8 条目 | 状态 | 证据 / 说明 |
|---|---|---|---|
| 1 | **Tracking Recovery**（大误差处补梯度，与 C 共享实现） | ✅ **已实现**并接入 A/B | `diff_vla_wbc/tracking_recovery_tail.py`（`u=relu(z−z0)`、Huber δ=1、additive，不改 J20）；A/B 与 C 共用同一份代码，只各自标定 λ |
| 2 | **困难状态采样**（让 Root/Heading 偏差大的窗口更常出现） | ✅ **已实现**（70% 普通 + 30% 困难） | `[v5][hard]` 日志：阈值 0.3197、n_hard 70,584/235,280；困难度 = `‖state[:3]−ref[:3]‖ + 0.5‖rot−ref‖` |
| 3 | **离线 WM Refresh**（旧 Replay 策略邻域重加权 → 候选 WM → 验证） | ❌ **没做，且上一版做法是错的** | 只重加权**旧数据**没有新真值 ⇒ 违反 V7 §3.1；脚本已加弃用声明。正确路线需 B 的 on-policy 真实数据（**目前不存在**） |
| 4a | **重标定物理 Limit**（校正单位后） | ✅ **已做**（96 窗口） | `λ_lim`：A 0.0528 / B 0.0570（旧的 0.0529 几乎一致；8 窗口的 0.1453 是离群） |
| 4b | **补充 Delta 新增越限约束** | ⚠️ **接了但语义不符、数值≈0** | `l_added_limit` 罚的是"恶化"（2.8% 样本），真正"新增"只有 **0.45%**；`‖g_added‖≈5e-4` ⇒ 反解 λ=210–323 会爆，现固定 0.053 |
| 5 | **保留网络、从 A/B best 继续训，不再从零重训** | ✅ **已做** | `A_v8`（200 步，从 A best@225）、`B_v8b`（200 步，从 B best@900） |
| 6 | **Trust 保持真 rad τ=0.02，去掉重复 L_same** | ✅ **已做** | `--trust-domain physical` + `λ_same` 强制 0 |
| 7 | **Fixed Peak 只当护栏，不收紧** | ✅ **已做** | 固定预算 mean 0.2216 rad；实测 `e_phys_absmax` 0.086–0.21 rad ⇒ **不激活** |
| 8 | **不升 Cap、不抄 C 的 τ=0.03** | ✅ **已做** | cap 全程 0.08 |
| 9 | **闭环验证**（首次尝试成功率） | ❌ **NOT_RUN** | 新 ckpt 一次都没上过 Isaac |
| 10 | **多 seed** | ❌ **NOT_RUN** | A/B 各 1 个 seed |
| — | **Fixed-WM vs Refreshed-WM 对照（V7 §13）** | ❌ **NOT_RUN** | 依赖 3 |

**两处对 reviewer 方案的主动偏差（需要你确认）**：

| 项 | reviewer 方案 | 实际 | 原因 |
|---|---|---|---|
| `z0` | 先用 `z0=2` 检查激活率 | **Train Q0.90 冻结**（0.645/2.510/1.456/3.142） | 实测 `z0=2` 时 `frac(z>2)≈0`（C 线 448 窗口实测为 0；我们 batch=256 的 `z_p95` 只有 0.8–1.0）⇒ `z0=2` 等于没加 |
| `λ_added` | 反解 `0.5·‖g_J‖/‖g_added‖` | **固定 0.053（=λ_lim 尺度）** | `‖g_added‖≈5e-4` 时反解病态（λ→210+），一旦出现越限会爆 |

---

## 2. 训练结果与效果

### 2.1 全部 pin 在 **WM0 = `wm_gain2.pt`（sha `fc5aeb30`）**

已逐 run 从 `config.json.wm_pin` 回读核验：**A/B 的全部结果、所有冒烟、C 的 pilot200 都是 WM0**。
（`wm_v000/wm.pt` 与 `a_prime_sonic_exc/wm_gain2.pt` **哈希完全相同**。）

### 2.2 结果（同一 WM0、同一 96 窗口；A/B 可直接横比）

| 模型 | 配置 | 步数 | val ΔJ（best@it） | **test ΔJ** | 相对原版 |
|---|---|---|---|---|---|
| A 原 | 5e-5/2e-5，drift 0.1，无 Recovery | 300 | +0.02962 @225 | **+0.02185** | — |
| **A_v8** | 3e-5/1e-5，+Recovery +困难采样 | +200 | **+0.03291 @100** | **+0.02477（best）/ +0.02577（final）** | **val +11% / test +13~18%** |
| B 原 | 3e-4/1e-4，无 Recovery | 960 | +0.02774 @900 | **+0.02074** | — |
| **B_v8b** | 3e-5/1e-5，+Recovery +困难采样 | +200 | **+0.03110 @25** | **+0.02416** | **val +12% / test +16.5%** |

**结论：A/B 在冻结 WM 上都拿到了 11–18% 的提升，且是独立 test 集。这部分是可信的。**

### 2.3 但"提升来源"与预期不符（**这是最重要的一条**）

| 指标 | A_v8（首5→末5） | B_v8b（首5→末5） |
|---|---|---|
| `L_recovery` | 20.9 → 14.9（−29%） | 16.0 → **16.3（+1.6%）** |
| `rec_root_pos_z_p95` | 0.880 → 0.884（+0.4%） | 0.820 → **0.885（+7.9%）** |
| `rec_root_ori_z_p95` | 1.69 → **1.92（+13.7%）** | 1.75 → **2.14（+22.3%）** |
| `rec_body_linvel_z_p95` | 1.57 → 1.54（−2%） | — |
| `rec_body_angvel_z_p95` | 2.81 → 2.68（−4.7%） | — |
| `e_phys_absmax`（rad） | 0.136 → **0.214（+57%）** | 0.164 → 0.199（+21%） |
| best 出现在 | it=100（共 200） | **it=25（共 200）** |

⇒ **两个 run 的"恢复尾巴"都没有改善（B 甚至全面变差）**，而 val/test 却提升了。
加上 **best 都出现在很早期**，说明：

> 提升更可能来自 **`L_lim` 的约束 + 困难采样改变了训练分布 + 低 lr 的局部精修**，
> **而不是** Recovery 真的教会了"从大误差恢复"。

这也解释了为什么"起点漂移 0.25"那组实验很关键 —— 它是唯一能让尾部样本真正出现在训练里的改动。

---

## 3. 问题清单（按严重度）

### P0 —— 阻塞 / 结论不成立

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| **P0-1** | **磁盘 100% 满**（10 TB 仅剩 **1.7 GB**） | `df -h`：`gpfs_hdd 10T 10T 1.7G 100%` | ①`A_d25_q90` 在 it≈70 时 **torch.save 写盘失败**（`PytorchStreamWriter failed`）⇒ run 崩溃、ckpt 损坏（已删）；②`A_d25_z2` 启动即 abort（`object refcount : 3`）；③**闭环会录视频，几乎肯定写不下** |
| **P0-2** | **Recovery 没压住尾部**（§2.3） | `train.jsonl` 的 `rec_*` | "加了尾部损失"≠"优化了尾部"；当前实现是 `Q0.90` 阈值 + Huber 梯度封顶(1) + batch 均值 ⇒ 只压"刚过线"的那批 |
| **P0-3** | **提升归因不清**：best 出现极早（A@100、B@25），恢复尾部变差 | §2.3 | 不能宣称"V8 配方教会了恢复能力" |
| **P0-4** | **闭环 NOT_RUN**；已有 9/18 是"重试取最好"，不是首次成功率 | `isaac/A_delta_continue/RESULTS.md` | 真实效果仍未知 |
| **P0-5** | **WM Refresh 完全没做**（V8 第 3 条） | §1 第 3 行 | B 的 on-policy 数据不存在；V7 §13 的 Fixed-vs-Refreshed 对照 NOT_RUN |
| **P0-6** | 单 seed（A/B 各 1） | — | 无误差棒 |

### P1 —— 影响结论强度

| # | 问题 | 说明 |
|---|---|---|
| **P1-1** | **物理放宽**：`e_phys_absmax` +57%(A) / +21%(B) | 仍远小于限位，但方向不利；`L_act/L_peak` 都不激活，没有任何东西约束它 |
| **P1-2** | **`L_added` 定义与名字不符** | 罚"恶化"(2.8%)，而"真正新增"只有 0.45%；数值≈0 ⇒ 不能当"新增越限已约束"的证据 |
| **P1-3** | **A/B 仍非单变量对照** | A 是 warm-start + 5e-5，B 是随机初始化 + 3e-4（且 B 的 lr 恰等于 v5 原配方）⇒ 无法归因到"初始化" |
| **P1-4** | **条件化只被验证到"不是常量"**（`cosZ≈0.65`，门禁阈值 <0.9） | 没有 reward 对齐证据 |
| **P1-5** | **日志单位标签历史性错误** | `train.jsonl` 的 `*_mrad` 实为归一化动作域（×0.396 才是 rad） |

### P2 —— 工程

| # | 问题 |
|---|---|
| **P2-1** | **WM1（`wm_c1_v001_published`）是"梯度探针未跑"就发布的**（`gate.json` 里 `override.reason=gradient_probe_not_run`）。本轮有一次 run 误 pin 到它（`B_v8`，已作废重跑）；此后所有 run 都加了 `--strict-wm-hash` + 回读校验 |
| **P2-2** | 闭环无逐 tick 落盘 |
| **P2-3** | 共享 worktree 有并发 agent，已发生一次"WM 被切换"事故 |

---

## 4. 未完成 / 阻塞清单

| 事项 | 状态 | 阻塞原因 |
|---|---|---|
| 起点漂移 0.25 的两组对照（`A_d25_q90` / `A_d25_z2`） | ❌ 崩溃 | **磁盘满**（P0-1） |
| 闭环：A_v8 / B_v8b / A原 / B原，首次尝试成功率 | NOT_RUN | 磁盘（录像）+ 尚未排期 |
| 多 seed（A/B 各再 2 个） | NOT_RUN | — |
| B 的 on-policy 采集（WM Refresh 前置） | NOT_RUN | 需 Isaac，25–60 min |
| 梯度探针（`ΔJ_WM` vs `ΔJ_Isaac`） | NOT_RUN | 从未做过 |
| V7 §13 Fixed-vs-Refreshed 对照 | NOT_RUN | 依赖 WM Refresh |
| `L_added` 语义三选一（改名/只罚新增/去掉） | 未定 | 待 reviewer |

---

## 5. 请 reviewer 回答的问题

**Q1 · 磁盘（P0-1）** 10 TB 用尽，`comimic/runs` 占 1.4 TB，其中可回收的大头：
`ablations/ 222G`、`S3_loop/ 126G`、`j1_scene2/ 73G`、`joint_ha_a1/ 61G`、若干 37G 的 s3v4/s3v6。
**哪些可以删？** 以及：闭环录视频每小时约占多少、要不要改成只录失败集？

**Q2 · Recovery 怎么才能真压尾部（P0-2）**
在"`z` 的分布尾部只有 1–2% 样本"的前提下，你要哪一种：
(a) `z0` 提到 Q0.97/Q0.99；(b) `δ` 调到 0.3–0.5；(c) 凸尾 `u²`；
(d) **尾部加权** `w ∝ relu(z−z0)`；(e) CVaR/分位数目标。请给公式 + 防爆措施。

**Q3 · 起点漂移（P0-2 的另一半）** `start_max_drift` 取 0.25 还是更大？
是否需要 root 横向/航向的**定向**扰动？如何判定"仍在 WM 支持范围内"？

**Q4 · 提升归因（P0-3）** 既然 best 出现在极早期、恢复尾部没改善，
**你是否同意"这 11–18% 主要来自 L_lim + 采样分布 + 低 lr 精修"**？
如果要拆开，最小实验集是什么（Recovery on/off × 困难采样 on/off 的 2×2？每组多少步？）

**Q5 · 闭环（P0-4）** 首次尝试成功率的判定与样本量（3 任务 × 多少 seed？），
以及必须落盘哪些逐 tick 字段才能在失败时归因。

**Q6 · 是否还值得继续 A/B** 还是把算力转到 C / WM Refresh？（A_v8 的 test +18% 是否足够"结项"）

---

## 6. 复现命令

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
source scripts/harness03_env.sh

# A_v8（已完成）
python3 scripts/train_delta_physical_bptt_v6.py --out runs/three_track_v7/ab_v8/A_v8 \
  --on-the-fly 1 --eval-set runs/temporal_ac_bptt_v5/eval_set --updates 200 --batch 256 \
  --init-delta-checkpoint runs/three_track_v6/delta_continue/seed0/checkpoint_actor_best.pt \
  --physical-weights runs/three_track_v6/calibration/physical_weights_delta.json \
  --limits-npz runs/three_track_v6/physical/limits_standard29.npz \
  --lambda-lim 0.0528 --lambda-added 0.053 --lambda-recovery 0.0019 \
  --recovery-z0-json runs/three_track_v7/ab_v8_smoke/recovery_z0.json \
  --start-max-drift 0.10 --hard-frac 0.3 \
  --trust-domain physical --peak-budget-mode fixed \
  --peak-budget runs/three_track_v7/ab_fix_smoke3/peak_budget.json \
  --head-lr 3e-5 --enc-lr 1e-5 --warmup-updates 20 \
  --eval-every 25 --cond-check-every 25 --seed 0 --device cuda:0 \
  --wm-version wm_v000 --strict-wm-hash fc5aeb30

# 起点漂移 0.25（崩溃，需先解决磁盘）
#   scripts/queue_ab_v8_wm0.sh 的 train A_d25_q90 / A_d25_z2 两行
```

**产物位置**

| 内容 | 路径 |
|---|---|
| A_v8 / B_v8b 的 ckpt、日志、val | `runs/three_track_v7/ab_v8/{A_v8,B_v8b}/` |
| A_v8 best 的 test（WM0） | `runs/three_track_v7/ab_v8/A_v8_best_test_wm0/test_once.json` |
| λ 标定（96 窗口） | `runs/three_track_v7/ab_diagnostics/calib_{A,B}_v8.json` |
| 真物理审计 | `runs/three_track_v7/ab_diagnostics/{v5,A,B}_phys_rad_v3.json` |
| 崩溃现场 | `runs/three_track_v7/ab_v8/{A_d25_q90,A_d25_z2}/train.log` |
