# COMIMIC 全会话交接文档（2026-11-08）

> 面向"接手的人/模型"，**不需要看过任何历史对话**。
> 内容：本次会话从头到尾做了什么、得到什么数字、改了哪些代码、踩了哪些坑、现在卡在哪、下一步怎么走。
> 所有结论标 `PASS / FAIL / BLOCKED / NOT_RUN`；所有数字来自真实日志（路径写在 §11）。

---

## 0. 十分钟速读（如果只看一节）

| 问题 | 答案 |
|---|---|
| 这条会话主要在干什么？ | 三条路线里的 **C 线（FT8000 + LoRA16）**：按外部 review 修配方 → 按 V8-C 任务书加"长尾恢复 + 物理修复 + 困难采样" → 训练 → 独立评测；同时做了 **V7 的一次 WM 离线刷新**（发布 WM1） |
| C 线最终成绩（WM0 固定，280 步） | val ΔJ **+0.01748**（Best@250）、**test ΔJ +0.01640，CI [+0.01217, +0.02035]**、越限 **2.53%**（pristine 2.59%） |
| 相对官方 `checkpoint-8000`（基座） | **+1.04%（CI 0.77%–1.29%）**；旧 C1 只有 **+0.36%** ⇒ **2.9 倍** |
| 结构对不对 | **主干 LoRA 占 80%、proj_out 占 20%**（旧 C1 是 proj 占 87%）⇒ review 要的反转已实现 |
| 物理达标吗 | **未达**（目标 <2.45%）。原因已定位：**原始 FT8000 自己就在 j4/j10 上越限 25.78%/22.55%**，这是"参考动作 vs 限位定义"的结构问题 |
| WM refresh 有用吗 | **对新策略分布有用**（20 步 root 误差 31.8→11.0 cm，−65.5%），但**对任意真实转移无效**（4.59→4.56 cm），对 Actor 收益只值 **+9.5%** |
| 闭环做了吗 | **没有跑成**（BLOCKED）：环境变量把 Isaac 的 onnxruntime 遮住，4 次尝试全部 `startup failed`。修法在 §9 |
| 最大的未验证项 | **真实 SONIC–Isaac 任务成功率**（18 对）；没有它，+1% J20 不能等同于真机收益 |

---

## 1. 会话目标与三份任务书

| 文档 | 作用 |
|---|---|
| `docs/THREE_TRACK_V6_DELIVERY_20261108.md` | 三条线（A Delta-Continue / B Delta-Retrain / C LoRA16-FT8000）的原始交付与历史结果 |
| `docs/TRACK_ISSUES_PARAMS_RESULTS_20261108.md` | 三条线的参数、结果、问题总表（本会话前半段产出） |
| **V7 任务书**（用户粘贴） | Mixed Replay Buffer + **周期** WM Refresh + A/B/C 稳定性修复 |
| **V8-C 任务书**（用户粘贴，最终以此为准） | 只做 C1：长尾 Recovery + 物理 Loss 修复 + 困难采样 + 正式 400 步；**§5 的"先刷一次再冻结"被用户改为用 V7 的周期刷新** |
| **V8.1-A/B 任务书**（用户粘贴，属 A/B 侧） | 尾部恢复 + Peak 预算 + 重训 200 步；**明确不涉及 C**。注意：A/B 在**同一个 worktree 上并行改代码**（见 §8 坑 9） |

**合并口径**：V8-C 全部（除 §5）+ V7 的周期刷新。计划文档：`docs/V8C_V7_MERGED_PLAN_20261108.md`。

---

## 2. C 线的最终配置（可复现）

```
Frozen FT8000: runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000（1,637,293,184 参数, fp32）
  + 218 层 LoRA16：216 个 attn1/attn2/ff + 2 个 proj_out（**原全秩权重冻结**）
    可训 16,891,904 = 主干 16,777,216 + proj_out 114,688（1.032%）
  + 4 步 Euler Flow Matching（可微）
  → token [B,40,64] → Frozen SONIC → q_raw = q_default + G1_ACTION_SCALE ⊙ a
  → **Frozen WM0 = runs/a_prime_sonic_exc/wm_gain2.pt（sha256 前16位 fc5aeb30e987f016）**
  → H=20 想象 rollout → J20（原 saturating 指数核，**未改**）
```

**损失（V8-C 七项 + P2/P4）**

```
L = −J20(原)
    + λ_R      · L_recovery(Worst-Window 聚合)      λ_R = 2.223e-3
    + λ_lim    · L_abs(q_raw)                        λ_lim = 0.03068
    + λ_added  · L_added(v_full − v_pristine)        λ_added = 6.78179
    + λ_margin · L_margin(κ=0.05)                    λ_margin = 0.2201
    + λ_peak   · L_peak(e_total, b_j)                λ_peak = 1.8e-5
    + λ_trust  · L_same(e_total, τ=0.03)             λ_trust = 0.01（校准点 INACTIVE，回退旧值）
```

| 参数 | 值 |
|---|---|
| Recovery `z0`（**Train-only Q70**） | root_pos **0.0992** / root_ori **0.2178** / body_linvel **0.2428** / body_angvel **0.3046** |
| Recovery 聚合 | **worst_window**：窗口内最差 25% tick + batch 最差 10% 窗口 |
| Peak 预算 `b_j` | **mean 0.0486、range [0.020, 0.120]、21 个不同值**（j4=0.12） |
| 训练采样 | 70% 普通 + **30% 困难**（root/heading/vel 误差 top-20%）+ **支持域门禁**（阈值 15.31，通过 **99.2%**） |
| LR/batch/steps | 3e-5（两组）→ cosine 5e-6；batch 32；warmup 20；grad clip 1.0 |
| 速度 | **7.7–8.0 s/步**（单卡 4090） |

---

## 3. 主要结果（按时间顺序）

### 3.1 旧 C1 配方（proj_out 全秩 + 单重 trust），200 步

| 指标 | 值 |
|---|---|
| val ΔJ（8 个点） | 25:+0.00504 / 50:+0.00385 / 75:+0.00460 / 100:+0.00520 / 125:+0.00524 / **150:+0.00607** / 175:+0.00575 / 200:+0.00565（**CI 全部不含 0**） |
| test | **+0.00568**，CI [+0.00412,+0.00712] |
| 越限 | 2.67–2.72% |
| 归因 | **proj_out 占 87%**（3.05e-2 vs LoRA 3.2e-3） |

### 3.2 WM 离线刷新（V7 流水线）→ 发布 WM1

| 项 | 值 |
|---|---|
| 数据 | 529,131 行旧 A′ + **366 段 / 7,552 tick 真 Isaac A 线数据**（12 episodes） |
| 训练 | 150 updates，70/30，11.4 s/步；single 4.96→1.20，kstep 74.3→15.8 |
| 门禁 | 旧 A′ val root 4.50→4.48 cm（+0.45%）；**新 on-policy val root 31.80→10.97 cm（−65.5%，CI [+0.676,+0.762]，仅 4 episodes）**；`dir_agree` **未跑 ⇒ override** |
| 版本 | `wm_v000`(fc5aeb30) → **`wm_c1_v001_published`(81d24caf)** |

### 3.3 V8-C 训练（三条 run，口径不同不可混比）

| Run | WM | Steps | val ΔJ | test ΔJ | test 越限 | 备注 |
|---|---|---|---|---|---|---|
| `actor_seed0_wm0_stopped_at190` | WM0 | 190 | +0.01631 | — | 2.57% | 中途停（改配方） |
| `actor_seed0`（WM1） | **WM1** | 400 | **+0.01838** | **+0.01796** | 2.60% | 门禁 override 发布 |
| `actor_wm0_seed0_stopped_at100` | WM0 | 100 | +0.00999@50 | — | 2.65% | 加困难采样前 |
| **`actor_wm0_p2p4`（最终）** | **WM0** | **280** | **+0.01748** | **+0.01640** | **2.53%** | P2+P4，停在平台 |

**最终 run 的独立 test（Best=it=250）**

| 指标 | 值 |
|---|---|
| J / J_base | 1.59213 / 1.57573 |
| **ΔJ** | **+0.01640**，CI **[+0.01217, +0.02035]**（55 簇） |
| **相对基座** | **+1.041%**（CI 0.772%–1.291%） |
| 越限 | **2.53%**（pristine 同 split 2.59%） |
| `q_raw_absmax` | 3.843（pristine 3.580） |
| 四路归因 | pristine 1.5706 / **仅主干 1.5862（token rms 0.0083, 80%）** / 仅 proj 1.5732（0.0027, 20%）/ 全开 1.5879 |

**逐关节 vs 原始 FT8000（决定性证据）**

| 关节 | pristine | 训练后 | 相对改善 |
|---|---|---|---|
| **j4** | **25.78%** | 24.95% | −3.2% |
| **j10** | **22.55%** | 21.88% | −3.0% |
| j5 | 13.33% | 11.82% | −11.3% |
| j11 | 9.64% | 8.44% | −12.4% |
| 全局（tick×关节） | 2.59% | 2.53% | −2.3% |

⇒ 越限主要由"参考动作 vs 限位定义"决定，**训练可改善但到不了 0**。

### 3.4 为什么停在 280（平台化）

| it | 100 | 150 | 200 | 250 | 趋势 |
|---|---|---|---|---|---|
| ΔJ | +0.01727 | +0.01470 | +0.01669 | +0.01748 | 无（+0.0165±0.0014） |
| 越限 | 2.56% | 2.66% | 2.61% | 2.62% | 无 |
| j4 | 22.6% | 23.1% | 22.8% | 22.8% | 无 |

### 3.5 WM 保真度（重要修正）

| 口径 | 数值 | 说明 |
|---|---|---|
| **原始 probe 协议**（`probe_wm_multistep.py`，策划好的 20 步窗口、喂录制动作） | 20 步 root 漂移 **想象 6.93 vs 真实 6.81 cm，ratio 1.02 PASS**；**端点 p50 2.11 cm / p90 5.93**；corr 0.929 | 今天实测当前 `wm_gain2.pt` |
| 旧文档记的 **0.89 cm** | **今天复现不出来** | 结合该文件 mtime=**2026-11-05 15:54**（重导出）与哈希争议（`c74060f1` vs `fc5aeb30`）⇒ 0.89 属于**重导出前**的旧版本 |
| V7 诊断（任意真实转移、逐 tick 均值） | **4.59 cm，p95 18.5 cm** | 更难的分布（含接触/转身） |
| 同诊断换 **WM1** | **4.56 cm**（−0.7%） | **刷新解决不了这一项** |

**结论**：WM 在稳定窗口上没问题（ratio 1.02），在部署会遇到的高动态分布上是 2–5 cm；这个量级与我们追求的改进同阶 ⇒ J20 优化到 +1% 后继续加码，很可能在拟合 WM 误差。

---

## 4. 代码改动清单（本会话）

### 4.1 新增文件

| 文件 | 作用 | 测试 |
|---|---|---|
| `diff_vla_wbc/wm_replay_buffer.py` | Mixed Replay（70/30、episode 防泄漏、动作域、k 步窗口、manifest） | 14 |
| `diff_vla_wbc/wm_version_store.py` | WM 版本存储（原子 publish/reject/rollback、sha 断言、pin） | 14 |
| `diff_vla_wbc/wm_refresh_gate.py` | 门禁 6 条 + episode 聚类 bootstrap CI + override | 12 |
| `diff_vla_wbc/vla_grad_sampler.py` | 现场可微 VLA 采样（C 的唯一阻塞点修复） | 含在 lora16 测试 |
| `diff_vla_wbc/tracking_reward_tail.py` | `exp(-z)` 的有理长尾（V8-Lite-R 遗留，未用于 C） | 12 |
| `diff_vla_wbc/tracking_recovery_tail.py` | **Recovery：Huber `relu(z−z0)`，支持 mean / worst_window** | 8 |
| `diff_vla_wbc/support_aware_sampler.py` | **困难起点难度打分 + SupportIndex + 70/30 混合** | 5 |
| `scripts/train_wm_refresh_v7.py` | 候选 WM 训练（single/kstep/both） | 真跑 |
| `scripts/eval_wm_refresh_v7.py` | 门禁执行（`--publish/--allow-prediction-only`） | 真跑 |
| `scripts/collect_wm_refresh_isaac.py` | **真 Isaac 逐 tick 采集 + tick→segment** | 3 |
| `scripts/v7_tick_record/sitecustomize.py` | **注入式 recorder**（不改 vendored 栈） | 真跑 |
| `scripts/make_segments_from_replay_v7.py` | 真物理分片 → segment | 真跑 |
| `scripts/diagnose_wm_shift_v7.py` | WM E1/5/10/20 诊断 | 真跑 |
| `scripts/calibrate_c_v8_losses.py` | **C 空间 λ + peak 预算 + z0 标定** | 真跑 |
| `scripts/eval_c_v8c_test.py` | 任意 C adapter 的独立 test（含四路归因） | 真跑 |
| `scripts/probe_wm_multistep.py`（已有） | WM 20 步保真（ratio/endpoint） | 真跑 |

### 4.2 修改文件

| 文件 | 改动 |
|---|---|
| `diff_vla_wbc/lora.py` | `set_lora_delta_scale(..., include=)` 支持按组切换 |
| `diff_vla_wbc/lora_ft8000_loader.py` | `C_RECIPES`（c0/c1/c2/c3）、按组参数统计、proj 冻结断言 |
| `diff_vla_wbc/physical_bptt_losses.py` | `l_same` 数值修复、`l_added_limit`、`l_margin` |
| `diff_vla_wbc/isaac_joint.py` | `reward_tail_mode=rational` 接入（默认 off，逐位不变） |
| `diff_vla_wbc/reward_cfg.py` | `REWARD_TAIL_*` 环境变量 |
| `scripts/train_sonic_lora16_ft8000_bptt.py` | **主要改动**：C1 配方、分组 LR、三路同噪声、总/边际 trust、Recovery/Added/Margin、固定 peak（perturb_q90 / self_q99）、困难采样 + 支持域、四路归因、per-joint 报告、严格 resume、聚类 CI |
| `tests/*` | 上述模块的单测（合计 **60+ 项**通过） |

### 4.3 关键 commit

`b9052fb` C代码+测试 · `0086b53` Phase0/1 · `20d73a2` Phase2a · `97b0646` Phase3 · `e49ab85` WM pinning · `1b106a8` 真物理分片转换 · `df744ba` C1 八点结果 · `7692bc1` 长尾奖励 · `8abaadb` V8-C 长尾+校准 · `bf42248` WM1 发布 · `2738e3e` 现状说明书 · `51920f4` 合并计划 · `05256c7` 补齐 1–5 · `9051332` P2/P4 · `c774bdc` 最终 test · `2d1f6af` C 线最终报告

---

## 5. 本会话踩过的坑（全部有证据，接手必读）

| # | 坑 | 现象 | 修法 |
|---|---|---|---|
| 1 | `l_same` 在 `e≡0` 时 NaN | zero-LoRA 首步 LoRA 梯度 NaN | 平方和 +1e-12 再开方 |
| 2 | `also_train` 是 no-op | "LoRA+proj_out"其实只训了 LoRA | 按全路径子串解冻，加测试 |
| 3 | server 的 LoRA scale 与训练不一致 | 训练 alpha32 而部署 alpha=None(scale 1) | 从 adapter meta 读 rank/alpha |
| 4 | 手动 DDP 端口占用 | `EADDRINUSE 29517` | 换端口（29541） |
| 5 | pipeline 自杀 | `pkill -f` 匹配到自己的命令行 | 按精确 pid kill；加 `flock` |
| 6 | **两个 Isaac 并行** | `onnxruntime is not available` | **真因是 PYTHONPATH 污染**（见 #7），不是并发 |
| 7 | **继承父进程 PYTHONPATH** | Isaac 里 `import onnxruntime` 解析到半残包（`onnxruntime.capi.onnxruntime_pybind11_state` 缺失） | **只给 Isaac 留 `v7_tick_record:MAIN`**，让它自己 source `isaac_env.sh` |
| 8 | 相对路径写进主仓 | Isaac 进程 cwd=`$MAIN`，tick 落到主仓 | 一律 `resolve()` 绝对路径 |
| 9 | **A/B 在同一 worktree 并行改代码** | `git status` 出现 A/B 的文件；`tracking_recovery_tail.py` 被对方改过（新增 worst_window） | **改共享文件前先 `git status`/读现状**；能复用就复用（本次 P2 直接复用 A/B 的实现） |
| 10 | 动作域合同 | A′ 的 `actions` = `(cmd_q − dq_ckpt)/sc_ckpt`，不是 decoder 归一化输出、不是 43 维 env action | 采集器按同一仿射变换记录（q50 2.918 vs A′ 2.93） |
| 11 | 多 seed 合成一个 episode | `_episode_id` 每进程从 0 数 | 按 `(run_id, episode_id)` 分组 |
| 12 | 动作域检查按 segment 判 | 20 行里 4 行越界 = 20% ⇒ 误判"域错" | 改成**整体**比例（0.5% 长尾） |
| 13 | `self_q99` 在 zero-LoRA 上退化 | 29/29 全落到 floor 0.02 | 必须用**有修正的参照 adapter** 标定 |
| 14 | `--strict-wm-hash 1` | 被当成期望哈希"1" | 传真实前缀 `fc5aeb30e987f016` |
| 15 | **闭环脚本环境**（当前阻塞） | `run_closedloop_C_lora16.sh:40` 的 `source harness03_env.sh` 污染 PYTHONPATH ⇒ 4 次尝试全 `startup failed` | 见 §9 |

---

## 6. 已完成 / 未完成（V8-C 验收表）

| # | 项 | 状态 |
|---|---|---|
| 1 | 218 层 LoRA 真注入、16,891,904、原 proj 冻结 | **PASS** |
| 2 | zero-LoRA 与 pristine 逐位一致 | **PASS**（max|Δ|=0.0） |
| 3 | 4 步 FM 梯度回 LoRA | **PASS**（测试） |
| 4 | Recovery 长尾 + z0 Train-only 冻结 | **PASS** |
| 5 | 固定 peak 预算（逐关节、Train-only） | **PASS**（21 个不同值） |
| 6 | Trust 去重（取消边际） | **PASS** |
| 7 | 困难采样 70/30 + 支持域 | **PASS**（99.2% 通过） |
| 8 | 四路归因 / per-joint 报告 | **PASS** |
| 9 | 严格 resume（含指纹校验） | **代码就绪，未真实 resume 过** |
| 10 | 400 步正式训练 | **停在 280（平台，用户决定）** |
| 11 | 独立 Test（一次） | **PASS**（+0.01640） |
| 12 | 物理目标 <2.45% | **FAIL**（2.53%），根因是结构性 j4/j10 |
| 13 | `L_rate` | **BLOCKED**（RobotSpec 无 velocity 字段） |
| 14 | **18 对真实闭环** | **BLOCKED**（环境问题，见 §9） |

---

## 7. 结论（可对外写的三句）

1. **OFFLINE_GAIN：PASS**。固定 `wm_gain2` 时，C 的独立 test 想象回报 **+0.01640（+1.04%，CI 0.77–1.29%）**，是旧 C1 的 2.9 倍；结构正确（主干 80%/proj 20%）。
2. **RECOVERY/PHYSICAL：部分改善**。越限 2.59%→2.53%，四个主要关节全降（−3%~−12%），但 j4/j10 结构性平台（原始模型自己 25.78%/22.55%）。
3. **CLOSED_LOOP_GAIN：NOT_RUN（阻塞）**。18 对闭环因环境问题没跑成 ⇒ **不能声称真机提升**。

---

## 8. 复现命令

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
source scripts/harness03_env.sh

# 训练（最终配方，WM0 固定，280 步可复现）
python3 scripts/train_sonic_lora16_ft8000_bptt.py --out <OUT> --updates 400 \
  --c-recipe c1_lora_proj --lora-lr 3e-5 --proj-lr 3e-5 --lr 3e-5 --lr-total-updates 400 \
  --min-lr 5e-6 --warmup 20 --grad-clip 1.0 --seed 0 --trust-dual 1 --tau-same 0.03 \
  --lambda-trust-total 0.01 --lambda-trust-lora 0 --lambda-peak-lora 0 \
  --lambda-recovery 0.002223 --recovery-aggregate worst_window \
  --recovery-z0-json <CAL>/recovery_z0.json \
  --lambda-added 6.78179 --lambda-margin 0.2201 --margin-kappa 0.05 \
  --lambda-peak-total 1.8e-5 --lambda-lim 0.03068 \
  --peak-budget-json <CAL>/peak_budget_self_q99.json \
  --hard-start-ratio 0.30 --support-gate 1 --physical-report 1 --attribution-4way 1 \
  --strict-resume 1 --wm-checkpoint $COMIMIC_MAIN/runs/a_prime_sonic_exc/wm_gain2.pt \
  --wm-version wm_v000 --wm-refresh-mode fixed --strict-wm-hash fc5aeb30e987f016 \
  --ft8000-checkpoint $COMIMIC_MAIN/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --vlm-path $COMIMIC_MAIN/groot_repro/nvidia/Cosmos-Reason2-2B-weights \
  --limits-npz runs/three_track_v6/physical/limits_standard29.npz \
  --eval-set runs/temporal_ac_bptt_v5/eval_set --device cuda:0 \
  --batch 32 --vla-batch 32 --fm-grad-backend checkpointed \
  --eval-every 50 --eval-batch 16 --cond-check-every 25 --save-every 50 --log-every 10

# 独立 test（任意 C adapter）
python3 scripts/eval_c_v8c_test.py --adapter <OUT>/checkpoint_lora_best.pt --split test \
  --device cuda:1 --out <OUT>/test_best

# WM 保真（ratio + 端点）
python3 scripts/probe_wm_multistep.py --ckpt $COMIMIC_MAIN/runs/a_prime_sonic_exc/wm_gain2.pt \
  --n 256 --k 20 --device cuda:1 --out <OUT>/probe
#   注：脚本内部路径写死 ROOT，需要 runs/{isaac_motion,S1_handover,a_prime_sonic_leash,a_prime_sonic_exc}
#   软链到 MAIN（本会话已建，只读）

# 单测（全部离线，不需要 Isaac）
python3 -m pytest tests/test_wm_replay_buffer.py tests/test_wm_version_store.py \
  tests/test_wm_refresh_gate.py tests/test_tracking_recovery_tail.py \
  tests/test_support_aware_sampler.py tests/test_physical_bptt_losses.py \
  tests/test_lora16_dummy.py tests/test_tracking_reward_tail.py -q
```

---

## 9. 当前唯一硬阻塞：闭环跑不起来（含精确修法）

**现象**：`scripts/run_closedloop_C_lora16.sh` 的 4 次尝试全部在 ~50 s 内失败：
`[sim_eval_vla] startup failed: RuntimeError: [SonicActionProvider] onnxruntime is not available`，
episode 日志里同时可见 `[audit] dep_onnxruntime=MISSING(ModuleNotFoundError)`。

**根因**：`scripts/run_closedloop_C_lora16.sh:40` 有 `source "$WT/scripts/harness03_env.sh"`，
它把 worktree 的 `PYTHONPATH` 顶到最前，**遮住 Isaac venv 自带的 onnxruntime**（坑 #7 的同源问题）。

**修法（10 分钟）**：把两个环境拆开——
1. **server**（需要 conda 的 torch）：显式 `PATH=/opt/miniconda3/envs/isaaclab/bin:$PATH` + `COMIMIC_MAIN`，**不要**套 worktree PYTHONPATH；
2. **Isaac**（`ha_isaac_closedloop.sh`）：**干净 PYTHONPATH**（只留 `$COMIMIC_MAIN`，让它自己 `source scripts/isaac_env.sh`）；
3. **先跑 1 集验证**（约 4 分钟），确认 `result.json` 有 `episode_steps > 100`，再放 18 集；
4. 双卡并行**不是问题**（两流水线失败的报错与单条完全相同），但需在 1 集验证通过后再开。

**18 集计划**（已验证可用的参数）：`TASKS="OpenDoor SitSofa VisionNavi"`、`SEEDS="0 1 2 3 4 5"`、`ARMS=lora`（＝V8-C Best）、`MAX_STEPS=3000`、`RECORD_VIDEO=1`。
要构成对照还需 `ARMS="lora base"`（base＝无 LoRA 的 FT8000）以及旧 C1 adapter 一次。

---

## 10. 下一步建议（按性价比）

| 优先级 | 做什么 | 成本 | 为什么 |
|---|---|---|---|
| **P0** | 修 §9 的环境，跑通 1 集，再放 18 集（V8-C） | 15 min + 1.5 h | 唯一能回答"真机是否有用"的实验 |
| P0 | 同时/随后跑 **Base（无 LoRA）18 集** | +1.5 h | 没有 Base 就无法说"LoRA 带来了提升" |
| P1 | **固定窗口尾部诊断（C 版）**：4 个 checkpoint 的 z_p50/p95/p99 | 30 min | 奖励均值饱和，尾部才是 Recovery 的战场 |
| P1 | **seed1** 同配方 280 步 | 40 min | 所有结论都是单 seed |
| P2 | 若闭环显示"漂移/站桩" ⇒ 上 **H=40 辅助视界 + 速度项加权** | 1–2 天 | 漂移是秒级现象，H=20 看不到 |
| P2 | 若闭环显示"任务阶段/抓取" ⇒ 审 **j4/j10 限位定义**、补恢复类参考动作 | 半天 + 数据 | 结构性，不是训练能解 |
| ❌ | 再刷一次 WM 指望涨点 | —— | 实测 4.59→4.56，只值 +9.5% Actor 收益 |
| ❌ | 继续加训练步数 | —— | it=100 后 5 点无趋势 |

---

## 11. 产物索引（全部在 worktree，旧目录只读）

```
runs/three_track_v7/
├ phase0_freeze/inventory.json                     # 20 产物 sha
├ wm_versions/{wm_v000, wm_c1_v001_candidate, wm_c1_v001_published, active.json}
├ buffer/cycle000_A/                               # 366 段真 Isaac A 线数据（12 episodes, 7,552 tick）
├ diagnostics/wm0_oldonly/{diagnostics.json,REPORT.md}
└ c1_lora_proj/pilot200/                           # 旧 C1（proj 全秩）200 步

runs/three_track_v8/c_tracking_recovery/
├ calibration/{physical_weights_v8c.json, recovery_z0.json,
│              peak_budget_train_v8.json, peak_budget_self_q99.json}
├ wm_refresh/{refresh.log, gate*/gate.json}
├ actor_seed0/                                     # WM1 400 步（test +0.01796）
├ actor_seed0_wm0_stopped_at190/                   # WM0 190 步（归档）
├ actor_wm0_seed0_stopped_at100/                   # WM0 100 步（归档）
├ actor_wm0_p2p4/                                  # **最终 run**
│   ├ train.log/train.jsonl/val.jsonl/cond_check.jsonl
│   ├ attribution.jsonl / physical_report.jsonl / stopped.json
│   ├ checkpoint_lora_best.pt(it=250) + 每 50 步
│   └ test_best/test.json                          # **独立 test**
├ diag_wm0_same/ diag_wm1_same/                     # 同协议 WM0 vs WM1
├ probe_wm0_multistep/probe.json                    # ratio 1.02 / 端点 2.11cm
└ closedloop_v8c/ cl_A/ cl_B/                      # 闭环尝试（失败，日志在）

docs/（worktree 与 comimic/docs 双份）
├ THREE_TRACK_V6_DELIVERY_20261108.md
├ TRACK_ISSUES_PARAMS_RESULTS_20261108.md
├ C_LORA_PROBLEM_STATEMENT_20261108.md
├ C1_AFTER_REVIEW_PROBLEM_20261108.md
├ V7_PROGRESS.md
├ V8C_V7_MERGED_PLAN_20261108.md
├ CURRENT_STATUS_FOR_REVIEW_20261108.md
├ C_STATUS_V8C_FINAL_20261108.md
└ HANDOVER_FULL_SESSION_20261108.md（本文）
```

**环境事实**：worktree `comimic-wt03`（分支 `harness/03-bptt-diagnosis`）；主仓 `comimic`（权重/数据，只读）；
2× RTX 4090（49 GB）；`scripts/harness03_env.sh` 设 `COMIMIC_MAIN`/PYTHONPATH；Isaac venv 在
`$COMIMIC_MAIN/third_party/isaacsim_env/bin/python`（自带 onnxruntime 1.22.1）；磁盘 `gpfs_hdd` 10 T 用 9.2 T（**剩 824 G**）。

**边界**：`runs/three_track_v6/`、`runs/a_prime_sonic_exc/` 只读；本轮无 Task Head / 无 Critic / 未升 cap / 未改 SONIC 与 FT8000。
