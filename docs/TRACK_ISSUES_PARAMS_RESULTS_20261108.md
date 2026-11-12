# 三路线现状：参数 / 结构 / 设计 / 结果 / 问题（2026-11-08 晚）

> 本文是 `docs/THREE_TRACK_V6_DELIVERY_20261108.md` 的**问题导向补充**：把 A / B / C 三条路线
> 的**结构、锁定参数、实测结果、当前问题**逐项摊开，每条结论都带**证据位置**（日志/字段/文件）。
> 状态口径沿用交付文档：`PASS / RUNNING / BLOCKED / NOT_RUN`，**没跑过的不写跑过**。
>
> 代码 worktree：`comimic-wt03`（commit `b9052fb` 之后继续改），数据/权重在只读的 `comimic`。

---

## 0. 状态矩阵（一眼看结论）

| 路线 | 代码 | 训练 | WM 评测 | 真实闭环 | 当前状态 |
|---|---|---|---|---|---|
| **A** Delta-Continue | 完成 | **PASS** 300 upd | val **+0.02823** / test **+0.02185** | **9/18**（最新一次 18 集） | **完成，可复用** |
| **B** Delta-Retrain | 完成 | seed0 到 **960/1000 被本会话手动停** | seed0 cond PASS | 未做 | **BLOCKED（人工停）**，seed1/2 NOT_RUN |
| **C** LoRA16-FT8000 | 完成 | 旧跑 700/800（lr1e-5）→ **新跑 RUNNING（DDP×2, bs64, lr1e-4, 400 步）** | 旧跑 val **+0.00548**（LoRA 边际）/ 总 +0.0374 | **未做**（脚本就绪） | **主跑 RUNNING**，闭环 NOT_RUN |

---

## 1. 共用骨架（三条路线同一套）

### 1.1 结构

```
冻结 FT8000 VLA (Gr00tN1d7)
  ├─ backbone (Qwen3-VL 2B, 冻结)  → 观测特征（每步 no_grad 缓存）
  └─ action_head = AlternateVLDiT (1.0917 B, 冻结)
        ├─ action_encoder / DiT blocks(attn1/attn2/ff) / action_decoder
        └─ 4 步 Euler flow-matching 采样 → x_norm (B,40,132)
              ↓ codec.decode（可微 min-max 反归一化）
        token = x_raw[..., :64]  (B,40,64)
              ↓
A/B: tokens + Δ(token, state64, hist10)   ← A/B 唯一可训的地方（1.74 M）
C  : tokens 直接来自 LoRA 化后的 head     ← C 可训 LoRA(16.8 M)+proj_out(6.3 M)
              ↓
SOINC 执行器（冻结 WBC, WBC_A24000_wmvel_cmd.pt）
              ↓ q_raw = q_default + G1_ACTION_SCALE ⊙ a   （正确物理口径）
A′ 世界模型（冻结, wm_gain2.pt）→ H=20 步想象 rolling → J20（折扣回报）
```

### 1.2 锁定参数（三条路线共用）

| 项 | 值 | 出处 |
|---|---|---|
| `T`（VLA chunk） | 40 | `config.json: action_horizon=40` |
| `K`（执行 token 数） | 20 | `HA_VLA_EXEC_STEPS` |
| `H`（可微 WM 步数） | 20 | `--horizon 20` |
| `FM steps` | 4（与部署一致） | `num_inference_timesteps=4` |
| `dt`（物理） | 0.02 s（50 Hz） | `l_rate` 口径 |
| `G1_ACTION_SCALE` | 0.075–0.548（standard 序） | `diff_vla_wbc/sonic_physical_action.py` |
| 物理限位 | `runs/three_track_v6/physical/limits_standard29.npz` | MJCF 90% soft |
| 评测集 | 448 窗口：256 train / 96 val / 96 test（375 train clips） | `runs/temporal_ac_bptt_v5/eval_set` |
| 训练起点 | `n_eligible=235,280`（只 train clips ⇒ 无 val/test 泄漏） | `train*.log: [C] 起点` |
| 部署口径 | `state_hist_10`（HA 序、减训练 default）、`--delta-v5-hist-order ha_v31` | server 启动门禁 |

### 1.3 测量协议（决定"能不能看见小效应"）

1. **成对同噪声**：同一批帧用同一 flow-matching 初噪采两次（`lora_scale=1` vs `=0`），
   差值才是"策略变化"，否则被采样噪声淹没（token 的采样噪声 ≈ 1e-3 量级，C 的效应量同阶）；
2. **cached-token 交叉检查**：`J_cached`（用评测集缓存 token）在**每次评测必须逐位相同**
   —— 它同时是"评测代码没被动过"的证据（实测 12 次都是 1.50594/1.57581）；
3. **物理量永远用 `q_raw`（裁剪前）+ 不对称限位 + 归一化尺度**；
4. 闭环判定用 `result.json.success`（sparse），**reward=0 ≠ 没进步**。

---

## 2. A：Delta-Continue（结论最硬的一条）

### 2.1 结构 / 参数

| 项 | 值 |
|---|---|
| Δ 网络 | `TemporalDeltaActorV5`：chunk-attention + history-GRU，**1,744,832** 参数，`d_in=400` |
| 输入 | `tokens(40,64)`、`state64(64)`、`hist10(10,58)` |
| 输出 | `Δ(40,64)`，`t≥K` 恒 0；`Δ = cap·tanh(u/cap)`，**cap=0.08** |
| 初始化 | v5 best warm-start（只加载 Actor 权重，优化器从 0） |
| lr | head **5e-5** / enc **2e-5**，warmup 20，batch **256**，300 updates |
| λ | `λ_lim=0.0529`、`λ_peak=8.4e-4`、`λ_same=0.010`（未激活）、`λ_rate=0`（BLOCKED）、`λ_raw=1e-3` |
| 损失 | `-J20 + λ_lim·L_lim + λ_peak·L_peak + λ_same·L_same + λ_raw·L_raw` |

### 2.2 结果（PASS）

| 指标 | 数值 | 证据 |
|---|---|---|
| WM val ΔJ | **+0.02823** | `runs/three_track_v6/delta_continue/seed0/val.jsonl` |
| WM test ΔJ | **+0.02185** | 同上 `test_once.json` |
| cond（cosZ / Δstd） | 0.655 / 通过 | `cond_check.jsonl` |
| ‖Δ‖（执行区） | 0.0779 | `train.jsonl` |
| ‖u‖/cap p50 / p95 | 2.47 / 4.85 | 同上（cap 前原始范数） |
| 物理：Δ@t=14 | **2.065 → 0.425 rad（−79%）** | 交付文档 §3A |
| 物理：限位违反率 | 2.74% → 2.90%（基本持平，绝对值由 Base 主导） | 同上 |
| 真实闭环 18 集 | **9/18**（OpenDoor 3/6、SitSofa 4/6、VisionNavi 2/6） | `isaac/A_delta_continue/RESULTS.md` |

### 2.3 问题（按严重度）

1. **P1 · cap 饱和（设计内，但压住了上限）**
   `‖u‖/cap p50 = 2.47`、p95 = 4.85、74% 的窗口 >2×cap。`tanh(2.5)` 的导数 0.027 vs 1×cap 处 0.42
   ⇒ **"改幅度"的有效梯度被压 ~16×**；`Δ` 的范数钉在 0.0779/0.08（97.5%）。
   证据：`train.jsonl: u_norm_over_cap_p50/frac_u_gt_2c`。
   **杠杆**：cap 0.08→0.10/0.12，或 `λ_raw` 1e-3→1e-2 量级（把 ‖u‖ 拉回 cap 附近）。
2. **P2 · 条件化偏弱（疑似 encoder lr 偏小）**
   cond `cos_shuffle_z = 0.655`：shuffle 掉 token 后 Δ 只掉 35%。enc-lr 只有 2e-5。
   **杠杆**：enc-lr 1e-3 跑 100 步看 `cos_shuffle_z` 是否显著下降（越低 = 越依赖输入）。
3. **P3 · 闭环方差大**：v5 best 曾在 OpenDoor 2/2 成功（929/1633 步）；A best 这次 3/6 且 seed0 需第 3 次尝试。
   同一任务同一策略两次跑差得多 ⇒ 单次 success 不能当结论，必须多 seed + 看步数分布。
4. **P4 · provider 未落盘逐 tick 物理量**（`q_raw/q_applied/q_real/限位/任务进展`），
   闭环只能看 sparse success ⇒ 无法回答"到底哪一步开始偏"。

---

## 3. B：Delta-Retrain

### 3.1 结构 / 参数

与 A **同一个训练器**（`scripts/train_delta_physical_bptt_v6.py`），区别只有：

| 项 | A | B |
|---|---|---|
| 初始化 | `--init-delta-checkpoint` = v5 best | **随机初始化**（不传该参数） |
| lr | head 5e-5 / enc 2e-5 | head **3e-4** / enc **1e-4** |
| warmup | 20 | 50 |
| updates | 300 | 1000 × seed0/1/2 串行 |
| batch | 256 | 256 |

### 3.2 结果（INCOMPLETE）

| 指标 | 值 | 证据 |
|---|---|---|
| seed0 进度 | **960/1000**（本会话 10:25 "全停"命令手动终止） | `runs/three_track_v6/QUEUE_STATUS.txt`、`delta_retrain/seed0/train.jsonl` |
| cond（~480 步时） | PASS，cosZ **0.624** | `cond_check.jsonl` |
| ‖Δ‖ | 0.0776（同样顶满 cap） | `train.jsonl` |
| ‖u‖/cap p50 | 2.36（frac>2c 70%，frac>3c 20%） | 同上 |
| seed1 / seed2 | **NOT_RUN** | — |

### 3.3 问题

1. **P0 · 被人工中断**：960/1000，差 40 步 + 收尾评测。**这不是失败，是本次"停一切"的副作用**，
   恢复只需重跑队列（见 §6）。
2. **与 A 同样的 cap 饱和**（p50 2.36）⇒ A 的结论对 B 同样成立：不是 LR 推不动，是预算用满后幅度饱和。
3. **B 的"随机初始化 + 3e-4"比 A 更激进**：cond cosZ 0.624 略低于 A 的 0.655，
   但两者的 cond 判据都只是"不是常量"，离"强条件化"还有距离。

---

## 4. C：LoRA16-FT8000（本次重点）

### 4.1 结构（可训部分）

```
FT8000 (1,637,293,184 参数, fp32, 冻结)
  action_head (AlternateVLDiT 1.0917 B)
    ├─ attn1 / attn2 / ff 共 **216 层** nn.Linear → 替换为 LoRALinear
    │     lora_A (16×in), lora_B (out×16), scale = alpha/r = 32/16 = **2.0**, B 零初始化
    │     可训 **16,777,216**
    └─ 输出块 proj_out_1 [3072,1536] + proj_out_2 [1024,1536]（含 bias）
          可训 **6,295,552**（全秩，`--lora-also-train proj_out`）
合计可训 **23,072,768 = 1.409%**
```

- `proj_out_1`：吃 timestep/embodiment 条件 `temb` → `shift, scale`，做**最后一步 adaLN 调制**
  `h = norm_out(h)·(1+scale)+shift`；
- `proj_out_2`：DiT hidden(1536) → 1024 维输出特征（再由 `action_decoder` 映到 132 维动作）。
- **为什么它必须可训**：这两层是"动作落在哪里"的最高杠杆旋钮；rank16 挂在 attn/ff 上只能间接影响。
- **为什么它必须被单独报告**：见 §4.5 归因。

### 4.2 数据 / 采样（C 与 A/B 的关键差别）

| 项 | A/B | C |
|---|---|---|
| token 来源 | 冻结 VLA + **独立 Δ 网** | **LoRA 化后的 VLA 自己**（4 步可微 FM） |
| 梯度路径 | `J → q_raw → Δ 网` | `J → q_raw → token → codec → 4 步 Euler → LoRA/proj_out` |
| 采样 | 每步新 flow 噪声；Δ 对 token 做残差 | 每步新噪声；**成对**同噪声采「LoRA」与「底模」 |
| 显存 | batch 256 也才 20 GB（Δ 网小） | **13 GB 权重** + 4 步图：bs16 pathwise 会 OOM，bs16+checkpointed ≈17 GB，DDP 每卡 bs32 ≈21 GB |

### 4.3 三次训练的配置与结果

| 跑 | 配置 | 状态 | val ΔJ | test ΔJ | 违规率 | q_raw 峰 | Δtoken rms |
|---|---|---|---|---|---|---|---|
| pilot 200 | lr1e-5, bs16, λ(A 臂) | PASS | +0.00164 | +0.00198 | 2.63% | 4.658 | 0.0014 |
| 主旧跑 | lr1e-5, bs16, λ(C 重标) | **700/800 人工停** | **+0.00548**(650) | — | **2.45%** | 4.651 | 0.0042 |
| **主新跑** | **DDP×2, bs64, lr1e-4, 400 步** | **RUNNING** | 见 `train400_lr1e-4_ddp/val.jsonl` | — | — | — | — |

底模参照（fresh-noise，96 窗口）：`val J_base = 1.49194`、`test J_base = 1.57558`、
`vio = 2.77% / 2.59%`、`q_raw_absmax = 4.603 / 3.580`。

旧跑 val ΔJ 曲线：`50:+0.00204 → 100:+0.00094 → 200:+0.00164 → 300:+0.00363 → 400:+0.00408
→ 450:+0.00526 → 550:+0.00518 → 600:+0.00506 → 650:+0.00548`（走平，不发散；cond **26/26 PASS**）。

### 4.4 λ 标定（C 手臂自己的梯度预算）

`scripts/calibrate_physical_bptt_weights.py --actor lora`（在 200 步 Adapter 上，16 窗口）：

| 项 | J | lim | peak | same | rate |
|---|---|---|---|---|---|
| grad norm | 1.157 | 15.68 | 1684.3 | 0 | 0 |
| ratio to ‖g_J‖ | 1× | 13.56 | 1456 | 0 | 0 |
| cos with g_J | — | +0.396 | **−0.443**（与 J 对抗） | — | — |
| λ | — | **0.0369** | **2.06e-4** | 0.010（未激活，用默认） | 0（BLOCKED） |

### 4.5 归因（本次最重要的发现）

在 it=600 的 Adapter 上，16 帧、同 seed：

| 对照 | token 变化 rms | 说明 |
|---|---|---|
| proj_out 单独（关 LoRA） | **3.053e-2** | 占绝对大头 |
| LoRA 边际（全开 vs 关 LoRA） | **3.212e-3** | 小一个数量级 |
| 全 Adapter vs pristine FT8000 | 3.275e-2 | ≈ proj_out 主导 |

WM 侧对应：`val J` 从 pristine 1.49194 → 全 Adapter 1.52938（**总 +0.0374**），
其中「只留 proj_out」1.52432（**+0.0324**）、**LoRA 边际仅 +0.0051**。
⇒ **"C 有效"这件事，主要功劳在 proj_out（全秩 6.3 M），不在 rank16 LoRA（16.8 M）。**
任何把 +0.0374 说成"LoRA 的效果"的表述都是错的。

### 4.6 C 的问题清单（含已修）

| # | 问题 | 状态 | 定位/证据 |
|---|---|---|---|
| C1 | **`also_train` 静默 no-op**：`apply_lora(..., also_train=("proj_out",))` 从未解冻任何参数 ⇒ S3 那批"LoRA+proj_out"实际只训了 LoRA | **已修**（本次=`lora.py` 按完整路径解冻；C 是第一次真的训到 proj_out） | `diff_vla_wbc/lora.py:83-95` |
| C2 | **`l_same` 在 e≡0 处反向 NaN**：`sqrt(mean(e²))` 的导数是 inf，zero-LoRA 首步 `e_same≡0` ⇒ `0·inf=NaN`，LoRA 梯度全毁 | **已修**（平方和 +1e-12；对 rms>1e-4 相对改动 <1e-8，A/B 结果不变） | `physical_bptt_losses.py:l_same`；回归测试 `tests/test_physical_bptt_losses.py` |
| C3 | **pathwise 4 步图 OOM**：bs16 pathwise 峰值 26.3 GB，与 B 训练同卡必炸 | **已修**（`--fm-grad-backend checkpointed`，数值与 pathwise 逐位等价，已有测试） | `tests/test_fm_grad_backends.py` |
| C4 | **zero-LoRA 恒等门禁挡住 resume**：续跑时 Adapter 已训练（B≠0），token 恒等必然不成立 | **已修**（只有 `fresh_adapter` 才硬门禁；续跑只记录并加上限 0.5） | `train_sonic_lora16_ft8000_bptt.py` |
| C5 | **server LoRA scale 必须来自 Adapter meta**：CLI 默认 `lora-rank=32, alpha=None(scale=1.0)`，会把训练时 scale=2.0 的增量**缩放一半** | **已修**（`--lora16-ckpt` 从 `meta.lora.rank/alpha/targets` 读；`--lora16-disable` 做 base 臂） | `scripts/ha_vla_latent64_server.py:_load_lora16` |
| C6 | **LR 偏小（主因之一）**：lr=1e-5 时 `lora_B` absmean 仅 1.79e-4、token 只挪 3.2e-3（A 的 Δ 是 0.078）；且 cosine→1e-6 让第 600 步只剩 1.5e-6 | **已改**（新跑 lr=1e-4、cosine→1e-5、warmup 20） | `train.jsonl: lr`；`checkpoint_lora_0600.pt` 直接数 |
| C7 | **batch 偏小**：旧跑 bs16（总样本 12.8k），A/B 用 256 | **已改**（DDP 双卡 × bs32 = **有效 64**，400 步 ⇒ 25.6k 样本） | `config.json: ddp` |
| C8 | **DDP 实现**：不能用 `DDP.forward`（要自己控 `backbone` no_grad 与 4 步 FM 子模块调用） | **已实现**：每 rank 独立前向 + **手动 all-reduce 梯度**；rank0 独占日志/评测/存档；早停用 all_reduce 同步 | `--ddp 1` + `torchrun --nproc_per_node=2` |
| C9 | **双卡端口冲突**：上次探针占用 29517 ⇒ torchrun `EADDRINUSE` | 已绕开（29541）；复跑请换端口 | `train400_lr1e-4_ddp/train.log` |
| C10 | **主跑尚未闭环**：`lora`（LoRA+proj_out）vs `base`（--lora16-disable=只留 proj_out）配对消融已就绪 | **NOT_RUN**（训练完立刻跑） | `scripts/run_closedloop_C_lora16.sh` |
| C11 | `L_rate` BLOCKED：`RobotSpec` 无 velocity 字段 | BLOCKED | 交付文档 §8 |

### 4.7 DDP 实测（速度/显存）

| 配置 | 显存/卡 | 步时 | 400 步预估 |
|---|---|---|---|
| 单卡 bs16 + checkpointed（旧跑） | ~17 GB | 7–12 s | — |
| **双卡 DDP，每卡 bs32（=有效 64）** | **~21 GB** | **~4–9 s**（首步 9.3 s 含预热） | **35–60 min** |

---

## 5. 跨路线问题（共同的坑）

### 5.1 "lr 太小 / batch 太小"到底怎么判断（三条路线各自的答案）

| 路线 | 瓶颈 | 证据 | 杠杆 |
|---|---|---|---|
| **A/B（Δ 网）** | **不是 lr，不是 batch**：batch 256（C 的 4×），且 Δ 已顶到 cap 97.5%，cap 前 ‖u‖/cap p50=2.4–2.5 | §2.2、§3.2 | **cap 0.08→0.10/0.12** 或 `λ_raw` 1e-3→1e-2；enc-lr 单独验条件化 |
| **C（LoRA）** | **是 lr（+ batch）**：lr1e-5 下 `lora_B` 只到 1.79e-4、token 挪 0.3%（A 是 7.8%） | §4.6 C6/C7 | lr **1e-4**、有效 batch **64**、400 步（本轮已在跑） |

### 5.2 归因污染（只有 C 有）

A/B 只训独立 Δ 网 ⇒ 效果天然归因清楚。C 解冻了 proj_out ⇒ **必须做 token 级分解**，
否则会把 proj_out 的 +0.0324 记成 LoRA 的功劳（§4.5）。

### 5.3 目标都是"冻结 WM 的想象回报" ⇒ 都有 reward hacking 风险

- 表现：`J` 上去但物理量变差（q_raw 峰↑、限位违反↑）、或 token 越出分布；
- 现有防线：`L_lim/L_peak/L_same`（物理预算）+ `L_raw`（Δ 幅度）+ cond 门禁 + **闭环才是裁判**；
- C 额外风险：proj_out 是高杠杆全秩层，更容易在 WM 上"作弊"⇒ 闭环是唯一硬证据。

### 5.4 采样/评测协议本身是个坑

- flow-matching 的采样噪声与 C 的效应量同阶（1e-3）⇒ **必须成对同噪声**；
- 评测集缓存 token（`J_cached`）在 12 次评测里逐位不变（1.50594 / 1.57581），这是评测代码没漂移的证据；
- 若用不同 seed 采 base/lora，ΔJ 会被噪声吃掉（旧跑 first eval 就差点误判）。

### 5.5 已修但值得记住的 bug 类

1. **静默 no-op**（`also_train`）——最危险：报告里写了"训练了 X"，代码其实没训；
2. **数值退化点**（`sqrt(0)`）——只在"新初始化状态"暴露，常规训练永远看不到；
3. **配置传递**（server 的 rank/alpha 必须从 ckpt meta 来）——否则 scale 静默减半；
4. **并行 Isaac** 会 `onnxruntime not available`（本次复现过），一次只能跑一个；
5. **多实例脚本**会互相写同一个 `OUT_ROOT` ⇒ 必须单实例锁（已加 `flock`）+ 端口清理。

---

## 6. 当前进程 / 复跑命令

### 6.1 正在跑

| 进程 | 内容 | 查看方式 |
|---|---|---|
| C 主新跑（DDP×2） | `train400_lr1e-4_ddp`，lr1e-4，有效 bs64，400 步 | `tail -f runs/three_track_v6/lora16_ft8000/train400_lr1e-4_ddp/train.log` |

### 6.2 已停（本会话 10:25 "全停"）但要恢复的

| 项 | 状态 | 恢复方式 |
|---|---|---|
| B seed0 | 960/1000 | 重跑 `scripts/train_delta_physical_bptt_v6.py`（`--updates 1000`，无 `--init-delta-checkpoint`） |
| B seed1/seed2 | NOT_RUN | 串行队列 `scripts/queue_A_lora.sh` 或手动两个 seed |

### 6.3 C 的完整命令

```bash
# 主跑（双卡 DDP，有效 batch 64，lr 1e-4）
source scripts/harness03_env.sh
O=runs/three_track_v6/lora16_ft8000/train400_lr1e-4_ddp
torchrun --nproc_per_node=2 --master_port=29541 \
  scripts/train_sonic_lora16_ft8000_bptt.py --out $O --updates 400 --ddp 1 \
  --lr 1e-4 --lr-total-updates 400 --min-lr 1e-5 --warmup 20 --grad-clip 1.0 --seed 0 \
  --ft8000-checkpoint $COMIMIC_MAIN/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --vlm-path $COMIMIC_MAIN/groot_repro/nvidia/Cosmos-Reason2-2B-weights \
  --limits-npz runs/three_track_v6/physical/limits_standard29.npz \
  --physical-weights runs/three_track_v6/calibration/lora16_weights.json \
  --eval-set runs/temporal_ac_bptt_v5/eval_set --device cuda:0 \
  --batch 32 --vla-batch 32 --accum 1 --fm-grad-backend checkpointed \
  --eval-every 50 --eval-batch 16 --cond-check-every 25 --save-every 50 --log-every 5 --base-eval 1

# 闭环（训练完）：lora 臂 → base 臂（只留 proj_out），同 seed 配对
ARMS="lora base" LORA=$O/checkpoint_lora_last.pt DEVICE=cuda:1 PORT=18096 \
  SEEDS="0 1 2" bash scripts/run_closedloop_C_lora16.sh
```

### 6.4 复现门禁

```bash
python3 -m pytest tests/test_lora16_dummy.py tests/test_four_step_fm_grad.py \
  tests/test_physical_bptt_losses.py -q          # 纯 CPU
COMIMIC_TEST_DEVICE=cuda:0 python3 -m pytest tests/test_sonic_wm_grad_to_lora.py -q  # 真实 WM
python3 scripts/audit_ft8000_lora16.py --ft8000 ... --vlm ... --out ...            # 真实审计
```

---

## 7. 下一轮实验建议（按性价比排序）

| 优先 | 实验 | 目的 | 成本 |
|---|---|---|---|
| 1 | C 闭环 `lora` vs `base`（本轮训练后立刻） | C 是否真的在真实任务上有用（唯一硬裁判） | ~2.5 h |
| 2 | C 的 **LoRA-only** 对照（`--lora-also-train ""`，200 步 lr1e-4） | 把 LoRA 的净贡献与 proj_out 分开 | ~25 min（DDP） |
| 3 | A/B 的 **cap 0.10/0.12** 或 `λ_raw 1e-2` 100 步 | 解 cap 饱和，看 ‖Δ‖ 能不能突破 0.08 | ~30 min |
| 4 | A/B 的 **enc-lr 1e-3** 100 步 | 验证"条件化偏弱"是不是 encoder 学得慢 | ~30 min |
| 5 | provider 逐 tick 落盘（`q_raw/q_applied/q_real`） | 闭环失败时能定位到"哪一 tick 开始偏" | 代码 ~1 h |
| 6 | `L_rate`（需要 Isaac actuator cfg 的 velocity 限） | 物理约束补全 | BLOCKED |

---

## 8. 一句话总结

- **A 是当前最可靠的成果**（WM ΔJ +0.028 / test +0.022 / 物理 t=14 尖峰 −79% / 闭环 9/18），
  代价是 cap 饱和把上限压住；
- **B 只差 40 步**，被人工叫停，不是失败；
- **C 的工程全部打通**（可微 4 步 FM → LoRA 梯度、真实审计、DDP、server 部署、闭环脚本），
  但**旧跑 lr=1e-5 明显偏小**，且**收益 85% 来自 proj_out 而不是 LoRA**；
  新跑（bs64 / lr1e-4 / DDP）正在验证"把 LR 与 batch 拉回正常量级后，LoRA 到底有没有用"。
