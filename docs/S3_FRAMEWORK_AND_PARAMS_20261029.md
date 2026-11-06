# S3 当前框架 / 架构 / 参数 —— 客观审计（2026-10-29）

> 目的：把现在**真正在跑**的东西写清楚，并逐条标出哪些参数是合理的、哪些有问题、问题多大。
> 所有数值都来自 `runs/S3_loop/*/args.json`、`v3_config.json`、`config.json` 和训练日志，不是回忆。

---

## 1. 系统架构（闭环数据流）

```
Isaac Sim 5.0 + IsaacLab 2.2（HumanoidArena，6 类任务）
  │  每 20 步问一次 VLA           HA_VLA_EXEC_STEPS=20
  │  ────────────────────────────────────────────────
  │  输入: ego RGB 224×224 + state46（本体感觉 46 维；手指恒 0，SM_HAND=0）
  │        ↓
  │  GR00T N1.7（Cosmos-Reason2-2B, bf16）
  │        backbone 冻结（tune_llm=False, tune_visual=False）
  │        action_head = 1.09B DiT + 0.20B self-attn   ← **只训这里的 LoRA**
  │        输出 action_horizon=40 步 × 78 维（64 token + 14 手指）
  │        flow-matching 采样，K=4 Euler 步，**每次重规划重新抽噪声**
  │        ↓ token(64)
  │  WBC（trackerlab policy，19M，obs 93 维 = 10 帧历史）
  │        TRAIN_WBC=0 冻结 / =1 一起训
  │        ↓ 29 关节目标 q_target
  └─ 50 Hz 执行 20 步 → 落盘 states / actions / tokens / frames / q_target_applied

落盘 dump（*.pt）
   → ① 打分 compute_collect_reward（训练同一个 reward）
   → ② 真转移 build_real_transitions_v2（B 线 on-policy 转移）
   → ③ 刷新 WM finetune_wm_velocity_policy（+ 门 eval_wm_multistep）
   → ④ 联合微调 train_joint_v2（V3：奖励项 + 锚定 + 可选 token BC）
   → ⑤ 回到 ①（策略变了就重采）
```

**三套模型与可训部分**

| 组件 | 规模 | 状态 |
|---|---|---|
| VLM backbone（Cosmos-Reason2-2B） | ~2 B | **冻结** |
| action_head | 1.09 B DiT + 0.20 B | **LoRA 33,554,432 / 1,654,070,400 = 2.03%** |
| WBC policy | 19 M | 默认冻结；`TRAIN_WBC=1` 时解冻 |
| WM（bundle.wm, explicit_dt0.02） | — | 冻结；每轮可刷新 |

LoRA 实测：`rank=32 alpha=32 layers=216 targets=attn1,ff proj_out_train=True`。

---

## 2. 训练参数全表（`runs/S3_loop/b1/train_main/args.json` 实拉）

| 参数 | 值 | 来源 | 客观评价 |
|---|---|---|---|
| `v3` | True | 文档要求 | ✅ |
| `horizon` H | 20 | 文档 / 与部署一致 | ✅ |
| `fm_steps` K | 4 | 文档 / 与 server 一致 | ✅ |
| `batch_size` | 4 | 文档 | ⚠️ 见 §3.1 |
| `grad_accum` | 8 | 文档 | 有效 batch 32，本身合理 |
| `steps` | 100–150 | 文档 | ❌ 见 §3.1 |
| `fm_lr` | 3e-6（温和）/ 1e-4（完整轮） | 我调 | ❌ 见 §3.2 |
| `lora_rank / alpha` | 32 / 32 | 文档 | ✅ |
| `lora_targets` | attn1,ff | 文档（trainer 默认） | ✅ |
| `amp` | bf16 | 文档 | ✅ |
| `weight_decay` | **0** | 默认 | ❌ 见 §3.5 |
| `obs_noise` | **0** | 默认 | ❌ 见 §3.5 |
| `dr` | False | 我关（`--no-dr`） | ⚠️ 小数据下等于零增强 |
| `lambda_r` | 0.3（完整轮）/ 0（近期） | 文档 / 我关 | ❌ 见 §3.4 |
| `lambda_hand` | **0.2** | 文档 | ❌ 见 §3.3 |
| `lambda_bc` | **0** | 文档（v3 下无效） | ✅（本来就无效，不背锅） |
| `token_bc_weight` | 0（已关） | 我加过、已关 | ✅ |
| `trust_center` | `prev`（完整轮）/ `label`（近期） | 文档 / 我改 | ❌ 见 §3.3 |
| `act_eps` / `act_weight` | **0.6** / 1.0 | 文档 | ❌ 见 §3.3 |
| `trust_eps` / `trust_weight` | 0 / 0（prev 下被强制） | — | ❌ 同上 |
| `train_wbc` / `wbc_lr` | 1 / **1e-6** | 我定 | ⚠️ 见 §3.6 |
| `start_from_b` | `runs/B_*/…/dump/*.pt`，stride 20 | 文档 | ✅ |
| `critic` | False | 门没过（v_ev 0.289<0.30） | ✅ 有依据 |
| `clip_head` / `fm_clip` | 10.0 / 0.5 | 默认 | ✅ |
| `reward_mode` | `saturating` | 文档 | ❌ 见 §3.4 |

---

## 3. 客观问题清单（按严重性）

### 3.1 【最严重】数据量 / 步数不匹配 → 必然过拟合

训练真正能抽到的**起点窗口**（`--start-from-b`，stride 20，H=20）：

| 数据集 | dump 数 | 可用起点窗口 |
|---|---|---|
| `B_r1`（19 dump） | 19 | **≈ 239** |
| `B_r*`（全部） | 43 | **≈ 838** |
| 完整轮 `B_F1`（6 dump） | 6 | **≈ 90** |

而 150 步 × batch 4 = **600 个样本**。⇒ 在 `B_F1` 上等于**把同一批 90 个起点过了 6–7 遍**；
即使用全部 B 数据也要过 1.4 遍。

**证据**：第 7 轮 `loss/token_bc` 降了 3 倍（0.075→0.026），同期
留出 `token_mse` **0.00848→0.01054（变差）**、闭环 **0.382→0.413（变差）**。
标准的过拟合曲线。

**结论**：在把数据堆到 **≥30 集（≥1000 窗口）** 之前，`steps` 必须压到 **30–50**（≤1 遍）。

### 3.2 【严重】`fm_lr` 与步数的乘积太大

实测权重相对变化：`fm_lr 1e-4 × 150 步` ⇒ **~10%**（`‖Δ‖/‖π_k‖`）。
对一个 2.03% 可训参数的 LoRA 来说，10% 是**重建级**的改动，不是微调。

温和档 `3e-6 × 20 步` ⇒ 变化约 1/170，配对结果显示"打平、不伤"（+0.020，t=0.94）。
⇒ **当前完整轮用的 `1e-4 × 150` 是偏大的**，这也是它大概率会退化的原因。

### 3.3 【严重】"锚定"在最常用的配置里其实是**失效**的

- `trust_center=prev` 时，`v3_step.py:136` 会把 `trust_eps` **强制置 0**
  ⇒ token 锚定**根本不存在**；
- `act_eps=0.6` 时，实测两臂的 `act/rmse` 只有 **0.09–0.13**（center=prev 下是和 π_k 比），
  远小于 0.6 ⇒ **动作 hinge 几乎从不触发**（`loss/act` ≈ 0.011，只占 22%）；
- `lambda_hand=0.2` 反而成了**最大的一项**（`loss/hand_fm` 0.19–0.22，占 ~78%）。

⇒ 第 1/2 轮实际上等于"**用 78% 的梯度训手部 + 22% 训一个不太触发的 hinge + 奖励项主导体干**"。

### 3.4 【严重】奖励项在 B 数据上**基本没有梯度**

`reward_mode=saturating` 的核是 `exp(-e/0.3²)`：

| root 偏差 e | 奖励值 | 梯度 |
|---|---|---|
| 0.1 m | 0.89 | 有 |
| 0.3 m | 0.37 | 有 |
| 0.5 m | 0.062 | 弱 |
| 0.8 m | 8e-4 | ≈0 |
| 1.2 m | 1e-7 | **0** |

而 B 线采到的起点 e_root 常见 **0.5–2 m**（B_r2 六任务 0.36–2.09）。
⇒ **λ_R=0.3 乘上去还是 ~0**，这一项对大部分样本无信息。

### 3.5 【中】零正则化

`weight_decay=0`、`obs_noise=0`、`dr=False`、`push_std=0`。
在 90–239 个起点窗口这么小的数据上，等于**完全没有抗过拟合手段**。

### 3.6 【中】`wbc_lr=1e-6 × 150 步` 可能"解冻了但没动"

文档记录：`wbc_lr=1e-5` **一步**就能把 `act/rmse` 推到 0.18（说明 19M 的 WBC 对 lr 极敏感）。
我用 1e-6 × 150 步 ≈ 1e-5 的 15 步量级——**可能太小**，测出来"解冻无用"会误导。
建议做一次 `wbc_lr ∈ {1e-6, 3e-6, 1e-5}` 的小扫。

### 3.7 【轻】WM 裁判精度与信号同量级

WM v6 的 20 步关节误差 **0.181 rad**，而策略本身的 joint 误差 **0.25–0.35 rad**。
裁判误差 ≈ 被裁判的量，这是 §3.4 之外奖励不可信的第二个原因。

### 3.8 【轻】输出维度浪费

`action_horizon=40`，闭环只用前 20 步（`HA_VLA_EXEC_STEPS=20`）。
不影响正确性，但 head 有一半输出永远用不到。

---

## 4. "我是怎么定的"——诚实交代来源

| 来源 | 参数 |
|---|---|
| **直接抄文档** `HANDOVER_4090_COMMANDS §0/§5.3` | H=20, K=4, batch 4, accum 8, lora 32, lambda_r 0.3, lambda_hand 0.2, act_eps 0.6（head 臂）, trust_center prev, amp bf16, no-dr, no-critic |
| **文档 + 我修** | save_every 50（防丢）、TEE=0（防 tee 死）、`--init-vla`（多轮链式） |
| **我加的，后被证否** | token BC（`--token-bc-weight`）、trust_center=label + token hinge、温和剂量 fm_lr 3e-6 |
| **我一直没动、但这次审计觉得可疑** | **lambda_hand=0.2**、**act_eps=0.6**、**weight_decay=0 / obs_noise=0**、**steps=150** |
| **我拍的** | `wbc_lr=1e-6`（文档只说 1e-5 太大，没给下界） |

**最该承认的一点**：`lambda_hand`、`act_eps`、正则化这三项我**从第 1 轮抄来就没再审过**，
而它们恰好都指向"梯度花在了闭环用不到的地方"。

---

## 5. 建议的修正参数（下一轮就该用）

| 参数 | 现在 | 建议 | 理由 |
|---|---|---|---|
| `lambda_hand` | 0.2 | **0** | 部署 `SM_HAND=0`，手部永远不用；且 LoRA 共享会污染体干（实测权重变化 10.7%） |
| `trust_center` | prev | **label** | prev 会强制关掉 token hinge |
| `act_eps` | 0.6 | **0.2** | 0.6 不触发（实测 0.09–0.13）；0.2 才有梯度 |
| `trust_eps / weight` | 0/0 | **0.03 / 1.0** | 给 token 一个可用的锚 |
| `fm_lr` | 1e-4 | **3e-6 ~ 1e-5** | 1e-4×150 = 10% 权重变化，实测必退 |
| `steps` | 150 | **30–50** | 数据只够 1 遍（90–239 窗口） |
| `weight_decay` | 0 | **1e-4** | 小数据抗过拟合 |
| `obs_noise` | 0 | **0.1** | 同上 |
| `lambda_r` | 0.3 | **0**（先）| 裁判不可信；等 WM 修好再开 |
| `reward_mode` | saturating | **linear** | 大漂移处仍有梯度 |
| B 起点过滤 | 无 | **e_root < 0.5 m** | 避开奖励死区（占 B 数据一大半） |
| 数据量 | 6 集 | **≥30 集** | 现在窗口池 90–239，过不了 3 遍 |

---

## 6. 一句话总结

架构是对的（冻结 backbone + LoRA head + 冻结 WBC + WM 裁判 + 闭环重采），
**问题全在"梯度花在哪"**：手部占 78%（部署不用）、锚定参数让 hinge 不触发、
奖励核在 B 数据的分布上是死的、正则化全 0、而数据只够 1 遍——却被跑了 3–7 遍。

---

## 7. 2026-10-29 追加：按臂 critic 开关与消融评测（`big2` 轮）

### 7.1 新增参数

| 参数 | 默认 | 作用 | 备注 |
|---|---|---|---|
| `CRITIC_ON_MAIN` / `CRITIC_ON_CTRL` | 空（继承 `CRITIC_ON`） | 每臂单独开关 critic | 消融必须同会话配对，一卡一臂 |
| `CRITIC_GRAD_STATE_MAIN` / `_CTRL` | 空（继承 `CRITIC_GRAD_STATE`） | 每臂单独开关"critic 对 s_H 留梯度" | 原来拼在**全局** `EXTRA` 里，关 critic 的臂也会拿到 ⇒ 已改为按臂拼接 |
| `EVAL_MODE` | `pi0` | `pi0` = π₀ 对 本轮；`ablation` = **本轮两臂互比** | `ablation` 不掺 π₀，避免跨会话绝对数不可比 |
| `LABEL_MAIN` / `LABEL_CTRL` | 空（自动由路径推） | 评测标签 | 写 `critic_ON` / `critic_OFF`，杜绝"main/ctrl 被当成策略名" |

### 7.2 `big2` 轮的配置（这轮要回答的问题：变差是不是 critic 造成的）

```bash
ROUND=big2 STEPS=200 BATCH=16 ACCUM=4 B_START_MAX_ERR=0.3 \
  CRITIC_ON=1 CRITIC_SOURCE=imagined CRITIC_GRAD_STATE=1 \
  CRITIC_ON_MAIN=1 CRITIC_ON_CTRL=0 CRITIC_GRAD_STATE_MAIN=1 CRITIC_GRAD_STATE_CTRL=0 \
  EVAL_MODE=ablation LABEL_MAIN="critic_ON" LABEL_CTRL="critic_OFF" \
  bash scripts/s3_pipeline.sh all
```

| 项 | 值 | 依据 |
|---|---|---|
| `STEPS` | **200** | `ckpt200_vs_600` 扫描：200 步 0.271 / 600 步 0.377（配对 +0.107，t=+1.86）⇒ 过训在退化 |
| `B_START_MAX_ERR` | **0.3** | 只训奖励核还有梯度的起点；`big1` 这轮是 `None`（全量喂，e>0.5 处梯度≈0 等于教噪声） |
| `BATCH×ACCUM` | 16×4 = 64 | 显存只由 micro-batch 决定（16 → 36.3 GB/卡）；1024 会 OOM |
| `H` | 5 | WM 精度：H=5 root 1.1 cm，H=20 6.9 cm |

### 7.3 判决规则（先写死，避免事后找理由）

用 `s3_pipeline.sh report` 的配对统计，`dif = ctrl − main = critic_OFF − critic_ON`：

- `mu > 0`（critic_OFF 漂移更大）且 `t ≥ +2`、≥5/6 同向 ⇒ **critic 有用**，方向是"把裁判做得更准"；
- `mu < 0`（critic_OFF 更好）且 `t ≤ −2`、≥5/6 同向 ⇒ **critic 是毒**，方向是"关掉裁判、只走奖励"，
  下一轮把 `STEPS` 提到 300–400 看是否继续改善；
- `|t| < 2` ⇒ 两臂**不可分辨**，而按 §B.2 的分辨率测算这是可以预期的
  （策略与 GT 的真实差只有 +0.0019，低于裁判噪声地板）⇒ 那就**不再在裁判身上投资**，
  转去压采样噪声（`--sample-avg`）和重规划盲区。

### 7.4 已排除的杠杆（别再回头试）

- **WM 混合比例**：B 线权重 0.2 → 0.5，k=20 关节误差 0.141 → 0.142（无变化）；
  `--extra-transitions` 在 `K>1` 时被代码直接跳过（只有一步目标）。
- **WM 步数**：`big1` 刷新 100 步、第 50 步早停（val20 0.2291），相对 base 只 −0.7%（k=20）。
  k=1 0.010 rad → k=20 0.142 rad，是**误差累积**，不是数据不够。

## 8. WM 刷新预算：照抄 mjlab 配方后的**反例**（2026-10-29 实测）

mjlab 线（`scripts/run_j1_rounds.sh` 第 4 步 `pretrain_wm.py`）的刷新预算是
**1000 步 × batch 2048 × lr 1e-4**；我们 S3 是 **100 步 × batch 256 × lr 1e-5，还 50 步早停**
（样本量差约 80 倍）。照抄配方后结果反转：

| ckpt | k=1 关节 | k=10 关节 | **k=20 关节（门口径）** | k=20 速度相关 |
|---|---|---|---|---|
| base（不刷新） | 0.010 | 0.096 | **0.142** | 0.230 |
| S3 100 步 / lr1e-5 | 0.011 | 0.097 | **0.141** | 0.242 |
| B 权重 0.5 | 0.011 | 0.098 | **0.142** | 0.262 |
| **mjlab 预算（1000×2048×1e-4）** | 0.015 | 0.114 | **0.154 ❌ FAIL** | **0.309 ✅** |

内部 `val20`（32 起点子集）从 0.2356 → **0.2066（−12%）**，看着大胜；但
**门用的 20 步关节 RMSE 反而 +9%，判 FAIL**。它用位置精度换了速度相关
（`--vel-lambda 5` 权重大，训久了往速度拟合偏）。

**教训两条：**
1. **内部 val20 不等于门口径**。`finetune_wm_velocity_policy.py` 的 `val20` 是 32 个起点的
   子集指标，`eval_wm_multistep.py` 才是全量 20 步 rollout。优化前者会把后者做坏。
2. **不能照抄 mjlab 的 WM 门口径**。他们的门是 `qd_corr`（单步速度相关），按那个口径这个
   ckpt 是 PASS；但想象训练里 critic 吃的是**整条 rollout**，必须用累积误差当门。

产物：`runs/S1_handover/wm_big1_budget.pt`，日志 `runs/S3_pipeline/wmmix/B_budget.log`。

## 9. 从 mjlab 九步里真正该抄的一项：**critic 门**

`run_j1_rounds.sh` 第 7 步：`fit_critic_real.py` 留出 `v_ev > 0.30` 才给训练加 `--critic`，
否则 `--no-critic`。我们是**反着来的**——`stage_joint` 里 `CRITIC_ON=1` 硬编码，
而离线 critic 实测 v_ev = 0.29，低于他们自己的门。

与已有证据一致：训练 200→600 步，`imagined reward` 平（0.0272→0.0271），
`boot` 从 −0.069 涨到 +0.044 ⇒ 策略在优化 critic 而非奖励。

**待办**：等 `big2`（同会话六种子配对：main=critic 开 / ctrl=critic 关）判决后，
把 critic 门加成 pipeline 独立阶段：

```bash
# 新阶段（plan）critic_fit + critic_gate：v_ev ≤ 0.30 ⇒ 强制 CRITIC_ON=0 并记录原因
PYTHONPATH=. python3 scripts/fit_critic_real.py --transitions <rt_v2_$ROUND.pt> \
    --out runs/wm_real/critic_b_$ROUND.pt --center linear --epochs 12 --device cuda:0
# 从日志抓 `[critic] best epoch=... val v_ev=+0.xxxx`，≤0.30 则 CRITIC_ON=0
```

其余可借鉴项（优先级低）：
- **双分布 WM 门**：他们 sim 分布是硬门、B 线是软门；我们只看 B 留出，漏了"刷新后遗忘 GT"。
  我们实测 k=1 从 0.010 升到 0.011~0.015，正是轻微遗忘的信号。
- **`--real-transitions`（真实 ret 而非想象 GAE）**：`fit_critic_real.py` 我们也有，
  可以给 critic 提供真实 return-to-go。
- **B 线起点按任务自己的库行区间均匀取**（`BLINE_STARTS_MODE=auto`）：他们修正过这个真 bug；
  我们目前是随机取。

## 10. `big3` 轮：判决规则（先写死）

配置：9 阶段全流程、`STEPS=200`、`BATCH=16×ACCUM=4`、`B_START_MAX_ERR=0.3`、
`CRITIC_VEV_MIN=0.30`（预估门判 FAIL ⇒ 两臂都关 critic，纯奖励 `lambda_r=0.3`）、
评测 = π₀ vs 本轮（`EVAL_MODE=pi0`，6 种子配对，两边 `--sample-avg 4`）。

`report` 的 `dif = ctrl − main`；`EVAL_MODE=pi0` 时 `main=base(pi0)`、`ctrl=本轮候选`，
所以 **`dif < 0` = 候选更好**。

| 结果 | 结论 | 下一步 |
|---|---|---|
| `dif ≤ −0.05` 且 `t ≤ −2`、≥5/6 同向 | 关掉裁判后训练**真的**压下了漂移 | 进第 4 轮：把 `STEPS` 加到 400–600，看是否继续改善 |
| `−0.05 < dif < 0` 但 `t > −2` | 方向对但幅度不够分辨 | 不动配方，靠**加轮次+扩池**积累；同时评估 `--sample-avg` 与重规划盲区 |
| `|dif| < 0.02`，`t` 接近 0 | 训练对闭环漂移**没有作用** | 停止在训练信号上投资，转去压采样噪声（`--sample-avg`）与盲区（`EXEC_STEPS`） |
| `dif ≥ +0.02` | 训练有害 | 回退，只用 π₀ + `--sample-avg`，把问题定位到奖励项本身 |

**同时看两件事作为交叉验证（不占额外机时）：**
1. 两臂配置相同 ⇒ `main` vs `ctrl` 的配对差应当 ≈ 0（**零假设对照**）。
   若出现显著非零，说明评测本身有偏，那一轮的判决作废。
2. `critic_gate.json` 的 `allow` 必须是 `false`，且训练 `args.critic=false`（验收器会查）。
   若门意外 `allow=true`（v_ev>0.30），本轮的"纯奖励"前提不成立，判决改按附录 C 的口径解读。

## 11. 勘误（2026-10-29 晚）：上面第 9 节的 critic 门**用错了对象**

第 9 节写的"`v_ev ≤ 0.30 ⇒ 强制 CRITIC_ON=0`"是**误读**了 mjlab 的口径。回看
`run_j1_rounds.sh` 第 7 步，那个门**只对 `CRITIC_SOURCE=real` 生效**：

```bash
if [[ "$CRITIC_SOURCE" == "imagined" && "$CRITIC_MODE" != "off" ]]; then
    CRITIC_ARGS=(--critic --critic-source imagined)   # GAE：不做 prefit、不过门
else
    ... fit_critic_real.py ...   # 这里才是 v_ev>0.30
fi
```

原因（mjlab §10.2）：real 监督结构性不可学（`v_ev` 0.088–0.152），而 imagined/GAE 的
`critic/v_ev` 能到 0.95–0.999。我们实测也是 **0.9946–0.9974**。

**后果**：`big3` 用 real 的 `v_ev=−0.146` 把健康的 imagined critic 误关了，那一轮白跑。
**已修**：`stage_critic` 现在按 `CRITIC_SOURCE` 分支，imagined 直接放行并把
`{"source":"imagined","allow":true,...}` 写进 `critic_gate.json`。

同时按 mjlab 交接文档改了另外三处（详见 `docs/HANDOVER_2GPU_MULTILOOP_20261028.md` 附录 D）：

| 项 | 旧 | 新 | 依据 |
|---|---|---|---|
| `TRAIN_WBC_MAIN/CTRL` | 1（解冻） | **0（冻结）** | §10.1 解冻 800 步 root_pos 4.16 → **89.2 cm**（21×） |
| 评测执行器 | `joint_final.pt` | **冻结的 `WBC_A24000_wmvel_v6.pt`** | 同上；只有显式解冻才用 joint_final |
| `H` | 5 | **8** | §10.3 H=5 太短（远端全靠 critic） |
| root 奖励 | `w=0.5, std=0.3/0.4` | 可覆盖，建议 **1.0 / 0.2 / 0.25** | §1 "root 项原来在 4 cm 处饱和" |

## 12. `big4` 轮：按对标结论重跑（预注册）

**这一轮是附录 D 四处修复后的第一次正式验证。** 与 `big1~big3` 的关键差异只有四项，其余流程不变：

```bash
ROUND=big4 STEPS=200 BATCH=16 ACCUM=4 H=8 B_START_MAX_ERR=0.3 \
  TRAIN_WBC_MAIN=0 TRAIN_WBC_CTRL=0 \
  REWARD_ROOT_WEIGHT=1.0 REWARD_ROOT_POS_STD=0.2 REWARD_ROOT_ORI_STD=0.25 \
  EVAL_MODE=pi0 bash scripts/s3_pipeline.sh all
```

| 项 | big1~big3 | **big4** | 依据 |
|---|---|---|---|
| WBC | 解冻（`train_wbc=True`） | **冻结** | mjlab §10.1：解冻 800 步 root 4.16 → 89.2 cm |
| 评测执行器 | `joint_final.pt` | **冻结的 `WBC_A24000_wmvel_v6.pt`** | 同上 |
| `H` | 5 | **8** | mjlab §10.3：H=5 太短 |
| root 奖励 | `w=0.5, std=0.3/0.4` | **`w=1.0, std=0.2/0.25`** | mjlab §1："root 项原来在 4 cm 处饱和" |
| critic | big1/2 硬开、big3 被误关 | **imagined 放行**（门改成按 source 分支） | mjlab 第 7 步 if 分支 |

冒烟已确认配置真的落地（`runs/S3_loop/smk_fix/train_main/args.json`）：
`train_wbc=False, horizon=8, critic=True(imagined), reward_root_weight=1.0,
reward_root_pos_std=0.2, reward_root_ori_std=0.25`，审计行
`[AUDIT][reward] root 覆盖 {...} ⇒ w_root=1.0/1.0 std=0.2/0.25`。

**先看一个指标就能判真假：`loss/reward_raw` 是否随训练往上走。**

- mjlab R1（800 步）该指标 0.0837 → **0.0914（+9%）**；
- 我们 big1~big3 是 0.188 → 0.182（**−3%，平**），因为核饱和、root 项只占一半权重。

判据（训练端，跑完 200 步即可看）：

| `reward_raw` 走势 | 结论 | 下一步 |
|---|---|---|
| 单调上升 ≥ +5% | 奖励**终于有梯度**，四处修复命中 | 继续加步数/加轮次 |
| 仍平（±2%） | 核仍然饱和 | 换 `REWARD_MODE=linear`（同误差、无死区）单独一轮 |
| 下降 | 奖励被 WM 误差带偏 | 暂停训练，先解决 WM 部署精度（0.736 vs 0.9925） |

闭环判决仍用 `s3_pipeline.sh report` 的 6 种子配对（dif = 本轮 − π₀，负=更好），
并按 §10 同一张表解读；另设一条**零假设对照**：本轮两臂配置完全相同，
`main` vs `ctrl` 的配对差应 ≈ 0，若显著非零说明评测有偏、该轮判决作废。
