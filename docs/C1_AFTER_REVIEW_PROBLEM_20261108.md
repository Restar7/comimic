# C1（按 review 改后的 LoRA16 配方）出现的新问题 + 我们希望达到的物理改善

> 写于 2026-11-08 12:20。上一份 `C_LORA_PROBLEM_STATEMENT_20261108.md` 是"改之前"的问题；
> **这一份是"按 review 改完并跑起来之后"出现的新问题**，以及我们对"物理变化"的目标定义与已有数据。
> 所有数字来自真实日志（`runs/three_track_v7/c1_lora_proj/pilot200/`），出处随文标注。
> 状态口径：`PASS / RUNNING / STOPPED / NOT_RUN`；**只跑了一半的实验不写成结论**。

---

## 0. 一句话结论

按 review 改完的 **C1 = FT8000 全冻结 + 主干(attn1/attn2/ff) LoRA16 + proj_out 也用 LoRA16 + 双重物理 Trust（相对 pristine 的总修正 / 相对 proj-only 的边际）+ 固定 peak 预算**
在 **100/200 步**时：

* **WM 回报真的显著改善了**：ΔJ = **+0.00504 / +0.00385 / +0.00460 / +0.00520**（it=25/50/75/100），
  按 clip 聚类（54 簇）的 95% CI **全部不含 0**；
* **但物理侧几乎没动**：限位违反率 **2.77% → 2.67%（−3.6%）**，
  而**旧配方**（全秩 proj_out、单重 trust）在 650 步时是 **2.45%（−12%）**；
* **根因不是"模型不想改"，是两件事把它按住了**：
  1. 新的总修正 trust **每一步都在起作用**（`rms_total` 15–47 mrad > τ=20 mrad），与 `L_lim` **抢梯度**；
  2. `L_lim` 是**纯 hinge**（只有越限才有梯度），对"正在逼近限位"没有任何预防作用。
* **归因确实反过来了**：token 位移 `lora 0.0029 / proj 0.0005`（主干占 85%），
  这正是 review 想要的方向（旧配方 proj 占 94%）。

---

## 1. C1 的完整设计（如果要复现，只看这一节）

### 1.1 结构 / 可训参数（运行时打印 + `config.json` 双证据）

| 项 | 值 |
|---|---|
| 底座 | FT8000 `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（fp32，全冻结） |
| LoRA 层数 | **218**（216 个 attn1/attn2/ff + **2 个 proj_out**） |
| 可训参数 | **16,891,904** = 主干 **16,777,216** + proj_out **114,688**（占全模型 **1.032%**） |
| rank / alpha / scale | 16 / 32 / **2.0**，dropout 0，bias none |
| `proj_out` 原全秩权重 | **严格冻结**（`proj_out_base_frozen=true`） |
| zero-LoRA 恒等 | `max|Δ| = 0.0`（逐位） |
| 注入配方 | `--c-recipe c1_lora_proj`（`diff_vla_wbc/lora_ft8000_loader.py:C_RECIPES`） |

> `proj_out` 的 114,688 = 16·(3072+1536) + 16·(1024+1536)，与 review 手算**完全一致**。

### 1.2 优化 / 训练

| 项 | 值 |
|---|---|
| 分组 LR | 主干 **3e-5**、proj_out **3e-5**（两个独立 param group，日志分别记录） |
| 调度 | warmup 20 → **cosine 3e-5 → 5e-6**（200 步预算） |
| 优化器 | AdamW(β=(0.9,0.95), wd=1e-4)，**grad clip 1.0** |
| batch | 32（单卡），每步**现场**抽帧 + 现场 4 步可微 FM |
| 数据 | on-the-fly，`n_eligible = 235,280` 起点（只 train clips，防泄漏 fail-closed） |
| 时间轴 | T=40 / K=20 / **H=20** / FM 4 步（`checkpointed` 反传） |
| 步数 | **RUNNING**（100/200 步完成） |
| 速度 | ~14 s/步（三路采样 + 三次 WM rollout） |

### 1.3 损失（这是 review 的核心改动）

```
L = -J20
    + λ_lim · L_lim(q_raw)                                  # λ_lim = 0.0369（Δ 臂标定值，见 §3 问题 P1）
    + λ_total · [ L_same(e_total) + L_peak(e_total, b_j) ]  # λ_total = 0.010
    + λ_lora  · [ L_same(e_lora)  + L_peak(e_lora,  b_j) ]  # λ_lora  = 0.0025
```

* **三路同噪声采样**（同帧、同 flow 初噪、同 `step_idx`）：
  * `tok_full`     = 全开（带图，训练对象）
  * `tok_proj`     = **关主干 LoRA、只留 proj_out LoRA**
  * `tok_pristine` = **全关**（= 原始 FT8000，`max|Δ|=0` 已验证）
* `e_total = q_full − q_pristine`（**总修正**，review 指出的、之前完全没被约束的量）
* `e_lora  = q_full − q_proj`（**LoRA 边际**）
* `L_same(e) = relu(rms(e)/τ − 1)²`，τ = **0.02 rad**；`L_peak(e,b_j) = mean max_{t,j} relu(|e|/b_j − 1)²`
* **固定 peak 预算**：`b_j = clamp(1σ_over(windows,t) q_raw_pristine[:,:,j], 0.01, 0.10)`，
  启动时用 32 个 val 窗口标定一次后**冻结**，存 `peak_budget.json`；实测 **29/29 关节都顶到 0.10 上限**（见 P3）
* 物理口径：`q_raw = q_default + G1_ACTION_SCALE ⊙ a`（standard 序，正确 G1 单位）

### 1.4 评测口径（review 要求的统计显著性）

* 成对同噪声：同一帧同一初噪采 `lora` 与 `pristine` 两次，`ΔJ = J_lora − J_pristine`；
* **按 episode/clip 聚类的 bootstrap CI**（2000 次重采样，96 个 val 窗口 → **54 个 clip 簇**）；
* 固定评测集 `runs/temporal_ac_bptt_v5/eval_set`（256/96/96 窗口），训练只用 train clips；
* pristine 基线：`val J = 1.49194`、`test J = 1.57558`、`vio = 2.77% / 2.59%`、`q_raw_absmax = 4.603`。

---

## 2. 已有结果（真实日志，RUNNING）

### 2.1 val ΔJ（成对同噪声 + 聚类 CI）

| it | ΔJ | 95% CI（54 簇） | Δtoken rms | 限位违反率 | `q_raw_absmax` |
|---|---|---|---|---|---|
| 25 | **+0.00504** | **[+0.00353, +0.00668]** | 0.00286 | 2.69% | 4.607 |
| 50 | +0.00385 | [+0.00244, +0.00540] | 0.00290 | 2.71% | 4.605 |
| 75 | +0.00460 | [+0.00318, +0.00605] | 0.00324 | 2.70% | 4.595 |
| 100 | **+0.00520** | **[+0.00343, +0.00716]** | 0.00324 | **2.67%** | 4.610 |

### 2.2 与旧配方（全秩 proj_out + 单重 trust，lr=1e-5）同 it 对照

| it | 旧配方 ΔJ | 旧配方 vio | **C1 ΔJ** | **C1 vio** |
|---|---|---|---|---|
| 25 | ≈ +0.0020（50 步） | 2.62% | **+0.00504** | 2.69% |
| 50 | +0.00204 | 2.62% | +0.00385 | 2.71% |
| 100 | +0.00094 | 2.73% | **+0.00520** | **2.67%** |

### 2.3 分组归因（每步都记）

| 组 | 可训参数 | grad_norm（mean / p90 / max） | update_norm（典型） | token 位移（典型） |
|---|---|---|---|---|
| `attn_ff` | 16,777,216 | **1.235 / 1.62 / 11.00** | 0.0150–0.0188 | **0.0029–0.0031** |
| `proj_out` | 114,688 | 0.594 / 0.91 / 3.55 | 0.0012–0.0016 | **0.0005** |

⇒ **主干 LoRA 是绝对主导（85% 位移）**，方向与 review 预期一致。

---

## 3. 改完之后**新出现**的问题（这是要评审的重点）

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| **P1** | **物理改善几乎消失** | C1 vio 2.77→**2.67%**（−3.6%）；旧配方 650 步到 **2.45%**（−12%） | 用户明确要"物理变化更大"，当前不满足 |
| **P2** | **trust 项常开，与 L_lim 抢梯度** | `rms_total` 15–47 mrad（>τ=20 mrad）；`L_same_total` 每步都在拉回 pristine | 修正幅度被压住 ⇒ 物理量改不动 |
| **P3** | **peak 预算被 clamp 到上限，等于没有分关节性** | `b_j` 实测 **29/29 = 0.10 rad**（1σ 全部 > 0.10） | 名义是"逐关节固定预算"，实际退化成"全局 0.10"，对最坏关节没有针对性 |
| **P4** | **proj_out 组的杠杆太小** | update 0.0016 vs 主干 0.0188（小 11×）；token 位移 0.0005 | review 期望"输出层低秩也能吃到收益"，现在它几乎不贡献 |
| **P5** | **梯度尖峰 + clip** | `attn_ff` grad_norm max **11.0**（mean 1.24，clip=1.0） | 尖峰步的更新被 clip 强剪，方向信息损失 |
| **P6** | **CI 只覆盖"单次评测内跨 clip"** | 4 个点都显著，但均值在 +0.0039~+0.0052 抖动 | 未覆盖"评测间"方差；不能声称趋势单调 |
| **P7** | **`L_lim` 是纯 hinge** | `L_lim` 只在越限 tick 有梯度（实测 0.6–4.1 波动） | 对"逼近限位"无预防作用 ⇒ 违反率降得慢 |
| **P8** | 训练变慢 | 14 s/步 vs 旧配方 ~8 s/步 | 三路采样 + 三次 rollout 的代价 |

---

## 4. 我们希望的"物理变化"到底指什么（请评审按这个验收）

用户的原话是"**希望物理上的变化多一些、改进大一些**"。我们把它拆成 4 个可测量目标：

| 目标 | 定义 | pristine | 旧配方（650 步） | **C1 目标** |
|---|---|---|---|---|
| **G1 更安全** | 限位违反率（`violation_frac`，27,648 个 job 的 tick 统计） | 2.77% | 2.45% | **< 2.30%** |
| **G2 峰值更低** | 全部 tick×关节的 `|q_raw|` 最大值 + per-joint p99 | 4.603 rad | 4.651 | **< 4.50 rad** |
| **G3 有实质修正** | `rms_total`（相对 pristine 的 same-obs 动作 RMS） | 0 | ~6 mrad | **10–30 mrad**（有动作但没疯） |
| **G4 不牺牲收益** | ΔJ 的 95% CI 下界 | — | +0.0055（均值） | **CI 下界 > 0** |

> 注意 G3：**"变化多一些"≠"越大越好"**——旧配方正是因为 proj_out 高杠杆把 token 推了 3e-2
> （是 C1 的 10 倍），才在 WM 上刷分而在物理上收益有限。我们要的是
> **"在 trust 域内、把修正用在限位/峰值这些真正约束物理的地方"**。

---

## 5. 候选杠杆（我列的，请评审排序/否决）

| 杠杆 | 具体改动 | 预期（对 G1/G2/G3） | 风险 |
|---|---|---|---|
| **A. 提高 λ_lim** | 0.0369 → **0.08~0.10**（其余不动） | G1 ↓↓（直接把梯度投向越限方向） | 低：trust 仍封顶幅度；但 λ 是从 Δ 臂搬来的，**需要按梯度预算重标** |
| **B. 加软限位项 `L_margin`** | `relu(0.15 − min((q−lo)/s, (hi−q)/s))²`，λ≈0.05 | G1/G2 ↓（在**逼近**限位时就给梯度，比 hinge 早 15% 量程） | 低：只加一个 hinge；但要防"过度保守"（G3 掉到 0） |
| **C. 放宽 τ** | 0.02 → 0.025/0.03 | G3 ↑（允许更大修正）、G1 可能 ↓ | 中：削弱 review 的信任域保证 |
| **D. 收紧 `b_j`** | 0.10 → **0.05**（或按 per-joint 分位） | G2 ↓（专治最坏关节） | 低；但 P3 的"分关节性"还没解决 |
| **E. 增加步数** | 200 → 400/800 | G1 需要时间（旧配方 600 步才到 2.45%） | 时间（~15 s/步） |
| **F. proj_out LR** | 3e-5 → 6e-5 | G3 ↑（输出层更能动） | **中高**：高杠杆层最容易骗 WM，review 明确警告过 |

---

## 6. 想问评审的 5 个具体问题

1. **λ_lim 该不该按 C1 的梯度预算重新反解？** 现在它沿用 Δ 臂的 0.0369（当时还没有总修正 trust）。
   我们已有的工具：`scripts/calibrate_physical_bptt_weights.py --actor lora`（能算 `‖g_Llim‖/‖g_J‖` 的比）。
   是否应该固定 `λ_lim` 使得 `‖g_Llim‖ ≈ 0.5·‖g_J‖`（与 Δ 臂同口径），然后**其它项不动**？
2. **固定 peak 预算 `b_j` 应该怎么标定？** 我们现在用 `1σ(q_raw_pristine)`，结果全被 clamp 到 0.10，
   等于没有分关节性。更好的选择是不是：`b_j = c·G1_ACTION_SCALE_j`（物理 scale 比例）、
   或 `b_j` = 旧配方在 200 步时 `|e|` 的 p90（数据驱动），或"限位裕度的 30%"？
3. **软限位 `L_margin` 的形式与权重**：`relu(κ − dist_norm)²` 的 κ（0.10/0.15/0.20）怎么选？
   会不会把策略推成"保守不动"（G3→0）？是否需要与 `L_same` 做互斥/权衡？
4. **proj_out 组要不要单独调 LR？** 现在它 115k 参数、更新比主干小 11×、token 位移 0.0005。
   是"给它更高 LR"，还是"干脆冻结（C0 对照）"更能回答"低秩输出适配有没有用"？
5. **验收门槛**：G1–G4 四个目标里，是否应该"物理优先"（例如 **G1 必须比 pristine 低 ≥8%** 才允许
   声称改进，ΔJ 只作为并列报告）？还是应该用 `ΔJ / Δvio` 的帕累托前沿来选点？

---

## 7. 复现入口

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
source scripts/harness03_env.sh

# C1 pilot（当前正在跑；配方 = c1_lora_proj + 双重 trust + 固定 peak 预算）
python3 scripts/train_sonic_lora16_ft8000_bptt.py \
  --out runs/three_track_v7/c1_lora_proj/pilot200 --updates 200 \
  --c-recipe c1_lora_proj --lora-lr 3e-5 --proj-lr 3e-5 \
  --lr 3e-5 --lr-total-updates 200 --min-lr 5e-6 --warmup 20 --grad-clip 1.0 --seed 0 \
  --trust-dual 1 --lambda-trust-total 0.010 --lambda-trust-lora 0.0025 \
  --lambda-peak-total 8.4e-4 --lambda-peak-lora 2.1e-4 --tau-same 0.02 \
  --ft8000-checkpoint $COMIMIC_MAIN/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --vlm-path $COMIMIC_MAIN/groot_repro/nvidia/Cosmos-Reason2-2B-weights \
  --limits-npz runs/three_track_v6/physical/limits_standard29.npz \
  --physical-weights runs/three_track_v6/calibration/lora16_weights.json \
  --eval-set runs/temporal_ac_bptt_v5/eval_set --device cuda:0 \
  --batch 32 --vla-batch 32 --fm-grad-backend checkpointed --budget-windows 32 \
  --eval-every 25 --eval-batch 16 --cond-check-every 25 --save-every 50 --log-every 5
```

**产物**

| 内容 | 路径 |
|---|---|
| C1 训练日志（含 per-group 梯度/更新/token 位移） | `runs/three_track_v7/c1_lora_proj/pilot200/train.jsonl` |
| C1 val（含聚类 CI） | `.../val.jsonl` |
| 参数合同（218 层 / 16,891,904 / proj 冻结） | `.../config.json` 的 `lora` 块 |
| 固定 peak 预算 | `.../peak_budget.json` |
| 旧配方对照 | `runs/three_track_v6/lora16_ft8000/train800/`、`.../train400_lr1e-4_ddp/` |
| 改动前的完整问题陈述 | `docs/C_LORA_PROBLEM_STATEMENT_20261108.md` |

---

## 8. 边界（别踩）

* 本文所有 C1 数字是 **200 步 pilot 的 4 个评测点**，`RUNNING`；**不能**当成收敛结论；
* **没有做闭环成功率**（C 线的 Isaac 闭环尚未跑；A 线做过 18 集 = 9/18）；
* `ΔJ` 是**冻结 WM 的想象回报**，不是真机成功率；
* 旧实验目录 `runs/three_track_v6/`、`runs/a_prime_sonic_exc/` 只读；新实验只写 `runs/three_track_v7/`；
* 任何"LoRA 有效"的表述必须区分：**pristine 对比** / **LoRA 边际**（同 proj）/ **token 位移量**。

---

## 9. 附：WM 版本核对（review 第 3 项，已查清）

review 指出"新问题文件写 `wm_gain2.pt` 的 SHA 前缀 `fc5aeb30…`，旧交接材料记录 `c74060f1d3706f63`"。
我们做了完整核对：

| 口径 | 值 |
|---|---|
| `runs/a_prime_sonic_exc/wm_gain2.pt` **文件级** sha256 | **`fc5aeb30e987f016d24a8392af9d0fecb3151e07dc33d49b71f0544f9621ca97`** |
| 同一文件 **权重级** sha256（按 key 排序 + float32 字节） | `f44abf2292c16450ba83027883b92b51…` |
| 文件大小 / mtime | 35,300,161 B / **2026-11-05 15:54** |
| `c74060f1d3706f63` 出现位置 | 只在 `comimic/docs/{DELTA_TOKEN_WORKLOG,REVIEW_HANDOFF,REVIEW_HANDOFF_V2}_20261107.md`，且都标"生成 38,400 条标签用的 WM" |
| 仓库里是否存在 `c74060f1…` 的文件 | **否**（`find` + 全量 sha 搜索均无） |

**结论**：`c74060f1…` 是**同一路径的旧版本**（在 2026-11-05 15:54 重新导出之前）的哈希，属于"记录过期"，
不是"实验加载错版本"。证据：A/B/C 三条线都通过
`runs/bptt1/args.json → wm_ckpt = runs/a_prime_sonic_exc/wm_gain2.pt` 加载（`diag_bptt_learnability.load_env`），
而该文件现在的 sha 就是 `fc5aeb30…`；V7 的 Phase 0 冻结清单（`runs/three_track_v7/phase0_freeze/inventory.json`）
与版本存储 `wm_v000` 也都用这个 sha 作为 anchor。

> 后续所有"换 WM"的动作都在 `runs/three_track_v7/wm_versions/` 里走版本号（`wm_v000` → `wm_v001_candidate`），
> 不再以文件路径当版本标识。
