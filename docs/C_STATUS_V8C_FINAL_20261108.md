# C 线当前状态与结论（V8-C + P2/P4 完成版）

> 写于 2026-11-08 17:50。**这份文档是"到目前为止 C 线（LoRA16）到底得到了什么"的完整交接**，
> 给外部 reviewer 看。所有数字来自真实日志；未做的实验明确标 `NOT_RUN`。
> 只读三行：**① 在固定 WM0（wm_gain2）下，C 的独立 test ΔJ = +0.01640（CI [+0.0122,+0.0204]），是旧 C1 的 2.9 倍；
> ② 物理越限 2.59% → 2.53%，四个主要关节都在降但已到平台（原始 FT8000 自己就在这四个关节上越限 25%/22%/13%/10%）；
> ③ 闭环 18 对**没做**，所以"真机是否更好"仍然未知。**

---

## 1. 一句话结论（三条，互不替代）

| 结论 | 判定 | 依据 |
|---|---|---|
| **OFFLINE_GAIN**（WM0 上的想象回报） | **PASS** | test ΔJ **+0.01640**，CI **[+0.01217, +0.02035]**（55 个 clip 簇）；旧 C1 同 WM0 是 +0.00568 |
| **PHYSICAL_SAFETY**（越限） | **部分改善 / 未达目标** | 2.59% → **2.53%**（目标 <2.45%，挑战 2.30%）；j4/j10 平台化，**原始模型自己就有 25.78%/22.55%** |
| **CLOSED_LOOP_GAIN**（真机成功率） | **NOT_RUN** | V8-C §8 的 18 对首次尝试闭环**尚未执行** |

---

## 2. 现在到底是什么配置（可复现）

```
Frozen FT8000 (1,637,293,184, fp32)
  + 218 层 LoRA16：216 个 attn1/attn2/ff + 2 个 proj_out（原全秩权重冻结）
    可训 16,891,904 = 主干 16,777,216 + proj_out 114,688（1.032%）
  + 4 步 Euler Flow Matching（可微）
  → token [B,40,64] → Frozen SONIC → q_raw = q_default + G1_ACTION_SCALE ⊙ a
  → **Frozen WM0 = runs/a_prime_sonic_exc/wm_gain2.pt（sha256 fc5aeb30e987f016…）**
  → H=20 想象 rollout → J20（原 saturating 指数核，未改）
```

**损失（V8-C 全套 + P2/P4）**

```
L = −J20(原)
    + λ_R      · L_recovery(Worst-Window 聚合)     λ_R = 2.223e-3
    + λ_lim    · L_abs(q_raw)                       λ_lim = 0.03068
    + λ_added  · L_added(v_full − v_pristine)       λ_added = 6.78179
    + λ_margin · L_margin(κ=0.05)                   λ_margin = 0.2201
    + λ_peak   · L_peak(e_total, b_j)               λ_peak = 1.8e-5
    + λ_trust  · L_same(e_total, τ=0.03)            λ_trust = 0.01（校准点 INACTIVE，回退旧值）
```

**关键参数（全部 Train-only 标定后冻结）**

| 项 | 值 |
|---|---|
| Recovery `z0`（Train Q70） | root_pos **0.0992** / root_ori **0.2178** / linvel **0.2428** / angvel **0.3046** |
| Recovery 聚合 | **worst_window**：窗口内最差 25% tick + batch 最差 10% 窗口（V8.1 P2） |
| Peak 预算 `b_j` | **mean 0.0486、range [0.020, 0.120]、21 个不同值**（j4 给到上限 0.12） |
| 训练采样 | 70% 普通 + **30% 困难**（root/heading/vel 误差 top-20%）+ 支持域门禁（阈值 15.31，通过 **99.2%**） |
| LR / batch / steps | 3e-5（两组）→ cosine 5e-6；batch 32；warmup 20 |
| 训练时长 | 280 步（停在平台，见 §4），**7.7–8.0 s/步** |

---

## 3. 结果

### 3.1 训练曲线（WM0 固定，val 96 窗口，成对同噪声 + 聚类 CI）

| it | ΔJ | 95% CI（54 簇） | 越限率 | Δtoken |
|---|---|---|---|---|
| 50 | +0.01194 | [+0.00916, +0.01503] | 2.67% | 0.0067 |
| **100** | **+0.01727** | [+0.01280, +0.02221] | **2.56%** | 0.0091 |
| 150 | +0.01470 | [+0.01094, +0.01867] | 2.66% | 0.0097 |
| 200 | +0.01669 | [+0.01255, +0.02084] | 2.61% | 0.0100 |
| **250** | **+0.01748**（Best） | [+0.01349, +0.02147] | 2.62% | 0.0100 |

### 3.2 独立 Test（一次，Best=it=250）

| 指标 | 值 |
|---|---|
| **ΔJ** | **+0.01640**，CI **[+0.01217, +0.02035]**（55 簇） |
| J / J_base | 1.59213 / 1.57573 |
| 越限率 | **2.53%**（pristine 同 split 2.59%） |
| `q_raw_absmax` | 3.843（pristine 3.580） |
| 最差关节 | **j4 24.95% / j10 21.88% / j5 11.82% / j11 8.44%** |

### 3.3 四路归因（同一个 checkpoint，无梯度）

| 组合 | Test J20 | token 位移 rms |
|---|---|---|
| pristine（全关 LoRA） | 1.5706 | — |
| **仅主干 LoRA** | **1.5862** | **0.0083（80%）** |
| 仅 proj_out LoRA | 1.5732 | 0.0027（20%） |
| 全开 | **1.5879** | 0.0104 |

⇒ **主干 LoRA 主导（80%）**；旧 C1 是 proj_out 占 87%。这是 review 想要的结构性反转。

### 3.4 逐关节：为什么"平台化"

| 关节 | **pristine（原始 FT8000）** | **训练后** | 相对改善 |
|---|---|---|---|
| j4 | **25.78%** | 24.95% | −3.2% |
| j10 | **22.55%** | 21.88% | −3.0% |
| j5 | 13.33% | 11.82% | −11.3% |
| j11 | 9.64% | 8.44% | −12.4% |
| 全局（tick×关节） | 2.59% | **2.53%** | −2.3% |

**结论：j4/j10 的越限主要来自"参考动作行程 vs 限位定义"，原始模型自己就在越限；LoRA 只能小幅改善，训练再多也下不去。**
（两个百分比不冲突：2.53% = 所有 (tick×关节) 组合的越限比例；24.95% = j4 单个关节在多少 tick 里越限。）

---

## 4. 为什么停在 it=280（平台化证据）

从 it=100 起 5 个评测点：

| 指标 | 100 | 150 | 200 | 250 | 趋势 |
|---|---|---|---|---|---|
| ΔJ | +0.01727 | +0.01470 | +0.01669 | +0.01748 | **无**（+0.0165 ± 0.0014） |
| 越限率 | 2.56% | 2.66% | 2.61% | 2.62% | **无**（2.61% ± 0.05） |
| j4 | 22.6% | 23.1% | 22.8% | 22.8% | **无**（±0.3pp） |
| j10 | 19.9% | 20.4% | 20.3% | 20.5% | **无** |

LR 此时已从 3e-5 余弦衰减到 ~1.24e-5，剩余 120 步只是低 LR 精修 ⇒ 主动停在 280（`stopped.json` 记录原因）。

---

## 5. 与历史 run 的对照（**口径不同，不能拼成一条曲线**）

| Run | WM | Steps | val ΔJ | test ΔJ | test 越限 | 备注 |
|---|---|---|---|---|---|---|
| 旧 C1 pilot200 | WM0 | 200 | +0.00607 | +0.00568 | 2.60% | proj_out **全秩**、单重 trust、无 recovery |
| V8-C（mean 聚合） | **WM1** | 400 | +0.01838 | +0.01796 | 2.60% | 换了 WM，不能与 WM0 混比 |
| V8-C（mean、无困难采样） | WM0 | 190 | +0.01631 | — | 2.57% | 中途停 |
| **V8-C + P2/P4（本条）** | **WM0** | **280** | **+0.01748** | **+0.01640** | **2.53%** | 本报告的最终结果 |

**关键推论**：WM1 那条的 test +0.01796 vs 本条 +0.01640 ⇒ **WM1 的额外收益只有 +9.5%**，
也就是说"C 的收益**不依赖**换 WM"；但反过来也说明 **WM refresh 单独带来的增量有限**（见 §7）。

---

## 6. 还没做的（V8-C 要求的交付缺口）

| # | 项 | 状态 | 说明 |
|---|---|---|---|
| 1 | **18 对真实 SONIC–Isaac 闭环**（旧 C1 vs V8-C） | **NOT_RUN** | 唯一能回答"真机是否更好"的实验 |
| 2 | 每 tick 物理落盘（q_raw/q_applied/q_real/qd_real/limit/task_stage） | **NOT_RUN** | 闭环诊断需要 |
| 3 | 固定窗口尾部诊断（z_p50/p95/p99 跨 checkpoint 对比） | **NOT_RUN**（C 版） | A/B 侧已有 `scripts/diag_v81_fixed_eval.py`，C 需要 LoRA 适配层 |
| 4 | 严格 resume 的首次真实验证 | **未验证** | 代码已接线（含抽帧指纹校验），尚未真正 resume 过 |
| 5 | `L_rate` | **BLOCKED** | RobotSpec 无 velocity 字段，审计未做 |

---

## 7. 关于"后续是否只能靠 WM refresh"——现有证据

| 证据 | 数字 | 含义 |
|---|---|---|
| WM1 门禁（新 on-policy 数据 20 步 root 误差） | **31.80 cm → 10.97 cm（−65.5%）** | 刷新对"新策略分布"的预测确实大幅改善 |
| WM1 对 Actor 的最终收益 | test +0.01640 → +0.01796（**+9.5%**） | **预测大幅改善 ⇒ 收益只涨 9.5%** |
| WM1 在旧 A′ held-out 上 | 4.50 → 4.48 cm（+0.45%） | 无遗忘 |
| 梯度方向探针 `dir_agree` | **未跑**（发布为 override） | 无法证明"梯度方向也变好了" |
| 本条（WM0）物理 | 2.53%（历史最低） | 换 WM 那条反而是 2.60% |

**判断**：
* **WM refresh 是有效杠杆，但按目前证据只值 ~10%**，不是"再涨一倍"；
* 更可能的下一步瓶颈是 **① 闭环评测缺失**（我们还不知道真机失败在哪：漂移？站桩？抓取？）、
  **② j4/j10 的限位/参考动作定义**（结构性，占 75% 的越限）、
  **③ 单 seed 方差**（所有结论都来自 seed0）；
* 若要靠 WM refresh 拿更大收益，前置条件是**先把 `dir_agree` 梯度探针补上**，
  否则"预测更准"和"梯度更有用"仍是两件事（V7/V8 文档都反复强调这点）。

---

## 8. 关键路径与复核命令

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
source scripts/harness03_env.sh

# 本条 run（最终结果）
runs/three_track_v8/c_tracking_recovery/actor_wm0_p2p4/
  ├ train.log / train.jsonl / val.jsonl / cond_check.jsonl
  ├ attribution.jsonl / physical_report.jsonl / stopped.json
  ├ checkpoint_lora_best.pt（it=250）/ checkpoint_lora_0{050,100,150,200,250}.pt
  ├ peak_budget_self_q99.json
  └ test_best/test.json          ← 独立 test（本报告 §3.2/3.3 来源）

# 标定
runs/three_track_v8/c_tracking_recovery/calibration/
  ├ physical_weights_v8c.json（λ 全部）
  ├ recovery_z0.json（Train Q70）
  └ peak_budget_self_q99.json（逐关节 21 值）

# 复核
cat runs/three_track_v8/c_tracking_recovery/actor_wm0_p2p4/val.jsonl
python3 -c "import json;print(json.load(open('runs/three_track_v8/c_tracking_recovery/actor_wm0_p2p4/test_best/test.json'))['results']['test']['dJ'])"
python3 -m pytest tests/test_tracking_recovery_tail.py tests/test_support_aware_sampler.py \
  tests/test_physical_bptt_losses.py tests/test_lora16_dummy.py -q   # 28 passed
```

**WM 版本（必须记）**

| 名称 | sha256(前16) | 角色 |
|---|---|---|
| `wm_v000` / `wm_gain2.pt` | `fc5aeb30e987f016` | **本条 run 使用的 WM** |
| `wm_c1_v001_published` | `81d24cafbd4b0953` | 之前那条 400 步用的 WM1（门禁 override 发布） |

---

## 9. 给 reviewer 的 5 个问题

1. **物理还能不能继续压？** j4/j10 在 pristine 就有 25.78%/22.55%，我们只降到 24.95%/21.88%。
   该认账（"结构性，训练到顶"），还是应该去查**限位定义/参考动作**（那属于数据/配置层，不是训练层）？
2. **闭环优先还是继续刷离线？** 离线已 +0.0164（2.9×旧 C1），但 18 对闭环没做。
   是否应该**先花 3–6 小时做闭环**，再决定还要不要动模型？
3. **WM refresh 的正确期望值**：新数据预测 −65.5% 只换来 +9.5% 的 Actor 收益。
   这个转化率是"正常"（说明预测已不是瓶颈），还是说明**WM 的梯度方向才是瓶颈**（需要 `dir_agree` 探针）？
4. **单 seed 能不能下结论？** 所有数字都是 seed0。要不要在闭环前先跑一个 seed1 的 280 步（约 40 分钟）看方差？
5. **`λ_added=6.78` 量级是否合理？** 校准反解出来的（因为该项梯度极小），但实际 `Ladd≈0.001` ⇒ 基本不生效。
   是否应该直接关掉（记为 INACTIVE），而不是留一个"看起来有、其实没有"的项？

---

## 10. 边界

* 本条所有数字都是 **WM0 口径**；与 WM1 那条（+0.0180）**不能混比**；
* `ΔJ` 是**冻结 WM 的想象回报**，不是真机成功率；
* 物理越限是"归一化超限量 > 0"的比例，两个百分比口径在 §3.4 已写明；
* 旧目录 `runs/three_track_v6/`、`runs/a_prime_sonic_exc/` 只读；
* 本轮**没有**做闭环、没有做多 seed、没有做单变量消融。
