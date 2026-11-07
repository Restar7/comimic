# v5 主线训练：现状 / 问题 / 结论（交给审阅者迭代用）

> 生成时间：2026-11-07 22:05（训练仍在跑，见 §4；**如果要修 cap，应当先停掉当前两个 run**）。
> 执行环境：worktree `/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03`
> （分支 `harness/03-bptt-diagnosis`，基线 `a4bd1ff`，改动已提交，主仓库 main **未动**）。
> 规范：`docs/TEMPORAL_AC_BPTT_V5_MAINLINE_SPEC.md`（同目录）。
> **本文是自包含的**：所有数字都是本机实测，标了「实测」；推断/假设标「推断」。

---

## 0. 给审阅者：一句话问题

**代码结构、梯度管线、部署契约、数据口径都已验证正确；但训练出来的 Actor 坍缩成了
"常数 Δ"——chunk-attention 上下文、History-GRU 历史、state64 三个输入在训练后完全不起作用
（打乱输入后输出几乎不变，cos≈0.9999）。根因是 `apply_norm_cap` 的 tanh 在训练早期
完全饱和（未 cap 的 `‖u‖≈6.6` vs `cap=0.08`，82×），把输入路径的梯度压小 ~80×，
优化器只能学最后一层的常数偏置。**

需要审阅者判断/迭代的是：**根因是否成立？修 cap 参数化的最佳方案？顺带的损失权重尺度、
梯度噪声、以及"防坍缩"验收门禁怎么设计？**

---

## 1. 背景与目标（规范摘要）

要训的东西：给冻结 VLA 输出的 **40 帧 motion token chunk** 加一个**逐帧**修正
`Δ_t (t=0..T−1, T=40)`，使**冻结世界模型（WM）**里滚 H=20 步的折扣回报 `J20` 变大，
同时动作变化别太大。

- 时间轴：`T=40`（一次 VLA chunk）、`K=20`（客户端实际执行 `HA_VLA_EXEC_STEPS=20`）、
  `H=20`（可微 WM 步数）、`ℓ(k)=k`；**`t≥K` 的 Δ 恒 0**（部署侧同样只在 `t<K` 加）。
- Actor 结构（规范 §5，已实现）：Chunk-Attention(1 block, d128, 4 head, FFN256, pre-LN)
  + HistoryGRU(58→128) + 位置 PE(40×128 / 40×16)
  + MLP `400→1024→1024→64`（SiLU），末层 `N(0,4e-4)`、bias=0，共 **1,744,832** 参数。
- 损失（规范 §7.1，锁定）：

```
L = −mean(J20)
    + 0.010 · L_act        # 29D rad 动作 trust：relu(RMS(e)/τ − 1)²，τ=0.020 rad
    + 0.002 · L_smooth
    + 0.001 · L_anchor     # ‖Δ‖²/cap²
```

- 优化（锁定）：AdamW，head lr 3e-4 / enc 1e-4，wd 1e-4（norm/bias 免衰减），
  grad clip 1.0，50 步 warmup + cosine 到 head 3e-5 / enc 1e-5。
- 预算（锁定）：updates=1000（上限 1200）、batch=256、val 每 50 步、早停=连续 6 次无改善、seeds 0/1。
- 数据（锁定）：**on-the-fly**——每个 optimizer step 在 338,402 个合格帧上
  `torch.multinomial(w, 256)` 抽起点 → 现场 burn-in(10 步, 扰动 5cm/0.1rad/0.05rad/0.3rad/s)
  → **现场跑冻结 VLA 采样新 token**（fm_steps=4，每次新流初噪）。
  （规范原本写 346,729 个合格帧 = 只按漂移筛；叠加 §3.4 的 `f≥10`、`f+H≤clip_end`
  两条窗口约束后是 338,402，`uniform_over_eligible=true`。）
- Critic：**关闭**（`K==H==20` ⇒ 尾部不可识别，日志写 `BOOTSTRAP_NOT_IDENTIFIABLE`）。
- 评测：固定 held-out 集 = 448 窗口（256 train / 96 val / 96 test，**uniform** 抽帧，
  val/test 各自 81 个独立 clip），每 50 步评 val；训练结束评 test。

---

## 2. 实现现状（关键代码）

### 2.1 Actor 的输入拼接（**这是问题 1 的现场**）

`diff_vla_wbc/delta_token_temporal_v5.py::forward`：

```python
hz   = self.token_proj(Z) + self.pe_chunk[:T][None]        # Z (B,T,64) -> (B,T,128)
c_t  = self.chunk_norm(self.chunk_encoder(hz))              # (B,T,128) 整段 attention 上下文
_,h_n= self.hist_gru(hist.float())                          # hist (B,10,58) -> (1,B,128)
parts = [Z, c_t, h_n[-1][:,None,:].expand(B,T,-1),
         state64[:,None,:].expand(B,T,-1), self.pe_pos[:T][None].expand(B,-1,-1)]
x = torch.cat(parts, -1)                                    # (B,T,400)
u = self.head(x)                                            # (B,T,64) 未 cap
d = apply_norm_cap(u, cap)                                  # 软上限（tanh）
d = d * mask[..., None]                                     # t>=K 严格 0
```

`apply_norm_cap`（`diff_vla_wbc/delta_token.py:38`，**训练与部署共用**）：

```python
def apply_norm_cap(delta, cap):
    n = delta.norm(dim=-1, keepdim=True) + 1e-9
    return delta * (torch.tanh(n / cap) * cap / n)
```

### 2.2 训练一步（`scripts/train_temporal_ac_bptt_v5.py`）

```python
batch = src.sample(256)                       # 现场抽帧 + burn-in + 现场 VLA
tokens = batch["tokens"].detach()             # Z (B,40,64) 冻结
delta  = actor(tokens, batch["state64"], batch["hist10"], mask, cap=cap_at(it))
with torch.no_grad():                         # Δ=0 的同起点参照（每步现算）
    q_base = q_target(rollout(tokens).actions)
J, ro = rollout(tokens + delta, H=20)         # 可微 WM + SONIC decoder + reward
e = q_target(ro.actions) - q_base             # 29D rad
L_act = mean_b relu(RMS_b(e)/0.020 - 1)^2     # 逐样本 hinge 再平均
L_sm, L_anc = ...
loss = -J.mean() + 0.010*L_act + 0.002*L_sm + 0.001*L_anc
loss.backward(); clip_grad_norm_(actor, 1.0); opt.step()
```

> 全流程的 `Δ → J` 是**真 BPTT**：梯度穿过 20 步 `decoder(obs994) → action → WM(step) → reward`
> 回到 Δ，再过 Actor 的各层（decoder/WM 参数冻结，但计算图保留）。

### 2.3 已实现并单测的其余部分

- 部署契约 `diff_vla_wbc/delta_token_io_v5.py`：`T/K/H`、per-frame Δ、`t≥K` 恒 0、
  `hist10 (10,58)`、**关节序换序**（训练 `standard` ⇄ 部署 `ha_v31`）、fail-closed；
- Critic 代码路径 `imagined_value_v5.py`（TDλ + EMA target + Value Gate，默认关闭）；
- 29D rad 动作预算 `action_trust_v5.py`；
- 6 个 v5 单测 + v4 契约测试：`70 passed / 14 skipped`；
- 评测集构建：`diag_bptt_learnability.py build --uniform-frames 1`。

---

## 3. 已验证正确的部分（不是问题所在，先排除）

### 3.1 数据/历史口径

- 起点合格帧：漂移筛选 **346,729**（与规范 §0-A 逐位一致）→ 叠加窗口约束 **338,402**；
- `history10` 强口径对拍（部署序 HA_V31 → 训练序 standard，两条部署路径）：
  `max|Δ| = 0.0`；`state` 与 hist 的 q 块偏移跨窗口恒定 `5.96e-8`；
- 关节序结论（实测 + 4 处代码证据）：**训练侧 = `standard`（MJCF/MDP）、部署 `state64` = `ha_v31`**；
  契约层换序后两条路径都逐位一致（这条修正了规范 §1.3 的绝对序表述）。

### 3.2 梯度管线（**不是梯度断了/符号反了**）

参数级中心差分（`J.mean()` 对 Actor 参数；取每个参数张量里梯度最大的元素）：

| 参数 | analytic | FD (h=0.03) | rel_err |
|---|---|---|---|
| `head.0.weight` | −5.194e-4 | −5.205e-4 | 2.3e-3 |
| `head.4.weight` | +6.826e-3 | +6.825e-3 | 2.2e-4 |
| `token_proj.weight` | −2.597e-4 | −2.583e-4 | 5.5e-3 |

（h=0.1 / 0.01 时也在 1e-2 内；只有取到 `|g|~1e-6` 的极小元素才会被 float32 舍入淹没。）

### 3.3 部署契约（真 checkpoint 自检通过）

`scripts/verify_v5_deploy_contract.py --ckpt seed0/checkpoint_actor_best.pt`：

```
params=1,744,832  spec: T/K/H=40/20/20 cap=0.08 out_mode=per_frame
state_order=ha_v31 hist_order=ha_v31 hist_clock=control_tick_50hz
Δ deploy-order vs 直连训练序: max|diff|=0.000e+00
t≥K 非零 ⇒ 正确抛错：DeltaTokenContractError
PASS：‖Δ‖max=0.0712 ≤ cap=0.08，t≥K 恒 0，换序一致
```

### 3.4 Critic / 早停 / 产物

- Critic 按规范关闭，`critic_gate.json` 写 `BLOCKED_BOOTSTRAP_NOT_IDENTIFIABLE`；
- 每个 run 已产出 `config.json / split_manifest.json / windows_meta.json / vla_token_manifest.json /
  history_contract.json / history_parity.json / train.jsonl / val.jsonl / critic_legacy_audit.md /
  checkpoint_actor_best.pt(.sha256) / deploy_contract_check.txt`。

---

## 4. 训练现状（2026-11-07 21:58）

| 项 | seed0 | seed1 |
|---|---|---|
| 进度 | ~it 420 / 1000 | ~it 415 / 1000 |
| 速度 | 14.7 s/step | 15.0 s/step |
| GPU | 27.9 GB / 49 GB，~100% | 27.9 GB / 49 GB，~99% |
| 预计结束 | ≈00:30 | ≈00:30 |

**held-out val ΔJ（每 50 步，96 窗口，base J=1.50594）**：

| it | seed0 | seed1 |
|---|---|---|
| 50 | +0.00041 | −0.00014 |
| 100 | +0.00018 | −0.00026 |
| 150 | −0.00013 | −0.00000 |
| **200** | **+0.00101** | −0.00003 |
| 250 | +0.00020 | −0.00009 |
| **300** | +0.00063 | **+0.00037** |
| 350 | +0.00037 | −0.00005 |

- val 的统计不确定度（实测，逐窗口 ΔJ std=0.0027 / √96）：**SE = 2.75e-4** ⇒
  seed0 的 +0.00101 是 **3.7σ**；改善窗口占比 70.8%；seed1 的 +0.00037 只有 1.3σ。
- 训练曲线的 batch-mean J 逐窗口 std=0.237、B=256 的 SE≈0.0148 ⇒ **训练日志看不出 0.001 级改善**。

**动作预算（it=50 的 checkpoint，val 96 窗口）**：

| 量 | 值 |
|---|---|
| 逐控制步 RMS | t=0..4: 19.2/23.0/24.6/24.3/27.2 mrad；t=19: 80.6 mrad |
| 全局 peak | 2.535 rad @ t=19, left_ankle_pitch |
| 两个 rollout 的终点状态差 | p50 2.8 mm / p90 5.7 mm / max 22 mm |

⇒ 晚期大峰**不是状态发散**，而是 decoder 对 token 微小差异的**相位敏感**（踝关节早/晚摆切换）；
前 5 步（部署最关心）RMS 19–27 mrad。

---

## 5. ★核心问题：Actor 坍缩成"常数 Δ"，输入实际未被使用

对 `seed0/checkpoint_actor_best.pt`（it=200）在 val 32 窗口上做**输入打乱/清零**
（同一 batch，只把某个输入沿 batch 维 shuffle）：

| 扰动 | cos(Δ_扰动, Δ_原) | max\|Δdiff\| |
|---|---|---|
| shuffle `Z`（整段 40 帧 chunk） | **0.999935** | 7.2e-4 |
| shuffle `state64` | **0.999711** | 1.6e-3 |
| shuffle `hist10`（GRU 输入） | **0.999996** | 1.5e-4 |
| 全部输入置 0 | **0.999805** | 1.7e-3 |

- batch 内 Δ 的逐元素 std = **8.98e-5**；
  注意：`‖Δ‖` 对**所有 t** 平均是 0.04（因为 t≥K 的 20 帧恒为 0），
  换算到 `t<K` 是 **≈0.08 = cap**（与训练日志 `|Δ|=0.0800` 一致）；
- 同一窗口内**帧间** Δ 差异：max 1.6e-3 / mean 5.8e-5 ⇒ **连 per-frame 都不是**。

⇒ 训练后的 Actor ≈ **常数函数**：不看 chunk、不看 state、不看 history、不随 t 变。
按规范 §11 风险表："学成常量（打不过 A2 或 shuffle 不掉分）⇒ 判 **FAIL**"。
**当前 +0.001 的 ΔJ 就是"常数解"的收益。**

**对照：初始化时输入是活的**（同结构随机权重，`torch.manual_seed(0)`）：

| 扰动 | cos(Δ_扰动, Δ_原) | max\|diff\| |
|---|---|---|
| shuffle `Z` | **0.2928** | 4.5e-3 |
| shuffle `state64` | 0.7611 | 3.0e-3 |
| shuffle `hist10` | 0.9781 | 8.8e-4 |

（初始化时 `‖Δ‖=0.0037`、未 cap `‖u‖=0.0074`；Δ 窗口间 std 4.4e-4。）
⇒ 坍缩是**训练过程**造成的，不是结构/接线的错。

---

## 6. 根因分析（推断，但有强证据）

### 6.1 直接原因：`apply_norm_cap` 的 tanh **完全饱和**

对同一 checkpoint（it=200）实测未 cap 的 head 输出：

| 量 | 值 |
|---|---|
| `‖u‖` mean / min / max | **6.607 / 4.910 / 16.474** |
| `cap` | 0.08 |
| `‖u‖/cap` | **≈82×** |
| `tanh(‖u‖/cap)` | **1.0000** |

在 `n≫cap` 时，`Δ = u·tanh(n/cap)·cap/n ≈ cap·u/n`，其 Jacobian 的量级 ≈ **`cap/n` ≈ 0.012**
（切向；径向 ≈ 0）⇒ **输入路径的梯度被压小 ~80×**。

### 6.2 后果：编码器梯度比 head 小 45–180×

`J.mean()` 反传得到的逐子模块梯度范数（实测）：

| 子模块 | 梯度范数 |
|---|---|
| `head`（MLP，1.53M 参数） | **4.78e-2** |
| `chunk_encoder`（attention） | **1.00e-3** |
| `token_proj` | 3.33e-4 |
| `hist_gru`（HistoryGRU） | **2.64e-4** |
| `chunk_norm` | 4.65e-5 |
| `pe_chunk` / `pe_pos` | 3.74e-5 / 3.78e-6 |

⇒ 优化器只能学"最后一层的常数偏置/常数方向"，编码器几乎不动。
而常数方向是训练早期的**噪声方向**（实测与 val 平均梯度 cos≈0.03），所以只拿到常数解的一小部分。

### 6.3 为什么 `‖u‖` 会涨到 6.6？

- 损失对"更大的 Δ"有利（exp0 的尺度标定：‖Δ‖=0.01→ΔJ+0.0037，0.08→+0.0241），
  在 `n<cap` 阶段径向梯度为正 ⇒ 幅度被推大；
- 进入 `n>cap` 后 tanh 的径向梯度迅速趋 0，但 AdamW 的动量/自适应 + `grad_clip=1.0`
  会让它继续过冲到 ~80× cap；
- 一旦饱和，方向学习速度 ∝ `cap/n`，而常数偏置（`head` 末层 bias/权重）不受这个压缩，
  于是"常数项越学越强、输入项越来越学不动"——正反馈。

### 6.4 另一个早期因素：正则项的梯度在前期是 J 的 27×

`loss_terms_grad`（val 32 窗口、it=50 ckpt）实测：

| 损失项 | 数值（batch mean） | 参数梯度范数 |
|---|---|---|
| `−J.mean()` | −1.475 | **0.369** |
| `0.010·L_act` | 0.0516 | **10.04** |
| `0.002·L_smooth` | 0.0116 | **2.78** |
| `0.001·L_anchor` | 0.00094 | 0.0026 |
| 合计 | — | 12.94 |

`cos(g_act, g_total)=0.999`、`cos(g_J, g_total)=0.418`；再叠加 `grad_clip=1.0`
⇒ 前 ~150 步只有 **~3% 的步长**在优化 J。`L_act` 是 hinge，`actRMS<τ` 后自动关断
（it≈160 起 actRMS 7–16 mrad、L_act 0.09–0.74、总梯度范数 0.10–1.7）——**这是"早期相位"，
不是永久问题**，但它和 §6.1 的饱和叠在一起，使得"方向"从未被有效学习。

### 6.5 次要因素：梯度信噪比低

固定评测 token、val 96 窗口拆两半（实测）：

| 量 | 值 |
|---|---|
| ‖mean grad(96)‖ | 0.252 |
| mean‖batch grad(8)‖ | 0.573（ratio 0.44） |
| cos(batch8, mean96) | 0.32 |
| **cos(g_half1, g_half2)（各 48 窗口）** | **0.131** |

⇒ 单窗口梯度方向主要被"窗口自身"的噪声主导；训练时每步还叠加**新的 VLA 流噪声**，
信噪比更低。这不是本轮的"主因"（结构/梯度已验证），但会显著拖慢方向学习。

---

## 7. 已经排除的解释

| 假设 | 证据 | 结论 |
|---|---|---|
| 梯度断了 / 符号反了 | 参数级 FD rel_err 2e-4~5e-3 | **排除** |
| 输入接错（没接 attention/GRU） | `d_in=400`，forward 拼接正确；初始化时 shuffle Z → cos 0.29 | **排除** |
| 数据/历史口径错 | hist10 强口径对拍两条路径 `max|Δ|=0` | **排除** |
| 部署契约错 | 真 ckpt 自检 PASS | **排除** |
| Critic 影响 | 关闭（`BOOTSTRAP_NOT_IDENTIFIABLE`） | **排除** |
| 评测集不敏感 | oracle 上界 +0.0298、96.9% 窗口为正 | **排除** |
| 只是"跑得少" | it=50/100/150/200 的 shuffle 敏感性：it=200 已坍缩；且编码器梯度小 45–180× | **不是唯一原因** |
| 学习率太低 | 不是主因：梯度被 cap 压缩 + 常数项占优；需要修复后再评估 lr | **待定** |

---

## 8. 可提升空间（同一评测集、cap=0.08）

| 方案 | ΔJ（val 96 窗口） | 说明 |
|---|---|---|
| Base（Δ=0） | 0 | 参照 |
| **当前 Actor（it=200）** | **+0.00101** | 3.7σ；≈常数解 |
| 与输入无关的"常数解"上界（沿窗口平均梯度走 cap） | **+0.0078** | 只需学一个 (t×64) 常数场 |
| **逐窗口 oracle 上界**（∂J/∂Δ 方向 + cap 范数） | **+0.0298** | p50 +0.0270，96.9% 窗口为正；**不可部署**，只是上界 |

⇒ 条件化 Actor 的理论空间很大（oracle 是常数解的 3.8×），但当前只拿到 oracle 的 **3.4%**。

---

## 9. 候选修复方案（待审阅者拍板）

### 方案 A：把"未 cap 的 `‖u‖`"按住（最小改动，不改部署语义）

- 在训练损失里加一项 `λ_u · mean(relu(‖u‖/(κ·cap) − 1)²)`，`κ≈2`；
  或在每步后对 head 末层做**范数投影**（让 `‖u‖ ≲ 2·cap`）。
- 优点：不动 `apply_norm_cap` 本身（部署契约不变：输出仍 ≤ cap），不加参数，单一超参；
- 风险：`λ_u` 需要调；需要验证"不会把 Δ 压成 0"；
- **验证门禁**：训练每 N 步跑 `shuffle(Z/state/hist)`，要求 `cos(Δ_perturbed, Δ) < 0.9`；
  否则判"坍缩"，早停。

### 方案 B：方向-幅度解耦（更彻底）

- `Δ = cap · tanh(ρ) · u/‖u‖`，其中 `ρ` 由**单独的标量头**给出（初始 `ρ=-2` ⇒ ‖Δ‖≈0.012·cap）；
- 方向 `u/‖u‖` 的梯度尺度是 O(1)，不受 `cap/‖u‖` 压缩；
- 代价：**参数量 +1（1,744,833）**，与规范锁定的 1,744,832 不符；需要审阅者决定是否接受。

### 方案 C：最后一层的权重归一化 / 更强的幅度正则

- 对 head 末层做 weight norm（把 `‖W_last‖` 固定），只学方向；
- 或把 `L_anchor` 从"对 cap 的软约束"改成"对未 cap `‖u‖` 的硬约束"；
- 风险：改变规范 §5 的结构描述（末层初始化/结构）。

### 顺带需要一起决定的（不改主因，但影响收敛）

1. **`λ_act` / `λ_smooth` 的尺度**：实测梯度是 J 的 27× / 7.5×（早期）。是否改成
   `λ_act=3e-4`、`λ_smooth=3e-5`（让 ‖g_reg‖≈0.1‖g_J‖），或改成"只在 `rms>τ` 的样本上回传"？
2. **`grad_clip`**：1.0 会把"J 的 0.369"和"正则的 10"一起归一化；是否改成按项归一化/不 clip？
3. **梯度噪声**：是否允许 `batch 512`/梯度累积（≠规范锁定 256）或 `cache-tokens` 消融？
4. **防坍缩门禁**：是否把 §9 的 shuffle 测试写进每 N 步的评测（不通过就早停并标记 `FAIL_CONSTANT`）？

---

## 10. 直接提给 GPT 的问题

1. **根因判断**：`apply_norm_cap` 的 tanh 饱和（`‖u‖≈82×cap`）+ 常数项占优，
   是否足以解释"输入全死 + 帧间常数"？还有没有别的机制（例如 `head` 末层 bias 的自由增长、
   `grad_clip` 的方向归一化、AdamW 对噪声的自适应）需要一起考虑？
2. **修法选择**：A/B/C 里推荐哪个？如果推荐 A，`κ` 和 `λ_u` 怎么定（有无解析上界）？
   如果推荐 B，能否在不增加参数的前提下实现（例如用 `‖u‖` 的 detached 标量）？
3. **顺序**：应该先修 cap 再调 lr/λ，还是先做一个"无 cap（或 cap 很大）+ 强 anchor"的对照
   来分离"饱和"与"优化噪声"两个因素？
4. **防坍缩验收**：shuffle 测试的具体阈值（cos < ? / 相对变化 > ?）和频率（每 N 步）怎么设？
5. **期望值**：给定 oracle=+0.0298、常数解上界=+0.0078，一个"真的条件化"的 Actor
   在 1000×256 样本、on-the-fly VLA 噪声下，合理的目标 ΔJ 是多少？是否应该改用
   cached tokens 先证明可学性、再回到 on-the-fly？
6. **是否需要"两阶段"**：先在 cached-pool（固定 token、固定窗口）上把条件化 Actor 训到
   明显超过常数解（验证结构与梯度），再切到 on-the-fly 做正式对照？规范允许
   `cache-tokens` 只用于 pilot（必须标 `data_mode=cached_pool`，不得与主表混）。

---

## 11. 复现命令与产物

```bash
# 环境
source scripts/harness03_env.sh        # worktree 代码 + 主仓库只读数据
# 单测
python3 -m pytest tests/test_temporal_v5_grad.py tests/test_value_td_lambda.py \
    tests/test_actor_tail_value_grad.py tests/test_history_and_token_index.py \
    tests/test_physical_action_trust.py tests/test_v5_fail_closed.py -q      # 70 passed

# 评测集（448 窗口，uniform）
python3 scripts/diag_bptt_learnability.py build --out runs/temporal_ac_bptt_v5/eval_set \
    --n-train 256 --n-val 96 --n-test 96 --batch 64 --vla-batch 4 --uniform-frames 1 --seed 0

# 主训练（2×4090 各一个 seed）
bash scripts/launch_v5_main.sh         # = 2× python3 scripts/train_temporal_ac_bptt_v5.py ...

# 诊断
python3 scripts/diag_oracle_upper_v5.py --eval-set runs/temporal_ac_bptt_v5/eval_set \
    --split val --ckpt runs/temporal_ac_bptt_v5/main/seed0/checkpoint_actor_best.pt \
    --out runs/temporal_ac_bptt_v5/main/seed0/oracle_upper_val.json
python3 scripts/diag_action_budget_v5.py --ckpt .../checkpoint_actor_best.pt \
    --eval-set runs/temporal_ac_bptt_v5/eval_set --split val --out .../action_budget_diag.json
python3 scripts/check_history_parity_v5.py --eval-set runs/temporal_ac_bptt_v5/eval_set \
    --out runs/temporal_ac_bptt_v5/main/seed0
python3 scripts/verify_v5_deploy_contract.py --ckpt .../checkpoint_actor_best.pt
python3 scripts/v5_run_report.py --run runs/temporal_ac_bptt_v5/main/seed0
```

产物目录：`comimic-wt03/runs/temporal_ac_bptt_v5/`
（`eval_set/`、`main/seed{0,1}/`、`main/wm_horizon_error.{json,csv}`、`main/wm_dump_check.json`、
`main/exp0/exp0.json`、`main/grad_fd.json`）。

---

## 附录：关键数字速查

| 量 | 值 |
|---|---|
| Actor 参数 | 1,744,832（T=40, use_history=1, d_in=400） |
| 训练 batch / 步数 | 256 / 1000（当前 ~420） |
| 合格帧 | 346,729（漂移）→ 338,402（+窗口约束） |
| val base J | 1.50594 |
| 当前 held-out ΔJ | seed0 +0.00101（3.7σ）/ seed1 +0.00037（1.3σ） |
| val ΔJ 的 SE | 2.75e-4 |
| oracle 上界 / 常数解上界 | +0.0298 / +0.0078 |
| 未 cap `‖u‖`（it=200） | 6.61（cap=0.08，82×，tanh=1.0） |
| 编码器 vs head 梯度 | 1.0e-3 vs 4.78e-2（45–180×） |
| shuffle Z / state / hist 的 cos（it=200） | 0.99994 / 0.99971 / 0.999996 |
| shuffle Z / state / hist 的 cos（初始化） | 0.293 / 0.761 / 0.978 |
| 正则项/J 的梯度比（早期） | 10.04 / 0.369 ≈ 27× |
| 梯度拆半 cos（48 vs 48 窗口） | 0.131 |
| 动作预算（it=50） | 前 5 步 RMS 19–27 mrad；t=19 peak 2.54 rad（相位切换） |
| WM 40 步（B 路线门槛） | dump 口径 20 步 p50 = 21.2 mm ⇒ `B_NOT_RUN` |
