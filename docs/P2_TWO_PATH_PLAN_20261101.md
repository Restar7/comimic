# P2 两条路并行：拆分 8.5（WBC-only） vs 联合解冻（VLA+WBC+KL）

> 日期：2026-11-01（**已更正**：Path B 是 VLA+WBC 联合训练，不是只训 WBC）
> 状态：方案（PLAN），代码未改。
> 目标：当前 8.5 是"VLA 冻结、只训 WBC"；再并行一条"VLA head + WBC 联合训练 + KL 锚"
> 的路，用同一 VLA 起点的 32 集配对闭环选更合适的那条。

---

## 1. 两条路（更正后）

| | **Path A（当前 8.5）** | **Path B（新，联合）** |
|---|---|---|
| VLA action head（LoRA+proj_out） | **冻结**（`FM_LR=0`） | **训练**（`FM_LR=1e-5`） |
| VLM backbone（2B） | 冻结 | 冻结（与所有 joint 轮一致） |
| WBC | 全参数/decoder 训练 | **decoder 训练**（或全参数） |
| KL 锚 | 无（只有 action hinge） | **VLA token KL + WBC action KL** |
| 步数 / lr | 200 / WBC 1e-6 | 100–200 / VLA 1e-5、WBC 5e-7 |
| 归因 | 干净：只换执行器 | 混淆：策略+执行器一起变 |
| 目的 | 尾部保险（已看到） | 看 co-adapt 能不能比拆分更好 |

> "VLA 解冻"在本项目里 = **action head（LoRA+proj_out）可训**；2B VLM backbone 仍然冻结
> （这是所有 joint 轮的共同前提；全量 2B 是另一个量级，不在这次范围）。

---

## 2. 两个 KL 锚

### 2.1 WBC action KL（防执行器漂）

```
a0 = WBC_start(s, token)         # 训练开始时 deep copy 的 WBC，冻结
loss_wbc_kl = λ_wbc * MSE(a_current(s, token), a0.detach())
```

### 2.2 VLA token KL（防策略漂）

联合训练如果只锚 WBC，VLA 可以"指挥"WBC 一起漂，历史上就是 21× 崩溃。所以再加一个
token 空间的行为锚：

```
tokens0 = VLA_start(obs, noise)   # 同一个 noise，冻结的起点 action head
loss_vla_kl = λ_vla * MSE(tokens_current, tokens0.detach())
```

- `tokens0` 用**同一个 `_noise`** 采样（`_sample_tokens(..., noise=_noise)`），所以对比是
  "同一状态、同一噪声下，起点 VLA 会输出什么"；
- 起点 action head 用 `copy.deepcopy(self.vlm.action_head)`，`requires_grad=False`；
- 额外一次 FM 采样（K=4），约 +0.2 s/iter；可接受；
- 现有的 `trust_eps=0.2 / trust_weight=1.0`（token 对 self 的铰链）**保留**，它是另一个锚
  （对数据采集策略），不是替代。

> 严格意义的 flow-matching KL 不好算；这里用"同噪声下的 token MSE"作为**行为 KL 代理**，
> 和 WBC 的 action KL 同一思路：尺度可解释、λ 可调。

---

## 3. 代码改动（完整清单）

### 3.1 `diff_vla_wbc/train_v2.py`

1. `JointV2Config` 增加：
   ```python
   wbc_unfreeze: str = "all"      # all | dec
   ```
   （`train_wbc` / `wbc_lr` 已有）
2. `_freeze_executor()`：`train_wbc=True` 时按 `wbc_unfreeze` 只解冻 `dec.*` 或全部。
   注意 `_freeze_executor` 在 `wbc_parameters` 收集**之前**调用，所以优化器/梯度同步
   自动只包含解冻的参数。

### 3.2 `diff_vla_wbc/rollout.py`

1. `Rollout` 增加 `anchor_actions: torch.Tensor | None = None`。
2. `imagine_with_tokens(..., anchor_policy=None)`：
   - 每个 step 用 `anchor_policy` 在**同一份 obs_flat/cond/ref** 下算 `a0`，
     `detach()` 后存进 `anchor_actions`；
   - 实现方式：临时 `bundle.policy = anchor_policy`，算完换回；
   - `anchor_policy=None` 时行为完全不变。

### 3.3 `diff_vla_wbc/v3_step.py`

1. `V3Config` 增加：
   ```python
   wbc_kl_anchor: float = 0.0
   vla_kl_anchor: float = 0.0
   ```
2. `__init__`：若 `vla_kl_anchor > 0`，建 `self._vla_ref = copy.deepcopy(self.vlm.action_head)`，
   `requires_grad_(False)`、`.eval()`；`_wbc_ref` 已存在（`train_wbc=True` 时 deep copy 起点 WBC）。
3. `train_step`：
   - 策略 rollout 传 `anchor_policy=(self._wbc_ref if wbc_kl_anchor>0 else None)`；
   - 算 `loss_wbc_kl = λ_wbc * MSE(rollout.actions, rollout.anchor_actions)`；
   - 若 `vla_kl_anchor>0`：用 `_vla_ref` + 同一个 `_noise` 再采一次 tokens，
     `loss_vla_kl = λ_vla * MSE(tokens, tokens0.detach())`；
   - `loss = loss + loss_wbc_kl + loss_vla_kl`，并记录 `loss/wbc_kl`、`loss/vla_kl`。
4. 反向仍然只回传到可训参数：VLA head（`FM_LR>0`）+ WBC 解冻部分。

### 3.4 CLI / env 透传

| 文件 | 改动 |
|---|---|
| `scripts/train_joint_v2.py` | `--wbc-unfreeze {all,dec}`、`--wbc-kl-anchor`、`--vla-kl-anchor`；写进 `args.json` |
| `scripts/launch_joint_ha_v2.sh` | `WBC_UNFREEZE`、`WBC_KL_ANCHOR`、`VLA_KL_ANCHOR` |
| `scripts/s3_loop.sh` | 训练 env 列表加这三个 |
| `scripts/s3_pipeline.sh` | `stage_joint` env 链加这三个 |
| `scripts/run_locked_h20_round.sh` | Path A（现 8.5）与 Path B（联合）两套 env |
| `scripts/eval_h20_8gpu.sh` | 不用改（已支持 `SM_CKPT_MAIN/CTRL`） |

### 3.5 断言

- `args.json` 的 `wbc_unfreeze / wbc_kl_anchor / vla_kl_anchor` 与实际一致；
- Path B 的非 `dec.*` WBC 参数逐位不变（若 `dec`）；
- 起点 action head 的 433 个张量在 **Path A** 后逐位不变；Path B 允许变化，但要把
  `loss/vla_kl` 和最终 token 漂移记下来；
- `loss/wbc_kl`、`loss/vla_kl` 有限且 >0。

---

## 4. 训练配方

共用：H=20、λ_r=0.3、act_weight=0.5、act_eps=0.6、yaw=0、BATCH=64、ACCUM=1、NPROC=8。

**Path A（当前 8.5）**

```
FM_LR=0
TRAIN_WBC_MAIN=1, WBC_LR=1e-6, WBC_UNFREEZE=all
WBC_KL_ANCHOR=0, VLA_KL_ANCHOR=0
STEPS=200
```

**Path B（联合 + KL）**

```
FM_LR=1e-5                 # 比标准 joint 的 3e-5 低，先稳
TRAIN_WBC_MAIN=1, WBC_LR=5e-7, WBC_UNFREEZE=dec
WBC_KL_ANCHOR=0.5, VLA_KL_ANCHOR=0.5
STEPS=100 起，最多 200；每 50 步存 ckpt
```

> λ 的量级：`MSE(action)≈0.25`、`MSE(token)` 量级类似；λ=0.5 时两项各 ~0.12，
> 和奖励/铰链同量级。若 WBC 不动，降 λ；若仍漂，升 λ。

---

## 5. 评测与判定

两条路都从**同一份锁定轮策略 + 上一轮最优执行器**出发，评测：

1. **Path A vs base**：同一 VLA，**上一轮最优执行器** vs A 的 `joint_final.pt`（32 集）；
2. **Path B vs base**：同一个起点系统，**上一轮最优执行器** vs B 的 `joint_final.pt`（32 集）；
3. **A vs B**：同一批 seed，逐集配对（32 集）。

> 这里的 "base" = **上一轮最优执行器**，不是 v6/π0。对 `h20r2`：
> 起点执行器是 **v7**（`h20_8g_b_wbc_joint/train_main/joint_final.pt`），
> 所以 8.5 是 **v8 vs v7**；策略比较是 **B vs h20r2，两边都用 v7**。
> d2b/π0 只作锚点/ sanity，不参与主判定。

判据（比 8.5 更严）：

- vs base：`mean<0` 且 `median<0`；
- **去掉 |diff|>5 cm 异常集后的核心中位也 <0**（区分尾部保险 vs 普遍提升）；
- 零翻倒、p90 不变差、`frac_out ≤ 0.55`；
- Path B 还要看 `L_fm` 不比起点差 >10%（VLA 真的在学，而不是被 KL 钉死）。

如果 A、B 都只赢尾部、核心中位 ≥0 ⇒ 两条都不进默认流程，保留 base 执行器。

---

## 6. 并行执行

- **4+4 并行**：A 占 0–3（`NPROC=4, BATCH=64` ⇒ 每卡 16），B 占 4–7；
  `DDP_GPUS` 分开、`MASTER_PORT` 不同、`SKIP_SELF_TARGETS=1` 共用 self 目标；
- 或 **8 卡顺序**（稳）：A 200 步 → B 100–200 步；
- 评测：4 条 sweep 占满 8 卡；先各 16 集定方向，胜者补到 32 集。

---

## 7. 风险与回滚

| 风险 | 处理 |
|---|---|
| 联合训练 21× 崩溃（历史） | FM_LR 降、WBC_LR 降、双 KL、每 50 步 ckpt + 快速闭环；`frac_out>0.55` 立即停 |
| VLA 被 KL 钉死、学不动 | `L_fm` 与 reward 都不动 ⇒ λ_vla 降一档 |
| WBC 被 KL 钉死 | WBC action 几乎不变 ⇒ λ_wbc 降一档 |
| 归因不清 | 只做 A/B 系统级比较；论文主结果仍用拆分的 A，B 作 exploratory |
| 两条路互相抢资源 | 4+4 时端口/日志/输出目录全隔开 |

---

## 8. 顺序

1. `h20r2` 完整轮：collect 42/42（v7）→ score/transitions → WM 刷新（A+B）→ gate PASS →
   joint 200/200 ✅ → pipeline eval（进行中）；
2. 跑 **Path A**（当前 8.5，无 KL）：200 步 + 32 集 A/B；
3. 改代码，跑 **Path B**（VLA+WBC 联合 + 双 KL）：100–200 步 + 32 集 A/B；
4. 按 §5 选下一轮；两条都只赢尾部 ⇒ 保留 base，转 WM/数据方向。

---

## 9. 一句话

**Path A = 冻结 VLA、只训 WBC（拆分 8.5）；Path B = VLA head + WBC 联合训练，用
“VLA token KL + WBC action KL”双锚防漂。** 两条都从同一个起点、同一批 32 集闭环比较；
联合路风险更高，所以只在 B 显著且稳定时才替换拆分路。
