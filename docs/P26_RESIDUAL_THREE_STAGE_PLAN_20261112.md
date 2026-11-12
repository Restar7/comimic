# P26 计划：三阶段 —— 残差快速判定 → p25（新 WM 联合微调）→ 冻结 VLA 训动作残差

> 生成：2026-11-12 08:10 ｜ 执行：C 线 ｜ 状态：**待执行**
> 一句话：先把「动作残差能不能真的降漂移」在 p21 上快速验证（阶段 0），
> 再走新 WM 的完整链路出正式 VLA（阶段 1），最后在冻结的 VLA 上训残差（阶段 2）。

---

## 0. 为什么是这个顺序（先回答“能不能复用 p21”）

**复用，但分两处用**：

| 阶段 | 用哪个 VLA | 为什么 |
|---|---|---|
| **阶段 0（快速判定）** | **p21**（现成的） | 残差能不能降漂移，和 VLA 是谁关系不大；p21 在手，**30–60 分钟就能出结论**，不用等 refresh |
| 阶段 1（正式 VLA） | **p25**（新 WM 重训的） | 正式交付要用最好的 VLA；这一步沿用已有计划 |
| 阶段 2（正式残差） | **p25 的 best** | 在最终 VLA 上重训残差（token 分布变了必须重训） |

**理由**：残差是全新的模块，先要回答“它到底有没有用”。用 p21 做这个问题最快；如果阶段 0 就失败，
阶段 1 的半天就不用花了。

---

## 1. 阶段 0：残差快速判定（p21 + 旧 WM，约 1 小时）

### 目标

在 **WM 的可微 rollout 里**，给动作加一个残差 `Δa`，看 `e_body_pos` 能不能从 **1.04 cm** 明显降下来。

- 目标：**≥ 15% 下降**（1.04 → ≤0.88 cm）；
- 不够（<5%）→ 残差这条路也基本没戏，止损。

### 做法

1. 给 `diff_vla_wbc/rollout.py::imagine_with_tokens` 加一个可选参数 `residual_fn`；
2. 在 **第 386 行** 之后注入：

```python
action = _call_act(bundle, obs_flat, cond, ref)
if residual_fn is not None:
    action = action + residual_fn(current, tokens[:, min(t, T - 1)])
next_state = bundle.step(current, action, xi, step_idx=t + 1)
```

3. 写 `/tmp/probe_residual.py`：冻结 VLA（用 p21 采 token，`grad=False`）、冻结 WBC/WM，
   只训残差网络，300 步，看训练窗口上的 `e_body_pos`。

> ⚠️ 这一步只是**训练态验证**（在 WM 的想象里）。它回答“动作层有没有空间”，
> 真正的结论要等阶段 3 的真仿真。

---

## 2. 阶段 1：p25（新数据 → refresh → critic → 联合微调）

**完全沿用** `docs/P25_WM_REFRESH_ON_REPEAT_PLAN_20261112.md`，不改动。产出：

- `p25/refresh3/wm_cl_v3.pt`（+ sha256）
- `p25/ft300/checkpoint_lora_best.pt`（+ val/test dJ）
- G3 判据：`dJ/J_base > 0.6%`

> 如果 G3 不过（大概率），**阶段 0 的残差仍然可以在 p21 上继续做**，两条线解耦。

---

## 3. 阶段 2：残差网络设计（核心）

### 3.1 输入 / 输出

| 项 | 维度 | 说明 |
|---|---|---|
| **输入 state** | **71** | rollout 的 `current`（root_pos3+quat4+q29+v3+w3+qd29），与 WM 同口径 |
| **输入 token** | **64** | 当前帧的 motion token（`tokens[:, min(t, T-1)]`） |
| 输入 prev_action | 29（可选） | `a_prev`，帮助残差知道当前控制量 |
| **输出 Δa** | **29** | 加在 WBC 动作上，单位与 action 相同 |

**最小版（阶段 0 用）**：`[state71 | token64 | prev_action29] = 164 维 → 29 维`。
**完整版（阶段 2 用）**：再加 10 帧 state64 历史（+640），共 804 维。

> 不加图像：残差是控制层的反馈律，图像对它没有增量信息；加了反而容易过拟合。

### 3.2 网络结构（小、有界、平滑）

```python
class ActionResidual(nn.Module):
    def __init__(self, d_state=71, d_token=64, d_act=29, hidden=256, max_res=0.05):
        super().__init__()
        self.max_res = max_res                      # 相对 action 归一化尺度的上限
        self.net = nn.Sequential(
            nn.Linear(d_state + d_token + d_act, hidden), nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d_act),
        )
        nn.init.zeros_(self.net[-1].weight)          # ★ 初始化 Δa=0，从“不加残差”开始
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, state, token, a_prev):
        return self.max_res * torch.tanh(self.net(torch.cat([state, token, a_prev], -1)))
```

**参数规模**：约 **5 万**（hidden 256）。部署开销 <0.1 ms/帧。

| 设计选择 | 理由 |
|---|---|
| `tanh × max_res` | 硬限幅，**残差不可能改变任务意图**（默认 ±5%） |
| 末层零初始化 | 训练第一步就等价于原策略，不会一上来把动作打崩 |
| LayerNorm | 状态各维量级差很大（位置 m、角度 rad、速度 m/s） |
| 不含图像 | 控制层反馈律，用不到视觉 |

### 3.3 损失

```python
loss = -J.mean()                                  # 主目标（和联合微调同一个 J）
     + λ_mag   * (a_res ** 2).mean()              # 幅度惩罚，λ_mag = 1e-3
     + λ_rate  * (a_res[1:] - a_res[:-1]).square().mean()   # 平滑惩罚，λ_rate = 1e-2
     + λ_lim   * limit_penalty                    # 复用现有 λ_lim 项（防越限）
```

- `λ_mag` 防止残差“独自干活”把 VLA 架空；
- `λ_rate` 防止 50 Hz 抖动；
- `λ_lim` 与训练主链一致，避免越限。

### 3.4 训练循环（阶段 2）

```python
vla.eval(); wbc.eval(); wm.eval()                 # 全冻
res = ActionResidual().to(dev)
opt = torch.optim.Adam(res.parameters(), lr=1e-3)
for step in range(int(args.steps)):               # 默认 300
    batch = provider.sample(32, seed=...)
    with torch.no_grad():
        tok = sampler.sample(batch["frames"], seed=..., grad=False)   # 冻结 VLA 的 token
    J, ex = window_returns(bundle, {**batch, "tokens": tok}, H, GAMMA,
                           want_actions=True, residual_fn=res)
    a_res = ex["residual"]                         # 需要在 rollout 里返回
    loss = -J.mean() + 1e-3*(a_res**2).mean() + 1e-2*(a_res[1:]-a_res[:-1]).square().mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
```

- **300 步**，batch 32，约 **10–15 分钟**（只训 5 万参数的 MLP）；
- 每 50 步报一次 `e_body_pos`（训练窗口）+ `‖Δa‖` 均值和最大值。

### 3.5 部署注入点

| 侧 | 位置 | 改法 |
|---|---|---|
| 训练 | `diff_vla_wbc/rollout.py:386`（`imagine_with_tokens` 的动作处） | 加 `residual_fn` 参数并注入 |
| 评测 | 闭环里 WBC 输出动作之后、写入 sim 之前（`ha_isaac_closedloop` 的动作路径） | 加钩子：`a = a + res(state64, token64, a_prev)` |
| 交付 | 一个 `residual.pt`（5 万参数）+ 一段前向代码 | 与 VLA ckpt 解耦，可单独开关 |

### 3.6 验收判据（阶段 2）

| 指标 | 阈值 | 判读 |
|---|---|---|
| `e_body_pos`（WM 训练窗口） | **≥15% 下降**（1.04 → ≤0.88 cm） | 残差有效 |
| `‖Δa‖` 均值 | ≤ 0.5 × `max_res` | 没有顶到上限（否则说明在硬撑） |
| 越限率（`λ_lim` 项） | 不高于 p21 的 6.8% + 2pp | 没有引入新风险 |
| **闭环（真仿真）配对** | 30 格 ≥6 净胜 | 真机也有效 |

---

## 4. 阶段 3：评测

1. **真仿真注入**：把训练好的残差接入闭环，跑 30 格（OpenDoor + Boxing，seed0–2 × repeat0–4），
   与「无残差」的同一 VLA 逐格配对；
2. **三方对照**：`p25` / `p25+残差` / `base`；
3. 报告 `e_body_pos`（真仿真）与成功率。

---

## 5. 风险与止损

| 风险 | 表现 | 止损 |
|---|---|---|
| 残差在 WM 里有效、真仿真无效 | 阶段 2 指标好看，阶段 3 不动 | 说明在利用 WM 误差 → 加“不许偏离数据流形”的约束（如对 `Δa` 加 KL 到 0，或限制 `max_res≤2%`） |
| 残差主导 | `‖Δa‖` 顶到上限、J 全靠残差 | 提高 `λ_mag`，或降到 `max_res=2%` |
| 抖动/不稳定 | 动作高频振荡、越限上升 | 提高 `λ_rate` |
| 阶段 0 就失败 | 1.04 → 0.98 都到不了 | **停止**，说明动作层也没有空间，漂移地板是参考/WBC 本身 |

---

## 6. 时间线

| 阶段 | 内容 | 时长 | 前置 |
|---|---|---|---|
| **0** | 残差快速判定（p21 + 旧 WM） | 1 h | 无 |
| **1** | p25：新数据 → refresh → critic → 联合微调 | 2 h | 阶段 0 结果（可并行） |
| **2** | 正式残差（p25 的 VLA） | 0.5 h | 阶段 1 产出 |
| **3** | 真仿真 30 格配对 | 2 h | 阶段 2 产出 |

**阶段 0 和阶段 1 可以并行**（不同 GPU），一个回答“残差有没有用”，一个回答“新 WM 有没有用”。

---

## 7. 交付物

- [ ] `rollout.py` 的 `residual_fn` 支持（改动 + 单测）
- [ ] 阶段 0 报告：WM 里 `e_body_pos` 下降幅度、`‖Δa‖` 分布
- [ ] p25 的 WM/critic/LoRA 三个 ckpt + `dJ/J_base`
- [ ] 阶段 2 的 `residual.pt` + 训练曲线
- [ ] 阶段 3 的 30 格三方配对表（含 McNemar）
- [ ] 一句话结论：**动作残差是否能把漂移从 1.04 cm 降下来；真仿真里是否成立**

---

## 8. 和现有结论的关系

| 已知事实 | 对本计划的影响 |
|---|---|
| VLA token 与标签 token 等价（1.040 vs 1.049 cm） | 漂移在 VLA 控制范围外 → 残差是唯一有信息的层 |
| 逐窗口 token 补偿不可从观测预测（R²<0） | 残差不需要预测，它**每帧看到误差** → 不受此限制 |
| p19–p24 在 J 上改目标/改梯度全部中性 | 本计划不碰目标，改的是**动作通路** |
| 30 格配对只能看 ≥30pp | 阶段 3 的结论要写成方向性判断 |
