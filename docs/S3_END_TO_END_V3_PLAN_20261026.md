# S3 端到端 v3 改造方案（2026-10-26）

> 目标：把联合微调从"BC 精修"改回**端到端闭环**语义 —— VLA 输出 token → 人形执行器(WBC)+可微动力学(WM)
> 真的执行 → 执行误差/回报反传回 action head；**全身（token）不做任何标签模仿**，只有手部保留标签监督。
> 本文给出 7 项改动的完整实现意见（文件 / 接口 / 超参 / 验收 / 风险）+ 梯度 clip 的评估。

---

## 0. 定位

| | 现在（v2） | v3 |
|---|---|---|
| token 监督 | FM-BC（标签）为主 | **无**（`λ_bc_token = 0`） |
| 手部 | 混在 78 维 BC 里 | **单独监督**（hand-only FM 掩码，权重 `λ_hand`） |
| 执行器 / WM | 冻结 | **WBC 冻结；WM 用真实转移"保鲜"** |
| 起点 | 参考帧（+WM 自走 burn-in） | **真录制机器人状态**（+扰动 + burn-in） |
| 信用分配 | 16 步 pathwise | **32 步 + token-critic bootstrap（GAE λ）** |
| 规模 | H=16 / batch 4 / accum 4 | **H=32 / batch 8 / accum 4** |
| 场景 | 5 cm 扰动、温吞 | **25 cm 扰动 + rollout 内推扰 + 偏离惩罚** |

**前提事实**（见 §1）：奖励对 token 的依赖是真实的（最难格 **39.8%**），但 VLA 目前离 GT 只有 **0–3.6%**
—— 所以 v3 的成败在于"把偏差/长程做出来之后，端到端能不能纠回显著的一块"。§1 的 oracle 数字就是天花板。

---

## 1. 已验证的事实（先看这张表再决定投多少）

`scripts/probe_headroom.py`（32 窗口，C6h，三臂同起点）：

| 格 | GT | VLA | 零 token | token 区分度 | **headroom(GT−VLA)** |
|---|---|---|---|---|---|
| 扰动 1× / H=16（现训练口径） | 1.2218 | 1.2188 | 1.1439 | +6.4% | **+0.2%** |
| 扰动 1× / H=32 | 2.0613 | 2.0609 | 1.6377 | +20.5% | +0.0% |
| 扰动 5× / H=16 | 0.8328 | 0.8267 | 0.7658 | +8.0% | +0.7% |
| **扰动 5× / H=32** | 1.2323 | 1.1880 | 0.7419 | **+39.8%** | **+3.6%** |

- **token 确实被使用**，而且"越脏越长"越明显 ⇒ 端到端反传有信息可用；
- 但 **VLA ≈ GT**（`token_mse(VLA vs label)=0.004`）⇒ 标称区间里没有误差可纠，这正是 λ_r 消融"看起来全是 BC"的原因；
- `scripts/probe_token_oracle.py`（已出两批，见下）：从偏离起点直接对 64 维 token 做奖励梯度上升，
  给"端到端最多能纠回多少"。**这个数 ≥10% 才值得投 §2；≤3% 就先改接口（§6）。**

---

## 2. 七项改动（逐条给实现意见）

### 2.1 WM 用**真实转移**保鲜（最关键）

**问题**：冻结 WM 是在 WBC 的旧策略分布上训的；token 一漂，想象就进入外推区。
SuperMimic 的做法是**每个 iter 用 replay buffer 的真仿真转移重训 WM**（`runner.py`：`wm_loss`）。

**我们不需要仿真器**：HA 录制数据里就有**真转移**（机器人真执行的状态 + 指令推出的动作）：

```
s_t     = [ pelvis_pos(t), pelvis_quat(t), joint_pos(t),          # 执行轨迹 ← data/npz/ha_tokens
            root_lin_vel(t), root_ang_vel(t), joint_vel(t) ]
a_t     = ( q_cmd(t+1) − default_q ) / action_scale               # 指令流 ← runs/isaac_motion/ha_cmd
s_{t+1} = s_{t+1}
```

这正是"部署时执行器接收的动作 → 机器人实际怎么动"的真数据（含跟踪滞后）。

**实现**
- 新脚本 `scripts/build_real_transitions.py` → `runs/wm_real/ha_transitions.pt`（字段：`states(N,71)`, `actions(N,29)`, `next_states(N,71)`, `clip_ids`）；
- 新模块 `diff_vla_wbc/wm_refresh.py`：
  ```python
  def refresh_wm(bundle, data, *, steps=200, batch=2048, lr=1e-4, vel_weight=5.0, kl_anchor=1e-3)
  ```
  - 损失 = 一步状态 MSE（pos/ori/q + `vel_weight`×速度）+ `kl_anchor`×‖θ−θ0‖²（**锚定原始 WM**，防漂移）；
  - **每 N 步（建议 100）刷新一次**，刷新时 `no_grad` 与策略更新解耦（stop-gradient）；
  - 审计行：`[AUDIT][wm] real_loss=..., 1step_pos=..., vel_ratio=...`。
- **验收**：刷新后 `qd_corr`（vs `ha_cmd`）≥0.9、`ratio∈[0.9,1.1]`；`imag_contrast`（§1 的 GT vs 零）不下降。

**风险**：真转移的 `a_t` 是"完美跟踪者的动作"，与 WBC 实际输出有分布差 → 先用 `kl_anchor` 压着，只做小幅修正。

### 2.2 起点用**真录制状态**

**问题**：v2 起点是参考帧（或 WM 自走 burn-in）→ 一起步就在"理想状态"，误差 = 0。
SuperMimic 的起点来自**真 replay buffer 的真状态**。

**实现**（`diff_vla_wbc/burnin.py`）
```python
def sample_burnin(..., start_source="recorded"):   # 新参数，默认 "reference" 保兼容
    if start_source == "recorded":
        state = recorded_state(npy_states, frames)      # ← 真执行状态（npz joint_pos/joint_vel + ha_cmd root）
    else:
        state = reference_start(...)
    state = perturb_state(state, xy=0.25, yaw=0.30, dq=0.05, dqd=0.30)   # 2.7 的幅度
    # 然后照旧：K 步 GT token burn-in（用仿真 anchor）
```
- 真状态获取：`data/npz/ha_tokens/<clip>.npz` 的 `body_pos_w[:,0] / body_quat_w[:,0] / joint_pos / joint_vel` +
  `ha_cmd` 的 root 线/角速度（两库 clip 数、每段长度逐位相同，已验）；
- 缓存成 `runs/wm_real/ha_recorded_states.pt`（或复用 2.1 的 `states`）。

### 2.3 token-critic（弥补我们漏掉的部件）

SuperMimic：`J += γ^H·c_H·V(s_H, f, ξ, goal)`，critic 用 **GAE(λ) 在想象转移上训**；
它**不吃 token**（"the critic never takes the token"），我们照抄。

**实现**（新模块 `diff_vla_wbc/token_critic.py`）
```python
class TokenCritic(nn.Module):          # ~ (71 + 6*10 + 33) -> 256 -> 256 -> 1  ≈ 1.2e5 参数
    def forward(self, state71, ref6x10, xi33): ...
def gae_targets(rewards, dones, values, gamma=0.99, lam=0.95): ...
```
- 策略损失：`L = −mean( J + γ^H · c_H · V(s_H) )`，`V` 用 `detach`；
- critic 每步用 GAE 目标 + minibatch 回归（`critic_epochs=5`, `minibatches=4`）、`clip_grad_norm_(critic, 1.0)`、`clip_grad_value_(critic, 0.1)`；
- **验收**：critic 的 explained-variance 在 200 步后 >0.3；有/无 critic 的 A/B（各 300 步）比 `imag_return` 与 token 漂移。

**注意**：critic 会**固化 WM 的偏差**（bootstrap 的是想象回报）。所以 2.1（保鲜）必须先做，否则 critic 会把模型利用放大。

### 2.4 想象规模：H=32 / batch 8 / accum 4

- 成本主要在 **FM 采样（K=4 次 DiT 前后向）**，而 H 只影响 WBC+WM 的 rollout（很便宜）；
  ⇒ **H 16→32 几乎免费**，batch 8 ≈ ×2 时间（约 2.5–3 s/微步，1000 步 ≈ 45–60 min）；
- 若 OOM：`--fm-grad-backend checkpointed`（显存 O(1)，慢 ~30%）；
- 审计行要打：`imagined_steps_per_s = batch*H / s_per_step`（目标 ≥100）。

### 2.5 取消 token BC，保留手部监督

用**同一个原生 FM 损失**，只改 `action_mask`：

```python
tgt, _ = bc_target(codec, label_tok, hands14, head_dim=132)   # 目标只用来做手部那 14 维
mask = torch.zeros_like(tgt); mask[..., 64:78] = 1.0          # ← 只监督手
L_hand = fm_bc_loss(head, bb_out, act_in, tgt, mask)
L = λ_hand * L_hand + λ_r * L_r                               # token 项完全不加
```
- `λ_hand` 建议 **0.2**（与 v1 的手部权重一致），`λ_r` 从 **0.3** 起（grad_ratio 目标 0.1–0.5）；
- **后果要接受**：token 唯一的监督就是奖励 → 漂移不受"标签"约束，只能靠 2.1/2.7 的可信度兜底；
- 建议同时加**无标签的先验界**（可选）：把 token 每维 clamp 到 base VLA（`checkpoint-8000`）在 HA 数据上的
  输出分布的 [q01,q99]（这是**底模自己的先验**，不是标签），存成 `runs/s3_token_prior.json`。

### 2.6 梯度 clip：**"1.0 / value 0.1" 只对 critic 合理，对 1.6B 的 head 不合适**

| 对象 | SuperMimic 设置 | 我们的建议 |
|---|---|---|
| critic（~1e5 参数 MLP） | `clip_grad_norm_(1.0)` + `clip_grad_value_(0.1)` | ✅ **照抄**（同构、同量级） |
| WBC policy（~1e7 MLP） | `clip_grad_norm_(1.0)` + `clip_grad_value_(0.1)` | 不适用（我们冻结） |
| **VLA head（1.62e9, AdamW, lr 1e-5）** | — | ❌ **不要用 1.0**：这个规模 + 这个 lr 的 ‖g‖ 通常是 O(10–100)，硬 clip 到 1.0 等于把有效 lr 压 10–100×（等效冻结） |
| value clip 0.1 | — | ❌ 不要用在 head 上（会把绝大多数元素截断）；只给 critic 用 |

> **修正（同日实测后）**：上面"clip 1.0 等于把有效 lr 压 10–100×"这句**只对 SGD 成立**。
> 我们用的是 **AdamW**，它对梯度做 `m/√v` 归一化，**全局常数缩放近似不变** —— 也就是说
> `clip_grad_norm_` 在 AdamW 下几乎不影响步长，只防数值尖峰。实测 pre-clip 范数 **3–9**，
> 所以默认设 `--clip-head 10`（只在尖峰时触发）。真正要小心的是**分项之间的相对权重**
> （奖励 vs 手部），那是 `λ` 的事，不是 clip 的事。

**正确做法（先测再定）**
1. 在门禁里打印 `‖∇_head‖` 的 p50/p99（`--log-grad-norm`）；
2. 若现在 `fm_clip=0.5` **经常触发**（>50% 的步），说明有效 lr 已被压 —— 把 clip 提到 `5×p99`；
3. **奖励项与手部项分别 clip**（两者量级差 5–10×，混 clip 会让大项决定阈值）；
4. 用 `torch.nn.utils.clip_grad_norm_(params, c)` 的返回值监控触发率（`total_norm`）并写进 metrics。

### 2.7 训练场景：把"误差"做出来

- burn-in 扰动：**xy 25 cm / yaw 0.30 rad / dq 0.05 / dqd 0.30**（现在 5 cm/0.1）；
- **rollout 内推扰**：`--push-std 0.15`（每步给 root 速度加 U(−0.15,0.15) m/s），制造中段偏离；
- 奖励加**偏离惩罚**（身体级）：`−w·(relu(tilt−0.5)² + relu(0.6−height)²)`（`isaac_joint.reward` 里已有 `posture_weight`，直接用）；
- 这样"token 好不好"才有差别（§1 实测：5× 扰动 + H=32 → 区分度 39.8%）。

---

## 3. v3 的训练步（伪代码）

```python
b = provider.sample()                      # 真录制起点 + 扰动 + K=10 burn-in（GT hold token）
act_in, bb_out = frozen_backbone(b.model_inputs)          # no_grad
x_norm = sample_once_with_grad(head, bb_out, act_in, K=4) # 一次采样
tok    = codec.decode(x_norm)[..., :64]
roll   = imagine_with_tokens(bundle, tok, **b.rollout_args, horizon=32, push_std=0.15)
ret    = discounted_return(roll, γ)                       # (B,)
V_H    = critic(roll.states[-1], b.ref_feats, b.xi).detach()
L_r    = −(ret + γ**32 * roll.survival[-1] * V_H).mean()  # ★ bootstrap
L_hand = fm_bc_loss(head, bb_out, act_in, tgt, hand_mask)  # 只监督 14 维手
L      = λ_r * L_r + λ_hand * L_hand                       # ← 没有 token BC
# 累积 / clip（分项）/ step
critic.update(gae_targets(roll.rewards, roll.dones, values))   # 每步
if step % 100 == 0: refresh_wm(bundle, real_transitions)        # 保鲜
```

---

## 4. 代码改动清单（按可交付顺序）

| # | 文件 | 改动 |
|---|---|---|
| 1 | `scripts/build_real_transitions.py`（新） | HA 录制 → `runs/wm_real/ha_transitions.pt`（`states/actions/next_states`） |
| 2 | `diff_vla_wbc/wm_refresh.py`（新） | `refresh_wm()`：一步状态 MSE + `vel_weight` + KL 锚定 |
| 3 | `diff_vla_wbc/token_critic.py`（新） | `TokenCritic` + `gae_targets` + `update()` |
| 4 | `diff_vla_wbc/burnin.py` | `start_source="recorded"`；扰动默认 25 cm/0.30 |
| 5 | `diff_vla_wbc/train_v2.py` | `JointV2Config` 加 `lambda_hand / mask_hand_only / critic_* / push_std / wm_refresh_every`；`train_step` 走 §3 |
| 6 | `diff_vla_wbc/rollout.py` | `imagine_with_tokens(..., push_std=0.0)`（每步给 root 速度加噪） |
| 7 | `scripts/train_joint_v2.py` | CLI：`--lambda-bc-token 0 --lambda-hand 0.2 --critic 1 --wm-refresh-every 100 --start-source recorded --push-std 0.15 --log-grad-norm` |
| 8 | `scripts/probe_token_oracle.py`（已有） | 门禁：判定"端到端天花板" |
| 9 | `scripts/probe_headroom.py`（已有） | 门禁：四格 contrast |

---

## 5. 运行命令

```bash
# ① 先造真转移（一次性，~3 min）
python3 scripts/build_real_transitions.py --out runs/wm_real/ha_transitions.pt

# ② 门禁（10 min）：天花板判定
python3 scripts/probe_token_oracle.py --windows 16 --horizon 32 --scale 5 --device cuda:0

# ③ v3 主跑（1000 步 ≈ 1 h）
export PYTHONPATH="$PWD:$PWD/Isaac-GR00T:$PWD/third_party/gr00t_deps:$PWD/third_party/gr00t_deps/_noflash"
export SUPERMIMIC_HA_ALIGN=1 SUPERMIMIC_ROBOT_USD="$PWD/trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"
python3 scripts/train_joint_v2.py \
  --wbc-ckpt runs/S1_handover/WBC_A20000_wmvel_cmd.pt \
  --vlm-ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --motion-packed runs/isaac_motion/ha_cmd/packed.pt \
  --lambda-bc-token 0 --lambda-hand 0.2 --lambda-r 0.3 \
  --critic 1 --critic-lr 3e-4 --gamma 0.99 --lam 0.95 \
  --start-source recorded --perturb-xy 0.25 --perturb-yaw 0.30 \
  --push-std 0.15 --horizon 32 --batch-size 8 --grad-accum 4 \
  --wm-refresh-every 100 --wm-refresh-steps 200 \
  --eval-every 200 --eval-batches 8 --save-every 500 \
  --log-grad-norm --out runs/s3_v3_noBC
```

---

## 6. 判读（三选一，先看 oracle 再看训练）

| oracle 结果（§1 的探针） | 结论 / 动作 |
|---|---|
| `oracle − label` **≥10%** | 端到端有真实空间 → 投 v3（§2 七项）；门禁看 `imag_return`↑、token 漂移 <2×、critic EV>0.3 |
| 3–10% | 空间很小 → 先做 2.1（WM 保鲜）+2.2（真起点）+2.7（脏场景），再测一次 oracle，别急着训练 |
| ≤3% | 端到端在这套执行器/奖励上没有增量 → 改接口（token 携带未来命令 / 观测补速度 / 让 WBC 也进回路），**不要**继续调 λ_r |

训练期的三条硬门禁（任何一条不过就停）：
1. `token_mse_decoded` 相对起点涨幅 **≤2×**（否则已出流形）；
2. `hand_gt170_pred` ≥ **0.09**（标签 0.1138，v2 是 0.096–0.102）；
3. `imag_contrast`（GT vs 零，同 H/扰动）**不下降**（说明 WM 保鲜有效）。

---

## 7. 风险与回退

| 风险 | 触发信号 | 回退 |
|---|---|---|
| critic 放大 WM 偏差 | `imag_return`↑ 但 `token_mse`↑>2×、`hand_gt170`↓ | 关 critic（`--critic 0`）先只做保鲜 |
| WM 保鲜把模型带偏 | 刷新后 `qd_corr` 掉、`imag_contrast` 掉 | 提高 `kl_anchor`（1e-3→1e-2）、降低刷新频率 |
| 纯奖励漂移 | `token_mse` 单调涨、`hand_gt170` 塌 | 打开底模先验界（§2.5 可选）或把 `λ_hand` 提到 0.5 |
| 显存不够 | OOM | `--fm-grad-backend checkpointed`、`--batch-size 4` |
| 训练/部署不一致 | 闭环差 | 部署端必须同一 `--fm-steps 4`、`--decode codec`、同一 WBC 逐字节一致 |

---

## 8. 与 SuperMimic 的逐项对照（这就是"照抄它、把策略换成 VLA"）

| 部件 | SuperMimic（WBC） | v3（VLA） | 状态 |
|---|---|---|---|
| 想象 rollout | WM 里 H=32 | WM 里 H=32 | ✅ 已有，扩到 32 |
| 起点 | replay buffer 真状态 | **真录制状态** + 扰动 | 🔧 2.2 |
| 模型保鲜 | 每 iter 用真转移重训 WM | **每 100 步用真 HA 转移刷新** | 🔧 2.1 |
| critic | GAE(λ) + `γ^H V(s_H)` bootstrap | **同**（token 版） | 🔧 2.3 |
| 梯度 clip | norm 1.0 + value 0.1（小 MLP） | **critic 照抄；head 先测再定** | 🔧 2.6 |
| 奖励 | tracking（reference） | 同（+ 偏离惩罚） | ✅ |
| 标签 | 无（它不用标签 token） | **全身不用标签**，仅手部 | 🔧 2.5 |
| 规模 | 8192 envs × 32 步/iter | batch 8 × grid 4 × H=32（≈100 步/s） | ⚠️ 差 2–3 个量级，需靠 critic 补 |

---

## 9. 不做的事

1. ❌ 不用仿真器在回路里（保持你的前提）；仿真只做**偶发验收**，不进梯度。
2. ❌ 不给 token 加任何标签/BC/锚定项（这是本次的核心约束）。
3. ❌ 不把 `clip_grad_norm_(head, 1.0)` 直接照抄（见 §2.6）。
4. ❌ 不改 token 口径（hold + `ha_cmd`）、不改 K=4、不改 WBC。
5. ❌ 不用 `imag_return` 单独当结论（它是代理；只有 oracle/闭环才是证据）。

---

## 10. oracle 验证结果（2026-10-26，`runs/S3_gates/oracle.json`）

5× 扰动（25 cm / 0.5 rad）、H=32、从偏离起点直接对 64 维 token 做奖励梯度上升（60 步）：

| 批 | GT token | VLA token | **Oracle** | vs GT | vs VLA | oracle 漂移 |
|---|---|---|---|---|---|---|
| #1 | 1.797 | 1.877 | **2.413** | **+34%** | +29% | 0.199（GT 0.060） |
| #2 | 1.815 | 1.838 | **2.392** | **+32%** | +30% | 0.202（GT 0.034） |
| 轻锚定 λ=0.1 | — | — | 2.414 / 2.393 | 同上 | 同上 | **0.183 / 0.187（更低）** |

**读法**：端到端反传在"偏离 regime"里有 **~30% 的天花板**（不是 3.6% —— 之前的 3.6% 是拿"同方向 GT token"当上限定错了）；
而且**轻锚定不损失回报、还能把漂移压低 8%** ⇒ "不学标签"也可以用软约束把 token 摁在流形附近。

---

## 11. 实施记录（v4，2026-10-26）：DR + 模型可信度 + 三个坑

### 11.1 新增/改动（全部已提交）

| 文件 | 内容 |
|---|---|
| `diff_vla_wbc/dr.py`（新） | **域随机化**：每个 batch 按 `nominal 25% / mid 45% / hard 30%` 分层抽扰动幅度（xy≤0.20 m、yaw≤0.25 rad、dq≤0.05、dqd≤0.30），并给出从真转移估量的参考值 |
| `diff_vla_wbc/wm_ensemble.py`（新） | **WM 集成不确定性**：加载已有的 2 个 WM（`WBC_A20000_obs93`、`ckpt_010500_physint_spec`），`u=Var_models(下一帧状态)`；奖励按 `r_t ← r_t·exp(−λ_u·u_t)` 门控（低可信度处不采信） |
| `diff_vla_wbc/v3_step.py` | 接入 `λ_u` / ensemble / 分桶指标（`bucket/{nominal,mid,hard}_{n,rew}`） |
| `diff_vla_wbc/burnin.py` | per-sample 扰动幅度（DR）、`perturb_state`/`_rand` 支持张量、generator 设备对齐 |
| `scripts/train_joint_v2.py` | `--dr/--no-dr`、`--dr-*-max`、`--lambda-u`、`--ensemble`、**固定 WM 快照评测**（`copy.deepcopy`，评测时临时换回），修 `v3_config.json` 的 Path 序列化 |

### 11.2 三个必须记住的坑（都已修）

1. **录制的 root 线/角速度全是 0**（`body_lin_vel_w[:,0].std()==0`）→ 真转移里速度通道是常数，
   WM 保鲜的 `1/std²` 会炸到 1e12。已改成**差分重建**（`np.gradient` 位置、四元数差分角速度），
   并存数据集级 `scales` 供归一化。
2. **WM 保鲜会改变评估尺度**：`ret_gt` 曾从 0.449 漂到 0.155 → 评测必须用**加载时的 WM 快照**。
3. **无约束的纯奖励训练会立刻奖励黑客**（实测 `runs/s3_v3_noBC`，200 步）：
   token 相对标签漂 **30×**、手闭合 **0.0001**，而它的想象回报是 GT 的 **5.8×**。
   ⇒ 这就是为什么 v4 必须带 DR + 不确定性门控；那 200 步留作"无约束"对照。

### 11.3 当前正在跑（v4）

```
runs/s3_v4_noBC   batch 8 × accum 4 · H=32 · K=4 · λ_r=0.3 · λ_hand=0.2 · token BC = OFF
起点 = 真录制状态 + DR 分层（nominal/mid/hard）· 过程推扰 0.03/0.05
λ_u=1.0 + WM 集成（2 个模型）· WM 每 50 步用真转移保鲜 200 步
评测 = 每 100 步 16 窗口（固定 WM 快照）
```
判据（分桶看，硬桶是关键）：
- `bucket/hard_rew` 上升；`imag_return_vla` ↑（固定快照后可比）；
- `token_mse_decoded ≤ 2×` 起点、`hand_gt170 ≥ 0.09`（这两条是防"奖励黑客"的闸门）；
- `unc/mean` 不单调上升（否则说明在往模型外推区跑）；
- `critic/v_ev` 转正（>0.3 更好）。

### 11.4 未做（诚实清单）

- **观测层 DR**（VLA 输入的 46 维状态噪声）只留了 `DRConfig.obs_noise` 字段，未接进 processor 输出——
  需要先确认 46 维分组布局，贸然加噪会伤底模先验；
- **前验软锚定**（`‖token − base_prior‖²`）没有默认开启：它的唯一"无标签"实例化在数值上≈标签
  （base ckpt 的 token_mse vs 标签只有 0.004），所以先用**不确定性门控**代替，等有明确需求再开；
- 闭环验收仍需能跑 Isaac 的容器（本机不可用）。
---

## 12. 口径更正：闭环 chunk **每 40 步**重取（不是 12，也不是 20）

代码事实（`third_party/HumanoidArena/.../action_provider_supermimic.py:_next_latent`）：
只有 chunk 队列**空了**才重新 `infer`，队列长度 = 服务端返回的 40 行。

* 实测：300 步 episode → `logs/ft2000_server2.log` 里 **8 次 infer**（300/40 = 7.5 → 8）✔
* 复核文档 J8 说"每 12 步重取"（源自旧 run 的 300/25）**不成立**；
* ⇒ **一次 chunk 的 40 帧会全部被执行**，所以想象步数应当 **H=40**（而不是 16/32）；
  H=32 只覆盖前 80%，最后 8 帧只受先验锚定、不受奖励。
* ⇒ 回放服务的 `--consume` 也应是 **40**。
