> **2026-11-03 更新（重要）**：本文档的 `linear`（`e/std²`）是 **R0 基线**；
> 最终多轮方案是 **Huber-on-error（`d=√e` + `h(d;δ)`）+ split 朝向 + δ 课程 + critic on + token trust off + action trust 0.2**，
> 以 **`docs/JOINT_MULTIROUND_PLAN_20261103.md`** 为准。
> 本文档的 §2.1（`e_*` 定义）和 §3（回报/总损失/反传）仍然适用；
> **§2.2 的 `e/std²` 只是 R0 基线，新线改成 `h(d;δ)`**（见多轮文档 §2）。

# 联合训练：奖励与损失（唯一口径）

> 2026-11-03。本文件是**当前联合微调**（P4）的唯一权威口径。
> 之前文档里出现过的 `exp` / `hinge` / `huber` / `mse` 都是历史迭代；**新线只用本文档的 `linear` 模式**。
> 依据：`diff_vla_wbc/isaac_joint.py::reward`、`diff_vla_wbc/v3_step.py::train_step`、
> `trackerlab-supermimic/.../mdp/supermimic.py::reward`。

---

## 0. 一句话

**从参考起点出发，用 VLA 生成的 64 维 token 驱动冻结 WBC，再用冻结 WM 想象 20 步；
奖励是每一步"跟踪误差的平方"（MSE），沿 20 步求和后直接反传（BPTT）；
总损失 = 0.3×(-Σr) + token trust + action trust。**

没有 PPO、没有 exp、没有 hinge、没有 critic、没有 BC/手部监督、没有物体奖励（物体奖励只评测）。

---

## 1. 数据流

```
VLA (LoRA + action head)
   │  64-d token π（FM 采样，K=4 步）
   ▼
冻结 WBC（Supermimic policy） ──► a_t ∈ R^29（归一化动作）
   │
   ▼
冻结 WM（8.82M MLP） ──► s_{t+1}, r_t
   │
   └── 20 步展开；loss_r.backward() 沿整条链回到 VLA 的 LoRA/head（BPTT）
```

- WBC 权重冻结；WM 权重冻结；**梯度穿过它们的前向图**（不是 PPO/REINFORCE）。
- 起点：参考状态（`START_FROM=reference`），可选 burn-in 10 步（见 §3.4 of the plan）。
- 窗口：`H=20` 步。

---

## 2. 单步奖励（`reward_mode="linear"`）

### 2.1 `e_*` 的定义（**都是平方误差**，来自 `mdp/supermimic.py::reward`）

| 符号 | 定义 | 单位 |
|---|---|---|
| `e_body_pos` | `mean_b ‖ref_bp_b − bp_b‖²`（30 个 body） | m² |
| `e_body_ori` | `mean_b rot_angle_sq(ref_bR_b, bR_b)` | rad² |
| `e_body_linvel` | `mean_b ‖ref_bv_b − bv_b‖²` | (m/s)² |
| `e_body_angvel` | `mean_b ‖ref_bw_b − bw_b‖²` | (rad/s)² |
| `e_root_pos` | `‖rp − p‖²` | m² |
| `e_root_ori` | `rot_angle_sq(rR, R)` | rad² |
| `r_rate` | `‖(a − a_prev)·scale‖²` | — |
| `r_lim` | `Σ(relu(q − hi) + relu(lo − q))` | rad |

> 关键：`e_*` 已经是**平方**；所以 `w·e/std²` = `w·(error/std)²`，这就是 **MSE**。
> 之前叫它 "linear" 是因为它对 `e` 是线性的（对误差是二次的）。

### 2.2 单步奖励公式（`linear` 分支，新线实际使用）

```text
r_t = -[ w_body_pos  · e_body_pos  / std_body_pos²
       + w_body_ori  · e_body_ori  / std_body_ori²
       + _vs · w_body_linvel · e_body_linvel / std_body_linvel²
       + _vs · w_body_angvel · e_body_angvel / std_body_angvel²
       + w_root_pos  · e_root_pos  / std_root_pos²
       + w_root_ori  · e_root_ori  / std_root_ori² ] · dt
     + w_action_rate · r_rate · dt
     + w_joint_limit · r_lim · dt
```

- `dt = 0.02 s`（CONTROL_DT）。
- `_vs = REWARD_VEL_SCALE`；新线 **0** ⇒ 速度两项乘 0，**不在奖励里**。
  （旧 saturating 模式里另有一段 exp 减法；非 saturating 模式已跳过，不会再混入。）
- `ROOT_ORI_MODE=mono` ⇒ 用上面的 `e_root_ori` 一项；`split`（tilt/yaw/yaw_rate 的 **exp** 核）
  **不用**。
- `TAIL_WEIGHT=0` ⇒ 没有 hinge；`POSTURE_WEIGHT=0` ⇒ 没有 posture 惩罚。

### 2.3 新线的权重 / 尺度（全部写死或显式传）

| 参数 | 值 | 说明 |
|---|---:|---|
| `w_body_pos` | 1.0 | ckpt 默认 |
| `body_pos_std` | **0.05 m** | `--reward-body-pos-std` |
| `w_body_ori` | 1.0 | ckpt 默认 |
| `body_ori_std` | **0.10 rad** | `--reward-body-ori-std` |
| `w_root_pos` | **1.0** | `REWARD_ROOT_WEIGHT=1.0` |
| `root_pos_std` | **0.05 m** | `--reward-root-pos-std` |
| `w_root_ori` | **1.0** | 同上 |
| `root_ori_std` | **0.10 rad** | `--reward-root-ori-std` |
| `w_action_rate` | **-0.1** | ckpt 默认 |
| `w_joint_limit` | **-2.0** | ckpt 默认 |
| `REWARD_VEL_SCALE` | **0** | 速度项关 |
| `TAIL_WEIGHT` | **0** | 无 hinge |
| `POSTURE_WEIGHT` | **0** | 无 posture |

**含义**：误差等于 std 时，该项奖励 = 1（乘 dt 前）；
root 位置 5 cm、root 朝向 0.1 rad（5.7°）、body 位置 5 cm、body 朝向 5.7° 就是"1 分"的尺度。

---

## 3. 回报与策略损失（`v3_step.py::train_step`）

### 3.1 回报

```text
J_b = Σ_{t=0}^{H-1} γ^t · c_t · r_t          H=20, γ=1
c_t = Π_{k≤t}(1 − d_k)                      （survival：摔倒后置 0）
loss_r = −(1/B) Σ_b J_b
```

- `γ=1`（`GAMMA=1.0`）⇒ `J_b = Σ_t c_t r_t`，20 步直接求和；
- `CRITIC_ON=0` ⇒ `boot=0`（没有 `γ^H V(s_H)` 自举）；
- `ADV_BASELINE=none` ⇒ 没有配对基线/advantage。

### 3.2 总损失

```text
L = λ_r · loss_r
  + w_trust · L_trust
  + w_act   · L_act
```

| 项 | 公式 | 参数 |
|---|---|---|---|
| `λ_r` | 0.3 | `LAMBDA_R=0.3` |
| `L_trust` | `mean_b [ relu( RMSE(π_tok − ref_tok) / 0.2 − 1 )² ]` | `TRUST_EPS=0.2`, `TRUST_WEIGHT=1.0`；`ref_tok = batch.port_tokens`（标签） |
| `L_act` | `w_act · mean_{t,b} [ relu( RMS(a_t − a_t^ref) / 0.6 − 1 )² ]` | `ACT_EPS=0.6`, `ACT_WEIGHT=0.6`；`a^ref` 用标签 token 跑同一段 rollout（no_grad） |

**关闭的项**：`LAMBDA_BC=0`、`TOKEN_BC=0`、`LAMBDA_HAND=0`、`LAMBDA_S=0`、`WBC_KL_ANCHOR=0`。

### 3.3 为什么没有 critic

- 目标是**有限时域（H=20）的跟踪和**，`γ=1`，没有"无限期价值"要估；
- critic 只是可选的 value baseline（降低方差），不是算法必需；
- 新线 `CRITIC_ON=0`：`loss_r = -Σ r_t`，最简单、无自举偏差；
- 想开可设 `CRITIC_ON=1`（代码支持），但会引入 `γ^H V(s_H)` 的自举项，需要额外调参。

---

## 4. 反传路径

1. VLA action head 采样 64 维 token（FM，`K=4` 步）；
2. token 作为条件输入冻结 WBC → 29 维动作；
3. 动作输入冻结 WM → 下一状态 + 奖励；
4. 20 步展开，`loss_r.backward()` 的梯度沿 WBC/WM 的前向图回到 token → VLA 的 LoRA + action head；
5. `L_trust`/`L_act` 直接作用在 token/动作上（trust region）。

**不是 RL**：没有 importance ratio、没有 clip、没有 old log-prob、没有 REINFORCE；
是"通过可微世界模型的解析梯度"（BPTT）。

---

## 5. 明确不用的东西

| 东西 | 状态 | 原因 |
|---|---|---|
| `saturating`（`exp(-e/std²)`） | **不用** | 在 cm 级平顶，梯度→0 |
| `huber` | 代码保留，**不用** | 它的 δ/cap 作用在**平方误差**上（δ=0.02 实际是 0.141 m，不是 2 cm）；新线不需要 |
| `root_ori_mode=split` | **不用** | 它是 tilt/yaw/yaw_rate 的 **exp** 核，不是 MSE |
| tail hinge | **不用** | 只对 >0.564 m 生效，20 步窗口不触发 |
| velocity 项 | **不用** | WM 速度通道不准（qd_corr≈0.64） |
| critic / adv baseline | **不用** | 有限时域直接求和 |
| BC / 手部 / state MSE | **不用** | `LAMBDA_BC=LAMBDA_HAND=LAMBDA_S=0` |
| 物体奖励 | **只评测** | 入篮是区域/事件，不是光滑距离 |

---

## 6. 代码位置

| 内容 | 文件 |
|---|---|
| `e_*` 定义 / `mdp.reward` | `trackerlab-supermimic/source/trackerLab/trackerLab/tracker_env/mdp/supermimic.py` (`reward`, 约 170–220 行) |
| 奖励模式（linear/huber/saturating） | `diff_vla_wbc/isaac_joint.py::reward`（约 525–575 行） |
| 回报 / 总损失 / 反传 | `diff_vla_wbc/v3_step.py::train_step`（约 477–790 行） |
| 参数 / 覆盖 | `scripts/train_joint_v2.py`（`--reward-*`、`--lambda-r`、`--trust-*`、`--act-*`） |
| 新线默认值 | `scripts/run_refstart_joint.sh` |
| A' 奖励（同一口径） | `scripts/compute_collect_reward.py --reward-mode linear --reward-root-pos-std 0.05 --reward-root-ori-std 0.10 --reward-body-pos-std 0.05 --reward-body-ori-std 0.10` |

---

## 7. 为什么之前会"一会儿 huber 一会儿 mse"

1. **`e_*` 是平方误差**：我一开始把它当成了距离，于是把 `linear` 当 L1、把 `(e/std)²` 当 MSE——
   实际上 `linear`（`e/std²`）才是误差的平方（MSE），而我加的 `mse` 是四次方（quartic）。**已删除 `mse` 模式**，统一用 `linear`。
2. **huber 的 δ/cap 单位**：它作用在 `e`（平方误差）上，所以 δ=0.02 对应 0.141 m，不是 2 cm；
   plan 里写的"δ=2 cm"是错的。新线**不用 huber**。
3. **`root_ori_mode=split` 是 exp 核**：它和"全 MSE"冲突；新线用 `mono`，走 `e_root_ori/std²`。
4. **代码 guard**：默认禁止 `mono`/`tail=0`；新线用 `--allow-legacy-reward` 显式放行（`posture=0`，不会重新打开旧项）。

---

## 8. 一句话总结

```text
r_t = -[ (e_root_pos/0.05)² + (e_root_ori/0.10)²
       + (e_body_pos/0.05)² + (e_body_ori/0.10)² ] · 0.02
     + (-0.1)·r_rate·0.02 + (-2.0)·r_lim·0.02

J_b = Σ_{t=0}^{19} r_t
L   = 0.3 · (-mean_b J_b)
    + 1.0 · mean_b [relu(RMSE_tok/0.2 - 1)²]
    + 0.6 · mean_{t,b} [relu(RMS_act/0.6 - 1)²]
```

这就是当前联合微调的**全部**奖励与损失；其它模式/项都是历史或可选对照，不参与新线。
