# 设计文档：联合微调的视界与监督（**H=20 / `loss_s` 已废弃**；现行方案 = 奖励内铰链）

> **⚠️ 2026-10-30 04:30 重大修订 —— 本文档原方案（H=20 + 独立 `loss_s`）已被实测否掉，保留原文仅作记录。**
>
> **现行方案（定稿）**：
> 1. **不做 `loss_s`**：它与"奖励里加铰链"是同一个机制（`relu(e−e₀)`），只走一条路
> 2. **不做 `H=20`**：实测 20-tick 开环窗口内误差**平坦**（`pos 1.00×`、`ori 1.04×`），9–20 帧不累积 ⇒ 无收益、成本 2.4×
> 3. **奖励内铰链**（已实现并实证）：`root_pos`（e₀=0.35 m）、`root_ori`（e₀=20°）、**新增分组 `body_pos`**、
>    以及 `tilt` / pelvis 高度每步惩罚
> 4. **`root_ori`（yaw 部分）暂缓**：93 维观测里没有"相对参考的根朝向"（99 维版有最后 6 维 rot6d），
>    而**部署时也拿不到那 6 维** ⇒ yaw 全程不可观测（实测占 ori 误差的 7.53°/10.24°），
>    惩罚它是注入噪声。**怎么处理 ori 待定**（见 §12）。
>
> 已验证：`root_pos` + `root_ori` 铰链把 `pos p90` 从 **52.1 → 26.9 cm（−48%）**、`pos p50` 5.31 → **4.41 cm**、`fell 0/80`
> （铰链臂 80 集，seed=2，从 π1 起训）。
>
> ---
>
> ## 原文（历史记录，勿照做）


> 版本：2026-10-30 03:55 · 状态：**待执行**（本文档只描述改动，不含已生效代码）
> 目标：让 8/9 联合微调**真正覆盖一次开环执行的全部 20 帧**，并给"跟踪误差"一条**非饱和**的梯度通道。

---

## 0. 一句话结论

| 项 | 现状 | 改成 |
|---|---|---|
| rollout 长度 | **8**（`HA_VLA_EXEC_STEPS=20`，只覆盖 40%） | **20** |
| `loss_r` 奖励项 | 前 8 帧（第 9–20 帧的奖励**算都不算**） | **全部 20 帧** |
| critic GAE 目标 | 8 步 | **20 步** |
| bootstrap | `γ⁸·(V(s₈)−V(ref₈))` | `γ²⁰·(V(s₂₀)−V(ref₂₀))` |
| `loss_s`（直接反传误差） | **v3 里空转**（只在 v2 的 `train_step` 被读） | **20 步、Huber 形状、复用奖励的 6 个误差项** |
| 关节角监督 | — | **不做**（手臂由 VLA 决定，见 §3.4） |

---

## 1. 为什么要改：三条实测证据

### 1.1 训练视界只有部署窗口的 40%

- 部署：`HA_VLA_EXEC_STEPS = 20`（`v3_step.py:76`、`ha_replay_token_server.py:8`）—— 一次 VLA 查询给出 token chunk，之后**开环执行 20 帧（0.4 s）**才重规划。
- 训练：`v3_step.py:367` `horizon=cfg.horizon`，`J1_H=8`。
- ⇒ 9–20 帧这一段，**奖励从来没被计算过**。

### 1.2 critic 那一段的带宽只有"1 个标量 × 0.1 梯度"

```python
ret  = Σ_{t=0}^{7} γ^t · r_t                       # 只对 8 个奖励求和
boot = γ^8 · survival · ( V(s₈) − V(ref₈) )        # 12 帧压成 1 个数
loss_r = -(ret + boot).mean()
# _sg = critic_state_grad_scale = 0.1（v3_step.py:98/382）⇒ 经 critic 回的梯度再 ×0.1
```

critic 能表达"长时程偏好"，**表达不了"9–20 帧每帧该怎么做"**。

### 1.3 奖励的形状对尾部结构性失明

`exp(-e/σ²)` 的均值。按误差分桶实测（`j1_scene2/round_5` 的 80 集）：

| 误差区间 | 占时刻 | 提供的位置梯度 |
|---|---|---|
| 0–0.1 m | 72.4% | **86.3%** |
| >0.4 m | **21.2%** | **0.22%** |

后果（同批 80 集，按集自助重采样 250 次的 95% CI）：

| 产物 | pos p50 (cm) | pos p90 (cm) | ori p50 (°) |
|---|---|---|---|
| π0 未微调 | 5.79 [5.40, 6.28] | **24.9** | 10.65 |
| π2 | **4.11** [3.67, 4.73] ✅ | 59.6 ❌ | 10.35 |
| π5 | 5.95 ❌ | 58.4 ❌ | **13.51** ❌ |

⇒ **中位真的能压下去（显著），但尾部同时翻 2.2 倍** —— 这是该目标的最优解，不是训练跑偏。

### 1.4 `lambda_s` 在 v3 里空转（与 `lambda_bc` 同一类坑）

`v3_step.py` 里 `lambda_s` 出现 **0 次**；它只在 `train_v2.py:345` 的 `JointTrainerV2.train_step` 被读，而 v3 覆写了 `train_step`。
⇒ `--lambda-s 1.0` 传进去**不产生任何梯度**，但 `v2_config.json` 照记。

### 1.5 库存的 `v_root`/`w_root` 通道是坏的

`data/library/ft318_compact.pt` 全库 **473,876 帧**实测 `|v_root| = |w_root| = 0.000000`（max 也是 0）。
⇒ 任何以状态 `[36:42]` 为目标的速度监督都会**教策略"别动"**。用根位置差分重建可得到 p99 = 0.269 m/s，但 90% 的帧为 0，信息量低。
**结论：`loss_s` 不碰状态里的速度通道**，改用奖励里那 6 个误差项（其速度项来自 `library.bodies(f)` 的 FK，**不是**那个坏通道）。

---

## 2. 核心设计：`loss_s` = 奖励跟踪项的 **Huber 版**

不给 `loss_s` 重新发明一套误差定义，而是**直接复用奖励里被 `exp` 包住的那 6 个原始误差**。

### 2.1 奖励的 6 个误差（`rewards.py` 里已有）

```python
e_root_pos = ((ma_pos - ra_pos) ** 2).sum(-1)              # ‖Δp_root‖²
e_root_ori = _sq_quat_err(ma_quat, ra_quat)                # θ_root²（平方测地角）
e_body_pos = ((rel_pos - robot_body_pos_w) ** 2).sum(-1).mean(-1)   # 14 body 的 ‖Δp‖² 均值
e_body_ori = _sq_quat_err(rel_quat, robot_body_quat_w).mean(-1)     # 14 body 的 θ² 均值
e_body_lin = ((ref_body_lin_vel_w - robot_body_lin_vel_w) ** 2).sum(-1).mean(-1)
e_body_ang = ((ref_body_ang_vel_w - robot_body_ang_vel_w) ** 2).sum(-1).mean(-1)
```

奖励把它们变成 `exp(-e_x / σ_x²)`；**`loss_s` 把它们变成 `Huber(√e_x / σ_x)`**。

### 2.2 公式

```
               1    6
loss_s(t) =  ───  Σ   w_x · Huber( √e_x(t) / σ_x , δ )
               6   x=1

Huber(z, δ) = 0.5·z²                 if |z| ≤ δ
            = δ·(|z| − 0.5·δ)       otherwise        （δ = 1.0，单位：σ）
```

`σ_x` 与 `w_x` **直接取奖励自己的 `std_*` / `w_*`**：

| 误差项 | σ_x（= 奖励的 std） | w_x | 含义（1σ） |
|---|---|---|---|
| `root_pos` | 0.20 m | 1.0 | 根位置偏 20 cm |
| `root_ori` | 0.25 rad (14.3°) | 1.0 | 根朝向偏 14° |
| `body_pos` | 0.30 m | 1.0 | 14 个 body 的 RMS 位置偏 30 cm |
| `body_ori` | 0.40 rad (22.9°) | 1.0 | body RMS 朝向偏 23° |
| `body_lin` | 1.00 m/s | 1.0 | body 线速度 RMS 差 1 m/s |
| `body_ang` | 3.14 rad/s | 1.0 | body 角速度 RMS 差 3.14 rad/s |

**为什么必须用奖励自己的 σ**：这样 `loss_s` 与 `loss_r` 的差别**只剩形状**（`exp` vs `Huber`），没有第二套尺度要去调，也不存在"两个项互相打架"的风险。

### 2.3 形状对比（这才是全部收益来源）

以根位置误差 d 为例（σ = 0.2 m）：

| d | 奖励项 `exp(-d²/σ²)` | 奖励梯度 `2d/σ²·exp(-d²/σ²)` | **Huber 项** | **Huber 梯度** |
|---|---|---|---|---|
| 5 cm | 0.939 | 2.35（最强） | 0.031 | 0.50 |
| 14 cm (=σ/√2) | 0.61 | 3.3（峰值） | 0.245 | 1.40 |
| 20 cm (1σ) | 0.368 | 1.84 | 0.50 | 1.00（拐点）|
| 40 cm (2σ) | 0.018 | 0.37 | 1.50 | **1.00（不再衰减）** |
| 80 cm (4σ) | 3e-6 | ~0 | 3.50 | **1.00** |
| 200 cm | ~0 | ~0 | 9.50 | **1.00** |

**关键：Huber 超过 1σ 后梯度恒为 1.0（不衰减），奖励则是指数衰减到 0。**

### 2.4 为什么用 Huber 而不是纯 MSE

我们数据里有 `max |Δp| = 2.7 m` 的样本。纯 MSE 的梯度 `2d/σ²` 在 2.7 m 处是 **135/σ**，会把单条极端样本变成主梯度；Huber 在 1σ 之后恒定，**不会被一条样本带偏**，同时还保留了"越远越拉"的方向（因为损失值仍线性增长）。

---

## 3. 五条设计理由（逐条对应前面某次分析）

### 3.1 帧率对齐
`HA_VLA_EXEC_STEPS = 20` ⇒ rollout 20。**既然要滚 20 步，`loss_r` 用前 8 步等于白扔 60% 的计算**。

### 3.2 锚定一致（最容易踩的坑）
奖励的 body 项**不是世界系绝对位置**，而是把参考重新锚定到机器人自己身上（`reanchor_reference`，`rewards.py:69`）：

```python
delta_pos = [robot_anchor_x, robot_anchor_y, ref_anchor_z]     # xy 取机器人、z 取参考
delta_ori = yaw(robot_anchor ⊗ ref_anchor⁻¹)                   # 只取 yaw
rel_pos   = delta_pos + R(delta_ori)·(ref_body_pos − ref_anchor_pos)
```

⇒ `motion_body_pos` 量的是**相对自身根部的姿态误差**，全局位置另由 `motion_global_root_pos` 管。
**`loss_s` 必须复用同一套误差（同一个 `reanchor_reference`），否则两个项会在不同的坐标系里互相拉扯。**
这就是为什么本文档选择"抽出奖励的误差"而不是"在 loss 里重算 body 位置"。

### 3.3 根位置**和**根朝向都要有
用户明确要求。上面的 6 项里 `root_pos`、`root_ori` 都在，且与奖励**同一坐标系、同一 σ**。
> ⚠️ 纯 body-keypoint 的做法（DeepMimic/PHC 风格）会因为 §3.2 的重锚定而**丢掉全局根位置误差** —— 所以**不能只做 body**，必须 `root_pos + root_ori + body×4` 六项齐全。

### 3.4 不监督关节角
`arm=vla` 时手臂**故意**偏离参考：实测 `right_shoulder_pitch` 偏差 **+34.7°（系统性偏置）**、`left_shoulder_pitch +33.6°`。
若用 29 维关节角 MSE，等于把"VLA 的正常工作方式"当误差惩罚。
而 body 项里 `shoulder_roll / elbow / wrist_yaw` 已经被覆盖（奖励本来就这样），34° 肩偏置折算成腕部位移 ~0.17 m，在 σ=0.3 下只是 **0.57σ** —— 与奖励的惩罚力度**完全一致**，不额外打压。
⇒ **`loss_s` 不含关节角项。**

### 3.5 避开坏的速度通道
`loss_s` 用奖励的 `body_lin/body_ang`（来自 `library.bodies(f)` 的 FK，正常），**不碰状态 `[36:42]`**（全零，见 §1.5）。

---

## 4. 逐文件改动

### 4.1 `third_party/SuperMimic-Reproduced/src/supermimic/envs/mdp/rewards.py`

**改动 1**：抽出 `tracking_errors()`，`tracking_reward_terms` 改为调用它（**行为逐位不变**）。

```python
def tracking_errors(
    *, robot_body_pos_w, robot_body_quat_w, robot_body_lin_vel_w, robot_body_ang_vel_w,
    ref_body_pos_w, ref_body_quat_w, ref_body_lin_vel_w, ref_body_ang_vel_w,
    anchor_index, joint_pos, soft_limits, action, prev_action,
) -> dict[str, torch.Tensor]:
    """返回奖励里被 `exp` 包住的**原始误差**（均为"平方"形式），供 loss_s 用 Huber 复用。

    与 `tracking_reward_terms` 共用同一份 `reanchor_reference` ⇒ 坐标系/锚定完全一致。
    """
    # ... 直接把 tracking_reward_terms 里 99-127 行那段原样搬进来 ...
    return {"root_pos": e_root_pos, "root_ori": e_root_ori,
            "body_pos": e_body_pos, "body_ori": e_body_ori,
            "body_lin": e_body_lin, "body_ang": e_body_ang}
```

`tracking_reward_terms` 内部改成：

```python
err = tracking_errors(...)
terms = {
    "motion_global_root_pos": dt * cfg.w_root_pos * torch.exp(-err["root_pos"] / cfg.std_root_pos**2),
    ...（其余同构）...
}
```

**改动 2**：`TrackingRewardCfg` 已含（本次已加，保留）
- `tail_pos_weight/e0/cap`、`tail_ori_weight/e0_rad/cap_rad`（尾部铰链）
- `w_tilt / tilt_e0_rad / w_height / height_min`（倾斜/高度每步惩罚）

### 4.2 `.../supermimic/world_model/heads.py`

新增透传（`SupermimicHeads` 里）：

```python
def reward_errors(self, s_next, ref, bodies=None):
    b = bodies if bodies is not None else self.robot_bodies(s_next)
    return tracking_errors(
        robot_body_pos_w=b["pos"], robot_body_quat_w=b["quat"],
        robot_body_lin_vel_w=b["lin_vel"], robot_body_ang_vel_w=b["ang_vel"],
        ref_body_pos_w=ref.body_pos_w, ref_body_quat_w=ref.body_quat_w,
        ref_body_lin_vel_w=ref.body_lin_vel_w, ref_body_ang_vel_w=ref.body_ang_vel_w,
        anchor_index=self.anchor_index, joint_pos=s_next[..., 7:36],
        soft_limits=self.soft_limits, action=None, prev_action=None,
    )
```

### 4.3 `diff_vla_wbc/mjlab_joint.py` + `supermimic_backend.py`

各加一个同名透传（与现有 `reward` / `robot_bodies` 同构）：

```python
# MjlabJointBundle
def reward_errors(self, state, reference, bodies=None):
    return self.assets.heads.reward_errors(state, reference, bodies)
```

> 目的：训练侧只需要 `bundle.reward_errors(s_pred, ref)`，**不需要知道后端的 FK/锚定细节**。

### 4.4 `diff_vla_wbc/rollout.py`

**改动 1**：保留已做的 `_state_error` 速度拆分（`vel_joint` / `vel_root`），但**本方案不再使用它**（留作离线诊断工具）。

**改动 2**：新增 Huber 版跟踪损失

```python
def tracking_huber_loss(bundle, rollout, *, delta: float = 1.0, weights=None, sigmas=None):
    """把奖励的 6 个原始误差换成 Huber 形状，对 rollout 的**每一步预测状态**求平均。

    形状：Huber(√e_x/σ_x, δ)，σ 与 w 默认复用奖励自己的 std_*/w_*。
    """
    H = len(rollout.references)
    terms = ("root_pos", "root_ori", "body_pos", "body_ori", "body_lin", "body_ang")
    w = weights or {"root_pos": 1.0, "root_ori": 1.0, "body_pos": 1.0,
                    "body_ori": 1.0, "body_lin": 1.0, "body_ang": 1.0}
    sg = sigmas or {"root_pos": 0.20, "root_ori": 0.25, "body_pos": 0.30,
                    "body_ori": 0.40, "body_lin": 1.00, "body_ang": 3.14}
    total = 0.0
    per_term = {k: [] for k in terms}
    for t in range(H):
        ref = rollout.references[t]
        err = bundle.reward_errors(rollout.states[t + 1], ref)     # 与奖励同一锚定
        for k in terms:
            z = err[k].clamp_min(0.0).sqrt() / sg[k]
            h = torch.where(z.abs() <= delta,
                            0.5 * z * z,
                            delta * (z.abs() - 0.5 * delta))
            per_term[k].append(h.mean())
            total = total + w[k] * h.mean()
    n = len(terms)
    out = {f"loss/state_{k}": float(torch.stack(v).mean().detach()) for k, v in per_term.items()}
    out["loss/state"] = float((total / n).detach())
    return total / n, out
```

### 4.5 `diff_vla_wbc/v3_step.py`

**改动 1**：rollout 长度用 `state_horizon`

```python
_S = int(getattr(cfg, "state_horizon", 0) or 0) or cfg.horizon
rollout = imagine_with_tokens(..., horizon=_S, ...)        # 原为 cfg.horizon
H = min(cfg.horizon, _S)                                   # 奖励只在前 H 步上取"价值"
```

**改动 2**：`loss_r` / `boot` 在**前 `H_r` 步**上算，`H_r = min(cfg.horizon, _S)`

```python
H_r = min(int(cfg.horizon), _S)
ret  = discounted_return(rollout, v3.gamma, h=H_r)        # 需给 discounted_return 加 h 参数
ref_states = self._ref_states(batch)                      # 长度 = H_r（见下方 ⚠️）
vH   = self.critic(rollout.states[H_r], ref_states[H_r - 1], xi)
v_ref= self.critic(ref_states[H_r - 1], ref_states[H_r - 1], xi)
boot = (v3.gamma ** H_r) * rollout.survival[H_r] * rel
```

> **默认实验用 `--horizon 20 --state-horizon 20` ⇒ `H_r = 20`，即奖励项也覆盖全部 20 帧**（用户要求）。
> 而代码写成 `H_r = min(...)` 的好处：§8 的退路 `--horizon 8 --state-horizon 20` **只是一个参数改动**，
> 奖励回到 8 步、监督仍是 20 步。

> ⚠️ **实现时必须先核对 `_ref_states` 的长度**：`v3_step.py:308-310` 里
> `H = len(batch.references); offs = arange(1, H+1)` ⇒ 返回的是 `frames+1 .. frames+H`，即 **H 个**状态
> （docstring 写的是 `(H+1,B,71)`，与实际可能不符）。若确实只有 H 个，则 `ref_states[H_r-1]` 就是
> 第 `H_r` 步的参考态 —— 与 `rollout.states[H_r]` 对应。**这一步用一个断言钉住**：
> `assert ref_states.shape[0] == len(batch.references)`。

**改动 2b**：`discounted_return` 加一个可选 `h`

```python
def discounted_return(rollout, gamma=0.99, bootstrap=None, h: int | None = None):
    h = rollout.rewards.shape[0] if h is None else int(h)
    ...（其余不变，把 rollout.rewards.shape[0] 换成 h）...
```

**改动 3**：`loss_s` 换成新的 Huber 版（替换掉我之前接的 `_state_error` 版本）

```python
if float(getattr(self.cfg, "lambda_s", 0.0)) > 0:
    from .rollout import tracking_huber_loss
    loss_s, sm = tracking_huber_loss(
        self.bundle, rollout,
        delta=float(getattr(self.v3, "state_huber_delta", 1.0)),
    )
    loss = loss + float(self.cfg.lambda_s) * loss_s
    metrics.update(sm)
```

**改动 4**：AUDIT 行打印 `lambda_s / state_horizon / huber_delta`，并在 `lambda_s > 0` 时确认 `teacher_states` 存在（否则 `_ref_states` 会静默降级）。

### 4.6 `scripts/train_joint_mjlab.py`

```python
ap.add_argument("--state-horizon", type=int, default=0,
                help="监督/critic 用的 rollout 长度；0=与 --horizon 相同。"
                     "建议 20（= HA_VLA_EXEC_STEPS）")
ap.add_argument("--lambda-s", type=float, default=0.0, help=...)   # 已存在，改 help
ap.add_argument("--state-huber-delta", type=float, default=1.0)     # 新
# h_req 必须把 state_horizon 算进去（否则 teacher_states 只有 horizon 帧）
h_req = max(int(args.horizon), int(args.eval_horizon or 0), int(args.state_horizon or 0))
h_eval = max(int(args.horizon), int(args.eval_horizon or 0), int(args.state_horizon or 0))
# ⚠️ 还有两处 provider 必须用 h_req 建，否则 batch.references 只有 horizon 条：
#    train_joint_mjlab.py:471  provider     = _mk_provider(int(args.horizon), ...)  → h_req
#    train_joint_mjlab.py:527  ref_provider = _mk_provider(int(args.horizon), ...)  → h_req
#    以及 :473  eval_provider = _mk_provider(h_eval, ...)（已是 h_eval，正确）
cfg = JointV2Config(..., lambda_s=args.lambda_s, state_horizon=int(args.state_horizon))
v3cfg = V3Config(..., state_huber_delta=float(args.state_huber_delta))
```

> ⚠️ `h_req` 那两行是**必须改的**：`teacher_states` 只有 `h_req` 帧，若 `state_horizon=20` 而 `h_req=8`，`_ref_states` 会在第 9 步越界或静默截断。

### 4.7 `scripts/run_j1_rounds.sh` / `launch_j1_rounds.sh`

```bash
J1_STATE_HORIZON=${J1_STATE_HORIZON:-20}
J1_LAMBDA_S=${J1_LAMBDA_S:-0}          # 已加
J1_HUBER_DELTA=${J1_HUBER_DELTA:-1.0}
# 第 8/9 步命令里追加：
--state-horizon "$J1_STATE_HORIZON" --lambda-s "$J1_LAMBDA_S" --state-huber-delta "$J1_HUBER_DELTA"
```

---

## 5. 权重、量纲与标定

### 5.1 量纲陷阱（必须记住）

`ret = Σ_{t<S} γ^t r_t` 是**求和**，`S` 变了它的量级就变：

| S | `ret`（r≈0.088, γ=0.99） |
|---|---|
| 8 | ≈ 0.68 |
| **20** | ≈ **1.60（2.4×）** |

⇒ `λ_r·loss_r` 从 0.195 变成 **0.48**。所以 `λ_s` 必须按新量级定，不能沿用旧值。

### 5.2 建议初值

| 参数 | 值 | 理由 |
|---|---|---|
| `lambda_r` | **0.3**（不变） | 奖励项自身的相对权重不动 |
| `lambda_s` | **0.3** | 估算：中位样本 `loss_s ≈ 0.6–1.0` ⇒ `λ_s·loss_s ≈ 0.2–0.3`，约为奖励项 0.48 的 40–60%；尾部（5σ）`loss_s ≈ 27` ⇒ `≈ 8`，是奖励项的 **17 倍** |
| `state_huber_delta` | **1.0** | δ=1σ：1σ 内保持平方（细粒度），之外线性（稳） |
| `state_horizon` | **20** | = `HA_VLA_EXEC_STEPS` |
| `gamma` | 0.99（不变） | γ²⁰ = 0.818，20 步仍有折扣但可接受 |
| `critic_state_grad_scale` | 0.1（不变） | 防 critic 主导 |
| `lambda_bc` | 0 | v3 不读，保持 0 以免配置撒谎 |
| `boot_one_sided` | **False** | 未证实有益 |

### 5.3 标定流程（不拍脑袋）

1. 跑 **20 步**，读 metrics 的 `loss/state` 与 `loss/reward_raw`；
2. 计算 `λ_r·loss_r / (λ_s·loss_s)`，目标 **0.4–0.6**（中位样本）；
3. 按此反算 `λ_s`，再正式跑 800 步。

---

## 6. 新增 metrics（必须能看见每一层）

| key | 含义 |
|---|---|
| `loss/state` | Huber 损失（6 项平均） |
| `loss/state_root_pos` / `_root_ori` | **根位置 / 根朝向**分项（用户明确要求可见） |
| `loss/state_body_pos` / `_body_ori` | 14 body 位置 / 朝向分项 |
| `loss/state_body_lin` / `_body_ang` | body 速度分项（来自 FK，非坏通道） |
| `loss/reward_raw` | `ret` 均值（已存在） |
| `loss/boot` | bootstrap（已存在） |
| `critic/v_adv_H` / `_raw` | 已存在 |
| `rollout/reward` | 已存在（注意：这是**奖励值**，跨 H 不可比） |

---

## 7. 执行方式（起点、判据、验证顺序）

### 7.1 起点：**π0**（不是 π1）

```
CUR_WBC  = third_party/SuperMimic-Reproduced/logs/wbc_ft318_mjlab/ckpt_0014000.pt
VLA_CKPT = /inspire/hdd/.../flowbody/FlowBody/runs/groot_sonic_vla_20k_frozen271_20261023/checkpoint-20000
```

理由：π1 的 `pos p90` 已经是 **52.1 cm（已坏）**，从它起等于"要求修复"，从 π0 起才是"要求别弄坏 + 把中位压下去"。π0 的 80 集基线：`pos p50 5.79 / p90 24.9 / ori p50 10.65 / ori p90 24.7 / fell 0/80`。

### 7.2 判据（四条全过才算成功）

| 指标 | π0 基线 | 通过线 |
|---|---|---|
| `pos p50` | 5.79 cm | **≤ 5.2**（要有真改善，且 CI 与 π0 不重叠）|
| `pos p90` | 24.9 cm | **≤ 30**（不许弄坏）|
| `ori p50` | 10.65° | **≤ 10.0** |
| `fell` | 0/80 | **0/80** |

### 7.3 验证顺序（每步不过就停，不要往下跑）

1. **语法/接线**：`ast.parse` + 真实 `import` + `--help` 里有新参数；
2. **20 步 smoke**：确认 `loss/state`、`loss/state_root_ori` 等 key 出现且**非零**；确认 `loss/state` 量级与 §5.2 估算一致；
3. **800 步单臂**（约 70 min）；
4. **80 集采集**（`scripts/collect_bline_arm.sh`，约 25 min）→ 按 §7.2 判；
5. 过则作为新基线继续，不过则回滚并定位（看 §6 哪个分项在涨）。

---

## 8. 风险与回滚

| 风险 | 表现 | 处置 |
|---|---|---|
| **20 步策略梯度不稳** | grad norm 爆炸、`rollout/reward` 崩 | `--horizon 8 --state-horizon 20`（奖励仍 8 步、监督 20 步）—— 这一组合**无需改代码**，只改参数 |
| **Huber 项过大** | `loss/state` 主导，动作变保守、中位退化 | 降 `lambda_s`（0.3 → 0.1 → 0.03） |
| **WM 20 步保真度不足** | `loss/state` 卡在 1–2 cm 下不去 | 这是地板，不是 bug；停止追求更小值 |
| **`h_req` 忘改** | 第 9 步之后参考状态越界/截断 | §7.3-1 的 `--help` + smoke 必须验证 `state_horizon` 生效 |
| **配置撒谎** | `v2_config.json` 记了但代码不读 | 已在 v3 AUDIT 行打印 `lambda_s / state_horizon`；`lambda_s>0` 时断言 `teacher_states is not None` |

**回滚**：本文档所有改动都在 6 个文件里，且**默认值保持旧行为**（`state_horizon=0`→等于 `horizon`；`lambda_s=0`→完全关闭）。所以"回滚"= 不起用这两个参数即可，无需 revert 代码。

---

## 9. 成本

| 项 | 现在 | 改后 |
|---|---|---|
| 8/9 微调（800 步） | ~30 min | **~70 min**（rollout 8→20，约 2.4×）|
| 80 集判据采集 | ~25 min | 不变 |
| **单臂总周期** | ~55 min | **~95 min** |

节流方案（若嫌慢）：**每 4 步做一次 20 步、其余步 8 步** ⇒ 单轮 ~42 min，仍需 §4.5 的同一套代码（只是 `_S` 按步号切换）。默认**先走全步 20**。

---

## 10. 与已有结论的冲突检查

| 已记录结论 | 是否冲突 | 说明 |
|---|---|---|
| A4：`H8 > H16 > H32 > H40` | ⚠️ **不冲突但需注明** | A4 是"固定 400 个优化步"比的，H 越大每步越贵 ⇒ 混着算力差；且当时 WM 远不如现在（现在 k=20 实测 `关节 0.5° / 根位 0.5–1.8 cm / qd_ratio 0.99`）。**若 20 步真的不稳，§8 给了不用改代码的退路。** |
| A5：`head_plus_wbc` 更好 | 不冲突 | 那是 RL 穿 WM 训 WBC，与本次无关 |
| WM 保鲜 A/B | 不冲突 | 本次不开 `WM_REFRESH_EVERY` |
| `std_root_ori` 0.15 无效且有害 | 不冲突 | 已回滚 0.25 |
| 库存 `v_root/w_root` 恒 0 | **已规避** | `loss_s` 不碰状态速度通道（§3.5）|
| `lambda_s` / `lambda_bc` 在 v3 空转 | **已修** | 本次把 `lambda_s` 真正接进 v3 |

---

## 11. 待办（本次不做，记在案）

1. **`wm_refresh.py` 的 `ori` 缩放缺陷**：`(R*S).sum(-1).sum(-1)` 对 (B,4) 等于把整个 batch 求和，
   实际是 `0.5·(3−Σ_b cos)`，梯度是应有值的 **B 倍**；`mag_weight=30` 是在这个错误缩放下调出来的，
   **改它必须重新调参** ⇒ 下次重构第一件事。
2. 逆动力学蒸馏（WM 只当标注器、退出梯度路径）：`a* = argmin_a ‖WM(s_real,a) − s_ref_next‖²`。
3. 缩短部署执行块（20 → 8 tick），让训练视界与部署窗口天然相等 —— 需先测 VLA 能否 6.25 Hz 出 token。

---

## 12. 2026-10-30 04:35 现行定稿（覆盖 §0–§11）

### 12.1 已完成的改动（除 ori 外全部改完）

| 改动 | 位置 | 状态 |
|---|---|---|
| `root_pos` 尾部铰链（λ=2, e₀=0.35 m, cap 1.0 m） | `rewards.py` | ✅ 已实现，实证 −48% 尾部 |
| `root_ori` 尾部铰链（λ=2, e₀=20°, cap 57°） | `rewards.py` | ✅ 已实现（yaw 问题待定，见 12.3）|
| **`body_pos` 分组铰链** | `rewards.py` | ✅ **本次新增** |
| `tilt`（e₀=20°）+ pelvis 高度（h_min=0.70 m）每步惩罚 | `rewards.py` | ✅ 已实现 |
| WM 阶段 2b（k=20, 1500 步, sim 交织 0.3） | `pretrain_wm.py` | ✅ 已实现，双门逐轮改善 |
| 坏集过滤（摔/倾角>45° 剔除出 `wm_mix`） | `bline_to_transitions.py` + `bline_health.py` | ✅ |
| 80 集健康硬门（`exit 8`） | `run_j1_rounds.sh` | ✅ |
| 断点续采 + `wait || true`（防整链被带走） | `run_j1_rounds.sh` | ✅ |
| `RESUME_FROM_STEP`（奖励实验复用 1–6 步） | `run_j1_rounds.sh` | ✅ |
| **固定起点评测集（`BLINE_EVAL_SEED=999` × 5 集/任务）** | `run_j1_rounds.sh` §9.5 + `collect_bline_arm.sh` | ✅ **本次新增** |
| `J1_LAMBDA_S` / `J1_STATE_VEL` / `--state-horizon` 接线 | `run_j1_rounds.sh` / `train_joint_mjlab.py` | ⚠️ 已接线但**默认关**，本轮不用 |

### 12.2 `body_pos` 分组铰链：为什么必须分组

实测（铰链臂 80 集，重锚定后的逐 body 位置误差）：

| body | 中位 | p90 | |
|---|---|---|---|
| right_wrist_yaw | **0.195 m** | 0.336 | 手臂 |
| right_elbow | 0.134 | 0.261 | 手臂 |
| left_wrist_yaw | 0.133 | 0.306 | 手臂 |
| left_elbow | 0.115 | 0.272 | 手臂 |
| right_ankle_roll | 0.072 | 0.114 | |
| 膝 | 0.050–0.059 | 0.09–0.10 | |
| pelvis / 髋 | 0.024–0.032 | 0.034–0.044 | |

**手臂 6 body 平均 0.126 m，腿+躯干 8 body 平均 0.046 m（差 2.8 倍）。**
若用"14 body 平均 + 单一阈值"，铰链**只会去拽手臂** —— 而那 0.13 m 是 `arm=vla` 的正常偏置
（肩俯仰 **+34.7°**），不是缺陷。所以按组各用自己尾部：

| 组 | bodies | 当前 p50 / p90 | `e₀` | `λ` |
|---|---|---|---|---|
| 腿+躯干 | 索引 0–7（8 个） | 0.046 / ~0.10 | **0.10 m** | 2.0 |
| 手臂 | 索引 8–13（6 个） | 0.126 / ~0.30 | **0.25 m** | 2.0 |

`TRACKED_BODIES` 顺序：`0 pelvis, 1-3 左腿, 4-6 右腿, 7 torso, 8-13 左右肩/肘/腕`。

**回归验证**：改动前后，原有 8 个 term（`motion_global_root_pos/ori`、`motion_body_pos/ori/lin_vel/ang_vel`、
`action_rate_l2`、`joint_limit`）在随机输入上**逐位相同**（只有刻意置零的铰链项不同）。

### 12.3 `root_ori` 待定（本次不动）

- `motion_global_root_ori` 用的是**全球系**四元数差，含 yaw；而 14 个 body 的朝向项经
  `reanchor_reference` 的 `yaw_quat` 已经把 yaw 消掉 ⇒ **同一个奖励里 body 项不管 yaw、root 项管 yaw，自相矛盾**
- 93 维观测里**没有** yaw 信息（`proj_grav` 只能给 roll/pitch），99 维版那最后 6 维
  `rot6d(RᵀR_ref)` 正是"相对参考的根朝向"，**但部署时拿不到**（VLA token 里也不带）⇒ 才被砍掉
- 实测 ori 误差分解：**yaw 7.53°（p90 24.7°）**、pitch 5.34°（偏置 4.59°）、roll 1.85°
  ⇒ **10.24° 里约 7.5° 不可观测**

**三个候选（待用户定）**：
1. **把 root ori 改成 yaw-invariant**（只比 `acos(up·up_ref)²`），与 body 项口径一致。
   代价：评测口径也要跟着拆成 `tilt` 和 `yaw`，否则会一直盯着修不了的数。
2. **把 6 维喂给 VLA head 的 state 输入** —— 但部署同样拿不到，**除非部署侧能提供**。
3. **接受** 7.5° yaw，只优化 pitch/roll。

### 12.4 十轮定稿配方（ori 待定，其余已定）

| 项 | 值 |
|---|---|
| 起点 | **π0**（W1 + 底模 VLA `groot_sonic_vla_20k_frozen271/checkpoint-20000`）|
| `H` | 8 |
| 奖励 | root pos/ori 铰链 + **分组 body_pos 铰链** + tilt/高度 |
| WM | 阶段 1+2+2b（k=20）|
| `loss_s` / `H=20` | ❌ 都不做 |
| `lambda_bc` / `boot_one_sided` | 0 / False |
| 训练 B 线 | `seed=$R`（多样性）|
| **固定评测集** | `seed=999` × 5 集/任务 = 20 集/轮 |
| 判据 | `pos p50 / pos p90 / tilt / fell`（yaw 只报不判）|

### 12.5 root 朝向拆分（2026-10-30 05:50，已实现并验证）

**为什么改**：在官方 `unitree_g1_sonic` 的 VLA 路径里，`state` 就是 8 项
（`left_leg/right_leg/waist/left_arm/right_arm/left_hand/right_hand/projected_gravity`），
**没有 `motion_anchor_orientation`**（anchor 只在 encoder 路径里给，见官方 Observation Configuration：
`token_state` 可由本地 encoder 产生，其 `encoder_observations` 含 `motion_anchor_orientation`）。
而 SONIC 的 token 按定义是**相对机器人自身朝向**的（`anchor = rot6d(quat_conj(q_robot) ⊗ q_ref)`）
⇒ **绝对 yaw 从来不进 token**。实测：token 对 yaw 误差的 5 折 CV R² = **0.226**（对 root_pos = 0.331）。

**改动**：`motion_global_root_ori`（含 yaw、满权重）→ 拆三项，**权重总和仍 = `w_root_ori`**
（⇒ `max_step_reward` 仍是 0.120，critic 的价值尺度不变）：

| 新 term | 权重 | 可观测性 |
|---|---|---|
| `motion_root_tilt`（`acos(up_r·up_ref)²`，只 roll/pitch） | `0.5·w_root_ori` | ✅ `proj_grav` 直接可观测 |
| `motion_root_yaw`（Δyaw²，绕回） | `0.2·w_root_ori` | ⚠️ 只有 ~23% 通道 |
| **`motion_root_yaw_rate`**（锚点 body 世界系角速度 z 差） | **`0.3·w_root_ori`** | ✅ **`w_b` 陀螺，就是"转向"** |

`motion_root_ori_tail`（铰链）改为作用在 **tilt** 上。

**验证**（mjlab venv，随机输入）：
- `max_step_reward` 前后都是 **0.12000** ✅
- 除 ori 三项外，其余所有 term **逐位相同** ✅
- `motion_root_tilt/yaw/yaw_rate` 均非零，`motion_root_ori_tail` 在 tilt 上咬合 ✅

**实测启动**：R1 用新奖励跑起来后 `rollout/reward` 0.0874→0.0901、30 步无 NaN/无崩 ✅

**评测口径**：以后报 ori 要拆成 `tilt` / `yaw` / `yaw_rate` 三个数；**判据只看 tilt 和 yaw_rate**，`yaw` 只报不判
（它是 VLA 路径结构上拿不到的量）。
