# 交接给 IsaacLab 线：MuJoCo 侧「联合微调」完整链路与架构（2026-10-30 版）

> 本文取代 `HANDOVER_TO_ISAACLAB_MUJOCO_JOINT_20261029.md`。
> 29 日那份写的是**当时**的 9 步流程 + 12 个坑；之后一天里改掉了三处**设计错误**、
> 补了四条数值稳定性修复、并把评测方法论整个换了一遍。本文按"**现在真实在跑的**"重写。
>
> 读法建议：§1 一页速览 → §3 契约 → §5 流程 → §6 奖励形状（**本次最大的改动**）
> → §10 数值稳定性（**照抄必看**）→ §11 评测方法论（**最容易继续踩**）→ §13 坑清单。

---

## 1. 一页速览

```
每轮 —— 10 步（原 9 步 + 新增固定评测集）
  1 场景采集（冻结 W1 WBC，带 4 类场景物体）           ~8 min   【与奖励无关，可跨轮复用】
  2 B 线回放采集（VLA 在环，4 任务 × 20 集 × 600 tick） ~25 min
  2.5 80 集健康硬门（fell 必须 0；奖励不得塌）          秒级
  3 bline→transitions + 混成 wm_mix（B 线占 44%）       ~2 min
  4 WM 三阶段训练（场景 3000 + mix 8000 + k=20 自回归 1500）~7 min
  5/6 WM 双分布验收门（sim + bline，单步 qd_corr/qd_ratio）~2 min
  7 critic（imagined 路线 = 无 prefit，只配 args）       秒级
  8 联合微调 800 步（LoRA + head，WBC/WM 冻结）          ~30 min
  9 导出 + 闭环验收（同一任务同起点 600 tick）           ~6 min
  9.5 固定起点评测集（seed=999 × 5 集/任务 = 20 集）      ~8 min   【跨轮可比的关键】
```

**累积链**：`CUR_WBC`（policy+WM）与 `CUR_VLA`（LoRA）**都**逐轮累积 —— R_N 用 R_{N-1} 的产物。
**单轮约 90 min**；只换奖励的实验可用 `RESUME_FROM_STEP=7` 复用 1–6 步，**降到 ~35 min**（§7.4）。

---

## 2. 两条栈，别混

| | MuJoCo（本文） | IsaacLab（你们） |
|---|---|---|
| 物理 | MuJoCo-Warp / `mjlab`（`third_party/SuperMimic-Reproduced/.venv`） | Isaac Sim |
| 状态布局 | `explicit_dt0.02`，71 维扁平 | 你们的 |
| 解释器 | **必须** `third_party/SuperMimic-Reproduced/.venv/bin/python`（`import mjlab` 只在那儿） | — |
| 关节序 | 标准 29 dof（`G1_JOINT_NAMES`） | 同名同序 |
| 资产 | `g1_29dof_with_hand.xml`（44 关节 / 43 电机 / 14 手） | 你们的 |

**跨栈不能直接拷 ckpt**：`is_mjlab_checkpoint()` 会拒掉 trackerlab 的 `{pi,wm}` 布局。
但**本文所有设计、奖励形状、流程、坑都与仿真器无关**，可直接借鉴。

---

## 3. 契约（这三条必须逐位对齐）

### 3.1 状态 / 观测 / 动作

| | 定义 |
|---|---|
| **state 71** | `[p(3), quat(4,wxyz), q(29), v_world(3), w_world(3), qd(29)]` |
| **obs 93** | term-major：`[q−q0(29), qd(29), w_body(3), proj_grav(3), a_prev(29)]`，`HISTORY=10` 帧 → 930 维 |
| **action 29** | **归一化关节偏置**：`q_target = default_q + action_scale · a`，`action_scale = 0.25·effort/kp` |
| **token 64** | SONIC latent，`HA_VLA_EXEC_STEPS=20` 帧一执行块（VLA 输出 40 帧 chunk） |

**实测校验过**：`q_target_applied == default_q + actions·action_scale`（B 线 dump 上 `max|diff| = 0.000e+00`）。
**注意**：库里的状态角速度是全球系（`w_world`），obs 与 WM 用 body 系 —— 转换要 `quat_apply_inverse`。

### 3.2 token：**它按定义不带绝对 yaw**

`sonic_tokens.py:180-188`：

```python
command = cat([dof_pos(29·nf), dof_vel(29·nf)])        # 参考的关节位置+速度
anchor  = _rot6d( quat_conj(q_robot) ⊗ q_ref )         # ← 参考被旋到「机器人自身朝向系」
return self.encoder(command, anchor)                   # token = enc(58, 6)
```

**⇒ token 表达的是"相对于我当前朝哪"，绝对 yaw 从来不进去。** 实测（80 集 48000 样本，ridge 5 折 CV）：
`token → yaw 误差 R² = 0.226`、`token → root_pos 误差 R² = 0.331`。

### 3.3 官方 `unitree_g1_sonic` 的 VLA 模态（**已逐字核对**）

`Isaac-GR00T/gr00t/configs/data/embodiment_configs.py:67`：

```python
"unitree_g1_sonic": {
    "video":  modality_keys=["ego_view"],
    "state":  modality_keys=["left_leg","right_leg","waist","left_arm","right_arm",
                             "left_hand","right_hand","projected_gravity"],     # 8 项，无 anchor
    "action": modality_keys=["motion_token","left_hand_joints","right_hand_joints"],
    "language": modality_keys=["annotation.human.task_description"],
}
```

**⇒ 我们的 VLA 输入和官方一字不差。** 朝向信息只可能从 `ego_view` 隐式推断。
官方的 tracking 部署另有 **encoder 路径**（`token_state` 由本地 ONNX encoder 产生，
其 `encoder_observations` 含 `motion_anchor_orientation`(60)，anchor 用的是**真实机器人 base quat**）
—— 那条路才能拿到绝对朝向。**两条路的取舍见 §6.4。**

---

## 4. 数据：四种，各管一段

| 数据 | 用途 | 规模 |
|---|---|---|
| **318 库** `data/library/ft318_compact.pt` | 参考动作 + token；`ref (473876,71)` | 318 clips |
| **场景采集** `round_N/transitions.pt` | WM 学"接触/物体"分布 | **1,023,700 行**（4 任务 × 256 envs × 1000 步）|
| **B 线 dump** `round_N/bline/b_*.pt` | 部署分布（VLA 在环真实轨迹） | 每轮 **80 集 × 600 tick** |
| **固定评测集** `round_N/eval_set/` | 跨轮同起点比较 | 20 集（5/任务，seed=999）|

**⚠️ 库里 `v_root`/`w_root` 通道全库恒为 0**（473,876 帧实测 `max = 0.000000`）。
任何以状态 `[36:42]` 为目标的速度监督都会教策略"别动"。参考的 body 角速度来自 `library.bodies(f)` 的 FK，**是正常的**。

---

## 5. 每轮 10 步：每步的输入/输出/判据

### 1/10 场景采集
- `collect_mjlab.py --ckpt $W1_CKPT`（**冻结 W1 WBC**，与奖励、与 VLA 都无关）→ `transitions_{task1..4}.pt` → concat
- **可跨轮硬链接复用**（WBC 冻结 ⇒ 数据等价）
- 4 类场景物体由 `scripts/extract_scene_objects.py` 从 SONIC 场景拆成 mjlab 可加载的 MJCF

### 2/10 B 线回放采集（**本轮唯一花时间的步骤**）
- `closed_loop_vla_supermimic.py --arm vla`，`CUR_VLA`(LoRA) × `CUR_WBC`(policy+WM)
- 4 任务 × 20 集 × 600 tick，4 并发
- **断点续采**：dump 文件名确定性（`b_${T}_f${F}_e${E}.pt`），已存在就跳过（§13.7）
- **奖励只被 `rec_rewards.append(...)` 记录，从不进控制回路** ⇒ 轨迹与奖励无关（重要！见 §7.4）

### 2.5/10 80 集健康硬门（**2026-10-30 新增，必须在这儿**）
- `scripts/bline_health.py`：`fell`、`max_tilt`、平均奖励
- 默认 `--max-fell 0`；不过就 `exit 8`
- **为什么必须**：`eval.json` 里的 `fell` 是**单集** eval（同任务同起点一条 600 tick），
  恰好不摔就一直是 `False`。真正的摔倒记录在每条 dump 的 `meta['fell']` 里。
  **我们据此漏看 4 小时**（R3–R6 摔倒 2→4/80 但日志一直显示 False）。

### 3/10 bline→transitions + mix
- `bline_to_transitions.py`：**坏集过滤**（摔 / 倾角>45° 的 episode 不许进 mix）+ 填真实 xi
- `make_wm_mix.py`：场景全量保留，B 线**重复 k 次**到目标占比 0.45
  （⚠️ 不要用 `--downsample-sim`：那会浪费 94% 的场景数据）

### 4/10 WM 三阶段（`pretrain_wm.py`）
| 阶段 | 数据 | 步数 | 作用 |
|---|---|---|---|
| 1 | 完整场景 `transitions.pt` | 3000 | 学接触/物体 |
| 2 | `wm_mix.pt` | 8000 | 适配部署分布；带**幅值比正则** `mag_weight=30`（**只压 qd**）|
| **2b** | B 线连续窗口，**k=20 自回归**，每步**交织一条场景单步样本** | 1500 | **修多步累积误差**（见 §8）|

- 总耗时 **~394 s**（单卡 4090）
- **阶段 2b 不能单独跑**：实测"纯 B 线窗口 600 步"会把 sim 单步门打坏
  （`corr 0.9896→0.9356`、`ratio 0.995→1.041` ⇒ `exit 2`）。**必须交织 sim 批**（`--roll-sim-weight 0.3`）。

### 5–6/10 WM 验收门
- `probe_wm_qd_corr.py`，判据：**`qd_corr ≥ 0.95` 且 `0.90 ≤ qd_ratio ≤ 1.10`**，`n ≥ 2000`
- 两个分布都要过：`transitions.pt`（sim）与 `bline_transitions.pt`（部署）
- **注意：这一步只量单步**，单步准 ≠ 多步准（§8）

### 7/10 critic
- `CRITIC_SOURCE=imagined` 时**没有 prefit**：critic 在微调里用想象 rollout 的 **GAE** 目标和策略一起更新
- 所以"没有 `critic.pt`"是正常的，不是失败
- （`real` 路线需要 `fit_critic_real.py`，实测 `v_ev` 只有 0.09–0.15 ⇒ 已弃用，§13.2）

### 8/10 联合微调（见 §6）
- `train_joint_mjlab.py --v3`，800 步，4 卡 DDP，`batch=16/rank`（32 会 OOM）
- 可训：**head 67.1M**（LoRA rank 64）；冻结：`WBC 8.96M` + `WM 8.82M`

### 9/10 导出 + 闭环验收
- `export_vla_ckpt.py` 把 LoRA 合并进 base → `ckpt_vla_r{N}`
- **自带 NaN 校验**（`[verify] LoRA 合并出 NaN`）—— 这条救了 §10 那次事故
- 闭环：同任务同起点 600 tick

### 9.5/10 固定起点评测集（**2026-10-30 新增**）
- `collect_bline_arm.sh`，`seed=999` 固定 × 5 集/任务
- **为什么必须**：训练用 B 线 `seed=$R`（每轮不同起点，保持多样性），
  但**跨轮比较**必须同起点。没有它，"起点不同"的噪声会掩盖早期退化（§11.2）

---

## 6. 联合微调架构（v3）

### 6.1 损失组成（**实测过的完整清单**）

```python
loss = 0.3 · loss_r                                   # 奖励（走冻结 WM 的 H=8 想象）
     + λ_trust · loss_trust                           # token 信任域（eps=0.2，中心=self）
     + loss_act                                       # 动作 hinge（eps=0.6, weight=1.0）
     + λ_anchor · loss_anchor                         # = 0（关）
     + λ_hand · loss_hand                             # = 0（关）
     + λ_token_bc · loss_token_bc                     # = 0（关，设计原则"token 不做标签监督"）
```

**⚠️ 三个"参数照收、代码不读"的坑（v3 覆写了 v2 的 `train_step`）：**

| 参数 | 状态 | 后果 |
|---|---|---|
| `--lambda-bc` | **v3 完全不读** | 传 `0.03` 也不产生任何梯度，但会写进 `args.json`/`summary.json` ⇒ 交接文档会撒谎 |
| `--lambda-s`（state MSE 项） | **v3 原本完全不读**（本次我们把它接上了，`v3_step.py` 里现在有 `lambda_s → _state_error` 分支，但 **默认 0** 且在最终配方里**不用** —— 因为"尾部"改由奖励内铰链承担，见 §6.5）| 原本传了也不生效 |
| `--lambda-a` | v3 用 `anchor_weight` 另设，`lambda_a` 不读 | 同上 |

**建议**：Isaac 侧若也用"v3 覆写"的结构，**必须让 v3 把 v2 的所有 loss 项显式声明**，
或者在 AUDIT 行里逐项打印"实际生效权重"，否则这类静默失效会一直存在。

### 6.2 信用分配：H=8 pathwise + critic bootstrap

```
rollout:  s_t --π(θ)--> a_t --可微 WM--> ŝ_{t+1},  t = 0..7
ret  = Σ_{t<8} γ^t · survival_t · r_t
boot = γ^8 · survival_8 · ( V(ŝ_8) − V(ref_8) )       # rel 形式，B2 修复
loss_r = −( ret + boot ).mean()
```

- `V` 对 `ŝ_8` **保留梯度**（straight-through，`critic_state_grad_scale=0.1`，只缩放梯度不缩放数值）
- critic 用 **GAE(γ=0.99, λ=0.95)** 目标，在想象 rollout 上更新
- **为什么 H=8**：A4 消融（各 400 步）`H8 +0.0820 > H16 +0.0778 > H32 +0.0734 > H40 +0.0681`。
  **注意这个比较是"固定 400 个优化步"，混着算力差**，不是纯粹的"长链不可优化"。
- **我们实测过 20 帧那个执行窗口内的误差是平的**（见 §8.3）⇒ **H=8 够用，不需要 20**。

### 6.3 信任域：`trust_center = self`

- `trust_eps=0.2` 是 token 空间相对先验的 RMSE 容差，`trust_weight=1.0`
- **中心必须是「采集时策略真正下发的 token」**（`--start-from-b` + `self_targets.pt`），
  不是数据集标签、也不是策略自己的当前输出 —— 否则约束会漂
- 这也是为什么 B 线必须 `--arm vla`：只有 vla 臂写 `tokens` 进 dump

### 6.4 朝向（root ori）：**这是本次最大的设计修正**

**问题**：`motion_global_root_ori` 原本是**全球系、含 yaw、满权重**。但：

1. 官方 `unitree_g1_sonic` 的 VLA 路径**没有 anchor 输入**（§3.3）⇒ 绝对 yaw 只能从相机猜
   （token → yaw 的 `R² = 0.226`）
2. SONIC token 按定义是 yaw-相对的（§3.2）⇒ 机器人会"以自己当前朝向为基准"执行
3. 实测 ori 误差分解：**yaw 7.53°（p90 24.7°）**、pitch 5.34°（偏置 4.59°）、roll 1.85°
   ⇒ 10.24° 里约 **7.5° 在架构上拿不到**

**改法**：拆三项，**权重总和 1.5×**（`max_step_reward` 0.12 → **0.13**）：

| term | 定义 | 权重 | 可观测性 |
|---|---|---|---|
| `motion_root_tilt` | `‖up_r − up_ref‖²`（≈θ_tilt²，只 roll/pitch）| **1.0×** | ✅ `proj_grav` 直接可观测 |
| `motion_root_yaw` | Δyaw²（绕回到 ±π）| 0.2× | ⚠️ 仅 23% 通道 |
| **`motion_root_yaw_rate`** | `(ω_yaw,ref − ω_yaw,robot)²`，锚点 body 的世界系角速度 z | **0.3×** | ✅ `w_b` 陀螺 |

**"转向"因此从 1/14 的弱监督（原本只有 `motion_body_ang_vel` 里 pelvis 那一份）
变成专门的 0.3× 项** —— 而丢掉的只是"面朝库世界系哪个绝对方位"。

**如果你们的部署有可信的机器人 yaw**（动捕/VIO/SLAM）：可以把 `motion_anchor_orientation`
作为输入喂给 VLA（偏离官方模态，需重训）或走 **encoder 路径**（官方支持，anchor 用真实 base quat）
—— 那时绝对 yaw 才值得满权重。

### 6.5 奖励形状：**主体 `exp` + 尾部 `hinge`（两段式）**

| 项 | σ / 阈值 | 权重 |
|---|---|---|
| `motion_global_root_pos` | σ=0.20 m | 1.0 |
| `motion_root_tilt` / `yaw` / `yaw_rate` | σ=0.25 rad / 0.25 rad / 1.0 rad·s⁻¹ | 1.0 / 0.2 / 0.3 |
| `motion_body_pos` / `_ori` / `_lin_vel` / `_ang_vel` | σ=0.30 m / 0.40 rad / 1.0 / 3.14 | 各 1.0 |
| `action_rate_l2` | — | **−0.1** |
| `joint_limit` | — | **−2.0** |
| **`motion_root_pos_tail`** | `relu(‖Δp‖−0.35 m)`，cap 1.0 m | λ=2.0 |
| **`motion_root_tilt_tail`** | `relu(θ_tilt−20°)`，cap 57° | λ=2.0 |
| **`motion_body_pos_tail_legs`** | `relu(RMS₈ −0.10 m)` | λ=2.0 |
| **`motion_body_pos_tail_arms`** | `relu(RMS₆ −0.25 m)` | λ=2.0 |

**为什么尾部必须加、而且必须是 hinge 不是 MSE/Huber：**

`exp(−e/σ²)` 的梯度是 `−2e/σ²·exp(−e/σ²)` —— **两端都趋 0**：
- `e→0`：梯度 0（**无害**，那是最优点）
- `e→∞`：梯度 0（**这是病** —— 均值目标会主动"拿尾部换中位"）

按误差分桶实测（`j1_scene2/round_5` 的 80 集）：

| 误差区间 | 占时刻 | 提供的位置梯度 |
|---|---|---|
| 0–0.1 m | 72.4% | **86.3%** |
| >0.4 m | **21.2%** | **0.22%** |

**后果（同批 80 集，按集自助重采样 250 次的 95% CI）**：

| 产物 | pos p50 (cm) | pos p90 (cm) |
|---|---|---|
| π0 未微调 | 5.79 [5.40, 6.28] | **24.9** |
| π2 | **4.11** [3.67, 4.73] ✅ | 59.6 ❌ |
| π5 | 5.95 ❌ | 58.4 ❌ |

⇒ 中位真的压下去了（显著），但尾部同时翻 2.2 倍。**这是该目标的最优解，不是训练跑偏。**

**三种尾部形状的对比（为什么选 hinge）：**

| 形状 | 0~e₀ | >e₀ | 与 `exp` 是否重复 |
|---|---|---|---|
| 纯 MSE | 二次 | 二次 | ✅ **完全重复**（`exp` 小误差展开就是二次）|
| Huber | 二次 | 线性 | ⚠️ **主体区间重复** |
| **hinge（用这个）** | **恒 0** | **线性** | ❌ **不重复** |

**为什么尾部用线性（而不是二次）**：线性 = 恒定拉力；二次的梯度 ∝ 距离，在 2.7 m 的极端样本上会到 5 倍、主导整个梯度。线性 + `cap` 是"有力但不失控"。

**`body_pos` 铰链必须分两组**（实测逐 body 重锚定位置误差）：

| 组 | bodies | p50 | p90 | `e₀` |
|---|---|---|---|---|
| 腿+躯干 | 索引 0–7 | 0.046 m | ~0.10 | **0.10 m** |
| 手臂 | 索引 8–13 | **0.126 m** | ~0.30 | **0.25 m** |

**手臂 6 body 平均 0.126 m，腿+躯干 8 body 平均 0.046 m（差 2.8 倍）**。
用"14 body 平均 + 单一阈值"的话，铰链只会去拽手臂 —— 而手臂那 0.13 m 是 `arm=vla` 的**正常偏置**
（肩俯仰 **+34.7°**，系统性），不是缺陷。

**已验证的实证**：`root_pos` + `root_tilt` 铰链把 `pos p90` 从 **52.1 → 26.9 cm（−48%）**、
`pos p50` 5.31 → **4.41 cm**、`fell 0/80`。

### 6.6 **绝对** tilt / pelvis 高度惩罚：**不要加**

我们踩过这个坑，数据如下（库里 318 clips / 473,876 帧的**参考**）：

| | 值 |
|---|---|
| root tilt p50 / p90 / **p99 / max** | 6.2° / 11.9° / **27.3° / 34.4°** |
| tilt >20° 的帧 | **3.2%**；自身 max>20° 的 clip：**56/318** |
| pelvis z p1 / **min** / p50 | **0.413 / 0.385 / 0.749** |
| pelvis z < 0.70 m 的帧 | **13.4%** |

**我们的任务集包含弯腰/蹲下**（`task_4_diningroom_can_low_chair` max tilt 34.4°、pelvis 最低 0.56 m）。
所以"相对竖直"的绝对 tilt 惩罚会压合法动作，"pelvis < 0.70 m 就罚"会压 13.4% 的合法帧。
**这两项我们已设为 0（关）**，摔倒改由"80 集硬门 + 坏集过滤 + 终止丢奖励"三道防线管。

---

## 7. 观测宽度 93 的含义（以及它带来的硬限制）

`OBS_TERM_DIMS_93 = (29, 29, 3, 3, 29)` = `[q−q0, qd, w_b, proj_grav, a_prev]`。

**里面没有根位置、没有根朝向、没有 heading。** `proj_grav` 只能给 roll/pitch。

Supermimic 原生是 **99 维**（`relative_ori99` = 93 + `rot6d(RᵀR_ref)`），那最后 6 维**正是相对根朝向**。
我们为了与 IsaacLab 对齐砍到 93 ⇒ **WBC 天生看不见朝向**。

**改回 99 的代价**：W1 的第一层是 `930 = 93×10`，换 99 必须**重训 WBC**（作废现有 W1）。
**折中方案**：把 6 维喂给 **VLA head 的 state 输入**（`max_state_dim=132`，现在只用了 46 维，有余量）
—— 但 §3.3 说明官方模态没有它，属于偏离。

---

## 8. 世界模型（WM）：三阶段 + 双门 + **同分布多步判据**

### 8.1 单步门禁（现用）

`qd_corr ≥ 0.95` 且 `0.90 ≤ qd_ratio ≤ 1.10`，`n ≥ 2000`，两分布都过。
**R1（从 W1 + 2b）实测：sim `0.9859 / 1.0020`、bline `0.9824 / 1.0390`。**

### 8.2 **单步门禁看不见多步误差**（重要）

修复前（`k` 步自回归，同分布）：

| k | qd_ratio |
|---|---|
| 1 | 0.977 ← **门禁只量这一步，所以永远 PASS** |
| 8 | **1.382** |
| 20 | **1.329** |

**根因**：`wm_refresh` 的损失是**纯单步**（`s_next = wm(state, action)`，输入永远是真实状态），
模型从没见过自己的输出 ⇒ 累积误差不受任何约束。

### 8.3 阶段 2b：k=20 自回归 + 交织 sim（已验证）

- 从 B 线取**连续窗口** `(k+1) 帧状态 + k 步动作`，**每步喂模型自己上一步的输出**
- 每步**交织一条真实场景单步样本**（否则会打坏 sim 分布）
- `k=20 / 1500 步 / sim_weight=0.3`，成本 **~330 s**

**效果（同分布，k=20）**：

| WM | 关节 mae | qd_corr | **qd_ratio** |
|---|---|---|---|
| 修复前（W1） | 0.63° | 0.905 | **1.260** |
| 修复后（W1+2b） | **0.42°** | **0.960** | **1.031** |

**而且 corr 不随 k 衰减**（k=1 → k=20 只掉 0.01），说明累积误差被压住了。

### 8.4 **判据的正确口径：同分布，不是跨轮**

同一份 WM 在"另一轮"的数据上量，数字会看起来差很多（因为那批数据是**旧策略**的行为）：

| 测法 | 结果 | 能不能用 |
|---|---|---|
| 每轮 WM 在**自己那一轮**数据上（同分布） | k=20 ratio **0.99–1.03** | ✅ **这才是判据** |
| 固定用 round_1 的数据测各轮 WM（跨轮） | 1.25 → 1.39 逐轮"变差" | ❌ **量的是策略演化，不是 WM** |

### 8.5 20 帧执行窗口内的误差是**平的**（⇒ 不需要 H=20）

B 线每个 600 tick 有 30 个重规划窗口。按"窗口内偏移 0–19"分桶（铰链臂 80 集）：

| 窗口内偏移 | pos 中位 | ori 中位 |
|---|---|---|
| 0（刚重规划）| 4.49 cm | 10.14° |
| 8 | 4.31 | 10.18 |
| 19（窗口末）| **4.54** | **10.51** |
| **末/首** | **1.01×** | **1.04×** |

**⇒ 9–20 帧没有累积误差 ⇒ 把 `H` 拉到 20 没有收益**（成本 2.4×）。
**剩余误差是"常数偏置"，不是累积** —— 偏置来源是冻结 WBC 的站姿倾向
（髋俯仰 −11.2°、膝 +7.1°、手臂 +34.7°，都是系统性偏置）。

---

## 9. 数值稳定性：**三个奇点 + 一个熔断**（照抄必看）

### 9.1 `_sq_quat_err` 用了 log-map ⇒ 朝向越准越容易 NaN

```python
# ❌ 原实现
v = quat_box_minus(q1, q2)          # → axis_angle_from_quat（log map）
return (v * v).sum(-1)              # θ→0 时是 0/0，梯度 NaN
```

**这是最阴的一个**：`axis_angle = 2·atan2(‖v‖,w)·v/‖v‖`，`v/‖v‖` 在 θ=0 是 0/0。
**而训练越好、朝向越准，就越贴近 θ=0。**

**实锤**：R1 第 **535** 步 `grad/head_preclip` 变 NaN，而那时 `rollout/reward = 0.1048` 是**本轮最高**。
此前 6 轮都能跑完 800 步，只是因为跟踪还差、从没接近过 θ=0。

**修法（处处可导、量纲不变）**：

```python
def _sq_quat_err(q1, q2):
    d2p = (q1 - q2).pow(2).sum(-1)
    d2m = (q1 + q2).pow(2).sum(-1)
    return 4.0 * torch.minimum(d2p, d2m)     # = 2−2|q1·q2| ≈ θ²，梯度线性有界
```

**量纲校验**：θ=10° → 新值 `0.030442` vs `θ² = 0.030462`（5 位有效数字一致）
⇒ **所有 σ 标定不用重调**。双覆盖形式同时消除 `q`/`−q` 的符号歧义。

### 9.2 `acos` 的两个奇点

| 位置 | 问题 |
|---|---|
| 倾斜惩罚 `torch.acos(up_z)` | 直立时 `up_z→1`，导数 `−1/√(1−up_z²)→∞` |
| `e_root_tilt = acos(dot)²` | 上下方向几乎平行（我们几乎总是）⇒ 同上 |

**修法**：都换成光滑形式
- 倾斜：`1 − up_z`（≈θ²/2），阈值换算成 `1−cos(20°)`
- tilt 误差：`‖up_r − up_ref‖²`（≈θ²）

### 9.3 `.sqrt()` 的无穷导数（已加固）

`relu(sqrt(e) − e₀)` 在 `e→0` 处导数为 ∞。所有 `sqrt` 都加了 `clamp_min(1e-8)`
⇒ 导数最大 ~5000× 系数（**有限**），且被 `clip_grad_norm_(1.0)` 兜住。

### 9.4 **NaN 熔断**（这次事故的教训）

```python
total = torch.nn.utils.clip_grad_norm_(self.fm_parameters, v3.clip_head)
if not torch.isfinite(torch.as_tensor(total)):
    raise RuntimeError(f"[NaN-guard] step={self.steps} grad 非有限 ⇒ 立即中止")
```

**没有它的时候**：NaN 从第 535 步持续到 800 步（白烧 **266 步 ≈ 15 min**），
而且**照常写出了 NaN ckpt** —— 直到 `export_vla_ckpt.py` 的 `[verify] LoRA 合并出 NaN` 才发现。

**上游必须有这一道**：`export_vla_ckpt.py` 的 merge 校验是最后一道闸，不能省。

### 9.5 修复后的实际梯度

```
真实 B 线 600 帧：梯度有限，max|g| = 1.32e-01
最坏情况（机器人==参考）：梯度有限，max|g| = 1.09e-09     ← 旧写法这里是 NaN
R1 训练 481 步：梯度有限 481/481，grad p50 = 0.003、max = 0.02
                （离 clip 阈值 1.0 有 50 倍余量 ⇒ 不再"贴着悬崖走"）
```

---

## 10. 评测方法论（**我们在这上面栽了三次**）

### 10.1 单集闭环 eval 有 ~30% 噪声 —— 不能当判据

| | 单集 eval（同任务同起点 600 tick） | 80 集采集分布 |
|---|---|---|
| R1 的 `pos p50` | **4.06 cm** | **5.79 cm** |

⇒ 单集和 80 集差 30%。我们据此一度得出"root_pos 回归 +34%"的错误结论。

### 10.2 `mean_reward` 跨轮**不可比**（尺子变过三次）

| 版本 | `max_step_reward` | 备注 |
|---|---|---|
| 最初 | 0.10 | `w_root_pos=0.5` |
| 之后 | 0.12 | root 权重 0.5→1.0 |
| **现在** | **0.13** | root ori 拆 1.5× |

再加上 `std_root_ori` 在 0.25/0.15 之间来回改过 ⇒ **同一个朝向误差在不同版本下奖励能差一倍**。
**判据只能用物理量**（`pos p50/p90`、`tilt 误差`、`fell`），不能用奖励。

### 10.3 尾部统计量必须报**置信区间**

`pos p90` 是按集重采样后的 CI 才有意义。实测：

| 产物 | p90 | 自助 95% CI |
|---|---|---|
| π0 | 24.9 | [16.7, 31.8] |
| π1 | 52.1 | **[33.1, 59.5]** ← 宽到不可判 |
| π3 | 53.7 | **[20.7, 57.6]** ← 同样宽 |

⇒ `n=80` 集时 p90 的 CI 宽达 ±15 cm。**不要让单点数字决定实验方向。**

### 10.4 `fell` 只有一个可信来源：`dump['meta']['fell']`

`eval.json` 里的 `fell` 是单集；**80 集采集数据里每条 dump 的 `meta`** 才是真的。

### 10.5 判据模板（拿去直接用）

| 指标 | 口径 | 通过线 |
|---|---|---|
| `pos p50` | 80 集，按集自助 CI | 显著优于基线（CI 不重叠）|
| `pos p90` | 同上 | **不退化**（CI 与基线重叠即可）|
| `tilt 误差 p50` | 80 集，**相对参考**（不是绝对竖直）| ≤ 基线 |
| `yaw` | 只报不判（架构上拿不到）| — |
| `fell` | `meta['fell']` | **必须 0/80** |
| 轮间一致性 | 固定起点 20 集（seed=999）| 退化 >10% 即停 |

---

## 11. 实测结果（供对标）

### 11.1 80 集闭环漂移（π0 基线，`seed=1` 起点）

| 指标 | 均值 | p50 | p75 | p90 |
|---|---|---|---|---|
| root_pos | 11.44 cm | 5.79 | 9.46 | 24.9 |
| root_ori | 14.34° | 10.65 | 16.93 | 24.7 |
| joint_rad | 0.335 | 0.309 | 0.411 | 0.513 |
| joint_vel | 0.315 rad/s | 0.222 | 0.398 | 0.635 |

### 11.2 六轮累积的退化（**这是最重要的教训**）

| 产物 | 配方 | pos p50 | pos p90 | ori p50 | fell |
|---|---|---|---|---|---|
| π0 | 未微调 | 5.79 | **24.9** | 10.65 | 0/80 |
| π1 (R1) | 原始 | 5.31 | **52.1** | 10.19 | 0/80 |
| π2 (R2) | 改配方① | **4.11** | 59.6 | 10.35 | 0/80 |
| π3 (R3) | 同① | 4.32 | 53.7 | 10.03 | **2/80** |
| π4 (R4) | 同① | 4.90 | 58.4 | **13.34** | 3/80 |
| π5 (R5) | 同① | 5.95 | 58.4 | 13.51 | **4/80** |
| π6 (R6) | 同① | 4.34 | 57.6 | 13.80 | 3/80 |

**读法**：中位在前 3 轮真的改善（4.11，CI 不重叠），但尾部**从 π1 就翻倍**；
第 4 轮起连中位也失守、并出现摔倒。**根因就是 §6.5 的"尾部失明"。**

### 11.3 铰链修复后的实测（从 π1 起，80 集，`seed=2`）

| 指标 | π1 基线 | 铰链臂 | |
|---|---|---|---|
| `pos p50` | 5.31 [4.83, 6.09] | **4.41 [4.04, 4.94]** | ✅ CI 不重叠 |
| **`pos p90`** | **52.1 [32.5, 59.5]** | **26.9 [18.5, 33.5]** | ✅✅ **−48%，CI 不重叠** |
| `ori p50` | 10.19 | 10.24 | ≈ |
| `fell` | 0/80 | **0/80** | ✅ |

**这是"奖励形状修尾部"的直接证据。**

---

## 12. 给 IsaacLab 侧的可移植性

### 可以直接搬（与仿真器无关）
1. **两段式奖励形状**（`exp` 主体 + `hinge` 尾部）+ 分组的 `body_pos` 铰链 + 分组阈值
2. **root ori 拆 tilt/yaw/yaw_rate** 的思路（按"可观测性"配权重）
3. **WM 阶段 2b**（k 步自回归 + 交织单步）—— 只要能取到连续窗口
4. **§9 的四个数值修复**（`_sq_quat_err` / `acos` / `sqrt` / NaN 熔断）—— **强烈建议全线照抄**
5. **§10 的评测方法论**（80 集、自助 CI、固定起点评测集、`meta['fell']`）
6. **健康硬门 + 坏集过滤 + 断点续采**
7. **"配置说一套代码做一套"的自查**（§6.1 那三个空转参数）

### 需要按 Isaac 侧改写
1. 资产/执行器换算（`action_scale` 的来源、armature/damping 的常量）
2. WM 的 `step/observation/reward` 适配层（你们的 `MDP` 语义）
3. `RESUME_FROM_STEP` 这类流程复用（依赖你们的数据是否与奖励无关 —— **先验证 §7.4 那条**）

### 建议 Isaac 侧优先验证的三件事
1. **`mean_reward` 在不同 `std`/权重下是否可比**（我们的答案：不可比）
2. **单集闭环 eval 的噪声有多大**（我们的答案：~30%，必须 80 集）
3. **`_sq_quat_err` 这类 log-map 在"完美跟踪"处是否 NaN**（我们的答案：会，且越训好越容易）

---

## 13. 坑清单（每条都真踩过）

| # | 坑 | 现象 | 修法 |
|---|---|---|---|
| 1 | 解冻 WBC（RL 穿 WM） | 闭环 `root_pos` 4.16 → **89.2 cm（21×）**，而权重 `max\|Δ\|` 只有 4.8e-4 | `TRAIN_WBC=0` |
| 2 | critic 用 real 长 return | `v_ev` 只有 0.09–0.15（输入里没有 clip 进度 ⇒ 结构性不可学）| 改 **GAE on imagined**，`v_ev` → 0.99 |
| 3 | `pipefail` + `set -e` | `ls ... \| wc -l` 在目录空时 `ls` 返回 2 ⇒ 整条赋值失败 ⇒ **脚本静默退出**（B 线一步没跑、日志无报错）| `\|\| true` |
| 4 | `wait` 在 `set -e` 下 | 任一子进程非零 ⇒ **整条链被带走** | `wait \|\| true` |
| 5 | `nvidia-smi \| head -1` 的 SIGPIPE | VRAM 读成两行 ⇒ 4090 降配逻辑静默失效 | 读全部输出再取第一行、剥非数字 |
| 6 | v3 覆写 `train_step` | `lambda_bc` / `lambda_s` / `lambda_a` **照收不读**，还写进 config | AUDIT 行打印"实际生效权重" |
| 7 | B 线 `frames` 只在重规划时刻记录 | (600,) vs (30,) 形状不匹配 | 按段展开到每 tick |
| 8 | 链被杀后重跑 | 半截 dump 被复用 ⇒ 混入不完整 episode | **断点续采 + 先删半截** |
| 9 | 库 `v_root/w_root` 恒 0 | 速度监督会教策略"别动" | 不碰状态 `[36:42]`；用 body 角速度（FK） |
| 10 | 跨轮比较 WM | 看起来逐轮变差，其实是策略演化 | 只用**同分布**判据 |
| 11 | 绝对 tilt/高度惩罚 | 库里参考自身 tilt 34.4°、pelvis 0.385 m ⇒ 压合法动作 | 关掉（§6.6）|
| 12 | log-map 奇点 | 训得越好越容易 NaN | §9.1 |
| 13 | NaN 无熔断 | 白烧 266 步 + 写出 NaN ckpt | §9.4 |
| 14 | `RESUME_FROM_STEP` 把 `LAUNCH` 也跳过了 | `Permission denied`（bash 直接 exec `.py`）| `LAUNCH` 必须定义在守卫之外 |
| 15 | `read -r -a` 只读第一行 | 起点列表只取到 1 个 | 用数组赋值 `STARTS=($OUT)` |

---

## 14. 成本与时间线

| 场景 | 耗时 |
|---|---|
| 完整一轮（10 步）| **~90 min** |
| 只换奖励（`RESUME_FROM_STEP=7`，复用 1–6）| **~35 min** |
| WM 重建（含 2b）| ~7 min |
| 80 集闭环采集 | ~25 min |
| 固定评测集 20 集 | ~8 min |
| 十轮累积 | **~15 h** |

---

## 15. 复现命令（单轮，从某个 ckpt 起）

```bash
cd <repo>
setsid nohup env \
  OUT=$PWD/runs/<exp> START_ROUND=1 ROUNDS=1 PATIENCE=99 \
  VLA_CKPT=<base or ckpt_vla_r{N}> \
  CUR_WBC=<W1 or round_{N}/train/joint_final.pt> \
  NGPU=4 GPUS=0,1,2,3 J1_BATCH=16 J1_LORA=64 J1_H=8 \
  WM_PRETRAIN_STEPS=8000 WM_BLINE_RATIO=0.45 \
  CRITIC_MODE=on CRITIC_SOURCE=imagined TRAIN_WBC=0 \
  BLINE_EVAL_SET=1 BLINE_EVAL_SEED=999 BLINE_EVAL_PER_TASK=5 \
  bash scripts/launch_j1_rounds.sh > logs/launch.log 2>&1 &
```

**只换奖励（复用 1–6 步）**：加 `RESUME_FROM_STEP=7`，并预先在该轮目录备好
`bline/ transitions.pt bline_transitions.pt wm_mix.pt wm_pre.pt self_targets.pt gate_wm_*.json`。

**一键审计**：

```bash
PYTHONPATH=$PWD python3 scripts/report_rounds.py --drift --kstep    # 门 / eval / 80集 p50-p90 / 同分布 k 步
```

---

## 16. 待办（我们知道但还没做）

1. **WBC 真实数据监督**：`scripts/pretrain_wbc.py` 还不存在。目前 WBC 永久冻结在 W1，
   而 A5 消融（Isaac 栈）显示"WBC 可训"能改善想象侧指标 —— **但那是 RL 穿 WM，
   我们的 mjlab 实测会炸闭环（坑 #1）**。真正没走过的是"用真实 `(s,a,s′)` 监督 WBC"，值得做。
2. **`wm_refresh.py` 的 `ori` 缩放缺陷**：`(R*S).sum(-1).sum(-1)` 对 `(B,4)` 等于**把整个 batch 求和**，
   实际是 `0.5·(3−Σ_b cos)`，梯度是应有值的 **B 倍**（B=2048 ⇒ 2048×）。
   `mag_weight=30` 是在这个错误缩放下调出来的 ⇒ **改它必须重新调参**。
3. **CVaR / 分位数聚合**：现在的尾部是"铰链近似的线性惩罚"，更彻底的做法是直接对最差 10% 样本加权。
4. **`body_ori` 的铰链**：目前只给 `body_pos` 加了分组铰链，`body_ori` 仍是纯 `exp`。
5. **或跳到 99 维观测 / 给 VLA 补 anchor 输入**：前提是部署能拿到可信 yaw（§6.4）。
