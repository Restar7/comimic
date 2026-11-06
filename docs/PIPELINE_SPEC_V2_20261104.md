# 全流程规范 v2（SONIC-decoder 口径）— 2026-11-04

> 这份文档是**唯一的执行口径**。任何一步与本文不符，一律以本文为准；改动必须先改本文。
>
> 状态：**待批准**。批准前不启动采集/训练/评估。

---

## 0. 五条铁律

| # | 铁律 | 为什么（事故记录） |
|---|---|---|
| **1** | **执行器统一 SONIC release decoder**（`model_decoder.onnx`），不用 SuperMimic WBC | 录制数据的参考轨迹就是 SONIC decoder 产出的。用 SuperMimic 追它 = 两个不同执行器互相比，实测段内 20 步漂移 7.54cm(p50)/22.4(p90)，关节 0.139 rad；换 SONIC 后 2.75cm/6.3(p90)、0.070 rad |
| **2** | **关节序统一 SONIC 序**（`SONIC_ISAACLAB_JOINT_ORDER` → `idx_s`），且**三处必须一致**：① decoder 观测读关节 ② 动作目标落位 ③ 段重置写回 | `standard` 与 `SONIC` 是两个不同的 29 关节序（`[0,3,6,9,13,17,1,4,7,10,…]`，**非恒等**）。曾经 obs 用 standard ⇒ 闭环 50 步发散；目标用 standard ⇒ 动作打到错关节；重置用 standard ⇒ 每 20 步瞬移到畸形态 |
| **3** | **一切不变量进契约门**（`scripts/verify_contracts.py` + 采集器运行时硬门禁），不过门不跑 | 已被"自洽地写错"骗过两次（`states==next_states`；"读回偏差 0.0000" 是循环论证） |
| **4** | **奖励 / 真回报 / critic 三者必须同口径**（`huber_err` + `split_huber` + 同一组 δ）。`saturating` 只允许做对照臂，且必须在 launcher 显式声明 | 第 1/2 轮用了 `saturating`+`split`+δ=0.10，与 `real_returns`/critic 的既定口径不一致，目标尺度错配 |
| **5** | **`WM_CKPT` / `REWARD_MODE` / `ADV_BASELINE` / `ACT_EPS` 在 launcher 启动时断言**，不允许落到"默认值" | 第 1 轮 `WM_CKPT` 为空 ⇒ 微调根本没加载刷新后的 WM；`ACT_EPS=0.10` ⇒ hinge 100% 出界。两次都是"默认值坑" |

术语：**ξ** = 机器人当前状态相对参考帧的偏移（root 位置差）；**segpos** = 段内第几步（0–19）；**H** = 微调/部署的滚动窗口 = 20；**δ** = 微调起点 ξ 的上限。

---

## 1. 采集（A'）

### 1.1 环境与执行器

| 项 | 值 |
|---|---|
| 仿真 | Isaac Sim 5.0 + IsaacLab 2.2，HA 任务场景（6 个任务各自一个场景） |
| 控制频率 | 50 Hz（`dt=0.02`，4 个 `0.005` 物理子步） |
| 执行器 | `--executor sonic` + `--sonic-decoder .../release/model_decoder.onnx` |
| 观测 | 994 维 = `[token(64)] [ang_vel(10×3)] [q−SONIC_DEFAULT(10×29)] [qd(10×29)] [last_action(10×29)] [gravity(10×3)]`；历史 10 帧、**新帧在最后**；`last_action` 是**上一步**的 decoder 原始输出（obs 在动作更新前拼接，t=0 全零） |
| 动作 | `q_target = SONIC_DEFAULT_POS + a × G1_ACTION_SCALE_ISAACLAB`（**SONIC 序**），按 `idx_s` 写入关节，clip 到关节限位 |
| 参考库 | `runs/isaac_motion/ha_cmd_exec/packed.pt`（= 录制 npz 的 `robot_qpos_before_decimation`，**SONIC 序**，fp16 量化） |
| token 流 | `runs/exp_tokens/<clip>.npy`（录制 SONIC encoder 的 64 维 latent，与帧 1:1） |

以上每一条都有离线数值证据，见 `scripts/verify_contracts.py` 的 22 项断言（`target = raw*scale+default` 的 max|Δ| = 0，`obs q 块 = qpos − SONIC_DEFAULT` 的 max|Δ| = 0 等）。

### 1.2 窗口枚举（覆盖口径）

```
每条 clip（长度 L）：
  窗口起点 = 0, STEPS, 2·STEPS, …  （STEPS = 780）
  若 L > STEPS 且末窗未盖住末尾，再补一个 start = L − STEPS 的尾窗口
长度门槛：L ≥ LEN_MIN（260）—— 与 STEPS **解耦**
```

| 参数 | 值 | 说明 |
|---|---|---|
| `STEPS` | **780** | 一个进程连续跑多少步（= 枚举跨度）。780 = 3 个 260 帧的覆盖块，启动开销摊薄 3 倍 |
| `LEN_MIN` | **260** | 最小可用 clip 长度。**不能再等于 STEPS**：270→810 若同时当门槛，可分 clip 从 599 掉到 221（36.9%）、帧覆盖只剩 50.8% |
| `--segment` | **20** | 段重置间隔 |
| `--num-envs` | **64** | 实测上限（128/255 卡死；与相机无关，是场景构建成本） |
| 覆盖结果 | 599/599 clip、484,386/484,386 帧 = **100%** | 875 个窗口 → **15 个进程** → 4 卡 **~7 min** |

> 关键点：`frames = 库起点 + start + step` 是**绝对帧号**，所以一个 env 连跑 780 步天然跨 3 个相邻窗口，**覆盖粒度不变、零数据损失**。

### 1.3 一个进程的时间线

```
启动 Isaac（~60 s：USD / 场景 / 64 个 env）  ← 无法避免，所以要让 STEPS 尽量大
warm-up 20 步（不录制）：让 state / token / 参考帧三者相位对齐
主循环 780 步（~33 s）：
    每步：记录 (s_before, a, s_after, frame, clip)
    每 20 步：换 token、重抽噪声、瞬移回参考帧
保存 → 契约自检 → 硬退出（Isaac teardown 会挂死 4 分钟）
```

### 1.4 一个 20 步段内部（8 步伪代码）

```python
for step in 0..779:
    # ① 抓 s（步之前）
    s_before = [root_p(3), root_quat(4), q[SONIC序](29), v(3), w(3), qd[SONIC序](29)]  # 71
    # ② token：每 20 步换一次（= 部署 receding horizon 0.4 s）
    if step % 20 == 0:
        tok = exp_tokens/<clip>.npy[clip_start + start + step]      # (64,)
    # ③ 拼 994 维观测（10 帧历史，q 块存 q − SONIC_DEFAULT_POS）
    obs = build_decoder_obs(tok, hist_ang_vel, hist_q, hist_qd, hist_last_a, hist_gravity)
    # ④ SONIC decoder → 29 维原始动作
    a = decoder(obs)
    # ⑤ 加噪（详见 §1.5）
    if step % 20 == 0:  delta = N(0, 0.05²)  # (29,) 独立高斯
    a = a + delta
    # ⑥ 动作 → 关节目标（SONIC 序）→ 写关节 → 推进 4 个物理子步
    q_target = SONIC_DEFAULT_POS + a * G1_ACTION_SCALE_ISAACLAB
    write_joint_target(q_target at idx_s); clamp(关节限位); physics_step(×4)
    # ⑦ 抓 s'（步之后），存一行
    save_row(s_before, a, s_after, frame, clip)
    # ⑧ 每 20 步：瞬移回参考帧（SONIC 序，三处一致）
    if (step+1) % 20 == 0:
        write_root_pose(ref[frame+1]); write_joint_state(ref[frame+1] at idx_s)
        write_root_velocity(ref v/w at [7:10],[10:13])
        非循环校验：正序读回 < 0.05 rad 且 错序 > 3×正序   # 违约立刻 SystemExit
```

### 1.5 加噪规范（你要求写清楚的部分）

| 项 | 规定 |
|---|---|
| 加在哪 | **SONIC decoder 的 29 维原始动作输出**上（`a` 之后、`q_target` 之前）。**不是**加在状态上（那是伪造 s'） |
| 分布 | 各向同性高斯 `δ ~ N(0, σ²)`，σ = **0.05**，29 个关节独立 |
| 时序 | **每 20 步重抽一次并保持**（`perturb_hold = 20`）。与 token 刷新、段重置**同相位** ⇒ 每个段是「从参考出发 + 一个固定的随机推力」 |
| 裁剪 | **不裁剪**（`action_clip = 0`）。这个 WBC/decoder 的动作是无界的（\|a\| 最大 26），clamp 到 ±1 会砍掉一半动作、40 步内必倒 |
| 量级依据 | σ=0.05 ≈ 动作幅度（\|a\|≈1.2）的 **4%**，折算关节 ≈0.017 rad。实测对段末漂移的贡献只有 **+0.18 cm**（2.57→2.75 cm）、关节 +0.018 rad |
| 目的 | 让 WM 见过「稍微偏一点」的**物理自洽**转移；**它不是多样性的主要来源**（多样性来自真实跟踪漂移：p50 2 cm / p90 7 cm / p99 16 cm） |
| 为什么不加大 | 加大只会把状态推到「参考解释不了」的地方，而 A' 的奖励/标签都以参考为锚；再大就退化成随机推机器人 |

### 1.6 落盘格式

一行 = 一个转移：
```
states (N,71) [p,quat,q,v,w,qd]   next_states (N,71) 同布局
actions (N,29)                    frames (N,) = 全局库帧号
clip_ids (N,)                     meta = {种子/噪声/扰动/执行器/关节序…}
```
**行是交错存的**：第 i 个 env 的行 = `i % num_envs`（所有下游统计必须按此还原，否则会得到"段内漂移 23.7cm"那种假数）。

### 1.7 采集期硬门禁（不通过就退出，不写数据）

1. 启动：`decoder_q_offset ≡ −SONIC_DEFAULT_POS`（<1e-5）；`idx_s ≠ idx_t`。
2. 每次段重置：正序读回 < 0.05 rad **且** 错序读回 > max(3×正序, 0.05)（反证，防"自洽地写错"）。
3. 落盘前：`max|s−s'| > 1e-2`（有转移）、`z_min > 0.20`（没摔）、全 `finite`。
4. 驱动器：**只认产物文件是否存在**（并行 Isaac 撞共享内存会 rc=0 但没产物）。

### 1.8 失败处理

单进程超时 900 s；产物缺失即视为失败并**自动重跑**（驱动器幂等：按文件名跳过已有分片）。

---

## 2. 合并与过滤

### 2.1 逐行漂移（唯一正确的残差口径）

```
err_t = ‖ (p_{t+1} − p_t) − (ref[f+1] − ref[f]) ‖      # 每步位移与参考位移之差
keep  = err_t ≤ 5 cm
```

- **不要**用"连续 20 行 = 一段"去统计（行是交错存的，那样算出来的是跨 env 的假段）。
- **不要**把跨窗口/跨段的长程累积漂移当质量指标（漂出去的轨迹也可能是合法解）。

### 2.2 阈值依据（新口径实测）

p50 = 0.44 cm、p90 = 1.17 cm、>2cm 占 8.4%、>5cm 占 0.5%。取 5 cm 只丢 0.5%，专门剔除接触失败/异常段。

### 2.3 覆盖校验

合并后必须报告：`覆盖 clip 数 / 可用 clip 数`、`帧覆盖 %`、`总行数`、`过滤丢弃 %`。任何一项低于 100%/99.5% 就要查。

---

## 3. WM 训练

### 3.1 结构与口径

| 项 | 值 |
|---|---|
| 世界模型 | `mlp`，`explicit_dt0.02` 积分（从 WBC ckpt 继承结构） |
| 输入 | `(s 71维, a 29维, xi)` → `s' 71维` |
| 训练数据 | §2 的 `filtered.pt`（A' 真实转移） |
| 单步超参 | steps 6000、batch 2048、lr 1e-4、`vel_weight 5.0`、`kl_anchor 1e-3` |

### 3.2 三个阶段

| 阶段 | 作用 | 本次是否开 |
|---|---|---|
| **阶段 2（单步 mix）** | 在 A' 全量上单步拟合（`t → t+1`，state 永远取真实值） | ✅ 开，6000 步 |
| **阶段 2b（k=20 自回归）** | 每步喂模型**自己的输出**，逐 20 步对齐真值 ⇒ 约束**累积误差** | ✅ **必须开**（以前被跳过，见下） |
| 阶段 3（纯 B 线幅值正则） | 压 `mean|qd_pred|/mean|qd_ref|` 到 1 | ❌ 无 B 线数据，不开 |

**为什么阶段 2b 必须开**：单步损失里 state 永远是真实状态，模型从没见过自己的误差。文档实测：k=1 的 `qd_ratio 0.977`（门禁只量这一步，永远 PASS）而 k=20 是 **1.329**。跳过 2b ⇒ WM 的 20 步想象漂移系统性**低估** ⇒ 用「想象的漂移」当奖励去微调，优化的是个乐观模型，传导不到真实。

### 3.3 阶段 2b 的窗口抽取（关键，容易写错）

需要 `S (N, 21, 71)` 和 `A (N, 20, 29)`——**连续 21 帧真实状态 + 20 个真实动作**。

A' 的每个 20 步段天生就是这样的窗口，但抽取时有**两个坑**：

1. **不能取相邻行**：分片行是交错存的（`row i = env i%64`），`st[t:t+21]` 是 21 个不同 env。
2. **不能跨段**：段边界是一次瞬移（物理上不连续），跨过去就是假转移。

抽取规则：
```
按 (clip_id) 分组 → 组内按 frames 排序
段边界识别：|q − q_ref| 在段首 < 0.03 rad（重置成功），据此切段
只取完整落在同一段内的 21 帧（段长恰好 20 ⇒ 每段正好 1 个窗口）
```
超参：`roll_steps 1500`、`roll_k 20`、`roll_lr 3e-5`、`roll_batch 64`、`roll_mag_weight 30`、`roll_sim_weight 0.3`。

### 3.4 多 seed + 固定评测集选优（必须）

`refresh_wm` 的 minibatch 采样曾**没有播种**，同一份数据跑两次 `qd_corr` 差 0.16（0.67 vs 0.51），比"改进"本身还大。

规范：
- `pretrain_wm --seed {0,1,2,3}`，产物写 `refresh_recipe`（含 seed）。
- 评测集**固定**：同一份 `filtered.pt`、`batch=65536`、`seed=20261103`（`scripts/eval_wm_refresh_ab.py`）。
- 按 held-out `qd_corr` 选最优，拷为 `wm_pre_full.pt`。

### 3.5 验收指标（两段）

**A. 单步质量**（`eval_wm_refresh_ab.py`）：`qd_corr` 越高越好、`qd_ratio` 越接近 1 越好；对照 = 刷新前的基线 ckpt。

**B. 累积质量（这一条才是关键，新增）**：在同一批 held-out 20 步段上，
```
imag_drift_20 = ‖p_pred(20) − p_start‖ ,  real_drift_20 = ‖p_real(20) − p_start‖
验收：imag/real ∈ [0.8, 1.2]（p50 与 p90 都要看）
```
**只有这个比值接近 1，微调「降想象漂移」才会真的降到真实漂移上。** 若阶段 2b 后仍 < 0.8，说明数据/容量不足，此时才讨论改建模（如给 VLA 加物体位姿输入）。

> **硬门**：`imag/real ∉ [0.8,1.2]` ⇒ **不进入 §4**。理由：微调的奖励项全部由 WM 的 20 步想象产生，
> WM 分不清 2 cm 和 6 cm 的话，后面所有奖励/权重/信任域的调整都是在不准的裁判上做优化。

---

## 4. 联合微调

### 4.1 参与组件

| 组件 | 状态 |
|---|---|
| VLA（Cosmos-Reason2-2B + action head LoRA，rank 32，targets `attn1,attn2,ff`，另训 `proj_out`） | **训练** |
| WM（来自 §3 的 `wm_pre_full.pt`） | **冻结**，但提供**可微**的想象 rollout（梯度回传到 token） |
| Critic（`critic_real.pt`，真回报拟合） | 训练/使用（`CRITIC_ON=1`） |
| WBC / SONIC decoder | 不参与（离线数据已由它产生） |

**必须显式传 `WM_CKPT`**（第 1 轮的教训：默认空 ⇒ 微调根本没加载刷新过的 WM）。

### 4.2 奖励函数（**这是 v3 的奖励口径，不是旧的 exp 版**）

奖励在 `docs/JOINT_MULTIROUND_PLAN_20261103.md` §2 重写过，当前实现：
`REWARD_MODE=huber_err` + `ROOT_ORI_MODE=split_huber`（`diff_vla_wbc/isaac_joint.py:583-599/641-661`）。

**基本量** `e = 平方误差`，取 `d = √e`（米 / 弧度）：

```text
h(d; δ) = 0.5·d²/δ      if d ≤ δ      # 近场二次，梯度 d/δ（在 0 处平滑）
          d − 0.5·δ     if d > δ      # 远场线性，梯度恒为 1（不饱和、无上界）

r_t = −[ w_body_pos·h(d_body_pos; δ_p) + w_body_ori·h(d_body_ori; δ_o)
       + w_root_pos·h(d_root_pos; δ_p) + w_root_ori·h(d_root_ori; δ_o) ] · dt
      + w_action_rate·r_rate·dt + w_joint_limit·r_lim·dt
```

- 朝向拆三项（`split_huber`）：`tilt 1.0 / yaw 0.2 / yaw_rate 0.3`（token 是 yaw-relative，绝对 yaw 降权）。
- **速度项不用**（`reward_vel_scale = 0`）；posture / hinge 尾部已删除（`tail_weight=0`）。
- `w_action_rate=-0.1`、`w_joint_limit=-2.0`、`dt=0.02`。
- **body 项的参考是重锚定到机器人自己的 xy+yaw 的**（BeyondMimic）；root 项的参考是**绝对**的 ⇒ root 位置漂移由 `d_root_pos` 单独承担。

**为什么不用 `saturating`（=旧的 `exp(−d²/σ²)`，ckpt 里 σ=0.3 m）**：

| 误差 d | `saturating`(σ=0.3) | `huber_err`(δ=0.05) |
|---|---|---|
| 2 cm（实测 p50） | 0.9956 | −0.0040 |
| 6.3 cm（p90） | 0.9608 | −0.0383 |
| 12 cm | 0.852 | −0.095 |
| 30 cm | 0.368 | −0.275 |
| 50 cm | **0.062（趋平）** | −0.475（线性继续） |

- **在 2–12 cm 这一段两者判别力相当**（所以要诚实：换核**不是**"信号回来了"）。
- 换核的真正理由是**远场**：开 `start_source=recorded` + δ=0.10 后，起点本身可到 10 cm，再叠 WM 的 20 步模型误差，想象误差经常跑到 20–50 cm；此时 `saturating` 已经掉到 0.06–0.37 且趋平（**分不清"很差"和"灾难"**），而 `huber_err` 保持恒定梯度、惩罚无上界。**这是鲁棒性选择，不是主信号修复。**
- 附带好处：奖励、`real_returns`、critic 三者口径一致。

`δ` 退火：`δ_p = δ_o = 0.05`（R1）→ `0.02`（R2）→ `0.005`（R3）。

### 4.3 总损失与梯度路径

```text
L = λ_r · [ −( ret + boot ) ]          # ① 奖励项（唯一"真目标"）
  + λ_bc · L_fm                        # ② 标签回归（锚，不是目标）
  + w_act · hinge(act_rms / act_eps − 1)   # ③ 动作信任域（约束，不教东西）
  + λ_hand · L_hand                    # ④ 手部专项（只监督 14 维）
  + trust_weight · hinge(...)          # ⑤ 策略锚（防遗忘）
```

**① 奖励项的三条梯度路径**（这是回答"root/body 位置的梯度在哪"）：

```python
rollout = imagine(tokens=VLA(obs), start_state, references, H=20)   # WM 可微
ret     = discounted_return(rollout, γ)                             # 20 步 imagined 回报
vH      = critic(rollout.states[-1], ref_states[-1])                # 想象末状态的价值
v_ref   = critic(ref_states[-1],     ref_states[-1])                # 参考末状态的价值
rel     = min(vH − v_ref, 0)          # ★ 单边、相对参考：只罚"比参考差"
boot    = γ**H · survival · rel
loss_r  = −(ret + boot)
```

- `ret` 里每一步都由 `reward(s_t, ref[f+t])` 组成 ⇒ **root 位置、root 朝向(tilt/yaw/yaw_rate)、30 个刚体的位置/朝向、动作平滑、关节软限位** 全部进梯度；
- body 项多一层 `q → FK → body_p`（FK 可微）；
- **`boot` 的语义**：不是"尾部绝对价值"，而是 **"在第 20 步把状态带得比参考差多少"** —— 即"别把烂状态交给下一个重规划窗口"。部署是窗口链，所以这一项有意义。
- **`boot` 的权重问题**：`γ=0.99, H=20 ⇒ γ^20 = 0.82`，实测 `ret+boot ≈ 1.57` ⇒ **bootstrap 占目标约一半**。而 critic 是在**真实状态**上拟合、被用在**想象状态**上。所以规定：
  **`γ` 从 0.99 降到 0.95（γ^20 = 0.36）**，让 20 步内的 imagined 回报成为主信号；并且开 `CRITIC_ON` 前先量"critic 在想象末状态上的估值误差"。

**② `λ_bc` 不能设为 0（有实验依据）**。`v3_step.py:648` 记录：去掉 token BC 之后
> "6 轮训练全在 π₀ 附近震荡（合并 n=24 反而更差 0.101 m）"

因为奖励项只给"往哪偏一点点"的局部梯度（实测 VLA 相对 GT 的优势仅 −0.014，返程尺度 1.5 ⇒ 1%），**不回答"这个 drifted 状态下该输出什么"**。规定 `λ_bc ∈ [0.05, 0.1]`。

**③ `act` 信任域的单位**：`act_rms` 是 **29 维动作的 RMS**，不是米/弧度。
- `act_eps=0.10` ⇒ 100% 出界、`loss/act≈8`（第 1 轮实测）；
- `act_eps=0.40` ⇒ `frac_out=0.45`（第 2 轮实测，act/rmse 均值 0.557）；
- 规定 **`act_eps = 0.50`**，并把 **`act/frac_out ∈ [0.3, 0.5]` 当硬监控**（不满足就调 ε，别拍脑袋）。

**④ `λ_hand`**：`loss_hand` 是 **hand-only flow-matching，掩码只有 `64:78`（14 维手）**，64 维 token 完全不受它监督（`v3_step.py:641-644`）。因为**奖励里没有手部项**，而 `λ_bc` 降到 0.1 后手部监督会不够 ⇒ 规定 **`λ_hand = 0.05`**。

**⑤ 策略锚**：`trust_center = "prev"`（冻结本轮的 π_k，用**同一条噪声轨迹**采样比较，等价 KL 的抽样估计，`v3_step.py:196-203/739-746`）。多轮迭代时这是最合适的中心；`label` 等于再做一次 BC、`base` 只锚预训练策略。`trust_weight` 取 0.05 量级。

**明确不用的东西**（避免又走回头路）：

| 项 | 规定 | 理由 |
|---|---|---|
| `adv_baseline` | **`none`** | 目标是"最大化想象回报"，不是"赢过 GT token"。detached 的 GT 回报是控制变量，**在无 tanh 时对梯度没有任何影响**；只有开 tanh 才会把目标变成"赢过演示"——那不是我们要的 |
| `adv_tau` | **0**（关 tanh） | 同上；开了只会引入一个必须匹配 advantage 尺度的魔法阈值 |
| 当 critic 要降方差时 | 用 `adv = ret − V̂(s_0)`（critic 基线） | **不用 GT token 当基线** |

---

### 4.4 H=20 与起点分布（本规范的核心设计）

**为什么不从头到尾 on-policy**：一条长轨迹训练时用不上（VLA 每次生成的 token 都不同），而且会漂 1–3 m 采不完。所以用 20 步 teacher forcing，把误差关在信任域内，**逐段降漂移**。

**为什么起点不能是"纯参考"**：部署时第 k 个窗口的起点是第 k−1 个窗口漂移后的残留（实测段内 p50 2 cm / p90 7 cm）。从纯参考出发只能优化"不存在的一类窗口"。

**起点的三条规定**：

| 项 | 规定 | 依据 |
|---|---|---|
| `start_source` | **`recorded`**（A' 的真实状态，按帧索引） | 物理自洽；覆盖 p50 2 / p90 7 / p99 16 cm |
| `start_max_drift δ` | **0.10 m**（硬封顶，只从 ξ ≤ δ 的帧采） | 保留 93.3% 的转移行；比部署段内 p90（7 cm）宽，仍在"参考有意义"的信任域内；避免采到 53 cm 的异常段 |
| `DR`（对参考态叠加的随机扰动） | **缩小**：`xy 0.03 / yaw 0.05 / dq 0.02 / dqd 0.10`（原 0.20/0.25/0.05/0.30） | DR 是"把参考态平移一下"，速度/关节历史不跟着变 ⇒ 物理不自洽。20 cm 是部署实测的 3–10 倍，太大 |

日志必须打印起点分布审计（`[start] drift-cap δ=… 保留…%`）。

### 4.5 训练预算

| 项 | 值 | 说明 |
|---|---|---|
| 步数 | **≥ 3000** | 第 1/2 轮只跑 300 步 × batch 4 × accum 4 = **4800 条序列 ≈ 0.9% 一个 epoch**，那是欠训练，不是"没效果" |
| batch / accum | 4 / 4（4 卡 DDP） | 显存上限决定 |
| 关键监控 | `L_fm`、`token_mse`、`act/frac_out`、起点分布、`imag_delta_win` | |

### 4.6 验收

1. `L_fm` / `token_mse` 在训练集上明显下降（证明优化器在工作）。
2. **`act/frac_out ∈ [0.3, 0.5]`**、`loss/act` 随步数下降。
3. `adv/*` 不打印时至少打印 `critic/v_adv_H`，其量级应与 `imag_delta` 同阶（判断 critic 有没有主导目标）。
4. **部署侧**：闭环成功率 > 基线，且 replan 边界的 ξ 分布**不劣化**（这才是最终判据）。
5. 起点分布审计日志必须出现：`[start] drift-cap δ=… 保留…%`。

---

## 5. 部署闭环

| 项 | 规定 |
|---|---|
| 执行器 | **SONIC decoder**（`HA_ALLOW_SONIC=1` + `action_provider_sonic`），**不再用 SuperMimic provider** |
| VLA 服务 | `scripts/ha_vla_latent64_server.py --ckpt <VLM> --head-ckpt <FT 的 vla_final.pt> --seed 0` |
| 重规划 | `HA_VLA_EXEC_STEPS = 20`（0.4 s），推理实测 244 ms ⇒ 够用 |
| 回合长度 | ≥ 600 步（12 s），建议 1500 步以暴露累积漂移 |
| 度量 | ① 官方 `success` ② 每个 replan 边界的 ξ（p50/p90）③ 是否摔倒（`z_min`、`up`） |
| dump | 每步 `(s, a, s', frame)` 落盘 ⇒ 既是 **on-policy 数据**（可混进 WM 训练），又是 ξ 曲线来源 |

**ξ 回灌规则**（每轮）：
```
δ_{k+1} = max(0.10, p90(ξ_k))          # 让下一轮的起点分布追上部署的真实累积
if p90(ξ_k) > 0.15: 需补一份 leash 采集（|ξ|>δ 才重置，产生"累积漂移"状态）
```
注意：光"测 ξ"不等于刷新 WM；必须把闭环轨迹**落盘并混进 WM 训练集**才算。

---

## 6. 每轮迭代做什么

```
round k:
  1. 采集 A'(STEPS=780, LEN_MIN=260, 64env, σ=0.05, seg=20)
  2. 契约门 → 合并 → 逐行过滤 5cm
  3. 抽 k=20 连续窗口（不跨段、不取相邻行）
  4. WM：阶段2(6000) → 阶段2b(k=20, 1500) → 4 seed → 固定评测集选优
  5. 验收 B：imag/real 20 步漂移 ∈ [0.8,1.2]（不过则不进入训练）
  6. 联合微调：奖励口径 huber_err/split_huber（δ=0.05）+ `recorded` 起点(δ=0.10) + 小 DR
     + `λ_bc 0.05~0.1 / λ_r 1.0 / λ_hand 0.05 / act_eps 0.50 / γ 0.95 / trust_center=prev`
     + `adv_baseline=none, adv_tau=0` + ≥3000 步（先把 `real_returns`/critic 按同口径重建）
  7. 长闭环（≥600 步）+ ξ dump + 成功率
  8. 用实测 ξ 更新 δ_{k+1}；把闭环 dump 混进下一轮 WM 数据
```

---

## 7. 回归清单（每次改动后必须全绿）

```bash
bash scripts/check_all_contracts.sh
```

22 项断言，覆盖：ref 布局切片、decoder offset、994 维 obs 五个分块、`target=raw*scale+default`、无延时后处理、`last_action` 滞后一帧、SONIC 序 ≠ standard 序、`A∘B=恒等`、有真实转移、无摔倒、逐行漂移、段首复位正序小且错序大。

采集期另有 3 条运行时硬门禁（§1.7）。

微调 launcher 的启动断言（铁律 5，缺一不可）：

| 断言 | 通过条件 | 不通过的表现 |
|---|---|---|
| `WM_CKPT` | 文件存在且含 `wm` 权重；打印 `refresh_recipe`（含 seed） | 第 1 轮：静默用了 WBC 自带的旧 WM |
| `REWARD_MODE` | `huber_err`（或显式声明为对照臂） | 第 1/2 轮：`saturating` ⇒ 远场退化 |
| `ACT_EPS` | ≥ 0.20，且训练中 `act/frac_out ∈ [0.3,0.5]` | `0.10` ⇒ `frac_out=1.00`、`loss/act≈8` |
| `ADV_BASELINE` | `none`（或显式声明对照臂） | `label` ⇒ 目标变成"赢过演示"，不是降漂移 |
| `REAL_RETURNS` 口径 | 与 `REWARD_MODE`/δ 一致 | critic 学到另一把尺 |
| 契约门 | `check_all_contracts.sh` PASS | —— |

---

## 8. 已知风险与未解项（诚实列表，不藏着）

| # | 风险 | 现状 | 计划 |
|---|---|---|---|
| 1 | **WM 的 20 步想象漂移可能仍低估真实** | 阶段 2b 以前被跳过，从未训练过累积误差 | 本轮开启并量化 `imag/real` |
| 2 | **手部抓取从不出现**（闭环 `|hand|` 0.05 vs 数据集 0.29，`hand_gt170=0`） | 已排除"状态自锁"假设（`SM_HAND_STATE=cmd` 无效）；放大 8 倍能到 0.75 rad 但仍不成功 | 若 §4 后仍如此，考虑 `λ_hand>0` 或接录制手部流 `--sm_hands_npy` |
| 3 | **token 目标依赖 VLA 观测不到的信息**（录制人体运动） | 实测 token 对 46 帧运动历史窗口末端 R²=0.965（因果历史，不是偷看未来） | 若成功率上不去，考虑给 VLA 加物体相对位姿/阶段输入（改建模，需另开一轮） |
| 4 | **累积漂移分布 vs 训练起点分布** | 微调起点 ξ ≤ δ=0.10；部署 30 个窗口可能累积到 14cm（随机游走）甚至更多（系统偏置） | 靠 §5 的 ξ dump 实测并回灌 δ；必要时补 leash 采集 |
| 5 | 采集吞吐上限（64 env/进程、860 s/15 进程） | 一轮 ~7 min | 若要百万级行：拉长 STEPS 到 1560，或找 HA env 的无相机配置 |
| 6 | 官方 eval `max_reward=0` 容易误读 | 这个 harness 的 Move 任务走 `success` 谓词，`reward_manager` 无 active 项 | 只看 `success` 与 ξ，别看 reward |
| 7 | **`λ_bc` 降到 0.05 后可能"推不动"** | 历史上 `λ_bc=0` 出现 6 轮原地震荡；0.05 没有先例 | 盯 `act/frac_out`、`L_fm` 与闭环 ξ；若不动，升到 0.1 或加 `token_bc_weight` |
| 8 | **`γ` 降到 0.95 后 bootstrap 变弱** | `γ^20=0.36`（原 0.82）⇒ 对"末尾状态变差"的惩罚减弱 | 用闭环 ξ 曲线判断：若 ξ 逐窗上升，说明需要更强 bootstrap，回升 γ 或加系数 |
| 9 | **奖励是"跟踪"目标，不是"任务"目标** | 参考距离奖励不保证抓取成功 | 成功率单独看；要突破需要改建模（§10.3） |

---

## 9. 时间预算

| 阶段 | 时间 |
|---|---|
| 采集（15 进程 / 4 卡 / 780 步） | **~7 min** |
| 合并 + 过滤 + 覆盖校验 | ~2 min |
| k=20 窗口抽取 | ~2 min |
| WM：阶段2(6000) + **阶段2b(k=20, 1500)** + 4 seed 选优 | ~12 min |
| imag/real 漂移验收 | ~2 min |
| 联合微调（≥3000 步，4 卡 DDP；含 real_returns/critic 重建） | ~45–70 min |
| 长闭环 + ξ dump | ~10 min |
| **一轮合计** | **~85–110 min** |

---

## 10. 待你确认的决策点

### 10.1 已按讨论改定的（如需回退请指出）

| 项 | 定值 |
|---|---|
| 采集 | `STEPS=780 / LEN_MIN=260 / 64 env`（100% 覆盖，~7 min/轮） |
| 加噪 | `σ=0.05 / hold=20 / 不裁剪`（实测贡献 +0.18 cm） |
| WM | 阶段 2（6000 步，单步）+ **阶段 2b（k=20 自回归，1500 步）**，4 seed 固定评测集选优 |
| WM 门槛 | `imag/real 20 步漂移 ∈ [0.8, 1.2]` 不过则不进微调 |
| 奖励口径 | `huber_err` + `split_huber`；`δ_pos=δ_ori=0.05`（R1）→0.02（R2） |
| 回报/critic | 与奖励**同口径**重建 `real_returns` 并重拟合 critic |
| 微调起点 | `start_source=recorded` + `start_max_drift δ=0.10` + DR 缩小（`0.03/0.05/0.02/0.10`） |
| 损失权重 | `λ_bc ∈ [0.05,0.1]`、`λ_r=1.0`、`λ_hand=0.05`、`w_act=0.2`、`act_eps=0.50` |
| 优势/基线 | `adv_baseline=none`、`adv_tau=0`（**不用 GT 当基线、不开 tanh**） |
| bootstrap | 保留（单边、相对参考），但 `γ: 0.99 → 0.95`（`γ^20 = 0.36`）让 imagined 20 步为主信号 |
| 策略锚 | `trust_center=prev`（冻结本轮 π_k，同噪声采样比较），`trust_weight≈0.05` |
| 微调步数 | ≥ 3000；eval ≥ 64 窗口 |
| 部署 | 切 **SONIC decoder provider**，验收 = 成功率 + replan 边界 ξ |
| 门禁 | 契约门 22 项 + launcher 启动断言（`WM_CKPT`/`REWARD_MODE`/`ADV_BASELINE`/`ACT_EPS`） |

### 10.2 还需要你拍板的 3 个

1. **`γ`**：按上面用 **0.95**（一个参数搞定 bootstrap 降权），还是保持 0.99 + 额外乘 0.3~0.5 的 bootstrap 系数？
2. **`λ_bc` 具体值**：**0.05**（更靠奖励）还是 **0.1**（更稳，接近历史可用配置）？
3. **本轮是否接受"先只把 WM 阶段 2b + imag/real 门槛跑通、暂不动 VLA 权重"**（我建议这个顺序：先确保裁判准，再动策略）。

### 10.3 明确不在本轮范围（避免范围膨胀）

- 手部抓取的"均值塌陷"根因（token 目标依赖 VLA 观测不到的人体运动）——需要给 VLA 加物体位姿/阶段输入，属于改建模。
- 从头到尾 on-policy 采集（已论证会漂 1–3 m、采不完）。
- 真机部署。

---

## 11. 修订记录

### v2.1（2026-11-04 晚，按讨论修正）

| # | 位置 | 改前 | 改后 | 依据 |
|---|---|---|---|---|
| 1 | §0 | 三条铁律 | **五条**（+奖励口径一致、+launcher 启动断言） | 第 1/2 轮的坑都是"默认值坑" |
| 2 | §4.2 | 未写实际奖励口径（暗示 exp 版） | 写清 **`huber_err` + `split_huber`** 的公式、权重、δ 退火；并**诚实标注**"换核不是信号回来的主因，是远场鲁棒性" | `isaac_joint.py:583-599/641-661`；σ=0.3m 与 2–6cm 漂移的尺度对比 |
| 3 | §4.3 | `L = λ_r·(−V̂) + λ_bc·L_fm + …` 一句话带过 | 展开**三条梯度路径**（root/body/FK 全在 `ret` 里）、`boot` 的真实语义（单边、相对参考）、`γ^20=0.82 ⇒ bootstrap 占一半` 的量化 | `v3_step.py:536-564,608-614` |
| 4 | §4.3 | `act_eps=0.40`、`frac_out 0.3–0.7` | **`act_eps=0.50`**、`frac_out ∈ [0.3,0.5]`，并说明它是**动作 RMS**不是厘米 | 第 2 轮实测 `act/rmse` 均值 0.557、`frac_out(0.40)=0.45` |
| 5 | §4.3 | `λ_bc` 可设 0 | **`λ_bc ∈ [0.05,0.1]`（不能设 0）** | `v3_step.py:648`：去掉 BC 后"6 轮全在 π₀ 附近震荡" |
| 6 | §4.3 | —— | 新增 **`λ_hand=0.05`**，并注明它只监督 `64:78`（14 维手） | 奖励无手部项；λ_bc 降权后手部无人监督 |
| 7 | §4.3 | 基线可用 `label`（VLA vs GT） | **`adv_baseline=none`、`adv_tau=0`**（撤回） | detached GT 回报在无 tanh 时**对梯度零影响**；开 tanh 会把目标变成"赢过演示" |
| 8 | §4.3 | —— | 策略锚改为 **`trust_center=prev`** | `prev` 是真正的 KL 型约束（同噪声采样比较），适合多轮迭代 |
| 9 | §4.6 | 验收带 `frac_out 0.3–0.7` | `0.3–0.5` + 新增"起点分布审计必须打印" | —— |
| 10 | §6/§7 | 微调一行带过 | 补全口径 + **launcher 6 条启动断言表** | —— |
| 11 | §8 | 6 条风险 | **9 条**（+λ_bc=0.05 推不动、+γ=0.95 削弱 bootstrap、+奖励是跟踪非任务目标） | —— |
| 12 | §9 | 一轮 75–95 min | 一轮 **85–110 min**（含阶段 2b 与更长微调） | —— |
| 13 | §10 | 7 条待确认 | **10.1 已改定 / 10.2 剩 3 个 / 10.3 明确不在范围** | —— |

### 尚未落地为代码的项（批准后做）

| 项 | 工作量 | 影响 |
|---|---|---|
| `build_wm_roll_windows.py`（抽 k=20 连续窗口，按段切） | ~60 行 | 阶段 2b 的前置 |
| `pretrain_wm` 阶段 2b 支持 3-D 窗口文件 | ~10 行 | 同上 |
| `imag/real 20 步漂移` 验收脚本 | ~80 行 | 微调的硬门 |
| launcher 6 条启动断言 | ~40 行 | 防止默认值坑复发 |
| 部署切 SONIC provider + replan 边界 ξ dump | 中等 | 最终验收 |
