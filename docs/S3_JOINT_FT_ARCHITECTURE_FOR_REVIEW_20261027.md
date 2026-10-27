# S3 联合微调：问题清单 + 当前架构（外部评审用，2026-10-27）

> 用途：把**我们踩过的坑、当前的实现、以及为什么这么做**一次性讲清，交给外部评审判断
> 架构是否合理、哪里还有问题。
>
> 阅读顺序建议：§1 目标与方法定位 → §2 问题清单（含实测数字）→ §3 当前架构（逐层）→
> §4 尚未解决的问题（请重点评审）→ §5 附录（仪器口径与可复现命令）。

---

## 1. 目标与方法定位

### 1.1 我们想做什么

联合微调一个 **VLA（GR00T N1.7）** 与一个**全身控制器（SuperMimic WBC）**，让 VLA 输出的
64 维运动 token 不只是"像人类演示"（模仿已接近满分），而是"这台身体真的能执行、能把任务做成"。

端到端链条（**A 方案**，全程可微，仿真器不在梯度路径上）：

```
VLA(backbone 冻结 + action head 可训)
   └─► 64d token ─► WBC(冻结, SuperMimic) ─► 29d 动作
        └─► 可微世界模型 WM(冻结) ─► 状态序列 ─► 解析跟踪奖励 ─► ∂J/∂θ 回传到 action head
```

### 1.2 参考实现（SuperMimic）是怎么做的

源码：`third_party/SuperMimic-Reproduced/src/supermimic/`

训练循环（`training/loop.py`，论文 Algorithm A.1）：**collect → fit world model → improve policy**

| 步 | 内容 | 关键参数 |
|---|---|---|
| ① collect | Isaac Lab 全并行，当前策略在跑，真物理给出 `(s,a,s',r)` 进 GPU replay buffer | `num_envs=4096` |
| ② fit WM | 每 UTD 在真转移上拟合 WM（速度通道标定） | `updates_per_env_step=0.4`，window 8 |
| ③ improve policy | **在冻结 WM 的想象里**做 actor-critic | `horizon=32`, `batch=4096`, γ0.99, GAE λ0.95 |

目标函数（`imagination/objective.py`，原文注释）：

```
J = mean_n [ Σ_t γ^t c_t r_{t+1} + γ^H c_H V_H ]      (SHAC-style; V_H keeps the gradient w.r.t. s_H)
```

**要点**：SuperMimic **不是 PPO**（`policy_trainer.py` 里没有 ratio/log_prob/clip_ratio，
Policy 输出确定性动作）。它靠的是：①WM 准（4096 环境真转移持续拟合）②batch 大（4096，把单批
模型误差平均掉）③短视界+价值自举。

### 1.3 我们与它的差距（这是全部问题的根源）

| 维度 | SuperMimic | 我们 | 后果 |
|---|---|---|---|
| 每样本要穿过的参数 | 4.8M（小 MLP） | **~3.6B（2B VLM + 1.62B DiT）** | 每样本算力约 750× |
| batch | 4096 | 4（LoRA 后 32，×accum 2 = 64） | 梯度噪声大 |
| WM 拟合数据 | **策略自己的** on-policy 真转移 | HA 控制器（sonic）的 off-policy 轨迹 + 参考库 | 裁判在"我们的域"外 |
| 真奖励 | 仿真器算 | **WM 想象里算** | 可能被利用（reward hacking）|
| 策略在环 | ✅ 仿真器在环 | ❌ 环里没有仿真器 | 无法在线纠正 |

---

## 2. 问题清单（现象 / 实测 / 根因 / 解法 / 状态）

> 状态标记：✅ 已解决 · 🟡 部分解决 · ❌ 未解决（评审请重点看 ❌/🟡）

### P1 ✅ 训练在 step ~140 崩溃（CUDA device-side assert）
- **现象**：两臂在相近步数同时死，`burnin.py` 的 `sample_burnin` 报 `device-side assert`。
- **根因**：`runs/wm_real/ha_transitions.pt` 是**按转移存行**（633,317 行，每个 clip 末帧没有后继），
  而代码用**帧号**（0‥634,015）索引它 → 尾部越界；不越界时也读错行（起点状态来自别的 clip）。
- **解法**：用 transitions 自带的 `frames` 字段建"帧→行"表，把 699 个 clip 末帧用同 clip 上一帧补上；
  并在 `burnin.py` 加 fail-fast（行数 ≠ `library.num_frames` 直接报错）。
- **验证**：`states_full[frames] == states` 逐元素相等；`recorded_states 按帧对齐: (634016, 71)`。

### P2 ✅ 手部 BC 污染 token（用户关键指正）
- **现象**：`λ_r=0` 的对照臂（**没有任何奖励梯度**）`tok_mse` 也从 ×1.00 漂到 **×1.39**。
- **根因**：手部 14 维与 token 64 维共享同一个 action head 的 trunk；手部 BC（`λ_hand=0.2`）
  的梯度是奖励梯度的 **37 倍**（`grad/r_over_hand = 0.027` 的倒数），通过 trunk 串扰 token。
  而部署时手部**根本不进 WBC**（HA provider 里走独立的 `_hand_queue` 14 维通道）。
- **解法**：`λ_hand = 0`（body-only）。代价：手会塌（`hand170 0.093→0.017`），部署抓取需另加独立支路。
- **验证**：关掉手部后，`λ_r=0` 臂的 `tok_mse` 稳定不动；漂移只剩奖励这一条通路。

### P3 🟡 Reward hacking：无约束的奖励梯度必然把 token 推离流形
- **现象（三次复现）**：
  | 配置 | `tok_mse` | `ret_vla` vs `ret_gt` |
  |---|---|---|
  | 无约束、DR 开（`s3_v3_noBC`）| **×52.9** | 0.853 vs −0.093 |
  | 无约束、DR 开（`s3_v4_noBC`）| ×22.6 | 0.240 vs 0.111 |
  | 无约束、H=20（A1）| ×3.88 | +0.403 vs +0.171 |
- **根因**：奖励在 WM 的想象里算，WM 在外推区会给"假的好分数"；**那个方向对优化器最便宜**。
- **已有解法（三代，逐步加强）**：
  1. `λ_a` 先验锚定（`0.05·MSE(token, prior)`）——**失败**：数值上只有奖励项的 1/300。
  2. **token 空间 hinge trust region**（`relu(RMSE/eps−1)²`）——**部分失败**：`eps=0.15` 时
     `trust/frac_out` 全程为 0（墙从未触发），而 token 的 RMSE 只走到 0.114 就已把 `tok_mse` 顶到 ×2.9。
  3. **动作空间 hinge trust region**（现用）：中心 = **用标签 token 跑同一段 rollout 得到的动作序列**，
     约束 `RMS(a − a_ref) ≤ eps`。理由：决定 WBC/WM 输入的是**动作**，不是 token；
     实测底模自身动作偏差 0.21，取 `eps=0.30`（1.4×）或 0.60。
     - 实测 `act/frac_out`：eps=0.30 → 0.5–0.9（顶住）；eps=0.60 → 0.0–0.2（几乎不顶）。
- **仍未解决**：约束一旦真的顶住，奖励分支的净作用就接近 0（见 P4）。

### P4 ❌ **核心开放问题：短视界没信号，长视界必作弊**
- **现象（同一套护栏下只改视界）**：
  | 训练/评测视界 | `ret_vla` 相对 GT | `tok_mse` | 判定 |
  |---|---|---|---|
  | **5** 步 | **+2.0%**（峰值 +3.3%）| ×1.43 | 无 hack，但收益在噪声内 |
  | **20** 步（eps 0.60）| **+12.7%** | ×1.4 | 疑似 hack |
  | **20** 步（eps 1.00）| **+24.0%** | ×2.0 | 疑似 hack（约束越松涨越多）|
- **根因**：要优化的差异（VLA vs GT）在想象回报上只有 **0.6%–2%**；而 WM 的误差与被追踪信号的
  量级之比在 3% 以上 ⇒ **裁判的分辨率不足以分辨要优化的信号**；视界一长，模型误差累积并被放大成"分数"。
- **为什么不能靠调约束解决**：约束只能限制"作弊幅度"，不能提高"裁判分辨率"。
- **候选解法（按可信度）**：
  - **A. 用真物理当裁判/环境**（Isaac，见 P17）——唯一能提高分辨率的路；
  - **B. best-of-K 筛选 + 蒸馏**：不穿梯度过 WM，只在邻域采样候选、用真物理排序后蒸馏；
  - **C. 提高信号**：换更动态的动作片段，让 VLA 与 GT 的差距从 0.6% 变成几个百分点。

### P5 🟡 WM 的域不一致（裁判在"别人的域"里训练）
- **现象**：同一仪器（`eval_wm_velocity`，ha_cmd 库，label token）：
  | WM | `qd_corr` | `qd_ratio` | pos_err | rot_err |
  |---|---|---|---|---|
  | 原始（未标定）| **−0.042** | 10.95 | 6.0 mm | 3.53° |
  | C6（在 ha_tokens 标定）| 0.217 | 3.64 | 2.4 mm | 0.81° |
  | **C6h（在 ha_cmd 标定，现用）** | **0.752** | 1.115 | 1.5 mm | 0.687° |
- **根因**：C6h 的 WM 是在 **HA 的 sonic 控制器**的执行轨迹 + 参考库上标定的；我们的 WBC 的真实
  状态分布从未被覆盖。
- **解法**：`collect`——让我们的 WBC 在 Isaac 里走，采 `(s,a,s',r)`，再拟合 WM。**尚未完成**。

### P6 ✅ in-loop WM 刷新有害（已关闭）
- **现象**：用 `ha_transitions` 刷新 200 步后，同一仪器：
  `qd_corr 0.752 → **0.589**`，`qd_ratio 1.115 → 1.700`，rollout reward 0.0844 → 0.0769。
- **根因**：那份数据是 off-policy（sonic 的执行），刷新把 WM 从"我们的域"拽向"别人的域"。
- **解法**：`wm_refresh_every = 0`，直接复用已标定的 WM。**教训**：SuperMimic 的"每迭代刷新 WM"
  前提是数据 on-policy；我们有数据不等于数据对。

### P7 ✅ critic 从"拟合想象"改为"拟合真数据"
- **现象**：旧 critic（在想象转移上做 GAE）`critic/v_ev` 长期为**负**（−2.0 ~ −5.1）。
- **根因**：它的监督目标来自 WM 的想象，等于把 WM 的偏差学进价值函数，再自举回策略梯度。
- **解法**：`scripts/build_real_returns.py` 用**真录制轨迹**算每帧的 `return-to-go`
  （奖励只需要状态，完全不经 WM），critic 在这个真目标上训练，并**按 clip 去时间趋势**
  （否则 V 会变成"剩余步数"，诱导策略"让状态看起来像更早的帧"= 滞后）。
- **验证**：`critic/v_ev` **+0.65 ~ +0.81**（3 步内转正）。

### P8 ✅ 评估视界与部署视界不一致
- **现象**：训练用 H=5 时，评估也用 5 步，`survival = 0.9999` —— 完全测不到"会不会摔"。
- **解法**：`--eval-horizon` 与 `--horizon` 解耦；**评估固定 20**（= 部署 `HA_VLA_EXEC_STEPS=20`）。
- **附带修正**：评测 provider 的参考帧数必须够 eval horizon（否则 `need 20 reference frames, got 5`）。

### P9 ✅ 评测窗口太少（16 → 64）+ 增加配对统计
- **现象**：16 个窗口上 2% 的差异无法判断。
- **解法**：`--eval-batches 16` × batch 4 = **64 窗口**，并输出逐窗口配对统计：
  `imag_delta_mean`（VLA−GT 的均值）、`imag_delta_std`、`imag_delta_win`（VLA 胜率）、`imag_delta_win_n`。

### P10 ✅ batch 太小（4）→ LoRA + 混合精度
- **根因**：全量训 1.62B = 权重 6.5G + 梯度 6.5G + AdamW 动量 13G ≈ **26 GB**，加冻结 backbone 8G
  与激活，48 GB 卡只能 batch 4。
- **解法**：
  - `diff_vla_wbc/lora.py`：只在 216 个 linear 上挂 LoRA（`B` 零初始化 ⇒ 训练起点与底模**逐位相同**）。
    `r=32` → 可训 **33,554,432 / 1,654,070,400（2.03%）**，显存代价 0.33 GB。
  - **bf16**：GR00T 的 ckpt 原生就是 `dtype=bfloat16`（我们之前全程 fp32，既非原生口径又慢）；
    现在 backbone 前向 + head 的 FM 采样/损失走 autocast，主权重 fp32，可微 rollout 仍 fp32。
  - 实测：batch 32 + H=20 + K=4 ≈ 16–18 s/step（fp32 同条件约 35 s）。
- **注意**：LoRA 省显存、不省算力（反向仍要穿过全模型）；速度来自 bf16。

### P11 ✅ 采样步数必须与 GR00T 原生一致
- **事实**：`checkpoint-8000/config.json` 里 `num_inference_timesteps = 4`。
- **结论**：`--fm-steps 4` 是原生口径，**不可为提速降到 2**（我一度提议 K=2，已撤回）。

### P12 ✅ 双仪器混用导致的口径混乱
两套仪器**不可互换**：

| 仪器 | 测什么 | 典型值 |
|---|---|---|
| `eval_wm_velocity.py`（ha_cmd + label + policy rollout）| WM 一步预测 vs **真实录制状态** | C6h **0.752** |
| `finetune_wm_velocity.py` 的 check（`corr(pred_qd, ref_qd)`，数据是**参考库**，`a_t = dq_t`）| WM 拟合**参考轨迹**的自洽性 | C6 **0.906** |
| `eval_wm_real`（ha_transitions 随机批）| off-policy 一步误差 | 0.545 → 刷新后 0.743 |
| `runs/S1_ctrl/*_cmd_eval.json`（699 clips 部署口径）| WBC 策略质量 | A@20000 **0.8673** |
| `S1_ctrl/haTOK_*`（旧 ha_tokens 口径）| 已废弃 | A@20000 0.6178（**方向与真口径相反**）|

**重要澄清**：那个 0.906 **不是"WM 预测真机动力学"** 的指标（脚本 docstring 原文：
`Data: reference transitions (S_t, a_t = dq_t, S_{t+1}) from the packed HA motion library`），
所以它**不能**替代 collect。

### P13 ✅ 执行器选择的证据链（结论：现用 C6h 正确）
- 部署口径（ha_cmd，699 clips）：**A@20000 = 0.8673，0 摔** > A@18500 0.8604 > A@16120 0.8203；
  B@14800 0.8675（与 A@20000 打平）。
- 旧仪器（ha_tokens）给出的"A 系越训越差"是**测反了**（A@16120 0.7094 > A@20000 0.6178）。
- C6h = A@20000 的策略（`pi_md5=993c8f7a`）+ 在 **ha_cmd 域**重标定的 WM（`wm_md5=bd045ffb`）。

### P14 ✅ 闭环 harness 的结构性低效
- **现象**：一次 300 步 episode 要 ~10 分钟，其中 **62–106 s 是 Isaac 启动**（旧 harness **每个
  episode 重启一次**），且 token 走 **HTTP** 服务。
- **解法**：改常驻进程 + 同进程取 token（回放 npy），或用 `--episode_batch_json`（单进程多集复用场景）。

### P15 ✅ 进程生命周期（工具调用会把后台任务连带杀掉）
- **现象**：两次训练"启动 20 秒后静默死亡"，无 traceback、无 core。
- **根因**：`nohup ... &` 没有脱离进程组，工具调用结束时整组被 kill。
- **解法**：一律 `setsid nohup ... & disown`；杀进程用 `pgrep -f "[t]rain_joint_v2.py"` 的括号技巧，避免匹配到自己的 shell。

### P16 ✅ token 前端口径（S0 审计结论）
- 标签 token = npz 的 `g1_motion_token_64`（= 录制 encoder latent）；
- **hold 窗口** = HA 录制/部署口径；**future 窗口** = 我们早期的错误装配（与标签 corr 仅 0.057，已退役）；
- 参考库必须用**命令流** `runs/isaac_motion/ha_cmd/packed.pt`（`kind=command`），
  不能用执行轨迹库 `ha_tokens`（8 帧滞后）。
- 审计脚本每次启动都校验：`wbc_ckpt_window=hold`、`lib_kind=command`、`bc=label`。

### P17 ❌ 本机 Isaac 渲染器不可用 ⇒ 在线环受阻
- **实测**：Isaac Sim 5.0 能启动（`app ready` 62–106 s），物理能跑（256 物体 **15,281 转移/秒**）；
  但 `omni.kit.renderer.plugin: Graphics plugins not available`，带 `--enable_cameras` 时
  PhysX 报 `no suitable CUDA GPU`。
- **为什么这会阻塞**：我们 VLA **是图像条件**的（npz 的 `g1_ego_view` 指向 front RGB mp4，
  训练时 `TaskObservationBuilder` 会解码进 `video.ego_view`）⇒ 在线环必须在 Isaac 里实时渲染。
- **正在做**：装 Xvfb（已装好）+ `DISPLAY=:99` 重试相机。**未验证成功**。
- **回退**：回放模式（token 预先算好，不需要相机）——用于 WM 标定与 GT/ORACLE 对比。

### P18 ❌ collect / on-policy 缺失（这是当前主要缺口）
详见 §3.4。

---

## 3. 当前架构（逐层）

### 3.1 模型栈与冻结关系

| 组件 | 文件 / 选择 | 状态 |
|---|---|---|
| VLA backbone | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000` | **冻结**（backbone_trainable_params_fp32=true，原生 dtype=bf16）|
| VLA action head | 同上 ckpt 的 `action_head`（1.62B，32 层 DiT，`num_inference_timesteps=4`）| **可训**：LoRA r=32 后 33.55M（2.03%）|
| WBC（执行器）| `runs/S1_handover/WBC_A20000_wmvel_cmd.pt`（C6h，`pi_md5 993c8f7a`，iter 20000，obs93，token_window **hold**）| **冻结** |
| WM（可微动力学）| 同 ckpt 内的 `wm`（`wm_md5 bd045ffb`，**ha_cmd 域标定**）| **冻结**（`wm_refresh_every=0`）|
| 奖励 | trackerlab 解析跟踪奖励（`bundle.reward`），6 个 `exp(−e/std²)` 核 + action_rate/joint_limit | 解析、无参数 |
| critic | `TokenCritic`（state71 + ref71 + xi33 → 1），~1.4e5 参数 | **可训**，监督目标 = **真录制 return-to-go**（按 clip 去趋势）|

### 3.2 训练目标（当前）

```python
# 每个 batch：
tokens = FM_sample(action_head, backbone_out, K=4, bf16)          # 64d token（可微）
roll  = unroll(frozen WBC, frozen WM, tokens, H=20)               # 可微 rollout
ret   = Σ_{t<20} γ^t · survival_t · r_t                           # 解析跟踪奖励
boot  = γ^20 · survival_20 · V(s_20, ref_20, ξ)                   # grounded critic 自举
loss  = -λ_r·(ret + boot)                                          # λ_r = 0.1
loss += λ_act · mean(relu(RMS(a_t − a_ref_t)/eps − 1)²)            # 动作空间信任域，eps=0.60
# 手部监督：λ_hand = 0（body-only）
```

其中 `a_ref` = **用标签 token 跑同一段 rollout 得到的动作序列**（无梯度参照）。

**关键超参**：H_train = 20 = H_eval = 20（部署口径）；γ=0.99 ⇒ γ^20=0.818；K=4（GR00T 原生）。

### 3.3 约束层级（我们试过三代，只保留第三代）

| 代 | 形式 | 结果 |
|---|---|---|
| 1 | `λ_a · MSE(token, prior)` | ❌ 劲太小（奖励项的 1/300）|
| 2 | `relu(RMSE(token−prior)/eps − 1)²`，eps=0.15 | ❌ `frac_out` 全程 0（从未触发）|
| 3 | **`relu(RMS(a − a_ref)/eps − 1)²`，eps=0.60** | 🟡 有效（`frac_out` 0.0–0.2 @0.60；0.5–0.9 @0.30），但奖励净作用仍接近 0（P4）|

### 3.4 collect / on-policy（当前设计）

**背景**：SuperMimic 的 WBC 训练时，**仿真器一直在环里**（4096 envs 边跑边采边训），所以它不需要
单独的 collect 数据集；我们的 A 方案**把仿真器换成了 WM**，于是"采集"从环里掉出来了。

**两条轨道（A 已实现未跑通，B 需渲染）**：

#### A. 回放式 collect（不需要渲染器）——正在调试
```
Isaac 进程（scripts/isaac_g1_collect.py，自包含，只 import torch + trackerLab 的 Policy）
  ├─ 场景：地面 + N 个 G1（`g1_29dof_rev_1_0.usd`），200 Hz 物理 / 50 Hz 控制
  ├─ token：从 runs/arm_tokens/{clip}/{arm}.npy **回放**，每 20 步换一个 chunk
  ├─ 冻结 WBC：obs93 = [q−default, dq, ang_vel_b, gravity_b, last_action]，history 10 帧
  ├─ 动作 → `set_joint_position_target(default_q + a*action_scale)`
  └─ 落盘 (s_t(71), a_t(29), s_{t+1}(71), frame)
系统 python（scripts/compute_collect_reward.py，待写）
  └─ 用**训练同一个 bundle.reward** 补算 r（奖励只依赖状态+参考，不需要在 Isaac 里算）
```
- 优点：不需要渲染；三臂（GT/ORACLE/VLA）**只差 token 流**，手部/起点/种子全同 ⇒ 可比性最强。
- 局限：token 是**开环**的（不随观测变化），所以**绝对任务成功率不可信**，但
  ①WM 在"我们 WBC 动作分布"上的标定 ②三臂相对比较 都是有效的。

#### B. 在线环（完全对齐 SuperMimic）——被渲染阻塞
```
每 20 步：VLA(图+状态) → token → WBC → 动作 → 物理 → (s,a,s',r) 进 buffer
每 N 轮：在 buffer 上 fit WM；每 M 轮：A 方案策略更新（冻结 WM 内 H=20 pathwise + γ^H·V）
```
- 规模换算（从 SuperMimic 的 4096 envs / UTD 0.4 缩到 128 envs）：
  等价 UTD = 0.0125 次更新/env step（**每 80 env step 一次更新**）；
  每采集轮 128×20 = 2560 条转移；VLA 推理是瓶颈 ⇒ **~430 转移/秒**，1 小时约 100 万转移。
- 前置：Isaac 渲染器（P17）。

### 3.5 评测（当前口径）

| 项 | 值 |
|---|---|
| 评测视界 | **20 步**（= 部署）|
| 窗口数 | **64**（`eval-batches 16 × batch 4`），固定种子 20261026 |
| 指标 | `token_mse_decoded`、`hand_mse_decoded`、`token_shift`、`imag_return_vla/gt`、`imag_survival`、`imag_state`、**逐窗口配对** `imag_delta_{mean,std,win,win_n}` |
| 评测用 WM | **固定快照**（`_eval_wm = deepcopy(bundle.wm)`），避免训练中 WM 变化污染曲线 |
| 起点 | 真录制状态 + burn-in 10 步 |

### 3.6 监督与环境配置（防止"看起来在跑但口径错"）

- `scripts/s3_audit_config.py`：**39 项**启动前审计（H=20 / chunk 40 / consume 20 / K=4 / codec.decode /
  label+hold+ha_cmd / C6h md5 / burn-in 10 / DR 分桶 / γ^H / 手部掩码 / 固定 WM 快照 / 真回报文件 /
  动作约束参数 / bf16 / LoRA rank …），**有 FAIL 就拒绝启动**。
- 每次运行的日志里有一行 `[AUDIT][v3]`，把全部口径打成一行（token_BC / hand_mask / λ_r / anchor /
  trust / act / dr buckets / start / burnin / push / critic 来源 / γ^H / wm_refresh / eval_H / amp / LoRA）。

---

## 4. 尚未解决的问题（请评审重点看这几条）

### Q1 这个代理到底能不能分辨我们要优化的东西？
- 要优化的量：VLA vs GT 的想象回报差 **0.6%（H=5）/ 2%（H=5 峰值）**；
- 裁判误差：WM 一步 `pos 1.5 mm / rot 0.69° / qd_corr 0.752`，20 步累积量级与被追踪信号的 3% 相当。
- **问**：在 batch 只有 32–64（vs 参考 4096）、每样本算力 750× 的前提下，这条路还成立吗？
  还是应该直接换成 best-of-K + 真物理排序？

### Q2 collect 的两种形态够不够
- 回放式（开环 token）能标定 WM、能做相对比较，但**测不到"闭环纠偏"**；
- 在线环需要渲染，目前不通。
- **问**：如果只有回放式，A 方案（梯度穿 WM）的可信度上限在哪里？

### Q3 bf16 + LoRA 是否引入了未测量的偏差
- LoRA 只改一个低秩子空间（rank 32），bf16 只在 backbone + FM 采样段；
- 已验证：LoRA 初始与底模**逐位相同**；冒烟无 NaN，`act/rmse`/`surv` 正常。
- **问**：是否需要做 bf16 vs fp32 的 token 一致性对拍（同 seed 比 `token_mse` 与 `imag_return`）？

### Q4 手部怎么办
- body-only 是当前口径（`λ_hand=0`），手会塌（`hand170 0.093→0.017`），部署抓取会失败；
- 手部不进 WBC，所以对全身动力学无影响。
- **问**：应该用"独立手部 head / 独立支路"来隔离，还是干脆在部署侧用规则/开环手部动作？

### Q5 评测的统计强度
- 64 窗口、同一 seed、同一批窗口重复评；
- 已证明管线**确定性**（v4 两臂 1000 步逐位相同），所以差异不是种子噪声，但**外部效度**（真物理）未知。
- **问**：在拿到真物理之前，`imag_delta_mean` 这类指标能作为决策依据到什么程度？

---

## 5. 附录

### 5.1 关键实测数字汇总

| 量 | 值 | 来源 |
|---|---|---|
| 训练帧数 / clip 数 | 634,016 帧 / 699 clips | motion library |
| 转移行数 | 633,317（每 clip 末帧无后继）| `ha_transitions.pt` |
| VLA 与标签 token MSE | 0.0047（H=20 时 0.0045–0.0063）| `eval_curve.jsonl` |
| VLA vs GT 想象回报（H=5）| +0.1701 vs +0.1705（−0.2%）| 同上 |
| VLA vs GT 想象回报（H=20, e60）| +0.3994 vs +0.3545（**+12.7%**）| `s3v8_e60` |
| critic v_ev（grounded）| +0.65 ~ +0.81 | 训练日志 |
| critic v_ev（imagined，旧）| −2.0 ~ −5.1 | 旧日志 |
| WM qd_corr（C6h，ha_cmd）| 0.752 / ratio 1.115 | `eval_wm_velocity` |
| WM qd_corr（刷新后，同仪器）| 0.589 / ratio 1.700 | 刷新实验 |
| WBC norm_return（A@20000，ha_cmd）| 0.8673，0 摔 | `S1_ctrl/A_20000_cmd_eval.json` |
| oracle gain_rel | +19.8% / +20.7%（6 步优化、漂移 0.112）| `runs/oracle_tokens/*.json` |
| LoRA r=32 | 33.55M / 1.654B = 2.03% | `[AUDIT][lora]` |
| Isaac 物理吞吐 | 15,281 env-transitions/s（256 物体）；单环境 51–60 步/s | bench |
| Isaac 启动 | 62–106 s | bench |
| SuperMimic 规模 | 4096 envs, H=32, batch 4096, UTD 0.4 | 源码 |

### 5.2 代码地图

| 路径 | 作用 |
|---|---|
| `scripts/train_joint_v2.py` | 训练入口（所有 CLI 开关 + 口径审计行）|
| `diff_vla_wbc/train_v2.py` | `JointTrainerV2`：冻结执行器、FM-BC、rollout、`fixed_eval`（含 eval_horizon 与配对统计）|
| `diff_vla_wbc/v3_step.py` | `JointTrainerV3` + `V3Config`：token BC 关、手部掩码、grounded critic、动作信任域、critic 自举 |
| `diff_vla_wbc/lora.py` | LoRA（按完整模块路径匹配 216 层；`B` 零初始化）|
| `diff_vla_wbc/burnin.py` | burn-in、DR 扰动、帧→行对齐 fail-fast |
| `diff_vla_wbc/rollout.py` | 可微 rollout（`imagine_with_tokens`）、`discounted_return` |
| `diff_vla_wbc/token_critic.py` | critic + GAE + `critic_fit_supervised`（真数据监督）|
| `diff_vla_wbc/wm_refresh.py` | WM 保鲜（当前关闭）+ `eval_wm_real` |
| `diff_vla_wbc/isaac_joint.py` | Isaac bundle（WBC+WM+奖励+FK+残差项）|
| `scripts/build_real_returns.py` | 真录制 `return-to-go`（critic 的 grounded 目标）|
| `scripts/build_real_transitions.py` | 真转移 `(s,a,s')`（当前只用于诊断）|
| `scripts/eval_wm_velocity.py` | **WM 速度通道仪器**（ha_cmd/label/rollout）|
| `scripts/finetune_wm_velocity.py` | 早期 WM 速度标定（**数据是参考库，`a_t=dq_t`，口径不同**）|
| `scripts/dump_arm_tokens.py` | 三臂 token 流落盘（gt / oracle / vla）|
| `scripts/isaac_g1_collect.py` | **回放式 collect**（最小 G1 环，自包含）|
| `scripts/isaac_replay_2arm.sh` | 两臂回放 runner（未通，被渲染/启动问题阻塞）|
| `patches/sm_token_npy.md` | provider 的 `npy` token 源补丁留档（third_party 被 gitignore）|
| `docs/S3_REAL_JUDGE_PLAN_20261027.md` | 真物理裁判改造方案（含判据与工时）|
| `scripts/s3_audit_config.py` | **39 项启动前口径审计**（fail-closed）|

### 5.3 已知的"表面现象 vs 真因"陷阱（写给评审）

1. `tok_mse` 变大 ≠ 变差：标签不是最优解；但**约束下它仍是有用的有效性代理**。
2. `ret_vla > ret_gt` 通常是 hack 信号而不是改进（因为 GT 在物理上已接近最优）。
3. WM 的 `qd_corr 0.906` 与 `0.752` **不是同一件事**（前者是拟合参考库的自洽性，后者是预测真机）。
4. `λ_r=0` 的对照臂**也会漂**——漂移不一定来自奖励（手部串扰曾造成 ×1.39）。
5. 训练日志里的 `loss/reward_norm` 是 **−(ret+boot)**，符号容易读反。
6. `imag_return` 只反映"WM 认为的好坏"；**唯一可信判定是真物理**。
