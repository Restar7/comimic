# v5 当前问题、架构与新方案（给 GPT 的完整上下文，2026-11-08）

> 目的：把**当前训练/部署架构、已测数据、闭环失败现象、以及我们想改的东西（H=16+critic、单关节越限惩罚、LoRA）**一次讲清，
> 请据此判断"改哪些、怎么改、先改哪个"。
> 本文所有数字都是本机实测，不是估计；路径可直接查。

---

## 0. TL;DR（先看这 8 条）

1. **v5 的 Δtoken 方案在冻结 WM 上已经接近饱和**：val `ΔJ=+0.0288`，oracle 上界 `+0.0298`（96.5%）。
2. **但闭环（真 Isaac + 真 VLA + 真 WBC）6/6 全失败**：OpenDoor / SitSofa / VisionNavi × seed0/1，各 3000 步、`timeout`、`reward=0`。
   - 这些任务的 reward 是 **sparse**（成功才给 1.0），所以 reward=0 = 一次都没成功；**没有摔倒**。
   - **没有跑 Base 对照**（用户要求只跑微调后），所以现在**无法归因**是 Δ 有害还是任务本身难。
   - 每步控制频率只有 **7.2–7.5 Hz**（VLA 推理延迟），但物理 dt 固定 0.02 s，时间口径自洽，只是评测慢。
3. **实测到严重问题：训练后的 Δ 会打出单关节 5.89 rad 的动作目标**（left_knee；G1 膝关节限位约 [-0.09, 2.87] rad），
   逐控制步的单关节 max：`0.18 → 4.45(t=5) → 3.40(t=10) → 5.89(t=14) → 4.36(t=19)` rad。
   单步 RMS 也从 32 mrad 涨到 214 mrad。**这几乎必然导致闭环抖动/撞限位/不完成任务。**
4. **H=20 是重要嫌疑**：H=20 让模型有 20 帧去积分，越晚的帧幅度越大；而 **WM 奖励对"单关节越限"没有任何惩罚**。
5. **H 扫描（96 held-out 窗口）**：奖励质量几乎均匀铺在 20 步（前 8 步 42%、前 12 步 62%、前 16 步 81%）；
   oracle 增益：H=5/8/10/12/14/16/20 = 13%/25%/37%/52%/66%/77%/100%。
   ⇒ **不要 5/8/10；建议 H=16 + critic**（尾部只占 19%）。
6. **想改的两件事**（用户已确认方向）：
   - **M1：H=16 + critic**（K=20 部署不变，尾部 `γ^16·V(y_16)`；部署侧只施加 `t<16` 的 Δ）；
   - **M2：给 WM 奖励加"单关节越限"惩罚**（训练侧 hinge，部署侧硬限幅兜底）。
7. **LoRA 方案**（去掉 Δtoken，直接训 action head 的 LoRA，梯度穿过 4 步 flow-matching 回传）**可行**，
   现有代码有 S3 A 路径（r=32、33.5M、bf16、batch 32×2、H=20）；但它**在 WM 指标上没有提升空间**（上限就是 +0.0298），
   其价值只在"改表示/换映射"，而当前闭环证据（6/6 失败）说明**瓶颈可能不在参数化**。
8. **需要 GPT 帮我们拍板**：M1/M2 的具体公式与权重、H 是否 16、LoRA 现在做还是先修闭环、以及"单关节惩罚"该加在哪一层。

---

## 1. 系统全貌与文件地图

| 组件 | 路径（worktree = `comimic-wt03`） |
|---|---|
| Δtoken Actor（v5） | `diff_vla_wbc/delta_token_temporal_v5.py` |
| 部署契约（per-frame、hist10、关节序） | `diff_vla_wbc/delta_token_io_v5.py` |
| Critic（TDλ+EMA+Gate，默认关） | `diff_vla_wbc/imagined_value_v5.py` |
| 动作 trust / 平滑 / 锚 | `diff_vla_wbc/action_trust_v5.py` |
| 主训练器 | `scripts/train_temporal_ac_bptt_v5.py` |
| 启动器 | `scripts/launch_v5_main_v2.sh`、`scripts/launch_v5_pilot.sh` |
| H 扫描 | `scripts/diag_horizon_v5.py` |
| 动作预算/关节幅度 | `scripts/diag_action_budget_v5.py` |
| oracle 上界 | `scripts/diag_oracle_upper_v5.py` |
| 权重标定 | `scripts/calibrate_v5_weights.py` |
| 部署自检 | `scripts/verify_v5_deploy_contract.py` |
| 闭环评测（自动） | `scripts/run_closedloop_v5_eval.sh` |
| VLA server（v5 分支） | `scripts/ha_vla_latent64_server.py`（wt03 副本；main 里也有改动） |
| HA client（hist10 字段） | `third_party/HumanoidArena/isaaclab_twist2_g1/action_provider/{action_provider_supermimic.py,lerobot_vla_http_client.py}` |
| 训练产物 | `runs/temporal_ac_bptt_v5/main_v2/seed{0,1}/` |
| 闭环产物 | `runs/closedloop_v5/` |

---

## 2. 当前架构 A：Δtoken（已训成、已部署验证）

### 2.1 结构（1,744,832 参数）

```
输入:
  Z      (B,40,64)   冻结 VLA 的整段 chunk token
  state64(B,64)      [rotlocal6(6) | q(29) | qd(29)]，**训练=standard 序**
  hist10 (B,10,58)   过去 10 tick [q−g1_default(29) | qd(29)]，**standard 序**，最老→最新

计算:
  hz   = token_proj(Z) + PE_chunk[:40]                  # 64→128
  c_t  = LayerNorm(TransformerEncoder(1 block, d128, 4 head, FFN256, pre-LN))(hz)
  h    = GRU(58→128)(hist10).last                        # 广播到每帧
  x_t  = [z_t(64) | c_t(128) | h(128) | state64(64) | PE_pos(t)(16)]   # d_in=400
  u_t  = MLP(400→1024→1024→64, SiLU)                     # 末层 N(0,4e-4)、bias=0
  Δ_t  = apply_norm_cap(u_t, cap=0.08) · 1[t<20]         # **逐帧**、t≥K 恒 0
```

参数分解：`token_proj 8,320 + PE_chunk 5,120 + Transformer 132,480 + LayerNorm 256
+ GRU 72,192 + PE_pos 640 + MLP 1,525,824 = 1,744,832`。

### 2.2 训练管线

```
采样（防泄漏）：w = 漂移合格 ∩ f≥10 ∩ f+H≤clip_end ∩ **clip ∈ 评测集 train_clips**
              → 235,280 帧 / 375 clips（原 338,402，含 30.5% val/test 泄漏，已修）
burn-in k=10（5cm/0.1rad/0.05rad/0.3rad/s 扰动）→ 起点
现场 VLA（fm_steps=4, 每次新初噪）→ Z
损失: L = −mean(J20) + 0.010·L_act + 0.002·L_smooth + 0.001·L_anchor + 1e-3·L_raw
  J20   = Σ_{k<20} γ^k·survival_k·reward_k           γ=0.99
  L_act = mean relu(RMS_j(e_same)/τ − 1)²            τ=0.020 rad，**同观测**动作差
  L_sm  = Σ(Δe_same²/29·M)/ΣM/τ²
  L_anc = Σ(Δ²·mask)/(K·B)/cap²
  L_raw = Σ_{t<K} relu(‖u_t‖/(2·cap) − 1)²/(K·B)
优化: AdamW head 3e-4 / enc 1e-4，wd 1e-4，clip 1.0，50 warmup + cosine
```

**同观测动作 trust**（关键修复）：rollout 期间拦 `executor.act(token, obs93)` 记录每步 obs93，
结束后用**同一 obs93 + 基础 token** 重解 action 作参照：
`e_same = q_target(a^Δ_k) − q_target(exec_act(Z_k, obs93_k))`。
旧实现是两条 rollout 的动作差（混入轨迹发散），实测其梯度是 J 的 27×，是 v1 坍缩的主因。

### 2.3 门禁 / 评测 / 部署

- **条件化门禁**（每 25 步）：`cos(shuffle Z)<0.9`、`std_win>1e-4`、`std_t>1e-4` → `cond_check.jsonl`；
- **评测**（每 50 步）：96/96 held-out 窗口，`ΔJ = mean J(Δ) − mean J(0)`；同时报
  **同观测**（`same_obs_rms/peak/p99`）与**轨迹级**（`rms/peak/p99`）两套动作预算；
- **部署契约**：`T/K/H=40/20/20`、`t≥K 恒 0`、`hist10` 必须存在、`state64` 是 `ha_v31` →
  契约层 `reorder(ha_v31→standard)`；**两个 default**（hist 用 `g1_default_joint_pos()`，
  动作用 `bundle.default_joint_pos`）；
- **运行期一致性门禁**（新增）：`hist10[-1]` 与 `state64` 的 q/qd 必须对得上
  （`err_q/err_qd < 5e-3`），否则抛 `DeltaTokenContractError`。
  部署链路对拍：**max|Δ|=1.5e-8**（部署序 vs 训练序直连）。

---

## 3. 已有结果（实测）

### 3.1 v2 主训练（on-the-fly，2 seeds，5×4090 中的两张）

| it | seed0 ΔJ | seed1 ΔJ |
|---|---|---|
| 50 | +0.01728 | +0.01610 |
| 100 | +0.01859 | +0.01948 |
| 200 | +0.02231 | +0.02491 |
| 300 | +0.02871 | +0.02691 |
| 400 | +0.02847 | +0.02583 |
| 450 | **+0.02877（best）** | +0.02370 |
| 500 | +0.02802 | **+0.02757（best）** |

- base J（val）= 1.50594；oracle = +0.0298 ⇒ best ≈ **96.5% / 92.5%**；
- cond PASS（`cosZ=0.62`、`std_win=4.0e-3`）；‖Δ‖=0.0785；同观测 actRMS≈30 mrad；
- 训练在 it≈530 被**外部 SIGKILL**（日志无 traceback，非本项目脚本），best ckpt 保留在
  `runs/temporal_ac_bptt_v5/main_v2/seed0/checkpoint_actor_best.pt`。

### 3.2 Pilot（cached_pool，200 步，同数据同预算）

| 臂 | trust | raw | test ΔJ | cond cosZ | ‖u‖/cap |
|---|---|---|---|---|---|
| p0_old | 两轨迹 | ✗ | +0.00099 | 0.9999 | 11.5 |
| p1_trust | 同观测 | ✗ | **+0.02671** | 0.60 | 2.24 |
| p2_cap | 两轨迹 | ✓ | +0.00224 | 0.978 | 3.08 |
| p3_both | 同观测 | ✓ | **+0.02640** | 0.68 | 1.80 |
| p4_postpl | 常量模板 | — | +0.00970 | — | — |

⇒ 同观测 trust 是**决定性修复**；raw hinge 单独不够，但把 u 从 2.2c 按到 1.8c。

### 3.3 H 扫描（96 held-out 窗口，cap=0.08）

| H | oracle ΔJ | 占 H=20 | 奖励质量累计 | 尾部（critic 兜） |
|---|---|---|---|---|
| 5 | +0.00392 | 13.2% | 26.6% | 73.4% |
| 8 | +0.00751 | 25.2% | 42.0% | 58.0% |
| 10 | +0.01115 | 37.4% | 51.9% | 48.1% |
| 12 | +0.01549 | 52.0% | 61.8% | 38.2% |
| 14 | +0.01952 | 65.5% | 71.6% | 28.4% |
| 16 | +0.02289 | **76.9%** | 81.3% | **18.7%** |
| 20 | +0.02978 | 100% | 100% | 0% |

（奖励质量几乎均匀铺满 20 步，短 H 真的丢信号。）

### 3.4 权重标定（cached，B=64，`‖g_term(λ=1)‖/‖g_J‖`）

| ‖u‖/cap | ‖g_J‖ | act（同观测） | smooth | anchor | raw |
|---|---|---|---|---|---|
| 0.09 | 0.987 | 0× | 13.2× | 5.3× | 0× |
| 0.54 | 0.671 | 0× | 14.6× | 21.1× | 99× |
| 1.61 | 0.351 | 2.19× | 23.1× | 16.4× | 2781× |
| 3.57 | 0.211 | 6.43× | 28.8× | 1.03× | 19527× |

⇒ 锁定 λ 有效：act 0.02–0.06×、smooth 2.6–5.8%、anchor 0.1–2%、raw 在 2c 处 ≈2.8×。

---

## 4. 当前问题（按严重度）

### P1 ★闭环 6/6 失败（最高优先级）

| task | seed | steps | success | reward | reason |
|---|---|---|---|---|---|
| OpenDoor | 0/1 | 3000 | ❌ | 0.0 | timeout |
| SitSofa | 0/1 | 3000 | ❌ | 0.0 | timeout |
| VisionNavi | 0/1 | 3000 | ❌ | 0.0 | timeout |

- 任务奖励是 **sparse**：`compute_reward_open_door` = `1.0 if success else 0`；
  ⇒ reward=0 = **一次都没成功**，不是"没记录"；
- `failure_reason=timeout`、`episode_steps=3000` ⇒ **没摔倒**，是活着但没完成任务；
- **没有 Base 对照**（用户要求只跑微调后）⇒ 现在无法归因：
  (a) Δ 有害；(b) 任务本身对 base VLA 也很难；(c) 评测口径（7 Hz / exec 20 / 相机）问题；
- 控制频率 **7.2–7.5 Hz**（VLA 推理延迟），物理 dt 固定 0.02 s，时间口径自洽，但评测慢
  （3000 步 ≈ 7 min/run）；
- **没带视频**（我设了 `RECORD_VIDEO=0`），需要重跑带视频。

### P2 ★单关节动作越限（实测，很可能是 P1 的直接原因）

用 **当前 best ckpt**（it=450）在 val 96 窗口上测（`diag_action_budget_v5.py`）：

| 步 t | 单步 RMS (mrad) | **单关节 max (rad)** |
|---|---|---|
| 0 | 32 | 0.18 |
| 5 | 96 | **4.45** |
| 10 | 143 | 3.40 |
| 14 | 182 | **5.89（left_knee）** |
| 19 | 214 | 4.36 |

- **5.89 rad（338°）**，G1 膝关节限位约 `[-0.09, 2.87]` rad ⇒ **超限 2 倍**；
- 逐关节 peak：left_knee 5.89、right_hip_yaw 2.23、right_hip_roll 2.09 …
- 对比 it=50 时（peak 2.5 rad）：**训练越久越糟**；
- 这些是**动作目标**（decoder 输出 → `q_target = default + scale·a`），不是 WM 状态；
- 训练里的 `L_act` 是 **29 关节 RMS**（30 mrad），**完全掩盖单关节尖峰**。

### P3 ★WM 奖励对"单关节越限"没有惩罚

- 奖励是 tracking 的 Huber/exp 形式（`huber_delta=0.02, huber_cap=0.15` 等），对**单关节大幅偏移**
  只给饱和惩罚，梯度趋 0；
- decoder 对 token 的某些方向**极敏感**（小 token 变化 → 大关节动作），所以优化器可以"低成本"刷奖励；
- 现有正则项（L_act/L_sm/L_anc/L_raw）都**不直接约束单关节动作**。

### P4 ★H=20 的晚期幅度积分

- H=20 ⇒ 20 帧都能贡献梯度，越晚的帧越容易被放大；
- 实测单关节 max 从 t=0 的 0.18 涨到 t=14 的 5.89；
- H=16 可以把"最晚 4 帧"移出优化范围，但**部署侧仍会执行 t<20** ⇒ 需要部署侧只施加 `t<16`
  （或把 t≥16 的 Δ 置零）。

### P5 部署口径（已修，但需保持）

- 训练 `standard` vs 部署 `ha_v31`：契约层换序，已对拍 1.5e-8；
- **两个 default**：hist 用 `g1_default_joint_pos()`（−0.312/0.669/−0.363），
  动作用 `bundle.default_joint_pos`（SONIC −0.2/0.42/−0.23）；差 ≈0.27 rad；
- HA client 的 obs 历史是 **standard 序**，且它的 default 是 **HA/trackerlab**（−0.2/0.42/−0.23/0.87…），
  与训练 hist 的 default 不同 ⇒ 我们改成 **client 发绝对 q + server 减训练 default**；
- 新增运行期门禁 `hist10[-1]` vs `state64`（err<5e-3），顺序声明错会直接抛。

### P6 LoRA 方案的风险与历史

- **方案**：冻 VLM+base DiT，LoRA r=32（attn1/attn2/ff + proj_out，33.5M），
  可微采样 fm_steps=4 → WM → J20 → 反传；S3 A 已有路径（bf16、batch 32×2）。
- **历史**：`v13 = action_target + LoRA` 的**幅值排序指标只有 0.09**（用户口径；不是残差/均值），
  且当时闭环成功率 **30.8% vs base 58.3%**（LoRA 是负资产）。
  用户认为**当前 v5 的结构/参数设计已经解决了 v13 的问题**（本轮的失败模式是 trust/泄漏/cap 饱和，已修）。
- **风险**：33.5M（19×）且**无界**（没有 cap），同 WM 奖励下更容易 exploitation；
  需要 token 变化 cap / KL / 同观测 trust / 门禁。
- **成本**：4 步 FM 反传 + 1.09B DiT，每步样本数比 Δtoken 少一个量级（B=32 vs 256）。

---

## 5. 我们想改什么（候选方案）

### M1 ★H=16 + critic（用户已确认方向）

- **训练**：`H_actor=16`，`K=20` 不变；
  `L = −[J16 + γ^16·V(y_16)] + 正则`；`y_k = [y_robot(71) | y_task]`；
- **critic**：`imagined_value_v5.py`（TDλ+EMA+Gate）；`K>H` ⇒ 可识别；
  门禁：`v_ev≥0.5`、方向一致性≥0.6、`v_adv` 与真实 return 的 cos；
- **部署**：**只施加 t<16 的 Δ**（`--delta-v5-max-t 16`），t≥16 保持 base token；
- **为什么 16**：H 扫描（§3.3）——16 拿 77% oracle、尾部只有 19%，且排除 t=18–19 的 2.2–5.9 rad 爆炸点；
- **注意**：H=16 后，t≥16 的 Δ 在训练里没有奖励约束，**必须在损失里 anchor 到 0**（或部署置零），
  否则会变成"没人管"的自由变量。

### M2 ★单关节越限惩罚（用户已确认方向）

三种候选（可叠加）：

```
(1) 关节限位 hinge（推荐，物理口径最硬）
    q_tgt_t = default + scale·a_t                         # (B,H,29)
    L_lim = mean_{t<H,j} relu(|q_tgt_tj| − q_max_j)²      # q_max 用 soft_joint_limits(0.95)

(2) 单关节幅度 hinge（不依赖限位表）
    L_pk = mean_{t<H} relu( max_j |q_tgt_tj| − q_pk )²    # q_pk 取 base 策略的 99 分位 + 余量

(3) 直接对动作归一化值
    L_a = mean relu(|a_t| − a_max)²                       # a_max 取语料 99.9 分位
```

- 权重标定：沿用 `calibrate_v5_weights.py` 的"每项梯度 ≈ ρ·‖g_J‖"（在饱和 ckpt 上测，因为那里才有越限）；
- 目标：把 P2 的 5.89 rad 压到 ≤ 限位（膝 ≈2.87）；
- **部署侧兜底**：`q_target = clamp(q_target, q_min·0.98, q_max·0.98)` + 单步变化率限幅
  （这是安全层，不替代训练惩罚）。

### M3 部署侧只施加 t<H（与 M1 配套）

- server 增加 `--delta-v5-max-t`，`apply_delta_per_frame` 只写前 `max_t` 帧；
- 好处：即使训练没收敛好，也把"晚期帧"的爆炸隔离掉；
- 代价：后 4 帧回到 base VLA（我们本来也没优化它们）。

### M4 LoRA 方案（可选，作为容量/表示消融）

```
冻：VLM backbone + base DiT
训：LoRA r=32（attn1/attn2/ff + proj_out，33.5M）
前向：VLM 特征 no_grad 缓存 → DiT(4 步 FM，可微) → tokens
损失：−[J16 + γ^16 V(y_16)] + L_act(同观测) + L_smooth
      + λ_tok·‖Z'−Z_base‖² + λ_raw·hinge(‖Z'−Z_base‖/(2c_tok))
      + L_lim（M2，同样加）
门禁：cond、幅值排序、动作预算、闭环成功率
```

- 成本：S3 A 用 batch 32×2、bf16、H=20；需要先 benchmark 步时/显存；
- 风险：无界容量 + WM 奖励 ⇒ 先加 token cap/KL，再谈效果；
- 判据：**闭环成功率**优先，WM ΔJ 只作参考（因为 Δtoken 已 96.5% oracle，没有再往上刷的意义）。

### M5 闭环评测修正（不做这些，M1/M2 的效果也说不清）

1. **补 Base 对照**（同 3 任务、同 3000 步、同 server 口径、同 seed）；
2. **带视频**（`RECORD_VIDEO=1`），存 `videos_failure/success`；
3. **确认任务可赢**：先确认 base VLA 在这些任务上不是 0（已知 DoubleDesk 是 +0.06 NR 量级）；
4. **控制频率**：7.2–7.5 Hz 是 VLA 推理延迟；确认 replan 口径（`HA_VLA_EXEC_STEPS=20`）与
   训练 H=20 一致；
5. **reward 口径**：这些任务 sparse（成功才 1.0），所以"reward=0"不能当"没进步"，
   需要辅助指标（门角度、到目标距离、存活步数）；
6. **Δ 的施加范围**：M3 的 `t<H`。

### M6 其他可选

- **token 变化 cap**（LoRA 用）：`‖Z'−Z_base‖ ≤ c_tok`，c_tok 用 base token 的 p99；
- **KL 到 base**：`KL(π_LoRA ‖ π_base)`（采样分布层面）或对 token 的 `‖·‖²`；
- **动作变化率惩罚**：`Σ‖a_t−a_{t−1}‖²`（现有 L_smooth 是 token 差，不是动作差）；
- **critic 的 real-return 对照**：S3 用 `--critic-source real`（`runs/wm_real/real_returns.pt`），
  可作为 critic 偏差的校准。

---

## 6. 需要 GPT 回答的问题

1. **H 取 16 是否合适**？还是 14（更保守）？critic 的尾部 19% 是否会引入过大偏差？
2. **M2 的单关节惩罚该选哪个公式**（限位 hinge / 峰值 hinge / 动作归一化）？权重怎么定？
   是加在 `q_target` 上还是加在 `a_norm` 上更稳？
3. **`L_act`（29 关节 RMS）是否应该改成"最大关节"口径**？RMS 30 mrad 掩盖了 5.89 rad 的尖峰。
4. **部署侧是否应该硬限幅**（clamp 到限位 + 变化率）？会不会掩盖训练问题？
5. **H=16 后 t≥16 的 Δ 怎么处理**：训练 anchor 到 0 + 部署置零，还是让它们自由？
6. **LoRA 现在做还是先修闭环**？如果做，容量/无界风险怎么控（token cap / KL / 先小规模）？
7. **闭环 6/6 失败**：在没有 Base 对照的情况下，最省时的归因实验是什么？
8. **WM 奖励本身要不要改**（例如对单关节越限直接给负奖励）？还是只在 loss 里加正则？

---

## 7. 复现命令（关键）

```bash
# 训练（v2 口径；H=16 版本待实现）
bash scripts/launch_v5_main_v2.sh

# H 扫描（§3.3 的数据）
python3 scripts/diag_horizon_v5.py --eval-set runs/.../eval_set --split val \
    --batch 24 --h-list 5,8,10,12,14,15,16,20 --out .../horizon_diag_fine.json

# 单关节幅度（§P2 的数据）
python3 scripts/diag_action_budget_v5.py \
    --ckpt runs/.../main_v2/seed0/checkpoint_actor_best.pt \
    --eval-set runs/.../eval_set --split val --out .../action_budget_v2.json

# 部署自检
python3 scripts/verify_v5_deploy_contract.py --ckpt <ckpt>

# 闭环（自动，6 run × 3000 步，Δ only）
bash scripts/run_closedloop_v5_eval.sh

# 单测
python3 -m pytest tests/test_temporal_v5_grad.py tests/test_v5_fail_closed.py \
    tests/test_physical_action_trust.py tests/test_history_and_token_index.py \
    tests/test_delta_token_io.py tests/test_deploy_audit.py -q
```

---

## 8. 附：当前闭环产物与日志

```
runs/closedloop_v5/
  eval_suite.log
  server_delta_18090.log
  tasks_delta/<Task>__seed{0,1}__.../result.json
  closedloop_summary.json
```

`result.json` 字段：`success / failure_reason / episode_steps / final_reward / max_reward / duration_sec`；
**没有视频**（`RECORD_VIDEO=0`），需要重跑带视频。

---

## 9. 一句话总结

- **WM 指标已经到 96.5% oracle**，继续刷它意义不大；
- **闭环 6/6 失败** + **单关节 5.89 rad 越限** 是当前真正的两个问题；
- **H=16 + critic** 和 **单关节越限惩罚** 是对症的两个修改（前者减少晚期积分，后者直接压尖峰）；
- **LoRA** 是"表示能力"的赌注，不是当前瓶颈的解；如果要做，必须在修好闭环归因之后、并带 token cap/KL。

---

## 10. P0 核实结果（2026-11-08 03:0x，写完上文后立刻补测）

### 10.1 "5.89 rad" 是**单位口径错**，不是物理关节目标

- 训练 bundle（`SonicIsaacBundle`）的 `action_scale` 实测是 **1.0（全关节）**；
- 而物理换算（SONIC/IsaacLab 口径）应是 `G1_ACTION_SCALE`
  = `[0.351,0.351,0.548,0.351,0.439,0.439, ...]`（0.075–0.548）；
- 诊断脚本 `q_target_from_action(default, bundle.action_scale, a)` 用的是 **1.0** ⇒ 把动作放大了
  **1/0.35 ≈ 2.9 倍**（部分关节 13 倍）。

用**正确 scale** 重算（同一个 best ckpt、同一批 24 个 val 窗口）：

| 量 | 值 |
|---|---|
| 原始动作（`SonicExecutor` 输出） | min **−15.79** / max **15.99** / std **2.09** |
| 物理 `q_target(Δ)`（`default + G1_ACTION_SCALE·a`） | min −2.05 / max **4.53 rad** |
| 物理 `q_target(Base)` | min −2.05 / max **4.47 rad** |
| 物理 `|q_Δ − q_Base|` 逐帧 max | 0.08(t0) … 0.37(t5) … **0.90(t10)** … **2.07(t14)** … 1.83(t16) … 1.60(t18) |

⇒ **结论 1**：5.89 rad 是"scale=1.0"口径下的假数；物理口径下 **Δ 的最大贡献是 2.07 rad @ t=14**。
⇒ **结论 2**：**Base 自己就到了 4.47 rad**（t=0），说明"目标越限"主要不是 Δ 引入的，而是
**训练侧 decoder 的动作输出本来就大**（raw action ±16、std 2.09，远超归一化动作的合理范围）。
⇒ **结论 3**：H=16 去不掉 t=14（在 H=16 之内）——审阅意见正确，H=16 只能作为对照/消融。

### 10.2 这暴露了一个更根本的口径问题：训练 decoder ≠ 部署 WBC

- **训练**：`bundle.act` = **SONIC decoder**（`SonicExecutor`），输出 ±16 的 raw action；
- **部署**：`action_provider_supermimic.py` 用的是 **trackerlab WBC（A24000）**，
  输出归一化动作，再 `q_target = default + G1_ACTION_SCALE·a`；
- 两者是**不同的 token→action 映射**，所以：
  1. 训练里的"同观测动作 trust"（30 mrad）**不是**部署侧的物理动作预算；
  2. 训练里测到的"单关节 2.07 rad"也**不能直接**归因到部署侧；
  3. 闭环失败的原因可能在这个映射差异上，而不是 Δ 本身。

### 10.3 据此修正 P0 的执行项

1. **必须在部署侧 instrumentation**（provider 每步落盘）：
   `q_raw`（WBC 原始目标）、`q_applied`（安全裁剪后）、`q_real`（实际关节）、
   `q_base_raw`（同 obs 用 base token 重解）、关节上下限、控制时刻、以及任务进展
   （门角度 / 到目标距离 / 接触状态 / 跟踪误差）；
2. **先跑 Base 已知有进展的场景**（DoubleDesk，历史 +0.06 NR 量级）做短程配对（Base / v5 / v5+安全层 / v5+只前16帧），
   带视频、同 VLA flow seed、同起点；
3. `L_lim` 要按"**绝对物理 q_target**"算、按**不对称上下限**（`l_j/u_j` + 归一化尺度 `s_j`）、
   用审阅给的 max 形式（`1/2·max² + 1/(2K)Σ_t max_j²`）；
4. WM 前向用**安全裁剪后的动作**推进，`L_lim` 用裁剪前的 `q_raw` 算（训练/部署同口径）；
5. H=16 + 4 步 tail critic 降级为 **P3 消融**（方案 B/C），先做方案 A（H20 + 安全约束）。

> 在部署侧 instrumentation 出来之前，**不要**再引用"5.89 rad 导致闭环失败"这类因果结论。

---

## 11. 修正记录与"当前真实问题"（2026-11-08 03:3x）

> 本节是对前文的**勘误 + 重新定级**。用户明确：**部署永远用 SONIC decoder，不用 Supermimic，订死**。

### 11.1 修正 1：6/6 闭环失败是**无效实验**（跑错执行器）

- 第一次用 `SONIC=1` 跑 → `action_provider_sonic.py` 还没发 `state_hist_10`，server 拒绝请求；
- 为了跑通，改成了 **`SONIC=0`（Supermimic provider）** —— 所以那 6 个 run 是 **Supermimic 执行器**的结果；
- **部署口径是 SONIC decoder（锁死）** ⇒ **"闭环 6/6 失败"不能作为 SONIC 部署的结论，已撤回**；
- 同时撤回"训练 decoder ≠ 部署 WBC"的说法：**训练与部署都是同一个 SONIC decoder**。

**已修**：
| 改动 | 验证 |
|---|---|
| `action_provider_sonic.py` 发 `state_hist_10`（10 帧 `[q_abs(29) | qd(29)]`，**HA 序**） | 取自 `_robot_joint_pos_hist/_robot_joint_vel_hist` |
| server 在 `hist_joint_order=ha_v31` 时把训练 default（`g1_default`）换到 HA 序再减 | 代码已改 |
| 端到端对拍（SONIC-provider 路径 vs 训练直连） | **max\|diff\| = 1.49e-8** |

⇒ 现在正在跑 **P0：DoubleDesk，SONIC=1，Base vs Δ，各 1000 步，带视频**。

### 11.2 修正 2："5.89 rad" 是 scale 口径错

- 训练 bundle 的 `action_scale = 1.0`（API 兼容值），但物理换算是
  `target = default + G1_ACTION_SCALE · a`（`G1_ACTION_SCALE` = 0.075–0.548）；
  SONIC 部署侧（`action_provider_sonic.py:2841`）用的就是这一套；
- 用正确 scale 重算（同一 best ckpt、24 个 val 窗口）：

| 量 | 值 |
|---|---|
| raw decoder 动作 | min −15.79 / max **15.99** / std **2.09** |
| 物理 `q_target(Δ)` | min −2.05 / max **4.53 rad** |
| 物理 `q_target(Base)` | min −2.05 / max **4.47 rad** |
| 物理 `\|q_Δ − q_Base\|` 逐帧 max | 0.37(t5) … **0.90(t10)** … **2.07(t14)** … 1.83(t16) … 1.60(t18) |

⇒ **物理上 Δ 的最大贡献是 2.07 rad @ t=14，不是 5.89**；但 **Base 自己也到 4.47 rad** ——
这需要**部署侧每步落盘**才能判断是 WM 漂移、decoder 对 VLA token 的输出范围，还是真实风险。

### 11.3 之前列的问题，哪些还在、哪些已撤销

| 之前的问题 | 现状 |
|---|---|
| Δ 学成常数（shuffle cos 0.9999） | ✅ **已解决**：v2 cos≈0.62、val +0.0288 |
| 训练采样泄漏 val/test（30.5%） | ✅ 已修：235,280 帧 / 375 train clips |
| 两轨迹动作 trust 导致坍缩 | ✅ 已修：同观测 trust（pilot 27×） |
| cap 饱和（‖u‖=82×cap） | ✅ 已修：raw hinge，‖u‖≈1.8c |
| **5.89 rad 单关节越限** | ⚠️ **部分撤销**：物理口径 **2.07 rad @ t=14**（仍偏大，需约束），不是 5.89 |
| **闭环 6/6 失败** | ❌ **撤销**：跑的是 Supermimic，非 SONIC；P0 正在重跑 |
| 训练 decoder ≠ 部署 decoder | ❌ **撤销**：都是 SONIC |
| H=16 能消除晚期爆炸 | ❌ **不成立**：t=14 峰值在 H=16 之内；H=16 只能当消融（方案 B/C） |
| "LoRA 没有提升空间" | ❌ **不严谨**：oracle 只是"冻结 WM + 残差预算"下的局部参考，不是 DiT LoRA 的数学上界 |
| WM 指标与真机脱节 | ⚠️ **待定**：等 SONIC 的 P0 Base/Δ 对照 |
| 控制频率 7.2–7.5 Hz | ⚠️ **待核实**：需确认 `env.step()` 是否严格同步、有无过期观测 |

### 11.4 当前真实问题（更新后，按优先级）

1. **P0-A 部署侧缺每步物理量**：`q_raw / q_applied / q_real / 关节上下限 / 控制时刻 / 任务进展`
   没有落盘 ⇒ 谁也无法判断"越限"是真实发生还是离线推算。
2. **P0-B SONIC 的 Base/Δ 对照未出**（正在跑）：DoubleDesk 1k 步 × {Base, Δ}，带视频。
3. **P1 Δ 的逐帧物理贡献随 t 增长**（0.37@t5 → 0.90@t10 → 2.07@t14 → 1.83@t16）：
   需要 `L_lim`（不对称限位 + 归一化 + max 形式）+ 部署安全层（限位 + 变化率），
   且 **WM 用裁剪后动作推进、L_lim 用裁剪前 `q_raw`**。
4. **P1-B Base 的绝对目标 4.47 rad** 需确认来源（WM 漂移 / decoder 输出范围 / 口径）。
5. **P1-C 动作 trust 的 τ=20mrad 是"raw action 差"口径**：`q_target_from_action` 用了 scale=1.0，
   物理差 ≈ 0.35–0.55×，所以 τ 实际更严/更松取决于关节；**需按 `G1_ACTION_SCALE` 重标 τ**。
6. **P2** H20 vs 前16帧 mask（同 ckpt，不重训）；**P3** H16 + 4 步 tail critic（先比 ∇V 与真实 4 步回报梯度）；
   **P4** LoRA（等 P0–P2 通过）。

### 11.5 现在在跑的 P0（SONIC=1）

```
runs/closedloop_v5/p0_sonic/
  server_18090.log   # v5 Δ（hist_order=ha_v31, hist_absolute=1, delta-v5-ckpt=best）
  server_18091.log   # Base（无 Δ）
  run_base.log  run_delta.log   # DoubleDesk seed0, 1000 步, RECORD_VIDEO=1
  base/ delta/       # result.json + videos
```

判据（先看这 4 个）：
1. Base 的 `success / final_reward / 存活步数`；
2. Δ 的同上；
3. `raw_range`（provider 自带日志）与每步 `q_target` 是否越限；
4. 视频里是否出现"抖动/撞限位/站桩"。
