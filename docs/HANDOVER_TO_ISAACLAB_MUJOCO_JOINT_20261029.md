# 交接给 IsaacLab 线：MuJoCo 侧「联合微调」的架构 / 参数 / 流程（2026-10-29 定稿）

> 目的：把我们在 **mjlab + MuJoCo** 这条线上跑通的「VLA head + SuperMimic WBC + 学习式世界模型」联合微调，
> 逐项写清**结构、契约、数据流、参数、判据、实测数字、踩过的坑**，供 IsaacLab 侧参考/对齐。
> 阅读顺序建议：§1 一页速览 → §3 契约 → §5 流程 → §6 参数 → §8 实测 → §10 坑（最省时间）。

---

## 1. 一页速览

```
318 运动库(473,876 帧) ─┐
ft318_tokens(64 维)     ├─► SuperMimic WBC(8.96M, 29 dof) ──► 世界模型 WM(8.82M) ──► tracking reward
ego 相机 ─► VLA(head 只训 LoRA 67M) ─► 40×64 token ─┘                    │
                                                                         └─► critic(0.235M, GAE)
闭环：MuJoCo 43-dof 场景（29 身体 + 14 指 + 桌子/物体），50 Hz 控制，PD 500 Hz
```

| 项 | 我们锁死的选择 | 一句话理由 |
|---|---|---|
| 观测 | **93 维**（term-major 29/29/3/3/29）+ 10 帧历史 ⇒ 单帧 93 | 与 Isaac 侧对齐 |
| 动作 | **29 维**归一化关节偏置（不含手） | WBC 只控身体 |
| token | **64 维**，空间 = 数据集 label/hold | 与数据集录制 token 同一空间 |
| 信用分配 | **H=8 pathwise（梯度穿 WM）+ critic GAE bootstrap** | H=5 太短、远端全靠 critic；H=20 太慢 |
| critic | **GAE（imagined 源）**，`v_ev 0.95–0.999` | real 监督只有 0.09–0.15，结构性不可学（见 §10.2） |
| WBC | **冻结** | 解冻后闭环崩 21×（见 §10.1） |
| WM | **每轮开头用「25% B 线 + 75% 仿真」混合数据训 1000 步 → 整轮冻结** | 只在仿真数据上训的 WM 在部署分布 `qd_corr` 只有 0.638 |
| reward | body 4×1.0 + **root 2×1.0**，`std_root 0.2 / 0.25` | root 项原来在 4 cm 处饱和 |
| 手 | VLA 直给 14 维关节目标，`λ_hand=0` | 你的口径 |

---

## 2. 组件与仓库位置

| 组件 | 文件 | 说明 |
|---|---|---|
| mjlab 侧 bundle | `diff_vla_wbc/mjlab_joint.py` | 把 supermimic 的 `Policy/WM/AnalyticHeads` 包成统一接口：`step()`、`ref_state_at()`、`obs_from_states()`、`save_checkpoint()` |
| 联合训练主循环 | `diff_vla_wbc/v3_step.py` | pathwise rollout + reward + trust/act hinge + critic 更新 |
| 想象 rollout | `diff_vla_wbc/rollout.py` | `imagine_with_tokens()`：**无 detach**，H 步串成一张图（真 BPTT） |
| WM 保鲜/预训练 | `diff_vla_wbc/wm_refresh.py`、`scripts/pretrain_wm.py` | 用真转移训 WM，训完 `freeze_world_model()` |
| 联合训练入口 | `scripts/train_joint_mjlab.py` | 单次微调（含门禁、critic、WM 预训练参数） |
| 多轮编排 | `scripts/run_j1_rounds.sh` | 每轮 9 步 + 累积 + 判停 |
| 安全启动器 | `scripts/launch_j1_rounds.sh` | `setsid+nohup` + 20 s 自检 + 参数透传 |
| 闭环采集/评测 | `scripts/closed_loop_vla_supermimic.py` | VLA→token→WBC→MuJoCo，写 dump（states/actions/tokens/frames/rewards） |
| 漂移度量 | `scripts/eval_realized_drift.py` | **六维**真实执行漂移（root pos/ori、29 关节角/速度、root 线/角速度） |
| 数据混合 | `scripts/bline_to_transitions.py`、`scripts/make_wm_mix.py` | B 线 dump → transitions 格式；按比例混合 |
| 资产 | `scripts/make_g1_43dof_asset.py` | 生成 43 关节 position 执行器资产（mjlab 用无 actuator 版） |

---

## 3. 契约（这三条必须逐位对齐）

### 3.1 状态 / 观测 / 动作

```
state  (71) = [p(3), quat wxyz(4), q(29), v_world(3), w_world(3), qd(29)]
obs    (93) = [q − q0(29), qd(29), w_body(3), projected_gravity(3), a_prev(29)]   （term-major）
action (29) = 归一化关节位置偏置，q_target = q_default + action_scale · a
              action_scale = 0.25 · effort_limit / kp   （逐关节）
```

> **坑**：motion library 里存的是 **world 系**角速度，而 obs/WM 用 **body 系** ⇒ 交换时必须
> `quat_apply_inverse(quat, w_world)`（我们踩过，见 §10.6）。

### 3.2 token

- `token_dim = 64`，空间 = **数据集录制的 label/hold token**（`ft318_tokens.pt`，473,876×64）。
- WBC 的 decoder 第一层宽度 = `64 + 930`（64 token + 930 = 93×10 历史）。
- VLA 每次输出 **40×64**（动作 horizon 40 帧 @50 Hz）。训练时取前 `K` 个 token 用于 H 步 rollout。

### 3.3 执行器（**这一条 MuJoCo/Isaac 差异最大**）

| | W1 训练（mjlab） | 闭环（MuJoCo 场景） |
|---|---|---|
| 关节数 | 29（无手） | **43**（29 + 14 指） |
| 执行器 | `BuiltinPositionActuatorCfg`（位置伺服） | 43 个 `<motor>`（力矩）+ 脚本内 PD |
| 增益 | `kp = armature·ω²`（ω=10 Hz·2π）、`kd = 2ζ·armature·ω`（ζ=2），腰 pitch/roll + 双踝 ×2 | **同一组数**（`groot_repro/g1_sim.py` 用同一公式算 `KP/KD/ACTION_SCALE`） |

数值（供 Isaac 侧核对）：`5020 kp14.3/kd0.91`、`7520_14 40.2/2.56`、`7520_22 99.1/6.31`、
`4010 16.8/1.07`、并联组 ×2 = `28.6/1.82`；手 `kp2.0/kd0.05`，力矩限幅 ±2.4。

我们另外生成了 `g1_43dof_position.xml`（43 个 `<position>`，供"训练/闭环同一份资产"），
但**闭环目前仍走 motor+PD**（数值等价，改动收益≈0）。

---

## 4. 数据：四种，各管一段（**容易混，务必分清**）

| 数据 | 规模/轮 | 谁产生 | 用来干什么 |
|---|---|---|---|
| `transitions.pt` | **51 万行**（512 envs × 1000 tick） | `collect_mjlab.py`：**WBC + 动作噪声**在 mjlab 里并行推进，**不含 VLA、平地无物体** | WM 训练的主干数据；critic 真 return 的来源 |
| **B 线 dump** `b_*.pt` | **4.8 万步**（4 任务 × 20 集 × 600 tick） | `closed_loop_vla_supermimic.py --arm vla`：**VLA 真推理、任务场景、有物体** | ① `self` 锚（trust_center=self）② `--start-from-b` 训练起点 ③ **WM 混合训练** ④ 闭环评测 |
| 318 库 `ft318_compact.pt` | 473,876 帧 / 318 clips / 9625 bins | 数据集（真机录制） | 参考动作、`ref_state_at`、reward 的参考帧 |
| `ft318_tokens.pt` | 473,876×64 | 数据集 | token 空间定义（label/hold） |

> **常见误解**：B 线 dump **不会**加入 318 库（库固定）；B 线也**不是** WM 的唯一数据
> （它只占混合集的 25%）。

---

## 5. 每轮 9 步流程（`run_j1_rounds.sh`）

| # | 步骤 | 关键点 | 产物 |
|---|---|---|---|
| 1/9 | 仿真采集 | WBC 冻结 ⇒ 第 2 轮起**硬链接复用**第 1 轮（省 2–3 min） | `transitions.pt` |
| 2/9 | **B 线闭环采集** | `arm=vla`；起点**每轮随机**（`--seed $R`，任务帧区间内分层随机，保证覆盖 + 跨轮不同） | `bline/b_*.pt`、`self_targets.pt` |
| 3/9 | 混合 WM 数据集 | B 线 → transitions 格式（补 `ret/xi/ref_rows/clip_ids`），与仿真按 **B 线 25%** 混合 | `bline_transitions.pt`、`wm_mix.pt` |
| 4/9 | **训练前把 WM 训稳** | 用 `wm_mix` 训 **1000 步**，训完冻结；之后门/训练/评测都用它 | `wm_pre.pt` |
| 5/9 | WM 双分布门 | sim 门（`qd_corr ≥ 0.95` **硬停**）+ B 线部署分布门（记录趋势） | `gate_wm_sim.json`、`gate_wm_bline.json` |
| 6/9 | 五道门禁 | `A_codec / B_freeze / C_burnin / C2_token_contrast / D_fm_bc`（小卡 OOM 记 skipped 继续） | `gates/*.json` |
| 7/9 | critic | **GAE（imagined）**；目标来自 4/9 训好的 WM 的想象 rollout | `critic.pt` |
| 8/9 | 联合微调 | 800 步，**WM 全程冻结**（`--wm-pretrain-steps 0 --wm-refresh-every 0`） | `train/joint_final.pt`、`train/vla_final.pt` |
| 9/9 | 导出 + 评测 + 接下一轮 | 合并 LoRA → `ckpt_vla_r$R`；600 tick 闭环评测；训练后**再测** bline 门；把 `joint_final.pt` 接给下一轮 | `ckpt_vla_r$R`、`eval.json`、`gate_wm_bline_after.json` |

### 5.1 累积链（**重点**：VLA 和 WM 都逐轮累积）

```
R1: VLA=原始 ckpt        WBC/WM=W1 最佳 ckpt  → ckpt_vla_r1 + joint_final_1
R2: VLA=ckpt_vla_r1      WBC/WM=joint_final_1（再 +1000 步 wm_mix） → ckpt_vla_r2 + joint_final_2
R3: VLA=ckpt_vla_r2      WBC/WM=joint_final_2（再 +1000 步 wm_mix） → …
```

- **为什么 B 线每轮必须重采**：VLA head 变了 ⇒ 闭环行为、`self` 锚（"当时真正下发的 token"）、
  真实漂移分布都变；复用旧 B 线 = 拿旧 head 的锚训新 head。
- **为什么仿真采集只采一次**：`collect_mjlab.py` 只跑 WBC（冻结）⇒ 每轮数据等价。

---

## 6. 联合微调参数（锁死值 + 依据）

```bash
# 单次微调（脚本内调用；等价的命令行）
python3 -m torch.distributed.run --nproc_per_node 4 scripts/train_joint_mjlab.py \
  --wbc-ckpt  ${RD}/wm_pre.pt --motion data/npz/g1_tasks --tokens data/library/ft318_tokens.pt \
  --vlm-ckpt ${CUR_VLA} --vlm-path groot_repro/nvidia/Cosmos-Reason2-2B-weights \
  --out ${RD}/train --steps 800 --v3 \
  --lambda-bc 0 --lambda-hand 0 --lambda-r 0.3 --fm-lr 1e-4 \
  --critic --critic-source imagined \
  --act-eps 0.6 --act-weight 1.0 --act-hinge-cap 2 \
  --vla-state-from-burnin \
  --real-transitions ${RD}/wm_mix.pt --start-source reference \
  --start-from-b '${RD}/bline/b_*.pt' --b-self-targets ${RD}/self_targets.pt \
  --wm-pretrain-steps 0 --wm-refresh-every 0 \
  --anchor-weight 0 --trust-eps 0.2 --trust-weight 1.0 --trust-center self \
  --batch-size 16 --grad-accum 1 --horizon 8 --lora-rank 64 --amp bf16 --device cuda
```

| 参数 | 值 | 依据（都是实测） |
|---|---|---|
| `--steps` | 800 | 4×4090 约 55 min |
| `--v3` + `backend=pathwise` | 开 | 梯度穿 WM，H 步真 BPTT |
| `--horizon` | **8** | H=5 时 >0.1 s 的后果全压给 critic（而 critic 当时不可用）；H=20 慢 2.5× |
| `--lambda-r` | 0.3 | 奖励项权重 |
| `--lambda-bc` | **0** | token 不做 BC；在漂移状态上拉向 label 反而变差 |
| `--lambda-hand` | **0** | 手部不训（部署用 `hand_mode` 决定） |
| `--anchor-weight` | **0** | 关锚定（参考 token 不是纠偏方向） |
| `--trust-eps / weight / center` | **0.2 / 1.0 / self** | 信任域中心 = B 线采集时**真正下发**的 token |
| `--act-eps / weight / hinge-cap` | **0.6 / 1.0 / 2** | 动作层信任域（归一化动作的逐关节 RMS） |
| `--lora-rank` | 64 | 67,108,864 可训参数（head 的 3.977%） |
| `--batch-size` | 16/rank（4 卡） | 4090 单卡 32 必 OOM |
| `--fm-lr` | 1e-4 | 实际经 cosine 调度（见 §10.4） |
| `--critic` / `--critic-source` | on / **imagined** | GAE；real 监督 v_ev 只有 0.09–0.15 |
| `--lam` / `--gamma` | 0.95 / 0.99 | GAE |
| `--wm-pretrain-steps` / `--wm-refresh-every` | **0 / 0** | WM 已在 4/9 训好并冻结 ⇒ reward 固定靶 |

### 6.1 reward 配置（`TrackingRewardCfg`）

| 项 | 权重 | std |
|---|---|---|
| `motion_body_pos` / `_ori` / `_lin_vel` / `_ang_vel` | 1.0 / 1.0 / 1.0 / 1.0 | 0.3 / 0.4 / 1.0 / 3.14 |
| **`motion_global_root_pos` / `_ori`** | **1.0 / 1.0** | **0.2 / 0.25** |
| `action_rate_l2` | −0.1 | — |
| `joint_limit` | −2.0 | — |
| `dt` | 0.02 | — |
| ⇒ `max_step_reward` | **0.12** | `dt·Σw` |

跟踪的 14 个 body（**含末端**）：`pelvis / torso_link`、双 `hip_roll/knee/ankle_roll`（脚）、双 `shoulder_roll/elbow/wrist_yaw`（腕）。

> **注意**：reward 里**没有关节角跟踪项**（只有 `joint_limit` 罚越界）。所以 `joint_rad 0.3 rad`
> 这种偏大不会自动被优化 —— 它是冗余自由度分配差异，不等于动作错（见 §9）。

---

## 7. 三个关键机制（照抄时最容易漏）

### 7.1 pathwise + critic bootstrap

```
loss_reward = −λ_r · Σ_{t=0..H−1} γ^t · r_t  −  λ_r · γ^H · V(s_H)      （V 来自 critic，detach）
```

- `imagine_with_tokens()` 里**没有 detach/no_grad**，H 步共用一张图 ⇒ 真 BPTT；
- H 步之外完全靠 `V(s_H)` ⇒ **critic 不准，这套就废**（见 §10.2）。

### 7.2 `trust_center = self`

- 中心 = **B 线采集时策略真正下发的 token**（`batch.self_tokens`，来自 `build_b_self_targets.py`）；
- 用它对同一段 rollout 再跑一遍（`self_tokens` 解码出的动作），得到 `a_ref`；
- token hinge：`relu(‖tok − tok_self‖/eps − 1)`；动作 hinge：`relu(rms(a − a_ref)/eps_a − 1)`；
- **不要用 label/prev 当中心**：在已经漂了的状态上，参考 token 不是纠偏方向。

### 7.3 WM 必须吃「部署分布」的数据

我们实测（同一个 WM，三种数据）：

| 测试分布 | `qd_corr` | `qd_ratio` |
|---|---|---|
| 仿真 transitions（平地、无 VLA） | 0.9929 | 1.006 |
| **B 线（任务场景 + VLA 在环）** | **0.6380** | **2.9475** |
| 混 25% B 线后训 1000 步，再测 B 线 | **0.9586** | 1.2621 |

⇒ 只用仿真数据的 WM 在真正要服务的分布上**速度高估 3 倍**。混 25% B 线 + 1000 步即可拉到 0.96。
（`qd_ratio` 仍 1.26 ⇒ 还有 26% 高估，下一轮把 B 线占比提到 0.35 再压。）

---

## 8. 实测数字（供 Isaac 侧对标）

### 8.1 训练端（R1，800 步，H=8）

| 指标 | 起（前 5 采样） | 末（后 5 采样） | 变化 |
|---|---|---|---|
| `rollout/reward` | 0.0837 | **0.0914**（峰值） | **+9%** |
| `act/rmse`（解码动作 vs 锚定动作） | 0.297 | 0.216–0.27 | −27% ~ −9% |
| `trust/rmse`（token vs self 锚） | 0.104 | 0.082–0.138 | −21% |
| `critic/v_ev` | 0.729 | **0.9963** | — |
| `loss/boot` | −0.096 | −0.19 ~ −0.29 | critic 在起作用 |
| `codec/out_of_range` | 0.038 | **0.097** | ⚠️ 恶化，需干预（§10.5） |

### 8.2 真实执行漂移（80 集 × 600 tick = 48,000 时刻，六维）

| 分量 | p50 | p90 |
|---|---|---|
| root 位置 | 5.8–6.0 cm | 26 cm |
| root 朝向 | 9.9–10.4° | 26–29° |
| 29 关节角 RMS | 0.306–0.309 rad | 0.51–0.54 rad |
| 29 关节速度 | 0.221 rad/s | 0.64 rad/s |
| root 线速度 | 0.020 m/s | 0.134 m/s |
| root 角速度（body 系） | 0.054 rad/s | 0.43 rad/s |
| 摔倒率 | 0–2 / 80 | — |

### 8.3 WM 门历史

| 轮 | `qd_corr`（sim） | `qd_ratio` | critic `v_ev`（real 监督） |
|---|---|---|---|
| R1（旧流程） | 0.9925 | 1.006 | 0.1524 ❌ |
| R2（旧流程） | 0.9933 | 1.005 | 0.0880 ❌ |
| R3（旧流程） | 0.9929 | 1.006 | — |

---

## 9. 判据与验收

| 门/指标 | 阈值 | 不过怎么办 |
|---|---|---|
| WM sim 门 | `qd_corr ≥ 0.95` | **硬停** |
| WM B 线门 | 记录；第 2 轮起应 > 0.85 | 抬 `WM_BLINE_RATIO` 到 0.35–0.4 |
| 训练后 − 训练前 bline 门 | ≥ −0.02 | 更差 ⇒ 说明 policy 漂移把 WM 带偏，考虑轮内重训（并同步重训 critic） |
| 主效果 | 六维漂移 p50/p90 逐轮降 | 用 `eval_realized_drift.py` |
| 辅效果 | `rounds.csv` 的 `mean_reward`（**新尺度 0.12**） | 判停：连续 2 轮主指标不改善 |
| 健康 | `codec/out_of_range` | >8% 加 `λ_bc=0.03`；>12% 再收 `act-eps` 到 0.4 |

> ⚠️ **reward 尺度**：因为 `w_root` 从 0.5 抬到 1.0，`max_step_reward` 由 0.10 → **0.12**。
> 旧轮的 `mean_reward`（如 0.0732 / 0.0598）**不可与新轮直接比**。

---

## 10. 踩过的坑（**照抄前务必看，能省一天**）

### 10.1 解冻 WBC → 闭环崩 21 倍

同 head 只换 WBC（600 tick，task1，起点 100）：

| WBC | root_pos p50 | root_ori p50 | joint_rad |
|---|---|---|---|
| 冻结（W1 ckpt） | **4.16 cm** | 6.37° | 0.596 |
| 解冻 800 步后 | **89.2 cm** | 16.15° | 0.645 |

权重只变了 **max|Δ| 4.8e-4（相对 4.8e-4）** 就崩了 —— 因为 J1 的 reward 是 **WM 想象**里的 tracking，
梯度穿过 WBC 时它只需要"让 WM 里的 reward 变大"，于是学会**骗 WM**。当时 WM 在部署分布上
`qd_corr` 只有 0.638 ⇒ 必然跑偏。**要用想象 reward 解冻 WBC，先保证 WM 在部署分布上准。**

### 10.2 critic 用「real 长 return」监督 ≈ 不可学

`--critic-source real`（回归 1000 步 return-to-go）：`v_ev` 只有 **0.088–0.152**，且逐轮下降。原因定量：

- `ret` 分布 mean 9.225 / std 1.182（变异系数仅 13%）⇒ `v_ev` 的分母本来就小；
- 单个 episode 内 `ret` 几乎不变（9.653→9.724），主要由"clip 剩余长度/难度"决定；
- `corr(ret, 剩余步数) = 0.16`，单变量线性回归 `v_ev = −16`；
- 而 critic 输入只有 `state(71)+ref(71)+xi(33)`，**看不出 clip 身份与进度** ⇒ 结构性不可学。

**改用 GAE（`--critic-source imagined`）后 `v_ev` 立刻到 0.95–0.999。**

### 10.3 H=5 太短

H=5 覆盖 0.10 s，远端信用全压给 critic。当时 critic `v_ev=0.088` ⇒ R2 出现 reward −18%、
tilt 7.3°→12.5°、root_ori +25%。**H=8 + 好 critic 才稳。**

### 10.4 LR 调度的隐形陷阱

`lr_decay_fraction = 250000/320000 = 0.78125`，`decay_steps = total × 0.78125`：

| total | 有效衰减段 | 现象 |
|---|---|---|
| 800 | 625 步 | step 410 时 LR 只剩 **28.7%**，200 步后收益递减（看起来"平台"，其实是调度） |
| 17000（续训用） | 13281 步 | 从 14050 续训 ⇒ 全程恒为 `base×0.03 = 9e-7` ⇒ **等于没训**（参数只动 4.8e-4） |

⇒ 续训/微调时 **必须**同时调 `total_iterations` 或直接把 `LR` 除以 `lr_final_ratio`。

### 10.5 `codec/out_of_range` 是什么、怎么治

- 定义：解码后归一化动作超出 `[-1, 1]` 的比例（边界来自数据统计 `raw ±1.5`）。
- 危害：① `clamp` 导数为 0 ⇒ 超界部分梯度白给；② 训练/部署语义漂移；③ 越界动作是 WBC 没见过的
  输入，执行不可预测（实测 >8% 时 tilt 上升、出现摔倒）；④ 与 WM 偏差相互作用会诱发 reward hacking。
- **解冻 WBC 治不了它**（边界是数据统计量，WBC 只是接收端）；reward 里也没有越界惩罚。
- 治法：`λ_bc=0.03` / `act-eps 0.6→0.4` / 直接加 `max(0,|x|−1)` 进 loss。

### 10.6 其它工程坑（每条都真踩过）

| 现象 | 根因 | 修法 |
|---|---|---|
| `frames` 30 vs `states` 600 尺寸不匹配 | vla 臂只在**重规划时刻**记 `frames`（600/20=30） | 按 `execute_steps` 展开到每 tick |
| `self_targets` 算出来的 `realized_err` 全错 | 默认 `--ref` 指向了另一个 634k 帧的库 | 显式传 318 库 |
| `qd_ratio` 与训练用的 WM 不匹配 | WM 门测的是 `W1_CKPT` 而非当轮实际用的 WM | 门也测 `CUR_WBC` |
| H2H 上 `eglQueryString` 是 None | PyOpenGL 选错平台 / `libEGL` 缺失 | `PYOPENGL_PLATFORM=egl`；先跑 `scripts/probe_gl.py` |
| `--wm-pretrain-steps` 报 unrecognized | argparse 只加了使用端，没加声明 | 静态检查脚本里每个 flag 是否在 `--help` 里 |
| 续训立即"成功退出" | `total_iterations < resumed iteration` | 设成 `resume + N` |
| 续训 `replay buffer has no sampleable rows` | resume 后第一轮就更新 WM，但 buffer 是空的 | `warmup_iterations = resume + 50` |

---

## 11. 给 IsaacLab 侧的直接建议

### 可以直接搬（与仿真器无关）

1. `trust_center=self` 的**语义**：用"采集时真正下发且变好"的 token 当中心，而不是 label/prev。
2. **critic 走 GAE**，不要用长 return-to-go 监督（除非给 critic 加 clip 进度/身份特征）。
3. `H ≥ 8`（配合可信 critic）；
4. **WM 训练数据必须混入闭环（VLA 在环）数据**，占比 20–30% 起步；
5. **reward 的 std 收紧**（root 0.2/0.25）而不是只抬权重 —— 抬权重不解决饱和；
6. 判据用**六维真实执行漂移**，而不是只看 mean reward（reward 在中位数处饱和）；
7. `out_of_range` 要当一等领域指标看。

### 需要按 Isaac 侧改写

| 我们 | Isaac 侧对应 |
|---|---|
| `diff_vla_wbc/mjlab_joint.py`（mjlab bundle） | trackerlab bundle（`isaac_joint.py` 已有对应实现） |
| `AnalyticHeads`（mjlab 可微 FK） | trackerlab 的解析件（注意：Isaac 侧没有可微 MuJoCo，pathwise 穿的是**解析 FK + 简化动力学**） |
| motor+PD vs position servo | USD 资产用 position actuator 配置，注意 kp/kd 要与 Isaac 的 actuator 配置一致 |
| `collect_mjlab.py`（mjlab 并行采集） | Isaac 侧的采集脚本（注意：Isaac 单步慢，采样量要重新估算） |

### 建议 Isaac 侧优先验证的三件事

1. **WM 在你��的部署分布上 `qd_corr` 是多少**（我们未混数据前是 0.638）——这是 pathwise 可信度的天花板；
2. **critic `v_ev` 能否 > 0.9**（GAE 版）——不够就别开 H<12；
3. **解冻 WBC 前先测"同 head 只换 WBC"**（3 min，600 tick）——别直接进多轮。
