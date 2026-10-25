# 交接：SuperMimic 整网微调（HA 数据 + 速度通道修复）

> 写于 2026-10-25 09:10。**接收方**：另一台 **2×RTX 4090** 容器。
> **本容器（卡 6）**不参与这件事，继续跑联合微调（见 §6）。
> 一句话：**WBC（policy）没有训错**；出问题的是两个"条件/度量口径"，需要在 IsaacLab 里
> 对 **policy + world model 整网微调**才能治本。不需要从头训，**从 `ckpt_013500.pt` 续训 2–3k 迭代**。

---

## 1. 现在到底哪里坏了（三处，全部有实测数字）

### ① token 条件空间不一致（最重要）
| 项 | 事实 |
|---|---|
| HA npz 的标签 | `g1_motion_token_64` ← 部署侧 `encoder_input (1762)` 前端（人体 SMPL + anchor + wrist 一起进 encoder） |
| our WBC 的条件 | trackerlab 的**在线 640 装配**：10 帧未来参考 `q/qd`(IsaacLab 序, 580) + anchor(60) |
| 差异 | `max|Δtoken| = 1.0625`（HA，≈17 个 FSQ 格）/ `0.4375`（ft318，7 格） |
| 复现 | `python3 trackerlab-supermimic/scripts/rls/supermimic/check_sonic_token.py --skip dyn --npz_dir ../data/npz/ha_tokens` → 打印 `[FAIL] SONIC token port disagrees — do not train` |
| 后果 | `token_mse`（对 HA 标签）**不是**保真度指标；用标签喂 bundle 基本跟不住参考；联合微调只能把 token 推到 WBC 的端口空间（token_mse 0.069→0.53 是这么来的） |

> 注：`bones_seed_token_ha_v2` 的 `args.json` 里 `motion_packed = bones_seed/bones_seed_compact.pt`
>（只有 71 维 ref，**没有 token 字段**），所以它回落到**在线 encoder** → 学到的是端口空间。
> **run 名里的 `_ha_` 有误导性，它并没有用 HA 的标签 token 训练。**

### ② world model 的速度通道没学出来
| 项 | 事实 |
|---|---|
| 位置/朝向/关节输出 | 参考动作下 1e-4 / 0.035 / 0.030（**很好**） |
| 速度输出 | 动作给 0 和给 2 倍，预测 `|qd|` 从 1.24 只变到 0.99（参考 0.48）；与参考相关 **0.09**，幅值 **2.5×** |
| 后果 | `state` 项 96% 的能量来自不可学的速度噪声；降权 0.05 后 `state_term` 2.87 → 0.18（本容器的规避方案） |
| 根源 | WM 是"一步预测"训的，位置项主导 loss；速度被 PhysX 里带 `action_noise=0.1` 的策略噪声主导 → 回归成均值场 |

### ③ 动作空间还没定论（警告，别踩）
- 原始代码把"参考关节目标→动作"写成 `a = Δq / action_scale`（per-joint 0.0745–0.5476）；本容器改成 `a = Δq`。
- 但**两者都不能证明是对的**：用 `Δq` 去标定 WM（`scripts/finetune_wm_velocity.py`）会让 WM 的
  动作敏感度与 policy 失配 —— 参考动作看着很好（state 2.62→0.36），**policy 自己的 rollout 反而崩到 29.6**。
- 结论：**WM 的动作输入空间 = policy 输出的动作空间**；任何标定/微调都只能把
  `a = π(obs, token)`（policy 实际吐的动作）当输入，不能用 Δq 之类的手算量。
- 本容器产出的 `ckpt_013500_wmvel*.pt` 是**无效标定的产物，不要用**。

---

## 2. 接收方的任务：整网微调 v1（不要从头训）

### 2.1 数据准备（CPU，10–20 min）
```bash
cd <comimic>
python3 trackerlab-supermimic/scripts/rls/supermimic/prepare_token_data.py \
  --src data/npz/ha_tokens --out data/token_motions/ha
```
- 每条 clip 产出 `<clip>.csv`（36 列：root xyz + quat xyzw + 29 joints）和
  `<clip>.tokens.npy`（**HA 的 `g1_motion_token_64` 原样，零归一化**）。
- 本容器只跑出 **424/699**（被中断），重跑补齐即可（已存在的会覆盖，幂等）。
- 699 clips / 634,016 frames。

### 2.2 代码改动（**必须做**，否则整网微调也白搭）

#### 2.2.0 根因：Isaac 移植把 mjlab 的 `explicit_integration` 丢了（**首选修法**）

mjlab 的 WM（`third_party/SuperMimic-Reproduced/src/supermimic/world_model/state_spec.py:55,103`）：
```python
explicit_integration: bool = True                 # 默认开
dt = self.dt if cfg.explicit_integration else 0.0
p_new  = p  + dt * v_new + quat_apply(hq, dp_h)   # ← 位置含 dt×速度
q_new  = normalize(q * rotvec(dt * w_new + dth))  # ← 朝向含 dt×角速度
qj_new = qj + dt * qd_new + dq                    # ← 关节含 dt×关节速度
```
→ 位置/朝向/关节的更新里**含 `dt × 速度`**：速度一旦错，位置立刻错，所以**速度通道被位置误差
带着学**，不需要任何额外监督。

trackerlab/Isaac 版（`trackerLab/algorithms/supermimic/networks.py:105-115`）：
```python
p1 = p + H @ dp ; R1 = R @ so3_exp(dth) ; q1 = q + dq
v1 = v + R @ dv  ; w1 = w + R @ dw       ; qd1 = qd + dqd
```
→ **六路输出完全独立**，没有 dt 耦合。位置头能拟合到 1e-4，速度头可以是均值场——不会被惩罚。
PhysX 的速度噪声（`action_noise=0.1` + 接触/PD）只是让这个缺陷更容易暴露，不是根因。

**修法（推荐，约 5 行）**：把 step 改成 mjlab 的半隐式积分，再小步微调 WM（参数含义变了，必须微调）：

> **可直接照抄的实现（含 `targets` 的逆变换、预热、验收、回退）见
> [`patches/wm_explicit_integration.md`](../patches/wm_explicit_integration.md)** —— 注意 `step` 与
> `targets` 必须成对改，只改一个会让网络输出与积分矛盾。
```python
def step(self, S, a, xi_n):
    p, R, v, w, q, qd = S
    d = run_mlp(self.net, self.in_norm(self.inputs(S, a, xi_n))) * self.out_norm.std() + self.out_norm.mean
    H = yaw_mat(R)
    dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, self.num_joints, self.num_joints), -1)
    v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
    w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
    qd1 = qd + dqd
    dt = CONTROL_DT                                  # 0.02（task cfg）
    p1 = p + dt * v1 + (H @ dp.unsqueeze(-1)).squeeze(-1)
    R1 = R @ so3_exp(dt * w1 + dth)
    return (p1, R1, v1, w1, q + dt * qd1 + dq, qd1)
```
- 验收：参考动作下 `corr(pred_qd, ref_qd) > 0.9`、幅值比 ∈[0.9,1.1]（现在 0.09/2.5×）；
- 本容器已实测：**只要速度有正确监督，600 步有监督微调就能把 corr 拉到 0.93**，所以恢复结构 +
  几百步微调足够，不需要重训；
- 注意：这会改变 WM 输出的**含义**（`dq` 变成"残差"而不是整步增量），必须重训/微调，且**不能**
  直接拿旧 `wm_vel*.pt` 混用。

**✅ 本容器已落地并验收（2026-10-25 09:45，卡 6）**
- 代码：`networks.py` 已支持显式积分（`step` + `targets` 成对），结构由 ckpt 的
  `wm_integration` 键自动选择（`isaac_joint.py`）；默认 OFF → 官方 ckpt 行为不变。
- 代数：`max|step(targets(S,S1)) − S1| = 1.19e-7`（两种结构都过）。
- 端到端标定（`scripts/finetune_wm_velocity_policy.py --steps 300`，用 policy 自己的动作）：
  | 指标 | 旧结构（residual） | 新结构·未标定 | **新结构 + 300 步标定** |
  |---|---|---|---|
  | policy rollout `state_term`（`state_vel=1.0`） | 2.87 | 3.79 | **0.59** |
  | `rollout_reward` | 0.0703 | 0.0552 | **0.0757** |
  | WM `qd` 相关 / 幅值比 | 0.155 / 2.55 | 0.065 / 4.43 | **0.90 / 0.92** |
  | 关节 / 位置 MSE | 0.0025 / 0 | 0.0159 / 0 | **1e-4 / 0** |
- 产物：`trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500_physint.pt`
  （带 `wm_integration="explicit_dt0.02"`，加载即自动启用新结构）。
- 结论：**速度问题已由结构修复解决**（同一权重口径下 state_term 2.87→0.59）；
  **token 空间问题仍在**，需要整网微调（`tokens.npy` 条件）或 VLA 重训。

#### 2.2.1（备选）如果不想改结构：给速度维加权 + 参考速度辅助项
下面两条是"不改结构"的补丁，效果取决于数据；**优先级低于 2.2.0**。

现状：每维误差下限 `ERR_FLOOR` 里 `dv/dw/dqd` 是 2e-2/5e-2（比位置大 6–17 倍），再对 70 维**等权平均**
（`runner.py:228` 的 `_wm_err`），速度梯度被双重压小。
```python
VEL_W = torch.cat((torch.ones(6), torch.full((6,), args.wm_vel_weight),
                   torch.ones(spec.num_joints), torch.full((spec.num_joints,), args.wm_vel_weight))).to(dev)
e = (e * VEL_W).mean(-1)          # _wm_err 里唯一改动
```
再加上"参考速度辅助项"（动作必须用 `pi(obs_ref, token_ref)`，不可用手算 Δq）：
```python
def ref_vel_aux(n=1024):
    b = sample_reference_windows(n)
    with torch.no_grad():
        a = pi(b.obs_hist, b.token)
    S_hat = wm.step(b.S, a, b.xi)
    e = wm.state_error(S_hat, b.S1) / torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    return (VEL_W * e.square()).mean()
loss = wm_err(...) + args.wm_ref_vel_weight * ref_vel_aux()
```
> ⚠️ 这一项的代价：它把 WM 从"仿真动力学模型"改造成"参考跟踪动力学模型"，会削弱对真实噪声/
> 接触扰动的建模。**只用于离线，不做论文口径**；如果 2.2.0 能落地，就不要用它。
```python
# runner.py:215  —— 每维误差下限（70 维：dp3 dθ3 dv3 dw3 dq29 dqd29）
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3),
                       torch.full((3,), 2e-2), torch.full((3,), 5e-2),   # dv, dw 的 floor 很大
                       torch.full((num_joints,), 3e-3),
                       torch.full((num_joints,), 5e-2))).to(dev)

# runner.py:228 —— 损失 = 归一化后 **70 维等权平均**
def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    e = (wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2   # (B,k,70)
    e = e.mean(-1)                                                # ← 等权！
    return (e * mask).sum() / s1.shape[0], e[:, -1]
```
速度的 35 维（`dv 6:9` / `dw 9:12` / `dqd 41:70`）不但和位置等权，而且 `dv/dw/dqd` 的
`ERR_FLOOR` 被设成 2e-2/5e-2（比位置大 6–17 倍）→ 归一化后误差被压小 → 梯度更弱。

#### 2.2.2 第一步：给速度维加权（必要，但**大概率不够**）
```python
# 70 维权重：dp/dθ/dq 权重 1；dv/dw/dqd 权重 args.wm_vel_weight（建议 5–10）
VEL_W = torch.cat((torch.ones(6),
                   torch.full((6,), args.wm_vel_weight),
                   torch.ones(spec.num_joints),
                   torch.full((spec.num_joints,), args.wm_vel_weight))).to(dev)

def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    e = (wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2
    e = (e * VEL_W).mean(-1)          # ← 唯一改动
    return (e * mask).sum() / s1.shape[0], e[:, -1]
```
**为什么说不够**：WM 是在 PhysX 一步转移上回归的，而仿真里的速度增量被
`action_noise=0.1` + PD/接触噪声主导，`(S, a, xi) → Δv` 的条件期望本来就接近"均值场"。
单纯加权只会让它把**仿真噪声**拟合得更像，**不会自动对齐到参考速度**（我们离线用的定义）。
本容器实测：动作给 0 给 2 倍，预测 `|qd|` 只从 1.24 变到 0.99（参考 0.48）。

#### 2.2.3 第二步（关键）：加一个"参考速度"辅助项
在 `wm_update()` 里加一项，用**参考转移**训练速度通道（rollout-free，开销很小）：
```python
def ref_vel_aux(n=1024):
    # 1. 从 HA motion 库随机取参考窗口（含 tokens.npy 的每帧标签）
    b = sample_reference_windows(n)              # 参考帧 f、S、S1、obs_hist、token
    # 2. 动作必须用 **policy 自己吐的**（不可用手算 Δq！本容器踩过：会把 policy 一致性搞坏）
    with torch.no_grad():
        a = pi(b.obs_hist, b.token)              # ← 唯一合法的动作输入
    # 3. 只对速度块加权
    S_hat = wm.step(S, a, xi)
    e = wm.state_error(S_hat, S1) / torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    return (VEL_W * e.square()).mean()

# wm_update() 里：
loss, e_last = wm_err(S_hat, w["s1"], w["mask"])
if args.wm_ref_vel_weight > 0:
    loss = loss + args.wm_ref_vel_weight * ref_vel_aux()
```
- 新增参数：`--wm-vel-weight`（第一段，建议 5）、`--wm-ref-vel-weight`（第二段，建议 1–3）；
- 参考窗口可以用现成的 `motion`（`data/token_motions/ha` 的 CSV 库）+ `tokens.npy` + `mdp.obs`；
- **验收判据只认这一条**：用参考动作 / 用 policy 动作各跑 8 步想象 rollout，
  `corr(pred_qd, ref_qd) > 0.9` 且 `|pred_qd|/|ref_qd| ∈ [0.9,1.1]`，且 `sim_rew` 不掉。
- 本容器的经验：**这个辅助项单独离线跑 600 步就能把 corr 从 0.155 拉到 0.93**（
  `scripts/finetune_wm_velocity.py`），所以整网微调里只需很小权重即可，不必重构网络。

#### 2.2.4 如果两段都做不到（回退方案）
保持 WM 原样、**离线不用它的速度通道**：`train_joint.py --state-vel-weight 0.05`
（本容器现在的做法，`state_term` 2.87→0.18），速度精度以 A1 闭环成功率为准。
这条不是治本，但可交付、可复现。

#### 2.2.5 条件 token 用 `.tokens.npy`（接口治本）
trackerlab 原生支持（`trackerlab/motion_buffer/motion_lib/supermimic.py:134` 检测
`*.tokens.npy` 并直接当条件），只要 `--motion_dir` 指向 `data/token_motions/ha`、
`--interface token`，就**不会再回落到在线 encoder**。
启动后请确认日志里没有回落到 `SonicTokenBuilder`（可在 `token_of` 处打一行，或 grep
`meta.json` 里是否记录 `tokens: true`）。

### 2.3 启动（2 卡 4090，续训）
以 `trackerlab-supermimic/logs/bones_seed_token_ha_v2/args.json` 为模板，关键改动：
```bash
cd trackerlab-supermimic
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 scripts/rls/supermimic/train.py \
  --interface token \
  --motion_dir  <comimic>/data/token_motions/ha \
  --resume     <comimic>/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500.pt \
  --num_envs 8192 \
  --iterations 2500 \
  --lr 5e-5 \
  --pi_batch 4096 --wm_batch 2048 --wm_batches 10 \
  --horizon 32 --goal_frames 10 --hist 10 \
  --log_dir <comimic>/trackerlab-supermimic/logs/ha_token_v1 \
  --ckpt_every 500
```
- 原 run：8192 envs、lr 2e-4、32k 迭代、峰值 22 GB/卡、10.6 h（2 卡）。
- 微调建议：**lr 降到 5e-5**、**2–3k 迭代**（≈1–2 h，2 卡），`--ckpt_every 500` 便于挑。
- 参数名以 `train.py --help` 为准（`num_envs/iterations/lr/pi_batch/wm_batch/wm_batches` 均来自 args.json）。
- 若显存吃紧：`--num_envs 8192` 单卡可能到 44 GB，2 卡各 4096 稳一些（原 run 就是每卡 4096）。

### 2.4 验收线（不过不许交接）
1. `train_log.csv` 里 `sim_rew` **不低于 0.84**（原 32k 是 0.8488），`term_frac` 保持 ~0.002；
2. 用本容器的探针验证**速度通道**（同一 packed `runs/isaac_motion/ha_tokens/packed.pt`）：
   `corr(pred_qd, ref_qd) > 0.9`、`|pred_qd|/|ref_qd| ∈ [0.9, 1.1]`（现在 0.09 / 2.5×）；
3. 用**policy 自己的动作**跑 8 步想象 rollout（`scripts/probe_port_tokens.py` 的 score 部分），
   `state_term` 明显低于 0.18、`rollout_reward ≥ 0.07`（这是端到端，唯一算数的口径）；
4. 产物：`logs/ha_token_v1/ckpt_*` + 一份 `docs/ABLATION_SM_FINETUNE_HA.md`（含三条验收数字）。

### 2.5 交付物（回传给本容器）
- 新的 WBC ckpt 路径（建议 `trackerlab-supermimic/logs/ha_token_v1/ckpt_001500.pt` 之类）
- 验收数字（上面 3 条）+ 训练日志
- 明确说明：**条件 token = HA 标签（tokens.npy）**，还是别的空间 —— 这决定本容器要不要继续用
  端口 token 侧车（`data/port_tokens/ha_tokens/`）。

---

## 3. 不要做的事（都是本容器踩过的坑）
1. ❌ 用 `ckpt_013500_wmvel*.pt`（无效标定，policy 一致性被破坏）。
2. ❌ 用 `a = Δq` 或 `Δq/action_scale` 当 WM 输入去标定/微调 WM（必须用 policy 的动作）。
3. ❌ 把 `token_mse`（对 HA 标签）当保真度指标（两套空间）。
4. ❌ 在 `train_joint.py` 的离线想象里指望速度项（WM 速度通道现在不可用，本容器降到
   `--state-vel-weight 0.05`；整网微调验收通过后才可改回 1.0）。
5. ❌ 用 mjlab（`third_party/SuperMimic-Reproduced`）的权重/解析头（永久禁用，见
   `docs/DATA_POLICY_HA_ONLY.md`）。

---

## 4. 参考数字（复现用）
| 检查 | 命令 | 期望/现状 |
|---|---|---|
| token parity | `check_sonic_token.py --skip dyn --npz_dir ../data/npz/ha_tokens` | 现状 `[FAIL]` 1.0625；修好后应 ~1e-7 |
| 端口 token vs 标签 | `scripts/probe_port_tokens.py` | 端口 0.070/2.87，标签 0.052/6.48（旧 WM） |
| 速度通道 | `scripts/finetune_wm_velocity.py --steps 0`（或看其 `[before]`） | corr 0.155 / ratio 2.55（坏） |
| 动作尺度证据 | `scripts/probe_action_decomp.py` | 参考动作 `Δq`：+0.0686 / 末步 0.047 |
| 联合微调起点 | `scripts/probe_reward_landscape.py --alphas 1.0` | 端口 token + 降权：state 0.18 |

---

## 5. 背景：为什么"整网"比"只改 WM"对
- policy 和 WM 是一起训出来的，**只改 WM 会让这一对失配**（本容器实测：WM 标定后 policy rollout 2.87 → 29.6）；
- 条件 token 是两个部件共享的接口，改它必然要动 policy；
- 但**不需要从头训**：`--resume ckpt_013500.pt` + 小 lr + 2–3k 迭代足够（policy 已收敛，只是换分布）。

---

## 6. 本容器（卡 6）在做什么（避免重复劳动）
- 跑**联合微调**（`scripts/launch_joint_ha.sh`，4000 步，纯 HA，`ckpt_013500.pt` 冻结）：
  - 动作换算：`a = Δq`（`COMIMIC_ACTION_SCALE=one`，默认）
  - 条件 token：**端口 token 侧车** `data/port_tokens/ha_tokens/*.npy`（`RecordedTokenSource` 自动优先）
  - `--state-vel-weight 0.05`
  - 日志 `logs/joint_ha_a1.log`，产物 `runs/joint_ha_a1/vla_004000.pt`
- 训完自动接力 A1 闭环（`scripts/a1_pipeline_after_train.sh` → `runs/ckpt_vla_ha_ft4000` → 六任务×3 seeds）。
- 已知局限：速度通道未治本、token 空间靠侧车绕开。**若接收方的整网微调验收通过，本容器应改用新 WBC 并把
  `--state-vel-weight` 恢复 1.0、删除端口侧车**（更新 `runs/OFFICIAL_SUPERMIMIC.json` 与
  `scripts/launch_joint_ha.sh` 的 `SM_CKPT` 默认值）。

---

## 7. 相关文档
- `docs/ISAAC_ADAPTER_FIXES_20261025.md` —— 本容器已做的三处修复与开关
- `docs/DATA_POLICY_HA_ONLY.md` —— 纯 HA / IsaacLab 铁律
- `docs/ABLATION_ALLOCATION.md` §5 —— 接口对齐规格（1054 vs 994、LayerNorm 等）
- `docs/HANDOVER_ISAAC_CLOSEDLOOP.md` —— 闭环踩坑史（含 `w` 体坐标 vs 世界坐标的警告）
