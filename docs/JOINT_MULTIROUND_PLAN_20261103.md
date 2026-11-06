# 联合微调多轮完整方案（唯一执行版）

> 2026-11-03。本文档把**当前所有已改/应改未改**、**每一轮的完整架构**、**2 卡产物到 A/A'/WM 的完整流程**、
> **奖励函数最终设计**、**WM refresh 的启动时机**全部写死。
> 与之冲突的旧描述（plan §8.1、spec 里的 `linear`/`mse`/`huber`/`hinge`）以本文档为准。
>
> ---
> **2026-11-04 修订（本次）**，三件事：
> 1. **原文 §2.5 / §7 两张"应改未改"清单已过期** —— 逐项核对代码后 P0 全部已落地，
>    重写成 §7「实施状态」（已实现 / 未实现 / 已修但原文没写）；
> 2. **修掉一条致命错误**：§4.3 里 `a=(a+δ).clamp(-1,1)` 是错的，代码早在 2026-11-03 就删了；
> 3. **新增 §11–§15**：token 量化口径、**执行器等价性（本文档最大的风险项）**、
>    三个已关闭的疑虑（覆盖/相位/平均）、修正后的行动顺序、本次改动清单。

---

## 0. TL;DR

| 项 | 结论 |
|---|---|
| 奖励 | **Huber-on-error**（`d=√e`，远端线性），**split** 朝向（tilt/yaw/yaw_rate），**无 exp/hinge/velocity/posture** |
| 多轮 | **4 轮必做 + 第 5 轮看地板**：δ = 5cm → 2cm → 5mm → 2mm →（1mm） |
| 形状 | 所有轮**同一个公式**，只退火 δ（一脉相承） |
| critic | **开**（`CRITIC_ON=1`，`CRITIC_SOURCE=imagined`，`ADV_BASELINE=none`） |
| token trust | **关**（它是把 token 往标签上拉，不是执行梯度） |
| action trust | **收紧**：`ACT_EPS=0.10`、`ACT_WEIGHT=1.0`（2026-11-04 用户口径：起点永远从参考出发 ⇒ 20 tick 偏差只有几 cm ⇒ 信任域必须远小于 base 偏差 0.21；球内仍为 0） |
| WM refresh | **R1 不刷**（隔离变量、先压误差）；**R2 起**每轮先用 10–20 集 VLA 闭环（B）刷新 |
| 2 卡 WBC | 用 **two-lib** 的 `exp_WBC_exec_cmd.pt`（`token_lib_train=ha_cmd`）；先过 G1 |
| A / A' | A = 新 WBC 在 `ha_cmd_exec` 上的**名义回放**；A' = A + L1 动作噪声 + L2 起点扰动 + 多 seed |
| 起点加权 | joint FT 的参考起点用 **phase/task 加权**（`exp_start_weights.pt`）+ 覆盖审计 |
| 起点（R2+） | **仍 100% 参考**；B 只给 WM refresh；B 起点（20%）是 R2 的**可选恢复臂**，默认不开 |
| **token 量化** | `--token-quant=none`（实测亚格外推：WBC rel 0.83% / SONIC decoder rel 0.49% ⇒ 不需要 STE，见 §11） |
| **执行器** | **仍是重训 WBC，未换 release decoder** —— 这是本文档最大的风险项（§12） |
| **G1** | `exp_WBC_exec_cmd.pt` **未交付**、G1 **未跑**（§7.2）；现有执行器唯一的 G1 记录是 FAIL |
| **待办** | 只有 4 项真缺：G1 / `exp_WBC_exec_cmd.pt` / B 采集 / `START_FROM=mix`（§7.2） |

---

## 1. 当前状态（2026-11-03）

| 阶段 | 状态 | 产物/证据 |
|---|---|---|
| P0a 本机代码 | ✅ | `START_FROM`/`MOTION_PACKED`/`library_kind=command_exec`/评测 `SM_PACKED`/`run_refstart_joint.sh`/`run_a_prime_seeds.sh`/`reward_from_objects.py`/`isaac_collect_perturbed.py`/`eval_drift_curve.py`/起点加权+审计 |
| P0b `ha_cmd_exec` | ✅ G0 PASS（2 卡） | 699 clips / 634,016 帧；FK 脚 2.4 mm；索引与旧库逐项全等；`mean|Δq|=0.336 rad` |
| P1a 训练库 | ✅（2 卡） | `bones_seed_plus_ha_exec_x20.pt`：143,923 clips / 60.16 M 帧 |
| P1b 第一版 28k | ❌ **token 口径错**（`token_lib_train=None`） | G1：`npy` 228 步摔（0.560 m）、`online` 1.526 m（基线 0.321 m） |
| P1b 修正版 two-lib | ❌ **未交付**（2026-11-04 核实） | `runs/S1_handover/exp_WBC_exec_cmd.pt` **不存在**；`ckpt_027500.pt` 在 27.5k 被 Killed。见 §7.2 第 7/8 项 |
| P1c G1 | ❌ **未跑** | `runs/exp_g1/` 目录不存在；现有执行器（`exp_WBC_exec_haonly_best.pt`）唯一 G1 记录：root **0.182 m、1472 步摔 → FAIL** |
| G2 参考成功 | ✅ DoubleDesk 99/100 | `runs/wm_real/exp_object_metrics.json` |
| P2–P6 | 待 G1 | 本文档 |

**禁止使用**：`exp_WBC_exec.pt`（第一版，执行 token 空间）。

> ⚠️ **2026-11-04 状态更正**：表里 P1b/P1c 原来写"🔄 在跑 / ⏳ 等修正版"，实际是
> **未交付 + G1 未跑**。所以本文件 §4 之后所有 A'/WM/joint 的步骤，
> **目前跑出的结果都建立在一个没过 G1 的执行器上**（`exp_WBC_exec_haonly_best.pt`）。
> 这不影响"流程能不能跑通"，但**任何效果数字都不能当结论**。

---

## 2. 奖励函数（最终设计）

### 2.1 基本量（`e_*` 是平方误差）

| 符号 | 定义 | 单位 |
|---|---|---|
| `e_root_pos` | `‖p − rp‖²` | m² |
| `e_root_ori` | `rot_angle_sq(R, rR)` | rad² |
| `e_body_pos` | `mean_b ‖ref_bp_b − bp_b‖²` | m² |
| `e_body_ori` | `mean_b rot_angle_sq(ref_bR_b, bR_b)` | rad² |
| `e_body_linvel/angvel` | 速度平方误差 | —（**不用**） |
| `r_rate` | `‖(a − a_prev)·scale‖²` | — |
| `r_lim` | `Σ(relu(q − hi) + relu(lo − q))` | rad |

来源：`trackerlab-supermimic/.../mdp/supermimic.py::reward`。

### 2.2 Huber-on-error（不是 on squared error）

```text
d_root_pos  = sqrt(e_root_pos)          # m
d_root_ori  = sqrt(e_root_ori)          # rad（mono）
d_body_pos  = sqrt(e_body_pos)          # m
d_body_ori  = sqrt(e_body_ori)          # rad

h(d; δ) = 0.5·d²/δ        if d ≤ δ       # 近场二次，梯度 = d/δ
          d − 0.5·δ       if d > δ       # 远场线性，梯度 = 1（常数）

# 注：下面是 mono 版（仅展示）；实际用 §2.3 的 split 替换
#      w_root_ori·h(d_root_ori; δ_o) 这一项
r_t = -[ w_root_pos·h(d_root_pos; δ_p)
       + w_body_pos·h(d_body_pos; δ_p)
       + w_root_ori·h(d_root_ori; δ_o)      # ← mono；实际用 §2.3 split
       + w_body_ori·h(d_body_ori; δ_o) ] · dt
     + w_action_rate·r_rate·dt
     + w_joint_limit·r_lim·dt
```

- `dt = 0.02`；`w_action_rate=-0.1`、`w_joint_limit=-2.0`；
- **无 exp、无 hinge、无 posture、无 velocity**。

### 2.3 朝向：split（不用 mono）

```text
d_tilt     = sqrt(tilt²)        # rad
d_yaw      = |Δyaw|             # rad（wrap 后 abs）
d_yawrate  = sqrt(yawrate²)     # rad/s

r_ori = -[ w_tilt·h(d_tilt; δ_o) + w_yaw·h(d_yaw; δ_o) + w_yawrate·h(d_yawrate; δ_o) ]
w_tilt = 1.0, w_yaw = 0.2, w_yawrate = 0.3
```

- 用 split 而不是 mono：**绝对 yaw 不可观测**（token 是 yaw-relative），降权到 0.2；
- 仍然是 Huber-on-error，**不是 exp**；
- 实现：`reward_shape.py` 新增 `ori_split_huber`（或 `tilt_yaw_yawrate_err` + 在 `isaac_joint.py` 里套 `h`）。

### 2.4 权重 / 参数表

| 参数 | 值 | 说明 |
|---|---:|---|
| `w_root_pos` / `w_root_ori` | 1.0 / 1.0 | `REWARD_ROOT_WEIGHT=1.0` |
| `w_body_pos` / `w_body_ori` | 1.0 / 1.0 | ckpt 默认 |
| `w_tilt / w_yaw / w_yawrate` | 1.0 / 0.2 / 0.3 | split |
| `w_action_rate` / `w_joint_limit` | -0.1 / -2.0 | ckpt 默认 |
| `REWARD_VEL_SCALE` | 0 | 速度项不用 |
| `TAIL_WEIGHT` / `POSTURE_WEIGHT` | 0 / 0 | 无 hinge/posture |
| `θ_p`（位置 δ） | R1 5cm → R2 2cm → R3 5mm → R4 2mm → R5 1mm | `--reward-delta-pos` |
| `θ_o`（朝向 δ） | 同上（rad） | `--reward-delta-ori` |

### 2.5 代码状态（**2026-11-04 逐项核对后：全部已落地**）

| 改动 | 状态 | 代码位置 / 证据 |
|---|---|---|
| `REWARD_MODE="huber_err"`（`d=√e` + `h(d;δ)`） | ✅ **已实现** | `isaac_joint.py:583-599`；核函数 `_huber_on_error` (`:73`)，公式与 §2.2 逐项一致 |
| `split_huber`（tilt 1.0 / yaw 0.2 / yaw_rate 0.3，Huber 不是 exp） | ✅ **已实现** | `isaac_joint.py:641-661` + `reward_shape.tilt_yaw_yawrate_err`；有 guard：`split_huber` 只允许配 `huber_err` |
| `--reward-delta-pos/--reward-delta-ori` | ✅ **已实现** | `train_joint_v2.py:404,406`（默认 0.05）→ `:589-590` 写进 `bundle.huber_err_delta_*` |
| `ROOT_ORI_MODE` 含 `split_huber` | ✅ **已实现** | `choices` 在 `train_joint_v2.py:427`；`run_refstart_joint.sh:36` 默认已设为 `split_huber` |
| `REWARD_MODE="linear"`（R0 基线） | ✅ 保留 | 作为对照臂 |
| `huber`（对平方误差） | ⚠️ 存在但**不用** | δ 单位是 m²，不是 m，会被 `huber_err` 取代 |
| `mse` | ✅ 已删除 | 曾是四次方 |
| body std 参数 | ✅ 已加 | `--reward-body-pos-std/--reward-body-ori-std` |
| velocity `_vs` 修正 | ✅ 已修 | `linear` 下乘 0；非 saturating 不再做 exp 减法 |

> **原文要求的"R1 前必须先实现"已经满足**（`run_refstart_joint.sh:35-48` 的默认值就是新口径：
> `REWARD_MODE=huber_err` / `ROOT_ORI_MODE=split_huber` / `δ=0.05` / `CRITIC_ON=1` /
> `TRUST_WEIGHT=0` / `ACT_EPS=0.6` / `ACT_W=0.2`）。

---

## 3. 总损失与反传（所有轮不变）

```text
J_b = Σ_{t=0}^{19} γ^t c_t r_t          γ=1, H=20, c_t=survival
loss_r = -(1/B) Σ_b J_b

L = λ_r·loss_r
  + w_act·mean_{t,b}[ relu(RMS(a_t − a_t^ref)/0.6 − 1)² ]      # action trust（开，0.2）
  + 0( token trust 关 )
  + γ^H·c_H·(V(s_H) − V(ref_H)) 的 critic bootstrap            # CRITIC_ON=1
```

- `λ_r=0.3`；
- `CRITIC_ON=1`、`CRITIC_SOURCE=imagined`、`ADV_BASELINE=none`；
- `TRUST_WEIGHT=0`（token trust 关）、`ACT_EPS=0.6`、`ACT_WEIGHT=0.2`；
- `LAMBDA_BC=TOKEN_BC=LAMBDA_HAND=LAMBDA_S=0`；
- 反传：VLA(LoRA+head) → 64d token → 冻结 WBC → 29d a → 冻结 WM → 20 步；BPTT，不是 PPO；
- 物体奖励只评测。

**critic 注意**：每轮 δ 变化会改变 return 尺度；critic **warm start + 小 LR**，或每轮重拟合一次。

---

## 4. 2 卡产物 → A → A' → WM（完整流程）

### 4.1 2 卡交付

| 产物 | 路径 | 验收 |
|---|---|---|
| 执行参考库 | `runs/isaac_motion/ha_cmd_exec/packed.pt` | G0 PASS（FK 2.4 mm） |
| 训练库 | `trackerlab-supermimic/data/bones_seed/bones_seed_plus_ha_exec_x20.pt` | 143,923 / 60.16M |
| 新 WBC（two-lib） | `runs/S1_handover/exp_WBC_exec_cmd.pt` | `token_lib_train=ha_cmd`；G1 |
| G1 报告 | `runs/exp_g1/` | root <0.1 m、0 摔、走完整条 |

**G1 命令（必须）**：

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
SM_CKPT=runs/S1_handover/exp_WBC_exec_cmd.pt \
SM_PACKED=runs/isaac_motion/ha_cmd_exec/packed.pt \
TOKEN_LIB=runs/isaac_motion/ha_cmd/packed.pt \
ARMS="npy" MAX_CLIP_STEPS=0 TAG=r1_g1 \
bash scripts/p1c_g1_replay.sh
```

### 4.2 A = 名义回放（**先做 A，再加噪**）

- 用**新 WBC**（`exp_WBC_exec_cmd.pt`）+ **命令 token npy**（`runs/exp_tokens/<clip>.npy`）+ `ha_cmd_exec` 参考；
- 每条 clip 从 row 0 **整条回放**（~900 步），记录 `(s, a, s')`；
- 工具：`scripts/isaac_collect_perturbed.py --action-noise-std 0 --start-perturb-xy 0 ...`（即 A）；
- 产物：`runs/wm_real/exp_a_prime/<clip>/A.pt`（或直接进 A' 流程）。

### 4.3 A' = A + L1 + L2 + 多 seed

```bash
TASK=DoubleDesk CLIP=<clip> \
TOKEN_NPY=runs/exp_tokens/<clip>.npy \
WBC=runs/S1_handover/exp_WBC_exec_cmd.pt \
PACKED=runs/isaac_motion/ha_cmd_exec/packed.pt \
SEEDS="0 1 2 3" STEPS=<clip_len> \
bash scripts/run_a_prime_seeds.sh
```

- **L1 动作噪声**：`a = pol(...)` 之后，每 `hold=20` 步重采 `δ~N(0,σ²)`，`a = a + δ`；
  `σ=0.05`；**记录加噪后的 a、last_a 也用加噪后的**（`isaac_collect_perturbed.py:269,286,288`）；
  > ⚠️ **原文写的 `(a+δ).clamp(-1,1)` 是致命错误，代码已于 2026-11-03 删除**
  > （`isaac_collect_perturbed.py:270-276` 有完整注释）：该 WBC 训练时 `action_limit=None`，
  > 实测 `|a|` 最大 **18.6**、**34%** 的分量超过 1.0 —— clamp 到 ±1 会砍掉大半动作，
  > 机器人 **40 步内必倒**（证据：σ 从 0.05 降到 1e-6，倒伏比例不变，**69% vs 70%**，
  > 说明倒伏与噪声无关，是 clamp 干的）。现在只有显式 `--action-clip` 才截断，默认 **0 = 不截断**。
  > **且**：σ=0.05 相对 `|a|~O(1..18)` 只是 **0.3%–5%** 的轻扰动；它的真实覆盖能力要用
  > **A(σ=0) vs A'(σ=0.05) 的轨迹差**直接量，不要靠推理（见 §13）。
- **L2 起点扰动**：episode 开始 `xy += N(0,0.1)`、`yaw += N(0,0.1)`、`dq += N(0,0.02)`；
  > ⚠️ 这个 **0.10 m 是拍的**。它应该等于**实测累积漂移的 p90**（用 `eval_drift_curve.py` 量），
  > 否则"A' 的覆盖"和"部署时的起点分布"对不上（见 §13 / §14）。
- **`--packed ha_cmd_exec`**（已修：原来硬编码旧 `ha_cmd`）；
- **不做 L3 链式回放**（WBC 重训本身解决漂移）；
- 每 clip **1 起点 × 4 seed**；体积 699 × 4 × ~900 ≈ **250 万行**；
- 产物：`runs/wm_real/exp_a_prime/<clip>/merged.pt` → 全局合并 `runs/wm_real/exp_ha_exec_transitions.pt`。

### 4.3.1 A' 的实际构成（R1，2026-11-03）

| 维度 | 值 |
|---|---|
| clip | **6 个**（有 token npy 的） |
| 起点 | 每 clip **1 个**：clip 的 row 0（`START_FRAME=0`） |
| seed（CLI） | **2 个**（`SEEDS="0 1"`） |
| env | **8 个/seed**（每个 env 独立 L1/L2 噪声） |
| 有效噪声轨迹 | 6 clip × 2 seed × 8 env = **96 条** |
| 长度 | **全长**（`A_STEPS=0` ⇒ 每条 clip 的 length；6 条合计 8,490 帧） |
| 行数 | 8,490 × 2 × 8 = **135,840 行** |

**⚠️ frames 局部/全局 bug（2026-11-03 发现并修）**：

- `isaac_collect_perturbed.py` 原先把 `frames` 记成 `start_frame+step`（**局部** 0..L-1），
  而 `compute_collect_reward` 按**全局 packed 行**查参考；
- 结果：**只有 DoubleDesk（global start=0）的 A' 奖励是对的**，其余 5 个 clip 的奖励查到了 DoubleDesk 的行；
- 影响：A' 的 **states/actions 是对的**（仿真数据）；错的是 **rewards**（WM 的 reward head）和 A' 奖励分析；
  **joint 的奖励来自 `bundle.reward`（真实奖励函数）**（`rollout.py:218`），**不读 WM reward head**，
  WM 的状态动力学也只用 states/actions ⇒ **R1 的 joint 效果仍有效**；
- 修复：`F.append(_frame_offset + start_frame + step)`，`_frame_offset = starts[ci]`；
  下一轮 A' 用修复版；本轮 R1 继续（reward head 本来也不被 joint 使用）。

### 4.4 A' 的奖励（必须与训练同口径）

```bash
python3 scripts/compute_collect_reward.py \
  --reward-mode huber_err --reward-delta-pos 0.05 --reward-delta-ori 0.05 \
  --reward-root-pos-std 0.05 --reward-root-ori-std 0.10 \
  --reward-body-pos-std 0.05 --reward-body-ori-std 0.10 \
  --packed runs/isaac_motion/ha_cmd_exec/packed.pt \
  --in <A' merged>.pt --out <A' with reward>.pt
```

- A' 的奖励**只用于训 WM**；joint FT 不直接用 A'；
- 物体奖励（`reward_from_objects.py`）只用于评测/分级成功。

### 4.5 WM 训练

```bash
python3 scripts/make_wm_mix.py --sim runs/wm_real/exp_ha_exec_transitions.pt \
  --bline <B 或留空> --ratio 0.25 --out runs/wm_real/exp_wm_mix.pt
python3 scripts/pretrain_wm.py --joint-ckpt runs/S1_handover/wm_903.pt \
  --transitions runs/wm_real/exp_wm_mix.pt --out runs/S1_handover/exp_wm_exec.pt \
  --steps 6000 --stage1-steps 3000 --roll-steps 1500 --roll-k 20
```

- WM 8.82M MLP，先不动；`reward head` 必须用**同一 huber_err/δ** 的 A' 重训；
- **G3a**：20 步预测误差（真实动作、held-out、p90）；**这是 mm 目标的地板**；
- **G3b**：VLA 闭环 20 步，WM vs 真 sim + 到 A' 管的距离。

### 4.6 一键整轮：`scripts/run_exp_round.sh`

```bash
# 冒烟（1 clip × 1 seed × 20 步；验证 A' -> merge -> reward）
ROUND=smoke WBC_CKPT=$PWD/runs/S1_handover/exp_WBC_exec.pt SEEDS="0" A_STEPS=20 NUM_ENVS=1 MAX_CLIPS=1 STAGES="aprime,merge,reward" bash scripts/run_exp_round.sh

# 正式 R1（有 token npy 的 clip × 2 seed × 整条；δ=5cm）
ROUND=r1 WBC_CKPT=$PWD/trackerlab-supermimic/logs/exp_wbc_exec_cmd/ckpt_027500.pt SEEDS="0 1" STAGES="" bash scripts/run_exp_round.sh
```

- 阶段：`aprime`（A'）→ `merge`（`a_prime.pt`）→ `reward`（`huber_err` + δ）→
  `wm`（`runs/S1_handover/exp_wm_<ROUND>.pt`）→ `joint`（`run_refstart_joint.sh`）→
  `eval`（`eval_drift_curve.py`）；
- 产物全部 `exp_` 前缀：`runs/wm_real/exp_round_<ROUND>/`、`runs/S3_loop/exp_ref_<ROUND>/`；
  **不碰旧 B 采集线**；
- `DRY_RUN=1` 只打印命令；`STAGES="aprime,merge,reward"` 可只跑前半段；
- `WBC_CKPT` 是唯一需要替换的旋钮（以后有更好的 WBC 直接换）。

**冒烟已验证（2026-11-03）**：A'（20 行）→ merge → reward（`a_prime_reward.pt`）全过。
冒烟修掉的 4 个坑：

1. Isaac `app.close()` 在本容器偶发挂起 → `isaac_collect_perturbed.py` 保存后 `os._exit(0)`；
2. 驱动以**输出文件**判成功（`seed*.pt` 存在即 OK，rc≠0 不算失败）；
3. `compute_collect_reward.py` 必须传 `--wbc-ckpt`（默认会加载另一个 WBC）；
4. 奖励/采集要带 `SUPERMIMIC_HA_ALIGN=1` + dex3 USD（`g1_29dof_with_dex3_rev_1_0.usd`）。

**joint / WM 也冒烟过了（2026-11-03 03:20–03:25）**：

- `run_refstart_joint.sh` **不再走 `s3_pipeline.sh joint`**（旧 pipeline 有 critic gate、双 arm main/ctrl、
  B 线 fresh_glob，会覆盖 REWARD_MODE/CRITIC_ON/ACT_WEIGHT）——已改成**直连 `launch_joint_ha_v2.sh`**；
- 2 步 joint：`reward=(huber_err,0.0)`、`act=(0.6,0.2)`、`critic=True/imagined`、`trust=(0.0,0.0)`、
  `start=reference`、`sampler=phase`，2 步 loss 正常；
- 1 步 joint + `WM_CKPT=exp_wm_smoke.pt`：`[wm] swapped from ...`（**之前 WM_CKPT 是空转**，已接通 `--wm-ckpt`）；
- `pretrain_wm.py` 增加 **Isaac backend**（原来只认 mjlab；现在 `is_isaac_checkpoint` → `load_isaac_bundle`）；
  `SUPERMIMIC_HA_ALIGN=1` + dex3 USD → `[AUDIT][spec] robot_spec 对齐 PASS`；
- `deps_env.sh`：py3.11 下**前置 `third_party/av_py311`**（`gr00t_deps/av` 是 3.12 构建，会 `ImportError`）；
- `run_exp_round.sh`：`STAGES=""` 视为**全跑**（之前空会跳过所有阶段）。

**WBC 选择（2026-11-03 最新）**：

| ckpt | 数据 | token | G1 | 用途 |
|---|---|---|---|---|
| `runs/S1_handover/exp_WBC_exec_haonly_best.pt`（HA-only 28k） | ha_cmd_exec x20 | `ha_cmd` | root 0.182 m，1472 步摔 → FAIL | **✅ 选定**（2026-11-03 起 A'/奖励/WM/joint 全用它） |
| `trackerlab-supermimic/logs/exp_wbc_exec_cmd/ckpt_027500.pt`（two-lib，混合） | bones+ha_exec | `ha_cmd` | 本机复核 root 0.400 m，1436 步摔 → FAIL | ❌ 弃用（比 HA-only 差 2.2×） |
| `runs/S1_handover/exp_WBC_exec.pt`（28k） | — | `None`（错） | 1.526 m | 只用于管线冒烟 |

- **G1 必须带 `TOKEN_LIB=ha_cmd`**（`p1c_g1_replay.sh` 默认 `TOKEN_LIB=$SM_PACKED`，two-lib 会错）；
- A'/WM/joint **三处同源**；换 WBC 必须重跑 A'；
- `ckpt_8000` 不是 WBC（是 **VLA** 的 checkpoint-8000）；`haonly_long` 没有 8k ckpt（28k→43.5k+）。

---

## 5. WM refresh 的启动时机（回答"是不是第二轮才开始"）

**同意：R1 不刷，R2 起再刷。**

| 轮 | B 采集 | WM refresh | 理由 |
|---|---|---|---|
| **R1** | **0**（可选结尾 5–10 集只做诊断） | **不刷** | 隔离变量：先看"纯执行梯度 + 加权起点"能把误差压到多少；量 WM 地板 |
| **R2** | **10–20 集** | ✅ 刷一次 | 策略已变，A' 不含新分布；用 B 补 25% |
| **R3** | 20–30 集 | ✅ | 同上 |
| **R4** | 20–30 集 | ✅ | 同上 |
| **R5**（条件） | 30 集 | ✅ | 只有地板 <1mm 才开 |

**B 的作用（重要）**：B **只用于 WM refresh**；**策略训练起点全程仍是参考**（加权）。
B 起点（DAgger）是 R2 的**可选恢复臂**，默认不开；只有 R1 闭环漂移曲线显示累积时才加 20%，且是**混在参考里**，不是替换参考。

**refresh 的正确姿势**：

- 不是"拿同一份 A' 再训一遍"（分布没变，没用）；
- 是 **A' + 当前策略的 B（10–20 集）**，B 占 25%；
- `wm_refresh` 失败不能挡训练；触发条件：G3b 的 20 步误差 > 阈值（如 5 mm 或当前 δ 的 2 倍）。

---

## 6. 多轮完整架构

### 6.1 不变量（所有轮）

| 项 | 值 |
|---|---|
| 奖励形状 | Huber-on-error + split |
| 朝向 split | tilt 1.0 / yaw 0.2 / yaw_rate 0.3 |
| 总损失 | `0.3·(-Σr) + 0.2·action trust` + critic bootstrap |
| token trust | 0 |
| critic | on |
| γ / H / λ_r | 1 / 20 / 0.3 |
| 数据 | A' + L1/L2（R2+ 再加 B 25%） |
| 起点 | 参考起点 + phase/task 加权（51,200 个，`STEPS=400`、`ACCUM=16`） |
| **只变** | **δ**、B 采集、WM refresh |

### 6.2 R1（完整步骤）

**目标**：把每 20 步漂移从 2.33 cm 压到 **2–3 cm**（或更好），打通全流程。

| # | 步骤 | 命令/产物 |
|---|---|---|
| 1 | G1 过：修正版 WBC | `exp_WBC_exec_cmd.pt`；`npy` + `SM_PACKED=ha_cmd_exec`；<0.1 m、0 摔 |
| 2 | 实现 `huber_err` + `split_huber` + `--reward-delta-*` | 见 §2.5 |
| 3 | 生成 A'（先 2 seed） | `run_a_prime_seeds.sh SEEDS="0 1"`；`--packed ha_cmd_exec` |
| 4 | 算 A' 奖励 | `compute_collect_reward.py --reward-mode huber_err --reward-delta-pos 0.05 ...` |
| 5 | 训 WM | `pretrain_wm.py`；G3a 量地板 |
| 6 | 联合微调 | `run_refstart_joint.sh`：`STEPS=400 ACCUM=16`、`REWARD_MODE=huber_err`、`DELTA_POS=0.05`、`ROOT_ORI_MODE=split_huber`、`CRITIC_ON=1`、`TRUST_WEIGHT=0`、`ACT_WEIGHT=0.2`、`START_SAMPLER=phase`、`START_AUDIT=1` |
| 7 | 训练中 eval | `fixed_eval`（uniform 8 batch）+ 关键段子集；看 `imag_state`/`imag_delta` |
| 8 | 闭环评测 | 14 集闭环 + `eval_drift_curve.py`；分级成功 |
| 9 | 覆盖审计 | `start_coverage.json`：unique frames/clips/per-task |
| 10 | 结论 | 是否进 R2；量 WM 地板、真实漂移、成功率 |

**R1 不采 B、不刷 WM。**

### 6.3 R2（只写改动）

- **δ：5 cm → 2 cm**（pos/ori 同步）；
- **起点**：默认仍 **100% 参考**（phase/task 加权）；
  若 R1 闭环漂移曲线**线性涨** → 加 **20% B 起点**（DAgger，`START_FROM=mix`，可选臂），**不替换参考**；
- **B：10–20 集**（当前 VLA 闭环，`ha_cmd_exec` 参考），**只喂 WM refresh**；
- **WM refresh**：A' + B 25%，刷一次；
- 联合微调其余不变；
- 目标：**1 cm**；
- 触发：若 R1 的闭环漂移曲线仍线性涨 → 加 20% 漂移起点（DAgger）作为 R2 的可选臂。

### 6.4 R3（只写改动）

- **δ：2 cm → 5 mm**；
- **B：20–30 集**；refresh；
- 若 G3a 地板 >5 mm → **先升级 WM**（更多 A' seed / 更大 MLP / 更长训练），否则 R3 意义有限；
- 目标：**5 mm**。

### 6.5 R4（只写改动）

- **δ：5 mm → 2 mm**；
- B 20–30；refresh；
- 目标：**2 mm**；
- 若地板 >2 mm → 停在这里，不再往下。

### 6.6 R5（条件轮）

- **δ：2 mm → 1 mm**；
- **只有 G3a 的 20 步 p90 < 1 mm** 才开；
- B 30；refresh；目标 **1 mm（0.1 cm）**；
- 若地板不够：改用真实仿真梯度（不用 WM）或更小 H（5–10 步）——这是**另一个项目**，不在本方案内。

---

## 7. 实施状态（2026-11-04 逐项核对代码后重写）

> 原文这张表叫"当前应改未改清单（全部）"。**逐项核对代码后结论是：P0 全部已落地。**
> 下面分成"已实现 / 未实现 / 已修但原文没写"三组，每行附代码位置。
> **只有 §7.2 那四项是真的待办。**

### 7.1 已实现（原文标 ❌ 或未提，但代码已做）

| # | 项 | 位置 / 证据 |
|---|---|---|
| 1 | `REWARD_MODE="huber_err"` | `isaac_joint.py:583-599` + `_huber_on_error` (`:73`) |
| 2 | `split_huber`（tilt 1.0 / yaw 0.2 / yaw_rate 0.3） | `isaac_joint.py:641-661`（含"只允许配 `huber_err`"的 guard） |
| 3 | `--reward-delta-pos/--reward-delta-ori` | `train_joint_v2.py:404,406` → `:589-590` |
| 4 | `ROOT_ORI_MODE` 的 `split_huber` | `train_joint_v2.py:427`（choices）+ `run_refstart_joint.sh:36`（默认） |
| 5 | refstart 默认全套 | `run_refstart_joint.sh:35-48` |
| 6 | A' 奖励同口径（`huber_err` + δ） | `compute_collect_reward.py:38,41` |
| 9 | A' 的 `--packed ha_cmd_exec` | `isaac_collect_perturbed.py:56-57`（默认就是 `ha_cmd_exec`） |
| 10 | L1/L2/seed | 同文件 `:47-54`；`exec_steps=perturb_hold=20` **同相**（关键对齐） |
| 11 | 起点加权 + 覆盖审计 | `data.py` / `train_joint_v2.py` |
| 12 | `SM_PACKED`/`START_FROM`/`library_kind`/`MOTION_PACKED` | 已改 |
| 13 | `fixed_eval` | `train_v2.py:682` |
| 14 | `eval_drift_curve.py` | `scripts/eval_drift_curve.py` 存在 |
| 16 | WM refresh（**部分**） | `v3_step.py:941` + `train_joint_v2.py:308-311`（`--wm-refresh-every/-steps/-lr`）；**缺**"失败不挡训练"与"G3b 阈值触发"，现在只有固定周期 |
| 19-24 | 其余 ✅ 项 | 与原文一致 |

### 7.2 未实现（**这才是真正的待办**）

| # | 项 | 现状 | 阻塞什么 |
|---|---|---|---|
| 7 | **G1 没跑过（本文档声称的 G1 门）** | `runs/S1_handover/exp_WBC_exec_cmd.pt` **不存在**（2 卡 27.5k 被 Killed）；`runs/exp_g1/` **目录都不存在**。仓库里有 64 个 `runs/exp_g1_*` 子目录（旧 WBC 的 G1 痕迹），但**没有一个是 `exp_WBC_exec_cmd.pt` 的 G1** | **§4 整条 A'→WM→joint 都建立在一个未过 G1 的执行器上** |
| 8 | `exp_WBC_exec_cmd.pt` 交付 | 同上 | 同上 |
| 15 | B 采集 10–20 集 + 25% 混入 | 机制在（`make_wm_mix.py --bline`），但 **R1 不采 B ⇒ 从未验证** | R2 的 refresh 没有实测依据 |
| 18 | `START_FROM=mix`（DAgger 起点，80% 参考 + 20% B） | **只有"全量替换"，没有"按比例混合"**：`--start-from-b` + `b_starts`（`train_v2.py:517,579,633`）一给就把**所有** batch 换成 B 起点（`_sample_from_b`），`--start-source` 也只有 `{reference,recorded}`（`train_joint_v2.py:263`） | drift curve 线性涨时，**只能二选一**（全参考 or 全 B），要么学不到纠偏、要么忘掉参考；`mix` 这一臂现在开不了 |

### 7.3 已修但原文没写（会误导复现）

| 项 | 位置 | 说明 |
|---|---|---|
| A' 的 `.clamp(-1,1)` **已删** | `isaac_collect_perturbed.py:270-276` | 见 §4.3 的警告框；保留原文会让复现者把机器人弄倒 |
| A' 的 `frames` 写全局 packed 行 | 同文件 `:287` | `_frame_offset + start_frame + step`（原局部 0..L-1 会让 starts>0 的 clip 奖励查错参考） |

---

## 8. 门（G0–G5）

| 门 | 判据 |
|---|---|
| G0 | `ha_cmd_exec` FK <3 cm（实测 2.4 mm）✅ |
| G1 | 命令 token 回放 root <0.1 m、0 摔、走完整条 |
| G2 | A' 帧覆盖 100%、动作分布合理、参考成功 >0（DoubleDesk 99/100）✅ |
| G3a | WM 20 步误差 p90（R1 <5 cm；R3 <5 mm；R5 <1 mm） |
| G3b | VLA 闭环 20 步：`e_wm` 小、`d_tube` 在管内 |
| G4 | 联合微调后 `imag_state`/关键段误差下降 |
| G5 | 闭环漂移曲线、分级成功、总漂移（R1 <50 cm；R3 <10 cm；R5 <1 cm） |

---

## 9. 风险与回退

| 风险 | 处理 |
|---|---|
| G1 不过 | 先查 `SM_TOKEN_LIB_TRAIN` 是否开；再查 WBC 训练库/obs；最后换 `bones_seed_retrain_1025b`（relative_ori99） |
| WM 地板 > 目标 δ | 停在该轮，先升级 WM（更多 A' seed/refresh/更大 MLP），不要硬跑下一轮 |
| 1 mm 目标不可达 | R5 不开；改为真实仿真梯度或 H≤10 |
| B 采集太贵 | 每轮只 10–30 集，专喂 WM refresh，不用于策略训练 |
| yaw 不可学 | split 已降权 0.2；若仍主导，进一步降到 0.1 或从奖励里去掉 |
| critic 不稳定 | warm start + 小 LR；每轮 δ 变化后重拟合 |
| 策略被 trust 绑死 | `ACT_WEIGHT=0.2` 已很弱；看 `act_rms` 是否贴 0.6 → 提 eps 或降 weight |
| 旧产物被覆盖 | 所有新路径 `exp_` 前缀；旧线只读 |

---

## 9.1 R1 启动状态（2026-11-03 03:26）

- 命令：`ROUND=r1 WBC_CKPT=.../ckpt_027500.pt SEEDS="0 1" JOINT_STEPS=400 ACCUM=16 STAGES="" bash scripts/run_exp_round.sh`
- 日志：`logs/exp_round_r1.log`（第一次）；`logs/exp_round_r1_resume.log`（续跑）
- **A' + 奖励已完成**：`a_prime_reward.pt` = **135,840 行**（6 clip × 2 seed × 8 env）；
- **第一次在 WM 阶段失败**：`make_wm_mix.py` 强制 `--bline`，pipeline 没传；且 `run()` 不 fail-fast，
  导致 joint 找不到 `exp_wm_r1.pt` 才报错。**已修**：R1 无 B 直接用 A'（`--transitions a_prime_reward.pt
  --stage1-steps 0 --roll-steps 0`）+ `run()` 失败即停；从 `STAGES="wm,joint"` 续跑成功。
- **R1 已在 step ~336/400 手动停掉**：6-clip 只是子集冒烟；step 0→50 的 `imag_state_vla`
  0.8474→0.8472（几乎没动），**不能当方案效果**；4 卡的 699 全量才是。
  > **2026-11-04 补充（重要）**：这个"几乎没动"有**第二种解释，而且可能是主因**——
  > 如果 20 步漂移只有 2.33 cm，那么在漂移状态上"标签动作"仍近似最优，
  > **outcome 信号本来就几乎没有可学的方向**。所以不能只归因为"6 clip 太少"。
  > 判据见 §14.1 的第一个实验（`check_label_optimality`）。
- **699 个 token npy 已补齐**（`runs/exp_tokens/` 699），4 卡用 `scripts/run_a_prime_4gpu.sh`。
- **当前只覆盖 6 个有 token npy 的 clip**（不是 699）；A'/WM 是子集，joint 的参考起点仍是全库 634k（加权）。
- 完整 699 需要先生成 699 个 token npy；本轮的定位是**全链路实跑**，效果只作参考。
- 2 卡的 two-lib 修正版在 27.5k 被 Killed（28k 未交付）；`ckpt_027500.pt` 是当前最好的 two-lib WBC；
  **G1 还没在这个 ckpt 上跑过**。

## 10. 一句话

> **R1：不采 B、不刷 WM，用 huber_err δ=5cm + split + critic + 0.2 action trust，把漂移压到 2–3 cm；
> R2–R4：δ 逐步减到 2 mm，每轮 10–30 集 B + 一次 WM refresh；
> R5 只在 WM 20 步地板 <1 mm 时开；奖励形状全程不变，只退火 δ。**

---

# 附录 A（2026-11-04 新增）：本轮补齐的口径与结论

> 以下五节是 2026-11-03 版本里**完全没有**的内容。它们不是"再改一版方案"，
> 而是把几个一直在讨论、但文档没写死的问题定下来，避免后面重复绕。

## 11. Token 空间的 FSQ 量化（新口径，默认 `none`）

**三条事实**（都可核）：

1. token 是**整数格**：`q = n/16`，`n ∈ [-16, 15]`，值域 `[-1.0, 0.9375]`，每维 **32 级**
   （release encoder 导出图，复刻见 `sonic_encoder.py:151`）；
2. 执行器**只在格点上训练过**（HA 标签 `g1_motion_token_64` 是 FSQ 输出；实测 `decoder_obs[:, :64]`
   的 `on_grid` 比例 = **1.0**）；
3. VLA 输出是**连续值**（`codec.decode` 是纯仿射，不 clamp、不量化）。

⇒ `∂a/∂token` 在格点之间是**外推**。为此新增开关（**训练 + 部署两侧都已接**）：

| `--token-quant` | 前向 | 反向 | 用途 |
|---|---|---|---|
| `none`（**默认**） | 连续 | 正常 | 当前执行器实测可用，就用这个 |
| `ste` | 格点 | 恒等（`∂τ_use/∂τ ≡ 1`） | 执行器**不可重训**（如 release decoder）或外推不可信时 |
| `hard` | 格点 | 零 | 消融：应当"训不动" |

**实测结论：两个执行器的亚格外推都极好 ⇒ 默认 `none` 是对的，`ste` 暂时不需要。**

| 执行器 | cos @ 半格 | rel @ 半格 | 判定 |
|---|---:|---:|---|
| SuperMimic WBC（8.83M） | 1.0000 | **0.83%** | OK |
| SONIC release decoder（10.18M） | 1.0000 | **0.49%** | OK |

- 判据工具：`scripts/check_token_jacobian.py`（解析 Jacobian vs 跨格差分，多 δ 扫描 + 同帧配对）；
- 口径全文：`docs/TOKEN_QUANT_STE_20261104.md`；
- 部署校验：`vla_meta.json` 的 `token_quant` 与 `--token-quant` **逐项比对**，
  不一致直接退出（放行开关 `ALLOW_TOKEN_QUANT_MISMATCH=1`），两个 server 共用同一份实现
  （`quant.check_meta_mode`）。

## 12. 执行器等价性 —— **本文档最大的风险项**

L2 用的是**重训的 SuperMimic WBC**，而 HA 数据的原始执行器是 **release SONIC decoder**：

| | 动作偏差 | 证据 |
|---|---|---|
| WBC vs 旧库的命令流 | ~**1.1 RMS** | `v3_step.py:709-711` |
| 新旧库关节差 | `mean\|Δq\| = 0.336 rad` | §1 |
| label-vs-port token | MSE 0.074 / corr 0.157 | `S0.4` |

**这件事现在有解了**：release decoder 的**可微 torch 版已就绪** —— `diff_vla_wbc/sonic_decoder.py`
（ONNX 逐算子复刻，与 onnxruntime 差 **3.3e-6**；与 HA `decoder_target_action` 的逐关节仿射残差
**5e-6 度**）。

⇒ **L1' 路线（VLA → token → SONIC decoder → WM → 奖励 → 反传）** 的阻塞点已逐项拆掉：

| 阻塞点 | 状态 |
|---|---|
| 可微 torch 版 | ✅ `diff_vla_wbc/sonic_decoder.py`（与 ORT 差 3.3e-6；与录制动作差 5e-6 度） |
| **994 维输入布局** | ✅ **已逐位反解并验证（`max|Δ| = 0.0`）**，见下 |
| batch=1 前向 | ✅ **修掉一个静默 bug**：opset 13 的 `Squeeze` 把 axes 放在**第二个输入**，旧实现取不到就 `squeeze()` 删掉所有 1 维 ⇒ **batch=1 时输出从 (1,29) 变 (29,)**；部署侧正是 batch=1。已修 + 加回归测试 |
| 重采 A' + 重训 WM | ⬜ 工程活（换执行器必须做，见 §4.3） |

**994 布局（逐位验证，`decoder_obs` 完全可重建）**：

```
[  0 :  64)  token                      （FSQ 1/16 格，[-1, 0.9375]）
[ 64 :  94)  his_base_angular_velocity  10×3   由旧到新 [t-10 … t-1]
[ 94 : 384)  his_body_joint_positions   10×29
[384 : 674)  his_body_joint_velocities  10×29
[674 : 964)  his_last_actions           10×29  = decoder_raw_action[t-10 : t]  ★
[964 : 994)  his_gravity_dir            10×3
```

- **块优先（block-major）**，不是帧优先；
- 每块内部**帧优先、由旧到新、不含当前帧**；
- `his_last_actions` 是**解码器自己的原始输出**（不是执行后的动作、不是别的归一化空间）——
  这一块是六个里**最敏感**的（清零 → 输出变 75%，随机化 → 106%），所以口径必须对，
  不能猜。rollout 时把上一步输出滚进窗口即可。

> 复现/自检：`diff_vla_wbc.sonic_decoder.{build_obs_994, split_obs_994,
> supermimic_obs_to_blocks}` + `tests/test_sonic_decoder.py`（含 batch=1 回归）。

> **在此之前，任何 joint 结果都无法归因**——分不清漂移里多少是 VLA 的错、多少是执行器不等价的错。
> 这是本文档（原版）最大的空白。

## 13. 三个已关闭的疑虑（别再绕）

| 疑虑 | 结论 | 依据 |
|---|---|---|
| **A' 覆盖不到部署的漂移** | **单窗口够**（2.33 cm vs L2 覆盖 10 cm，**4× 余量**；且 WM 只被问 20 步）；**累积偏差覆盖不到**，但那是"训练起点"问题、不是采集问题 | §4.3 / §6.2 |
| **相位错配**（`lag +8 帧`） | **不成立**：参考已换成执行轨迹（`ha_cmd_exec`），`(s[t], τ[t], s[t+1])` 同一次录制、自洽 | `sonic_frontend.py` |
| **加噪被"平均"** | **不成立**：噪声是**在仿真器里 replay** 的，`(s, a_noisy, s')` 逐条真值一一匹配，且 `last_a`/obs 同步用加噪值 | `isaac_collect_perturbed.py:269,277-288` |

> 剩下的"平均"只来自**部分可观**（71 维 state 不是 Markov），由 **G3a** 量着，与噪声无关。

## 14. 修正后的行动顺序（R1 之前）

原文顺序是「实现 huber_err → 采 A' → 训 WM → joint」。**修正两条：**

### 14.1 先量 drift curve，再决定要不要动数据

原 §6.3 把"要不要 DAgger 起点"放到 R2，但它的**判据是 R1 的曲线**。

```
drift curve 平坦 / 有界 → 不用扩覆盖、不用 B、不用 mix
                          （A' 以参考为中心 ⇒ 策略越好，老 WM 越准，这是自增强的好方向）
drift curve 线性涨       → 只降"每窗口误差"无效（45 个窗口的乘积还是 N×）
                          ⇒ 必须动**起点**（放大 L2 / START_FROM=mix），不是动 δ
```

**注意 `START_FROM=mix` 目前没实现**（§7.2 第 18 项），所以线性涨的情况下这一臂现在开不了。

### 14.2 `ACT_EPS` 已从 0.6 收回 **0.30**（2026-11-04 改）

- A' 实际采到的动作半径是 **0.05**，而 trust region 允许 **0.6**（12×）；
- 0.6 是后来无依据放松的：**标定记录（commit `9e8c4cf`）是
  "base policy's own deviation 0.21, eps=0.30 (~1.4x), frac_out 0.15–0.30"**；
- 目标是压到 mm 级，这个球必须**真的咬住**，否则它既无约束力、又掩盖真实违约。

**已按用户口径收紧到 `ACT_EPS=0.10`、`ACT_WEIGHT=1.0`**（`run_refstart_joint.sh:55-56`、
`launch_joint_ha_v2.sh:140,143`）。

理由：起点永远从参考出发 ⇒ 20 tick 内的偏差本来就是几 cm ⇒ 信任域应当**远小于** base policy
自身的偏差 0.21（而 0.21 里还有一大半是 FM 采样噪声）。

**预期与观察点**：`frac_out≈1.0` 是**正常的**（它本来就该一直咬住）。真正要盯的是
`act/rms` —— 如果它长期贴在 0.10 不动（被钉死），说明奖励没有活动空间，此时应降 `ACT_WEIGHT`
或把 eps 略放，而不是继续加权重。

### 14.3 两个待做的判据实验（都比继续讨论便宜）

| 实验 | 回答什么 | 成本 |
|---|---|---|
| **`check_label_optimality`** | **漂移状态下"标签动作"是否仍最优？** 这是"joint FT 到底有没有意义"的唯一判据 | 半天，现成 A' + 现成 WM |
| **WM 均值回归偏置** | 从 A' 的漂移状态出发、用**同一动作序列**滚 20 步：WM 预测的误差衰减是否快于真实 | 半天，`eval_wm_multistep.py` 已有骨架 |

> 第一个实验若显示"误差主要由状态偏离决定、与选哪个动作几乎无关"，**joint FT 的增量天然很小**——
> 这与 §9.1 的 `0.8474→0.8472` 是**一致**的，不只是"样本少"。

## 15. 2026-11-04 第二批修复（链路审计后，对应 1–4 号问题）

| # | 问题 | 修法 | 位置 |
|---|---|---|---|
| **1** | 我一度把 `CRITIC_SOURCE` 默认改成 `real`（依据 `9e8c4cf` 的旧实测）——**这是错的，已回退**：`METHOD_LOCK_H20_20261101` 明确锁定 **imagined GAE**（无 prefit、与策略在线共同更新 ⇒ 自洽，不依赖任何离线产物），Nov-3 的 plan 与 refstart 都继承了它 | 保持 `imagined` 为默认，只把硬编码改成**可覆盖**（`CRITIC_SOURCE=real` 作为可选臂） | `run_refstart_joint.sh:86` |
| **1b** | （仅当显式走 `real` 臂时）`real_returns.pt` 的口径检查是**死代码**（写在 `raise` 之后，永不执行）⇒ 现有的 `runs/wm_real/real_returns.pt`（`reward_mode='saturating'`, `vel_scale=1.0`）一直没被拦下 | 把检查移到加载之后，逐项比对 `reward_mode / vel_scale / δ_pos / δ_ori / root_ori_mode`，不符直接退出并打印重建命令 | `train_joint_v2.py`（critic 加载块） |
| **1c** | `build_real_returns.py` 只能产 `saturating/linear` 回报，做不出 `huber_err` 目标 | 加 `--reward-mode huber_err` + `--reward-delta-pos/--reward-delta-ori/--root-ori-mode`，并把口径写进 `meta` | `build_real_returns.py` |
| **2** | `ADV_TAU` 在 `ADV_BASELINE=none` 下是死参数 | **只加了能力、没改默认**：新增 `label` 基线（用**标签 token**跑同一段想象，不需要 B dump）。默认**仍是 `none`**（plan 口径）。要用就 `ADV_BASELINE=label ADV_TAU=0.02`，代价是每步多一次 no_grad rollout；注意 pathwise 下减常数不改梯度 ⇒ 收益只来自 tanh 的重加权 | `v3_step.py`、`train_joint_v2.py` |
| **3** | L1 噪声是**绝对** σ=0.05，但 `action_scale` 实测 **0.167–0.750（4.5×）**，同一 σ 在最大关节上是 2.15°、在最小关节上只有 0.48° | 新增 `--action-noise-rad`（推荐 0.014 ≈ 旧口径中位等效）：`δ_j = N(0,(rad/scale_j)²)`，噪声按**角度**统一 | `isaac_collect_perturbed.py`、`run_a_prime_seeds.sh` |
| **4** | `perturb_hold == exec_steps == 20` **同相** ⇒ 噪声永远落在 chunk 边界，WM 会学"扰动只出现在 t≡0(mod 20)"的伪相关 | 改成"下一次重采 = 当前步 + hold"，并加 `--perturb-hold-jitter`（默认 5 ⇒ hold∈{15..25}）；不再用 `step % hold` | 同上 |

**这批修复改变了 A' 的生成口径**，所以：

> ⚠️ **已有的 A'（135,840 行）与 WM 必须按新口径重采/重训**，否则新老口径混在一份 WM 里。
> 旧 A' 仍可用于对照，但不能再和新 A' 合并训练。

**还没做（按优先级）**：

| # | 项 | 为什么 |
|---|---|---|
| 5 | `REWARD_VEL_SCALE=0` ⇒ 关节速度没有奖励压力 | 需要判读（可用差分重建速度项） |
| 6 | `L2=0.10` 应等于实测漂移 p90 | 等 `eval_drift_curve.py` |
| — | ~~root 权重~~ | **撤回**：WM 输出含 root 位置、误差可反传；只是杠杆可能是 O(dt) 量级，需先量 |

---

## 16. 本次修订改动清单（第一批）

| # | 改动 | 影响 |
|---|---|---|
| 1 | §2.5 从"应改未改"改为"全部已落地"（附行号） | 复现者不会再去找不存在的工作 |
| 2 | §7 重写为「已实现 / 未实现 / 已修但原文没写」 | 真正的待办只剩 4 项（G1、`exp_WBC_exec_cmd.pt`、B 采集、`START_FROM=mix`） |
| 3 | **§4.3 删掉 `(a+δ).clamp(-1,1)`** | 这是会**直接把机器人弄倒**的错误 |
| 4 | §4.3 给 L2 的 0.10 加了"应按实测 p90 定"的警告 | 覆盖与起点分布要对齐 |
| 5 | §9.1 补了 `imag_state` 不动的第二种解释 | 防止把"没信号"误判成"样本少" |
| 6 | 新增 §11 token 量化 | 训练 + 部署两侧已接，默认 `none`（有实测支撑） |
| 7 | 新增 §12 执行器等价性 | 标为最大风险项；可微 SONIC decoder 已就绪 |
| 8 | 新增 §13 三个已关闭疑虑 | 防止再绕覆盖/相位/平均 |
| 9 | 新增 §14 行动顺序 | 先 drift curve，再动数据；`ACT_EPS` 该收小 |
| 10 | 新增 §16（本节） | — |
| 11 | 新增 §15（第二批修复：critic/adv/L1/hold） | 详见 §15 |
