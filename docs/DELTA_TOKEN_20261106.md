# 命令层残差（delta token）：`token' = token + g(state)`

> 2026-11-06。与 `LoRA 微调` / `state_extra` / `反常量项` **并行**存在，默认关闭，
> 用环境变量选路。本文自包含：动机、架构、部署口径、命令行、判据、风险。

---

## 1. 为什么做这条（四轮实测之后才轮到它）

同一天的三个「给更多信息，指望模型自己学」的路，全部没有把状态相关性推上去。
判据是**落地探针的 `残差/均值`（L2 口径，阈值 0.5）**：

| 臂 | 改了什么 | ‖Δa‖ | **残差/均值** | **残差/噪声** | cos vs v13 |
|---|---|---:|---:|---:|---:|
| `v13` | action_target + LoRA | 0.132 | **0.098** | **3.84** | 1.00 |
| `p1b` | 关 AT + 反常量项(w=0.05) + 分母下限 | 0.132 | 0.164 | ~4.0 | 0.956 |
| `p1c` | p1b **+ `state_extra=rotlocal6` 全强度** | 0.122 | **0.170** | **3.76** | 0.952 |

**关键读数不是「残差/均值只涨了 1.7 倍」，而是「残差/噪声一点没涨」（3.84→3.76）。**
说明涨出来的那部分「残差」是**采样噪声按比例放大**，不是状态依赖。

⇒ **p1c 已经排除了「模型看不见偏差」这个解释**：真实姿态偏差以 scale=1.0 全强度
塞进了 VLA 输入，结果完全没变。**卡点在结构，不在信息量。**

---

## 2. 架构

```
x_raw(78) ──► token_base = x_raw[..., :64]            （冻结的 VLA，no_grad）
                        │
   [token_base(64), rot6d(6), dof_vel(29)] ──► g ──► Δtoken(64)
                        │
              token = token_base + Δtoken             ★ 在进 WBC 之前
                        │
              WBC(token) → action → WM → reward → loss
                        │
                 梯度只回到 g（VLA 冻结）
```

- `g` = `99 → 1024 → 1024 → 64` 的 MLP，**1,217,600 参数**（LoRA 33.5M 的 **3.6%**），末层**极小初始化**（`std=4e-4`）⇒ 初始 `‖Δtoken‖ ≈ 0.0095`，而 `‖base token‖ ≈ 7.98`（**0.12%**）⇒ 等价于「从底模出发」。
- ⚠️ **为什么不用严格零初始化**：链式法则 `dL/dW_前层 ∝ W_out`，严格 0 会让前两层在第 0 步**完全没有梯度**（实测 `|grad| = 0.000e+00`，只有输出层在动）。4e-4 的 std 让三层从第 0 步一起可训，初始扰动仍在 0.1% 量级。
- 把 `token_base` 也当输入：**token 本身就是参考运动的 encoder latent，天然带相位**，
  同一个偏差在支撑相/摆动相该补的东西不同。
- 与 `state_extra` 的区别：那个是「改 VLA 的输入分布，指望 LoRA 学」；这里是
  **把状态相关写进结构** —— `g` 的输入就是偏差，同一个 token 在不同偏差下必然给不同命令。

---

## 3. 部署口径（**第三方客户端零改动**）

线上 64 维 state 的真实组成（`action_provider_sonic.py:1351-1354`）：

```
state64 = [ root_rot6d (0:6) | dof_pos_29 (6:35) | dof_vel_29 (35:64) ]
```

`g` 只用其中两块：

| 特征 | 来源 | 说明 |
|---|---|---|
| `rot6d` (6) | `state64[0:6]` | heading-canonical，回合起始为 identity；与训练侧 `extra6_for("rotlocal6")` **同一函数** |
| `dof_vel` (29) | `state64[35:64]` | **当前 state46 根本没用这 29 维** ⇒ 白送的额外信号 |

服务器 `scripts/ha_vla_latent64_server.py` 的改动：
- 从 ckpt 读 `delta_token_state` / `delta_token_meta`（含 `hidden/depth/cap`）；
- 在解码前 `chunk64 += apply_norm_cap(g(token0, feats), cap)`；
- ckpt 里没有就完全跳过（老 ckpt 逐位不变）。

**训练/部署口径的唯一防线**：软上限 `apply_norm_cap()` 在两端调**同一个函数**
（`diff_vla_wbc/delta_token.py`）。训练时它是损失的一部分，部署不跟着做就是漂移。

---

## 4. 命令行选路（默认全关，现有行为逐位不变）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

# ① 现在这条（LoRA + 反常量项 + state_extra）—— 什么都不用加
NPROC=2 DDP=1 DDP_GPUS=0,1 BATCH=16 ACCUM=3 STEPS=300 RUN=joint_p1b \
  WM_CKPT=runs/a_prime_sonic_exc/wm_gain2.pt TERM_OFF=1 \
  VAR_WEIGHT=0.05 ACT_WEIGHT=0.3 STATE_EXTRA=rotlocal6 \
  bash scripts/run_p1_round_b.sh

# ② 命令层残差（推荐口径：VLA 全冻，只训 g + critic）
NPROC=2 DDP=1 DDP_GPUS=0,1 BATCH=16 ACCUM=3 STEPS=300 RUN=joint_p1d \
  WM_CKPT=runs/a_prime_sonic_exc/wm_gain2.pt TERM_OFF=1 \
  DELTA_TOKEN=1 FREEZE_VLA=1 DELTA_TOKEN_HIDDEN=512 DELTA_TOKEN_CAP=0.6 \
  VAR_WEIGHT=0 ACT_WEIGHT=0.3 STATE_EXTRA=none \
  bash scripts/run_p1_round_b.sh

# ③ 先核对命令行（不占 GPU、不建 run 目录）
DRY_RUN=1 DELTA_TOKEN=1 FREEZE_VLA=1 bash scripts/run_p1_round_b.sh
```

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `DELTA_TOKEN` | `0` | 1 = 开启命令层残差 |
| `DELTA_TOKEN_HIDDEN` | `1024` | 99→h→h→64；1024×2 = **1,217,600** 参数 |
| `DELTA_TOKEN_DEPTH` | `2` | |
| `DELTA_TOKEN_INIT_STD` | `4e-4` | 末层初始化 std；0 = 严格零初始化（前两层无梯度，仅 A/B 用） |
| `DELTA_TOKEN_CAP` | `0` | `‖Δtoken‖` 软上限（tanh 收缩，保留梯度）；建议 `0.6` |
| `FREEZE_VLA` | `0` | 1 = `lora_rank` 强制 0（VLA 全冻），只训 `g` + critic |

**为什么 `FREEZE_VLA=1` 是推荐口径**：LoRA 在闭环上是负资产（v13 30.8% vs base 58.3%，
26 集）；冻掉它，归因干净、每步快数倍、也不会把底模的 token 带偏。

---

## 5. 判据（与之前一致，不新增口径）

| 门 | 指标 | 阈值 |
|---|---|---|
| G2 | `act/rmse` | **0.05–0.15** |
| G3 | `critic/v_ev` | **> 0.90** |
| G4 | `critic/v_adv_H_raw > 0` 占比 | **< 80%** |
| 新 | `delta_token/norm` | 从 0 单调上去（>0 才说明这条路在动） |
| **G5** | 落地探针 **`残差/均值`（L2）** | **> 0.5**（v13 = 0.098 / p1b = 0.164 / p1c = 0.170） |
| G5 | 落地探针 ‖Δa‖ | **0.10–0.20** |
| G5 | `cos(v13, 新臂)` | 应**明显低于 0.95**（否则还是那根常向量） |

探针（必须**同批带 v13 标定**，否则口径会算错）：

```bash
python3 scripts/probe_ft_landing.py --dump runs/closedloop_r3/base_s0 \
  --arms base= v13=runs/joint_v13/vla_000100.pt dt=runs/joint_p1d/vla_final.pt \
  --device cuda:0 --n-noise 3 --max-req 24
```

⚠️ 训练侧 `action_var/ratio` 是 **RMS 分母**，探针是 **L2 分母**，差 **sqrt(29)=5.39 倍**。
不要把训练侧的 0.5 当探针的 0.5 用。

---

## 6. 已知风险 / 检查清单

| # | 风险 | 现象 | 处置 |
|---|---|---|---|
| 1 | **token 出流形** | `act/rmse` 冲过 0.15，或闭环开始抽风 | 调小 `DELTA_TOKEN_CAP`（0.6→0.3），或降 `g` 的 lr |
| 2 | **容量不够** | `delta_token/norm` 在 step 50 前平台、奖励不降 | `DELTA_TOKEN_HIDDEN=2048`（4.85M） |
| 3 | **不是容量问题** | norm 平稳升但探针残差/均值仍 <0.2 | 别再加大；上**历史特征**（最近 2–3 帧 rot6d+dof_vel → 189 维） |
| 4 | **`delta_token/norm` 全程 0** | 零初始化后梯度没通 | 查 `g` 是否进了 optimizer、`delta_token` 是否真的进 `_DT_ARG`（用 `DRY_RUN=1` 核对） |
| 5 | **冻结后学不动** | `FREEZE_VLA=1` 时 loss 不动 | 确认 `delta_token` 的 lr = `fm_lr`（当前如此）；必要时单独放大 |

## 7. 文件索引

| 用途 | 路径 |
|---|---|
| 网络 + 软上限（**两端共用**） | `diff_vla_wbc/delta_token.py` |
| 训练侧接入（特征/前向/保存） | `diff_vla_wbc/v3_step.py`（`_delta_token_features` / `_delta_token_delta` / `save_vla`） |
| CLI | `scripts/train_joint_v2.py`（`--delta-token*`、`--freeze-vla`） |
| 环境变量 → 参数 | `scripts/launch_joint_ha_v2.sh`（`_DT_ARG`，`DRY_RUN`） |
| 本轮配置 | `scripts/run_p1_round_b.sh` |
| 部署（**客户端零改动**） | `scripts/ha_vla_latent64_server.py` |
| 探针（按臂装载） | `scripts/probe_ft_landing.py` |
