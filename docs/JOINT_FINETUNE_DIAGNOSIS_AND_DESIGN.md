# 联合微调诊断与重设计（2026-10-25）

> 位置：`docs/JOINT_FINETUNE_DIAGNOSIS_AND_DESIGN.md`
> 范围：IsaacLab 闭环（HumanoidArena 资产 + SuperMimic WBC + GR00T N1.7 VLA）与联合微调。
> 目的：①记录已解决/未解决的问题；②SuperMimic 侧的分析结论；③**联合微调应该怎么重新设计**，
> 尤其是"被平均"这一类问题。

---

## 0. 一页摘要

| 状态 | 结论 |
|---|---|
| ✅ **摔倒已解决** | 根因是闭环 `action_scale` 口径错（BeyondMimic 0.395/0.548 → HA 0.3789/0.75）；修后 PPBox 从"278 步摔倒"变成"**300 步跑满、不摔**" |
| ✅ 其它已修 | HA 默认姿态 / HA 驱动 kp-kd / ref6d 常量选型 / 联合微调 token 漂移（锚定解决）/ 数据完整性确认 |
| ❌ **任务不成功** | 两臂都 `timeout`、`max_reward=0.0`：走到目标附近但**不抓取** |
| ❌ **ft 臂行为退化** | 官方 VLA 会走向目标；联合微调后的 VLA（2000 与 anchor4000）在目标间乱走/侧倾 |
| ❌ 手部未生效（已修通链路） | 之前闭环**完全不驱动手指**；已实现"VLA 14 维手部直通"，但手指仍未见闭合 |
| ⚠️ 执行器能力不足 | 评测 WBC 只训过 BONES-SEED locomotion（129,943 clips），HA 操作数据只用了 318/699 |

**一句话**：摔倒是"口径错"，不是"策略差"；剩下的失败是"**执行器不会操作** + **联合微调目标把行为训坏了**"。

---

## 1. 已解决的问题（含证据）

### 1.1 闭环 action_scale 口径错 → 摔倒（最重要）
| 项 | 值 |
|---|---|
| 训练侧（`SUPERMIMIC_HA_ALIGN=1`） | mean **0.3789** / max **0.7500** |
| 闭环旧口径（BeyondMimic） | mean **0.3950** / max **0.5475** |
| 更糟：锁的 import 写错 → 兜底 | **0.250 / 0.250**（`ModuleNotFoundError: trackerTask.trackerTask`） |
| 逐关节后果 | `hip_pitch` 放大 **2.85×**、`wrist_yaw/pitch` 缩小 **10×** |
| 修复后实测 | PPBox seed0：**278 步摔倒 → 300 步跑满、`timeout`（不摔）** |

修复位置：
- `action_provider_supermimic.py`：模块级强制 `SUPERMIMIC_HA_ALIGN=1` + `_resolve_action_scale()` 校验；
- **修正 import 路径**（`source/trackerTask` 加入 `sys.path`，`from trackerTask.supermimic.robots import get_robot`）；
- 兜底分支加 `traceback`（以后不会静默退 0.25）。

### 1.2 P0：执行器接口对齐（三项，均带回退）
| 项 | 新默认 | 回退开关 |
|---|---|---|
| 初始姿态 + 观测偏移 | 训练侧 `DEFAULT_JOINT_POS_HA`（膝 0.42 / 踝 −0.23 / 肘 0.87） | `SM_DEFAULT_POSE=asset` |
| 驱动 kp/kd | HA 表（踝 60/6.5、膝 160/4.5、腰 150） | `SM_DRIVE=asset` |
| obs99 末尾 6 维 | 常量 `zeros`（A/B：zeros 114/124 步 vs identity 48/47） | `SM_REF6D=identity` |

> 结论：这三项**都不是摔倒主因**（改后 74→76 步，噪声级），但接口口径已与训练一致。

### 1.3 联合微调的 token 漂移 → 已解决
| 配置 | `token_mse` 轨迹 |
|---|---|
| 无锚定 | 0.077（2000 步）→ 0.093（3000，vel=1.0）/ 0.108（vel=0） |
| **token 锚定 λ=1.0** | 0.077 → **0.019（2500）→ 0.018（3000）→ 0.015（4000）** |
| 从头 + 锚定（1000 步） | **0.020** |

结论：**锚定能钉住 token 空间**，但（见 §3）**不解决行为**。

### 1.4 数据完整性（核实）
- BONES-SEED packed：**129,943 clips / 6.3 GB**，中位 789 帧 → **完整**；
- HA 699 条：`g1_hand_joints_14`、`g1_motion_token_64`、`body_quat_w` 全在；
- **但** `data/token_motions/g1_tasks` 只有 **318/699**（HA stage2 训练只用了 45%）。

---

## 2. 当前仍存在的问题

### 2.1 任务不成功（reward 恒 0）
- base（官方 checkpoint-8000）与 ft（anchor4000 / vla2000）都是 `timeout` + `max_reward=0.0`；
- 行为差异见 §2.2；抓取从未发生。

### 2.2 ft 臂行为退化（A1 的核心风险）
| VLA | 行为 |
|---|---|
| base（官方） | **走向白桌子（箱子所在，y≈−5.4）**，围着目标转 |
| anchor4000（锚定） | 桌子 / 墙角 / 蓝色货架之间**打转**（货架 y≈−2.7，是错误目标） |
| vla2000（无锚定） | **相机全程侧倾**、乱走，姿态不稳 |

→ **不是 SuperMimic 的问题**（两臂同一 WBC），**是联合微调把 VLA 的任务行为训坏了**。

### 2.3 手部
- 旧闭环：`full[:, body_idx] = body_target`，**手指永远 default**（从不闭合）；
- 已实现直通（§4.5），但 DoubleDesk 里手指**未见闭合**，需查三件事（VLA 手输出是否≈0 / 侧内关节顺序 / 腕部是否到位）。

### 2.4 执行器能力（WBC）
- 评测 WBC = `bones_seed_token_ha_v2`（BONES-SEED locomotion）；
- HA 版 WBC `ha_stage2_hatoken_1025_0217` 只训了 **318/699、13.3k iters**；
- 参考：同 run 的 `sim_r` 峰值在 **12.5k–17.5k**（ep_len 223→208），**32k 退化到 ep_len 108**。

---

## 3. SuperMimic 侧分析出的问题（逐条）

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| S1 | obs99 末尾 6 维是**参考朝向**，部署无真值 | 训练：`mat_to_6d(Rt @ root_R[f])`；Arena 无参考动作 | 只能常量，分布外（但策略不敏感） |
| S2 | 闭环 `action_scale` 口径错 | §1.1 | **摔倒主因**（已修） |
| S3 | 闭环 kp/kd 与训练不同 | Arena 踝 20/1.0 vs 训练 60/6.5 | 噪声级（已对齐） |
| S4 | 闭环 reset 姿态与训练默认不同 | TWIST2 vs `DEFAULT_JOINT_POS_HA` | 噪声级（已对齐） |
| S5 | 训练数据是 BONES-SEED locomotion | packed clip 名 `Idle_Left_...` | 不会操作/抓取 ← **主因之一** |
| S6 | WBC 训练 episode 上限 500 步 | `env.py:146` 默认 | 长时程训练不足 |
| S7 | 手部不在 WBC 里、闭环也没执行 | provider 只写 body；client 丢 `hand_binary_chunk` | 抓取必失败（**已修通**） |
| S8 | VLA chunk 每 12 步重取一次 | 300 步 / 25 次 infer | 可能与 40 帧 token 语义不符（待查） |
| S9 | 数据集 14 维手指为 `[left7, right7]`，asset 是左右交错 | provider 打印的 finger 顺序 | 已做左右分组，**侧内顺序待验证** |
| S10 | HA 数据里 3/7 任务手全 0 | pp_box 2.5%、open_door 0.9% | 手部头"被平均"，见 §4.6 |

---

## 4. 联合微调的设计（重点）

### 4.1 现状（代码级）
```
ego 图 + 64 状态 + 指令
  → GR00T N1.7（backbone 冻结，只训 action head）
  → 64 维 token chunk（40 帧）
  → SuperMimic WBC policy（10 帧 obs + token；lr 1e-5 微动）
  → 29 维动作
  → 世界模型推演 H=8（**冻结**，`--wm-update-every 0`）
  → 每步解析奖励 → 折扣回报 → 反传 VLA head
```

**损失组合**（`compute_joint_loss` + trainer）：
| 项 | 权重 | 说明 |
|---|---|---|
| reward | 1.0 | `−discounted_return(γ=0.99)` |
| state | 0.5 | pos + ori(geodesic) + q + vel×`state_vel` |
| smooth | 1e-3 | 动作差分平方 |
| safety | 0.01 | 终止概率均值 |
| hand | 0.2 | 14 维手 MSE（mask 0.05、grasp boost 5） |
| token_anchor | 0（可设） | λ·MSE(token, 端口 token) |
| goal/task/fm | 0 | 未传 teacher 时跳过 |

**奖励本体**（`MDP.reward`）：
- 6 项跟踪核 `exp(−e/std²)`：body_pos(1.0, 0.3) / body_ori(1.0, 0.4) / body_linvel(1.0, 1.0) /
  body_angvel(1.0, 3.14) / root_pos(0.5, 0.3) / root_ori(0.5, 0.4)；
- 2 项惩罚：`action_rate −0.1`、`joint_limit −2.0`；
- 全部 × `CONTROL_DT=0.02` → 每步上限 **0.10**，实测 **0.083**；
- 参考**重锚到机器人 root（xy+yaw）**；终止 `term_height 0.25 / term_tilt 0.8`（几乎不触发）。

### 4.2 现存缺陷：四个"被平均 / 被吞掉"
| # | 现象 | 数值 | 本质 |
|---|---|---|---|
| D1 | **奖励被饱和平均** | 好/坏 token 只差 12%（0.083 vs 0.074），batch 噪声 ±40% | `exp(−e/std²)` 在小平米误差处饱和；8 步内 token 差异进不了核的敏感区 |
| D2 | **摔倒被平均掉** | `survival≈0.99996`、`term≈0` | 想象里几乎不终止 → 无"别摔"梯度 |
| D3 | **手部被平均成 0** | 3/7 任务手全 0；`hand_mse` 0.006–0.014 | 全帧 MSE + 任务混合 → 手部头输出接近常数 0 |
| D4 | **行为被 token 锚定平均** | token_mse 0.077→0.016，但导航退化（走错目标） | 锚定目标=数据集 token，不是"任务做对" |

另外两个放大器：
- **batch=2** → 梯度噪声大（100 步内 reward 0.055–0.096 波动）；
- **H=8** → 误差未累积，看不出好坏 token 的长期差异（A4：H32 的 token_mse 0.103 vs H8 0.167）。

### 4.3 新设计总览（保住可微 WM 的优势）
> 核心原则：**主梯度仍来自可微世界模型的反传**（不用仿真器），但要让奖励**有信息**；
> 监督项只做"防退化锚"，不替代行为优化；所有项都**带掩码/退火**，避免"被平均"。

```
L = w_a · L_action_distill        # 新增：执行器一致性（行为级监督，低成本）
  + w_t · L_token_anchor(退火)     # 1.0 → 0.1
  + w_s · L_state(vel=0.1)        # 0.5
  + w_h · L_hand(masked)          # 0.5，只算"有手部动作"的帧
  + w_r · L_reward(去饱和)         # 0.3（先小，等奖励有信息再升）
  + w_p · L_posture               # 0.5（tilt/height 连续惩罚）
  + w_c · L_contrast              # 对比式白化（见 4.4）
```

### 4.4 让奖励重新"有信息"（治 D1/D2）
1. **去饱和**：把 `exp(−e/std²)` 换成 `−e`（或 std 缩到 1/3），误差立刻产生梯度；
2. **加绝对项**：root 位置/朝向的**世界坐标**误差（现在参考重锚到机器人，允许自由平移）；
3. **姿态惩罚**：`−w·relu(tilt−0.5)`、`−w·relu(0.6−height)`（cfg 已有 `term_tilt/term_height`）；
4. **对比白化**：同一状态采 K=4 个 token（VLA token + 3 个扰动/打乱），
   `advantage_i = (r_i − mean(r))/std(r)`，用 `softmax(advantage/T)` 加权；所有 token **共用同一 `xi`**（公共随机数）；
5. **H 8 → 32**（A4 证据），K=2 个 xi 采样；
6. **梯度累积 4–8**（等效 batch 8–16），把 ±40% 噪声压到 ±10%。

### 4.5 手部（治 D3，且你指出的"手不进 SuperMimic"）
**已实现**（本次）：
- server：`hands14_chunk`（VLA 第 64–78 维）；
- client：缓存 `last_hands14_chunk`；
- provider：`SM_HAND=1` 直写 finger target（左右分组；`SM_HAND_SCALE`）。

**待补**：
1. 侧内 7 维顺序校准（用一帧录制数据逐维对名字）；
2. **手部监督要"按任务、按帧"而非全局平均**：
   - 只在 `|hand_gt|>阈值` 的帧算（已有一半 mask）；
   - **按任务均衡采样**（7 任务 × 100 clips，别让 3 个零手任务稀释）；
   - 权重 0.2 → 0.5；可加"手部开合二分类"辅助损失（`max(fingers)>1.70`）；
3. **任务选择**：抓取验收用 **HOI_double_desk（手 67.6%）/ HOI_grap_cup（36.5%）**，
   PPBox（2.5%）只当"导航 + 不摔"任务。

### 4.6 治 D4：锚定与行为的平衡
1. token 锚定**退火**（1.0 → 0.1，500 步后线性降）；
2. 新增 **action distill**：`‖π(VLA_token) − π(rec_token)‖²`（WBC 冻结，port token 侧车 699 条已齐）；
   它给"行为级"监督，且完全在可微链路里（不过仿真）；
3. **闭环选 ckpt**：每 500 步跑 1 任务 × 1 seed（≈4 min），按"能否接近目标/是否摔"挑，
   不按 `token_mse` 挑。

### 4.7 实施清单（文件级）
| 文件 | 改动 |
|---|---|
| `diff_vla_wbc/rollout.py` | 去饱和奖励选项、`state_vel` 默认 0.1、姿态惩罚项、对比白化 |
| `diff_vla_wbc/train.py` | `token_anchor`（已有）、新增 `action_distill`、`hand` mask 按任务、梯度累积 |
| `scripts/train_joint.py` | `--action-distill-weight`、`--token-anchor-anneal`、`--reward-mode linear`、`--posture-weight`、`--grad-accum`、`--closed-loop-every` |
| `action_provider_supermimic.py` | 手部直通（已加）、侧内顺序表、finger 日志 |
| `scripts/ha_fast_eval.sh` | `--sm_obs_mode` 可配（为 93 维 WBC 铺路） |

### 4.8 验收门禁（每步都设"过/不过"）
| 阶段 | 指标 | 门禁 |
|---|---|---|
| 数据 | HA token 侧车 | **699/699** |
| WBC stage2 | `sim_r` / `ep_len` | `sim_r ≥ 0.85` 且 `ep_len` 不下降 |
| base 闭环 | 能否走到目标/抓 | 至少"接近 + 手闭合" |
| 联合微调 | `token_mse` / 闭环行为 | `token_mse ≤0.02` **且** 导航不退化（对比 base） |
| A1 | 6 任务 × 3 seeds 成功率 | 两臂同 WBC、同 locked scale |

---

## 5. 复现与运行（当前可用命令）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

# VLA server（示例：anchor4000）
PYTHONPATH=$PWD/third_party/gr00t_deps/_noflash python3 scripts/ha_vla_latent64_server.py \
  --ckpt runs/ckpt_vla_ha_anchor4000 --port 18087 --device cuda:0 --bf16 \
  --horizon 40 --fm-steps 4 > logs/anchor4000_server.log 2>&1 &

# 单集闭环（必须看到 [SuperMimic][LOCKED] 与 hand passthrough）
SM_CKPT=$PWD/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt \
TASK=DoubleDesk SEED=0 MAX_STEPS=300 SERVER_URL=http://127.0.0.1:18087 \
OUT_ROOT=$PWD/runs/anchor4000_hand_eval2 \
  bash scripts/ha_isaac_closedloop.sh
```

**审计行（缺一不可）**：
```
[SuperMimic][LOCKED] HA_ALIGN=1 action_scale mean=0.3789 max=0.7500
[SuperMimic] hand passthrough ON (scale=1.0)
```

---

## 6. 回退开关一览（全部可回退）

| 开关 | 默认 | 回退效果 |
|---|---|---|
| `SM_DEFAULT_POSE` | `ha` | `asset` = 旧行为 |
| `SM_DRIVE` | `ha` | `asset` = 旧行为 |
| `SM_REF6D` | `zeros` | `identity` |
| `SM_HAND` | `1` | `0` = 不驱动手指 |
| `SM_HAND_SCALE` | `1.0` | 缩放手指命令 |
| `SM_ALLOW_SCALE_MISMATCH` | 未设 | `1` = 跳过 action_scale 强制锁（**仅 A/B**） |
| `--state-vel-weight` | 0（新标准） | 1.0 = 含速度地板 |
| `--token-anchor-weight` | 0 | 1.0 = 锚定 |
| `TOKEN_ANCHOR_WEIGHT` | 0 | 同上（launcher 透传） |

---

## 7. 待办优先级（建议执行顺序）

1. **P0**：补 699/699 token 侧车（20–30 min）；
2. **P0**：WBC stage2（BONES-SEED → HA 699，`--obs_mode paper_obs93`，locked scale），4 卡；
3. **P1**：手部三项（侧内顺序、掩码+均衡、权重 0.5）；
4. **P1**：联合微调重设计（§4.3–4.6：去饱和奖励、对比白化、H32、action distill、锚定退火、梯度累积、闭环选 ckpt）；
5. **P2**：A1 两臂 × 6 任务 × 3 seeds（locked、新 WBC、新目标），出成功率表。

> 原则：**每一步都要有"过/不过"的门禁**（§4.8），不过就不往下走——
> 这轮踩过的坑（action_scale、静默兜底、手部未执行）都是"没有审计行"导致的。

---

## 8. 实施状态（2026-10-26 代码已落地，默认全部关闭）

| 改动 | 文件 | 开关 | 默认 | 冒烟验证 |
|---|---|---|---|---|
| 奖励去饱和（`exp` → `-(e/std²)`） | `diff_vla_wbc/isaac_joint.py` | `--reward-mode linear` | `saturating`（旧行为） | ✅ 4 步无 NaN，`rollout/reward` 出现负值（0.024 → −0.004/0.043） |
| 连续姿态惩罚（tilt/height） | 同上 | `--posture-weight` | 0 | ✅ 同上 |
| 动作级蒸馏 `π(VLA tok) ≈ π(rec tok)` | `diff_vla_wbc/train.py` | `--action-distill-weight` | 0 | ✅ `loss/action_distill` 正常输出（2.5–7.3） |
| token 锚定退火 | `scripts/train_joint.py` | `--anchor-anneal-steps` | 0 | ✅ 与锚定共存 |
| 梯度累积 | `diff_vla_wbc/train.py` | `--grad-accum` | 1（=旧行为） | ✅ accum=2 跑通 |
| 手部直通（VLA 14 维 → 手指） | server / client / `action_provider_supermimic.py` | `SM_HAND` / `SM_HAND_SCALE` | 1 / 1.0 | ✅ DoubleDesk 300 步跑通（手指未见闭合，待查侧内顺序） |
| action_scale 强制锁 + 正确 import | `action_provider_supermimic.py` | 固定 | 强制 | ✅ `[LOCKED] mean 0.3789 max 0.75` |

**安全性**：所有开关默认关闭 → 与旧代码逐位等价（`grad_accum=1` 时 backward/step 与原逻辑一致；
`reward_mode=saturating` 且 `posture_weight=0` 时 reward 直通不改）。

**注意**：`--reward-mode linear` 下 `rollout_reward` 的量纲变了（可正可负），**不能与旧的
0.083/0.079 直接比较**；换口径时必须用同一 reward_mode 重跑对照。
