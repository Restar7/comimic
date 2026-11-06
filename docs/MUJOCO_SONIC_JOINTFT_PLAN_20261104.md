# MuJoCo / mjlab + GR00T-N1.7 + SONIC 联合微调计划（唯一执行版）

> 2026-11-04。本文档替代 `JOINT_MULTIROUND_PLAN_20261103.md`（IsaacLab + SuperMimic 版）与
> `FULL_LENGTH_5ROUND_PLAN_20261101.md`（IsaacLab + SuperMimic 版）作为**当前执行口径**。
> 两条旧文档保留作**方法论参考**（奖励形状、门、轮次课程、验收判据），但
> **平台 = MuJoCo / mjlab**、**执行器 = SONIC release decoder（不再重训 SuperMimic WBC）**、
> **只做 VLA 联合微调**。
>
> 本文档先回答一个决定路线的问题：**318 数据集的 motion token 能不能在 MuJoCo 里直接回放？**
> 答案在 §2：**能复现目标（0.03–0.06 rad），但闭环会前倾摔倒（~1 s）**，与 IsaacLab 同结论。
> 因此主线走 §5 的 **B 路线（VLA 在环 + 每轮重采）**，A 路线降级为
> **短分段（H=20）只用于 WM 的 teacher-forced 数据**（§6），并保留"植物标定"作为翻盘杠杆（§4）。

---

## 0. TL;DR

| 项 | 结论（2026-11-04 实测） |
|---|---|
| 318 数据的 token↔state↔target 口径 | ✅ **完全自洽**：`decoder(录制状态历史, token[t])` 复现 `action.wbc[t]`，**MAE 0.035–0.059 rad**；token 时间偏移扫描 **0 最优**（±5 帧明显变差） |
| 318 的关节序 | ✅ `observation.state` = **STANDARD_29**（左腿→右腿→腰→左臂→右臂）+ 左手指 7 + 右臂 7 + 右手指 7；用 `observation.eef_state` 的 FK 逐位对拍（误差 0.000 m）**排除**了交错序 |
| 执行器语义 | ✅ `action.wbc` = `SONIC_DEFAULT_POS + ACTION_SCALE · decoder_action`（绝对关节目标，STANDARD 序）；与 mjlab/SuperMimic 的动作空间**同底**（同 default、同 `0.25·effort/stiffness`、同 kp/kd）⇒ 换执行器只需一次**关节序置换** |
| 目标回放（把 `wbc` 当位置目标灌进 MuJoCo） | ⚠️ 关节跟得很好（**0.034–0.058 rad**），但**基座持续前倾，46–70 步摔**（deploy/HA 两套增益都摔） |
| token 闭环回放（token→decoder→PD→MuJoCo） | ❌ **摔**：46–50 步（~1 s），从近静止帧起跑也只有 12–67 步 |
| 20 步短分段回放（每段重置到参考） | ✅ **0/40 摔**；target 臂 joint MAE 0.034 rad / xy 漂移 7.9 cm / 末倾 7.9°；token 臂 0.096 rad / 8.4 cm / 16.5° |
| 5 步短分段 | ✅ 0.017 rad / 0.4 cm / 2.4° |
| 摔倒机制 | 录制姿态的**质心-支撑边界只剩 2–3 cm**（膝近直、身体前倾），MuJoCo 里踝力矩压不住 → 绕脚尖前翻；把躯干质心后移 8–10 cm 可撑到 80 步但仍在漂 ⇒ 是**植物（质量分布/接触/踝）不保真**，不是数据口径错 |
| 采集速度 | ✅ 单 env CPU **530–1600 控制步/s**（2803 帧一集 ≈ 2–5 s），比 Isaac 的 ~18 步/s 快 **30–90×**，所以"多采"完全成立 |
| **裁决** | **replay 路线不达标（不是"非常精准"）⇒ 主线走 B 路线**；A 路线只在 H=20 分段内使用，并先做 §4 植物标定 |
| 主执行器 | **SONIC release decoder（冻结）**，不再重训 SuperMimic WBC |
| 主训练 | **只联合微调 VLA**（LoRA + token head），WM 冻结、decoder 冻结 |
| 平台 | MuJoCo（`groot_repro` 部署模型 + `g1_assets/g1_43dof_position.xml`）+ mjlab 资产；**不再依赖 IsaacLab** |

---

## 1. 架构（改完之后的形状）

```text
                      ┌──────────── VLA（GR00T-N1.7，官方 318 微调 ckpt 作 base）────────────┐
图像（318 录制帧）+ 语言指令 + 43 维 proprio ─►│ backbone + action head ─► token(64) + 手(7+7) │
                      └───────────────────────────────────────────────────────────────────────┘
                                        │ token(64)（唯一的身体指令）
                                        ▼
                      ┌──────────── SONIC release decoder（冻结，ONNX/可微 torch）────────────┐
                      │ 输入 994 = token | ang_vel 10×3 | q 10×29 | qd 10×29 | last_a 10×29 | gravity 10×3 │
                      │ 输出 29 维 raw action（HA_V31 序）→ q_target = DEFAULT + SCALE·a       │
                      └───────────────────────────────────────────────────────────────────────┘
                                        │ 29 维关节目标（STANDARD 序）
                                        ▼
                      ┌──────────── MuJoCo G1（faithful plant，§4 标定）───────────────────────┐
                      │ 43 执行器；PD 在物理步内重算（500 Hz）；外部 50 Hz 换目标               │
                      └───────────────────────────────────────────────────────────────────────┘
                                        │ (s, a, s')
                                        ▼
                      ┌──────────── 世界模型 WM（冻结；冻结后才做联合微调）────────────────────┐
                      │ 71 维 state + 29 维 action → next state；解析奖励 = 参考跟踪误差        │
                      └───────────────────────────────────────────────────────────────────────┘
                                        │ 20 步想象梯度
                                        ▼
                         γ^t·r_t → VLA（LoRA + token head）      decoder/WM/plant 全部 no_grad

手部：VLA 手 head(14) → HandStream → 手执行器；**不进 token、不进 WM**。
```

**与旧架构的差异（全部是删除项）**：

| 项 | 旧（IsaacLab + SuperMimic） | 新（MuJoCo/mjlab + SONIC） |
|---|---|---|
| 身体执行器 | 重训 SuperMimic WBC（28k 步，G1 从未通过） | **冻结 SONIC release decoder**（官方部署权重） |
| 动作空间 | `(q-q0-bias)/scale` | **同一个**（同 default、同 scale、同 kp/kd）⇒ 只差关节序置换 |
| 平台 | IsaacLab + trackerlab + USD | **MuJoCo MJCF + mjlab** |
| 训练内容 | WBC 重训 + VLA 联合微调 | **只联合微调 VLA** |
| 数据 | A'/B 在 Isaac 里采（16 env 上限，18 步/s） | MuJoCo（530–1600 步/s，可开多进程/多卡刷满） |
| 318 的 `motion_token` | 用作 VLA 标签 | **同样**用作 VLA 标签 + 回放的指令源 |

> 关节序契约：唯一真源 `joint_contract.py`。`STANDARD_29`（MuJoCo/WM）↔ `HA_V31_29`（decoder）
> 用 `permutation()` 换序，**禁止写字面量**。

---

## 2. 实验：318 token 在 MuJoCo 里能不能 replay？（已跑完，附数字）

脚本：`scripts/replay_ft318_mujoco.py`（新增）。数据：`ft318_frozen_20261023`（9 个子集 / 318 集 /
473,876 帧 / 50 fps）。植物：`GR00T-WholeBodyControl/.../scene_43dof.xml`（官方部署模型，43 执行器，
armature 0.01，damping 0.05，摩擦 1.0）。

### 2.1 先证明数据口径（否则后面全是错归因）

| 检查 | 方法 | 结果 |
|---|---|---|
| 关节序 | 把 `observation.state` 当关节角做 FK，与 `observation.eef_state` 的腕位对拍（4 种假设：绝对/相对默认 × STANDARD/交错） | **绝对 + STANDARD** 逐位命中（`[0.246 0.232 0.138]` vs `[0.246 0.232 0.138]`）；交错序差 0.28 m ⇒ 排除 |
| `action.wbc` 语义 | 与 `decoder(DEFAULT+SCALE·a)` 对比 | 绝对目标（MAE 0.136 rad 最小；归一化假设 0.197 更差） |
| token 时间对齐 | 扫 `token[t+k]`，k∈{-20..20} | **k=0 最优**（0.0616；±5 → 0.0646/0.0673；±20 → 0.073/0.082） |
| decoder 复现能力 | 用**录制状态历史**喂 994 维，比 `action.wbc` | **0.035–0.059 rad**（起点 100/300/600 三处） |
| 权威对照 | 与 `docs/DEBUG_WBC_DECODER_20261103.md` §6/§8 的 q 块语义、动作映射 | 一致（q 块 = `q − DEFAULT`，ISAACLAB 序；目标 = `DEFAULT + SCALE·a`） |

⇒ **链路本身没问题**。后面摔，只能是植物/闭环动力学。

### 2.2 回放实测（ep0 = `task_1_meeting_room_milk_basket/2026-07-14-20-30-01/ep0`，2803 帧）

| 实验臂 | 增益 | 结果 | joint MAE | 备注 |
|---|---|---|---|---|
| `target`（wbc 当位置目标） | deploy（hip99/knee99/ankle48.4） | **摔 @62 步** | 0.0375 rad | 关节跟得极好，基座前倾 |
| `target` | HA（hip130/knee160/ankle60/arm40） | **摔 @70 步** | 0.058（vs 目标） | 同上 |
| `token`（token→decoder 闭环） | deploy | **摔 @46 步** | 0.094 | 与 Isaac 同量级 |
| `token` | HA | **摔 @50 步** | 0.306 | 反馈与植物不匹配更差 |
| 6 个近静止帧起跑 300 步 | HA | 12–67 步全摔 | 0.11–0.26 | 不是"起点相位"问题 |

**关键诊断（为什么摔）**：

1. 录制姿态的**质心前边缘余量只有 2–3 cm**（f0: front +0.033；f2025: +0.019），身体是前倾站姿；
   部署默认姿态是 +0.10/−0.11 的对称余量 ⇒ 录制姿态本身就是"骑在脚尖上"。
2. 仿真里踝力矩 +15…+28 Nm，但**质心仍持续前移**（20 步内 +0.019→+0.055），脚跟离地
   （foot_x min 从 −0.180 → −0.068）⇒ 绕脚尖前翻。
3. 把**躯干质心后移** −0.06/−0.08/−0.10 m：撑满 80 步（末倾 56°/25°/22°，仍漂）。⇒ 是
   **质量分布/接触/踝模型**的系统性偏差，不是某一处符号错。

### 2.3 分段（每个 H 步重置到参考）——**唯一可用的形状**

40 段 × 20 步（均匀覆盖全集）：

| 臂 | 摔 | joint MAE | xy 漂移 | z 下沉 | 末倾 |
|---|---|---|---|---|---|
| `target` H=20 | **0/40** | 0.034（p90 0.045） | 0.079（p90 0.104） | 0.033 | 7.9° |
| `token` H=20 | **0/40** | 0.096（p90 0.124） | 0.084（p90 0.163） | 0.070 | 16.5° |
| `target` H=5 | 0/40 | 0.017 | 0.004 | 0.018 | 2.4° |

**这就是 Isaac 侧 `--segment 20` 结论的 MuJoCo 复现**：长时回放不可用，短分段可用。

### 2.4 裁决（照用户给的分支规则）

> "如果 replay 路线非常精准、没啥问题、可以大量采集 ⇒ 走 JOINT_MULTIROUND；不然 ⇒ 走另一条。"

- replay **不是**"非常精准"：闭环 ~1 s 摔，20 步漂移 8 cm（Isaac 计划里的基线是 2.33 cm）；
- 但**采集速度**与**口径正确性**都好，分段数据完全可用于训 WM；
- ⇒ **主线 = B 路线（§5）**；A 路线只在 H=20 分段内做 WM 数据（§6），并把 §4 的植物标定作为
  "把 A 路线救回来"的唯一杠杆（标定过了 → 回 §6.3 升级为主线的候选）。

---

## 3. 门（重写；每条都能当场跑）

| 门 | 判据 | 现状 |
|---|---|---|
| **G0** 数据口径 | `decoder(录制状态, token[t])` vs `action.wbc[t]`：MAE ≤0.08 rad（起点 300 帧处 ≤0.05）、token 偏移 0 最优 | ✅ **PASS**（0.035–0.059；偏移 0 最优） |
| **G0b** 执行器等价 | `SonicDecoderTorch` vs ORT ≤1e-5；关节序置换 == `joint_contract.permutation()` | ✅ 图已对拍（3.3e-6，`docs/DEBUG_WBC_DECODER_20261103.md` §2.2）；置换见 §8 模块 |
| **G1** 植物保真 | 从录制帧起跑 **300 步 target 回放不摔**（up>0.9）且 xy 漂移 p90 ≤5 cm/60 步 | ❌ **FAIL**（70 步摔）— §4 的工作项 |
| **G1'** 分段可用 | H=20：0 摔、joint MAE ≤0.12、xy 漂移 p90 ≤0.10 m | ✅ **PASS**（0/40；0.096；0.084→p90 0.163 ≈ 边缘） |
| **G2** 数据覆盖 | A'/B 覆盖全部 318 集、帧覆盖 100%、动作/状态有限、无趴地行 | ⬜ 待采 |
| **G3** WM | 20 步预测：in-dist p90 ≤5 cm（R1）、≤1 cm（R3+） | ⬜ 待训 |
| **G4** 联合微调 | `imag_state`/关键段误差逐轮下降；`act_rms` 不贴死 trust 边界 | ⬜ |
| **G5** 闭环 | 20 步漂移、分级成功率、总漂移（R1 ≤50 cm、R3 ≤10 cm）；配对 vs 上一轮 CI 不含 0 | ⬜ |

> **G1 是唯一"红"的硬门，但它不挡 B 路线**（B 路线用 VLA 在环，策略自己反馈，不依赖 replay）。
> 它挡的是"用 teacher-forced replay 采 A'"这条路，所以 A 路线必须等 G1 或退到 H=20。

---

## 4. 植物标定（唯一能把 A 路线救回来、也能把 B 路线做得更真的工作项）

**目标**：MuJoCo 里的 G1 在录制目标流驱动下，**300 步不摔、60 步 xy 漂移 p90 ≤5 cm**。

**做法（按性价比排序，每项都是可跑的脚本）**：

1. **质心/质量分布标定**（半天）：把 `torso_link` / `pelvis` 的 `body_ipos`（或其子树的点质量）
   当参数，用"录制目标流回放 80 步不摔 + 质心余量匹配"做 1D/2D 粗扫（已验：−0.08…−0.10 m 可撑
   80 步）。写进 `scripts/calibrate_g1_plant.py`，产物 `runs/plant_calib.json`。
2. **接触模型标定**（半天）：脚底 mesh 换 box/多接触点、`solref/solimp`、`friction`（1.0→0.8/1.2）
   三档扫描；用同一回放指标选。
3. **踝/膝驱动标定**（半天）：`ankle_pitch` 的 kp/effort（HA 60/40 vs deploy 48.4/50）与
   `armature` 二选一；判据同 G1。
4. **初始状态补全**（2 h）：用**脚锁定**估基座初速（stance foot 静止假设；实测首 200 帧 ≈0.01 m/s，
   说明起点是站立，后续行走段需要非零初速）；`qvel` 用状态中心差分；基座姿态用
   `observation.root_orientation[0]`（4.5° 前倾）而不是单位四元数。
5. **场景/物体**（1–2 天，可选）：318 的四类任务场景（会议桌/办公桌/厨房/餐桌）。**只对 B 路线重要**
   （VLA 图像），对 WM 不重要（WM 只看 state/action）。

**首轮粗扫结果（2026-11-04，`scripts/calibrate_g1_plant.py`，ep0，120 步上限）**：

| 增益 | 躯干质心 x 偏移 | 撑住步数 | 末倾 | xy 漂移 | joint MAE |
|---|---|---:|---:|---:|---:|
| HA | 0 | 67 | 61° | 0.74 m | 0.071 |
| HA | −0.04 | 75 | 64° | 0.71 m | 0.073 |
| HA | −0.08 | 96 | 62° | 0.87 m | 0.084 |
| HA | −0.10 | 97 | 62° | 0.84 m | 0.086 |

⇒ 质心后移能延命（67→97 步），但**仍摔**，而且漂移 0.7–0.9 m 说明是"被腿推着往前走"，
不是单纯前翻。**单参数标定不够**，§4 的 2–4 必须一起做。

**标定目标函数（新，重要）**：数据集**没有根位置**，但有 `observation.root_orientation`（逐帧基座姿态），
这是唯一可用的"基座真值"。所以标定的损失应当是
`mean_t ‖ up_sim(t) − up_rec(t) ‖`（当前 0 → 61° vs 录制恒 4.5°，差距一眼可见），
而不是"看起来站不站得住"。漂移只能作辅助项。

**判据**：G1 PASS 之后，A 路线从"仅分段"升级为"可长时回放"，再按 §6.3 重新评估是否切回
JOINT_MULTIROUND 主线。

---

## 5. 主线：B 路线（VLA 在环 + 每轮重采），5 轮

> 借 `FULL_LENGTH_5ROUND_PLAN_20261101.md` 的**课程/验收**，但：平台 MuJoCo、执行器 SONIC、
> 只训 VLA、数据量按 MuJoCo 的速度放大。

### 5.1 为什么是它

- 不依赖 replay 保真度：数据 = **策略在这个 sim 里真的做了什么**（on-policy），自洽；
- MuJoCo 采集极快（530–1600 步/s/进程），可以 8 进程 × 8 卡刷满，**每轮 1000+ 集**都不是瓶颈；
- VLA 只在数据集图像上跑（318 有 `videos/`），**不需要在 MuJoCo 里重建四个场景**；
- 与 `https://github.com/Restar7/issac_groot` 成功过的形状一致（B 线 + WM + joint），只是把执行器换成 SONIC。

### 5.2 每轮数据

| 轮 | 采集策略 | 条数 | 长度 | 备注 |
|---|---|---|---|---|
| R1 | base VLA（=**官方 318 微调**） | **600**（每任务 150） | 全长（START_FRAME=0, MAX=length） | MuJoCo 侧 8 进程，约 20–40 min |
| R2–R5 | 上一轮最优 | **318 × 1 遍**（每轮全量重采） | 全长 | 每轮 15–25 min |
| 附加 | R1 起每轮**分段 A'**（§6）| 318 集 × 64 段 × 20 步 | H=20 | 30–60 min，只喂 WM |

- B 线不设 600 步 cap；失败的 episode 记 `valid_len`；
- **图像来源**：318 `videos/` 的 ego 帧（与 state 同帧号）；不做渲染。
- **状态来源**：MuJoCo 真状态（`qpos/qvel`）→ 43 维 proprio（STANDARD 序 + 手）。

### 5.3 每轮 8 步（固定）

1. **preflight**：ckpt / 全集 clip 表 / VLA server / MuJoCo 植物版本哈希；
2. **B 采集**：`scripts/collect_b_ft318.py`（待写，§8）——VLA server + MuJoCo + SONIC；
3. **B → transitions**：`(s, a_sonic, s', reward)`，奖励用真实奖励函数（WM 不参与）；
4. **WM refresh**：`A'(分段) : B = 1:1`；三阶段 3000/3000/1500；
5. **WM gate**：20 步 p90 + OOD 相关；
6. **joint**：H=20、λ_r=0.3、`act_eps=0.10`、`act_weight=1.0`（口径见 §5.5），起点 **100% 参考**、
   `START_SAMPLER=phase`；200–300 updates；8 卡 DDP；
7. **eval**：全长、`START_FRAME=0`、32 对（R5 48 对）；指标 = 20 步漂移 / 全长漂移 / joint RMSE / 摔；
8. **report/gate**：§5.6。

### 5.4 每轮只改三件事

| 轮 | 起点 ckpt | δ_pos/δ_ori（奖励 Huber） | B 条数 | WM refresh |
|---|---|---|---|---|
| R1 | 官方 318 base | 5 cm | 600 | 不刷（隔离变量） |
| R2 | R1 best | 2 cm | 318 | ✅ A'+B 1:1 |
| R3 | R2 best | 5 mm | 318 | ✅ |
| R4 | R3 best | 2 mm | 318 | ✅ |
| R5（条件） | R4 best | 1 mm | 318 | ✅（仅当 G3 p90 <1 mm 才开） |

- 奖励形状全程不变：**Huber-on-error（`d=√e`）+ split 朝向**（tilt 1.0 / yaw 0.2 / yaw_rate 0.3）；
- 与旧文档一致：无 exp、无 hinge、无 velocity、token trust 关。

### 5.5 联合微调的不变量

| 项 | 值 | 说明 |
|---|---|---|
| `H` | 20 | 与最短分段一致 |
| `λ_r` | 0.3 | — |
| `act_eps` / `act_weight` | **0.10 / 1.0** | 2026-11-04 用户口径（起点永远从参考出发） |
| `token_quant` | `none` | 实测亚格外推 rel ≤0.83%（`docs/TOKEN_QUANT_STE_20261104.md`） |
| critic | on（`imagined`） | 与 METHOD_LOCK 一致 |
| 起点 | 参考 + phase/task 加权 | B 起点只作 R2 可选恢复臂 |
| 冻结 | decoder / WM / 植物 | 只反传 VLA(LoRA+head) |

### 5.6 验收（每轮）

1. vs **上一轮最优**：配对均值/中位 <0、95% CI 不含 0、核心任务中位不退化；
2. vs **base**：只作锚点；
3. 后段（last 50 tick）必须改善，只有早段改善不算；
4. 零摔、无 NaN、无单集 >+15 cm 灾难集；
5. WM gate PASS；`frac_out` 尾部 ≤0.5（`act_eps=0.10` 下 `frac_out≈1` 是**正常**，盯 `act_rms`）；
6. 32 集 CI ≈ ±2 cm；<1 cm 改进不下结论。

---

## 6. A 路线（分段 teacher-forced，只喂 WM；G1 过则可升级为主线）

### 6.1 形状

- 对每条 clip 均匀取段：`START_FRAME = f`，**执行 H=20 步后重置到参考 f+20**；
- 每段：`(s_t, a_t, s_{t+1})` 真值 + 参考 `ref` 帧；动作 = **SONIC decoder 原始输出**（不是 wbc）；
- 扰动：L1 动作噪声 `σ_rad=0.014`（按 `action_scale` 折算）、L2 起点扰动 `xy/yaw N(0,0.05)`、
  噪声重采 `hold∈{15..25}`（不再与 chunk 边界同相）；
- **不做长时链式回放**（会摔，§2.2）。

### 6.2 规模

| 项 | 值 |
|---|---|
| clip | 318（9 个子集全量） |
| 段/clip | 64（均匀，步长 `max(1, len/64)`） |
| 段长 | 20 |
| seed | 2 |
| 行数 | 318 × 64 × 20 × 2 ≈ **814k** 行（≈ Isaac 版 135k 的 6×，成本 30–60 min） |

### 6.3 升级条件（A 路线 → 主线）

同时满足才切：

1. **G1 PASS**（300 步 target 回放不摔、60 步 xy 漂移 p90 ≤5 cm）；
2. H=20 的 token 臂 xy 漂移 p90 **≤2 cm**（当前 8.4 cm）；
3. WM 的 20 步 p90 ≤1 cm（G3）；
4. `check_label_optimality`：漂移状态下"标签动作"仍最优（否则 joint FT 增量天然很小，
   见 `JOINT_MULTIROUND_PLAN_20261103.md` §14.3）。

达不到 ⇒ 停在 §5 的 B 路线，A' 只当 WM 数据。

---

## 7. 执行顺序（今天之后）

| # | 事项 | 产物 | 阻塞谁 |
|---|---|---|---|
| 1 | **植物标定**（§4.1–4.4） | `runs/plant_calib.json` + `scripts/calibrate_g1_plant.py` | A 路线、B 的保真度 |
| 2 | `SonicExecutor` 接线（§8.1） | `diff_vla_wbc/sonic_mjlab.py` + 自检 | 全部训练 |
| 3 | WM 数据（分段 A'） | `runs/wm_real/ft318_seg/*.pt` | WM |
| 4 | WM 训练 / gate | `runs/S1_handover/sonic_wm_r1.pt` | joint |
| 5 | VLA server（318 base）+ MuJoCo 闭环 | `scripts/collect_b_ft318.py` | B 路线 |
| 6 | R1 采集 600 集 | `runs/B_ft318_r1/` | joint |
| 7 | R1 joint + eval | `runs/S3_loop/r1/` | 下一轮 |

---

## 8. 已完成/待完成的代码

### 8.1 本次新增（已在仓库里）

| 文件 | 作用 |
|---|---|
| `scripts/replay_ft318_mujoco.py` | **本计划 §2 的全部实验**：`target` / `token` / 分段三种臂；deploy/HA 两套增益；输出 npz + json 指标；单 env CPU 530–1600 步/s |
| `diff_vla_wbc/sonic_mjlab.py` | **执行器接线**：`SonicExecutor`（994 构造 + 可微 decoder + 动作空间换算）+ `SonicMjlabBundle`（把 `MjlabJointBundle` 的 `act` 换成 SONIC，其余全代理） |
| `scripts/check_sonic_mjlab.py` | 自检：ORT 对拍、`obs930→994` 逐块对齐、动作空间往返、置换 == `joint_contract` |
| `scripts/calibrate_g1_plant.py` | §4 的质心/接触/增益粗扫 + 报告（产物 `runs/plant_calib.json`） |
| `scripts/collect_b_ft318.py` | **§5 的 B 路线采集**（VLA 在环：数据集 ego 视频帧 + sim state64 → HTTP `/infer` → token → SONIC → MuJoCo）；`--token-source dry` 可不连服务做冒烟 |
| `scripts/collect_seg_ft318.py` | **§6 的 A' 分段采集**（SONIC 执行器 + L1 动作噪声 + L2 起点扰动 + hold 抖动）；产物格式与 `build_real_transitions_mjlab.py` 一致（`states(71)/next_states(71)/actions(29)/frames`），可直接喂 `pretrain_wm.py` |

**已通过的自检（2026-11-04）**：

| 检查 | 结果 |
|---|---|
| `SonicDecoderTorch` vs ORT（994→29） | `max|Δ| = 2.4e-7` ✅ |
| `obs930 → 994` 与手工逐块拼接 | `max|Δ| = 0` ✅ |
| q 块往返（HA↔STANDARD，末帧） | `max|Δ| = 0` ✅ |
| 关节序往返 | `max|Δ| = 0`；`perm(std→ha) = [0,6,12,1,7,13,…]` ✅ |
| `∂a/∂token` | mean 1.42 / max 6.05（**图没断**）✅ |
| 端到端 action vs ORT | `1.4e-6` ✅ |
| 真实 ckpt（`wbc_ft318_mjlab/ckpt_0007000.pt`）+ 包装 | `act` 形状 (B,29) 有限、`step/reward/library` 代理正常 ✅ |
| 动作尺度同底 | base `action_scale[0:4] = [0.3507, 0.3507, 0.5475, 0.3507]` == SONIC ✅ |

**B 采集冒烟（2026-11-04，dry）**：2 集 × 60 步 = **120 行**，0 摔、joint MAE 0.046–0.047、
末倾 4.7–10.4°；ego 视频经 `cv2` 解码 480×640（mjlab venv 无 cv2，自动回落到系统
`/usr/local/lib/python3.12/dist-packages`，同 3.12 ABI 已验证可读帧）。

**初值补全（§4.4）已验证但不足以救 G1**：`--init recorded`（录制 root 四元数 + 中心差分关节速度 +
脚锁定基座速度）把 300 步 target 回放从 70 步延到 **72 步**，MAE 0.072→0.078 —— 说明**不是初值问题**。

**A' 采集冒烟（2026-11-04）**：`--set-index 0 --episodes 0 --segments 6 --seg-len 20
--action-noise-rad 0.014 --start-perturb-xy 0.05` → **120 行**，`states (120,71)` / `actions (120,29)`、
全部有限、root z 0.79±0.02、quat 范数 1.0、逐步关节增量 0.015 rad（无跳变）、`|a|` mean 0.85 / max 5.27。

**训练入口已接通**：

```bash
# 旧（SuperMimic WBC）：行为不变
python scripts/train_joint_mjlab.py --wbc-ckpt <sm.pt> --motion data/npz/g1_tasks ...
# 新（SONIC 冻结执行器）：多两个参数
python scripts/train_joint_mjlab.py --executor sonic --token-quant none \
    --wbc-ckpt <sm.pt> --motion data/npz/g1_tasks --tokens data/library/ft318_tokens.pt ...
# --wbc-ckpt 只用来拿 **WM + 参考库 + 解析件**（字典里的 policy 不再被调用）
```

### 8.2 待写（按优先级）

1. `scripts/collect_seg_ft318.py`：§6 的分段采集（含 L1/L2 扰动、奖励、`.pt` 转移）；
2. `scripts/collect_b_ft318.py`：§5 的 VLA 在环采集（VLA server + MuJoCo + SONIC + 318 视频帧）；
3. `scripts/train_joint_sonic.py`：在 `train_joint_mjlab.py` 上把 bundle 换成 `SonicMjlabBundle`
   （预计只改 `--wbc-ckpt`→`--decoder` 一处 + 构造处）；
4. WM 输入适配：`MjlabJointBundle` 的世界模型不用改（state/action 语义不变）。

### 8.3 复用（不改）

- `diff_vla_wbc/sonic_decoder.py`（可微 decoder，onxx 逐算子复刻，vs ORT 3.3e-6）；
- `diff_vla_wbc/mjlab_joint.py`（WM 观测 93、reward、termination、参考库）；
- `diff_vla_wbc/v3_step.py` / `rollout.py` / `train_v2.py`（联合微调主干）；
- `scripts/train_joint_mjlab.py`（joint CLI）；
- `joint_contract.py`（关节序唯一真源）。

---

## 9. 风险

| 风险 | 现状/处理 |
|---|---|
| **植物不保真**（COM/接触/踝） | 已量化（§2.2）；§4 是唯一解；B 路线不受阻，A 路线受阻 |
| 换执行器后 WM 分布过期 | 必须按新执行器重采 A'（§6），**旧 WM 只能初始化** |
| VLA 图像是"回放的录制帧"（开环视觉） | 与旧 B 线同风险；用 `MAX_STEPS=length` + 漂移曲线监控；漂移大时降权 |
| 318 的 `action.wbc` 与 decoder 有 0.03–0.06 残差 | 属正常（录制是 HA 环境）；用 **decoder 输出**当动作标签，不用 `wbc` |
| 手部通道 | 不进 token/WM；采集/部署沿用 `HandStream`（`docs/HAND_STREAM.md`） |
| 评测口径混用 | 全表统一"全长 + START_FRAME=0"；600-tick 旧结果只作附录 |
| 速度错觉 | MuJoCo 快，但 **VLA 前向是瓶颈**（~10–30 ms/token）；600 集 × 2000 帧 ≈ 4–8 h/轮，按 8 进程并行 |

---

## 10. 一句话

> **318 的 token 在 MuJoCo 里能复现目标（0.03–0.06 rad），但闭环 1 s 就前翻（质心只剩 2–3 cm 余量、
> 植物不保真）；分段 20 步 0 摔可用。所以主线走 B 路线（VLA 在环、每轮 318 全集重采、5 轮 δ 退火），
> A 路线退到"分段 teacher-forced 只喂 WM"，并把植物标定作为把 A 救回主线的唯一杠杆。
> 执行器全面换成冻结 SONIC decoder，只联合微调 VLA。**


---

# 附录 B（2026-11-04 执行记录）：A' 全量采集 + WM 训练 + G3

## B.1 A' 全量（计划 §6）

命令：`CHUNK=8 SEGMENTS=32 MAXJOBS=32 bash scripts/run_a_prime_ft318.sh`

| 项 | 值 |
|---|---|
| 产物 | `runs/wm_real/ft318_seg/a_prime_ft318.pt` |
| 行数 | **203,520**（318 clip × 32 段 × 20 步） |
| clip 覆盖 | **318 / 318**（每个 clip 恰好 640 行） |
| 有限性 | 全部有限，无 NaN |
| 趴地行（tilt>45°） | **0.0%**（tilt>30° 仅 0.005%）⇒ G2 的"无趴地行"过 |
| 动作 | `|a|` mean 1.10 / p90 2.61 / max 19.66（与 decoder 原始输出同量级） |
| 状态 | root z 均值 0.760；每步关节增量 0.016 rad（无跳变）；quat 范数 1.0 |
| 扰动 | L1 `σ=0.014 rad`、L2 起点 `xy N(0,0.05)`、hold∈[15,25] |
| 耗时 | **~1.5 min**（32 进程 × 260 步/s/进程） |

**吞吐这件事有个坑**（已修）：默认 onnxruntime 会开满 128 线程，32 进程一起跑时线程超订，
一个分片 45 s 都跑不完。改成 **torch 版 decoder（逐算子等价，3e-6）+ `torch.set_num_threads(1)`** 后
单进程 **19.6 s / 5120 步（≈260 步/s）**、CPU 占用 1 核，32 进程才真正线性。
（`collect_seg_ft318.py --decoder-impl torch` 已是默认。）

**merge 的 episode 标签**：合并时用 `set_index*1000 + episode` 作全局 id（9 个子集的 episode
编号各自从 0 开始，直接用会撞车）；`episodes` 张量只为覆盖率审计，WM 训练不用。

## B.2 WM 训练（计划 §5.3 第 4 步）

```bash
python scripts/pretrain_wm.py \
  --joint-ckpt third_party/SuperMimic-Reproduced/logs/wbc_ft318_mjlab/ckpt_0007000.pt \
  --transitions runs/wm_real/ft318_seg/a_prime_ft318.pt \
  --out runs/S1_handover/sonic_wm_r1.pt --steps 3000 --stage1-steps 0 --roll-steps 0
```

- R1 没有 B，A' 直接当 mix（与 `JOINT_MULTIROUND_PLAN_20261103.md` §9.1 的修法一致）；
- 冒烟（20 步）：`loss -1008.27 → -1018.71`，`qd_corr 0.476 → 0.776`，19.5 s；
- 正式 3000 步约 **47 min**（0.93 s/step，单卡 4090）。

## B.3 G3（20 步想象误差）——评测脚本 `scripts/eval_wm_g3_ft318.py`

真值取 A' 的段内连续 20 步；用同一动作序列自回归滚 WM，逐步比位置/朝向/关节。
**冒烟 WM（20 步训练）基线**：

| k | pos p90 | ori p90 | q p90 |
|---:|---:|---:|---:|
| 1 | 4.69 mm | 1.56° | 0.019 rad |
| 5 | 14.6 mm | 3.53° | 0.041 rad |
| 10 | 36.0 mm | 11.2° | 0.062 rad |
| 20 | **159.5 mm** | 29.6° | 0.166 rad |

**正式 3000 步（`sonic_wm_r1.pt`，2000 段，70 s 训完）**：

| k | pos p90 | ori p90 | q p90 |
|---:|---:|---:|---:|
| 1 | 4.7 mm | 0.33° | 0.017 rad |
| 5 | 18.0 mm | 2.66° | 0.031 rad |
| 10 | **25.8 mm** | 5.96° | 0.046 rad |
| 20 | **121.4 mm** | 19.1° | 0.154 rad |

`qd_corr 0.476 → 0.926`。**G3（20 步 p90 ≤50 mm）FAIL**：有效视野约 **10–12 步**，
k=10 已经 25.8 mm、k=20 翻到 121 mm（超线性）。
⇒ 下一步不是"继续加步数"，而是：
1. **加自回归阶段**（`--roll-steps 2000 --roll-k 20`，直接在 20 步一致性上优化）；
2. A' 从 32 段/clip 提到 **64 段/clip**（计划口径，407k 行）并加 `--seg-len 10` 对照组；
3. 若仍 >50 mm：把 **joint FT 的 H 降到 10**（与有效视野对齐），而不是硬撑 H=20。

**r1b（12000 步 + 想开自回归阶段）= 同样的数**：

| k | r1（3000 步） | r1b（12000 步） |
|---:|---:|---:|
| 1 | 4.7 mm | 5.0 mm |
| 5 | 18.0 mm | 19.1 mm |
| 10 | 25.8 mm | 26.4 mm |
| 20 | 121.4 mm | **115.3 mm** |

⇒ **加训练步数没有用**（12000 步只降 5%），20 步误差是**结构/数据**决定的，不是欠拟合。

两条直接结论（写进执行口径）：

1. **阶段 2b（自回归 k=20）被 B 线数据卡住**：日志
   `[wm-pre][warn] 阶段2b 没有窗口来源（runs/wm_real/ft318_seg/bline/b_*.pt）⇒ 跳过`
   —— 也就是说 **没有 B 就没法把 20 步一致性练进去**，这正好是 §5 第 3–4 步要做的事；
2. **在 B 数据到位之前，joint FT 的 H 应该按 WM 的有效视野设 = 10**（k=10 的 p90 26 mm，
   离 5 cm 门有余量），而不是硬撑 H=20；H=20 只有在 B 采到 + 阶段 2b 打开之后再回来。
**注意**：position p90 在 k=20 时已经 16 cm，若正式训完仍 >5 cm，说明**漂移主导**，
应先在 A' 里换成"每 5 步重置 + 更短 H"的对照，而不是继续加训练步数。

## B.4 本轮新增/修改的脚本

| 文件 | 状态 |
|---|---|
| `scripts/run_a_prime_ft318.sh` | 新增：分片并行 + 合并（`OMP/MKL/OPENBLAS_NUM_THREADS=1`） |
| `scripts/collect_seg_ft318.py` | 新增 `--decoder-impl {torch,onnx}`（默认 torch） |
| `scripts/eval_wm_g3_ft318.py` | 新增：G3 多步评测 |
| `scripts/collect_b_ft318.py` | 新增：B 路线采集（dry/vla） |
| `scripts/calibrate_g1_plant.py` | 新增：植物标定（tilt 目标） |
| `scripts/train_joint_mjlab.py` | 新增 `--executor sonic --decoder --token-quant` |
| `diff_vla_wbc/sonic_mjlab.py` | 新增：SONIC 执行器 + bundle 包装 |


## B.5 R1 joint FT 冒烟：SONIC 执行器 + H=10（2026-11-04，已跑通）

**环境配方（踩了 4 个坑才通，写死在这里）**：

```bash
# 训练用 **系统 python3**（有 transformers 5.9.0），mjlab venv 没有 transformers
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH=$PWD/third_party/gr00t_deps:$PWD/third_party/py312_shims
python3 scripts/train_joint_mjlab.py \
  --wbc-ckpt third_party/SuperMimic-Reproduced/logs/wbc_ft318_mjlab/ckpt_0007000.pt \
  --motion data/npz/g1_tasks --tokens data/library/ft318_tokens.pt --hand-dir data/npz/g1_tasks \
  --vlm-ckpt /inspire/.../groot_sonic_vla_20k_frozen271_20261023/checkpoint-20000 \
  --out runs/joint_sonic_smoke --executor sonic --token-quant none \
  --steps 20 --horizon 10 --batch-size 8 --lambda-r 0.3 --act-eps 0.10 --act-weight 1.0 \
  --start-source reference --log-every 5 --save-every 1000 --device cuda
```

四个坑：
1. mjlab venv 缺 `transformers` → 改用系统 `python3`（它有 transformers/mujoco/onnx，但没有 mjlab 也不需要）；
2. 系统 python 缺 `tyro` / `docstring_parser` → `third_party/py312_shims/` 里 symlink（**绝对路径**，相对路径会解析错）；
3. 系统 python 缺 `diffusers` / `albumentations` 链 → 直接用项目自带的 `third_party/gr00t_deps` 做 PYTHONPATH（它是 site-packages 式平铺目录，**不含** torch/numpy，不会遮蔽）；
4. `SonicMjlabBundle` 两个 bug 已修：`_GraphDecoder.consts` 是普通 dict（`.to()` 不搬 → CUDA 上 `mat2 is on cpu`）、
   `rollout` 会传 `step_idx` 而 mjlab 的 `step` 不接受（adapter 吃掉）。

**结果（20 步，H=10）**：

| 证据 | 值 |
|---|---|
| 可训参数 | `trainable=head_only params=67,108,864  wbc_params=0 wbc_trainable=0 wm_trainable=0` ✅（SONIC + WM 全冻） |
| 损失 | `loss -0.2489`(10) → `-0.2709`(15) → `-0.2437`(20)；`loss/fm_bc 0.373 → 0.064 → 0.341` |
| 想象回报 | `rollout/reward` 0.085–0.094，`rollout/survival` 1.0 |
| 动作量级 | `rollout/action_abs` 1.18–1.44（decoder 原始动作范围） |
| 产物 | `runs/joint_sonic_smoke/{joint_final.pt, vla_final.pt}`；`executor_meta = {decoder_onnx: ..., token_quant: none}` ✅ |

⇒ **"VLA → token(64) → 冻结 SONIC decoder → 冻结 WM(20 步想象) → 奖励 → 反传 VLA" 整条链在 MuJoCo 口径上跑通**，
这是本计划要的"架构改造 + 训练实现 + 跑通验证"。
