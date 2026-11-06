# 给 IsaacLab / trackerlab 侧的问题清单：root x/y、yaw 与相关设计

> 写于 2026-10-30。MuJoCo/mjlab（SuperMimic 复现）侧在把 318 数据集接进
> 「token → WBC(decoder) → 闭环 → 联合微调」这条链路时，踩到的一系列**数据契约级**
> 的问题。它们大多**不是实现 bug**，而是"数据里没有那个量"导致的设计约束。
>
> 我们把每条的**实测证据**、**试过的解法及其结论**、**最终的处置**都写下来，
> 并在每一节末尾列出**希望 IsaacLab 侧确认的具体问题**（他们很可能面对同一批问题）。
>
> 所有数字都可用仓库里的脚本复现（脚本路径写在每节末尾）。

---

## 0. 摘要：六条，按严重程度排

| # | 问题 | 性质 | 我们的处置 |
|---|---|---|---|
| 1 | **数据集里完全没有 root x/y**（473,876 帧，x/y 恒为 0） | 数据缺失 | 用官方 planner 重建（唯一可用路线） |
| 2 | **绝对 yaw 对策略不可观测**（93 维观测里没有 yaw） | 架构限制 | 保留奖励项、接受它只能当先验；官方发布版同样如此 |
| 3 | **绝对位置对策略不可观测**（93 维里没有位置/线速度） | 架构限制 | 同上；官方也如此（992 维 decoder 观测无位置） |
| 4 | **位置奖励项在落后 >0.45 m 时完全饱和** ⇒ 欠走自我强化 | 奖励几何 | 待改：σ 0.30 → 0.55 |
| 5 | **318 语料 72% 是原地片段**（clip 平均速度 p50 = 0.019 m/s） | 数据配比 | 待改：走路 clip 加权 ×2.5 |
| 6 | **评测 spawn 逻辑在"手臂插进桌子"时把整机抬高 87 cm** ⇒ 悬空摔倒 | 评测 bug（已修） | 已修；建议 IsaacLab 检查同类逻辑 |

**最要紧的一句**：第 1 条如果不解决，`motion_global_root_pos` 这一项在训练里
教的不是"走过去"，而是"留在原点"——它会**主动摧毁**走路能力（实测步幅塌 4.4 倍）。

---

## 1. 问题 1：数据集里完全没有 root x/y

### 1.1 实测证据

`data/library/ft318_compact.pt`（由 318 条 LeRobot 录制导出，473,876 帧）：

```
ref[:, 0:2]（root x/y）:  x span = 0.000000 m,  y span = 0.000000 m   （全部帧）
ref[:, 2]  （root z）  :  [0.378, 0.756] m                            （有值）
ref[:, 3:7]（root quat）: 有值
ref[:, 7:36]（29 关节） : 有值
ref[:, 36:42]（root v/w）: 恒为 0
```

源 LeRobot 日志的字段（官方部署栈 `gear_sonic/run_data_exporter.py` 产出）：

```
observation.state            (43)   关节角（29 身体 + 14 手）  ✅ 真值
observation.root_orientation (4)    根朝向                    ✅ 真值
observation.eef_state        (14)   手腕 pos(3)+quat(4) ×2    ⚠️ 是**基座相对**（z 可到 -0.11）
observation.projected_gravity(3)    重力方向                  ✅
action.wbc                   (43)   关节目标                  ✅ 真值
action.motion_token          (64)   录制 token                ✅ 真值
teleop.planner_movement      (3)    走路方向（22.9% 帧非零）    ✅ 真值（**意图**）
teleop.planner_facing        (3)    期望朝向                  ✅
teleop.planner_speed         (1)    走路速度（0.117–0.60 m/s） ✅ 真值（**意图**）
teleop.planner_mode/stream_mode      模式
observations.images.ego_view        视频
—— 没有 root position，也没有 smpl_transl ——
```

**结论：不是我们导出时丢的，是官方采集器本来就不记。** 设计上位置归 planner 管
（`is_using_encoder_ = true` 的路径），策略（decoder）不消费位置。

而官方**训练 decoder** 用的参考是另一份工件：`.pkl` 运动文件（`MotionLib` 加载，
含 `global_translation`），是**采集时由 planner 实时产出**的。我们手里没有这批 `.pkl`
（`SONIC_DATA` 里只有 1 个人类 SMPL demo）。

### 1.2 我们试过的五条重建路线（全部实测，别再重复踩）

| 路线 | 做法 | 实测结果 | 判定 |
|---|---|---|---|
| **a. 指令积分** | `Σ speed·dt`（planner_speed 非零时） | 694–934 cm/12 s | ❌ **不是距离，是人的意愿**；重放只到它的 **0.23 倍** |
| **b. planner 原始输出** | 用录制指令驱动 `planner_sonic.onnx`，逐 0.1 s 重规划，输出含 root 的全身 qpos | 112–317 cm/12 s；朝向对拍（vs 录制 `root_orientation`）**MAE 5–7°** | ✅ **唯一可用**（官方同源） |
| c. planner + 整集配速缩放 | 把 b 的路径缩到 a 的长度 | 缩放因子 0.46–1.0，事后发现**基准（a）本身是错的** | ❌ 用错误基准做校正 |
| d. 逐窗口配速缩放 | 每 5 s 滑窗缩放 | **爆掉**：放大 2.7–8 倍，路径 p90 从 500 → 1371 cm | ❌ 有除法，小分母发散 |
| e. 开环物理重放 | 把录制 `action.wbc` 当 `q_target` 灌进 MuJoCo | **机器人倒地**（tilt 中位 90°、朝向 MAE 86°、净位移全部朝同一方向 R=0.97） | ❌ 开环关节 PD 没有平衡控制器 |
| f. 闭环重放（`--arm gt`） | 用**录制 token** 驱动我们训好的 WBC 闭环跑 | 平衡、可达 | ⚠️ 可用，但那是"我们自己策略的输出" ⇒ 当参考会变成自模仿（学不到新东西） |

**另一个必须记住的坑**：路线 e 产生的"重放/planner = 0.65"曾被我们当成
"参考与关节不自洽、路径比天花板只有 0.65"。**这个结论作废** —— 它测的是
"躺在地上的机器人蹭了多远"。**不要引用那个 0.65。**

### 1.3 最终处置

```
参考的 x/y  := planner 原始输出（路线 b），不缩放
             := `data/library/ft318_compact_v8.pt`（= ft318_compact_v2.pt，逐位相同）
只填 x/y，z / quat / 29 关节 / 关节速度 逐位保持录制真值
每集按"录制朝向首帧"做刚体归一化（旋转到 yaw=0、平移到原点）
```

验收：朝向对拍 MAE **7.0°（p90 11.1°）**；归一化后首帧 yaw = 0.00°、x/y 精确为 0。

复现：

```bash
PY=third_party/SuperMimic-Reproduced/.venv/bin/python
$PY scripts/probe_planner.py --check        # planner ONNX I/O 与语义
$PY scripts/probe_planner.py --smoke        # idle 1.4cm/2.1s；walk 默认 0.573 m/s
$PY scripts/build_ref_with_planner.py --workers 7 --library data/library/ft318_compact.pt \
    --out data/library/ft318_compact_v8.pt --report runs/ref_v8/report.json   # 约 2 分钟
```

### 1.4 想请 IsaacLab 确认的问题

1. **你们的训练参考（motion pkl / MotionLib）里，root 的 x/y 是从哪来的？**
   是 planner/retarget 的输出，还是 SMPL 的真实平移（`smpl_transl`）？
2. 如果你们的 LeRobot 录制数据**也没有** root position（我们判断这是官方设计），
   那你们训练 decoder 时用的参考是不是完全没有这个字段？
3. 你们有没有**采集时把 planner 输出也记下来**？
   我们在采集侧的建议（还没做）：exporter 多写一列 `planner_qpos_future (64×36)`。
   它本来就在内存里（encoder 的 `g1_qdq_future` 就是它的切片），加一列**永久解决**这个缺失。
4. 你们是否也用 `tracking_anchor_pos`（官方 `reward/tracking/base.yaml`，weight 0.5, std 0.3）？
   如果参考的 x/y 不可靠，这一项在你们那边是怎么处理/降权的？

---

## 2. 问题 2：绝对 yaw 对策略不可观测

### 2.1 观测契约（我们与官方**逐项一致**）

```
decoder 观测 (994) = token(64) + 10 × [ base_ang_vel(3), q(29), dq(29), last_action(29), proj_gravity(3) ]
                    ↑ 我们用 930 + 64 = 994，项序不同但内容完全相同
```

`proj_gravity` 只含 roll/pitch ⇒ **任何绝对 yaw 都不在观测里**。

### 2.2 token 里的朝向：官方三种 anchor 模式（`g1_deploy_onnx_ref.cpp:610-692`）

```cpp
// 0: full base quaternion          (motion_anchor_ori_b)            ← 官方发布用这个
// 1: robot heading only            (motion_anchor_ori_heading)
// 2: reference first-frame heading (motion_anchor_ori_refheading)
left_quat = base_quat / heading(base_quat) / heading(ref@current_frame)
anchor    = rot6d( conj(left_quat) ⊗ ref_future )
```

| 模式 | token 里含什么 | VLA 能否预测 |
|---|---|---|
| **0 `_b`** | **tilt 误差 + yaw 误差**（相对机器人自身） | 需要从图像推断机器人朝向 |
| 1 `_heading` | 只有 yaw 误差 | 同上 |
| 2 `_refheading` | **纯参考内部旋转**，完全不含机器人状态 | ✅ 只依赖参考 |

**关键**：模式 0/1 **是带着 yaw 误差的**；模式 2 才是把 yaw 误差去掉。
（我们一度说反过，以本节为准。）

### 2.3 那 yaw 到底谁管

```
遥操作/采集：encoder 手里有机器人真实四元数 ⇒ token 正确 ⇒ decoder 能纠
VLA 推理 ：VLA 只知道 proj_gravity ⇒ 只能**看着 ego_view 用视觉推断朝向**
```

**官方发布版同样是模式 0、同样没有 yaw 观测**，所以它也是靠视觉兜。
`_refheading` 只是代码里留的选项（注释说给 `g1_dyn decoder in the refheading setting` 用），
我们手里**没有那份权重**（`SONIC_DATA` 里没有任何 `*dyn*` / `*refheading*`）。

### 2.4 我们这边的处置

- 联合微调奖励把朝向拆成三项（`tilt 1.0 / yaw 0.2 / yaw_rate 0.3`，σ 0.25）；
  依据是"可观测性"：`proj_grav` 直接可见 tilt（满权重），绝对 yaw 只有 ~23% 可学（降权），
  "转向"的正确落点是 **yaw 角速度**（`w_b` 在观测里）。
- **不改 token anchor 模式**（换模式要重训 encoder + 重训 VLA 的 action head，
  且模式 2 会丢 heading 误差通道 ⇒ 不是免费修复）。

### 2.5 想请 IsaacLab 确认的问题

1. 你们训练 decoder 时用的是哪种 anchor 模式？是不是也是 `motion_anchor_ori_b`（模式 0）？
2. 如果是模式 0：**你们怎么处理"VLA 无法观测 yaw"这件事？**
   - 靠视觉（我们猜是这样）？
   - 还是给 VLA 加了 anchor/朝向输入？（注意官方 `unitree_g1_sonic` 的 state 是 8 项，
     不含 `motion_anchor_orientation` —— anchor 只在 encoder 路径上）
3. 你们有没有试过 `_refheading`（模式 2）的权重？效果如何？
4. 你们的绝对 yaw 误差在闭环里是什么量级？会不会累积（因为 token 是 yaw 相对的）？

---

## 3. 问题 3：绝对位置同样不可观测（与 yaw 同源）

```
93 维观测 = [ q−q0(29), qd(29), w_b(3), proj_grav(3), a_prev(29) ]
            没有位置、没有线速度
```

后果：**奖励可以惩罚位置误差，但策略无法用它做反馈**，只能学一个"先验"。
官方 decoder 完全一样（994 里没有位置/线速度），它靠两样东西在系统层面兜：

| 兜法 | 机制 | 我们的状态 |
|---|---|---|
| **kinematic planner** | 每 0.1 s 用**真实世界位姿**（`context_mujoco_qpos[1,4,36]` 含 root 位置）重规划 | ❌ 不在 VLA 路径上（官方也是：外部 token ⇒ `is_using_encoder_ = false`） |
| **视觉** | VLA 看 ego_view 自己判断离目标多远 | ⚠️ 有，但监督质量取决于 token 训练 |

**注意**：官方 encoder 输入里有 `motion_root_z_position`（高度），但**没有 x/y** ——
这正是"x/y 由 planner 负责"的证据。

### 想请 IsaacLab 确认的问题

1. 你们的 VLA 推理路径是不是也把 planner 关掉了（`is_using_encoder_=false`）？
2. 如果是：**你们靠什么让机器人"走到正确位置"？** 视觉？还是把位置误差塞进观测？
3. 你们有没有试过"给 decoder 加 3 维位置误差条件输入"（`R_rootᵀ(p_ref − p_robot)`，
   真机上由里程计/视觉给出）？效果如何？

---

## 4. 问题 4：位置奖励在"落后"时完全饱和 ⇒ 欠走自我强化

官方口径（也是我们现在 WBC 训练用的）：

```yaml
tracking_anchor_pos:  weight 0.5   std 0.30     # = 0.5 · exp(-e²/0.30²)
```

代入实测落后量：

| 落后 e | 该项得分（满分 0.5） | 占正奖励预算（≈5.0） |
|---|---|---|
| 0.20 m | 0.32 | 6.4% |
| 0.30 m | 0.18 | 3.7% |
| **0.45 m** | **0.05** | **1.0%** ← 实际上已死 |
| **0.60 m** | **0.009** | **0.2%** |

而我们的实测 `pos p50` = **0.47 / 0.53 / 0.75 / 1.17 m**（4 个窗口）——**全在死区**。
一旦落后超过 ~0.45 m，奖励里**没有任何东西在奖励它往前走**，而关节项站着就能满足
⇒ **欠走是自我强化的**。

实测后果（同一批 token、同一把尺子）：

```
干净基座（bones_seed，60% 走路样本）  ：路径比 1.51   ← 走过头
318 微调 8k 之后                     ：路径比 0.34   ← 走不够（4.4 倍塌陷）
```

**这不是"忘了怎么走"，是"训练里没有东西奖励它走"。**

处置（待做）：`std_root_pos 0.30 → 0.55`（落后 0.55 m 时仍有 0.37 的得分）。
我们的**联合微调**奖励里另有一套尾部铰链（`-λ·relu(|Δp| − e₀)`，λ2.0、e₀0.35m），
实测能把 p90 砍 48%，但那是另一份奖励（见 §6），WBC 训练走的是官方口径、没有铰链。

### 想请 IsaacLab 确认的问题

1. 你们的 `tracking_anchor_pos` 用的是 std=0.3 吗？闭环里位置误差的典型量级是多少？
2. 如果误差经常 >0.45 m，你们有没有遇到我们这种"欠走"？
   有没有加尾部铰链（线性惩罚）之类的机制？

---

## 5. 问题 5：318 语料 72% 是原地片段

| 运动含量 | `ft318_compact_v8` | `bones_seed`（我们的基座） |
|---|---|---|
| 帧级速度 >0.15 m/s | **12.8%** | 39.9% |
| 逐 clip 平均速度 p50 | **0.019 m/s** | 0.156 m/s |
| clip 平均速度 >0.1 m/s | **28.0%** | 60.4% |
| clip 平均速度 >0.2 m/s | 12.9% | 44.8% |

在 72% 的样本教"别动"的情况下，微调会把走路能力往下拉（见 §4 的 1.51 → 0.34）。

处置（待做）：给走路 clip 加权（`clip mean speed > 0.1 m/s` 的加权 ×2.5，
采样占比 28% → 约 50%）。

### 想请 IsaacLab 确认的问题

1. 你们用 318（或同类任务数据）微调时的**采样策略**是什么？均匀？按速度？按任务？
2. 你们有没有观察到同类"微调后步幅变小"的现象？

---

## 6. 附：我们的两份奖励（避免混淆）

很多讨论的混乱来自"两份奖励被当成一份"。它们在代码里是**分开的两套**：

| | WBC(decoder) 训练 | VLA 联合微调 |
|---|---|---|
| 定义位置 | `supermimic/envs/tracking_env_cfg.py`（mjlab `RewardTermCfg`） | `supermimic/envs/mdp/rewards.py::tracking_reward_terms` + `TrackingRewardCfg` |
| 谁在用 | mjlab 训练环境 | 联合微调的 **WM 想象 rollout**（`heads.reward ← bundle.reward ← rollout.py`） |
| 内容 | 官方 8 项：anchor_pos .5/σ.30、anchor_ori .5/σ.40、body(pos/ori/lin/ang) 1/1/1/1、action_rate −.1、joint_limit −10 | 我们的：root_pos 1.0/σ.20、root 朝向拆 tilt1.0/yaw0.2/yaw_rate0.3（σ.25）、body 四项 1/1/1/1、action_rate −.1、joint_limit −2 + **4 条尾部铰链** |
| `max_step_reward` | 0.10 | 0.13 |
| 缺项 | `undesired_contacts`（−0.1，需要接触传感器，未实现） | — |

`REWARD_TERM_NAMES`（`imagination/compiled.py`）是**写死**的项名元组，改奖励项必须同步，
否则重新训 WBC 一启动就 `KeyError`（2026-10-30 实际踩过）。

---

## 7. 附：评测 harness 的一个 bug（已修，建议 IsaacLab 自查同类逻辑）

原来"出生时把脚从地板里抬出来"的逻辑是：

```python
for _ in range(4):
    deepest = min(c.dist for c in data.contact)   # 任何接触里最深的穿透
    data.qpos[2] += -deepest + 1e-3               # 把整机往上抬"那么深"
```

**问题**：如果初始姿态把**手臂插进了桌子**（穿透 −34.9 cm），整机会被抬高 **87 cm**
⇒ 出生高度 1.649 m（正常 0.78）⇒ 自由落体 ⇒ 摔成一坨（tilt 94–115°）。

这直接污染了历史数字：**"摔倒"和部分"坏集"里有一部分是这个伪影**，
而且它让某集的路径比虚高到 0.78（躺在地上滑出来的），修复后真实值只有 **0.10**。

修法：单次抬升 ≤3 cm、累计 ≤5 cm，超限只告警不再抬高度。
建议 IsaacLab 检查自己的 reset/spawn 里有没有同类"无条件按穿透量抬升"的写法。

---

## 8. 我们现在定版的配置（供对照）

```
参考库    data/library/ft318_compact_v8.pt      （planner 原始 x/y，不缩放）
          = ft318_compact_v2.pt（逐位相同）
npz 参考  data/npz/g1_tasks_v8                  （318/318，与库对齐误差 0.00e+00）
WBC 训练  干净基座 e9_0024000（bones_seed，未受"x/y≡0"污染）+ 官方 8 项奖励
          LR：4 卡 × 1e-5 = 有效 2e-5；cosine，total 20000
联合微调  奖励 = §6 右列（**不动**）；critic：`CRITIC_SOURCE=real` + 目标 MC（可选 TD(λ)）
评测      `--ref` 全部指向 v8
```

复现整套「重建参考 → 校验」：

```bash
PY=third_party/SuperMimic-Reproduced/.venv/bin/python
$PY scripts/build_ref_with_planner.py  --library data/library/ft318_compact.pt \
     --out data/library/ft318_compact_v8.pt --report runs/ref_v8/report.json
python3 scripts/patch_npz_root_xy.py --library data/library/ft318_compact_v8.pt \
     --src data/npz/g1_tasks --dst data/npz/g1_tasks_v8
python3 scripts/eval_oracle_path.py --ref data/library/ft318_compact_v8.pt --dump '<B线 dump>'
```

---

## 9. 一句话总结（给 IsaacLab 的核心问题）

> **这批数据里没有 root x/y，也没有任何绝对位置的观测量。**
> 官方发布版同样如此——它的 x/y 由 planner 在生产参考时给出，deploy 时由视觉兜住。
> **问题在于：当你要把"位置"变成一条训练奖励时，参考的 x/y 必须可靠；
> 而这份参考只能靠 planner 重建，且它的绝对尺度无法用数据验证。**
>
> **所以我们最想知道 IsaacLab 那边三件事：**
> 1. 你们的训练参考 `.pkl` 里 root x/y 到底来自 planner 还是 SMPL 真实平移？
> 2. 你们的 decoder 训练有没有 `tracking_anchor_pos`？参考不可靠时怎么处理？
> 3. 你们闭环里"走到正确位置"是靠视觉、靠 planner、还是靠给 decoder 加位置输入？
