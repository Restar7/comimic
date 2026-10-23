# 手部 14 维独立通道（decoupled hand stream）

> 结论先说：**手部不走 SuperMimic**。身体继续用 29 维动作（`ACT_DIM=29`，跟踪 14 个刚体、最远到 `wrist_yaw_link`），手部是**独立的 14 维关节角通道**，由 `diff_vla_wbc/hand_stream.py` 驱动，15 个测试全过。
> 这套做法不是新发明，是对齐本工作区里 SONIC/`gear_sonic` 的现成设计，只不过那边早就实现、SuperMimic 这边没有。

---

## 1. 两条线的对应关系

| | SONIC / `gear_sonic` | SuperMimic（本文改造后） |
|---|---|---|
| 全身表示 | **64 维 motion token**（latent，`num_tokens × token_dim`） | **29 维关节目标**（无 tokenizer） |
| 手部 | **14 维关节角**，独立流 | **14 维关节角**，独立流（本模块） |
| 手部怎么下发 | `set_joint_position_target` + soft limit clamp | PD 力矩 `kp(q_des-q) - kd·q̇` + `actuatorfrcrange` clip |
| 是否改动作空间 | 不改（注释原文：*without changing SONIC's 29-DOF action space*） | 不改（`ACT_DIM` 仍是 29） |
| 数据前提 | motion 里要有 `hand_dof_pos`（43-DOF motion） | 有就用 `source="motion"`，没有就用 `source="pose"` 手型库 |

SONIC 侧原始实现（`gear_sonic/envs/wrapper/manager_env_wrapper.py`）：

```python
:97   if self._hand_joint_indices is None or len(self._hand_joint_indices) != 14:
:238  """Compute joint indices for mapping body (29 DOF) and hand (14 DOF) actions."""
:246  def _apply_motion_hand_dof_targets(self):
          """Apply the external hand stream without changing SONIC's 29-DOF action space."""
          hand_targets = self.motion_command.hand_dof_pos
          hand_targets = torch.clamp(hand_targets, min=limits[..., 0], max=limits[..., 1])
          robot.set_joint_position_target(hand_targets, joint_ids=self._hand_joint_indices)
```

另外 `gear_sonic/trl/modules/universal_token_modules.py:90` 注释直接写明 `hand_action_dim=None,  # For separate hand decoder (e.g., 14 hand joints)`，`:354` 默认 `"hand_action": 14`。

### 1.1 本仓库的接法：VLA 双输出，手部**不经过** SuperMimic

```text
VLM backbone ─┬─► flow-matching head ─► tokens [B,40,64] ─► MotionGoalAdapter ─► SuperMimic(WBC/WM) ─► 29 身体关节
              └─► hand head           ─► hands  [B,H,14]  ────────────────────────────────────────► 14 手部关节
```

- `diff_vla_wbc/hand_head.py`：`HandHead`（masked-mean 池化 + MLP，输出 `[B, H, 14]` 弧度）、`HandHead.loss`（SmoothL1）、`attach_hand_head`。
- `diff_vla_wbc/gr00t_vlm.py`：`Gr00tVLM(..., hand_head=...)` 与 `forward_with_hands()`，一次返回 `(tokens, hands)`；手部吃的是 action head 的 `backbone_features`，**不经过** adapter、世界模型和奖励。
- `diff_vla_wbc/hand_stream.py`：`HandStreamCfg(source="vla")` + `hand.command(hands)`，14 维**直接使用**，只在执行层 clamp 到关节限位。
- 同一份 backbone features 上还有 `diff_vla_wbc/goal_head.py` 直接回归 SuperMimic 的 10×68 物理 goal，
  也就是说 **VLA 不再经过 motion token**；理由与用法见 [`GOAL_INTERFACE.md`](GOAL_INTERFACE.md)。

运行时一共这几行：

```python
tokens, hands = vlm.forward_with_hands(inputs)     # 64-D token + 14-D 手部关节角
goal = adapter(tokens)                             # 只有 token 进入 SuperMimic
body_action = policy(goal, obs)                    # 29 维身体动作
hand.command(hands[:, 0, :])                       # 14 维直接下发，不经 SuperMimic
hand.step(dt)
ctrl = hand.assemble_ctrl(body_tau_29, data=data)  # 43 维（29 身体 + 14 手部）
```

训练时 `HandHead.loss(pred, target)` 单独反传；没有手部标签就不参与，身体分支完全不受影响（`forward_with_hands` 在没挂 hand head 时等价于原来的 `forward`）。

---

## 2. 43 维 ctrl 的真实布局（实测，别拼错）

`reproduce_scenes/g1/g1_29dof_with_hand.xml`：`nq=50`（7 浮动 + 29 身体 + 14 手指）、`nv=49`、`nu=43`。

| ctrl 下标 | 内容 |
|---|---|
| `0:22` | 左腿 + 腰 + 左臂（`G1_JOINT_NAMES[0:22]`） |
| **`22:29`** | **左手 7 个手指** |
| `29:36` | 右臂（`G1_JOINT_NAMES[22:29]`） |
| **`36:43`** | **右手 7 个手指** |

手指执行器是**插在左右臂之间**的，所以：

```python
ctrl = np.concatenate([body_tau_29, hand_tau_14])   # ❌ 错，29/14 都会错位
ctrl = hand.assemble_ctrl(body_tau_29, hand_tau_14) # ✅ 按名字散射，模块里已校验
```

关节地址同理：身体 qpos 不是 `qpos[7:36]`，而是 `[7..28] + [36..42]`（`G1BodyIndex` 已按名字解好）。

---

## 3. 手部关节与限位（来自 MJCF）

| 关节 | 左手范围 | 右手范围（镜像） |
|---|---|---|
| `thumb_0` | `[-1.0472, +1.0472]` | 同（对称） |
| `thumb_1` | `[-0.7243, +1.0472]` | `[-1.0472, +0.7243]` |
| `thumb_2` | `[0, +1.7453]` | `[-1.7453, 0]` |
| `middle_0` | `[-1.5708, 0]` | `[0, +1.5708]` |
| `middle_1` | `[-1.7453, 0]` | `[0, +1.7453]` |
| `index_0` | `[-1.5708, 0]` | `[0, +1.5708]` |
| `index_1` | `[-1.7453, 0]` | `[0, +1.7453]` |

右手是几何镜像，**同一个抓握手势在右手是反号**（`right = -left`），模块的 `HAND_POSES` 已按此生成。

---

## 4. 内置手型（`HAND_POSES`，单位 rad）

| 名字 | 左手指令（thumb_0/1/2, middle_0/1, index_0/1） |
|---|---|
| `open` | `0, 0, 0, 0, 0, 0, 0` |
| `grasp` | `0.55, 0.85, 1.25, -1.15, -1.35, -1.15, -1.35` |
| `pinch` | `0.90, 0.50, 1.45, -0.25, -0.35, -0.95, -1.25` |
| `fist` | `1.00, 1.00, 1.70, -1.50, -1.70, -1.50, -1.70` |

全部经过 `tests/test_hand_stream.py::test_pose_values_respect_joint_limits` 校验，落在关节限位内。

---

## 5. 用法

### 5.1 手型库模式（无手部数据也能用）

```python
import mujoco
from diff_vla_wbc.hand_stream import HandStream, HandStreamCfg, G1BodyIndex, load_hand_model

model = load_hand_model()                      # 43 执行器（29 身体 + 14 手）
data = mujoco.MjData(model)
body = G1BodyIndex(model)                      # 按名字解身体 qpos/dof/ctrl 下标
hand = HandStream(model, HandStreamCfg(pose="open", kp=12.0, kd=0.5, slew_rate=8.0))

hand.set_pose("grasp")                         # 平滑过渡（slew_rate 限速）
hand.step(0.02)                                # 每个控制周期推进一次
body_tau = kp * (q0 - body.q(data)) - kd * body.dq(data)   # 29 维，来自策略或 PD
ctrl = hand.assemble_ctrl(body_tau, data=data) # 43 维，手部力矩自动算好
data.ctrl[:] = ctrl
```

一条命令可直接看效果（不需要 checkpoint、不需要数据）：

```bash
python scripts/hand_stream_demo.py --poses open grasp open --hold 0.6 --save runs/hand_stream/demo
```

### 5.2 运动表模式（有 43-DOF 手部数据时）

```python
motion = np.load("hand_dof_pos.npy")           # (T, 14) 弧度，HAND_JOINT_NAMES 顺序
hand = HandStream(model, HandStreamCfg(source="motion", motion=motion, motion_fps=50.0))
q_des = hand.targets(frame)                    # 越界自动 clamp 到关节限位
```

### 5.3 29-DoF 模型上自动禁用

拿 `g1_29dof.xml` 构建时 `HandStream.enabled == False`，`pd_torque` 返回全零、`assemble_ctrl` 原样返回 29 维，**旧链路行为完全不变**。

---

## 6. 控制律与增益

```text
τ = kp · (q_des − q) − kd · q̇        # 再 clip 到 actuatorfrcrange
kp = 8.0 Nm/rad（默认）, kd = 0.4 Nm·s/rad
```

- 手指执行器力限：`thumb_0 ±2.45 Nm`，其余 `±1.4 Nm`（MJCF 的 `actuatorfrcrange`）。
- 增益是**初值**，不是实机标定值；demo 里用 `kp=12, kd=0.5` 收敛到 `err < 0.01 rad`。
- 位置伺服是逐关节 PD，不做指尖 IK；要精确抓取姿态时，在 `q_des` 层面做手指 IK 即可，接口不变。

---

## 7. 已验证 / 未验证

**已验证（`pytest tests/test_hand_stream.py -q` → 15 passed）**

- 43-DoF MJCF 的 14 个手指关节顺序、限位、执行器优先级；`hand_ctrl_ids == [22..28, 36..42]`。
- 身体执行器顺序 == `G1_JOINT_NAMES`，`ctrl` 按名字散射不重不漏。
- 右手镜像符号（`right = -left`）与所有手型都在限位内。
- 闭环 MuJoCo：`grasp` 3 s 内收敛（最大误差 < 0.25 rad、均值 < 0.08），`open` 保持 < 0.05 rad，身体默认姿态漂移 0.001 rad。
- 29-DoF 模型自动禁用、行为不变；motion 源索引与越界 clamp。
- **VLA 双输出**（`pytest tests/test_vla_hand_head.py -q` → 10 passed）：`HandHead` 形状/掩码池化/梯度、
  `forward_with_hands` 返回 `(tokens, hands)`、hand head 参数与 adapter 完全不相交、
  手部输出直连 43 维 ctrl 的 14 个手部槽位且**不改变身体槽位**。
- demo 实跑：`open → grasp → open`，末态误差 ≤ 0.005 rad/关节。

**未验证 / 已知缺口**

- 手部**不参与** SuperMimic 的世界模型与奖励（14 个 tracked 刚体不含手指）→ 策略不知道手在做什么，重物/强接触下不会补偿。
- 训练/部署模型不一致（29-DoF 训练、43-DoF 部署）的 sim-to-sim 差异未量化。建议训练时也用 43-DoF 模型并把手指锁死。
- 未做指尖 IK、物体资产、接触模型 → 抓取任务还不能闭环评测。
- 现成 checkpoint 与 benchmark 都只覆盖 29 维身体，手部**没有评测指标**（需要指尖位置误差、抓握准确率、接触成功率）。
- 增益、`slew_rate` 未做敏感性测试。

---

## 8. 文件清单

| 文件 | 内容 |
|---|---|
| `diff_vla_wbc/hand_stream.py` | `HAND_JOINT_NAMES` / `HAND_POSES` / `HandStream` / `HandStreamCfg` / `G1BodyIndex` / `load_hand_model` |
| `diff_vla_wbc/hand_head.py` | VLA 侧 `HandHead`（14 维直出）+ `HandHead.loss` + `attach_hand_head` |
| `diff_vla_wbc/gr00t_vlm.py` | 新增 `hand_head=` 参数与 `forward_with_hands()`（返回 tokens + hands） |
| `tests/test_hand_stream.py` | 15 个测试（布局、镜像、限位、slew、motion 源、闭环收敛） |
| `tests/test_vla_hand_head.py` | 10 个测试（hand head 形状/掩码/梯度、双输出、直连手部槽位） |
| `scripts/hand_stream_demo.py` | 无数据无 checkpoint 的可跑 demo，可存轨迹 npz |
| `reproduce_scenes/g1/g1_29dof_with_hand.xml` | 43 执行器模型（原有资产，未改动） |

---

## 9. 下一步（按优先级）

1. 用训练好的 29-DoF 策略 + `hand_stream` 跑 `eval_mujoco.py` 同款 episode（需要给 `G1MujocoEnv` 加 `xml_path` + `hand` 两个可选参数，或直接用 `G1BodyIndex` 写一个 43-DoF 版）。
2. 给 `HandHead` 接监督：43-DOF motion 的 `hand_dof_pos`（或手指 retarget 结果）作为 `[B,H,14]` 标签，加进 `JointTrainer`。
2. 手部参考数据：接 SONIC 的 `hand_dof_pos` 通道（43-DOF motion），或做手指 retarget。
3. 物体 + 接触：才谈得上抓取任务，`docs/FRAMEWORK.md` 里已标注未做。
4. 训练时启用 43-DoF 模型（手指锁死）以消除训练/部署的模型差异。
