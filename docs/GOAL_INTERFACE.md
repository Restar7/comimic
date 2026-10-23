# VLA → SuperMimic 的接口：直接输出 68×10 物理 goal

> 结论：**VLA 不走 token 了，直接回归 SuperMimic 要的物理参考量**（10 帧 × 68），手部 14 维另走独立头。
> 这样 VLA 的输出**原样**就是 SuperMimic 的输入，中间没有任何学习式转换。

---

## 1. 为什么放弃 token 路线

SuperMimic 的策略只认一种 goal（`motion_library.py:66`）：

```text
每帧 68 维 = [ q_ref − q_default (29, rad) | q̇ (29, rad/s) | 重力投影 (3, 单位向量)
              | 机体系角速度 (3, rad/s) | 航向系线速度 (3, m/s) | 根高度 (1, m) ]
10 帧拼成 680 维，喂 SupermimicBundle.act(obs_930, goal_680) -> 29 维动作
```

原来的做法是 `GR00T tokens [B,40,64] --MotionGoalAdapter--> [B,10,68]`，问题在：

| 问题 | 具体 |
|---|---|
| 语义不对齐 | 64 维是动作头隐空间，无单位、无物理含义；68 维是 rad / rad/s / m/s / m / 单位向量 |
| token 没有被训出来 | 仓库没有 GR00T checkpoint，action head 随机初始化 → token 空间本身是任意的 |
| 监督不可辨识 | adapter 只从下游 task loss 拿梯度，还要穿 10 步想象 + BPTT；64→680 的多对多映射没有锚点 |
| 时间口径硬凑 | token 40 帧 vs goal 10 帧，靠 `pool(4:1)+transformer` 猜 |

对比 SONIC：它的 64 维是 **motion tokenizer 在动作数据上训出来的**，decoder 有重建监督，
所以 token 有物理锚点。我们没有这个锚点，硬套就等于"用一个随机隐空间去回归物理量"。

**那 B（补 tokenizer+decoder）行不行？** 技术上可以，但它只是把"学习式转换"挪到 decoder 里，
而且必须先训 tokenizer、再训 VLA，两阶段；在没有预训练 token 语义的前提下，收益只有"输出维度低"，
代价是把一个可监督的物理目标变成两级间接监督。既然目标是**进入 SuperMimic**，那条路就不划算。

---

## 2. 现在的结构

```text
VLM backbone ─┬─► goal head  ─► goal  [B, 10, 68] 物理量 ─► flatten ─► SuperMimic(WBC/WM) ─► 29 身体关节
              └─► hand head  ─► hands [B, H, 14] 关节角  ────────────► HandStream(PD) ────► 14 手部关节
```

- **goal head**（`diff_vla_wbc/goal_head.py`）：masked-mean 池化 VLM token → MLP → `[B,10,68]`，输出**物理量**；
  最后一层零初始化 + 先验（直立重力 `[0,0,-1]`、高度 0.76 m），训练初期不是随机抖。
- **hand head**（`diff_vla_wbc/hand_head.py`）：同一份 backbone features → `[B,H,14]` 关节角，直接下发。
- 两个头都吃 `Gr00tVLM.backbone_features()`，互不干扰；`diff_vla_wbc/adapter.py` 保留但不再是默认路径。

---

## 3. 归一化（只发生在 head 内部）

一个头同时回归 rad、rad/s、m/s、单位向量、米，尺度差两个数量级，必须条件化。`GoalSpec` 用的典型量级：

| 项 | 维度 | 尺度 |
|---|---|---|
| `joint_pos`（q − q0） | 29 | 0.5 rad |
| `joint_vel` | 29 | 5 rad/s |
| `gravity` | 3 | 1.0 |
| `root_ang_vel` | 3 | 5 rad/s |
| `root_lin_vel` | 3 | 1.0 m/s |
| `root_height` | 1 | 0.1 m |

`GoalSpec.normalize/denormalize` 负责换算，**对外永远是物理量**，和 SuperMimic 内部
`_goal_scale_vector`（`policy/nets.py:44`）的单位完全一致；loss 在归一化空间算（Smooth-L1）。

---

## 4. 用法

```python
from diff_vla_wbc.goal_head import GoalHead, GoalHeadCfg, flatten_goal
from diff_vla_wbc.hand_head import HandHead, HandHeadCfg

vlm.goal_head = GoalHead(GoalHeadCfg(in_dim=2048, hidden=512))
vlm.hand_head = HandHead(HandHeadCfg(in_dim=2048, hidden=512, horizon=1))

goal, hands = vlm.forward_goal_and_hands(inputs)      # [B,10,68] + [B,H,14]
body_action = bundle.act(obs_hist_flat, flatten_goal(goal))   # (B, 29)，直接进 SuperMimic
hand.command(hands[:, 0, :])                          # 14 维直接下发
ctrl = hand.assemble_ctrl(body_tau_29, data=data)     # 43 维
```

`forward_goal_and_hands` 只跑一次 backbone（`backbone_features()`），不做 flow-matching 采样，
比原来的 token 路径还省一次 40 步积分。

---

## 5. 已验证

`pytest tests/test_goal_head.py -q` → **13 passed**：

- goal 每帧布局与 `GOAL_SLICES` 一致、680 展平是帧主序、`unflatten` 可逆；
- 各物理项尺度正确（0.5 / 5 / 1 / 5 / 1 / 0.1）；
- 初始化输出 == 物理先验（零关节偏移、重力 `[0,0,-1]`、0.76 m）；
- 掩码池化忽略 padding、loss 有限、梯度可达输入与参数；
- `forward_goal_and_hands` 返回 `(goal, hands)`，两个头都可选；
- **端到端**：`GoalHead` 的 680 维输出直接喂 `SupermimicBundle.act(obs_930, goal)` → `(1, 29)` 动作，数值有限。

---

## 6. 下一步

1. **给 goal head 配监督**：BONES-SEED npz 已经在仓库里，`MotionLibrary.goal_window(frames, 10)`
   可以直接产 `[B,10,68]` 标签 → 用 teacher forcing 预训 goal head，再接下游 task loss 微调。
2. 手部同理：43-DOF motion 的 `hand_dof_pos`（或手指 retarget 结果）作为 `[B,H,14]` 标签。
3. 把两路 loss 挂进 `JointTrainer`（有标签才算，没有就跳过），身体分支不受影响。
4. `adapter.py` 若要彻底移除，先确认没有脚本/README 还在引用（`docs/FRAMEWORK.md` 有图示）。
