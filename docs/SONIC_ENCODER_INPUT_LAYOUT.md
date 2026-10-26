# SONIC Encoder 输入布局（2026-10-26 实测破译）

> 验证脚本：`scripts/verify_sonic_encoder_layout.py`（实测 token 相等比例 **0.9995–1.000**）
> 数据：`data/ha_raw/HOI_double_desk/tw/*.npz` 的录制帧；ONNX：`model_encoder.onnx`（release）

## 1. 结论（760 → 640 → 64）

```
obs_dict (1, 1762)
  ├─ col[4:584]    = [ q_t (29) × 10 | qd_t (29) × 10 ]   ← 580，**当前参考帧重复 10 次**
  ├─ col[601:661]  = anchor_rot6d (6) × 10                 ← 60，**10 帧恒定**（当前锚点）
  └─ 其余列不参与 g1 分支（teleop/smpl 等）

g1 分支输入 = 640 = 580 + 60
  → MLP(640→2048→1024→512→512→64) → FSQ(round(tanh(x+off)*scale−bias)/16) → token(64)
```

| 分量 | 来源字段（HA raw npz） | 备注 |
|---|---|---|
| `q_t` | `human_joint_pos[t]` | 重定向参考，SDK/MJCF 顺序，**不做 perm** |
| `qd_t` | `encoder_motion_joint_vel_hist[t, -1]` | 与 `q_t` 同帧 |
| `anchor_rot6d` | `consumed_anchor_rot6d[t]` | 6 维；窗口内恒定；**公式未知**（见 §3） |

**验证事实**：
- `encoder_input` 的 646 个非零列 = `[4:584]`(580) + `[595:661]`(66)；
- 用 `[4:584] + [601:661]` 重建（其余置 0）→ ONNX token 与 `encoder_latent` **逐位相等**；
- 直接把录制的 1762 喂 ONNX → token 与 `encoder_latent` **Δ=0**（encoder 完全理解）。

## 2. 旧 `SonicTokenBuilder` 错在哪（这就是端口 token corr 0.057 的根因）

| 项 | 旧 builder | 真值 |
|---|---|---|
| q/qd | `frame + arange(10)`（**未来 10 帧**） | **当前帧重复 10 次** |
| anchor | 未来 10 帧各算一个 `rot6d(conj(q_robot)*q_ref)` | **当前锚点 6 维，10 帧恒定** |
| 顺序 | frame-major `cat([pos(580), vel(580)])` | 同（term-major 的 580 没问题）；但**代码**一度是**按帧交错**（每帧 `[pos_j, vel_j]`）→ 已修 |
| perm | 对 q/qd 做 `PERM_MJCF_TO_ISAACLAB` | 取决于库的口径：HA 录制是 encoder 序（**不置换**）；`build_ha_command_lib` 存 MJCF 序，则 `build_motion_windows` 内部置换**是对的** |

> 因此端口 token 与标签 token（= `encoder_latent`）MSE 0.085、corr 0.057 —— 不是"两个前端"，
> 而是**喂给 encoder 的量本身错了**。

## 3. anchor 6 维的公式（2026-10-26 **已解**）

```
anchor6 = rot6d(conj(q_robot) ⊗ q_ref_aligned)        # 3×3 的前两列，行优先展平
q_robot       = robot_root_orientation               # wxyz，机器人（仿真）根朝向
q_ref_aligned = human_body_quat_w_aligned            # wxyz，heading-aligned 参考根朝向
```

实测（`data/ha_raw` 7 个任务各取首条 clip）：

- 与录制 `consumed_anchor_rot6d`：**mean|Δ| ≤ 5.5e-5**（float32 噪声级）；
- 录制 `encoder_anchor_window[:, 0]` == `consumed_anchor_rot6d`（≤1e-4）⇒ 两者是**同一时刻同一个量**，
  不是"另一个时刻的量"（此前 S1 交接里那句话作废）；
- 用"命令流 q/qd（当前帧 ×10）+ 公式 anchor"重算 token 与 `encoder_latent`：
  **exact_frac = 1.0000（7/7 任务）**。

⇒ **BONES-SEED 的硬缺口已闭合**：在环 B 模式里 `q_robot` = 仿真机器人根朝向、
`q_ref` = 参考运动的 heading-aligned 根朝向，就能造出与 HA 标签**同一空间**的 token。
（离线打标若用"完美跟踪"假设，anchor 恒等 ≠ 部署分布；在环训练才是对的口径。）

门禁：`tests/test_sonic_token_contract.py`（anchor 公式 + 逐位 parity，数据缺失时 skip）。

## 4. 对 S1 的直接含义

1. **HA 数据不需要 builder**：直接用 npz 里的 `g1_motion_token_64`（= encoder_latent）当 WBC 的 token 条件；
2. **BONES-SEED 现在可以用"正确 builder"造同空间 token**（公式见 §3），混库不再有空间冲突；
3. builder 改动就三处：窗口 hold、`anchor = rot6d(conj(robot) ⊗ ref_aligned)`、按库口径决定是否置换；
   580 维必须 term-major（`diff_vla_wbc/sonic_encoder.py:build_motion_windows` 已修 + 有测试锁住）。
