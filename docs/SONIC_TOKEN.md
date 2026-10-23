# SONIC motion token：冻结 encoder 的复现与 B 模式接入

> 结论：SONIC 的 64 维 motion token **已经能用纯 PyTorch 精确复现**（与部署 ONNX 逐位一致），
> 且按 **B 模式**（用仿真里的机器人状态在环算 anchor）接入 SuperMimic 的路径已打通。
> BONES-SEED 的数据管线也已跑完：142,220 → 保留 129,943 条 / 47.6M 帧 / 263.75 h / 紧凑库 6.75 GB。

---

## 1. token 是怎么来的

`GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx`
（1762 → 64，三个模式 g1 / teleop / smpl 由 `encoder_index` 选择）里的 **g1 分支**：

```text
输入（10 帧 × 64，逐帧拼接）
    [dof_pos(29), dof_vel(29), anchor_6d(6)] × 10 帧
    ↑ IsaacLab 关节顺序          ↑ 参考 root 与机器人 root 的 6D 朝向差
MLP: 640 → 2048 → 1024 → 512 → 512 → 64（SiLU）
FSQ: token = round(tanh(h + 0.032237) × 15.515501 − 0.5) / 16
```

权重直接取自 ONNX initializer（共 4,231,232 参数），**没有单独的 checkpoint**。

### 坑：偏移要 +1

ONNX 图先做 `obs[1:]`（丢掉 `encoder_index`），再切 g1 字段，所以真实偏移比图里的常量多 1：

| 字段 | 图内常量 | 实际 obs 偏移 |
|---|---|---|
| `encoder_index` | 0 | 0 |
| `command_multi_future_nonflat` | [3, 583) | **[4, 584)** |
| `motion_anchor_ori_b_mf_nonflat` | [600, 660) | **[601, 661)** |

按图内常量填会得到"看起来合理但全错"的 token（实测 maxdiff 1.5），偏移 +1 后与 onnxruntime **逐位相等**。

---

## 2. 代码

`diff_vla_wbc/sonic_encoder.py`：

| 接口 | 作用 |
|---|---|
| `G1MotionEncoder.from_onnx()` | 从 ONNX 抽权重，得到纯 PyTorch 模块（可在 GPU、训练环里批量跑） |
| `forward(command, anchor_mf)` | `(B,10,58) + (B,10,6)` → `(B,64)` 量化 token |
| `forward_raw` / `fsq` | 量化前连续 latent / 单独的 FSQ 步 |
| `build_onnx_input(...)` | 拼 1762 维输入（自检/对齐用） |
| `build_motion_windows(...)` | 从 motion + 机器人 root 朝向生成 `command`/`anchor`（含 clip 末尾 clamp） |
| `tokens_from_motion(...)` | 整条 clip → `(T,64)`（离线用） |
| `rot6d` / `quat_to_rotmat` | anchor 的 6D 朝向差（`R_robot^T R_ref` 前两列） |

验证：`pytest tests/test_sonic_encoder.py -q` → **12 passed**（随机/零输入与 onnxruntime 逐位一致、
FSQ 网格、29 关节顺序、identity/90° yaw 的 6D、末尾 clamp、整条 clip 编码）。

---

## 3. B 模式：为什么用机器人状态在环算 anchor

`motion_anchor_ori_b_mf_nonflat` 的语义是**参考朝向相对机器人当前朝向**的差值（`root_rot_dif_l_multi_future`），
它携带跟踪误差反馈。所以：

- **A（离线、假设完美跟踪）**：anchor 恒为 identity → token 只编码参考 motion；与部署分布不一致，**只用于冒烟**；
- **B（在环）**：每个训练 step 用仿真里的机器人 root 朝向算 anchor → token 分布与 SONIC 训练/部署一致，
  也和 ft318 里记录的真实 token 同口径。**这是正式路线。**

两个实现要点：

1. encoder 在训练环里是 `nn.Module`（GPU、按 env 批量），不是 onnxruntime；
2. token 必须 `detach`（no_grad）——它是条件输入，不能让梯度穿过 encoder 回到机器人状态，否则策略会"骗 encoder"；
   量化（round 到 1/16）在 detach 之后做，与部署一致。

---

## 4. 数据现状（BONES-SEED 管线已完成）

```text
datasets/bones-seed/v1/g1/**/*.csv        142,220 条（解压完成）
data/npz/bones_seed/*.npz                 142,220 个（97 GB，50 Hz）
data/npz/bones_seed/manifest_filtered.csv 保留 129,943 条（论文 129,514）
data/library/bones_seed_compact.pt        129,943 clips / 47,475,680 帧 / 263.75 h（论文 264.56）
```

B 模式**不需要**给 BONES-SEED 离线批量产 token：token 在训练环里由（参考未来 10 帧 + 仿真机器人状态）实时算出。

---

## 5. 接入 SuperMimic 还要改什么

| 改动 | 内容 |
|---|---|
| policy 条件 | 删掉 `MotionEncoder`（680→32），`Decoder` 输入 32→64，直接吃 token |
| 环内 token | 每个 step：`MotionLibrary` 取未来 10 帧参考 + 仿真机器人 root 朝向 → `build_motion_windows` → encoder → detach |
| critic | 训练时继续吃 680 维 GT goal（部署不需要 critic），可不动 |
| 奖励 / 世界模型 | 不动（世界模型 goal-free，奖励用 GT 参考刚体） |
| VLA | 加 64 维 token head（标签：ft318 的真实 token），手部 14 维 head 已就绪 |
