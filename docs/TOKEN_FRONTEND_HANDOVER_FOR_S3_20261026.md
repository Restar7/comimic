# 给 S3 的交接：**token 口径错了**（encoder 输入装配）及其影响（2026-10-26）

> 读者：执行 **S3（联合微调 v2）** 的人/agent。
> 结论先行：**S1 的 WBC 与 HA 的标签 token 不在同一个"前端口径"上**，我们已定位到**具体错在哪、
> 怎么修、修完哪些东西必须跟着换**。S3 的所有"锚定 / FM-BC 目标 / burn-in / 门禁"都受影响，
> 按本文的清单改，否则会重演"训练指标很好、闭环不生效"。
> 证据等级：✅ 逐位实测（max|Δ| = 0）；🟡 需上机确认。

---

## 0. 一句话

SONIC 的 64 维 motion token 是 `frozen encoder(1762 维观察) → FSQ`。**1762 维里有 580 维是"10 帧命令窗口"**。
我们的实现把这个窗口填成 **未来 10 帧**；而 HA 录制/部署侧填成 **当前帧重复 10 次（hold）**。
同一个网络、不同填法 → 两套 token **corr 只有 0.05**。S1 的旧 WBC 学的是前者，VLA 与 HA 标签是后者。

---

## 1. 1762 维的精确布局（实测，✅）

来源：`GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx`（1762→64，三模式由 `encoder_index` 选）。

| 段 | 维度 | 内容 | 我们的库/录制里 |
|---|---|---|---|
| `encoder_mode` | 4 | 模式选择（joint29 遥操作 = `[0,0,0,0]`，SMPL = `[2,0,0,0]`） | 0 |
| `command_multi_future_nonflat` | **580** | **10 帧 ×（29 关节位置 + 29 关节速度）**，encoder 顺序 | **[4, 584)** |
| `motion_root_z_*` | 10 + 1 | 参考 root 高度 | 0（该模式不用） |
| `motion_anchor_orientation` | 6 | 机器人 root 与参考 root 的 6D 姿态差 | [595, 601) |
| `motion_anchor_orientation_10frame` | **60** | 10 帧 × 6D anchor | **[601, 661)** |
| lowerbody / vr / smpl / wrist | 120+120+9+12+720+60+60 | 其它模式用 | 0 |

- 直接用 `encoder_index` 后的偏移（ONNX 图里先做 `obs[1:]`，**所有偏移 +1**）：
  `COMMAND_OFFSET = 4`、`ANCHOR_MF_OFFSET = 601`（`diff_vla_wbc/sonic_encoder.py`）。
- 640 维（g1 分支）里 **只有这两段被 encoder 读**；`rot6d` = `R_robot^T R_ref` 的前两列（行主序）。

---

## 2. 错在哪：同一个槽位，两种填法

### 2.1 HA 录制/部署（**hold**，✅ 逐位）

`third_party/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_sonic.py`（joint29 遥操作分支）：

```python
motion_joint_window     = build_latest_hold_window(latest_joint_frame,     _STEP5_FRAMES)  # 10
motion_joint_vel_window = build_latest_hold_window(latest_joint_vel_frame, _STEP5_FRAMES)
anchor_window           = build_latest_hold_window(latest_anchor_frame,    _STEP5_FRAMES)
# 注释：we want the freshest intent ... Repeat the latest reference as a hold-future window
```

**原因**：实时遥操作（PICO VR）没有未来帧，sender 只能给"最新意图"，所以把 10 个槽位用当前帧填满。

**实测（全量，✅）**：`data/ha_raw/<task>/*.npz` 的 `encoder_input`，7 任务 × 6 clip：

| 检查 | 结果 |
|---|---|
| `[4:294)` 的 10 行是否相同 | **0.00e+00（全等）** |
| `[294:584)` 的 10 行是否相同 | **0.00e+00** |
| `[4:584)` vs `repeat(motion_joint_pos_hist[t], 10)` | **0.00e+00** |
| `[601:661)` vs `repeat(anchor[t], 10)` | 行间 Δ = 0 |
| 用上述配方跑 encoder → `encoder_latent` | **max\|Δ\| = 0.000e+00（bit-exact）** |

录制 npz 存的就是运行时那个数组（`"encoder_input": self._latest_encoder_input.copy()`），所以这是**代码行为**，不是后处理。

### 2.2 SONIC 训练环境（**真未来**，设计语义）

`gear_sonic/envs/manager_env/mdp/commands.py`（`TrackingCommand`）：

```python
self.future_motion_ids = self.motion_ids.repeat_interleave(self.num_future_frames)
frame_choices = torch.arange(min_frames, num_future_frames + 1, step=...)   # step5 采样
```

⇒ `.onnx` 这一侧本来是"未来 10 帧、每 5 帧取一个（覆盖 0.9 s）"。**两边都"对"，只是口径不同。**

### 2.3 我们之前的实现（❌）

`SonicTokenBuilder` / `build_motion_windows` 用 `idx = frame .. frame+9`（未来），anchor 也逐未来帧算。
⇒ 与 HA 标签 **MSE 0.085 / corr 0.057**（`docs/S0_AUDIT_REPORT_20261026.md` §4）。
这就是"联合微调 token 锚定 0.0152、闭环却不生效"的上游原因。

---

## 3. 已落地修复（S1 侧，S3 直接受益）

| 文件 | 改动 |
|---|---|
| `trackerlab-supermimic/.../algorithms/supermimic/sonic_token.py` | `SonicTokenBuilder(window=...)`，**默认 hold**（`SONIC_TOKEN_WINDOW=future` 可复现旧口径） |
| `diff_vla_wbc/sonic_encoder.py` | `build_motion_windows(..., window="hold")`，command 与 anchor 同步 hold |
| `scripts/build_ha_command_lib.py` | 用**命令流**重建 HA packed 库 → `runs/isaac_motion/ha_cmd/packed.pt` |
| `trackerlab-supermimic/data/bones_seed/bones_seed_plus_ha_cmd_x20.pt` | BONES 79.5% + HA(命令流)×20 的混合库 |
| `scripts/verify_sonic_frontend.py` | 前端复现探针（记录配方 + legacy 对照） |

⚠️ hold 下**离线拿 identity anchor** 无法逐位复现标签（实测 corr 0.97/0.50/0.40）：**anchor 是 token 的实质成分**。
训练必须用**环内机器人 anchor**（B 模式，默认），离线侧车只可用于 A 模式冒烟。

---

## 4. 第二个坑：HA 库的参考轨迹与标签不同源（✅）

| 对比（`HOI_double_desk__tw__ep000000`，2080 帧） | mean\|Δ\| |
|---|---|
| `data/npz/ha_tokens` 的 `joint_pos` vs raw `robot_qpos_before_decimation` | **0.000（逐位相等）** ← 库是**执行轨迹** |
| 同上 vs raw `encoder_motion_joint_pos_hist[-1]`（**命令流**，标签就是用它算的） | 0.410（最佳 lag **+8 帧**） |

⇒ 用旧库配标签 token，等于"看 t 时刻的码、追 t−0.16 s 的姿态"。
⇒ 修法：`runs/isaac_motion/ha_cmd/packed.pt`（`q/qd` 取命令流，root p/v/w 用执行轨迹作代理）。
⇒ **旧库没坏**：端口空间训练/评测自洽（0/699 摔是真的），只是不能和标签配对。

---

## 5. S3 必须跟着改的清单（**逐条**）

### 5.1 权重：只能用 label 空间的 WBC

- `--supermimic-ckpt` 指向 **`wbc_hold_*`**（`SONIC_TOKEN_WINDOW=hold` 训出来的，ckpt 里 `provenance` 会记 `token_window`）。
- **禁止**用 `ckpt_010500_physint*` / `ckpt_016000*`（端口空间，future 窗口）跑 S3：burn-in 与 reward 都在另一个空间。
- 加载时断言：`ckpt["args"]["obs_mode"]`、`robot_spec`（已有，`isaac_joint.py:289-318`）、`provenance.token_window == "hold"`。

### 5.2 锚定 / FM-BC 的目标：换标签 token

| 旧（作废） | 新 |
|---|---|
| `data/port_tokens/ha_tokens/*.npy` | **`data/npz/ha_tokens/*.npz` 的 `g1_motion_token_64`**（hold，逐位对的上 encoder_latent） |
| `RecordedTokenSource(port_dir)` | 读 npz 的 label 列（或先跑一次性导出到 `data/label_tokens/ha_tokens/*.npy`，命名与 clip 对齐） |
| `tgt[..., :78] = codec.encode(port_tok ⊕ hand14)` | `codec.encode(label_tok ⊕ hand14)`（**归一化空间目标**，S0.2 的 codec） |

- **删掉** sample-MSE 的 anchor/hand 主项，改 **FM-BC**（`vlm.action_head` 原生损失），与 V2 §3.5 一致。
- 采样后 **必须先 `codec.decode(x_norm)` 再喂 WBC**（J2）；`--decode none` 只用于复现旧结果，不得混表。

### 5.3 burn-in / reward 的 tokenizer

- `diff_vla_wbc/burnin.py` 里构造历史的 token 必须来自 **hold** builder（当前默认），且 `anchor` 用 burn-in 时的**仿真机器人 root**。
- `--fm-steps`（K）训练与部署必须一致（S0.3：K=4）。
- 奖励、`imag_rollout`、`bundle.step/act` 全部沿用冻结 WBC（S3.2），WBC/WM 不更新。

### 5.4 S3 的门禁（换成 label 口径）

| 门禁 | 命令要点 | 通过线 |
|---|---|---|
| HA-699 执行器评测 | `evaluate.py --ckpt <wbc_hold> --motion_packed runs/isaac_motion/ha_tokens/packed.pt` | `n_failed ≤ 5%` |
| BONES-500 防遗忘 | 同脚本 `--motion_packed data/bones_seed/bones_seed_compact.pt --clip_limit 500` | `norm_return` 掉点 ≤0.02 |
| WM 速度 | `scripts/eval_wm_velocity.py --ckpt <wbc_hold> --tokens label` | `qd_corr ≥ v1 同仪器值（≈0.74）`、`ratio ∈[0.9,1.1]` |
| 联合训练自检 | `--lambda-r 0` 对照 + 梯度比 0.1–0.5 | 与 V2 §3.5 相同 |

⚠️ `eval_wm_velocity.py` 默认 `--tokens port`，label 空间必须显式 `--tokens label`。
⚠️ 同一 ckpt 跑 HA 与 BONES 会覆盖 `_eval.json`，跑完立刻改名。

### 5.5 表与口径

- 所有旧数字（`A1/A2/A4/A5/anchor`、`0/699`、`145/699`）标 **`token_space=port, window=future`**，**不进 A1 正式表**；
- 新数字标 **`token_space=label, window=hold`**；
- 复现旧行为：`SONIC_TOKEN_WINDOW=future`。

---

## 6. 当前状态与数字（供 S3 判断能否开工）

| 项 | 状态 |
|---|---|
| 前端逐位复现 | ✅（hold + 录制 anchor，max\|Δ\| = 0） |
| hold 版 WBC 重训 | 🟢 进行中（`logs/wbc_hold_x20`，warm start `ckpt_016000`，BONES + `ha_cmd`×20，lr 2e-5，it 16120→18500） |
| 冒烟（120 iter，纯 rollout） | ✅ `sim_rew 0.84 / term 0.005 / wm_loss 0.047` ⇒ 换 tokenizer 不破坏身体 |
| 三格门禁 | ⏳ 训练完跑（HA-699 / BONES-500 / WM 速度） |
| ft318 的 hold/future | 🟡 推断为 hold（同 FSQ 网格、`teleop.*` 现场流），待补 anchor 后实测 |

**S3 的开工条件**：`wbc_hold_*` 三格门禁过线 + `provenance.token_window == hold` 在 ckpt 里可见。
在此之前 S3 可以做的：把 5.2 / 5.3 的代码改完，用 `--lambda-r 0`（纯 BC）在**旧端口 ckpt** 上做管道冒烟，**但结果不得当结论**。

---

## 7. 别做的事

1. ❌ 用 `data/port_tokens` 做 S3 的锚定目标（已在 S0.4 证伪）。
2. ❌ 混用两套 token（标签 token 的 VLA + 端口 token 的 WBC）。
3. ❌ 拿 identity anchor 造的离线 token 去当训练/门禁目标（anchor 是实质成分）。
4. ❌ 把 `token_mse` / 想象回报当门禁（V2 §5.6）；只看闭环与执行器评测。
5. ❌ 把 hold 与 future 的结果放同一张表。
