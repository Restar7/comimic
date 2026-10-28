# token 空间 / anchor / 真物理插桩：**结论锁定**（2026-10-28）

> 目的：把过去两周反复出现的口径混乱**一次性钉死**，以后任何人（包括 agent）不得再靠"文档说"
> 或"我印象里"推翻下表。每一条都给出**可复现的验证命令**和**实测数字**。
>
> 判定规则：**只有实测数字能推翻本表**。文档/注释/记忆都不能。

---

## 0. 三条"不许再说"的谣言（已撤回）

| 错误说法 | 为什么错 | 实测 |
|---|---|---|
| "C6h 是混合包：策略吃侧车、WM 吃 label，两个空间不一致" | 侧车（`data/port_tokens`）是**从执行轨迹算的诊断文件**，S1 训练**从未读过它**（`runner.py:232` 在线从参考库算 token） | 参考库现算 vs npz 标签 **MSE 0.00069 / mean\|Δ\| 0.0110** |
| "VLA 的监督目标（label）和执行器的输入空间不一致，要换成侧车/要重训" | 二者是同一份（上面的 0.00069） | 同上 |
| "要把 WM 重新标定到侧车 / 要把 S1 重训到 label" | 前提（空间不一致）不成立 | 同上 |

**追加撤回**：`gt 0.0402 / oracle 0.0162 / port 0.0507` 那张三臂真物理表**全部作废**，
原因是喂 token 时用了预计算 npy（**anchor 被冻在录制时的机器人朝向**）。

---

## 1. 锁定的七条事实

### F1. HA 的 motion token = 命令流，经部署前端（1762 维）编码
- npz `g1_motion_token_64` 与 raw npz 的 `encoder_latent` **逐位相等**（`atol 1e-6`）。
- raw npz 里 `encoder_input` 是 **(T, 1762)** —— 部署前端输入；`encoder_motion_joint_pos_hist`
  是命令流；npz 的 `joint_pos` 是**实际执行轨迹**（`robot_qpos_before_decimation`），
  与命令流 mean\|Δ\| 0.39 rad、滞后 8 帧（`scripts/build_ha_command_lib.py` 开头）。
- 验证：
  ```bash
  python3 -c "
  import numpy as np,glob
  z=np.load('data/npz/ha_tokens/HOI_double_desk__tw__ep000000.npz')
  r=np.load(sorted(glob.glob('data/ha_raw/HOI_double_desk/tw/*.npz'))[0])
  print(r['encoder_input'].shape, np.allclose(z['g1_motion_token_64'], r['encoder_latent'][:len(z['g1_motion_token_64'])], atol=1e-6))"
  # → (2080, 1762) True
  ```

### F2. SuperMimic 训练吃的 token = **在线从参考库算**的 hold token（也是命令流）
- `trackerlab-supermimic/.../supermimic/runner.py:232`：
  `token_fn = SonicTokenBuilder(motion, dev, k_window=_k)` —— `motion` 是 `motion_packed` 参考库。
- 该库 `bones_seed_plus_ha_cmd_x20.pt` 由 `scripts/build_ha_command_lib.py` 从**命令流**建。
- 验证：见 F3。

### F3. F1 与 F2 是**同一个空间**（这是全部混乱的正解）
```
参考库现算(hold)  vs  npz 原装 label ：MSE 0.00069   mean|Δ| 0.0110
```
（0.011 在 fp16 量化噪声量级；npz token 本体就是 fp16 粒度 0.001 存出来的）

验证：
```bash
python3 /tmp/cmp_tokens.py     # 或在 H3/L1 run 里看 token_mse 基线
# → MSE(C vs A) = 0.00069   mean|d| = 0.0110
```

### F4. `data/port_tokens/ha_tokens/*.npy`（"侧车"）是**执行轨迹**编码，不是训练空间
- `scripts/build_port_tokens.py` 读 npz 的 `joint_pos / joint_vel / body_quat_w`（= 执行轨迹），
  走 640 维切片 → 64 维。
- 与 label 的差：**MSE 0.0212 / mean\|Δ\| 0.1117**（约 30 倍）。
- **用途**：只能当"另一条命令流口径？"的诊断；**禁止**当作执行器的条件空间或 VLA 的监督目标。
- 判别口诀：**训练吃什么 = 参考库现算；盘上的 .npy 侧车不是训练吃的。**

### F5. token 里含 **anchor**，anchor 用**机器人实时 root quat**
- `sonic_token.py::_tokens_one`：`anchor = rot6d(quat_conj(q_robot) ⊗ q_ref)`，
  `q_robot` 是**当时机器人**的 root quat（训练时=仿真机器人，部署时=真实机器人）。
- 因此：**任何预计算、anchor 冻住的 token 流，都不能用于真物理回放**——
  机器人朝向一旦偏离录制朝向，token 指向的就是错误的世界方向。
- 实测代价：冻 anchor → 根误差 **3.49 m**；实时 anchor → **1.0 cm**。

### F6. 真物理回放的正确喂法 = `SM_TOKEN_SOURCE=online`
- provider 每步：`frame = library_row(cursor)`，`robot_quat = robot.data.root_quat_w`（实时），
  `token = SonicTokenBuilder(参考库, window="hold")(frame, robot_quat=q_robot)`。
- 光标 `SM_TOKEN_START`（默认 0）/ `SM_TOKEN_STRIDE`（默认 1）；库 `SM_TOKEN_LIB`
  （默认 `runs/isaac_motion/ha_cmd/packed.pt`）。
- 状态落盘里 `frames` = **真实库行号**（`_ref_frame`），奖励补算直接用它。
- 代码：`third_party/HumanoidArena/.../action_provider/action_provider_supermimic.py`
  （`_next_latent_online` / `_build_online_token_builder` / `_OnlineLibView`），
  以及 `.../script/eval_scripts/sonic/sim_eval_vla.py` 的 `choices += "online"`。
- **启动时必须 `env -u LD_LIBRARY_PATH -u LD_PRELOAD`**，否则 Isaac 的 `carb` 子进程
  `fork` 报 EAGAIN（污染的系统 python torch 库路径）。
- 验证（第一次可信的真物理读数）：
  ```
  r_mean 0.06869  track 0.0721  e_root_pos 0.0100 m  e_body_pos 0.0085
  ```

### F7. 参考库必须是**命令流**库
- `runs/isaac_motion/ha_cmd/packed.pt`（命令流）用于部署口径 / 真物理 / 奖励。
- `runs/isaac_motion/ha_tokens/packed.pt`（执行轨迹）是旧表，**只在**"旧口径对照"里用；
  两库同一个 ckpt 能差 **0.33 NR**，混用必出反向结论。

---

## 2. 当前配置的合法组合（不许自创）

| 组件 | 值 | 备注 |
|---|---|---|
| VLA 起点 | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000` | 在 label 空间微调过 |
| 执行器 | `runs/S1_handover/WBC_A20000_wmvel_cmd.pt` 或 `WBC_A24000_wmvel_cmd.pt` | 两者 provenance 都是 `token_window=hold / obs=paper_obs93 / lib=bones_seed_plus_ha_cmd_x20 / token_dim=64`，是同一个口径 |
| VLA 监督标签 | npz `g1_motion_token_64`（`--bc-token-space label`）| **不要**改 port |
| WBC 条件 token | 参考库在线算（`--wbc-token-space` 随 `window=hold` 落 label）| 与 F3 一致 |
| 参考库 | `runs/isaac_motion/ha_cmd/packed.pt` | F7 |
| 真物理评测 | `SM_TOKEN_SOURCE=online` | F6 |

---

## 3. 已被实测否定的"优化方向"（别再做）

| 方向 | 结果 |
|---|---|
| 用 `data/port_tokens` 当 VLA 监督（`--bc-token-space port`）| token_mse 基线从 0.0047 变 0.0757；学得会但**目标是错的**（执行轨迹口径），没有任何部署意义 |
| 把 WM 重标定到侧车 | 前提不成立 |
| 用预计算 npy 做真物理 A/B | 冻 anchor ⇒ 假数（3.49 m 伪影） |
| 信任"想象 return 涨"作为进展证据 | 必须配 P0-2 的真物理校准（见 `docs/S3_FIX_PLAN_AFTER_REVIEW_20261027.md`）|

---

## 4. 采集（collect）口径：**只允许一种**（2026-10-28 起）

真物理采集/回放**必须**用下表；旧 `isaac_replay_2arm.sh`（npy 三臂）**永久停用**。

| 项 | 必须值 | 不这样会怎样 |
|---|---|---|
| token 源 | `SM_TOKEN_SOURCE=online`（每步库现算 + 实时 anchor）| 冻 anchor ⇒ 按错的世界方向走（实测 3.49 m）|
| 参考库 | `SM_TOKEN_LIB=runs/isaac_motion/ha_cmd/packed.pt` | 执行轨迹库 ⇒ 差 0.33 NR |
| reset 扰动 | `SM_RESET_NOISE=1 SM_RESET_SEED=<seed> SM_RESET_SCALE=1.0` | 不开 ⇒ 不同 seed 逐位相同（max\|Δa\|=0.0），n=1 |
| 动作记录 | `q_target_applied`（clip 后的 29 维 PD 目标，弧度）| 记 `a_norm` ⇒ WM 学"没真正下发过的动作"（实测 \|a_norm\| 可达 8.1，被 clip 到限位）|
| token 记录 | `tokens`（每步 64 维）| 无法审计/复算 |
| 帧号 | `frames` = **真实库行号** | 换 clip 必错 |
| 启动环境 | `env -u LD_LIBRARY_PATH -u LD_PRELOAD` | Isaac `carb` 子进程 fork 报 EAGAIN |
| 驱动脚本 | `scripts/isaac_replay_online.sh` | — |

### 采集验收（三条必须同时满足）
1. `e_root_pos ≤ 0.05 m`
2. 关节速度均值与录制同量级（录制 `qd_abs_mean 0.381`）
3. 不同 seed 的 `max|Δa| > 1.0`（否则 DR 没生效）

### 已验收的基线（`runs/isaac_online_lockdown`，执行器 = `WBC_A24000_wmvel_cmd`，300 步）

| seed | r_mean | track | e_root_pos | e_body_pos | qd 均值 | 起点 xy | 终点 xy |
|---|---|---|---|---|---|---|---|
| 0 | +0.06935 | 0.0733 | **0.0104 m** | 0.0084 | — | (-3.70,-2.15) | (-3.21,-2.12) |
| 1 | +0.07021 | 0.0738 | **0.0076 m** | 0.0078 | — | (-3.65,-2.24) | (-3.14,-2.21) |
| 2 | +0.06913 | 0.0717 | **0.0043 m** | 0.0082 | — | (-3.68,-2.22) | (-3.20,-2.19) |

- 录制参考：`qd_abs_mean 0.3810`；回放 `0.3715`（**97%**）✔
- 种子间 `max|Δa|` = 8.13 / 6.86 / 6.22 ✔（旧采集是 0.000）
- 新字段：`q_target_applied (300,29)`、`tokens (300,64)`、`meta.cfg`（11 项采集参数）✔
- 注意：`joint_pos` 的一阶差分 0.019 vs 录制 0.0065（含 PD 微跟踪），但**同号率 0.75**（不是随机抖动），
  且关节速度均值与录制 97% 吻合 ⇒ 视为合格；报指标时用 `qd` 而不是 `Δq`。

> 这 3 条 dump 是**第一批口径正确的采集数据**：`(s_t, q_target_applied_t, s_{t+1})` 可直接用于
> WM 保鲜/真转移（P0-3），`tokens` 可用于审计，`frames` 可直接补算奖励。
> 旧 `runs/isaac_replay_20261027_*` 下的 dump **只保留 transitions 的物理自洽性**，行为分布作废。

---

## 5. P0-2 结果（2026-10-28）：**真实 headroom 只有 ~3%**，这是本项目的现状判据

同一批窗口、同一个 bundle（`WBC_A24000_wmvel_cmd`）、同 seed、同 reset 扰动：

| 臂 | token 来源 | `imag_reward`（想象）| `state_term` | **真物理 r_mean** | `e_root_pos` |
|---|---|---|---|---|---|
| GT | 参考库在线算（hold + 实时 anchor）| 0.08740 | 0.3374 | **+0.06935** | 0.0104 m |
| VLA 底模 | `checkpoint-8000` 自己出的 token（npy 回放）| 0.08677 | 0.3041 | **+0.06740** | 0.0185 m |

```
Δ_imag(VLA vs GT) = -0.7%      ← 想象
Δ_real(VLA vs GT) = -2.8%      ← 真物理（同 seed、同扰动，配对）
```

**两条结论**：
1. **想象和真实这次是一致的**（−0.7% vs −2.8%，同量级）⇒ 世界模型在**这个**比较上不乐观，之前的
   "想象涨 12.7%/24%" 不是"想象偏乐观"造成的，而是**超过了 GT 上限** ⇒ 是 surrogate 被利用（漂移/边界外）。
2. **奖励几乎饱和**：底模 VLA 已经达到 GT 的 **97.2%**。用"跟踪奖励"当目标的联合微调，
   真物理最多还能拿 **~3%**。**继续优化这个目标不会有实质收益。**

> 因此后续联合微调的价值**不能**建立在"提高跟踪奖励"上，必须换目标：
> **扰动下的恢复能力 / 任务成功率 / 鲁棒性**。对应做法 = 把 `SM_RESET_SCALE` 拉大去测：
> GT 该顶住、底模该崩 —— 那才是 headroom 所在，也是 A1（VLA 看到被扰动的状态）真正要修的东西。
