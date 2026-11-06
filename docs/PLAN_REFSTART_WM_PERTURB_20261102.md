# 完整计划 v4：分工版（本机 1×4090 + 2 卡容器，共享代码目录）

> 更新：2026-11-02
> **两台容器共享同一个代码目录**：
> `/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic`
> 不需要拷贝代码；产出一律写到约定的 `runs/` 路径，双方直接读。
>
> **原则**：新线全并行、旧线（R1–R4 / B_90x / ha_cmd / WBC_A24000）一个不动。
> 每一步有验收门，不过门不进入下一步。

---

## 0. 一句话

**本机（1×4090）负责所有代码修改、真 A'、WM、联合微调、闭环评测；
2 卡容器负责构建 `ha_cmd_exec` 参考库、构建 WBC 训练库、微调 WBC。**
WBC 训完写回共享 `runs/S1_handover/exp_WBC_exec.pt`，本机接进去继续 P2–P6。

**并行实现（硬约束，贯穿全文）**：

- 所有新产物带 `exp_` 前缀；旧线 `901fix4_partial / 902_joint / 903_joint / B_90x /`
  `ha_cmd / bones_seed_plus_ha_cmd_x20 / WBC_A24000 / sweeps/full_*` **只读**；
- **默认行为不变**：`START_FROM` 默认 `b`、`SM_PACKED` 默认 `ha_cmd`、`library_kind` 只新增
  `command_exec` 类型（旧判断不变）、`SM_HAND` 默认从 `0` 改为 `1`（只影响闭环，旧数据回放显式 `0`）；
- 两台容器共享代码目录，各写各的 `runs/exp_*`，不互相覆盖。

---

## 0.1 当前进度（2026-11-02 晚）

| 阶段 | 状态 | 证据 |
|---|---|---|
| **P0a 本机代码** | ✅ 已落地 | `START_FROM` / `MOTION_PACKED` / `library_kind=command_exec` / 评测 `SM_PACKED` / `run_refstart_joint.sh` / `reward_from_objects.py`（实测 DoubleDesk 99/100）/ `isaac_collect_perturbed.py`（L1+L2）/ `run_a_prime_seeds.sh`（**默认 4 seed**）/ **起点加权采样 + 覆盖审计**（`exp_start_weights.pt`：634k 帧、mean 1.0、max 3.79）/ `eval_drift_curve.py`（实测 main 0.947 / ctrl 0.843）/ R0 基线 `linear` 已存在；**最终 `huber_err` + `split_huber` + `--reward-delta-*` 待实现**（多轮文档 §7 P0）；完整方案见 `docs/JOINT_MULTIROUND_PLAN_20261103.md` |
| **P0b `ha_cmd_exec`** | ✅ **G0 PASS**（2 卡） | FK 脚误差 **2.4 mm**；clip/帧/`starts/lengths` 与旧库逐项全等；`mean\|Δq\| = 0.336 rad`；root 逐位相同 |
| **P1a 训练库** | ✅（2 卡） | `bones_seed_plus_ha_exec_x20.pt`，143,923 clips / 60.16M 帧 |
| **P1b 微调 WBC** | 第一版 28k 完成；**修正版在跑** | 第一版 20:00 完成，但 **`token_lib_train=None`**（执行 token，无 two-lib）→ G1 FAIL（见 P1c）；已重启修正版 `exp_wbc_exec_cmd`（`token_lib_train=ha_cmd`，24,050/28,000，ETA ≈ 21:55），输出 `runs/S1_handover/exp_WBC_exec_cmd.pt` |
| **P1c G1** | ❌ 第一版 FAIL（token 口径错） | **本机正式 G1（命令 token + 2080 步）**：`npy` 臂 228/2080 步**摔倒**（root 0.560 m）；`online` 臂 2080/2080 全程（root **1.526 m**、joint 0.503）——基线是 0.321 m / 1327 步摔。**根因已确认：第一版 ckpt 的 `token_lib_train=None`**（训练用执行 token，评测用命令 token，空间错位）。修正版 `exp_wbc_exec_cmd` 在跑，ETA ≈ 21:55 |
| **P2–P6** | ⏳ 等 G1 | — |
| **G2 参考成功率（本机）** | ✅ DoubleDesk **99/100 = 99%** | `scripts/reward_from_objects.py` 全量扫 `data/ha_raw`：100 条有 hammer/basket 字段的 clip 中 99 条 place=True（mean 60.2 步）；其余 599 条非 DoubleDesk 任务无该字段，需任务专属判据 |

**多轮完整方案（最终执行版）**：`docs/JOINT_MULTIROUND_PLAN_20261103.md`（R1 完整步骤 + R2–R5 改动 + 全部应改未改清单）。

**关键发现（2 卡 handover §4）**：训练 token 不是读 sidecar，而是 `SonicTokenBuilder` 用**训练库那一行的 q**
在线算。把 ref 换成执行 q 后，训练 token 也变成执行空间，和部署的命令 token 对不上
（dry-run 1.7 m 就是这么来的）。2 卡加了**默认关闭**的 `TwoLibTokenView` +
`SM_TOKEN_LIB_TRAIN=<命令流库>`，让"ref=执行、token=命令"真正成立。**各环节的 token 来源必须分清**：
- **WBC 训练（2 卡）**：`SM_TOKEN_LIB_TRAIN=runs/isaac_motion/ha_cmd/packed.pt`；
- **本机 P2（A' 回放）**：`isaac_g1_collect.py --token-npy` 用命令流 label ⇒ 本来就是命令 token；
- **本机 P4（联合微调）**：token 来自策略（VLA head），不读库 ⇒ 不需要 `SM_TOKEN_LIB_TRAIN`；
- **回放/评测**：token 用命令流（`SM_TOKEN_SOURCE=npy`，或 online 的 `SM_TOKEN_LIB=ha_cmd`），
  参考/评分用 `SM_PACKED=ha_cmd_exec`（本机已把这两个解耦）。

**策略梯度口径（2026-11-02 纠正）**：WBC 和联合微调都不是 PPO。`runner.py::policy_update()` 是
`loss = -J.mean()`，梯度**沿 20 步想象 rollout 反传**（BPTT / analytic gradient）：WM 权重
`set_requires_grad(wm, False)` 冻结，但前向图可微；critic 只是 **value baseline**（GAE 风格），
`NO_CRITIC=1` = 不用 baseline，不是"关 PPO"。没有 ratio / clip / old_logprob。
→ 因此 **WM 的准确性/覆盖直接决定策略梯度**（G3、A' 的覆盖比"PPO 调参"重要得多）。

> 2 卡容器因共享目录里原本没有 P0a 的 `build_ha_exec_lib.py` / `check_exec_lib.py` / `build_wbc_lib.py`，
> 已按 §3.1/§4/§6.1 的规格**补写**（纯新增，未改任何旧脚本）；本机只补了 §3.2/§3.3/§3.7/§3.8 与
> 5 处补丁（`START_FROM`/`MOTION_PACKED`/`library_kind`/`SM_PACKED`/奖励模式）。

---

## 1. 根因（为什么必须换参考库）

| 事实 | 证据 |
|---|---|
| 当前参考库是混血：q 来自**命令流**、根来自**执行轨迹** | `build_ha_command_lib.py` 注释 |
| 两者差 **mean\|Δq\|=0.39 rad、lag +8 帧（0.16 s）** | 同上 |
| WBC 训练库 79% bones_seed（FK 自洽 0.6 mm）+ 21% HA-cmd（矛盾目标） | `bones_seed_plus_ha_cmd_x20.pt`：12.68M/60.16M 帧 |
| 矛盾目标无解 → WBC 只能折中 → 参考回放 0.32 m（xy）、1327 步摔 | 实测 `ref_online` 回放 |
| 执行 q / 执行根 / 物体轨迹**都在原始 npz 里** | `robot_qpos_before_decimation`、`robot_root_position/orientation`、`env_obj_hammer_position` |
| 训练采样是"帧号 → 图像 + state_46 + 指令"整套取的；视觉来自 HA ego-view mp4 | `g1_tasks.py:198` `TaskObservationBuilder.build` |

**结论**：训练目标应该是 **(执行 q, 执行根)**，token（命令流编码）只当输入条件；
不需要反解"命令根"。

**实测每窗口漂移（B_903 DoubleDesk，696 个 20 步窗口）**：

| 量 | 数值 |
|---|---:|
| 每 20 步增量 median | **+2.33 cm** |
| mean | +4.07 cm |
| p90 | **+18.13 cm** |
| 变差窗口占比 | **64.4%** |
| 104 窗口线性累积 | **≈1.75 m** |

→ 每窗口只有 2 cm，但 64% 的窗口在变差，104 个窗口就累积到 1.75 m。
**参考起点训练 + 2 cm 动作噪声覆盖不到累积后的 0.5–1 m 漂移状态**——所以要么
P5 实测证明累积可控，要么混 20% 漂移起点（DAgger）。

**另一点（和起点强相关）**：图像的 `frames` 是参考帧，state 输入会被 burn-in/起点覆写
（`vla_state_from_burnin=True`）。B 起点会让"图像（参考帧）↔ state（漂移状态）"错位；
参考起点把**图像、state、参考**三者对齐。

---

## 2. 分工总览

| 阶段 | 任务 | 机器 | 产出（共享路径） | 依赖 |
|---|---|---|---|---|
| **P0a** | 写代码：`build_ha_exec_lib.py`、`reward_from_objects.py`、`run_refstart_joint.sh`、`run_a_prime_seeds.sh`、`s3_loop.sh` 的 `START_FROM`、`library_kind` 扩展、评测 `SM_PACKED`、`isaac_collect_perturbed.py`、`eval_drift_curve.py`、`compute_collect_reward.py` 的奖励模式（linear/huber） | **本机** | `scripts/*` | — |
| **P0b** | 构建 `ha_cmd_exec`（699 clip）+ FK 校验 | **2 卡容器** | `runs/isaac_motion/ha_cmd_exec/packed.pt` + `report.json` | P0a 脚本 |
| **P0c** | 旧 WBC × `ha_cmd_exec` 快速对照（10 min） | **本机** | 对照数字 | P0b |
| **P1a** | 构建 WBC 训练库 `bones_seed_plus_ha_exec_x20.pt` | **2 卡容器** | `trackerlab-supermimic/data/bones_seed/…` | P0b |
| **P1b** | 微调 WBC（`NPROC=2`） | **2 卡容器** | `runs/S1_handover/exp_WBC_exec.pt` + 日志 | P1a |
| **P1c** | G1 参考回放验收 | **2 卡容器** | `runs/exp_g1/*` 指标 | P1b |
| **P2** | 真 A'（WBC 回放）+ L1 动作噪声 + L2 起点扰动 + 物体奖励 | **本机** | 每 clip `runs/wm_real/exp_a_prime/<clip>/merged.pt` → 合并 `runs/wm_real/exp_ha_exec_transitions.pt` | P1c |
| **P3** | 重训 WM | **本机** | `runs/S1_handover/exp_wm_exec.pt` | P2 |
| **P4** | 从 base、参考起点联合微调 | **本机** | `runs/S3_loop/exp_ref_base/` | P3 |
| **P5** | 闭环 + 分级成功 | **本机** | `runs/S3_loop/sweeps/exp_*` | P4 |
| **P6** | 手部（`SM_HAND=1`） | **本机** | 结果 | P5 |

**时序**：

```
本机:   P0a 写代码 ──┐                    ┌─ P0c 对照 ─┐
                     │                    │            │
2卡:                 └─ P0b ha_cmd_exec ──┴─ P1a 库 ── P1b 微调 ── P1c G1 ──┐
                                                                          │
本机:                                                          P2 ─ P3 ─ P4 ─ P5 ─ P6
```

- P0a 先做（本机，0 GPU）；
- 2 卡容器等 P0a 的脚本落盘后，立刻开 P0b；
- 本机在等待 P1b 的时间做：物体奖励脚本、A' 脚本、评测分析脚本、文档；
- P1c 的 G1 过了，本机接 P2。

---

## 3. P0a 本机：代码修改清单

### 3.1 `scripts/build_ha_exec_lib.py`（新增，2 卡容器执行；已由 2 卡补写并跑通）

读原始 npz，写**执行参考库**：

```text
输入: data/ha_raw/<TASK>/<tw|yb>/<clip>.npz
输出: runs/isaac_motion/ha_cmd_exec/packed.pt
      {ref (N,71) float16, names, starts, lengths}
      ref 布局: [p(3), quat_wxyz(4), q(29), v_w(3), w_b(3), qd(29)]

字段映射:
  p     <- robot_root_position
  quat  <- robot_root_orientation (wxyz)
  q     <- robot_qpos_before_decimation
  v_w   <- robot_root_lin_vel_world
  w_b   <- robot_root_ang_vel_world
  qd    <- robot_qvel_before_decimation
  names <- <原始 clip 名>（`packed.pt`，699 clip）
         `packed_x20.pt` 才是 `<clip>__HA_CMDr0..r19`（13,980 clip，WBC 训练库用）
```

**关节顺序（关键）**：npz 的 `robot_qpos_before_decimation` 已经是 **STANDARD_29（MJCF）序**，
**不要**再套 `JC.to_standard()`（会二次置换 ~0.5 m）；2 卡用 `G1Kinematics` FK **零误差**复现
`body_pos_w` 验证过。

- **不覆盖** `runs/isaac_motion/ha_cmd/packed.pt`；
- 输出 `report.json`：每条 clip 的 FK 脚误差、根高度差、帧数。

### 3.2 `scripts/reward_from_objects.py`（新增）

用物体真值算**任务奖励 + 分级成功**：

```text
输入: 原始 npz 的 env_obj_hammer_position / env_obj_basket_position /
       episode_init_env_obj_*
输出: 每帧 reward（0/1 或分级） + 每集 success/grasp/lift/place 标签
```

- 成功判据：锤子中心进入篮子区域（阈值用 episodes 的 init 位置标定）；
- 分级：接触（手–锤子距离）/ 抬起（锤子 z 超过台面）/ 搬运 / 入篮；
- 用途：P2 的 A' 奖励、P5 的评测指标、验证"参考本身是不是成功"。

实际 CLI（已实现并实测）：

```bash
python3 scripts/reward_from_objects.py --npz-dir data/ha_raw \
  --packed runs/isaac_motion/ha_cmd_exec/packed.pt \
  --out runs/wm_real/exp_object_metrics.json \
  --r-xy 0.15 --dz 0.25 --lift 0.03 --reach 0.20
```

- 只有 DoubleDesk 的 npz 有 hammer/basket 字段；其余任务 `n_skipped_no_object`（需任务专属判据）；
- **实测：DoubleDesk 99/100 = 99% 参考成功**（mean 60.2 步 place）。

### 3.3 `scripts/run_refstart_joint.sh`（新增）

只跑 joint stage，参考起点、从 base 起：

```bash
ROUND=1001 RUN_LABEL=exp_ref_base STEPS=400 \
INIT_VLA="" \
SM_CKPT="$PWD/runs/S1_handover/exp_WBC_exec_cmd.pt" \
MOTION_PACKED="$PWD/runs/isaac_motion/ha_cmd_exec/packed.pt" \
ADV_BASELINE=none CRITIC_ON=0 GAMMA=1.0 \
START_SAMPLER=phase START_WEIGHTS="$PWD/runs/wm_real/exp_start_weights.pt" START_AUDIT=1 \
bash scripts/run_refstart_joint.sh
```

> `SM_CKPT` 必须是 **G1 通过**的 ckpt（当前候选是 two-lib 的 `exp_WBC_exec_cmd.pt`，
> 第一版 `exp_WBC_exec.pt` 是执行 token 空间，不要用）。

- 内部 `NPROC=1 BATCH=8 ACCUM=8`（本机单卡，保持全局 batch 64）；
- `START_FROM=reference`（见 3.4）；
- `WM_REFRESH_STEPS=0`（隔离变量）；
- **奖励默认见多轮文档**：`huber_err`（`d=√e` + `h(d;δ)`）+ `split_huber`，R1 δ=5cm；`CRITIC_ON=1`、`TRUST_WEIGHT=0`、`ACT_WEIGHT=0.2`；完整公式见 `docs/JOINT_MULTIROUND_PLAN_20261103.md`。

### 3.3.1 起点覆盖：加权采样 + 审计（2026-11-02 新增）

全库均匀随机时，200 步 × 64 = 12,800 个起点 ≈ **2%** 的 634k 帧，关键抓取/放置段更稀。
现在改为**按阶段/任务加权**：

| 项 | 默认 |
|---|---|
| `--start-sampler` | `uniform`；新线 `phase` |
| `--start-weights` | `runs/wm_real/exp_start_weights.pt`（`reward_from_objects.py --weights-out` 生成） |
| 权重 | `(1 + 4*phase) × 任务均衡`；phase = reach/lift/place 任一为 1（5 帧平滑） |
| 实测（生成文件） | 634,016 帧，mean 1.0，max 3.79；DoubleDesk mean 2.0，其余 6 任务各 0.49–1.32 |
| `STEPS=400` × global 128（`ACCUM=16`） | **51,200 个起点（8.1%）**，DoubleDesk 关键段权重 2× |
| `START_AUDIT=1` | 训练结束写 `start_coverage.pt/json`（unique frames / unique clips / per-task） |

代码：
- `diff_vla_wbc/data.py`：`MotionBatchProvider(start_weights, start_audit)`，有权重走 `torch.multinomial`；
- `diff_vla_wbc/train_v2.py`：`BurnInBatchProvider` 透传 + `audit_frames()`；
- `scripts/train_joint_v2.py`：`--start-sampler {uniform,weights,phase,task}` / `--start-weights` /
  `--start-audit`；`task` 模式按 7 个任务均衡（无需权重文件）。

### 3.4 `scripts/s3_loop.sh`：加 `START_FROM`

joint stage（约 307 行）：

```bash
if [[ "${START_FROM:-b}" == "reference" ]]; then _SFB=""; else _SFB="$B_GLOB"; fi
env V3=1 TRUST_CENTER="$TRUST_CENTER" START_FROM_B="$_SFB" ...
```

- 默认 `b` → 旧行为不变；
- `reference` → 不传 `--start-from-b` → 起点 = 参考（`isaac_joint.py:648`）；
- **burn-in**：默认 10 步，起点 = 参考帧 `f−10` + 冻结 WBC 滚 10 步（贴部署）；
  要严格"起点偏差 0"设 `BURNIN=0`，起点 = `reference_start` 原值；
- 参考起点模式下 `b_start_max_err`（E_start）**不生效**，帧是全库均匀随机；
- 评测 provider 走同一条路 → `fixed_eval` 也在参考起点上，训练/评测同分布。

### 3.4.1 视觉 / 状态 / 指令必须同源

`TaskObservationBuilder.build(frames)`（`g1_tasks.py:198`）按帧号一次取三样：

| 输入 | 来源 |
|---|---|
| 视觉 `video.ego_view` | HA 录制的 **ego-view mp4**（`g1_ego_view` / `vision_rgb_video_path`） |
| `state_46` | HA npz（`g1_state_46`），**会被 burn-in 起点覆写**（`vla_state_from_burnin=True`） |
| 指令 | HA npz（`g1_instruction`） |

- **参考起点** → 图像、state、参考同帧，一致；
- **B 起点** → 图像是参考帧、state 是漂移状态，**错位**；
- 评测用 Isaac `front` 相机，和 HA ego-view 仍有视角差（已知 gap，消不掉，只能减小）。

### 3.5 `diff_vla_wbc/sonic_frontend.py`：`library_kind` 扩展

现在只认 `ha_cmd`（command）/ `ha_tokens`（executed），其它返回 `unknown`；
`train_joint_v2.py:722` 会拒绝非 command 库。

- 新增：路径含 `ha_cmd_exec` → 返回 `"command_exec"`（token 仍是命令流，参考 q 是执行）；
- `train_joint_v2.py:722` 的断言放宽到 `{"command","command_exec"}`；
- 或者临时把新库命名为 `ha_cmd_exec`（含 `ha_cmd`）复用 `command`，但**不推荐**，语义会混。

### 3.6 评测侧参考库覆盖

`action_provider_supermimic.py` 里参考库原来硬编码 `runs/isaac_motion/ha_cmd/packed.pt`。
**已改**：`_packed_ref_path()`（118 行）读 `SM_PACKED`，替换了 `_load_lib_once`（765）和
`_http_ref_last`（821）两处；**token 库仍读 `SM_TOKEN_LIB`（484 行）**，两者解耦。新增：

```text
--sm_packed / SM_PACKED   # 参考库（默认 ha_cmd；新线传 ha_cmd_exec）
```

- 只影响 reward/frames/误差；**token 仍来自命令流**（`SM_TOKEN_SOURCE=npy` 用 `encoder_latent`，
  或 online 用 `ha_cmd` 的命令库）；
- 训练侧用 `--motion-packed`（已有）指向 `ha_cmd_exec`。

### 3.7 `scripts/isaac_collect_perturbed.py`（新增，P2 的 L1）

复制 `isaac_g1_collect.py`，加：

```text
--action-noise-std 0.05   --noise-mode gauss   --perturb-hold 20
```

- 实际插在 `a = pol(...)` 之后、`set_joint_position_target` 之前：按 `perturb_hold` 重采 δ，`a=(a+δ).clamp(-1,1)`；
- **记录加噪后的 `a`、`last_a` 也用加噪后的**（否则 WM 学的动力学是错的）；
- 新增 `--seed` + **L2 起点扰动** `--start-perturb-xy/yaw/dq`（xy 0.1 / yaw 0.1 / dq 0.02）；
- 多 seed 驱动：`scripts/run_a_prime_seeds.sh`（默认 `SEEDS="0 1 2 3"`，合并成 `merged.pt`）；
- 奖励用 `compute_collect_reward.py --reward-mode huber_err --reward-delta-pos <δ> --reward-delta-ori <δ>`（与训练同口径）或 `reward_from_objects.py` 补。

### 3.8 `scripts/eval_drift_curve.py`（新增，P5）

- 输入 sweep 的 dump；
- 输出每条 episode 的 `root error vs t`、首次发散时间、分级成功；
- 不再只看整集均值。

---

## 4. P0b 2 卡容器：构建 `ha_cmd_exec`

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
python3 scripts/build_ha_exec_lib.py \
  --npz-dir data/ha_raw \
  --out runs/isaac_motion/ha_cmd_exec \
  --x20-repeats 20
python3 scripts/check_exec_lib.py --packed runs/isaac_motion/ha_cmd_exec/packed.pt \
  --report runs/isaac_motion/ha_cmd_exec/report.json
```

### G0 验收门

| 指标 | 目标 |
|---|---|
| FK(执行 q, 执行根) vs `body_pos_w` 脚位置 | max < 3 cm |
| 根高度与执行根 | < 2 cm |
| 帧数 / clip 数 | 699 clips，帧数与原 npz 一致 |
| 与 `ha_cmd` 对比 | q 差 ~0.39 rad、根相同（预期） |

**实测（2 卡，2026-11-02）**：FK 脚 **2.4 mm**、根高 0.24 mm、699 clips / 634,016 帧、
`mean\|Δq\|=0.336 rad`、root 逐位相同、`index_structure_identical=true` → **PASS**。

**硬约束（否则图像/指令/sidecar 全错位）**：

- clip 名必须保持 `...__HA_CMDr0..r19`；
- 每条 clip 的**帧数**、`starts/lengths` 排布必须和原库**逐项一致**；
- 只替换 `ref` 里的 q / root / 速度数值，**不动索引结构**；
- 原因：`TaskObservationBuilder` 用 `frames` 去查 ego-view mp4、`state_46`、指令和 token sidecar，
  索引一变，图像和状态就对不上。

不过 G0 不进入 P1。

---

## 5. P0c 本机：旧 WBC × `ha_cmd_exec` 快速对照（10 min）

```bash
# 用旧 WBC、新参考库、命令 token（npy）
SM_TOKEN_SOURCE=npy \
SM_TOKEN_NPY=runs/exp_tokens/HOI_double_desk__tw__ep1776268633456387.npy \
SM_PACKED=runs/isaac_motion/ha_cmd_exec/packed.pt \
SM_CKPT=runs/S1_handover/WBC_A24000_wmvel_v6.pt \
SM_START_FRAME=0 SM_INIT_FROM_LIB=1 SM_HAND=1 RECORD_VIDEO=0 \
SM_DUMP_STATES=runs/B_success/replay/exp_exec_oldwbc/dump/DoubleDesk \
bash scripts/ha_isaac_closedloop.sh
```

**口径警告**：必须用 `SM_TOKEN_SOURCE=npy`（命令 token）。若 online 从 `ha_cmd_exec` 算，
得到的是**执行空间 token**，旧 WBC 会在 200 步内跑偏 1.7 m（2 卡 dry-run 实测）。

判读：

| 结果 | 含义 |
|---|---|
| root 误差从 0.32 m 明显下降（< 0.15） | 混血是主因，微调只是收尾 |
| 几乎不变 | 旧 WBC 跟不上执行 q，P1b 要下更大功夫 |

---

## 6. P1 2 卡容器：训练库 + 微调 WBC

### 6.1 构建训练库

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
python3 scripts/build_wbc_lib.py \
  --bones trackerlab-supermimic/data/bones_seed/bones_seed_compact.pt \
  --ha-exec runs/isaac_motion/ha_cmd_exec/packed.pt \
  --x20 20 \
  --out trackerlab-supermimic/data/bones_seed/bones_seed_plus_ha_exec_x20.pt
```

- bones_seed 129,943 clips 原样；
- HA 部分替换为 `ha_cmd_exec` 的 699 clips ×20，clip 名保持 `__HA_CMDr0..r19`；
- **token 口径（关键，已由 2 卡修正）**：训练 token 不是读 sidecar，而是 `SonicTokenBuilder`
  用训练库那一行的 q 在线算。若直接用执行库，token 会变成**执行空间**，与部署的命令 token 对不上
  （2 卡 dry-run 1.7 m 的根因）。必须开**双库视图**：
  `SM_TOKEN_LIB_TRAIN=runs/isaac_motion/ha_cmd/packed.pt`（默认关闭，数值等价于在命令库上算 token，
  同时 ref 仍是执行库）；ckpt 的 `provenance.token_lib_train` 会记录。

### 6.1.1 WBC 从哪个起点微调（已核对）

| ckpt | 迭代 | 训练库 | obs_mode | 能不能直接接 |
|---|---:|---|---|---|
| `bones_seed_retrain_1025b/ckpt_012500.pt` | 12,500 | **bones_seed-only** | `relative_ori99` | ⚠️ obs 口径不同 |
| `bones_seed_token_2x8192/ckpt_002000.pt` | 2,000 | bones_seed-only | relative_ori99 | ❌ 太早 |
| `runs/S1_handover/WBC_A24000_wmvel_v6.pt` | 24,000 | `bones_seed_plus_ha_cmd_x20` | **paper_obs93** | ✅ 兼容，但带 HA-cmd 折中 |
| `WBC_A20000_*` / `A24000_wmvel_cmd` | 20–24k | `bones_seed_plus_ha_cmd_x20` | paper_obs93 | ✅ 兼容，同样带折中 |

- **首选 A**：从 `WBC_A24000_wmvel_v6` 起，在 `bones_seed + ha_exec` 上微调（兼容、便宜）；G1 过了就说明折中被洗掉；
- **备选 B**：从 `bones_seed_retrain_1025b/ckpt_012500`（bones_seed-only）起，但要把训练/评测的 obs 切到 `relative_ori99` 或多一步转换；G1 不过再考虑；
- **不推荐 C**：从零重训 24k（2 卡成本太高）。

### 6.2 微调 WBC（2 卡）

> **必须设 `TOKEN_LIB_TRAIN=runs/isaac_motion/ha_cmd/packed.pt`（two-lib）**。
> 第一版漏了 → ckpt 的 `token_lib_train=None` → 训练 token 是执行空间，G1（命令 token）1.526 m。
> 修正版：`logs/exp_wbc_exec_cmd` / `runs/S1_handover/exp_WBC_exec_cmd.pt`。

实际（2 卡已跑，warm start 24k → 28k）：

```bash
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC/trackerlab-supermimic"
MODE=full NPROC=2 ENVS_PER_RANK=4096 ITERATIONS=28000 CKPT_EVERY=500 LR=2e-05 \
LIBRARY=$COMIMIC/trackerlab-supermimic/data/bones_seed/bones_seed_plus_ha_exec_x20.pt \
INTERFACE=token COMPILE=0 \
RESUME=$COMIMIC/runs/S1_handover/WBC_A24000_wmvel_v6.pt \
OBS_MODE=paper_obs93 MAX_EPISODE_STEPS=0 NO_CRITIC=1 FREEZE_WM=0 \
SM_TOKEN_LIB_TRAIN=$COMIMIC/runs/isaac_motion/ha_cmd/packed.pt \
LOG_DIR=$COMIMIC/trackerlab-supermimic/logs/exp_wbc_exec \
bash scripts/rls/supermimic/train_bones_seed.sh
```

- 也可以直接 `bash scripts/p1b_finetune_wbc_exec.sh`（2 卡封装的同一配方）；
- `RESUME` 是 warm start：`ITERATIONS` 必须给**绝对目标** 28000（`start_it >= iterations` 会报错）；
- `LR=2e-5`（不是 from-scratch 的 2e-4）；`OBS_MODE=paper_obs93`（与部署/provider 一致）；
- `NO_CRITIC=1`（warm start 时 critic 是新的，bootstrap 会拽崩）；`FREEZE_WM=0`（WM 跟执行库小幅更新）；
- 产出 ckpt 复制到 `runs/S1_handover/exp_WBC_exec.pt`（共享路径，`exp_` 前缀）。

### 6.3 G1 验收（2 卡容器跑，结果共享）

参考回放：`SM_TOKEN_SOURCE=npy`（命令 token）+ `SM_PACKED=ha_cmd_exec` + `SM_CKPT=exp_WBC_exec`，
`SM_START_FRAME` 取 clip 起点，跑完整条。

> **`online` 臂的坑**：2 卡的 `scripts/p1c_g1_replay.sh` 里 `TOKEN_LIB` 默认 = `SM_PACKED`
> （即 `ha_cmd_exec`）⇒ `online` 算出来是**执行 token**，不是命令 token。
> 公平对照必须显式 `TOKEN_LIB=runs/isaac_motion/ha_cmd/packed.pt`；
> `npy` 臂读录制的 `encoder_latent`，不受这个默认影响，**才是部署口径**。

| 指标 | 目标 |
|---|---|
| root 误差（xy） | **< 0.10 m**（现在 0.32） |
| 关节 RMSE | < 0.2 rad |
| 摔倒 | **0** |
| 走完整条 clip | 是 |

**第一版 28k 实测（本机，`npy` + `online`，2080 步）**：

| 臂 | 步数 | root xy | joint RMSE | 摔倒 |
|---|---:|---:|---:|---|
| `npy`（命令 token，冻结 anchor） | 228/2080 | 0.560 m | 0.456 rad | **是** |
| `online`（命令 token，实时 anchor） | 2080/2080 | **1.526 m** | 0.503 rad | 否 |
| 基线（旧 WBC × ha_cmd，online） | 1327 步摔 | 0.321 m | 0.42 rad | 是 |

→ 第一版 ckpt 是**执行 token** 训的（`token_lib_train=None`），评测用**命令 token**，空间错位。
**这不是 G1 的最终结论**；修正版 `exp_WBC_exec_cmd` 出来后再跑同一口径。

不过 G1 不进入 P2；把 `report.json` + 视频 + 指标写到 `runs/exp_g1/`。

---

## 7. P2 本机：真 A'（WBC 回放 + L1/L2 覆盖，**只给 WM**）

> **A' 只用于训练 WM**，不参与联合微调。联合微调用的是 `ha_cmd_exec` 参考库 + 重训后的 WM。
> **并行保证**：产物全部写 `runs/wm_real/exp_*`，不覆盖 `ha_transitions.pt` / `real_transitions_v2b.pt`。

### 7.1 数据生成（1 起点 + 4 seed；G2 先用 2 seed）

- **起点**：每条 clip 从起点（row 0）**回放整条**（~900 步），天然覆盖全部阶段；
- **seed**：4 组独立噪声（L1+L2），每组一条轨迹；
- **不采 B、不做 L3 链式回放**：WBC 重训本身就是修漂移，L3 不再需要；
- **驱动**：`scripts/run_a_prime_seeds.sh`（每 clip 跑 N 个 seed → `runs/wm_real/exp_a_prime/<clip>/merged.pt`）；
  全部 clip 跑完后合并成 `runs/wm_real/exp_ha_exec_transitions.pt`（全局文件，给 P3）。

体积：699 clip × 4 seed × ~900 步 ≈ **250 万行**（单卡 Isaac 8 env，约 4–8 小时）。
先用 2 seed 过 G2，再用 **1→2→4 的 G3a 曲线**决定是否加到 8——不要一上来堆满。

### 7.2 L1 动作噪声（`scripts/isaac_collect_perturbed.py`）

在 `a = pol(hist, tok)` 之后、`set_joint_position_target` 之前：

```python
# 实际代码（scripts/isaac_collect_perturbed.py）
if float(args_cli.action_noise_std) > 0:
    _hold = max(1, int(args_cli.perturb_hold))
    if step % _hold == 0 or _delta is None:          # hold=20，匹配部署重规划节奏
        if args_cli.noise_mode == "gauss":
            _delta = torch.randn_like(a) * float(args_cli.action_noise_std)
        else:
            _delta = (torch.rand_like(a) * 2.0 - 1.0) * float(args_cli.action_noise_std)
    a = (a + _delta).clamp(-1.0, 1.0)                # 归一化动作空间，clip 防越界
robot.set_joint_position_target(default_q + a * scale, ...)
A.append(a.cpu())        # ★ 记录加噪后的 a
last_a = a.detach()      # ★ WBC 历史也看到扰动
```

- σ=0.05（≈0.019 rad ≈1.1°），hold=20；
- warm-up 不加噪；每 env 独立；clip [-1,1]；
- 先 0.05，G3a 好再加 0.1；不要 0.2+（会摔）。

### 7.3 L2 起点扰动

- 在写根/关节初值之后：`xy += N(0, 0.1 m)`、`yaw += N(0, 0.1 rad)`、`q += N(0, 0.02 rad)`；
- 3 档：0 / 0.05 / 0.1 m，覆盖参考附近的小漂移；
- **不做 L3**（累积漂移由 WBC 重训解决）。

### 7.4 A' 里存两份奖励

- **跟踪奖励**：`compute_collect_reward.py --reward-mode huber_err --reward-delta-pos 0.05 --reward-delta-ori 0.05 --packed runs/isaac_motion/ha_cmd_exec/packed.pt`（**mode/δ 必须与训练一致**；多轮文档 §4.4）；
- **任务奖励/分级成功**：`reward_from_objects.py`（锤子是否入篮等），只用于评测，不进 WM 训练；
- **奖励形状必须在生成 A' 之前定稿**，否则 WM 的 reward head 要重训两次。

### G2 验收

- 帧覆盖：每条 clip 0..length 全被走过（全 clip 回放天然满足）；
- 动作覆盖：`|a|` 分位数与 WBC 名义动作接近（噪声没把分布拉飞）；
- **参考本身 success 比例 > 0**（验证 `ha_cmd_exec` + 新 WBC 确实能做成任务）；
  **DoubleDesk 参考库实测 99/100 = 99%**（`reward_from_objects.py`），G2 的下限已满足。

---

## 8. P3 本机：重训 WM

### 8.1 奖励与损失（最终 → 见多轮文档）

> **最终执行版：`docs/JOINT_MULTIROUND_PLAN_20261103.md`**。
> 本节只保留 R0 基线与指针；新线以多轮文档为准。

**最终设计（所有轮同一个形状，只退火 δ）**：

- 奖励：**Huber-on-error**（`d=√e`，`h(d;δ)`，远端线性；不是 `e/std²`、不是 exp、不是 hinge）；
- 朝向：**split**（tilt 1.0 / yaw 0.2 / yaw_rate 0.3，每个轴都用 `h`）；
- 速度 / tail / posture / token trust：**全关**；
- critic：**开**（`CRITIC_ON=1`，`CRITIC_SOURCE=imagined`，`ADV_BASELINE=none`）；
- action trust：`ACT_EPS=0.6`、`ACT_WEIGHT=0.2`（球内为 0）；
- `γ=1`、`H=20`、`λ_r=0.3`；
- BC / hand / state / anchor：**全 0**；
- 物体奖励：只评测。

**δ 课程**：

| 轮 | δ_pos / δ_ori | 目标 | B 采集 | WM refresh |
|---|---|---|---|---|
| R1 | 5 cm / 0.05 rad | 2–3 cm | 0 | 不刷 |
| R2 | 2 cm / 0.02 rad | 1 cm | 10–20 集 | 刷一次 |
| R3 | 5 mm / 0.005 rad | 5 mm | 20–30 集 | 刷 |
| R4 | 2 mm / 0.002 rad | 2 mm | 20–30 集 | 刷 |
| R5（条件） | 1 mm / 0.001 rad | 1 mm | 30 集 | 刷（仅当地板 <1mm） |

**代码状态**：`huber_err` + `split_huber` + `--reward-delta-*` **待实现**（R1 前）；
现有 `linear` 只作 R0 基线；`huber`（对平方误差）不用；`mse` 已删除。

### 8.2 WM 大小

- **保持 8.82M MLP，先不动**；
- 判断：训练损失低但 20 步误差高 → 分布/数据问题，加参数无用；
- 训练损失也高 → 才考虑加大；
- **sim-to-sim 的接触动力学 gap 是更大的风险**，不是 MLP 容量。

### 8.3 训练

```bash
# --sim = 由 exp_a_prime/<clip>/merged.pt 全局合并来的 exp_ha_exec_transitions.pt
python3 scripts/make_wm_mix.py \
  --sim runs/wm_real/exp_ha_exec_transitions.pt \
  --bline <B 线转移或留空> --ratio 0.25 --out runs/wm_real/exp_wm_mix_exec.pt
python3 scripts/pretrain_wm.py \
  --joint-ckpt runs/S1_handover/wm_903.pt \
  --transitions runs/wm_real/exp_wm_mix_exec.pt \
  --out runs/S1_handover/exp_wm_exec.pt \
  --steps 6000 --stage1-steps 3000 --roll-steps 1500 --roll-k 20
```

- 新奖励形状 → WM 的 reward head 会用 A' 的新奖励重训；
- B 线留空 = 纯 A'；有 B 时保持 25%；
- **并行保证**：新 WM 写 `runs/S1_handover/exp_wm_exec.pt`，`wm_903.pt` 只读。

### G3（两段：WM 预测 + 闭环覆盖）

**G3a：WM 的 20 步预测误差（held-out 真动作）**

- 随机 held-out 参考起点（6 任务分层，每任务 ≥50 个）；
- 取**真实动作序列**（A'/执行回放，不是策略），真 sim 与 WM 各滚 20 步；
- 报告 `mean / p90 / max`，门控看 **p90 < 0.05 m**；
- 和 `wm_903` 对比，不能更差。

**G3b：VLA 闭环 20 步的偏差是否在 WM 覆盖内**

> 训练时每个起点只推 20 步（H=20），VLA 部署也是每 20 步重规划一次；
> 所以覆盖测试的窗口就是 **20 步**，不是 200。200/2080 步的累积漂移属于 P5，不作为 G3 的门。

- 每个随机参考起点，VLA（当前 head + 新 WBC）**闭环 20 步**（一个完整部署窗口）；
- 起点分两种（都要测）：
  - (a) **参考起点**：训练分布，测"WM 在训练窗口内准不准"；
  - (b) **漂移起点**（可选）：从上一次闭环结束的状态起步，测"部署时起点已经漂了，WM 还准不准"（暴露偏差）；
- 记录三样：
  1. `e_actual = ||s_real − s_ref||`：实际执行 vs 参考（窗口内真实漂移）；
  2. `e_wm = ||s_wm − s_real||`：同样起点+同样动作下，WM 想象 vs 真 sim（模型误差）；
  3. `d_tube`：实际访问的 state/action 到 A'+噪声管道的距离（覆盖度）；
- 判读：

| 观察 | 结论 |
|---|---|
| `e_wm` p90 小（< 0.05 m），且 `d_tube` 在管道内 | WM 覆盖了 20 步窗口的实际偏差 → 可以冻结 WM |
| `e_wm` 大、`d_tube` 出管道（尤其是 (b) 漂移起点） | WM 不覆盖 → 触发一次 on-policy 刷新 |
| `e_actual` 本身很大 | 是策略/参考问题，不是 WM 问题，先回 P1/P2 |

- 成本：16–32 个起点 × 20 步，单卡约 15–30 分钟（可复用 `s3_eval_sweep` 的 dump）。

> G3a 是"WM 像不像仿真器"，G3b 是"训练/部署的 20 步窗口内，WM 有没有覆盖 VLA 的实际状态"。两个都过才算 G3 过。

---

## 9. P4 本机：从 base、参考起点联合微调

```bash
ROUND=1001 RUN_LABEL=exp_ref_base STEPS=200 \
INIT_VLA="" \
SM_CKPT="$PWD/runs/S1_handover/exp_WBC_exec.pt" \
MOTION_PACKED="$PWD/runs/isaac_motion/ha_cmd_exec/packed.pt" \
WM_CKPT="$PWD/runs/S1_handover/exp_wm_exec.pt" \
ADV_BASELINE=none \
bash scripts/run_refstart_joint.sh
```

- 单卡：`NPROC=1 BATCH=8 ACCUM=8`（全局 64），200 步约 2 h；
- token 来自策略（不是库），P4 **不需要** `SM_TOKEN_LIB_TRAIN`；需要的是 `MOTION_PACKED=ha_cmd_exec`；
- 变体：`LAMBDA_S=0.2`（E2）、`TOKEN_BC=1 w=0.1`（E4）；
- **burn-in 两档**：
  - `BURNIN=0`：起点 = 参考原值（严格 0 偏差，验证"起点对齐"本身）；
  - `BURNIN=10`：起点 = 参考 + 冻结 WBC 滚 10 步（贴部署，验证"WBC 漂移"）；
  - 两档都跑，才能分清是"起点偏差"还是"WBC 漂移"；
- 该模式下 `b_start_max_err`（E_start）不生效，帧是全库均匀随机；
- 图像/state/参考同帧（`vla_state_from_burnin=True` 会覆写 state 为起点）。

### G4

- `imag_state`（参考起点 eval）比旧路径明显低，不是 0.002 级。

---

## 10. P5 本机：闭环 + 分级成功

```bash
TASK=DoubleDesk TAG=exp_ref_base_vs_base SEEDS="0 1 2 3 4 5 6 7" \
START_FRAME=0 MAX_STEPS=2080 \
HEAD_MAIN=base HEAD_CTRL="$PWD/runs/S3_loop/exp_ref_base_joint/train_main/vla_final.pt" \
SM_CKPT_MAIN="$PWD/runs/S1_handover/WBC_A24000_wmvel_v6.pt" \
SM_CKPT_CTRL="$PWD/runs/S1_handover/exp_WBC_exec.pt" \
bash scripts/s3_eval_sweep.sh
python3 scripts/eval_drift_curve.py --sweep runs/S3_loop/sweeps/exp_ref_base_vs_base_s*
```

**单卡注意**：两个 server + Isaac 同卡可能 OOM；先试 `EXTRA_*=--sample-avg 1`，
或主/控分两次跑（`GPU_MAIN=GPU_CTRL=0` 串行）。

### G5

| 观察 | 结论 |
|---|---|
| 漂移曲线平 | 参考起点 + 20 步分段成立，可停 B 采集 |
| 曲线线性/指数涨 | 累积误差主导，加 20% 漂移起点 |
| 成功率 > 0 | 第一次真成功，继续 |
| 仍 0 | 回 P1/P2 |

---

## 11. P6 本机：手部

- `s3_collect_arm.sh` 已默认 `SM_HAND=1`；
- 旧 B 数据回放要显式 `SM_HAND=0`；
- 参考手部：`g1_hand_joints_14` 作 `SM_HANDS_NPY`；
- 训练侧 `LAMBDA_HAND=0.1`（base 78 维 head 的 64:78）；
- WM 71 维不含手，手只能靠 BC，奖励里没有抓取信号。

---

## 12. 交接协议（共享目录，无需拷贝）

| 产物 | 谁写 | 路径 | 本机如何验收 |
|---|---|---|---|
| `ha_cmd_exec/packed.pt` | 2 卡 | `runs/isaac_motion/ha_cmd_exec/packed.pt` | `torch.load` 查 shape/names；看 `report.json`；跑 P0c |
| `report.json` | 2 卡 | 同上目录 | FK 误差 < 3 cm |
| WBC 训练库 | 2 卡 | `trackerlab-supermimic/data/bones_seed/bones_seed_plus_ha_exec_x20.pt` | 查 clip 数 143,923、HA 帧 12.68M |
| 新 WBC | 2 卡 | `runs/S1_handover/exp_WBC_exec.pt` | `torch.load` 有 `pi/wm/robot_spec`；跑 G1 复验 |
| G1 报告/视频 | 2 卡 | `runs/exp_g1/` | root < 0.1 m、0 摔倒 |
| 训练日志 | 2 卡 | `trackerlab-supermimic/logs/exp_wbc_exec/train_log.csv` | 曲线不崩、无 NaN |
| A' 转移 | 本机 | 每 clip `runs/wm_real/exp_a_prime/<clip>/merged.pt` → 合并 `runs/wm_real/exp_ha_exec_transitions.pt` | 行数、帧覆盖、奖励分布 |
| 新 WM | 本机 | `runs/S1_handover/exp_wm_exec.pt` | G3a 20 步误差、G3b 覆盖 |

**约定**：两个容器都用**同一绝对路径**；如果某台机器挂载点不同，先 `readlink -f` 对齐再写。

---

## 13. 时间线（并行）

| 时间 | 本机 | 2 卡容器 |
|---|---|---|
| Day 1 上午 | P0a 写代码（3.1–3.8） | 等脚本 |
| Day 1 下午 | 写物体奖励/A'/评测脚本 | P0b 构建 `ha_cmd_exec` + G0 |
| Day 1 晚 | P0c 旧 WBC 对照 | P1a 构建训练库 |
| Day 2 | 写 WM/A' 脚本、文档 | P1b 微调 WBC（warm start 24k → 28k） |
| Day 3 | 等 | P1c G1 验收 |
| Day 3–4 | P2 A'（4 seed ≈ 4–8 h，可先 2 seed）+ P3 WM（1 h） | 待命 |
| Day 4 | P4 联合微调（2 h） | — |
| Day 5 | P5 闭环 + 分析 | — |
| Day 5+ | P6 手部 | — |

---

## 14. 风险与回退

| 风险 | 处理 |
|---|---|
| `ha_cmd_exec` FK 残差大（G0 不过） | 检查 q/root 的坐标系；退回 `ha_cmd` 并记录 |
| `library_kind` 拒绝新库 | **已打** 3.5 的补丁（`command_exec`）；默认行为不变 |
| 2 卡训练 OOM | `ENVS_PER_RANK` 4096→2048 |
| 从 `WBC_A24000` 微调洗不掉旧折中 | G1 不过时改用 `bones_seed_retrain_1025b/ckpt_012500`（bones_seed-only，但要切 `relative_ori99` obs 口径） |
| 奖励形状改了但 WM reward head 没重训 | 生成 A' 前先定稿 Huber δ/cap，再一次性训 reward head |
| WBC 微调后参考回放仍 > 0.1 m | 加步数/加 HA 采样权重；**先查 `SM_TOKEN_LIB_TRAIN` 双库是否开**（否则 token 是执行空间）——第一版就踩了这个坑（G1 1.526 m） |
| 单卡联合微调太慢 | `BATCH=16 ACCUM=4`；或借 2 卡容器跑 |
| 单卡评测 OOM | 主/控分两次跑；`--sample-avg 1` |
| 旧产物被覆盖 | 所有新路径带 `exp_`；旧 ckpt 只读 |

---

## 15. 明确不做

- 不覆盖 `ha_cmd/packed.pt`、`bones_seed_plus_ha_cmd_x20.pt`、`WBC_A24000_wmvel_v6.pt`、`90x_*`、`B_90x`；
- 不把参考 q 换成命令 q（新库用执行 q）；
- 不用闭环 root-mean 做模型选择；
- 不在 G1 不过时进入 P2；
- 不用 `TOKEN_BC_SOURCE=self`（模仿失败）。

---

## 16. 一句话

> **本机写代码 + 后训练/评测；2 卡容器构建 `ha_cmd_exec` + 微调 WBC；共享目录直接交接。
> 先过 G0（参考自洽）、G1（参考回放 < 0.1 m、0 摔倒），再谈 A'/WM/联合微调。**
