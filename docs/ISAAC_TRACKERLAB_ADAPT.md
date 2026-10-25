# 把我们的 SuperMimic 经验迁到 Isaac Lab 版（`trackerlab-supermimic`）

> 目标：在 **Isaac Sim 5.0.0 + Isaac Lab release/2.2.0**（与 HA 数据集一致的唯一版本）上，复现我们已经在
> MuJoCo 上跑通的 token 条件化训练，并把 **64 维 SONIC motion token** 作为策略
> 的 conditioning（而不是上游的「未来 K 帧参考」goal）。
>
> 状态标记：**[已验证]** = 本机 CPU 实跑过；**[未验证]** = 只有静态代码依据，
> 必须在装了 Isaac Lab 的机器上跑第一条命令才算数。

---

## 0. 两套栈的对照

| 维度 | 我们现有（MuJoCo 侧） | `trackerlab-supermimic`（Isaac Lab 侧） |
|---|---|---|
| 物理 | mjlab / MuJoCo-Warp（训练世界模型） | **PhysX**（Isaac Sim 5.0.0，与 HA 数据集一致） |
| 机器人 | G1 29 DoF（+14 手，仅我们这条链有） | G1 29 DoF（无手） |
| 策略条件 | **64-d SONIC token**（`PolicyCfg.token_dim`） | K=10 帧未来参考 → 730 维 goal |
| 世界模型 | `supermimic.world_model.WorldModel`（神经网络） | `networks.WorldModel`（同族：invariant features → 增量） |
| 观测 | 93 维（joint_pos/vel, ang_vel, gravity, last_action）× 10 帧 | 93 / 99 维（多 6 维参考相对朝向）× 10 帧 |
| 动作 | 归一化 29 维 → `default_q + action_scale * a` | 同（PhysX joint target） |
| 训练算法 | BPTT（可微世界模型反传） | BPTT（同上，runner.py） |
| 奖励 | 6 项 tracking + action rate + joint limit | **完全同构**（`mdp/supermimic.py::MDP.reward`） |

结论：**算法同族、奖励同构、观测同构**，唯一的实质差异就是「策略的条件输入」。
所以适配的核心是给 MotionLib/Policy 接一条 token 通路，其余不动。

---

## 1. 硬件/软件前提（本机预检结果）

| 项 | 本机状态 | 结论 |
|---|---|---|
| GPU | 2× `NVIDIA GeForce RTX 4090`（**49140 MiB**，48 GB 版），驱动 595.71.05 | ✅ 有 RT core，Isaac Sim 支持 |
| Vulkan | `libvulkan.so.1` ✅，ICD `/etc/vulkan/icd.d/nvidia_icd.json` ✅ | ✅ 渲染可用 |
| EGL | `libEGL.so.1` ✅ | ✅ 无头渲染可用 |
| Python | 系统只有 3.12.3 | ❌ 需要 **3.11** 环境（Isaac Lab 要求） |
| Isaac Sim | **5.0.0 已装**（`third_party/isaacsim_env`） | ✅ 与 HA 数据集录制版本一致，见 §8.7 |
| 显存预算 | 上游参考：8096 envs 时 PyTorch 峰值 **18.7 GiB** | 48 GB 单卡可跑 8096，甚至更多 |

### H200 能不能训？——**不能**

Isaac Sim 要求 **RTX 系列 GPU（RT core）**，H200/H100 这类数据中心卡没有 RT core，
Isaac Sim 起渲染设备时会直接失败（典型报错 `gpu.foundation.plugin: No device could
be created`）。所以这套 Isaac Lab 训练**只能在 4090 上跑**，H200 那条路走不通。
参考：[Isaac Sim Requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)。

---

## 2. 我们改了什么（逐文件）

所有改动都在 `trackerlab-supermimic/` 里，共 6 个文件 + 3 个新文件：

| 文件 | 改动 | 作用 |
|---|---|---|
| `source/trackerLab/.../motion_lib/supermimic.py` | `__init__` 加 `token_dir`；载入 `<clip>.tokens.npy` → `self.tokens`；新增 `token(f)` / `token_dim`；`from_dir` 读 `meta.json` 的 `src_fps` 与 token 开关 | 数据侧支持 64 维 token |
| `source/trackerLab/.../motion_buffer/supermimic.py` | 新增 `token()` | 当前帧 token |
| `source/trackerLab/.../managers/supermimic.py` | 新增 `token()` | env 侧取命令 |
| `source/trackerLab/.../algorithms/supermimic/runner.py` | `TOKEN_MODE`；`GOAL_DIM = 64`；新增 `goal_of(frames)`；3 处取 goal 改为 `goal_of` / `motion_manager.token()` | 训练图接 token |
| `scripts/rls/supermimic/train.py` | 新增 `--interface {motion,token}` | 实验开关 |
| `scripts/rls/supermimic/evaluate.py`、`export_policy.py` | 按保存的 `interface` 选择 conditioning 与 `goal_dim` | 评测/导出一致 |
| **新** `scripts/rls/supermimic/prepare_token_data.py` | 我们的 npz → 上游 MotionLib 格式 + token sidecar | 数据转换 |
| **新** `scripts/rls/supermimic/check_token_interface.py` | 不依赖 Isaac 的自检 | 见 §4 |
| **新** 本文档 | — | — |

关键设计：**token sidecar 与 CSV 同名同目录**（`<clip>.csv` + `<clip>.tokens.npy`），
`meta.json` 里记 `src_fps` 和 `token_dim`，因此 `MotionLib.from_dir()` 能自动识别，
不需要新增命令行参数。

---

## 3. 数据转换

我们的 npz（`data/npz/g1_tasks` 或 `data/npz/ha_tokens`）里已经有三样东西：
`body_pos_w/body_quat_w`（取 `pelvis` 当 root）、`joint_pos`（29）、
**`g1_motion_token_64`（64 维 token）**。

```bash
cd trackerlab-supermimic

# 我们的 318 条 SONIC 任务 clip
python3 scripts/rls/supermimic/prepare_token_data.py \
    --src ../data/npz/g1_tasks --out data/token_motions/g1_tasks

# HumanoidArena 的 699 条
python3 scripts/rls/supermimic/prepare_token_data.py \
    --src ../data/npz/ha_tokens --out data/token_motions/ha
```

输出：

```
<out>/meta.json            {"src_fps": 50, "num_clips": N, "token_dim": 64, ...}
<out>/<clip>.csv           T×36  root_xyz(3) + root_quat_xyzw(4) + joint_pos(29)
<out>/<clip>.tokens.npy    T×64
<out>/<clip>.hands.npy     T×14（可选）
<out>/manifest.jsonl       每条 clip 的帧数/时长/指令
```

三点必须知道的口径：

1. **四元数顺序**：上游 CSV 要 **XYZW**，我们的 npz 是 Isaac 的 **WXYZ**，转换器已换；
2. **帧率**：我们数据是 50 Hz，上游 `MotionLib` 默认 `src_fps=30`；转换器把 50 写进
   `meta.json`，`from_dir()` 会读，**不要**手工传 30；
3. **关节顺序**：已核对我们的 `joint_names` 与上游 `G1_JOINT_NAMES` **完全一致**
   （29 个，逐项相同）[已验证]。

---

## 4. 不装 Isaac 也能验的部分

```bash
python3 scripts/rls/supermimic/check_token_interface.py \
    --motion_dir /tmp/tm_smoke --src_dir ../data/npz/g1_tasks --limit 3
```

本机实跑结果 [已验证]：

```
[data] clips=3 frames=9466 token_dim=64 src_fps=50.0
[motion] frames=9466 clips=3 token_dim=64
[token] frames 0..3 aligned, mid-frame 4733 -> clip 1[1740] aligned
[policy] Policy(obs_dim=99, goal_dim=64) -> action (5, 29)
[goal] motion interface dim=730, token interface dim=64
[ok] token interface plumbing verified (CPU only, no Isaac required)
```

它验证了：数据转换 → `MotionLib` 载入 token → 帧对齐（含跨 clip 偏移）→
`Policy(obs_dim, 64)` 前向出 29 维动作。

---

## 5. 训练命令（装好 Isaac Lab 之后）

```bash
cd trackerlab-supermimic
python3 -m pip install -r requirements.txt && python3 -m pip install --no-deps -e .

# 先 4 环境冒烟（上游自带 preset），确认机器能起 Isaac Sim
python3 scripts/rls/supermimic/train.py --task g1 --num_envs 4 --iterations 2 \
    --motion_dir data/token_motions/g1_tasks --interface token \
    --log_dir logs/token_smoke

# 正式跑：token 条件化，8096 环境（48 GB 卡余量足够）
python3 scripts/rls/supermimic/train.py --task g1 \
    --num_envs 8096 --iterations 320000 \
    --motion_dir data/token_motions/g1_tasks --interface token \
    --goal_frames 10 --hist 10 --horizon 32 \
    --utd 0.4 --lr 2e-4 --seed 0 \
    --log_dir logs/token_g1_tasks

# 评测 / 导出（会自动按保存的 interface 走 token 通路）
python3 scripts/rls/supermimic/evaluate.py --task g1 --run_dir logs/token_g1_tasks \
    --motion_dir data/token_motions/g1_tasks
python3 scripts/rls/supermimic/export_policy.py --run_dir logs/token_g1_tasks
```

`--interface token` 时 `GOAL_DIM=64`，`goal_frames` 对策略无效（仍会写进 task_cfg，
不影响）；`critic` 输入维度自动跟着变。

---

## 6. 哪些「MuJoCo 经验」真能搬，哪些搬不了

**能搬（已经/直接可用）**

| 经验 | 迁移方式 |
|---|---|
| 奖励与终止设计 | 上游 `MDP.reward/termination` 与本项目同构，直接用；我们的 `state_term` 权重表可对照 |
| 世界模型思路（invariant features → 增量） | 上游已是同族实现，无需改 |
| **token 条件化接口** | 本次改造（§2） |
| 数据管线（npz：token + refpose + 手） | `prepare_token_data.py` |
| 训练配方（BPTT、H=32、utd=0.4、lr 2e-4、warmup、wm_only 500 步） | 用上游默认 + 我们在 `joint_sm24k_hand` 的 `--hand-loss-weight 0.2 / grasp-boost 5` 类比 |
| 评测口径（rollout reward、state term） | 上游 `metrics/supermimic.py` 已实现 |

**搬不了**

| 项 | 原因 |
|---|---|
| 我们的 Policy/WorldModel **权重** | 层形状不同（我们 `encoder_hidden/decoder_hidden`；上游 `[2048,1024,512,256]`），不能直接加载，只能**重训** |
| 手部 14 维 | 上游 G1 没有手；我们的 `g1_hand_joints_14` 只做了 sidecar，尚未接入 actor 输出 |
| MuJoCo 的执行层数值 | PhysX 的接触/驱动参数不同，同一个策略的绝对 reward 会漂 |

所以这次的定位是：**用 Isaac Lab 重训一个 token 条件化的 SuperMimic WBC**，
它和我们 MuJoCo 侧的策略共享接口与奖励口径，但不共享权重。

---

## 7. 验收清单（按顺序做）

1. [ ] 建 Python 3.11 环境，装 **Isaac Lab release/2.2.0 + Isaac Sim 5.0.0**（与 HA 数据集一致，见 §9）；
2. [ ] `python3 -c "import isaaclab"` 通过；跑上游 `--num_envs 4` 冒烟；
3. [ ] `prepare_token_data.py` 转换 318 条，`check_token_interface.py` 通过；
4. [ ] `--interface token --num_envs 64` 跑 50 迭代，确认 loss 有限、`goal_dim=64` 生效；
5. [ ] 8096 环境正式训练，先跑 2000 迭代看 `full-clip normalized return`（上游参考：2000 迭代
       0.5416 / 12-40 clips 失败；32000 迭代 0.8695 / 0 失败）；
6. [ ] 训练完导出 ONNX/portable policy，接我们的闭环（届时执行层就是同一个策略，训练=部署）。

---

## 8. 4 卡方案：B + C 一起走（**代码已落地，CPU 部分已验证**）

### 8.1 架构

```
torchrun --nproc_per_node=4
│
├─ rank 0: Isaac 仿真(envs/4, clip 0,4,8…) + 本地 replay buffer ─┐
├─ rank 1: Isaac 仿真(envs/4, clip 1,5,9…) + 本地 replay buffer ─┤ 梯度 all_reduce
├─ rank 2: Isaac 仿真(envs/4, clip 2,6,10…)+ 本地 replay buffer ─┤ (B) → 各自 step
└─ rank 3: Isaac 仿真(envs/4, clip 3,7,11…)+ 本地 replay buffer ─┘
```

- **C（分片仿真）**：每 rank 一个 Isaac 实例 → 环境数随卡数线性扩展；每 rank 只载 `1/N` 运动库
  → **同时解决显存**。
- **B（DDP 更新）**：world model / policy / critic 的梯度在 rank 间平均（**先平均、后裁剪**，
  与 Isaac Lab rsl_rl 口径一致），等效于单进程大 batch。
- 只有 rank 0 写日志/checkpoint/summary；每个 rank 都参与梯度归约（不会被跳过）。

### 8.2 改动清单

| 文件 | 类型 | 内容 |
|---|---|---|
| `source/.../algorithms/supermimic/distributed.py` | 新增 | `init_distributed` / `broadcast_modules` / `allreduce_grads` / `average_scalars` / `shard_clips` / `shard_clip_count` |
| `source/.../motion_buffer/motion_lib/compact.py` | 新增 | `CompactMotionLib`：只存 `ref (N,71) fp16`，`body_*` 走 `kin.forward` 按需算；其余公开接口与 `MotionLib` 完全一致 |
| `source/.../algorithms/supermimic/runner.py` | 修改 | dist 初始化 + 每 rank 设备；运动库分片/紧凑化；初始参数广播；**3 处 optimizer.step 前 allreduce**；所有文件 IO 收敛到 rank 0 |
| `scripts/rls/supermimic/train.py` | 修改 | `--compact_library/--no-compact_library`、`--shard_motion`、`--shard_mode`；torchrun 下按 `LOCAL_RANK` 设 `args.device`（Isaac 也渲染在对应卡上） |
| `scripts/rls/supermimic/check_compact_library.py` | 新增 | 与预计算版逐字段比对 + 显存报告 |
| `scripts/rls/supermimic/check_distributed.py` | 新增 | 2 进程 gloo 验证广播/归约/分片 |
| `scripts/rls/supermimic/train_4gpu.sh` | 新增 | `MODE=bc`（B+C）/`seeds`（4 seed）/`prof`（瓶颈测量 + env 扫描） |

### 8.3 显存账（BONES-SEED 全量 47,475,680 帧；数字为**实测**，按 14 个 tracked body）

| 项 | 单卡（不分片） | 4 rank 分片 |
|---|---|---|
| 预计算 FK 的 `MotionLib`（**实测 1340 B/帧**，14 个 tracked body） | **59.2 GiB** ❌ | 14.8 GiB/卡（勉强，且 buffer 会顶到边缘） |
| **紧凑库**（`ref` fp16，142 B/帧） | **6.3 GiB** ✅ | **1.6 GiB/卡** ✅ |
| replay buffer（T=1000，N=envs） | — | ~4.4 GiB/卡 @4096 envs（~1.07 KB/环境步） |
| 网络 + 优化器（wm/pi/critic） | — | ~0.5 GiB/卡 |
| PhysX/Isaac 自身 | — | 参考：8096 envs 时 PyTorch 峰值 18.7 GiB |

→ 48 GB 卡上，B+C 用 `ENVS_PER_RANK=4096…8096` 都有充足余量。

### 8.4 预期加速与前提

- 数据侧（采集）随卡数**近线性**加速；更新侧（wm/policy/critic）加速取决于梯度归约占比
  → 端到端预计 **2.5–3.5×**（4 卡）。
- **前提是先测清瓶颈**：`MODE=prof` 会给每个 env 数打印
  `[prof] collect … | wm … | policy+critic …`。
  - `collect` 占大头 → C 的收益最大；
  - `wm`+`policy` 占大头 → B 的收益最大（此时也可考虑把 `--wm_batches` 调大/调小）。
- 为什么不能指望 4×：Isaac 的仿真步进在时间维度上是串行的，梯度归约与 buffer 采样也有开销。

### 8.5 命令

```bash
# 0) 数据（见 §3）
python3 scripts/rls/supermimic/prepare_token_data.py \
    --src ../data/npz/ha_tokens --out data/token_motions/ha

# 1) 先测瓶颈（60 迭代 × 3 个 env 数，输出 [prof] 三段耗时）
MODE=prof bash scripts/rls/supermimic/train_4gpu.sh

# 2) B+C：4 rank × 4096 envs，分片库 + DDP
MODE=bc NPROC=4 ENVS_PER_RANK=4096 MOTION=data/token_motions/ha \
    LOG_DIR=logs/bc_ha bash scripts/rls/supermimic/train_4gpu.sh

# 3) A：4 个 seed 并行（要统计量时）
MODE=seeds NPROC=4 ENVS_PER_RANK=8096 LOG_DIR=logs/seed0 bash scripts/rls/supermimic/train_4gpu.sh
```

### 8.6 验证状态

| 项 | 状态 |
|---|---|
| 紧凑库与预计算库逐字段一致（root/body/goal/token/ref_state） | **[已验证]** fp16 舍入内，max 1.5e-3 |
| 紧凑库显存报告（BONES-SEED 110.2 GiB → 6.3 GiB） | **[已验证]** |
| 广播 / 梯度平均（含某 rank `grad=None` 的边界）/ 分片覆盖 / 标量平均 | **[已验证]** 2 进程 gloo |
| runner/train 的分布式接线 | **[未验证]** 需要装好 Isaac Lab 才能真跑 |
| 4 卡端到端加速比 | **[待测]** 由 `MODE=prof` 决定，预计 2.5–3.5× |

---

## 9. 版本统一（最终决定，2026-10-24）

**训练与评测都跑 `isaacsim 5.0.0` + `Isaac Lab release/2.2.0`。**

- 依据：HA 数据集就是这两个版本录的（`ISAACSIM_VERSION=5.0.0` / `ISAACLAB_GIT_REF=release/2.2.0`）。
- trackerlab 上游 pin 是 5.1/2.3.2，但它对 Isaac Lab 的 API 依赖只有 **6 个模块 / 10 个符号**，
  **在 2.2 全树都能解析**（静态比对，模块级差异 0），因此训练侧落到 2.2 没有 API 阻塞。
- 5.0 venv 里已知要补的 python 依赖：`flatdict`（2.2 的 `isaaclab.sim` 需要），
  加上 `trackerlab-supermimic/requirements.txt` 的 `mujoco / scipy / matplotlib / pillow /
  imageio-ffmpeg / opencv-python-headless`。
- **残余风险**：本机没有在 5.0 上实跑过（按要求不跑训练），第一次在目标容器跑时先过
  `scripts/rls/supermimic/smoke_isaac.sh`；不要拿上游 5.1 的 README 数值当基线。
