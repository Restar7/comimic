# 跨容器交接与纠错（给另一个容器里的 DSH）

> 这份文档是给**不在我这台机器上**的 DSH 看的。我这边装的东西、跑的进程、绝对路径，对你不生效；
> 你要拿的是**判断和契约**，路径按你容器重映射。
>
> 阅读顺序：§1 纠错清单（先看，你很可能已经想错了）→ §2 事实基座 → §3 版本一致性 →
> §4 你必须在目标容器自查的 8 项 → §5/§6 契约与边界 → §7 消融口径 → §8 卡数与时间。

标记约定：
- **[实测]** = 我在本容器真跑过、有输出为证
- **[上游依据]** = 来自仓库代码/文档，可复核
- **[待你确认]** = 依赖你容器的环境，必须自己跑一遍

---

## 1. 纠错清单（如果你之前理解成左边，那就是错的）

| # | ❌ 错误理解 | ✅ 正确事实 | 依据 |
|---|---|---|---|
| 1 | 「SuperMimic 是在 Isaac Lab 里训的」 | **SuperMimic 官方口径是 mjlab / MuJoCo-Warp**。mjlab 只是 **Isaac-Lab 风格的 API**，物理是 MuJoCo Warp。公开复现全是 mjlab/UniLab 系，**没有 Isaac Lab 物理的实现** | `docs/REPRODUCTION_PLAN.md`、`docs/PAPER_SPEC.md`、`bones_seed` 训练实跑环境 `mujoco 3.16 / mjlab 1.6 / warp 1.17` **[上游依据]** |
| 2 | 「训练时用的是 SONIC 的 decoder」 | 训练时 encoder/decoder **都是 SuperMimic 自己的 MLP**（`PolicyCfg.encoder_hidden/decoder_hidden`）。训练链：`VLA token → SuperMimic Policy → SuperMimic WorldModel → 解析 reward → 反传`，**纯 torch、无可微物理引擎、无 SONIC decoder** | `supermimic/policy/nets.py`；`build_g1_assets()` 只加载 MuJoCo 模型+FK，零 ONNX **[实测]** |
| 3 | 「SONIC decoder 是 SuperMimic 的」 | SONIC 提供两样：64 维 token **encoder**（`model_encoder.onnx`，两仓库同一文件）、部署用 **decoder**（`model_decoder.onnx`，输入 = token64 + 10 帧历史 = 994 维 → 29 关节）。后者只出现在**闭环部署**，不在训练里 | `groot_repro/sonic_loop.py:4` **[上游依据]** |
| 4 | 「评测用 MuJoCo 就行」 | 只要你要**和 HA 论文/数据集对得上**，评测必须在 **Isaac Sim 5.0 + Isaac Lab 2.2** 里渲染 —— HA 的训练观测是作者在 Isaac 里渲染的 mp4，换渲染器就是把 VLA 推到 OOD 输入上 | HA `docs/04_environment_setup.md` **[上游依据]** |
| 5 | 「训练侧用 5.1、评测侧用 5.0 就行」 | **不要两套版本**。已判定：统一到 **isaacsim 5.0.0 + Isaac Lab 2.2.0**（= HA 数据集的录制版本）。trackerlab 上游 pin 的 5.1/2.3.2 只是它的验证环境，静态比对显示它只用 6 个 isaaclab 模块 / 10 个符号，**2.2 里全都有**，所以训练也能上 5.0/2.2。依据与残余风险见 §3 | 静态比对 + `resources.lock.json` |
| 6 | 「H200 也能训 Isaac Lab」 | **不能**。Isaac Sim 要求 RTX 系列（RT core），H100/H200 无 RT core，起渲染设备直接失败。**这套只能在 4090 上训** | Isaac Sim Requirements **[上游依据]** |
| 7 | 「4 卡能让单次训练快 4 倍」 | trackerlab 的 runner **单进程单卡**（代码里没有 torchrun/DDP/world_size 任何入口）。4 卡要么并行 4 个 seed（墙钟不变），要么用我们实现的 **B+C**（分片仿真 + 梯度平均）拿 **2.5–3.5×** | `grep -rniE "torchrun|distributed|world_size|ddp"` 在 `source/ scripts/ docs/ README` 里零命中 **[实测]** |
| 8 | 「BONES-SEED 直接丢进 trackerlab 的 MotionLib 就行」 | **装不下**。trackerlab 的 `MotionLib` 预计算 FK，**实测 1340 B/帧（14 tracked body）** → BONES-SEED 47,475,680 帧 = **59.2 GiB**（48 GB 卡放不下）。必须用**紧凑库**（只存 `ref (N,71) fp16` = 6.3 GiB，FK 按 batch 实时算） | `check_compact_library.py` **[实测]** |
| 9 | 「帧数 = floor((T-1)/src_fps/CONTROL_DT)+1 没问题」 | **有 bug**：50 Hz 数据上浮点误差会少 1 帧（`78.24/0.02 = 3911.9999999999995`），**318 条里 27 条**中招；token sidecar 按源帧数写 → 直接错位一帧/断言失败。修法见 §5.3 | **[实测]** 27/318 → 修后 0/318 |
| 10 | 「`vla_<step>.pt` 可以直接当 checkpoint 推理」 | 不行。它是 `{"step","fm"}`（只含可训练参数，6.4 GB），必须用 `scripts/export_vla_ckpt.py` 合并回 GR00T checkpoint 目录（只重写被改的 shard，其余软链） | **[实测]** 537/537 张量校验通过 |
| 11 | 「`import isaacsim` 报 EOFError = 安装失败」 | 那是 **EULA 交互提示**。`export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y` 即可，包其实已经装好了 | **[实测]** |
| 12 | 「BONES-SEED ≈ 一个 40h 数据集，训练时长线性于数据量」 | 论文自己的扫描显示**饱和**：到目标回报的 GPU 小时 = 0.5 / 1.3 / 1.8 / **1.9**（对应 1 / 7 / 40 / 264 h）。所以 40 h 子集和全量几乎同价 —— **省时间要动吞吐（envs/卡），不是砍数据** | `PAPER_SPEC.md` Fig.3b **[上游依据]** |

---

## 2. 事实基座（组件与数据，别搞混）

### 2.1 谁是谁

| 组件 | 归属 | 训练/来源 | 出现在哪 |
|---|---|---|---|
| VLA（Cosmos-Reason2-2B + GR00T-N1.7 action head） | 我们 | 官方 GR00T 微调（纯监督，无物理引擎） | 训练 + 评测 |
| 64 维 motion token **encoder** | SONIC | Isaac Lab（PPO+aux）导出 ONNX | 只用于**造 token 监督** |
| **部署 decoder**（token→29 关节） | SONIC | 同上 | 只在**闭环部署** |
| **WBC 策略**（token 条件化 `Policy`）+ **世界模型** | SuperMimic | mjlab / MuJoCo-Warp | 只在**训练图** |

⚠️ 由此得出本项目**当前最大的方法学软肋**：训练优化的是 SuperMimic 那套，部署执行的是 SONIC decoder —— **训练代理 ≠ 部署执行器**。三种收口方式见 §3.2。

### 2.2 两个数据集别混

| 用途 | 数据集 | 规模 | 存储形态 |
|---|---|---|---|
| **WBC 训练** | **BONES-SEED** | 129,943 clips / 47,475,680 帧 / **263.75 h** | `bones_seed_compact.pt` = 6.28 GiB，只有 `ref (47.5M, 71) float16`（root p/quat/v/w + 29 关节 q/qd），FK 实时算 |
| **VLA 微调** | SONIC ft318 | 318 clips / 473,876 帧 / 2.63 h | npz（token64 + 手 14 + ego_view + 指令） |

### 2.3 「和 arena 完全统一」到底指什么：四层核对表

**先说结论：版本那一层（L1）已经统一到 arena 的 Isaac Sim 5.0.0 + Isaac Lab 2.2.0，控制频率（L2）也一致；唯一还有差异的是机器人资产（L3，两边不是同一个 USD），场景/道具（L4）天然不同且不需要一致。**

下表全部来自**读仓库 pin / 读代码**（未跑仿真）：

| 层 | 项 | trackerlab（训练侧） | HumanoidArena | 判定 | 该做什么 |
|---|---|---|---|---|---|
| **L1** | 仿真器版本 | **已跑在 `isaacsim_env` = Isaac Sim 5.0.0 + `isaaclab_220` = Isaac Lab release/2.2.0**（2026-10-24 09:39 起的最新训练，日志实证） | `isaacsim 5.0.0` + `IsaacLab release/2.2.0`（HA `setup_humanoidarena_envs.sh`、`docs/04_environment_setup.md`） | ✅ **已统一（与 arena 同版本）** | 无需换版本。上游 pin 的 5.1/2.3.2 只是它的验证组合，§3 的静态判定说明为什么 2.2 也能跑 |
| **L2** | 控制频率 | `physics_dt=0.005`、`control_decimation=4`、`control_dt=0.02`（`task_cfg.py`） | `sim.dt=0.005`、`decimation=4`（= 50 Hz） | **一致** ✅ | 不用动；换版本后复核一遍即可 |
| **L3** | 机器人资产 | `assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd`，**29 关节、无手**，14 个 tracked body | `g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd`，**29 体关节 + 2×7 手 = 43 DoF**，base-fixed | **不一致**（不同 USD 文件；关节数同为 29，但质量/惯量/碰撞体不保证相同） | 二选一：① 接受（WBC 只跟踪 29 个体关节，手不参与）；② 换成 arena 的 USD，并核对 29 个关节名与顺序 |
| **L4** | 场景 / 道具 / 成功判定 | **无**（空场景，纯运动跟踪） | 7 个任务场景 + 道具，成功判定读 USD prim 的 AABB/距离/力 | **天然不一致，且不需要一致** | 论文里写明分层：WBC 在空场景学跟踪；场景只影响 VLA 的视觉与任务成功判定 |
| — | 观测/动作接口 | 本体感知 99×10 → 29 关节目标 | front RGB + 64 维 state → 40 维 refpose（+2 手） | **不需要一致**（分层结构：VLA 在上，WBC 在下） | — |

**一句话回答「是不是完全统一」**：
> **L1 已统一**（最新训练实测跑在 Isaac Sim 5.0.0 + Isaac Lab 2.2.0，与 arena 同版本）；**L2 一致**；
> **L3 资产不同、需人工决定要不要换**；**L4 不需要统一**。
> 所以「版本/环境对齐 arena」这件事**已经做到**；剩下的只有 L3（机器人资产）这一个可选项。

### 2.4 最新训练实测数据（2026-10-24 09:39–09:59，来自 runner 日志）

| 项 | 实测值 |
|---|---|
| 环境 | `third_party/isaacsim_env`（**isaacsim 5.0.0**）+ `third_party/isaaclab_220`（**Isaac Lab release/2.2.0**） |
| 并行 | **B+C**：2 ranks × 8192 envs，动作库按 rank 分片，网络 DDP 平均梯度 |
| 动作库 | packed BONES-SEED（`data/bones_seed/bones_seed_compact.pt`）：129,943 clips / 47,475,680 帧 / 263.75 h → 每 rank 64,972 clips / **131.8 h** |
| 速度 | **≈1.06 s/iteration**（稳态 0.87 it/s；policy 更新从 iter 500 起，之前更快） |
| 显存 | peak **20.2 GiB / rank**（8192 envs） |
| 换算 | 1000 迭代 = 0.295 h → **20k 迭代 ≈ 5.9 h**，**32k 迭代 ≈ 9.4 h** |
| 学习信号 | iter 500 起 `pi` 开始更新（pi loss −1.05 → −5.26），`surv` 0.30 → 0.94，`ep_len` 38 → 215，`ep_ret` 0.44 → 0.52 |

⚠️ 两处容易误读的地方：

1. `<run>/args.json` 里的 `motion_dir` 仍是 `data/retargeted/lafan1_g1`（该目录里 0 个 csv），
   但**实际加载的是 packed BONES-SEED** —— 以 `motion_library.json` 的 `provenance` 为准
   （`packed library bones_seed_compact.pt`）。这是字段滞后，建议把 `args.json` 的 motion 字段改成实际 packed 路径。
2. 训练在**另一台服务器**上、写的是共享盘，所以**本机 `pgrep` 看不到进程**；
   判断「是否在跑」只能看日志 mtime / 最后一行迭代数，不能看进程表。

### 2.5 手的问题：无手数据 / 带手资产 / 直出手，各影响哪一层

已核实的三条事实：

| 事实 | 证据 |
|---|---|
| **BONES-SEED 没有手** | `bones_seed_compact.pt` 的 `ref (47,475,680, **71**)` = 13(root) + 29(q) + 29(qd)，无手维度 |
| **我们的手标签来自另一套数据** | `data/npz/*` 的 `g1_hand_joints_14 (T, 14)`（7+7）；HA raw 里是 `hand_action_left/right (T,7)` |
| **VLA 直出手，不经过 WBC** | `train_joint.py` 的手辅助损失直接回归 `full[..., 64:78]`；WBC 的 action 只有 29 体关节 |
| arena 资产**带手** | `closed_loop/assets/ha_assets/robots/g1-29dof_wholebody_dex3/...`（Dex3 手 mesh + URDF） |

**结论分三层，影响完全不同：**

| 层 | 影响 | 判断 |
|---|---|---|
| **L-数据**「BONES-SEED 无手」 | 对 WBC：无影响（输出 29 体关节，手不在动作空间）<br>对 VLA：无影响（手标签来自 HA/`g1_tasks`） | ✅ 不构成缺陷 |
| **L-资产**「训练 USD 无手 vs arena USD 带手」 | 腕部多出 7×2 个连杆（数百克级）→ 臂摆惯量、腕力矩、手-物接触都变 | ⚠️ **小而可测的动力学差**；建议直接用带手 USD 训练消除 |
| **L-接口**「手动作语义」 | 我们 VLA 出 **14 维连续**（7+7 关节角）；arena 官方契约 `vla_action[38:40]` 是 **2 维二值**（开合） | ❗**最要紧的一条**：抓取类任务（抓杯/搬箱/双桌）成败几乎全看手 |

**为什么不建议靠推理给「资产差」结案**：tracking reward 主要吃 root 与关节位置，腕部质量只通过手臂链间接影响；量级大概率小，
但**具体多少只能 A/B 实测**（无手 vs 带手 USD，各 1–2k 迭代，比 tracking return 与关节误差）。

**建议（按优先级）**
1. **WBC 训练换 arena 的带手 USD**，手关节用默认位形 + 高刚度 PD 锁住、不进 policy action —— 训练动力学与评测一致，且不给策略增加学习负担。
2. **手输出按评测契约决定**：要保持与 arena 论文可比 → 把 14 维连续映射成 2 维二值（阈值化或加 hand head）；要保留 14 维连续 → 在 arena 侧实现 Dex3 关节驱动，并标注**绝对成功率不可与论文比**。
3. 资产差的量化：跑一次 1–2k 迭代的 A/B 即可。

---

**目标服务器上怎么取证据（只读，不训练、不启动仿真）**：

```bash
# 版本 + 控制参数 + 机器人契约 + 运动数据，一次打出来，FAIL 项直接列出
python3 trackerlab-supermimic/scripts/rls/supermimic/check_env_consistency.py
```
（该脚本只 import 配置、读 `meta.json`，**不创建 env、不 step 物理、不训练**。它已经把上表的
L1/L2/L3 逐项对 arena 参考值打 PASS/FAIL。）

**5.0 venv 里已知要补的依赖**（5.1→5.0 换版本时唯一的额外工作）：
`flatdict`（Isaac Lab 2.2 的 `isaaclab.sim` 需要）+ `trackerlab-supermimic/requirements.txt` 的
`mujoco==3.3.7 / scipy / matplotlib / pillow / imageio-ffmpeg / opencv-python-headless`。

---

## 3. 版本决策：统一到 Isaac Sim 5.0.0 + Isaac Lab 2.2.0

### 3.1 决策

**训练与评测都用 `isaacsim 5.0.0` + `Isaac Lab release/2.2.0`，不保留第二个版本。**

理由：**HA 数据集是 Isaac Sim 5.0 + Isaac Lab 2.2 录制的**（HA 仓库
`isaaclab_twist2_g1/tools/setup_humanoidarena_envs.sh` 里 `ISAACSIM_VERSION=5.0.0`、
`ISAACLAB_GIT_REF=release/2.2.0`；`docs/04_environment_setup.md` 同）。观测、资产、物理
三者只有和录制版本一致才对得上，所以 5.0/2.2 是基准，而不是可选项。

### 3.2 为什么训练侧也能落 5.0/2.2（静态判定，未在此机器实跑）

trackerlab-supermimic 的上游 pin 是 `isaacsim 5.1.0.0 + Isaac Lab v2.3.2`
（`scripts/rls/supermimic/resources.lock.json` 的 `runtime`），但它的**真实 API 依赖面很小**：

| 检查项 | 结果 |
|---|---|
| 从 `isaaclab` 导入的模块 | **6 个**：`actuators` / `app` / `assets` / `scene` / `sim` / `utils` |
| 从这些模块导入的符号 | **10 个**：`AppLauncher`、`ArticulationCfg`、`AssetBaseCfg`、`DCMotorCfg`、`ImplicitActuatorCfg`、`InteractiveScene`、`InteractiveSceneCfg`、`SimulationCfg`、`SimulationContext`、`configclass` |
| 这些模块在 2.2 与 2.3.2 中是否都存在 | **都存在**（模块级差异 0；符号级在 2.2 全树可解析） |
| 有没有用到 2.3 独有 API | **静态扫描未发现** |

所以 trackerlab 落到 2.2 上是**没有 API 阻塞**的；上游选 5.1 只是他们自己的验证组合。

### 3.3 残余风险（必须在目标容器第一次跑时确认）

1. **未在 5.0/2.2 上实跑过**（本机按要求不跑训练）。我这边尝试时唯一的报错是**缺 python 依赖**
   `ModuleNotFoundError: No module named 'flatdict'`（2.2 的 `isaaclab.sim.simulation_context` 需要它），
   **不是 API 不兼容**。目标容器装齐下面这些即可：
   ```bash
   uv pip install flatdict hidapi prettytable toml einops transformers                   mujoco==3.3.7 scipy matplotlib pillow imageio-ffmpeg opencv-python-headless
   ```
   （`mujoco`/`scipy`/`matplotlib`/`pillow`/`imageio-ffmpeg`/`opencv-python-headless` 来自
   `trackerlab-supermimic/requirements.txt`；`flatdict` 等来自 Isaac Lab 2.2 自身依赖。）
2. **行为差异**：API 存在 ≠ 配置字段默认值/弃用行为完全一致。所以第一次跑要按 §4 的 8 项自查，
   重点看 `smoke_isaac.sh` 是否一次通过。
3. **不要拿上游 5.1 的数值当基线**：trackerlab README 里 `0.5416 @2000 it`、`0.8695 @32000 it`
   是他们在 5.1 上跑的，我们换到 5.0 只能做**内部对比**，不能直接对齐那些数字。

### 3.4 和「执行器不一致」这件事要分开看

版本统一解决的是**同版本物理**；它**不**解决另一个更根本的问题：

```
训练图：VLA token → SuperMimic Policy → SuperMimic WorldModel → reward → 反传
部署链：VLA token → SONIC decoder(ONNX) → 29 关节
```

这两个执行器不是同一个东西。修法（与 Isaac 版本无关，可独立做）：把闭环执行器换成
SuperMimic Policy —— `G1Assets.q_target(action) = default_joint_pos + action_scale * action`
和 `kp/kd/action_scale` 都已备好，930 维观测（10 帧 × 93）也现成。

## 4. 目标容器必须自查的 8 项（别照抄我的路径）

```bash
# 1) Isaac 两栈自检（EULA 变量必须先导出）
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
python -c "import torch, isaacsim, isaaclab; print(torch.__version__, isaaclab.__version__)"

# 2) GPU 是否 RTX（连 RT core 都没有就别折腾 Isaac）
nvidia-smi --query-gpu=name,memory.total --format=csv

# 3) robot 资产（116 个文件，含 mjcf/g1_29dof_rev_1_0.xml、usd/g1_29dof.usd）
python3 scripts/rls/supermimic/prepare_resources.py --task g1 --verify_only --assets_only

# 4) 我们的 token 数据（318 clips / 473,876 帧 / token_dim=64 / src_fps=50）
python3 scripts/rls/supermimic/prepare_token_data.py --src <npz_dir> --out data/token_motions/g1_tasks

# 5) 紧凑库与预计算库逐字段一致（这条能吃下 90% 的"数据加载"类问题）
python3 scripts/rls/supermimic/check_compact_library.py --motion_dir data/token_motions/g1_tasks --real_kin

# 6) token 通路（不装 Isaac 也能跑）
python3 scripts/rls/supermimic/check_token_interface.py --motion_dir data/token_motions/g1_tasks

# 7) 分布式管线（2 进程 gloo）
python3 -m torch.distributed.run --nproc_per_node=2 scripts/rls/supermimic/check_distributed.py

# 8) Isaac 端到端冒烟（2 envs / 1 iter，峰值应 < 1 GiB）
bash scripts/rls/supermimic/smoke_isaac.sh
```

任一项失败时的最短定位：
- 1 失败 → 版本/venv 混了，或 EULA 未导出
- 3 失败 → 资产不是 git checkout 或文件不全（脚本已支持"普通拷贝"安装）
- 5 失败 → 帧数 off-by-one（§5.3）或 token sidecar 帧数不匹配
- 8 的 `No device could be created` → 卡不是 RTX / 驱动太老

---

## 5. 必须打在自己代码上的三个契约

### 5.1 token 接口（让策略吃 64 维 token）

数据侧：每个 `<clip>.csv` 旁放 `<clip>.tokens.npy (T,64)`，目录里放
`meta.json = {"src_fps": 50, "token_dim": 64, ...}`。

代码侧（trackerlab 里我们已改的 6 个文件，等价改动）：

| 位置 | 改动 |
|---|---|
| `MotionLib.__init__` / `from_dir` | 支持 `token_dir`，载入 `tokens.npy` → `self.tokens`；新增 `token(f)` / `token_dim` |
| `motion_buffer` / `managers` | 各加一个 `token()` 透传 |
| `runner`（策略条件处） | `GOAL_DIM = 64`；把取 goal 的 3 处改成取 token |
| `train.py` | `--interface {motion,token}`；**torchrun 下按 `LOCAL_RANK` 设 device** |
| `evaluate.py` / `export_policy.py` | 按保存的 `interface` 走 token 通路，`goal_dim=64` |

### 5.2 显存契约（BONES-SEED 必须用紧凑库）

| 项 | 单卡 | 4 卡分片 |
|---|---|---|
| 预计算 FK 的 `MotionLib`（**实测 1340 B/帧**，14 tracked body） | **59.2 GiB** ❌ | 14.8 GiB/卡（勉强） |
| **紧凑库**（`ref` fp16，142 B/帧） | **6.3 GiB** ✅ | **1.6 GiB/卡** ✅ |
| replay buffer（T=1000 @4096 envs） | — | ~4.3 GiB/卡 |
| 实测整卡占用（4096 envs） | — | **17–18 GiB/卡** **[实测]** |
| 实测整卡占用（2 envs） | — | **0.77 GiB** **[实测]** |

### 5.3 帧数 off-by-one（**必须修，否则 token 错位一帧**）

```python
# ❌ 旧：50 Hz 数据上有浮点误差，318 条里 27 条少一帧
n = int(np.floor((T_src - 1) / src_fps / CONTROL_DT)) + 1

# ✅ 新：源帧率是控制帧率的整数倍时，重采样是恒等映射
ratio = src_fps * CONTROL_DT
if abs(ratio - round(ratio)) < 1e-9 and round(ratio) >= 1:
    n = T_src
else:
    n = int(np.floor((T_src - 1) / src_fps / CONTROL_DT)) + 1
```
自检：`准备数据后，每条 clip 的 tokens.npy 行数必须等于 CSV 行数`（我们的脚本会断言）。

---

## 6. 已验证 / 未验证的边界（别把没验的当结论）

| 项 | 状态 | 证据 |
|---|---|---|
| 紧凑库 ≡ 预计算库（root/body/goal/token/ref_state，真实 G1 kinematics） | **已验证** | 逐字段 max\|Δ\| ≤ 1.5e-3（fp16 舍入内） |
| token 通路（数据 → MotionLib → `Policy(obs,64)` → 29 维动作） | **已验证** | CPU 自检 |
| 分布式管线（广播/梯度平均/含 `grad=None` 边界/分片覆盖） | **已验证** | 2 进程 gloo |
| Isaac 端到端（2 envs/1 iter，含紧凑库 + token） | **已验证** | `peak 0.77 GiB`，退出码 0，写出 `last.pt` |
| 4 卡 B+C 的真实加速比 | **未验证** | 需要先跑 `MODE=prof` |
| 真实 s/iteration（4090） | **外推值** | 0.75 s @4096 envs / 1.07 s @8096（上游 23.71 min/2000 it @8096 envs on RTX PRO 6000 × 1.5），**±40%** |
| trackerlab 在 Isaac Lab 2.2 上能否跑（= S1 方案可行性） | **未验证** | 我这边刚开始测就被叫停，没有结论 |

---

## 7. 三个消融的实验口径（跑错就白跑）

| # | 问题 | 处理组 vs 对照组 | 指标 | 硬约束 |
|---|---|---|---|---|
| 1 | 我们的反传微调有没有用 | `vla_004000.pt` 合并后的 ckpt vs 官方 20k VLA | **闭环成功率 + Wilson 95% CI** | episode 是二值的：20 单臂粒度 5%，**两臂差 ≥2 个 episode 才算信号**；要写论文请 40+ 单臂 |
| 2 | FM 反传方法怎么选 | `pathwise / checkpointed / truncated / adjoint`（`qvgm_lite` 真实栈不可用，会显式报错） | `rollout_reward` 曲线 + 显存 + 单步耗时 | **同数据、同 seed、同墙钟**；adjoint 会 detach backbone（不要配 `--train-vlm`） |
| 3 | 固定墙钟下 BPTT 与 PPO 谁学得出来 | `train_joint.py`（精确反传） vs `train_ppo.py`（同一 reward 的 score-function） | 同墙钟 reward 曲线 + 能否闭环 | 两臂各 **2 h**；PPO 有效 horizon = 1（一次只吐一个 token chunk），是 T=1 的 clipped PG + 学习基线，**不要写成多步 GAE** |

**消融 1 的任务集**：用户明确要求**在 HA 7 任务上**做（视觉必须一致）；SONIC 4 任务的 MuJoCo 闭环只能作为"内部可比"的代理，**不能当 HA 口径交差**。

---

## 8. 卡数与时间：什么时候 4 卡有用

| 目标 | 4 卡的作用 | 墙钟 |
|---|---|---|
| 单次训练（固定迭代数） | **不缩短**；每卡一个仿真，4 卡 = 每迭代吃 4× 数据 | 由 s/iteration 决定：20k 迭代 ≈ **4.2 h** @4096 envs/rank、**5.9 h** @8096（外推，±40%） |
| 单次训练（固定数据量/质量） | **4× 减少所需迭代** | 追平现有 `ckpt_0024000.pt` 的训练量 ≈ 5–6k 迭代 ≈ **1–1.5 h** |
| 要统计量 | 4 个 seed 并行 | 同上墙钟，拿 4 条曲线 |

**加速的两个真旋钮**（按收益排序）：
1. **num_envs**（4096 → 8096/12288）：唯一能改 s/iteration 的杠杆；
2. **B+C**（分片仿真 + 梯度平均）：把"数据侧"分摊到 4 卡，实测前先跑 `MODE=prof`。

`MODE=prof` 会打印 `[prof] collect … | wm … | policy+critic …`：
- `collect` 占大头 → C 的收益最大；
- `wm`+`policy` 占大头 → B 的收益最大。

---

## 9. 跨容器通用坑

1. **`nohup … &` 起的进程可能被外层工具在命令结束时回收** → 长任务用受管后台任务或 `setsid`，否则会看到"跑到一半无声消失"（我这边就是这么丢了两次跑）。
2. **一张卡别塞两个 Isaac 训练**：4096 envs 就要 17–18 GiB，和别的训练共存前先看 `nvidia-smi`。
3. **`isaaclab.sh --install` 可能静默装到系统 python**（它用 PATH 里的 python）：装完必须
   `uv pip list | grep isaaclab` 复核，缺就用 `uv pip install -e source/isaaclab*` 补。
4. **trackerlab 的 python 依赖要单独装**：`mujoco==3.3.7 scipy matplotlib pillow imageio-ffmpeg opencv-python-headless`。
5. **`WXYZ` vs `XYZW`**：我们的 npz 是 WXYZ，trackerlab 的 CSV 要 XYZW，转换器已处理，别手写。
6. **上游 `prepare_resources.py` 默认会下 LAFAN1**（我们不用）：加 `--assets_only`，否则会白等/报错。

---

## 10. 我这条线产出的文件（可直接移植）

| 文件 | 作用 | 可移植性 |
|---|---|---|
| `scripts/export_vla_ckpt.py` | `vla_*.pt` → GR00T checkpoint 目录（含逐张量校验） | 与版本无关，直接用 |
| `diff_vla_wbc/fm_grad.py` | 4 种 FM 反传后端（pathwise/checkpointed/truncated/adjoint） | 与版本无关 |
| `diff_vla_wbc/ppo.py` + `scripts/train_ppo.py` | PPO 那一臂（固定墙钟对比用） | 与版本无关 |
| `scripts/success_rate.py` | 多 seed 闭环成功率 + Wilson CI + 断点续跑 | 需按你容器的 runner 路径改 `RUNNER` |
| `scripts/run_ablations.sh` / `ablation_report.py` | 三个消融的统一入口与汇总 | 同上 |
| `trackerlab-supermimic/.../motion_lib/compact.py` | 紧凑运动库（59.2 GiB → 6.3 GiB） | **与 version 无关，优先移植** |
| `trackerlab-supermimic/.../algorithms/supermimic/distributed.py` | B+C：分片 + 梯度平均 + 广播 | 同上 |
| `trackerlab-supermimic/scripts/rls/supermimic/{prepare_token_data,check_token_interface,check_compact_library,check_distributed}.py`、`smoke_isaac.sh`、`train_4gpu.sh` | 数据转换 / 四类自检 / 冒烟 / 4 卡启动 | 路径改用你的 `$REPO` |
| `docs/ABLATIONS.md`、`docs/ISAAC_TRACKERLAB_ADAPT.md` | 消融协议、Isaac 适配（含 §8 B+C 与显存账） | 文档直接读 |

---

## 11. 如果你只记三句话

1. **视觉一致性 = 评测渲染器 = Isaac Sim 5.0/2.2；WBC 用什么版本训不影响它。**
2. **BONES-SEED 必须走紧凑库**（59.2 GiB → 6.3 GiB），并且**帧数 off-by-one 必须修**（27/318 会错位）。
3. **4 卡不缩短固定迭代数的墙钟**，它买的是"每迭代 4× 数据"；要加速只能动 `num_envs`（和 B+C 的数据侧分摊）。
