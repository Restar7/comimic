# Isaac Lab 侧 BONES-SEED 训练：装机 + 数据 + 开训（2026-10-24/25）

> 目标：在**本机 2×RTX 4090（48 GB 版）**上，把 `trackerlab-supermimic`（Isaac Lab 2.3.2 /
> Isaac Sim 5.1 / PhysX）跑在 **BONES-SEED**（129,943 clips / 47,475,680 帧 / 263.75 h）上，
> 双卡 DDP + 分片动作库开训。
>
> 标记：**[已验证]** = 本机真跑过；**[未验证]** = 只有静态依据。

---

## 0. 一句话现状

| 项 | 状态 |
|---|---|
| Isaac Sim 5.1.0 + Isaac Lab 2.3.2（Python 3.11） | **独立 venv** `third_party/trackerlab_env`，见 §2 |
| BONES-SEED 紧凑库 → Isaac 侧可用的 packed 库 | ✅ `trackerlab-supermimic/data/bones_seed/bones_seed_compact.pt`（6.28 GiB，canonical 布局） |
| 数据正确性（关节序 / 四元数序 / MJCF / 50 Hz） | **[已验证]** 与 MuJoCo npz 的 14 个跟踪刚体逐帧对齐，最大位置误差 **6.2e-4 m**（fp16 舍入量级） |
| 4 env 冒烟 / 双卡开训 | 见 §5、§6（本轮实测结果记在那两节） |

**为什么不复用并发会话的 `third_party/isaacsim_env`**：那是 Isaac Sim **5.0.0 + Isaac Lab 2.2**
（HA 场景闭环用的），版本与本线不一致；而且它的 Python 依赖 `~/.local/share/uv/python/cpython-3.11`
已被清掉，`bin/python` 是断链，**该 venv 目前是死的**（21 GB，可删，但归另一条会话）。

---

## 1. 版本矩阵（来自 `trackerlab-supermimic/docs/installation.md`）

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.11.16（uv-managed） | Isaac Lab 2.3 只支持 3.11 |
| Isaac Sim | **5.1.0**（`isaacsim[all,extscache]==5.1.0`，pypi.nvidia.com） | pip 装法，非 workstation |
| Isaac Lab | **v2.3.2** = commit `37ddf626871758333d6ed89cf64ad702aef127d0`（pip 包版本号 0.54.2） | 只装 `source/isaaclab`（任务只依赖核心包） |
| PyTorch | 2.7.0+cu128 | Isaac Sim 5.1 配套 |
| 驱动 | 595.71.05（本机） | ≥ 580 即可 |
| MuJoCo | 3.3.7 | 只用于 FK/离线渲染，训练中的动力学是 PhysX |

---

## 2. 装机（一条命令，可重复执行）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
bash scripts/install_trackerlab_isaac.sh            # 全流程，日志 logs/isaaclab_232_install.log
STEP=isaacsim bash scripts/install_trackerlab_isaac.sh   # 单步重试
```

脚本做的事（幂等）：

1. `uv venv --python 3.11 third_party/trackerlab_env`
   （uv 的 3.11 装在 `third_party/uv_python`，不再依赖 `~/.local/share/uv/python`）；
2. `pip/setuptools==80.9.0/wheel==0.45.1` + `flatdict==4.0.1 --no-build-isolation`；
3. `torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0`（`download.pytorch.org/whl/cu128`）；
4. `isaacsim[all,extscache]==5.1.0`（`--extra-index-url https://pypi.nvidia.com`，
   `--index-strategy unsafe-best-match`）；
5. `git clone` IsaacLab + `checkout 37ddf62…` + `pip install -e source/isaaclab` + `ipython/psutil`；
6. `pip install -r requirements.txt` + `pip install --no-deps -e .`（trackerlab-supermimic）；
7. 自检 `import torch, isaacsim, isaaclab, trackerLab`。

非交互相关的环境变量（脚本里已设，手工跑要自己 export）：

```bash
export OMNI_KIT_ACCEPT_EULA=YES   # 否则 import isaacsim 会卡在 EULA 提问上（上次就是这么挂的）
export ACCEPT_EULA=Y
```

### 2.1 系统依赖（这台机器原本缺）

```bash
apt-get install -y libxt6 libglu1-mesa libegl1 libvulkan1 libvulkan-dev vulkan-tools \
    libxrender1 libxi6 libsm6 libice6 libgomp1 \
    libxrandr2 libxinerama1 libxcursor1 libxkbcommon0 libgbm1 libxcb1 libxext6 libxfixes3
```

**已验证的显卡栈**：`vulkaninfo --summary` 能枚举两块 4090（`apiVersion 1.4.329`），
ICD 是 `/etc/vulkan/icd.d/nvidia_icd.json` → `libGLX_nvidia.so.0`；`libEGL.so.1` 可加载；
`libcuda.so.1` 在 ldconfig 里。若被 mesa 软件设备抢走，显式指定：

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

### 2.2 root 容器

Kit 拒绝以 root 运行，除非 `--allow-root`。本仓库的 `train.py/evaluate.py` 通过
`--kit_args=--allow-root` 传（`supervise.py` 会自动判断 euid），我们的
`train_bones_seed.sh` 默认就带这个参数。

---

## 3. 数据：BONES-SEED 怎么进 Isaac 侧

### 3.1 两边的数据口径

| | MuJoCo 侧（已有） | Isaac 侧（本仓库） |
|---|---|---|
| 文件 | 一个 `bones_seed_compact.pt`：`ref (47,475,680, 71) fp16` + `names/starts/lengths` | 逐 clip 36 列 CSV（30 Hz 源）→ 载入时重采样到 50 Hz |
| 每帧内容 | `[root_pos(3), root_quat_wxyz(4), q(29), root_lin_vel_w(3), root_ang_vel_w(3), qd(29)]` | 只有 pos+quat+q；速度用中心差分现算 |
| 帧率 | 已重采样到 **50 Hz**（BONES-SEED 原始 120 fps） | 控制 50 Hz |
| 刚体状态 | 不存，在线 FK | 默认**预计算全部刚体**（623 float/帧）→ BONES-SEED 要 **110.2 GiB** ❌ |

所以 Isaac 侧的做法是：**直接吃 packed 张量**（不导 13 万条 CSV、不再重采样），
并复用上一轮已经写好的 `CompactMotionLib`（只存 ref，`body_*` 按需 FK）。

### 3.2 生成 Isaac 侧用的 canonical 库

新增 `scripts/rls/supermimic/prepare_bones_seed.py`：校验 + 重排 + 落盘 + 侧车 `meta.json`。

```bash
cd trackerlab-supermimic
python3 scripts/rls/supermimic/prepare_bones_seed.py \
  --src  ../third_party/SuperMimic-Reproduced/data/library/bones_seed_compact.pt \
  --out  data/bones_seed/bones_seed_compact.pt \
  --npz_dir ../third_party/SuperMimic-Reproduced/data/npz/bones_seed --fk_clips 6
```

输出：

```
data/bones_seed/bones_seed_compact.pt   6.28 GiB  ref (47475680, 71) fp16，
                                        canonical 布局 [p, quat_wxyz, v, w, q, qd]
data/bones_seed/meta.json               num_clips / num_frames / hours_50hz / src_fps /
                                        ref_layout=canonical / joint_names / tracked_bodies /
                                        fk_cross_check（逐 clip 误差）
```

**为什么要重排**：`supermimic` 的 `build_library.py` 存的是
`[p, quat, q, v, w, qd]`，而 `CompactMotionLib` 内部统一用 `[p, quat, v, w, q, qd]`
（这样 `_Gather` 的切片和 `MotionLib` 一致）。重排一次 2.6 s，之后每个 rank 直接 memcpy 到显存。

### 3.3 正确性验证（不需要 Isaac）

两个脚本，都是 CPU 就能跑：

```bash
# 1) 数据侧：packed 库 + FK 与 MuJoCo npz 逐帧对比
python3 scripts/rls/supermimic/prepare_bones_seed.py --verify_only \
  --src <raw.pt> --npz_dir <npz> --fk_clips 6

# 2) 代码侧：from_packed 的索引表 / goal / 刚体形状 / 显存报告
python3 scripts/rls/supermimic/check_bones_seed_loader.py \
  --library data/bones_seed/bones_seed_compact.pt --npz_dir <npz> --clips 8
```

**[已验证]**（本机，CPU）：

```
[lib] 8 clips / 5797 frames / token_dim=None
[index] clip/bin tables consistent (120 bins, 50 frames per bin)
[shapes] goal=(97, 730) (K=10), body_p=(97, 14, 3), obs-ready
[fk] Idle_Left_001__A017: 3827 frames, body_pos err 3.22e-04 m
[fk] ... worst 1.08e-03 m（fp16 参考张量的量级）
[ok] packed BONES-SEED loader verified (CPU only, no Isaac required)
```

这一步同时证明了四件事：**关节顺序**与 `G1_JOINT_NAMES` 一致、**四元数是 wxyz**、
**MJCF 与 npz 的来源模型一致**、**数据已经是 50 Hz**（否则帧数会对不上）。

> 顺带：`prepare_bones_seed.py` 的校验覆盖 1.3e5 条 clip 的索引连续性、NaN/inf、
> 四元数模长（最大偏差 **4.16e-4**，fp16 正常量级）。

---

## 4. 机器人资产

`RobotKinematics` 需要 **MJCF**（FK），PhysX 需要 **USD**：两者都在
`Renforce-Dynamics/assetslib@9df79278` 的 `unitree/unitree_g1` 下。

```bash
# 稀疏克隆（prepare_resources.py 的做法），只取 unitree_g1
cd trackerlab-supermimic
git init data/assetslib && git -C data/assetslib remote add origin https://github.com/Renforce-Dynamics/assetslib.git
git -C data/assetslib sparse-checkout init --cone
git -C data/assetslib sparse-checkout set unitree/unitree_g1
git -C data/assetslib fetch --depth 1 origin 9df792784e3642163f03390bf6ae16cf88b222a5
git -C data/assetslib checkout --detach 9df792784e3642163f03390bf6ae16cf88b222a5
```

或用等价的一条命令（本次用的）：

```bash
curl -sSL https://codeload.github.com/Renforce-Dynamics/assetslib/tar.gz/9df79278… \
 | tar xz -C /tmp --wildcards "*/unitree/unitree_g1/*"
cp -r /tmp/assetslib-*/unitree/unitree_g1 trackerlab-supermimic/data/assetslib/unitree/
```

校验：`urdf/g1_29dof_rev_1_0.urdf`(36 KB)、`mjcf/g1_29dof_rev_1_0.xml`(24 KB)、
`usd/g1_29dof_rev_1_0.usd`(1.6 KB 入口) + `usd/configuration/g1_29dof_rev_1_0_{base,physics}.usd`
(28 MB / 10 KB) 都在，且不是 Git-LFS 指针（首字节 `PXR-USDC`）。**[已验证]**

---

## 5. 冒烟（4 env）

```bash
cd trackerlab-supermimic
MODE=smoke bash scripts/rls/supermimic/train_bones_seed.sh
```

等价于：

```bash
third_party/trackerlab_env/bin/python scripts/rls/supermimic/train.py \
  --task g1 --motion_packed data/bones_seed/bones_seed_compact.pt \
  --num_envs 4 --iterations 3 --interface motion \
  --goal_frames 10 --hist 10 --horizon 32 --utd 0.4 --lr 2e-4 \
  --buffer_steps 64 --warmup_steps 16 --wm_only_iters 0 \
  --wm_batches 1 --wm_batch 8 --pi_batch 8 --log_every 1 --ckpt_every 5 \
  --amp bf16 --no-compile --kit_args=--allow-root \
  --log_dir logs/bones_seed_smoke
```

看什么：`[motion] compact: 129943 clips / 47475680 frames`、PhysX 起来、
loss 有限、`logs/bones_seed_smoke/args.json` 里 `motion_packed` 正确。

---

## 6. 双卡开训（当前正在跑的就是这一条）

```bash
cd trackerlab-supermimic
setsid --fork env MODE=train NPROC=2 ENVS_PER_RANK=8192 ITERATIONS=32000 COMPILE=1 \
  LOG_DIR=logs/bones_seed_2x8192_compile PORT_BASE=29810 \
  bash scripts/rls/supermimic/train_bones_seed.sh
```

> `setsid --fork` 不是装饰：`nohup ... &` 起的进程仍留在调用它的 shell 的进程组里，
> 上游被打断（Ctrl-C / 工具超时 / 会话收尾）时会一起收到 SIGTERM。本机已经踩过一次。

- 每个 rank **一张卡、一份 Isaac 仿真、一份动作库分片**（`--shard_motion --shard_mode interleave`），
  world model / policy / critic 的梯度 `allreduce` 后平均（先平均、后裁剪）；
- `logs/bones_seed_2x8192_compile/`: `args.json` / `train_log.csv` / `motion_library.json` /
  `ckpt_XXXXXX.pt` / `last.pt`（只有 rank 0 写盘）；
- 进度查看、指标解读、续训命令：见 **[BONES_SEED_PROGRESS.md](BONES_SEED_PROGRESS.md)**。

### 6.0 配置是怎么选出来的（实测，不是拍脑袋）

| 配置 | 稳态 it/s | 每迭代数据（env-steps） | PyTorch 峰值 | 结论 |
|---|---|---|---|---|
| 2 rank × 4096 | **0.57** | 8,192 | 21.3 GiB | 基线 |
| 2 rank × **8192** | **0.56** | **16,384** | **25.5 GiB** | **同样时间一倍数据 → 选它** |
| 单 rank × 4096（冒烟） | 1.02（wm-only 阶段） | 4,096 | 10.5 GiB | 只用来验证链路 |

迭代内部拆解（`SM_PROFILE=1` 实测，2×8192）：`imagine fwd 260–290 ms` +
`imagine bwd 213–229 ms` + `critic 115–160 ms` ≈ **0.63 s**；每进程 GPU 占用只有 **35–50%**，
所以「加环境数」几乎免费（仿真不是瓶颈），真正的杠杆是 `torch.compile` 压更新段。

### 6.1 显存预算（单卡 48 GB，BONES-SEED 全量，2 rank）

| 项 | 每 rank |
|---|---|
| packed 库分片（ref fp16 一半 + int64 索引） | **~4.1 GiB** |
| `root_v_local/root_w_local/root_gz`（fp32） | ~0.33 GiB |
| replay buffer（`--buffer_steps 1000`, 8192 envs） | ~8.8 GiB |
| world model + policy + critic + optimiser | ~0.5 GiB |
| PhysX / Isaac Lab（8192 envs） | ~12–14 GiB |
| **合计** | **实测 25.5 GiB**（8192 envs）；4096 envs 时 21.3 GiB |

不分片时库本体是 **8.24 GiB**（`check_bones_seed_loader.py` 实测报告）；
上游预计算 FK 布局要 **110.2 GiB** ❌。

---

## 7. 已知坑

1. **EULA**：新 shell 里手工跑必须先 `export OMNI_KIT_ACCEPT_EULA=YES`，否则 `import isaacsim`
   直接 `EOF when reading a line`。装 `isaaclab` 时报 `No module named pip` 也是同一个原因的连带
   （venv 里先装 pip）。
2. **root**：`--kit_args=--allow-root`，否则 Kit 启动即退出。
3. **Isaac Lab 只装核心包**：`pip install -e third_party/IsaacLab_232/source/isaaclab` 就够，
   不要跑 `./isaaclab.sh --install`（它会去找 conda / `_isaac_sim/python.sh`，本机没有）。
4. **不要用 `--motion_dir` 指到 BONES-SEED**：那条路会逐 clip `np.loadtxt`，13 万条 CSV 不现实；
   BONES-SEED 一律 `--motion_packed`。
5. **评测**：`evaluate.py` 是「一个 clip 一个环境」，13 万 clip 起不来；
   用 `--motion_packed <lib> --clip_limit 4096` 取均匀子集（新增参数）。
6. `torch.compile`：长跑建议开（`COMPILE=1`），首次编译几分钟；冒烟务必 `--no-compile`。
