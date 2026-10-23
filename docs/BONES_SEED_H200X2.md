# SuperMimic BONES-SEED 全量训练手册（2×H200 / 本机 1×4090 双版本）

> **目标**：在 **2 卡 H200** 上跑 **BONES-SEED 全量**（142,220 条原始 clip → 过滤后 129,943 clip / 263.75 h）的 SuperMimic 论文口径训练。
> **⚠️ 先读 §0**：本仓库所在容器（2026-10-23 实测）**只有 1×RTX 4090 48 GB，没有 2×H200；`/inspire/dataset/bones-seed/v1` 里也一条 G1 CSV 都没有**。所以本文 §4.1 的 H200 命令是"目标机器"规格书，**在本机跑不了**；本机能跑的是 §4.2 的单卡降配版，且必须先按 §1/§2 补数据、装环境（约 23.5 GB 数据 + 8 GB 依赖）。
> 环境：**不用 uv**，系统 `python3.12`（`/usr/bin/python3.12`，3.12.13）建 venv + `pip` + `torchrun`。
> 代码：`third_party/SuperMimic-Reproduced`（CuboiLeo 复现，mjlab `8bb2bc9` + MuJoCo Warp）。

---

## 0. 本机现状核查（2026-10-23，复核命令见 §8）

| 项 | 本机实际（已核实） | 本手册要求 | 判定 |
|---|---|---|---|
| 硬件 | k8s 容器 `flowbody128--…`，**1×RTX 4090 48 GB**（49140 MiB，sm_89，UUID `GPU-6533f836…`，`NVIDIA_VISIBLE_DEVICES` 只给这一张），128 核 / 1007 GB 内存，无 slurm | 2×H200（141 GB/卡） | ❌ 既不是 H200，也只有 1 张卡 |
| 驱动 / CUDA | `570.124.06` / CUDA 12.8 | torch 2.14 的 **cu13x** 轮子要驱动 ≥ 580 | ⚠️ 装 torch 必须选 **cu126**（`2.14.1` 有 cu126 轮子；cu128 源最高只到 2.11.0） |
| 数据 | `/inspire/dataset/bones-seed/v1` 共 **261 KB**：`LICENSE.md` + `README.md` + **空** `metadata/` + `.cache` 里 0 字节的 `g1.tar.gz.lock`；`find -name '*.csv' \| wc -l` → **0** | 142,220 条 G1 CSV | ❌ 数据没下（`g1.tar.gz` 实测 23,499,973,647 B ≈ 23.5 GB，且数据集 **gated=auto**） |
| Python 环境 | 只有 conda `isaaclab`：`torch 2.7.0+cu128` / `mujoco 3.10.0` / `warp-lang 1.10.1` / `numpy 1.26` / `pandas 2.3.3` / `pyarrow 24` / `tensorboard 2.20`；**无 mjlab、无 mujoco-warp、无 rsl-rl-lib、无 tyro**；仓库内**无 `.venv`**、无 `data/`、无 `logs/` | `torch>=2.14`、mjlab `8bb2bc9`、mujoco-warp 3.16、warp 1.18 | ❌ 现有环境不能训练 |
| 磁盘 | overlay 1.4 TB 可用 | 数据 + npz + 库 ≈ 100–120 GB | ✅ 够 |
| 网络 | pypi / github / huggingface 均可达；`git 2.34.1`、`hf` CLI 在；**没有 HF token** | — | ✅（数据要登录 + 同意 gated 协议）|

### 0.1 直接回答："能不能在当前环境里训练？"

**不能，而且没有"零下载"的路径。** 三条独立原因，缺一不可：

1. **硬件不对**：本机只有 1×4090 48 GB。2 卡 H200 的命令（`torchrun --nproc_per_node 2`）在这里要么直接找不到第二张卡，要么两个 rank 挤一张卡。单卡能跑，但要把命令换成 §4.2。
2. **数据是空的**：`bones-seed/v1` 只有壳。`--data.library data/library/bones_seed_compact.pt` 不存在，训练第一句就 `FileNotFoundError`。
3. **环境没装**：mjlab 1.6.0 的 `pyproject.toml` 硬性要求 `torch>=2.14.0` / `mujoco~=3.16.0` / `mujoco-warp~=3.16.0` / `warp-lang>=1.15.0`；现有 `isaaclab` 环境是 torch 2.7 / mujoco 3.10 / warp 1.10，**装不进去也跑不起来**，`import mjlab` 直接 ImportError。必须新建 venv（约 7–8 GB 下载）。

### 0.2 要跑起来，最少补三样

| 补什么 | 大小 | 命令 |
|---|---|---|
| BONES-SEED `g1.tar.gz` + 解压 | 23.5 GB + 25–40 GB | §1.2 |
| venv + mjlab 全套（本机用 cu126 torch） | 约 7–8 GB | §2.3 |
| 数据管线三步（142k CSV → npz → 紧凑库） | 45–50 GB + 6.74 GB | §3 |

### 0.3 本机能跑什么

- ✅ **可以**：单卡 1×4090 48 GB 跑论文口径同款配置（§4.2，`batch-size` 降到 2048），显存够；按 5090 上 4.3 it/s 的 60–75% 外推，**约 2.5–3.5 it/s → 320k 迭代 25–36 h**（未实测）。
- ❌ **不可以**：2 卡数据并行、H200 口径的 11–15 h 时长。
- 🚫 **不要指望复用 `isaaclab` conda 环境**：torch/mujoco/warp 三件套版本都对不上 mjlab 1.6，硬装会把该环境搞坏（它还被 `scripts/smoke_vlm.py` 等 VLA 脚本用着）。

---

## 1. 数据：`/inspire/dataset/bones-seed/v1`（实测只有壳）

### 1.1 实测状态

```bash
BONES_SEED=/inspire/dataset/bones-seed/v1

du -sh $BONES_SEED                              # 261K
ls -la $BONES_SEED                              # LICENSE.md README.md metadata/ .cache/
ls -A  $BONES_SEED/metadata                     # 空
find $BONES_SEED -name '*.csv' | wc -l          # 0
ls -la $BONES_SEED/.cache/huggingface/download  # 只有 *.lock（0 字节），没有 g1.tar.gz
```

结论：**目录和文档在，G1 动作数据不在**。`.cache/.../g1.tar.gz.lock` 是上次 `hf download` 启动后中断留下的 0 字节锁文件，不是数据。`metadata/` 也是空目录。

### 1.2 补齐数据（HF gated，23.5 GB）

> **现状（2026-10-24）**：数据已经下到 **`/inspire/hdd/.../wsc-workspace/datasets/bones-seed/v1/g1.tar.gz`**（字节数已核对 = 23,499,973,647），解压到 8,776/142,220 条时暂停。`/inspire/dataset/bones-seed/v1` 那个目录**仍是空壳，别再用**。续解压与后续步骤见 [`BONES_SEED_HANDOVER.md`](BONES_SEED_HANDOVER.md) §4.2。

`bones-studio/seed` 是 **gated=auto** 数据集：必须先同意协议 + 用 token 登录，否则 `hf download` 返回 401；而且 token 必须带 `canReadGatedRepos=true`（fine-grained token 默认 false，点了协议也会 403）。

```bash
# ① 浏览器打开 https://huggingface.co/datasets/bones-studio/seed ，点 "Agree and access repository"
# ② 本机登录（token 需 canReadGatedRepos；本机 HF_HOME 里原有的 token 没有这个权限，所以 403）
hf auth login                    # 旧版 CLI 用 huggingface-cli login
#   或：export HF_TOKEN=hf_xxx

# ③ 只下 g1（别下 soma_uniform/soma_proportional.tar.gz，各约 45 GB，本管线用不到）
BONES_SEED=/inspire/dataset/bones-seed/v1
hf download bones-studio/seed g1.tar.gz \
  --repo-type dataset --local-dir $BONES_SEED

# ④ 解压到数据根，先确认顶层目录名，再解压
cd $BONES_SEED
tar -tzf g1.tar.gz | head -3     # 期望形如 g1/csv/{date}/{motion}.csv
tar -xzf g1.tar.gz

# ⑤ 校验：必须是 142,220
find $BONES_SEED/g1 -name '*.csv' | wc -l
du -sh $BONES_SEED               # 解压后估计 25–40 GB
```

- 断点续传：`hf download` 中断后重跑同一条命令即可；旧锁文件不碍事。
- **别用错数据**：`/inspire/.../wsc-workspace/oli/.smpl_download/bones_seed_smpl/` 下那 31 GB 分卷（顶层是 `smpl_filtered/`）是 SOMA/SMPL 格式，**不是** G1 关节 CSV，喂不了本管线，别花时间解它。
- 磁盘预算：`g1.tar.gz` 23.5 GB + 解压 25–40 GB + npz 45–50 GB + 紧凑库 6.74 GB ≈ **100–120 GB**（本机 1.4 TB 可用，够）。解压完可删 `g1.tar.gz` 省 23.5 GB。

### 1.3 布局与坑（原口径不变）

- G1 动作 CSV 在 **`$BONES_SEED/g1/`** 下（`g1/*.csv` 或 `g1/csv/*.csv` 两种布局都行）。
- `metadata/` 是 BONES 的标注表（`move_duration_frames` 等），里面也有 csv，**不能**混进转换。
  因此 §3 的转换用 `--pattern '**/g1/**/*.csv'`：只吃 `g1/` 下的 csv，两种嵌套深度都能匹配，自动跳过 `metadata/`。
- 产物统一落在本仓库 `data/` 下（已在 `.gitignore`）：`data/npz/bones_seed/`、`data/library/bones_seed_compact.pt`。

---

## 2. 环境（venv + pip，不用 uv）

### 2.1 为什么必须新建 venv

| 包 | 现有 `isaaclab` 环境 | mjlab 1.6.0 要求 | 结论 |
|---|---|---|---|
| torch | 2.7.0+cu128 | `>=2.14.0` | ❌ |
| mujoco | 3.10.0 | `~=3.16.0` | ❌ |
| mujoco-warp | 未装 | `~=3.16.0` | ❌ |
| warp-lang | 1.10.1 | `>=1.15.0` | ❌ |
| mjlab | 未装 | `8bb2bc9` | ❌ |

（要求取自 `https://raw.githubusercontent.com/mujocolab/mjlab/8bb2bc9…/pyproject.toml`。）

### 2.2 本机驱动 → torch 必须锁 cu126（和原手册唯一的实质差别）

- 本机驱动 `570.124.06`（CUDA 12.8）。**cu130 的 torch 轮子要驱动 ≥ 580**，装上后 `import torch` 即报 `CUDA driver version is insufficient`。
- `download.pytorch.org/whl/cu128` 最高只有 torch 2.11.0，满足不了 `>=2.14`；**`cu126` 有 `torch-2.14.1+cu126-cp312`** 轮子（已核实存在）。
- 换到驱动 ≥ 580 的 H200 机器时，才按原手册用 cu130。

### 2.3 安装命令（本机可直接照抄）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/third_party/SuperMimic-Reproduced

# ① 本机没有 /etc/pip/constraint.txt，这一步是保险；若在别的镜像里装了，必须先解除，
#    否则 torch 会被钉在 2.8.0a0+nv，和 mjlab 的 torch>=2.14 直接冲突
unset PIP_CONSTRAINT

# ② 系统自带 python3.12（/usr/bin/python3.12，3.12.13），建仓库内 venv（.venv/ 已在 .gitignore）
python3.12 -m venv .venv
. .venv/bin/activate
pip install -U pip

# ③ 关键：先把 torch 装成 cu126 版（本机驱动 570 的上限），避免后面被解析成 cu130/cu12x 默认包
pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126

# ④ 装 mjlab（钉在 SuperMimic 复现用的 commit），torch 已满足 >=2.14，不会被换掉
pip install "mjlab @ git+https://github.com/mujocolab/mjlab@8bb2bc9e187797689ec82c78c35019afd2fd0376"

# ④b 关键：warp 必须降回 1.17（本机实测）。mjlab 会拉 warp-lang 1.18，
#     而 1.18 的 runtime 是 CUDA 13.4，要求驱动 >= 13.0；本机 570（12.8）下
#     wp.init() 只列出 cpu，**不报错**但 mujoco-warp 静默退化到 CPU。
#     1.17 用 CUDA 12.9 runtime，驱动 12.8 可用；mjlab 要求 >=1.15，1.17 满足。
pip install "warp-lang==1.17.0"

# ⑤ 装 supermimic 本体：--no-deps 跳过它 pyproject 里的钉版（会被 ③④ 覆盖，重复装反而打架）
pip install -e . --no-deps

# ⑥ mjlab 不带、但本仓库脚本要用的包
pip install pandas pyarrow huggingface-hub

# ⑦ 自检（三层：torch 能建 CUDA context + warp 枚举到 4090 + G1 模型能加载）
python -c "import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
python -c "import warp as wp; wp.init(); print(wp.get_devices())"
python -c "from supermimic.data.motion_convert import load_g1_model; m=load_g1_model(); print('nq', m.nq, 'njnt', m.njnt)"
python -c "import mjlab, supermimic; print('mjlab+supermimic ok')"
```

期望输出（本机实测）：`2.14.1+cu126 1 NVIDIA GeForce RTX 4090`、warp 列出 `['cpu', 'cuda:0']` 且标 `47 GiB, sm_89`、`nq 36 njnt 30`、最后一行打印 ok。
**warp 那行只有 `['cpu']` 就是装成了 1.18，必须降回 1.17**（见 ④b）。
首次安装约 7–8 GB（torch + CUDA 轮子），慢是正常的。

> 若在驱动 ≥ 580 的 2×H200 机器上，把 ③ 换成
> `pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130` 即可（与原手册一致）。

### 2.4 每个新 shell 都要先激活

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/third_party/SuperMimic-Reproduced
. .venv/bin/activate
unset PYTHONPATH        # 本机当前未设；若节点全局设了 ROS 之类 python3.10 的 PYTHONPATH，必须先清掉
```

确认 `which python` / `which torchrun` 指向 `.venv/bin/` 下的那个（不是 `/opt/miniconda3/envs/isaaclab/bin`）。

---

## 3. 数据管线（全量，一次性；在 venv 内跑）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/third_party/SuperMimic-Reproduced
. .venv/bin/activate
BONES_SEED=/inspire/dataset/bones-seed/v1

# (1) 142,220 条 120 fps G1 CSV（厘米 / 欧拉 xyz 度 / 关节度）-> 50 Hz npz，约 287.6 h
#     --pattern 只匹配 g1/ 下的 csv，避开 metadata/
python scripts/data/csv_to_npz_batch.py \
  --input-dir $BONES_SEED \
  --pattern '**/g1/**/*.csv' \
  --output-dir data/npz/bones_seed \
  --input-fps 120 --output-fps 50 \
  --workers 64

# (2) 名字过滤 + 可行性过滤 -> manifest_filtered.csv
#     门槛：名字后 130,137（论文 129,667）；可行性后 129,943（论文 129,514）、263.75 h（论文 264.56）
python scripts/data/filter_bones_seed.py \
  --npz-dir data/npz/bones_seed --workers 64 \
  --out data/npz/bones_seed/manifest_filtered.csv

# (3) 打包紧凑 fp16 动作库：目标特征 + 14 个跟踪刚体状态在线 FK
#     预期输出：47.6M 帧 / 263.75 h / 6.74 GB
python scripts/data/build_library.py \
  --npz-dir data/npz/bones_seed \
  --manifest data/npz/bones_seed/manifest_filtered.csv \
  --out data/library/bones_seed_compact.pt \
  --workers 32
```

**磁盘预算**：npz ≈ 45–50 GB / 紧凑库 6.74 GB（原始 CSV 已在 `/inspire`，不再占本仓库）。
训练只读那一个 `.pt`，本机 `/inspire` 与仓库同盘（overlay），不需要再搬。

---

## 4. 训练

### 4.1 目标机器：多卡 H200，一进程一卡（token 条件化，离线）

> **当前正式路线是 token 条件化**（VLA 直出 64 维 SONIC motion token，SuperMimic 直接消费；
> 不再用 680 goal 做 policy 条件）。4×H200 离线启动脚本已就绪：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

# ① 离线预检（不联网；检查 4×H200/sm_90、warp、encoder、紧凑库）
cd third_party/SuperMimic-Reproduced
.venv/bin/python ../../scripts/preflight_4xh200_offline.py --expect-gpus 4
cd ../..

# ② 三档 200 迭代基准 → 决定用哪档（看 it/s 与显存峰值）
PROFILE=paper ITER=200 RUN=bench_paper bash scripts/launch_token_train_4xh200.sh
PROFILE=high  ITER=200 RUN=bench_high  bash scripts/launch_token_train_4xh200.sh
PROFILE=max   ITER=200 RUN=bench_max   bash scripts/launch_token_train_4xh200.sh

# ③ 正式 320k（论文口径优先）
PROFILE=paper bash scripts/launch_token_train_4xh200.sh

# ④ 续训
PROFILE=paper bash scripts/launch_token_train_4xh200.sh \
  --resume logs/e9_bones_seed_token_h200x4/ckpt_latest.pt
```

利用率档位（脚本默认 `high`）：

| 档位 | 每卡 envs / policy batch / WM batch | 估算显存/卡 | 320k 预计（估） |
|---|---|---|---|
| `paper` | 4096 / 4096 / 2048 | ~25 GB（18%） | ~13–20 h |
| `high` | 8192 / 8192 / 4096 | ~55 GB（40%） | ~20–32 h |
| `max` | 16384 / 8192 / 8192 | ~90–110 GB（70%） | ~28–45 h |

> `max` 只是更"满"，**不是更快**（每次迭代更新次数固定，加 batch 只让迭代更重）。
> 正式结果建议 `paper`；要利用率用 `high`；`max` 只做吞吐基准。
> TF32 默认开启；**不要用 `--policy-trainer.compile reduce-overhead`**（会卡 cudagraphs），compile 必须开。

下面第 4.1 节保留的手写命令是旧 680-goal 口径，仅作参考；token 版以脚本为准。

**前置**：驱动 ≥ 580（cu130 torch）、2 张 H200、§3 已跑完。

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/third_party/SuperMimic-Reproduced
. .venv/bin/activate
nvidia-smi -L                    # 必须列出 2 张 H200，否则别用这条命令

OMP_NUM_THREADS=16 torchrun --nproc_per_node 2 --master_port 29500 \
  -m supermimic.scripts.train \
  --run-name e8_bones_seed_h200x2 \
  --seed 0 \
  --data.library data/library/bones_seed_compact.pt \
  --env.num-envs 4096 \
  --env.future-frames 10 \
  --policy-trainer.horizon 32 \
  --policy-trainer.batch-size 4096 \
  --schedule.total-iterations 320000 \
  --schedule.updates-per-env-step 0.4 \
  --schedule.checkpoint-interval 4000
```

### 4.2 本机可跑版：1×RTX 4090 48 GB 单卡（**在这台机器上就用这条**）

单卡不需要 torchrun（`world=1` 时 lr 不放大，走的就是论文的单卡口径）；批降到 2048（5090 32 GB 的实测口径），48 GB 显存更宽松。

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/third_party/SuperMimic-Reproduced
. .venv/bin/activate

# 先冒烟：300 迭代、小环境数，确认全链路通（约 5–10 分钟）
OMP_NUM_THREADS=32 python -m supermimic.scripts.train \
  --run-name smoke_bones_seed_4090 \
  --data.library data/library/bones_seed_compact.pt \
  --env.num-envs 512 \
  --policy-trainer.batch-size 512 \
  --schedule.total-iterations 300 \
  --schedule.checkpoint-interval 100

# 正式跑：320k 迭代
OMP_NUM_THREADS=32 python -m supermimic.scripts.train \
  --run-name e8_bones_seed_4090x1 \
  --seed 0 \
  --data.library data/library/bones_seed_compact.pt \
  --env.num-envs 4096 \
  --env.future-frames 10 \
  --policy-trainer.horizon 32 \
  --policy-trainer.batch-size 2048 \
  --schedule.total-iterations 320000 \
  --schedule.updates-per-env-step 0.4 \
  --schedule.checkpoint-interval 4000
```

- 想只跑一段看曲线：加 `--schedule.max-hours 8`（到点自动停，`ckpt_latest.pt` 可在 §5 续训）。
- 4090 的 48 GB 是"魔改显存"，算力仍是 sm_89（≈5090 的 55–70%）：预计 **2.5–3.5 it/s，320k 迭代 25–36 h**（未实测，跑起来看 TensorBoard 的 `iter/s` 校准）。
- 单卡不是论文的多卡口径，结果不要和 Table A.8 的 2 卡数字逐位对比。

### 4.3 参数为什么这么配（均已对代码核实）

| 项 | H200×2 | 本机 4090×1 | 依据 |
|---|---|---|---|
| 启动方式 | `torchrun --nproc_per_node 2` | `python -m`（world=1） | `src/supermimic/training/distributed.py:26-27`：非 torchrun 直接 `(0,0,1)` |
| rank 绑卡 | 自动 `CUDA_VISIBLE_DEVICES=LOCAL_RANK` | 不需要 | `src/supermimic/scripts/train.py:14-31`，**不要**手动设 `CUDA_VISIBLE_DEVICES`，否则 MuJoCo Warp 把全部 rank 挤到 GPU 0 → OOM |
| lr 缩放 | 自动 ×√2 | 自动 ×1（不放大） | `training/loop.py:37-40`；单卡就是论文基学习率 |
| `--data.library` | `bones_seed_compact.pt` | 同 | `training/loop.py:49-51` 检测到 library 会**自动**置 `env.compact_library=True`；不要再传 `--data.npz-dir` / `--data.manifest` |
| 每卡环境数 | 4096 | 4096 | 论文 Table A.5；48 GB 够 |
| 策略批 / 视野 | 4096 / H=32 | **2048** / H=32 | 论文 Table A.5；5090 32 GB 的实测口径，4090 沿用更稳 |
| UTD | 0.4 | 0.4 | `--schedule.updates-per-env-step` |
| 迭代 | 320k（无 `max-hours` 上限） | 320k（可加 `--schedule.max-hours`） | 全量跑满，默认 `max_hours=None` |
| 观测口径 | 论文（`anchor_obs=False`，默认） | 同 | EXPERIMENTS.md「正式结果统一论文观测口径」 |
| 检查点 | 每 4000 迭代 | 同 | `ckpt_latest.pt` + `ckpt_XXXXXXX.pt`（仅 rank0 写） |

**预期**：2×H200 单卡约 6–8 it/s → 320k 迭代约 **11–15 h**；显存 ≈ 6.7 GB（库）+ 20 GB 左右，141 GB 远够。
日志：`logs/<run-name>/tb`（TensorBoard）、配置快照 `logs/<run-name>/config.json`。

想更快可以把 `--env.num-envs` / `--policy-trainer.batch-size` 加到 8192 —— **但那是偏离论文的改动**，正式结果别这么跑。

---

## 5. 续训 / 评测 / 可视化

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/third_party/SuperMimic-Reproduced
. .venv/bin/activate

# 续训（单卡；从 ckpt_latest.pt 继续，迭代计数接着走）
OMP_NUM_THREADS=32 python -m supermimic.scripts.train \
  --run-name e8_bones_seed_4090x1 \
  --data.library data/library/bones_seed_compact.pt \
  --policy-trainer.batch-size 2048 \
  --schedule.total-iterations 320000 \
  --resume logs/e8_bones_seed_4090x1/ckpt_latest.pt

# 2 卡续训：把 python -m 换成 OMP_NUM_THREADS=16 torchrun --nproc_per_node 2 --master_port 29500 \
#   -m supermimic.scripts.train，--resume 指向 logs/e8_bones_seed_h200x2/ckpt_latest.pt

# 全量 benchmark（论文 Appendix F；评测走 npz 目录，不是 compact 库）
python scripts/eval_benchmark.py \
  --ckpt logs/e8_bones_seed_4090x1/ckpt_latest.pt \
  --npz-dir data/npz/bones_seed \
  --manifest data/npz/bones_seed/manifest_filtered.csv \
  --num-envs 1024

# 2000 条固定子集快速对比（先建列表再评测）
python scripts/eval_benchmark.py \
  --ckpt logs/e8_bones_seed_4090x1/ckpt_latest.pt \
  --npz-dir data/npz/bones_seed \
  --clip-list data/npz/bones_seed/subset2000.txt \
  --num-envs 1024

# 交互式 replay（无显示时 --viewer viser -> http://localhost:8080）
python scripts/play.py \
  --ckpt logs/e8_bones_seed_4090x1/ckpt_latest.pt --clips dance1_subject1
```

---

## 6. 排错对照

| 现象 | 原因 / 处理 |
|---|---|
| `import torch` 报 `CUDA driver version is insufficient` | 装了 cu130 版 torch，而本机驱动是 570（CUDA 12.8）→ 重装 cu126：`pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126` |
| `ModuleNotFoundError: mjlab` | 用了 `isaaclab` conda 环境；必须 `. .venv/bin/activate`，且在 venv 里装过 mjlab |
| `pip install` 报 `ResolutionImpossible ... torch>=2.14.0` | ① 装了旧 torch（2.7）；② 镜像的 `/etc/pip/constraint.txt` 钉死 torch → `unset PIP_CONSTRAINT` 后先按 §2.3 ③ 装 cu126 torch |
| `FileNotFoundError: no clips under ...` | `--data.library` 路径错，或 §3 (3) 没跑完（本机当前就是这种情况：`.pt` 根本不存在） |
| `find $BONES_SEED -name '*.csv' \| wc -l` 为 0 | 数据没下：按 §1.2 走 gated 下载（23.5 GB） |
| `hf download` 返回 401 / `GatedRepoError` | 没同意数据集协议或没登录 → 浏览器点 Agree + `hf auth login` |
| `python: No module named supermimic` | 没激活 venv，或漏了 `pip install -e . --no-deps` |
| `torchrun: command not found` | 同上（torchrun 随 torch 装进 `.venv/bin`） |
| `pip: command not found` / 装到系统里去了 | `. .venv/bin/activate` 后再 `pip`，确认 `which pip` 在 `.venv/bin` 下 |
| 训练一启动就崩在 import | 节点全局 `PYTHONPATH` 里有其它 Python 版本的包 → `unset PYTHONPATH` |
| 多卡时全部 rank 抢 GPU 0、OOM | 手动设了 `CUDA_VISIBLE_DEVICES` → 去掉，让 `src/supermimic/scripts/train.py` 按 `LOCAL_RANK` 绑卡 |
| 转换时抛 `csv_format` / 列名错误 | `--pattern` 把 `metadata/*.csv` 吃进去了 → 保持 `--pattern '**/g1/**/*.csv'` |
| 训练早期仿真慢 10× | MuJoCo Warp solver 溢出 printf；本仓库已在 `put_model` 后关闭（`EXPERIMENTS.md`），若自改过 `SimulationCfg` 需重新确认 |
| 4090 上显存不足 | 降 `--policy-trainer.batch-size`（1024）或 `--env.num-envs`（2048）；48 GB 正常不会碰到 |
| 4090 比预期慢很多 | 单卡不是论文硬件；先看 `iter/s` 是否落在 2.5–3.5，再看是不是被别人占了卡（`nvidia-smi`） |
| 训练回报/评测偏低的误判 | BONES-SEED clip 中位长度仅 5.8 s，漂移影响很小；LAFAN1 长 clip（均值 220 s）不能用同一套预期 |

---

## 7. 参考资料

- 数据 / 过滤 / 紧凑库口径：[`third_party/SuperMimic-Reproduced/README.md`](../third_party/SuperMimic-Reproduced/README.md)、[`docs/EXPERIMENTS.md`](../third_party/SuperMimic-Reproduced/docs/EXPERIMENTS.md)
- BONES-SEED 数据集说明（142,220 条 / 288 h / 三种格式）：[`GR00T-WholeBodyControl/docs/source/user_guide/training_data.md`](../GR00T-WholeBodyControl/docs/source/user_guide/training_data.md)
- 论文数值与协议：[`docs/PAPER_SPEC.md`](../third_party/SuperMimic-Reproduced/docs/PAPER_SPEC.md)
- 本工作区接入框架：[`docs/FRAMEWORK.md`](FRAMEWORK.md)
- SuperMimic 论文：<https://arxiv.org/abs/2610.28480>
- BONES-SEED 数据集页（gated）：<https://huggingface.co/datasets/bones-studio/seed>

---

## 8. 核查清单（换机器/换环境时先跑这 8 条）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/third_party/SuperMimic-Reproduced

# ① 卡型与张数（2×H200 应列出两张 H200；本机只有一张 4090）
nvidia-smi --query-gpu=index,name,memory.total --format=csv
nvidia-smi -L

# ② 驱动版本（cu130 轮子要求 >= 580；本机 570 → 只能 cu126）
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# ③ 数据是否下全（期望 142220）
find /inspire/dataset/bones-seed/v1/g1 -name '*.csv' 2>/dev/null | wc -l

# ④ 紧凑库是否建好（期望存在且约 6.74 GB）
ls -lh data/library/bones_seed_compact.pt

# ⑤ venv 是否建好且装对版本
. .venv/bin/activate 2>/dev/null && python -c "import torch, mjlab, supermimic; print(torch.__version__, torch.cuda.device_count())"

# ⑥ warp 能否枚举设备
python -c "import warp as wp; wp.init(); print(wp.get_devices())"

# ⑦ 单测是否过（首次装完环境后跑一次）
env -u PYTHONPATH python -m pytest tests -q

# ⑧ 冒烟训练（300 迭代）
python -m supermimic.scripts.train --run-name smoke --data.library data/library/bones_seed_compact.pt \
  --env.num-envs 256 --policy-trainer.batch-size 256 --schedule.total-iterations 300
```
