# SuperMimic BONES-SEED 全量训练手册（4×H200 · 复用本机环境 · 全程离线）

> **目标机器**：4×H200（143 GB/卡），**没有外网**，**不能装任何东西**，只能用当前这套已经装好并跑通的环境。
> **一句话结论**：H200 那台机器**只干"训练"这一件事**。数据、紧凑动作库、全部 Python 依赖、代码路径都已经在本机（2×4090）备好并校验过，H200 上**不需要 pip / 不需要 huggingface / 不需要 apt**。
> 配套：[`BONES_SEED_HANDOVER.md`](BONES_SEED_HANDOVER.md)（本机 2×4090 现状与数据管线）、[`BONES_SEED_H200X2.md`](BONES_SEED_H200X2.md)（2×H200 目标规格）
> 文档时间：2026-10-24，基于本机实测（环境已做成自包含，`docs` 里唯一需要"信我"的部分都给了校验命令）。

---

## ⚠️ 先读这一段：主路线已定为 **2×H200**，入口在 `BONES_SEED_H200X2_OFFLINE.md`

**2026-10-24 决定：正式结果走 2×H200**（论文口径：lr ×√2、样本量 1:1）。
👉 **离线机器上的完整命令看 [`BONES_SEED_H200X2_OFFLINE.md`](BONES_SEED_H200X2_OFFLINE.md)** —— 正式跑就是一条：
`bash scripts/launch_token_train_2xh200.sh`。

本文角色降级为：**4 卡加速（可选）+ 离线环境细节**。架构以 [`FINAL_HANDOVER.md`](FINAL_HANDOVER.md) 为准 = **token 条件化**（`--policy.token-dim 64`）。

| 事项 | 去哪 |
|---|---|
| **2×H200 正式训练（主路线）** | [`BONES_SEED_H200X2_OFFLINE.md`](BONES_SEED_H200X2_OFFLINE.md) §0 |
| 4×H200 加速（样本量 2×，别对论文数字） | 本文 §0.1 / §4.2 |
| 离线环境保障 / 排错 / ETA 换算 | 本文 §1–§3、§7、§8 |

> 本文与 `BONES_SEED_HANDOVER.md` §6 里出现的 680-goal 命令（`e8_bones_seed_4090x2` 那套）**已废弃**：
> token 版 `Policy`/`Decoder` 形状不同，**checkpoint 互不兼容**，不要互相 `--resume`。

---

## 0. 分工：哪些事已经做完，H200 上只需要做什么

| 事项 | 在哪做 | 状态 |
|---|---|---|
| 依赖环境（torch 2.14.1+cu126 / warp 1.17 / mjlab 1.6 / mujoco 3.16） | 本机 | ✅ 装好并**做成自包含**（§2） |
| `g1.tar.gz` 23.5 GB 下载 + 解压 142,220 条 CSV | 本机 | ✅ 完成 |
| CSV → 50 Hz npz（142,220 条 / 97 GB） | 本机 | ✅ 完成，`data/npz/bones_seed/` |
| 名字 + 可行性过滤（129,943 条 / 263.75 h） | 本机 | ✅ 完成，`manifest_filtered.csv` |
| 紧凑 fp16 动作库（47,475,680 帧 / 6.74 GB） | 本机 | ✅ 完成，`data/library/bones_seed_compact.pt` |
| **4×H200 训练** | **H200** | ⬜ 本文 §4 |
| 评测 / 回放 | H200 或本机 | ⬜ 本文 §6 |

**绝对路径（两台机器应当一致；不一致就按 §1.1 改 `REPO`/`DATA` 变量）**

```bash
REPO=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/third_party/SuperMimic-Reproduced
DATA=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/datasets/bones-seed/v1
LIB=$REPO/data/library/bones_seed_compact.pt
```

### 0.1 最短路径（token 架构 · 在 H200 上按这个顺序走）

```bash
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
REPO=$COMIMIC/third_party/SuperMimic-Reproduced
cd $REPO && unset PYTHONPATH

# ① 离线预检（唯一权威脚本）：4 张 H200 / warp 4 设备 / encoder 逐位一致 / 库 / 离线资产
.venv/bin/python ../../scripts/preflight_4xh200_offline.py --expect-gpus 4

# ② 三档 200 迭代基准，选 it/s 最高且不 OOM 的档（见 FINAL_HANDOVER §6.5.3）
cd $COMIMIC
PROFILE=paper ITER=200 RUN=bench_paper bash scripts/launch_token_train_4xh200.sh
PROFILE=high  ITER=200 RUN=bench_high  bash scripts/launch_token_train_4xh200.sh
PROFILE=max   ITER=200 RUN=bench_max   bash scripts/launch_token_train_4xh200.sh

# ③ 正式 320k（论文口径；建议单独再跑一行 nohup 落到日志文件）
cd $COMIMIC
setsid nohup env PROFILE=paper bash scripts/launch_token_train_4xh200.sh \
  > /tmp/train_token_h200x4.log 2>&1 < /dev/null &
```

launcher 自动带上：`--policy.token-dim 64`、`SONIC_ENCODER_ONNX`、`SM_TF32=1`、
`HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1`、`PYTHONPATH=`、`OMP_NUM_THREADS=16`、
`--policy-trainer.horizon 32`、`--schedule.updates-per-env-step 0.4`、`--schedule.checkpoint-interval 4000`。
档位（每卡）：`paper` 4096 env / batch 4096 / wm 2048；`high` 8192 / 8192 / 4096；`max` 16384 / 8192 / 8192。

ETA 用这条算：`320000 / it_s / 3600` 小时（§7 有对照表；注意 `paper` 档最快，加 batch 只提高利用率不加速度）。

---

## 1. 到 H200 后的 8 条离线自检（**先跑这一整块，全过再谈训练**）

```bash
REPO=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/third_party/SuperMimic-Reproduced
cd $REPO
unset PYTHONPATH                 # 节点全局 PYTHONPATH 里有别的 python 会直接崩，必须先清

# ① 四张卡（不是 4 张就别往下走）
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# ② 驱动（torch cu126 要驱动 ≥ 525；warp 1.17 要能编 sm_90）
nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u

# ③ venv 解释器（关键：base_prefix 必须指向 .venv-python，不能是 /root/...）
.venv/bin/python -c "import sys; print(sys.version); print('base_prefix =', sys.base_prefix)"

# ④ torch 看到 4 张卡
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"

# ⑤ warp 枚举到 cuda:0..3 —— 只看到 cpu 就是白装，必须停
.venv/bin/python -c "import warp as wp; wp.init(); print(wp.get_devices())"

# ⑥ 动作库在、字节数对、md5 对
ls -l $LIB
md5sum $LIB        # 期望 f82de7fae8574aaeb17330dcc4e778ef

# ⑦ 包版本没被别的环境串味
.venv/bin/pip list | grep -iE "^(torch|mujoco|mujoco-warp|warp-lang|mjlab|rsl-rl-lib|numpy|pandas|pyarrow) "

# ⑧ 编译工具链在（inductor 首次编译要 gcc，不要网）
which gcc g++ && gcc --version | head -1
```

**期望输出**

| # | 期望 |
|---|---|
| ① | 4 行 `GPU 0..3: NVIDIA H200`，`memory.total` ≈ `143771 MiB` |
| ② | 单个版本号，**≥ 525**（建议 ≥ 570；本机是 595.71.05） |
| ③ | `3.12.14`，`base_prefix = .../SuperMimic-Reproduced/.venv-python` |
| ④ | `2.14.1+cu126 4` + `NVIDIA H200` |
| ⑤ | `['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']`，每张标 `H200 (…)` |
| ⑥ | `6.3G`，md5 = `f82de7fae8574aaeb17330dcc4e778ef` |
| ⑦ | torch 2.14.1+cu126 / mujoco 3.16.0 / mujoco-warp 3.16.0 / warp-lang 1.17.0 / mjlab 1.6.0 / rsl-rl-lib 5.5.1 / numpy 2.5.4 / pandas 3.1.0 / pyarrow 26.0.0 |
| ⑧ | `/usr/bin/gcc`、`/usr/bin/g++`，gcc 13.x |

### 1.1 如果路径不一致

`REPO`、`LIB` 是**绝对路径写死在** `.venv/pyvenv.cfg`、`.venv/bin/*` 的 shebang、`src/supermimic.egg-link`/`.pth`、`logs/` 里。若 H200 上挂载点不同（比如 `/mnt/...`），**不要重装**，做符号链接即可：

```bash
sudo mkdir -p /inspire/hdd/global_user/liumengfan-253108110079   # 视权限而定
ln -s /你的/实际/挂载点 /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace
# 或者反过来：在 H200 上把同一份目录 bind mount 到老路径
#   sudo mount --bind /mnt/xxx/wsc-workspace /inspire/hdd/.../wsc-workspace
```

只要 `REPO` 这个绝对路径能解析，`.venv`、`.venv-python`、editable 安装、`data/`、`logs/` 就全都自动正确，**不需要 pip install 任何东西**。

---

## 2. 环境为什么能直接搬：`.venv-python`（本次新增的关键东西）

原问题：交接过来的 `.venv/bin/python` 只是个软链，指向 uv 管理的独立 CPython `/root/.local/share/uv/python/cpython-3.12.13.../bin/python3.12`。**换一台机器 `/root` 不一样，整个 venv 直接 `No such file or directory`**，而 H200 没网，`uv python install` 也救不了。

本次已经修好，做法是把解释器本体搬进共享目录，venv 从此自包含：

```bash
# 已完成（无需在 H200 上重做）：
#   cp -a /root/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu  $REPO/.venv-python
#   .venv/bin/python3.12 -> $REPO/.venv-python/bin/python3.12
#   .venv/pyvenv.cfg  home = $REPO/.venv-python/bin
```

现在结构是：

```
$REPO/.venv/           11 GB   所有依赖（torch/mujoco/mjlab/warp/...）
$REPO/.venv-python/   210 MB   独立 CPython 3.12.14（自包含，随共享盘一起走）
$REPO/data/            ~104 GB 动作库 + npz
$REPO/logs/                    训练日志/ckpt（rank0 写）
```

**验证过一次就算过**：`.venv/bin/python -c "import torch, mujoco, mjlab, supermimic, warp"` 全通，且 `torchrun` 的 shebang 指向 `.venv/bin/python`（相对共享盘），所以 H200 上不需要改任何脚本。

### 2.1 万一 H200 上 `.venv/bin/python` 还是起不来

按顺序试，**全程不联网**：

```bash
# 诊断
ls -l $REPO/.venv/bin/python3.12
cat $REPO/.venv/pyvenv.cfg
ls -l $REPO/.venv-python/bin/python3.12

# ① `.venv-python` 存在但软链断了 → 重指
ln -sfn $REPO/.venv-python/bin/python3.12 $REPO/.venv/bin/python3.12

# ② `.venv-python` 整个丢了 → 从本机（或同镜像的 /root）再拷一次
cp -a /root/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu $REPO/.venv-python
ln -sfn $REPO/.venv-python/bin/python3.12 $REPO/.venv/bin/python3.12

# ③ 最后一招：用 H200 机器自带的 3.12（只要 3.12.x，cp312 的二进制轮子全兼容）
H200PY=$(command -v python3.12)
ln -sfn $H200PY $REPO/.venv/bin/python3.12
sed -i "s|^home = .*|home = $(dirname $H200PY)|" $REPO/.venv/pyvenv.cfg

# 每次改完都验一遍
$REPO/.venv/bin/python -c "import sys, torch; print(sys.version, torch.__version__, torch.cuda.device_count())"
```

> 这三步都不需要 root 以外的特殊权限，也不写 `/root`，全部落在共享盘里。

---

## 3. 离线兜底（H200 上不会联网，这些必须在跑之前想清楚）

| 风险 | 离线处置 | 是否已验证 |
|---|---|---|
| 少了 `libEGL.so.1`，`import mujoco` 时报 `OpenGL.raw.EGL ... 'NoneType' object has no attribute 'eglQueryString'` | **`export MUJOCO_GL=disable`**。训练/评测完全不用渲染（play.py 的可视化才需要），本机实测 `MUJOCO_GL=disable` 下 `import mujoco + load_g1_model()` 正常（`nq=36`） | ✅ 已在本机实测 |
| warp / Triton / inductor 首次编译 | 全部在**本地**生成，只写 `/root/.cache/warp`、`/tmp/torchinductor_root`、`~/.triton`，**不联网**。第一次跑前几分钟慢是这个原因，不是卡坏了 | ✅ 本机实测 |
| mjlab 资产（G1 的 XML/mesh） | 已经打包在 wheel 里（`mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml`，65 MB），**运行时不下载** | ✅ 已核实文件在 |
| 数据集 gated / HF token | 训练**完全不碰** HF。`--data.library` 只读本地那一个 `.pt` | ✅ |
| `torch.compile` 需要 gcc | 自检 ⑧ 已确认 | ✅ |
| 万一真想补装东西 | 本机 `/root/.cache/pip`（6.1 GB）里有全部轮子，**离线**拷过去可以 `pip install --no-index --find-links` | 备用 |

---

## 4. 训练（4 卡）

### 4.0 启动前最后一眼

```bash
cd $REPO
unset PYTHONPATH
export OMP_NUM_THREADS=16          # 128 核 / 4 rank = 32 核每 rank，脚本还会自动切 affinity
nvidia-smi --query-gpu=index,memory.used --format=csv      # 4 张卡应基本空着
```

> **不要手动设 `CUDA_VISIBLE_DEVICES`**。`src/supermimic/scripts/train.py:20-37` 会按 `LOCAL_RANK` 自动把每个 rank 钉到一张卡上（并设 `SM_GPU_PINNED=1`），手动设会把 4 个 rank 挤到同一张卡上直接 OOM。

### 4.1 冒烟（300 迭代，约 5–15 分钟，含首次编译）

> 用 launcher 跑冒烟最省事（自动带 token 参数）；下面的手动版**必须**保留 `--policy.token-dim 64` 与 `SONIC_ENCODER_ONNX`，否则测的是旧架构。

```bash
cd $COMIMIC && setsid nohup env NPROC=4 PROFILE=paper ENVS=1024 BATCH=1024 WMB=1024 ITER=300 \
  RUN=smoke_token_h200x4 bash scripts/launch_token_train_4xh200.sh \
  > /tmp/smoke_token_h200x4.log 2>&1 < /dev/null &

grep -E "\[launch\]|\[sonic\]|^it " /tmp/smoke_token_h200x4.log   # 期望 [sonic] token conditioning enabled (dim=64)
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv   # 4 张卡都要有显存
```

手动等价命令（排查 launcher 问题时用）：

```bash
cd $REPO && unset PYTHONPATH && export OMP_NUM_THREADS=16 \
  SONIC_ENCODER_ONNX=$PWD/data/assets/model_encoder.onnx SM_TF32=1 HF_HUB_OFFLINE=1

setsid nohup .venv/bin/torchrun --nproc_per_node 4 --master_port 29500 \
  -m supermimic.scripts.train \
  --run-name smoke_token_h200x4 \
  --data.library data/library/bones_seed_compact.pt \
  --policy.token-dim 64 \
  --env.num-envs 1024 --policy-trainer.batch-size 1024 --wm-trainer.batch-size 1024 \
  --schedule.total-iterations 300 --schedule.checkpoint-interval 100000 \
  > /tmp/smoke_token_h200x4.log 2>&1 < /dev/null &

tail -f /tmp/smoke_token_h200x4.log        # 等第一行 `it 50 | ... it/s ...`
```

看到 `params: world model 8.82M, policy 8.89M, critic 0.235M` 和 `it 50 | … 2.4 it/s …` 形式的日志就说明全链路通了。**记下这里的 `it/s` 和显存峰值**，用来算正式跑的 ETA（§7）。

### 4.2 正式跑：4×H200 · token 架构 · 320k 迭代

**推荐：用 launcher 脚本（唯一权威入口，参数已内置）**

```bash
cd $COMIMIC
setsid nohup env PROFILE=paper bash scripts/launch_token_train_4xh200.sh \
  > /tmp/train_token_h200x4.log 2>&1 < /dev/null &
```

`PROFILE=paper`（论文口径，最快收工）／`high`／`max`（更吃满 GPU，但更慢）。自定义覆盖示例：

```bash
NPROC=4 PROFILE=paper ENVS=4096 BATCH=4096 WMB=2048 ITER=320000 \
  RUN=e9_bones_seed_token_h200x4 bash scripts/launch_token_train_4xh200.sh
```

**手动等价命令**（launcher 出问题时的排错路径，token 参数一个都不能少）

```bash
cd $REPO && unset PYTHONPATH && export OMP_NUM_THREADS=16 \
  SONIC_ENCODER_ONNX=$PWD/data/assets/model_encoder.onnx SM_TF32=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

setsid nohup .venv/bin/torchrun --nproc_per_node 4 --master_port 29500 \
  -m supermimic.scripts.train \
  --run-name e9_bones_seed_token_h200x4 --seed 0 \
  --data.library data/library/bones_seed_compact.pt \
  --policy.token-dim 64 \
  --env.num-envs 4096 \
  --policy-trainer.horizon 32 \
  --policy-trainer.batch-size 4096 \
  --wm-trainer.batch-size 2048 \
  --schedule.total-iterations 320000 \
  --schedule.updates-per-env-step 0.4 \
  --schedule.checkpoint-interval 4000 \
  > /tmp/train_token_h200x4.log 2>&1 < /dev/null &

echo "launched"
```

> **为什么用 `setsid nohup … < /dev/null &`**：裸 `nohup … &` 挂在一个会被回收的 shell 会话里，会话一断整组 SIGTERM，训练会“莫名其妙消失”（本机踩过一次）。`setsid` 开独立会话，断开也不影响。
> **不要用 `--policy-trainer.compile reduce-overhead`**：本机实测会卡在 cudagraphs 的慢路径（比 baseline 慢 4×）。保持默认 `default`。

### 4.3 只想先跑一段（可续训）

```bash
# 加 --schedule.max-hours 8，到点自动存 ckpt_latest.pt 并退出
... --schedule.max-hours 8 ...
# 之后用 §5 的 --resume 接着跑，迭代计数和 lr 曲线都接着走
```

### 4.4 参数为什么这么配（已对代码核实）

| 参数 | 4×H200 取值 | 依据 / 说明 |
|---|---|---|
| 启动方式 | `torchrun --nproc_per_node 4` | `training/distributed.py:24-38`，一进程一卡 |
| rank 绑卡 | 自动（`CUDA_VISIBLE_DEVICES=LOCAL_RANK`） | `scripts/train.py:23-24` |
| 学习率 | **自动 ×√4 = ×2** | `training/loop.py:37-40`；单卡是 ×1。**不要手改 lr** |
| `--env.num-envs` | 4096 / 卡 | 论文 Table A.5「每 GPU 4096」 |
| `--policy-trainer.batch-size` | 4096 / 卡 | 与 2×H200 口径一致；H200 141 GB 显存无压力（预估 ≤30 GB） |
| `--env.future-frames` | 10 | 论文 |
| `--policy-trainer.horizon` | 32 | 论文 H=32 |
| `--schedule.updates-per-env-step` | 0.4 | 论文 UTD |
| `--schedule.total-iterations` | 320000 | 论文 |
| 检查点 | 每 4000 迭代 | `ckpt_latest.pt` + `ckpt_XXXXXXX.pt`，仅 rank0 写 |
| 日志 | `logs/<run-name>/tb` + `config.json` | TensorBoard 直接读 |

> ⚠️ **4 卡 vs 论文 2 卡的口径差异（必须知道）**：`total_iterations` 是**每 rank** 的；4 卡 × 4096 env × 320k 迭代 = **5.2×10⁹ 环境步**，正好是论文 2 卡（2.6×10⁹）的 **2 倍**。也就是说 4×H200 跑 320k 属于“更多数据”的加强版，**不要**和论文 Table A.8 的数值逐位对比。要严格对齐论文的数据量，得用 2 卡跑 320k（见 `BONES_SEED_H200X2.md` §4.1）。

---

## 5. 断点续训 / 跨卡数接续

### 5.1 同配置接着跑（4 卡，token 版）

```bash
cd $COMIMIC
setsid nohup env PROFILE=paper bash scripts/launch_token_train_4xh200.sh \
  --resume logs/e9_bones_seed_token_h200x4/ckpt_latest.pt \
  > /tmp/train_token_h200x4_resume.log 2>&1 < /dev/null &
```

续训会恢复 wm / policy / critic 权重 + 三个优化器状态 + `bin_return`/`bin_memory` + `iteration`/`num_updates`，lr 余弦曲线从原迭代计数继续。

### 5.2 ⚠️ 旧架构 checkpoint 不能接续到 token 版

`logs/e8_bones_seed_4090x2/ckpt_latest.pt` 是**旧 680-goal 架构**（`token_dim=0`，Decoder 输入 `latent+930`）。
token 版（`token_dim=64`）的 `Policy`/`Decoder` 形状不同，**`--resume` 会直接 shape mismatch**。
FINAL_HANDOVER §7 也明确写了"token 版策略与旧 680-goal checkpoint 不兼容，需要重训"。

- 旧 ckpt 只能当**参照物**（世界模型、奖励、世界模型收敛曲线），不能当初始化；
- token 版必须在 4×H200 上从头训（或从它自己的 ckpt 续）；
- 同理：两版不能混跑对比 loss/return。

跨卡数接续（4 卡 → 2 卡 → 1 卡）对 token 版依然成立，只改 `NPROC`：

```bash
NPROC=2 PROFILE=paper bash scripts/launch_token_train_4xh200.sh \
  --resume logs/e9_bones_seed_token_h200x4/ckpt_latest.pt
```

### 5.3 降卡接续（token 版，4 卡 → 2 卡 → 1 卡都行）

只改 `NPROC` 和 `--resume`，token 参数由 launcher 自动带：

```bash
# 2 卡
NPROC=2 PROFILE=paper bash scripts/launch_token_train_4xh200.sh \
  --resume logs/e9_bones_seed_token_h200x4/ckpt_latest.pt

# 单卡（手动版，world=1 时 lr 不放大；注意仍要带 --policy.token-dim 64 + encoder）
cd $REPO && export PYTHONPATH= SM_TF32=1 SONIC_ENCODER_ONNX=$PWD/data/assets/model_encoder.onnx
OMP_NUM_THREADS=32 .venv/bin/python -m supermimic.scripts.train \
  --run-name e9_bones_seed_token_h200x1 --data.library data/library/bones_seed_compact.pt \
  --policy.token-dim 64 \
  --env.num-envs 4096 --policy-trainer.batch-size 2048 --wm-trainer.batch-size 2048 \
  --schedule.total-iterations 320000 \
  --resume logs/e9_bones_seed_token_h200x4/ckpt_latest.pt
```

---

## 6. 监控 / 评测 / 回放

```bash
cd $REPO && unset PYTHONPATH

# 训练曲线（文本，无需浏览器）
.venv/bin/python scripts/tb_tail.py --run-dir logs/e9_bones_seed_token_h200x4 --last 3
.venv/bin/python scripts/tb_tail.py --run-dir logs/e9_bones_seed_token_h200x4 --filter policy/

# 关键指标的日志行（每 50 迭代一行）
grep -E "^it " /tmp/train_h200x4.log | tail -5

# TensorBoard（有端口映射时）
.venv/bin/tensorboard --logdir logs/e9_bones_seed_token_h200x4/tb --port 6006 --bind_all

# 看 ckpt
ls -lh logs/e9_bones_seed_token_h200x4/

# 全量 benchmark（论文 Appendix F，走 npz 目录，不是紧凑库）
.venv/bin/python scripts/eval_benchmark.py \
  --ckpt logs/e9_bones_seed_token_h200x4/ckpt_latest.pt \
  --npz-dir data/npz/bones_seed \
  --manifest data/npz/bones_seed/manifest_filtered.csv \
  --num-envs 1024

# 交互回放（无显示时用 viser；注意 eval/play 走 npz 目录，不是紧凑库）
.venv/bin/python scripts/play.py --ckpt logs/e9_bones_seed_token_h200x4/ckpt_latest.pt \
  --npz-dir data/npz/bones_seed --viewer viser
```

日志里每行的含义（`loop.py:247-253`）：

```
it 500 | 0.06 h | 2.4 it/s | updates 184 | train_ret 0.507 (n=11962) |
wm nse1 0.0228 nse8 0.0477 | imag_ret 0.889 surv 0.324 | t sim/wm/pi 2.4/5.0/13.5s
      ↑迭代      ↑已跑小时  ↑速度      ↑梯度更新数  ↑训练回合归一化回报  ↑世界模型 1/8 步误差 ↑想象回报/存活 ↑50 迭代内 仿真/世界模型/策略 的耗时
```

---

## 7. 要跑多久（实测校准，别照抄估算）

**公式**：`ETA(小时) = 320000 / it_s / 3600`，`it_s` 直接取第一行 `it 50 | … X it/s`。

| H200×4 实测 `it/s` | 320k 迭代 ETA | 说明 |
|---|---|---|
| 4 | 22.2 h | 偏慢，先查是不是 CPU 抢核/别的任务在跑 |
| 6 | 14.8 h | 手册对 2×H200 的估算值（6–8 it/s） |
| 8 | 11.1 h | 乐观 |
| 12 | 7.4 h | 需要更快的世界模型/策略步（batch 更大、编译更好） |

参考基准（**同一份代码、本机 2×4090 实测**）：`2.3–2.4 it/s` → 320k 要约 **37 h**，显存 19.6 GB/卡，GPU 利用率 52–90%，功耗 218/450 W。4090 的 TF32 算力约为 H200 的 1/6，所以 H200 上落进 6–8 it/s 是合理预期。

⚠️ **4 卡不会减少迭代数**：`total_iterations` 是每 rank 计的，加卡等于“每次迭代喂更多数据”，不改变“跑 320k 次”这件事本身的耗时（和本机 1 卡 vs 2 卡的关系一样）。想让墙钟变短只有：**降 total_iterations** 或 **加 `--schedule.max-hours` 分批跑**。

---

## 8. 离线排错表（H200 专属，没有网 = 不能现场补装）

| 现象 | 原因 | 处置 |
|---|---|---|
| `.venv/bin/python: No such file or directory` | `.venv/bin/python3.12` 断链 / `.venv-python` 没随盘过来 | §2.1 三步，先 `ln -sfn` 再考虑拷解释器 |
| `base_prefix` 指向 `/root/...` 而不是 `.venv-python` | `pyvenv.cfg` 的 `home` 还是老的 | `sed -i "s|^home = .*|home = $REPO/.venv-python/bin|" .venv/pyvenv.cfg` |
| `wp.init()` 只有 `cpu` | warp 要的 CUDA runtime 和驱动不匹配 | **无网不能降级**：确认 ② 驱动版本；warp 1.17 = CUDA 12.9 runtime，驱动 ≥ 525；不满足只能换驱动/换机器 |
| `import torch` 报 `CUDA driver version is insufficient` | 驱动低于 cu126 要求 | 同上，只能换驱动；**别试图 pip 换 torch（无网）** |
| `import mujoco` 报 `OpenGL.raw.EGL` / `eglQueryString` | 机器没装 `libEGL.so.1` | `export MUJOCO_GL=disable`（训练不需要渲染）✅ 已实测 |
| `torchrun: command not found` | 没走 venv | 用 `.venv/bin/torchrun` 绝对路径 |
| 4 个 rank 全挤 GPU0 / OOM | 手动设了 `CUDA_VISIBLE_DEVICES` | 去掉该变量，让 `train.py` 按 `LOCAL_RANK` 绑卡 |
| NCCL 初始化卡住 / `Address already in use` | 端口被占（本机用过 29500） | 换 `--master_port 29511`；必要时 `NCCL_P2P_DISABLE=1` |
| `FileNotFoundError: no clips under ...` | `--data.library` 路径不对 / 库不在 | §1 ⑥ 校验；或改用 `--data.npz-dir data/npz/bones_seed` |
| 多卡启动很慢、卡在 import | 节点全局 `PYTHONPATH` 串了别的 python | `unset PYTHONPATH` |
| 前几分钟 GPU 利用率很低、慢了 5–10 倍 | warp kernel + Triton/inductor **首次本地编译** | 正常，等第一行 `it 50`；缓存落在 `/root/.cache/warp`、`/tmp/torchinductor_root`，重启 `torchrun` 会命中 |
| 训练进程“莫名消失” | 裸 `nohup … &` 被会话回收连带 SIGTERM | 用 `setsid nohup … < /dev/null &`（§4.2） |
| 想杀进程结果把自己的 shell 杀了 | `pkill -f "包含完整命令行的模式"` 会匹配到自己 | 先 `pgrep -f '关键字'` 看 PID，再 `kill <PID>`；或用 `pkill -f "_bench_[c]udagraph"` 这种方括号技巧 |
| 曲线和论文对不上 | 4 卡 × 320k = 论文 2 倍的样本量；lr 又放大到 ×2 | 见 §4.4 的口径说明；要严格对齐就用 2 卡 |

---

## 9. 产物清单与校验值（H200 上核对用）

| 产物 | 路径 | 大小 | 校验 |
|---|---|---|---|
| 紧凑动作库 | `$REPO/data/library/bones_seed_compact.pt` | 6.3 GB | `md5 = f82de7fae8574aaeb17330dcc4e778ef`，47,475,680 帧 / 263.75 h |
| 过滤清单 | `$REPO/data/npz/bones_seed/manifest_filtered.csv` | ~3 MB | 129,943 行 keep=1（+1 行表头）；总时长 263.75 h |
| npz 动作 | `$REPO/data/npz/bones_seed/*.npz` | 97 GB | `find … -name '*.npz' \| wc -l` → 142,220 |
| 原 CSV | `$DATA/g1/csv/*/*.csv` | ~52 GB | 142,220 条 |
| 原始 tar | `$DATA/g1.tar.gz` | 23.5 GB | 确认 142,220 已解压后可删 |
| 训练日志 | `$REPO/logs/<run-name>/` | — | `tb/`、`config.json`、`ckpt_latest.pt` |

> 磁盘占用汇总：npz 97 GB + CSV 52 GB + tar 23.5 GB + 库 6.7 GB + 依赖 11 GB ≈ **190 GB**。

---

## 10. 参考

- 本机现状与数据管线：[`BONES_SEED_HANDOVER.md`](BONES_SEED_HANDOVER.md)
- 2×H200 目标规格书（严格论文口径）：[`BONES_SEED_H200X2.md`](BONES_SEED_H200X2.md)
- 上游复现仓库与论文数值：`third_party/SuperMimic-Reproduced/README.md`、`docs/EXPERIMENTS.md`、`docs/PAPER_SPEC.md`
- SuperMimic 论文：<https://arxiv.org/abs/2610.28480>
