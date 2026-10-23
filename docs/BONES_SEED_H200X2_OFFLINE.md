# SuperMimic BONES-SEED 训练手册（**2×H200 · 全程离线 · token 架构**）

> **最终决定**：走 **2×H200 路线**（论文口径：每卡 4096 env、320k 迭代、样本量与论文 1:1、lr 自动 ×√2）。
> **目标机器无外网**：不装任何东西、不访问 HF/数据集，**全部复用当前这份已装好并做成自包含的环境**。
> **入口**：`scripts/launch_token_train_2xh200.sh`（薄封装，钉死 `NPROC=2 PROFILE=paper`）→ 内部 exec `scripts/launch_token_train_4xh200.sh`（**唯一含全部训练参数的脚本**，token 参数/ TF32 / 离线环境变量都在那里）。
> **唯一权威的架构说明**：[`FINAL_HANDOVER.md`](FINAL_HANDOVER.md)（token 条件化 + 手部独立通道）；本文只负责**这台离线机器上怎么跑起来**。
> 文档时间：2026-10-24。本文每条命令都在本机（2×4090）实跑验证过；2×H200 的实测 it/s 已于同日补齐（2.3–2.7 it/s，见 §8.1）。
>
> 🚑 **起不来先看 §0.3（三条命令收现场 + 一键打包）**；`nvidia-smi` 连卡都列不出 → **§0.4 整机 GPU 没起来**。
>
> ⏱ **实测 2×H200 = 2.3–2.7 it/s → 320k 约 37 h**；为什么换 H200 也没变快、以及三条提速路 → **§8.1 / §8.2**（瓶颈在 host 单线程，不在 GPU）。

---

## 0. 一页纸：**只跑正式训练（最常见）**

```bash
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
REPO=$COMIMIC/third_party/SuperMimic-Reproduced

# ① 离线预检（不联网；exit 0 才继续）
cd $REPO && unset PYTHONPATH
.venv/bin/python ../../scripts/preflight_4xh200_offline.py --expect-gpus 2

# ② 正式 320k（论文口径）—— 一条命令，前台跑更安全（Ctrl-C 可控；见 §0.2 后台版）
cd $COMIMIC
bash scripts/launch_token_train_2xh200.sh 2>&1 | tee /tmp/train_token_h200x2.log
```

**另开一个终端**看它有没有真的起来：

```bash
sleep 90
grep -E "\[launch\]|\[sonic\]|params:|^it " /tmp/train_token_h200x2.log
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv
```

### 0.1 想先跑基准再正式（**两件事必须串行**）

```bash
cd $COMIMIC
# 先基准（200 迭代，约 5–15 分钟含首次编译），等它跑完或手动停掉
ITER=200 RUN=bench_2xh200 bash scripts/launch_token_train_2xh200.sh 2>&1 | tee /tmp/bench_2xh200.log
# 记下第一个 `it 50 | … X it/s` 和 nvidia-smi 的显存峰值，再起正式：
bash scripts/launch_token_train_2xh200.sh 2>&1 | tee /tmp/train_token_h200x2.log
```

> ⚠️ **绝对不要同时起两个**。它们会抢同一组卡和同一个 `PORT`，结果是"两个都变慢"或"后起的那个静默死掉"。
> launcher 现在自带两道守卫：显存 >8 GB 被占用时直接报错退出（`FORCE=1` 可强制），端口被占则自动 +1 换端口。

### 0.2 后台跑（断开终端也不停）

```bash
cd $COMIMIC
setsid nohup bash scripts/launch_token_train_2xh200.sh > /tmp/train_token_h200x2.log 2>&1 < /dev/null &
```

> 🔴 **`setsid … &` 立刻显示 `Done` 是正常的，不代表成功**：`setsid` 会 fork 一个父进程再退出，
> shell 看到的是"父进程正常退出"。真正的进程是它的子进程，**必须用下面的方法确认**：

```bash
pgrep -af "supermimic[.]scripts[.]train"                    # 有 2~5 行 → 在跑
grep -E "\[launch\]|\[sonic\]|^it |Error|Traceback|Address already in use|CUDA out of memory" /tmp/train_token_h200x2.log | tail -20
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv   # 几十秒后应看到 ~20 GB/卡
```

`bash scripts/launch_token_train_2xh200.sh` 自动等价于：

```
NPROC=2  PROFILE=paper  ITER=320000
  ENVS=4096 (每卡)  BATCH=4096 (policy, 每卡)  WMB=2048 (world model)
  RUN=e9_bones_seed_token_h200x2   PORT=29500   CKPT_INTERVAL=4000
```

### 0.3 🚑 起不来时：先收现场（三条命令 + 一键打包）

**启动后 GPU 没动静 / `it` 行不出来 / `Done` 但显存 0 MiB —— 先别重复重启，按这三条收现场。**

```bash
# ① 进程到底还在不在（`setsid … &` 报 Done 是假象，必须看这个）
pgrep -af "supermimic[.]scripts[.]train" | head

# ② 日志里到底报了什么（这几类关键字覆盖 95% 的启动失败）
grep -nE "\[launch\]|\[sonic\]|params:|^it |Error|Traceback|Address already in use|CUDA out of memory|NCCL|insufficient" \
  /tmp/train_token_h200x2.log /tmp/bench_2xh200.log 2>/dev/null | tail -40

# ③ 卡上有没有真的被占用（起来的进程会占 ~20–25 GB/卡）
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv
```

**一键打包（把输出整段贴给对接人，比挤牙膏快）**

```bash
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
{
  echo "########## 1. 进程 ##########"
  pgrep -af "supermimic[.]scripts[.]train" || echo "(没有训练进程)"
  echo "########## 2. 日志尾 ##########"
  tail -60 /tmp/train_token_h200x2.log 2>/dev/null || echo "(没有日志文件)"
  echo "########## 3. 关键错误 ##########"
  grep -nE "\[launch\]|\[sonic\]|params:|^it |Error|Traceback|Address already in use|CUDA out of memory|NCCL|insufficient" \
    /tmp/train_token_h200x2.log 2>/dev/null | tail -40
  echo "########## 4. GPU ##########"
  nvidia-smi -L 2>&1
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,driver_version --format=csv 2>&1
  echo "########## 5. 占用进程 ##########"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1
  echo "########## 6. 环境 ##########"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "PYTHONPATH=${PYTHONPATH:-<unset>}"
  echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
  echo "########## 7. warp/torch 复述 ##########"
  cd $COMIMIC/third_party/SuperMimic-Reproduced 2>/dev/null && \
    .venv/bin/python -c "import torch, warp as wp; print('torch', torch.__version__, 'cuda', torch.cuda.device_count()); wp.init(); print('warp', wp.get_devices())" 2>&1 | tail -15
} 2>&1 | tee /tmp/diag_2xh200.txt
```

**照症状对号入座**

| 现场特征 | 结论 | 处置 |
|---|---|---|
| ① 没进程，② 里 `Address already in use` | 两条训练撞了同一个 `PORT`（最常见，尤其是"基准+正式"一起起） | 只起一条；launcher 已加端口自动避让，手敲命令时用 `PORT=29511` |
| ① 没进程，② 里 `CUDA out of memory` | 上一条没死透还占着卡，或用了 `max` 档 | `pgrep -af supermimic` → `kill <PID>`；改 `ENVS=2048 BATCH=2048` |
| ① 没进程，② 里 `No module named 'supermimic'` | 用了系统 `torchrun`（`/usr/local/bin/torchrun`，shebang 是 `/usr/bin/python`） | 必须用 `.venv/bin/torchrun`；launcher 已修，别手敲 |
| ① 没进程，② 里 `warp 只看到 0 张 CUDA 卡` / `需要 2 张卡` | 驱动或 warp 版本问题 | 见 §1、§9；无网不能重装，确认 `warp-lang 1.17.0` |
| ① 没进程，② 里 `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` | **机器缺 `libEGL.so.1`**（2×H200 实证，两个 rank 都死在 import） | launcher 已默认 `MUJOCO_GL=disable`；手敲命令要自己 `export MUJOCO_GL=disable`，详见 §3.1 |
| ① 有进程，③ 显存 20 GB+ 但 `it` 还不出现 | **正常**：首次 warp kernel + Triton/inductor 本地编译（5–15 分钟） | 等；`tail -f` 日志，别 kill |
| ① 有进程，③ 显存 0 MiB 且一直不动 | 卡在 NCCL 建链或加载动作库 | `NCCL_P2P_DISABLE=1 PORT=29511` 再试；看 ② 里最后一行 |
| ③ `nvidia-smi` 本身报 `No devices were found` / 没有任何 GPU | **整机 GPU 没起来**（不是训练的问题） | 见下面 §0.4 |

### 0.4 整机 GPU 没起来（`nvidia-smi` 列不出卡）

这类问题跟训练脚本无关，先定性：

```bash
nvidia-smi -L                                    # 期望列出 2 行 GPU
ls -l /dev/nvidia* 2>&1 | head                   # 期望有 /dev/nvidia0 /dev/nvidia1 /dev/nvidiactl
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
cat /proc/driver/nvidia/version 2>&1 | head -3   # 内核模块版本 vs 用户态驱动是否一致
dmesg 2>/dev/null | grep -i -E "nvidia|nvrm|xid" | tail -20   # 需要权限
ldconfig -p 2>/dev/null | grep -c libnvidia      # 期望 > 20
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"   # 容器里被限制最常见的元凶
nvidia-smi -q 2>&1 | grep -iE "mig|pending|failed|ecc mode" | head
```

| 现象 | 多半是 | 能做什么 |
|---|---|---|
| `nvidia-smi: command not found` | 驱动没装/容器没挂 | 找运维；离线环境装不了 |
| `No devices were found` 但 `/dev/nvidia*` 在 | 用户态驱动与内核模块版本不匹配（`/proc/driver/nvidia/version` 与 `nvidia-smi --version` 对比） | 找运维统一版本；**不要试图 pip 装 torch 来绕** |
| `/dev/nvidia*` 缺失、`NVIDIA_VISIBLE_DEVICES=none/void` | 容器启动时没给 GPU（`--gpus` / device plugin 没生效） | 重启容器时带上 GPU；这是唯一"重进容器就解决"的一类 |
| `nvidia-smi` 正常但没有 `dmesg` 权限 | 正常现象 | 忽略，用 `nvidia-smi -q` 代替 |
| 卡在 `MIG` / `pending` / ECC 报错 | 硬件/驱动状态异常 | 找运维 `nvidia-smi -r`（需要权限） |

> 记住一条：**判断顺序永远是"机器有没有卡 → 卡能不能被 torch/warp 看到 → 训练进程有没有起来 → 有没有在算"**。
> 前三步任何一步不过，外面再重跑 320k 都是白跑；对应命令分别是 §0.4 / §2 ④ / §0.3 ① / §0.3 ③。

---

## 1. 为什么"没有网络"也能跑：逐项对照

| 依赖 | 本地位置 | 大小 | 离线自检 |
|---|---|---|---|
| Python 解释器 | `$REPO/.venv-python/`（自包含 CPython 3.12.14，不依赖 `/root`） | 210 MB | `.venv/bin/python -c "import sys; print(sys.base_prefix)"` → 必须指向 `…/.venv-python` |
| 全部 Python 依赖 | `$REPO/.venv/`（torch 2.14.1+cu126 / mujoco 3.16 / mjlab 1.6 / warp-lang 1.17 / numpy 2.5.4 / pandas / pyarrow …） | 11 GB | `.venv/bin/python -c "import torch,mujoco,mjlab,supermimic,warp"` |
| token encoder（ONNX） | `$REPO/data/assets/model_encoder.onnx` | 48 MiB，**4,231,232 参数** | preflight §3 会与 PyTorch 复现**逐位比对** |
| 紧凑动作库 | `$REPO/data/library/bones_seed_compact.pt` | 6.28 GiB，**129,943 clips / 47,475,680 帧 / 263.75 h** | `md5sum` = `f82de7fae8574aaeb17330dcc4e778ef` |
| mjlab 机器人资产（G1 XML/mesh） | 已打包在 wheel 内 `mjlab/asset_zoo/robots/unitree_g1/` | 65 MB | 运行时不下载（已核实文件在） |
| warp / Triton / inductor 编译缓存 | `~/.cache/warp/1.17.0`、`/tmp/torchinductor_root`、`~/.triton` | 首次自动生成 | **本地编译，不联网**；只是第一次跑慢几分钟 |
| HF 数据集 / token | — | — | **训练完全不碰**；launcher 已 `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` |
| 原始 CSV / npz | `$COMIMIC/third_party/.../data/npz/bones_seed/`（97 GB，142,220 条） | 只在**评测/回放**用 | 训练只读上面那一个 `.pt` |

**结论**：训练路径上没有任何一步需要网络。`gcc/g++` 必须在（inductor 首次编译要用），这是唯一"系统级"依赖。

---

## 2. 到机器上的 8 条自检（先跑这一整块）

```bash
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
REPO=$COMIMIC/third_party/SuperMimic-Reproduced
cd $REPO
unset PYTHONPATH                       # 节点全局 PYTHONPATH 串了别的 python 会直接崩

# ① 卡数必须是 2
nvidia-smi -L | grep -c '^GPU'
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# ② 驱动（torch cu126 要 ≥ 525；warp 1.17 要能编 sm_90）
nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u

# ③ 解释器自包含
.venv/bin/python -c "import sys; print(sys.version); print('base_prefix =', sys.base_prefix)"

# ④ torch / warp
.venv/bin/python -c "import torch, warp as wp; print(torch.__version__, torch.cuda.device_count()); wp.init(); print(wp.get_devices())"

# ⑤ 动作库
ls -l data/library/bones_seed_compact.pt && md5sum data/library/bones_seed_compact.pt

# ⑥ token encoder
ls -l data/assets/model_encoder.onnx

# ⑦ 编译工具链
which gcc g++ && gcc --version | head -1

# ⑧ 路径是否和本机一致（venv 里写死了绝对路径）
grep -n "^home" .venv/pyvenv.cfg
```

**期望**：① 输出 `2`，两张 `NVIDIA H200`（≈143771 MiB）；② 单个版本号 ≥ 525；③ `3.12.14` 且 `base_prefix` 以 `.venv-python` 结尾；④ `2.14.1+cu126 2`，`wp.get_devices()` 含 `cuda:0`,`cuda:1`；⑤ `6.3G` + md5 匹配；⑥ 约 `48M`；⑦ gcc 13.x；⑧ `home` 指向 `$REPO/.venv-python/bin`。

**⑧ 路径不一致怎么办（离线，不能重装）**：venv 里全是绝对路径。用符号链接或 bind mount 让老路径能解析即可：

```bash
# 方式 A：把共享盘挂到老路径
sudo mkdir -p /inspire/hdd/global_user/liumengfan-253108110079
sudo ln -s /实际挂载点/wsc-workspace /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace
# 方式 B：bind mount
sudo mount --bind /实际/挂载点/wsc-workspace /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace
```

---

## 3. 离线预检（一条命令，`exit 0` 才往下走）

```bash
cd $REPO && .venv/bin/python ../../scripts/preflight_4xh200_offline.py --expect-gpus 2
```

它检查并逐项打印 `[ok]`／`[FAIL]`：GPU 数量与型号、驱动、mujoco、warp 的 CUDA 设备数、**encoder 参数 4,231,232 + token 落在 1/16 网格 + 与 ONNX 逐位一致（偏移 `[4:584)`/`[601:661)`）**、动作库 clips/frames/hours、离线资产。

任何一项 `[FAIL]` → 直接停，不要起 320k。本机实跑输出（2 卡）：

```
[1] GPU      [ok] torch 可见 2 张卡 / driver 595.71.05
[2] warp     [ok] mujoco 3.16.0 / warp 1.17.0 CUDA 设备 2 张
[3] encoder  [ok] 4,231,232 参数 · token 在 1/16 网格 · 与部署 ONNX 逐位一致
[4] library  [ok] 129,943 clips / 47,475,680 帧 / 263.75 h
[5] offline  [ok] 训练路径不访问网络
```

### 3.1 ⚠️ 预检通过 ≠ 训练能起：`libEGL` 这个坑（2×H200 实证）

**2026-10-24 在 2×H200 上真实发生**：预检 5 项全 `[ok]`，一跑训练两个 rank 立刻都死在 import：

```
AttributeError: 'NoneType' object has no attribute 'eglQueryString'
  File ".../mujoco/rendering/classic/gl_context.py", line 41, in <module>
  File ".../mujoco/egl/__init__.py", line 31, in <module>
  File ".../OpenGL/raw/EGL/_types.py", line 87, in <module>
```

原因：那台机器**没有 `libEGL.so.1`**（`ctypes.util.find_library('EGL')` → `None`）。`mujoco/__init__.py` 只在 `except ImportError` 时跳过渲染模块，而这个错是 `AttributeError`，所以直接抛穿。

处置（训练不需要渲染，离线也没有 apt 可装）：

```bash
export MUJOCO_GL=disable     # launcher 已默认，用脚本就不用管
```

验证方式（本机 2×4090 实测全过）：

```bash
MUJOCO_GL=disable .venv/bin/python -c "import mujoco, mjlab; from supermimic.data.motion_convert import load_g1_model; print(load_g1_model().nq)"
MUJOCO_GL=disable FORCE=1 ITER=60 ENVS=256 BATCH=256 WMB=256 bash scripts/launch_token_train_2xh200.sh \
  --policy-trainer.compile None --wm-trainer.compile None     # 期望出现 it 50 / updates 4
```

> 只有要**可视化**（`scripts/play.py --viewer native`）时才需要 GL；那时才去补 `libEGL`（离线机器只能从有网机器拷 `libEGL.so.1*` + 设 `LD_LIBRARY_PATH`，不建议）。

---

## 4. 200 迭代基准（决定要不要降配）

```bash
cd $COMIMIC
setsid nohup env ITER=200 RUN=bench_2xh200 bash scripts/launch_token_train_2xh200.sh \
  > /tmp/bench_2xh200.log 2>&1 < /dev/null &
sleep 300; grep -E "\[launch\]|\[sonic\]|^it |Error|OOM" /tmp/bench_2xh200.log
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

**要记下三件事**：① `it/s`；② 每卡显存峰值；③ 有没有 OOM。

- 看到 `[sonic] token conditioning enabled (dim=64)`（两个 rank 各一行）+ `[INFO] params: world model 8.82M, policy 8.96M, critic 0.235M` 就是**架构对了**（policy 8.96M 是 token 版；旧 680-goal 版是 8.89M）。
- `paper` 档每卡预占 ~25 GB（141 GB 很宽松）。若 OOM：`ENVS=2048 BATCH=2048 bash scripts/launch_token_train_2xh200.sh`。
- 首次跑前 5–15 分钟慢是 **warp kernel + Triton/inductor 本地编译**，不是卡坏了；`it` 行出来之前不要动它。

---

## 5. 正式 320k（论文口径）

```bash
cd $COMIMIC
setsid nohup bash scripts/launch_token_train_2xh200.sh \
  > /tmp/train_token_h200x2.log 2>&1 < /dev/null &

sleep 60; tail -5 /tmp/train_token_h200x2.log
```

> 为什么用 `setsid nohup … < /dev/null &`：裸 `nohup … &` 挂在会被回收的 shell 会话里，会话一断整组 SIGTERM，训练会"莫名其妙消失"（本机踩过）。`setsid` 开独立会话，断开不影响。

### 5.1 launcher 自动带上的参数（已对代码核实，别手改）

| 参数 | 值 | 说明 |
|---|---|---|
| `--policy.token-dim` | **64** | token 架构开关；`0` 就是旧 680-goal 架构 |
| `SONIC_ENCODER_ONNX` | `data/assets/model_encoder.onnx` | 训练环内现算 token（B 模式，detach） |
| `SM_TF32` | `1` | `train.py:47` 读；Hopper 上 BPTT 更新快 1.3–1.8× |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | `1` | 离线闸门 |
| `MUJOCO_GL` | `disable` | 无网机器常缺 `libEGL.so.1`；不关 GL 会在 import 时崩（2×H200 实证，见 §3.1）|
| `PYTHONPATH` | 清空 | 防串环境 |
| `OMP_NUM_THREADS` | 16 | `train.py` 还会按 `LOCAL_RANK` 自动切核 |
| `--env.num-envs` | 4096 | 论文 Table A.5「每 GPU 4096」 |
| `--policy-trainer.batch-size` | 4096 | 论文 |
| `--wm-trainer.batch-size` | 2048 | 论文（每次更新 10 个 batch） |
| `--policy-trainer.horizon` | 32 | 论文 H=32 |
| `--schedule.updates-per-env-step` | 0.4 | 论文 UTD |
| `--schedule.total-iterations` | 320000 | 论文 |
| `--schedule.checkpoint-interval` | 4000 | 约 30–60 分钟一个 ckpt |
| lr | **自动 ×√2** | `loop.py:37-40`；2 卡正好是论文口径，**不要手改 lr** |
| `--env.future-frames` | 10 | 默认值（tyro 不传即 10） |

### 5.2 跑起来后应该看到的日志

```
[launch] gpus=2 (NVIDIA H200,NVIDIA H200) nproc=2 profile=paper envs/rank=4096 batch/rank=4096 wm-batch=2048 iters=320000
[launch] run=e9_bones_seed_token_h200x2
[launch] torch=2.14.1+cu126 cuda_visible=2 warp_cuda=2
[sonic] token conditioning enabled (dim=64)          ← 两个 rank 各一行
[INFO] params: world model 8.82M, policy 8.96M, critic 0.235M
it 50 | 0.01 h | 6.4 it/s | updates 4 | train_ret ... | wm nse1 ... | t sim/wm/pi ...
```

每 50 迭代一行；`t sim/wm/pi` 是这 50 次迭代里仿真/世界模型/策略的耗时分解，出问题先看这三项。

---

## 6. 续训 / 换 run 名 / 降卡

```bash
cd $COMIMIC

# 续训（迭代计数、优化器状态、lr 余弦曲线都接着走）
setsid nohup bash scripts/launch_token_train_2xh200.sh \
  --resume logs/e9_bones_seed_token_h200x2/ckpt_latest.pt \
  > /tmp/train_token_h200x2_resume.log 2>&1 < /dev/null &

# 限时跑（到点自动存 ckpt 退出，服务器不让长跑时用）
setsid nohup bash scripts/launch_token_train_2xh200.sh --schedule.max-hours 8 \
  > /tmp/train_token_h200x2_8h.log 2>&1 < /dev/null &

# 换 run 名 + 改规模（任何一项都能单独覆盖）
RUN=e9_token_h200x2_b ENVS=8192 BATCH=4096 WMB=4096 \
  bash scripts/launch_token_train_2xh200.sh

# 只剩 1 张卡时（world=1，lr 不放大；仍必须保留 token 参数）
cd $REPO && export PYTHONPATH= SM_TF32=1 SONIC_ENCODER_ONNX=$PWD/data/assets/model_encoder.onnx
OMP_NUM_THREADS=32 .venv/bin/python -m supermimic.scripts.train \
  --run-name e9_bones_seed_token_h200x1 \
  --data.library data/library/bones_seed_compact.pt --policy.token-dim 64 \
  --env.num-envs 4096 --policy-trainer.batch-size 2048 --wm-trainer.batch-size 2048 \
  --schedule.total-iterations 320000 \
  --resume logs/e9_bones_seed_token_h200x2/ckpt_latest.pt
```

> ⚠️ **旧 680-goal 的 checkpoint（如 `logs/e8_bones_seed_4090x2/`）不能 `--resume` 到 token 版**：`Policy`/`Decoder` 形状不同，会 shape mismatch。它只能当曲线/世界模型参照物。

---

## 7. 监控 / 评测 / 回放

```bash
cd $REPO && unset PYTHONPATH

# 文本曲线（不需要浏览器）
.venv/bin/python scripts/tb_tail.py --run-dir logs/e9_bones_seed_token_h200x2 --last 3
.venv/bin/python scripts/tb_tail.py --run-dir logs/e9_bones_seed_token_h200x2 --filter policy/

# 日志行
grep -E "^it " /tmp/train_token_h200x2.log | tail -5

# TensorBoard
.venv/bin/tensorboard --logdir logs/e9_bones_seed_token_h200x2/tb --port 6006 --bind_all

# ckpt
ls -lh logs/e9_bones_seed_token_h200x2/
```

**评测需要 npz 目录（97 GB，142,220 条）**，不是那个紧凑库：

```bash
.venv/bin/python scripts/eval_benchmark.py \
  --ckpt logs/e9_bones_seed_token_h200x2/ckpt_latest.pt \
  --npz-dir data/npz/bones_seed \
  --manifest data/npz/bones_seed/manifest_filtered.csv \
  --num-envs 1024
```

> `eval_benchmark.py` / `play.py` 目前**还没接 token 条件路径**（FINAL_HANDOVER §7 第 4 条列为待办）。
> 若跑出 shape 错，就是这个问题，不是 checkpoint 坏；在此之前只能用训练曲线和 `wm/nse*` 判断收敛。

---

## 8. 要跑多久（以实测为准）

**公式**：`ETA(小时) = 320000 / it_s / 3600`。

**2×H200 已实测的基准（2026-10-24，token 版 / `paper` 档）：稳态 2.3–2.7 it/s → 320k ≈ 37 h。**

| 2×H200 实测 `it/s` | 320k ETA | 说明 |
|---|---|---|
| **2.5（实测）** | **≈ 37 h** | 现状。首个 `it 50` 的 `0.1 it/s` 是编译，别看那行 |
| 4 | 22.2 h | 要更快只能动 host 开销或 batch（见 §8.2）|
| 6 | 14.8 h | ~~早期手册估算 6–8 it/s~~ → **实测不成立**，见 §8.1 |
| 8 / 10 | 11.1 h / 8.9 h | 需要先解决 §8.1 的瓶颈 |

> ⚠️ **加卡/加 batch 不减少迭代数**：`total_iterations` 是每 rank 计的，加大 batch 只会让每次迭代更重（更"吃满"但更慢）。要缩短墙钟只有降 `ITER`、`--schedule.max-hours` 分批，或解决 §8.1 的 host 瓶颈。

### 8.1 ⚠️ 为什么这么慢：实测证明瓶颈在 **host**，不在 GPU

**同一份代码，2×H200 和 2×4090 跑出一样的速度**（这是决定性证据）：

| 每 50 迭代的 `t sim/wm/pi` | **2×H200（sm_90 / 140 GB / driver 595）** | **2×4090（sm_89 / 48 GB / driver 595）** |
|---|---|---|
| `sim` | **2.3 s** | **2.3 s** ← 分毫不差 |
| `wm` | 4.0 s | 5.0 s |
| `pi` | **15.0 s** | **13.5 s** ← 反而更慢 |
| 稳态 it/s | 2.3–2.7 | 2.3–2.4 |

H200 的 TF32 算力约 4–6×、HBM 带宽 4.8 vs 1.0 TB/s，**却完全没换来速度**（`sim` 一模一样，`pi` 还慢 1.5 s）→ 瓶颈不在算力。

**每次迭代 ≈ 0.40–0.43 s 的构成**：

| 环节 | 耗时 | 占比 | 性质 |
|---|---|---|---|
| 物理仿真（4096 env 一步） | ~46 ms | 11% | mjlab **已经开着 CUDA graph**（`sim.py:_should_use_cuda_graph`，驱动 ≥12.4 + mempool → True），基本到底了 |
| 世界模型更新（0.4×10 个 backward batch） | ~80 ms | 19% | 已是 compile 后的形态 |
| **策略想象 + critic** | **~300 ms** | **70%** | **32 步想象 × 每步几百个小 kernel** + 20 个 critic 微批 → 纯发射开销 |

**已排除的路（别重走）**：

| 方案 | 结论 |
|---|---|
| `torch.compile` | ✅ 已开。实测 eager **3.887 s** → compiled **0.485 s**（8.0×），这条路已经吃满 |
| mjlab sim CUDA graph | ✅ 已默认开启（驱动 ≥12.4），`sim` 的 46 ms 已是极限 |
| `--policy-trainer.compile reduce-overhead`（CUDA graphs） | ❌ 不可用：`cudagraph_trees` 报 *"outputs from a previous step still require backward"*（BPTT 跨 32 步，图不能复用），退化到慢路径，**比 baseline 还慢约 4×** |
| 把整个 32 步 rollout 编成一个图 | ❌ 试过：inductor 卡在编译不动（>10 分钟无进展），放弃 |
| 换更好的 GPU（H200） | ❌ 实测无效（就是上面那张表）|

**结论一句话**：这份实现是"每迭代几万次小 kernel 发射"的形态，**单线程 host 决定上限**，所以 H200 ≈ 4090 ≈ 2.5 it/s。早期手册里"H200 6–8 it/s → 11–15 h"的估算是**乐观了 3 倍**。

#### 8.1.1 ⚠️ 别把两条 run 的 it/s 直接对比（配置不同，会看反）

2×H200 上跑的是 **token 版 + policy batch 4096**；本机 e8 是 **旧 680-goal 版 + policy batch 2048、没有 token encoder**。按"每次 update"拆开才是同口径：

| 阶段 | 2×H200（batch 4096 + 每步 encoder） | 2×4090 e8（batch 2048，无 encoder） | 读法 |
|---|---|---|---|
| `sim`（同 envs 4096） | 46 ms | 46 ms | 完全一样 → 发射/CUDA graph 限定 |
| `wm` update（同 batch 2048） | **200 ms** | **245 ms** | H200 快 **1.22×** ← GPU 优势只在算得动的地方出现 |
| `policy` update | **610 ms** | **670 ms** | H200 **多干 2 倍 batch + 32 次 encoder** 还快 9% → 单位算力约 2× |
| 每迭代 | 46+0.4×(200+610) = **370 ms → 2.7 it/s** | 46+0.4×(245+670) = **412 ms → 2.4 it/s** | **H200 其实更快** |

**结论：H200 不是更慢，是同时在干两倍的活。** 另外两边 CPU 同代（8558 vs 6530，都是 4.0 GHz turbo，H200 侧 load 反而更低），所以也不是 CPU 拖累。

**这里还藏着一个关键实测**：policy batch 2048 → 4096（×2）时，单次 update 是 670 → 610 ms，**基本没变贵**。这就是"发射开销固定"的直接证据，也说明 §8.2 的 B 方案（加 batch + 减 ITER）大概率是赚的。

### 8.2 想更快：三条路（按性价比排）

**A. 什么都不改 —— 37 h。** 论文口径的代价；ckpt 每 4000 迭代，随时可续。

**B. 把 H200 的算力变成速度：每迭代喂更多数据。**
因为 host-bound，batch 翻倍时单迭代耗时涨得远小于 2 倍。先花 10 分钟实测（**必须先停掉正式跑**，否则显存守卫会拦住；`FORCE=1` 会让测量失真）：

```bash
# 停掉正式跑
pgrep -af "supermimic[.]scripts[.]train"      # 拿 PID
kill <PID>

# 实测 high 档（每卡 ENVS 8192 / policy batch 8192 / WM batch 4096）
ITER=200 PROFILE=high RUN=bench_high bash scripts/launch_token_train_2xh200.sh 2>&1 | tee /tmp/bench_high.log
```

**判据**：`it/s` 若只掉到 **1.8–2.0**（而不是 1.2），则"batch 翻倍 + **ITER 减半到 160k**"能在**相同总样本量**（160k×2×8192 = 320k×2×4096）下把墙钟压到 **≈ 25 h**。
代价：梯度更新次数减半（64k vs 128k），**偏离论文口径**，结论里必须注明。

**C. 打 critic-compile 补丁：+15–20%（37 h → ≈ 31 h）。**
critic 每次 policy update 走 20 个微批 + 20 次 all-reduce，是 `pi` 里最碎的一段。纯提速、不改数学，但需重启一次（有 ckpt，可续）。

**不建议**：`PROFILE=max`（更满但更慢）、`reduce-overhead`、`--policy-trainer.compile None`（eager 慢 5×）。

> 还想试「2 进程/卡把发射间隙填满」？**已实测排除**，原因（NCCL 不允许同卡多 rank + 容器 `/proc` 卡数不一致的坑）见 §8.4，别再花时间。

**B 的决策表**（先跑完 §8.2 的 200 迭代实测再对照）：

| `PROFILE=high` 实测 it/s | 建议 | 320k 墙钟 |
|---|---|---|
| ≥ 2.2 | 值 → `high` + `ITER=160000`（总样本量不变）| ~20–24 h |
| 1.8 – 2.2 | 边缘 → 自己权衡（省 30–40% 但更新次数减半）| ~24–29 h |
| < 1.8 | 不值 → 保持现在这条 320k 跑完 | 37 h |

### 8.3 H200 节点的实测证据（2026-10-24，训练中采样）

```
index, utilization.gpu [%], power.draw [W], power.limit [W], memory.used [MiB]
0, 46 %, 228.92 W, 700.00 W, 24243 MiB
1, 40 %, 233.55 W, 700.00 W, 24243 MiB

CPU(s): 192   Model name: INTEL(R) XEON(R) PLATINUM 8558   (48 核/路, 2 线程/核, max 4000 MHz)
load average: 10.35    ← 192 核里的负载
```

**读法（这是 host-bound 的铁证）**：

| 观测 | 数值 | 含义 |
|---|---|---|
| GPU 利用率 | 40% / 46% | 一半以上时间在等 host 派活 |
| GPU 功耗 | 228 / 234 W **÷ 700 W = 33%** | 连功耗都上不去，不是热/功耗墙 |
| 显存 | 24.2 / 140 GB = **17%** | 完全不是显存瓶颈（`paper` 档估算 ~25 GB，吻合） |
| CPU 占用 | 192 核里只有 **~2 核**在跑（load 10，含系统噪音） | 主循环是**单线程**，其余 190 核全闲 |
| 单核型号 | Xeon 8558 @ 4.0 GHz turbo | 单核不弱，所以"比 4090 慢 1.5 s"只能用发射开销解释，不能用 CPU 差解释 |

结论：**卡在等 CPU，CPU 有 190 个核闲着**。这就是为什么换 H200 没用、加 batch 也不线性加速。

同一天在 2×4090 上采样（e8 独占跑）：`38% / 40% util，220 W / 450 W` —— 情况完全一样，说明这是实现形态问题，不是某台机器的问题。

### 8.4 ❌ 已实测排除：「2 个进程/卡（4 rank on 2 GPU）」这条常被想到的路

直觉是"一个 GPU 塞两个 Python 进程，发射间隙就能被填满"。**本机实测走不通**，两个独立原因，都留了证据：

**① `NCCL` 不允许两个 rank 共用同一张物理卡**

```
[rank2]/[rank3] torch.distributed.DistBackendError: NCCL error in: .../NCCLUtils.cpp:92,
                invalid usage (run with NCCL_DEBUG=WARN for details), NCCL version 2.29.3
```
而 `gloo` 回退又**不能对 CUDA tensor 做 all_reduce**（`allreduce_grads` 发的就是 CUDA tensor）。
（能绕的路只有 MPS / 独立进程各自训再合并，工程量远大于收益，不建议。）

**② 顺带挖到一个容器里的真坑：`/proc/driver/nvidia/gpus` 和 `nvidia-smi` 数量不一致**

```
ls /proc/driver/nvidia/gpus | wc -l   → 8      ← train.py 的 _n_gpus() 看的是这个（整机物理卡）
nvidia-smi -L | grep -c '^GPU'        → 2      ← 容器实际只给了 2 张（NVIDIA_VISIBLE_DEVICES）
```
所以在这类"整机 8 卡、容器给 2 卡"的节点上，**只要 `NPROC > 容器可见卡数`**，`train.py:23-24` 就会把
`CUDA_VISIBLE_DEVICES` 设成 `"2"`/`"3"` 这种不存在的编号 → 子进程直接
`RuntimeError: No CUDA GPUs are available`。

- 正常跑 `NPROC=2` 不受影响（rank 0/1 → `"0"`/`"1"` 合法）；
- 真要试多进程，必须**先显式 `export CUDA_VISIBLE_DEVICES=0,1`**（让 `_preset` 生效、跳过自动绑卡）；
- 但即使修好这一点，①② 里的 NCCL 限制仍然拦着 → **结论：2 进程/卡不可行**。

**这条路已经实证封死，别再花时间。** 剩下的只有 §8.2 的 B（加 batch + 减 ITER）和 C（critic 编译）。

### 8.5 「要不要训完 320k？」——曲线饱和的实测证据 + LR 陷阱

**实测（本机 2×4090，旧 680-goal 版，steady 2.3–2.4 it/s，已跑 32k 迭代）**：

| 迭代 | `train_ret` | `wm nse1` | `wm nse8` | `imag_ret` | `surv` |
|---|---|---|---|---|---|
| 50 | 0.459 | 0.166 | 0.672 | 0.111 | 0.000 |
| 500 | 0.507 | 0.0228 | 0.0477 | 0.889 | 0.324 |
| 2,000 | 0.83 (波动) | 0.0264 | 0.0509 | 0.830 | 0.746 |
| 4,000 | — | 0.0099 | 0.0251 | **2.010** | **0.994** |
| 8,000 | — | 0.0060 | 0.0098 | 2.396 | 0.998 |
| 16,000 | 0.786 | 0.0056 | 0.0113 | 2.710 | 0.999 |
| 30,000 | 0.799 | 0.0046 | 0.0105 | 2.818 | 0.999 |
| 32,150 | 0.785 | 0.0047 | 0.0082 | **2.833** | **1.000** |

**读法**：

- `surv`（想象 rollout 活满 H=32 的比例）在 **~4k 迭代就到 0.994**，之后是 0.999/1.000 → **饱和**；
- `imag_ret` 在 **~8k 迭代拿到最终值的 ~85%**，16k 之后每 1 万迭代只涨 0.01 → **进入平台**；
- `train_ret` 在 **~15–20k 迭代后 0.79–0.81 横盘**（有 ±0.02 抖动）；
- `wm nse8` 在 ~8k 之后在 0.008–0.011 之间徘徊 → **世界模型也到平台**。

**也就是说：训练侧指标在 20–30k 迭代（占 320k 的 6–9%）就基本平了，之后 290k 迭代是低学习率精修。**

### 8.5.1 🚨 但"早停"有个大坑：LR 是相对 `total_iterations` 定的

`--schedule.lr_decay_fraction = 0.7812`（= 250k/320k）、`lr_final_ratio = 0.03`：

```
ITER = 320000  →  lr 在 250,000 迭代才衰减到 3%
```

所以：

| 做法 | 结果 |
|---|---|
| 设 `ITER=320000`，跑到 30k 觉得曲线平了就停 | ❌ lr 还在接近最大值，策略**没 settle**，拿这个 ckpt 评测会明显偏弱，也**不能**和论文数字比 |
| 设 `ITER = 你真能跑完的步数`（例如 130k），让它跑满 | ✅ lr 在最后刚好退到 3%，**这个 ckpt 是"完成态"的**，可以直接当最终结果 |

**因此结论：不要"设 320k 然后早停"，而要"先定预算，再把 ITER 设成预算"**。

### 8.5.2 推荐的三段式（务实版）

| 阶段 | 做什么 | 时间 |
|---|---|---|
| ① 定档 | `ITER=200 PROFILE=high` 实测 it/s（§8.2 B）| 10 min |
| ② 出第一个"可评测" ckpt | 用定好的档跑 `ITER=30000`（lr 会在 23.4k 退到底），拿到一个**真正 settle 的** ckpt 去做 benchmark | `high` ~4.6 h / `paper` ~3.3 h |
| ③ 决定是否续 | 若 benchmark 达标 → 停；若明显不够 → 用 `--resume` 续到 130k / 160k（此时 ITER 要一起改成目标值，lr 会重新排程）| 视情况 |

**好消息：评测路径已经支持 token 版了**（`src/supermimic/eval/benchmark.py:64-93`、`scripts/play.py:29-40` 里都有 token 分支）。
> 更正：`FINAL_HANDOVER.md` §7 那条"待补 token 评测路径"已经过期，代码里已经实现。

**2026-10-24 实测**：用 H200 那条跑在 **iteration 4000** 存下的 ckpt（`logs/e9_bones_seed_token_h200x2/ckpt_latest.pt`）直接评测成功：

```bash
cd $REPO && export PYTHONPATH= MUJOCO_GL=disable SONIC_ENCODER_ONNX=$PWD/data/assets/model_encoder.onnx
.venv/bin/python scripts/eval_benchmark.py \
  --ckpt logs/e9_bones_seed_token_h200x2/ckpt_latest.pt \
  --npz-dir data/npz/bones_seed \
  --manifest data/npz/bones_seed/manifest_filtered.csv \
  --num-envs 1024                      # 全量；先小规模可以 --num-envs 64
```

小规模试跑（8 条 clip）结果：

```
8 clips | failed 0 | mean norm return 0.9156 | step-weighted 0.9159
  motion_global_root_pos  0.0097 (上限 0.01)   motion_body_pos  0.0192 (上限 0.02)
  motion_body_ori         0.0136                motion_body_lin_vel 0.0200
  motion_body_ang_vel     0.0199                action_rate_l2 -0.00001  joint_limit 0.0
```

（这 8 条是最简单的 `Idle_Left` 系列，代表"上限水平"；真实结论要跑全量或混合子集。）

### 8.5.2.1 判读口径（怎么算"够用了"）

| 指标 | 含义 | 判读 |
|---|---|---|
| `mean_norm_return` | 归一化回报（每步 reward / 理论上限，逐项相加） | **> 0.85 已相当好**；论文附录 F 的对比要同 clip 子集 |
| `failed` | 是否提前掉出（root 高度/倾角/超时） | 必须是 **0**，非 0 说明这几条没跟住 |
| `motion_body_pos` / `motion_body_ori` | 14 个被跟踪刚体的位置/朝向项 | 越接近上限越好（上限 0.02 / 0.02） |
| `motion_body_lin_vel` / `ang_vel` | 速度项（平滑度） | 同上 |
| `steps / frames` | 是否跑完整条 clip | 应等于 frames |

### 8.5.2.2 因此"要不要训完"变成可回答的问题

1. 每 4000 迭代就有一个 ckpt，**随时可以量化**（一次全量评测 ~1024 envs）；
2. 把 `4k / 20k / 60k` 三个 ckpt 各评一遍，看 `mean_norm_return` 还在不在涨：
   - 涨得动 → 继续到 130k/160k（记得 ITER 一起改，让 lr 退火刚好在终点）；
   - 20k 与 60k 差 < 0.01 → **已经到平台，可以停**，不用跑满 320k。

### 8.5.3 一句话结论

- **训练侧曲线 20–30k 迭代就平**（surv 4k 饱和、imag_ret 8k 到 85%、train_ret 15–20k 横盘）；
- 但"平了"不等于"能用了"，验收要看 **tracking benchmark**（已支持 token 版，§8.5.2）；
- **要早停就把 ITER 设成要跑的步数**，别设 320k 然后半路杀；`paper` 档先跑 30k（~3.3 h）出一个 settled ckpt 是完全合理的起点。

---

## 9. 离线排错表（没有网 = 不能现场补装）

> **先收现场再对症**：进程 / 日志 / 显存三条 → §0.3；`nvidia-smi` 连卡都列不出 → §0.4。下表是收完现场后的具体修法。

| 现象 | 原因 | 处置 |
|---|---|---|
| **起不来、`Done` 但显存 0 MiB** | `setsid &` 报 Done 是假象 / 两条训练撞端口 | §0.3 三条命令 → 大概率是 `Address already in use`，只起一条，`PORT=29511` |
| **`nvidia-smi` 列不出卡** | 容器没给 GPU / 驱动没起来 | §0.4 |
| `.venv/bin/python: No such file or directory` | 解释器软链断了 / `.venv-python` 没随盘过来 | `ln -sfn $REPO/.venv-python/bin/python3.12 $REPO/.venv/bin/python3.12`；`pyvenv.cfg` 的 `home` 也要指向 `$REPO/.venv-python/bin` |
| `base_prefix` 是 `/root/...` 或 `/usr` | `pyvenv.cfg` 的 `home` 不对 | `sed -i "s#^home = .*#home = $REPO/.venv-python/bin#" .venv/pyvenv.cfg` |
| `wp.init()` 只列 `cpu` | warp 版本/驱动不匹配 | 无网不能降级；确认 `warp-lang 1.17.0`（`.venv/bin/pip list \| grep warp`）与驱动版本。改不了就换机器 |
| `import torch` 报 `CUDA driver version is insufficient` | 驱动 < 525 | 同上，只能换驱动 |
| `import mujoco` 报 `OpenGL.raw.EGL … 'NoneType' object has no attribute 'eglQueryString'` | 机器缺 `libEGL.so.1`（**2×H200 那次起不来就是它**） | `export MUJOCO_GL=disable`（训练不需要渲染）。launcher 已内置为默认值，用脚本就不会再犯；详见 §3.1 |
| `ModuleNotFoundError: No module named 'supermimic'` | 用了系统 `torchrun`（`/usr/local/bin/torchrun` 是 `#!/usr/bin/python`） | launcher 已在 2026-10-24 修成 `exec "$SUPERMIMIC/.venv/bin/torchrun"`；手敲命令时也务必用 `.venv/bin/torchrun` |
| 两个 rank 都占 GPU0 / OOM | 手设了 `CUDA_VISIBLE_DEVICES` | 去掉；`train.py` 按 `LOCAL_RANK` 自动绑卡 |
| `NCCL … Address already in use` | 端口被占 | `PORT=29511 bash scripts/launch_token_train_2xh200.sh` |
| `FileNotFoundError: … no clips` | 库路径不对 | §2 ⑤ 校验；或 `LIBRARY=/绝对路径/bones_seed_compact.pt` |
| 首次跑很慢、利用率低 | warp + Triton/inductor 本地编译 | 正常；等 `it 50` 那行 |
| 训练进程"消失" | 裸 `nohup … &` 被会话回收 | 用 `setsid nohup … < /dev/null &` |
| 想杀进程结果把自己 shell 杀了 | `pkill -f` 匹配到自己命令行 | `pgrep -af "supermimic[.]scripts[.]train"` 看 PID，再 `kill <PID>` |
| policy 参数量是 8.89M 而不是 8.96M | 没带 `--policy.token-dim 64`（跑成旧架构了） | 用 launcher，别手敲 |
| 编译卡住 >10 分钟 | 用了 `--policy-trainer.compile reduce-overhead` | **不要用**，本机实测卡在 cudagraphs 慢路径（比 baseline 慢 4×）；保持默认 `default` |

---

## 10. 绝对禁止清单（离线机器上做了就救不回来）

1. ❌ `pip install` / `apt-get install` / `hf download` / `git clone` —— 没网，且会污染已锁定的版本。
2. ❌ 升级/降级 `torch`、`warp-lang`、`mujoco`、`mjlab` —— 版本一换就静默退化（warp 1.18 只会剩 `cpu` 设备）。
3. ❌ 手动 `export CUDA_VISIBLE_DEVICES` —— 会和 `LOCAL_RANK` 绑卡打架。
4. ❌ `--policy-trainer.compile reduce-overhead`。
5. ❌ `--wm-trainer.compile None`/`--policy-trainer.compile None` 跑正式长跑 —— eager 比 compile 慢约 5×（仅冒烟可用）。
6. ❌ 删除 `.venv/`、`.venv-python/`、`data/library/bones_seed_compact.pt`、`data/assets/model_encoder.onnx`（离线机器上无法重建）。
7. ❌ 两台机器同时往同一个 `--run-name` 写（ckpt 互相覆盖）。

---

## 11. 产物与校验值

| 产物 | 路径 | 校验 |
|---|---|---|
| 紧凑动作库 | `$REPO/data/library/bones_seed_compact.pt` | `md5 = f82de7fae8574aaeb17330dcc4e778ef`；6.28 GiB；129,943 clips / 47,475,680 帧 / 263.75 h |
| token encoder | `$REPO/data/assets/model_encoder.onnx` | 48 MiB；**4,231,232 参数**；与 PyTorch 复现逐位一致 |
| 训练产物 | `$REPO/logs/e9_bones_seed_token_h200x2/` | `tb/`、`config.json`、`ckpt_latest.pt`、`ckpt_XXXXXXX.pt`（每 4000 迭代，仅 rank0 写） |
| 启动脚本 | `$COMIMIC/scripts/launch_token_train_2xh200.sh` | 薄封装，`NPROC=2 PROFILE=paper` |
| 参数脚本 | `$COMIMIC/scripts/launch_token_train_4xh200.sh` | 唯一含 token/TF32/离线参数的地方 |
| 预检脚本 | `$COMIMIC/scripts/preflight_4xh200_offline.py` | `--expect-gpus 2` |

**已验证（2026-10-24，本机 2×4090）**

- 预检全过：encoder 4,231,232 参数、token 1/16 网格、与 ONNX 逐位一致；库 129,943/47,475,680/263.75 h。
- `bash scripts/launch_token_train_2xh200.sh`（小配置）端到端跑通：2 rank NCCL、两 rank 均 `[sonic] token conditioning enabled (dim=64)`、`params policy 8.96M`。
- token 路径**确实在更新**：eager 小配置实跑 `it 50 | updates 4`（无报错），说明 warmup 之后的世界模型/策略更新链路正常。

**未验证（H200 到手第一件事）**

- 2×H200 上的真实 `it/s`、显存峰值、`paper` 档是否 OOM（4090 48 GB 上的等价档未满配验证）。
- H200 侧驱动版本是否 ≥ 525。

---

## 12. 附录：与 4 卡 / 旧架构的关系

| 路线 | 命令 | 口径 | 备注 |
|---|---|---|---|
| **2×H200（本文，主路线）** | `bash scripts/launch_token_train_2xh200.sh` | 论文 1:1（lr ×√2、样本量 1:1） | 正式结果用这个 |
| 4×H200（可选加速） | `PROFILE=paper bash scripts/launch_token_train_4xh200.sh` | 320k 迭代但样本量 = 论文 2 倍，lr ×2 | 更快收工，但**别和论文 Table A.8 逐位比** |
| 2×4090（本机验证/降配） | `NPROC=2 PROFILE=paper ENVS=2048 BATCH=1024 bash scripts/launch_token_train_4xh200.sh` | 同上缩规模 | 已实跑；用于验证链路 |
| ~~旧 680-goal 架构~~ | ~~`--policy.token-dim` 不传~~ | — | **已废弃**，checkpoint 与 token 版不兼容 |

其他文档：[`FINAL_HANDOVER.md`](FINAL_HANDOVER.md)（架构唯一权威）、[`BONES_SEED_HANDOVER.md`](BONES_SEED_HANDOVER.md)（数据管线与本机历史）、[`BONES_SEED_H200X4.md`](BONES_SEED_H200X4.md)（4 卡附带条件与离线环境细节）、[`BONES_SEED_H200X2.md`](BONES_SEED_H200X2.md)（环境安装/排错长文）。
