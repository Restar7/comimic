# BONES-SEED / SuperMimic 环境交接文档

> 交接时间：2026-10-24 00:40（本机时间）
> 一句话现状：**这台机器没有 2×H200（实测 1×RTX 4090 48 GB / 驱动 570.124.06），但 SuperMimic 整套训练环境已经在仓库内装好并验证可用**；BONES-SEED 的 `g1.tar.gz`（23.5 GB）已完整下载，解压到 **8,776 / 142,220** 条时被叫停。接手的人从"续解压"往下跑即可。
> **最新架构与全部决策请看 [`FINAL_HANDOVER.md`](FINAL_HANDOVER.md)**（锁定版）；本文保留为早期环境交接记录。
> 配套手册：`docs/BONES_SEED_H200X2.md`（2×H200 目标机器规格 + 本机单卡版命令 + 排错表）

---

## 0. 现状一览（可直接接手）

| 组件 | 位置 | 状态 | 校验方式 |
|---|---|---|---|
| GPU | 本机 | 1×RTX 4090 48 GB（49140 MiB，sm_89），驱动 570.124.06 / CUDA 12.8，无 slurm | `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv` |
| venv | `third_party/SuperMimic-Reproduced/.venv`（11 GB） | ✅ 装好、已跑通 | 见 §6「已验证」 |
| 源码 | `third_party/SuperMimic-Reproduced` | ✅ 可编辑安装（`pip install -e . --no-deps`） | `python -c "import supermimic"` |
| 数据 tar | `../datasets/bones-seed/v1/g1.tar.gz` | ✅ **完整**：23,499,973,647 B | `stat -c%s` 对比上面字节数 |
| 数据解压 | `../datasets/bones-seed/v1/g1/csv/{date}/*.csv` | ✅ **142,220 / 142,220** 条 | `find ... -name '*.csv' \| wc -l` |
| BONES-SEED 管线 | `data/npz/bones_seed/` + `data/library/bones_seed_compact.pt` | ✅ 142,220 npz（97 GB）→ 保留 129,943 → 紧凑库 47,475,680 帧 / 263.75 h / 6.75 GB | `torch.load(...)["lengths"]` |
| 下载脚本 | `scripts/fetch_bones_seed.sh` | ✅ 可用（自检 403 → 断点续传 → 解压 → 校验） | `bash -n` 通过 |
| VLA→SuperMimic 接口 | `diff_vla_wbc/goal_head.py` | ✅ VLA 直接输出 10×68 物理 goal（不经 token/adapter），手部 14 维独立头；38 个测试全过 | `pytest tests/test_hand_stream.py tests/test_vla_hand_head.py tests/test_goal_head.py -q` |
| SONIC motion token | `diff_vla_wbc/sonic_encoder.py` | ✅ 冻结 encoder 纯 PyTorch 复现，与部署 ONNX **逐位一致**；12 个测试 | `pytest tests/test_sonic_encoder.py -q` |
| SONIC 微调数据 | `data/sonic_datasets.json` + `diff_vla_wbc/sonic_dataset.py` | ✅ 已锁定 `groot_sonic_vla_ft318_frozen_20261023`：**4 类任务 / 9 子集 / 318 episode / 473,876 帧 / 2.633 h**，含 64 维 token 与手部 7+7 | `pytest tests/test_sonic_dataset.py -q`（11 passed） |
| 手册 | `docs/BONES_SEED_H200X2.md` | ✅ 已按本机现实重写 | — |

绝对路径（本机）：

```bash
REPO=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/third_party/SuperMimic-Reproduced
DATA=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/datasets/bones-seed/v1
```

> 注意：`/inspire/dataset/bones-seed/v1` 是**空的壳**（261 KB，0 条 CSV），别再用它。真正数据在 wsc-workspace 下的 `datasets/bones-seed/v1`。

---

## 1. 这台机器上"能不能跑"的结论

- ✅ **能跑**：单卡 1×4090 48 GB，`torch 2.14.1+cu126 + mjlab 1.6.0 + mujoco-warp 3.16 + warp 1.17.0` 全部跑通，测试 `7 passed / 17 skipped`，官方 CSV 试转 20 条 **0 失败**。
- ❌ **不能跑 2 卡**：物理上只有一张卡，`torchrun --nproc_per_node 2` 无意义。
- ⚠️ **不能照抄 1.18 warp / cu130 torch**：见 §2 的三根钉子，照抄会"装成功但静默跑 CPU / import 就报驱动不足"。

---

## 2. 三根钉子（本机专属，务必别改）

| 组件 | 本机必须 | 原因 | 装错后的症状 |
|---|---|---|---|
| torch | **2.14.1+cu126** | 驱动 570 = CUDA 12.8；cu130 轮子要求驱动 ≥ 580；`cu128` 源最高只有 2.11.0（不满足 mjlab 的 `torch>=2.14`） | `import torch` 报 `CUDA driver version is insufficient` |
| warp-lang | **1.17.0**（不是 1.18） | warp 1.18 用 CUDA 13.4 runtime，要求驱动 ≥ 13.0 | **不报错**，但 `wp.init()` 设备列表只有 `cpu` → mujoco-warp 静默退化，训练慢到不可用 |
| pyarrow | 从 **pypi.org** 装 | 内网 Nexus 镜像拉 47 MB 的 pyarrow 轮子会 15 s 超时重试 5 次后失败 | `error: connection-timeout ... nexus.sii.shaipower.online` |

关键自检（**必须看到 `cuda:0`**，只看 import 成功不算数）：

```bash
cd $REPO && export PYTHONPATH=
.venv/bin/python -c "import warp as wp; wp.init(); print(wp.get_devices())"
# 期望：['cpu', 'cuda:0']，且打印 (47 GiB, sm_89, NVIDIA GeForce RTX 4090)
```

---

## 3. 从零重建环境（约 15 分钟，只有这一步需要联网）

```bash
REPO=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/third_party/SuperMimic-Reproduced
cd $REPO
unset PIP_CONSTRAINT            # 本机没有 /etc/pip/constraint.txt，保险起见
export PYTHONPATH=

python3.12 -m venv .venv        # 系统 python3.12 = /usr/bin/python3.12 (3.12.13)
. .venv/bin/activate
pip install -U pip

# ① torch 锁 cu126。
#    坑：本机从 download.pytorch.org 单连接只有 0.3 MB/s（869 MB 要 ~50 分钟）。
#    推荐用 aria2c 多线程下轮子，再本地装（实测 20 MB/s）：
mkdir -p /tmp/whl && cd /tmp/whl
aria2c -x16 -s16 -k8M -c --file-allocation=none \
  -o "torch-2.14.1+cu126-cp312-cp312-manylinux_2_28_x86_64.whl" \
  "https://download.pytorch.org/whl/cu126/torch-2.14.1%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl"
cd $REPO && .venv/bin/pip install /tmp/whl/torch-2.14.1+cu126-cp312-cp312-manylinux_2_28_x86_64.whl

# ② mjlab（钉 commit），会把 warp-lang 拉成 1.18 —— 下一步降回去
.venv/bin/pip install "mjlab @ git+https://github.com/mujocolab/mjlab@8bb2bc9e187797689ec82c78c35019afd2fd0376"

# ③ warp 降到 1.17（本机驱动 12.8 的上限），mjlab 要求 >=1.15，1.17 满足
.venv/bin/pip install "warp-lang==1.17.0"

# ④ supermimic 本体（--no-deps，避免它的 pyproject 钉版打架）
.venv/bin/pip install -e . --no-deps

# ⑤ 三个 mjlab 不带的包；pyarrow 走 pypi.org 绕开 Nexus
.venv/bin/pip install --index-url https://pypi.org/simple pandas pyarrow huggingface-hub
.venv/bin/pip install pytest

# ⑥ 自检（三条都要过）
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
.venv/bin/python -c "import warp as wp; wp.init(); print(wp.get_devices())"
.venv/bin/python -c "from supermimic.data.motion_convert import load_g1_model; m=load_g1_model(); print('nq', m.nq, 'njnt', m.njnt)"
```

装完的实际版本（2026-10-24 实测，可直接当基线）：

```text
Python 3.12.13 | torch 2.14.1+cu126 | mujoco 3.16.0 | mujoco-warp 3.16.0
warp-lang 1.17.0 | mjlab 1.6.0 | rsl-rl-lib 5.5.1 | numpy 2.5.4
pandas 3.1.0 | pyarrow 26.0.0 | scipy 1.18.1 | tyro 1.0.16 | tensorboard 2.21.0
```

> pip 默认走内网 Nexus（`/root/.config/pip/pip.conf` → `http://nexus.sii.shaipower.online/repository/pypi/simple/`）。小包走它很快（实测 270 MB/s），大包偶尔超时，超时就加 `--index-url https://pypi.org/simple`。

---

## 4. 数据

### 4.1 已就位的文件

```bash
DATA=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/datasets/bones-seed/v1
ls -la $DATA
# g1.tar.gz  23,499,973,647 B（已完整，和 HF 上的 size 一致）
# g1/csv/{date}/{motion}.csv  —— 当前 8,776 条，解压未完成
```

### 4.2 续解压（接手第一件事，约 5–10 分钟）

```bash
cd $DATA && tar -xzf g1.tar.gz
find $DATA/g1 -name '*.csv' | wc -l      # 期望 142,220
du -sh $DATA/g1                           # 预计 30–40 GB
rm -f g1.tar.gz                           # 确认无误后再删，省 23.5 GB
```

> 坑：不要用 `tar -xzf g1.tar.gz <单个成员>` 去"偷看"——指定的成员不在开头时，tar 会把整个 23.5 GB 扫一遍，看起来像卡死。**要看格式直接解压完看。**

### 4.3 官方 CSV 格式（已实测）

```text
36 列，带表头：Frame, root_translateX/Y/Z（**厘米**）, root_rotateX/Y/Z（**欧拉 xyz 度**）,
之后 29 个关节列，命名 `<joint>_joint_dof`（步长 120 fps）
```

SuperMimic 的 `supermimic/data/motion_convert.py:160-183` 按这个格式解析（厘米→米、欧拉→四元数、度→弧度），**官方文件能直接吃**：实测转 20 条 `20/20 done, 0 failures`。

### 4.4 数据源的坑（重要）

- **HF 是 gated**：`bones-studio/seed` 为 `gated: auto`，需要①在数据集页点 "Agree and access repository" 填表，②token 具备 `canReadGatedRepos`。本机 `HF_HOME=/inspire/hdd/.../cache/huggingface`，里面原 token 属于 `MfanLiu`，`canReadGatedRepos=False` → 403；后来用的 token 属于 `xingchen07`，`canReadGatedRepos=True` → 200。**token 别外传，也别写进仓库。**
- **oli 目录里那 142,220 条 CSV 不能用**：`oli/workspace/soma-retargeter/assets/motions/bones-seed-export/bvh/{date}/*.csv` 确实有 142,220 个文件，但列名是 `left_hip_pitch_joint`（没有 `_joint_dof` 后缀）、还多出 `head_yaw_joint` / `head_pitch_joint` 两列（38 列），跟解析器对不上，硬用会直接 `ValueError: joint columns differ from G1 order`。
- `oli/.smpl_download/bones_seed_smpl/` 那 31 GB 分卷是 **SMPL 过滤版**（顶层 `smpl_filtered/`，且是**未压缩的 tar**，`tar -tf` 而不是 `-tzf`），跟 G1 CSV 无关。

### 4.5 下载命令（换机器/重下时用）

```bash
export HF_TOKEN=hf_xxx        # 需要 canReadGatedRepos=true
bash $REPO/../scripts/fetch_bones_seed.sh       # 或 comimic/scripts/fetch_bones_seed.sh
```

实测下载速度：`hf download`（xet 多连接）约 12–70 MB/s，23.5 GB 约 **8–30 分钟**；若只有 0.5 MB/s 说明退化成单连接，改用 aria2c `-x16`（见 oli 的 `docs/DOWNLOAD.md`）。

---

## 5. 数据管线（三步，接着跑）

```bash
cd $REPO && export PYTHONPATH= && . .venv/bin/activate
DATA=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/datasets/bones-seed/v1

# (1) 142,220 条 120 fps CSV -> 50 Hz npz（实测 3.6 clips/s @4 workers → 64 workers 约 40–70 分钟）
python scripts/data/csv_to_npz_batch.py \
  --input-dir $DATA --pattern '**/g1/**/*.csv' \
  --output-dir data/npz/bones_seed \
  --input-fps 120 --output-fps 50 --workers 64

# (2) 名字 + 可行性过滤（预期 129,943 条 / 263.75 h；论文 129,514 / 264.56）
python scripts/data/filter_bones_seed.py \
  --npz-dir data/npz/bones_seed --workers 64 \
  --out data/npz/bones_seed/manifest_filtered.csv

# (3) 紧凑 fp16 动作库（预期 47.6M 帧 / 263.75 h / 6.74 GB）
python scripts/data/build_library.py \
  --npz-dir data/npz/bones_seed \
  --manifest data/npz/bones_seed/manifest_filtered.csv \
  --out data/library/bones_seed_compact.pt --workers 32
```

实测参考（20 条试点）：npz 平均 **0.34 MB/条** → 全量约 **48 GB**；产物 key 为 `joint_pos (T,29)`、`body_pos_w (T,30,3)` 等，与 mjlab `MotionLoader` 对齐。

磁盘预算：tar 23.5 GB + 解压 30–40 GB + npz ~48 GB + 紧凑库 6.7 GB ≈ **110 GB**（当前 `/inspire` 剩 1.3 TB，够）。

---

## 6. 训练（单卡 1×4090）

```bash
cd $REPO && export PYTHONPATH= && . .venv/bin/activate

# 冒烟：300 迭代，~5–10 分钟，验证全链路
OMP_NUM_THREADS=32 python -m supermimic.scripts.train \
  --run-name smoke_bones_seed_4090 \
  --data.library data/library/bones_seed_compact.pt \
  --env.num-envs 512 --policy-trainer.batch-size 512 \
  --schedule.total-iterations 300 --schedule.checkpoint-interval 100

# 正式：320k 迭代（单卡 batch 2048，48 GB 显存宽松）
OMP_NUM_THREADS=32 python -m supermimic.scripts.train \
  --run-name e8_bones_seed_4090x1 --seed 0 \
  --data.library data/library/bones_seed_compact.pt \
  --env.num-envs 4096 --env.future-frames 10 \
  --policy-trainer.horizon 32 --policy-trainer.batch-size 2048 \
  --schedule.total-iterations 320000 \
  --schedule.updates-per-env-step 0.4 \
  --schedule.checkpoint-interval 4000
```

- 单卡用 `python -m`（world=1，lr 不放大）；**2 卡才用 torchrun**，且要驱动 ≥580 或同样锁 warp 1.17。
- 日志：`logs/<run-name>/tb`（TensorBoard）+ `logs/<run-name>/config.json`；检查点 `ckpt_latest.pt` / `ckpt_XXXXXXX.pt`（仅 rank0 写）。
- 预估：4090 约 **2.5–3.5 it/s → 320k 迭代 25–36 h**（未实测，跑起来看 TensorBoard 的 iter/s 校准）。
- 想先跑一段：加 `--schedule.max-hours 8`，之后用 `--resume logs/<run>/ckpt_latest.pt` 续。

---

## 7. 已验证 / 未验证（交接时必须说清）

**已验证（2026-10-24）**

- `import torch/mujoco/mjlab/supermimic` 全通；torch 看到 1 张卡；`mujoco 3.16` 与 `warp 1.17` 正常。
- `wp.get_devices()` → `['cpu', 'cuda:0']`（RTX 4090 47 GiB, sm_89）。
- `load_g1_model()` → `nq=36, njnt=30, nbody=31`（mjlab 资产 zoo 里的 G1 XML 可用）。
- `pytest tests -q` → **7 passed, 17 skipped**（skip 是需要数据/长跑的用例）。
- 官方 CSV → npz 试点 **20/20 成功、0 失败**，产物 shape/key 与 mjlab 期望一致。
- `g1.tar.gz` 字节数与 HF 完全一致。
- **手部 14 维独立通道**（`diff_vla_wbc/hand_stream.py`）实现并验证：`pytest tests/test_hand_stream.py -q`
  → **15 passed**；43 维 ctrl 布局、右手镜像符号、手型限位、闭环抓握收敛（<0.25 rad）、
  29-DoF 模型自动禁用都已覆盖。设计见 [`HAND_STREAM.md`](HAND_STREAM.md)。

**未验证（接手先做）**

- 全量解压（当前 8,776/142,220）→ 全量转换 → 过滤 → 紧凑库。
- 训练能否起、显存峰值、实际 it/s（`--env.num-envs 4096` 在 48 GB 上够不够）。
- benchmark / play（`scripts/eval_benchmark.py`、`scripts/play.py`）未跑过。
- 多卡路径完全没验证（本机只有 1 卡）。

---

## 8. 排错速查（本机专有，其它见手册 §6）

| 症状 | 原因 | 处置 |
|---|---|---|
| `wp.init()` 只有 `cpu` | warp ≥1.18 要驱动 ≥13.0（本机 12.8） | `.venv/bin/pip install "warp-lang==1.17.0"` 后重验 |
| `import torch` 报 CUDA driver 不足 | 装了 cu130 轮子 | 卸掉装 cu126 轮子（§3 ①） |
| pip 拉 torch 只有 0.3 MB/s | pytorch CDN 单连接限速 | aria2c `-x16` 下轮子再本地装（§3 ①） |
| pyarrow 装不上、`nexus...connection-timeout` | 内网镜像大包超时 | `--index-url https://pypi.org/simple pyarrow` |
| HF 403 `not in the authorized list` | 没点 Agree 或 token 无 `canReadGatedRepos` | 点协议 + 换 token；用 `whoami-v2` 查 `canReadGatedRepos` |
| `tar -xzf ... <单个文件>` 卡几分钟 | 指定成员在归档尾部，tar 扫全档 | 直接全量解压 |
| `ValueError: joint columns differ from G1 order` | 用了 oli 那份 38 列 CSV | 用官方 `g1/csv/**`（36 列、`_joint_dof`） |
| 转换报 `No module named pandas/pyarrow` | 少了 mjlab 不带的包 | 按 §3 ⑤ 装 |

> 小提醒：`pkill -f "某个包含完整命令行的模式"` 会连自己这条 shell 一起杀掉（SIGTERM），清理进程时用 PID 或 `pgrep` 先确认。

---

## 9. 交接清单（Checklist）

- [x] venv 建好、版本钉住（§3 基线）
- [x] warp 1.17 验证 CUDA 可见
- [x] `g1.tar.gz` 完整下载（23,499,973,647 B）
- [x] 官方 CSV 格式与解析器匹配（20 条试点 0 失败）
- [x] 手部 14 维独立通道（`hand_stream`）+ demo（`scripts/hand_stream_demo.py`）
- [ ] 续解压到 142,220 条（§4.2）
- [ ] 跑数据管线三步（§5），产出 `data/library/bones_seed_compact.pt`
- [ ] 冒烟训练 300 迭代（§6）
- [ ] 正式 320k 迭代 / 或先 `--max-hours 8`
- [ ] 全量 benchmark（`scripts/eval_benchmark.py`）
- [ ] 若要 2×H200：按 `docs/BONES_SEED_H200X2.md` §4.1，换机器后先跑该文 §8 的 8 条核查

**别删的东西**：`.venv/`（11 GB，重装要联网 15 分钟）、`g1.tar.gz`（在确认 142,220 条解压完成前）、`data/npz/bones_seed/`（重跑要 1 小时）。

---

## 10. 参考

- **🎯 主路线：训练手册（2×H200 · 全程离线 · token 架构）**：[`BONES_SEED_H200X2_OFFLINE.md`](BONES_SEED_H200X2_OFFLINE.md) —— 一条 `bash scripts/launch_token_train_2xh200.sh` 即正式跑
- 架构唯一权威（token 条件化 / 手部独立通道 / 代码改动清单）：[`FINAL_HANDOVER.md`](FINAL_HANDOVER.md)
- 训练手册（4×H200 加速 + 离线环境细节）：[`BONES_SEED_H200X4.md`](BONES_SEED_H200X4.md)
- 训练手册（2×H200 + 单卡双版本、排错表、核查清单）：[`BONES_SEED_H200X2.md`](BONES_SEED_H200X2.md)
- 手部 14 维独立通道（设计 + 用法 + 与 SONIC 的对应关系）：[`HAND_STREAM.md`](HAND_STREAM.md)
- VLA→SuperMimic 的 goal 直连接口（为什么放弃 token adapter）：[`GOAL_INTERFACE.md`](GOAL_INTERFACE.md)
- SONIC 数据集固定版本（ft318 frozen 2026-10-23，318 episode，含指纹与任务指令）：[`SONIC_DATASET_PIN.md`](SONIC_DATASET_PIN.md)
- SONIC motion token（encoder 复现、偏移 +1 的坑、B 模式在环接入）：[`SONIC_TOKEN.md`](SONIC_TOKEN.md)
- 工作区接入框架：[`FRAMEWORK.md`](FRAMEWORK.md)
- 上游复现仓库：`third_party/SuperMimic-Reproduced/README.md`、`docs/EXPERIMENTS.md`
- BONES-SEED 数据集页（gated）：<https://huggingface.co/datasets/bones-studio/seed>
- 下载提速与 gating 说明（oli 项目的实测记录）：`wsc-workspace/oli/limx_oli_soma_target/docs/DOWNLOAD.md`
- SuperMimic 论文：<https://arxiv.org/abs/2610.28480>
