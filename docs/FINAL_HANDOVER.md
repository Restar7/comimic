# 最终交接文档（架构锁定版）

> 生成时间：2026-10-24
> 适用工作区：`/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic`
> 本文是**唯一权威**的交接记录：锁定版本、锁定数据、锁定架构、改过的每一处代码、以及**所有被排除的方案及原因**。
> 配套细节文档：[`BONES_SEED_H200X2.md`](BONES_SEED_H200X2.md)（环境/排错）、[`SONIC_TOKEN.md`](SONIC_TOKEN.md)（token 复现）、[`SONIC_DATASET_PIN.md`](SONIC_DATASET_PIN.md)（数据固定）、[`HAND_STREAM.md`](HAND_STREAM.md)（手部通道）、[`GOAL_INTERFACE.md`](GOAL_INTERFACE.md)（被 token 路线取代的 goal 直连，保留为 fallback）。

---

## 0. TL;DR：最终架构

```text
                         ┌──────── VLA（GR00T N1.7，已微调；max_action_dim=132，horizon=40）────────┐
图像 + 语言 ────────────► │ backbone + action head ─► motion_token(64) + 左右手(7+7)，共 78 维有效 │
                         └──────────────────────────────────────────────────────────────────────────┘
                                      │ token（唯一进入 SuperMimic 的信号）        │ hand（直接下发）
                                      ▼                                            ▼
        ┌──────────────── SuperMimic（WBC + 世界模型，token 条件化）─────────┐   ┌──────────────┐
        │ Policy.decoder(token 64 + obs 930) ─► 29 维身体动作              │   │ HandStream   │
        │ Critic(feat 71 + GT goal 680 + xi 33 + e_root 4)（仅训练用）      │   │ PD → 14 手关节│
        │ 世界模型 71→70（goal-free）／解析奖励（GT 参考刚体）               │   └──────────────┘
        └──────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                          43 维执行：ctrl[0:22]+[29:36] 身体，ctrl[22:29]+[36:43] 手部
```

- **token 是 VLA 直接输出的**，SuperMimic 不再有自己的 goal encoder；
- **token 在训练环内由冻结 SONIC encoder 计算**（B 模式：参考未来 10 帧 + 仿真机器人 root 朝向），detach；
- 手部 14 维**不进入 token、不进入 SuperMimic**，由 VLA 的 hand head 直出、HandStream 直接执行；
- 奖励/世界模型/critic 保持原样（世界模型 goal-free，奖励用 GT 参考刚体）。

---

## 1. 锁定的环境（改一次就崩，别动）

| 项 | 锁定值 | 为什么（实测） |
|---|---|---|
| 硬件 | 1×RTX 4090 48 GB（sm_89，49140 MiB），驱动 570.124.06 / CUDA 12.8 | 只有一张卡；无 H200、无 slurm |
| Python | 3.12.13（`/usr/bin/python3.12`），venv 在 `third_party/SuperMimic-Reproduced/.venv`（11 GB） | — |
| **torch** | **2.14.1+cu126** | 驱动 570 不支持 cu130 轮子（需 ≥580）；cu128 源最高只到 2.11（不满足 `>=2.14`） |
| **warp-lang** | **1.17.0** | warp 1.18 runtime 是 CUDA 13.4，要求驱动 ≥13.0；1.18 下 `wp.get_devices()` 只剩 `cpu`（不报错！） |
| mujoco / mujoco-warp | 3.16.0 / 3.16.0 | mjlab 1.6 要求 `~=3.16` |
| mjlab | 1.6.0（commit `8bb2bc9`） | 上游复现口径 |
| 其它 | numpy 2.5.4、pandas 3.1.0、pyarrow 26.0.0、scipy 1.18.1、tyro 1.0.16、tensorboard 2.21.0、rsl-rl-lib 5.5.1 | — |
| 装包注意 | torch 轮子用 aria2c 下（pip 单连接仅 0.3 MB/s）；**pyarrow 必须走 pypi.org**（内网 nexus 会超时）；`pip install -e . --no-deps` | 见 `BONES_SEED_H200X2.md` §2.3 |

自检（必须看到 `cuda:0`）：

```bash
cd third_party/SuperMimic-Reproduced && export PYTHONPATH=
.venv/bin/python -c "import warp as wp; wp.init(); print(wp.get_devices())"
```

---

## 2. 锁定的数据

### 2.1 BONES-SEED（预训练）

| 项 | 值 |
|---|---|
| 原始 | `/inspire/hdd/.../wsc-workspace/datasets/bones-seed/v1/g1/csv/**`，**142,220** 条 CSV（`g1.tar.gz` 23,499,973,647 B，已解压） |
| npz（50 Hz） | `third_party/SuperMimic-Reproduced/data/npz/bones_seed/`，**142,220 个（97 GB）** |
| 过滤 | `manifest_filtered.csv`：保留 **129,943** 条（论文 129,514） |
| 紧凑库 | `data/library/bones_seed_compact.pt`，**129,943 clips / 47,475,680 帧 / 263.75 h / 6.75 GB**，`ref` 为 `(47475680, 71)` fp16 |
| 关键点 | 官方 CSV 是 36 列、`*_joint_dof` 命名、厘米/欧拉/度、120 fps —— 与解析器一致（20 条试点 0 失败） |

### 2.2 SONIC 四类任务（微调）

| 项 | 值 |
|---|---|
| 根 | `/inspire/hdd/.../flowbody/FlowBody/dataset/groot_sonic_vla_ft318_frozen_20261023` |
| 规模 | **4 任务 / 9 个 LeRobot 子集 / 318 episode / 473,876 帧 / 2.633 h @50fps** |
| 任务拆分 | task_1 **89**（5+84）、task_2 70、task_3 65、task_4 94 |
| 字段 | `action.motion_token[64]`（1/16 网格）、`action.wbc[43]`、左右手 `[7]+[7]`、`observation.state[43]`、任务指令在 `manifest.json` |
| 过滤凭证 | 每个子集有 `review_receipt.json`（例：task1 的 `old_failed_indices` 38 条被剔除，122→84） |
| 固定方式 | `data/sonic_datasets.json`（路径/条数/帧数/指令/sha256 指纹）+ `diff_vla_wbc/sonic_dataset.py` + `scripts/check_sonic_dataset.py` |

**锁定校验命令**：`python scripts/check_sonic_dataset.py --fingerprints` → 9 个子集全 `[ok]`，`318 episodes / 473876 frames / 2.633 h`。

---

## 3. 最终模型架构与接口

### 3.1 SONIC motion token（冻结 encoder）

| 项 | 内容 |
|---|---|
| 来源 | `GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx` 的 **g1 分支** |
| 输入 | 10 帧 × `[dof_pos(29), dof_vel(29), anchor_6d(6)]`（IsaacLab 关节顺序，逐帧拼接 = 640） |
| 网络 | 640→2048→1024→512→512→64（SiLU），**4,231,232 参数** |
| FSQ | `token = round(tanh(h + 0.032237395) × 15.515501 − 0.5) / 16` |
| **偏移（大坑）** | ONNX 先做 `obs[1:]` 再切片 → 真实偏移 = 图内常量 **+1**：command `[4:584)`、anchor `[601:661)` |
| 复现 | `diff_vla_wbc/sonic_encoder.py`（纯 PyTorch，GPU 可批跑）+ `tests/test_sonic_encoder.py`（12 passed，与 onnxruntime **逐位相等**） |
| 环内 | `diff_vla_wbc/sonic_token.py`（B 模式）+ `supermimic/envs/mdp/sonic_tokens.py`（训练树内实现） |
| anchor 语义 | `R_robot^T R_ref` 前两列 6D（body 帧）；**必须用仿真机器人 root 朝向**，不能用 identity |

### 3.2 SuperMimic（token 条件化）

| 组件 | 改动 |
|---|---|
| `Policy` | `PolicyCfg.token_dim`（0=原 680 goal，64=token）；`forward(obs, x, interface="token")` 时 token 直接作 latent |
| `Decoder` | 输入 `latent_dim + 930`；token 模式 latent=64 |
| `rollout.imagine` | 新增 `token_fn`：每步 `token_fn(frame, ref, s)` → `policy(obs, token, interface="token")`；`ImagRollout.tokens` 记录 |
| `PolicyTrainer` | 透传 `token_fn` |
| `training/loop.py` | `cfg.policy.token_dim > 0` 时构建 `SonicTokenBuilder` |
| critic / 奖励 / 世界模型 | **未改**（critic 训练时吃 GT 680 goal；WM goal-free；奖励解析式用 GT 参考刚体） |

### 3.3 手部（独立通道）

| 项 | 内容 |
|---|---|
| 模型 | `reproduce_scenes/g1/g1_29dof_with_hand.xml`：`nq=50 / nu=43`（29 身体 + 14 手） |
| ctrl 布局 | `0:22` 左腿+腰+左臂、**`22:29` 左手**、`29:36` 右臂、**`36:43` 右手**（插在中间，**不能 concat**） |
| 手部代码 | `diff_vla_wbc/hand_stream.py`（`HAND_POSES`/PD/`assemble_ctrl`/`G1BodyIndex`）+ `hand_head.py`（VLA 14 维头） |
| 右手 | 几何镜像，`right = -left`（限位反号） |
| 数据标签 | ft318 的 `teleop.left/right_hand_joints`（7+7） |
| 验证 | `tests/test_hand_stream.py` 15 passed、`test_vla_hand_head.py` 10 passed；demo 末态误差 ≤0.005 rad |

### 3.4 VLA 输出

**VLA 已经直接输出 motion token，不需要新加 head**（证据：`groot_repro/repro_infer.py` 部署契约
"Action contract: motion_token(64) + left_hand_joints(7) + right_hand_joints(7), horizon 40"；
数据侧 `meta/modality.json` 的 action 段同样把 `action.motion_token`(0:64) 与左右手(各 7)列为动作键；
基座 checkpoint config：`max_action_dim=132 / max_state_dim=132 / action_horizon=40`）。

| 输出 | 维度 | 标签来源 | 状态 |
|---|---|---|---|
| `motion_token` | 64 | ft318 `action.motion_token` | ✅ **已存在**（GR00T action head 直接输出） |
| `left/right_hand_joints` | 7+7 | ft318 `teleop.*_hand_joints` | ✅ **已存在**，由 `HandStream` 直接执行 |
| goal（fallback） | 10×68 | `MotionLibrary.goal_window` | ✅ 可选（`goal_head.py`），token 路线下不用 |

也就是说：**VLA 侧不需要再训练一个新头**；需要做的是"消费"——把 `motion_token` 作为 SuperMimic 的条件
（本次已打通），把 14 维手部交给 `HandStream`（已就绪）。

---

## 4. 代码改动清单（全部已落地）

### 4.1 SuperMimic-Reproduced（上游树）

| 文件 | 改动 |
|---|---|
| `src/supermimic/policy/nets.py` | `PolicyCfg.token_dim`；`Decoder(latent_dim=)`；`Policy` 支持 `interface="token"`（token 直接作 latent，无学习式 goal encoder） |
| `src/supermimic/imagination/rollout.py` | `imagine(..., token_fn=None)`；token 模式调用 `policy(..., interface="token")`；`ImagRollout.tokens` |
| `src/supermimic/imagination/policy_trainer.py` | 透传 `token_fn` 给 `imagine` |
| `src/supermimic/training/loop.py` | `token_dim > 0` 时构建 `SonicTokenBuilder` |
| `src/supermimic/envs/mdp/sonic_tokens.py`（新） | 上游树内的 encoder 抽权重 + `SonicTokenBuilder`（B 模式，detach） |

### 4.2 comimic 工作区

| 文件 | 作用 |
|---|---|
| `diff_vla_wbc/sonic_encoder.py`（新） | 冻结 encoder 纯 PyTorch 复现 + `build_motion_windows`/`rot6d`/`tokens_from_motion` |
| `diff_vla_wbc/sonic_token.py`（新） | `SonicTokenProvider`（B 模式，从 `MotionLibrary` + 71 维状态产 token） |
| `diff_vla_wbc/sonic_dataset.py`（新） | ft318 数据固定 resolver（`verify`/`task_counts`/`instructions`/`fingerprint_set`） |
| `diff_vla_wbc/hand_stream.py`（新） | 14 维手部通道（PD、ctr 布局、右手镜像、`G1BodyIndex`） |
| `diff_vla_wbc/hand_head.py`（新，可选） | VLA 14 维手部头 + loss；现有 GR00T 已输出手部，仅在"自建头"实验时使用 |
| `diff_vla_wbc/goal_head.py`（新，fallback） | VLA 直出 10×68 goal（含归一化/先验/flatten） |
| `diff_vla_wbc/gr00t_vlm.py` | 新增 `hand_head`/`goal_head` 参数、`backbone_features()`、`forward_goal_and_hands()`、`forward_with_hands()` |
| `data/sonic_datasets.json`（新） | SONIC 数据锁定清单（active=ft318，含指纹/任务/指令） |
| `scripts/check_sonic_dataset.py`（新） | 数据预检（条数/帧数/列/指纹，失败 exit 1） |
| `scripts/fetch_bones_seed.sh`（新） | BONES-SEED gated 下载/解压/校验脚本 |
| `scripts/hand_stream_demo.py`（新） | 手部通道 demo（无数据无 checkpoint） |
| `supermimic/eval/benchmark.py`（改） | token 条件策略的评测路径：`token_fn(cmd.frame, robot_quat)` → `policy(..., interface="token")` |
| `supermimic/scripts/play.py`（改） | `_DictPolicy` 支持 token 条件（从 env command 现算） |
| `groot_repro/supermimic_backend.py`（新） | 部署用解码器：`token(64)+930 历史 → q_target(29)`，替换 SONIC 的 `model_decoder.onnx` |
| `scripts/prepare_g1_tasks_npz.py` + `data/npz/g1_tasks_ft318` | SONIC **ft318** → SuperMimic motion npz（**318 条**，含 token/手部/指令/ego_view） |
| `tests/test_supermimic_deploy_backend.py`（新） | 部署后端 smoke（obs 930、q_target、torque），训练 venv 下 3 passed |
| `tests/test_sonic_encoder.py`（12） | 与 onnxruntime 逐位一致、FSQ 网格、偏移、6D、窗口 clamp |
| `tests/test_sonic_token.py`（9） | B 模式 provider、A/B 等价性、policy token 接口与反传 |
| `tests/test_sonic_dataset.py`（11） | 数据锁定（318/task1=89/指纹/列） |
| `tests/test_hand_stream.py`（15） | ctrl 布局、右手镜像、限位、PD 收敛 |
| `tests/test_vla_hand_head.py`（10） | 双输出、手部直连、与 adapter 不相交 |
| `tests/test_goal_head.py`（13） | goal 直连（fallback 路线） |
| 文档 | `FINAL_HANDOVER.md`(本文)、`SONIC_TOKEN.md`、`SONIC_DATASET_PIN.md`、`HAND_STREAM.md`、`GOAL_INTERFACE.md`、`BONES_SEED_H200X2.md`、`BONES_SEED_HANDOVER.md` |

**测试现状**（2026-10-24 05:45）：comimic `pytest tests -q` → **94 passed / 3 skipped**；
SuperMimic `pytest tests -q` → **7 passed / 17 skipped**；
部署后端（训练 venv）`SUPERMIMIC_DEPLOY_TEST=1 pytest tests/test_supermimic_deploy_backend.py -q` → **3 passed**。

> 注：`data/npz/lafan1` 被并行会话改成了指向 `bones_seed` 的软链（05:22），导致依赖 LAFAN1 的用例
> 找不到 `walk1_subject1`；已给这些用例加了"无 clip 则 skip"的保护（上游 tests 5 个文件），套件恢复绿色。

---

## 5. 已排除的方案与原因（重要，别再走回头路）

| 被排除的 | 原因 | 最终选择 |
|---|---|---|
| 2×H200 训练 | 本机只有 1×4090 48G，容器 `NVIDIA_VISIBLE_DEVICES` 单个 UUID | 1 卡训练（2 卡脚本保留在 `BONES_SEED_H200X2.md` §4.1，换机器再用） |
| torch cu130 | 驱动 570（12.8）不支持，`import torch` 即报错 | cu126（`2.14.1+cu126`） |
| warp-lang 1.18 | runtime CUDA 13.4，要驱动 ≥13.0；**不报错**但设备只剩 cpu | warp 1.17.0 |
| 复用 `isaaclab` conda 环境 | torch 2.7 / mujoco 3.10 / warp 1.10，无 mjlab，版本全不满足 | 独立 `.venv` |
| `pyarrow` 走内网 nexus | 47 MB 轮子 15 s 超时重试 5 次失败 | `--index-url https://pypi.org/simple` |
| `onnx2torch` 抽 encoder | 依赖的 torchvision ABI 与 torch 2.14 不匹配（`torchvision::nms` 崩） | 手工从 ONNX initializer 抽权重 |
| 按 ONNX 图内常量填偏移 | 图先 `obs[1:]` 再切片，真实偏移 **+1**；按常量填 token 全错（maxdiff 1.5） | `[4:584)` / `[601:661)`，逐位对齐 |
| 用 `action.wbc` 当 encoder 输入 | 它是 WBC decoder 的输出动作，不是参考 motion | encoder 只吃**参考 motion**（BONES-SEED 的 G1 轨迹 / 库里的未来参考） |
| A 模式（identity anchor）做正式预训练 | token 只编码参考、误差通道恒 0，与部署分布不一致 | **B 模式**：仿真机器人状态在环算 anchor；A 只做冒烟 |
| 离线给 BONES-SEED 批量产 token | B 模式下 token 在训练环现算，不需要落盘 | 环内 `SonicTokenBuilder` |
| 离散 FSQ 做端到端 | 量化不可导；但我们端到端**不经过 encoder**（VLA 直出 token） | 连续 token + STE 量化（或 decoder 适应），FSQ 只用于产标签 |
| token→goal 的学习式 adapter（680→32 反向） | token 无预训练语义（无 GR00T 权重）、64→680 映射不可辨识、监督仅来自 task loss | 先用 goal 直连（fallback），最终 **token 直连**（VLA 出 token、SuperMimic 吃 token） |
| 用 `SONIC_DATA/.incoming`（134 条）当训练集 | 过滤前原始导入，不是训练口径 | **ft318**（10/23 frozen，318 条） |
| 用 `SONIC_DATA` 的 four_task（229）/ 全部（363） | 同一批数据的未冻结版本，大小/口径不对 | ft318（task1=89 等） |
| 用 `oli/.../bones-seed-export/bvh` 那 142,220 条 CSV | 列名是 `*_joint`（不是 `*_joint_dof`）、多 2 个 head 关节（38 列），解析器会 `ValueError` | 官方 `g1.tar.gz`（36 列） |
| 用 `oli/.smpl_download/bones_seed_smpl` | 31 GB 分卷是 **SMPL 过滤版**（未压缩 tar，顶层 `smpl_filtered/`），与本管线无关 | 忽略 |
| 离线给 BONES-SEED 打 goal 标签训 goal head | 被 token 路线取代 | `goal_head.py` 保留为 fallback |
| 新做一个"VLA token head" | **VLA 已经直接输出 64 维 `motion_token`**（GR00T action head，horizon 40，见 `repro_infer.py` 部署契约），再做一个是重复造轮子 | 直接消费现有输出 |

---

## 6. 日常命令（环境 / 数据 / 测试 / 单卡训跑）

```bash
# ── 0) 统一进入方式（所有命令都从这里开始）
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/third_party/SuperMimic-Reproduced
export PYTHONPATH=
. .venv/bin/activate
```

```bash
# ── 1) 环境自检：必须看到 cuda:0
.venv/bin/python -c "import torch, warp as wp; wp.init(); print(torch.__version__, torch.cuda.device_count(), wp.get_devices())"

# ── 2) SONIC 数据锁定预检（318 条 / 4 任务 / 指纹）
cd ../.. && /opt/miniconda3/envs/isaaclab/bin/python scripts/check_sonic_dataset.py --fingerprints

# ── 3) 测试（comimic 94 passed；SuperMimic 7 passed / 17 skipped）
/opt/miniconda3/envs/isaaclab/bin/python -m pytest tests -q
cd third_party/SuperMimic-Reproduced && .venv/bin/python -m pytest tests -q

# ── 4) SONIC encoder 与部署 ONNX 逐位对齐（12 passed）
cd ../.. && /opt/miniconda3/envs/isaaclab/bin/python -m pytest tests/test_sonic_encoder.py -q

# ── 5) 手部通道 demo（无数据、无权重）
/opt/miniconda3/envs/isaaclab/bin/python scripts/hand_stream_demo.py --poses open grasp open --hold 0.6
```

```bash
# ── 6) 单卡 4090 训练（论文口径；先 20k 看曲线）
cd third_party/SuperMimic-Reproduced && export PYTHONPATH= SM_TF32=1
SONIC_ENCODER_ONNX=$PWD/data/assets/model_encoder.onnx \
nohup .venv/bin/python -u -m supermimic.scripts.train \
  --run-name e9_bones_seed_token \
  --data.library data/library/bones_seed_compact.pt \
  --policy.token-dim 64 \
  --env.num-envs 4096 --policy-trainer.batch-size 2048 \
  --schedule.total-iterations 20000 --schedule.checkpoint-interval 2000 \
  > /tmp/e9_train.log 2>&1 &

# 监控
tail -f /tmp/e9_train.log
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 2

# 续到 320k
SONIC_ENCODER_ONNX=$PWD/data/assets/model_encoder.onnx SM_TF32=1 \
.venv/bin/python -u -m supermimic.scripts.train \
  --run-name e9_bones_seed_token \
  --data.library data/library/bones_seed_compact.pt --policy.token-dim 64 \
  --env.num-envs 4096 --policy-trainer.batch-size 2048 \
  --schedule.total-iterations 320000 --schedule.checkpoint-interval 4000 \
  --resume logs/e9_bones_seed_token/ckpt_latest.pt
```

> 训练里的 token 由 `SonicTokenBuilder` 实时算（B 模式），**不需要离线批量产 token**；
> 若换机器，用 `$SONIC_ENCODER_ONNX` 指向 `model_encoder.onnx` 即可。

```bash
# ── 7) token 版策略评测（需 token checkpoint；例：20k 跑出来的）
cd third_party/SuperMimic-Reproduced && export PYTHONPATH=
.venv/bin/python scripts/eval_benchmark.py \
  --ckpt logs/e9_bones_seed_token/ckpt_latest.pt \
  --npz-dir ../../data/npz/g1_tasks_ft318 --num-envs 32

# ── 8) 可视化（token 版；无显示用 viser）
.venv/bin/python scripts/play.py --ckpt logs/e9_bones_seed_token/ckpt_latest.pt \
  --npz-dir ../../data/npz/g1_tasks_ft318 --clips <clip_name> --viewer viser

# ── 9) SONIC ft318 → SuperMimic npz（需要时重转）
cd ../.. && /opt/miniconda3/envs/isaaclab/bin/python scripts/prepare_g1_tasks_npz.py \
  --root /inspire/hdd/global_user/liumengfan-253108110079/flowbody/FlowBody/dataset/groot_sonic_vla_ft318_frozen_20261023 \
  --out-dir data/npz/g1_tasks_ft318 --overwrite
```

---

## 6.5 4×H200 离线训练（无网络，共享当前容器/挂载）

> **2026-10-24 补充：正式结果的主路线已定为 2×H200（论文口径，样本量 1:1、lr ×√2）**，
> 完整离线命令见 [`BONES_SEED_H200X2_OFFLINE.md`](BONES_SEED_H200X2_OFFLINE.md)（入口 `bash scripts/launch_token_train_2xh200.sh`）。
> 本节保留为**4 卡加速方案**（样本量 = 论文 2 倍，别和 Table A.8 逐位比）。
> 另：`launch_token_train_4xh200.sh` 的 `exec torchrun` 已修为 `exec "$SUPERMIMIC/.venv/bin/torchrun"`（系统 `/usr/local/bin/torchrun` 的 shebang 是 `/usr/bin/python`，没有 supermimic）。

### 6.5.1 前置（缺一不可）

1. **同一挂载、同一绝对路径**：`/inspire/hdd/.../wsc-workspace/comimic` 与
   `third_party/SuperMimic-Reproduced/.venv` 必须在 H200 机器上挂到**相同路径**
   （venv 里是绝对路径；离线机器不能 `pip install`，不能重建）。
   路径不同就用环境变量覆盖：`REPO_ROOT` / `SUPERMIMIC` / `LIBRARY` / `ENCODER`。
2. **资产已内置，不依赖网络**：
   - `third_party/SuperMimic-Reproduced/data/assets/model_encoder.onnx`（48 MiB，token encoder 副本）；
   - `data/library/bones_seed_compact.pt`（6.28 GiB，129,943 clips / 47,475,680 帧）；
   - `warp-lang 1.17.0`（sm_90 可用）；H200 侧驱动 ≥ 535，cu126 轮子向后兼容。
3. 离线脚本：`scripts/preflight_4xh200_offline.py`、`scripts/launch_token_train_4xh200.sh`。

### 6.5.2 第一步：离线预检（不联网）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/third_party/SuperMimic-Reproduced
.venv/bin/python ../../scripts/preflight_4xh200_offline.py --expect-gpus 4
# 检查：4×H200/sm_90、warp CUDA 4 张、encoder 4,231,232 参数且与 ONNX 逐位一致、
#       库 129,943 clips / 47,475,680 帧、离线资产齐全；任何一项失败 exit 1
```

### 6.5.3 第二步：三档 200 迭代基准（决定用哪档）

```bash
# 每档约 5–15 分钟（含 compile），跑完看 it/s 与显存峰值
PROFILE=paper ITER=200 RUN=bench_paper bash scripts/launch_token_train_4xh200.sh
PROFILE=high  ITER=200 RUN=bench_high  bash scripts/launch_token_train_4xh200.sh
PROFILE=max   ITER=200 RUN=bench_max   bash scripts/launch_token_train_4xh200.sh

# 同时开一个显存/利用率观察（另开终端）
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv -l 2
```

判定：在**不 OOM、不编译超时**的前提下选 `it/s` 最高的档位。

### 6.5.4 第三步：正式 320k 训练

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

# 论文口径（推荐做正式结果）
PROFILE=paper bash scripts/launch_token_train_4xh200.sh

# 更吃满 GPU（可接受偏离时）
PROFILE=high  bash scripts/launch_token_train_4xh200.sh

# 尽量吃满显存（只建议做吞吐基准，OOM 风险最高）
PROFILE=max   bash scripts/launch_token_train_4xh200.sh

# 自定义（覆盖档位里的任何一项）
NPROC=4 PROFILE=high ENVS=8192 BATCH=4096 WMB=4096 ITER=320000 \
RUN=e9_token_h200x4 bash scripts/launch_token_train_4xh200.sh
```

各档位等价的核心参数（脚本自动套用）：

| 档位 | 每卡 envs | policy batch | WM batch | 估算显存/卡 | 320k 预计（4×H200，估） |
|---|---|---|---|---|---|
| `paper` | 4096 | 4096 | 2048 | ~25 GB / 141 GB（18%） | **~13–20 h** |
| `high`（默认） | 8192 | 8192 | 4096 | ~55 GB（40%） | ~20–32 h |
| `max` | 16384 | 8192 | 8192 | ~90–110 GB（70%） | ~28–45 h |

> 注意方向：**加 batch/env 提高的是"利用率"，不是"完成速度"**。每次迭代的更新次数固定（UTD 0.4），
> 加 batch 只会让每次迭代更重，所以 `max` 更满但更慢；要更快完成，用 `paper` + TF32 + compile。

### 6.5.5 续训 / 监控 / 排错

```bash
# 续训（迭代计数接着走）
PROFILE=paper bash scripts/launch_token_train_4xh200.sh \
  --resume logs/e9_bones_seed_token_h200x4/ckpt_latest.pt

# 日志与曲线
tail -f logs/e9_bones_seed_token_h200x4/tb 2>/dev/null || true
tensorboard --logdir logs/e9_bones_seed_token_h200x4/tb --port 6006

# 进程 / 显存
pgrep -af "supermimic.scripts.train"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -l 2
```

| 现象 | 处理 |
|---|---|
| 第一次 policy 更新 OOM（多在 `max`） | 降档：`PROFILE=high`，或 `ENVS=8192 BATCH=4096` |
| 编译卡住 >10 分钟 | **不要用 `--policy-trainer.compile reduce-overhead`**（4090 实测卡死）；保持默认 `default` |
| `warp` 只看到 cpu | warp 不是 1.17.0，或驱动不足；预检会直接报错 |
| `FileNotFoundError: SONIC encoder ONNX` | `ENCODER=/path/model_encoder.onnx` 或设 `$SONIC_ENCODER_ONNX` |
| `torchrun: command not found` | 没激活 venv 或用错 python；用 `.venv/bin/torchrun` |
| 多卡抢同一张卡 | 不要手设 `CUDA_VISIBLE_DEVICES`，`train.py` 按 `LOCAL_RANK` 自动绑卡 |

### 6.5.6 已验证 / 离线说明

- **多卡代码路径已在 1 卡机器上用 `torchrun --nproc_per_node 2`（gloo 回退）跑通 50 迭代**：
  两个 rank 都打印 `[sonic] token conditioning enabled (dim=64)`、`updates 8`，梯度同步正常；
  真机 4×H200 走 NCCL（每 rank 独占一卡）。
- 训练路径**不访问网络**（脚本已设 `HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1`）；
  warp 首次运行用镜像里的 `nvcc` 编译 kernel 到 `~/.cache/warp/1.17.0`，只需本地 CUDA toolkit。
- TF32 默认开启（`SM_TF32=1`）：Hopper 上 policy 的 32 步 BPTT 更新预计快 1.3–1.8×（H200 上以实测为准）。
- compile 必须开：4090 实测关掉 0.5 it/s、开启 2.2 it/s（4.7×）。

---

## 7. 当前状态与下一步

**已完成**

- [x] 环境锁定（cu126 torch / warp 1.17 / mjlab 1.6），在 4090 上 `cuda:0` 可用
- [x] BONES-SEED 全量管线：142,220 CSV → 142,220 npz → 保留 129,943 → 紧凑库 47.6M 帧 / 263.75 h
- [x] SONIC 数据锁定（ft318：318 条 / 4 任务 / 指纹 / 预检脚本）
- [x] 冻结 encoder 纯 PyTorch 复现（与 ONNX 逐位一致，偏移 +1 已修正）
- [x] SuperMimic token 条件化接口（Policy/Decoder/rollout/trainer/loop 全部打通）
- [x] 手部 14 维独立通道 + VLA hand head
- [x] goal 直连（fallback）与 hand stream 的完整测试

**链路审计（2026-10-24 05:30 更新）**

| 链路 | 状态 | 说明 |
|---|---|---|
| BONES-SEED 数据/紧凑库 | ✅ | 129,943 clips / 47.6M 帧 / 263.75 h |
| SONIC ft318 锁定 | ✅ | 318 条 / 4 任务 / 指纹 |
| 冻结 token encoder | ✅ | 与 ONNX 逐位一致 |
| VLA 输出 token+手部 | ✅ | 既有 GR00T 微调（`repro_infer` 契约） |
| SuperMimic token 训练 | ✅ 接口+冒烟+单卡 20k（已停） | 4×H200 脚本就绪，待实跑 |
| **token 评测（benchmark/play）** | ✅ **本次接上** | ft318 全 318 条跑通（随机策略 318 fail / ret 0.58，路径正确） |
| **ft318 → SuperMimic npz** | ✅ **本次补齐** | `data/npz/g1_tasks_ft318` 318 条；旧 `data/npz/g1_tasks` 是 ft271 的 271 条（**别用错**） |
| **部署解码器（VLA token → 动作）** | ✅ 后端就绪 + smoke 通过 | `groot_repro/supermimic_backend.py`（obs 930/token→q_target 29/torque 均验证）；A/B 实跑未做 |
| 部署脚本路径 | ✅ 已通（shim） | `/home/wusichen/SONIC_DATA → 本地 SONIC_DATA` 软链 |
| 手部执行 | ✅ | `HandStream`；手部评测指标 ❌ |
| 端到端联合训练 | ❌ | 未做 |

**仍未闭环（明确）**

1. **部署后端的观测数值 parity**：新解码器的 930 维 obs 是按训练代码复刻的，尚未用一条训练期日志做逐元素对拍（仅形状/有限性验证过）；
2. **A/B 实跑**：同一段 VLA token，`model_decoder.onnx` vs SuperMimic 策略的任务成功率对比未做；
3. **手部指标**：指尖误差/抓握成功率未定义；
4. **端到端联合训练**：VLA → token → SuperMimic → 奖励反传未做；
5. **ft318 微调**：数据已转好（318 npz），微调本身未跑。

**当前运行状态（2026-10-24 04:30）**

- **没有训练进程在跑**：4090 上的 20k 验证跑在调参实验中被停止（最后一次记录：`it 1450`，
  `wm nse1 0.0179 / nse8 0.0414`、`imag_ret 0.858`、`surv 0.428`、~2.2 it/s，曲线健康）。
- 4090 上的调参结论（已固化）：**compile 必须开**（关 0.5 it/s vs 开 2.2 it/s，4.7×）；
  **不要用 `reduce-overhead`**（会卡在 cudagraphs）；`SM_TF32=1` 已内置。
- **4×H200 待启动**：先预检，再按 §6.5.3 跑三档 200 迭代基准，最后按 §6.5.4 起正式 320k。

**下一步（按顺序）**

1. 4×H200 到手：`preflight_4xh200_offline.py --expect-gpus 4` → 三档基准 → `PROFILE=paper`（或按基准选档）起 320k；
2. （可选）在 4090 上重跑 20k 验证跑，确认 token 版长跑曲线；
3. **端到端联合**：`VLA → token(STE) → SuperMimic decoder → 动作 → 世界模型 → 奖励 → loss` 反传到 VLA（小 batch/短 horizon 起步）；
4. **评测**：补 `eval_benchmark.py` / `play.py` 的 token 条件路径（否则 token 版 checkpoint 无法量化）；
   手部补新指标（指尖误差/抓握准确率）；
5. **ft318 微调数据转换**：把 SONIC 的 43 维动作转成 SuperMimic 的 motion npz/库，然后同配置（`--policy.token-dim 64`）微调。

**冒烟训练踩到的坑（已修）**

- env 侧 `Trainer.act` 原来直接喂 680 goal，token 模式下会 shape 崩（32+930 vs 64+930）→ 改为用 `self.command.frame` + `robot_root_quat_w` 现算 token；
- `PolicyTrainer` 的 `token_fn` 只加了参数没赋值 → 补 `self.token_fn = token_fn`；
- 回放池 warmup 至少要 ~60 env step（64 env 时 5 步不够，报 "no sampleable rows"）；
- `--wm-trainer.compile None --policy-trainer.compile None` 可关掉 torch.compile，冒烟快很多。

**部署接口细节（重要）**

- VLA 一次输出 **40 步**的 token chunk（50 Hz），SuperMimic 每步消费一个 token：
  部署时每 40 步跑一次 VLA，chunk 内逐步喂给 policy；训练时（B 模式）每步现算 token，语义一致。
- 现有 VLA 的 action 维度是 **132 维 padded**，有效键只有 `motion_token(64)` + 左右手 `(7+7)`；
  取 token 时按 processor 的 action 键切片，不要直接取前 64 维（padding 顺序由 processor 决定）。

**已知风险**

- 本机单卡 4090，320k 迭代预计 25–36 h（未实测）；
- **2026-10-24 实测补齐：2×H200（token 版 / paper 档）稳态 2.3–2.7 it/s → 320k 约 37 h，和 2×4090 的 2.3–2.4 it/s 几乎一样**。
  原因已定位：瓶颈在 **host 单线程发射开销**，不在 GPU（`sim` 分毫不变 2.3 s/50 迭代，`pi` 300 ms/迭代占 70%）；
  `torch.compile`（8×）与 mjlab 的 CUDA graph 都已开到位。分析、排错与三条提速路见
  [`BONES_SEED_H200X2_OFFLINE.md`](BONES_SEED_H200X2_OFFLINE.md) §8.1–§8.3；同时该文修正了"2×H200 6–8 it/s"的旧估算（乐观 3 倍）；
- `SonicTokenBuilder` 在训练环内每步跑一次 4.2M 参数 MLP，需在真实 batch 下测开销；
- token 版策略与旧 680-goal checkpoint 不兼容，需要重训；
- ft318 只有 2.63 h，微调要冻结 backbone、只训 head/adapter。

---

## 8. 快速索引

| 想找什么 | 去哪 |
|---|---|
| 环境安装/排错（cu126/warp 1.17/pyarrow） | `docs/BONES_SEED_H200X2.md` |
| 最早的环境交接（历史） | `docs/BONES_SEED_HANDOVER.md` |
| token 复现细节（偏移/FSQ/parity） | `docs/SONIC_TOKEN.md` |
| SONIC 数据锁定（318 条/指纹/任务指令） | `docs/SONIC_DATASET_PIN.md` + `data/sonic_datasets.json` |
| 手部通道（ctrl 布局/镜像/手型库） | `docs/HAND_STREAM.md` |
| goal 直连（被取代，fallback） | `docs/GOAL_INTERFACE.md` |
| 代码入口 | `diff_vla_wbc/sonic_encoder.py`、`sonic_token.py`、`hand_stream.py`、`hand_head.py`；`third_party/SuperMimic-Reproduced/src/supermimic/envs/mdp/sonic_tokens.py` |
| 测试 | `tests/test_sonic_encoder.py`、`test_sonic_token.py`、`test_sonic_dataset.py`、`test_hand_stream.py`、`test_vla_hand_head.py`、`test_goal_head.py` |
| 4×H200 离线训练 | `scripts/launch_token_train_4xh200.sh` + `scripts/preflight_4xh200_offline.py` |
| 论文修改方案（idea 定位/逐节改写/数据修正） | `docs/PAPER_REVISION_PLAN.md` |

---

## 9. 全部命令速查（复制即用）

> 所有命令的工作目录：`/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic`
> 单卡用 conda `isaaclab` 的 python 跑工作区脚本，训练用 `third_party/SuperMimic-Reproduced/.venv`。

### 9.1 环境与自检

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/third_party/SuperMimic-Reproduced
export PYTHONPATH=
.venv/bin/python -c "import torch, warp as wp; wp.init(); print(torch.__version__, torch.cuda.device_count(), wp.get_devices())"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
```

### 9.2 数据（BONES-SEED 下载 / 管线 / SONIC 锁定）

```bash
# BONES-SEED 下载（gated；需要 canReadGatedRepos=true 的 token）
HF_TOKEN=hf_xxx bash scripts/fetch_bones_seed.sh
# 或手动：解压 + 校验 142,220
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/datasets/bones-seed/v1
tar -xzf g1.tar.gz && find g1 -name '*.csv' | wc -l

# 数据管线三步（生成 6.28 GiB 紧凑库；已完成，重跑时用）
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/third_party/SuperMimic-Reproduced && . .venv/bin/activate
python scripts/data/csv_to_npz_batch.py --input-dir /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/datasets/bones-seed/v1 \
  --pattern '**/g1/**/*.csv' --output-dir data/npz/bones_seed --input-fps 120 --output-fps 50 --workers 64
python scripts/data/filter_bones_seed.py --npz-dir data/npz/bones_seed --workers 64 \
  --out data/npz/bones_seed/manifest_filtered.csv
python scripts/data/build_library.py --npz-dir data/npz/bones_seed \
  --manifest data/npz/bones_seed/manifest_filtered.csv \
  --out data/library/bones_seed_compact.pt --workers 32

# SONIC 微调数据锁定预检（318 条 / 4 任务 / 指纹）
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic && /opt/miniconda3/envs/isaaclab/bin/python scripts/check_sonic_dataset.py --fingerprints
```

### 9.3 测试

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
/opt/miniconda3/envs/isaaclab/bin/python -m pytest tests -q                       # 94 passed
/opt/miniconda3/envs/isaaclab/bin/python -m pytest tests/test_sonic_encoder.py -q  # 12 passed
cd third_party/SuperMimic-Reproduced && .venv/bin/python -m pytest tests -q        # 7 passed / 17 skipped
```

### 9.4 手部通道 demo

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
/opt/miniconda3/envs/isaaclab/bin/python scripts/hand_stream_demo.py --poses open grasp open --hold 0.6
```

### 9.5 单卡 4090（验证/调参用）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/third_party/SuperMimic-Reproduced
export PYTHONPATH= SM_TF32=1 SONIC_ENCODER_ONNX=$PWD/data/assets/model_encoder.onnx
nohup .venv/bin/python -u -m supermimic.scripts.train \
  --run-name e9_bones_seed_token \
  --data.library data/library/bones_seed_compact.pt \
  --policy.token-dim 64 \
  --env.num-envs 4096 --policy-trainer.batch-size 2048 \
  --schedule.total-iterations 20000 --schedule.checkpoint-interval 2000 \
  > /tmp/e9_train.log 2>&1 &

tail -f /tmp/e9_train.log
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 2
```

### 9.6 4×H200 离线（正式）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

# ① 预检（不联网）
cd third_party/SuperMimic-Reproduced
.venv/bin/python ../../scripts/preflight_4xh200_offline.py --expect-gpus 4
cd ../..

# ② 三档 200 迭代基准（看 it/s / 显存峰值）
PROFILE=paper ITER=200 RUN=bench_paper bash scripts/launch_token_train_4xh200.sh
PROFILE=high  ITER=200 RUN=bench_high  bash scripts/launch_token_train_4xh200.sh
PROFILE=max   ITER=200 RUN=bench_max   bash scripts/launch_token_train_4xh200.sh

# ③ 正式 320k（论文口径 / 高利用 / 吃满显存）
PROFILE=paper bash scripts/launch_token_train_4xh200.sh
PROFILE=high  bash scripts/launch_token_train_4xh200.sh
PROFILE=max   bash scripts/launch_token_train_4xh200.sh

# ④ 续训
PROFILE=paper bash scripts/launch_token_train_4xh200.sh \
  --resume logs/e9_bones_seed_token_h200x4/ckpt_latest.pt

# ⑤ 监控
tail -f logs/e9_bones_seed_token_h200x4/../*.log 2>/dev/null || true
tensorboard --logdir logs/e9_bones_seed_token_h200x4/tb --port 6006
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -l 2

# ⑥ 自定义覆盖（任何一项都能单独改）
NPROC=4 PROFILE=high ENVS=8192 BATCH=4096 WMB=4096 ITER=320000 \
RUN=e9_token_h200x4 bash scripts/launch_token_train_4xh200.sh
```

### 9.6b 部署后端（SuperMimic 当解码器）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/groot_repro
../third_party/SuperMimic-Reproduced/.venv/bin/python - <<'EOF'
import sys, numpy as np
sys.path.insert(0, '.')
from supermimic_backend import SuperMimicController
ctrl = SuperMimicController('../logs/<run>/ckpt_latest.pt', device='cpu')  # 换成真实 token ckpt
q = ctrl.q0_isaac[ctrl.isaac_to_mujoco].copy()
qt = ctrl.step(np.zeros(64, np.float32), q, np.zeros(29), np.array([1.0,0,0,0]), np.zeros(3))
print('q_target', qt.shape, 'finite', np.isfinite(qt).all())
EOF

# 标准化 smoke（训练 venv，3 passed）
cd .. && SUPERMIMIC_DEPLOY_TEST=1 third_party/SuperMimic-Reproduced/.venv/bin/python \
  -m pytest tests/test_supermimic_deploy_backend.py -q
```

> 部署脚本的 `/home/wusichen/SONIC_DATA` 硬编码已用软链打通：
> `/home/wusichen/SONIC_DATA -> /inspire/hdd/.../wsc-workspace/SONIC_DATA`（换机器时同样建一次即可）。

### 9.7 排错/清理

```bash
# 停训练（按 PID，别用 pkill -f 匹配自己命令行）
pgrep -af "supermimic.scripts.train"
kill <PID>

# warp 版本/设备（必须 1.17.0 且有 cuda:0..3）
.venv/bin/python -c "import warp as wp; wp.init(); print(wp.get_devices())"

# 磁盘 / 显存
df -h /inspire
nvidia-smi
```

---

## 10. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-24 | 初版：环境/数据锁定、token 复现、SuperMimic token 条件化、手部通道、4×H200 离线脚本与全部命令 |
| 2026-10-24 07:10 | **交接复核（§11）**：全量实测核实数据/环境/测试；更新 2×H200 已跑到 20,300 步的真实进度与指标；修正 §7 两条已过时的「下一步」；修 .gitignore（`g1_tasks*`）与 FRAMEWORK.md 的 ft271→ft318 |


---

## 11. 交接复核（2026-10-24 07:10 CST 实测）

> 复核方式：逐条在机上重跑，不看描述只看结果。本节结论**优先于** §7 的"当前状态"块。

### 11.1 已核实（与本文一致 ✅）

| 项 | 实测命令要点 | 结果 |
|---|---|---|
| 训练 venv | `.venv/bin/python -c ...` | python 3.12.13 / torch **2.14.1+cu126** / CUDA 可用 / mujoco 3.16.0 / mujoco_warp 3.16.0 / warp **1.17.0** / mjlab 已装 |
| 本机 GPU | `nvidia-smi` | 1×RTX 4090 48 GB（489 MiB 占用，空闲） |
| BONES 原始 | `find …/g1 -name '*.csv' \| wc -l` | **142,220**（`g1.tar.gz` 23,499,973,647 B 在） |
| BONES npz | `ls data/npz/bones_seed \| grep -c npz` | **142,220 npz / 97 GB** |
| 紧凑库 | `ls -l data/library/bones_seed_compact.pt` | **6,748,141,482 B（6.28 GiB）** |
| SONIC ft318 指纹 | `scripts/check_sonic_dataset.py --fingerprints` | 9 个子集全 `[ok]`，**318 eps / 473,876 frames / 2.633 h @50fps / 4 tasks** |
| ft318 → npz | 逐文件读 `joint_pos.shape[0]` 求和 | **318 条 / 473,876 帧**（与锁定值逐帧一致），每条含 `g1_state_46`/`g1_motion_token_64` |
| 工作区测试 | `pytest tests -q` | **94 passed / 3 skipped** |
| SuperMimic 测试 | `.venv/bin/python -m pytest tests -q` | **7 passed / 17 skipped** |
| 部署后端 | `SUPERMIMIC_DEPLOY_TEST=1 pytest tests/test_supermimic_deploy_backend.py` | **3 passed** |
| 新模块 | `ls diff_vla_wbc/{sonic_encoder,sonic_token,sonic_dataset,hand_stream,hand_head,goal_head,g1_tasks}.py` | 全部存在 |
| token 评测路径 | grep `token_fn`/`interface="token"` | `src/supermimic/eval/benchmark.py`、`scripts/play.py` 均已支持 |

> ⚠️ 路径提醒：BONES 的 npz 与紧凑库在 **`third_party/SuperMimic-Reproduced/data/…`** 下，
> **不在仓库根 `data/`**。按根目录去找会误判成"数据缺失"（本次复核第一遍就踩了）。

### 11.2 与 §7 不一致（认知需要更新）

1. **训练进度比 §7 记录的超前得多。** §7 只记了 4090 那次 `it 1450` 已停；实际还有一次
   **2×H200 的 token 版跑到了 20,300 迭代**：

   ```text
   run      logs/e9_bones_seed_token_h200x2        (token_dim=64, total_iterations=320000, envs=4096)
   last     step 20300 @ 2.42 h，≈2.33 it/s（tb: sim .047 + wm .078 + policy .300 s/iter）
   ckpt     ckpt_0020000.pt（183 MB, 07:06）+ ckpt_latest.pt（= 20300 步，07:06）
   状态     07:08 后无写入，07:10 复核时无进程 —— 已停（日志在 H200 那台机器的 /tmp，本机看不到停机原因）
   ```

   关键指标（step 20300）：`train/episode_norm_return **0.8108**`、`train/episode_len 109.6`、
   `wm/nse_1step 0.0052`、`wm/nse_kstep 0.0101`、`policy/imag_return 2.81`、
   `policy/imag_survival_H 0.9995`、`policy/action_abs 0.93`、`critic_loss 0.0159`。
   对比 4090 的 `it 1450`（ret 0.499 / surv 0.428 / nse1 0.0179），**H200 这轮明显更好**。

   按 2.33 it/s 估算，20.3k → 320k 还需 **≈299.7k 迭代 ≈ 35.7 h**。

2. **§7「下一步」第 4 条已过时**：token 版 `eval_benchmark.py` / `play.py` 的路径已经写好了
   （§7 审计表 ✅ 是对的，"下一步"清单没同步）。

3. **§7「下一步」第 5 条已过时**：ft318 → SuperMimic npz **已完成**，318 条、帧数与锁定值一致。

4. **旧 run 不可复用**：`logs/e8_bones_seed_4090x2`（44k ckpt）的 `policy.token_dim` 是 `None`，
   即 680-goal 版本，与 token 条件策略不兼容（§7 已提示，本次确认）。

### 11.3 本次顺手修的两处

| 问题 | 修法 |
|---|---|
| `data/npz/g1_tasks_ft318/` 未进 `.gitignore`（318 个 npz ≈700 MB 会被误提交） | `.gitignore` 改为 `data/npz/g1_tasks*/` |
| `docs/FRAMEWORK.md` 仍指向已废弃的 ft271（`data/npz/g1_tasks`、271 条） | 全部改为 ft318 / `data/npz/g1_tasks_ft318`（8 处，ft271 引用清零） |

### 11.4 仍成立的风险 / 未闭环（与 §7 一致，均未变化）

| # | 项 | 证据 / 说明 |
|---|---|---|
| 1 | **`wm/term_acc = 1.0`、`term_bce = 0.0001` 可能是平凡满分** | 需确认终止标签的正样本比例；若全为负，`safety` 项等于没信号 |
| 2 | 320k 只到 **6.3%** | 距论文口径（Table A.8）还很远 |
| 3 | **端到端联合训练未做** | VLA → token → SuperMimic → 奖励反传 |
| 4 | 部署 A/B 未做 | `model_decoder.onnx` vs SuperMimic policy 的任务成功率对比 |
| 5 | 手部指标未定义 | 指尖误差 / 抓握成功率 |
| 6 | `data/npz/g1_tasks`（ft271、271 条、597 MB）仍在盘上 | §7 已标注"别用错"，建议删除或改名 `_deprecated` |

### 11.5 一句话现状

> **数据、环境、token encoder、SuperMimic token 条件化、手部通道、评测路径、部署后端全部就位且实测通过；
> 唯一缺的是"训练量"和"端到端联合"** —— 一个 20.3k/320k 的 token 版 checkpoint 已经证明曲线健康
> （`episode_norm_return 0.81`、world model `nse_1step 0.0052`），但还没跑完，也还没接上 VLA 反传。

### 11.6 修订后的下一步（替换 §7 的旧清单）

1. **先查 termination 正样本率**（10 分钟）——确认 `safety` 项不是空信号，再谈长跑；
2. **续训 320k**：`PROFILE=paper` 从 `logs/e9_bones_seed_token_h200x2/ckpt_latest.pt` 续，
   约 36 h；同步记录 `episode_norm_return` 是否向 0.90 收敛；
3. **端到端联合训练**（当前最大空白）：`VLA(20k ckpt) → token → SuperMimic(token ckpt) → WM → 奖励 → 反传`，
   用 ft318 的 318 条、小 batch/短 horizon 起步；`diff_vla_wbc/` 里 `--vlm --vlm-dataset` 已能喂真实 ego_view；
4. **部署 A/B**：同一段 VLA token，`model_decoder.onnx` vs SuperMimic 策略；
5. **手部指标**：指尖误差 / 抓握成功率；
6. 清理 `data/npz/g1_tasks`（ft271）避免用错。
