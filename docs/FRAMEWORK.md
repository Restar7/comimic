# Differentiable VLA–Whole-Body Co-Training Framework

> 状态：**可运行的接入框架 + 已通过真实 GR00T action head 的端到端梯度校验**；
> 顶层 VLM 已接上本机真实的 **Cosmos-Reason2-2B** 权重（`diff_vla_wbc/gr00t_vlm.py`），
> `RGB + instruction -> VLM -> FM head -> tokens -> adapter -> WBC -> WM -> loss` 可跑通并反传到 action head。
> 尚未完成：真实 GR00T checkpoint 的 action head 权重、BONES-SEED 全量预训练、带物体/接触的动力学的抓取任务闭环。
> 这些限制不是接口缺失，而是缺少权重/数据/算力，下面会逐条写清楚。

## 1. 选型：为什么是 CuboiLeo/SuperMimic-Reproduced

在 GitHub 上对比了三个 `supermimic` 实现：

| 仓库 | 数据 | 报告效果 | 结论 |
|---|---|---|---|
| **CuboiLeo/SuperMimic-Reproduced** | LAFAN1 + **BONES-SEED 129.9k clips** 管线 | LAFAN1 0.904；BONES 子集 0.935 | **选中**：唯一有完整 BONES 数据管线，效果最好，模块最干净 |
| Motphys/Uni-SuperMimic | LAFAN1 40 clips | 40 clips 0.829 | 接口接近原 bridge，但没有 BONES |
| zita-ch/supermimic_vibed | LAFAN1 40 clips | 0.829，0/40 fail | 依赖 Isaac Lab 3 + Docker，工程重 |

已 clone 到：

```text
third_party/SuperMimic-Reproduced
commit d9be385b3eeb13a638fb33fb200d3bfd9be9839f
```

**重要事实**：三个仓库都**没有随代码分发预训练 checkpoint**。BONES-SEED 本身是
gated dataset（Hugging Face `bones-studio/seed`，需接受许可）。所以“clone 下来就有
BONES 预训练权重”不存在；本框架把加载、微调、评测路径接好，权重需要自己训练或申请。

## 2. 计算图

```text
RGB + language + state
        │
        ▼
Cosmos-Reason2-2B backbone  (本地权重, gr00t_vlm.Gr00tVLM)
        │  backbone_features [B,T,2048]
        ▼
Flow Matching action head  (DiT, Euler ODE, 4/8 steps)
        │  a_theta  [B, 40, 64]
        ▼
MotionGoalAdapter  (pool 4 tokens/frame + temporal transformer)
        │  g_eta    [B, 10, 68]  (680-D)
        ▼
SuperMimic WBC policy  obs_hist [B,10,93] + g_eta
        │  u_psi    [B, 29] normalized joint offsets
        ▼
Learned differentiable world model F_phi  (frozen params, live inputs)
        │  s_{t+1}  [B, 71]
        ▼
Analytic FK + tracking reward / termination head
        │
        ▼
L = L_task + L_track + L_goal + L_smooth + L_FM
        │
        └── loss.backward() ──► adapter / WBC / FM DiT / (optionally VLM)
```

关键点：`F_phi` 的参数冻结，但 **forward 没有 `torch.no_grad()`**，所以梯度仍能穿过
世界模型回到动作。世界模型本身用真实 transition 单独更新。

## 3. 张量契约（CuboiLeo 版本）

```text
state s        [B,71] = root_pos(3) | root_quat wxyz(4) | q(29)
                      | root_lin_vel_w(3) | root_ang_vel_b(3) | dq(29)
observation o  [B,93] = (q + bias - q_default)(29) | dq(29)
                      | ang_vel_b(3) | projected_gravity(3) | prev_action(29)
obs history    [B,10,93] -> flatten term-major -> [B,930]
goal g         [B,10,68] = per frame:
                      (q - q_default)(29) | dq(29) | gravity(3)
                      | ang_vel_b(3) | lin_vel_heading(3) | height(1)
action u       [B,29] = normalized joint-position offset
q_target       = q_default + G1_ACTION_SCALE * u
```

G1 MJCF：`GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml`
（29 DoF，nq=36，nv=35，actuator=29）。`g1/meshes` 指向
`reproduce_scenes/g1/meshes`（脚本 `scripts/setup_assets.sh` 可重建）。

**注意：163 MB 的 G1 mesh 资产没有上传到 Git。** 先准备
`reproduce_scenes/g1/meshes/`（可从 Unitree G1 / LAFAN1_Retargeting_Dataset 获取），
再运行 `bash scripts/setup_assets.sh` 创建 MJCF 需要的软链。没有 mesh 时，
`build_g1_assets()` 会给出明确的缺失路径错误，但无法编译 MuJoCo 模型。

## 4. 目录

```text
diff_vla_wbc/
  compat.py               mjlab.math shim / G1 资产 / action scale / 默认姿态
  gr00t_fm.py             无 no_grad 的 Euler FM 采样（真实 Gr00tN1d7ActionHead）
  gr00t_vlm.py            顶层 VLM：Cosmos-Reason2-2B + action head 的本地加载与 batch 构造
  gr00t_tiny.py           用真实 action head 做小尺寸梯度测试
  adapter.py              64-D token -> 10x68 motion goal
  supermimic_backend.py   加载/构建上游 Policy + WorldModel + AnalyticHeads
  rollout.py              FM -> adapter -> WBC -> WM -> reward 的可微 rollout
  train.py                冻结 WM 的联合训练 + 单独 WM 更新
  data.py                 npz 转换、MotionLibrary batch、缓存 token
  task_losses.py          末端/根高/姿态任务损失示例
  mujoco_env.py           CPU MuJoCo 闭环评测
scripts/
  prepare_motion_npz.py   将本工作区 G1 动作 CSV 转成 SuperMimic npz
  train_joint.py          联合训练 / 梯度 smoke CLI
  eval_mujoco.py          MuJoCo 闭环评测
  smoke_vlm.py            真实 VLM 冒烟：RGB+instruction -> tokens -> goal（含梯度检查）
  link_vlm_weights.py     把本地 HF 缓存里的 Cosmos 权重软链成仓库内视图
tests/                    9 个测试，包含真实 GR00T head 的反传
```

## 5. 快速运行

依赖：Python 3.12+、PyTorch、MuJoCo、transformers、diffusers、dm-tree、numpy/scipy。
当前测试环境：Python 3.13、torch 2.11、mujoco 3.16、transformers 5.1、diffusers 0.41。

```bash
# 0) G1 mesh 资产不在仓库里；先把 meshes 放到 reproduce_scenes/g1/meshes，
#    再创建 MJCF 需要的软链
bash scripts/setup_assets.sh

# 1) 把真机 G1 任务录制转成 MotionLibrary npz（318 条 / 473,876 帧 / 50 fps）
#    产出约 1.8 GB，已 git-ignore。每个 npz 里除了 SuperMimic 需要的
#    joint_pos/joint_vel/body_*，还带 ego_view 路径、46 维 state、64 维
#    motion token 和该 episode 的指令，供 VLM 侧按 frame 对齐。
python scripts/prepare_g1_tasks_npz.py \
  --root /inspire/.../flowbody/FlowBody/dataset/groot_sonic_vla_ft318_frozen_20261023 \
  --out-dir data/npz/g1_tasks_ft318

# 2) 用真实 GR00T action head（tiny DiT）跑通完整联合梯度
python scripts/train_joint.py \
  --motion data/npz/g1_tasks_ft318 \
  --steps 20 --batch-size 2 --horizon 4 --small --device cpu \
  --real-gr00t-head --save --out runs/joint_smoke

# 3) MuJoCo 闭环（随机/训练后的 WBC 都会跑，随机策略会摔倒，这是预期的）
python scripts/eval_mujoco.py \
  --motion data/npz/g1_tasks_ft318 --joint-ckpt runs/joint_smoke/joint_latest.pt \
  --small --clip-index 0 --max-steps 100 --out runs/eval_smoke

# 3b) 控制器基线：直接用下一帧参考关节角作为 q_target，不经过策略
python scripts/eval_mujoco.py \
  --motion data/npz/g1_tasks_ft318 --teacher --clip-index 0 --max-steps 100

# 3c) 顶层 VLM：真实 Cosmos-Reason2-2B -> FM head -> adapter -> WBC
python scripts/link_vlm_weights.py --help     # 可选：建立仓库内权重视图（软链）
python scripts/smoke_vlm.py --device cuda:0   # 只验 VLM 这一段（约 2 分钟）
python scripts/train_joint.py \
  --motion data/npz/g1_tasks_ft318 --steps 1 --batch-size 1 --horizon 4 --small \
  --device cuda:0 --vlm --vlm-steps 2         # 全链 1 步（合成图，仅接线）

# 3d) 全链 + 真实观测：每步按采样到的 motion 窗口取对应的 ego_view 帧、
#     46 维 state 与该 episode 的指令，走 checkpoint 自带的 Gr00tN1d7Processor
python scripts/train_joint.py \
  --motion data/npz/g1_tasks_ft318 --steps 2 --batch-size 2 --horizon 4 --small --device cuda:0 \
  --supermimic-ckpt <supermimic.pt> \
  --vlm --vlm-dataset --vlm-ckpt <gr00t>/checkpoint-20000 --vlm-steps 2

# 4) 测试
python -m pytest tests -q
```

当前 smoke 输出（CPU，tiny head）证明：

```text
FM DiT grad        > 0
adapter grad       > 0
policy grad        > 0
frozen WM param grad = None
```

## 6. 接入真实 GR00T / 真实 SuperMimic 权重

### 6.1 真实 GR00T

```python
from diff_vla_wbc.gr00t_fm import sample_model_with_grad

# 你的完整 Isaac-GR00T 环境里：
model = Gr00tN1d7.from_pretrained(ckpt_path, ...)   # 真实 2B VLA
model.action_head.set_trainable_parameters(
    tune_projector=True, tune_diffusion_model=True, tune_vlln=False
)

tokens = sample_model_with_grad(
    model,
    batch,                    # 与 Gr00tN1d7.forward 相同的预处理 batch
    num_steps=8,              # 4 也可；8 更稳
    initial_noise=fixed_noise # 训练时固定噪声便于复现
)
```

注意：
- **不要**调用 `model.get_action()` / `action_head.get_action()`，它们带 `@torch.no_grad()`。
- `sample_model_with_grad` 内部会调用 `prepare_input` → `backbone` → 无 no_grad 的
  Euler 循环。
- 全量反传 2B VLM 很吃显存；建议只解冻 action head 的 projector/DiT，VLM 冻结。
- 如果 8 步反传不稳定，可把 `num_steps` 降到 4，或后续换成 Q-VGM / adjoint 后端；
  接口已经隔离在 `sample_fm_with_grad`。

### 6.2 真实 SuperMimic checkpoint

```python
from diff_vla_wbc.supermimic_backend import build_g1_assets, SuperMimicBundle

assets = build_g1_assets(device="cuda")
bundle = SuperMimicBundle.from_checkpoint(
    "runs/supermimic_mjwarp_001/ckpt_012000.pt", assets
)
```

checkpoint 必须来自 CuboiLeo/SuperMimic-Reproduced（含 `config.policy` /
`config.world_model` / `policy` / `wm`）。Motphys 的 checkpoint 接口不同，不能直接加载。

### 6.3 联合训练

```python
from diff_vla_wbc.adapter import AdapterConfig, MotionGoalAdapter
from diff_vla_wbc.data import MotionBatchProvider, sample_reference_windows
from diff_vla_wbc.gr00t_fm import Gr00tTokenSource
from diff_vla_wbc.train import JointTrainer, JointTrainerConfig

adapter = MotionGoalAdapter(AdapterConfig(token_dim=64, token_horizon=40))
trainer = JointTrainer(
    bundle, adapter,
    fm_sampler=Gr00tTokenSource(gr00t_model, num_steps=8),
    fm_parameters=[p for p in gr00t_model.action_head.parameters() if p.requires_grad],
    cfg=JointTrainerConfig(horizon=8),
    device="cuda",
)

# 先用真实 transition 预训练/更新 WM
trainer.train_world_model(sample_reference_windows(bundle, library, batch_size=8, window=8))
# 再冻结 WM，用生成目标做 BPTT
trainer.train_step(batch)   # batch 来自你的 RGB/language/motion 对齐数据集
```

真实数据需要提供 `start_state / obs_hist / history / xi / references /
teacher_goals / teacher_goal_sequence / teacher_states / fm_tokens`。本仓库的
`MotionBatchProvider` 已实现从 motion library 构造这些字段；把
`token_source` 换成真实 GR00T token 即可。

### 6.4 顶层 VLM（Cosmos-Reason2-2B，已接上）

框架原来从 FM token 起步，VLM 这一段是空的。现在 `diff_vla_wbc/gr00t_vlm.py`
把 **Isaac-GR00T 的 Cosmos-Reason2-2B backbone + Gr00tN1d7ActionHead** 组装成
`Gr00tVLM`，它就是 `sample_model_with_grad` 要的 `gr00t_model`：

```python
from diff_vla_wbc.gr00t_vlm import Gr00tVLM, Gr00tVLMTokenSource

vlm = Gr00tVLM.from_pretrained(device="cuda:0")        # 自动解析本地 Cosmos 权重
batch = vlm.build_batch(image, ["Pick up the carton."], state)   # state: [B,43]
trainer = JointTrainer(
    bundle, adapter,
    fm_sampler=Gr00tVLMTokenSource(vlm, num_steps=2),
    fm_parameters=vlm.trainable_parameters(),
    device="cuda",
)
```

**权重解析顺序**（`resolve_vlm_source`，全部 `local_files_only`）：
`model_name=` 参数 → `$GR00T_VLM_PATH` → 仓库内视图
`groot_repro/nvidia/Cosmos-Reason2-2B-weights/`（`scripts/link_vlm_weights.py`
用软链生成，`.gitignore` 已忽略）→ 本机 Hugging Face 缓存里的
`models--nvidia--Cosmos-Reason2-2B/snapshots/*`（会顺带扫描 checkout 上层的
`cache/huggingface/hub`）→ 裸 repo id。本机命中的是扫描到的缓存快照。

**为什么不用 `Gr00tN1d7.from_pretrained`**：它要求完整 GR00T checkpoint（2.5 GB
head + gated 的 `nvidia/Cosmos-Reason2-2B`），而 `get_action` 带 `@torch.no_grad()`。
`Gr00tVLM` 只重写了「加载 + prepare_input」这一层，backbone / action head / Euler
采样全部是 vendored 的原版代码。

**从 flowbody 的 Isaac-GR00T 补齐的东西**（本目录原来缺这些，缺了连
`import gr00t` 都过不去）：

| 路径 | 作用 |
| --- | --- |
| `Isaac-GR00T/gr00t/data/` | VLM 数据管线：embodiment tag、state/action processor、collator、dataset |
| `Isaac-GR00T/gr00t/configs/data/` | `DataConfig` / modality config（`base_config` 依赖） |
| `Isaac-GR00T/tests/gr00t/data/` | 上面两块的测试 |
| `gr00t/configs/finetune_config.py`、`gr00t/experiment/launch_finetune.py` | 与 flowbody 一致的 `dataset_specs_json` 改动 |

**在本机环境（transformers 5.9）上必须的兼容补丁**，都写了注释：

1. `gr00t/configs/model/gr00t_n1d7.py`：5.9 的 `PreTrainedConfig.__init_subclass__`
   会给子类自动 `@dataclass`，再叠一层 `@dataclass` + `field(default_factory=...)`
   会在 import 期报 `non-default argument 'diffusion_model_cfg' follows default
   argument`。改成「按版本条件加装饰器 + 模块级默认 dict + `__init__` 里 deepcopy」。
2. `gr00t/model/modules/qwen3_backbone.py`：5.x 把 text stack / vision tower 挪到了
   `model.model.language_model` / `model.model.visual`，用一个 `_language_model()` /
   `_vision_tower()` 访问器同时兼容 4.57 与 5.x。
3. 同上文件的 `forward`：5.x 的 Qwen3-VL 需要 processor 新给出的
   `mm_token_type_ids`（M-RoPE），改成把 processor 产出的可选多模态张量一并透传。

**已验证**（2×RTX 4090，`scripts/smoke_vlm.py`、`scripts/train_joint.py --vlm`）：

```text
VLM source        .../models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a19...
batch             input_ids[1,81] mm_token_type_ids[1,81] pixel_values[256,1536] image_grid_thw[1,3]
backbone layers   16（28 层 Cosmos 按 GR00T N1.7 习惯截断）
总参数            2.70 B（VLM + 550 M DiT head），trainable 869 M（head；VLM 冻结）
tokens            [1,40,64] bfloat16，requires_grad=True
goal              [1,10,68] float32
梯度               action head |grad| 和 > 0，backbone |grad| = 0（冻结时）
联合训练 1 步      loss 0.6746（reward 0.533 / goal 0.0989 / state 0.0762 / safety 0.4968 全链 BPTT）
```

**边界（别当成已完成）**：

- 基础用法（`--vlm` 不带 `--vlm-dataset`）**每一步都喂同一张合成图**，只适合验证
  梯度管道。**用这种输入去微调 head 会把预训练学到的视觉条件化抹平** —— 因为输入
  是常量时，loss 的唯一下降方向就是让 head 忽略图像输出常数。真实训练必须用 6.5。
- 显存：bf16 加载 + 冻结 VLM 时 2.7 B 模型约 6 GB；`--train-vlm` 反传 16 层
  backbone 需要更多显存和更小 batch。

### 6.5 真实观测（ego_view）接入 —— `--vlm-dataset`

6.4 的 `Gr00tVLM.build_batch` 走的是**裸 `AutoProcessor` + 手写 chat template +
原始 state**。这在没有 checkpoint 时够用，但一旦加载真 checkpoint 就是错的：那个
head 是在 `Gr00tN1d7Processor` 里训出来的，它会把图像按 `shortest_edge 256 -> crop
230 -> 256` 缩放、把 state 做**百分位归一化**并补到 132 维、把语言 `formalize`。
喂原始图 + 原始关节角等于让它分布外推理。

`--vlm-dataset` 换回 checkpoint 自己的预处理链路：

```text
MotionBatchProvider 采到 window (frames)
        |
        v
TaskFrameIndex: 全局 frame -> (episode npz, 集内 frame)
        |
        +--> EgoViewImages: mp4 seek 到该帧，解码出 480x640 RGB
        +--> npz 里的 g1_state_46 / g1_instruction
        |
        v
Gr00tN1d7Processor.process_observation(obs, UNITREE_G1_SONIC)
        |
        v
BatchFeature(input_ids, pixel_values, state[B,1,132], embodiment_id=11, action_mask)
        |
        v
sample_model_with_grad -> tokens[B,40,132][..., :64] -> adapter -> WBC -> WM -> loss
```

实现：`diff_vla_wbc/g1_tasks.py`（索引 + 视频随机访问 + 观测拼装），npz 由
`scripts/prepare_g1_tasks_npz.py` 生成。每个 npz 额外存了 `g1_state_46`、
`g1_motion_token_64`、`g1_hand_joints_14`、`g1_instruction`、`g1_ego_view`。

**录制数据没有 root 位置/速度**（平移编码在 motion token 里，由 SONIC 自己的 WBC
解码），所以转换器只能重建：

| 量 | 做法 | 代价 |
| --- | --- | --- |
| `root_quat` | 用实测 `observation.root_orientation` | 准确（机器人直立） |
| `root_z` | **足端接触估高**：FK 后把最低 ankle_roll 放到地面 | 跟随蹲起，误差小 |
| `root_xy` | 钉在原点 | **导航类任务（task_1）会失真**，原地任务可接受 |
| root 速度 | 置零 | 跟踪奖励少了根速度项 |

所以现在的参考适合 `task_2/3/4`（办公室、厨房、餐厅，基本原地），`task_1` 含导航段，
暂不适用。要彻底解决需要 SONIC 的 token 解码器或带 root 轨迹的录制。

**已实测**（真实 ckpt + 真实 SuperMimic + 真实帧，2 步 × batch 2）：

```text
[info] real observations: 2 episodes / 7191 frames, processor=Gr00tN1d7Processor
[step 1/2] loss 3.9197  (reward -0.0411 / goal 0.1016 / state 7.7153 / safety 0.1528)
[step 2/2] loss 3.6703  (reward -0.0267 / goal 0.0632 / state 7.2645 / safety 0.1515)
```

`loss/state` 仍在 7 左右：那是 200 步训出来的玩具 world model，不是链路问题。

## 7. BONES-SEED

- 数据集：`bones-studio/seed`，约 288 小时、142,220 条 120 fps G1 动作。CuboiLeo 的管线把它转成 50 Hz，过滤后 129,943 条 / 263.75 h。
- **数据未就位（2026-10-23 复核）**：`/inspire/dataset/bones-seed/v1` 共 261 KB，只有 `LICENSE.md` / `README.md` 和一个**空的** `metadata/`；`g1/` 下 0 条 CSV，`g1.tar.gz`（23.5 GB，HF gated）尚未下载。补齐步骤见 [`BONES_SEED_H200X2.md`](BONES_SEED_H200X2.md) §1.2。仓库本身仍**不包含**权重：三个复现仓库都不分发 checkpoint，BONES 预训练权重需要自己训练。
- 全量训练（含数据管线三步 + 2×H200 训练/续训/评测命令）见 [`BONES_SEED_H200X2.md`](BONES_SEED_H200X2.md)。
- 只查脚本用法：

```bash
cd third_party/SuperMimic-Reproduced
. .venv/bin/activate          # 环境安装见 BONES_SEED_H200X2.md 第 2 节（venv + pip，不用 uv）
python scripts/data/csv_to_npz_batch.py --help
python scripts/data/filter_bones_seed.py --help
python scripts/data/build_library.py --help
```

然后本框架加载：

```python
from diff_vla_wbc.supermimic_backend import load_motion_library
library = load_motion_library(
    ["third_party/SuperMimic-Reproduced/data/library/bones_seed_compact.pt"],
    assets, compact=True,
)
```

`load_motion_library(compact=True)` 会把上游 `motion_convert.load_g1_model`
重定向到本仓库的 G1 XML，避免安装 mjlab。

## 8. 已实际验证 / 未验证

已运行：

- `pytest tests -q`：**18 passed**（9 个原有用例 + 9 个 `tests/test_gr00t_vlm.py`）。
- 真实 `Gr00tN1d7ActionHead`（tiny DiT，65k 参数）+ 真实 SuperMimic
  `Policy` / `WorldModel` / `AnalyticHeads` + 真实 G1 MJCF：
  - FM 采样 3 步反传到 DiT：grad > 0；
  - adapter grad > 0；
  - WBC policy grad > 0；
  - 冻结 WM 参数 grad = None；
  - MuJoCo 29-DoF 闭环可 step / rollout（补齐 deploy MJCF 缺失的 armature，
    并使用 implicitfast 积分器；否则 10 Hz PD 增益在 5 ms 步长下会数值发散）。
- 真机 G1 任务录制（271 条 / 404,989 帧 / 50 fps）已由 `scripts/prepare_g1_tasks_npz.py`
  转成 `data/npz/g1_tasks_ft318/*.npz`（318 条，约 700 MB，git-ignored）。
- compact BONES-style `.pt` library 的加载/目标窗口/参考状态接口已用
  合成 packed library 验证。
- **顶层 VLM（真实 Cosmos-Reason2-2B 权重）+ 真实 `Gr00tN1d7ActionHead` 结构**：
  `scripts/smoke_vlm.py` 跑通 `RGB + instruction + state -> backbone(16 层) ->
  FM head -> tokens [1,40,64](bf16) -> MotionGoalAdapter -> goal [1,10,68]`，
  action head 反传 grad > 0；`scripts/train_joint.py --vlm` 跑通 1 步全链
  （VLM -> FM -> adapter -> WBC -> WM -> task loss）。
- 本目录资产/入口已适配：G1 mesh 从本机 Unitree G1 描述目录补齐（
  `reproduce_scenes/g1/meshes`，`.gitignore` 忽略），deploy MJCF 的 `meshes`
  软链到它；缺的 `gr00t/data`、`gr00t/configs/data` 已从 flowbody 的
  Isaac-GR00T 补齐；`groot_repro/prepare_ckpt.py` 的 donor-config 路径改为
  仓库内相对路径。

未验证：

- 真实 GR00T checkpoint（2.5 GB action head）。VLM 用的是真权重，但 head 目前是
  随机初始化；`Gr00tVLM.from_pretrained(checkpoint=...)` 路径未在本机跑过。
- 与 motion 对齐的 RGB/language 训练数据（CLI 用合成图 / `--vlm-image` 单图）。
- 真实 2B VLA 端到端任务成功率（本机当前只有 1×RTX 4090 48 GB，仓库也没有任务 checkpoint）。
- 真实 BONES-SEED 全量训练：数据**还没下载**（`/inspire/dataset/bones-seed/v1` 只有 261 KB 的空壳），
  且本机是 1×4090 48 GB 而非 2×H200；两套命令与前置条件见
  [`BONES_SEED_H200X2.md`](BONES_SEED_H200X2.md) §0/§1.2/§4，尚未在本机实跑。
- 抓取任务的对象/接触世界模型；当前 WM 只预测机器人 71 维状态。
- 论文级 4096 env、数小时训练；本机只能做 smoke/小数据验证。

## 9. 论文依据

- SuperMimic: <https://arxiv.org/abs/2610.28480>
- VITA: <https://arxiv.org/abs/2507.13231>
- Q-VGM: <https://arxiv.org/abs/2606.08015>
- Flow Policy Gradients: <https://arxiv.org/abs/2602.02481>
- CuboiLeo reproduction: <https://github.com/CuboiLeo/SuperMimic-Reproduced>
- Motphys reproduction: <https://github.com/Motphys/Uni-SuperMimic>
