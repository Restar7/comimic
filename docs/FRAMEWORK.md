# Differentiable VLA–Whole-Body Co-Training Framework

> 状态：**可运行的接入框架 + 已通过真实 GR00T action head 的端到端梯度校验**。
> 尚未完成：真实 GR00T 2B checkpoint 的联合训练、BONES-SEED 全量预训练、带物体/接触的动力学的抓取任务闭环。
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
GR00T N1.7 VLM backbone
        │
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

## 4. 目录

```text
diff_vla_wbc/
  compat.py               mjlab.math shim / G1 资产 / action scale / 默认姿态
  gr00t_fm.py             无 no_grad 的 Euler FM 采样（真实 Gr00tN1d7ActionHead）
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
tests/                    9 个测试，包含真实 GR00T head 的反传
```

## 5. 快速运行

依赖：Python 3.12+、PyTorch、MuJoCo、transformers、diffusers、dm-tree、numpy/scipy。
当前测试环境：Python 3.13、torch 2.11、mujoco 3.16、transformers 5.1、diffusers 0.41。

```bash
# 0) 如果 meshes 软链不存在
bash scripts/setup_assets.sh

# 1) 把工作区里的 G1 动作导出为 npz（已生成 13 条 example）
python scripts/prepare_motion_npz.py \
  --csv-root GR00T-WholeBodyControl/gear_sonic_deploy/reference/example \
  --out-dir data/npz/example

# 2) 用真实 GR00T action head（tiny DiT）跑通完整联合梯度
python scripts/train_joint.py \
  --motion data/npz/example \
  --steps 20 --batch-size 2 --horizon 4 --small --device cpu \
  --real-gr00t-head --save --out runs/joint_smoke

# 3) MuJoCo 闭环（随机/训练后的 WBC 都会跑，随机策略会摔倒，这是预期的）
python scripts/eval_mujoco.py \
  --motion data/npz/example --joint-ckpt runs/joint_smoke/joint_latest.pt \
  --small --clip-index 0 --max-steps 100 --out runs/eval_smoke

# 3b) 控制器基线：直接用下一帧参考关节角作为 q_target，不经过策略
python scripts/eval_mujoco.py \
  --motion data/npz/example --teacher --clip-index 0 --max-steps 100

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

## 7. BONES-SEED

- 数据集：`bones-studio/seed`，gated，需接受 license；约 288 小时、142,220 条
  120 fps G1 动作。CuboiLeo 的管线把它转成 50 Hz，过滤后 129,943 条 / 263.75 h。
- 当前仓库**不包含** BONES 数据或权重，也没有下载 gated 数据。
- 用上游脚本在有权重的环境里准备 compact library：

```bash
cd third_party/SuperMimic-Reproduced
uv run python scripts/data/csv_to_npz_batch.py --help
uv run python scripts/data/filter_bones_seed.py --help
uv run python scripts/data/build_library.py --help
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

- `pytest tests -q`：**9 passed**。
- 真实 `Gr00tN1d7ActionHead`（tiny DiT，65k 参数）+ 真实 SuperMimic
  `Policy` / `WorldModel` / `AnalyticHeads` + 真实 G1 MJCF：
  - FM 采样 3 步反传到 DiT：grad > 0；
  - adapter grad > 0；
  - WBC policy grad > 0；
  - 冻结 WM 参数 grad = None；
  - MuJoCo 29-DoF 闭环可 step / rollout（补齐 deploy MJCF 缺失的 armature，
    并使用 implicitfast 积分器；否则 10 Hz PD 增益在 5 ms 步长下会数值发散）。
- 13 条工作区 G1 动作已转成 `data/npz/example/*.npz`。
- compact BONES-style `.pt` library 的加载/目标窗口/参考状态接口已用
  合成 packed library 验证。

未验证：

- 真实 2B VLA checkpoint（本机 8 GB RTX 5060 Laptop 放不下，仓库也没有 checkpoint）。
- 真实 BONES-SEED 训练（gated + 无权重）。
- 抓取任务的对象/接触世界模型；当前 WM 只预测机器人 71 维状态。
- 论文级 4096 env、数小时训练；本机只能做 smoke/小数据验证。

## 9. 论文依据

- SuperMimic: <https://arxiv.org/abs/2610.28480>
- VITA: <https://arxiv.org/abs/2507.13231>
- Q-VGM: <https://arxiv.org/abs/2606.08015>
- Flow Policy Gradients: <https://arxiv.org/abs/2602.02481>
- CuboiLeo reproduction: <https://github.com/CuboiLeo/SuperMimic-Reproduced>
- Motphys reproduction: <https://github.com/Motphys/Uni-SuperMimic>
