# MuJoCo 路线迁移方案：93 维观测 + 318 续训 + IsaacLab 联合微调架构移植

> 版本 **v2** · 2026-10-28 · 本文件是修改方案；**已落地两个数据产物**：`data/library/ft318_compact.pt`（参考库）与 `data/library/ft318_tokens.pt`（录制 token，§5.2.1）
>
> **v2 修订**（用户 2026-10-28 确认）：① 去掉端口侧车，token 统一为 **hold = label**；
> ② 确认 318 是 hold 且前瞻来自 **token 序列**（40 个）；③ 明确 318 **没有命令流**，
> 给出 A/B 两套续训方案；④ 补上 W1 的第一个硬坑（bin 表尺寸不匹配）与预处理脚本；
> ⑤ 新增 §6.9 与 IsaacLab WBC 微调流程的逐项对齐表。
>
> **v3 修订**（用户 2026-10-28 追问）：**直接在数据集上训练，用录制 token，不重建 hold**。
> §5.2 换成「读 token 源」（不再需要 hold 前端）；§6.2 说明为什么这是 318 上的正确做法；
> 代价是 anchor 与录制姿态绑定（用关掉姿态扰动 + 对照臂处置）。
>
> 读者：执行迁移的人、评审的人。
> 目标一句话：**在 MuJoCo/mjlab 路线上，用 318 数据集续训 SuperMimic（WBC），并把
> IsaacLab 主线（`scripts/train_joint_v2.py` / `diff_vla_wbc/train_v2.py` / `v3_step.py`）
> 的联合微调**思想、方法、架构**完整搬过来，观测固定 93 维。**
>
> 相关文档：
> * `docs/DATA_POLICY_HA_ONLY.md`（本方案要修改它的 §0.2 禁令）
> * `runs/OFFICIAL_SUPERMIMIC.json`（接口差异清单，本方案 §3 的输入）
> * `docs/HANDOVER_SUPERMIMIC_FINETUNE.md`（历史踩坑：token 空间 / 速度通道 / action 语义）
> * `docs/S3_END_TO_END_PLAN_20261028.md`（Isaac 主线现状，本方案移植的是它）
> * `docs/FT318_HAND_JOINT_1000_20261025.md`（318 时代 MuJoCo 路线唯一跑过的联合微调）

---

## 0. 摘要

### 0.1 要做的三条流水线

| # | 流水线 | 原来在哪 | 迁移后在哪 | 产物 |
|---|---|---|---|---|
| **W1** | 318 数据集 → SuperMimic(WBC) **续训** | 只有 Isaac 有（`trackerlab/logs/wbc_*`，且从未用过 318） | mjlab/MuJoCo-Warp（`sm-train --data.library ft318_compact.pt --resume ...`） | `third_party/SuperMimic-Reproduced/logs/wbc_ft318_mjlab/ckpt_*.pt` |
| **J1** | 联合微调（VLA head ← 奖励/BC 反传，VLA+WBC+WM） | Isaac（v2/v3，obs 99） | mjlab 后端 + **obs 93**（同一套 `train_v2/v3`） | `runs/joint_mjlab_ft318/joint_*.pt` + `vla_*.pt` |
| **E1** | MuJoCo 闭环评测 | 只有 decoder 路线与旧 joint ckpt | 吃 J1 产物，93 维口径 | `runs/joint_mjlab_ft318/cl/qc_*.json` |

### 0.2 明确不做

1. **不改 Isaac 主线**：`train_joint_v2.py` / `isaac_joint.py` / HA 数据 / `checkpoint-8000` 全部保持现状，本方案不动它们。
2. **不把 93 改成 99**：mjlab 侧的 policy 首层是 `(2048, 994)=64+930`，改 99 等于换模型；本方案所有 obs 相关代码固定 93。
3. **不混用 third_party 权重与解析头**：mjlab 的权重只配 mjlab 的解析件（FK/奖励/终止/obs），Isaac 的权重只配 trackerlab 的解析件。历史上的 `trackerlab_bundle.py` 桥已经不存在，也不恢复。
4. **不改 upstream `sm-train` 的语义**：W1 只传参，不改 `training/loop.py`、`imagination/policy_trainer.py`。
5. **不碰 BONES-SEED 原始库**（`bones_seed_compact.pt`，6.7 GB）：W1 是"续训"，参考库换成 318。

### 0.3 五个需要拍板的决定（见 §11.D）

| 编号 | 决定 | 默认取值（推荐） |
|---|---|---|
| D1 | W1 从哪个 ckpt 续训 | `e9_bones_seed_token_h200x2/ckpt_0024000.pt`（318 时代联合微调用的就是它） |
| D2 | WBC 的条件 token | **直接用数据集录制 token**（hold 空间、逐帧，不重建；§5.2/§6.2） |
| D3 | 是否解锁 MuJoCo/318 | 是，需改 `docs/DATA_POLICY_HA_ONLY.md` 并加新的一节例外（§5.14） |
| D4 | J1 的 VLA 起点 | `flowbody/.../groot_sonic_vla_20k_frozen271_20261023/checkpoint-20000`（318 监督版） |
| D5 | J1 是否加手部监督 | 否（mjlab policy 无手部输出；`g1_hand_joints_14` 只对 VLA 头有意义，留作可选） |

---

## 1. 不变式（红线，改代码时不许破坏）

### 1.1 维度契约

| 量 | 值 | 定义位置 | 说明 |
|---|---|---|---|
| `STATE_DIM` | **71** | `diff_vla_wbc/rollout.py:31`、`world_model/state_spec.py:1-4` | `[p(3), quat wxyz(4), q(29), v_w(3), w_b(3), qd(29)]` |
| `OBS_DIM` | **93** | `world_model/heads.py:21-22`、`OBS_TERMS` | `[q-q0(29), qd(29), w_b(3), proj_grav(3), a_prev(29)]` |
| `term_dims` | **(29,29,3,3,29)** | `rollout.py:57` 的默认值 | 历史 99 维走 `relative_ori99`，**mjlab 一律用默认** |
| `ACTION_DIM` | 29 | `rollout.py:29` | |
| `HISTORY` | 10 | `rollout.py:30` | policy 输入 `10×93=930` |
| `token` | 64 | `policy/nets.py:32`（`token_dim`） | `PolicyCfg.token_dim=64` → decoder 首层 `64+930=994` |
| `WM state` | 70 差分量 / 71 状态 | `world_model/state_spec.py` | `explicit_integration=True`，`dt=0.02` |
| `xi` | 33 | `state_spec.py:57` | `[0]=0.75` 摩擦，`[1:30]` 编码器偏置，`[30:33]` CoM |
| 每步奖励上界 | **0.1** | `TrackingRewardCfg.max_step_reward`（`rewards.py:42-43`）= `dt*(1+1+1+1+0.5+0.5)` | 与 Isaac 的 `_MAX_STEP_REWARD=0.1` 一致，`train_v2.py:60` 的常量不用改 |

### 1.2 观测 93 的装配（**唯一实现**）

```python
# world_model/heads.py:30  analytic_observation(s, prev_action, xi, default_joint_pos)
q, qd, w_b = s[..., 7:36], s[..., 42:71], s[..., 39:42]
obs = cat([q + xi[...,1:30] - q0, qd, w_b, projected_gravity(s[...,3:7]), prev_action])  # (N,93)
# 历史 flatten（term-major）：world_model/heads.py:57 flatten_history_term_major
flat = cat([hist[:, :, a:b].permute(1,0,2).reshape(N, H*d) for each term])  # (N,930)
```

**红线**：joint FT 里 `bundle.act(obs_hist_flat, token)` 的 `obs_hist_flat` 必须由
`bundle.flatten_history(bundle.observation(...))` 产生；不许从 VLA 的 46 维状态拼接，也不许从 npz 原始
`g1_state_46` 硬拼（两者都不在 policy 的观测空间里）。

### 1.3 状态元组的角速度是**体系**（这是本方案最大的坑，见 §10 R3）

- MuJoCo 自由关节 `qvel` 的角速度 = **体系**（`envs/state_io.py:20-27` 直接 `qpos|qvel` 拼成 71）。
- WM 的 `state_spec` 也写成 `root_ang_vel_b`（`world_model/state_spec.py:3`）。
- 环境 obs 用 `base_mdp.base_ang_vel`（`envs/tracking_env_cfg.py:101`）= 体系。
- **但** 运动库存的是世界系：`RefState.root_ang_vel_w`（`motion_library.py:54`），`build_library.py`
  直接搬 npz 的 `body_ang_vel_w`。

⇒ 任何"从运动库造状态元组"的代码都必须做 `w_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)`。
upstream `SuperMimicBundle.reference_start` / `teacher_states`（`diff_vla_wbc/supermimic_backend.py:376-460`）
**没有做这个转换**，它自己的注释也写了"仅供 evaluation/smoke"。本方案的
`MjlabJointBundle` 必须转换，并在 M0 用探针证明（§9 M0-G2）。

### 1.4 动作语义（mjlab 侧固定，不许切换到 `a = Δq`）

```
q_target = default_joint_pos + action_scale * a        # diff_vla_wbc/supermimic_backend.py:87
```
`action_scale` 来自 `build_g1_assets` 的 `G1_ACTION_SCALE`（per-joint）。
历史教训（`docs/HANDOVER_SUPERMIMIC_FINETUNE.md:33-39`）：把 `a` 改成 `Δq` 会让 WM 的动作敏感度
与 policy 失配，policy rollout 从 2.87 崩到 29.6。**mjlab 路线禁止 `COMIMIC_ACTION_SCALE=one`。**

### 1.5 checkpoint 三种布局，互不兼容

| 布局 | 顶层键 | 谁产出 | 消费者 |
|---|---|---|---|
| **B1 mjlab 训练 ckpt** | `{iteration, num_updates, wm, policy, critic, wm_opt, policy_opt, critic_opt, bin_return, bin_memory, config}` | `sm-train` | `sm-train --resume`、`SuperMimicBundle.from_checkpoint` |
| **B2 联合微调 ckpt** | `{adapter, adapter_config, policy, wm, policy_config, world_model_config, config, steps}` | `diff_vla_wbc/train.py:JointTrainer.save` | `scripts/eval_mujoco.py --joint-ckpt` |
| **B3 Isaac 训练 ckpt** | `{pi, wm, args, iteration}` | trackerlab | `load_isaac_bundle` |

`is_mjlab_checkpoint()` 只能认 **B1/B2**（有 `policy`+`wm`），**不能**认 B3（有 `pi`）。
`is_isaac_checkpoint()`（`isaac_joint.py:644`）只能认 B3。两个判定必须在入口处互斥检查。

### 1.6 token 只有一个空间：hold（= `label`），**侧车已废弃**

Isaac 侧已经判过：`scripts/train_joint_v2.py:518` 的审计行写着
`S0.4: 标签==录制 encoder_latent；端口装配 corr=0.057 已退役`；S3 的锁定文档也禁止再喂
`data/port_tokens`。所以本方案**不引入任何侧车**，只有一个 token 空间：

| 名字 | 定义 | 生产者 |
|---|---|---|
| `label`（= hold） | `FSQ(MLP640)`；窗口 = **当前帧重复 10 次**；anchor = `rot6d(conj(q_robot)·q_ref)` | 部署前端（录制时算好，写进 HA npz 的 `g1_motion_token_64`；318 的对应列是 parquet `action.motion_token`） |

**为什么 mjlab 侧也能直接落进这个空间**（逐行核对过，不是推测）：

| 项 | mjlab | trackerlab | 结论 |
|---|---|---|---|
| rot6d | `supermimic/envs/mdp/sonic_tokens.py:210-222` | `trackerLab/algorithms/supermimic/sonic_token.py:124-136` | **实现完全一致**（旋转矩阵前两列） |
| 关节顺序 | `perm`（`sonic_tokens.py:113`） | `PERM_MJCF_TO_ISAACLAB`（`sonic_token.py:79`） | 同一个置换 |
| 编码器 | `default_encoder_path()` → `model_encoder.onnx` | 同名函数，同一文件 | 同一份权重 |
| FSQ | `round(tanh(x+off)*scale-bias)/16` | 同 | 一致 |
| **窗口** | `frame..frame+9`（未来 10 帧，`:170-176`） | `window="hold"`：`frame` 重复 10 次（`:229-232`） | **唯一差异** |

⇒ **端口侧车当初存在，只是因为 mjlab 的 WBC 用了 future 窗口。W1 改成 hold 训练之后，
WBC 的输入空间就是标签空间，侧车从根上不需要。**

前瞻在哪：单个 token 是 hold（窗口内没有未来帧），**前瞻来自 token 序列**——一次给 40 个
token（`token_horizon=40`，0.8 s），每个都是各自时刻的 hold token
（`sonic_token.py:243-247` 的 `k_window` 注释）。这也正是"HA 是 hold 又有前瞻"的含义。

### 1.7 两个解释器，不许混跑

| 阶段 | 解释器 | 理由 |
|---|---|---|
| W1（WBC 续训） | `third_party/SuperMimic-Reproduced/.venv/bin/python`（torch 2.14.1+cu126 / mujoco 3.16 / mjlab 1.6.0 / warp 1.17） | mjlab + MuJoCo-Warp 只能在这里 |
| J1 / E1（联合微调、评测） | 系统 `python3` + `PYTHONPATH=.../third_party/gr00t_deps:.../? `（见 §7.5 命令） | GR00T/transformers 与 mjlab venv 的 torch 版本不兼容 |

`diff_vla_wbc/compat.py:201 ensure_supermimic_importable(lightweight=True)` 提供了 mjlab 的
轻量 shim + `mjlab.utils.lab_api.math` 假模块，所以系统 python 可以直接 import `supermimic.*` 的
纯 torch 模块（`policy`/`world_model`/`kinematics`/`envs.mdp`），**不需要装 mjlab**。

---

## 2. 现状盘点

### 2.1 代码

| 文件 | 职责 | 本方案怎么用 |
|---|---|---|
| `diff_vla_wbc/supermimic_backend.py` | mjlab 后端：`G1Assets` / `SuperMimicBundle` / `build_g1_assets` / `load_motion_library` | **复用**（`MjlabJointBundle` 包它） |
| `diff_vla_wbc/mujoco_env.py` | CPU MuJoCo 闭环（`mj_step` + PD） | 复用（E1） |
| `diff_vla_wbc/train.py` | `JointTrainer`（318 时代的联合微调；**会训 WBC policy**） | 保留为对照组（§8.3），不作主线 |
| `diff_vla_wbc/isaac_joint.py` | Isaac 后端 bundle + `canonical_to_flat` 等 | **只作接口模板**；`canonical_to_flat` 需提到公共模块 |
| `diff_vla_wbc/train_v2.py` | `JointTrainerV2`（只训 VLA head） | **移植主体**，改 6 处解耦 |
| `diff_vla_wbc/v3_step.py` | `JointTrainerV3`（v2 + critic/anchor/trust region/DR/burn-in/WM 保鲜） | **移植主体**，改 3 处解耦 |
| `diff_vla_wbc/rollout.py` | 可微想象 rollout + `freeze_world_model` | 复用（默认 93 已经完全对上） |
| `diff_vla_wbc/burnin.py` | 真录制状态 burn-in + 扰动 | 改 2 处（`bundle.mdp` / `motion.ref` 索引） |
| `diff_vla_wbc/data.py` | `MotionBatch` / `MotionBatchProvider` / `RecordedTokenSource` | 复用（已与后端解耦，lazy import） |
| `diff_vla_wbc/fm_bc.py` / `fm_grad.py` / `lora.py` / `token_critic.py` / `dr.py` / `wm_refresh.py` / `prefetch.py` | BC 损失、FM 梯度后端、LoRA、critic、域随机化、WM 保鲜、预取 | **全部复用，一行不改** |
| `diff_vla_wbc/action_codec.py` / `gr00t_vlm.py` | VLA 侧 codec / GR00T 包装 | 复用 |
| `scripts/train_joint_v2.py` | Isaac 联合微调入口（CLI 全表） | **模板**，新入口照抄（§5.5） |
| `scripts/train_joint.py` | 旧入口：`JointTrainer` + **Isaac-only 门禁**（`:283-289`） | 门禁要允许 `--backend mjlab`（§5.12） |
| `scripts/eval_mujoco.py` / `closed_loop/code/closed_loop_mujoco.py` | MuJoCo 评测 | 复用（E1） |
| `scripts/build_port_tokens.py` | trackerlab 端口侧车 | **不改、不用**（侧车已废弃，§1.6） |
| `third_party/SuperMimic-Reproduced` | mjlab 训练器 + policy + WM + 解析件 | 只调用，不改（除非 D5 要手部） |

### 2.2 权重

| 用途 | 路径 | 关键事实 |
|---|---|---|
| mjlab WBC 底模 | `third_party/SuperMimic-Reproduced/logs/e9_bones_seed_token_h200x2/ckpt_0024000.pt`（~191 MB） | 布局 B1；`policy.token_dim=64`；参考库 `bones_seed_compact.pt`；obs 93 |
| mjlab WBC 最新 | 同目录 `ckpt_0156000.pt` / `ckpt_latest.pt` | 训练到 ~156k iter |
| mjlab 联合微调产物 | `runs/joint_sm24k_hand/joint_001000.pt`、`runs/joint_diff_ft318/joint_*.pt` | 布局 B2；policy 已被反传改过（10/22 张量，max|Δ|=0.0054 / 0.0894） |
| Isaac WBC（对照） | `trackerlab-supermimic/logs/wbc_*/ckpt_*.pt` | 布局 B3；其中 `wbc_mix_obs93` 用 `bones_seed_plus_ha_x20.pt`、`wbc_ha_obs93` 用 `runs/isaac_motion/ha_tokens/packed.pt` |
| VLA（318 监督） | `/inspire/.../flowbody/FlowBody/runs/groot_sonic_vla_20k_frozen271_20261023/checkpoint-20000` | GR00T N1.7，20k 步 |
| VLA（HA 监督，Isaac 主线） | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000` | 不动 |

### 2.3 数据

| 数据集 | 规模 | 字段 | 本方案用途 |
|---|---|---|---|
| `data/npz/g1_tasks` | **318 clips / 473,876 帧 / 2.633 h / 4 任务** | `joint_pos(29)`, `joint_vel(29)`, `body_pos_w(30,3)`, `body_quat_w(30,4)`, `body_lin_vel_w`, `body_ang_vel_w`, `fps=50`, `g1_state_46`, `g1_motion_token_64`, `g1_hand_joints_14`, `g1_ego_view`, `g1_instruction` | W1 参考库 + J1 参考库/标签 |
| `data/npz/g1_tasks_ft318` | 318（同批命名副本） | 同上 | 同上（去重后只留一个） |
| `data/npz/ha_tokens` | 699 clips / 634,016 帧 | 同上 + HA 标签 token | 不动 |
| `third_party/.../data/library/bones_seed_compact.pt` | 6.7 GB fp16，71 维 | compact | W1 的起点库（不重训） |
| `data/port_tokens/ha_tokens` | HA 的 trackerlab 端口侧车 | (T,64) | **已废弃**（Isaac 判定 corr 0.057）；本方案不产出、不消费 |

### 2.4 环境

| 组件 | 状态 |
|---|---|
| mjlab venv | ✅ `.venv/lib/python3.12/site-packages`: `mjlab-1.6.0`, `mujoco-3.16.0`, `torch-2.14.1+cu126`, `warp_lang 1.17` |
| SONIC encoder ONNX | ✅ `third_party/SuperMimic-Reproduced/data/assets/model_encoder.onnx`（50 MB） |
| G1 MJCF + mesh | ✅ `scripts/setup_assets.sh` 建立 `g1/meshes` 软链 |
| GR00T 依赖 | ✅ `third_party/gr00t_deps`（系统 python 走 PYTHONPATH） |

---

## 3. 差距表（Isaac 有 → mjlab 缺什么）

| # | 能力 | Isaac 实现 | mjlab 现状 | 迁移动作 |
|---|---|---|---|---|
| 1 | bundle 统一接口 | `isaac_joint.IsaacJointBundle`（`:224-611`） | `SupermimicBackend.SuperMimicBundle`（`:174-460`）**接口 80% 已对齐** | 新增 `mjlab_joint.MjlabJointBundle` 包一层（§5.1） |
| 2 | `.mdp`（给 burn-in 造 obs） | `trackerlab` MDP | 无 | `MDPShim`，走 `analytic_observation`（§5.1） |
| 3 | `.motion`（参考库） | trackerlab library | `bundle.library`（mjlab `MotionLibrary`，API 不同） | 属性 `motion` 返回 library + `ref_joint_pos()` 适配（§5.1） |
| 4 | `.spec`（robot spec 快照） | `robot_spec_snapshot` | 无 | 用 `asdict(assets)` 的最小快照（§5.1） |
| 5 | `.teacher_goal` token 路由 | 直接 raise | 走 `library.goal_window`（680 维） | token 路由下返回 `None`（`data.py:404` 已判 `token_dim`） |
| 6 | 状态元组角速度 | world（Isaac 库）→ 由 bridge 转换 | **world 未转换** | 本方案在 bundle 内做转换（§1.3） |
| 7 | 训练入口 | `scripts/train_joint_v2.py` | 无（旧 `train_joint.py` 是 `JointTrainer` 且 Isaac-only） | 新增 `scripts/train_joint_mjlab.py`（§5.5） |
| 8 | ~~端口 token 侧车~~ | `scripts/build_port_tokens.py` | 不需要 | **删除**：hold 口径下标签空间即 WBC 空间（§1.6） |
| 9 | 条件 token 来源 | 录制标签（trackerlab 原生支持 `*.tokens.npy`） | mjlab 只能在线算 | 新增 `mjlab_tokens.py` + `--data.tokens`（§5.2） |
| 10 | 真转移（B 线采集） | `scripts/build_real_transitions*.py` + Isaac 采集 | 无 | M4 阶段复刻（§7.4），本方案先跑通"无真转移"版 |
| 11 | 闭环评测 | Isaac Sim | MuJoCo（`mujoco_env.py` / `closed_loop/`） | 复用 + 新 joint ckpt 加载（§5.13） |
| 12 | 政策/门禁 | `_forbid_legacy_dataset`、`is_isaac_checkpoint` | 同一套（禁止 mjlab/318） | 加例外条款（§5.14） |

---

## 4. 目标架构

### 4.1 数据流

```
W1  ┌ data/npz/g1_tasks (318) ──build_library.py──> data/library/ft318_compact.pt (fp16, 71d)
    │                                                     │
    └ e9/ckpt_0024000.pt ──sm-train --data.library ... --resume ...──> logs/wbc_ft318_mjlab/ckpt_*.pt
                                                                         │  (B1 布局, obs93)
J1  ┌ VLA(318 20k 或 HA-8000) ──LoRA(head)──> tokens(64)                 │
    │        │                                    │                     │
    │        └─ BC 标签: npz g1_motion_token_64（hold，无侧车）          │
    │                                             ▼                     ▼
    │                        imagine_with_tokens(bundle=MjlabJointBundle)
    │                            a = policy(obs_hist_93, token)   [冻结]
    │                            s' = wm.step(s, a, xi)           [冻结]
    │                            r  = heads.reward(..)            [解析]
    │                            V  = TokenCritic(s, ref, xi)     [可训]
    │                                             │
    │                    loss = λ_bc·FMBC + λ_r·(−R̂) + λ_hand·hand + λ_a·hinge + λ_a'(anchor)
    └──────────────────────────── grad → VLA action head ────────────────┘
E1  joint_*.pt + vla_*.pt ──export_vla_ckpt──> MuJoCo 闭环 (93 obs) ──> qc_*.json / mp4
```

### 4.2 分层（与 Isaac 完全同构，只是把后端换掉）

| 层 | Isaac | mjlab（本方案） |
|---|---|---|
| 执行器（冻结） | `IsaacJointBundle`（trackerlab） | `MjlabJointBundle`（mjlab） |
| 可微动力学（冻结） | `wm`（trackerlab，explicit_dt0.02） | `wm`（mjlab，explicit_dt0.02） |
| 解析件 | trackerlab reward/term/FK | mjlab `AnalyticHeads` / `G1Kinematics` |
| VLA（可训 head） | `Gr00tVLM` + LoRA | **同一份代码** |
| 训练器 | `JointTrainerV2/V3` | **同一份代码**（解耦 6+3 处） |
| 数据 | `MotionBatchProvider` + `RecordedTokenSource` | **同一份代码** |

> 设计原则：**只替换 bundle**。`train_v2/v3` 不出现任何 `mjlab`/`trackerlab` 字样（现在也没有），
> 因此迁移成本集中在 `mjlab_joint.py` 这一个文件。

---

## 5. 代码改动清单

> 约定：`新增` = 新文件；`改N处` = 最小手术式改动；每处都给出锚点（`文件:行`）与前后差异。

### 5.1 新增 `diff_vla_wbc/mjlab_joint.py`（**核心，~260 行**）

作用：把 mjlab 的 `SuperMimicBundle` 适配成 `train_v2/v3/rollout/burnin/data` 需要的鸭子类型，
并**修掉 §1.3 的角速度帧问题**。

#### 5.1.1 常量与判定

```python
OBS_DIM = 93                                   # 与 world_model/heads.py:21 OBS_DIM 一致，禁止改
TERM_DIMS = (29, 29, 3, 3, 29)                 # 与 rollout.py:57 默认值一致
STATE_DIM = 71
NUM_JOINTS = 29
XI_DIM = 33


def is_mjlab_checkpoint(path) -> bool:
    """B1（sm-train）或 B2（联合微调）布局：有 policy + wm 且**没有** pi。

    B1: {iteration, num_updates, wm, policy, critic, config, ...}
    B2: {adapter, policy, wm, config, steps, ...}
    """
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    return ("policy" in ck and "wm" in ck) and ("pi" not in ck)
```

#### 5.1.2 状态元组的角速度归一化（关键修正）

```python
def _ref71_body(library, frames: torch.Tensor) -> torch.Tensor:
    """库里的 71 维参考 → **MuJoCo 状态语义**（角速度换到体系）。

    library.ref71 的 [39:42] 是世界系（body_ang_vel_w）；state_spec/obs/仿真状态都是体系。
    上游 reference_start/teacher_states 没做这一步（supermimic_backend.py:376-460），
    直接拿来用会让 obs 的 base_ang_vel 与训练分布错位（§10 R3）。
    """
    r = library.ref71(frames)                                  # (N,71) fp32
    q = r[:, 3:7]
    w_b = quat_apply_inverse(q, r[:, 39:42])                   # world -> body
    return torch.cat([r[:, :39], w_b, r[:, 42:71]], dim=-1)
```

#### 5.1.3 `MjlabJointBundle`（接口表）

| 成员 | 签名 | 实现要点 |
|---|---|---|
| `policy` / `wm` | — | 直接来自 `SuperMimicBundle` |
| `assets` | `G1Assets` | `heads`（FK/奖励/终止）、`action_scale`、`default_joint_pos` |
| `device` | property | `assets.device` |
| `motion` | property → `MotionLibrary` | 同名属性给 `v3_step` / `LibraryTokenBuilder` 用 |
| `spec` | dict | `{"robot":"g1","joints":29,"obs":93,"token":64,"action_scale":[...]}`（只做审计） |
| `num_joints` | int | 29 |
| `ref_joint_pos(frames)` | `(N,29)` | `library.ref71(frames)[:, 7:36]`（**Isaac 是 `[:,13:42]`**） |
| `observation(state, a_prev, xi)` | `(N,93)` | `analytic_observation(...)`；断言 `shape[-1]==93` |
| `flatten_history(hist)` | `(B,930)` | `flatten_history_term_major(hist)` |
| `act(obs_hist_flat, token)` | `(B,29)` | `policy(obs_hist_flat, token, interface="token")`；断言 `obs_hist_flat.shape[-1]==930` |
| `step(state, a, xi)` | `(B,71)` | `wm.step` |
| `reward(state, a, a_prev, ref, bodies=None)` | dict | `heads.reward(...)`；返回 `{"total":...}`（与 Isaac 同形） |
| `termination_logits(state, ref)` | `(B,)` | `wm.termination_logits` |
| `robot_bodies(state)` | dict | `heads.robot_bodies`（可微 FK） |
| `reference_start(library, frames, history=10, xi=None)` | dict | **用 `_ref71_body` 造 states**；action 用 `(q_ref-q0)/action_scale`；返回键与 Isaac 完全一致：`state/hist_states/hist_actions/observations/obs_hist/prev_action/xi` |
| `teacher_states(library, frames, horizon)` | `(B,H,71)` | 同上，`_ref71_body` |
| `teacher_references(library, frames, horizon)` | `list[ReferenceFrame]` | `library.bodies(idx)`（compact 模式自动走 FK） |
| `teacher_goal(library, frames)` | `None` | token 路由下不造 680 维目标 |
| `mdp` | property → `MDPShim` | `mdp.obs(states, actions, xi, frames)` → `observation(...)`（给 `burnin.py:193`） |
| `save_checkpoint(path, **extra)` | — | B2 布局：`{policy, wm, config:{policy,world_model}, adapter?, ...extra}` |
| `reference_frame(library, frame)` | `ReferenceFrame` | 直接转 upstream |

#### 5.1.4 加载器

```python
def load_mjlab_bundle(ckpt, *, device="cuda", npz=None, packed=None, token_dim=64,
                      xml_path=None) -> MjlabJointBundle:
    ensure_supermimic_importable(lightweight=True)      # compat.py:201（带 mjlab shim）
    from .supermimic_backend import SuperMimicBundle, build_g1_assets, load_motion_library

    assets = build_g1_assets(xml_path, device)
    base = SuperMimicBundle.from_checkpoint(ckpt, assets, device=device)   # 只走 B1/B2 分支
    if int(base.policy.token_dim) != int(token_dim):
        raise SystemExit(f"{ckpt}: token_dim={base.policy.token_dim} != {token_dim}")
    library = load_motion_library(npz, assets, compact=bool(packed), device=device) if npz else None
    return MjlabJointBundle(base, library, assets)
```

> ⚠️ `SuperMimicBundle.from_checkpoint` 里有一条通向已删除模块的分支
> （`supermimic_backend.py:253` → `trackerlab_bundle`）。本方案**不修复**它，只在
> `load_mjlab_bundle` 里先用 `is_mjlab_checkpoint()` 挡住 B3，避免落到那条 ImportError。

#### 5.1.5 `MDPShim`

```python
class MDPShim:
    """burnin.py 只用到 bundle.mdp.obs(...)，这里给最小面。"""
    def __init__(self, bundle): self._b = bundle
    def obs(self, states, actions, xi, frames=None):
        return self._b.observation(states, actions, xi)     # (N,93)，忽略 frames
```

### 5.2 给 mjlab 训练器装一个「读数据集 token」的源（**不需要任何 token 前端**）

**先纠正一个设计**：既然 318 已经带**逐帧的部署 token**（`g1_motion_token_64`，与 parquet
`action.motion_token` 逐位相等，见 §6.1），就**不该在线重算**。在线重算需要命令流（318 没有）
和仿真 anchor（会漂），是纯粹的损失。用录制 token 有两个好处：

1. **token 空间精确**：喂给 WBC 的就是部署时喂的那 64 维，不存在「近似标签空间」；
2. **少一个模块**：不需要 `sonic_frontend_mjlab.py`，也不需要改 upstream 的窗口语义。

唯一代价：录制 token 的 anchor 编码的是**录制时的机器人姿态**，训练中若给 sim 机器人加姿态扰动
（reset noise / DR），这个 anchor 会「过期」（§10 R13）。处置见 §6.2。

#### 5.2.1 数据侧（已产出 ✅）

```
data/library/ft318_tokens.pt   60.7 MB
  tokens: (473876, 64) fp16      # 与 ft318_compact.pt 的全局帧轴逐帧对齐
  names:  318 个 clip 名          # 与库 names 完全相同、同序（已断言）
  starts/lengths: 同库
  space: "label/hold"
```

#### 5.2.2 代码侧：`diff_vla_wbc/mjlab_tokens.py`（~60 行）+ upstream 3 处小改

```python
class LibraryTokens:
    """按库帧号直读录制 token；签名与 upstream SonicTokenBuilder 一致，便于替换。"""

    def __init__(self, path, device, num_future: int = 10):
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.tok = d["tokens"].to(device)                     # (N,64)
        self.n = int(self.tok.shape[0])

    def __call__(self, frame, ref=None, state=None, robot_quat=None):
        return self.tok[frame].float()                        # (B,64)；忽略 ref/state
```

upstream 改动（3 处，~10 行，默认行为不变）：

| 文件 | 改动 |
|---|---|
| `config.py:DataCfg` | 新增 `tokens: Path \| None = None`（`--data.tokens`） |
| `training/loop.py:89-96` | `if cfg.data.tokens is not None: token_fn = LibraryTokens(...)`，否则原逻辑 |
| `training/loop.py:152-159` | `token_builder` → 统一成 `self.token_source` |
| `imagination/rollout.py:48` | 不用改（`token_fn(frame, ref, s)` 签名已兼容） |

**注意**：这不是「侧车」。侧车是**另算一套空间**的派生数据（corr 0.057，已废弃）；
这里是**数据集自带的标签**，与 BC 用的是同一个东西（§1.6）。

### 5.3 新增 `scripts/prepare_wbc_resume.py`（W1 的第一道坎）

`sm-train --resume` 会把 ckpt 里的 bin 采样表原样拷回（`training/loop.py:143-145`）：
```python
self.command.bin_return.copy_(ck["bin_return"])
self.command.bin_memory.copy_(ck["bin_memory"])
```
而 bin 数由**参考库**决定（`commands.py:70-71`，`bin_frames=50`；公式是
`bins_per_clip = (len + 49) // 50`，见 `motion_library.py:160-163`——**不是** `ceil((len-1)/50)`，
后者是 `merge_libraries.py` 的粗估，会少算，实测 9,621 vs 正确 9,625，直接导致 size mismatch 报错）：

| 库 | clips | frames | num_bins |
|---|---|---|---|
| `bones_seed_compact.pt`（e9 训练库） | 129,943 | 47.5 M | **1,013,724**（已核对 e9 ckpt 的 tensor 形状） |
| `ft318_compact.pt`（本方案，已产出） | 318 | 473,876 | **9,625** |

⇒ 直接 `--resume e9/ckpt_0024000.pt` 会在 `load()` 里 **size-mismatch 崩掉**。

本脚本只改 6 个键，其余逐位保留：
```python
ck = torch.load(src, map_location="cpu", weights_only=False)
nb = sum((int(L) + 49) // 50 for L in library["lengths"])   # ceil(L/50)，不是 ceil((L-1)/50)
ck["bin_return"] = torch.full((nb,), 10.0)      # commands.py:415 g_max=10.0
ck["bin_memory"] = torch.full((nb,), 1.0 / nb)
ck["iteration"] = 0; ck["num_updates"] = 0      # 让 --schedule.total-iterations 是绝对语义
torch.save(ck, out)                             # 优化器状态保留（与库无关，可复用）
```
产出 `runs/wbc_resume/e9_0024000_ft318.pt`（~191 MB）。**原 ckpt 不动，可回滚。**

### 5.4 新增 `scripts/build_mjlab_library.py`（薄封装，~80 行）

只是给 upstream `scripts/data/build_library.py` 加校验与 provenance：

```
python scripts/build_mjlab_library.py --npz-dir data/npz/g1_tasks --out data/library/ft318_compact.pt
```
校验项（缺一即失败）：
1. 每个 npz 有 `body_names` 且含 `pelvis`（318 实测 ✅ 30 个 body，第 1 个就是 pelvis）；
2. `fps==50`；
3. `Σ joint_pos.shape[0] == 473876`（与 `docs/FINAL_HANDOVER.md:74` 的锁定值逐帧一致）；
4. 产物 `ref.shape == (473876, 71)`，`names` 长度 318；
5. 写 `ft318_compact.provenance.json`（源目录、文件数、帧数、sha256、mjlab commit）。

### 5.5 新增 `scripts/train_joint_mjlab.py`（~700 行，**从 `train_joint_v2.py` 派生**）

保留 `train_joint_v2.py` 全部逻辑，只替换 4 个装配点：

| 装配点 | `train_joint_v2.py` | `train_joint_mjlab.py` |
|---|---|---|
| bundle | `load_isaac_bundle(...)` | `load_mjlab_bundle(...)`（§5.1） |
| 参考库 | `--motion-packed runs/isaac_motion/ha_cmd/packed.pt` | `--motion data/npz/g1_tasks`（**full 库**，参考帧/body 都要） + 可选 `--motion-packed data/library/ft318_compact.pt` |
| token 侧车 | `--port-tokens data/port_tokens/ha_tokens` | **删除该参数**（hold 口径 ⇒ 直接用 npz 标签） |
| 后端检查 | `is_isaac_checkpoint` | `is_mjlab_checkpoint`（互斥报错） |

**CLI 与 v2 逐项对齐**（名字/默认值都照抄，避免两套心智模型）：
```
--wbc-ckpt --vlm-ckpt --vlm-path --vlm-select-layer --motion --motion-packed --port-tokens
--bc-token-space {label,port}          # mjlab 入口默认 label；`port` 仅供历史复现
--wbc-token-space {auto,label,port} --wbc-token-window {hold,future}
--out --steps --batch-size --grad-accum --horizon --gamma --lambda-bc --lambda-r --lambda-s --lambda-a
--lora-rank --lora-alpha --lora-targets --amp {off,bf16} --fm-lr --fm-steps --clip-head
--v3 --critic --no-critic --critic-lr --lam --critic-source {imagined,real}
--dr --no-dr --dr-xy-max --dr-yaw-max --dr-dq-max --dr-dqd-max --dr-p-nominal --dr-p-mid
--push-std --push-ang --anchor-weight --trust-eps --trust-weight --trust-center
--act-eps --act-weight --act-hinge-cap --start-source {reference,recorded} --real-transitions
--wm-refresh-every --wm-refresh-steps --wm-refresh-lr --eval-every --eval-batches --eval-out
--save-every --save-vla-every --gate-only --gate-out --grad-log-every --state-vel-weight
```
新增 3 个 mjlab 专属参数：
```
--xml-path            默认 third_party/SuperMimic-Reproduced/.../g1.xml（由 setup_assets.sh 保证）
--state-angvel-frame  {body,world}  默认 body；world 只用于复现上游 reference_start 的旧行为
--wbc-action-scale    {mjlab,one}   默认 mjlab；one 仅用于对照实验（§1.4 禁止默认用 one）
```

**门禁照搬**（`train_joint_v2.py` 已有的 4 道）：
1. `token_space` 不一致且 `lambda_r>0` → `SystemExit`（`:533-545`）；
2. `motion` 含 `mixed/ft318` → 由政策决定是否放行（§5.14 会改成白名单）；
3. `frame_index.num_frames != library.num_frames` → 退出（`train_joint_v2.py:381-383`）；
4. `--gate-only` 四道自检（codec / 冻结 / burn-in / FM-BC）。

### 5.6 新增 3 个 launcher

| 脚本 | 作用 |
|---|---|
| `scripts/launch_wbc_ft318_mjlab.sh` | W1：建库（若缺）→ `sm-train` 续训 → 打印验收数字 |
| `scripts/launch_joint_mjlab_ft318.sh` | J1：两阶段（BC warmup 800 步 → +reward 2000 步），照 `launch_joint_ha_v2.sh` 的结构 |
| `scripts/run_e1_mujoco_qc.sh` | E1：4 任务 × 3 seeds 闭环 + 视频 + `qc_*.json` |

launcher 内必须写死：
```bash
VENV=$COMIMIC/third_party/SuperMimic-Reproduced/.venv/bin/python   # W1 用
PY=$COMIMIC/third_party/gr00t_deps/../..                            # J1/E1 用系统 python3
export SONIC_ENCODER_ONNX=$REPO/data/assets/model_encoder.onnx
export MUJOCO_GL=disable
```

### 5.7 改 `diff_vla_wbc/train_v2.py`（6 处）

| # | 锚点 | 现状 | 改成 |
|---|---|---|---|
| 1 | `:60 _MAX_STEP_REWARD = 0.1` | 常量 | `MAX_STEP_REWARD` 可从 `bundle.max_step_reward` 读，缺省 0.1（mjlab 恰好也是 0.1，见 §1.1） |
| 2 | `:374 extra["robot_spec"]` | 直接 `_rm.robot_spec_snapshot(bundle.spec)` | `getattr(bundle, "spec", None)` 为空则跳过 |
| 3 | `:110 _freeze_executor` | 遍历 policy/wm | 保持；额外断言 `bundle.observation(...).shape[-1]==93` |
| 4 | `:280 _rollout` | `imagine_with_tokens(bundle, ...)` | 保持；`term_dims=getattr(bundle,"term_dims",(29,29,3,3,29))` 显式传 |
| 5 | `:296-302 _state_error` 调用 | 用 `batch.teacher_states` | 保持（mjlab 的 teacher_states 已是体系，见 §5.1.2） |
| 6 | `:307 self.bundle.act(oh, tokens[:,0])`（`lambda_a` 分支） | 直接用 | 保持；`oh` 必须是 930 维（加断言） |

### 5.8 改 `diff_vla_wbc/v3_step.py`（3 处）

| # | 锚点 | 现状 | 改成 |
|---|---|---|---|
| 1 | `:138 motion = self.bundle.motion` | 假定 trackerlab library | `bundle.motion` 属性（mjlab 也提供，§5.1.3） |
| 2 | `:249 motion = self.bundle.motion` | 同上 | 同上 |
| 3 | `:195 from .trackerlab_adapter import quat_to_mat` | 依赖 Isaac 侧数学库 | 复制到 `diff_vla_wbc/rotations.py`（新增 8 行）或改用 `supermimic.kinematics.rotations.matrix_from_quat` |

> `v3_step.py` 里其余对 `motion` 的使用（`clip_start` / `clip_id` / `ref`）必须改走
> bundle 适配方法：`bundle.clip_start_of(frames)` 与 `bundle.ref_joint_pos(frames)`（§5.1.3 已给）。

### 5.9 改 `diff_vla_wbc/burnin.py`（2 处）

| # | 锚点 | 现状 | 改成 |
|---|---|---|---|
| 1 | `:190 q_cmd = motion.ref[hrows, 13:13+29]` | 写死 Isaac 库布局（13 起） | `q_cmd = bundle.ref_joint_pos(hrows)` |
| 2 | `:193 obs = bundle.mdp.obs(...)` | 假定有 `.mdp` | `mdp = getattr(bundle,"mdp",None)` 为空则 `bundle.observation(states.reshape(-1,71), a_hist.reshape(-1,29), xi_rep)` |

> `burnin.py` 的 `clip_first64 = motion.clip_start[motion.clip_id[f0]]`（:184）也要换：
> mjlab 有 `clip_start` 与 `frame_clip`，适配成 `bundle.clip_start_of(f0)`。

### 5.10 改 `diff_vla_wbc/rollout.py`（2 处）

| # | 锚点 | 现状 | 改成 |
|---|---|---|---|
| 1 | `:55 _obs_terms_from_flat` 默认 term | `(29,29,3,3,29)` | 保持默认；从 `bundle.term_dims` 取，缺省同值 |
| 2 | `bundle.assets`（`:97` 附近注释 + head 维） | 假定 Isaac assets | `getattr(bundle,"assets",None)`；为空用 `bundle.robot_bodies` |

> ⚠️ `rollout.py:97` 的注释说"trackerlab bridge 会把 world ω 转 body"。**mjlab 没有这个 bridge**，
> 所以 §5.1.2 的转换必须由 bundle 自己做，不能指望 rollout。

### 5.11 改 `diff_vla_wbc/data.py`（1 处）

| # | 锚点 | 现状 | 改成 |
|---|---|---|---|
| 1 | `:477 _reference_state_at` | `library.ref(frames)` 属性式 | 已兼容 mjlab（`MotionLibrary.ref` 返回 `RefState` 同名属性）✅ **无需改**；只把 `_reference_state_at` 的 `[39:42]` 也走 `_ref71_body`（与 §5.1.2 一致） |

### 5.12 改 `scripts/train_joint.py`（门禁）

| # | 锚点 | 现状 | 改成 |
|---|---|---|---|
| 1 | `:54 ISAAC_SM_CKPT` + `:283-289 is_isaac` 硬门禁 | 拒绝一切非 trackerlab ckpt | `--backend {isaac,mjlab}`，默认 `isaac`；`mjlab` 时走 `load_mjlab_bundle` 并保留 `SystemExit` 提示 |
| 2 | `:60 _forbid_legacy_dataset` | 禁 `mixed/ft318/g1_tasks` | 改为**白名单**：`--backend mjlab` 时允许 `g1_tasks`（318），仍禁 `mixed*` |
| 3 | 文件头注释（"只有一条线路：Isaac Lab"） | — | 更新为两条线 + 各自口径 |

> 保守做法：**只加 `--backend mjlab` 开关**，不动默认行为——Isaac 主线的任何脚本/CI 都不会受影响。

### 5.13 改评测入口

| 文件 | 改动 |
|---|---|
| `scripts/eval_mujoco.py` | `--joint-ckpt` 支持 B2（已支持）；新增 `--expect-obs 93` 断言；`--token-source {label,online}` |
| `closed_loop/code/closed_loop_mujoco.py` | 读 `runs/OFFICIAL_*` 的速度/动作口径；新增 `--wbc-ckpt` 显式传（不再默认 e9） |
| `scripts/success_rate.py:543` | MuJoCo 门禁：允许 E1 的评测（`COMIMIC_ALLOW_MUJOCO_ROUTE=1` 保持显式） |

### 5.14 政策与文档（**必须先做，否则后面每一步都会被门禁拦住**）

| 文件 | 改动 |
|---|---|
| `docs/DATA_POLICY_HA_ONLY.md` | 追加 §7「MuJoCo/318 例外」：<br>① 允许 `--backend mjlab` 使用 `data/npz/g1_tasks`（318）与 mjlab 权重；<br>② 仅限"MuJoCo 路线"，Isaac 主线仍永久禁止 mjlab 权重与 318；<br>③ 两条线的产物必须带 `backend` 字段，禁止交叉喂 token；<br>④ **`mixed_ha_ft318` 仍永久禁止**（混合库会让口径不可归因） |
| `runs/OFFICIAL_SUPERMIMIC.json` | 追加 `mjlab` 段：`{"checkpoint":"...sm-train 产物","obs":93,"token_space":"label(hold)","action_scale":"mjlab"}` |
| 新增 `runs/OFFICIAL_SUPERMIMIC_MJLAB.json` | 与上面同源，供脚本读取（不改旧文件的消费方） |

### 5.14b M1 落地记录（2026-10-28，已实测）

| 项 | 结果 |
|---|---|
| 新增 | `diff_vla_wbc/mjlab_joint.py`（`MjlabJointBundle` / `MDPShim` / `load_mjlab_bundle` / `is_mjlab_checkpoint` / `_ref71_body`） |
| 改动 | `burnin.py` 4 处（`bundle.default_xi/clip_start_of/ref_joint_pos/obs_from_states`）；`isaac_joint.py` +4 个通用方法（纯新增） |
| **不需要改** | `train_v2.py`、`v3_step.py`、`rollout.py`、`data.py` —— 它们早已鸭子类型化；`bundle.spec` 的 `robot_spec` 快照有 try/except，mjlab 侧只打一条 warning |
| 端到端冒烟（CPU，318 全库） | 加载 10.6 s；`MotionLibrary(full): 318 clips, 473876 frames, 9625 bins`；batch `(4,71)/(4,930)/(4,40,64)`；rollout `states (5,4,71)`、`reward 0.088`、`surv 1.0`；**token.grad 非零（范数 0.17）** |
| 回归测试 | `tests/test_mjlab_joint_grad.py` 3 passed（obs 93 / 体系角速度 / 奖励反传），4.5 s |
| 全量测试 | 126 passed, 3 skipped, **1 failed** —— `test_joint_gradient.py::test_real_gr00t_head_to_supermimic`（`proj_out_2.weight.grad is None`）。该测试依赖的文件（`train.py`/`fm_grad.py`/`gr00t_tiny.py`/`adapter.py`）**均无本地改动**，且失败点在 GR00T head 的梯度，与 mjlab 迁移无关；另有一次运行报 `No module named 'tyro'`（环境缺 `gr00t_deps`），属环境问题 |

#### 5.14b.1 顺带发现的数据性质（影响 J1 的验收口径）

`scripts/prepare_g1_tasks_npz.py:215` 把根的线速度/角速度**写死为 0**：

```python
qvel = np.concatenate([zero_root, zero_root, joint_vel], axis=1)   # root lin/ang vel = 0
root_pos = np.concatenate([np.zeros((frames, 2), np.float32), root_z[:, None]], axis=1)  # root xy = 0
```

实测 318 npz： pelvis 的 `body_lin_vel_w` / `body_ang_vel_w` **恒为 0**，其他 body 正常；
根高度用"最低踝贴地"启发式合成（`g1_root_height_mode=foot`）。

后果与对策：

| 影响 | 说明 | 对策 |
|---|---|---|
| W1 训练 | WM 的转移来自**仿真**（不是库），所以 v/w 通道不退化 ✅ | 照常跑 |
| 奖励 | `motion_body_lin_vel/ang_vel` 对 pelvis 的参考速度是 0 → 轻微偏置"根别动" | 这 4 个任务是站姿操作，可接受；若后续做行走任务必须补根速度 |
| J1 的 state 项 | `teacher_states` 的 `[36:42]` 全 0 | **不要开 `--lambda-s`**（默认 0）；验收不许用速度维度 |
| 参考根 xy | 恒为 0（相对原点） | `motion_global_root_pos` 奖励等价于"别漂太远"，符合当前任务 |

> 如果以后要修：从 `body_pos_w[:, 0]` / `body_quat_w[:, 0]` 有限差分算根速度，重跑
> `scripts/prepare_g1_tasks_npz.py` + `build_library.py`（~2 min）。

### 5.15 测试（新增 6 个用例，`tests/`）

| 测试 | 断言 |
|---|---|
| `test_mjlab_bundle_obs93.py` | `bundle.observation(...).shape[-1]==93`；`flatten_history` 后 930；term-major 顺序与 `OBS_TERMS` 逐项相等 |
| `test_mjlab_state_frame.py` | 对同一帧：`_ref71_body` 的 `[39:42]` == `quat_apply_inverse(quat, world_ang_vel)`，误差 <1e-6；且与 `library.ref(.)` 的其余维逐位相等 |
| `test_mjlab_joint_grad.py` | `imagine_with_tokens` 全程可微：`∂reward/∂token ≠ 0`、`policy.requires_grad==False`、`wm.requires_grad==False` |
| `test_mjlab_ckpt_roundtrip.py` | `save_checkpoint` → `load_mjlab_bundle` 逐张量相等；`B1`/`B2` 都能加载；`B3` 被 `is_mjlab_checkpoint` 拒绝 |
| `test_mjlab_hold_token.py` | `MjlabHoldTokenBuilder` 的 hold 窗口 = 当前帧重复 10 次；同一帧在 `future`/`hold` 下 token 不同；hold token 值域落在 FSQ 格点 `round(x*16)/16` |
| `test_mjlab_batch_contract.py` | `MotionBatchProvider.sample()` 所有字段形状符合 §7.1 表 |

---

## 6. W1：318 在 SuperMimic 上的续训（policy **和** world model）

### 6.1 先确认三件事（都有代码/数据证据）

| 问题 | 答案 | 证据 |
|---|---|---|
| 318 的 token 是不是 hold？ | **是**。① 318 parquet 的 `action.motion_token` 由部署前端算出，前端窗口 = 当前帧重复 10 次；② 实测 `data/npz/g1_tasks/*.npz` 的 `g1_motion_token_64` 与 parquet 的 `action.motion_token` **逐位相等**（2993 帧 max\|Δ\|=0.0），且全部落在 FSQ 格点 `k/16` 上 | `sonic_token.py:229-232`（`idx = frame.expand(n,nf)`）；本地实测（2026-10-28） |
| 有没有前瞻？ | **有，但前瞻在 token 序列上**：单个 token 是 hold（窗口内无未来帧），一次喂 40 个 token（0.8 s）提供前瞻 | `sonic_token.py:243-247`（`k_window`：「前瞻来自 token 序列，不改变单个 token 的口径」）；`data.py` 的 `token_horizon=40` |
| 要不要端口侧车？ | **不要**。hold 口径下 WBC 输入空间 == 标签空间 | §1.6 的逐行核对；`train_joint_v2.py:518` 的 S0.4 结论 |

### 6.2 为什么直接吃录制 token（而不是在线重建 hold）

HA 能建 `runs/isaac_motion/ha_cmd/packed.pt`，是因为 `data/ha_raw` 的 npz 里有
`encoder_motion_joint_pos_hist`（**命令流**，见 `scripts/build_ha_command_lib.py:1-15`）。
**318 的原始 parquet 只有 31 列，没有 `encoder_*`**（已逐列核对）：

```
observation.state, observation.eef_state, action.wbc, action.motion_token,
observation.root_orientation, observation.projected_gravity, ..., teleop.planner_*, timestamp, ...
```

所以在 mjlab 里「在线重建 hold token」只能拿**执行轨迹**当命令流用 → 与真标签差
mean\|Δ\|=0.39 rad / 最佳 lag +8 帧（`build_ha_command_lib.py:5-8`），纯属往下走。
**直接读录制 token 才是这个数据集上的正确做法**，而且更省事：

| 方案 | token | 参考 | 是否精确 | 改动 |
|---|---|---|---|---|
| **B（采用）** | 录制标签（逐帧，`ft318_tokens.pt`） | 执行轨迹 | ✅ 与部署完全一致 | §5.2（~10 行 upstream + 60 行新模块） |
| A（仅作对照/回退） | 在线 hold（执行轨迹 + 仿真 anchor） | 执行轨迹 | ❌ 命令流与 anchor 双近似 | 原来的 hold 前端方案 |

**B 的唯一代价（写进验收）**：录制 token 里的 anchor 是**录制时**的机器人姿态。训练中若给
sim 机器人加姿态扰动，anchor 会过期——token 会「以为」机器人还在录制姿态，于是丢掉一部分
「参考相对我偏了多少」的修正信号。处置：

1. **W1 先把姿态扰动关掉**（`--env.no-reset-noise`，保留 `action_noise_std` 与 obs noise），
   让 anchor 与录制姿态一致；
2. 若要开 DR，就跑一条对照臂量 `sim_rew` 掉多少（预期 <5%），掉太多再切 A（§10 R13）。

> 参考：trackerlab 早就有「直接读 tokens」的官方支持（`motion_buffer/motion_lib/supermimic.py:134`
> 检测 `*.tokens.npy` 并当条件），文档里称这是「接口治本」。B 就是把这个做法搬到 mjlab。

### 6.3 数据准备（第 1 步已完成 ✅）

```bash
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
VENV=$COMIMIC/third_party/SuperMimic-Reproduced/.venv/bin/python

$VENV third_party/SuperMimic-Reproduced/scripts/data/build_library.py \
  --npz-dir $COMIMIC/data/npz/g1_tasks \
  --out     $COMIMIC/data/library/ft318_compact.pt
# 实测输出（2026-10-28）：318 clips / 473876 frames (2.63 h) / 0.07 GB fp16
# 校验：ref.shape=(473876,71) fp16；names=318；num_bins=9621；与 FINAL_HANDOVER 锁定的
#       318 eps / 473,876 帧 逐帧一致

# ② 录制 token 包（与库同序同帧）—— 条件 token 的唯一来源（§5.2）
$VENV scripts/build_mjlab_tokens.py \
  --npz-dir $COMIMIC/data/npz/g1_tasks \
  --library $COMIMIC/data/library/ft318_compact.pt \
  --out     $COMIMIC/data/library/ft318_tokens.pt
# 实测输出（2026-10-28）：tokens (473876,64) fp16 / 60.7 MB；names 与库逐项相等、同序；
#       全部落在 FSQ 格点 k/16 上
```

### 6.4 续训 ckpt 预处理（§5.3，必须先做）

```bash
$VENV scripts/prepare_wbc_resume.py \
  --src     third_party/SuperMimic-Reproduced/logs/e9_bones_seed_token_h200x2/ckpt_0024000.pt \
  --library data/library/ft318_compact.pt \
  --out     runs/wbc_resume/e9_0024000_ft318.pt
# 不做这一步 → load() 里 bin_return.copy_ 直接 size-mismatch（1,013,724 vs 9,625）
```

### 6.5 训练命令

```bash
cd $COMIMIC/third_party/SuperMimic-Reproduced
TORCHRUN=$COMIMIC/third_party/SuperMimic-Reproduced/.venv/bin/torchrun
# 条件 token = 数据集录制 token（§5.2），不需要任何在线前端/环境开关
export SONIC_ENCODER_ONNX=$COMIMIC/third_party/SuperMimic-Reproduced/data/assets/model_encoder.onnx

CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node 2 --master_port 29640 \
  -m supermimic.scripts.train \
  --run-name    wbc_ft318_mjlab \
  --log-root    logs \
  --data.library $COMIMIC/data/library/ft318_compact.pt \
  --data.tokens  $COMIMIC/data/library/ft318_tokens.pt \
  --resume       $COMIMIC/runs/wbc_resume/e9_0024000_ft318.pt \
  --env.num-envs 4096 \
  --env.action-noise-std 0.1 \
  --env.no-reset-noise \

  --policy.token-dim 64 \
  --schedule.total-iterations 60000 \
  --schedule.checkpoint-interval 2000 \
  --schedule.log-interval 200 \
  --schedule.warmup-iterations 40 \
  --wm-trainer.batch-size 2048 --wm-trainer.batches-per-update 10 \
  --wm-trainer.lr 1.4e-5 --policy-trainer.lr 1.4e-5
# 注1：Trainer 在分布式下会把 lr 乘 sqrt(world)=1.41 ⇒ 有效 2e-5（training/loop.py:41-44）
# 注2：total-iterations 是**绝对** env step（load() 会把 iteration 恢复成 ckpt 里的 24000，
#       循环条件是 iteration < total），所以 60000 = 续训 36k 步
# 注3：env.obs-noise / domain-randomization 默认 True，别写 `--env.obs-noise True`
#       （tyro 的 bool 是开关，关闭用 `--env.no-xxx`）；`--env.no-reset-noise` 是 §6.2 的决定：
#       录制 token 的 anchor 绑定录制姿态，关掉姿态扰动才能自洽（§10 R13）
```

为什么这么配：

| 参数 | 值 | 理由 |
|---|---|---|
| `--resume` | e9 的 24k 步 ckpt（经 §5.3 预处理） | 318 时代的联合微调用的就是它（`docs/FT318_HAND_JOINT_1000_20261025.md:15`）；没被后来的 Isaac 侧改动污染 |
| `--data.library` | `ft318_compact.pt` | 续训的参考分布换成 318（不再混 BONES-SEED，`mixed` 仍禁） |
| `--data.tokens` | `ft318_tokens.pt` | 条件 token 就是数据集自带的部署 token（§5.2）；不重建、不近似 |
| lr | 有效 2e-5 | 与 trackerlab 的 WBC 微调一致（`trackerlab/logs/wbc_mix_obs93/args.json: lr=2e-05`）；原训练是 2e-4 |
| total | 60k env steps | 318 只有 BONES-SEED 1% 的帧量；e9 训到 156k，续训 36k 是"再走一段"的量级 |
| 4096 envs × 2 卡 | — | 与 e8/e9 同配置（峰值 ~22 GB/卡），不改 WBC 的采样分布 |
| `--env.no-reset-noise` | 关姿态扰动 | 录制 token 的 anchor 绑定录制姿态，关掉才能与 token 自洽（§6.2 / §10 R13） |

### 6.6 policy 与 WM 到底怎么更新（这就是"世界模型也要对应训练"）

`sm-train` 每次迭代 = **1 个 env step**（`training/loop.py:176-227`）：

1. `act(obs)`：hold token + obs93 → 29 维动作（`training/loop.py:152-159`）；
2. `env.step`：MuJoCo-Warp 里 4096 个 G1 并行前进一步，动作带 `action_noise_std=0.1`；
3. 过了 `warmup_iterations` 后按 `updates_per_env_step=0.4` 累积，每满 1.0 做一次更新：
   * `wm_trainer.update(buffer)`：10 个 batch × 2048 条 8 步窗口，**用仿真真转移**拟合
     `(s, a, xi) → s'`（`wm_trainer.py:23-31`）；
   * `policy_trainer.update(buffer)`：想象 H=32 步 + critic（`policy_trainer.py:19-38`）。

⇒ **policy 和 WM 在同一次更新里一起训**，WM 吃的是 318 参考下的真仿真转移，不是离线拟合。
Isaac 侧的对应关系：

| 概念 | trackerlab（Isaac） | mjlab（本方案） |
|---|---|---|
| WM 热身 | `wm_only_iters`（0 / 200） | 无单独开关，用 `--wm-trainer.batches-per-update` 控制占比 |
| WM 每次样本 | `--wm-batch 2048 --wm-batches 10` | `--wm-trainer.batch-size 2048 --wm-trainer.batches-per-update 10` |
| policy 更新频率 | `policy_updates_per_wm_update` | `--schedule.policy-updates-per-wm-update 1` |
| 学习率 | `--lr 2e-5` | `--wm-trainer.lr` × `--policy-trainer.lr`（同值） |
| 显式积分 WM | 需要打补丁（Isaac 移植丢了） | **原生就有**（`state_spec.py:55 explicit_integration=True`，e9 训练时即开启） |

### 6.7 预算

| 项 | 估计 |
|---|---|
| 显存 | 4096 envs/卡，4096×2，峰值 ~22 GB/卡 |
| 时间 | 36k env steps ≈ 1.2–1.5 h（2×H200）/ ≈ 2.5–3 h（2×4090） |
| 磁盘 | 每 2000 步一个 ckpt × 191 MB ≈ 3.4 GB |

### 6.8 验收（4 条，任一不过就回滚到 e9）

| # | 判据 | 命令 | 通过线 |
|---|---|---|---|
| W1-1 | 318 参考跟踪不退 | `$VENV third_party/SuperMimic-Reproduced/scripts/eval/benchmark.py`（`--library ft318_compact.pt`） | `sim_rew` ≥ 底模（同库同种子） |
| W1-2 | token 条件有效 | 同上，token 打乱 | 相对下降 ≥ 30% |
| W1-3 | 四个任务逐个不退 | `python3 scripts/eval_mujoco.py --motion data/npz/g1_tasks --clip-index {0..17}` | 每任务 `state_term` ≤ 底模 +5% |
| W1-4 | 产物可复现 | `train_log.csv` / `config.json` / `tb/` 齐备；`ckpt_latest.pt` 能被 `SuperMimicBundle.from_checkpoint` 加载 | 无 NaN，`iteration` 单调 |

### 6.9 与 IsaacLab WBC 微调流程的逐项对齐

| 步骤 | Isaac（trackerlab） | mjlab（本方案） | 状态 |
|---|---|---|---|
| ① 原始数据 | `data/ha_raw`（**含** encoder 命令流） | 318 parquet（**无** encoder 流） | ⚠️ 客观差异，§6.2 给了 A/B 两方案 |
| ② 参考库 | `ha_cmd/packed.pt`（命令流，canonical 71 维） | `ft318_compact.pt`（执行轨迹，canonical 71 维） | ✅ 已产出 |
| ③ 样本权重 | `merge_libraries.py --extra-repeat 20`（HA 只占 1.24%） | 318 单库，无需重复 | ✅ |
| ④ 条件 token | `SonicTokenBuilder(window="hold")` + 实时 anchor（sim 里没有逐帧标签，只能在线算） | **直读录制 token**（318 自带逐帧标签）：§5.2 | ✅ 数据已产出 / 🔧 代码待改 |
| ⑤ 续训起点 | `ckpt_010500_physint.pt` / `ckpt_016000_spec.pt` | `e9/ckpt_0024000.pt`（24k 步） | ✅ 已核对 |
| ⑥ 续训配置 | `lr 2e-5`、8192 envs、`wm_only_iters 0` | `lr 2e-5`（有效）、4096×2 envs、`batches_per_update 10` | ✅ |
| ⑦ 验收 | `sim_rew ≥ 基线`、token 对比度、四任务 | §6.8 的 W1-1..4 | ✅ |
| ⑧ 产物交接 | `logs/<run>/ckpt_*.pt` + 验收数字 | `logs/wbc_ft318_mjlab/ckpt_*.pt` + `runs/wbc_resume/` 记录 | ✅ |

## 7. J1：联合微调移植

### 7.1 batch 契约（`MotionBatchProvider.sample()` 产物）

| 字段 | 形状 | 来源（mjlab） | 校验 |
|---|---|---|---|
| `start_state` | `(B,71)` | `reference_start()["state"]`（**体系 ω**） | `[3:7]` 四元数模长 1±1e-5 |
| `obs_hist` | `(B,930)` | `flatten_history(observations)` | `= B×10×93` |
| `history` | `(10,B,93)` | `observations` | term-major 与 `OBS_TERMS` 对齐 |
| `prev_action` | `(B,29)` | `(q_ref-q0)/action_scale` 的 t-1 帧 | 与 `action_scale` 同设备 |
| `xi` | `(B,33)` | `default_xi` | `[:]=0.75` |
| `references` | `list[H]` of `ReferenceFrame` | `library.bodies(frame+t)` | 每项 4 个张量 |
| `teacher_states` | `(B,H,71)` | `_ref71_body` | 同上体系 |
| `fm_tokens` | `(B,40,64)` | `RecordedTokenSource(token_space="label")`（npz 的 `g1_motion_token_64`） | 值域 `round(x*16)/16` |
| `hand_ref` | `None` | D5=否 | — |

### 7.2 损失与梯度

```
x_norm = FM_sample(head, bb_out)          # 4 步 Euler，梯度只回 head（fm_grad 后端）
token  = codec.decode(x_norm)[..., :64]
a_t    = policy(obs_hist_t, token)        # 冻结 WBC，梯度穿过
s_t+1  = wm.step(s_t, a_t, xi)            # 冻结 WM（explicit dt0.02）
r_t    = heads.reward(s_t+1, a_t, a_t-1, ref_t)      # 解析，无需梯度
V_t    = critic(s_t, ref_t, xi)           # 可训（v3）
L      = λ_bc·FMBC(x_norm, port_token) + λ_r·(−R̂) + λ_hand·hand + λ_a·hinge(a) + λ_a'·anchor
```
`R̂ = Σ γ^t r_t / (H·0.1)`（`train_v2.py:273`，分母用 `_MAX_STEP_REWARD`）。

### 7.3 v3 的六个机制 → mjlab 落点

| 机制 | Isaac 落点 | mjlab 落点 | 是否要改 |
|---|---|---|---|
| 只训 VLA head | `train_v2._freeze_executor` | 同（bundle 换成 mjlab） | 否 |
| token BC | `fm_bc.bc_target/loss` | 同（标签直接用 npz `g1_motion_token_64`） | 不换（label 空间） |
| 想象反传奖励 | `rollout.imagine_with_tokens` | 同（默认 93 已经对上） | 否 |
| critic / GAE / 真 return-to-go | `token_critic.py` | 同（输入是 `state(71)+ref+xi`） | 否 |
| 真录制状态 burn-in | `burnin.sample_burnin` | 改 2 处（§5.9） | 是 |
| DR + 过程扰动 | `dr.sample_batch` + `perturb_state` | 同（状态 71 语义一致） | 否 |
| WM 真转移保鲜 | `wm_refresh.refresh_wm` | 需要 mjlab 的真转移（M4） | 是（M4） |
| trust region / anchor | `v3_step.V3Config` | 同 | 否 |

### 7.4 两阶段配方

**阶段 A：BC warmup（`lambda_r=0`，800 步）** —— 先让 token 落到 label(hold) 空间，不碰奖励。
```bash
--lambda-r 0 --lambda-bc 1.0 --lora-rank 32 --horizon 16 --batch-size 4 --grad-accum 4
--bc-token-space label
```
验收：`token_mse` 相对起点下降 ≥ 40%；`codec/out_of_range` < 1%。

**阶段 B：+奖励（`lambda_r=0.3`，2000 步）**
```bash
--v3 --lambda-r 0.3 --lambda-hand 0 --critic --critic-source real
--anchor-weight 1e-3 --trust-eps 0.2 --trust-weight 1.0 --act-eps 0.15 --act-weight 0.5
--start-source recorded --real-transitions runs/wm_real/mjlab_transitions.pt   # M4 才需要
```
验收见 §7.5。

### 7.5 命令（可直接抄）

```bash
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
export PYTHONPATH=$COMIMIC:$COMIMIC/Isaac-GR00T:$COMIMIC/third_party/gr00t_deps:$COMIMIC/third_party/gr00t_deps/_noflash
export MUJOCO_GL=disable
export SONIC_ENCODER_ONNX=$COMIMIC/third_party/SuperMimic-Reproduced/data/assets/model_encoder.onnx

cd $COMIMIC && python3 scripts/train_joint_mjlab.py \
  --wbc-ckpt $COMIMIC/third_party/SuperMimic-Reproduced/logs/wbc_ft318_mjlab/ckpt_00XXXX.pt \
  --vlm-ckpt /inspire/hdd/global_user/liumengfan-253108110079/flowbody/FlowBody/runs/groot_sonic_vla_20k_frozen271_20261023/checkpoint-20000 \
  --vlm-path $COMIMIC/groot_repro/nvidia/Cosmos-Reason2-2B-weights \
  --motion   $COMIMIC/data/npz/g1_tasks \
  --bc-token-space label --wbc-token-window hold --wbc-token-space label \
  --horizon 16 --batch-size 4 --grad-accum 4 --v3 \
  --lambda-bc 1.0 --lambda-r 0.3 --lambda-hand 0 \
  --anchor-weight 1e-3 --trust-eps 0.2 --trust-weight 1.0 \
  --lora-rank 32 --amp bf16 \
  --steps 2800 --eval-every 200 --eval-batches 4 \
  --save-every 200 --save-vla-every 400 \
  --out $COMIMIC/runs/joint_mjlab_ft318
```

### 7.6 J1 验收门（G1–G5）

| 门 | 判据 | 读数来源 |
|---|---|---|
| G1 | `token_mse` 800 步内降到起点 60% 以下 | `eval_curve.jsonl` |
| G2 | `rollout/reward` 在 H=16 上单调上升，且 `state_term` 下降 | 同上 |
| G3 | token 不漂移：`‖token−prior‖` RMSE 不超 `trust_eps` 的 1.5× | 训练日志 `trust/excess` |
| G4 | 冻结真值：`policy.requires_grad` 全 False，`wm` 梯度范数 = 0 | `--gate-only` 输出 `gates.json` |
| G5 | 闭环：MuJoCo 4 任务 × 3 seeds，成功率不低于底模 -2pp | E1 的 `qc_*.json` |

---

## 8. E1：MuJoCo 闭环评测

### 8.1 口径

| 项 | 值 |
|---|---|
| 执行器 | J1 的 `joint_*.pt`（冻结 WBC）+ J1 的 `vla_*.pt`（合并进 VLA ckpt：`scripts/export_vla_ckpt.py`） |
| 环境 | `diff_vla_wbc/mujoco_env.py`（CPU MuJoCo，`mjINT_IMPLICITFAST`，20 ms 控制周期） |
| obs | 93（由 bundle 产生，**不许**用 npz 的 46 维） |
| 步数 | 4 任务 × 3 seeds × 20 s |
| 指标 | `state_term`、`rollout_reward`、root 位置误差、关节 RMSE、摔倒率、成功率 |

### 8.2 产物

```
runs/joint_mjlab_ft318/cl/
  qc_task{1..4}__seed{0,1,2}.json     # 每 episode 的状态轨迹 + 指标
  summary.json                         # 聚合 + 与底模的 Δ
  videos/task{k}__seed{s}.mp4          # 可选
```

### 8.3 与 `JointTrainer`（旧路）的对照（必做，否则无法归因）

`diff_vla_wbc/train.py` 的旧路线**会训 WBC policy**（`policy_lr=1e-5`，`:47,132`），本方案不训。
为了说明"只训 head"更好/不更差，必须跑一条对照臂：`scripts/train_joint.py`（旧 `JointTrainer`，
**会训 WBC policy**，`policy_lr=1e-5`，`:47,132`）跑 500 步、同数据、同 seed，与 J1 阶段 A 比
`token_mse` / `state_term` / `rollout_reward`。

---

## 9. 里程碑（每步都有回滚点）

| 里程碑 | 内容 | 产出 | 回滚点 |
|---|---|---|---|
| **M0** | 口径探针（**不训练**） | `tests/test_mjlab_state_frame.py` 等 6 个测试全绿；`gates/m0_probe.json` 记录：<br>① `library.ref71` vs 仿真状态在 20 帧上的 max\|Δ\|（除 `[39:42]`）<br>② `[39:42]` 转换前后的 max\|Δ\|<br>③ obs93 与仿真 `env.obs["policy"]` 的 max\|Δ\| | 无代码风险，纯只读 |
| **M1** | `mjlab_joint.py` + 解耦改动（§5.7–5.11） | **✅ 已完成 2026-10-28**：`diff_vla_wbc/mjlab_joint.py` 新增；`burnin.py` 改成后端无关（4 处）；`isaac_joint.py` 补 4 个通用方法；`train_v2/v3/rollout/data` **零改动**；`tests/test_mjlab_joint_grad.py` 3 passed（4.5 s，CPU） | git revert 单个 commit |
| **M2** | W1：318 续训（含 WM） | `runs/wbc_resume/*` + `logs/wbc_ft318_mjlab/ckpt_*` + W1-1..4 | 换回 e9 原 ckpt 即可 |
| **M3** | 阶段 A（BC warmup，label 空间） | `runs/joint_mjlab_ft318/joint_000800.pt` | 无需侧车；ckpt 独立 |
| **M4** | （可选）真转移采集 + WM 保鲜 | `runs/wm_real/mjlab_transitions.pt` + `wm_refresh` 验收 | 关 `--wm-refresh-every` |
| **M5** | 阶段 B（+奖励）+ v3 机制全开 | `joint_002800.pt` + G1–G4 | 回退到阶段 A 的 ckpt |
| **M6** | E1 闭环 + 对照臂 | `cl/summary.json` + videos | — |

**M0 的三条探针（验收线）**：

```python
# ① 参考状态 vs 仿真状态（同一帧，无噪声 reset）
env = make_env(library); env.reset_to_frame(ids, frame=f)
s_sim = StateIO(env).read()                       # (N,71) 体系 ω
s_ref = library.ref71(f)                          # 世界 ω
d_world = (s_sim[:, 39:42] - s_ref[:, 39:42]).abs().max()
d_body  = (s_sim[:, 39:42] - _ref71_body(library, f)[:, 39:42]).abs().max()
assert d_body < d_world        # 期望 d_world 大一个量级、d_body <1e-4

# ② 解析 obs vs 环境 obs
o_analytic = bundle.observation(s_sim, prev_action, xi)        # (N,93)
o_env = env.obs["policy"].float()                              # (N,93)
assert (o_analytic - o_env).abs().max() < 1e-3

# ③ token 三个空间的相关性（确认不可互换）
corr(label, mjlab_port) < 0.3
```

---

## 9.1 J1 已实现（2026-10-28）：入口 + 采集 + 真转移 + 验收门

```bash
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
export PYTHONPATH=$COMIMIC:$COMIMIC/Isaac-GR00T:$COMIMIC/third_party/gr00t_deps:$COMIMIC/third_party/gr00t_deps/_noflash

# ① 门禁（不训练）：codec / 冻结 / burn-in / token 对比度 / FM-BC
python3 scripts/train_joint_mjlab.py \
  --wbc-ckpt <mjlab wbc ckpt> --motion data/npz/g1_tasks \
  --tokens data/library/ft318_tokens.pt \
  --vlm-ckpt <vla ckpt> --vlm-path groot_repro/nvidia/Cosmos-Reason2-2B-weights \
  --out runs/joint_mjlab_ft318 --gate-only

# ② BC warmup（800 步）
python3 scripts/train_joint_mjlab.py ... --steps 800 --lambda-r 0 --lora-rank 32 --amp bf16

# ③ 采集（B 线，给 WM 保鲜 / critic / b_start）
python3 scripts/closed_loop_vla_supermimic.py --vla-ckpt <ckpt> --joint-ckpt <mjlab joint ckpt> \
  --task task1 --steps 600 --out-video runs/mjlab_collect/t1.mp4 --dump runs/mjlab_collect/t1.pt

# ④ 真转移
python3 scripts/build_real_transitions_mjlab.py --dump 'runs/mjlab_collect/*.pt' \
  --ckpt <wbc ckpt> --motion data/npz/g1_tasks --out runs/wm_real/mjlab_transitions.pt --with-rewards

# ⑤ WM 速度通道验收门（F）
python3 scripts/probe_wm_qd_corr.py --ckpt <wbc ckpt> --motion data/npz/g1_tasks \
  --tokens data/library/ft318_tokens.pt --n 64 --h 8
```

实测（2026-10-28，RTX 4090 / 1 步）：

| 项 | 结果 |
|---|---|
| BC 臂 | `loss 0.0504`（`loss/fm_bc` 同值，`rollout/skipped 1.0`） |
| 奖励臂（`--lambda-r 0.3`） | `rollout/reward 0.0879`、`loss/reward_norm 0.875`、`loss -0.227` |
| LoRA | rank 8 → 8,388,608 / 1,628,904,576（0.515%）；rank 32 约 16.8M |
| 存档 | `joint_*.pt` 可被 `JointTrainer.load_bundle_from_checkpoint` 加载（逐张量相等） |
| WM 保鲜 | `pos_mse 2.0e-4 → 5.2e-6`（2 步；合成真转移） |
| **qd_corr 基线（e9 在 318 上）** | **0.359 / ratio 4.55 / pos 0.034 m ⇒ FAIL** —— W1 后必须重测并过 0.9 / [0.9,1.1] |

> 采集合成验证用了一个"回放式" dump（`frames` 有效）；**自由闭环** dump 的 `frames=-1`，
> 因此 `--with-rewards` 会被跳过、`b_start` 也不能直接用（需要回放式采集或补帧号）。

## 10. 风险与坑（按历史踩坑写）

| # | 风险 | 症状 | 处置 |
|---|---|---|---|
| **R1** | **obs 93 与 99 混用** | 首层 `(2048,994)` 加载 `(2048,1054)` 直接报错；但 `decoder.scale` 长度不对时会**静默**广播错 | 入口断言 `930`/`994`；`test_mjlab_bundle_obs93` |
| **R2** | **token 空间混用** | `token_mse` 看着在降，闭环却跟不住 | §1.6 三空间表；入口断言 `--bc-token-space` 与 `--wbc-token-space` 一致才允许 `lambda_r>0`（照抄 `train_joint_v2.py:531-545`） |
| **R3** | **角速度帧不一致（世界 vs 体系）** | 训练 loss 正常，闭环整体漂 ~5–10°；obs 与仿真不同分布 | M0 探针 ①②；`_ref71_body` 统一在 bundle 内转换 |
| **R4** | 动作语义飘到 `a=Δq` | policy rollout 奖励骤降（历史：2.87→29.6） | `--wbc-action-scale mjlab` 默认；`one` 只做对照 |
| **R5** | `explicit_integration` 结构不匹配 | 加载 ckpt 时 `wm_integration` 键缺失 → 用错积分 | 从 ckpt 的 `config.world_model.state_spec.explicit_integration` 读；加载后断言 |
| **R6** | 参考库格式（compact vs full） | compact 库没有 `body_*`，`library.bodies()` 靠 FK（`motion_library.py:182-190`），慢但正确；若 FK 配置错会静默 | J1 **用 full 库**（npz 直接喂），W1 才用 compact；两者不要混 |
| **R7** | 两个解释器混跑 | `import supermimic` 成功但 torch 版本冲突，报 CUDA/算子错 | §1.7 表；launcher 里写死解释器路径 |
| **R8** | 政策门禁拦路 | `SystemExit: 只接受 Isaac/trackerlab ckpt` | §5.14 先改政策与门禁 |
| **R9** | 318 与 BONES-SEED 混库 | 口径不可归因（`mixed_ha_ft318` 已禁） | 一库一 run；run 名带库名 |
| **R10** | 只训 head 但 WBC 是旧的 | 奖励梯度在旧 WBC 上无效（trackerlab 历史结论） | 必须**先 W1 再 J1**；W1 没过不要跑 J1 |
| **R11** | 图像输入缺失 | 318 有 `g1_ego_view`，C 线采集没有 | J1 阶段若只跑离线 rollout（无 `--vlm-dataset`），可先用标签驱动；上真图前先验 VLA 侧 forward |
| **R13** | 录制 token 的 anchor 与仿真姿态脱钩 | 开了姿态扰动后 `sim_rew` 掉、闭环走偏 | W1 关 `--env.no-reset-noise`；开 DR 必须跑对照臂；掉太多切 §6.2 的 A 方案 |

---

## 11. 附录

### 11.A 命令速查

```bash
# M0 探针
python3 scripts/probe_mjlab_state_frame.py --ckpt <e9 ckpt> --motion data/npz/g1_tasks   # 新增
# M1 门禁
python3 scripts/train_joint_mjlab.py --gate-only --gate-out runs/gates/m1.json ...
# M2 W1
bash scripts/launch_wbc_ft318_mjlab.sh
# M3 阶段 A（无侧车）
bash scripts/launch_joint_mjlab_ft318.sh STAGE=A
# M5 阶段 B
bash scripts/launch_joint_mjlab_ft318.sh STAGE=B
# M6 闭环
bash scripts/run_e1_mujoco_qc.sh runs/joint_mjlab_ft318
```

### 11.B 文件清单

**新增（9 个源码 + 6 个测试）**
```
diff_vla_wbc/mjlab_joint.py
diff_vla_wbc/mjlab_tokens.py                # LibraryTokens：按帧直读录制 token
diff_vla_wbc/rotations.py                   # 从 trackerlab_adapter 搬 8 行
scripts/build_mjlab_library.py              # 薄封装 + provenance 校验
scripts/build_mjlab_tokens.py               # npz 标签 → ft318_tokens.pt（已用内联脚本产出，可固化）
scripts/prepare_wbc_resume.py               # 续训 ckpt 预处理（bin 表重写）
scripts/train_joint_mjlab.py
scripts/launch_wbc_ft318_mjlab.sh
scripts/launch_joint_mjlab_ft318.sh
scripts/run_e1_mujoco_qc.sh
tests/test_mjlab_{bundle_obs93,state_frame,joint_grad,ckpt_roundtrip,batch_contract}.py
tests/test_mjlab_hold_token.py
```
**修改（10）**
```
diff_vla_wbc/train_v2.py        (6 处)
diff_vla_wbc/v3_step.py         (3 处)
diff_vla_wbc/burnin.py          (2 处)
diff_vla_wbc/rollout.py         (2 处)
diff_vla_wbc/data.py            (1 处)
scripts/train_joint.py          (门禁 + backend)
scripts/train_joint_v2.py       (仅加 --backend isaac 显式参数，默认不变)
scripts/eval_mujoco.py          (断言 + token 源)
closed_loop/code/closed_loop_mujoco.py  (--wbc-ckpt)
docs/DATA_POLICY_HA_ONLY.md     (例外条款)
runs/OFFICIAL_SUPERMIMIC.json   (追加 mjlab 段)
```
**已产出（数据）**：`data/library/ft318_compact.pt`（67 MB）、`data/library/ft318_tokens.pt`（60.7 MB）。

**不动**：`third_party/SuperMimic-Reproduced/**`（除 §5.2 路线② 的 3 行窗口开关 + 新增日志目录）、`diff_vla_wbc/isaac_joint.py`、
`diff_vla_wbc/train.py`（只加 backend 分支）、Isaac 主线的所有 launcher 与 HA 数据。

### 11.C 字段字典

| 名字 | 形状/取值 | 出处 |
|---|---|---|
| `state` | `(71,)` `[p3,q4,q29,v_w3,w_b3,qd29]` | `state_spec.py:3` |
| `obs` | `(93,)` `[q-q0(29),qd(29),w_b(3),pg(3),a_prev(29)]` | `heads.py:22,30` |
| `obs_hist_flat` | `(930,)` term-major | `heads.py:57` |
| `token` | `(64,)`，录制标签（hold 空间），FSQ 格点 `round(x*16)/16` | `sonic_tokens.py:142`、`sonic_token.py:229-232` |
| `xi` | `(33,)` = [0.75, bias×29, com×3] | `state_spec.py:57` |
| `action` | `(29,)`，`q_target=q0+scale*a` | `supermimic_backend.py:87` |
| `reference` | `ReferenceFrame{bodies pos/quat/lin/ang, root pos/quat}` | `heads.py:69-78` |
| `r` | 标量，`Σ` 上界 0.1/步 | `rewards.py:42-43` |

### 11.D 待拍板（回答后即可开工）

| 编号 | 问题 | 影响面 |
|---|---|---|
| **D1** | W1 起点用 `ckpt_0024000`（318 时代口径）还是 `ckpt_0156000`（最新、但没在该口径上验证过）？ | W1 全部数字 |
| **D2** | ~~future vs hold~~ → **已定：hold**（用户 2026-10-28 确认；侧车因此废弃） | 影响 §5.2 的 3 行 upstream 开关 |
| **D3** | 是否正式解除 `DATA_POLICY` 对 MuJoCo/318 的封锁（需要你签字）？ | 所有门禁 |
| **D4** | J1 的 VLA 起点：318 监督 20k，还是 HA 的 `checkpoint-8000`（可与 Isaac 主线对比）？ | J1 的对照可比性 |
| **D5** | 要不要在 mjlab policy 上加手部输出？（318 有 `g1_hand_joints_14`，但 mjlab policy 只有 29 维） | 需改 upstream `policy/nets.py`，风险高 |

> 建议：D1=`ckpt_0024000`、D2=`future`（先跑通，hold 作为 M5 之后的 A/B）、D3=是、D4=`318-20k`、
> D5=否（手部只喂 VLA 头，不进 WBC）。
