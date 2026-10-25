# 消融实验全套设计与执行手册（v2 · 2026-10-24）

> 用途：后续实验执行 + 论文写作的唯一依据。每条消融都给出**变量、对照、命令、指标、输出路径、判读口径、成本、代码状态**。
> 先读 §1（现状核对）：**现在还不能开跑**，有两个前置阻塞。

---

## 0. 一页速览

| # | 消融 | 要回答的问题 | 处理组 | 对照组 | 主指标 | 代码状态 |
|---|---|---|---|---|---|---|
| **A1** | 有无联合微调 | 把下游奖励反传回 VLA 有没有用？ | 联合微调后的 VLA（如 `vla_004000.pt` 合并版） | 官方 20k VLA（纯行为克隆） | **闭环成功率 + Wilson CI** | ✅ 代码就绪（评测环境受阻，见 §1） |
| **A2** | FM 反传方法 | 5 种梯度估计怎么选？ | `pathwise / checkpointed / truncated / adjoint` | 互为对照 + 已有 `qvgm_lite` toy 结果 | `rollout_reward` 曲线、显存、s/step | ✅ 就绪 |
| **A3** | PPO vs BPTT | 固定墙钟下，policy gradient 能不能学出来？ | `train_ppo.py`（score-function，同一 reward） | `train_joint.py`（精确反传） | 同墙钟 reward 曲线 + 成功率 | ✅ 就绪（缺一次 GPU 冒烟） |
| **A4** | 世界模型冻结 × horizon | WM 要不要一起更新？horizon 多长最合适？ | WM 更新 / horizon ∈ {8,16,32,40} | WM 冻结 / horizon=8 | `rollout_reward`、`state_term`、闭环成功率 | ✅ 就绪（CLI 已支持） |
| **A5** | 冻结 SuperMimic 只训 head | 收益来自 VLA head 还是 WBC 一起动？ | 只更新 action head（`--policy-lr 0`） | 默认（head + policy 一起更新） | 同上 + **参数变化审计** | ✅ 就绪（纯 CLI） |

统一约定：**除被消融的那个变量，其余全部相同**（数据、VLM 权重、batch、horizon、手部辅助损失、seed、卡数、墙钟）。

---

## 0.5 阻塞解除命令清单（照抄执行，按顺序）

> 全部是"装东西/拷资产/改路径"，**不含训练**。执行完 §1.4 的验收命令即可确认。

```bash
# ── B1 · HA 的 7 任务场景包（必须在能访问 Google Drive 的机器上跑）─────────────
python3 -m pip install gdown
gdown "https://drive.google.com/file/d/1TCa_aVRmFrZs_l4wlxkqanNebvDtChNk/view?usp=sharing" \
      -O humanoidarena_assets.zip --continue --retries 5        # ~1–2 GB；gdown 6.x 没有 --id
# 传到训练机后：
cd <comimic>/third_party/HumanoidArena/isaaclab_twist2_g1
unzip -q ha_assets.zip -d assets
# 验收：下面必须无输出（无 MISS）
for p in $(grep -rhoE "assets/[A-Za-z0-9_./-]+\.usd" tasks/common_scene/*.py | sort -u); do
  [ -f "$p" ] || echo "MISS $p"; done

# ── B2 · HA 的 python 依赖装进 Isaac 5.0 venv（本机已装，命令留档）────────────
PY=<comimic>/third_party/isaacsim_env/bin/python
uv pip install -p "$PY" pyzmq==27.0.0 logging_mp==0.1.5 onnxruntime==1.22.1 \
                       pynput==1.8.1 redis
# 注意：rerun-sdk 需要 numpy>=2，会与 isaaclab/trackerlab 的 numpy==1.26 冲突 ——
#       headless 评测链路（sim_eval_vla / eval_vla_suite / action_provider）**不 import rerun、
#       也不 import pynput**，所以不要为它们升级 numpy。
# 验收：
PYTHONPATH=<...>/isaaclab_twist2_g1:$PYTHONPATH "$PY" -c \
  "import action_provider.create_action_provider, action_provider.action_provider_sonic, \
          action_provider.lerobot_vla_http_client; print('OK')"

# ── B3 · 给 HA 评测起一个我们自己的 VLA HTTP server（官方只给 client）─────────
#    协议见 action_provider/lerobot_vla_http_client.py；返回 latent64 / latent64_chunk
python3 scripts/ha_vla_server.py --ckpt runs/ckpt_vla_ft4000 --port 18080 --device cuda:0

# ── B4 · SuperMimic 版 action provider（替换 SONIC decoder）──────────────────
#    见 trackerlab 的 G1Assets.q_target(action) = default_joint_pos + action_scale * a
#    在 create_action_provider.py 里注册 provider=supermimic 即可

# ── B5 · Isaac 训出的 WBC → 联合微调（TrackerLabBundle 适配器）────────────────
python3 scripts/check_trackerlab_bundle.py \
  --ckpt trackerlab-supermimic/logs/bones_seed_token_2x8192_isaac50/last.pt
#    验收：同一帧 + 同一 token，适配器与 trackerlab 原生网络输出逐元素一致（1e-5）

# ── B6 · 手部接口决策（14 维连续 vs arena 2 维二值）────────────────────────
#    默认：把 14 维按阈值映射成 2 维二值，保持与 arena 评测契约可比
#    见 scripts/hand_interface.py --mode binary

# ── B7 · 机器人资产换成 arena 的带手 USD（消除 L3 差异）─────────────────────
# 注意用 **wholebody（浮动基座）** 那个，不要用 base-fix（固定基座不能训 WBC 的 root 运动）：
A=<comimic>/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots
export SUPERMIMIC_ROBOT_USD=$A/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd
#    trackerlab 的 robot.py 已支持该环境变量，并会自动置 HAS_HANDS=True（43 DoF：
#    29 体关节 + 2x7 手指；手指保持默认位形、不进入 action 空间）
#    验收：python -c "from trackerTask.supermimic.robot import USD_PATH, HAS_HANDS; print(USD_PATH, HAS_HANDS)"
```

---

## 1. 现状核对（开跑前必须确认的事）

### 1.1 已确认 OK 的

| 项 | 状态 | 证据 |
|---|---|---|
| Isaac 版本与 arena 一致 | ✅ `isaacsim 5.0.0 + Isaac Lab release/2.2.0` | 最新训练日志：`third_party/isaacsim_env` + `third_party/isaaclab_220` |
| 控制频率与 arena 一致 | ✅ `physics_dt 0.005 × decimation 4 = 0.02 s`（50 Hz） | `trackerlab-supermimic/.../task_cfg.py` |
| BONES-SEED 训练在跑、指标在动 | ✅ 2 ranks × 8192 envs，B+C，≈1.06 s/iter，peak 20.2 GiB | `logs/bones_seed_token_2x8192_isaac50.launcher.log` |
| 四个 FM 后端在真实 head 上梯度一致 | ✅ adjoint/checkpointed 与 pathwise 相对误差 0；truncated cos 0.9999 | `tests/test_fm_grad_backends.py`（12 用例） |
| PPO 训练器数学正确 | ✅ 6 个 CPU 单测（logprob/ratio/σ/value/flat-advantage） | `tests/test_ppo_trainer.py` |

### 1.2 两个前置阻塞（**不解决不能开跑**）

#### 阻塞 P0-1：arena 的 7 个任务场景 **0/7 就位**

- 现状：`closed_loop/assets/` 只有机器人资产（`ha_assets`，131 文件）；**没有任何任务场景 USD**；
  `closed_loop/assets/ha_isaac` 目录已不存在。全仓库搜 `double_desk / grap_cup / pp_box / boxing /
  open_door / sit_sofa / vision_navi` 只命中 BONES-SEED 的动作文件。
- **连带影响**：机器人初始位姿、道具初始位姿、成功判定谓词都定义在 HA 任务的
  `tasks/common_scene/base_scene_*.py` + `*_hw_env_cfg.py` 里 —— 场景没到位，这些**都无法确认**，
  A1 的「arena 口径成功率」也就跑不了。
- 解除方式：从 HA 仓库跑 `fetch_assets.sh` 拉 `assets.zip`（Google Drive，>1 GB），按
  `base_scene_*.py` 引用的路径落位。
- 验收（不训练，只加载）：7 个 scene USD 都能 `open_stage` 成功；7 个任务 `gym.make` 后
  `env.reset()` 各跑一次不报错；打印每个任务的机器人初始位姿，与 HA cfg 对齐。

#### 阻塞 P0-2：Isaac 训出的 WBC → 联合微调的**桥不存在**

- 现状：Isaac 侧（trackerlab）的 WBC 是 `algorithms/supermimic/networks.py` 的
  `Policy(enc 64→32, dec 32+obs×10→29)` + `WorldModel(71 维状态增量)`；
  而联合微调 `scripts/train_joint.py` 吃的是 `supermimic` 包（mjlab 口径）的
  `SuperMimicBundle`（`.act/.step/.reward/.observation/.termination_logits/...`）。
  **两者层形状与接口都不同，目前没有转换层。**
- 解除方式：写一个 `TrackerLabBundle` 适配器，把 trackerlab 的 `Policy` + `WorldModel` +
  `RobotKinematics` + `MDP` 包成 `SuperMimicBundle` 的 7 个方法：
  | 需要实现 | trackerlab 侧对应 |
  |---|---|
  | `act(obs_flat, cond)` | `Policy(obs_hist, goal)`，`goal` = 64 维 token |
  | `step(state, action, xi)` | `WorldModel.step((p,R,v,w,q,qd), a, xi)` |
  | `reward(next_state, action, a_prev, ref, bodies)` | `MDP.reward`（签名需对齐） |
  | `observation(next_state, action, xi)` | `MDP.obs` |
  | `termination_logits(next_state, ref)` | `MDP.termination`（硬判定 → 用 soft 版做 logits） |
  | `flatten_history` / `assets.heads.robot_bodies` | `robot.py` 的 FK + `OBS_TERMS` 展平 |
  | `policy` 的 `token_dim` | `Policy` 的 goal 维度 = 64（token 直入） |
- 工作量估计：**1–2 天**（纯适配，不训练）。验收：用同一帧、同一 token，对比
  trackerlab 与适配器输出的动作**逐元素一致**（fp32 容差 1e-5）。

### 1.3 一个必须澄清的概念

「联合微调」在本仓库的**现有实现是离线可微训练**：VLA 的观测来自 npz 里**录好的** ego_view 图像，
反传经过 SuperMimic 的 Policy + WorldModel，**训练时不启动 Isaac**。
如果你要的是「联合微调也在 Isaac 里在线渲染/闭环反传」，那是另一套架构（数周工作量），**当前没有实现**。
本手册的 A1–A5 全部基于现有离线实现。

### 1.4 2026-10-24 资产下载进展（已做 / 仍缺）

**已下载并解压**（`third_party/HumanoidArena/isaaclab_twist2_g1/`）：

| 包 | 来源 | 结果 |
|---|---|---|
| HumanoidArena 仓库 | `git clone -b release/open-source-prep` | ✅ 1347 文件：7 个任务的 `g1_tasks/*`（含 `mdp/rewards.py` 成功判定）、`tasks/common_scene/base_scene_*.py`、`script/eval_scripts/sonic/*` 7 个启动脚本 |
| Unitree 仿真资产 `assets.zip` | **ModelScope 镜像**（HF 原站限速 ~0.5 MB/s，ModelScope 3.4 MB/s） | ✅ 1.31 GB → `assets/{model,objects,robots}`，562 文件；含 `g1-29dof-dex3-base-fix` 与 `g1-29dof_wholebody_dex3`（**带 Dex3 手**） |

**仍缺（硬阻塞）：HA 的 7 个任务场景资产**

- 官方位置：Google Drive `1TCa_aVRmFrZs_l4wlxkqanNebvDtChNk`（`docs/04_environment_setup.md` §5）。
- 本机状态：元数据页可达（HTTP 200），但真实下载域名 `doc-0c-*-docs.googleusercontent.com`
  **连接超时** → **本机下不了**；HF / ModelScope 无镜像（HF `author=HumanoidArena` 无 dataset）。
- 缺的正是这 10 个 USD（`base_scene_*.py` 引用但包里没有）：

```
assets/boxing_bag/boxing_bag_physics.usd
assets/objects/small_warehouse/SM_Crate_A14_Gray_01/SM_Crate_A14_Gray_01_physics.usd
assets/objects/small_warehouse/football_scene/interaction_obj/football_goal/football_goal_physics.usd
assets/objects/small_warehouse/football_scene/interaction_obj/soccer_ball/soccer_ball_physics.usd
assets/objects/small_warehouse/hammer/hammer.usd
assets/objects/small_warehouse/small_cart.usd
assets/objects/small_warehouse/small_warehouse_digital_twin_boxtarget.usd
assets/objects/small_warehouse/small_warehouse_doubledesk/{small_doubledesk.usd,interaction_obj/hammer/hammer.usd}
assets/objects/small_warehouse/small_warehouse_opendoor/small_warehouse_digital_twin_opendoor.usd
```

- **解除方式（在任何能访问 Google Drive 的机器上，一条命令）**：
```bash
python3 -m pip install gdown
gdown "https://drive.google.com/file/d/1TCa_aVRmFrZs_l4wlxkqanNebvDtChNk/view?usp=sharing" \
      -O humanoidarena_assets.zip --continue --retries 5
# 传到训练机后：
cd /path/to/HumanoidArena/isaaclab_twist2_g1 && unzip ha_assets.zip -d assets
# 验收：下面 10 条必须全部存在
for p in $(grep -rhoE "assets/[A-Za-z0-9_./-]+\.usd" tasks/common_scene/*.py | sort -u); do
  [ -f "$p" ] || echo "MISS $p"; done
```

### 1.5 评测链路的真实形状（决定「怎么把我们的策略接进去」）

HA 的官方评测是**客户端-服务器**，不是把模型塞进 sim：

```
Isaac Lab（client, script/eval_scripts/sonic/sim_eval_vla.py）
   │  --lerobot_server_url http://127.0.0.1:18080    （HTTP，返回 latent64 / latent64_chunk）
   ▼
LeRobot VLA HTTP server（官方不提供实现，需要我们自己起）
   │  latent64
   ▼
action_provider/action_provider_sonic.py   （SONIC decoder → 29 关节）
   ▼
Isaac Lab 物理 + 成功判定
```

由此得到两个**明确的接入点**（比"重写环境"轻得多）：

| 要接的东西 | 接在哪 | 工作量 |
|---|---|---|
| **我们的 VLA（GR00T）** | 起一个 HTTP server，实现 `LeRobotVLAHttpClient` 期望的响应格式（`latent64`/`latent64_chunk`） | 1–2 天 |
| **我们的 SuperMimic WBC**（替换 SONIC decoder） | 照着 `action_provider_sonic.py` 写一个 `action_provider_supermimic.py`（token64 → 29 关节），在 provider 工厂里注册 | 1–2 天 |

**顺带解决 L3 资产差**：`g1-29dof_wholebody_dex3`（带手）已经在刚下载的资产包里，
可以直接替换 trackerlab 的 `assetslib` 机器人路径，让 WBC 训练与评测用同一个 USD。



---

## 2. 五个消融（逐个给全）

> 所有命令里的占位符：
> `$SM_CKPT` = SuperMimic WBC checkpoint；`$VLM_PATH` = `groot_repro/nvidia/Cosmos-Reason2-2B-weights`；
> `$VLM_CKPT` = `runs/ckpt_vla20k`；`$MOTION` = `data/npz/g1_tasks`（或 HA 版 `data/npz/ha_tokens`）。

### A1 · 有无联合微调（主结果）

| 项 | 内容 |
|---|---|
| **变量** | VLA 是否经过「奖励反传到 action head」的联合微调 |
| **处理组** | `runs/ckpt_vla_ft4000`（由 `joint_diff_ft318/vla_004000.pt` 合并，537 张量已校验） |
| **对照组** | `runs/ckpt_vla20k`（官方 20k，纯监督） |
| **指标** | 每任务成功率、总体成功率、Wilson 95% CI、未摔倒率、task3/4 的连续量（dx/dy/z_rel） |
| **代码** | `scripts/export_vla_ckpt.py`（已完成合并）+ `scripts/success_rate.py`（多 seed + CI + 断点续跑） |
| **阻塞** | 需要 arena 7 任务场景（P0-1）。当前只有 SONIC 4 任务可跑，**不能当 arena 口径** |

```bash
# 处理组 checkpoint（如尚未生成）
python3 scripts/export_vla_ckpt.py --base runs/ckpt_vla20k \
  --vla runs/joint_diff_ft318/vla_004000.pt --out runs/ckpt_vla_ft4000 --force

# 多 seed 成功率（--tasks 换成 arena 的 7 个任务名后即为正式口径）
python3 scripts/success_rate.py \
  --arm base=runs/ckpt_vla20k --arm finetuned=runs/ckpt_vla_ft4000 \
  --tasks task1 task2 task3 task4 --seeds 0 1 2 3 4 --chunks 40 \
  --out runs/ablations/A1_joint_finetune --video
```

- **输出**：`runs/ablations/A1_joint_finetune/{<arm>/<task>_s<seed>/meta.json, state.jsonl, run.log}`，
  汇总 `rows.json` / `summary.json` / `SUMMARY.md`；视频 `A1_joint_finetune/<arm>/videos/*.mp4`。
- **判读**：episode 是二值的 → 20 单臂粒度 5%，**两臂差 ≥2 个 episode 才算信号**；论文请用
  `--seeds 0..9`（40 单臂）。
- **成本**：≈70 s/episode；4 任务 × 5 seed × 2 arm ≈ 47 min。

### A2 · FM 反传方法

| 项 | 内容 |
|---|---|
| **变量** | 梯度如何穿过 FM 的 K 步 Euler 链 |
| **臂** | `pathwise`（默认，精确 BPTT）/ `checkpointed`（精确，O(1) 激活）/ `truncated`（有偏，廉价）/ `adjoint`（精确离散伴随）；`qvgm_lite` 需 chunk critic，真实栈**不提供**（会显式报错），只在 `ablation_fm_grad/` 的 toy/tiny 里保留 |
| **指标** | `rollout_reward`（主）、`state_term`、`token_mse`、峰值显存、s/step、以及（可选）闭环成功率 |
| **代码** | `diff_vla_wbc/fm_grad.py` + `--fm-grad-backend`；`tests/test_fm_grad_backends.py`（12 用例，含真实 head） |
| **注意** | `adjoint` 会 detach backbone 特征 → **不要**同时用 `--train-vlm` |

```bash
for b in pathwise checkpointed truncated adjoint; do
  O=runs/ablations/A2_fm_backend/$b
  /usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29641 \
    scripts/train_joint.py \
      --motion $MOTION --supermimic-ckpt $SM_CKPT \
      --vlm --vlm-path $VLM_PATH --vlm-ckpt $VLM_CKPT --vlm-dataset --vlm-steps 2 \
      --batch-size 2 --horizon 8 --steps 1000000 \
      --fm-grad-backend $b --time-budget-minutes 30 \
      --hand-loss-weight 0.2 --hand-grasp-boost 5 \
      --eval-every 300 --eval-batches 2 --eval-out $O/eval_curve.jsonl \
      --save-every 300 --save-vla-every 0 --out $O
done
```

- **输出**：`runs/ablations/A2_fm_backend/<backend>/eval_curve.jsonl` + `joint_*.pt` + 训练日志。
- **判读**：先比 `rollout_reward` 的收敛斜率与终值；`truncated` 应更快但更差（有偏）；`adjoint`
  与 `pathwise` 应几乎重合（若明显偏离 → 实现 bug）。
- **成本**：4 × 30 min ≈ 2 h（两卡）。

### A3 · PPO vs BPTT（固定墙钟）

| 项 | 内容 |
|---|---|
| **变量** | 梯度的来源：精确反传 vs score-function |
| **臂** | `train_joint.py`（pathwise） vs `train_ppo.py`（`a = FM_θ(obs,z) + σ·ε`，`z` 每 batch 固定，PPO clip + 学习基线） |
| **指标** | 同墙钟 `rollout_reward` 曲线、`ppo/sigma`、`ppo/clip_frac`、能否闭环 |
| **代码** | ✅；PPO 的**有效 horizon = 1**（VLA 一次吐一个 token chunk，世界模型用同一个 chunk 滚 H 步）→ 是 T=1 的 clipped PG，**论文里不能写成多步 GAE** |
| **注意** | 两臂必须**同一墙钟**（`--time-budget-minutes` 相同），并且都带手部辅助损失（λ=0.2）以保持公平 |

```bash
# BPTT
/usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29642 \
  scripts/train_joint.py --motion $MOTION --supermimic-ckpt $SM_CKPT \
    --vlm --vlm-path $VLM_PATH --vlm-ckpt $VLM_CKPT --vlm-dataset --vlm-steps 2 \
    --batch-size 2 --horizon 8 --steps 1000000 --fm-grad-backend pathwise \
    --time-budget-minutes 120 --hand-loss-weight 0.2 --hand-grasp-boost 5 \
    --eval-every 300 --eval-batches 2 \
    --eval-out runs/ablations/A3_bptt_vs_ppo/bptt/eval_curve.jsonl \
    --out runs/ablations/A3_bptt_vs_ppo/bptt

# PPO
/usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29643 \
  scripts/train_ppo.py --motion $MOTION --supermimic-ckpt $SM_CKPT \
    --vlm --vlm-path $VLM_PATH --vlm-ckpt $VLM_CKPT --vlm-steps 2 \
    --batch-size 2 --horizon 8 --ppo-epochs 4 --sigma-init 0.25 \
    --time-budget-minutes 120 --hand-loss-weight 0.2 \
    --eval-every 100 --eval-batches 2 \
    --eval-out runs/ablations/A3_bptt_vs_ppo/ppo/eval_curve.jsonl \
    --out runs/ablations/A3_bptt_vs_ppo/ppo
```

- **输出**：两个 `eval_curve.jsonl` + `vla_final.pt`（可直接接 A1 的成功率流程）。
- **判读**：先看**能不能动**（reward 是否离开 step-0 基线），再比同墙钟终值；`ppo/sigma` 若塌到下限
  说明策略退化到确定性。
- **成本**：2 × 120 min = 4 h。

### A4 · 世界模型冻结/更新 × imagination horizon

| 项 | 内容 |
|---|---|
| **变量 1** | WM 是否参与联合更新：`--wm-update-every 0`（冻结，默认） vs `--wm-update-every 100 --wm-update-warmup 500`（在线更新） |
| **变量 2** | imagination horizon：`--horizon {8,16,32,40}`（默认 8） |
| **为什么上界是 40** | VLA 一次输出 **40 帧 token chunk**；`imagine_with_tokens` 对 step t 取 `tokens[:, min(t, T-1)]`，所以 horizon=40 正好用完整个 chunk，>40 会退化成重复最后一帧 → **40 是原则性的上界** |
| **代码** | ✅ 纯 CLI（`--wm-update-every/-warmup/-batch/-window`、`--horizon` 都已接线到训练循环） |
| **指标** | `rollout_reward`、`state_term`、`rollout/survival`、显存、s/step |

```bash
# 网格：WM 冻结 × H{8,16,32,40}；WM 更新 × H{8,40}（6 格，够写趋势）
for H in 8 16 32 40; do
  O=runs/ablations/A4_wm_horizon/wm_frozen_H$H
  /usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29644 \
    scripts/train_joint.py --motion $MOTION --supermimic-ckpt $SM_CKPT \
      --vlm --vlm-path $VLM_PATH --vlm-ckpt $VLM_CKPT --vlm-dataset --vlm-steps 2 \
      --batch-size 2 --horizon $H --steps 1000000 --wm-update-every 0 \
      --time-budget-minutes 40 --hand-loss-weight 0.2 --hand-grasp-boost 5 \
      --eval-every 300 --eval-batches 2 --eval-out $O/eval_curve.jsonl --out $O
done
for H in 8 40; do
  O=runs/ablations/A4_wm_horizon/wm_live_H$H
  /usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29644 \
    scripts/train_joint.py --motion $MOTION --supermimic-ckpt $SM_CKPT \
      --vlm --vlm-path $VLM_PATH --vlm-ckpt $VLM_CKPT --vlm-dataset --vlm-steps 2 \
      --batch-size 2 --horizon $H --steps 1000000 \
      --wm-update-every 100 --wm-update-warmup 500 --wm-update-batch 8 \
      --time-budget-minutes 40 --hand-loss-weight 0.2 --hand-grasp-boost 5 \
      --eval-every 300 --eval-batches 2 --eval-out $O/eval_curve.jsonl --out $O
done
```

- **输出**：`runs/ablations/A4_wm_horizon/{wm_frozen_H*,wm_live_H*}/eval_curve.jsonl`；
  WM 更新臂的日志里会多出 `loss/wm_*` 字段，用来确认 WM 真的在更新。
- **判读**：horizon 变大 → 反传链变长、显存与时间上升；看 reward 提升是否值得。WM 在线更新若
  让 reward 抖动，说明 WM 与策略在互相追逐（需要更小的 `--wm-update-every` 频率或更大 warmup）。
- **成本**：6 × 40 min ≈ 4 h。

### A5 · 冻结 SuperMimic，只训 VLA action head

| 项 | 内容 |
|---|---|
| **变量** | 联合微调时是否也更新 WBC（SuperMimic policy） |
| **处理组** | `--policy-lr 0 --adapter-lr 0`（只更新 action head，`--fm-lr 1e-5`） |
| **对照组** | 默认 `--policy-lr 1e-5`（head + policy 一起更新） |
| **指标** | `rollout_reward`、`state_term`、闭环成功率 + **参数变化审计** |
| **代码** | ✅ 纯 CLI。注意：token 路线**不使用 adapter**，所以 `--adapter-lr 0` 只是保险；VLM backbone 本来就被 `freeze_vlm=True` 冻结 |

```bash
# 处理组：只训 head
O=runs/ablations/A5_only_head/head_only
/usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29645 \
  scripts/train_joint.py --motion $MOTION --supermimic-ckpt $SM_CKPT \
    --vlm --vlm-path $VLM_PATH --vlm-ckpt $VLM_CKPT --vlm-dataset --vlm-steps 2 \
    --batch-size 2 --horizon 8 --steps 1000000 --policy-lr 0 --adapter-lr 0 \
    --time-budget-minutes 40 --hand-loss-weight 0.2 --hand-grasp-boost 5 \
    --eval-every 300 --eval-batches 2 --eval-out $O/eval_curve.jsonl --out $O

# 对照组：head + policy
O=runs/ablations/A5_only_head/head_plus_wbc
... 同上，去掉 --policy-lr 0 --adapter-lr 0 ...
```

- **参数变化审计**（证明冻结生效，跑完必做）：
```bash
python3 - <<'PY'
import torch
a = torch.load('runs/ablations/A5_only_head/head_only/joint_latest.pt', map_location='cpu', weights_only=False)
b = torch.load('$SM_CKPT', map_location='cpu', weights_only=False)
pa, pb = a['policy'], sorted(b['policy'].keys())
diff = max(float((a['policy'][k] - b['policy'][k]).abs().max()) for k in pb if k in a['policy'])
print('policy 参数最大变化 =', diff, '（应 == 0 才说明真的冻结）')
PY
```

- **判读**：若 head-only 已经能涨到和 head+WBC 接近，说明收益主要来自 VLA head，**可以省掉 WBC 更新**
  （也顺带避免了"训练代理与部署执行器不一致"的风险放大）。
- **成本**：2 × 40 min ≈ 1.5 h。

---

## 3. 指标字典（论文里每个数字从哪来）

| 指标 | 来源 | 含义 / 怎么比 |
|---|---|---|
| `rollout_reward` | `eval_curve.jsonl` | 想象 rollout 的解析 tracking reward 均值，**主训练指标** |
| `state_term` | 同上 | 想象轨迹与教师状态的误差（越低越好） |
| `token_mse` | 同上 | VLA 输出 token 与录制 token 的 MSE（看是否偏离监督） |
| `hand_mse` | 同上（开手部损失时） | 手部回归误差 |
| `loss/*`（reward/state/smooth/safety） | 训练日志 | 各损失项分量，用来解释 reward 变化的来源 |
| `rollout/survival` | 训练日志 | 想象轨迹的存活率（世界模型是否自洽） |
| `ppo/*`（ratio/clip_frac/sigma/entropy） | PPO 日志 | 判断 PPO 是否健康（ratio→1、clip_frac 低、σ 不塌） |
| 成功率 + Wilson CI | `summary.json` | **A1/A3/A5 的最终裁决指标** |
| `full-clip normalized return` / failed clips | Isaac 侧 `train_log.csv` / `evaluate.py` | WBC 训练质量（与上游可比性用） |
| `it/s`、`peak GiB` | Isaac 侧日志 | 成本核算与规模外推 |

---

## 4. 输出目录规范

```
runs/ablations/
├── A1_joint_finetune/           <arm>/<task>_s<seed>/{meta.json,state.jsonl,run.log}
│                                <arm>/videos/*.mp4
│                                rows.json  summary.json  SUMMARY.md
├── A2_fm_backend/<backend>/     eval_curve.jsonl  joint_*.pt  训练日志
├── A3_bptt_vs_ppo/{bptt,ppo}/   eval_curve.jsonl  vla_final.pt
├── A4_wm_horizon/<arm>_H<N>/    eval_curve.jsonl  joint_*.pt
└── A5_only_head/<arm>/          eval_curve.jsonl  joint_latest.pt
```

汇总（任何时候可跑）：
```bash
python3 scripts/ablation_report.py --root runs/ablations --out docs/ABLATION_REPORT_AUTO.md
```

---

## 5. 执行顺序与卡时预算（两卡 4090）

| 顺序 | 消融 | 成本 | 前置 |
|---|---|---|---|
| 1 | **A5**（只训 head vs head+WBC） | 1.5 h | 无 |
| 2 | **A4**（WM × horizon，6 格） | 4 h | 无 |
| 3 | **A2**（4 后端） | 2 h | 无 |
| 4 | **A3**（BPTT vs PPO） | 4 h | 一次 `train_ppo.py --steps 2` 冒烟 |
| 5 | **A1**（成功率，主结果） | 1 h（不含 arena 场景接入） | **P0-1 场景** |

合计 ≈ **12.5 h**（不含 WBC 训练与场景接入）。建议先跑 A5/A4（便宜、能立刻验证链路），最后跑 A1。

---

## 6. 论文写作映射（fairness 注意事项）

| 消融 | 对应论文元素 | 必须写明的公平性约束 |
|---|---|---|
| A1 | 主表：有无联合微调的成功率 | 两臂同一 WBC、同一评测器、同一 seed 集合；**绝对值不可与 arena 论文比**（版本/资产差见 §1） |
| A2 | 方法对比表（梯度估计） | 同数据、同 seed、同墙钟；`truncated` 注明有偏；`qvgm_lite` 只在 toy 上有，不能混进真实栈结论 |
| A3 | "为什么用 BPTT 而不是 PPO" | 同墙钟、同 reward、同辅助损失；PPO 是 **T=1** 的 clipped PG（不是多步 GAE），必须如实描述 |
| A4 | horizon / WM 消融曲线 | horizon 上界 40 由 token chunk 决定；WM 在线更新要报告更新频率与 warmup |
| A5 | "收益来自哪部分参数" | 用参数审计证明冻结生效；报告头参数量与可训练参数量 |

---

## 7. 尚未实现 / 需人工决策的清单（诚实版）

> 2026-10-24 更新：B2 / B5 / B6 / B7 **已解决并验证**；B1 依赖外部下载；B3 / B4 是剩下的两件工程。

| # | 阻塞 | 状态 | 证据 / 下一步 |
|---|---|---|---|
| **B1** | arena 7 任务场景资产 | ✅ **已完成**（SJTU 网盘包，2.68 GB，解压后 `assets/` 共 4.8 GB；7 个任务逐项验收 **0 缺失**） | 见 §7.5 的下载方式与验收脚本 |
| **B2** | HA 的 python 依赖 | ✅ **已解决** | 装进 `third_party/isaacsim_env`；`pyzmq / logging_mp / onnxruntime / pynput / redis` 可导入；`action_provider.*` 三个关键模块导入通过。`rerun-sdk` 需 numpy>=2 **故意不装**（headless 评测链路不 import 它） |
| **B3** | HA 评测的 VLA HTTP server | ✅ **已实现 + 协议自检通过** | `scripts/ha_vla_server.py`：`POST /infer` → `latent64_chunk (40,64)`、`POST /reset`、`GET /health`。**用 HA 官方客户端实测通过**（见 §7.4） |
| **B4** | SuperMimic 版 action provider | ✅ **已实现 + 自检通过** | `action_provider/action_provider_supermimic.py`（拼 SuperMimic 顺序的 93/99 观测 → Policy → `default_q + action_scale*a` → 43 维关节目标）。**SONIC 已被默认禁用**（见 §7.2） |
| **B5** | Isaac WBC → 联合微调的桥 | ✅ **已实现并数值验证** | `diff_vla_wbc/trackerlab_bundle.py` + `scripts/check_trackerlab_bundle.py`；对着真实 ckpt 跑：状态往返 2.4e-7、policy **0.0**、world model **0.0** |
| **B6** | 手部语义（14 维连续 vs 2 维二值） | ✅ **已实现** | `scripts/hand_interface.py`：双向映射 + 按任务标定阈值；真实数据显示 task_1 **右手不动**（单臂任务），已自动回退到安全阈值 |
| **B7** | 机器人资产换成 arena 带手 USD | ✅ **已实现并验证** | `SUPERMIMIC_ROBOT_USD` 指向 `g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd`（**浮动基座**，不是 base-fix）；实测 `HAS_HANDS=True` |
| — | `train_ppo.py` 端到端 GPU 冒烟 | ⏳ 未做 | A3 开跑前必须做（单测只覆盖数学） |
| — | 联合微调「在 Isaac 里在线训练」 | ❌ 未实现（本手册不用） | 现有实现是离线可微训练（§1.3） |

### 7.1 B5 的接线细节（用的时候必须知道）

Isaac 侧 ckpt（`logs/bones_seed_token_2x8192_isaac50/ckpt_*.pt`）的 `pi` 只有 `dec.*`，
**没有 encoder** —— trackerlab 的 token 路线把 64 维 token 直接当 latent（`Policy(..., token_dim=64)`）。
适配器已按这个变体加载，验证通过。

该 ckpt 的 `obs_mode = relative_ori99`：每帧观测比我们多 6 维
（`mat_to_6d(R_robot^T R_ref)` = 参考根朝向在机器人系下的表达）。**已解决，而且不用重训**：

- `TrackerLabBundle.observation(next_state, action, xi, ref)` 会**现算**这 6 维（用 step 后的
  机器人姿态和当帧 reference），并把观测宽度暴露成 `bundle.obs_width = 99`；
- `flatten_history` 按 `(29,29,3,3,29,6)` 做 term-major 展平（99 宽可用）；
- `diff_vla_wbc/rollout.py` 已改成把 `ref` 透传给 `bundle.observation/act`
  （`_call_observation/_call_act` 带 `TypeError` 回退，所以对老 bundle 完全兼容），
  历史宽度也从 `bundle.obs_width` 读取，不再硬编码 93。

验证（`scripts/check_trackerlab_bundle.py --obs-mode relative_ori99`，对真实 ckpt）：

```
[1/3] state round-trip: rotation max|Δ| = 2.384e-07, other fields 0.000e+00
[2/3] policy            max|Δ| = 0.000e+00
[3/3] world model       max|Δ| = 0.000e+00
[4/4] 6-dim ref-relative max|Δ| = 0.000e+00; flatten round-trip 0.000e+00; policy(99宽) 0.000e+00
```

> 备选（如果将来想要更干净的图）：让 Isaac 侧用 `--obs_mode paper_obs93` 重训，观测就是 93。

### 7.2 SONIC decoder 已从评测链路中移除（默认）

用户要求「把 SONIC 的 decoder 去掉」。做法是**门控而不是删文件**（保留上游代码便于将来对账）：

| 位置 | 改动 |
|---|---|
| `script/eval_scripts/sonic/sim_eval_vla.py` `_normalize_control_routing` | 原来硬编码 `args.action_source = "sonic_wholebody"`；改成默认 `supermimic_wholebody`，SONIC 需 `HA_ALLOW_SONIC=1` 才可用 |
| `action_provider/create_action_provider.py` | 新增 `supermimic_wholebody` 分支并作为默认；`sonic*` 在未放行时 **`SystemExit`** 并给出两条出路 |
| `action_provider/action_provider_supermimic.py` | 新文件：`ActionProvider.get_action(env)` → 43 维关节目标 |

自检（不需要 Isaac）：

```bash
cd third_party/HumanoidArena/isaaclab_twist2_g1
python action_provider/action_provider_supermimic.py --self-test
# [ok] SuperMimicActionProvider self-test passed (action (43,), obs 93, hist 10,
#      physics 8 steps + 2 renders)
```

派发行为实测：

```
default            -> 进 SuperMimicActionProvider（随后因缺 env.scene["robot"] 报 KeyError，符合预期）
sonic_wholebody    -> SystemExit: The SONIC decoder provider is disabled in this project ...
unknown            -> SystemExit: unknown action source: 'nope' (supported: supermimic_wholebody | ...)
```

⚠️ 两点要说清：

1. **MuJoCo 旧闭环**（`closed_loop/code/mosaic_n17/tools/closed_loop_mujoco.py`）里仍然会加载
   `release/model_decoder.onnx`。它已经不是我们的评测路径（评测走 Isaac），所以没有动它；
   要用它跑历史视频就仍然会走 SONIC decoder。
2. 若要和 arena 论文做「同执行器」对比，可以临时 `export HA_ALLOW_SONIC=1` 跑一档 SONIC 基线，
   但必须在论文里标明两档的执行器不同。

### 7.2.1 A1 的两条臂各走自己的执行器（2026-10-24 确认）

这是**设计决定**，不是权宜之计：

| 臂 | VLA checkpoint | 执行器 | 理由 |
|---|---|---|---|
| **A（base）** | `runs/ckpt_vla20k`（官方 20k，未联合微调） | **SONIC decoder**（HA 的 `sonic_wholebody`，即 GR00T N1.7 官方下游链路） | 官方 20k 本来就是**为 SONIC decoder 部署**训的 |
| **B（ft）** | `runs/ckpt_vla_ft4000`（联合微调 4000 步） | **SuperMimic WBC**（`supermimic_wholebody`） | 联合微调正是**为这个 WBC** 优化的 |

一键两臂：

```bash
bash scripts/run_a1_arena.sh ARM=base TASK=HOI_pp_box SEEDS="0 1 2 3 4"   # 自动设 HA_ALLOW_SONIC=1
bash scripts/run_a1_arena.sh ARM=ft   TASK=HOI_pp_box SEEDS="0 1 2 3 4"   # 默认 supermimic
```

输出：`runs/ablations/A1_joint_finetune/{base,ft}/<TASK>/seed_<n>/{result.json,videos/,run.log}`

⚠️ **论文里必须写明**：A1 比较的是**两套完整系统**（VLA + 各自的部署执行器），不是"只比 VLA 权重"。
因为两条臂的执行器不同，这会带来"执行器贡献"和"VLA 贡献"的混淆。

**若要分离这两个因素（推荐，代价只是多跑两格）**：2×2 交叉

| 格 | VLA | 执行器 | 用来回答 |
|---|---|---|---|
| ① | base | SONIC | 官方系统基线 |
| ② | ft | SuperMimic | 我们的系统 |
| ③ | base | SuperMimic | **执行器**换掉但不微调 |
| ④ | ft | SONIC | **微调**了但用回官方执行器 |

- ②−① = 我们的整套收益
- ③−① = 单纯换执行器的贡献
- ④−① = 单纯联合微调的贡献（**这是"反传微调本身有没有用"最干净的对照**）

实现上就是 `--action_source` 与 `HA_ALLOW_SONIC` 的组合，`run_a1_arena.sh` 已支持（臂名与执行器解耦可按需扩展）。

### 7.3 接上 B3 之后 A1 的完整链路（预告）

```
HA eval (Isaac 5.0/Lab 2.2)
  └─ sim_eval_vla.py --action_source supermimic_wholebody
       ├─ HTTP VLA server（B3，返回 latent64）            ← 我们的 VLA（ckpt_vla20k / ft4000）
       └─ SuperMimicActionProvider（B4）
            ├─ 10 帧 × 93(99) 观测历史
            ├─ trackerlab Policy（--sm_ckpt，我们 IsaIac 训的 WBC）
            └─ q_target = default_q + action_scale * a  → 43 维关节 → env.step
```

### 7.4 B3 用法与自检结果

```bash
# 协议自检（不需要 GPU/模型：起一个 dry-run server，用 HA 官方客户端打进去）
python3 scripts/ha_vla_server.py --self-test
# [ok] /infer -> latent64 chunk (40, 64); /reset ok; deterministic

# 真跑：给 A1 的两条臂各起一个（不同 port / 不同 ckpt / 不同卡）
python3 scripts/ha_vla_server.py --ckpt runs/ckpt_vla_ft4000 --port 18080 --device cuda:0
python3 scripts/ha_vla_server.py --ckpt runs/ckpt_vla20k   --port 18081 --device cuda:1

# 客户端（HA 评测）指向它
HA_ALLOW_SONIC=0 bash scripts/run_a1_arena.sh ARM=ft TASK=HOI_pp_box SERVER_URL=http://127.0.0.1:18080
HA_ALLOW_SONIC=1 bash scripts/run_a1_arena.sh ARM=base TASK=HOI_pp_box SERVER_URL=http://127.0.0.1:18081
```

协议细节（**逐字对齐** `action_provider/lerobot_vla_http_client.py`）：

| 项 | 值 |
|---|---|
| 请求 | `POST /infer`，JSON：`observation.images.front.{shape,dtype,data_b64}` + `observation.state`（64 维）+ `robot_type` + `return_chunk` + `task` |
| 响应 | `{"latent64_chunk": [[64 floats] × 40], "meta": {...}}` |
| 复位 | `POST /reset` `{"seed": n}` → `{"ok": true}` |
| 健康 | `GET /health` → `{"ok": true, "infer": n, "errors": n}` |
| 实现 | `Gr00tVlaPolicy`：RGB+vla_state+instruction → GR00T processor → FM 采样 `(40,132)` → 取 `[:, :64]` |

### 7.5 B1：arena 场景包下载命令（**必须在能访问 Google Drive 的机器上执行**）

> 本机实测：`drive.google.com` 元数据页可达（HTTP 200），但真实内容域名
> `doc-0c-*-docs.googleusercontent.com` **连接超时** → 在这台机器上下不了。

```bash
# ① 装下载器（6.x 与 5.x 的 CLI 不一样：6.x 没有 --id，改成位置参数）
python3 -m pip install -U gdown

# ② 下载（两种写法，任选；URL 形式两个版本都认，推荐）
mkdir -p ~/ha_dl && cd ~/ha_dl
gdown "https://drive.google.com/file/d/1TCa_aVRmFrZs_l4wlxkqanNebvDtChNk/view?usp=sharing" \
      -O humanoidarena_assets.zip --continue --retries 5
# 等价（gdown 6.x 位置参数）：
#   gdown 1TCa_aVRmFrZs_l4wlxkqanNebvDtChNk -O humanoidarena_assets.zip
# 等价（gdown 5.x 旧写法，仅旧版）：
#   gdown --id 1TCa_aVRmFrZs_l4wlxkqanNebvDtChNk -O humanoidarena_assets.zip

# ③ 校验压缩包（能列出内容 = 没下坏；期望 GB 级，不是几百 KB）
ls -lh humanoidarena_assets.zip
unzip -t humanoidarena_assets.zip | tail -3

# ④ 传到训练机
scp humanoidarena_assets.zip <user>@<train-host>:/tmp/

# ⑤ 在训练机上解压（仓库里已有 Unitree 的 assets/{model,objects,robots}，这里是**合并**）
cd <comimic>/third_party/HumanoidArena/isaaclab_twist2_g1
unzip -q /tmp/humanoidarena_assets.zip -d assets      # 若包顶层就是 objects/ robots/
# 若解出来是 assets/assets/...（包顶层带 assets/）：
#   mv assets/assets/* assets/ && rmdir assets/assets

# ⑥ 验收 A：只查这次真正缺的 10 个 USD（无输出 = 全齐）
for p in assets/boxing_bag/boxing_bag_physics.usd \
         assets/objects/small_warehouse/SM_Crate_A14_Gray_01/SM_Crate_A14_Gray_01_physics.usd \
         assets/objects/small_warehouse/football_scene/interaction_obj/football_goal/football_goal_physics.usd \
         assets/objects/small_warehouse/football_scene/interaction_obj/soccer_ball/soccer_ball_physics.usd \
         assets/objects/small_warehouse/hammer/hammer.usd \
         assets/objects/small_warehouse/small_cart.usd \
         assets/objects/small_warehouse/small_warehouse_digital_twin_boxtarget.usd \
         assets/objects/small_warehouse/small_warehouse_doubledesk/small_doubledesk.usd \
         assets/objects/small_warehouse/small_warehouse_doubledesk/interaction_obj/hammer/hammer.usd \
         assets/objects/small_warehouse/small_warehouse_opendoor/small_warehouse_digital_twin_opendoor.usd; do
  [ -f "$p" ] || echo "MISS $p"; done

# ⑦ 验收 B：全量对照 base_scene 引用（无输出 = 全齐）
for p in $(grep -rhoE "assets/[A-Za-z0-9_./-]+\.usd" tasks/common_scene/*.py | sort -u); do
  [ -f "$p" ] || echo "MISS $p"; done

# ⑧ 验收 C：带手机器人 USD（B7 依赖）与 7 个任务场景关键词
ls assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd
for k in boxing doubledesk opendoor football sit_sofa; do
  printf "%-12s %s\n" "$k" "$(find assets -iname "*${k}*" | wc -l)"; done
```

**下载失败时的备选**：

1. `gdown` 报 `Too many users have viewed or downloaded this file` → 用浏览器打开
   `https://drive.google.com/file/d/1TCa_aVRmFrZs_l4wlxkqanNebvDtChNk/view`，
   右上角「下载」→ 再 `scp` 过来（最稳）。
2. 想用 `curl` 直连：大于 100 MB 的文件需要确认令牌，建议直接用 `gdown` 或浏览器，别手搓。
3. 如果拿到的是 `assets.zip`（顶层就是 `objects/`、`robots/`）：直接 `unzip -q -d assets`；
   如果顶层是 `assets/`：解到 `isaaclab_twist2_g1` 而不是 `assets/`。

### 7.6 B1 实际下载记录（2026-10-24，SJTU 网盘 · 已成功）

用户提供了交大网盘分享，比 Google Drive 快很多（Drive 在本机不可达）。**完整可复现步骤**：

```bash
# ① 分享信息
#    https://pan.sjtu.edu.cn/web/share/0dbbecb53f8f2d5f960004aed3206a40   提取码 4jo7
#    文件名 humanoidarena_assets.zip，2.68 GB

UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120"

# ② 用提取码换 accessToken（有效期 600 s）
AT=$(curl -s -A "$UA" \
  "https://pan.sjtu.edu.cn/user/v1/share/verify-extraction-code/0dbbecb53f8f2d5f960004aed3206a40?extraction_code=4jo7" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['accessToken'])")

# ③ 换成 S3 预签名直链（302）
URL=$(curl -s -A "$UA" -o /dev/null -w "%{redirect_url}" \
  "https://pan.sjtu.edu.cn/api/v1/file/smh2ax67srucy60s/space01dtsn01rbb7w/humanoidarena_assets.zip?access_token=$AT&user_id=310829&action=info")

# ④ 16 路并发下载（单连接被限速 ~3.5 MB/s；aria2c -x16 实测 ~21 MB/s，6 倍）
aria2c --allow-overwrite=true --auto-file-renaming=false \
       -x16 -s16 -k1M --max-connection-per-server=16 --min-split-size=1M \
       -o ha_assets.zip --dir=. "$URL"

# ⑤ 校验 + 解压合并（仓库里已有 Unitree 的 assets/{model,objects,robots}）
unzip -t ha_assets.zip | tail -1        # 期望 "No errors detected"
cd third_party/HumanoidArena/isaaclab_twist2_g1
unzip -q -o _dl/ha_assets.zip -d assets
```

**验收结果（逐任务、剔除注释行后）**：

| 任务 | 场景文件 | 引用 USD | 缺失 |
|---|---|---|---|
| HOI_double_desk | `base_scene_pickplace_doubledesk.py` | 3 | **0** |
| HOI_pp_box | `base_scene_pickplace_box.py` | 0（路径拼接） | **0** |
| HSI_boxing | `base_scene_boxing_bag.py` | 0（路径拼接） | **0** |
| HSI_open_door | `base_scene_open_door.py` | 1 | **0** |
| HSI_sit_sofa | `base_scene_smalllivingroom_sitingsofa.py` | 0 | **0** |
| HSI_vision_navi | `base_scene_small_warehouse_vision_navigation.py` | 0 | **0** |
| HOI_football | `base_scene_football_single_cfg_wholebody.py` | 2 | **0** |

> 注意两个「假缺失」，之前按全仓库 grep 会误报：
> 1. `small_cart.usd` 只出现在**注释**里（`# usd_path=...small_cart.usd`），实际用的是 `small_doubledesk.usd`（存在）；
> 2. `boxing_bag_physics.usd` 属于 **legacy** 变体 `base_scene_boxing_bag_cfg_wholebody.py`，
>    而 `move_boxing_bag_..._wholebody` 任务实际 import 的是 `base_scene_boxing_bag`（新布局，资产齐全）。
>
> 验收脚本务必**剔除注释行**，否则会误判。

资产总量：`assets/` = **4.8 GB**（Unitree 包 1.31 GB + HA 场景包 2.68 GB 解压后合并）。

### 7.7 ⚠️ 并发会话对我这些文件的改动（重要）

2026-10-24 12:26，另一条会话给 `action_provider_supermimic.py` 加了 `_step_physics`，内容很关键：

- HA 的评测循环跑在 `ControlConfig(use_rl_action_mode=True)` 下，**会跳过 `env.step(action)`**，
  所以 provider 必须自己推进物理（上游 SONIC provider 也是这么做的）；否则关节目标永远写不进去，
  机器人站着不动、每次都是 0 分。
- `sim.step(render=False)` 不会刷新 RTX 相机，必须每控制步 `sim.render()` 一次，
  否则 `front_camera` 输出冻结在第一帧，VLA 会一直看到同一张图（chunk 重复）。

我这边只做了**加法**（把 self_test 的假 env 补齐 `cfg.decimation / physics_dt / sim / scene`，
并把断言扩展到"物理真的推进了"），**没有改动他们的逻辑**。改完自检：
`(action (43,), obs 93, hist 10, physics 8 steps + 2 renders)`。

> 交接提醒：这个文件是**两条会话共同编辑**的。下次改之前先 `git -C ... diff` 或看 mtime，
> 避免互相覆盖。
