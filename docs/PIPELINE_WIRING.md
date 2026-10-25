# 管线对接清单（只做接线，不跑任何训练）

> 目标：**SuperMimic 训完之后**（Isaac Sim / Isaac Lab，另一个容器里跑的那份），
> ① 能直接接上联合微调；② 能直接在 **Isaac Lab 里闭环推理**，且**场景、初始位姿与训练完全一致**。
>
> 本文只描述接口与接线，每条都标了「已验证 / 待实现」。

---

## 0. 三个系统，各自负责什么

| 系统 | 在哪 | 输入 → 输出 | 现状 |
|---|---|---|---|
| **SuperMimic 训练**（trackerlab-supermimic，IsaacLab 2.3.2 + Isaac Sim 5.1 + PhysX） | **另一个容器**（我们不动它） | BONES-SEED 数据 → `pi`（token→29 动作）+ `wm`（70 维动力学） | 在训；产出 `ckpt_XXXXXX.pt` / `last.pt` |
| **我们的 VLA**（stage O：GR00T-N1.7-3B 微调） | 本容器 | 图像 + 46 维状态 + 指令 → 78 维 chunk（64 token + 14 手） | ✅ 训完并**写死**：`runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（见 `runs/OFFICIAL_CKPT.json`） |
| **HumanoidArena Isaac 评测** | 本容器（Isaac Lab 2.2.0 / Isaac Sim 5.0.0，已装） | `sim_main.py` + 官方场景 | 环境装好，未跑 |

---

## 1. 接缝 A：VLA → Isaac Lab 闭环推理（**已完成并验证协议**）

HumanoidArena 的运行时用纯 HTTP 向策略要动作
（`isaaclab_twist2_g1/action_provider/lerobot_vla_http_client.py`）：

```
POST /infer
{ "observation": { "images": { "front": {shape,dtype,data_b64} }, "state": [64 floats] },
  "robot_type": "...", "return_chunk": true, "task": "<instruction>" }
→ { "latent64_chunk": [[64 floats] ...] }        # SONIC_VLA_ACTION_FORMAT=latent64
```

我们这边的实现：**`scripts/ha_vla_latent64_server.py`**（新增）

| 事项 | 处理方式 |
|---|---|
| 状态维度不一致 | 他们的 `vla_state`(64) = rot6d(6)+dof_pos(29)+dof_vel(29)；我们的 `state_46` = 关节29 + 手14 + 重力3。29 关节直通，重力由 rot6d 算（`R^T[0,0,-1]`，row 布局），**手不在他们的 state 里** → 用上一帧自己发出的 14 维回灌（`--hand-source feedback`）或置零 |
| 手 14 → 2 | `max(手指) > 1.70` → binary；实测与 HA 自己的 `vla_action_hand_binary` **100% 一致** |
| 输出 | `latent64_chunk`（horizon × 64）+ 附带 `hand_binary_chunk`（本容器内可读；官方 runtime 只吃 64 或 66） |
| 已验证 | 用 **HA 自己的 `LeRobotVLAHttpClient`** 打我们的 server（`--dry-run`，返回零）：解析出 `(40, 64) float32`；`gravity_from_rot6d([1,0,0,0,1,0]) = [0,0,-1]`；`state64→46` 得 46 维。**协议层完全对得上** |

启动方式（推理时用，不是训练）：

```bash
# 真实 VLA（占 1 张卡的推理显存）
python3 scripts/ha_vla_latent64_server.py \
  --ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 --port 18080

# Isaac 侧（官方入口）
SONIC_VLA_ACTION_FORMAT=latent64 \
python isaaclab_twist2_g1/sim_main.py --input_source vla --gmt_backend sonic \
  --lerobot_server_url http://127.0.0.1:18080 --sonic_decoder_path <model_decoder.onnx> \
  --task Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody --robot_type g129 --enable_cameras
```

**待实现（1 处）**：`--dry-run` 只验证了协议；真实推理需要一次 GPU 冒烟，确认
`process_observation` 对 `video.ego_view` 的尺寸/layout 期望与我们的 `(1,1,H,W,3) uint8` 一致
（H,W 目前取自 processor 的 `image_target_size`，若为空默认 224×224）。

---

## 2. 接缝 B：SuperMimic（Isaac 版）ckpt → 联合微调（**待实现适配层**）

两边的 ckpt 结构不同，这是唯一需要写代码的地方：

| | trackerlab（Isaac 版，在训的那份） | 我们的 `diff_vla_wbc`（联合微调用） |
|---|---|---|
| 文件 | `ckpt_XXXXXX.pt` / `last.pt` | `ckpt_0024000.pt`（SuperMimic-Reproduced 导出） |
| 键 | `pi` / `wm` / `critic` / `args` / `reward_cfg` | `policy` / `wm` / `critic` / 优化器 |
| 模块命名 | `enc.*` / `net.*` | SuperMimic 自己的模块名 |
| 观测 | `observation_mode: relative_ori99`，hist 10，goal_frames 10 | `obs_flat` = 10×93 = 930，token 64 |
| 动作 | 29（`JointPositionAction`） | 29（同） |
| 奖励 | `reward_cfg`：`w_body_pos/w_body_ori/w_body_linvel/w_body_angvel/w_root_pos/w_root_ori` + 各 std | 8 项解析奖励（SuperMimic 论文权重） |

适配层要做的三件事（全部离线、不训练）：

1. **读**：`torch.load(ckpt)` → 取 `pi` / `wm` 的 state_dict + `args`（含 `num_envs/horizon/hist/goal_frames`）+ `reward_cfg`；
2. **映射**：把 `pi.enc.*` / `wm.net.*` 的键映射到 `SuperMimicBundle` 的模块命名，并**逐层比对 shape**（形状不一致就报错而不是静默加载）；
3. **对齐观测/归一化**：`relative_ori99` 的观测布局 + critic 的 `norm.mean/var` 必须与我们的 `obs_flat` 构造一致；不一致时保存一份「观测构造映射表」，而不是靠猜。

验证方式（不需要 GPU）：加载后跑一次前向，比较同一 token 下 Isaac 版 pi 与我们 pi 的输出差；若差异过大，说明映射错。

---

## 3. 接缝 C：场景与初始位姿统一到 HA（**待接**）

这里的「不一致」有两个来源，都要以 HA 为准：

| 来源 | 现状 | 统一做法 |
|---|---|---|
| 训练场景 | SuperMimic 训练用的是 LaFAN1/BONES-SEED 的动作跟踪场景（`mjlab`/trackerlab 的默认安全地板） | 闭环推理与后续训练统一用 **HA 的 7 个任务场景**：`isaaclab_twist2_g1/tasks/...`，任务名如 `Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody` |
| 初始位姿 | 我们之前的 MuJoCo 代理场景是自造的 | 直接用 **HA 录制里的真值**：`robot_root_position[0]`（机器人初始根位姿）、`episode_init_env_obj_*`（每个物体初始位姿）、以及 `tasks/common_env_config/*_sonic.yaml` 里的 `pose_range`（reset 范围）与 `--seed` |

具体落点：

1. 场景：`sim_main.py --task <HA 任务名> --env_config <对应 *_sonic.yaml>`；
2. 机器人初始位姿：HA 的 `episode_object_seed` / `deterministic_object_resets` 机制已支持；
   要复现**某一条录制**的初始状态，用 raw npz 里的 `episode_init_env_obj_*` + `robot_root_position[0]` 写成一个 seed 覆盖；
3. 相机：HA 的 `front` 相机参数在任务配置里，推理必须用同一个（否则视觉分布又变了）；
4. 校验：录一段 replay，把首帧图像与我们训练数据首帧（`front_rgb.mp4` 第 0 帧）做像素级对比，
   **不通过就不算"场景一致"**。

---

## 4. 端到端状态表

| 接缝 | 状态 | 缺什么 |
|---|---|---|
| VLA → Isaac（`/infer` 协议 + 64 维 token） | ✅ 协议已验证；实现已落地 | 一次 GPU 推理冒烟（非训练） |
| SuperMimic(Isaac) ckpt → 联合微调 | ✅ 适配器已落地并实测（`scripts/adapt_trackerlab_ckpt.py`） | 等 token 接口 ckpt 出现后跑一次转换；当前 motion 接口 ckpt 实测 26 层 1:1、8 层不匹配（见下） |
| SuperMimic(Isaac) ckpt → Isaac 闭环（当 WBC） | ❌ 未实现 | 同上适配层 + 在 `sim_main` 侧提供 `token→29` 的本地策略接口（替代 ONNX decoder） |
| HA 场景 + 初始位姿统一 | ✅ 工具已落地并实测（`scripts/ha_scene_bootstrap.py`） | 在 Isaac 侧把 JSON 喂给 reset；首帧像素校验需等 Isaac 跑起来 |

---

## 5. 本轮落地的两个工具（都已在真实数据上验证，且不训练）

### `scripts/adapt_trackerlab_ckpt.py`（接缝 B）

实测（`logs/bones_seed_bench_2x4096/ckpt_000060.pt`，motion 接口）：

```
mapped 1:1 : 26      # dec.0.bias + dec.2..8 + enc.2..8 + WM 全部线性层
mismatched : 8
  pi.dec.0.weight [2048,1022] vs [2048,994]    # obs 历史布局不同（99 vs 93 /帧）
  pi.enc.0.weight [2048,730]  vs [2048,680]    # goal 布局不同（73 vs 68 /帧）
  wm.in_norm / out_norm (6 个)                  # 他们的 WM 有 I/O 归一化，我们没有
notes:
  他们的 WM 用 LayerNorm（net.1/4/7，6 个张量），我们的 dynamics MLP 没有
  token 接口会移除 encoder（enc 输入宽度变成 64）→ token 版 ckpt 会更接近我们的布局
```

行为：**任何一层形状不一致就拒绝转换**（`--allow-partial` 仅作检查用），并输出 JSON 审计报告。
将来 token 接口 ckpt 出来后：先跑审计 → 只要剩下的是"I/O 归一化/LayerNorm"这类结构性差异，
就用 `--allow-partial` 生成一份"能 1:1 的层全拷、其余随机初始化"的 warm start 文件，
并在联合微调的日志里显式记录哪些张量是重新初始化的。

### `scripts/ha_scene_bootstrap.py`（接缝 C）

三条子命令，实测输出：

```
init --npz <ep>      → robot [-3.7, -2.2, 0.8]
                       basket: start [-2.83, -0.1, 0.75]  + recorded pos/ori/vel
                       hammer: start [-2.63, -3.13, 0.80] + recorded pos/ori/vel
                       yaml: doubledesk_sonic.yaml (2 reset blocks: hammer / basket)
first-frame --npz <ep> → runs/ha_bootstrap/first_frame.png（480×640，录制第 0 帧）
compare --reference <png> --sim <png>
                       → mae / psnr / hist_corr / edge_corr + MATCH|MISMATCH
                         自比: mae=0.0, psnr=inf, MATCH  ✅
```

用法：Isaac 侧把 `init` 的 JSON 当作 reset 覆盖（机器人位姿 + 每个物体的初始位姿/朝向/速度），
跑起来后导出首帧，用 `compare` 与录制首帧比；**MISMATCH 就不算"场景一致"**。
