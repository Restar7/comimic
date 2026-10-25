# 交接文档：HumanoidArena Isaac 闭环（Isaac Sim 5.0 + 我们的 VLA）

> **用途**：在 **570 驱动**的环境里把 HumanoidArena 的 Isaac Sim 闭环跑起来，驱动我们的 VLA
> （出视频 + 成功率）。写入时间：2026-10-24。
>
> **为什么换环境**：本容器驱动是 **595.71.05**，实测 **Isaac Sim 5.0 起不来**（最小
> `SimulationApp` 都段错误：`Warp CUDA error: Failed to get driver entry point
> 'cuDeviceGetUuid'`），而同一容器里 **Isaac Sim 5.1 能正常启动**（`SIMAPP_51_OK`）。
> 结论：595 驱动与 Isaac Sim 5.0 不兼容 → 需要 570 驱动的容器跑 5.0。

---

## 0. 分工（谁跑在哪）

```
[570 容器] Isaac Sim 5.0 + IsaacLab 2.2.0
    sim_main.py  --input_source vla --gmt_backend sonic
        │  POST /infer  (HTTP, 图像 + 64 维 state + 指令)
        │  ← {latent64_chunk: [[64 floats] × 40]}
        ▼
[任意容器 / 本容器] 我们的 VLA 服务
    scripts/ha_vla_latent64_server.py
        → GR00T-N1.7-3B（stage O 微调后的 checkpoint）
        → 64 维 token + 14 维手（内部压成 2 维 binary）
```

两边只通过 **HTTP** 耦合，所以可以跨容器/跨机器；除端口外无需共享内存。

---

## 1. 570 容器的环境安装（按序执行）

目标版本（HumanoidArena 官方口径）：**Python 3.11 / torch 2.7.0+cu128 / Isaac Sim 5.0.0 /
IsaacLab release/2.2.0**。

```bash
# 1) PyTorch
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128

# 2) Isaac Sim 5.0.0（约 15 GB）
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

# 3) Isaac Lab release/2.2.0
git clone --depth 1 -b release/2.2.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && ./isaaclab.sh --install      # ← 若报 torch 版本冲突，用下面的替代方案
```

### 1.1 三个必踩的坑（我们全踩过，照抄可省时间）

| 症状 | 原因 | 处理 |
|---|---|---|
| `EOFError: Do you accept the EULA?` | 无交互 shell 下 import isaacsim 会问 EULA | 所有命令前置 `export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y` |
| `isaaclab.sh --install` 报 `ResolutionImpossible: torch==2.8.0a0+...nv25.6` | isaaclab-rl 要 torch>=2.7，Isaac Sim 侧钉 nv 版 torch | 绕过解析器逐包装：`pip install --no-deps --no-build-isolation -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_tasks` |
| `No module named pip` | uv venv 默认不带 pip | `uv pip install -p <venv>/bin/python pip setuptools wheel` |
| `ModuleNotFoundError: carb` | `carb` 只在 Kit app 启动后才可 import | 正常现象：脚本必须先 `AppLauncher` 再 import isaaclab（`sim_main.py` 就是这么写的） |

### 1.2 系统库（headless 渲染必需）

```bash
apt-get install -y --no-install-recommends \
  libxt6 libglu1-mesa libglfw3 libx11-6 libxi6 libxrandr2 libxinerama1 \
  libxcursor1 libsm6 libice6 libegl1 libgl1 libvulkan1 mesa-vulkan-drivers \
  cyclonedds-dev
```

### 1.3 HumanoidArena 的 Python 依赖

`unitree_sdk2py` **不在 PyPI**。HumanoidArena 自带一份，直接进 `PYTHONPATH`（不用编译）：

```bash
export PYTHONPATH=<HumanoidArena>/external/unitree_sdk2_python:$PYTHONPATH
python -c "import unitree_sdk2py.core.channel"   # 应无报错
```

它依赖 `cyclonedds`（Eclipse 绑定）：`pip install cyclonedds`（需要 `cyclonedds-dev`）；
若构建时找不到库，把 `CYCLONEDDS_HOME` 指到含 `include/dds` 与 `lib/libddsc.so` 的前缀
（我们用的办法：软链 `/usr/include` 与 `/usr/lib/x86_64-linux-gnu` 到一个合成前缀）。

---

## 2. 场景资产

```bash
export PROJECT_ROOT=<HumanoidArena>/isaaclab_twist2_g1
ls $PROJECT_ROOT/assets/{objects,robots}     # 3.8 GB，已解好，无需再下载
```

代码里按 `{PROJECT_ROOT}/assets/objects/...` 拼路径，所以 `PROJECT_ROOT` 必须指向
`isaaclab_twist2_g1`。资产包来源：官方 Drive
`humanoidarena_assets_isaaclab_twist2_g1_20260703_release_clean.zip`（2.68 GB）。

---

## 3. 我们这边：VLA 服务（已在跑，可直接复用）

**位置**：`comimic/scripts/ha_vla_latent64_server.py`
**当前状态**：在本容器已加载模型并监听（`curl http://127.0.0.1:18080/` → `{"ok": true, ...}`）。

### 3.1 启动

```bash
export PYTHONPATH=<comimic>/third_party/gr00t_deps:<comimic>/third_party/gr00t_deps/_noflash
python3 scripts/ha_vla_latent64_server.py \
  --ckpt <comimic>/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \   # ← 写死的官方微调产物（runs/OFFICIAL_CKPT.json）
  --port 18080 --device cuda:0 --bf16 --horizon 40 --fm-steps 4
```

`PYTHONPATH` 两段都不能少：
- `gr00t_deps`：`tyro / transformers / onnxruntime` 等 GR00T 私有依赖（缺了报 `No module named 'tyro'`）；
- `gr00t_deps/_noflash`：**故意抛 ImportError 的假 flash_attn**，逼 transformers 回落 sdpa。反传微调/推理里 VLM 输入是 fp32，真 FlashAttention 只收 fp16/bf16，会直接崩。

### 3.2 协议（与 `action_provider/lerobot_vla_http_client.py` 完全对齐）

```
POST /infer
{ "observation": { "images": { "front": {shape:[H,W,3], dtype:"uint8", data_b64:"..."} },
                   "state": [64 floats] },
  "robot_type": "unitree_g1_refpose_v3_1", "return_chunk": true, "task": "<instruction>" }
→ 200 { "latent64_chunk": [[64 floats] × 40], "hand_binary_chunk": [[2 floats] × 40], "dt_ms": ... }
```

服务端内部两处转换（都测过）：

1. **64 → 46 维状态**：他们的 `vla_state` = rot6d(6) + dof_pos(29) + dof_vel(29)；
   我们的模型要 `joints29 + hands14 + projected_gravity(3)`。
   - 29 关节直通；
   - 重力 = `Rᵀ[0,0,-1]`（rot6d **row 布局**，实测 `identity → [0,0,-1]`）；
   - 手不在他们的 state 里 → 用自己上一帧输出的 14 维回灌（`--hand-source feedback`）。
2. **14 → 2 维手**：`max(手指) > 1.70` 判为闭合。实测与 HA 自己的 `vla_action_hand_binary`
   **100% 帧一致**（手指量程 ±1.74）。

### 3.3 单独自测（不需要 Isaac）

```bash
# 协议自测（无模型，返回零）
python3 scripts/ha_vla_latent64_server.py --dry-run --port 18091 &
python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0,'<HumanoidArena>/isaaclab_twist2_g1')
from action_provider.lerobot_vla_http_client import LeRobotVLAHttpClient
print(LeRobotVLAHttpClient("http://127.0.0.1:18091").infer_chunk(
    np.zeros((480,640,3),np.uint8), np.zeros(64,np.float32),
    robot_type="unitree_g1_refpose_v3_1", task="pick up the box").shape)   # 期望 (40, 64)
PY
```

---

## 4. Isaac 侧：闭环命令

### 4.1 先验证场景能起（replay，不接 VLA）

```bash
cd $PROJECT_ROOT
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export PYTHONPATH=$PROJECT_ROOT/../external/unitree_sdk2_python:$PYTHONPATH
python sim_main.py --headless --enable_cameras \
  --task Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody --robot_type g129 \
  --input_source replay --replay_file <一条 HA raw npz> \
  --gmt_backend sonic --sonic_decoder_path <model_decoder.onnx> --seed 0
```

`model_decoder.onnx` 取 `models--nvidia--GEAR-SONIC/snapshots/*/low_latency/model_decoder.onnx`。

### 4.2 接我们的 VLA（latent64）

```bash
SONIC_VLA_ACTION_FORMAT=latent64 \
python sim_main.py --headless --enable_cameras \
  --task Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody --robot_type g129 \
  --input_source vla --gmt_backend sonic \
  --lerobot_server_url http://<VLA 服务地址>:18080 \
  --sonic_decoder_path <model_decoder.onnx> --seed 0
```

官方说明：该接口接受 **64 维 latent**，或 **66 维 = 64 + 左右手 binary**；我们返回的是
`latent64_chunk`（64 维），手部以 `hand_binary_chunk` 附带返回（如需 66 维，把两者拼接即可）。

### 4.3 七个任务的 task 名

```
HOI_double_desk : Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody
HOI_grap_cup    : Isaac-Move-ArtVIP-Livingroom-GrapCup-G129-Dex3-Wholebody
HOI_pp_box      : Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody
HSI_boxing      : Isaac-Move-Boxing-...-G129-Dex3-Wholebody
HSI_open_door   : Isaac-Move-Open-Door-G129-Dex3-Wholebody
HSI_sit_sofa    : Isaac-Move-Sit-Sofa-...-G129-Dex3-Wholebody
HSI_vision_navi : Isaac-Move-Vision-Navigation-...-G129-Dex3-Wholebody
```

任务名以 `isaaclab_twist2_g1/tasks/` 里实际注册的为准（`grep -r "gym.register" tasks/`）。
对应的 env 配置在 `tasks/common_env_config/*_sonic.yaml`（`--env_config` 可覆盖）。

---

## 5. 场景/初始位姿一致性校验（**必做**，否则"闭环"没有意义）

训练数据是 Isaac Sim 录的，闭环必须用**同一个场景 + 同一初始位姿**。工具：
`comimic/scripts/ha_scene_bootstrap.py`

```bash
# (1) 从 raw npz 抽出这一集的初始状态（机器人 + 每个物体）
python3 scripts/ha_scene_bootstrap.py init --npz <raw npz> --task HOI_double_desk
#   → runs/ha_bootstrap/...json：robot pos/ori、basket/hammer 的 measured start + recorded init pos/ori/vel
#     + yaml 里的 reset 范围（doubledesk_sonic.yaml 的 hammer/basket 两个 block）

# (2) 录制第 0 帧导出（作为参照）
python3 scripts/ha_scene_bootstrap.py first-frame --npz <raw npz> --out runs/ha_bootstrap/first_frame.png

# (3) Isaac 跑起来后导出仿真的首帧，比对
python3 scripts/ha_scene_bootstrap.py compare \
  --reference runs/ha_bootstrap/first_frame.png --sim <sim_first_frame.png>
#   → {"mae":..., "psnr_db":..., "hist_corr":..., "edge_corr":..., "verdict":"MATCH|MISMATCH"}
#   自比基线：mae=0.0, psnr=inf, MATCH
```

**规则：`verdict` 不是 `MATCH` 就不算"场景一致"**，先修场景/相机/seed，再谈成功率。

---

## 6. Checkpoint 清单（用哪个）

| 路径 | 说明 | 训练 loss |
|---|---|---|
| `runs/groot_sonic_vla_ha_b64/checkpoint-10000` | stage O 第一轮终值（**历史，不再使用**） | 0.0460 |
| `runs/groot_sonic_vla_ha_b64_cont/checkpoint-500/1000/1500` | lr=1e-4 续训（**跑坏了，仅作对照**） | 0.062 |
| **`runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（写死）** | **官方微调产物 = `runs/OFFICIAL_CKPT.json`；所有联合微调 / 消融 / VLA 服务默认都用它** | **0.0401 ± 0.0041**（该 run 现存最优） |

**推荐**：等 lr=1e-5 续训跑完（10000 步）用它的 `checkpoint-10000`；中途想跑闭环就用最新的
`checkpoint-3000` 之类，效果差别在几个百分点内。

---

## 7. 已知坑与诊断表

| 现象 | 结论 |
|---|---|
| `Warp CUDA error: Failed to get driver entry point 'cuDeviceGetUuid'` + `librtx.scenedb.plugin.so` 段错误（**连最小 SimulationApp 都复现**） | **Isaac Sim 5.0 与 595 驱动不兼容**；同一容器里 5.1 正常。解法：换 570 驱动环境（本文档的前提） |
| `/proc/driver/nvidia/gpus` 条目数 ≠ `nvidia-smi -L` 条目数 | 容器 GPU 可见性不一致，是潜在隐患（我们环境是 8 vs 2） |
| `GLInteropContext::init: carb::windowing is not available` | 缺 X11/GL 库，按 §1.2 装齐即可（headless 下是无害 warning） |
| 0 字节的 `libcuda.so.550.*`/`570.*` 残留 | 会误导按版本号探测驱动的代码；我们已把它们移到 `comimic/third_party/disabled_libs/` |
| `No module named 'tyro'`（我们的服务端） | 少了 `gr00t_deps` 到 PYTHONPATH |
| `FlashAttention only support fp16 and bf16`（我们的服务端） | 少了 `gr00t_deps/_noflash` 到 PYTHONPATH |

---

## 8. 验收清单（按顺序）

1. ☐ `import unitree_sdk2py.core.channel` 通过
2. ☐ `sim_main.py --help` 退出码 0
3. ☐ replay 一条 HA episode 能起、能出画面（证明场景+机器人+资产 OK）
4. ☐ `ha_scene_bootstrap.py compare` 的 `verdict == MATCH`（证明场景与训练一致）
5. ☐ VLA 服务自测返回 `(40, 64)`；Isaac 侧日志里出现对 `/infer` 的请求
6. ☐ 单任务 vla 闭环跑完，出视频
7. ☐ 七个任务批量，统计成功率

---

## 9. 本容器当前状态（交接时点）

| 项 | 状态 |
|---|---|
| lr=1e-5 续训 | 运行中，step ~3000/10000，loss 0.038–0.042，ETA ≈ 1 小时 50 分 |
| VLA 服务 | 运行中（GPU0，13 GB），`{"ok": true}` |
| Isaac Sim 5.0 | ❌ 本容器不可用（595 驱动）→ 交接到 570 环境 |
| Isaac Sim 5.1 | ✅ 已装（`third_party/isaacsim51_env`），最小 app 可启动（备用路线） |
| IsaacLab 2.2.0 | ✅ 已装（本容器，配 5.0 用；5.0 起不来所以未验证） |
| 相关脚本 | `scripts/ha_vla_latent64_server.py`、`scripts/ha_scene_bootstrap.py`、`scripts/adapt_trackerlab_ckpt.py` |
| 相关文档 | `docs/PIPELINE_WIRING.md`（三处接缝全貌）、`docs/HA_RAW_PIPELINE.md`（数据→微调链路） |

---

## 10. 570 环境实测结论（2026-10-24，已跑通）

**结论：本文档的前提成立。`570.124.06` 驱动下 Isaac Sim 5.0.0 + IsaacLab 2.2.0 可以起，
闭环可以跑通。** 595 上那个 `cuDeviceGetUuid` 段错误在本容器不存在。

| 验证项 | 命令/证据 | 结果 |
|---|---|---|
| 驱动 | `nvidia-smi` | 570.124.06 / CUDA 12.8 / 2× RTX 4090 |
| 最小 `SimulationApp` | `/tmp/simapp_min.py` | ✅ 构造 55.8s，`app.update()` ×5 正常，`close()` 退出码 0 |
| IsaacLab 2.2.0 | `import isaaclab`（配 5.0 venv） | ✅ editable 安装可用 |
| 场景 | doubledesk | ✅ `create environment success`，前视相机 640×480 |
| 我们 VLA 真推理 | HA client → `/infer` | ✅ `(40,64)`，稳态 **~150 ms/次**（首帧 0.9 s） |
| 单集闭环 | `scripts/ha_isaac_closedloop.sh` | ✅ 300 步跑满、出 mp4 + `result.json` |

### 10.1 本次补掉的 5 个缺口（交接文档里没写）

| # | 症状 | 根因 | 修复 |
|---|---|---|---|
| 1 | `third_party/isaacsim_env/bin/python: No such file or directory` | venv 的解释器软链指向 `/root/.local/share/uv/python/cpython-3.11-*`，新容器里没有 | `ln -sfn $COMIMIC/third_party/uv_python/cpython-3.11.16-linux-x86_64-gnu /root/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu` |
| 2 | RTX 报 `libneuray.so: libGLU.so.1: cannot open shared object file` | 只装了部分 X11 库 | `apt-get install -y libglu1-mesa libegl1 libxt6 libglfw3 libxi6 libxcursor1 libxinerama1 libxrandr2 libsm6 libice6` |
| 3 | `Failed to create environment: USD file not found ... small_doubledesk.usd` | 资产只有 **1.7 GB**（解压中断），场景 USD 缺 | 用官方包补全：`unzip -n humanoidarena_assets_isaaclab_twist2_g1_20260703_release_clean.zip -d assets` → 现 **4.8 GB**（`qwk-workspace/` 里就有那份 zip） |
| 4 | 日志 `Skipping NVIDIA GPU due CUDA being in bad state` | carb 不认 `CUDA_VISIBLE_DEVICES` | **不要**用它选卡，只能用 `--device cuda:N` |
| 5 | `[SonicActionProvider] model path is empty` | 上游 provider 同时加载 encoder **和** decoder | `--sonic_encoder_path` 也必须给（本文档 §4.1 只写了 decoder） |

校验资产齐不齐（7 个任务）：
```bash
cd $PROJECT_ROOT
python3 - <<'PY'
import re, glob, os, pathlib
need = set()
for d in ("move_pickplace_doubledesk_g1_29dof_dex3_wholebody","move_pickplace_box_g1_29dof_dex3_wholedoby",
          "move_boxing_bag_g1_29dof_dex3_wholebody","move_open_door_g1_29dof_dex3_wholebody",
          "move_sit_sofa_g1_29dof_dex3_wholebody","move_small_warehouse_vision_navigation_g1_29dof_dex3_wholebody"):
    for py in glob.glob(f"tasks/g1_tasks/{d}/*.py"):
        for m in re.finditer(r"from (tasks\.common_scene\.[\w.]+) import", open(py, errors="replace").read()):
            need.add(m.group(1).replace(".", "/") + ".py")
missing = [(p, u) for p in need for u in
           re.findall(r"assets/[A-Za-z0-9_./-]*?\.usd", open(p, errors="replace").read())
           if not pathlib.Path(u).exists() and not u.startswith("assets/objects/small_warehouse/small_cart")]
print("MISSING:", missing or "none")
PY
```

### 10.2 执行器：默认是**我们自己的 WBC**，不是上游 SONIC

`action_provider/create_action_provider.py` 把项目默认改成了 `supermimic_wholebody`
（`action_provider/action_provider_supermimic.py`，我们写的胶水层）：
`VLA latent64 → SuperMimic policy(obs99, token64) → 29 关节目标`。
上游 SONIC decoder 必须显式 `export HA_ALLOW_SONIC=1` 才会加载 —— 这是故意的，避免评测
悄悄换成"我们没训过的执行器"。

因此 **`--sm_ckpt` 是必填**（provider 里 `SystemExit: --sm_ckpt is required`）。我们用的是
token-conditioned 那一版：
```
$COMIMIC/trackerlab-supermimic/logs/bones_seed_token_2x8192_isaac50/last.pt   # 含 "pi" 键
```
`ckpt_002500.pt` / `last.pt` 都能用（前者是轻量导出快照）。

为此给官方 eval 脚本补了 4 个参数（`script/eval_scripts/sonic/sim_eval_vla.py`）：
`--sm_ckpt --sm_obs_mode --sm_token_source --sm_token_dim`，默认值可用
`SM_CKPT / SM_OBS_MODE / SM_TOKEN_SOURCE / SM_TOKEN_DIM` 环境变量给。
obs 维度由 ckpt 反推：`dec.0.weight` 的输入 = `hist(10)×obs_dim + token_dim` =
`10×99 + 64 = 1054` → `obs_mode=relative_ori99`（provider 默认值正好对）。

### 10.3 语言指令原来恒为空（已补）

`sim_eval_vla.py` 的 `_normalize_control_routing()` 会把 `language_instruction` 清空，
provider 又把 `sv_task_name`（全仓库没人设置过）当指令传给 `/infer` → **VLA 一直收到空指令**。
训练时用的是 `scripts/prepare_ha_sessions.py` 里的 `TASK_INSTRUCTIONS` 表，闭环必须发同一句。

修复：`sim_eval_vla.py` 新增 `--sv_task_name`（默认取 `HA_TASK_INSTRUCTION`），
`scripts/ha_isaac_closedloop.sh` 按任务自动带上那一句。

### 10.4 闭环里两个**真 bug**（都在我们的 provider 里，已修）

1. **机器人完全不动、reward 恒 0**
   `sim_eval_vla.py` 用 `ControlConfig(..., use_rl_action_mode=True)`，`RobotController.step()`
   因此**跳过 `env.step(action)`**，物理必须由 provider 自己步进（上游 SONIC provider 就是这么做的）。
   我们的 provider 只算关节目标、没 step 物理 → 目标从没写进仿真。
   → 新增 `SuperMimicActionProvider._step_physics()`：`set_joint_position_target → write_data_to_sim
   → sim.step(render=False) → scene.update(dt=physics_dt)`，循环 `env.cfg.decimation` 次。
   *判据*：修之前 120 帧只有 1847 个像素变化（0.6%，等于静止），修之后机器人会动、会摔。

2. **前视相机画面冻结，VLA 永远看到首帧**
   `sim.step(render=False)` 不触发 RTX 渲染，`camera.data.output["rgb"]` 一直是第一帧。
   → 每个控制步末尾补一次 `env.sim.render()`（与上游 SONIC provider 的 `env.sim.render()` 一致）。
   *判据*：服务端日志 `img_mean/img_std` 从恒定变成逐次变化。

   服务端现在每收到一次请求就打一行，便于自查观测是否在动：
   ```
   [server] infer #4 dt=185.4ms img=(480, 640, 3) img_mean=... img_std=... tok_mean=... state=64 task='Pick up the object...'
   ```

### 10.5 怎么跑（TL;DR）

```bash
# 1) VLA 服务（GPU1；自动取 run 里编号最大的 checkpoint）
RUN=$COMIMIC/runs/groot_sonic_vla_ha_b64_lr1e-5
CKPT=$(ls -d $RUN/checkpoint-* | sort -t- -k2 -n | tail -1)
PYTHONPATH=$COMIMIC/third_party/gr00t_deps:$COMIMIC/third_party/gr00t_deps/_noflash \
python3 scripts/ha_vla_latent64_server.py --ckpt "$CKPT" --port 18080 \
  --device cuda:1 --bf16 --horizon 40 --fm-steps 4

# 2) 单集闭环（Isaac 在 GPU0，出 mp4 + result.json）
bash scripts/ha_isaac_closedloop.sh
TASK=Boxing SEED=1 REPEAT=2 bash scripts/ha_isaac_closedloop.sh

# 3) 批量（消融用）：多 variant × 多任务 × 多 seed，自动出 SUMMARY.md
VARIANTS="base=http://127.0.0.1:18080,abl=http://127.0.0.1:18081" \
TASKS="DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi" \
SEEDS="0 1 2" REPEATS=5 bash scripts/ha_isaac_batch.sh
```

产物：`runs/ha_isaac_eval/<TASK>__seed<k>__repeat<k>__<stamp>/`
（`result.json` / `videos_success|videos_failure/*.mp4`）；
批量汇总：`runs/ha_isaac_batch_<stamp>/{summary.json,SUMMARY.md}`。

### 10.6 当前数值水平（**这是模型问题，不是链路问题**）

| 项 | 现状 |
|---|---|
| 闭环链路 | ✅ 通（300 步、mp4、`result.json`、VLA 每 40 步被调用一次） |
| 成功率 | 0/N —— 机器人 50~60 步（≈1 s）内摔倒 |
| 原因 1 | WBC 只训到 **2500/32000** 迭代（`train_log.csv` 里 sim_rew 已从 0.44 涨到 107，在学，但远不够） |
| 原因 2 | VLA ckpt 是进行中的 `lr1e-5` 续训（交接时说 3500，现在已刷到 6500+，**目录在滚动**） |

所以：**链路可以拿来做消融了，但要先等 WBC 训到一个能站住的 ckpt**，否则所有 variant 的成功率
都是 0，消融没有区分度。最小可用判据：`SM_TOKEN_SOURCE=zeros` 下机器人能撑满 300 步不摔。

### 10.7 仍需处理的缺口

1. **`HOI_grap_cup`（Isaac-Move-ArtVIP-Livingroom-GrapCup-…）跑不了**：本 checkout（shallow，
   HEAD `6847928`）既没注册这个任务、也没有 ArtVIP 客厅资产。要么拉上游分支补
   `tasks/g1_tasks/move_grapcup_*` + 资产，要么先只跑其余 6 个任务。
2. **贴图是关的**：测试 yaml 里 `overrides.wait_for_textures: false`，画面是灰模（见 §5 的
   首帧比对），与训练数据（有贴图）视觉分布不同。正式评测应把这行改成 `true` 再跑。
3. **checkpoint 滚动**：另一容器里 `lr1e-5` 还在训，每 ~8 分钟新增一个 `checkpoint-*`，
   消融必须**显式固定 CKPT 路径**，否则两次跑的不是同一个模型。
4. **reward 是稀疏的**（`compute_reward` 只在锤子 CoM 进入篮筐时给 1.0）。`max_reward` 可当
   "有没有推进"的弱信号，但成功与否只能看 `success`。

### 10.8 六任务出片（2026-10-24 13:00）

用 `scripts/ha_isaac_batch.sh` + `scripts/ha_collect_videos.py` 跑完 6 个任务
（VLA=`groot_sonic_vla_ha_b64_lr1e-5` 当时最新 ckpt，WBC=`bones_seed_token_ha_v2/ckpt_005500.pt`），
视频在 `runs/ha_arena_videos/`，索引 `runs/ha_arena_videos/INDEX.md`：

| 任务 | 结果 | 步数 |
|---|---|---|
| HOI_double_desk | fall | 33 |
| HOI_pp_box | fall | 39 |
| HSI_boxing | fall | 59 |
| HSI_open_door | fall | 41 |
| HSI_sit_sofa | fall | 39 |
| HSI_vision_navi | fall | 40 |

**注意 `failure_reason` 的含义**：`fall` = 摔倒检测触发；`timeout` = 未摔倒但 300 步内没达成。
所以这批视频只有 0.7~1.2 s —— 链路是对的，是 WBC 还站不住（见 §10.6）。

### 10.9 另外两个协议缺口（已补）

1. **`POST /reset`**：上游 LeRobot server 有这个端点，HumanoidArena 的 SONIC provider
   **每集开始会调**，我们原来只实现 `/infer` → `HTTP 404 ... /reset` → 整集直接失败。
   `scripts/ha_vla_latent64_server.py` 现在返回 `{"ok": true}` 并清空手部回灌缓冲。
2. **`SONIC_VLA_ACTION_FORMAT`**：走上游 SONIC provider 必须 `=latent64`（我们返回 64 维
   latent）。不设的话它按 `semantic_v3` 期待 40 维，报
   `Expected VLA action chunk shape [N, 40] ... got (40, 64)`。`ha_isaac_closedloop.sh` 现在默认导出。

### 10.10 脚本清单（本次新增）

| 脚本 | 干什么 |
|---|---|
| `scripts/ha_isaac_closedloop.sh` | 单集闭环：任务别名 → 任务 id/yaml/指令，调官方 `sim_eval_vla.py`，出 mp4+json |
| `scripts/ha_isaac_batch.sh` | 批量：多 variant（各自 URL）× 多任务 × 多 seed × 多 repeat，自动出 `SUMMARY.md` |
| `scripts/ha_collect_videos.py` | 把一次批量产物收成"一任务一个视频 + INDEX.md" |
| `scripts/ha_vla_latent64_server.py` | 加了 `/reset`、每次请求的 `img_mean/img_std/tok_mean` 日志 |

### 10.11 瓶颈定位：是 WBC，不是 VLA（决定性对照，2026-10-24 13:25）

摔了以后必须先分清"VLA 预测的 token 太差"还是"执行器 WBC 站不住"，否则后续消融全是噪声。
做法：加一个只回放数据集真值 token 的服务 `scripts/ha_gt_token_server.py`
（协议和我们的 VLA 服务完全一致，闭环保留不变，只换 `SERVER_URL`）：

```bash
python3 scripts/ha_gt_token_server.py \
  --npz data/ha_raw/HOI_double_desk/tw/<一条>.npz --port 18081     # 用 encoder_latent (T,64)
SERVER_URL=http://127.0.0.1:18081 bash scripts/ha_isaac_closedloop.sh
```

| 输入 token | 结果 |
|---|---|
| 我们 VLA（ckpt-6500） | fall @ 33~64 步 |
| **数据集真值 `encoder_latent`** | **fall @ 51 步** |
| 零 token（对照） | fall @ 67 步 |

结论：**执行器才是瓶颈**。换成数据集里一模一样的 token，机器人照样 1 秒内摔；
反过来，VLA 的 token 分布不算离谱（`std 0.33` vs 数据集 `0.16`，范围同量级）。

于是得到**跑 VLA 消融前的前置门槛**：

> `ha_gt_token_server.py` + WBC 必须能撑满 300 步不摔（或至少明显 > 100 步），
> 否则 VLA 的任何改动都测不出来——因为执行器本身 1 秒就倒了。

当前 WBC 状态：`bones_seed_token_ha_v2` 约 5500/32000 迭代（`sim_r≈0.84`，
`term_frac≈0.00`，训练里看着挺好），但部署到 HA 场景仍站不住。下一步要么把训练跑完，
要么对齐训练/部署的观测（`relative_ori99` 末尾 6 维参考朝向在我们这里是置零的，
训练时来自 motion 的 `root_R[f]`——这是目前最大的一处已知不对齐）。

### 10.12 真正的根因：**关节顺序不一致**（2026-10-24 13:26，已修）

前面所有"机器人 1 秒就摔"的怀疑都指向 WBC 太弱，但部署侧有一个更硬的问题：

| | 顺序 |
|---|---|
| **trackerlab 策略输出**（训练时） | `L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee, L_ankle_pitch, L_ankle_roll, R_*…, waist_yaw/roll/pitch, L_shoulder_*…, R_shoulder_*…`（左腿→右腿→腰→左臂→右臂） |
| **HumanoidArena 资产**（`robot.data.joint_names`） | `L_hip_pitch, R_hip_pitch, waist_yaw, L_hip_roll, R_hip_roll, waist_roll, L_hip_yaw, R_hip_yaw, waist_pitch, L_knee, R_knee, L_shoulder_pitch, R_shoulder_pitch, L_ankle_pitch, R_ankle_pitch, …`（左右交替） |

胶水层原来按"第 j 个 body 关节 ← 策略第 j 维"直接对齐 → **每一条动作都打到错误的关节上**，
所以无论 token 是 VLA 预测的还是数据集真值，机器人都必摔。

修复（`action_provider_supermimic.py`）：
- `_canonical_joint_names()` 给出策略的 29 关节顺序（优先从 `trackerTask.supermimic.robots` 读，
  读不到用内置常量）；
- `self.perm` = 按名字求出 `canonical → asset` 的索引；
- 观测（`joint_pos/joint_vel/default`）和动作目标都在 canonical 顺序下计算，动作再按 `perm` 散射回资产顺序；
- 名字对不上直接 `RuntimeError` 并把两边的名字打出来（不允许静默错位）。

效果（DoubleDesk / seed 0 / 同一 WBC）：

| 配置 | 存活步数 |
|---|---|
| 修之前（VLA token） | 33 |
| 修之前（数据集真值 token） | 51 |
| **修之后（VLA token）** | **92** |
| **修之后（数据集真值 token）** | **165** |

翻 2~3 倍，但还没站住 → 剩下的差距是 WBC 训练量（约 5500/32000）和
`relative_ori99` 末尾 6 维参考朝向（部署时置零）这两件事，见 §10.11。

### 10.13 修复后的六任务结果 + ref-relative 6 维的 A/B

关节顺序修好之后重跑（VLA=`lr1e-5` ckpt6500，WBC=`bones_seed_token_ha_v2/ckpt_005500`）：

| 任务 | 修之前步数 | 修之后步数 |
|---|---|---|
| HOI_double_desk | 33 | **125** |
| HOI_pp_box | 39 | 43 |
| HSI_boxing | 59 | 83 |
| HSI_open_door | 41 | 80 |
| HSI_sit_sofa | 39 | **130** |
| HSI_vision_navi | 40 | **156** |

视频：`runs/ha_arena_videos_fixed/`（`INDEX.md` 里还有每条的结果）。
仍然全是 `fall` —— 但视频里能看到机器人真的在动、在走，而不是原地僵住。

**obs99 末尾 6 维（参考朝向）**：训练时是 `mat_to_6d(Rt @ motion.root_R[f])`，部署时没有
motion 参考，只能取常量。做了 2 seed 的 A/B（DoubleDesk，ckpt7000）：

| 常量 | seed1 | seed2 |
|---|---|---|
| `zeros`（现在默认） | 114 | 124 |
| `identity` = `[1,0,0,0,1,0]`（理论上的单位旋转） | 48 | 47 |

结论：**默认用 zeros**（`SM_REF6D=identity` 可切回去做后续实验）。理论上 identity 才对，
实测反过来，说明这 6 维在策略里的用法和我们假设的不同——这是一个**待查项**，不影响当前链路。

### 10.14 最终出片（2026-10-24 13:42，锁定版本）

用户要求"用当前效果最好的 checkpoint 出各任务视频"，最终锁定（**都是显式固定的路径**）：

| 组件 | 路径 |
|---|---|
| VLA | **`runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（写死；见 `runs/OFFICIAL_CKPT.json`）** |
| WBC | `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_007500.pt` |
| provider | `joint order` 已修；`SM_REF6D=zeros`（默认） |

产物：`runs/ha_arena_videos_final/`（含 `INDEX.md`），批量原始结果
`runs/ha_isaac_batch_20261024_134218/`。

| 任务 | 视频时长 | 步数 | 结果 |
|---|---|---|---|
| HOI_double_desk | 2.97 s | 88 | fall |
| HOI_pp_box | 1.77 s | 52 | fall |
| HSI_boxing | 3.47 s | 103 | fall |
| HSI_open_door | 2.80 s | 83 | fall |
| HSI_sit_sofa | 2.57 s | 76 | fall |
| HSI_vision_navi | **9.17 s** | **274** | fall |

成功率仍为 0。`HSI_vision_navi` 撑到 274/300 步，说明链路本身没问题、
其余任务的差距主要在 WBC（约 7500/32000 迭代）——**这是下一步唯一真正的瓶颈**。

复现这一批（VLA 服务已在 18080）：
```bash
SM_CKPT=$COMIMIC/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_007500.pt \
VARIANTS="best=http://127.0.0.1:18080" \
TASKS="DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi" \
SEEDS="0" REPEATS=1 bash scripts/ha_isaac_batch.sh
python3 scripts/ha_collect_videos.py runs/ha_isaac_batch_<stamp> --out runs/ha_arena_videos_final
```

### 10.15 两条执行器口径的对比（2026-10-24 14:16）

**注意：仓库现在的默认执行器不是 SONIC decoder。** `action_provider/create_action_provider.py`
（12:06 改动）把 `action_source` 默认改成了 `supermimic_wholebody`（我们自己的 WBC），
SONIC 必须 `export HA_ALLOW_SONIC=1` 才生效；`script/eval_scripts/sonic/sim_eval_vla.py`
里 `_normalize_control_routing()` 同样如此。所以 §10.14 那批视频用的是 SuperMimic WBC。

同一套 VLA（`checkpoint-9500`）、同一 seed，两条执行器各跑 6 个任务：

| 任务 | SONIC decoder（官方） | SuperMimic WBC（我们） |
|---|---|---|
| HOI_double_desk | 35 步 | 88 步 |
| HOI_pp_box | 50 步 | 52 步 |
| HSI_boxing | 38 步 | 103 步 |
| HSI_open_door | 36 步 | 83 步 |
| HSI_sit_sofa | 51 步 | 76 步 |
| HSI_vision_navi | 61 步 | **274 步** |

视频：`runs/ha_arena_videos_sonic/`（SONIC）、`runs/ha_arena_videos_final/`（SuperMimic）。

怎么选：

| 目的 | 用哪条 | 怎么开 |
|---|---|---|
| 和 HumanoidArena 官方榜单/别人结果对比 | SONIC decoder（数据集录制时的执行器） | `export HA_ALLOW_SONIC=1` + `SONIC_VLA_ACTION_FORMAT=latent64` |
| 评估"我们自己的 VLA+WBC 系统"、做联合消融 | SuperMimic WBC（默认） | 直接跑，`SM_CKPT` 指 trackerlab ckpt |

两条都摔，但 SuperMimic 明显更耐摔——它是**在 HA 数据上训的**，而 SONIC decoder 吃的是
我们 VLA 预测的 latent（未经 SONIC 编码器的在线对齐）。要正式报"官方口径成功率"，
SONIC 那条路还需要确认 `--sonic_pose_source` / heading-align 等参数是否和官方 runbook 一致，
目前只是"能跑并且不比 WBC 好"。

---

## 11. 关节顺序契约（**锁死**，2026-10-24）

闭环里"关节角顺序"这件事从今天起只有一个定义：`comimic/joint_contract.py`。
任何 provider / 数据脚本 / 部署脚本都不许再自带一份关节名字面量。

### 11.1 两套顺序，各有各的用处

| 名字 | 内容 | 谁在用 |
|---|---|---|
| `STANDARD_29` | 左腿 → 右腿 → 腰 → 左臂 → 右臂 | MuJoCo MJCF 声明顺序、trackerlab `get_robot("g1").joint_names`（WBC 训练/推理）、HA `TWIST2_G1_JOINT_NAMES_29`、`groot_repro/g1_sim.BODY_JOINT_NAMES` |
| `HA_V31_29` | 左右交错（`L_hip_pitch, R_hip_pitch, waist_yaw, L_hip_roll, …`） | HumanoidArena `vla_state` 的 29 维、Isaac 资产 `robot.data.joint_names` 里这 29 个、我们 VLA 的 `state46.joints29` |

> 坑：HumanoidArena 把**交错**那套叫 "canonical"，trackerlab/MuJoCo 把**另一套**叫 canonical。
> 所以本项目禁止用 "canonical" 这个词，统一 `STANDARD_29` / `HA_V31_29`。

### 11.2 锁在哪

| 位置 | 锁法 |
|---|---|
| `joint_contract.py` | `assert_order(names, expected, who=...)` / `match_order(names)`：名字对不上、顺序对不上都**当场抛错**并把 perm 打出来 |
| Isaac `action_provider_supermimic.py` | `asset_order = JC.match_order(asset_body_names)`，再 `perm = JC.permutation(asset_order, "standard")`；启动日志打印 `joint order locked: asset=… policy=standard perm=…` |
| MuJoCo `groot_repro/g1_sim.py` | `BODY_JOINT_NAMES` 直接由 `JC.STANDARD_29` 生成（不再有字面量） |
| 数据 `prepare_ha_sessions.py` / `prepare_ha_tokens_npz.py` | 读 npz 后先 `JC.to_standard(jp_ha)` 再做 FK / 手腕切片；FK 模型顺序不等于 `STANDARD_29` 直接报错；`state_46.joints29` 明确保持 HA 顺序（train == deploy） |
| VLA 服务 `ha_vla_latent64_server.py` | 新增 `--state-order {ha_v31,standard}`（默认 `ha_v31`，即当前 ckpt 训练时的顺序） |
| 环境 `scripts/isaac_env.sh` | `PYTHONPATH` 加 `$COMIMIC`，任何入口（包括裸跑 `sim_main.py`）都能 import 契约 |

### 11.3 回归测试

```bash
python3 -m pytest tests/test_joint_contract.py -q      # 8 passed
```

测试直接比对**三个真源**，谁改了都会红：

1. MuJoCo MJCF（`reproduce_scenes/g1/g1_29dof_with_hand.xml`）的关节声明顺序 == `STANDARD_29`
2. `trackerlab` 的 `get_robot("g1").joint_names` == `STANDARD_29`
3. HumanoidArena `vla_smpl_runtime.py::CANONICAL_G1_JOINT_NAMES_29` == `HA_V31_29`

外加：两套顺序必须真的不同、置换可逆、顺序错位会抛错、provider/`g1_sim` 不许有私有字面量。

### 11.4 新增一个消费者时怎么做

```python
import joint_contract as JC

order = JC.match_order(runtime_names, who="my_provider")   # 自动判断是哪套，否则报错
perm  = JC.permutation(order, "standard")                  # 统一到策略顺序
q_std = q_raw[perm]
```

**不要**写 `joint_names.index(...)`、不要写死 `[0:6]` 这种切片、不要复制名字列表。
