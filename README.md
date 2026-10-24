# Differentiable VLA–Whole-Body Co-Training Framework

> **新增主线（2026-10）**：把 GR00T N1.7 的 Flow Matching action head 与
> CuboiLeo/SuperMimic-Reproduced 的可微 WBC + world model 接成一张计算图，并把
> **顶层 VLM（Cosmos-Reason2-2B）**接在它的上游：
> `RGB + instruction -> Cosmos-Reason2-2B -> FM head -> motion goal adapter -> SuperMimic WBC -> differentiable world model -> task loss`。
> 代码在 `diff_vla_wbc/`（VLM 在 `gr00t_vlm.py`），说明与复现命令见
> [`docs/FRAMEWORK.md`](docs/FRAMEWORK.md)。
>
> 已通过真实 `Gr00tN1d7ActionHead`（tiny DiT）+ 真实 SuperMimic `Policy/WorldModel` +
> 真实 G1 MJCF 的端到端梯度测试；顶层 VLM 用本机真实的 Cosmos-Reason2-2B 权重跑通
> `backbone -> FM head -> tokens -> adapter` 并反传到 head（`scripts/smoke_vlm.py`）。
> **尚未**有 GR00T 的 action head checkpoint、与 motion 对齐的 RGB 训练数据、BONES-SEED
> 全量训练或带物体接触的世界模型。边界写在框架文档第 8 节。

```text
diff_vla_wbc/             可微 VLA-WBC 联合训练框架
third_party/
  SuperMimic-Reproduced/  clone 的 CuboiLeo 实现（BONES-SEED 管线）
scripts/                  数据转换 / 联合训练 / MuJoCo 评测
data/npz/g1_tasks/       271 条真机 G1 录制转成的 npz（git-ignored，约 1.8 GB）
docs/FRAMEWORK.md         架构、张量契约、真实权重接入、限制
tests/                    18 个测试（含真实 GR00T head 反传、VLM 接线）
```

> G1/YCB mesh 资产（约 170 MB）**不进 Git**，保留在本地磁盘；首次运行执行
> `bash scripts/setup_assets.sh` 创建 MJCF 需要的 `g1/meshes` 软链（缺失时按脚本提示从
> Unitree G1 描述目录补齐）。
>
> 顶层 VLM 权重不随仓库分发：`python scripts/link_vlm_weights.py` 会把本地
> Hugging Face 缓存里的 `nvidia/Cosmos-Reason2-2B` 软链成仓库内视图
> （`groot_repro/nvidia/Cosmos-Reason2-2B-weights/`，git-ignored）。

---

# issac_groot — GR00T N1.7 推理复现与闭环控制

这个仓库把我们复现 **GR00T N1.7（SONIC）推理链路** 的完整过程整理了出来，重点回答一个问题：

> 一个 VLA checkpoint，从「给一张 ego_view + 机器人状态」到「机器人在 MuJoCo 里真的动起来」，中间到底要经过什么？

三个目录：

| 目录 | 内容 | 说明 |
|---|---|---|
| `Isaac-GR00T/` | 官方训练/推理框架 | 策略服务端 `Gr00tPolicy`、模型定义、processor |
| `GR00T-WholeBodyControl/` | 官方部署仓库 | SONIC C++ 控制器、MuJoCo 仿真、PICO 遥操作、LeRobot 导出 |
| `groot_repro/` | **我们自己的复现工作** | 关键脚本、排坑记录、结果 |
| `reproduce_scenes/` | **场景与初始位姿** | 部署评测用的 MJCF + 网格 + 初始状态，可原样重建 |

> 只上传了代码与配置。venv、TensorRT、ONNX 权重、数据集、视频都没有进仓库（见文末「没有上传的东西」）。

---

## 一、推理链路（重点）

```
                     ┌──────────────────────────────────────────┐
 ego_view (640x480)  │  Gr00tPolicy.get_action()                │
 robot state (43)  → │    processor → VLM backbone → DiT x4 步   │ → motion_token (40 x 64)
 language prompt     │    decode_action                         │    + 左右手各 7 维
                     └──────────────────────────────────────────┘
                                      │
                                      ▼  50 Hz
                     ┌──────────────────────────────────────────┐
                     │  SONIC decoder (model_decoder.onnx)      │
 10 帧状态历史      → │   994 维观测 → 29 维关节目标             │ → q_target
                     └──────────────────────────────────────────┘
                                      │
                                      ▼
                     tau = kp*(q_target - q) - kd*dq     (500 Hz)
                                      │
                                      ▼
                     MuJoCo mj_step  →  head_camera 渲染  →  回到最上面
```

两边都是 50 Hz：policy 每 0.8 s（40 步）出一次 token 块，decoder 每个控制周期消费一个 token。

### decoder 的观测契约（994 维）

```
token(64)
+ 10 帧 × base_ang_vel(3)      = 30
+ 10 帧 × q(29)                = 290     ← 注意是 q - default_angles
+ 10 帧 × dq(29)               = 290
+ 10 帧 × last_action(29)      = 290
+ 10 帧 × projected_gravity(3) = 30
                                 ─────
                                 994
```

来自 `GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/observation_config_sonic_release.yaml`，
C++ 侧对应 `g1_deploy_onnx_ref.cpp` 的 `GatherHis*` 系列函数。

### 关节目标怎么算

```cpp
// g1_deploy_onnx_ref.cpp:3132
action_value = action[isaaclab_to_mujoco[i]] * g1_action_scale[i];
q_target[i]  = default_angles[i] + action_value;
last_action[i] = action[i];          // 存的是 isaaclab 序
```

`g1_action_scale = 0.25 * effort_limit / stiffness`，kp/kd 由电机 armature 按二阶临界阻尼推出来 —— 全部在
`policy_parameters.hpp` 里。

---

## 二、踩过并写进文档的坑

这些是复现时最容易卡住的地方，`groot_repro/README.md` 里有完整记录：

1. **decoder 历史里的 q 必须减 `default_angles`**
   `body_q[i] = q[mujoco_to_isaaclab[i]] - default_angles[...]`。
   漏掉这一步，decoder 会以为机器人比实际高 0.3 rad，一直过量补偿 → 一秒倒地。
   看起来像「decoder 不做平衡」，其实是观测错了。

2. **部署 MJCF 的关节/执行器顺序不是 MuJoCo 顺序**
   是 `腿、腰、左臂、左手、右臂、右手`（数据集顺序）。
   必须按名字查 `jnt_qposadr` / actuator id，不能按切片写 `ctrl[:29]`。

3. **planner 路径里必须有 `V2`**
   C++ 用路径字符串判断 planner 版本：`planner/target_vel/V2/planner_sonic.onnx`，换个路径就报
   `Unsupported planner version`。

4. **VLM backbone 是 gated 仓库**
   `nvidia/Cosmos-Reason2-2B` 需要授权；但权重其实全在 checkpoint shard 里（493 个张量）。
   用公开的 Qwen3-VL-2B 配置把 `num_hidden_layers` 改成 16 即可，张量名/形状 100% 对齐。

5. **8 GB 显卡放不下 6.29 GiB 模型**
   用 accelerate 的 `device_map` 按层拆分（`groot_repro/gpu_patch.py`）：
   纯 CPU 25–28 s/次 → GPU 分片 **2.8–4 s/次**。

6. **CycloneDDS 版本不一致**
   C++ 侧 vendored `libddsc.so` 是 0.10.2，Python 侧 pip `cyclonedds` 是 11.0.1；
   回环不支持组播，换网卡又在 XTypes 类型对象处崩。这是官方 C++ 控制器和 Python 仿真互联的最后一个坎。

---

## 三、怎么跑

### 纯离线复现（不需要仿真、不需要 PICO）

```bash
# 1) 把 checkpoint 变成可离线加载的视图（软链权重，只改两个 json 里的 model_name）
python groot_repro/prepare_ckpt.py \
    --source <checkpoint-50000> --dest ckpt_sirius_50k

# 2) 对录制数据做推理，和真值对比
python groot_repro/repro_infer.py \
    --ckpt ckpt_sirius_50k --frames 0 800 1600 2400 --samples 3

# 3) 出图
python groot_repro/plot_result.py --frame 800
```

### decorer 闭环 + MuJoCo 渲染

```bash
# 只回放录制 token（快速验证闭环）
python groot_repro/render_recorded_physics.py

# policy 在环路里（GPU 分片）
python groot_repro/render_policy_physics.py --chunks 12 \
    --scene <scene_43dof_two_table.xml> --prompt "Pick up the carton ..." \
    --device cuda:0 --gpu-gib 2.4GiB --out out/carton.mp4
```

### 场景复现

```bash
cd reproduce_scenes
python verify_scenes.py --render /tmp/preview   # 8 个场景全部加载 + 渲染
```

`reproduce_scenes/README.md` 里有场景↔任务对照表、初始位姿格式和 43 维关节顺序的坑。

### 编译 x86 版 C++ 控制器

```bash
bash groot_repro/build_deploy_x86.sh
```
（脚本里写清了每一处依赖复用了本机什么，没下 7 GB）

---

## 四、目录说明（`groot_repro/`）

| 文件 | 作用 |
|---|---|
| `README.md` | **完整复盘**：SONIC_DATA 盘点、复现结果、四任务视频、踩坑 |
| `g1_sim.py` | PD 增益 / action scale / 关节映射 / 重力投影，与部署代码逐项对齐 |
| `sonic_loop.py` | SONIC decoder 的 Python 版控制环（994 维观测打包） |
| `repro_infer.py` | 离线推理 + 与录制真值对拍 |
| `render_policy_physics.py` | **policy + 物理闭环 + MuJoCo 渲染 → mp4** |
| `render_recorded_physics.py` | 录制 token 的闭环回放（验证用） |
| `validate_decoder.py` | 录制 token 灌进 decoder，对拍 `action.wbc`（MAE 2.5°/corr 0.956） |
| `prepare_ckpt.py` / `meta_patch.py` / `gpu_patch.py` | 离线加载、省内存、GPU 分片 |
| `build_deploy_x86.sh` | x86 编译 SONIC C++ 控制器 |
| `start_stack.sh` | 一键起 sim + 控制器 |
| `scenes/` | 自建/合并的 MJCF 场景 |

---

## 五、没有上传的东西

| 内容 | 大小 | 怎么获取 |
|---|---|---|
| TensorRT 10.13.3.9 | 4.4 GB | `pip install tensorrt-cu12==10.13.3.9` |
| `model_decoder.onnx` / `model_encoder.onnx` | 41 / 50 MB | HF `nvidia/GEAR-SONIC`（`download_from_hf.py`） |
| `planner_sonic.onnx` | 774 MB | 同上（注意要放在路径含 `V2` 的目录下） |
| checkpoint（`checkpoint-50000` 等） | 8.9 GB / 个 | 内部数据，见 `groot_repro/README.md` |
| 录制数据集（LeRobot） | 数十 GB | 内部数据 |
| venv / conda 环境 | — | `install_scripts/*.sh` |
| 渲染视频（mp4） | 数百 MB | 用上面的脚本重新生成 |

---

## 六、已验证的结论

- 权重完整性：`sha256sum -c` 全过；载入 `missing/unexpected/mismatched` 全 0
- decoder 对拍：录制 token → decoder，与录制 `action.wbc` 比 **MAE 2.5°、corr 0.956**
- 闭环：机器人能站立、行走（12 s 位移 0.27 m），policy 在环路里 9.6 s 全程不摔
- 任务成功率：**0/4**（apple / carton / red cup / handover，各 60 s）
  —— 和部署自己的评测（carton 6/50、can 8/33、handover 0/89）一致，属于低成功率任务

详见 `groot_repro/README.md`。
