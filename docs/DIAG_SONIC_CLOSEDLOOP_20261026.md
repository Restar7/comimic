# SONIC 闭环“怎么都站不住”——进程级排查报告（2026-10-26）

> 触发：用户质疑“所有运行都摔倒，不可能是模型问题，先查接口”。
> 本文只写**有数据支撑**的结论；没跑成的部分明确标注。
>
> 一句话结论：**视觉、频率、关节顺序、decoder 输入布局、动作换算全部对得上；
> 但 VLA 的状态向量、token 反归一化、`SONIC_HA_ALIGN` 执行器口径、decoder 历史初值
> 这四处存在硬性接口错位，任何一处单独都足以让机器人从“走不好”变成“必摔”。**

---

## 1. 今天实际跑成/没跑成什么

| 项 | 状态 |
|---|---|
| Isaac 闭环（GT token，300 步，逐帧转储） | ❌ **本节点 Isaac 起不来**：新 pod（Ubuntu 24.04）缺 NVIDIA Vulkan ICD；补 ICD + libXt6/libGLU 后仍然 `PhysXFoundation: No devices found by renderer::IGpuFoundation` / `CUDA libs are present, but no suitable CUDA GPU was found`，RTX viewport 建了 3.5 万次空转。torch CUDA 正常（2×4090 可见），**是 Isaac 的 Vulkan/RTX 枚举问题，不是显存** |
| MuJoCo 替代闭环（官方 decoder + 官方 PD 常数，CPU） | ✅ 跑通并逐帧转储，但**该模型 plant 与被测 Isaac 资产不同**：连“录制里的 29 维 joint target 原样回放”都会在 43 步内倒（录制里同一串 target 站了 2080 帧）。**故它只能做相对对照，不能用来判定 Isaac 摔倒原因** |
| 离线对拍（录制 npz ↔ 线上代码） | ✅ 全部做完，结论见 §2/§3 |

产物：
- 逐帧转储/分析：`scripts/dbg_mujoco_sonic_loop.py`、`scripts/dbg_analyze_sonic_dump.py`、
  `runs/dbg_mujoco/*.npz`、`runs/dbg_sonic_gt/frames/*.png`
- provider 逐帧转储（Isaac 用，env `SONIC_DEBUG_DUMP=<dir>`）已内置，等 Isaac 可用即可用

---

## 2. 已经验证“对齐”的部分（这些不是原因）

1. **decoder 输入 994 维布局**（`latent64 + 10×(ang_vel + joint_pos_delta + joint_vel + last_action + gravity)`）：
   用录制 `data/ha_raw/HOI_double_desk/tw/*.npz` 反解，provider 的拼接顺序逐块 **maxdiff = 0.0**；
   唯一差异是 `last_action` 块相差 1 帧（录制 `obs[t]` 用的是 `a[t-1]`，provider 同语义，正确）。
2. **动作换算**：`target = raw * G1_ACTION_SCALE_ISAACLAB + SONIC_DEFAULT_POS` 与录制
   `decoder_target_action` 逐关节拟合 **残差 0（scale 0.3507/0.5475/0.4386/0.0745，
   偏置 −0.312/0.669/−0.363/0.6）**；`final_body_action_29dof == decoder_target_action`（无裁剪/无延迟）。
3. **关节顺序**：`SONIC_ISAACLAB_JOINT_ORDER` 与 `joint_contract.HA_V31_29` **完全相同**；
   MuJoCo/STANDARD 与该序的映射由 `policy_parameters.hpp` 的 perm 表给出，可用。
4. **观测约定**：`ang_vel` = body 系（录制 `robot_root_ang_vel_local` 逐帧 maxdiff 0.0）；
   `gravity` = `R^T·[0,0,−1]`（maxdiff 1e-7）；`joint_pos` 是 **相对 default 的 delta**（已验证）。
5. **控制频率**：录制 `meta_control_dt=0.02 / decimation=4 / physics 0.005`，闭环
   `control_hz=50`、`decimation=4`、`sim.dt=0.005` —— **一致**。
6. **视觉**：录制首帧 ego 画面 vs 闭环首帧前视画面（`runs/dbg_sonic_gt/frames/`）是**同一机位、
   同一灰模场景、同一分辨率 480×640**（训练数据本身就是灰模），不是视觉域偏移问题。

---

## 3. 查出来的硬 bug（按影响排序）

### B1 VLA 的 46 维 state 分块错位（两块互换，影响两条线所有运行）

训练数据 `data/ha_sessions` 的 `observation.state`(43) + gravity(3) 布局（`meta/modality.json`）：

```
left_leg 0:6 | right_leg 6:12 | waist 12:15 | left_arm 15:22 |
left_hand 22:29 | right_arm 29:36 | right_hand 36:43 | gravity 43:46
```

用 parquet 逐帧对拍确认：`state = [jp_ha[0:22], left_hand, jp_ha[22:29], right_hand]`
（HA_V31 关节序），51 个 episode 全部 **maxdiff = 0.0**。

而 `scripts/ha_vla_latent64_server.py` 的 `state64_to_state46()` 原来拼的是
`[jp(29), left_hand, right_hand, gravity]` ⇒ **槽位 22:29 与 29:36 互换**：

| 组 | 训练/部署一致？ | 实测最大偏差 |
|---|---|---|
| left_leg / right_leg / waist / left_arm / right_hand / gravity | ✅ | 0.0 |
| **right_arm (22:29)** | ❌ 部署喂的是**左手手指** | **1.74 rad** |
| **left_hand (29:36)** | ❌ 部署喂的是**右臂关节** | **1.74 rad** |

⇒ VLA 有 **14/46 维**(30%) 的 proprio 是错的 → 预测 token 系统性跑偏。
**已修**：`--state-layout train_v1`（默认）按训练布局拼，`legacy` 保留旧行为。

### B1b `projected_gravity` 的 rot6d 解码方式是错的（另 3/46 维）

`ha_vla_latent64_server.py` 原来假设“两行/两列基向量”打包，但 HA 的
`quat_to_rot6d_wxyz()` 是 `rot[:, :2].reshape(...)`，即 **前两列 row-major**：
`[R00, R01, R10, R11, R20, R21]`。单位姿态在录制里就是 `[1,0,0,1,0,0]`，
按“两行”解会退化（`gravity = [0, nan, nan]`）。

用录制 `vla_state_root_rot6d` 逐帧解重力，与 `decoder_gravity_dir_hist`
（= 机器人真实朝向的 `R^T·[0,0,−1]`）对拍：

| 解码方式 | max\|Δ\| |
|---|---|---|
| 前两列 row-major（本次修正） | **1.2e−7** |
| 旧“两行/两列”（现状） | tilt 帧上偏 **0.5 rad**，单位姿态直接 NaN/0 |

`projected_gravity` 是 46 维里唯一直接告诉模型“我歪没歪”的量，喂错等于把平衡感知拿掉。
**已修**：`rot6d_to_matrix(layout="ha_cols")`（`rows` 保留做 A/B）。

### B1+B1b 合计：46 维里 **17 维（37%）** 是错的

修好后端到端验证（同一集、同一帧，与训练 parquet 的 43+gravity 对拍）：

```
state46(train_v1, 新 rot6d) vs 训练布局 : max|Δ| = 0.0        ✅
state46(legacy,   旧 rot6d) vs 训练布局 : max|Δ| = 1.74 rad   ❌
```

### B2 10-25 的 L1 批次 token 没做反归一化（codec）

`runs/compare_two_routes/L1/server.log` 里 661 次推理 `tok_mean` 均值 **+0.088**
（录制 token 的 per-frame 均值 **−0.016**，std 0.163）——差的正是“min-max 归一化空间”的偏置。
10-26 加了 `--decode codec` 后 `logs/base8000_ha_align_server.log` 里
`tok_mean≈0.00, tok_absmean=0.108~0.141`，与录制（absmean 0.127）同量纲。

⇒ **10-25 那批 1/70 的数字不能代表官方口径**，它跑的是偏置 token。

### B3 `SONIC_HA_ALIGN=1` 把 decoder 的语义改掉了（10-26 那批 0/30 的直接原因）

`SONIC_HA_ALIGN=1` 同时改了 3 处、而 decoder 是按官方 deploy 语义训出来的：

| 项 | 官方/录制 | HA_ALIGN | 后果 |
|---|---|---|---|
| `default_q`（decoder 观测里的 delta 基准、动作的偏置） | hip −0.312 / knee **0.669** / ankle −0.363 / elbow 0.6 | −0.2 / **0.42** / −0.23 / 0.87 | knee **−14.3°**、elbow **+15.5°**、hip +6.4°、ankle +7.6° 的系统性姿态偏差（观测与指令各偏一次） |
| `action_scale` | 0.3507/0.5475/0.4386/0.0745 | mean 0.3789 max 0.7500 | 与官方逐关节比值 **0.38× ~ 10.1×**（腕部被放大 10 倍） |
| kp/kd | `policy_parameters.hpp` | HA 资产表 | 换了执行器 |

⇒ 这条臂上 decoder 既“看错自己的姿态”又“下错幅度的指令”，成功率 0 是必然的，
**不能当作“官方微调 + SONIC 控制器”的成绩**。

### B4 decoder 历史缓冲初值用错了量纲

`_setup_buffers()` 原来：

| 缓冲 | 原值 | 应该 | 影响 |
|---|---|---|---|
| `last_action_hist` | `_sonic_default_np`（关节角 −0.312/0.669…） | **0**（raw action 空间） | 前 10 拍 decoder 以为刚下过 0.67 rad 的动作 |
| `grav_dir_hist` | `0` | **[0,0,−1]** | 前 10 拍等于告诉 decoder“姿态未知/自由落体” |

10 拍 = 0.2 s，足以打出一个大修正。替代 plant 上的 A/B（同 token、同 plant，只换初值）：
**首摔 21 步（provider 初值） vs 68 步（按录制初值）**。
**已修**：默认 `SONIC_HIST_INIT=faithful`，`legacy` 可回退 A/B。

### B5（次要，但要记账）PD 增益被手改过

`SONIC_PD_KP` 里有注释掉的官方值和改后的值：

| 关节 | 官方（注释行） | 现行 | 偏差 |
|---|---|---|---|
| ankle_pitch | 28.445627 | **48.445627** | +70% |
| shoulder_roll | 14.222814 | **34.222814** | +141% |
| shoulder_yaw | 14.222814 | 24.222814 | +70% |
| elbow | 14.222814 | 19.222814 | +35% |
| wrist pitch/yaw | 14.222814 | 16.77876 | +18% |

录制动作是在官方 kp 下产生的；改 kp 会改变闭环响应。是否保留应作为**单独一臂**显式记录，
不能混在“官方口径”里。

---

## 4. 已改的代码（都可回退）

| 文件 | 改动 | 回退 |
|---|---|---|
| `scripts/ha_vla_latent64_server.py` | `state64_to_state46()` 增加 `layout`；默认 `train_v1`（按训练布局） | `--state-layout legacy` |
| 同上 | `rot6d_to_matrix()/gravity_from_rot6d()` 改为 HA 的“前两列 row-major”解码（原来会把重力解成 0/NaN） | `layout="rows"` |
| `third_party/HumanoidArena/.../action_provider_sonic.py` | `_grav_dir_hist=[0,0,−1]`、`_last_action_hist=0` | `SONIC_HIST_INIT=legacy` |
| 同上 | 新增 `SONIC_DEBUG_DUMP=<dir>` 逐帧转储（latent/obs994/raw/target/q/qd/root/…） | 不设即关闭 |
| `scripts/dbg_mujoco_sonic_loop.py` | MuJoCo 闭环台架（4 个实验臂 + 三种历史初值） | — |
| `scripts/dbg_analyze_sonic_dump.py` | 转储 ↔ 录制逐块对拍 | — |

契约校验仍通过：`python3 scripts/pin_ha_closedloop.py --check` ✅

---

## 5. Isaac 可用后必须补的实验（一次 8 分钟，能一锤定音）

同一任务（DoubleDesk, seed 0, 300 步）、同一 pin 契约，四臂：

| 臂 | 执行器口径 | token | 期望 |
|---|---|---|---|
| A | `SONIC_HA_ALIGN=0`（官方） | GT（`ha_gt_token_server.py --horizon 20`） | 能站/能按录制动 → 证明链路通 |
| B | `SONIC_HA_ALIGN=0` | 修好后的 VLA | 与 A 的差距 = 纯 VLA 差距 |
| C | `SONIC_HA_ALIGN=0` | VLA + `--state-layout legacy` | 与 B 的差距 = B1 的代价（应显著更差） |
| D | `SONIC_HA_ALIGN=1` | 同 B | 验证 B3 的代价（应最先摔） |

命令骨架（VLA 服务需先起，另卡）：

```bash
# 臂 A
python3 scripts/ha_gt_token_server.py --npz <同任务一条录制>.npz --key encoder_latent --horizon 20 --port 18081
HA_ALLOW_SONIC=1 SONIC_HA_ALIGN=0 SERVER_URL=http://127.0.0.1:18081 \
SONIC_DEBUG_DUMP=$PWD/runs/dbg_sonic_gt/dump TASK=DoubleDesk SEED=0 MAX_STEPS=300 \
  bash scripts/ha_isaac_closedloop.sh
# 分析
python3 scripts/dbg_analyze_sonic_dump.py --dump runs/dbg_sonic_gt/dump --npz <同一条录制>.npz
```

判据：
- A 站住 ⇒ 执行器链路无责，锅在 VLA 输入（B1）与 token（B2）；
- A 也摔 ⇒ 用转储比对 `obs994` 各块与录制、`|target−q|` 跟踪误差、`q_target` 是否越界，
  再看 `SONIC_PD_KP`（B5）与资产 effort/armature 是否与 deploy 一致。


---

## 7. 修完重跑（2026-10-27，同一节点 Isaac 已可用）

> 用户口径：**B3（`SONIC_HA_ALIGN=1`）和 B5（PD 增益手改）是刻意改动，保留不动**；
> 只修 B1/B1b/B4 三个接口 bug。以下结果都在 `SONIC_HA_ALIGN=1` 下跑（即项目现行口径）。

VLA 服务：`runs/ckpt_vla_ha_8000_local`（= checkpoint-8000，`--decode codec --fm-steps 4 --horizon 40`），
Isaac GPU0 / VLA GPU1，逐帧转储 `SONIC_DEBUG_DUMP`。

| 臂（DoubleDesk） | 修前（10-26 legacy） | 修后（10-27） | 修后 z min | 修后最大倾角 | 修后位移 |
|---|---|---|---|---|---|
| seed 0 | timeout 300 | **timeout 300（不倒）** | 0.731 | 12.4° | 1.06 m |
| seed 1 | **fall @269** | **timeout 300（不倒）** | 0.740 | 9.4° | 1.72 m |
| seed 2 | **fall @181** | fall @198 | 0.21 | 87° | 2.13 m |
| seed 7 | **fall @223** | **timeout 300（不倒）** | 0.749 | 7.8° | 1.45 m |
| seed 8 | **fall @269** | fall @125 | 0.18 | 78° | 1.19 m |
| 官方口径 seed 0（对照） | — | timeout 300（不倒） | 0.775 | **4.9°** | 0.20 m |

即：**原来会摔的 4 个 seed 里，2 个（1、7）现在跑满 300 步不倒**；
2 个（2、8）仍然摔，但摔倒位置从 181/269 变成 198/125，且都发生在**已正常行走 1.2~2.1 m
（步速 1.3~1.6 m/s）之后**——已经不是原来“2~6 s 必摔”的形态。

录像（前视逐帧）：
`runs/dbg_fix/DoubleDesk__seed{0,1,2,7,8}__*/videos_{success,failure}/*.mp4`、
`runs/dbg_fix_official/...`；抽帧对比 `runs/dbg_fix/frames/`
（`new_s1_f200.png` = 修后 seed1 直立行走；`old_s1_fall_f255.png` = 修前同 seed 摔倒视角）。

**还没解释的两处**（诚实地留着）：
1. seed 2 / 8 的触发点没定死。排查过“latent 断点”这个假设：**所有** run（含不倒的）
   都有 1.2~1.9 的单步 latent 跳变，所以断点不是判别量；两例的前兆都是 tilt 在约 12 步内
   从 25° 涨到 55°，同时踝跟踪误差 0.3~0.7 rad（踝在使劲但跟不上），伴随 1.3~1.6 m/s 的行走速度。
2. `HA_ALIGN=1` 下腕部目标全天顶在 ±11 rad（见下表），所以**抓取类动作不可能成立**；
   但这条不影响站/走，属于 B3 口径的已知代价。

**结论**：接口 bug 修掉后，机器人从“几乎必摔”变成“能站、能走、部分 seed 跑满 300 步”，
画面里能看到它直立走向桌子停在锤子前（`frames/new_s1_f200.png`）。

**但任务仍然 0 成功（`max_reward=0`），原因可量化**：HA_ALIGN 的 `action_scale` 是刻意的，
但它把**腕部**放大到官方的 10 倍，实测腕关节目标：

| 关节 | 目标 \|mean\| | 目标 \|max\| | 越限比例 | 跟踪误差 |
|---|---|---|---|---|
| L/R wrist_pitch | 8.66 / 9.67 | 9.87 / 11.37 | **99% / 97%** | 7.1 / 8.1 rad |
| L/R wrist_yaw | 9.75 / 8.57 | 11.40 / 10.22 | **99% / 98%** | 8.2 / 7.0 rad |
| knee / ankle / elbow | 0.32 / 0.26 / 1.19 | ≤1.59 | 0% | 0.05~0.41 rad |

即：**腿和臂都在正常跟踪 → 站得住、走得动；腕子全天顶在限位上 → 抓不了**。
这是 B3 这一口径选择的直接代价，不是 bug。

口径对照（同 seed 同 VLA）也印证：`HA_ALIGN=0` 时跟踪误差 0.086 rad（腕目标不再爆），
机器人几乎原地（位移 0.20 m）；`HA_ALIGN=1` 时它走得更远（1.06 m）但腕子饱和。
两条臂的预测 token 统计都在录制分布内（\|lat\| 0.13 / 0.09 vs 录制 0.127，
与录制最近邻距离中位数 0.77 / 0.33），说明 VLA 出的是“像样但不对任务”的 latent——
下一步要提成功率，得在 VLA/数据侧，而不是执行器接口。

---

## 6. 环境备忘（别再踩）

- 本 pod：Ubuntu 24.04、2×4090、driver 570.124.06、`/dev/nvidia1` 缺失、**无 NVIDIA Vulkan ICD**。
  已补 `/usr/share/vulkan/icd.d/nvidia_icd.json`、`libxt6`、`libglu1-mesa`，Isaac 仍枚举不到 GPU。
  ⇒ **Isaac 闭环要回老节点跑**（老节点：Ubuntu 22.04、10 核配额、Isaac 正常）。
- MuJoCo 场景 `GR00T-WholeBodyControl/gear_sonic/.../scene_43dof.xml` 的执行器/关节里**手部插在中间**
  （body 执行器 = `range(0,22)+range(29,36)`），用 `qpos[7:36]` / `ctrl[:29]` 会把左手指当右臂——
  这与录制 `observation.state` 的 `[jp[0:22], hl, jp[22:29], hr]` 布局是同一件事。

---

## 8. 录像链路修复（2026-10-27，用户反馈“太卡、10 s 就停、不清晰”）

| 症状 | 原因 | 修法 |
|---|---|---|
| 10 s 就自动结束 | `MAX_STEPS=300`（50 Hz → 6 s 仿真；按 30 fps 编码正好 10 s 视频） | 探索上限改 `MAX_STEPS=6000`（= 2 min 仿真/视频） |
| 画面卡、像慢动作 | ① 相机数据只有 `sim.render()` 后才刷新，而 provider 只在**换 VLA chunk**时渲染（每 20 步）→ 同一帧重复 20 次；② 录制 50 Hz 却按 30 fps 编码 → 播放速度 1.67× 慢 | ① 录制时**每步** `env.sim.render()`；② `--video_fps 50`（与 50 Hz 控制同帧率） |
| 糊 | `SimpleVideoRecorder` 用 `mp4v`（码率极低） | 收尾时用 imageio-ffmpeg 的 **libx264 CRF 18 / yuv420p** 重编码 |
| 分辨率低 | 前视相机写死 640×480 | 改为 `HA_FRONT_CAM_W/HA_FRONT_CAM_H` 环境变量覆盖（默认不变），本次用 **960×720** |

涉及文件：`script/eval_scripts/sonic/sim_eval_vla.py`（每步渲染 + h264 重编码）、
`tasks/common_config/camera_configs.py`（分辨率可覆盖）。**控制口径未改**；
因为评测脚本哈希变了，按 `docs/HA_CLOSEDLOOP_PIN.md` 的流程 `pin_ha_closedloop.py --write`
重新固化，`git diff configs/ha_closedloop_pin.json` 只有 `created_utc` 与 `script_sha256` 两行。

### 8.1 交付：Boxing 2 分钟录像（2026-10-27）

```
runs/dbg_boxing/Boxing__seed0__repeat0__20261027_113956/
  videos_failure/vla_boxing_2min__seed_0__repeat_0__episode_0__timeout.mp4
  result.json          # episode_steps=6000, failure_reason=timeout, max_reward=0
  ../dump/steps.jsonl  # 6000 帧逐帧数据
```

| 指标 | 值 |
|---|---|
| 时长 / 帧数 / fps | **120.0 s** / 6001 帧 / **50 fps** |
| 分辨率 | **960×720**（原来是 640×480） |
| 编码 | h264 CRF 18 / yuv420p，2.2 Mbps（原来 mp4v，~0.5 Mbps） |
| 文件 | 33.6 MB |
| 探索上限 | 6000 步 = 2 min（50 Hz） |
| 实际摔倒 | **step 4296（85.9 s）**，`hard_tilt`；`HA_FALL_TERMINATE=0` 所以继续录到 120 s |
| 任务 | Boxing seed 0，`max_reward=0`（没打中袋） |

wall clock ≈ 70 min（每步 ~0.70 s，主要是每步 960×720 渲染；640×480 时是 0.32 s/步）。

**注意**：同一个 seed 连跑两次，摔倒时刻差很多（第一次 9.7 s，这次 85.9 s）——VLA 的
flow-matching 采样是随机的，闭环轨迹不可复现；要比摔倒率必须多 seed。
