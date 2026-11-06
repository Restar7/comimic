# 回放关节序错误：定位、修正与实测（2026-11-03 晚）

对象：`scripts/isaac_a_collect_batch.py` 的"回放录制目标"诊断路径。
结论：**关节序确实是错的，已改并验证**；但改完之后回放**仍会在约 1.4 s 倒下**，
而这次倒下的原因**不是关节序**（见 §4 的三组对照）。

---

## 1. 事实：仿真里的关节序 = SONIC/canonical 序

从实跑日志直接打印出来的 `robot.joint_names`（HA 场景、
`g1_29dof_with_dex3_rev_1_0.usd`，29 个身体关节）：

```
[ 0] left_hip_pitch      [ 1] right_hip_pitch     [ 2] waist_yaw
[ 3] left_hip_roll       [ 4] right_hip_roll      [ 5] waist_roll
[ 6] left_hip_yaw        [ 7] right_hip_yaw       [ 8] waist_pitch
[ 9] left_knee          [10] right_knee          [11] left_shoulder_pitch
[12] right_shoulder_pitch[13] left_ankle_pitch   [14] right_ankle_pitch
[15] left_shoulder_roll [16] right_shoulder_roll [17] left_ankle_roll
[18] right_ankle_roll   [19] left_shoulder_yaw   [20] right_shoulder_yaw
[21] left_elbow         [22] right_elbow         [23] left_wrist_roll
[24] right_wrist_roll   [25] left_wrist_pitch    [26] right_wrist_pitch
[27] left_wrist_yaw     [28] right_wrist_yaw
```

**逐项等于 `SONIC_ISAACLAB_JOINT_ORDER`**，也等于 `vla_smpl_runtime.CANONICAL_G1_JOINT_NAMES_29`。
所以 `idx_s = [isaac_names.index(n) for n in SONIC_ISAACLAB_JOINT_ORDER]` 是 **恒等映射**，
而 `idx_t = [isaac_names.index(j) for j in ckpt_spec_joint_names]` 是一个**置换**。

## 2. 事实：录制 npz 里的 q / action 也是 canonical 序

`data/ha_raw/...` 的 `robot_qpos_before_decimation`、`final_body_action_29dof`
（= `decoder_target_action`）都是 canonical 序。四项独立验证：

| # | 验证 | 结果 |
|---|---|---|
| 1 | `decoder_obs` 的 q 块末帧 − `robot_qpos` | 时间上恒为常数（std **5e-6**）⇒ 同序 |
| 2 | 该常数逐关节 | = `−SONIC_DEFAULT_POS`；非零位恰是 0/1、9/10、11/12、13/14、**15/16（左右反号）**、21/22 |
| 3 | `final_body_action_29dof` vs `SONIC_DEFAULT_POS + decoder_raw_action*G1_ACTION_SCALE_ISAACLAB` | **max|Δ| = 0.0** |
| 4 | 帧 0 姿态 | canonical 读 = 正常站姿；standard 读会出现"左膝 0、右膝 0.4、左肘 0、左腕偏航 1.2" |

而 `runs/isaac_motion/ha_cmd_exec/packed.pt` 的 q 是这些值的**原样搬运**
（与 npz 逐帧 max|Δ| = 0.0005，即 fp16 量化），但下游按 ckpt 的 `joint_names`（standard 序）解读。

## 3. 修正

`scripts/isaac_a_collect_batch.py`：

1. 新增 `_rtgt_ids`：回放时 = `idx_s`（默认），可用 `SM_REPLAY_IDS=ckpt` 回到旧行为做 A/B；
2. 回放写目标、写初始关节状态、初值读回校验，**全部统一用 `_rtgt_ids`**；
3. 回放初值改为**取被回放 npz 的第 0 帧**（之前取的是库里随机 clip 的初值——
   实测拿 `HSI_open_door` 的初值去回放 `double_desk` 的目标）；
4. 目标键按官方优先级取：`final_body_action_29dof` → `decoder_target_action` → `vla_action_joint_pos_29`；
5. 每步打印 `关节 mae/max`、`root drift`（已扣掉 env 原点常数）、`z_err`、`up`；
6. 诊断开关：`SM_HOLD_Q=1`（保持第 0 帧静态）、`SM_HOLD_Q=2`（保持 env 默认站姿）、
   `SM_REPLAY_KEY=<npz 键名>`、`SM_DUMP_JOINTS=1`。

## 4. 实测（1 env，DoubleDesk，clip `...ep1776268633456387`，600 步）

### 4.1 修正前后（同一数据、同一场景）

| step | 修前 关节 mae | 修后 关节 mae | 修后 root drift |
|---|---|---|---|
| 0 | 0.264 rad | **0.031 rad** | **0.0000 m** |
| 40 | — | 0.056 rad | 0.117 m |
| 80 | — | 0.061 rad（人已经在倒） | 0.930 m |

修后第 0 步比**录制自身**的跟踪误差还小（录制 mean 0.146 rad），说明序和数据都对上了。

### 4.2 但 1.4 s 之后仍然倒（倒的过程中关节照样跟得住）

| step | z | up | 说明 |
|---|---|---|---|
| 0 | 0.797 | 0.999 | 与录制 0.800 对齐 |
| 40 | 0.774 | 0.996 | 漂 0.117 m |
| 70 | 0.620 | 0.756 | 开始倾 |
| 85 | 0.113 | 0.111 | 已倒 |

### 4.3 三组对照：倒下的原因**不是关节序、也不是目标数据**

| 对照 | 做法 | 结果 |
|---|---|---|
| A | 回放**录制到的实际 q**（`robot_qpos_before_decimation`）而不是 decoder 目标 | 关节 mae **0.026**（几乎复现），仍然 step 60–75 倒 |
| B | `SM_HOLD_Q=1`：目标恒定 = 第 0 帧，**完全不动** | 仍然 ~1.5 s 倒 |
| C | `SM_HOLD_Q=2`：目标恒定 = env 默认站姿 | 仍然 ~1.2 s 倒 |

另外还排除了：HA 场景 vs 裸地面（两者都倒）、机器人 USD（`_m2` 与非 m2 逐位相同的曲线）、
PD 增益（读回 kp mean 71.4 / kd 3.60，与 HA spec 一致）。

**⇒ 机器人站在那儿不动也会倒。** 这是"纯位置 PD + 自由浮动基"的本征不稳定：
本体的平衡是**不稳定平衡**，PD 只有关节反馈、没有平衡反馈，任何微小扰动都会指数增长。
录制那 41 s 能站住，是因为录制时 decoder 是**状态反馈控制器**
（`decoder_obs` 含 q/qd/重力方向/角速度），而 `direct_replay` 把反馈丢掉了，只喂录制的绝对目标。

## 5. 结论与建议

1. **关节序这条已经修好并验证**：第 0 步关节 0.031 rad、root 0.000 m，且逐关节无系统性偏置
   （最大 0.10 rad，全是重力下垂量级）。之前"回放怎么都对不上"里，序的问题是真的。
2. **"回放绝对不会倒"这个前提不成立**：开环喂关节目标时，人形是开环不稳定系统；
   官方 `direct_replay` 是同一套开环方案，我们早先在 `sim_eval_vla` 路径上测到的
   "50 步 6 cm、99 步摔"与这里的量级一致。
3. **不会倒的是闭环**：`--executor wbc`（WBC 在环）在同一个场景里能站 400+ 步（up 0.84）。
   要"物理真实且不摔"的 A'，只能用闭环 + 每 20 步重置到参考。
4. 若要拿 100% 官方基线，仍需把 `sim_main.py --input_source replay` 跑通
   （它依赖 redis + DDS + image server 服务拓扑，headless 独立跑会停在 provider 创建前）。

## 6. 复现命令

```bash
C=$PWD
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PROJECT_ROOT=$C/third_party/HumanoidArena/isaaclab_twist2_g1
export PYTHONPATH="$C/third_party/unitree_sdk2_python:$PROJECT_ROOT:$C:\
$C/trackerlab-supermimic/source/trackerTask:$C/trackerlab-supermimic/source/trackerLab"
export SUPERMIMIC_HA_ALIGN=1
NPZ=$C/data/ha_raw/HOI_double_desk/tw/Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody_sonic_1776268633456387.npz

SM_REPLAY_IDS=sonic SM_REPLAY_KEY=final_body_action_29dof \
$C/third_party/isaacsim_env/bin/python -u scripts/isaac_a_collect_batch.py \
  --enable_cameras --headless --wbc-ckpt runs/S1_handover/exp_WBC_exec_haonly_best.pt \
  --packed runs/isaac_motion/ha_cmd_exec/packed.pt --tokens-dir runs/exp_tokens --token-source npy \
  --num-envs 1 --steps 600 --start-mode first --clip-index 0 --warmup 0 \
  --executor sonic --replay-target-npz "$NPZ" \
  --ha-task Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody \
  --ha-yaml tasks/common_test_config/base_test/doubledesk_sonic_test.yaml \
  --seed 0 --out runs/calib/replay_sonic.pt
```

`scripts/_ab_replay.sh <sonic|ckpt> <log>` 是封装好的 A/B 脚本。

---

## 7. 同一问题对下游的影响（待处理，未改）

`runs/isaac_motion/ha_cmd_exec/packed.pt` 的 q/qd 是 **canonical 序的值 + standard 序的标签**
（`build_ha_exec_lib.py:18` 的注释写的是 `STANDARD_29 / MJCF 序`，与实测不符）。

受影响的地方（都还在用 `idx_t`）：

1. `isaac_a_collect_batch.py` 的 **A' 采集初始化** `q0[:, idx_t] = rows[:,13:13+nj]`；
2. `--segment N` 分段重置里的同一句；
3. 任何"按 `meta['joint_names']` 去对齐库 q"的奖励/WM 状态构造。

**为什么没有直接改**：WBC ckpt 的 `robot_spec.joint_names` 也是 standard 序，
它的 obs/动作口径是按旧约定训出来的；改映射要连 `build_wbc_lib.py` 一起核对，
否则会把"训练口径"和"仿真口径"再对错一次。

**建议的验证顺序**（每步 3–5 min，一次 1 env）：
1. A' 采集里把 `q0[:, idx_t]` 改成 `q0[:, idx_s]`，跑 50 步看 `z/up` 是否变干净；
2. 若变干净 → 同步改 `--segment` 重置与奖励参考；
3. 最后再决定 WBC 是否需要按新口径重训（或只在推理侧做一次固定置换）。

---

## 8. 官方 replay 二进制：终于跑通了（同一个 env，官方代码路径）

### 8.1 之前"起不来/卡住"的真正原因

`action_provider/create_action_provider.py:12` 有门禁：

```python
SONIC_ALLOWED = os.environ.get("HA_ALLOW_SONIC", "0") == "1"
```

没导出 `HA_ALLOW_SONIC=1` 时，`create_action_provider` 直接 `raise SystemExit(...)`；
主线程退出，但 **ImageServer 是非守护线程**，于是进程不退出、只反复刷同一帧画面，
看起来像"卡在 provider 创建前"。**加上这个环境变量就通了。**

### 8.2 跑通用的命令

```bash
export HA_ALLOW_SONIC=1 SUPERMIMIC_HA_ALIGN=1
export PYTHONPATH="$C/third_party/unitree_sdk2_python:$PROJECT_ROOT:$C:\
$C/trackerlab-supermimic/source/trackerTask:$C/trackerlab-supermimic/source/trackerLab"
$ISAACSIM_PY -u sim_main.py --device cuda:0 \
  --env_config_yaml tasks/common_env_config/doubledesk_sonic.yaml \
  --task Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody \
  --robot_type g129 --input_source replay --gmt_backend sonic \
  --sonic_encoder_path .../model_encoder.onnx --sonic_decoder_path .../model_decoder.onnx \
  --replay_file <npz> --replay_mode direct_replay \
  --seed 42 --headless --enable_cameras --exit_when_replay_complete
```

注意 `--enable_cameras` 必须带（任务场景里 spawn 了相机，否则 `Failed to create environment`）。

### 8.3 官方给的对拍指标（`_log_replay_joint_position_error`，每 10 帧一条）

| frame | 官方 mae | 官方 max | 我们（同 clip，修序后）mae |
|---|---|---|---|
| 0 | **0.0014 rad** | 0.0099 | 0.031 |
| 10 | 0.0029 | 0.0136 | — |
| 30 | 0.0027 | 0.0139 | — |
| 50 | 0.0115 | 0.0865 | 0.056（step 40） |

**官方在第 0 帧比我们准 20 倍，前 50 帧一直是 mrad 量级**。

### 8.4 还有的速度问题（未解决）

官方主循环实测 **约 5.5 s/控制步**（2080 帧要 3 小时），而我们的采集器同样场景
16 env 是 ~18 步/s。已排除：`env.scene.get_state()`+JSON 序列化（已加
`input_source == "replay"` 时跳过，见 `sim_main.py:1283`，对速度无影响）。
最可能是 `env.step()` 里的渲染（我们采集器走的是 `sim.step(render=False)`）。

### 8.5 对 `sim_main.py` 的本地改动（仅 replay 模式生效）

* `sim_main.py:1283`：replay 模式下跳过每 10 步的整场景 DDS 序列化。
  （注：不要再往前走一步去跳过整个 `if not args_cli.replay_data:` 块——
  它的 `else` 是**旧 JSON replay 分支**，跳过去会走到 `FileActionProviderReplay`，
  报 `'SonicActionProvider' object has no attribute 'get_start_loop'`。）

---

## 9. 与官方逐项对齐后的定位（2026-11-03 深夜）

### 9.1 已在我们这边补齐的"同通路"开关

| 开关 | 作用 |
|---|---|
| `SM_STEP_MODE=official` | 逐字节照抄官方推进循环：每物理步 `set_joint_position_target(43 维)` → `write_data_to_sim` → `sim.step(render=False)` → `scene.update(dt=physics_dt)` |
| `SM_SKIP_INIT_WRITE=1` | 不写 npz 初值，直接用环境 reset 后的状态（官方做法） |
| `SM_OFFICIAL_RESET=1` + `SM_EPISODE_SEED` | `sim.reset → env.reset → reset_all_self`（官方 `_perform_complete_reset`） |
| `SM_REPLAY_KEY` / `SM_REPLAY_IDS` | 目标键、关节序可切换 |
| 手部 | 从 npz 的 `hand_action_left/right` 写入 43 维 action（官方 `_apply_hand_targets`） |

### 9.2 结论：序、初值、推进循环、USD、增益 **都不是** 剩下的差异

四组对照的 step0 关节 mae **完全相同（0.0303）**：
（a）手写 29 维通路、（b）`env.step(43)`、（c）官方推进循环、（d）官方循环 + 不写初值 + 官方 reset。

### 9.3 差异被定位到 **手臂**，而且是**力矩**

同一帧数据、同一指令（左肘 `cmd=0.338`，初值 `q=1.200`）：

| | 一步后角速度 | 一步后角度 |
|---|---|---|
| 录制（`robot_qvel_before_decimation[1]`） | **−12.8 rad/s** | 0.978 |
| 我们（`robot.data.joint_vel[0,21]`） | **−4.0 rad/s** | 1.117 |

同时确认这些量**完全一致**：`kp=40.0`、`kd=5.00`、`armature=0.0`、`friction=0.0`、
`vel_limit=37`、`dt=0.02`、USD（`_m2` 与非 m2 曲线逐位相同）、指令值。

按 `I·α` 估算，我们的实际力矩只有指令的 **≈1/3**（指令 40×(1.2−0.338)=34.5 Nm）。
⇒ **最后一个嫌疑是关节力矩被限幅**（`effort_limit_sim` / USD drive `maxForce`）。

### 9.4 下一步（一条命令级验证）

1. 打印 `robot.data.applied_torque[0,21]` 与 `actuator.effort_limit_sim`；
2. 若被限在 ~11 Nm：用 HA spec 的 `effort`（肘 60 Nm）显式
   `robot.write_joint_effort_limit_to_sim(...)`，再对拍 step0；
3. 若限幅不是原因，则只剩"录制侧 target→q 的语义"（可能录制时手臂走的是另一条
   通道/带一帧延迟），再用官方 `SONIC_DEBUG=1` 的 `[SONIC][TRACKING_TOPK]` 对比目标。

### 9.5 影响评估（对 A' 流水线）

手臂力矩差 3 倍会直接改变 contact/平衡，因此：
* 我们采集的 A' 数据里手臂动力学与录制**不同**；
* 开环回放在 ~1.4 s 倒下、官方能撑到 ≥2.4 s，与此一致。
**在 9.4 修好之前，A' 数据的"物理真实性"仍然不达标。**

---

## 10. 官方基准 vs 我们：逐项排除后的最终定位（2026-11-03 深夜）

### 10.1 官方基准（真的跑起来了）

`HA_ALLOW_SONIC=1` 之后，官方 `sim_main.py --input_source replay --replay_mode direct_replay`
在**同一个 npz、同一个任务**上给出：

```
frame=0  mae=0.001362 rad  max=0.009897
frame=30 mae=0.002678     max=0.013901
frame=120 mae=0.057272    max=0.508781   ← 2.4 s 仍未崩
```

我们同一份数据、同样修好关节序后：**frame0 mae=0.0303**（差 20 倍），~85 帧倒地。

### 10.2 下列全部被**排除**（四组不同实现给出**逐位相同**的 0.0303）

| 变量 | 试过的两种做法 | 结果 |
|---|---|---|
| 初值 | 写 npz 第 0 帧 / `SM_SKIP_INIT_WRITE` 不写（靠 env reset） | 相同 |
| reset | 不 reset / `SM_OFFICIAL_RESET=1` + `episode_object_seed` | 相同 |
| action 维度 | 只写 29 维 / 写满 43 维（含手部 `hand_action_left/right`） | 相同 |
| 推进循环 | 手写 `write_data_to_sim+sim.step+scene.update` / `env.step(action)` / 官方逐物理步循环 | 相同 |
| 机器人 USD | 默认 / `ROBOT_USD_OVERRIDE=..._m2.usd` | 相同 |
| PD 增益 | 写 `_ha_spec` / 不写 | 相同（env 默认已等于 HA spec：hip 130 / knee 160 / ankle 60 / arm 40） |
| 手部/左右 | 左右手臂互换假设 | 排除（identity 明显更优：0.194 vs 0.499） |

### 10.3 差异被**精确定位到手臂的力矩/惯量**

同一指令、同一初值（左肘 `cmd=0.338`，`q=1.200`），一步（20 ms）之后：

| | 角速度 | 位置 |
|---|---|---|
| 录制 `robot_qvel_before_decimation[1]` | **−12.78 rad/s** | 0.978 |
| 我们 `robot.data.joint_vel[0,21]` | **−4.01 rad/s** | 1.117 |

实测 `applied_torque[21] = −11.44 Nm`，与 `kp·err − kd·v = 40×(0.338−1.117) − 5×(−4.0) = −11.1`
吻合 ⇒ **力矩没有被限幅，PD 就是按 kp=40 在算**。
⇒ 差的那 3 倍只能来自**等效惯量**（我们比录制大 ~3 倍），而不是增益、不是限幅、不是序。

### 10.4 为什么这一条重要

手臂惯量差 3 倍 ⇒ 同样指令下手臂运动学完全不同 ⇒ 接触/平衡完全不同 ⇒
这就是"我们 1.4 s 倒、官方 ≥2.4 s 不倒"的直接原因，也是 A' 数据"物理不真实"的直接原因。

### 10.5 下一步（唯一还没做的两件事）

1. **在官方进程里打开 `SONIC_DEBUG=1`**，拿到它的 `[SONIC][TRACKING_TOPK]`
   （`pos_error = current_pos − sonic_targets`），和我们同一帧逐关节对拍，
   确认它的手臂是否也"一步走 0.22"；
2. **在我们进程里直接实例化官方 `SonicActionProvider`**（而不是自己拼循环），
   若它的结果也是 0.0303，则差异 100% 在 env 构造（`apply_env_config_yaml` 之外
   还有 `task_runtime_profiles.apply_task_runtime_profile` 那一步我们没做）；
   若它的结果是 0.0014，则差异在我们缺失的某个 provider 内部步骤。

---

## 11. 官方 `SONIC_DEBUG=1` 的逐关节证据（决定性）

同一帧（frame 0，左肘 `target=0.3377`，初值 1.2）：

```
官方 [SONIC][TRACKING_TOPK]
  target=[..., left_elbow_joint=+0.3377, ...]
  actual=[left_elbow_joint=+0.9781, right_elbow_joint=+0.9416, ...]
  dq    =[right_elbow=-14.67, left_elbow=-12.75, left_shoulder_pitch=+5.24, ...]
  → 官方"与录制 q"的误差 mae=0.001362
我们  same step: left_elbow=1.117, dq=-4.01, 与录制 q 的误差 mae=0.0303
```

⇒ 官方的手臂在**同样的目标、同样的 env、同样的任务**下一步就到 0.978（−12.75 rad/s），
我们只有 1.117（−4.0 rad/s）。**这不是"我们的循环写法"问题**（§10.2 已逐项排除），
而是**机器人手臂关节的响应本身不同**。

同时实测我们侧 `applied_torque[21]=−11.44 Nm`，
与 `kp·err−kd·v = 40×(0.338−1.117) − 5×(−4.01) = −11.1` 吻合 ⇒ PD 是按规定算的，**没有限幅**。
⇒ 结论锁定为 **等效惯量/关节参数**（我们比官方大约 3 倍），而不是顺序、初值、增益或限幅。

### 下一实验（唯一还没做的一条）

在我们进程里**直接实例化官方 `SonicActionProvider`**（构造 + 每步 `get_action()`），
它的 env 与我们的完全一样：

* 若它给出 **0.978 / mae 0.0014** ⇒ 差异在我们 env 构造之外（provider 内部还有一步我们没复制）；
* 若它给出 **1.117 / mae 0.0303** ⇒ 差异 100% 在 env 构造（我们和 sim_main 之间仍有未对齐的
  场景/机器人参数）。

这一步是"二选一"的判决性实验，约 30 分钟。

---

## 12. 判决性实验：官方 provider 跑在**我们的进程**里（结论确定）

新增开关 `SM_USE_OFFICIAL_PROVIDER=1`：直接实例化官方 `SonicActionProvider`，
由它自己的 `get_action()` 推进仿真（env 仍是我们构造的那个）。

### 12.1 结果：0.0014 —— 和官方二进制**逐位一致**

```
[replay-off] step=  0 mae=0.0014 max=0.010 | z=0.797  写入目标[21]=0.3377 实测[21]=0.9781
[replay-off] step=  1 mae=0.0030 ...        | z=0.790
[replay-off] step= 20 mae=0.0026 ...        | z=0.789
[replay-off] step= 40 mae=0.0058 ...        | z=0.790
```

⇒ **env 构造是对的、数据是对的、关节序是对的**；
我们手写的那条 replay 循环（0.0303）就是剩下的 bug。
官方写进去的目标同样是 `0.3377`，但它的实测是 `0.978`，我们手写循环只有 `1.117`。

### 12.2 但"不会倒"依然不成立 —— 官方实现也会倒

同一条 900 步长跑（官方 provider 推进）：

| step | mae | z |
|---|---|---|
| 0 | 0.0014 | 0.797 |
| 40 | 0.0058 | 0.790 |
| 80 | 0.0498 | 0.788 |
| 120 | 0.0573 | 0.767 |
| **140** | 0.0580 | **0.678** |
| **160** | 0.1047 | **0.089（已倒）** |

⇒ **开环回放这条 41 s 的片段，官方实现也只能撑到约 3 s**。
我们手写循环撑 ~1.7 s（85 步），官方 3 s（150 步）——差的是精度，不是"倒不倒"。

### 12.3 结论（可以写进验收）

1. **回放精度**：用官方 provider 驱动 = `0.0014 rad`（与我们手写循环的 `0.0303` 差 20 倍）；
2. **稳定性**：开环关节回放 ~3 s 必倒，**这不是我们代码的 bug，官方同样如此**；
3. 想要"物理真实且不摔"的 A'，仍然只能走**闭环 + 每 20 步重置到参考**。

### 12.4 用法

```bash
SM_USE_OFFICIAL_PROVIDER=1 SM_OFFICIAL_RESET=1 \
SM_EPISODE_SEED=<npz 的 episode_object_seed> SM_SKIP_INIT_WRITE=1 \
SM_REPLAY_IDS=sonic ... --executor sonic --replay-target-npz <npz>
```

---

## 13. 闭环（A'）现在的情况：初值关节序修好后**全程不摔**

### 13.1 修的是什么

A' 采集的初值 `q0[:, idx_t] = ref_q`，而库里的 q 是 **canonical 值 + standard 标签**。
改成 `SM_INIT_IDS=sonic`（用 `idx_s` 写初值）后：

| 配置 | 结果 |
|---|---|
| 旧（idx_t 写初值） | z 0.80 → 0.60@160 → 0.26@180 → **摔倒** |
| 新（idx_s 写初值） | **2080 帧全程不摔**：z ∈ [0.752, 0.798]，up ≥ 0.936 |

### 13.2 但"偏差大"还在（量化）

同一条 clip 全长 2080 帧（存了 1362 行有效数据）：

| step | root 漂移 | 关节 mae（vs 参考） |
|---|---|---|
| 0 | 0.000 m | 0.049 rad |
| 100 | 0.297 m | 0.245 |
| 300 | 0.970 m | 0.300 |
| 600 | 1.275 m | 0.273 |
| 1000 | 3.739 m | 0.228 |
| 1361 | 3.194 m | 0.363 |

* root 误差 < 5 cm 的帧：**0.6%**（奖励基本恒 0）
* z：参考 0.780 / 我们 0.777，几乎无偏
* **站得住 ≠ 跟得上**

### 13.3 已经把 WBC 口径也试过（排除）

`SM_WBC_IDS=sonic`（obs/动作都换成 SONIC 序 + `default_s`/`scale_s`）→ **第 100 步就倒地**
（z=0.089, up=−0.135）。说明 WBC ckpt 的 ckpt/standard 口径是**对的**，
偏差不是映射错，而是 **WBC 自身的跟踪精度**（0.28 rad 量级）。

### 13.4 结论

1. **闭环全程不摔**：已解决（新修的初值关节序）；
2. **漂移 0.3→3.7 m**：这是 WBC 跟踪精度问题（0.28 rad），不是数据/映射 bug；
3. 想拿到有用的 A'
 * 短期：**每 20 步重置到参考**（段内偏差只剩 0.05–0.1 m，奖励才有信号）；
 * 长期：把 WBC 的关节跟踪 RMSE 从 0.28 压到 <0.1 rad。
