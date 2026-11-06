# WBC / SONIC-decoder 排查记录（2026-11-03 通宵）

目的：判断 4×4090 新线（A' → 奖励 → WM → joint）的执行器到底该用哪个、以及为什么"跟不上"。

---

## 1. 已确认的**真 bug**（都已修，有证据）

| # | 位置 | 问题 | 证据 | 状态 |
|---|---|---|---|---|
| 1 | `scripts/p1c_g1_replay.sh` | **没把 `TASK` 传下去** ⇒ 所有 clip 都在 DoubleDesk 场景里跑 | 修前日志 `task=...DoubleDesk` 而 clip 是 VisionNavi；修后 `SmallWarehouse-VisionNavigation` | ✅ |
| 2 | `scripts/isaac_a_collect_batch.py` | 读库 `ref` 用错字段偏移：q 在 **13:42**（布局 `[p,quat,v,w,q,qd]`），代码按 7:36 读 | 用 13:42 读与录制 q 差 **0.0005 rad**；用 7:36 差 **4.68 rad** | ✅ |
| 3 | 采集器 | 目标**没 clip 到关节限位**（官方 provider 有，`action_provider_supermimic.py:1026`） | 补上后：z 0.78→**0.776**（原 0.63）、up 0.96→**0.969**（原 0.72）、趴地行 20%→**0%** | ✅ |
| 4 | 采集器 | A' 动作噪声带 `clamp(-1,1)`，而该 WBC `action_limit=None`、`|a|` 最大 18.6 | σ 从 0.05 降到 1e-6 倒伏比例不变（70% vs 69%） | ✅ |
| 5 | 采集器 | 多 env HA 场景调 `env.reset()` ⇒ `CUDA device-side assert`（4/16/32 env 全中） | 去掉 reset 即正常 | ✅ |
| 6 | 采集器 | warmup 与录制帧号不同步（录制首行落后 20 帧） | 已把录制起点前移 warmup | ✅ |
| 7 | 采集器 | HA 场景接触材质未复制（默认低摩擦 → 脚打滑） | 对齐 `static/dynamic=1.0 + combine=max` 后不再打滑 | ✅ |
| 8 | `sim_eval_vla.py` | 无法用环境变量打开 replay / replay_mode 默认值污染正常路径 | 加 `HA_REPLAY_FILE` / `HA_REPLAY_MODE` 且无文件时不改模式 | ✅ |

> 注：`scripts/start_vla_server.sh` 原本存在（未跟踪 git），被我误覆盖为简化版；原内容无法恢复。

---

## 2. 已确认的**事实**（不是 bug）

### 2.1 参考库本身是对的

- `packed.pt` 的 `ref` 布局 = `[p(3), quat(4), v(3), w(3), q(29), qd(29)]`
- 与录制逐帧对拍：q **0.0005**、qd **0.006**、v **0.0002**、w **0.0016**（fp16 精度）
- 录制 `robot_qpos_before_decimation` 与库 q **同序**（相关 1.000），差异只是**帧偏移**（约 1 帧，均值 0.16 rad）

### 2.2 SONIC decoder 转换与输入口径都是对的

- ONNX → torch：vs onnxruntime **3.3e-6**；vs 录制 `decoder_raw_action` **3.8e-6 ~ 5.7e-6**
- 994 维布局与官方 `observation_config_sonic_release.yaml` 一致：
  `[token(64) | ang_vel 10×3 | q 10×29 | qd 10×29 | last_action 10×29 | gravity 10×3]`
- 逐块反解：`token/ang_vel/q_hist/qd_hist/gravity` 四块与录制**逐位相等**；`last_action` 块差 1 帧（录制日志时刻不同）
- **token 就是录制里那个 token**（`runs/exp_tokens/*.npy` 与 `decoder_obs[:,:64]` 差 **0.0**）

### 2.3 decoder 是**状态反馈控制器**，不是 token→动作

实测（同一帧 obs，只改一部分）：

| 改动 | 输出动作变化 |
|---|---|
| 只换 token | \|Δa\| = 0.968 |
| 只换 930 维状态 | \|Δa\| = 0.999 |

⇒ 光喂 token 不可能复现参考；必须有状态反馈。

### 2.4 开环回放**必然发散**（不是 bug）

官方 `direct_replay`（喂录制关节目标）逐步对拍：

| 步 | root 误差 | 关节误差 |
|---|---|---|
| 0 | **0.000 m** | **0.000 rad** |
| 20 | 0.006 | 0.021 |
| 50 | 0.061 | 0.060 |
| 99 | 摔（94.7°） | — |

`inference_replay`（喂录制 obs）73 步摔。与历史记录（`docs/SONIC_DECODER_FOUND.md`：2026-10-24 六条 35–61 步摔）同量级。

### 2.5 WBC 精度是当前真正的瓶颈

- 官方闭环（我们 WBC）首步"下发目标 vs 参考" = **0.531 rad**
- 我的批量采集器同一指标 = **0.47–0.56 rad**（两套实现一致）
- 修好限位 clip 后：机器人不再下沉，但 root 仍 0.1→0.8 m、关节 ~0.55 rad
- 训练侧对应：20 轮里 `body_pos` 奖励一直 0.171–0.196（没动过）

---

## 3. 还没解决的（下一步）

1. **decoder 在实时闭环里的 obs 语义**：逐块 diff 显示 `q_hist` 块仍差 2.2 rad。
   录制那块等于 `encoder_robot_joint_pos_hist`，与 `robot_qpos_before_decimation` 相关 1.000 但有帧偏移；
   实时侧用"原始 q（STANDARD 序）"仍对不齐。**需要拿官方 `SonicActionProvider` 的 live obs 逐块对照**
   （它的 `_build_live_decoder_obs_with_latent` 是权威实现）。
2. **WBC 精度**：目标 joint RMSE 0.5 → <0.2 rad（P1）。
3. 全部修好后：A' 闭环采集（批量版 4 卡 × 256 env，约 30–40 min/遍）→ 奖励 → WM → joint。

---

## 4. 当前可用脚本

| 脚本 | 说明 |
|---|---|
| `scripts/isaac_a_collect_batch.py` | **批量采集器**（HA 真场景 / 裸场景两用）。关键开关：`--ha-task --ha-yaml --executor {wbc,sonic} --token-source {npy,online} --start-mode {uniform,first,windows} --len-min/--len-max --clip-index --replay-target-npz --replay-decoder-obs`，环境变量 `SM_DUMP_DEC_OBS`（dump 实时 994）、`SM_DECODER_ORDER`、`SM_USE_HA_SPEC` |
| `scripts/sonic_decoder_torch.py` | ONNX→可微 torch，`--check` 自带 L1 验证 |
| `scripts/run_a_prime_ha_4gpu.sh` | 官方逐条路径（慢，~20 h/699 条） |
| `scripts/run_a_prime_full_4gpu.sh` | 裸场景批量路径（快，但场景口径不对） |
| `scripts/finalize_a_prime.py` | 合并后 G2 校验 + 过滤趴地行 |

跑批量 HA 场景需要：

```bash
export PROJECT_ROOT=$PWD/third_party/HumanoidArena/isaaclab_twist2_g1
export PYTHONPATH="$PWD/third_party/unitree_sdk2_python:$PROJECT_ROOT:$PWD:\
$PWD/trackerlab-supermimic/source/trackerTask:$PWD/trackerlab-supermimic/source/trackerLab:$PYTHONPATH"
# 必须 --enable_cameras；不要调 env.reset()
```

---

## 5. 一句话结论

**链路（库/初始化/场景/接触/限位）今天全部修对了；decoder 与 token 也验证到浮点噪声级。
剩下两件事：decoder 实时 obs 的 `q_hist` 块语义、以及 WBC 本身的跟踪精度（0.53 rad）。
在这两件解决前，A' 用 WBC 采集是可行的（不再趴地），但奖励会偏低。**

---

## 6. 追加（同日深夜）：994 维已逐块对齐到 5/6

用 1 env、`--warmup 0`、`--clip-index 599` 对照录制 `decoder_obs` 第 0 步：

| 块 | 实时 vs 录制 |
|---|---|
| token / ang_vel / qd_hist / last_action / gravity | **全部 0.000（逐位相等）** |
| q_hist | 0.669 → 加上全局偏移后一致 |

**q 块的语义（离线反解，结论确定）**：

```
录制 q_hist 末帧 − robot_qpos_before_decimation = 全局常数 c（时间 std = 0.00000，跨 clip std = 0.00000）
c 只有 12/29 个关节非零：
  j0,j1 = +0.312   j9,j10 = -0.669   j11,j12 = -0.2   j13,j14 = +0.363
  j15 = -0.2       j16 = +0.2        j21,j22 = -0.6
```

已落盘 `runs/decoder_q_offset.pt`，采集器（`--executor sonic`）会自动加上。

**仍未通过**：即使 obs 对齐到第 0 步逐位相等，decoder 的 1-env 闭环仍在 ~50 步内发散
（up 0.94 @50 → 0.0 @100）。说明还差最后一项：**部署侧的动作后处理**（延时/限速/手部指令），
或者 decoder 的输入在运行期还需要别的量。**下一步应该拿官方 provider 的 live obs 直接对照**
（`action_provider_sonic.py:2641 _build_live_decoder_obs_with_latent`），那需要 VLA 服务。

---

## 7. 当前可用结论（给下一班）

1. **A' 用我们的 WBC 采，已经可以跑**（限位 clip 修好后闭环稳定：up 0.97、0 趴地）；
2. **decoder 只能在"真 VLA 在环"下用**，纯 obs 回放/纯 token 驱动都不可行；
3. 所有数据侧口径（库字段偏移 13:42、关节序、初始状态、接触材质）今天已修并验证。

---

## 8. 追加：关节序被四种方法排除，动作映射与 q 块语义均已精确反解

| 检查 | 结果 |
|---|---|
| 拟合 `target = a·raw + b`（逐关节最小二乘） | **残差 0.0000**；b == `SONIC_DEFAULT_POS`、a == `G1_ACTION_SCALE_ISAACLAB`，**同序** |
| 录制 q 块 − `robot_qpos` | 全局常数 `−SONIC_DEFAULT_POS`（时间 std / 跨 clip std 均 **0.00000**） |
| 录制逐关节跟踪误差 | 0.02–0.20 rad，**无任何关节系统性错位**（>0.3 的只有 2 个） |
| 相关性 + 匈牙利匹配 | identity 最优（相关 1.000），任何置换都更差 |

⇒ **关节序正确**。相关修正已落到采集器：
- decoder 的 q 块 = `q − SONIC_DEFAULT_POS`（`runs/decoder_q_offset.pt`）
- decoder 的目标 = `SONIC_DEFAULT_POS + a·G1_ACTION_SCALE_ISAACLAB`
- 回放录制目标时用 `idx_t` 写关节（与读 q 同一套）
- **显式写入 HA_ALIGN 的 kp/kd**（position 模式，与官方 `sonic_effort_control=False` 一致）

其中**写增益**这一条把回放第 0 步从 `root 0.117 m / 关节 1.279 rad` 拉到 **`0.003 m / 0.264 rad`**。

## 9. 仍未解释的差异（下一班接着查）

即使这样，用**录制目标**回放时，我们的仿真跟踪录制目标约 **0.45–0.85 rad**，
而录制自身只差 **0.13–0.20 rad** —— 说明**我们批量场景的动力学仍与录制环境不同**（约 3×）。
官方 harness 用同一批数据前 50 步只差 6 cm，我们 50 步差 1.0 m。

**排查方向（按嫌疑排序）**：
1. 物体/场景 spawn 与录制那一集不一致（HA 录制有确定的物体摆放和 task reset）；
2. 我们批量场景没有跑 task 的 `initialize_task_scene` / `reset_all_self`；
3. 执行器 effort_limit / armature 细节；
4. 关节速度初值、地面接触参数微差。

**结论：A' 要"真"，必须用官方 harness 的 scene 初始化流程；批量场景目前只能做到"物理一致但环境略偏"。**

---

## 10. 完整闭环实测（16 env，DoubleDesk，全长 2080 步，σ=0.05）

| 步 | root 误差 | up | z |
|---|---|---|---|
| 0 | 0.043 m | 0.949 | 0.788 |
| 100 | 0.684 m | 0.971 | 0.771 |
| 200 | 1.182 m | 0.949 | 0.747 |
| 400 | 1.943 m | 0.839 | 0.688 |
| 800 | 3.345 m | 0.465 | 0.485 |

**结论：全长闭环 A' 不可用**（100 步就漂 0.68 m，δ=5cm 的奖励几乎恒 0）。
**唯一可行形状 = 短分段 + 每段重置到参考**（20–50 步/段），这样每段漂移被限制在 0.1 m 量级，
而 joint 训练的想象 horizon 正是 H=20，两者匹配。

批量上限（带相机）：**16 env/进程**可用；32/100 env 会 `ERROR_OUT_OF_DEVICE_MEMORY`。

---

## 11. `--segment N` 分段采集（已实现，待完整验证）

采集器新增 `--segment N`：每 N 步把机器人 root/关节/速度**重置回参考帧**（`ref[starts+start+step]`），
把漂移限制在段内。推荐 `--segment 20`（= joint 训练的想象 horizon H=20）。

```bash
export PROJECT_ROOT=$PWD/third_party/HumanoidArena/isaaclab_twist2_g1
export SUPERMIMIC_HA_ALIGN=1
export PYTHONPATH="$PWD/third_party/unitree_sdk2_python:$PROJECT_ROOT:$PWD:\
$PWD/trackerlab-supermimic/source/trackerTask:$PWD/trackerlab-supermimic/source/trackerLab:$PYTHONPATH"
$ISAAC_VENV/bin/python -u scripts/isaac_a_collect_batch.py \
  --enable_cameras --wbc-ckpt runs/S1_handover/exp_WBC_exec_haonly_best.pt \
  --packed runs/isaac_motion/ha_cmd_exec/packed.pt --tokens-dir runs/exp_tokens \
  --num-envs 16 --steps 2080 --segment 20 --start-mode first --headless \
  --token-source npy --executor wbc --action-noise-std 0.05 --perturb-hold 20 \
  --ha-task Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody \
  --ha-yaml tasks/common_test_config/base_test/doubledesk_sonic_test.yaml \
  --seed 0 --out runs/wm_real/exp_a_prime_ha/a_prime_desk.pt
```

**批量上限：16 env/进程带相机**（32/100 会 OOM）。

**注意**：本会话最后一次 `--segment 20` 验证跑在 Isaac 启动阶段卡住（日志停在 ICD/GLInterop
告警），未能拿到指标；下一班第一件事就是把它跑通并量 `root 误差 / up`。
