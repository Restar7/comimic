# A4 运行报告：世界模型冻结/更新 × imagination horizon（**Isaac 栈 / 纯 HA / 冻结代码快照**）

> 本机 = A4 的 **4、5 号卡**（2× RTX 4090 48 GB，49140 MiB）。
> 定义性运行 = **v3**：`runs/_code_a4_20261025/scripts/run_a4_v3.sh`（代码快照 + `CODE_MANIFEST.sha256`）。
> 每格固定墙钟 **30 min**，六格 ≈ 3.2 h；`--steps 1000000` 由墙钟截断。

---

## 0. 口径（写死）

| 部件 | 来源 |
|---|---|
| WBC policy / 世界模型权重 | `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt`（= `runs/OFFICIAL_SUPERMIMIC.json`，iteration 10500） |
| 99 维观测 / 奖励 / 软终止 | `trackerLab.tracker_env.mdp.supermimic.MDP`（`relative_ori99`、Table A.2、×dt） |
| 可微 FK（14 tracked bodies） | `trackerLab.kinematics.supermimic.g1.G1Kinematics` |
| 参考运动 + 世界系 root 速度 | `trackerLab.motion_buffer.motion_lib.CompactMotionLib`（`runs/isaac_motion/ha_tokens/packed.pt`） |
| 默认姿态 / 关节限位 | `trackerTask.supermimic.robots.get_robot('g1')` |
| VLA | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（`runs/OFFICIAL_CKPT.json`） |
| 数据 | `data/npz/ha_tokens`（纯 HA，699 clips / 634,016 frames）；**mixed / ft318 永久禁用**（加载器与打包器会硬性拒绝） |

没有任何 MuJoCo/mjlab 实现或权重参与：旧的「Isaac 权重 + mjlab 解析层」混用桥
（`diff_vla_wbc/trackerlab_bundle.py`）**已删除**，代码入口只有 `diff_vla_wbc/isaac_joint.py`。

### 0.1 官方底模是移动靶

该 Isaac WBC run 仍在**另一个容器里继续训**（实测 2026-10-24 16:27 还在写 `ckpt_016500.pt`）。
A4 六格统一钉在 `runs/OFFICIAL_SUPERMIMIC.json` 记录的 `ckpt_010500.pt`。换底模必须整格重跑。

---

## 1. 为什么要跑三代（v1 / v2 作废的原因）

| 版本 | 目录 | 状态 | 不作数的原因 |
|---|---|---|---|
| v1 | `runs/ablations/A4_pilot_v1/` | 保留参考 | ①评测窗口随 H 变（抽帧上界含 horizon）→ 各臂不是同一批样本；②指标是 H 步均值，跨 H 不可比；③每点只有 2 个窗口，噪声过大 |
| v2 | `runs/ablations/A4_v2_contaminated/` | **作废** | 跑到第 4 格时**另一路 agent 并发改了 4 个核心文件**（`isaac_joint.py` / `data.py` / `rollout.py` / `train_joint.py`，00:21–00:45）→ 前后格子跑的不是同一份代码；且当时动作换算口径是错的 |
| **v3** | `runs/ablations/A4_wm_horizon/` + `runs/_code_a4_20261025/` | **定义性** | 从冻结快照运行（开跑前 `sha256sum -c CODE_MANIFEST.sha256`）；使用 2026-10-25 修好的动作 / token / 速度口径 |

v2 的污染是怎么发现的：审计脚本比对**跨格 step-0 `token_mse`**（同一批固定评测窗口应该逐位相同），
前 3 格是 `0.053290`、后 2 格是 `0.150018` —— 一眼看出中间换了代码。

**固化的规矩**：这类固定墙钟多格实验必须从**代码快照**跑（`runs/_code_<实验>_<日期>/`），
并在每格开始前校验清单；`args.json` 指纹 + step-0 一致性是必须的收尾审计。

---

## 2. 2026-10-25 的三处口径修复（另一路 agent 查出，我独立复现后采纳进 v3）

三处都是"从 mjlab 迁到 Isaac 时积累的适配层口径错误"，不是训练脚本错、也不是 SuperMimic 学错。

### 2.1 动作换算多除了一次 `action_scale`（主因）

- `tracker_env/supermimic/env.py` 的语义是 `q_target = default_q + xi_bias + a * action_scale`，
  所以代码一直按 `a = (q_ref - default_q - bias) / action_scale` 反推参考动作。
- 但被测 ckpt 的实际训练口径是 `a = Δq`（相对默认姿态的弧度，**scale = 1**）：
  参考动作 rollout 用 `/scale` → reward **−0.129**、末步状态误差 1.31；
  用 `Δq` → **+0.0686**、末步 **0.047**。
- 修法：换算默认 `action_scale = ones`；奖励平滑项里的 `(a − a_prev) * action_scale` 仍用 spec。
  开关：`COMIMIC_ACTION_SCALE=spec`（旧行为，仅 A/B）、`COMIMIC_REWARD_ACTION_SCALE=spec|one`。
- **对 A4 的影响**：v1/v2 的观测历史里 `a_prev` 被放大 4–8 倍 → WBC 收到分布外输入 → 想象 rollout 全歪。
  v3 才是在正确口径上做消融。`check_isaac_bundle.py` 的"隐含关节目标"断言（6e-8）钉住这条。

### 2.2 HA 标签 token ≠ WBC 条件空间（只影响诊断量）

- HA npz 的 `g1_motion_token_64` 来自部署侧 `encoder_input(1762)` 前端，而 trackerlab WBC 条件在
  **640 维端口装配**；parity 检查 `max|Δtoken| = 1.0625`（HA）/ `0.4375`（ft318）。
- 所以 `token_mse` 过去是在**错误空间**里量的。修法：`scripts/build_port_tokens.py` 用端口装配重算
  699 条 clip → `data/port_tokens/ha_tokens/*.npy`，`RecordedTokenSource` 优先读侧车。
- A4 的训练条件是 **VLA 自己输出的 token**，不受这条影响；eval 的 `token_mse` 现在是"到端口 token 的距离"，
  仍然是**诊断量**（训练回路里没有 token 保真项）。

### 2.3 `state` 项被 WM 的速度通道主导（度量修正）

- 参考动作下位置/朝向/关节误差只有 1e-4 / 0.035 / 0.030，但 WM 想象的 `ang_vel` / `joint_vel` 误差是
  1.755 / 2.896（与参考速度相关仅 0.077）—— WM 是在 PhysX 里带 `action_noise=0.1` 的策略 rollout 上
  学出的"真实速度"，与光滑参考速度不同相。
- 修法：`LossWeights.state_vel` + `train_joint.py --state-vel-weight`（默认 **0.05**）。
- **对 A4 的影响**：v3 六格统一 `--state-vel-weight 0.05`（与 A1/A5 主线一致）。因此 `state_term`
  绝对值比 v1/v2 小一个量级（v3 起点 0.87 vs v1 17.5）——这是口径变化，不是训练变好。

---

## 3. v3 网格与协议

| 格子 | 变量 | 参数 | eval 间隔 |
|---|---|---|---|
| `wm_frozen_H8` | WM 冻结 × H=8 | `--wm-update-every 0` | 100 |
| `wm_frozen_H16` | WM 冻结 × H=16 | 同上 | 80 |
| `wm_frozen_H32` | WM 冻结 × H=32 | 同上 | 50 |
| `wm_frozen_H40` | WM 冻结 × H=40 | 同上 | 40 |
| `wm_live_H8` | WM 在线更新 × H=8 | `--wm-update-every 50 --wm-update-warmup 100 --wm-update-batch 8 --wm-update-window 8` | 100 |
| `wm_live_H40` | WM 在线更新 × H=40 | 同上 | 40 |

六格逐字相同的部分（`args.json` 里可以直接 diff 验证）：
`--batch-size 1`（2 卡 DDP → 全局 2）、`--seed 0`、`--vlm-dataset --vlm-steps 2`、
`--hand-loss-weight 0.2 --hand-grasp-boost 5 --state-vel-weight 0.05`、`--eval-batches 6`、`--save`、
`--time-budget-minutes 30`，同一 Isaac ckpt / 纯 HA 数据 / 打包运动库 / VLA ckpt。

**评测口径（跨 H 可比）**

- 抽帧与 H 无关：评测提供器固定 `horizon=40`（token chunk 长度），rollout 用各格自己的 H。
  （`MotionBatchProvider.sample` 的抽帧上界含 horizon，否则同一固定种子在不同 H 下会抽到不同帧。）
- 每点固定 **6 个窗口**（v1 只有 2 个）。
- 同时报满 H 与**固定 8 步前缀**：`rollout_reward@8` / `state_term@8` / `survival@8`。
  H 步均值随 H 变化（窗口越长 WM 漂移越多、连乘存活越小），跨 H 只能看前缀口径与 Δ/min。

**为什么 H 上界是 40**：VLA 一次输出 40 帧 token chunk，`imagine_with_tokens` 用
`tokens[:, min(t, T-1)]`，t>40 只是重复最后一帧。

**显存**：`--vlm` 全量训练 1.62 B 的 action head → fp32 参数 6.5 GB + 梯度 6.5 GB + AdamW 两矩 13 GB。
bs=2 实测峰值 46/48 GB（无余量）；bs=1 + 分块梯度 all-reduce（≤256 MB/块，数学等价）压到 **39.5 GB**。
launcher 另有显存门禁：可用 < 42 GB 直接拒跑，避免半路 OOM 浪费 30 min。

**墙钟退出是两卡协同的**：原来各 rank 各自判 `--time-budget-minutes`，先到墙钟的先 `break` → 另一个
rank 会在下一步的 `sync_grads → all_reduce` 上忙等死锁（实测复现：一个进程 38 GB 卡 9 分钟）。
v3 每步用 `all_reduce(MAX)` 协商停止。

---

## 4. 起跑前后必须看的证据

### 4.1 适配层数值自检（9 项，快照里跑过）

```bash
python3 scripts/check_isaac_bundle.py \
  --ckpt trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt \
  --packed runs/isaac_motion/ha_tokens/packed.pt
```

ckpt 键严格一致 / 状态布局往返 2.4e-7 / `step` 0.0（布局互换负向对照 4.14，证明守卫有效）/
`observation`·`reward`·`termination` 对原生 `MDP`：0.0 / 0.0 / 1.6e-12 /
99 维历史 flatten 0.0 / policy 前向 0.0 / 参考起始态 0.0、隐含关节目标 6e-8 /
参考态跟踪项 = 每步最大 0.1（3.7e-7）/ WM 窗口 0.0 / npz 帧数 = 运动库帧数 634,016 且 clip 名对齐。

### 4.2 跑完必跑审计

```bash
python3 scripts/audit_a4.py --root runs/ablations/A4_wm_horizon --logs logs/a4
```

逐格：Isaac ckpt / 纯 HA / 无 mixed·ft318 / 2 个 rank 都起来 / 有 `[budget]` 墙钟截断 / 无
Traceback·CUDA OOM·NaN / 评测点数量与单调性 / 格内 horizon 恒定 / 峰值显存；
**参数变化审计**：读每格 `joint_latest.pt` 与源 ckpt 逐张量比 —— `wm_frozen` 必须
`max|ΔWM| == 0`（真冻结），`wm_live` 必须 `> 0`（真更新），两者 `max|Δpolicy| > 0`（policy 在训）。
跨格：**除被消融的键外参数逐字相同**，且 step-0 `token_mse` 完全一致（= 同一批评测窗口）。
有 FAIL 时退出码 1。

### 4.3 汇总

```bash
python3 scripts/summarize_a4.py --root runs/ablations/A4_wm_horizon
```

输出满 H 与固定前缀两张表 + 峰值显存 / WM 更新次数。

---

## 5. 判读（写进论文前必须知道）

1. **`token_mse` 会上升**：训练回路里只有"奖励 + 状态（速度降权 0.05）+ 手部"，没有 token 保真项，
   所以 VLA 会**用 token 保真度换 reward**。`rollout_reward` 与 `token_mse` 必须**一起看**：
   frozen 臂 reward 涨而 token_mse 暴涨 = 典型 reward hacking；live 臂若 token_mse 涨得少而 reward 涨得多，
   才说明在线更新修正了代理偏差。
2. **H 步均值不能跨 H 比**，跨 H 只看 `@8` 前缀口径与 Δ/min。
3. **观测历史是"参考态播种"**：Isaac 主线用 PhysX 回放缓冲里的真实 `(state, a_prev, xi, f)`；我们没有在线
   仿真，历史用参考运动构造（六格完全一致）。rollout_reward 的绝对值不是 Isaac 训练曲线的延续。
4. **WM 在线更新是"参考转移上的有监督拟合"**，不经策略梯度、不来自 PhysX 状态；每次更新后立刻重新冻结。
5. **WM 更新 batch 只有 8**（Isaac 自己用 10×2048 窗口/更新）：每个更新要和 2.5e9 参数的 VLA 梯度步抢
   2 s 预算。若 live 臂"看起来没差别"，需 `--wm-update-batch 256` 做敏感性复查再下结论。
6. **终止/存活是解析软终止**（`MDP.soft_termination`），不是学出来的终止头 —— trackerlab 的 WM 没有终止头。
7. **A4 不含闭环成功率**：成功率是 A1 的口径；每格收尾落 `vla_final.pt`（6.1 GB）可接 A1 的成功率流程。
8. **动作 clamp**：Isaac env 把动作 clamp 到 ±20，我们没 clamp；实测 `rollout/action_abs ≈ 0.8–1.6`，
   离 20 有一个量级，不影响结论。
