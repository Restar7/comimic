# 交接：ĝ 蒸馏（把 WM 的 20 步梯度当标签训 Δtoken）

> 写于 2026-11-07 01:25。训练正在跑（`runs/gt6ddp`，2 卡 DDP，150 步，10.3 s/步，预计 01:47 结束）。
> 如果被中断，本文档包含：怎么续、怎么判、所有踩过的坑。

---

## 0. 一句话结论

**WM 的梯度方向在真机上是好的**（H=2 96/96、H=20 63/64 窗口变好），但
**WM 高估幅度 1.8×**，而且**"逐窗口学 ĝ"只能拿到 cos 0.3~0.5**。
所以本轮把训练目标从「用报酬反传（跨窗口求和 → 常向量）」换成
「**回归 ĝ 标签**（逐窗口、不求和）」，即 **蒸馏**。

---

## 1. 今天的实测（决定性证据，全部可复现）

### 1.1 真物理 ±δ 检验（判"梯度能不能信"）

脚本：`scripts/probe_wm_gradient_truth.py plan|score` + `scripts/isaac_probe_action_rollout.py`
数据：`runs/probe_truth/exc_dump.pt`（由 `all_transitions.pt` 反置换生成，口径已验证）
日志：`logs/truth_probe.log`（H=2）、`logs/truth_probe_h20.log`（H=20）

| 视界 | 符号一致率 | 真物理变好比例 | ΔJ 相关 | J_true 均值 | J_WM 均值 |
|---|---|---|---|---|---|
| H=2（96 窗口） | 1.000 | **1.000** | +0.964 | +0.0076 | +0.0087 |
| H=20（64 窗口） | 0.984 | **0.984** | +0.092 | **+0.059** | **+0.128** |

⇒ ① **方向可用**（连 H=20 都可）；② **WM 高估约 1.8×**（0.128 vs 0.059）。
⇒ 这条直接解释了 v13 为什么比 base 差（28.6% vs 66.7%）：每一步都在过冲。

### 1.2 可学性（判"能不能学"）

详见 `docs/GRAD_OPTIMUM_WHAT_IT_DEPENDS_ON_20261106.md`。要点（n=3072，8 折 CV）：

| 目标 | 特征 | cos（ridge / MLP） |
|---|---|---|
| `∂r₀/∂token`（单步） | token | 0.19 / 0.18 |
| | state | 0.39 / 0.42 |
| | state+act0 | 0.47 / **0.50** |
| `∂J₂₀/∂token`（20 步） | token | 0.21 / 0.24 |
| | state+token | 0.23 / 0.33 |

⇒ **20 步标签可学性差（0.22~0.35），单步好（0.47~0.50）**；token 本身几乎没信息。
⇒ 之前"R² 全 ≤0"是 n=256 的小样本伪影，n=3072 后转正（state+act0 R² 0.195）。

### 1.3 在线（推理期现场算）的成本 —— 实测

`OnlineGradCorrector.grad_action`（batch=1，含一次 autograd）：

| H | 前向+反向 | 对照：现有闭环 2.6 s/chunk |
|---|---|---|
| 2 | 231 ms | +9% |
| 5 | 439 ms | +17% |
| 10 | 987 ms | +38% |
| 20 | **1869 ms** | +72% |
| 只前向 H=20 | 88 ms | —— |
| **Δtoken 一次前向**（蒸馏版） | **0.105 ms** | ≈0% |

⇒ 在线**只能当离线标签采集器**，不能当交付形态（否则闭环慢近一倍）。
⇒ 交付形态是蒸馏版 Δtoken（推理零负担、不带 WM）。

---

## 2. 代码改动（全部默认关闭，老行为逐位不变）

| 文件 | 改动 | 开关 |
|---|---|---|
| `diff_vla_wbc/v3_step.py` | ① `V3Config` 新增 `delta_token_fullstate` / `grad_target_weight` / `grad_target_n` / `grad_target_eta` / `grad_target_horizon` / `grad_target_every`；② `_delta_token_features` 支持返回**部署 state64 全 64 维**；③ 新增 `_grad_target_loss()`（稠密 ĝ 监督）；④ 新指标进 metrics | 见右 |
| `diff_vla_wbc/delta_token.py` | 新增 `STATE64_DIM = 64` | —— |
| `scripts/train_joint_v2.py` | `--delta-token-fullstate`、`--grad-target-weight/-n/-eta/-horizon`；写进 `V3Config` | —— |
| `scripts/launch_joint_ha_v2.sh` | env：`DELTA_TOKEN_FULLSTATE`、`GRAD_TARGET_WEIGHT/N/ETA/HORIZON` | —— |
| `diff_vla_wbc/online_grad.py` | **新增**：`OnlineGradCorrector`（现场算 `∂J/∂a` 并施加 η·unit），离线验证通过 | 仅离线用 |
| `scripts/build_closedloop_dump.py` | **新增**：把 `dump/steps.jsonl` 拼成探针 dump + 口径自检 | —— |
| `third_party/.../action_provider_sonic.py` | 曾加过在线 hook，**已回滚**（现 6022 行，与原文件逐位一致；备份 `/tmp/ap_sonic_backup.py`） | —— |

### 新指标（关键判据）

| 指标 | 含义 | 判据 |
|---|---|---|
| `delta_token/grad_cos` | Δ 与 ĝ 的余弦 | 老路 ≈0；**目标 >0.3 且上升** |
| `delta_token/grad_norm_ratio` | ‖Δ‖/η（η=0.08） | 老路 1.9（过冲）；目标 0.7~1.1 |
| `delta_token/state_sens` | 打乱状态后 Δ 的变化比例 | >0 才说明在看状态 |
| `delta_token/norm` | ‖Δtoken‖ | 应贴近 0.08 |

---

## 3. 正在跑的这一轮（如果被中断，按 §4 续）

```
RUN=gt6ddp   STEPS=150   NPROC=2 DDP=1 DDP_GPUS=0,1   BATCH=512(全局) ACCUM=2  → 有效 1024
WM_CKPT=runs/a_prime_sonic_exc/wm_gain2.pt   TERM_OFF=1   LAMBDA_R=0   VAR_WEIGHT=0
ACT_WEIGHT=0.15 ACT_EPS=0.30  TRUST_CENTER=base TRUST_EPS=0.05 TRUST_WEIGHT=0.2
DELTA_TOKEN=1 DELTA_TOKEN_FULLSTATE=1 DELTA_TOKEN_LR=5e-4 DELTA_TOKEN_INIT_STD=4e-4
GRAD_TARGET_WEIGHT=1.0 GRAD_TARGET_N=8 GRAD_TARGET_ETA=0.08 GRAD_TARGET_HORIZON=20
FREEZE_VLA=1 START_MAX_DRIFT=0.10 INIT_VLA=  EXTRA="--real-transitions runs/a_prime_sonic_exc/all_transitions.pt --boot-one-sided" H=20
```

**为什么这么配**
- `LAMBDA_R=0`：学生完全不穿 WM ⇒ 彻底移除"跨窗口求和 → 常向量"的机制（纯蒸馏，归因最干净）；
- `GRAD_TARGET_HORIZON=20`：标签视界与 rollout/部署的 20 步**对齐**（之前试过 k=2，与 H=20 不对齐）；
- `GRAD_TARGET_ETA=0.08`：= 0.15 ÷ 1.8（真机实测的幅度高估）；
- `DELTA_TOKEN_FULLSTATE=1`：输入补上 dof_pos 29 维（cos 0.24 → 0.39+）。

**早期趋势（01:25，step 22/150）**

```
step  1  cos −0.027  norm 0.008  ratio 0.09
step  5  cos −0.011  norm 0.065  ratio 1.33
step 10  cos +0.066  norm 0.059  ratio 0.55
step 15  cos +0.172  norm 0.057  ratio 0.43
step 22  cos +0.152  norm 0.057  ratio 0.70
```
⇒ **幅度已经落在 η 附近（老路是 1.9×）**，cos 从负转正到 ~0.15（还在爬）。

---

## 4. 中断了怎么办

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
# 续训（从 runs/gt6ddp 的最后一个 ckpt 起）
INIT_VLA=runs/gt6ddp/vla_000100.pt RUN=gt6ddp2 STEPS=150 ...   # 其余同上
# 或者重跑（150 步只要 ~26 分钟，2 卡）
```

判断"是否还在跑"：`ps -eo args | grep train_joint_v2`、`tail runs/gt6ddp/train_metrics.jsonl`。
ckpt 落盘：`runs/gt6ddp/vla_000050.pt / vla_000100.pt / vla_final.pt`（SAVE_EVERY=50）。

---

## 5. 跑完之后的验收流程

```bash
# ① 探针（30 s）：看残差/均值、‖Δa‖/η
python3 scripts/probe_ft_landing.py --arms base= v13=runs/joint_v13/vla_000100.pt \
    new=runs/gt6ddp/vla_final.pt --n-noise 3 --max-req 24 --device cuda:0
#   判据：‖Δa‖ ≤ 1.5η；残差/均值 明显脱离 0.098

# ② 闭环（2 任务 × 1 seed，约 15 min）
HA_PIN_BYPASS= ... bash scripts/r3_closedloop_ab.sh  # TAG=new TASK=SitSofa|VisionNavi HEAD=runs/gt6ddp/vla_final.pt
#   判据：成功率 ≥ base（SitSofa 66.7% / VisionNavi 50.0%）
```

**闭环 launcher 的两个坑**（今天踩过）：
1. `r3_closedloop_ab.sh` 的 `SRV_PY` 里带了 `third_party/skimage_py311`（cp311）⇒ 在本机 py3.12 下
   `ModuleNotFoundError: skimage.measure._ccomp`。**用 `scripts/r3_sr_cell.sh`（py312_shims 版）**；
2. `r3_sr_cell.sh` **不会**自动导出 sonic 栈环境变量，必须显式给：
   `HA_ALLOW_SONIC=1 HA_GMT_BACKEND=sonic HA_ACTION_SOURCE=sonic_wholebody
    SONIC_VLA_ACTION_FORMAT=latent64 SUPERMIMIC_HA_ALIGN=1 HA_VLA_EXEC_STEPS=20`
   （缺了会**静默回落到 SuperMimic provider**）。

---

## 6. 尚未解决的问题（下一个人接手要知道）

1. **闭环 dump 的关节口径没定死**：`runs/closedloop_r3/*/dump/steps.jsonl` 的 `q`
   与运动库 ref 的关节**对不上**（一步预测误差 0.415 rad；加 `ha_v31→standard` 置换更差 1.48 rad）。
   ⇒ 任何"用闭环录制数据评估 WM/漂移"的结论都不可信（我今天因此**撤回**过一组漂移分解数据）。
   ⇒ 想解决：改 `action_provider_sonic.py` 加一个**只 dump 不改动作**的 hook（代码已写过、位置已知），
      在运行时把 `robot.data.joint_pos / root_state / 当前参考` 一次对齐录下来。
2. **参考帧来源未确认**：`isaac_replay_online.sh` 的 `LIB` 默认是 `runs/isaac_motion/ha_cmd/packed.pt`，
   但用 `frame` 去索引两种库都匹配不上（最近邻 0% 命中）⇒ 运行时的参考到底是哪份数据还没查清。
3. **在线修正的接线**（若将来要用）：服务端只有 35 维状态、没有参考帧 ⇒ 需要在 HA 侧算（provider 里
   有完整状态和参考），或给 payload 加旁路字段（**不进模型输入**，所以**不需要重训 VLA**）。
4. `GRAD_TARGET_HORIZON=2` 与 `H=20` 不对齐的教训：短视界好学但目标不一致，已改回 20。
   （若 20 步学不动，下一步试 k=5，并同步把 rollout/reward 的视界也调成 5，保持对齐。）

---

## 7. 关键文件/日志索引

| 内容 | 路径 |
|---|---|
| 本轮训练 | `runs/gt6ddp/`（`args.json` 有全量配置）、`logs/gt6ddp.log` |
| 上一轮（k=2, λ_r=5，已废弃） | `runs/gt5cut/`、`logs/gt5cut.log`（可用于对比"不对齐"的坏结果） |
| 真机 ±δ 检验 | `logs/truth_probe.log`、`logs/truth_probe_h20.log`、`runs/probe_truth/exc_dump.pt` |
| 可学性/视界扫掠 | `docs/GRAD_OPTIMUM_WHAT_IT_DEPENDS_ON_20261106.md`、`runs/gradtok2/` |
| 在线成本基准 | 本文 §1.3（脚本内联在会话里；`diff_vla_wbc/online_grad.py` 可直接复测） |
| v13/base 闭环基线 | `runs/closedloop_r3/{v13,base}_s*/`（v13 30.8% / base 58.3%） |
