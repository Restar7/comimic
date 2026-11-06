# RUNBOOK · J1 全流程一条命令（2×H200）

> 版本 v1 · 2026-10-28 · 前置：**W1 已跑完**，用 `ckpt_0007000` 作为底模。
> 脚本：`scripts/run_j1_rounds.sh`（**多轮自动闭环**，推荐）/ `scripts/run_j1_pipeline.sh`（单轮版）。
> 两者任一门不过都会**立刻停**，不会带着坏口径往下跑。

---

## 0.1 W1 结果与最佳 ckpt（2026-10-28，finish 20000）

W1 已经跑完（`logs/wbc_ft318_mjlab/`，20 个 ckpt，2.48 h，2.1 it/s）。训练曲线在 **~12k 之后完全平**：
`imag_ret` 3.059–3.061、`wm nse1 0.0009–0.0010`、`train_ret 0.890–0.892`、`surv 1.000`。

选 ckpt 只认**环境内 benchmark**（同库同 seed，另 2 条 clip 与 8 条 clip 各跑一遍）：

| ckpt | 2 clips（task_1） | 8 clips（4 任务）mean | 8 clips step-weighted |
|---|---|---|---|
| 007000 | 0.8779 | 0.7601 | 0.7874 |
| 010000 | **0.9103** | 0.7572 | 0.7969 |
| **014000** | 0.9046 | **0.7607** | **0.7985** |
| 017000 | 0.8995 | — | — |
| 020000 | 0.9025 | 0.7582 | 0.7963 |

⇒ **取 `ckpt_0014000`**（8 clip 上 mean/step-weighted 双第一；与 10k/20k 差 <0.5%，属噪声）。
底模 e9 同口径是 **0.7009**（2 clips）⇒ W1 带来 **+29%**。

### 一个必须知道的问题：**task_4 会摔（与选哪个 ckpt 无关）**

8 clips 里 `failed 2`，两条都是 **task_4 `diningroom_can_low_chair`**。我把四个 ckpt 都跑了一遍：

| ckpt | failed | 失败的 clip / fail_step |
|---|---|---|
| 007000 | 2 | T4 @ **634**, T4 @ **837** |
| 010000 | 2 | T4 @ **636**, T4 @ **837** |
| **014000** | 2 | T4 @ **632**, T4 @ **825** |
| 020000 | 2 | T4 @ **631**, T4 @ **825** |

⇒ **不是 14000 独有的问题**：四个 ckpt 在几乎同一帧（631–636 / 825–837）摔，其余 6 条全部跑完全长。

**参考数据不是坏点**（对比 `|Δq|max` / `|qd|max`）：
```
T4  0.146 / 5.72        T1  0.329 / 9.52        T3  0.293 / 9.49
```
task_4 的参考反而**更平滑**。失败点在该 clip 的 ~55%（632/1121、825/1477），
对应"去低凳取罐"的**深蹲/大幅重心转移**段 —— 更像是该动作段对 WBC 跟踪太难，
而不是训练或数据坏了。

**对 J1 的含义**：task_4 的 `fell=true` **不能**当作"J1 把策略训差了"的证据（W1 之前就这样）。
第一轮验收看 task_1/2/3 的 `reward` 与 task_4 的 `max_tilt_deg` 是否下降；
若四轮后 task_4 仍摔，单独立项诊断（深蹲段的动作可行性 / 参考根高启发式）。

## 0.2 与另一张 4090 的锁定口径逐条对齐（2026-10-28 复核）

对照物：`docs/HANDOVER_4090_COMMANDS_20261028.md`（他们那边实际在跑的 Isaac 流程）。
**原则：思想与锁定项必须一致，只允许"后端不同"和"能更大就更大"。**

| 他们锁定项 | 值 | 本流程 | 一致？ |
|---|---|---|---|
| `LAMBDA_BC` | **0**（token 不做 BC） | `--lambda-bc 0` | ✅ |
| `lambda-hand` | **0.2**（hand-only 掩码） | `--lambda-hand 0.2` | ✅ |
| `LAMBDA_R` | **0.3** | `--lambda-r 0.3` | ✅ |
| `--critic` | **关**（`--no-critic`；v_ev 为负） | `--no-critic` | ✅（已改） |
| 动作信任域 | `ACT_EPS=0.6 ACT_WEIGHT=1.0` | `--act-eps 0.6 --act-weight 1.0 --act-hinge-cap 2` | ✅（已加） |
| H（重规划周期） | **20** | `J1_H=20` | ✅（已改，原来 16） |
| `fm-lr` | **1e-4** | `J1_FMLR=1e-4` | ✅（已改） |
| A1（burn-in 末状态写回 VLA） | `--vla-state-from-burnin` | 同 | ✅（已加） |
| 起点 | `START_FROM_B`（B 线真实重规划状态） | 第 1 轮 = 参考帧 + DR；第 2 轮起接 B（见 §7） | ⚠️ 见下 |
| DR | `--no-dr`（因为起点已是真实漂移） | 起点=参考帧时**保留 DR**；接 B 起点时改 `--no-dr` | ⚠️ 条件一致 |
| LoRA | 32 | **64**（用户要求加大；H200 有余量） | 有意偏离 |
| batch | 4 × accum 8 = 有效 32 | H200：64 × 1 = 有效 64（单卡 64 / 双卡各 64） | 有意放大 |
| 采集 | 6 任务 × 3 集 × 600 步 ≈ 1.1 万步（Isaac，2 并发 45 min） | **4096 envs × 4000 步 ≈ 1640 万条**（向量化，几分钟） | 放大 1000×（但见 §7 语义差别） |

### 两个必须记住的语义差别（不是"不一致"，是后端决定的）

1. **我们的 WBC 冻结、他们可解冻**：他们 §5.3 有 `TRAIN_WBC=1` 的对照臂；我们 J1 只训 VLA head
   （WBC 由 W1 单独续训）。这对应他们表里"`--train-wbc` 是可选对照臂"，不是主线。
2. **采集的驱动源不同**：
   * 他们的 B 线是 **VLA 驱动**（server 出 token），所以能采到"VLA 自己把机器人跑偏"的状态；
   * 我们的 `collect_mjlab.py` 是 **WBC 参考跟踪**（记录 token），拿到的是"跟踪分布"，
     **只适合 WM 保鲜**，不能替代 B 线当 `b_start`/critic 的数据源。
   * ⇒ 本流程第 1 轮先用它做 WM 保鲜；B 线（VLA 驱动）仍需 `closed_loop_vla_supermimic.py`（§7）。

## 0.3 λ 到底管什么（**Isaac v3 口径 = token 不做 BC**）

你记的没错：**当前 IsaacLab 那条线（S3 v3）不做 token BC**——
`docs/S3_END_TO_END_PLAN_20261028.md:214` 写死了 `--lambda-bc 0`，
`docs/S3_END_TO_END_V3_PLAN_20261026.md:223` 的配方是
`--lambda-bc-token 0 --lambda-hand 0.2 --lambda-r 0.3`。

```
L = λ_bc  · FMBC( head(x) → token )      ← λ_bc = 0（token **不**用标签监督）
  + λ_hand· FMBC( hand 维, hand-only 掩码 mask[...,64:78]=1 )   ← 0.2（手部才监督）
  + λ_r   · ( −R̂ 想象反传 ) + 锚定/信任域/critic               ← 0.3
```

| 名字 | 含义 | 本仓默认 | 为什么 |
|---|---|---|---|
| **λ_bc** | token 的**监督**（标签）项 | **0** | 与 Isaac v3 一致：token 只由**奖励**监督；`>0` 只剩诊断用途（纯 BC 对照臂） |
| **λ_hand** | 手部辅助项（hand-only 掩码） | **0.2** | 手部没有奖励信号，只能靠标签监督 |
| **λ_r** | **奖励**项 | **0.3** | 想象闭环反传 |
| λ_a / anchor / trust | 动作 hinge、token 锚定、信任域 | 0 / 1e-3 / (0.2,1.0) | 防止奖励把 token/动作推出分布 |

### 那"域"是什么？

我们**不用侧车**，token 空间是 **数据集自带的 label/hold 域**
（`g1_motion_token_64` == parquet `action.motion_token`，逐位相等）。
但"在这个域里"**不等于**"要用标签去训"：VLA 底模（`checkpoint-8000` / 318 的 20k）
本来就是**在这个域里做的 SFT**，所以 J1 起点就已经在正确的域里，
`λ_bc=0` 不会让它跑出域——奖励这一项足以在域内挪动它（再加上 token 锚定/信任域兜底）。

> 反过来，`λ_bc=1` 会把它往"录制标签"上拽，而录制标签在闭环里本来就会被 VLA 自己的
> 预测取代 —— 这就是 Isaac 那轮把 token BC 关掉的原因。

## 0.4 B 线回放采集（最后补上的一块）

之前只有"WBC 参考跟踪"的向量化采集（适合 WM 保鲜，但**没有 VLA 漂移**，不能当 b_start/critic 数据）。
现在 `closed_loop_vla_supermimic.py` 加了**回放模式**，一行命令就能产出"真库行号 + 真奖励"的 B 线数据：

```bash
python3 scripts/closed_loop_vla_supermimic.py \
  --vla-ckpt <VLA or 合并后的 ckpt> --joint-ckpt <W1 ckpt> \
  --task task1 --steps 600 --device cuda:0 --no-video \
  --replay-lib data/library/ft318_compact.pt --start-frame <库行号> \
  --arm gt --tokens data/library/ft318_tokens.pt \
  --dump runs/bline/gt_f<行号>.pt
```

| 参数 | 含义 |
|---|---|
| `--replay-lib` | 给参考库 ⇒ 进回放模式：root+29 关节+qd **一起**从库行读（= 4090 交接的 `SM_INIT_FROM_LIB=1`） |
| `--start-frame` | 起始**库行号**（随机起点就多跑几条不同行号） |
| `--arm gt` | 用**录制 token** 驱动（不加载 VLA，快；用来验链路 + 产干净参考数据） |
| `--arm vla` | 真实 VLA 推理（采"VLA 把机器人跑偏"的漂移数据；需要 GPU + EgoView 渲染） |
| `--dump` | 落 `states/next_states/actions/tokens/q_target_applied/frames/rewards` |

**实测（2026-10-28，4090，20 tick）**：
```
frames  [101, 102, ..., 120]        ← 真库行号
rewards [0.0565, -0.0979, 0.0473, ...]
meta.frames_source = library_row     ← BStartSource 的 strict 检查认这个
meta.cfg.SM_TOKEN_SOURCE = online
```
⇒ `--start-from-b 'runs/bline/gt_f*.pt'` 可以直接用（strict 门可过）。

## 0.5 最终口径（2026-10-29 定稿，**与下面的命令一致**）

| 项 | 值 | 为什么 |
|---|---|---|
| **WBC** | **解冻**（`--train-wbc --wbc-lr 1e-6`，`TRAIN_WBC=1`） | 10-29 口径：执行器也要跟着奖励走 |
| 门禁 | 解冻后 **跳过 `B_freeze`**（否则必然 FAIL）：`A_codec / C_burnin / C2 / D_fm_bc` | 自动联动 |
| **trust_center** | **`self`**（= 采集时策略真正下发的 token） | `label` 属另一个执行器；`prev` 只保证"别离开上一轮"。Isaac 侧目前只有 label/base/prev，`self` 是本仓新加 |
| **token 锚定** | **关**（`--anchor-weight 0`） | 数值上比奖励项小 2–3 个量级，挡不住梯度；容器换成 hinge 信任域 |
| **token 信任域** | 开（`--trust-eps 0.2 --trust-weight 1.0`，中心=self） | 无量纲 hinge |
| 动作信任域 | 开（`--act-eps 0.6 --act-weight 1.0 --act-hinge-cap 2`） | 与 4090 锁定一致 |
| **lambda_hand** | **0（关）** | 10-29 口径 |
| lambda_bc / lambda_r | 0 / 0.3 | 与 4090 锁定一致 |
| **B 线采集** | **`arm=vla`**（**round 1 也用 vla**）；`4 任务 × 20 集 × 600 tick = 48,000 步` | 只有 vla 臂的 dump 里 `tokens` 才是"真正下发的 token"，`self` 中心才有意义 |

**采集规模/耗时（实测组件）**：单条 ≈ **35–45 s**（启动 25 s + 600 tick 3.2 s + 30 次 VLA 前向）

| 规模 | 步数 | 单进程 | 2 并发 | 4 并发 |
|---|---|---|---|---|
| 4×10×600 | 24k | ~27 min | ~14 min | ~8 min |
| **4×20×600（默认）** | **48k** | ~53 min | **~27 min** | ~15 min |
| 6×20×600 | 72k | ~80 min | ~40 min | ~22 min |

> 想再多就调 `BLINE_TASKS/BLINE_EPISODES`（H200 上 2 卡并发，成本很低）。

## 0.5.5 开跑前 30 秒：GL 自检（**换机器/新镜像第一件事**）

闭环要渲 ego 视角喂 VLA。渲染后端每台机器状态都不一样，已经踩过 3 次（§5.2 第 4/5/7 条），
所以开跑前先探一遍——比跑到 B 线才发现 80 集全挂省 20 分钟：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

# ① 三个后端逐个真渲一帧（纯 CPU，不加载 VLA，约 20 s）
python3 scripts/probe_gl.py

# ② 系统里到底有没有这两个库
ldconfig -p | grep -E "libEGL|libOSMesa" || echo "[none] libEGL 和 libOSMesa 都没有"
```

`probe_gl.py` 会打印它用的 mujoco 路径，必须是 `third_party/gr00t_deps/mujoco`（和闭环一致；
脚本自己也把它插到 `sys.path` 最前了）。

| ① 的输出 | 含义 | 你要做的 |
|---|---|---|
| `[ok] egl render=(64, 64, 3)` | 最理想（4090 那台就是这样） | 什么都不用做 → §0.6 |
| `[fail] egl …` + `[ok] osmesa` | 缺 libEGL，但有 OSMesa | 什么都不用做：闭环按 `egl → osmesa` 自动回落（日志打 `[gl] backend=osmesa`）；想强制就加前缀 `SM_GL_BACKEND=osmesa` |
| 三个都 `[fail]` | 缺库 | 走 ③ 装库，再回 ① |
| 没 root / 装不了 / 无网 | — | 走 ④，B 线换机器采 |

```bash
# ③ 缺库时（需要 root。H200 上"eglQueryString 是 None"就是缺 libEGL）
sudo apt-get update
sudo apt-get install -y libegl1 libegl-mesa0 libosmesa6
python3 scripts/probe_gl.py                 # 装完再验；出现任意一个 [ok] 就够

# ③' apt 装不上、但驱动自带 libEGL 时（ldconfig 没收录）
find / -name "libEGL*.so*" 2>/dev/null | head
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
python3 scripts/probe_gl.py

# ④ plan B：本机完全渲不出来 ⇒ B 线换到 4090 那台采（实测 [ok] egl）
#    runs/ 是共享盘，dump 直接落在同一路径，H200 只做训练
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
BLINE_JOBS=2 BLINE_EPISODES=20 BLINE_TICKS=600 ROUNDS=1 REWARD_STEPS=20 \
  bash scripts/launch_j1_rounds.sh          # ← 在 4090 那台执行这一条
```

探完就往下走 §0.6 金丝雀。

## 0.6 金丝雀：先跑 10 分钟（**必做**，过了再跑 §1）

新口径（WBC 解冻 + hand 0 + anchor 0 + `trust_center=self` + `arm=vla` 采集）**一次都还没跑过**。
直接上 6 轮，万一第一步就崩，你会白等 27 分钟采集。所以先跑这个：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
ROUNDS=1 REWARD_STEPS=20 NGPU=2 GPUS=0,1 \
BLINE_EPISODES=2 BLINE_TICKS=600 \
  bash scripts/launch_j1_rounds.sh
```

规模：采集 `4 任务 × 2 集 × 600 tick = 8 条` ≈ **3 min**；训练 20 步 ≈ **1–2 min**；合计约 **6–8 min**。

### 必须看到这 4 行（缺哪行 → 看右边）

| # | 期望输出 | 含义 | 没看到说明 |
|---|---|---|---|
| 1 | `[launch] 采集 4 任务 x 2 集 x 600 tick (arm=vla, jobs=2)` 且同一行有 `WBC 1`、`batch=64 lora=64 H=20` | launcher 把口径传进去了 | 你直接跑了 `run_j1_rounds.sh`（没透传）或环境变量写错 |
| 2 | `[b-start] self_targets: N 条 dump 带真值` | `trust_center=self` 的前提成立 | `build_b_self_targets.py` 没跑成，或 dump 路径不匹配 → 会**降级成 label**（日志有 `[TRUST][self][warn]`） |
| 3 | `[AUDIT][v2] trainable=head_only ... wbc_trainable=8960000` | **WBC 真的解冻了** | 是 `0` ⇒ `TRAIN_WBC` 没传进训练（解冻后这个数必须 >0） |
| 4 | `[j1 1/20] {... "loss/reward_raw": ~1.7, "rollout/reward": ~0.09, "trust/rmse": 0.03–0.06, "loss/trust": 0.0, "loss/anchor": 0.0}` | 奖励分支在跑；两个 hinge（trust eps=0.2 / act eps=0.6）都在域内 ⇒ **loss 为 0 是正常的**，看 `trust/rmse` 有没有值 | `loss/anchor` >0 ⇒ anchor 没关；`trust/rmse` 缺失 ⇒ `--trust-center self` 没接上 `self_targets.pt`；`rollout/survival` <1 ⇒ 策略一上来就摔 |

再加一条通道检查（采集是否真在跑 vla 臂）：

```bash
grep -l "arm=vla" runs/j1_rounds/round_1/bline/log_*.log | wc -l    # 期望 = 8
tail -3 runs/j1_rounds/round_1/bline/log_task1_e1.log              # 应能看到 [sim]/[sm] 而非 "[vla] skipped"
```

### 金丝雀通过后

再跑 §1 的 6 轮正式命令（不用改任何东西，只是 `ROUNDS=6 REWARD_STEPS=800`）。
如果金丝雀就崩了，把 `logs/j1_rounds_*.log` 的尾部 + `round_1/` 下已生成的文件列表发我，我直接定位。

## 1. 一条命令（2×H200）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
ROUNDS=6 REWARD_STEPS=800 NGPU=2 GPUS=0,1 J1_BATCH=64 J1_LORA=64 ENVS=1024 STEPS=1000 \
  bash scripts/launch_j1_rounds.sh      # ← 必须用 launcher（setsid+nohup），别直接跑 run_j1_rounds.sh
# ↑ 默认 6 轮（≈4.5–7 h）；连续 2 轮闭环 mean reward 不涨就自动停，通常 3–5 轮就收工
# ⚠️ 必须用 launch_j1_rounds.sh：它 setsid+nohup 起来并做 20 s 自检。
#    直接跑 run_j1_rounds.sh 会在关终端/Ctrl-C 时被一起杀掉（2026-10-28 18:19 就这么丢过一轮）
# 想每轮训更久（≈1.5–2.3 h/轮，6 轮最多 ≈9–14 h）：把 REWARD_STEPS 换成 2000
# 只跑一轮 / 复现某轮：ROUNDS=1 bash scripts/run_j1_rounds.sh
```

采集规模：**`ENVS=1024 × STEPS=1000 = 102 万条`**（dump 1.2 GB，1–2 min）。
不是 4096×4000——那会产 **1638 万条 / 19 GB**，比整个 318 库（47 万帧）还多 35 倍，
而 WM 保鲜只需要 41 万样本、critic 只需 ~100 万行。

`run_j1_rounds.sh` **每一轮**依次做 7 件事：

| # | 步骤 | 产物 | H200 耗时 |
|---|---|---|---|
| 1 | **仿真采集**（WBC 在 mjlab 里推 1024 envs × 1000 tick，动作加噪；**不含 VLA**） | `round_r/transitions.pt`（≈1.3 GB / 102 万行） | 2–3 min |
| 2 | **WM 验收门**（拿第 1 步的真转移比 WM 积分：`qd_corr`/`qd_ratio`） | `round_r/gate_wm_qd.json` | 10–20 s |
| 3 | **B 线回放采集**（`arm=vla`：ego 相机 → VLA → token → WBC → MuJoCo，闭环） | `round_r/bline/b_*.pt` + `self_targets.pt` | **20–30 min**（`BLINE_EPISODES=20` ⇒ 4×20=**80 集** / 2 并发）；`BLINE_JOBS=4` ⇒ **10–15 min** |
| 4 | **门禁**（`A_codec`、`C_burnin`、`C2_token_contrast`、`D_fm_bc`；WBC 解冻后自动跳过 `B_freeze`） | `round_r/gates/gates.json` | 8–12 min（串行） |
| 5 | **J1 联合微调**（`REWARD_STEPS` 步，`λ_bc=0`/`λ_hand=0`/`λ_r=0.3`/`trust_center=self`/WBC 解冻） | `round_r/train/vla_final.pt` + `b_*.pt` | **25–35 min**（800 步，batch 64，2 卡） |
| 6 | **合并 head → 闭环评测** 600 tick | `ckpt_vla_r*/`、`cl_task1.json`、mp4 | 2–3 min |
| 7 | **判停 / 汇总**（连续 `PATIENCE` 轮不涨 >1% 就停） | `rounds.csv` | — |

**总计 ≈ 60–80 min/轮**（H200 双卡，800 步，默认参数），6 轮 ≈ 6–8 h；
按 §1.2 的高压参数 ≈ 40–55 min/轮，6 轮 ≈ 4–5.5 h。

> ⚠️ B 线是大头之一（80 集），**别只调 `REWARD_STEPS`**：一轮里它和门禁合起来 ≈ 一半墙钟。

## 1.1 暂停 / 恢复（**Ctrl-C 不够**，必须 pkill）

launcher 是 `setsid + nohup`：`^C` 只中断前台那个 shell，**子进程原地活着**。
2026-10-29 04:10 就是这样——你以为停了，`nvidia-smi` 里还挂着两个各 60.6 GB 的
`python3`（509744/509745），下一轮一启动就直接抢显存。

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
# ① 停调度层（launcher → run_j1_rounds）
pkill -f "[r]un_j1_rounds"; pkill -f "[l]aunch_j1_rounds"; sleep 2
# ② 停可能正在跑的 4 类重进程
pkill -f "[t]rain_joint_mjlab"; pkill -f "[c]losed_loop_vla_supermimic"
pkill -f "[c]ollect_mjlab";     pkill -f "[p]robe_wm_qd_corr"
pkill -f "[e]xport_vla_ckpt";   sleep 3
# ③ 自检：两条都应该是空/小数字
ps -eo pid,cmd | grep -E "[r]un_j1_rounds|[t]rain_joint_mjlab|[c]losed_loop|[c]ollect_mjlab" | head
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

- ③ 还看到 60 GB 但 ② 之后进程列表是空的 ⇒ 用 `kill -9 <pid>`（pid 从 `nvidia-smi` 的 Processes 表拿）。
- 中途按过 `^C` 的，**一定要先跑 ① ②**，否则会同时跑两份训练。

恢复分三种：

| 情况 | 命令 |
|---|---|
| 本轮跑到一半崩了，`transitions.pt` 已在（省掉 2–3 min 采集 + WM 门） | `REUSE=1 ROUNDS=1 REWARD_STEPS=800 bash scripts/launch_j1_rounds.sh` |
| 只有 B 线要重跑（`[bline] 0 条`，见 §5.2） | `REUSE=1 ROUNDS=1 REWARD_STEPS=20 BLINE_EPISODES=2 BLINE_TICKS=600 bash scripts/launch_j1_rounds.sh` |
| 第 1 轮已产出 `ckpt_vla_r1`，接着跑第 2..6 轮 | `START_ROUND=2 ROUNDS=5 VLA_CKPT=runs/j1_rounds/ckpt_vla_r1 bash scripts/launch_j1_rounds.sh` |

## 1.2 H200 高压配置（显存用满 + 墙钟更短）

默认那套是按 4090 的安全值写的（batch 64 ≈ 60 GB/卡，143 GB 的 H200 只用了 4 成）。H200 上有余量：

```bash
ROUNDS=6 REWARD_STEPS=800 NGPU=2 GPUS=0,1 \
J1_BATCH=128 J1_LORA=64 J1_H=20 \
ENVS=2048 STEPS=1000 \
BLINE_JOBS=4 BLINE_EPISODES=20 BLINE_TICKS=600 \
WM_REFRESH_EVERY=500 WM_REFRESH_STEPS=100 \
  bash scripts/launch_j1_rounds.sh
```

| knob | 默认 | 高压 | 为什么 | 出问题就退 |
|---|---|---|---|---|
| `J1_BATCH` | 64（60 GB/卡） | **128**（≈110 GB/卡） | 训练占 60% 墙钟；batch 翻倍才能把 2B+1.6B 的 MFU 拉起来 | `J1_BATCH=96`（OOM 时先减这个） |
| `ENVS` | 1024 | **2048** | 采集并行度翻倍，102→204 万条（dump 2.5 GB） | `ENVS=1536` |
| `BLINE_JOBS` | 2 | **4** | 80 集从 40 波变 20 波（≈27 min → ≈13 min）；每卡 2 个闭环进程 ≈ 40 GB | `BLINE_JOBS=3` |
| `BLINE_STARTS_MODE` | `auto` | **`auto`**（保持） | 起点按任务自己的库区间取（§5.2 第 6 条）；改成 `explicit` 会退回"用全库固定行号给所有任务"，会跨任务错配 | 别动 |
| `WM_REFRESH_EVERY/STEPS` | 500/200 | **500/100** | 保鲜由 `step % every == 0` 触发（step 从 1 数）：800 步里 `every=500` 只触发 1 次，把每次的步数从 200 降到 100 ⇒ 墙钟省 ~12%。**别写 `every=800`**：那只会在第 800 步（最后一步）触发，等于白做 | WM 门 `qd_corr` < 0.95 ⇒ 回 500/200 |

预计：**6 轮从 6–8 h 压到 4–5.5 h**（每轮 ≈40–55 min）。

还能再快、但要改代码/装包（**我都没动**，要就说话）：

| 项 | 现价 | 能省 |
|---|---|---|
| 门禁 4 道**串行**（每道都重载 2B+1.6B） | 8–12 min/轮 | 拆到 2 张卡并行 ⇒ **省 4–6 min/轮**（加 `GATES_JOBS=2`） |
| `flash_attn is not installed` → sdpa | backbone 是大头 | 装 flash-attn 通常再快 **20–40%**（要在 H200 上编译，属环境改动） |


---

## 1.3 一条龙（停机 → GL 自检 → 金丝雀 → 6 轮，**复制粘贴即可**）

从"上一轮还在跑/刚被 Ctrl-C"到"正式 6 轮开跑"，按顺序贴这 4 段：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

# ── ① 停机（Ctrl-C 杀不掉 setsid 的进程，必须 pkill）──────────────────
pkill -f "[r]un_j1_rounds"; pkill -f "[l]aunch_j1_rounds"; sleep 2
pkill -f "[t]rain_joint_mjlab"; pkill -f "[c]losed_loop_vla_supermimic"
pkill -f "[c]ollect_mjlab"; pkill -f "[p]robe_wm_qd_corr"; pkill -f "[e]xport_vla_ckpt"; sleep 3
ps -eo pid,cmd | grep -E "[r]un_j1_rounds|[t]rain_joint_mjlab|[c]losed_loop|[c]ollect_mjlab" | head
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # 两行都该是空的/小数字

# ── ② GL 自检（必须有一个 [ok]；否则按 §0.5.5 装库）────────────────────
python3 scripts/probe_gl.py
ldconfig -p | grep -E "libEGL|libOSMesa" || echo "[none] 两个库都没有"

# ── ③ 金丝雀（复用已采的真转移，只重跑 B 线→门禁→20 步；约 5–8 min）────
rm -rf runs/j1_rounds/round_1/bline          # 清掉上一轮 0 条的残留日志/dump
REUSE=1 ROUNDS=1 REWARD_STEPS=20 BLINE_EPISODES=2 BLINE_TICKS=600 \
  bash scripts/launch_j1_rounds.sh
# 必须看到： [gl] backend=…  ·  [bline] taskN 起点(2) 四条  ·  [bline] 8 条  ·  [j1 …] "loss/trust" > 0

# ── ④ 金丝雀过了 → 正式 6 轮（高压）──────────────────────────────────
ROUNDS=6 REWARD_STEPS=800 NGPU=2 GPUS=0,1 \
J1_BATCH=128 J1_LORA=64 J1_H=20 ENVS=2048 STEPS=1000 \
BLINE_JOBS=4 BLINE_EPISODES=20 BLINE_TICKS=600 \
WM_REFRESH_EVERY=500 WM_REFRESH_STEPS=100 \
  bash scripts/launch_j1_rounds.sh
```

③ 里任何一条不对就停（`pkill -f "[r]un_j1_rounds"`），把 `logs/j1_rounds_*.log` 尾部和
`runs/j1_rounds/round_1/bline/log_task1_e1.log` 发我，此时只损失 5–8 分钟。

## 2. 每一步的判定线（不过就停）

| 步 | 判据 | 期望值（W1 it7000 实测） |
|---|---|---|
| 1 采集 | 行数 ≈ `num_envs × min(steps, capacity)` | 4096×4000 = **16.4M 行** |
| 2 WM 门 | `qd_corr ≥ 0.9` 且 `qd_ratio ∈ [0.9,1.1]` | **0.969 / 1.029 PASS**（4090 上已实测） |
| 3 门禁 | 5 项全 PASS | codec/freeze 已实测 PASS；其余 H200 上验 |
| 4 联合微调 | `rollout/reward` 上升、`state_term` 下降、token 不漂（锚定/信任域兜底） | 参照 `eval_curve.jsonl` |
| 5 闭环 | 不摔（tilt < 60°、z > 0.45） | 与 W1 同起点比 |

---

## 2.1 时间预算（H200 双卡，一轮 = 7 步）

| # | 步骤 | 耗时 | 依据 |
|---|---|---|---|
| 1 | 采集 1024 envs × 1000 步（102 万条） | **≈1 min** | **实测**：1024×300 = **25 s**（含 10 s 建环境，线性外推） |
| 2 | WM 验收门 | ≈20 s | 实测 |
| 3 | B 线回放 5 起点 × 600 tick | **≈3 min** | 实测 20 tick 的启动≈25 s + 600 tick ≈10 s ⇒ ~35 s/条 |
| 4 | 五道门禁 | ≈2 min | 4090 上 codec+freeze 秒级；burn-in/FM-BC 稍重 |
| 5 | critic 拟合（102 万行，12 epoch） | ≈2 min | 小 MLP + 1M×71 预处理 |
| 6 | **联合微调**（`REWARD_STEPS` 步） | **每步 3–4.5 s** → 800 步 **35–60 min**；2000 步 **1.5–2.3 h** | 4090 实测 3.9 s/步（batch16/H16/λ_r=0.3）；H200 batch64/H20 约 ×1.3–1.5 再 ÷2 |
| 7 | 合并 head（写 7 GB）+ 闭环评测 600 tick | ≈3 min | — |

**单轮合计**：
* `REWARD_STEPS=800` → **≈45–70 min**
* `REWARD_STEPS=2000` → **≈1.5–2.3 h**

**`ROUNDS=6` 总计（含早停）**：

| 配置 | 上限（跑满 6 轮） | 实际上大概率 |
|---|---|---|
| 800 步/轮 | ≈4.5–7 h | **3–5 轮 ⇒ ≈2.5–5.5 h** |
| 2000 步/轮 | ≈9–14 h | 3–4 轮 ⇒ ≈4.5–9 h |

> 早停规则（`PATIENCE=2`，`MIN_GAIN=0.01`）：每轮闭环在**同一任务、同一库起点**上跑 600 tick，
> 取 dump 里逐步奖励的**均值**；相对历史最佳提升 **>1%** 才算"涨"，连续 2 轮不涨就停。
> 结果写 `runs/j1_rounds/rounds.csv`（`round,mean_reward,fell,max_tilt_deg,final_z`）。

> **第 1 轮请先用 800 步**，跑完看 `round_1/train/train_metrics.jsonl` 里的 `s_per_step`：
> 时间 = `步数 × s_per_step`，再决定后面几轮要不要加到 2000。
> 采集/门禁/评测加起来只有 ~8 min，**时间几乎全在联合微调上**。

## 3. 关键参数（H200 值 / 4090 对照）

| 参数 | H200 | 4090 | 说明 |
|---|---|---|---|
| `NGPU` | **2** | 1 或 2 | 2 = torchrun + 梯度同步（已实测两 rank 参数逐位相同） |
| `J1_BATCH` | **64**（每卡，等效 128） | **16** | 显存：batch16=32.5GB；**batch32 在 4090 直接 OOM** |
| `J1_LORA` | **64**（67.1M / 3.98%） | 32 或 16 | rank128=134M 也可，代价是显存 |
| `ENVS` | **1024** | 512 | **采集**只需覆盖足够状态（1024×1000=102 万条）；4096×4000 会产 1640 万条、dump 19 GB，纯浪费 |
| `STEPS` | **1000** | 1000 | 每 env 推进的控制 tick 数（**并行推进 1000 次**，每次所有 env 同时动；不是串行 4000 步）|
| `CAP` | **1000** | 1000 | 环形缓冲长度；能存住的转移数 ≈ `ENVS×min(STEPS,CAP)` |
| `REWARD_STEPS` | 2000 | 1000 | 阶段 B（每轮不必 2000，300–1000 也可） |

> 显存提示：H200(141GB) 上 `J1_BATCH=64` + `J1_LORA=64` 峰值约 60–70 GB；
> 若想更大就 `J1_BATCH=128`（约 120 GB，接近上限）或开 grad-accum。

---

## 4. 循环怎么转（第 2 轮起）

`launch_j1_rounds.sh` 默认一口气跑 `ROUNDS` 轮：每轮第 1 步用**上一轮导出的新 head**
（`rounds.csv` 里记着 `ckpt_vla_r<k>`）重新采集，因为 VLA head 变了 ⇒ 闭环状态分布也变了。
所以正常情况**什么都不用做**，它自己会转。

只有"中断后要接着跑"才手动指定，用 `START_ROUND`（目录号继续，不会覆盖 `round_1`）：

```bash
# 第 1 轮已产出 runs/j1_rounds/ckpt_vla_r1，接着跑第 2..6 轮
START_ROUND=2 ROUNDS=5 VLA_CKPT=runs/j1_rounds/ckpt_vla_r1 \
J1_BATCH=128 ENVS=2048 BLINE_JOBS=4 \
  bash scripts/launch_j1_rounds.sh
```

- `VLA_CKPT` 必须指**上一轮导出的整 ckpt**（`ckpt_vla_r*`，已把 LoRA 合并进 base），
  不是 `round_*/train/vla_final.pt`（那只是 LoRA 增量，单用加载不了）。
- 同一轮重跑（不换 VLA）：加 `REUSE=1` 复用该轮 `transitions.pt`。
- `W1_CKPT` **不动**（W1 是最好的 WBC+WM，`ckpt_0014000`）；只有换参考库（重训 W1）才换。
- 停止条件：连续 `PATIENCE`（默认 2）轮闭环 mean reward 提升 < `MIN_GAIN`（默认 1%）就自动停。

---

## 5. 想单独跑某一步

```bash
# 只采集（不训练）
cd third_party/SuperMimic-Reproduced && env -u PYTHONPATH .venv/bin/python scripts/collect_mjlab.py \
  --ckpt logs/wbc_ft318_mjlab/ckpt_0007000.pt \
  --library ../../data/library/ft318_compact.pt --tokens ../../data/library/ft318_tokens.pt \
  --num-envs 4096 --steps 4000 --replay-capacity 4000 \
  --out ../../runs/wm_real/mjlab_transitions.pt

# 只跑 WM 门
python3 scripts/probe_wm_qd_corr.py --ckpt <W1 ckpt> --motion data/npz/g1_tasks \
  --tokens data/library/ft318_tokens.pt --transitions runs/wm_real/mjlab_transitions.pt

# 只跑门禁
python3 scripts/train_joint_mjlab.py --wbc-ckpt <W1 ckpt> --motion data/npz/g1_tasks \
  --tokens data/library/ft318_tokens.pt --vlm-ckpt <VLA> --vlm-path groot_repro/nvidia/Cosmos-Reason2-2B-weights \
  --out runs/joint_mjlab_ft318 --gate-only

# 复用已有采集（跳过第 1 步）
SKIP_COLLECT=1 bash scripts/run_j1_pipeline.sh
```

---

## 5.1 2026-10-28 晚：端到端试跑发现并修掉的 6 个真 bug

本机（4090，单卡）用 `SMOKE=1` 跑完整 7 步，一路上撞到并修掉：

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | 门禁 `burn-in` OOM（45.7/46 GB，即使中间 gc+empty_cache） | freeze 门之后 PyTorch 仍占 45 GB，同进程无法回收 | 门禁改成**一道门一个进程**（`--gate-subset`）；并加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 2 | 4090 上仍会 OOM（尤其 `D_fm_bc`） | 2B backbone + 1.6B head 在 46 GB 卡上就是紧 | **按显存自适应**：`<100 GB ⇒ GATES=auto`（OOM 记 `gates_skipped.txt` 并继续）；H200 默认 `strict`（不过就停） |
| 3 | `RuntimeError: Tensors must have same number of dimensions: got 2 and 3` | `b_start` 把 trackerlab 的 6 元组喂给 `bundle.mdp.obs`，元组里 `R` 是 3×3 | `b_start` 改用 **`bundle.obs_from_states(flat, a, xi, frames)`**（两个 bundle 都实现） |
| 4 | `AttributeError: 'function' object has no attribute 'device'` | `motion.ref.device`：mjlab 的 `library.ref` 是**方法** | 设备取 `motion.device`（方法/张量两种布局都兼容） |
| 5 | `AttributeError: 'ReferenceFrame' object has no attribute 'f'` | `v3._ref_states` 用了 trackerlab 的 `FrameRef.f` + `motion.ref` 张量 | 新增 `bundle.ref_state_at(frames)`，`_ref_states` 走它（老路径保留兜底） |
| 6 | `export_vla_ckpt` 报 `432/432 tensors absent from the base index` → **LoRA 训完的 head 无法产出可部署 ckpt** | LoRA 的 `lora_A/B` 在 base ckpt 里不存在，必须**合并**进 `xxx.weight` | `export_vla_ckpt.py` 加 **`W ← W + B@A`** 合并（默认 `alpha=rank ⇒ scale=1`）、missing 检查排除 LoRA 键、verify 校验合并后权重 |

**实测确认（4090，SMOKE）**：
```
WM 门          qd_corr 0.9707 / ratio 1.0282  PASS
A_codec        PASS
B_freeze       PASS（随后进程在 teardown OOM，auto 模式记 skipped 继续）
C_burnin       PASS  start_xy_err=0.0726m survival 1.0
C2_contrast    PASS  wbc-space contrast +20.0%（t=36.6）
训练 4 步       rollout/reward 0.0871 · hand_fm 0.185 · anchor 0.0198 · act/rmse 0.094
LoRA 合并      216 个 base 权重 · base vs merged max|Δ| = 9.0e-05 · finite
```

> 另外两条操作教训（写进脚本了）：
> * `pkill -f run_j1_rounds` 会**匹配到自己的命令行**把当前 shell 杀掉 → 用 `pkill -f "[r]un_j1_rounds"`；
> * `launch_j1_rounds.sh` 现在会把 `SMOKE/GATES/PATIENCE/MIN_GAIN/TASK` 一起透传（否则 launcher 的 SMOKE 传不到内层）。

## 5.2 2026-10-29 凌晨：真机试跑撞到的 7 个 bug（**都已修/已定位，命令一个字没改**）

你 03:42 那条金丝雀命令报的错就是第 1 条；顺着往下又挖出 3 条，全部会**直接卡死正式 6 轮**
（前两条当场死，后两条是采集静默 0 条）。第 6 条是**语义级**的，不报错但会污染 B 线数据。
改动落在 `scripts/launch_j1_rounds.sh`、`run_j1_rounds.sh`、`closed_loop_vla_supermimic.py`，
并新增 `scripts/bline_starts.py`。

| # | 现象（原文） | 根因 | 修法 |
|---|---|---|---|
| 1 | `scripts/launch_j1_rounds.sh: line 47: BLINE_TASKS: unbound variable` | launcher 开头 `set -euo pipefail`；`BLINE_TASKS/…` 只出现在 `CMD=(env X="${X:-def}" …)` 里，那只影响**子进程环境**，本 shell 里仍是未定义 → 第 47 行进度 echo 一读就退出 | 所有 knob 先在 launcher 里落成 shell 变量，再塞进 `CMD` |
| 2 | `[[: 49140⏎0: syntax error in expression (error token is "0")`（连报 4 行后，静默按 `GATES=strict`+`batch 64` 开跑） | `VRAM_MB=$(nvidia-smi … \| head -1 \|\| echo 0)`：`pipefail` 下 nvidia-smi 写第 2–4 张卡收 SIGPIPE(141) ⇒ 管道判失败 ⇒ `\|\| echo 0` 再追一行 ⇒ 变量变 `"49140\n0"` ⇒ 所有 `[[ -gt ]]` 语法错、显存自适应全废 | 全量读入再切首行 + 去非数字；并加显存自适应 `J1_BATCH 64→16 / ENVS 1024→512`（<100 GB 时） |
| 3 | B 线静默 `[bline] 0 条`，每集日志尾部 `RuntimeError: Cannot download the VLM backbone 'nvidia/Cosmos-Reason2-2B' … gated` | ① `third_party/gr00t_deps/sitecustomize.py` 在**解释器启动阶段**就 import transformers → huggingface_hub，`HF_HUB_CACHE` 当场定死成 `~/.cache`，python 里 `os.environ.setdefault("HF_HOME", …)` 是死代码；② 即便 cache 里有那 4.88 GB 权重，ckpt 的 `config.json` / `processor_config.json` 写死 gated repo id，tokenizer 路径照样 HEAD hub → 401 | ① `run_j1_rounds.sh` 在**父进程**导出 `HF_HOME/HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE`；② 闭环新增 `_offline_vla_view()`：给 ckpt 建 symlink 视图，只把 `model_name` 换成本地 `groot_repro/nvidia/Cosmos-Reason2-2B-weights` |
| 4 | B 线 `AttributeError: module 'mujoco.rendering.classic.gl_context' has no attribute 'GLContext'` | 脚本全局 `export MUJOCO_GL=disable`，而闭环**必须**渲 ego 视角喂 VLA，`setdefault("MUJOCO_GL","egl")` 顶不住已存在的 `disable` | 闭环里把 `disable`/空 强制成 `egl` |
| 5 | 改完 4 之后 B 线换一种死法：`AttributeError: 'NoneType' object has no attribute 'eglQueryString'`（8 集全挂） | PyOpenGL 3.1.10 在 linux 上的平台选择是「**`$DISPLAY` 存在就选 GLX**」，GLX 平台没有 EGL 符号；我这边 `nohup` 起的进程没有 `DISPLAY`（选 EGL、通了），你终端里有 `DISPLAY` ⇒ 选 GLX ⇒ `import mujoco` 当场炸 | 强制 `MUJOCO_GL=egl` 时同时 `PYOPENGL_PLATFORM=egl`（不再看 `DISPLAY`） |
| 6 | B 线**不报错**，但 task3/task4 的 episode 看着"很奇怪"（初始化就在跳别的动作、很快掉 reward） | `--start-frame` 是**全库行号**，跟 `--task` 无关；而库是 4 个任务首尾相接的：`task1 [0,183567)`、`task2 [183567,275453)`、`task3 [275453,336100)`、`task4 [336100,473876)`。旧代码给每个任务喂同一组 `BLINE_STARTS="100 2000 …"`，于是 **task3/task4 的起点 0 个落在自己区间内**（task2 也只有 4/20），机器人从 task_1 的动作中起跑，场景物体位置全不对 ⇒ reward 和 `self` token 都是噪声 | 新增 `scripts/bline_starts.py`：按 clip 名前缀算出该任务的 `[lo,hi)`，在区间内均匀取 `BLINE_EPISODES` 个起点（并保证一集 600 tick 不跨出任务边界）；`run_j1_rounds.sh` 默认 `BLINE_STARTS_MODE=auto`，旧行为用 `BLINE_STARTS_MODE=explicit` |
| 7 | H200 上 `MUJOCO_GL=egl` + `PYOPENGL_PLATFORM=egl` 都设对了，仍然 `'NoneType' object has no attribute 'eglQueryString'` | 注意报错细节：不是"选错 GLX"（那会报 `GLXPlatform has no attribute EGL`），而是 `.EGL` **存在但等于 None** ⇒ PyOpenGL 的平台已经切到 EGL，只是 **本机加载不到 `libEGL.so.1`**（H200 镜像里缺 `libegl1`，或 ldconfig 找不到） | 闭环改成**真试**：`_import_mujoco()` 依次 import `egl` → `osmesa`，哪个能 import 用哪个（`SM_GL_BACKEND=<b>` 可强制）；新增 `scripts/probe_gl.py`：在目标机上 20 s 报出每个后端能不能真渲出像素 |
**修完实测（`DISPLAY=:0` 复现你的 shell 条件，GPU3，`--arm vla --steps 20`）**：

```
[vla-view] model_name → …/groot_repro/nvidia/Cosmos-Reason2-2B-weights（离线）
[plan 1] tick=0 token=[-0.46,0.52]
[summary] {"ticks": 20, "plans": 1, "min_z": 0.746, "max_tilt_deg": 8.12, "fell": false}
[dump] 20 ticks -> runs/canary_b_test2.pt
```

> 采集已经跑完、只想重跑后面几步时：加 **`REUSE=1`**（`round_N/transitions.pt` 存在就跳过
> 第 1 步采集和 WM 门）。例如 B 线崩了重跑：
> `REUSE=1 ROUNDS=1 REWARD_STEPS=20 BLINE_EPISODES=2 BLINE_TICKS=600 bash scripts/launch_j1_rounds.sh`

> 排查顺序：先 grep `logs/j1_rounds_*.log` 有没有 `[bline] 0 条`（有 ⇒ 直接看
> `round_1/bline/log_task1_e1.log` 的 traceback 尾部），再看 `round_1/gates/gates.json`。

**GL 后端先探一遍（换机器/换镜像第一件事）**：

```bash
python3 scripts/probe_gl.py          # 20 s，CPU，不加载 VLA
# 期望：[ok] egl render=(64, 64, 3)  ← 或者 [ok] osmesa
ldconfig -p | grep -E "libEGL|libOSMesa"      # 两个都没有 ⇒ 装 libegl1 / libosmesa6
```

闭环现在按 `egl → osmesa` 自动试，日志会打 `[gl] backend=egl` 或
`[gl][warn] egl 不可用 → 试下一个`；想强制指定就用 `SM_GL_BACKEND=osmesa`。

## 6. 已验证 / 待验证

| 项 | 状态 |
|---|---|
| **端到端 7 步（SMOKE，4090 单卡）** | ✅ 已跑通到"合并 head"（§5.1 的 6 个 bug 全修完） |
| 门禁策略 | 小卡 `<100 GB` 自动 `auto`（OOM 记 skipped）；H200 `strict` |
| 采集（向量化，带帧号/奖励） | ✅ 4090：256 envs × 300 步 = **76,544 行**，~40 s |
| WM 门（正确口径） | ✅ **qd_corr 0.969 / ratio 1.029 → PASS** |
| 门禁 codec / freeze | ✅ PASS（4090） |
| 门禁 burn-in / token 对比度 / FM-BC | ⚠️ 4090 上 LoRA64 时 OOM（显存 45/46GB）；H200 无此问题，但**首次请在 H200 上看这 3 项** |
| BC 臂 / 奖励臂（单步） | ✅ 4090：1.15 s/step（BC）、3.5 s/step（奖励，H16） |
| 双卡梯度同步 | ✅ 两 rank 参数逐位相同 |
| 闭环评测（`closed_loop_vla_supermimic` + export） | ⚠️ 需 H200 上首次跑通（4090 同一套代码，未跑完整 600 tick） |

---

## 7. 出问题先看这三行

```bash
tail -50 runs/joint_mjlab_ft318/gates.json          # 门禁谁没过
cat  runs/gates/wm_qd_corr.json                     # WM 门的四个数
cat  runs/joint_mjlab_ft318/rl/summary.json         # 阶段 B 的最终指标
```

常见：
* `[tokens] ... space=label/hold` 没出现 → `--data.tokens` 没传或路径错（会退化成在线 future token，口径全错）
* 门禁 `freeze FAIL pi_identical=False` → `--wbc-ckpt` 不是 mjlab 布局（需要有 `policy`+`wm`）
* 阶段 B 的 `wm-refresh` 报 `TypeError: tuple indices...` → `wm_refresh` 的 `step_fn` 没接上（本仓已修）
