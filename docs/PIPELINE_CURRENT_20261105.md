# 当前唯一执行流程（2026-11-05 定稿）

> 本文档**取代** `FULL_LENGTH_5ROUND_PLAN_20261101.md` 与 `JOINT_MULTIROUND_PLAN_20261103.md`
> 中与下文冲突的部分（差异见 §7）。以本文为准；旧文档的"奖励公式/损失公式/契约门"等
> 仍然有效（见 §7 的对照表）。
>
> 一句话：**一轮长训 + 周期性用"策略真实转移"刷新 WM**，不再走"多轮 × 每轮采集"。

---

## 1. TL;DR

| 项 | 现在的做法 | 为什么 |
|---|---|---|
| 执行器 | **SONIC release decoder**（`HA_ALLOW_SONIC=1`）| 录制数据就是这个执行器产的；WBC 那条线口径不同 |
| WM 的角色 | **可微 plant**（梯度来源） | 没有它就没有 pathwise 梯度 |
| WM 的数据 | A'（录制+白噪声）**+ 策略真实转移（20–30%）** | 只靠 A' 时 WM 出现"伪方向"（实测转移率 0.09）|
| WM 何时更新 | **训练中周期性刷新**（`--wm-refresh-every`），不再"训一次冻到底" | 上游 SuperMimic 是每个 iteration 都刷 |
| 轮数 | **一轮**（可训 2000+ 次更新），不做 4–5 轮 | "多轮"只是为了刷新 WM + 课程；刷新可以在轮内做 |
| 起点 | 录制状态（带 ≤10 cm 漂移）| 与 A' 的漂移分布一致；参考按**帧号**对齐，语义自洽 |
| 奖励 | 想象 20 步执行 vs **理想命令流** 的漂移 | 与用户口径一致 |
| 交付指标 | 闭环 net 位移 / 示范进度 / Boxing 成功率（多 seed）| DoubleDesk 20 s 内做不完，不能用 success 当唯一判据 |

---

## 2. 端到端流程（7 步）

### ① 环境/执行器（固定）
```
HA_ALLOW_SONIC=1
SONIC_VLA_ACTION_FORMAT=latent64
SONIC_DIR=<release decoder onnx 目录>
HA_VLA_EXEC_STEPS=20        # 部署口径：每 20 步重新推理（2.5 Hz）
SUPERMIMIC_HA_ALIGN=1
```
- 执行器是 **SONIC release decoder**（`model_decoder.onnx`），不是 SuperMimic WBC。
- 闭环入口：`scripts/ha_isaac_closedloop.sh`（官方 `sim_eval_vla.py`）。
- 我们自己的封装：`scripts/r3_closedloop_ab.sh`（起 VLA 服务 + 跑闭环 + 体检）。

### ② 数据 A'（已完成，产物 `runs/a_prime_sonic_exc/`）
| 参数 | 值 |
|---|---|
| 任务 | 6 个（DoubleDesk/PPBox/Boxing/OpenDoor/SitSofa/VisionNavi）|
| 分片 | **50 个** `*_off*.pt`，合计 **529,131 条转移** |
| 噪声 | `--action-noise-std 0.15`，**`--perturb-hold 1`**（逐步白噪声）|
| leash | `--reset-max-drift 0.15` |
| token | `--exec-steps 1`（逐帧 token）|
| 驱动 | `scripts/run_a_prime_sonic_exc.sh`（两组并行）|
| 窗口 | `roll_windows_s1.pt`（k=20、**stride 1**、**92,785 个窗口**）|

采集成本实测：单 shard 1.9 min；50 shard（两组并行）≈ **37 min**。
若用 `PER=64`（每进程 64 env）可把 shard 数降到 ~15，墙钟 ≈13 min（下次记得改）。

### ③ WM 训练（离线阶段）
- 脚本：`scripts/pretrain_wm_scratch_ddp.py`（**4 卡 DDP，从零**）
- 关键实测：**本训练是 kernel-launch 瓶颈**（batch 1024→8192，每步耗时都是 ~200 ms）
  ⇒ 大 batch 近乎免费。单卡 batch 8192 比 4 卡 DDP（每卡 1024）还快 1.9 倍。
- 当前参数：
  ```
  --steps-single 12000 --steps-multi 4000 --k 20
  --batch 4096（每卡，等效 16384）--lr 3e-4（单步）--lr-multi 3e-5
  --inc-weight 2.0 --mag-weight 10.0
  单步数据 = 全量 52.9 万条真实转移；多步数据 = 92,785 个窗口
  ```
- 耗时：单步 206 s + 多步 1444 s ≈ **27 min**；产物 `runs/a_prime_sonic_exc/wm_scratch_ddp.pt`
- 损失：原 SuperMimic 逐维归一化平方误差（`ssm_wm_loss`）+ 增量损失 ×2 + 幅值 ×10

### ④ WM 门禁（三道，缺一不可）
| 门 | 脚本 | 判据 | 现状 |
|---|---|---|---|
| 多步保真 | `probe_wm_multistep.py` | 20 步 imag/real ratio ∈ [0.8,1.2] | ✅ 1.07 cm / corr 0.995 |
| 自由 headroom（参考项）| `probe_wm_headroom.py` | 只作警示 | ⚠ 89% |
| **交叉验证转移率（主判据）** | `probe_wm_headroom_transfer.py` | **≥0.5** | ❌ **0.09** |
| 通道可分辨 | `probe_wm_channels.py` | 误差÷单步变化 <0.5 才适合做奖励 | 位置 2.07 / qd 0.88 / angvel 0.55 |

**转移率的意思**：把数据按任务分成互不相交 A/B 两半各训一个 WM，用 `WM_A` 自由优化出动作，
看它在 `WM_B` 上是否同样变好。**≥0.5 = 方向是真实动力学；≤0.2 = 模型专属伪方向。**

### ⑤ 联合微调（一轮，可长训）
脚本：`scripts/run_joint_ft_round5.sh` → `scripts/launch_joint_ha_v2.sh`（DDP 4 卡）

| 参数 | 当前值 | 依据 |
|---|---|---|
| `H`（想象视界）| **20** | = 部署节奏（2.5 Hz、每 20 步重规划）|
| `START_SOURCE` / `START_MAX_DRIFT` | `recorded` / **0.10** | 起点=带 ≤10 cm 漂移的真实执行状态 |
| 奖励 | `huber_err` + `split_huber` + 死区 0.30 + **速度项** | 速度项是今天新增（原来 `huber_err` 分支没有速度）|
| `REWARD_DELTA_POS/VEL` | 0.05 / 0.05 | — |
| `REWARD_VEL_SCALE` | **1.0** | base 速度只有录制的 1/2，位置项 0.4 s 内看不见 |
| `LAMBDA_R` | **0.3**（建议）；曾用 5.0（激进 17×）| 旧计划口径 |
| act hinge | `ACT_EPS=0.6`、`ACT_WEIGHT=1.0`（建议）；曾用 0.3/0.2 | 旧计划口径 |
| token 信任域 | `TRUST_CENTER=base`、`TRUST_EPS=0.05`、`TRUST_WEIGHT=0.5` | v3 里 token 的**唯一**锚（`lambda_bc` 在 v3 不生效）|
| 手部 | **`--hand-from-base`**（服务端用 base 头出 64:78）| 手不参与微调，避免训坏（0.567 → 1.068）|
| `GAMMA` / critic | 0.95 / 关 | — |
| 步数 | **建议 2000**（原来 400 太短）| 有刷新兜底后可以长训 |

每一微调步在做什么（`diff_vla_wbc/v3_step.py::train_step`）：
1. 从 A' 采起点 `start_state` + 从命令库采 20 帧理想参考 + 10 帧观测历史；
2. VLA head（LoRA）采样 40 帧 token chunk（带噪声）；
3. `imagine_with_tokens`：20 步 → `SONIC decoder(token, obs994)` → 动作 → `WM(s,a)` → 下一步；
4. 每步算"想象执行 vs 理想"的漂移，`ret = Σ γ^t r_t`；
5. `loss = λ_r·(−ret) + act hinge + hand FM + token trust`，只反传进 LoRA；
6. 每 `--wm-refresh-every` 步刷新 WM（见 ⑥）。

### ⑥ WM 周期刷新（本轮新增，**代码已有钩子**）
`scripts/train_joint_v2.py:1352`：
```python
if args.v3 and args.wm_refresh_every and step % args.wm_refresh_every == 0:
    trainer.refresh_wm(_real, steps=args.wm_refresh_steps, lr=args.wm_refresh_lr, ...)
```
- **现状**：`wm_refresh_every=0`（冻结）且 `_real` 只有 A'。
- **目标**：`_real` = **A'（70–80%）+ 策略真实转移（20–30%）**，`wm_refresh_every=100`。
- 策略真实转移从哪来：每 ~100 次更新跑一次闭环（4 min），把 dump 转成
  `states/actions/next_states/frames`（格式见 `filtered_std.pt`）。
  - 转换要点：dump 的 `root_lin_vel_b/root_ang_vel_b` 是**机体系**，数据里是**世界系**
    （实测 `corr(dp/dt, slot36:39)=+0.953`）⇒ 必须按四元数旋转，否则喂错输入。
- **待做（唯一缺的工程）**：`dump → transitions` 转换器 + 外层循环脚本。

### ⑦ 评测（交付判据）
```bash
TAG=v5 HEAD=runs/joint_v5/vla_final.pt GPU=3 PORT=18100 MAX_STEPS=1000 SEED=0 \
  SRV_EXTRA="--hand-from-base" bash scripts/r3_closedloop_ab.sh
python3 scripts/eval_closedloop_progress.py --runs base=... v5=...
```
| 指标 | 说明 |
|---|---|
| `net`（净位移）| 站立/走动的粗判据 |
| `prog_end`（示范进度）| 沿示范路径的弧长进度，比 net 更贴任务 |
| Boxing `success` | **唯一能做成功的任务**（示范 9.7 s；DoubleDesk 20 s 内做不完）|
| `z_min` / 翻倒帧 | 安全 |
| `hand14_max` | 手是否闭合（阈值 1.70）|

判据：**多 seed（≥3，建议 4 卡并行 4 seed）** 下 `net`/`prog_end` 不差于 base，
且 Boxing 成功率 ≥ base。

---

## 3. 当前实测基线（写在墙上，防止再拿单 seed 说事）

| 配置 | 闭环 net | prog_end | 手 max | 备注 |
|---|---|---|---|---|
| base（checkpoint-8000 原样）| **1.000 ± 0.042**（3 seed）| **0.314 ± 0.040** | 1.826 | Boxing seed0 **成功**（648 步）|
| round-3（λ_bc=0, trust 0.15/0.05）| 0.111 | — | 0.149 | 崩 |
| round-4（trust 0.05/0.5 + base 手）| 1.031 ± 0.145 | 0.330 ± 0.094 | 1.068 | ≈ base，无显著提升；Boxing 0/2 |
| 真值 token 上界 | 1.223（600 步）| — | — | 证明执行器/接线是对的 |

**结论：目前没有任何一版联合微调在统计上超过 base。**

---

## 4. 已知的失败机理（今天查出来的，全部可复现）

1. **`state46` 布局拼错**（21/46 维）⇒ VLA 吐常量 token ⇒ 机器人站桩。
   已修（`legacy` 默认）+ 启动硬校验 + 契约门 3b。
2. ~~`--fm-steps 4` 把 chunk 采平~~ **（2026-11-05 撤回）**：那次 A/B 与布局修复混在一起；用已修布局重测 `fm=4` 走 1.165 m / `fm=20` 走 1.134 m，两者都正常，且 fm=4 快 2.4×。⇒ **`--fm-steps` 保持官方 4**。
3. **HA 的 64 维 vla_state 没有手** ⇒ 手 14 维恒 0 ⇒ 永不闭合。已修（`hands14` 透传 +
   `--hand-from-base`）。
4. **WM 的"伪方向"**：自由优化动作能把 20 步误差降 89%，但**跨模型只转移 8%**（转移率 0.09）；
   只改 2% 的动作就能降 82%。白噪声重采、从零训都**没能修掉**。
   ⇒ 用 WM 反传当奖励 = 优化一个假目标，这是"训练指标变好、真机变差"的机制。
5. **奖励缺速度维度**：base 速度只有录制的 1/2（0.050 vs 0.102 m/s），而 `huber_err` 分支
   原来只有位置/朝向项 ⇒ 对"走慢了"失明。已补速度项。

---

## 5. "合理吗"——逐条自评

| 决定 | 依据等级 | 说明 |
|---|---|---|
| 执行器用 SONIC decoder | **实测强** | 录制数据同源；闭环 1.0 m |
| A' 加白噪声（σ0.15/hold1）| 推理合理、效果待证 | 目的：让 `a` 不再只是"意图提示"；单独做没修掉伪方向 |
| 从零训 + DDP + 大 batch | **实测强**（吞吐）/ 效果待证 | batch 8192 吞吐 7.6×，DDP 4 卡约 2× |
| H=20（不拉长）| **实测强** | 与部署 2.5 Hz 一致；拉长 60 会用到"不存在的 token 冻结"|
| 起点 recorded / ≤10 cm | **实测强** | A' 漂移分布 p50 3.8 / p90 8.0 cm |
| 奖励 = 想象漂移 | 设计意图明确，但**依赖 WM 可信** | 现在 WM 不可信（转移率 0.09）⇒ 这是最大风险 |
| 周期刷新 WM（策略真实转移）| **上游做法 + 推理合理**，我们未验证 | 这是当前**唯一未验证的关键假设** |
| 一轮长训（2000 步）| 未验证 | 前提是刷新有效 |

**最大风险**：周期刷新能否把转移率从 0.09 拉起来。**验证成本 30 分钟**（用已有 8 份闭环
dump 混进 WM 重训 → 测转移率），**建议在开 2000 步长训之前先做这一步**。

---

## 6. 下一步（明确、可执行）

1. **[30 min，零采集]** 把已有 8 份闭环 dump 转成转移，混进 A'（20%）重训 WM
   （`finetune_wm_multistep --batch 8192 --steps 1500`）→ 测**转移率**。
   - ≥0.5 ⇒ 机理成立，进第 2 步；
   - 不动 ⇒ 停下，考虑"模型集成 + 不确定性惩罚"或"放弃 WM 反传"。
2. **[半天工程]** 写 `dump → transitions` 转换器 + 外层循环（每 100 步闭环一次 → 重建 mix）。
3. **[1.5 h]** 一轮 2000 步联合微调（参数按 §2⑤ 建议值）+ 4 卡并行 4 seed 验收。
4. 若仍不如 base：停止在 WM 路线上投时间，转为交付 base + 把"可迁移 WM"当独立课题。

---

## 7. 与两份旧文档的对照

| 旧文档条目 | 现在是否有效 | 说明 |
|---|---|---|
| **奖励公式**（`JOINT_MULTIROUND_PLAN` §2：`e_*`、huber-on-error、split、δ 退火）| ✅ 有效 | 实现与之一致；今天只**新增速度项**（δ_vel=0.05）|
| **总损失**（§3：`J_b=Σγ^t c_t r_t`、H=20、γ=1）| ⚠ 部分 | γ 我们实际用 **0.95**（不等于 1）|
| **critic**（warm start/重拟合）| ❌ 暂不用 | 我们 `CRITIC_ON=0` |
| **δ / λ 五轮退火**（5cm→2cm→5mm→2mm）| ❌ 作废 | 与"一轮长训 + 刷新"不兼容；如要课程，改成轮内按步退火 |
| **多轮架构 R1–R5、每轮采 B**（两份文档都是）| ❌ 作废 | 改为**轮内周期刷新**；"B 线"数据仍要，但改成 20 步短窗、在线采 |
| **A' = 名义回放 + L1/L2 + 多 seed**（`JOINT_MULTIROUND_PLAN` §4.2/4.3）| ✅ 有效 | 我们就是这么采的（加了白噪声与 leash）|
| **WM 三阶段 3000/3000/1500、A:B=1:1** | ⚠ 部分 | 我们：单步 12000 + 多步 4000（batch 大得多）；配比改成 A' 80% + 策略 20% |
| **"WM refresh 从 R2 起、每轮 10–20 集闭环"** | ⚠ 改动 | 提前到**轮内每 100 步**，用 20 步短窗而非整集 |
| **采集/评测工具、契约门、pin 配置** | ✅ 有效 | 全部保留 |
| **执行器 WBC(v6)** | ❌ 作废 | 改 SONIC release decoder（`docs/CLOSEDLOOP_INFERENCE_FIX_20261104.md`）|
| **全长评测、START_FRAME=0、32/48 对** | ⚠ 部分 | 评测入口仍用官方脚本；但 **DoubleDesk 20 s 内做不完**，成功率只在 Boxing/PPBox 上有意义 |

---

## 8. 相关文档

| 文档 | 内容 |
|---|---|
| `docs/WHY_FT_DEGRADED_20261104.md` | 根因链（含三次撤回）、WM 保真/headroom/转移率实验、四种改法 |
| `docs/CLOSEDLOOP_INFERENCE_FIX_20261104.md` | 布局/fm_steps/手部三处修复与验收 |
| `docs/HA_CLOSEDLOOP_PIN.md` | 闭环场景/资产/初始位姿的锁 |
| `configs/ha_closedloop_pin.json` | 上面那条的机器可读版本 |

---

## 9. 吞吐真相（2026-11-05，为什么"4 卡 DDP 还这么慢"）

### 9.1 先排除的两个错误答案

- **不是"没用满 4 卡"**：确实只用了 **3 卡**。GPU1 上 43.5 GB 是同机**其它容器**的三个进程
  （`nvidia-smi` 能看到 pid，但 `/proc/<pid>` 在本 PID namespace 里不存在 ⇒ 不是我们的、也杀不掉），
  且 GPU1 利用率一直 16–19% ⇒ 它在算，不是空转等人收。
- **不是"卡不够"**：耗时几乎正比于视界长度（H=10 → 2.64 s，H=20 → 4.06–5.11 s），
  说明瓶颈是**串行链长度**，不是并行算力。

### 9.2 真凶：每步是 launch 绑定，batch=1 等于让 GPU 空转

`J1_STEP_PROFILE=1` 的分段（每卡 batch=1，H=20）：

```
backbone=0.05s  fm_sample=0.22s  rollout=0.94s  bwd_opt=2.45s   （合计 ~4.2 s/次更新）
```

**batch 加大到 8 倍，耗时几乎不变**（实测，3 卡 DDP，全局窗口数 = 3 × 每卡 batch × accum）：

| 每卡 batch | accum | 每次更新的窗口数 | 每次更新耗时 | 有效吞吐（窗/s） |
|---|---|---|---|---|
| 1 | 4 | 12 | 4.16 s | 0.72 |
| 2 | 2 | 12 | 4.16 s | 1.44 |
| 4 | 1 | 12 | 3.97 s | 3.02 |
| 8 | 1 | 24 | 4.27 s | **5.62** |

显存 27 GB(batch1) → 33.7 GB(batch8)，48 GB 卡安全。
⇒ 这段训练是 **kernel 发射 / Python 逐层开销绑定**，不是 FLOP 绑定；
`batch=1 + accum=4` 的组合把吞吐浪费了约 **8 倍**。

**等价性**：`v3_step` 里 `(loss/accum).backward()`，loss 是 batch 内 mean，每个窗口的
rollout 完全独立（无跨窗耦合）⇒ 改 microbatch 切法**不改变数学**，只改变归约顺序。
所以现在用 **每卡 batch 8 × accum 1**（每次更新 24 窗）替代原 `batch 1 × accum 4 × 4 卡`（16 窗），
`s_per_step` 的口径也从"一个 microbatch"变成"一次优化器更新"。

### 9.3 顺手挖出的地雷：启动脚本在 `set -e` 下静默退出

`scripts/launch_joint_ha_v2.sh` 是 `set -euo pipefail`，而 `_ARGS` 里有一句

```bash
$([[ "$TERM_OFF" == "1" ]] && echo --term-off)
```

`TERM_OFF≠1` 时这个替换返回 1 ⇒ **整条数组赋值失败 ⇒ 进程零报错退出（rc=1）**，
日志停在一行无关的 `[deps_env] ...` 上。2026-11-05 一次 4 组对照全灭，还被误读成"跑得很快"。
已修为 `$( ... || true )`，并加了静态门 **`check_all_contracts.sh` 3c/4**
（launch/driver 里所有 `$( [[ ... ]] && ... )` 必须带 `|| true`），启动横幅也打出 `term_off=`。

### 9.4 顺手清掉的"永远红的门"

`check_all_contracts.sh` 默认分片目录原来是 `runs/a_prime_sonic`，它 **`z_min=0.238`**
（<0.25 的「没有摔倒」门 ⇒ 里面有倒地帧），而 round-5 任何一步都不用它。
一个每次都被忽略的红灯本身就是 bug 的来源 ⇒ 该目录已移入
`runs/_quarantine/`（含 README 与复修步骤），默认口径改成**现役两份语料**：

| 语料 | 有没有真实转移 | 没有摔倒 | 逐行漂移 p50 |
|---|---|---|---|
| `a_prime_sonic_exc`（WM 数据） | ✅ 63.54 | ✅ z_min=0.636 | 0.44 cm |
| `a_prime_sonic_leash`（起点采样） | ✅ 121.09 | ✅ z_min=0.561 | 0.36 cm |

### 9.5 现在这一轮（`runs/joint_v5`）的实际配置

```
每卡 batch 8 × accum 1 × 3 卡 = 24 窗/更新   STEPS=500（= 500 次更新、12,000 窗）
WM=wm_gain2.pt  TERM_OFF=1  START_MAX_DRIFT=0.05  --hand-from-base
实测 steady-state 4.43 s/更新 ⇒ 500 步 ≈ 37 分钟（原配置同等工作量约 2.6 小时，慢 4.2 倍）
```

判读口径不变：`rollout/survival≡1`、`act/rmse≈0`（hinge 未触发）、`trust/rmse 0.05–0.08`、
`grad/head_preclip 1–3`；之后仍是 **4 seed 闭环 A/B vs base**（`scripts/r3_closedloop_ab.sh`
+ `scripts/eval_closedloop_progress.py`），单 seed 一律不接受。

---

## 10. 「越联合微调越差」的重新归因：可能是更新次数，不是 WM（2026-11-05）

### 10.1 三轮并排看

| 轮次 | 闭环真实结果 | act/rmse(末) | act/frac_out | **优化器更新次数** | 想象 win 率 |
|---|---|---|---|---|---|
| round-3 | **0.111 m（崩）** | 1.463 | 0.90 | **1000** | **0.88（全场最好看）** |
| round-4 | 1.014 m ≈ base | 0.055 | 0.00 | **100** | 0.69 |
| round-5 | 待测 | 0.92 @step250 | 0.85 | 500 | 0.875 |

两条结论：

1. **想象指标是负信号**：崩得最惨的 round-3，恰恰是 eval 最好看的那一轮（win 0.88、想象回报
   单调改善）。所以「回报在涨 / win 率在涨」**不能**用来判成功，只能用来报警。
2. **漂移是"更新次数"的函数**：round-4 之所以没坏，很可能只是因为它 **400 steps / accum 4
   = 只有 100 次优化器更新**就停了；round-3 跑了 1000 次；round-5 跑到 250 次时已经 0.92。
   `act_eps=0.30 / trust_eps=0.05` 是**软墙**（权重 0.2 / 0.5），不是硬约束，撞穿要几百次更新。

⇒ 之前一直在怀疑「采集 / WM / 损失哪一环错了」，但按这个读法，**更可能只是训练超过了可信域**
（早期停止问题）。这跟"WM 学得不够好"是两件事，验收方式也完全不同。

### 10.2 两种解释目前不可区分，必须闭环

- **(A) 薅WM**：动作跑到冻结 WM 编出来的流形外，执行器跟不上 ⇒ 闭环崩；
- **(B) 正常代价**：round-5 **新加了速度项**（`REWARD_VEL_SCALE=1`，因为 base 速度只有录制的
  一半）⇒ 要改速度动作就必须动，0.9 的漂移正是"被奖励要求的"。

唯一判据是闭环。所以新增 **检查点闭环扫描**（`scripts/sweep_ckpt_closedloop.sh`）：
同一 seed、同一 `MAX_STEPS=1000`，把 `base` 与 `vla_000100/200/300/400/final` 一起跑，
用 `eval_closedloop_progress.py` 判读 net 位移 + 是否走在示范路径上 + 进度。
**若悬崖存在，它会在曲线上显形**——这也直接回答"最多能训多少次"。

### 10.3 新增的体检门：`scripts/check_ft_run.py`

把上面三条变成可执行判据（`--gate` 时出流形区 exit 3）：

```
[ft] joint_v5: steps=500 batch=8 accum=1 ⇒ **更新次数 500**
[ft] 动作漂移 act/rmse(近30均) = 0.806   frac_out = 0.79
[ft] ⚠ **动作已出可信域** …这一型与 round-3（闭环 0.111 m 崩）同构，不是与 round-4 同构
     **想象指标再好也不能宣布成功**：必须闭环 A/B，且优先测早期检查点
```

自检（对历史数据回放）：round-4 → `rc=0`（在域内）、round-3 → `rc=3`（判为崩前形态）。
门里同时报 `更新次数`（`steps / grad_accum`）——这一步最容易搞错，round-4 的 400 其实是 100。

### 10.4 顺手修掉的部署侧坑

`scripts/r3_closedloop_ab.sh` 里服务端卡号原来是**写死 `cuda:1`**，而本机 GPU1 被别的容器占了
43.5/48 GB ⇒ 2B 服务端必然放不下。改成 `SRV_GPU=${SRV_GPU:-$GPU}`（默认与 Isaac 同卡），
启动行也打出 `isaac_gpu/srv_gpu/seed/max_steps`，避免"两条臂卡号不同"这种口径漂移。
