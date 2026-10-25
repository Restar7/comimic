# Token 锚定损失（token anchor）：解决"越训越差"

> 2026-10-25。代码已实现，**尚未启动训练**（等指令）。

## 1. 为什么需要它（问题）
联合微调的 loss 里**没有任何 token 保真项**（只有 reward/state/smooth/safety/hand）。reward 在
1000–2000 步就到顶（0.0819 → 0.0816 → 0.0822），之后模型在 **reward 平坦区里随机漂移**：

| step | `token_mse`（对端口 token） | `rollout_reward` |
|---|---|---|
| 1000 | 0.0964 | +0.0819 |
| 2000 | **0.0773**（最好） | +0.0816 |
| 3000（vel=1.0） | 0.0925（↑20%） | +0.0822 |
| 3000（vel=0 那次，已归档） | 0.1080（↑40%） | +0.0841 |

mjlab 那轮之所以"越训越好"，是因为它的**标签空间≈执行器空间**（MSE 0.005），漂向 reward 时
token_mse 恰好也在降；我们这边两个空间差 16 倍（0.088），漂向 reward 就是漂离标签。

## 2. 改了什么（代码已就位）
| 文件 | 改动 |
|---|---|
| `diff_vla_wbc/train.py` | `JointTrainer(..., token_anchor=<callable>, token_anchor_weight=λ)`；`train_step` 里在 hand loss 之后加 `λ·MSE(tokens, anchor)`，并输出 `loss/token_anchor` |
| `scripts/train_joint.py` | 新增 `--token-anchor-weight`（默认 **0 = 关闭**，旧行为不变）；>0 时用 `RecordedTokenSource`（自动优先端口侧车 `data/port_tokens/ha_tokens/`）作为锚定目标；日志打印 `[anchor] ON: λ=… src=… port_dir=…` |
| `scripts/launch_joint_ha.sh` | 透传 `TOKEN_ANCHOR_WEIGHT`（默认 0） |
| `scripts/train_ppo.py` | （A3 的 PPO 臂暂未接入；它用 score-function，没有 token 梯度路径，先不加） |

实现要点：
- 锚定目标**不用 grad**（从 npz / 侧车读），只对 `tokens` 求梯度 → 直接影响 action head；
- 形状 `(B, 40, 64)`，取 `min(T)` 对齐；
- `λ` 与 reward 项同量级比较：起点 `token_mse ≈ 0.17`、`loss/reward ≈ -0.4` → **λ=1.0 时两项同量级**；
- 与原 `--recorded-tokens` 路径互不干扰（那条路径的 token 无梯度，锚定项自然为 0 梯度）。

## 3. 怎么用（**尚未执行**，等指令）
```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
RUN=joint_ha_anchor STEPS=2000 \
STATE_VEL_WEIGHT=1.0 TOKEN_ANCHOR_WEIGHT=1.0 PREFETCH_VLA=1 \
setsid nohup bash scripts/launch_joint_ha.sh > logs/joint_ha_anchor.launcher.log 2>&1 &
```
- 从**官方底模** `checkpoint-8000` 重训（不带 `RESUME_STEP`）；
- 2000 步（上一轮的平台点），`vel=1.0`（保留速度约束），预取开；
- 产物在 `runs/joint_ha_anchor/`，不覆盖已有 `runs/joint_ha_a1/`（那里有目前最好的 `vla_002000.pt`）。

## 4. 验收标准（跑起来后必须核对）
| 项 | 期望 |
|---|---|
| `loss/token_anchor` | 从 ~0.17 单调下降（1000 步 <0.10、2000 步 <0.05） |
| `token_mse`（固定 eval） | **单调下降**（不再出现 2000→3000 的反涨） |
| `rollout_reward` | ≥0.08（不能为锚定牺牲 reward） |
| kinematic | ≤0.006（oracle 0.0063） |
| 速度分项 `state_vel` | 保持地板 ~0.41–0.45 |

## 5. 风险与回退
- **风险**：λ 太大会把 token 拉向"标签/端口 token 的均值"，抑制 reward 探索 → reward 掉；
  太大时 `token_mse` 好看但跟踪变差。**先 λ=1.0**，若 reward <0.075 就降到 0.3。
- **回退**：`TOKEN_ANCHOR_WEIGHT=0` 即退化成当前行为（代码路径完全等价）。
- 与 `--state-vel-weight 0` 的组合**不建议**：那样既没速度约束、锚定又只在 token 上，
  等于两条约束都变弱；建议 `vel=1.0 + anchor=1.0` 一起用。

---

## 6. 续训 vs 从头重训：结论与理由（2026-10-25）

### 6.1 结论：**先续训（RESUME_STEP=2000 + anchor），不要从头重训**

| | A. 从头重训（0→2000，带 anchor） | **B. 续训（2000→3000/4000，带 anchor）** |
|---|---|---|
| 成本 | ~70 分钟（2000 步 × ~2 s + 加载） | **~35 分钟**（1000 步） |
| 回答什么问题 | "带锚定的完整曲线长什么样"（与前 2000 步无锚定不可直接比，因为起点相同但 recipe 不同 → 其实**可以**比） | **"锚定能不能阻止 2000 步之后的漂移"** ← 这是当前最要紧的问题 |
| 风险 | 低，但花 70 分钟才知道 λ 合不合适 | 低；λ 不合适可以立刻调（只损失 35 分钟） |
| 产物 | 一条干净的 anchor 曲线 | 一个"锚定后不再漂"的 ckpt（直接可进 A1） |

**判断**：当前的核心未知是"锚定是否有效"，而不是"完整曲线"。先用 B 花 35 分钟拿到答案：
- **若有效**（token_mse 守住 ≤0.077、reward ≥0.08、kinematic ≤0.006）→ 锚定方案成立，
  再按需决定是否补一次 A（从头、2000 步）作为论文用的干净曲线；
- **若无效**（token_mse 仍涨 或 reward 掉）→ 说明 λ=1.0 不合适，调 λ（0.3 / 3.0）再试，
  此时只损失 35 分钟——如果先做 A，就是白烧 70 分钟。

### 6.2 命令

**B（推荐，先跑这个）**
```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
RUN=joint_ha_a1 RESUME_STEP=2000 STEPS=3000 \
STATE_VEL_WEIGHT=1.0 TOKEN_ANCHOR_WEIGHT=1.0 PREFETCH_VLA=1 \
setsid nohup bash scripts/launch_joint_ha.sh > logs/joint_ha_anchor.log 2>&1 &
# 续训会自动：合并 vla_002000 → runs/ckpt_vla_ha_ft2000；载入 joint_002000（含 wm_integration）
# 注意：产物会写回 runs/joint_ha_a1/（会在 step 3000 覆盖旧的 joint_003000/vla_003000，
#       旧的那份已在 _attempt5_vel0_2000to3000/ 归档）
```

**A（备选，B 通过且需要干净曲线时）**
```bash
RUN=joint_ha_anchor STEPS=2000 \
STATE_VEL_WEIGHT=1.0 TOKEN_ANCHOR_WEIGHT=1.0 PREFETCH_VLA=1 \
setsid nohup bash scripts/launch_joint_ha.sh > logs/joint_ha_anchor.log 2>&1 &
# 从官方底模 checkpoint-8000 重训 2000 步，产物在 runs/joint_ha_anchor/
```

### 6.3 决策判据（到 step 3000 的 eval 就够）
| 观察 | 结论 | 下一步 |
|---|---|---|
| `token_mse ≤ 0.08` 且 `reward ≥ 0.08` | 锚定有效 | 继续到 4000，或直接用 step 3000 的 ckpt 进 A1 |
| `token_mse` 仍 > 0.09 | 锚定偏弱 | λ → 3.0 再跑 1000 步（从同一个 3000 存档续） |
| `reward < 0.075` | 锚定束缚了 reward | λ → 0.3 再跑 |
| kinematic > 0.008 | 执行器在漂 | 检查 `policy_lr`（1e-5）是否该降到 2e-6 |

---

## 7. 执行记录（2026-10-25 07:04，comimic4 / 4 卡）：**已起跑，B 方案扩到 4000**

- **选型**：按 §6.1 选 **B（续训 2000→…）**，但终点直接给到 **4000**（少开关一次进程；
  `EVAL_EVERY=500` ⇒ 2500/3000/3500/4000 四个点，§6.3 判据在 3000 那点照样能判）。
- **卡数安排（这 4 张卡都在跑别的消融，必须排队）**：
  卡 0,1 = A2 四臂 → A5 两臂（≈09:15 收）；卡 2,3 = A3-BPTT（120 min，≈07:56 收）；
  ⇒ anchor **只占单卡（卡 2）**，**等 chain2 一结束自动起跑**（排队器里等，不需要人盯）；
  **卡 3 留空给 A1 的 Isaac 闭环**（Isaac Sim 要独占一张卡，之前 base 臂就是因为没卡被 `Terminated`）。
- **为什么单卡而不是双卡**：主线 `runs/joint_ha_a1/args.json` 实测
  `_world_size=1 / batch_size=2 / prefetch_vla=True / state_vel_weight=1.0`。
  本跑**完全同形**（唯一差别 = `--token-anchor-weight 1.0`）⇒ 3000/4000 两点与主线曲线
  **单变量可比**。顺带把 §5 的纠结结掉：**主线续训用的是 `vel=1.0`**，所以"vel=1.0+anchor=1.0"
  正是与主线同一 recipe，不存在"两条约束都变弱"的问题。
- **启动**（已起，排队中）：
  ```bash
  setsid nohup bash scripts/launch_anchor_1025.sh > logs/anchor_queue.out 2>&1 &
  # 关键参数：RESUME_STEP=2000 STEPS=4000 NGPU=1 TOKEN_ANCHOR_WEIGHT=1.0 \
  #           STATE_VEL_WEIGHT=1.0 PREFETCH_VLA=1 EVAL_EVERY=500  CUDA_VISIBLE_DEVICES=2
  ```
- **产物隔离（重要）**：写 `runs/joint_ha_anchor_2000to4000/`，**不写回 `runs/joint_ha_a1/`**。
  §6.2 的原始命令会把主线的 `joint_003000/vla_003000` 覆盖掉（主线现在正跑到 3100+），
  所以 2000 的两个存档用**硬链接**放进新目录（主线目录被动也不影响续训）。
  日志：`logs/joint_ha_anchor_2000to4000.log`（训练）、`.queue.log`（排队）、`.launcher.log`。
- **起跑自检**：日志头部必须出现 `[anchor] ON: λ=1.0 ...` 与
  `[resume] joint <- .../joint_002000.pt (step=2000)`，且 `[eval 2000]` 与主线
  `token_mse 0.077285 / rollout_reward +0.081578` 一致（说明续训基线没跑偏）。

### 7.1 实际起跑（07:17:27，卡 2，单卡）与三项自检结果

> 用户指令：**停掉 A3-BPTT，把卡让给本续训**。执行顺序 = 先停 chain2（BPTT 在 step 2100 停），
> 再起 anchor，避免两个 120 min 任务抢同一对卡。

| 自检 | 实测 | 判定 |
|---|---|---|
| `[anchor] ON` | `λ=1.0 src=data/npz/ha_tokens port_dir=data/port_tokens/ha_tokens` | ✅ 走端口侧车 |
| `[resume]` | `joint_002000.pt (step=2000)`；vla `tensors=537 missing=494 unexpected=0` | ✅ |
| `[eval 2000]` | `token_mse 0.077289 / state_term 0.452769 / rollout_reward +0.081576` vs 主线 `0.077285 / 0.452769 / +0.081578` | ✅ **逐位一致**，单变量（只有 λ）成立 |

**对照基线（主线 `runs/joint_ha_a1`，同 recipe、无 anchor）**：

| step | `token_mse` | `state_term` | `rollout_reward` |
|---|---|---|---|
| 2000 | 0.0773 | 0.4528 | +0.08158 |
| 3000 | **0.0925（↑20%）** | 0.4191 | +0.08223 |

⇒ 本跑的判据就一条：**3000 步的 `token_mse` 能不能守住 ≤0.08**（同时 `reward ≥0.08`）。
`state_vel_weight=1.0` 与主线一致（主线 `args.json` 实测 `_nproc_per_node=1 / batch_size=2 /
state_vel_weight=1.0 / prefetch_vla=True`），所以不存在 §5 说的"两条约束都变弱"。

### 7.2 顺手修掉的一个真 bug（否则本容器起不来）

`scripts/launch_joint_ha.sh` 的 PYTHONPATH 注释里写了 `gr00t_deps`，但 export 行**漏了它**：

```diff
- export PYTHONPATH="$COMIMIC:$COMIMIC/Isaac-GR00T:$COMIMIC/third_party/gr00t_deps/_noflash..."
+ export PYTHONPATH="$COMIMIC:$COMIMIC/Isaac-GR00T:$COMIMIC/third_party/gr00t_deps:$COMIMIC/third_party/gr00t_deps/_noflash..."
```

症状：`ModuleNotFoundError: No module named 'tyro'` → `ImportError: Could not import the GR00T action head`
（07:15 第一次起跑即 exit=1）。在主线上能跑是因为那台容器的 site-packages 里本来就有 tyro；
本容器（comimic4）只有 `third_party/gr00t_deps` 里有。已修，其余容器无需改动（多一个 PYTHONPATH 项无害）。

---

## 8. 结论（2026-10-25 08:43，跑完 4000）：**token 锚定有效，且没有牺牲 reward / 跟踪**

产物 `runs/joint_ha_anchor_2000to4000/`：`joint_003000/004000.pt`、`vla_003000/004000.pt`、`vla_final.pt`
（07:17:27 起，08:43 收，单卡 ≈86 min / 2000 步；`--save-every 1000` 落盘齐全）。

### 8.1 硬数字（同一固定 eval batch，唯一差别 = λ）

| step | **anchor token_mse** | 主线 token_mse | anchor reward | 主线 reward | anchor pos/ori/q | 主线 pos/ori/q |
|---|---|---|---|---|---|---|
| 2000 | 0.0773 | 0.0773（同点） | +0.08158 | +0.08158 | 3.8e-5 / 0.0009 / 0.0035 | 同 |
| 2500 | **0.0193** | — | +0.08333 | — | 4.0e-5 / 0.0005 / 0.0030 | — |
| 3000 | **0.0181** | **0.0925（↑20% 漂移）** | **+0.08344** | +0.08223 | 4.2e-5 / 0.0012 / 0.0027 | 4.3e-5 / **0.0022** / 0.0032 |
| 3500 | 0.0183 | — | +0.08176 | — | 2.7e-5 / 0.0019 / 0.0030 | — |
| 4000 | **0.0152** | 未跑到（主线 06:38 被 SIGTERM 杀在 step ≈3100） | +0.08226 | — | 3.5e-5 / 0.0010 / 0.0030 | — |

### 8.2 验收（对照 §4 的表）

| 项 | 期望 | 实测 | 判定 |
|---|---|---|---|
| `loss/token_anchor` | ~0.17 → 单调降 | 0.0231 → 0.0207 → 0.0171 → … → 0.0153 | ✅ 降 |
| `token_mse`（固定 eval） | 单调降、无 2000→3000 反涨 | 0.0773→0.0193→0.0181→0.0183→**0.0152** | ✅ **−80%**，全程守 ≤0.02 |
| `rollout_reward` | ≥0.08 | **+0.0823**（且 3000 点比主线还高 +0.0012） | ✅ 没为锚定牺牲 reward |
| kinematic `pos+ori+q` | ≤0.006（oracle 0.0063） | **0.0040**（3.5e-5+0.0010+0.0030） | ✅ 与 oracle 同量级 |
| `state_vel` 地板 | 0.41–0.45 | **0.418** | ✅ |
| 额外收益 | — | `hand_mse 0.0114 → 0.0079`（−31%） | ✅ 手部辅助项一起变好 |

### 8.3 怎么读（别过度解读的地方）

1. **`token_mse` 是锚定项本身，不能当独立证据**。锚定把 token 拉回录制端口 token，MSE 自然降；
   真正证明"没学坏"的是**独立量**：`reward` 不降、`ori/q` 反而比主线好（0.0012/0.0027 vs 0.0022/0.0032）、
   `hand_mse` 下降。⇒ 结论成立：**锚定压住了 2000 步之后的 reward 平坦区随机漂移**。
2. λ=1.0 一档就够，**不需要再跑 λ=3.0**（§6.3 的"token_mse 仍 >0.09"分支未触发）；
   λ=0.3 也不必——reward 没有掉。
3. **主线 `runs/joint_ha_a1` 是残缺的**：`joint_ha_a1.log` 末尾是 `SignalException: Process … got signal: 15`，
   06:38:45 被杀在 step ≈3100，最后落盘 = `joint_003000/vla_003000`。
   ⇒ **A1 的 ft 臂要用 4000 步模型时，应用本跑的 `vla_004000.pt`**：
   ```bash
   python3 scripts/export_vla_ckpt.py --base runs/ckpt_vla_ha_8000_local \
     --vla runs/joint_ha_anchor_2000to4000/vla_004000.pt --out runs/ckpt_vla_ha_anchor4000 --force
   ```
4. 复现：`RESUME_STEP=2000 STEPS=4000 NGPU=1 TOKEN_ANCHOR_WEIGHT=1.0 STATE_VEL_WEIGHT=1.0 PREFETCH_VLA=1`，
   见 `scripts/launch_anchor_1025.sh`。

---

## 9. 两条 anchor 线的区别 + "A 方案为什么停过"（2026-10-25 08:55）

### 9.1 先分清两条线（别混）

| 线 | 目录 | 内容 | 状态 |
|---|---|---|---|
| **B（续训）** | `runs/joint_ha_anchor_2000to4000/` | 2000→4000，**已跑完**，`token_mse 0.0773→0.0152` | ✅ 见 §8 |
| **A（从头）** | `runs/joint_ha_anchor/`（旧）→ `runs/joint_ha_anchor_from0/`（今天的重跑） | 官方底模 `checkpoint-8000` 冷启 0→2000，全程带锚定 | 🟢 08:53 起跑（见 §9.3） |

### 9.2 A 方案早先那次**不是跑完、也不是代码错，是被外部 SIGTERM 杀掉**

证据（`logs/joint_ha_anchor.log` + `runs/joint_ha_anchor/args.json`）：

- `args.json` 落盘 **06:39**（`steps=2000 / token_anchor_weight=1.0 / state_vel_weight=1.0 / nproc=1`）；
- 日志里没有 `eval 0`、没有 `step 1`、没有 ckpt —— 说明**还在加载权重阶段**；
- 唯一异常是 `W1025 14:40:02 … Received 15 death signal` → `SignalException: Process … got signal: 15`
  （本地时间 14:40:02 = **UTC 06:40:02**，即启动后约 1 分钟）；
- **没有** OOM、没有 `ModuleNotFoundError`、没有 traceback 里的业务异常。
- 旁证：那次用的解释器是 `/opt/miniconda3/envs/isaaclab`（py3.11），**不是**联合训练该用的
  `/usr/bin/python3`（3.12）；且 06:39–06:40 这个点本容器 **4 张卡全被 A2 / A3-BPTT 占满**。
  ⇒ 典型情形：起在了没有卡的场景、随后被外部清理（或被人手 kill）。

### 9.3 重跑（A 方案，卡 2 空闲，08:53 起）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
RUN=joint_ha_anchor_from0 OUT=$PWD/runs/joint_ha_anchor_from0 STEPS=2000 NGPU=1 PORT=29674 \
  CUDA_VISIBLE_DEVICES=2 STATE_VEL_WEIGHT=1.0 TOKEN_ANCHOR_WEIGHT=1.0 PREFETCH_VLA=1 EVAL_EVERY=500 \
  setsid nohup bash scripts/launch_joint_ha.sh > logs/joint_ha_anchor_from0.launcher.log 2>&1 &
```

- 冷启（**不**给 `RESUME_STEP`），底模 = `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`；
- 与 B / 主线同形：`NGPU=1`、`batch 2`、`prefetch on`、`vel=1.0`、`λ=1.0`；`EVAL_EVERY=500`
  ⇒ 0/500/1000/1500/2000 五个点；
- 已核对起跑锚点：`[anchor] ON: λ=1.0 … port_dir=data/port_tokens/ha_tokens`；
  `[eval 0] token_mse 0.169811 / state_term 3.044407 / rollout_reward +0.040185`
  与 A2/A5 的 eval 0 指纹**逐位一致**（同一底模/同一 eval batch）；`loss/token_anchor = 0.2099`（起点值）。
- 用途：给论文一张**从 0 就带锚定**的干净曲线（B 的 0→2000 段是无锚定的，B 只能证明"锚定能止损"，
  A 才能说明"锚定从一开始就不漂"）。ETA ≈50–60 min → ≈09:50。
- 判据：四点 `token_mse` 单调降、2000 点 ≤0.08（B 在 2500 已是 0.0193，A 应远低于 0.08）、`reward ≥0.08`。

---

## 10. "续训到平台期为止" + "从 0 就带锚定"：两条线两卡并行（2026-10-25 09:06）

用户要求：① B 继续续到**平台期**为止；② 同时**从 0** 起一条带锚定的；③ 两条**并行在两卡**上跑。

### 10.1 卡位

| 卡 | 线 | 产物目录 | 起点 | 上限 |
|---|---|---|---|---|
| **2** | **A：从 0 带锚定**（0→2000，08:53 起）→ 自动接力 2000→平台 | `runs/joint_ha_anchor_from0/` → `runs/joint_ha_anchor_from0_2000to8000/` | 官方底模 `checkpoint-8000` | 8000（实际由平台判据停） |
| **3** | **B-cont：续训到平台**（4000→） | `runs/joint_ha_anchor_4000to8000/` | B 的 `joint/vla_004000`（**硬链接**） | 8000（实际由平台判据停） |
| 0,1 | A5 `head_plus_wbc`（既有消融，≈09:25 收） | — | — | — |

### 10.2 平台判据（`scripts/plateau_guard.sh`，两条线共用）

最近 **3 个 eval 点**同时满足才停（避免把 batch 噪声当平台）：

1. 都没有把"历史最好 `token_mse`"再压低 **≥ 0.002**（`IMPROVE`）；
2. `rollout_reward` 极差 **≤ 0.0025**（`RANGE`）。

⇒ 语义 = "训练目标（reward）已饱和 + 保真项（token）不再有实质进步"。
命中后按**进程组**停（`kill -TERM -- -PID`，setsid 保证 pgid=pid），只杀该 run，
最近一次 `--save-every 1000` 的 ckpt 保留。日志：`logs/<tag>.guard.log`。

### 10.3 起跑核对（两条线都必须过）

- **B-cont**：`[resume] joint_004000 (step=4000)`、vla `tensors=537`、
  `[eval 4000] token 0.015151 / reward +0.082262` ⇒ 与 B 的末点**逐位一致**（续训基线没跑偏）。
  ⚠️ 基座载入必须是 `runs/ckpt_vla_ha_anchor4000`：`launch_joint_ha.sh` 原本把 merge 目录写死成
  `runs/ckpt_vla_ha_ft${RESUME_STEP}`，而 `ckpt_vla_ha_ft4000` 里是**最早主线的 step-2000 权重**
  （07:00 那次 A1 接力留下的），直接用它 = 静默错配。为此新增 `MERGED_CKPT` 环境变量覆盖。
- **A 线**：`[eval 0] token 0.169811 / reward +0.040185`（与 A2/A5 指纹一致）、`[anchor] ON: λ=1.0`。

### 10.4 A 线早期数据（很重要）

| step | `token_mse` | `rollout_reward` | kinematic(pos+ori+q) | `hand_mse` |
|---|---|---|---|---|
| 0（官方底模） | 0.1698 | +0.04019 | 0.06849 | 0.6513 |
| **500** | **0.0233** | **+0.07985** | **0.00493** | **0.0090** |

⇒ **从 0 就带锚定，500 步（≈13 min）就走到 B 花了 2500 步才到的水平**。如果 1000/1500/2000 继续降，
那"最终模型"应该用 A 的（训练量省一半、recipe 干净、无混血）。

### 10.5 复现命令

```bash
# B：继续续训到平台（卡 3）
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
mkdir -p runs/joint_ha_anchor_4000to8000
ln runs/joint_ha_anchor_2000to4000/joint_004000.pt runs/joint_ha_anchor_4000to8000/joint_004000.pt
ln runs/joint_ha_anchor_2000to4000/vla_004000.pt   runs/joint_ha_anchor_4000to8000/vla_004000.pt
RUN=joint_ha_anchor_4000to8000 OUT=$PWD/runs/joint_ha_anchor_4000to8000 \
  RESUME_STEP=4000 STEPS=8000 NGPU=1 PORT=29675 CUDA_VISIBLE_DEVICES=3 \
  MERGED_CKPT=$PWD/runs/ckpt_vla_ha_anchor4000 \
  STATE_VEL_WEIGHT=1.0 TOKEN_ANCHOR_WEIGHT=1.0 PREFETCH_VLA=1 EVAL_EVERY=250 \
  setsid nohup bash scripts/launch_joint_ha.sh > logs/joint_ha_anchor_4000to8000.launcher.log 2>&1 &
bash scripts/plateau_guard.sh <PID> runs/joint_ha_anchor_4000to8000/eval_curve.jsonl logs/joint_ha_anchor_4000to8000.guard.log &

# A：从 0 → 2000 → 自动接力到平台（卡 2）
RUN=joint_ha_anchor_from0 OUT=$PWD/runs/joint_ha_anchor_from0 STEPS=2000 NGPU=1 PORT=29674 \
  CUDA_VISIBLE_DEVICES=2 STATE_VEL_WEIGHT=1.0 TOKEN_ANCHOR_WEIGHT=1.0 PREFETCH_VLA=1 EVAL_EVERY=500 \
  setsid nohup bash scripts/launch_joint_ha.sh > logs/joint_ha_anchor_from0.launcher.log 2>&1 &
setsid nohup bash scripts/queue_anchor_from0_phase2.sh > logs/anchor_from0_phase2.out 2>&1 &
```

### 10.6 结果：**B 在 4000 步就是平台**（09:22 哨兵自动停）

| step | `token_mse` | `rollout_reward` | kin(pos+ori+q) | `hand_mse` | `state_vel` |
|---|---|---|---|---|---|
| **4000** | **0.0152** | +0.08226 | 0.00405 | 0.0079 | 0.418 |
| 4250 | 0.0171 | +0.08278 | 0.00437 | 0.0081 | 0.403 |
| 4500 | 0.0165 | +0.08385 | 0.00378 | 0.0048 | 0.377 |
| 4750 | 0.0169 | +0.08311 | 0.00387 | 0.0046 | 0.386 |

判据命中（`logs/joint_ha_anchor_4000to8000.guard.log`，09:22:00）：token 从 4000 的 0.0152 起
**再没创新低**（0.0171/0.0165/0.0169 全部 > 0.0152−0.002），reward 极差仅 **0.0011 ≤ 0.0025**。

⇒ **`reward` 与 `token` 双双饱和，B 的平台点就是 4000 步**；继续训只有 hand 还在小幅降（0.0081→0.0046），
那是辅助项、不改变结论。**B 的最终产物 = `vla_004000.pt`**（09:00 已 merge 成
`runs/ckpt_vla_ha_anchor4000`，537/537 校验通过）。这一轮 4000→4750 没落新 ckpt
（`save-every 1000`，停在 4750 < 5000），也不需要——最好的点正是 4000。

### 10.7 A 线（从 0）当前读数（09:25，step 1400/2000）

| step | `token_mse` | `reward` | kin | `hand_mse` | `state_vel` |
|---|---|---|---|---|---|
| 0 | 0.1698 | +0.04019 | 0.06849 | 0.6513 | 2.976 |
| 500 | 0.0233 | +0.07985 | 0.00493 | 0.0090 | 0.499 |
| **1000** | **0.0204** | **+0.08245** | **0.00407** | 0.0116 | 0.388 |

⇒ 1000 步时：**reward / kinematic 已经追平 B 的 4000 步**（+0.0825 vs +0.0823、0.0041 vs 0.0040），
**token 还差一档**（0.0204 vs 0.0152）。所以"从 0 锚定"能不能替代 B，关键看它在 2000（接力段 3000/4000）
能否把 token 压到 ≤0.017 且不再往回走。
