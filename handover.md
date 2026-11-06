# Handover：S3 五轮全长 VLA + WBC + WM 联合微调

> 生成时间：**2026-11-02 02:45 UTC**
> 工作区：`/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic`
> 本文是给下一个接手人的**当前状态 + 架构 + 结论 + 踩坑**总览。
> 长期计划以 `docs/FULL_LENGTH_5ROUND_PLAN_20261101.md` 为准；hinge 标定以
> `docs/HINGE_RECALIBRATION_5ROUNDS_20261101.md` 为准。若本文与旧文档冲突，以本文的“实际 args / 实际结果”为准。

---

## 0. 一句话现状

- **R1、R2、R3 已完成**；R3 在 DoubleDesk 上比 R2 配对平均好 **−0.0602 m**（22/32 更好），但 95%CI `[−0.1229, +0.0032]` 跨 0，且 R3 出现 **1 次摔倒**（R2 为 0），严格安全门未过。
- **R4 正在跑**：2026-11-02 02:40:54 启动，round `904`，配置 `E_start=1.6 / e_h=0.632 / λ=0.70 / cap=1.490 / act=0.5 / 200 steps`；当前在采 84 条 fresh（6 任务 ×14），随后自动跑 WM/critic/joint/32 对评测。
- **评测覆盖有硬伤**：此前 R1/R2/R3 的正式门控**全部只评了 DoubleDesk**。OpenDoor 目前只有 1 对有效 probe；PPBox、Boxing、SitSofa、VisionNavi 尚未做闭环评测。**不能把 DoubleDesk 的结论外推到六任务。**
- **所有 DoubleDesk/OpenDoor 闭环 episode 目前都是 `timeout`、`success=0`**。0.85 m 的全集平均误差主要来自“到桌前失败后停住、参考继续走”的约 2 m 平台，不是持续漂移。
- 当前最需要解决的不是“再训几步”，而是：**抓取/搬运这一段没有成功、训练数据本身是失败轨迹、评测只覆盖一个任务**。

---

## 1. 正在运行的 R4（先看这里）

| 项 | 值 |
|---|---|
| 启动时间 | 2026-11-02 02:40:54 UTC |
| harness job | `bash-69`（后台 job；日志见下） |
| 轮次 | `ROUND=904`，`N=4` |
| 起点 | `runs/S3_loop/903_joint/train_main/vla_final.pt`（R3 final） |
| 配置 | `E_START=1.6`，`EH=0.632`，`LAM=0.7000`，`CAP=1.490`，`ACT_W=0.5`，`STEPS=200`，`EVAL_N=32`，`SEED=4` |
| 数据 | `B_904`：84 条 fresh（6 任务 ×14），`B_GLOB=B_90[1-4]`，fresh weight 3.0 |
| 当前阶段 | 采集。02:43 时 **8/84**；每条约 130–160 s，8 卡并行 |
| 日志 | `logs/auto_ladder/r4_round904.log` |
| 监控 | `tail -f logs/auto_ladder/r4_round904.log`；`find runs/B_904 -path '*/dump/*.pt' | wc -l` |

启动命令（如果进程死了、需要原样续跑）：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
ROUND=904 N=4 PREV_ROUND=903 \
PREV_HEAD="$PWD/runs/S3_loop/903_joint/train_main/vla_final.pt" \
E_START=1.6 EH=0.632 LAM=0.7000 CAP=1.490 ACT_W=0.5 STEPS=200 EVAL_N=32 SEED=4 \
bash scripts/run_full_round_auto.sh
```

> `run_full_round_auto.sh` 会先检查 `B_904` 是否已有 84 条；不足才采，够了就直接进入 WM/pipeline。
> 它最后会用 `eval_h20_8gpu.sh` 跑 32 对，并做方向性 + 安全门控；门控失败会写
> `runs/S3_loop/round_904_gate_fail.txt` 并以非零退出，**不会自动进 R5**。
>
> R4 是在 R3 安全门出现 1 次摔倒后、由用户明确要求“先启动 R4”而启动的；
> 这不是“R3 已严格过门”的结论。R4 若失败，仍应停在 R4 修，不要进 R5。
> R4 仍是 200 步；用户关心的“每轮多训一会儿（400 步）”尚未执行，见 §8 P3。

---

## 2. 总体架构

### 2.1 组件

```text
图像 + 语言指令
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│ VLA：Cosmos-Reason2-2B + LoRA(r=32) + action head            │
│  - LoRA targets: attn1, ff；另外训练 proj_out                │
│  - base ckpt: runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 │
│  - 输出：64 维 motion token（latent），不是直接 29 维关节    │
└──────────────────────────────────────────────────────────────┘
      │ 64-d token
      ▼
┌──────────────────────────────────────────────────────────────┐
│ 冻结执行器 WBC v6：WBC_A24000_wmvel_v6.pt                   │
│  - token + 本体观测 → 29 维身体关节目标                      │
│  - 主计划全程冻结；joint 训练不解冻 WBC                      │
│  - 评测也用这个冻结执行器（不是 joint_final.pt）             │
└──────────────────────────────────────────────────────────────┘
      │ 29 维关节目标
      ▼
┌──────────────────────────────────────────────────────────────┐
│ IsaacLab / HumanoidArena 闭环仿真                            │
│  - 6 个任务：DoubleDesk / PPBox / Boxing / OpenDoor /        │
│    SitSofa / VisionNavi                                     │
│  - 参考：runs/isaac_motion/ha_cmd/packed.pt（真实 motion lib）│
│  - dump 的 frames 是 library row，用于计算 root 误差          │
└──────────────────────────────────────────────────────────────┘
```

训练时还有两个冻结件：

- **世界模型 WM**：8.82M MLP；输入当前状态/动作，预测下一步状态和奖励；用于 H=20 的 imagined rollout。
- **token critic**：在想象 rollout 上做 GAE，给策略提供长视界信用分配；不是部署件。

### 2.2 数据流（B 线）

每条 B episode 保存：

| 字段 | 形状/含义 |
|---|---|
| `states` | `[T,71]`：root xyz(3)+quat wxyz(4)+q(29)+v(3)+w(3)+qd(29)，ActionCodec 顺序 |
| `actions` | `[T,29]`：实际下发的 29 维目标（WBC 输出/应用后的目标） |
| `q_target_applied` | `[T,29]`：真正进 PD 前的 29 维目标 |
| `tokens` | `[T,64]`：每 20 步重规划时喂给策略/执行器的 token |
| `frames` | `[T]`：真实参考库行号（`library_row`），用于 root/joint 误差 |
| `meta` | 任务、layout、reward 配置、执行器 ckpt 等 |

`packed.pt` 是全局 motion library，包含：

```python
ref     # [634016, 71] 参考轨迹
names   # 100+ clips per task
starts  # 每段 clip 的全局起始 row
lengths # 每段帧数
```

**关键：`START_FRAME` 是全局 library row，不是任务内 offset。** 第 0 行是 DoubleDesk 的第一帧；OpenDoor 的起点在 `371449+` 之后，详见 §4.2。

### 2.3 joint 训练目标（实际配置）

- 输入：B/G 的真实转移 + WM 的 H=20 imagined rollout；
- 动作损失：GR00T 原生 flow-matching（`fm_bc`）；
- 根位置尾部 hinge：`λ_tail · mean(max(0, e − e_h) capped)`，`e` 为 H=20 内 root 位置误差；
- 奖励项：WM 想象回报，`λ_r=0.3`；
- token trust region：`trust_center=label`，`trust_eps=0.2`，`trust_weight=1.0`；
- critic：`CRITIC_SOURCE=imagined`，GAE，`CRITIC_GRAD_STATE=1`，`state_grad_scale=0.1`；
- 不做 token BC（`TOKEN_BC=0`）、不做 hand loss、不做 posture/tilt penalty、不解冻 WBC；
- 8 卡 DDP：`NPROC=8`，`BATCH=64`，`ACCUM=1`，全局有效 batch 64；`H=20`；`STEPS=200`（R5 计划 300）；
- dtype bf16；LoRA rank 32，alpha 64；head 以 `vla_final.pt`（约 129–134 MB）保存，base ckpt 约 6.5 GB。

---

## 3. 五轮状态与超参

### 3.1 轮次总表（实际跑的值）

| 轮 | 起点 | B 数据 | `act_weight` | `E_start` | `e_h` | `λ` | `cap` | steps | 结果 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| base | — | — | — | — | — | — | — | — | 锚点 |
| **R1** | base VLA | B_901 180（6×30） | **0.5** | 1.0 | 0.401 | **1.0** | 0.971 | 200 | n44 gate PASS |
| **R2** | R1 final | B_902 84（6×14） | 0.8 | 1.2 | 0.485 | **0.2474** | 1.165 | 200 | n32 directional PASS |
| **R3** | R2 final | B_903 84（6×14） | 0.6 | 1.4 | 0.564 | **0.80** | 1.327 | 200 | n32 方向 PASS，安全门 1 fall 未过 |
| **R4** | R3 final | B_904 84（采集中） | 0.5 | 1.6 | 0.632 | **0.70** | 1.490 | 200 | **运行中** |
| R5 | R4 最优 | 84 fresh | 0.5 | 1.8 | 0.679 | 0.65 | 1.637 | 300 | 未开始 |

注意：

- **R1 的 `act_weight=0.5` 是实际打赢的 R1-fix4 配置**；计划文档表格里的 `1.0` 是早期名义值，不要拿它复现 R1。
- **R2 的 `λ=0.2474` 是 v2 标定值**；按 2026-11-02 的 v3 p90 原则，R2 应该是 `0.85`。R2 没有按 v3 重训，保留实际值。R3+ 已按 v3 使用 `0.80/0.70/0.65`。
- 所有轮的 `trust_center=label`、`act_eps=0.6`、`tail_cap_slope=0.5`、`yaw_hinge=0`、`yaw_rate=0`。

### 3.2 B 数据池

| 池 | 条数 | 组成 | 用途 |
|---|---:|---|---|
| `B_901` | 180 | 6 任务 ×30 | R1 静态池；后续累计 |
| `B_902` | 84 | 6 任务 ×14 | R2 fresh |
| `B_903` | 84 | 6 任务 ×14 | R3 fresh |
| `B_904` | 84（采集中） | 6 任务 ×14 | R4 fresh |

累计 glob：

```text
R2: runs/B_90[1-2]/*/dump/*.pt
R3: runs/B_90[1-3]/*/dump/*.pt
R4: runs/B_90[1-4]/*/dump/*.pt
```

每轮 fresh 权重 3.0；`B_START_MAX_ERR` 依次为 1.0/1.2/1.4/1.6/1.8。

### 3.3 关键 artifact

| 项 | 路径 |
|---|---|
| base VLA head | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000` |
| R1 final head | `runs/S3_loop/901fix4_partial/train_main/vla_final.pt` |
| R1 pointer | `runs/S3_loop/901fix4_best.txt` |
| R2 final head | `runs/S3_loop/902_joint/train_main/vla_final.pt` |
| R2 pointer | `runs/S3_loop/902_best.txt` |
| R3 final head | `runs/S3_loop/903_joint/train_main/vla_final.pt` |
| R3 旧 λ=0.1931 run（停止，仅存档） | `runs/S3_loop/903_prelambda/` |
| R3 WM | `runs/S1_handover/wm_903.pt` |
| R2 WM | `runs/S1_handover/wm_902.pt` |
| 评测执行器 | `runs/S1_handover/WBC_A24000_wmvel_v6.pt` |
| 配置/args | 各 run 的 `train_main/args.json`、`v3_config.json`、`summary.json` |

`vla_final.pt` 是 head delta（约 129–134 MB），加载方式：

```bash
HEAD_CKPT=runs/S3_loop/903_joint/train_main/vla_final.pt
# 或评测时 HEAD_MAIN / HEAD_CTRL
```

---

## 4. 评测协议与致命坑

### 4.1 有效协议（当前）

- 任务：默认 **DoubleDesk**（其他任务必须显式传 `TASK` 和正确的 `START_FRAME`）；
- 每条 episode：`START_FRAME` 指定全局 library row，`MAX_STEPS=2080`，跑到 2080 步或提前结束；
- 动作采样：`--sample-avg 4`；
- 执行：server 出 token → 冻结 WBC v6 → Isaac；`EXEC_STEPS=20`（每 20 步重规划）；
- 并行：4 条 sweep ×2 GPU，每条 sweep 8 个 seed，合计 **32 对**；
- 指标：
  - **root mean**：每步 `||state.xyz − ref[frame].xyz||` 的整集平均（m）；
  - **joint RMSE**：29 关节误差；
  - **tilt max**：整集最大倾角，`>45°` 计为摔倒；
  - **success/failure_reason**：来自 `result.json`。

### 4.2 ★ 非 DoubleDesk 任务必须用任务自己的起始 row

`START_FRAME=0` **只对 DoubleDesk 有效**，因为 `packed.pt` 第 0 行是 `HOI_double_desk` 的第一帧。

OpenDoor 的正确入口：

```bash
python3 scripts/sample_start_row.py --task OpenDoor --seed 0 --margin 2160
# 输出 373209（最长 OpenDoor clip，长度 2069；2080 会略超出 11 帧）
```

2026-11-02 的教训：用 `TASK=OpenDoor START_FRAME=0` 跑了 8 对，机器人被放在 DoubleDesk 起点、开门场景里几乎不动，参考却沿 DoubleDesk 轨迹走。得到：

```text
R2 1.698 m vs R3 1.782 m，配对 +0.0841 m，R3 2/8 更好，全部 timeout，0 fall
```

**这组数字无效，不能引用。** 正确起点后 1 对 probe：

```text
R2 0.505 m vs R3 0.499 m，配对 −0.006 m，R3 略好；两边都 timeout，0 fall；
root 位移 1.33/1.43 m（说明确实在动）
```

但 n=1，不能判优劣。OpenDoor 需要至少 8–16 对有效数据，其他 4 个任务目前完全没有闭环结果。

### 4.3 门控

R2–R4 使用 **directional + safety gate**：

```text
mean(ctrl − main) < 0
median(ctrl − main) < 0
better >= 55% of pairs
falls(ctrl) <= falls(main)
```

R1 由于 n=44，额外看 bootstrap 95%CI；R1 的严格门是 `CI upper <= +0.01 m`。

注意：**这个门只看 DoubleDesk**。在补完其他任务之前，不能把它当成“六任务通过”。

### 4.4 指标的最大陷阱：全集平均被失败尾巴主导

DoubleDesk 目前所有 episode 都是 `timeout`、`success=0`。典型好集：

| 时刻 | 机器人 | 参考 | 根误差 |
|---|---|---|---|
| t=800 | (-2.62,-2.55)，-59° | (-2.61,-2.58)，-68° | 0.03 m |
| t=1000 | (-2.60,-2.53)，-69° | (-2.59,-2.58)，-78° | 0.05 m |
| t=1200 | (-2.60,-2.52)，-65° | (-2.51,-2.58)，**+22°** | 0.11 m |
| t=1600 | (-2.61,-2.52)，-64° | (-2.28,-0.67)，+119° | **1.89 m** |
| t=2079 | (-2.61,-2.51)，-66° | (-2.60,-0.60)，+104° | **1.92 m** |

机器人在桌前停住，参考在 t≈1200 转身走向篮子（约 1.9 m）。所以：

- 前 1000 步误差约 **0.34–0.41 m**；
- 后 40% 误差饱和到约 **1.9–2.0 m**；
- 全集平均 0.85 m 里很大一块是“失败后静止 vs 参考继续走”。

这不是纯漂移，也不是“主动判定完成”；是**抓取/放置没有完成，后续转身搬运没有发生**。训练数据 `B_903` 的 DoubleDesk 也是同样结尾：终点误差 1.9–2.35 m，根节点只移动约 1 m。**继续在失败数据上多训，不会自己长出成功抓取。**

---

## 5. 结果（DoubleDesk，配对）

### 5.1 三轮主结果

| 对比 | n | 平均配对差 | 中位配对差 | 候选更好 | bootstrap 95%CI | falls |
|---|---:|---:|---:|---:|---|---|
| R1 − base | 44 | **−0.0962 m** | −0.0958 | **27/44 (61.4%)** | **[−0.1922, −0.0010]** | — |
| R2 − R1 | 32 | **−0.0433 m** | −0.012 | 18/32 (56.3%) | [−0.1602, +0.0675] | 7 → **0** |
| R3 − R2 | 32 | **−0.0602 m** | −0.068 | **22/32 (68.8%)** | [−0.1229, +0.0032] | 0 → **1** |

绝对均值（R3 场，n32）：

```text
R2  0.9178 m  (sd 0.1659, median 0.894, p90 1.141, joint 0.3592, fell 0)
R3  0.8576 m  (sd 0.1493, median 0.826, p90 1.063, joint 0.3618, fell 1)
```

判读：

- **R1 是唯一 CI 不跨 0 的轮次**（n44，且 64 对 finalizer 被中断，只用了 44 对）。
- R2 方向对、摔倒 7→0，但 n32 CI 跨 0，属“方向性通过”。
- R3 方向更强、22/32，但 **1 次真实摔倒（seed31, tilt 92°）**，安全门 `falls not worse` 严格失败。
  用户明确要求继续启动 R4，所以 R4 已跑，但交接时不要把 R3 写成“安全门通过”。

### 5.2 早期跟踪 vs 全集

R3 相对 R2 的改善主要来自“到桌前这一段”和“卡住位置”，不是完成了后续：

| 指标 | R2 | R3 | 差 | R3 更好 |
|---|---:|---:|---:|---:|
| 前 1000 步根误差 | 0.409 | 0.369 | **−0.040 m** | 17/32 |
| 前 1200 步根误差 | 0.410 | 0.373 | −0.037 m | 17/32 |
| 整集平均 | 0.918 | 0.858 | −0.060 m | 22/32 |

所以“到桌子这段降低明显”这个说法要谨慎：**只有约 4 cm，且 17/32，五五开偏一点**；全集 6 cm 更多是因为卡住的位置/时机稍好，不是质变。

### 5.3 OpenDoor

| 评测 | n | R2 | R3 | 差 | 结果 |
|---|---:|---:|---:|---:|---|
| 错误起点（START_FRAME=0） | 8 | 1.698 | 1.782 | **+0.084**（R3 更差） | 全部 timeout，0 fall；**无效** |
| 正确起点（373209）probe | 1 | 0.505 | 0.499 | **−0.006**（R3 略好） | 两边 timeout，0 fall |

**只能引用第二行，且必须标注 n=1。** 第一行是 wrong-start artifact，不要写进任何结论表。

### 5.4 其他任务

| 任务 | 闭环评测状态 |
|---|---|
| DoubleDesk | R1/R2/R3 有 32/44 对；R4 评测将再跑 32 对 |
| OpenDoor | 1 对有效 probe；需补 8–16 对 |
| PPBox | **未评测** |
| Boxing | **未评测** |
| SitSofa | **未评测** |
| VisionNavi | **未评测** |

训练数据六任务都有，但评测尚未覆盖。**当前所有“轮次提升”只能主张 DoubleDesk。**

---

## 6. 为什么到后 40% 会停在原地

结论：**不是策略主动停下，而是桌前抓取/操作没有完成，后续“转身—搬运—走向篮子”没有触发。**

证据：

1. 好集 `t=800–1000` 根误差只有 0.03–0.05 m，说明接近段跟得很好；
2. `t≈1200` 参考 yaw 从 −78° 转到 +22°，开始走向篮子；机器人 yaw 一直 −65°，根位置从 t=1200 到结束几乎不变；
3. 根速度约 3.5 cm/s，但手臂关节仍有小幅运动、action 幅值约 0.8：它还在桌前“尝试”，不是完全静止；
4. 更差的集会更早偏航，卡在 (-1.84,-2.6) 之类的位置，从 t=400 起就不动；
5. `B_903` 的 DoubleDesk 训练轨迹也是同样结尾（终点误差 1.9–2.35 m，根位移约 1 m），说明数据里没有成功的抓取后搬运片段。

因此：

- 多训 200→400 步，可能把前 1000 步再压一点，但**不会自动补出缺失的“抓取成功→转身搬运”**；
- 要真正降低全集平均，需要**成功的 DoubleDesk 示范**（尤其是抓取后转身、走到篮子），或对参考轨迹做成功片段/分阶段监督；
- 评测也应从“全集平均”改为同时看 **前 1000 步误差、首次发散时间、success/timeout**，否则一个 2 m 平台会把所有改进淹没。

---

## 7. 已排除 / 已踩坑（不要重复）

| 项 | 结论 |
|---|---|
| d2b / h20r2 当 baseline 或起点 | 旧 600-tick 协议 checkpoint，不适配全长；baseline 必须是 base VLA `checkpoint-8000` |
| 旧 936 条 600-tick B | 不进入五轮主计划，避免协议污染 |
| hinge v1（`e_h=p75`, R1 0.738） | 制造 0.4–0.7 m 中近场死区，策略站立不动；已废弃 |
| hinge v2（`e_h=p45`, R1 λ=0.3592） | R1 实际用 λ=1.0 才过；v2 的 λ 对后续轮太弱，R3 用 0.1931 时 `imag_state` 退化到 1.54，已停 |
| R1-fix2/3/5/6/7 | 分别因 no-filter、act=1.0+λ1.0 崩塌、hybrid trust、act=1.0 label、8-pair 全负等被停；只有 **fix4** 赢 |
| R3 λ=0.1931 | 20/32 平均 −0.0017，10/20 更好；停止，旧 run 存 `903_prelambda`，改用 λ=0.8 |
| `START_FRAME=0` 跑 OpenDoor | 会把机器人放 DoubleDesk 起点，数据无效；必须用任务自己的 row |
| `pgrep -f 'train...'` 自匹配 | watcher 会永远等自己；用 `pgrep -f '[t]rain...'` |
| 工具超时杀进程 | 长评测/训练必须用 managed background job 或 `setsid`；普通 tool 超时会连带杀掉子进程（本次 4 条 eval 被误杀过） |
| `s3_pipeline.sh` 只跑第一个 stage | 驱动曾只读 `$1`；已修为循环 `collect score transitions wm_refresh wm_gate critic_gate joint` |
| WM symlink double-prefix | stage2 的 `traj_901_train/v_100.pt` 找不到；已修绝对/相对路径归一化 |
| WM ratio 审计错池 | 曾把 B150/A333 轨迹池当 batch row；已修为解析 `[mix]` A=128/B=128 |
| `act_weight` 自动 curriculum 静默 0.125 | 已改为显式 `ACT_WEIGHT_SCHEDULE` |
| `trust_center=hybrid`、`act_weight=1.0` | R1-fix6/7 都更差，已排除 |
| `s3_eval_sweep.sh` 按进程名全杀 server | 8 卡并行时会互相杀；已收紧到本 sweep 的 `PORT_MAIN/PORT_CTRL` |

---

## 8. 当前未决问题 / 下一步优先级

### P0：看住 R4

1. 等 `B_904` 到 84/84；
2. 盯 `wm_gate`、`critic_gate`、`joint`；
3. 32 对 R4 vs R3 出来后，按 directional+safety 门判；
4. 若 R4 失败：停在 R4，修当前轮，不要进 R5。

### P1：补多任务评测（必须做，不能再只用 DoubleDesk）

每个任务都必须：

```bash
python3 scripts/sample_start_row.py --task <TASK> --seed <s> --margin 2160
```

然后：

```bash
TASK=<TASK> TAG=<tag> SEEDS="0 1 2 3 4 5 6 7" \
START_FRAME=<task-specific-row> MAX_STEPS=2080 \
HEAD_MAIN=<prev> HEAD_CTRL=<cur> \
SM_CKPT_MAIN=$PWD/runs/S1_handover/WBC_A24000_wmvel_v6.pt \
SM_CKPT_CTRL=$PWD/runs/S1_handover/WBC_A24000_wmvel_v6.pt \
EXTRA_MAIN="--sample-avg 4" EXTRA_CTRL="--sample-avg 4" \
bash scripts/s3_eval_sweep.sh
```

OpenDoor 已知可用 row：`373209`。其他任务先用 `sample_start_row.py --list` 看 clip 长度，再选 margin。

### P2：解决“成功片段缺失”

当前 B 数据是策略自己的失败轨迹，`scored.json` 只有 `r_mean`，没有成功标签；DoubleDesk `result.json` 全是 timeout。要真正降低后 40% 的 2 m 平台，需要：

- 成功的 DoubleDesk 示范（reference/teleop/更好的策略）；
- 或对参考轨迹做成功片段监督（token/action BC），尤其抓取后转身搬运段；
- 或在数据里按阶段重采样，而不是只增加训练步数。

### P3：多训 400 步的对照（用户明确关心）

目前没有 400 步实验。已有证据：

- R3 step100→200：imagined return 0.652→0.660，`imag_delta` 反而 +0.005；
- R2 step100→200：0.723→0.724，基本平；
- 所以 200 附近已接近平台，单纯加步数不保证改善。

如果要做，建议控制变量：从 R3-200 继续 200 步（总 400），保存 300/350/400，在同一批 16 个种子上与 R3-200 闭环配对；只有 400 明确赢，才把 R4/R5 改成 400。不要直接在所有轮上加步数，否则无法归因。

### P4：修 hinge 标定脚本的 v3 一致性

`docs/HINGE_RECALIBRATION_5ROUNDS_20261101.md` 已写 v3 p90 原则：

```text
λ_round = 0.517 / (p90_round − e_h_round)
R1 1.0 / R2 0.85 / R3 0.80 / R4 0.70 / R5 0.65
```

但 `scripts/calibrate_hinge_5round.py` **仍只输出 v2 的 λ**（解 `mean(hinge)/mean(exp)=0.30`），没有输出 v3 的 p90 λ。下一轮重新标定时必须补上，避免又拿回 0.19/0.16/0.14。

### P5：R3 的 1 次摔倒

严格安全门未过。可选：

- 补 16–32 对确认摔倒率是否稳定；
- 或在 R4/R5 加稳定性约束/降低动作激进程度；
- 不要因为“均值更好”就忽略安全门。

---

## 9. 命令速查

### 9.1 监控 R4

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

tail -f logs/auto_ladder/r4_round904.log
find runs/B_904 -path '*/dump/*.pt' | wc -l
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

### 9.2 合并/判读 DoubleDesk 结果

```bash
python3 scripts/merge_eval_sweeps.py \
  runs/S3_loop/sweeps/full_r3lam08_vs_r902_s0 \
  runs/S3_loop/sweeps/full_r3lam08_vs_r902_s1 \
  runs/S3_loop/sweeps/full_r3lam08_vs_r902_s2 \
  runs/S3_loop/sweeps/full_r3lam08_vs_r902_s3 \
  --out runs/S3_loop/sweeps/full_r3lam08_vs_r902_merged.json
```

### 9.3 单任务有效评测（以 OpenDoor 为例）

```bash
TASK=OpenDoor TAG=opendoor_valid_probe SEEDS="0" START_FRAME=373209 MAX_STEPS=2080 \
HEAD_MAIN="$PWD/runs/S3_loop/902_joint/train_main/vla_final.pt" \
HEAD_CTRL="$PWD/runs/S3_loop/903_joint/train_main/vla_final.pt" \
SM_CKPT_MAIN="$PWD/runs/S1_handover/WBC_A24000_wmvel_v6.pt" \
SM_CKPT_CTRL="$PWD/runs/S1_handover/WBC_A24000_wmvel_v6.pt" \
EXTRA_MAIN="--sample-avg 4" EXTRA_CTRL="--sample-avg 4" \
GPU_MAIN=0 GPU_CTRL=1 PORT_MAIN=18400 PORT_CTRL=18401 \
bash scripts/s3_eval_sweep.sh
```

### 9.4 重新标定 hinge（注意：当前脚本只有 v2 λ）

```bash
python3 scripts/calibrate_hinge_5round.py \
  --out runs/S1_handover/hinge_calibration_5rounds.json
```

### 9.5 查看 R3 训练曲线

```bash
python3 - <<'PY'
import json
for line in open('runs/S3_loop/903_joint/train_main/eval_curve.jsonl'):
    r=json.loads(line)
    print(r['step'], r.get('imag_return_vla'), r.get('imag_delta_mean'), r.get('imag_state_vla'))
PY
```

---

## 10. 文件索引

### 文档

| 文件 | 内容 |
|---|---|
| `docs/FULL_LENGTH_5ROUND_PLAN_20261101.md` | 五轮全长计划（注意 R1 表里的 act=1.0 不是实际赢的 fix4） |
| `docs/HINGE_RECALIBRATION_5ROUNDS_20261101.md` | v2 标定 + v3 p90 λ 原则 |
| `docs/HANDOVER_8GPU_JOINTFT_20261101.md` | 8 卡 joint 的工程细节 |
| `docs/H20_8GPU_JOINTFT_20261101.md` | H=20 的早期闭环证据 |
| `docs/FINAL_HANDOVER.md` | 更早的架构锁定版（SuperMimic 时代） |
| **`handover.md`** | 本文件：当前状态与结论 |

### 脚本

| 脚本 | 用途 |
|---|---|
| `scripts/run_full_round_auto.sh` | 一轮完整流程：采 84 → WM → joint → 32 对评测 → 门控 |
| `scripts/run_full_r1fix.sh` | R1-fix 的 joint-only 启动器（参数化） |
| `scripts/s3_pipeline.sh` | 单 stage/全流程：collect/score/transitions/wm_refresh/wm_gate/critic_gate/joint/eval |
| `scripts/s3_loop.sh` | 底层 train/eval/collect；`RUN_LABEL` 控制输出目录 |
| `scripts/train_joint_v2.py` | v2/v3 joint 训练入口 |
| `diff_vla_wbc/v3_step.py` | V3 训练步：FM + hinge + trust + critic |
| `diff_vla_wbc/isaac_joint.py` | Isaac 联合数据/动作映射 |
| `scripts/s3_eval_sweep.sh` | 2 卡 ×多种子闭环 sweep |
| `scripts/eval_h20_8gpu.sh` | 4×2 卡 ×多种子 32/48 对评测 |
| `scripts/merge_eval_sweeps.py` | 合并多条 sweep，输出配对统计 |
| `scripts/sample_start_row.py` | 按任务采样合法起始 row |
| `scripts/collect_full_episodes.py` | 六任务 ×14 条 fresh 采集 |
| `scripts/calibrate_hinge_5round.py` | hinge 标定（当前 v2 λ，需补 v3） |

### 结果 / 日志

| 路径 | 内容 |
|---|---|
| `runs/S3_loop/sweeps/base_vs_r1fix4_44partial.json` | R1 n44 主结果 |
| `runs/S3_loop/sweeps/full_r2_vs_r901_merged.json` | R2 n32 主结果 |
| `runs/S3_loop/sweeps/full_r3lam08_vs_r902_merged.json` | R3 n32 主结果 |
| `runs/S3_loop/sweeps/opendoor_valid_probe/summary.json` | OpenDoor 1 对有效 probe |
| `runs/S3_loop/sweeps/opendoor_r3_vs_r2_merged.json` | OpenDoor 8 对**无效**结果（wrong start），仅存档 |
| `logs/auto_ladder/r4_round904.log` | R4 当前日志 |
| `logs/auto_ladder/r3_round903.log` | R3 流水线日志 |
| `logs/auto_ladder/r2_round902.log` | R2 流水线日志 |
| `logs/r3_lambda08_joint.log` | R3 λ=0.8 joint 日志 |

---

## 11. 交接检查清单

- [ ] R4 是否到 84/84，WM/critic/joint 是否 PASS；
- [ ] R4 32 对是否完成，门控 `mean<0 / median<0 / better≥55% / falls≤R3` 是否过；
- [ ] 若 R4 失败：停在 R4，修当前轮，不要进 R5；
- [ ] OpenDoor 补到 ≥8 对有效（正确 row 373209），再判断；
- [ ] PPBox/Boxing/SitSofa/VisionNavi 各补 8–16 对；
- [ ] 所有非 DoubleDesk 评测必须显式传 `TASK` 和任务起点 row；
- [ ] 报告里不要把 `opendoor_r3_vs_r2_merged.json` 的 +0.084 当成 OpenDoor 结论；
- [ ] 报告里不要把 R3 写成“安全门通过”（1 fall）；
- [ ] 任何“提升”必须标注 n、CI、任务、是否 success；
- [ ] 若要加训练步数，先做 400 步对照，不要直接全轮加；
- [ ] 若要重新标定 hinge，补 v3 p90 λ，不要退回 v2 的 0.19/0.16/0.14。

---

## 12. 最短结论（给赶时间的人）

1. **R1 是唯一统计上站得住的轮次**（n44，均值 −0.096 m，CI 不跨 0）。
2. **R2、R3 在 DoubleDesk 上方向正确但证据弱**：R2 −0.043 m（18/32），R3 −0.060 m（22/32）；R3 CI 仍跨 0，且有 1 次摔倒。
3. **0.85 m 的全集平均不是纯漂移**：前 1000 步只有 0.34–0.41 m，后 40% 是“抓取失败后停在桌前、参考继续走”的约 2 m 平台。
4. **训练数据本身是失败轨迹**（B_903 DoubleDesk 终点误差 1.9–2.35 m），所以“多训”不能自动产生成功抓取/搬运。
5. **评测只覆盖了 DoubleDesk**；OpenDoor 只有 1 对有效 probe；其他四任务未评测。当前不能声称多任务提升。
6. **R4 正在跑**，配置 `E=1.6 / λ=0.70 / act=0.5 / 200 steps`；先看住 R4，再补多任务与成功数据。
