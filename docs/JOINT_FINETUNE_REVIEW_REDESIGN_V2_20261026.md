# 联合微调 + IsaacLab SuperMimic 训练：复核与重设计 v2（2026-10-26）

> 对象：`Restar7/comimic`（main 分支快照）+ 上游 `ZouYan93/trackerlab-supermimic`（main）。
> 输入：`docs/JOINT_FINETUNE_DIAGNOSIS_AND_DESIGN.md`（下称"原文档"）——**只作线索，不当结论**。
> 读者：执行者（DeepSeek）。本文第 4 节是逐步执行清单，每步有"改什么 / 怎么验 / 不过怎么办"。

**证据等级**（全文统一）：

- ✅ **代码已核实**：在仓库代码里直接看到，给出 `文件:行号`；
- 🟡 **需上机确认**：依赖本地改过的 `trackerlab-supermimic`、HumanoidArena provider、ckpt 里的 stats 等仓库外内容，我看不到，必须先跑第 4 节的探针；
- ❌ **原文档结论不成立**：附理由。

---

## 0. 结论速览

1. **原文档把"ft 臂行为退化"归因于"联合微调目标把行为训坏了"，方向对，但具体机制没找准。** 真正的机制是下面几个**口径/估计器错误**叠加。它们和"奖励饱和"无关，改奖励核救不回来：
   - **(J2) VLA 输出从未反归一化** ✅：GR00T 训练时动作经 min-max 归一化到 [-1,1]，但 `ha_vla_latent64_server.py` 和 `train_joint.py` 都把 head 的原始输出直接当 token/手指用。手指阈值 `>1.70` 在归一化空间里**永远到不了**，这一点很可能就是"手从不闭合"的直接原因（🟡 量级需上机读 stats 确认）。
   - **(J4) token 锚定和手部监督都是"对一次 FM 采样结果做 MSE"** ✅。这种做法会把条件分布压成均值（导航这种多模态行为就表现为"在目标之间打转"），这才是"被平均"的本质。正确做法是用 GR00T 自己的 flow-matching 损失做 BC。
   - **(J3) 训练用 2 步 Euler，服务端默认 4 步** ✅。梯度优化的是 2 步采样出来的东西，部署时跑的是另一条积分路径。
   - **(J5) 联合微调同时在冻结 WM 上更新 WBC（lr 1e-5）** ✅。你们自己在 `PLAN_HA_WBC_FINETUNE §7.5` 里已经证明"冻结 WM + 更新策略 = 模型利用"。更糟的是，评测时 ft 臂又换回原始 WBC。
   - **(J1) 联合训练没有导出 `SUPERMIMIC_HA_ALIGN=1`** ✅/🟡：`isaac_joint.py` 的 `default_q`/`action_scale` 都来自 `get_robot("g1")`，这个值随环境变量变化。所以原文档"联合训练不受 action_scale 影响"的说法**不成立** ❌。
2. **A1 的两臂都不是干净对照** ✅：base 臂是官方 VLA，输出 HA 标签 token 空间，但 WBC 吃的是端口 token 空间，两者相关只有 0.06–0.19；ft 臂的 WBC 训练时被改过、评测时没改。所以"base 走向桌子、ft 打转"不能归因于"奖励反传"。
3. **SuperMimic（Isaac 侧）训练这条线**：域差已实测（HA 上摔倒率 22.6% vs 自域 0.49%）；"冻结 WM 微调策略"已被证伪；"混合库 + WM 共训"是目前唯一健康的路线（HA 探针 0.541→0.694、摔倒率 0.226→0.046）。**继续这条路**，但要补三件事：执行器 spec 写进 ckpt、显式积分随训随标、HA 场景下的 GT-token 回放门禁。
4. **重设计核心**（§3）：
   - **先统一接口**：同一个 token 空间、同一个归一化口径、同一个 executor spec、同一个 FM 步数。
   - **主收益来自 "base_aligned"**：用端口 token 重标数据，再跑一次官方 GR00T 微调，用的是 FM 损失。这一步同时就是 A1 的正确对照臂。
   - **联合微调 v2 只做小幅精修**：FM-BC 正则（不是 sample-MSE）+ 冻结 WBC 与 WM + 扰动起点和 burn-in 历史 + H=16 + 单次采样 + 梯度累积。
5. **对期望要诚实**：可微 WM 这条梯度优化的是"跟踪录制示范动作的可执行性"，**教不会"去对的目标、去抓"**。任务成功率主要取决于 BC 数据质量和执行器能力。联合微调能做的是让 token 更容易被 WBC 执行好，所以门禁必须用闭环指标。

---

## 1. 复核范围

| 部分             | 看了什么                                                     | 看不到（→🟡）                                                 |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 联合微调         | `diff_vla_wbc/{isaac_joint,rollout,train,data,gr00t_fm,fm_grad}.py`、`scripts/{train_joint.py,launch_joint_ha.sh,ha_vla_latent64_server.py,build_port_tokens.py,probe_port_tokens.py}` | 各 run 的真实 `args.json`、日志                              |
| GR00T            | `Isaac-GR00T/gr00t/{configs/data/embodiment_configs.py,data/state_action/state_action_processor.py,model/gr00t_n1d7/gr00t_n1d7.py}` | `checkpoint-8000` 里的 `statistics`（min/max/q01/q99 的具体数值） |
| SuperMimic Isaac | 上游 `trackerlab-supermimic` 的 `runner.py / networks.py / mdp/supermimic.py / robot.py / robots.py`；`scripts/launch_sm_finetune.sh`；文档 `PLAN_HA_WBC_FINETUNE / HANDOVER_* / TOKEN_ANCHOR / SONIC_TOKEN` | 你们本地改过的版本：token 接口、`HA_ALIGN`、`DEFAULT_JOINT_POS_HA`、`SonicTokenBuilder`、`--freeze_wm`、显式积分；HA 的 `action_provider_supermimic.py` |

---

## 2. 问题清单

### 2.1 联合微调链路（J*）

| #      | 问题                                                         | 证据                                                         | 后果                                                         | 修法（§4 步骤）                                       |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------- |
| **J1** | 联合训练进程**没有**导出 `SUPERMIMIC_HA_ALIGN=1` / `SUPERMIMIC_ROBOT_USD` | `launch_joint_ha.sh:69-87`、`train_joint.py` 全文都没有；`isaac_joint.py:262-264` 用 `get_robot("g1").default_q`；`HANDOVER_CLOSEDLOOP_ACTION_SCALE §3`：默认 USD 无手 → `HA_ALIGN=0` | 若 WBC 训练用的是 HA 默认姿态，那么 WM 想象里每一帧 obs 的 `q − default_q` 都整体偏移（README 已有前车之鉴：偏 0.3 rad 会 1 秒倒地）；`reference_start` 里 `a_prev=(q_ref−default)/scale` 的口径也跟着错 | S0.1                                                  |
| J1'    | 文档之间对"HA 默认姿态"的说法互相矛盾                        | 原文档 §1.2 说"膝 0.42/踝 −0.23/肘 0.87"；`PLAN_HA_WBC_FINETUNE §4` 说"HA 默认姿态（−0.312/0.669/−0.363）"，而这组数恰好是上游 BeyondMimic 默认值（`robot.py:107-116`） | 没人能确定 WBC 训练时到底用的哪一套 → 🟡                      | S0.1：从 WBC run 的 launcher 日志/代码取真值写进 ckpt |
| J1''   | `COMIMIC_ACTION_SCALE=one` 这个结论是在 `HA_ALIGN=0` 的 BM 表下比出来的（spec 对比 ones） | `isaac_joint.py:265-287` 注释：spec 是 0.0745–0.5476（=BM 表）。HA 表（0.3789/0.75）从没被比过 | "ones 最好"可能只是"两个错的里面错得少的那个"                | S0.1 + S3.4（burn-in 让 a_prev 不再依赖换算）         |
| **J2** | **head 输出未反归一化**，训练和服务端都一样                  | `embodiment_configs.py:85-108`（motion_token / left_hand / right_hand，无 mean_std 键）→ `state_action_processor.py:404-415` 做 min-max 到 [-1,1] 并 clip；`ha_vla_latent64_server.py:205-213` 直接 `full[:, :64]`、`full[:, 64:78]`，手指阈值 1.70；`train_joint.py:348-362` 同样直接切片；MuJoCo 闭环却调用了 `processor.decode_action`（`closed_loop/code/closed_loop_mujoco.py:764`） | ① 手：归一化输出 ≤1 → `>1.70` 恒不成立 → 二值手恒张开；直通时只给到约 1/1.74 的幅度。② token：base 臂喂给 WBC 的是归一化后的 token。③ 训练时 anchor 和 hand 用**原始空间**标签去回归**归一化空间**的输出 → 第 0 步 `hand_mse=0.65` 的量级很可能就来自这里（`TOKEN_ANCHOR.md §10.4`），head 被迫整体换尺度 | S0.2                                                  |
| **J3** | 训练用 `--vlm-steps 2`，服务端默认 `--fm-steps 4`            | `launch_joint_ha.sh:77`；`ha_vla_latent64_server.py:296`；原文档 §5 的命令也是 `--fm-steps 4` | 反传优化的是 2 步 Euler 的终点，部署跑的是 4 步，head 被改过之后两者不等价 | S0.3                                                  |
| **J4** | anchor 和 hand 都是对**单次 FM 采样**做 MSE；token 和手各自采样一次，噪声不同 | `train.py:258-293`；`train_joint.py:348-359`（每次调用都新 `randn`），hand 分支再调一次 `_sample_full`（`train_joint.py:470-476`） | 对生成式 head 的样本做 MSE，在多模态条件下的最优解是条件均值 → 行为被"平均"（走到两个目标中间、打转）；手部被压向均值（多数帧为 0）；另外 1.6B head 每步多付一次前向+反传 | S3.3：改 FM-BC 损失、单次采样                         |
| **J5** | WBC 参数在冻结 WM 上被更新，评测又换回原 WBC                 | `train.py:116-133`（policy 参数进了 optimizer，`policy_lr=1e-5`）；原文档 §5 的评测用 `ckpt_010500_physint.pt` | ① 模型利用（`PLAN_HA_WBC_FINETUNE §7.5` 已实证）；② VLA 是和"被改过的 WBC"共同适配的，部署的却是另一个 → 训练和部署不一致 | S3.2：WBC 完全冻结（只让梯度穿过）                    |
| **J6** | 多卡时所有 rank 同 seed → 每张卡采同一批帧、同一份噪声       | `train_joint.py:222`（`torch.manual_seed(args.seed)`）、`:446-456`（`MotionBatchProvider(seed=args.seed)`）、`data.py:337-339` | 数据并行等于白算，有效 batch 仍是 2；目前主线 `NGPU=1` 不受影响，但 A2/A3 那些 2 卡跑法受影响 | S3.6                                                  |
| J7     | 想象起点 = **参考帧本身**，历史也是参考 + 换算出来的 a_prev  | `isaac_joint.py:438-473`（`state = rows[:, -1]`）            | 起点误差为 0，8 步内几乎不偏 → 好坏 token 的回报差很小（原文档 D1 的 12%）、存活≈1（D2）。**这才是 D1/D2 的根因，不是 exp 核饱和** | S3.4：扰动起点 + WBC 自身 burn-in                     |
| J8     | 奖励梯度只到 `tokens[:, :H]`（H=8），一个 chunk 有 40 帧     | `rollout.py:334-335`                                         | 闭环每 12 步重取一次 chunk（原文档 S8），实际执行的是前 12 帧，H=8 连执行段都覆盖不全 | S3.5：H=16                                            |
| J9     | 评测只有 2 batch × 2 = **4 个窗口**；平台判据就基于它        | `launch_joint_ha.sh:79`、`plateau_guard.sh`                  | token_mse / reward 的波动基本是噪声；平台判停不可靠          | S3.7                                                  |
| J10    | 目标函数本质是"让 WBC+WM 跟踪录制示范的执行动作"             | `MDP.reward` 对 `ref.f`（录制帧）                            | 最多把可执行性优化到示范水平，不会产生"去对目标/抓取"的新能力 → 期望与门禁要按这个来设 | §3.1                                                  |

### 2.2 IsaacLab SuperMimic 训练线（W*）

| #      | 问题                                                         | 证据                                                         | 建议                                                         |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **W1** | 执行器 spec（default_q / action_scale / kp-kd / USD）**全靠环境变量**，ckpt 里不存 | `HANDOVER_CLOSEDLOOP_ACTION_SCALE §3`；上游 `runner.py:426-436` 存的 ckpt 键里没有 spec | 训练端把 spec 快照写进 ckpt（`robot_spec`），所有消费方（evaluate / provider / isaac_joint）加载时逐项断言。provider 那把锁已经有了，但联合训练没有 → S1.1 |
| **W2** | 域差已实测：HA 上 `term_frac 22.6%`、`sim_r 0.541`；自域是 0.49% / 0.877 | `PLAN_HA_WBC_FINETUNE §7.9`                                  | 必须做 HA 适配，这一点结论成立                               |
| **W3** | "冻结 WM + 策略 RL"已证伪（`sim_r` 塌、`imag_rew` 反升）     | `PLAN_HA_WBC_FINETUNE §7.5`                                  | 永久禁止；联合微调里同样禁止更新 WBC（见 J5）                |
| W4     | "混合库（BONES+HA×20）+ WM 共训"有效但在收敛：0.680→0.694、0.064→0.046 | `PLAN §8.3/8.5`                                              | 继续这条线，加码方向见 S1.2（提高 HA 权重/lr、扰动与负载随机化） |
| **W5** | 两条 token 路线**互相打架**：`launch_sm_finetune.sh` stage2 让 WBC 吃 **HA 标签 token**；联合微调又把 VLA 拉向**端口 token** | `launch_sm_finetune.sh:8-12,46`；`TOKEN_ANCHOR.md`；`PLAN §2` 约束 #2 | **只能选一个 token 空间**。推荐端口空间（理由见 §3.2），stage2 的"标签 token WBC"停用 |
| W6     | 端口 token ≠ 标签 token 的根因没有定论：一种说法是"两个前端"（部署侧 1762 维含 SMPL/wrist/过去帧），另一种是端口装配的 bug | `HANDOVER_LATEST_FINDINGS §0.1`、`probe_port_tokens.py`      | 🟡 S0.4 用 HA raw 自带的 `encoder_input (1762)` 逐字段比对，十几分钟能判清，结果决定走 §3.2 的哪条路 |
| W7     | WM 一共训，"800 步速度标定"就失效                            | `PLAN §8.4`                                                  | 训练全程开显式积分（`SUPERMIMIC_EXPLICIT_INTEGRATION=1` 已在 `launch_sm_finetune.sh:29`），每个候选 ckpt 都过一遍 `eval_wm_velocity.py` 门禁，不再单独做"事后标定" |
| W8     | 训练里没有物体/负载/推扰；HA 任务要搬箱、抓杯                | 上游 `make_g1_cfg` 只有自身动力学                            | P2：手/腕负载质量随机化（0–2 kg）+ 躯干外力推扰（PPBox 这类任务要扛得住） |
| W9     | HA 闭环里没有 GT-token 回放的**上限门禁**；现有"真值 token 回放"是在接线 bug 修复前做的（51→165 步） | `HANDOVER_ISAAC_CLOSEDLOOP §10`                              | S1.3：每个 WBC 候选都跑"端口 GT token 开环回放"；另外 SONIC release decoder + GT `encoder_latent` 作为录制执行器的参照 |

### 2.3 原文档结论逐条校正

| 原文档                                                       | 判定            | 理由                                                         |
| ------------------------------------------------------------ | --------------- | ------------------------------------------------------------ |
| §1.1/交接文档："联合训练不受 action_scale 影响"              | ❌               | `isaac_joint.py` 的 default_q 和 action_scale 都依赖 `HA_ALIGN`（J1） |
| §2.2："不是 SuperMimic 的问题（两臂同一 WBC），是联合微调把 VLA 训坏了" | ❌（前提不成立） | ft 臂训练时 WBC 也在被改（J5）；base 臂 token 空间不匹配（§0.2）；两臂都不是干净对照 |
| D1："exp 核在小误差处饱和 → 换成 −e"                         | ❌ 归因错        | `exp(−e/σ²)` 在 e→0 时对 e 的斜率最大，并不饱和；对 Δ 的梯度变小是因为 e=Δ²、离最优点近，换成 −e 同样 ∝Δ。真正的原因是起点=参考、H=8（J7）。−e 还没有上界，在 WM 外推区会放大误差梯度 |
| 4.4-2："加世界坐标 root 项"                                  | ❌ 已存在        | `mdp/supermimic.py:177-178`：`e_rp/e_ro` 本来就是世界坐标，只有 body 项重锚（`:168-172`） |
| 4.4-4："对比白化 + softmax(advantage) 加权"                  | ❌ 不推荐        | 把 score-function 的权重套在 pathwise 梯度上，概念混用；成本 ×K；pathwise 下降噪靠 batch/累积就够 |
| 4.4-5："H 8→32"                                              | 🟡 部分对        | 方向对，但 32 步会把 WM 复合误差带进来；取 H=16（覆盖 12 步重取间隔）+ 可选 critic 自举 |
| D3/4.5："只在有手部动作的帧算 hand loss"                     | ❌               | 零手帧是合法标签（导航时就该张手）；mask 掉以后模型学不到"不抓时张开"。根因是 J2+J4；用 FM 损失 + 任务均衡采样 |
| D4/4.6："锚定退火 + action distill"                          | 🟡               | 锚定本身是 sample-MSE（J4），应换成 FM-BC 且权重恒定；action-distill 可作为可选的次要项（行为空间度量），但它同样是 sample-MSE，不能当主项 |
| 4.8："token_mse ≤ 0.02 做门禁"                               | ❌               | 它就是被训练的目标（TOKEN_ANCHOR §8.3 自己也承认），不能当独立证据；门禁改成闭环指标 |
| 4.6-3："闭环选 ckpt"                                         | ✅ 同意          | 保留，扩大样本量                                             |
| 4.5-3："抓取验收用 DoubleDesk/GrapCup"                        | ✅ 同意          | 保留                                                         |

---

## 3. 重设计

### 3.1 设计原则

1. **接口先于目标函数**：token 空间、归一化、executor spec、FM 步数、关节顺序，每一项都必须"写进 ckpt + 加载时断言 + 日志打审计行"。这一轮踩的坑全是口径问题。
2. **生成式 head 只用 flow-matching 损失做监督**（BC 正则）。sample-MSE 一律不当主项。
3. **执行器（WBC）和动力学（WM）在 VLA 微调时完全冻结**，只让梯度穿过。
4. **可微 WM 的作用定位为"可执行性精修"**，量级受控（梯度比监控），不承担学行为的职责。
5. **门禁用闭环**：执行器上限（GT token 回放）→ base_aligned → ft，逐级过线。

### 3.2 统一 token 接口（决策点，由 S0.4 决定）

| S0.4 结果                                                    | 走哪条                                    | 做法                                                         |
| ------------------------------------------------------------ | ----------------------------------------- | ------------------------------------------------------------ |
| **(a) 标签来自非 g1 分支（teleop/smpl，`encoder_index≠g1`）或依赖离线拿不到的输入** | **路线 II（默认推荐）**：VLA 改说端口语言 | 用端口 token 重标 HA 数据的 `action.motion_token`，跑官方 GR00T 微调，得到 **base_aligned**（§3.3）。WBC 不动 token 接口 |
| (b) 标签就是 g1 分支，只是端口装配错了（步长/偏移/关节序/anchor 约定） | 路线 I'：修端口                           | 修 `SonicTokenBuilder` 直到 port==label 逐位一致 → WBC 用修好的 builder 重训（混合库）→ 官方 VLA 天然就是合法的 base 臂；仍建议补 §3.3 来修 J2 |

为什么默认选 II：WBC 的 token 是在训练环里用"未来参考 + 机器人朝向"实时算的（B 模式），BONES-SEED 这 263 h 只能产出端口空间的 token。HA 标签空间只有 3.5 h，撑不起一个通用 WBC。stage2（`launch_sm_finetune.sh 2`）就是在走这条撑不起来的路，停掉。

### 3.3 阶段 A：base_aligned（主要收益来源，也是 A1 的正确对照臂）

- **数据**：`data/ha_sessions` → `data/ha_sessions_port`（视频硬链接）。parquet 里的 `action.motion_token` 换成 `data/port_tokens/ha_tokens/<clip>.npy`，手部列不变，**重算 `stats.json` 里 motion_token 的统计量**。
- **训练**：官方 `examples/finetune.sh`（就是 `launch_ha_official_20k.sh` 那套），从 `checkpoint-8000` 热启，4k–8k 步，全局 batch 32–64。
  - ⚠️ **必须让新 stats 覆盖旧 stats**：`state_action_processor.py:128-143` 写明，ckpt 里已有 `unitree_g1_sonic` 的 stats 时，新 stats 默认会被**丢弃**（只打一行 warning）。要打开 `override_pretraining_statistics=True`（或等价开关），并在训练后检查 ckpt 里存的 motion_token min/max 已经换成端口 token 的统计量。
- **服务端**：先 decode 再输出（S0.2）；`--fm-steps` 与训练配置一致（S0.3）。
- **验收**：离线 held-out episode 上，`decode(sample)` 与端口 token 的 MSE 明显低于官方 ckpt 的 0.17；闭环见 §3.7。

### 3.4 阶段 B：WBC 的 HA 适配（Isaac 训练线，与 A 并行）

- 沿用 `PLAN §8` 的配方：obs93 手术件热启、库 = BONES-SEED + HA×20、**WM 共训**、显式积分开、`MAX_EPISODE_STEPS=0`、`HA_ALIGN=1`。
- 加码（二选一先试）：HA 权重 ×40 / lr 3e-5；P2 再加负载/推扰随机化（W8）。
- 每个候选 ckpt 都要：①写入 `robot_spec` 和 `wm_integration`；②跑 HA-only fresh 探针 + BONES 子集探针（防遗忘）；③WM 速度门禁；④HA Isaac 场景 GT-token 开环回放（S1.3）。
- 端口 token 的 builder 不变，所以阶段 A 的重标数据在 WBC 换代后依然有效。

### 3.5 阶段 C：联合微调 v2（小幅精修）

**计算图**（与现在相同的部分不再赘述）：

```
obs → GR00T(backbone 冻结) → head → x_norm (B,40,132)        ← 每步只采样一次，K = 服务端 fm_steps
x_raw = codec.decode(x_norm)                                  ← 可微反归一化（S0.2）
token = x_raw[..., :64]  →  WBC(冻结) → WM(冻结) → reward     ← 起点：扰动 + burn-in（下文）
L_bc  = GR00T FM 损失(head, 目标 = encode(端口 token ⊕ 手部标签))   ← 替代 token_anchor 和 hand MSE
```

**损失**：

```
L = λ_bc · L_FM                                # 主项，恒定权重（不退火）
  + λ_r  · ( − J / (H · MAX_STEP_REWARD) )     # 归一化回报
  + λ_s  · L_state(pos, ori, q)                # 可选，小权重；速度通道不进
  [+ λ_a · ‖π(token_vla) − π(token_port)‖²]    # 可选次要项，WBC 冻结
J = Σ_{t<H} γ^t · c_t · r_t   （可选 + γ^H · c_H · V_ψ(s_H)，V 取自 WBC 的 last.pt critic，见 S3.5）
```

**超参起点**：H=16；batch 4/卡 × 梯度累积 4；head lr 1e-5；`λ_bc=1`。`λ_r` 用**梯度比**来定：每 50 步打印 `‖∇_head L_r‖ / ‖∇_head L_bc‖`，把它控制在 0.1–0.5 之间。扫 `λ_r ∈ {0, 0.1, 0.3, 1.0}`，其中 `λ_r=0` 的那一臂就是"继续 BC"的对照。

**起点（治 J7、J1''）**：

1. 参考帧 f 往前退 K=10 帧到 f₀（夹到 clip 起点），先用参考构造初始历史；
2. 对 f₀ 的状态加扰动：root xy ±5 cm、yaw ±0.1 rad、关节 q ±0.05 rad、qd ±0.3 rad/s；
3. `no_grad` 下让**冻结的 WBC** 拿 f₀..f₀+K−1 的端口 GT token 在 WM 里走 K 步。这样得到的状态、历史和 a_prev 都是 WBC 自己产生的，和换算口径无关；
4. 从这个状态开始，用 VLA 的 token 做 H 步有梯度的 rollout。

### 3.6 手部

- FM-BC 损失里把手部 14 维一起监督（GR00T 本来就是这么训的），**不 mask 零手帧**。
- 采样按任务均衡：每个 batch 内 7 个任务等概率。可以写一个 `TaskBalancedSampler` 给 `MotionBatchProvider` 用；阶段 A 走官方 dataloader，用 `EPISODE_SAMPLING_RATE` 和任务权重实现。
- 服务端先 decode 再做阈值判断；直通 finger target 用 decode 之后的值。侧内 7 维顺序按原文档 §4.5-1 校准（��）。

### 3.7 A1 实验设计（重排）

| 臂    | VLA                               | 执行器                    | 用途                                       |
| ----- | --------------------------------- | ------------------------- | ------------------------------------------ |
| O1    | GT 端口 token（录制回放）         | WBC_v2                    | **执行器上限**。这一臂都过不了，就别训 VLA |
| O2    | GT `encoder_latent`               | SONIC release decoder     | 录制执行器的参照（HA 原生链路）            |
| R     | 官方 checkpoint-8000（decode 后） | SONIC release decoder     | 官方系统基线                               |
| **B** | **base_aligned**（阶段 A）        | WBC_v2                    | **正确的对照臂**                           |
| **F** | base_aligned + 联合 v2（阶段 C）  | WBC_v2（与 B 逐字节相同） | 唯一变量 = 奖励反传                        |

- 任务：PPBox（导航+不摔）、DoubleDesk、GrapCup（抓）+ 其余 3 个；每个任务 3 seeds（选 ckpt 时 1 task × 2 seeds，正式表 6 × 3）。
- 指标：成功率、`max_reward`、摔倒率、到目标最小距离、手闭合事件数。

---

## 4. 执行清单（给 DeepSeek，按顺序；每步都有门禁）

> 通用规则：①每个改动都要打一行 `[AUDIT]` 日志，并把对应字段写进 ckpt/args.json；②新产物用新目录名，不覆盖旧文件；③任何门禁没过，就停下回报数字，不要往下跑；④停任务按 PID/进程组，不要用 `pkill -f`。

### S0 审计与探针（不训练，约 1–2 h，最高优先级）

**S0.1 执行器 spec 一致性（J1）**

1. 从 WBC 训练 run（`trackerlab-supermimic/logs/bones_seed_token_ha_v2/` 的 launcher 日志 / `args.json`，以及本地 `trackerTask/.../robot.py`）确定训练时的 `HA_ALIGN`、`default_q`(29)、`action_scale`(29)、kp/kd、USD 路径。

2. 写脚本 `scripts/stamp_robot_spec.py`：给现有 WBC ckpt 加一个 `robot_spec` 键（另存为新文件，例如 `ckpt_010500_physint_spec.pt`）：

   ```python
   ck = torch.load(src, map_location="cpu", weights_only=False)
   ck["robot_spec"] = {"ha_align": 1, "usd": USD, "default_q": [...29], "action_scale": [...29],
                       "kp": [...29], "kd": [...29], "joint_names": [...29]}
   torch.save(ck, dst)
   ```

3. `scripts/train_joint.py` **最顶部**（任何 `diff_vla_wbc`/`trackerLab` import 之前）：

   ```python
   import os
   os.environ.setdefault("SUPERMIMIC_HA_ALIGN", "1")
   os.environ.setdefault("SUPERMIMIC_ROBOT_USD", "<与 WBC 训练相同的 USD>")
   ```

   `launch_joint_ha.sh` 也 export 一遍（双保险）。

4. `diff_vla_wbc/isaac_joint.py`，在 `IsaacJointBundle.__init__` 里取得 spec 之后：

   ```python
   rs = ck.get("robot_spec")
   dq = torch.tensor(self.spec.default_q); sc = torch.tensor(self.spec.action_scale)
   print(f"[AUDIT][spec] HA_ALIGN={os.environ.get('SUPERMIMIC_HA_ALIGN')} "
         f"default_q[knee]={dq[3]:.3f} action_scale mean={sc.mean():.4f} max={sc.max():.4f}", flush=True)
   if rs is not None:
       assert torch.allclose(dq, torch.tensor(rs["default_q"]), atol=1e-6), "default_q 与 WBC 训练不一致"
       assert torch.allclose(sc, torch.tensor(rs["action_scale"]), atol=1e-6), "action_scale 与 WBC 训练不一致"
   elif os.environ.get("COMIMIC_ALLOW_NO_SPEC") != "1":
       raise SystemExit("WBC ckpt 没有 robot_spec；先跑 scripts/stamp_robot_spec.py")
   ```

5. **门禁**：在 HA_ALIGN=0 和 =1 两种设置下各打印一次 default_q / action_scale，和第 1 步的真值对表。把这张对表回报。

6. 在正确的 spec 下重跑 `scripts/probe_action_decomp.py`，比较 {HA spec, ones} 两种换算的参考动作 rollout reward / 末步误差。回报（这一步只作诊断；S3.4 的 burn-in 会让这个换算不再关键）。

   **S0.2 动作归一化（J2）**

7. 加载 `checkpoint-8000` 的 processor，打印 `processor.state_action_processor.use_percentiles / clip_outliers`，以及 `norm_params["unitree_g1_sonic"]["action"][k]["min"/"max"]`（k ∈ motion_token、left_hand_joints、right_hand_joints），同时打印 HA 数据里 `g1_motion_token_64`、`g1_hand_joints_14` 的实际范围。

8. 写 `diff_vla_wbc/action_codec.py`（可微、torch 实现）：

   ```python
   class ActionCodec:
       KEYS = ("motion_token", "left_hand_joints", "right_hand_joints")
       def __init__(self, processor, tag="unitree_g1_sonic", device="cuda"):
           sap = processor.state_action_processor
           cfg = processor.modality_configs[tag]["action"]
           assert not getattr(cfg, "mean_std_embedding_keys", None), "有 mean_std 键，需另写分支"
           p = sap.norm_params[tag]["action"]
           lo = np.concatenate([np.asarray(p[k]["min"]).reshape(-1) for k in self.KEYS])
           hi = np.concatenate([np.asarray(p[k]["max"]).reshape(-1) for k in self.KEYS])
           self.lo = torch.tensor(lo, dtype=torch.float32, device=device)
           self.rng = torch.tensor(np.maximum(hi - lo, 1e-8), dtype=torch.float32, device=device)
           self.dim = self.lo.numel()              # 期望 78
       def decode(self, x):                        # x: (..., >=78) 归一化空间
           y = x[..., : self.dim].float()
           return (y + 1.0) * 0.5 * self.rng + self.lo   # 训练时不 clamp，保留梯度；单独记录越界比例
       def encode(self, y):                        # y: (..., 78) 原始空间
           return (2.0 * (y.float() - self.lo) / self.rng - 1.0).clamp(-1, 1)
   ```

9. **单测（门禁）**：随机 x∈[-1,1]，`codec.decode(x)` 要和 `processor.decode_action(x, EmbodimentTag("unitree_g1_sonic"))` 拼出的 78 维逐元素一致（误差 <1e-5）；`encode(decode(x))≈x`。

10. `ha_vla_latent64_server.py`：加 `--decode {codec,none}`，默认 `codec`。`chunk64` 和 `fingers` 都取 decode 之后的值；启动时打印 `[AUDIT][server] decode=codec fm_steps=K`。**旧的联合微调 ckpt（anchor4000、vla2000）是在"原始空间回归"下训的，必须用 `--decode none` 评测**，并在结果目录名里标出来。

11. **门禁**：用官方 ckpt 在 3 帧 HA 数据上推理，打印 decode 前后手指的 max，以及 `>1.70` 的比例，和录制标签的 `vla_action_hand_binary` 对比。回报。

    **S0.3 FM 步数一致（J3）**

- 训练端 `--vlm-steps` 写进 `vla_*.pt`（`{"fm_steps": K}`）；服务端加载时读出来，与 `--fm-steps` 不一致就报错（留 `--allow-fm-steps-mismatch` 只给 A/B 用）。

- 本轮统一用 K=4（与官方推理一致）。显存不够就用 `--fm-grad-backend adjoint` 或 `checkpointed`（`fm_grad.py` 现成；backbone 冻结时 adjoint 合适）。

  **S0.4 token 空间根因（W6，决定 §3.2 走哪条）**

1. 从 `data/ha_raw/<task>/*.npz` 取 3 条，读 `encoder_input (T,1762)` 和 `encoder_latent (T,64)`。

2. 打印 `encoder_input[:, 0]`（`encoder_index`，看是 g1/teleop/smpl 哪个分支）。

3. 如果是 g1 分支：取 `encoder_input[t, 4:584]` reshape 成 (10,58)，拆出 dof_pos(29)、dof_vel(29)；取 `[601:661]` reshape 成 (10,6) 作 anchor（偏移 +1 的坑见 `SONIC_TOKEN.md §1`）。再和 `SonicTokenBuilder` 在同一帧装出来的 640 维逐字段比，看帧步长、未来帧 vs 过去帧、关节顺序、anchor 约定差在哪。

4. **判定**：非 g1 分支或输入不可离线复现 → 路线 II；g1 分支且能修到逐位一致 → 路线 I'。回报判定依据（字段级差异表）。

   **S0.5 多卡 seed（J6）**

- `train_joint.py`：先广播参数，然后 `torch.manual_seed(args.seed + 1000*rank)`；`MotionBatchProvider(seed=args.seed + 1000*rank)`；eval provider 的 seed 保持固定，只在 rank0 上跑。
- **门禁**：2 卡 smoke，打印各 rank 第 1 步的 `frames`，确认不同。

### S1 WBC 训练线（Isaac；与 S2 并行）

**S1.1 spec 写进训练端（W1）**：本地 `runner.py` 保存 `ckpt_*.pt` / `last.pt` 时加 `robot_spec`（字段同 S0.1），加上已有的 `wm_integration`。`evaluate.py` 和 provider 加载时断言。

**S1.2 继续混合库 + WM 共训（W4）**：

```
RESUME=<PLAN §8 的 ckpt_012250>  OBS_MODE=paper_obs93  FREEZE_WM=0  NO_CRITIC=1(热启 critic 时)
LIBRARY=bones_seed_plus_ha_x40.pt   # 或 x20 + LR=3e-5，两个格子二选一先跑
MAX_EPISODE_STEPS=0  HA_ALIGN=1  SUPERMIMIC_EXPLICIT_INTEGRATION=1  CKPT_EVERY=250
```

每 500 iter 对候选 ckpt 跑：

- HA-only fresh 探针（同 PLAN §7.8 协议）：目标 `sim_r ≥ 0.80`、`term_frac ≤ 0.02`；

- BONES 子集探针：`sim_r ≥ 0.85`（防遗忘）；

- `scripts/eval_wm_velocity.py`：`qd_corr ≥ 0.9`、幅值比 ∈ [0.9, 1.1]。

  **S1.3 执行器上限门禁（W9，O1/O2）**：

- O1：新写 `scripts/ha_port_token_server.py`（照 `ha_gt_token_server.py` 改）。按 episode 回放 `data/port_tokens/ha_tokens/<clip>.npy`，HA Isaac 闭环 + WBC 候选，跑 PPBox/DoubleDesk 各 3 seeds。看存活步数、是否跟着录制轨迹走到目标附近。

- O2：`HA_ALLOW_SONIC=1` + release decoder + `ha_gt_token_server.py`（`encoder_latent`）。

- **门禁**：O1 在 PPBox 上 300 步不摔（≥2/3 seeds），并能走到录制目标附近。过不了就回到 S1.2 加码，不要进 S3。

### S2 阶段 A：base_aligned（路线 II）

1. `scripts/relabel_sessions_port.py`：遍历 `data/ha_sessions/*`，找到 episode 对应的 clip 名（**先验证映射**：抽 5 条，比较 parquet 的 state 关节列和 npz `joint_pos` 在同一帧是否一致、帧数是否相等），把 `action.motion_token` 换成端口 token，写到 `data/ha_sessions_port/`（视频硬链接），重算 `meta/stats.json` 和 `episodes_stats.jsonl` 里 motion_token 的 min/max/mean/std/q01/q99。
2. 训练：`BASE=runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 ROOTS=<port roots> MAX_STEPS=6000 RUN=groot_sonic_vla_ha_port bash scripts/launch_ha_official_20k.sh`，并打开 **override stats**（在 `examples/finetune.sh` / 数据集配置里找 `override_pretraining_statistics`，没有就在 processor 加载后显式 `set_statistics(..., override=True)`）。
3. **门禁**：
   - 训完的 ckpt 里 motion_token 的 stats 必须等于端口数据重算出来的值（打印对表）；
   - 留 10% episode 做 held-out：用 4 步采样，`MSE(codec.decode(x)[..., :64], port)` 明显低于官方 ckpt；手指 decode 后 `>1.70` 的比例与标签一致（±5%）；
   - 闭环 B 臂（base_aligned + 当前最好的 WBC）跑 PPBox/DoubleDesk × 3 seeds，回报。

### S3 阶段 C：联合微调 v2

**S3.1 新入口**：复制 `scripts/train_joint.py` → `scripts/train_joint_v2.py`，旧脚本不动，便于对照复现。

**S3.2 冻结执行器**：

```python
for p in bundle.policy.parameters(): p.requires_grad_(False)
bundle.policy.eval(); freeze_world_model(bundle.wm)
# JointTrainer: optimizer 只放 fm_parameters（adapter 在 token 路线不用，也不放）
```

**门禁**：训练结束后 `joint_*.pt["pi"]` 与源 WBC ckpt 逐张量相等（`torch.equal`）。

**S3.3 单次采样 + FM-BC**：

```python
# 每步：backbone 冻结 → 只算一次，在 no_grad 下
bb_in, act_in = vlm.prepare_input(dict(inputs))
with torch.no_grad():
    bb_out = vlm.backbone(bb_in)
# (a) 有梯度的采样：喂 BatchFeature 的浅拷贝（process_backbone_output 会原地改写字典）
x_norm = sample_fm_with_grad(vlm.action_head, BatchFeature(dict(bb_out)), copy_action_input(act_in),
                             initial_noise=noise, num_steps=K, grad_backend=args.fm_grad_backend)
x_raw = codec.decode(x_norm)
tokens = x_raw[..., :64]
# (b) FM-BC 损失（GR00T 原生）：目标 = encode(端口 token ⊕ 手部标签)，pad 到 head_action_dim
tgt = torch.zeros(B, H_chunk, head_dim, device=dev)
tgt[..., :78] = codec.encode(torch.cat([port_tok, hand14], -1))
mask = torch.zeros_like(tgt); mask[..., :78] = 1.0
act_bc = copy_action_input(act_in); act_bc["action"] = tgt; act_bc["action_mask"] = mask
L_fm = vlm.action_head(BatchFeature(dict(bb_out)), act_bc)["loss"]
```

注意：`action_head.forward` 会原地 `view` 掉 `action_input.state`（`gr00t_n1d7.py:214`），所以 (a) 和 (b) 各用一份 `copy_action_input`（浅拷贝字典 + `state.clone()`）。**删掉**旧的 `token_anchor` 和 `fm_full_sampler` 手部分支。

**S3.4 起点 burn-in**（§3.5 那 4 步）：在 `MotionBatchProvider` 里新增 `sample_burnin(K=10, perturb=...)`，用 `bundle.step` 和 `bundle.act` 在 `torch.no_grad()` 下走 K 步，返回 `start_state / history / prev_action / references(从 f 开始)`。
**门禁**：用 GT 端口 token 跑 H=16 的想象 rollout，和"起点=参考"相比：reward 均值应当下降（说明有误差可修），存活率 ≥0.95（说明 WBC 救得回来），回报两组数。

**S3.5 回报与可选 critic**：

- H=16，`tokens[:, :16]`；回报除以 `H·MAX_STEP_REWARD` 归一化；

- 可选：如果 WBC 的 `last.pt` 有 critic（上游 `runner.py:253` 的 bootstrap 用法），加 `γ^H · c_H · V(s_H)`。critic 输入要 GT goal（680 维，从 motion 库取）。**先做 A/B**：critic 版和非 critic 版在 S3.7 的闭环选择上谁更好就用谁；没有 critic 就跳过。

  **S3.6 batch / 累积 / 梯度比**：`--batch-size 4 --grad-accum 4`（需要实现：累积 k 次 backward 再 sync、clip、step）；每 50 步打印 `grad_ratio = ‖∇L_r‖/‖∇L_fm‖`（在 head 参数上分别 `autograd.grad`，只在打印步做）。

  **S3.7 评测与选 ckpt**：

- 固定 eval：64 个窗口（不是 4 个），报 `L_fm(val)`、想象回报（burn-in 起点）、`decode` 之后 token 与端口的 MSE（只作诊断）；

- 每 500 步一次 mini 闭环（PPBox + DoubleDesk，各 1 seed，约 8 min），**按闭环挑 ckpt**；

- 扫 `λ_r ∈ {0, 0.1, 0.3, 1.0}`，每格 2000 步，`λ_r=0` 是对照。

  **S3.8 门禁**：F 臂在 6 任务 × 3 seeds 上相对 B 臂成功率不降，且（摔倒率↓ 或 到目标最小距离↓ 或 抓取任务手闭合事件↑）里至少一项显著改善。否则结论就是"奖励反传无增益"，如实上报。

### S4 A1 正式表

按 §3.7 的五臂跑。所有臂的 `[AUDIT]` 行（spec / decode / fm_steps / WBC ckpt sha256）统一汇总进 `runs/A1_v2/audit.json`。

---

## 5. 不要做的事

1. ❌ 对 FM 采样结果做 MSE 当主监督（token_anchor / hand MSE 这类）。
2. ❌ 联合微调时更新 WBC 或 WM；或者训练用一个 WBC、评测换另一个。
3. ❌ 训练和服务端的 `fm_steps` 不一致，或者一边 decode 一边不 decode。
4. ❌ 在冻结 WM 上对 WBC 做策略 RL（已证伪）。
5. ❌ 同时维护两套 token 空间（stage2 标签 token WBC + 端口 token VLA）。
6. ❌ 拿 `token_mse` 或想象回报当门禁，只看闭环。
7. ❌ 把 `--decode none` 下的旧 ckpt 结果和新口径结果混在一张表里。
8. ❌ 改奖励核为 `−e`、加 softmax 对比加权（理由见 §2.3）。

---

## 6. 回报模板（每完成一步贴一次）

```
[S0.1] HA_ALIGN 训练真值=?  default_q(knee/ankle/elbow)=?/?/?  action_scale mean/max=?/?
       joint 进程实测: HA_ALIGN=? 是否一致=?   probe(HA spec vs ones): reward=?/?
[S0.2] use_percentiles=?  motion_token lo/hi 范围=?  hand lo/hi=?  codec 单测=pass/fail
       官方 ckpt 手指 max(decode 前/后)=?/?  >1.70 比例(前/后/标签)=?/?/?
[S0.4] encoder_index=?  判定=路线 II / I'   字段差异表: ...
[S1.x] ckpt=? HA sim_r/term=?/?  BONES sim_r=?  qd_corr/ratio=?/?  O1 PPBox 存活=?/?/?
[S2]   stats 覆盖=yes/no  held-out MSE(官方→aligned)=?→?  B 臂闭环: ...
[S3.x] λ_r=?  grad_ratio=?  L_fm(val)=?  mini 闭环: ...
```

---

# 附录 A：执行反馈（2026-10-26，S0 已完成）

> 本附录是**执行侧的回填**：S0 审计的实测结果、对 S1/S2/S3 的具体影响、以及各阶段当前状态。
> 详细探针记录见 `docs/S0_AUDIT_REPORT_20261026.md`。

## A.1 S0 五项的实测回报（按 §6 模板）

```
[S0.1] HA_ALIGN 训练真值=1（launcher 日志）  default_q(knee/ankle/elbow)=0.420/−0.230/0.870
       action_scale mean/max=0.3789/0.7500
       joint 进程实测: HA_ALIGN=1（已锁） 是否一致=是 → [AUDIT][spec] robot_spec 对齐 PASS
       ⏳ probe(HA spec vs ones) 尚未重跑（要 GPU）
[S0.2] use_percentiles=True  clip_outliers=True
       motion_token 原始范围=[−0.6875, 0.6250]（逐维）｜hand 原始范围=±1.740
       codec 单测=PASS（decode vs processor.decode_action: max|Δ|=1.19e-07）
       官方 ckpt 手指 max(decode 前/后)=待对照｜>1.70 比例(前/后/标签)=0 / 0 / 0.020
       实跑 tok_absmean=0.132（录制 latent 0.127）→ decode 生效
[S0.4] encoder_index(cols1:4)=0,0,0（录制帧）｜判定：**标签空间 = encoder latent（逐位相等）**
       端口 token vs 标签: MSE=0.0854, corr=0.0568 → **端口装配错**
       字段布局：encoder_input[4:584]=10×58 参考 qdq；[595:655]=anchor 窗口（与 encoder_anchor_window Δ=0）
[S1.x] 未开始（等 builder 修好）
[S2]   未开始；**依据 S0.4 降级为可选**
[S3.x] 未开始
```

## A.2 对 S1/S2/S3 的影响（相对 §3/§4 的增量）

### A.2.1 新增依赖链（顺序不可乱）
```
S0 ✅
 └─> 0. 修 pin（否则闭环数字不可用）
       └─> 1. 修 builder：SonicTokenBuilder 在 HA 上复现 encoder_latent（逐位）
             ├─ 失败 → 走路线 II（重标端口 + VLA 微调）→ S2 恢复原样
             └─ 成功 → 2. S1：WBC 用"修好的 builder + HA 标签 token"混合库重训
                       └─> 3. O1 门禁（GT 标签 token 回放）
                             ├─ 不过 → 回 S1 加码
                             └─ 过 → 4. S3 v2（FM-BC + decode + K=4 + H=16 + 冻结）
```

### A.2.2 S1（WBC 训练线）
| 项 | 变化 |
|---|---|
| spec | ✅ 每个 ckpt 必须写 `robot_spec`（工具：`scripts/stamp_robot_spec.py`）+ `wm_integration` |
| **token 源** | ❗新增硬前置：**builder 必须先修**；修好前只做"HA-only 标签 token"版（699 条），BONES-SEED 暂不参与 |
| 数据 | 混合库不变；HA 部分**必须用标签 token**（npz 的 `g1_motion_token_64`） |
| 门禁 | O1 改成 **GT 标签 token 回放**（不是端口 token） |

### A.2.3 S2（base_aligned）—— 降级为可选
| 原计划 | 现在 |
|---|---|
| 重标 `action.motion_token` → 端口 token，重算 stats，官方 GR00T 4–8k 步微调 | ❌ **不需要**：标签本来就是 encoder latent，VLA 已在该空间 |
| base_aligned 作为 A1 对照臂 | ✅ **官方 checkpoint-8000 + decode + 修好的 WBC** 即合法 base 臂 |
| 收益 | 省掉重标脚本 + stats + 1–2 天微调 |
| 代价 | ft 臂必须建立在"修好的 builder"上，否则两臂仍不同空间 |

> builder 修不出来 → 退回复核路线 II（S2 恢复原样，VLA 需重标重训）。

### A.2.4 S3（联合微调 v2）
沿用 §3.5，但明确八项：
1. 采样后 **`codec.decode()` 再喂 WBC**；
2. 监督改 **FM-BC**（`codec.encode(标签 token ⊕ 手部标签)`，归一化目标），删掉 sample-MSE 的 anchor/hand 主项；
3. K=4（训练=部署）；
4. burn-in 起点（扰动 + WBC 自走 K 步）；
5. H=16；
6. WBC 与 WM **完全冻结**；
7. 锚定目标改 **标签 token**（npz），不是端口侧车；
8. `--action-scale` 在 **HA spec** 下重测 {ones, spec}；奖励核不改，归一化回报 + 梯度比 0.1–0.5；按闭环选 ckpt。

## A.3 各阶段当前状态

| 阶段 | 状态 | 已完成 | 待做 |
|---|---|---|---|
| **S0 审计** | ✅ 完成 5/5 | spec / codec / fm_steps / token 根因 / 多卡 seed | `--decode none` 对照（被 pin 阻塞）、spec 下重跑 `probe_action_decomp`、2 卡 seed smoke |
| **0. 修 pin** | 🔴 阻塞 | 已定位：`sim_eval_vla.py` 08:01 被外部改动（仓库未纳入 git，无法审查） | 确认改动意图；有意则 `pin_ha_closedloop.py --write`，否则回滚 |
| **1. 修 builder** | ⬜ 未开始 | S0.4 已给出布局证据（anchor 段 Δ=0；qdq 段 Δ=0.167） | 对齐帧步长/顺序 → HA 上逐位复现 `encoder_latent` |
| **S1 WBC 重训** | ⬜ 未开始（等 builder） | spec 工具、混合库配方（`PLAN §8`） | 混合库 + HA×20/40 + 显式积分 + `MAX_EPISODE_STEPS=0` + spec 落盘 |
| **S2 base_aligned** | 🟡 降级为可选 | — | 仅当 builder 修不出来才执行 |
| **S3 联合 v2** | ⬜ 未开始 | 代码骨架（`train_joint.py` 可复制为 v2） | 按 A.2.4 八项改；先跑 λ_r=0 的 BC 对照臂 |
| **A1 正式表** | ⬜ 未开始 | 五臂设计（O1/O2/R/B/F） | 等 S1/S3 |

## A.4 已落地的 S0 代码/产物（可直接复用）

| 产物 | 路径 | 作用 |
|---|---|---|
| spec 戳 + 断言 | `scripts/stamp_robot_spec.py`、`diff_vla_wbc/isaac_joint.py`、`scripts/train_joint.py` 顶部 env 锁 | J1 |
| 可微归一化 codec | `diff_vla_wbc/action_codec.py`（`python -m diff_vla_wbc.action_codec` 自测） | J2 |
| 服务端 decode 开关 | `scripts/ha_vla_latent64_server.py --decode {codec,none}` + 审计行 | J2 |
| FM 步数一致性 | `vla_*.pt` 的 `fm_steps`、`export_vla_ckpt` 的 `vla_meta.json`、服务端校验 | J3 |
| 带 spec 的 WBC | `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint_spec.pt`（+`.spec.json`） | S1/S3 直接可用 |
| 多卡 seed | `train_joint.py` 的 `seed + 1000*rank` | J6 |

## A.5 旧数字的处理（重要）

- 旧 ft ckpt（`anchor4000`、`vla2000`）是在**未 decode** 的 token 上评测的 → **作废**；
- 按复核 §5.7：这些 ckpt 只能用 `--decode none` 复现旧结果，**不得与新口径混表**；
- pin 修好前的闭环数字同样不得与 07:4x 之前混表。

## A.6 S2 判定：**退役**（不是"可选"）—— 标签 token 已在环逐位复现

> 执行反馈。判据：`docs/S2_BASE_ALIGNED_PLAN_20261026.md` v2 +
> `runs/S2_gates/g0b_token_reproduction.json`（探针 `scripts/probe_token_reproduction.py`）。

A.2.3 把 S2 降级成"仅当 builder 修不出来才执行"。**这个触发条件已经有答案：builder 修得出来，
只差三处口径。** 用 `G1MotionEncoder`（权重取自部署 ONNX，另有 pytest 与 onnxruntime 逐位对拍）
实测 4 条 clip：

| 实验 | 做法 | 结果 |
|---|---|---|
| **A** | 录制 `encoder_input` 的 `[4:584]`+`[601:661]` → encoder | 与 `encoder_latent` **exact_frac = 1.0000**（个别帧差 1 个 FSQ 格 6.2e-2） |
| **B** | 拆 command 的 580 维 | 布局 = **字段主序** `[pos×10, vel×10]`；10 份 pos 全 = `human_joint_pos[t]`（**当前**参考帧）、10 份 vel 全 = 当前参考速度；anchor 60 维 = 10 份同一个**在环相对朝向**，且**就是 `consumed_anchor_rot6d`**（mean|Δ| ≤1e-4；用它重建 token 的 exact 0.9996–0.9999） |
| **C** | 只用"当前参考帧×10 + anchor×10"重建 | 与 `encoder_latent` **exact_frac = 1.0000** ⇒ **训练环里可复现** |
| **D** | 端口口径（未来 10 帧升序 + `PERM_MJCF_TO_ISAACLAB` + 恒等 anchor） | exact ≈ 0.09–0.12；与端口侧车的 corr(侧车, label) = 0.057–0.192 完全对应 |

⇒ **判定**：
1. **S2 退役**（重标端口 + VLA 微调没有收益）。标签空间才是 VLA 的原生空间，WBC 侧复现它只要
   "当前帧复制 10 份 + 不置换关节 + 用仿真机器人朝向算 anchor"；
2. 端口侧车（`data/port_tokens` / `build_port_tokens.py` / `SonicTokenBuilder` 的窗口·置换·anchor 三处）
   **退役**；`scripts/relabel_sessions_port.py`、`data/ha_sessions_port/`、`scripts/launch_ha_port_6000.sh`
   **保留为唯一 fallback**（触发条件见 S2 v2 §3），**不跑**；
3. **给 S1/S3 的两条硬约束**：
   - 在环 token = **当前帧复制**（不是未来窗口）；HA 数据**不做** `PERM_MJCF_TO_ISAACLAB`（BONES 需另测，
     按数据源分开）；
   - `diff_vla_wbc/sonic_encoder.py:build_motion_windows` 的 580 维布局是 **per-frame 交错**，与 encoder 的
     字段主序不符 —— 任何用它产 token 的地方（S3 锚定/探针）必须按字段主序修；
4. **新发现（landmine）**：`data/ha_sessions` 的 `observation.state` 实际是 **HA_V31（Isaac）序**，而
   `info.json` 声明的名字是 STANDARD_29 序（会话建于 10-24 05:32；`to_standard(jp_ha, src="ha_v31")` 是
   10-26 00:15 的 `4adbce1` 才加的）⇒ 现有 VLA 的 state 口径是 HA_V31（与服务端一致，闭环没问题），
   但**重跑 `prepare_ha_sessions.py` 会静默换成 STANDARD 序**，重跑前必须先定口径。

---

# 附录 B：修 pin + 修 builder 的执行结果（2026-10-26）

## B.1 pin ✅ 已修复
- `sim_eval_vla.py` 在 08:01 的外部改动是**有意的**：新增 `--sm_obs_mode auto`（跟随 ckpt 的
  `args.obs_mode`）+ 显式 A/B 才放行的越界保护；
- 已 `pin_ha_closedloop.py --write` 重新固化，`--check` **通过**（7 任务 / 16 资产 / init pose / decoder=release）；
- 闭环恢复可用；**此前的 pin 阻断解除**。

## B.2 builder：输入布局已破译（100% 可复现），锚点公式仍缺

| 项 | 旧 builder | 实测真值 |
|---|---|---|
| q/qd | 未来 10 帧 | **当前参考帧重复 10 次** |
| anchor | 未来 10 帧各算一个 | **当前锚点 6 维，10 帧恒定** |
| perm | 对 q/qd 做 MJCF→IsaacLab | **不做** |

- 重建 `[4:584] + [601:661]`（其余置 0）→ ONNX token 与 `encoder_latent` **逐位相等**
  （`scripts/verify_sonic_encoder_layout.py`，0.9995–1.0 PASS）；
- 直接把录制 1762 喂 ONNX → Δ=0；
- **端口 token 与标签 corr 0.057 的根因找到了**：不是"两个前端"，是喂给 encoder 的量错了。

**唯一未解**：`anchor_rot6d` 的计算公式（现有四元数组合最好 Δ=0.34）。HA 录制数据里有现成的
`consumed_anchor_rot6d`；BONES-SEED 没有 → **混库前必须解出**。

## B.3 对依赖链的更新（替换附录 A 的 A.2.1）

```
S0 ✅ → pin ✅
  └─> builder：布局 ✅ / 锚点公式 ⬜
        ├─ 现在就能做：S1-α = **HA-only 标签 token** 重训（699 条，npz 的 g1_motion_token_64，不需要 builder）
        └─ 混库（BONES-SEED + HA）需等锚点公式解出 → S1-β
  └─> S3 v2 依赖 S1-α 的产物；S2 仍为可选
```

## B.4 新增产物
| 产物 | 路径 |
|---|---|
| encoder 布局文档（含验证方法） | `docs/SONIC_ENCODER_INPUT_LAYOUT.md` |
| 布局验证脚本 | `scripts/verify_sonic_encoder_layout.py` |
| 重新固化的 pin | `configs/ha_closedloop_pin.json` |


---

# 附录 C：S3 按 S0 结论的修订与执行结果（2026-10-26）

> 执行报告（含全部数字与复现命令）：`docs/S3_JOINT_V2_REPORT_20261026.md`
> 产物：`runs/s3_gates/gates_label_bc.json`、`runs/s3_v2_bc/`、`runs/s3_v2_lamr03/`、`runs/S3_gates/contrast_*_64.json`

## C.1 对 §3.5 / §4 S3 的修订（只列被 S0 结论改掉的部分）

| 项 | §3.5/§4 原写法 | 修订后 | 依据 |
|---|---|---|---|
| BC 目标空间 | 端点 token（§3.3 重标后） | **标签 token = npz `g1_motion_token_64`（=`encoder_latent`）** | S0.4 |
| burn-in 的 GT conditioning | 端口 GT token | **跟随冻结 WBC 的训练空间**（当前=端口；S1 换标签 WBC 后 `--wbc-token-space label`） | WBC 只在自己训练过的空间里可被条件化 |
| 两套空间不一致时的 λ_r | 直接扫 {0,0.1,0.3,1.0} | **硬门禁**：`bc != wbc` 且 `λ_r>0` 直接退出（`--allow-token-space-mismatch` 仅诊断） | 本轮实测代价（C.2） |
| S2 base_aligned | 阶段 A（主要收益来源） | **降级为可选**；官方 ckpt + decode 就是合法 base 臂 | 附录 A.2.3 |
| 门禁 C2（新增） | 原 §4 没有 | **想象回报对 token 的区分度**，用**配对统计**（t ≥ 3、胜率 ≥ 0.8）；必须分 WBC 空间 / VLA 空间两次报 | 8 窗口版本会给假阴性 |

## C.2 实测（64 窗口，配对检验，冻结 WBC v2 = `wbc_mix2_obs93/ckpt_016000_spec`）

```
ret  gt_wbc=1.1132  mean=0.9477  shuffled=0.9288  gt_bc=0.7572  zeros=0.7428  randn=-1.0403
wbc-space contrast = +0.3704 (+33.3%)  paired t=21.4  win=64/64   → 奖励通道有信号
bc-space  contrast = +0.0144 (+1.9%)   paired t=1.53  win=53%     → 跨空间后信号消失
```
⇒ 复核 §0 里"奖励返传救不回行为"的判断要细化成两半：
**通道本身是好的**（t=21），坏的是**两套 token 空间**（W5/W6）。
只要 S1 把 WBC 换成标签空间，λ_r 就可以按 §3.5 开；在那之前，λ_r>0 的臂只能当诊断。

## C.3 两条 λ_r 臂（H=16、K=4、batch 4×accum 4、1000 步、固定 64 窗口评测）

| 臂 | λ_r | `L_fm(val)` | decode 后 token MSE | 手闭合率 pred/label | 想象回报 |
|---|---|---|---|---|---|
| B0（交付物） | 0 | 0.1365 → 0.0265 | 0.0042 → 0.0051 | 0.102 / 0.114 | 0.732 |
| B3（诊断，空间冲突） | 0.3 | 0.1365 → 0.0303 | 0.0042 → **0.0598（+14×）** | **0.000** / 0.114 | **1.006（+38%）** |

B3 把"空间不一致时开 λ_r"的代价量化了：想象回报涨 38%，但 token 偏离标签 14 倍、**手部闭合直接塌到 0**
（奖励只依赖 64 维 token，手部没有任何奖励梯度）。所以 §3.5 的 F 臂**必须**在空间统一之后才做。

## C.4 S3 执行状态（对照 §4 的 S3.1–S3.8）

| 项 | 状态 |
|---|---|
| S3.1 新入口 | ✅ `scripts/train_joint_v2.py`（旧 `train_joint.py` 未改） |
| S3.2 冻结执行器 | ✅ 门禁 `pi` 逐张量相等 + `wbc/wm` 反传后零梯度 |
| S3.3 单次采样 + FM-BC | ✅ 删除 token_anchor / hand-MSE；L_bc = GR00T 原生 FM |
| S3.4 burn-in 起点 | ✅ 起点 xy 误差 5.9 cm、survival 0.9997；对比度 +11%（burn-in vs 参考起点） |
| S3.5 H=16 + 归一化回报 | ✅ |
| S3.6 batch/accum/梯度比 | ✅ `grad_ratio(λ_r=0.3)=0.133`（落在 0.1–0.5） |
| S3.7 64 窗口评测 | ✅；mini 闭环脚本就绪但本容器 Isaac 不可用（见报告 §5.1） |
| S3.8 F vs B 闭环 | ⛔ 前置未满足：等 S1 的标签空间 WBC |

## C.5 追加修订（token 前端交接，2026-10-26 晚）：hold 口径把 §C.2 的结论翻过来了

交接文档：`docs/TOKEN_FRONTEND_HANDOVER_FOR_S3_20261026.md`（174–177 行给了 S3 的逐条清单）。

| 项 | C.1–C.4 的写法 | 交接后 |
|---|---|---|
| 参考库 | `runs/isaac_motion/ha_tokens/packed.pt` | **`runs/isaac_motion/ha_cmd/packed.pt`**（命令流；旧库的 `joint_pos` 是执行轨迹，lag +8 帧） |
| WBC | `wbc_mix2_obs93/ckpt_016000_spec.pt` | **`logs/wbc_hold_*/ckpt_*.pt`**（hold 前端训的；旧 ckpt 是 future 窗口） |
| burn-in 的 token | 预计算侧车 | **环内 hold builder + 当时仿真机器人 anchor**（`--wbc-token-source online`，交接 §5.3） |
| `--wbc-token-space` | 手填 label/port | `auto`：hold ⇒ label（并加"窗口口径必须一致"硬门禁） |
| C.2 的"bc-space 只有 1.9%" | 结论：跨空间无信号 | 在 hold 口径下 **bc-space 与 wbc-space 等价**（1.318 vs 1.336；配对 t=18.6 / 19.5），1.9% 那个数字是**future 窗口的错配**造成的 |
| C.3 的 λ_r 结论 | 只能等 S1 | hold 中间 ckpt（it 17250）上 `grad_ratio=0.2866`（落在 0.1–0.5），奖励项可以开了；正式表仍要等 `wbc_hold_*` 过三格门禁 |

新门禁产物：`runs/s3_gates_hold/gates_hold_label.json`（codec/freeze/burn-in/contrast/FM-BC 全 PASS）。
代码：`diff_vla_wbc/sonic_frontend.py` + `burnin.py(token_fn)` + `train_joint_v2.py`(库/窗口/自动发现)
+ `data.py(resolve_clip_files)`。
