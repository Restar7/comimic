# C-V8 闭环现状 · WM 信号 · 失败解剖（提交外部评审）

> **提交对象**：外部评审（GPT）。**自包含**，不需要看仓库或历史对话。
> **写于**：2026-11-08 20:50。
> **一句话**：**世界模型（WM）没有被训坏——它的梯度方向在真物理上验证过是有效的（69% / 98.4% 同号）；
> 真正的问题是我们在用"0.4 秒的跟踪奖励"去优化一个"12–54 秒的任务"，而且 WM 的幅度高估约 1.8×。**
>
> 证据等级：【FACT】有落盘文件；【INFER】从 FACT 推出；【GUESS】未验证猜测（附证伪方法）。

---

## 0. 一页速读

| 问题 | 答案 | 等级 |
|---|---|---|
| 闭环跑完了吗？ | **跑完了**：4 任务 × 6 seeds = **24 集**，2 小时（18:39→20:36） | FACT |
| 成功多少？ | **12/24（50%）**：Boxing 5/6、VisionNavi 3/6、OpenDoor 2/6、SitSofa 2/6 | FACT |
| 失败长什么样？ | **12 次失败里 11 次是 timeout（跑满 3000 步）、只有 1 次摔倒** | FACT |
| 这说明什么？ | **机器人站得稳，但"不干活"**——任务进度推不动，不是平衡问题 | 【INFER】 |
| WM 训坏了吗？ | **没有**。WM 梯度方向在真物理上验证有效：48 窗口 69% 同号、真实漂移 **−1.23 cm**；H=20 复测 **98.4%** 同号 | FACT（2026-11-05 / 11-07 两次实验） |
| WM 的缺陷是什么？ | **幅度高估 1.79×（中位）/ 2.17×（均值）**；而且只验证到 **20 步 = 0.4 秒** | FACT |
| 离线收益多大？ | test ΔJ **+0.01640（+1.04%，CI 0.77–1.29%）**，是旧配方的 2.9 倍；但只占 J 剩余 headroom 的 **6.7%** | FACT |
| V8-C 比 base 好吗？ | **同场景配对后仍分不出来**：OpenDoor 的 4 个不一致对全部是"base 成功、V8-C 失败"（McNemar 精确 p=**0.125**）；SitSofa 2:1、VisionNavi 0:1（p=1.00）。**场景已证明完全相同**（`episode_object_seed` 逐 seed 一致） | FACT |
| 最大的混杂是什么？ | base 是 **torch 2.8.0a0 + python3.12**，V8-C 是 **torch 2.7.0 + python3.11**（Isaac/SONIC 侧**完全相同**） | FACT |
| 最可能的问题排序？ | ① 目标—指标错配 ② 幅度过冲×长累积 ③ 统计噪声/环境差异 | 【INFER】 |

---

## 1. 闭环最终结果（24 集，逐集）

**配置**：V8-C 最新 adapter `actor_wm0_p2p4/checkpoint_lora_best.pt`（it=250，test ΔJ +0.01640，
sha256 `ac85a95c29c5bc13…`），**只跑 V8-C，不跑 base、不跑旧配方**；
双卡两条流水线（每卡 1 个 VLA server + 1 个 Isaac），`MAX_STEPS=3000`，seeds 0–5，录像全开。

### 1.1 分任务汇总

| 任务 | n | 成功 | 成功率 | 平均步数 | 终止原因 |
|---|---|---|---|---|---|
| **Boxing** | 6 | **5** | **83%** | 998 | success×5, timeout×1 |
| **VisionNavi** | 6 | **3** | **50%** | 2352 | success×3, timeout×3 |
| **OpenDoor** | 6 | **2** | **33%** | 2302 | success×2, timeout×4 |
| **SitSofa** | 6 | **2** | **33%** | 2519 | success×2, timeout×3, fall×1 |
| **合计** | **24** | **12** | **50.0%** | — | **timeout×11, fall×1** |

### 1.2 逐集明细

| 卡 | 任务 | seed | 成功 | 步数 | 原因 | 时长(s) |
|---|---|---|---|---|---|---|
| p0 | Boxing | 0 | ✅ | 370 | success | 104 |
| p0 | Boxing | 1 | ✅ | 1622 | success | 433 |
| p0 | Boxing | 2 | ✅ | 372 | success | 100 |
| p0 | Boxing | 3 | ❌ | **3000** | timeout | 726 |
| p0 | Boxing | 4 | ✅ | **180** | success | 51 |
| p0 | Boxing | 5 | ✅ | 441 | success | 121 |
| p1 | OpenDoor | 0 | ❌ | 3000 | timeout | 709 |
| p1 | OpenDoor | 1 | ❌ | 3000 | timeout | 674 |
| p1 | OpenDoor | 2 | ❌ | 3000 | timeout | 582 |
| p1 | OpenDoor | 3 | ✅ | **616** | success | 89 |
| p1 | OpenDoor | 4 | ✅ | 1195 | success | 171 |
| p1 | OpenDoor | 5 | ❌ | 3000 | timeout | 429 |
| p1 | SitSofa | 0 | ❌ | 2679 | **fall** | 653 |
| p1 | SitSofa | 1 | ✅ | 1618 | success | 435 |
| p1 | SitSofa | 2 | ❌ | 3000 | timeout | 636 |
| p1 | SitSofa | 3 | ❌ | 3000 | timeout | 718 |
| p1 | SitSofa | 4 | ❌ | 3000 | timeout | 726 |
| p1 | SitSofa | 5 | ✅ | 1816 | success | 474 |
| p0 | VisionNavi | 0 | ✅ | **2708** | success | 713 |
| p0 | VisionNavi | 1 | ❌ | 3000 | timeout | 794 |
| p0 | VisionNavi | 2 | ❌ | 3000 | timeout | 757 |
| p0 | VisionNavi | 3 | ❌ | 3000 | timeout | 794 |
| p0 | VisionNavi | 4 | ✅ | 941 | success | 271 |
| p0 | VisionNavi | 5 | ✅ | 1466 | success | 399 |

* 成功步数：**中位 1068**，范围 **180–2708**；
* 失败步数：11 个 3000 + 1 个 2679（摔倒）；
* 总仿真时长 3.21 h（双卡并行，墙钟 1 h 57 min）；控制频率 3.7–4.7 步/s。
* 产物：`runs/three_track_v8/c_tracking_recovery/closedloop_v8c_final/{run.log,RESULTS.md,summary.json,pipe0/,pipe1/}`

---

## 2. 失败解剖（这是本文最重要的一节）

### 2.1 失败形态：11/12 是"站着不动/无效运动"，只有 1 次摔倒

| 失败类型 | 次数 | 说明 |
|---|---|---|
| **timeout** | **11** | 跑满 3000 步（≈13 min 仿真）仍没触发任务成功判定；**机器人没有倒** |
| fall | 1 | SitSofa seed0，第 2679 步，`tilt_deg=90.5`（完全躺平） |

⇒ **【INFER】策略不是"控制不稳"，而是"不推进任务"**。J20 里的稳定项（survival / 姿态 / 速度）已经接近满分，
但任务进度（走到沙发、穿门、到目标点）没有任何项在约束。

### 2.2 视频证据（人工抽帧）

SitSofa seed0（唯一摔倒的那集）抽 9 帧：
* 第 0 步：站在房间，沙发在正前方；
* 第 400–2000 步：一直在房间里移动，**沙发距离几乎没变**（没走过去）；
* 第 2400 步：终于靠近沙发；
* 第 2600 步：画面地平线开始倾斜；
* 第 2670 步：画面横 90°（摔倒，与 `tilt_deg=90.5` 完全一致）。

⇒ 这集是"**走了 2600 步才靠近沙发，然后在坐下前翻倒**"。和"导航/接近慢"的判断一致。

### 2.3 与 base（官方 FT8000）的对照

`runs/ab36/base`（2026-11-07，**同一套 Isaac/SONIC 执行器**、同 3 任务、同 seeds、同 `MAX_STEPS=3000`、
`ckpt=checkpoint-8000, head=none, delta_token_enabled=False`）：

| 任务 | base（11-07） | V8-C（今天） | Fisher 精确检验 |
|---|---|---|---|
| OpenDoor | **6/6（100%）** | 2/6（33%） | **p = 0.061** |
| SitSofa | 3/6（50%） | 2/6（33%） | p = 1.00 |
| VisionNavi | 2/6（33%） | **3/6（50%）** | p = 1.00 |
| 三任务合计 | 11/18（61%） | 7/18（39%） | p = 0.32 |

**【FACT】统计上没有一个任务达到显著（OpenDoor 也只是边缘 p=0.061）。**
**这条必须强调：以现在 n=6/任务 的样本量，"V8-C 变差"这个说法在统计上不成立。**

#### 2.3.1 配对分析（同场景）——比上面那张表更该看

**场景确实是配对的**：`result.json` 里的 `episode_object_seed` 在 base 与 V8-C 之间**逐 (task, seed) 完全相同**
（OpenDoor seed0–5：`11400714785074694791` … `…4790`，`source=env_seed`）
⇒ 同一 seed 下物体摆放、朝向、场景随机化**逐位一致**，差别只来自策略本身 + 采样噪声。

同场景配对（McNemar 精确检验，只看"一边成功一边失败"的对）：

| 任务 | base 赢 / V8-C 赢（不一致对） | 逐 seed 结果（base → V8-C） | McNemar 精确 p |
|---|---|---|---|
| **OpenDoor** | **4 / 0** | s0 ✅→❌、s1 ✅→❌、s2 ✅→❌、s3 ✅→✅、s4 ✅→✅、s5 ✅→❌ | **0.125** |
| SitSofa | 2 / 1 | s0 ❌→❌、s1 ✅→✅、s2 ✅→❌、s3 ✅→❌、s4 ❌→❌、s5 ❌→✅ | 1.000 |
| VisionNavi | 0 / 1 | s0 ❌→✅、s1 ❌→❌、s2 ❌→❌、s3 ❌→❌、s4 ✅→✅、s5 ✅→✅ | 1.000 |

**读法**：
* OpenDoor 是唯一有**一致方向**的任务：4 个不一致对**全部**是"base 成功、V8-C 失败"，**没有一个反向**。
  精确 p=0.125（n 太小），但"4:0"这个方向性比 p 值更值得注意；
* SitSofa / VisionNavi 是**双向噪声**（2:1、0:1），没有任何结论；
* 场景配对只能排除"物体摆放差异"，**排除不了** server 采样噪声（seed=-1）和 torch 版本差异——
  同一场景下轨迹照样会分叉，所以这叫"**配对场景、非配对轨迹**"。

### 2.4 唯一接近显著的一项（OpenDoor）与它的混杂

OpenDoor 是 base **6/6**、V8-C **2/6**，差得最多。但：

* **环境不同**：base 的 VLA server 是 `/usr/bin/python3 3.12.3 + torch 2.8.0a0+…nv25.06`；
  V8-C 是 `conda 3.11.14 + torch 2.7.0+cu128`。**Isaac 侧（SONIC release ONNX、WBC ckpt、pin）逐字相同**，
  server 代码也只差 LoRA 加载那 55 行（已 diff）。
* 4 步 Flow-Matching 的采样对 torch 版本/kernel/TF32 敏感；闭环 3000 步是反馈回路，微小差异会放大。
* 另一侧证据：**VisionNavi 今天 3/6 反而比 base 2/6 好** ⇒ 不像"环境系统性拖累今天"。

⇒ **【GUESS】OpenDoor 的差距可能是真效应（LoRA 损害了"接近门+推门"），也可能是环境+噪声。
必须用同环境跑一次 base 才能判。**

### 2.5 一个重要的评测缺口

**这次 24 集没有开轨迹 dump**（只有 `result.json` + 视频）。因此**无法计算"任务进度"**
（`scripts/eval_closedloop_progress.py` 需要逐帧状态：`d_path` 到示范路径的距离、`prog_end` 路径进度）。
base 的 `ab36` 反而有 `dump/`。

⇒ 现在的"timeout"只能说明"没触发成功判定"，**不能区分"走错了路"和"只差一点但超时"**。
这是一个应该在下一轮补上的关键观测。

---

## 3. WM 的全部信号（保真度 / 方向 / 幅度 / 刷新）

### 3.1 结构（回顾）【FACT】

* MLP **133 → 2048 → 2048 → 70**，8.8M 参数；`s_{k+1} = s_k + dt·f(s_k, a_k)`，dt=0.02（50 Hz）。
* 输入 133 = 71 状态 + 29 动作 + 33 ξ（训练时 ξ=0）。
* 版本：**WM0 = `wm_gain2.pt`，sha256 `fc5aeb30e987f016…`，mtime 2026-11-05 15:54**（本次 V8-C 训练全程冻结它）。

### 3.2 保真度：三种口径、三个答案【FACT】

| # | 协议 | 20 步 root 误差 | 备注 |
|---|---|---|---|
| A | `probe_wm_multistep.py`：256 个策划窗口，**喂录制动作** | 想象 **6.93 cm** vs 真实 **6.81 cm**，**ratio 1.018**；端点 p50 **2.11** / p90 **5.93 cm**；corr **0.929** | **PASS**（判据 ratio ∈ [0.8,1.2]） |
| B | `diagnose_wm_shift_v7.py`：任意真实转移，逐 tick 均值 | h=1 **2.80** → h=5 3.00 → h=10 3.55 → **h=20 4.59 cm**（p95 18.5） | 高动态分布上 2–5 cm |
| C | 旧文档（2026-11-05）记的端点保真 | **0.89 cm** | **今天复现不出来**（权重重导出/哈希争议；mtime 11-05 15:54） |

* 绝对误差增长：20 步 4.59 cm ≈ **每步 2.3 cm 的漂移**（50Hz 下 0.4 s）。
* 一个 1000 步的闭环 episode = 50 个这样的 20 步窗口。

### 3.3 方向：**这是"WM 没训错"的核心证据**【FACT】

**实验 ①（2026-11-05，`docs/CURRENT_APPROACH_FOR_REVIEW_20261105.md` §12.2）**
48 个窗口、δ = 0.15 SONIC 单位（≈3.3°/关节）、各滚 20 步，在**真物理（Isaac）**里结算：

| 方向 | WM 判断 | 真物理里"+"更好 | 真实漂移变化 |
|---|---|---|---|
| **∂J_WM/∂a（WM 梯度）** | + | **33/48 = 69%** | **−1.23 cm（变好）** |
| FT 实际造成的动作位移（0.17 单位 ≈4.3°/关节） | +（ΔJ_WM=+0.029） | **17/48 = 35%** | **−0.11 cm（没变好）** |
| 参照：WM 自身 20 步保真误差 | — | — | 0.89 cm（旧口径） |

结论原话：**"WM 没训错：它的 20 步梯度方向在真机上是有效的（69% 同号、漂移 −1.23 cm、信号/误差 ≈1.4）"**；
**"错在把方向落实成参数：FT 实际走的方向，不是动作空间里的上升方向"**。

**实验 ②（2026-11-07，`docs/REVIEW_TWO_ROUTES_BPTT_VS_DISTILL_20261107.md` §3.2）**
64 窗口、`a0 ± δ·ĝ`（δ=0.03 rad）、H=20，真物理结算：

| 视界 | 窗口 | 符号一致率 | 真物理变好比例 | ΔJ 相关 | J_true 均值 | J_WM 均值 |
|---|---|---|---|---|---|---|
| H=2 | 96 | 1.000 | **1.000** | +0.964 | +0.0076 | +0.0087 |
| H=20 | 64 | 0.984 | **0.984** | +0.092 | **+0.0587** | **+0.1277** |

⇒ **方向可用（H=20 也可用）；幅度高估：中位 1.79×、均值 2.17×**（所以当时取 η=0.08 ≈ 0.15/1.8）。

### 3.4 幅度：WM 说"多推一点"，真机只需要 0.55 倍【FACT + INFER】

* 上表：`|ΔJ_WM| / |ΔJ_true|` = **1.79×（中位）**；
* 本次 V8-C 的 LoRA 是**每步都施加** token 偏移（rms 0.010），不是一次性 δ 扰动；
* 【INFER】如果策略沿 WM 方向但幅度按 1.8× 过冲，**3000 步的累积会把"局部最优"变成"系统性偏离"**。

### 3.5 历史已暴露的同类失败（必须一起读）【FACT】

* 2026-11-05：**FT 联合微调用 0.17 的动作位移（≈4.3°/关节），在真机上把漂移抬了 +4.5 cm**（8 对配对，只有 2/8 变好）。
* 早期四种参数化（LoRA / 反常量项 / state_extra / delta-token）**全部收敛到 0.08–0.17 的常向量**
  ——因为跨窗口把每窗口各自的方向平均掉了。
* 扭曲最可能来自 **token → 4 步流匹配采样 → 动作** 这条链的雅可比。

⇒ 也就是说：**"WM 方向对" 和 "训出来的策略闭环变好" 之间，历史上已经断过一次。**

### 3.6 WM 刷新（WM1）的信号【FACT】

| 项 | 结果 |
|---|---|
| 新 on-policy 数据 | 366 段 / 7,552 tick / 12 episodes（真 Isaac 逐 tick） |
| 训练 | 150 updates，942 s；val single 2.196→1.204，kstep 29.44→15.82 |
| 门禁 | 新分布 root **31.80 → 10.97 cm（−65.5%）**，旧 A′ 4.500→4.480（+0.45%） |
| **但** | ① 只有 **4 个 episode** 进验证；② 旧 A′ 的**成对 bootstrap CI = [−0.63%, −0.14%]（略变差）**；③ **`dir_agree` 没跑 → override 发布** |
| 对 Actor 的收益 | WM1 那条 test **+0.01796** vs WM0 本条 **+0.01640**（+9.5%）——**但步数 400 vs 280、聚合 mean vs worst_window，不是干净 A/B** |
| 在"任意真实转移"上 | WM1 **4.56 cm** vs WM0 4.59 cm（−0.7%），但 ∑ssm **+13.6% 更差**、q MAE **+32% 更差** |

⇒ **刷新让 WM 更贴合当前策略，但没让它成为"更准的世界模型"。**

---

## 4. 诊断：问题到底在哪（三层，按证据强度）

### 第 1 层（最强）：**目标函数与任务指标错配**

| | J20（训练在优化的） | 闭环成功（在考的） |
|---|---|---|
| 时间尺度 | **20 步 × 0.02 s = 0.4 秒** | **600–2700 步 = 12–54 秒** |
| 语义 | 6 个 exp 核的跟踪积分 | "走过去坐下 / 开门穿门 / 到目标点" |
| 上限 | **1.821**（基座已 86.5%，训练后 87.4%） | — |
| 失败可见性 | 看不见"没完成任务" | 11/12 失败都是它 |

**决定性证据**：11 次 timeout 的机器人**站满了 3000 步没倒**——J20 里"稳定"这一项已经满分，
但 J20 **完全不奖励任务进度**。所以 **J20 +1% 完全可以对应任务成功率不变**。

### 第 2 层：**幅度过冲 × 长累积**

WM 幅度高估 **1.79×**（实测），策略每步施加偏移，3000 步累积。
历史先例：FT 的位移方向对、真机漂移反而 +4.5 cm。

### 第 3 层：**统计噪声 + 环境混杂**

* n=6/任务：Fisher p = 0.061 / 1.00 / 1.00；三任务合计 p=0.32；
* base 与 V8-C 的 **torch/python 版本不同**（Isaac 侧相同）；
* 训练只有 **seed 0**。

---

## 5. 为什么"当时觉得效果蛮好"和"现在闭环一般"不矛盾

三件事同时为真：

1. **WM 方向**确实好（69% / 98.4% 同号，真物理漂移 −1.23 cm）——但这是**单次 δ 扰动、20 步、0.4 秒**的尺度；
2. **离线 J20** 确实涨（+1.04%，CI 不含 0，2.9× 旧配方）——但这是 **WM 自己算的 0.4 秒跟踪回报**；
3. **闭环任务成功率**是 **12–54 秒的语义完成度**——**没有任何一个实验直接测过这个尺度**。

⇒ 我们缺的从来不是"WM 好不好的证据"，而是**"0.4 秒的 WM 信号 → 50 秒的任务成功"这条传导链**的证据。

---

## 6. 证据边界（评审请当作硬约束）

| # | 缺口 | 状态 |
|---|---|---|
| 1 | **同环境的 base 臂**（今天的 server env） | **NOT_RUN** —— 唯一能归因的实验 |
| 2 | 闭环轨迹 dump（进度 `prog_end` / 路径距离 `d_path`） | **NOT_RUN** —— 本次没开，无法区分"走错"与"只差一点" |
| 3 | WM 在 **H>20**（0.4 s 以上）的保真/方向 | **NOT_RUN** |
| 4 | LoRA 幅度扫描（scale 0 / 1.0 / 2.0） | **NOT_RUN** —— 可直接验证"1.8× 过冲" |
| 5 | `dir_agree` 梯度探针（WM1 门禁） | **NOT_RUN**（override 发布） |
| 6 | 训练多 seed | **NOT_RUN**（全部 seed 0） |
| 7 | 任务级奖励（progress/success） | **未实现** |
| 8 | pristine 越限率 2.59% 的落盘 | **PROVENANCE 弱**（文档记录，未落 JSON） |

---

## 7. 下一步实验（按性价比，写清"能证伪什么"）

| 优先级 | 实验 | 成本 | 能回答 |
|---|---|---|---|
| **P0** | **同环境 base 臂**：用今天的 server/Isaac 环境跑 OpenDoor 6 集 | ~1 h（1 卡） | OpenDoor 6/6→2/6 是真效应还是环境/噪声 |
| **P0** | **LoRA 幅度扫描**：`--lora16-scale` ∈ {0（=base）, 1.0, 2.0} × OpenDoor 6 seeds | ~2.5 h（双卡） | 18 个"1.8× 过冲"假设；scale=0 顺便就是 base 臂 |
| **P1** | **开 dump 重跑 6–8 个失败集**，算 `prog_end / d_path` | ~1.5 h | "走错路" vs "只差一点超时"（决定要不要加步数/加进度奖励） |
| **P1** | **H=40 / H=100 的 WM 想象回报**（同一 checkpoint 重评） | 半天 | 0.4 s 视界是否就是瓶颈 |
| **P2** | 任务级 progress reward 替代/补充 J20 | 2–3 天 | 根治"指标错配" |
| **P2** | 多 seed 重训（seed 1/2） | 2 × 47 min | 训练方差 |

> **P0 的两条其实是同一个实验设计**：把 adapter 幅度当一个扫描轴（0 / 1.0 / 2.0），
> 一次就同时回答"LoRA 有没有用"和"幅度是不是过头"，比再跑 6 个 seed 的信息量大得多。

---

## 8. 给评审的 8 个问题

1. **诊断对不对**：11/12 失败是 timeout、只有 1 次摔倒，是否足以支持"策略稳定但不推进任务"这个结论？
   还有什么替代解释（例如成功判定太严、参考动作本身不可完成）？
2. **WM 方向已验证、但闭环没变好**：你认为断点更可能在
   (a) 0.4 s → 50 s 的视界外推、(b) 幅度过冲、(c) token→FM→动作的雅可比，还是 (d) 指标本身？
3. **J20 该不该被替换**？如果换，你会用什么任务级信号（progress / success / sparse reward）构造可微目标？
4. **单 seed + n=6/任务**：在这种样本量下，你会怎么设计下一轮实验才能得到可判定的结论？
   需要几个 seed / 多少个 episode？
5. **环境混杂**：torch 2.8 vs 2.7（Isaac 侧完全相同）会不会足以让 OpenDoor 从 6/6 掉到 2/6？
   你会怎么验证/排除？
6. **幅度扫描**：把 LoRA delta scale 从 2.0 降到 1.0/0.5，你认为这是"合理的工程修正"还是"掩盖问题"？
7. **`dir_agree` 探针**：在只有 12 episodes 的新数据上，这个探针有意义吗？还是应该先采更多 on-policy 数据？
8. **如果要写论文/对外汇报**：以现有证据，哪三句话可以写、哪三句不能写？

---

## 9. 证据索引

```
WT = /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
MAIN = /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

# 闭环（本次，24 集）
$WT/runs/three_track_v8/c_tracking_recovery/closedloop_v8c_final/
  ├ run.log / RESULTS.md / summary.json
  ├ pipe0/runs/{Boxing,VisionNavi}__seed*__*/{result.json,videos_*}
  └ pipe1/runs/{SitSofa,OpenDoor}__seed*__*/{result.json,videos_*}

# base（官方 FT8000，同 harness/同 3000 步/6 seeds）
$MAIN/runs/ab36/base/{OpenDoor,SitSofa,VisionNavi}/seed*/**/result.json
$MAIN/runs/ab36/base/*/server.log        # ckpt=checkpoint-8000 head=none
$MAIN/runs/ab36/delta/                   # 同批 A 线 Δtoken 臂（9/18）

# WM 方向（两次真物理实验）
$MAIN/docs/CURRENT_APPROACH_FOR_REVIEW_20261105.md   # §12.2 69% / −1.23cm
$MAIN/docs/REVIEW_TWO_ROUTES_BPTT_VS_DISTILL_20261107.md  # §3.2 98.4% / 1.79×
$MAIN/runs/probe_truth/exc_dump.pt
$MAIN/logs/truth_probe.log  $MAIN/logs/truth_probe_h20.log

# WM 保真 / 刷新
$WT/runs/three_track_v8/c_tracking_recovery/probe_wm0_multistep/probe.json
$WT/runs/three_track_v8/c_tracking_recovery/diag_wm0_same/diagnostics.json
$WT/runs/three_track_v8/c_tracking_recovery/diag_wm1_same/diagnostics.json
$WT/runs/three_track_v8/c_tracking_recovery/wm_refresh/gate_publish/gate_report.json
$MAIN/runs/a_prime_sonic_exc/wm_gain2.pt              # WM0, sha fc5aeb30…

# V8-C 训练与离线评测
$WT/runs/three_track_v8/c_tracking_recovery/actor_wm0_p2p4/
  ├ val.jsonl / train.jsonl / cond_check.jsonl
  ├ attribution.jsonl / physical_report.jsonl / stopped.json
  └ test_best/test.json            # ΔJ +0.01640

# 相关文档
$MAIN/docs/C_V8_FOR_GPT_REVIEW_20261108.md    # 训练+WM 全量证据（上一份）
$MAIN/docs/C_STATUS_V8C_FINAL_20261108.md
$WT/docs/V7_PROGRESS.md
```

---

## 10. 术语

| 术语 | 含义 |
|---|---|
| **J20** | 冻结 WM 上 20 步想象回报（γ=0.99，6 个 exp 跟踪核），**不是真机回报**，理论上限 1.821 |
| **ΔJ** | adapter 与 pristine 在同一 batch/同一 flow 噪声下的成对差 |
| **timeout** | 跑满 `MAX_STEPS=3000` 仍未触发任务成功判定 |
| **fall** | 姿态终止（`hard_tilt`，本批唯一一次 `tilt_deg=90.5`） |
| **dir_agree** | WM 梯度方向与真物理改进方向的一致性（本批未测，WM1 门禁为 override） |
| **prog_end / d_path** | 终点在示范路径上的进度 / 轨迹到示范路径的距离（本轮**未采集**） |
