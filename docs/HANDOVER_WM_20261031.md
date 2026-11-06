# 交接文档：世界模型（WM）精度攻坚 · 2026-10-31

> 面向**新对话**的完整交接。只写"现在是什么状态、为什么、下一步怎么做"，不写过程叙事。
> 关联文档：`docs/ABLATION_SUITE.md`、`docs/PAPER_METHOD_S3_CLOSED_LOOP.md`、`docs/DESIGN_LOSS_S_AND_H20_20261030.md`

---

## 0. 一页纸现状

| 项 | 状态 |
|---|---|
| 闭环策略 d2b / d3 | **d2b 0.153 m（最好）**、d3 0.150 m；同会话 A/B 判 d3 **不优于** d2b（root +0.020 m，t=+1.10；**tilt +1.95°，t=+2.58，仅 1/12 更好**） |
| 执行器天花板 | **GT 命令臂 0.108 ± 0.038 m**（6 集）⇒ 破 0.108 必须重训 WBC，不是改 head |
| 当前 WM | `runs/S1_handover/wm_d3.pt`（k=1 root 0.1 cm / k=20 **6.5 cm**），**仍是唯一可用裁判** |
| 反事实数据（token/动作空间扰动） | **四次实验全部失败**（详见 §3），路线**已判定不成立** |
| 正在做 | **GRU 版 WM 已判定：不能降低误差**（见 §11 与 `docs/WM_GRU_ABLATION_20261031.md`）。下一步按 §8 转路线 A/C |
| 卡住的根因 | WM 长视界精度比需求差 **5–6 倍**（需 k=8 ≤0.3 cm，现 ~1.2 cm），而想象梯度要能分辨 1–5 mm 的策略差异 |

---

## 1. 为什么卡在 WM 精度（问题的物理来源）

1. 训练用的"执行误差"只能靠 WM 想象出来：`tokens → WBC 解码 → WM 预测状态的后续 K 步 → 奖励(状态 vs 参考)`。
2. 策略在 8 步（0.16 s）里能改变的位置只有 **1–5 mm**；WM 在 8 步的预测误差 **~2 cm** ⇒ 信噪比 **1:4~1:20**，梯度方向是噪声。
3. 实测判别力：同一批 128 窗口，用 WM 判断"VLA 臂 vs GT 臂"谁更好，**VLA 赢 46%（掷硬币）**，而真实相差 **39%**（0.150 vs 0.108）。
4. 因 H=8 只覆盖 20 步动作块（0.4 s）的 40%，且长视界信用靠 critic（其梯度被 `--critic-state-grad-scale 0.1` 削 10 倍）。**H 的上限 = WM 的可靠步数**。

**判据（这两条是后续一切实验的验收线）**
- k=8 root_err **≤0.5 cm**（现 ~1.2 cm）
- k=20 root_err **≤1.0 cm**（现 6.5 cm）
- 附加：k=1 不得退化（基线 **0.1 cm**）；关节 k=20 不得差于 0.144 rad 太多

---

## 2. 已确认的测量事实（可直接引用，勿重复测）

### 2.1 WM 分档精度（`wm_d3.pt`，in-distribution 留出）

| k | joint_rmse | root_err | rot_err | qd_corr | root_w_corr |
|---|---|---|---|---|---|
| 1 | 0.008 rad | **0.1 cm** | 0.5° | 0.739 | 0.816 |
| 5 | 0.050 | 0.9 cm | 3.1° | 0.492 | 0.554 |
| 10 | 0.096 | 2.4 cm | 7.6° | 0.213 | 0.331 |
| 20 | 0.144 | **6.5 cm** | 13.9° | 0.250 | 0.212 |

### 2.2 训练窗口的工作点（B 线池，829 dump / 5,724 近场起点）

| 量 | 值 |
|---|---|
| 候选起点 | 20,642；`e0 ≤0.3` 通过 **27.7%**（5,724）；中位 e0 **0.681 m** |
| 训练起点 e0 | mean **0.152** / p50 0.143 / p90 0.263 |
| 8 步内漂移 | +15 mm（0.152→0.167）；20 步 +48 mm |
| 闭环工作点 | root 0.150 m、tilt 4.9°、\|Δyaw\| 11.5–13.5° |
| 奖励核 | 位置 exp σ=0.2（梯度峰在 0.141 m，工作点恰好在此）；hinge 门槛 0.35 m **训练中永不触发**；tilt hinge 20°、yaw hinge 15° 同样不触发 |

### 2.3 WM 的输入/结构

- `net.0 (2048,133)`、`net.3 (2048,2048)`、`net.6 (2048,2048)`、`net.9 (70,2048)` ⇒ **133→2048→2048→70**，2D 参数 **8,804,352**
- `explicit_dt0.02`：`s_{k+1} = s_k + dt·f(s_k,a_k)`，dt=0.02（50 Hz）
- 输入 133 = 71 状态 + 29 动作 + 33 ξ；**训练时 ξ 用 zeros**；不使用 obs_hist

---

## 3. 四次反事实数据实验（**全部失败**，路线已判定不成立）

目的：WM 训练集里没有"token/动作稍有不同会怎样"的样本，试图补上。

| # | 产物 | 扰动空间 | σ | 占批比 | 进哪阶段 | K | k=1 root | k=5 | k=10 | k=20 | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `wm_d3.pt` | — | — | — | — | 20 | **0.1 cm** | 0.9 | 2.4 | **6.5** | 基线 |
| 1 | `wm_wmuse_p2.pt` | token | 0.03–0.30 | 25% | 阶段2 | 1 | 2.4 | 12.1 | 26.3 | 65.7 | 失败 |
| 2 | `wm_wmuse2.pt` | token | ≤0.12 | 8% | 阶段2 | 1 | 1.2 | 5.4 | 9.2 | 13.3 | 失败 |
| 3 | `wm_wmuse3.pt` | token | ≤0.12 | 8% | 阶段2b | **8** | **0.3** | 1.9 | 4.7 | 12.0 | 失败 |
| 4 | `wm_wmuse4.pt` | **动作** | 0.02/0.05/0.10 | 8% | 阶段2b | **8** | **0.3** | 2.0 | 4.9 | 12.3 | 失败 |

**归因结论（已排除的假设）**

1. "幅度太大"——σ 0.30→0.12 只把伤害从 ×10 降到 ×2，**不解释**残余差距。
2. "占比太高"——25%→8% 同理，**不解释**。
3. "放错阶段"——单步→多步把 k=1 从 ×12 修到 ×3，**但多步仍差 ~2 倍**。
4. "扰动空间错（token 隐空间不在流形上）"——**动作空间（物理合法、幅度仅策略自身变化的 4%–20%）得到完全一样的数**（③ 与 ④ 逐档重合：0.3/1.9/4.7/12.0 vs 0.3/2.0/4.9/12.3）⇒ **该假设被否掉**。
5. **未排除的混淆**：③④ 同时把 `K` 从 20 改成 8（为容纳 20-tick 段），所以 ③④ 相对基线的退化可能部分来自 K。**用户口径：K 一律钉死 20，不再跑 K=8。**

**结论：任何合成扰动数据（token 或动作空间）都无法提升 WM 的 k 步精度，且会损害短视界精度。此路线终止。**

---

## 4. 关键 bug 清单（本轮发现并修复/记录）

### 4.1 已修复

| # | 文件 | 问题 | 修复 |
|---|---|---|---|
| 1 | `scripts/isaac_bw_pert_collect.py`（新） | `sel`（候选索引，0..7000+）被用于索引形状为 `n` 的 env 缓冲区 ⇒ CUDA `device-side assert`，堆栈指在无关行 | 区分 `sel`（候选）与 `eid`（env，`arange(n)`） |
| 2 | 同上 | `write_*_to_sim(..., env_ids=<CUDA tensor>)` 触发 assert | 全量写，不传 `env_ids`（与 A 线采集器一致） |
| 3 | 同上 | `torch.device("cuda")` 无序号，Isaac 切卡后出现 cuda:0/cuda:1 混用 | 显式 `cuda:0` + `torch.cuda.set_device` |
| 4 | 同上 | `set_joint_position_target(_q_tgt[:, idx_std], joint_ids=idx_std)` 越界 | 传 29 宽张量本身 + `joint_ids=idx_std` |
| 5 | `scripts/s3_pipeline.sh` | P 占比断言把"本阶段按设计不用 P（P=0）"判成失败 ⇒ 管线自伤中止 | `if _np > 0 and _want_p > 0 ...` |
| 6 | `scripts/s3_pipeline.sh` | 缺少 `runs/S3_pipeline/<ROUND>/state.json` 时静默空跑（只留 `DONE rc=1`） | 需先 seed（见 §6） |
| 7 | `scripts/s3_pipeline.sh` | 新增 `WM_SKIP_P1=1`：阶段1产物存在则复用，省 4 分钟 | 已加 |
| 8 | `scripts/finetune_wm_velocity_policy.py` | 新增 `--pert-dump/--mix-pert/--pert-seg-len/--a-order`，第三池 P 与三元配比打印+断言 | 已加 |

### 4.2 **已发现但未修（重要）**：A 线关节顺序不一致

- A 线文件（`runs/wm_real/ac_w*.pt`）的 `meta.joint_names` 声明 **STANDARD_29**，实测数据也是 STANDARD（`corr(|mean q|,|default_q|)=0.62`；按 ha_v31 换序降到 0.03）。
- 但 `finetune_wm_velocity_policy.py` 对 A 线**也套用了** `state_to_standard(perm=ha_v31→standard)` ⇒ **A 线的 29 个关节在 WM 训练里被打乱过**（A 线占阶段2 一半）。
- 已加开关 `--a-order {ha_v31,declared}`（**默认 `ha_v31` = 沿用历史行为**，保证与 `wm_d3` 可比）。**未验证翻转后的效果**——这可能是 WM 长视界误差的一个隐藏来源，值得作为独立实验。

### 4.3 操作踩坑（避免重犯）

- `pkill -f <script>` 会匹配到自己的 shell/进程 ⇒ **只能按 PID 杀**；本轮曾因它误杀正在跑的训练。
- 长命令里的 `VAR="...(通配符)..."` 变量展开会把 `*.pt` 展开成几百个路径 ⇒ 必须用数组+引号，或直接走管线。
- 长时间任务用 `setsid nohup ... &`，否则会被 shell 退出/后续命令带的 kill 波及。
- 直接调 `eval_wm_multistep.py` 必须导出：`SUPERMIMIC_HA_ALIGN=1` 与 `SUPERMIMIC_ROBOT_USD=<g1-29dof_wholebody_dex3 USD>`，否则 `[spec] default_q 与 WBC 训练不一致` 直接退出。

---

## 5. 新增/修改的文件

| 文件 | 作用 |
|---|---|
| `scripts/isaac_bw_pert_collect.py`（新） | B 分布 + 扰动的向量化采集（支持 token 空间与**动作空间**两种扰动，`--action-sigmas`） |
| `scripts/run_bw_pert_batch.sh`（新） | 批启动器（`GPU=` 钉卡、`SMOKE=1`、前后断言 B 池文件数不变） |
| `scripts/abl_diff_args.py`（新） | 两次训练 `args.json` 的机械 diff，`--expect <knob>` 断言"只翻了一个旋钮" |
| `scripts/s3_pipeline.sh` | `WM_PERT_GLOB/WM_PERT_SHARE/WM_PERT_SEG_LEN`；P 只进阶段2b；`WM_SKIP_P1`；P 配比断言 |
| `scripts/finetune_wm_velocity_policy.py` | 第三池 P（`--pert-dump/--mix-pert/--pert-seg-len`，窗口不跨段）、`--a-order` |
| `docs/ABLATION_SUITE.md` | 参考臂从 d3 改为"d4 配方"；新增 §2.7 陷阱表（含"跨会话数字不可比"） |
| `runs/Bw_pert/`（3.69M 行） | token 空间扰动（σ 0.03–0.30），**负结果，勿用** |
| `runs/Bw_pert_low/`（2.21M 行） | 上面筛 σ≤0.12，**负结果，勿用** |
| `runs/Bw_pert_a/`（3.69M 行） | 动作空间扰动（σ_a 0.02/0.05/0.10），**负结果，勿用** |

**所有 P 相关默认应关掉**（`WM_PERT_SHARE=0`），避免误用。

---

## 6. 如何复现/续跑（命令照抄）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=$PWD/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd

# 只跑 WM 刷新（阶段1/2/2b），需要先 seed 状态文件（transitions 产物由 d3 复用）
mkdir -p runs/S3_pipeline/<ROUND>
python3 -c "import json,pathlib;pathlib.Path('runs/S3_pipeline/<ROUND>/state.json').write_text(json.dumps({'stages':{'transitions':{'artifact':'runs/wm_real/rt_v2_d3.pt','at':'x','epoch':0,'extra':''}}},indent=1))"
ROUND=<ROUND> CUDA_VISIBLE_DEVICES=1 bash scripts/s3_pipeline.sh wm_refresh

# 复用阶段1并跳过（阶段1 产物复制成 wm_<ROUND>_a.pt 即可）
cp runs/S1_handover/wm_d3.pt runs/S1_handover/wm_<ROUND>_a.pt   # 或复用已训好的 _a
ROUND=<ROUND> WM_SKIP_P1=1 CUDA_VISIBLE_DEVICES=1 bash scripts/s3_pipeline.sh wm_refresh

# 评估（同一留出集，与基线并排）
PYTHONPATH=. python3 scripts/eval_wm_multistep.py \
  --ckpt runs/S1_handover/wm_d3.pt runs/S1_handover/wm_<ROUND>.pt \
  --dump "runs/wm_real/traj_wmuse2_val/*.pt" --horizon 20 --stride 20 --device cuda:1
```

**留出集口径**：`runs/wm_real/traj_wmuse2_val/`（= SitSofa 全池留出，134 条）。所有 WM 实验必须用**同一**留出集，否则不可比。

---

## 7. 当前正在做的事：GRU 版 WM（已定稿，实现中）

| 项 | 取值 |
|---|---|
| 架构 | **GRU 1 层，hidden = 1024**（用户定） |
| 头部 | **保持 3 层：1024 → 2048 → 2048 → 70**（与现有 MLP 同宽，输出状态增量，dt=0.02 不变） |
| 参数量 | ≈ 9.98M vs 基线 8.80M（**+13%**，断言阈值 ±20%） |
| K | **20，钉死**（不再用 K=8） |
| 数据/超参 | 与 `wm_d3` 完全一致 |
| 关键点 | 隐状态跨 20 步保留并**全程反传**（这是"加记忆"的全部意义） |
| 保护 | 梯度裁剪 + lr 预热；加载/保存兼容 `--arch mlp`（默认，逐位不变） |
| 自测 | `scripts/test_wm_arch.py`（纯 CPU：参数量、前向形状、"改隐状态必须改输出"） |
| 状态 | **已交子代理实现（agent id `849723b3-54b7-48a3-9bf2-7615f300a066`），仅实现+自测+dry-run，未启动训练** |

**为什么用 GRU（而不是加宽 MLP）**：现有 WM 是**一阶马尔可夫、无记忆**的积分器，20 步误差 65× 于 1 步（0.1 cm→6.5 cm），符合"误差有偏 + 状态漂移放大"的特征。惯性、接触记忆（几百 ms）在原理上一阶模型表达不了。GRU 提供隐状态 + 多步可导，且**头部宽度不变** ⇒ 提升可归因到"记忆"而非"容量"。

**启动条件**：子代理交付后先看 `[wm-arch]` 参数打印与自测，再跑阶段1（从头训，因结构不同不能从 `wm_d3` 初始化）→ 阶段2/2b（K=20）→ 用 §6 命令评估三点（k=1/8/20）。

**判定**：k=1 ≤0.2 cm **且** k=20 ≤3 cm ⇒ 记忆假设成立、继续加数据/加深；否则说明瓶颈是**数据规模**（现 63 万条 vs SuperMimic 10⁹），转下面的路线 B。

---

## 8. 备选路线（GRU 失败时按序尝试）

| 路线 | 做法 | 预期 | 成本 |
|---|---|---|---|
| **A. 数据放大（最像 SuperMimic）** | A 线 512→4096 env、8 波→40 波（≈4,900 万行，实测 **30 s/波**）；WM 步数 ×3–10；**K=20** | 可能把 k=20 压到 3–4 cm，**仍不够**（需 ≤1 cm） | 采集 20 min + 训练 1–1.5 h |
| **B. 扩 B 线** | B 从 42 集/轮 → 300 集（≈4 h，VLA 速度限制 ~1 min/集）；WM 混合里 **B 占 70%**（`WM_AB_RATIO` 1→0.43） | 分布更贴部署；k=20 可能改善 | 4 h 采集 |
| **C. H 压到 2–4 + critic 用真实回报** | 承认 WM 只准 1–2 步（k=1 0.1 cm、k=2 ~0.3 cm），长视界交给**真实数据学的 V**（`--critic-source real`，代码已实现、四轮都没用过）；`--critic-state-grad-scale 0.1→1.0` | **唯一能立刻落地且保住端到端梯度**的方案 | 需重建 `real_returns.pt`（现有是 10-26 的旧文件，WBC 与奖励配置都变了） |
| D. A 线关节顺序修正 | 见 §4.2，`--a-order declared` | 未知，可能显著 | 10 分钟/次 |

**注意**：C 中"critic 用真实回报"不替换 WM，只是把**长期价值**换成真实数据学的；梯度路径仍是 `tokens → WBC → WM(H 步) → V → 反传`，**这是目前唯一同时满足"端到端"与"用真实信息"的方案**。

---

## 9. 用户口径（决策记录，勿违背）

1. **每次启动训练前必须先问**：说明改了什么、准备怎么启动，得到同意再跑。
2. 不做 AWR / DAgger 作为主方案——要的是**完整端到端梯度反传**。
3. **K=20 钉死**，不再用 K=8（K=8 是我为塞下 20-tick 段引入的，已废弃）。
4. 执行器 = 仿真里的真实执行（物理不可微），**WM 是唯一可微替代**；执行器网络本身（WBC）可微、权重冻结但梯度可穿。
5. 口径：不要"改了啥"式的创新点叙述；只写架构与创新点本身。
6. 论文里负结果要如实写（I5：想象回报不能当裁决器，现在扩展到"想象梯度在本设置下贡献可测为零"）。

---

## 10. 一句话给接手者

> **WM 是这条路线唯一的瓶颈。** 现在 `wm_d3.pt`（k=1 0.1 cm / k=20 6.5 cm）是唯一可用裁判；四次"补反事实数据"的尝试全部失败并已排除扰动空间/幅度/占比/阶段四个解释；下一步是 **GRU 版（1 层 1024 + 3 层 2048 头，K=20，参数 +13%）**，若它把 k=20 压到 ≤3 cm 就继续投，否则转 **数据放大（A 线 4900 万行）** 或 **H=2 + critic=real**（唯一保住端到端的落地方案）。

---

## 11. 附：GRU 版交付状态（2026-10-31 晚，已验收）

子代理已交付并自测通过：

| 项 | 结果 |
|---|---|
| 文件 | `scripts/finetune_wm_velocity_policy.py`（改）、`scripts/test_wm_arch.py`（新，纯 CPU） |
| 参数账 | `[wm-arch] arch=mlp params=8804352` / `arch=gru params=9989120`（**+13.46%**，守卫阈值 ±20%，超限直接拦） |
| GRU 形状 | `nn.GRU(133,1024,num_layers=1)` → `Linear(1024,2048)+LN+SiLU` → `Linear(2048,2048)+LN+SiLU` → `Linear(2048,70)`；末层零初始化；`_integrate` 与 MLP 的 `step` 逐行一致，dt=0.02 不变 |
| 记忆接线 | `step_h(S,a,xi,h)->(state,h)`，K 步循环后**一次 backward**（整段 BPTT）；自测证明"第 K 步的梯度只能经 h 回到第 1 步的动作"（无状态对照梯度为 0） |
| 兼容 | `--arch mlp` 逐位不变；GRU 从 MLP ckpt 初始化时只复用 `net.9→head.6` 与 `in_norm/out_norm`，其余 FRESH（会打印清单）；存档含 `wm_arch="gru"`，可严格重载 |
| BPTT 保护 | `--clip-norm`（默认 1.0，新增可调）、GRU 专属 lr 线性预热 `min(200,steps)` |
| 自测 | `python3 scripts/test_wm_arch.py` PASS；`--help` 可见 `--arch {mlp,gru}` |

**⚠️ 注意（子代理给的示例命令不要直接用）**：它的 `--rollout-steps 3 --steps 600 --ckpt WBC...` 只是语法示例，与我们的实验口径不符。**正确启动必须走管线**（§6），命令为：

```bash
# 阶段1（A 线，从头训 GRU；不能从 wm_d3 初始化，只复用末层与归一化统计）
# 实现方式：给 s3_pipeline.sh 的 stage_wm 加 WM_ARCH 透传后
ROUND=wmgru WM_ARCH=gru CUDA_VISIBLE_DEVICES=1 bash scripts/s3_pipeline.sh wm_refresh
#   阶段2/2b 必须 K=20（WM_ROLLOUT_STEPS 默认即 20，不要再改 8）
# 评估（与 wm_d3 同留出集）
PYTHONPATH=. python3 scripts/eval_wm_multistep.py \
  --ckpt runs/S1_handover/wm_d3.pt runs/S1_handover/wm_wmgru.pt \
  --dump "runs/wm_real/traj_wmuse2_val/*.pt" --horizon 20 --stride 20 --device cuda:1
```

**尚未做**：`s3_pipeline.sh` 里还没有 `WM_ARCH` 透传（需在 stage_wm 三处命令行加 `--arch "$WM_ARCH"`，并让阶段1 也允许 `--arch gru`）。这是接手者的**第一件事**，改动约 6 行。

**评估注意**：GRU 的 `val_horizon=20` 数值**不能直接和 MLP 比**（前者带记忆、后者不带），必须用 `eval_wm_multistep.py` 在同一留出集上跑 k=1/5/10/20 与 `wm_d3` 并排比较。

---

## 11. GRU 版结论（2026-10-31 已跑完，**负结果，勿重复**）

实现侧接管审查发现 4 个真 bug（评估侧读不了 GRU ckpt、cuDNN eval 模式不能反向、`bundle.step`
不传记忆、续训 strict=False 吞错），已全部修复并把结构代码收进 `diff_vla_wbc/wm_arch.py`
（训练/评估同一份）。自测 `scripts/test_wm_arch.py` 覆盖：参数量（8,804,352 / 9,989,120，+13.46%）、
MLP↔GRU 积分等价、BPTT 穿 K 步、隐状态生命周期、ckpt 往返。**MLP 基线回归门：评测数字逐位不变（max|Δ|=0.0）**。

**精度结论（同一把裁判：134 条留出、k=1/5/10/20）**：

| | k=20 root | k=20 joint | k=20 rot |
|---|---|---|---|
| MLP 最好（`wm_d3_k20last.pt`） | **5.7 cm** | **0.158** | **12.4°** |
| GRU 最好（`wm_mem4_k20.pt`） | 7.3 cm（+28%） | 0.185（+17%） | 14.1°（+14%） |

- **记忆热启动诊断**（`eval_wm_multistep --warmup 10 --warmup-mode teacher`：用真实历史把隐状态填满，
  对 MLP 逐位零变化）⇒ GRU **反而更差**（k=20 9.4→9.7 cm）⇒ **隐状态里没有可用信息**，不是接线问题。
- **配方事实**：d3 的阶段1/2 是 K=1 ⇒ `weight_hh` 梯度严格为 0（自测里有断言），
  7500 步里前 6000 步对记忆无效；改成 K=4 从第一阶段就训记忆（`mem4` 线）也只把 k=20 位置做到 8.8→7.3 cm，
  同时 k=1 退化到 0.2 cm —— **仍不如 MLP**。
- **附带收获（可直接用）**：d3 的 2b 早停用**关节 RMSE**，与位置/姿态**反向**。
  同一条轨迹跑满 1500 步 → root **6.5→5.7 cm（−12%）**、rot **13.9→12.4°（−11%）**（代价 joint +10%）。
  ⇒ 早停指标应改成 `root + λ·rot`（与奖励同口径）。ckpt 已存为 `runs/S1_handover/wm_d3_k20last.pt`。

**复现**：`ROUND=d4gru|mem4 ARCH=gru bash scripts/run_wm_gru_ablation.sh {stage12|2b <tag> <steps> <lr>|eval}`；
汇总成表：`python3 scripts/compare_wm_eval.py --baseline wm_d3_k20last.pt runs/S1_handover/eval_wm_*.json`。
