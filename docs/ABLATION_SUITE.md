# 消融实验（当前版）

> 只保留**能改变结论**的消融。每个消融回答一个具体的因果问题；回答不了问题的、或者已经被更便宜的证据回答过的，一律列在 §4「不做」里并写明理由。
> 10-24 的旧版（A1–A5：有无联合微调 / FM 反传方式 / PPO-BPTT / WM 冻结 / 冻结执行器）已归档为
> `docs/ABLATION_SUITE_20261024_archive.md` —— 那批针对的是"训练架构"，不是当前的闭环组件。

## 0. 一页速览

| 编号 | 消融（只翻一个旋钮） | 回答什么问题 | 成本 |
|---|---|---|---|
| **AB1** | 关**动作锚**：`ACT_WEIGHT 1.0 → 0` | 闭环提升到底是不是锚点带来的？（I4 的核心） | ~45 min |
| **AB2** | 关**rotlocal6**：`STATE_EXTRA rotlocal6 → none` | 把"相对开局朝向"喂进 state，值多少？（I7） | ~45 min |
| **AB3** | 关**想象奖励**：`LAMBDA_R 0.3 → 0` | 奖励的 pathwise 梯度到底有没有在干活？ | ~45 min |
| **AB4** | 关**配对基线**：`ADV_BASELINE self → none` | `adv = ret − ret_exec` 这层降噪值多少？ | ~45 min |
| AB5（可选） | 关**critic 自举**：`CRITIC_ON 1 → 0` | `boot` 项（占目标约 30%）是否必要 | ~45 min |
| AB6（可选） | 关**yaw 远场 hinge**：`YAW_HINGE_LAMBDA 0.3 → 0` | 两段式核的"梯度覆盖"是否真有用（I3） | ~45 min |

成本口径 = 单臂 800 步（≈30 min，单卡）+ 12 集配对闭环评测（≈15 min，双卡并行）。
两条消融可**同时**跑（各占一张卡，见 §2.6）⇒ 4 条必做 ≈ 1.5–2 h。

**铁律（所有消融共用）**

1. **只翻一个旋钮**，其余与"完整方法"逐参数相同：同一份数据（`runs/B_abl`，829 个 dump）、
   同一个 WM（`runs/S1_handover/wm_d3.pt`，**不重训**）、同一个起点
   （`runs/S3_loop/d2b/train_main/vla_final.pt`）、同样 800 步、同样 `H=8 / batch 16×4 / lr 3e-5`。
   理由：本项目的最大教训就是"一次改 5 样 ⇒ 提升归因不了"——d3 就是这么跑的，
   AB1–AB4 存在的唯一目的就是把它拆开。
2. **参考臂是"完整方法"，不是 π₀**。要回答的是"这个组件值多少"，
   而"整套方法值多少"已由 d1/d2/d2b 的链式迭代回答：

   > ⚠ 2026-10-31 订正：**参考臂不是 d3**。同会话 A/B（d2b vs d3，12 集）判 d3 不优于 d2b：
   > root +0.020 m（t=+1.10，CI 跨 0）、**tilt +1.95°（t=+2.58，仅 1/12 更好）**——因为 d3 把
   > `W_TILT` 从 0.5 砍到 0.2，闭环正好兑现了这个交换（tilt 变差、位置没换来）。
   > ⇒ AB1–AB4 的参考臂与起点改为 **d4 的配方**（d2b head + W_TILT 回 0.5 + 新数据），
   > 下面命令里的 `d3_joint` 路径在 d4 跑完后同步替换。
3. 配对差 **> 0** ⇒ 该组件必需；**≈ 0** ⇒ 可去掉或收益未观测；**< 0** ⇒ 该组件有害。
   判决口径与主结果一致：配对差 + `t` + 胜率（x/12）+ 中位 + p90 + 摔倒率；`t ≥ +2 且 ≥9/12` 才算"组件必需"。

| | 参考臂 | 消融臂 |
|---|---|---|
| head | `runs/S3_loop/d3_joint/train_main/vla_final.pt` | `runs/S3_loop/ablN_joint/train_main/vla_final.pt` |
| 执行器 | `WBC_A24000_wmvel_v6.pt`（同一份冻结） | 同一份 |
| 种子 | `0..11`，两边**同序** | 同序 |

---

## 1. 每个消融的判读表

### AB1 · 动作锚有没有用（测 I4）

| 结果 | 解释 | 动作 |
|---|---|---|
| 消融臂显著更差（差 > +0.03，t > 2） | 锚点是主驱动，与 d1–d2b 的推断一致 | 保留，并按 `act_weight` 课程退场，看收益是否交接给奖励 |
| 两者相当 | **锚点是安慰剂**；d 链的增益来自"on-policy B 数据 + 更强奖励" | 撤回 I4、重估整条路线；退场直接做掉（不占算力） |
| 消融臂更好 | 锚点在拖后腿（把策略钉在漂移行为附近） | 立即降/关锚点 |

这是**唯一能证伪我们核心结论**的实验，优先级最高。

### AB2 · rotlocal6 值多少（测 I7）

- 已知 `scale_absmax` 全程 1.0000→1.0013 ⇒ 6 维只是"喂进去了"，没有被学会调节；
  本消融量的正是**喂进去本身**的收益。
- 若配对差 ≈ 0：说明"相对朝向"在当前任务上没有可测收益（视觉通道已部分覆盖），
  I7 降级为"设计正确、收益未观测"，并**停止**在它上面追加投入（如可训练编码器）。

### AB3 · 奖励有没有在干活

- 关掉 `λ_r` 后唯一的训练信号是动作锚 ⇒ 等价于"纯自模仿"。
- 若与完整方法相当 ⇒ **奖励仍是摆设**（与 I5 的负结果一致）；后续应把力气放在
  "提高奖励可分辨性"或"改执行锚点"，而不是继续改奖励项。
- 若明显更差 ⇒ **第一次拿到"奖励梯度有用"的证据**；这不与 I5 矛盾——
  I5 限制的是"用回报裁决"，不是"用回报求梯度"。

### AB4 · 配对基线值多少

- 内部读数已有：`adv/std_ratio` ≈ 0.055（方差降到 5.5%）。但**降噪 ≠ 提升**，必须闭环回答。
- 若 ≈ 0：说明方差不是瓶颈（或已被 Adam 吸收），可以去掉这层复杂度。

### AB5 / AB6（可选）

- **AB5**：`boot` 约占目标的 30%（`loss/boot −0.015` vs `rollout/reward 0.039`），但依赖 critic 的
  准确度（当前 `v_ev 0.996`）。若 AB3 显示奖励有用，再单独测 boot。
- **AB6**：活性审计已证明远场有梯度、且占朝向奖励 35%；要证明"它比没有更好"仍需闭环。

---

## 2. 怎么跑（照抄即可，2026-10-31 全部 dry-run 验证过）

### 2.1 一次性：钉住参照数据（只做一次）

消融轮**不采新数据**，用的必须是 d3 当时那一份 829 个 dump。`runs/B_*` 会随后续轮次变长，
所以用符号链接树把它冻住（不占空间）：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
mkdir -p runs/B_abl
for d in runs/B_*; do b=$(basename "$d"); [ "$b" = B_abl ] || { [ -d "$d" ] && ln -sfn "$PWD/$d" "runs/B_abl/$b"; }; done

# 验收：必须 = 829（= d3 的池子；换数字说明池子变了，先别跑）
ls runs/B_abl/*/vla_*_f*_r*/dump/*.pt | grep -v '_r.pt$' | wc -l
```

### 2.2 公共环境块（每一项都取自 d3 的 `args.json`）

```bash
COMMON=(
  ROUND=__ABL__ \
  ABL_ROUND=1 DRY_RUN=0 \
  WM_PIN=$PWD/runs/S1_handover/wm_d3.pt \
  INIT_VLA=$PWD/runs/S3_loop/d2b/train_main/vla_final.pt \
  B_GLOB=runs/B_abl/*/vla_*_f*_r*/dump/*.pt \
  B_FRESH_GLOB=$PWD/runs/B_d3/*/dump/*.pt B_FRESH_PIN=1 B_FRESH_WEIGHT=3.0 \
  STEPS=800 BATCH=16 ACCUM=4 H=8 TRAIN_ARMS=main \
  ACT_EPS=0.60 ACT_WEIGHT=1.0 \
  STATE_EXTRA=rotlocal6 ADV_BASELINE=self ADV_TAU=0.02 \
  TAIL_CAP_SLOPE=0.5 TAIL_WEIGHT=2.0 ROOT_ORI_MODE=split \
  REWARD_ROOT_WEIGHT=1.0 REWARD_ROOT_POS_STD=0.2 REWARD_ROOT_ORI_STD=0.25 \
  B_START_MAX_ERR=0.3 LAMBDA_R=0.3 YAW_HINGE_LAMBDA=0.3 CRITIC_ON=1
)
```

> `ABL_ROUND=1` 做两件事：① 允许 `WM_PIN` 跳过 collect/gate（正常轮次不允许，双保险）；
> ② **冻结 `act_weight` 课程**——否则它会按 d3 的 `frac_out=0.2266<0.35` 自动退到 0.5，
> 等于每条消融都多翻一个旋钮（2026-10-31 用 dry-run 抓到，之前差点全部作废）。

### 2.3 六条命令（每条只改一个变量）

```bash
R=abl1; env "${COMMON[@]/__ABL__/$R}" ACT_WEIGHT=0  bash scripts/s3_pipeline.sh joint  # AB1 关动作锚
R=abl2; env "${COMMON[@]/__ABL__/$R}" STATE_EXTRA=none       bash scripts/s3_pipeline.sh joint  # AB2 关 rotlocal6
R=abl3; env "${COMMON[@]/__ABL__/$R}" LAMBDA_R=0             bash scripts/s3_pipeline.sh joint  # AB3 关想象奖励
R=abl4; env "${COMMON[@]/__ABL__/$R}" ADV_BASELINE=none      bash scripts/s3_pipeline.sh joint  # AB4 关配对基线
R=abl5; env "${COMMON[@]/__ABL__/$R}" CRITIC_ON=0            bash scripts/s3_pipeline.sh joint  # AB5 关 critic 自举（可选）
R=abl6; env "${COMMON[@]/__ABL__/$R}" YAW_HINGE_LAMBDA=0     bash scripts/s3_pipeline.sh joint  # AB6 关 yaw 远场 hinge（可选）
```

产物：`runs/S3_loop/${R}_joint/train_main/{vla_final.pt,args.json,train_metrics.jsonl}`。

### 2.4 三道闸门（**顺序不能省**，漏一道就可能白跑 30 分钟）

**闸 1 · 先 dry-run 看生效值**（不启动训练，10 秒）

```bash
DRY_RUN=1 R=abl1; env "${COMMON[@]/__ABL__/$R}" DRY_RUN=1 ACT_WEIGHT=0 bash scripts/s3_pipeline.sh joint
# 必须出现：joint 生效值  : act_eps=0.60, act_w=0, lambda_r=0.3, yaw_hinge_lambda=0.3, ...
# 六个消融的期望值（act_w 全部 = 1.0，除 AB1）：
#   AB1 act_w=0 | AB2 state_extra=none | AB3 lambda_r=0 | AB4 adv_baseline=none
#   AB5 critic=0 | AB6 yaw_hinge_lambda=0
```

**闸 2 · 跑完对 args.json 做机械 diff**（证明真的只翻了一个旋钮）

```bash
python3 scripts/abl_diff_args.py \
  runs/S3_loop/d3_joint/train_main/args.json \
  runs/S3_loop/abl1_joint/train_main/args.json --expect act_weight
# 期望：✅ 只翻了期望的旋钮: ['act_weight']
# 退出码 1 = 期望之外还动了别的（这条消融作废）；2 = 期望的键没变（开关没接通）
```

期望键对照：AB1 `act_weight` / AB2 `state_extra` / AB3 `lambda_r` / AB4 `adv_baseline` /
AB5 `critic` / AB6 `yaw_hinge_lambda`。

**闸 3 · 配对闭环评测（消融臂 vs 完整方法，12 集）**

```bash
TAG=abl1_vs_d3 SEEDS="0 1 2 3 4 5 6 7 8 9 10 11" START_FRAME=0 \
  HEAD_MAIN=$PWD/runs/S3_loop/d3_joint/train_main/vla_final.pt \
  HEAD_CTRL=$PWD/runs/S3_loop/abl1_joint/train_main/vla_final.pt \
  SM_CKPT_MAIN=$PWD/runs/S1_handover/WBC_A24000_wmvel_v6.pt \
  SM_CKPT_CTRL=$PWD/runs/S1_handover/WBC_A24000_wmvel_v6.pt \
  LABEL_MAIN="full(d3)" LABEL_CTRL="no-anchor" \
  EXTRA_MAIN="--sample-avg 4" EXTRA_CTRL="--sample-avg 4" \
  bash scripts/s3_eval_sweep.sh
# 汇总：runs/S3_loop/sweeps/abl1_vs_d3/summary.json（含每一对的配对差、CI、胜率）
```

> 注意方向：这里 `main` = 完整方法、`ctrl` = 消融臂。`Δ = ctrl − main`，所以
> **Δ>0 才表示"关掉这个组件更差"**（组件必需）；Δ<0 表示关掉它更好。

### 2.5 一句话封版检查

```bash
for r in abl1 abl2 abl3 abl4; do
  printf "%-6s " $r
  python3 scripts/abl_diff_args.py runs/S3_loop/d3_joint/train_main/args.json \
    runs/S3_loop/${r}_joint/train_main/args.json --expect "$(case $r in abl1) echo act_weight;; abl2) echo state_extra;; abl3) echo lambda_r;; abl4) echo adv_baseline;; esac)" \
    | tail -1
done
```

### 2.6 两条消融同时跑（每张卡一条，时间减半）

```bash
R=abl1; env "${COMMON[@]/__ABL__/$R}" GPU_MAIN=0 ACT_WEIGHT=0 bash scripts/s3_pipeline.sh joint &
R=abl2; env "${COMMON[@]/__ABL__/$R}" GPU_MAIN=1 STATE_EXTRA=none bash scripts/s3_pipeline.sh joint &
wait
```

> 只在**两个不同 ROUND** 之间并行；同一 ROUND 的两臂并行还是走 `TRAIN_ARMS="main ctrl"`。

### 2.7 已知陷阱（都是踩过的）

| 现象 | 原因 | 现在由谁挡住 |
|---|---|---|
| 消融臂 act_w 静默变成 0.5 | `act_weight` 课程按上轮 `frac_out<0.35` 退场，d3 结束时刚好 0.2266 | `ABL_ROUND=1` 冻结课程；dry-run 打印 |
| AB3 跑完和完整方法一模一样 | `LAMBDA_R` 曾写死在 stage 里（0.3） | 改成环境变量 + 闸 2 的 `--expect lambda_r` |
| 训练侧 `fresh glob` 0 匹配直接崩 | `runs/B_ablN/` 不存在（消融不采数据） | `B_FRESH_PIN=1` 显式钉住 + 断言非空 |
| 训练输出跑到 `runs/S3_loop/r99/` | env 续行链里插了注释，`\` 被吃掉 | 启动自检 + `[[ -d .../$jl/train_main ]]` 断言 |
| `rm -rf` 把上次的 head 删了 | `s3_loop` 每臂开跑前清空 `train_<arm>` | 别复用 ROUND 名；重跑就换 `abl1b` |
| 跨会话比两个号 | d3 在两场里的均值是 0.150 / 0.176，波动 ≈ 效应量本身；跨会话准配对给 −3 mm、同会话 A/B 给 +20 mm，**符号相反** | 只认同会话配对 A/B；`s3_eval_sweep.sh` 两臂同场同时跑 |

---

## 3. 预算

| 方案 | 单次（训练+评测） | 4 个必做 | 6 个全做 |
|---|---|---|---|
| 单卡串行 | ~45 min | ~3 h | ~4.5 h |
| **两卡并行**（§2.6） | ~45 min | ~1.5 h | ~2.2 h |

## 4. 不做（以及为什么）

| 曾经的候选 | 为什么不做 |
|---|---|
| `--sample-avg 1 vs 4` | 已有统计结论：配对 **−0.097**（avg4 更好），重做无新信息 |
| WM 数据里混不混参考库 | 已由门的 **OOD 栏**每轮监测；且它回答"泛化"，不是"方法组件" |
| `w_lambda` / `yaw_rate` 权重 | `yaw_rate` 因幅度比 1.40、相关性 0.39 **未达准入**，当前为 0；对关着的项做消融无意义 |
| `adv_tau`（0.02 vs 0.005） | 实测 `frac_inside_tau = 1.0` ⇒ `tanh` 全在线性区；换 τ 等价于换常数缩放，Adam 下无区别 |
| `tail_cap_slope`（0 vs 0.5） | 只影响远场（误差 > cap）的坡度；判决指标是中位/p90，远场贡献可忽略 |
| A 线 4 波 vs 8 波 | 这是**数据量**不是方法组件；A 线采集 8 分钟，要测可直接重采 |
| 200 / 400 / 800 步 | 已由 d1(800) / d2(200) / d2b(+600) 给出"数据比步数更值钱"的证据 |
| 10 轮 × 10 阶段的进度表 | 不是消融对象；它由 `s3_pipeline.sh` 锁死 |
| 旧 A1–A5（10-24 版） | A1 已由每轮 π₀ 配对臂回答（12 集，比原计划 6 集更强）；A5 的反面臂有 §10.1 的 21× 崩塌记录；A2/A3/A4 是**已冻结的设计选择**，不是论文声称的创新点 |

## 5. 论文写作映射

| 论文里的说法 | 需要的证据 | 消融 |
|---|---|---|
| "执行锚点提供稠密可执行信号" | 关掉后闭环显著变差 | AB1 |
| "相对朝向使 yaw 可观测" | 关掉后显著变差 | AB2 |
| "想象奖励提供可用的策略梯度" | 关掉后显著变差 | AB3 |
| "配对基线降低梯度方差" | 关掉后变差，或给出方差读数 + 等效结论 | AB4 |
| "两段式核覆盖远场梯度" | 关 hinge 后 p90 变差 | AB6 |
| "想象回报不能用作裁决器"（负结果） | 128 窗 `imag_delta` 在噪声地板内 | 无需额外跑（已有） |
