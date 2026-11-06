# 交接：联合微调（想象梯度）迁到 8×4090 + 已知问题/解决/实验计划

> 日期：2026-11-01 03:20（写于 2 卡 4090×48GB 环境，任务迁到 **8 卡 4090** 继续）
> 关联：`docs/WM_JOINT_ORDER_BUG_20261031.md`（394 行，全部正/负结果与证据链）、
> `docs/WM_GRU_ABLATION_20261031.md`、`docs/HANDOVER_WM_20261031.md`
>
> **★ 一句话现状（2026-11-01 04:36 更新）**：
> 1. WM 已修好（k=20 位置 −40%）、**梯度方向被真物理验证 98–100% 可信（H=2/8/16/20）**、
>    策略对 token 的权限足够（余量 6×）；
> 2. **把 H 从 8 提到 20（信用覆盖 40%→100%）拿到了本轮第一次统计显著的闭环改善**：
>    `d5jH20` **0.1373 m vs d2b 0.1525 m**，配对差 **−0.0157 m**，**95%CI [−0.0249, −0.0076]（不含 0）**，
>    10/12 集更好，**tilt 7.9°→6.3°**；
> 3. 但**单纯放大模型项影响力是有害的**（`grad_scale ×10` ⇒ **+0.022 m 显著变差**）；
> 4. ⇒ 结论修正为：**"信用覆盖"是关键，"影响力大小"不是**。8 卡上要做的第一件事是
>    **把 H=20 的结果做稳健性确认**（换 NPROC / 换种子集 / 加到 20 集），然后 P1（λ_r 折回 0.12，
>    分离"覆盖"与"放大"）、P2（H=20 + aw=0.5）、再谈 WBC。

> **2026-11-01 04:20 更新（P0 已跑完，8 卡 DDP）**：H=20、权重不变在 **32 集配对闭环**上
> **−0.95 cm**（95%CI [−1.55, −0.27]，28/32 更好）；改善集中在后段：200–299 步 **−1.68 cm**、
> 最后 50 步 **−2.10 cm**；训练内 `L_fm −33%`、`imag_ret +7.8%`、`imag_delta 改善 52%`。
> ⇒ **信用覆盖本身确实带来闭环收益**，"闭环收益 ≤0"在该旋钮上不再成立。
> 详见 `docs/H20_8GPU_JOINTFT_20261101.md`（含 DDP 修复、逐步 profile、复现命令）。
> ⚠️ 8 卡 `NPROC=8, accum=1` 时 `STEPS` 必须用 **200**（=200 次更新 × 有效 batch 64）；
> 照抄 §0 的 `STEPS=800` 会变成 800 次更新 / 51,200 样本（4× 训练量），与历史臂不可比。
> 下一步建议：P1（λ_r=0.12，分离"覆盖 vs 影响力"）→ P2（WBC，仍是最大杠杆）。

> **2026-11-01 05:10 更新（第二轮：放锚点 vs 加 λ_r）**：H=20 + `act_weight 1.0→0.5`
> （λ_r 仍 0.3）⇒ **−1.36 cm**（16 集，95%CI [−2.05,−0.74]，14/16；后段 −3.52 cm、
> 最后 50 步 −5.46 cm），比 P0 的 −0.95 cm 更大；同轮 `λ_r 0.3→0.6` + act_w=0.5（AB）
> ⇒ **+1.08 cm 且失控**（`frac_out=0.51`，含 +26 cm 异常集）。
> ⇒ **放大杠杆是“放开动作锚点”，λ_r 维持 0.3；继续加 λ_r 进入不稳定区。**
> 注：本节上方 04:36 的 `d5jH20 −1.57 cm / 12 集` 与 04:20 的 `−0.95 cm / 32 集` 是
> **不同 episode 集**的两次配对（32 集含更难的 seeds 12–31），方向一致不矛盾；
> 同一 16 集（seeds 0–15）上 P0 为 −0.84 cm、B 为 −1.35 cm。

> **2026-11-01 05:30 更新（方案 LOCKED）**：后续所有联合微调轮次默认走
> **H=20 + λ_r=0.3 + act_weight=0.5 + 200 updates × 有效 batch 64**；
> 起点 = 上一轮最优，执行器冻结 `WBC_A24000_wmvel_v6`，评测 **32 集起（48 集出论文）**。
> 启动器：`scripts/run_locked_h20_round.sh`；完整规范/验收/停止判据：
> `docs/METHOD_LOCK_H20_20261101.md`。
> `act_weight` 暂固定 0.5；要降到 0.25 必须先跑固定臂 + 32 集闭环，再上“双闸门”课程
> （训练内 `frac_out<0.45` + 闭环不变差），不能只按步数退场。
> P2（WBC）作为**独立执行器 track**，排在锁定策略轮之后，不混进 joint。
> **第三池 P 已删除**（`runs/Bw_pert*` 是负结果，见 `docs/HANDOVER_WM_20261031.md`）：
> pipeline 不再有 `--pert-*` 接线，WM 只走 **A+B**；数据已归档到 `runs/_deprecated_pert/`。

---

## 0. 到 8 卡上第一件事（照抄）

> **先看结论**：本机（2 卡）已跑出 **H=20 有效**（−1.57 cm，CI 不含 0）。到 8 卡上**第一件事是复现它**：
> 用 `NPROC=4`（或 8）重跑同配置 + 12 集/20 集配对。若复现，再按 §5 的 P1/P2 往下推；
> 若不复现，先查"是不是 DDP 度/种子集造成的"。

```bash
cd <repo>
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=$PWD/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd

# ① H=20、权重不变（λ_r=0.3, aw=1.0）、8 卡 DDP（每卡 batch 8，全局 16×8=128？见 §5 的账）
ROUND=h20_8g ABL_ROUND=1 H=20 DDP=1 NPROC=8 MASTER_PORT=29521 \
  WM_PIN=$PWD/runs/S1_handover/wm_d5_k20.pt \
  INIT_VLA=$PWD/runs/S3_loop/d2b/train_main/vla_final.pt \
  STEPS=800 FM_LR=3e-5 ACT_EPS=0.6 ACT_WEIGHT=1.0 YAW_HINGE_LAMBDA=0 \
  REWARD_ROOT_POS_STD=0.2 REWARD_ROOT_ORI_STD=0.25 REWARD_ROOT_WEIGHT=1.0 \
  CRITIC_ON_MAIN=1 CRITIC_GRAD_STATE_MAIN=1 CRITIC_STATE_GRAD_SCALE=0.1 TRAIN_ARMS=main \
  PORT_MAIN=18097 RUN_LABEL=h20_8g \
  B_FRESH_GLOB="$PWD/runs/B_d3/*/dump/*.pt" B_FRESH_PIN=1 \
  BATCH=64 ACCUM=1 \
  bash scripts/s3_pipeline.sh joint

# ② 闭环配对评测（12 集；**务必**成对、同执行器）
TAG=d2b_vs_h20_8g SEEDS="0 1 2 3 4 5 6 7 8 9 10 11" START_FRAME=0 \
  HEAD_MAIN=$PWD/runs/S3_loop/d2b/train_main/vla_final.pt \
  HEAD_CTRL=$PWD/runs/S3_loop/h20_8g_joint/train_main/vla_final.pt \
  SM_CKPT_MAIN=$PWD/runs/S1_handover/WBC_A24000_wmvel_v6.pt \
  SM_CKPT_CTRL=$PWD/runs/S1_handover/WBC_A24000_wmvel_v6.pt \
  LABEL_MAIN=d2b LABEL_CTRL=h20_8g \
  EXTRA_MAIN="--sample-avg 4" EXTRA_CTRL="--sample-avg 4" \
  PORT_MAIN=18150 PORT_CTRL=18151 SRV_SEED=0 \
  bash scripts/s3_eval_sweep.sh
# 判据：配对差 ≤ −0.013 m 且 95%CI 不含 0 ⇒ 成立；否则见 §5 决策树
```

---

## 1. 现状：什么已经修好了（可直接引用）

| 项 | 数值 | 说明 |
|---|---|---|
| **A 线关节顺序 bug** | 已修 + 已加自检 | 见 §2；这是本轮最大发现 |
| **最好的 WM** | `runs/S1_handover/wm_a24_k20.pt` | k=20 root **3.9 cm** / joint **0.067 rad** / rot **7.3°** |
| 上一代 WM（对照） | `runs/S1_handover/wm_d5_k20.pt` | 4.1 cm / 0.066 / 7.3°（8 波 A 线） |
| 原基线 | `runs/S1_handover/wm_d3.pt` | 6.5 cm / 0.144 / 13.9°（**作废，勿再用**） |
| 动作雅可比 | 对角命中率 **0.966**（旧 0.724）、增益 ×2.7 | `probe_wm_action_jacobian.py` |
| **梯度方向** | 真物理 **98.3% / 100% / 98.3% / 100%**（H=2/8/16/20） | `probe_wm_gradient_truth.py` |
| **token→action 权限** | 只需 **0.15×** 数据自然幅度（余量 6×） | `probe_token_action_gain.py` |
| 起点策略 | `runs/S3_loop/d2b/train_main/vla_final.pt`（闭环 **0.153 m**） | 所有臂的共同起点 |
| 闭环上限 | GT 命令臂 **0.108 m** | 执行器物理极限，占闭环误差 ~70% |

---

## 2. 本轮修掉的 bug（**必读，避免重踩**）

### 2.1 A 线关节顺序错位（真 bug，影响所有历史 WM）

* 现象：`scripts/finetune_wm_velocity_policy.py` 默认 `--a-order ha_v31`，会对 **A 线**（`runs/wm_real/ac_w*.pt`）
  的 `states` 做 `ha_v31→standard` 换序，但**不动 `actions`**。
* 实测证据：A 线文件 `meta.joint_names` **与 spec 逐项相同**（standard），
  `corr(mean q, default_q)=+0.698`（原样）/ **+0.034**（换序后）；
  `corr(mean action, mean q−default)=+0.832`（原样）/ +0.055（换序后）。
  B 线恰好相反：`states` 是 ha_v31（过 perm 后相关 0.749）、`actions` 是 standard（H2 检验 +0.439 vs +0.021）。
* 后果：**阶段1 是 100% A 线**，3000 步全在学"动作 j → 别的关节"；阶段2 里 A 线占一半。
  而想象/评估侧一直是 standard ⇒ 训练/推理口径冲突。
* 修法：**`--a-order declared`（新实验必须用）**；并在 A 线加载处加自检
  （打印 `[a-order] 选用 declared: corr=+0.680（另一种 -0.009）`，选错直接 `SystemExit`）。
* 收益：WM k=20 位置 **6.5→4.1 cm**、关节 **0.144→0.066**、姿态 **13.9→7.3°**；
  雅可比命中率 **0.724→0.966**；想象 `state_term 0.760→0.593`、`reward +8.5%`。
* **影响面**：`ABLATION_SUITE` 的参考臂、四次反事实数据实验、I5 的旧解释、GRU 消融 —— 都需按新口径重看。

### 2.2 其它已修（实现层）

| # | 问题 | 修法 |
|---|---|---|
| 1 | GRU 结构只写在训练脚本里，评估侧（`isaac_joint.load_isaac_bundle`）读不了 GRU ckpt | 提到 `diff_vla_wbc/wm_arch.py`，`load_isaac_bundle` 按 `detect_wm_arch(ck)` 造模型 |
| 2 | cuDNN 的 RNN **反向只允许 training 模式**（bundle 加载后被 `.eval()`） | 训练循环每轮 `wm.train()` |
| 3 | `bundle.step` 不带隐状态 ⇒ 想象/闭环用 `h=0`（训练带记忆、部署不带） | `step_idx` 当生命周期信号：`==1` 清零 / `>1` 延续 / `None` 无状态 |
| 4 | 续训 GRU ckpt 用 `strict=False` 吞掉 pi 缺键 | 统一走 `load_isaac_bundle`（它自己判 arch），strict 恢复 |
| 5 | 复用 MLP 的 `in_norm/out_norm` 只搬 `mean/var`、没搬 `count` | 连 `count` 一起搬（否则下一次 `update()` 会把标定冲掉） |
| 6 | 2b 早停用**关节 RMSE**，与位置/姿态**反向** | 跑满 1500 步（`VAL_EVERY=0`）反而更好：root 6.5→5.7 cm、rot 13.9→12.4°（joint 0.144→0.158） |
| 7 | `--critic-state-grad-scale` 硬编码 0.1 | 加 `CRITIC_STATE_GRAD_SCALE`（默认仍 0.1） |
| 8 | VLA server 的 flow-matching 采样**没播种** ⇒ 同策略同 seed 三次跑出 0.064/0.143/0.152 | 加 `--seed`（`SRV_SEED`），残余 ~4.6% 来自 Isaac 内核 |
| 9 | sweep 的两个槽位**硬编码 GPU 0/1** ⇒ 一卡被训练占着就没法评测 | 加 `GPU_MAIN/GPU_CTRL`（两槽可以放同一张卡） |

---

## 3. 闭环实验的全部结果（同一批 episode、同一执行器、配对）

| 臂 | 相对 d5jT 的改动 | 闭环 root | 相对 `d2b` | 显著? | 训练内 |
|---|---|---|---|---|---|
| `d2b` | —（起点） | 0.1517–0.1536 | — | — | `L_fm=0.3891`（起点） |
| `d5jT` | **只换 WM**（关节顺序修好） | 0.169 | +0.014 m（9 集） | 否 | `L_fm −7.0%`、`imag_ret +3.2%` |
| `d5jR` | + `ACT_WEIGHT 1.0→0.5` | 0.1556 | **+0.004 m** | 否（CI ±1.3 cm） | `L_fm −2.1%`、`frac_out 0.32→0.37` |
| `d5jG` | + `grad_scale 0.1→1.0`（影响力 ×10） | **0.1758** | **+0.0223 m** | **是（更差）** | `L_fm +15%`、`frac_out→0.50` |
| lead β=0.3 | 只给 d2b 输出加提前量 | 0.1511 | **0.0000 m** | 否 | —（同策略，纯闭环） |
| `d5jC`（对照） | 新代码 + 旧 WM | 0.170（9 集） | +0.016 m | 否 | 与 d5jT 同起点 |

**同集配对的逐集差（d5jR − d2b）**：`[+0.014, −0.026, −0.004, +0.007, +0.032, +0.031, −0.044,
−0.018, +0.004, +0.010, +0.024, +0.016]`，均值 **+0.0038**，d5jR 更好 4/12。

**结论**：**影响力越大闭环越差**（单调）；WM 修好带来的正收益**没有**传到闭环。

---

## 4. 判定过的假设（不要重复）

| 假设 | 实验 | 结论 |
|---|---|---|
| WM 不准 ⇒ 梯度无效 | 修关节顺序 | ❌ 排除：WM −40%、梯度真物理 98–100% 可信 |
| 梯度是噪声 | 真物理 ±δ 线搜索 | ❌ 排除：方向 98–100% 对、幅度差 11–21% |
| 需要记忆（GRU） | 4 组配方 | ❌ k=20 joint **+70%**、rot +40% |
| 数据不够 | A 线 8→24 波（3×） | ❌ 只 **−6.6%**（到 1 cm 需 ~3^20 量级） |
| 锚点（动作球）太硬 | `ACT_WEIGHT 0.5` | ❌ 闭环 +0.004（CI 跨 0） |
| 长视界信用被削 10× | `CRITIC_STATE_GRAD_SCALE=1.0` | ❌ **闭环显著更差**（+2.2 cm） |
| 误差是系统滞后 | lead β=0.3（同策略配对） | ❌ 均值完全相同 |
| token 接口太小（权限不足） | `probe_token_action_gain` | ❌ 权限足够（只需 0.15× 数据幅度） |
| H 太短（信用只覆盖 40%） | H=20 梯度验证 | ✅ 方向可信、信号 3.5×；**训练臂正在跑/待跑** |

---

## 5. 8 卡上的实验计划（按优先级，含判据）

### 5.0 DDP 的账（**先算清楚再跑**）

`launch_joint_ha_v2.sh` 的约定：`DDP=1` ⇒ `torchrun --nproc_per_node=$NPROC`，
**每卡 batch = BATCH / NPROC**，梯度 all_reduce（`--sync-gradients`），每 rank 种子 +rank。
有效 batch = `per_rank_batch × NPROC × ACCUM`（DDP 已跨卡求平均）。

* 现有 2 卡跑的配置：`BATCH=16 ACCUM=4` ⇒ 有效 batch **64**、800 步。
* **8 卡保持有效 batch 不变**：`BATCH=64 ACCUM=1 NPROC=8` ⇒ 每卡 8、有效 **64** ✔
* ⚠️ **显存警告**：本机 2 卡是 48 GB（modded），2 卡 DDP 时每卡占用约 **30 GB**。
  **标准 4090 只有 24 GB** ⇒ 每卡 batch 8 **很可能会 OOM**。对策（按顺序试）：
  1. `BATCH=16 ACCUM=4 NPROC=8`（每卡 2，有效 64）—— 先保能跑；
  2. 关掉训练内评测的频率：`EVAL_EVERY=200`（评测也吃显存）；
  3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
* ⚠️ **可比性**：DDP 度（NPROC）不同，数值轨迹就不同（每卡独立抽样 + all_reduce 求和顺序）。
  ⇒ **同一轮的 A/B 必须用同一个 NPROC**；跨轮比较只看"配对差 + CI"，不看逐位。

### P0（先跑，~10 min 训练 + 20 min 评测）：**H=20 + 权重不变**

* 目的：验证"信用覆盖 40%→100%"本身是否带来收益（用户假设）。
* 命令：见 §0。
* 注意：H=20 下想象回报是**累加**的（`imag_ret` 从 0.34 → **0.79**，2.3×）⇒ 奖励项影响力自动 2.5×，
  这正是"权重不变"的含义。
* 判据：`frac_out` 别冲过 0.5；闭环配对差 **≤ −0.013 m 且 CI 不含 0** 才算成立。

### P0 的结果（本机已跑，2 卡 DDP）：**有效**

| 臂 | n | 闭环 root | tilt | p90 |
|---|---|---|---|---|
| `d2b` | 12 | 0.1525 ± 0.0328 | 7.9° | 0.191 |
| **`d5jH20`（H=20，λ_r=0.3，aw=1.0）** | 12 | **0.1373 ± 0.0381** | **6.3°** | **0.180** |

配对差 **−0.0157 m**，sem 0.0050，**95%CI [−0.0249, −0.0076] 不含 0**，**10/12 更好**，中位 −0.0150。
训练内：`L_fm_val 0.3511→0.2294`（比 H=8 的 d5jT 再好 **20%**，全程单调下降），
但 `act_frac_out=0.444`（逼近 0.5 护栏）、`codec_oor` 涨 5 倍 ⇒ 奖励项影响力已接近上限。

### P0.5（8 卡必做，~40 min）：**复现 H=20**
* 同配置换 `NPROC=4` + 12 集配对；再用 20 集把 CI 压到 ~1 cm。判据：配对差仍 ≤ −1.0 cm 且 CI 不含 0。

### P1（~30 min）：**H=20 + λ_r 折回 0.12**（影响力对齐 H=8）

* 若 P0 出现 `frac_out>0.45` 或 `L_fm` 明显变差，说明"权重不变"已过头 ⇒ 用 λ_r=0.12 把影响力拉回 H=8 水平，
  **单独**验证"覆盖增益"。
* 命令：P0 的命令 + `LAMBDA_R=0.12`（注意 pipeline 里 `lambda_r` 由 `LAMBDA_R_MAIN` 控制，见 `s3_loop.sh:43`）。

### P2（~2 h，最确定）：**执行器（WBC）**

* 理由：闭环 0.1536 里 **0.108 是 GT 命令臂的水平**（占 ~70%）。WM 已够用（k≤2 = 0.1–0.3 cm，梯度已验证）。
* 做法（二选一）：
  1. **单独**在部署分布上微调 WBC：冻结 VLA，用 VLA 实际产出的 token 流在 Isaac 里跑跟踪奖励，
     只训执行器（先 KL 锚住原 WBC，lr 1e-6，200–500 步）；
  2. 若一定要联合解冻：**只解冻解码器 + KL 锚 + 小 lr**，并且**每 50 步**量一次闭环（历史上全解冻 800 步
     直接崩到 89 cm，见交接文档 §10.1）。
* 判据：闭环 ≤ **0.135 m**（比 d2b 好 1.5 cm 以上且 CI 不含 0）。

### P3（~1 h）：**信用分配换机制**（不用 WM 反传）

* `--critic-source real`（`scripts/build_real_returns.py` 重建，现有的是 10-26 的旧文件）
  + H=2~4（WM 在 k≤2 是 0.1–0.3 cm）+ `CRITIC_STATE_GRAD_SCALE=1.0`。
* 或者直接 on-policy RL（真实回报），放弃反传。
* 判据同上。

### P4（评测基建，30 min）：**把尺子做到 ±1 cm**

* 当前配对 CI ≈ **±1.3 cm**（12 集，sem 0.0067）。要分辨 1 cm 需要 ~20–25 集；
* 先做**确定性诊断**：同策略同 seed 跑两次，比对 900 步轨迹是否逐位一致（现在残余 ~4.6%）。
  若不一致，找出是 physx 还是 server 采样；一致则以后 12 集就够。

---

## 6. 评测协议（**必须成对、同执行器**）

```bash
TAG=<name> SEEDS="0 1 2 3 4 5 6 7 8 9 10 11" START_FRAME=0 \
  HEAD_MAIN=<对照 ckpt> HEAD_CTRL=<候选 ckpt> \
  SM_CKPT_MAIN=runs/S1_handover/WBC_A24000_wmvel_v6.pt \
  SM_CKPT_CTRL=runs/S1_handover/WBC_A24000_wmvel_v6.pt \
  LABEL_MAIN=<对照名> LABEL_CTRL=<候选名> \
  EXTRA_MAIN="--sample-avg 4" EXTRA_CTRL="--sample-avg 4" \
  PORT_MAIN=18150 PORT_CTRL=18151 GPU_MAIN=0 GPU_CTRL=1 SRV_SEED=0 \
  bash scripts/s3_eval_sweep.sh
```

* 输出：`runs/S3_loop/sweeps/<TAG>/summary.json`（含 `per_seed`、`paired_vs_ctrl`）。
* **只看配对差**（`mean`/`ci95_boot`/`better`），不要拿不同批 episode 的绝对值横向比。
* 12 集 ⇒ CI ≈ ±1.3 cm；**效应 <1 cm 不要下结论**。
* 每集 ≈ 80 s/槽位（900 步），12 集 ≈ 20 min/两条槽位（并行）。
* 一条命令跑完 ≈ 20 min，8 卡机器上可以同时开多组（用不同的 `PORT_MAIN/PORT_CTRL` 与 `GPU_MAIN/GPU_CTRL`）。

---

## 7. 仪器（全部可复用，命令照抄）

| 仪器 | 回答什么 | 命令 |
|---|---|---|
| `probe_wm_gradient_truth.py`（plan/score）+ `isaac_probe_action_rollout.py` | **WM 梯度方向在真物理里可信吗**（符号一致率/相关/幅度比） | `plan → isaac roll → score`（见下） |
| `probe_wm_action_jacobian.py` | ∂q/∂a 的**方向**（对角命中率）与增益 | `--ckpt A B --dump ... --horizon 1` |
| `probe_token_action_gain.py` | **token→action 权限**（策略能不能改变行为） | 见下 |
| `eval_wm_multistep.py` | WM 的 k 步精度（**固定 134 集留出**） | 见下 |
| `compare_wm_eval.py` | 多个评测 json → markdown 对比表 | `--baseline wm_d3.pt runs/S1_handover/eval_wm_*.json` |
| `test_wm_arch.py` | WM 结构自测（mlp/gru，8 项断言） | `python3 scripts/test_wm_arch.py` |

```bash
# 梯度可信度（H=20 为例）
PYTHONPATH=. python3 scripts/probe_wm_gradient_truth.py plan \
  --ckpt runs/S1_handover/wm_a24_k20.pt --dump "runs/wm_real/traj_wmuse2_val/v_11*.pt" \
  --horizon 20 --windows 60 --delta 0.03 --out /tmp/job_h20.pt --device cuda:0
source scripts/isaac_env.sh
$ISAAC_VENV/bin/python -u scripts/isaac_probe_action_rollout.py --job /tmp/job_h20.pt --out /tmp/roll_h20.pt --device cuda:1 --headless
PYTHONPATH=. python3 scripts/probe_wm_gradient_truth.py score --ckpt runs/S1_handover/wm_a24_k20.pt \
  --job /tmp/job_h20.pt --roll /tmp/roll_h20.pt --device cuda:0

# token→action 权限
PYTHONPATH=. python3 scripts/probe_token_action_gain.py \
  --ckpt runs/S1_handover/WBC_A24000_wmvel_v6.pt --dump "runs/wm_real/traj_wmuse2_val/v_11*.pt" \
  --windows 64 --horizon 1 --device cuda:1

# WM 精度（固定留出，与历史可比）
PYTHONPATH=. python3 scripts/eval_wm_multistep.py \
  --ckpt runs/S1_handover/wm_d3.pt runs/S1_handover/wm_a24_k20.pt \
  --dump "runs/wm_real/traj_wmuse2_val/*.pt" --horizon 20 --stride 20 --device cuda:0 \
  --out runs/S1_handover/eval_wm_check.json
```

---

## 8. 踩过的坑（12 条，省你们几个小时）

1. **`pkill -f <script>` 会匹配到自己/正在跑的训练** ⇒ 只按 PID 杀。
2. **长命令里 `VAR="…*.pt"` 会被 shell 展开** ⇒ 用数组或显式 glob 字符串。
3. **`runs/B_<round>` 用 symlink 时 `resolve()` 会把它指回原目录** ⇒ `b_fresh_glob` 匹配 0 条、训练直接 abort。
   用 `B_FRESH_GLOB=<绝对路径> B_FRESH_PIN=1` 显式钉住。
4. **sweep 的两个槽位抢同一个端口**（默认 18100/18101）⇒ 两条 sweep 并发时**互相 kill**，
   结果作废。必须给不同 `PORT_MAIN/PORT_CTRL`。
5. **给 sweep 加 `CUDA_VISIBLE_DEVICES` 会让 ctrl 槽起不来**（它固定用 GPU 索引 0/1），
   要用新加的 `GPU_MAIN/GPU_CTRL`。
6. **`s3_pipeline.sh` 里 `ABL_ROUND=1` 会冻结锚点课程**（`act_weight` / `act_eps` 的自动退场）——单旋钮干净，
   但要知道"课程"在正式轮次里会自动退（`frac_out<0.35 ⇒ ×0.5`）。
7. **`H` 的默认值在 `s3_pipeline.sh:240` 是 8**，`launch_joint_ha_v2.sh` 是 20 ⇒ 不显式给 `H=20` 就跑到 H=8。
8. **`WM_PIN` 必须 `ABL_ROUND=1`**（否则 `die`）。
9. **cuDNN RNN 反向只允许 training 模式**（GRU 相关）。
10. **`bundle.reward` 返回 dict，不是标量**（取 `["total"]`）。
11. **obs 扁平化必须用 `bundle.flatten_history`**（term-major），直接 `flatten` 会让 batch 维错位。
12. **2b 的早停指标（关节 RMSE）与位置/姿态反向** ⇒ 想优化位置就别用它选点。

---

## 9. 文件清单（本轮新增/修改）

| 文件 | 作用 |
|---|---|
| `diff_vla_wbc/wm_arch.py`（新） | `GRUWorldModel` / `build_wm_arch` / `detect_wm_arch` / `load_wm_weights` / `report_wm_arch` |
| `diff_vla_wbc/isaac_joint.py` | WM 按 ckpt 的 arch 构造；`step()` 用 `step_idx` 管隐状态；`save_checkpoint` 写 `wm_arch` |
| `scripts/finetune_wm_velocity_policy.py` | `--arch/--clip-norm/warmup`；`wm.train()`；**A 线顺序自检**；`a-order declared` |
| `scripts/s3_pipeline.sh` | `WM_ARCH/WM_CLIP_NORM`；`ACT_WEIGHT_SCHEDULE`（已有）；`H`（240 行） |
| `scripts/s3_loop.sh` | `CRITIC_STATE_GRAD_SCALE`（新） |
| `scripts/s3_eval_sweep.sh` | `GPU_MAIN/GPU_CTRL`（新）、`SRV_SEED` |
| `scripts/start_vla_server.sh` / `ha_vla_latent64_server.py` | `--seed`（采样可复现）、`--lead-beta`（提前量） |
| `scripts/run_wm_gru_ablation.sh`（新） | WM 消融驱动（`ROUND/ARCH/GPU/K1/K2/A_ORDER/VAL_EVERY`） |
| `scripts/probe_wm_gradient_truth.py` + `isaac_probe_action_rollout.py`（新） | **梯度方向判据**（真物理线搜索） |
| `scripts/probe_wm_action_jacobian.py`、`probe_token_action_gain.py`、`compare_wm_eval.py`、`test_wm_arch.py` | 诊断/汇总/自测 |
| `docs/WM_JOINT_ORDER_BUG_20261031.md` | **394 行主文档**：证据链、全部正负结果、复现命令 |
| `docs/WM_GRU_ABLATION_20261031.md` | GRU 接管审查 + 精度消融 |

**关键 ckpt**：`runs/S1_handover/{wm_d3.pt（旧，勿用）, wm_d5_k20.pt, wm_a24_k20.pt（最好）, wm_d3_k20last.pt}`
**关键策略**：`runs/S3_loop/{d2b, d5jT_joint, d5jR_joint, d5jG_joint}/train_main/vla_final.pt`

---

## 10. 论文口径（避免写错）

1. **可以写**：A 线关节顺序的 bug 被定位并修复，WM 的 k 步误差下降 40–53%；
   **WM 的梯度方向在真物理中被验证 98–100% 可信**（这是少见的、可直接引用的验证方法）。
2. **可以写**：I5 的强化 —— "在本执行器（GT 命令臂 0.108 m）与本数据规模下，
   经想象梯度优化的闭环收益为 **负或不可测**"（5 条独立证据，其中 `grad_scale ×10` **显著变差**）。
3. **不要写**"想象梯度是噪声"——已被真物理证伪。
4. **不要写**"加记忆/加数据能解决长视界"——GRU +70% 更差、数据 3× 只 −6.6%。
5. 闭环所有数字必须带**配对 CI**，并注明 episode 集数（不同批不可横向比）。

---

## 11. 待回答的开放问题（8 卡上可做）

1. H=20 的**信用覆盖**本身（保持影响力不变）有没有收益？ ← P0/P1
2. **跨 batch 的梯度一致性**（同一策略、不同 batch 的 `ĝ` 有多一致）——决定"方向对但学不快"，
   30 min 可测（未做）。
3. WBC 在**部署分布**（VLA token 流）上微调后，闭环能降到多少？ ← P2，最确定的一条
4. 真实回报 critic / on-policy RL 能否突破 0.135？ ← P3
5. Isaac 的**逐位确定性**能否修好（决定评测成本）？ ← P4
