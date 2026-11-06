# 固定方案：H=20 + λ_r=0.3 + act_weight=0.5（LOCKED）

> 日期：2026-11-01 05:30
> 状态：**LOCKED**。以后所有联合微调轮次默认走这个配方；任何改动必须先做单旋钮消融，
> 再用 **32 集起**的配对闭环判定。依据：`docs/H20_8GPU_JOINTFT_20261101.md`。

---

## 1. 锁定配方

| 项 | 值 | 说明 |
|---|---|---|
| 想象视界 `H` | **20** | 覆盖 20 步执行窗口（H=8 只覆盖 40%）；H>20 是吃 WM 噪声（k=40 root 12.8 cm） |
| 奖励权重 `LAMBDA_R` | **0.3** | 稳定点；0.6 出现 +26 cm 灾难集（AB 臂） |
| 动作锚点 `ACT_WEIGHT` | **0.5** | P0(1.0)→B(0.5) 把同种子收益 −0.84→−1.35 cm |
| `ACT_EPS` | **0.6** | 与历史一致 |
| 更新次数 | **200 updates × 有效 batch 64** | 与 `d5jT/d5jR/d5jG` 的 800 iter×batch16×accum4 严格等价 |
| 梯度累积 | `ACCUM=1`，`BATCH=64` | 8 卡 DDP 时每卡 8；有效 batch 64 |
| `FM_LR` / `K` | 3e-5 / 4 | 与历史一致 |
| 奖励口径 | root pos std 0.2 / ori std 0.25 / weight 1.0 / split / tail 2.0 | 与 P0/B 一致 |
| critic | imagined GAE，`grad_state=1`，`state_grad_scale=0.1` | 与 P0/B 一致 |
| trust | `self`，eps 0.2，weight 1.0 | 与 P0/B 一致 |
| `STATE_EXTRA` | `rotlocal6` | 与 P0/B 一致 |
| 执行器 WBC | **冻结** `WBC_A24000_wmvel_v6.pt` | 保证与历史读数可比 |
| 起点 | 上一轮最优策略（当前 `runs/S3_loop/h20_8g_b_joint/train_main/vla_final.pt`） | 逐轮递进 |
| 评测 | **32 集起，48 集出论文**；4 条 sweep 占满 8 卡 | 16 集只能分辨 ~0.9 cm |

### 证据（全部 vs d2b，同一执行器）

| 臂 | λ_r | act_w | n | 配对差 | 95%CI | 更好 | 备注 |
|---|---|---|---|---|---|---|---|
| P0 | 0.3 | 1.0 | 32 | −0.95 cm | [−1.55,−0.27] | 28/32 | H=20 覆盖收益 |
| **B（锁定）** | **0.3** | **0.5** | 16 | **−1.36 cm** | **[−2.05,−0.74]** | 14/16 | 后段 −3.52、last50 −5.46 |
| AB | **0.6** | 0.5 | 16 | **+1.08 cm** | [−2.15,+5.27] | 10/16 | `frac_out=0.51`，含 +26 cm 异常集 |

B vs P0 直接配对（同 16 集）：−0.90 cm，CI [−2.32,+0.34]；差中差 −0.52 cm。
**B 明确优于 d2b；B 对 P0 的点估计更好但 16 集尚不可分**（需要 48 集）。

---

## 2. 一轮怎么跑

### 2.1 消融轮（固定 WM，单旋钮验证/复现）

```bash
ROUND=<name> MODE=ablation \
  INIT_VLA=$PWD/runs/S3_loop/h20_8g_b_joint/train_main/vla_final.pt \
  bash scripts/run_locked_h20_round.sh
```

它会：预生成 self 目标 → 8 卡 DDP joint（H=20/λ_r=0.3/act_w=0.5/200 updates）→
4 条 sweep 的 32 集配对评测（自动合并）。

### 2.2 完整轮（on-policy 数据 + WM 刷新）

```bash
ROUND=<name> MODE=full COLLECT_EPISODES=7 \
  INIT_VLA=$PWD/runs/S3_loop/h20_8g_b_joint/train_main/vla_final.pt \
  bash scripts/run_locked_h20_round.sh
```

`MODE=full` 走 `s3_pipeline.sh all` 的 10 个阶段（collect/score/transitions/wm_refresh/
wm_gate/critic/joint/eval/report），joint 参数由本方案锁定；跑完再补 32 集配对评测。

### 2.3 手工等价命令（便于核对）

```bash
OMP_NUM_THREADS=8 ROUND=<name> ABL_ROUND=1 H=20 DDP=1 NPROC=8 \
  WM_PIN=$PWD/runs/S1_handover/wm_d5_k20.pt \
  INIT_VLA=$PWD/runs/S3_loop/h20_8g_b_joint/train_main/vla_final.pt \
  STEPS=200 EVAL_EVERY=100 FM_LR=3e-5 ACT_EPS=0.6 ACT_WEIGHT=0.5 LAMBDA_R=0.3 \
  YAW_HINGE_LAMBDA=0 REWARD_ROOT_POS_STD=0.2 REWARD_ROOT_ORI_STD=0.25 REWARD_ROOT_WEIGHT=1.0 \
  CRITIC_ON_MAIN=1 CRITIC_GRAD_STATE_MAIN=1 CRITIC_STATE_GRAD_SCALE=0.1 TRAIN_ARMS=main \
  RUN_TAG=<name> RUN_LABEL=<name> \
  B_FRESH_GLOB="$PWD/runs/B_d3/*/dump/*.pt" B_FRESH_PIN=1 SKIP_SELF_TARGETS=1 \
  BATCH=64 ACCUM=1 bash scripts/s3_pipeline.sh joint
```

---

## 3. 验收 / 停止判据

**必需（全部满足才算过）：**

1. 对**上一轮最优**：配对差 < 0 且 95%CI 不含 0（至少 32 集）；
2. 对 **d2b 锚点**：保持 ≤ −1.3 cm 的强结论（48 集）；
   > ★ 2026-11-01（用户口径）：**主对照永远是上一轮最优**，不是 d2b/π0。
   > `run_locked_h20_round.sh` 的 `BASELINE_HEAD` 已改成自动取最近一轮
   > `*_joint/train_main/vla_final.pt`（排除本轮），`BASELINE_LABEL` 同步；
   > d2b/π0 只作锚点/ sanity check，不再当主读数。
3. `act/frac_out` 最后 50 步均值 **≤ 0.5**；
4. `L_fm_val` 不比上一轮最优差 10% 以上；
5. 无 NaN、零翻倒；
6. **后段（200–299 步）和最后 50 步必须改善**；若只有早段改善、尾部不变，视为“没修到问题”。

**立即停：** `frac_out > 0.5` 持续、NaN、或出现单集 > +15 cm 的灾难集 ⇒ 该臂作废。

**不要用 n<32 下结论。** 16 集的 CI ≈ ±0.9 cm，只够看方向。

---

## 4. act_weight 继续下降（课程）

**锁定方案里 act_weight 固定 0.5，不做自动退场。** 后续允许按两步走：

1. **固定臂先验证**：`act_w=0.25`（λ_r=0.3 不变），32 集配对。
   若不稳定或不变好 ⇒ 0.5 就是下界，停止。
2. 若 0.25 稳定且更好，才引入课程 `0.5 → 0.25`，并且必须带**双闸门**：
   - 训练内：`frac_out < 0.45`、`L_fm` 不比上一档差 10%、无 NaN；
   - **闭环**：每降一档前用 32 集配对确认没变差（不可省）。

> ⚠️ 为什么不能只看训练内：AB 的训练内 `imag_ret` 是三臂最高（+9.9%）、`imag_delta` 最好，
> 但闭环里 seed 4 是 +26 cm。**锚点退场的风险在闭环尾部，不在训练均值。**
> 现有 pipeline 的自动规则是 `frac_out<0.35 ⇒ ×0.5`，B 的 0.43 不会触发，所以要自定义 schedule。

---

## 5. P2（WBC 执行器）：独立 track，不塞进 joint

**结论：放进流程，但作为独立的“执行器 track”，在锁定策略轮之后做。**

为什么值得：
- B 的闭环 0.1455 m vs GT 命令臂 0.108 m，执行器仍占 ~70% 误差；
- B 证明策略能补偿执行器（后段 −3.52 cm、last50 −5.46 cm），WBC 变好可与策略补偿叠加；
- 交接文档把 P2 列为“最确定”的剩余杠杆。

为什么必须独立：
- WBC 是**所有策略共享**的；换 WBC 后 d2b/B/P0 的历史数字全部要重测，否则不可比；
- joint 阶段**故意冻结 WBC**，混进去会破坏单旋钮归因；
- WM 是在旧 WBC 的数据上刷的；换 WBC 后必须重新刷 WM。

流程：

```
锁定策略轮（B 配方）
  → P2：WBC v7（冻结 VLA；KL 锚 v6；lr 1e-6；200–500 步；每 50 步量闭环）
  → 验收：(B 策略, v7) vs (B 策略, v6) 32 集配对 ≤ −1.0 cm 且 CI 不含 0，零翻倒
  → 用 v7 的数据重新刷 WM
  → 再跑一轮锁定策略（WBC 换成 v7，并重测 d2b/B 基线）
```

在 P2 通过前，锁定方案的 WBC 保持 v6，历史读数不变。

### 5.1 已接进启动器（自动跑）

`scripts/run_locked_h20_round.sh` 现在含 stage 8.5：

- `MODE=full` 默认 `WITH_WBC=1`：`joint → P2(WBC-only, 200 步) → 同一 VLA 的 base/v_new 32 集 A/B → 策略评测`；
- `MODE=ablation` 默认 `WITH_WBC=0`，需要时显式 `WITH_WBC=1`；
- **保守换执行器判据**：A/B 的 `paired.mean < 0` **且** `paired.median < 0` 才换；只赢尾部（中位≈0）不换，避免 AB 式的异常集假赢；
- 训练参数：`FM_LR=0`（VLA 冻结）、`WBC_LR=1e-6`、`STEPS=200`、`ACT_WEIGHT=0.5`、`YAW_HINGE_LAMBDA=0`；
- 产物：`runs/S3_loop/<round>_wbc_joint/train_main/joint_final.pt`、
  `runs/S3_loop/sweeps/<round>_wbc_vs_base_merged.json`。

当前 `h20r2` 是在接这个 stage 之前启动的，所以它的 P2 会在 pipeline 的 eval 之后作为**下一步自动命令**执行；
从下一轮开始，一条 `run_locked_h20_round.sh` 就会自动包含 8.5。

---

## 6. 明确不要做的事

1. 不要再扫 `λ_r=0.4/0.5/0.6`——0.6 已证明是悬崖，0.3 是稳定点。
2. 不要把 H 加到 30/40——k=40 的 WM root 误差 12.8 cm，且执行窗口只有 20 步。
3. 不要用 <32 集的闭环下结论。
4. 不要跨 DDP 度直接比 `L_fm_val` 绝对值（DDP=8 是 64 窗口、DDP=4 是 128 窗口）。
5. 不要把 WBC 训练混进 joint 阶段（见 §5）。
6. 不要在 `frac_out > 0.5` 时继续加奖励权重或降锚点。
7. **不要再用第三池 P（`runs/Bw_pert*`）**：token/动作扰动反事实是**负结果**
   （`docs/HANDOVER_WM_20261031.md`）。2026-11-01 已从 pipeline 删除接线，
   WM 只走 **A+B**；数据已移到 `runs/_deprecated_pert/`，任何 glob 都不再命中。

---

## 7. 产物与工具

| 文件 | 作用 |
|---|---|
| `scripts/run_locked_h20_round.sh` | 锁定方案的启动器（ablation/full） |
| `scripts/eval_h20_8gpu.sh` | 4 条 sweep 并行 + 自动合并 |
| `scripts/merge_eval_sweeps.py` | 多 sweep 配对差合并 + 自助 CI |
| `scripts/compare_train_metrics.py` | 跨臂训练指标对比表 |
| `docs/H20_8GPU_JOINTFT_20261101.md` | 两轮结果与证据 |
| `runs/S3_loop/sweeps/d2b_vs_h20_b_merged.json` | B 的 16 集配对统计 |
