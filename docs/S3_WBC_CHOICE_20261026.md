# S3 联合微调选哪个 WBC + "先训 WM 再联合"是否合适（2026-10-26）

> 输入：`docs/S1_WBC_DECISION_DOSSIER_20261026.md`（5 个候选 + 全量指标）。
> 本文只回答两件事：**选哪个**、**先跑什么 WM 训练**。所有数字都是本机实测（`runs/s3_gates_*/gates.json`、
> `logs/wm_*`），口径 = S3 现在实际用的那套：`--motion-packed runs/isaac_motion/ha_cmd/packed.pt`
> （命令流）、`--wbc-token-window hold`、`--bc-token-space label`、H=16、K=4。

---

## 1. 结论：**选 C2 = `runs/S1_handover/WBC_A20000_obs93.pt`**

| 候选 | S3 五道门禁 | 奖励通道对比度（wbc-space） | burn-in | `grad_ratio` | WM qd_corr（vs **ha_cmd** / vs ha_tokens） |
|---|---|---|---|---|---|
| **C2 `WBC_A20000_obs93.pt`** | **5/5 PASS** | **+0.3941（+29.3%），t=19.10，win=100%** | 0.0901→0.0889，surv 0.9999 | 0.2856 | −0.069 / 11.2 ｜ 0.111（dossier，旧库） |
| C1 `WBC_A20000_pi_WM10500.pt` | **contrast FAIL** | **−0.0134（−2.0%），t=−0.42，win=50%** | 0.0678→0.0524，surv 0.9958 | 0.2035 | 0.119 / 4.24 ｜ **0.70（dossier，旧库）** |
| A@18500（中间产物，作对照） | 5/5 PASS | +0.3901（+29.0%），t=19.63 | 0.0898→0.0887 | 0.2735 | −0.069 / 11.4 |
| A@17250（中间产物，作对照） | 5/5 PASS | +0.3886（+29.0%），t=19.49 | — | 0.2866 | — |

**为什么不是 C1**（dossier 的首选）：C1 的"WM 标定好（0.70/1.00）"是在**旧参考库**
`runs/isaac_motion/ha_tokens/packed.pt`（执行轨迹）上量的；S3 现在的参考/奖励是**命令流库**
（三处口径自洽那条路线）。换成命令流后 C1 的拼接 WM 立刻露馅：

- 奖励通道对比度 **−1.3%（t=−0.42）**：GT token 与零 token 的回报没有区别 ⇒ 奖励反传没有方向；
- burn-in 起点回报 0.0678 → 0.0524（**−23%**，survival 0.9958）：WM 认为"起点变差了"，但这份变差
  对 token 不敏感 —— 相当于给了一个**常值偏移**，梯度里没有"哪个 token 更好"的信息。

**为什么是 C2**：dossier §3.1/§3.3 里它是**部署口径（新表）最好**的策略（NR 0.8673 / 0 摔 / joint RMSE 0.084 /
BONES-500 9/500），`provenance.token_window="hold"`、`robot_spec` 齐全、库 = `bones_seed_plus_ha_cmd_x20.pt`，
与 S3 的 hold/命令流口径**完全对齐**（我的硬门禁零 override）；而且在它上面奖励通道是**强信号**
（t=19.1，64/64 窗口胜出），`grad_ratio=0.286` 落在复核要求的 0.1–0.5。

> 一句话：**C1 是"旧参考口径"下的最优解，C2 是我们这条路线的唯一自洽选择。**
> 如果哪天决定退回执行轨迹参考（`--motion-packed ha_tokens`），C1/C4 才有意义 —— 那等于把
> S0.4/交接文档已经修好的口径又拆回去，不建议。

---

## 2. "先跑 WM 训练，再联合训练"——**合适，但必须是"速度标定"，不是"再共训一轮"**

### 2.1 该做：一次针对速度通道的 WM 微调（20 min）

```bash
cd <repo>
export PYTHONPATH="$PWD:$PWD/Isaac-GR00T:$PWD/third_party/gr00t_deps:$PWD/third_party/gr00t_deps/_noflash"
export SUPERMIMIC_HA_ALIGN=1
python3 scripts/finetune_wm_velocity_policy.py \
  --ckpt runs/S1_handover/WBC_A20000_obs93.pt \
  --out  runs/S1_handover/WBC_A20000_wmvel.pt \
  --steps 800 --vel-lambda 5 --device cuda:0

# 复测：**两个参考库都看**（这一点很关键，见 §2.3）
python3 scripts/eval_wm_velocity.py --ckpt runs/S1_handover/WBC_A20000_wmvel.pt \
  --tokens label --packed runs/isaac_motion/ha_cmd/packed.pt   --samples 16 --device cuda:0
python3 scripts/eval_wm_velocity.py --ckpt runs/S1_handover/WBC_A20000_wmvel.pt \
  --tokens label --packed runs/isaac_motion/ha_tokens/packed.pt --samples 16 --device cuda:0
```

**验收（两条都要）**：
1. `qd_corr` 相对 C2 的 −0.069 **显著上升**、`qd_ratio` 从 11 降到接近 1（ha_cmd 口径）；
   同时 ha_tokens 口径不许变差（防"修好一边、坏了另一边"）；
2. **S3 的 contrast 门禁不许退化**（现在是 +0.394 / t=19.1）：
   ```bash
   python3 scripts/train_joint_v2.py --gate-only --batch-size 4 --gate-batches 8 --lambda-r 0.3 \
     --wbc-ckpt runs/S1_handover/WBC_A20000_wmvel.pt --vlm-ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
     --out runs/s3_gates_C2wmvel
   ```
   过不了就**不要**换 WM —— 奖励通道才是联合微调的实际入口。

### 2.2 不该做

- ❌ **再跑一轮"WM 与策略共训"**：dossier §4.5 已实测，正是这种共训把 `qd_corr` 从 0.737 打到 0.09–0.14；
  而训练侧 `wm_loss` 在所有臂都收敛到 0.0315–0.0329（**看不出好坏**），继续训只会更久、不会更准。
- ❌ 用 `--samples 8` 的单次读数拍板：dossier §8.2 建议 ≥32 采样本；我这边 16 采样已经够看出 11× 的 ratio。

### 2.3 一个必须知道的口径坑：`ha_cmd` 的 `qd` 是**命令速率**，不是实际关节速率

`runs/isaac_motion/ha_cmd/packed.pt` 的 `q/qd` 来自 encoder 的 command 流，而 WM 预测的是**机器人实际**
关节速度。两者在物理上不是同一个量（实测：连"为自己口径训的" A@20000 WM 也只有 qd_corr −0.069 / ratio 11.2）。
⇒ **`eval_wm_velocity` 不能单独当 hold 路线的 WM 门禁**，必须配 contrast 门禁 + 长程 rollout 检查
（`--horizon 32`）一起看。dossier 里 C1 的 0.70/1.00 是"旧库 + 旧 WM"的配对结果，不能直接搬到 hold 路线。

---

## 3. 推荐执行顺序（总耗时 ≈ 1.5–2 h，不含闭环）

| # | 步骤 | 命令/产物 | 通过线 |
|---|---|---|---|
| 1 | （可选，20 min）WM 速度微调 | `finetune_wm_velocity_policy.py` → `WBC_A20000_wmvel.pt` | §2.1 两条 |
| 2 | S3 五道门禁 | `--gate-only`（`runs/s3_gates_C2` 已经是 5/5，若换 WM 需重跑） | 5/5 PASS |
| 3 | λ_r 扫描 | `LAMBDA_R=0 / 0.1 / 0.3 / 1.0` × 1000 步（2 卡并行，λ_r=0 先跑作对照） | `grad_ratio ∈ [0.1,0.5]`；不达标就调 λ_r |
| 4 | mini 闭环挑 ckpt | `scripts/mini_closedloop_v2.sh`（需要能跑 Isaac 的容器） | F 臂 vs B 臂成功率不降 + 一项显著改善 |

**B 臂（对照）** = 官方 `checkpoint-8000` + 同一个 C2 执行器；**F 臂** = S3 训练产物 + 同一个 C2（逐字节相同）。

---

## 4. 追加（同日）：C6（`A20000_wmvel`）不合适，但"按我们口径重标"的版本合适

### 4.1 在 hold/命令流口径上实测三个候选

| ckpt | 五道门禁 | 奖励通道 contrast（wbc-space） | burn-in | `grad_ratio` | WM `qd_corr/ratio`（ha_cmd） | 结论 |
|---|---|---|---|---|---|---|
| C2 `WBC_A20000_obs93.pt` | 5/5 PASS | **+0.394（+29.3%），t=19.1，win 100%** | 0.0901→0.0889（−1.3%）surv 0.9999 | 0.286 | −0.069 / 11.2 | 信号最强，但 **WM 速度通道坏** |
| C1 `WBC_A20000_pi_WM10500.pt` | contrast FAIL | −0.013（−2.0%），t=−0.42 | 0.0678→0.0524（−23%）surv 0.996 | 0.204 | 0.119 / 4.24 | ❌ |
| **C6 `WBC_A20000_wmvel.pt`**（dossier 新首选） | contrast **FAIL** | **−0.003（−0.6%），t=−0.10，win 62%** | 0.0663→0.0514（−22%）surv 0.963 | 0.205 | 0.746（旧域标的） | ❌ 在我们口径上奖励通道已死 |
| **C6h `WBC_A20000_wmvel_cmd.pt`**（我按 ha_cmd 域重标） | **5/5 PASS** | **+0.117（+9.4%），t=7.82，win 91%** | 0.0913→0.0835（**−8.5%**）surv 0.9999 | 0.149 | **0.972 / 0.986** | ✅ **推荐** |

### 4.2 为什么 C6 不行、C6h 行

S1 的 `finetune_wm_velocity_policy.py` 原来**写死旧口径**（`data/npz/ha_tokens` + `runs/isaac_motion/ha_tokens/packed.pt`），
所以 C6 的 WM 是"为执行轨迹参考标的"：**qd_corr 0.746 但动作敏感性归零** ——
换到我们的命令流参考后，GT token 与零 token 的想象回报一模一样（contrast −0.6%，t=−0.10），
绝对想象回报从 1.35 掉到 0.55。**这是"指标被刷高、能力没跟上"的典型。**

我给该脚本加了 `--packed/--npz-dir/--port-dir/--token-space`（默认行为不变），在 **ha_cmd 域**上重标：

| | 标定前（C2） | 标定后（C6h，800 步） |
|---|---|---|
| `qd_corr` / `qd_ratio` | −0.002 / 4.07 | **0.972 / 0.986**（dossier 旧域版只有 0.746） |
| `joint_mse` | 1.3e-3 | **1e-4** |
| policy rollout `state_term` | 0.733 | **0.076**（8.5×） |
| reward | +0.0885 | **+0.0927** |

### 4.3 裁决：**用 C6h（`runs/S1_handover/WBC_A20000_wmvel_cmd.pt`）**

理由（按重要性）：
1. **只有它的 WM 和我们口径一致**（qd_corr 0.97 / ratio 0.99 / joint_mse 1e-4 / state_term 0.076）。
   C2 的 WM 速度通道差 11 倍 → 奖励里的 linvel/angvel 两项是**错的**，梯度大但有偏；
   复核 §0.5 的定位是"奖励只做可执行性精修"，用有偏的大梯度不如用干净的小梯度。
2. **奖励通道仍然成立**：+9.4%、t=7.82、91% 窗口胜出（不是噪声）；`grad_ratio=0.149` 在 0.1–0.5 带内，
   想要更靠近 0.3 就把 λ_r 从 0.3 提到约 0.6。
3. **策略与 C2 完全相同**（只换了 wm）→ 执行器指标不变：新表 NR 0.8673 / 0 摔 / BONES-500 9/500 / joint RMSE 0.084。
4. `provenance.token_window="hold"`、`robot_spec`、`bones_seed_plus_ha_cmd_x20` 库齐全 → 我的门禁零 override。

保留 C2 作为**A/B 对照臂**：如果 λ_r 扫描在 C6h 上没有闭环增益，可以回到 C2 验证
"更大但更有偏的奖励梯度是否反而有效"。两臂只差 `--wbc-ckpt`，其余逐字相同。

> ⚠️ 给 S1 的提醒：`qd_corr` 是**可被标定刷高而动作敏感性归零**的指标（C6 就是证据）。
> WM 门禁必须配 S3 的 contrast 测试；标定时务必用**目标部署口径**的参考库（现在是 `ha_cmd`）。
