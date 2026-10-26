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
