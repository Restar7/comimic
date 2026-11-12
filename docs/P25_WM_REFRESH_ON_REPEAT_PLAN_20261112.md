# P25 计划：用**目标分布闭环数据**刷新 WM → 重训 critic → 联合微调 → 配对评测

> 生成：2026-11-12 05:40 ｜ 执行：C 线 ｜ 状态：**待执行**
> 一句话：把 seed0–2 repeat 协议新采的闭环数据混进 WM refresh（refresh #3），然后按老流程
> （critic 预训 → 联合微调 → val 选优 → 闭环配对）走一遍，**与 p21 构成严格的单变量对照**。

---

## 0. 30 秒速览

| 项 | 值 |
|---|---|
| 与 p21 的差异 | **只有 WM**（p21 = 旧 `wm_cl_candidate`；本单 = 在目标分布上再刷一次的 v3） |
| 起点 LoRA | **c_base `critic10/lora300_ddp/checkpoint_lora_best.pt`**（sha `6942eab9`） |
| 步数 / 配方 | 300 步、λ_lim 0.0049、β 0.975、critic v5、其余逐项同 p21 |
| 数据 | seed0–2 repeat 闭环（新）+ 已有 `collect_tok / ab2way_* / s7_12_container*` |
| 成本 | 数据 10 min + refresh 10 min + critic 5 min + 微调 35 min + 评测 2 h ≈ **半天** |
| 主判据 | ① val 相对 dJ（`dJ/J_base`）> **0.6%**（p21 是 0.38%）；② 闭环配对不劣于 base |

---

## 1. 为什么做这一单（以及为什么先验不高）

今晚已经证明的（探针与训练日志都在 `runs/dual_actor_wm6/c_lora/p2{2,3,4}/` 与 `/tmp/probe_*.log`）：

| 结论 | 证据 |
|---|---|
| 在旧 WM 上改目标/改梯度全部无效 | p22 噪声 0.25%、p23 锐化 0.24–0.32%、p24 lookahead **−0.28%**，p19/p20/p21 中性 |
| token 模仿层面已饱和 | 标签 token 与 VLA token 的跟踪误差 **1.049 vs 1.040 cm** |
| 逐窗口补偿在 WM 里有效但**不可学** | oracle −24% 误差；但最优系数 out-of-sample R² **全为负**（state/token/差分特征都试过） |
| 唯一可学的部分 ≈ 全局共享分量 | 全局偏移 +9.6% J / −9.8% 误差（20% cap） |

**本单唯一没试过的轴：WM 没见过目标分布（repeat 派生场景）。** 如果 critic 的排序在目标场景上更准，
联合微调**有可能**拿到那 5–10% 的共享分量。**不要期待 24%**——信息墙和 WM 精度无关。

---

## 2. 步骤 1：构建闭环转移对（新数据 + 老数据）

1. 复制 `scripts/c_wm6/build_cl_pairs.py` → `scripts/c_wm6/build_cl_pairs_v3.py`，改两处常量：

```python
OUT  = WT / 'runs/dual_actor_wm6/c_lora/p25/refresh3'
RUNS = [
    # 新：目标分布的闭环（seed0-2 repeat 协议）
    'p21_seed012_r0_5', 'p21_seed012_r6_11', 'p21_seed012_r12_17',
    'p21_recheck_r0_5',
    # 旧：录制/早期闭环，防止过拟合到单一任务
    'collect_tok', 'ab2way_s35_50', 's7_12_container1', 's7_12_container2', 'ab2way_0_7',
]
```

2. 跑（**不需要 GPU**，约 5–10 min）：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh
python3 scripts/c_wm6/build_cl_pairs_v3.py
```

3. 验收：`p25/refresh3/cl_pairs.pt` 存在，且 `cl_pairs_meta.json` 里

- `train + val` 条数 ≥ **35 万**（新数据约 10 万 + 老数据 40 万，去重后）；
- 闭环 ep 数 ≥ **250**（新 200+，老 184 里的训练部分）；
- 每个 episode 的 `uid % 5 == 0` 进 val（防泄漏，脚本已内置）。

---

## 3. 步骤 2：WM refresh #3

复制 `scripts/c_wm6/p21_refresh2_cl.py` → `p25_refresh3_cl.py`，改三处常量：

```python
INIT = WT / 'runs/dual_actor_wm6/c_lora/p19/refresh2/wm_cl_candidate.pt'   # 从 refresh2 出发
OUT  = WT / 'runs/dual_actor_wm6/c_lora/p25/refresh3'                       # 新目录
DEV  = 'cuda:0'
```

跑（loss 与 refresh#2 **完全相同**：`ssm_wm_loss + 2.0 × increment_loss`，这是 p21 基线高的那个配方）：

```bash
python3 scripts/c_wm6/p25_refresh3_cl.py --steps 3000 --lr 1e-5 --cl-frac 0.5
```

**通过的判据（沿用 refresh#2，两个都要过）**：

| 指标 | 阈值 |
|---|---|
| 闭环 val 零动作误差比 | **≥ 1.15** |
| 闭环 val 打乱动作比 | **≥ 1.15** |
| 闭环 val 单步误差 | 比 refresh2（0.2515）**不退化超过 20%** |

产出：`p25/refresh3/wm_cl_v3.pt`，记录 **sha256 前 8 位**（后面 `--strict-wm-hash` 要用）。

```bash
sha256sum runs/dual_actor_wm6/c_lora/p25/refresh3/wm_cl_v3.pt | cut -c1-8
```

---

## 4. 步骤 3：用新 WM 重训 v5 critic

从 `run_p23_std_critic.sh` 复制成 `run_p25.sh`，**删掉 4 个 std 参数**（本单不改奖励），并改：

```bash
OUT="$C/p25/ft300"
--wm-checkpoint "$C/p25/refresh3/wm_cl_v3.pt"
--strict-wm-hash <上面算出的 8 位>
--critic-pretrain-out "$OUT/critic_pretrain.pt"
```

只跑阶段 1（critic 预训）：

```bash
python3 -m torch.distributed.run --standalone --nproc_per_node=2 --master_port=29593 \
  scripts/train_sonic_lora16_ft8000_bptt.py "${COMMON[@]}" --critic-enabled 1 \
  --critic-pretrain-epochs 20 --critic-pretrain-batch 256 \
  --critic-pretrain-out "$OUT/critic_pretrain.pt"
```

验收：`v_ev` ≥ **0.30**（p23 那次是 0.930；低于 0.3 说明 critic 没学好，后面 β 会带偏）。

---

## 5. 步骤 4：联合微调（p25 = c_base 起步）

其余逐项照 p21，**只把 WM 和 critic 换掉**：

```bash
--init-lora  "$C/critic10/lora300_ddp/checkpoint_lora_best.pt"    # c_base，不是 p21
--wm-checkpoint "$C/p25/refresh3/wm_cl_v3.pt"
--strict-wm-hash <sha8>
--critic-warm-from "$OUT/critic_pretrain.pt"
--critic-enabled 1
--updates 300 --lr-total-updates 300 --warmup 20 --lr 1e-4 --min-lr 1e-6
--batch 16 --accum 1 --save-every 25 --eval-every 50
--lambda-lim 0.00490 --beta 0.975 --beta-warmup-updates 100 --beta-gate none
```

**为什么起点是 c_base 而不是 p21**：p21 已经是「c_base + 旧 WM + 同配方」，所以
`c_base + 新 WM` 与它构成**严格单变量对照**。先看这一条；有效再谈叠加。

---

## 6. 步骤 5：选优

- 自动按 val `dJ` 选 best（`checkpoint_lora_best.pt`）；
- 记录三个数：`val dJ`、`test dJ`、`J_base`（**必须报相对量 `dJ/J_base`**，因为不同 WM 的 J 尺度不同，绝对 dJ 不可比）。

**判据**：`dJ/J_base` **> 0.6%** 才算有信号（p21 = 0.38%，p22 = 0.25%，p23 = 0.24–0.32%）。

---

## 7. 步骤 6：闭环配对评测

用和 base 交接单**完全相同的 30 个格**（OpenDoor + Boxing × seed0/1/2 × repeat0–4），
这样可以直接三方对照：**p25 vs p21 vs base**。

```bash
C2_SERVERS=c C2_PORT_C=18080 C2_GPU_C=0 C2_ISAAC_GPU=0 C2_SERVER_SEED=20261109 \
python3 scripts/c_wm6/c2_driver.py --tasks OpenDoor --seeds 0 1 2 --repeat-start 0 --repeats 5 \
  --arms c --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/p25_repeat/card0 \
  --lora-ckpt runs/dual_actor_wm6/c_lora/p25/ft300/checkpoint_lora_best.pt --lora-tag p25
# card1 换 Boxing / 端口 18081 / GPU 1
```

汇总脚本直接复用交接单 §8 的那段（把 `base_repeat_s012_r0_4` 换成 `p25_repeat`）。

---

## 8. 预注册判据（跑之前定死）

| 关卡 | 阈值 | 不通过怎么办 |
|---|---|---|
| G1 WM refresh | 零动作比、打乱比 ≥1.15 且误差不退化 >20% | 调 `--cl-frac` 或步数，最多重试 1 次；仍不过就**停止本单** |
| G2 critic | `v_ev ≥ 0.30` | 重训一次；仍不过就关 critic 跑（并标注） |
| G3 训练 | `dJ/J_base > 0.6%` | **停止**，本线收尾 |
| G4 闭环 | 30 格配对：p25 的净胜 ≥ 6 格（即 McNemar p<0.10） | 结论写“与 p21/base 无差异” |

**禁止**：只重跑失败格、取 best-of、事后换判据。

---

## 9. 风险与止损

1. **最可能的结局仍是中性**：信息墙（逐窗口补偿不可预测）不会因为 WM 更准而消失。所以本单的价值是**排除最后一个变量**，不是指望翻盘。
2. **新数据可能把 WM 带偏**：repeat 闭环里失败集占多数（失败=卡住/超时），动力学分布偏向"失败行为"。如果 G1 的零动作比下降，说明 WM 学到了坏动力学 → 立即停下，混回旧数据重刷。
3. **30 格配对的判别力只有 ≥30pp**：所以 G4 即使"净胜 6 格"也只是弱信号，必须在报告里写明。
4. **PPBox / DoubleDesk**：用已有采集数据，不新增；这两个任务的闭环成功率接近 0，本单不指望它们。

---

## 10. 时间线（半天）

| 时间 | 步骤 | 产出 |
|---|---|---|
| +0:00–0:15 | 构建转移对 | `cl_pairs.pt` + meta |
| +0:15–0:30 | WM refresh #3 | `wm_cl_v3.pt` + sha |
| +0:30–0:40 | critic 预训 | `critic_pretrain.pt`（v_ev） |
| +0:40–1:20 | 联合微调 300 步 | `checkpoint_lora_best.pt` + val/test |
| +1:20–1:30 | 判定 G3 | 继续 / 停止 |
| +1:30–3:30 | 30 格配对评测 | `p25_repeat/` + 配对表 |

---

## 11. 交付物清单

- [ ] `p25/refresh3/{cl_pairs.pt, cl_pairs_meta.json, wm_cl_v3.pt}` + **sha256**
- [ ] G1 三项指标（零动作比 / 打乱比 / 单步误差）对照 refresh2
- [ ] `p25/ft300/{critic_pretrain.pt, checkpoint_lora_best.pt, checkpoint_lora_best.json}`
- [ ] `val / test` 的 `dJ` 与 **`dJ/J_base`**
- [ ] `p25_repeat/` 30 集 + 与 p21 / base 的三方配对表（含 McNemar）
- [ ] 一句话结论：**新 WM 是否让联合微调超过 p21 的 0.38%**

---

## 12. 如果这单也中性（大概率）

那就把结论定死，写进总报告：

> 在 HumanoidArena-lite repeat 协议下，C 线所有变体（C、p19–p25）相对官方 FT8000 base **均无统计显著提升**；
> 机制上，VLA token 在模仿层面已与标签 token 等价，跟踪误差地板由冻结 WBC 决定，
> 而逐窗口的最优补偿不可从观测预测（out-of-sample R² < 0）——**amortized 梯度训练在这条链上没有可用信号**。

之后的资源应转向：**换 WBC / 参考动作重定向**（改传动比），或**直接对成功率建模**（value 模型 / bandit），
而不是继续在「VLA token × 跟踪奖励」这个组合上加力。
