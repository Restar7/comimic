# 现状与结论（2026-11-07 04:40）—— 供外部评审

> 前一份 `REVIEW_TWO_ROUTES_BPTT_VS_DISTILL_20261107.md`（02:15）写了两条路线的**设计与历史**。
> 本文是它的**更新版**：加入 02:15 之后的所有新结果、当前瓶颈、闭环验收现状、以及工程问题清单。
> 若只读一份，读这份。

---

## 0. 一页结论（TL;DR）

| # | 结论 | 证据强度 |
|---|---|---|
| 1 | **WM 梯度方向在真物理里可用**（H=2 96/96、H=20 63/64 窗口变好） | ✅ 真机实测 |
| 2 | **WM 幅度高估 ≈1.8×**（预测 ΔJ +0.128 vs 真机 +0.059，H=20） | ✅ 真机实测 |
| 3 | **BPTT（端到端报酬反传）在本配置下仍失败**：回报 +1.5% 换来动作偏离 +98% | ✅ 43 步实测 |
| 4 | **把 ĝ 当标签做蒸馏可行**：5,632 条标签 → 留出簇 cos **0.416**（训练内 0.591） | ✅ 离线实测 |
| 5 | **标签量是当前第一瓶颈**：3,072→5,632 条让留出 cos 0.256→0.416，尚未收敛 | ✅ 学习曲线 |
| 6 | **部署侧第一次"量级合格"**：‖Δa‖/η 0.30→0.60、效应/噪声 0.72→1.39 | ✅ 探针实测 |
| 7 | **闭环验收证据不足**：1 个 timeout（奖励恒 0）+ 1 个 sim_error，无视频无逐步数据 | ⚠️ 无效 |

**当前的交付候选**：`runs/distill5k_eta15.pt`（官方 VLA + 我们训的 Δtoken）。
**兜底**：官方 VLA（base）闭环 58.3%（SitSofa 66.7% / VisionNavi 50.0%）。

---

## 1. 部署形态与判据（不变）

```
相机 + state46 ──► 官方 VLA（GR00T Cosmos-Reason2-2B + 1.09B DiT，**权重未动**）
                    └─► 64 维 token
服务端：token' = token + Δtoken_net(token, state64)      ← 唯一增量（1,247,296 参数）
                    └─► SONIC release 解码器（ONNX/CPU）──► 29 维关节目标 ──► PD/Isaac
```

- `state64 = [root_rot6d(6) | dof_pos(29) | dof_vel(29)]`；Δtoken 输入 = `token(64) + state64(64) = 128`
- Δtoken 一次前向 **0.105 ms**（实测），相对闭环 2.6 s/chunk 可忽略；**推理不带 WM**
- 网络：`128 → 1024 → SiLU → 1024 → SiLU → 64`，末层 `normal(std=4e-4)`；`apply_norm_cap` 软上限（训练/部署同一函数）
- 判据：闭环成功率 **≥ base（SitSofa 66.7% / VisionNavi 50.0%，合并 58.3%）**

---

## 2. 已验证的四个事实（全部可复现）

### 2.1 WM 前向
20 步位置误差 **0.82 cm**（exc 起点）；1 步 **0.05 cm / 0.018 rad**（`all_transitions.pt`，口径已验证）。

### 2.2 真物理 ±δ 检验（判"梯度能不能信"）
脚本 `scripts/probe_wm_gradient_truth.py plan|score` + `scripts/isaac_probe_action_rollout.py`
数据 `runs/probe_truth/exc_dump.pt`；日志 `logs/truth_probe.log`、`logs/truth_probe_h20.log`

| 视界 | 窗口 | 符号一致率 | **真物理变好比例** | ΔJ 相关 | J_true 均值 | J_WM 均值 |
|---|---|---|---|---|---|---|
| H=2 | 96 | 1.000 | **1.000** | +0.964 | +0.0076 | +0.0087 |
| H=20 | 64 | 0.984 | **0.984** | +0.092 | **+0.0587** | **+0.1277** |

⇒ 方向可用；幅度中位比 **1.79×**、均值比 **2.17×** ⇒ 部署 η 取 `0.15/1.8 ≈ 0.08`（现已试到 0.15）。

### 2.3 可学性（特征 → ĝ 的余弦，n=3072，8 折 CV）

| 目标 | 特征 | ridge cos | MLP cos |
|---|---|---|---|
| `∂r₀/∂token` | token | 0.194 | 0.175 |
| | state | 0.393 | 0.418 |
| | state+act0 | **0.472** | **0.500** |
| `∂J₂₀/∂token` | token | 0.206 | 0.236 |
| | state+token | 0.232 | 0.334 |
| | state+actH | 0.215 | 0.351 |

残差/均值（L2）：k=1 **2.835**、k=5 1.442、k=20 1.073 ⇒ 理想修正**不是常数**。

### 2.4 成本（batch=1，含一次 autograd）

| 做法 | 耗时 | vs 2.6 s/chunk |
|---|---|---|
| 在线 ĝ，H=2 / 5 / 10 / **20** | 231 / 439 / 987 / **1869 ms** | +9% / +17% / +38% / **+72%** |
| Δtoken 前向（蒸馏版） | **0.105 ms** | ≈0% |

⇒ 在线（推理期算梯度）**只能当离线标签生成器**，不能当交付形态。

---

## 3. 路线 A：BPTT（端到端报酬反传）—— 现状：仍失败

`runs/bptt1`（02:07 起，43 步后人为停止；2 卡 DDP，λ_r=2.8 + `--window-norm` + fullstate）：

| 步数 | `rollout/reward`（在优化的） | `act/rmse`（动作偏离 base） | ‖Δtoken‖ |
|---|---|---|---|
| 1–10 | 0.08337 | 0.0903 | 0.0754 |
| 11–20 | 0.08327 | 0.1336 | 0.1186 |
| 21–30 | 0.08467 | 0.1609 | 0.1520 |
| **31–43** | **0.08466（+1.5%）** | **0.1789（+98%）** | **0.1707** |

**判读**：用翻倍的动作偏离只换来 1.5% 的想象回报 ⇒ 典型的"薅 WM"形态，与 p1e（norm 0.36~0.44、探针 ‖Δa‖=4.83η）同病。

失败机制（已有数学解释）：参数梯度 ≈ `Σ_w g_w ⊗ φ(x_w)`；早期输出对所有窗口近似相同 ⇒ 一步跳到**平均方向**，之后状态相关部分只剩很小的残差梯度；正则项只能强制"有方差"，不能强制"方差有用"（`VAR_WEIGHT` 0.03/0.1/0.3 三档均失败）。

---

## 4. 路线 B：蒸馏 —— 现状：唯一在推进的路线，且已有真实进展

### 4.1 管线（离线化是关键改动）

```
① 标签：从 exc（529,131 行真 Isaac 转移）采窗口
        token = port_tokens（数据集 token，**注意不是 VLA token**，见 §6-风险1）
        ĝ = ∂J₂₀/∂token（一次 autograd，20 步 WM 链）
        → 存成 npz（scripts/... 的 J1_GRAD_DUMP / J1_GRAD_DUMP_TOK）
② 训练：scripts/train_distill_offline.py（**训练阶段完全不碰 WM/VLA/闭环**）
        按 ref0 聚类 → 留出 2 个簇 → 训 1.2M 学生 → 存 head ckpt
③ 验收：探针 probe_ft_landing.py（部署口径）→ 真机 ±δ（可选）→ 闭环
```

### 4.2 天花板实验（`scripts/probe_distill_ceiling.py`，3,072 条标签）

| 输入 | 维度 | ridge cos | MLP 随机切分 | **MLP 跨簇留出（10 折）** |
|---|---|---|---|---|
| **token + state64（部署口径）** | 128 | 0.229 | 0.360 ± 0.006 | **0.256 ± 0.034**（min −0.033 / max 0.348） |
| state71 | 71 | 0.226 | 0.338 | 0.240 |
| token + [rot6d+dof_vel]（老口径） | 99 | 0.197 | 0.282 | 0.219 |
| token 单独 | 64 | 0.201 | 0.298 | 0.207 |

⇒ ① 跨运动泛化掉 ~0.10，但仍有 0.256；② **第 2 刀（补 state64）有效**（0.256 vs 0.219）；③ 有些簇样本极少（簇大小 62~567）⇒ 需分层采样。

### 4.3 离线训练结果（`scripts/train_distill_offline.py`，60 epoch 内收敛）

标签池：`runs/gradtok2`（3,072，20 步 token 空间）+ 今夜新生成（2,560） = **5,632 条**
切分：按 `ref0` 聚 10 簇，**留出 2 个簇**（跨运动口径）

| ckpt | η | train cos | **held-out cos** | 备注 |
|---|---|---|---|---|
| `runs/distill5k.pt` | 0.08 | 0.519 | **0.403** | |
| **`runs/distill5k_eta15.pt`** | 0.15 | 0.591 | **0.416** | 交付候选 |

- 40 epoch 后 held-out 已平（0.401→0.416），train 仍升 ⇒ **瓶颈是标签量/多样性，不是训练时长**；
- `train cos 0.59` 提示：**现有输入的可学上限 ≈0.6**，加标签最多把留出推到 0.5 左右。

### 4.4 部署口径探针（`scripts/probe_ft_landing.py`，16 请求 × 2 噪声，与 base 同状态配对）

| 臂 | `|Δa|` RMS/关节 | `‖Δa‖` | `‖Δa‖/η` | 噪声内 sd | **效应/噪声** | 残差/均值(部署) |
|---|---|---|---|---|---|---|
| base | 0 | 0 | 0 | 0.550 | 0 | —— |
| gt7fix（在线蒸馏，~2.4k 在线标签） | 0.0134 | 0.0450 | 0.30（判"太弱"） | 0.0092 | 0.72 | 0.673 |
| distill5k（η=0.08） | 0.0110 | 0.0518 | 0.35 | 0.0047 | 1.35 | **0.427** |
| **distill5k_eta15** | 0.0199 | **0.0905** | **0.60（判"量级够"）** | 0.0091 | **1.39** | —— |

对照：v13（早期 LoRA 微调）同一测法 残差/均值 = **0.098**（≈常数）。

### 4.5 当前瓶颈与下一步（按证据排序）

| # | 动作 | 依据 | 预期 |
|---|---|---|---|
| 1 | **标签加厚到 1.5~3 万**（按簇分层） | 3k→5.6k 让留出 0.256→0.416，未收敛 | 留出 0.45~0.5 |
| 2 | **补 `state − ref` 显式误差特征** | 现输入的可学上限 ~0.6；`ref` 单独喂 cos 仅 0.18，但"差量"没试过 | 可能再 +0.05~0.1 |
| 3 | η 扫档（0.08 / 0.15 / 0.20） | 0.60η 判定"量级够"，但闭环是否过冲未知 | 闭环最优 η |
| 4 | 标签视界 k：20（对齐）vs 5（更可学） | §2.3 | 需 A/B |

---

## 5. 闭环验收现状（**证据不足，不能下结论**）

`runs/ab_d15k/`（04:09 起，HEAD=`runs/distill5k_eta15.pt`，SitSofa + VisionNavi × seed 0/1，1000 步）

| episode | 结果 | 关键字段 |
|---|---|---|
| SitSofa seed0 | **timeout** | 1000/1000 步、440 s、**`final_reward=0.0`、`max_reward=0.0`** |
| SitSofa seed1 | **sim_error**（第 204 步） | Isaac 日志：`urllib.error.URLError: Connection refused` ⇒ **服务端先被杀了**（脚本编排问题） |
| VisionNavi ×2 | 未跑（被我手动停止） | —— |

- **`max_reward = 0.0`** ⇒ 整段 1000 步任务奖励从未非零 ⇒ 机器人**从未接近目标**；但只有 1 个样本，且 base 在 SitSofa 也有 2/6 失败 ⇒ 不构成结论；
- **本轮没有视频、没有逐步 `steps.jsonl`**（`RECORD_VIDEO=0` 且未设 `SONIC_DEBUG_DUMP`）⇒ 连"往哪走/是否站桩"都无法判读。

**复跑前必须补**：`RECORD_VIDEO=1`、`SONIC_DEBUG_DUMP=<dir>`、服务端生命周期与各 seed 解耦（避免 Connection refused）。

---

## 6. 风险与未解问题（评审重点）

1. **标签点 vs 部署点不一致（最大风险）**
   所有标签都在 **`port_tokens`（数据集 token）** 上算，而部署时学生吃到的是 **VLA 输出 token**。
   两者分布相近（VLA 的 token MSE≈0.0046）但未验证；若差异大，则"留出 cos 0.416"不能直接外推到部署。

2. **任务分布未覆盖**
   标签来自运动库（走/跟踪），部署任务是 SitSofa / VisionNavi / OpenDoor / DoubleDesk 的参考轨迹。
   跨任务泛化**从未测过**（`runs/gradtok2` 的 `frames` 与运行时参考库还对不上，见下条）。

3. **闭环 dump 的关节口径未定死**
   `runs/closedloop_r3/*/dump/steps.jsonl` 的 `q` 与运动库 ref 一步预测误差 **0.415 rad**（加 `ha_v31→standard` 置换更差 1.48 rad）。
   ⇒ 任何用闭环录制数据评估 WM/漂移的历史结论**一律作废**（已撤回一组"漂移分解"数据）。

4. **运行时参考库来源未确认**
   `isaac_replay_online.sh` 的 `LIB` 默认 `runs/isaac_motion/ha_cmd/packed.pt`，但用 `frame` 索引两种库都 0% 命中（最近邻 ~1 rad）⇒ 未查清。

5. **可学上限**：`train cos 0.591` ⇒ 现有输入下天花板 ~0.6；想再上必须换输入（§4.5-2）。

---

## 7. 今天修掉的工程问题（都记档，避免重复踩）

| # | 问题 | 修法 |
|---|---|---|
| 1 | 蒸馏标签算在**错误 token**（`port_tokens` vs VLA 输出）上 | `_dt_base_tok`，`grad_cos` 从 0.05 恢复到 0.09~0.14 |
| 2 | 探针/服务端把 Δtoken 输入宽度**硬编码 99** | 从 `state["net.0.weight"].shape[1] - 64` 反推；服务端特征按 `feat_dim` 二选一 |
| 3 | 服务端 `NameError: _d` | 变量名改回 `d` |
| 4 | DDP 下 dump 文件同名互相覆盖 | 文件名加 `RUN` 标签 + rank |
| 5 | `window_norm` 缺实现 | 新增 `--window-norm`（逐窗口归一化） |
| 6 | 缺"快速标签生成"开关 | 新增 `J1_GRAD_DUMP_N`（只标子批，成本按 batch 线性下降） |
| 7 | 闭环 A/B 环境坑 | 必须用 `scripts/r3_sr_cell.sh`（py312 shims），且显式导出 sonic 栈环境变量（缺了会静默回落 SuperMimic provider） |

**未修**：标签生成仍跑在训练 harness 里（每步额外一次 20 步带梯度 rollout），实测 **3.5 min/步 @ batch 256**；`J1_GRAD_DUMP_N` 已加但尚未实测提速倍数。

---

## 8. 复现命令

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
# ① 天花板（3,072 缓存标签，含跨簇留出）
python scripts/probe_distill_ceiling.py --dir runs/gradtok2 --target g_tokH --split cluster
# ② 离线蒸馏训练（5,632 标签，η=0.15，留出 2 簇）
python scripts/train_distill_offline.py --dir runs/labels_all --out runs/distill5k_eta15.pt \
       --epochs 40 --holdout-clusters 2 --eta 0.15
# ③ 部署口径探针
python scripts/probe_ft_landing.py --arms base= e15=runs/distill5k_eta15.pt --n-noise 2 --max-req 16
# ④ 标签生成（提速版：只标 128 个/步）
J1_GRAD_DUMP=<dir> J1_GRAD_DUMP_TOK=1 J1_GRAD_DUMP_N=128 ... RUN=gen15k GPU=1 bash scripts/launch_joint_ha_v2.sh
# ⑤ 闭环（务必带视频 + 逐步 dump）
RECORD_VIDEO=1 SONIC_DEBUG_DUMP=<dir> TASK=SitSofa HEAD=runs/distill5k_eta15.pt ... bash scripts/r3_sr_cell.sh
```

**产物索引**：`runs/distill5k_eta15.pt`（候选）、`runs/labels_all`（5,632 标签）、`runs/gradtok2`（3,072）、
`runs/ab_d15k`（闭环，含失败原因）、`docs/REVIEW_TWO_ROUTES_BPTT_VS_DISTILL_20261107.md`（02:15 版设计文档）、
`docs/HANDOVER_ONLINE_GRAD_DISTILL_20261107.md`（交接）。

---

## 9. 希望评审回答的问题

1. **BPTT 还有没有标准救法？**（在"共享参数 + 跨窗口求和 + 输入可解释性仅 0.3~0.6"的前提下；per-window 梯度均衡 / 独立 head / 蒸馏初值 + BPTT 微调，哪种值得试？）
2. **标签点不一致（§6-1）怎么低成本消除？** 是否应该让标签也在 VLA token 上算（需要离线跑 VLA 前向，我们没有 exc 的图像，需要想别的办法）？
3. **加标签 vs 加特征，哪个先做？** 现证据是"标签未收敛（0.256→0.416）"和"输入上限 ~0.6"两条并存。
4. **验收的统计功效**：base 58.3%、单 episode 1000 步、每臂 2 任务 × 1~2 seed。要多少 episode 才能看出 +10 个点？有没有更便宜的中间指标（如真机 ±δ 的"变好比例"）可以先筛？
5. **是否有我们没想到的第三条路**：例如直接用 ĝ 训练一个"只在低频（每 N 个 chunk）运行"的在线修正，把 1.87 s 成本摊薄到 +10%？（需要服务端拿到完整状态与参考帧，工程上未完成）
