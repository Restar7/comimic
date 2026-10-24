# FM 梯度反传消融（`ablation_fm_grad/`）

> **论文用报告：`ablation_fm_grad/REPORT.md`**（英文、带 Related Work / 表格 / LaTeX / 引用）。
>
> 目标：把"execution loss → flow-matching 动作头"的几种梯度估计方式放在**同一套模型、同一套 loss、同一份随机种子**下对比，回答"现在该用哪种 FM 反传"。
> 当前默认实现（`diff_vla_wbc/gr00t_fm.py::sample_fm_from_features`）就是 **pathwise / unrolled Euler + autograd**；本目录把其他方案补齐成可切换后端。

---

## 0. 客观结论：现在能做什么、不能做什么

| | 结论 | 依据 |
|---|---|---|
| ✅ **现在就能做** | 梯度正确性（与 pathwise 的 cosine / 相对误差）、显存/时间、短程学习曲线、K 的 scaling | 本目录 `toy` + `tiny` 两种模式已跑通并给出结果 |
| ⚠️ **需要先补一步** | 真实 **3B GR00T + 训练好的 SuperMimic WBC + 真实任务损失** 的端到端对比 | 缺三样：① token-conditioned SuperMimic checkpoint（20k 跑在 it1450 被停，没有可用 ckpt）；② 世界模型是 robot-only、无对象状态，任务损失目前只能是 tracking/safety；③ 4090 48GB 上 full GR00T + H=8 的显存未验证（需要 H200 或 checkpointing） |
| ❌ **现在不能下结论** | "Q-VGM 更好/更差"的论文级结论 | 本目录的 `qvgm_lite` 是**机制复现**（非官方代码），critic 用 oracle/回归替身；只能验证"局部值梯度方向对不对"，不能替代官方 IQL/TD chunk critic |

**一句话**：方法选型实验现在可以做；论文里"我们比较了 Q-VGM / adjoint / checkpointing"这句话，要等真实 checkpoint + 真实任务损失那一步跑完才能写。

---

## 1. 五种后端

| 后端 | 原理 | 梯度 | 激活显存 | 计算 | 适用 |
|---|---|---|---|---|---|
| `pathwise` | 全图 unrolled Euler（当前原型） | 精确 BPTT | O(K) | 1× | 默认，先验证 4–8 步 |
| `checkpointed` | 每步 `torch.utils.checkpoint` | 精确（重算） | O(1) 存储 + 重算 | ~2× | 长 K / 大 DiT，省显存 |
| `truncated` | 前 K−tail 步 `no_grad` | 有偏（只到最近 tail 步） | O(tail) | 1× | 廉价 warm-start / 对照 |
| `adjoint` | 离散伴随：前向 no_grad 存 state，反向逐步 vjp | 与 BPTT 数学等价（浮点误差内） | O(1) 步图 + K 个 state | ~1.5–2× | 长链、显存吃紧 |
| `qvgm_lite` | 晚步 clean-action 估计 + critic 值梯度，只回传单步 velocity | 有偏（局部近似） | O(1) | 1× | 长 K 且 critic 可靠时 |

> 说明：VITA 的 "flow latent decoding" 本质也是**把 reconstruction loss 穿过 ODE solver**，梯度机制与 `pathwise` 相同（差异在架构/loss 设计），因此不作为独立后端；Q-VGM 的官方实现没有公开代码，`qvgm_lite` 按论文的 local approximation 复现（见 `papers/NOTES.md`）。

---

## 2. 实测结果（本机 4090，2026-10-24）

复现命令：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
/opt/miniconda3/envs/isaaclab/bin/python -m ablation_fm_grad.run_ablation \
  --mode toy --steps 2 4 8 --trials 3 --train-steps 30 --device cuda --qvgm-critic oracle
/opt/miniconda3/envs/isaaclab/bin/python -m ablation_fm_grad.run_ablation \
  --mode tiny --steps 4 --trials 1 --train-steps 20 --lr 1e-2 --device cuda --qvgm-critic oracle
```

### 2.1 toy（MLP velocity + 二次执行损失，K=2/4/8）

见 `results/toy_2-4-8.md`。要点：

```text
pathwise     cos=+1.0000  rel=0.00e+00  train 70.1 → 2.12
checkpointed cos=+1.0000  rel=0.00e+00  train 70.1 → 2.12   (时间 ~2×)
adjoint      cos=+1.0000  rel=0.00e+00  train 70.1 → 2.12   (时间 ~1.5–2×)
truncated    cos=+0.9984  rel=8.75e-01  train 70.1 → 2.16
qvgm_lite    cos=+0.9998  rel=5.00e-01  train 70.1 → 2.10   (oracle critic)
```

### 2.2 tiny（真实 GR00T N1.7 action head 类，tiny 维度，K=4）

见 `results/tiny_4.md`：

```text
pathwise     724.33 → 15.57    cos=1.0000  rel=0
checkpointed 724.33 → 15.57    cos=1.0000  rel=0     (时间 ~1.8×)
adjoint      724.33 → 15.57    cos=1.0000  rel=0     (时间 ~1.4×)
truncated    724.33 → 31.93    cos=0.9975  rel=0.75
qvgm_lite    724.33 → 100.12   cos=0.9973  rel=0.50
```

### 2.3 真实 1.09B GR00T action head（4090 实测）

直接加载官方 checkpoint 的 **`action_head.*` 权重**（1,091,722,240 参数，bf16，32 层 DiT，
`max_action_dim=132, H=40`），不加载 3B VLM，构造 dummy backbone/state 特征，K=4、batch=1：

```text
pathwise      loss=1.0010  peak=5811 MB  time=0.280 s
checkpointed  loss=1.0010  peak=8259 MB  time=0.404 s   (step-level torch.utils.checkpoint)
grad max|diff|=0.0   token max|diff|=0.0
```

**结论（重要）**：

- 真实 action head 的 K=4/batch=1 **pathwise 只要 ~5.8 GB**，4090 48 GB 完全放得下；
- **step-level checkpointing 在这个模型上反而涨到 8.3 GB（+42%）、慢 1.44×**——
  原因是 checkpoint 只包住整段 DiT 调用，重算时同时存在前向重算激活与反向张量，
  并没有按 block 粒度切；**不要盲目用我们这版 `checkpoint_steps` 去省显存**；
- 要真正省显存需要 **per-block checkpointing**（在 DiT 的 transformer block 级别包），
  这是后续工作，不是当前主线。

### 2.4 真实规模档位（`--mode scale`，隔离进程测显存）

为逼近真实 action head 的显存行为，用**真实 GR00T action-head 类**放大到
`H=40 / hidden=512 / 8 layers / batch=4`，每个后端单独进程跑（避免 CUDA 分配器串味）：

| 后端 | 峰值显存 (MB) | 单步时间 (s) | 相对 pathwise 显存 | 相对时间 | 梯度 cos / rel |
|---|---|---|---|---|---|
| pathwise | 575.6 | 0.062 | 1.00× | 1.00× | 1.0000 / 0 |
| checkpointed | 448.8 | 0.106 | **0.78×** | 1.71× | 1.0000 / 0 |
| adjoint | 633.6 | 0.089 | 1.10× | 1.44× | 1.0000 / 0 |
| truncated | 390.0 | 0.029 | 0.68×（有偏） | 0.47× | 0.8869 / 0.76 |

（`results/scale_4_large.md`；K=4，单 seed。）

**关键观察**：在这个规模下 checkpointed 只省 ~22% 显存、慢 1.7×；
adjoint **没有省显存**（K 个 forward state + 每步 vjp 的临时张量），但梯度精确。
真实 3B action head 的每步激活会大一个量级，checkpoint/adjoint 的显存优势才会明显；
在 141 GB 的 H200 上这个优势不重要。

### 2.4 主线选择（结论）

| 硬件/场景 | 主线选择 | 理由 |
|---|---|---|
| **4×H200（141 GB，正式训练）** | **`pathwise`** | 显存不是瓶颈；精确、最快（无重算）；K=4–8 的图完全可以放下 |
| **4090 48 GB（微调，当前实际场景）** | **`pathwise`** | **实测 1.09B 真实 action head：K=4/batch=1 仅 5.8 GB**；加上 frozen VLM(~6 GB)、梯度/Adam(~7 GB)、SuperMimic rollout，仍 < 48 GB |
| 4090 + 更大 batch/K 导致 OOM | **per-block checkpointing**（待实现），或降 batch/K | 实测 step-level `checkpoint_steps` 反而 8.3 GB（+42%）、慢 1.44×，**当前这版不能用来省显存** |
| K 很大（16/32）且 per-block 仍 OOM | `adjoint` | 数学等价；但当前实现不省显存、慢 ~1.4×，仅作最后手段 |
| 只要便宜 warm-start | `truncated` | 有偏（rel 0.76），只做 baseline，不能当主线 |
| 研究分支（论文对比） | `qvgm_lite` | 需要真实 IQL/TD chunk critic 才能公平比较；当前只做机制验证 |

**因此：主线 = `pathwise`（就是当前 `sample_model_with_grad` 的实现）。4090 上它也装得下（实测 5.8 GB/1.09B head/K4/batch1）；`checkpointed` 当前实现不省显存，只有改成 per-block 后才作为 fallback；`adjoint` 仅作长 K 的最后手段；`truncated`/`qvgm_lite` 只进消融表。**

### 2.5 解读（诚实版）

1. **`pathwise` / `checkpointed` / `adjoint` 三个精确后端梯度完全一致**（cos=1，rel=0），说明实现正确、可作为互相的 sanity check；
2. **`checkpointed` 用 ~2× 时间换 O(1) 激活**，在 toy/tiny 上显存差异被 CUDA context（~18 MB）淹没，**必须用真实 DiT/H=8 才能看出收益**；
3. **`adjoint` 与 BPTT 等价但慢 1.4–2×**（K 次 vjp），适合"长 K + 显存紧"而非"更快"；
4. **`truncated` 方向对、幅值偏**（rel 0.5–0.88），学习略慢——可作为 cheap baseline，不能当主方法；
5. **`qvgm_lite` 方向几乎对（cos 0.997–0.9998）、幅值差 0.5**；在 tiny/lr=1e-2 下反而变差（724→100），说明**局部近似 + 无校准 critic 对步长敏感**——这正是 Q-VGM 论文强调 critic 训练的原因，别用这个结果去否定 Q-VGM。

> 注意：toy 的"loss"和 tiny 的"loss"都是替代任务，不是 MuJoCo 任务成功率；这些数字只用于**选梯度后端**，不能作为论文性能结果。

---

## 3. 代码结构

```text
ablation_fm_grad/
  backends.py          # pathwise / checkpointed / truncated / adjoint / qvgm_lite
  harness.py           # toy + tiny(GR00T action head) 两种 setup、critic、指标、学习循环
  run_ablation.py      # CLI：跑 K 列表、所有后端，输出 md/json
  results/             # 实测结果（toy_2-4-8.*, tiny_4.*）
  papers/NOTES.md      # Q-VGM / VITA / adjoint / checkpointing 论文要点
  tests/test_backends.py
```

关键接口（都在 `harness.py`）：

```python
setup = make_toy_setup(K, device)          # 或 make_tiny_gr00t_setup(K, device)
res = make_backend("pathwise").run(setup.field, setup.a0, K, setup.task, critic=None)
# res.grads: {param_name: grad}，可与 pathwise 比较 cosine / rel err
```

`make_tiny_gr00t_setup` 直接复用 `diff_vla_wbc.gr00t_tiny` 的**真实 action head 类**（`action_encoder → DiT → action_decoder`），只是把维度缩小；因此 tiny 结果比 toy 更接近真实图，但仍不是 3B 权重。

---

## 4. 接到真实 GR00T + SuperMimic（下一步，待跑）

### 4.1 需要的前置

1. **一个可用的 token-conditioned SuperMimic checkpoint**
   ```bash
   cd third_party/SuperMimic-Reproduced
   .venv/bin/python -m supermimic.scripts.train --run-name token_5k \
     --data.library data/library/bones_seed_compact.pt --policy.token-dim 64 \
     --env.num-envs 4096 --policy-trainer.batch-size 2048 \
     --schedule.total-iterations 5000 --schedule.checkpoint-interval 1000
   ```
   （约 35–40 分钟；产出 `ckpt_latest.pt`）
2. **world model 的 71 维 state / 680 goal 切片**（`supermimic.eval.benchmark.load_policy_from_checkpoint` 已能加载）；
3. 真实任务损失：当前先接 **tracking + safety**（`imagine_with_tokens` + `compute_joint_loss`），对象/抓握损失需要 `F_obj` 扩展（未实现）。

### 4.2 替换 harness 的 task

把 `QuadraticTask` 换成：

```python
from diff_vla_wbc.rollout import imagine_with_tokens, compute_joint_loss
from supermimic.eval.benchmark import load_policy_from_checkpoint

bundle = ...  # SuperMimicBundle(policy, world_model, assets, ...)
policy, _ = load_policy_from_checkpoint(ckpt, "cuda")
def loss_fn(tokens):                       # tokens: [B,40,64]
    ro = imagine_with_tokens(bundle, fm_tokens=tokens, state=..., obs_hist=...,
                             references=refs, horizon=H)
    loss, _ = compute_joint_loss(ro, weights)
    return loss
```

然后：
- `pathwise`：直接 `sample_fm_from_features`（已有）；
- `checkpointed`：把 `sample_fm_from_features` 的 loop 换成 `checkpoint(step)`（或在 `Gr00tVelocity` 外包一层）；
- `adjoint`：用本目录 `AdjointBackend` 的逻辑，`field` 换成 `Gr00tVelocity`；
- `qvgm_lite`：需要一个 **action-sensitive chunk critic**（论文用 IQL/TD），当前 `OracleCritic` 只能做机制验证，不能做论文对比。

### 4.3 显存/时间预算（4090 48GB，估算）

| 配置 | 激活 | 备注 |
|---|---|---|
| full 3B GR00T + K=4 + H=4 + batch=1，pathwise | ~25–35 GB | 需实测；可能 OOM |
| 同上 + checkpointed | ~15–25 GB | 慢 ~2×，优先试 |
| tiny action head（本目录） | ~1 GB | 只验证图是否正确 |
| 4×H200 | 141 GB/卡 | 正式消融建议在这里跑 |

---

## 5. 协议与验收标准（建议写进论文附录）

1. **梯度正确性**：`pathwise` vs `checkpointed` vs `adjoint` 的 rel err < 1e-3（fp32）；
2. **有限差分**：对 3 个参数/方向，中心差分误差 < 2%；
3. **显存**：checkpointed 相对 pathwise 的峰值激活下降 ≥30% 才算有效；
4. **时间**：adjoint 允许 ≤2× pathwise；超过则不适合；
5. **稳定性**：K∈{2,4,8,16} 下 grad norm、NaN 率、loss 曲线；
6. **真实任务**：同预算 A/B（pathwise vs truncated vs qvgm）在 held-out episodes 上的成功率/跟踪误差，而不是 toy loss。

---

## 6. 与论文 claim 的关系

- 现在可以写："we compare exact BPTT, gradient checkpointing, discrete adjoint, and a Q-VGM-style local value-gradient estimator on a controlled testbed; exact methods agree to machine precision, checkpointing trades compute for memory, and the local estimator's direction is accurate but magnitude-sensitive."
- **不能**写："Q-VGM improves success" / "adjoint is faster" / "checkpointing is free" —— 这些都还没有真实任务/真实 3B 证据。

---

## 7. 已知限制

- `qvgm_lite` 是论文 **local approximation** 的复现，不是官方实现；critic 的 IQL/TD 训练、chunk 级 action-sensitive 设计未实现；
- `truncated` 的梯度有偏，不能用于"精确梯度"结论；
- toy/tiny 的 memory 数字被 CUDA context 主导，不能外推；
- 所有结果都是单 seed；正式对比需要 ≥3 seeds + 置信区间。
