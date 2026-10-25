# 三个消融实验：协议与运行手册

> 对应 `scripts/run_ablations.sh`（MODE=1/2/3/all）。
> 本文只写"做什么、怎么算、花多少卡时"，结果见 `runs/ablations/` 与
> `python3 scripts/ablation_report.py --root runs/ablations` 生成的汇总。

---

## 0. 三条消融各自回答什么问题

| # | 问题 | 处理组 | 对照组 | 指标 |
|---|---|---|---|---|
| 1 | 我们的反传微调到底有没有用？ | `vla_004000.pt`（4000 步可微联合微调） | 官方 20k VLA（只做行为克隆） | 闭环**成功率**（Wilson 95% CI） |
| 2 | 五种 FM 反传估计方式怎么选？ | pathwise / checkpointed / truncated / adjoint（+ 已有的 qvgm_lite toy 结果） | 同数据、同 seed、同墙钟预算 | `rollout_reward` 曲线、显存、单步耗时 |
| 3 | 固定时间下，精确反传（BPTT）和 PPO 谁训得出来？ | `train_joint.py`（pathwise 精确梯度） | `train_ppo.py`（同一 reward 的 score-function 梯度） | 同墙钟下 reward 曲线 + 能否闭环 |

三者的共同点：**除了被消融的那一个变量，其余全部相同**（数据 npz、SuperMimic WBC
checkpoint、世界模型、奖励函数、手部辅助损失 λ=0.2、种子、卡数）。

---

## 1. 消融 1：有/无反传微调 → 成功率

### 1.1 两臂

```bash
CKPT_BASE=runs/ckpt_vla20k        # 官方 20k VLA（SONIC 271 eps），不含我们的反传微调
CKPT_FT  =runs/ckpt_vla_ft4000    # joint_diff_ft318/vla_004000.pt 合并回的 checkpoint
```

`runs/ckpt_vla_ft4000` 由 `scripts/export_vla_ckpt.py` 生成：把 `vla_004000.pt`
里的 537 个张量（1.62 B 参数，全部落在 shard 2/3）写回基线 checkpoint，其余分片
软链，并用 `vla_merge_report.json` + 逐张量比对留证（实测 537/537 一致）。

### 1.2 协议

```bash
python3 scripts/success_rate.py \
    --arm base=runs/ckpt_vla20k --arm finetuned=runs/ckpt_vla_ft4000 \
    --tasks task1 task2 task3 task4 --seeds 0 1 2 3 4 \
    --chunks 40 --execute-steps 20 --hand-mode trend \
    --out runs/ablations/a1_success_rate --video
```

* 一个 episode = 40 chunks = 32 s 部署节奏（2.5 Hz 重规划，执行前 20 tick）；
* 成功判定直接复用部署侧谓词（`closed_loop_mujoco.py::_check_success`），
  task1/2 走官方 `check_success.py`，task3/4 用 place 谓词；
* 每个 (arm, task, seed) 一个独立进程：OOM/崩溃只影响该 episode，可续跑；
* 汇总写 `summary.json` / `SUMMARY.md`（Wilson 区间）。

### 1.3 样本量与判读

二值 episode → 粒度 1/n。`4 tasks × 5 seeds = 20` 单臂只有 5% 粒度，
**两臂差 ≥ 2 个 episode 才算有信号**；要写进论文请用 `--seeds 0..9`（40 单臂，
约 45 min/臂）。成本 ≈ 70 s/episode（实测 task2 66 s、task3 65 s、task4 60 s）。

### 1.4 ⚠️ 任务集：HA 7 任务是 Isaac Sim only（重要）

如果是想在 **HumanoidArena 的 7 个任务**上出成功率，必须先知道：

* HA 官方用 **Isaac Sim 5.0 + Isaac Lab 2.2（PhysX）**，任务定义在
  `tasks/g1_tasks/<pkg>/*_hw_env_cfg.py`，成功判定是 `compute_success_mask` 二值奖励；
* **仓库不提供场景 USD/MJCF**（`assets/` 被 gitignore，要单独下 1 GB 的 assets.zip），
  MJCF 只有机器人本体；
* 唯一能脱离 Isaac 的入口是 `visualize_rotlocal_v3_mujoco.py`：纯运动学回放，
  没有道具、没有物理、没有谓词。

所以要在本机跑 HA 成功率，只有三条路：

| 方案 | 代价 | 能否和论文数字比 |
|---|---|---|
| **A. 用现成 SONIC 4 任务 MuJoCo 闭环**（`closed_loop_mujoco.py`） | 今天就能跑，代码已就绪 | 不能比绝对值；但 **A/B 两臂内部可比**，消融结论有效 |
| **B. 自建 HA 版 MuJoCo 场景**（移植 7 个 AABB/距离/角度/接触力谓词 + 自窗口场景道具） | 谓词好移植（~600 行）；**场景/道具要自己建**，约 2–4 周 | 不能：MuJoCo vs PhysX 的接触/摩擦不同，抓取、门闩、坐沙发结果会偏 |
| **C. 本机装 Isaac Sim 5.0 + Isaac Lab** | 安装 + 资产多 GB；需要 RTX 渲染卡（本机 2×4090 够） | 最忠实，但装机与调试风险最高 |

**建议先跑 A**：它不改动任何实验变量，只换测量环境，照样能回答"反传微调有没有涨"。

---

## 2. 消融 2：FM 反传方法

### 2.1 已有的部分（不用重跑）

`ablation_fm_grad/`：toy（MLP velocity）、tiny（真实 GR00T head 类）、真实
1.09 B action head 三种规模上的**梯度正确性 / 显存 / 时间**，以及 K 的 scaling。
结论已经写在 `ablation_fm_grad/REPORT.md`：pathwise ≡ checkpointed ≡ adjoint（rel=0），
truncated 有偏（rel≈0.67–0.88，cos>0.99），qvgm_lite 只在 oracle critic 下可用。

### 2.2 本次新增：真实训练对比

```bash
MODE=2 AB2_MIN=30 bash scripts/run_ablations.sh
```

给 `scripts/train_joint.py` 加了 `--fm-grad-backend {pathwise,checkpointed,truncated,adjoint}`
（`diff_vla_wbc/fm_grad.py`），每个后端**固定 30 min 墙钟**、同数据同种子跑真训练，
比较 `eval_curve.jsonl` 里的 `rollout_reward` / `state_term` / `token_mse`。

正确性由 `tests/test_fm_grad_backends.py` 钉住（12 个用例，含真实 tiny GR00T head：
adjoint/checkpointed 与 pathwise 相对误差 0，truncated cos=0.9999、rel=0.67）。

**注意**：`adjoint` 只覆盖 ODE 链，会 detach backbone 特征 → **不要**和 `--train-vlm`
同时用（会告警）。`qvgm_lite` 需要 chunk critic，真实栈里没有，故不提供（会显式报错）。

### 2.3 成本

4 后端 × 30 min = **2 h / 2 卡**。要更长预算就调 `AB2_MIN`。

---

## 3. 消融 3：BPTT vs PPO（固定墙钟）

```bash
MODE=3 AB3_MIN=120 bash scripts/run_ablations.sh
```

* 两臂各 **120 min**（用户口径），同一份数据/模型/reward/手部辅助损失；
* BPTT：`train_joint.py --fm-grad-backend pathwise`（现有训练器）；
* PPO：`scripts/train_ppo.py`，把 action head 的输出包成对角高斯策略
  `a = FM_θ(obs,z) + σ·ε`，`z` 每个 batch 抽一次并**存下来**，update 时用同一个 `z`
  重算均值 → ratio 是真 PPO ratio；
* 由于 VLA 一次只吐一个 40×64 chunk、世界模型用同一个 chunk 滚 H 步，
  MDP 的有效 horizon = 1，所以这是 **T=1 的 PPO（带学习基线 + clip）**，
  实现里也这么写，不冒充多步 GAE。

### 3.1 判读顺序

1. **先看两臂是否"能动"**：`rollout_reward` 曲线是否显著离开 step-0 基线；
2. 再看同墙钟下谁的 reward 高、`eval` 的 `state_term` 是否下降；
3. 最后（可选）把两臂各自的 `vla_final.pt` 走消融 1 的闭环测成功率。

成本 4 h / 2 卡（2 h × 2 臂）。

---

## 4. GPU 预算与排队

单机 2×4090（各 48 GB）。实测显存（batch 2 / horizon 8 / vlm-steps 2）：

| 任务 | 每卡占用 | 备注 |
|---|---|---|
| `train_joint.py`（BPTT） | ~46 GB | 全链反传，独占两卡 |
| `train_ppo.py` | 明显更低（rollout 走 no_grad） | 只需 action head 的反传图 |
| 闭环推理 | ~12–20 GB | 可与轻任务共存，但别和训练抢 |

**总预算**：消融 1（推理）≈ 1 h → 消融 2 ≈ 2 h → 消融 3 ≈ 4 h，合计 **约 7 h**。
官方 HA 20k 微调跑完（约 08:10）之后按 1→2→3 顺序排即可。

---

## 5. 一键复现

```bash
# 先确认要用的 checkpoint 都在
ls runs/ckpt_vla20k runs/ckpt_vla_ft4000

DRY=1 MODE=all bash scripts/run_ablations.sh      # 只打印命令
MODE=1 bash scripts/run_ablations.sh              # 消融1（~1 h）
MODE=2 AB2_MIN=30 bash scripts/run_ablations.sh   # 消融2（~2 h）
MODE=3 AB3_MIN=120 bash scripts/run_ablations.sh  # 消融3（~4 h）

python3 scripts/ablation_report.py --root runs/ablations \
    --out docs/ABLATIONS_REPORT.md
```
