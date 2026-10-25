# 联合微调 + 消融实验：七卡分卡方案（含命令）

> **⚠️ 2026-10-24 硬修订（优先于本文所有旧命令）**：A1 旧处理组 `runs/ckpt_vla_ft4000`
> 是 **mixed ft318**（1017 条混合数据），已**永久禁用**；MuJoCo 闭环也**永久禁用**。
> A1 正式口径改为 **纯 HA 数据 + IsaacLab / HumanoidArena**，见
> `docs/DATA_POLICY_HA_ONLY.md` 与 `scripts/run_a1_ha_isaac.sh`。
> 本文 §2 A1 的 MuJoCo 命令与 `ckpt_vla_ft4000` **仅作历史记录，不得再执行**。

> **2026-10-25 05:53 起跑（优先于本文的卡位表）**：A2（4 后端）、A5（2 臂）、A3 的 **BPTT** 臂
> 已在 `comimic4` 的 4 张卡上开跑（chain1=卡 0,1：A2→A5；chain2=卡 2,3：A3-BPTT），
> 产物目录改用 `runs/ablations/A2_fm_backend_v2_physint/`、`runs/ablations/A5_only_head_v2_physint/`、
> `runs/ablations/A3_bptt_vs_ppo/bptt`；实时状态与结论见 **`docs/ABLATION_STATUS_20261025.md`**。
> 本文 §2/§3 的旧命令、旧目录名与 `$SM_CKPT=ckpt_010500.pt` 一律作废（口径以
> `docs/HANDOVER_MAINLINE_AND_ABLATIONS_20261025.md` 为准）。
>
> 写死于 2026-10-24。两个 ckpt 已固定，见 `runs/OFFICIAL_CKPT.json` 与 `runs/OFFICIAL_SUPERMIMIC.json`。
>
> **当前唯一阻断项**：Isaac 版 SuperMimic 的接口与我们的 `diff_vla_wbc` bundle **尚未完全一致**
> （详见 §1），在对齐层写出来之前**不要起训练**，否则是静默跑偏。

---

## 0. 两个写死的 ckpt（唯一来源）

| 角色 | 路径 | 说明 |
|---|---|---|
| **官方微调产物（VLA）** | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000` | step 8000，训练 loss 窗口均值 **0.0401 ± 0.0041**（该 run 现存最优）。元数据：`runs/OFFICIAL_CKPT.json` |
| **官方 SuperMimic（底层+世界模型）** | `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt` | **Isaac Sim/PhysX 训练**，token 接口，iteration 10500，HA 数据。元数据：`runs/OFFICIAL_SUPERMIMIC.json` |

**规则（写死）**：
- SuperMimic 的官方 ckpt **只从 `trackerlab-supermimic/logs/<最新 run>/ckpt_*.pt` 取**；
  `third_party/SuperMimic-Reproduced/logs/*` 是 mjlab(MuJoCo-Warp) 版，**只能作历史对照**，不得作为官方底模。
- 两个 launch 脚本的默认值都已指向上面这两个路径，只有显式 `VLM_CKPT=...` / `SM_CKPT=...` 才会覆盖。

---

## 1. 接口一致性现状（这是起训练前的必过关卡）

对 `ckpt_010500.pt`（Isaac 版，token 接口）做了逐层比对：

| 项 | Isaac 版（theirs） | 我们 bundle（ours） | 状态 |
|---|---|---|---|
| policy 解码器首层 | `dec.0.weight (2048, 1054)` = 64 token + 10×**99** obs | `(2048, 994)` = 64 token + 10×**93** obs | ❌ 每帧差 6 维 |
| policy 隐层 | 2048→1024→512→256→32→29 | 同 | ✅ 完全一致 |
| policy encoder | **不存在**（token 接口移除） | `encoders.motion.*`（680 维 goal） | ✅ token 路由下不使用 |
| policy obs 归一化 | `obs_norm` RunningNorm(99) | `decoder.scale`(930) | ❌ 形式与维度都不同 |
| WM 线性层 | 133→2048→2048→2048→70（net.0/3/6/9） | 同维度（dynamics.mlp.net.0/2/4/6） | ✅ 维度一致 |
| WM 归一化/结构 | LayerNorm(net.1/4/7) + `in_norm`(133) + `out_norm`(70) | **无** LayerNorm、无 in/out norm | ❌ 需补壳 |
| 终止头 | **不存在** | `term_head` 4→64→64→1 | ❌ theirs 缺（可随机初始化） |

**结论**：三处不一致（policy 首层 1054 vs 994、obs 归一化、WM 的 LayerNorm/in-out-norm），
另有一处缺失（终止头）。必须写一个 **对齐层**，做三件事：

1. 我们的 obs 构造器要能产出 **99 维/帧**（先确认那多出来的 6 维是什么：读
   `trackerlab-supermimic/source/trackerLab/.../` 里 env 的 `obs_dim` 定义）；
2. 把 their `obs_norm(mean,var)` 折进我们 `decoder.scale`（或新增一个 running-norm 包装）；
3. WM 包一层：`in_norm` → Linear+LayerNorm×3 → `out_norm`，权重从 `net.*` 直接搬；
   终止头随机初始化，并在日志里登记「该项是新的」。

对齐层写好后的验收：**同一 token + 同一 obs 下，两个 policy 前向输出差 < 1e-4**（否则映射错）。

> 脚本已有雏形：`scripts/adapt_trackerlab_ckpt.py` 能审计并做「形状对得上的层全拷、其余显式重初始化」的部分加载，
> 但它**不能**替代上面的对齐层（属于"部分加载"，不是"接口一致"）。

---

## 2. 五个消融（逐个给全）

> 占位符（都已写死，见 `runs/OFFICIAL_SUPERMIMIC.json` / `runs/OFFICIAL_CKPT.json`）：
> - `SM_CKPT` = `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt`（Isaac Lab/PhysX 版 SuperMimic WBC）
> - `VLM_PATH` = `groot_repro/nvidia/Cosmos-Reason2-2B-weights`
> - `VLM_CKPT` = `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（官方微调产物，写死）
> - `MOTION` = `data/npz/mixed_ha_ft318`（HA+ft318 混合，1017 条）或 `data/npz/ha_tokens`（纯 HA，699 条）

### 卡位总表

| 组 | 卡 | 是否需训练卡 | 预估墙钟 |
|---|---|---|---|
| **A1** 有无联合微调（主结果） | **卡 6** | 否（纯评测；与训练并行时独占 6） | ≈47 min（4 任务×5 seed×2 arm） |
| **A2** FM 反传方法 | **卡 0,1** | 是（2 卡 DDP） | ≈2 h（4 臂 × 30 min） |
| **A3** PPO vs BPTT | **卡 2,3** | 是（2 卡 DDP） | ≈4 h（2 臂 × 120 min） |
| **A4** WM 冻结/更新 × horizon | **卡 4,5** | 是（2 卡 DDP） | ≈4 h（6 格 × 40 min） |
| **A5** 冻结 SuperMimic 只训 head | **卡 2,3**（A3 之后）或 **0,1**（A2 之后） | 是 | ≈1.5 h（2 臂 × 40 min） |

### A1 · 有无联合微调（主结果）

| 项 | 内容 |
|---|---|
| **卡** | 6（评测） |
| **变量** | VLA 是否经过「奖励反传到 action head」的联合微调 |
| **处理组** | `runs/ckpt_vla_ft4000`（由 `joint_diff_ft318/vla_004000.pt` 合并，537 张量已校验） |
| **对照组** | `runs/ckpt_vla20k`（官方 20k，纯监督） |
| **指标** | 每任务成功率、总体成功率、Wilson 95% CI、未摔倒率、task3/4 的连续量（dx/dy/z_rel） |
| **代码** | `scripts/export_vla_ckpt.py`（已完成合并）+ `scripts/success_rate.py`（多 seed + CI + 断点续跑） |
| **阻塞** | 需要 arena 7 任务场景（P0-1）。当前只有 SONIC 4 任务可跑，**不能当 arena 口径** |

```bash
# 处理组 checkpoint（如尚未生成）
python3 scripts/export_vla_ckpt.py --base runs/ckpt_vla20k \
  --vla runs/joint_diff_ft318/vla_004000.pt --out runs/ckpt_vla_ft4000 --force

# 多 seed 成功率（--tasks 换成 arena 的 7 个任务名后即为正式口径）
python3 scripts/success_rate.py \
  --arm base=runs/ckpt_vla20k --arm finetuned=runs/ckpt_vla_ft4000 \
  --tasks task1 task2 task3 task4 --seeds 0 1 2 3 4 --chunks 40 \
  --out runs/ablations/A1_joint_finetune --video
```

- **输出**：`runs/ablations/A1_joint_finetune/{<arm>/<task>_s<seed>/meta.json, state.jsonl, run.log}`，
  汇总 `rows.json` / `summary.json` / `SUMMARY.md`；视频 `A1_joint_finetune/<arm>/videos/*.mp4`。
- **判读**：episode 是二值的 → 20 单臂粒度 5%，**两臂差 ≥2 个 episode 才算信号**；论文请用
  `--seeds 0..9`（40 单臂）。
- **成本**：≈70 s/episode；4 任务 × 5 seed × 2 arm ≈ 47 min。

### A2 · FM 反传方法

> **✅ 2026-10-24 17:20 起跑（本容器 wscoli2 = 第 1、2 卡，2× RTX 4090）**
> 下面 §A2 的旧命令已过期（`$MOTION` 是混合数据、`$SM_CKPT` 还是 mjlab 版），**实际入口**：
>
> ```bash
> # 冻结代码快照（快照里带 CODE_MANIFEST.sha256，启动时会自校验）
> runs/_code_a2_20261024
> # 启动（4 臂 × 30 min 同墙钟，串行；单实例 flock 保护；10 s 显存采样）
> setsid nohup bash runs/_code_a2_20261024/scripts/launch_a2_fm.sh > logs/a2_master.log 2>&1 &
> # 训练结束后自动出报告（docs/ABLATION_A2_RUN.md + runs/ablations/A2_fm_backend/A2_FINALIZED.txt）
> setsid nohup bash runs/_code_a2_20261024/scripts/finalize_a2_fm.sh > logs/a2_finalize.log 2>&1 &
> ```
>
> 与旧命令的硬性差异：数据 `data/npz/ha_tokens`（纯 HA，699 clips）+ 打包库
> `runs/isaac_motion/ha_tokens`；SuperMimic = Isaac `{pi,wm}` `ckpt_010500.pt`（iteration 10500）；
> VLA = `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000` 且 **VLM backbone 冻结**（不设 `--train-vlm`）；
> 共享超参 `--batch-size 2 --horizon 8 --hand-loss-weight 0.2 --hand-grasp-boost 5`，
> **唯一变量 = `--fm-grad-backend`**；输出 `runs/ablations/A2_fm_backend/<backend>/`。
>
> 起跑前自检（都已通过）：4 个后端各 2 步冒烟全 `rc=0`、日志回显 `backend=<b>` 与请求一致、
> 手部标签 = `data/npz/ha_tokens`（无 ft318）、`eval 0` 与另一容器 A3 的 `eval 0` 逐位相同
> （token_mse 0.069362 / state_term 18.617166 / rollout_reward 0.018358 / hand_mse 0.756087），
> 说明底模 / 数据 / 评测 batch 三处口径一致。

| 项 | 内容 |
|---|---|
| **卡** | 0,1 |
| **变量** | 梯度如何穿过 FM 的 K 步 Euler 链 |
| **臂** | `pathwise`（默认，精确 BPTT）/ `checkpointed`（精确，O(1) 激活）/ `truncated`（有偏，廉价）/ `adjoint`（精确离散伴随）；`qvgm_lite` 需 chunk critic，真实栈**不提供**（会显式报错），只在 `ablation_fm_grad/` 的 toy/tiny 里保留 |
| **指标** | `rollout_reward`（主）、`state_term`、`token_mse`、峰值显存、s/step、以及（可选）闭环成功率 |
| **代码** | `diff_vla_wbc/fm_grad.py` + `--fm-grad-backend`；`tests/test_fm_grad_backends.py`（12 用例，含真实 head） |
| **注意** | `adjoint` 会 detach backbone 特征 → **不要**同时用 `--train-vlm` |

```bash
for b in pathwise checkpointed truncated adjoint; do
  O=runs/ablations/A2_fm_backend/$b
  CUDA_VISIBLE_DEVICES=0,1 /usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29641 \
    scripts/train_joint.py \
      --motion $MOTION --supermimic-ckpt $SM_CKPT \
      --vlm --vlm-path $VLM_PATH --vlm-ckpt $VLM_CKPT --vlm-dataset --vlm-steps 2 \
      --batch-size 2 --horizon 8 --steps 1000000 \
      --fm-grad-backend $b --time-budget-minutes 30 \
      --hand-loss-weight 0.2 --hand-grasp-boost 5 \
      --eval-every 300 --eval-batches 2 --eval-out $O/eval_curve.jsonl \
      --save-every 300 --save-vla-every 0 --out $O
done
```

- **输出**：`runs/ablations/A2_fm_backend/<backend>/eval_curve.jsonl` + `joint_*.pt` + 训练日志。
- **判读**：先比 `rollout_reward` 的收敛斜率与终值；`truncated` 应更快但更差（有偏）；`adjoint`
  与 `pathwise` 应几乎重合（若明显偏离 → 实现 bug）。
- **成本**：4 × 30 min ≈ 2 h（两卡）。

### A3 · PPO vs BPTT（固定墙钟）

| 项 | 内容 |
|---|---|
| **卡** | 2,3 |
| **变量** | 梯度的来源：精确反传 vs score-function |
| **臂** | `train_joint.py`（pathwise） vs `train_ppo.py`（`a = FM_θ(obs,z) + σ·ε`，`z` 每 batch 固定，PPO clip + 学习基线） |
| **指标** | 同墙钟 `rollout_reward` 曲线、`ppo/sigma`、`ppo/clip_frac`、能否闭环 |
| **代码** | 已就绪；PPO 的**有效 horizon = 1**（VLA 一次吐一个 token chunk，世界模型用同一个 chunk 滚 H 步）→ 是 T=1 的 clipped PG，**论文里不能写成多步 GAE** |
| **注意** | 两臂必须**同一墙钟**（`--time-budget-minutes` 相同），并且都带手部辅助损失（λ=0.2）以保持公平 |

```bash
# BPTT
CUDA_VISIBLE_DEVICES=2,3 /usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29642 \
  scripts/train_joint.py --motion $MOTION --supermimic-ckpt $SM_CKPT \
    --vlm --vlm-path $VLM_PATH --vlm-ckpt $VLM_CKPT --vlm-dataset --vlm-steps 2 \
    --batch-size 2 --horizon 8 --steps 1000000 --fm-grad-backend pathwise \
    --time-budget-minutes 120 --hand-loss-weight 0.2 --hand-grasp-boost 5 \
    --eval-every 300 --eval-batches 2 \
    --eval-out runs/ablations/A3_bptt_vs_ppo/bptt/eval_curve.jsonl \
    --out runs/ablations/A3_bptt_vs_ppo/bptt

# PPO
CUDA_VISIBLE_DEVICES=2,3 /usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29643 \
  scripts/train_ppo.py --motion $MOTION --supermimic-ckpt $SM_CKPT \
    --vlm --vlm-path $VLM_PATH --vlm-ckpt $VLM_CKPT --vlm-steps 2 \
    --batch-size 2 --horizon 8 --ppo-epochs 4 --sigma-init 0.25 \
    --time-budget-minutes 120 --hand-loss-weight 0.2 \
    --eval-every 100 --eval-batches 2 \
    --eval-out runs/ablations/A3_bptt_vs_ppo/ppo/eval_curve.jsonl \
    --out runs/ablations/A3_bptt_vs_ppo/ppo
```

- **输出**：两个 `eval_curve.jsonl` + `vla_final.pt`（可直接接 A1 的成功率流程）。
- **判读**：先看**能不能动**（reward 是否离开 step-0 基线），再比同墙钟终值；`ppo/sigma` 若塌到下限
  说明策略退化到确定性。
- **成本**：2 × 120 min = 4 h。

### A4 · 世界模型冻结/更新 × imagination horizon

> **⚠️ 本节的命令已被取代（2026-10-24 更新）**：A4 现在跑在**完整 Isaac Lab 栈 + 纯 HA 数据**上，
> 一键脚本 = `scripts/run_a4_wm_horizon.sh`，运行报告 = `docs/ABLATION_A4_RUN.md`。
> 与该节下面原文的差异（都是有理由的，见报告）：
> 1. **不再混用 mjlab 解析层**：观测 / 奖励 / 软终止 / 可微 FK / 参考运动库全部来自
>    `trackerlab-supermimic`（`diff_vla_wbc/isaac_joint.py`）；旧的
>    `diff_vla_wbc/trackerlab_bundle.py` 已删除。
> 2. **数据写死纯 HA**：`data/npz/ha_tokens`（699 clips / 634,016 frames）。
>    **mixed / ft318 永久禁用**（`train_joint.py` 与打包器都会硬性拒绝）。
> 3. **`MOTION` 不再是 `data/npz/mixed_ha_ft318`**。
> 4. **batch 全网格统一 1**（2 卡 DDP → 全局 2）：实测 bs=2 单卡峰值 46/48 GB，bs=1 降到 39 GB。
> 5. **WM 在线更新参数改小**：`--wm-update-every 50 --wm-update-warmup 100`（原 100/500 会让
>    H=40 在 40 min 内一次都不更新，该臂静默退化成冻结臂）。
> 6. **墙钟退出改为两卡协同**（原来先到墙钟的 rank 退出会让另一张卡死锁在 `all_reduce`）。
> 7. 自检：`python3 scripts/check_isaac_bundle.py --ckpt <Isaac ckpt> --packed runs/isaac_motion/ha_tokens/packed.pt`
>    （9 项全部 0 或 fp32 噪声级）。

| 项 | 内容 |
|---|---|
| **卡** | 4,5 |
| **变量 1** | WM 是否参与联合更新：`--wm-update-every 0`（冻结，默认） vs `--wm-update-every 50 --wm-update-warmup 100`（在线更新） |
| **变量 2** | imagination horizon：`--horizon {8,16,32,40}`（默认 8） |
| **为什么上界是 40** | VLA 一次输出 **40 帧 token chunk**；`imagine_with_tokens` 对 step t 取 `tokens[:, min(t, T-1)]`，所以 horizon=40 正好用完整个 chunk，>40 会退化成重复最后一帧 → **40 是原则性的上界** |
| **代码** | 纯 CLI（`--wm-update-every/-warmup/-batch/-window`、`--horizon` 都已接线到训练循环） |
| **指标** | `rollout_reward`、`state_term`、`rollout/survival`、显存、s/step |

```bash
# 网格：WM 冻结 × H{8,16,32,40}
for H in 8 16 32 40; do
  O=runs/ablations/A4_wm_horizon/wm_frozen_H$H
  CUDA_VISIBLE_DEVICES=4,5 /usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29644 \
    scripts/train_joint.py --motion $MOTION --supermimic-ckpt $SM_CKPT \
      --vlm --vlm-path $VLM_PATH --vlm-ckpt $VLM_CKPT --vlm-dataset --vlm-steps 2 \
      --batch-size 2 --horizon $H --steps 1000000 --wm-update-every 0 \
      --time-budget-minutes 40 --hand-loss-weight 0.2 --hand-grasp-boost 5 \
      --eval-every 300 --eval-batches 2 --eval-out $O/eval_curve.jsonl --out $O
done
# 网格：WM 更新 × H{8,40}
for H in 8 40; do
  O=runs/ablations/A4_wm_horizon/wm_live_H$H
  CUDA_VISIBLE_DEVICES=4,5 /usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29644 \
    scripts/train_joint.py --motion $MOTION --supermimic-ckpt $SM_CKPT \
      --vlm --vlm-path $VLM_PATH --vlm-ckpt $VLM_CKPT --vlm-dataset --vlm-steps 2 \
      --batch-size 2 --horizon $H --steps 1000000 \
      --wm-update-every 100 --wm-update-warmup 500 --wm-update-batch 8 \
      --time-budget-minutes 40 --hand-loss-weight 0.2 --hand-grasp-boost 5 \
      --eval-every 300 --eval-batches 2 --eval-out $O/eval_curve.jsonl --out $O
done
```

- **输出**：`runs/ablations/A4_wm_horizon/{wm_frozen_H*,wm_live_H*}/eval_curve.jsonl`；
  WM 更新臂的日志里会多出 `loss/wm_*` 字段，用来确认 WM 真的在更新。
- **判读**：horizon 变大 → 反传链变长、显存与时间上升；看 reward 提升是否值得。WM 在线更新若
  让 reward 抖动，说明 WM 与策略在互相追逐（需要更小的 `--wm-update-every` 频率或更大 warmup）。
- **成本**：6 × 40 min ≈ 4 h。
- **显存注意**：`--horizon 8 --batch-size 2` 实测 46.6/49 GB；H=16/32/40 需把 `--batch-size` 降到 1
  （否则 OOM），并在日志里记录实际显存峰值。

### A5 · 冻结 SuperMimic，只训 VLA action head

| 项 | 内容 |
|---|---|
| **卡** | 2,3（A3 之后）或 0,1（A2 之后） |
| **变量** | 联合微调时是否也更新 WBC（SuperMimic policy） |
| **处理组** | `--policy-lr 0 --adapter-lr 0`（只更新 action head，`--fm-lr 1e-5`） |
| **对照组** | 默认 `--policy-lr 1e-5`（head + policy 一起更新） |
| **指标** | `rollout_reward`、`state_term`、闭环成功率 + **参数变化审计** |
| **代码** | 纯 CLI。注意：token 路线**不使用 adapter**，所以 `--adapter-lr 0` 只是保险；VLM backbone 本来就被 `freeze_vlm=True` 冻结 |

```bash
# 处理组：只训 head
O=runs/ablations/A5_only_head/head_only
CUDA_VISIBLE_DEVICES=2,3 /usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29645 \
  scripts/train_joint.py --motion $MOTION --supermimic-ckpt $SM_CKPT \
    --vlm --vlm-path $VLM_PATH --vlm-ckpt $VLM_CKPT --vlm-dataset --vlm-steps 2 \
    --batch-size 2 --horizon 8 --steps 1000000 --policy-lr 0 --adapter-lr 0 \
    --time-budget-minutes 40 --hand-loss-weight 0.2 --hand-grasp-boost 5 \
    --eval-every 300 --eval-batches 2 --eval-out $O/eval_curve.jsonl --out $O

# 对照组：head + policy
O=runs/ablations/A5_only_head/head_plus_wbc
CUDA_VISIBLE_DEVICES=2,3 /usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29645 \
  scripts/train_joint.py --motion $MOTION --supermimic-ckpt $SM_CKPT \
    --vlm --vlm-path $VLM_PATH --vlm-ckpt $VLM_CKPT --vlm-dataset --vlm-steps 2 \
    --batch-size 2 --horizon 8 --steps 1000000 \
    --time-budget-minutes 40 --hand-loss-weight 0.2 --hand-grasp-boost 5 \
    --eval-every 300 --eval-batches 2 --eval-out $O/eval_curve.jsonl --out $O
```

- **参数变化审计**（证明冻结生效，跑完必做）：见 `scripts/` 下的对照脚本；要点是
  `max|joint_latest.policy[k] − SM_CKPT.policy[k]| == 0` 才算真的冻结。

- **判读**：若 head-only 已经能涨到和 head+WBC 接近，说明收益主要来自 VLA head，**可以省掉 WBC 更新**
  （也顺带避免了"训练代理与部署执行器不一致"的风险放大）。
- **成本**：2 × 40 min ≈ 1.5 h。

---

## 3. 排期建议（7 卡满载）

| 时间窗 | 卡 0,1 | 卡 2,3 | 卡 4,5 | 卡 6 |
|---|---|---|---|---|
| 第 1 段 | **A2**（4×30 min） | **A3**（2×120 min） | **A4**（6×40 min） | **A1**（评测 47 min，可并行） |
| 第 2 段 | A5（head_only / head+WBC） | （接 A5 亦可） | （补 A4 剩余格） | A1 的 `--seeds 0..9` 扩样 |

> 所有训练命令都用 `--time-budget-minutes` 固定墙钟，保证跨臂可比；`--steps` 给一个极大值即可（由墙钟截断）。

---

## 4. 预飞结论（2026-10-24 实测，2 卡，每条 6–10 步）

| 预飞 | 命令形状 | 结果 |
|---|---|---|
| A0 主线形状 | `train_joint.py --horizon 8 --batch-size 2`（SM ckpt 用可加载版） | ✅ exit 0，`[step 1..10]` 正常 |
| A2 `truncated` 后端 | `--fm-grad-backend truncated` | ✅ exit 0 |
| A4 `horizon 16` | `--horizon 16 --batch-size 1` | ✅ exit 0（显存可承受） |
| A5 冻结 policy | `--policy-lr 0 --adapter-lr 0` | ✅ exit 0 |
| A3 PPO 臂 | `train_ppo.py` | ❌ `ValueError: initial_noise must be (2,40,132), got (2,40,64)` |
| **主线 + Isaac 版 SM ckpt** | `--supermimic-ckpt trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt` | ❌ `ValueError: ... has no 'config' entry; not a SuperMimic checkpoint` |

### 4.1 已修掉的两个 bug

1. `train_ppo.py`：`sample_model_with_grad` 在 `torch.no_grad()` 里被调用 → `RuntimeError: sample_model_with_grad must be called with grad enabled`。已在该调用处局部 `torch.enable_grad()`。
2. `train_ppo.py` 的噪声宽度：写死/按 `action_head.action_dim` 造噪声与实际期望（`max_action_dim=132`）不一致。已改为从 batch 的 action 维度推导 **（仍不通过，见 4.2）**。

### 4.2 两个未解决的阻断

**阻断 1：A3 PPO 臂**
症状：`initial_noise must be (2,40,132), got (2,40,64)`。已定位到采样器内部按 `max_action_dim=132` 校验，而 PPO 侧构造的 batch 里 action 宽度是 64；`train_joint.py` 用同一套 `processed_batch` 却能跑通，说明 `train_ppo.py` 里 `Gr00tVLM` 的加载参数（`max_action_dim`）与 `train_joint.py` 不一致。
处理建议：把 `train_ppo.py` 的模型加载改成与 `train_joint.py` 同一路径（`from_pretrained(checkpoint=...)` 走 checkpoint 自带 config），而不是走默认 `max_action_dim=64`。

**阻断 2：Isaac 版 SuperMimic ckpt 不能直接加载**
`ckpt_010500.pt` 的键是 `{pi, wm, critic, args, reward_cfg}`，我们的 loader 要求 `{config, policy, wm}`；且结构上还差：
- policy 首层 `(2048,1054)` vs 我们 `(2048,994)`（观测每帧 99 vs 93，多出的 6 维 = `rot6d(Rᵀ·R_ref)`，**可计算**）；
- obs 归一化 `obs_norm(mean,var)` vs 我们的 `decoder.scale`；
- WM 有 `LayerNorm(net.1/4/7)` + `in_norm/out_norm`，我们无；
- 终止头我们有无（they 用规则式，影响可忽略）。

处理建议（按工作量排序）：
1. **写转换器 + 观测扩展到 99 维**（推荐）：改 `analytic_observation` 拼 6 维相对朝向 → policy 首层自动 1054；WM 加 3 个 LayerNorm 与 in/out norm（权重直搬）；再产出一个 `config`+`policy`+`wm` 的 ckpt。验收：同一 token/obs 下两 policy 前向差 < 1e-4。
2. **在 trackerlab runner 里接 VLA 反传**（架构最干净，但工作量大）。

> 在阻断 2 解决前，**A1/A2/A3/A4/A5 里凡是用 `$SM_CKPT = ckpt_010500.pt` 的命令都跑不起来**；
> 用 `third_party/SuperMimic-Reproduced/logs/e9_bones_seed_token_h200x2/ckpt_0024000.pt`（mjlab 版）
> 则全部可跑 —— 但那是"MuJoCo 动力学"的底模，不是 Isaac 口径。

---

## 5. 接口对齐实施规格（全用 Isaac 版，不做混用）

**决策（写死）**：SuperMimic 只用 `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt`
（Isaac Lab/PhysX 版）；**旧的 mjlab 版 `ckpt_0024000` 不再使用、不做混用对照**。

### 5.1 从源码抠出来的确切规格（不是猜）

**Policy（token 接口，无 encoder）**
```python
# networks.py: Policy.__init__ → self.dec = mlp(z_dim + obs_dim*hist, [2048,1024,512,256], 29)
# 输入 = [token(64) , obs_hist(10 × 99)]  → 64 + 990 = 1054      与 ckpt dec.0.weight (2048,1054) 一致
# obs(99) = [q - q0 - dq0 (29), qd (29), Rᵀw (3), g_b (3), a_prev (29), rot6d(RᵀR_ref) (6)]
# 隐层 mlp(..., act=ELU, layer_norm=False)；obs 归一化 = RunningNorm(99) 于解码器输入前
```

**WorldModel**
```python
in_dim  = (13 + 2*29) + 29 + (29 + 4) = 71 + 29 + 33 = 133
out_dim = 12 + 2*29 = 70
self.net = mlp(133, [2048,2048,2048], 70, act=nn.SiLU, layer_norm=True)   # LN 在每层后
self.in_norm  = RunningNorm(133);  self.out_norm = RunningNorm(70, eps=1e-4)
step(S, a, xi_n):
    y = net(in_norm(cat(invariant_features(S)=71, a=29, xi_n=33)))        # 133 → 70
    d = y * out_norm.std() + out_norm.mean
    dp, dθ, dv, dw, dq, dqd = d.split((3,3,3,3,29,29))
    p1 = p + H @ dp ;  R1 = R @ so3_exp(dθ) ; v1 = v + R @ dv ; w1 = w + R @ dw
    q1 = q + dq ; qd1 = qd + dqd                                          # H = yaw_mat(R)
```

**终止**：规则式（`height_thr` / `tilt_thr` + 关节限位），**没有学习头** → 我们侧 term_head 保持随机初始化即可（loss 权重 0.01/0.0，影响可忽略）。

### 5.2 要新增的东西（**只新增，不删旧路径**）

| 文件 | 内容 | 行数 |
|---|---|---|
| `diff_vla_wbc/trackerlab_adapter.py` | `TLPolicy`、`TLWorldModel`（复刻上面两段规格，含 RunningNorm，权重从 `pi`/`wm` 直搬） | ~150 |
| `diff_vla_wbc/obs99.py` | 在 93 维后追加 `rot6d(Rᵀ·R_ref)`（复用 `heads.py` 的旋转工具 + `references[t].root_quat_w`） | ~40 |
| `scripts/convert_trackerlab_ckpt.py` | 产出 `{config, policy, wm}` 的 bundle ckpt（写新文件，原 ckpt 只读） | ~80 |
| `tests/test_trackerlab_parity.py` | 前向等价验收 | ~80 |

### 5.3 验收（不过就不许开训）

1. **policy 前向**：同一 `obs_hist(10,99)` + 同一 token(64) 输入下，
   `TLPolicy` 与 trackerlab 的 `Policy` 输出最大绝对差 **< 1e-4**；
2. **WM 单步**：同一 `(S, a, xi_n)` 下，`TLWorldModel.step` 与 `WorldModel.step` 的
   `s'`（71 维扁平化后）最大绝对差 **< 1e-4**；
3. **obs99**：我们算的 6 维与 `mat_to_6d(Rᵀ·R_ref)` 逐元素差 **< 1e-6**；
4. 三项全绿 → 用转换后的 bundle 起主线（卡 0,1），此时 `--supermimic-ckpt` 指向**转换产物**。

### 5.4 影响面（再确认一次）

| 对象 | 是否改动 |
|---|---|
| VLA ckpt（`checkpoint-8000`，64 token + 14 手） | 不动 |
| `train_joint.py` / loss / reward / rollout / eval | **不动**（只把 bundle 里的 policy/WM 换成 adapter 实现，接口 `act()/step()/reward()` 不变） |
| 旧的 mjlab 路径 | 保留但不再使用（按决策不混用） |

### 5.5 已确认的两处口径细节（写代码时按这个来）

1. **71 维 invariant 特征（两边顺序相同）**：
   trackerlab `mdp.invariant_features` = `[p_z(1), mat_to_6d(HᵀR)(6), Rᵀv(3), Rᵀw(3), q(29), qd(29)]`；
   我们的 `spec.canonical` = `[p_z, rot6d(q_local), v_h, w, q, qd] / in_scale` —— **结构相同**，
   区别是归一化方式（in_norm vs in_scale）与 **w 的坐标系**（他们世界系、我们体坐标系）。
2. **状态换系**：我们扁平 71 维里 `ang_vel` 是**体坐标系**，trackerlab 状态里 `w` 是**世界系**；
   跨实现时必须 `w_world = R · w_body`，否则 WM 前向差会很大（parity 测试会直接抓到）。

**状态布局（我们的扁平 71 维）**：`[root_pos(0:3), root_quat wxyz(3:7), joint_pos(7:36),
lin_vel_w(36:39), ang_vel_b(39:42), joint_vel(42:71)]`。
