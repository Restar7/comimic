# （历史文档）

> ⚠️ 本文记录的是 ft318 时代的联合微调路径。**官方微调产物已写死为 `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`**
> （选择依据见 `runs/OFFICIAL_CKPT.json`：ckpt-8000 的训练 loss 窗口均值 0.0401±0.0041，
> 是现存最优；7500 更优但已被 save_total_limit 删除）。联合微调默认值已指向它，
> 本文里的 ft318 checkpoint 仅作历史对照。

# 反传微调路径（锁定版）：20k VLA → token → SuperMimic → 世界模型 → 奖励 → 梯度回 VLA

> 本文把这条链路**钉死**：命令、路径、评测口径、每步产物。以后只按这里跑，不要再从 GR00T 基座重训。
> 文档时间：2026-10-24。

## 0. 一句话分工（这是之前搞错过的地方）

```
① GR00T-N1.7-3B 基座
      └─ 官方流程 + 我们的 318 条 ft318 数据    →  微调 20,000 步  →  ②
② 「我们的 VLA」= groot_sonic_vla_20k_frozen271_20261023/checkpoint-20000
      直接输出 motion_token(64) + 左右手(7+7)         ✅ 已有产物，不再重训
③ 反传微调（本文这一条）：②的 VLA 现场出 token
      → SuperMimic(token 条件化) → 世界模型 → 解析奖励 → loss → 梯度回到 VLA head  ⬅ 现在跑的就是这个
```

- **BONES-SEED（129,943 clips）只用于 ③ 里面的 SuperMimic 预训练**（已训到 20k 步，见 §5）。
- **ft318（318 eps / 473,876 帧 / 2.633 h / 4 任务）是我们 VLA 的真实训练集**，也是 ③ 的参考轨迹来源。

## 1. 锁定路径（全部绝对路径，抄错就没有第二次机会）

```bash
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
REPO=$COMIMIC/third_party/SuperMimic-Reproduced
DEPS=$COMIMIC/third_party/gr00t_deps                      # GR00T 依赖（--no-deps 装在独立目录）
VLA_CKPT=/inspire/hdd/global_user/liumengfan-253108110079/flowbody/FlowBody/runs/groot_sonic_vla_20k_frozen271_20261023/checkpoint-20000
VLM_PATH=$COMIMIC/groot_repro/nvidia/Cosmos-Reason2-2B-weights   # 注意：必须带 -weights 后缀
SM_CKPT=$REPO/logs/e9_bones_seed_token_h200x2/ckpt_0020000.pt    # SuperMimic token 条件化（BONES-SEED 预训练）
MOTION=$COMIMIC/data/npz/g1_tasks                                # 318 条 ft318 → SuperMimic 格式（含 g1_motion_token_64）
OUT=$COMIMIC/runs/joint_diff_ft318
```

| 组件 | 是什么 | 为什么是它 |
|---|---|---|
| `VLA_CKPT` | 20,000 步微调后的 GR00T N1.7（含 optimizer，24 GB） | **我们的 VLA**，直接输出 motion_token；**不要**再从基座重训 |
| `VLM_PATH` | Cosmos-Reason2-2B 权重（4.6 GB，HF 缓存软链） | 顶层视觉语言骨干 |
| `SM_CKPT` | SuperMimic token 条件化 ckpt（20k 步，BONES-SEED） | 把 token 解码成 29 维身体动作 |
| `MOTION` | 318 条 ft318 的 MotionLibrary npz | 参考轨迹 + 录制 token（评测基线） |

## 2. 启动命令（**唯一入口**，12000 步）

```bash
cd $COMIMIC
setsid nohup env \
  PYTHONPATH=$DEPS/_noflash:$DEPS:$COMIMIC \
  MUJOCO_GL=disable \
  CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=8 \
  HF_HOME=/inspire/hdd/global_user/liumengfan-253108110079/cache/huggingface \
  SONIC_ENCODER_ONNX=$REPO/data/assets/model_encoder.onnx \
  /usr/bin/python3 scripts/train_joint.py \
    --motion $MOTION \
    --supermimic-ckpt $SM_CKPT \
    --vlm --vlm-path $VLM_PATH --vlm-ckpt $VLA_CKPT --vlm-dataset \
    --vlm-steps 2 --batch-size 2 --horizon 8 \
    --steps 12000 --log-every 100 \
    --eval-every 1000 --eval-batches 2 --eval-out $OUT/eval_curve.jsonl \
    --save --save-every 1000 --save-vla-every 4000 --out $OUT \
  > /tmp/joint_diff_12000.log 2>&1 < /dev/null &
```

**每个参数为什么这么填**

| 参数 | 值 | 理由 |
|---|---|---|
| `--vlm --vlm-path ... --vlm-ckpt ... --vlm-dataset` | 三件套 | token 由 **VLA 现场推理**产生（真实 ego 视频 + 46 维状态 + 指令），不是数据集里的录制 token |
| `--vlm-steps 2` | 2 | flow-matching 去噪步数；2 步足够、并行度好（4 步更准但慢一倍） |
| `--batch-size 2` / `--horizon 8` | 2 / 8 | 48 GB 卡上的稳定档；显存紧张时先降 horizon |
| `--eval-every 1000 --eval-batches 2` | 固定 2 个批次 | 同一批帧、同一随机种子（seed=12345），保证曲线可比 |
| `--save-every 1000` | 每 1000 步 | `joint_001000.pt`… 可单独拿出 ckpt 做对比 |
| 环境变量 `_noflash` | **必须** | 反传微调输入是 fp32，FlashAttention 只支持 fp16/bf16；用 stub 挡掉后 transformers/GR00T 自动走 sdpa |
| `MUJOCO_GL=disable` | 必须 | 训练不需要渲染 |

## 3. 评测口径（每 1000 步自动打一行）

`--eval-out` 写入 JSONL，同时打印到日志：

```
[eval 0]    {"step": 0,    "token_mse": 0.06013, "state_term": 2.75248, "rollout_reward": 0.057589}
[eval 1000] {"step": 1000, ...}
```

| 指标 | 定义 | 怎么读 |
|---|---|---|
| `token_mse` | VLA 出的 token 与数据集里**录制 token**（`g1_motion_token_64`）的 MSE | **主指标**：微调有没有让 VLA 更会出 token；越小越好 |
| `state_term` | 用这批 VLA token 走 SuperMimic + 世界模型 8 步后，状态项误差 | 越小越好；录制 token 的参照值约 **0.30** |
| `rollout_reward` | 想象 rollout 的每步平均奖励 | 越大越好；录制 token 参照约 **+0.094**，随机 token 是 **−0.854** |

**参照基准（已实测，用来判断微调到底有没有用）**

| token 来源 | `state_term` | `rollout_reward` |
|---|---|---|
| 随机 token | 65.8 | −0.854 |
| **数据集录制 token（上界参照）** | **0.30** | **+0.094** |
| 20k VLA 现场出 token（**微调前 = `[eval 0]`**） | **2.75** | **+0.058** |

→ 微调前的 VLA 已经远好于随机、但离录制 token 还有明显差距；**这就是反传微调要填的 gap**。

## 4. 已知的三个坑（都是实测踩过的，别再踩）

| 坑 | 症状 | 解法 |
|---|---|---|
| `--vlm-path` 指到 `Cosmos-Reason2-2B`（不带 `-weights`） | `OSError: no file named model.safetensors` | 必须指 `Cosmos-Reason2-2B-weights` |
| FlashAttention + fp32 输入 | `RuntimeError: FlashAttention only support fp16 and bf16` | `PYTHONPATH` 最前面放 `$DEPS/_noflash`（stub 让 import 失败 → sdpa） |
| 在 `main()` 里局部 `import MotionBatchProvider` | `UnboundLocalError` | 只在需要处 import `RecordedTokenSource` |

## 5. 上游产物（本路径依赖的、已经完成的）

| 产物 | 路径 | 状态 |
|---|---|---|
| SuperMimic token 条件化（BONES-SEED 预训练） | `$REPO/logs/e9_bones_seed_token_h200x2/ckpt_0020000.pt`（20k 步，2×H200，2.5 it/s） | ✅ 在跑/已到 20k；全量评测 129,943 条：`mean 0.6476`（4k ckpt）|
| ft318 → MotionLibrary npz | `$COMIMIC/data/npz/g1_tasks/` **318 个**（758 MB，含 `g1_motion_token_64`、手部 14 维、指令、ego 视频路径）| ✅ |
| 我们的 VLA（20k 步） | `$VLA_CKPT`（24 GB）| ✅ |

## 6. 运行状态与结果（持续更新）

- 吞吐：**~2 s/步**（batch 2 / horizon 8 / vlm-steps 2，单卡 4090 48 GB）→ 12000 步 ≈ **6.7 h**
- 评测点：step 0 / 1000 / 2000 / … / 12000，共 13 个点
- 日志：`/tmp/joint_diff_12000.log`；曲线：`$OUT/eval_curve.jsonl`；ckpt：`$OUT/joint_XXXXXX.pt`


---

## 7. 审查记录（2026-10-24，启动前逐项核过）

| # | 审查项 | 结论 | 处理 |
|---|---|---|---|
| 1 | 进程是否真在跑 | ✅ `train_joint` 2 进程、GPU 40.7 GB / 46–100% | — |
| 2 | token 是否由 VLA 现场产生 | ✅ `--vlm-dataset` 走 `TaskObservationBuilder`（真实 ego 视频 + 46 维状态 + 指令）→ `sample_model_with_grad` | — |
| 3 | SuperMimic 走的是 token 路线（不是 goal） | ✅ ckpt 的 `policy.token_dim=64` → `train_step` 的 `token_route=True` → `imagine_with_tokens` | — |
| 4 | 梯度是否真的回传到 VLA | ✅ `fm_parameters` = 1,620,515,968 个可训练参数（DiT 1.09B + SelfAttention 0.20B + 其余），已加入 AdamW 的 `fm_lr` 组并参与 `clip_grad_norm_` | — |
| 5 | VLA 有没有被冻住（`--train-vlm` 未开） | 冻结的是 **backbone**（`tune_llm/tune_visual=False`）；**action head 可训练** —— 这正是"反传微调"要更新的部分 | — |
| 6 | **评测帧与录制 token 是否对齐** | ✅ `RecordedTokenSource` 取 `g1_motion_token_64[f : f+40]`，VLA 输出的就是这个 chunk（同 `action_horizon=40`），逐帧同起点 | — |
| 7 | **评测是否有随机性** | ❌ 原实现每次 `torch.randn` 初始化噪声 → 曲线抖动 | ✅ **已修**：评测前 `manual_seed(1234+i)` + 评测后恢复 CPU/CUDA RNG；两次独立运行 `[eval 0]` **完全一致**（`0.061862 / 2.668902 / 0.058795`）|
| 8 | **ckpt 是否包含微调后的 VLA** | ❌ `JointTrainer.save()` 只存 `adapter/policy/wm`，**不含 VLA 头** → 12000 步白跑 | ✅ **已修**：新增 `--save-vla-every`，每 4000 步写 `vla_XXXXXX.pt`（可训练参数，fp32，~6.5 GB/个）|
| 9 | 上次跑的 12000 步任务 | 用的是修好之前的代码，且占 39 GB 导致二次加载 OOM | 已停、重起（新版代码 + `--save-vla-every 4000`）|
| 10 | 显存 | 40.7 / 48 GB | 安全；若要加大 batch 先降 `--horizon` |
| 11 | 评测集是否 hold-out | ❌ 不是：eval 批次与训练同库采样（只固定了帧与种子）| **如实记录**：这条曲线衡量"拟合与改善趋势"，不是泛化精度；泛化要另做 hold-out 会话 |
| 12 | 速度 | ~2 s/步（batch 2 / horizon 8 / vlm-steps 2）| 12000 步 ≈ 6.7 h |
