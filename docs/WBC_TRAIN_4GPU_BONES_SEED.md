# SuperMimic WBC 训练（4×RTX 4090 / 8192 envs per GPU / 完整 BONES-SEED）

> 2026-10-25。给 4 卡容器（comimic4）用的 IsaacLab SuperMimic WBC 训练命令。
> 单卡容器（本机）只能跑 `NPROC=1`，命令同样适用。

---

## 0. TL;DR（复制即用）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic

export SONIC_ENCODER_ONNX=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx

MODE=train NPROC=4 ENVS_PER_RANK=8192 ITERATIONS=32000 \
MAX_EPISODE_STEPS=0 \
LIBRARY=data/bones_seed/bones_seed_compact.pt \
INTERFACE=token COMPILE=1 HA_ALIGN=1 CKPT_EVERY=500 \
PORT_BASE=29660 LOG_DIR=logs/bones_seed_token_fullclip_4x8192 \
setsid nohup bash scripts/rls/supermimic/train_bones_seed.sh \
  > logs/bones_seed_fullclip_4x8192.launcher.log 2>&1 &
```

- `MAX_EPISODE_STEPS=0` → **跑到整条 clip 结束**（最长 10000 帧 ≈ 200 s 的那条也会跑完）
- 保守替代：`MAX_EPISODE_STEPS=1200` → 覆盖 p90（1416 帧）的 85%，不被极端长 clip 拖死
- 冒烟（先验证链路，4 envs / 3 iter / 1 卡）：
  ```bash
  MODE=smoke NPROC=1 bash scripts/rls/supermimic/train_bones_seed.sh
  ```

---

## 1. 这条命令用的是什么数据（完整 BONES-SEED）

| 项 | 值 |
|---|---|
| 库文件 | `data/bones_seed/bones_seed_compact.pt`（6.75 GB） |
| clip 数 | **129,943**（完整 BONES-SEED，不是 699 条 HA） |
| 频率 | 50 Hz（`ref / names / starts / lengths`） |
| clip 长度 | min **29** / p10 149 / 中位 **288** / p90 628 / p99 1,531 / 最长 **8,979** 帧（mean 365.4 帧 ≈ 7.3 s） |
| 参考 run | `logs/bones_seed_token_ha_v2`（`num_envs 8192`、`iterations 32000`、`interface token`、`compact_library`、`shard_motion interleave`） |

> ⚠️ **2026-10-25 11:15 勘误（已按库文件逐项复核）**：本表原先写的"中位 **789** 帧（15.8 s）、均值 **907**、最长 **10000**"
> 是 **HA 库**的统计（`data/npz/ha_tokens`，699 clips / 634,016 帧：median 789 / mean 907.0 / max 10000），
> 被误贴到 BONES-SEED 这节。BONES-SEED 实测：clips **129,943**、frames **47,475,680**
> （= `ref.shape[0]`，逐位一致）、`names` 唯一 129,943 个（无重复段）、`starts` 严格连续（无空洞/重叠）。
> 数据完整性结论不变，**只是分位数换成上面这行**。

> 结论：数据是**完整**的；之前的问题不是"数据不全"，而是 **episode 上限 500 步**（= clip 中位长度的 63%）+ 训练停在退化的 32k 点。

---

## 2. 这条命令展开成什么

```bash
<trackerlab_env>/python -m torch.distributed.run --nproc_per_node 4 --master_port 29660 \
  scripts/rls/supermimic/train.py \
    --task g1 \
    --motion_packed data/bones_seed/bones_seed_compact.pt \
    --interface token --goal_frames 10 --hist 10 --horizon 32 \
    --utd 0.4 --lr 2e-4 --max_episode_steps 0 \
    --compact_library --compile \
    --num_envs 8192 --iterations 32000 \
    --shard_motion --shard_mode interleave \
    --ckpt_every 500 --amp bf16 \
    --kit_args=--kit_args=--allow-root \
    --log_dir logs/bones_seed_token_fullclip_4x8192
```

| 参数 | 含义 |
|---|---|
| `--interface token` | 64 维 SONIC token 由 `SonicTokenBuilder` **在线**从参考帧算出（不需要 `.tokens.npy` 侧车）；需要 `SONIC_ENCODER_ONNX` |
| `--shard_motion --shard_mode interleave` | 129,943 条 clip 按 rank 交错分片（每卡 ~32k 条） |
| `--max_episode_steps 0` | **本次新增**：跑到 clip 结束（见 §5 代码改动） |
| `--num_envs 8192` | 每个 rank 的环境数（总 4×8192 = 32,768） |
| `HA_ALIGN=1` | 训练侧驱动 = HA/SONIC 的 PD + 默认姿态 + 摩擦组合（`trackerTask/supermimic/robot.py:149` 的 `ha_gains`） |

---

## 3. 资源预算（48 GB 卡）

| 项 | 4096 env/rank | 8192 env/rank |
|---|---|---|
| packed library 分片 | ~1.6 GiB | ~1.6 GiB |
| replay buffer（1000 步） | ~4.4 GiB | ~9 GiB |
| 网络 + 优化器 | ~0.5 GiB | ~0.5 GiB |
| PhysX / Isaac Lab | ~10–12 GiB | ~20–24 GiB |
| **合计** | **~20 GiB** | **~30–35 GiB** |

- 8192 env/rank 在 48 GB 上可行；**OOM 就降** `ENVS_PER_RANK=6144` 或 `4096`。
- 墙钟参考：8192 env × 32k iter ≈ **10.6 h**（单卡实测）。DDP 不缩短单 iter 墙钟，但总样本 ×4。

---

## 4. 跑起来之后怎么盯

```bash
# 训练日志（每 50 iter 一行）
tail -f logs/bones_seed_token_fullclip_4x8192/*.log
# 关键列：sim_r（真实仿真奖励）、ep_len、ep_ret、term
```

**选 ckpt 的判据（重要）**：按 **`sim_r` + `ep_len`** 挑，不要用 `last.pt`。
参考 run（`bones_seed_token_ha_v2`）的实测轨迹：

| iter | sim_r | ep_len | ep_ret |
|---|---|---|---|
| 10,500 | 0.877 | 216.6 | 0.858 |
| 12,500 | 0.876 | **223.2** | 0.857 |
| **17,500** | **0.888** | 208.1 | 0.837 |
| 25,550 | 0.870 | 177.0 | 0.730 |
| 32,000 | 0.849 | **108.5** | **0.350** |

→ **峰值在 12.5k–17.5k，32k 明显退化**。所以 `ITERATIONS=32000` 只是沿用旧设置，
真正该用的是"按 sim_r/ep_len 选出的 ckpt"；**建议同时把 `CKPT_EVERY` 保持在 500**。

---

## 5. 本次代码改动（已合并，py_compile 通过）

| 文件 | 改动 |
|---|---|
| `scripts/rls/supermimic/train.py` | 新增 `--max_episode_steps`（默认 500；**≤0 = 跑到 clip 结束**） |
| `source/trackerLab/trackerLab/algorithms/supermimic/runner.py` | `task_cfg.max_episode_steps = 2**31-1 if ≤0 else 值` |
| `scripts/rls/supermimic/train_bones_seed.sh` | 透传 `MAX_EPISODE_STEPS`（默认 500） |

语义：`env.py:343-348` 的结束条件为
```python
truncated = (f + 1 >= clip_end) | (ep_step + 1 >= max_episode_steps)
```
即"clip 走完"或"步数上限"二者先到者。`≤0` 时只由 clip 结束触发。

---

## 6. 这条训练**不能**解决的事（重要，别混）

当前闭环"一直摔"的根因不在 episode 长度，也不在数据完整性。按优先级：

| 优先级 | 问题 | 现状 |
|---|---|---|
| **P0** | 闭环 reset 用的是 Arena 初始姿态（TWIST2 −0.2/0.4/−0.2），而 WBC 训练是**参考帧 reset** + HA 默认姿态（−0.312/0.669/−0.363） | ❌ 未修 |
| **P0** | obs99 末尾 6 维（参考朝向）在部署是**常量**（`SM_REF6D=zeros`），训练时是 `mat_to_6d(Rt @ root_R[f])` | ❌ 未修 |
| P1 | eval 侧 Arena 资产驱动 vs 训练 `HA_ALIGN=1` 的 kp/kd 是否逐项一致 | 待核对 |
| P2 | 训练 episode 上限（本文档要解决的） | ✅ 已改 |

**结论**：本训练是"让 WBC 学到长时程稳定"的正确方向，但闭环摔倒要先修 P0 的两项，否则训得再好也可能照样摔。
