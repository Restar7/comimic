# 提速（预取/缓存）+ 续训（resume）使用说明

> 2026-10-25。对应实测瓶颈：卡 6 上联合微调 **步速 2.0–2.8 s/step**，但 GPU 利用率只有
> **23–31%**（有时 2–6%）——每步 GPU 只忙 ~0.6 s，其余时间在等 CPU：
> `TaskObservationBuilder.build`（mp4 随机访问 ~40 ms/帧）+ `Gr00tN1d7Processor`
> （图像 transform / tokenize）。而且**同一组 frames 每步被算两遍**（token 路 + 手部辅助路）。

## 1. 改了什么

| 文件 | 改动 | 作用 |
|---|---|---|
| `diff_vla_wbc/prefetch.py`（新） | `StepMemo`（步骤内缓存）+ `BatchPrefetcher`（后台单线程预取，**只碰 CPU**） | 一步只处理一次 + 主线程 backward 时把下一步的预处理做掉 |
| `scripts/train_joint.py` | 新增 `--prefetch-vla`（**默认开**）/`--no-prefetch-vla`；provider 改为 `token_source=None`，VLA 采样走 `batch.model_inputs`；循环改"提前采样下一步" | 所有走 `train_joint.py` 的消融（A1/A2/A4/A5 + A3 的 BPTT 臂）**自动吃到** |
| `scripts/train_ppo.py` | 同样的预取（默认开） | A3 的 PPO 臂 |
| `scripts/train_joint.py` | `--resume-joint CKPT` / `--resume-vla CKPT` / `--start-step N` | 续训 |
| `diff_vla_wbc/train.py` | `JointTrainer.load` 同时支持 `{policy,...}`(mjlab) 与 `{pi,...}`(Isaac)，并透传 `wm_integration`；返回 ckpt | 续训时不会把显式积分结构读成残差结构 |
| `scripts/check_prefetch_parity.py`（新） | CPU 侧等价性检查 | 预取结果必须与现算 `torch.equal` |
| `scripts/smoke_resume_and_prefetch.sh`（新） | GPU 冒烟 | ①预取开/关逐位一致 ②续训起点 eval 与原始 run 一致 |

## 2. 怎么用

### 2.1 正常训练（无需改命令）
```bash
bash scripts/launch_joint_ha.sh                 # 预取默认开
bash scripts/launch_a2_fm.sh                    # 同样自动生效
STATE_VEL_WEIGHT=0 bash scripts/launch_a3_a5.sh  # 同样
```
日志里会多一行 `[prefetch] enabled=True`。要排查时加 `--no-prefetch-vla` 关掉。

### 2.2 续训（从任意 `joint_XXXXXX.pt` / `vla_XXXXXX.pt` 继续）
```bash
# 1) 把 VLA 存档合并回一个可加载的 ckpt 目录
python3 scripts/export_vla_ckpt.py \
  --base runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --vla  runs/joint_ha_a1/vla_002000.pt --out /tmp/ckpt_ft2000 --force

# 2) 续跑：--steps 用"总步数"（例如从 2000 续到 4000 → --steps 4000）
COMIMIC=$PWD
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 -m torch.distributed.run --nproc_per_node 1 --master_port 29642 \
  scripts/train_joint.py \
    --motion data/npz/ha_tokens \
    --supermimic-ckpt trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt \
    --vlm --vlm-path groot_repro/nvidia/Cosmos-Reason2-2B-weights --vlm-ckpt /tmp/ckpt_ft2000 \
    --vlm-dataset --vlm-steps 2 \
    --batch-size 2 --horizon 8 --steps 4000 --state-vel-weight 1.0 \
    --resume-joint runs/joint_ha_a1/joint_002000.pt \
    --resume-vla   runs/joint_ha_a1/vla_002000.pt \
    --eval-every 500 --eval-batches 2 --eval-out runs/joint_ha_a1/eval_curve.jsonl \
    --hand-loss-weight 0.2 --hand-grasp-boost 5 \
    --save --save-every 1000 --save-vla-every 1000 --out runs/joint_ha_a1_r2
```
要点：
- `--steps` 是**总步数**语义；`--resume-joint` 会把 `_start_step` 设成 ckpt 里记录的步数，
  所以 eval 打点 / 文件命名（`joint_0003000.pt`）会接着原编号走；
- `--start-step` 可手工覆盖起点（比如 ckpt 里步数丢了）；
- 续训起点的 `[eval <start>]` 会先打一行，等于记录"续训基线"；
- `joint_*.pt` 现在带 `wm_integration`（显式积分结构），续训时自动继承。

C 语言版一句话：**预取默认开、续训靠 `--resume-joint/--resume-vla`，`--steps` 给总步数。**

## 3. 验证状态（诚实记录）

| 检查 | 状态 |
|---|---|
| `processed_batch` 预取/现算逐位一致（CPU） | ✅ 通过：`scripts/check_prefetch_parity.py`，7 个字段 `torch.equal` |
| CLI 参数解析（`--help`） | ✅ 通过（顺手修了 `--state-vel-weight` 帮助串里 `%` 未转义导致 `--help` 崩的问题） |
| `joint_*.pt` 带 `wm_integration` | ✅ 当前 run 的 `joint_001000/002000.pt` 已验证 |
| **预取开/关 2 步逐位一致（GPU）** | ⏳ 待跑：`bash scripts/smoke_resume_and_prefetch.sh`（需要空闲卡） |
| **续训续接（GPU）** | ⏳ 待跑：同一脚本的 ② 段 |

> ⚠️ 顺序建议：等当前主线跑到 4000 步（卡空出来）→ 先跑 `scripts/smoke_resume_and_prefetch.sh`，
> 两项都 PASS 再把 A2–A5 放出去；**没 PASS 就不许用 `--resume-*` 跑正式实验**。

## 4. 预期收益（保守估计）
- 步骤内缓存：每步少一次 processor+视频读取（原两次 → 一次）；
- 跨步预取：把剩下的 CPU 时间藏到 GPU backward 后面；
- 合计预期 **1.5–2.5×**（2.0–2.8 s/step → 0.8–1.4 s/step），具体看主机 CPU 争抢；
- 若还嫌慢：下一步才考虑"帧级 LRU 缓存"（内存换速度，随机采样命中率不高，性价比一般）。

---

## 5. 实测（2026-10-25 12:43，卡 6）

**续训正确性（`RESUME_STEP=2000`）**：从 `joint_002000.pt` + `vla_002000.pt` 续训，续训起点
`[eval 2000]` 与**原始 run（未实现预取时）**的 `[eval 2000]` 逐字段一致：

| 字段 | 原始 run | 续训（预取开） |
|---|---|---|
| `token_mse` | 0.077285 | **0.077285** ✅ |
| `state_pos` / `ori` / `q` | 3.8e-5 / 0.001459 / 0.003467 | **完全相同** ✅ |
| `state_vel` | 0.439181 | **0.439181** ✅ |
| `rollout_reward` | +0.081623 | **+0.081623** ✅ |
| `hand_mse` | 0.011385 | **0.011385** ✅ |

⇒ 这同时**端到端证明预取在数值上中性**（原始 run 没有预取，续训有预取，eval 逐位相同）。

**步速**：预取开启后 **1.50 s/step**（原 2.0–2.8 s/step）→ 约 **1.4–1.9×**。
GPU 利用率仍只有 13–29%，说明剩余瓶颈是"VLA fp32 反传 + 主线程串行"，不是纯 CPU 预处理；
想要更快得上"帧级 LRU + 多候批"或减少 `--vlm-steps`（都会动语义，需单独评估）。

**口径切换记录（重要）**：A1 主线的 step 0–2000 使用 `--state-vel-weight 1.0`，
step 2000–4000 使用 **0**（标准口径）。切换依据：速度项在 1000 步已贴地板（0.44，梯度≈0）。
因此该 run 的 `state_term` 曲线在 2000 步有断点（0.444 → 0.005），**看走势请用
`state_pos/ori/q` 分项或 `token_mse`/`rollout_reward`**；报告里必须标注这一点。
