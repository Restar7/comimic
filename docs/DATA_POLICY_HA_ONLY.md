# 硬性数据/链路政策（永久锁定）

> 写死于 2026-10-24。**任何代码、任何训练、任何评测都不得违反本文。**
> 违反时入口脚本必须直接报错退出，不允许静默回退。

## 0. 两条铁律

1. **只走纯 HA（HumanoidArena）数据。**
   - 唯一允许的 VLA/联合微调数据根：`data/npz/ha_tokens`（699 episodes / 634,016 frames）。
   - 监督微调数据根：`data/ha_sessions`（由 `data/ha_raw` 转换）。
   - **永久禁止**：`data/npz/mixed_ha_ft318`、`data/npz/g1_tasks`、`data/npz/g1_tasks_ft318`、
     任何名字含 `mixed` 或 `ft318` 的数据集；以及用它们训出来的
     `runs/joint_diff_ft318/*`、`runs/ckpt_vla_ft4000`、`runs/ckpt_vla20k`（ft318 时代的 20k）。
2. **只走 IsaacLab / PhysX 链路（`trackerlab-supermimic`）。**
   - WBC + 世界模型 + 99 维观测 + 奖励 + 软终止 + 可微 FK：全部来自
     `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_*.pt`（Isaac Lab 训练，纯 HA）。
   - **永久禁止**：`third_party/SuperMimic-Reproduced`（mjlab / MuJoCo-Warp）的权重/解析头；
     `closed_loop/` 下的 MuJoCo 闭环；任何 `mujoco` 评测结果作为正式口径。

## 0.5 两条路线（**写死，任何文档/脚本/评测都不得混用**）

本项目**一直有两条并行路线**，A1 的对比就是它们：

| # | 路线 | 执行链 | VLA 训练 | 执行器 | 产物 |
|---|---|---|---|---|---|
| **L1** | **官方线** | VLA → 64 token（HA 标签空间）→ **SONIC release decoder** → 29 动作 | 官方监督微调（`checkpoint-8000`） | `model_decoder.onnx`（**release 真权重已就位**，复现 HA 录制动作 1e-5 度，见 `docs/SONIC_DECODER_FOUND.md`）；`runs/sonic_decoder_final.pt` 是蒸馏替代，**只作离线对照** | 官方基线成功率 |
| **L2** | **联合微调线（我们的贡献）** | VLA → 64 token（**装配空间**，端口侧车）→ **SuperMimic WBC(policy)** → WM 想象 → reward → BPTT 回 VLA head | 监督微调 + **奖励反传联合微调** | `ckpt_010500_physint.pt`（WM 冻结） | 论文主结果 |

**铁律**：
1. 两条线的 token 空间**不同**（L1=部署标签空间；L2=装配空间），**不得互相喂 token**；
2. 同一次对比里**两条臂必须同一个执行器**（要么都 decoder，要么都 WBC）；
3. 写任何报告必须标清楚这条结果属于 L1 还是 L2。

## 1. 唯一的正式链路

```
HA raw (data/ha_raw)
  ├─ prepare_ha_sessions.py  -> data/ha_sessions     -> 官方 GR00T 监督微调
  └─ prepare_ha_tokens_npz.py-> data/npz/ha_tokens   -> SuperMimic / 联合微调
                                                      |
官方 VLA 底模: runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000  (HA 监督, 写死)
联合微调:    scripts/train_joint.py --motion data/npz/ha_tokens
             --supermimic-ckpt trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_*.pt
评测:        scripts/run_a1_ha_isaac.sh  (Isaac Sim 5.0 + IsaacLab 2.2 + HumanoidArena)
```

## 2. 代码里的锁

| 入口 | 锁 | 行为 |
|---|---|---|
| `scripts/train_joint.py` | `_forbid_legacy_dataset()` | `--motion` / `--recorded-tokens` 含 `mixed`/`ft318` → `SystemExit` |
| `scripts/train_joint.py` | `is_isaac_checkpoint()` | `--supermimic-ckpt` 不是 trackerlab（无 `pi`/`wm`）→ `SystemExit`，不回退 mjlab |
| `scripts/success_rate.py` | MuJoCo 门禁 | 未设 `COMIMIC_ALLOW_MUJOCO_ROUTE=1` 时直接退出（只允许历史复现） |
| `scripts/run_a1_ha_isaac.sh` | 臂/ckpt 审计 | 处理臂必须由纯 HA 联合微调产出，且 merge base == 对照组底模 |
| `scripts/launch_joint_ha.sh` | 默认值 | `MOTION=data/npz/ha_tokens`、`SM_CKPT=.../bones_seed_token_ha_v2/ckpt_*.pt` |

`train_joint.py` 顶部写死：

```python
ISAAC_SM_CKPT = .../trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt
HA_MOTION_DIR = .../data/npz/ha_tokens
FORBIDDEN_DATA_TOKENS = ("mixed", "ft318")
```

## 3. 自检命令（CI/交接前必跑）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

# 1) 训练入口必须拒绝 mixed/ft318
python3 scripts/train_joint.py --motion data/npz/mixed_ha_ft318 --steps 1 2>&1 | grep -q "永久只走纯 HA" \
  && echo "PASS: mixed rejected"

# 2) 非 Isaac WBC 必须拒绝
python3 scripts/train_joint.py --supermimic-ckpt third_party/SuperMimic-Reproduced/logs/e9_bones_seed_token_h200x2/ckpt_0020000.pt --steps 1 2>&1 | grep -q "只接受 Isaac/trackerlab" \
  && echo "PASS: mjlab WBC rejected"

# 3) MuJoCo 评测默认关闭
python3 scripts/success_rate.py --arm base=runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 --dry-run 2>&1 | grep -q "permanently disabled" \
  && echo "PASS: MuJoCo disabled"

# 4) 纯 HA 数据计数
ls data/npz/ha_tokens/*.npz | wc -l     # 期望 699
```

## 4. A1 的纯 HA 口径（card 6）

| 项 | 值 |
|---|---|
| 对照组 | `runs/ckpt_vla_ha_8000_local`（= `checkpoint-8000` 的本地路径视图，权重逐字节相同） |
| 处理组 | `runs/ckpt_vla_ha_ft4000`（由 `runs/joint_ha_a1/vla_004000.pt` 合并回**同一**底模） |
| WBC | `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500.pt`（两臂共用，冻结） |
| 任务 | HumanoidArena 6 个可跑任务（`GrapCup` 缺资产，不跑） |
| 指标 | 成功率 + Wilson CI、未摔倒率、平均存活步数、`max_reward`、配对 McNemar |
| 入口 | `bash scripts/run_a1_ha_isaac.sh` |

> 旧 `docs/ABLATION_ALLOCATION.md` 里 A1 的 `runs/ckpt_vla_ft4000`（mixed ft318）和
> MuJoCo 命令**作废**，以后只按本文与 `run_a1_ha_isaac.sh` 执行。

## 5. 本次 A1 运行状态（2026-10-25）

| 项 | 值 |
|---|---|
| 处理臂训练 | `runs/joint_ha_a1`，4000 步，日志 `logs/joint_ha_a1.log` |
| 训练数据 | `data/npz/ha_tokens`（699 eps） |
| 底模 | `runs/ckpt_vla_ha_8000_local`（= `checkpoint-8000` 本地视图） |
| WBC（冻结） | `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500.pt` |
| 自动接力 | `scripts/a1_pipeline_after_train.sh` |
| 评测入口 | `scripts/run_a1_ha_isaac.sh` |
| 评测协议 | 6 个 HA 任务 × seeds 0/1/2 × 1 repeat，`MAX_STEPS=300` |
| 汇总 | `runs/ablations/A1_joint_finetune_ha/SUMMARY.md` |

训练侧第一个 eval（step 500）出现 **token_mse 0.069 → 0.399** 而
`state_term 21.9 → 1.11` / `rollout_reward 0.014 → 0.073` / `hand_mse 0.756 → 0.010`：
这是典型的“想象 reward 变好、录制 token 保真度变差”。闭环 A1 以
`SUMMARY.md` 的 success / 存活步数 / `max_reward` 为准，不以训练侧 reward 代替成功率。

## 6. 卡 6 训练重启记录（2026-10-25 06:45）

| 项 | 值 |
|---|---|
| 中断原因 | 容器重启（PID 1 起于 01:49:10），训练进程在 step ≈1150 被带走，日志无 traceback |
| 第 1 轮残留 | `runs/joint_ha_a1/_attempt1_20261025/`（`joint_001000.pt` / `vla_001000.pt` / `eval_curve.jsonl`），日志 `logs/joint_ha_a1_attempt1.log` |
| 重启命令 | `bash scripts/launch_joint_ha.sh`（`STEPS=4000`，纯 HA，WBC=`ckpt_013500.pt`） |
| 训练进程 | `TRAIN_PID=1326602`，日志 `logs/joint_ha_a1.log` |
| 自动接力 | `TRAIN_PID=1326602 bash scripts/a1_pipeline_after_train.sh`，日志 `logs/a1_after_train.log` |
| 起点校验 | 重启后 `eval 0` 与第 1 轮逐位相同（token_mse 0.069364 / state_term 21.861635 / rollout_reward 0.014182 / hand_mse 0.756083）；step 100 loss -0.119617 vs 第 1 轮 -0.119618 |
| 预计 | 4000 步 ≈ 2 h（≈1.8 s/step），随后自动 merge → `run_a1_ha_isaac.sh`（6 任务 × seeds 0/1/2） |
