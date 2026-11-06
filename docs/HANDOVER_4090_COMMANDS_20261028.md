# 4090 迁移交接：**锁定口径 + 全部命令**（2026-10-28）

> 目标：换到一张**能跑 Isaac 5.0 闭环**的 4090 后，按本文从上到下执行，**不需要再判断口径**。
> 配套：`docs/RUNBOOK_S3_HANDOFF_20261028.md`（流程/门）、`docs/S3_E2E_PLAN_V4_20261028.md`（方案/O1–O15）。
> **规则：本文里的环境变量就是口径；不要临时改。**

---

## 0. 锁定口径（**不要改**）

| 项 | 值 |
|---|---|
| 仓库路径 | `/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic`（不同挂载时 `export COMIMIC=<新路径>`） |
| **plant** | **HA dex3**：`third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd`（= A24000 的 `robot_spec.usd`） |
| WBC | `runs/S1_handover/WBC_A24000_wmvel_cmd.pt`（**作为 base WM**；刷新件 v4 另说） |
| **起点** | `SM_INIT_FROM_LIB=1 SM_START_FRAME=<库行号>`：root + 29 关节 + qd **一起**从库里读 |
| **手** | `SM_HAND=0`（训练时手指锁 0；喂 VLA 手指是训练没见过的条件） |
| 重规划 | `HA_VLA_EXEC_STEPS=20` |
| 录像 | `RECORD_VIDEO=0`（采集）；验收再开 |
| token 源 | GT 臂 `online`（不需要 server）；VLA 臂 `http`（需要 server） |
| 会诊器 | `runs/isaac_motion/ha_cmd/packed.pt`（命令流） |
| 驱动 | **570.124.06**（595.x 上 Isaac 5.0 会崩，见 O15） |

---

## 1. 迁移后第一件事：环境自检（5 分钟）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
export COMIMIC=$PWD

nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
# 期望：driver_version = 570.124.06；显存 49140 MiB

bash scripts/isaac_env.sh check
# 期望：torch / isaacsim / isaaclab 全 OK
```

**冒烟（30 步，GT 臂，不需要 server）**：

```bash
mkdir -p logs
ARM=gt TASK=DoubleDesk SEED=0 START_FRAME=0 MAX_STEPS=30 GPU=0 \
  OUT=$PWD/runs/probe_isaac bash scripts/s3_collect_arm.sh 2>&1 | tee logs/probe_isaac.log
grep -c "Warp CUDA error" logs/probe_isaac.log        # 期望 0
grep -E "episode_start|init from library|episode_end" logs/probe_isaac.log   # 期望都出现
```

- 若出现 `Warp CUDA error ... cuDeviceGetUuid` → **环境未好（O15）**，停，先换驱动/容器；
- 若正常 → 继续。

---

## 2. 起 VLA server（VLA 臂需要；GT 臂可跳过）

```bash
mkdir -p runs/B_r1/vla_in
setsid nohup env PORT=18080 GPU=0 LOG_DIR=$PWD/runs/B_r1/vla_in \
  bash scripts/start_vla_server.sh > logs/server_B_r1.log 2>&1 < /dev/null & disown
sleep 60; grep -E "listening|model loaded" logs/server_B_r1.log
# 期望：[server] listening on http://127.0.0.1:18080/infer
```

> `--log-dir` 会落 `req_*.npz`（rgb / state64 / state46 / chunk），**本轮训练的 VLA 输入就靠它**。
>
> **server 占 ~10 GB 显存**。采集时它和某一个 Isaac 共用一张卡（10+20=30 GB 可以）；
> 如果那张卡还要跑第二个 Isaac（+20 GB），就会 OOM。分辨方法：`nvidia-smi` 看占用。

---

## 3. 采集：**4 env 并行 × 六类任务 × 随机起点**（对齐 SuperMimic）

> **SuperMimic 的 reset**（源码 `supermimic/envs/mdp/commands.py`）：
> 随机 clip（`randint(0, num_clips)`）+ 随机帧（`randint(0, num_frames)`）+ reset 扰动。
> **没有"枚举 env seed"**——seed 只是 RNG 种子。
>
> **我们为什么只需要改 provider 这一侧**：HA 的任务 env **没有运动参考**
> （`commands=None`、`terminations` 为空、`rewards` 是物体任务奖励、摔倒靠 tilt），
> 所以"从第 N 帧开始"只影响 token/初始化/我们自己的漂移与奖励——**env 不需要改**。
>
> | SuperMimic | 我们 | 实现 |
> |---|---|---|
> | 随机 clip | `sample_start_row.py` 在该任务的 100 条 clip 里随机挑 | ✅ |
> | 随机帧 | 在该 clip 内均匀随机，留 `MAX_STEPS+80` 帧余量 | ✅ |
> | reset 扰动 | `SM_RESET_NOISE=1 SM_RESET_SEED=<episode>`（xy 0.10m / yaw 0.25rad / q 0.05 / qd 0.30） | ✅ |
> | env seed | **固定 0** | ✅ |

### 3.1 六类任务（默认：每条 episode 随机 clip + 随机帧 + 随机扰动）

```bash
export COMIMIC=$PWD
TASKS="DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi" \
EPISODES=3 ARMS="vla" JOBS=2 NUM_GPUS=2 MAX_STEPS=600 \
ROOT=$PWD/runs/B_r2 LOGD=$PWD/logs/B_r2 \
  bash scripts/s3_collect_parallel.sh
```

- **6 任务 × 3 集 × 600 步 = 18 条 / 10,800 步**（约 45–50 min，2 并发）；
- 每集：随机 clip + 随机帧（`seed=episode`）+ 不同 reset 扰动；
- 输出到 **`runs/B_r2/`**（`runs/B_r1` 的 300 步数据只作管线验证与 A/B 基线，不用于训练）。

> **显存红线（2026-10-28 踩过）**：一张 4090 只有 49 GB。VLA server ~10 GB，
> 每个 Isaac ~15–20 GB。**`JOBS` 不要超过 `NUM_GPUS`**（一卡一个机器人）；
> 想 4 并发就先确认 `nvidia-smi` 每卡 < 40 GB。**`JOBS > NUM_GPUS` 会 OOM**：
> 现象是日志停在 `provider 对齐 PASS`、**没有 `episode_start`**、`dump/` 为空。
> 跑起来后 1 分钟看一眼：`nvidia-smi` 每卡 < 40 GB，`ls runs/B_r2/*/dump/` 开始出 `.pt`。

- `START_FRAMES` 默认 **`random`**：每条 episode 由 `sample_start_row.py` 按任务挑行号，
  打印在 log 的 `[collect] random start -> row=<N>`；
- 产出：`runs/B_r2/vla_<Task>_f<row>_r<ep>/{dump,runs}` + `logs/B_r2/...log`；
- **18 条 × 600 步、2 并发 ≈ 45–50 min**（EPISODES=3）；单条 6 任务 ≈ 8 min；
- 想看/复现某个行号：`python3 scripts/sample_start_row.py --task DoubleDesk --seed 0 --margin 380`。

### 3.2 公平对照（GT vs VLA，同一条起点、同一扰动）

```bash
ARMS="gt vla" TASKS="DoubleDesk" EPISODES=1 START_FRAMES="0" JOBS=2 NUM_GPUS=2 MAX_STEPS=300 \
  bash scripts/s3_collect_parallel.sh
```

- **A/B 固定用行 0**（和已有的 GT 基线 0.19 m 可比）；批量采集才用 `random`；
- 两条臂用**同一个 `RESET_SEED`** ⇒ 同一个行号 + 同样的 reset 扰动；
- 只有 token 源不同（GT=online / VLA=http），plant/手/exec/起点全部锁定。

### 3.3 采集验收（三条）

1. 每条都有 `dump/*.pt`，`states/actions/tokens/frames/q_target_applied` 齐全；
2. `grep -c "Warp CUDA error" logs/B_r2/*.log` 全为 0，且每条有 `episode_end`；
3. 每条 log 里有 `[collect] random start -> row=...`、`[SuperMimic] init from library: row=...`
   和 `RESET NOISE seed=...`（起点、扰动都生效）。

### 3.4 为什么**一个 Isaac 进程只跑一集**（多集的危害）

| 危害 | 说明 |
|---|---|
| **起始帧退化成固定** | `SM_START_FRAME` 是**进程级**环境变量：一个进程内所有 episode 只能用同一个起始行，"随机起点"就没了，只剩 reset 扰动 |
| **一次崩溃丢多集** | 第 3 集崩了，前 2 集虽已 flush，但第 3 集丢；进程级 OOM/段错误时很难定位是哪一集 |
| **集间 reset 未必干净** | 物体摆放、相机、RNG、接触缓存都要重置；HA 的 `--episode_batch_json` 路径能用但**没像单集那样验证过**，残留状态会静默污染数据 |
| **长进程缓慢涨显存** | Isaac/RTX 长时间跑有内存增长，多集更明显（OOM 风险） |
| **日志/归因变难** | 一个 log 里 3 集，出问题要按时间切 |

**收益**：省掉每集 ~2 min 的启动（18 集 ≈ 省 30 min 墙钟）。
**结论**：现阶段**一进程一集**（18 个进程、2 并发）；等把"每集起始帧"做成队列（~20 行）再考虑多集。

## 4. 打分 + 画漂移图（**归因：1 m 是谁的锅**）

```bash
# 打分（用训练同一个 reward；脚本内部已把 asset 序转 standard）
SUPERMIMIC_HA_ALIGN=1 PYTHONPATH=.:third_party/SuperMimic-Reproduced/src \
  python3 scripts/compute_collect_reward.py \
    --in runs/B_r1/gt_DoubleDesk_f0_r0/dump/*.pt --out runs/B_r1/gt_DoubleDesk_f0_r0/scored.pt --device cuda:0
SUPERMIMIC_HA_ALIGN=1 PYTHONPATH=.:third_party/SuperMimic-Reproduced/src \
  python3 scripts/compute_collect_reward.py \
    --in runs/B_r1/vla_DoubleDesk_f0_r0/dump/*.pt --out runs/B_r1/vla_DoubleDesk_f0_r0/scored.pt --device cuda:0

# 漂移曲线（root / joint / tilt 三张图）
PYTHONPATH=. python3 scripts/plot_drift.py \
  --dump runs/B_r1/gt_DoubleDesk_f0_r0/dump/*.pt \
  --dump runs/B_r1/vla_DoubleDesk_f0_r0/dump/*.pt \
  --out runs/B_r1/drift_gt_vs_vla.png --title "DoubleDesk start=0"
```

**判决（G1）**

| 结果 | 结论 | 下一步 |
|---|---|---|
| t=0 root <1 cm、joint <0.02 rad | 起点对齐 ✅ | 继续 |
| VLA 臂 200 步漂移 **0.1–0.2 m** | 之前 1 m 是**口径问题** | 采集重跑即可 |
| VLA 臂 200 步漂移 **仍 >0.5 m** | 是**策略闭环发散** | 正是联合微调要修的，继续第 5 步 |
| 任一臂摔 | 记录，别丢数据；≥2/3 集不摔才继续 | 检查手/asset/起点 |

---

## 5. 正式一轮（六类采集 → 一个共享 head → 先验 DoubleDesk）

### 5.1 采集（§3.1 已产出六类任务；这里只补更多 episode）

```bash
# 第 1 轮用 §3.1 的 6 条即可；要更稳就每个任务多跑 2 条（不同 reset 扰动）
EPISODES=3 TASKS="DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi" \
START_FRAMES="random" ARMS="vla" JOBS=2 NUM_GPUS=2 MAX_STEPS=600 \
ROOT=$PWD/runs/B_r2 LOGD=$PWD/logs/B_r2 \
  bash scripts/s3_collect_parallel.sh
```

### 5.2 打分 → 真转移 → WM 刷新（多步 + B 轨迹）

```bash
OUTS=$(ls -d runs/B_r2/vla_*_f*_r*); SC=()
for d in $OUTS; do
  SUPERMIMIC_HA_ALIGN=1 PYTHONPATH=.:third_party/SuperMimic-Reproduced/src \
    python3 scripts/compute_collect_reward.py --in $d/dump/*.pt --out $d/scored.pt --device cuda:0
done

PYTHONPATH=. python3 scripts/build_real_transitions_v2.py \
  --out runs/wm_real/real_transitions_v2b.pt \
  --extra-dumps 'runs/B_r2/vla_*_f*_r*/scored.pt'

PYTHONPATH=. SUPERMIMIC_HA_ALIGN=1 python3 scripts/finetune_wm_velocity_policy.py \
  --ckpt runs/S1_handover/WBC_A24000_wmvel_cmd.pt \
  --out  runs/S1_handover/WBC_A24000_wmvel_v5.pt \
  --packed runs/isaac_motion/ha_cmd/packed.pt --token-space label \
  --rollout-steps 3 --steps 300 --batch 256 --lr 3e-5 --vel-lambda 5 \
  --traj-dump 'runs/B_r2/vla_*_f*_r*/dump/*.pt' --traj-frac 0.2 --traj-val-frac 0.25 \
  --val-horizon 20 --val-every 50 --val-max-starts 32 --device cuda:0
```

**WM 门（G0-b）**

```bash
PYTHONPATH=. SUPERMIMIC_HA_ALIGN=1 python3 scripts/eval_wm_multistep.py \
  --ckpt runs/S1_handover/WBC_A24000_wmvel_cmd.pt runs/S1_handover/WBC_A24000_wmvel_v5.pt \
  --dump 'runs/B_r2/vla_Boxing_*/dump/*.pt' \
  --horizon 20 --stride 20 --device cuda:0
```

判据：20 步关节误差 **≤ 1.05× base** 才用 v5；否则回 base。

### 5.3 训练（H=20、中心 π_k、起点 = B 重规划）

```bash
V3=1 TRUST_CENTER=prev START_FROM_B='runs/B_r2/vla_*_f*_r*/dump/*.pt' \
ACT_EPS=0.6 ACT_WEIGHT=1.0 LAMBDA_R=0.3 LAMBDA_BC=0 \
RUN=s3_r1 BATCH=4 ACCUM=8 H=20 K=4 STEPS=300 EVAL_EVERY=100 EVAL_BATCHES=8 SAVE_EVERY=100 \
WBC_CKPT=$PWD/runs/S1_handover/WBC_A24000_wmvel_v5.pt \
EXTRA="--amp bf16 --lora-rank 32 --fm-lr 1e-4 --no-dr --vla-state-from-burnin \
       --no-critic --act-hinge-cap 2 --lambda-hand 0.2" \
bash scripts/launch_joint_ha_v2.sh 2>&1 | tee logs/s3_r1.log
```

WBC 对照臂（可选，另一张卡）：再加 `TRAIN_WBC=1 WBC_LR=1e-6 ACT_EPS=0.3 RUN=s3_r1_wbc`。

**时间**：300 步 × 16–18 s/步 ≈ **85 min**（这是整轮的主要开销）。
想更快只有三条：① 降到 200 步（约 55 min，轮数略多）；② 两卡 DDP（未实现，约 1.7×）；
③ 只训最后 k 个 block（未实现）。**采集已经不是瓶颈**（六类 4 并发约 8 min）。

### 5.4 闭环验收（换 head 进 server）

```bash
# 关掉旧 server，起新的（base ckpt + 训好的 head + 落盘）
pkill -f "[h]a_vla_latent64_server" || true
HEAD_CKPT=$PWD/runs/s3_r1/vla_000300.pt PORT=18081 GPU=0 \
LOG_DIR=$PWD/runs/s3_r1/vla_eval_in bash scripts/start_vla_server.sh &
sleep 60

# 用训练后的 WBC/WM（joint_final 里的 pi/wm）+ 新 server 跑闭环
ARM=vla TASK=DoubleDesk SEED=0 START_FRAME=0 MAX_STEPS=600 GPU=0 \
SERVER_URL=http://127.0.0.1:18081 \
SM_CKPT=$PWD/runs/s3_r1/joint_final.pt \
OUT=$PWD/runs/B_eval_r1 bash scripts/s3_collect_arm.sh
```

然后画图对比：

```bash
PYTHONPATH=. python3 scripts/plot_drift.py \
  --dump runs/B_r1/gt_DoubleDesk_f0_r0/dump/*.pt \
  --dump runs/B_r1/vla_DoubleDesk_f0_r0/dump/*.pt \
  --dump runs/B_eval_r1/dump/*.pt \
  --out runs/B_eval_r1/drift_round1.png
```

---

## 6. 门与止损（写死）

| 门 | 判据 | 不过怎么办 |
|---|---|---|
| 环境 | `Warp CUDA error` = 0，`episode_start` 出现 | 换驱动/容器，停 |
| G1 | t=0 root <1 cm；≥2/3 集不摔 | 修采集，别带病训练 |
| G0-b | 20 步关节 ≤ 1.05× base | 回 base WM |
| G4 | 无 NaN；`token_mse` ≤1.5×；`frac_out` 不贴 1 | 降 `fm_lr` / `lambda_r` |
| **G5** | `e_root` 比基线低 **且超过 seed 波动**，摔率不升 | 两轮不降 ⇒ 停 |

---

## 7. 产物清单

| 路径 | 内容 |
|---|---|
| `runs/B_r1/vla_<Task>_f<frame>_r<ep>/dump/*.pt` | 采集（六类任务；states/actions/tokens/frames/q_target） |
| `runs/B_r1/*/scored.pt` | 用训练同一 reward 补算的奖励 |
| `runs/B_r1/drift_gt_vs_vla.png` | 归因图 |
| `runs/wm_real/real_transitions_v2b.pt` | 真转移（含 B on-policy extra） |
| `runs/S1_handover/WBC_A24000_wmvel_v5.pt` | 多步刷新件 |
| `runs/s3_r1/{joint,vla}_*.pt` | 训练产物（joint=WBC+WM，vla=head 更新） |
| `runs/s3_r1/eval_curve.jsonl` | 想象评测曲线（只作 canary） |
| `runs/B_eval_r1/drift_round1.png` | 第 1 轮闭环验收图 |

---

## 8. 别踩的坑（都已修，别改回去）

1. **plant**：务必 HA dex3（`s3_collect_arm.sh` 已锁）；用 trackerlab 29dof 资产会掉 0.062 NR。
2. **起点**：`SM_INIT_FROM_LIB=1` + 同一个 `SM_START_FRAME`；没有它就是 env 默认姿态，两条臂不可比。
3. **手**：采集 `SM_HAND=0`；手指锁 0 是训练条件。
4. **`--v3`**：训练必须 `V3=1`，否则 `TRUST_CENTER/ACT_EPS` 空转。
5. **`--critic`**：trainer 默认**开**；启动器默认传 `--no-critic`（别删）。
6. **WM 动作语义**：`q_target − default_q`；不要用 dump 的 `a_norm` 混。
7. **B dump 的 `states` 是 asset(HA_V31) 序**：打分/转移脚本已自动转 standard；自己写脚本时别忘了。
8. **`server --head-ckpt` 要 `vla_*.pt`**（含 `fm` 状态字典），不是 `joint_*.pt`。
9. **`--traj-frac` 别开太大**（0.2）：单集 B 会过拟合，把录制域刷坏。
10. **`pkill -f` 别写全名**：用 `[h]a_vla_latent64_server`，否则会杀掉自己的 shell。
11. **显存 OOM**：`JOBS > NUM_GPUS`（多个 Isaac 挤一张卡）+ server 同卡 ⇒ 日志停在
    `provider 对齐 PASS`、没有 `episode_start`、`dump/` 为空、进程静默消失。修法：
    `JOBS <= NUM_GPUS`（一卡一个），或把 server 挪到另一张卡；跑起来 1 分钟后 `nvidia-smi` 复核。
