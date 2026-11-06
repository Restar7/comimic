# S3 当前状态 / 问题 / 交接命令（2026-10-28 17:30）

> 本文替代口头交接：**现状 → 当前问题（按优先级）→ 解决方案 → 逐条命令 → 产物**。
> 配套：`docs/S3_E2E_PLAN_V4_20261028.md`（方案/O1–O16）、`docs/HANDOVER_4090_COMMANDS_20261028.md`（采集命令）、`docs/RUNBOOK_S3_HANDOFF_20261028.md`（门与止损）。

---

## 0. 一句话

**采集侧口径已全部统一并验证；公平对照的真实差距是 2.4×（GT 0.082 m / VLA 0.200 m，不是之前的 10×）。
B 数据、真转移、WM v5 都已就绪；卡在"联合微调环境缺 `av`/`skimage` + 存盘频率太稀，跑 58 步就没了、无 ckpt"。**

---

## 1. 已验证的东西（可直接用）

### 1.1 采集口径（逐条在 log 里核对过）

| 项 | 值 | 证据 |
|---|---|---|
| plant | **HA dex3**（= A24000 的 `robot_spec.usd`） | `[closedloop] usd=...g1_29dof_with_dex3...` |
| 起点 | 库行 + root/关节/qd 一起写 | `init from library: row=...` |
| 随机起点 | 按任务随机 clip + 随机帧 | `random start -> row=...` |
| reset 扰动 | xy 0.10 / yaw 0.25 / q 0.05 / qd 0.30 | `RESET NOISE seed=...` |
| 手 | **关**（训练时手指锁 0） | 无 `hand passthrough ON` |
| WBC | A24000 | `policy loaded from ...A24000...` |
| 重规划 / 录像 | 20 步 / 关 | `HA_VLA_EXEC_STEPS=20`、`record_video_every_n=0` |
| 参考库 | `ha_cmd`（命令流） | `frames_source=library_row` |

### 1.2 关键数字：公平对照（DoubleDesk，同一行 0、同扰动、同 plant、同手，300 步）

| 臂 | root 均值 | p50 / p90 | joint | 奖励 r | 摔 |
|---|---|---|---|---|---|
| **GT online** | **0.082 m** | 0.071 / 0.134 | 0.243 | 0.09405 | 否 |
| **VLA π₀** | **0.200 m** | 0.135 / 0.461 | 0.257 | 0.07876 | 否 |

⇒ **真实差距 2.4×（root）/ 6%（joint）/ −16%（奖励）**。
之前"2 m vs 0.19 m、10×"是旧口径（起点/手/plant 不一致）造成的假差距，**已作废**。

### 1.3 六类任务（随机起点，各 300 步）

| 任务 | root 均值 | joint | tilt max | 摔 |
|---|---|---|---|---|
| OpenDoor | **0.108** | 0.321 | 23.9° | 否 |
| Boxing | 0.256 | 0.326 | 16.3° | 否 |
| PPBox | 0.353 | 0.343 | 19.9° | 否 |
| VisionNavi | 0.452 | 0.591 | 88.5° | **是（110 步）** |
| DoubleDesk | 0.721 | 0.438 | 18.5° | 否 |
| SitSofa | **0.880** | 0.277 | 22.2° | 否 |

⇒ 5/6 不摔；**随机起点比行 0 难得多**（同任务 0.721 vs 0.200）。

### 1.4 数据与模型产物

| 产物 | 路径 | 说明 |
|---|---|---|
| B 数据 | `runs/B_r1/vla_*/dump/*.pt`（**26 条**） | 6 任务×3 集×600 步 + 早期 6×300 + A/B 2 条 |
| 公平对照 dump | `runs/B_r1/{gt,vla}_DoubleDesk_f0_r0/` | 同口径 A/B |
| 真转移 | `runs/wm_real/real_transitions_v2b.pt` | 含 B on-policy extra |
| WM v5 | `runs/S1_handover/WBC_A24000_wmvel_v5.pt` | 多步刷新，早停第 100 步 |

---

## 2. 当前问题（按优先级）与解决方案

### P0-1｜训练环境缺 `av` / `skimage`（**直接阻塞训练**）
- **现象**：两次训练尝试失败，失败目录名就是异常：`runs/_failed_s3_r1_av`、`runs/_failed_s3_r1_skimage`；
  本机 `python3 -c "import av, skimage"` → 两个都 `ModuleNotFoundError`。
- **根因**：`diff_vla_wbc/g1_tasks.py:142` 用 `import av` 解码视频帧（VLA 的图像输入）；
  GR00T processor 的图像处理链还需要 `skimage`。
- **方案**：在**训练用的那个 python** 里装：
  ```bash
  pip install av scikit-image
  python3 -c "import av, skimage; print('deps OK')"
  ```
- **判据**：上面那条打印 `deps OK`；训练能过 step 1 不报 ImportError。
- **成本**：5 min。**阻塞？** 是。

### P0-2｜训练跑到 step 58 就停，且**没存 ckpt**
- **现象**：`runs/s3_r1/train_metrics.jsonl` 只有 58 行；`runs/s3_r1/` 里**没有 `joint_*`/`vla_*`**；
  `eval_curve.jsonl` 只有 step 0；`logs/s3_r1.log` 停在 step 50（`tee` 被 SIGPIPE 弄死，日志冻住）。
- **根因**：① `save_every=100`，58 步 < 100 ⇒ 没有 ckpt；② 训练进程在 16:45 停掉（很可能还是 P0-1 的依赖问题在某个 batch 触发）。
- **方案**：
  1. 先解决 P0-1；
  2. **`SAVE_EVERY=50 EVAL_EVERY=50`**，保证 100 步内至少有一个 ckpt；
  3. 启动用 `setsid nohup ... > log 2>&1 < /dev/null &`，**不要 `| tee`**（tee 死会带走训练）。
- **判据**：`runs/s3_r1/joint_0050.pt` 与 `vla_0050.pt` 存在；log 持续增长到 300 步。
- **成本**：配置 1 行。**阻塞？** 是。

### P1-1｜WM v5 还没过"多步 + episode 级留出"的门
- **现象**：v5 的早停是按**同一集的时间尾段**（val20 0.2814 → 0.2447，−13%），不是另一条 episode；
  而且刷新时的 `--traj-dump 'runs/B_r1/vla_*'` **把 r0/r1/r2 都喂进去了**，没有干净留出。
- **影响**：v5 可能是"见过这一集"的假提升；训练是用它当裁判的。
- **重要**：v5 的 `--traj-dump 'runs/B_r1/vla_*'` **把 r0/r1/r2 全喂进去了**，
  所以拿 r2 评 v5 是**泄漏**的，不能当门。
- **干净做法（推荐，5–10 min）**：重刷 100–150 步，**r0 训、r1 留出**：
  ```bash
  PYTHONPATH=. SUPERMIMIC_HA_ALIGN=1 python3 scripts/finetune_wm_velocity_policy.py \
    --ckpt runs/S1_handover/WBC_A24000_wmvel_cmd.pt \
    --out  runs/S1_handover/WBC_A24000_wmvel_v6.pt \
    --packed runs/isaac_motion/ha_cmd/packed.pt --token-space label \
    --rollout-steps 3 --steps 150 --batch 256 --lr 3e-5 --vel-lambda 5 \
    --traj-dump 'runs/B_r1/vla_*_r0/dump/*.pt' --traj-frac 0.2 --traj-val-frac 0.25 \
    --val-dump 'runs/B_r1/vla_*_r1/dump/*.pt' \
    --val-horizon 20 --val-every 50 --val-max-starts 32 --device cuda:0
  ```
  然后用 **r2**（既没训也没早停）做最终门：
  ```bash
  PYTHONPATH=. SUPERMIMIC_HA_ALIGN=1 python3 scripts/eval_wm_multistep.py \
    --ckpt runs/S1_handover/WBC_A24000_wmvel_cmd.pt runs/S1_handover/WBC_A24000_wmvel_v6.pt \
    --dump 'runs/B_r1/vla_*_r2/dump/*.pt' --horizon 20 --stride 20 --device cuda:0
  ```
- **判据**：20 步关节误差 **≤ 1.05× base**；不达标就用 base。
- **成本**：10 min。**阻塞？** 阻塞第 1 轮训练的"裁判可信度"。

### P1-2｜critic 门仍未过（但数据已经够了）
- **现状**：上次（只有 1 集 B）留出 `v_ev` 最佳 +0.22（<0.30），B 留出无法判定。
- **变化**：现在有 **6 任务 × 3 集**，可以做**episode 级留出**（比如每个任务留 r2）。
- **方案**：
  ```bash
  PYTHONPATH=. python3 scripts/fit_critic_real.py \
    --transitions runs/wm_real/real_transitions_v2b.pt \
    --out runs/wm_real/critic_real_v2.pt --center mean --use-diff --epochs 3
  ```
  判据：**B 留出 `v_ev > 0.30`** 才启用 `CRITIC_ON=1`；否则保持关。
- **成本**：10 min。**阻塞？** 否（第 1 轮可关）。

### P1-3｜O1（ORACLE 三臂）还没跑 —— **决定"值不值得继续投"**
- **现象**：token 有没有权限把漂移压到 GT 以下，从未验证；旧的三臂数据无效（npy 模式 frames 是步号）。
- **方案**：`GT / ORACLE / VLA` 各 2000 步 online 闭环，同 harness 同起点，比 `e_root(t)` 与摔率。
- **判据**：ORACLE 的 `e_root` 显著低于 GT ⇒ 继续；否则止损。
- **成本**：1 h（要 Isaac）。**阻塞？** 不阻塞第 1 轮，但阻塞"要不要投第 2 轮"。

### P2-1｜奖励在大漂移处饱和（O16）
- **现象**：root 奖励核 `exp(-e/0.3²)`：0.8 m → 8e-4、1.2 m → 1e-7、2 m → 5e-20（**梯度为 0**）。
- **影响**：600 步的 B 集后半段是"死区"；训练起点若落在那里等于白训。
- **方案**：
  1. **起点过滤**：只用 `e_root < 0.5 m` 的 B 重规划时刻（给 `b_start.py` 加过滤，~10 行）；
  2. 或训练用 `--reward-mode linear`（大误差仍有梯度，代价是奖励无界、需重标定）。
- **判据**：训练日志里 `rollout/reward` 不再被死区拉平；`frac_out` 不贴 1。
- **成本**：10 行 / 1 个开关。**阻塞？** 否，但影响学习信号。

### P2-2｜`--head-ckpt` 还没被真实 head 验证过
- **现状**：server 的 `--head-ckpt` 已实现（加载 `vla_*.pt` 的 `fm`、自动挂 LoRA、unexpected≠0 报错），但还没有 ckpt 试过。
- **方案**：拿到第一个 `vla_0050.pt` 后立刻做一次 server 自检：
  ```bash
  HEAD_CKPT=runs/s3_r1/vla_0050.pt PORT=18081 bash scripts/start_vla_server.sh
  ```
  判据：日志出现 `[server][head] loaded ... unexpected=0`，且同一输入 token 与离线推理一致。
- **成本**：5 min。**阻塞？** 阻塞闭环验收。

### P2-3｜闭环验收基线要统一
- **现状**：新的行 0 基线是 **GT 0.082 / VLA 0.200 m**（300 步）。
- **方案**：验收也用 **同 row 0、同扰动、同 plant、300 步**；主指标 `e_root` 均值 + 前 100 步斜率 + 摔率。
- **判据**：VLA π₁ 的 `e_root` 从 0.200 降到 **<0.15**（且不摔）。

---

## 3. 交接命令（按顺序执行）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
export COMIMIC=$PWD

# ---------- 0) 环境自检 ----------
python3 -c "import av, skimage; print('deps OK')"     # 不 OK 就 pip install av scikit-image
nvidia-smi --query-gpu=index,driver_version,memory.used --format=csv,noheader

# ---------- 1) WM 干净留出：r0 训、r1 早停、r2 最终门 ----------
PYTHONPATH=. SUPERMIMIC_HA_ALIGN=1 python3 scripts/finetune_wm_velocity_policy.py \
  --ckpt runs/S1_handover/WBC_A24000_wmvel_cmd.pt \
  --out  runs/S1_handover/WBC_A24000_wmvel_v6.pt \
  --packed runs/isaac_motion/ha_cmd/packed.pt --token-space label \
  --rollout-steps 3 --steps 150 --batch 256 --lr 3e-5 --vel-lambda 5 \
  --traj-dump 'runs/B_r1/vla_*_r0/dump/*.pt' --traj-frac 0.2 --traj-val-frac 0.25 \
  --val-dump 'runs/B_r1/vla_*_r1/dump/*.pt' \
  --val-horizon 20 --val-every 50 --val-max-starts 32 --device cuda:0

PYTHONPATH=. SUPERMIMIC_HA_ALIGN=1 python3 scripts/eval_wm_multistep.py \
  --ckpt runs/S1_handover/WBC_A24000_wmvel_cmd.pt runs/S1_handover/WBC_A24000_wmvel_v6.pt \
  --dump 'runs/B_r1/vla_*_r2/dump/*.pt' --horizon 20 --stride 20 --device cuda:0
# 判据：20 步关节误差 ≤ 1.05× base，否则用 base；通过后把下面命令的 v5 换成 v6

# ---------- 2) critic 重评（有 3 集/任务了）----------
PYTHONPATH=. python3 scripts/fit_critic_real.py \
  --transitions runs/wm_real/real_transitions_v2b.pt \
  --out runs/wm_real/critic_real_v2.pt --center mean --use-diff --epochs 3
# 判据：B 留出 v_ev > 0.30 才开 critic

# ---------- 3) 联合微调（300 步、每 50 存、setsid 防被杀）----------
setsid nohup env V3=1 TRUST_CENTER=prev \
  START_FROM_B='runs/B_r1/vla_*_f*_r*/dump/*.pt' \
  ACT_EPS=0.6 ACT_WEIGHT=1.0 LAMBDA_R=0.3 LAMBDA_BC=0 \
  RUN=s3_r1b BATCH=4 ACCUM=8 H=20 K=4 STEPS=300 EVAL_EVERY=50 EVAL_BATCHES=8 SAVE_EVERY=50 \
  WBC_CKPT=$PWD/runs/S1_handover/WBC_A24000_wmvel_v5.pt \
  EXTRA="--amp bf16 --lora-rank 32 --fm-lr 1e-4 --no-dr --vla-state-from-burnin \
         --no-critic --act-hinge-cap 2 --lambda-hand 0.2" \
  bash scripts/launch_joint_ha_v2.sh > logs/s3_r1b.log 2>&1 < /dev/null & disown
# 看：每 50 步出 joint_/vla_ ckpt；无 NaN；token_mse ≤1.5×；frac_out 不贴 1

# ---------- 4) server 换 head 自检 ----------
HEAD_CKPT=$PWD/runs/s3_r1b/vla_0050.pt PORT=18081 GPU=0 \
  bash scripts/start_vla_server.sh > logs/server_r1b.log 2>&1 < /dev/null & disown
sleep 60; grep -E "\[server\]\[head\]|listening" logs/server_r1b.log

# ---------- 5) 闭环验收（同 row 0 基线）----------
setsid nohup env ARM=vla TASK=DoubleDesk SEED=0 START_FRAME=0 MAX_STEPS=300 GPU=1 \
  SERVER_URL=http://127.0.0.1:18081 \
  SM_CKPT=$PWD/runs/s3_r1b/joint_0050.pt \
  OUT=$PWD/runs/B_eval_r1b bash scripts/s3_collect_arm.sh > logs/eval_r1b.log 2>&1 < /dev/null & disown

# ---------- 6) 漂移对比图 ----------
PYTHONPATH=. python3 scripts/plot_drift.py \
  --dump runs/B_r1/gt_DoubleDesk_f0_r0/dump/*.pt \
  --dump runs/B_r1/vla_DoubleDesk_f0_r0/dump/*.pt \
  --dump runs/B_eval_r1b/dump/*.pt \
  --out runs/B_eval_r1b/drift_r1b.png --title "GT / pi0 / pi1 @ row0"
# 判据：pi1 的 e_root <0.15（π0 是 0.200），且不摔
# 早期用 joint_0050.pt；训练跑完后换成 joint_final.pt 再验一次
```

---

## 4. 产物与目录约定

| 路径 | 内容 |
|---|---|
| `runs/B_r1/` | 采集（26 dump）+ A/B 基线 + scored |
| `runs/B_r2/` | （预留）下一轮采集；命令见 HANDOVER_4090 §3.1 |
| `runs/wm_real/real_transitions_v2b.pt` | 真转移（含 B on-policy） |
| `runs/S1_handover/WBC_A24000_wmvel_v5.pt` | WM 刷新件（待过门） |
| `runs/s3_r1b/` | 训练产物（joint=WBC+WM，vla=head 更新） |
| `runs/B_eval_r1b/` | 闭环验收 + 漂移图 |

---

## 5. 新踩的坑（已写进本文，别再踩）

1. **`| tee` 会带走训练**：tee 死于 SIGPIPE 后 log 冻住、训练也可能跟着断。用 `setsid nohup ... > log 2>&1 & disown`。
2. **`save_every` 要 ≤ 步数/4**：300 步配 100 看似合理，但进程一挂就什么都没有；用 50。
3. **`av` / `skimage` 缺失**：训练/评测的图像链需要，装进**对应 python**。
4. **WM 刷新的留出必须"没参与训练"的 episode**：`--traj-dump` 不要把留出集喂进去。
5. **起点过滤**：`e_root > 0.5 m` 的 B 起点在奖励死区，别用（或改 `--reward-mode linear`）。
6. **采集 `JOBS ≤ NUM_GPUS`**；server 占 ~10 GB；一卡一个 Isaac。
