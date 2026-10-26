# 方案：用 HA 微调 WBC（ckpt_010500_physint）→ 再做联合微调（2026-10-25 评估）

> 用户提的方案：拿 HA 数据集微调当前 `ckpt_010500_physint.pt`，训好之后再做联合微调。
> 本文回答：**能不能做 / 值不值得做 / 按什么约束做 / 先做什么才不会白烧卡**。

---

## 0. 结论（三句话）

1. **技术上完全可行，而且数据事实比原先设想的更支持它**：现 WBC **从没见过 HA 动作**
   （只在 BONES-SEED 上训过，`runs/OFFICIAL_SUPERMIMIC.json` 明确更正过"run 名带 `_ha_`"的误导），
   而 HA 的 699 条里有 **636 条（91%）比 10 s 长**、最长 **200 s** —— 正好是旧训练回合上限（10 s）砍不到的地方。
2. **但有一个前置诊断必须先做（≈30 分钟）**：用现成的 WBC 在 HA 动作上做**零样本 rollout**，看
   `sim_r / term_frac`。如果它在 HA 上本来就跟踪得不错，那域差不是主要矛盾，**微调会白烧 2–4 h**。
3. **如果做，四条硬约束不能破**（否则重演已知失败）：**WM 冻结 / 端口 token / `a=Δq`+spec reward /
   微调后重做 800 步速度标定**。并且换底模 = A1/A2/A4/A5/anchor 全部表作废，要成批重跑（≈10 h）。

---

## 1. 数据事实（这是支持该方案的核心理由）

| 域 | clips | 帧数 | 时长 | clip 长度 median | p90 | max | >10 s 的 clip |
|---|---|---|---|---|---|---|---|
| **BONES-SEED**（现 WBC 训练域） | 129,943 | 47,475,680 | 263.75 h | 288 帧 = **5.76 s** | 628 帧 | 8,979 帧 = 179.58 s | 21,714（16.7%） |
| **HA**（部署域，未训过） | **699** | 634,016 | **3.52 h** | **789 帧 = 15.78 s** | 1,431 帧 = 28.6 s | **10,000 帧 = 200 s** | **636（91.0%）** |

- HA 的 Isaac 打包库现成可用：`runs/isaac_motion/ha_tokens/packed.pt`（699 clips / 634,016 帧，
  与 BONES-SEED 同一个 compact 布局 `ref/names/starts/lengths`，**能直接喂 WBC runner 的 `--motion_packed`**）。
- 端口 token 侧车 699 条现成：`data/port_tokens/ha_tokens/*.npy`（**必须用这个空间，不能用 HA npz 标签 token**）。
- ⇒ 两个缺口同时存在：**① 域差**（BONES-SEED 的 mocap locomotion vs HA 的任务动作）；
  **② 长时程**（HA 中位 15.8 s，而旧训练回合最多 10 s ⇒ 就算训 HA，用 500 也不会覆盖）。

---

## 2. 四条硬约束（破一条就白跑）

| # | 约束 | 为什么（项目里已经踩过） |
|---|---|---|
| 1 | **WM 必须冻结**（不得把 WM 放回 sim 转移上更新） | 800 步速度标定是在"参考窗口 + policy 自己的动作"上做的；用 PhysX 一步转移续训 300 iter 会把标定洗掉：`qd_corr 0.897 → 0.095`、`ratio 0.93 → 4.27`、`state 0.555 → 3.575` |
| 2 | **token 用端口空间**（`data/port_tokens/ha_tokens` 或 runner 的 `--interface token` 在线编码） | HA npz 的 `g1_motion_token_64` 与 WBC 端口 token **不是同一空间**（`max|Δ| 1.06–1.19`、corr 0.06–0.19）；直接喂 = 给 policy 随机码（实测 `sim_r 0.52→0.15`、`term 0.17→0.55`） |
| 3 | **动作 `a=Δq`、奖励用 spec scale**，且与联合侧 `COMIMIC_ACTION_SCALE=one` 一致 | 旧代码多除一次 per-joint scale，参考跟踪 reward `+0.069 → −0.129` |
| 4 | **policy 变了之后必须重做 800 步速度标定**，产出**新文件名**（如 `ckpt_010500_physint_ha.pt`） | 标定是对"该 policy 产生的动作分布"做的；不重标 = 拿旧标定配新 policy。**绝不覆盖 `ckpt_010500_physint.pt`** |

---

## 3. 先做这个：30 分钟零样本探针（决定要不要微调）

用**现有** `ckpt_010500_physint`（不动权重）在 **HA 动作**上跑 30 个 iter 的纯 rollout：
`--wm_only_iters` 相对 start_it = 500 ⇒ `iterations = 10500 + 30` 时 **policy 根本不更新**，
`--ckpt_every` 设大 ⇒ 不落盘、不污染任何文件。

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic
export SONIC_ENCODER_ONNX=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx
MODE=train NPROC=1 ENVS_PER_RANK=2048 ITERATIONS=10530 \
MAX_EPISODE_STEPS=0 \
LIBRARY=$PWD/runs_ha_probe_placeholder   # 见下：LIBRARY 指向 HA 打包库
```
> 实际 `LIBRARY` 用 **`/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/runs/isaac_motion/ha_tokens/packed.pt`**
> （脚本内部会 `cd $REPO`，所以建议给绝对路径），`RESUME=` 指向
> `.../trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt`，
> `INTERFACE=token COMPILE=0 HA_ALIGN=1 CKPT_EVERY=100000 PORT_BASE=29690 LOG_DIR=logs/ha_zeroshot_probe`

**判读门槛**（对着 BONES-SEED 上的参考：`sim_r ≈0.877`、`term_frac ≈0.49%`）：

| 零样本结果（HA 动作） | 结论 | 下一步 |
|---|---|---|
| `sim_r ≥ 0.80` 且 `term_frac ≤ 1%` | HA 上本来就跟踪得住，**域差不是主要矛盾** | **不要微调**；去修 §4 的 P0（reset 姿态 / obs99） |
| `sim_r 0.6–0.8` 或 `term_frac 1–5%` | 有域差但不致命 | 先修 P0，再考虑微调 |
| `sim_r < 0.6` 或 `term_frac > 5%` | **域差确认**（WBC 不认识 HA 动作） | **值得微调**，按 §5 做 |

---

## 4. 但先记住：闭环摔倒的主因很可能不在 WBC 域差

`docs/WBC_TRAIN_4GPU_BONES_SEED.md` §6 的 P0 两项至今未修：

1. 闭环 reset 用 Arena 初始姿态（TWIST2 −0.2/0.4/−0.2），训练是参考帧 reset + HA 默认姿态（−0.312/0.669/−0.363）；
2. obs99 末尾 6 维（参考朝向）部署侧是常量 `SM_REF6D=zeros`，训练侧是 `mat_to_6d(Rᵀ·R_ref)`。

### 4.1 实测复核（flowbody128 上的闭环日志，2026-10-25 09:39–11:42）

用户指出"obs99 / PD 这些已在 flowbody128 上解决过，但仍不行"。**核对日志后的确切事实**：

| 时间 | WBC | VLA | PPBox seed0 存活步数 | 结果 |
|---|---|---|---|---|
| 10:06 | `ckpt_008500`（BONES-SEED，未标定） | ft2000 | **74** | fall |
| 11:29 | `ckpt_008500` + **p0fix** | ft2000 | **57** | fall |
| 11:32 | `ckpt_008500` + **p0fix2** | ft2000 | **76** | fall |
| 11:42 | **`ckpt_010500_physint`** + p0fix | **ft4000** | **278** | fall（差 22 步到 timeout） |
| 09:39 | **`ckpt_010500_physint`** + p0fix | **anchor4000** | **291** | fall（差 9 步） |

**诚实结论**：
- **P0 修复本身没有带来提升**（ft2000 同组合：74 → 57/76，同量级甚至更差）；
- 真正把存活从 ~75 拉到 ~290 的是 **换成 `ckpt_010500_physint`（显式积分 + 速度标定）** 与 **更好的 VLA（ft4000/anchor4000）**；
- 也就是说：**"执行器的动作保真度（尤其速度/接触）"才是主导因素** —— 这恰好是 HA 微调能改善的方向，但**也说明"域差"这个说法目前仍是推断**。

### 4.2 "域差很大"是推断，不是实测（必须承认这一点）

我查遍了 `logs/` 与 `docs/`：**没有任何一次把 HA 库（`runs/isaac_motion/ha_tokens/packed.pt`）喂给 WBC runner**
的记录（`grep` 命中的都是联合训练用的 `--motion-packed-dir`）。现有的"域差"证据全是**间接**的：

1. 用**数据集真值 token** 也摔（joint-order 修前 51 步、修后 165 步）⇒ 执行器跟不上 HA 动作；
2. P0 修 + physint WBC 后最长 **291/300**，仍 `fall`；
3. 同一个 WBC 在**自己的训练域**里 `term_frac 0.49%`、`sim_r 0.877`（站得住）。

⇒ **30 分钟探针（§3）就是把这件事从"推断"变成"数字"的最小代价**，别跳过。

### 4.3 一个被忽略的 obs 域差（比"没训过 HA"更便宜可修）

逐维对比训练 `obs99`（`tracker_env/mdp/supermimic.py:154-164`）与部署侧（`action_provider_supermimic.py:_obs_frame`）：

| 维度 | 训练 | 部署 | 是否一致 |
|---|---|---|---|
| 0–28 | `q − default_q − dq0`（`dq0` 是 ±0.01 rad 的域随机） | `q − default_body` | ✅ 基本一致 |
| 29–57 | `qd` | `joint_vel` | ✅ |
| 58–60 | `Rᵀw` | `root_ang_vel_b` | ✅ |
| 61–63 | `−R[...,2,:]` | `projected_gravity_b` | ✅ |
| 64–92 | `a_prev` | `last_action` | ✅ |
| **93–98** | **`mat_to_6d(Rᵀ·R_ref[f])`（每帧真实参考朝向）** | **常量 `SM_REF6D`（zeros 或 identity）** | ❌ **唯一一处硬不一致** |

而这个不一致**实测就值 2–3 倍生存步数**（`docs/HANDOVER_ISAAC_CLOSEDLOOP.md` §10.13：zeros 114/124 vs identity 48/47）。

⇒ 由此得到两个**比"HA 微调"更便宜的候选**：
- **A. obs 对齐训练**：训练时把最后 6 维随机置零/置为单位阵（data augmentation），让 policy 对"部署侧假参考"鲁棒 —— 改 1 处、重训 500–1000 iter 即可验证；
- **B. 部署侧重建参考**：用 token 推一个参考朝向（或直接用上一帧目标），让 6 维有真值 —— 改 provider，不重训。

**最佳组合**：HA 微调时**同时**加 obs 对齐增强（HA 数据 + ref6d 随机化 + WM 冻结 + 只训 policy），
一次把"没见过的动作"和"部署侧的假参考"两个缺口都补掉。

---

## 5. 若决定微调：执行方案（成本 ≈2–4 h/次）

| 项 | 取值 | 理由 |
|---|---|---|
| 起点 | `ckpt_010500_physint.pt`（`RESUME=`） | 显式积分 + 已标定的 WM/policy |
| 库 | `runs/isaac_motion/ha_tokens/packed.pt`（699 / 634,016） | 部署域 |
| `MAX_EPISODE_STEPS` | **0**（→ 2^31-1） | HA 中位 15.8 s、最长 200 s；用 500 会砍掉 91% 的 clip |
| 解冻范围 | **只 policy**（WM 不更新、encoder 不用） | 保 WM 标定；encoder 在 token 路由本来不用 |
| lr / schedule | `lr 2e-5` 起步、500–1,500 iter、`ckpt_every 100` | HA 只有 3.5 h（BONES-SEED 的 1/75）→ 长训必过拟合/遗忘通用 locomotion |
| 环境数 | 8192/rank（2 卡则 2×8192） | HA 只有 699 clips，不必 4 卡 |
| 产物 | **`logs/wbc_ha_ft/`**，最终 `ckpt_*_physint_ha.pt` | 不覆盖任何现有 ckpt |
| 微调后 | **重做 800 步速度标定**（`scripts/finetune_wm_velocity_policy.py`）→ `ckpt_010500_physint_ha.pt` | 约束 #4 |
| 验收 | ① HA 上 `sim_r` 明显↑、`term_frac`↓；② **同时**在 BONES-SEED 子集上不掉（防遗忘）；③ `eval_wm_velocity.py` 的 `qd_corr/ratio` 仍 ≥0.74/≈0.94 | 只看 HA 会掩盖遗忘 |

**之后再做联合微调**：把 `--supermimic-ckpt` 换成新标定件，其余 recipe 不变（anchor 那套：端口 token / `a=Δq` / WM 冻结 / λ=1.0）。
⚠️ 这样 A1/A2/A4/A5/anchor 的旧表全部作废（文档硬规则：一次对比只能有一个 WM/一个底模），
**要么整批重跑（≈10 h），要么在论文里明确分成"底模 v1 / v2"两批，不混表。**

---

## 6. 建议的执行顺序（省时间版）

1. **零样本 HA 探针**（≈10–30 min，1 卡）→ 看 §3 门槛；
2. **A1 闭环 base vs ft**（≈1.5 h，1 卡；ft 用 `runs/ckpt_vla_ha_anchor4000`）→ 这是唯一独立指标，至今没跑完；
3. （原计划）若 ① 域差大 且 ② 摔不是 P0 引起 → 才做 HA 微调；之后重标定 → 重跑联合 + 消融批次。
   > **2026-10-25 14:24 更新**：第 1 条已经用"warm start 在 HA 上跑 50 iter"的形式**直接量到**——
   > `term_frac 0.58` vs 自域 `0.005`（约 100 倍）、`sim_r 0.449` vs `0.877`。**域差已确认**，
   > 于是按用户决定直接进了 obs93 + 冻结 WM 的 HA 微调（见 §7）。

---

## 7. 执行记录：obs93 + 冻结 WM 的 HA 微调（2026-10-25 14:24 起，4 卡）

用户决定：**不用 ref6d 那 6 维**，直接删掉 → 用 obs93 重新训 SuperMimic，在 HA 数据上。

### 7.1 代码改动（3 处，都已完成）

| 文件 | 改动 |
|---|---|
| `trackerlab-supermimic/scripts/convert_obs99_to_obs93.py`（新） | 手术式转换：`dec.0.weight (2048,1054) → (2048,994)`（每帧删最后 6 列 ×10），`obs_norm.mean/var → 93`，`args.obs_mode="paper_obs93"`；**自检：WM 20 个张量逐张量不变** |
| `scripts/rls/supermimic/train.py` | 新增 `--freeze_wm` |
| `runner.py` | `--freeze_wm` 时：① 跳过 `wm_update()`（参数+优化器不动）；② 跳过 `wm.in_norm/out_norm` 的 RunningNorm 更新 |
| `scripts/rls/supermimic/train_bones_seed.sh` | 新增透传 `OBS_MODE / LR / FREEZE_WM / WM_ONLY_ITERS` |

产物：`trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint_obs93.pt`
（`iteration=10500`、`wm_integration=explicit_dt0.02`、`obs_mode=paper_obs93` 均保留/正确）。

### 7.2 起跑（4 卡，HA 数据，只训 policy）

```bash
cd trackerlab-supermimic
export SONIC_ENCODER_ONNX=.../gear_sonic_deploy/policy/release/model_encoder.onnx
RESUME=$PWD/logs/bones_seed_token_ha_v2/ckpt_010500_physint_obs93.pt \
OBS_MODE=paper_obs93 FREEZE_WM=1 WM_ONLY_ITERS=0 LR=2e-5 \
MODE=train NPROC=4 ENVS_PER_RANK=8192 ITERATIONS=11300 \
MAX_EPISODE_STEPS=0 \
LIBRARY=$COMIMIC/runs/isaac_motion/ha_tokens/packed.pt \
INTERFACE=token COMPILE=1 HA_ALIGN=1 CKPT_EVERY=100 PORT_BASE=29660 \
LOG_DIR=logs/wbc_ha_obs93 \
setsid nohup bash scripts/rls/supermimic/train_bones_seed.sh > logs/wbc_ha_obs93.launcher.log 2>&1 &
```

起跑核对（`args.json` + 日志）：

| 项 | 实测 |
|---|---|
| `obs_mode` | **paper_obs93** ✅ |
| `freeze_wm` | **True** ✅ |
| `max_episode_steps` | **2147483647**（全 clip）✅ |
| HA 库是否真的加载 | **每 rank 175 clips / 154,092 帧**（4×175≈699）✅ 不是 BONES-SEED |
| warm start | 日志 `Training iterations 10501..11300; policy updates start after 10500` ✅ |
| WM 冻结是否生效 | 首个日志点 `wm_loss = 0, wm_err8 = 0` ✅ |

### 7.3 **决定性数字：域差被直接量到了（约 100 倍）**

同一个 policy（`ckpt_010500_physint`，warm start 起点）在两个域上的实测：

| 域 | `term_frac`（摔倒占完成回合） | `sim_r` | `ep_len` |
|---|---|---|---|
| **BONES-SEED**（自己训过的域） | **0.49%** | **0.877** | 216.6 |
| **HA**（部署域，本次 it 10550 读数） | **58.0%** | **0.449** | 85.7 |

⇒ **"域差很大"从推断变成了实测**：同一个执行器在 HA 动作上 **58% 的回合以摔倒结束**，
`sim_r` 几乎腰斩。**HA 微调不是可选项，是必须的** —— 这一条同时解释了闭环为什么在 ~290 步时倒。

### 7.4 本轮要看的验收信号（**附一条重要修正**）

> ⚠️ **run 内的 `term_frac` / `sim_r` 是非平稳的**：`sample_frames` 的起点分布含
> `0.4 · softmax(-25·bin_returns)`，而 `bin_returns` 由"实际回合回报"在线更新 ⇒
> policy 差 → 差 bin 被加权 → 采到更难的起点 → `term_frac` 更高。这是**反馈回路**，
> 所以**不能**拿 run 内的 `0.58 → 0.88` 直接断言"策略变差"（`imag_rew` 同期只掉 5%）。
> 正确做法：**每个候选 ckpt 各跑一次"fresh 50-iter 探针"**（新起进程 ⇒ `bin_returns` 回到均匀，
> 与 baseline 的 0.58/0.449 同口径），或直接看闭环。

| # | 信号 | 口径 |
|---|---|---|
| 1 | **fresh 探针的 `term_frac`**（新进程头 50 iter） | 与 baseline **0.580 / 0.449** 同口径比较 ← **主判据** |
| 2 | `wm_loss` 恒为 **0** | 证明冻结没破（已确认） |
| 3 | `imag_rew`（回放缓冲里的想象奖励） | 比 `sim_r` 平稳，可作为"策略有没有真变差"的旁证 |
| 4 | 每 100 iter 的 ckpt（`ckpt_01100 … ckpt_01300`） | 每个都可做 fresh 探针 |
| 5 | 闭环 PPBox seed0 | 对比现基线 **278/291 步** ← 最终判据 |

**当前读数（14:32，it 10550/10600/10650）**：`sim_r 0.449 → 0.266 → 0.265`、`term_frac 0.580 → 0.779 → 0.883`、
`imag_rew 0.391 → 0.348 → 0.372`、`pi_gradnorm 18.8 → 15.1 → 9.1`、`wm_loss` 恒 0、无 skip/NaN。
按上面的口径，这**还不能**判定策略变差（imag_rew 只掉 5%），要等 fresh 探针。


### 7.5 ⚠️ 关键负面结果：**在冻结 WM 上做 policy RL = 模型利用（model exploitation）**

两版都跑过、两版都退化，**已全部停止**（GPUs 14:49 释放）：

| 版本 | 配置 | 结果（`sim_r` 是每窗 400 万步，不是小样本噪声） |
|---|---|---|
| v1（`logs/wbc_ha_obs93`） | lr 2e-5 + critic 参与 bootstrap | `sim_r 0.449 → −0.153`（600 iter 单调塌）、`term_frac 0.58 → 0.88` |
| v2（`logs/wbc_ha_obs93_v2`） | **`--no_critic`** + lr 1e-5 | `sim_r 0.527 → 0.404 → 0.367 → 0.322 → 0.278 → 0.236 → 0.183 → **0.116**`（400 iter 单调塌）、`ep_len 86 → **32**`（0.6 s 就倒） |

**决定性证据是"方向相反"**：同一时间段里
- 真实仿真：`sim_r` ↓、`ep_len` ↓、`term_frac` ↑
- **世界模型里的想象奖励：`imag_rew 0.465 → 0.588` ↑**

⇒ 策略在**把想象奖励刷高**的同时**在真实 PhysX 里越来越差** = 典型的"利用冻结 WM 的模型误差"。
`--no_critic` 只去掉了 critic bootstrap，**挡不住这件事**；lr 从 2e-5 降到 1e-5 也只是让它慢一点。

**为什么必然**：`docs/HANDOVER_MAINLINE_AND_ABLATIONS_20261025.md` §4.1 已经记录过反向版本
（"WM 在 sim 转移上更新会破坏 policy↔WM 一致性"）。这里是同一件事的另一半：
**policy 动了、WM 不动，一致性同样被破坏**（800 步标定只对"当时的动作分布"有效）。

⇒ **结论：`HA 数据 + obs93 + 冻结 WM + policy RL` 这条路不成立，已废弃。**
两条正确路线（待选）：

| 路线 | 做法 | 优点 | 代价 |
|---|---|---|---|
| **A（推荐）BC 监督适配** | 不复用 WM/reward，直接让 policy 回归参考关节目标（`a_ref = (q_ref − default_q)/scale`），HA 窗口上做监督 | 不经 WM ⇒ 无法被利用；便宜、可复用；正是文档 §10 "路线 A：先 BC 再 PPO" | 需要写 ~60–80 行采样+训练循环 |
| **B WM 一起解冻** | 在 HA 上按 BONES-SEED 那套 policy+WM 联合训，训完**重做 800 步速度标定** | 不用写 BC；复用现成 runner | WM 会变 ⇒ 必须重标定 + 重跑消融批；仍可能"训得好但闭环仍摔" |

### 7.6 已确认的好消息：**删掉那 6 维（obs93）本身是有利的**

同一 policy、同一协议、唯一差别 = 删不删 ref6d（首个日志窗口，4×8192 envs）：

| 观测维度 | `term_frac` | `sim_r` |
|---|---|---|
| obs99（原 ckpt） | 0.580 | 0.449 |
| **obs93（手术件）** | **0.278** | **0.527** |

（注意：这个"A/B"里两边都含 ~50 iter 的策略更新，所以**不是纯净对比**。为此正在跑
`scripts/probe_ha_obs_ab.sh` 的**纯 rollout 探针**（policy/WM/critic 全部不更新，各 50 iter）：
`logs/probe_ha_ab/{obs99_none,obs93_none}` —— 那才是"删 6 维净效果"的干净数字。）

### 7.7 当前状态（14:52）

- 所有 HA 微调**已停止**；4 卡空闲。
- 保留资产：`ckpt_010500_physint_obs93.pt`（手术件，可用）、
  `logs/wbc_ha_obs93*/ckpt_*`（退化样本，**不要用**，但可作"模型利用"的证据）。
- 正在跑：`scripts/probe_ha_obs_ab.sh`（obs99 vs obs93 纯 rollout A/B，≈15 min，两轮）。

### 7.8 ✅ 纯净探针结果（policy/WM/critic 全不更新，各 50 iter，4×8192 envs）

| 观测 | `sim_r` | `ep_len` | `term_frac` |
|---|---|---|---|
| **obs99**（原 `ckpt_010500_physint`） | **+0.541** | 86.7 | **0.226** |
| **obs93**（删掉 ref6d 6 维） | **+0.536** | 86.0 | **0.254** |

⇒ **删掉那 6 维对 HA 上的表现几乎零损失（差异在噪声内）**
⇒ 同时**更正 §7.6**：那里写的"obs93 0.278 vs obs99 0.580"是被**50 iter 的坏更新**污染的，
不是 6 维的净效果。**干净数字是 0.226 vs 0.254。**
⇒ 结论：**obs93 这个决定是安全的**，可以直接用在从头训里（部署侧不再需要伪造参考）。

（`logs/probe_ha_ab/{obs99_none,obs93_none}`；脚本 `scripts/probe_ha_obs_ab.sh`，≈6 min/轮。）

### 7.9 域差的**准确**幅度（用纯净探针口径）

| 域 | `term_frac` | `sim_r` |
|---|---|---|
| BONES-SEED（训练域，参考 run @10.5k） | **0.49%** | **0.877** |
| **HA（部署域，obs99 纯净探针）** | **22.6%** | **0.541** |

⇒ 域差 = **摔倒率 46 倍**、`sim_r` 掉 38%。之前说的"约 100 倍"是拿训练首窗（含坏更新）比出来的，**已更正**。

---

## 8. 混合库续训（warm start，WM 一起训）：**第一次在 HA-only 上量到实质进步**

### 8.1 配置（15:00:47 起，4 卡）

```
warm start = ckpt_010500_physint_obs93.pt   # obs93 手术件（那 6 维已删）
库         = data/bones_seed/bones_seed_plus_ha_x20.pt   # 79.51% BONES-SEED + 20.49% HA(x20)
FREEZE_WM  = 0    # ★ WM 一起训（关键：不再重演 model exploitation）
NO_CRITIC  = 1    # warm start 时 critic 是新的，不做 bootstrap
WM_ONLY_ITERS = 200   # 先让 WM 在混合数据上适配，再动 policy
LR = 2e-5   MAX_EPISODE_STEPS = 0（全 clip）   NPROC=4 × ENVS_PER_RANK=8192
ITERATIONS 10500→14500（4000 iter，≈65 min）   CKPT_EVERY=250
```

混合库由 `scripts/merge_libraries.py` 生成（自检：`starts` 连续、帧数与 `ref` 一致）：
129,943 + 699×20 clips、60,156,000 帧、7.96 GiB；HA 占 **20.49%** 的 bin（不复制的话只有 1.24%）。

### 8.2 训练侧（混合口径，**不能**直接当 HA 指标）

it 10700（policy 开始更新）之后单调变好、**没有崩**：
`sim_r 0.761 → 0.812`、`term_frac 0.058 → 0.008–0.012`、`ep_len 235 → 331–347`、`wm_loss 0.163 → 0.083`（WM 仍在适配）。
对比昨天两版（冻结 WM + policy RL）：`sim_r` 一路塌到 −0.15 / 0.116、`imag_rew` 反而升 —— 那是 model exploitation；
这一版 `sim_r` 与 `imag_rew` **同向**，是健康信号。

### 8.3 ★ HA-only 探针：真进步（**这是本次最关键的数字**）

同一套"纯 rollout 探针"（policy/WM/critic 全不更新、`MAX_EPISODE_STEPS=0`、HA-only 库、
`logs/probe_ha_*`；`wm_only_iters` 设成远大于 iterations ⇒ 只测不训）：

| 权重 | `sim_r` | `ep_len` | `term_frac`（HA 上的摔倒率） |
|---|---|---|---|
| warm start（it 10500，`ckpt_010500_physint` obs99） | +0.541 | 86.7 | **0.226** |
| **混合续训到 it 11250（`ckpt_011250`，obs93）** | **+0.680** | 87.1 | **0.064** |
| 相对改善 | **+26%** | — | **摔倒率 −72%（3.5×）** |

⇒ **HA 域确实在被学会**：`sim_r +0.14`、摔倒率从 22.6% 降到 6.4%。而且这一版**只用 750 iter**。
（探针口径：1 rank × 4096 envs；baseline 是 4 × 8192 —— 同协议、样本量不同，期望可比，噪声略大。）

### 8.4 仍未解决 / 待做

1. `sim_r 0.680` 仍低于训练域 0.877、`term_frac 0.064` 仍是 0.005 的 13 倍 ⇒ **还有空间**，继续训到 14500 再看；
2. **速度标定已失效**（WM 被混合 sim 数据训过）⇒ 进联合训练前必须重做 800 步标定；
3. 闭环 PPBox 仍未用新 ckpt 复测（基线 278/291 步）；
4. 换底模 ⇒ A1/A2/A4/A5/anchor 的旧表要整批重跑或标成"底模 v1/v2"。

### 8.5 追加探针点（同一探针协议，HA-only）：**改善在收敛**

| 权重 | `sim_r` | `term_frac` | 相对上一行 |
|---|---|---|---|
| warm start（it 10500） | +0.541 | 0.226 | — |
| it 11250（`ckpt_011250`） | +0.680 | 0.064 | **+0.139 / −0.162** |
| **it 12250（`ckpt_012250`）** | **+0.694** | **0.046** | +0.014 / −0.018 |

⇒ 前 750 iter 拿到了绝大部分收益，之后 1000 iter 只再涨 0.014 ⇒ **在当前 lr=2e-5 / HA×20 的配方下，
HA 域正在趋向 ~0.69 / 0.046 的平台**（对比训练域 0.877 / 0.005：摔倒率仍差 **9 倍**）。
⇒ 决策含义：继续用同样配方再加 iter 收益很小；要么**提高 lr / 提高 HA 权重**再推一档，要么
**先拿这个 ckpt 去跑闭环**（闭环才是最终判据），按闭环结果再决定要不要加码。

### 8.6 加码一档：**lr 2e-5 → 5e-5、HA ×20 → ×40**（续训，不重开）

从 `logs/wbc_mix_obs93/last.pt`（it 13000，含 π/wm/critic/优化器）**真·续训**到 16000：
`LR=5e-5`（日志 lr 列实测 **4.956e-5**，余弦因子 0.99）、新库 `bones_seed_plus_ha_x40.pt`
（BONES 65.99% + **HA 34.01%** bins，每 rank 39,476 clips / 18.19M 帧）、`FREEZE_WM=0`、`NO_CRITIC=1`、全 clip。
⚠️ 两个工程事实：
1. **resume 不校验库文件**（`runner.py:200-208` 只校验 obs_mode / reward_cfg / hist / goal_frames / action_limit / bounded_critic）
   ⇒ **换库续训是合法的**；
2. **lr 可以在续训时改**：优化器 state 会带回旧 lr，但训练循环每步都用 `lr_at(it)` 重设
   （`runner.py:512-515`）⇒ CLI 的 `--lr` 生效；
3. `--log_dir` **必须换新的**（runner 硬性拒绝复用，防止覆盖旧产物）⇒ 曲线在 `logs/wbc_mix2_obs93`。

**HA-only 探针（同协议）四连点**：

| 权重 | `sim_r` | `term_frac`（HA 摔倒率） | 与训练域的倍数差 |
|---|---|---|---|
| warm start（it 10500） | +0.541 | 0.226 | 45× |
| it 11250（lr 2e-5, HA×20） | +0.680 | 0.064 | 13× |
| it 12250（同上） | +0.694 | 0.046 | 9× |
| **it 15000（lr 5e-5, HA×40）** | **+0.717** | **0.028** | **5.6×** |

⇒ **加码有效**：摔倒率 `0.046 → 0.028`（−39%），`sim_r 0.694 → 0.717`。
训练域（BONES-SEED）参考是 `0.877 / 0.005`，所以 HA 上还有差距，但已经从 45 倍收敛到 5.6 倍。
