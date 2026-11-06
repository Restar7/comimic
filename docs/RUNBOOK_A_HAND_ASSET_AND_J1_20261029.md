# RUNBOOK · A：带手资产统一 + W1 续训 + J1 六轮（4×4090 / 2×H200）

> 2026-10-29 定 · 触发问题：「Supermimic 的 HA 微调到底带不带手？闭环带手、训练不带手，一个资产一个不带，任务能成吗？」
> 结论：**之前确实不对称**，本 runbook 就是把它抹平的完整步骤。

## 0. 不对称在哪（已核实）

| | W1/SuperMimic 训练（旧） | 闭环（旧） |
|---|---|---|
| 机器人资产 | mjlab `unitree_g1` → **29 关节，无手指** | `g1_29dof_with_hand.xml` → **44 关节 = 29 + 14 指** |
| 执行器 | `BuiltinPositionActuatorCfg`（位置伺服） | 43 个 `<motor>`（力矩）+ 脚本里手算 PD |
| 手 | 不存在于模型 | 14 指，VLA 给目标、力矩 PD 驱动 |
| 场景 | 平地，无物体 | 桌子/抽屉/盘/罐 |

后果：WBC 从没见过手的质量/惯量；手部在我方 `--lambda-hand 0` 下完全不训；两套执行器实现靠人工同步。

**增益本身是同一组数**（`g1_sim.KP/KD` 就是 `armature·ω²`、`2ζ·armature·ω`，并联腰/踝 ×2 与 mjlab 一致），所以差异只有"资产有没有手"和"PD 在仿真器内还是脚本里"。

---

## 1. 阶段 A1：生成带手 · position 版资产（**已完成**）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
python3 scripts/make_g1_43dof_asset.py --check
```

输出（实测）：

```
[ok] …/SONIC_DATA/deployment/gear_sonic/data/robot_model/model_data/g1/g1_43dof_position.xml  body_act=29 hand_act=14
[check] nq=50 nv=49 nu=43 fixedgain_actuators=43 timestep=0.002
[check] 关节数 = 43（应 43：29 身体 + 14 手）
```

- 29 个身体执行器：mjlab 四组 kp/kv（`5020 14.3/0.91`、`7520_14 40.2/2.56`、`7520_22 99.1/6.31`、
  `4010 16.8/1.07`，腰 pitch/roll 与双踝 ×2 = `28.6/1.82`），`forcerange` 取 effort limit。
- 14 个手指执行器：`kp=2.0 kv=0.05 forcerange=±2.4`（**与闭环现有手部 PD 完全一致**）。
- robot-only，无桌子/物体；`ctrlrange` 顺带补上（旧资产没有角度限位）。

> 资产必须放在 `SONIC_DATA/.../g1/` 下：xml 里 `meshdir="meshes"` 是相对路径，仓里的
> `groot_repro/scenes/meshes` 只有 `ycb`，挪过去加载会 `Error opening … .STL`。

---

## 2. 阶段 A2：让 W1 训练吃这份资产（`SM_G1_XML` 开关）

```bash
# 训练时指到新资产；不设这个环境变量 = 旧的 29-dof 行为（可回滚）
export SM_G1_XML=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/SONIC_DATA/deployment/gear_sonic/data/robot_model/model_data/g1/g1_43dof_position.xml
```

改动点（`third_party/SuperMimic-Reproduced/src/supermimic/envs/tracking_env_cfg.py`）：

1. `get_g1_robot_cfg()` 在 `SM_G1_XML` 存在时，用 `MjSpec.from_file(xml)` 建 43 个
   `BuiltinPositionActuatorCfg`（同一套分组），`init_state` 里手 14 关节取 0；
2. `actions["joint_pos"].actuator_names` 从 `(".*",)` 改成**排除 `_hand_` 的 29 个**，
   `scale` 沿用 `0.25·effort/kp`；
3. 手不参与动作 ⇒ 每个控制步 `ctrl=0`，手指被 `kp=2` 的位置伺服**定在零位**（质量/惯量进模型，
   但手本身不动）。

**验收（必须全绿再往下）**：

```bash
# obs 必须还是 93、动作必须还是 29、nu 必须 43
python3 - <<'PY'
import os
os.environ["SM_G1_XML"] = ".../g1_43dof_position.xml"
import sys; sys.path.insert(0, "third_party/SuperMimic-Reproduced/src")
from supermimic.envs.tracking_env_cfg import make_env
from supermimic.config import SuperMimicEnvCfgArgs
env = make_env(SuperMimicEnvCfgArgs(num_envs=4), device="cuda:0")
print("obs", env.observation_space, "act", env.action_space)
PY
```

| 期望 | 不满足说明 |
|---|---|
| `obs` 里 policy 维度 = **93** | 手被误加进观测（`actuator_names` 过滤没生效） |
| `act` 维度 = **29** | 手被执行器正则匹配进去了 |
| `env.sim.mj_model.nu == 43` | 资产没被真正加载 |

---

### A2 实施记录：43 关节会一路撞到"29 vs 43"（都已修）

把资产换成 43 关节后，代码里所有"默认 29 关节"的地方都会炸，实测依次踩到并修掉：

| # | 位置 | 现象 | 修法 |
|---|---|---|---|
| 1 | `tracking_env_cfg.py` obs 的 `joint_pos/joint_vel` | 观测混进 14 个手关节 | 传 `asset_cfg=SceneEntityCfg("robot", joint_names=(BODY_JOINT_REGEX,))` |
| 2 | 同上 `joint_limit` 奖励 | 同上 | 同一 asset_cfg |
| 3 | `mdp/commands.py` `default_joint_pos` | `ref.joint_pos(29) - default(43)` | 按 `_hand_` 过滤后再交给库 |
| 4 | 同上 `_write_reference_state_to_sim` | `clip` 到 43 维 soft limits 失败 | 29→43 补默认角再写，指标侧 `robot_joint_pos` 反向切片 |
| 5 | `envs/state_io.py` | `assert qpos==36` | **状态契约保持 71 维**：按关节名挑 29 个身体关节，读时取子集、写时散回去 |
| 6 | `envs/sim_context.py` | `xi` 维度 33 断言失败（bias 43） | bias 只取身体关节 |
| 7 | `training/loop.py` `AnalyticHeads` | WM reward 里 `joint_pos(29) - lo(43)` | soft limits / default 同样过滤 |
| 8 | `make_g1_43dof_asset.py` | mjlab 按 geom 名过滤碰撞，而 SONIC 的 103 个 geom **全匿名** | 生成时按 mjlab 约定命名（`*_collision`、脚 `left_foot1..4_collision`） |

验收脚本：`python3 scripts/check_hand_asset.py` → 期望
`obs.policy(单帧)=93 × hist10 · goal=680 · act=29 · nu=43 · 手部 actuator 14`。

### A3 的 2 个续训坑

1. **`--schedule.total-iterations` 必须大于 resume 的 iteration**：从 `ckpt_0014000` 续训要写
   `STEPS=17000`（跑 3000 步）。写 `STEPS=3000` 会**干净退出**（14000 ≥ 3000），日志里没有任何报错。
2. **`WARMUP` 要抬到 `resume_iteration + 几十`**（如 `WARMUP=14060`）：replay buffer 是空的，
   而 loop 在第一轮就要更新 WM ⇒ `replay buffer has no sampleable rows yet`。

## 3. 阶段 A3：W1 续训（**从 `ckpt_0014000` 微调，不重训**）

为什么从 14000：W1 20k 曲线在 12k 后平台，`ckpt_0014000` 是 8-clip 最优
（mean **0.7607** / step-weighted **0.7985**，比 base e9 的 0.7009 高 **+29%**）。

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
# 4×4090：3000 步 ≈ 30–50 min（第一次带 compile warmup 会多几分钟）
SM_G1_ASSET=43dof RUN=wbc_hand_ft318 STEPS=17000 ENVS=2048 LR=3e-5 \
NPROC=4 GPUS=0,1,2,3 WARMUP=14060 CKPT_EVERY=15500 LOG_EVERY=50 PORT=29670 \
RESUME=$PWD/third_party/SuperMimic-Reproduced/logs/wbc_ft318_mjlab/ckpt_0014000.pt \
  bash scripts/launch_w1_ft318.sh
```

看进度：`grep -E "^it " logs/wbc_hand_ft318.log | tail -3`（期望从 `it 14050` 往上走到 17000）。

| 判据 | 阈值 | 不过怎么办 |
|---|---|---|
| 8-clip mean（同一评测脚本，同 clip 集） | ≥ **0.74**（允许比 0.7607 掉 2.5%） | 掉太多 ⇒ 手质量带来的分布偏移过大，改为**手 kp 降到 0.5**（近被动）再试，或 `STEPS=8000` |
| 闭环 20 tick（`--arm vla --steps 20`） | `fell=false` 且 `max_tilt < 12°` | 同上 |
| WM 门 `qd_corr` | ≥ 0.95 | 续训把 WM 也带偏了 ⇒ 只更新 policy，WM 冻结 |

---

## 4. 阶段 B：J1 六轮（critic 自动门）

```bash
ROUNDS=6 REWARD_STEPS=800 NGPU=4 GPUS=0,1,2,3 \
J1_BATCH=16 J1_LORA=64 J1_H=20 ENVS=512 STEPS=1000 \
BLINE_JOBS=4 BLINE_EPISODES=20 BLINE_TICKS=600 \
WM_REFRESH_EVERY=500 WM_REFRESH_STEPS=100 \
  bash scripts/launch_j1_rounds.sh
```

每轮 7 步与判据见 `RUNBOOK_J1_PIPELINE_H200_20261028.md` §1/§2。**每轮新增的第 5 步**：

```
R$R · 5/7 critic 拟合（门：留出 v_ev > 0.30）
[critic] 本轮 critic=on(v_ev>0.30)      ← 只有过了才开
[critic] 本轮 critic=off(v_ev<=0.30)    ← 不过就自动退回 --no-critic
```

### critic 之前为什么"效果不好"（别重复踩）

| 之前的做法 | 结果 |
|---|---|
| `--critic-source imagined`（在 WM 想象的 rollout 上拟合） | 没有真返回，值函数学到的是 WM 的偏差 |
| 只有 1 集 B 线数据做留出 | `v_ev` 最佳 **+0.22**；后来 +0.2888，都 **< 0.30** ⇒ 一直关 |
| 奖励核在大漂移处饱和（`exp(-e/0.3²)`，0.8 m ⇒ 8e-4，梯度≈0） | 后半段是死区，critic 学到的 return 也被拉平 |

**现在不同**：`scripts/collect_mjlab.py` 的 transitions **带 `ret`（真 return-to-go）**，
`fit_critic_real.py --transitions round_r/transitions.pt` 是用**真返回**拟合，留出 `v_ev>0.30`
才启用 `--critic --critic-source real --critic-ckpt`。所以「开 critic」= 每轮让它自动试一次，
不过就自动关，日志里能看到 `[critic] …`。

要看的具体数字：`[critic] v_ev=…`（训练日志里 `critic/v_ev`）、`loss/boot`。

---

## 5. 时间预算（4×4090，单轮）

| 步骤 | 时间 |
|---|---|
| 1 采集 512×1000 | 2–3 min（`REUSE=1` 可跳过） |
| 2 WM 门 | 20 s |
| 3 B 线 80 集 / 4 并发 | 13–18 min |
| 4 门禁 4 道（auto，`D_fm_bc` 会 OOM 跳过） | 6–9 min |
| 5 critic 拟合 | 2–3 min |
| 6 J1 800 步（batch 16×4 rank） | 50–55 min |
| 7 导出 + 闭环评测 600 tick | 3–5 min |
| **合计** | **≈80–90 min/轮** → 6 轮 **8–9 h** |

想压时间：`REWARD_STEPS=500`（≈35 min，判停靠闭环 reward 仍够用）或门禁并行（待做，见 §6）。

---

## 6. 已知未做（按价值排序）

1. **门禁 4 道并行到 4 张卡**：现在串行，8–12 min/轮（4090 上 `D_fm_bc` 还 OOM）。
2. **闭环切换到同一份 position 资产**：现在闭环仍是 motor+PD；两者增益等价，但"字面同一份 xml"
   要改 `g1_sim.set_ctrl` → 写 `q_target`。收益小、风险中，放后面。
3. **W1 在任务场景里训练**：现在 W1 是平地无物体，只有 J1 的 B 线见过桌子。
4. **手部监督**：`--lambda-hand 0` 保持不变（你的口径），若要手真正学会抓，需要 ≥0.2。

---

## 7. 锁死的完整流程（2026-10-29 定稿 · **不再改动**）

一条命令（4×4090）跑完整轮次；参数已按下面的表锁死，`launch_j1_rounds.sh` 的默认值就是它们。

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
VLA0=/inspire/hdd/global_user/liumengfan-253108110079/flowbody/FlowBody/runs/groot_sonic_vla_20k_frozen271_20261023/checkpoint-20000
setsid nohup env OUT=$PWD/runs/j1_rounds_final ROUNDS=3 NGPU=4 GPUS=0,1,2,3 \
  J1_BATCH=16 J1_LORA=64 J1_H=8 ENVS=512 STEPS=1000 \
  BLINE_JOBS=4 BLINE_EPISODES=20 BLINE_TICKS=600 \
  WM_REFRESH_EVERY=0 WM_PRETRAIN_STEPS=1000 WM_BLINE_RATIO=0.25 \
  CRITIC_MODE=on CRITIC_SOURCE=imagined TRAIN_WBC=0 \
  VLA_CKPT=$VLA0 \
  bash scripts/launch_j1_rounds.sh > logs/j1_final_launch.log 2>&1 < /dev/null &
```

### 7.1 每轮 9 步（顺序是重点）

| # | 步骤 | 关键动作 | 产物 |
|---|---|---|---|
| 1/9 | 仿真采集 | `collect_mjlab.py`，512 envs × 1000 tick（**只跑 WBC、不含 VLA**）；WBC 冻结 ⇒ 第 2 轮起**硬链接复用**第 1 轮 | `transitions.pt` |
| 2/9 | **B 线闭环采集** | `arm=vla`，4 任务 × 20 集 × 600 tick，起点**每轮随机**（`--seed $R`，任务区间内分层） | `bline/b_*.pt` + `self_targets.pt` |
| 3/9 | 混合 WM 数据集 | B 线 → transitions 格式（补 `ret/xi/ref_rows`），与仿真按 **B 线 25%** 混合 | `bline_transitions.pt` / `wm_mix.pt` |
| 4/9 | **训练前把 WM 训稳** | `pretrain_wm.py`，用 `wm_mix` 训 **1000 步**，产物即本轮后续唯一使用的 WBC/WM | `wm_pre.pt`（→ `CUR_WBC`） |
| 5/9 | WM 双分布门 | sim 门（≥0.95 **硬停**）+ B 线部署分布门（记录，用于跨轮看趋势） | `gate_wm_sim.json` / `gate_wm_bline.json` |
| 6/9 | 门禁 | `A_codec / C_burnin / C2_token_contrast / D_fm_bc`（`GATES=auto`：OOM 记 skipped） | `gates/*.json` |
| 7/9 | critic | **GAE**（`--critic --critic-source imagined`），目标来自 4/9 训好的 WM 的想象 rollout | `critic.pt` |
| 8/9 | 联合微调 | 800 步，`--wm-pretrain-steps 0 --wm-refresh-every 0`（**WM 全程冻结，reward 固定靶**） | `train/joint_final.pt` / `train/vla_final.pt` |
| 9/9 | 导出 + 评测 + 接下一轮 | 合并 LoRA → `ckpt_vla_r$R`；600 tick 闭环评测；**训练后**再测 bline WM 门；`CUR_WBC=joint_final.pt`、`CUR_VLA=ckpt_vla_r$R` | `ckpt_vla_r$R` / `eval.json` / `gate_wm_bline_after.json` |

### 7.2 锁死的参数

| 参数 | 值 | 定这个值的依据 |
|---|---|---|
| `J1_H` | **8** | pathwise 窗口；H=5 时 >0.1 s 的后果全压给 critic |
| critic | **GAE（imagined）** | real 监督的 `v_ev` 只有 0.09–0.15（输入缺 clip 身份/进度，结构性不可学） |
| `TRAIN_WBC` | **0（冻结）** | 实测解冻 800 步后同 head 闭环 root_pos 4.2 cm→89 cm（21×），是 WM 偏差下的 reward hacking |
| `w_root_pos / w_root_ori` | **1.0 / 1.0** | root 权重占比 20%→33% |
| `std_root_pos / std_root_ori` | **0.2 / 0.25** | std=0.3/0.4 时 4 cm 处 reward 0.982（饱和）；收紧后 0.961，小误差重新有梯度 |
| `WM_PRETRAIN_STEPS` | **1000** | WM 只有 8.8M，训得快；一次训稳后整轮冻结 |
| `WM_BLINE_RATIO` | **0.25** | 纯仿真 WM 在 B 线分布上 `qd_corr` 只有 **0.638**（`ratio 2.95`），混 25% 拉到 0.92 |
| `WM_REFRESH_EVERY` | **0** | 训练中不再动 WM（避免 reward 移动靶） |
| `max_step_reward` | **0.12** | `dt × Σw`；reward 尺度已变，早间 R1=0.0732 / R2=0.0598 **不可比** |

### 7.3 判据

| 门 | 阈值 | 不过怎么办 |
|---|---|---|
| WM sim 门 | `qd_corr ≥ 0.95` | **硬停**（exit 2） |
| WM B 线门 | 记录 `qd_corr` / `ratio` | 第 1 轮低是正常的；第 2 轮起应 > 0.85，否则 `WM_BLINE_RATIO` 抬到 0.4 |
| 主效果指标 | 六维漂移 `root_pos` / `root_ori` / `joint_rad` / `joint_vel` / `lin` / `ang` 的 p50 与 p90 | 用 `scripts/eval_realized_drift.py` 在每轮 B 线上算 |
| 辅指标 | `rounds.csv` 的 `mean_reward`（新尺度）、critic `v_ev` | 判停：连续 2 轮主指标不改善 |

### 7.4 为什么"每轮都要重采 B 线，但仿真采集只采一次"

- **B 线必须重采**：VLA head 每轮都变 ⇒ 闭环行为、`self` 锚（"当时真正下发的 token"）、真实漂移分布都变；
- **仿真采集不用重采**：`collect_mjlab.py` 只跑 WBC（冻结），数据每轮等价 ⇒ 硬链接复用，省 2–3 min/轮。

### 7.5 累积链（VLA + WM 都逐轮累积）

```
R1: VLA=原始           WBC/WM=ckpt_0014000        → ckpt_vla_r1 + joint_final_1
R2: VLA=ckpt_vla_r1    WBC/WM=joint_final_1(+1000步wm_mix)  → ckpt_vla_r2 + joint_final_2
R3: VLA=ckpt_vla_r2    WBC/WM=joint_final_2(+1000步wm_mix)  → ckpt_vla_r3 + joint_final_3
```

（B 线 dump 不加入 318 库，库始终是固定 318 clips；B 线只喂 `self` 锚、训练起点、WM 混合集与评测。）

---

## 8. 2026-10-29 追加：bootstrap 改成单边（`boot_one_sided`）

**症状**（R2 实测）：`critic/v_adv_H` 从第 0 步就翻正，80% 的步为正，`max` 摸到 +0.76；
而 R1 同一指标从第一桶就是负的、一路走到 −0.45。按同步骤对齐：

| steps | R1 | R2 |
|---|---|---|
| 0–49 | −0.073 | **+0.152** |
| 50–99 | −0.191 | **+0.192** |
| 100–149 | −0.314 | **+0.165** |
| 750–799 | −0.449 | — |

**为什么正号是错的**：`v_adv_H = V(s_H) − V(ref_H)`，而 `V(ref_H) = critic(ref_H, ref_H, xi)`
是"状态差为 0"那个点，跟踪奖励在它最大 ⇒ 正确的 critic 必须给 `rel ≤ 0`。
`critic/v_ev = 0.9951`（留出集）说明 critic 拟合得很好 ⇒ 问题在**目标**（想象 rollout 的
GAE），不在 critic。WM 的 2×2 实测给出可乘之机：`qd_ratio` 样本内 1.12–1.26，
**分布外 1.33–1.42**（关节速度被高估 33–42%）。

**改动**：`V3Config.boot_one_sided = True`（`diff_vla_wbc/v3_step.py`），
`rel := min(rel, 0)`——只惩罚"比参考差"，不为"比参考好"付钱。
原始值仍记录为 `critic/v_adv_H_raw`，**不要删**：它就是判断这套护栏有没有掩盖真信号的唯一依据。

**生效范围**：R2 的 8/9 已经加载了旧模块 ⇒ R2 是**未修的最后一轮**，可当 A/B 的对照；
R3 起生效。每轮 `v3_config.json` 都会 dump `boot_one_sided`，AUDIT 行也打 `boot_1side=`，
所以哪轮用了哪种模式是可审计的。

## 9. 2026-10-29 追加：WM 阶段 2b（k 步自回归）与它的坑

**问题**：`wm_refresh` 的损失是**纯单步**（`s_next = wm(state, action)`，输入永远是真实状态），
模型从未见过自己的输出 ⇒ 累积误差不受约束。实测 held-out：k=1 的 `qd_ratio` 0.977
（门禁只量这一步，永远 PASS），而 k=20 是 **1.329**。

**第一版做法（错的，已废弃）**：单独加一段"纯 B 线连续窗口滚 k 步"的阶段。
600 步、lr 3e-5 就把 **sim 单步门打坏**：`corr 0.9896→0.9356`、`ratio 0.995→1.041` ⇒ `exit 2`。
这正是本文档记过的灾难性遗忘，只是这次发生在阶段 2b 而不是阶段 3。
**教训：只要在 B 线窗口上单独训练，就一定会伤 sim 分布。**

**现行做法**：阶段 2b 每步**交织一条真实场景单步样本**（`--roll-sim-weight`，默认 0.3）。
终版参数 `--roll-k 20 --roll-steps 1500 --roll-sim-weight 0.3`，两个官方门都过：

| 门 | 无 2b | 破坏版 2b | 交织版 2b（定稿） |
|---|---|---|---|
| sim corr | 0.9896 | 0.9356 ❌ | **0.9878** |
| sim ratio | 0.9950 | 1.0412 | **1.0064** |
| bline corr | 0.9856 | — | **0.9824** |
| bline ratio | 1.0310 | — | **1.0296** |
| k=20 qd_ratio（跨轮 held-out） | ~1.26 | 1.178 | **1.175** |

**判据（2026-10-29 21:10 定稿，之前用错了）**：只认**同分布**（每轮 WM 在**自己那一轮**的
dump 上测）+ 双门，**跨轮数字无意义**。实测（k=20，同分布）：

| WM | joint_mae | qd_corr | **qd_ratio** |
|---|---|---|---|
| round_1（修复前） | 0.63° | 0.905 | **1.260** ❌ |
| round_2 | 0.47° | 0.956 | **0.990** ✅ |
| round_3 | 0.47° | 0.932 | **1.003** ✅ |
| round_4 | 0.52° | 0.938 | **0.991** ✅ |

⇒ **qd_ratio 从 1.260 修到 0.99–1.00，连续三轮稳定不掉，问题已解决。**

**为什么不能看跨轮**：在 round_1 的 dump（= π0 的行为）上，四轮 WM 的 k=20 ratio 是
1.252 → 1.165 → 1.267 → 1.386，看着像"越训越差"。实际上策略每轮都在变，越往后的 WM
越**不该**匹配 π0 的过期行为。WM 的职责是贴当前策略的分布；跨轮数字量的是策略演化，不是 WM 质量。

**待办（下次重构第一件事）**：`wm_refresh.py` 的 `ori` 项把**四元数**当**旋转矩阵**用
（`(R*S).sum(-1).sum(-1)` 对 (B,4) 等于把整个 batch 求和），实际是 `0.5·(3−Σ_b cos)`，
梯度是应有值的 **B 倍**（B=2048 ⇒ 2048×），而 pos/qj/vel 都是 mean 尺度。
证据：预训练日志 `loss_first=−1021.9` 正好等于该式在 B=2048 下的值；实测 WM 单步朝向
误差 0.08°（近完美）而 qd 是唯一烂通道。注意模型输出头会归一化四元数，所以**不是坏模型，
是权重失衡**；但 `mag_weight=30` 是在这个错误缩放下调出来的，改它必须重新调参。

## 10. 2026-10-29 追加：`mean_reward` 从 R2 起不可跨轮比较

R2 把 `std_root_ori` 从 0.25 收到 0.15，于是同一个朝向误差的奖励值会**机械性下降**：
以 R1 的 8.81° 代入，`exp(-e²/0.25²)=0.685`，而 `exp(-e²/0.15²)=0.350`。

**所以：R2 的 `mean_reward 0.0764` vs R1 的 `0.0879` 不能解读为变差。**
跨轮比较**只能用 `eval_realized_drift.py` 的物理分量**（root_pos / root_ori / joint / vel）。

连带影响：`run_j1_rounds.sh` 用 `mean_reward` 做 `BEST_R` 早停判定，从 R2 起这个比较
本身就跨标尺了。当前 `PATIENCE=99` 让它不生效，**但下次重启前必须关掉或换指标**，
否则会把"换了尺子"误判成"没涨"。

**另一个坑：单集 eval 的 p50 有 ~30% 的噪声。** R1 的单集 eval 给 root_pos p50 = 4.06 cm，
而 R1 的 80 集采集分布是 p50 = 5.8 cm、p90 = 24.9 cm。所以**永远用 80 集采集分布判**，
不要用 9/9 那一条 600 tick 的轨迹下结论。

## 11. 2026-10-29 追加：80 集趋势 —— 中位数改善、尾部恶化（**用 p50+p90 双指标**）

| 指标（80 集 / 48000 tick） | π0 训练前 | π1 = R1 产物 | π2 = R2 产物 |
|---|---|---|---|
| root_pos **p50** | 5.8 cm | 5.3 cm | **4.1 cm** ✅ |
| root_pos **p90** | 24.9 cm | **52.1 cm** | **59.6 cm** ❌ |
| root_ori **p50** | 10.7° | 10.2° | 10.3° ≈ |
| root_ori **p90** | 24.7° | 25.8° | **30.7°** ❌ |
| 坏集数（p50>10cm） | 19/80 | 22/80 | 19/80 |

结论：
1. **单集 eval 的 p50 有 ~30% 噪声**（R1 单集 4.06 cm vs 80 集 5.3 cm），判轮次只能用 80 集分布。
2. **尾部从 R1 就在恶化**（24.9→52.1→59.6 cm），坏集数稳定 ⇒ 是坏集冲得更深，不是更多集变坏。
   根因：奖励是 `exp(-e²/std²)` 的**均值**，没有 CVaR / 最坏情况项，也没有针对漂移的终止惩罚。
3. **`std_root_ori 0.25→0.15` 在总体层面无效**：单集 eval 8.81°→7.32° 是噪声，
   80 集 p50 是 10.2°→10.3°（平），p90 反而变差。**root orientation 问题仍未解决。**

**停止线**（2026-10-29 20:53 更新，四条任一触发即停链并改为在奖励里加尾部项，不再靠多训硬扛）：
1. `root_pos p90 > 80 cm`（现 53.7，π3 已回落）
2. **`root_ori p90 > 45°`**（现 36.0，**单调恶化**：24.7→25.8→30.7→36.0）
3. `fell > 0`（至今全 0）
4. `root_pos p50` 或 `root_ori p50` 连续两轮回退

**π3（R3 产物）的 80 集**：pos p50 4.3 cm / p90 53.7 cm；ori p50 **10.0°（四轮最好）** / p90 36.0°。
即"中位在变好、朝向尾部在变坏"。

**每轮必报**：80 集 `root_pos` / `root_ori` 的 **p50 和 p90**（只用 p50 会掩盖尾部）。

## 12. 2026-10-30 01:50 停链复盘 + 尾部铰链实验

### 归因纠正（重要）
我一度把退化归因于"R2 起改的配方（boot_one_sided + WM 阶段 2b + ori 0.15）"。**这是错的。**
`pos p90` 在 **π1（R1 的产物、原始配方、还没做任何改动）就已经从 24.9 翻到 52.1**。
所以 **尾部恶化是训练目标本身的性质，与 2b / boot / ori 无关**；我的改动对应的是后来的
**摔倒**（π3 起 2/80）和**中位失守**（π4 起）。

### 根因：奖励是"饱和高斯 + 均值"，对尾部结构性失明
按误差分桶实测（σ=0.2 m）：

| 误差区间 | π0 占比 | π4 占比 | π4 梯度贡献 |
|---|---|---|---|
| 0–0.1 m | — | 72.4% | **86.3%** |
| >0.4 m | 3.1% | **21.2%** | **0.22%** |

`exp(-e²/σ²)` 在 0.6 m 处已是 `1.2e-4` ≈ 0。**21% 的时刻只提供 0.22% 的位置梯度** ⇒
均值目标主动"拿尾部换中位"：p50 5.79→4.11（CI 不重叠，**显著改善**）的同时
p90 24.9→52–60（**2.2 倍恶化**）。这不是噪声，是这个目标的最优解。

critic 不是元凶（`v_ev` 留出集 0.99，忠实拟合），WM 是放大器（分布外 `qd_ratio` 1.33–1.42
vs 分布内 0.99–1.00 ⇒ 深漂移在想象里更便宜）。

### 修复（本轮唯一的新变量）
`TrackingRewardCfg` 增加**尾部铰链**：
`-tail_pos_weight · relu(‖Δp‖ − tail_pos_e0)`，λ=2.0、e₀=0.35 m、cap 1.0 m。
实测（数值验证）：**`e ≤ 0.35 m` 时梯度逐位不变**；`e > 0.35` 时梯度从 ~1e-4 变成恒定 −0.04。
`max_step_reward` 仍 0.120。

### 本轮设计（`runs/j1_tail/round_1`，从 π1 起）
| 项 | 值 | 理由 |
|---|---|---|
| `std_root_ori` | 0.25 | 0.15 实测无效且有害，已回滚 |
| `boot_one_sided` | False | 未经验证为有益，回滚 |
| WM 阶段 2b | **开** | 已证清白（尾部在 π1 就坏，当时 2b 不存在）；且让想象更忠实 |
| 尾部铰链 | **开** | 本轮要测的东西 |
| 坏集过滤 | 开 | 摔/倾角>45° 的 episode 不进 `wm_mix` |
| 80 集硬门 | 开（`exit 8`） | `fell` 必须 0；reward ≥ 0.0800 |

起点与 R1 相同、**起点帧完全相同**（`BLINE_STARTS_MODE=auto` 确定性），所以是同批同起点对比。

### 判据（提前写死）
`pos p90 ≤ 35 cm`、`pos p50 ≤ 5.9 cm`、`fell = 0/80`、`>0.4 m` 占比 ≤ 8%。

## 13. 修复的性质分层（2026-10-30 02:10）

防止把"止损"当成"根治"，逐项定性：

| 改动 | 性质 | 说明 |
|---|---|---|
| **pos / ori 尾部铰链** | **根治（机制层）** | 缺陷在**目标函数**里：`exp(-e²/σ²)` 均值 0.6m 处≈0，"拿尾部换中位"是它的数学最优解。铰链改变目标本身 ⇒ 同一病理不可能再经该通道产生 |
| 坏集过滤 | 半根治 | 断掉"摔的集回流 → WM 学到摔 → 更差"的正反馈，但不阻止策略产出坏行为 |
| 80 集硬门 | **亡羊补牢** | 只止损。存在的理由就是我漏看 4 小时的教训 |
| `ori` std 回滚 / `boot_one_sided` 回滚 | 撤销 | 未经证实有益 |

**铰链管不到的三个放大器（下一步必须补）：**
1. **H=8 看不见摔倒**：0.16 s 视野，摔倒要 ~1 s 成形 ⇒ 需加倾斜/摔倒终止项，或把 H 提到 16–20。
2. **WM 轮内不保鲜**：`WM_REFRESH_EVERY=0`，策略 800 步对着过期模型优化 ⇒ 开 `WM_REFRESH_EVERY=150`。
3. **WBC 冻结**：跟踪上限被锁死，头部只能 re-target 不能提高跟踪精度。

**铰链自身的局限**：它只是给尾部**加权重**，聚合仍是"均值"。
真正对应"避免最坏情况"的聚合是 **CVaR / 分位数**；铰链是它的线性近似。
若本轮有效，下一版应把聚合换成 CVaR，而不是继续调 λ。

**注意**：两个铰链都让奖励整体下移（罚项只减不加）⇒ **`mean_reward` 从本轮起又是新尺子，
不可与 `j1_scene2` 的任何历史数字比较**（那里还有 ori 0.25/0.15 两把尺）。

## 14. 2026-10-30 02:15 更正 §13 的两条建议（与已跑消融冲突）

§13 我提了"H 拉到 16–20"和"开 `WM_REFRESH_EVERY=150`"，**两条都与已跑完的消融记录冲突**，更正如下。

### 14.1 H 不能拉长（A4 已否）
`docs/ABLATION_MASTER_SUMMARY_20261025.md`：A4（WM 冻结 × horizon，各 400 步）
`H8 +0.0820 > H16 +0.0778 > H32 +0.0734 > H40 +0.0681`，**单调下降**；
A4 报告写明机理"朝向误差按 `dt·|w_b−w_world|`/步累积，H 越大越偏"。⇒ **保持 H=8。**

**正确入口：给奖励补 tilt / pelvis 高度每步惩罚。**
`tracking_env_cfg.py` 里终止条件已有 `root_tilt_threshold=0.8`(≈45.8°) 和 pelvis 高度，
但 `rewards.py` 里 **`grep -niE "tilt|height|fell|fall"` 零命中** ⇒ 摔倒只在终止时通过
"丢后续奖励"间接惩罚，而 H=8=0.16 s 根本走不到终止（摔倒要 ~1 s 成形）。
把前兆做成**每步**惩罚，8 步就能看见，且不与 A4 冲突。

### 14.2 WM 保鲜的禁令是本栈特有的
A4 里 `wm_live` 是 **⛔ 政策禁止**（不是"跑了变差"），理由："在线更新会把**混帧补偿**当物理洗掉"。
混帧补偿是 **Isaac 栈**的产物；mjlab 是 `explicit_dt0.02` 显式状态且 `refresh_wm` 带 `kl_anchor`。
⇒ 该禁令**不必然适用本栈**，但本栈也**没有任何证据支持有益**。**必须做独立单变量 A/B，不能当默认改。**

### 14.3 WBC 冻结：A5 的结论是"要训"，我们是"证据不足"
A5（900 步 × 2 臂）：`head_only` token 0.3721 / reward +0.0796；
`head_plus_wbc` token **0.1161（3.2×）** / reward **+0.0834**。
记录结论："更新 WBC 明显救 token 与 reward ⇒ 选 `head_plus_wbc`"。

我们 `TRAIN_WBC=0` 的唯一反证是那次闭环爆炸（root_pos 4.16→89.2 cm），但那是 **n=1 单集**。
2026-10-30 已两次被 n=1 误导（`pos p90` 噪声、`eval.json` 的 `fell` 字段看错整晚）。
⇒ **WBC 问题是"未决"，不是"已关闭"**，须用 **80 集判据 + 每 100 步闭环门**重测。
A5 记录的代价（hand 变差）在本栈不存在（`SM_HAND=0`，动作 29 维，手不在动作空间）。

### 14.4 执行顺序
A 尾部铰链（在跑）→ **B tilt/高度每步惩罚**（同族病，紧跟）→ C 重测 `TRAIN_WBC=1`（带 80 集门）
→ D `WM_REFRESH_EVERY` A/B（未证实）。坏集过滤 + 80 集硬门全程保留。

## 15. 2026-10-30 02:20 再更正：A5 **不是**"WBC 该训"的证据

§14.3 我引用 A5 说"记录结论是要训 WBC"，**这条引用被过度解读了**，查原文后更正。

### A5 到底训了什么
`docs/ABLATION_A5_RUN.md`：
```
| 唯一变量 | --policy-lr/--adapter-lr：head_only = 0/0；head_plus_wbc = 默认 1e-5 |
| WM       | 冻结（--wm-update-every 0，两臂相同） |
```
⇒ **WBC 是被"奖励穿过冻结 WM 的 H=8 想象"更新的**，损失与 head 同一份。
**没有 transitions、没有 BC、没有任何真实动作监督。**

判据全是**想象侧**：`rollout_reward` / `state_term` / `token_mse` / `hand_mse`。**没有闭环指标。**

A5 自己贴的口径警告（绝对值不可用）：
1. 条件 token 用 HA 标签 vs 底模的 trackerlab 端口 token，**相关系数仅 0.06–0.19**；
2. `--state-vel-weight=1.0`，而"WM 速度通道 ~96% 能量是不可学噪声"；
3. 动作还是旧换算 `a = Δq/action_scale`。

### 正确的读法
| | A5（Isaac） | 我们（mjlab） |
|---|---|---|
| WBC 更新 | RL 穿冻结 WM | RL 穿冻结 WM |
| 判据 | **想象** reward / token MSE 变好 | **闭环** root_pos 4.16→89.2 cm |
| WBC 位移 | `max\|Δ\| = 4.6e-3` | `max\|Δ\| = 4.8e-4`（**小 10 倍仍炸**）|

⇒ A5 与 mjlab 结果**不矛盾**：A5 证明"让 WBC 动会提高**想象**奖励"，这正是 reward hacking 的签名；
mjlab 证明同一机制在**闭环**上是灾难。**想象指标结构上看不见闭环崩溃。**

### 因此 C 项重新定义
- ❌ 不是"重测 `TRAIN_WBC=1`"（RL 穿 WM 的路子，机制没变，只会重演）
- ✅ 是**新写 `scripts/pretrain_wbc.py`：用真实 (s,a,s′) 监督 WBC**（手上现成：每轮 1,023,700 行
  场景数据 + 80 条 B 线），配**每 100 步的 80 集闭环回滚门**
- 现状：`--real-transitions` 在本仓只用于 WM 预训练/保鲜、`critic-source real`、
  `--start-source recorded`；**没有任何路径用真实动作监督 WBC**，`pretrain_wbc.py` 不存在。

**执行顺序更新**：A 尾部铰链（在跑）→ B tilt/高度每步惩罚 → C 写 `pretrain_wbc.py` 并离线验证
（不占主链 GPU 时间，与 B 并行）。

## 16. 2026-10-30 02:57 换成奖励实验时**步骤 1–6 可完全复用**

在 `closed_loop_vla_supermimic.py:480-481` 里，奖励只被写进 `rec_rewards`，**从不参与动作生成**。
因此 B 线轨迹与奖励函数无关。逐项核对：

| 步骤 | 依赖奖励？ | 换奖励后 |
|---|---|---|
| 1/9 场景采集 | ❌ | 复用 |
| 2/9 B 线 | ❌（奖励只记录） | 复用 |
| 3/9 bline→transitions + mix | ❌（只搬 states/actions） | 复用 |
| 4/9 WM 预训练 | ❌（只吃 states/actions/next_states） | **复用 `wm_pre.pt`** |
| 5–6/9 双门 | ❌ | 复用 |
| 7/9 critic、8/9 微调、9/9 验收 | ✅ | **必须重跑** |

⇒ **换奖励只需重跑 7-8-9，约 35 min（原 2 h）。** 待办：给 runner 加 `RESUME_FROM_STEP`。

**副作用**：复用的 dump 里 `rewards`/`ret` 是旧尺子。我们走 `critic-source imagined` 不受影响；
健康门的 `fell`/`max_tilt` 与尺子无关，**只有"奖励塌"那一项要按相对值看，不可跨尺子比**。

### 健壮性修复（2026-10-30 02:15 事故）
那次链条在 B 线第 28/80 条时被外部杀掉（主机内存充足、无 OOM、episode 日志停在中途）。
加了两处：
1. **断点续采**：dump 文件名确定（`b_${T}_f${F}_e${E}.pt`），已存在就跳过；
2. **`wait || true`**：原来 `set -e` 下任何子进程非零都会把整条链带走（很可能就是死因）。

### 优先级修正（原 §14.4）
**A 尾部铰链（在测）→ B tilt/高度每步惩罚（必须测）→ C 真实数据监督 WBC（必建、离线、不占主链）
→ D WM 保鲜：降级为"仅当 A/B 都失败才动"**（本栈零证据；且双门显示 WM 对新策略数据仍拟得好：
bline ratio 1.01–1.02、corr 0.987，"过期"并不明显）。
