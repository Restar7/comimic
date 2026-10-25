# 交接：主线联合训练 + 纠错消融（2026-10-25，出卡：wscoli2 / 卡 0,1）

> **收件人**：所有还在跑 A1–A5 的容器（卡 0,1 / 2,3 / 4,5 / 6）。
> **一句话**：底模换了（显式物理积分 + 800 步速度标定）、三处口径写死（**端口 token 条件 / `a=Δq` / WM 冻结**）、
> **之前的消融曲线全部作废**，按本文重跑。唯一**不能**开的是 **A4 的 `wm_live` 两格**。
> **作用域**：本文**只覆盖 L2（联合微调 + SuperMimic WBC）这条线**。L1（官方 VLA + SONIC decoder）见
> `docs/HANDOVER_LATEST_FINDINGS_FOR_4090_20261025.md` §1.1/§2.B，**不要把两条线的 token 混用**。
> 本文数字全是本卡实测（复现命令见 §9），不是推测。
>
> **2026-10-25 05:30 补（2×4090 容器）**：WM 的**朝向通道坐标系**已修正（世界系 `w` 不能直接进右乘指数），
> 实测**旧/新标定差别可忽略、真正拉开数字的是"残差→显式积分"**。哪些格子能换底模、哪些不能，见 **§11**；
> **不需要再为了"换标定"重跑任何东西**。

---

## 0. 生效版本（先对齐，否则白跑）

| 项 | 值（写死） |
|---|---|
| **官方 WBC（唯一）** | `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt` |
| 结构开关 | `wm_integration="explicit_dt0.02"`（加载即自动启用；`joint_latest.pt` 也会存这个键） |
| WBC 原底模 | `ckpt_010500.pt`（BONES-SEED 预训练；**run 名里的 `_ha_` 有误导性**，见 `runs/OFFICIAL_SUPERMIMIC.json`） |
| **WM 使用** | **冻结**：`--wm-update-every 0`（**禁止**再丢回 trackerlab runner 续训，见 §1.3） |
| **条件 token** | **端口空间**：`data/port_tokens/ha_tokens/*.npy`（`RecordedTokenSource` 自动优先，缺失才回退 npz 标签） |
| **动作** | `a = Δq`：`COMIMIC_ACTION_SCALE=one`、`COMIMIC_REWARD_ACTION_SCALE=spec`（默认，别改） |
| 速度项权重 | **`--state-vel-weight 0`**（速度**不参与** `state` 正则；只作为 `state_vel` 诊断量打印） |
| 数据 | `data/npz/ha_tokens`（699 clips / 634,016 帧，纯 HA）；packed：`runs/isaac_motion/ha_tokens/packed.pt` |
| VLA 底模 | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000` |
| 联合训练环境 | `/usr/bin/python3`（3.12 + torch 2.8）+ `PYTHONPATH=…/gr00t_deps:…/gr00t_deps/_noflash`，2 卡 torchrun |
| WBC/Isaac 环境 | `third_party/isaacsim_env` = **Isaac Sim 5.0.0** + `third_party/isaaclab_220` = **IsaacLab 2.2.0** |
| eval 协议 | `EVAL_HORIZON=40` 抽帧 + `@8` 前缀指标 + **新增分项** `state_pos / state_ori / state_q / state_vel` |

### 0.1 ⚠️ 两条路线（写死，别再混）
| 路线 | 链路 | 执行器 | 用途 |
|---|---|---|---|
| **L1 官方线** | VLA → 64 token（**HA 标签空间**）→ **SONIC decoder** → 动作 | `runs/sonic_decoder_final.pt`（蒸馏，corr 0.989）或真 `model_decoder.onnx` | 官方基线成功率 |
| **L2 联合微调线（我们）** | VLA → 64 token（**装配空间**，端口侧车）→ **SuperMimic WBC** → WM → reward → BPTT | `ckpt_010500_physint.pt`（WM 冻结） | 论文主结果 / A1–A5 |

- 两条线 token 空间不同，**不得互相喂 token**；
- 一次对比里两臂**必须同一执行器**；
- A1 目前实现的是 **L2 内部的"有无奖励反传"**（`base` vs `ft`，都用 WBC）；
- "L1 vs L2" 的 headline 对比要**分别跑两条线**：L1 臂（官方 VLA + decoder）单独跑，L2 臂（联合 VLA + WBC）单独跑，
  **两条线不交叉**（联合微调的 VLA 配 decoder 是错的）。臂表见 §4.3。

### 0.2 为什么是 Isaac 5.0，不是 5.1（别再搞混）
HA 那批 SuperMimic run（`bones_seed_token_ha_2x8192` / `_ha_full_2x8192` / **`_ha_v2`（我们的底模）**）的 launcher 日志都是
`third_party/isaaclab_220/apps/isaaclab.python.headless.kit` + `Isaac-Sim/5.0`。
5.1/2.3.2（`third_party/trackerlab_env`）只用于更早的 **BONES-SEED** run，**不要**拿它跑 HA 相关的 WBC 训练。

### 0.3 ⚠️ 不要再用旧代码快照
`runs/_code_a2_20261024`、`runs/_code_a5_20261024`、`runs/_code_a3a5_20261024` 都是**旧口径**（HA 标签 token、
`state-vel-weight 0.05`、旧动作换算、缺 `wm_integration` 保存）。要么直接从仓库根目录跑（共享盘同一份），
要么先按 §5 的清单重新打一个带 `CODE_MANIFEST.sha256` 的快照。

---

## 1. 三条口径（之前所有坑的根）

### 1.1 条件 token 必须是 WBC 的端口空间
- HA npz 的 `g1_motion_token_64`（部署侧 encoder 输出）与 WBC 训练时的**端口 token** 不是同一空间：
  实测 `max|Δ| = 1.06–1.19`、**相关系数只有 0.06–0.19**。
- ⇒ 把 HA 标签 token 直接当联合训练的条件 = 给 policy 喂随机码 → 直接崩（实测 `sim_r 0.52 → 0.15`、`term 0.17 → 0.55`，40 iter）。
- 正确做法：`data/port_tokens/ha_tokens/`（699 条侧车）；`RecordedTokenSource` 会优先读它。
- ⇒ **`token_mse` 只对端口 token 有意义**；对 HA 标签的 `token_mse` 没有保真度含义（训练中上涨是正常现象，不是 bug）。
- ⇒ VLA 侧必须产出端口 token（侧车/转换链路），**不能**直接喂标签 token。

### 1.2 动作换算 `a = Δq`
- 旧代码多除了一次 per-joint `action_scale`（0.0745–0.5476），把参考动作放大 4–8 倍：
  参考跟踪 reward `+0.069 → −0.129`，末步状态误差 1.31。
- 现在 `COMIMIC_ACTION_SCALE=one`（`a = Δq`）为默认；**奖励内部**的平滑项仍用 spec scale
  （`COMIMIC_REWARD_ACTION_SCALE=spec`）——这两个默认值都不要改。

### 1.3 WM 必须冻结（这是速度修复能不能保住的前提）
- 速度通道是在**参考窗口 + policy 自己的动作**上标定的（800 步）；PhysX 一步转移的速度增量被
  `action_noise=0.1` + 接触/PD 噪声主导，用 sim 数据继续训 **300 iter** 就会把它洗回均值场：
  `qd_corr 0.897 → 0.095`、ratio `0.93 → 4.27`、state `0.555 → 3.575`。
- ⇒ 联合训练一律 `--wm-update-every 0`；
- ⇒ **不要**把 `ckpt_010500_physint.pt` 丢回 `trackerlab-supermimic/scripts/rls/supermimic/train.py` 续训（那会把标定洗掉）。

---

### 1.4 速度不进 loss ≠ 结构/标定不重要（为什么仍要保留 explicit 结构 + 标定）
- `--state-vel-weight 0` 只是**不拿速度当 `state` 正则**；
- 但 WM 预测的速度仍然进两个关键通道：
  1. **reward**：`reward_cfg` 里有 `w_body_linvel` / `w_body_angvel`，tracking 奖励用到速度；
  2. **rollout 的下一步 obs**：99 维里有 `qd`(29) + `Rᵀw`(3)，速度错了下一步 policy 的输入就错了；
  3. （可能）`soft_termination` 用到速度/倾角。
- ⇒ 显式积分结构 + 800 步标定（`qd_corr 0.90`）仍然是 **reward 可信**的前提，不是白做的；
- **A/B 验证（30 min × 2，可选但建议）**：同一 `state_vel=0` 配置下
  `ckpt_010500_physint.pt`（显式+标定） vs `ckpt_010500.pt`（残差+未标定），比 `rollout_reward` 与 `token_mse` 曲线。
  若 A 明显好 → 结构/标定的价值通过 reward/obs 体现；若差不多 → 说明可以简化（也值得知道）。

## 2. 参照数字（判断「训对没有」的锚，全部实测）

> 2026-10-25 口径变更：**训练用 `--state-vel-weight 0`**（速度不进 `state` 正则）。
> 下表同时给两套读数：`state(vel=0)` = **现行口径**（pos+ori+q）；`state(vel=1)` = 历史口径（含速度地板）。

| 条件 token | **state(vel=0)**（现行） | state(vel=1)（历史） | rollout_reward | 说明 |
|---|---|---|---|---|
| **录制端口 token（oracle）** | **0.0063** | 0.398 | **+0.0786** | 上界参照 |
| 全零 token | 0.0263 | 1.600 | +0.0520 | — |
| 随机 token | 0.0266 | 7.905 | +0.0249 | 下界 |
| **基座 VLA token（起点，卡 6 实测）** | **0.0713** | 3.053 | **+0.0391** | 训练要填的 gap |

> ⚠️ **必读**：在 `state(vel=0)` 口径下，`state` 项对 token 质量**区分度很弱**（oracle 0.0063 vs 随机 0.0266，
> 零 token ≈ 随机），因为 8 步内位置/朝向主要由 policy 的动作决定、WM 运动学又很准。
> ⇒ **`state` 项现在只当"稳定正则"，主学习信号是 `rollout_reward`**（oracle +0.0786 vs 随机 +0.0249，3.2 倍区分度）。
> 起点 0.0713 与上面三行不是同一 batch（卡 6 的 eval batch），只能看趋势。

**分项（oracle，8 步，实测）**：`pos 2e-5（0.6 cm）/ ori 0.0037 / q 0.0026 / vel 0.372（只作诊断，不进 loss）`。

**旧 mjlab 栈对照（ft318，历史口径含速度）**：oracle `0.30 / +0.094`；训练 1000 步后 VLA `0.183 / +0.096`。
注意 ft318 近静止（\|dq/dt\| 0.16 rad/s），HA 动态（0.54–0.68），**绝对值不可跨栈比较**。

---

## 3. 主线联合训练（可直接跑）

### 3.1 启动（卡 6 的 A1 主线 / 任意 2 卡容器）
```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
# 默认值已改成新底模 + state-vel-weight=1.0（见 scripts/launch_joint_ha.sh 的 SM_CKPT / STATE_VEL_WEIGHT）
setsid nohup bash scripts/launch_joint_ha.sh > logs/joint_main.log 2>&1 &
# 训完自动接力 A1 闭环（可选）：
# TRAIN_PID=<pid> bash scripts/a1_pipeline_after_train.sh
```
等效的显式命令（便于核对）：
```bash
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29642 \
  scripts/train_joint.py \
  --motion data/npz/ha_tokens --motion-packed-dir runs/isaac_motion/ha_tokens \
  --supermimic-ckpt trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt \
  --vlm --vlm-path groot_repro/nvidia/Cosmos-Reason2-2B-weights \
  --vlm-ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 --vlm-dataset --vlm-steps 2 \
  --batch-size 2 --horizon 8 --steps 4000 \
  --wm-update-every 0 --state-vel-weight 0 \
  --hand-loss-weight 0.2 --hand-grasp-boost 5 \
  --eval-every 500 --eval-batches 2 --eval-out runs/joint_main/eval_curve.jsonl \
  --log-every 50 --save --save-every 0 --out runs/joint_main
```

### 3.2 期望与验收
| 指标 | 起点（eval 0） | 目标（1000–4000 步） |
|---|---|---|
| `state_term`（`state_vel=0` 口径：pos+ori+q） | **≈0.071** | 往 oracle **≈0.0065** 靠；1000 步内 < 0.04，4000 步内 < 0.02 |
| `state_vel`（分项，**仅诊断**） | ~2.9 | 不参与优化；只用来观察速度通道（见 §6.6） |
| `rollout_reward` | **+0.039** | 往 **+0.079** 走 |
| `token_mse`（端口空间） | 0.170 | **下降**（旧栈 0.062→0.030；若上涨说明条件空间又错位） |
| `(vla_XXXX.pt` / `joint_latest.pt`） | — | 两份都落盘；**`joint_latest.pt` 必须带 `wm_integration`** |

> **判读纪律**：`state(vel=0)` 对 token 质量的区分度很弱（见 §2 的 ⚠️），所以**不要把它当主指标**——
> 主指标是 `rollout_reward`（+0.039 → +0.079）和 `token_mse`（0.170 ↓）；`state` 只用来发现"想象发散"。
> 参考：旧 mjlab 栈 1000 步把 state 从 2.67 压到 0.183（那是含速度的历史口径）；我们 40 min（≈900 步）只能看趋势。

---

## 4. 纠错消融：每卡做什么

| 组 | 卡 | 现在能不能跑 | 入口 | 关键改动 |
|---|---|---|---|---|
| **A1** 有无联合微调（主结果） | 6 | ✅ 能（= 主线 4000 步 + Isaac 闭环） | `scripts/launch_joint_ha.sh` → `scripts/a1_pipeline_after_train.sh` → `scripts/run_a1_ha_isaac.sh` | 已指向 physint；两臂共用同一冻结 WBC |
| **A2** FM 反传 4 后端 | 0,1 | ✅ 能 | `bash scripts/launch_a2_fm.sh` | 已改新底模 + `state-vel-weight 0` |
| **A3** BPTT vs PPO | 2,3 | ✅ 能 | `MODE=all bash scripts/launch_a3_a5.sh` | 已改新底模 + `state-vel-weight 0`。⚠️ PPO 臂走 `diff_vla_wbc/ppo.py`（reward-only，score-function），**不接受** `--state-vel-weight`，这是正常的；两臂同墙钟、同 reward、同手部辅助损失即可 |
| **A4** WM 冻结 × horizon | 4,5 | ✅ **只能跑 `wm_frozen_H{8,16,32,40}` 四格** | `bash scripts/run_a4_wm_horizon.sh`（默认 `SKIP_WM_LIVE=1`） | 已改新底模 + `state-vel-weight 0` |
| **A4** `wm_live_H{8,40}` | 4,5 | ❌ **禁止**（`SKIP_WM_LIVE=0` 才会跑） | — | 见下 |
| **A5** 冻结 WBC 只训 head | 0,1（A2 之后） | ✅ 能 | `MODE=a5 bash scripts/launch_a3_a5.sh` 或 `bash scripts/launch_a5_only_head.sh` | 已改新底模 + `state-vel-weight 0` |

### 4.1 为什么 `wm_live` 必须暂停（这是唯一不能做的消融）

> **05:30 补**：`wm_live` 不是永远不能做——修好处方（§11.1）后，可以用**显式积分结构 + 正确帧**的底模做；
> 但**旧 `physint`（旧帧标定）不能**，见 §11.4。
joint 侧的 WM 更新走 `IsaacJointBundle.sample_windows`，里面的动作是**手算的参考动作**
```python
actions = (q[:, 1:] - self.default_joint_pos) / self.action_scale   # = Δq（手算）
```
而 rollout 里喂给 WM 的是 **policy 的真实动作**。两者分布不一致 →
① 破坏 policy↔WM 一致性（交接文档 §1③ 实测：WM 这样标定后 policy rollout 2.87→29.6）；
② 把速度标定洗掉（300 iter：corr 0.90→0.095）。
⇒ 恢复条件：把 `sample_windows` 的动作改成 `a = π(obs_ref, port_token)`，并加"参考速度辅助项"（见
`docs/HANDOVER_SUPERMIMIC_FINETUNE.md` §2.2.3）；在此之前 `wm_live` 一律 `SKIP_WM_LIVE=1`。

### 4.2 各组命令（复制即用）
### 4.3 A1 的臂怎么摆（**两条路线，不存在交叉**）
**铁律：联合微调的 VLA 只连 SuperMimic WBC，配 SONIC decoder 是错的**（token 空间不同，且从未这样设计）。

| 臂 | 路线 | VLA | 执行器 | 回答的问题 |
|---|---|---|---|---|
| **L1** | L1 官方线 | 官方监督 `checkpoint-8000` | **SONIC decoder**（`runs/sonic_decoder_final.pt`） | 官方基线成功率（headline 的一半） |
| **L2-ft** | L2 联合线 | **联合微调** VLA（奖励反传） | **SuperMimic WBC**（`physint`，WM 冻结） | 我们的完整线（headline 的另一半） |
| **L2-base**（可选、隔离用） | L2 联合线 | 官方 VLA + **纯监督 token 对齐**（对齐到装配空间） | 同上 | 只隔离出"奖励反传"的净效应 |

- **L1 vs L2-ft** = 论文 headline（两条完整路线之比；训练方式和执行器都不同，这是"我们的线 vs 官方线"，不是单项隔离）；
- **L2-base vs L2-ft** = 干净的"奖励反传有没有用"（同执行器、同数据、只差是否用奖励反传；L2-base 需要先做一次纯监督 token 对齐，见卡 6 文档 §2.C）；
- **现有脚本** `run_a1_ha_isaac.sh` 跑的是 `base`(=官方监督 VLA, 走 WBC) vs `ft`(=联合 VLA, 走 WBC)：它属于 **L2 内部**，若要严格当 L2-base，需要先把 base 的 token 头对齐到装配空间（否则 base 臂本身就带着 token 空间错配）。

```bash
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd $COMIMIC

# A1 主线（卡 6）：4000 步 + 自动接力
setsid nohup bash scripts/launch_joint_ha.sh > logs/joint_main.log 2>&1 &

# A2（卡 0,1）：4 后端 × 30 min
setsid nohup bash scripts/launch_a2_fm.sh > logs/a2_master.log 2>&1 &

# A3 + A5（卡 2,3）：BPTT 120 + PPO 120 + A5 两臂 40×2
setsid nohup bash scripts/launch_a3_a5.sh > logs/a3a5_master.log 2>&1 &

# A4（卡 4,5）：只跑 wm_frozen 四格（SKIP_WM_LIVE 默认 1）
setsid nohup bash scripts/run_a4_wm_horizon.sh > logs/a4_wm_horizon.log 2>&1 &

# A5（卡 0,1 的独立入口，等效）
setsid nohup bash scripts/launch_a5_only_head.sh > logs/a5_master.log 2>&1 &
```

---

## 5. 代码改动清单（**必须同步到各卡**，否则结果不可比）

| 文件 | 改动 | 为什么 |
|---|---|---|
| `diff_vla_wbc/isaac_joint.py` | `save_checkpoint` 写 `wm_integration` | 否则从 `joint_*.pt` 再加载会**静默退回残差结构**（权重是按显式积分训的） |
| `diff_vla_wbc/isaac_joint.py` | `COMIMIC_ACTION_SCALE=one` 默认（`a=Δq`）；奖励仍用 spec scale | 旧的多除一次 action_scale 是**漂移主因** |
| `diff_vla_wbc/isaac_joint.py` | 读 `ckpt["wm_integration"]` 自动启用显式积分；**加载时打印审计行** `[bundle] world-model structure = …` | 结构随 ckpt 走，且日志可审计 |
| `scripts/train_joint.py` | `--state-vel-weight`（默认 0.05）；`wm_integration` 透传；eval 新增 `state_pos/ori/q/vel` 分项 | 速度项可监控；分项避免被合成标量误导 |
| `scripts/train_joint.py` | eval 用 `EVAL_HORIZON=40` 抽帧 + `@8` 前缀指标 + `args.json` 参数指纹 | 跨格/跨 H 可比、可审计 |
| `trackerlab-supermimic/.../runner.py` | `import glob`（CSV 分支之前根本没被走过） | `--motion_dir` 路径会 `NameError` |
| `trackerlab-supermimic/.../runner.py` | resume 支持"评估导出件"（只载 pi/wm，critic+优化器新建） | `ckpt_*.pt` 不能直接 resume（缺 critic/opt） |
| `trackerlab-supermimic/.../runner.py` | `wm_only_iters` 改为 **相对 start_it**；`--iterations` 必须写**总迭代数** | 否则续训时 warmup 被跳过：policy 从第 1 iter 就更新 → 崩 |
| `scripts/launch_joint_ha.sh` / `launch_a2_fm.sh` / `launch_a3_a5.sh` / `run_a4_wm_horizon.sh` / `launch_a5_only_head.sh` / `run_a1_ha_isaac.sh` | `SM_CKPT → ckpt_010500_physint.pt`、`--state-vel-weight 0`、A4 默认 `SKIP_WM_LIVE=1` | 统一底模与口径 |
| `runs/OFFICIAL_SUPERMIMIC.json` | 指向 physint + 标定/参照数字 + **BONES-SEED 训练事实**（原写 HA 是从 run 名推的） | 唯一来源，防止再引用错底模 |

> 同步方式：各卡 `git` 不在版本里也没关系，直接比 `sha256sum` 这几个文件，或用共享盘同一份（都在
> `/inspire/.../comimic`，天然共享；只要别用旧快照 `runs/_code_*` 跑新口径）。

---

## 6. 已知坑（症状 → 原因 → 处理）

| # | 症状 | 原因 | 处理 |
|---|---|---|---|
| 6.1 | `NameError: name 'glob' is not defined` | runner 的 CSV 分支缺 import | 已修（§5） |
| 6.2 | 一启动就 `Resume needs last.pt...` | `ckpt_*.pt` 是导出件（无 critic/opt） | 已修：warm start 分支；或显式给 `last.pt` |
| 6.3 | `--iterations must exceed the resumed checkpoint iteration` | `iterations` 语义是**总数** | 写 `start + N`（例：10500+2500=13000） |
| 6.4 | 续训后 policy 立刻崩（`sim_r 0.52→0.13`） | `wm_only_iters` 按绝对迭代数判断 → warmup 被跳过 + 新 critic | 已修（相对 `start_it`）；且**必须**留 warmup |
| 6.5 | policy 学会"崩"（term 0.9）但 Isaac 起不来报 `No device could be created` | 5.1 venv 在 595 驱动上的渲染告警 / 用错栈 | HA 相关一律用 `third_party/isaacsim_env`（5.0）；渲染告警可忽略（PhysX 仍在 GPU，实测 0.69 it/s @4096 envs） |
| 6.6 | `state_term` 的读数/量级看着怪 | 它曾经是 pos+ori+q+vel 的合成标量（速度占 98%）；现在训练用 `state_vel=0` → 只剩 pos+ori+q | 分项字段 `state_pos/ori/q/vel` 都会打印；`state_vel` 只作诊断（地板 0.39 = mocap-FD vs sim 的口径差） |
| 6.7 | 训练后 `token_mse` 上涨 | 对标的是 HA 标签（跨空间） | 确认 `RecordedTokenSource` 走端口侧车；对端口 token 应该下降 |
| 6.8 | WBC 训练"用 PPO"的说法 | **错**：两套 SuperMimic 都是 **BPTT**（`J→loss=-J→backward`），critic 只做 bootstrap | 见 §9.4；项目里唯一的 PPO 是 A3 的对照臂（VLA head 的 score-function） |
| 6.9 | 把 physint ckpt 拿去 trackerlab 续训 | 会在 sim 转移上把速度标定洗掉 | 见 §1.3：WM 冻结，不续训 |

---

## 7. 起跑前自检（每条都要过）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

# 1) 底模结构/标定
python3 scripts/finetune_wm_velocity_policy.py \
  --ckpt trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt \
  --steps 0 --device cuda:0 2>&1 | grep -E "structure|before"
# 期望：[structure] ... explicit_dt0.02 ; qd_corr ≈0.90 ; ratio ≈0.93 ; policy state ≈0.4–0.6 / reward ≈+0.075

# 2) 端口 token 侧车齐全
ls data/port_tokens/ha_tokens/*.npy | wc -l      # 期望 699

# 3) 联合训练日志开头必须出现这一行（速度通道结构审计）
#    [bundle] world-model structure = explicit_dt0.02 (ckpt wm_integration='explicit_dt0.02')

# 4) 联合训练 2 步冒烟（约 3–4 min）
#    见 §3.1 的命令，把 --steps 4000 换成 --steps 2、--eval-every 1、--out /tmp/smoke
# 期望：eval0 state_term≈0.071（pos+ori+q；state_vel≈2.98 仅诊断）、token_mse≈0.17、rollout_reward≈+0.039；
#       无报错；joint_latest.pt 里 wm 与底模 max|Δ|=0（冻结）；日志有 [bundle] world-model structure = explicit_dt0.02

# 5) Isaac 5.0 栈（只在真跑 WBC/闭环时）
ls third_party/isaacsim_env/lib/python3.11/site-packages/isaacsim-5.0.0.0.dist-info
```

---

## 8. 旧结果处置（重要）
- **A2（19:24 那组）、A5（00:34 那组）、A4 pilot、A3 的 BPTT 单臂** 都是在**旧口径**下跑的
  （HA 标签 token + `state-vel-weight 0.05` + 旧动作换算 / 或旧底模）：**绝对数字作废**，
  仅保留"组内相对结论"作为参考（例如 A5：`head_plus_wbc` 三项都优于 `head_only`）。
- 重跑时**不要**覆盖旧目录（便于对照），用新目录名（各 launcher 已带时间戳或固定新路径）。
- `docs/ABLATION_A5_RUN.md` 顶部已加口径警告；A2/A3/A4 的报告在重跑后请同样加。

---

## 9. 附：本卡今天的关键实测（可复现）

### 9.1 速度标定（800 步，基于 `ckpt_010500`）
```
[structure] explicit_dt0.02
[800] wm qd_corr=0.9018 qd_ratio=0.9145 joint_mse=1e-4 pos=0.0
[after policy] state_term=0.5285 reward=+0.07589   →  产物 ckpt_010500_physint.pt
```

### 9.2 参照与起点（标定后 WM，实测）
| 条件 | state(vel=0) | state(vel=1) | rollout_reward | 分项 pos/ori/q |
|---|---|---|---|---|
| recorded(port) | **0.0063** | 0.398 | **+0.0786** | 2e-5 / 0.0037 / 0.0026 |
| zeros | 0.0263 | 1.600 | +0.0520 | — |
| random | 0.0266 | 1.905–7.905\* | +0.0249 | — |
| 基座 VLA token（卡 6，eval batch 不同） | 0.0713 | 3.053 | +0.0391 | 0.00015 / 0.0373 / 0.0338 |

\* random 在两次测量中为 1.905 / 7.905（不同随机种子与 batch）；趋势一致：rand 差、recorded 好。
**结论**：`reward` 有 3.2 倍区分度；`state(vel=0)` 只有 ~4 倍且零 token ≈ 随机 → 只当稳定正则。

### 9.3 联合训练 2 步冒烟（新底模）
```
[eval 0] token_mse=0.1698 state_term=3.0528 rollout_reward=+0.0391     ← 该次冒烟用 state_vel=1.0 跑
         （同一批数据在 state_vel=0 下的读数：state_term≈0.0713，state_pos 0.00015/ori 0.0373/q 0.0338/vel 2.981）
WM 冻结验证：max|Δ|=0.000e+00（20 张量）；policy 2e-5（policy_lr=1e-5）
```

### 9.4 两套 SuperMimic 都是 BPTT（不是 PPO）
- `trackerlab-supermimic/.../runner.py`：`J=(disc*c_t*r).sum(0)` → `loss=-J.mean()` → `loss.backward()`；
  全文无 `log_prob / ratio / importance`。
- `third_party/SuperMimic-Reproduced/src/supermimic/imagination/policy_trainer.py`：`loss=-J/horizon` → `backward()`。
- 项目里唯一的 PPO = `scripts/train_ppo.py`（A3 对照臂，针对 VLA head）。

### 9.5 为什么 calibration 会被"洗掉"
参考窗口（干净速度监督）vs PhysX 一步转移（速度被 `action_noise=0.1` 主导）——两个分布；
用后者继续训 300 iter（610 万样本）就回到均值场。所以：**标定 + 冻结 WM** 必须成对使用。

---

## 10. 联系人/决策记录
- 出卡：wscoli2（卡 0,1）。决策：走"路线 B"——**不重训 WBC**（不跑 trackerlab runner），
  直接 `physint` 底模 + 端口 token + 冻结 WM + BPTT 联合微调。
- "路线 A"（让 WBC 原生吃 HA 标签 token）需要先做**监督适配（BC on reference windows）**再 PPO，
  今天不做；不要再用"直接切 token + BPTT 续训"试（实测 policy 必崩）。

---

## 11. 补（2026-10-25 05:30，2×4090 卡实测）：WM 积分帧修正 + 底模可替换性矩阵

> 本节**不改**前面 §0–§10 的口径（端口 token / `a=Δq` / 数据 / eval 协议一律不动），只回答两个问题：
> ①"朝向通道坐标系"这个 bug 修了之后哪些数字会变；②**哪些格子可以换底模、哪些不能**。
> 一句话：**不要再"换标定"了**（旧/新标定差别可忽略）；要换只换"结构/底模"。

### 11.1 修掉的一个真 bug（朝向通道的坐标系）
- 事实：状态里的 `w` 是**世界系**（`tracker_env/supermimic/env.py:222 → data.root_link_ang_vel_w`），
  但 `algorithms/supermimic/networks.py` 的 explicit 分支写成 `R1 = R @ exp(dt*w + dθ)`——
  **右乘**指数里的 `dt*w` 必须是**体系**角速度（mjlab `state_spec.py` 的 `root_ang_vel_b` 约定）。
- 修法（成对）：`step`: `w_b = Rᵀw` → `R @ exp(dt*w_b + dθ)`；`targets`: `dθ = log(RᵀR1) − dt·w_b`。
- ⚠️ 原来的"代数互逆 1.19e-7"**对任何坐标系都成立**（`dθ` 会把误差吸收），证明不了坐标对。
  新验收 `scripts/check_wm_physics_interface.py` 会"咬人"（本卡全过）：
  互逆 `7.7e-7`；零输出 = **世界系**匀速外推 `1.2e-7`；旧写法 `R@exp(dt·w)` 差 `4.5e-2`（必须被抓住）；
  `state_error(S,S)=0`；`R(Rᵀw)=w` `4.8e-7`。

### 11.2 现在有 4 个底模，身份别混
| 名字 | 结构 | 数据 | 状态/用途 |
|---|---|---|---|
| `ckpt_010500.pt` | **残差**（速度无耦合） | BONES-SEED | 历史底模。实测 `qd_ratio 4.14 / root_w_corr 0.185 / rot_err 2.57°` → **新格子不要再用** |
| `ckpt_010500_physint.pt` | explicit（**旧帧**） | BONES-SEED + HA 窗口 800 步标定 | 卡 6 的 A1 正在用；**旧口径，已冻结，别重启那条进程** |
| `ckpt_010500_physint_fix.pt` | explicit（**新帧**） | 同上，05:11 用修好的公式重标 800 步 | 新口径等效底模；与旧版差别见 §11.3 |
| `logs/bones_seed_retrain_1025b/ckpt_*` | explicit（新帧） | **BONES-SEED 从零重训**（32000 iter，进行中） | 真正"学出来的物理"；A4 长 horizon / `wm_live` 的最终底模 |

### 11.3 旧标定 vs 新标定：差别可忽略（所以不需要"再换标定"）
同一协议实测（HA 端口 token、8 批 × 每批 8、H=8；`scripts/eval_wm_velocity.py`）：

| 指标 | `physint`（旧帧） | `physint_fix`（新帧） | 残差底模 `010500` |
|---|---|---|---|
| `qd_corr` | 0.7438 | 0.7438 | 0.0789 |
| `qd_ratio` | 0.9384 | 0.9385 | **4.14** |
| `root_w_corr` | 0.8122 | 0.8122 | 0.185 |
| `pos_err` | 1.01 mm | 1.01 mm | 4.44 mm |
| `rot_err`（单步） | 0.4998° | 0.4262° | **2.566°** |
| 想象 `state_term` | 0.00492 | 0.00433 | 0.05813 |
| 想象 `reward` | 0.07807 | 0.07821 | 0.06229 |

**结论**：真正拉开数字的是**"残差 → 显式积分"**（`qd_ratio` 4.14→0.94、`root_w_corr` 0.185→0.81、
`rot_err` 2.57°→0.43°、`reward` 0.0623→0.0782）；帧修正只值 0.07° / 12% `state_term`。

### 11.4 可替换性矩阵（"哪些能换、哪些不能"）
| 格子 | 能换底模？ | 换成什么 | 理由 |
|---|---|---|---|
| **A1**（H=8，base vs ft） | ✅ | `physint` 或 `physint_fix`，**二选一** | 两版差异远小于 20 ep 的抽样噪声（≈±20%）；**两臂必须同一个** |
| **A2**（FM 4 后端，H=8） | ✅ | 同上 | 臂内必须同一个 |
| **A3**（BPTT vs PPO，H=8） | ✅ | 同上 | 同上 |
| **A5**（head-only vs head+WBC，H=8） | ✅ | 同上 | 同上 |
| **A4 短 horizon**（H=8/16） | ✅ | 同上 | 朝向误差不累积到有害量级 |
| **A4 长 horizon**（H=32/40） | ❌ 旧 `physint` 不行 | `physint_fix`（过渡）→ **优先** BONES-SEED 重训 | 朝向误差按 `dt·|w_b−w_world|`/步累积，H 越大越偏 |
| **`wm_live`（world 不冻结）** | ❌ 旧口径不能做 | **BONES-SEED 重训底模**（`physint_fix` 只能当过渡） | 在线更新会把"混帧补偿"当物理洗掉（§1.3/§9.5） |
| 数据 / token 空间 / `a=Δq` / reward 配置 / eval 协议 | ❌ **不能换** | — | 这些不是"换底模"，换了就是另一个实验 |

三条硬规则：
1. **一次对比里只能有一个 WM**；`joint_latest.pt` / `vla_*.pt` 只能与产生它的那个 WM 比。
2. **卡 6 正在跑的 A1 不要重启**：它进程里是"旧代码 + 旧 ckpt"，自洽；重启会变成"新公式 + 旧标定"的混搭。
3. 换底模 = 换一整批；旧 `runs/ablations/*` 只能当"旧口径"归档，**不能和新口径混表**。

### 11.5 BONES-SEED 从头重训（进行中，回答"结构能不能学出速度"）
- 命令：`trackerlab-supermimic/scripts/rls/supermimic/train_bones_seed.sh`，
  `MODE=train NPROC=2 ENVS_PER_RANK=8192 ITERATIONS=32000` + `SUPERMIMIC_EXPLICIT_INTEGRATION=1`；
  数据 `data/bones_seed/bones_seed_compact.pt`（129,943 clips / 47,475,680 帧）；`--interface token --lr 2e-4`；
  日志 `logs/bones_seed_retrain_1025b/`；ETA ≈ 11 h。
- 它刻意**不加任何速度 loss 权重**（不加 `λ`），只靠"位置/朝向/关节要被预测对"来逼出速度。
- 验收点：**5k / 15k / 32k** 各跑一次 `scripts/eval_wm_velocity.py`，看 `root_w_corr`、`qd_ratio`、`rot_err`；
  判定 `root_w_corr > 0.9`、`qd_ratio ∈ [0.9,1.1]` 才算学到。若 5k 仍低 → 说明"光靠结构不够"，
  到时候再议（**不擅自加 loss 项**）。

---

## 12. 补（2026-10-25 05:53）：A2 / A5 / A3-BPTT 在 4 卡容器上起跑

> 状态板（谁在跑 / 跑完与否 / 结论）= **`docs/ABLATION_STATUS_20261025.md`**，本节只记口径与位置，不复述数字。

盘点结论（05:38 快照）：A5、A2、A3 的 **BPTT 臂**、A4 v6 的 **H16/H32/H40**、A1 的 **Isaac 闭环**都**没开始跑**；
真正在跑的只有 A1 主线训练、A3-PPO 臂、A4-v6 的 H8。

05:53 在 `comimic4`（4×49 GB）上按两条链起跑，每组内部唯一变量遵守 §4 的定义：

| 链 | 卡 | 内容 | 产物（新目录，旧结果不覆盖） |
|---|---|---|---|
| chain1 | 0,1 | A2 四后端 × 30 min → A5 `head_only`/`head_plus_wbc` × 40 min | `runs/ablations/A2_fm_backend_v2_physint/`、`runs/ablations/A5_only_head_v2_physint/` |
| chain2 | 2,3 | A3 **BPTT** 臂 120 min（与在跑的 PPO 臂同墙钟） | `runs/ablations/A3_bptt_vs_ppo/bptt` |

- 总控脚本（新）：`scripts/run_abl_1025_queue.sh`（串行 + 每臂重试 2 次，**不改训练语义**）；
  日志 `logs/run_1025_v2/`。
- 起跑指纹（两链 `eval 0` 完全一致，证明底模/数据/eval batch 同口径）：
  `token_mse 0.169811 / state_term 0.068489 / rollout_reward 0.040185 / hand_mse 0.65125`。
- **预取**：A2/A5/A3-BPTT 均为 `[prefetch] enabled=False`（`train_joint.py` 默认关，只有 `launch_joint_ha.sh`
  传 `--prefetch-vla`）⇒ 既避开 04:54/04:58 后台 PyAV 线程 SIGSEGV 的类别，又让 A3 两臂
  （PPO 也是 `--no-prefetch-vla`）预处理路径对称。
- **仍未开跑**：A4 v6 的 H16/H32/H40（在 `run_a4_v6.sh` 里排队，另一容器）、A1 的 Isaac 闭环
  （base 臂 5/18 episode 后被 `Terminated`，且 `runs/ckpt_vla_ha_ft4000` 里现在是 step-2000 权重，
  **必须等 `vla_004000.pt` 重新合并**）、`wm_live` 两格（§4.1 政策仍禁止）。
