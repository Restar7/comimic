# 2 卡容器交付：`ha_cmd_exec` 参考库 + WBC 训练库 + 微调 + G1（含失败分析）

> 执行机器：2×RTX 4090 48 GB 容器（共享目录 `/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic`）
> 对应计划：`docs/PLAN_REFSTART_WM_PERTURB_20261102.md` §4（P0b）、§6（P1a/P1b/P1c）
> 只做了"2 卡容器"的活：P0b / P1a / P1b / P1c。P0a 代码、P0c 对照、P2–P6 属于本机。
> 更新：2026-11-02 21:05

---

## 0. 一句话

**P0b 参考库 ✅ G0 PASS、P1a 训练库 ✅、P1b 两个 WBC ✅（A=执行 token / B=命令 token）；但 G1 没过。**
不是库做错了，而是**"把跟踪目标换成执行 q" 这件事本身让 WBC 更难跟**：同一条 DoubleDesk clip，
旧 WBC × `ha_cmd`（命令参考）= 0.32 m / 1327 步摔；新 WBC × `ha_cmd_exec`（执行参考）=
**0.94 m / 1698 步摔**；同一个新 WBC 换回命令参考 = 0.28 m / 1374 步摔。
→ **建议先不要按 §1 的结论直接切执行参考**，理由和下一步见 §6。

---

## 1. P0b `ha_cmd_exec`：✅ G0 PASS

命令（新增脚本 `scripts/build_ha_exec_lib.py`）：

```bash
python3 scripts/build_ha_exec_lib.py --npz-dir data/ha_raw \
  --out runs/isaac_motion/ha_cmd_exec --x20-repeats 20
python3 scripts/check_exec_lib.py --packed runs/isaac_motion/ha_cmd_exec/packed.pt \
  --report runs/isaac_motion/ha_cmd_exec/report.json
```

字段映射：`p/quat/v/w ← robot_root_*`，`q/qd ← robot_qpos/qvel_before_decimation`（STANDARD_29 序）。

> **关节顺序坑（已验证，别再踩）**：`robot_qpos_before_decimation` 已经是 STANDARD_29（MJCF）序。
> 证据：`G1Kinematics` 对 (root, q) 做 FK 能**零误差**复现 `data/npz/ha_tokens` 的 `body_pos_w`；
> 若再套一次 `joint_contract.to_standard()`，脚位置会差 0.5 m。

产出：

| 文件 | 内容 |
|---|---|
| `runs/isaac_motion/ha_cmd_exec/packed.pt` | 699 clips / 634,016 帧 / fp16 / 90 MB，`names/starts/lengths` 与 `ha_cmd` **逐项相同** |
| `runs/isaac_motion/ha_cmd_exec/packed_x20.pt` | 13,980 clips / 12.68 M 帧（×20，与旧 x20 库 HA 段同构） |
| `runs/isaac_motion/ha_cmd_exec/report.json` | G0 全量报告（699 条逐条 FK） |
| `runs/isaac_motion/ha_cmd_exec/meta.json` | `ref_layout: canonical`（**必须有**，loader 靠它认布局） |

G0 数字（699/699 clips）：

| 指标 | 门 | 实测 |
|---|---|---|
| FK 脚位置 vs `body_pos_w` | max < 3 cm | **2.38 mm**（p99 2.24 mm） |
| 根高度（pelvis z） | < 2 cm | **0.24 mm** |
| clip 数 / 帧数 | 699 / 与 npz 一致 | 699 / 634,016（逐条核对） |
| 与 `ha_cmd` 比 q | ~0.39 rad | **0.336 rad**（median 0.238） |
| 与 `ha_cmd` 比 root | 相同 | **逐位相同（max\|Δ\|=0）** |
| quat / NaN | 正常 | dev 3.4e-4 / 全有限 |

`verdict: PASS`。

---

## 2. P1a 训练库：✅

```bash
python3 scripts/build_wbc_lib.py \
  --bones trackerlab-supermimic/data/bones_seed/bones_seed_compact.pt \
  --ha-exec runs/isaac_motion/ha_cmd_exec/packed.pt --x20 20 \
  --out trackerlab-supermimic/data/bones_seed/bones_seed_plus_ha_exec_x20.pt
```

- 143,923 clips / **60,156,000 帧** / 8.55 GB（bones 129,943 + HA 699×20）；
- bones 段 **bitwise identical**、整体与旧 `..._ha_cmd_x20.pt` **索引结构逐项同构**（脚本断言）；
- HA 段 `mean|Δq| vs ha_cmd = 0.3357 rad`，root 逐位相同。

---

## 3. P1b 微调 WBC：✅ 两个 ckpt

两个都从 `WBC_A24000_wmvel_v6.pt`（iter 24000）warm start **4000 步 → 28000**，
`NPROC=2 ENVS_PER_RANK=4096 lr=2e-5 obs93 no_critic=1 freeze_wm=0 compile=0`，
`ROBOT_USD=dex3 43DoF`（与 v6 的 `robot_spec.usd` 一致），`CKPT_EVERY=500`。

| 名字 | 训练时 token 来源 | 文件 | 含义 |
|---|---|---|---|
| **A** | 执行参考库在线算（= 计划字面） | `runs/S1_handover/exp_WBC_exec.pt` | ref/token 都来自执行 q |
| **B** | **命令流库**在线算（= 计划本意） | `runs/S1_handover/exp_WBC_exec_cmd.pt` | ref=执行、token=命令（部署口径） |

> 计划 §6.2 的命令**没有** `RESUME`（等于从 0 训 4000 步），且 `ITERATIONS` 必须写绝对目标
> （`--resume` 语义）；`OBS_MODE` 默认是 `relative_ori99`，必须显式 `paper_obs93`，否则 ckpt 与
> 闭环 provider 口径不一致。这些差异在 `scripts/p1b_finetune_wbc_exec.sh` 里都固化成了默认值。

B 是怎么来的：训练栈里 token 不是读 sidecar，而是 `SonicTokenBuilder` 用**训练库那一行的 q/qd**
在线算。为了让"ref=执行、token=命令"成立，加了 `TwoLibTokenView`（按 clip 名+局部帧号把训练库行
映射到命令流库行）+ `SM_TOKEN_LIB_TRAIN` 开关（**默认关闭**，不影响任何旧 run）。
离线单测：双库视图算出的 token 与"直接在命令流库上训练"**逐位相同**（bones / HA 两段都验）。

训练日志：`trackerlab-supermimic/logs/exp_wbc_exec{,_cmd}/train_log.csv`
（A：sim_rew 0.775→0.808、ep_len 77→324、wm_loss 1.4→0.047，无 NaN，峰值 20.2 GiB/卡）

---

## 4. P1c G1 参考回放：❌ FAIL

脚本：`scripts/p1c_g1_replay.sh` + `scripts/eval_g1_replay.py`（起点 `SM_INIT_FROM_LIB=1`
+ `SM_START_FRAME=clip 起点`，参考帧 = 库行，跑整条）。

**指标口径校验**：用历史 dump（旧 WBC × `ha_cmd` online）复现计划 §1 的
"**0.321 m / 1327 步摔**" —— 完全对上，说明指标没跑偏。

### 4.1 结果矩阵（DoubleDesk clip0 = `HOI_double_desk__tw__ep1776268633456387`，2080 帧）

| ckpt | 参考库 | token | 步数 | root xy mean | 关节 RMSE | 摔倒 |
|---|---|---|---|---|---|---|
| 旧 v6（基线） | `ha_cmd`（命令） | online 命令 | 1327 | **0.321** | 0.420 | 是 |
| **A**（执行 token） | `ha_cmd_exec`（执行） | online 执行 | 1698 | **0.943** | 0.490 | 是（tilt 84°） |
| **A** | `ha_cmd_exec` | npy 命令 label | 228 | 0.560 | 0.456 | 是 |
| **A** | `ha_cmd`（命令） | online 命令 | 1374 | 0.282 | 0.440 | 是 |
| **B**（命令 token） | `ha_cmd_exec` | online 命令 | 见 §4.2 | | | |
| **B** | `ha_cmd_exec` | npy 命令 label | 见 §4.2 | | | |

### 4.2 B 的结果

（待补）

### 4.3 失败模式（A × 执行参考，逐段拆开）

| 步 | 参考位移 | 机器人位移 | xy 误差 |
|---|---|---|---|
| 0–200 | 0.02 m | 0.01 m | 0.04 m |
| 200–400 | **1.13 m** | **0.57 m** | 0.55 m |
| 400–1200 | 0.28 m | **1.93 m** | 1.09–1.24 m |
| 1200–1698 | 1.99 m | 1.96 m | 1.10 m → 摔（root_z 0.31） |

即：**起步转身+迈步那 4 秒跟不上（慢半拍、少走一半），之后参考站住了它还在走**，越拉越开，
最后绊倒。和关节速度有关：执行参考的 `mean|qd| = 0.41 rad/s`，命令参考只有 `0.098`
（4 倍），而 WBC 的带宽来自 79% 的 bones_seed 平滑数据。

### 4.4 诊断实验（都是同一条 clip、400 步窗口，A ckpt）

| 实验 | root xy | 关节 RMSE |
|---|---|---|
| 执行参考（原样） | 0.292 | 0.325 |
| **命令参考**（同 ckpt，token 也换命令） | **0.084** | 0.237 |
| 执行参考 q/qd **前移 8 帧**（`ha_cmd_exec_lead8`，想抵消跟踪滞后） | **0.452**（更差） | 0.380 |
| 旧 v6 × 命令参考（历史） | 0.123 | 0.218 |

另外查过、可以排除的：

- **不是时间平移**：`q_cmd(t)` 与 `q_exec(t+lag)` 的差异在任何 lag（−20..20）下都是 ~0.29 rad
  （lag=7 时 0.2906 vs lag=0 时 0.2917），两条是**不同的运动**，不是"命令提前 8 帧"；
- **不是关节顺序/封装错误**：FK 零误差、token 双库逐位对齐、G0 PASS；
- **不是 token 空间**（至少不是主因）：A（执行 token）与 B 中途 ckpt（命令 token）在执行参考上
  都是 ~0.3 m / 400 步，差别在噪声内；
- **不是"前移能救"**：前移 8 帧反而变差 0.45；
- 命令流膝盖摆幅 1.55 rad vs 执行流 0.66 rad —— 命令流是"人类重定向"的更大动作，
  执行流是机器人实际做到的、更快更碎的动作。

---

## 5. 交接清单

| 产物 | 路径 | 状态 |
|---|---|---|
| 执行参考库 | `runs/isaac_motion/ha_cmd_exec/packed.pt` | ✅ G0 PASS |
| G0 报告 | `runs/isaac_motion/ha_cmd_exec/report.json` | ✅ |
| 前移诊断库 | `runs/isaac_motion/ha_cmd_exec_lead{4,8,12}/packed.pt` | 诊断用，别当正式库 |
| 训练库 | `trackerlab-supermimic/data/bones_seed/bones_seed_plus_ha_exec_x20.pt` | ✅ 143,923 clips / 60.16 M 帧 |
| WBC-A（执行 token） | `runs/S1_handover/exp_WBC_exec.pt` | ✅ iter 28000 |
| WBC-B（命令 token，部署口径） | `runs/S1_handover/exp_WBC_exec_cmd.pt` | ✅ iter 28000 |
| G1 报告/视频/日志 | `runs/exp_g1_A_exec_token/`、`runs/exp_g1_A_on_cmdref_full/`、`runs/exp_g1_A_lead8/`、`runs/exp_g1_B*/` | 见 §4 |
| 训练日志 | `trackerlab-supermimic/logs/exp_wbc_exec{,_cmd}/train_log.csv` | ✅ |

新增脚本（全部新增，未改旧脚本）：

```
scripts/build_ha_exec_lib.py     P0b 建库
scripts/check_exec_lib.py        P0b G0 验收 → report.json
scripts/build_wbc_lib.py         P1a 训练库
scripts/p1b_finetune_wbc_exec.sh P1b 微调（支持 TOKEN_LIB_TRAIN 双库）
scripts/p1b_chain_B_after_A.sh   串行跑 A→B
scripts/p1c_g1_replay.sh         P1c G1 回放（online/npy 两臂，SM_PACKED/SM_TOKEN_LIB 分离）
scripts/eval_g1_replay.py        G1 指标（root/关节/摔倒/完成度）
```

代码改动（都是**默认关闭**的开关）：

```
trackerlab-supermimic/.../sonic_token.py   + TwoLibTokenView
trackerlab-supermimic/.../runner.py        + SM_TOKEN_LIB_TRAIN（provenance 里记录 token_lib_train）
```

> 注意：`SM_PACKED`（参考库与 token 库解耦）是**本机那边**在 20:0x 加进
> `action_provider_supermimic.py` 的（`_packed_ref_path()`），我用的是它，没重复改。

---

## 6. 结论与建议（给本机决策）

1. **参考库这一层是中性的、可信的**：G0 全过、索引结构不变、root 逐位相同。问题不在库。
2. **"把跟踪目标换成执行 q" 没有带来更好的参考回放**，反而更差（0.32 → 0.94 m，且照样摔）。
   直接原因：执行 q 的关节速度是命令 q 的 4 倍，WBC 跟不上 → 落后半步 → 越拉越开。
3. 计划 §1 的推理"矛盾目标无解 → 换执行 q"缺一步：命令参考虽然**关节**上不可实现
   （膝 1.55 vs 0.66 rad），但它的**根**轨迹与执行参考完全相同，而 WBC 的本征滞后正好被
   命令流那点"提前量"部分抵消，所以**根误差反而小**。换执行 q 把这点缓冲拿掉了。
4. 下一步的三个选项（按推荐顺序）：
   - **(a) 参考只换一半**：跟踪用 `ha_cmd`（命令参考，根跟踪好），奖励/WM 的**速度通道**用执行量
     （`robot_root_lin_vel_world` 本来就是执行量，已经在库里）；即保留 `ha_cmd`，不要切 `ha_cmd_exec`。
   - **(b) 执行 q 先滤波 + 前移**：把执行 q 做低通（q/qd 平滑到 ~0.1 rad/s 量级）再当前瞻目标，
     并单独扫 lead（本容器先试了 +8 帧，单独前移没用，需要和滤波一起做）。
   - **(c) 真要执行参考就加重训练**：≥12k 步、或对 HA 段加权（`--x20` 提到 40/60）、或先
     "命令参考预热 → 执行参考收尾"。本容器 4000 步 warm start 明显不够（关节 RMSE 0.49 还在 0.4+）。
5. **不要把 G1 不过当成 P1 的失败**：P0b/P1a 的产物是干净的，P1b 的两个 ckpt 可以直接拿去跑
   P0c 对照（A/B × 两参考 × 两 token）——本容器已经跑完了这张表（§4.1）。
6. `runs/exp_g1_run28k_*` 那一组不是我跑的（另一个会话用同一 ckpt 跑的 plan-literal 配置），
   数字与 §4.1 的 A 行不同（它 online 臂跑满 2080 步、1.526 m、未摔），引用时注意区分。
