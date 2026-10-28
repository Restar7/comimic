# S3 联合微调：**端到端执行方案**（2026-10-28 定稿版）

> 目的：把"从采集到联合微调再到验收"的完整流程写死，保证按本文执行不会踩已知的坑。
> 配套文档：
> * `docs/TOKEN_SPACE_ANCHOR_LOCKDOWN_20261028.md` —— token 空间 / anchor / 真物理插桩的**结论锁定**
> * `docs/S3_FIX_PLAN_AFTER_REVIEW_20261027.md` —— 外部评审提出的算法问题清单（本文给出每项的现状）
>
> **规则**：本文里每个数字都是本机实测，不是估计；任何执行阶段只认"验收标准"，不认"看起来涨了"。

---

## 0. 一句话流程

```
[0] 预检（一次性）        真转移 + WM 保鲜 + critic 训练 + A1 验证
        ↓
[1] B 线采集              真物理闭环跑 5 条 episode，dump (s, a_applied, s', tokens, 真奖励)
        ↓
[2] 用采集数据刷新 WM      build_real_transitions_v2 → finetune_wm → 验收 qd_corr
        ↓
[3] 训练 critic           用"录制 634k 帧 + B 线漂移帧"算 return-to-go → 验收 v_ev > 0.3
        ↓
[4] 联合微调              想象闭环里反传：λ_r·(−Σγ^t r) + λ_hand·手部，token 不做 BC
        ↓
[5] 闭环验收              B 线跑训练后的 head → 与基线比 r / 根误差
        ↓
   涨 → 回到 [1]（用新策略采）    不涨 → 止损，换目标（任务成功）
```

一轮（第一轮）约 40 分钟；后续轮约 20 分钟。

---

## 1. 锁定的口径（**不得更改**，详见 lockdown 文档）

| 项 | 值 |
|---|---|
| VLA 起点 | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000` |
| 执行器 | `runs/S1_handover/WBC_A24000_wmvel_cmd.pt`（`hold / paper_obs93 / bones_seed_plus_ha_cmd_x20 / 64`）|
| VLA 监督标签 | npz `g1_motion_token_64`（`--bc-token-space label`）；**禁止**用 `data/port_tokens` |
| 执行器条件 token | 参考库**在线**算（`window=hold` + 实时 anchor）|
| 参考库 | `runs/isaac_motion/ha_cmd/packed.pt`（命令流）|
| 真物理 token 喂养 | GT 臂用 `SM_TOKEN_SOURCE=online`；VLA 臂用 `SM_TOKEN_SOURCE=http` |
| VLA 服务 | **`scripts/ha_vla_latent64_server.py`**（已验证：6 任务 0 摔）。**不要用** `scripts/vla_token_server.py`（未验证，会摔）|
| 闭环步数上限 | 官方部署口径 `HA_VLA_EXEC_STEPS=20`；**不要**用 40 当基线（40 是更宽松的条件）|

---

## 2. 现状清单（2026-10-28）

### 2.1 已经做对、可以直接用的
| 项 | 证据 |
|---|---|
| token 空间对齐（VLA 标签 ≈ 执行器条件，MSE 0.00069）| lockdown F3 |
| 真物理插桩修好（online + 实时 anchor，根误差 1.0 cm）| lockdown F6 |
| B 线闭环打通（http 实时推理，104 次查询/2080 步）| `runs/B_baseline_2080` |
| 采集 dump 字段齐全（`q_target_applied` / `tokens` / `frames`=真库行号 / meta）| 同上 dump |
| 参考执行动作库（634,016×29，覆盖 100%）| `runs/wm_real/ref_action_lib.pt` |
| P1-1（critic 梯度穿 s_H + 相对形式）| `--critic-grad-through-state` / `--critic-rel-clip` |
| P1-4（hinge 中心换底模）| `--trust-center base` / `--act-hinge-cap` |
| P1-6/A1（burn-in 状态写回 VLA）| `--vla-state-from-burnin` |

### 2.2 **还没做**（本文要做的）
| 项 | 状态 | 阻塞第 1 轮？ |
|---|---|---|
| **P0-3** 用正确动作标签重建真转移 → WM 保鲜 | 没做（`build_real_transitions.py` 仍用 `q_cmd` 反推）| **是** |
| critic 重新训练（真实 return-to-go）| 没做（现有 `v_ev` 为负：−0.47/−1.29/−1.84）| 否（第 1 轮先关） |
| 已验证 server 支持 **换 action head**（评测 S3 产出的 head）| 没做 | 否（第 5 阶段前必须） |
| VLA 臂"随机起始帧"（采集覆盖）| 没做 | 否（第 2 轮起有意义） |
| P1-5 bf16 vs fp32 对拍 | 没做 | 否 |
| P0-1 批量多窗口评测器 | 没做（用 B 线闭环替代）| 否 |

### 2.3 基线数字（第 1 轮要打败的）
```
闭环（24k + 有手 + EXEC_STEPS=20 + 已验证 server，2080 步）：
    不摔（timeout）；env 判定 success=False
    r_mean(整段) = +0.05108；前 300 步 = +0.04844；末 600 步 = +0.04623
    关节 RMSE vs 参考 = 0.399 rad
GT 上限（参考库现算 token，300 步）：
    r_mean = +0.06935；根误差 0.010 m
⇒ 闭环差距：前 300 步 -30%
```

---

## 3. 需要写的代码（3 处，都很小）

### 3.1 `scripts/build_real_transitions_v2.py`（**必须，P0-3**）
用**真正下发的动作**重建真转移，一次产出两样东西（WM 保鲜数据 + critic 监督目标）。

```
输入：runs/isaac_motion/ha_cmd/packed.pt 的 states（634,016 帧，命令流）
      runs/wm_real/ref_action_lib.pt     （decoder_target_action，canonical 序）
      B 线 dump（可选，`--extra-dumps 'runs/B_*/dump/*.pt'`，用 q_target_applied）

动作换算（WM 的动作语义 = 归一化 q_target）：
      a_t = q_target_applied_t − default_q          （action_scale = 1.0，实测）
      录制侧：a_t = ref_action_lib.actions[t]        （已是 canonical、asset→canonical 已转）

每步奖励：bundle.reward(state_t, a_t, a_{t-1}, FrameRef(t))   （与训练/评测同一个函数）

输出 runs/wm_real/real_transitions_v2.pt：
      states (N,71) / actions (N,29) / next_states (N,71) / frames (N,) / clip_ids (N,)
      rewards (N,) / ret (N,) / ref_rows (N,71)      ← ret/ref_rows 给 critic 用
```
验收：
- `actions` 与 `q_target_applied − default_q` 口径一致（抽查 10 帧，误差 < 1e-5）
- `frames` 是真实库行号，`next_states[f] == states[f+1]`（除 clip 末帧）
- 覆盖 100%（`valid` 全 True）

### 3.2 给已验证 server 加 `--head-ckpt`（**第 5 阶段前必须**）
`scripts/ha_vla_latent64_server.py` 现在只能加载完整 checkpoint，无法替换 action head 权重。
改动（~10 行）：`--head-ckpt` 指向 S3 的 `joint_*.pt`，加载后 `model.action_head.load_state_dict(sd, strict=False)` 并打印 `missing/unexpected`。
**不要**改它的 state46 构造与手部逻辑（那是已验证的部分）。

### 3.3（可选，第 2 轮再做）VLA 臂随机起始帧
`sim_eval_vla.py` 增加 `--ref-start-frame`，让 env 的参考从任意帧开始；
GT 臂已有 `SM_TOKEN_START`，不需要改。

---

## 4. 执行阶段（逐条命令 + 验收）

### Stage 0｜预检（一次性，约 40 分钟）

**0.1 建真转移（3.1 的脚本）**
```bash
PYTHONPATH=. python3 scripts/build_real_transitions_v2.py \
    --out runs/wm_real/real_transitions_v2.pt
```
验收：见 3.1 的三条。

**0.2 WM 保鲜**（用刚建的真转移，**动作标签已正确**）
```bash
PYTHONPATH=. SUPERMIMIC_HA_ALIGN=1 python3 scripts/finetune_wm_velocity_policy.py \
    --ckpt runs/S1_handover/WBC_A24000_wmvel_cmd.pt \
    --out  runs/S1_handover/WBC_A24000_wmvel_v2.pt \
    --packed runs/isaac_motion/ha_cmd/packed.pt \
    --token-space label --steps 800 --batch 256 --lr 1e-4 --vel-lambda 5
```
**验收（硬门槛）**：用同一个仪器比 `qd_corr`
```bash
for c in runs/S1_handover/WBC_A24000_wmvel_cmd.pt runs/S1_handover/WBC_A24000_wmvel_v2.pt; do
  PYTHONPATH=. python3 scripts/eval_wm_velocity.py --ckpt $c \
      --packed runs/isaac_motion/ha_cmd/packed.pt --tokens label --tag $(basename $c)
done
```
- 门槛：`qd_corr ≥ 0.70`（C6h 基线 0.752）
- **不到 0.70 → 回滚，用原 ckpt**（上次用错动作标签刷新掉到 0.589，就是这条没验）

**0.3 训练 critic（可与 0.2 并行，另一张卡）**
```bash
# 用 0.1 的 ret/ref_rows/states 直接喂给 --critic-real（格式已兼容）
PYTHONPATH=. python3 scripts/build_critic_return_targets.py \
    --inputs 'runs/wm_real/real_transitions_v2.pt' \
    --out runs/wm_real/critic_mc_v2.pt --gamma 0.99
```
验收：在留出窗口上算 `v_ev`（`train_joint_v2.py` 会把 `critic/v_ev` 打进日志）
- **门槛：`v_ev > 0.3` 才允许在第 2 轮启用 critic**；否则永久关闭
- 现有 critic 的 `v_ev = −0.47/−1.29/−1.84`（比均值还差），**第 1 轮必须 `--no-critic`**

**0.4 A1 验证（10 分钟）**
同 seed 跑两个 30 步 run：`--vla-state-from-burnin` 开/关，比 `imag_return_norm` 和 `token_shift`。
- 若两者**完全一致** ⇒ A1 没生效（bug），必须先修，否则 VLA 永远看不到偏差
- 验收：开的那个 `token_shift` 或 `imag_return` 有可测变化

### Stage 1｜B 线采集（5 条 × 400–600 步，约 10 分钟）
```bash
# 每条：不同 clip（或同 clip 不同 seed），必须落盘绝对路径
OUT=$PWD/runs/B_r1_$(date +%H%M); mkdir -p $OUT/dump
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  TASK=<DoubleDesk|PPBox|Boxing|OpenDoor|SitSofa> SEED=0 MAX_STEPS=600 ISAAC_GPU=0 \
  MODEL_LABEL=vla MODEL_LABEL=$RUN OUT_ROOT=$OUT/runs \
  SM_CKPT=$PWD/runs/S1_handover/WBC_A24000_wmvel_v2.pt \
  SM_TOKEN_SOURCE=http SERVER_URL=http://127.0.0.1:18080 SM_HAND=1 \
  SM_DUMP_STATES=$OUT/dump/<task> \
  HA_VLA_EXEC_STEPS=20 \
  bash scripts/ha_isaac_closedloop.sh > logs/$OUT/$(basename $OUT)_<task>.log 2>&1
```
采集后立刻补算真奖励（与训练同一个函数）：
```bash
superMIMIC_HA_ALIGN=1 PYTHONPATH=.:third_party/SuperMimic-Reproduced/src \
  python3 scripts/compute_collect_reward.py --in $OUT/dump/*.pt --out $OUT/scored --device cuda:1
```
**采集验收（三条同时满足）**
1. 每条都落盘，`states` 行数 = 实跑步数（不是 0）
2. `q_target_applied` 非空、`tokens` 非空、`frames` 连续
3. **不摔的比例**：至少 4/5 条是 timeout（摔了的那条照样入库，但标记出来）

### Stage 2｜用新数据刷新 WM（约 20 分钟）
重复 0.1/0.2，但把 `--extra-dumps` 指向 Stage 1 的 dump（on-policy 漂移数据进真转移）。
验收同 0.2：`qd_corr ≥ 0.70` 且**不比上一版低 0.03 以上**。

### Stage 3｜critic 目标刷新（约 5 分钟）
同 0.3，`--inputs` 同时给录制侧和 Stage 1 的 `scored/*_r.pt`。
验收：`v_ev > 0.3`（达不到就不启用）。

### Stage 4｜联合微调（约 10 分钟）
```bash
GPU=0 STEPS=300 LAMBDA_R=0.3 LAMBDA_BC=0 LAMBDA_HAND=0.2 \
RUN=s3_r1 BATCH=4 ACCUM=8 H=20 K=4 EVAL_EVERY=100 EVAL_BATCHES=8 SAVE_EVERY=100 \
WBC_CKPT=$PWD/runs/S1_handover/WBC_A24000_wmvel_v2.pt \
EXTRA="--amp bf16 --lora-rank 32 --fm-lr 1e-4 --no-dr \
       --vla-state-from-burnin --no-critic \
       --trust-center base --act-hinge-cap 2 --act-eps 0 \
       --lambda-aref 0" \
bash scripts/launch_joint_ha_v2.sh
```
说明：
- `--lambda-bc 0`：**不做 token BC**（按用户口径）
- `--lambda-hand 0.2`：手部保留 hand-only BC（唯一例外，否则手无监督）
- `--no-critic`：第 1 轮关（v_ev 为负）；第 2 轮若达标再开
- `--no-dr`：DR 对 VLA 不可见，只注入噪声
- `--lambda-aref 0`：动作误差项暂不作为主/辅目标（代码在，默认关）

**训练验收（三条，缺一不可）**
1. 无 NaN；`codec/out_of_range` 不随步数单调上升
2. `token_mse` 变化 ≤ 1.5×（告警线；超过要查，别当成功）
3. `imag_return_norm` 上升（**但不作为结论**，只作为"训练没崩"的迹象）

### Stage 5｜闭环验收（约 5 分钟/条）
先在 server 上加载训练后的 head（3.2 的 `--head-ckpt`），然后跑 B 线：
```bash
# 新 server 进程（与基线同一个脚本，只多 --head-ckpt）
python3 scripts/ha_vla_latent64_server.py \
  --ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --head-ckpt runs/s3_r1/joint_000300.pt --device cuda:0 --port 18081
# 评测（与基线同一任务、同一 seed、同一 MAX_STEPS）
env -u LD_LIBRARY_PATH -u LD_PRELOAD TASK=DoubleDesk SEED=0 MAX_STEPS=600 ISAAC_GPU=0 \
  MODEL_LABEL=s3r1 OUT_ROOT=$PWD/runs/B_eval_r1/runs \
  SM_CKPT=$PWD/runs/S1_handover/WBC_A24000_wmvel_v2.pt \
  SM_TOKEN_SOURCE=http SERVER_URL=http://127.0.0.1:18081 SM_HAND=1 \
  SM_DUMP_STATES=$PWD/runs/B_eval_r1/dump/DoubleDesk_s3r1 HA_VLA_EXEC_STEPS=20 \
  bash scripts/ha_isaac_closedloop.sh
```
**验收（唯一判据）**
```
闭环 r_mean 相对基线提升 ≥ +5%（基线前 300 步 0.0484）
且 根误差不增大、不摔
```
提升 < 5% → 判为"无实质改进"。

---

## 5. 决策门（写死，避免反复）

| 门 | 判据 | 不通过怎么办 |
|---|---|---|
| G0-a | WM 保鲜后 `qd_corr ≥ 0.70` | 回滚 WM，`--extra-dumps` 去掉 on-policy 数据重试一次；仍不行就不用新鲜转移 |
| G0-b | A1 有可测影响 | 修 A1（否则第 4 阶段意义有限）|
| G0-c | critic `v_ev > 0.3` | 第 1 轮不用 critic（不阻塞） |
| G1 | 采集 4/5 条不摔 | 检查手部限幅/`EXEC_STEPS`，别带着摔的数据训练 |
| G4 | 训练三条验收 | 不通过就调 `fm_lr` / 降 `lambda_r`，不要硬跑 |
| **G5** | **闭环 r 提升 ≥ 5%** | **止损**：第 2 轮若仍无提升，改目标（任务成功信号 / 动作误差低权重辅助项）|

---

## 6. 明确**不做**的事（已排除的坑，别再试）

| 不做 | 原因 |
|---|---|
| 用 `data/port_tokens` 当 VLA 监督或执行器条件 | 它是执行轨迹编码的诊断文件，MSE 0.021/0.11，与训练口径差 30 倍 |
| 用预计算 npy 做 GT 臂真物理回放 | 冻 anchor ⇒ 根漂 3.49 m；必须 `SM_TOKEN_SOURCE=online` |
| 把 `HA_VLA_EXEC_STEPS` 设成 40 当基线 | 40 是更宽松条件；官方部署是 20 |
| 用 `scripts/vla_token_server.py` | 未验证，实测同配置下 166 步摔 |
| 开 DR / push | VLA 观测不到扰动（①），只注入噪声 |
| 用 critic 之前不验 `v_ev` | 现有 critic `v_ev < 0`，带着它等于注入错误自举 |
| 把"想象 return 涨"当成功 | 必须由 Stage 5 闭环验收 |
| 给 VLA/WBC 加 yaw / ω_plan（暂时）| 现无证据表明朝向是瓶颈；等 G5 通过后作为鲁棒性增强再评估 |

---

## 7. 已知债务（不阻塞，但记录在案）

1. **P1-5 未做**：bf16 是"按 GR00T 原生 dtype"选的，未做 fp32 对拍。若出现"想象与真实系统性不一致"，第一个怀疑它。
2. **P1-3 未做 A/B**：因为已有 `v_ev < 0` 的硬证据，直接关 critic，省了 A/B。
3. **P0-1 批量评测器未建**：用 B 线闭环评测替代（更真实，但更贵；每条 600 步 ~3 分钟）。
4. **① 图像通道无解**：想象中的 VLA 看不到图像，所以"纯视觉纠偏"学不到；A1 只补本体感觉通道。
5. **`frames` 口径**：online 模式=真实库行号；npy 模式=步数。所有新采集都必须走 online/http，不要混。
6. **并行采集未做**：现在单 env ≈ 3.6 步/s。方案 A（多进程各跑不同 clip，共用一个 server）零代码改动，显存够（GPU0 空闲 40 GB）；等 G5 通过后再做提速。

---

## 8. 时间预算

| 阶段 | 第一轮 | 后续轮 |
|---|---|---|
| Stage 0 预检 | 40 min（一次性）| — |
| Stage 1 采集 | 10 min | 5–10 min |
| Stage 2 WM 刷新 | 20 min | 3–5 min（增量）|
| Stage 3 critic | 5 min（并行）| 5 min |
| Stage 4 联合微调 | 10 min | 10 min |
| Stage 5 闭环评测 | 5 min | 5 min |
| **合计** | **~50 min**（预检不计入则 ~50 min/轮）| **~25 min/轮** |

---

## 9. 一句话总结

**链路已经通了（真物理闭环 2080 步不摔、基线 0.048/GT 0.069）**，剩下的不是"能不能跑"，
而是三件数据/口径的活：**用 `q_target_applied` 重建真转移（P0-3）→ 保鲜 WM 并验收 `qd_corr` →
训 critic 并验收 `v_ev`**。这三件做完，联合微调才有意义；做完后唯一判据是 **Stage 5 闭环 r 提升 ≥ 5%**。

---

## 10. 当前**未解决**问题清单（阻塞项与风险，2026-10-28 22:00）

> 本节是本文最重要的部分：**流程能跑 ≠ 问题解决了**。下面每一项都写清
> 现象 / 影响 / 是否阻塞 / 缓解 / 判据。执行时按"是否阻塞"排序处理。

### A. 评测口径不一致（**阻塞 G5 的判据有效性**）
- **现象**：我们的离线奖励（`bundle.reward` + `ha_cmd` 库）用**世界坐标**比较；而 HA 官方
  环境的奖励/终止把参考**重锚到机器人 xy+yaw**（`supermimic.py`: `dR = yaw_mat(R @ rR^T)`）。
  另外 B 线（http）模式机器人起点是 env 的 init pose，与库 clip 起点不同 ⇒ 绝对根误差有常数偏移
  （实测前 200 步就有 0.37 m）。
- **影响**：我们的 `r / e_root` 不能与 env 的成败直接对应；跨臂比较只有在"同任务同起点同 seed"下才有效。
- **阻塞？** 是（G5 用我们的数字判提升，必须先让数字可比）。
- **缓解**：
  1. 采集与评测**统一用 `SM_INIT_Q_NPY` 把机器人 reset 到参考帧的 q/qd**（provider 已支持），
     这样世界坐标比较才有意义；
  2. G5 一律 **≥3 seeds 配对**，阈值改为"超过 seed 间波动"而不是拍 5%；
  3. 同时报 env 自己的 `success`（虽然目前恒 False，但它是唯一任务级信号）。
- **判据**：同一配置 3 seeds 的 `r` 波动 < 20% 时，才算"数字稳定可比"。

### B. A1 未验证 + 图像通道不可预测（**决定方法上限**）
- **现象**：VLA 的输入是录制帧缓存（图像不更新）；A1 只能把**本体感觉**写回。
- **影响**：若 A1 无效，想象里学到的只是"稳态 token"，学不到"偏了怎么纠"；联合微调的收益上限很低。
- **阻塞？** 不阻塞启动，但**决定是否值得迭代**。
- **缓解**：G0-b 先验证 A1 有无影响；若无效 → 方法降级，改走 on-policy DAgger（需要"漂移状态下的正确 token"标签，目前无解）或任务成功奖励。
- **判据**：`--vla-state-from-burnin` 开/关两个 30 步 run 的 `token_shift` / `imag_return_norm` 有可测差异。

### C. 手部无感知 + 无安全保护（**阻塞 Stage 1 的稳定性**）
- **现象**：`WBC_A24000` 的 obs93 **不含手指关节**；provider 只把**身体 29 关节**clip 到限位，
  **手部 14 维既不 clip 也不限速**。
- **影响**：手部命令在重规划边界跳变 → 戳到桌面 → 掀翻。实测：`EXEC_STEPS=20 + 有手` 在
  **未验证 server** 下 166 步摔，**无手**则不摔；换成已验证 server 后同配置不摔。
- **阻塞？** 是（采集里若出现摔，会污染 WM/critic 的分布）。
- **缓解**：provider 给手部加 **clip 到手指关节限位 + 限速**（约 10 行），Stage 1 之前做。
- **判据**：手部目标每步变化 ≤ 0.2 rad，且 5 条采集里 ≥4 条不摔。

### D. 两个 server 的差异**尚未定位**（信息缺失，可能藏着别的坑）
- **现象**：已验证的 `ha_vla_latent64_server.py`（state64→state46，hands=feedback/zeros）不摔；
  我新写的 `vla_token_server.py`（喂机器人**真实手指**状态 + `sample_once_with_grad`）摔。
- **影响**：不确定"喂真实手指"是不是有害；也不知道还有没有别的行为差异（FM 采样入口、noise、state 组装）。
- **阻塞？** 否（已改用验证过的 server），但要记录：**以后任何 server 改动都必须先跑 300 步闭环 A/B**。
- **判据**：新 server 与已验证 server 在同一配置下，300 步内 `r` 差 < 5% 且都不摔，方可替换。

### E. critic 可能根本修不好（**决定第 2 轮配置**）
- **现象**：现有 `critic/v_ev = −0.47 / −1.29 / −1.84`（比预测均值还差）。
- **影响**：自举项 `γ^H·(V(s_H) − V(ref_H))` 注入错误信号。
- **阻塞？** 否（第 1 轮 `--no-critic`），但若想用长时域优化就必须修。
- **缓解**：用"录制 634k 帧 + B 线漂移帧"的真实 return-to-go 重训；**v_ev > 0.3 才启用**，否则永久关闭。
- **判据**：留出窗口 `v_ev ≥ 0.3`。

### F. WM 偏差对梯度的污染（**无法消除，只能量化**）
- **现象**：C6h 的 WM `qd_corr 0.752`、位置误差 ~1.5 mm；想象 H=20 会累积。
- **影响**：梯度方向有偏；"想象涨、真物理不动"的直接来源。
- **阻塞？** 否，但它是**所有想象训练的固有偏差**。
- **缓解**：① WM 保鲜必须验收 `qd_corr` 不掉；② 可做 H=10 vs H=20 的对比，看收益是否依赖长程想象；③ 最终只认闭环。
- **判据**：保鲜后 `qd_corr ≥ 0.70`；H=10 与 H=20 的闭环收益差 < 30%（否则说明严重依赖 WM 长程）。

### G. 任务成功率维度缺失（**战略级**）
- **现象**：6 任务 `success = False ×6`；跟踪奖励提升 ≠ 任务做成。
- **影响**：可能出现"闭环 r 涨、任务仍然 0/6"的结局，那时本文的 G5 判据会误导。
- **阻塞？** 否，但必须**双轨报告**。
- **判据**：Stage 5 同时报 `r` 和 `success`；若 r 涨 ≥5% 而 success 仍 0，则进入"换目标"分支（任务成功/物体状态奖励）。

### H. 采集覆盖窄（**影响 WM/critic 泛化**）
- **现象**：每轮 5 条；VLA 臂起点固定（env init pose），只能覆盖该段运动。
- **影响**：WM/critic 只见过局部状态，换 clip 就退化。
- **阻塞？** 否（第 1 轮够用）。
- **缓解**：第 2 轮起做"随机起始帧"（GT 臂 `SM_TOKEN_START` 已支持；VLA 臂需 10 行改动），
  并轮换 clip。
- **判据**：连续两轮的 clip 不重叠，且 WM 保鲜后 `qd_corr` 不降。

### I. bf16 未对拍（低风险，但会干扰归因）
- **现象**：直接按 GR00T 原生 dtype 选了 bf16，没做 fp32 对拍。
- **缓解**：出现"想象与真实系统性不一致"时，第一时间做同 seed 的 fp32/bf16 64 窗口对拍。
- **判据**：token 差 ≪ 0.004，`imag_return` 差 ≪ 1%。

### J. 并行采集未做（**只影响速度，不影响正确性**）
- 现状单 env ≈ 3.6 步/s；GPU0 空闲 40 GB，可同时跑 3–4 个 Isaac 进程（零代码改动）。
- 等 G5 通过后再做，避免为一个还没验证的方法投入工程。

---

### 10.1 阻塞项优先级（执行顺序）
```
C（手部安全）        → Stage 1 之前必须做，10 行
A（评测口径）        → Stage 1 之前定好（SM_INIT_Q_NPY + ≥3 seeds）
B（A1 验证）         → Stage 0.4，10 分钟，决定方法上限
E（critic v_ev）     → Stage 0.3，不阻塞第 1 轮
F/G（WM 偏差/任务成功）→ 持续监控，Stage 5 双轨报告
D/H/I/J              → 记录在案，按需处理
```
