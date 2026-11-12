# COMIMIC 唯一交接文档（ALL-IN-ONE）· 2026-11-08/09

> 这份文档**取代**此前所有分线交接（`HANDOVER_20261108_MASTER_A_WM6.md`、`SUPERVISION_WM6_20261108.md`、
> `WM6_FULL_HANDOVER_FOR_GPT_20261108.md`）作为**唯一入口**：读完这一份就能接手。
> 作者：容器 A 接手 Agent（DeepSeek）｜覆盖两条线：**WM6 动力学线** 与 **C 线联合微调线**。
> 主仓：`/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic`（只读权重/资产）
> WM6 工作树：`…/comimic-wt03` ｜ C 线工作树：`…/comimic-c-wm6`（分支 `experiment/c-wm6-h10-critic`）

---

## 0. 30 秒状态

| 事项 | 状态 |
|---|---|
| base6 数据 | **36/36 集**、75,216 tick、逐集复核全过、3739 窗口逐位 0 坏 |
| WM6（`wm_base6_best`, s350） | **门禁 PASS 6/6**：新分布 e20 **+38.59%**、旧 A′ 退化 **+2.72%**、`dir_agree` **0.9792 ≥ 0.9271** |
| 发布 | **未发布**（`active.json` 与公共 WM 指针都没动） |
| C 线（C-WM6 主线） | **C0 PASS 15/15 → C1 DONE（100 updates）→ `C2_READY`** |
| C1 数字 | J10 **0.783→0.821**、critic v_loss **0.062→0.037**、td_err 0.609→0.541、**β=1.0** |
| 部署侧 LoRA16 | **已存在且键空间匹配**（`action_head.*`, 440 张量），无需改代码 |
| 唯一未过判据 | **P5-4（收益排序）**：随机 91 窗符号率 0.44–0.56 ≈ 抛硬币；需要在线共训 critic |
| 下一步 | **C2 真闭环门禁**（Base vs C，同任务同 seed）→ C3（~280 updates）→ 六任务 seals 6–11 闭环 |

**接手先读这三行**：状态机 `comimic-c-wm6/runs/dual_actor_wm6/control/readiness.json`；
步骤清单 `…/c_lora/NEXT_STEPS.md`；本文其余部分按需查。

---

## 1. 资产与角色（完整 SHA）

```
图像+语言+state46 →[FT8000 frozen]→ token[40,64] →[SONIC release]→ q_target(29,HA) @50Hz, 每 K=20 tick 重规划
   → 叠加 Δ(1.744M) 或 LoRA16(16.89M) →[WM6 8.82M 可微裁判, H 步 BPTT] (+ 在线 TokenCritic)
```

| 对象 | 路径 | SHA256（完整） |
|---|---|---|
| **WM6 裁判**（s350） | `comimic-wt03/runs/wm_base6_v1/wm_candidates/wm_base6_best_candidate/wm.pt` | `8edb59cf2adddf0a2f65b6bb573e7c529e9d2a3b9e15ee621f050802aaa4af74` |
| WM0（基线） | `comimic/runs/three_track_v7/wm_versions/wm_v000/wm.pt` | `fc5aeb30e987f016d24a8392af9d0fecb3151e07dc33d49b71f0544f9621ca97` |
| v001（**INVALID**） | `wm_versions/wm_c1_v001_published` | `81d24caf…`（数据合同错，勿用） |
| VLA FT8000 | `comimic/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000` | `config.json` = `7d06d0da7baf0873016f2172fd2df5478f52a92e1c90e94ce2585e10f170e249` |
| SONIC release decoder | `../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx` | `c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed` |
| SONIC low_latency（**禁用**） | `policy/low_latency/model_decoder.onnx` | `c4ac2e74…`（与 release 差 5°，会毁一致性） |
| WBC 执行器 | `comimic/runs/S1_handover/WBC_A24000_wmvel_cmd.pt` | `2390dc295b03f14429a63d6ea1ff78d2997a1419ee45a127977877a9ea2eadc7` |
| packed motion | `comimic/runs/isaac_motion/ha_cmd/packed.pt`（含 `ref` (634016,71)） | `3f444d50b28906f3582e1218049bd2f2d19a6ca3a864d87df502bcb75292c9b1` |
| A′ 语料（50 shard 合并） | `comimic-wt03/runs/a_prime_sonic_exc/all_transitions.pt` | `418c310b…` |
| 离线 critic（**仅 A′ 分布可用**） | `comimic-wt03/runs/wm_real/critic_a_prime.pt` | v_ev=0.3507，γ=0.95，center=linear，hidden 256/depth 2/use_diff=false |

**角色分工**：WM6 与 critic 只参与**训练/诊断**，**绝不进入 Isaac 部署决策**；部署链路只有
`FT8000(+LoRA) → token → SONIC release → Isaac`。

---

## 2. P0 动作合同（唯一真相，不可再猜）

```
state71 = [ root_pos(3) | root_quat_wxyz(4) | q_STANDARD(29) | v_world(3) | w_world(3) | qd_STANDARD(29) ]
action29_model = (q_target_HA − dq_HA) / sc_HA        dt = 0.02 s
PERM_HA_TO_STD = [0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28]
PERM_STD_TO_HA = [0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28]
dq_STD = [-0.2,0,0,0.42,-0.23,0, -0.2,0,0,0.42,-0.23,0, 0,0,0, 0.35,0.18,0,0.87,0,0,0, 0.35,-0.18,0,0.87,0,0,0]
sc 均值 0.3789（逐关节取自 WBC robot_spec，记录器写进 tick meta 的 dq_ha/sc_ha）
xi33 = 全 0（normalize_xi 后仍是 0 ⇒ WM 看不到参考命令）
```

### ★ 两个必须区分的 step 调用（本轮实测，代价 5.5×）

| 场景 | 正确调用 | 证据（64 窗 H=10 on base6 val） |
|---|---|---|
| **记录数据**（base6 tick：state STD + action HA 归一化） | `getattr(bundle,'base',bundle).step`（`IsaacJointBundle`） | e10 = **1.75 cm** |
| C 线内部（token → `bundle.policy` 解码 → 动作） | `bundle.step`（`SonicIsaacBundle`，内部再换序一次） | 该路径自洽（`imagine_with_tokens` 用它） |

若把记录数据喂 `bundle.step` ⇒ e10 = **9.56 cm**（5.5×）；反之把 C 线标准序动作喂 `base.step` 同理错。
**结论已写进** `control/wm_lock.json.wm_step_call` 与 `control/shared_eval_contract.json.wm_io`。

### 独立复核（A′ 64 窗 × H=20，冻结 WM0）

| 组合 | root e20 |
|---|---|
| **正确（state 原样 + action 原样）** | **3.79 cm** |
| 状态换序 HA→STD / STD→HA | 10.95 / 12.10 cm |
| 动作换序 | 18.54 cm |

---

## 3. 数据

### 3.1 base6（本轮新采，36 集）

| 任务 | 采集方 | 成功/超时 | 备注 |
|---|---|---|---|
| open_door | A | 4 / 2 | |
| sit_sofa | A | 4 / 2 | s5 为 592 tick 短集 |
| vision_navi | A | 4 / 2 | 历史最好 |
| pp_box | B | 2 / 4 | 与基线一致 |
| double_desk | B | 0 / 6 | base 做不到（Δ 的靶子） |
| **boxing** | **A 接手**（B 未采，避免重复） | 0 / 6 | 6 集全 3000 步超时 |

- 每集 15 字段 tick：`states_before(71) states_after(71) actions_model(29) actions_raw(29) q_target(29) q_applied(29) task_id episode_uid seed tick frame_count dt done done_reason meta{}`
- **`tick` 是分块局部序号（0..15）**，全局时序只能用 `frame_count`
- **`result.json` 在嵌套目录** ⇒ 采集器的 `finalize_ticks()` 拿不到 ⇒ `done_reason` 全 `unknown`；
  已用 `scripts/annotate_episode_outcome.py` 事后回填（只写 `episode.json`，不改 Raw）
- 单集墙钟：快速成功 1.5–4 min；超时 10–12 min；`--video on` 约 +3–4×

### 3.2 其它语料

| 语料 | 用途 | 关键字段 |
|---|---|---|
| A′ `a_prime_sonic_exc/` | WM0 训练 + 旧分布门禁 + dir_agree | `states/actions/next_states/frames(全局库行号)/clip_ids`；**有 Reference** |
| base6 tick | WM1 refresh + 新分布门禁 | **无 Reference** ⇒ 不能算 tracking 判据 |
| `runs/temporal_ac_bptt_v5/eval_set` | C 线训练/评测窗口 | 448 窗，split 0/1/2 = 256/96/96；`refs_f` 是**参考帧号**不是状态 |

### 3.3 合并与复核（结果）

`merge`：**train 2859 段/24 集、val 426/6、test 454/6**（每任务 4/1/1，窗口 stride=H 且不跨集）。
`verify_merged_wm6.py --all`：**3739/3739 段与 Raw 逐位一致，0 坏段**。

---

## 4. 工具链（全部在 `comimic-wt03/scripts/`，C 线副本在 `comimic-c-wm6/scripts/`）

| 脚本 | 作用 |
|---|---|
| `collect_base_wm_v2.py` | 唯一 base 采集入口（fail-closed + server 托管 + 事后补 done + manifest） |
| `annotate_episode_outcome.py` | 从嵌套 `result.json` 回填 `done_reason/success`（只写 episode.json） |
| `audit_base_wm_v2.py` / `merge_base_wm_v2.py` | P3 审计 / 合并（按 seed 升序，最后两个 = test,val） |
| `verify_merged_wm6.py` | merged↔Raw 逐位核对（`--all`） |
| `wm6_coverage.py` | 六任务覆盖度闸门（`--brief` 给 bash 用） |
| `supervise_wm6_snapshot.py` | 采集期监督快照（进度/ETA/独立合同复核） |
| `train_wm_refresh_v7.py`（已改） | WM refresh（`--save-every/--save-dir` 检查点；`WM6_WINDOW_CACHE=1` 采样池化 20×） |
| `select_wm6_checkpoint_v2.py` | **双口径**选优（官方 holdout 约束 + 稠密口径报告） |
| `register_wm6_candidate.py` | 检查点 → 版本（sha 可追溯） |
| `run_wm6_stage.sh` | 单阶段长训 + 选优 + 登记 |
| `supervise_wm_base6_v3.sh` | 端到端监督（等采集→审计/合并→逐位核对→覆盖闸门→长训→选优→门禁三份） |
| `probe_wm6_dir_agree.py` | dir_agree：WM 梯度 → 精确换算 → job / score |
| `p54_rank_calib.py` | P5-4 复评：符号率+CI、幅度校准、置信门限精确率 |
| `p54_composite_judge.py` | 复合裁判 `J10 + γ^H·(V(s_H)−V(ref_H))` |
| `eval_wm6_per_task.py` | 六任务逐任务 val（P5-2） |
| `train_sonic_lora16_ft8000_bptt.py`（已改） | **C 线训练器**：FT8000+LoRA16 + WM BPTT + **在线 TokenCritic** |
| `train_temporal_ac_bptt_v5.py`（已改） | Temporal-v5 训练器：`--critic-kind token` + `--dry-run` |
| `c_wm6/c0_tests.py` | C0 门禁 15 项（锁/合同/step-call/H10 梯度/critic 梯度规则） |

---

## 5. WM6 结论（哪些成立、哪些被推翻）

### 5.1 门禁（主口径 `--val-fraction 0.2`，带 dir_agree 探针）—— **PASS 6/6**

| 判据 | 值 | 阈值 |
|---|---|---|
| finite / action_contract / no_leakage | ok | True |
| new_e20_rel_improve | **+38.59%**（ci_lo **+7.36%**） | ≥5% 且 ci_lo>0 |
| old_e20_rel_degrade | **+2.72%** | ≤5% |
| dir_agree_new ≥ old | **0.9792 ≥ 0.9271** | ≥ |

逐任务 val（e20 8.26→4.27 cm）：vision_navi +60.0%、double_desk +48.9%、sit_sofa +46.6%、
boxing +45.6%、pp_box +39.8%、open_door +38.2%。

### 5.2 逐 H 精度（base val 426 窗，WM1 vs WM0）

| H | 1 | 2 | 3 | 5 | 8 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|
| WM1 (cm) | 0.17 | 0.39 | 0.63 | 1.10 | 1.70 | **2.10** | 3.10 | 4.27 |
| WM0 (cm) | 0.32 | 0.73 | 1.21 | 2.21 | 3.57 | 4.37 | 6.18 | 8.26 |

⇒ **WM1 在每个 H 上 +45–52%**（这是最硬、n 最大的结论）。

### 5.3 dir_agree（A′ holdout 96 窗，δ=0.03 rad RMS）—— 梯度方向

| H | 2 | 5 | **10** | 20（C-V8 口径） |
|---|---|---|---|---|
| WM0 | 0.927 | 0.990 | **0.979** | 0.458 |
| WM1 | **0.979** | 0.990 | 0.969 | 0.500 |

⇒ **H≤10 时 WM 梯度可信（0.97–0.99）；H=20 崩到 ~0.5。**

### 5.4 P5-4（"反号/抛硬币"）—— **未通过，且已定性**

- 原始 24 窗是**按 cos 取 4 低 + 4 高**的极端子集（`probe_c_vs_base_isaac_plan.py:19,111`），
  WM0：`J_WM=+0.00949` vs 真 `−0.00733`，符号率 **0.458**；WM1：**0.500**。
- 随机扩样（test 45 + val 46 窗，H=10，`--select random`）：
  | 集 | 真值 | WM0 | WM1 |
  |---|---|---|---|
  | test 45 | ΔJ −0.00232，变好 33% | 0.556 CI[0.400,0.711] | 0.444 CI[0.311,0.600] |
  | val 46 | ΔJ −0.00255，变好 30% | 0.500 CI[0.348,0.652] | 0.457 CI[0.326,0.609] |
- 配对检验：仅 WM0 对 7、仅 WM1 对 2 ⇒ **McNemar p=0.180（无差异）**；两 WM 的 ΔJ 相关 **+0.705**；
  去任务偏置后 0.622 / 0.556；每任务仿射校准（乐观上限）0.733 / 0.644。
- 幅度校准：c = 0.267(WM0) / 0.568(WM1)；MAE 7.6→6.8 / 7.1→6.9 mm ⇒ **校准不改善符号**。
- **置信门限（实用口径）**：只在 `|c·ΔJ_wm| > τ` 判断时
  | τ | WM0 覆盖率/精确率 | WM1 |
  |---|---|---|
  | 0.25×中位|ΔJ_true| | 36% / **0.688** | 49% / 0.545 |
  | 0.50× | 11% / **0.800** | 20% / **0.667** |
- **机制（24 窗分解）**：分支差只有绝对 J 的 0.4–2%，而 WM 的绝对 J 误差是 0.35–0.88（差 20–100 倍）；
  WM1 在分布外更"乐观"（预测 J 0.549 vs 真 0.166）⇒ 公共模式抵消更差 ⇒ 相关性 0.574→0.456。
- ⇒ **结论：WM 不能当收益比较器；只能当"带弃权的粗筛"，收益侧必须交给 critic。**

---

## 6. Critic（离线已证不可用；在线已接线）

### 6.1 离线 critic（A′ 分布）

管线：`compute_collect_reward.py`（A′ 逐行奖励）→ `build_critic_dataset.py`（MC return-to-go + ref_rows，
自动推断 stride=16；**502,636 行 / 1369 段**；reward mean +0.0101；ret mean +0.193）
→ `fit_critic_real.py`（`--center linear --epochs 12`）⇒ **v_ev = +0.3507**（留出 96,357，门限 >0.30）。

**负结果（重要）**：把它当裁判做复合 `J10 + γ¹⁰(V(s₁₀)−V(ref₁₀))`：

| WM | J10 单独 | J10 + 离线 critic | J20 |
|---|---|---|---|
| WM0 | **0.583**（corr +0.489） | 0.542（+0.271） | 0.458（+0.257） |
| WM1 | **0.583**（+0.285） | 0.458（+0.096） | 0.500（+0.132） |

⇒ **离线 critic 跨分布失效**：V 是策略相关的，必须在联合微调内共训。

### 6.2 在线 critic（C 线，已接线并跑通）

- 结构：`diff_vla_wbc/token_critic.py::TokenCritic V(state71, ref71, xi33)`（111,105 参数）；
  `diff_vla_wbc/imagined_value_v5.py` 提供 `td_lambda_targets / ema_update / bootstrap_identifiable`。
- 目标：`J_AC = Σ_{t<H} γ^t r_t + β·γ^H·survival_H·(V(s_H) − V(ref_H))`；
  `H=10, K=20(=exec_k), γ=0.99, λ=0.95, EMA=0.995, β 0→1 在 100 update 内`。
- 梯度纪律（C0 已测）：actor 更新时 critic 参数冻结但 **∂V/∂s10 保留**；critic 目标 detach；优化器隔离。
- **结构门槛**：`bootstrap_identifiable(K,H) = K > H`；`K==H` 时 β 被强制 0（这是 v5 之前一直关着 critic 的原因）。

---

## 7. C-WM6 主线（工作树与执行记录）

| 项 | 值 |
|---|---|
| 工作树 | `comimic-c-wm6`（`git worktree`）｜分支 `experiment/c-wm6-h10-critic` |
| commit | `d5f267a`（基线）→ `2280109`（C0 15/15）→ `362b906`/`c76ed5c`/`9356070`（在线 critic 接线跑通） |
| 共享控制 | `runs/dual_actor_wm6/control/{wm_lock,shared_eval_contract,clip_split,resource_map,readiness}.json` |
| 输出目录 | `runs/dual_actor_wm6/c_lora/`（只写这里；主仓与 wt03 只读） |
| GPU | 两张 4090 UUID：`GPU-272c06bc-4338-1cec-7fa0-298692b90d75` / `GPU-12b4dd7c-c3b7-67f6-815c-00e274c95d83`（以 UUID 判隔离） |

### 7.1 C0（门禁 15/15 PASS）

`runs/dual_actor_wm6/c_lora/c0/c0_report.json`：WM6 sha 锁(s350)、FT8000 config sha、SONIC sha、
置换互逆、**step_call 合同**、h10 前向（e1 0.25 / e10 2.04 / p90 3.36 cm）、H10 反传到动作、
critic ∂V/∂s10 非零、ctx 内参数冻结、优化器隔离、`bootstrap_identifiable(20,10)`、γ=0.99 一致。

### 7.2 C1（100 updates，14 min）

`runs/dual_actor_wm6/c_lora/h10_critic/seed0/`：

| 指标 | 前 25 | 后 25 |
|---|---|---|
| J10 | 0.7831 | **0.8205** |
| critic v_loss | 0.0624 | **0.0366** |
| critic td_error | 0.6091 | **0.5405** |
| β 末值 / J_ac−J10 | **1.0** / **+0.0805** | |

检查点：`checkpoint_lora_{0025,0050,0075,0100,best,last}.pt` + `trainer_state.pt`（严格 resume）；
日志：`train.jsonl` / `critic.jsonl`(17 字段) / `physical_report.jsonl` / `attribution.jsonl` / `val.jsonl`。
配置：`batch=32, lr=3e-5, warmup=20, critic_lr=3e-4, critic_batch=2048, critic_updates=4`。

### 7.3 部署侧 LoRA16（**无需改代码**）

`scripts/ha_vla_latent64_server.py` 已有 `--lora16-ckpt` → `_load_lora16()` →
`apply_lora(self.model.action_head, ...)` + `load_lora(..., strict=True)`；
C1 检查点的 `lora` 键前缀正是 `action_head.*`（440 张量）⇒ **键空间匹配**。
⚠️ 必须显式 `--lora-rank 16`（服务器默认 32）；`--lora-alpha`（训练用 32）待核实。

---

## 8. 下一步（按顺序，命令可直接跑）

### C2 真闭环门禁（必须先做）

```bash
# 1) 起 LoRA server（GPU0）
cd /inspire/.../comimic-c-wm6
set +u; source scripts/harness03_env.sh
python3 scripts/ha_vla_latent64_server.py \
  --ckpt ../comimic/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 --port 18080 --horizon 40 \
  --lora16-ckpt runs/dual_actor_wm6/c_lora/h10_critic/seed0/checkpoint_lora_best.pt \
  --lora-rank 16 --lora-targets attn1,attn2,ff --lora-also-train proj_out
# 2) Base 对照：同命令**去掉** --lora16-ckpt（server 起在另一端口）
# 3) 每任务同 seed 跑闭环（开发集 seeds 0-5）
TASK=OpenDoor SEED=0 MAX_STEPS=3000 SERVER_URL=http://127.0.0.1:18080 \
  OUT_ROOT=runs/dual_actor_wm6/c_lora/closedloop/dev/C \
  bash scripts/ha_isaac_closedloop.sh
```

必须记录：真实 J10/J20、后 10 tick 变化、任务进度、超时/摔倒、raw/applied joint target、
关节限位事件、root 轨迹、模型/环境/seed/代码 SHA。
**判据**：C 相对 Base 不出现严重负收益；出现则按 §10 的 B/D 分支排查，不得直接进 C3。

### C3 正式重训 + 六任务验收

```bash
# C3：把 C1 的 --updates 换成 280，--save-every 25，续跑用 --strict-resume 1 --resume <out>
# 六任务：seeds 6-11 × 6 任务 × ≥6 episode，Base 与 C 配对
python3 scripts/summarize_closedloop.py <OUT_ROOT>     # 现有汇总脚本
```
最终交付：`runs/dual_actor_wm6/c_lora/final_report/REPORT.md`（含 PASS/FAIL/INCONCLUSIVE）。

### 放行指标（缺一不可）

1. `dir_agree(H=10) ≥ 0.65` 且不退化（现状 0.97）；
2. critic 在其**训练分布**留出集 `v_ev > 0.30`；
3. **≥91 窗独立真物理**上 `J10+critic` 的符号率与相关**都**优于 J10 单独；
4. C2 无严重负收益；最终六任务成功率/进度相对 Base 不劣化（给显著性）。

---

## 9. 结论总表（成立 / 被推翻）

| 结论 | 状态 | 依据 |
|---|---|---|
| WM1 状态预测比 WM0 好 ~50%（每 H 都是） | ✅ | e20 4.27 vs 8.26 cm |
| H≤10 梯度方向可信（0.97–0.99） | ✅ | A′ holdout |
| H=20 梯度方向不可信（~0.46–0.50） | ✅ | A′ 扫描 + C-V8 |
| **"降 H 就能解决收益排序/反号"** | ❌ 被推翻 | 随机 91 窗仍 0.44–0.56 |
| **"旧 A′ 退化 33%"** | ❌ 口径错 | 官方 holdout 只 +2.72%；稠密口径 +26% 是另一回事 |
| 旧 A′ 稠密（快速段）动力学确实退化 | ✅ | +26.4%（lag=16 holdout 口径） |
| 离线 critic（A′）能提升收益排序 | ❌ 反了 | 0.583→0.54/0.46 |
| critic 必须联合微调内共训 + K>H | ✅ | 结构门槛 + 本轮负结果 |
| base6 无 Reference ⇒ 收益判据只能 A′ 上算 | ✅ | tick 无 frames |
| 部署侧已有 LoRA16 支持且键匹配 | ✅ | `--lora16-ckpt` + `action_head.*` |

---

## 10. 坑与事故（全部真踩过）

| # | 现象 | 根因 | 正确做法 |
|---|---|---|---|
| 1 | 监督脚本突然 `syntax error near fi` 退出 | **在 bash 运行时原地改同一脚本** | 写新文件 + 重启 |
| 2 | `pkill -f <pattern>` 把自己杀了 | pattern 出现在自己命令行里 | 用 `pgrep` 取 PID 再 `kill`，或把 pattern 拼开 |
| 3 | `ModuleNotFoundError: diff_vla_wbc.wm_replay_buffer` | `isaac_env.sh/harness03_env.sh` 把主仓旧包顶前 | Isaac 只在**子 shell** source；训练/诊断用干净 PYTHONPATH（wt03 第一） |
| 4 | `No module named 'tyro'`（载 FT8000） | 需要 harness 依赖 | `source scripts/harness03_env.sh` |
| 5 | Isaac rollout 存完 `roll.pt` 后进程**永不退出** | `app.close()` 卡住 | "输出文件出现 + sleep 6 → kill" |
| 6 | `d03.load_env()` 报"找不到 motion npz" | 没传 `--bptt-args` | 一律传 `comimic/runs/bptt1/args.json` |
| 7 | `MixedReplay` 报"动作越出旧 A′ 域" | 类默认 `action_max_outside_frac=0`（严格），训练器 CLI 默认 0.005 | 自写脚本显式传 0.005 |
| 8 | 训练偶发 loss=830/57（gn 878） | 极端姿态批次 | 已由 grad_clip 兜住；建议异常段过滤 |
| 9 | `--skip-gradient` 门禁恒 FAIL | 缺 dir_agree | 只在诊断用 |
| 10 | 门禁新分布 `eps=1`、CI 抖动 | `MixedReplay` 内部再切 20% | 同时给 `--val-fraction 0.9` 版；改进 eval 不二次切分 |
| 11 | `--val-fraction 0.9` 的旧 A′ 数字显示"+32% 改善" | val 与训练 clip 重叠 | 该跑只取新分布数字 |
| 12 | A′ 20 步窗口池只有 567 个 | `contiguous_runs` 按 (clip,frame) 排序，A′ 是 (step-major,env-minor) | 用 lag=16 轴（398,946 窗）；如改需重算旧基线 |
| 13 | `bundle.step` 把记录数据算差 5.5× | SonicIsaacBundle 内部再换序 | 记录数据用 `base.step` |
| 14 | `batch["refs_f"]` 是**帧号**不是 71 维状态 | 采样器语义 | critic 输入用 `packed['ref'][frames]` 查表 |
| 15 | `batch["frames"]` 0 维 / xi 1 维 / `L_rec` 是张量 | 采样器与库的形状约定不统一 | `_frames_seq/_xi_for/_norm_rec` 归一化（已在 C 训练器里） |
| 16 | `rok.states[ti]` 得到 3 维 | 高级索引 | `rok.states[ti, bi]` |

---

## 11. 禁止事项

- ❌ `git add -A`（主仓/别的会话在同一 worktree 提交；`runs/` 被 gitignore）
- ❌ 改 `three_track_v7/wm_versions/active.json` 或任何公共 WM 指针（发布需人工）
- ❌ 在 **H=20** 用 WM 做收益排序/决策
- ❌ 在 base6（无 Reference）上算 tracking 类判据
- ❌ 用旧 C-V8 checkpoint 续训、用 J10-only 冒充 "H10+Critic"、用旧离线 critic 冒充当轮
- ❌ 原地编辑正在运行的 bash/python 脚本
- ❌ 改 B 容器源文件/ckpt；改机械限位来让 loss 好看

---

## 12. 文件与复现索引

```
# WM6 线（comimic-wt03）
runs/wm_base6_v1/{collector_A,collector_B}/                36 集原始 tick
runs/wm_base6_v1/{audit,merged,gates,wm_candidates,train_ckpts,dir_agree,supervision}/
  gates/wm_base6_best_eval_with_probe/gate.md              ★ PASS 6/6
  gates/gradient_probe.json                                {dir_agree_old:0.9271,new:0.9792}
  supervision/{merged_verify,per_task_val,h_sweep,cv8_h_sweep,p54_calib_test45,
               p54_calib_val46,p54_pairdiag,p54_composite_cv8,ckpt_selection_v2_stageA}.json
runs/dual_actor_wm6/control/                               5 份共享合同（C 线工作树内）
runs/dual_actor_wm6/c_lora/{C0_REPORT.md,c0/,NEXT_STEPS.md,h10_critic/seed0/,c1_report.json}
runs/wm_real/{transitions_a_prime.pt,critic_a_prime.pt}    离线 critic 数据/权重
runs/a_prime_sonic_exc/rewards_current/                    A′ 逐行奖励（429 MB）

# 复现关键命令
python3 scripts/verify_merged_wm6.py --all                 # 3739 段逐位核对
WM6_WINDOW_CACHE=1 python3 scripts/train_wm_refresh_v7.py … # 600 步 5.5 min
python3 scripts/select_wm6_checkpoint_v2.py --ckpts …      # 双口径选优
python3 scripts/eval_wm_refresh_v7.py … --gradient-probe-result supervision/gradient_probe.json
python3 scripts/c_wm6/c0_tests.py                          # C0 15 项
# C1 完整命令见 runs/dual_actor_wm6/c_lora/NEXT_STEPS.md
```

---

## 13. 指标定义

- **root e20**：`‖p̂_H − p_H‖` 的均值（H 步自回归）。
- **dir_agree**：沿 WM 梯度走 ±δ，真物理"变好"的比例（符号一致率）。
- **ΔJ**：`Σ_{t<H} bundle.reward(s_t, a_t, a_{t−1}, FrameRef(frame_t))["total"]`（无 γ、无 survival）。
- **J_AC**：`Σ_{t<H} γ^t r_t + β·γ^H·survival_H·(V(s_H) − V(ref_H))`。
- **v_ev**：`1 − MSE(V, ret)/Var(ret)`（留出 episode）。
- **bootstrap_identifiable(K,H)** = `K > H`（否则 `V(s_H)` 不可识别）。
- **旧 A′-官方口径**：`contiguous_runs` 在 clip 哈希 holdout 上切窗（567 窗）；
  **旧 A′-稠密口径**：lag=16 轴 × holdout clip（40 万候选）。
- **P5-4 覆盖率/精确率**：只在 `|c·ΔJ_wm| > τ` 时判定"更好/更差"，`coverage=n_conf/n`，`precision=符号正确率`。

---

## 14. 时间线（2026-11-08）

16:41 接手 → 16:46 换监督 v2 → 17:31 **A 18/18 完成** → 17:37–18:02 **A 接手 boxing** →
18:02 36/36 → 18:06 merge 24/6/6 + 3739 段逐位 0 坏 → 18:11 长训 600 步完成（5.5 min）→
18:12–18:26 门禁 PASS + dir_agree 真跑 → 18:45–19:15 **H 扫描 + P5-4 随机复评** →
19:12–19:18 离线 critic（v_ev 0.3507）但跨分布失效 → 19:22 token critic 接入 v5 训练器（dry-run）→
19:41 `comimic-c-wm6` 工作树 + 5 份合同 → 19:43 **C0 15/15** → 20:0x–20:23 C 训练器接在线 critic（修 8 个 shape 坑）→
20:23–20:37 **C1 100 updates 完成** → 20:5x 确认部署侧 LoRA16 兼容 → **C2_READY**
