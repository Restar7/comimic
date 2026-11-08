# 主控容器 A 交接文档：P0 动作合同 + Base 逐 tick 采集 + WM1 候选流水线

> 时间：2026-11-08 16:40 ｜ 作者：主控容器 A 的 DeepSeek Agent ｜ 对象：下一位接手的 Agent/工程师
> 读完本文**不需要翻任何聊天记录**即可继续；所有数字都可复核（路径写在 §12）。
>
> **与旧文档的关系**：`docs/HANDOVER_FULL_SESSION_20261108.md`（360 行）是**另一条线（C 线）**的交接，
> 讲 C 线 LoRA 训练与闭环阻塞；**本文是"主控 A + WM6 动力学采集"这条线**，两者互补，都要读。
> 仓库：主仓 `comimic`（HEAD `1ea002e`）、主控 worktree `comimic-wt03`（HEAD `43b9f6b`）。
> 共享盘：`/inspire/hdd/global_user/liumengfan-253108110079`（10 T，剩 1.5 T）。

---

## 0. 三分钟速读

**链路**：冻结 FT8000 VLA → SONIC release decoder → （A/B 线）1.744 M Δ 残差网 /（C 线）LoRA16 → 用 8.82 M 世界模型 WM 当可微裁判做 H20 BPTT 局部 tracking。

**本会话主线**：怀疑 WM（裁判）不准 → 审计发现**动作/状态关节序合同错了** → 修正合同 → 重采六任务 base 逐 tick 数据 → 训候选 WM1 → 物理门禁 → 过门禁才开 C 线。

**最重要的单条结论**（可复核）：
> A′ 及一切合法数据的合同是 **`state71` 的 q/qd 用 STANDARD 序 × `action` 用 HA(=SONIC) 序归一化**；
> 旧的 v7 on-policy 采集器**两处都错**（q/qd 存成 HA 却当 STANDARD；归一化常量用 STANDARD 去减 HA 的 q_target）。
> 后果量化：换序让冻结 WM0 的 20 步 root 误差 **5.63 cm → 12.75~18.79 cm**；
> 旧 on-policy 数据实测 e20 **0.2573 m**，按正确合同重算是 **0.0909 m**（差 2.8×）。
> ⇒ 之前「WM 在新分布 31.8 cm」**主要是合同错配**，不是分布偏移；因此 `wm_c1_v001_published` 判 **INVALID**，`active.json` 已回滚到 `wm_v000`。

**此刻正在跑**

| 进程 | 作用 | 进度（16:36） |
|---|---|---|
| `collect_base_wm_v2.py`（w0, cuda:0） | open_door + sit_sofa × 6 seed | open_door s0(1456 tick)✓ s1(2992 tick)✓ |
| `collect_base_wm_v2.py`（w1, cuda:1） | vision_navi × 6 seed | vision_navi s0(1632 tick)✓ |
| VLA server（:18220） | FT8000，**无 Δ** | 服务中 |
| `supervise_wm_base6.sh`（pid 4037545） | 采集结束自动接力：审计→合并→训练→门禁 | 等待中 |

**下一步**：采集收尾（ETA 17:40–17:50，共 18 集）→ 监督脚本自动跑 → 人工确认门禁 → 过则开 C 线。

---

## 1. 系统全链路与四个角色

```
图像+语言+state46 → [VLA: Gr00tN1d7 @ checkpoint-8000] → 64-d token chunk (T=40)
                  → [SONIC release decoder] → q_target(29, rad, HA 序) @50 Hz，每 K=20 tick 重规划
                  → 叠加 Δ(1.744 M, cap=0.08 rad) 或 LoRA16
                  → [WM: DynamicsMLP 8.82 M] 当可微裁判，H20 BPTT
```

| 角色 | 实体 | 参数 | 状态 |
|---|---|---|---|
| VLA | `comimic/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（23 文件/35.7 GB，`config.json` sha `7d06d0da…`） | 1.63 G | 冻结 |
| SONIC decoder | `../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx`（40,900,688 B，sha **`c7241a12…`**） | — | 冻结 |
| Δ 残差网 | token_proj(64→128)+Transformer(1L,d128,h4)+hist_gru(58→128)+MLP(400→1024→1024→64) | 1.744 M | A/B 线 |
| WM | `133→2048→2048→2048→70` + LayerNorm；termination head `4→64→64→1`；analytic heads 无参 | 8.82 M | 本轮重训 |

⚠️ **禁用**：`policy/low_latency/model_decoder.onnx`（149,763,762 B，sha `c4ac2e74…`）——与 release 残差差 5°，会毁掉一致性。

---

## 2. P0 动作合同（唯一真相，**不可再猜**）

```
state71 (71) = [ root_pos(3) | root_quat_wxyz(4) | q_STANDARD(29) | v_w(3) | w_w(3) | qd_STANDARD(29) ]
                idx 0:3       3:7              7:36            36:39    39:42      42:71
action29_model (29) = (q_target_HA − dq_HA) / sc_HA
dt = 0.02 s
```

* 置换（互逆）：`PERM_HA_TO_STD = [0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28]`，`PERM_STD_TO_HA = [0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28]`，`dst[i] = src[PERM[i]]`。
* `dq_HA[i] = dq_STD[PERM_STD_TO_HA[i]]`，`sc_HA` 同理；`dq_STD`（HA_ALIGN g1 spec）= `[-0.2,0,0,0.42,-0.23,0, -0.2,0,0,0.42,-0.23,0, 0,0,0, 0.35,0.18,0,0.87,0,0,0, 0.35,-0.18,0,0.87,0,0,0]`，`sc` 均值 **0.3789**。

### 证据 A：张量实测（`scripts/audit_action_contract.py` → `runs/wm_base6_v1/audit/action_contract.json`）

| 项 | 结果 |
|---|---|
| A′ 行布局 | `(step-major, env-minor)`，`num_envs=16`；lag=16 同 clip 最多、`frames` 步长 1 |
| states q/qd 顺序 | **STANDARD**（`corr(mean q, dq_STD)=0.62` vs `dq_HA=-0.06`；膝 0.331/0.411、踝 −0.166/−0.217 落在 STD 槽） |
| actions 顺序 | **HA**（`corr(a_i,q_std[i])=+0.017` vs `corr(a_i,q_std[P_i])=+0.469`） |
| 常量顺序 | **HA**（判别关节 OLS 截距：与 `dq_HA` 误差 **0.159**，与 `dq_STD` 0.386） |
| 还原 q_target 误差 | SONIC 常量 0.334 < 混序 0.454 < 纯 STD 0.506 rad |

### 证据 B：代码审计（带 file:line）

* `runs/a_prime_sonic_exc/all_transitions.pt` ← `scripts/isaac_a_collect_batch.py`：states 来自已 `to_standard` 的 `joint_pos`；**actions = decoder raw = HA 序**（`:1216,1239`）。
* `scripts/v7_tick_record/sitecustomize.py:52-62` 用 `provider._sonic_idx`（HA）读 q/qd 却当 STANDARD 存 —— **0 次换序（需要 1 次）**。
* 同文件 `:147` `a_wm=(q_target_HA − _DQ)/_SC`，`_DQ/_SC` 来自 WBC `robot_spec`（STANDARD）—— **一个动作混两种序**。
* `SonicIsaacBundle.step` 对动作做 1 次 STD→HA；`SonicMjlabBundle.step` 不做 —— 以 A′ 为准，`sonic_isaac.py` 正确。

### 换序代价（冻结 WM0，真实 A′ 64 窗口 × 20 步）

| 条件 | root e1 | root e20 |
|---|---|---|
| 正确 | 0.0005 m | **0.0563 m** |
| 状态换序（HA 当 STD） | 0.0054 m (10.8×) | 0.1879 m (3.3×) |
| 状态换序（STD 当 HA） | 0.0050 m (10×) | 0.1539 m (2.7×) |
| 动作换序 | 0.0022 m (4.4×) | 0.1275 m (2.3×) |

### 旧 on-policy（`cycle000_A`）实测（冻结 WM0，96 窗口 × 20 步）

| 组合 | root e20 |
|---|---|
| 现状（双向错） | **0.2573 m** |
| 只修 state | 0.2010 m |
| 只修 action | 0.1082 m |
| **都修（v2 合同）** | **0.0909 m** |

---

## 3. 数据清单

### 3.1 base 闭环成功率（官方 FT8000，无 Δ）

| 任务 | `comimic/runs/ab36/base` | `comimic/runs/closedloop_r3/base6x6` |
|---|---|---|
| PPBox | 2/6（1227、948 步） | 2/6（724、1118 步） |
| DoubleDesk | **0/6** | **0/6** |
| Boxing | 4/6（240、248、277、1821） | 4/6（176、233、276、298） |
| OpenDoor | 6/6（974–1923） | 5/6（785、572、780、2690、2936） |
| SitSofa | 3/6（487、963、2324） | 2/6（1496、1551） |
| VisionNavi | 2/6（1678、701） | 3/6（707、1233、703） |
| 合计 | **17/36** | **16/36** |

* provenance 三条硬证据：VLA=`checkpoint-8000`（逐文件 sha）、decoder=`release`(`c7241a12…`)、`head_ckpt=MISSING` + `server.log` 无 `delta-v5]`。
* **同 seed 跨进程不可复现**（PPBox 本会话成功 s3/s4，ab36 是 s3/s5）⇒ 只按 n/6 统计。
* DoubleDesk 在 base 下 **37 集全 0**（含早期 25 集）——系统性做不到，验证 Δ 价值的最佳靶子。
* 早期 sr3/sr4：SitSofa 4/6、VisionNavi 3/6（无 per-run sha，证据中等，已弃用）。
* ⚠️ `runs/ab36/base` 由**另一个会话**写入；`comimic` 仓还有别的会话在提交 → **绝不 `git add -A`**。

### 3.2 A′（WM0 语料）

`comimic-wt03/runs/a_prime_sonic_exc/`：50 shard、**529,131** 转移、370.6 MB、合并 sha `418c310b…`；`source=a_line_wbc_only`、`action_noise_std=0.15`、`reset_max_drift=0.15`、`n_leash_resets=189`、100 clips × 6 任务 × offsets{0,16,…,112}；50 Hz；字段 `states/actions/next_states/frames/clip_ids/meta`；`a_t` = t+1 命令。

### 3.3 旧 on-policy（**已判不可用**）

`runs/three_track_v7/buffer/cycle000_A/`：A_delta、仅 OpenDoor、seeds 2–11、12 进程、7,552 tick、366 段。合同错（§2.4）⇒ 只能当诊断样本。

### 3.4 本轮新采（进行中）

`runs/wm_base6_v1/collector_A/<task>/seed<k>/`：`ticks/ticks_<pid>_<chunk>.pt`（`.partial`→原子 rename）+ `episode.log` + `episode.json` + `result.json`。
每 tick 15 字段：`states_before(71) states_after(71) actions_model(29) actions_raw(29) q_target(29) q_applied(29) task_id episode_uid seed tick frame_count dt done done_reason meta{}`。
已验证：`state_after[t]==state_before[t+1]`（max diff **0.0**）、`frame_count` 步长 `{1}`、`actions_model==(q_target−dq_ha)/sc_ha`（err 1.2e-5）、state 序 STANDARD（corr 0.735 vs 0.038）。

---

## 4. WM 现状

### 4.1 架构与口径

* 输入 `133 = canonical(71) + action(29) + xi(33)`；输出 70 维增量；`StateSpec.integrate` 显式积分 dt=0.02。
* `StateSpec.split`：`p 0:3, quat 3:7, q 7:36, v 36:39, w 39:42, qd 42:71`；`canonical()` = heading-invariant 特征（root_z + rot6d + v_h + w + q + qd）再除 `in_scale`。
* **`xi` 恒 0**：`default_xi=[0.75,0…]` → `normalize_xi` 后全 0（`xi_scale=(0.45,(0.01)*29,(0.05)*3)`）。训练/推理一致，但 WM 看不到参考命令 —— 评估时必须记住。
* 损失：`ssm`（WM 自己的 `out_norm.std()` + `ERR_FLOOR` 逐维归一化平方误差）+ 单步 + k=20 自回归；ckpt meta `stage1=800 stage2=800 gain_weight=2.0 gain_pairs=runs/paired_probe2/gain_pairs.pt`。

### 4.2 版本

| 版本 | sha 前缀 | 状态 |
|---|---|---|
| `wm_v000` | **`fc5aeb30`** | **active**（A′ 训练，冻结参考） |
| `wm_c1_v001_published` | `81d24caf` | **INVALID**（`INVALID.json`，训练数据合同错） |
| `wm_probe_candidate/gate` | — | 历史探针 |
| `active.json.bak_1625` | — | 回滚前备份 |

私有候选商店 `runs/wm_base6_v1/wm_candidates/`：`active.json` → `init_from_wm0`（= `wm_v000` 拷贝，sha `fc5aeb30`）。

### 4.3 数字

| 指标 | WM0 | v001 | 来源 |
|---|---|---|---|
| 旧 A′ val h=1 root | 0.0280 m | — | `diagnostics/wm0_oldonly/REPORT.md` |
| 旧 A′ val h=20 root | **0.0459 m** | 0.0448 m | 同上 + `gate.json` |
| 旧 A′ val h=20 p95 | 0.1854 m | — | 同上 |
| 旧 A′ val h=20 ∑ssm | 89.13 | — | 同上 |
| 新分布（旧错数据） | 0.3180 m | 0.1097 m | 外部门禁报告 |
| 新分布（正确合同） | — | — | **0.0909 m**（`tick_contract_diag.json`） |

v001 gate：`new_e20 +65.5%`、`old_e20 −0.45%`，但 **`dir_agree` FAIL(OVERRIDE)**（梯度探针从未跑）⇒ PASS 不成立。

---

## 5. 本会话新增/修改的代码

| 文件 | 作用 |
|---|---|
| `scripts/collect_base_wm_v2.py` | 唯一 base 逐 tick 采集入口（固定 CLI + fail-closed + server 托管 + 事后补 `done` + manifest） |
| `scripts/wm_base6_record/sitecustomize.py` | v2 记录器（PYTHONPATH 注入；15 字段；原子写；episode 边界 flush；SIGTERM/atexit 保底） |
| `scripts/audit_action_contract.py` | P0 合同审计（序/常量/换序代价），不成立返回 2 |
| `scripts/diag_tick_contract_v2.py` | 4 种序组合 e20 对照（找真实合同） |
| `scripts/audit_base_wm_v2.py` | P3 数据审计（schema/finite/四元数/连续性/dt 自检/动作域/裁剪/静止尾段/坏文件） |
| `scripts/merge_base_wm_v2.py` | P3 合并（episode 级 4:1:1、窗口不跨界、输出兼容 `train_wm_refresh_v7.py`） |
| `scripts/run_wm_base6_pipeline.sh` | P3–P5 流水线（audit/merge/train/gate） |
| `scripts/supervise_wm_base6.sh` | 后台监督 + 安全阀（4 h、val 空或 train<500 段拒训、失败即停） |
| `scripts/summarize_closedloop.py` | 闭环结果汇总（三种目录布局；复现 SitSofa 4/6、VisionNavi 3/6） |
| `scripts/collect_base_6x6.sh` | 旧的成功率采集（36/36 已完，备查） |
| `docs/BASE_WM_STATUS_FOR_GPT_20261108.md` | 给外部 GPT 的现状清单（含 tick 字段覆盖度表 + 11 个待答问题） |
| `docs/P0_CONTRACT_AND_COLLECTOR_DELIVERY_20261108.md` | P0 审计 + 采集器交付报告 |
| **本文** | 全会话交接 |

会话早段的物理损失修复（已 commit，后续训练要用）：`train_delta_physical_bptt_v6.py` 增 `--trust-domain/--peak-budget-mode/--peak-agg/--lambda-rate-soft/--recovery-agg/--wm-version/--strict-wm-hash`，`default_joint_pos` 改用 `env.bundle.default_joint_pos`，`_phys_log` 并入 `train.jsonl`；`tracking_recovery_tail.py` 加 `top_mean()` 与 `worst_window`；`physical_bptt_losses.py` 加 `l_peak_topk`、`l_rate_soft`。物理量：`e_phys = G1_ACTION_SCALE ⊙ (a_Δ − a_base)`（真 rad）；`L_lim = mean[0.5·max_{t,j}v² + 0.5·mean_t max_j v²]`；严格违反分解满足 `mod = base − fixed + new`（残差 ~1e-9）。

---

## 6. 采集器 CLI 与交接协议

```bash
python3 scripts/collect_base_wm_v2.py \
  --tasks <pp_box|double_desk|boxing|open_door|sit_sofa|vision_navi> ... \
  --seeds 0,1,2,3,4,5 --max-steps 3000 --sim-gpu 0 --vla-gpu 1 --video off \
  --port 18220 --out-root runs/wm_base6_v1/collector_A
# 可选 --smoke-steps N  --dry-run  --reuse-server  --server-url
```
任务名→HA `TASK` 是显式映射（`pp_box→PPBox … vision_navi→VisionNavi`），`task_id` 0–5 写进每个 tick。

**容器 A（本机，正在跑）**
```bash
# w0（cuda:0）
python3 scripts/collect_base_wm_v2.py --tasks open_door sit_sofa --seeds 0,1,2,3,4,5 \
  --max-steps 3000 --sim-gpu 0 --vla-gpu 1 --video off --port 18220 \
  --out-root runs/wm_base6_v1/collector_A
# w1（cuda:1，复用同一 server）
python3 scripts/collect_base_wm_v2.py --tasks vision_navi --seeds 0,1,2,3,4,5 \
  --max-steps 3000 --sim-gpu 1 --vla-gpu 1 --video off --reuse-server --port 18220 \
  --out-root runs/wm_base6_v1/collector_A
```
**容器 B（只写 collector_B，端口 18222）**
```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
source runs/wm_base6_v1/control/collector_B.env
python3 scripts/collect_base_wm_v2.py --tasks pp_box double_desk boxing \
  --seeds 0,1,2,3,4,5 --max-steps 3000 --sim-gpu 0 --vla-gpu 1 --video off \
  --port 18222 --out-root runs/wm_base6_v1/collector_B
```
B 的边界：只读 READY/脚本；只写 `collector_B/`；不改代码/Git/模型/A 目录；不训练。

**READY**：`runs/wm_base6_v1/control/READY.json`（git commit、collector/recorder sha256、VLA+decoder sha、**合同版本+两个置换表**、换序代价、六任务映射、两容器 env 与命令、schema 版本、run id `wm_base6_v1_20261108`）。

---

## 7. 流水线与监督

```bash
bash scripts/run_wm_base6_pipeline.sh audit|merge|train|gate|all
```
* `train`：`--version-store runs/wm_base6_v1/wm_candidates --wm-old init_from_wm0`，A′:`base` = 80:20，`mode=both, k=20, lr=1e-5, updates=150, batch=256, window_batch=8`，`--eval-every 25`，`device cuda:1`。
* `gate`：`eval_wm_refresh_v7.py --wm-old init_from_wm0 --wm-candidate <ver> --old-replay A′ --onpolicy-data merged/val --skip-gradient`（只出诊断）。
* **发布规则**：必须 `probe_wm_gradient_truth.py` 产出 `dir_agree` 后用 `--gradient-probe-result` 重跑，**禁止 `--allow-prediction-only`**。

`supervise_wm_base6.sh`（pid 4037545，日志 `runs/wm_base6_v1/supervise.log`）：
等采集退出（≤4 h）→ audit → merge → 若 `val` 空或 `train<500` 段则**只交付**、不训练 → train → gate。

---

## 8. 环境与坑（踩过的全在这）

| 坑 | 现象 | 正确做法 |
|---|---|---|
| server/client 环境隔离 | client 若 source `harness03_env.sh` → `dep_onnxruntime=MISSING`，全 0 步；server 不 source → `No module named tyro` | server 在**子 shell** source 后 exec；client `PYTHONPATH` 必须干净 |
| PYTHONPATH 顺序 | `import diff_vla_wbc` 命中主仓旧包（缺 `wm_replay_buffer`） | wt03 排第一：`$WT:$C/trackerlab-supermimic/source/trackerTask:$C/trackerlab-supermimic/source/trackerLab:$C` |
| package 路径 | `ModuleNotFoundError: trackerLab.kinematics` | 真实包在 `source/trackerLab/trackerLab/`，要加 `source/trackerLab` |
| Isaac 强制退出跳过 atexit | 尾块丢、`done` 全 0 | 记录器 episode 边界 flush + SIGTERM；**采集器在集结束后由外部进程补 `done`**（已实现） |
| Isaac python | `ha_isaac_closedloop.sh` 是 `exec python` | `third_party/isaacsim_env/bin` 放 `PATH` 首 |
| RTF≈0.05–0.1 | 墙钟 0.43 s/tick 但 dt 仍是 0.02 s | 用速度一致性验证 dt，别用墙钟推 |
| 视频 | 开录像 ~14 min/集 | 正式 `--video off`（4–10 min/集） |
| 端口 | 多 server 抢端口 | A=18220、B=18222；旧 18210 已停 |
| 磁盘 | 曾 100% 满写坏 ckpt | 采集前 preflight 查剩余（现 1.5 T） |
| git 并发 | 另一会话在同 worktree 提交 | 只 `git add <具体路径>`；`runs/` 被 gitignore（需要时 `git add -f`） |
| `runs/` 被 gitignore | READY.json 不在 git | 交接以盘上文件为准 |

---

## 9. 诚实清单：失效 / 待复核

| 结论 | 状态 |
|---|---|
| `wm_c1_v001_published` gate PASS（新分布 −65.5%） | **失效**（数据合同错）→ INVALID + 回滚 |
| 「WM 新分布 31.8 cm = 分布偏移」 | **改写**：约 2.8× 是合同错配；修正后 9.1 cm |
| `probe_wm_gradient_truth.py` 69% / `probe_wm_action_jacobian.py` 98.4% | **待复核**：两者都有 `actions`(归一化) 与 `q_target_applied`(rad) 混用 fallback |
| 旧 `cycle000_A` | **不可用**（合同错） |
| 「base PPBox ≈ 0」 | **错**（只看前 2 seed），实际 2/6 |
| SitSofa base 稳定基线 | **不稳**：4/6（sr4）→3/6（ab36）→2/6（base6x6），yaml 未变 |
| 同 seed 可复现 | **不可**（跨进程） |

---

## 10. 未决问题 / 风险

1. `q_applied` 只是近似（`robot.data.joint_pos_target`，provider 侧裁剪后；无 PhysX 回读）。
2. `xi` 恒 0：WM 看不到参考命令，"局部 tracking 裁判"是否够用存疑；要改需重训并证明旧数据不退化。
3. 静止尾段：DoubleDesk 长期 timeout 的后半段几乎静止；`merge` 已记录 `still_frac_qd_lt_0.1`，**降权策略未定**。
4. 门禁三（`dir_agree`）**无实现**：先修 `probe_wm_gradient_truth.py` 的动作口径，再用 Isaac 有限差分产出 `{"dir_agree_old","dir_agree_new"}`。
5. 门禁五（真闭环 H20）需完整物理状态；目前只能恢复 71 维机器人状态 ⇒ 必须显式标注限制。
6. C 线训练前要修 `strict-resume` 采样指纹 FAIL（采样器进度、Batch 顺序、RNG、Flow 噪声、优化器/调度器状态）。
7. 容器 B 是否就绪未知（READY 已发布，未回报）。

---

## 11. 下一步与放行条件

| 阶段 | 内容 | 放行条件 |
|---|---|---|
| P3（进行中） | 采集完成 → audit → merge | 6 任务 × ≥6 集；问题 episode=0；val/test 非空 |
| P4 | 候选 `wm_base6_001`：A′:base=80:20，init=`fc5aeb30`，k=20 both，lr 1e-5，100–150 updates，val 早停 | loss 有限、val e20 改善、可复现 |
| P5-1 | 旧 A′ val H1/H20 root/heading/q/qd 退化 ≤5%（逐项） | 通过 |
| P5-2 | 六任务独立 val H1/H20（分任务 + 分位数） | 不显著变差 |
| P5-3 | **动作增益/收益排序**：Isaac 有限差分 vs WM 梯度，`dir_agree` 真跑 | 达阈值，且能判断近邻动作哪个真实更好 |
| P5-4 | C-V8 24 窗口困难案例（WM0 `+0.00950` vs Isaac `−0.00733`） | 不再把负收益判成正 |
| P5-5 | 真闭环 H20（可恢复范围内），WM 与 Isaac 同动作/参考/Reward（修 `bptt_args` 的 `term_off`/`d_max`） | 通过 + 标注限制 |
| P6 | 门禁过后才开 C 线：FT8000 + 零增量 LoRA16 + 冻结 WM1/SONIC + H20 J20，与 WM0 版同起点/同 batch/同种子对照 | 先修 `strict-resume`、先对拍 reward 与部署链路 |

**不通过就不发布**：保留 WM0，`active.json` 不动，候选留 `wm_candidates/` 并写清失败原因。

---

## 12. 附录

### 12.1 关键路径

```
VLA            comimic/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000
decoder        ../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx
WBC 执行器      comimic/runs/S1_handover/WBC_A24000_wmvel_cmd.pt
packed motion  comimic/runs/isaac_motion/ha_cmd/packed.pt  (sha 3f444d50…)
A′ 语料         comimic-wt03/runs/a_prime_sonic_exc/
WM0            comimic-wt03/runs/three_track_v7/wm_versions/wm_v000/wm.pt
任务 yaml       comimic/third_party/HumanoidArena/isaaclab_twist2_g1/tasks/common_test_config/base_test/*_sonic_test.yaml
闭环入口        comimic/scripts/ha_isaac_closedloop.sh
新采集/流水线    comimic-wt03/runs/wm_base6_v1/
```

### 12.2 sha256 前缀

| 对象 | 前缀 |
|---|---|
| VLA `config.json` | `7d06d0da` |
| SONIC release decoder | `c7241a12` |
| SONIC low_latency（禁用） | `c4ac2e74` |
| packed motion | `3f444d50` |
| A′ `all_transitions.pt` | `418c310b` |
| WM0 | `fc5aeb30` |
| v001（INVALID） | `81d24caf` |

### 12.3 本会话 git 提交（`comimic-wt03`）

```
43b9f6b wm_base6_v1 P3/P4/P5: 审计+合并+候选训练+门禁流水线
38a2bf0 wm_base6_v1: P0 动作合同审计 + v2 逐tick采集器 + READY 交接
537dd27 docs: 补 tick 字段覆盖度表 + 修正 A' 动作域为归一化
afc455f docs: base 采集/WM 训练/A' 语料现状清单（给 GPT）
0a5bc26 scripts: base 6x6 采集 + 闭环结果汇总 + 任务分解诊断
（更早）941f7a9 物理损失修复；d76e029 C 线全会话交接
```

### 12.4 时间线

| 时间 | 事件 |
|---|---|
| 12:57 | 启动 base 6×6 成功率采集（36/36 完成） |
| 14:30 | 发现 `runs/ab36/base` 已是完整 6×6（另一会话 11-08 补 3 任务）；两套互相印证 |
| 14:29 | 另一会话发布 `wm_c1_v001_published`（后判 INVALID） |
| 15:25 | 写 `BASE_WM_STATUS_FOR_GPT`（给外部 GPT 的现状清单） |
| 16:00–16:20 | **P0 合同审计**（张量 + 代码双证据）；换序代价 5.6→19 cm；旧数据 0.2573→0.0909 m |
| 16:20 | v2 记录器 + 采集器交付，冒烟 6 项全绿 |
| 16:22 | 发布 READY + 两容器 env |
| 16:25 | `active.json` 回滚 `wm_v000`；v001 标 INVALID |
| 16:27 | `wm_base6_v1` 正式采集开跑（双卡） |
| 16:30 | P3 审计/合并跑通（0 问题 episode） |
| 16:34 | 后台监督启动（自动接力） |

---

**接手第一件事**：`tail -40 runs/wm_base6_v1/supervise.log`。若监督已死而数据齐全：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
bash scripts/run_wm_base6_pipeline.sh merge
bash scripts/run_wm_base6_pipeline.sh train
```
