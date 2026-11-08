# P0 动作合同审计 + Base 逐 tick 采集器交付（wm_base6_v1）

> 2026-11-08 16:20，主控容器 A。所有数字都是实测（脚本可复跑），不是推测。
> 产物：`runs/wm_base6_v1/{control,audit,collector_A}`、`scripts/{audit_action_contract,diag_tick_contract_v2,collect_base_wm_v2}.py`、`scripts/wm_base6_record/sitecustomize.py`

---

## 1. 资源与路径审计

| 项 | 实测值 |
|---|---|
| 主仓 | `/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic`（HEAD `1ea002e`） |
| 主控 worktree | `.../comimic-wt03`（HEAD `537dd27`，与主仓共享 `shared` 无冲突；只 shadow `diff_vla_wbc/`+`scripts/`） |
| GPU | 2× RTX 4090 49140 MiB；空闲时 483 MiB |
| Isaac venv | `comimic/third_party/isaacsim_env/bin/python`（3.11） |
| 深度学习栈 | torch 2.8.0a0+nv25.06、onnxruntime 1.31.0、transformers 4.57.3 |
| VLA ckpt | `comimic/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（23 文件/35.7 GB；`config.json` sha `7d06d0da7baf0873…`） |
| SONIC decoder | `GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx` sha **`c7241a12…`** ✅ |
| ⚠️ 禁用项 | `policy/low_latency/model_decoder.onnx` sha `c4ac2e74…`（149.8 MB，另一个模型） |
| A′ 语料 | `comimic-wt03/runs/a_prime_sonic_exc/`：50 shard / 529,131 转移 / 370.6 MB，合并 sha `418c310b…` |
| WM0 | `runs/three_track_v7/wm_versions/wm_v000/wm.pt` sha `fc5aeb30…`，8.82 M 参数 |

---

## 2. P0：唯一状态—动作合同（**实测结论**）

### 2.1 A′ 语料的真实口径

| 项 | 实测 | 证据 |
|---|---|---|
| 行布局 | `(step-major, env-minor)`，`num_envs=16` | lag=16 时同 clip 比例最高、`frames` 步长 = 1；lag=1 时 root 位移中位数 0.83 m（跨 env 跳变） |
| `states` q/qd 顺序 | **STANDARD**（mjlab 29 名列表） | `corr(mean q, default_q_STD)=0.62` vs `corr(mean q, default_q_HA)=-0.06`；膝 0.331/0.411、踝 -0.166/-0.217 落在 STANDARD 槽位 |
| `actions` 关节顺序 | **HA / SONIC(decoder) 序** | `corr(a_i, q_std[i]) = +0.017`（29 关节均值）而 `corr(a_i, q_std[P_i]) = +0.469`，P = HA→STD 索引置换 |
| 归一化常量顺序 | **HA 序**（`dq_ha = dq_std[P]`） | 判别关节（膝/踝/肩/肘/腕）逐关节 OLS 截距：与 `dq_ha` 误差 **0.159**，与 `dq_std` 误差 0.386 |
| 动作语义 | `a_t = (q_target(t+1)_HA − dq_ha)/sc_ha` | 还原 `q_target` 误差：SONIC 常量 0.334 rad < 混序 0.454 < 纯 STANDARD 0.506 |
| dt | 0.02 s | `meta.exec_steps=1`、`frames` 步长 1、`|v|` 与相邻位移/0.02 同量级 |

**所以冻结合同 = `states(STANDARD) × actions(HA 序, 归一化)`。**

### 2.2 换序的代价（冻结 WM0，真实 A′ 窗口 64×20 步）

| 条件 | root e1 | root e20 |
|---|---|---|
| 正确合同 | 0.0005 m | **0.0563 m** |
| 状态换序（HA 当 STD） | 0.0054 m（10.8×） | 0.1879 m（3.3×） |
| 状态换序（STD 当 HA） | 0.0050 m（10×） | 0.1539 m（2.7×） |
| 动作换序 | 0.0022 m（4.4×） | 0.1275 m（2.3×） |

⇒ **关节序错一次，20 步误差就从 5.6 cm 变成 13–19 cm。这不是风格问题，是量级问题。**

### 2.3 现有 on-policy 数据（`cycle000_A`）确实是错的（实测量化）

冻结 WM0、96 个真实窗口、20 步自回归：

| 组合 | root e20 |
|---|---|
| 现状（state=HA 当 STD，action 用 STANDARD 常量减 HA 的 q_target） | **0.2573 m** |
| 只修 state（→STANDARD） | 0.2010 m |
| 只修 action（→HA 常量） | 0.1082 m |
| **state + action 都修（本 v2 合同）** | **0.0909 m** |

⇒ 旧 on-policy 数据「31.8 cm」里，**约 2.8× 是合同错配造成的**，修好后是 9.1 cm（仍高于 A′ val 的 4.6 cm，属正常分布外）。

### 2.4 代码级根因（与 tensor 实测互相印证）

1. `scripts/v7_tick_record/sitecustomize.py:52-62` 用 `provider._sonic_idx` 取 q/qd（HA 序）却当成 A′ 的 STANDARD 存 —— **0 次换序，需要 1 次**；
2. 同文件 `:147` `a_wm=(q_target_HA − _DQ)/_SC`，而 `_DQ/_SC` 来自 WBC `robot_spec`（STANDARD 序）—— **一个动作里混了两种序**；
3. `probe_wm_gradient_truth.py:60-63`、`probe_wm_action_jacobian.py:56` 存在 `q_target_applied`（rad）与 `actions`（归一化）**混用 fallback** ⇒ 此前 69% / 98.4% 的梯度方向结论**标记为待复核**，不作为门禁依据；
4. `SonicIsaacBundle.step` 对动作做 1 次 STD→HA，而 `SonicMjlabBundle.step` 不做 —— 两个 wrapper 合同相反，以 A′（HA 动作）为准，`sonic_isaac.py` 正确。

---

## 3. P1：采集器交付（已通过冒烟）

### 3.1 新增/修改的文件（最小集）

| 文件 | 作用 | 状态 |
|---|---|---|
| `scripts/wm_base6_record/sitecustomize.py` | **新**：v2 逐 tick 记录器（注入式，不改 vendored） | 新增 |
| `scripts/collect_base_wm_v2.py` | **新**：唯一 CLI 入口（前置 fail-closed + server 托管 + 逐集采集 + 事后 done 标记 + manifest） | 新增 |
| `scripts/audit_action_contract.py` | **新**：合同审计（A′ 序/常量/换序代价），返回码 2 = 合同不成立 | 新增 |
| `scripts/diag_tick_contract_v2.py` | **新**：新数据 4 种序组合的 e20 对照 | 新增 |
| `runs/wm_base6_v1/control/READY.json` + `collector_{A,B}.env` | 交接协议 | 已发布 |

**没有改动**：vendored Isaac/HA 代码、SONIC decoder、WM0、`active.json`、A′ 原始文件。

### 3.2 固定 CLI（容器 A/B 同一份代码）

```
python3 scripts/collect_base_wm_v2.py \
  --tasks <pp_box|double_desk|boxing|open_door|sit_sofa|vision_navi> \
  --seeds 0,1,2,3,4,5 --max-steps 3000 \
  --sim-gpu 0 --vla-gpu 1 --video off \
  --out-root <container 独占目录>          # 可选 --smoke-steps / --dry-run / --reuse-server
```

任务名 → HumanoidArena `TASK` 显式映射（不靠字符串猜）：
`pp_box→PPBox, double_desk→DoubleDesk, boxing→Boxing, open_door→OpenDoor, sit_sofa→SitSofa, vision_navi→VisionNavi`。

### 3.3 冒烟实测（open_door seed0，100 步）

```
[base6v2] 16:17:14 open_door seed0 rc=0 0.8min ticks=6 文件
[base6v2] 16:17:16 open_door seed0 连续性: frames 1..96 step=[1] done_reason=unknown
```

逐项验证：

| 检查 | 结果 |
|---|---|
| 字段齐全 | `states_before/after(71)`、`actions_model(29)`、`actions_raw(29)`、`q_target(29)`、`q_applied(29)`、`task_id/episode_uid/seed/tick/frame_count/dt/done/done_reason/meta` ✅ |
| `state_after[t] == state_before[t+1]` | 最大差 **0.0**（严格对齐，不是猜的）✅ |
| 连续性 | `frame_count` 1..96，步长集合 `{1}`，无丢 tick ✅ |
| 动作域 | `actions_model == (q_target − dq_ha)/sc_ha`，最大误差 1.2e-5 ✅ |
| 状态序 | `corr(mean q, dq_std)=0.735` vs `dq_ha=0.038` ⇒ 已是 STANDARD ✅ |
| 动作/状态对齐 | `corr(a_i, q_HA_i)` 均值 0.414（与 A′ 的 0.47 同量级）✅ |
| `.partial` 残留 | 0 ✅ |
| 无 Δ | server 命令行无 `--delta-v5-ckpt`；日志无 `delta-v5]`；`head_ckpt MISSING` ✅ |

### 3.4 已知限制（如实记录，不掩盖）

1. **`q_applied`** 取的是 `robot.data.joint_pos_target`（provider 侧裁剪后的目标，rad/HA 序）——ISAAC 没有 PhysX 目标回读，所以这是"最接近实际下发"的值，不是执行器内部量；
2. **`done`** 由采集器在 episode 结束后**外部补写**（Isaac 强制退出会跳过 `atexit`，实测过）；`done_reason` 来自 `result.json`；
3. **尾部 tick** 最多丢 `chunk-1 = 15` 个（默认 chunk 16）；正式跑 3000 步时 <0.5%；
4. **`xi`(33)** 仍是 `default_xi`（[0.75, 0…0]），`normalize_xi` 之后恒为 0 —— 与 WM0 训练时一致，但不携带参考信息；
5. 视频默认关闭（`--video off`），只保留 `result.json` + tick + 日志。

---

## 4. 交接协议（容器 B 只需三条）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
source runs/wm_base6_v1/control/collector_B.env
python3 scripts/collect_base_wm_v2.py --tasks pp_box double_desk boxing \
  --seeds 0,1,2,3,4,5 --max-steps 3000 --sim-gpu 0 --vla-gpu 1 --video off \
  --port 18222 --out-root runs/wm_base6_v1/collector_B
```

容器 B 的边界：只写 `runs/wm_base6_v1/collector_B/`；不改代码/Git/模型/A 目录；不训练。
容器 A 命令（本机）：同二进制，`--tasks open_door sit_sofa vision_navi --port 18220 --out-root runs/wm_base6_v1/collector_A`。

预估：单集约 3000 步 ≈ 12–14 min（含 server 复用），18 集/容器 ≈ **4 h**（双卡各 1 Isaac）。

---

## 5. 下一阶段放行条件

| 阶段 | 放行条件 | 现状 |
|---|---|---|
| P0 合同 | `audit_action_contract.py` 返回 0；换序代价已量化 | ✅ 通过 |
| P1 采集器 | 冒烟 6 项检查全绿；READY 已发布 | ✅ 通过（open_door） |
| P1 全量采集 | A/B 各 18 集 × 3000 步，`.partial`=0、连续性全 `{1}` | ⏳ 待启动 |
| P3 审计/合并 | `audit_base_wm_v2.py` + `merge_base_wm_v2.py`（待写） | 未开始 |
| P4 候选 WM | 80/20 混采、从 `fc5aeb30` 初始化、100–150 updates | 未开始 |
| P5 物理门禁 | 5 道门禁，`dir_agree` 必须真跑（禁止 `allow_missing_gradient_probe`） | 未开始 |

**必须先解决的一件事**：`wm_c1_v001_published`（当前 active，sha `81d24caf`）是在**错误的 on-policy 数据**上训出来的（见 §2.3），它的 gate PASS 不成立。候选 WM 一律从 `wm_v000`(`fc5aeb30`) 初始化，并把 active 指针回滚到 `wm_v000` 或至少标记 v001 为 invalid。
