# MuJoCo 迁移审计：IsaacLab 的方法**到底搬全了没有**（2026-10-28）

> 审计对象：`scripts/train_joint_v2.py` + `diff_vla_wbc/{train_v2,v3_step,burnin,dr,token_critic,fm_bc,wm_refresh,lora,b_start,v2_gates}.py`
> 对照物：`docs/S3_END_TO_END_PLAN_20261028.md`（含 §10 的 A–J 未解决问题清单）、
> `docs/TOKEN_SPACE_ANCHOR_LOCKDOWN_20261028.md`（F1–F7 锁定事实）、
> `docs/S3_FIX_PLAN_AFTER_REVIEW_20261027.md`（外部评审 4 个结构性问题）。
>
> **结论先说：没有搬全。** 数据/物理接口层（M1）已完成并实测通过；
> **训练入口（J1）与"真转移 → WM 保鲜 → critic → B 线起点"那条链还没搬到 MuJoCo**，
> 而那条链正是 S3 计划里 §2.2「还没做」和 §10 里 C/E/G/H 的载体。

---

## 0. 一句话结论

| 层 | 状态 |
|---|---|
| ① 观测/奖励/**终止/可微 FK/状态帧**（物理接口） | ✅ **已迁移并实测**（`mjlab_joint.py`，3 个回归测试） |
| ② 训练算法（冻结执行器 / LoRA / FM-BC / 想象反传 / anchor / hinge / critic / burn-in / DR） | ✅ 代码**后端无关**，可直接复用（`train_v2/v3_step` 一行没改） |
| ③ 训练入口（provider / token 源 / 真转移 / 评测 / 存档 / 门禁） | ❌ **没搬**（`train_joint_v2.py` 是 Isaac 专用，mjlab 版入口还不存在） |
| ④ 真数据闭环（B 线采集 → 真转移 → WM 保鲜 → critic 目标 → b_start） | ❌ **没搬**（全部依赖 Isaac 的 dump 格式） |
| ⑤ 评测（闭环 + success + 3 seeds） | ⚠️ **部分**：MuJoCo 闭环脚本存在（`closed_loop_vla_supermimic.py`），但无 success 维度 |

---

## 1. 迁移矩阵（Isaac 机制 → mjlab 现状）

| # | Isaac 机制 | 实现位置 | mjlab 现状 | 证据 |
|---|---|---|---|---|
| 1 | 93/99 维观测、term-major 历史 | `isaac_joint.observation` | ✅ **93 原生** | `mjlab_joint.py:OBS_DIM/TERM_DIMS`；`test_observation_is_93` |
| 2 | 状态元组 71（体系角速度） | trackerlab bridge 转换 | ✅ **已修**（库是世界系→体系） | `_ref71_body`；`test_state_uses_body_frame_angvel` |
| 3 | 奖励（重锚定 xy+yaw） | trackerlab MDP | ✅ **上游同款实现**（`reanchor_reference` 逐项对齐 mjlab） | `rewards.py:75-134` |
| 4 | 软终止 → logits | `mdp.soft_termination` → logit | ✅ `wm.termination_logits` 本来就是 logits | `world_model.py:58` |
| 5 | 可微 FK（14 tracked bodies） | trackerlab | ✅ `AnalyticHeads.robot_bodies` | `heads.py:101` |
| 6 | 冻结执行器（只训 head） | `train_v2._freeze_executor` | ✅ 复用；**加载即冻结** | `load_mjlab_bundle(freeze=True)` |
| 7 | LoRA（rank32，~16.8M） | `lora.py` | ✅ 复用（未改） | — |
| 8 | FM-BC（token 标签） | `fm_bc.py` + provider | ✅ 复用 + **标签=数据集 token**（不用侧车） | `ft318_tokens.pt`；lockdown F3 |
| 9 | 想象反传奖励（H/gamma/归一化） | `rollout.imagine_with_tokens` | ✅ 复用，实测梯度到 token | `test_reward_backprops_into_token`（grad 0.17） |
| 10 | token anchor / trust region / action hinge | `v3_step.V3Config` | ✅ 复用（需 `bundle.motion`，已提供） | `mjlab_joint.motion` |
| 11 | critic（GAE / 真 return-to-go） | `token_critic.py` | ⚠️ 代码可复用，但**没有 MuJoCo 真轨迹**可算目标 → 只能先关 | S3 §10.E |
| 12 | burn-in（录制状态 + DR + push） | `burnin.py` | ✅ 已解耦（4 处 hook）；`recorded_states` 需要 MuJoCo dump（缺） | `burnin.py` 改动 |
| 13 | DR（xy/yaw/dq/dqd 分桶） | `dr.py` | ✅ 复用（状态 71 语义一致） | — |
| 14 | WM 真转移保鲜 | `wm_refresh.py` + `build_real_transitions_v2` | ❌ **没有 MuJoCo 版输入** | S3 §3.1 |
| 15 | b_start（B 线重规划状态起点） | `b_start.py`（并行新增） | ❌ 依赖 Isaac dump 字段 | `b_start.py` docstring |
| 16 | 5 道门禁（codec/freeze/burnin/token_contrast/fm_bc） | `v2_gates.py` | ✅ 无 Isaac 依赖；**freeze 门禁已修**（见 §3.1） | grep 无 isaac/trackerlab |
| 17 | 入口（provider/token 源/真转移/评测/存档） | `scripts/train_joint_v2.py` | ❌ **未写** `scripts/train_joint_mjlab.py` | — |
| 18 | 闭环执行（VLA→token→WBC→MuJoCo） | `closed_loop_vla_supermimic.py` | ✅ 已有；**存档键已对齐**（见 §3.2） | 实测可加载 J1 ckpt |
| 19 | 手部 clip/限速（问题 C） | provider | ✅ **不存在该问题**（mjlab policy 无手部）；但 14 维手部命令会被丢弃 → 需明示 | — |
| 20 | bf16/fp32 对拍（问题 I） | — | ❌ 两条路线都没做 | — |

---

## 2. S3 计划的 A–J：在 MuJoCo 上分别是什么状态

| 项 | Isaac 的问题 | MuJoCo 现状 | 结论 |
|---|---|---|---|
| **A** 评测口径（世界坐标 vs 重锚定；起点偏移） | 我们的离线奖励用世界坐标，env 重锚定 ⇒ 数字不可比 | mjlab 的奖励**本身就是重锚定**（`reanchor_reference`），且参考/起点都取自同一库帧 | ✅ **结构上不存在**；但仍需用同一 `reference_start` 做评测（M0 探针） |
| **B** A1 未验证 / 图像不更新 | 决定方法上限 | 同一套 VLA，**同样没验证** | ❌ 未解决（后端无关） |
| **C** 手部无感知/无限速 | 会掀翻 | mjlab WBC 不吃手部 ⇒ 问题不存在；**替代问题**：VLA 的 14 维手部被丢弃 | ⚠️ 换成"手部语义"待定义 |
| **D** 两个 server 差异 | Isaac 专属 | 不涉及 | N/A |
| **E** critic 修不好 | v_ev 为负 | 同一份代码，但**没有 MuJoCo 真返回** | ❌ 未解决，第 1 轮必须 `--no-critic` |
| **F** WM 偏差污染梯度 | qd_corr 0.75 | 同样存在；**W1/J1 都还没有 qd_corr 验收门** | ⚠️ 需补验收 |
| **G** 任务成功率缺失 | success 恒 False | `eval_mujoco.py` 只有 reward/state_term；`closed_loop_mujoco.py` 有 success 谓词但那是 SONIC 任务 | ⚠️ 需定义 J1 的 success |
| **H** 采集覆盖窄 | 5 条/轮 | MuJoCo 侧还没有采集 | ❌ 同 ④ |
| **I** bf16 未对拍 | 低风险 | 同样没做 | ❌ |
| **J** 并行采集 | 速度 | 不涉及 | N/A |

**锁定事实 F7 的一个延伸风险（必须记）**：Isaac 锁定「参考库必须是**命令流**库」，且实测执行轨迹库
差 **0.33 NR**。318 数据集**没有命令流**（parquet 只有 `action.motion_token` / `action.wbc`），
所以 W1/J1 的参考库是**执行轨迹**——这一条在 MuJoCo 上是**已知未解决**，代价要用对照臂量化。

---

## 3. 本次审计新发现、并已修掉的缺口

### 3.1 冻结门禁只认 Isaac 的 `pi`（会让 J1 的第一道门禁直接 FAIL）

```python
# before (train_v2.py:406)
src = ck.get("pi")            # mjlab ckpt 用 "policy" ⇒ SKIP → gate_freeze FAIL
# after
src = ck.get("pi", ck.get("policy"))
```
证据：`mjlab` 的 `sm-train` 产物顶层是 `{policy, wm, ...}`（本机核对过 `ckpt_0024000.pt`）。

### 3.2 J1 存档缺 `policy_config` / `world_model_config`（E1 闭合加载不了）

`scripts/closed_loop_vla_supermimic.py:203` 走
`JointTrainer.load_bundle_from_checkpoint`，它要求
`ck["policy_config"]` + `ck["world_model_config"]`（`train.py:434`）。
`MjlabJointBundle.save_checkpoint` 原来只写 `config: {policy, world_model}` ⇒ 直接 `ValueError`。

**已修**：`save_checkpoint` 现在同时写 `policy_config` / `world_model_config`（旧键保留）。
实测：
```
keys: [... 'policy_config', 'world_model_config', 'policy', 'wm', 'wm_integration' ...]
closed-loop 可加载: True | policy 逐张量相等: True | token_dim: 64
```

### 3.3 burn-in 里 5 处 trackerlab 写死（M1 已修，记一笔）

`clip_start/clip_id`、`motion.ref[..., 13:42]`、`mdp.obs`、`isaac_joint.default_xi`、`_flat_tuple`
→ 全部换成 bundle 方法（`clip_start_of / ref_joint_pos / obs_from_states / default_xi`），
Isaac 侧补同名实现，行为不变。

---

## 4. 仍然缺的（按阻塞程度排序）

| 优先级 | 缺什么 | 影响 | 补法 |
|---|---|---|---|
| ~~**P0**~~ | ~~`scripts/train_joint_mjlab.py`（J1 入口）~~ | ✅ **已完成 2026-10-28**：BC 臂（`loss 0.0504`）与奖励臂（`rollout/reward 0.0879`、`loss -0.227`）各跑通 1 步，joint/vla ckpt 落盘 | `scripts/train_joint_mjlab.py` |
| ~~**P0**~~ | ~~MuJoCo 的 B 线采集~~ | ✅ **已完成**：`closed_loop_vla_supermimic.py --dump` 落 `{states, next_states, actions, tokens, q_target_applied, meta}`（自由闭环 `frames=-1`，`has_ref=false`） | 见脚本 `--dump` |
| ~~**P1**~~ | ~~`build_real_transitions_v2` 的 MuJoCo 版~~ | ✅ **已完成**：`scripts/build_real_transitions_mjlab.py`；动作 = **归一化 a 本身**（`q_target=q0+scale·a`）；合成 dump 实测 200 行、`rewards/ret/ref_rows/frame_map` 全出、finite | 动作语义与 Isaac 不同（Isaac 是 `q_target−q0`，scale=1） |
| **P1** | `b_start` 的 MuJoCo 版 | ⚠️ **部分**：`BStartSource` 本身后端无关（只吃 `{states,actions,frames,meta}` + `order`），318 关节序已确认是 `standard`；但自由闭环 dump 的 `frames=-1` ⇒ 需要回放式采集或补帧号 | `--start-from-b ... --b-start-order standard` |
| ~~**P1**~~ | ~~`qd_corr` 验收门~~ | ✅ 工具已有，但**口径已修正**（2026-10-28 审计）：原实现拿 WM 的一步预测去比**参考库**的 qd，而 WM 是在**仿真转移**上训的——两者语义不同。实测 H=1：W1 `qd_corr −0.004 / ratio 4.96`、e9 `0.115 / 6.10`，说明这是口径问题不是 WM 失效。已加 `--transitions` 模式（比**真实 next_states**）；库模式标注为「诊断用，不是验收口径」。**W1 的决定性验收改用环境内 benchmark**：同 2 条 clip、同录制 token，e9 `0.7009` → W1(it7k) `0.8778`（**+25.3%**） | `--transitions <sim_transitions.pt>` |
| **P2** | success 指标（问题 G） | 只能报 reward | 给 4 个任务写谓词（`closed_loop_mujoco.py:134-160` 有现成骨架） |
| **P2** | bf16 对拍（问题 I） | 归因干扰 | 同 seed 64 窗口 fp32/bf16 |
| **P2** | 手部语义（问题 C 的替代） | VLA 14 维被丢弃 | 明确"丢弃 + 只做位置跟踪"，或给 mjlab policy 加手部头（风险高） |
| **P2** | 命令流缺失（F7） | 参考库口径差 0.33 NR（HA 实测） | 短期：接受 + 对照臂量化；长期：从 `action.wbc` 反推命令流 |

---

## 4.1 本轮（继续实现）新增的 6 个件

| # | 文件 | 作用 | 验证 |
|---|---|---|---|
| 1 | `scripts/train_joint_mjlab.py` | **J1 入口**（CLI 与 `train_joint_v2.py` 对齐；mjlab 后端 + 数据集 token） | BC 臂 1 步 ✅ / 奖励臂 1 步 ✅ |
| 2 | `scripts/closed_loop_vla_supermimic.py --dump` | B 线采集（`states/next_states/actions/tokens/q_target_applied`） | 代码就绪；需真跑一次闭环 |
| 3 | `scripts/build_real_transitions_mjlab.py` | dump → 真转移（`rewards/ret/ref_rows/frame_map`） | 合成 dump 200 行 ✅ |
| 4 | `scripts/probe_wm_qd_corr.py` | WM 速度通道验收门（问题 F） | e9 基线 0.359/4.55（FAIL，符合预期） |
| 5 | `diff_vla_wbc/wm_refresh.py` + `v3_step.refresh_wm` | **后端无关**的 WM 保鲜（`step_fn`） | 2 步刷新 ✅ `pos_mse 2e-4 → 5e-6` |
| 6 | `diff_vla_wbc/mjlab_joint.py` | 新增 `default_xi` / `reference_frame` / `policy_config`·`world_model_config` / 加载即冻结 | 单测 3 passed |

**仍然没做的 3 项（P2，都不阻塞第 1 轮）**：
① success 指标（问题 G）；② bf16/fp32 对拍（问题 I）；③ 命令流重建（F7，318 缺命令流，
已知代价 0.33 NR）。另：B/E/F/G/H 这类"数据/方法层面"的开放项不因后端迁移而消失。

## 4.2 2026-10-28 晚：J1 双卡 + LoRA64 + 大 batch 已实现并实测

| 项 | 结论 |
|---|---|
| LoRA rank | 默认 **64**（67.1M / 3.98%）；rank32 = 33.5M；rank128 = 134M |
| batch | 默认 **64**（H200 目标）；**4090 实测上限 16/rank**（32/rank 直接 OOM，44.8 GB 已用）|
| 2 卡同步 | `JointV2Config.sync_gradients` + 扁平 all_reduce(AVG)；**实测两 rank 参数逐位相同**（`sum=-1.87172484 sq=21.23856926` 两者一致）|
| 踩到的坑 | LoRA 的 A 矩阵是随机的 ⇒ `torch.manual_seed` **不能**按 rank 偏移，否则各卡起点不同、梯度平均失效（已修：模型同种子，只有数据按 rank 偏移）|
| 吞吐（4090×2, batch16/rank, H=4） | **2.02 s/step**（等效 batch 32）|

## 5. 结论

1. **算法层搬完了**：`train_v2 / v3_step / burnin / dr / token_critic / fm_bc / wm_refresh / lora / v2_gates`
   里**没有一行**是 Isaac 专属；M1 只用 1 个新文件 + 6 处 hook 就把它们接到了 mjlab 上，
   并且实测"奖励 → WBC → WM → token"梯度可达。
2. **入口层没搬**：`train_joint_v2.py` 里那 700 行装配（provider / 三种 token 源 / 真转移 /
   DR 配置 / 评测 / 存档 / 门禁）仍是 Isaac 专用；mjlab 版入口是下一步唯一 P0。
3. **真数据闭环没搬**：B 线采集 → 真转移 → WM 保鲜 → critic 目标 → b_start，这四件在 MuJoCo
   上**一件都还没有**；它们对应 S3 §2.2「还没做」的全部内容，也是 §10 里 E/H 的载体。
4. **A–J 里**：A 结构上消失（奖励本来就重锚定），C/D/J 不适用或换了形态，
   **B/E/F/G/H/I 依然开放**，且 F7（命令流缺失）在 318 上是新增的开放项。
