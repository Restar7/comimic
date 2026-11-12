# B-WM6 主线：完整评审材料（给 GPT 修改用）

> 生成：2026-11-09 10:57 CST｜容器 2（B 线）｜分支 `experiment/b-wm6-h10-critic`
> worktree：`/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-b-wm6`
> **本文所有数字都可用 §10 的命令复跑，没有估计值。**
> 状态：B0/B1/B3 已完成；`c3_v5`（β=1 + v5 critic）**正在跑**，最新观测到 it=370；B4 六任务闭环 Base 臂 18/18 完成、B 臂只跑了 4 集（被叫停，理由见 §8.1）。

---

## 0. 给 GPT 的 6 个问题（先看这个）

| # | 问题 | 我的倾向 | 卡在哪 |
|---|---|---|---|
| **Q1** | **β=1 明显在伤 J10**（证据见 §7.3）：β=1 时 val ΔJ 从 +0.00914 掉到 +0.00235；β=1 到底该给多少？是否应该"β 上限由 critic 健康度动态定"？ | 降到 0.3–0.5，或改成 `β = β_max·1[v_ev≥τ]` 且 β_max<1 | 任务书要求 β 目标 1.0，但实测 1.0 有害 |
| **Q2** | critic 的 `v_ev` 随策略漂移单调衰减（0.62→0.40→0.32，10 步一个点）。**在线共训的频率/容量/目标**该不该改（更多 critic step？更短 EMA？冻结 actor 的 critic-only 预热阶段？） | 每 N 个 actor update 做一次"critic-only 追赶"（actor 冻结） | 现在 4 次 critic step/actor step 追不上 |
| **Q3** | `imagined_value_v5` 的 `plan` 特征是 `Z_{t..t+3}`（4 个 token）。**要不要加大到 8/10 个 token 或加 `remaining_steps` 的显式编码**？ | 加到 8 个 | 未见消融 |
| **Q4** | 真物理 open-loop 的 ΔJ 小于 A/A 噪声（§6.3）。**是否应该直接用闭环成功率做唯一判据**（放弃 open-loop 作为门禁）？ | 是 | open-loop 需要每臂独立进程，成本高且信号弱 |
| **Q5** | `rot6` 在部署侧是"相对回合首帧"，训练侧是"相对 clip 首帧参考"（§5.4）。实测敏感度只有 4.1%，**要不要干脆把 rot6 通道置零**（训练+部署都置零，彻底消除这个隐患）？ | 置零 | 需要改 client（`action_provider_sonic`），属于跨容器改动 |
| **Q6** | 六任务闭环每任务只有 3 seeds（18 集/臂）。**统计上够不够下 PASS/FAIL？** 若不够，应补到 6 seeds（再多 ~3 h）还是用"任务级进度+失败模式"定性判定？ | 补到 6 seeds 才有资格判 PASS/FAIL | 时间 |

---

## 1. 任务边界与"本轮唯一必要的大修改"

任务书原始要求：用新 WM6 从零初始化原 B 路线的 `TemporalDeltaActorV5`，**同时在线联合训练独立 TokenCritic**，H10 可微 Tracking + Critic 尾部价值监督，实际部署 K20，最后在六任务真实 VLA–SONIC–Isaac 闭环里验证相对官方 Base 的效果。

我把它落成 **5 条** 唯一必要的大修改（其余一律"恢复原 B 配方"）：

1. 新 WM6（`8edb59cf`, s350）替换旧 `wm_gain2`；
2. `H_train` 20 → **10**；
3. 打开**在线联合** critic（β 0→1，受独立验证门控）；
4. 修正**训练/部署时序**与**动作 rad 口径**；
5. **强制真实 Isaac 闭环验收**。

**明确不继承**：A 原版 V5 warm-start、B 旧 checkpoint、B_v8b/V8.2 的 worst-window recovery、C 线 LoRA16 配方、WM 自动重训 / active 指针自动发布。

---

## 2. 资源与隔离

| | 本容器（B） | C 容器 |
|---|---|---|
| GPU0 UUID | `GPU-618f618a-b436-477c-83cd-037d1e9fe845` | `GPU-272c06bc-4338-1cec-7fa0-298692b90d75` |
| GPU1 UUID | `GPU-0e64fd48-2e1d-74ba-7200-671951b1e1e7` | `GPU-12b4dd7c-c3b7-67f6-815c-00e274c95d83` |

* **按 UUID 判断隔离**（不看 `cuda:0` 编号），无重叠。
* 训练卡位（实测显存）：**GPU0 = 冻结 FT8000 在线 token 生成**（峰值 27.3 GB）；**GPU1 = Δ Actor + Critic + 冻结 SONIC + WM6 BPTT**（峰值 ~2.4 GB）。VLA 是瓶颈，实测 **12.7–14.0 s/update（B=256, H=10）**。
* 共享控制文件 `runs/dual_actor_wm6/control/` 是**软链**到 `comimic-c-wm6/.../control`（单一真相源，只读）。
* 16 核 → 实际 128 核 / 1 TB RAM；Isaac 单实例 ~15 GB 显存、~7 GB RSS，**同卡最多 2 实例**。

---

## 3. 版本锁（B0 逐项重算 sha，与 C 容器一致）

| 对象 | sha256 前缀 |
|---|---|
| WM6 `wm_base6_best_candidate/wm.pt`（s350） | `8edb59cf` |
| FT8000 `checkpoint-8000/config.json` | `7d06d0da` |
| SONIC Release `model_decoder.onnx` | `c7241a12`（低延迟 `c4ac2e74` 禁用） |
| WBC executor `WBC_A24000_wmvel_cmd.pt` | `2390dc29` |
| packed motion `ha_cmd/packed.pt` | `3f444d50` |
| eval_set `windows.pt`（256/96/96） | `e2f8586e` |

记录：`runs/dual_actor_wm6/b_delta/config/b_lock.json`（含物理合同、GPU UUID、control 快照）。

---

## 4. 架构

### 4.1 两条链

```
部署链:  Isaac RGB + 指令 + 本体状态 → 官方 FT8000(冻结, 无 LoRA) → Z_base [40,64]
        → Δ V5 (1,744,832) → SONIC Release(冻结) → Isaac 真物理
        → 每 20 tick 重规划（HA_VLA_EXEC_STEPS=20）；history 逐 tick 更新
        ⚠️ 部署链**不含** WM6 / Critic

训练链:  同一 Z_base → Z_base + Δ → SONIC(冻结) → WM6(冻结, 8.82M) → J10 + β·γ^10·V(y10) → Δ
```

### 4.2 Actor（`diff_vla_wbc/delta_token_temporal_v5.py`，未改动）

```
输入  Z (B,40,64) | state64 (B,64) | hist10 (B,10,58)
hz   = token_proj(Z) + PE_chunk[:40]                  # 64→128
c_t  = LayerNorm(TransformerEncoder(1 block, d128, 4 head, FFN256, pre-LN))(hz)
h    = GRU(58→128)(hist10).last                       # 广播到每帧
x_t  = [ z_t(64) | c_t(128) | h(128) | state64(64) | PE_pos(t)(16) ]      # d_in=400
u_t  = MLP(400→1024→1024→64, SiLU)                    # 末层 N(0,4e-4)、bias=0
Δ_t  = apply_norm_cap(u_t, cap=0.08) · 1[t < delta_k]
```

参数分解：`token_proj 8,320 + PE_chunk 5,120 + Transformer 132,480 + LN 256 + GRU 72,192 + PE_pos 640 + MLP 1,525,824 = **1,744,832**`。

**初始化**：从零（`--actor-init scratch`），末层 `N(0,4e-4)` —— **刻意不零初始化**，因为零初始化会让上游第一步梯度全 0。实测 init `|Δ|max = 1.85e-3`、`mean_norm 2.7e-3`，且 `∂L/∂token` 非零（B0 #12）。

### 4.3 张量合同（B0 逐位验证）

| 接口 | 定义 | 验证 |
|---|---|---|
| WM `state71` | `[root_pos3 | quat_wxyz4 | q_STD29 | v_w3 | w_w3 | qd_STD29]`，dt=0.02 | 与共享锁一致 |
| WM `action29` | **HA(SONIC) 序**、`(q_target_HA − dq_HA)/sc_HA` | 记录数据**逐位复现**（max 4.2e-6，640 tick） |
| Δ `state64` | `[rot6d6 | q_STD29 | qd_STD29]`，训练侧 standard；入站 ha_v31 由 spec 转 | B0 #15 |
| Δ `hist10` | `[q − **g1_default**(29) | qd(29)]`，**standard 序**，最老→最新 | B0 #15 逐位 0 差；部署转换 #16 逐位 0 差 |
| SONIC | torch 实现 vs `onnx.reference.ReferenceEvaluator`（独立 NumPy） | **max|Δ| = 4.8e-7** |

⚠️ 注意 `hist` 的 q 偏移是 **`g1_default_joint_pos()`**（hip −0.312 / knee 0.669 / elbow 0.6），**不是** `bundle.default_joint_pos`（SONIC 口径 hip −0.2 / knee 0.42 / elbow 0.87）。两者 elbow 差 **0.27 rad**；用错就会静默错位（B0 #15 的 `gap_if_wrong_default=0.27` 就是这条）。

### 4.4 已知近似：rot6 口径（必须记录，不是零）

部署侧 `state64[:6]` 是 **相对回合首帧朝向**的 rotlocal6；训练侧是**相对 clip 首帧参考朝向**。HA 会在回合开始把机器人 reset 到参考起始位姿，所以近似相等，残差来自 burn-in 扰动。
**量化**：把 rot6 整块置零，Δ 的 cosine = **0.9991**、相对变化 **4.1%** ⇒ 影响小但非零。

### 4.5 Critic（本轮换了两次）

| | `TokenCritic`（第一版，已弃用） | `ImaginedValueV5`（当前） |
|---|---|---|
| 输入 | `s71 + ref71 + xi33`（+可选 s−ref） | `s71+ref71+(s−ref)71+xi33+hist10(580)+plan Z_{t..t+3}(256)+extra4` |
| in_dim | 175 / 246 | **1086** |
| 参数 | 111,105 | **819,713**（hidden 512×2） |
| 目标 | TD(λ) 0.95 + EMA 0.995 target | 同 |
| 独立 clip `v_ev` | **+0.067**（最好）/ 在线 +0.145 / 真物理 **−6.06** | **+0.626**（离线 best）/ 在线 +0.62→+0.32（衰减） |

---

## 5. 代码改动清单

新增文件（**不改运行中的 v5 训练器**，遵守"运行中的脚本不原地编辑"）：

| 文件 | 作用 |
|---|---|
| `scripts/b_env.sh` | B 线环境：`$WT` 顶到 PYTHONPATH 最前 |
| `scripts/b_wm6/contracts.py` | 版本锁 + G1 物理合同（MJCF 真实限位；真 rad 尺度） |
| `scripts/train_b_wm6_h10_critic.py` | **B 线正式训练器**（H10+K20+在线 critic，`--critic-kind {token,v5}`，严格 resume） |
| `scripts/b_wm6/b0_tests.py` | B0 门禁 27 项 |
| `scripts/b_wm6/pretrain_critic.py` | TokenCritic 离线预训练（MC/TDλ 可选） |
| `scripts/b_wm6/critic_v5_probe.py` | **ImaginedValueV5 先验实验**（MC 目标，独立 clip dev） |
| `scripts/b_wm6/analyze_run.py` | 逐任务 ΔJ / Δ 分段 / rot6 敏感度 / 梯度分离 |
| `scripts/b_wm6/build_replay_job.py` / `score_replay.py` / `score_pair.py` | 真 Isaac open-loop 回放 + 配对打分 + A/A 噪声 |
| `scripts/b_wm6/run_isaac_roll.sh` | 跑一个回放 job，**输出出现即杀进程**（`app.close()` 会永久挂住） |
| `scripts/b_wm6/export_deploy_ckpt.py` | 训练态 → 部署可用 v5 checkpoint |
| `scripts/b_wm6/run_isaac_pair.sh` | Base vs B 真 Isaac 闭环驱动 |
| `scripts/b_wm6/summarize_b4.py` | 六任务结果汇总（同 (arm,task,seed) 保留最新） |

### 5.1 修掉的 5 个 v5 真 bug（都会让"H10+Critic"名不副实）

| # | v5 原写法 | 后果 | 本实现 |
|---|---|---|---|
| 1 | `states[H].detach()` 进 critic | `β·γ^H·V(s_H)` 对 Δ **梯度恒 0**（"开了 critic 只当常数"） | 不 detach；`FrozenParamCtx` 只关参数 `requires_grad` |
| 2 | `refs_f[:, H]` | `refs_f` 只有 H 列 ⇒ `IndexError`（该分支从未跑过）；且 critic 要的是 **71 维参考状态**，不是帧号 | 现取 `teacher_references(..., H+1)[H]` + `ref_state_at()` |
| 3 | `adapter.as_dict()`（token 分支 `adapter=None`） | `it%50==0` 时 `critic_gate.json` 写入崩溃 | token 分支不再引用 adapter |
| 4 | `beta` 只在 `it%50==0` 更新 | 前 49 步恒 0、之后 50 步一台阶 | 逐 update `min(1, it/W)` |
| 5 | checkpoint 不含 critic/target/优化器/RNG/β | **无法 resume** | 全量保存 + locks sha 对账；B0 #25 验 **bitwise** 复现 |

### 5.2 动作 rad 口径修正（重要，影响所有"mrad"数字）

* 记录器 meta 的真相：`q_target_ha = dq_ha + sc_ha · actions_model`，**逐位复现 max|Δ|=4.2e-6**。
* `sc_ha` 均值 **0.378852**；`dq_ha[HA→STD]` 与 `bundle.default_joint_pos` **逐位相同**。
* 但 `bundle.action_scale` 是**全 1.0**（`action_scale_mode=one`）。
  原 B 的 `action_trust_v5.q_target_from_action(default, bundle.action_scale, a)` = `default + 1.0·a`
  ⇒ **把归一化动作当 rad**，量级偏大 ≈ 1/0.3789 = **2.64×**。
* 本轮：`L_act/L_smooth` 仍在**原 B 的归一化单位**里算（τ=0.020 归一化 ≈ 7.6 mrad 物理 RMS），
  **所有报表**改用 `q_target_std = dq_std + sc_std·a` 的真 rad；
  `token_delta_norm` / `action_normalized_delta` / `joint_target_delta_rad` 三者严格分开。

### 5.3 修掉的 3 个环境 bug（真阻塞，全部实测定位）

| 症状 | 根因 | 修法 |
|---|---|---|
| Isaac 卡死在 `Starting the simulation`，永远到不了第一次 VLA 请求 | 容器里 `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` 指向**不存在的文件** ⇒ `vkCreateInstance` 报 `ERROR_INCOMPATIBLE_DRIVER` ⇒ `carb::graphics::createInstance failed` / `no suitable CUDA GPU` ⇒ PhysX 退软件并卡死 | `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` |
| `[SonicActionProvider] onnxruntime is not available` | B 线自己的 `PYTHONPATH` 含 `third_party/gr00t_deps`（内含**坏的** onnxruntime，缺 `onnxruntime_pybind11_state`），盖住 `isaacsim_env` 里好的副本 | 跑 Isaac 时 `env -u PYTHONPATH -u VIRTUAL_ENV` |
| 回合跑完但找不到 `result.json` | `ha_isaac_closedloop.sh` 会 `cd $PROJECT_ROOT` 再 exec python，**相对** `OUT_ROOT` 被重新解析到 HumanoidArena 目录下 | `OUT` 一律转绝对路径 |

另外修了 `run_isaac_pair.sh` 的**死锁**：`wait`（无参）会把同 shell 后台起的 VLA server 也等进去（server 要等 `stop_server`，而 `stop_server` 在 `wait` 之后）⇒ 只 `wait` episode 的 PID。

### 5.4 修掉的两处**会伪造正收益**的方法学 bug（最该被 review 的部分）

1. `build_replay_job.actions_of()` 忘了把 `tokens` 传进 `masked_return` ⇒ Base/B 两臂**动作逐位相同**（`max|Δa|=0`）。
   **第一版报告的 "ΔJ10=+0.0014, CI>0" 就是假信号。**
2. `isaac_probe_action_rollout.py` 在同一进程里先后跑两臂，`write_state()` 只重置位姿/关节/速度，
   **不重置接触缓存与执行器内部状态** ⇒ 两臂动作**完全相同**时末状态仍差 **6.84**（比真实 Δ 大 3 个数量级）。
   ⇒ 现在**每臂一个全新进程**，并额外跑一份 `--dup` 的 A/A job。

---

## 6. 已完成的实验与结果

### 6.1 B0 门禁：**27/27 PASS**

`runs/dual_actor_wm6/b_delta/c0/c0_report.json`。关键项：

* `04_action_domain_exact` max|Δ|=4.17e-6（640 tick 反解）
* `09_sonic_torch_vs_onnx` 4.8e-7
* `10_actor_fresh_init` 1,744,832 / `d_in=400` / 末层 `N(0,4e-4)`
* `11_tail_strict_zero` `t≥K` 的 Δ **恰好 0**；`18_grad_leaks_to_tail?` `∂J/∂Δ[t≥K]` **恰好 0**
* `12_actor_init_near_zero` init|Δ|max=1.85e-3（≈0 但非零 ⇒ 首步有梯度）
* `13_cap_stability` 极端输入下 ‖Δ‖ ≤ cap
* `15/16_hist` 训练/部署口径逐位一致
* `17/19/20` 三路梯度：`Δ←J10`、`V←s_H`、`Δ←β·γ^H·V(s_H)` **全部非零**
* `21/22/23` TDλ 能更新、EMA 比正确、**critic loss 不改 actor 参数（恰 0）**
* `25_resume_bitwise` 参数与 forward max|Δ| **= 0**
* `27_full_trainer_step_2gpu` 真实 1-update 全链路 rc=0

### 6.2 B1（100 updates，21.9 min，**token critic**，tail 自由，v1 配方）

| it | val ΔJ | same-obs RMS | β | critic v_ev |
|---|---|---|---|---|
| 25 | +0.00574 | 7.95 mrad | 0.25 | −1.253 |
| 50 | **+0.00579** | 8.84 | 0.50 | −0.170 |
| 75 | +0.00470 | 13.4 | 0.75 | **+0.145** |
| 100 | +0.00403 | 14.2 | 1.00 | −0.231 |

* test（独立 clip，96 窗）：`J=0.83416 vs Base 0.83040 ⇒ ΔJ=+0.00376`
* 逐任务 val ΔJ **6/6 全正**：pp_box +0.00720 / sit_sofa +0.00632 / open_door +0.00586 / vision_navi +0.00584 / double_desk +0.00581 / boxing +0.00472
* Δ 分段（t<10 / 10–20 / ≥20）：**0.0736 / 0.0731 / 0.0000** ⇒ 无监督的后 10 帧和前半一样大
* `‖u‖/cap` 中位数 **2.28** ⇒ tanh 饱和（原 B v2 的 `L_raw` 就是修这个）
* 梯度分离：`‖grad βV‖/‖grad J10‖` 到 **2.69**，`cos` 只有 **0.085** ⇒ 价值梯度近乎正交

### 6.3 B2 真物理 open-loop（48 窗 × H=20，**每臂独立进程**）

| 指标 | Base | B(Δ) | Δ | 95% CI | 胜率 |
|---|---|---|---|---|---|
| 真 J10 | 0.55514 | 0.55561 | **+0.00047** | [+0.00011, +0.00099] | **72.9%** |
| 真 J20 | — | — | **+0.00043** | [+0.00008, +0.00093] | — |

**A/A 噪声底**（同 job 内 plus-vs-minus，动作逐位相同）：mean **+0.00143**、`|mean|` 0.00193、std 0.00431。

⇒ **ΔJ 与 A/A 噪声同量级甚至更小 ⇒ INCONCLUSIVE。** 方向正面（73% 窗口变好），但幅度只有 base J10 的 **0.08%**，且 A/A 有 **+0.0014 系统性正偏**（第二臂继承第一臂残余）。**不得**据此宣称真物理有正收益。

关节安全：两臂 `limit_viol=44.8`、`peak|q−q_default|=4.64 rad` **完全相同** ⇒ 越限来自 **SONIC decoder 原始输出**（部署侧由 `sonic_physical_action` 安全层裁剪），**不是 Δ 引入**；Δ 只叠加 ~7 mrad RMS。

### 6.4 B3 正式（300 updates，65.2 min，**token critic**，`tail_control=mask10`，v2 配方）

配置：`λ_raw=1e-3, κ=2, cap 固定 0.08`；`delta_k=10`（Δ 只在 t<10 生效，监督区间 == 生效区间）；critic 从零在线共训；β warmup 100 + **严格门控**（最近一次验证 FAIL ⇒ β 立即 0）。

| it | val ΔJ | same RMS | peak | β | critic v_ev |
|---|---|---|---|---|---|
| 25 | +0.00463 | 6.85 | ~21 | 0.25 | −1.252 |
| 50 | — | — | — | **0** | −0.176 |
| 100 | +0.00727 | 6.59 | ~22 | 0 | −0.141 |
| 200 | +0.00818 | 6.89 | ~23 | 0 | −7.70 |
| 275 (best) | **+0.00922** | — | — | 0 | — |
| 300 | +0.00914 | 6.78 | 23.5 | 0 | −23.5 |

* test：`J=0.83825 vs 0.83040 ⇒ **ΔJ=+0.00785**`
* 逐任务 val ΔJ **6/6 全正**：sit_sofa **+0.01314** / pp_box +0.00991 / vision_navi +0.00988 / open_door +0.00877 / boxing +0.00847 / double_desk +0.00697
* 真 rad 预算：same RMS **6.71 mrad**、peak 43.5、p99 21.3（B1 是 14.0 / 74.0 ⇒ **幅度减半**）
* `‖Δtoken‖(t<10)=0.0698`（cap 0.08）
* **B1→B3 的 2.1× 提升来源**：`tail mask`（去无监督 10 帧）+ `L_raw`（去饱和）+ `β=0`（不用未验证价值梯度）

### 6.5 v5 critic 先验实验（关键正结果）

同一批策略匹配 WM6 想象数据、同一 WM6、MC 目标、**独立 clip 的 val 96 窗**：

| critic | 特征 | dev `v_ev` | corr | mae（const 0.229） |
|---|---|---|---|---|
| TokenCritic(use_diff=0) | s+ref+xi | **+0.067** | 0.27 | 0.227（≈基线） |
| TokenCritic(use_diff=1) | +（s−ref） | −0.20（过拟合，train loss 0.0002） | — | — |
| **ImaginedValueV5** | +hist10+plan4+extra | **+0.6259** | **0.804** | **0.131（比基线好 43%）** |

⇒ **根因确认：TokenCritic 的信息集缺"未来 plan token / 运动相位 / 显式 s−ref"。**

### 6.6 `c3_v5`：β=1 + v5 critic 的续训（**正在跑**，从 c2b it=300 续到 420）

`--resume c2b/checkpoint_last.pt --critic-kind v5 --critic-init ckpt --critic-ckpt critic_v5_b3.pt --beta 1.0`

| it | β | J10 | J_ac | critic `v_ev` | val ΔJ | same RMS |
|---|---|---|---|---|---|---|
| 310 | 0 | 0.8016 | 0.8016 | **+0.620 PASS** | — | 6.59 |
| 320 | 1.00 | 0.7832 | 0.9004 | +0.592 PASS | **+0.00474** | 9.66 |
| 330 | 1.00 | 0.8121 | 0.9699 | +0.560 PASS | — | 11.6 |
| 340 | 1.00 | 0.7874 | 0.9425 | +0.521 PASS | **+0.00235** | 11.8 |
| 350 | 1.00 | 0.7897 | 0.9605 | +0.466 PASS | — | 12.9 |
| 360 | 1.00 | 0.8274 | 1.0150 | +0.402 PASS | +0.00270 | 15.1 |
| 370 | 1.00 | 0.7955 | 1.0064 | **+0.324 PASS** | — | 17.4 |
| 380 | 1.00 | 0.7831 | 1.0112 | **+0.246 ⇒ FAIL**（β 被门控归 0） | **+0.00166** | 17.5 |

**对照：β=0 的 it=300 时 val ΔJ = +0.00914、same RMS 6.78 mrad。**

### 6.7 B4 六任务真实闭环（Base 臂完整，B 臂被叫停）

部署链已确认**真的生效**（不是空跑）：
```
[server][delta-v5] #121 |Δ|mean(t<K)=0.0542 max=0.0569 tail_zero=True hist_last_q0=+0.177
```
`tail_zero=True` ⇒ t≥10 的 mask 在部署端也生效；`hist/state` 一致性门禁没报错。

| 任务 | Base（18 集，seeds 6/7/8） | B(Δ)（用 c2b 的 ckpt，只跑 4 集） |
|---|---|---|
| PPBox | 1/3 | **0/3** |
| DoubleDesk | 0/3 | 0/1 |
| Boxing | 2/3 | — |
| OpenDoor | 2/3 | — |
| SitSofa | 2/3 | — |
| VisionNavi | 1/3 | — |
| **合计** | **8/18 = 44%** | **0/4** |

---

## 7. 结论

### 7.1 站得住的

1. **工程链完全打通**：从零 Δ Actor → 冻结 SONIC → 冻结 WM6 可微 BPTT → J10/β·V(s10) 反传 → 严格 resume → 真 Isaac 部署（含 tail mask、hist/state 一致性门禁）→ 真闭环任务评测。
2. **actor 侧有稳定的 WM6 想象收益**：held-out test ΔJ **+0.00785**，6/6 任务为正，动作修正只有 6.7 mrad RMS。
3. **critic 的"能不能用"这关解决了**：换成 v5 imagined 特征后独立 clip `v_ev` **+0.626**、在线 +0.62→+0.32，**全程 > 0.30 门限**。

### 7.2 站不住的（必须写进最终判定）

1. **H10+Critic 不能判 PASS**：β=1 让 val ΔJ 从 +0.00914 掉到 +0.00235–0.00270，动作幅度从 6.8 涨到 15–17 mrad。critic 可用，但**β=1 这个权重本身有害**。
2. **真物理信号低于噪声底**：open-loop ΔJ10 = +0.00047 vs A/A +0.00143 ⇒ INCONCLUSIVE。
3. **六任务闭环没有形成结论**：Base 8/18，B 只有 4 集（0 胜），n 太小；且这 4 集用的是 **β=0 的 c2b**，不是 β=1 的 c3_v5。

### 7.3 β 的剂量-反应（本轮最有用的定量结论）

| β | val ΔJ | same RMS | critic `v_ev` | 结论 |
|---|---|---|---|---|
| 0.25–0.50 | **+0.0058** | 8–9 mrad | −1.25→−0.17（不可用） | 收益最高但 critic 不可信 |
| 0 | **+0.00914**（it=300） | 6.8 | −0.14 | **actor 最优**（纯 J10） |
| 1.00（critic 合格） | **+0.00235–0.00270** | 15–17 | +0.62→+0.32 | **β 伤 J10** |

⇒ 三点共线：**β 越大，ΔJ 越低，动作幅度越大**。而 critic 的 `v_ev` 在 β=1 下还在单调衰减（策略漂移）。
**这不是"critic 坏"，是"β·V 的梯度与 J10 冲突，且 β=1 的权重压过了 J10"。**

---

## 8. 当前问题清单

### 8.1 阻塞性（必须先解决才能判 PASS/FAIL）

1. **β 上限**（Q1）：β=1 有害。需要定一个"既不浪费 critic 又不动摇 J10"的 β，并且判据必须是**闭环成功率**而不是 ΔJ。
2. **critic 在线漂移**（Q2）：`v_ev` 每 10 步掉 ~0.04；β=1 下策略漂得比 critic 追得快。需要"critic-only 追赶阶段"或更密的 critic 更新。
3. **闭环样本量**（Q6）：18 集/臂 在 44% 基线下只能分辨 >25% 的绝对差；要判 PASS 需 ≥6 seeds（36 集/臂）。
4. **B4 的 B 臂还没在最终 ckpt 上跑过**：已跑的 4 集用的是 c2b（β=0）；`c3_v5`（β=1）还没进闭环。

### 8.2 非阻塞但需要记录

5. `rot6` 训练/部署定义不同（影响 4.1%，Q5）。
6. SONIC decoder 原始输出会越 MJCF 限位（7.7% joint-step、peak 4.64 rad），部署侧靠 `sonic_physical_action` 安全层裁剪；**WM 看到的是未裁剪的归一化动作** ⇒ 训练/部署之间还有一层"裁剪"没有被建模。
7. `L_lim` 在 Δ≈0 时就已经是 24.9（说明 base decoder 本身越限），所以本轮把它当**监控**而不是 loss（`--lambda-limit 0`）。
8. 后半 chunk 若不用 tail mask，Δ 会和无监督段一样大（B1 实测 0.0731 vs 0.0736）——本轮已用 `mask10` 消掉。
9. `hist` 的 default 是 `g1_default_joint_pos()`，与 `bundle.default_joint_pos` 差 0.27 rad（elbow）；`assert_hist_state_consistent` 的 tol=5e-3 会挡住错配。
10. 环境层：`VK_ICD_FILENAMES` 是容器级错误，**C 容器同样会中招**（如果它要跑 Isaac 闭环）；建议让两边都用 `/usr/share/vulkan/icd.d/nvidia_icd.json`。

---

## 9. 建议给 GPT 的修改方向（我的候选方案）

| 方案 | 做法 | 预期 | 风险 |
|---|---|---|---|
| **A. β 限额 + 健康度**（推荐） | `β = β_max · 1[v_ev ≥ 0.30]`，`β_max ∈ {0.3, 0.5}`；v_ev 掉到 0.45 以下就停止增加 β | 保住 ΔJ≈+0.008 的同时让 critic 有贡献 | 需要 2 个 β 臂各 300 updates（~2.5 h） |
| **B. critic-only 追赶** | 每 20 个 actor update 冻结 actor、只训 critic 200 步，把 `v_ev` 拉回 >0.5 再继续 | 抑制漂移 | 墙钟 +30% |
| **C. 用 v5 critic 但只做"筛选"不做"梯度"** | critic 只用于在若干候选 Δ 里选（不进 loss） | 完全绕开 β 冲突 | 与任务书"β 进 loss"不符 |
| **D. 加 plan 长度** | `plan_steps 4 → 8` | 价值更准（Q3） | 未见消融，需重训 critic |
| **E. rot6 置零** | 训练+部署都把 `state64[:6]` 置 0 | 消除 Q5 | 需改 C 侧 client（跨容器） |

---

## 10. 复现命令

```bash
WT=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-b-wm6
cd $WT && source scripts/b_env.sh

# B0 门禁（27 项，含 1 个完整 trainer step / 两张卡）
python3 scripts/b_wm6/b0_tests.py --device cuda:1 --vla-device cuda:0 --batch 32

# B3 正式（β 被门控否决那条线）
python3 scripts/train_b_wm6_h10_critic.py --out runs/dual_actor_wm6/b_delta/c2b/seed0 \
  --device cuda:1 --vla-device cuda:0 --updates 300 --batch 256 --horizon 10 --exec-k 20 \
  --tail-control mask10 --tail-cut 10 --lambda-raw 1e-3 --raw-norm-kappa 2 \
  --cap-warmup-updates 0 --beta 1.0 --beta-warmup-updates 100 --beta-gate soft \
  --critic-enabled 1 --critic-kind token --critic-updates 4 \
  --on-the-fly 1 --eval-set runs/temporal_ac_bptt_v5/eval_set

# v5 critic 先验
python3 scripts/b_wm6/critic_v5_probe.py --out runs/dual_actor_wm6/b_delta/critic_v5_b3.pt \
  --actor-ckpt runs/dual_actor_wm6/b_delta/c2b/seed0/checkpoint_best.deploy.pt \
  --train-windows 256 --dev-windows 96 --epochs 2000 --batch 1024 --lr 1e-3 --device cuda:0

# β=1 + v5 critic 续训
python3 scripts/train_b_wm6_h10_critic.py --out runs/dual_actor_wm6/b_delta/c3_v5/seed0 \
  --resume runs/dual_actor_wm6/b_delta/c2b/seed0/checkpoint_last.pt --updates 420 \
  --device cuda:1 --vla-device cuda:0 --batch 256 --horizon 10 --exec-k 20 \
  --tail-control mask10 --lambda-raw 1e-3 --raw-norm-kappa 2 --cap-warmup-updates 0 \
  --beta 1.0 --beta-warmup-updates 100 --beta-gate soft \
  --critic-enabled 1 --critic-kind v5 --critic-init ckpt \
  --critic-ckpt runs/dual_actor_wm6/b_delta/critic_v5_b3.pt \
  --critic-v5-hidden 512 --critic-v5-depth 2 --critic-updates 4 --critic-validate-every 10 \
  --on-the-fly 1 --vla-device cuda:0 --eval-set runs/temporal_ac_bptt_v5/eval_set

# 真物理 open-loop（每臂独立进程 + A/A）
python3 scripts/b_wm6/build_replay_job.py --ckpt <ckpt>.deploy.pt --out job_base.pt --dup --split val --n 48 --horizon 20
python3 scripts/b_wm6/build_replay_job.py --ckpt <ckpt>.deploy.pt --out job_b.pt    --split val --n 48 --horizon 20
bash scripts/b_wm6/run_isaac_roll.sh job_base.pt roll_base.pt 1
bash scripts/b_wm6/run_isaac_roll.sh job_b.pt    roll_b.pt    1
python3 scripts/b_wm6/score_pair.py --job-a job_base.pt --roll-a roll_base.pt \
  --job-b job_b.pt --roll-b roll_b.pt --b-ckpt <ckpt>.pt --out pair.json

# 六任务闭环（Base vs B；OUT 必须绝对路径）
W=$(pwd); TASKS="PPBox DoubleDesk Boxing OpenDoor SitSofa VisionNavi" SEEDS="6 7 8" \
  MAX_STEPS=3000 ARMS="base,d" PAR=2 ISAAC_GPU=0 SRV_GPU=1 SERVER_PORT=18242 RECORD_VIDEO=0 \
  D_CKPT=$W/runs/.../checkpoint_best.deploy.pt OUT=$W/runs/dual_actor_wm6/b_delta/isaac/final2 \
  bash scripts/b_wm6/run_isaac_pair.sh
python3 scripts/b_wm6/summarize_b4.py --root runs/dual_actor_wm6/b_delta/isaac/final2
```

---

## 11. 文件 / 日志索引

```
runs/dual_actor_wm6/b_delta/
  README.md                         产物索引
  report/REPORT.md                  正式报告（增量更新）
  report/REVIEW_FOR_GPT.md          ← 本文
  config/b_wm6_h10_critic.json      唯一正式配置
  config/b_lock.json               版本锁 + 物理合同 + GPU UUID
  c0/c0_report.json                 B0 27 项
  c0/action_domain.json            动作域反解真值（dq_ha/sc_ha）
  c1/seed0/                         B1（token critic, tail 自由）
  c2b/seed0/                        B3 正式（token critic, mask10, β 被否决）
  c3_v5/seed0/                      β=1 + v5 critic 续训（正在跑）
  critic_warm_b1.pt                 TokenCritic 预训练（v_ev 0.067）
  critic_v5_b3.pt                   ImaginedValueV5 预训练（v_ev 0.626）
  isaac/job_*.pt roll_*.pt pair_*.json   真物理 open-loop（含 A/A）
  isaac/final2/tasks_base/          B4 Base 臂 18 集
  isaac/final2/tasks_b/             B4 B 臂（只 4 集，被叫停）
  logs/                             全部原始日志（含被修掉的失败尝试）
```

关键日志：`logs/c2b_seed0.log`、`logs/c3_v5_seed0.log`、`logs/critic_v5_probe.log`、
`logs/critic_warm_b1*.log`、`logs/b4_final2.log`、`logs/b4_delta_arm.log`、
`logs/server_d_18242.log`（含 `[server][delta-v5] #N ... tail_zero=True`）、`logs/smoke_v5*.log`。

---

## 12. 一句话总结给 GPT

> 工程链和 actor 侧都成立了（held-out test ΔJ +0.00785、6/6 任务为正、动作 6.7 mrad）；
> **critic 换成 v5 imagined 后也成立了（v_ev +0.626）**；
> **唯一没成立的是"β=1 的价值梯度"——它把 ΔJ 从 +0.00914 打到 +0.00235，同时把动作幅度放大 2.5 倍。**
> 请重点裁决 Q1（β 上限与门控形式）和 Q6（闭环样本量是否够下结论），
> 其余 Q2–Q5 是次级优化。
