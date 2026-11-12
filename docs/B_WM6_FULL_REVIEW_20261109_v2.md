# B-WM6 全线评审材料 v2（给 GPT 修改用）

> 生成：2026-11-09 12:25 CST｜容器 2（B 线）｜分支 `experiment/b-wm6-h10-critic`
> worktree：`/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-b-wm6`
> 本文**取代** `B_WM6_REVIEW_FOR_GPT_20261109.md`（那版写于 P0–P9 之前）。
> 所有数字都可用 §11 的命令复跑；没有估计值。

---

## 0. TL;DR（十行看完）

1. **架构合规**：Frozen FT8000/ SONIC / WM6 + `TemporalDeltaActorV5(1,744,832)` + H10 + K20 + Δ 只在 t<10。B0 门禁 **27/27 PASS**。
2. **Actor 侧成立**：held-out test ΔJ **+0.00785**（6/6 任务为正），真 rad 动作修正 RMS **6.7 mrad**。
3. **Critic 换对之后成立**：`TokenCritic` 在独立 clip 上 `v_ev` 最好 **+0.067**（真物理 −6.06）；换成 `ImaginedValueV5`（多吃 hist10+plan4+s−ref）后 **+0.626**（corr 0.804）。
4. **β=1 曾经伤人**：旧实现按**全局 update** 算 warmup，换 critic 后 β **一步 0→1**；`c3_v5` 的 test ΔJ 从 +0.00785 掉到 **+0.00588**，动作幅度 6.7→11.6 mrad。
5. **β 调度已修**：按 `critic_activation_update` 起算 + 每步限速 0.05 + 滞回门禁（0.35/0.28）；单测 **6/6**，resume 后 β 曲线 **max diff = 0.0**。
6. **加了梯度预算**：价值梯度按范数封顶 ≤ **20%**×‖g_J10‖（实测只用到 3–4%）。
7. **修复版对照**：`expB`（β=**0.975**，v_ev 0.57）test ΔJ **+0.00788**、动作 **6.56 mrad**，与 `expA`（β=0）的 **+0.00767 / 6.65 mrad** 打平 ⇒ **非零价值梯度不再伤 Actor**。
8. **真物理**：跨进程 A/A **恰为 0.0**；ΔJ10 **+0.00049 [+0.00011, +0.00106]**、胜率 **79.2%**、ΔJ20 **+0.00056** ⇒ 小而**统计显著**的正收益（0.09%）。
9. **但 WM↔Isaac 幅度差 ~10×**（想象 +0.95% vs 真物理 +0.088%），critic 绝对值不可迁移（真物理 `v_ev=−4.53`、corr 仅 0.555、v_mean **−0.156** vs 真值 **+0.545**）。
10. **六任务闭环 = INCONCLUSIVE**（按指令跳过）：Base 8/18、B 只 4 集且用的是旧 ckpt。**对外不能宣布 B 线成功。**

**总判定：`PASS_WITH_CAVEATS`（机制层 PASS；任务层 INCONCLUSIVE）。**

---

## 1. 给 GPT 的问题清单（更新版）

| # | 问题 | 现状/证据 | 我的建议 |
|---|---|---|---|
| **Q1** | **critic 真物理绝对值不可迁移**（corr 0.555 但 v_mean −0.156 vs target +0.545） | §7.3 | 把 `β·V` 改成**相对形式** `V(s_H,ref_H) − V(ref_H,ref_H)`；或 per-task 仿射标定后再进 loss |
| **Q2** | **WM6 把 Δ 的收益放大约 10×**（0.95% vs 0.088%） | §7.2 | 不再用 WM6 ΔJ 当验收；是否对 Δ 动作做**奖励尺度校准**（现有 `p54_rank_calib` 有 `c=0.27~0.57` 的收缩系数可复用）？ |
| **Q3** | β 上限该给多少 | `expB` β=0.975 与 β=0 打平；`c3_v5` β 突跳时掉 0.002 | 保持 1.0 + 预算，或做 β∈{0.3,0.5,1.0} 消融 |
| **Q4** | critic 在线漂移（0.62→0.57 over 40 步；c3_v5 里 0.62→0.25） | §6.4 | 启用已有的 `--critic-catchup-every`（actor 冻结追赶）；建议 20 步一次、每次 100 步 |
| **Q5** | 六任务闭环样本量 | Base 8/18；每任务 3 seed | 判 PASS 至少 6 seeds（36 集/臂，≈3 h） |
| **Q6** | `rot6` 训练/部署定义不同（cos 0.9968，差 6.9%） | §9 | 不是主因，不置零；要彻底统一需改 C 侧 client（跨容器） |
| **Q7** | 是否该采新数据刷 WM2 | WM 在 Δ 动作上系统性乐观 | **暂不**：按指令须先排除合同/仿真错误（P5/P9 已排除），这属于 WM 的 OOD 乐观，需新策略数据才判 |
| **Q8** | 版本号命名 | 我这轮是**从零重训的 B-WM6 正式线**，未继承 V8 recovery | 请指定对外叫法（B-WM6 / v7 / 其他），我按你的口径改标题与判定表 |

---

## 2. 资源与隔离

| | 本容器（B） | C 容器 |
|---|---|---|
| GPU0 UUID | `GPU-618f618a-b436-477c-83cd-037d1e9fe845` | `GPU-272c06bc-4338-1cec-7fa0-298692b90d75` |
| GPU1 UUID | `GPU-0e64fd48-2e1d-74ba-7200-671951b1e1e7` | `GPU-12b4dd7c-c3b7-67f6-815c-00e274c95d83` |

* **按 UUID 判断隔离**（不看 `cuda:0`），无重叠。
* 训练卡位（实测）：**GPU0 = 冻结 FT8000 在线 token 生成**（峰值 **27.3 GB**）；**GPU1 = Δ+C + SONIC + WM6 BPTT**（峰值 **~2.4 GB**）。
* VLA 是瓶颈：**12.7–14.0 s/update（B=256, H=10）**。加梯度预算（两次 backward）后 **~13.2 s**。
* 128 核 / 1 TB；Isaac 单实例 ~15 GB 显存 / ~7 GB RSS，**同卡最多 2 实例**。
* 共享控制 `runs/dual_actor_wm6/control/` 是指向 C 容器的软链（只读）。

---

## 3. 版本锁（B0 逐项重算，与 C 一致）

| 对象 | sha256 前缀 |
|---|---|
| WM6 `wm_base6_best_candidate/wm.pt`（s350） | `8edb59cf` |
| FT8000 `checkpoint-8000/config.json` | `7d06d0da` |
| SONIC Release `model_decoder.onnx` | `c7241a12`（`c4ac2e74` 低延迟禁用） |
| WBC `WBC_A24000_wmvel_cmd.pt` | `2390dc29` |
| packed motion `ha_cmd/packed.pt` | `3f444d50` |
| eval_set `windows.pt`（256/96/96） | `e2f8586e` |

记录：`b_delta/config/b_lock.json`（含物理合同、GPU UUID、control 快照）。

---

## 4. 架构与张量合同

### 4.1 两条链

```
部署: Isaac RGB + 指令 + 本体状态 → FT8000(冻结,无LoRA) → Z_base[40,64]
      → ΔV5(1.744M) → SONIC Release(冻结) → Isaac 真物理
      → 每 20 tick 重规划；history 逐 tick 更新；**不含 WM6/Critic**

训练: 同一 Z_base → Z_base+Δ → SONIC(冻结) → WM6(冻结,8.82M)
      → J10 + β·γ^10·surv_10·V(y10) → Δ
```

### 4.2 Actor（`diff_vla_wbc/delta_token_temporal_v5.py`，未改动）

```
Z (B,40,64) | state64 (B,64) | hist10 (B,10,58)
hz   = token_proj(Z) + PE_chunk[:40]                  # 64→128
c_t  = LayerNorm(TransformerEncoder(1 block,d128,4head,FFN256,pre-LN))(hz)
h    = GRU(58→128)(hist10).last
x_t  = [ z_t(64) | c_t(128) | h(128) | state64(64) | PE_pos(t)(16) ]   # d_in=400
u_t  = MLP(400→1024→1024→64, SiLU)                    # 末层 N(0,4e-4), bias=0
Δ_t  = apply_norm_cap(u_t, cap=0.08) · 1[t < delta_k]
```

参数：`8320+5120+132480+256+72192+640+1525824 = 1,744,832`。
初始化：**从零**，末层 `N(0,4e-4)`（**刻意不零初始化**，否则上游首步无梯度）；实测 `|Δ|max=1.85e-3`。

### 4.3 合同（B0 逐位验证）

| 接口 | 定义 | 验证 |
|---|---|---|
| WM `state71` | `[root_pos3|quat4|q_STD29|v3|w3|qd_STD29]`，dt=0.02 | 与锁一致 |
| WM `action29` | **HA(SONIC) 序**，`(q_target_HA − dq_HA)/sc_HA` | 记录数据**逐位复现 max 4.2e-6**（640 tick） |
| Δ `state64` | `[rot6d6|q_STD29|qd_STD29]`（训练 standard，入站 ha_v31 由 spec 转） | B0 #15 |
| Δ `hist10` | `[q − **g1_default**(29) | qd(29)]`，standard 序，最老→最新 | B0 #15 逐位 0；部署转换 #16 逐位 0 |
| SONIC | torch vs `onnx.reference.ReferenceEvaluator`（独立实现） | **4.8e-7** |

⚠️ `hist` 的偏移是 **`g1_default_joint_pos()`**（hip −0.312/knee 0.669/elbow 0.6），
**不是** `bundle.default_joint_pos`（SONIC 口径 −0.2/0.42/0.87）；elbow 差 **0.27 rad**，用错会静默错位。

### 4.4 物理尺度（真 rad 的唯一真相）

`q_target_std = dq_std + sc_std · a_std`，`sc_std = sc_ha[HA_TO_STD]`，均值 **0.378852**。
`bundle.action_scale` 是 **全 1.0**（`action_scale_mode=one`）⇒ **不能当 rad 尺度**（会把量级记错 2.64×）。
真限位来自 `reproduce_scenes/g1/g1_29dof_with_hand.xml` 的 `joint range`（未改动）。

### 4.5 Critic（两代，最终用 v5）

| | `TokenCritic`（弃用） | `ImaginedValueV5`（当前） |
|---|---|---|
| 输入 | `s71+ref71+xi33(+s−ref)` | `s71+ref71+(s−ref)71+xi33+hist10(580)+plan4×64(256)+extra4` |
| in_dim / 参数 | 175 或 246 / 111,105 | **1086 / 819,713**（hidden 512×2） |
| 目标 | TD(λ 0.95) + EMA(0.995) | 同 |
| 独立 clip `v_ev` | **+0.067**（最好） | **+0.6259**（corr 0.804） |
| 真物理 `v_ev` | **−6.06** | **−4.53**（corr **+0.555**） |

---

## 5. 全部参数（锁定配置）

### 5.1 时间轴 / RL

| 项 | 值 |
|---|---|
| T / K_exec / H_train | **40 / 20 / 10** |
| Δ 生效帧 `delta_k` | **10**（`tail_control=mask10`）；B1 曾是 20 |
| dt / control | 0.02 s / 50 Hz；每 20 tick 重规划 |
| γ | 0.99 |
| β_max / warmup / slew / 滞回 | 1.0 / **从 `critic_activation_update` 起算 40–100 步** / **0.05/step** / **on 0.35, off 0.28** |
| 梯度预算 | ‖g_V_scaled‖ ≤ **0.20 × ‖g_J10‖** |
| Critic γ / λ / EMA | 0.99 / 0.95 / 0.995 |
| Critic 更新 | 4 次/actor step，batch 2048，lr 3e-4，wd 1e-4，clip 1.0 |
| Critic 独立验证 | 每 5–25 步，指标 = 独立 clip 上的想象 MC 回报 `v_ev` |

### 5.2 Actor 优化 / 损失

| 项 | 值 |
|---|---|
| 优化器 | AdamW，head 3e-4 / enc 1e-4，min 3e-5 / 1e-5，wd 1e-4，clip 1.0 |
| warmup / decay | 50 步 linear + cosine |
| cap | 0.08（**固定**，`--cap-warmup-updates 0`，B1 曾是 0.04→0.08 over 100） |
| λ_raw / κ | **1e-3 / 2.0**（B1 曾是 0） |
| λ_action / τ | 0.010 / **0.020（归一化动作单位，≈7.6 mrad 物理 RMS）** |
| λ_smooth / λ_anchor | 0.002 / 0.001 |
| λ_limit / λ_peak | **0 / 0**（原 B 无；只监控） |
| batch | 256（seeds 固定、on-the-fly 抽帧） |

### 5.3 采样 / 数据

| 项 | 值 |
|---|---|
| 起点 | `multinomial(w, B)` over **237,880** 合格帧（漂移 ≤0.10 m、f≥10、f+H+1≤clip_end、**仅 train clips**） |
| burn-in | 10 步；扰动 xy 0.05 m / yaw 0.10 rad / dq 0.05 rad / dqd 0.30 rad/s |
| VLA | 冻结 FT8000，`fm_steps=4`，**每次 update 新噪声**，vla_batch 128 |
| eval_set | 256/96/96（train/val/test），训练只允许 train clips |

### 5.4 各 run 的实际参数差异

| run | updates | H/K | delta_k | β | critic | λ_raw | cap warmup | 备注 |
|---|---|---|---|---|---|---|---|---|
| `c1/seed0` | 100 | 10/20 | 20 | 0→1 | token(256×2), scratch | 0 | 100 | B1 短程 |
| `c2b/seed0` | 300 | 10/20 | 10 | 0→1（门控后归 0） | token(256×2), scratch | **1e-3** | **0** | **B3 正式** |
| `c3_v5/seed0` | 300→420 | 10/20 | 10 | **1（突跳）** | **v5**(512×2) warm | 1e-3 | 0 | **失败样本** |
| `expA_beta0` | 300→340 | 10/20 | 10 | **0** | 关 | 1e-3 | 0 | P6 对照 |
| `expB_fixed` | 300→340 | 10/20 | 10 | **0→0.975** | v5 warm + 预算 | 1e-3 | 0 | **P6 修复版（候选）** |

---

## 6. 代码改动清单

| 文件 | 作用 |
|---|---|
| `scripts/b_env.sh` | B 线环境（`$WT` 顶到 PYTHONPATH 最前） |
| `scripts/b_wm6/contracts.py` | 版本锁 + G1 物理合同（MJCF 真限位、真 rad 尺度） |
| `scripts/train_b_wm6_h10_critic.py` | **正式训练器**（`--critic-kind {token,v5}`、严格 resume、β 控制器、梯度预算、critic 追赶） |
| `scripts/b_wm6/beta_schedule.py` | **P0** β 纯状态机（激活起算+warmup+滞回+限速），可单测 |
| `scripts/b_wm6/test_beta_schedule.py` | P0 单测 6 项 |
| `scripts/b_wm6/baseline_audit.py` | P0 只读基线审计（79 个 ckpt） |
| `scripts/b_wm6/audit_critic_grad.py` | P1 三通路梯度归因 |
| `scripts/b_wm6/audit_target_phys_rot6.py` | P2 + P5 + P9 三合一审计 |
| `scripts/b_wm6/critic_v5_probe.py` | v5 critic 先验（MC 目标 + 独立 clip dev） |
| `scripts/b_wm6/pretrain_critic.py` | TokenCritic 离线预训（对照） |
| `scripts/b_wm6/b0_tests.py` | B0 门禁 27 项 |
| `scripts/b_wm6/build_replay_job.py` / `score_replay.py` / `score_pair.py` | 真 Isaac open-loop 回放 + 配对 + A/A |
| `scripts/b_wm6/run_isaac_roll.sh` | 回放 job（输出出现即杀，绕 `app.close()` 挂死） |
| `scripts/b_wm6/run_isaac_pair.sh` | Base vs B 真闭环驱动 |
| `scripts/b_wm6/summarize_b4.py` | 六任务汇总（同 (arm,task,seed) 取最新） |
| `scripts/b_wm6/export_deploy_ckpt.py` | 训练态 → 部署 v5 ckpt |
| `scripts/b_wm6/analyze_run.py` | 逐任务 ΔJ / Δ 分段 / rot6 敏感度 |

---

## 7. 排查记录：**9 个已定位的 bug**（这是本文最重要的部分）

### 7.1 v5 训练器遗留的 5 个真 bug（都会让"H10+Critic"名不副实）

| # | 原写法 | 后果 | 修法 |
|---|---|---|---|
| 1 | `states[H].detach()` 进 critic | `β·γ^H·V(s_H)` 对 Δ **梯度恒 0**（"开了 critic 只当常数"） | 不 detach；`FrozenParamCtx` 只关参数 `requires_grad` |
| 2 | `refs_f[:, H]` | `refs_f` 只有 H 列 ⇒ `IndexError`；且 critic 要的是 **71 维参考状态**不是帧号 | `teacher_references(...,H+1)[H]` + `ref_state_at()` |
| 3 | `adapter.as_dict()`（token 分支 `adapter=None`） | `it%50==0` 时写 `critic_gate.json` 崩溃 | token 分支不再引用 adapter |
| 4 | `beta` 只在 `it%50==0` 更新 | 前 49 步恒 0、之后 50 步一台阶 | 逐 update（后升级为 BetaController） |
| 5 | checkpoint 不含 critic/target/优化器/RNG/β | **无法 resume** | 全量保存 + locks sha 对账；B0 #25 验 **bitwise** |

### 7.2 本轮新发现的 4 个 bug

| # | bug | 证据 | 影响 | 修法 |
|---|---|---|---|---|
| 6 | **β warmup 用全局 update 算** | 从 c2b it=300 续训 + 换 critic 时 `min(1,301/100)=1` ⇒ β **一步 0→1**（日志 it310 β=0 → it320 β=1） | `c3_v5` test ΔJ 掉到 +0.00588、动作 6.7→11.6 mrad | `BetaController`：激活起算 + slew 0.05 + 滞回（§8） |
| 7 | **resume 时 `beta/gate_status` 被 checkpoint 无条件覆盖** | 覆盖发生在"架构不一致重置"**之后**，把重置冲掉 | 同上 | 调整顺序，改由控制器持有 |
| 8 | **真物理 open-loop 两臂动作逐位相同** | `build_replay_job.actions_of()` 忘把 `tokens` 传进 `masked_return` | 第一版报告的 "ΔJ10=+0.0014 显著" 是**假信号** | 传参 + 加 `--dup` A/A 模式 |
| 9 | **同进程顺序跑两臂互相污染** | 同动作时末状态仍差 **6.84**（比真实 Δ 大 3 个数量级）；ΔJ10 偏置 **+0.00143** | 之前判 INCONCLUSIVE 的直接原因 | **每臂独立进程**；实测跨进程 A/A **恰为 0.0** |

### 7.3 3 个环境 bug（真阻塞）

| 症状 | 根因 | 修法 |
|---|---|---|
| Isaac 卡死在 `Starting the simulation`，永远到不了第一次 VLA 请求 | 容器里 `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` 指向**不存在的文件** ⇒ `vkCreateInstance` 报 `ERROR_INCOMPATIBLE_DRIVER` ⇒ `carb::graphics::createInstance failed` / `no suitable CUDA GPU` ⇒ PhysX 退软件并卡死 | `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`（已写进 `run_isaac_pair.sh`） |
| `[SonicActionProvider] onnxruntime is not available` | B 线 `PYTHONPATH` 含 `third_party/gr00t_deps`（**坏的** onnxruntime，缺 `onnxruntime_pybind11_state`），盖住 `isaacsim_env` 的好副本 | 跑 Isaac 时 `env -u PYTHONPATH -u VIRTUAL_ENV` |
| 回合跑完找不到 `result.json` | `ha_isaac_closedloop.sh` 会 `cd $PROJECT_ROOT` 再 exec，**相对** `OUT_ROOT` 被重解析到 HumanoidArena 下 | `OUT` 一律转绝对路径 |

另：`run_isaac_pair.sh` 的 **`wait` 死锁**（无参 `wait` 会等同 shell 后台起的 VLA server，
而 server 要等 `stop_server`，`stop_server` 在 `wait` 之后）⇒ 只 `wait` episode 的 PID。

### 7.4 一处"差点被误判为 bug"的口径混淆

`spec.K` = **Δ 生效帧数**（`mask10` 下 =10），而 critic 的 plan 窗口用的是
**实际执行的 token 数 `exec_k` = 20**。用 `spec.K` 去 build critic 特征会把
`t=H=10` 的 plan 全补 0（我第一版 P1 审计脚本就踩了这个坑，已修并写入文档）。
**训练器用的是 `exec_k=20`，是对的。**

---

## 8. P0：β 调度修复（核心）

```
β_sched = β_max · clamp((it − critic_activation_update) / W, 0, 1)
gate    : v_ev ≥ 0.35 ⇒ ok；v_ev < 0.28 ⇒ 不 ok；中间**保持**（滞回）
β_eff   : 每 update 最多变 0.05（限速）
```

* `critic_activation_update` 在以下四种情况重置为当前 it，**不动 Actor 全局 step**：
  新建 critic / `--critic-init ckpt` / `--critic-warm-from` / resume 时架构不一致。
* `{activation, gate_ok, β_eff, β_sched}` 写入 checkpoint，`--resume` 后逐位复现。

**单测 6/6（`recovery_h10_critic/audit/audit_beta_resume.json`）**：

| # | 检查 | 结果 |
|---|---|---|
| 01 | it=300 换新 critic ⇒ β 起始 ≈0 | **0.0**（旧实现同点 = 1.0） |
| 02 | 之后 10 步不跳到 1 | 0, 0.01, …, 0.10（单调、每步 +0.01） |
| 03 | warmup 中断再 resume 曲线逐位复现 | **max diff = 0.0**（51 步） |
| 04 | 门禁抖动无突跳 | 每步 ≤ slew；0.30 在死区时保持 PASS |
| 05 | 不隐式改 LR/Scheduler | 控制器源码不含 `.lr`/`optimizer`/`scheduler` |
| 06 | 旧实现门禁一翻就满功率 | legacy@301 = **1.0** vs 新实现首步 **0.0** |

---

## 9. 审计结论（P1/P2/P5/P9）

### P1 梯度通路（`critic_grad/audit_critic_gradient_paths.json`）

| 通路 | ‖g‖ | 占全通路 |
|---|---|---|
| A：只 `s10` 回传 | **0.7462** | 1.000 |
| B：只 plan 特征回传 | **0.0000** | 0.000 |
| C：都开 | 0.7462 | 1.000 |

`cos(A,B)=0`、`cos(A,C)=1.000` ⇒ **唯一通路 `s10`，无 plan 捷径**。`z = tokens + Δ.detach()` 生效。

### P2 价值目标（`target_audit/audit_critic_target.json`）

- TD(λ) 手算 vs 实现：**max diff 0.0**
- 末步自举 = `r_{H−1} + γ V̄_H` ✅
- `J10` 重算：**0.0**（γ 折扣 + survival 一致）
- MC 目标：`MC[t]=Σ_{k≥t}γ^{k−t}r_k`，从 t 起算 ✅
- ⇒ `J_AC = J10 + βγ^H surv V(y_H)` **无重复计数**；J10 只覆盖 t<H，V 只覆盖 t≥H

### P5 物理动作（`phys/audit_physical_action.json`）

| 项 | 值 |
|---|---|
| base decoder 自带越限（joint-step 占比） | **7.58%** |
| Δ 臂总越限 | 7.63% |
| **Δ 新增**越限 | **0.05%** |
| Base↔Δ 关节目标差（真 rad） | RMS **16.2 mrad** |
| 越限来源 | **SONIC decoder 原始输出**，部署侧 `sonic_physical_action` 裁剪 |
| WM6 输入 | **仍是原始归一化 command**（不改，符合 WM6 训练合同） |

### P9 rot6（`rot6/audit_rot6.json`）

| 对比 | cos(Δ) | 相对变化 |
|---|---|---|
| 训练（相对 clip 首帧参考） vs 部署（相对回合首帧） | **0.9968** | 6.9% |
| 训练 vs rot6 置零 | 0.9977 | 6.0% |

⇒ 非主因，**不置零**。

---

## 10. 全部实验结果

### 10.1 B0 门禁：27/27 PASS

`04_action_domain` 4.17e-6 / `09_sonic_onnx` 4.8e-7 / `10_init` 1,744,832 / `11_tail_zero` 恰 0 /
`13_cap` ≤cap / `15/16_hist` 逐位一致 / `17/19/20` 三路梯度非零 / `21/22/23` TDλ+EMA+优化器隔离 /
`25_resume` bitwise 0 / `27_full_step` rc=0。

### 10.2 Actor 各版本（held-out test，独立 clip 96 窗）

| run | β | test ΔJ | same-obs RMS | peak | 备注 |
|---|---|---|---|---|---|
| `c1`（token, tail 自由, v1） | 1.0 | **+0.00376** | 13.98 mrad | 74.0 | 101 步 |
| `c2b`（token, mask10, v2） | 0（门控） | **+0.00785** | **6.71 mrad** | 43.5 | **300 步，actor 最优** |
| `c3_v5`（v5, β 突跳） | 1→0 | +0.00588 | 11.58 | — | 失败样本 |
| `expA`（β=0 续训） | 0 | +0.00767 | 6.65 | — | P6 对照 |
| `expB`（v5, 修复） | **0.975** | **+0.00788** | **6.56** | — | **P6 修复候选** |

逐任务 val ΔJ（`c2b`）：sit_sofa **+0.01314** / pp_box +0.00991 / vision_navi +0.00988 /
open_door +0.00877 / boxing +0.00847 / double_desk +0.00697 —— **6/6 全正**。

### 10.3 v5 critic 先验（同一批策略匹配数据，独立 clip）

| critic | dev `v_ev` | corr | mae（const 0.229） |
|---|---|---|---|
| TokenCritic(use_diff=0) | **+0.067** | 0.27 | 0.227（≈基线） |
| TokenCritic(use_diff=1) | −0.20（train loss 0.0002，过拟合） | — | — |
| **ImaginedValueV5** | **+0.6259** | **0.804** | **0.131（比基线好 43%）** |

### 10.4 expB（修复版）逐步轨迹

```
it=310 β=0.05  v_ev=0.6206  val ΔJ=+0.00926  sameRMS=6.76
it=320 β=0.47  v_ev=0.6093  val ΔJ=+0.00894  sameRMS=6.75
it=330 β=0.72  v_ev=0.5925  val ΔJ=+0.00898  sameRMS=6.69
it=340 β=0.97  v_ev=0.5689  val ΔJ=+0.00909  sameRMS=6.70
```

梯度预算实测：`‖g_J10‖=0.19–0.30`、`‖g_V‖=0.005–0.013`、ratio **2.9%–4.2%**、cos **−0.12 / +0.08**（近正交）。

### 10.5 P7 真物理（48 窗 × H=20，每臂独立进程）

| 指标 | Base | B | Δ | 95% CI | 胜率 |
|---|---|---|---|---|---|
| **真 J10** | 0.55514 | 0.55563 | **+0.00049** | **[+0.00011, +0.00106]** | **79.2%** |
| **真 J20** | — | — | **+0.00056** | **[+0.00011, +0.00119]** | — |
| **A/A（跨进程，同动作）** | — | — | **0.0** | — | — |
| A/A（同进程顺序两臂，旧做法） | — | — | +0.00143 | — | — |

**关键修正**：Isaac **跨进程完全确定性**（同动作末状态差 0.0），所以旧报告的
"+0.00143 噪声 > 信号"是 harness 污染，不是物理噪声。**真实 A/A = 0**。

### 10.6 六任务真实闭环（未完成）

| 任务 | Base（3 seeds） | B（旧 ckpt，被叫停） |
|---|---|---|
| PPBox | 1/3 | 0/3 |
| DoubleDesk | 0/3 | 0/1 |
| Boxing | 2/3 | — |
| OpenDoor | 2/3 | — |
| SitSofa | 2/3 | — |
| VisionNavi | 1/3 | — |
| **合计** | **8/18 = 44%** | **0/4** |

部署链已确认真生效：`[server][delta-v5] #121 |Δ|mean(t<K)=0.0542 tail_zero=True`。

---

## 11. 复现命令

```bash
WT=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-b-wm6
cd $WT && source scripts/b_env.sh

python3 scripts/b_wm6/b0_tests.py --device cuda:1 --vla-device cuda:0 --batch 32   # B0 27 项
python3 scripts/b_wm6/test_beta_schedule.py                                        # P0 6 项
python3 scripts/b_wm6/baseline_audit.py                                            # P0 基线
python3 scripts/b_wm6/audit_critic_grad.py --actor-ckpt <ckpt> --critic-ckpt <critic> --n 24
python3 scripts/b_wm6/audit_target_phys_rot6.py --actor-ckpt <ckpt>

# P6 B（修复版）
python3 scripts/train_b_wm6_h10_critic.py \
  --out runs/dual_actor_wm6/b_delta/recovery_h10_critic/expB_fixed \
  --resume runs/dual_actor_wm6/b_delta/c2b/seed0/checkpoint_last.pt --updates 340 \
  --device cuda:1 --vla-device cuda:0 --batch 256 --horizon 10 --exec-k 20 \
  --tail-control mask10 --lambda-raw 1e-3 --raw-norm-kappa 2 --cap-warmup-updates 0 \
  --beta 1.0 --beta-stage-warmup 40 --beta-slew 0.05 --beta-gate-on 0.35 --beta-gate-off 0.28 \
  --beta-grad-budget 0.20 --critic-enabled 1 --critic-kind v5 --critic-init ckpt \
  --critic-ckpt runs/dual_actor_wm6/b_delta/critic_v5_b3.pt \
  --critic-v5-hidden 512 --critic-v5-depth 2 --critic-updates 4 --critic-batch 2048 \
  --critic-validate-every 5 --on-the-fly 1 --vla-batch 128 --seed 0 \
  --eval-set runs/temporal_ac_bptt_v5/eval_set --eval-every 10 --log-every 10

# 真物理配对（每臂独立进程！）
bash scripts/b_wm6/run_isaac_roll.sh job_base.pt roll_base.pt 1
bash scripts/b_wm6/run_isaac_roll.sh job_b.pt    roll_b.pt    1
python3 scripts/b_wm6/score_pair.py --job-a job_base.pt --roll-a roll_base.pt \
  --job-b job_b.pt --roll-b roll_b.pt --b-ckpt <ckpt>.pt --out pair.json
```

---

## 12. 产物索引与提交

```
runs/dual_actor_wm6/b_delta/
├── BASELINE_AUDIT.md / report/{REPORT.md, REVIEW_FOR_GPT.md}
├── c0/  c1/  c2b/  c3_v5/                       历史 run（只读）
├── critic_warm_b1.pt  critic_v5_b3.pt           critic 预训权重
├── recovery_h10_critic/
│   ├── BASELINE_AUDIT.md                        79 ckpt 全表
│   ├── audit/{audit_beta_resume,baseline_audit}.json
│   ├── critic_grad/  target_audit/  phys/  rot6/   四份审计
│   ├── expA_beta0/  expB_fixed/                 P6 对照 / 修复候选
│   └── final_report/REPORT.md                   P0–P9 报告
├── isaac/{job_*,roll_*,pair_*.json}             真物理 open-loop
└── isaac/final2/{tasks_base,tasks_b}/           六任务闭环（部分）
```

git：`d367454` → `c62f1fd` → `b033b3a`（P0–P9 全部代码与审计）。
主仓 `docs/` 同步：`B_WM6_REVIEW_FOR_GPT_20261109.md`、`B_WM6_RECOVERY_REPORT_20261109.md`、
`B_WM6_FULL_REVIEW_20261109_v2.md`（本文）。

---

## 13. 一句话给 GPT

> **工程与目标全部对齐、H10+Critic 的"非零 β 且不伤 Actor"已经做到（expB：β=0.975、test ΔJ +0.00788、动作 6.56 mrad），
> 真物理也有小而干净的正收益（ΔJ10 +0.00049、CI 不含 0、A/A=0、胜率 79%）。**
> **唯一没解决的是"价值/收益的尺度"：WM6 把收益放大约 10×，critic 的真物理绝对值符号都反了（corr 0.555 但 v_mean −0.156 vs +0.545）。**
> 请裁决 **Q1（β·V 改相对形式）** 和 **Q5（闭环要不要补到 6 seeds）**，其余 Q2–Q4、Q6–Q8 是次级。
