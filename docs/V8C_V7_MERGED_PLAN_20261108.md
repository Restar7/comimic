# COMIMIC 合并架构计划：V8-C 训练 + V7 周期 WM Refresh

> 写于 2026-11-08 15:00。**这是"完整架构 + 当前代码修改清单 + 重训计划"的唯一口径文档。**
> 规则：V8-C 任务书**除 §5（WM refresh）外全部采用**；§5 替换为 **V7 的双时间尺度周期刷新**。
> 状态标记：`PASS / FAIL / BLOCKED / NOT_RUN`；未跑过的不写成跑过。

---

## 0. 一句话

冻结 FT8000 + 218 层 LoRA16（含 proj_out LoRA）的 C 线，在 **V8-C 的损失**（原 J20 + 长尾 Recovery + 绝对/额外限位 + 弱 Margin + 固定 peak 预算 + 总 Trust）下训练 400 updates；
训练切成 **100 步一个 segment**，每个 segment 内 **WM 冻结**；每到边界做一次 V7 的诊断/按需刷新/门禁/发布，**换 WM 只发生在 segment 边界**，且每个 segment 的 WM 版本与 sha 都写进日志。最终做 **Fixed-WM vs Refreshed-WM 对照** + 独立 Test，证明"刷新真的帮到 Actor"。

---

## 1. 完整架构

```
                       ┌──────────────────────────────────────────────┐
                       │  Frozen FT8000 VLA (1.637B, fp32)             │
                       │   backbone(冻结) + AlternateVLDiT(冻结)      │
                       └───────────────┬──────────────────────────────┘
                                       │  + LoRA16 on 216 attn1/attn2/ff 层
                                       │  + LoRA16 on proj_out_1/2（原全秩权重冻结）
                                       │  可训 = 16,777,216 + 114,688 = 16,891,904
                                       ▼
                             4-step Euler Flow Matching（可微）
                                       ▼
                                token [B,40,64]
                                       ▼
                            Frozen SONIC（可微执行器）
                                       ▼
                          q_raw = q_default + scale ⊙ a
                                       ▼
                     ┌──────────── Frozen WM（版本化）─────────────┐
                     │  wm_v000(fc5aeb30) 或 wm_c1_v001(81d24caf)  │
                     └───────────────┬──────────────────────────────┘
                                     ▼
                        H=20 想象 rollout → J20(原指数核)
                                     │
        ┌────────────────────────────┼─────────────────────────────┐
        ▼                            ▼                             ▼
   L_recovery(长尾)           L_lim + L_added + L_margin      L_peak + L_same
   （ReLU(z−z0) Huber）       （绝对/额外/软限位）            （固定 b_j / 总 Trust τ=0.03）
        └────────────────────────────┴─────────────────────────────┘
                                     ▼
                        Backprop **只**到 LoRA A/B
```

### 1.1 双时间尺度（V7 §2 的机制，接在 V8-C 训练里）

```
segment 0 : 100 Actor updates（WM = WM_k 冻结，hash 记录）
   │
   ├─ 保存快照：actor + optimizer + scheduler + RNG + WM 版本
   ├─ 便宜诊断（每 100 步都做，几分钟）：
   │    token 分布 / 物理动作 / q_raw 限位 / E1·E5·E10·E20（当前 on-policy 数据）
   ├─ 触发判定（V7 §12）：
   │    E20 明显恶化 或 状态-动作滑出 A′ 支持域 → 需要刷新
   │    否则 → 进入下一个 100 步（不刷新）
   ▼ 需要刷新
真实 Isaac 短轨迹采集（**用当前 C 快照**，SONIC 执行，20 步/起点，~128 起点）
   │  → onpolicy/C/cycleNNN（累加，不覆盖）
   ▼
Mixed Replay：70% A′（529,131 行）+ 30% on-policy（累积）
   ▼
候选 WM：copy(WM_k) → 150~300 updates（1e-5，多步 1/5/10/20）
   ▼
门禁 6 条（episode 聚类 CI）：新数据 E20 改善 / 旧 A′ 不退化 / dir_agree 不差 /
   │                          finite / 动作合同 / 无泄漏
   ├─ PASS → publish WM_{k+1}（原子、可回滚）→ 下一 segment 加载新版本
   └─ FAIL → REJECTED → 继续用 WM_k（不算失败）
   ▼
segment 1 : 100 Actor updates（WM = 新版本，hash 记录）
```

**硬约束（V7 §2/§12）**
1. 单次 BPTT 内**绝不**换 WM（只在 segment 边界换）；
2. WM 更新时 Actor 冻结，Actor 更新时 WM 冻结，但梯度可穿过 WM/SONIC 输入；
3. 每个 run 记录自己用的 WM 版本 + sha；不同 segment 的 ΔJ **不拼成一条曲线**；
4. 不覆盖 `wm_gain2.pt`，一律走版本目录。

---

## 2. 锁定参数（V8-C §1/§6.2）

| 项 | 值 |
|---|---|
| 底座 | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（fp32，全冻结） |
| Adapter | 218 层 LoRA16（216 attn1/attn2/ff + 2 proj_out），原 proj_out 全秩**冻结** |
| rank / alpha / scale / dropout | 16 / 32 / 2.0 / 0 |
| 可训参数 | **16,891,904**（主干 16,777,216 + proj 114,688，占 1.032%） |
| Token T / 执行 K / WM H / FM | 40 / 20 / 20 / 4 步 |
| 控制 dt | 0.02 s |
| Batch | 32（单卡）；有效 batch = 32 |
| Updates | **400**（4 × 100 segment） |
| LR | 主干 3e-5 / proj_out 3e-5（两 group 独立）；warmup 20；cosine → 5e-6；grad clip 1.0 |
| AdamW | β=(0.9, 0.95)，wd=1e-4 |
| 采样 | on-the-fly（Train clips only，`n_eligible=235,280`） |
| WM | 每 segment 冻结一个版本（初值 `wm_c1_v001_published` / `81d24caf`） |
| 物理算符 | `q_raw = q_default + G1_ACTION_SCALE ⊙ a`（standard 序，唯一入口） |

---

## 3. 损失（V8-C §2–§3.6，已实现）

```
L = −J20(原指数核, 不变)
    + λ_R      · L_recovery        # Σ_t γ^t Σ_k w_k φ(relu(z_k − z0_k)), φ=Huber(δ=1)
    + λ_lim    · L_lim(q_raw)      # 绝对越限（不对称限位）
    + λ_added  · L_added           # E[relu(v(q_full) − v(q_pristine))²]，base detach
    + λ_margin · L_margin          # relu(κ − d)², κ=0.05（弱软限位）
    + λ_peak   · L_peak(e_total,b_j)   # 固定逐关节预算 b_j（Train-only）
    + λ_trust  · L_same(e_total,τ)     # τ=0.03 rad，**总修正**信任域
```

* **没有边际 Trust**（`λ_trust_lora = λ_peak_lora = 0`）⇒ 训练时**跳过 proj-only 参考支路**（省 1/3 采样；实测 8.0 s/step）；
* `e_total = q_full − q_pristine`（三路**同噪声**采样，pristine = 全 LoRA 关闭，逐位等于 FT8000）；
* **z0 必须由数据定**：实测 `z0=2` 时 `frac(z>2)=0.000`（H=20 想象 rollout 与真实窗口都一样），所以用 **Train Q90 冻结**：
  `root_pos 0.2059 / root_ori 0.5521 / body_linvel 0.5982 / body_angvel 0.7205`；
* **b_j 由扰动响应定**：δZ=0.05·std 经正式 SONIC 测逐关节 `|Δq|`，Q90 + floor 0.02 → mean 0.0225、range [0.020, 0.0376]、**11 个不同值**；
* **λ 由梯度预算反解**（48 Train 窗口，旧 C1 adapter 作参照，rms_total=12.2 mrad）：
  `λ_lim 0.03068 / λ_added 6.78179 / λ_margin 0.2201 / λ_recovery 1.913e-3 / λ_peak 1.8e-5 / λ_trust 0.01（fallback）`
  —— `λ_trust` 在校准点 INACTIVE（rms<τ），按 V8-C §3.6 记 INACTIVE，但**安全项回退旧值**，绝不静默关掉。

---

## 4. V7 WM Refresh 子系统（替换 V8-C §5）

### 4.1 组件（都已实现并通过测试）

| 组件 | 文件 | 测试 |
|---|---|---|
| Mixed Replay（70/30、90/10、episode 防泄漏、动作域、k 步窗口、manifest） | `diff_vla_wbc/wm_replay_buffer.py` | 14 |
| 版本存储（原子 publish / reject / rollback / resolve / sha 断言） | `diff_vla_wbc/wm_version_store.py` | 14 |
| 门禁（6 条 + episode 聚类 bootstrap CI + override 记录） | `diff_vla_wbc/wm_refresh_gate.py` | 12 |
| 候选 WM 训练（single/kstep/both，K=20） | `scripts/train_wm_refresh_v7.py` | 真跑 150 步 |
| 门禁执行（`--publish/--allow-prediction-only`） | `scripts/eval_wm_refresh_v7.py` | 真跑 PASS |
| 真 Isaac 逐 tick 采集（注入式，不改 vendored 栈） | `scripts/v7_tick_record/sitecustomize.py` + `scripts/collect_wm_refresh_isaac.py` | 3 |
| WM0 偏移诊断（E1/5/10/20、root/heading、支撑度） | `scripts/diagnose_wm_shift_v7.py` | 真跑 |
| 真物理转移 → segment | `scripts/make_segments_from_replay_v7.py` | 真跑 |

### 4.2 数据合同（踩过的坑，已固化）

1. **动作域**：A′ 的 `actions` = `(cmd_q − dq_ckpt)/sc_ckpt`（`build_real_transitions.py:58-106`），不是 decoder 归一化输出、不是 43 维 env action；实测新数据 `q50=2.918` vs A′ `2.93`；
2. `state71` 布局 `[p(3)|quat(4)|q_sonic(29)|v(3)|w(3)|qd_sonic(29)]` 与 A′ 逐位一致；
3. 多 seed 采集按 `(run_id, episode_id)` 分组（否则两个 seed 合成一个 episode）；
4. 动作域检查按**整体**比例（0.5% 长尾），不按 segment（20 行里 4 行越界会误判成"域错"）；
5. 采集必须：绝对路径（Isaac cwd=`$MAIN`）、不继承父进程 `PYTHONPATH`（否则 `onnxruntime` 半残）。

### 4.3 版本与发布（现状）

```
runs/three_track_v7/wm_versions/
  wm_v000/                  # WM0 副本，sha fc5aeb30e987f016
  wm_c1_v001_candidate/     # 150 步候选，sha 81d24cafbd4b0953
  wm_c1_v001_published/     # ← 当前 active
  wm_probe_candidate/ wm_probe_gate/   # 早期 3 步 smoke
```

`wm_c1_v001` 门禁结果：旧 A′ root 4.50→4.48 cm（rel +0.45%，≤5% 门槛）；新 on-policy root **31.80→10.97 cm（rel +65.5%，CI [+0.676,+0.762]，4 episodes）**；`dir_agree` **未跑 ⇒ override**（发布记录写明"不代表梯度方向已验证"）。

---

## 5. 当前代码审计（已有 / 缺失）

### 5.1 已有（PASS）

| 能力 | 位置 |
|---|---|
| LoRA16 注入 / 218 层 / 参数合同 / zero-LoRA 恒等 | `diff_vla_wbc/lora_ft8000_loader.py`（`C_RECIPES`、`trainable_by_group`、`proj_out_base_frozen`） |
| 4 步 FM 可微 → token | `diff_vla_wbc/vla_grad_sampler.py`、`gr00t_fm.py` |
| WM/SONIC 冻结、梯度穿透 | 由 `test_sonic_wm_grad_to_lora` 守住 |
| 三路同噪声采样 + 总/边际 trust | `train_sonic_lora16_ft8000_bptt.py` |
| 分组 LR + per-group grad/update/token 诊断 | 同上（`grad_norm_by_group`、`update_norm_by_group`、`dtok_lora/proj`） |
| Recovery 长尾 + z0 冻结 | `diff_vla_wbc/tracking_recovery_tail.py`（7 测试） |
| Added Limit / 弱 Margin | `diff_vla_wbc/physical_bptt_losses.py`（`l_added_limit`、`l_margin`，7 测试） |
| C 空间 λ 校准 + peak 预算（扰动响应） | `scripts/calibrate_c_v8_losses.py` |
| 按 clip 聚类 bootstrap CI（val/test） | 训练器 `eval_split` + `wm_refresh_gate.bootstrap_ci` |
| WM 版本 pin（`--wm-version/--strict-wm-hash`，DDP 一致性） | 训练器 + `wm_version_store` |

### 5.2 缺失（本轮要写）

| # | 缺口 | 影响 | 优先级 |
|---|---|---|---|
| G1 | **`scripts/schedule_wm_refresh_v7.py`**：每 100 步诊断 → 触发判定 → 采集 → WM 训练 → 门禁 → 发布，产出 `cycleNNN.json` | 没有它就没有"周期" | P0 |
| G2 | 训练器 **`--wm-refresh-mode gated`** 的真实实现：在 segment 边界**重新加载**已发布 WM、打印/记录 `wm_version+sha`、拒绝在 segment 内换 | 现在只记录参数、行为仍是 fixed | P0 |
| G3 | **C 侧 on-policy 采集**：用 `checkpoint_lora_XXXX.pt` 起带 LoRA 的 server（`ha_vla_latent64_server.py --lora16-ckpt` 已支持）→ Isaac 采 20 步 | 现在 Buffer 里只有 A 的数据 | P0 |
| G4 | **梯度真值探针**：`probe_wm_gradient_truth.py` + `isaac_probe_action_rollout.py` → `dir_agree_old/new` JSON，喂给门禁去掉 override | 门禁第 3 条现在缺失 | P1 |
| G5 | **困难起点采样 70/30**（V8-C §4）：on-the-fly 起点从"合法 Train 起点"改为 70% 普通 + 30% 高 root/heading/vel 误差；支持域检查 | 训练分布仍偏简单 | P1 |
| G6 | **Support Gate**（V8-C §4.3）：标准化 (s,a,ref) 近邻距离，阈值来自 A′ 验证分布 p99 | 防"想象结果当标签" | P1 |
| G7 | **Fixed-WM vs Refreshed-WM 100 步对照**（V7 §13） | 唯一能证明 refresh 有用的实验 | P1 |
| G8 | **严格 resume**（optimizer/scheduler/RNG/sampler/update 编号） | V8-C §11-13 | P2 |
| G9 | 每 tick 物理落盘（`q_raw/q_applied/q_real/qd_real/limit/task_stage`） | V8-C §11；也直接喂 Buffer | P2 |

### 5.3 明确不做（V8-C §13 的纪律）

只跑 **C 一条正式线**；不跑 A/B；不做 rank 扫描、LR 矩阵、Task Head、Critic；不改 SONIC/FT8000；不升 cap；不用 Val/Test 标定 peak 预算；不用想象状态当 WM 真值。

---

## 6. 代码修改清单（按文件，精确）

### 6.1 `scripts/schedule_wm_refresh_v7.py`（新，P0）

```
python3 scripts/schedule_wm_refresh_v7.py \
  --actor-snapshot runs/.../checkpoint_lora_0200.pt \
  --policy-id C --cycle 1 \
  --version-store runs/three_track_v7/wm_versions \
  --replay-old $COMIMIC_MAIN/runs/a_prime_sonic_exc/all_transitions.pt \
  --onpolicy-root runs/three_track_v8/onpolicy/C \
  --collect-windows 128 --collect-seeds 8 --collect-max-steps 600 \
  --old-ratio 0.7 --new-ratio 0.3 --wm-updates 150 \
  --device-train cuda:1 --device-eval cuda:1 \
  --out runs/three_track_v8/c_tracking_recovery/cycles/cycle001
```
行为：① 便宜诊断（token/物理/E20，当前 on-policy 数据）→ `diag.json`；② 若未触发且非强制 → `status=SKIPPED` 退出；③ 触发 → 起带 LoRA 的 server（GPU1, 端口 18098）→ 采 `--collect-windows` 个 20 步起点 → 转 segment → 与旧 replay 合并训练候选 → 门禁 → `publish|reject` → `cycle001.json`（含所有 hash/数字/结论）。

### 6.2 `scripts/train_sonic_lora16_ft8000_bptt.py`（改，P0/P1）

| 改动 | 说明 |
|---|---|
| `--wm-refresh-mode gated` 真实实现 | 每个 segment 开始时 `resolve(--wm-version)` → 校验 sha（`--strict-wm-hash`）→ 重新加载 WM 权重到 bundle → 日志 `[C] segment k WM=... sha=...`；segment 内**拒绝**任何换 WM 的请求 |
| `--refresh-check-every 100` | 到边界时若 `--cycle-plan <json>` 存在，读取该 cycle 的结论（publish/reject），据此决定下一个 segment 的 WM 版本 |
| `--hard-start-ratio 0.30`（G5） | FrameSource 里混 30% 高难度起点（root/heading/vel 误差分位高），带 `--hard-start-support-gate 1` |
| `--save-optimizer 1`（G8） | checkpoint 存 optimizer/scheduler/RNG/sampler 状态与 update 编号 |
| `--tick-physics-dump <dir>`（G9） | 每 tick 落 `q_raw/q_applied/q_real/qd_real/limit/task_stage` |
| 评测 | 每 50 步：原 J20 / Recovery / 物理 8 项 / 困难与普通窗口分开 / per-group 4 路（pristine/仅主干/仅 proj/全开，无梯度） |

### 6.3 `scripts/collect_wm_refresh_isaac.py`（改，P0）

* `--policy-kind {delta,lora}` + `--lora16-ckpt`：C 侧用 LoRA server（复用 `ha_vla_latent64_server.py --lora16-ckpt`）；
* 输出目录按 `onpolicy/<POLICY>/cycle<NNN>` 组织，**累加**而不是覆盖（`MixedReplay` 已支持多目录）；
* 采集完成后自动 `--convert`（已有）+ 动作域检查（已有）+ 写 `policy_manifest.json`（已有）。

### 6.4 `scripts/compare_actor_wm_versions_v7.py`（新，P1）

从同一 `checkpoint_lora_<boundary>.pt` 分两条 100 步：**Fixed-WM(WM_k)** 与 **Refreshed-WM(WM_{k+1})**，相同 batch/噪声/种子/损失；两者都在 WM0 与 WM1 上交叉评估 + 在真实短轨迹上比 E20 与物理动作。输出 `compare.json`。

### 6.5 测试（新增）

```
test_scheduler_triggers              # 触发判定：E20 恶化 → refresh；稳定 → skip
test_segment_wm_switch_atomicity     # segment 内换 WM 必须抛；边界换必须记录 hash
test_collect_lora_policy             # LoRA server + 采集 → segment 合同
test_hard_start_sampling             # 70/30 比例、只用 Train clips、支持域拒绝
test_fixed_vs_refreshed_parity       # 两条臂除 WM 外完全一致
```

---

## 7. 执行计划（含当前 run 的处理）

### Phase 0：冻结（已完成）

* `runs/three_track_v7/phase0_freeze/inventory.json`（20 产物，18 FOUND）；
* WM0 `fc5aeb30…`、A best `5c318a5a…`、C 旧 last `81f88ab0…`（注意：与 WM1 候选 `81d24caf…` 不同）。

### Phase 1–3（已完成）

Buffer / 版本存储 / 门禁 / 诊断 / 第一次离线 refresh（`wm_c1_v001_published`）。

### Phase 4：C 正式训练（进行中，**切 segment 模式**）

当前状态（15:00）：`actor_seed0`，it≈180/400，WM1 冻结，8.0 s/step，val ΔJ **+0.01233 (50) / +0.01494 (100)**，vio 2.84–2.86%。

**按用户决定（方案①）**：
1. 让当前 run 跑到 **it=200** 自动保存 `checkpoint_lora_0200.pt` 与快照，然后**停**；
2. 跑 **cycle001**（G1+G3）：诊断 → 用 it=200 的 C 快照采 128 起点 → 累加 Buffer → 训候选 WM2（150 步，70/30）→ 门禁（含 override 标注）→ publish/reject；
3. 从 `checkpoint_lora_0200.pt` **严格 resume**（G8）到 segment 2（it 200→300），WM 版本 = cycle001 的结论；
4. it=300 边界重复 cycle002；it=400 结束（不再刷新，只做最终评测）。

> 时间注：100 步训练 ≈ 13 分钟，而一个完整 cycle（采集 40–60 min + WM 25 min + 门禁 5 min）≈ 70–90 分钟。
> 所以按 V7 §12 的口径：**每 100 步都做"便宜诊断"，但只在触发时才做完整 refresh**；否则只花 2–3 分钟。

### Phase 5：对照与验收（P1）

* **V7 §13**：`checkpoint_lora_0200` → Fixed-WM vs Refreshed-WM 各 100 步 → 四向交叉评估；
* **V8-C §7**：最终 Best 在独立 Test 上评一次（原 J20 / Recovery / 物理 / 困难窗口 / per-group 4 路）；
* （可选，明确不计入本轮交付）18 对真实闭环。

---

## 8. 验收表（每项标 PASS/FAIL/BLOCKED/NOT_RUN）

| # | 判据 | 当前 |
|---|---|---|
| 1 | 218 层 LoRA 真注入、16,891,904 可训、原 proj 冻结 | **PASS**（日志+`config.json`） |
| 2 | zero-LoRA 与 pristine 逐位一致 | **PASS**（`max|Δ|=0.0`） |
| 3 | 4 步 FM 梯度回到 LoRA | **PASS**（`test_four_step_fm_grad`、`test_sonic_wm_grad_to_lora`） |
| 4 | Recovery 在大误差处给非零梯度、z0 由 Train 冻结 | **PASS**（7 测试 + 校准 JSON） |
| 5 | peak 预算只用 Train、逐关节不同值 | **PASS**（11 个不同值） |
| 6 | 总 Trust 不与主干边际重复惩罚 | **PASS**（`λ_lora=λ_peak_lora=0`，跳过 proj 支路） |
| 7 | 新旧 Reward 分开报告 | **PASS**（`J` 与 `J_pristine_base` 同时记录） |
| 8 | 训练不用 Val/Test 统计量 | **PASS**（`clip_filter=only train clips`；预算/λ 全部 Train） |
| 9 | **周期 refresh（每 100 步检查 + 触发即刷）** | **NOT_RUN**（G1/G3 待写） |
| 10 | **segment 边界换 WM 且记录 hash** | **NOT_RUN**（G2 待写） |
| 11 | **dir_agree 梯度探针** | **BLOCKED**（G4 待跑；当前 refresh 为 override） |
| 12 | Fixed-WM vs Refreshed-WM 对照 | **NOT_RUN**（G7 待写） |
| 13 | 严格 resume（opt/RNG/sampler） | **NOT_RUN**（G8 待写） |
| 14 | 物理限位违反率目标（<2.45%，挑战 2.30%） | **FAIL 暂**（2.84–2.86%） |
| 15 | Same-Obs RMS 落在 10–30 mrad | **PASS 暂**（~12–20 mrad） |
| 16 | 独立 Test（一次） | **NOT_RUN**（训练未结束） |

---

## 9. 时间预算（实测校准）

| 项 | 实测/估计 |
|---|---|
| C 训练 1 步（V8-C 损失、跳过 proj 支路、batch 32） | **8.0 s** → 100 步 ≈ 13 min，400 步 ≈ 53 min |
| WM 候选 150 步（70/30，batch 256，K=20） | **11.4 s/步 → 28 min**（实测） |
| 门禁（96 窗口 × 2 WM × 4 horizon） | **≈ 4 min**（实测） |
| 采集 1 个起点（20 步 + 启动开销约 30 s） | 约 35–50 s ⇒ **128 起点 ≈ 75–110 min**（串行、Isaac 独占） |
| 梯度探针（±扰动） | 约 15–20 min |
| **一个完整 cycle（采集 128 + WM 150 步 + 门禁）** | **≈ 110–145 min** |
| 便宜诊断（只看 E20/token/物理） | 2–3 min |

> 结论：**100 步训练（13 min）远小于一次 refresh（2 小时）**。所以必须"每 100 步诊断、按需刷新"（V7 §12 的原意），不能无条件每 100 步都刷。若触发频繁，采集规模先降到 64 起点。

---

## 10. 风险与纪律

| 风险 | 处置 |
|---|---|
| C 侧采集要换 server 策略、Isaac 独占 | 采集与训练**分 GPU**（训练 GPU0、Isaac+server GPU1）；串行、加锁（已有 `flock`） |
| 中途换 WM 导致曲线不可比 | 每 segment 记录 `wm_version+sha`；报告里按 segment 分列，不拼一条线 |
| 门禁 override 被误读成"验证过" | `gate.json` 里 `override.reason=gradient_probe_not_run`；报告必须原样引用 |
| 采集数据动作域不符 | 采集器已 fail-closed（整体 0.5% 长尾阈值） |
| 版本覆盖 | `wm_version_store` 拒绝同名候选；publish 校验 gate sha == 权重 sha |
| 旧目录被写 | 只写 `runs/three_track_v7/`、`runs/three_track_v8/`；`runs/three_track_v6/`、`runs/a_prime_sonic_exc/` 只读 |

---

## 11. 交付物

1. 本计划 `.md`（本文）；
2. 代码 diff（G1–G9）+ 新增测试（5 个）；
3. 每 cycle 的 `cycleNNN.json`（诊断数字、采集 manifest、WM 训练曲线、门禁 6 条、发布/拒绝）；
4. 400 步训练日志（按 segment 分列，含 `train.jsonl/val.jsonl/cond_check.jsonl`）；
5. `peak_budget_train_v8.json`、`recovery_z0.json`、`physical_weights_v8c.json`；
6. 独立 Test 结果（原 J20 / 物理 / 困难窗口 / per-group 4 路）；
7. Fixed-WM vs Refreshed-WM 对照 `compare.json`；
8. 时间测量（每步、每次采集、每次 WM 更新、每次门禁）；
9. 验收表（§8）逐项 PASS/FAIL/BLOCKED/NOT_RUN。

---

## 12. 执行顺序（不再分叉）

```
[已完成] Phase 0 冻结 → Phase 1 Buffer/采集器 → Phase 2 诊断 → Phase 3 首次 refresh
[进行中] Phase 4a: C 训练 it=0→200（WM1 冻结）        ← 当前 it≈180
[下一步] Phase 4b: 停 at 200 → 写 G1/G3 → cycle001（诊断+采集+WM2+门禁）
[下一步] Phase 4c: 从 200 严格 resume → it=200→300（WM 版本=cycle001 结论）
[下一步] Phase 4d: cycle002（同上）→ it=300→400
[下一步] Phase 5 : 独立 Test + Fixed-vs-Refreshed 100 步对照
[下一步] Phase 6 : 按需启用周期调度（默认 100 步诊断、事件触发刷新）
```
