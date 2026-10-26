# S1（WBC 训练栈）后续计划 — 2026-10-26

> 对象：`trackerlab-supermimic` 的 G1 SuperMimic WBC（token 条件 / obs93 / HA 对齐执行器）。
> 依据：`docs/JOINT_FINETUNE_REVIEW_REDESIGN_V2_20261026.md` §4 的 **S1.1 / S1.2 / S1.3**。
> 只讲 S1。S0（探针）、S2（base_aligned）、S3（联合 v2）不在本文范围。
> 证据等级：✅ 代码或日志里直接看到（给 `文件:行号` / 路径）；🟡 需上机确认。

---

## 0. 一句话

混合库 + WM 共训**已经跑完两段（it 10500 → 16000）**，是这条线唯一"健康且有效"的证据：
HA-only 探针 `sim_r 0.541 → 0.717`、摔倒率 `0.226 → 0.028`（**8 倍**），且训练侧 `imag_rew` 与 `sim_r` 同向（不是 model exploitation）。

但 **S1 的三个门禁一个都还没真正过**：① ckpt 里没有执行器 spec/provenance；② 没有官方协议下的
"HA 全 clip / BONES 防遗忘"数字；③ 没有 O1/O2 执行器上限，**新 WBC 从没进过闭环**。

⇒ **下一步不是继续猛训，而是把它收口成一个可交付的 `WBC_v2`**（1 张卡、约 1.5 h、不训练），
再按门禁数字决定要不要加训一档。

---

## 1. 现状（全部已核实）

### 1.1 训练史：两段 warm start 续训

| 段 | run | 区间 | 库 | lr | WM | 卡 | 时长 | 训练侧末点（混合域，**不能当 HA 指标**） |
|---|---|---|---|---|---|---|---|---|
| 基线 | `bones_seed_token_ha_v2/ckpt_010500_physint.pt`（obs99） | 10500 | BONES-SEED | — | 冻结 | — | — | 自域 `sim_r 0.877 / term 0.0049` |
| 手术件 | 同目录 `ckpt_010500_physint_obs93.pt` | 10500 | 同上 | — | — | — | — | — |
| **seg1** | `logs/wbc_mix_obs93/` | **10500 → 13000** | `bones_seed_plus_ha_x20.pt`（143,923 clips / 60.16 M 帧，HA≈20.5% 帧） | 2e-5 | **共训** | 4×8192 | ≈46 min（1.0 it/s） | `sim_r 0.761→0.816 / term 0.06→0.01 / ep_len 235→383` |
| **seg2** | `logs/wbc_mix2_obs93/` | **13000 → 16000** | `bones_seed_plus_ha_x40.pt`（157,903 clips / 72.84 M 帧，HA≈34.0% 帧） | **5e-5** | **共训** | 4×8192 | **3329.8 s（55.5 min）** | `sim_r 0.790 / term 0.010 / ep_len 418 / wm_loss 0.0883→0.0478` |

- 两段都 `FREEZE_WM=0`、`NO_CRITIC=1`、`MAX_EPISODE_STEPS=0`（全 clip）、`obs93`、`HA_ALIGN=1`、`--interface token`（端口 token）。
- seg2 续训合法且 lr 生效：`runner.py:200-208` 不校验库文件；`runner.py:512-515` 每步用 `lr_at(it)` 重设 lr。
- **32k 从零 BONES 全 clip 重训没成**：`logs/bones_seed_fullclip_4x8192/` 在 it 500 被 `Killed`（11:12），没有底模 v2 产物。

### 1.2 HA-only 纯 rollout 探针（policy/WM/critic 全不更新）

| 权重 | `sim_r` | `term_frac` | 备注 |
|---|---|---|---|
| it 10500 obs99（4×8192） | +0.541 | 0.226 | `docs/PLAN_HA_WBC_FINETUNE_20261025.md` §7.8 |
| it 10500 obs93（4×8192） | +0.536 | 0.254 | 同上，删 6 维净损失≈0 |
| it 11250（x20, lr2e-5） | **+0.680** | **0.064** | `logs/probe_ha_ckpt11250.log` |
| it 12250（x20, lr2e-5） | **+0.694** | **0.046** | `logs/probe_ha_ckpt12250.log` |
| **it 15000（x40, lr5e-5）** | **+0.717** | **0.028** | `logs/probe_ha_ckpt15000.log` |
| it 16000（final） | 🟡 **没跑** | 🟡 | 本次要补 |

- 探针口径：`1 rank × 4096 envs`、50 iter、`WM_ONLY_ITERS=999999`、`FREEZE_WM=1`、`MAX_EPISODE_STEPS=0`、HA 库。
- 训练域参考 `0.877 / 0.005` ⇒ HA 摔倒率差距已从 **45× 收敛到 5.6×**。

### 1.3 关键产物

| 类别 | 路径 |
|---|---|
| WBC 候选 | `trackerlab-supermimic/logs/wbc_mix2_obs93/{ckpt_015000,ckpt_015750,ckpt_016000,last.pt}`、`logs/wbc_mix_obs93/{ckpt_012250,ckpt_013000,last.pt}` |
| 基线 + 手术件 | `logs/bones_seed_token_ha_v2/{ckpt_010500_physint.pt, ckpt_010500_physint_obs93.pt, ckpt_010500_physint_spec.pt(+.spec.json)}` |
| 混合库 | `trackerlab-supermimic/data/bones_seed/{bones_seed_compact.pt, bones_seed_plus_ha_x20.pt, bones_seed_plus_ha_x40.pt}` |
| HA 动作库 / token | `runs/isaac_motion/ha_tokens/packed.pt`（699 clips / 634,016 帧）、`data/port_tokens/ha_tokens/*.npy`（699） |
| 官方评测基线 | `logs/bones_seed_token_ha_v2/ckpt_010500_eval.json`（40 BONES clips：`n_failed 0/40`、`norm_return 0.9036`、326 s） |
| WM 速度榜 | `logs/wm_ckpt_leaderboard.jsonl`（`physint_13500`: qd_corr **0.707** / ratio **0.931**；`retrain_8500`: 0.092 / 5.09） |
| 闭环基线 | PPBox seed0 **278 / 291 步**（`ckpt_010500_physint` + ft4000/anchor4000，**旧 WBC、obs99**） |

### 1.4 S1 门禁完成度

| 文档项 | 要求 | 现状 | 缺什么 |
|---|---|---|---|
| **S1.1** spec 进 ckpt | 训练端保存 `robot_spec`（+`wm_integration`） | ❌ `runner.py:563-576` 的 ck 字典里没有；只有 `scripts/stamp_robot_spec.py` 手工回填了**一个** ckpt（23:45） | 训练端写、消费端断言、其余 5 个候选回填 |
| **S1.2** 混合库共训 | 每 500 iter 三门禁 | 🟡 只有 HA 探针（且只测了 3 个点） | `ckpt_016000` 探针、**BONES 子集防遗忘**、**WM 速度** |
| **S1.3** O1/O2 上限 | 端口 GT token 回放 + SONIC decoder | ❌ `scripts/ha_port_token_server.py` 不存在；现有 `ha_gt_token_server.py` 只吃 npz 的 `encoder_latent` | 新 server、`--sm_obs_mode` 修复、3 seeds × 2 任务 |

---

## 2. 三个必须先说清的事实（本轮新发现）

**F1｜ckpt 不带 spec（✅ 代码核实）**
`runner.py:563-576` 保存的 `small` = `pi/wm/iteration/args/reward_cfg/wm_integration`，**没有** `robot_spec`。
→ 加训得到的 `ckpt_016000` 和 `ckpt_010500` 一样，消费方只能靠环境变量猜 HA_ALIGN/USD。
（`diff_vla_wbc/isaac_joint.py:304-318` 已经在等这个键，缺了会 `SystemExit`，除非 `COMIMIC_ALLOW_NO_SPEC=1`。）

**F2｜闭环脚本硬编码 obs99（✅ 代码核实）**
`scripts/ha_isaac_closedloop.sh:152`、`scripts/ha_fast_eval.sh:131` 都写死 `--sm_obs_mode relative_ori99`；
而新 WBC 是 `paper_obs93`（`args.json` 里 `obs_mode=paper_obs93`，`Policy` 输入 93 维）。
→ **新 WBC 一进闭环就会 obs 维度不匹配**（`action_provider_supermimic.py:219` 按 args 建 hist）。
修法：从 ckpt 的 `args["obs_mode"]` 读，与 CLI 不一致就报错（A/B 才允许覆盖）。

**F3｜HA 探针的 `ep_len` 可疑（🟡 需上机核验）**
所有 HA 探针 `ep_len ≈ 86–87` 步（1.7 s），但按库里 1 s bin 均匀取起点算，**期望剩余 606 帧**；
而 `term_frac` 只有 2.8% ⇒ 绝大多数回合不是"摔倒"结束，却也不是"播到 clip 末"。
→ 说明有东西在 ~87 步处截断（`env.py:346-352` 的 `truncated` 分支）。
**这不影响 ckpt 之间的相对比较，但影响 `term_frac` 的时窗解释**（1.7 s 内的摔倒率 ≠ 长程执行能力）。
→ 对策：**主门禁用官方 `evaluate.py`**（从第 0 帧播到 clip 末、无随机起点、无截断），探针降级为快速看门狗。

---

## 3. 计划（四步，按顺序）

### S1.0 前提核验（15 min，不训练）

1. 记录当前机器事实：**本容器只有 2×RTX 4090（48 GB）**（`nvidia-smi`），而 seg1/seg2 是 4×8192 跑的
   （`logs/wbc_mix2_obs93.launcher.log:2`）。后续按 2 卡排时：4 卡 1.11 s/iter ⇒ 2 卡约 2.2 s/iter（或 `ENVS_PER_RANK=16384` 抵一半）。
2. F3 的三行 debug：跑 3 iter，打印 `truncated / terminated` 次数、`f` 的分布、`clip_end[clip_id[f]]`，
   判清 87 步截断是不是 `clip_end` 口径问题。**判不清也不挡后面**（主门禁已改 evaluate.py）。
3. 把"口径不变量"写成审计行：`HA_ALIGN=1` / USD=`g1_29dof_rev_1_0.usd` / `obs_mode` / `interface=token` / token_dim=64。

**产物**：`runs/S1_audit/audit.json`（spec + 生成环境 + 2 卡事实）。

### S1.1 spec / provenance 写进训练端（45 min）

1. `runner.py`（保存前构造一次）：
   ```python
   spec = {"ha_align": int(rm.HA_ALIGN), "usd": os.environ.get("SUPERMIMIC_ROBOT_USD", ""),
           "joint_names": [...], "default_q": [...], "action_scale": [...], "kp": [...], "kd": [...]}
   prov = {"library": args.motion_packed, "library_sha256": <前 1 MiB 的 sha256 或文件大小+mtime>,
           "obs_mode": args.obs_mode, "interface": args.interface, "token_dim": 64,
           "lr": lr, "iterations": args.iterations, "git": <repo HEAD>, "wm_frozen": args.freeze_wm}
   ck["robot_spec"], ck["provenance"] = spec, prov          # last.pt 与 ckpt_*.pt 都写
   ```
   （`small` 里也要有 `robot_spec`——evaluate / provider / isaac_joint 读的都是 `ckpt_*.pt`。）
2. `scripts/stamp_robot_spec.py` 加 `--batch`，回填既有候选（**另存新文件，不覆盖**）：
   `ckpt_010500_physint_obs93.pt`、`wbc_mix_obs93/{ckpt_012250,ckpt_013000}`、`wbc_mix2_obs93/{ckpt_015000,ckpt_015750,ckpt_016000}`。
3. 消费端断言：`evaluate.py` 与 `action_provider_supermimic.py` 载入时比对 `default_q / action_scale`（1e-6），
   不一致直接 `SystemExit`（`isaac_joint.py:304-318` 已有，照抄）。

**门禁**：6 个候选逐项打印 `[AUDIT][spec] HA_ALIGN=1 default_q[3]=0.420 action_scale mean=0.3789 max=0.7500`，
与 `ckpt_010500_physint_spec.pt.spec.json` 逐字段相等。

### S1.2 官方协议门禁三格（≈1 h，1 张卡）

用**官方 `evaluate.py`**（paper F.1：每条动作从第 0 帧播到 clip 末、无 DR、无噪声、摔倒即失败）：

| 格 | 命令要点 | 基线（v1） | v2 目标 |
|---|---|---|---|
| **HA 全 699 clip** | `--ckpt logs/wbc_mix2_obs93/ckpt_016000.pt --motion_packed runs/isaac_motion/ha_tokens/packed.pt --run_dir logs/wbc_mix2_obs93` | `ckpt_010500_physint_obs93`（预期 fail 高） | `n_failed ≤ 5%（≤34/699）`、`norm_return ≥ 0.85` |
| **BONES 500 子集**（防遗忘） | 同 ckpt，`--motion_packed data/bones_seed/bones_seed_compact.pt --clip_limit 500` | v1 同协议重测（参考：40 clip 时 0/40、0.9036） | `norm_return` 掉点 ≤0.02、`n_failed ≤1%` |
| **WM 速度** | `python3 scripts/eval_wm_velocity.py --ckpt .../ckpt_016000.pt --tag mix2_16000` | `physint_13500`：0.707 / 0.931 | `qd_corr ≥0.9`、`ratio ∈[0.9,1.1]` |

⚠️ **evaluate.py 的输出名只按 ckpt 名生成**（`evaluate.py:274-278` → `<ckpt>_eval.json`）⇒ 同一 ckpt 跑 HA 和 BONES
会**互相覆盖**。每跑完一格立刻改名（`cp ... ckpt_016000_HA699_eval.json`）。

⚠️ WM 这轮**被训过**（`wm_loss 0.0883 → 0.0478`）⇒ 800 步速度标定**必然失效**。若 `qd_corr < 0.9`：
```bash
python3 scripts/finetune_wm_velocity_policy.py --ckpt .../ckpt_016000.pt --out .../ckpt_016000_wmvel.pt --steps 800
python3 scripts/eval_wm_velocity.py --ckpt .../ckpt_016000_wmvel.pt   # 复测，必须过 0.9
```
**没过就不要把这个 WBC 交给 S3**（S3 的奖励梯度建立在 WM 上）。

**产物**：`runs/S1_gates/{v1,v2}/{HA699,BONES500}_{eval.json,log}` + `wm_velocity.json`。

### S1.3 闭环上限门禁 O1/O2（≈1.5 h，1 卡 + Isaac）

1. **修 F2**：两个闭环脚本改成从 ckpt 读 `obs_mode`（默认跟随 ckpt，`SM_OBS_MODE` 才可覆盖），并打印 `[AUDIT][obs]`。
2. **新写 `scripts/ha_port_token_server.py`**（照 `ha_gt_token_server.py` 改，协议不变）：
   数据源改成 `data/port_tokens/ha_tokens/<clip>.npy`（(T,64)），加 `--npy`/`--clip`，其余（cursor、horizon=40、`hand_binary_chunk`）不变。
3. **O1**（执行器上限）：`SM_CKPT=<v2> SM_TOKEN_SOURCE=http SERVER_URL=<port-token server>`，PPBox / DoubleDesk 各 3 seeds、300 步。
4. **O2**（录制执行器参照）：`HA_ALLOW_SONIC=1` + release decoder + `ha_gt_token_server.py`（`encoder_latent`）。
5. **对照**：旧 WBC（`ckpt_010500_physint`，obs99）同协议跑一遍，确认"换代"的净收益。

**门禁**：O1 在 PPBox **≥2/3 seeds 300 步不摔**，且终点到录制目标的距离明显小于旧 WBC；
过了才能进 S3；不过就回 S1.2 加码。

### S1.4 定版与交接（20 min）

1. `WBC_v2 = ckpt_016000(_wmvel).pt`（spec+provenance 齐全、三道门禁过）→ 复制到 `runs/wbc_v2/wbc_v2.pt`。
2. 写 `runs/A1_v2/wbc_v2.json`：sha256、spec、三道门禁数字、生成 run。
3. 声明底模换代：**A1/A2/A4/A5/anchor 旧表 = 底模 v1**；闭环 A1_v2 与 S3 联合 v2 = **底模 v2**，不混表。
4. 给 S3 的交接：用 `bundle.step/act` 做 burn-in；`isaac_joint.load_isaac_bundle` 已按 ckpt `args["obs_mode"]` 自动切 93 维（`isaac_joint.py:254`），无需改。

---

## 4. 决策树（数字触发，别凭感觉）

| 条件（S1.2/S1.3 的读数） | 动作 | 成本 |
|---|---|---|
| HA fail ≤5% 且 BONES 掉点 ≤0.02 且速度门禁过 | **定版 WBC_v2** → 直接进 S1.3 闭环 | 1.5 h |
| HA fail >5%，但失败集中在少数 clip 家族（看 `per_clip`） | 按家族提权重/补库（HA×40 已 34%，可到 ×60 或按任务均衡） | 改库 10 min + 2 h 训练 |
| HA fail >5% 且分散 | 从 `ckpt_016000` 续训 `lr 1e-5` 2000–3000 iter（不要再用 5e-5） | 2 卡 ~1.5–2 h |
| BONES 掉点 >0.02（遗忘） | 回 x20 或 x30 混比续训 1000 iter | ~1 h |
| 速度门禁不过 | 先做 800 步重标定，再看闭环；**不许**把未标定 WM 交给 S3 | 20 min |
| O1 300 步仍摔（≥2/3 seeds 摔） | 回到 S1.2 加码（不是去改 VLA）；补 W8 负载/推扰随机化 | 代码 1 h + 训练 2 h |

---

## 5. 不做的事

1. ❌ **不再从头跑 32k BONES 全 clip 重训**（4 卡 ≈10 h、本机 2 卡 ≈20 h；而混合库已含 66% BONES，且 WM 已被共训过）。
   需要"底模 v2"的 A4 `wm_live` 格子用 WBC_v2 + 重标定即可，不必从零。
2. ❌ 不动 token 接口（端口 64 维），不引入第二个 token 空间。
3. ❌ 不把 obs99 和 obs93 的结果混在一张表（闭环脚本必须先按 F2 修）。
4. ❌ 不拿 run 内 `term_frac` 当指标（`sample_frames` 有 bin_returns 反馈回路，`PLAN §7.4`），只看 fresh 探针 / 官方评测 / 闭环。
5. ❌ 不在 S1 里更新 WBC 的 token 编码器，也不碰 VLA（那是 S2/S3）。

---

## 6. 成本与排期（2 张卡）

| 步 | 内容 | 卡 | 墙钟 |
|---|---|---|---|
| S1.0 | 前提核验 + F3 三行 debug | 1 | 15 min |
| S1.1 | spec/provenance 进训练端 + 回填 6 个 ckpt | — | 45 min |
| S1.2 | HA699 ×2 + BONES500 ×2 + WM 速度 ×2 | 1 | ≈1 h |
| S1.3 | 修 obs_mode + port-token server + O1/O2 + 对照 | 1 (+Isaac) | ≈1.5 h |
| S1.4 | 定版 + 交接 | — | 20 min |
| **合计** | 不含加训 | | **≈4 h** |
| （可选）加训一档 | x40 / lr1e-5 / 3000 iter | 2 | 1.5–2 h |

---

## 7. 需要拍板的两件事

1. **先补门禁（本计划，推荐）还是先加训一档？**
   我建议先补门禁：现在 `0.717 / 0.028` 已经接近平台（`0.694 → 0.717` 用了 2750 iter），
   而"HA 全 clip 失败率"和"闭环"这两个真正决定成败的数字**还是空白**——先测再决定加不加，避免又在没有判据的情况下烧 4 h。
2. **能不能拿回 4 张卡？** 能拿回来的话，若决策树走到"加训"，墙钟砍半。
