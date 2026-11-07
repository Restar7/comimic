# v5 主线交接文档（v2 定稿版）：改了什么 / 完整配置 / 结构设计 / 参数 / 怎么跑 / 怎么验收

> 用途：**其他会话按本文就能复现、改造、训练**，不需要再翻聊天记录。
> 生成时间：2026-11-07 23:5x（mainline v2 正在跑，见 §8）。
> 规范：`docs/TEMPORAL_AC_BPTT_V5_MAINLINE_SPEC.md`；
> 诊断与演进过程：`docs/TEMPORAL_AC_BPTT_V5_DIAGNOSIS_AND_ITERATION.md`（§13 是 v2 更新）；
> 执行记录：`docs/TEMPORAL_AC_BPTT_V5_EXECUTION_REPORT.md`。
> **本文以 v2（修复后）为准**；v1（坍缩版）只作为负基线保留。

---

## 0. 一句话结论（先看这个）

- v1 训练出来的 Actor **坍缩成常数 Δ**（shuffle 输入 cos=0.9999、编码器梯度小 45–180×）。
- **两个 bug 级修复**（都已实测验证）：
  1. **动作 trust 改成"同观测"**：旧实现比的是两条 rollout 的动作差（混入轨迹发散）；
     改成"同一 SONIC obs93、只换 token"重解 action ⇒ **这是决定性修复**；
  2. **训练采样防泄漏**：旧实现从全部 338,402 个合格帧抽样，其中 30.5% 属于 val/test clip；
     改成只取评测集的 **375 个 train clip（235,280 帧）**。
- 加上**未 cap 范数 hinge**（`λ_raw=1e-3, κ=2`）把 `‖u‖` 按在 ~2c。
- **效果**（held-out）：
  - 同数据同预算（cached pilot，200 步，test 96 窗口）：`+0.00099 → +0.02671`（**27×**），
    且超过逐帧常量模板 `+0.00970`（**2.75×**）；
  - on-the-fly 主训练（v2）：val `+0.0004~0.001 → +0.0186~0.0195`（**18–45×**），
    达 oracle 上界的 **62–65%**；条件化门禁 PASS（cos(shuffle chunk)=0.66）。
- Critic 仍**关闭**（`K==H==20` 不可识别）；教师蒸馏留到下一阶段。

---

## 1. 仓库 / 分支 / 目录约定

| 项 | 值 |
|---|---|
| 代码 worktree | `/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03` |
| 分支 | `harness/03-bptt-diagnosis`（基线 `a4bd1ff`） |
| 最新提交 | `2378771`（"eval 增加同观测动作预算"） |
| 主仓库 `comimic` | **只读**：不改代码；只把**文档**同步到 `comimic/docs/`（用户要求） |
| 数据/权重 | 都在主仓库 `runs/`、`data/`（只读） |
| 训练产物 | **worktree 内** `comimic-wt03/runs/temporal_ac_bptt_v5/`（`runs/` 被 gitignore） |
| 环境 | `source scripts/harness03_env.sh`（worktree 代码 + 主仓库只读数据；Python 3.11） |

> ⚠️ 主仓库 `runs/bptt1/**`、`runs/harness_0{1,2,3}_*`、他人 checkpoint **只读，禁止覆盖**。

---

## 2. 改了什么（逐文件）

### 2.1 新增

| 文件 | 作用 |
|---|---|
| `diff_vla_wbc/delta_token_temporal_v5.py`（已有，v2 加 `PositionTemplateActor`） | 条件化 Actor + 逐帧常量模板 |
| `diff_vla_wbc/delta_token_io_v5.py` | v5 部署契约：per-frame Δ、`hist10`、`T/K/H`、关节序换序、fail-closed |
| `diff_vla_wbc/imagined_value_v5.py` | Imagined Critic（TDλ+EMA+Gate）——**代码完整、默认关闭** |
| `diff_vla_wbc/action_trust_v5.py` | 29D rad 动作 trust / 平滑 / anchor / 预算统计 |
| `scripts/train_temporal_ac_bptt_v5.py`（重写） | 主训练器（on-the-fly / cached 双模式、同观测 trust、raw 范数、cond 门禁） |
| `scripts/launch_v5_main_v2.sh` | **v2 主训练启动器**（2 seeds） |
| `scripts/launch_v5_pilot.sh` | 5 臂 pilot（旧 trust / 同观测 / raw / both / 常量模板） |
| `scripts/calibrate_v5_weights.py` | 按梯度尺度标定 λ（act/smooth/anchor/raw） |
| `scripts/verify_v5_deploy_contract.py` | 真 ckpt 的部署契约自检 |
| `scripts/check_history_parity_v5.py` | hist10 强口径对拍（部署序 → 训练序，两条路径） |
| `scripts/diag_oracle_upper_v5.py` | 评测集上 Δtoken 的 oracle 上界 + Actor 方向对齐 |
| `scripts/diag_action_budget_v5.py` | 逐关节/逐控制步动作预算 + 轨迹发散诊断 |
| `scripts/wm_horizon_error_v5.py` / `wm_dump_check_v5.py` | WM 40 步误差（B 路线门槛；两种口径） |
| `scripts/v5_run_report.py` | 运行中/已完成 run 的 markdown 报表 |
| `tests/test_temporal_v5_grad.py` 等 6 个 | 规范 §9 的测试（含防坍缩/契约/fail-closed） |

### 2.2 修改

| 文件 | 改了什么 |
|---|---|
| `scripts/diag_bptt_learnability.py` | `eligible_start_mask(..., allowed_clips=)`（**防泄漏**）、`--uniform-frames`、`--h-tail`、§3.4 窗口硬约束、`FrozenVLASampler`（**冻结全部 VLA 参数**、检查点指纹）、py3.11 f-string 修复 |
| `scripts/train_temporal_ac_bptt_v5.py` | 见 §4/§5/§6（on-the-fly、同观测 trust、raw 范数、cond 门禁、双口径预算、pos_template 臂） |

### 2.3 从 `788d1ee` 取用（v4 部署契约，不改）

`diff_vla_wbc/delta_token_io.py`、`scripts/verify_delta_token_deploy.py`、
`tests/test_delta_token_io.py`、`tests/test_deploy_audit.py`。

### 2.4 明确**没有**动

`diff_vla_wbc/v3_step.py`（会话 02）、`scripts/ha_vla_latent64_server.py`、
`scripts/probe_ft_landing.py`（会话 01）——v5 用新模块接缝，部署集成等会话 01。

---

## 3. 结构 / 设计（精确）

### 3.1 Actor（条件化，1,744,832 参数）

```
输入:
  Z     (B,40,64)   冻结 VLA 本次 chunk（整段）
  state64 (B,64)    [rot6d(6) | dof_pos(29) | dof_vel(29)]，**训练侧 standard 序**
  hist10 (B,10,58)  过去 10 个控制 tick 的 [q−default(29) | qd(29)]，最老→最新

计算:
  hz    = token_proj(Z) + PE_chunk[:40]                  # 64→128
  c_t   = LayerNorm(TransformerEncoder(1 block, d128, 4 head, FFN256, pre-LN))(hz)
  h     = GRU(58→128)(hist10).last                        # (B,128)，广播到每帧
  x_t   = [ z_t(64) | c_t(128) | h(128) | state64(64) | PE_pos(t)(16) ]   # d_in=400
  u_t   = MLP(400→1024→1024→64, SiLU)                     # 末层 N(0,4e-4), bias=0
  Δ_t   = apply_norm_cap(u_t, cap) * mask_t               # mask_t = 1[t<K]
```

参数：`token_proj 8,320 + PE_chunk 5,120 + Transformer 132,480 + LayerNorm 256
+ GRU 72,192 + PE_pos 640 + MLP 1,525,824 = 1,744,832`。

**逐帧常量模板（防伪对照 A2）**：`PositionTemplateActor`，唯一参数 `template (40,64)`，
不看任何输入；其余（cap/mask/优化器/数据）完全相同。

### 3.2 限幅：`apply_norm_cap`（训练/部署**同一函数**）

```python
def apply_norm_cap(delta, cap):
    n = delta.norm(dim=-1, keepdim=True) + 1e-9
    return delta * (torch.tanh(n / cap) * cap / n)
```

- `cap = 0.08`（v2 **固定**，不做 0.04→0.08 warmup）；
- **v1 的坍缩根因**：`‖u‖` 涨到 6.6（82×cap）⇒ tanh 饱和 ⇒ 输入路径梯度被压 ~80×；
- v2 用 **raw 范数 hinge** 把 `‖u‖` 按住（见 §3.3），实测 `‖u‖/cap ≈ 1.8–2.2`。

### 3.3 损失（v2 定稿）

```
J20   = Σ_{k=0}^{H-1} γ^k · survival_k · reward_k        γ=0.99, H=20
L_act = mean_b relu( RMS_b(e_same) / τ − 1 )²            τ=0.020 rad（逐样本 hinge 再平均）
L_sm  = Σ (Δe_same²/29 · M) / Σ M / τ²
L_anc = Σ(Δ²·mask) / (K·B) / cap²
L_raw = Σ_{t<K} relu(‖u_t‖/(κ·cap) − 1)² / (K·B)          κ=2
L     = −mean(J20) + 0.010·L_act + 0.002·L_sm + 0.001·L_anc + 1e-3·L_raw
```

其中 **`e_same` 是同观测动作差**（见 §3.6），`M` 是 survival 权重。

**权重标定依据**（`calibrate_v5_weights.py`，cached B=64，报 `‖g_term(λ=1)‖/‖g_J‖`）：

| ‖u‖/cap | ‖g_J‖ | act（同观测） | smooth | anchor | raw |
|---|---|---|---|---|---|
| 0.09 | 0.987 | 0× | 13.2× | 5.3× | 0× |
| 0.54 | 0.671 | 0× | 14.6× | 21.1× | 99× |
| 1.61 | 0.351 | 2.19× | 23.1× | 16.4× | 2781× |
| 3.57 | 0.211 | 6.43× | 28.8× | 1.03× | 19527× |

⇒ `λ_act=0.010` 只贡献 0.02–0.06×g_J（**旧口径是 27×**）；`λ_smooth=0.002`→2.6–5.8%；
`λ_anchor=0.001`→0.1–2%；`λ_raw=1e-3` 在 2c 处 ≈2.8×g_J（能按住）、在 0.6c 处 ≈0.1×（不干扰）。

### 3.4 优化器 / 调度（锁定）

| 项 | 值 |
|---|---|
| 优化器 | AdamW |
| 学习率 | head `3e-4` / encoder `1e-4`（`weight_decay=1e-4`，**norm/bias 免衰减**） |
| warmup / cosine | 50 步线性；cosine 到 head `3e-5` / enc `1e-5` |
| grad clip | `1.0`（v2 实测总梯度范数 0.1–0.3，基本不触发） |
| batch | `256` |
| H / K / T / γ | `20 / 20 / 40 / 0.99` |

### 3.5 数据流（**on-the-fly**，v2 防泄漏）

```
w = 0/1 掩码：漂移合格帧 (‖pos−ref‖≤0.10) ∩ f≥10 ∩ f+H≤clip_end ∩ **clip ∈ 评测集 train_clips**
idx = torch.multinomial(w, 256, replacement=True)        # 235,280 帧上均匀
bi  = sample_burnin(bundle, lib, idx, k=10, perturb(5cm,0.1rad,0.05rad,0.3rad/s), recorded_states)
Z   = frozen_vla.sample(idx, seed=每步新)                # 现场 VLA，fm_steps=4，每次新流初噪
s64 = state64_from(bundle, bi.state, idx)                # rotlocal6 + q + qd（standard 序）
h10 = bi.history[:, :, :58]                              # [q−default | qd]
```

- 合格帧：**346,729**（只按漂移，与规范一致）→ 叠加窗口约束 **338,402** →
  **防泄漏后 235,280**（375 个 train clip）；`windows_meta.json` 记录
  `n_eligible / n_allowed_clips / clip_filter`；
- 评测集（`runs/temporal_ac_bptt_v5/eval_set`）：**448 窗口 = 256/96/96**，
  按 clip 切 375/81/81，`--uniform-frames 1`，固定缓存 VLA token（**只用于评测**）。

### 3.6 ★同观测动作 trust（v2 的核心修复）

```python
# 修正 rollout 期间：拦执行器级接口（SonicIsaacBundle.act 是类方法，拦 bundle.act 无效）
orig = env.bundle.executor.act
def rec(token, obs93):  recorded.append(obs93);  return orig(token, obs93)
env.bundle.executor.act = rec
J, extra = imagine_with_tokens(tokens = Z + Δ, ...)
env.bundle.executor.act = orig

# 同一 obs93、只换回基础 token 重解（参照 = 常数，no_grad）
q_base_k = q_target( orig(Z_k, obs93_k) )                # 同观测
e_same   = q_target(a^Δ_k) − q_base_k                     # ← L_act / L_smooth 用它
e_traj   = q_target(a^Δ_k) − q_target(a^0_k)              # 旧口径，仅诊断（act_rms_traj_mrad）
```

- 成本：每步多 20 次 decoder forward（小），换来"局部 trust"的干净定义；
- 实测：旧口径 act 梯度 = 27×g_J（把轨迹发散算成动作越界）；同观测口径 = 0.02–0.06×。

### 3.7 条件化门禁（防"学成常量"，规范 §11）

每 `--cond-check-every 25` 步，在 val 96 窗口上：

```
delta_std_across_windows, frame_to_frame_std,
cos(shuffle Z), cos(shuffle state64), cos(shuffle hist10), cos(zero_all)
passed := cos(shuffle Z) < 0.9  and  std_win > 1e-4  and  std_t > 1e-4
```

写入 `cond_check.jsonl`。**不**硬性要求 hist 的 cos < 0.9（历史可能确实无额外信息）。
参考值：v1 坍缩时 cosZ=0.9999；v2 正常时 cosZ=0.62–0.68、std_win≈3.5e-3、std_t≈4.3e-3。

### 3.8 评测协议（每 50 步 val，结束 test）

- 固定 96/96 窗口（独立 clip），`cap=0.08`；
- `ΔJ = mean(J(Δ)) − mean(J(0))`；base J：val **1.50594**、test **1.57581**；
- **动作预算双口径**（`val.jsonl` / `test_once.json`）：
  - 轨迹级：`rms_mrad / peak_mrad / p99_mrad`（部署会真正看到的差，含轨迹发散）；
  - **同观测**：`same_obs_rms_mrad / same_obs_peak_mrad / same_obs_p99_mrad`（局部 trust）。
  - 参考：v2 @it100 同观测 30 mrad / 轨迹级 96–107 mrad / peak ~4 rad（晚期相位差）。

### 3.9 部署契约（`delta_token_io_v5.py`）

- `T=40, K=20, H=20`，`Δ_t` 只在 `t<K` 非零（服务端**按帧加**，禁止广播）；
- `hist10 (10,58)` 必须存在（缺 → 抛/4xx），dtype float32；
- **关节序（重要）**：训练侧 = `standard`（MJCF/MDP）；**部署 `state64`/`state_hist_10` = `ha_v31`**；
  契约层 `reorder_joints(ha_v31 → standard)` 后喂网络（强口径对拍两条路径 `max|Δ|=0`）；
- **两个 default 不是同一个**：hist 的 q 偏移用 `g1_default_joint_pos()`
  （`[-0.312,0,0,0.669,-0.363,…]`，knee@3）；动作换算用 `bundle.default_joint_pos`
  （SONIC，`[-0.2,0,0,0.42,-0.23,…]`）。差 ≈0.27 rad，混用会整体平移；
- ckpt schema：`delta_token_v5_state / delta_token_v5_spec（含 T/K/H/关节序/cap）/ delta_token_v5_meta`
  （含两个 default、data_mode、seed、commit）+ `ckpt_sha256`；
- 真 ckpt 自检：`scripts/verify_v5_deploy_contract.py`（应打印 `PASS`）。

### 3.10 Critic（默认关闭）

`imagined_value_v5.py` 完整实现（TDλ、EMA target、Value Readiness Gate、`in_dim` 运行时给出=1086），
**但 v2 不启用**：`K==H==20` ⇒ 尾部区间 0 ⇒ `V(y_H)` 不可识别，日志写
`BOOTSTRAP_NOT_IDENTIFIABLE`。打开条件：`K>H`（如 `H_actor=12` 的 S1 方案）且 Gate 通过。

---

## 4. 完整配置（v2 主训练，逐参数）

```bash
source scripts/harness03_env.sh
bash scripts/launch_v5_main_v2.sh          # 等价于下面两条（GPU0/GPU1 各一个 seed）
```

每条（seed 0/1）：

```bash
CUDA_VISIBLE_DEVICES=$gpu python3 scripts/train_temporal_ac_bptt_v5.py \
  --out  $WT/runs/temporal_ac_bptt_v5/main_v2/seed$s \
  --seed $s --device cuda:0 \
  --on-the-fly 1 --eval-set $WT/runs/temporal_ac_bptt_v5/eval_set \
  --updates 1000 --batch 256 --horizon 20 --exec-k 20 --beta 0 \
  --trust-mode same_obs --lambda-raw 1e-3 --raw-norm-kappa 2 \
  --cap-warmup-updates 0 --cond-check-every 25 \
  --cap 0.08 --lambda-action 0.010 --tau-action 0.020 \
  --lambda-smooth 0.002 --lambda-anchor 0.001 \
  --head-lr 3e-4 --enc-lr 1e-4 --min-lr-head 3e-5 --min-lr-enc 1e-5 \
  --weight-decay 1e-4 --grad-clip 1.0 --warmup-updates 50 \
  --vla-batch 128 --fm-steps 4 --burnin-steps 10 \
  --start-max-drift 0.10 --perturb-xy 0.05 --perturb-yaw 0.10 \
  --perturb-dq 0.05 --perturb-dqd 0.30 \
  --eval-every 50 --eval-batch 48 --early-stop-patience 6 --log-every 10
```

### 4.1 与规范 §0-A 的偏差（v2，全部显式记录）

| # | 偏差 | 原因 | 影响 |
|---|---|---|---|
| 1 | 输出在 **worktree** `runs/` 而非 `$COMIMIC_MAIN/runs/` | 用户要求"不改 main" | 仅路径 |
| 2 | `--trust-mode same_obs`（规范伪代码是两轨迹差） | 规范原意是"同 SONIC obs"；旧实现混淆轨迹发散，实测导致坍缩 | **决定性修复** |
| 3 | `λ_raw=1e-3, κ=2` 新增 | 防 tanh 饱和（v1 的 ‖u‖=82×cap） | 保持 ‖u‖≈2c |
| 4 | `--cap-warmup-updates 0`（固定 cap=0.08） | 可学习性试验；不做 0.04→0.08 | 更简单，pilot 已验证 |
| 5 | 采样只取 375 train clips（235,280 帧） | **防 val/test 泄漏**（旧实现 30.5% 泄漏） | 数字更保守 |
| 6 | `--cond-check-every 25` | 规范 §11 的"学成常量"门禁 | 新增诊断 |
| 7 | 部署关节序 = 训练 `standard` / 部署 `ha_v31` | 实测（规范 §1.3 绝对序表述有误） | 契约层换序 |
| 8 | FD 在 h=1e-2 时 3.2%（规范 ≤3%） | float32；h=3e-3 时 0.4% | 脚注 |
| 9 | `L_act` 逐样本 hinge 再平均（规范伪代码是整批 RMS） | 更严格的"动作预算" | 更保守 |
| 10 | `--vla-batch 128`（规范未规定） | 显存/吞吐 | 不影响分布 |

---

## 5. 已验证（门禁清单，其他会话必须复跑）

```bash
source scripts/harness03_env.sh
python3 -m pytest tests/test_temporal_v5_grad.py tests/test_value_td_lambda.py \
    tests/test_actor_tail_value_grad.py tests/test_history_and_token_index.py \
    tests/test_physical_action_trust.py tests/test_v5_fail_closed.py \
    tests/test_delta_token_io.py tests/test_deploy_audit.py -q
# => 70 passed, 14 skipped
python3 scripts/check_history_parity_v5.py --eval-set runs/.../eval_set --out <run_dir>
# => passed=true（两条部署路径 max|Δ|=0）
python3 scripts/verify_v5_deploy_contract.py --ckpt <run>/checkpoint_actor_best.pt
# => PASS：1,744,832 参数、T/K/H=40/20/20、‖Δ‖≤cap、t≥K 恒 0、换序 diff=0
```

**梯度管线**（参数级 FD，`head.0/head.4/token_proj` 最大梯度元素）：rel_err **2e-4–5e-3**。

---

## 6. v2 Pilot（证据，cached_pool，200 步；**只用于分离因素，勿与 on-the-fly 混表**）

| 臂 | trust | raw | test ΔJ | cond cosZ@125 | ‖u‖/cap p50 |
|---|---|---|---|---|---|
| p0_old | 两轨迹 | ✗ | +0.00099（坍缩） | 0.9999 | 11.5 |
| p1_trust | 同观测 | ✗ | **+0.02671** | 0.60 | 2.24 |
| p2_cap | 两轨迹 | ✓ | +0.00224（坍缩） | 0.978 | 3.08 |
| p3_both | 同观测 | ✓ | **+0.02640** | 0.68 | 1.80 |
| p4_postpl | — | — | +0.00970（常量模板） | — | — |

> p4 的轨迹级动作预算 = **99.5 mrad**（与 actor 的 96–107 mrad 相当）；同观测口径待补测。

结论：**同观测 trust 是决定性修复**（27×）；raw 单独不够；raw 把 u 从 2.2c 按到 1.8c；
条件化 Actor 在**同一 test、同一 cap** 下比常量模板好 **2.75×**。

---

## 7. 当前主训练 v2（2026-11-07 23:53）

| | seed0 | seed1 |
|---|---|---|
| 进度 | ~it 145 / 1000 | ~it 150 / 1000 |
| 速度 | 15.1 s/step | 14.8 s/step |
| 预计结束 | ≈04:00 | ≈04:00 |
| val ΔJ @50 / @100 | +0.01728 / **+0.01859** | +0.01610 / **+0.01948** |
| cond @100 | cosZ 0.675 PASS | cosZ 0.663 PASS |
| 同观测 actRMS | 30.8 mrad | 29.3 mrad |

oracle 上界（同评测集 val）= **+0.0298** ⇒ 当前 ≈ **62–65%**。
base J：val 1.50594 / test 1.57581。产物：`runs/temporal_ac_bptt_v5/main_v2/seed{0,1}/`。

---

## 8. 怎么跑（其他会话照抄）

### 8.1 一次性准备（已存在，可复用）

```bash
OUT=$WT/runs/temporal_ac_bptt_v5
python3 scripts/diag_bptt_learnability.py build --out $OUT/eval_set \
    --n-train 256 --n-val 96 --n-test 96 --batch 64 --vla-batch 4 \
    --uniform-frames 1 --seed 0
```

### 8.2 主训练（v2）

```bash
cd $WT && bash scripts/launch_v5_main_v2.sh
```

### 8.3 Pilot（复现四臂 + 常量模板）

```bash
cd $WT && bash scripts/launch_v5_pilot.sh        # cached_pool，200 步
```

### 8.4 标定（改任何 λ 之前先跑）

```bash
python3 scripts/calibrate_v5_weights.py \
    --windows $MAIN/runs/harness_03_bptt/big/windows.pt \
    --vla-tokens $MAIN/runs/harness_03_bptt/big/vla_tokens.pt \
    --batch 64 --target-ratio 0.1 --out $OUT/calib.json
```

### 8.5 诊断

```bash
python3 scripts/diag_oracle_upper_v5.py --eval-set $OUT/eval_set --split val \
    --ckpt $OUT/main_v2/seed0/checkpoint_actor_best.pt --out .../oracle_upper_val.json
python3 scripts/diag_action_budget_v5.py --ckpt ... --eval-set $OUT/eval_set --split val
python3 scripts/v5_run_report.py --run $OUT/main_v2/seed0
```

### 8.6 监控

```bash
tail -f $WT/runs/temporal_ac_bptt_v5/main_v2/seed{0,1}.log
cat  $WT/runs/temporal_ac_bptt_v5/main_v2/seed0/val.jsonl
cat  $WT/runs/temporal_ac_bptt_v5/main_v2/seed0/cond_check.jsonl
```

---

## 9. 任务定义（能直接照做的清单）

### 9.1 必须遵守

1. **不改结构**（Attention/GRU/MLP/参数量 1,744,832）；只改训练侧超参/损失时要重新标定（§8.4）；
2. 动作 trust **必须**是同观测（`--trust-mode same_obs`）；`traj` 只用于诊断/复现 v1；
3. 训练采样**必须**只取评测集 train clips（`--eval-set` 且 `windows_meta.train_clips` 存在，否则 fail-closed）；
4. `t≥K` 的 Δ **恒 0**（训练 mask + 部署契约双重断言）；
5. `data_mode` 必须记录；`cached_pool` 与 `on-the-fly` 的数字**不得混表**；
6. Critic 默认关闭；打开需 `K>H` + Gate 通过；
7. 不覆盖 `runs/bptt1/**`、他人 run/ckpt；不改主仓库代码。

### 9.2 待办（下一步实验）

| 优先级 | 任务 | 判据 |
|---|---|---|
| P0 | 等 v2 跑完 | held-out test ΔJ、cond 曲线、双口径预算 |
| P0 | **同预算对齐 vs 常量模板** | 轨迹级已经接近（actor 96–107 mrad vs 模板 **99.5 mrad**）；严格口径需给模板**补测同观测 same_obs 预算**（p4 跑时还没加该字段），若仍不齐再用 `λ_act` 调 |
| P1 | 提速：**VLA 预取**（线程重叠 burn-in） | 目标 15 → 10–11 s/step |
| P1 | seed2（第三张卡/排队） | 3 seeds 的均值±std |
| P2 | `H_actor=12` + Critic（S1） | Gate：v_ev≥0.5、方向一致性≥0.6 |
| P2 | 教师初始化（WM 反传方向蒸馏） | 只在 cap/trust 修好之后 |
| 阻塞 | M3 真 Isaac 配对 / M4 部署 | 需要会话 01 的服务端 per-frame 施加 + `state_hist_10` |

---

## 10. 踩坑清单（其他会话最容易再踩的）

1. **`SonicIsaacBundle.__setattr__` 会把属性赋值转发给 `base`**，而 `act` 是类方法 ⇒
   想拦动作必须拦 `bundle.executor.act(token, obs93)`；
2. `Gr00tVLM.from_pretrained(freeze_vlm=True)` **只冻 backbone**，action head（1.09B）
   默认可训；不显式冻结会在 B=256 时 OOM；
3. **两个 `default_joint_pos` 不是同一个**（hist 用 `g1_default_joint_pos()`，动作用
   `bundle.default_joint_pos`，差 ≈0.27 rad）；
4. 训练侧关节序是 **standard**，部署 `state64` 是 **ha_v31**；
5. `diag_bptt_learnability.py` 原有 py3.12 嵌套 f-string，Python 3.11 下会语法错误（已修）；
6. **数据泄漏**：只按漂移筛帧会把 val/test clip 混进训练（30.5%），必须 `allowed_clips`；
7. `apply_norm_cap` 饱和后输入梯度 ≈ cap/‖u‖；必须监视 `‖u‖/cap` 与 shuffle 敏感性；
8. 旧动作 trust（两轨迹差）在小 Δ 时几乎为零、在轨迹发散时爆炸 ⇒ 不能当"局部 trust"用；
9. `Rollout.history` 只有最终 10 帧；同观测参照必须在 rollout 内记录每步 obs；
10. 评测集 `windows_meta.json` 必须有 `train_clips`（trainer 靠它做防泄漏，缺则 fail-closed）。

---

## 11. 产物清单（每个 run）

```
runs/temporal_ac_bptt_v5/main_v2/seed{0,1}/
  config.json  git_revision.txt  split_manifest.json  windows_meta.json
  vla_token_manifest.json  history_contract.json  history_parity.json
  critic_gate.json  critic_legacy_audit.md  deploy_contract_check.txt
  train.jsonl  val.jsonl  cond_check.jsonl
  test_once.json  action_budget.csv        # 结束时写
  checkpoint_actor_best.pt(.sha256)  checkpoint_actor_last.pt
  reproduce.sh  final_report.md            # 结束时写
```

（`cond_check.jsonl` 是 v2 新增；`test_once.json` 现在同时含 `rms_mrad`（轨迹级）
与 `same_obs_rms_mrad`（同观测）。）
