# v5 主线实现规范：**per-frame Δ Actor + Imagined Critic**（可交付执行书）

> **给执行 Agent 的话**：这是一份**可独立执行**的规范，不是研究报告。
> 只做**一条主线**：逐帧 Δtoken（per-frame）+ 可微 WM 的 H 步 BPTT + Imagined Target Critic。
> **不做** Constant / 共享 Δ / 位置模板等对照臂（用户已明确：主线优先）。
> 每条"现状"结论都带 **文件:行号** 或 **可复现命令**；查不到的一律标 `NOT_FOUND` / `AMBIGUOUS`。
>
> **机器/目录**：2×4090 与当前 `comimic` **共目录**（同一份磁盘）。
> 代码改在**独立 worktree/分支**（`comimic-wt03` / `harness/03-bptt-diagnosis`），
> **不改主仓库运行时文件**，不合入 main，除非集成人明确要求。
> **禁止**：覆盖 `runs/bptt1/**`、`runs/harness_0{1,2,3}_*`、他人的 checkpoint / 未提交代码；禁止联网 clone/pull。

---

## §0-**A 本版锁定的默认配置（2026-11-07 定稿，执行时**不要**自行改动）**

```
数据   ：on-the-fly 均匀抽帧（346,729 个合格帧，multinomial）+ 现场 burn-in(10) + 现场 VLA（每步新噪声）
        不建训练池；窗口约束 f>=10 且 f+H<=clip_end
Actor  ：per-frame Δ；**输入整段 40 帧 chunk**（attention 看全段）；**输出只在 t<K=20 生效**，t>=20 恒 0
        结构/参数：Chunk-Attention(1 block,d128,4head,FFN256) + HistoryGRU(58→128) + MLP 400→1024→1024→64
        = 1,744,832 参数；cap=0.08（前 100 updates 从 0.04 线性升）；mask t<K
训练   ：H=20（可微 WM 步数）、γ=0.99、K=20（=客户端 HA_VLA_EXEC_STEPS）、ℓ(k)=k
损失   ：L = -mean(J20) + 0.010·act_trust(29D rad, τ=0.020) + 0.002·smooth + 0.001·anchor
Critic ：**关闭**（critic.enabled=false, beta=0，日志写 BOOTSTRAP_NOT_IDENTIFIABLE(K==H)）
        代码路径与单测仍要实现（供备选 A/B 使用），但**不接入 Actor loss**
预算   ：updates=1000（上限 1200）、batch=256、val 每 50 步、早停=连续 6 次无改善、seeds 0/1/2
        预计 ≈9–10 s/step ⇒ ≈2.6 h/seed；2×4090 建议两个独立进程各跑一个 seed
验收   ：held-out(独立 clip) 的 ΔJ vs Base、动作 29D rad RMS/peak、以及 §5.5.5 的 40 步 WM 误差测量
```

**本版明确不做**：Constant / 共享 Δ / 位置模板 等对照臂；DiT LoRA Stage II；多段 BPTT；
Critic 的 β>0 阶段（原因见 §5.5.3 / §6.1）。

## §0 执行前 60 秒自检

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
git rev-parse --short HEAD          # 期望 >= 788d1ee（主仓库）
git status --short | head           # 记下别人的未提交改动，别碰
cd ../comimic-wt03 && git rev-parse --abbrev-ref HEAD && git rev-parse --short HEAD
# 期望：harness/03-bptt-diagnosis / a4bd1ff
source scripts/harness03_env.sh     # 统一运行环境（worktree 代码 + 主仓库只读数据）
python3 -c "import diff_vla_wbc,sys;print(diff_vla_wbc.__file__)"   # 必须解析到 wt03
```

**两卡怎么用（重要）**：本任务的瓶颈是 **CPU 侧的 op 派发**（`bundle.reward` 单次 ~24 ms 而 CUDA 只有 2.6 ms，实测 profiler：CPU 352 ms vs CUDA 25.6 ms / 10 次）。
⇒ **DDP 的加速比只有 1.3~1.6×；直接开两个进程跑两个 seed 更快、更简单。**
推荐：`CUDA_VISIBLE_DEVICES=0` 跑 seed0、`=1` 跑 seed1，各自独立 `--out`。
（若确实要 DDP，按 §7.4 的约束做，否则更新计数会错。）

---

## §1 现状审计（FOUND / NOT_FOUND / AMBIGUOUS）

### 1.1 `bptt1`（历史最后一个 BPTT 主臂）的真实开关

| 项 | 值 | 证据 | 判定 |
|---|---|---|---|
| `critic` | `True`，`critic_source=imagined` | `runs/bptt1/args.json` | FOUND |
| **`critic_grad_through_state`** | **`False`** | `runs/bptt1/v3_config.json`；分支在 `diff_vla_wbc/v3_step.py:1332` | FOUND |
| ⇒ Critic 对 Actor 的贡献 | **恒为 0**（走 `with torch.no_grad()`，`V(x_H)` 是常数） | `v3_step.py:1332-1345` | FOUND |
| `boot_one_sided` / `critic_rel_clip` / `critic_state_grad_scale` | `True` / `5.0` / `0.1` | `v3_config.json` | FOUND |
| Critic 输入 | `(state 71, ref 71, xi 33)`，**无 future-plan / 无 history** | `diff_vla_wbc/token_critic.py:24-46` | FOUND |
| `v_ev`（train） | 首 10 步 **+0.52** → 末 10 步 **+0.90**（峰 0.9335） | `runs/bptt1/train_metrics.jsonl` 的 `critic/v_ev` | FOUND |
| 旧 Critic 权重 | **`critic_ckpt=None`，从未保存** | `args.json` + `ls runs/bptt1` | **NOT_FOUND ⇒ 不可复用** |
| 采样 | 每步 512 新鲜窗口（2 rank×256），实际 **59** 步 ⇒ ≈30,208 次抽样 | `args.json` + `train_metrics` 行数 | FOUND |
| 结束原因 | 计划 150 步，实际 59 步；日志尾 `SignalException: got signal 15` | `logs/bptt1.log` | FOUND（被 SIGTERM，非崩溃） |
| **WM 回报（J，H20 折扣和）** | **1.51847 → 1.53543 = +1.12%** | `loss/reward_raw` 首/末 10 步均值 | FOUND |
| `‖Δtoken‖` | 0.0754 → **0.1488**（=1.9η，**无 cap**） | `delta_token/norm` | FOUND |
| `window_norm` | 开（`loss/reward_norm` ≈ −0.99，几乎全正窗口） | `v3_config.json` | FOUND |
| `start_max_drift` / 转移数据 | `0.10` / `runs/a_prime_sonic_exc/all_transitions.pt` | `args.json` | FOUND |
| `v13` 的起点 | **不同**：`runs/a_prime_sonic_leash/filtered_std.pt` + `start_max_drift=0.05` | `runs/joint_v13/args.json` | FOUND |

**结论（写作约束）**：v5 的 Critic 是**首次真正接入梯度**，不是"修复旧 bug"；
旧 Critic 权重不存在，**禁止**在文档/日志里写"复用旧 critic / 旧 v_ev 一直很好"。

### 1.2 时间轴合同 `T / H / K / ℓ(k)`

| 量 | 真值 | 证据 |
|---|---|---|
| `T`（一次 VLA 生成的 token 数） | **40** | `Gr00tVLM(action_horizon=40)`；`runs/bptt1/args.json` 的 `horizon=20` 是 WM 步数，不是 T |
| `H`（可微 WM 物理步数） | **20** | `V2Config.horizon=20` |
| **`K`（一次 chunk 实际执行的 token 数）** | **20** | 客户端 `HA_VLA_EXEC_STEPS=20`：`action_provider_supermimic.py:911 rows = arr[cur:cur+exec_steps]` |
| `ℓ(k)`（WM 步 k → token 位置） | **`ℓ(k)=k`（k<T），k≥T 钳到 T−1** | `diff_vla_wbc/rollout.py:366-367` `token_at(step)=tokens[:,min(step,T-1)]` |
| 被训练的 token 位置 | **只有 0..19** | 同上（H=20） |
| 被修改的 token 位置 | **0..39**（服务端广播） | `scripts/ha_vla_latent64_server.py:671` `chunk64 = chunk64 + _d[None,:]` |

⇒ **历史 bug 级问题**：`bptt1` 在 20..39 上**改了 Δ 但从未训练**。
v5 必须 `mask_t = 1[t<K]`、`t≥K` 的 Δ 恒 0，且**部署侧同样只在 t<K 加**。

### 1.3 观测/历史口径（数值已逐位对拍，不是读注释）

环境事实（`python3 -c` 实测）：`bundle.obs_width=93`、`bundle.hist=10`、`term_dims=(29,29,3,3,29)`、
`default_joint_pos[:5]=[-0.2, 0, 0, 0.42, -0.23]`（SONIC 默认角）、`len(action_scale)=29`。

| 检查 | 结果 | 方法 |
|---|---|---|
| `history[:, -1, 0:29]`（obs93 的 q 块） vs `state[7:36] − default` | **max\|Δ\| = 0.0** | 逐位相减 |
| `history[:, -1, 29:58]`（qd 块） vs `state[42:71]` | **max\|Δ\| = 0.0** | 逐位相减 |
| 错序相减（`q_ha − default_ha`） | max\|Δ\| = 1.03 ⇒ **不成立** | 同上 |

⇒ **训练侧 obs93 的 `q/qd` 与 `state64` 的 `dof_pos/dof_vel` 是同一套关节序（HA_V31）**，
只差一个 `−default` 常数偏移。**"一个是 STANDARD、一个是 HA_V31"的说法只适用于
`SonicExecutor.obs994()` 里给 ONNX decoder 的那次换序**（`sonic_mjlab.py:203-212` 的 `index_select(std_to_ha)`），
**不适用于** `history` 与 `state64` 之间。**这条必须写进报告，否则会误导后续实现。**

---

## §2 与主线的 **bug 修复** 对齐（只取修复，不迁移主线）

### 2.1 差异清单（可复现）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
git log --oneline a4bd1ff..788d1ee          # 8 个提交
git diff --name-status a4bd1ff..788d1ee | grep diff_vla_wbc/
```

实测结果（2026-11-07 08:5x）：

```
A  diff_vla_wbc/delta_token_io.py        ← 会话 01 的部署契约（新文件）
A  diff_vla_wbc/teacher_probe.py         ← 会话 02 的 teacher 探针（主线不用）
M  diff_vla_wbc/v3_step.py               ← 会话 02 的 trainer（主线**不经过**它）
```

**并且**：`rollout.py / isaac_joint.py / sonic_isaac.py / sonic_mjlab.py / sonic_decoder.py /
reward_cfg.py / burnin.py / data.py / delta_token.py / prefetch.py / train_v2.py`
在 `a4bd1ff..788d1ee` 之间 **逐文件无改动**（`git diff --stat` 为空）。

⇒ **主线的运行时依赖没有被别人改过**，所以"我的历史结果"与"主线代码"不冲突；
需要补的 bug 修复**只集中在部署契约层**（`delta_token_io.py` + 服务端 + 探针 + 测试）。

### 2.2 **必须取过来**的修复（一次 checkout，5 个文件）

```bash
# 只取这一批（来自 b74d0bf：Δtoken 加载/特征契约统一，修复 35/64 与关节序口径）
git checkout 788d1ee -- \
    diff_vla_wbc/delta_token_io.py \
    scripts/verify_delta_token_deploy.py \
    tests/test_delta_token_io.py \
    tests/test_deploy_audit.py
# 服务端与 landing 探针**不要直接 checkout**（会话 01 还在改）；先看 §2.4 的协调规则
```

| 修复内容 | 为什么主线需要它 |
|---|---|
| `DeltaTokenSpec` + `infer/load/build` + `DeltaTokenContractError` | v5 的 ckpt **必须**声明 `feat_dim/关节序/cap`，否则部署必错 |
| `delta_features_from_state64` 的 **35/64 两口径**与**关节换序** | 历史 bug：35 维与 64 维特征在不同代码路径下不一致 |
| `sha256` / `ckpt_sha256_of_tensor_dict` | §11 的 run manifest 需要 |
| `tests/test_delta_token_io.py` / `test_deploy_audit.py` | 取过来后立刻能跑，作为基线门禁 |

### 2.3 **明确不要取**的（避免把主线迁移进来）

| 文件/提交 | 为什么不取 |
|---|---|
| `diff_vla_wbc/v3_step.py`（062e363/dcdad61 + 未提交改动） | 会话 02 在改；v5 主线**不经过**它（自己调 `imagine_with_tokens`） |
| `diff_vla_wbc/teacher_probe.py`、`scripts/harness02_*`、`train_distill_offline.py` | 教师/蒸馏路线，v5 主线**没有教师** |
| `diff_vla_wbc/deploy_obs.py` + `tests/test_deploy_obs_alignment.py` | **会话 01 未提交的 WIP**。可作**参考**（它的块序文档有用），但不要复制；等它提交后再取 |
| 我的 `scripts/diag_bptt_learnability.py`（已进 788d1ee 的旧快照） | **过期**（缺 per-frame/chunk/gradframe 全部功能）；以 worktree 版本为准 |

### 2.4 服务端/探针的协调规则（不能硬 checkout）

`scripts/ha_vla_latent64_server.py` 与 `scripts/probe_ft_landing.py` 归**会话 01**且仍在改动。
主线需要它们**新增**的能力，只允许以"**最小追加**"形式进：

```
server：新增 delta_from_chunk_full_v5(...)（per-frame、按帧加、hist fail-closed）＋ 新 spec 分支
probe ：新增 --action-rad 报告（29D q_target 增量 RMS/peak），不改既有输出键
```

**在会话 01 交付前，主线可以只跑训练与 WM 评测；部署集成标 `BLOCKED_ON_SESSION01`。**

### 2.5 取完之后的验收（必须全绿）

```bash
python3 -m pytest tests/test_delta_token_io.py tests/test_deploy_audit.py -q
python3 scripts/verify_delta_token_deploy.py --help     # 能起来即可
git status --short | grep -E "v3_step|teacher_probe"     # 必须为空（证明没把主线迁移进来）
```

---

## §3 数据合同：**起点与 `bptt1` 逐项一致**

### 3.1 `bptt1` 的起点配方（原样，逐字段）

| 项 | 值 | 代码位置 |
|---|---|---|
| 转移数据 | `runs/a_prime_sonic_exc/all_transitions.pt`（529,131 行 / 634,016 帧 / 375 个可用 clip） | `args.json` |
| 漂移阈值 | `start_max_drift = 0.10` m | `train_joint_v2.py:~1050` 的 `_w` 计算 |
| 漂移参考库 | `runs/isaac_motion/ha_cmd_exec/packed.pt` 的 `ref`（**注意不是 `ha_cmd`**） | 同上：`_pk = torch.load(ROOT/"runs/isaac_motion/ha_cmd_exec/packed.pt")` |
| 合格判据 | `‖state[:3] − ref[frame][:3]‖ ≤ 0.10` **且** 该帧被真采到过（`_covered`） | 同上 |
| 抽样 | **`torch.multinomial(w, B, replacement=True)`**，`w ∈ {0,1}` ⇒ **在全部合格帧上均匀** | `diff_vla_wbc/data.py: MotionBatchProvider.sample` |
| 坑（照抄，勿"修正"） | `w==0` 的帧用**前一帧**的权重回填（`_w[zero]=_w[(zero-1).clamp(0)]`）⇒ 会让个别不合格帧变成合格 | 同上 |
| burn-in | `k=10` 步，`sample_burnin` | `diff_vla_wbc/burnin.py:131` |
| 扰动 | `xy=0.05 m, yaw=0.10 rad, dq=0.05 rad, dqd=0.30 rad/s` | `args.json` |
| 每步窗口数 | 512（2 rank×256） | `args.json` |

### 3.2 必须改掉的两处偏离（我上一版的问题）

| 项 | 我上一版 | 改成 | 为什么 |
|---|---|---|---|
| 抽样分布 | **按 clip 轮流分层**（每 clip ≈8 个窗口） | **`multinomial(w, B)` 在 346,729 个合格帧上均匀** | 分层会压平 clip 长度权重 ⇒ 优化的是**另一个分布**；均匀才是部署真实分布、且梯度无偏 |
| 数据组织 | 预先建 3,072 / 10,000 窗口"池"，再从池里抽 | **不建池**：每步从合格帧**现场抽** B 个起点 → 现场 burn-in → 现场跑 VLA | 池是"缓存 token"的副产品；见 §3.3 |

> **关于"多样性"**：分层**降低**多样性（每 clip 只 8 个窗口），均匀 + 现场抽样**提高**多样性
> （1000 updates × 256 = 25.6 万次抽样，几乎不重复）。所以"为了多样性"的正确答案是**均匀 + 不建池**。

### 3.3 主训练数据流（**on-the-fly，与 `bptt1` 同构**）

```python
# 每个 optimizer step：
w   = start_weight_mask()                        # §3.1 的 0/1 掩码（含"前一帧回填"那个坑）
idx = torch.multinomial(w, 256, replacement=True) # ← 均匀抽帧
bi  = sample_burnin(bundle, lib, idx, k=10, perturb=True, ...)   # burn-in(10 步) → state/history/refs
Z   = frozen_vla.sample(rgb=frame(idx), state46=..., task=..., noise=randn)  # 现场采样，每步新噪声
# 之后：Actor → Z+Δ → imagine_with_tokens(H) → J → backward
```

**成本（实测口径，两个来源分开标）**：

| 环节 | 成本 | 来源 |
|---|---|---|
| burn-in + WM/reward/decoder 前反向 | ≈ **5–6 s/step** @ B=256, H=20 | 我本轮实测（`diag` 每一步的 `secs`） |
| VLA 前向（2B VLM + 1.09B DiT，4 步 FM，冻结） | **+≈4 s/step** @ B=256 | `docs/REVIEW_TWO_ROUTES…md`：bptt1 实测"每步 ~4 s" |
| 合计 | **≈9–10 s/step** ⇒ 1000 updates ≈ **2.6 h / seed** | |

**⇒ 明确取舍：on-the-fly 比"缓存池"慢约 1.7×，但它是 `bptt1` 的真实口径，并且每步窗口与 token 噪声都是新的。这是主训练的唯一口径。**

**允许的例外（必须标注）**：`--cache-tokens 1` 时用预先缓存的 token chunk（建池 10k ≈ 1.5–3 h），
只用于**快速 pilot / 单元测试 / 反复调代码**；此类 run 的报告必须写
`data_mode=cached_pool`，**不得与 `on-the-fly` 的数字放在同一张对照表里**。

### 3.4 窗口筛选的硬约束（新增）

$$f \ge 10,\qquad f + H + H_{\text{tail}} \le \text{clip\_end}$$

- 下限 `f ≥ 10`：`history` 需要前 10 帧；
- 上限 `f + H + H_tail ≤ clip_end`：**尾部参考帧必须真实存在**，否则 `teacher_references` 会把它们
  clamp 成重复帧 ⇒ 尾部价值退化成一个常数，Critic 学到的是假信号；
- 同时要求 **`f+H+1` 帧与 `f` 同一 clip**（不能跨 clip 借参考）。

---

## §4 **History10 对齐规范**（训练 / 推理 / 部署三处必须逐位一致）

### 4.1 三处口径的**精确块序**（全部实测/读码确认）

| 处 | 宽度 | 每帧块序 | 关节序 |
|---|---|---|---|
| **训练侧 obs93**（`windows.pt['history']`） | 93 | `q−default(29) \| qd(29) \| ω_body(3) \| g_b(3) \| a_prev(29)` | **HA_V31** |
| **部署侧 994**（喂 ONNX decoder） | 994 | `token(64) \| ω(10×3) \| (q−default)(10×29) \| qd(10×29) \| last_action(10×29) \| g(10×3)` | 见 §4.6 |
| **`state64`**（客户端发） | 64 | `rot6d(6) \| dof_pos(29) \| dof_vel(29)` | **HA_V31** |

### 4.2 逐位验证（必须由执行 Agent 复跑一遍）

```bash
python3 - <<'PY'
import torch
from diff_vla_wbc.compat import g1_default_joint_pos
d = torch.load('runs/harness_03_bptt/big/windows.pt', map_location='cpu', weights_only=False)
h, s = d['history'].float(), d['state'].float()
print('q 块 max|Δ| =', float((h[:, -1, 0:29] - (s[:, 7:36] - g1_default_joint_pos())).abs().max()))
print('qd块 max|Δ| =', float((h[:, -1, 29:58] - s[:, 42:71]).abs().max()))
PY
# 期望：两行都是 0.0（或 ≤1e-6）
```

### 4.3 训练侧提取（**一行**，不要自己重算）

```python
hist10 = batch.history[:, :, :58].contiguous()      # (B,10,58) = [q−default | qd]，oldest→newest
# 块序：先 29 维 q，再 29 维 qd；帧序：最老→最新
```

### 4.4 部署侧提取（**从 decoder 已有的 buffer 取，不重算**）

客户端本来就在每个控制步为 SONIC decoder 拼 `(q−default)` 与 `qd` 两块历史（10×29 各一块）。
**直接复用那两块**，按 `[q_block(29) | qd_block(29)]` 拼成 `(10,58)` 放进请求的 `state_hist_10` 字段——
这样训练与推理**共用同一份张量来源**，不存在"重新实现一遍导致块序/关节序错位"的风险。

### 4.5 对齐测试（**必须新增并通过**）

`tests/test_history_and_token_index.py::test_hist10_parity`：

1. 取 N=32 个真实窗口；
2. 用 §4.3 得到 `hist10_train`；
3. 用 §4.4 的部署路径（同一状态序列）得到 `hist10_deploy`；
4. 断言 `max|hist10_train − hist10_deploy| ≤ 1e-6`；
5. 断言块序：把 `hist10` 的前 29 维与 `state64[6:35] − default` 对齐、后 29 维与 `state64[35:64]` 对齐；
6. 断言帧序：`hist10[:, -1]` 对应**决策时刻**那一帧（不是 t+1）。

### 4.6 不可用块与顺序风险的诚实说明

| 块 | 能不能用 | 原因 |
|---|---|---|
| `q−default`、`qd` | ✅ 用 | 两侧同源、逐位验证过 |
| `a_prev` | ❌ 不用 | 训练侧是**参考库推的**（`burnin.py:189` `a_hist=(q_cmd−default)/scale`），部署侧是**实际下发动作** ⇒ 口径不同 |
| `gravity`、`ω_body` | ❌ 不用（首轮） | 同物理量、不同估计管线；要用必须另做对拍 |
| **994 的第 3~5 块顺序** | ⚠️ 必须用**会话 01 的对拍脚本**确认 | `SonicExecutor.obs994()` 里对 `joint_pos/joint_vel/last_action` 做了 `index_select(std_to_ha)`；**部署 provider 的 concat 顺序与它是否一致**，只能靠 `scripts/probe_decoder_alignment.py`（会话 01）判定，不能靠读注释 |

### 4.7 fail-closed 规则（缺一即抛，不许降级）

1. `hist10.shape == (10,58)` 且 `dtype=float32`；
2. `spec.hist_clock == "control_tick_50hz"` 且请求确实来自连续控制 tick（不是 chunk 级 2.6 s 快照）；
3. 请求缺 `state_hist_10` ⇒ **4xx**；**禁止**用全 0 / 上一份值 / `hist64` 冒充；
4. 关节序不符（`reorder_joints` 后不一致）⇒ 抛 `DeltaTokenContractError`；
5. 训练侧 `use_history=1` 时 `hist10 is None` ⇒ 抛错（`delta_token_temporal_v5.py` 已实现）。

---

## §5 Actor 实现（已写好并单测通过，可直接复用）

**文件**：`diff_vla_wbc/delta_token_temporal_v5.py`（worktree 里**已存在**，1,744,832 参数）

```
token_proj 64→128 (8,320) + PE128(40×128) (5,120)
+ TransformerEncoder(1 block, d128, 4 heads, FFN256, pre-LN, dropout0) (132,480)
+ HistoryGRU(58→128) (72,192) + PE16(40×16) (640)
+ MLP 400→1024→1024→64 (1,525,824)
= 1,744,832 参数
```

- `d_in = 400`（`z_t 64 + c_t 128 + h 128 + s0 64 + PE16 16`），与原 1024×2 骨架兼容；
- 末层 `N(0,4e-4)`、bias 0 ⇒ 初始 `|Δ|max ≈ 2.0e-3`（贴近 Base）；
- `mask_t=1[t<K]`，`t≥K` 输出严格 0（已单测）；
- `use_history=False` 自动退化成 **独立 no-history 架构**（1,541,568 参数，`d_in=272`），**不是**用零历史冒充；
- 已通过：形状/`d_in`/mask/各层非零有限梯度/缺 hist 抛错。

**初始化**：只有末层做 `N(0,4e-4)`；其余 PyTorch 默认（**不要**整网 `init_std`）。

---

## §5.5 **Critic 决策：H 选多少？先看历史与本轮证据**

### 5.5.1 历史 H 评估（本地文档，非记忆）

| 时间 | 结论 | 出处 |
|---|---|---|
| 2026-10-30 之前 | `J1_H=8`（rollout 8 步）⇒ **部署窗口 20 帧里第 9–20 帧的奖励从来没被计算过** | `docs/DESIGN_LOSS_S_AND_H20_20261030.md` §1.1 |
| 2026-10-30 提议 | 改成 `H=20`（为了覆盖全部 20 帧奖励、并给 critic 20 步 GAE） | 同上 §0 |
| **2026-10-30 04:30 修订** | **"不做 H=20"**：实测 20 tick 开环误差**平坦**（`pos 1.00×`、`ori 1.04×`），9–20 帧不累积 ⇒ **无额外收益、成本 2.4×** | 同上 顶部修订 |
| 2026-11-06/07 `bptt1` | 仍用 `H=20`（理由是**奖励覆盖**整个执行窗口，不是误差累积） | `runs/bptt1/args.json` |
| 本轮（会话 03）固定窗口消融 | `H=2: const +0.0007 / mlp +0.0033`、`H=5: +0.0028 / +0.0097`、`H=20: +0.0281 / +0.0725`；**按各自 J_H 归一化**后 H=2≈+2.2%、H=5≈+2.6%、H=20≈+4.8% | `runs/harness_03_bptt/arms_runs_w6.jsonl` |

⇒ **历史与本轮一致的两条事实**：① 长 H **不会**带来更强的状态误差惩罚（误差平坦）；
② 长 H 唯一不可替代的作用是**让整段 20 帧的奖励都进目标函数**（否则 9–20 帧完全无监督）。

### 5.5.2 四因素权衡

| 因素 | H 小（5/8/10） | H=20 |
|---|---|---|
| 反向成本 | 低（≈线性） | **高**（≈2.4× vs H=8） |
| 20 帧奖励覆盖 | **不完整**（除非有 Critic 补尾部） | **完整** |
| WM 误差累积惩罚 | 无差别（实测误差平坦） | 无差别 |
| 与 Critic 搭配 | 尾部区间大 ⇒ Critic 有活干，但**尾部最多到 K=20** | 尾部区间 = 0（`K=H`）⇒ **Critic 无可识别目标** |

**关键不等式**：当前运行时 `K = 20`（客户端 `HA_VLA_EXEC_STEPS=20`，20 帧后重新规划）
⇒ **任何 Critic 的合法尾部长度 ≤ `K − H` ≤ 20 步；且必须在同一 chunk 内**。
所以 Critic **不可能**看到"20 步以后"的世界（那需要未来的 VLA 输入，见 §6.1 的 S4/S5）。

### 5.5.3 结论（本规范采用）

| 方案 | 配置 | 何时用 | 判定 |
|---|---|---|---|
| **默认（推荐）** | **`H=20`，`β=0`，不做 Critic** | 当前运行时 `K=20` | ✅ 主线先跑这个：奖励覆盖完整、无 Critic 偏置、无 Gate 失败模式 |
| 备选 A | `H=10`，Critic 尾部 10 步（`K=20`） | 只有当初算力成为瓶颈时 | 🟡 覆盖仍是 20 帧、反向成本减半，但引入 Critic 偏置，**必须与默认臂同预算 A/B** |
| 备选 B | `H=20` + 尾部 20 步 | **只有把运行时改成 `K=40`** 之后 | ✅ 唯一能让 Critic 真正"看更远"（40 步）的合法路径 |
| 不许 | `H=20` 且拿 `Z[20:40]` 当未来 | — | ❌ 那是"另一个策略"（不 replan），Critic 会评估错策略，见 §6.1 S5 |

**为什么默认不做 Critic（客观理由，不是保守）**：
1. `K=H=20` 时尾部**不可识别**（§6.1）；
2. 即使改成 `H=10+尾部`，总视界仍是 20 帧 —— Critic 只是**用近似换算力**，不会延长视界；
3. 我们的 WM 在 20 步内**误差平坦**（`pos 1.00×`、`ori 1.04×`，且 20 步端点 0.82–0.89 cm），
   ⇒ **精确回报比学习出来的 Value 更可靠**；引入 Critic 是拿"可靠的精确量"换"有偏的估计量"；
4. 已有实测正面结果（逐帧 + 全帧 chunk，test `+0.0347`）**就是在 `β=0` 下得到的**，没有任何 Critic 参与。

**Critic 代码路径仍要实现并单测**（因为备选 A/B 都需要它，且要能证明"加了到底有没有用"），
但**默认关闭**：`critic.enabled=false, beta=0`，日志写明 `BOOTSTRAP_NOT_IDENTIFIABLE (K==H)`。

### 5.5.4 三条路线（本版选 A，B/C 留档）

| 路线 | 运行时 | 视界 | 代价 | 本版 |
|---|---|---|---|---|
| **A（本版采用）** | `K=20` 不变 | 20 步（0.4 s） | 无 | ✅ 已验证方向（逐帧+全帧 chunk，cached-pool 口径 test `+0.0347`，2 seed） |
| B（留档） | **`K=40`**（客户端 `HA_VLA_EXEC_STEPS=40`） | 40 步（0.8 s） | ① 开环翻倍 ⇒ 闭环行为改变，必须 Isaac 重测 ② **40 步 WM 误差未测** ③ 可用起点变少 | ⬜ 不进本版 |
| C（留档） | `K=20` 不变，训练时用**数据集同一 clip 的未来图像**造下一段 chunk | 任意长 | 图像不是想象轨迹的真实视角 ⇒ **近似**，必须标注 | ⬜ 不进本版 |

> **注意**：A 的"+0.0347"是**cached-pool + 分层抽样**口径得到的；本版改成 on-the-fly + 均匀抽样后，
> **必须在同一新口径下重测**，旧数字只能当方向性参照，**不得写进同一张对照表**。

### 5.5.5 若要走 B（K=40），先做这个测量（本版也建议顺手做，成本 ~20 min）

**目标**：回答"WM 滚 40 步还可信吗"。这是 B 的唯一前置门槛。

```python
# 数据：data 里同一 clip 的相邻帧就是连续控制步
# 对 N=256 个起点 f：
s = recorded_states[f]                       # 真实状态
for k in range(40):
    s = bundle.step(s, recorded_actions[f+k], xi, step_idx=k+1)   # 单步 WM，喂**录制动作**
    err[k] = ‖s.pos − recorded_states[f+k+1].pos‖, quat_err[k], ‖s.q − q_ref‖
# 输出：err 在 k=1,5,10,20,40 的分位数
```

**验收阈值（建议）**：

| 指标 | 允许上限 | 依据 |
|---|---|---|
| `pos` p50 @ k=20 | ≤ **1.5 cm** | 历史实测 20 步端点 0.82–0.89 cm |
| `pos` p50 @ k=40 | ≤ **3 cm** | 若超过，40 步想象不可用于价值 |
| 误差随 k 的**增长率** | `err[40]/err[20] ≤ 2.5×` | 线性增长可接受；指数增长说明不能滚 40 步 |

不过阈值 ⇒ **B 直接 `NOT_RUN`**，本版继续走 A（这条写进风险表）。

## §6 Critic：**本版默认关闭**（先解决"尾部价值是否可定义"，再谈实现）

> **本版（A 路线）状态**：`critic.enabled=false`、`beta=0`。原因不是"保守"，而是 §6.1 的硬约束：
> `K==H==20` ⇒ 尾部区间为 0 ⇒ `V(y_H)` 不可识别。
> **仍要交付**：`diff_vla_wbc/imagined_value_v5.py` 的完整实现 + 6 个单测（供备选 A/B 打开时使用），
> 以及日志里明确打 `BOOTSTRAP_NOT_IDENTIFIABLE (K==H)`。

### 6.1 关键结论（由 §1.2 的 K 审计直接推出）

`T=40`、`K=20`（执行 20 帧后重新规划）、`H=20`（BPTT 深度）：

> **当 `H == K` 时，rollout 结束时"本次 chunk 已执行完"，下一步需要一次新的 VLA 调用
> （需要未来的 RGB/任务上下文），在想象里**无法合法构造**。
> ⇒ 按本任务书 §4.3 的规则，此时 `V(y_H)` 不可识别，`β` 必须为 0，Stage 2 标 `BLOCKED(BOOTSTRAP_NOT_IDENTIFIABLE)`。**

**两条合法出路（二选一，必须写进 config）**：

| 方案 | 做法 | 代价 | 可识别性 |
|---|---|---|---|
| **S1（推荐）** | `H_actor = 12`（或 10/15），`K = 20`；尾部 = 用**同一 chunk 剩余的 8 个已执行 token**（`t=12..19`）无梯度续跑 WM，得到真实奖励尾部 | 每步多 8 次 WM forward（≈ +40% 时间） | ✅ 合法、无需新 VLA |
| S2 | `H = K = 20`，`β = 0`，只训 No-Critic | 无 | ❌ 但完全合法（先跑通主线） |

**必须一起说清楚的天花板**：S1 的尾部最多覆盖 `K−H ≤ 8` 步（0.16 s），
**因此 Imagined Critic 在本运行时里"延长视界"的收益上限就是这 8 步**——它的主要作用是
**方差压缩/计算节省**，不是"看到更远的未来"。**不要把它宣传成 SHAC 式的长视界收益。**

### 6.2 Critic 网络与输入（复用旧实现，别重写）

```python
# 复用 diff_vla_wbc/token_critic.py:24 的 TokenCritic 作为 backbone
y_t = [ state71 , ref71 , (state−ref)71(use_diff=True) , xi33 ,
        hist10 摊平 58*10=580 , plan_tokens(Z_{ℓ(t)..ℓ(t)+3}) 4*64=256 ,
        remaining_steps 1 , t/T/K 3 ]
in_dim = 71+71+71+33+580+256+1+3 = 1086
ValueMLP：in_dim → 512 → 512 → 1（SiLU）        # 415,745 参数级
```
- **`in_dim` 必须由 `CriticFeatureAdapter` 运行时给出**，不许在配置文件里写死；
- 只写"**实际仍会执行**"的 plan token（`t..K−1`），超过 `K−1` 的位置**不得**填入；
- `ref/xi` 只用于 Critic 训练与回报定义，**不得**进入 Actor 输入。

### 6.3 TD(λ) + EMA Target（精确实现）

```
G_t = r_t + γ·m_t·[(1−λ)·sg(V̄(y_{t+1})) + λ·G_{t+1}]，  γ=0.99, λ=0.95, m_t=1−done_t
G_L = 0（真终止） 或 sg(V̄(y_L))（合法未终止，L = H_actor 的段尾）
L_V = smooth_l1(V_ψ(y_t), sg(G_t), beta=1.0)          # 只更新 ψ，不回传 WM/Actor
V̄ ← 0.995·V̄ + 0.005·V_ψ                              # 每个 critic optimizer step
```
- 另存一份**不依赖自身 bootstrap 的 MC 目标**（用 S1 的真实尾部奖励）算 `v_ev`，
  避免"自己 bootstrap 自己"造成的 EV 虚高（`token_critic.mc_targets_grouped` 可复用）。

### 6.4 Value Readiness Gate（`β` 的唯一入口）

| 条件 | 阈值 | 不满足时 |
|---|---|---|
| 合法 continuation 存在 | S1 生效 | `β=0`，标 `BLOCKED(BOOTSTRAP_NOT_IDENTIFIABLE)` |
| held-out（独立 clip）`v_ev` | **≥ 0.5**（按任务分别报） | `β=0` |
| `MAE/RMSE` 优于"常量均值 Value"基线 | 必须 | `β=0` |
| 末状态方向的**方向一致性**（与独立较长 MC 回报） | **≥ 0.60**，32–64 个未训练窗口 | `INCONCLUSIVE` ⇒ `β=0` |
| 加入尾部后 | 短程 WM 改善不以动作 RMS 暴涨/末状态 Value 异常为代价 | 回退 `β=0` 并回滚 bestVal |

PASS 后 `β: 0 → 1` 在 **100 actor updates 内线性**提升；任一步失败 ⇒ 回 0 + 日志 `VALUE_GATE_FAIL`。

---

## §7 训练循环（两个 optimizer + 正确的 stop-grad）

### 7.1 每步伪代码（映射到真实函数）

```python
idx   = multinomial(pool_w, 256)                      # population（§3.1 口径）
rep   = random.choice([tokens_vla, tokens_vla_rep2])  # 噪声循环（§3.4）
Z     = rep[idx].detach()                             # 真 VLA，冻结
s0    = state64[idx];  h10 = history[idx][:, :, :58]
mask  = ones(T); mask[K:] = 0
raw   = actor(Z, s0, h10, mask, cap=cap_at(it))       # per-frame Δ
Zc    = Z + raw
ro    = imagine_with_tokens(bundle, tokens=Zc, state=state, obs_hist=None,
                            history=hist_930, xi=xi, references=refs,
                            prev_action=prev_action, horizon=H)      # 可微
J20   = discounted_return(ro, 0.99)
y_H   = critic_adapter.build(ro, batch, t=H)
with freeze_params(target_critic):      # 冻结参数，**保留** ∂V/∂y_H
    tail = target_critic(y_H) if beta > 0 else 0
J_ac  = J20 + beta * (0.99**H) * ro.survival[H] * tail
# 物理动作 trust（同 SONIC obs，29D rad）
q_corr = bundle.default_joint_pos + bundle.action_scale * ro.actions   # (H,B,29) rad
q_base = build_base_q_target(idx, burnin)   # Δ=0 的同起点/同 obs 动作（每步现算，见下注）
e      = (q_corr - q_base).permute(1,0,2)                              # (B,H,29)
R_act  = sqrt((e²·M).sum() / (29·M.sum()))
L_act  = relu(R_act/tau_act − 1)²                      # tau_act = 0.020 rad
L_sm   = mean((e[:,1:]−e[:,:-1])²/29) / tau_act²
L_anc  = (Δ²·mask).sum() / (K·B) / cap²
loss   = −J_ac.mean() + 0.010·L_act + 0.002·L_sm + 0.001·L_anc
loss.backward(); assert_finite_nonzero(actor); clip(1.0); actor_opt.step()
# Critic：detach 轨迹 → TDλ → 4 次更新 → EMA
for _ in range(4): ...（§6.3）
每 50 步：held-out 评测 + Gate 判定 + 存 bestVal
```

> **`q_base` 怎么来（on-the-fly 版）**：动作 trust 的参照必须是"同起点、同 SONIC obs、Δ=0"的动作。
> on-the-fly 下没有预计算池，所以**每步多做一次 Δ=0 的 rollout**（H 步，无梯度）——
> 实测这部分约占每步 forward 的 1/3（backward 才是主项），可接受；
> 若要省，可把它与下一次训练的 rollout 合并（同一批 idx 复用），但**不许换成"上一批的参照"**。

### 7.2 stop-grad 规则（必须逐条实现，写进注释）

| 张量 | 规则 |
|---|---|
| `Z` | detach（VLA 冻结，且 token 来自缓存） |
| `hist930`（给 decoder 的） | 不 detach（它在 rollout 内被更新，需要梯度） |
| `h10`（给 Actor 的） | 不 detach（是输入，无梯度来源） |
| `y_H` → `target_critic` | **不 detach**（否则 Actor 拿不到尾部梯度） |
| `target_critic` 参数 | 冻结（`requires_grad_(False)`，**不是** `no_grad()`） |
| `L_V` 的输入/目标 | 全 detach |
| `e`（动作差）里的 `q_base` | detach（参照是常数） |
| `q_corr` | 不 detach（要回传到 Δ） |

### 7.3 训练超参（v5 首轮，写进 config 而不是硬编码）

| 组 | 参数 | 值 |
|---|---|---|
| Actor | `model` | 见 §5（已实现） |
| Actor | `cap` / warmup | `0.08`；前 100 updates 从 `0.04` 线性升到 `0.08` |
| Actor | `mask` | `t<K=20` |
| BPTT | `H` / `γ` | `20`（S2）或 `12`（S1，带尾部）；`0.99` |
| Optimizer | head / encoder LR | `3e-4` / `1e-4`，AdamW，`weight_decay=1e-4`（Norm/Bias 免衰减） |
| Optimizer | grad clip | `1.0` |
| Schedule | warmup / cosine | 50 updates；cosine 降到 head `3e-5` / enc `1e-5` |
| 损失 | `λ_act / τ_act / λ_smooth / λ_anchor` | `0.010` / `0.020 rad` / `0.002` / `0.001` |
| 数据 | 采样 / batch | **on-the-fly 均匀抽帧（§3.3）/ `256`**；不建池 |
| Actor | 输入 / 输出 | 输入 **整段 40 帧** chunk；输出 `(B,20,64)` 有效（`t>=20` 恒 0） |
| Critic | enabled / beta | **false / 0**（`BOOTSTRAP_NOT_IDENTIFIABLE(K==H)`） |
| 训练 | updates / val_interval / early_stop | `1000`（上限 1200）/ `50` / 连续 6 次无改善 |
| 训练 | seeds | `0,1,2` |
| Critic | lr / wd / clip / upd per actor / TDλ / EMA | `3e-4` / `1e-4` / `1.0` / `4` / `0.95` / `0.995` |
| Critic | buffer / fresh_frac / batch | `100,000` 条 imagined step / `0.75` / `2048` |
| 诊断 | `window_norm` / `teacher_aux` / `anti_constant` | **全部 off** |

### 7.4 两卡说明（若真要 DDP）

- 数据按 rank 切分，梯度 all-reduce；`updates` 按**全局**计数；
- 用 `no_sync` 做梯度累积时，**每 rank 的 micro-batch 必须相同**，否则有效 batch 会错；
- 每 rank 的 `metric` 必须 `all_reduce(mean)` 后再写日志（`bptt1` 的 teacher 指标就吃过这个亏，见 `bc861b7`）；
- 仍然建议**两个独立进程跑两个 seed**（§0 的 CPU-bound 结论）。

---

## §8 里程碑与命令

### M0 审计（本文件即交付物）
- 产出：本文 + `docs/LEGACY_BPTT_CRITIC_AUDIT.md`（把 §1 的表格 + 命令输出落盘）；
- **判定**：`critic_grad_through_state=false` ⇒ 旧 critic 对 Actor 零贡献（FOUND）；
  `critic_ckpt=None` ⇒ 旧权重不可复用（NOT_FOUND）。
- `M0 FAIL`：任何一项无法核实 ⇒ 不许进 M2。

### M1 接口 + 单测
```bash
source scripts/harness03_env.sh
python3 -m pytest tests/test_temporal_v5_grad.py tests/test_value_td_lambda.py \
    tests/test_actor_tail_value_grad.py tests/test_history_and_token_index.py \
    tests/test_physical_action_trust.py -q
python3 scripts/diag_bptt_learnability.py exp0 --out <OUT> --n 8 --fd-at grad
# 期望：FD 相对误差 ≤3%（h=1e-2/3e-2；更小的 h 会被 float32 舍入支配）
```

### M2 建池 + 主训练
```bash
OUT=$COMIMIC_MAIN/runs/temporal_ac_bptt_v5
# ① 只建**评测集**（val/test 各 96 个固定窗口，供每 50 步 held-out 评测；不建训练池）
python3 scripts/diag_bptt_learnability.py build --out $OUT/eval_set \
    --n-train 256 --n-val 96 --n-test 96 --batch 64 --vla-batch 4 \
    --uniform-frames 1 --seed 0
# ② 主训练（on-the-fly 抽帧 + 现场跑 VLA；默认 β=0，不做 Critic —— 见 §5.5）
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_temporal_ac_bptt_v5.py \
    --out $OUT/main/seed0 --updates 1000 --batch 256 --horizon 20 --beta 0 --seed 0 \
    --on-the-fly 1 --eval-set $OUT/eval_set &
CUDA_VISIBLE_DEVICES=1 python3 scripts/train_temporal_ac_bptt_v5.py \
    --out $OUT/main/seed1 --updates 1000 --batch 256 --horizon 20 --beta 0 --seed 1 \
    --on-the-fly 1 --eval-set $OUT/eval_set &
```

### M3 真 Isaac 配对（会话 04 的门）
- 24–64 个独立 clip 窗口，同起点 / 同 `Z` / 同 hist / 同 seed；
- 报告 `J_Isaac(Δ) − J_Isaac(Base)`、29D `q_target` 增量 RMS/peak、jerk、跌倒/接触/超时；
- clip 级 bootstrap 95% CI；不足则写 `INCONCLUSIVE`。

### M4 集成
- 新权重只走 feature flag；默认仍 Base；
- 保留 `disable_delta()` / `strict_no_history_fallback`；
- 部署改动等**会话 01** 落地后再做（§2.4）。

---

## §9 测试清单

| 文件 | 断言 |
|---|---|
| `tests/test_temporal_v5_grad.py` | `[B,T,64]+[B,64]+[B,10,58]→[B,T,64]`；attention/GRU/MLP 各层非零有限梯度；`mask_t=0` 处梯度为 0 |
| `tests/test_history_and_token_index.py` | §4.5 的 6 条；`ℓ(k)=k`；`K=min(H,T)`；`reorder_joints` 往返 |
| `tests/test_value_td_lambda.py` | TDλ 递推、真终止 vs timeout、EMA 语义（保留系数）、target detach |
| `tests/test_actor_tail_value_grad.py` | Actor 更新冻结 Critic 参数但 `dJ/dy_H ≠ 0`；`β=0` 时无尾部梯度 |
| `tests/test_physical_action_trust.py` | `q_target` 单位 rad；同 obs 下 `e = scale·(a_corr−a_base)`；peak/smooth 手算一致 |
| `tests/test_v5_fail_closed.py` | 缺 hist / 关节序不符 / T 不符 / arch 不符 / `t≥K` 非零 ⇒ 全部抛异常 |

---

## §10 每个 run 的最低交付物

```
runs/temporal_ac_bptt_v5/<arm>/<seed>/
  config.json  git_revision.txt  split_manifest.json  windows_meta.json
  vla_token_manifest.json  history_contract.json  critic_legacy_audit.md
  train.jsonl  val.jsonl  test_once.json  grad_fd.json
  history_parity.json  action_budget.csv  wm_isaac_paired.csv  wm_horizon_error.csv
  checkpoint_actor_best.pt  checkpoint_actor_last.pt
  checkpoint_critic_best.pt  checkpoint_target_critic.pt     # 仅 S1
  reproduce.sh  final_report.md
```

---

## §11 风险与立即阻断

| 风险 | 检测 | 动作 |
|---|---|---|
| 起点分布与 `bptt1` 不一致 | `windows_meta.json.uniform_over_eligible` 必须为 true | 未改 = 结果不可与历史比较 |
| 假 history | 请求缺 `state_hist_10` | `FAIL`；只允许 no-history 模型 |
| history 块序/关节序错位 | `history_parity.json` 的 `max|Δ|` | >1e-6 ⇒ `FAIL`，不许开训 |
| `t≥K` 仍有 Δ | 单测 + 部署端断言 | `FAIL` |
| 尾部价值不可识别 | `K==H` 且 β>0 | `BLOCKED(BOOTSTRAP_NOT_IDENTIFIABLE)` |
| WM reward hack | WM ΔJ↑ 但动作 RMS 越界 | 暂停，标 `WORLD_MODEL_EXPLOITATION_RISK` |
| 把主线迁移进来 | `git status` 出现 `v3_step.py`/`teacher_probe.py` | 立即回退 |
| 梯度非有限 | `assert_finite_nonzero` | 阻断该 run，记录 seed/idx |
| 长视界 WM 不可信 | 未做 §5.5.5 测量就上 K=40/H_tail | **禁止**；B 路线必须先用 §5.5.5 过阈值 |
| 把 cached-pool 数字与 on-the-fly 数字混表 | 报告里 `data_mode` 字段 | 两张表分开，禁止合并比较 |

---

## §12 文件改动清单（精确路径）

| 类别 | 文件 | 动作 | 现状 |
|---|---|---|---|
| Actor | `diff_vla_wbc/delta_token_temporal_v5.py` | **新增** | ✅ 已写并单测（wt03） |
| 训练器 | `scripts/train_temporal_ac_bptt_v5.py` | **新增** | 🟡 已写；需改成 **on-the-fly 抽帧**（当前读缓存池）+ 补 hist/Critic |
| 评测集构建 | `scripts/diag_bptt_learnability.py` | **改** `build`：加 `--uniform-frames`（评测集用固定窗口即可，训练不建池） | ⬜ 待改（当前是分层抽样） |
| 部署契约 | `diff_vla_wbc/delta_token_io.py` | **从 `788d1ee` 取** | ⬜ |
| 部署契约 v5 | `diff_vla_wbc/delta_token_io_v5.py` | **新增**（per-frame 输出 + hist 字段 + `T/K/H`） | ⬜ |
| Critic | `diff_vla_wbc/imagined_value_v5.py` | **新增**（TDλ + EMA + adapter + Gate） | ⬜ |
| 测试 | `tests/test_temporal_v5_grad.py` 等 6 个 | **新增** | ⬜ |
| 服务端 | `scripts/ha_vla_latent64_server.py` | **最小追加**（等会话 01） | ⛔ |
| 文档 | `docs/TEMPORAL_AC_BPTT_V5_RUNTIME_CONTRACT.md` | 新增（history/schema/回滚） | ⬜ |

---

## §13 一句话交接

**先把 §2 的 4 个文件取过来 + §3 的起点改成"合格帧上均匀抽样" + §4 的 hist10 逐位对齐测试跑绿**，
然后再谈 Critic —— 因为按 §6.1 的审计，**在 `K=H=20` 的当前运行时里，Critic 的尾部价值根本不可识别**，
主线必须先以 `β=0`（S2）跑通并拿到 held-out 数字，再用 `H_actor=12 < K=20`（S1）才允许开 `β>0`。
