# v5 运行时合同（history / schema / 回滚）

> 对应规范：`docs/TEMPORAL_AC_BPTT_V5_MAINLINE_SPEC.md` §4 / §6 / §8 M4 / §12。
> 代码：`diff_vla_wbc/delta_token_io_v5.py`（部署契约）、
> `diff_vla_wbc/delta_token_temporal_v5.py`（Actor）、
> `diff_vla_wbc/action_trust_v5.py`（物理动作预算）。
> **本文件只写"必须满足什么"**；服务端改动等会话 01 落地（规范 §2.4，标
> `BLOCKED_ON_SESSION01`），本版只跑训练 + WM 评测。

---

## 1. 时间轴合同（`T / K / H / ℓ(k)`）

| 量 | 值 | 含义 | 来源 |
|---|---|---|---|
| `T` | **40** | 一次 VLA 生成的 token 数（整段 chunk） | `Gr00tVLM(action_horizon=40)` |
| `K` | **20** | 一次 chunk **实际执行**的 token 数 | 客户端 `HA_VLA_EXEC_STEPS=20` |
| `H` | **20** | 可微 WM 的物理步数（BPTT 深度） | `--horizon 20` |
| `ℓ(k)` | **`k`**（`k<T`；`k≥T` 钳到 `T−1`） | WM 步 `k` 用哪个 token | `rollout.py:366-367` / `delta_token_io_v5.token_index` |

**Per-frame Δ 的施加规则（硬约束）**

```
Δ = actor(Z, state64, hist10, mask)        # (T,64)
chunk64[:K] += Δ[:K]
chunk64[K:] 保持不变                        # Δ 在 t≥K **恒为 0**
```

* 训练：`mask_t = 1[t<K]`，网络在 `t≥K` 输出严格 0（单测 `test_temporal_v5_grad`）；
* 部署：`apply_delta_per_frame(chunk64, Δ, spec)` 会**再断言一次** `Δ[K:] == 0`，
  非零直接 `DeltaTokenContractError`（禁止广播、禁止静默截断）；
* 服务端**必须按帧加**：`chunk64 = chunk64 + Δ`（逐帧），不是 `chunk64 + Δ[0]`。

---

## 2. History10 合同（训练 / 推理 / 部署逐位一致）

### 2.1 块序与帧序

```
hist10  : float32, shape (10, 58)
每帧    : [q_off(29) | qd(29)]          # q_off = q − hist_default_joint_pos
帧序    : 最老 → 最新；hist10[-1] 是**决策时刻**那一帧（不是 t+1）
关节序  : ★ **训练侧 = `standard`**（MJCF/MDP）；**部署侧 state64 / state_hist_10 = `ha_v31`**
          契约层 `reorder_joints(ha_v31 → standard)` 之后才喂网络。
```

> ★ **关节序审计（修正规范 §1.3 的绝对序结论）**：§1.3 的逐位测量只能证明
> "obs93 的 q/qd 与 state64 的 dof_pos/dof_vel **同序**、只差一个 `−default`"，
> 不能区分"都是 HA"还是"都是 STANDARD"。四处代码证据表明**训练侧是 STANDARD**：
> ① `MDP.obs` 直接算 `q − default_q`（`supermimic.py:171`），而
> `bundle.default_joint_pos[:6]=[-0.2,0,0,0.42,-0.23,0]` 的 **knee 在 index 3**（= STANDARD 序）；
> ② `SonicIsaacBundle.step` 注释："状态的关节序天生是 STANDARD（语料/VLA 观测/FK 逐关节限位全按它）"；
> ③ `SonicExecutor.obs994` 对 q/qd/last_action 做 `index_select(std_to_ha)` 才喂 SONIC decoder；
> ④ `delta_token_io`（v4）把训练侧约定写成 `standard`。
> ⇒ **部署契约必须声明 `state_joint_order=ha_v31` 并把入站量换到 standard**
> （`LoadDeltaTokenV5` 已实现；`hist10_from_hist64` 先换序再减 standard 序 default）。

> ★ **两个 `default_joint_pos` 不是同一个东西**（本机实测，混用会产生 ≈0.27 rad 的整体偏移）：
>
> | 用途 | 值（前 5 维） | 来源 |
> |---|---|---|
> | **hist10 的 q 块**（obs93 / SONIC decoder 994） | `[-0.312, 0, 0, 0.669, -0.363, …]` | `compat.g1_default_joint_pos()`（mjlab `KNEES_BENT`） |
> | **动作换算** `q_target = default + scale·a` | `[-0.2, 0, 0, 0.42, -0.23, …]` | `bundle.default_joint_pos`（SONIC/MDP，`SonicIsaacBundle` 构造时换过） |
>
> ckpt meta 同时写 `hist_default_joint_pos`（前者的）与 `default_joint_pos` / `action_scale`
> （后者的，只给动作 trust 的部署侧用）。`hist10_from_hist64` 优先读前者。

* 训练侧一行提取：`hist10 = batch.history[:, :, :58]`（`history` 是 obs93 的
  `[q−default|qd|ω|g|a_prev]`，只取前两块）；
* **不可用块**：`a_prev`（训练侧是参考库推的、部署侧是实际动作）、`gravity`/`ω_body`
  （不同估计管线）——规范 §4.6；
* 部署侧**不要重算**：直接用客户端 decoder 已有的 `(q−default)`、`qd` 两块
  （`hist10_from_blocks`），或由 10 帧 `state64` 序列取 `[6:35]/[35:64]`
  （`hist10_from_hist64`，需要 `default_joint_pos`，已写进 ckpt meta）。

### 2.2 fail-closed（缺一即抛，禁止降级）

| 条件 | 行为 |
|---|---|
| `hist10` 缺失（`use_history=True`） | `DeltaTokenContractError`（训练/推理都抛） |
| 形状 ≠ `(10,58)` 或 dtype ≠ `float32` | 抛（**不接受** float64 自动降级） |
| `hist_clock != "control_tick_50hz"` | 加载 spec 即抛（chunk 级快照不许冒充） |
| 关节序不符（`reorder_joints` 后不一致） | 抛 |
| 请求缺 `state_hist_10` | 服务端 **4xx**；禁止全 0 / 上一份值 / `hist64` 冒充 |
| 网络 `use_history` 与 spec 不一致 | 抛（拒绝静默退化成 no-history） |

对拍产物：每个 run 的 `history_parity.json`（训练侧 vs 部署路径的 `max|Δ|`，要求 ≤1e-6；
实测 `0.0`）。单测：`tests/test_history_and_token_index.py::test_hist10_parity_real_windows`。

---

## 3. Checkpoint schema

`torch.save({...})`：

```python
{
  "delta_token_v5_state": {k: Tensor},        # TemporalDeltaActorV5 的 state_dict
  "delta_token_v5_spec":  {...},              # 见下（含 T/K/H/关节序/cap）
  "delta_token_v5_meta":  {...},              # default_joint_pos / action_scale / seed / commit
  "actor_params": 1744832,
  "ckpt_sha256": "<sha256(tensor dict)>",
  # + extra：it / dJ_val / data_mode ...
}
```

`delta_token_v5_spec` 必含：`arch="temporal_v5"`、`tok_dim/d_model/n_heads/ffn/n_blocks`、
`hist_frames=10`、`hist_dim=58`、`state_dim=64`、`hidden/depth`、`cap`、`out_mode="per_frame"`、
**`T/K/H`**、`state_joint_order`、`hist_joint_order`、`hist_clock`、`sources`。

装载：`load_delta_token_v5_from_checkpoint(obj)` → `(net, spec)`；缺失/形状不符/时间轴不符
一律 `DeltaTokenContractError`。**没有 v5 字段的旧 ckpt 返回 `(None, None)`**（调用方必须显式
落到 Base，不能默认当成 v5）。

---

## 4. 请求 / 响应（**待会话 01 落地**）

```jsonc
POST /infer
{
  "rgb": [...],
  "state64": [64],                  // [rot6d(6) | dof_pos29 | dof_vel29]，HA_V31
  "state46": [46],
  "state_hist_10": [[64], ...],     // (10,64) 连续 50Hz 控制 tick；v5 必需
  "task": "...",
  "flow_seed": 12345
}
```

服务端最小追加（规范 §2.4，**由会话 01 实现**）：

```python
d = delta_from_chunk_full_v5(chunk64, state64, hist10, net, spec)   # (40,64)
chunk64 = apply_delta_per_frame(chunk64, d, spec)                    # 只在 t<K 加
```

* 若 `spec.hist_clock == "control_tick_50hz"` 而请求没有 `state_hist_10` ⇒ **HTTP 4xx**；
* 若 `spec.out_mode == "per_frame"` ⇒ **禁止广播**；
* 任一不满足 ⇒ 只允许加载 **no-history** 模型（单独训练、单独 schema、报告单独一行）。

---

## 5. 回滚

1. **默认仍是 Base**：新权重只走 feature flag（如 `HA_DELTA_TOKEN_MODE=v5`），
   未显式打开时 `disable_delta()` 语义不变；
2. 加载失败 **不降级**：抛错 → 回落 Base 必须由**运维显式**选择并记录日志；
3. 旧 checkpoint（v4 `delta_token_state`）与 v5 互不兼容，两套加载器并存，
   禁止把 v4 权重喂给 v5 网络；
4. 训练侧保留 `use_history=0` 的独立 no-history 架构（1,541,568 参数，`d_in=272`），
   **不是**用零历史冒充（规范 §5）；
5. 回滚清单：`runs/temporal_ac_bptt_v5/<arm>/<seed>/checkpoint_actor_best.pt` +
   `ckpt_sha256` + `git_revision.txt` + `config.json` 一起归档。

---

## 6. 验收口径（每 run 必看）

| 产物 | 内容 |
|---|---|
| `test_once.json` | held-out（独立 clip）`J`、`J_base`、`ΔJ`、29D rad `RMS/peak/p99` |
| `action_budget.csv` | 每 50 步的 val `ΔJ` 与动作预算（看"WM 涨、动作越界"的 reward hack） |
| `history_parity.json` | `max|Δ|`（≤1e-6） |
| `wm_horizon_error.csv` | §5.5.5 的 40 步开环误差（B 路线门槛） |
| `critic_legacy_audit.md` | §1 的审计表（旧 critic 对 Actor 零贡献） |
| `summary.json` | `data_mode`、`uniform_over_eligible`、critic 状态 |

**纪律**：`data_mode=cached_pool` 的数字**不得**与 `on-the-fly` 放在同一张对照表里
（规范 §3.3 / §11）。
