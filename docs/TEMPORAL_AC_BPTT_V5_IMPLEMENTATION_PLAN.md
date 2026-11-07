# v5 · Temporal-Conditioned Imagined Actor–Critic BPTT —— **适配当前代码**的实现计划

> 生成时间 2026-11-07 08:40。分支：`harness/03-bptt-diagnosis`（worktree `comimic-wt03`，基线 commit `a4bd1ff`）。
> 本文是**落地计划**（不是已完成报告）。所有"现状"结论都带**文件:行号**，全部现场读代码/日志得到；
> 查不到的一律标 `AMBIGUOUS` / `NOT_FOUND`，不猜。
>
> **本文相对你给的 v5 任务书做了三件事**：① 把每个组件映射到本仓库真实符号；② 补上任务书要求但本地缺失的审计结论；
> ③ 给出**不改会话 01/02 文件**的接缝方案（避免和他们正在改的代码冲突）。

---

## §0 一句话结论（先看这个）

v5 的架构（Chunk-Attention + History-GRU + 原 1024×2 头）、损失（H20 回报 + 可信 Value 尾部 + 物理动作 trust）、
门控（Value Readiness Gate）**都能在当前代码上实现**，但要先解决 **4 个真实存在的阻塞点**（不是设计问题，是代码/协议问题）：

| # | 阻塞点 | 现状（代码证据） | 必须做什么 |
|---|---|---|---|
| B1 | **`bptt1` 的 Critic 根本没有梯度进 Actor** | `diff_vla_wbc/v3_step.py:100` 默认 `critic_grad_through_state=False`；`runs/bptt1/v3_config.json` 里它就是 `false` ⇒ 走 `torch.no_grad()` 分支，`V(x_H)` 对 Δ **恒为常数** | v5 必须新写 Target Critic 的**可微尾部项**（冻结参数、保留 `∂V/∂y_H`） |
| B2 | **`DeltaTokenSpec` 只支持 feat_dim ∈ {35,64}、单向量输出** | `diff_vla_wbc/delta_token_io.py:65-88`（`feat_layout` 由 `FEAT_LAYOUTS[feat_dim]` 决定）、`:351` `delta_from_token_chunk` 返回 `(64,)` | 新增 `DeltaTokenSpecV5` + `delta_from_token_chunk_v5` 返回 `(T,64)`；服务端按帧加 |
| B3 | **History10 在服务端拿不到** | `scripts/ha_vla_latent64_server.py:522` `infer(rgb, state64, ...)` 每次请求只有**一份** state64；10 帧 50Hz 历史只在客户端（`action_provider_supermimic.py:_next_latent_impl` 每 chunk 才请求一次） | 要么客户端多传一个 `(10,58)` 字段（约 20 行），要么只允许 `no_history` 模型上线；**禁止假历史** |
| B4 | **T=40 但只有前 20 个 token 被训练** | `diff_vla_wbc/rollout.py:366-367`：`token_at(step)=tokens[:,min(step,T-1)]`，H=20 ⇒ 只用到 t=0..19；而服务端 `chunk64 = chunk64 + _d[None,:]` 把 Δ 加到**全部 40 帧** | v5 必须显式定义 `K`、`ℓ(k)`、mask：默认 `K=min(H,T)=20`、`ℓ(k)=k`、`t≥K` 的 Δ 强制为 0（并在部署侧同样只在 t<K 加） |

---

## §1 审计结果（任务书 §2.1 要求的那张表）

### 1.1 `bptt1` 的 Critic / 采样 / 口径 —— 全部可查，附更正

| 项目 | 值 | 来源 | 判定 |
|---|---|---|---|
| `critic` 开关 | `True` | `runs/bptt1/args.json` | **FOUND** |
| `critic_source` | `imagined` | 同上 + `v3_config.json` | **FOUND** |
| `critic_grad_through_state` | **`False`** | `runs/bptt1/v3_config.json` + `args.json` | **FOUND** |
| Critic 是否有梯度进 Actor | **没有**（`v3_step.py:1332` 的 `if critic_grad_through_state` 为假 ⇒ 走 `with torch.no_grad()`，`boot` 对 Δ 是常数） | `diff_vla_wbc/v3_step.py:1332-1345` | **FOUND（关键）** |
| `boot_one_sided` | `True`（`rel=min(rel,0)`，只惩罚"比参考差"） | `v3_step.py:1348-1355` | **FOUND** |
| `critic_rel_clip` / `critic_state_grad_scale` | `5.0` / `0.1` | `v3_config.json` | **FOUND** |
| `critic_hidden` / `depth` / `lr` | `256` / 2（`TokenCritic` 默认）/ `3e-4` | `v3_config.json` + `token_critic.py:27` | **FOUND** |
| Critic 输入 | **`(state 71, ref_state 71, xi 33)`**，无 future-plan、无 history | `diff_vla_wbc/token_critic.py:24-46` | **FOUND** |
| `v_ev`（train） | 首 10 步 **+0.52** → 末 10 步 **+0.90**（峰值 0.9335） | `runs/bptt1/train_metrics.jsonl` 的 `critic/v_ev` | **FOUND** |
| `v_loss` | 0.152 → 0.030 | 同上 | **FOUND** |
| `v_adv_H_raw` | +0.094 → +0.153（`v_adv_H` = 一侧裁剪后 −0.016 → −0.043） | 同上 | **FOUND** |
| Critic checkpoint | `critic_ckpt=None`（**没有独立保存过 critic 权重**），只有 `runs/bptt1/vla_000050.pt`（含 Δ 权重的 `delta_token_state`） | `args.json` + `ls runs/bptt1` | **NOT_FOUND（旧 Critic 不可复用）** |
| 采样 | 每步 512 个新鲜窗口（2 rank×256），实际 59 步 ⇒ **≈30,208 次**；池 = 全部合格帧 | `args.json`（`batch_size=256`,`NPROC=2`）+ `train_metrics` 步数 | **FOUND** |
| 步数/结束原因 | 计划 150，实际 **59**，日志尾部是 `SignalException: got signal 15`（被 SIGTERM，非崩溃） | `logs/bptt1.log` | **FOUND** |

#### ⚠️ 两处必须更正（相对我 04:33 那版报告）

| 项 | 我之前写的 | **实测真值** |
|---|---|---|
| `bptt1` 的 WM 回报 | "**0 改善**（0.0837→0.0837）" | **`loss/reward_raw`（= J，H=20 折扣和）1.51847 → 1.53543 = +0.01696（+1.12%）**；`rollout/reward` 0.08337→0.08431（+1.13%）。我上一版把 4 位小数看成了"没变"，**错了** |
| 含义 | "bptt1 完全没动" | 它**动了 +1.12%**，正好是"一个共享常向量"的量级（我的 Constant 臂 held-out +0.0141/1.5054 = **+0.94%**）⇒ 历史结论"收敛到常向量"是对的，但**不是"零改善"** |

### 1.2 时间轴合同：`T / H / K / ℓ(k)`（任务书 §2.3）

| 量 | 真值 | 代码证据 |
|---|---|---|
| `T`（VLA 一次生成的 token 数） | **40**（历史值；运行时必须记录） | `args.json: horizon=20`；probe 里 `Gr00tVLM(action_horizon=40)`；`rollout.py` 收到的是 `(B,T,64)` |
| `H`（本轮可微 WM 物理步数） | **20** | `V2Config.horizon=20` |
| `K`（本次 chunk 真正被执行/需修正的位置数） | **20**（= `min(H,T)`，因为 WM rollout 只吃前 20 个 token） | `rollout.py:366` `token_at(step)=tokens[:,min(step,T-1)]` |
| `ℓ(k)`（WM step k → token 位置） | **`ℓ(k)=k`（k<T），k≥T 时钳到 T−1** | 同上（**不是**默认假设，是读代码确认） |
| 未训练位置 | **t=20..39 全部未被训练**，但服务端会把 Δ 加到这 20 帧上（`server:671 chunk64 = chunk64 + _d[None,:]`） | §B4 |
| 真实控制 tick | 0.02 s（50 Hz），H=20 ⇒ 0.4 s；chunk 内执行 20 步（`HA_VLA_EXEC_STEPS=20`） | `control_dt=0.02`；`action_provider_supermimic.py:911` |

⇒ **v5 的硬规则**：`K=min(H,T)`；`mask_t = 1[t<K]`；`t≥K` 的 Δ **恒为 0**；部署侧必须用同一个 mask（否则训练/部署不一致）。若要覆盖 20..39，见 §6 的"分段 BPTT"（M2 之后）。

### 1.3 History10 可行性（任务书 §2.2）

| 项 | 结论 | 证据 |
|---|---|---|
| 训练侧有没有 10 帧历史 | **有**：`windows.pt['history']` = `(N,10,93)`，term 序 `[q(29)\|qd(29)\|ang_vel(3)\|gravity(3)\|a_prev(29)]` | `rollout.py:53 _obs_terms_from_flat` |
| 关节序是否与 `state64` 一致 | **一致（HA_V31）**：`obs93.q − default == state64.dof_pos − default`，`max|Δ|=0.0`；`qd` 同样 `max|Δ|=0.0`（我逐位跑过） | 见 §1.3 验证脚本输出 |
| 单位/偏移 | obs93 的 q 是 **`q − SONIC_default`**；`state64.dof_pos` 是**绝对角**；两者只差常数 | `sonic_mjlab.py:203-212`；`supermimic_obs_to_blocks` |
| `a_prev` 块可用吗 | **不可用**：burn-in 里由参考库推出（`burnin.py:189-197` 的 `a_hist=(q_cmd−default)/scale`），不是实际下发动作 | `burnin.py:189` |
| 部署侧能拿到 10 帧吗 | **现在不能**：`infer()` 每次只收 1 份 state64；50 Hz obs 历史只在客户端（它本来每步都在算，为跑 decoder） | `ha_vla_latent64_server.py:522`；`action_provider_supermimic.py:988` |
| 允许的折中 | 客户端加一个 `(10,58)` 请求字段（q\|qd） ⇒ 逐位同口径；**不加就只能跑 `no_history` 模型** | §B3 |

### 1.4 部署 IO 契约的现状（决定 B2 的改法）

| 符号 | 位置 | 限制 |
|---|---|---|
| `DeltaTokenSpec`（frozen dataclass） | `delta_token_io.py:65` | 字段固定：`token_dim/feat_dim/hidden/depth/init_std/cap/feat_joint_order/state_joint_order/layer_shapes/sources/meta` |
| `feat_layout` | `delta_token_io.py:78` | **只认 `FEAT_LAYOUTS[feat_dim]`，即 35 或 64** |
| `delta_features_from_state64` | `delta_token_io.py:326` | 只从**一份** state64 造 35/64 维 |
| `delta_from_token_chunk` | `delta_token_io.py:351` | 返回 `(64,)`，服务端 `chunk64 + _d[None,:]` **广播** |
| `build_delta_token_net` | `delta_token_io.py:247` | 只造 `DeltaTokenNet`（MLP） |
| 服务端施加点 | `ha_vla_latent64_server.py:651-676` | 用本次 `state64` + `chunk64[0]` 算 Δ，广播到 40 帧 |

⇒ v5 必须**新增**（不修改既有符号，避免和会话 01 冲突）：
`DeltaTokenSpecV5`、`delta_features_from_state64_v5`、`delta_from_chunk_full_v5`、`build_temporal_actor_v5`，
并在 `spec.meta` 里落 `arch/output_mode/T/K/H/hist_layout/normalizer_hash`。

### 1.5 可直接复用的现成件（不重写）

| 用途 | 现成符号 | 位置 |
|---|---|---|
| Critic 网络（V(s, s_ref, xi)） | `TokenCritic(state_dim=71, ref_dim=71, xi_dim=33, hidden, depth, use_diff)` | `token_critic.py:24` |
| Critic GAE 目标 + 更新 | `gae_targets` / `critic_update`（支持 `epochs/minibatches/clip_value`） | `token_critic.py:48/66` |
| Critic 监督拟合（可做独立校验集） | `critic_fit_supervised` | `token_critic.py:134` |
| MC 回报目标（**用于"不依赖自身 bootstrap"的校验**） | `mc_targets_grouped` / `gae_targets_padded` | `token_critic.py:226/166` |
| 可微 WM rollout（token 路线） | `imagine_with_tokens(bundle, tokens, state, history, xi, references, prev_action, horizon)` | `rollout.py:321` |
| 折扣回报 | `discounted_return(rollout, gamma, steps=)` | `rollout.py:298` |
| 软范数上限（训练/部署同一个） | `apply_norm_cap(delta, cap)` | `delta_token.py:38` |
| 物理动作换算 | `q_target = default_q + xi + a*action_scale` | `isaac_joint.py:754`；`supermimic_backend.py:87`；`action_scale_sonic()` `sonic_mjlab.py:91` |
| burn-in 起点 | `sample_burnin(...)` | `burnin.py:131` |
| 现有最好 Actor（基线） | `DeltaParam(kind='mlp_frame_flat')` | `scripts/diag_bptt_learnability.py` |

---

## §2 v5 组件 → 当前代码 的逐项映射

| v5 组件 | 当前代码里对应什么 | 动作 |
|---|---|---|
| `token_proj 64→128` + PE128 + 1×TransformerEncoder | 无（现在是 `DeltaTokenNet` 直接吃拼接向量） | **新增** `diff_vla_wbc/delta_token_temporal_v5.py` |
| History GRU 58→128 | 无（history 现在只喂 decoder） | 同上 |
| 每位置 PE16 | 无 | 同上 |
| 原 1024×2 头 | `delta_token.py:DeltaTokenNet.net`（`Linear→SiLU→Linear→SiLU→Linear`） | **复用同一 skeleton**，改成 `400→1024→1024→64` |
| `apply_norm_cap(raw,0.08)` + mask | `delta_token.py:38` + 新增 mask | 复用函数 + 新增 mask |
| Imagined Target Critic（TDλ + EMA） | `token_critic.py` 有 Critic/GAE，**无 EMA target、无 richer features** | **新增** `diff_vla_wbc/imagined_value_v5.py`（内部复用 `TokenCritic` 作 backbone） |
| Value Readiness Gate / β 调度 | 无 | 新增（训练脚本内） |
| 物理动作 trust（29D rad，同 SONIC obs） | 无（bptt1 的 act hinge 用的是 `rollout.actions` 的 RMS，未换算 rad） | **新增** `physical_action_trust()` 在训练脚本 |
| population 采样（池 3072/10k） | `scripts/diag_bptt_learnability.py --resample --bs` | **复用并扩池** |
| 真 VLA chunk（不换成重复 token0） | 无（现有缓存是我用 `Gr00tVLM` 采的 448/3264 窗口） | **新增** `scripts/cache_real_vla_chunks_v5.py`（带 manifest/hash） |
| `ℓ(k)` / `K` / mask | 无 | 训练脚本内计算（`rollout.py` 不改） |
| 分阶段训练（β gate） | `v3_step.py` 有 critic 但梯度断开、且该文件归会话 02 | **不碰 v3_step**；在 `scripts/train_temporal_ac_bptt_v5.py` 内实现双 optimizer |

**边界（我不动的文件）**：`diff_vla_wbc/v3_step.py`（会话 02）、`scripts/ha_vla_latent64_server.py`（会话 01）、
`scripts/probe_ft_landing.py`（会话 01 正在改）。v5 通过**新模块 + 独立训练脚本**接缝，上线集成时再由你指定谁改服务端。

---

## §3 新模块设计（精确签名）

### 3.1 `diff_vla_wbc/delta_token_temporal_v5.py`

```python
@dataclass(frozen=True)
class TemporalActorCfg:
    tok_dim: int = 64
    d_model: int = 128          # token_proj 输出 / 上下文宽度
    n_heads: int = 4
    ffn: int = 256
    n_blocks: int = 1
    hist_frames: int = 10
    hist_dim: int = 58          # q(29)+qd(29)
    state_dim: int = 64
    pe_pos_dim: int = 16
    hidden: int = 1024          # ★ 骨架保持 1024×2
    depth: int = 2
    init_std: float = 4e-4
    cap: float = 0.08
    out_mode: str = "per_frame"  # "per_frame" | "shared"

class TemporalDeltaActorV5(nn.Module):
    """Z[B,T,64] + state64[B,64] + hist[B,10,58] -> Delta[B,T,64]（masked & capped）"""
    def __init__(self, cfg: TemporalActorCfg): ...
    def forward(self, Z, state64, hist, mask=None) -> torch.Tensor:
        # c = LayerNorm(Transformer(W_Z Z + PE128))(B,T,128)
        # h = GRU_128(hist).last                                  (B,128)
        # x_t = [z_t(64); c_t(128); h(128); s0(64); PE16(t)(16)]   (B,T,400)
        # u   = MLP_400_1024_1024_64(x_t)
        # Δ   = mask_t * apply_norm_cap(u, cap)
    @property
    def n_params(self) -> int: ...
```

**实测参数量（已算）**：

| 块 | 参数 |
|---|---|
| `token_proj 64→128` | 8,320 |
| `PE128 (T=40, learned)` | 5,120 |
| `TransformerEncoder ×1 (d128/4head/FFN256/pre-LN)` | 132,480 |
| `GRU(58→128)` | 72,192 |
| `MLP 400→1024→1024→64` | 1,525,824 |
| `PE16 (T=40, learned)` | 640 |
| **合计** | **1,744,576** |

（对比：我实测最好的 `mlp_frame_flat` 是 3,868,736 —— v5 反而**更小**，因为 chunk 走 attention 而不是 flatten 进第一层。）

**初始化**：末层 `N(0,4e-4)`、bias 0（与 `DeltaTokenNet` 一致，保证第 0 步≈Base）；其余 PyTorch 默认。
**不使用 CNN**（按任务书）。

### 3.2 `diff_vla_wbc/imagined_value_v5.py`

```python
@dataclass(frozen=True)
class CriticCfgV5:
    state_dim: int = 71          # WM 的 flat state
    ref_dim: int = 71
    xi_dim: int = 33
    hist_dim: int = 58*10        # 可选，0 = 不用历史
    plan_dim: int = 64*4         # Z_{ℓ(k):ℓ(k)+4}（"还会执行的那几帧"）
    extra_dim: int = 8           # remaining_steps/T/K/task_id/phase...
    hidden: int = 512
    depth: int = 2
    use_diff: bool = True        # 复用 TokenCritic 的 s−ref
    gamma: float = 0.99
    td_lambda: float = 0.95
    ema_keep: float = 0.995
    huber_delta: float = 1.0

class ImaginedValueV5(nn.Module):
    """y_t -> V(y_t)（实值 raw return，不缩放）。in_dim 由 adapter 运行时决定，不写死。"""
    def forward(self, y: torch.Tensor) -> torch.Tensor: ...

class CriticFeatureAdapter:
    """把 (imagined rollout, batch, t) → y_t；**只允许合法可得字段**。
    明确禁止：未来真实 state、未来 RGB、未执行 token 的 label。"""
    def build(self, imagined, batch, t: int) -> torch.Tensor: ...

def td_lambda_targets(rewards, dones, values_bar, *, gamma, lam) -> torch.Tensor:
    """G_t = r_t + γ m_t[(1−λ)·sg(V̄_{t+1}) + λ·G_{t+1}]，末步按 done/合法 bootstrap 处理。"""

def ema_update(target: nn.Module, online: nn.Module, keep: float) -> None: ...
```

**复用**：backbone 直接用 `TokenCritic(...)`（`token_critic.py:24`），只把输入宽度换成 `CriticFeatureAdapter.in_dim`；
目标函数**不用** `gae_targets`（那是 GAE 用同一网络当 baseline），改用上面的 `td_lambda_targets` + EMA target。
**校验**：必须同时算"不依赖自身 bootstrap 的 MC 目标"（`mc_targets_grouped`）作为独立 EV 校验集。

### 3.3 `diff_vla_wbc/delta_token_io_v5.py`（新文件，不修改 v4 的 `delta_token_io.py`）

```python
@dataclass(frozen=True)
class DeltaTokenSpecV5:
    arch: str                  # "temporal_v5"
    tok_dim: int; d_model: int; n_heads: int; ffn: int; n_blocks: int
    hist_frames: int; hist_dim: int; state_dim: int; pe_pos_dim: int
    hidden: int; depth: int; init_std: float; cap: float
    out_mode: str              # "per_frame"
    T: int; K: int; H: int     # ★ 时间轴合同（部署必须校验）
    feat_joint_order: str      # "ha_v31"
    state_joint_order: str     # "ha_v31"
    hist_clock: str            # "control_tick_50hz" | "chunk_rate"
    sources: dict; meta: dict  # meta 里带 normalizer_hash / train_commit / windows_sha

def build_temporal_actor_v5(spec) -> nn.Module
def load_temporal_actor_v5(sd, spec)                        # strict，形状不符直接抛
def delta_features_from_state64_v5(state64, spec) -> np.ndarray          # (64,)
def delta_hist_from_history10(hist10, spec) -> np.ndarray                # (10,58) 或抛
def delta_from_chunk_full_v5(Z, state64, hist10, net, spec) -> np.ndarray # (T,64)，含 mask+cap
```

**fail-closed 规则（缺一就抛 `DeltaTokenContractError`）**：

1. `Z.shape == (T,64)` 且 `T == spec.T`；
2. `hist10` 存在且 `shape==(10,58)`；`spec.hist_clock` 与运行时收集方式一致；
3. `state64` 关节序 == `spec.state_joint_order`（用 `reorder_joints`，不匹配就抛）；
4. 输出 `(T,64)`，`t≥K` 的列**恒为 0**；服务端必须按帧加（禁止广播）；
5. `spec.cap / hidden / arch` 与 checkpoint 的 `layer_shapes` 一致（`infer_*` 已有同类校验，v5 照抄）。

### 3.4 `scripts/cache_real_vla_chunks_v5.py`

```python
# 冻结官方 VLA（checkpoint-8000）在**真实窗口**上采样并缓存；绝不使用 port_tokens 冒充
# 产物：runs/<run>/vla_chunk_manifest.json + chunks_<shard>.pt
# 每个窗口记录：frame, task, flow_seed, fm_steps, Z(40,64), hand14(40,14), sha256, vlm_ckpt_sha
```

> 我现有的 `diag_bptt_learnability.py build` 已经在做这件事（`_cache_vla_tokens`），v5 版只是把它独立成脚本 +
> 加 manifest（`flow_seed` / `vlm_ckpt_sha` / `windows_sha`），便于审计"token 来源"。

### 3.5 服务端 history 合同（不落代码，先落"必须满足的条件"）

```
POST /infer { rgb, state64, state46, state_hist_10(10,64 | 10,58), task, ... }
```
- 若 `spec.hist_clock == "control_tick_50hz"` 而请求里没有 `state_hist_10` ⇒ **HTTP 4xx + 明确错误**，不许用全 0 或上一份值代替；
- 若 `spec.out_mode == "per_frame"` ⇒ 服务端 `chunk64[:K] += Δ[:K]`，`chunk64[K:]` **不改**；
- 若以上任一不满足 ⇒ 只允许加载 `no_history` 模型（单独训练、单独 checkpoint schema、报告里单独一行）。

---

## §4 训练循环（映射到真实函数）

```python
# 冻结：VLA 权重、SONIC decoder 参数、WM 参数、Target Critic 参数（但保留 ∂V/∂y_H）
# 可训：Actor θ；online Critic ψ
for it in range(1, 1201):
    batch = pool.sample(batch=256)                 # ← population：池 10k~30k，每步重抽
    Z     = batch.vla_chunk.detach()               # (B,T,64) 真 VLA，来自 cache_v5 manifest
    s0, hist = batch.state64, batch.hist10          # (B,64) / (B,10,58)
    mask  = batch.exec_mask                        # (B,T) 1[t<K]
    raw   = actor(Z, s0, hist, mask)               # (B,T,64)
    delta = apply_norm_cap(raw, cap_schedule(it)) * mask[..., None]
    Zc    = Z + delta

    # --- 可微 WM+Sonic 前向（复用 imagine_with_tokens，ℓ(k)=k）---
    ro = imagine_with_tokens(bundle, tokens=Zc, state=batch.state, obs_hist=None,
                             history=batch.history_Bx10xW, xi=batch.xi,
                             references=batch.refs, prev_action=batch.prev_action,
                             horizon=H)            # H=20
    J20 = discounted_return(ro, gamma=0.99)        # (B,)

    # --- 尾部 Value（冻结参数、保留梯度）---
    y20   = critic_adapter.build(ro, batch, t=H)
    beta  = gate.beta(it)                           # Gate 前 0；PASS 后 100 步线性 →1
    with freeze_params(target_critic):
        V20 = target_critic(y20)
    J_ac = J20 + beta * (0.99**H) * ro.survival[H] * V20

    # --- 物理动作 trust（同 SONIC obs，29D rad）---
    q_corr = bundle.default_joint_pos + bundle.action_scale * ro.actions      # (H,B,29) rad
    q_base = precomputed_base_q_target(batch)                                 # 同起点/同 obs 的 Δ=0 动作
    e = q_corr - q_base
    act_rms = sqrt((e.pow(2).sum(-1) * M).sum() / (29*M.sum() + eps))
    L_act = relu(act_rms/tau_act - 1).pow(2)                                  # τ=0.020
    L_smooth = mean((e[1:]-e[:-1]).pow(2).mean(-1) / tau_act**2)
    L_anchor = (delta.pow(2).sum(-1) / cap**2).mean()

    loss = -J_ac.mean() + 0.010*L_act + 0.002*L_smooth + 0.001*L_anchor
    actor_opt.zero_grad(set_to_none=True); loss.backward()
    assert_finite_nonzero_grad(actor)               # ★ M1 的门禁
    clip_grad_norm_(actor.parameters(), 1.0); actor_opt.step()

    # --- Critic：detach 轨迹 + TDλ + EMA（4 次更新）---
    recs = detach_records(ro, batch, valid_only=True, version=it)
    vbuf.add(recs)
    for _ in range(4):
        vb = vbuf.sample(2048, fresh_frac=0.75)
        with torch.no_grad():
            G = td_lambda_targets(vb.r, vb.done, target_critic(vb.y_next), gamma=.99, lam=.95)
        lv = F.smooth_l1_loss(online_critic(vb.y), G, beta=1.0)
        critic_opt.zero_grad(set_to_none=True); lv.backward()
        clip_grad_norm_(online_critic.parameters(), 1.0); critic_opt.step()
        ema_update(target_critic, online_critic, 0.995)

    if it % 50 == 0:
        val = validate_heldout(raw_j20=True, action_budget=True)
        gate.update(val)                    # 只用 val 的独立 clip；绝不用 test
        save_best(val); save_last()
```

**与现有代码的接缝**：
- `imagine_with_tokens` 的 `history` 参数要的是 **WBC 的 `(10,B,93|99)`**，不是 v5 的 `(B,10,58)` —— 两者都要传，别混：`hist10` 给 Actor，`history` 给 decoder。
- 基线 `q_base` 必须用**同一批起点、同一 obs、Δ=0** 跑一次得到（我 `_eval_arm` 里已经有这个逻辑，可直接抽函数）。
- `bundle.actions` 是 **STANDARD 序、归一化**，乘 `action_scale` 之前**不要**再换序（`act()` 已返回 STANDARD；`default_joint_pos` 已被 `SonicIsaacBundle` 换成 SONIC 默认角）。

---

## §5 数据与采样

| 项 | v5 取值 | 与现有实现的关系 |
|---|---|---|
| 起点 | 真 Isaac 转移，漂移 ≤0.10 m（346,729 帧合格） | 已有（`start_weights`） |
| burn-in | 10 步，扰动 5cm/0.1rad | 已有（`sample_burnin`） |
| **独立窗口池** | 先 3,072（已有缓存）→ 扩到 **10,000**（VLA 采样约 1.5 h）→ 资源许可 30k | 现有 `big/windows.pt` 是 3,264（3,072 train） |
| 采样 | 每步从池中重抽，`global_batch=256` | 已有（`--resample --bs 256`） |
| 切分 | 按 clip 分组，375/81/81 | 已有 |
| VLA chunk | 真 VLA 采样并缓存，带 manifest | 已有（`_cache_vla_tokens`），v5 版加 manifest |
| 覆盖审计 | 记录每个 token 被训练次数（`t<K` 才计数） | **新增** |

---

## §6 实验矩阵（任务书 §10 的 A0–A10 → 用我现有 harness 怎么跑）

| ID | 臂 | 我现在的实现 | 状态 |
|---|---|---|---|
| A0 | Δ=0 | `--arms base` | ✅ 已跑（J: train 1.5061 / val 1.5086 / test 1.4554） |
| A1 | Constant64 | `--arms constant` | ✅ 已跑（test +0.0141） |
| **A2** | **逐帧位置模板 `c_t`** | **新增** `--arms pos_template`（参数 `(T,64)`，只随 t 变，不看任何输入） | ⬜ 必做（**核心防伪对照**） |
| A3 | 共享 Δ MLP（token0+state） | `--arms mlp` | ✅ 已跑（test +0.0195） |
| A4 | per-frame + flatten chunk | `--arms mlp_frame_flat` | ✅ 已跑（test +0.0347，2 seed） |
| A5 | v5 Actor（无 history） | 新 Actor + `hist=0` | ⬜ |
| **A6** | **v5 Actor + Target Critic + 真 history** | 主方案 | ⬜ |
| A7 | A6 但 β=0（No-Critic） | 同一 Actor，`beta_schedule=0` | ⬜ |
| A8 | A6 + Actor 输入 history 打乱 | `--ablate shuffle_hist` | ⬜ |
| A9 | A6 + Actor 的条件 chunk 打乱 | `--ablate shuffle_chunk` | ⬜ |
| A10 | A6 + 动作约束=0 | `--lambda-action 0` | ⬜ |

**A2 必须先跑**，否则无法排除"位置模板就能拿到大部分收益"——这也是任务书点名的"防伪条件化对照"。
**A8/A9 的评测纪律**：只换 Actor 的额外条件特征，`Z`、起点、`history(decoder)`、奖励、参数全部不变（我 `_patch_inputs` 已经是这个语义，扩一下即可）。

---

## §7 里程碑与命令

### M0 静态审计（本文 + 一个落盘文件）
- 交付 `docs/LEGACY_BPTT_CRITIC_AUDIT.md`：把 §1.1 的表格 + 证据行号落盘。
- **本文已完成 §1 的审计**；`v_ev` 与 `critic_grad_through_state` 是本次新查出来的，不是猜的。
- **M0 判定**：`critic_grad_through_state=false` ⇒ **旧 critic 对 Actor 零贡献（FOUND）**；`critic_ckpt=None` ⇒ **旧 critic 权重不可复用（NOT_FOUND）**。

### M1 数据/模型接口 + 单测（不需要 GPU 大算力）
```bash
source scripts/harness03_env.sh
python3 -m pytest tests/test_temporal_v5_grad.py tests/test_value_td_lambda.py \
    tests/test_actor_tail_value_grad.py tests/test_history_and_token_index.py \
    tests/test_physical_action_trust.py -q
# FD：8 个真窗口，h=1e-2/3e-2，对比 ∂J/∂Δ_t 与中心差分（我 exp0 的框架已可用）
python3 scripts/diag_bptt_learnability.py exp0 --out $OUT --n 8 --fd-at grad
```

### M2 复现 + 主训练
```bash
OUT=$COMIMIC_MAIN/runs/temporal_ac_bptt_v5
# ① 先复现旧 population 基线（A1/A3/A4）——协议必须与 08:10 报告一致
python3 scripts/diag_bptt_learnability.py arms --out $OUT --windows $BIG --vla-tokens $BTOK \
    --sizes 3072 --resample --bs 256 --steps 250 --lrs 3e-3 --arms constant mlp mlp_frame_flat
# ② 位置模板 A2（必做）
python3 scripts/diag_bptt_learnability.py arms ... --arms pos_template
# ③ v5 主训（A6/A7 并行，3 seeds）
python3 scripts/train_temporal_ac_bptt_v5.py --out $OUT --arm A6 --seed 0 --updates 1000 \
    --pool 10000 --batch 256 --beta-gate on --lambda-action 0.010 --tau-action 0.020
python3 scripts/train_temporal_ac_bptt_v5.py --out $OUT --arm A7 --seed 0 --updates 1000 --beta-gate off
```

### M3 真 Isaac 配对（会话 04 的门）
- 24–64 个独立 clip 窗口，同起点/同 chunk/同 seed，比较 `J_Isaac(Δ)-J_Isaac(Base)`；
- 同时报告 29D q_target 增量 RMS/peak、jerk、跌倒/接触/超时；
- clip 级 bootstrap 95% CI；`INCONCLUSIVE` 也要写。

### M4 集成/回滚
- 新 v5 权重只走 feature flag；默认仍 Base；
- 保留 `disable_delta()` / `load_previous_checkpoint()` / `strict_no_history_fallback`。

---

## §8 测试清单（任务书 §12）

| 测试文件 | 必须断言 |
|---|---|
| `tests/test_temporal_v5_grad.py` | `[B,T,64]+[B,64]+[B,10,58] → [B,T,64]`；attention/GRU/MLP **每层**都有非零有限梯度；`mask_t=0` 的位置梯度为 0 |
| `tests/test_value_td_lambda.py` | TDλ 递归、真终止 vs timeout、EMA 保留系数语义、target detach |
| `tests/test_actor_tail_value_grad.py` | Actor 更新时 Critic **参数不变**但 `dJ_ac/dy_H ≠ 0`；`β=0` 时 tail 项无梯度 |
| `tests/test_history_and_token_index.py` | `ℓ(k)=k`；`K=min(H,T)`；hist 的 q/qd 与 `state64` 同序（逐位 `max|Δ|=0`）；HA 与 STANDARD 的 `reorder_joints` 往返 |
| `tests/test_physical_action_trust.py` | `q_target` 单位是 rad；同 obs 下 Base vs Corr 的差 = `scale*(a_corr−a_base)`；peak/smooth 与手算一致 |
| `tests/test_vla_real_chunk_manifest.py` | manifest 的 `flow_seed/vlm_ckpt_sha/windows_sha` 可复现；禁止 `port_tokens` 冒充 |
| `tests/test_fail_closed_v5.py` | 缺 hist / 关节序不符 / T 不符 / arch 不符 **全部抛异常**，不静默降级 |

---

## §9 风险与阻断条件（fail-closed）

| 风险 | 检测点 | 阻断动作 |
|---|---|---|
| 假 history | 请求无 `state_hist_10` | `FAIL`：只允许 `no_history` 模型 |
| 旧 critic 不可用却被当成可用 | `critic_ckpt=None`（已确认） | 只能新建 critic；**禁止**写"复用旧 critic" |
| Value 目标不可定义（无法构造合法未来） | `K>H` 且无合法 replan | `BOOTSTRAP_NOT_IDENTIFIABLE`，`β=0` |
| WM reward hack | WM `ΔJ`↑ 而 Isaac 不涨 / 动作 RMS 越界 | 暂停，标 `WORLD_MODEL_EXPLOITATION_RISK` |
| 学成常量 | 打不过 A2(`c_t`) 或 shuffle 不掉分 | 判 `FAIL` |
| 未执行 token 被改 | 部署 `t≥K` 有非零 Δ | 直接 `FAIL`（契约违反） |
| 梯度非有限 | `assert_finite_nonzero_grad` | 阻断该 run，记录 seed/clip |

---

## §10 每个 run 的最低交付物

```
runs/temporal_ac_bptt_v5/<arm>/<seed>/
  config.yaml  git_revision.txt  split_manifest.json  vla_token_manifest.json
  history_contract.json  critic_legacy_audit.md
  train.jsonl  value_train.jsonl  val.jsonl  test_once.json
  grad_fd.json  critic_gradient_check.json  wm_isaac_paired.csv
  per_window_returns.csv  grad_stats.csv  learning_curves.csv
  checkpoint_actor_best.pt  checkpoint_critic_best.pt  checkpoint_target_critic.pt
  reproduce.sh  final_report.md
```

---

## §11 与其它会话的边界

| 文件 | 归属 | 我怎么处理 |
|---|---|---|
| `diff_vla_wbc/v3_step.py` | 会话 02 | **不修改**。v5 用独立训练脚本调用 `rollout.imagine_with_tokens` |
| `scripts/ha_vla_latent64_server.py` | 会话 01 | **不修改**。只在 §3.5 写"必须满足的合同"，由你指定谁改 |
| `scripts/probe_ft_landing.py` | 会话 01（正在改） | **不修改**。物理 rad 判据复用它的口径说明 |
| `diff_vla_wbc/delta_token_io.py` | 会话 01 新增 | **不修改**。v5 另开 `delta_token_io_v5.py` |
| 我自己的 `scripts/diag_bptt_learnability.py` | 会话 03 | 扩展（新增 `pos_template` / `shuffle_hist` / `shuffle_chunk` / `lambda_action`） |
| `runs/bptt1/**`、`runs/harness_0{1,2}_*` | 历史/他人 | **只读**，绝不覆盖 |

---

## §12 立刻可执行的下一步（等你点头）

1. **A2 位置模板**（1 条臂，25 分钟）—— 没有它，A6 的"条件化"不成立；
2. **v5 Actor + 双 optimizer + β gate 的最小实现**（M1，1.5 小时含单测）；
3. **把 bptt1 完整配方（含它的 window_norm/act hinge/grad_target，无 cap）在同一个池/评测下复刻**（1 条臂，25 分钟）
   —— 用来确认 §1.1 的"+1.12% ≈ 常向量水平"这个判断，并把"哪一项让它在 59 步内只拿到 1.1%"钉死；
4. 之后再开 10k 池 + 1000 updates 的正式训练。

> **不声称已成功**：v5 的收益目前**只有间接支持**（A4 的 test +0.0347、Oracle 上界 +0.0730），
> Critic 的尾部价值在当前代码里**从未真正参与过 Actor 梯度**（§1.1 已证），所以 v5 的 Critic 部分是**首次实现**，不是"修复"。
