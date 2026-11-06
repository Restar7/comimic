# Token 空间的 FSQ 网格量化 + STE（唯一口径）

> 2026-11-04。回答一个问题：**"反传穿过执行器"这一步，token 要不要先量化？**
> 代码：`diff_vla_wbc/quant.py`（网格/STE/统计）、`scripts/check_token_jacobian.py`（前置判据）。
> 开关：`--token-quant {none,ste,hard}`（launcher 环境变量 `TOKEN_QUANT`）。

---

## 0. 一句话

**SONIC 的 token 是 `1/16` 的整数格，执行器只在格点上训练过。** 所以"VLA 输出连续 token
→ 冻结执行器"这一步的 `∂a/∂token` 是**插值外推**。要不要把它掰回格点，取决于一个可测的数：
**解析 Jacobian 与"跨一格真实差分"的偏差**（见 §6）。测完再选，别凭感觉。

当前实测（WBC `exp_WBC_exec_haonly_best.pt`，8 帧 × 16 维，同 clip#0 同帧配对）：

| δ（格距倍数） | cos | rel | gain | 结论 |
|---:|---:|---:|---:|---|
| 0.01 | 1.0000 | 0.0016 | 1.5834 | 解析 Jacobian **正确** |
| 0.25 | 1.0000 | 0.0042 | 1.5839 | |
| 0.5（最坏亚格偏移） | **1.0000** | **0.0083** | 1.5844 | **可接受** |
| 1.0 | 0.9998 | 0.0166 | 1.5853 | 跨一格 |

`rel` 随 δ 线性翻倍（0.0042 → 0.0083 → 0.0166），是**曲率**的典型签名
（`rel ≈ |f''|δ/2|f'|`），不是外推失效。⇒ **这个执行器（SuperMimic WBC）在
`token_quant=none` 下可以接受**（亚格偏移最坏情况只带来 0.8% 的动作方向误差）。

同一把尺子量 **SONIC release decoder**（`model_decoder.onnx` → 逐算子复刻的可微 torch 版，
994→2048→2048→1024→1024→512→512→29，10.18M 参数；输入直接取 HA raw 的 `decoder_obs`，
口径自检残差 **5e-6 度**）：

| δ（格距倍数） | cos | rel | gain |
|---:|---:|---:|---:|
| 0.01 | 1.0000 | 0.0029 | 1.3631 |
| 0.25 | 1.0000 | 0.0025 | 1.3635 |
| **0.5** | **1.0000** | **0.0049** | 1.3638 |
| 1.0 | 1.0000 | 0.0098 | 1.3644 |

⇒ **两个执行器都 OK**，SONIC decoder 甚至比 WBC 更线性（0.5% vs 0.8%）。
JSON：`runs/token_jacobian/sonic_release.json`。

注意这是**一个 clip / 一个 npz** 的结果；正式门禁要 `--frames 8 --dims 64` 跑多个 clip。
JSON：`runs/token_jacobian/wbc_haonly_best.json`。

---

## 1. 三条事实（决定了后面全部结论）

| # | 事实 | 证据 |
|---|---|---|
| ① | token 是 `q = n/16` 的整数格，`n ∈ [-16,15]`，值域 `[-1.0, 0.9375]`，每维 32 级 | release encoder 导出图：`q = round(tanh(h+0.032237)*15.515501 − 0.5)/16`，复刻见 `sonic_encoder.py:151` |
| ② | 执行器**只在格点上训练过**：HA 标签 `g1_motion_token_64` 是 FSQ 输出；A' 采集喂的 token 库同样是 FSQ 输出 | `docs/SONIC_DECODER_FOUND.md`、`sonic_frontend.py` |
| ③ | VLA 输出是**连续值**：`codec.decode` 是纯仿射（**不 clamp、不量化**） | `action_codec.py:47` |

于是 `∂a/∂token` 在格点之间**没有数据约束**——这是本文件唯一要解决的问题。

---

## 2. 三种模式（必须显式选）

| `--token-quant` | 前向喂什么 | 反向 | 用途 |
|---|---|---|---|
| `none` | 连续 token | 正常 | **默认**。执行器可重训 + dither 时用它；Jacobian 校验 OK 时也用它 |
| `ste` | **格点**（执行器训练分布内） | **恒等**（`∂τ_use/∂τ ≡ 1`） | 执行器**不可重训**时（如 release SONIC decoder）；或 Jacobian 校验 FAIL 时 |
| `hard` | 格点 | **零** | 纯消融：应当"训不动"。若它也能降 loss ⇒ 梯度从 trust/BC 项漏进来，该轮结论不可用 |

实现在 `quant.py`：

```python
τ_q   = fsq_grid(τ)                 # round(τ*16).clamp(-16,15)/16 ，不带梯度
τ_use = τ + (τ_q - τ).detach()      # forward: τ_q   backward: 1
```

`∂τ_use/∂τ = 1 + ∂[常数]/∂τ = 1` —— **严格恒等**，不是近似。所以
`reward → WM → 执行器 → τ_use → τ → decode → DiT×4 → head` 这条链**一个断点都没有**；
唯一变化的是"执行器在哪个点求 Jacobian"（从格点之间挪到格点上，而那是它训练过的点）。

---

## 3. 必须避开的写错法

FSQ 公式里的 `tanh` 作用在 **encoder 内部的连续 embedding `h`** 上，**不是 token**。
VLA 输出的 token 已经在 `q` 空间里，所以量化只能是**绝对网格**：

```python
q = clamp(round(τ * 16), -16, 15) / 16          # ✅
q = round(tanh(τ + 0.0322) * 15.5155 - 0.5)/16  # ❌ tanh 会把 [-1,0.9375] 压成 [-0.77,0.73]
```

`quant.py::fsq_grid` 已经写死成前者，并且 `tests/test_quant.py` 用逐点已知值钉住了它
（`-1.0→-1.0`、`0.031→0`、`0.0375→0.0625`、`0.9→0.875`、`1.2→0.9375`）。

---

## 4. 三处必须一致（混搭会静默错）

| | A' 采集（真实仿真） | joint 训练（想象） | 部署/闭环评测 |
|---|---|---|---|
| `token_quant=none` | 格点（来自 token 库） | **格点之间** | **格点之间** |
| `token_quant=ste` | 格点 | 格点 | **必须也量化** |

**规则：训练的前向和部署必须一致，只有反向可以骗。**

| 训练 | 部署 | 后果 |
|---|---|---|
| 连续 | 连续 | ⚠️ 前向永远在分布外，靠外推硬撑（要 §6 的 `rel` 小才成立） |
| `ste`（前向格点） | **连续** | ❌ 训练时执行器看到格点、部署时看到中间值 ⇒ 学到的补偿在部署时失效 |
| `ste`（前向格点） | 量化 | ✅ 自洽，且与数据生成系统 bit-exact |

**部署侧已经接好了**（2026-11-04）：

- 服务端（**两个都接好了**）：
  `scripts/ha_vla_latent64_server.py`（主）与 `scripts/ha_vla_server.py`（A1 arena 的旧 server）
  都支持 `--token-quant {none,ste,hard}`（也认环境变量 `TOKEN_QUANT`，所以 15 个 launcher 不用逐个改）；
  两者的口径校验共用 `diff_vla_wbc/quant.py::check_meta_mode` / `resolve_deploy_mode`，不会漂移；
- 量化位置在 **lead 外推之后**（`tok + β(tok − tok₋₁)` 一定造出格点之间的值，
  只在采样后量化等于没量化；旧 server 没有 lead，直接在出口量化）；
- 口径校验：读 `vla_meta.json` 的 `token_quant`，与 `--token-quant` **逐项比对**，
  不一致直接 `SystemExit`（放行开关 `ALLOW_TOKEN_QUANT_MISMATCH=1`，和 `fm_steps` 同构）；
- `vla_meta.json` 由 `scripts/export_vla_ckpt.py` 从 `vla_*.pt` 的 `v2.token_quant` 带出；
- 推理没有梯度，所以 `ste` 与 `hard` 的**前向完全一样**（都投射到格点）；两档都接受只是
  为了和训练侧同名。

launcher：

```bash
TOKEN_QUANT=ste HEAD_CKPT=runs/exp_ref_r1/vla_000400.pt bash scripts/start_vla_server.sh
# 服务端 GET / 会回报 {"token_quant": "ste", "token_quant_pre_dist": 0.0123}
```

---

## 5. 指标怎么读

每次采样都会记（`none` 模式不记）：

| 键 | 含义 | 健康值 |
|---|---|---|
| `quant/dist_mean` / `dist_p90` / `dist_max` | VLA 输出到最近格点的距离 | 值域内 ≤ 0.03125；长期贴 0.03125 = 在格点边界磨 |
| `quant/near_edge` | 距格点 > 1/4 格的元素占比 | 看**趋势**，不设硬阈值 |
| `quant/out_range` | 落在 `[-1, 0.9375]` 之外的占比 | 越界会被夹住 ⇒ `dist_max` 会 > 半格 |
| `quant/cell` | 格距常数（0.0625） | 常数 |

**`dist_mean` 同时是 STE 偏差的度量**（偏差 ≈ 格距 × 局部曲率）。若它持续 > 0.05，
说明 reward 在把 token 推离格点，STE 的近似在变差 → 要么加软正则，要么检查 §6。

另注：`token_quant=ste` 会让有效学习率"被量化"（单步更新 ≪ 半格时前向不动），
loss 曲线呈阶梯是**预期现象**，不是断图。判据：`grad_norm` 正常但 loss 不动 ⇒ 先怀疑这个。

---

## 6. 前置判据：Jacobian 有限差分校验（**先做这个**）

```bash
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=$PWD/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody-dex3/g1_29dof_with_dex3_rev_1_0.usd

python3 scripts/check_token_jacobian.py \
  --ckpt runs/S1_handover/exp_WBC_exec_haonly_best.pt \
  --packed runs/isaac_motion/ha_cmd_exec/packed.pt \
  --frames 8 --dims 64 --out runs/token_jacobian/wbc_haonly_best.json
```

它量的是 `∂a/∂τ_j · δ` ≟ `a(τ + δe_j) − a(τ)`，**同一个 clip 内同帧配对**（token 行号按
`frame − clip_start` 对齐；第一版脚本 token/obs 不配对，量到的其实是 OOD 饱和，已修）。

判据：

| 条件 | 结论 |
|---|---|
| `cos(δ=0.01) < 0.99` | **BROKEN**：不是量化问题，是 autograd/配对/饱和有问题，先修 |
| `cos(δ=0.5) ≥ 0.99` 且 `rel ≤ 0.10` | `token_quant=none` 可接受 |
| `cos(δ=0.5) ≥ 0.95` 且 `rel ≤ 0.30` | 建议 `token_quant=ste` |
| 其它 | `ste` + 对执行器 dither 重训 |

**这个数比 G3a（WM 20 步误差）更靠前**：如果这里就失真，后面所有 reward 形状、δ 课程、
WM refresh 都是在优化一个错误的梯度。

---

## 7. 如果执行器**不可重训**（例如 release SONIC decoder）

结论会反转：**这时应该量化（`ste`）而不是连续喂。**

理由：HA 数据本身就是 `motion → encoder → FSQ → decoder` 跑出来的，**量化是还原，不是妥协**。
不能重训就没有"把训练分布扩到格点之间"（dither）这条路，只剩"把输入掰回格点"这一条：

- 前向量化 ⇒ 执行器精确落在训练分布上；
- 反向 STE ⇒ 梯度仍然到得了 VLA；
- 部署也量化 ⇒ 与数据生成系统一致。

代价：优化的是"连续松弛"的梯度（真实函数的导数 a.e. 为 0），所以 **`rel` 的测量仍然必要**
（它告诉我们这个松弛有多可信）。

---

## 8. 消融怎么跑

```bash
# 基准（当执行器 Jacobian 校验 OK，或执行器可 dither 重训）
TOKEN_QUANT=none bash scripts/launch_joint_ha_v2.sh

# STE 臂（执行器不可重训 / 校验 FAIL）
TOKEN_QUANT=ste  bash scripts/launch_joint_ha_v2.sh

# hard 臂：判据是"训不动"。它若降 loss ⇒ 梯度从 trust/BC 项漏进来，该轮结论作废
TOKEN_QUANT=hard bash scripts/launch_joint_ha_v2.sh
```

日志里看 `[launch-v2] token_quant=...`、`metrics['quant/*']`、`summary.json.token_quant`。

---

## 9. 未决项

- [x] **部署侧量化**：`ha_vla_latent64_server.py` + `start_vla_server.sh` + `export_vla_ckpt.py` 已接。
- [x] `scripts/ha_vla_server.py`（旧 server，A1 arena 用）已补 `--token-quant`（与主 server 共用口径校验）。
- [x] **SONIC release decoder 的 Jacobian 已测**（`diff_vla_wbc/sonic_decoder.py` + `check_token_jacobian.py --sonic-decoder`），结果 OK。
- [x] release SONIC decoder 的 Jacobian 已测（ONNX→可微 torch 在 `diff_vla_wbc/sonic_decoder.py`；994 维输入直接用 HA raw 的 `decoder_obs`，不需要重建布局）。
- [ ] `dither` 重训执行器的对照臂（`token + U(−0.031, 0.031)`）尚未做；它能让 `token_quant=none` 真正成立。
- [ ] `token_quant=ste` 下的 `quant/dist_mean` 长期漂移阈值还没定，先按"看到持续 >0.05 就报警"跑。
