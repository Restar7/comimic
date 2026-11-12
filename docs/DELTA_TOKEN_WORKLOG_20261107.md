# Δtoken 教师蒸馏：全部改进 · 修改 · 结果 · 问题（2026-11-07 工作总账）

> 这份文档把**这次全部工作**按"改了什么 / 测出什么 / 踩了什么坑 / 最后结论"整理成一份。
> 每个数字都标了脚本或数据路径；凡未测的写"未测"。所有结论都可以按 §7 的命令复现。

---

## 0. 一句话

冻结 VLA 出 64 维 token，我们在 token 上加小残差 Δ（"Δtoken"），Δ 由**可微世界模型反传得到的教师标签**监督。
本次把这条链从"到处是口径错配"梳理到"**同一协议、同一样本量下的可比结论**"：
**教师弱正（0.551，n=256）、学生无信号（0.449，n=256）；所有"调幅度/调视界/调损失/加信息"的修法都被同一协议否掉。**

---

## 1. 系统与口径

### 1.1 部署链路（推理**不含**世界模型）

```
相机 256² + state46 → GR00T N1.7（Cosmos-Reason2-2B frozen + 1.09B DiT 微调）
   → 64 维 token chunk (T=40 @ 50 Hz = 0.8 s) → token' = token + Δ（唯一被训练的模块）
   → SONIC decoder（ONNX/CPU）→ 29 关节目标 → Isaac/真机
```

### 1.2 关键参数

| 项 | 值 |
|---|---|
| DiT | 32 层，inner 1536，output 1024，**1,091.17 M**；`tune_diffusion_model=true`（action head 全参微调）、`tune_llm/visual=false` |
| LoRA 换算 | rank32 全线性 34.1 M / rank16 17.0 M / rank8 仅 attention 3.4 M |
| token / chunk | 64 维/帧，T=40，50 Hz（H=20 步 = 0.4 s） |
| 标签 | `runs/labels_v3`：300 dump × 128 = **38,400** 条；`g_tok_pert` = `∂J₂₀/∂token_t`（γ=0.99，discounted，K=H=20） |
| 蒸馏口径 | `Y = unit(g)·η`；η 已从 0.15 改为 **0.06 = 0.75·cap**；cap = **0.08**（逐帧 L2 soft-cap） |
| 成本 | H=20 梯度 1869 ms / H=2 231 ms；闭环 3 任务 × 6 seed × 3000 步 ≈ 3 h |

### 1.3 ⚠️ 三个**不同**的世界模型（最容易踩的坑）

| 文件 | 角色 | sha256(16) |
|---|---|---|
| **`runs/a_prime_sonic_exc/wm_gain2.pt`** | **生成 38,400 标签（A′ 线）** | `c74060f1d3706f63` |
| `runs/S1_handover/wm_a24_k20.pt` | 历史真值探针（98.4%） | `419c34f007cb8cd5` |
| `WBC_A24000_wmvel_cmd.pt` 内 wm | bundle 默认，**不是标签口径** | `52267bf1357c58ad` |

标签口径实锤：`runs/lab3_g0/args.json` 的 `"wm_ckpt": "runs/a_prime_sonic_exc/wm_gain2.pt"`。

---

## 2. 本次全部代码修改

| # | 文件 | 改了什么 | 为什么 | 影响 |
|---|---|---|---|---|
| 1 | `scripts/train_distill_v3.py` | 新增 `--mask-frames`（默认 20），损失/cos 只在有效帧上算，并打印旧口径 | 逐帧标签在 t≥20 **精确为 0**，旧口径把 cos 稀释 2× | 冒烟：新 0.412 / 旧 0.206（正好 2×） |
| 2 | `scripts/pretrain_v5_supervised.py` | `--eta` 默认 **0.15 → 0.06**；`η > 0.9·cap` 时告警 | soft-cap 是 `‖Δ‖=cap·tanh(‖u‖/cap) < cap`，目标 = cap 不可达；0.15 > 0.08 会被**结构性压扁 1.875×** | 学生第一次能表达目标幅度 |
| 3 | `scripts/opt_teacher_wm.py` | **新建**：WM 内约束轨迹优化教师（单窗口 U，(40,64)） | 验证"摊销轨迹优化"假设 | 见 §3.2，**结论：不优于单次梯度** |
| 4 | 同上 | 教师限幅 `0.08·tanh(u)`（逐元素，最大 **0.64**）→ **`apply_norm_cap`**（逐帧 L2，最大 0.08） | 教师曾比学生大 **8×**，在学生表达能力之外优化 | 契约统一 |
| 5 | 同上 | 修 **3 个 NaN 源**：①`u·(c/‖u‖)` 在 0 处导数 8e10 ②`sqrt(0)`×relu 零梯度 = 0·∞ ③WM 状态通道 inf/nan | 否则第一次 Adam 步就回退到零修正 | 优化器可用 |
| 6 | 同上 | 关节限位改"动作空间信任域 0.03 + 状态可行性（骨盆高度/R22）" | 实测动作**不是**关节偏置（dump 的 `prev_action` RMS 1.38 vs 参考 0.24） | 约束层可正确施加 |
| 7 | `scripts/make_teacher_isaac_job.py` | **新建**：教师/学生 Δ → Isaac 可执行两臂 job；支持 `--wm-ckpt`/`--teacher-mode grad`/`--alpha`/`--reward-steps` | 真机验收 | 见 §3.2 |
| 8 | `scripts/scan_delta_amplitude.py` | **新建**：固定教师方向、扫 α 的幅度曲线 | 查"幅度是否过大" | §3.2 |
| 9 | `scripts/scan_student_amplitude.py` | **新建**：**学生**的 α 扫描 + 一阶方向量 `D_i = Σ⟨∇J,Δ_θ⟩` + Isaac job | 教师方向不能替学生说话 | §3.2 |
| 10 | `scripts/train_dir_mag.py` | **新建**：方向头（unit）+ 独立幅度门控（≤0.75·cap），纯 cos 损失 | 情况 A 的修复尝试 | §3.3，**cos 无改善** |
| 11 | `scripts/analyze_label_horizon.py` | **新建**：H 曲线（cos/比值/尾巴/J 份额） | 选视界 | §3.4 |
| 12 | `scripts/probe_label_learnability.py` | **新建**：统一 mask 的 k1 vs k20 可学性 | 查旧对比是否口径不一致 | §3.3 |
| 13 | `scripts/probe_label_generalization.py` | **新建**：留族 vs 留运动 | 查"任务没覆盖"假设 | §3.3 |
| 14 | `scripts/probe_vision_information.py` | **新建**：视觉特征增量（含打乱对照） | 查 LoRA/视觉是否有额外信息 | §3.5 |
| 15 | `scripts/eval_body_drift_vs_ref.py` | **新建**：root + 14 body 相对示教参考的漂移 | 查闭环误差结构 | §3.6 |

---

## 3. 全部实测结果

### 3.1 标签本身

| 项 | 值 |
|---|---|
| 数量 / 覆盖 | 38,400 条；6 族全有（double_desk 23.8% / vision_navi 19.1% / pp_box 17.0% / open_door 14.9% / boxing 13.0% / sit_sofa 12.1%）；533/699 运动 |
| 零标签 | **50.0%**，全部来自 t≥20（J₂₀ 影响不到更远 token） |
| 幅值分布 | `‖g‖` p50 0.0009 / p90 0.133 / p99 0.51 / **max 9.73**；`‖token‖` p50 1.30 |
| 留出口径 | `frame` = **运动库行号**；"按 frame 留前 2 段" = **留出整个 HOI_double_desk 族**（84 条运动） |

### 3.2 真机（Isaac）验收 —— 全部同一协议：32/256 窗口、20 步、同一 reward

| 臂 | n | **真物理变好比例** | 符号一致率 | ΔJ 相关 |
|---|---|---|---|---|
| 教师 H=20，δ=0.08 | 32 | 0.688 | 0.688 | +0.472 |
| **教师 H=20，δ=0.08** | **256** | **0.551** | **0.617** | **+0.390** |
| 教师 H=2，δ=0.08 | 256 | **0.512** | — | — |
| WM 内 20 步优化教师 | 32 | 0.656 | 0.656 | +0.525 |
| 教师 δ 扫描 α=0.125/0.25/0.5/1 | 32 | 0.531 / 0.531 / 0.562 / **0.688** | — | — |
| **学生 α=0.25** | **256** | **0.449** | — | — |
| 学生 α=0.25 / 0.5 / 1 | 32 | 0.625 / 0.625 / 0.500 | — | — |

**读法**：① 教师只有**弱正**（0.551，CI≈±0.061）；② **n=32 的 0.688 是小样本乐观偏差**；
③ 短视界教师更差（0.512）；④ 多步优化教师不优于单次梯度；⑤ 教师幅度**越大越好**（0.531→0.688，n=32）；
⑥ 学生（任何幅度、n=256）**无信号**。

### 3.3 学生的可学性（四种参数化，同一套留出、只算有效帧）

| 方案 | 输出 | 损失 | held-out **cos** |
|---|---|---|---|
| ridge（3554 维特征） | — | — | 0.376 |
| v5 头（现役） | 逐帧 L2 soft-cap | masked MSE on `η·unit(g)`，η=0.15 | 0.397 |
| flat MLP（v3） | 无 cap | 同上 | 0.412 |
| **方向 + 幅度门控** | `m·unit(u)`，m≤0.06 | **纯 cos** | **0.407** |

* 逐帧（20 步）0.313 / 共享（64）0.411 / 第 0 帧（1 步）0.508 / 动作（1 步）0.505。
* 留出**整族** 0.301 vs 留出**运动**（族见过）0.341 ⇒ **任务覆盖只值 0.04**。
* 学生一阶方向量 `D>0 = 0.50` ⇒ 学生的修正与 WM 梯度在"一阶收益"上完全抵消。

### 3.4 视界 H 曲线（512 窗口，窗口级梯度，K=20）

| H | cos(g_H,g₂₀) | ‖g_H‖/‖g₂₀‖ | 尾巴（critic 需补） | J 份额 |
|---|---|---|---|---|
| 2 | +0.355 | 0.038 | 99% | 0.111 |
| 5 | +0.665 | 0.106 | 93% | 0.298 |
| 8 | +0.803 | 0.208 | 84% | 0.467 |
| 10 | +0.862 | 0.298 | 76% | 0.574 |
| 15 | +0.965 | 0.607 | 44% | 0.808 |
| 20 | 1.000 | 1.000 | 0% | 1.000 |

⇒ 降 H 只有 H=15 勉强可接受，但它与 H=20 已 cos 0.965（**换了也换不到可学性**）。

### 3.5 视觉 / 参考（隐变量排查）

| 试验 | 结果 |
|---|---|
| 视觉 `backbone_features(2048)` 加入（PCA 16/32/64 + **打乱对照**） | +0.005 / +0.007 / +0.007，**与打乱对照完全重叠** ⇒ 无额外信息 |
| `ref0`（神谕参考动作） | token +0.04、动作 +0.07 ⇒ 参考不是缺失的关键隐变量 |

### 3.6 闭环 A/B 与漂移（36 集 × 3000 步，带视频与逐步 dump）

| 指标 | base | delta（S 臂，cos 0.459） | 统计 |
|---|---|---|---|
| 成功率 | **11/18 = 61.1%** | **9/18 = 50.0%** | McNemar p=0.625；配对差 CI [−11,+33] 点 |
| root 漂移 p50 | 0.48 / 0.44 / 1.02 m | 0.48 / 0.32 / 1.33 m | 配对中位 +0.029 m，p=0.41 |
| body 漂移 p50 | 0.483 / 0.427 / 1.015 m | 0.492 / 0.317 / 1.315 m | 配对中位 +0.034 m，p=0.24 |
| **站桩占比** | 0.24 / 0.30 / 0.24 | 0.34 / 0.40 / 0.31 | **13/18 变高，p=0.048** |

结构发现：**漂移 = 整体平移/航向错，不是姿势错**（14 个 body 的 p50 全在 0.62–0.68 m，`body−root` max 0.037 m）；
**横向 > 纵向**（cross 0.28–0.63 m > along 0.17–0.55 m）；漂移随时间单调累积（0.22 → 1.23 m）。

### 3.7 部署探针（`probe_ft_landing.py`）

| 臂 | ‖Δa‖ | /η | 效应/噪声 |
|---|---|---|---|
| 在线蒸馏 | 0.0450 | 0.30 | 0.72 |
| 离线 η=0.08 | 0.0518 | 0.35 | 1.35 |
| 离线 η=0.15 | 0.0905 | 0.60 | 1.39 |

---

## 4. 所有问题（bug / 口径 / 误判）

### 4.1 真 bug（已修）

| bug | 现象 | 修法 |
|---|---|---|
| `train_distill_v3.py` 逐帧路径缺 mask | 一半输出维度在学"输出 0"；cos 被稀释 2× | `--mask-frames`（默认 20）+ 打印旧口径 |
| η(0.15) > cap(0.08) | 学生被结构性压扁 1.875×，MSE 永远到不了 0；**cos 尺度不变 ⇒ 完全看不出来** | η → 0.75·cap = 0.06 + 告警 |
| 教师限幅逐元素 tanh | 教师最大 ‖Δ‖ 0.64 = 学生 8×，优化到学生表达不了的集合 | 统一走 `apply_norm_cap` |
| `cap` 在 ‖u‖=0 处导数 8e10 | 优化器第一步梯度爆掉、回退到零 | `cap·tanh(u)` 参数化 |
| `sqrt(0)` × relu 零梯度 | 0·∞ = NaN，第一次 Adam 步即废 | 用**均方**比较 |
| WM 状态通道 inf/nan | 污染整条 loss | `nan_to_num` + 有界替换 |
| MJCF 无限制关节 `jnt_range=(0,0)` | 把**未扰动基线**判成越界 0.31 rad/关节 | 真限位掩码 |
| Isaac job 的 `frames` numpy/转置 | `AttributeError` / shape 不匹配 | 统一转 tensor + `(N,H)` |

### 4.2 口径问题（会直接改变结论）

1. **`frame` 是运动库行号**，不是 episode 帧号 ⇒ "按 frame 分 10 段留出"其实是**留出一个完整任务族**（double_desk）。
2. **`g1` 是 1 步奖励梯度**（`∂r₀/∂a`），不是 20 步 ⇒ 动作空间 0.505 不是"空间优势"。
3. **`tokenwise_reward_grad().g` 是 (B,64)**（对 T 求和），不是逐帧。
4. **三个不同的 WM**（见 §1.3）：标签用 `wm_gain2`，历史 98.4% 用 `wm_a24_k20`，bundle 默认是第三个。
5. **动作不是关节偏置**：dump 的 `prev_action` RMS 1.38 vs 参考推出 0.24（差 5.7×）。
6. `token_chunk(2400)` 是笔误，应为 **2560 = 40×64**。

### 4.3 我的误判与更正（写清楚，避免再被引用）

| 曾说过 | 更正 |
|---|---|
| "H=20 方向只有 +0.092 相关" | 那是 **ΔJ 幅值**相关；符号一致率 **0.617（n=256）/0.984（历史 WM）** |
| "动作空间 0.505 是空间优势" | 它是 **1 步**标签；同视界下 token 1 步是 0.508 |
| "5 臂数字被稀释 2×，作废" | `train_distill_temporal.py` / `pretrain_v5_supervised.py` **本来就有 mask**，没作废 |
| "标签用 bundle 默认 WM，那个 WM 的梯度是噪声" | 标签用 `wm_gain2`；当时测错了 WM |
| "部署幅度是验证尺度的 10–23 倍" | 把两套 WM/口径拼在一起，**不成立**，需同批重测 |
| "缩小教师幅度会更安全" | 教师方向上是**反的**（0.531→0.688 单调上升）；但对**学生**曾看似有效（n=32） |
| "学生 ×0.25 能到 0.625，这条路可救" | **n=256 掉到 0.449**，n=32 是噪声 |
| "可学性 0.41 = 观测信息不足" | 未定论：四种参数化都 0.41，更像"拟合能力上限"而非单一信息源缺失 |

---

## 5. 最终结论

1. **教师方向只有弱正信号**：H=20 的 **0.551**（n=256，CI≈±0.061）；H=2 的 **0.512**；n=32 的 0.688 是乐观偏差。
2. **学生学不到教师方向**：离线 cos 天花板 **0.41**（ridge 0.376 / v5 0.397 / flat MLP 0.412 / 方向+门控 0.407），
   一阶 `D>0 = 0.50`；真机 **0.449**（n=256）。
3. **所有修法都被同一协议否掉**：多步优化教师、缩小（教师/学生）幅度、方向+幅度门控、加视觉、加参考、降视界。
4. ⇒ **"教师标签蒸馏 Δtoken"在现有 WM 与标签下没有可证明的正收益。**
   **本项目核心贡献（可微 WM 把全身执行误差反传到 token 空间）没有被否**，
   被否的是"把它蒸馏成每步小 Δ"这一步用法。
5. 唯一未被否掉但**缺数据**的路径：**候选 best-of-N 筛选**（只需符号；需要"每状态多候选 + 真机回报"的新 dump）。

---

## 6. 未验证 / 未做

| 项 | 状态 |
|---|---|
| **在线 Gate B**（真实 token→SONIC→因果历史→PD/Isaac 闭环） | **未做**（本次是离线 SONIC 复刻 + 固定动作序列回放） |
| token 空间 ΔJ 配对（`J_isaac(+Δ)` vs `J_isaac(0)`，在线） | 未做 |
| 帧间视觉信息（运动方向/目标接近） | 未做（只测单帧 + 均值池化） |
| 大样本闭环 | 只有 n=18（分辨率 >25 个点）；建议 48 对起 |
| 部署状态 OOD（漂移 0.45–1.3 m 后学生是否还成立） | 从未量化 |
| best-of-N | 未做（缺数据形态） |
| 教师 n=256 在其他 WM（`wm_a24_k20`）上 | 未做 |

---

## 7. 复现清单

```bash
# 1) 真机验收（教师/学生/短视界），核心命令
python3 scripts/make_teacher_isaac_job.py --n 256 --teacher-mode grad --alpha 1.0 \
    --out runs/teacher256                       # H=20 教师
python3 scripts/make_teacher_isaac_job.py --n 256 --teacher-mode grad --reward-steps 2 \
    --out runs/teacher256_h2                    # H=2 教师
python3 scripts/scan_student_amplitude.py --n 256 --alphas 0.25 --out runs/student256
$ISAAC_VENV/bin/python -u scripts/isaac_probe_action_rollout.py \
    --job <job>.pt --out <roll>.pt --num-envs 256 --device cuda:0 --headless
python3 scripts/probe_wm_gradient_truth.py score \
    --ckpt runs/S1_handover/WBC_A24000_wmvel_cmd.pt --job <job>.pt --roll <roll>.pt

# 2) 可学性 / 泛化 / 视界 / 视觉
python3 scripts/probe_label_learnability.py --dir runs/labels_v3
python3 scripts/probe_label_generalization.py --mode family      # 或 motion
python3 scripts/analyze_label_horizon.py --hs 2 5 8 10 15 20 --batch 32
python3 scripts/probe_vision_information.py extract --n 2000 --out runs/vision_probe
python3 scripts/probe_vision_information.py probe --dir runs/vision_probe

# 3) 方向+幅度门控重训
python3 scripts/train_dir_mag.py

# 4) WM 内优化教师
python3 scripts/opt_teacher_wm.py --windows 128 --iters 0 5 10 20 --lr 0.05

# 5) 闭环与漂移
TASK=OpenDoor HEAD=none GPU=0 PORT=18241 SEEDS="0 1 2 3 4 5" MAX_STEPS=3000 \
  CELLDIR=runs/ab36/base/OpenDoor bash scripts/r3_sr_cell.sh
python3 scripts/eval_body_drift_vs_ref.py --cells base=runs/ab36/base delta=runs/ab36/delta
```

---

## 8. 文件与产物

**文档**
* `docs/REVIEW_HANDOFF_V2_20261107.md` —— 评审用（§1–10 + 附录 A.1–A.11，含更正/作废清单）
* `docs/LABEL_HORIZON_ANALYSIS_20261107.md` —— 视界 H 的完整分析
* `docs/VISION_PROBE_AND_MASK_FIX_20261107.md` —— 视觉探针 + mask 修复
* **本文件** —— 工作总账

**数据**
* `runs/labels_v3/`（38,400 标签）、`runs/teacher256/`、`runs/teacher256_h2/`、`runs/student256/`
* `runs/alpha_isaac/a{0.125,0.25,0.5,1.0}/`、`runs/student_isaac/`、`runs/teacher_isaac_gain2/`
* `runs/ab36/`（36 集闭环 + `body_drift.json`）、`runs/label_horizon.json`、`runs/label_learnability.json`、`runs/vision_probe/vision_probe.json`、`runs/alpha_scan.json`、`runs/dir_mag/model.pt`

**脚本**：§2 表中 15 个（其中 8 个为本次新建）
