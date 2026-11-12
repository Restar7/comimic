# P26-WM 重训计划：`wm_gain3`（gain_loss 保方向 + A'/闭环双数据 + 3000 步）

> 生成：2026-11-12 12:20 ｜ 状态：**待执行**
> 目标：训出一个**方向可信 + 绝对误差可接受**的 WM，用于训练动作残差。
> 判据：**dir_agree ≥ 0.90、corr_dJ ≥ 0.28**（方向），录制误差 ≤ 0.20、闭环误差 ≤ 0.30（精度）。

---

## 1. 为什么这样重训（代码证据，不是猜）

| 事实 | 来源 |
|---|---|
| `wm_gain2.pt` 是唯一带 **gain_loss** 训出来的 WM | `scripts/finetune_wm_gain.py`（`--pairs` 配对探针 + `gain_loss`） |
| wm_base6 = wm_gain2 + **350 步 ssm-only**，方向靠“短训低 lr”保住 | `comimic-wt03/docs/B_WM6_REVIEW_FOR_GPT_20261109.md`（“WM6 `8edb59cf` s350 替换旧 wm_gain2”） |
| refresh1/2/3 **全部没有 gain_loss**（grep 0 次） | `scripts/c_wm6/p21_refresh2_cl.py` |
| 方向随刷新单调退化 | dir_agree：0.979（base6）→ 0.885（r1）→ 0.771（r2）→ 0.750/0.635（r3） |

**结论：方向来自 `gain_loss`（同一状态 × K 个动作的配对监督），refresh 把它丢了。重训必须把它带回来。**

---

## 1.5 ★ 口径修正（2026-11-12 新发现，必须遵守）

**`xi` 不是中性输入**：`default_xi()` 里 `xi[0] = 0.75`（地面摩擦系数），其余为零（编码器偏置/质心偏移）。
数据是在摩擦 0.75 下采的，而 **refresh1/2/3 全部用 `torch.zeros(...)` 训练和评测**（= 摩擦 0 的冰面），
导致它们的“绝对误差改善”是在错误口径下自证的（换回 0.75 后 refresh2 的误差从 0.16 变 1.65，差 10 倍）。

| 环节 | 之前 | 本计划 |
|---|---|---|
| 训练 | refresh: `zeros` ❌ | **`b.default_xi()`** ✅（`finetune_wm_gain.py` 本来就是这样） |
| 绝对误差评测 | refresh gate: `zeros` ❌ | **`default_xi` 重写评测** |
| dir_agree 探针 | `default_xi` ✅ | 保持 |
| C 线训练 | `default_xi`（eval_set xi=0.75）✅ | 保持 |

**所有对比必须在 `default_xi` 下进行。**

---

## 2. 必须恢复的三样东西（refresh 线丢掉的）

| # | 丢了什么 | refresh 的做法 | 本计划 |
|---|---|---|---|
| 1 | **`default_xi`（摩擦 0.75）** | `torch.zeros(...,33)`（摩擦 0） | **`b.default_xi()`** 全程 |
| 2 | **多步自回归（k=20）** | 只有单步（`p21_refresh2_cl.py`） | **single + k 步**（`--mode both` 等价） |
| 3 | **`gain_loss`（配对探针）** | 完全没有（grep 0 次） | **`--gain-weight 2.0`** |

对照：`train_wm_refresh_v7.py`（wm_base6 的 350 步）**三样都在**（`--mode both --k 20` + `default_xi`），所以 dir_agree 保住了 0.979；
`p21_refresh2_cl.py`（refresh1/2/3）**三样全丢**，dir_agree 掉到 0.635。

---

## 2. 训练配方（一次定死）

| 项 | 值 |
|---|---|
| **初始化** | `comimic/runs/a_prime_sonic_exc/wm_gain2.pt`（sha `fc5aeb30`） |
| **A' 数据** | `comimic/runs/a_prime_sonic_exc/roll_windows_s1.pt`（92,785 窗口，`states(92785,21,71)`、`actions(92785,20,29)`） |
| **闭环数据** | `runs/dual_actor_wm6/c_lora/p25/refresh3/cl_pairs.pt`（820,384 转移；train 658,802 / val 161,582） |
| **配对探针** | `comimic/runs/paired_probe2/gain_pairs.pt`（7456 组 × 4 动作） |
| **stage1（1000 步）** | 单步：A' **70%** + 闭环转移 **30%**（batch 级） |
| **stage2（2000 步，k=20）** | 多步：A' **70%** + 闭环窗口 **30%**（`--windows2`/`--mix2 0.3`） |
| **loss** | `ssm_wm_loss + 2.0 × increment_loss + 2.0 × gain_loss` |
| **lr** | stage1 `2e-5`、stage2 `1.5e-5` |
| **batch / gain_batch / k** | 8192 / 64 / 20 |
| **存盘** | **每 500 步一个 ckpt**（选型用） |
| **xi** | **`default_xi` 全程（0.75）** |

---

## 3. 需要补的工程（两处）

### 3.1 `scripts/c_wm6/build_cl_windows.py`（新写，~40 行）

从 `cl_pairs.pt` 切出 stage2 用的窗口：

```
输入: states(820384,71) actions(820384,29) next_states ep_ids
做法: 按 ep_ids 分组 → 组内校验 states[i+1]≈next_states[i] → 滑窗 k=20（stride 20）
输出: cl_windows.pt
      states  (N,21,71)
      actions (N,20,29)
      ep_ids  (N,)
      meta    {k, stride, n_episodes, n_windows}
```

### 3.2 `scripts/finetune_wm_gain.py`（加 ~20 行）

1. `--cl-transitions <cl_pairs.pt>` + `--mix-cl 0.3`：stage1 按 30% 从转移里抽 `(s,a,s')`；
2. stage2 直接用现成机制：`--windows2 <cl_windows.pt> --mix2 0.3`（逻辑已在脚本里）；
3. 每 `--save-every 500` 步存 ckpt。

---

## 4. 训练命令

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh

# 1) 切闭环窗口（约 20 min）
python3 scripts/c_wm6/build_cl_windows.py   --pairs runs/dual_actor_wm6/c_lora/p25/refresh3/cl_pairs.pt   --out runs/c_wm6/cl_windows.pt --k 20 --stride 20

# 2) 训练 3000 步（4–6 h）
python3 scripts/finetune_wm_gain.py   --init-wm   /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/runs/a_prime_sonic_exc/wm_gain2.pt   --windows   /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/runs/a_prime_sonic_exc/roll_windows_s1.pt   --cl-transitions runs/dual_actor_wm6/c_lora/p25/refresh3/cl_pairs.pt --mix-cl 0.3   --windows2  runs/c_wm6/cl_windows.pt --mix2 0.3   --pairs     /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/runs/paired_probe2/gain_pairs.pt   --stage1 1000 --stage2 2000 --batch 8192 --gain-batch 64   --lr1 2e-5 --lr2 1.5e-5 --gain-weight 2.0 --inc-weight 2.0 --k 20   --save-every 500   --out runs/dual_actor_wm6/c_lora/p26/wm_gain3/wm_gain3.pt --device cuda:0
```

---

## 5. 评测协议（两组数，缺一不可）

### 5.0 前置：所有评测必须用 `default_xi`（0.75），禁止 `zeros`

### 5.1 绝对误差（WM 自己的想象，快速）

复用 refresh 的门禁口径（`p21_refresh2_cl.py` 的 eval 部分或 `train_wm_refresh_v7.py` 的 eval）：

```bash
python3 -m ... eval_wm_refresh_v7.py --wm-candidate <ckpt> ...
```

产出四项：录制单步误差、闭环单步误差、零动作比、打乱比。

### 5.2 方向可信度（dir_agree，必须做）

三步（与 `p19/diag/dir_agree_3wm` 完全同协议：n=96、horizon=2、delta=0.03）：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03

# 1) 生成扰动 job（依赖该 WM 的梯度方向）
PYTHONPATH=. python3 scripts/probe_wm_gradient_truth.py plan \
  --ckpt <ckpt> --dump "runs/wm_real/traj_wmuse2_val/v_11*.pt" \
  --horizon 2 --windows 96 --delta 0.03 --out /tmp/job_gain3.pt --device cuda:0

# 2) 真物理 rollout（Isaac，约 5–10 min）
source scripts/isaac_env.sh
$ISAAC_VENV/bin/python -u scripts/isaac_probe_action_rollout.py \
  --job /tmp/job_gain3.pt --out /tmp/roll_gain3.pt --device cuda:1 --headless

# 3) 打分
PYTHONPATH=. python3 scripts/probe_wm_gradient_truth.py score \
  --ckpt <ckpt> --job /tmp/job_gain3.pt --roll /tmp/roll_gain3.pt --device cuda:0
```

输出：`dir_agree`、`corr_dJ`。

---

## 6. 门禁与止损（跑之前定死）

| 关卡 | 阈值 | 不过怎么办 |
|---|---|---|
| **G1 方向** | `dir_agree ≥ 0.90` 且 `corr_dJ ≥ 0.28` | 用 s500 / s1000 的中间 ckpt 再测；都不到 → 降 `--mix-cl` 到 0.15 重训 |
| **G2 精度** | 录制误差 ≤ 0.20 且 闭环误差 ≤ 0.30 | 加 500 步 stage2 续训；仍不过 → 报告 |
| **G3 残差** | 用该 WM 训的残差，真机 30 格配对净胜 ≥6 | 判为“WM 利用”，停 |

**禁止**：只看绝对误差选模型；G1 不过还硬上残差。

---

## 7. 时间估算

| 步骤 | 时长 |
|---|---|
| 写 `build_cl_windows.py` + 跑 | 20 min |
| 改 `finetune_wm_gain.py`（`--cl-windows`、中途存盘） | 20 min |
| 训练 3000 步（stage1 1000 + stage2 2000×k20） | **4–6 h**（单卡） |
| 评测（绝对误差 + dir_agree ×3 个 ckpt） | 1 h |
| 合计 | **约 6–8 h** |

> 想快：把 stage2 的 `--k 20` 降到 10，时间减半，但和 wm_gain2 的配方就不再严格一致（建议不改）。

---

## 8. 产出与后续

- [ ] `runs/dual_actor_wm6/c_lora/p26/refresh4/wm_gain3.pt`（+ 每 500 步的中间 ckpt）
- [ ] 四项绝对误差 + `dir_agree`/`corr_dJ`
- [ ] 若 G1/G2 通过 → 用它训残差：
      `python3 scripts/c_wm6/train_residual.py --use-vla 1 --wm-ckpt .../wm_gain3.pt ...`
- [ ] 残差真机 30 格配对（与已有 base 30 格比）

---

## 9. 一句话

**用 wm_gain2 的配方（ssm + inc + gain）在 A'+闭环数据上重训 3000 步，靠 gain_loss 保住方向、靠闭环数据降绝对误差，选型只认 `dir_agree`。**
