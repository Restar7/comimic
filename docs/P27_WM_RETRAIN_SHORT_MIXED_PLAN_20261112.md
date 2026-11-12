# P27 计划：WM 重训 —— 保留 inc + gain，短训 + 五五混数据 + 按 dir_agree 选型

> 生成：2026-11-12 ｜ 状态：**待执行** ｜ 前置：`runs/c_wm6/cl_windows.pt`（22,698 个真机闭环窗口，已生成）
> 一句话：**不改数据来源、不改 loss 组成，只改三件事——补上 stage1 的 increment、把数据混成 50/50、把 3000 步砍到 400 步并按 dir_agree 选中间 checkpoint。**

---

## 1. 为什么这样改（有依据的部分 / 没验证的部分分开写）

### 有依据

| 观察 | 数据 |
|---|---|
| 方向能保住的是「短训」 | wm_base6 = wm_gain2 + **350 步** → dir 0.979 |
| 训得越久方向越差 | refresh1 800 步 0.885 → refresh2 3000 步 0.771 → wm_gain3 3000 步 **0.517** |
| inc 能给动作分支供梯度 | `increment_loss` 的 docstring：不缩放的话动作效应只有 2.6e-6 |
| gain 能钉动作方向 | `gain_loss`：同状态 × K 动作的配对监督 |
| inc + gain **可以**训出好方向 | **wm_gain2 就是 ssm + inc + gain 训出来的**，wm_base6 的 0.979 继承自它 |

### 没验证（本次要排除的变量）

1. wm_gain3 崩到 0.517，**主因未知**：可能是 3000 步、可能是混了 30% 闭环、也可能是 inc 只在 stage2 生效；
2. 因此本次只改这三个变量，**不再叠加别的改动**。

---

## 2. 代码修改（3 处，必须先做）

### 2.1 `scripts/finetune_wm_gain.py`：stage1 补上 increment

现状（stage1）：

```python
l_wm = ssm_wm_loss(wm, pred, s1)
l_g  = one_gain(...) if gain_weight > 0 else 0
loss = l_wm + float(a.gain_weight) * l_g          # ← 没有 increment
```

改为：

```python
l_wm = ssm_wm_loss(wm, pred, s1)
l_inc = increment_loss(pred, s0, s1, vel_weight=1.0) if float(a.inc_weight) > 0 \
        else pred.new_zeros(())
l_g  = one_gain(...) if gain_weight > 0 else 0
loss = l_wm + float(a.inc_weight) * l_inc + float(a.gain_weight) * l_g
```

并打印：`[s1] wm=… inc=… gain=…`

### 2.2 `scripts/finetune_wm_gain.py`：数据混合比例改 50/50

- stage1：`--mix-cl 0.5`（从 `cl_pairs` 抽，已有 `--cl-transitions`）
- stage2：`--mix2 0.5`（从 `cl_windows` 抽，已有 `--windows2`）

> 注意：现在 stage1 的 `_n_cl = batch * mix_cl`；stage2 的 `_n2 = batch * mix2`。两处都改成 0.5 即可，无需新代码。

### 2.3 `scripts/finetune_wm_gain.py`：存盘间隔改 100

```bash
--save-every 100
```

---

## 3. 训练配置

| 项 | 值 | 依据 |
|---|---|---|
| 初始化 | `comimic/runs/a_prime_sonic_exc/wm_gain2.pt` | 方向源头 |
| 数据（stage1） | A' 窗口 **50%** + 闭环转移 `cl_pairs` **50%** | A' 保动作覆盖，闭环保部署分布 |
| 数据（stage2，k=20） | A' 窗口 **50%** + 闭环窗口 `cl_windows` **50%** | 同上 |
| 配对探针 | `runs/paired_probe2/gain_pairs.pt` | gain 的唯一来源 |
| loss | `ssm + 2.0 × increment + 2.0 × gain`（**两个阶段都生效**） | wm_gain2 原配方 |
| 步数 | stage1 **200** + stage2 **200** = **400** | wm_base6 是 350 步量级 |
| lr | stage1 `1e-5`、stage2 `1e-5`（比 wm_gain2 的 2e-5 更保守） | 少训就要低 lr，避免一步跨太远 |
| batch / gain_batch / k | 8192 / 64 / 20 | wm_gain2 原值 |
| xi | **`default_xi`（0.75）全程** | 修正 refresh 线的口径错误 |
| 存盘 | 每 **100** 步 → s100 / s200 / s300 / s400 | 按 dir_agree 选 |

> 如果 400 步的 dir_agree 还在 0.9 以上、且误差仍在降，可以再续 400 步（同配置、从最优 ckpt 继续）。

---

## 4. 训练命令

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh

python3 scripts/finetune_wm_gain.py \
  --device cuda:0 \
  --wbc-ckpt /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/runs/S1_handover/WBC_A24000_wmvel_cmd.pt \
  --init-wm  /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/runs/a_prime_sonic_exc/wm_gain2.pt \
  --windows  /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/runs/a_prime_sonic_exc/roll_windows_s1.pt \
  --cl-transitions runs/dual_actor_wm6/c_lora/p25/refresh3/cl_pairs.pt --mix-cl 0.5 \
  --windows2 runs/c_wm6/cl_windows.pt --mix2 0.5 \
  --pairs    /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/runs/paired_probe2/gain_pairs.pt \
  --stage1 200 --stage2 200 --batch 8192 --gain-batch 64 --k 20 \
  --lr1 1e-5 --lr2 1e-5 --gain-weight 2.0 --inc-weight 2.0 \
  --save-every 100 --log-every 25 \
  --out runs/dual_actor_wm6/c_lora/p27/wm_gain4/wm_gain4.pt
```

预计 **3–5 分钟**。

---

## 5. 评测（每个 checkpoint 都做）

### 5.1 多步误差（快，2 分钟/个）

```bash
python3 scripts/c_wm6/eval_wm_multistep2.py --device cuda:0 \
  --ckpts runs/dual_actor_wm6/c_lora/p27/wm_gain4/wm_gain4_s100.pt \
          runs/dual_actor_wm6/c_lora/p27/wm_gain4/wm_gain4_s200.pt \
          runs/dual_actor_wm6/c_lora/p27/wm_gain4/wm_gain4_s300.pt \
          runs/dual_actor_wm6/c_lora/p27/wm_gain4/wm_gain4_s400.pt
```

报：`rec H1 / rec H10 / rec H20 / cl H1 / cl H10 / cl H20`。

### 5.2 动作敏感性（快）

```bash
python3 /tmp/probe_sens.py     # 需要把 ckpt 列表换成 s100..s400
```

报：`零动作比 / 打乱动作比`（要 >1，否则动作分支没学到）。

### 5.3 dir_agree（慢，10 分钟/个；只测前两名）

```bash
cd ../comimic-wt03
# 0) 先把 WM 合进带 pi 的完整 ckpt（探针只接受完整 Isaac ckpt）
python3 - <<'PY'
import torch
base = torch.load('/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/runs/S1_handover/WBC_A24000_wmvel_cmd.pt', map_location='cpu', weights_only=False)
wm = torch.load('runs/.../wm_gain4_s200.pt', map_location='cpu', weights_only=False)['wm']
ck = dict(base); ck['wm'] = wm
torch.save(ck, '/tmp/ckpt_gain4_s200.pt')
PY

# 1) plan
export PYTHONPATH="/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic/source:$PYTHONPATH"
PYTHONPATH=.:$PYTHONPATH python3 scripts/probe_wm_gradient_truth.py plan \
  --ckpt /tmp/ckpt_gain4_s200.pt \
  --dump "/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/runs/wm_real/traj_wmuse2_val/v_11*.pt" \
  --horizon 2 --windows 96 --delta 0.03 --out /tmp/job_g4.pt --device cuda:0

# 2) Isaac 物理 rollout
source scripts/isaac_env.sh
$ISAAC_VENV/bin/python -u scripts/isaac_probe_action_rollout.py \
  --job /tmp/job_g4.pt --out /tmp/roll_g4.pt --device cuda:1 --headless

# 3) score
PYTHONPATH=.:$PYTHONPATH python3 scripts/probe_wm_gradient_truth.py score \
  --ckpt /tmp/ckpt_gain4_s200.pt --job /tmp/job_g4.pt --roll /tmp/roll_g4.pt --device cuda:0
```

---

## 6. 选型判据（跑之前定死）

| 指标 | 阈值 | 权重 |
|---|---|---|
| **dir_agree** | **≥ 0.90** | 一票否决 |
| corr_dJ | ≥ 0.28 | 次优先 |
| 打乱动作比 | ≥ 1.15 | 动作分支必须真的学到 |
| rec H10 | 越小越好 | 精度 |
| cl H20 | 越小越好 | 精度 |

**选型顺序**：先筛 dir_agree ≥ 0.90 → 再按 cl H20 最低 → 并列时看 rec H10。

**若 4 个 checkpoint 全部 dir_agree < 0.85** → 停止，结论写成「短训 + inc + gain + 五五混数据也保不住方向」，转二选一（wm_base6 当裁判 / refresh2 当裁判）。

---

## 7. 通过后的后续（不在本单内，但先写清）

1. 用选出的 WM 训残差：`train_residual.py --wm-ckpt <胜出者>`（训练窗口同时混 A' 与 `cl_windows`）；
2. 真机 30 格配对（vs 已有 base 30 格）；
3. 若残差真机有效 → 再做联合微调（起点 `c_base`）。

---

## 8. 风险

| 风险 | 处理 |
|---|---|
| 400 步不够降误差 | 若 dir_agree 仍 ≥0.9 且误差在降，续 400 步（同配置） |
| 50% 闭环数据让方向掉 | 降 `--mix-cl/--mix2` 到 0.3 重跑，对比 dir_agree |
| gain 在 stage2 与 inc 冲突 | 分别单独关掉 inc / gain 各跑一版（各 5 分钟），定位是谁在破坏方向 |

---

## 9. 一句话

**保留 inc + gain（别去掉），但只训 400 步、A'/闭环各半、每 100 步存盘、按 dir_agree 选——把"训太久"这个唯一确定的可疑因素排除掉。**
