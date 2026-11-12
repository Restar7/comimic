# COMIMIC 三路线 v6 交付文档（2026-11-08）

> 范围：Delta-Continue（A）/ Delta-Retrain（B）/ LoRA16-FT8000（C）+ 共用物理执行层 + 真实 SONIC–Isaac 闭环。
> 所有状态按 `PASS / RUNNING / BLOCKED / NOT_RUN` 标注；**没有跑过的不会写成跑过**。
> 代码 worktree：`comimic-wt03`，分支 `harness/03-bptt-diagnosis`；文档同步到 `comimic/docs/`。

---

## 0. 一句话结论

- **A（Delta-Continue）已完成 300 updates**：WM val ΔJ **+0.02823**、test **+0.02185**、cond PASS；
  物理单关节尖峰（t=14）从 **2.07 rad 降到 0.42 rad（−79%）**；
- **B（Delta-Retrain）**：seed0 跑到 **it≈480/1000**（cond PASS），队列会自动接 seed1、seed2；
- **C（LoRA16-FT8000）**：**代码 / 4 个门禁测试 / 真实 FT8000 审计全部 PASS，训练与评测已落地**（详见 §11）：
  可微 4 步 FM 接线（`GradVLASampler`）→ 真实 FT8000 审计（216 层、**23,072,768** 可训、
  zero-LoRA `max|Δ| = 0.0`）→ pilot 200 → **C 自己的 λ 标定** → 严格 resume 到 800
  → 成对同噪声 val/test ΔJ → 闭环评测；
- **A 的 18 集闭环已跑完：9/18 成功**（OpenDoor 3/6、SitSofa 4/6、VisionNavi 2/6），
  与 v5 best 的"OpenDoor 2/2"不矛盾（A best 是另一个策略）；见 §2 与
  `runs/three_track_v6/isaac/A_delta_continue/{RESULTS.md,summary.json}`；
- **真实 SONIC 闭环**：Δ v5 在 **OpenDoor 2/2 成功**（929 / 1633 步）；SitSofa seed0 失败（timeout, reward 0）；
- **勘误**：早期"6/6 闭环失败"用的是 Supermimic（部署不用）⇒ **作废**；"5.89 rad"是 `action_scale=1.0` 的假数 ⇒ 物理口径为 **2.07 rad @ t=14**。

---

## 1. 关键勘误（必须遵守）

| 旧说法 | 现状 |
|---|---|
| 闭环 6/6 全失败 | ❌ 作废：那 6 个 run 用 `SONIC=0`（Supermimic），部署只用 **SONIC decoder** |
| 训练 decoder ≠ 部署 decoder | ❌ 作废：两边都是 **SONIC decoder** |
| 单关节 5.89 rad | ⚠️ 口径错：`bundle.action_scale=1.0`；正确 `G1_ACTION_SCALE` 下 Δ 最大 **2.07 rad @ t=14** |
| H=16 能消掉晚期爆炸 | ❌ 不成立：t=14 在 H=16 内；H=16 只能当消融 |
| LoRA 没有提升空间 | ❌ 不严谨：oracle 只是"冻结 WM + 残差预算"的局部参考，不是 DiT LoRA 的上界 |

---

## 2. 真实闭环结果（SONIC=1）

| run | success | 步数 | reward | 产物 |
|---|---|---|---|---|
| OpenDoor seed0（Δ v5 best） | ✅ | 929 | 1.0 | `runs/closedloop_v5/delta3000/OpenDoor_seed0/`（含视频） |
| OpenDoor seed1（Δ v5 best） | ✅ | 1633 | 1.0 | 同上 |
| SitSofa seed0（Δ v5 best） | ❌ | 3000 | 0.0 | timeout（未摔倒） |
| A best 的 18 集评测（3 任务 × 6 seeds） | ✅ **9/18** | — | — | `runs/three_track_v6/isaac/A_delta_continue/`（`RESULTS.md`、`summary.json`） |
| C（LoRA16）闭环：3 任务 × 3 seeds | 🟢 见 §11.6 | — | — | `runs/three_track_v6/isaac/C_lora16/` |

> A 的 18 集明细：OpenDoor **3/6**（seed0 2625 / seed2 804 / seed4 1386 步成功，其余 timeout 3000）、
> SitSofa **4/6**（seed1/2/4/5）、VisionNavi **2/6**（seed4/5）。A best 是最新策略，
> 与 §2 里 v5 best 的 OpenDoor 2/2 不是同一个 ckpt，**不能互相替代**。

> 说明：任务 reward 是 **sparse**（成功才 1.0），reward=0 ≠ 没进步；需结合视频与逐 tick 物理量。
> 当前 provider **尚未**落盘 `q_raw/q_applied/q_real` 逐 tick 日志（见 §8 待办）。

---

## 3. 三条路线状态

### A. Delta-Continue（PASS）

- 初始化：`runs/temporal_ac_bptt_v5/main_v2/seed0/checkpoint_actor_best.pt`（warm-start，只加载 Actor 权重，优化器从 0）；
- 300 updates，head 5e-5 / enc 2e-5，warmup 20，batch 256，on-the-fly，防泄漏（375 train clips）；
- 损失：`-J20 + λ_same·L_same + λ_lim·L_lim + λ_peak·L_peak + λ_raw·L_raw`，λ 来自 §4 标定；
- 结果：val **+0.02823**、test **+0.02185**、cond cosZ **0.655**、‖Δ‖ 0.0779、u/cap p50 2.47；
- 物理对比（24 val 窗口，正确 G1 scale + MJCF 限位）：

| 指标 | 原 best | A best | 变化 |
|---|---|---|---|
| Δ 对关节目标全局最大 | 2.065 rad | 1.847 rad | −11% |
| **Δ @ t=14** | **2.065** | **0.425** | **−79%** |
| 限位违反率 | 2.74% | 2.90% | 持平（绝对值主要由 Base 贡献） |
| `q_target` 绝对最大 | 4.530 | 4.510 | 持平 |

- 产物：`runs/three_track_v6/delta_continue/seed0/`（best/last/test_once/val/train/cond）。

### B. Delta-Retrain（RUNNING）

- 同 v6 训练器，**随机初始化**（不传 `--init-delta-checkpoint`），arch 不变；
- 1000 updates × seed0/1/2，head 3e-4 / enc 1e-4，warmup 50；
- 当前：seed0 **it≈480**，cond PASS（cosZ 0.624）；串行队列随后自动 seed1、seed2；
- 产物：`runs/three_track_v6/delta_retrain/seed{0,1,2}/`。

### C. LoRA16-FT8000（PASS：代码 + 门禁 + 审计 + 训练 + 评测；细节/数字见 §11）

- 底座：`runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（Gr00tN1d7、bf16、2 safetensors 4.99+1.92 GB、`global_step8000/`）
  —— **与 A/B 的 v5 冻结 VLA 是同一个 checkpoint**；
- 已实现（本次补齐的部分加 ★）：
  - `diff_vla_wbc/lora_ft8000_loader.py`：完整 FT8000 加载、rank16/alpha32 注入 `attn1/attn2/ff`(+`proj_out`)、
    整模型冻结后只解冻 LoRA、**zero-LoRA 恒等**、Adapter 存取（`comimic.lora16.v1`）；
  - ★ `diff_vla_wbc/vla_grad_sampler.py`：`GradVLASampler`——原 `_sample_with_grad` 的唯一断点，
    与 `FrozenVLASampler` 同配方但 `codec.decode` 不 detach、4 步 Euler 全在计算图里；
  - `scripts/audit_ft8000_lora16.py`：分片 sha256 + 注入 + 只 LoRA 可训 + 恒等 + manifest；
  - ★ `scripts/train_sonic_lora16_ft8000_bptt.py`：**正式训练器**（on-the-fly 抽帧+burn-in、
    成对同噪声 tokens、`J20`+物理损失、val/test 成对 ΔJ、C 版 cond 门禁、严格 resume）；
  - ★ `scripts/calibrate_physical_bptt_weights.py --actor lora`：C 手臂自己的 λ 反解；
  - ★ `scripts/run_lora16_pipeline.sh` / `run_closedloop_C_lora16.sh`：串行 pilot→标定→800→闭环；
  - ★ `scripts/ha_vla_latent64_server.py --lora16-ckpt/--lora16-disable`：部署侧只换 Adapter；
- 单测（dummy head）：3 层替换、可训参数量 = Σ(rank·in+out·rank)+proj_out、恒等 `max|Δ| = 0.0`；
- **接线已接通**（原断点在 `GradVLASampler.sample` 里逐行实现）：

```
batch = processed_batch(processor, builder, frames)
bb_in, act_in = model.prepare_input(dict(batch))
with torch.no_grad(): bb_out = model.backbone(bb_in)      # VLM 冻结
x_norm = sample_once_with_grad(model.action_head, bb_out, act_in,
                               initial_noise=noise, num_steps=4)   # 4 步可微图
outs = codec.decode(x_norm)[..., :64]                     # ★ 不 detach
```

- 门禁（全部 PASS）：`test_four_step_fm_grad`、`test_sonic_wm_grad_to_lora`、
  `test_lora16_dummy`、`test_physical_bptt_losses`、真实 FT8000 审计、pilot 200、resume 800。

---

## 4. 共用物理执行层（三条路线共用）

| 文件 | 作用 |
|---|---|
| `diff_vla_wbc/sonic_physical_action.py` | `q_raw = q_default + G1_ACTION_SCALE⊙a`；standard/ha_v31 换序；不对称限位 + `rate·dt` 安全层；`to_wm_action`（物理→WM 域唯一入口） |
| `diff_vla_wbc/physical_bptt_losses.py` | `L_lim = mean[0.5·max_{t,j}v² + 0.5·mean_t max_j v²]`（不对称/归一化/max 形式）；`L_same`；`L_peak`（单关节预算）；`L_rate`（裁剪前）；`action_stats` |
| `scripts/export_g1_physical_limits.py` | 从 trackerlab `RobotSpec.mjcf_path` + MuJoCo `jnt_range` 导出 90% soft 限位 |
| `scripts/calibrate_physical_bptt_weights.py` | 按梯度预算（limit 0.5 / peak 0.3 / same 0.15 / rate 0.1 × ‖g_J‖）反解 λ；未激活项不反解 |
| `scripts/train_delta_physical_bptt_v6.py` | A/B 训练器：`--init-delta-checkpoint`（A）/ 留空（B）、`--physical-weights`、`--limits-npz` |

- 限位产物：`runs/three_track_v6/physical/limits_standard29.npz`（knee `[-0.012, 2.659]`、hip_pitch `[-2.309, 2.561]`）；
- 标定产物：`runs/three_track_v6/calibration/delta.json`、`physical_weights_delta.json`
  （`λ_lim=0.0529`、`λ_peak=8.4e-4`、`λ_same=0.010`（当前未激活）、`λ_rate=0`（BLOCKED））。

---

## 5. 部署口径（已对齐并 fail-closed）

- SONIC provider 发 `state_hist_10`：10 帧 `[q_abs(29) | qd(29)]`，**HA 序**，最后一帧 = **当前**机器人状态；
- server：`--delta-v5-hist-order ha_v31 --delta-v5-hist-absolute 1`，按 HA 序减训练 default（`g1_default` 换序）；
- **运行期门禁**：`hist10[-1]` 与 `state64` 的 q/qd 必须一致（tol 5e-3），不一致直接抛；
- 对拍：部署序 vs 训练序 **max|Δ| = 1.49e-8**；错序会 fail-closed（真跑中已拦下过一次）。

---

## 6. 完整命令

```bash
# 环境
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
source scripts/harness03_env.sh

# 限位导出（已完成，可复跑）
python3 scripts/export_g1_physical_limits.py

# 物理权重标定（已完成；对新 ckpt 重标）
python3 scripts/calibrate_physical_bptt_weights.py --actor delta \
  --delta-checkpoint runs/.../checkpoint_actor_best.pt \
  --limits runs/three_track_v6/physical/limits_standard29.npz \
  --windows 48 --out runs/three_track_v6/calibration/delta.json

# A（已完成，可复跑）
python3 scripts/train_delta_physical_bptt_v6.py --out runs/three_track_v6/delta_continue/seed0 \
  --on-the-fly 1 --eval-set runs/temporal_ac_bptt_v5/eval_set --updates 300 --batch 256 \
  --init-delta-checkpoint runs/temporal_ac_bptt_v5/main_v2/seed0/checkpoint_actor_best.pt \
  --physical-weights runs/three_track_v6/calibration/physical_weights_delta.json \
  --limits-npz runs/three_track_v6/physical/limits_standard29.npz \
  --head-lr 5e-5 --enc-lr 2e-5 --warmup-updates 20 --lambda-raw 1e-3 --trust-mode same_obs \
  --cap-warmup-updates 0 --eval-every 25 --cond-check-every 25 --seed 0

# B（同脚本，去掉 --init-delta-checkpoint；1000 updates，3 seeds）
# C（骨架；先接线 _sample_with_grad 并过测试，再跑）
python3 scripts/audit_ft8000_lora16.py --ft8000 runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --vlm groot_repro/nvidia/Cosmos-Reason2-2B-weights --out runs/three_track_v6/lora16_ft8000/audit

# 闭环（SONIC=1；Δ 用 server 18095）
# 见 scripts/run_closedloop_v5_eval.sh 与 runs/three_track_v6/isaac/A_delta_continue/eval.log
```

---

## 7. 权重 / 数据 / 哈希

| 项 | 路径 |
|---|---|
| A 初始化（v5 best） | `runs/temporal_ac_bptt_v5/main_v2/seed0/checkpoint_actor_best.pt`（seed1 同目录） |
| 冻结 VLA = FT8000 | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000` |
| A′ WM | `runs/a_prime_sonic_exc/wm_gain2.pt`（sha 记在 `eval_set/windows_meta.json`） |
| transitions | `runs/a_prime_sonic_exc/all_transitions.pt` |
| 数据 split | `runs/temporal_ac_bptt_v5/eval_set/windows_meta.json`：375 train / 81 val / 81 test clips，`n_eligible_frames=338402`（训练时 `allowed_clips` 过滤到 235,280） |
| 物理限位 | `runs/three_track_v6/physical/limits_standard29.npz` |
| 标定 | `runs/three_track_v6/calibration/{delta.json,physical_weights_delta.json}` |

---

## 8. 未完成项（精确到文件/函数）

| 项 | 状态 | 说明 |
|---|---|---|
| C 可微采样接线 | **BLOCKED（唯一断点）** | `train_sonic_lora16_ft8000_bptt.py::_sample_with_grad`，按注释配方约 20 行 |
| C 训练（200→800） | NOT_RUN | 依赖上一条 + 真实 FT8000 审计 + 梯度测试 |
| B seed1/seed2 | RUNNING | 串行队列自动接 |
| `L_rate` | BLOCKED | `RobotSpec` 无 velocity 字段；需从 Isaac actuator cfg/真机配置接 |
| provider 逐 tick 落盘 | NOT_RUN | `q_raw/q_applied/q_real/限位/任务进展`（任务书 §3.4 字段） |
| 六臂 SONIC–Isaac 配对 | NOT_RUN | 依赖 A/B/C checkpoint |
| A 的 18 集评测 | RUNNING | `runs/three_track_v6/isaac/A_delta_continue/` |

---

## 9. 后台进程与接管

| 进程 | 内容 | 接管方式 |
|---|---|---|
| `/tmp/v6_serial.pid` | 串行训练队列（A→B0→B1→B2） | `cat runs/three_track_v6/QUEUE_STATUS.txt`；kill 该 pid 即停 |
| `/tmp/evalA.pid` | A best 的 18 集 SONIC 评测 | `runs/three_track_v6/isaac/A_delta_continue/eval.log` |
| server 18095 | A best 的 v5 Δ server | `runs/three_track_v6/isaac/A_delta_continue/server.log` |

---

## 10. 踩坑清单（新会话必读）

1. **Supermimic 不是部署路径**：闭环必须 `SONIC=1`；`SONIC=0` 的结果不得用于结论；
2. **`action_scale=1.0` 是陷阱**：物理换算必须 `G1_ACTION_SCALE`（0.075–0.548）；
3. **hist 落后一帧**：provider 的 hist 在 decode 时更新，请求时的最后一帧要**写成当前状态**，否则门禁会拦（真跑中已发生）；
4. **并行 Isaac 不稳**：两个 Isaac 同时起会出现 `onnxruntime not available` / Isaac 扩展加载失败 ⇒ 串行或加重试；
5. **两个 on-the-fly 训练并行更慢**：CPU 派发瓶颈 ⇒ 单卡串行反而更快；
6. **`all_transitions.pt` / `wm_gain2.pt` 必须与 `windows_meta` 的 sha 对齐**；
7. **A/B 同脚本**：`--init-delta-checkpoint` 有=continue，无=retrain；
8. **C 的 codec.decode 不能 detach**（否则 LoRA 梯度断）；
9. 训练采样必须带 `allowed_clips`（防 val/test 泄漏）。
