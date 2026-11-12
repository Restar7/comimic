# three_track_v6 — M0/M1 审计与状态报告（2026-11-08 04:0x）

## 0. 状态总表

| 项 | 状态 | 证据/路径 |
|---|---|---|
| worktree | PASS | `comimic-wt03` @ `harness/03-bptt-diagnosis`，M1 提交 `e8d8ba5` |
| Delta best（A 初始化） | PASS | `runs/temporal_ac_bptt_v5/main_v2/seed0/checkpoint_actor_best.pt`（6.99 MB）；seed1 同目录，均只读 |
| v5 冻结 VLA | PASS | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（`eval_set/windows_meta.json: vlm_ckpt`） |
| FT8000（C 底座） | PASS | **与 v5 同一个** `checkpoint-8000`：`Gr00tN1d7`、bf16、2 个 safetensors（4.99+1.92 GB）、`global_step8000/`（DeepSpeed 状态）⇒ 完整模型 |
| A′ WM | PASS | `runs/a_prime_sonic_exc/wm_gain2.pt`（35 MB；sha 记在 windows_meta） |
| transitions | PASS | `runs/a_prime_sonic_exc/all_transitions.pt`（370 MB） |
| 数据 split | PASS | `eval_set/windows_meta.json`：train_clips=375、val/test 各 81、`n_eligible_frames=338402`；训练用 `allowed_clips` 过滤到 235,280 |
| LoRA 基建 | PASS | `diff_vla_wbc/lora.py`（LoRALinear，A=kaiming / B=0 ⇒ 初始恒等） |
| SONIC hist10 | PASS | `action_provider_sonic.py::_vla_hist10`（绝对 q+qd，HA 序，最后一帧=当前状态）；HA↔STANDARD 对拍 1.49e-8 |
| 物理动作/损失模块 | PASS（本轮到） | `diff_vla_wbc/sonic_physical_action.py`、`diff_vla_wbc/physical_bptt_losses.py`，数值自检通过 |
| **权威关节限位 / rate limit** | **BLOCKED** | 训练 bundle 不暴露 MuJoCo `jnt_range`（只有 `world_model_cfg`）；trackerlab 有 `RobotKinematics`（`kinematics/supermimic/robot.py` 从 `mj_range` 读 lo/hi）但需要 robot cfg；未接通前 `L_lim` 不应开训 |
| A/B/C 训练 | **NOT_RUN** | 需 v6 训练器（`train_delta_physical_bptt_v6.py`、`train_sonic_lora16_ft8000_bptt.py`）；未写、未跑，不伪造 |
| 真实 SONIC 闭环（Δ v5） | 部分 PASS | `runs/closedloop_v5/delta3000/`：**OpenDoor seed0 success/929 步、seed1 success/1633 步**；SitSofa/VisionNavi 在跑 |

## 1. 本轮闭环关键结果（SONIC=1，3000 步上限）

| run | success | 步数 | reward | 视频 |
|---|---|---|---|---|
| OpenDoor seed0 | ✅ | 929 | 1.0 | ✅ |
| OpenDoor seed1 | ✅ | 1633 | 1.0 | ✅ |
| SitSofa seed0 | 运行中 | — | — | — |
| SitSofa seed1 / VisionNavi 0,1 | 排队 | — | — | — |

（早前 `p0_sonic/base` 的 DoubleDesk Base 是 0 奖励/timeout，仅 1000 步；不同任务，不可直接对比。）

## 2. M1 已实现的接口（三条路线共用）

- `q_target_raw(a) = q_default + G1_ACTION_SCALE ⊙ a`（禁止 scale=1.0）；
- `standard/ha_v31` 显式换序；
- `safety_bounds/apply_safety`：不对称上下限 + margin + `rate·dt`（dt 必须是物理步长）；
- `to_wm_action(q, mode)`：物理动作 → WM 训练域的唯一入口；
- `L_lim = mean[0.5·max_{t,j}v² + 0.5·mean_t max_j v²]`（不对称、归一化、max 形式）；
- `L_same`、`L_peak`（单关节）、`L_rate`（裁剪前）、`action_stats`。

## 3. 下一步（按任务书顺序，需要新会话继续）

1. 接通权威限位/rate：用 `trackerLab.kinematics.supermimic.robot.RobotKinematics` 的 `q_lo/q_hi`（MJCF `jnt_range`）+ 90% soft limit；rate limit 从实际 actuator 配置读；产出 `runs/three_track_v6/physical/limits_standard29.npz`；
2. 写 `scripts/calibrate_physical_bptt_weights.py`（Delta best / random Delta / zero-LoRA 三类参数，测 Reward 与各物理项的梯度范数/夹角，输出 `calibration/{delta,lora}.json`）；
3. 写 `scripts/train_delta_physical_bptt_v6.py`（A: `--init-mode checkpoint`；B: `--init-mode random`；物理损失 + safe-WM 包装 + strict clips）；
4. 写 `scripts/train_sonic_lora16_ft8000_bptt.py`（FT8000 + rank16/alpha32 注入、4 步可微 FM、零 LoRA 恒等对拍、物理损失）；
5. 三条各跑 Pilot（A 100 / B 200 / C 200），过 Gate 后扩训，再做真实 SONIC–Isaac 六臂配对。
