# 卡 2,3 的消融结果：A3（BPTT vs PPO）+ A5（冻结 WBC）

> 自动生成：`python3 scripts/a3_a5_report.py`。所有数字直接来自
> `runs/ablations/**/eval_curve.jsonl` 与训练日志，未做任何挑选或平滑。

## 0. 协议（两臂只差被消融的那一个变量）

| 项 | 取值 |
|---|---|
| 数据 | **纯 HA**：`data/npz/ha_tokens`（699 条 / 634,016 帧 / 3.52 h）。不含任何 ft318、不含 mixed_ha_ft318 |
| SuperMimic WBC + 世界模型 | `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt`（Isaac Lab/PhysX，token 接口，iteration 10500） |
| 联合微调 VLA | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（官方微调产物） |
| 世界模型 | 冻结（`--wm-update-every 0`） |
| 手部辅助损失 | λ=0.2，boost=5 |
| batch / horizon | 2 per rank（2 卡 → 全局 4）/ 8 |
| 卡 | 2 卡 DDP（本容器 = 2×RTX 4090 48 GB） |

**整条链路都是 Isaac Lab 的**（`diff_vla_wbc/isaac_joint.py`），不是 mjlab/MuJoCo 旧链路：
policy 与世界模型 = `ckpt_010500.pt` 的 `pi`/`wm` 张量（逐元素相同，max|Δ|=0）；
观测（99 维 `relative_ori99`）、奖励（`MDP.reward`，用 ckpt 里记录的 `reward_cfg`）、
软终止（`MDP.soft_termination`）、可微 FK（`G1Kinematics`）、参考运动库
（`CompactMotionLib`，由 `data/npz/ha_tokens` 打包，699 条）全部来自
`trackerlab-supermimic`。门禁脚本 `scripts/verify_isaac_chain.py` 逐条断言通过。

## 1. A3 · 精确反传（BPTT）vs score-function（PPO），同墙钟 120 min/臂

| 指标 | BPTT (pathwise, exact) | PPO (score function, T=1) |
|---|---|---|
| eval 次数 | 6 | 6 |
| 训练步数 | — | 500 |
| 墙钟（s，最后一次 eval） | 4173 | 2456 |
| rollout_reward（首次 → 末次） | 0.0808 → 0.0844 | 0.0454 → -0.0069 |
| rollout_reward（末 3 次均值） | 0.0840 | -0.0095 |
| rollout_reward（最佳） | 0.0844 | 0.0454 |
| state_term（首次 → 末次） | 0.0062 → 0.0079 | 1.7886 → 5.1347 |
| token_mse（首次 → 末次） | 0.1477 → 0.0928 | 0.1885 → 1.6625 |
| hand_mse（首次 → 末次） | 0.0091 → 0.0032 | — → — |
| rollout/survival（末 3 次均值） | 1.0000 | — |
| ppo/sigma（末 3 次均值） | — | — |
| ppo/clip_frac（末 epoch，末 3 次均值） | — | — |
| ppo/clip_frac（**第 0 个 epoch**，末 3 次均值） | — | — |
| ppo/ratio（第 0 个 epoch，末 3 次均值） | — | — |
| 训练 loss（末 3 次均值） | -0.6419 | — |

### 0.1 链路门禁输出（`scripts/verify_isaac_chain.py`）

```text
[1] SuperMimic checkpoint: /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt
  [ok]   checkpoint is the Isaac layout {pi, wm} (trackerlab/Isaac Lab, not mjlab {policy, wm, config})
  [ok]   policy = token-interface decoder only (dec.0.bias …): the goal encoder is removed
  [ok]   policy owns its RunningNorm over 99 obs dims
  [ok]   world model is the LayerNorm MLP (wm.net.*) of the Isaac implementation
  [ok]   interface = 'token'
  [ok]   obs_mode = 'relative_ori99'
  [ok]   hist = 10
  [ok]   checkpoint carries the reward config the Isaac run used

[2] training bundle provenance (every consumer must be trackerlab)
  [ok]   bundle policy keys == ckpt['pi'] keys (no extra/replaced layers)
  [ok]   bundle world-model keys == ckpt['wm'] keys
  [ok]   policy tensors bit-identical to ckpt['pi'] (max|Δ| = 0.000e+00, 13 tensors)
  [ok]   WM tensors bit-identical to ckpt['wm'] (max|Δ| = 0.000e+00, 20 tensors)
  [ok]   no mjlab keys (encoders.* / decoder.mlp.* / dynamics.mlp.*) in the bundle

[3] analytic layer (obs / reward / termination / FK / motion)
  [ok]   MDP is trackerlab's (trackerLab.tracker_env.mdp.supermimic.MDP)
  [ok]   FK is trackerlab's (trackerLab.kinematics.supermimic.g1)
  [ok]   reference motion is trackerlab's CompactMotionLib (trackerLab.motion_buffer.motion_lib.compact)
  [ok]   robot contract from trackerTask get_robot('g1') (trackerTask.supermimic.robots)
  [ok]   per-frame obs width = 99 (relative_ori99)
  [ok]   obs history = 10
  [ok]   obs term dims = (29, 29, 3, 3, 29, 6)
  [ok]   reward()/observation() dispatch to MDP.obs / MDP.reward

[4] dataset: pure HA only (no ft318, no mixed)
  [ok]   motion path has no ft318/mixed tag: /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/data/npz/ha_tokens
  [ok]   packed Isaac motion library built from HA only: ['data/npz/ha_tokens']
  [ok]   packed library clips = 699 (pure HA set)
  [ok]   packed library frames = 634016
  [ok]   699 HA episodes found
  [ok]   episode metadata has no ft318 reference (HOI_double_desk__tw__ep000000.npz)
  [ok]   episode carries the recorded 64-d SONIC token
  [ok]   episode carries the 14-d hand labels

==============================================================================
[GATE PASSED] Isaac Lab chain end-to-end (policy/WM/obs/reward/termination/FK/motion from trackerlab-supermimic) + pure HA data (699 clips). No mjlab weights, no ft318 episode.
```

## 2. A5 · 只训 VLA head vs head + WBC（40 min/臂）

| 指标 | head only (policy frozen) | head + WBC policy |
|---|---|---|
| eval 次数 | 4 | 4 |
| 训练步数 | — | — |
| rollout_reward（首次 → 末次） | 0.0074 → 0.0670 | 0.0074 → 0.0732 |
| rollout_reward（末 3 次均值） | 0.0652 | 0.0715 |
| state_term（首次 → 末次） | 20.5813 → 1.0035 | 20.5813 → 0.6693 |
| token_mse（首次 → 末次） | 0.0558 → 1.5125 | 0.0558 → 0.4504 |
| hand_mse（首次 → 末次） | 0.6513 → 0.0088 | 0.6513 → 0.0037 |

## 3. 显存峰值（nvidia-smi 采样）

| GPU | 峰值占用 (MiB) |
|---|---|
| 0 | 46046 |
| 1 | 46084 |

## 5. 原始产物

| 臂 | 目录 | eval 曲线 | 权重 |
|---|---|---|---|
| A3/bptt | `runs/ablations/A3_bptt_vs_ppo/bptt` | ✅ (6 行) | — |
| A3/ppo | `runs/ablations/A3_bptt_vs_ppo/ppo` | ✅ (6 行) | — |
| A5/head_only | `runs/ablations/A5_only_head/head_only` | ✅ (4 行) | joint_latest.pt |
| A5/head_plus_wbc | `runs/ablations/A5_only_head/head_plus_wbc` | ✅ (4 行) | joint_latest.pt |

逐行数据：`docs/ABLATION_A3_RUN_PHYSINT_20261025.csv`
