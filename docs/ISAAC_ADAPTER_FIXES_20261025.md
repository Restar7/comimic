# Isaac 适配层三处口径修复（2026-10-25）

> 起因：卡 6 的 A1 处理臂（纯 HA 联合微调）越训 `token_mse` 越差（0.069 → 0.53），
> 而 `rollout_reward` 反而涨到 0.074–0.094。逐层分解后确认**不是训练脚本或 SuperMimic 学错**，
> 而是从 mjlab 迁到 Isaac 时积累的三处适配层口径问题。每处都有可复现脚本与验收数字。

## 1. 动作换算多除了一次 `action_scale`（主因）

| 项 | 内容 |
|---|---|
| 位置 | `diff_vla_wbc/isaac_joint.py`（`self.action_scale`、`reference_start`、`_wm_windows`） |
| 症状 | 观测历史里的 `a_prev` 与参考动作被放大 4–8 倍 → WBC 每步收到分布外输入 |
| 证据 | 参考动作 rollout：除以 scale → reward **−0.129** / 末步误差 1.31；`a = Δq` → **+0.0686 / 0.047** |
| 修法 | 换算默认 `a = Δq`（`self.action_scale = ones`）；奖励里的 `(a−a_prev)*scale` 仍用 spec |
| 开关 | `COMIMIC_ACTION_SCALE=spec`（旧行为，仅对照）/ `COMIMIC_REWARD_ACTION_SCALE=spec|one` |

## 2. Token 空间不一致（HA 标签 ≠ WBC 条件空间）

| 项 | 内容 |
|---|---|
| 事实 | HA npz 的 `g1_motion_token_64` 来自部署侧 `encoder_input (1762)` 前端；trackerlab WBC 条件在 **640 维端口装配**（10 帧未来参考 + anchor）。parity 检查 `max|Δtoken| = 1.0625`（HA）/ `0.4375`（ft318），脚本自己打印 `[FAIL] ... do not train` |
| 影响 | `token_mse` 不是保真度指标；用标签 token 喂 bundle 几乎跟不住参考，只有被奖励优化过的 token 能"补偿" |
| 修法 | `scripts/build_port_tokens.py` 用端口装配重算 699 条 HA clip 的 token → `data/port_tokens/ha_tokens/*.npy`；`RecordedTokenSource` 优先读侧车（`COMIMIC_PORT_TOKENS` 可覆盖）→ eval / `token_mse` / 参考 token 统一到 WBC 空间 |
| 证据 | 动作换算修好后：端口 token **0.070 / 2.87**，标签 token 0.052 / 6.48，参考动作上界 0.069 |

> **未做、也不建议做**：重训 SuperMimic。WBC 与 MuJoCo 参考实现逐位一致、32k 收敛
> （`sim_rew` 0.849），错的是我们喂给它的观测/动作口径。

## 3. `state` 项被 WM 的速度通道主导（度量修正）

| 项 | 内容 |
|---|---|
| 事实 | 参考动作下位置/朝向/关节误差 1e-4 / 0.035 / 0.030，但 `ang_vel` 1.755、`joint_vel` 2.896；WM 想象速度与参考速度相关仅 **0.077**（位置→速度派生相关 0.075） |
| 原因 | WM 在 PhysX 里带 `action_noise=0.1` 的策略 rollout 上训练，学到的是"仿真机器人真实速度"（幅值约 2×、含噪声），与光滑参考速度不同相；位置仍贴合（1e-4） |
| 修法 | `LossWeights.state_vel`（默认 1.0 保持旧行为）+ `train_joint.py --state-vel-weight`（默认 **0.05**），把 `v/w/qd` 通道降权；`launch_joint_ha.sh` 显式传 `0.05` |
| 验收 | 端口 token 下 `state_term` **2.87 → 0.1816**（低于 mjlab 时代的上界 0.30），reward 不变 0.0703 |

## 验收线（以后改这几处必须重跑）

```bash
# 动作口径：参考动作应 ≈ +0.069 reward / 末步误差 0.05
python3 scripts/probe_action_decomp.py
# token 空间：端口 token 应明显优于标签 token
python3 scripts/probe_port_tokens.py
# 汇总：state_term(port token) 必须 ≈0.18，绝不能停在 2.8
python3 scripts/probe_reward_landscape.py --alphas 1.0
```

## 修复后的起点（2026-10-25 08:48，卡 6 重跑）

```
[eval 0] token_mse 0.1698  state_term 0.9927  rollout_reward 0.0281  hand_mse 0.6512
```

- `token_mse` 现在是"到端口 token 的距离"（起点 0.17，应该**下降**）；
- `state_term` 起点 0.99，向端口 token 的 0.18 收敛；
- `rollout_reward` 起点 0.028，向 0.070 收敛。
- 旧的两轮（口径错误）产物在 `runs/joint_ha_a1/_attempt{1,2}_20261025/`。
