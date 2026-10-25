# A5 运行报告：冻结 SuperMimic，只训 VLA head（Isaac 栈，纯 HA）

> ⚠️ **口径警告（2026-10-25 01:15 补）**：本次 A5（23:12–00:34）跑的是**修复前**的代码——
> ① 条件 token 用 HA 标签 `data/npz/ha_tokens`，而底模 `ckpt_010500` 的条件是 trackerlab 端口 token
> （实测两者 `max|Δ|=1.06~1.19`、相关系数只有 **0.06~0.19**，见 `docs/HANDOVER_SUPERMIMIC_FINETUNE.md` §1①）；
> ② `--state-vel-weight` 仍是 1.0（WM 的速度通道没学会，其 ~96% 能量是不可学噪声，见 §1②）；
> ③ 动作换算是旧的 `a = Δq/action_scale`。
> 因此：**曲线绝对值（含 `token_mse`、`state_term`）不可用**；两臂之差因为同底模、同口径，
> **相对结论仍成立**（`head_plus_wbc` 三项都优于 `head_only` → 不能靠"只训 head"省掉 WBC 更新）。
> §2 的冻结审计（`pi` 逐张量 `max|Δ|`）不受影响。新底模 `logs/ha_token_v1` 验收通过后 A5 需用
> `tokens.npy` + `--state-vel-weight 0`（速度不参与 state 正则，仅诊断）重跑。

> 自动生成：`python3 scripts/a5_report.py --root runs/ablations/A5_only_head`

| 项 | 值 |
|---|---|
| 卡 | 本容器（wscoli2）2× RTX 4090 48 GB，`CUDA_VISIBLE_DEVICES=0,1`，2 卡 DDP |
| 数据 | `data/npz/ha_tokens`（纯 HA；mixed / ft318 被入口硬拒） |
| 底模（WBC + WM） | `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt`（iteration 10500） |
| VLA | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`，VLM backbone 冻结 |
| 共享超参 | `--batch-size 2 --horizon 8 --hand-loss-weight 0.2 --hand-grasp-boost 5`，同墙钟 40 min/臂 |
| 唯一变量 | `--policy-lr/--adapter-lr`：head_only = 0/0；head_plus_wbc = 默认 1e-5 |
| WM | 冻结（`--wm-update-every 0`，两臂相同） |

## 1. 曲线（同墙钟）

| 臂 | 评测点 | 末步 | 墙钟 s | rollout_reward 首→末 (best) | Δ/min | state_term 首→末 | token_mse 首→末 | hand_mse 末 |
|---|---|---|---|---|---|---|---|---|
| head_only | 4 | 900 | 1819.85 | 0.007391 → 0.06701 (0.06701) | 0.001766 | 20.58 → 1.004 | 0.05578 → 1.512 | 0.0088 |
| head_plus_wbc | 4 | 900 | 1824.98 | 0.007391 → 0.0732 (0.0732) | 0.001958 | 20.58 → 0.6693 | 0.05578 → 0.4504 | 0.00372 |

## 2. 冻结审计（A5 结论成立的前提）

源底模：`trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt`；每臂 `joint_latest.pt` 的 `pi`/`wm` 与之逐张量比较。

| 臂 | pi 张量 | pi 变化张量数 | pi max\|Δ\| | wm 变化张量数 | wm max\|Δ\| | steps | 判定 |
|---|---|---|---|---|---|---|---|
| head_only | 13 | 0 | 0.000e+00 | 0 | 0.000e+00 | 1189 | ✅ policy 未动 |
| head_plus_wbc | 13 | 10 | 4.642e-03 | 0 | 0.000e+00 | 1184 | ✅ policy 在更新 |

## 3. 训练侧证据

| 臂 | 日志步数 | 墙钟退出 | 错误行 | rc |
|---|---|---|---|---|
| head_only | 24 | [budget] 40 min reached after 1189 steps (2399s) | 0 | — |
| head_plus_wbc | 24 | [budget] 40 min reached after 1184 steps (2399s) | 0 | — |

显存采样（整轮最高，48 GB 卡）：**gpu0=40732 MiB, gpu1=40680 MiB**。

## 4. 判读

- 若 head_only 已经能到 head_plus_wbc 同量级的 `rollout_reward`，说明收益主要来自 VLA head，可以省掉 WBC 更新（顺带避免“训练代理与部署执行器不一致”的风险放大）。
- 两臂同墙钟、同 batch/horizon/辅助损失，只有 `--policy-lr` 不同；`wm` 两臂都必须 `max|Δ| == 0`。
- 冻结审计里 `max|Δ| == 0` 是**精确**相等（fp32 逐元素），不是阈值判断。

