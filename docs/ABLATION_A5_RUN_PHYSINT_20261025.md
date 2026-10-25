# A5 运行报告：冻结 SuperMimic，只训 VLA head（Isaac 栈，纯 HA）

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
| head_only | 4 | 900 | 1842.35 | 0.04019 → 0.07963 (0.07963) | 0.001177 | 0.06849 → 0.006605 | 0.1698 → 0.3721 | 0.008258 |
| head_plus_wbc | 4 | 900 | 1869.72 | 0.04019 → 0.08338 (0.08338) | 0.001269 | 0.06849 → 0.007182 | 0.1698 → 0.1161 | 0.02347 |

## 2. 冻结审计（A5 结论成立的前提）

源底模：`/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt`；每臂 `joint_latest.pt` 的 `pi`/`wm` 与之逐张量比较。

| 臂 | pi 张量 | pi 变化张量数 | pi max\|Δ\| | wm 变化张量数 | wm max\|Δ\| | steps | 判定 |
|---|---|---|---|---|---|---|---|
| head_only | 13 | 0 | 0.000e+00 | 0 | 0.000e+00 | 1174 | ✅ policy 未动 |
| head_plus_wbc | 13 | 10 | 4.188e-03 | 0 | 0.000e+00 | 1161 | ✅ policy 在更新 |

## 3. 训练侧证据

| 臂 | 日志步数 | 墙钟退出 | 错误行 | rc |
|---|---|---|---|---|
| head_only | 24 | [budget] 40 min reached after 1174 steps (2400s) | 0 | — |
| head_plus_wbc | 24 | [budget] 40 min reached after 1161 steps (2400s) | 0 | — |

显存采样（整轮最高，48 GB 卡）：**gpu0=41338 MiB, gpu1=41404 MiB, gpu2=39852 MiB, gpu3=39894 MiB**。

## 4. 判读

- 若 head_only 已经能到 head_plus_wbc 同量级的 `rollout_reward`，说明收益主要来自 VLA head，可以省掉 WBC 更新（顺带避免“训练代理与部署执行器不一致”的风险放大）。
- 两臂同墙钟、同 batch/horizon/辅助损失，只有 `--policy-lr` 不同；`wm` 两臂都必须 `max|Δ| == 0`。
- 冻结审计里 `max|Δ| == 0` 是**精确**相等（fp32 逐元素），不是阈值判断。

