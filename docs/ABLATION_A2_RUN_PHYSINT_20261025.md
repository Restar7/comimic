# A2 运行报告：FM 反传方法（pathwise / checkpointed / truncated / adjoint）

> 自动生成：`python3 scripts/a2_fm_report.py --root runs/ablations/A2_fm_backend`

| 项 | 值 |
|---|---|
| 卡 | 本容器（wscoli2）2× RTX 4090 48 GB，`CUDA_VISIBLE_DEVICES=0,1`，2 卡 DDP |
| 数据 | `data/npz/ha_tokens`（纯 HA；mixed / ft318 被入口硬拒） |
| 打包运动库 | `runs/isaac_motion/ha_tokens/packed.pt`（trackerlab CompactMotionLib） |
| SuperMimic（冻结底模） | `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt` |
| VLA 头 | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（VLM backbone 冻结，不设 `--train-vlm`） |
| 共享超参 | `--batch-size 2 --horizon 8 --hand-loss-weight 0.2 --hand-grasp-boost 5`，同墙钟 `--time-budget-minutes 30` |
| 唯一变量 | `--fm-grad-backend` |

## 1. 曲线（同墙钟比较）

| 臂 | 评测点 | 末步 | 墙钟 s | rollout_reward 首→末 (best) | Δ/min | state_term 首→末 | token_mse 首→末 | hand_mse 末 |
|---|---|---|---|---|---|---|---|---|
| pathwise | 4 | 900 | 1795.02 | 0.04019 → 0.0834 (0.0834) | 0.001318 | 0.06849 → 0.007245 | 0.1698 → 0.1149 | 0.02209 |
| checkpointed | 3 | 600 | 1303.41 | 0.04019 → 0.0808 (0.0808) | 0.001871 | 0.06849 → 0.006228 | 0.1698 → 0.1472 | 0.009317 |
| truncated | 4 | 900 | 1689.37 | 0.04019 → 0.08292 (0.08292) | 0.001377 | 0.06849 → 0.007044 | 0.1698 → 0.174 | 0.009415 |
| adjoint | 4 | 900 | 1780.65 | 0.04019 → 0.08357 (0.08357) | 0.001325 | 0.06849 → 0.007436 | 0.1698 → 0.1213 | 0.00818 |

## 2. 训练侧证据（每臂真的按指定后端在跑）

| 臂 | backend 回声 | 日志步数 | 墙钟退出 | 错误行 | rc |
|---|---|---|---|---|---|
| pathwise | （pathwise 是默认值，代码不打印回声；见 launcher 的 requested 行） | 19 | [budget] 30 min reached after 903 steps (1799s) | 0 | — |
| checkpointed | [fm-grad] backend=checkpointed: per-step torch.utils.checkpoint (exact, ~2x time, O(1) activations) | 17 | [budget] 30 min reached after 828 steps (1799s) | 0 | — |
| truncated | [fm-grad] backend=truncated: first K-tail steps under no_grad (biased, cheap, warm-start style) | 20 | [budget] 30 min reached after 959 steps (1800s) | 0 | — |
| adjoint | [fm-grad] backend=adjoint: discrete adjoint / vjp replay (exact, ~2x time, O(K) tiny states) | 19 | [budget] 30 min reached after 910 steps (1799s) | 0 | — |

显存采样（`gpu_mem.csv`，整轮 A2 的最高值，48 GB 卡）：**gpu0=47624 MiB, gpu1=47110 MiB, gpu2=39894 MiB, gpu3=489 MiB**。每臂起点/终点见 `logs/a2.launcher.log`，采样时间列可切成对应窗口。

## 3. 口径自检（每条都必须成立）

共享底模门禁（来自 `logs/run_1025_v2/chain1.out`，一次、对所有臂生效）：`[gate] Isaac WBC ok: iteration=10500, obs_mode=relative_ori99, interface=token`

| 检查 | 结果 | 证据 |
|---|---|---|
| pathwise: launcher 请求的后端 = pathwise | ✅ | [arm a2_pathwise] requested 2026-10-25T05:54:01+00:00 --fm-grad-backend=pathwise batch=2 horizon=8 |
| pathwise: 日志没有其他后端回声（pathwise 是默认值，不打印） | ✅ | 无回声（符合预期） |
| checkpointed: launcher 请求的后端 = checkpointed | ✅ | [arm a2_checkpointed] requested 2026-10-25T06:27:25+00:00 --fm-grad-backend=checkpointed batch=2 horizon=8 |
| checkpointed: 日志回显 `backend=checkpointed` | ✅ | [fm-grad] backend=checkpointed: per-step torch.utils.checkpoint (exact, ~2x time, O(1) activations) |
| truncated: launcher 请求的后端 = truncated | ✅ | [arm a2_truncated] requested 2026-10-25T06:58:55+00:00 --fm-grad-backend=truncated batch=2 horizon=8 |
| truncated: 日志回显 `backend=truncated` | ✅ | [fm-grad] backend=truncated: first K-tail steps under no_grad (biased, cheap, warm-start style) |
| adjoint: launcher 请求的后端 = adjoint | ✅ | [arm a2_adjoint] requested 2026-10-25T07:30:00+00:00 --fm-grad-backend=adjoint batch=2 horizon=8 |
| adjoint: 日志回显 `backend=adjoint` | ✅ | [fm-grad] backend=adjoint: discrete adjoint / vjp replay (exact, ~2x time, O(K) tiny states) |
| pathwise: 数据 = 纯 HA 699 clips | ✅ | clips=699, frames=634016 |
| pathwise: 底模 = Isaac ckpt iteration 10500 | ✅ | [gate] Isaac WBC ok: iteration=10500, obs_mode=relative_ori99, interface=token |
| pathwise: 手部辅助监督用 HA 标签 | ✅ | [hand] 辅助监督开启: λ=0.2 boost=5.0 labels=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/data/npz/ha_tokens |
| pathwise: 无 OOM / traceback | ✅ | 错误行=0 |
| checkpointed: 数据 = 纯 HA 699 clips | ✅ | clips=699, frames=634016 |
| checkpointed: 底模 = Isaac ckpt iteration 10500 | ✅ | [gate] Isaac WBC ok: iteration=10500, obs_mode=relative_ori99, interface=token |
| checkpointed: 手部辅助监督用 HA 标签 | ✅ | [hand] 辅助监督开启: λ=0.2 boost=5.0 labels=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/data/npz/ha_tokens |
| checkpointed: 无 OOM / traceback | ✅ | 错误行=0 |
| truncated: 数据 = 纯 HA 699 clips | ✅ | clips=699, frames=634016 |
| truncated: 底模 = Isaac ckpt iteration 10500 | ✅ | [gate] Isaac WBC ok: iteration=10500, obs_mode=relative_ori99, interface=token |
| truncated: 手部辅助监督用 HA 标签 | ✅ | [hand] 辅助监督开启: λ=0.2 boost=5.0 labels=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/data/npz/ha_tokens |
| truncated: 无 OOM / traceback | ✅ | 错误行=0 |
| adjoint: 数据 = 纯 HA 699 clips | ✅ | clips=699, frames=634016 |
| adjoint: 底模 = Isaac ckpt iteration 10500 | ✅ | [gate] Isaac WBC ok: iteration=10500, obs_mode=relative_ori99, interface=token |
| adjoint: 手部辅助监督用 HA 标签 | ✅ | [hand] 辅助监督开启: λ=0.2 boost=5.0 labels=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/data/npz/ha_tokens |
| adjoint: 无 OOM / traceback | ✅ | 错误行=0 |

## 4. 判读

- 主指标：`rollout_reward` 的同墙钟斜率（Δ/min）与终值；`token_mse` 只衡量 Euler 链输出、不衡量梯度，不能单独当结论。
- `truncated` 预期更快（步数更多）但更差（有偏）；`adjoint` 与 `pathwise` 应几乎重合，若明显偏离 → 实现 bug，先查 `diff_vla_wbc/fm_grad.py` 而不是下结论。
- `checkpointed` 与 `pathwise` 数学等价，差异应只在吞吐（步数）与显存。
- `qvgm_lite` 需要 chunk critic，真实栈**不提供**（会显式报错），只在 `ablation_fm_grad/` 的 toy 里保留，不得混进本表。
- VLM backbone 两臂都冻结：`adjoint` 会 detach backbone 特征，所以本消融不叠加 `--train-vlm`。

