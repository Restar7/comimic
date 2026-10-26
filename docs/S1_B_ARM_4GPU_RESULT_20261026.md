# S1 B 臂结果（4×4090，warm start=10500，hold token）

> **⚠️ 2026-10-26 06:38 追加诊断，本节结论推翻下面 §"门禁"的表述。**
> 把 6 个 ckpt × 2 套 HA 参考库用同一仪器跑成矩阵后（原始 json 在 `runs/S1_ctrl/`）：
>
> | ckpt | `ha_tokens`（mocap 关节，§1/§3 用的库） | `ha_cmd`（命令流，训练库同源） |
> |---|---|---|
> | ctrl 10500（B 起点） | 140/699, nr **0.5156** | 0/699, nr **0.8166** |
> | B@12600（本次） | 105/699, nr **0.5269** | 1/699, nr **0.8636** |
> | A@16120（A 起点≈16000） | **0/699, nr 0.7094** | — |
> | A@18500（x20 末） | 2/699, nr **0.6272** | 0/699, nr **0.8604** |
> | A@20000（x20b 末） | 1/699, nr **0.6178** | — |
>
> 1. **训练库的 HA 片段逐位等于 `ha_cmd`**（q/v/w/qd 全 0 差，已验证），而两库关节差
>    mean 0.29 rad / qd 0.43 rad·s⁻¹ / max 35。§1/§3 指 `ha_tokens` 是**旧口径**，
>    用它评 B 等于拿没见过的关节轨迹考它 → 105 摔被夸大。
> 2. **在训练同口径（ha_cmd）上：B 起点 0.8166 → B@12600 0.8636（+0.047），
>    与 A@18500 的 0.8604 打平。** 不是"B 训崩了"。
> 3. `ha_tokens` 的 105 摔里 **85 个是 HSI_sit_sofa 一个动作族**（B 起点 97/100 摔，
>    训完 85/100）；A 系的 2 摔也全是 sit_sofa。是单族能力差，不是整体崩。
> 4. **真正的问题：这套 S1 配方在把部署域指标往下拉。** A 系同一血脉
>    16120 → 18500 → 20000 = nr **0.7094 → 0.6272 → 0.6178**（单调下降），
>    同期 ha_cmd 与仿真读数在上升。所以"继续加训到平台"方向可疑。
> 5. 起点结论不变（用 A 系 16000/16120），但**理由是 sit_sofa 族鲁棒性 + 别继续盲加训**，
>    不是"10500 起步训不出来"。



> 2026-10-26 04:52 UTC 起跑 → 05:31 训练结束（12600）→ 05:44 三门禁跑完。
> 配方与 A 臂 `logs/wbc_hold_x20` 逐键相同（`reward_cfg` 已比对一致），唯一差别：起点 10500 vs 16000。

## 起跑核对（交接文档 §2 的 4 行，全部 PASS）

```
[AUDIT][spec] HA_ALIGN=1 usd=g1_29dof_rev_1_0.usd default_q[knee/ankle/elbow]=0.420/-0.230/0.870
              action_scale mean/max=0.3789/0.7500 kp mean=71.4
[AUDIT][prov] obs=paper_obs93 iface=token lib=bones_seed_plus_ha_cmd_x20.pt lib_bytes=8549654882
library: .../bones_seed_plus_ha_cmd_x20.pt  iterations: 12600  compile: 1
Training iterations 10501..12600; policy updates start after 10500.
```
`--freeze_wm` 未出现；`WM_ONLY_ITERS=0` 生效（`pi` 从第一个 log 起就非零）；未设 `SONIC_TOKEN_WINDOW`（默认 hold）。

## 训练（4×8192，耗时 38.8 min，~1.0 it/s）

| | it 10550 | it 12600 |
|---|---|---|
| sim_rew | 0.837 | **0.856** |
| term | 0.01 | 0.01 |
| wm_loss | 0.173 | 0.0369 |
| ep_len | 77 | 354 |
| peak GPU | 17.7 GiB / 48 GB（`nvidia-smi` 26.4 GiB 实占） | 同 |

## 门禁（同协议同仪器，`runs/S1_gates/B_*.json`）

| 门禁 | B@12600 (hold) | 对照 |
|---|---|---|
| HA-699 | **105/699 摔，norm_return 0.5269**，survival 0.5337，cov 0.850 | A@18500 (hold)：2/699，0.6272 |
| BONES-500 | **8/500 摔，norm_return 0.8804**（framew 0.8944），cov 0.984 | A@18500：8/500，0.8798；v1 基线 0.8836/6-500 |
| WM 速度（label） | **qd_corr 0.0954，qd_ratio 3.739**；joint_mse 0.0042，pos_err 0.0045 m，rot_err 2.09°，rollout surv 0.99997 | v1 基线 0.737/0.947 |

失败集合交叉比对（HA-699，`failed` 列表）：

- v1@10500(future 口径)：145 摔；B@12600：105 摔；**重叠 95**
- B 修好了 10500 的 50 个，但新摔 10 个 → 方向对、绝对水平仍差一个档
- v2@16000(future 口径)：0 摔 / 0.7374

## 判读

- **B 在部署域明显更低** → 按 §5 第三行：HA 适配必须保留，主线继续 A（10500 起点不行）。
- 关键点：B 的**仿真读数**（sim_rew 0.856、term 1%）与 A（0.8566、0.27%）几乎一样，
  但 `norm_return` 0.527 vs 0.627、摔 105 vs 2 —— **sim_rew 分辨不出这个差别，只有 HA-699 能**。
- BONES-500 与 A 打平（8/500、0.8804 vs 0.8798）→ B 没遗忘，只是没补上 HA 适配。
- WM 速度 0.095/3.74 明显低于 v1 基线 0.737/0.947；该指标用策略动作做一步预测，
  在"策略与 hold token 不匹配"时会连带塌陷，不宜单独判 WM 坏了。

## 两个记录坑（判读前必须知道）

1. **A 臂的 hold 版 HA-699 不在 `runs/S1_gates`**：`v2hold_HA699_eval.json` 与 `v2hold_BONES500_eval.json`
   md5 完全相同（都是 500 clips / 8 摔 / 0.8798）——被 §3 警告的覆盖坑吃掉了。A 臂的 2/699、0.6272 需从 A 机补回。
2. **`runs/S1_gates/v1|v2` 是 00:10 / 01:56 的旧批**，早于 `sonic_token.py` 的 hold 默认（02:14），
   即它们是 **future 口径**。只有 `B_*` 三条与 A 臂 hold 读数同口径；混表会误判。

## 复现

```bash
bash scripts/rls/supermimic/launch_B_10500_4x8192.sh   # 起跑（12600 自动收）
bash scripts/rls/supermimic/run_B_gates.sh             # 等训练退出后三门禁 + 汇总
```
