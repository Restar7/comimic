# 评审材料：Δtoken 部署链路一致性审计（会话 01 结果）

> 写于 2026-11-07。**用途**：把本轮审计的全部事实、数字、修了什么、还剩什么，
> 一次性摊给外部评审（GPT / 人），供判断"该怎么改"。
> 前置材料：`docs/REVIEW_TWO_ROUTES_BPTT_VS_DISTILL_20261107.md`（训练路线 A/B 的设计与结果）。
> 本文只讲**部署链路一致性**：训练学到的 Δtoken 修正，到底有没有按训练语义在真机链路上执行。
>
> 代码基准：分支 `harness/01-deployment`，基线 `a4bd1ff`，提交 `9165043`。
> 原始产物：`runs/harness_01_deploy/`（`decoder_alignment.csv`、`metrics.json`、`test.log`、
> `runtime_contract.md`、`issues.md`、`report.md`）。

---

## 0. 一句话结论

**修复前：Δtoken 在真实链路上"装不上 + 装错"，所以任何"Δ 没效果"的观测都是假失败。**
**修复后：语义链条正确且可逐位对拍（数值差 ≤2e-6），但"能不能提高成功率"仍未实测（NOT_RUN）。**

---

## 1. 系统全貌（评审需要的最小上下文）

```
相机 → VLA(GR00T 2B) → 78 维 = [token64 | hand14]
                          ↓
        token' = token + Δtoken(token[:,0], feat(state64))     ← 我们唯一的改动点
                          ↓
   SONIC release decoder（ONNX，obs_dict[1,994] → action[1,29]）
                          ↓
        q_target = action × G1_ACTION_SCALE + SONIC_DEFAULT_POS
                          ↓
                        PD ⊂ Isaac
```

* `state64 = [root_rot6d(6) | dof_pos(29) | dof_vel(29)]`，由第三方客户端发送，**不改协议**。
* 训练侧（BPTT）不经过 ONNX：用 `SonicDecoderTorch` —— ONNX 图的**逐算子 torch 复刻**，
  权重逐位拷贝、冻结、可微。所以"代理执行器"就是真解码器本身（这一点本轮被数值确认）。
* Δtoken 网络：`(token64 ⊕ feat) → Linear → SiLU → Linear → SiLU → Linear(64)`，2×1024。
  两套输入口径：`feat_dim=35`（`rot6d+dof_vel`，旧）/ `feat_dim=64`（整个 state64，第 2 刀）。
  幅度约束 `apply_norm_cap`（tanh 软上限，`cap=0` ⇒ 恒等）。

---

## 2. 本轮审计的问题清单与结论

| # | 问题 | 结论 | 状态 |
|---|---|---|---|
| 1 | 训练/推理的 Δ 网络是否完全一致？ | 是（同一份权重，严格加载；sha256 已记录） | PASS |
| 2 | 输入状态是否完全一致？ | **否**，29 维关节块顺序错（训练 STANDARD / 部署 HA_V31） | 已修 |
| 3 | token chunk 的修正方式是否一致？ | 是（都用 chunk 第 0 帧算 Δ，广播到整段；不碰 hand14） | PASS |
| 4 | 训练代理 WBC 与真实 SONIC decoder 行为是否一致？ | 是（同一张图，`max|Δa|=1.9e-06`）；差异只在"观测怎么拼" | PASS |
| 5 | Isaac 真实执行时到底哪个 provider？ | **SONIC**（实测 88/88 日志）；但导出责任在调用方，叶子脚本单独跑会静默降级 | 已加门禁 |
| 6 | reference 库 / 关节索引 / 状态坐标系是否一致？ | 坐标系一致；**帧索引有真 bug**（pp_box 从 246502 行起，默认 0） | 已加断言 |
| 7 | checkpoint 是否按预期加载、有无静默回退？ | **修复前：64 维 ckpt 直接崩**；现在严格加载 + 显式"未启用"日志 | 已修 |

---

## 3. 事实（全部可复现，带数字）

### 3.1 三套关节顺序（后面一切对照的基础）

| 名字 | 用在哪 | 定义位置 |
|---|---|---|
| `STANDARD_29`（= MJCF = trackerlab） | **训练**：motion library / 转移 / WM / 奖励 / Δtoken 特征 | `joint_contract.py:38` |
| `HA_V31_29`（= HA "canonical" = Isaac 资产） | **部署**：`state64[6:35]`、VLA `state46[:29]`、SONIC decoder I/O | `joint_contract.py:70` |
| SONIC 994 观测布局 | 解码器内部（块优先、块内旧→新，历史 10 帧） | `sonic_decoder.py:60-70` |

置换：`x_std = x_ha[p]`，`x_ha = x_std[argsort(p)]`，

```
p = [ 0  3  6  9 13 17  1  4  7 10 14 18  2  5  8 11 15 19 21 23 25 27 12 16 20 22 24 26 28]
```

**三条独立证据**（不是猜的）：
1. `scripts/build_ha_command_lib.py:60-64`：`q_cmd = encoder_motion_joint_pos_hist[...][:, inv]`，
   注释写"命令流（encoder 顺序）→ **MJCF 顺序**"。
2. 数据驱动：训练库 `packed.pt` 的 `q` 块与录制里 HA 序的机器人 q，在**该置换下** 25/29 维最佳
   匹配落在 `p` 上（最高相关 0.99）；不置换时平均值相关只有 0.16（有回归测试）。
3. 部署 `state64[6:35]` 与录制 `robot_qpos_before_decimation` **逐位相等**（`max|Δ|=0.0`），
   而 provider 送进 state64 的正是 `robot.joint_pos[0, self._sonic_idx]`（HA 序）。

### 3.2 真实 checkpoint 的事实

| 项 | `runs/bptt1/vla_000050.pt` | `runs/gt7fix/vla_final.pt` |
|---|---|---|
| ckpt sha256（前 16） | `32b9be2ceba19ac0` | `670a220ff6f4f7bc` |
| Δ 权重 sha256（前 16） | `af64b14d47a8f80f` | `98cc6ef6e45e3b2b` |
| 首层 `net.0.weight` | `(1024, 128)` | `(1024, 128)` |
| metadata 原文 | `{hidden:1024, init_std:4e-4, depth:2, cap:0}` | 同 |
| **metadata 里有 feat_dim 吗** | **没有** | **没有** |
| 正确推断 | `feat_dim=64`（state64 全量），params 1,247,296 | 同 |
| **旧服务端路径** | **崩**：`size mismatch net.0.weight [1024,128] vs [1024,99]` | **崩**（同） |

⇒ 训练保存端**不写 `feat_dim`**（`v3_step.py:489-494`），而部署端写死 35 维。
当前 BPTT 配置正是 `DELTA_TOKEN_FULLSTATE=1` ⇒ **每一份真实产物都装不上**。

### 3.3 特征错序的量化影响（真实 ckpt，真实线上 `state64`）

| 指标 | bptt1 | gt7fix |
|---|---|---|
| 特征向量差 `mean|Δ|` / `max|Δ|` | 0.4764 / 7.3332 | 同（与网络无关） |
| 网络输出方向 `cos(Δ_correct, Δ_wrong)` | **+0.7472**（p05 +0.3810） | **+0.8075**（p05 +0.5222） |
| `‖Δ_wrong‖/‖Δ_correct‖` | 0.9879 | 1.0855 |
| 经真实 decoder 后 `cos(Δa_correct, Δa_wrong)` | **+0.7830** | **+0.5437** |

### 3.4 代理执行器 ↔ 真实 SONIC decoder 对拍（核心实验）

数据：`data/ha_raw/.../…_sonic_1776268633456387.npz` 里的 `decoder_obs`（**部署时真正喂
ONNX 的 994 维**），24 帧。ONNX sha256 `c7241a12…`（40,900,688 B，release 变体）。

**E1 图级等价**
```
torch 复刻(训练) vs onnxruntime(部署) : max|Δa| = 1.907e-06, mean = 2.208e-07
torch 复刻 vs 录制时的真实动作         : max|Δa| = 3.815e-06, mean = 3.259e-07
```

**E2 token 有限差分**（δ ∈ {0.01,0.03,0.08} × 8 个单位方向 × 24 帧）
```
cos(Δa_proxy, Δa_sonic) 最小 = 0.99999975
‖Δa_sonic‖/‖Δa_proxy‖ ∈ [0.99991, 1.00010]
```
⇒ token 通道的 Jacobian 没有被"代理"歪掉。

**E3 观测拼装口径的敏感度**（同一物理状态，只改一种拼法）

| 口径错误 | ‖Δa‖ 平均 | 最大单关节 | **Jacobian cos 均值/最小** | ‖J‖比 |
|---|---|---|---|---|
| 关节序按 STANDARD 拼 | **1.0552** | 6.8344 | **+0.4855 / +0.3640** | 0.313 |
| q 块忘减 `SONIC_DEFAULT` | 0.6797 | 4.1357 | +0.7652 / +0.7440 | 1.600 |
| `last_action` 错一帧 | 0.0312 | 0.3060 | +0.9986 / +0.9944 | 1.000 |
| token 落 1/16 格点 | 0.0000 | 0.0000 | +1.0000 | 1.000 |

⇒ **绝对动作误差小 ≠ Jacobian 相同**：`last_action` 错一帧绝对误差只有 0.031，但方向已偏离 0.14%。

**E4 真实 Δtoken 经真实 decoder 的落地量**

| 臂 | ‖Δtoken‖ | ‖Δa‖ | RMS/关节 | 动作增益 | cos(正确序,错误序) | 最大关节误差 |
|---|---|---|---|---|---|---|
| bptt1 | 0.1146 | 0.2903 | 0.0545 | 2.53 | **+0.7830** | 0.252 |
| gt7fix | 0.0473 | 0.1137 | 0.0222 | 2.40 | **+0.5437** | 0.157 |

换成物理单位（SONIC 平均动作尺度 0.3814 rad/unit）：

| 臂 | RMS/关节（弧/度） | 相对你们自己的"过冲"判据 `1.5η=0.225` |
|---|---|---|
| bptt1 | 0.0208 rad = **1.19°** | **1.29×** ⇒ 略过冲 |
| gt7fix | 0.0085 rad = **0.49°** | **0.51×** ⇒ 偏小，闭环大概率看不见 |

**E5 端到端（把录制约 994 反拼成训练侧 930 观测）**
```
重建 994 自检 max|Δ| = 0.0
训练执行器 vs 真实 ONNX（做了关节序置换） = 1.43e-06
                     （忘了置换）           = 8.2062
```

---

## 4. 我们修了什么（本轮提交 `9165043`）

| 文件 | 改动 |
|---|---|
| `diff_vla_wbc/delta_token_io.py`（新） | 唯一加载/特征契约：`feat_dim` 先读 metadata、缺失则从**首层输入宽度**反推；`hidden/depth` 同样反推并与 meta 交叉校验；键集合/逐层形状链/输出宽度三重校验，冲突即抛错；特征按 `ha_v31→standard` 置换；cap 与训练同一函数 |
| `scripts/ha_vla_latent64_server.py` | 用上面的加载器；新增 `--delta-feat-joint-order`；加载失败 `SystemExit`；无 Δ 时显式打印；新增 `runtime_audit()`（解释器/依赖/ckpt+head sha256/Δ spec/cap/feat_dim 全落日志） |
| `scripts/probe_ft_landing.py` | 同一套加载规则（含关节序） |
| `scripts/_deploy_audit.sh`（新） | provider 判定（`sonic/supermimic/mixed/unknown`，不符或判不了即非零退出）；Δ+非 SONIC 执行器硬失败；参考库 clip 起始行断言 |
| `r3_sr_cell.sh` / `ha_isaac_closedloop.sh` / `ha_isaac_batch.sh` / `isaac_replay_2arm.sh` / `isaac_replay_online.sh` | 接入审计；`SONIC=1` 显式导出三件套（不再依赖调用方）；release 权重缺失不再静默降到 low_latency |

**测试**：`test_delta_token_io.py` 25 passed、`test_deploy_audit.py` 14 passed、
`test_sonic_decoder.py` 8 passed（回归）。全部 CPU，日志 `runs/harness_01_deploy/test.log`。

---

## 5. 对历史结论的影响（重要）

| 历史结果 | 是否受本轮影响 | 理由 |
|---|---|---|
| base 58.3% / v13 30.8% | **不受影响** | 那两批 `head=none` / `head=runs/joint_v13/vla_final.pt`，ckpt 里**没有** `delta_token_state`，从未走 Δ 路径 |
| 真物理 ±δ 判据（96/96、63/64） | **不受影响** | 走的是 WM 梯度 + Isaac，不经 Δtoken |
| Δtoken 各条训练曲线（cos/‖Δ‖） | 数值本身有效，但**"部署没效果"的推论作废** | 因为部署端根本装不上/装错 |
| 闭环 dump 的漂移分解 | **仍作废** | dump 的 q 是 HA 序、库是 STANDARD 序，逐帧对应未复核（BLOCKED） |

---

## 6. 还没解决的问题（评审可以直接打）

1. **BLOCKED-1**：Δtoken 的真实闭环成功率 **NOT_RUN**。本会话不跑 GPU 仿真（机器上已有别人的
   Isaac 进程）。⇒ "有没有帮助"目前只能给到"链路口径正确"这一层。
2. **BLOCKED-2**：闭环 `dump/steps.jsonl` 的 `q` 与参考库帧的逐帧对应关系没复核。强怀疑就是
   第 3.1 节那个置换，但未逐帧验证。影响：任何用 dump 做的漂移分解。
3. **BLOCKED-3**：训练 BPTT 的 decoder 观测来自 **WM 预测状态**，部署是实测状态；这部分误差
   对 Jacobian 的影响**没量化**（E3 只量化了"拼装口径"）。
4. **未修**：`SM_PACKED`（奖励参考）与 `SM_TOKEN_LIB`（token 参考）是两条独立默认路径；
   `rotlocal6` 的锚点训练用"参考库首帧"、部署用"机器人首请求"；`vla_meta.json` 缺失时
   FM 步数/量化只 WARN；`token_quant` 与 Δ 的施加顺序训练/部署相反（当前 `none` 不触发）；
   `data/npz/ha_tokens` 的 `joint_pos` 数值是 HA 序但 `joint_names` 写 STANDARD（数据集陈旧）。

---

## 7. 想请评审回答的问题

1. **在链路已经对齐的前提下**，Δtoken 这条路还值不值得投？
   已测：`‖Δa‖` 对应 **1.19°/关节**（bptt1，1.29× 过冲阈）/ **0.49°/关节**（gt7fix，0.51×）。
   而 base 的 flow-matching 采样噪声本身就有 ~0.06–0.15 m 的闭环差异。
   **1°/关节 的系统性偏移，在 500–1000 步的 episode 里能改变成功率吗？还是必然被噪声吞掉？**
2. **功率问题**：按 `n = 7.85·(p₁q₁+p₂q₂)/Δ²`，58.3% 基线下 +20pp 要约 **91 集/臂**、
   +10pp 要约 **361 集/臂**。我们只有 4–6 集。**有没有更便宜的中间判据**能先筛掉"注定看不见"
   的臂？（现在有：30 秒的 `probe_ft_landing.py` 给 `‖Δa‖/1.5η` 与跨臂 cos）
3. **Δ 的幅度该怎么定？** 现在 bptt1 过冲 1.29×、gt7fix 欠幅 0.51×，两条臂都"不对"。
   按你们 H=20 的真机结论（WM 高估 1.8× ⇒ η=0.08），落到部署侧的合适幅度到底是
   `1.5η` 还是别的？是否有更稳的口径（例如按 `‖Δa‖` 而不是 `‖Δtoken‖` 做 cap）？
4. **BLOCKED-3（WM 预测观测）值不值得修？** 部署用实测、训练用 WM 预测，理论上会让
   ∂a/∂token 在训练时评估在"想象的观测"上。要不要在训练里给 obs 注入实测噪声 / 用混合观测？
5. **I-8（quant 与 Δ 顺序）**：如果以后开 `TOKEN_QUANT=ste`，训练是"先量化再加 Δ"、
   部署是"先加 Δ 再量化"。哪个才是对的？（我们倾向部署口径，因为部署的 token 必须落在格点上）

---

## 8. 复现命令

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt01   # 分支 harness/01-deployment
ROOT=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

python3 -m pytest tests/test_delta_token_io.py -q      # 25 passed
python3 -m pytest tests/test_deploy_audit.py -q        # 14 passed

python3 scripts/verify_delta_token_deploy.py --root $ROOT \
  --arms bptt1=runs/bptt1/vla_000050.pt gt7fix=runs/gt7fix/vla_final.pt \
  --out $ROOT/runs/harness_01_deploy/delta_token_verify.json

python3 scripts/probe_decoder_alignment.py --root $ROOT --frames 24 \
  --arms bptt1=runs/bptt1/vla_000050.pt gt7fix=runs/gt7fix/vla_final.pt
```

---

## 9. 术语表（给外部评审）

| 词 | 含义 |
|---|---|
| Δtoken | 命令层残差网络：`token' = token + g(token, state)`，输出 64 维加到运动 token 上 |
| η | 蒸馏标签的目标幅度，`η=0.08`（token RMS）= 0.15/1.8，已按"WM 高估 1.8×"折过 |
| `1.5η` | 你们自己的"过冲"判据：`‖Δa‖ > 1.5η = 0.225` 视为不可部署 |
| feat_dim 35 / 64 | Δ 网络的输入特征宽度：`rot6d+dof_vel` / 整个 `state64` |
| rotlocal6 | `rot6d(conj(heading(q_init))·q_now)`，与线上 `state64[:6]` 同定义 |
| obs994 | SONIC decoder 的输入：`[token64 \| ω(10×3) \| q−default(10×29) \| qd(10×29) \| a_prev(10×29) \| g(10×3)]` |
| 1.5η 之外的两个判据 | `残差/均值`（跨窗口状态相关性）与 `grad_cos`（Δ 与 ĝ 的余弦） |
