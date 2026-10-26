# S2（阶段 A：base_aligned / 路线 II）计划 v2 — **判定：退役**（2026-10-26，S0 回填后）

> 依据：`docs/JOINT_FINETUNE_REVIEW_REDESIGN_V2_20261026.md` 附录 A（S0 全部完成后的回填）
> + `docs/S0_AUDIT_REPORT_20261026.md` §4。
> 本文替换 v1（v1 的前提是"标签来自不可复现的前端 ⇒ 只能让 VLA 说端口语言"）。
> 证据等级：✅ 本机实跑逐位数字；🟡 需上机；❌ 结论不成立。

---

## 0. 一句话

**S2 不做了，也不是"可选"——它被证明没有收益。** 因为标签 token（= `encoder_latent`）
在训练环里能用**当前参考帧**逐位复现（`A`/`C` 两个实验 exact_frac = 1.0000），
所以"让 WBC 说 VLA 的语言"（S1 的正确路线）是可行且更省的；把 VLA 重标成端口语言既没必要，
端口侧车本身还有三处口径错误（下文 §2）。

v1 里已经建好的东西（重标注脚本 + `data/ha_sessions_port/` + 官方微调 launcher）
**保留为唯一 fallback**，但**现在不跑**。

---

## 1. 判定依据（本机实跑，`scripts/probe_token_reproduction.py`）

产物：`runs/S2_gates/g0b_token_reproduction.json`。encoder = `diff_vla_wbc.sonic_encoder.G1MotionEncoder`
（权重从部署 ONNX 的 g1 分支抽出，4,231,232 参数；另有一份已验证与 onnxruntime 逐位一致的 pytest）。

| clip | T | **A**：录制 `obs[4:584]`+`obs[601:661]` → encoder vs `encoder_latent` | **C**：只用当前参考帧重建 → vs latent | **D**：端口口径（未来窗口+PERM+恒等 anchor）vs latent | 端口侧车 vs label |
|---|---|---|---|---|---|
| HOI_double_desk__tw | 2080 | max 0.0 / exact **1.0000** | max 0.0 / exact **1.0000** | max 1.1 / exact 0.11 | MSE 0.0854, corr 0.057 |
| HOI_pp_box__yb | 1917 | max 6.2e-2（1 个 FSQ 格）/ exact **1.0000** | 同上 **1.0000** | max 0.69 / exact 0.12 | corr 0.192 |
| HSI_vision_navi__yb | 945 | max 0.0 / exact **1.0000** | max 0.0 / exact **1.0000** | max 0.94 / exact 0.09 | corr 0.151 |
| HSI_boxing__zz | 701 | max 0.0 / exact **1.0000** | max 0.0 / exact **1.0000** | max 1.2 / exact 0.11 | corr −0.039 |

`exact_frac` = 64 维里与 `encoder_latent` 逐元素差 < 1e-9 的比例（token 是 FSQ 量化值，
所以"逐位"是硬判据；6.2e-2 = 1/16 一个格）。

### 1.1 布局（实测，与 `docs/SONIC_TOKEN.md` 的"未来 10 帧"语义不同）

```
encoder_input (1762):
  [0]        encoder_index（录制里恒 0，不可用于判分支）
  [4:584]    command：**字段主序** = 10×dof_pos(29) 接 10×dof_vel(29)
             且 10 份 pos 全部 == human_joint_pos[t]（当前参考帧），
             10 份 vel 全部 == encoder_motion_joint_vel_hist[t,45]（当前参考速度）
  [601:661]  anchor：10×6，10 份完全相同 == encoder_anchor_window[t,0]
  [其他]     全 0
```
- **关节顺序**：encoder 要的就是录制里的顺序（HA_V31 / Isaac 资产序）——`C` 实验里**没有做任何置换**
  才能逐位复现；换成 `PERM_MJCF_TO_ISAACLAB` 后 exact 掉到 0.10。
- **anchor 就是 `consumed_anchor_rot6d`**（逐帧、10 份复制）：与 `encoder_anchor_window` 比
  mean|Δ| = 0.0000（3 条 clip 里 2 条逐位相等、1 条 1e-4）。用
  `anchor = repeat(consumed_anchor_rot6d[t])` 复现 token：exact_frac **0.9996–0.9999**
  （差的 0.01% 是 FSQ 边界上的 1 格，6.25e-2）。它**不是** `rot6d(q_robot, human_body_quat_w)`
  （mean|Δ| 0.3–0.75），是"机器人根朝向 vs 参考/目标根朝向"的在环相对朝向。

⇒ **标签 token = f(当前参考关节位, 当前参考关节速, 当前相对朝向)**，逐帧、逐位可复现：

```python
token[t] = encoder(                       # G1MotionEncoder，权重取自部署 ONNX
    command = concat([ref_pos[t]] * 10, [ref_vel[t]] * 10),   # 字段主序，当前帧复制，不置换
    anchor  = concat([consumed_anchor_rot6d[t]] * 10),
)
```

**command 必须来自参考轨迹（目标），不是机器人执行轨迹** ✅：用 raw 的 `human_joint_pos` +
`encoder_motion_joint_vel_hist[-1]` 装 command，与标签 exact_frac = **0.9997–1.0000**（6 条 clip，
max|Δ| = 6.25e-2，即极少数帧差 1 个 FSQ 格）；换成 npz 的 `joint_pos/joint_vel`（机器人**实测**执行轨迹）
只能到 **0.35–0.46** —— 因为 command 携带的是"要去哪"，不是"走到哪"。

**它是一个"当前姿态目标 embedding"，不是未来轨迹 embedding**（这一点对 S1/S3 的口径很重要，见 §4）。

---

## 2. 端口侧车为什么错（根因，三条都实测）

| # | 端口口径（`SonicTokenBuilder` / `build_port_tokens.py`） | 录制口径 | 后果 |
|---|---|---|---|
| 1 | 窗口 = **未来 10 帧升序** `frame+[0..9]` | **当前帧复制 10 份** | token 语义完全不同 |
| 2 | 对已经是 encoder 顺序的 HA 数据再乘 **`PERM_MJCF_TO_ISAACLAB`** | 不置换 | 关节乱序 |
| 3 | anchor = **恒等**（`build_port_tokens.py` 把 `robot_quat` 传成 `body_quat_w[:,0]`，而 `ShimLib.root_quat` 返回同一个 quat ⇒ `rot6d(q,q)=id`） | 每帧相对朝向 ×10 | 丢掉朝向反馈 |

`SonicTokenBuilder` 的 580 维**布局是对的**（field-major）；`diff_vla_wbc/sonic_encoder.py:build_motion_windows`
的布局是 **per-frame 交错**（`cat([pos[idx], vel[idx]], -1)`），这个是错的（另一个人用它的地方要注意）。
两处实现各对一半，合起来就是 corr 0.057 的来源。

---

## 3. v1 已完成/保留的资产（备而不发）

| 资产 | 路径 | 状态 | 说明 |
|---|---|---|---|
| 重标注脚本 | `scripts/relabel_sessions_port.py` | ✅ 通过自检 | 699/699 映射校验（帧数、port shape、`source` 字段、旧 token == `encoder_latent`）、只改 token 列、train/val 双树、stats 只重算 token |
| 端口数据树 | `data/ha_sessions_port/` | ✅ 已生成（train 630 ep / 575,449 帧，val 69 ep / 58,567 帧） | 视频是硬链接（不额外占空间），parquet ~6 GB |
| 官方微调 launcher | `scripts/launch_ha_port_6000.sh` | ✅ 冒烟通过（2 步，loss 0.387，`[AUDIT][data]`/`[AUDIT][s2]` 行齐） | 6000 步 b64 lr1e-5 ≈ 1.6 h |
| 触发判定探针 | `scripts/probe_token_reproduction.py` | ✅ | 本文 §1 的数字由它产出 |

**fallback 触发条件（只有这两条同时成立才回来跑 S2）**
1. 按 §4 的配方修好 builder 后，WBC 在**标签空间**上训不出来（HA 探针 `sim_r` 或闭环不达标），
   且能证明是"条件信号太弱"而不是执行器/奖励问题；
2. 或者发现 HA 的"当前帧重复"确实是录制端 bug，需要恢复未来窗口语义。
→ 满足任一条才重跑：`bash scripts/launch_ha_port_6000.sh`（数据已就绪，不用重标）。

`data/ha_sessions_port/**` 上加了 `README_FALLBACK.md`，防止有人误当正式数据训练。

---

## 4. 交接：让 WBC 说标签语言（S1 的 4 行改动 + 验证）

S1 要在环里产 token（当前是未来窗口 + 置换 + 恒等 anchor），改成与录制/部署一致：

```python
# trackerlab-supermimic/.../supermimic/sonic_token.py :: SonicTokenBuilder.__call__
idx = frame[:, None].expand(-1, nf)            # 1) 当前帧复制 10 份（不是 frame+[0..9]）
...
pos = joint_pos.reshape(n, nf * 29)            # 2) 不做 perm（HA 数据已经在 encoder 顺序）
vel = joint_vel.reshape(n, nf * 29)
anchor = quat_to_rot6d(_quat_mul(_quat_conj(q_robot), q_ref))   # 3) q_robot 必须是仿真机器人 root
                                                                #    （不能和 q_ref 同源，否则恒等）
                                                                #    目标口径 = 录制里的 consumed_anchor_rot6d
```

验证（不训练，`~2 min`）：

```bash
# 1) 录制口径自检：A/C 两个实验都应为 1.0000，D（端口口径）应 ≈0.10
python3 scripts/probe_token_reproduction.py --clips 1

# 2) 修好 builder 后的验收（参考轨迹 + anchor → token，与 npz 标签逐位比）：期望 exact ≥ 0.999
python3 scripts/probe_token_reproduction.py --clips 1 --from-token-npz 8 \
    --out runs/S2_gates/g0b_token_reproduction.json
```

### 4.1 ❗`build_motion_windows` 现在的 `window="hold"` 只修了一半（实测）

S1 刚给 `diff_vla_wbc/sonic_encoder.py:build_motion_windows` 加了 `window="hold"`（窗口语义对了），
但**置换**和**布局**还没改。用 HOI_double_desk 首条 clip 逐位对拍（anchor 用录制值）：

| 变体 | exact_frac vs `encoder_latent` |
|---|---|
| **hold + 字段主序 + 不置换**（正确） | **1.0000** |
| hold + `PERM_MJCF_TO_ISAACLAB` + 字段主序 | 0.1191 |
| hold + 不置换 + **per-frame 交错**（当前实现） | 0.2796 |
| hold + `PERM` + per-frame 交错（当前实现） | 0.1304 |

⇒ 还差两处（各 2 行）：

```python
# 1) 不置换（对 HA 数据；BONES 需要单独验证）
pos = joint_pos_mjcf                                  # 删掉 [..., perm]
vel = joint_vel_mjcf
# 2) 字段主序：先 10 帧 pos、再 10 帧 vel
command = torch.cat([pos[idx].reshape(T, -1), vel[idx].reshape(T, -1)], dim=-1).reshape(T, num_future, 58)
```

改完用 §4 的两条命令验收（期望 A/C = 1.0000，`--from-token-npz` ≥ 0.999）。

⚠️ **BONES-SEED 数据的关节顺序要单独验证**：BONES 的 CSV 是 MuJoCo 分组序，HA 录制是 Isaac 序；
`PERM_MJCF_TO_ISAACLAB` 对 BONES 可能是必需的、对 HA 一定不能加。**按数据源分开，别一刀切。**

---

## 5. 新发现（顺带，影响 S1/S3/A1，必须写进交接）

**F-A｜`data/ha_sessions` 的 `observation.state` 是 HA_V31（Isaac）序，但 `info.json` 声明的名字是 STANDARD_29 序** ✅
- 会话建于 `2026-10-24 05:32`，而 `to_standard(jp_ha, src="ha_v31")` 是 `4adbce1`（`2026-10-26 00:15`）才加的；
- 实测 `session.joints == raw.robot_qpos_before_decimation`（不是 `to_standard(raw)`）；
- ⇒ **现有 VLA 的 state 输入是 HA_V31 序**（与服务端 `--state-order ha_v31` 一致，闭环没问题）；
- ⇒ **但重跑 `prepare_ha_sessions.py` 会静默改成 STANDARD 序**，让 ckpt 与部署两端错位；重跑前必须先决定口径并改 `info.json` 的 names。

**F-B｜`build_motion_windows` 的布局与 encoder 不一致**（§2 表），任何用它的地方（joint 训练的 token 锚定、
`probe_*` 系列）都要按 §1.1 修。

---

## 6. S2 残余工作（不阻塞，等 S1）

只有一项：**B 臂闭环协议**（新口径下的"官方 ckpt + decode + WBC_v2"），现在被两件事挡着：
- S1 的 `WBC_v2`（spec + provenance 齐全、三门禁过）还没定版；
- 闭环 pin 校验失败（`docs/S0_AUDIT_REPORT_20261026.md` §8，`sim_eval_vla.py` 08:01 被外部改动）。

就绪后一条命令即可（口径已锁：`decode=codec`、`fm_steps=4`、`SM_OBS_MODE=auto`）：

```bash
python3 scripts/ha_vla_latent64_server.py \
  --ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --decode codec --fm-steps 4 --port 18080 &
SM_CKPT=runs/wbc_v2/wbc_v2.pt SERVER_URL=http://127.0.0.1:18080 \
  bash scripts/ha_isaac_closedloop.sh PPBox 0
```

---

## 7. 不做的事（v2 明确禁止）

1. ❌ 不跑 `launch_ha_port_6000.sh` 正式训练（S2 已退役；只有 §3 的 fallback 触发才允许）；
2. ❌ 不用端口侧车做 token 锚定 / 评测目标（S3 的锚定目标 = npz 的 `g1_motion_token_64`）；
3. ❌ 不把 `data/port_tokens` 的数据混进任何"正式"表（它是错误装配的产物，只作反例）；
4. ❌ 不在 HA 数据上使用 `PERM_MJCF_TO_ISAACLAB`；
5. ❌ 不重跑 `prepare_ha_sessions.py`（会改关节序），除非先按 F-A 定口径。

---

## 8. 回报模板（本次已填）

```
[S2 判定] 退役（非"可选"）
  证据 A: 录制 1762 输入 → encoder → latent  exact_frac = 1.0000（4 clips）
  证据 C: 仅"当前参考帧×10 + anchor×10"重建  exact_frac = 1.0000（4 clips）
  证据 C': 参考轨迹+consumed_anchor_rot6d → token vs npz 标签  exact 0.9997–1.0000（6 clips）
          （换成机器人执行轨迹 joint_pos 只有 0.35–0.46 ⇒ command 必须用参考轨迹）
  证据 D: 端口口径（未来窗口+PERM+恒等 anchor）exact ≈ 0.09–0.12，corr(侧车, label) 0.057–0.192
[资产] relabel 脚本 699/699 通过；data/ha_sessions_port（train 630/val 69）；launch_ha_port_6000.sh 冒烟通过 → 备而不发
[交接] SonicTokenBuilder 三处口径（窗口/置换/anchor）+ build_motion_windows 布局
[新发现] 会话 state 是 HA_V31 序但 info.json 写 STANDARD 序（重跑会静默改口径）
[S2 残余] B 臂闭环（等 WBC_v2 + pin 修复）
```
