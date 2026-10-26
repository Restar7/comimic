# S1 重做（S0 之后）：让 WBC 说**标签 token** 的语言（2026-10-26）

> 触发：`docs/S0_AUDIT_REPORT_20261026.md` §4 推翻了复核 V2 §3.2 的前提。
> 范围：只做 S1（WBC 训练栈）。S2（base_aligned）按 A.2.3 降级为可选；S3 等 S1 的 token 定版。

---

## 0. 一句话

S0.4 的结论是 **"标签 = `encoder_latent`（逐位相等），我们的端口侧车算错了"**，所以 S1 的方向要从
"改 VLA 去说端口语言" 反过来变成 **"让 WBC 说标签语言"**。本轮把这句话变成了可执行的三件事：

1. **token 前端已经逐位对上**（新配方：command = 当前参考帧 hold ×10；anchor = 当前帧 anchor ×10）——
   用 HA 录制数据实测 `max|Δ token − encoder_latent| = 0.0`（3/3 clips）；
2. 但**数据血缘有个必须先修的坑**：`data/npz/ha_tokens` 的 `joint_pos` 是**机器人实际执行轨迹**
   （`robot_qpos_before_decimation`，逐位相等），**不是** encoder 吃的 command 流
   （`encoder_motion_joint_pos_hist`，两者 mean|Δ| = 0.39 rad、最佳 lag **+8 帧**）。
   用现在这个库去配"标签 token"，token 与参考会系统性错位 ≈8 帧；
3. 于是 S1 的下一步不是"直接重训"，而是 **先按 command 流重建 HA 库 + 用新配方产 token，再混 BONES 重训**。

---

## 1. S0 结论对 S1 的影响（逐条）

| S0 结论 | 对 S1 的直接后果 |
|---|---|
| 标签 token ≡ `encoder_latent`（MSE 0 / corr 1.0） | 不需要重标数据、不需要 S2；VLA 的 64 维 head 已经在正确空间 |
| 端口侧车 vs 标签：MSE 0.085、corr 0.057 | **端口 token 路线退役**（含 `data/port_tokens`、旧 `SonicTokenBuilder` 的 future 窗口） |
| `encoder_input` 布局实测（`[4:584)` command、`[601:661)` anchor，[595:601) 当前 anchor） | 前端可以离线逐位复现 → 可以给 BONES-SEED 造同空间 token |
| builder 修不出来 → 走路线 II | ✅ **不需要**：前端已复现，路线 II 与 S2 都不做 |
| pin 08:01 被改导致对照被拒 | 已确认改动是 S1.3 的 `--sm_obs_mode auto`（本会话有意改动）→ `pin_ha_closedloop.py --write` 重新固化，`--check` PASS |

---

## 2. 本轮新增的硬证据（都可复跑）

### E1 前端配方逐位复现（`scripts/verify_sonic_frontend.py` 思路，实测 3/3）

```
command[t] = repeat( encoder_motion_joint_pos_hist[t][-1], 10 )   ‖
             repeat( encoder_motion_joint_vel_hist[t][-1], 10 )     # 顺序就是 encoder 的 58 维
anchor[t]  = repeat( encoder_anchor_window[t][0], 10 )              # 行间 Δ = 0（hold 语义）
其余字段（root_z / lowerbody / vr / smpl / wrist）= 0
=> token == encoder_latent，max|Δ| = 0.000e+00（3/3 clip）
```

- 对照：旧端口配方的 anchor 用 `rot6d` 逐未来帧算 + command 取未来 10 帧，与标签 **corr 0.05**；
- ~~用 `consumed_anchor_rot6d` 代替 `encoder_anchor_window[t][0]` 会差 0.019–0.063，说明 `consumed_*`
  是另一个时刻的量~~ **（2026-10-26 更正，勿再引用）**：逐帧对齐后
  `consumed_anchor_rot6d == encoder_anchor_window[:, 0]`（mean|Δ| ≤1e-4），二者是**同一时刻同一个量**；
  此前那个 0.019–0.063 是**episode/帧错位**造成的（拿别的 clip 或错一帧比）。
- **anchor 公式已解**（见 `docs/SONIC_ENCODER_INPUT_LAYOUT.md` §3）：
  `anchor6 = rot6d(conj(robot_root_orientation) ⊗ human_body_quat_w_aligned)`，
  与录制值 mean|Δ| ≤5.5e-5；用它重算 token 与标签 **exact 1.0000（7/7 任务）**。
  ⇒ 不需要录制的 anchor，**BONES-SEED 也能造同空间 token**（在环 B 模式：`q_robot`=仿真根朝向，
  `q_ref`=参考运动的 heading-aligned 根朝向）。门禁：`tests/test_sonic_token_contract.py`。

### E2 数据血缘：库里的参考 ≠ 标签的参考（必须先修）

| 对比（`HOI_double_desk__tw__ep000000`，2080 帧） | mean\|Δ\| |
|---|---|
| prepared `joint_pos` vs raw `robot_qpos_before_decimation` | **0.000**（逐位相等） |
| prepared `joint_pos` vs raw `encoder_motion_joint_pos_hist[-1]`（command 流） | 0.410（最佳 lag **+8** 帧时 0.390） |
| prepared `joint_pos` vs raw `human_joint_pos` | 0.410 |

⇒ 现状是"**跟踪执行轨迹、却给描述命令轨迹的 token**"。8 帧 = 0.16 s 的前瞻偏差，足以让
`token → 关节目标` 的映射学成一个"提前量"，闭环里表现为过冲/甩。

---

## 3. 已落地的代码改动（本轮）

| 文件 | 改动 | 兼容 |
|---|---|---|
| `trackerlab-supermimic/.../sonic_token.py` | `SonicTokenBuilder` 新增 `window`；默认 **hold**（= 标签空间）；`idx = frame×10` | 旧 ckpt 用 `SONIC_TOKEN_WINDOW=future` 复现 |
| `diff_vla_wbc/sonic_encoder.py` | `build_motion_windows(..., window="hold")`，command 与 anchor 同时 hold | 离线产 token / 联合训练一致 |
| `scripts/verify_sonic_frontend.py` | 前端复现探针（记录配方 + 与 label 对比 + legacy 对照） | — |
| `scripts/stamp_robot_spec.py`、`runner.py`、`evaluate.py`、provider | spec/provenance 落盘 + 断言（S1.1，已完成） | `COMIMIC_ALLOW_NO_SPEC=1` |

---

## 4. 当前 WBC 的数字（口径必须标清）

| 权重 | token 空间 | HA-699 官方评测 | BONES-500 | WM qd_corr/ratio |
|---|---|---|---|---|
| `ckpt_010500_physint_obs93`（v1 基线） | **端口**（future 窗口） | **145/699 摔**，`norm_return 0.520` | `0.8836`，6/500 | 0.737 / 0.947 |
| `ckpt_016000`（混合库续训，端口） | 端口 | **0/699 摔**，`0.7374` | `0.8900`，10/500 | 0.154 / 2.79（未标定） |
| `ckpt_016000_wmvel2`（+1600 步速度标定） | 端口 | 同上（`pi` 逐位相同） | 同上 | **0.753 / 0.968** |

- 这三行是 **端口空间内部自洽** 的结果（训练和评测都用端口 token），所以 0/699 是真的；
  但它们**不能被 VLA 驱动**（VLA 输出在标签空间）——这正是 W1/W5/W6 的病灶。
- 结论：这批数字保留为"**执行器能力上限**"的证据（O1 的端口版），**不进 A1 正式表**。

---

## 5. S1 新执行清单（按顺序，每步有门禁）

### S1-A 按 command 流重建 HA 库（≈40 min，无训练）

1. `scripts/build_ha_command_lib.py`：从 `data/ha_raw/<task>/*.npz` 抽
   `encoder_motion_joint_pos_hist[-1]` / `velocity[-1]`（顺序 = encoder order，**不要 perm**）、
   root quat、`encoder_anchor_window[t][0]`、`encoder_latent`，按 BONES 的 canonical 布局打包；
   clip 名沿用 `HOI_*__tw__epNNNNNN`（与 `data/npz/ha_tokens` 的 source 映射对齐）。
2. **门禁**（已实测，注意口径）：命令流库 + `window="hold"` + **完美跟踪 anchor（identity）**
   → 与 `encoder_latent` 的 corr = **0.973 / 0.498 / 0.403**（3 条 clip），最大偏差 0.5–0.8。
   ⇒ 说明 **anchor 是 token 的重要成分**（它编码跟踪误差），离线拿不到机器人状态时不能指望逐位复现；
   逐位复现只在"用录制 anchor"时成立（E1：max|Δ|=0）。所以：
   - **训练侧**（B 模式）：`SonicTokenBuilder` 直接用仿真机器人 root 算 anchor —— 与部署同分布，正确；
   - **离线侧车/回放**：必须带 anchor（录制值），否则只能当 A 模式冒烟，不得用于门禁。

### S1-B 混合库重训（S1.2）：label 空间

```
库 = BONES-SEED(129,943) + HA-command(699×20 或 ×40)      # 与 PLAN §8 同配方
warm start = logs/wbc_mix2_obs93/ckpt_016000_spec.pt      # token 层要重学，lr 从 2e-5 起
FREEZE_WM=0  NO_CRITIC=1  MAX_EPISODE_STEPS=0  obs93  HA_ALIGN=1
SONIC_TOKEN_WINDOW=hold（默认）
```
**门禁（沿用 S1.2）**：HA-699 官方 `n_failed`、BONES-500 `norm_return` 掉点 ≤0.02、
`eval_wm_velocity` 的 `qd_corr` 不低于 v1（0.737）——每次换代都要重标定 800 步。

### S1-C O1 门禁改成**标签 token 回放**（A.2.2）

- 现有 `scripts/ha_port_token_server.py` 改成读 `data/npz/ha_tokens/*.npz` 的
  `g1_motion_token_64`（或加 `--key g1_motion_token_64`），其余不变；
- 门禁：PPBox / DoubleDesk × 3 seeds、300 步，≥2/3 seeds 不摔。

### S1-D 定版 WBC_v2(label) + 交接

- `runs/wbc_v2_label/wbc_v2.pt`（spec + provenance + token_window=hold 写进 ckpt）；
- `runs/A1_v2/wbc_v2.json` 记录 sha256 / 三道门禁；
- 旧端口 ckpt 一律标 `token_space=port`，**不得与 label 混表**。

### S1-E（P2，非阻塞）负载/推扰随机化（W8）

只在闭环暴露接触类失败（抓取滑落、推箱推不动）时做；当前证据不足，先不做。

---

## 6. 需要拍板

1. **混合比例**：先 `HA×20`（与 §8 首次成功一致）还是直接 `×40`（更快但要重标定更勤）？建议 ×20 起点。
2. **是否保留端口 ckpt 作为 O1 上限参照**：建议保留（它证明执行器本身能 0/699 摔，问题只在 token 口径）。
3. 旧 `data/port_tokens/*` 是否删除：建议**保留但标记 deprecated**，A/B 用。
