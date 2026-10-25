# 交接：最新技术结论 + 给 2×4090 的任务清单（2026-10-25 11:15）

> ⚠️ **2026-10-25 口径更新（以 `docs/HANDOVER_MAINLINE_AND_ABLATIONS_20261025.md` 为准）**：
> `--state-vel-weight` 已从 1.0 改为 **0**（速度不参与 joint 训练的 `state` 正则，仅作为 `state_vel` 诊断量打印）；
> reward 不变（环境自带的速度项不动）。本文下面仍写 1.0 的地方按 0 执行。

> **收件人**：2×4090 容器的 DSH。**发件人**：卡 6（wscoli2 共享同一份代码/数据盘）。
> **配套文档（先读）**：`docs/HANDOVER_MAINLINE_AND_ABLATIONS_20261025.md`（wscoli2 出的新标准，
> 口径写死：`ckpt_010500_physint.pt` + 端口 token + `a=Δq` + WM 冻结 + `state-vel-weight 1.0`）。
> 本文只写**那份文档之后新查清/新做出来的东西**，以及建议你们接手的活。

---

## 0. 我们这轮查清的三件事（都有实测数字）

### 0.1 token 空间：不是"我们端口写错了"，是**两个前端**
| 命题 | 结论 | 证据 |
|---|---|---|
| HA 的 `encoder_latent` = 本地 `model_encoder.onnx(它自带的 encoder_input 1762)` | ✅ 成立 | 原文档逐帧 bit-exact |
| 我们端口（640 维自建输入）算的 token = HA 的 `encoder_latent` | ❌ **不成立** | `check_sonic_token.py` → `[FAIL] max|Δ|=1.06` |
| **数据距离**（标签 ↔ 端口 token，纯数据属性、与训练无关，60 clips 实测） | HA **MSE 0.0877**（= 标签方差的 **245%**）<br>ft318 **MSE 0.0054**（= 方差的 **12%**） | 见 §5 复现 |

差 **16 倍**——这解释了两件历史疑案：
1. 为什么 mjlab 时代同样架构没炸：ft318 的标签本来就贴着执行器空间（12%），对齐几乎免费；
2. 为什么 HA 闭环 0/23 成功：标签离执行器空间 245%，等于喂乱码。

**原因**：同一网络、同一权重，但输入装配不同 —— 部署侧用 **过去 10 帧真实机器人 q/qd + consumed anchor + SMPL/wrist（1762 维）**；端口用 **未来 10 帧参考运动 + 参考四元数 anchor（640 维）**。

### 0.2 WM 的速度通道：根因是 Isaac 移植丢了 `explicit_integration`
| 项 | mjlab（`SuperMimic-Reproduced`） | trackerlab/Isaac |
|---|---|---|
| 位置更新 | `p + dt*v_new + H@dp`（`state_spec.py:55,103`，`explicit_integration=True`） | `p + H@dp`（六路完全独立） |
| 后果 | 速度错 → 位置立刻错 → 速度被隐式监督 | 速度可以独立烂掉，不被任何项惩罚 |
| 实测 | — | 动作给 **0** 和给 **2 倍**，预测 `\|qd\|` 只从 1.24 变到 0.99；与参考相关 **0.09**、幅值 **2.5×** |

**这不是 BONES-SEED 没见过 HA 速度**，是结构缺陷；PhysX 的 `action_noise=0.1` 只是让退化更快发生。
→ 已在 `networks.py` 实现显式积分（`step` + `targets` 成对），结构由 ckpt 的
`wm_integration="explicit_dt0.02"` 键自动选择；官方旧 ckpt 无此键 → 自动保持旧语义。
代数互逆检验：`max|step(targets(S,S1))−S1| = 1.19e-7`。

### 0.3 动作换算 `a = Δq`（已进标准，不重复论证）
参考动作 rollout：除以 per-joint `action_scale` → reward **−0.129**/末步误差 1.31；`a=Δq` → **+0.069/0.047**。

---

## 1. 我们这轮新增的产物（可直接用）

| 产物 | 路径 | 验收 |
|---|---|---|
| 显式积分实现 | `trackerlab-supermimic/.../supermimic/networks.py`（`step`/`targets`） | 互逆 1.19e-7 |
| ckpt 结构自动识别 | `diff_vla_wbc/isaac_joint.py`（读 `wm_integration`） | 官方 ckpt→`residual`；physint→`explicit_dt0.02` |
| 本机等价标定 ckpt | `.../logs/bones_seed_token_ha_v2/ckpt_013500_physint.pt` | policy rollout `state_term 0.59`、`reward +0.0757`、`qd_corr 0.90`、`ratio 0.92`（同权重 `state_vel=1.0`；旧结构同口径为 2.87/0.0703） |
| 端口 token 侧车 | `data/port_tokens/ha_tokens/*.npy` | **699/699** |
| eval 分项 | `state_pos/state_ori/state_q/state_vel` | 已在 `train_joint.py` |
| **官方 decoder 蒸馏** | `runs/sonic_decoder_final.pt` + `scripts/distill_sonic_decoder.py` | 蒸馏 `decoder_obs(994) → final_body_action_29dof`：**val_mae 0.031**（动作幅值 0.258，≈12%）、**corr 0.989**（150/699 文件、300 步） |

### 1.1 官方评测链路的关键事实（新发现）
- 官方部署是 **VLA → 64 维 token → release decoder → 动作**，**不是** SuperMimic WBC；
- HA raw 逐帧记录了 `decoder_obs (T,994)`、`decoder_raw_action`、`decoder_target_action`（≡ `final_body_action_29dof`）、`meta_decoder_path`；
- `decoder_obs` 布局已从 release YAML 确认：**token(64) + 10 帧 × [base_ang_vel 3 + joint_pos 29 + joint_vel 29 + last_action 29 + gravity_dir 3] = 994**；
- 官方解码器权重（`model_decoder.onnx`）**不在本工作区**（release 目录只有 yaml）→ 所以我们用记录蒸馏替代；如有真权重，优先用真权重并做一次 val 对齐。

---

## 2. 给 2×4090 的任务（按优先级，任选/并行）

### A. 【推荐】整网微调，让 WBC 直接吃 HA 标签 token（"路线 A"）
目标：把 token 空间**按定义对齐**（数据距离 0），这样官方 VLA 天然合法、A1 的对照臂也合法。

1. `python3 trackerlab-supermimic/scripts/rls/supermimic/prepare_token_data.py --src data/npz/ha_tokens --out data/token_motions/ha`（699 条；本机只跑到 424，需补齐）
2. 训练：`--motion_dir data/token_motions/ha --interface token --resume .../ckpt_010500.pt --lr 5e-5 --iterations ~2500`
   - ⚠️ **必须同时给 WM 速度通道加监督**（参考窗口 + policy 自己的动作 + 归一化速度 MSE，λ 5–10）；
     否则整网训练会把 `physint` 的速度标定**洗掉**（wscoli2 实测：sim 数据续训 300 iter → `qd_corr 0.897→0.095`）；
   - ⚠️ `iterations` 是**总数**（续训写 `start+N`）；`wm_only_iters` 有绝对/相对迭代数的坑（见主文档 §6）。
3. 验收三条：`sim_rew ≥0.84`；参考窗口速度 `corr(qd) > 0.9`、幅值比 ∈[0.9,1.1]；端到端用 policy 动作 rollout `state_term` 明显低于 0.18、`reward ≥0.07`。
4. 产出 ckpt 必须带 `wm_integration` 键（`save_checkpoint` 已支持）。

### B. 【可选】官方 decoder 线路（基线）
- 用 `runs/sonic_decoder_final.pt`（994→29）+ §1.1 的 93 维观测布局，在闭环 env 里做 `VLA → token → decoder → 动作`；
- 先扩到 699 clips / 更多步把 `val_mae` 压到 ≤0.02，或用真 `model_decoder.onnx`；
- 这条只用于**官方基线成功率**，不要和 WBC 那条混着比（执行器不同）。

### C. 【可选】给 A1 补"对照臂"（若不做 A）
如果坚持 WBC 保持端口空间（路线 B），那 A1 的 base 臂（官方 VLA）必摔且原因不同 → 需补一个
**纯监督 token 对齐臂**（只训 VLA 的 token 头，MSE 到端口 token，不用 WM/奖励，约 1–2 h）：
- `base_aligned`（对齐后） vs `ft`（对齐 + 奖励反传）→ 唯一变量才是"奖励反传"。

---

## 3. 验收线（三条，缺一不可）
1. 参考窗口 + policy 动作：`corr(pred_qd, ref_qd) > 0.9`、`|pred_qd|/|ref_qd| ∈ [0.9,1.1]`；
2. 端到端想象 rollout（policy 动作，`state_vel=1.0`）：`state_term` 明显低于 0.18、`rollout_reward ≥ 0.07`；
3. RL 侧：`sim_rew ≥ 0.84`、`term_frac ~0.002`。

## 4. 不要做的事
1. ❌ 用 `ckpt_013500_wmvel*.pt`（无效标定：用 Δq 当 WM 输入，会破坏 policy 一致性，实测 policy rollout 2.87→29.6）。
2. ❌ 把 `physint` ckpt 丢回 trackerlab runner 续训而不加速度监督（标定会被洗掉）。
3. ❌ 把 `token_mse`（对 HA 标签）当保真度 —— 只对**端口 token** 有意义。
4. ❌ 用 mjlab 的权重/解析头跑 HA 闭环（永久禁用）；但**可以**用它做 parity 参考实现。
5. ❌ 用 `pkill -f train_joint.py` 这种宽匹配停任务（会误杀自己的 shell）。

## 5. 复现命令（本文所有数字）
```bash
# 数据距离（标签 vs 端口 token）—— 60 clips，纯 CPU
python3 - <<'PY'
# SonicTokenBuilder(参考 jp/jv/quat) vs npz 的 g1_motion_token_64
# 期望：ft318 MSE≈0.0054（12% 方差）；HA MSE≈0.0877（245%）
PY

# token 空间 parity（仓库自带门禁）
cd trackerlab-supermimic && python3 scripts/rls/supermimic/check_sonic_token.py \
  --skip dyn --npz_dir ../data/npz/ha_tokens      # 期望 [FAIL] max|Δ|≈1.06（未对齐）

# WM 结构 + 速度标定验收
python3 scripts/finetune_wm_velocity_policy.py --steps 300 \
  --out /tmp/ckpt_test.pt                        # 期望 qd_corr>0.9、policy state<0.6、reward≈+0.076

# 官方 decoder 蒸馏
python3 scripts/distill_sonic_decoder.py --files 699 --steps 800 \
  --target-key final_body_action_29dof --out runs/sonic_decoder_final.pt
```

## 6. 本机（卡 6）在跑什么
- 新标准主线联合微调（已重启，旧口径那轮归档到 `runs/joint_ha_a1/_attempt4_oldconfig_20261025/`）：
  `ckpt_010500_physint.pt` + 端口 token 侧车 + `a=Δq` + `--state-vel-weight 1.0` + WM 冻结；
  `TRAIN_PID=3564585`，日志 `logs/joint_ha_a1.log`，产物 `runs/joint_ha_a1/vla_004000.pt`；
  训完自动接力 A1 闭环（`logs/a1_after_train.log`，PID 3565647）。
- 旧口径（013500 + 速度降权 0.05）的观察，仅作对照：`token_mse 0.170→0.308`、`reward 0.028→0.079`
  —— **它在往 WM 喜欢的方向跑，不是往端口 token 跑**，这正是必须切新口径的直接证据。
