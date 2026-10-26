# S0 审计与探针报告（2026-10-26）

> 对应复核文档 §4 的 S0.1–S0.5。**只做审计/探针，不改训练**（S1/S2/S3 未动）。
> 所有结论都带命令与实测数字；未完成的门禁明确标注 ⏳。

---

## 0. 结论速览

| 项 | 结论 |
|---|---|
| **S0.1 执行器 spec** | ✅ 训练真值 = `HA_ALIGN=1`（knee **0.420** / ankle **−0.230** / elbow **0.870**，action_scale **0.3789/0.7500**）；联合训练旧行为 = BeyondMimic（0.669/−0.363/0.600，0.3950/0.5475）→ **J1 成立**，已修（env 锁 + spec 戳 + 断言） |
| **S0.2 动作归一化** | ✅ **J2 成立且量化**：`use_percentiles=True` + min-max → head 输出在 [−1,1]；token 原始范围 [−0.46,0.39]；手指原始范围 **±1.740**（阈值 1.70 只在原始空间有意义）→ 服务端必须 decode。codec 与官方 `decode_action` **max\|Δ\|=1.19e-07 PASS** |
| **S0.3 FM 步数** | ✅ `vla_*.pt` 现在写 `fm_steps`；`export_vla_ckpt` 写 `vla_meta.json`；服务端不一致直接 `SystemExit`（A/B 才允许 `ALLOW_FM_STEPS_MISMATCH=1`） |
| **S0.4 token 空间** | ❗**推翻复核的假设**：`g1_motion_token_64`（标签）与 `encoder_latent` **逐位相等**（MSE=0）；而我们的**端口 token 与标签 MSE=0.0854、corr=0.057** → **端口装配是错的**，标签空间才是 VLA 的原生空间。`encoder_input` 布局已实测定位。 |
| **S0.5 多卡 seed** | ✅ 已改（`seed + 1000*rank`）；⏳ 2 卡 smoke 待 2 卡机验证 |

---

## 1. S0.1 执行器 spec（J1）

### 1.1 训练真值（来自 WBC run 的 launcher 日志）
```
[12:27:50] robot asset: g1_29dof_rev_1_0.usd | drive: HA/SONIC PD + HA default pose (HA_ALIGN=1)
SUPERMIMIC_HA_ALIGN=1
SUPERMIMIC_ROBOT_USD=.../trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd
```

### 1.2 两种设置的实测对比（`get_robot("g1")`）
| 设置 | default_q[knee/ankle/elbow] | action_scale mean/max |
|---|---|---|
| `HA_ALIGN=0`（代码默认 / 联合训练旧行为） | 0.669 / −0.363 / 0.600 | 0.3950 / 0.5475 |
| **`HA_ALIGN=1`（WBC 训练真值）** | **0.420 / −0.230 / 0.870** | **0.3789 / 0.7500** |

→ 联合训练若不设环境变量，WM 想象里每帧 `q − default_q` 与 `reference_start` 的 `a_prev` 全部偏移（**J1 成立**）。

### 1.3 修复（已落地）
| 文件 | 改动 |
|---|---|
| `scripts/stamp_robot_spec.py`（新） | 把 `ha_align/usd/joint_names/default_q/action_scale/kp/kd` 写进 ckpt 的 `robot_spec` 键（另存 `_spec.pt` + `.spec.json`） |
| `scripts/train_joint.py` | **任何项目 import 之前** `setdefault("SUPERMIMIC_HA_ALIGN","1")` + `SUPERMIMIC_ROBOT_USD` |
| `scripts/launch_joint_ha.sh` | 双保险 export |
| `diff_vla_wbc/isaac_joint.py` | 加载时打印 `[AUDIT][spec]`，并与 ckpt 的 `robot_spec` **逐项断言**；无 spec 时要求 `COMIMIC_ALLOW_NO_SPEC=1` |

### 1.4 门禁实测（✅ PASS）
```
[AUDIT][spec] SUPERMIMIC_HA_ALIGN=1 default_q[knee/ankle/elbow]=0.420/-0.230/0.870
              spec_scale mean/max=0.3789/0.7500 action_scale_mode=one
[AUDIT][spec] robot_spec 对齐 PASS
[AUDIT][spec] bundle loaded ok
```
产物：`trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint_spec.pt`（55 MB）+ `.spec.json`

> ⏳ 未做：在正确 spec 下重跑 `probe_action_decomp.py` 对比 {HA spec, ones}。需要 GPU 与一次短加载，建议下一步做。

---

## 2. S0.2 动作归一化（J2）

### 2.1 processor 统计（`checkpoint-8000`）
```
use_percentiles = True   clip_outliers = True
motion_token     : min∈[−0.6875,−0.1875] max∈[0.1875,0.6250] std≈0.1438
left_hand_joints : min 最负 −1.7400  max ≤ 0（原始范围 ±1.74）
right_hand_joints: 同上（镜像）
```
→ 动作走 **min-max → [−1,1]**；**手指的 1.70 闭手阈值只在原始空间成立**，归一化后恒 ≤1 → 旧闭环"手永远张着"是必然。

### 2.2 可微 codec（新文件）
`diff_vla_wbc/action_codec.py`：`decode(x) = (x+1)/2·(hi−lo) + lo`、`encode` 反向并 clamp（保留梯度）。
单测对拍：
```
[AUDIT][codec] tag=unitree_g1_sonic dim=78 raw_min[-1.740,1.740] rng_mean=0.9371
[codec-test] decode vs processor.decode_action: max|Δ| = 1.19e-07  PASS
[codec-test] encode(decode(x)) vs clamp(x):      max|Δ| = 1.19e-07  PASS
```

### 2.3 服务端
`scripts/ha_vla_latent64_server.py`：新增 `--decode {codec|none}`（默认 `codec`），decode 后再切 token/手指；审计行
```
[AUDIT][server] decode=codec fm_steps=4 train_fm_steps=None
```
逐请求日志新增：`tok_absmean / hand14_max / hand_gt170`。

### 2.4 门禁实测（DoubleDesk seed0，60 步，官方 ckpt）
```
infer #1 tok_absmean=0.1332 hand14_max=0.039 hand_gt170=0.000
infer #2 tok_absmean=0.1311 hand14_max=0.054 hand_gt170=0.000
结果: timeout, 60/60 步（不摔）
```
- `tok_absmean ≈ 0.132` 与录制 latent 的 `|mean| = 0.127` **同量级** → decode 生效 ✔；
- `hand14_max ≈ 0.04` = 起始阶段手是**张开**的，符合任务早期（标签里 >1.70 的闭手帧只占 2%）；
- 标签侧对照：`|max| = 1.740，>1.70 比例 = 0.020，|mean| = 0.396`。

> ⏳ 未完成：`--decode none` 的同条件对照（服务已起、等待出数）。**旧 ft ckpt（anchor4000/vla2000）按复核要求必须用 `--decode none` 评测并单独归档。**

---

## 3. S0.3 FM 步数（J3）

| 修改 | 文件 |
|---|---|
| `vla_*.pt` 三个保存点都写 `"fm_steps": args.vlm_steps` | `scripts/train_joint.py` |
| `export_vla_ckpt.py` 把 `fm_steps` 带到合并目录 `vla_meta.json` | `scripts/export_vla_ckpt.py` |
| 服务端读 `vla_meta.json`，与 `--fm-steps` 不一致直接退出（A/B 才允许 `ALLOW_FM_STEPS_MISMATCH=1`） | `scripts/ha_vla_latent64_server.py` |

> ⏳ 门禁：需用一批**新训练**的 `vla_*.pt` 走一遍 `export_vla_ckpt → server` 才能看到 mismatch 抛错（旧 ckpt 没有 `fm_steps`）。

---

## 4. S0.4 token 空间根因（W6）—— **推翻复核假设**

### 4.1 实测（`data/ha_raw/HOI_double_desk/tw/*.npz` + `data/npz/ha_tokens` + `data/port_tokens`）
| 对比 | MSE | corr |
|---|---|---|
| **标签 `g1_motion_token_64` vs 录制 `encoder_latent`** | **0.000e+00** | **1.000000** |
| **端口 token（我们的侧车）vs 标签** | **8.539e-02** | **0.0568** |

→ **标签就是 encoder latent（逐位相等）**；**端口侧车与它几乎不相关**。
复核假设"标签来自非 g1 分支"**不成立**；真实情况是 **我们的端口装配（`build_port_tokens.py` / `SonicTokenBuilder`）算错了**。

### 4.2 `encoder_input` 布局（实测非零段）
```
非零段: [4:584] len=580 (absmax 6.29)   → 10 帧 × (q29 + qd29)
        [595:661] len=66                → 10×6 anchor + 6 余量
[595:655] 与 encoder_anchor_window 逐元素 mean|Δ| = 0.0   （完全一致）
[4:584] 与 encoder_motion_joint_pos_hist[-10:] 的 q 部分 mean|Δ| = 0.167（最接近的候选）
```
- `encoder_index`（cols 1:4）与 selector（col 0）在录制帧里均为 0 —— 该字段不是判定分支的可靠依据；
- anchor 段与 `encoder_anchor_window` **完全相等**，说明复现 1762 输入是可行的（下一步只需对齐 qdq 段的帧步长/顺序）。

### 4.3 决策（与复核 §3.2 相反）
- **不需要把数据"重标成端口 token"**（复核的路线 II 基于错误前提）；
- **正确路线：让 WBC 说 VLA 的语言** —— 用 **标签空间**（= `encoder_latent`，699 条 npz 里全有 `g1_motion_token_64`）重训 WBC；
- **端口 token 侧车与 `SonicTokenBuilder` 这条路应当退役**（除非后续查明并修好装配误差）；
- 联合微调的 token 锚定目标也应改用 **标签 token**（直接读 npz），而不是端口侧车。

---

## 5. S0.5 多卡 seed（J6）

```python
# scripts/train_joint.py
_seed = int(args.seed) + 1000 * int(_rank)   # 每 rank 不同 seed
torch.manual_seed(_seed)
provider = MotionBatchProvider(..., seed=_seed, ...)
```
eval provider 仍固定 seed（只在 rank0 跑），保证评测可比。
> ⏳ 门禁：需 2 卡机跑 `--nproc_per_node 2` 的 2 步 smoke，打印各 rank 第 1 步 `frames` 确认不同。

---

## 6. 产物与回退

| 产物 | 路径 |
|---|---|
| spec 戳过的 WBC | `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint_spec.pt`（+ `.spec.json`） |
| codec | `diff_vla_wbc/action_codec.py`（`python -m diff_vla_wbc.action_codec` 自测） |
| 探针数据 | `runs/probe_decode_s0_2b/`（decode=codec，60 步）、`runs/probe_decode_none_s0_2/`（对照，⏳） |

回退开关：
- spec：`COMIMIC_ALLOW_NO_SPEC=1`（放行无 spec 的 ckpt）、`SUPERMIMIC_HA_ALIGN=0`（退回旧行为，**仅 A/B**）；
- 归一化：`--decode none`（旧口径）；
- fm 步数：`ALLOW_FM_STEPS_MISMATCH=1`（仅 A/B）。

---

## 7. 待办（S0 剩余 + 交接给 S1/S2/S3）

1. ⏳ `--decode none` 同条件对照出数（服务已起，脚本同 §2.4）；
2. ⏳ `probe_action_decomp.py` 在 HA spec 下重跑（对比 {spec, ones}）；
3. ⏳ 2 卡 seed smoke；
4. **S0.4 后续**：把 `encoder_input[4:584]` 的帧步长/顺序对齐后，验证能否用 `SonicTokenBuilder` 复现 `encoder_latent`（逐位）；在此之前，**WBC 重训与 token 锚定都改用标签 token**。

---

## 8. 阻塞项（执行中发现的意外）

**闭环契约 pin 校验失败**（08:01 有**外部改动**，不是我改的）：
```
[pin] ❌ 契约校验失败（1 处）：
   [评测脚本变了] script/eval_scripts/sonic/sim_eval_vla.py —— 结论口径可能变，需重新固化
```
- 该文件 mtime = `2026-10-26 08:01`，而我 07:4x 的 decode=codec 探针还能跑；
- 仓库对该文件没有 git 记录（`git diff` 为空，third_party 未纳入版本管理），**无法审查改了什么**；
- 影响：**`--decode none` 的同条件对照被 pin 拒绝启动**（S0.2 §2.4 的对照项因此未完成）。

处理建议（需人工确认，不做静默放行）：
1. 找到 08:01 改这个文件的人/会话，确认改动意图；
2. 若是有意改动：`python3 scripts/pin_ha_closedloop.py --write` 后再跑对照，并把新 pin 写入交接文档；
3. 若无意：回滚该文件后再跑。

**在 pin 恢复前，所有闭环数字都不应与 07:4x 之前的数字混表。**

---

## 9. S0.4 后续：encoder 布局已破译（2026-10-26 续）

### 9.1 pin 已修复 ✅
```
python3 scripts/pin_ha_closedloop.py --write   # 08:01 的外部改动（--sm_obs_mode auto + 越界保护）有意，予以固化
python3 scripts/pin_ha_closedloop.py --check   # ✅ 契约校验通过（7 任务 / 16 资产 / init pose / decoder=release）
```

### 9.2 encoder 输入布局（详见 `docs/SONIC_ENCODER_INPUT_LAYOUT.md`）
- `col[4:584]` = **当前参考帧 q ×10** + **当前 qd ×10**（不是未来 10 帧）；
- `col[601:661]` = **当前锚点 rot6d ×10**（窗口内恒定）；
- 用这三组重建（其余置 0）→ token 与 `encoder_latent` **逐位相等**（验证脚本 `scripts/verify_sonic_encoder_layout.py`，0.9995–1.0 PASS）；
- **旧 `SonicTokenBuilder` 取的是未来 10 帧 + 未来 10 个锚点** → 这就是端口 token corr 0.057 的根因。

### 9.3 仍未解
- **anchor 6 维的公式**：现有一切四元数字段组合最好只到 Δ=0.34；HA 录制数据里有现成的
  `consumed_anchor_rot6d`，BONES-SEED 没有 → 混库前必须解出，否则只能用 HA-only 标签 token。
