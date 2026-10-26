# 交接：联合微调用哪个 WBC（S1 收尾，2026-10-26，**已按口径对齐修正**）

> 给做**联合微调**（VLA ↔ 冻结 WBC/世界模型，`scripts/train_joint.py`）的同学直接用。
> 所有数字都是 2026-10-26 在本机 4×4090 上、同一仪器跑出来的；原始 json 在
> `runs/S1_gates/`、`runs/S1_ctrl/`；权重已拷到 `runs/S1_handover/`（含 `MD5SUMS.txt`）。

---

## 0. 先把口径对齐（这决定了后面所有结论）

**联合微调吃的参考库 = 旧表 `ha_tokens`（mocap），不是新表 `ha_cmd`。** 代码依据：

```python
# diff_vla_wbc/isaac_joint.py
DEFAULT_MOTION_SET = "ha_tokens"
out_dir = DEFAULT_PACKED_ROOT / DEFAULT_MOTION_SET   # → runs/isaac_motion/ha_tokens/packed.pt

# scripts/train_joint.py
HA_MOTION_DIR = ROOT / "data/npz/ha_tokens"          # 699 npz；mixed / ft318 硬性拒绝
load_isaac_bundle(..., npz=files, packed_motion=None)  # → ensure_packed_motion() 命中上面那个缓存
```

`runs/isaac_motion/ha_tokens/meta.json` 的 `npz_dirs` 就是 `data/npz/ha_tokens`，
缓存条件 `num_clips == 699` 完全命中 → **联合微调不会重建库，直接用这一份**。

| 表 | 文件 | 是什么 | 用在哪儿 |
|---|---|---|---|
| **旧表** | `runs/isaac_motion/ha_tokens/packed.pt` | mocap 重定向参考（699 clips，源 `data/npz/ha_tokens`） | **联合微调的训练/评测口径** + 文档 §3 旧门禁 |
| **新表** | `runs/isaac_motion/ha_cmd/packed.pt` | HA 命令流参考（`data/ha_raw encoder_motion_joint_*`） | 部署口径（机器人真正下发的命令流） |

两库 root 轨迹完全一致，但关节差 mean 0.29–0.34 rad、qd 差 0.60–0.74 rad/s（max 35）→
**同一个 ckpt 在两表上能差 0.33 NR**，混表会直接得出相反结论。

---

## 1. 结论（一句话）

**联合微调用 `runs/S1_handover/WBC_A20000_obs93.pt`（A 臂 it=20000），并且把跟踪参考换成命令流
`--motion-packed runs/isaac_motion/ha_cmd/packed.pt`。**

理由（三处口径对齐）：VLA 的 token 标签 = `encoder_latent`（命令流编的，与 npz 标签逐位相等，
见 `diff_vla_wbc/data.py:274`）→ WBC 的 S1 训练库也是命令流 → 真机部署喂的也是命令流。
把参考也换成命令流后，条件与参考同口径，这时对应的仪器就是**新表**，而 A@20000 是新表最优
（0 摔 / 0.8673 / joint RMSE 0.084 rad / BONES 0.8799，已到平台）。

**只有在你们坚持沿用 `data/npz/ha_tokens` 的执行轨迹参考（不改参考）时，才退而用
`WBC_A16120_obs93.pt`** —— 它在那条口径（旧表）上 0 摔 / 0.7094 / joint RMSE 0.155，但 token 侧仍错位。

**不要用 `WBC_B14800_obs93.pt`**：旧表 70/699 摔、joint RMSE 0.295 rad（16.9°），
而旧表正是联合微调的内环口径。

---

## 2. 三个候选的完整指标

| ckpt | **旧表（联合微调口径）** NR / 摔 | 旧表 joint RMSE | 新表（部署口径） NR / 摔 | 新表 joint RMSE | BONES-500 | WM qd_corr |
|---|---|---|---|---|---|---|
| A@16120（不换参考时用） | **0.7094 / 0** | **0.155 rad (8.9°)** | 0.8203 / 1 | 0.117 rad | 10/500, 0.8628 | **0.142** |
| **A@20000 ← 联合微调首选** | 0.6178 / 1 | 0.239 rad (13.7°) | **0.8673 / 0** | **0.084 rad (4.8°)** | 9/500, **0.8799** | 0.111 |
| B@14800 ← 不推荐 | 0.5341 / **70** | 0.295 rad (16.9°) | 0.8675 / 0 | 0.084 rad | 10/500, 0.8809 | 0.090 |

其他可用参照：

- 10500 起点（B 臂 warm start）：旧表 140/699 / 0.5156；新表 0/699 / 0.8166；BONES 6/500 / 0.8836
- 旧记录 WM `qd_corr` 曾报 **0.737** —— **不可复现**（见 §5.2），当前所有 ckpt 都只有 0.09–0.14

---

## 3. RMSE 怎么读（别被"1 m+"吓到）

`evaluate.py` 在 `per_motion[*].tracking` 里给三个量：

```
root_position_rmse_m     1.1 ~ 3.3 m    ← 绝对世界坐标误差，别拿它判好坏
root_velocity_rmse_m_s   0.27 ~ 0.64 m/s
joint_position_rmse_rad  0.084 ~ 0.295 rad   ← 最有判别力，用这个
```

**为什么 root 会到 1 m 以上**：奖励/终止都是**把参考重锚到机器人 xy+yaw** 之后算的
（`supermimic.py`: `dR = yaw_mat(R @ rR^T)`、`anchor = cat(p[:2], rp[2:3])`），
**绝对世界路径本来就不受约束**；参考库每 clip 起始点离原点平均 3.5 m、clip 内还要走 2 m。
所以"整体漂移 1 m"正常，跨表也不可比（同一策略旧表 3.0 m vs 新表 0.40 m，是行为差不是模型坏）。

---

## 4. 平台期（三点几何拟合，判断"还要不要加训"）

| 口径 | 曲线 | 渐近 | 现状 | 结论 |
|---|---|---|---|---|
| 新表 A 系 | 0.8203(16120) → 0.8604(18500) → 0.8673(20000) | 0.870 | 0.8673 | 已到平台（差 0.003） |
| 新表 B 系 | 0.8166(10500) → 0.8636(12600) → 0.8675(14800) | 0.868 | 0.8675 | 已到平台（差 0.0003） |
| 旧表 A 系 | 0.7094 → 0.6272 → 0.6178 | 0.616 | 0.6178 | 已封顶（在缓慢退） |
| **旧表 B 系** | 0.5156 → 0.5269 → 0.5341 | **0.545** | 0.5341 | **还差 0.011，约 2–3 轮 ×2200 iter** |

**这条很重要**：在联合微调口径（旧表）上，**A 系已经封顶（0.616，且这是 S1 训练把它从 0.709 拉下来的）**，
真正还有空间的是 B（渐近 0.545）。但 B 的 70 摔/内环精度差是结构性的，靠加训到 0.545 也追不上 A@16120 的 0.7094。

---

## 5. 已知坑（都会咬人）

1. **`sit_sofa` 一族是判据问题**：`HSI_sit_sofa` 参考根高落差均值 **0.277 m** / 最大 0.351 m，而终止阈值
   `term_height=0.25`（判 `|ref_z − robot_z|`）。所以这一族"不真坐下去必摔"，其余 6 族落差 ≤0.235 m 能糊过去。
   报指标时建议**同时给"含 sit"和"剔 sit"两版**（B@14800：70/699 → 剔 sit 10/599；A@16120 两版都是 0 摔）。
2. **WM 速度 0.737 不可复现**：`--tokens port` 走 `data/port_tokens/ha_tokens`，**该目录是空的**（699 个 clip 全找不到），
   所以旧记录那个 0.737 无法复现；当前 label 口径全在 0.09–0.14。这是独立待查项（`scripts/build_port_tokens.py` 可补）。
3. **obs99 / obs93 不能混**：`ckpt_010500_physint.pt` 是 obs99；本文三个候选都是**原生 obs93**（`obs93_converted_from` 为空），可直接用。
4. **`evaluate.py` 输出名只按 ckpt 名生成** → 同一 ckpt 跑两套库会互相覆盖。每跑一格立刻改名
   （`runs/S1_gates/v2hold_HA699_eval.json` 就是这么丢的）。
5. **训练/评测口径已置换**：S1 的训练库 HA 片段是 `ha_cmd`（命令流），旧门禁还指 `ha_tokens`（执行轨迹）→
   所以"训练一直在把旧表分数往下拉"。这不是模型退化，是两张表在量不同的东西。
6. **⚠️ 联合微调当前数据存在一处口径错位（实测，建议尽快修）**：
   `data/npz/ha_tokens/*.npz` 里同时装了两个不同口径的东西 ——
   - `joint_pos` = **机器人实际执行轨迹**（`robot_qpos_before_decimation`，实测 `mean|Δ| = 0.00000`，逐位相等）
   - `g1_motion_token_64` = **命令流编出来的 token**（实测与 `data/ha_raw` 的 `encoder_latent` **逐位相等**）
   两者差 `mean|Δ| = 0.109 rad`（单 clip）/ 0.29–0.34 rad（全库），最佳 lag **+8 帧 = 0.16 s**。
   也就是说：**VLA 输出的 token 说的是"我命令 X"，而 WBC 被要求跟踪的是"机器人实际做出来的 X′"**。
   索引对齐已验：两库 `starts`/`lengths` 完全相同、root `p`+`quat` 在 fp16 精度内一致
   （max|Δ|=4.9e-4，超过 1e-3 的帧 0/634016）→ clip 是 1:1 对应的，换 `--motion-packed` 不会错位。
   修法二选一：
   (a) 参考库换成命令流口径 —— `scripts/build_ha_command_lib.py --out runs/isaac_motion/ha_cmd`（已存在），
       联合微调时传 `--motion-packed runs/isaac_motion/ha_cmd/packed.pt`（注意这会让内环指标从"旧表"变"新表"）；
   (b) 保持执行轨迹参考，但 token 也用执行轨迹重编（不一致性更大，不推荐）。

---

## 6. 怎么接进联合微调（照抄即可）

### 6.1 环境变量（缺一不可）

```bash
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd
export SONIC_ENCODER_ONNX=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx
# 不要设 SONIC_TOKEN_WINDOW：默认 hold 才是当前口径
```

### 6.2 必须改的两处 guard（否则脚本直接 exit 1）

`scripts/launch_joint_ha.sh`：

1. `SM_CKPT` 默认值目前是 `.../bones_seed_token_ha_v2/ckpt_010500_physint.pt`（**obs99 老件**）
   → 改成 `runs/S1_handover/WBC_A20000_obs93.pt`。
2. 白名单只放行 `*bones_seed_token_ha_v2*`：
   ```bash
   case "$SM_CKPT" in *bones_seed_token_ha_v2*) ;; *) echo "[fail] ..."; exit 1 ;; esac
   ```
   → 显式允许 `runs/S1_handover/WBC_*.pt`。这个 guard 的本意是"别混非 HA 血统的 WBC"，不是"只准用 10500"。

`scripts/train_joint.py` 里 `ISAAC_SM_CKPT` 默认也指向 10500 → 用 `--supermimic-ckpt` 覆盖。

### 6.3 起跑（关键：多传一个 `--motion-packed`，把参考换成命令流）

`scripts/launch_joint_ha.sh` 里 `train_joint.py` 的调用行加一行：

```bash
    --motion-packed runs/isaac_motion/ha_cmd/packed.pt \
```

然后：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
SM_CKPT=$PWD/runs/S1_handover/WBC_A20000_obs93.pt \
STEPS=4000 RUN=joint_ha_s1_wbcA20000_haCMD \
bash scripts/launch_joint_ha.sh
```

若不改参考（沿用执行轨迹），则 `SM_CKPT` 用 `WBC_A16120_obs93.pt`、不要传 `--motion-packed`。

### 6.4 起跑后 3 行必须核对

```
[bundle] Isaac/trackerlab ckpt runs/S1_handover/WBC_A20000_obs93.pt -> full Isaac stack
         (... packed motion .../isaac_motion/ha_cmd/packed.pt)        ← 换了参考就该是 ha_cmd
[bundle] world-model structure = explicit_dt0.02 ...
[info] ... token_dim=64 / [AUDIT][spec] ... 对齐 PASS
```

三个候选我都实测加载过：`pi`+`wm` 齐全、原生 obs93、`explicit_dt0.02`、hist=10 / goal_frames=10 / token 接口。

---

## 7. 建议的下一步

1. **先用 500 步 A/B 裁决"换不换参考"（约 30 min）**：
   `A20000 + ha_cmd 参考` vs `A16120 + npz 执行轨迹参考`，其余参数/seed 全同，比较末段
   `imag_return_norm_vla`、`token_mse_decoded`、`hand_mse_decoded`，再用
   `scripts/eval_joint_token_mujoco.py --supermimic-ckpt <ckpt>` 比 `joint_rmse`/`root_rmse`。
   预期：A@16120 在内环指标上更好（旧表 0 摔 / 0.155 rad），A@20000 在部署指标上更好。
2. 定案后跑满 4000 步，与现存基线 `joint_ha_a1`（用 10500 WBC）同 seed 对比。
3. **WM 速度专项**：补 `data/port_tokens/ha_tokens`（`scripts/build_port_tokens.py`），
   才能判断 `qd_corr 0.09~0.14` 是口径问题还是 WM 退化 —— 联合微调的 `imag_*` 全依赖它。
4. **若以后想让旧表分数也上去**：只能动数据（把 mocap 口径的 HA 参考混进训练库 / HA 家族加权），
   靠加 iter 不行（旧表上 A 系已封顶 0.616，且训练本身在把它往下拉）。

---

## 8. 文件清单

```
runs/S1_handover/WBC_A20000_obs93.pt   it=20000  55 MB  ← 联合微调首选（新表/命令流口径最优）
runs/S1_handover/WBC_A16120_obs93.pt   it=16120 126 MB  ← 仅当不换参考（沿用执行轨迹）时用
runs/S1_handover/WBC_B14800_obs93.pt   it=14800  55 MB  ← 不推荐（旧表 70 摔）
runs/S1_handover/MD5SUMS.txt
runs/S1_ctrl/*_eval.json     6 ckpt × 2 套参考库 的诊断矩阵（口径证据）
runs/S1_gates/B*_eval.json   B 臂正式门禁（旧表 HA-699 / BONES-500）
docs/S1_B_ARM_4GPU_RESULT_20261026.md   B 臂全过程 + 归因
```
