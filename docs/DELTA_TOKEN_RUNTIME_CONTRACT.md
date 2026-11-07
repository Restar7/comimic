# Δtoken 运行时契约（训练 / 部署必须逐位一致的部分）

> 任何一条不满足 ⇒ 不要跑，不要"猜一个继续"。本文是**唯一**契约说明；
> 代码实现见 `diff_vla_wbc/delta_token_io.py`、`diff_vla_wbc/deploy_obs.py`，
> 测试见 `tests/test_delta_token_io.py`、`tests/test_delta_token_schema_v2.py`、
> `tests/test_deploy_obs_alignment.py`。

---

## 1. 施加方式

```
token'_t = token_t + Δ,  t = 0..T-1      # Δ 是**共享**的 64 维，广播到整段 chunk
hand14   = hand14                        # 手部 14 维原样通过，永不修改
Δ        = apply_norm_cap(Δ_raw, cap)    # 训练/部署**同一个函数**（tanh 软上限，cap=0 ⇒ 恒等）
```

- 服务端一处施加：`scripts/ha_vla_latent64_server.py`；
- 训练一处施加：`diff_vla_wbc/v3_step.py::_delta_token_delta`（`d.unsqueeze(1)` 广播）。

---

## 2. 输入 `state64`

```
state64 = [ root_rot6d(6) | dof_pos(29) | dof_vel(29) ]
```

| 项 | 规定 |
|---|---|
| 关节序（线上） | **HA_V31_29**（= `SONIC_ISAACLAB_JOINT_ORDER` = Isaac 资产 `joint_names`） |
| 关节序（训练特征内部） | **STANDARD_29**（trackerlab / MuJoCo 序），由 `joint_contract` 转换 |
| rot6d 元素顺序 | `[m00, m01, m10, m11, m20, m21]`（`R[:, :2]` **行优先**）<br>⚠️ 不是 `[m00,m10,m20,m01,m11,m21]`（列优先是**错的**） |
| rot6d 语义 | 线上是"相对回合起始朝向"（`rotlocal6`，开局 identity，见 `state_extra.rotlocal6`） |

**转换只允许走** `joint_contract`（`to_standard` / `from_standard` / `permutation`）与
`delta_token_io.delta_features_from_state64`，禁止在别处写字面量顺序。

---

## 3. 特征宽度（两种口径，由 ckpt 决定）

| `feat_dim` | 内容 | 说明 |
|---|---|---|
| **35** | `[rot6d(6) , dof_vel(29)]` | 老口径（不含 dof_pos） |
| **64** | `[rot6d(6) , dof_pos(29) , dof_vel(29)]` | 部署 `state64` 全量（推荐） |

网络输入宽度 = `64 (token) + feat_dim`。

---

## 4. checkpoint 契约（schema v2）

ckpt 里必须带：

```python
"delta_token_state" : state_dict          # 形状自解释（首层宽度 - 64 = feat_dim）
"delta_token_meta"  : {
    "schema_version": 2,
    "hidden", "depth", "init_std", "cap",
    "feat_dim", "token_dim",
    "input_features", "input_features_version",   # 人类可读 + 版本号
    "feat_joint_order",            # "standard"
    "apply", "norm_cap",           # 语义标签，必须与实现一致
    "teacher_objective", "teacher_reward", "teacher_horizon", "teacher_eta", "teacher_n",
    "grad_target_weight", "trained_steps", "git_commit",
}
```

规则：
1. **新 ckpt 必须写 v2**；缺字段 ⇒ 加载器退回 legacy 形状推断（有告警，且记录 `schema_version=1`）；
2. **meta 与权重形状矛盾 ⇒ fail-closed**（`DeltaTokenContractError`），不许猜；
3. 审计：`python3 scripts/harness02_check_delta_ckpt.py <ckpt>`，`status` 必须是 `ok`；
4. 旧 ckpt（无 `delta_token_state`）⇒ **Δ 模块整体关闭**，等价于 base（这条是有意的，不要当错误）。

---

## 5. 执行器与解码器

| 项 | 规定 |
|---|---|
| 部署执行器 | **SONIC release ONNX decoder**（`model_decoder.onnx`，`onnxruntime` CPU provider） |
| 部署 obs | **994 维** = `motion_token(64) + 930`（见 §6） |
| 训练侧 rollout 用的策略 | `trackerlab` WBC（torch，`WBC_A24000_wmvel_cmd.pt`），obs 为 **93/99 维 × 10 帧** |
| 两者关系 | 用 `scripts/probe_decoder_alignment.py` 对拍（历史 max 误差 ~1.9e-6）。**对拍不通过就不许声称"训练=部署"** |
| provider 门禁 | `HA_ACTION_SOURCE` 必须解析到 SONIC provider；否则 `scripts/_deploy_audit.sh` 直接失败，**不许静默回落**（历史事故：静默回落 SuperMimic provider，导致"没效果"被误读） |

---

## 6. 部署那 930 维（`diff_vla_wbc/deploy_obs.py`）

```
930 = [10×ang_vel(3)] + [10×(joint_pos − default_q)(29)] + [10×joint_vel(29)]
    + [10×last_action(29)] + [10×gravity(3)]

块顺序 == action_provider_sonic.py 的 concat 顺序：
    latent(64) → ang_vel_hist → joint_pos_hist → joint_vel_hist → last_action_hist → grav_dir_hist
关节序 == HA_V31
gravity == gravity_dir_from_quat_wxyz(q) == -R[...,2,:]
```

训练侧转换入口：

```python
to_deploy_obs_hist(hist_10x93, xi_dq0=xi[:, 1:30])   # → (B,930)
```

**逐位保证**：`tests/test_deploy_obs_alignment.py`（6 项，atol 1e-12，含"忘了换关节序会明显不同"的负向测试）。

---

## 7. 参考帧 / clip

| 项 | 现状 |
|---|---|
| 服务端 | **不接收** `frame`；服务端 Δ 只用 `state64` ⇒ 与参考无关 |
| provider | 持有参考时间线（`joint_pos / joint_vel / body_quat_w`，**无 root p/v**） |
| 采样侧 | 必须显式给 `--real-transitions`（起点集合）与 `SM_PACKED` / `SM_TOKEN_LIB`；不确定就记录为 UNVERIFIED，不许默认 PASS |

---

## 8. 量化（`TOKEN_QUANT`）

- 当前 `TOKEN_QUANT=none` ⇒ 不触发；
- 若将来启用 STE/hard，**必须先定义**"Δ 在量化前还是后施加"，并同步训练侧；
  否则训练/部署口径会分叉（这一条写进代码注释与测试之前，量化保持关闭）。

---

## 9. 默认策略（重要）

1. 生产默认路径必须能明确跑 **Base（Δ 关闭）**：不传 `--head-ckpt`，或 ckpt 里没有 `delta_token_state`；
2. 实验性 Δ head 只能**显式选择**才启用；
3. **未通过 Gate C（闭环）之前，任何 Δ ckpt 都不得作为默认上线 head。**

---

## 10. 已知限制（不是 bug，是当前管线的边界）

1. **root p/v 的 10 帧历史拿不到**（数据源没有）：Δ 的输入不能包含 root 位置/速度历史；
2. **参考帧历史**：只落 `frame` + `ref0`；需要历史时可回运动库按 `frame` 重取；
3. 闭环录制数据（`runs/closedloop_r3/*/dump/steps.jsonl`）的关节口径**未定死**（一步误差 0.415 rad），
   不可用于评估 WM/漂移。
