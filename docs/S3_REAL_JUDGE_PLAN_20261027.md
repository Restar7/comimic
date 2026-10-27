# S3 真物理裁判改造 —— 完整修改意见（2026-10-27）

> 本文取代 `docs/S3_REAL_JUDGE_PLAN_20261027.md`（那份是初稿），本文是**可执行的完整版**：
> 根源问题 → 能/不能解决什么 → 目标架构 → 文件级改动 → 执行顺序 → 判据 → 风险 → 事实附录。

---

## 0. 一句话结论

**根源问题不是"梯度反传不work"，而是"我们用一个学出来的模型当裁判，去判一个它分辨不出的差异"。**
本改造把**判定权**从模型换到真物理（Isaac），并用真物理采到的 **on-policy 转移**把模型裁判校准到它该有的精度；
但它**不改变**端到端反传的路径，也**不保证**反传一定学到东西——那是改造后要用真裁判去回答的问题。

---

## 1. 根源问题（全部用今天的实测数字定义）

### 1.1 已经被确证的六个事实

| # | 事实 | 数字 |
|---|---|---|
| F1 | **H=5 可微视界**：不 hack，但收益不可分辨 | `ret_vla` 比 GT **+2.0%**；`tok_mse` ×1.43 |
| F2 | **H=20 可微视界**：hack 回来，且约束越松涨越多 | `ret_vla` **+12.7%（e60）/ +24.0%（e100）**；`tok_mse` ×1.4 / **×2.0**；`surv` 0.9999→0.96 |
| F3 | **手部 BC 污染 token**：λ_r=0 的对照组也漂 | λ_r=0 臂 `tok_mse` ×1.39（它没有任何奖励梯度）|
| F4 | **in-loop WM 刷新有害**（同一仪器前后对比）| `qd_corr` 0.752 → **0.589**，`qd_ratio` 1.115 → 1.700 |
| F5 | **executor 选择正确**（部署口径仪器，699 clips）| A@20000 `norm_return` **0.8673 / 0 摔**；A@18500 0.8604；A@16120 0.8203 |
| F6 | **"A 系越训越差"是旧仪器测反了** | 旧仪器(ha_tokens) A@16120 0.7094 > A@20000 0.6178；真口径(ha_cmd) 正好相反 |

### 1.2 根因

> 我们要优化的差异（VLA vs GT 在想象回报上的差）在 H=5 时只有 **0.6–2%**，
> 而**世界模型自己的误差与被追踪信号的量级之比**在同一数量级甚至更大。
> 于是：**视界短 → 裁判分辨率不够，看不到信号；视界长 → 裁判误差累积，优化器把它放大成"分数"。**
> 这不是调参能解决的，是"模型当裁判"的内生矛盾。

---

## 2. 这个改造能解决什么、不能解决什么

| 类别 | 内容 |
|---|---|
| ✅ **能解决** | ①**判定权**：三臂（GT/ORACLE/VLA）在真物理里比，答案不再依赖模型；②**WM 的域偏差**：用我们控制器自己的 on-policy 真转移重标定（现在 0.752，目标 ≥0.85，参考实现 0.90）；③**奖励的可靠性**：真奖励来自 env 自己的 reward（与 SuperMimic 的 `r_{t+1}` 同源）|
| ❌ **不能解决** | ①**梯度路径的模型误差**：A 方案仍要穿过 WM，若真裁判判"无收益"，则必须换 best-of-K 蒸馏或真仿真 in-the-loop RL；②**batch 规模**（我们 4 vs 他们 4096）——这是 1.62B action head 的显存硬约束；③**渲染**（在线环前置条件，见 §4.G）|

**所以准确的期待**：改造后我们**第一次能知道答案是对是错**；至于答案是不是我们想要的，得等结果。

---

## 3. 目标架构：双轨（在线环 / 回放环）

两条轨道**共用**同一套数据格式、记录字段、判据。

### 3.1 在线环（需要渲染器；与 SuperMimic 同构）

```
loop 每轮：
  (1) VLA 实时推理（batch = envs）→ token chunk（每 20 步一次）
  (2) 环境执行 20 步（真物理），写 (s,a,s',r,frame) 进 replay buffer
  (3) 每 N 轮：在 replay buffer 上 fit WM（真转移）
  (4) 每 N 轮：A 方案策略更新（冻结 WM 内 H=20 pathwise + γ^H·V(s_H)）
```

规模换算（从 SuperMimic 的 `num_envs=4096, UTD=0.4` 缩到我们的 128 envs）：

| 量 | SuperMimic | 我们（128 envs）|
|---|---|---|
| 等价 UTD | 0.4 次更新 / 迭代（4096 env steps）| **0.0125 次更新/env step → 每 80 env step 一次** |
| 每采集轮 | — | 1 次 VLA 推理 + 20×128 = **2560 条转移** |
| 更新频率 | — | **每 4 轮做 1 次策略更新** |
| 吞吐 | ~273k 转移/秒 | **~430 转移/秒** |
| 1 小时能跑 | — | **~100 万转移 + ~100 次策略更新** |

### 3.2 回放环（**不需要渲染器**；今天就能出真物理答案）

```
每 episode：(clip, arm, seed) → 从 runs/arm_tokens/{clip}/{arm}.npy 回放 token
            → 冻结 WBC 在 Isaac 里执行 300 步 → 记 success/fall/reward/跟踪误差
```
* **三臂唯一差异 = token**（手部、起点帧、随机种子全同）；
* 同时把 `(s,a,s',r)` 落盘 → 给 §4.F 用。

---

## 4. 文件级修改清单

### A. token 流生成（离线，无相机）

**新增** `scripts/dump_arm_tokens.py`

```python
--clip HOI_double_desk__tw__ep000000 --out runs/arm_tokens
# 产出 runs/arm_tokens/{clip}/{gt,oracle,vla}.npy  (T,64) float32 + {arm}.json
```
| 臂 | 来源 | 备注 |
|---|---|---|
| gt | `data/npz/ha_tokens/{clip}.npz` 的 `g1_motion_token_64` | 与训练标签同源（S0.4）|
| oracle | `runs/oracle_tokens/{clip}.npy` | **已存在** |
| vla | 离线跑 VLA（`--vla-ckpt` base + `--head-ckpt vla_final.pt`），观测 = 录制 mp4 帧 + 46 维状态 | 纯推理，不占 Isaac |
| hands（三臂共用）| npz 的 `g1_hand_joints_14` | 否则比的是手不是 token |

`.json` 元数据：`source_ckpt`、`md5`、`T`、`token_mean/std`、`n_chunks`、生成命令。

### B. provider 增加 `npy` token 源（回放环用）

**改** `third_party/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_supermimic.py`

| 改动 | 细节 |
|---|---|
| 新增 CLI | `--sm_token_npy PATH`、`--sm_hands_npy PATH` |
| `_make_token_source` | 新增 `mode == "npy"`，返回 `("npy", (tokens_T64, hands_T14))` |
| `_next_latent` | npy 模式：chunk 空时从 `self._np_cursor` 切 40 行填充，光标按 `HA_VLA_EXEC_STEPS`（默认 20）推进 |
| 手部 | npy 模式从 `_sm_hands_npy` 同一光标取 14 维，**不读 front_camera** |
| 关键约束 | **npy 模式禁止调用 `_front_rgb()`**（渲染器不可用）|

**留档**：`patches/sm_token_npy.md`（`third_party/` 被 gitignore，今天已因 GPFS 丢过一次改动）。

### C. provider 增加 `inline` VLA 源（在线环用）

| 改动 | 细节 |
|---|---|
| 新增 `mode == "inline"` | 同进程加载 `Gr00tVLM` + `ActionCodec` + `TaskObservationBuilder`，直接产 (64,) token |
| 依赖 | `PYTHONPATH` 加 `Isaac-GR00T`/`gr00t_deps`（与训练一致）|
| 前置 | **需要 ego-view 图像** ⇒ 必须先修渲染（§4.G）|
| 显存 | VLA 2B fp32 ≈ 8 GB + Isaac 场景 ≈ 6 GB，单卡 48 GB 足够 |

### D. runner：常驻进程跑多 episode

**新增** `scripts/isaac_replay_run.py` + `scripts/isaac_3arm.sh`

| 项 | 约定 |
|---|---|
| 目标 | **Isaac 只启动一次**（旧 harness 每 episode 重启，62–106 s × 18 次 ≈ 半小时纯等待）|
| 入参 | `--clips c1 c2 --arms gt oracle vla --seeds 0 1 2 --max-steps 300 --out runs/isaac_3arm` |
| 每 episode | 重置场景 → 起始帧由 seed 决定（同一 clip 三臂同帧）→ 回放 token → 跑 max_steps |
| 早停 | 终止项触发即结束，记录 `episode_steps` |
| 输出 | 每 episode 一个 JSON + 一份 `SUMMARY.md`（按 clip × arm 聚合，逐 seed 明细）|
| 前置校验 | `python3 scripts/pin_ha_closedloop.py --check` PASS |

### E. 每 episode 记录什么（真物理口径）

| 字段 | 来源 | 用途 |
|---|---|---|
| `success` / `failure_reason` | HA 的 `_build_result_payload` | **主判据** |
| `fall` / `terminated` | env 终止项 | **主判据**（历史失败主因）|
| **`reward_sum` / `norm_ret`** | **env 自己的奖励函数** | **主判据**（= SuperMimic 的 `r_{t+1}` 同源）|
| `joint_rmse`、`root_pos_err`、`tilt_deg` | 逐帧对参考 | 诊断"跟得准"还是"摔倒" |
| `episode_steps` | — | 是否提前终止 |
| `(s71, a29, s71', r)` | 每步读 robot data | **给 §4.F** |

**状态布局必须与 `scripts/build_real_transitions.py` 逐位一致**：
`[pelvis_pos(3), quat_wxyz(4), joint_pos(29), root_lin_vel_w(3), root_ang_vel_w(3), joint_vel(29)]`（共 71）。

### F. WM 用 on-policy 数据重标定

**新增** `scripts/calibrate_wm_onpolicy.py`（复用 `wm_refresh.py` + `refresh_wm_ckpt.py` 的结构）

| 项 | 约定 |
|---|---|
| 输入 | `runs/wm_real/onpolicy_*.pt`（§4.E 产出）|
| 输出 | `runs/wm_real/C6h_onpolicy.pt` |
| 协议 | lr 1e-4、KL anchor 1e-3、**600–800 步**（对齐 C6 的标定协议）|
| 验收 | `eval_wm_velocity.py --ckpt C6h_onpolicy.pt --packed ha_cmd --tokens label --samples 32 --batch 32` → **`qd_corr ≥ 0.85`、`qd_ratio ∈ [0.9,1.15]`** |
| 反面案例 | 用 ha_transitions（off-policy）刷新 ⇒ 0.752→0.589（F4）。**必须用 on-policy 数据** |

### G. 修渲染（在线环的前置）

| 步 | 动作 |
|---|---|
| G1 | `apt-get install -y xvfb`；`Xvfb :99 -screen 0 1920x1080x24 &`，`export DISPLAY=:99` |
| G2 | 确认 NVIDIA Vulkan ICD：`ls /etc/vulkan/icd.d/nvidia_icd.json`（已存在，**不要再往 `/usr/share/vulkan/icd.d/` 加第二份**，会报 multiple ICD）|
| G3 | `$ISAAC_VENV/bin/python <cam_test>.py --headless --enable_cameras`，判据：能拿到 `(64,64,3)` RGB 且非全零 |
| G4 | 失败则退：**离线验证图像通道重要性**（同帧 真实 ego-view vs 全零图像，比 token 差异）；若差异小，在线环可喂零图像，不需要渲染 |

**启动须知（已实测）**：
* 必须 `unset CUDA_VISIBLE_DEVICES`（carb 不认，会判 "CUDA in bad state"）；
* 必须走 IsaacLab `AppLauncher`（裸 `SimulationApp` 会卡在 `wait_for_viewport`）；
* `isaaclab.sh` 在 `TERM=dumb` 下 `tabs` 失败 ⇒ 直接用 `$ISAAC_VENV/bin/python`。

---

## 5. 执行顺序与工时

| 步 | 内容 | 预计 |
|---|---|---|
| 0 | 停 v8（F1/F2 结论已足够），腾显存 | 2 min |
| 1 | **§4.B** provider npy 模式 + `patches/` 留档 | 30 min |
| 2 | **§4.A** token 流生成（三臂）| 30 min |
| 3 | **§4.D** runner + JSON + SUMMARY | 45 min |
| 4 | **smoke**：1 clip × 1 seed × GT × 200 步（验状态布局与终止项）| 15 min |
| 5 | **全量回放三臂**：2 clip × 3 seed × 3 arm | 40–60 min |
| 6 | **§4.F** on-policy 转移 → WM 重标定 → 复测 | 30 min |
| 7 | **§4.G** 修渲染（与 5/6 并行）| 30 min |
| 8 | 在线环（若 G 成功）：128 envs × 100 轮 + 25 次更新 | 20 min |
| 9 | 结果写文档 | 20 min |

**总计约 3.5 小时，GPU 时间 ~1.5 小时。**

---

## 6. 判据（先写死，避免事后解释）

### 6.1 回放三臂（主判据）

| 结果 | 结论 | 下一步 |
|---|---|---|
| **ORACLE > GT**（真奖励或存活率，3 seed 配对）| WM 的**排序有真信号** | 继续在 WM 里优化（A 方案），H 用 20 |
| **VLA > GT** | 联合微调**真的产生了改进** | 扩大规模 + 进长跑 |
| 三臂打平（差异 < 1σ）| WM 排序无信号 | **停止"用 WM 当环境"的路线**，改 best-of-K 蒸馏或真仿真 RL |
| VLA < GT | 联合微调有害 | 回退 base ckpt，查 token 空间/口径 |

统计口径：每臂每 clip **≥3 seeds**，逐 seed 配对；报 `mean ± std` 与逐 seed 明细，**不报单点聚合**。

### 6.2 WM 标定（辅助判据）

`qd_corr ≥ 0.85` 且 `qd_ratio ∈ [0.9, 1.15]`（仪器：`eval_wm_velocity`，ha_cmd，label，`samples 32 × batch 32`）。

### 6.3 在线环（如启用）

每 20 轮检查：`tok_mse` 是否 ≤1.5×、`act/frac_out` 0.1–0.5、真奖励是否单调上升；
任何一项破了就停，记录当时的 `tok_mse`——这本身就是"有效域边界"的实测值。

---

## 7. 风险与回退

| 风险 | 现状 | 缓解 / 回退 |
|---|---|---|
| 渲染修不好 | `Graphics plugins not available`、Vulkan 缺失 | **回放环不依赖渲染**，主判据照跑；在线环延后 |
| G1 场景步进异常（PhysX `no suitable CUDA GPU`）| 物理能跑（15,281 转移/秒，256 物体），但 G1 场景未验 | smoke（步 4）先验；异常则查 PhysX device 设置 |
| 三臂打平（最可能的结果）| — | 这不是失败，是**结论**：A 方案在 WM 里不可判；转 §6.1 第三行 |
| `third_party` 改动丢失 | 今天已丢过一次 | 补丁落 `patches/*.md` + commit |
| 共享 GPFS 上其他 agent 在跑 | S1/S2 同工作区 | 只改自己的文件；不 `git checkout/clean`；按 PID 杀进程 |

---

## 8. 附录：已确认的事实（避免再次混淆）

### 8.1 执行器 / WM 的最终选择与证据

| 项 | 值 | 证据 |
|---|---|---|
| WBC | `runs/S1_handover/WBC_A20000_wmvel_cmd.pt`（C6h，`pi_md5 993c8f7a`，iter 20000，obs93，token_window **hold**）| `runs/S1_ctrl/A_20000_cmd_eval.json`：`norm_return 0.8673`、`n_failed 0` |
| WM | 同上文件里的 `wm`（`wm_md5 bd045ffb`，在 **ha_cmd** 域标定）| `eval_wm_velocity`：`qd_corr 0.752`、`qd_ratio 1.115`、`pos_err 1.5 mm`、`rot_err 0.687°` |
| WM 保鲜 | **关**（`wm_refresh_every=0`）| 用 ha_transitions 刷新 ⇒ 0.752→0.589（F4）|
| 速度奖励 | **开**（`reward_vel_scale=1.0`）| 原始 WM 的 qd_corr 是 **−0.042**，C6h 是 0.752 ⇒ 速度通道可用 |
| critic | **真数据 return-to-go**（`critic_source=real`，按 clip 去时间趋势）| `v_ev 0.79`（旧想象 critic 是 **−2.0**）|
| 手部监督 | **关**（`lambda_hand=0`）| 手不走 WBC；开着会污染 token（F3）|

### 8.2 各仪器口径（不可混用）

| 仪器 | 测什么 | 典型值 |
|---|---|---|
| `eval_wm_velocity.py`（ha_cmd + label + policy rollout）| WM 速度通道 | C6h **0.752**；C6 在 ha_tokens 上标定到 **0.906** |
| `eval_wm_real`（ha_transitions 随机批）| WM 在 off-policy 数据上的一步误差 | 0.545 → 刷新后 0.743（**与上一行不可比**）|
| `runs/S1_ctrl/*_cmd_eval.json`（699 clips 部署口径）| WBC 策略质量 | A@20000 **0.8673** / 0 摔 |
| `S1_ctrl/haTOK_*`（旧 ha_tokens 口径）| 已废弃 | A@20000 0.6178（**方向与真口径相反**）|

### 8.3 与 SuperMimic 的对应关系（回答"我们是不是和它一致"）

| SuperMimic | 我们 | 现状 |
|---|---|---|
| `collect`（4096 envs，当前策略，真奖励）| 回放环 / 在线环 | **本次改造要补的就是这一环** |
| `fit WM`（每 UTD 在真转移上）| on-policy 重标定（§4.F）| 同上 |
| `improve policy`（冻结 WM，H=32，`J=Σγ^t c_t r + γ^H c_H V_H`）| **同形**（H=20，`loss_r = -(ret + γ^H V)`）| 已一致 |
| 无 PPO（`policy_trainer.py` 里没有 ratio/log_prob）| 无 PPO，SHAC 式 pathwise | 已一致 |
| batch 4096 | batch 4 + accum 8 | **硬约束，不可对齐** |
