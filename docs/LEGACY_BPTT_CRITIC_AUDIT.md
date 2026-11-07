# 旧 BPTT Critic 审计（规范 §1 / §8 M0 落盘件）

> 规范：`docs/TEMPORAL_AC_BPTT_V5_MAINLINE_SPEC.md`（§1.1 / §1.2 / §1.3 / §8 M0）。
> 本文件把 §1 的表格 + **本机复跑的实测输出**落盘。凡未亲自复跑的一律标 `SPEC_CITED`。
> 执行 worktree：`comimic-wt03`（分支 `harness/03-bptt-diagnosis`，基线 `a4bd1ff`）；
> 主仓库 `comimic`（main）**只读**。

---

## 0. M0 判定（先看这个）

| 判定项 | 结论 | 状态 |
|---|---|---|
| `bptt1` 的 `critic_grad_through_state` | **`False`** ⇒ 旧 Critic 对 Actor 的贡献**恒为 0**（`V(x_H)` 是常数） | **FOUND** |
| 旧 Critic 权重 | **`critic_ckpt=None`，从未保存** | **NOT_FOUND ⇒ 不可复用** |
| `K == H == 20` ⇒ 尾部区间 | `K − H = 0` ⇒ `V(y_H)` **不可识别** | **BLOCKED(BOOTSTRAP_NOT_IDENTIFIABLE)** |
| 训练侧 `history` 的 q/qd 与 `state64` 的 dof_pos/dof_vel | **同一套关节序（HA_V31）**，只差 `−default`；逐位 `max|Δ| ≈ 1.2e-7` | **FOUND（本机复跑）** |

⇒ **M0 PASS**。写作约束：v5 的 Critic 是**首次真正接入梯度**，不是"修复旧 bug"；
禁止在文档/日志里写"复用旧 critic / 旧 v_ev 一直很好"。

---

## 1. `bptt1` 的真实开关（§1.1）

命令：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
python3 -c "import json;d=json.load(open('runs/bptt1/args.json'));print({k:d.get(k) for k in
  ['critic','critic_source','critic_grad_through_state','critic_ckpt','batch_size','horizon',
   'start_max_drift','start_sampler','vla_state_from_burnin','delta_token_fullstate',
   'delta_token_cap','window_norm']})"
python3 -c "import json;d=json.load(open('runs/bptt1/v3_config.json'));print(d)"
sed -n '1330,1348p' diff_vla_wbc/v3_step.py
```

实测（本机复跑）：

| 项 | 值 | 证据 | 判定 |
|---|---|---|---|
| `critic` | `True` | `runs/bptt1/args.json` | FOUND |
| `critic_source` | `imagined` | `args.json` + `v3_config.json` | FOUND |
| **`critic_grad_through_state`** | **`False`** | `v3_config.json`；`v3_step.py:1333` | FOUND |
| ⇒ Critic 对 Actor | **恒 0**（`else: with torch.no_grad(): vH = self.critic(rollout.states[-1].detach(), …)`） | `v3_step.py:1345-1348` | FOUND |
| `boot_one_sided` / `critic_rel_clip` / `critic_state_grad_scale` | `True` / `5.0` / `0.1` | `v3_config.json` / `args.json` | FOUND |
| Critic 输入 | `(state 71, ref 71, xi 33)`，无 future-plan / 无 history | `diff_vla_wbc/token_critic.py:24-30` | FOUND |
| **旧 Critic 权重** | `critic_ckpt=None`，从未保存 | `args.json` + `ls runs/bptt1` | **NOT_FOUND** |
| 采样 | 每步 512 新鲜窗口（2 rank×256） | `args.json`（`batch_size=256`，`NPROC=2`） | FOUND |
| 结束原因 | 计划 150 步、实际 59 步，日志尾 `SignalException: got signal 15`（SIGTERM，非崩溃） | `logs/bptt1.log` | FOUND |
| `v_ev`（train） | 首 10 步 **+0.52** → 末 10 步 **+0.90**（峰 0.9335） | `runs/bptt1/train_metrics.jsonl` 的 `critic/v_ev` | SPEC_CITED |
| WM 回报 J（H20 折扣和） | **1.51847 → 1.53543 = +1.12%** | `train_metrics.jsonl` 的 `loss/reward_raw` 首/末 10 步 | SPEC_CITED |
| `‖Δtoken‖` | 0.0754 → **0.1488**（=1.9η，无 cap；`delta_token_cap=0.0`） | `delta_token/norm` + `args.json` | FOUND（cap=0 本机复跑） |
| 起点配方 | `all_transitions.pt`、`start_max_drift=0.10`、`start_sampler=uniform`、burn-in k=10、扰动 5cm/0.1rad/0.05rad/0.3rad/s | `args.json` | FOUND |
| VLA 状态口径 | **`vla_state_from_burnin=false`**（VLA 吃录制帧的 state，不吃 burn-in 末状态） | `args.json` | FOUND |
| `delta_token_fullstate` | `true`（state64 = rotlocal6+q+qd） | `args.json` | FOUND |

**结论**：旧 Critic 的 `v_ev` 看着"很好"，但它**从未进入 Actor 梯度**（`no_grad` 分支）；
且权重不存在。v5 只能新建 Critic，且默认关闭（`K==H` 不可识别）。

---

## 2. 时间轴合同（§1.2）

| 量 | 真值 | 证据 | 判定 |
|---|---|---|---|
| `T`（一次 VLA 生成的 token 数） | **40** | `Gr00tVLM(action_horizon=40)`；`runs/bptt1/args.json` 的 `horizon=20` 是 WM 步数 | FOUND |
| `H`（可微 WM 步数） | **20** | `V2Config.horizon=20` | FOUND |
| `K`（一次 chunk 实际执行的 token 数） | **20** | 客户端 `HA_VLA_EXEC_STEPS=20`（`action_provider_supermimic.py:911`） | SPEC_CITED（client 文件不在本仓库路径内，未复跑） |
| `ℓ(k)`（WM 步 k → token 位置） | **`ℓ(k)=k`**，`k≥T` 钳到 `T−1` | 本机复跑：`diff_vla_wbc/rollout.py:366-367` `token_at(step)=tokens[:,min(step,T-1)]` | FOUND |
| 被训练的 token 位置 | **只有 0..19**（H=20） | 同上 | FOUND |
| 被修改的 token 位置（旧） | **0..39**（服务端广播 `chunk64 = chunk64 + _d[None,:]`） | `scripts/ha_vla_latent64_server.py:671` | SPEC_CITED |

⇒ **历史 bug 级问题**：`bptt1` 在 20..39 上**改了 Δ 但从未训练**。
v5 的处理：`mask_t = 1[t<K]`，`t≥K` 恒 0，部署侧同样只在 `t<K` 加
（`delta_token_io_v5.apply_delta_per_frame` 会再断言一次）。

---

## 3. 观测/历史口径（§1.3）——本机逐位复跑

### 3.1 环境事实（`python3 -c` 实测）

| 项 | 值 |
|---|---|
| `bundle.obs_width` | 93 |
| `bundle.hist` | 10 |
| `term_dims` | `(29,29,3,3,29)` |
| `len(action_scale)` | 29 |
| `default_joint_pos` | SONIC 口径（≠ `g1_default_joint_pos()` 的 mjlab `KNEES_BENT_KEYFRAME`） |

### 3.2 `history` vs `state64`（真实窗口 N=32）

脚本（与规范 §4.2 同口径，但把"未知 default"换成"从数据反解常数偏移"）：

```python
import torch
d = torch.load("runs/harness_03_bptt/big/windows.pt", map_location="cpu", weights_only=False)
h, st = d["history"].float()[:32], d["state"].float()[:32]
ht = h[:, :, :58]
off = ht[:, -1, :29] - st[:, 7:36]          # = −default（应与窗口无关）
d_est = -off.mean(0)
print((off - off.mean(0, keepdim=True)).abs().max())      # 偏移是否跨窗口恒定
print((ht[:, -1, :29] - (st[:, 7:36] - d_est)).abs().max())
print((ht[:, -1, 29:] - st[:, 42:71]).abs().max())
```

实测输出：

| 检查 | `max|Δ|` | 判定 |
|---|---|---|
| `history[:, -1, 0:29]`（q 块） vs `state[7:36] − default` | **1.19e-7** | FOUND |
| `history[:, -1, 29:58]`（qd 块） vs `state[42:71]` | **0.0** | FOUND |
| 偏移跨窗口恒定（关节序一致性） | **5.96e-8** | FOUND |
| `hist10_train` vs 部署路径 `hist10_from_hist64`（state64 的 q/qd 块） | **1.19e-7** | FOUND |

★ **这个 default 是 `g1_default_joint_pos()`（mjlab `KNEES_BENT`，前 5 维
`[-0.312, 0, 0, 0.669, -0.363]`），不是 `bundle.default_joint_pos`（SONIC/MDP 动作路径，
`[-0.2, 0, 0, 0.42, -0.23]`）。两者差 ≈0.27 rad；§4.2 的核对脚本用 `g1_default_joint_pos()`
是对的，而 §1.3 里"`default_joint_pos[:5]=[-0.2,0,0,0.42,-0.23]`"描述的是**动作路径**，
不要拿它去算 hist10 的 q 偏移（本机实测：用 SONIC default 会得到 `max|Δ|=0.27` 的假失败）。

⇒ **训练侧 obs93 的 `q/qd` 与训练侧 `state64` 的 `dof_pos/dof_vel` 是同一套关节序**，
只差一个 `−default` 常数偏移。

★ **但"同一套"是哪一套？——本机四处代码证据判定为 `standard`（MJCF/MDP），不是 HA_V31**：
① `MDP.obs` 直接算 `q − default_q`（`supermimic.py:171`），而 `bundle.default_joint_pos`
的 **knee 在 index 3**（`[-0.2,0,0,0.42,-0.23,0]`）= STANDARD 序；
② `SonicIsaacBundle.step` 注释："状态的关节序天生是 **STANDARD**（语料/VLA 观测/FK 全按它）"；
③ `SonicExecutor.obs994` 对 q/qd 做 `index_select(std_to_ha)` 才喂 SONIC decoder；
④ `delta_token_io`（v4）训练侧约定 = `standard`。
⇒ 规范 §1.3 的"两者都是 HA_V31"只对**相对**关系成立（同序、差 `−default`），
**绝对序 = STANDARD**。**部署侧 `state64` 才是 HA_V31**，所以 v5 契约必须声明
`state_joint_order=ha_v31` 并在边界换到 standard（`delta_token_io_v5.py` 已实现，
强口径对拍 `scripts/check_history_parity_v5.py` 两条部署路径均 `max|Δ|=0`）。

### 3.3 起点分位数（§3.1 / §3.4，本机复跑）

```bash
python3 -c "
import sys; sys.path.insert(0,'scripts')
import diag_bptt_learnability as d03
env = d03.load_env('cuda:0', d03.MAIN_REPO/'runs/bptt1/args.json')
w, info = d03.eligible_start_mask(env, horizon=20, drift=0.10, h_tail=0)
print(info)"
```

实测输出：

```
{'n_frames': 634016, 'n_eligible': 338402, 'n_drift_eligible': 346729,
 'horizon': 20, 'h_tail': 0, 'drift': 0.1, 'min_frame': 10,
 'uniform_over_eligible': True, 'eligible_per_clip': 338402}
```

| 量 | 值 | 说明 |
|---|---|---|
| 总帧数 | 634,016 | 375 个可用 clip |
| **漂移合格帧**（`‖pos−ref‖≤0.10` 且真采到过，含 `w==0` 前一帧回填的 bptt1 原坑） | **346,729** | **与规范 §0-A 逐位一致** |
| **再叠加 §3.4 窗口约束后**（`f≥10` 且 `f+H≤clip_end`） | **338,402** | on-the-fly `multinomial` 的**实际**采样域 |

> 说明：规范 §0-A 的 346,729 是"漂移合格帧"；§3.4 新增了
> `f≥10` 与 `f+H+H_tail≤clip_end` 两条硬约束（尾部参考帧必须真实存在），
> 因此 v5 的实际采样域是 **338,402**（−2.4%）。
> `windows_meta.json.uniform_over_eligible` 在 on-the-fly 与 `build --uniform-frames 1`
> 两种路径下都为 `true`。

### 3.4 关节序换序的适用范围（必须写进报告）

| 处 | 宽度 | 块序 | 关节序 |
|---|---|---|---|
| 训练侧 obs93（`windows.pt['history']`） | 93 | `q−default(29) \| qd(29) \| ω(3) \| g(3) \| a_prev(29)` | **HA_V31** |
| `state64`（客户端发） | 64 | `rot6d(6) \| dof_pos(29) \| dof_vel(29)` | **HA_V31** |
| 部署侧 994（喂 ONNX decoder） | 994 | `token(64) \| ω(10×3) \| (q−default)(10×29) \| qd(10×29) \| last_action(10×29) \| g(10×3)` | 见 §4.6（需会话 01 的 `probe_decoder_alignment.py` 判定） |

⇒ **只有 994 的第 3~5 块**存在 `index_select(std_to_ha)` 的换序问题；
`hist10` 与 `state64` 之间**没有**换序问题（3.2 已逐位证明）。

---

## 4. 本版新增的桥接产物（§2.5 / §12）

| 文件 | 作用 | 状态 |
|---|---|---|
| `diff_vla_wbc/delta_token_io.py` | v4 部署契约（从 `788d1ee` 取） | ✅ 已取，`tests/test_delta_token_io.py` 25 passed |
| `diff_vla_wbc/delta_token_io_v5.py` | v5 per-frame 契约（T/K/H + hist10 + fail-closed） | ✅ 新增 |
| `diff_vla_wbc/delta_token_temporal_v5.py` | per-frame Actor（1,744,832 参数） | ✅ 已有（本机复核参数/梯度/mask） |
| `diff_vla_wbc/imagined_value_v5.py` | TDλ + EMA target + Gate（默认关闭） | ✅ 新增 |
| `diff_vla_wbc/action_trust_v5.py` | 29D rad 动作 trust（手算对拍） | ✅ 新增 |
| `docs/TEMPORAL_AC_BPTT_V5_RUNTIME_CONTRACT.md` | history / schema / 回滚 | ✅ 新增 |

> **未取**（规范 §2.3）：`diff_vla_wbc/v3_step.py`、`teacher_probe.py`、
> `scripts/harness02_*`、`train_distill_offline.py`、`deploy_obs.py`。
> 验证：`git status --short | grep -E "v3_step|teacher_probe"` 必须为空。
