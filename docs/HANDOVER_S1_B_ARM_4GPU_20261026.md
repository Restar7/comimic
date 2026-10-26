# 交接：在另一台 **4 卡** 容器跑 S1 的 B 臂（warm start=10500，hold token）

> 写于 2026-10-26。本机（2 卡）在跑 A 臂；请你在 **4 卡容器**跑 B 臂，两边共用同一个工作区目录。
> 目的：判定 **"换 token 口径后，从哪个权重起步更好"** —— A=16000（已适配 HA 但被 future 口径训了 5500 iter）、
> B=10500（BONES+HA 手术件，`ckpt_010500_physint_obs93`）。其余配方**完全相同**。

---

## 0. 背景（30 秒版）

- HA 的 64 维 token 是 **hold** 语义（当前帧重复 10 次）；我们之前编的是**未来 10 帧** → corr 0.05。
- 已修好前端（`SONIC_TOKEN_WINDOW=hold` 是默认值），并重建了 HA 参考库（命令流）：`runs/isaac_motion/ha_cmd/packed.pt`。
- A 臂（本机）读数：`sim_rew 0.8399→0.8566`、`term 1.02%→0.27%`、官方 HA-699 **2/699 摔**、但 `norm_return 0.627`（偏低，怀疑是"future 惯性"或"口径固有代价"）。
- 所以需要 B 臂：**同样的 hold 配方，但从不曾被 future 口径深度适配的 10500 起步**。

---

## 1. 环境与路径（都已是共享目录）

| 项 | 路径 |
|---|---|
| 工作区 | `/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic` |
| 训练入口 | `trackerlab-supermimic/scripts/rls/supermimic/train_bones_seed.sh` |
| 混合库 | `trackerlab-supermimic/data/bones_seed/bones_seed_plus_ha_cmd_x20.pt`（BONES 79.5% + HA 命令流 ×20） |
| B 臂 warm start | `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint_obs93_spec.pt` |
| encoder ONNX | `/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx` |
| HA 评测库 | `runs/isaac_motion/ha_tokens/packed.pt`（699 clips） |
| BONES 评测库 | `trackerlab-supermimic/data/bones_seed/bones_seed_compact.pt` |

---

## 2. B 臂起跑（4 卡，1500–2000 iter）

```bash
cd /inspire/.../comimic/trackerlab-supermimic
export SONIC_ENCODER_ONNX=/inspire/.../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx

RESUME=$PWD/logs/bones_seed_token_ha_v2/ckpt_010500_physint_obs93_spec.pt \
OBS_MODE=paper_obs93 FREEZE_WM=0 NO_CRITIC=1 WM_ONLY_ITERS=0 LR=2e-5 \
MODE=train NPROC=4 ENVS_PER_RANK=8192 ITERATIONS=12600 \
MAX_EPISODE_STEPS=0 \
LIBRARY=$PWD/data/bones_seed/bones_seed_plus_ha_cmd_x20.pt \
INTERFACE=token COMPILE=1 HA_ALIGN=1 CKPT_EVERY=250 PORT_BASE=29750 \
LOG_DIR=logs/wbc_hold_B_10500 \
setsid nohup bash scripts/rls/supermimic/train_bones_seed.sh > logs/wbc_hold_B_10500.launcher.log 2>&1 &
```

**起跑后必须核对这 4 行**（在 `.launcher.log` 里）：

```
[AUDIT][spec] HA_ALIGN=1 usd=g1_29dof_rev_1_0.usd default_q[knee/ankle/elbow]=0.420/-0.230/0.870
              action_scale mean/max=0.3789/0.7500
[AUDIT][prov] obs=paper_obs93 iface=token lib=bones_seed_plus_ha_cmd_x20.pt ...
library: .../bones_seed_plus_ha_cmd_x20.pt
--freeze_wm 不出现（应为 False） / warm start: policy updates start after 10500
```

⚠️ 四个**必须**（踩过的坑）：
1. `WM_ONLY_ITERS=0` —— 默认是 500，会让策略在头 500 iter 完全不更新（`pi_loss` 恒 0）；
2. `LOG_DIR` 必须是**新目录**（runner 拒绝复用旧目录）；
3. 不要设 `SONIC_TOKEN_WINDOW`（默认就是 `hold`）；设了 `future` 就变成旧口径，白跑；
4. 不要混 `obs99` 的 ckpt（`ckpt_010500_physint.pt` 是 obs99，必须用 `_obs93_spec.pt`）。

**iter 目标**：12600（= 10500 + 2100）。若 1500 步就够看趋势，可以在 12000 停（`CKPT_EVERY=250`，随时可停：`touch` 一个 stop file 或 kill 进程组）。
参考耗时：4 卡 × 8192 ≈ **1.0 it/s** ⇒ 1500 步 ≈ **25 min**，2100 步 ≈ **35 min**。

---

## 3. 门禁（跑完 B 臂后，每格都要，且与 A 臂**同协议同仪器**）

```bash
PY=../third_party/trackerlab_env/bin/python
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y

# ① HA-699（部署域）—— A 臂：n_failed 2/699，norm_return 0.6272
$PY scripts/rls/supermimic/evaluate.py --run_dir logs/wbc_hold_B_10500 \
  --ckpt logs/wbc_hold_B_10500/ckpt_012<N>.pt \
  --motion_packed ../runs/isaac_motion/ha_tokens/packed.pt --device cuda:0 --kit_args=--allow-root

# ② BONES-500（防遗忘；基线 v1 = 0.8836 / 6-500）
$PY scripts/rls/supermimic/evaluate.py --run_dir logs/wbc_hold_B_10500 \
  --ckpt logs/wbc_hold_B_10500/ckpt_012<N>.pt \
  --motion_packed data/bones_seed/bones_seed_compact.pt --clip_limit 500 --device cuda:1 --kit_args=--allow-root

# ③ WM 速度（label 口径！）—— A 基线：v1 0.737/0.947；A 臂未标定前 0.15/2.8
$PY ../scripts/eval_wm_velocity.py --ckpt trackerlab-supermimic/logs/wbc_hold_B_10500/ckpt_012<N>.pt --tokens label
```

⚠️ `evaluate.py` 的输出名只按 ckpt 名生成（`<ckpt>_eval.json`），**①和②会互相覆盖** → 每格跑完立刻
`cp <ckpt>_eval.json ../runs/S1_gates/B_HA699_eval.json` / `B_BONES500_eval.json`。

---

## 4. 回报模板（贴这一张就够）

```
[B] warm start=10500  iter=12xxx  卡数=4x8192
    run 内: sim_rew=0.xxx  term=0.xx%  ep_len=xxx  wm_loss=0.xxx
    HA-699 : n_failed=?/699  norm_return=0.xxx  (A: 2/699, 0.6272)
    BONES-500: n_failed=?/500 norm_return=0.xxx (基线 0.8836/6-500)
    WM: qd_corr=0.xxx ratio=0.xxx
    起跑核对: spec/audit 行是否 PASS=？
```

---

## 5. 判读与去向

| B vs A（HA-699 `norm_return`） | 结论 |
|---|---|
| B 明显更高（≥ +0.03） | 起点选错：后续主线的 warm start 改用 10500 系，A 臂作废 |
| B 与 A 相当（±0.02） | 起点不重要 → 用 A（已适配 HA 的权重），加训推进即可 |
| B 明显更低 | 说明"HA 适配"必须保留 → 继续 A，并加训到平台 |

**不要**把 B 的结果和端口空间（`token_space=port`）的任何数字混表；B 属于 `token_space=label, window=hold`。

---

## 6. 与本机 A 臂的进度同步（供你判断要不要等）

- A 臂：`logs/wbc_hold_x20`（18500 完成）→ `logs/wbc_hold_x20b`（→20000，进行中）；
- 已出：HA-699 `2/699, norm_return 0.6272`；
- 队列中：BONES-500(18500) + **hold 基线**（`ckpt_016000_spec` 在 hold 下不训练直接评测 HA-699）——
  基线数是判断"0.627 是口径代价还是训练不足"的关键，出来会同步给你。
