# S3 交接 Runbook（2026-10-28）

> 目的：把"能不能继续、什么时候停、看什么数"写死，让后面**不需要人判断**也能跑。
> 配套：**`docs/S3_E2E_PLAN_V4_20261028.md`（唯一端到端方案，未解决问题清单 O1–O14）**、
> `docs/TOKEN_SPACE_ANCHOR_LOCKDOWN_20261028.md`（口径锁定）。
> **规则：每个门只看它写死的指标，不看训练日志里的"想象回报涨了"。**

---

## 0. 环境门（先过这个）

Isaac 5.0 闭环栈在本机反复死在 `warp.cu cuda_init`：
`Warp CUDA error: Failed to get driver entry point 'cuDeviceGetUuid'` → SIGSEGV。
**旧命令同样崩 ⇒ 与代码无关**（见 v4 方案 O15）。驱动要求：**570.124.06**（当前宿主 595.71.05）。
在这条修好之前，Stage 1（B 采集）与 Stage 5（真物理闭环验收）都跑不了；
训练/评测栈（trackerlab，Isaac 5.1）不受影响。

---

## 0.1 一句话

链路是通的，**没有发现起点/参考/token 的 bug**（见 §6 证据）；本次 GT 臂摔倒 = 跟踪漂移导致的
后仰失衡，**不是碰桌子**。所以后面只做一件事：**把执行漂移压下去**。训练前必须先过 §2 的 G0。

---

## 1. 现状（2026-10-28）

| 项 | 状态 |
|---|---|
| token 通路（fed vs label） | ✅ `mean|Δ|=0.023`，corr **0.925**（差的是实时 anchor，设计如此） |
| 参考帧号 | ✅ online/http 模式 `frames` = 真实库行号（http 侧已修，见 §5） |
| 起点对齐 | ✅ GT 臂 t=0 根误差 **0.002 m** |
| GT 臂摔倒原因 | ✅ 漂移后仰（tilt 6.8°→76°，在两张桌子中间的空地），**非接触** |
| VLA 臂"不摔" | ⚠️ 假象：200 步后站住不动，根误差 2.07 m（没干活） |
| `trust_center=prev` | ✅ 已实现（LoRA 快照 = π_k，同噪声、无标签） |
| 训练起点 = B 重规划状态 | ✅ 已实现（`--start-from-b`，不 burn-in、不叠 DR） |
| WM 刷新验收 | 🟡 已跑（`docs` 见 `logs/wm_finetune_v2b.log`），门见 §2 G0-b |
| 物体/接触 WM | ❌ 未做（本阶段不需要） |

---

## 2. 三个门（按顺序，过一个才进下一个）

### G0-a｜token 有没有"权限"（1 小时，**最重要**）

三臂 online 各 2000 步，同一 harness/起点/seed：

```
GT online / ORACLE（想象里优化跟踪误差的 token）/ VLA π0
指标：e_root(t) @ 300 / 600 / 2000、tilt、摔率
```

| 结果 | 动作 |
|---|---|
| ORACLE 的 e_root 显著低于 GT（> seed 间波动） | **GO**：token 是杠杆，继续 §3 |
| ORACLE ≈ GT 或更差 | **STOP**：token 没权限，回去改 WBC，不再训 head |

> 为什么这是第一门：head 能学到的最好结果就是 ORACLE。上界不行，训练一定白跑。

### G0-b｜WM 刷新验收

```bash
for c in runs/S1_handover/WBC_A24000_wmvel_cmd.pt runs/S1_handover/WBC_A24000_wmvel_v2.pt; do
  PYTHONPATH=. python3 scripts/eval_wm_velocity.py --ckpt $c \
      --packed runs/isaac_motion/ha_cmd/packed.pt --tokens label --tag $(basename $c)
done
```

| 结果 | 动作 |
|---|---|
| 刷新版 `qd_corr ≥ 0.70` 且不比 base（**0.834**）低 0.03 以上 | 用刷新版 |
| 否则 | 回滚，用 base，**不要**带病刷新 |

**2026-10-28 实测（Stage 0.2，只用录制库）**

| ckpt | qd_corr | qd_ratio | pos_err | rot_err | rollout reward |
|---|---|---|---|---|---|
| base `WBC_A24000_wmvel_cmd.pt` | 0.83406 | 0.9987 | 1.41 mm | 0.552° | 0.09104 |
| refreshed `WBC_A24000_wmvel_v2.pt` | **0.82698** | 1.0020 | 1.30 mm | 0.594° | 0.08803 |

判定：`qd_corr` −0.0071（阈值 0.03 内）⇒ 一步口径 PASS；但 **多步口径 FAIL**：

```
scripts/eval_wm_multistep.py --dump 'runs/B_*/dump/*.pt' --horizon 20
base: k=20 joint 0.154 rad / root 9.2 cm
v2  : k=20 joint 0.214 rad (+39%) / root 5.8 cm (-37%)
```

⇒ **第 1 轮用 base WM**；刷新件 v2 不进流程。

**2026-10-28 方案 1（多步 + B 轨迹）已跑通**：

```
finetune_wm_velocity_policy.py --rollout-steps 3 --traj-dump 'runs/B_*/dump/*.pt' \
  --traj-frac 0.2 --traj-val-frac 0.25 --val-every 50 --steps 300 --lr 3e-5
→ 留出 B 尾段 20 步关节 0.168 → 0.129 rad (-23%)，根 9.3 → 4.0 cm (-57%)  PASS
产物：runs/S1_handover/WBC_A24000_wmvel_v4.pt
```

### G0-b′｜critic 能不能用（10 分钟，留出集才算）

```bash
PYTHONPATH=. python3 scripts/fit_critic_real.py \
  --transitions runs/wm_real/real_transitions_v2b.pt \
  --out runs/wm_real/critic_real_v1.pt --center mean --use-diff --epochs 3
```

门：**留出 episode 上 `v_ev > 0.30`**（不是训练日志里的 `critic/v_ev`）。
2026-10-28 实测：linear 去趋势 **−0.06 / −0.13**；mean 去均值 **+0.21** ⇒ **未过门**，
默认仍 `--no-critic`；想试就开臂 C（`CRITIC_ON=1 CRITIC_CKPT=...`），判据看真物理 `e_root`。

### G0-c｜A1 是否生效（10 分钟）

同 seed 两个 30 步 run，`--vla-state-from-burnin` 开/关，比 `token_shift` / `imag_return_norm`。
完全一致 ⇒ A1 没生效，先修；有可测差异 ⇒ 过。

---

## 3. 每轮流程（G0 过后）

```bash
# 1) B 线采集（5 条 × 400–600 步）—— 必须 online/http，dump 才会带真实库行号
OUT=$PWD/runs/B_r1_$(date +%H%M); mkdir -p $OUT/dump
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  TASK=DoubleDesk SEED=0 MAX_STEPS=600 ISAAC_GPU=0 \
  MODEL_LABEL=vla OUT_ROOT=$OUT/runs \
  SM_CKPT=$PWD/runs/S1_handover/WBC_A24000_wmvel_v2.pt \
  SM_TOKEN_SOURCE=http SERVER_URL=http://127.0.0.1:18080 SM_HAND=0 \
  SM_INIT_FROM_LIB=1 SM_START_FRAME=2080 \
  SM_DUMP_STATES=$OUT/dump/DoubleDesk HA_VLA_EXEC_STEPS=20 \
  bash scripts/ha_isaac_closedloop.sh > logs/$(basename $OUT)_DoubleDesk.log 2>&1

# 2) 补算真奖励（注意：脚本已自动把 asset 序转 standard）
SUPERMIMIC_HA_ALIGN=1 PYTHONPATH=.:third_party/SuperMimic-Reproduced/src \
  python3 scripts/compute_collect_reward.py --in $OUT/dump/*.pt --out $OUT/scored --device cuda:1

# 3) 重建真转移（含 B 线 on-policy）
PYTHONPATH=. python3 scripts/build_real_transitions_v2.py \
  --out runs/wm_real/real_transitions_v2.pt --extra-dumps "$OUT/dump/*.pt"

# 4) WM 刷新 + G0-b 验收（--extra-transitions 把 B 线 on-policy 混进去）
PYTHONPATH=. python3 scripts/finetune_wm_velocity_policy.py \
  --ckpt runs/S1_handover/WBC_A24000_wmvel_cmd.pt --out runs/S1_handover/WBC_A24000_wmvel_v2.pt \
  --packed runs/isaac_motion/ha_cmd/packed.pt --token-space label --steps 800 --batch 256 \
  --lr 1e-4 --vel-lambda 5 \
  --extra-transitions runs/wm_real/real_transitions_v2.pt --extra-frac 0.5

# 5) 联合微调（中心 = π_k，起点 = B 重规划状态，无标签，无 critic）
GPU=0 STEPS=200 V3=1 TRUST_CENTER=prev START_FROM_B="$OUT/dump/*.pt" \
  LAMBDA_R=0.3 LAMBDA_BC=0 ACT_EPS=0.6 ACT_WEIGHT=1.0 \
  BATCH=2 ACCUM=1 H=20 K=4 EVAL_EVERY=50 EVAL_BATCHES=8 \
  EXTRA="--amp bf16 --lora-rank 32 --fm-lr 1e-4 --no-dr --vla-state-from-burnin \
         --no-critic --act-hinge-cap 2 --lambda-hand 0.2" \
  RUN=s3_r1 bash scripts/launch_joint_ha_v2.sh

# 6) 闭环验收（head 换进 server 后跑 B 线，≥3 seeds，同 harness）
python3 scripts/ha_vla_latent64_server.py \
  --ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --head-ckpt runs/s3_r1/joint_000200.pt --device cuda:0 --port 18081
```

**每轮验收（唯一判据）**

```
e_root(t) 的 300/600/2000 步值 + tilt + 摔率，≥3 seeds
π_k 比 π_{k-1} 低、且超过 seed 间波动 ⇒ 继续
否则 ⇒ 见 §4 止损
```

---

## 4. 止损门

| 条件 | 动作 |
|---|---|
| 连续两轮 `e_root` 不降 | **停**：不再加步数、不再调超参 |
| 想象误差降、真物理不降 | 停训练，回到 WM 刷新（G0-b）；WM 在 B 状态上不准 |
| 训练开始后 `token_mse` > 1.5× 或 `codec/out_of_range` 单调升 | 停：已离开 WM 覆盖区（canary） |
| GT 臂在**同起点**也压不下漂移 | 停：瓶颈是 WBC/接触，回 WBC（不在本 Runbook 范围） |

---

## 5. 已修的坑（本轮）

| 坑 | 修复 |
|---|---|
| 启动器 `WBC_CKPT` 默认旧 ckpt（wbc_mix2） | 默认改成 `WBC_A24000_wmvel_cmd.pt` |
| 启动器默认传 `--port-tokens data/port_tokens`（与 lockdown 冲突） | 默认不传；`--bc-token-space label` 显式写死；`PORT_DIR` 空 = 不用 |
| `trust_center` 只有 label/base | 加 `prev`（LoRA 快照 = π_k，同噪声采样，无标签） |
| 训练起点 = 录制状态（梯度没信号） | 加 `--start-from-b`：B 线真实重规划状态，不 burn-in、不叠 DR |
| B dump 的 `states` 是 **asset(HA_V31) 序**，却直接喂 bundle | `compute_collect_reward.py` / `build_real_transitions_v2.py` 统一先转 standard |
| http 模式 dump 的 `frames` 是**步号**（只有跑 clip 0 才碰巧对） | provider 的 http 分支补了参考行号游标（`SM_TOKEN_START` + 步数，夹到 clip 末），meta 加 `frames_source` |
| `v2_gates` 在 B 起点下 `burnin=None` 会崩 | 加空值保护 |
| 启动器默认**不传 `--v3`** ⇒ `--trust-center/--act-*/--lambda-hand` 全是空转 | 加 `V3=1`；`V3=0` 且用了这些参数时**打 WARN** |
| WM 刷新脚本不读 `real_transitions_v2.pt`，B 线数据永远进不了 WM | 加 `--extra-transitions/--extra-frac`（已冒烟） |
| 启动器不传 `--v3`（见上）；另外 WBC 解冻开关也没有 | 加 `V3=1`、`TRAIN_WBC=1 WBC_LR=...`；解冻臂的参考 rollout 用 π_k 的 WBC 快照（已冒烟，WM 逐位不变） |
| `compute_collect_reward` 把 asset 序当 standard 序算奖励 | 统一先转 standard（与 `--start-from-b` 同一个函数） |
| WM 动作语义搞混：录制侧是 `q_target − default_q`（弧度），dump 的 `actions` 是 `a_norm`，差 per-joint scale（mean 0.379） | `build_real_transitions_v2` 的 extra 侧统一换算成 `q_target − default_q`（不要用 `e["actions"]`） |
| trainer 的 `--critic` **默认是开**，不带 `--no-critic` 就会用随机 critic | 启动器默认传 `--no-critic`，只有 `CRITIC_ON=1` 才开 |

---

## 6. 本次排查的证据（可复现）

```text
token：dump.tokens vs npz.g1_motion_token_64[0:1437]
       mean|Δ|=0.023，corr=0.925（差的是 live anchor）
起点：GT dump t=0 根误差 0.002 m
摔倒：t=1397 tilt 6.8° → t=1432 tilt 76.1°，根位置在 (-2.4,-1.2)
       两个桌子物体在 y≈-3.05 / y≈-0.04，摔倒点在中间空地
ego   ：1408/1416 画面朝天（后仰），不是撞桌
漂移：根误差 300 步 0.10 m → 600 步 0.20 m → 摔倒 0.82 m
```

---

## 7. 已知债务（不阻塞）

1. 图像通道仍是录制 ego 帧（B 线当时渲染画面没落盘）；本体感觉已用真实状态（A1）。
2. `--start-from-b` 的训练/评测切分是**同一批 dump** 按行号取模 5 切的（不是跨 episode 留出）。
3. ORACLE 三臂还没在 online 模式下重跑（旧的 npy 回放数据无效）。
4. critic 仍关（`v_ev < 0`）；本阶段不用。
