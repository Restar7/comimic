# C 线 λ 修正主线 · 完整交接计划（新会话可直接接手）

> 更新：2026-11-10 02:40 CST ｜ 执行者：C 线 Agent
> **本文自包含**：读完这一份就能接着跑，不需要翻别的文档。
> 交付物保护：`lora300_ddp/checkpoint_lora_best.pt` **全程不动**；新主线单独目录；任一 Gate 不过即回滚。

---

## 0. 30 秒状态（接手先看这里）

| 事项 | 状态 |
|---|---|
| **正在跑** | `p14/main300`：**2 卡 DDP、从 scratch 300 步**，`λ_lim=0.00832` + `λ_margin=0.0679`（2026-11-10 **02:33 起**，约 6.1 s/步 ⇒ **~03:10 完成**） |
| 日志 | `runs/dual_actor_wm6/c_lora/p14/train_main.log` |
| 输出 | `runs/dual_actor_wm6/c_lora/p14/main300/` |
| 已完成 | ① H=10/WM6 **重标定**（`p13/calib_h10/`）；② 发现了 `λ_lim` 超罚 3.7 倍的证据 |
| 待做 | 训练健康检查 → Isaac 开环门 → **先单评 VisionNavi（6 seeds）** → 不行调 λ_recovery → 过了再护栏 → **6 任务 × 6 seeds（只跑 C，不跑 base）** |
| 起步检查 | `ls runs/dual_actor_wm6/c_lora/p14/main300/summary.json` 是否已生成 |

---

## 1. 必读事实（不要重新猜）

### 1.1 资产与 SHA

| 对象 | 路径 | SHA / 说明 |
|---|---|---|
| **C 交付 ckpt（旧）** | `comimic-c-wm6/runs/dual_actor_wm6/c_lora/critic10/lora300_ddp/checkpoint_lora_best.pt` | sha256 `6942eab9a72f3d39…`；218 层 / rank16 / alpha32 / 16,891,904 参数；meta `{update:100, dJ_val:0.0045587}` |
| **WM6** | `comimic-wt03/runs/wm_base6_v1/wm_candidates/wm_base6_best_candidate/wm.pt` | sha `8edb59cf2adddf0a…`（**必须 `--strict-wm-hash 8edb59cf`**） |
| FT8000 | `comimic/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000` | config sha `7d06d0da…` |
| VLM | `comimic/groot_repro/nvidia/Cosmos-Reason2-2B-weights` | |
| SONIC decoder | `GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx` | sha `c7241a12…`（**不是** low_latency） |
| critic 预训 | `comimic-c-wm6/runs/dual_actor_wm6/c_lora/critic_pretrain_tail10/critic_pretrain.pt` | `in_dim=1470`、`plan_steps=10`、`H=10/K=20` |
| **重标定产物** | `comimic-c-wm6/runs/dual_actor_wm6/c_lora/p13/calib_h10/` | 见 1.2 |

### 1.2 重标定结果（H=10 / WM6 / 当前 adapter，48 窗）

| 项 | ‖g‖ | cos(g,gJ) | **λ（标定）** | V8C 旧 λ（H20/WM0） |
|---|---|---|---|---|
| J（基准） | 0.0390 | — | — | 0.0962 |
| **margin** | 0.0429 | **+0.463** | **0.067918** | 0.2201 |
| recovery | 2.235 | +0.192 | **0.005214** | 0.002223 |
| abs(lim) | 1.868 | +0.417 | **0.008321** | 0.03068 |
| added | 0.00713 | **−0.464** | 0.817 | 6.782 |
| peak | 1231 | **−0.480** | — | 1.8e-5 |

**新 z0**（`p13/calib_h10/recovery_z0.json`，**不能再用 V8C 的**，差 1.8 倍）：
`root_pos 0.1784 / root_ori 0.4649 / body_linvel 0.5730 / body_angvel 0.9380`

**核心结论**：现配方 `λ_lim=0.0307` 的梯度占比 = `0.0307×1.868/0.039` = **147%**（同向分量 61%）⇒ **限位惩罚主导，是当前收益饱和的嫌疑主因**。改成标定值 `0.00832` 后回到 40% 预算。

### 1.3 当前闭环基线（**按 seed 去重**，成功率=该 seed 至少成功一次）

| 任务 | C(lora300_ddp) | C-V8C | base(B线) | base(r3) |
|---|---|---|---|---|
| OpenDoor | **6/6** | 4/6 | 2/3 | 6/6 |
| SitSofa | **5/6** | 2/6 | 2/3 | 2/6 |
| Boxing | 4/6 | 5/6 | 2/3 | 4/6 |
| VisionNavi | **3/6（+2 摔）** | 4/6 | 1/3 | 3/6 |
| PPBox | 2/6 | — | 1/4 | 2/6 |
| DoubleDesk | 0/6 | — | 0/3 | 0/6 |

**问题靶子**：VN 3/6 + 2 次摔倒；Boxing 4/6。
**护栏**：OpenDoor 6/6、SitSofa 5/6 不能掉超过 1 个 seed。

### 1.4 环境与踩过的坑（不读会浪费一小时）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
source scripts/harness03_env.sh        # 提供 trackerTask/trackerLab 的 PYTHONPATH
```

| # | 坑 | 处理 |
|---|---|---|
| 1 | `pkill -f` 匹配到自己的命令行 | 用 `pgrep -f "train_sonic_lora16_ft8000_bptt[.]py"`（方括号破坏自匹配） |
| 2 | 闭环 driver 起 server 报 `cd: too many arguments` | 已在 `c2_driver.py` 修复（缺 `&&`）；需 commit ≥ `8a8d97e` |
| 3 | 同 seed 结果不可复现 | 已固定 server 采样种子 `C2_SERVER_SEED=20261109` |
| 4 | Isaac 报告脚本 `NameError: MJTNUM_BYTES` | **不要 source harness**，用干净 PYTHONPATH：`PYTHONPATH=<wt03>:<main>/trackerlab-supermimic/source/{trackerLab,trackerTask} python3 …` |
| 5 | Isaac 回放存完 `roll.pt` 不退出 | 轮询文件出现 + `sleep 8` 后 `kill -9` |
| 6 | 视频只有 20 s | 那是 `__tmp.mp4`（仍在写）；完成品是 3001 帧 / 100 s / 30 fps |
| 7 | `--record-video N` 不抽帧 | 这条链路实际每步都录；想快就设 0 |
| 8 | 训练 OOM | 闭环在跑时不要同卡训练；训练 batch 8+accum4 可省一半显存 |
| 9 | `c2_reward.py` 的 J10/J20 | **不可引用**（`frame_count` 是局部计数器）；只用 `success/episode_steps/joint_limit/action_rate` |

---

## 2. 完整执行流程（逐阶段、带 Gate）

```
[A] 等 p14 训练完成
      ↓ Gate A：训练健康度（L_margin↓、λ 占比、越限不恶化、无 NaN）
[B] Isaac 开环门（15 min）
      ↓ Gate B：ΔJ_Isaac ≥ +0.00329（=旧 ckpt 的 90%）
[C] ★ 先单评 VisionNavi（6 seeds，1.2 h）
      ↓ Gate C：VN ≥ 4/6（现在是 3/6）且 fall ≤1（现在是 2）
      ├─ 过 → [D]
      └─ 不过 → [C'] 加 λ_recovery（或其他）→ 从 scratch 重训 300 步 → 回到 [B][C]
[D] 护栏回归：OpenDoor/SitSofa/Boxing（各 4-6 seeds）
      ↓ Gate D：OpenDoor ≥5/6、SitSofa ≥4/6、Boxing ≥4/6
[E] 全六任务 × 6 seeds（只跑 C，不跑 base）
      ↓ Gate E：逐任务不降 + VN/Boxing 至少 +2 seed + fall 减少
[F] 写终稿 + 决定是否替换交付物
```

---

## 3. 阶段 A：等训练完成 + 健康检查

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
ls runs/dual_actor_wm6/c_lora/p14/main300/summary.json     # 出现即完成
grep -oE "it= +[0-9]+ J=[0-9.-]+ .*Llim=[0-9.]+.*Lmar=[0-9.]+" runs/dual_actor_wm6/c_lora/p14/train_main.log | tail -3
python3 -c "
import json
d=json.load(open('runs/dual_actor_wm6/c_lora/p14/main300/summary.json'))
print('updates',d['updates'],'best',d['best']); print('test',{k:d['test_once'][k] for k in ('dJ','dJ_ci_lo','dJ_ci_hi','limit_violation_frac','q_raw_abs_max')})"
```

**Gate A（全过才继续）**

| # | 判据 | 阈值 |
|---|---|---|
| 1 | `updates` | == 300 |
| 2 | `L_margin` | 首段→末段**下降**。⚠️ 它的**损失贡献很小是正常的**（实测 `Lmar≈0.013`、`λ·L≈0.0009`，占 J 仅 0.1%）——margin 的强度是**梯度预算 7.5%**（标定保证），**不要用损失占比判断**；只看它是否在下降、以及越限率是否改善 |
| 3 | `λ_lim·L_lim` | 损失占 J 的 **2–5%**（旧配方 4–8%）；梯度占比应为 **~40%**（旧配方 147%） |
| 4 | val 越限率 | ≤ **2.9%**（pristine 2.81%） |
| 5 | `q_raw_absmax` | ≤ **4.5** |
| 6 | NaN / grad 爆炸 | 无 |

> 若 Gate A 的 4/5 恶化：λ_lim 折中到 **0.012**（60% 预算）重训，其余不变。

---

## 4. 阶段 B：Isaac 开环门（15 min）

```bash
# B1 生成 job（~3 min）
export WT=$PWD; set +u; source scripts/harness03_env.sh
python3 scripts/c_wm6/probe_c_vs_base_isaac_plan.py \
  --adapter runs/dual_actor_wm6/c_lora/p14/main300/checkpoint_lora_best.pt \
  --eval-set ../comimic-wt03/runs/temporal_ac_bptt_v5/eval_set \
  --H 10 --per-task 8 --select random --split-key test --seed 777 \
  --wm ../comimic-wt03/runs/wm_base6_v1/wm_candidates/wm_base6_best_candidate/wm.pt \
  --strict-wm-hash 8edb59cf \
  --out runs/dual_actor_wm6/audit/isaac_wm6_h10_p14/job.pt --device cuda:0

# B2 真物理回放（~3 min，带 hang guard）
cd ../comimic && set +u; source scripts/isaac_env.sh >/dev/null 2>&1
unset LD_PRELOAD CUDA_VISIBLE_DEVICES
export DISPLAY=${DISPLAY:-:99} XDG_RUNTIME_DIR=/tmp/xdg; mkdir -p "$XDG_RUNTIME_DIR"
D=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6/runs/dual_actor_wm6/audit/isaac_wm6_h10_p14
rm -f "$D/roll.pt"
$ISAAC_VENV/bin/python -u scripts/isaac_probe_action_rollout.py --job "$D/job.pt" --out "$D/roll.pt" --num-envs 24 --device cuda:0 --headless > /tmp/roll_p14.log 2>&1 &
PID=$!; for i in $(seq 1 120); do [ -f "$D/roll.pt" ] && break; sleep 5; done
[ -f "$D/roll.pt" ] && { sleep 8; kill -9 $PID; echo ROLL_OK; } || { kill -9 $PID; echo ROLL_TIMEOUT; }

# B3 报告（★ 用干净 PYTHONPATH，否则 mujoco 报 MJTNUM_BYTES）
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
export PYTHONPATH="$PWD:/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic/source/trackerLab:/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic/source/trackerTask"
unset VIRTUAL_ENV
python3 scripts/probe_isaac_h20_report.py --job $D/job.pt --roll $D/roll.pt \
  --ckpt /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/runs/S1_handover/WBC_A24000_wmvel_cmd.pt \
  --wm-ckpt runs/wm_base6_v1/wm_candidates/wm_base6_best_candidate/wm.pt \
  --device cuda:0 --out $D/report.json
```

**Gate B**：`ΔJ_Isaac > 0` **且 ≥ +0.00329**（旧 ckpt `+0.003646` 的 90%）。
`ΔJ_WM` 只记录，**不参与判断**（corr 0.14–0.17，会误导）。

---

## 5. 阶段 C：★ 先单评 VisionNavi（这是"是否有效"的第一判据）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
source scripts/harness03_env.sh
C2_SERVERS=c C2_PORT_C=18080 C2_GPU_C=0 C2_ISAAC_GPU=0 \
nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks VisionNavi --seeds 0 1 2 3 4 5 --arms c \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 2 \
  --out runs/dual_actor_wm6/c_lora/c2/p14_nav \
  --lora-ckpt runs/dual_actor_wm6/c_lora/p14/main300/checkpoint_lora_best.pt \
  --lora-tag p14 > runs/dual_actor_wm6/c_lora/c2/p14_nav.log 2>&1 &
```

- 单卡 6 集，约 **1.2 h**；
- 结果看 `p14_nav/c/VisionNavi_seed*/…/result.json` 与 `p14_nav.log` 的 `rc=` 行；
- 失败集也会跑满 3000 步（~9–12 min），成功集会提前结束。

**Gate C（硬判据）**

| 判据 | 阈值 |
|---|---|
| **成功率（通过）** | **≥ 4/6**（当前 3/6，+1 seed 即算改善） |
| **成功率（最优）** | 5/6 以上更好，但不是必须 |
| **摔倒** | **≤ 1 次**（当前 2 次） |
| **部分通过** | 3/6（不变）但摔倒 ≤1 → 安全面改善、成功率没动 ⇒ 进 [C'] 只调 recovery |
| **未通过** | ≤2/6 或摔倒仍 ≥2 |

### 5.1 阶段 C'：没过怎么办（按顺序试，每次只改一个变量）

| 顺序 | 改什么 | 配置 | 说明 |
|---|---|---|---|
| **C'-1** | **加 λ_recovery** | `--lambda-recovery 0.005214`、`--recovery-z0-json p13/calib_h10/recovery_z0.json`、`--recovery-aggregate worst_window --recovery-worst-t 0.25 --recovery-worst-b 0.10` | 预算 30%；**从 scratch 300 步重训**（保持血缘干净） |
| **C'-2** | λ_recovery 降到一半 | `0.0026` | 若 C'-1 让护栏掉 |
| **C'-3** | λ_margin 加倍 | `0.1358` | 只针对摔倒 |
| **C'-4** | λ_lim 再降 | `0.0042`（20% 预算） | 若怀疑限位仍主导 |

**注意**：每次重训后必须**重跑 Gate B（开环门）+ Gate C（nav 6 seeds）**，不允许只看训练曲线。

---

## 6. 阶段 D：护栏回归（Gate C 通过后）

```bash
C2_SERVERS=c C2_PORT_C=18081 C2_GPU_C=1 C2_ISAAC_GPU=1 \
python3 scripts/c_wm6/c2_driver.py --tasks OpenDoor SitSofa Boxing --seeds 0 1 2 3 --arms c \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 2 \
  --out runs/dual_actor_wm6/c_lora/c2/p14_guard \
  --lora-ckpt runs/dual_actor_wm6/c_lora/p14/main300/checkpoint_lora_best.pt --lora-tag p14
```

**Gate D**：OpenDoor **≥5/6**（当前 6/6）、SitSofa **≥4/6**（当前 5/6）、Boxing **≥4/6**（当前 4/6）。
任一掉 >1 个 seed ⇒ **判失败**，回 [C'] 调 λ（先减 margin，再考虑减 recovery）。

---

## 7. 阶段 E：全六任务 × 6 seeds（**只跑 C，不跑 base**）

**触发**：Gate C + Gate D 全过。

两卡并行，共 **36 集**，约 3–4 h：

```bash
# 卡0
C2_SERVERS=c C2_PORT_C=18080 C2_GPU_C=0 C2_ISAAC_GPU=0 \
python3 scripts/c_wm6/c2_driver.py --tasks PPBox DoubleDesk SitSofa --seeds 0 1 2 3 4 5 --arms c \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 2 \
  --out runs/dual_actor_wm6/c_lora/c2/final_p14_A \
  --lora-ckpt runs/dual_actor_wm6/c_lora/p14/main300/checkpoint_lora_best.pt --lora-tag p14

# 卡1
C2_SERVERS=c C2_PORT_C=18081 C2_GPU_C=1 C2_ISAAC_GPU=1 \
python3 scripts/c_wm6/c2_driver.py --tasks VisionNavi OpenDoor Boxing --seeds 0 1 2 3 4 5 --arms c \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 2 \
  --out runs/dual_actor_wm6/c_lora/c2/final_p14_B \
  --lora-ckpt runs/dual_actor_wm6/c_lora/p14/main300/checkpoint_lora_best.pt --lora-tag p14
```

**Gate E（最终）**

| 判据 | 阈值 |
|---|---|
| 逐任务不下降 | 相对 1.3 表 ≤1 seed 波动 |
| 主目标 | VN 或 Boxing **≥ +2 seed** |
| 安全 | fall 总数 **下降**（当前 5 次，VN 占 4） |
| 对外表述 | 与 base 的比较标注"**非配对（不同 seed）**" |

---

## 8. 汇总与判读脚本（可直接用）

```bash
python3 - <<'PY'
import json, glob, os
from collections import defaultdict
base="runs/dual_actor_wm6/c_lora/c2"
agg=defaultdict(dict)
for d in ("final_p14_A","final_p14_B","p14_nav","p14_guard"):
    for f in glob.glob(f"{base}/{d}/c/**/result.json", recursive=True):
        r=json.load(open(f)); agg[d].setdefault(r["model_label"],[]).append(
            (r["seed"], r["success"], r["failure_reason"], r["episode_steps"]))
for d,tasks in agg.items():
    print("==",d)
    for t,rows in sorted(tasks.items()):
        # 按 seed 去重：一个 seed 成功过即算成功
        by={}
        for s,ok,fr,st in rows: by[s]=by.get(s,False) or ok
        ok=sum(by.values()); n=len(by)
        print(f"  {t:11s} {ok}/{n}  " + " ".join(f"s{s}={'Y' if v else 'x'}" for s,v in sorted(by.items())))
PY
```

---

## 9. 回滚与产物管理

| 情况 | 动作 |
|---|---|
| 任一 Gate 不过 | 丢弃该阶段目录（`p14/` 可整体删）；`lora300_ddp` 不动 |
| 代码 | 全部为**可选开关**（`--lambda-lim/--lambda-margin/--lambda-recovery` 等），不传即原配方 |
| 脚本 | `runs/` 被 gitignore ⇒ `run_p13_ddp.sh` 只在本机；**换机器需重新落盘**（内容见附录 A） |
| 交付物 | Gate E 通过前始终是 `lora300_ddp/checkpoint_lora_best.pt`；通过后再决定替换与写报告 |

---

## 10. 时间线

| 阶段 | 成本 |
|---|---|
| A 训练完成 + 健康检查 | ~35 min（进行中） |
| B Isaac 开环门 | ~15 min |
| **C nav 6 seeds** | **~1.2 h** |
| C'（如需要，每次） | 重训 35 min + 门 15 min + nav 1.2 h ≈ **2 h/轮** |
| D 护栏 | ~1.5 h |
| E 全六任务 36 集 | ~3–4 h |

---

## 11. 新会话接手清单（照着做）

1. `date`、`nvidia-smi`：看有没有残留进程/占卡；
2. `ls runs/dual_actor_wm6/c_lora/p14/main300/summary.json`：训练是否已完成；
3. 若完成 → 走 §3 Gate A；若还在跑 → 等（`grep "it="` 看进度）；
4. 若发现进程死了且没有 summary → 看 `p14/train_main.log` 尾部，**不要自动重跑**，先判因（OOM / NaN / 被杀）；
5. 之后严格按 §4 → §5 → §6 → §7 的顺序，**不要跳 Gate**。

---

## 附录 A：`run_p13_ddp.sh` 关键内容（换机器时重建）

```bash
#!/usr/bin/env bash
set +u
WT=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
MAIN=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
WT03=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
source "$WT/scripts/harness03_env.sh"; cd "$WT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset CUDA_VISIBLE_DEVICES
mkdir -p "$OUT"
exec python3 -m torch.distributed.run --standalone --nproc_per_node=2 --master_port=29561 \
  scripts/train_sonic_lora16_ft8000_bptt.py \
  --ft8000-checkpoint "$MAIN/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000" \
  --vlm-path "$MAIN/groot_repro/nvidia/Cosmos-Reason2-2B-weights" \
  --eval-set "$WT03/runs/temporal_ac_bptt_v5/eval_set" \
  --limits-npz "$WT03/runs/three_track_v6/physical/limits_standard29.npz" \
  --out "$OUT" --bptt-args "$MAIN/runs/bptt1/args.json" \
  --packed-motion "$MAIN/runs/isaac_motion/ha_cmd/packed.pt" \
  --wm-checkpoint "$WT03/runs/wm_base6_v1/wm_candidates/wm_base6_best_candidate/wm.pt" \
  --strict-wm-hash 8edb59cf2adddf0a2f65b6bb573e7c529e9d2a3b9e15ee621f050802aaa4af74 \
  --critic-grad-measure-every 5 --critic-plan-steps "${PLAN_STEPS:-10}" \
  --horizon 10 --rollout-k 20 --critic-enabled 1 --critic-kind v5 \
  --critic-warm-from "${WARM_FROM}" --beta 0.975 --beta-warmup-updates 100 --beta-gate none \
  --critic-grad-budget 0.20 --critic-updates 4 --critic-batch 2048 --critic-lr 3e-4 \
  --updates "${UPDATES:-300}" --lr-total-updates "${UPDATES:-300}" --warmup 20 \
  --lr "${LR:-1e-4}" --min-lr 1e-6 --batch "${RANK_BATCH:-16}" --accum 1 \
  --save-every 25 --eval-every 50 --c-recipe c1_lora_proj --seed "${SEED:-0}" --ddp 1 \
  ${EXTRA_ARGS:-}
```

调用示例（本次主线）：

```bash
OUT=runs/dual_actor_wm6/c_lora/p14/main300 LR=1e-4 UPDATES=300 PLAN_STEPS=10 RANK_BATCH=16 \
WARM_FROM=runs/dual_actor_wm6/c_lora/critic_pretrain_tail10/critic_pretrain.pt \
EXTRA_ARGS='--lambda-lim 0.00832 --lambda-margin 0.067918 --margin-kappa 0.05 \
            --recovery-z0-json runs/dual_actor_wm6/c_lora/p13/calib_h10/recovery_z0.json' \
bash runs/dual_actor_wm6/c_lora/run_p13_ddp.sh
```

---

## 附录 B：一句话总结

**λ_lim 从超罚 3.7 倍拉回 0.00832、加 margin 0.0679，从 scratch 300 步重训 → 开环门 → 先单评 VisionNavi 6 seeds（要 ≥5/6 且摔倒 ≤1）→ 不过就加 λ_recovery（0.00521 + 新 z0）重训重评 → 过了再跑护栏（OpenDoor/SitSofa/Boxing）→ 都不破坏才做六任务 × 6 seeds、只跑 C。**
