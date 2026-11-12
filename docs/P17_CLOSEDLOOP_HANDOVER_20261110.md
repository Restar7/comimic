# p17 闭环评测交接（2 卡 4090 · 六任务 × 6 seeds）

> 生成于 2026-11-10 ｜ 目标机器：**2 × RTX 4090（同目录结构）** ｜ 只跑推理，不训练
> 一句话：用 **p17 的 LoRA 适配器**在 Isaac 里跑 `OpenDoor / SitSofa / Boxing / VisionNavi / PPBox / DoubleDesk` 各 **seeds 0–5**，重点是量化"VN 变强"与"OpenDoor 退化"这笔取舍。

---

## 0. 30 秒速览

| 事项 | 内容 |
|---|---|
| 适配器 | `comimic-c-wm6/runs/dual_actor_wm6/c_lora/p17/dyn300/checkpoint_lora_best.pt` |
| sha256 前 16 | `fc64ce28310abb17` |
| 期望 LoRA meta | `update=300`、`dJ_val≈0.0039896`、`physical.lim=0.00832`、440 tensors / 218 层 / rank16 / alpha32 |
| p17 是什么 | **λ_lim 下调到 0.00832 + 不加 margin + 动态起点加权**（S1 参考加速度 + S2 闭环失败状态就近；权重文件 `p17/dyn_weights.pt`） |
| 已测结论（本次会话） | VN **4/6、0 摔**（交付物 3/6、摔 2，且把两次摔倒的 seed 直接变成成功）；**OpenDoor seed0 超时**；PPBox seed3 超时 |
| 本交接要补的 | OpenDoor seeds 1–5（**量化代价**）、Boxing、SitSofa、PPBox 其余 seed、DoubleDesk |
| 规模/耗时 | 36 集；两卡并行各 18 集，**约 1.5–2.5 小时** |
| 判据 | VN 是否稳在 4/6 左右且 0 摔；**OpenDoor 掉多少**（这是决定能不能替换交付物的关键） |

---

## 1. 资产与前置检查

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6

# 1) 适配器在位、哈希对得上（必须一致，否则不是 p17）
ls -la runs/dual_actor_wm6/c_lora/p17/dyn300/checkpoint_lora_best.pt
sha256sum runs/dual_actor_wm6/c_lora/p17/dyn300/checkpoint_lora_best.pt | cut -c1-16
#   → 期望 fc64ce28310abb17

# 2) meta 自检
python3 -c "
import torch;d=torch.load('runs/dual_actor_wm6/c_lora/p17/dyn300/checkpoint_lora_best.pt',map_location='cpu',weights_only=False)
m=d.get('meta',{});print('update',m.get('update'),'dJ',m.get('dJ_val'),'phys',m.get('physical'))"
#   → update 300 / dJ 0.0039895722 / phys {'lim': 0.00832,...}

# 3) 其它必需资产（与 p16 相同）
ls -d /inspire/.../comimic/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000
ls -d /inspire/.../comimic/groot_repro/nvidia/Cosmos-Reason2-2B-weights
ls -d /inspire/.../comimic-wt03/runs/temporal_ac_bptt_v5/eval_set
ls    /inspire/.../comimic-wt03/runs/three_track_v6/physical/limits_standard29.npz
ls    /inspire/.../comimic/scripts/ha_isaac_closedloop.sh
ls -d /inspire/.../comimic/third_party/isaacsim_env/bin
ls -d /inspire/.../comimic/third_party/HumanoidArena
```

> ⚠️ `comimic/scripts/ha_isaac_closedloop.sh` 里 `COMIMIC=` 是写死的绝对路径，换机器必须先改。
> 注：**评测只需要适配器 `.pt`**，`dyn_weights.pt` 是训练期采样权重，闭环评测**不需要**它。

---

## 2. 一次性准备

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u
source scripts/harness03_env.sh
export C2_SERVER_SEED=20261109          # ★ 固定采样种子
```

**冒烟测试（必做，约 5 分钟）**

```bash
C2_SERVERS=c C2_PORT_C=18080 C2_GPU_C=0 C2_ISAAC_GPU=0 \
python3 scripts/c_wm6/c2_driver.py --tasks Boxing --seeds 2 --arms c --max-steps 3000 \
  --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/p17_smoke \
  --lora-ckpt runs/dual_actor_wm6/c_lora/p17/dyn300/checkpoint_lora_best.pt \
  --lora-tag p17
```

通过标志：① 日志出现 **`[server][lora16] loaded ... meta={'update': 300, ...}`**；② `共 N 个 (task,seed) × 1 臂 = N 集`；③ 有 `rc=0` 且 `result.json` 落盘。
（参考：Boxing seed2 在 p17 上是成功，159 步量级。）

---

## 3. 正式跑：两卡两 driver

**分工**

| 卡 | 端口 | 任务 | seeds |
|---|---|---|---|
| 0 | 18080 | `DoubleDesk Boxing VisionNavi` | 0–5 |
| 1 | 18081 | `OpenDoor SitSofa PPBox` | 0–5 |

**卡 0**

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh; export C2_SERVER_SEED=20261109
C2_SERVERS=c C2_PORT_C=18080 C2_GPU_C=0 C2_ISAAC_GPU=0 \
nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks DoubleDesk Boxing VisionNavi --seeds 0 1 2 3 4 5 --arms c \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/p17_6task_A \
  --lora-ckpt runs/dual_actor_wm6/c_lora/p17/dyn300/checkpoint_lora_best.pt \
  --lora-tag p17 > runs/dual_actor_wm6/c_lora/p17_6task_A.log 2>&1 &
```

**卡 1**

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh; export C2_SERVER_SEED=20261109
C2_SERVERS=c C2_PORT_C=18081 C2_GPU_C=1 C2_ISAAC_GPU=1 \
nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks OpenDoor SitSofa PPBox --seeds 0 1 2 3 4 5 --arms c \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/p17_6task_B \
  --lora-ckpt runs/dual_actor_wm6/c_lora/p17/dyn300/checkpoint_lora_best.pt \
  --lora-tag p17 > runs/dual_actor_wm6/c_lora/p17_6task_B.log 2>&1 &
```

> 若你只关心"取舍"这一件事，可以**先只跑 OpenDoor + VisionNavi（各 6 seeds，两卡各一个任务）**，约 40–60 分钟就能得到关键结论，其余任务再补。

---

## 4. 监控

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
grep -hE "rc=0" runs/dual_actor_wm6/c_lora/p17_6task_{A,B}.log | tail -6
for f in runs/dual_actor_wm6/c_lora/p17_6task_{A,B}.log; do echo -n "$f "; grep -c "rc=0" $f; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

---

## 5. 汇总

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
python3 scripts/c_wm6/agg_c2_episodes.py --roots p17_6task_A p17_6task_B --arm c --fall all
```

---

## 6. 基线对照（交付物 `lora300_ddp`，seeds 0–5，复用，不必重跑）

| 任务 | 交付物 | p17 目前已测 |
|---|---|---|
| OpenDoor | **6/6**（1013/1192/2214/548/772/2515） | **✗ seed0 超时**（seeds 1–5 待测 → **本次最关键的数据**） |
| SitSofa | **5/6**（s5 超时） | 未测 |
| Boxing | **4/6**（s0/s1 超时） | 未测（seed2 已确认成功 159 步） |
| VisionNavi | **3/6，摔 2**（s2 摔1832、s3 摔1006） | **4/6，0 摔**（s2 ✓2556、s3 ✓2824 把两次摔倒变成成功；s4 ✓1250、s5 ✓741；s0/s1 超时） |
| PPBox | **2/6**（✓s1 616、✓s4 683） | ✗ seed3 超时 |
| DoubleDesk | **0/6** | 未测 |

**判读口径**

- **主结论看两个数**：`VN 成功率/摔倒` 与 `OpenDoor 成功率`；
- 若 OpenDoor ≥5/6 ⇒ p17 只掉 1 个 seed，而 VN +1 seed 且 0 摔 ⇒ **这笔取舍是正的**；
- 若 OpenDoor ≤3/6 ⇒ p17 只能当"VN 专项"，交付物仍用旧 ckpt；
- SitSofa/Boxing/PPBox 用来确认退化不是普遍性的；DoubleDesk 已知全栈 0/6，可作噪声参考。

---

## 7. 已知坑

| # | 现象 | 处理 |
|---|---|---|
| 1 | 启动即报 `c.log 里没有 '[server][lora16] loaded'` | 适配器路径/哈希不对；按 §1 核对（p17 的 meta 是 **update=300**，别和 p16 的 100 混淆） |
| 2 | `FileNotFoundError: 'ss'` | 缺 `iproute2`；本仓库 `c2_driver.py` 已加兜底 |
| 3 | 端口冲突 | 换端口或清理残留 driver（`pkill -f "c2_driver[.]py"`，方括号防自匹配） |
| 4 | 两卡并发变慢 2× | Isaac 物理抢 CPU，正常 |
| 5 | 4090 24 GB | 每卡 server(~13.4 GB) + Isaac(~4–8 GB)，勉强够；先看 `nvidia-smi` |
| 6 | PPBox/DoubleDesk 大量超时 | 已知现象（交付物也是 2/6、0/6），不是环境坏了 |
| 7 | 别用 `c2_reward.py` 的 J10/J20 | `frame_count` 是局部计数器；只用 `success/episode_steps/failure_reason` |
| 8 | 磁盘 | 每集 ~190 个 `ticks_*.pt`，36 集预留 ≥20 GB |

---

## 8. 交付什么

1. `runs/dual_actor_wm6/c_lora/c2/p17_6task_{A,B}/`（含 `jobs.jsonl`、每集 `result.json`、`ticks/`）；
2. §5 聚合表；
3. 一句话结论：**VN 是否仍 4/6 + 0 摔、OpenDoor 掉到几个 seed**；
4. 异常集附 `episode.log` 尾部 20 行。
