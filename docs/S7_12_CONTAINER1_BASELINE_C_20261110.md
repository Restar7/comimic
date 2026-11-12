# 六任务闭环 · 容器 1：**基线 C 交付物**（seeds 7–12）

> 生成于 2026-11-10
> 目标：**容器 1（2 × RTX 4090）** 跑**基线 C 交付物 `lora300_ddp`** 的**全部六个任务**
> 配对关系：与**容器 2（官方 FT8000 base，不挂 LoRA）**使用**同一批 seeds 7–12、同一批任务**，
> 两容器结果合并即为**逐 (任务, seed) 配对**的 "C 微调 vs 官方 base" 对比。

---

## 0. 30 秒速览

| 事项 | 内容 |
|---|---|
| 评测对象 | `comimic-c-wm6/runs/dual_actor_wm6/c_lora/critic10/lora300_ddp/checkpoint_lora_best.pt` |
| sha256 前 16 | **`6942eab9a72f3d39`** |
| 期望 meta | `update=100`、`dJ_val≈0.0045587`、`physical.lim=0.03067937671760282`、440 tensors / 218 层 / rank16 / alpha32 |
| 臂 | `--arms c`（**要挂 LoRA**，与容器 2 的 `base` 臂区分） |
| seeds | **7 8 9 10 11 12**（预注册，跑完全报） |
| 规模 | **6 任务 × 6 seeds = 36 集**，两卡并行，各约 **2.2 小时** |
| 前置 | 两处补丁必须在位（§1），否则 seed 对不齐 |

---

## 1. 前置补丁（不满足就别开跑）

本评测的可比性依赖 driver/server 里的**每集独立播种**（`/reseed`）。
在容器 1 上确认：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
grep -c "EPISODE_SEED_BASE"  scripts/c_wm6/c2_driver.py          # 期望 3
grep -c 'path == "/reseed"'  scripts/ha_vla_latent64_server.py   # 期望 1
```

任一为 0 ⇒ 从主机拷这两个文件：
`scripts/c_wm6/c2_driver.py`、`scripts/ha_vla_latent64_server.py`。

**可复现性的准确表述**

- ✅ 同一 (task, seed) 的**采样噪声流**已确定（`seed = 20261109 + task_id*1000 + seed`），与队列位置无关；
- ⚠️ 单集步数仍有 **±30–60%** 抖动（CUDA kernel / Isaac 接触求解），小样本成功率偶有翻面；
- ⇒ 只按**多 seed 成功率**下结论，步数不作判据。

---

## 2. 一次性准备 + 冒烟

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u
source scripts/harness03_env.sh
export C2_SERVER_SEED=20261109

# 冒烟：Boxing seed7（快）
C2_SERVERS=c C2_PORT_C=18080 C2_GPU_C=0 C2_ISAAC_GPU=0 \
python3 scripts/c_wm6/c2_driver.py --tasks Boxing --seeds 7 --arms c --max-steps 3000 \
  --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/s7_12_container1/smoke \
  --lora-ckpt runs/dual_actor_wm6/c_lora/critic10/lora300_ddp/checkpoint_lora_best.pt \
  --lora-tag base
```

通过标志：① `[server][lora16] loaded ... meta={'update': 100, 'dJ_val': 0.0045587...}`；
② server 日志有 `[server] reseed seed=2026xxxxx`；③ 有 `rc=0` + `result.json`。

---

## 3. 正式跑（两卡各 3 个任务）

分工按历史耗时平衡（DoubleDesk 55′、PPBox 60′、VN 45′、SitSofa 40′、OpenDoor 35′、Boxing 30′）：

| 卡 | 任务（按此顺序） | 预计 |
|---|---|---|
| 0 | `DoubleDesk Boxing VisionNavi` | ≈130 min |
| 1 | `PPBox OpenDoor SitSofa` | ≈135 min |

**卡 0**

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh; export C2_SERVER_SEED=20261109
C2_SERVERS=c C2_PORT_C=18080 C2_GPU_C=0 C2_ISAAC_GPU=0 \
nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks DoubleDesk Boxing VisionNavi --seeds 7 8 9 10 11 12 --arms c \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/s7_12_container1/card0 \
  --lora-ckpt runs/dual_actor_wm6/c_lora/critic10/lora300_ddp/checkpoint_lora_best.pt \
  --lora-tag base > runs/dual_actor_wm6/c_lora/c2/s7_12_container1/card0.log 2>&1 &
```

**卡 1**

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh; export C2_SERVER_SEED=20261109
C2_SERVERS=c C2_PORT_C=18081 C2_GPU_C=1 C2_ISAAC_GPU=1 \
nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks PPBox OpenDoor SitSofa --seeds 7 8 9 10 11 12 --arms c \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/s7_12_container1/card1 \
  --lora-ckpt runs/dual_actor_wm6/c_lora/critic10/lora300_ddp/checkpoint_lora_best.pt \
  --lora-tag base > runs/dual_actor_wm6/c_lora/c2/s7_12_container1/card1.log 2>&1 &
```

---

## 4. 监控

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
grep -hE "rc=0" runs/dual_actor_wm6/c_lora/c2/s7_12_container1/*.log | tail -8
for f in runs/dual_actor_wm6/c_lora/c2/s7_12_container1/card*.log; do echo -n "$f "; grep -c "rc=0" "$f"; done
grep -h "reseed" runs/dual_actor_wm6/c_lora/c2/s7_12_container1/card*/servers/c.log | tail -6
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

---

## 5. 汇总（本容器）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
python3 scripts/c_wm6/agg_c2_episodes.py \
  --roots s7_12_container1/card0 s7_12_container1/card1 --arm c --fall all
```

上报：**逐任务成功数 / 摔倒次数 / 超时次数**（6 任务各一行）。

---

## 6. 参考值（同一 ckpt，seeds 0–5，旧播种协议）

| 任务 | 基线 C | 说明 |
|---|---|---|
| OpenDoor | **6/6** | 1013/1192/2214/548/772/2515 |
| SitSofa | **5/6** | s5 超时 |
| Boxing | **4/6** | s0/s1 超时 |
| VisionNavi | **3/6，摔 2** | s2、s3 摔 |
| PPBox | **2/6** | s1✓616、s4✓683 |
| DoubleDesk | **0/6** | 全超时（base 也 0/6） |
| 合计 | **20/36 (56%)** | |

新协议（`/reseed`）已观测：OpenDoor s0 ✓1079、s1 ✗超时；VisionNavi s0 ✗超时（未摔）。

---

## 7. 已知坑

| # | 现象 | 处理 |
|---|---|---|
| 1 | `c.log 里没有 '[server][lora16] loaded'` | 路径/哈希错；基线 meta 是 update=100 |
| 2 | 没有 `reseed` 日志 | 补丁缺失 ⇒ 回 §1 |
| 3 | 误用了 base 臂 | 本容器必须 `--arms c` + `C2_SERVERS=c`，否则跑的是官方 base |
| 4 | `FileNotFoundError: 'ss'` | 缺 `iproute2`；本仓库已有兜底 |
| 5 | 4090 24 GB 紧 | 每卡只放一个 driver，先看 `nvidia-smi` |
| 6 | 大量超时 | DoubleDesk/PPBox 属正常；先确认 `[server] infer #N` 在涨 |
| 7 | 别用 `c2_reward.py` 的 J10/J20 | `frame_count` 是局部计数器 |
| 8 | 磁盘 | 36 集预留 ≥25 GB |

---

## 8. 交付物

1. `runs/dual_actor_wm6/c_lora/c2/s7_12_container1/{card0,card1}/`（`jobs.jsonl` + 每集 `result.json` + `ticks/`）；
2. §5 的六行汇总表；
3. 异常集附 `episode.log` 尾部 20 行 + 对应 `reseed` 行。
