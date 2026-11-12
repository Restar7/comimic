# 基线六任务闭环 · 容器 B 交接（seeds 7–12）

> 生成于 2026-11-10 ｜ 目标：**容器 B（2 × RTX 4090）**
> 任务：`VisionNavi`、`Boxing`（卡0）与 `OpenDoor`、`SitSofa`（卡1）
> 评测对象：**基线交付物 `lora300_ddp`**（不是 p14/p16/p17/p18）
> 目的：与容器 A 一起把基线 seed 池扩到 **7–12**，与主机 seeds 0–5 合并统计

---

## 0. 30 秒速览

| 事项 | 内容 |
|---|---|
| 适配器 | `comimic-c-wm6/runs/dual_actor_wm6/c_lora/critic10/lora300_ddp/checkpoint_lora_best.pt` |
| sha256 前 16 | **`6942eab9a72f3d39`** |
| 期望 meta | `update=100`、`dJ_val≈0.0045587`、`physical.lim=0.03067937671760282` |
| 配方来历 | λ_lim=0.030679、无 margin、无动态加权、H=10 / WM6（`8edb59cf`）、300 步的 best（it=100） |
| seeds | **7 8 9 10 11 12**（预注册，跑完全报） |
| 本容器规模 | 4 任务 × 6 seeds = **24 集**（卡0 两任务、卡1 两任务） |
| 耗时 | 卡0 VisionNavi+Boxing ≈75 min；卡1 OpenDoor+SitSofa ≈75 min（并行） |
| 前置（关键） | 两处**评测协议补丁**必须在位（见 §1） |

---

## 1. 协议补丁（不做这步，结果与其他容器不可对齐）

server 每个 episode 由 driver 调 `/reseed`，用 `seed = 20261109 + task_id*1000 + seed` 重新播种，
使同一 (任务, seed) 的噪声流与"排在第几集、前面跑过什么"无关。

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
grep -c "EPISODE_SEED_BASE"  scripts/c_wm6/c2_driver.py          # 期望 3
grep -c 'path == "/reseed"'  scripts/ha_vla_latent64_server.py   # 期望 1
```

任一为 0 ⇒ 从主机拷这两个文件过去：
`scripts/c_wm6/c2_driver.py`、`scripts/ha_vla_latent64_server.py`。

**可复现性的准确说法**：

- ✅ 同一 (任务, seed) 的**采样噪声流**已确定，与队列位置无关；
- ⚠️ 单集步数仍有 **±30–60%** 抖动（CUDA/Isaac 硬件级非确定性），小样本下成功率偶有翻面；
- ⇒ 只按**多 seed 成功率**下结论。

---

## 2. 一次性准备

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u
source scripts/harness03_env.sh
export C2_SERVER_SEED=20261109      # 会被 /reseed 覆盖，保留即可
```

**冒烟测试（必做，约 5 分钟）**

```bash
C2_SERVERS=c C2_PORT_C=18080 C2_GPU_C=0 C2_ISAAC_GPU=0 \
python3 scripts/c_wm6/c2_driver.py --tasks Boxing --seeds 7 --arms c --max-steps 3000 \
  --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/base_s7_12/smoke \
  --lora-ckpt runs/dual_actor_wm6/c_lora/critic10/lora300_ddp/checkpoint_lora_best.pt \
  --lora-tag base
```

通过标志：① `[server][lora16] loaded ... meta={'update': 100, ...}`；② server 日志有 `[server] reseed seed=...`；③ 有 `rc=0` + `result.json`。

> 4090 24 GB：每卡 server ≈13.4 GB + Isaac ≈4–8 GB，**很紧**，开跑前确认卡空。

---

## 3. 正式跑（两卡并行，各两个任务）

**卡 0：VisionNavi → Boxing（顺序执行，共 12 集，约 75 min）**

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh; export C2_SERVER_SEED=20261109
C2_SERVERS=c C2_PORT_C=18080 C2_GPU_C=0 C2_ISAAC_GPU=0 \
nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks VisionNavi Boxing --seeds 7 8 9 10 11 12 --arms c \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/base_s7_12/containerB_vn_bx \
  --lora-ckpt runs/dual_actor_wm6/c_lora/critic10/lora300_ddp/checkpoint_lora_best.pt \
  --lora-tag base > runs/dual_actor_wm6/c_lora/c2/base_s7_12/containerB_vn_bx.log 2>&1 &
```

**卡 1：OpenDoor → SitSofa（顺序执行，共 12 集，约 75 min）**

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh; export C2_SERVER_SEED=20261109
C2_SERVERS=c C2_PORT_C=18081 C2_GPU_C=1 C2_ISAAC_GPU=1 \
nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks OpenDoor SitSofa --seeds 7 8 9 10 11 12 --arms c \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/base_s7_12/containerB_od_ss \
  --lora-ckpt runs/dual_actor_wm6/c_lora/critic10/lora300_ddp/checkpoint_lora_best.pt \
  --lora-tag base > runs/dual_actor_wm6/c_lora/c2/base_s7_12/containerB_od_ss.log 2>&1 &
```

---

## 4. 监控

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
grep -hE "rc=0" runs/dual_actor_wm6/c_lora/c2/base_s7_12/*.log | tail -8
for f in runs/dual_actor_wm6/c_lora/c2/base_s7_12/containerB_*.log; do echo -n "$f "; grep -c "rc=0" "$f"; done
grep -h "reseed" runs/dual_actor_wm6/c_lora/c2/base_s7_12/containerB_*/servers/c.log | tail -6
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

---

## 5. 汇总与上报

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
python3 scripts/c_wm6/agg_c2_episodes.py \
  --roots base_s7_12/containerB_vn_bx base_s7_12/containerB_od_ss \
  --arm c --fall all
```

上报格式：

| 任务 | seeds 7–12 成功数 | 摔倒次数 | 超时次数 |
|---|---|---|---|
| VisionNavi | x/6 | | |
| Boxing | x/6 | | |
| OpenDoor | x/6 | | |
| SitSofa | x/6 | | |

---

## 6. 基线参考值（同一 ckpt）

| 任务 | seeds 0–5 历史（旧播种协议） | 新协议（`/reseed`）已知 |
|---|---|---|
| VisionNavi | **3/6，摔 2**（✓s1 675 / s4 1648 / s5 923；s2、s3 摔；s0 超时） | s0 ✗超时（未摔） |
| Boxing | **4/6**（✓s2 169 / s3 594 / s4 191 / s5 418；s0、s1 超时） | — |
| OpenDoor | **6/6**（1013/1192/2214/548/772/2515） | **s0 ✓1079、s1 ✗超时**（与历史一致：s0 快、s1 曾成功过） |
| SitSofa | **5/6**（✓s0 365 / s1 811 / s2 452 / s3 485 / s4 2416；s5 超时） | — |
| DoubleDesk / PPBox（容器 A） | 0/6、2/6 | — |

**重点观察项**

1. **VisionNavi 的 s? 是否还摔**——若 seeds 7–12 上出现摔倒，说明"摔"是基线固有的低频事件，不能拿 p 系列的"0 摔"当结论；
2. **OpenDoor 是否仍接近满堂红**——它是基线最强的任务（历史 6/6），若这里掉到 ≤4/6，说明新 seed 段更难。

---

## 7. 已知坑

| # | 现象 | 处理 |
|---|---|---|
| 1 | 报 `c.log 里没有 '[server][lora16] loaded'` | 路径/哈希错；基线 meta 是 **update=100** |
| 2 | 没有 `reseed` 日志 | 补丁缺失 ⇒ 回 §1 |
| 3 | `FileNotFoundError: 'ss'` | 缺 `iproute2`；仓库 driver 已兜底 |
| 4 | 端口冲突 | 换 18090/18091 或清理残留 |
| 5 | 4090 24 GB 紧 | 每卡一个 driver |
| 6 | 大量超时 | VN/PPBox 常态；先确认 `reseed` 与 `[server] infer` 都有输出 |
| 7 | 别用 `c2_reward.py` 的 J10/J20 | `frame_count` 是局部计数器 |
| 8 | 磁盘 | 24 集预留 ≥15 GB |

---

## 8. 交付物

1. `runs/dual_actor_wm6/c_lora/c2/base_s7_12/containerB_{vn_bx,od_ss}/`（`jobs.jsonl` + 每集 `result.json` + `ticks/`）；
2. §5 的四行汇总表；
3. 异常集附 `episode.log` 尾部 20 行 + 对应 `reseed` 行。
