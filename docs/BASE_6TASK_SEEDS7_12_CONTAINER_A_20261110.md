# 基线六任务闭环 · 容器 A 交接（seeds 7–12）

> 生成于 2026-11-10 ｜ 目标：**容器 A（2 × RTX 4090）**
> 任务：`DoubleDesk`、`PPBox`（本容器负责这两个任务的全部 6 个 seed）
> 评测对象：**基线交付物 `lora300_ddp`**（不是 p14/p16/p17/p18）
> 目的：把基线的 seed 池从 0–5 扩到 **7–12**，与其他容器/主机的数据合并后用于统计对照

---

## 0. 30 秒速览

| 事项 | 内容 |
|---|---|
| 适配器 | `comimic-c-wm6/runs/dual_actor_wm6/c_lora/critic10/lora300_ddp/checkpoint_lora_best.pt` |
| sha256 前 16 | **`6942eab9a72f3d39`** |
| 期望 meta | `update=100`、`dJ_val≈0.0045587`、`physical.lim=0.03067937671760282`、440 tensors |
| 配方来历 | λ_lim=0.030679、无 margin、无动态加权、H=10 / WM6（`8edb59cf`）、300 步训练的 best（it=100） |
| seeds | **7 8 9 10 11 12**（预注册，跑完全报，不许挑） |
| 本容器规模 | 2 任务 × 6 seeds = **12 集** |
| 耗时 | 卡0 DoubleDesk ≈55 min；卡1 PPBox ≈60 min（并行） |
| **前置（关键）** | 两处**评测协议补丁**必须已存在，否则 seed 与其他容器不可对齐（见 §1） |

---

## 1. 协议补丁（不做这一步，结果不可对齐）

评测的可比性依赖两个文件里的 `/reseed` 机制：server 每个 episode 用 `seed = 20261109 + task_id*1000 + seed` 重新播种，
使同一 (任务, seed) 的噪声流与"这是第几集、前面跑过什么"无关。

**在两台机器上分别确认补丁在位**：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
grep -c "EPISODE_SEED_BASE"            scripts/c_wm6/c2_driver.py            # 期望 3
grep -c 'path == "/reseed"'            scripts/ha_vla_latent64_server.py     # 期望 1
```

- 两个数都对 ⇒ 补丁在位，直接往下；
- 任一为 0 ⇒ 从本机拷贝这两个文件过去（它们都是**可选开关**，不影响别的路径）：
  - `scripts/c_wm6/c2_driver.py`
  - `scripts/ha_vla_latent64_server.py`

**可复现性的准确说法（别误解）**：

- ✅ **同一 (任务, seed) 的采样噪声流现在是确定的**，不随队列位置/前面跑过几集而变；
- ⚠️ **单集步数仍有 ±30–60% 抖动**（CUDA kernel 选择、Isaac 接触求解等硬件级非确定性），成功率在小样本下偶尔翻面；
- ⇒ 所以结论只能建立在**多 seed 的成功率**上，单集步数不能当判据。

---

## 2. 一次性准备

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u
source scripts/harness03_env.sh
export C2_SERVER_SEED=20261109      # 会被 /reseed 按 (task,seed) 覆盖，保留即可
```

**冒烟测试（必做，约 5 分钟）**——确认基线适配器被正确加载：

```bash
C2_SERVERS=c C2_PORT_C=18080 C2_GPU_C=0 C2_ISAAC_GPU=0 \
python3 scripts/c_wm6/c2_driver.py --tasks DoubleDesk --seeds 7 --arms c --max-steps 3000 \
  --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/base_s7_12/smoke \
  --lora-ckpt runs/dual_actor_wm6/c_lora/critic10/lora300_ddp/checkpoint_lora_best.pt \
  --lora-tag base
```

通过标志：

1. 日志有 **`[server][lora16] loaded ... meta={'update': 100, 'dJ_val': 0.0045587...}`**；
2. server 日志出现 **`[server] reseed seed=2026xxxxx`**（说明每集独立播种生效）；
3. 有 `rc=0` 且 `result.json` 落盘。

> ⚠️ 本机型（4090 24 GB）每卡需要 VLA server ≈13.4 GB + Isaac ≈4–8 GB ≈ 18–21 GB，**close to 上限**；
> 开跑前 `nvidia-smi` 确认卡是空的，别共卡。

---

## 3. 正式跑（两卡并行）

**卡 0：DoubleDesk（6 集，多为 3000 步超时，约 55 min）**

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh; export C2_SERVER_SEED=20261109
C2_SERVERS=c C2_PORT_C=18080 C2_GPU_C=0 C2_ISAAC_GPU=0 \
nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks DoubleDesk --seeds 7 8 9 10 11 12 --arms c \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/base_s7_12/containerA_doubledesk \
  --lora-ckpt runs/dual_actor_wm6/c_lora/critic10/lora300_ddp/checkpoint_lora_best.pt \
  --lora-tag base > runs/dual_actor_wm6/c_lora/c2/base_s7_12/containerA_dd.log 2>&1 &
```

**卡 1：PPBox（6 集，多为超时，约 60 min）**

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh; export C2_SERVER_SEED=20261109
C2_SERVERS=c C2_PORT_C=18081 C2_GPU_C=1 C2_ISAAC_GPU=1 \
nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks PPBox --seeds 7 8 9 10 11 12 --arms c \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/base_s7_12/containerA_ppbox \
  --lora-ckpt runs/dual_actor_wm6/c_lora/critic10/lora300_ddp/checkpoint_lora_best.pt \
  --lora-tag base > runs/dual_actor_wm6/c_lora/c2/base_s7_12/containerA_ppbox.log 2>&1 &
```

---

## 4. 监控

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
grep -hE "rc=0" runs/dual_actor_wm6/c_lora/c2/base_s7_12/*.log | tail -8
for f in runs/dual_actor_wm6/c_lora/c2/base_s7_12/containerA_*.log; do echo -n "$f "; grep -c "rc=0" "$f"; done
# 每集都应看到一次 reseed
grep -h "reseed" runs/dual_actor_wm6/c_lora/c2/base_s7_12/containerA_*/servers/c.log | tail -6
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

---

## 5. 汇总与上报

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
python3 scripts/c_wm6/agg_c2_episodes.py \
  --roots base_s7_12/containerA_doubledesk base_s7_12/containerA_ppbox \
  --arm c --fall all
```

上报格式（只要这三类，**不要报单集步数当判据**）：

| 任务 | seeds 7–12 成功数 | 摔倒次数 | 超时次数 |
|---|---|---|---|
| DoubleDesk | x/6 | | |
| PPBox | x/6 | | |

---

## 6. 基线参考值（同一 ckpt，供对齐 sanity check）

| 任务 | seeds 0–5 历史（旧播种协议） | 新协议（`/reseed`）已知 |
|---|---|---|
| DoubleDesk | **0/6**（全超时；base 也 0/6，属任务级难题） | — |
| PPBox | **2/6**（s1 ✓616、s4 ✓683；s0 另有一次 sim_err） | — |
| OpenDoor（本容器不跑） | 6/6 | s0 ✓1079 |
| VisionNavi（本容器不跑） | 3/6，摔 2 | s0 ✗超时 |
| SitSofa（本容器不跑） | 5/6 | — |
| Boxing（本容器不跑） | 4/6 | — |

**判读**：DoubleDesk 若出现成功、或 PPBox 明显高于 2/6，说明新 seed 段更难/更易；
baseline 在 seeds 7–12 上的数将与本机 seeds 0–5 合并成 12 集的估计（CI 更窄）。

---

## 7. 已知坑

| # | 现象 | 处理 |
|---|---|---|
| 1 | 启动报 `c.log 里没有 '[server][lora16] loaded'` | 适配器路径错；核对 §1 的 sha。基线 meta 是 **update=100** |
| 2 | 没有任何 `reseed` 日志 | 补丁没打 / 用的不是本 worktree 的 driver ⇒ 回 §1 |
| 3 | `FileNotFoundError: 'ss'` | 缺 `iproute2`；本仓库 driver 已兜底（老版本需打补丁或安装） |
| 4 | 端口冲突 | 换端口（18080/18081 → 18090/18091）或清理残留 `c2_driver` |
| 5 | 4090 24 GB 紧 | 每卡只放一个 driver；先 `nvidia-smi` |
| 6 | 大量 3000 步超时 | DoubleDesk/PPBox 的正常现象（基线也是） |
| 7 | 别用 `c2_reward.py` 的 J10/J20 | `frame_count` 是局部计数器 |
| 8 | 磁盘 | 每集 ~190 个 `ticks_*.pt`；12 集预留 ≥8 GB |

---

## 8. 交付物

1. `runs/dual_actor_wm6/c_lora/c2/base_s7_12/containerA_{doubledesk,ppbox}/`（含 `jobs.jsonl`、每集 `result.json`、`ticks/`）；
2. §5 的两行汇总表（成功数 / 摔倒 / 超时）；
3. 若出现异常集，附 `episode.log` 尾部 20 行 + 该集的 `reseed` 行。
