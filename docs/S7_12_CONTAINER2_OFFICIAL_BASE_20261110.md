# 六任务闭环 · 容器 2：**官方 FT8000 base（不挂 LoRA）**（seeds 7–12）

> 生成于 2026-11-10
> 目标：**容器 2（2 × RTX 4090）** 跑**官方微调底模 `checkpoint-8000`（base 臂，不挂任何 LoRA）**的**全部六个任务**
> 配对关系：与**容器 1（基线 C 交付物 `lora300_ddp`）**使用**同一批 seeds 7–12、同一批任务、同一套补丁**，
> 两边结果合并即为**逐 (任务, seed) 配对**的 "C 微调 vs 官方 base" 对比。

---

## 0. 30 秒速览

| 事项 | 内容 |
|---|---|
| 评测对象 | **官方 FT8000 底模**：`comimic/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（**不挂 LoRA**） |
| 臂 | **`--arms base`**（与容器 1 的 `--arms c` 对应） |
| 本地 LoRA ckpt | **不需要**（base 臂不加载 adapter，也不需要 `--lora-ckpt`） |
| tick 里的策略标识 | 自动写 `base_ft8000`（便于与容器 1 区分） |
| seeds | **7 8 9 10 11 12**（预注册，与容器 1 完全相同） |
| 规模 | **6 任务 × 6 seeds = 36 集**，两卡并行，各约 **2.2 小时** |
| 前置 | 两处补丁必须在位，否则 **base-only 会 fail-closed 拒绝评测**（§1） |

---

## 1. 前置补丁（base 臂比 C 臂多一个硬性依赖）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
grep -c "EPISODE_SEED_BASE"   scripts/c_wm6/c2_driver.py          # 期望 3
grep -c 'path == "/reseed"'   scripts/ha_vla_latent64_server.py   # 期望 1
grep -c 'if "c" in SERVER_ARMS' scripts/c_wm6/c2_driver.py        # 期望 1  ← base-only 必需
```

三个数都对 ⇒ 可以跑。
第三个数为 0 ⇒ 你会在启动时看到：

```
[c2][fail-closed] c.log 里没有 '[server][lora16] loaded' ⇒ LoRA 没装上，拒绝评测
```

这是**只跑 base 臂**时的老 bug（保护检查无条件要求 c.log），本仓库已修成"仅当装了 c 臂才检查"。
把 `scripts/c_wm6/c2_driver.py`、`scripts/ha_vla_latent64_server.py` 两个文件拷过来即可。

**可复现性**：同一 (task, seed) 的采样噪声流由 `/reseed` 固定（`20261109 + task_id*1000 + seed`）；
但单集步数仍有 ±30–60% 硬件级抖动 ⇒ 只按多 seed 成功率下结论。

---

## 2. 一次性准备 + 冒烟

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u
source scripts/harness03_env.sh
export C2_SERVER_SEED=20261109

# 冒烟：base 臂 Boxing seed7（快）
C2_SERVERS=base C2_PORT_BASE=18080 C2_GPU_BASE=0 C2_ISAAC_GPU=0 \
python3 scripts/c_wm6/c2_driver.py --tasks Boxing --seeds 7 --arms base --max-steps 3000 \
  --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/s7_12_container2/smoke
```

通过标志：

1. **不能**出现 `[server][lora16] loaded`（出现说明挂上了不该挂的 adapter）；
2. server 日志有 `[server][cfg] ... seed=20261109` 以及 **`[server] reseed seed=2026xxxxx`**；
3. 出现 `共 N 个 (task,seed) × 1 臂 = N 集`，随后有 `rc=0` + `result.json`。

> 注意：base 臂**不传** `--lora-ckpt`；`--arms base` 与 `C2_SERVERS=base` 必须同时给。

---

## 3. 正式跑（与容器 1 完全相同的任务/seed 划分）

| 卡 | 任务（按此顺序） | 预计 |
|---|---|---|
| 0 | `DoubleDesk Boxing VisionNavi` | ≈130 min |
| 1 | `PPBox OpenDoor SitSofa` | ≈135 min |

**卡 0**

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh; export C2_SERVER_SEED=20261109
C2_SERVERS=base C2_PORT_BASE=18080 C2_GPU_BASE=0 C2_ISAAC_GPU=0 \
nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks DoubleDesk Boxing VisionNavi --seeds 7 8 9 10 11 12 --arms base \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/s7_12_container2/card0 \
  > runs/dual_actor_wm6/c_lora/c2/s7_12_container2/card0.log 2>&1 &
```

**卡 1**

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh; export C2_SERVER_SEED=20261109
C2_SERVERS=base C2_PORT_BASE=18081 C2_GPU_BASE=1 C2_ISAAC_GPU=1 \
nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks PPBox OpenDoor SitSofa --seeds 7 8 9 10 11 12 --arms base \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/s7_12_container2/card1 \
  > runs/dual_actor_wm6/c_lora/c2/s7_12_container2/card1.log 2>&1 &
```

---

## 4. 监控

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
grep -hE "rc=0" runs/dual_actor_wm6/c_lora/c2/s7_12_container2/*.log | tail -8
for f in runs/dual_actor_wm6/c_lora/c2/s7_12_container2/card*.log; do echo -n "$f "; grep -c "rc=0" "$f"; done
# base 臂的 server 日志在 servers/base.log
grep -h "reseed" runs/dual_actor_wm6/c_lora/c2/s7_12_container2/card*/servers/base.log | tail -6
grep -h "server\] infer #" runs/dual_actor_wm6/c_lora/c2/s7_12_container2/card*/servers/base.log | tail -3
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

---

## 5. 汇总（本容器）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
python3 scripts/c_wm6/agg_c2_episodes.py \
  --roots s7_12_container2/card0 s7_12_container2/card1 --arm base --fall all
```

上报：**逐任务成功数 / 摔倒次数 / 超时次数**（6 任务各一行）。

---

## 6. 参考值（官方 base 的历史数据，**非配对、不同 harness**，只作量级参考）

| 任务 | base 历史（本仓库 c2 驱动，n 很小） | 备注 |
|---|---|---|
| OpenDoor | 0/0（无本仓库数据） | B 线另一套 harness 记过 5/6 |
| SitSofa | **1/4** | s2 ✓1564 |
| Boxing | 0/0 | B 线记过 5/6（不同 harness） |
| VisionNavi | **1/2** | s1 ✓2758 |
| PPBox | **0/2** | 全超时 |
| DoubleDesk | 0/0 | B 线记过 0/6 |

⇒ 本容器的价值不在"和旧 base 数字比"，而在于**产出与容器 1 同 seeds、同协议、可逐 seed 配对**的 base 数据。

**判读要点**

- 看 **C 相对 base 的逐 seed 胜负**（配对），以及六任务的合计成功率差；
- base 臂出现成功率异常低时，先查 §2 的三条冒烟标志（尤其"没有 lora16 loaded"这一条）。

---

## 7. 已知坑

| # | 现象 | 处理 |
|---|---|---|
| 1 | `fail-closed c.log 里没有 lora16 loaded` | base-only 保护补丁缺失 ⇒ §1 第三个 grep |
| 2 | 误把 `--arms c` 用在本容器 | 那样跑的是基线 C，不是官方 base；本容器必须 `--arms base` |
| 3 | 传了 `--lora-ckpt` | base 臂不需要；传了也不会挂载，但容易误判，建议不传 |
| 4 | 没有 `reseed` 日志 | 补丁缺失 ⇒ §1 |
| 5 | `FileNotFoundError: 'ss'` | 缺 `iproute2`；本仓库已兜底 |
| 6 | 4090 24 GB 紧 | 每卡一个 driver |
| 7 | 别用 `c2_reward.py` 的 J10/J20 | `frame_count` 是局部计数器 |
| 8 | 磁盘 | 36 集预留 ≥25 GB |

---

## 8. 交付物

1. `runs/dual_actor_wm6/c_lora/c2/s7_12_container2/{card0,card1}/`（`jobs.jsonl` + 每集 `result.json` + `ticks/`）；
2. §5 的六行汇总表；
3. 异常集附 `episode.log` 尾部 20 行 + 对应 `reseed` 行 + `servers/base.log` 里该集的 `infer` 计数。
