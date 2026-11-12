# P21 × HumanoidArena 三容器交接（repeat_idx 0–17）

日期：2026-11-11
模型：p21（refresh#2 联合微调 best@150，update=150，dJ=+0.0021717）

本机（container 0）：**repeat_idx 0–5**，已在跑。
容器 A：**repeat_idx 6–11**。
容器 B：**repeat_idx 12–17**。

三个容器必须使用**同一份打过补丁的 driver + wrapper**，否则 repeat_idx 不生效（会退化成同一个 repeat 重复跑）。

---

## 1. 目标与口径

| 项 | 值 |
|---|---|
| seeds | **0 / 1 / 2**（官方 `EVAL_SEEDS="0 1 2"`） |
| repeat_idx | **0–17**，每容器 6 个（0–5 / 6–11 / 12–17） |
| 任务 | card0：OpenDoor + SitSofa；card1：Boxing + VisionNavi |
| 每容器 | 2 任务 × 3 seeds × 6 repeats = **36 集/卡**，**72 集/容器** |
| 三容器合计 | 3 seeds × 18 repeats × 4 任务 = **216 集**；**54 集/任务** |
| 官方 HumanoidArena | 3 seeds × 20 repeats = **60 集/任务**（7 任务、4 test modes + cross-GMT） |

> 本批是官方的**轻量版**：3×18（少 2 个 repeat）、4 个 HSI 任务（少 Football/DoubleDesk/P&PBox）、只跑 base mode + SONIC。
> 结果只能当趋势，不能直接当 leaderboard 分数。

---

## 2. 关键定义（先看懂再跑）

- `seed` = 评测 seed 组（0/1/2），**组内固定**；
- `repeat_idx` = 组内第几个 repeat（0–17 全局编号）；
- `episode_seed` 用官方公式派生：

```python
def derive_episode_seed(task_name: str, group_seed: int, repeat_idx: int) -> int:
    payload = f"{task_name}|{int(group_seed)}|{int(repeat_idx)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[-4:], "big", signed=False) & 0x7FFFFFFF
```

- `task_name` 用**完整 Isaac task id**（如 `Isaac-Move-Open-Door-G129-Dex3-Wholebody`）；
- driver 在每集启动前 `POST /reseed {"seed": episode_seed}`（policy 采样种子）；
- wrapper 把同一个 `episode_seed` 通过 `--episode_seed` 传给 `sim_eval_vla.py`（场景随机化）；
- **两者必须相同**。sim 的 `/reset` 不 reseed（自定义 server 的设计），driver 的 `/reseed` 是权威。

`persist_sim=0`：本批**每集一个 Isaac 进程**；官方默认 `persist_sim=1`（一个 `(model, seed)` 组一个进程）本批未实现。

---

## 3. 必须打的补丁（三容器完全一致）

### 3.1 `comimic/scripts/ha_isaac_closedloop.sh`（driver 实际执行的是这个）

这是 `c2_driver.py` 里 `MAIN / "scripts/ha_isaac_closedloop.sh"` 指向的文件；**只改 `comimic-c-wm6/scripts/` 下的副本没用**。

改动 1：在 `REPEAT=${REPEAT:-0}` 后加一行：

```bash
REPEAT=${REPEAT:-0}
EPISODE_SEED=${EPISODE_SEED:-$SEED}
```

改动 2：`sim_eval_vla.py` 的启动参数加 `--episode_seed`：

```bash
  --seed "$SEED" --repeat_idx "$REPEAT" --episode_seed "$EPISODE_SEED" \
```

（同时把 `comimic-c-wm6/scripts/ha_isaac_closedloop.sh` 保持同步，避免其他工具用到旧副本。）

### 3.2 `comimic-c-wm6/scripts/c_wm6/c2_driver.py`

必须包含以下改动（可直接从本机复制该文件）：

1. `import hashlib`；
2. `TASK_ID_MAP`（完整 Isaac task id）；
3. `derive_episode_seed()`（上面的官方公式）；
4. `run_episode(..., repeat, ...)`：
   - `rec` 增加 `repeat`；
   - `_ep_seed = derive_episode_seed(TASK_ID_MAP[ha_task], seed, repeat)`；
   - `post_reseed(..., _ep_seed)`；
   - env 增加 `"REPEAT": str(repeat)`、`"EPISODE_SEED": str(_ep_seed)`；
   - `run_dir` 带 `r{repeat}`；
   - `WM6_TICK_EPISODE` 带 `r{repeat}`，新增 `WM6_TICK_EPISODE_SEED`；
5. `worker` 解包 `(task, seed, repeat)`；
6. `main` 新增 `--repeat-start` / `--repeats`，`jobs` 变三元组，`summary.json` 的 rows 带 `repeat`。

### 3.3 自检（必须通过，否则不要开跑）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
python3 - <<'PY'
import sys; sys.path.insert(0, 'scripts/c_wm6')
from c2_driver import derive_episode_seed, TASK_ID_MAP
for t in ['OpenDoor', 'SitSofa', 'Boxing', 'VisionNavi']:
    print(t, [derive_episode_seed(TASK_ID_MAP[t], 0, r) for r in range(6)])
PY
```

seed=0、repeat 0–5 的期望值：

```text
OpenDoor   [1165506222, 1366698720, 1318335773, 384699587, 50686553, 394936585]
SitSofa    [1737204259, 572408881, 394648599, 1496534366, 860295903, 1890201208]
Boxing     [1049058202, 12283769, 2035381933, 587028425, 262552480, 2090369674]
VisionNavi [1305788479, 1490334495, 914829829, 1112945881, 34406482, 1432329048]
```

如果算出来不同，说明 driver 没打补丁。

---

## 4. 启动后 30 秒硬校验

第一集（r0）启动后：

```bash
# 1) sim 的 startup 行
grep "startup seed=" card0/c/OpenDoor_seed0/try1_r0_*/episode.log
# 期望：startup seed=0 repeat_idx=0 episode_seed=1165506222

# 2) sim 命令行
ps -eo cmd | grep -o -- "--seed [0-9]* --repeat_idx [0-9]* --episode_seed [0-9]*" | head
# 期望：--seed 0 --repeat_idx 0 --episode_seed 1165506222

# 3) server reseed 日志
grep reseed card0/servers/c.log | tail -1
# 期望：[server] reseed seed=1165506222
```

**三处必须是同一个数字**。如果是 `episode_seed=0`，说明 `comimic/scripts/ha_isaac_closedloop.sh` 没打补丁，立刻停，改完再跑。

再确认队列：

```bash
grep "共 .* 集" card0.log card1.log
# 期望：共 36 个 (task,seed,repeat) × 1 臂 = 36 集
```

---

## 5. 容器 A：repeat_idx 6–11

### 5.1 环境

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
export WT=$PWD
source scripts/harness03_env.sh
export C2_SERVER_SEED=20261109
# Vulkan ICD（曾出现 0 字节导致 PhysX 起不来）
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
CKPT=$PWD/runs/dual_actor_wm6/c_lora/p21/ft300/checkpoint_lora_best.pt
OUT=$PWD/runs/dual_actor_wm6/c_lora/c2/p21_seed012_r6_11
mkdir -p "$OUT"
```

> 也可以直接用本仓已备好的脚本：`runs/dual_actor_wm6/c_lora/run_p21_seed012_r6_11.sh`（容器 A）和 `run_p21_seed012_r12_17.sh`（容器 B）。脚本里的 `ROOT`/`CKPT` 按容器实际路径改即可。

### 5.2 启动 card0（OpenDoor + SitSofa，GPU0，端口 18080）

```bash
C2_SERVERS=c C2_PORT_C=18080 C2_GPU_C=0 C2_ISAAC_GPU=0 \
  nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks OpenDoor SitSofa --seeds 0 1 2 --repeat-start 6 --repeats 6 \
  --arms c --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out "$OUT/card0" --lora-ckpt "$CKPT" --lora-tag p21 \
  > "$OUT/card0.log" 2>&1 &
```

### 5.3 启动 card1（Boxing + VisionNavi，GPU1，端口 18081）

```bash
C2_SERVERS=c C2_PORT_C=18081 C2_GPU_C=1 C2_ISAAC_GPU=1 \
  nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks Boxing VisionNavi --seeds 0 1 2 --repeat-start 6 --repeats 6 \
  --arms c --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out "$OUT/card1" --lora-ckpt "$CKPT" --lora-tag p21 \
  > "$OUT/card1.log" 2>&1 &
```

预计：36 集/卡，约 3–6 小时（每集 ~5–10 min）。

---

## 6. 容器 B：repeat_idx 12–17

与容器 A 完全相同，只改两处：

```bash
OUT=$PWD/runs/dual_actor_wm6/c_lora/c2/p21_seed012_r12_17
# 两个 driver 命令里：
--repeat-start 12 --repeats 6
```

其余（seeds、tasks、ports、GPU、ckpt、wrapper 补丁）完全一样。

---

## 7. 聚合口径（三容器跑完后）

每个容器的 `summary.json` 里 `rows` 字段包含：

```text
arm, task, ha_task, seed, repeat, success, failure_reason,
episode_steps, n_ticks, frame_step_set, duration_sec, rc, reseed
```

合并三容器后：

- 每个任务：3 seeds × 18 repeats = **54 集**；
- 每个 seed：18 repeats；
- 报告：每任务 SR（mean ± std **across seeds**）、每 seed SR；
- 摔倒率 AFR 如需要，用 `failure_reason == "fall"` 统计；
- **不要**挑窗口/挑 seed；官方是 3×20，本批 3×18，差 2 个 repeat。

与官方差距（写报告时必须注明）：

| 项 | 官方 | 本批 |
|---|---|---|
| seeds | 3 | 3 |
| repeats/seed | 20 | **18** |
| 集/任务 | 60 | **54** |
| 任务 | 7（3 HOI + 4 HSI） | 4（全 HSI） |
| 评测轴 | base/semantic/vision/execution + cross-GMT | **只 base** |
| GMT | TWIST2 + SONIC | **只 SONIC** |
| Isaac 进程 | `persist_sim=1`（组内复用） | **`persist_sim=0`（每集重启）** |

---

## 8. 常见坑

1. **只改了 `comimic-c-wm6/scripts/ha_isaac_closedloop.sh`**：driver 实际执行的是 `comimic/scripts/...`，必须改主仓那份。
2. **没传 `--episode_seed`**：sim 的 `episode_seed` 会退化成 `--seed`（0/1/2），repeat_idx 只改文件名，不改随机化。
3. **用老的重复 seed 写法**（`--seeds 0 0 0 0 0 ...`）：driver 现在应使用 `--seeds 0 1 2 --repeat-start X --repeats 6`；老写法只会得到同一个 repeat 的多次拷贝。
4. **server 端口冲突**：card0 用 18080，card1 用 18081；两个容器不要共用一台机器的端口。
5. **Vulkan ICD 0 字节**：`ls -l /etc/vulkan/icd.d/nvidia_icd.json`，0 字节就从 `/usr/share/vulkan/icd.d/` 拷一份。
6. **旧僵尸进程**：跑之前 `ps -eo pid,cmd | grep -E "sim_eval_vla|ha_vla_latent64_server"`，确认没有残留；否则 GPU 显存会被吃。

---

## 9. 本机当前状态（container 0）

- 运行中：`c2/p21_seed012_r0_5/`，repeat 0–5，36 集/卡；
- 补丁文件：
  - `comimic-c-wm6/scripts/c_wm6/c2_driver.py`
  - `comimic/scripts/ha_isaac_closedloop.sh`（主仓，driver 实际执行）
  - `comimic-c-wm6/scripts/ha_isaac_closedloop.sh`（同步副本）
- 自检通过：OpenDoor/seed0/r0 的 sim `episode_seed=1165506222` 与 server `reseed` 一致；Boxing/seed0 的 r0=1049058202、r1=12283769，repeat_idx 确实在变。
- 第一集结果：Boxing seed0 r0 成功（334 步）。
- 已备好 A/B 脚本：`runs/dual_actor_wm6/c_lora/run_p21_seed012_r6_11.sh`、`run_p21_seed012_r12_17.sh`。
