# SONIC 数据集：固定版本（2026-10-24 锁定）

> 结论：SONIC 微调数据**固定用 `groot_sonic_vla_ft318_frozen_20261023`**（2026-10-23 冻结、已剔除失败 episode）：
> **4 类任务 / 9 个 LeRobot 子数据集 / 318 episode / 473,876 帧 / 2.633 h @50fps**。
> 路径、条数、任务指令、文件指纹全部记录在 [`data/sonic_datasets.json`](../data/sonic_datasets.json)，
> 代码通过 `diff_vla_wbc.sonic_dataset` 读取，不再有任何脚本硬编码路径。

---

## 1. 锁定的 active 集

```text
active = ft318_frozen_20261023
root   = /inspire/hdd/global_user/liumengfan-253108110079/flowbody/FlowBody/dataset/groot_sonic_vla_ft318_frozen_20261023
manifest = <root>/manifest.json   (schema: flowbody.groot_sonic_vla_roots.v1)
```

| 任务 | episode | 子数据集 | 任务指令（manifest.json 原文） |
|---|---|---|---|
| `task_1_meeting_room_milk_basket` | **89**（5 + 84） | 2026-07-14-19-15-12 / 2026-07-14-20-30-01 | Navigate to the table ahead, grasp the milk carton, then turn around and proceed to the other table to place the carton into the basket. |
| `task_2_office_red_cup_drawer` | **70**（56 + 14） | 2026-07-22-19-33-29 / 2026-07-22-20-13-55 | Place the red cup into the drawer and close the drawer completely. |
| `task_3_kitchen_plate_milk_transfer` | **65** | 2026-07-23-19-14-37 | Retrieve the milk carton from the right plate and place it onto the left plate. |
| `task_4_diningroom_can_low_chair` | **94**（18 + 18 + 29 + 29） | 2026-07-27-20-38-22 / -20-38-30 / 2026-07-29-19-01-38 / -19-01-58 | Squat down to grasp the aluminum can on the low chair, then stand up and place it on the table ahead. |
| **合计** | **318** | **9** | 473,876 帧 = 2.633 h |

**为什么是这个版本**：

- 目录时间 2026-10-23 16:45，是 SONIC 四类任务集在**剔除失败 episode 之后的冻结版**；
- 它的 `manifest.json` 逐条记录了 `source_root`（对应 `SONIC_DATA/` 下的 2026-07-14、20260722、20260723、20260727、20260729）、
  episode/frames/fps 和 `task_config_sha256`，来源可追溯；
- 与同一次冻结的 `groot_sonic_vla_ft271_frozen_20261023`（271 episode，task_4 只保留 47）对照，
  **318 版是完整版**，因此作为 active；271 版只作为备用记录。

> 说明：之前先固定过的 `SONIC_DATA/.incoming/20260714_*`（134 条）是**未过滤原始导入**，不是训练口径，
> 现已从清单中移除，只保留在 git 历史里。

## 2. 数据内容（每个子数据集都是 LeRobot v2.1，50 fps，31 列）

| 列 | 维度 | 用途 |
|---|---|---|
| `action.motion_token` | `[64]` | SONIC FSQ motion token（浮点、步长 1/16，可导） |
| `action.wbc` | `[43]` | 29 身体 + 14 手部关节目标 |
| `teleop.left_hand_joints` | `[7]` | 左手关节角（rad） |
| `teleop.right_hand_joints` | `[7]` | 右手关节角（镜像符号，与 MJCF 一致） |
| `observation.state` | `[43]` | 本体状态（29 身体 + 14 手） |

## 3. 代码里怎么用

```python
from diff_vla_wbc.sonic_dataset import (
    active_datasets, dataset_entries, episode_files, task_counts, instructions, verify,
)

report = verify()            # 校验磁盘：条数/帧数/token/手部列
assert report["ok"], report["errors"]
print(report["episodes"], report["frames"], report["hours_at_50fps"])   # 318 473876 2.633
print(task_counts())          # {'task_1_...': 89, 'task_2_...': 70, ...}
print(instructions()["task_3_kitchen_plate_milk_transfer"])

for ds in active_datasets():
    for parquet in episode_files(ds):
        ...
```

- 清单：`comimic/data/sonic_datasets.json`（`active` 决定用哪套；`sets` 里含条目、任务指令、指纹）；
- 测试：`pytest tests/test_sonic_dataset.py -q` → **11 passed**（318 条、task1=89、指纹稳定、token/手部列存在）。

## 4. 注意 / 待办

1. **不要再用 `SONIC_DATA/` 下那几份原始导入**做训练：它们是过滤前的版本（如 134 / 229 / 363 条），只作溯源。
2. **`groot_repro/` 里还有 6 处硬编码 `/home/wusichen/SONIC_DATA/...`**（`repro_infer.py`、`replay_recorded*.py`、
   `render_*.py`、`gpu_bench.py`、`server_check.py`、`test_hold.py`），应改为从 `sonic_dataset` resolver 取路径。
3. **任务指令在 `manifest.json` 里**（不在 LeRobot `tasks.jsonl`，那里全是 `"demo"`），做语言条件时用 `instructions()`。
4. token 是 1/16 网格上的浮点值（非离散索引），可直接回归、可反传；若做端到端不要把中间量化成 index。
