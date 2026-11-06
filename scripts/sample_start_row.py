#!/usr/bin/env python3
"""给某个任务随机挑一个**起始库行号**（= SuperMimic 的"随机 clip + 随机帧"）。

为什么只需要这一处：
    HA 的任务 env **没有运动参考**：`commands = None`、`terminations` 为空、
    `rewards` 是物体任务奖励（hammer → basket）；摔倒检测是 `sim_eval_vla.py` 里的 tilt。
    所以"随机起始帧"不需要改 env —— 参考只在我们的 provider 里（token + `frames`）。

输出：一行整数（库行号），直接给 `SM_START_FRAME` / `SM_TOKEN_START`。

用法：
    python3 scripts/sample_start_row.py --task DoubleDesk --seed 0 --margin 360
    python3 scripts/sample_start_row.py --list
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

#: 任务别名 → 库里的 clip 名前缀（`build_ha_command_lib.py` 的命名）
TASK_PREFIX = {
    "doubledesk": "HOI_double_desk",
    "ppbox": "HOI_pp_box",
    "boxing": "HSI_boxing",
    "opendoor": "HSI_open_door",
    "sitsofa": "HSI_sit_sofa",
    "visionnavi": "HSI_vision_navi",
    "grapcup": "HOI_grap_cup",   # 环境里没注册，仅列出
}
ALIASES = {
    "double_desk": "doubledesk", "hoi_double_desk": "doubledesk",
    "pp_box": "ppbox", "hoi_pp_box": "ppbox",
    "hsi_boxing": "boxing",
    "hsi_open_door": "opendoor",
    "hsi_sit_sofa": "sitsofa",
    "hsi_vision_navi": "visionnavi",
    "hoi_grap_cup": "grapcup",
}


def canon(task: str) -> str:
    t = task.strip().lower().replace("-", "")
    t = ALIASES.get(t, t)
    if t not in TASK_PREFIX:
        raise SystemExit(f"[start-row] 未知任务 {task!r}；可选：{sorted(TASK_PREFIX)}")
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--task", default="DoubleDesk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=int, default=360,
                    help="该行之后至少还要有多少帧（≈ MAX_STEPS + token 窗口）")
    ap.add_argument("--list", action="store_true", help="列出任务与可用 clip 数")
    a = ap.parse_args()

    d = torch.load(str(a.packed), map_location="cpu", weights_only=False)
    names, starts, lens = d["names"], [int(x) for x in d["starts"]], [int(x) for x in d["lengths"]]

    if a.list:
        for t, pre in TASK_PREFIX.items():
            idx = [i for i, n in enumerate(names) if n.startswith(pre)]
            ok = [i for i in idx if lens[i] >= a.margin]
            print(f"{t:12s} prefix={pre:18s} clips={len(idx):3d} clips>={a.margin}f={len(ok):3d}")
        return 0

    t = canon(a.task)
    pre = TASK_PREFIX[t]
    idx = [i for i, n in enumerate(names) if n.startswith(pre)]
    if not idx:
        raise SystemExit(f"[start-row] 库里没有 {pre} 的 clip")
    ok = [i for i in idx if lens[i] >= a.margin]
    if not ok:
        # 没有够长的：退回最长的那条，并把起点压到末尾之前
        i = max(idx, key=lambda k: lens[k])
        f = max(0, lens[i] - a.margin)
    else:
        rng = random.Random(int(a.seed))
        i = rng.choice(ok)
        f = rng.randrange(0, max(1, lens[i] - a.margin))
    row = starts[i] + f
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
