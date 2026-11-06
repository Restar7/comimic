#!/usr/bin/env python3
"""Sample full-clip start rows for B-line collection.

Unlike sample_start_row.py (which requires `length >= margin` and samples a
start that leaves `margin` frames), this helper uses the *whole* clip:

  prefix  : start at clip frame 0, run the full clip length
  uniform : start uniformly in the clip, run to the clip end

It prints one JSON object per episode with `row`, `length`, `start`,
`remaining`, `max_steps` and `clip`, so the collector can set a per-episode
MAX_STEPS instead of a global 600.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TASK_PREFIX = {
    "doubledesk": "HOI_double_desk",
    "ppbox": "HOI_pp_box",
    "boxing": "HSI_boxing",
    "opendoor": "HSI_open_door",
    "sitsofa": "HSI_sit_sofa",
    "visionnavi": "HSI_vision_navi",
    "grapcup": "HOI_grap_cup",
}


def canon(task: str) -> str:
    t = task.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "doubledesk": "doubledesk",
        "ppbox": "ppbox",
        "boxing": "boxing",
        "opendoor": "opendoor",
        "sitsofa": "sitsofa",
        "visionnavi": "visionnavi",
        "grapcup": "grapcup",
    }
    if t not in aliases:
        raise SystemExit(f"unknown task {task!r}; choose from {sorted(TASK_PREFIX)}")
    return aliases[t]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--task", default="Boxing")
    ap.add_argument("--num", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-mode", choices=["prefix", "uniform"], default="prefix")
    ap.add_argument("--cap", type=int, default=0, help="0 = no cap (run to clip end)")
    ap.add_argument("--index", default="", help="comma-separated clip indices (0-based within task)")
    args = ap.parse_args()

    d = torch.load(str(args.packed), map_location="cpu", weights_only=False)
    names = list(d["names"])
    starts = [int(x) for x in d["starts"]]
    lengths = [int(x) for x in d["lengths"]]
    pre = TASK_PREFIX[canon(args.task)]
    idx = [i for i, n in enumerate(names) if n.startswith(pre)]
    if not idx:
        raise SystemExit(f"no clips for {pre}")

    rng = random.Random(int(args.seed))
    if args.index.strip():
        picked = [idx[int(x)] for x in args.index.split(",") if x.strip()]
    else:
        pool = list(idx)
        rng.shuffle(pool)
        picked = pool[: max(1, int(args.num))]

    for i in picked:
        L = lengths[i]
        if args.start_mode == "prefix":
            f = 0
        else:
            f = rng.randrange(0, max(1, L))
        remaining = max(1, L - f)
        max_steps = remaining if int(args.cap) <= 0 else min(remaining, int(args.cap))
        print(json.dumps({
            "clip": names[i],
            "row": starts[i] + f,
            "length": L,
            "start": f,
            "remaining": remaining,
            "max_steps": max_steps,
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
