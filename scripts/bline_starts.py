#!/usr/bin/env python3
"""Pick B-line replay start rows that actually belong to the task being collected.

Why this exists (2026-10-29): ``--start-frame`` in ``closed_loop_vla_supermimic.py`` is a
**global library row**, while ``--task`` only selects the scene XML + language prompt.  The
first version of the round loop handed *every* task the same ``BLINE_STARTS`` list, but the
compact library is concatenated per task, so most of those rows fall inside ``task_1``:

    task_1  [     0, 183567)  89 clips
    task_2  [183567, 275453)  70 clips
    task_3  [275453, 336100)  65 clips
    task_4  [336100, 473876)  94 clips

With ``BLINE_STARTS="100 2000 9000 … 240000"`` and 20 episodes/task, ``task_3`` and ``task_4``
had **zero** in-range starts: every episode was initialised from a ``task_1`` motion, i.e. the
robot spawned mid-dance in a scene where the objects were somewhere else.  Those episodes'
rewards and ``self`` tokens were noise (and they can fall over immediately).

Usage:
    python3 scripts/bline_starts.py --library data/library/ft318_compact.pt \\
        --task task3 --episodes 20 --ticks 600
    # -> 20 rows, evenly spread inside task_3's own range, none within `ticks` of a boundary
"""
from __future__ import annotations

import argparse
import os
import sys

import torch


def task_prefix(task: str) -> str:
    """``task3`` -> ``"task_3"`` (the clip-name prefix used by the 318 dataset)."""
    digits = "".join(ch for ch in task if ch.isdigit())
    return f"task_{digits}"


def task_range(pack: dict, task: str) -> tuple[int, int, int]:
    """Return ``(lo, hi, clips)`` = the library row range owned by ``task``."""
    prefix = task_prefix(task) + "_"
    names = pack["names"]
    starts = pack["starts"]
    lengths = pack["lengths"]
    picked = [i for i, name in enumerate(names) if str(name).startswith(prefix)]
    if not picked:
        alts = sorted({str(n).split("__")[0].rsplit("_", 1)[0] for n in names})
        raise SystemExit(f"[fail] 库里没有 {prefix}* 的 clip；有这些前缀: {alts}")
    lo = min(int(starts[i]) for i in picked)
    hi = max(int(starts[i]) + int(lengths[i]) for i in picked)
    return lo, hi, len(picked)


def even_starts(lo: int, hi: int, episodes: int, ticks: int, seed: int = 0) -> list[int]:
    """分层取起点：把可用区间切成 ``episodes`` 段，每段取一点。

    ``seed == 0`` → 取段中点（旧行为，跨轮完全可比）；
    ``seed != 0`` → 段内随机（**每轮不同**，6 轮累计覆盖 6×20 个不同起点，泛化面更宽）。
    """
    """``episodes`` rows spread over ``[lo, hi - ticks)`` so no episode runs off the task."""
    head = 50                      # 跳过 clip 最开头几十帧（落地/初始化抖动）
    usable_lo = lo + head
    usable_hi = hi - ticks - 1     # 一集 ticks 个控制步，末帧必须还在本任务内
    if usable_hi <= usable_lo:
        raise SystemExit(f"[fail] 任务区间 [{lo},{hi}) 只有 {hi - lo} 帧，放不下 {ticks} tick 的一集")
    span = usable_hi - usable_lo
    if seed == 0:
        return [usable_lo + int((i + 0.5) * span / episodes) for i in range(episodes)]
    import random

    rng = random.Random(seed)
    out = []
    for i in range(episodes):
        lo_i = usable_lo + int(i * span / episodes)
        hi_i = usable_lo + int((i + 1) * span / episodes)
        out.append(rng.randint(lo_i, max(lo_i, hi_i - 1)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--task", required=True, help="task1 / task2 / task3 / task4")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--seed", type=int, default=int(os.environ.get("BLINE_SEED", "0") or 0),
                    help="0=段中点（跨轮可比）；非 0=段内随机（每轮换一批起点）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    pack = torch.load(args.library, map_location="cpu", weights_only=False)
    lo, hi, clips = task_range(pack, args.task)
    starts = even_starts(lo, hi, args.episodes, args.ticks, seed=args.seed)
    if not args.quiet:
        print(
            f"[bline-starts] {args.task}: seed={args.seed} {clips} clips, rows [{lo},{hi}) "
            f"({hi - lo} 帧 = {(hi - lo) / 50:.0f} s) → {args.episodes} 个起点 "
            f"[{starts[0]},{starts[-1]}]",
            file=sys.stderr,
        )
    print("\n".join(str(s) for s in starts))


if __name__ == "__main__":
    main()
