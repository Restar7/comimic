#!/usr/bin/env python3
"""把一次批量闭环的产物收成"一任务一个视频 + 索引"的目录。

用法：
    python3 scripts/ha_collect_videos.py runs/ha_isaac_batch_20261024_125538 \
        --out runs/ha_arena_videos

产出：
    <out>/<task alias>__<success|fall|timeout>.mp4
    <out>/INDEX.md            任务 / 模型 / 结果 / 步数 / 视频
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ALIAS = {
    "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody": "HOI_double_desk",
    "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby": "HOI_pp_box",
    "Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody": "HSI_boxing",
    "Isaac-Move-Open-Door-G129-Dex3-Wholebody": "HSI_open_door",
    "Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody": "HSI_sit_sofa",
    "Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody": "HSI_vision_navi",
    "Isaac-Move-ArtVIP-Livingroom-GrapCup-G129-Dex3-Wholebody": "HOI_grap_cup",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for f in sorted(args.batch_dir.glob("runs/**/result.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        vid = d.get("video_path") or ""
        rows.append((d, Path(vid) if vid else None))

    if not rows:
        print(f"no result.json under {args.batch_dir}/runs")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    index = ["# HumanoidArena 闭环视频（Isaac Sim 5.0 + IsaacLab 2.2 + 我们的 VLA）", "",
             f"来源：`{args.batch_dir}`", "",
             "| 任务 | VLA/WBC | 结果 | 步数 | max_reward | 视频 |",
             "|---|---|---|---|---|---|"]
    for d, vid in rows:
        task = str(d.get("task", "?"))
        alias = ALIAS.get(task, task.split("-")[2] if task.count("-") >= 3 else "task")
        reason = str(d.get("failure_reason") or ("success" if d.get("success") else "unknown"))
        label = str(d.get("model_label") or "?")
        name = f"{alias}__{reason}.mp4"
        if vid and vid.is_file():
            shutil.copy2(vid, args.out / name)
            link = f"[{name}]({name})"
        else:
            link = "（无视频）"
        index.append(
            f"| {alias} | {label} | {'✅ success' if d.get('success') else reason} "
            f"| {d.get('episode_steps')} | {d.get('max_reward')} | {link} |")
    index += ["", "成功率由 `sim_eval_vla.py` 的 `success` 字段给出（episode 内 reward 曾 ≥ 1.0）。",
              "视频是前视相机（VLA 的观测）逐帧录制，fall/timeout 会提前截断。"]
    (args.out / "INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print("\n".join(index))
    print(f"\n-> {args.out}/INDEX.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
