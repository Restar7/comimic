#!/usr/bin/env python3
"""把两个闭环录像拼成**左右并排**的对照视频（左=骨架/基线，右=候选）。

为什么要它：`runs/DEMO_*` 里每个任务每个臂各一条 mp4，肉眼看两条太麻烦；
并排 + 顶部标注（模型名 + 帧号）能一眼看出差别。

用法：
    python3 scripts/make_sidebyside_video.py \
        --left  runs/DEMO_ALLTASKS/DoubleDesk_base_vs_ours_t60/main_r0/runs/*/videos_failure/*.mp4 \
        --right runs/DEMO_ALLTASKS/DoubleDesk_base_vs_ours_t60/ctrl_r0/runs/*/videos_failure/*.mp4 \
        --label-left "baseline (base+v6) 1.384m" \
        --label-right "ours (h20r2+v7) 1.238m" \
        --out runs/DEMO_ALLTASKS/DoubleDesk_sidebyside.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--label-left", default="left")
    ap.add_argument("--label-right", default="right")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=0, help="0=全部")
    a = ap.parse_args()

    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    lpath, rpath = Path(a.left).expanduser(), Path(a.right).expanduser()
    for p in (lpath, rpath):
        if not p.exists():
            print(f"[sidebyside] 找不到 {p}", file=sys.stderr)
            return 2
    caps = [cv2.VideoCapture(str(lpath)), cv2.VideoCapture(str(rpath))]
    if not all(c.isOpened() for c in caps):
        print("[sidebyside] 打不开输入视频", file=sys.stderr)
        return 2
    n = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps))
    if a.max_frames:
        n = min(n, int(a.max_frames))
    w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 两路分辨率不同时按左边对齐
    for i, c in enumerate(caps):
        if int(c.get(cv2.CAP_PROP_FRAME_WIDTH)) != w or int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)) != h:
            print(f"[sidebyside][warn] 第 {i} 路分辨率不同，缩放对齐到 {w}x{h}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(a.fps), (w * 2, h))
    if not vw.isOpened():
        print("[sidebyside] VideoWriter 打不开（缺 mp4v 编码器）", file=sys.stderr)
        return 3
    labels = (a.label_left, a.label_right)
    for i in range(n):
        tiles = []
        for k, c in enumerate(caps):
            ok, img = c.read()
            if not ok:
                img = np.zeros((h, w, 3), np.uint8)
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h))
            cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
            cv2.putText(img, f"{labels[k]}   frame {i}", (6, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        vw.write(np.hstack(tiles))
    vw.release()
    for c in caps:
        c.release()
    print(f"[sidebyside] {out}  帧数={n}  尺寸={w*2}x{h}  {out.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
