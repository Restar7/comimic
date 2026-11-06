#!/usr/bin/env python3
"""把仿真采集 + B 线（VLA 在环、任务场景）按比例混成一个 WM 训练集。

为什么：WM 只在平地的仿真 transitions 上学 ⇒ 没见过物体接触与 VLA 分布，`qd_ratio` 偏 16%。
混 20–30% 的 B 线数据，让 WM 的想象覆盖它实际要服务的分布。

    python3 scripts/make_wm_mix.py --sim round_1/transitions.pt \
        --bline round_1/bline_transitions.pt --ratio 0.25 --out round_1/wm_mix.pt

`--ratio` 是 **B 线占比**（按行数）；`--seed` 控制下采样，保证可复现。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

KEYS = ("states", "next_states", "actions", "clean_actions", "xi",
        "rewards", "ret", "frames", "ref_rows", "clip_ids", "ep_ids")
# frame_map 是"录制帧 -> 行号"的索引表，混合后语义失效，写 arange 占位
KEYS_INDEX = ("frame_map",)


def take(d: dict, idx: torch.Tensor) -> dict:
    out = {}
    for k in KEYS:
        if k in d:
            out[k] = d[k][idx]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="collect_mjlab 的 transitions.pt")
    ap.add_argument("--bline", required=True, help="bline_to_transitions.py 的产物")
    ap.add_argument("--ratio", type=float, default=0.25, help="B 线占混后总行数的比例")
    ap.add_argument("--downsample-sim", action="store_true",
                    help="旧行为：下采样场景来凑比例；默认改用**重复 B 线**（场景一行不丢）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sim = torch.load(args.sim, map_location="cpu", weights_only=False)
    bl = torch.load(args.bline, map_location="cpu", weights_only=False)
    n_bl = int(bl["states"].shape[0])
    n_sim = int(sim["states"].shape[0])
    r = min(max(float(args.ratio), 0.0), 0.9)

    # 目标：n_bl_use / (n_sim_use + n_bl_use) = r
    if r <= 0:
        mix = take(sim, torch.arange(n_sim))
        print(f"[mix] ratio=0 ⇒ 只用仿真 {n_sim} 行")
    elif args.downsample_sim:
        # 旧行为：下采样仿真来凑比例（会丢掉大量场景/接触样本）
        n_sim_use = min(int(round(n_bl * (1 - r) / r)), n_sim)
        g = torch.Generator().manual_seed(int(args.seed))
        idx_sim = torch.randperm(n_sim, generator=g)[:n_sim_use]
        mix = take(sim, idx_sim)
        for k in KEYS:
            if k in bl and k in mix:
                mix[k] = torch.cat([mix[k], bl[k]])
            elif k in bl:
                mix[k] = bl[k]
        print(f"[mix] B 线全部 {n_bl} 行 + 仿真下采样 {n_sim_use}/{n_sim} 行 "
              f"⇒ B 线实际占比 {100 * n_bl / (n_bl + n_sim_use):.1f}%")
    else:
        # ★ 默认：仿真**全量保留**，把 B 线重复 k 次来达到目标占比。
        #   原因（2026-10-29）：场景采集 102 万行里最稀缺的是"物体接触"，下采样会丢掉 94%；
        #   而 B 线只有 4.8 万行，重复它不会带来新信息损失，却能给出足够的损失权重。
        k = max(1, int(round(r * n_sim / max((1.0 - r) * n_bl, 1.0))))
        mix = take(sim, torch.arange(n_sim))
        for kk in KEYS:
            if kk in bl and kk in mix:
                mix[kk] = torch.cat([mix[kk]] + [bl[kk]] * k)
            elif kk in bl:
                mix[kk] = torch.cat([bl[kk]] * k)
        print(f"[mix] 仿真全量 {n_sim} 行 + B 线 {n_bl} 行 ×{k} = {n_bl * k} 行 "
              f"⇒ B 线等效占比 {100 * n_bl * k / (n_sim + n_bl * k):.1f}%")

    n = int(mix["states"].shape[0])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(args.seed) + 1))
    mix = {k: v[perm] for k, v in mix.items()}
    mix["frame_map"] = torch.arange(n, dtype=torch.long)
    mix["meta"] = {
        "source": "sim+bline mix",
        "sim": str(args.sim), "bline": str(args.bline),
        "ratio_bline_target": r, "rows": n,
        "rows_sim": n_sim, "rows_bline": n_bl,
        "note": "B 线来自闭环（VLA 在环、任务场景），xi 置零",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mix, str(args.out))
    print(f"[mix] -> {args.out}  rows={n}  ret mean={float(mix['ret'].mean()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
