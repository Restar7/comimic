#!/usr/bin/env python3
"""A' 全量采集的后处理：合并 + 过滤坏段 + 统计（2026-11-03）。

输入：``runs/a_prime_seg20/off*.pt``（每个 = 16 env × 200 步）
输出：
  * ``merged.pt``   合并后的全部转移（键与采集器一致）
  * ``filtered.pt`` 去掉"段内漂移过大"的坏段（默认阈值 0.30 m）
  * 打印段内漂移分布 / 坏段比例 / 每条 clip 的覆盖情况

为什么要过滤：实测 off0 里 23% 的 20 步段内最大漂移 > 30 cm
（机器人没跟上参考、甚至趔趄），这些段的 (s,a,s') 仍在参考附近之外，
喂给 WM 会学到"跑偏的动力学"，喂给奖励会恒 0。
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "runs/a_prime_seg20")
    ap.add_argument("--packed", type=Path,
                    default=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt")
    ap.add_argument("--segment", type=int, default=20)
    ap.add_argument("--max-drift", type=float, default=0.30,
                    help="段内 root 最大漂移超过这个值就整段丢弃（米）")
    ap.add_argument("--pattern", type=str, default="*_off*.pt",
                    help="分片 glob（新命名是 <Task>_off<N>.pt）")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(str(args.dir / args.pattern)))
    if not files:
        raise SystemExit(f"没有找到 {args.dir}/off*.pt")
    parts = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    keys = ("states", "actions", "next_states", "frames", "clip_ids")
    cat = {k: torch.cat([p[k] for p in parts]) for k in keys}
    meta = dict(parts[0]["meta"])
    meta["n_shards"] = len(files)
    print(f"[merge] {len(files)} 个分片 -> {int(cat['states'].shape[0])} 行")

    ref = torch.as_tensor(
        torch.load(str(args.packed), map_location="cpu", weights_only=False)["ref"]
    ).float()
    fr = cat["frames"].long()
    p = cat["states"][:, :3].numpy()
    pr = ref[fr][:, :3].numpy()

    n = len(p)
    # ★ 2026-11-03 修正：分片里的行是 **16 个 env 交错**存的（row_i = env(i % n_env), step(i // n_env)），
    #   按"连续 20 行"分段会把不同 env 拼在一起 → 段内漂移统计和过滤**全是错的**。
    #   改用**逐行**口径：每个 (s, s') 与参考帧 (f, f+1) 的位移差。
    #   （s' 的帧号 = frames+1；段末那一行 s' 会被重置，误差天然大，单独标记）
    nn = fr + 1
    valid = nn < len(ref)
    d1 = np.zeros(n, dtype=np.float32)
    d1[valid] = np.linalg.norm((cat["next_states"][:, :3].numpy()[valid] - p[valid])
                               - (ref[nn[valid]][:, :3].numpy() - pr[valid]), axis=-1)
    keep = d1 <= args.max_drift
    print(f"[filter] 逐行漂移（s→s' 与参考 f→f+1 的位移差）: "
          f"p50={np.percentile(d1,50)*100:.2f}cm p90={np.percentile(d1,90)*100:.2f}cm "
          f"max={d1.max()*100:.1f}cm")
    for th in (0.02, 0.05, 0.10, 0.30):
        print(f"[filter]   逐行漂移 > {th*100:>4.0f}cm: {100*(d1>th).mean():5.1f}%")
    print(f"[filter] 阈值 {args.max_drift*100:.0f}cm -> 丢弃 {100*(~keep).mean():.1f}% 的行，"
          f"保留 {int(keep.sum())}/{n}")

    out_all = args.out or (args.dir / "merged.pt")
    cat_all = dict(cat)
    cat_all["meta"] = meta
    torch.save(cat_all, str(out_all))
    print(f"[save] {out_all}  {tuple(cat_all['states'].shape)}")

    filt = {k: cat[k][torch.from_numpy(keep)] for k in keys}
    filt["meta"] = dict(meta, max_drift_filter=float(args.max_drift),
                        kept_rows=int(keep.sum()), total_rows=int(n))
    out_f = args.dir / "filtered.pt"
    torch.save(filt, str(out_f))
    print(f"[save] {out_f}  {tuple(filt['states'].shape)}")

    clips = sorted({int(c) for c in cat["clip_ids"].tolist()})
    print(f"[cover] 覆盖 clip 数 = {len(clips)} / 699")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
