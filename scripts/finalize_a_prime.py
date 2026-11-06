#!/usr/bin/env python3
"""A' 收尾：G2 校验 + 过滤趴地行 + 统计。

用法：
  python3 scripts/finalize_a_prime.py --in runs/wm_real/exp_a_prime_full/a_full/a_prime.pt \
      --out runs/wm_real/exp_a_prime_full/a_full/a_prime_clean.pt [--z-min 0.4] [--up-min 0.5]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def root_up(state: torch.Tensor) -> torch.Tensor:
    q = state[:, 3:7]
    return 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", dest="dst", type=Path, required=True)
    ap.add_argument("--packed", type=Path, default=Path("runs/isaac_motion/ha_cmd_exec/packed.pt"))
    ap.add_argument("--z-min", type=float, default=0.4, help="根高低于它算摔倒（默认 0.4 m）")
    ap.add_argument("--up-min", type=float, default=0.5, help="朝上分量低于它算倒下（默认 0.5）")
    a = ap.parse_args()

    d = torch.load(a.src, map_location="cpu", weights_only=False)
    lib = torch.load(a.packed, map_location="cpu", weights_only=False)
    starts = torch.as_tensor(lib["starts"]).long()
    lengths = torch.as_tensor(lib["lengths"]).long()

    S, F = d["states"].float(), d["frames"].long()
    C = d.get("clip_ids")
    n = S.shape[0]
    up = root_up(S)
    z = S[:, 2]
    keep = (z >= a.z_min) & (up >= a.up_min)

    print(f"[final] 总数 {n} 行，clip {len(C.unique()) if C is not None else 'NA'}")
    print(f"[final] 摔倒/倾倒行 {int((~keep).sum())} ({(~keep).float().mean():.2%})  →  保留 {int(keep.sum())}")
    print(f"[final] z   mean={z.mean():.3f}  min={z.min():.3f}   up mean={up.mean():.3f}")

    # G2：frames 必须在全局 packed 行范围内、且在各自 clip 内
    fmin, fmax = int(F.min()), int(F.max())
    print(f"[final] frames 范围 ({fmin}, {fmax})   期望上限≈{int(starts[-1] + lengths[-1] - 1)}")
    if C is not None:
        lo = starts[C]
        hi = lo + lengths[C]
        bad = int((F < lo).sum() + (F >= hi).sum())
        print(f"[final] 越界 frame {bad}（必须 0）")
    # 覆盖率：这 634,016 个库里有多少帧出现过
    covered = int(torch.unique(F).numel())
    total = int(lengths.sum())
    print(f"[final] 覆盖帧 {covered}/{total} ({covered / total:.2%})")

    out = {k: v[keep] for k, v in d.items() if torch.is_tensor(v) and v.shape[:1] == (n,)}
    for k, v in d.items():
        if k not in out:
            out[k] = v
    meta = dict(d.get("meta") or {})
    meta.update({"finalize": {"z_min": a.z_min, "up_min": a.up_min,
                              "dropped": int((~keep).sum()), "kept": int(keep.sum())}})
    out["meta"] = meta
    a.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(a.dst))
    print(f"[final] saved {a.dst}  rows={out['states'].shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
