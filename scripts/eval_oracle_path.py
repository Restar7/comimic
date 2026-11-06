#!/usr/bin/env python3
"""Oracle 路径比评测：喂**录制 token**（`--arm gt`，VLA 不参与）时，机器人走出的轨迹
和参考轨迹的比值 —— 判断 WBC「走够了没有」的唯一直接量。

为什么需要（2026-10-30）：
  参考库补上根轨迹（`ft318_compact_v2.pt`）之后，奖励里的 `motion_global_anchor_position`
  第一次有真目标。但"能不能走够"必须单独量，不能只看训练 return：
  用录制 token 关掉 VLA 这一层，剩下的差异就只属于 WBC 的步态标定。

用法：
    python3 scripts/eval_oracle_path.py --ref data/library/ft318_compact_v2.pt \\
        --dump 'runs/ref_v2/oracle/gt_*.pt'
输出每个 dump 的：机器人路径 / 参考路径 / 比值 / 净位移 / pos 误差分位 / tilt。
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from eval_realized_drift import per_tick_frames  # noqa: E402


def _up(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([2.0 * (x * z + w * y),
                        2.0 * (y * z - w * x),
                        1.0 - 2.0 * (x * x + y * y)], -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs="+", required=True)
    ap.add_argument("--ref", type=Path, default=ROOT / "data/library/ft318_compact_v2.pt")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    ref = torch.load(str(a.ref), map_location="cpu", weights_only=False)["ref"].float()
    files: list[str] = []
    for g in a.dump:
        files += sorted(glob.glob(g))
    if not files:
        raise SystemExit("[fail] 没有匹配到 dump")

    rows = []
    print(f"{'dump':<22}{'S路径':>8}{'R路径':>8}{'比值':>7}{'R净移':>8}{'S净移':>8}"
          f"{'pos_p50':>8}{'pos_p90':>8}{'tilt_p50':>9}{'tilt_p90':>9}  单位 cm/°")
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        S = d["states"].float()
        PF = per_tick_frames(d["frames"], S.shape[0])
        R = ref[PF]
        ps, pr = S[:, 0:2], R[:, 0:2]
        path_s = float((ps[1:] - ps[:-1]).norm(dim=-1).sum() * 100)
        path_r = float((pr[1:] - pr[:-1]).norm(dim=-1).sum() * 100)
        net_r = float((pr[-1] - pr[0]).norm() * 100)
        net_s = float((ps[-1] - ps[0]).norm() * 100)
        pos = (S[:, 0:3] - R[:, 0:3]).norm(dim=-1) * 100
        tilt = torch.acos((_up(S[:, 3:7]) * _up(R[:, 3:7])).sum(-1).clamp(-1, 1)) * (180 / 3.141592653589793)
        rows.append({"file": Path(f).name, "path_s": path_s, "path_r": path_r,
                     "ratio": path_s / max(path_r, 1e-6), "net_s": net_s, "net_r": net_r,
                     "pos_p50": float(pos.median()), "pos_p90": float(pos.quantile(0.9)),
                     "tilt_p50": float(tilt.median()), "tilt_p90": float(tilt.quantile(0.9))})
        r = rows[-1]
        print(f"{r['file']:<22}{path_s:>8.1f}{path_r:>8.1f}{r['ratio']:>7.2f}{net_r:>8.1f}"
              f"{net_s:>8.1f}{r['pos_p50']:>8.1f}{r['pos_p90']:>8.1f}"
              f"{r['tilt_p50']:>9.1f}{r['tilt_p90']:>9.1f}")

    n = len(rows)
    ratios = [r["ratio"] for r in rows]
    print(f"\n[汇总] n={n} 路径比: 中位 {sorted(ratios)[n // 2]:.2f} "
          f"min {min(ratios):.2f} max {max(ratios):.2f}")
    print(f"       判定：中位比 0.9–1.1 ⇒ WBC 步态标定到位；< 0.7 ⇒ 仍走不够；> 1.3 ⇒ 走过头")
    if a.out:
        import json
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"rows": rows}, indent=1, ensure_ascii=False))
        print(f"[out] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
