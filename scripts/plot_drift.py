#!/usr/bin/env python3
"""从一条 B dump 画漂移曲线（root / joint / tilt），并保存 json + png。

用法：
    PYTHONPATH=. python3 scripts/plot_drift.py --dump runs/B_r1/vla_DoubleDesk/dump/*.pt \
        --ref runs/isaac_motion/ha_cmd/packed.pt --out runs/B_r1/drift_vla.png
    # 多条 dump 叠一张图：--dump 'a.pt' --dump 'b.pt' ...（可重复）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True, help="B dump 路径（可重复）")
    ap.add_argument("--ref", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/drift.png")
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    lib = torch.load(str(a.ref), map_location="cpu", weights_only=False)["ref"].float()
    n_lib = int(lib.shape[0])
    curves = []
    for p in a.dump:
        d = torch.load(p, map_location="cpu", weights_only=False)
        S = d["states"].float()
        fr = d["frames"].long().clamp(0, n_lib - 1)
        ref = lib[fr]
        e_root = (S[:, :3] - ref[:, :3]).norm(dim=-1)
        joint = ((S[:, 7:36] - ref[:, 7:36]) ** 2).mean(-1).sqrt()
        q = S[:, 3:7]
        w, x, y, z = q.unbind(-1)
        up = 1 - 2 * (x * x + y * y)
        tilt = torch.rad2deg(torch.acos(up.clamp(-1, 1)))
        curves.append({"name": Path(p).stem, "root": e_root, "joint": joint, "tilt": tilt})

    for c in curves:
        r, j, t = c["root"], c["joint"], c["tilt"]
        print(f"[drift] {c['name']:28s} n={len(r):5d} "
              f"root mean={float(r.mean()):.3f} p50={float(r.quantile(.5)):.3f} "
              f"p90={float(r.quantile(.9)):.3f} | joint mean={float(j.mean()):.3f} "
              f"| tilt max={float(t.max()):.1f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[drift] 没装 matplotlib，只出 json: {exc}")
        a.out.with_suffix(".json").write_text(json.dumps(
            {c["name"]: {"root": [round(float(x), 4) for x in c["root"][::5]],
                         "joint": [round(float(x), 4) for x in c["joint"][::5]],
                         "tilt": [round(float(x), 2) for x in c["tilt"][::5]]}
             for c in curves}, indent=1))
        return 0

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for c in curves:
        x = range(len(c["root"]))
        ax[0].plot(x, c["root"], label=c["name"], lw=1)
        ax[1].plot(x, c["joint"], label=c["name"], lw=1)
        ax[2].plot(x, c["tilt"], label=c["name"], lw=1)
    ax[0].set_title("root error (m) vs reference")
    ax[1].set_title("joint RMSE (rad)")
    ax[2].set_title("tilt (deg)")
    for a_ in ax:
        a_.set_xlabel("control step")
        a_.grid(alpha=.3)
        a_.legend(fontsize=8)
    ax[2].axhline(60, ls="--", c="r", lw=.8)
    if a.title:
        fig.suptitle(a.title)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"[drift] saved {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
