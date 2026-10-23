#!/usr/bin/env python
"""Plot the reproduction: policy prediction vs. teleoperated ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="/home/wusichen/groot_repro/out/inference_result.json")
    ap.add_argument("--frame", type=int, default=1000)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", default="/home/wusichen/groot_repro/out/reproduction.png")
    args = ap.parse_args()

    res = json.loads(Path(args.result).read_text())
    npz = np.load(Path(args.result).with_suffix(".npz"))
    frame = args.frame
    s = args.sample

    rgb = npz[f"frame_{frame}_rgb"]
    pred = npz[f"frame_{frame}_sample_{s}_motion_token"]
    targ = npz[f"frame_{frame}_target_motion_token"]
    mae_h = np.abs(pred - targ).mean(axis=1)

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb)
    ax.set_title(f"ego_view @ frame {frame}", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1:])
    vmax = float(max(np.abs(pred).max(), np.abs(targ).max()))
    im = ax.imshow(pred.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"predicted motion_token (sample {s})", fontsize=10)
    ax.set_xlabel("horizon step")
    ax.set_ylabel("token dim")
    fig.colorbar(im, ax=ax, fraction=0.025)

    ax = fig.add_subplot(gs[1, 0:2])
    im = ax.imshow(targ.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("recorded (teleoperated) motion_token", fontsize=10)
    ax.set_xlabel("horizon step")
    ax.set_ylabel("token dim")
    fig.colorbar(im, ax=ax, fraction=0.025)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(np.arange(40), mae_h, lw=1.6, color="#c0392b")
    ax.set_title(f"|error| over horizon (mean {mae_h.mean():.3f})", fontsize=10)
    ax.set_xlabel("horizon step")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.3)

    m = res["frames"][[f["frame_index"] for f in res["frames"]].index(frame)]["samples"][s][
        "metrics"
    ]["motion_token"]
    fig.suptitle(
        "GR00T N1.7 / UNITREE_G1_SONIC - sirius checkpoint-50000 reproduced on CPU\n"
        f"motion_token: MAE={m['mae']:.4f}  corr={m['correlation']:.3f}  "
        f"1/16-quantised match={m['quantized_1_16_match_fraction']:.2f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
