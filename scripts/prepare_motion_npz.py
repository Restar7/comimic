#!/usr/bin/env python3
"""Convert the G1 motion clips in this workspace into SuperMimic npz libraries.

Examples:
    # SONIC per-body CSV export directories
    python scripts/prepare_motion_npz.py \
        --csv-root GR00T-WholeBodyControl/gear_sonic_deploy/reference/example \
        --out-dir data/npz/example

    # LAFAN1 36-column G1 retarget CSVs
    python scripts/prepare_motion_npz.py \
        --lafan1-dir data/lafan1_g1 --out-dir data/npz/lafan1

BONES-SEED is a gated 120 fps dataset.  Use the upstream converter after you
have access:
    uv run python scripts/data/filter_bones_seed.py ...
    uv run python scripts/data/build_library.py ...
This repository then loads the packed library with ``compact=True``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diff_vla_wbc.data import motion_npz_from_body_csv_dir, motion_npz_from_lafan1_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-root", type=Path, help="Directory of SONIC per-body CSV clip dirs")
    ap.add_argument("--lafan1-dir", type=Path, help="Directory with LAFAN1 36-column CSV clips")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--xml", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--input-fps", type=float, default=30.0, help="LAFAN1 source fps")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    if args.csv_root:
        for clip_dir in sorted(p for p in args.csv_root.iterdir() if p.is_dir()):
            required = ["joint_pos.csv", "body_pos.csv", "body_quat.csv", "body_lin_vel.csv", "body_ang_vel.csv"]
            if not all((clip_dir / name).is_file() for name in required):
                print(f"[skip] {clip_dir.name}: missing per-body CSVs")
                continue
            out = args.out_dir / f"{clip_dir.name}.npz"
            motion_npz_from_body_csv_dir(
                clip_dir, out, xml_path=args.xml, fps=args.fps
            )
            print(f"[ok] {clip_dir.name} -> {out}")
            made += 1
    if args.lafan1_dir:
        for csv_path in sorted(args.lafan1_dir.glob("*.csv")):
            out = args.out_dir / f"{csv_path.stem}.npz"
            motion_npz_from_lafan1_csv(
                csv_path,
                out,
                xml_path=args.xml,
                input_fps=args.input_fps,
                output_fps=args.fps,
            )
            print(f"[ok] {csv_path.name} -> {out}")
            made += 1
    if made == 0:
        raise SystemExit("No clips converted; pass --csv-root and/or --lafan1-dir")
    print(f"converted {made} clip(s) into {args.out_dir}")


if __name__ == "__main__":
    main()
