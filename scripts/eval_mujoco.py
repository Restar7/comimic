#!/usr/bin/env python3
"""Closed-loop MuJoCo evaluation of the trained WBC / joint checkpoint.

Example:
    python scripts/eval_mujoco.py \
        --motion data/npz/example \
        --joint-ckpt runs/joint_smoke/joint_latest.pt \
        --clip-index 0 --max-steps 100

This is the G1 model from ``GR00T-WholeBodyControl/gear_sonic_deploy/g1`` with
the 29-DoF deploy actuator set; it is *not* mjlab/MuJoCo-Warp.  It is enough to
check that the learned policy actually moves the robot and to compare against a
frozen-WBC baseline.  A 0% success result with a randomly initialized policy is
expected and is not evidence about the method.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diff_vla_wbc.mujoco_env import G1MujocoEnv  # noqa: E402
from diff_vla_wbc.supermimic_backend import (  # noqa: E402
    SuperMimicBundle,
    build_g1_assets,
    load_motion_library,
)
from diff_vla_wbc.train import JointTrainer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", nargs="+", type=Path, required=True)
    ap.add_argument("--supermimic-ckpt", type=Path, default=None)
    ap.add_argument("--joint-ckpt", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--clip-index", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/eval"))
    ap.add_argument("--teacher", action="store_true",
                    help="use the next reference joint target as the action (controller baseline)")
    args = ap.parse_args()

    device = torch.device(args.device)
    assets = build_g1_assets(device=device)
    if args.joint_ckpt:
        # The joint checkpoint carries the exact policy/WM architecture, so it
        # is safe even when --small was used during training.
        bundle = JointTrainer.load_bundle_from_checkpoint(args.joint_ckpt, assets)
        adapter = JointTrainer.load_adapter_from_checkpoint(args.joint_ckpt, device)
        print(f"[info] loaded joint checkpoint {args.joint_ckpt} (adapter + policy + world model)")
    elif args.supermimic_ckpt:
        bundle = SuperMimicBundle.from_checkpoint(args.supermimic_ckpt, assets)
        adapter = None
    else:
        bundle = SuperMimicBundle.random(assets, small=args.small)
        adapter = None
        print("[warn] no --joint-ckpt: evaluating a randomly initialized policy; expect falls")

    files = []
    for p in args.motion:
        if p.is_dir():
            files.extend(sorted(p.glob("*.npz")))
        elif p.suffix == ".npz":
            files.append(p)
    if not files:
        raise SystemExit("no motion npz files found")
    library = load_motion_library(files, assets, device=device)
    print(f"[info] {library.summary()}")

    env = G1MujocoEnv(bundle)
    result = env.rollout(
        library,
        clip_index=args.clip_index,
        max_steps=args.max_steps,
        mode="teacher" if args.teacher else "policy",
    )
    print(json.dumps({k: v for k, v in asdict(result).items() if k != "trajectory"}, indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / f"eval_{result.clip}.json"
    out_json.write_text(json.dumps({k: v for k, v in asdict(result).items() if k != "trajectory"}, indent=2))
    import numpy as np

    for name, values in result.trajectory.items():
        if getattr(values, "size", 0) and values.ndim > 0:
            np.save(args.out / f"{result.clip}_{name}.npy", values)
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
