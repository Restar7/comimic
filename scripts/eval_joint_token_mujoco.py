#!/usr/bin/env python3
"""Closed-loop MuJoCo eval of a **token-conditioned** joint/SuperMimic checkpoint.

``scripts/eval_mujoco.py`` only knows the paper route (a 680-d goal window from
``MotionLibrary.goal_window``); every token-conditioned checkpoint we train
(``PolicyCfg.token_dim = 64``) raises
``ValueError: token must have 64 dims, got 680`` there.  This script feeds the
checkpoint what it was trained on instead: the **recorded** 64-d SONIC motion
token (``g1_motion_token_64``) of the ft318 npz, one token per control tick,
while MuJoCo integrates the real 29-DoF physics.

So the number it reports is "how well does this WBC track the ft318 clip when
the token is perfect" — the oracle-token closed loop, not the VLA closed loop.

Usage:
    python3 scripts/eval_joint_token_mujoco.py \
        --joint-ckpt runs/joint_sm24k_hand/joint_001000.pt \
        --clips data/npz/g1_tasks/task_1_*.npz ... --max-steps 300 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
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


def recorded_tokens(path: Path, device: torch.device) -> torch.Tensor:
    with np.load(path) as d:
        if "g1_motion_token_64" not in d:
            raise KeyError(f"{path.name} has no g1_motion_token_64")
        tok = np.ascontiguousarray(d["g1_motion_token_64"])
    return torch.as_tensor(tok, dtype=torch.float32, device=device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", type=Path, required=True)
    ap.add_argument("--joint-ckpt", type=Path, default=None,
                    help="joint trainer state (policy + WM config); token route")
    ap.add_argument("--supermimic-ckpt", type=Path, default=None,
                    help="plain SuperMimic ckpt (before any joint finetune)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-steps", type=int, default=300, help="control ticks (50 Hz) per clip")
    ap.add_argument("--teacher", action="store_true",
                    help="reference controller baseline (next ref joint target), not the policy")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/eval_joint_token")
    ap.add_argument("--tag", default="", help="label written into the summary (e.g. hand@1000)")
    args = ap.parse_args()

    device = torch.device(args.device)
    assets = build_g1_assets(device=device)
    if args.joint_ckpt:
        bundle = JointTrainer.load_bundle_from_checkpoint(args.joint_ckpt, assets)
        print(f"[info] joint ckpt {args.joint_ckpt}")
    elif args.supermimic_ckpt:
        bundle = SuperMimicBundle.from_checkpoint(args.supermimic_ckpt, assets)
        print(f"[info] supermimic ckpt {args.supermimic_ckpt}")
    else:
        raise SystemExit("pass --joint-ckpt or --supermimic-ckpt")
    if int(getattr(bundle.policy, "token_dim", 0)) != 64:
        raise SystemExit(f"policy token_dim={getattr(bundle.policy, 'token_dim', 0)}; "
                         "this script is for the 64-d token route")

    env = G1MujocoEnv(bundle)
    mode = "teacher" if args.teacher else "policy"
    rows = []
    for clip in args.clips:
        library = load_motion_library([clip], assets, device=device)
        tokens = recorded_tokens(clip, assets.device)
        # ``rollout`` asks the library for the conditioning every tick; here that
        # conditioning is the recorded token of the current frame.
        library.goal_window = lambda frames, k, _t=tokens: _t[frames.long().reshape(-1)]
        result = env.rollout(library, clip_index=0, max_steps=args.max_steps, mode=mode)
        row = {k: v for k, v in asdict(result).items() if k != "trajectory"}
        rows.append(row)
        print(f"[{clip.stem}] reward={row['mean_reward']:+.4f} norm_ret={row['norm_return']:+.4f} "
              f"joint_rmse={row['joint_rmse']:.4f} root_rmse={row['root_rmse']:.4f} "
              f"survival={row['survival']:.3f} failed={row['failed']}@{row['fail_step']}")

    def agg(key: str) -> float:
        vals = [r[key] for r in rows]
        vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "tag": args.tag,
        "mode": mode,
        "ckpt": str(args.joint_ckpt or args.supermimic_ckpt),
        "max_steps": args.max_steps,
        "clips": len(rows),
        "mean_reward": agg("mean_reward"),
        "norm_return": agg("norm_return"),
        "joint_rmse": agg("joint_rmse"),
        "root_rmse": agg("root_rmse"),
        "survival": agg("survival"),
        "failures": sum(1 for r in rows if r["failed"]),
        "per_clip": rows,
    }
    print("[summary] " + json.dumps({k: v for k, v in summary.items() if k != "per_clip"},
                                    ensure_ascii=False))
    args.out.mkdir(parents=True, exist_ok=True)
    name = (args.tag or "run").replace("/", "_")
    (args.out / f"{name}.json").write_text(json.dumps(summary, indent=2))
    print(f"[saved] {args.out / f'{name}.json'}")


if __name__ == "__main__":
    main()
