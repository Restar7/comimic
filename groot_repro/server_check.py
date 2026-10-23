#!/usr/bin/env python
"""Query a live GR00T PolicyServer exactly like the Sonic deployment client.

Starts nothing itself: point it at a running `gr00t/eval/run_gr00t_server.py`
and it will build the deployment observation dict for a recorded frame and call
`PolicyClient.get_action` over ZMQ, i.e. the same wire path used by

    GR00T-WholeBodyControl/gear_sonic/scripts/run_vla_inference.py
      -> gr00t.policy.server_client.PolicyClient
      -> PolicyServer on :5550
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from repro_infer import GROUPS, STATE_ORDER, build_observation, metrics, _col, _read_frame
from gr00t.policy.server_client import PolicyClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="/home/wusichen/SONIC_DATA/2026-07-14/datasets/2026-07-14-20-30-01",
    )
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=1000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5550)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument(
        "--prompt",
        default=(
            "Pick up the apple from the table in front of you, turn around, walk "
            "to the table behind you, and place the apple into the box."
        ),
    )
    ap.add_argument("--out", default="/home/wusichen/groot_repro/out/server_check.json")
    args = ap.parse_args()

    root = Path(args.dataset)
    ep = f"episode_{args.episode:06d}"
    table = pq.read_table(root / "data" / "chunk-000" / f"{ep}.parquet")
    q = _col(table, "observation.state")
    g = _col(table, "observation.projected_gravity")
    targets = {
        "motion_token": _col(table, "action.motion_token")[args.frame : args.frame + 40],
        "left_hand_joints": _col(table, "teleop.left_hand_joints")[
            args.frame : args.frame + 40
        ],
        "right_hand_joints": _col(table, "teleop.right_hand_joints")[
            args.frame : args.frame + 40
        ],
    }
    rgb = _read_frame(
        root / "videos" / "chunk-000" / "observation.images.ego_view" / f"{ep}.mp4",
        args.frame,
    )
    obs = build_observation(q[args.frame], g[args.frame], rgb, args.prompt)

    # CPU inference takes ~25 s, well past PolicyClient's 15 s default RCVTIMEO.
    client = PolicyClient(host=args.host, port=args.port, timeout_ms=300_000)
    if not client.ping():
        raise SystemExit(f"no policy server at {args.host}:{args.port}")
    print(f"connected to {args.host}:{args.port}")

    result = {"host": args.host, "port": args.port, "frame": args.frame, "samples": []}
    for s in range(args.samples):
        action, info = client.get_action(obs)
        flat = {}
        for key, dim in (("motion_token", 64), ("left_hand_joints", 7), ("right_hand_joints", 7)):
            v = np.asarray(action.get(key, action.get(f"action.{key}")), np.float32)
            while v.ndim > 2 and v.shape[0] == 1:
                v = v[0]
            flat[key] = v
            print(f"  {key}: {v.shape} finite={np.isfinite(v).all()}")
        result["samples"].append(
            {
                "sample_index": s,
                "metrics": {k: metrics(v, targets[k]) for k, v in flat.items()},
                "info_keys": sorted(info) if isinstance(info, dict) else [],
            }
        )
        print(
            f"  sample {s}: motion_token mae="
            f"{result['samples'][-1]['metrics']['motion_token']['mae']:.4f} "
            f"corr={result['samples'][-1]['metrics']['motion_token']['correlation']:.3f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
