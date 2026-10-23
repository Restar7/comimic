#!/usr/bin/env python
"""Offline reproduction of the Sonic / GR00T N1.7 deployment inference path.

For each requested frame of a recorded Sonic episode this script rebuilds the
*exact* observation dictionary the deployment client sent to the policy server
(see deployment/bin/offline_episode_policy_compare.py and
GR00T-WholeBodyControl/gear_sonic/utils/inference/vla_utils.py), runs the
checkpoint through ``Gr00tPolicy.get_action`` (the same class the policy server
serves), and scores the predicted action chunk against the action that was
actually recorded at teleoperation time.

Observation contract for EmbodimentTag.UNITREE_G1_SONIC:
    video   : {"ego_view": uint8 (1, 1, H, W, 3)}          single frame
    state   : left_leg(6) right_leg(6) waist(3) left_arm(7)
              right_arm(7) left_hand(7) right_hand(7)
              projected_gravity(3), each float32 (1, 1, D)
    language: {"annotation.human.task_description": [[prompt]]}
Action contract:
    motion_token(64) + left_hand_joints(7) + right_hand_joints(7), horizon 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

# Joint-group layout of observation.state, from the dataset's meta/modality.json.
GROUPS = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "right_arm": (29, 36),
    "left_hand": (22, 29),
    "right_hand": (36, 43),
}
# Order the model expects (processor_config.json -> unitree_g1_sonic.state.modality_keys)
STATE_ORDER = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "projected_gravity",
]
ACTION_DIMS = {"motion_token": 64, "left_hand_joints": 7, "right_hand_joints": 7}
HORIZON = 40


def _col(table, name: str) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), dtype=np.float32)


def _read_frame(video: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index} of {video}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_observation(q, gravity, rgb, prompt: str) -> dict:
    """Rebuild the deployment observation dict for one frame."""
    state = {}
    for name in STATE_ORDER:
        if name == "projected_gravity":
            vec = gravity
        else:
            lo, hi = GROUPS[name]
            vec = q[lo:hi]
        state[name] = np.asarray(vec, dtype=np.float32)[np.newaxis, np.newaxis]
    return {
        "video": {"ego_view": np.asarray(rgb, dtype=np.uint8)[np.newaxis, np.newaxis]},
        "state": state,
        "language": {"annotation.human.task_description": [[prompt]]},
    }


def metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    err = pred - target
    a = pred.ravel() - pred.mean()
    b = target.ravel() - target.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "max_abs_error": float(np.abs(err).max()),
        "correlation": float(a.dot(b) / denom) if denom > 0 else float("nan"),
        "quantized_1_16_match_fraction": float(
            np.mean(np.round(pred * 16) == np.round(target * 16))
        ),
        "pred_min": float(pred.min()),
        "pred_max": float(pred.max()),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/wusichen/groot_repro/ckpt_sirius_50k")
    ap.add_argument(
        "--dataset",
        default="/home/wusichen/SONIC_DATA/2026-07-14/datasets/2026-07-14-20-30-01",
    )
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 800, 1600, 2400])
    ap.add_argument("--embodiment-tag", default="unitree_g1_sonic")
    ap.add_argument(
        "--prompt",
        default=(
            "Pick up the apple from the table in front of you, turn around, walk "
            "to the table behind you, and place the apple into the box."
        ),
    )
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/wusichen/groot_repro/out/inference_result.json")
    args = ap.parse_args()

    # ---- recorded data -----------------------------------------------------
    root = Path(args.dataset)
    ep = f"episode_{args.episode:06d}"
    table = pq.read_table(root / "data" / "chunk-000" / f"{ep}.parquet")
    q = _col(table, "observation.state")
    gravity = _col(table, "observation.projected_gravity")
    recorded = {
        "motion_token": _col(table, "action.motion_token"),
        "left_hand_joints": _col(table, "teleop.left_hand_joints"),
        "right_hand_joints": _col(table, "teleop.right_hand_joints"),
    }
    video = root / "videos" / "chunk-000" / "observation.images.ego_view" / f"{ep}.mp4"
    print(f"episode {ep}: {table.num_rows} frames, q={q.shape}, video={video.name}")

    # ---- policy (identical class to the deployment's PolicyServer) ---------
    import meta_patch

    meta_patch.install()
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    t0 = time.monotonic()
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.ckpt,
        device=args.device,
        strict=True,
    )
    load_s = time.monotonic() - t0
    print(f"policy loaded on {args.device} in {load_s:.1f}s")

    result = {
        "checkpoint": args.ckpt,
        "embodiment_tag": args.embodiment_tag,
        "dataset": str(root),
        "episode": args.episode,
        "prompt": args.prompt,
        "device": args.device,
        "load_seconds": round(load_s, 1),
        "frames": [],
    }
    arrays: dict[str, np.ndarray] = {}

    for frame in args.frames:
        if frame + HORIZON > table.num_rows:
            raise ValueError(f"frame {frame} lacks {HORIZON} recorded targets")
        rgb = _read_frame(video, frame)
        obs = build_observation(q[frame], gravity[frame], rgb, args.prompt)
        targets = {k: v[frame : frame + HORIZON] for k, v in recorded.items()}
        arrays[f"frame_{frame}_rgb"] = rgb
        for k, v in targets.items():
            arrays[f"frame_{frame}_target_{k}"] = v

        entry = {"frame_index": frame, "image_shape": list(rgb.shape), "samples": []}
        for s in range(args.samples):
            torch.manual_seed(args.seed + 1000 * frame + s)
            np.random.seed(args.seed + 1000 * frame + s)
            t1 = time.monotonic()
            action, info = policy.get_action(obs)
            dt = time.monotonic() - t1
            preds = {}
            for key, dim in ACTION_DIMS.items():
                raw = np.asarray(action.get(key, action.get(f"action.{key}")), np.float32)
                while raw.ndim > 2 and raw.shape[0] == 1:
                    raw = raw[0]
                if raw.shape != (HORIZON, dim):
                    raise ValueError(f"{key}: {raw.shape} != {(HORIZON, dim)}")
                preds[key] = raw
                arrays[f"frame_{frame}_sample_{s}_{key}"] = raw
            entry["samples"].append(
                {
                    "sample_index": s,
                    "seconds": round(dt, 3),
                    "metrics": {k: metrics(preds[k], targets[k]) for k in preds},
                }
            )
            print(
                f"  frame {frame} sample {s}: {dt:.2f}s "
                f"motion_token mae={entry['samples'][-1]['metrics']['motion_token']['mae']:.4f} "
                f"corr={entry['samples'][-1]['metrics']['motion_token']['correlation']:.3f}"
            )
        result["frames"].append(entry)

    # ---- summary -----------------------------------------------------------
    summary = {}
    for key in ACTION_DIMS:
        maes = [
            s["metrics"][key]["mae"]
            for f in result["frames"]
            for s in f["samples"]
        ]
        cors = [
            s["metrics"][key]["correlation"]
            for f in result["frames"]
            for s in f["samples"]
        ]
        summary[key] = {
            "mae_mean": float(np.mean(maes)),
            "mae_min": float(np.min(maes)),
            "correlation_mean": float(np.mean(cors)),
        }
    result["summary"] = summary
    result["num_frames"] = len(result["frames"])
    result["num_samples_per_frame"] = args.samples
    times = [s["seconds"] for f in result["frames"] for s in f["samples"]]
    result["seconds_per_inference"] = {
        "mean": float(np.mean(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    print(json.dumps({"summary": summary, "timing": result["seconds_per_inference"]}, indent=2))
    print(f"wrote {out} and {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
